import datetime
import pprint
import random
import time
import sys
import numpy as np

from torch.utils.data import DataLoader

[sys.path.append(i) for i in ['.', '..']]

from model import vocab
from model.seq2seq_net import Seq2SeqNet
from train_eval.train_seq2seq import train_iter_seq2seq
from utils.average_meter import AverageMeter
from utils.vocab_utils import build_vocab

from config.parse_args import parse_args

from torch import optim

from twh_dataset_to_lmdb import target_joints
from data_loader.lmdb_data_loader import *
import utils.train_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_model(args, lang_model, pose_dim, _device):
    n_frames = args.n_poses
    generator = Seq2SeqNet(args, pose_dim, n_frames, lang_model.n_words, args.wordembed_dim,
                           lang_model.word_embedding_weights).to(_device)
    loss_fn = torch.nn.MSELoss()

    return generator, loss_fn


def train_epochs(args, train_data_loader, test_data_loader, lang_model, pose_dim, trial_id=None):
    start = time.time()
    loss_meters = [AverageMeter('loss'), AverageMeter('var_loss')]

    # interval params
    print_interval = int(len(train_data_loader) / 5)
    save_model_epoch_interval = 20

    # init model
    generator, loss_fn = init_model(args, lang_model, pose_dim, device)

    # define optimizers
    gen_optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    # training
    global_iter = 0
    for epoch in range(1, args.epochs+1):
        # evaluate the test set
        val_metrics = evaluate_testset(test_data_loader, generator, loss_fn, args)

        # save model
        if epoch % save_model_epoch_interval == 0 and epoch > 0:
            try:  # multi gpu
                gen_state_dict = generator.module.state_dict()
            except AttributeError:  # single gpu
                gen_state_dict = generator.state_dict()

            save_name = '{}/{}_checkpoint_{:03d}.bin'.format(args.model_save_path, args.name, epoch)

            utils.train_utils.save_checkpoint({
                'args': args, 'epoch': epoch, 'lang_model': lang_model,
                'pose_dim': pose_dim, 'gen_dict': gen_state_dict
            }, save_name)

        # train iter
        iter_start_time = time.time()
        for iter_idx, data in enumerate(train_data_loader, 0):
            global_iter += 1
            in_text, text_lengths, target_vec, in_audio, aux_info = data
            batch_size = target_vec.size(0)

            in_text = in_text.to(device)
            in_audio = in_audio.to(device)
            target_vec = target_vec.to(device)

            # train
            loss = train_iter_seq2seq(args, epoch, in_text, text_lengths, target_vec, generator, gen_optimizer)

            # loss values
            for loss_meter in loss_meters:
                name = loss_meter.name
                if name in loss:
                    loss_meter.update(loss[name], batch_size)

            # print training status
            if (iter_idx + 1) % print_interval == 0:
                print_summary = 'EP {} ({:3d}) | {:>8s}, {:.0f} samples/s | '.format(
                    epoch, iter_idx + 1, utils.train_utils.time_since(start),
                    batch_size / (time.time() - iter_start_time))
                for loss_meter in loss_meters:
                    if loss_meter.count > 0:
                        print_summary += '{}: {:.3f}, '.format(loss_meter.name, loss_meter.avg)
                        loss_meter.reset()
                logging.info(print_summary)

            iter_start_time = time.time()


def evaluate_testset(test_data_loader, generator, loss_fn, args):
    # to evaluation mode
    generator.train(False)

    losses = AverageMeter('loss')
    start = time.time()

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            in_text, text_lengths, target_vec, in_audio, aux_info = data
            batch_size = target_vec.size(0)

            in_text = in_text.to(device)
            target = target_vec.to(device)

            out_poses = generator(in_text, text_lengths, target, None)
            loss = loss_fn(out_poses, target)
            losses.update(loss.item(), batch_size)

    # back to training mode
    generator.train(True)

    # print
    elapsed_time = time.time() - start
    logging.info('[VAL] loss: {:.3f} / {:.1f}s'.format(losses.avg, elapsed_time))

    return losses.avg


def main(config):
    args = config['args']

    trial_id = None

    # random seed
    if args.random_seed >= 0:
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    # set logger
    utils.train_utils.set_logger(args.model_save_path, os.path.basename(__file__).replace('.py', '.log'))

    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA version: {}".format(torch.version.cuda))
    logging.info(pprint.pformat(vars(args)))

    # dataset
    train_dataset = TwhDataset(args.train_data_path[0],
                               n_poses=args.n_poses,
                               subdivision_stride=args.subdivision_stride,
                               pose_resampling_fps=args.motion_resampling_framerate,
                               data_mean=args.data_mean, data_std=args.data_std)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                              collate_fn=word_seq_collate_fn
                              )

    val_dataset = TwhDataset(args.val_data_path[0],
                             n_poses=args.n_poses,
                             subdivision_stride=args.subdivision_stride,
                             pose_resampling_fps=args.motion_resampling_framerate,
                             data_mean=args.data_mean, data_std=args.data_std)
    test_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                             shuffle=False, drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                             collate_fn=word_seq_collate_fn
                             )

    # build vocab
    vocab_cache_path = os.path.join(os.path.split(args.train_data_path[0])[0], 'vocab_cache.pkl')
    lang_model = build_vocab('words', [train_dataset, val_dataset], vocab_cache_path, args.wordembed_path,
                             args.wordembed_dim)
    train_dataset.set_lang_model(lang_model)
    val_dataset.set_lang_model(lang_model)

    # train
    train_epochs(args, train_loader, test_loader, lang_model,
                 pose_dim=len(target_joints)*12, trial_id=trial_id)


if __name__ == '__main__':
    _args = parse_args()
    main({'args': _args})
