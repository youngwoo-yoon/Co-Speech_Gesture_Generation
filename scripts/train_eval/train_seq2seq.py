import logging
import torch
import torch.nn.functional as F


loss_i = 0
def custom_loss(output, target, args):
    n_element = output.numel()

    # MSE
    l1_loss = F.l1_loss(output, target)
    l1_loss *= args.loss_l1_weight

    # continuous motion
    diff = [abs(output[:, n, :] - output[:, n-1, :]) for n in range(1, output.shape[1])]
    cont_loss = torch.sum(torch.stack(diff)) / n_element
    cont_loss *= args.loss_cont_weight

    # motion variance
    norm = torch.norm(output, 2, 1)
    var_loss = -torch.sum(norm) / n_element
    var_loss *= args.loss_var_weight

    loss = l1_loss + cont_loss + var_loss

    # inspect loss terms
    global loss_i
    if loss_i == 100:
        logging.debug('  (loss terms) l1 %.5f, cont %.5f, var %.5f' % (l1_loss.item(), cont_loss.item(), var_loss.item()))
        loss_i = 0
    loss_i += 1

    return loss


def train_iter_seq2seq(args, epoch, in_text, in_lengths, target_poses, net, optim):
    # zero gradients
    optim.zero_grad()

    # generation
    outputs = net(in_text, in_lengths, target_poses, None)

    # loss
    loss = custom_loss(outputs, target_poses, args)
    loss.backward()

    # optimize
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
    optim.step()

    return {'loss': loss.item()}
