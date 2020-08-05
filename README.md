# Co-Speech Gesture Generator

This is an implementation of *Robots learn social skills: End-to-end learning of co-speech gesture generation for humanoid robots* ([Paper](https://arxiv.org/abs/1810.12541), [Project Page](https://sites.google.com/view/youngwoo-yoon/projects/co-speech-gesture-generation))

The original paper used TED dataset, but, in this repository, we modified the code to use [Trinity Speech-Gesture Dataset](https://trinityspeechgesture.scss.tcd.ie/) for [GENEA Challenge 2020](https://genea-workshop.github.io/2020/).
The model is also changed to estimate rotation matrices for upper-body joints instead of estimating Cartesian coordinates.
  

## Environment
The code was developed using python 3.6 on Ubuntu 18.04. Pytorch 1.3.1 was used, but the latest version would be okay. 

## How to run

1. Install dependencies 
    ```
    pip install -r requirements.txt
    ```

1. Download the FastText vectors from [here](https://fasttext.cc/docs/en/english-vectors.html) and put `crawl-300d-2M-subword.bin` to the resource folder (`PROJECT_ROOT/resource/crawl-300d-2M-subword.bin`). 
You may use [the cache file](https://www.dropbox.com/s/9voiyhcgkg632hc/vocab_cache.pkl?dl=0) instead of downloading the FastText vectors (> 5 GB). Put the cache file into the LMDB folder that will be created in the next step. The code automatically loads the cache file when it exists (see `build_vocab` function). 

1. Make LMDB
    ```
    cd scripts
    python trinity_data_to_lmdb.py [PATH_TO_TRINITY_DATASET]
    ```

1. Update paths and parameters in `PROJECT_ROOT/config/seq2seq.yml` and run `train.py`
    ```
    python train.py --config=../config/seq2seq.yml
    ```

1. Inference
    ```
    python inference.py [PATH_TO_MODEL] [PATH_TO_AUDIO] [PATH_TO_TRANSCRIPT]
    ```
   We share the model trained on the training set of the GENEA challenge 2020.
[Click here to download](https://www.dropbox.com/s/2r19a34a9y5lg75/baseline_icra19_checkpoint_100.bin?dl=0)  


## License

Please see `LICENSE.md`


## Citation

```
@INPROCEEDINGS{
  yoonICRA19,
  title={Robots Learn Social Skills: End-to-End Learning of Co-Speech Gesture Generation for Humanoid Robots},
  author={Yoon, Youngwoo and Ko, Woo-Ri and Jang, Minsu and Lee, Jaeyeon and Kim, Jaehong and Lee, Geehyuk},
  booktitle={Proc. of The International Conference in Robotics and Automation (ICRA)},
  year={2019}
}
```
 

