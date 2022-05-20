# Co-Speech Gesture Generator

This is an implementation of *Robots learn social skills: End-to-end learning of co-speech gesture generation for humanoid robots* ([Paper](https://arxiv.org/abs/1810.12541), [Project Page](https://sites.google.com/view/youngwoo-yoon/projects/co-speech-gesture-generation))

The original paper used TED dataset, but, in this repository, we modified the code to use [Talking With Hands 16.2M](https://github.com/facebookresearch/TalkingWithHands32M) for [GENEA Challenge 2022](https://genea-workshop.github.io/2022/challenge/).
The model is also changed to estimate rotation matrices for upper-body joints instead of estimating Cartesian coordinates.
  

## Environment
The code was developed using python 3.8 on Ubuntu 18.04. Pytorch 1.5.0 was used.

## How to run

1. Install dependencies 
    ```
    pip install -r requirements.txt
    ```

2. Download the FastText vectors from [here](https://fasttext.cc/docs/en/english-vectors.html) and put `crawl-300d-2M-subword.bin` to the resource folder (`PROJECT_ROOT/resource/crawl-300d-2M-subword.bin`). 

3. Make LMDB
    ```
    cd scripts
    python twh_dataset_to_lmdb.py [PATH_TO_DATASET]
    ```

4. Update paths and parameters in `PROJECT_ROOT/config/seq2seq.yml` and run `train.py`
    ```
    python train.py --config=../config/seq2seq.yml
    ```

5. Inference. Output a BVH motion file from speech text (TSV file).
    ```
    python inference.py [PATH_TO_MODEL_CHECKPOINT] [PATH_TO_TSV_FILE]
    ```


## Sample result

TBA


## Remarks

* A vocab cache and pretrained model will be available when the full training dataset is released.
* I found this model was not successful when all the joints were considered, so I trained the model only with upper-body joints excluding fingers and used fixed values for remaining joints (using JointSelector in PyMo). You can easily try a different set of joints (e.g., full-body including fingers) by specifying joint names in `target_joints` variable in `twh_dataset_to_lmdb.py`. Please update `data_mean` and `data_std` in the config file if you change `target_joints`. You can find data mean and std values in the console output of the step 3 (Make LMDB) above.


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
 

