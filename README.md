# High-Order Attention for Visual Question Answering

High-order attention models are strong tool for tasks  with several data modalities inputs. This code is for Visual Question Answering Multiple Choice, which performs 3-Modality attention over Question, Image and Multiple-Choice Answers. 
This code achieves 69.4 on Multiple-Choice VQA.


### Requirements
* Code is in lua, and require [Torch](http://torch.ch/)
* The preprocssinng code is in Python, and you need to install [NLTK](http://www.nltk.org/) if you want to use NLTK to tokenize the question.
* [spectral-lib](https://github.com/jnhwkim/spectral-lib) by @mbhenaff for MCB CuFFT wrappers

You also need to install the following package in order to sucessfully run the code.

- [cudnn.torch](https://github.com/soumith/cudnn.torch)
- [torch-hdf5](https://github.com/deepmind/torch-hdf5)
- [lua-cjson](http://www.kyne.com.au/~mark/software/lua-cjson.php)
- [loadcaffe](https://github.com/szagoruyko/loadcaffe)

### Training

##### Download Dataset
The first thing you need to do is to download the data and do some preprocessing. Head over to the `data/` folder and run

```
$ python vqa_preprocess.py --download 1 --split 1
```
* `--download 1` means you choose to download the VQA data from the [VQA website](http://www.visualqa.org/) 
* `--split 1` means you use COCO train set to train and validation set to evaluation.
* `--split 2 ` means you use COCO train+val set to train and test-dev set to evaluate. 
* `--split 3 ` means you use COCO train+val set to train and test-std set to evaluate. 

After this step, it will generate two files under the `data` folder. `vqa_raw_train.json` and `vqa_raw_test.json`

##### Preprocess Image/Question Features

```
$ python prepro_vqa.py --input_train_json ./vqa_raw_train.json --input_test_json ./vqa_raw_test.json --num_ans 3000 --max_length 15 --test 1
```
* --num_ans specifiy how many top answers you want to use during training.
* --max_length specify question length.
* --test indicate wheter we are in test setup (i.e. split 2 or split 3)

This will generate two files in `data/` folder, `vqa_data_prepro.h5` and `vqa_data_prepro.json`.You will also see some question and answer statistics in the terminal output.


##### Download Image Model
Here we use VGG_ILSVRC_19_layers [model](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77) and Deep Residual network implement by Facebook [model](https://github.com/facebook/fb.resnet.torch). 

Head over to the `image_model` folder and run

```
$ python download_model.py --download 'VGG' 
```
This will download the VGG_ILSVRC_19_layers model under `image_model` folder. To download the Deep Residual Model, you need to change the `VGG` to `Residual`.

Then we are ready to extract the image features. Head back to the `data` folder and run (You can change the `-gpuid`, `-backend` and `-batch_size` based on your gpu.)

For **VGG** image feature:

```
$ th prepro_img_vgg.lua -input_json ./vqa_data_prepro.json -image_root XXXX -cnn_proto ../image_model/VGG_ILSVRC_19_layers_deploy.prototxt -cnn_model ../image_model/VGG_ILSVRC_19_layers.caffemodel
```
This will generate two output files: `vqa_data_img_vgg_train.h5 `, `vqa_data_img_vgg_test.h5 ` in `-out_path`. 

For **Deep Residual** image feature:
```
$ th prepro_img_residule.lua -input_json ./data/vqa_data_prepro.json  -image_root XXXX -residule_path ../image_model/resnet-200.t7
```
This will generate two output files: `cocoqa_data_img_residule_train.h5`, `cocoqa_data_img_residule_test.h5` in `-out_path`. 

##### Train the model

Back to the `main` folder

```
th train.lua  \
        -id XX \
        -start_from 0 \
        -dropout 5 \
        -save_checkpoint_every 3000 \
        -eval 0 \        
        -feature_type Residual \
        -hidden_size 512 \
        -hidden_last_size 8192 \
        -hidden_combine_size 8192 \
        -batch_size 250 \
        -losses_log_every 100  \
        -learning_rate 4e-4 \
        -output_size 3001 \
        -learning_rate_decay_every 1200 \
        -input_img_train_h5 XX/cocoqa_data_img_residule_train.h5 \
        -input_ques_h5 data/vqa_data_prepro.h5 \
        -input_json data/vqa_data_prepro.json 
    
```
* `-start_from` iteration to load model from
* `-eval` set to 1, in case of `split=1`(i.e. train/val split), and add `-input_img_test_h5`.

##### Note
- Deep Residual Image Feature is 4 times larger than VGG feature, make sure you have enough RAM when you extract or load the features.
- If you didn't have large RAM, replace the `require 'misc.DataLoader'` (Line 11 in `train.lua`) with `require 'misc.DataLoaderDisk`. The model will read the data directly from the hard disk *(SSD is prefered)*

### Evaluation

```
th eval.lua -id XX -start_from XX \
        -feature_type Residual \
        -input_img_test_h5 XX/cocoqa_data_img_residule_test.h5 \
        -input_ques_h5 data/vqa_data_prepro.h5 \
        -input_json data/vqa_data_prepro.json \
        -MC 1
```
* `-MC` true indicates evaluation on Multiple-Choice task
##### Evaluate using Pre-trained Model

You can find The pre-trained model here (Make sure you provide the mcb files):
https://www.dropbox.com/sh/u6s47tay8yx7p3i/AACndwMd6E_k_WPNS-Cc26Ega?dl=0



##### VQA on Single Image with Free Form Question

Soon

##### Attention Visualization

Soon

### Reference

If you use this code as part of any published research, please acknowledge the following paper

```
@inproceedings{schwartz2017high,
  title={High-Order Attention Models for Visual Question Answering},
  author={Schwartz, Idan and Schwing, Alexander and Hazan, Tamir},
  booktitle={Advances in Neural Information Processing Systems},
  pages={3665--3675},
  year={2017}
}
```
