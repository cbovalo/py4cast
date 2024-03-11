# How to launch a training ?

## 1. Prepare the dataset and the model chosed

To be explained.

```bash
runai exec_gpu python bin/prepare.py titan all
```

```bash
runai exec_gpu python bin/prepare.py nlam --dataset titan
```

## 2. Launch the training (via training.py)

Main options are :

    - --model  ["hi_lam","graph_lam"] : The model choosed
    - --dataset ["titan","smeagol"] : The dataset choosed
    - --data_conf  : The configuration file for the dataset (used only for smeagol right now).
    - --steps : Number of autoregressive steps 
    - --standardize : Do we want to standardize our inputs ? 

In dev mode : 

    - --subset 10 : Number of batch to use 
    - --no_log : If activated, no log are kept in tensorboard. Models are not saved. 


### Examples

```sh
    runai gpu_play_mono
    runai exec_gpu python bin/train.py --model hi_lam --dataset smeagol
```

```sh
    runai gpu_play_mono
    runai exec_gpu python bin/train.py --model hi_lam --dataset titan
```


## 3. Some information on training speed. 

A few test had been conducted in order to see if there is regression in training speed due to an increase complexity. 
Here are the command launch. 
Note that the seed is fixed by default so exactly the same random number had been used.


```sh 
runai gpu_play 4
runai exec_gpu python bin/train.py --model hi_lam --dataset smeagol --no_log --standardize --gpu 4 --subset 200 --step 1
runai exec_gpu python bin/train.py --model hi_lam --dataset smeagol --no_log --standardize --gpu 4 --subset 200 --step 3
runai exec_gpu python bin/train.py --model hi_lam --dataset smeagol --no_log --standardize --gpu 1 --subset 200 --step 1
runai exec_gpu python bin/train.py --model hi_lam --dataset smeagol --no_log --standardize --gpu 1 --subset 200 --step 3
```

NB : The it per second is increasing batch after batch. There seem to be an initial cost which vanish. 


** Other important factors  which may impact a lot : **
  - batch_size : 1
  - num_workers : 10
  - prefetch : 2 
  - Grid: 500 x 500


|  | 1 Step | 3 Steps |
|--|--|--|
|1 GPU | 1.53 it/s -> 2:10 | 0.59 it/s -> 5:36 |
|4 GPU | 0.78 it/s -> 1:04 | 0.44 it/s -> 1:54 |


Test conducted on MR !14 

|  | 1 Step | 3 Steps |
|--|--|--|
|1 GPU |                   | 0.59 it/s -> 5:39|
|4 GPU | 0.80 it/s -> 1:02 | 0.43 it/s -> 1:55| 