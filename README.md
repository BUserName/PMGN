# PMGN: Prototype-Matching Graph Network for Heterogeneous Domain Adaptation

This is the official implementation of MM'20 paper 'Prototype-Matching Graph Network for Heterogeneous Domain Adaptation'


Please install the required libaries first, and then run code

```
python train.py
```

We provide data of 3 HDA tasks for you to try.

Source domain | Target domain | Extra Parameter Setting| Result (Pytorch 1.6.0)
------------ | -------------  | ----------             | -----------
Amazon_Surf  | Webcam_Decaf   | --dis_loss=0.15        | 98.62
rand_GR      | rand_SP_10     |                        | 
NUSTAG       | IMGNET         |                        |  


Full dataset can be download at:
* Office-Caltech: https://www.dropbox.com/s/i87qac7kzkwy8d9/datasets.zip
* NUS-IMG: https://www.dropbox.com/s/i87qac7kzkwy8d9/datasets.zip
  * (Credits to the Transfer Nerual Trees Repo https://github.com/wyharveychen/TransferNeuralTrees)

* Multilingual Reuters Collection: 

