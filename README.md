# MANet
2020/6/7:
	These experiments are based on a 2080Ti and Pytorch 1.2.
	Download the Market_1501 or DukeMTMC-ReID and unzip to the data folder

	The pre-trained model of ResNet-50 can be downloaded from:
		https://download.pytorch.org/models/resnet50-19c8e357.pth

	The pre-trained model of ResNeSt-50 can be downloaded from:
		https://hangzh.s3.amazonaws.com/encoding/models/resnest50-528c19ca.pth
	
	The feature vector of the auxiliary branch can be set to 256, 512, and 1024. 
	The specific results are as follows:
		Market_1501:	85.7%/95.2% (256dims) | 85.9%/95.3% (512dims) | 87.3%/95.7% (1024dims)
		DukeMTMC-ReID:	74.5%/88.9% (256dims) |	75.5%/89.5% (512dims) | 75.6%/89.5% (1024dims)
		
	
	We also upload the log during training(256dims).

2020/6/19: 

	Reminded by a friend, we also test the effect of the model on MSMT17 (auxiliary branch vector size: 512dims): 
		mAP/rank1 : 56.7%/82.8%
	

	We also updata the training log of MSMT17.
	MSMT17 dataset can be downloaded from:
		链接：https://pan.baidu.com/s/12khamxao4fDXc4aA9rKEIA  提取码：axqt
		
		
PS:	

	The experiment is based on two 2080Ti and Pytorch 1.2.
	An example on Market_1501:
		train:
			python main_reid.py train --save_dir='./pytorch-ckpt/market-bfe' --max_epoch=400 --eval_step=30 --dataset=market1501 --test_batch=32 --train_batch=32 --optim=adam --adjust_lr	
		test:
			python main_reid.py train --save_dir='./pytorch-ckpt/market_bfe' --model_name=bfe --train_batch=32 --test_batch=32 --dataset=market1501 --pretrained_model='./pytorch-ckpt/market-bfe/model_best.pth.tar' --evaluate



