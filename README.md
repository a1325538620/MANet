# MANet

Download the Market_1501 or DukeMTMC-ReID and unzip to the data folder

The feature vectors of the auxiliary branches can be set to 256, 512, and 1024. 
The specific results are as follows:
		  mAP/rank1		  mAP/rank1               mAP/rank1
Market_1501:	85.7%/95.2% (256dims) | 85.9%/95.3% (512dims) | 87.3%/95.7% (1024dims)
DukeMTMC-ReID:	74.5%/88.9% (256dims) |	75.5%/89.5% (512dims) | 75.6%/89.5% (1024dims)

An example on Market_1501:
	train:
    		python main_reid.py train --save_dir='./pytorch-ckpt/market-bfe' --max_epoch=400 --eval_step=30 --dataset=market1501 --test_batch=32 --train_batch=32 --optim=adam --adjust_lr
    
	test:
		python main_reid.py train --save_dir='./pytorch-ckpt/market_bfe' --model_name=bfe --train_batch=32 --test_batch=32 --dataset=market1501 --pretrained_model='./pytorch-ckpt/market-bfe/model_best.pth.tar' --evaluate
		
		
We also uploaded the log during training.
