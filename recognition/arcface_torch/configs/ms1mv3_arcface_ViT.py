from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()

# Margin Base Softmax
config.margin_list = (1.0, 0.5, 0.0)
config.network = "vit_b_16"
config.resume = False
config.save_all_states = True
config.output = "ms1mv3_arcface_ViT"

config.embedding_size = 512

# Use fp16 for training
config.fp16 = True 

# Partial FC
config.sample_rate = 1.0
config.fp16 = False

# For SGD 
config.optimizer = "adamw"
config.lr = 0.001
config.momentum = 0.9
config.weight_decay = 0.1

config.batch_size = 128

config.verbose = 2000
config.dali = False

# dataload numworkers
config.num_workers = 12

config.rec = "./train/ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 40
config.warmup_epoch = config.num_epoch // 10
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
