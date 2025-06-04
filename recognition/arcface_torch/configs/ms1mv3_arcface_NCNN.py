from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()

# Margin Base Softmax
config.margin_list = (1.0, 0.5, 0.0)
config.network = "ncnn"
config.resume = False
config.save_all_states = True
config.output = "ms1mv3_arcface_NCNN"

config.embedding_size = 512

# Partial FC
config.sample_rate = 1.0
# Use fp16 for training
config.fp16 = False 

# For SGD 
config.optimizer = "sgd"
config.lr = 0.1
config.momentum = 0.9
config.weight_decay = 5e-4

config.batch_size = 128

config.verbose = 2000
config.dali = False

# dataload numworkers
config.num_workers = 12

config.rec = "./train/ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 30
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
