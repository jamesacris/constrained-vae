# shell script to train whole grid of models on one gpu
# 5 models at a time
# beta = .002
python train_beta_VAE.py -z 10 --norm_beta .002 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta .002 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta .002 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta .002 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta .002 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta .002 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta .002 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta .002 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta .002 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta .002 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta .002 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta .002 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta .002 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta .002 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta .002 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta .002 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta .002 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta .002 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta .002 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta .002 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = .004
python train_beta_VAE.py -z 10 --norm_beta .004 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta .004 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta .004 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta .004 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta .004 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta .004 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta .004 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta .004 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta .004 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta .004 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta .004 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta .004 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta .004 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta .004 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta .004 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta .004 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta .004 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta .004 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta .004 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta .004 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = .006
python train_beta_VAE.py -z 10 --norm_beta .006 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta .006 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta .006 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta .006 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta .006 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta .006 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta .006 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta .006 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta .006 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta .006 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta .006 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta .006 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta .006 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta .006 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta .006 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta .006 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta .006 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta .006 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta .006 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta .006 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = .008
python train_beta_VAE.py -z 10 --norm_beta .008 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta .008 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta .008 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta .008 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta .008 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta .008 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta .008 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta .008 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta .008 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta .008 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta .008 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta .008 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta .008 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta .008 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta .008 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta .008 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta .008 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta .008 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta .008 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta .008 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = .01
python train_beta_VAE.py -z 10 --norm_beta .01 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta .01 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta .01 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta .01 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta .01 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta .01 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta .01 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta .01 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta .01 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta .01 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta .01 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta .01 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta .01 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta .01 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta .01 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta .01 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta .01 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta .01 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta .01 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta .01 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = .02
python train_beta_VAE.py -z 10 --norm_beta .02 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta .02 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta .02 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta .02 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta .02 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta .02 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta .02 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta .02 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta .02 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta .02 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta .02 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta .02 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta .02 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta .02 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta .02 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta .02 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta .02 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta .02 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta .02 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta .02 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = .03
python train_beta_VAE.py -z 10 --norm_beta .03 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta .03 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta .03 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta .03 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta .03 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta .03 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta .03 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta .03 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta .03 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta .03 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta .03 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta .03 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta .03 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta .03 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta .03 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta .03 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta .03 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta .03 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta .03 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta .03 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = .04
python train_beta_VAE.py -z 10 --norm_beta .04 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta .04 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta .04 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta .04 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta .04 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta .04 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta .04 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta .04 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta .04 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta .04 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta .04 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta .04 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta .04 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta .04 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta .04 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta .04 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta .04 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta .04 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta .04 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta .04 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = .05
python train_beta_VAE.py -z 10 --norm_beta .05 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta .05 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta .05 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta .05 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta .05 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta .05 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta .05 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta .05 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta .05 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta .05 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta .05 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta .05 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta .05 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta .05 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta .05 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta .05 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta .05 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta .05 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta .05 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta .05 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = .06
python train_beta_VAE.py -z 10 --norm_beta .06 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta .06 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta .06 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta .06 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta .06 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta .06 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta .06 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta .06 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta .06 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta .06 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta .06 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta .06 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta .06 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta .06 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta .06 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta .06 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta .06 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta .06 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta .06 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta .06 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = .07
python train_beta_VAE.py -z 10 --norm_beta .07 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta .07 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta .07 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta .07 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta .07 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta .07 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta .07 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta .07 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta .07 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta .07 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta .07 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta .07 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta .07 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta .07 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta .07 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta .07 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta .07 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta .07 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta .07 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta .07 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = .08
python train_beta_VAE.py -z 10 --norm_beta .08 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta .08 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta .08 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta .08 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta .08 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta .08 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta .08 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta .08 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta .08 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta .08 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta .08 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta .08 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta .08 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta .08 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta .08 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta .08 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta .08 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta .08 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta .08 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta .08 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = .09
python train_beta_VAE.py -z 10 --norm_beta .09 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta .09 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta .09 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta .09 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta .09 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta .09 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta .09 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta .09 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta .09 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta .09 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta .09 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta .09 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta .09 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta .09 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta .09 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta .09 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta .09 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta .09 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta .09 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta .09 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = .1
python train_beta_VAE.py -z 10 --norm_beta .1 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta .1 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta .1 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta .1 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta .1 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta .1 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta .1 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta .1 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta .1 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta .1 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta .1 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta .1 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta .1 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta .1 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta .1 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta .1 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta .1 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta .1 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta .1 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta .1 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = .2
python train_beta_VAE.py -z 10 --norm_beta .2 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta .2 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta .2 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta .2 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta .2 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta .2 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta .2 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta .2 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta .2 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta .2 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta .2 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta .2 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta .2 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta .2 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta .2 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta .2 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta .2 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta .2 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta .2 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta .2 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = .3
python train_beta_VAE.py -z 10 --norm_beta .3 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta .3 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta .3 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta .3 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta .3 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta .3 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta .3 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta .3 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta .3 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta .3 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta .3 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta .3 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta .3 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta .3 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta .3 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta .3 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta .3 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta .3 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta .3 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta .3 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = .4
python train_beta_VAE.py -z 10 --norm_beta .4 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta .4 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta .4 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta .4 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta .4 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta .4 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta .4 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta .4 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta .4 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta .4 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta .4 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta .4 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta .4 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta .4 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta .4 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta .4 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta .4 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta .4 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta .4 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta .4 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = .5
python train_beta_VAE.py -z 10 --norm_beta .5 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta .5 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta .5 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta .5 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta .5 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta .5 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta .5 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta .5 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta .5 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta .5 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta .5 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta .5 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta .5 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta .5 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta .5 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta .5 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta .5 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta .5 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta .5 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta .5 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = .6
python train_beta_VAE.py -z 10 --norm_beta .6 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta .6 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta .6 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta .6 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta .6 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta .6 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta .6 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta .6 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta .6 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta .6 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta .6 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta .6 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta .6 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta .6 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta .6 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta .6 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta .6 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta .6 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta .6 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta .6 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = .7
python train_beta_VAE.py -z 10 --norm_beta .7 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta .7 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta .7 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta .7 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta .7 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta .7 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta .7 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta .7 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta .7 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta .7 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta .7 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta .7 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta .7 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta .7 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta .7 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta .7 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta .7 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta .7 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta .7 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta .7 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = .8
python train_beta_VAE.py -z 10 --norm_beta .8 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta .8 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta .8 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta .8 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta .8 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta .8 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta .8 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta .8 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta .8 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta .8 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta .8 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta .8 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta .8 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta .8 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta .8 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta .8 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta .8 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta .8 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta .8 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta .8 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = .9
python train_beta_VAE.py -z 10 --norm_beta .9 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta .9 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta .9 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta .9 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta .9 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta .9 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta .9 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta .9 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta .9 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta .9 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta .9 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta .9 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta .9 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta .9 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta .9 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta .9 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta .9 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta .9 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta .9 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta .9 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = 1.0
python train_beta_VAE.py -z 10 --norm_beta 1.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta 1.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta 1.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta 1.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta 1.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta 1.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta 1.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta 1.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta 1.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta 1.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta 1.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta 1.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta 1.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta 1.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta 1.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta 1.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta 1.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta 1.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta 1.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta 1.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = 2.0
python train_beta_VAE.py -z 10 --norm_beta 2.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta 2.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta 2.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta 2.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta 2.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta 2.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta 2.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta 2.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta 2.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta 2.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta 2.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta 2.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta 2.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta 2.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta 2.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta 2.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta 2.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta 2.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta 2.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta 2.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = 3.0
python train_beta_VAE.py -z 10 --norm_beta 3.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta 3.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta 3.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta 3.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta 3.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta 3.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta 3.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta 3.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta 3.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta 3.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta 3.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta 3.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta 3.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta 3.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta 3.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta 3.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta 3.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta 3.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta 3.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta 3.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = 4.0
python train_beta_VAE.py -z 10 --norm_beta 4.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta 4.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta 4.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta 4.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta 4.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta 4.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta 4.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta 4.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta 4.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta 4.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta 4.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta 4.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta 4.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta 4.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta 4.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta 4.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta 4.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta 4.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta 4.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta 4.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = 5.0
python train_beta_VAE.py -z 10 --norm_beta 5.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta 5.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta 5.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta 5.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta 5.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta 5.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta 5.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta 5.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta 5.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta 5.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta 5.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta 5.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta 5.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta 5.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta 5.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta 5.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta 5.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta 5.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta 5.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta 5.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = 6.0
python train_beta_VAE.py -z 10 --norm_beta 6.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta 6.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta 6.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta 6.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta 6.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta 6.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta 6.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta 6.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta 6.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta 6.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta 6.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta 6.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta 6.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta 6.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta 6.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta 6.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta 6.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta 6.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta 6.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta 6.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = 7.0
python train_beta_VAE.py -z 10 --norm_beta 7.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta 7.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta 7.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta 7.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta 7.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta 7.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta 7.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta 7.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta 7.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta 7.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta 7.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta 7.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta 7.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta 7.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta 7.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta 7.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta 7.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta 7.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta 7.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta 7.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = 8.0
python train_beta_VAE.py -z 10 --norm_beta 8.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta 8.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta 8.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta 8.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta 8.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta 8.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta 8.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta 8.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta 8.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta 8.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta 8.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta 8.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta 8.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta 8.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta 8.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta 8.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta 8.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta 8.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta 8.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta 8.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = 9.0
python train_beta_VAE.py -z 10 --norm_beta 9.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta 9.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta 9.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta 9.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta 9.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta 9.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta 9.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta 9.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta 9.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta 9.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta 9.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta 9.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta 9.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta 9.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta 9.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta 9.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta 9.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta 9.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta 9.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta 9.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
# beta = 10.0
python train_beta_VAE.py -z 10 --norm_beta 10.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 20 --norm_beta 10.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 30 --norm_beta 10.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 40 --norm_beta 10.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 50 --norm_beta 10.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 60 --norm_beta 10.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 70 --norm_beta 10.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 80 --norm_beta 10.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 90 --norm_beta 10.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 100 --norm_beta 10.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0
python train_beta_VAE.py -z 110 --norm_beta 10.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 120 --norm_beta 10.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 130 --norm_beta 10.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 140 --norm_beta 10.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 150 --norm_beta 10.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 
python train_beta_VAE.py -z 160 --norm_beta 10.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 170 --norm_beta 10.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 180 --norm_beta 10.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 190 --norm_beta 10.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0 &
python train_beta_VAE.py -z 200 --norm_beta 10.0 --epoch 20 --verbose-batch 1000000 --physical-gpu 0