CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
  --ndf 320 \
  --n_rkhs 1280 \
  --batch_size 480 \
  --tclip 20.0 \
  --n_depth 10 \
  --dataset C10 \
  --amp
