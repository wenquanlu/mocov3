python main_moco.py \
  -a vit_small -b 128 \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=200 --warmup-epochs=20 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --dist-url 'tcp://localhost:10001' --out=output_clean-200v2\
  --multiprocessing-distributed --world-size 1 --rank 0 \
  mini-imagenet


python main_moco.py \
    -b 128 \
    --epochs=200 --warmup-epochs=20 \
    --moco-m-cos --crop-min=.2 \
    --dist-url 'tcp://localhost:10001' \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    mini-imagenet

python main_lincls.py \
  -a vit_small --lr 0.1 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained output_clean-200/checkpoint_0199.pth.tar \
  mini-imagenet

  python main_lincls.py \
  -a vit_small --lr 0.1 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained output_clean-200v2/checkpoint_0199.pth.tar \
  mini-imagenet

  python main_moco.py \
  -a vit_small -b 128 \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=200 --warmup-epochs=20 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --dist-url 'tcp://localhost:10001' --out=output_gauss100-200\
  --multiprocessing-distributed --world-size 1 --rank 0 \
  noisy_mini-imagenet-gauss100

python main_lincls.py \
    -a vit_small --lr 0.1 \
    --dist-url 'tcp://localhost:10001' \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    --pretrained output_gauss100-200/checkpoint_0199.pth.tar \
    noisy_mini-imagenet-gauss100

python main_moco.py \
  -a vit_small -b 128 \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=200 --warmup-epochs=20 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --dist-url 'tcp://localhost:10001' --out=output_gauss100-200-denoised\
  --multiprocessing-distributed --world-size 1 --rank 0 \
  noisy_mini-imagenet-gauss100-denoised
