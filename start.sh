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

python main_lincls.py \
    -a vit_small --lr 0.1 \
    --dist-url 'tcp://localhost:10001' \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    --pretrained output_gauss100-200-denoised/checkpoint_0199.pth.tar \
    noisy_mini-imagenet-gauss100-denoised > output_gauss100-200-denoised/eval_log.txt

python main_moco.py \
  -a vit_small -b 128 \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=60 --warmup-epochs=10 --resume=output_gauss100-200-denoised/checkpoint_0139.pth.tar\
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --dist-url 'tcp://localhost:10001' --out=output_gauss100-resume-0-140-200-0-60-60 \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  noisy_mini-imagenet-gauss100

python main_lincls.py \
    -a vit_small --lr 0.1 \
    --dist-url 'tcp://localhost:10001' \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    --pretrained output_gauss100-resume-0-140-200-0-60-60/checkpoint_0059.pth.tar \
    noisy_mini-imagenet-gauss100 > output_gauss100-resume-0-140-200-0-60-60/eval_log.txt

python main_moco.py \
  -a vit_small -b 128 \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=40 --warmup-epochs=10 --resume=output_gauss100-200-denoised/checkpoint_0159.pth.tar\
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --dist-url 'tcp://localhost:10001' --out=output_gauss100-resume-0-160-200-0-40-40 \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  noisy_mini-imagenet-gauss100

python main_lincls.py \
  -a vit_small --lr 0.1 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained output_gauss100-resume-0-160-200-0-40-40/checkpoint_0039.pth.tar \
  noisy_mini-imagenet-gauss100 > output_gauss100-resume-0-160-200-0-40-40/eval_log.txt


python main_moco.py   -a vit_small -b 128   --optimizer=adamw --lr=1.5e-4 --weight-decay=.1   --epochs=90 --warmup-epochs=10 --resume=output_gauss100-200-denoised/checkpoint_0109.pth.tar  --stop-grad-conv1 --moco-m-cos --moco-t=.2 
  --dist-url 'tcp://localhost:10001' --out=output_gauss100-resume-0-110-200-0-90-90   --multiprocessing-distributed --world-size 1 --rank 0   noisy_mini-imagenet-gauss100

python main_lincls.py \
  -a vit_small --lr 0.1 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained output_gauss100-resume-0-110-200-0-90-90/checkpoint_0089.pth.tar \
  noisy_mini-imagenet-gauss100 > output_gauss100-resume-0-110-200-0-90-90/eval_log.txt




for step in 0059 0054 0049 0044 0039 0034 0029 0024 0019 0014 0009 0004
do 
  mkdir output_gauss100-resume-0-140-200-0-60-60-$step
  python main_lincls.py \
    -a vit_small --lr 0.2 \
    --dist-url 'tcp://localhost:10001' \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    --pretrained output_gauss100-resume-0-140-200-0-60-60/checkpoint_${step}.pth.tar \
    --out output_gauss100-resume-0-140-200-0-60-60-$step \
    noisy_mini-imagenet-gauss100 > output_gauss100-resume-0-140-200-0-60-60-${step}/eval_log.txt
done


for step in 0199 0194 0189 0184 0179 0174 0169 0164 0159 0154 0149 0144 0139 0134 0129 0124 0119 0114 0109 0104 0099 0094 0089 0084 0079 0074 0069 0064 0059 0054 0049 0044 0039 0034 0029 0024 0019 0014 0009 0004
do 
  mkdir output_gauss100-200-denoised-$step
  python main_lincls.py \
    -a vit_small --lr 0.2 \
    --dist-url 'tcp://localhost:10001' \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    --pretrained output_gauss100-200-denoised/checkpoint_${step}.pth.tar \
    --out output_gauss100-200-denoised-$step \
    noisy_mini-imagenet-gauss100-denoised > output_gauss100-200-denoised-${step}/eval_log.txt
done

for step in 0199 0194 0189 0184 0179 0174 0169 0164 0159 0154 0149 0144 0139 0134 0129 0124 0119 0114 0109 0104 0099 0094 0089 0084 0079 0074 0069 0064 0059 0054 0049 0044 0039 0034 0029 0024 0019 0014 0009 0004
do 
  mkdir output_gauss100-200-$step
  python main_lincls.py \
    -a vit_small --lr 0.2 \
    --dist-url 'tcp://localhost:10001' \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    --pretrained output_gauss100-200/checkpoint_${step}.pth.tar \
    --out output_gauss100-200-$step \
    noisy_mini-imagenet-gauss100 > output_gauss100-200-${step}/eval_log.txt
done


for step in 0044 0039 0034 0029 0024 0019 0014 0009 0004; do    mkdir output_gauss100-resume-0-140-200-0-60-60-$step;   python main_lincls.py     -a vit_small --lr 0.2     --dist-url 'tcp://localhost:10001'     --multiprocessing-distributed --world-size 1 --rank 0     --pretrained output_gauss100-resume-0-140-200-0-60-60/checkpoint_${step}.pth.tar     --out output_gauss100-resume-0-140-200-0-60-60-$step     noisy_mini-imagenet-gauss100 > output_gauss100-resume-0-140-200-0-60-60-${step}/eval_log.txt; done

for step in 0049 0044 0039 0034 0029 0024 0019 0014 0009 0004; do    mkdir output_gauss100-resume-0-140-200-0-60-60-$step;   python main_lincls.py     -a vit_small --lr 0.2     --dist-url 'tcp://localhost:10001'     --multiprocessing-distributed --world-size 1 --rank 0     --pretrained output_gauss100-resume-0-140-200-0-60-60/checkpoint_${step}.pth.tar     --out output_gauss100-resume-0-140-200-0-60-60-$step     noisy_mini-imagenet-gauss100 > output_gauss100-resume-0-140-200-0-60-60-${step}/eval_log.txt; done