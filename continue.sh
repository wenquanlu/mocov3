#for step in 0024 0019 0014 0009 0004; do    mkdir output_gauss100-resume-0-140-200-0-60-60-$step;   python main_lincls.py     -a vit_small --lr 0.2     --dist-url 'tcp://localhost:10001'     --multiprocessing-distributed --world-size 1 --rank 0     --pretrained output_gauss100-resume-0-140-200-0-60-60/checkpoint_${step}.pth.tar     --out output_gauss100-resume-0-140-200-0-60-60-$step     noisy_mini-imagenet-gauss100 > output_gauss100-resume-0-140-200-0-60-60-${step}/eval_log.txt; done

for step in 0179 0174 0169 0164 0159 0154 0149 0144 0139 0134 0129 0124 0119 0114 0109 0104 0099 0094 0089 0084 0079 0074 0069 0064 0059 0054 0049 0044 0039 0034 0029 0024 0019 0014 0009 0004
do 
  mkdir output_gauss100-200-$step
  python main_lincls.py \
    -a vit_small --lr 0.2 \
    --dist-url 'tcp://localhost:10002' \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    --pretrained output_gauss100-200/checkpoint_${step}.pth.tar \
    --out output_gauss100-200-$step \
    ../simsiam/noisy_mini-imagenet-gauss100 > output_gauss100-200-${step}/eval_log.txt
done
