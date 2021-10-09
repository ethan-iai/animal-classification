& D:/anaconda/envs/torch38/python.exe d:/animal-classification/main.py --seed 5 --name animals --batch-size 32 --finetune-batch-size 128 --workers 0 --expand-labels --data-path data --dataset animals --num-classes 22 --total-steps 300000 --eval-step 1000 --randaug 2 16 --teacher_lr 0.05 --student_lr 0.05 --weight-decay 5e-4 --ema 0.995 --nesterov --mu 2 --label-smoothing 0.15 --temperature 0.7 --threshold 0.6 --lambda-u 8 --warmup-steps 5000 --uda-steps 5000 --student-wait-steps 3000 --teacher-dropout 0.2 --student-dropout 0.2 --amp

python main.py --seed 5 --name animals --expand-labels --data-path data --dataset animals --num-classes 22 --total-steps 300000 --eval-step 1000 --randaug 2 16 --batch-size 128 --teacher_lr 0.05 --student_lr 0.05 --weight-decay 5e-4 --ema 0.995 --nesterov --mu 7 --label-smoothing 0.15 --temperature 0.7 --threshold 0.6 --lambda-u 8 --warmup-steps 5000 --uda-steps 5000 --student-wait-steps 3000 --teacher-dropout 0.2 --student-dropout 0.2 --amp

python imagenet data -j 0 -a resnet50 --gpu0

& D:/anaconda/envs/torch38/python.exe d:/animal-classification/imagenet.py data -j 0 -a resnet50 --gpu 0