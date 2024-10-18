#!/bin/bash -l
# L'argument '-l' est indispensable pour bénéficier des directives de votre .bashrc
 
# On peut éventuellement placer ici les commentaires SBATCH permettant de définir les paramètres par défaut de lancement :
#SBATCH --gres gpu:2
#SBATCH --time 1-23:50:00
#SBATCH --cpus-per-gpu 12
#SBATCH --mem-per-cpu 3G
#SBATCH --mail-type FAIL,END

conda activate myenv
python main.py mmvpr RGB --arch resnest269 --num_segments 8 --gd 20 --lr 0.01 --wd 1e-4 --lr_steps 20 40 --epochs 100 --batch-size 16 -j 32 --dropout 0.5 --consensus_type=avg --eval-freq=1 --shift --shift_div=8 --shift_place=blockres --npb
python3 generate_submission.py mmvpr --arch=resnest269 --csv_file=submission_resnest269_best.csv  --weights=checkpoint/TSM_mmvpr_RGB_resnest269_shift8_blockres_avg_segment8_e100/ckpt.best.pth.tar --test_segments=8 --batch_size=1 --test_crops=1
python3 generate_submission.py mmvpr --arch=resnest269 --csv_file=submission_resnest269_last.csv  --weights=checkpoint/TSM_mmvpr_RGB_resnest269_shift8_blockres_avg_segment8_e100/ckpt.pth.tar --test_segments=8 --batch_size=1 --test_crops=1

python main.py mmvpr RGB --arch resnext101_64x4d --num_segments 8 --gd 20 --lr 0.01 --wd 1e-4 --lr_steps 20 40 --epochs 100 --batch-size 16 -j 32 --dropout 0.5 --consensus_type=avg --eval-freq=1 --shift --shift_div=8 --shift_place=blockres --npb
python3 generate_submission.py mmvpr --arch=resnext101_64x4d --csv_file=submission_resnext101_64x4d_best.csv  --weights=checkpoint/TSM_mmvpr_RGB_resnext101_64x4d_shift8_blockres_avg_segment8_e100/ckpt.best.pth.tar --test_segments=8 --batch_size=1 --test_crops=1
python3 generate_submission.py mmvpr --arch=resnext101_64x4d --csv_file=submission_resnext101_64x4d_last.csv  --weights=checkpoint/TSM_mmvpr_RGB_resnext101_64x4d_shift8_blockres_avg_segment8_e100/ckpt.pth.tar --test_segments=8 --batch_size=1 --test_crops=1