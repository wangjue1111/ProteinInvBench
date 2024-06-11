cd /gaozhangyang/experiments/ProteinInvBench

# CUDA_VISIBLE_DEVICES=0 /root/anaconda3/envs/flash/bin/python ./train/main.py --ex_name pifold_ipa-lr1e-3  --use_dist 1 --use_product 1 --batch_size 32 --lr 1e-3 --epoch 20 --offline 0 --model_name PiFold-wj



CUDA_VISIBLE_DEVICES=0 /root/anaconda3/envs/flash/bin/python ./train/main.py --ex_name pifold_ipa_deepcross-lr1e-3-bs8  --use_dist 1 --use_product 1 --batch_size 8 --lr 1e-3 --epoch 20 --offline 0 --model_name PiFold-wj


CUDA_VISIBLE_DEVICES=0 /root/anaconda3/envs/flash/bin/python ./train/main.py --ex_name pifold_ipa_pairdist-lr1e-3-bs8  --use_dist 1 --use_product 1 --batch_size 8 --lr 1e-3 --epoch 20 --offline 0 --model_name PiFold-wj


CUDA_VISIBLE_DEVICES=0 /root/anaconda3/envs/flash/bin/python ./train/main.py --ex_name pifold_ipa_cdconv-lr1e-3-bs8  --use_dist 1 --use_product 1 --batch_size 8 --lr 1e-3 --epoch 20 --offline 0 --model_name PiFold-wj


CUDA_VISIBLE_DEVICES=0 /root/anaconda3/envs/flash/bin/python ./train/main.py --ex_name KWDesign_BlockGAT --batch_size 8 --lr 1e-3 --epoch 20 --offline 0 --model_name KWDesign_BlockGAT