cd /gaozhangyang/experiments/ProteinInvBench
CUDA_VISIBLE_DEVICES=0 /root/anaconda3/envs/flash/bin/python ./train/main.py --ex_name e3pifold --use_dist 1 --use_product 1 --batch_size 8 --lr 1e-3 --epoch 20 --offline 0 --model_name E3PiFold


cd /gaozhangyang/experiments/ProteinInvBench
CUDA_VISIBLE_DEVICES=0 /root/anaconda3/envs/flash/bin/python ./train/main.py --ex_name PiFold_vec_interact --use_dist 1 --use_product 1 --batch_size 8 --lr 1e-3 --epoch 20 --offline 0 --model_name PiFold-wj


cd /gaozhangyang/experiments/ProteinInvBench
CUDA_VISIBLE_DEVICES=0 /root/anaconda3/envs/flash/bin/python ./train/main.py --ex_name PiFold_vec_interact_deep_wa_condition --use_dist 1 --use_product 1 --batch_size 8 --lr 1e-3 --epoch 20 --offline 0 --model_name PiFold-wj


cd /gaozhangyang/experiments/ProteinInvBench
CUDA_VISIBLE_DEVICES=0 /root/anaconda3/envs/flash/bin/python ./train/main.py --ex_name PiFold_virtual_edge --use_dist 1 --use_product 1 --batch_size 8 --lr 1e-3 --epoch 20 --offline 0 --model_name PiFold-wj

cd /gaozhangyang/experiments/ProteinInvBench
CUDA_VISIBLE_DEVICES=0 /root/anaconda3/envs/flash/bin/python ./train/main.py --ex_name PiFold_virtual_edge_rm_pointinteract --use_dist 1 --use_product 1 --batch_size 8 --lr 1e-3 --epoch 20 --offline 0 --model_name PiFold-wj

cd /gaozhangyang/experiments/ProteinInvBench
CUDA_VISIBLE_DEVICES=0 /root/anaconda3/envs/flash/bin/python ./train/main.py --ex_name PiFold_virtual_edge_adaptnet --use_dist 1 --use_product 1 --batch_size 8 --lr 1e-3 --epoch 20 --offline 0 --model_name PiFold-wj


cd /gaozhangyang/experiments/ProteinInvBench
CUDA_VISIBLE_DEVICES=0 /root/anaconda3/envs/flash/bin/python ./train/main.py --ex_name PiFold_virtual_edge_normalize --use_dist 1 --use_product 1 --batch_size 8 --lr 1e-3 --epoch 20 --offline 0 --model_name PiFold-wj

CUDA_VISIBLE_DEVICES=0 /root/anaconda3/envs/flash/bin/python ./train/main.py --ex_name PiFold_virtual_edge_simplify --use_dist 1 --use_product 1 --batch_size 8 --lr 1e-3 --epoch 20 --offline 0 --model_name PiFold-wj


CUDA_VISIBLE_DEVICES=0 /root/anaconda3/envs/flash/bin/python ./train/main.py --ex_name PiFold_virtual_edge_simplify_alignEV_edgeupate --use_dist 1 --use_product 1 --batch_size 8 --lr 1e-3 --epoch 20 --offline 0 --model_name PiFold-wj



CUDA_VISIBLE_DEVICES=0 /root/anaconda3/envs/flash/bin/python ./train/main.py --ex_name PiFold_virtual_edge_rm_Tst --use_dist 1 --use_product 1 --batch_size 8 --lr 1e-3 --epoch 20 --offline 0 --model_name PiFold-wj

CUDA_VISIBLE_DEVICES=0 /root/anaconda3/envs/flash/bin/python ./train/main.py --ex_name PiFold_10_global_attn --use_dist 1 --use_product 1 --batch_size 8 --lr 1e-3 --epoch 20 --offline 0 --model_name PiFold-wj
