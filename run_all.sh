# homophilic graphs 



python -u main.py --dataset 'Cora' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/Cora_gamma_1.txt

python -u main.py --dataset 'Cora' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 2.0 --device 'cuda:0' | tee output/Cora_gamma_2.txt

python -u main.py --dataset 'Cora' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 3.0 --device 'cuda:0' | tee output/Cora_gamma_3.txt

python -u main.py --dataset 'Cora' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 4.0 --device 'cuda:0' | tee output/Cora_gamma_4.txt

python -u main.py --dataset 'Cora' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 5.0 --device 'cuda:0' | tee output/Cora_gamma_5.txt



python -u main.py --dataset 'Cora' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/Cora_alpha_1.txt

python -u main.py --dataset 'Cora' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 2.0 --gamma 1.0 --device 'cuda:0' | tee output/Cora_alpha_2.txt

python -u main.py --dataset 'Cora' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 3.0 --gamma 1.0 --device 'cuda:0' | tee output/Cora_alpha_3.txt

python -u main.py --dataset 'Cora' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 4.0 --gamma 1.0 --device 'cuda:0' | tee output/Cora_alpha_4.txt

python -u main.py --dataset 'Cora' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 5.0 --gamma 1.0 --device 'cuda:0' | tee output/Cora_alpha_5.txt




python -u main.py --dataset 'Citeseer' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/Citeseer_gamma_1.txt

python -u main.py --dataset 'Citeseer' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 2.0 --device 'cuda:0' | tee output/Citeseer_gamma_2.txt

python -u main.py --dataset 'Citeseer' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 3.0 --device 'cuda:0' | tee output/Citeseer_gamma_3.txt

python -u main.py --dataset 'Citeseer' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 4.0 --device 'cuda:0' | tee output/Citeseer_gamma_4.txt

python -u main.py --dataset 'Citeseer' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 5.0 --device 'cuda:0' | tee output/Citeseer_gamma_5.txt



python -u main.py --dataset 'Citeseer' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/Citeseer_alpha_1.txt

python -u main.py --dataset 'Citeseer' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 2.0 --gamma 1.0 --device 'cuda:0' | tee output/Citeseer_alpha_2.txt

python -u main.py --dataset 'Citeseer' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 3.0 --gamma 1.0 --device 'cuda:0' | tee output/Citeseer_alpha_3.txt

python -u main.py --dataset 'Citeseer' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 4.0 --gamma 1.0 --device 'cuda:0' | tee output/Citeseer_alpha_4.txt

python -u main.py --dataset 'Citeseer' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 5.0 --gamma 1.0 --device 'cuda:0' | tee output/Citeseer_alpha_5.txt




python -u main.py --dataset 'Pubmed' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/Pubmed_gamma_1.txt

python -u main.py --dataset 'Pubmed' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 2.0 --device 'cuda:0' | tee output/Pubmed_gamma_2.txt

python -u main.py --dataset 'Pubmed' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 3.0 --device 'cuda:0' | tee output/Pubmed_gamma_3.txt

python -u main.py --dataset 'Pubmed' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 4.0 --device 'cuda:0' | tee output/Pubmed_gamma_4.txt

python -u main.py --dataset 'Pubmed' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 5.0 --device 'cuda:0' | tee output/Pubmed_gamma_5.txt



python -u main.py --dataset 'Pubmed' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/Pubmed_alpha_1.txt

python -u main.py --dataset 'Pubmed' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 2.0 --gamma 1.0 --device 'cuda:0' | tee output/Pubmed_alpha_2.txt

python -u main.py --dataset 'Pubmed' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 3.0 --gamma 1.0 --device 'cuda:0' | tee output/Pubmed_alpha_3.txt

python -u main.py --dataset 'Pubmed' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 4.0 --gamma 1.0 --device 'cuda:0' | tee output/Pubmed_alpha_4.txt

python -u main.py --dataset 'Pubmed' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 5.0 --gamma 1.0 --device 'cuda:0' | tee output/Pubmed_alpha_5.txt



# -----------------------------------------



# python -u main.py --dataset 'Chameleon' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/Chameleon_gamma_1.txt

# python -u main.py --dataset 'Chameleon' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 2.0 --device 'cuda:0' | tee output/Chameleon_gamma_2.txt

# python -u main.py --dataset 'Chameleon' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 3.0 --device 'cuda:0' | tee output/Chameleon_gamma_3.txt

# python -u main.py --dataset 'Chameleon' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 4.0 --device 'cuda:0' | tee output/Chameleon_gamma_4.txt

# python -u main.py --dataset 'Chameleon' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 5.0 --device 'cuda:0' | tee output/Chameleon_gamma_5.txt



# python -u main.py --dataset 'Chameleon' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/Chameleon_alpha_1.txt

# python -u main.py --dataset 'Chameleon' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 2.0 --gamma 1.0 --device 'cuda:0' | tee output/Chameleon_alpha_2.txt

# python -u main.py --dataset 'Chameleon' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 3.0 --gamma 1.0 --device 'cuda:0' | tee output/Chameleon_alpha_3.txt

# python -u main.py --dataset 'Chameleon' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 4.0 --gamma 1.0 --device 'cuda:0' | tee output/Chameleon_alpha_4.txt

# python -u main.py --dataset 'Chameleon' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 5.0 --gamma 1.0 --device 'cuda:0' | tee output/Chameleon_alpha_5.txt

# # --------

# python -u main.py --dataset 'Squirrel' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/Squirrel_gamma_1.txt

# python -u main.py --dataset 'Squirrel' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 2.0 --device 'cuda:0' | tee output/Squirrel_gamma_2.txt

# python -u main.py --dataset 'Squirrel' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 3.0 --device 'cuda:0' | tee output/Squirrel_gamma_3.txt

# python -u main.py --dataset 'Squirrel' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 4.0 --device 'cuda:0' | tee output/Squirrel_gamma_4.txt

# python -u main.py --dataset 'Squirrel' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 5.0 --device 'cuda:0' | tee output/Squirrel_gamma_5.txt


# python -u main.py --dataset 'Squirrel' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/Squirrel_alpha_1.txt

# python -u main.py --dataset 'Squirrel' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 2.0 --gamma 1.0 --device 'cuda:0' | tee output/Squirrel_alpha_2.txt

# python -u main.py --dataset 'Squirrel' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 3.0 --gamma 1.0 --device 'cuda:0' | tee output/Squirrel_alpha_3.txt

# python -u main.py --dataset 'Squirrel' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 4.0 --gamma 1.0 --device 'cuda:0' | tee output/Squirrel_alpha_4.txt

# python -u main.py --dataset 'Squirrel' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 5.0 --gamma 1.0 --device 'cuda:0' | tee output/Squirrel_alpha_5.txt

# ----------------

# python -u main.py --dataset 'Film' --train_lr 0.1 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/Film_gamma_1.txt

# python -u main.py --dataset 'Film' --train_lr 0.1 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 2.0 --device 'cuda:0' | tee output/Film_gamma_2.txt

# python -u main.py --dataset 'Film' --train_lr 0.1 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 3.0 --device 'cuda:0' | tee output/Film_gamma_3.txt

# python -u main.py --dataset 'Film' --train_lr 0.1 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 4.0 --device 'cuda:0' | tee output/Film_gamma_4.txt

# python -u main.py --dataset 'Film' --train_lr 0.1 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 5.0 --device 'cuda:0' | tee output/Film_gamma_5.txt


# python -u main.py --dataset 'Film' --train_lr 0.1 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/Film_alpha_1.txt

# python -u main.py --dataset 'Film' --train_lr 0.1 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 2.0 --gamma 1.0 --device 'cuda:0' | tee output/Film_alpha_2.txt

# python -u main.py --dataset 'Film' --train_lr 0.1 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 3.0 --gamma 1.0 --device 'cuda:0' | tee output/Film_alpha_3.txt

# python -u main.py --dataset 'Film' --train_lr 0.1 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 4.0 --gamma 1.0 --device 'cuda:0' | tee output/Film_alpha_4.txt

# python -u main.py --dataset 'Film' --train_lr 0.1 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 5.0 --gamma 1.0 --device 'cuda:0' | tee output/Film_alpha_5.txt


# ---------------

# python -u main.py --dataset 'Wisconsin' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/wisconsin_gamma_1.txt

# python -u main.py --dataset 'Wisconsin' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 2.0 --device 'cuda:0' | tee output/wisconsin_gamma_2.txt

# python -u main.py --dataset 'Wisconsin' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 3.0 --device 'cuda:0' | tee output/wisconsin_gamma_3.txt

# python -u main.py --dataset 'Wisconsin' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 4.0 --device 'cuda:0' | tee output/wisconsin_gamma_4.txt

# python -u main.py --dataset 'Wisconsin' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 5.0 --device 'cuda:0' | tee output/wisconsin_gamma_5.txt


# python -u main.py --dataset 'Wisconsin' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/wisconsin_alpha_1.txt

# python -u main.py --dataset 'Wisconsin' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 2.0 --gamma 1.0 --device 'cuda:0' | tee output/wisconsin_alpha_2.txt

# python -u main.py --dataset 'Wisconsin' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 3.0 --gamma 1.0 --device 'cuda:0' | tee output/wisconsin_alpha_3.txt

# python -u main.py --dataset 'Wisconsin' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 4.0 --gamma 1.0 --device 'cuda:0' | tee output/wisconsin_alpha_4.txt

# python -u main.py --dataset 'Wisconsin' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 5.0 --gamma 1.0 --device 'cuda:0' | tee output/wisconsin_alpha_5.txt



# ---------------

# python -u main.py --dataset 'Texas' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/texas_gamma_1.txt 

# python -u main.py --dataset 'Texas' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 2.0 --device 'cuda:0' | tee output/texas_gamma_2.txt 

# python -u main.py --dataset 'Texas' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 3.0 --device 'cuda:0' | tee output/texas_gamma_3.txt 

# python -u main.py --dataset 'Texas' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 4.0 --device 'cuda:0' | tee output/texas_gamma_4.txt 

# python -u main.py --dataset 'Texas' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 5.0 --device 'cuda:0' | tee output/texas_gamma_5.txt 


# python -u main.py --dataset 'Texas' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/texas_alpha_1.txt 

# python -u main.py --dataset 'Texas' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 2.0 --gamma 1.0 --device 'cuda:0' | tee output/texas_alpha_2.txt 

# python -u main.py --dataset 'Texas' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 3.0 --gamma 1.0 --device 'cuda:0' | tee output/texas_alpha_3.txt 

# python -u main.py --dataset 'Texas' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 4.0 --gamma 1.0 --device 'cuda:0' | tee output/texas_alpha_4.txt 

# python -u main.py --dataset 'Texas' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 5.0 --gamma 1.0 --device 'cuda:0' | tee output/texas_alpha_5.txt 


# # -------------

# python -u main.py --dataset 'Cornell' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 1.0 --device 'cuda:0'  | tee output/cornell_gamma_1.txt

# python -u main.py --dataset 'Cornell' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 2.0 --device 'cuda:0'  | tee output/cornell_gamma_2.txt

# python -u main.py --dataset 'Cornell' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 3.0 --device 'cuda:0'  | tee output/cornell_gamma_3.txt

# python -u main.py --dataset 'Cornell' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 4.0 --device 'cuda:0'  | tee output/cornell_gamma_4.txt

# python -u main.py --dataset 'Cornell' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 5.0 --device 'cuda:0'  | tee output/cornell_gamma_5.txt


# python -u main.py --dataset 'Cornell' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 1.0 --device 'cuda:0'  | tee output/cornell_alpha_1.txt

# python -u main.py --dataset 'Cornell' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 2.0 --gamma 1.0 --device 'cuda:0'  | tee output/cornell_alpha_2.txt

# python -u main.py --dataset 'Cornell' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 3.0 --gamma 1.0 --device 'cuda:0'  | tee output/cornell_alpha_3.txt

# python -u main.py --dataset 'Cornell' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 4.0 --gamma 1.0 --device 'cuda:0'  | tee output/cornell_alpha_4.txt

# python -u main.py --dataset 'Cornell' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 5.0 --gamma 1.0 --device 'cuda:0'  | tee output/cornell_alpha_5.txt




# -----------------------------------------------------
# Large-scale heterophilous graphs


# python -u large_scale_graphs_main.py --dataset 'arxiv-year' --train_lr 0.01  --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/arxiv_year_alpha_1.txt

# python -u large_scale_graphs_main.py --dataset 'arxiv-year' --train_lr 0.01  --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --num_parts 100 --num_splits 5 --alpha 2.0 --gamma 1.0 --device 'cuda:0' | tee output/arxiv_year_alpha_2.txt

# python -u large_scale_graphs_main.py --dataset 'arxiv-year' --train_lr 0.01  --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --num_parts 100 --num_splits 5 --alpha 3.0 --gamma 1.0 --device 'cuda:0' | tee output/arxiv_year_alpha_3.txt

# python -u large_scale_graphs_main.py --dataset 'arxiv-year' --train_lr 0.01  --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --num_parts 100 --num_splits 5 --alpha 4.0 --gamma 1.0 --device 'cuda:0' | tee output/arxiv_year_alpha_4.txt

# python -u large_scale_graphs_main.py --dataset 'arxiv-year' --train_lr 0.01  --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --num_parts 100 --num_splits 5 --alpha 5.0 --gamma 1.0 --device 'cuda:0' | tee output/arxiv_year_alpha_5.txt


# python -u large_scale_graphs_main.py --dataset 'arxiv-year' --train_lr 0.01  --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/arxiv_year_gamma_1.txt

# python -u large_scale_graphs_main.py --dataset 'arxiv-year' --train_lr 0.01  --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 2.0 --device 'cuda:0' | tee output/arxiv_year_gamma_2.txt

# python -u large_scale_graphs_main.py --dataset 'arxiv-year' --train_lr 0.01  --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 3.0 --device 'cuda:0' | tee output/arxiv_year_gamma_3.txt

# python -u large_scale_graphs_main.py --dataset 'arxiv-year' --train_lr 0.01  --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 4.0 --device 'cuda:0' | tee output/arxiv_year_gamma_4.txt

# python -u large_scale_graphs_main.py --dataset 'arxiv-year' --train_lr 0.01  --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 5.0 --device 'cuda:0' | tee output/arxiv_year_gamma_5.txt





# python -u large_scale_graphs_main.py --dataset 'snap-patents' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 1000 --num_splits 5 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/snap_patents_alpha_1.txt

# python -u large_scale_graphs_main.py --dataset 'snap-patents' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 1000 --num_splits 5 --alpha 2.0 --gamma 1.0 --device 'cuda:0' | tee output/snap_patents_alpha_2.txt

# python -u large_scale_graphs_main.py --dataset 'snap-patents' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 1000 --num_splits 5 --alpha 3.0 --gamma 1.0 --device 'cuda:0' | tee output/snap_patents_alpha_3.txt

# python -u large_scale_graphs_main.py --dataset 'snap-patents' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 1000 --num_splits 5 --alpha 4.0 --gamma 1.0 --device 'cuda:0' | tee output/snap_patents_alpha_4.txt

# python -u large_scale_graphs_main.py --dataset 'snap-patents' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 1000 --num_splits 5 --alpha 5.0 --gamma 1.0 --device 'cuda:0' | tee output/snap_patents_alpha_5.txt



# python -u large_scale_graphs_main.py --dataset 'snap-patents' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 1000 --num_splits 5 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/snap_patents_gamma_1.txt

# python -u large_scale_graphs_main.py --dataset 'snap-patents' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 1000 --num_splits 5 --alpha 1.0 --gamma 2.0 --device 'cuda:0' | tee output/snap_patents_gamma_2.txt

# python -u large_scale_graphs_main.py --dataset 'snap-patents' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 1000 --num_splits 5 --alpha 1.0 --gamma 3.0 --device 'cuda:0' | tee output/snap_patents_gamma_3.txt

# python -u large_scale_graphs_main.py --dataset 'snap-patents' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 1000 --num_splits 5 --alpha 1.0 --gamma 4.0 --device 'cuda:0' | tee output/snap_patents_gamma_4.txt

# python -u large_scale_graphs_main.py --dataset 'snap-patents' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 1000 --num_splits 5 --alpha 1.0 --gamma 5.0 --device 'cuda:0' | tee output/snap_patents_gamma_5.txt





# python -u large_scale_graphs_main.py --dataset 'Penn94' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/Penn94_alpha_1.txt

# python -u large_scale_graphs_main.py --dataset 'Penn94' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 100 --num_splits 5 --alpha 2.0 --gamma 1.0 --device 'cuda:0' | tee output/Penn94_alpha_2.txt

# python -u large_scale_graphs_main.py --dataset 'Penn94' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 100 --num_splits 5 --alpha 3.0 --gamma 1.0 --device 'cuda:0' | tee output/Penn94_alpha_3.txt

# python -u large_scale_graphs_main.py --dataset 'Penn94' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 100 --num_splits 5 --alpha 4.0 --gamma 1.0 --device 'cuda:0' | tee output/Penn94_alpha_4.txt

# python -u large_scale_graphs_main.py --dataset 'Penn94' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 100 --num_splits 5 --alpha 5.0 --gamma 1.0 --device 'cuda:0' | tee output/Penn94_alpha_5.txt



# python -u large_scale_graphs_main.py --dataset 'Penn94' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/Penn94_gamma_1.txt

# python -u large_scale_graphs_main.py --dataset 'Penn94' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 2.0 --device 'cuda:0' | tee output/Penn94_gamma_2.txt

# python -u large_scale_graphs_main.py --dataset 'Penn94' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 3.0 --device 'cuda:0' | tee output/Penn94_gamma_3.txt

# python -u large_scale_graphs_main.py --dataset 'Penn94' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 4.0 --device 'cuda:0' | tee output/Penn94_gamma_4.txt

# python -u large_scale_graphs_main.py --dataset 'Penn94' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 5.0 --device 'cuda:0' | tee output/Penn94_gamma_5.txt






# python -u large_scale_graphs_main.py --dataset 'pokec' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 2000 --num_splits 5 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/pokec_alpha_1.txt

# python -u large_scale_graphs_main.py --dataset 'pokec' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 2000 --num_splits 5 --alpha 2.0 --gamma 1.0 --device 'cuda:0' | tee output/pokec_alpha_2.txt

# python -u large_scale_graphs_main.py --dataset 'pokec' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 2000 --num_splits 5 --alpha 3.0 --gamma 1.0 --device 'cuda:0' | tee output/pokec_alpha_3.txt

# python -u large_scale_graphs_main.py --dataset 'pokec' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 2000 --num_splits 5 --alpha 4.0 --gamma 1.0 --device 'cuda:0' | tee output/pokec_alpha_4.txt

# python -u large_scale_graphs_main.py --dataset 'pokec' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 2000 --num_splits 5 --alpha 5.0 --gamma 1.0 --device 'cuda:0' | tee output/pokec_alpha_5.txt



# python -u large_scale_graphs_main.py --dataset 'pokec' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 2000 --num_splits 5 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/pokec_gamma_1.txt

# python -u large_scale_graphs_main.py --dataset 'pokec' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 2000 --num_splits 5 --alpha 1.0 --gamma 2.0 --device 'cuda:0' | tee output/pokec_gamma_2.txt

# python -u large_scale_graphs_main.py --dataset 'pokec' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 2000 --num_splits 5 --alpha 1.0 --gamma 3.0 --device 'cuda:0' | tee output/pokec_gamma_3.txt

# python -u large_scale_graphs_main.py --dataset 'pokec' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 2000 --num_splits 5 --alpha 1.0 --gamma 4.0 --device 'cuda:0' | tee output/pokec_gamma_4.txt

# python -u large_scale_graphs_main.py --dataset 'pokec' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 2000 --num_splits 5 --alpha 1.0 --gamma 5.0 --device 'cuda:0' | tee output/pokec_gamma_5.txt






# python -u large_scale_graphs_main.py --dataset 'twitch-gamers' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 0.001 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/twitch_gamers_alpha_1.txt

# python -u large_scale_graphs_main.py --dataset 'twitch-gamers' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 0.001 --num_parts 100 --num_splits 5 --alpha 2.0 --gamma 1.0 --device 'cuda:0' | tee output/twitch_gamers_alpha_2.txt

# python -u large_scale_graphs_main.py --dataset 'twitch-gamers' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 0.001 --num_parts 100 --num_splits 5 --alpha 3.0 --gamma 1.0 --device 'cuda:0' | tee output/twitch_gamers_alpha_3.txt

# python -u large_scale_graphs_main.py --dataset 'twitch-gamers' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 0.001 --num_parts 100 --num_splits 5 --alpha 4.0 --gamma 1.0 --device 'cuda:0' | tee output/twitch_gamers_alpha_4.txt

# python -u large_scale_graphs_main.py --dataset 'twitch-gamers' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 0.001 --num_parts 100 --num_splits 5 --alpha 5.0 --gamma 1.0 --device 'cuda:0' | tee output/twitch_gamers_alpha_5.txt



# python -u large_scale_graphs_main.py --dataset 'twitch-gamers' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 0.001 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/twitch_gamers_gamma_1.txt

# python -u large_scale_graphs_main.py --dataset 'twitch-gamers' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 0.001 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 2.0 --device 'cuda:0' | tee output/twitch_gamers_gamma_2.txt

# python -u large_scale_graphs_main.py --dataset 'twitch-gamers' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 0.001 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 3.0 --device 'cuda:0' | tee output/twitch_gamers_gamma_3.txt

# python -u large_scale_graphs_main.py --dataset 'twitch-gamers' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 0.001 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 4.0 --device 'cuda:0' | tee output/twitch_gamers_gamma_4.txt

# python -u large_scale_graphs_main.py --dataset 'twitch-gamers' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 0.001 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 5.0 --device 'cuda:0' | tee output/twitch_gamers_gamma_5.txt






# python -u large_scale_graphs_main.py --dataset 'genius' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 30 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/genius_alpha_1.txt

# python -u large_scale_graphs_main.py --dataset 'genius' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 30 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 100 --num_splits 5 --alpha 2.0 --gamma 1.0 --device 'cuda:0' | tee output/genius_alpha_2.txt

# python -u large_scale_graphs_main.py --dataset 'genius' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 30 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 100 --num_splits 5 --alpha 3.0 --gamma 1.0 --device 'cuda:0' | tee output/genius_alpha_3.txt

# python -u large_scale_graphs_main.py --dataset 'genius' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 30 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 100 --num_splits 5 --alpha 4.0 --gamma 1.0 --device 'cuda:0' | tee output/genius_alpha_4.txt

# python -u large_scale_graphs_main.py --dataset 'genius' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 30 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 100 --num_splits 5 --alpha 5.0 --gamma 1.0 --device 'cuda:0' | tee output/genius_alpha_5.txt



# python -u large_scale_graphs_main.py --dataset 'genius' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 30 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/genius_gamma_1.txt

# python -u large_scale_graphs_main.py --dataset 'genius' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 30 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 2.0 --device 'cuda:0' | tee output/genius_gamma_2.txt

# python -u large_scale_graphs_main.py --dataset 'genius' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 30 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 3.0 --device 'cuda:0' | tee output/genius_gamma_3.txt

# python -u large_scale_graphs_main.py --dataset 'genius' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 30 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 4.0 --device 'cuda:0' | tee output/genius_gamma_4.txt

# python -u large_scale_graphs_main.py --dataset 'genius' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 30 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 5.0 --device 'cuda:0' | tee output/genius_gamma_5.txt




# ---------------------------------------------------------
# new heterophilous graphs


# python -u heterophilous_main.py --dataset 'Roman-empire' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/roman_empire_alpha_1.txt

# python -u heterophilous_main.py --dataset 'Roman-empire' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 2.0 --gamma 1.0 --device 'cuda:0'  | tee output/roman_empire_alpha_2.txt

# python -u heterophilous_main.py --dataset 'Roman-empire' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 3.0 --gamma 1.0 --device 'cuda:0' | tee output/roman_empire_alpha_3.txt

# python -u heterophilous_main.py --dataset 'Roman-empire' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 4.0 --gamma 1.0 --device 'cuda:0' | tee output/roman_empire_alpha_4.txt

# python -u heterophilous_main.py --dataset 'Roman-empire' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 5.0 --gamma 1.0 --device 'cuda:0' | tee output/roman_empire_alpha_5.txt


# python -u heterophilous_main.py --dataset 'Roman-empire' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/roman_empire_gamma_1.txt

# python -u heterophilous_main.py --dataset 'Roman-empire' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 2.0 --device 'cuda:0' | tee output/roman_empire_gamma_2.txt

# python -u heterophilous_main.py --dataset 'Roman-empire' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 3.0 --device 'cuda:0' | tee output/roman_empire_gamma_3.txt

# python -u heterophilous_main.py --dataset 'Roman-empire' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 4.0 --device 'cuda:0' | tee output/roman_empire_gamma_4.txt

# python -u heterophilous_main.py --dataset 'Roman-empire' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 5.0 --device 'cuda:0' | tee output/roman_empire_gamma_5.txt




# python -u heterophilous_main.py --dataset 'Amazon-ratings' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/amazon_ratings_alpha_1.txt

# python -u heterophilous_main.py --dataset 'Amazon-ratings' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 2.0 --gamma 1.0 --device 'cuda:0' | tee output/amazon_ratings_alpha_2.txt

# python -u heterophilous_main.py --dataset 'Amazon-ratings' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 3.0 --gamma 1.0 --device 'cuda:0' | tee output/amazon_ratings_alpha_3.txt

# python -u heterophilous_main.py --dataset 'Amazon-ratings' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 4.0 --gamma 1.0 --device 'cuda:0' | tee output/amazon_ratings_alpha_4.txt

# python -u heterophilous_main.py --dataset 'Amazon-ratings' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 5.0 --gamma 1.0 --device 'cuda:0' | tee output/amazon_ratings_alpha_5.txt


# python -u heterophilous_main.py --dataset 'Amazon-ratings' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/amazon_ratings_gamma_1.txt

# python -u heterophilous_main.py --dataset 'Amazon-ratings' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 2.0 --device 'cuda:0' | tee output/amazon_ratings_gamma_2.txt

# python -u heterophilous_main.py --dataset 'Amazon-ratings' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 3.0 --device 'cuda:0' | tee output/amazon_ratings_gamma_3.txt

# python -u heterophilous_main.py --dataset 'Amazon-ratings' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 4.0 --device 'cuda:0' | tee output/amazon_ratings_gamma_4.txt

# python -u heterophilous_main.py --dataset 'Amazon-ratings' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 5.0 --device 'cuda:0' | tee output/amazon_ratings_gamma_5.txt



# python -u heterophilous_main.py --dataset 'Minesweeper' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/minesweeper_alpha_1.txt

# python -u heterophilous_main.py --dataset 'Minesweeper' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 2.0 --gamma 1.0 --device 'cuda:0' | tee output/minesweeper_alpha_2.txt

# python -u heterophilous_main.py --dataset 'Minesweeper' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 3.0 --gamma 1.0 --device 'cuda:0' | tee output/minesweeper_alpha_3.txt

# python -u heterophilous_main.py --dataset 'Minesweeper' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 4.0 --gamma 1.0 --device 'cuda:0' | tee output/minesweeper_alpha_4.txt

# python -u heterophilous_main.py --dataset 'Minesweeper' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 5.0 --gamma 1.0 --device 'cuda:0' | tee output/minesweeper_alpha_5.txt


# python -u heterophilous_main.py --dataset 'Minesweeper' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/minesweeper_gamma_1.txt

# python -u heterophilous_main.py --dataset 'Minesweeper' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 2.0 --device 'cuda:0' | tee output/minesweeper_gamma_2.txt

# python -u heterophilous_main.py --dataset 'Minesweeper' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 3.0 --device 'cuda:0' | tee output/minesweeper_gamma_3.txt

# python -u heterophilous_main.py --dataset 'Minesweeper' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 4.0 --device 'cuda:0' | tee output/minesweeper_gamma_4.txt

# python -u heterophilous_main.py --dataset 'Minesweeper' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 5.0 --device 'cuda:0' | tee output/minesweeper_gamma_5.txt



# python -u heterophilous_main.py --dataset 'Tolokers' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/tolokers_alpha_1.txt

# python -u heterophilous_main.py --dataset 'Tolokers' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 2.0 --gamma 1.0 --device 'cuda:0' | tee output/tolokers_alpha_2.txt

# python -u heterophilous_main.py --dataset 'Tolokers' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 3.0 --gamma 1.0 --device 'cuda:0' | tee output/tolokers_alpha_3.txt

# python -u heterophilous_main.py --dataset 'Tolokers' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 4.0 --gamma 1.0 --device 'cuda:0' | tee output/tolokers_alpha_4.txt

# python -u heterophilous_main.py --dataset 'Tolokers' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 5.0 --gamma 1.0 --device 'cuda:0' | tee output/tolokers_alpha_5.txt


# python -u heterophilous_main.py --dataset 'Tolokers' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/tolokers_gamma_1.txt

# python -u heterophilous_main.py --dataset 'Tolokers' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 2.0 --device 'cuda:0' | tee output/tolokers_gamma_2.txt

# python -u heterophilous_main.py --dataset 'Tolokers' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 3.0 --device 'cuda:0' | tee output/tolokers_gamma_3.txt

# python -u heterophilous_main.py --dataset 'Tolokers' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 4.0 --device 'cuda:0' | tee output/tolokers_gamma_4.txt

# python -u heterophilous_main.py --dataset 'Tolokers' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 5.0 --device 'cuda:0' | tee output/tolokers_gamma_5.txt




# python -u heterophilous_main.py --dataset 'Questions' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/questions_alpha_1.txt

# python -u heterophilous_main.py --dataset 'Questions' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 2.0 --gamma 1.0 --device 'cuda:0' | tee output/questions_alpha_2.txt

# python -u heterophilous_main.py --dataset 'Questions' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 3.0 --gamma 1.0 --device 'cuda:0' | tee output/questions_alpha_3.txt

# python -u heterophilous_main.py --dataset 'Questions' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 4.0 --gamma 1.0 --device 'cuda:0' | tee output/questions_alpha_4.txt

# python -u heterophilous_main.py --dataset 'Questions' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 5.0 --gamma 1.0 --device 'cuda:0' | tee output/questions_alpha_5.txt


# python -u heterophilous_main.py --dataset 'Questions' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/questions_gamma_1.txt

# python -u heterophilous_main.py --dataset 'Questions' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 2.0 --device 'cuda:0' | tee output/questions_gamma_2.txt

# python -u heterophilous_main.py --dataset 'Questions' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 3.0 --device 'cuda:0' | tee output/questions_gamma_3.txt

# python -u heterophilous_main.py --dataset 'Questions' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 4.0 --device 'cuda:0' | tee output/questions_gamma_4.txt

# python -u heterophilous_main.py --dataset 'Questions' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 5.0 --device 'cuda:0' | tee output/questions_gamma_5.txt



#---------------------------------------------------------







