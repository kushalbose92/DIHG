# python -u main.py --dataset 'Cora' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 1.0 --device 'cuda:0'

# python -u main.py --dataset 'Citeseer' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.10 --train_w_decay 0.0005 --th 1.0 --alpha 1.0 --gamma 1.0 --device 'cuda:0'

# python -u main.py --dataset 'Pubmed' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.10 --train_w_decay 0.0005 --th None --alpha 1.0 -gamma 1.0 --device 'cuda:0'

# python -u main.py --dataset 'Chameleon' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee Chameleon.txt

# python -u main.py --dataset 'Chameleon' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 5.0 --gamma 1.0 --device 'cuda:0' | tee Chameleon.txt

# python -u main.py --dataset 'Chameleon' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 5.0 --device 'cuda:0' | tee Chameleon.txt


# python -u main.py --dataset 'Film' --train_lr 0.1 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 5.0 --device 'cuda:0' | tee Film.txt


# python -u main.py --dataset 'Squirrel' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005  --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee Squirrel.txt

# python -u main.py --dataset 'Cornell' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 1.0 --gamma 1.0 --device 'cuda:0' 


# python -u main.py --dataset 'Texas' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 0.0005 --alpha 1.0 --gamm 1.0 --device 'cuda:0' 


# python -u main.py --dataset 'Wisconsin' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --alpha 5.0 --gamma 1.0 --device 'cuda:0' 


# --------------------------------------------------------------------------------------------------------------------------

python -u large_scale_graphs_main.py --dataset 'arxiv-year' --train_lr 0.01  --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --num_parts 100 --num_splits 1 --alpha 1.0 --gamma 4.0 --device 'cuda:0' | tee output/arxiv_year.txt


# python -u large_scale_graphs_main.py --dataset 'snap-patents' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 1 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 1000 --num_splits 5 --alpha 5.0 --gamma 1.0 --device 'cuda:0' | tee output/snap_patents.txt

# python -u large_scale_graphs_main.py --dataset 'Penn94' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 200 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/Penn94.txt


# python -u large_scale_graphs_main.py --dataset 'pokec' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 500 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 2000 --num_splits 5 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/pokec.txt

# python -u large_scale_graphs_main.py --dataset 'twitch-gamers' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 0.001 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/twitch_gamers.txt


# python -u large_scale_graphs_main.py --dataset 'genius' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 30 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/genius.txt


# --------------------------------------------------

# python -u linkx_main.py --dataset 'arxiv-year' --train_lr 0.01  --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 0.0 --num_parts 100 --num_splits 5 --alpha 2.0 --gamma 1.0 --device 'cpu' | tee arxiv_year.txt

# python -u linkx_main.py --dataset 'snap-patents' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 0.0 --num_parts 1000 --num_splits 5 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee snap_patents.txt

# python -u linkx_main.py --dataset 'Penn94' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 0.0 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee Penn94.txt


# python -u linkx_main.py --dataset 'pokec' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 0.0 --num_parts 2000 --num_splits 5 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee pokec.txt

# python -u linkx_main.py --dataset 'twitch-gamers' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee twitch_gamers.txt

# python -u linkx_main.py --dataset 'genius' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 0.001 --num_parts 100 --num_splits 5 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee genius.txt



# ---------------------------------------------------------

# python -u heterophilous_main.py --dataset 'Roman-empire' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 2.0 --device 'cuda:0' | tee output/roman_empire.txt


# python -u heterophilous_main.py --dataset 'Amazon-ratings' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/aamzon_ratings.txt


# python -u heterophilous_main.py --dataset 'Minesweeper' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0  --gamma 1.0 --device 'cuda:0' | tee output/minesweeper.txt


# python -u heterophilous_main.py --dataset 'Tolokers' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/tolokers.txt


# python -u heterophilous_main.py --dataset 'Questions' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 300 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --alpha 1.0 --gamma 1.0 --device 'cuda:0' | tee output/questions.txt


# -----------------------------------------------------------------

# python -u tu_datasets_main.py --dataset 'ENZYMES' --train_lr 0.001 --seed 0 --num_layers 3 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --num_splits 25 --batch_size 8 --device 'cuda:0' | tee enzymes.txt

# python -u tu_datasets_main.py --dataset 'MUTAG' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th 0.0 --num_splits 25 --batch_size 8 --device 'cuda:0' | tee mutag.txt

# python -u tu_datasets_main.py --dataset 'PROTEINS' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th 0.0 --num_splits 25 --batch_size 8 --device 'cuda:0' | tee proteins.txt

# python -u tu_datasets_main.py --dataset 'COLLAB' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th 0.0 --num_splits 25 --batch_size 8 --device 'cuda:0' | tee collab.txt

# python -u tu_datasets_main.py --dataset 'REDDIT-BINARY' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th 0.0 --num_splits 25 --batch_size 4 --device 'cuda:0' | tee reddit-binary.txt

# python -u tu_datasets_main.py --dataset 'IMDB-BINARY' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th 0.0 --num_splits 25 --batch_size 64 --device 'cuda:0' | tee imdb-binary.txt