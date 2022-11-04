# nohup python -u train.py --dataset yelp2018 --device 4 --drop_ratio 0.05 --t 0.11 --a 20 --norm_type 0.6 --beta 0 > logs/yelp_beta_0_t_0.11_drop_0.05_a_20.log 2>&1 &
# nohup python -u train.py --dataset yelp2018 --device 7 --drop_ratio 0.1 --t 0.11 --a 20 --norm_type 0.6 --beta 0.05 > logs/yelp_beta_0.05_t_0.11_drop_0.1_a_20.log 2>&1 &
# nohup python -u train.py --dataset yelp2018 --device 4 --drop_ratio 0.05 --t 0.13 --a 20 --norm_type 0.6 --beta 1 --alpha 0 > logs/yelp2018/yelp_pt_onlyRD_t_0.13_drop_0.05_a_20.log 2>&1 &
# nohup python -u train.py --dataset yelp2018 --device 4 --drop_ratio 0.05 --t 0.09 --a 20 --norm_type 0.6 --beta 1 --alpha 0 > logs/yelp2018/yelp_pt_onlyRD_t_0.09_drop_0.05_a_20.log 2>&1 &
# nohup python -u train.py --dataset yelp2018 --device 1 --drop_ratio 0.05 --t 0.07 --a 20 --norm_type 0.6 --beta 1 --alpha 0 > logs/yelp2018/yelp_pt_onlyRD_t_0.07_drop_0.05_a_20.log 2>&1 &
# nohup python -u train.py --dataset yelp2018 --device 7 --drop_ratio 0.05 --t 0.15 --a 20 --norm_type 0.6 --beta 1 --alpha 0 > logs/yelp2018/yelp_pt_onlyRD_t_0.15_drop_0.05_a_20.log 2>&1 &

# beta
# nohup python -u train.py --dataset yelp2018 --device 0 --drop_ratio 0.1 --t 0.11 --a 20 --norm_type 0.6 --beta 0.2 > logs/yelp2018/beta/0.2.log 2>&1 &
# nohup python -u train.py --dataset yelp2018 --device 0 --drop_ratio 0.1 --t 0.11 --a 20 --norm_type 0.6 --beta 0.3 > logs/yelp2018/beta/0.3.log 2>&1 &
# nohup python -u train.py --dataset yelp2018 --device 5 --drop_ratio 0.1 --t 0.11 --a 20 --norm_type 0.6 --beta 0.4 > logs/yelp2018/beta/0.4.log 2>&1 &
# nohup python -u train.py --dataset yelp2018 --device 5 --drop_ratio 0.1 --t 0.11 --a 20 --norm_type 0.6 --beta 0.5 > logs/yelp2018/beta/0.5.log 2>&1 &
# nohup python -u train.py --dataset yelp2018 --device 7 --drop_ratio 0.1 --t 0.11 --a 20 --norm_type 0.6 --beta 0.6 > logs/yelp2018/beta/0.6.log 2>&1 &
# nohup python -u train.py --dataset yelp2018 --device 7 --drop_ratio 0.1 --t 0.11 --a 20 --norm_type 0.6 --beta 0.7 > logs/yelp2018/beta/0.7.log 2>&1 &
# nohup python -u train.py --dataset yelp2018 --device 4 --drop_ratio 0.1 --t 0.11 --a 20 --norm_type 0.6 --beta 0.8 > logs/yelp2018/beta/0.8.log 2>&1 &
# nohup python -u train.py --dataset yelp2018 --device 4 --drop_ratio 0.1 --t 0.11 --a 20 --norm_type 0.6 --beta 0.9 > logs/yelp2018/beta/0.9.log 2>&1 &
# nohup python -u train.py --dataset yelp2018 --device 3 --drop_ratio 0.1 --t 0.11 --a 20 --norm_type 0.6 --beta 1 > logs/yelp2018/beta/1.log 2>&1 &

# nohup python -u train.py --dataset amazon-book --device 4 --drop_ratio 0.05 --t 0.03 --a 20 --norm_type 0.6 --beta 1 --testbatch 20 > logs/amazon_beta_1_t_0.03_drop_0.05_a_20_norm_0.6.log 2>&1 &
# nohup python -u train.py --dataset amazon-book --device 5 --drop_ratio 0.05 --t 0.08 --a 20 --norm_type 0.6 --beta 1 --testbatch 20 > logs/amazon_beta_1_t_0.08_drop_0.05_a_20_norm_0.6.log 2>&1 &
# nohup python -u train.py --dataset amazon-book --device 6 --drop_ratio 0.05 --t 0.1 --a 20 --norm_type 0.6 --beta 1 --testbatch 20 > logs/amazon_beta_1_t_0.1_drop_0.05_a_20_norm_0.6.log 2>&1 &
# nohup python -u train.py --dataset amazon-book --device 7 --drop_ratio 0.05 --t 0.12 --a 20 --norm_type 0.6 --beta 1 --testbatch 20 > logs/amazon_beta_1_t_0.12_drop_0.05_a_20_norm_0.6.log 2>&1 &

# nohup python -u train.py --dataset kindle-store --device 0 --drop_ratio 0.05 --t 0.10 --a 0 --norm_type row --beta 0.2 > logs/kindle/beta_0.2_t_0.10_drop_0.05_a_0.log 2>&1 &
# nohup python -u train.py --dataset kuaishou-video --device 2 --drop_ratio 0.01 --t 0.10 --a 0 --norm_type row --beta 0. --testbatch 20 > logs/kuaishou/beta_0_t_0.10_drop_0.01_a_0.log 2>&1 &
# nohup python -u train.py --dataset kindle-store --device 7 --drop_ratio 0.05 --t 0.10 --a 0 --norm_type row --beta 0.4 > logs/kindle/beta_0.4_t_0.10_drop_0.05_a_0.log 2>&1 &

# nohup python -u train.py --dataset kindle-store --device 2 --drop_ratio 0.05 --t 0.10 --a 5 --norm_type row --beta 0.6 > logs/kindle/beta_0.6_t_0.10_drop_0.05_a_5.log 2>&1 &
# nohup python -u train.py --dataset kindle-store --device 4 --drop_ratio 0.05 --t 0.10 --a 5 --norm_type row --beta 0.7 > logs/kindle/beta_0.7_t_0.10_drop_0.05_a_5.log 2>&1 &
# nohup python -u train.py --dataset kindle-store --device 0 --drop_ratio 0.05 --t 0.10 --a 5 --norm_type row --beta 0.8 > logs/kindle/beta_0.8_t_0.10_drop_0.05_a_5.log 2>&1 &

# nohup python -u train.py --dataset kuaishou-video --device 0 --drop_ratio 0.01 --t 0.07 --a 0 --norm_type row --beta 0.2 --testbatch 20 > logs/kuaishou/beta_0.2_t_0.07_drop_0.01_a_0.log 2>&1 &
# nohup python -u train.py --dataset kuaishou-video --device 2 --drop_ratio 0.01 --t 0.09 --a 0 --norm_type row --beta 0.2 --testbatch 20 > logs/kuaishou/beta_0.2_t_0.09_drop_0.01_a_0.log 2>&1 &
# nohup python -u train.py --dataset kuaishou-video --device 3 --drop_ratio 0.01 --t 0.06 --a 0 --norm_type row --beta 0.2 --testbatch 20 > logs/kuaishou/beta_0.2_t_0.06_drop_0.01_a_0.log 2>&1 &
# nohup python -u train.py --dataset kuaishou-video --device 6 --drop_ratio 0.01 --t 0.10 --a 0 --norm_type row --beta 0.3 --testbatch 20 > logs/kuaishou/beta_0.3_t_0.10_drop_0.01_a_0.log 2>&1 &

# nohup python -u train.py --dataset kuaishou-video --device 2 --drop_ratio 0.01 --t 0.07 --a 20 --norm_type 0.5 --beta 0.2 --testbatch 20 > logs/kuaishou/beta_0.2_t_0.07_drop_0.01_a_20_norm_0.5.log 2>&1 &
# nohup python -u train.py --dataset kuaishou-video --device 1 --drop_ratio 0.01 --t 0.07 --a 20 --norm_type 0.5 --beta 0.3 --testbatch 20 > logs/kuaishou/beta_0.3_t_0.07_drop_0.01_a_20_norm_0.5.log 2>&1 &
# nohup python -u train.py --dataset kuaishou-video --device 2 --drop_ratio 0.01 --t 0.06 --a 20 --norm_type 0.5 --beta 0.3 --testbatch 20 > logs/kuaishou/beta_0.3_t_0.06_drop_0.01_a_20_norm_0.5.log 2>&1 &
# nohup python -u train.py --dataset kuaishou-video --device 3 --drop_ratio 0.01 --t 0.06 --a 20 --norm_type 0.5 --beta 0.4 --testbatch 20 > logs/kuaishou/beta_0.4_t_0.06_drop_0.01_a_20_norm_0.5.log 2>&1 &
# nohup python -u train.py --dataset kuaishou-video --device 4 --drop_ratio 0.01 --t 0.07 --a 20 --norm_type 0.5 --beta 0.4 --testbatch 20 > logs/kuaishou/beta_0.4_t_0.07_drop_0.01_a_20_norm_0.5.log 2>&1 &
# nohup python -u train.py --dataset kuaishou-video --device 6 --drop_ratio 0.01 --t 0.08 --a 20 --norm_type 0.5 --beta 0.2 --testbatch 20 > logs/kuaishou/beta_0.2_t_0.08_drop_0.01_a_20_norm_0.5.log 2>&1 &
# nohup python -u train.py --dataset kuaishou-video --device 5 --drop_ratio 0.05 --t 0.11 --a 20 --norm_type 0.55 --beta 1 --testbatch 20 > logs/kuaishou/beta_1_t_0.11_drop_0.05_a_20_norm_0.55.log 2>&1 &
# nohup python -u train.py --dataset kuaishou-video --device 4 --drop_ratio 0.05 --t 0.07 --a 20 --norm_type 0.55 --beta 1 --testbatch 20 > logs/kuaishou/beta_1_t_0.07_drop_0.05_a_20_norm_0.55.log 2>&1 &
# nohup python -u train.py --dataset kuaishou-video --device 5 --drop_ratio 0.05 --t 0.05 --a 20 --norm_type 0.55 --beta 0 --testbatch 20 > logs/kuaishou/beta_0_t_0.05_drop_0.05_a_20_norm_0.55.log 2>&1 &
# nohup python -u train.py --dataset kuaishou-video --device 4 --drop_ratio 0.05 --t 0.11 --a 20 --norm_type 0.55 --beta 0 --testbatch 20 > logs/kuaishou/beta_0_t_0.11_drop_0.05_a_20_norm_0.55.log 2>&1 &

# nohup python -u train.py --dataset homekitchen --device 0 --drop_ratio 0.2 --t 0.09 --a 20 --norm_type 1 --beta 0.8 > logs/homekitchen/beta_0.8_t_0.09_drop_0.2_a_20_norm_1.log 2>&1 &
# nohup python -u train.py --dataset homekitchen --device 1 --drop_ratio 0.2 --t 0.09 --a 20 --norm_type 1 --beta 0.9 > logs/homekitchen/beta_0.9_t_0.09_drop_0.2_a_20_norm_1.log 2>&1 &
# nohup python -u train.py --dataset homekitchen --device 2 --drop_ratio 0.2 --t 0.09 --a 20 --norm_type 1 --beta 1 > logs/homekitchen/beta_1_t_0.09_drop_0.2_a_20_norm_1.log 2>&1 &
# nohup python -u train.py --dataset homekitchen --device 6 --drop_ratio 0.2 --t 0.09 --a 20 --norm_type 1 --beta 1.2 > logs/homekitchen/beta_1.2_t_0.09_drop_0.2_a_20_norm_1.log 2>&1 &

#beta
# nohup python -u train.py --dataset homekitchen --device 0 --drop_ratio 0.2 --t 0.11 --a 20 --norm_type 1 --beta 0 > logs/homekitchen/beta/0.log 2>&1 &
# nohup python -u train.py --dataset homekitchen --device 1 --drop_ratio 0.2 --t 0.11 --a 20 --norm_type 1 --beta 0.1 > logs/homekitchen/beta/0.1.log 2>&1 &
# nohup python -u train.py --dataset homekitchen --device 2 --drop_ratio 0.2 --t 0.11 --a 20 --norm_type 1 --beta 0.2 > logs/homekitchen/beta/0.2.log 2>&1 &
# nohup python -u train.py --dataset homekitchen --device 3 --drop_ratio 0.2 --t 0.11 --a 20 --norm_type 1 --beta 0.3 > logs/homekitchen/beta/0.3.log 2>&1 &
# nohup python -u train.py --dataset homekitchen --device 4 --drop_ratio 0.2 --t 0.11 --a 20 --norm_type 1 --beta 0.4 > logs/homekitchen/beta/0.4.log 2>&1 &

# nohup python -u train.py --dataset gowalla --device 3 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --seed 108456 > logs/gowalla/seed_no_gamma/9.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 4 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --seed 629753 > logs/gowalla/seed_no_gamma/8.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 6 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --seed 844421 > logs/gowalla/seed_no_gamma/10.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 0 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --seed 89327489 > logs/gowalla/seed_no_gamma/6.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 5 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --seed 9732646 > logs/gowalla/seed_no_gamma/7.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 4 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --seed 75466 > logs/gowalla/seed_no_gamma/2.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 4 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --seed 86577 > logs/gowalla/seed_no_gamma/3.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 4 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --seed 786756 > logs/gowalla/seed_no_gamma/4.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 6 --drop_ratio 0.1 --t 0.04 --a 0 --norm_type 0.9 --beta 1 --alpha 0  > logs/gowalla/gowalla_no_oi_t_0.04_drop_0.1_a_0_norm_0.9.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 3 --drop_ratio 0.12 --t 0.08 --a 0 --norm_type 0.9 --beta 1 --alpha 0  > logs/gowalla/gowalla_no_oi_t_0.12_drop_0.1_a_0_norm_0.9.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 5 --drop_ratio 0.1 --t 0.14 --a 0 --norm_type 0.9 --beta 1 --alpha 0  > logs/gowalla/gowalla_no_oi_t_0.14_drop_0.1_a_0_norm_0.9.log 2>&1 &

# nohup python -u train.py --dataset gowalla --device 0 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --seed 629753 --epsilon 200 > logs/gowalla/epsilon/ep_200.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 0 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --seed 9512341 --epsilon 250 > logs/gowalla/epsilon/7_ep_250.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 1 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --seed 69741234 --epsilon 250 > logs/gowalla/epsilon/8_ep_250.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 4 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --seed 9324231 --epsilon 250 > logs/gowalla/epsilon/4_ep_250.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 6 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --seed 1834241 --epsilon 250 > logs/gowalla/epsilon/5_ep_250.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 7 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --seed 23423412 --epsilon 250 > logs/gowalla/epsilon/6_ep_250.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 2 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --seed 629753 --epsilon 300 > logs/gowalla/epsilon/ep_300.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 5 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --seed 629753 --epsilon 80 > logs/gowalla/epsilon/ep_80.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 7 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --seed 629753 --epsilon 100 > logs/gowalla/epsilon/ep_100.log 2>&1 &

# nohup python -u train.py --dataset gowalla --device 1 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --noise 0.05 > logs/gowalla/noise/noise_0.05.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 1 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --noise 0.1 > logs/gowalla/noise/noise_0.1.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 2 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --noise 0.15 > logs/gowalla/noise/noise_0.15.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 2 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --noise 0.2 > logs/gowalla/noise/noise_0.2.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 5 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --noise 0.25 > logs/gowalla/noise/noise_0.25.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 5 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --noise 0.3 > logs/gowalla/noise/noise_0.3.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 7 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --noise 0.35 > logs/gowalla/noise/noise_0.35.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 7 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --noise 0.4 > logs/gowalla/noise/noise_0.4.log 2>&1 &

# nohup python -u train.py --dataset yelp2018 --device 4 --drop_ratio 0.05 --t 0.11 --a 20 --norm_type 0.6 --beta 0.05 --noise 0.05 > logs/yelp2018/noise/noise_0.05.log 2>&1 &
# nohup python -u train.py --dataset yelp2018 --device 5 --drop_ratio 0.05 --t 0.11 --a 20 --norm_type 0.6 --beta 0.05 --noise 0.1 > logs/yelp2018/noise/noise_0.1.log 2>&1 &
# nohup python -u train.py --dataset yelp2018 --device 6 --drop_ratio 0.05 --t 0.11 --a 20 --norm_type 0.6 --beta 0.05 --noise 0.15 > logs/yelp2018/noise/noise_0.15.log 2>&1 &
# nohup python -u train.py --dataset yelp2018 --device 7 --drop_ratio 0.05 --t 0.11 --a 20 --norm_type 0.6 --beta 0.05 --noise 0.2 > logs/yelp2018/noise/noise_0.2.log 2>&1 &
# nohup python -u train.py --dataset yelp2018 --device 4 --drop_ratio 0.05 --t 0.11 --a 20 --norm_type 0.6 --beta 0.05 --noise 0.25 > logs/yelp2018/noise/noise_0.25.log 2>&1 &
# nohup python -u train.py --dataset yelp2018 --device 5 --drop_ratio 0.05 --t 0.11 --a 20 --norm_type 0.6 --beta 0.05 --noise 0.3 > logs/yelp2018/noise/noise_0.3.log 2>&1 &
# nohup python -u train.py --dataset yelp2018 --device 6 --drop_ratio 0.05 --t 0.11 --a 20 --norm_type 0.6 --beta 0.05 --noise 0.35 > logs/yelp2018/noise/noise_0.35.log 2>&1 &
# nohup python -u train.py --dataset yelp2018 --device 7 --drop_ratio 0.05 --t 0.11 --a 20 --norm_type 0.6 --beta 0.05 --noise 0.4 > logs/yelp2018/noise/noise_0.4.log 2>&1 &

# nohup python -u train.py --dataset homekitchen --device 4 --drop_ratio 0.2 --t 0.11 --a 20 --norm_type 1 --beta 1 --noise 0.25 > logs/homekitchen/noise/noise_0.25.log 2>&1 &
# nohup python -u train.py --dataset homekitchen --device 5 --drop_ratio 0.2 --t 0.11 --a 20 --norm_type 1 --beta 1 --noise 0.3 > logs/homekitchen/noise/noise_0.3.log 2>&1 &
# nohup python -u train.py --dataset homekitchen --device 6 --drop_ratio 0.2 --t 0.11 --a 20 --norm_type 1 --beta 1 --noise 0.35 > logs/homekitchen/noise/noise_0.35.log 2>&1 &
# nohup python -u train.py --dataset homekitchen --device 7 --drop_ratio 0.2 --t 0.11 --a 20 --norm_type 1 --beta 1 --noise 0.4 > logs/homekitchen/noise/noise_0.4.log 2>&1 &

# nohup python -u train.py --dataset amazon-cd --device 3 --drop_ratio 0.2 --t 0.13 --a 30 --norm_type 0.45 --beta 0.1 --noise 0.05 > logs/cd/noise/noise_0.05.log 2>&1 &
# nohup python -u train.py --dataset amazon-cd --device 3 --drop_ratio 0.2 --t 0.13 --a 30 --norm_type 0.45 --beta 0.1 --noise 0.1 > logs/cd/noise/noise_0.1.log 2>&1 &
# nohup python -u train.py --dataset amazon-cd --device 4 --drop_ratio 0.2 --t 0.13 --a 30 --norm_type 0.45 --beta 0.1 --noise 0.15 > logs/cd/noise/noise_0.15.log 2>&1 &
# nohup python -u train.py --dataset amazon-cd --device 4 --drop_ratio 0.2 --t 0.13 --a 30 --norm_type 0.45 --beta 0.1 --noise 0.2 > logs/cd/noise/noise_0.2.log 2>&1 &
# nohup python -u train.py --dataset amazon-cd --device 5 --drop_ratio 0.2 --t 0.13 --a 30 --norm_type 0.45 --beta 0.1 --noise 0.25 > logs/cd/noise/noise_0.25.log 2>&1 &
# nohup python -u train.py --dataset amazon-cd --device 5 --drop_ratio 0.2 --t 0.13 --a 30 --norm_type 0.45 --beta 0.1 --noise 0.3 > logs/cd/noise/noise_0.3.log 2>&1 &
# nohup python -u train.py --dataset amazon-cd --device 2 --drop_ratio 0.2 --t 0.13 --a 30 --norm_type 0.45 --beta 0.1 --noise 0.35 > logs/cd/noise/noise_0.35.log 2>&1 &
# nohup python -u train.py --dataset amazon-cd --device 2 --drop_ratio 0.2 --t 0.13 --a 30 --norm_type 0.45 --beta 0.1 --noise 0.4 > logs/cd/noise/noise_0.4.log 2>&1 &

# nohup python -u train.py --dataset gowalla --device 1 --drop_ratio 1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --noise 0.05 > logs/gowalla/noise/no_drop_noise_0.05.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 1 --drop_ratio 1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --noise 0.1 > logs/gowalla/noise/no_drop_noise_0.1.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 2 --drop_ratio 1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --noise 0.15 > logs/gowalla/noise/no_drop_noise_0.15.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 2 --drop_ratio 1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --noise 0.2 > logs/gowalla/noise/no_drop_noise_0.2.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 5 --drop_ratio 1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --noise 0.25 > logs/gowalla/noise/no_drop_noise_0.25.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 5 --drop_ratio 1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --noise 0.3 > logs/gowalla/noise/no_drop_noise_0.3.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 7 --drop_ratio 1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --noise 0.35 > logs/gowalla/noise/no_drop_noise_0.35.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 7 --drop_ratio 1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4 --noise 0.4 > logs/gowalla/noise/no_drop_noise_0.4.log 2>&1 &

# nohup python -u train.py --dataset yelp2018 --device 4 --drop_ratio 1 --t 0.11 --a 20 --norm_type 0.6 --beta 0.05 --noise 0.05 > logs/yelp2018/noise/no_drop_noise_0.05.log 2>&1 &
# nohup python -u train.py --dataset yelp2018 --device 5 --drop_ratio 1 --t 0.11 --a 20 --norm_type 0.6 --beta 0.05 --noise 0.1 > logs/yelp2018/noise/no_drop_noise_0.1.log 2>&1 &
# nohup python -u train.py --dataset yelp2018 --device 6 --drop_ratio 1 --t 0.11 --a 20 --norm_type 0.6 --beta 0.05 --noise 0.15 > logs/yelp2018/noise/no_drop_noise_0.15.log 2>&1 &
# nohup python -u train.py --dataset yelp2018 --device 7 --drop_ratio 1 --t 0.11 --a 20 --norm_type 0.6 --beta 0.05 --noise 0.2 > logs/yelp2018/noise/no_drop_noise_0.2.log 2>&1 &
# nohup python -u train.py --dataset yelp2018 --device 4 --drop_ratio 1 --t 0.11 --a 20 --norm_type 0.6 --beta 0.05 --noise 0.25 > logs/yelp2018/noise/no_drop_noise_0.25.log 2>&1 &
# nohup python -u train.py --dataset yelp2018 --device 5 --drop_ratio 1 --t 0.11 --a 20 --norm_type 0.6 --beta 0.05 --noise 0.3 > logs/yelp2018/noise/no_drop_noise_0.3.log 2>&1 &
# nohup python -u train.py --dataset yelp2018 --device 6 --drop_ratio 1 --t 0.11 --a 20 --norm_type 0.6 --beta 0.05 --noise 0.35 > logs/yelp2018/noise/no_drop_noise_0.35.log 2>&1 &
# nohup python -u train.py --dataset yelp2018 --device 7 --drop_ratio 1 --t 0.11 --a 20 --norm_type 0.6 --beta 0.05 --noise 0.4 > logs/yelp2018/noise/no_drop_noise_0.4.log 2>&1 &


# nohup python -u train.py --dataset homekitchen --device 4 --drop_ratio 1 --t 0.11 --a 20 --norm_type 1 --beta 1 --noise 0.05 > logs/homekitchen/noise/no_drop_noise_0.05.log 2>&1 &
# nohup python -u train.py --dataset homekitchen --device 5 --drop_ratio 1 --t 0.11 --a 20 --norm_type 1 --beta 1 --noise 0.1 > logs/homekitchen/noise/no_drop_noise_0.1.log 2>&1 &
# nohup python -u train.py --dataset homekitchen --device 6 --drop_ratio 1 --t 0.11 --a 20 --norm_type 1 --beta 1 --noise 0.15 > logs/homekitchen/noise/no_drop_noise_0.15.log 2>&1 &
# nohup python -u train.py --dataset homekitchen --device 7 --drop_ratio 1 --t 0.11 --a 20 --norm_type 1 --beta 1 --noise 0.2 > logs/homekitchen/noise/no_drop_noise_0.2.log 2>&1 &
# nohup python -u train.py --dataset homekitchen --device 4 --drop_ratio 1 --t 0.11 --a 20 --norm_type 1 --beta 1 --noise 0.25 > logs/homekitchen/noise/no_drop_noise_0.25.log 2>&1 &
# nohup python -u train.py --dataset homekitchen --device 5 --drop_ratio 1 --t 0.11 --a 20 --norm_type 1 --beta 1 --noise 0.3 > logs/homekitchen/noise/no_drop_noise_0.3.log 2>&1 &
# nohup python -u train.py --dataset homekitchen --device 6 --drop_ratio 1 --t 0.11 --a 20 --norm_type 1 --beta 1 --noise 0.35 > logs/homekitchen/noise/no_drop_noise_0.35.log 2>&1 &
# nohup python -u train.py --dataset homekitchen --device 7 --drop_ratio 1 --t 0.11 --a 20 --norm_type 1 --beta 1 --noise 0.4 > logs/homekitchen/noise/no_drop_noise_0.4.log 2>&1 &

# nohup python -u train.py --dataset gowalla --device 5 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.3 --noise 0.4 > logs/gowalla/beta_0.3_t_0.06_drop_0.1_a_0_norm_0.9.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 6 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.9 --noise 0.4 > logs/gowalla/beta_0.9_t_0.06_drop_0.1_a_0_norm_0.9.log 2>&1 &
# nohup python -u train.py --dataset gowalla --device 7 --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 1 --noise 0.4 > logs/gowalla/beta_1_t_0.06_drop_0.1_a_0_norm_0.9.log 2>&1 &

# nohup python -u train.py --dataset amazon-cd --device 0 --drop_ratio 0.2 --t 0.13 --a 30 --norm_type 0.45 --beta 0.2 --noise 0.4 > logs/cd/beta/0.2.log 2>&1 &
# nohup python -u train.py --dataset amazon-cd --device 0 --drop_ratio 0.2 --t 0.13 --a 30 --norm_type 0.45 --beta 0.3 --noise 0.4 > logs/cd/beta/0.3.log 2>&1 &
# nohup python -u train.py --dataset amazon-cd --device 1 --drop_ratio 0.2 --t 0.13 --a 30 --norm_type 0.45 --beta 0.4 --noise 0.4 > logs/cd/beta/0.4.log 2>&1 &
# nohup python -u train.py --dataset amazon-cd --device 5 --drop_ratio 0.2 --t 0.13 --a 30 --norm_type 0.45 --beta 0.5 --noise 0.4 > logs/cd/beta/0.5.log 2>&1 &
# nohup python -u train.py --dataset amazon-cd --device 5 --drop_ratio 0.2 --t 0.13 --a 30 --norm_type 0.45 --beta 0.6 --noise 0.4 > logs/cd/beta/0.6.log 2>&1 &
# nohup python -u train.py --dataset amazon-cd --device 6 --drop_ratio 0.2 --t 0.13 --a 30 --norm_type 0.45 --beta 0.7 --noise 0.4 > logs/cd/beta/0.7.log 2>&1 &
# nohup python -u train.py --dataset amazon-cd --device 6 --drop_ratio 0.2 --t 0.13 --a 30 --norm_type 0.45 --beta 0.8 --noise 0.4 > logs/cd/beta/0.8.log 2>&1 &
# nohup python -u train.py --dataset amazon-cd --device 7 --drop_ratio 0.2 --t 0.13 --a 30 --norm_type 0.45 --beta 0.9 --noise 0.4 > logs/cd/beta/0.9.log 2>&1 &
# nohup python -u train.py --dataset amazon-cd --device 7 --drop_ratio 0.2 --t 0.13 --a 30 --norm_type 0.45 --beta 1 --noise 0.4 > logs/cd/beta/1.log 2>&1 &

nohup python -u train.py --dataset homekitchen --device 2 --drop_ratio 0.2 --t 0.11 --a 20 --norm_type 1 --beta 1 --alpha 0 --noise 0.4 > logs/homekitchen/only_beta.log 2>&1 &
