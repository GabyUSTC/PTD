# PTD
This is the Pytorch implementation for our paper "Propagation then Distillation: Understanding and Improving Linear GCNs for Recommendation"

## Enviroment Requirement
`pip install -r requirements.txt`

## Dataset
We provide four processed datasets: Gowalla, Yelp2018, Home&Kitchen and Amazon-CD.

## Commands to Reproduce Our Results
Gowalla:
`python -u train.py --dataset gowalla --drop_ratio 0.1 --t 0.06 --a 0 --norm_type 0.9 --beta 0.4`

Yelp2018:
`python -u train.py --dataset yelp2018 --drop_ratio 0.1 --t 0.11 --a 20 --norm_type 0.6 --beta 1`

Home&Kitchen:
`python -u train.py --dataset homekitchen --drop_ratio 0.2 --t 0.11 --a 30 --norm_type 1 --beta 1`

Amazon_CD:
`python -u train.py --dataset amazon-cd --drop_ratio 0.2 --t 0.13 --a 30 --norm_type 0.45 --beta 0.1`
