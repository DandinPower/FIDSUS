#!/bin/bash
# algos=("FedAvg" "FIDSUS" "FedProx")
# datasets=("mnist" "FashionMNIST")
# client_activity_rates=("0.6" "0.8" "1")
algos=("FIDSUS")
datasets=("mnist")
client_activity_rates=("0.6")

jr="1"
batch_size="64"
num_clients=50
num_classes=10
model="microvit"

for algo in "${algos[@]}"; do
    for dataset in "${datasets[@]}"; do
        for client_activity_rate in "${client_activity_rates[@]}"; do
            go_value="_nc${num_clients}"
            python -u main.py -algo "$algo" -jr "$jr" -data "$dataset" -go "$go_value" -nc "$num_clients" \
                        -nb "$num_classes" -car "$client_activity_rate" -lbs "$batch_size" -m "$model"
        done
    done
done