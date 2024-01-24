data_dir="/datasets/CIFAR10"
amp="--amp"
opt="--opt sgd"
channels="--channels 128"
epochs="--epochs 64"

for tau in 1 3 5 8 10 15 20
do
    command="python train.py -data-dir $data_dir $amp $opt $channels $epochs -tau $tau"

    echo "Executing command: $command"
    $command
    echo ""
done
