nvidia-docker run -v $HOME:/mnt -v $HOME/dataset:/dataset chainermn \
  mpiexec --allow-run-as-root -n 2 \
  python3 /mnt/mobilenet/train_mobilenet_mn.py /dataset/imagenet/train.txt --mean /mnt/mobilenet/mean.npy --epoch 100 --batchsize 64 --out /mnt/mobilenet/result --gpu
