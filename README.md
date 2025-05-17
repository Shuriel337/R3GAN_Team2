환경설정

![환경설정](https://github.com/user-attachments/assets/e595dcf0-e999-4ee3-97db-436b77bfb28d)

데이터셋

FFHQ 256*256 출처: https://www.kaggle.com/datasets/denislukovnikov/ffhq256-images-only

70000개 짜리인데 저는 50000개만 사용했습니다.

실행 방법

CUDA_VISIBLE_DEVICES=0 python train.py --outdir=training-runs --data=ffhq256 --gpus=1 --cfg=paper256 --snap=10 --batch=32 --kimg=500 --metrics=fid50k_full --fp32=True

kimg를 상황에 맞게 조절하면 되는데 elice에서 kimg=500으로 하면 6시간 정도 걸렸습니다.

Config A 실행 로그

학습 안 했을 때: {"results": {"fid50k_full": 342.6042459543894}, "metric": "fid50k_full", "total_time": 407.3115086555481, "total_time_str": "6m 47s", "num_gpus": 1, "snapshot_pkl": "network-snapshot-000000.pkl", "timestamp": 1747018878.5895853}

kimg=500 학습 완료 했을 때: {"results": {"fid50k_full": 38.009531625838264}, "metric": "fid50k_full", "total_time": 401.24780344963074, "total_time_str": "6m 41s", "num_gpus": 1, "snapshot_pkl": "network-snapshot-000200.pkl", "timestamp": 1747133366.7522354}

자세한 내용은 StyleGAN Config A 코드 log.txt를 참고하시면 됩니다. (중간에 컴퓨터를 꺼야 할 일이 생겨서 300, 200으로 나누어 학습했습니다. resume을 사용하면 기존 pkl파일 기반으로 이어서 학습할 수 있습니다)


