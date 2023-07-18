# k-ium 의료 인공지능 경진대회

---

## 개발 환경

> ### CPU
> - Intel(R) Core(TM) i7-10700F CPU @ 2.90GHz   2.90 GHz
> - 64비트 운영 체제, x64 기반 프로세서
> ### OS
> - Window11 Home
> ### GPU
> - NVIDIA GeForce RTX 3060
> ---
> - pyenv-win version 3.1.1
> - python 3.9.13
> - 자세한 라이브러리는 Pipfile, Pipfile.lock을 통해 확인 가능
> - CUDA version 11.7
> - CuDnn version 8.3

## 사용방법

```bash
# 파이썬 버전 3.9.13 설치(pyenv 권장)
# 3.9.13 환경 위에 pipenv 설치

# 폴더 구조
ckpt
 -- pth file
data
 -- trian_set
  -- 이미지
  -- train.csv
...

git clone https://github.com/sihyeong671/k-ium-medical-AI-competition.git
cd k-ium-medical-AI-competition

pyenv local 3.9.13
pipenv install

pipenv shell
# 학습
# output pth이름은 trainer.py의 torch.save에서 이름 수정 필요
python main.py --mode=train
# 추론
# load_model funcion에서 사용할 모델이름 수정 필요
python main.py --mode=test
```


## 개요
저희는 부산대학교 인공지능 동아리에서 참여한 AID팀입니다. 혈관조영술 이미지만으로 뇌동맥류를 찾는 것은 숙련된 의료 전공자가 필수입니다. 이를 인공지능을 통해 정확하게 해석하고 자동화하는 것은 비용이 많이 드는 진단 기기를 다루기 힘든 지역에 기여할 수 있습니다. 

저희 팀은 크게 4가지의 방법을 시도 하였습니다. 
1) anomalib을 이용한 anomaly detection
2) 각 type 별 classification + grad-cam
3) multilable classification
4) binary classification


저희가 문제해결을 위해 anomaly detection, Classification과 CAM을 통한 위치지정 분류, 그리고 multilabel classification을 시도해보았습니다.

아래는 anomalib의 PaDim 모델을 통해 얻은 결과입니다.
![1428LI-A](anomaly_detection_img/1428LI-A.jpg)
![2408RV-B](anomaly_detection_img/2480RV-B.jpg)
![2565RI-B](anomaly_detection_img/2565RI-B.jpg)

첫번째 이미지처럼 구조의 큰 변화는 잘 찾는 걸 볼 수 있습니다. 하지만 두번째 이미지처럼 혈관이 아닌 이미지를 구분하지 못하고, 세번째 이미지처럼 작은 뇌동맥류를 찾지 못했습니다. 결과를 본 후 뇌동맥류를 찾는데 anomaly detection을 이용하기 힘들다고 판단하였습니다.

Classification과 CAM을 통한 위치지정 분류는 classification을 통해 학습한 모델이 집중한 부분을 특징지어 보려했습니다. 
![class_cam](anomaly_detection_img/class_cam.png)
하지만 위 이미지처럼 모델이 특정한 위치의 부분이 혈관을 가르키지 못했습니다. 데이터의 특성상 값이 0인 비율이 커서 모델 학습이 정확하지 못하다고 판단하였습니다.

Multilabel classification 레이블을 이용하여 supervised learning을 실행하였습니다. 영상 type마다 보이는 부분이 달라 LI-A, LI-B, LV-A, LV-B, RI-A, RI-B, RV-A, RV-B 각각에 대해 모델을 만들었습니다. 레이블의 불균형으로 인해 weighted bceloss를 사용했고 각 모델에 대한 결과값을 종합해보았지만 대부분의 결과값이 0.5를 넘지 못하였고 AUROC score가 0.52정도에 그쳐 다른방법을 선택하기로 했습니다.

## 최종 모델

최종적으로 저희팀은 AUROC 점수가 우선적으로 계산되기 때문에 위치보다 뇌동맥류 여부에 집중을 하기로 결정했습니다. 주어진 8장의 영상이미지를 합쳐 하나의 input으로 모델에 넣어서 최종적으로 뇌동맥류를 판단하도록 하는 모델로 구성했습니다.

모델은 작으면서 결과값이 안정적으로 나오는 Efficientnet b4를 선택했고 timm라이브러리의 사전학습된 모델을 사용했습니다.

### 학습 세부 사항

1. Augmentation
- ElasticTransform
- GaussianBlur
- 256 x 256 resize
- 8장의 이미지 GraySacle 변환후 concatenation하여 input으로 사용
2. HyperParameter
- epoch: 20
- optimizer: Adam
- learning rate: 0.001
3. 

### 결과

C-statistic(AUROC score):  