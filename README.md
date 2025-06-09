# 🫀 ECG 기반 심근 비대증(HCM) 조기 진단 및 클래스 불균형 문제 해결

-본 프로젝트는 심전도(ECG) 데이터를 이용해 **심근 비대증(Hypertrophic Cardiomyopathy, HCM)** 을 조기에 진단하고, 의료 데이터에서 흔히 발생하는 **클래스 불균형 문제**를 해결하기 위한 AI 모델 개발을 목표로 합니다.
---

## 🔍 배경

- **ECG(심전도)**는 비침습적으로 심장 상태를 평가할 수 있는 대표적인 생체신호입니다.  
- **심근 비대증(HCM)**은 심장 근육이 비정상적으로 두꺼워지는 질환으로,  
  조기 진단하지 않으면 심부전, 부정맥 등 치명적인 합병증으로 이어질 수 있습니다.  
- 하지만 실제 임상 데이터에서 **비대증 환자의 수는 매우 적어 클래스 불균형 문제가 존재**합니다.

---
## 🧪 프로젝트 개요

![image](https://github.com/user-attachments/assets/e4012787-57de-4de7-b419-ed160fa2da42)
- *데이터 수집부터 예측 모델 생성까지의 전체 흐름*

---
## 🩺 데이터 정보

- **출처**: [PTB-XL](https://physionet.org/content/ptb-xl/1.0.1/) (PhysioNet 공개 ECG 데이터셋)
- **환자 수**: 18,869명  
- **총 기록 수**: 21,799개의 10초 길이 ECG  
- **성비**: 52% 남성 / 48% 여성  
- **연령 분포**: 0 ~ 95세 (중간값 62, 분위수 범위 22)

<img src="https://github.com/user-attachments/assets/78647f3b-bd48-4086-9172-0943c5192670" width="200"/>

## 🏷️ 클래스 구성

![image](https://github.com/user-attachments/assets/654f8a8d-50ec-41eb-bab8-ea307f042f19)


| 클래스 | 설명 | 샘플 수 |
|--------|------|----------|
| NORM   | 정상 심전도 | 9514 |
| HYP    | 심근 비대증 | 2649 |
| 기타   | ST 변화, 전도 장애 등 | 나머지 |

- 본 연구에서는 **NORM vs HYP (정상 vs 비대증)** 이진 분류로 설정  
- 클래스 분포는 약 **4:1**로 **심한 클래스 불균형 존재**
<img src="https://github.com/user-attachments/assets/98680b26-6792-4a40-9806-77bdd5e22da3" width="200"/>
<img src="https://github.com/user-attachments/assets/53854bb5-9672-47da-909d-84d08084918a" width="200"/>

## ⚙️ 데이터 구성 및 처리

- 레이블: `HYP → 0`, `NORM → 1`로 변환
- 데이터 분할 비율: `train:val:test = 7:2:1`
- 의료 데이터 특성상 **데이터 증강은 적용하지 않음**

---

## 🧠 불균형 문제 해결 전략

| 기법 | 설명 |
|------|------|
| `Class Weighting` | 손실 함수에 클래스 비율 기반 가중치 부여 |
| `Focal Loss` | Hard example에 더 집중 |
| `Weighted Random Sampling` | 학습 시 소수 클래스 샘플 우선 선택 |
| `Cost-sensitive Learning` | FP/FN 비용 차이를 고려한 손실 함수 |
| `Under Sampling` | 다수 클래스 일부 제거하여 균형 조정 |

## 🧪 실험 결과

본 연구에서는 다양한 딥러닝 모델(RNN, CNN, ResNet, VGG 등)을 비교 분석했으며,  
CNN 계열의 모델에서 가장 우수한 성능을 보였고, **SEBlock**과 **Class Weight 조정**을 통해 불균형 문제를 효과적으로 해결했습니다.

<img src="https://github.com/user-attachments/assets/0d5c46f2-7d53-4ff8-be5a-d353e66e038e" width="350"/>
## 🧱 모델 아키텍처: CNN2D

본 연구에서 최종적으로 채택된 CNN2D 구조는 다음과 같습니다:

```python
import torch.nn as nn

class CNN2D(nn.Module):
    def __init__(self, num_classes):
        super(CNN2D, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(10, 10))
        self.conv2 = nn.Conv2d(in_channels=100, out_channels=250, kernel_size=(10, 10))
        self.conv3 = nn.Conv2d(in_channels=250, out_channels=500, kernel_size=(10, 10))
        self.conv4 = nn.Conv2d(in_channels=500, out_channels=1000, kernel_size=(10, 10))

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(100)
        self.bn2 = nn.BatchNorm2d(250)
        self.bn3 = nn.BatchNorm2d(500)
        self.bn4 = nn.BatchNorm2d(1000)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout1 = nn.Dropout(0.5)
        self.fc = nn.Linear(1000, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)

        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)

        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout1(x)

        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout1(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

### 📊 모델 비교 결과
- CNN 모델이 전반적인 Accuracy 및 Recall에서 가장 우수
- 특히 HCM 클래스(소수 클래스)의 Recall을 효과적으로 향상시킴
<img src="https://github.com/user-attachments/assets/ca406632-eff9-44b7-bd77-13a1a1d9884c" width="350"/>
---

### 🧪 CNN 구조별 성능 비교

- 다양한 CNN 구조 실험 결과, **Layer-5 / out_1024 / SEBlock** 모델이 가장 우수한 성능 기록
- Recall 및 Loss 모두 개선

---

### ✅ 최종 모델 성능
- **구성**: CNN + SEBlock  
- **설정**: Class Weight 3:1, Epoch 50, Batch Size 64, Learning Rate 0.002  
- **최종 성능**:
  - Accuracy: 0.9457
  - Loss: 0.1602
  - Recall: 0.8461 / 0.9701
  - **F1-Score: 0.8618 / 0.9666**
<img src="https://github.com/user-attachments/assets/be026a83-f4a7-4d6e-b610-5e3cb25ed903" width="350"/>
---

## 🧾 결론 및 기대 효과

- 다양한 **불균형 처리 기법과 아키텍처 최적화**를 통해  
  **정상 vs 심근 비대증(HCM)** 이진 분류에서 **소수 클래스의 성능을 효과적으로 향상**
- 의료 데이터 특성상 **데이터 증강 없이도** 신뢰성 있는 결과를 도출
- 앞으로의 연구에서는 해당 모델을 실시간 진단, 의료기기 연동 등 **의료 현장에 적용 가능한 형태로 확장**할 예정

> 본 연구는 AI 기반 의료 진단에서 **데이터 불균형이라는 현실적인 문제를 해결하며**,  
> **신뢰성 있고 임상적으로 활용 가능한 모델 설계의 가능성**을 보여주었습니다.




