###CatBoost Architecture+DiCE algorithm for ML###

모델 사용 및 역할 정의
이 모델은 ‘이직 성공 가능성’을 예측하고, 실패로 예측된 사용자에게 성공 가능성을 높이기 위한 행동 가이드(recourse)를 제공하는 모델이다.
- 1차 역할: 개인의 현재 상태를 기반으로 이직 성공 확률을 예측
- 2차 역할: 실패로 예측된 경우, 어떤 요소를 어떻게 바꾸면 성공으로 분류되는지를 제안

모델 목적 1문장
개인의 구직·교육·경력 관련 정보를 바탕으로 이직 성공 가능성을 예측하고, 성공 확률을 높이기 위한 최소한의 현실적 변화 시나리오를 제시하는 것이 목적이다.

타깃(y) 정의
타깃 변수: 이전직장사항_전직유무
0: 이직 실패
1: 이직 성공
이 변수는 실제 조사 데이터에서 관측된 이직 결과로, 모델은 이를 이진 분류(binary classification) 문제로 학습한다.

손실 함수 & 평가 지표 선택 이유
손실 함수 (Training)
Logloss (Binary Cross-Entropy)

선택 이유
단순 정답/오답이 아니라 확률 예측의 정확성을 학습할 수 있음. 이후 precision 중심 threshold 정책 및 recourse 생성에 적합

평가지표 (Metrics)
Precision (Class = 1, 이직 성공)
F1-score (보조 지표)
ROC-AUC (참고 지표)
<Precision을 중점으로 본 이유>
이직 성공으로 예측했는데 실제로 실패하는 경우(False Positive)는
사용자에게 잘못된 기대/결정 유도라는 리스크가 큼
따라서 “성공이라고 말할 때는 최대한 확실해야 함”
이 모델은 Recall보다 Precision을 더 중요하게 설계됨

Train / Test 성능 비교
Train Accuracy: 약 88%
Test Accuracy: 약 82%
Test Precision (이직 성공): 약 0.72
<해석>
전체 정확도는 안정적이며, 이직 성공 클래스에 대해 보수적인 예측(precision 중심) 성향을 보임
이는 서비스 목적(잘못된 성공 예측 최소화)에 부합

Overfitting 점검 결과
다음 기준으로 과적합을 점검하였다:
Train vs Test 성능 비교
Precision / Recall 간 괴리 확인
CatBoost depth, regularization(l2_leaf_reg) 튜닝 결과 확인

<결론>
Train 성능 대비 Test 성능이 급격히 하락하지 않음
모델은 과적합보다는 일반화 성향을 보임
일부 recall 저하는 의도된 설계(precision 우선 정책)의 결과로 해석 가능

시각화 등 (선택사항)
