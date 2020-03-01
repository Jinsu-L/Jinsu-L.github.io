---
title: Machine Learning Classification Matric
tags: MachineLearning
category: MachineLearing
---

>  머신 러닝에서 사용하는 여러가지 classification metric을  정리하기 위한 페이지
>
> 새로 알아가는 Metric이 있을 때 마다, 추가할 예정



## Classification Matric 

sklearn Classification metrics : https://scikit-learn.org/stable/modules/classes.html#classification-metrics



### Confusion matrix

라벨이 있는 경우에 대한 모델 평가 matric

* 데이터 상 라벨과, 모델이 예측한 결과에 대한 데이터를 4가지 케이스로 분리
* 양성인데, 양성으로 예측 -> True Positive (TP)
* 음성인데, 음성으로 예측 -> True Negative (TN)
* 양성인데, 음성으로 예측 -> False Negative (FN)
* 음성인데, 양성으로 예측 -> False Positive (FP)

#### Matrix

|                   | Pred Positive | Pred Negative |      |
| ----------------- | ------------- | ------------- | ---- |
| Observed Positive | TP            | FN            | P    |
| Observed Negative | FP            | TN            | N    |

#### 구현

```python
from sklearn.metrics import confusion_matrix

y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
confusion_matrix(y_true, y_pred)
# output
"""
	array([[2, 0, 0],
  	     [0, 0, 1],
    	   [1, 0, 2]])
"""

tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
(tn, fp, fn, tp)
# output
"""
	(0, 2, 1, 1)
"""

```

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix



### Accuracy - 정확도

정확도는 전체 데이터 중 제대로 분류된 데이터의 비율을 뜻한다. 높을 수록 좋은 Matric.

* *(TP + TN) / (P + N)*

```python
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
accuracy_score(y_true, y_pred)
# 0.5
accuracy_score(y_true, y_pred, normalize=False)
# 2
```



### Error Rate - 오류율

오류율은 정확도와 반대 지표, 전체 데이터 중에서 잘못 분류된 데이터의 비율을 뜻한다. 낮을 수록 좋은 Matric

* *(FP + FN) / (P + N)*


### Sensitivity(Recall) - 민감도(재현율)

재현율은 실제 positive 중에서 positive로 분류한 비율을 뜻한다. 높을 수록 좋은 Matric, 

검색 랭킹에서 사용되면, 질의와 관련 있는 문서들 중 실제로 검색된 문서들의 비율.

Sensitivity == Recall == True Positive Rate(TPR) 이라고도 한다.

* *(TP) / P*

```python
from sklearn.metrics import recall_score
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
recall_score(y_true, y_pred, average='macro')
# 0.33...
```



### Precision - 정밀도

positve 라고 예측한 데이터 중 실제 Positive의 비율을 뜻함. 높을 수록 좋은 Matric.

검색된 문서들 중 관련 있는 문서들의 비율.

* *TP / ( TP + FP)*

```python
from sklearn.metrics import precision_score
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
precision_score(y_true, y_pred, average='macro')
# 0.22...
```



### Specificity - 특이도

negative 데이터 중에서 negative로 분류한 것들의 비율

* TN / N

* Specificity == 1 - False Positive Rate == 1 - FP / N

```python
from sklearn.metrics import confusion_matrix
y_true = [0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 1, 0, 1, 0, 1, 0, 1]
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
specificity = tn / (tn+fp)
```



### F1 Score

Precision(정밀도) 와 Recall(재현율)의 가중 조화 평균을 F-Score라고 한다.

가중치를 정밀도에 주어지는 데, 가중치가 1인 경우 F1 Score 라고 한다.

> F = (1 + beta ^ 2) ( Precision X Recall ) / (beta ^2 Precision + Recall)
>
>  F1 = 2 * Precision * Recall / ( Precision + Recall )

```python
from sklearn.metrics import f1_score
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
f1_score(y_true, y_pred, average='macro')
# 0.26...
```



### ROC Curve

ROC(Receiver-Operating Characteristic curve) - 민감도(TPR)와 특이도(FPR)의 그래프.

각 지표를 축으로 하는 그래프로, ROC의 면적(AUC)이 1에 가까울 수록 좋은 성능

```python
import numpy as np
from sklearn import metrics
y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
fpr
# array([0. , 0. , 0.5, 0.5, 1. ])
tpr
# array([0. , 0.5, 0.5, 1. , 1. ])
thresholds
# array([1.8 , 0.8 , 0.4 , 0.35, 0.1 ])
```





### 알아두면 편리한 함수

#### Classification Report

------

Precision, Recall, f1-score를 한번에 계산

```python
# scikitlearn classification_report
from sklearn.metrics import classification_report
y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))
# output
"""
              precision    recall  f1-score   support
<BLANKLINE>
     class 0       0.50      1.00      0.67         1
     class 1       0.00      0.00      0.00         1
     class 2       1.00      0.67      0.80         3
<BLANKLINE>
    accuracy                           0.60         5
   macro avg       0.50      0.56      0.49         5
weighted avg       0.70      0.60      0.61         5
<BLANKLINE>
"""
```



#### ROC AUC Score

------

ROC 그래프의 AUC(Area Under the Curve) 스코어를 계산해주는 함수

```python
import numpy as np
from sklearn.metrics import roc_auc_score
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
roc_auc_score(y_true, y_scores)
# 0.75
```

