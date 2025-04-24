# Parkinson-Disease-Classification
This project aims to accurately classify Parkinson’s Disease by leveraging both classical ensemble learning and cutting-edge quantum machine learning techniques. Traditional ensemble methods like Random Forest and Gradient Boosting were employed to capture complex patterns in biomedical voice measurements. Additionally, a Quantum Neural Network (QNN) model was developed using parameterized quantum circuits to explore quantum advantages in handling high-dimensional data. The hybrid approach improves classification accuracy and robustness, offering a promising step towards early and reliable diagnosis of Parkinson’s Disease.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_curve, roc_auc_score


import seaborn as sns
import matplotlib.pyplot as plt 


# For Machine LEarning Pipeline
from sklearn.pipeline import make_pipeline 

# For Standardising The Data
from sklearn.preprocessing import StandardScaler 

# Different Machine Learning Models    
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC # Support vector machine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier,RidgeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,HistGradientBoostingClassifier,BaggingClassifier
from sklearn.ensemble import  AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score



from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
