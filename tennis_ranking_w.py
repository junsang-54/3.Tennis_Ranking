#라이브러리 설정
import sys
import sklearn
import numpy as np
import os

% matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

#변수 및 함수 설정
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("그림 저장:", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")


#csv데이터 불러오기위해 pandas 사용
import pandas as pd

tennis = pd.read_csv("/Users/jinchoi725/Desktop/atp_matches_winner.csv")
tennis


#불러온 csv파일에 column의 중 랭킹 예측에 필요한 12개의 데이터가 없는(nan값을 가진) row가 있어 dropna 함수를 사용하여 빈데이터 제거
tennis_NoNan = (tennis.dropna(thresh=12))
tennis_NoNan

#원본에서 정제된 데이터를 slice
true_copy_tennis_NoNan = tennis_NoNan.copy() 


#빈데이터(nan값)가 있으면 예측이 불가하여 빈데이터가 있는지 isnull함수로 확인하고, column별로 nan값의 갯수를 한눈에 보기 편하게 sum함수 사용
true_copy_tennis_NoNan.isnull().sum()

#describe 함수를 사용하여 데이터의 요약 정보 확인
true_copy_tennis_NoNan.describe() 


#정제된 데이터의 각 column별 데이터분포를 한눈에 보기위해 히스토그램 출력
%matplotlib inline
import matplotlib.pyplot as plt
true_copy_tennis_NoNan.hist(bins=100, figsize=(80, 60))
save_fig("attribute_histogram_plots")
plt.show()


# 모델을 훈련시키기위한 Train Set 과 해당 모델의 훈련 정확도를 확인을 위한 Test Set 분리
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(true_copy_tennis_NoNan, test_size=0.2, random_state=42)
test_set.head()


# 서브에이스값으로 Train & Test Set 분리되었는지 확인
true_copy_tennis_NoNan["w_ace"].hist()
train_set["w_ace"].hist()
test_set["w_ace"].hist()


# 서브에이스 카테고리 생성 ?
true_copy_tennis_NoNan["ace_cat"] = pd.cut(true_copy_tennis_NoNan["w_ace"],
                              bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                              labels=[1, 2, 3, 4,5])
true_copy_tennis_NoNan["ace_cat"].value_counts()

true_copy_tennis_NoNan["ace_cat"].hist()


# 승자의 랭킹과 다른 요소들의 상관 관계를 확인
corr_matrix = true_copy_tennis_NoNan.corr()
corr_matrix["winner_rank"].sort_values(ascending=False)


# 각 요소끼리 상관관계의 Scatter Matrix 시각화
from pandas.plotting import scatter_matrix
attributes = ["winner_rank", "w_df", "w_bpFaced",
              "w_bpSaved", "w_svpt", "w_1stIn", "w_2ndWon", "w_SvGms", "w_1stWon", "minutes", "w_ace", "winner_age" ]
scatter_matrix(true_copy_tennis_NoNan[attributes], figsize=(50, 30))
save_fig("scatter_matrix_plot")


# 모델 수행 전 예측할 column인 winner_rank를 제거
true_copy_tennis_NoNan = train_set.drop("winner_rank", axis=1)
true_copy_tennis_NoNan_labels = train_set["winner_rank"].copy()
true_copy_tennis_NoNan_labels


# 데이터 머신러닝 학습을 위한 파이프라인 준비
## 누락값을 대체 하는 함수인 SimpleImputer와 그 대체할 값을 평균과 표준편차를 활용해 표준화한 값을 만드는 StandardScaler함수를 사용하여 결측값을 채우고 정규화
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])
true_copy_tennis_NoNan_tr = num_pipeline.fit_transform(true_copy_tennis_NoNan)
true_copy_tennis_NoNan_tr

## ColumnTransformer를 사용하여 최종적으로 tennis_prepared 형태로 데이터 준비하여 파이프라인 준비 완료 ?
from sklearn.compose import ColumnTransformer

num_attribs = list(true_copy_tennis_NoNan)
cat_attribs = ["w_ace"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs)
    ])
tennis_prepared = full_pipeline.fit_transform(true_copy_tennis_NoNan)
print(tennis_prepared)
print(tennis_prepared.shape)


# Linear Regression 분석
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(tennis_prepared, true_copy_tennis_NoNan_labels)
# Linear Regression이 잘 학습되었는지 확인을 위해 몇 개의 샘플을 평가
some_data = true_copy_tennis_NoNan.iloc[:5]
some_labels = true_copy_tennis_NoNan_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("예측:", lin_reg.predict(some_data_prepared))
print("레이블:", list(some_labels))

## Linear Regression 분석 결과
from sklearn.metrics import mean_squared_error

tennis_predictions = lin_reg.predict(tennis_prepared)
lin_mse = mean_squared_error(true_copy_tennis_NoNan_labels, tennis_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

true_copy_tennis_NoNan_labels.hist()


# Decision Tree Regressor 분석
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(tennis_prepared, true_copy_tennis_NoNan_labels)

## Decision Tree Regressor 분석 결과
tennis_predictions = tree_reg.predict(tennis_prepared)
tree_mse = mean_squared_error(true_copy_tennis_NoNan_labels, tennis_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# Random Forest 분석
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(tennis_prepared, true_copy_tennis_NoNan_labels)


## Random Forest 분석 결과
tennis_predictions = forest_reg.predict(tennis_prepared)
forest_mse = mean_squared_error(true_copy_tennis_NoNan_labels, tennis_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# 교차 검증을 위해 cross_val_score함수 호출과 점수 표기를 위한 함수 정의
from sklearn.model_selection import cross_val_score
def display_scores(scores):
    print("scores:", scores)
    print("average:", scores.mean())
    print("standard deviation:", scores.std())


# Linear Regression 교차 검증
lin_scores = cross_val_score(lin_reg, tennis_prepared, true_copy_tennis_NoNan_labels,
                            scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# Decision Tree Regressor 교차 검증
tree_scores = cross_val_score(tree_reg, tennis_prepared, true_copy_tennis_NoNan_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
display_scores(tree_rmse_scores)


# Random Forest 교차 검증
forest_scores = cross_val_score(forest_reg, tennis_prepared, true_copy_tennis_NoNan_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# 최적의 모델을 만들기 위해 Grid Search를 통해 Random Forest Regressor 모델 세부 조정
from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]

forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                          scoring='neg_mean_squared_error',
                          return_train_score=True)
grid_search.fit(tennis_prepared, true_copy_tennis_NoNan_labels)

print(grid_search.best_params_)
print(grid_search.best_estimator_)


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
