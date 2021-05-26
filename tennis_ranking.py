#데이터 불러오기
import pandas as pd
tennis = pd.read_csv("./atp_matches_winner.csv")
tennis


#데이터 정제 : column, nan 제거
tennis_NoNan = (tennis.dropna(thresh=12))
tennis_NoNan

true_copy_tennis_NoNan = tennis_NoNan.copy()


#nan 제거 검증
true_copy_tennis_NoNan.isnull().sum()
true_copy_tennis_NoNan.describe()


# describeget_ipython().run_line_magic('matplotlib', 'inline')

#정제된 데이터 히스토그램 보기
import matplotlib.pyplot as plt
true_copy_tennis_NoNan.hist(bins=100, figsize=(80, 60))
save_fig("attribute_histogram_plots")
plt.show()


#Train & Test Set 분리
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(true_copy_tennis_NoNan, test_size=0.2, random_state=42)
test_set.head()


# 서브에이스값으로 Train & Test Set 분리 확인
true_copy_tennis_NoNan["w_ace"].hist()
train_set["w_ace"].hist()
test_set["w_ace"].hist()


# 서브에이스 카테고리 생성
import numpy as np
true_copy_tennis_NoNan["ace_cat"] = pd.cut(true_copy_tennis_NoNan["w_ace"],
                              bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                              labels=[1, 2, 3, 4,5])
true_copy_tennis_NoNan["ace_cat"].value_counts()
true_copy_tennis_NoNan["ace_cat"].hist()


# 승자의 랭킹과 다른 요소들의 상관 관계
corr_matrix = true_copy_tennis_NoNan.corr()
corr_matrix["winner_rank"].sort_values(ascending=False)


# Sactter Matrix 시각화
from pandas.plotting import scatter_matrix
attributes = ["winner_rank", "w_df", "w_bpFaced",
              "w_bpSaved", "w_svpt", "w_1stIn", "w_2ndWon", "w_SvGms", "w_1stWon", "minutes", "w_ace", "winner_age" ]
scatter_matrix(true_copy_tennis_NoNan[attributes], figsize=(50, 30))
save_fig("scatter_matrix_plot")


# 입력(x)과 레이블(y) 분리
true_copy_tennis_NoNan = train_set.drop("winner_rank", axis=1)
true_copy_tennis_NoNan_labels = train_set["winner_rank"].copy()
true_copy_tennis_NoNan_labels


# 데이터 파이프라인 준비
## SimpleImputer와 StandardScaler함수를 사용하여 결과값을 채우고 정규화
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])
true_copy_tennis_NoNan_tr = num_pipeline.fit_transform(true_copy_tennis_NoNan)
true_copy_tennis_NoNan_tr

## ColumnTransformer를 사용하여 최종적으로 tennis_prepared 형태로 데이터 준비하여 파이프라인 준비 완료
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

some_data = true_copy_tennis_NoNan.iloc[:5]
some_labels = true_copy_tennis_NoNan_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("예측:", lin_reg.predict(some_data_prepared))
print("레이블:", list(some_labels))


# Linear Regression 분석 결과
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


# Decision Tree Regressor 분석 결과
tennis_predictions = tree_reg.predict(tennis_prepared)
tree_mse = mean_squared_error(true_copy_tennis_NoNan_labels, tennis_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# Random Forest 분석
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(tennis_prepared, true_copy_tennis_NoNan_labels)


# Random Forest 분석 결과
tennis_predictions = forest_reg.predict(tennis_prepared)
forest_mse = mean_squared_error(true_copy_tennis_NoNan_labels, tennis_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# 교차 검증 준비
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


# Grid Search를 통해 Random Forest Regressor 모델 세부 조정
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
