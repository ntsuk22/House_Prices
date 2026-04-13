# House Prices - Kaggle Regression Project

---

## პროექტის სათაური
House Prices: Advanced Regression Techniques (Ames Housing)

---

## Kaggle-ის კონკურსის მოკლე მიმოხილვა
ამ კონკურსის მიზანია Ames, Iowa-ს საცხოვრებელი სახლების ფასების პროგნოზირება 79 ცვლადის გამოყენებით. ამოცანა არის რეგრესია.

---

## მიდგომა პრობლემის გადასაჭრელად
პროექტში გამოყენებულია ეტაპობრივი, ექსპერიმენტული მიდგომა:
ჯერ შევქმენი საბაზისო LinearRegression მარტივი preprocessing-ით.
შემდეგ შევცვალე preprocessing სქემა (ყველა სვეტი შევინარჩუნე და 0-ებს და None-ებს შევუსაბამე მნიშვნელობები), რათა მეტი ინფორმაცია შემოსულიყო.
ბოლოს დაემატა რეგულარიზაცია (Ridge) + Pipeline + GridSearchC`.
ასევე ექსპერიმენტები გავაკეთე  DecisionTreeRegressor სხვადასხვა კონფიგურაციით, მაგრამ არასტაბილური გენერალიზაციის გამო საბოლოო მოდელად არ დარჩა.

> **TL;DR:** საბოლოო მოდელი არის Ridge (SECOND TRAIN REGRESSION), რადგან ამ კონფიგურაციამ აჩვენა საუკეთესო სტაბილურობა CV + holdout შეფასებებში.

---

## რეპოზიტორიის სტრუქტურა
```text
House_Prices/
  model_experiment.ipynb ----- ტრენინგი, ექსპერიმენტები, მოდელების შედარებაMLflow logging
  model_inference.ipynb ----- საუკეთესო მოდელის ჩატვირთვა და test პროგნოზები
  submission.csv ----- Kaggle-ისთვის გენერირებული prediction ფაილი
  README.md ----- პროექტის საბოლოო დოკუმენტაცია
```

---

## Feature Engineering
### რეგრესიის ტრენინგები
 სვეტები რომლებშიც None ნიშვნელობები 80% აჭარბებდა გამოვტოვე ხოლო დანარჩენებში შევავსება მედიანით. მიუხედავად იმისა რომ აღნიშNული მოდელის შედეგიც საკმაოდ კარგი იყო მეორე ტრენინგისთვის გადავწყვიტე ყველა მონაცემი დამტოვებინა და შემევსო 0-ებით რადგნა ზოგი მონაცემის წაშლა საბოლოო შედეგზე დიდი გავლენის მოხდენა შეეცლო, მეორე ტრენინგმა უკეთესი შედეგი მიიღო.


### Decision Tree ტრენინგები
Decision Tree ექსპერიმენტებში გამოიყენებოდა:
TotalSF, TotalBath, ასაკობრივი ნიშნები, interaction ტერმები (OverallQual × TotalSF), skewed ფიჩერებზე log1p.
მიუხედავად სწორი მიმართულებისა, out-of-sample შედეგები არასტაბილური დარჩა (ზოგჯერ log მეტრიკა უმჯობესდებოდა, მაგრამ დოლარში holdout უარესდებოდა).

---

## კატეგორიული ცვლადების რიცხვითში გადაყვანა
ძირითადი მიდგომა: `pandas.get_dummies(..., drop_first=False)` (one-hot encoding).
train/test თანხვედრა: `X_test.reindex(columns=X_train.columns, fill_value=0)`.
დამატებით ერთ-ერთ ტრენინგში დაიტესტა ნაწილობრივი frequency encoding (`complete_cols`-ის კატეგორიებზე).

---

## Nan მნიშვნელობების დამუშავება
მედიანა უფრო უსაფრთხოა რადგან რიცხვით მნიშვნელობაში გადაყვანინსას დიდ გავლენას და ცვლილებას არ ახდენს.
0 ძირითადად სარისკო შეიძლება იყოს თუმცა ამ შმეთხვევაში მონაცემი რომელიც None არის შეიძ₾ება დიდ გავლენას ახდენდეს ფასზე ამიტომ მისი განულება რადგნა ცვლილეაბ შესამჩნევი იყოს შედეგიანია.

---

## Cleaning მიდგომები
Id ყოველთვის მოიხსნება feature-ებიდან.

SalePrice target-ად გამოიყოფა.

მონაცემი იყოფა train_test_split(test_size=0.2, random_state=42) სქემით.

Train 1-ში მაღალი null (>80%) სვეტები იშლებოდა (Alley, PoolQC, Fence,MiscFeature).

Train 2-ში ეს შეზღუდვა მოიხსნა და ყველა სვეტი დარჩა, შესაძლებელი იყო პირველი ორის დატოვება.

**რისკები:**
- მაღალი-null სვეტების წაშლა ამ შმეთხვევისთვის არა None-ებისთვის შეიძლება მნიშვნელოვან ინფორმაციას შლიდეს.

---

## Feature Selection
პირდაპირი feature selection (RFE, correlation filter, L1, PCA) პრაქტიკულად არ გამოყენებულა.

Train 1-ში ირიბი selection ხდებოდა მხოლოდ მაღალი null სვეტების წაშლით.

Train 2/3-ში თითქმის ყველა სვეტი რჩებოდა.

ამიტომ OLS-ზე overfitting-ის რისკი მაღალი იყო; Ridge-მა ეს რისკი L2 რეგულარიზაციით ნაწილობრივ დააბალანსა.

---

## გამოყენებული მიდგომები და მათი შეფასება
  Target transform	----- log1p(SalePrice)
  Categorical encoding ----- one-hot (get_dummies)
  Missing values	----- median ან 0/"None"
  Baseline model	----- LinearRegression
  Regularized model	----- Ridge
  Tree model	----- DecisionTreeRegressor

---

## Training
### რეგრესია — 3 ტრენინგი
| ტრენინგი | რა შეიცვალა | უარყოფითი მხარე |
|---|---|---|
| First Train Regression | baseline preprocessing + Linear/Ridge შედარება | OLS-ზე variance მაღალი |
| Second Train Regression (**Best**) | სრული სვეტები + Ridge CV tuning | feature space კვლავ დიდი და Collinearity |
| Third Train Regression | ratio/frequency დამატებები | `best_cv_rmse_log` გაუარესდა (~0.165) |

### Decision Tree — 4 ტრენინგის მოკლე ანალიზი
| ტრენინგი | დაკვირვება | მთავარი პრობლემა |
|---|---|---|
| DT Train 1 | train/CV gap მაღალი | overfitting (variance) |
| DT Train 2 | შიდა log მეტრიკა ნაწილობრივ უკეთესი | holdout $ RMSE მკვეთრი გაუარესება (~$35k) |
| DT Train 3 | log მხარეზე თითქოს პროგრესი | რეალურ დოლარში კვლავ სუსტი გენერალიზაცია |
| DT Train 4 | ძალიან ზედაპირული ხე | underfitting (high bias), holdout ~$43k+ |

> **დასკვნა:** `DecisionTreeRegressor` არც ერთ რანში არ აჩვენებს საკმარისად სტაბილურ out-of-sample ხარისხს, ამიტომ საბოლოო მოდელად არ დარჩა.

---

## ტესტირებული მოდელები
LinearRegression (OLS baseline)
Ridge (Pipeline-ში, რეგულარიზებული ხაზი)
DecisionTreeRegressor (სხვადასხვა depth/leaf/pruning კონფიგურაციით)

---

## Hyperparameter ოპტიმიზაციის მიდგომა
Ridge-ისთვის გამოყენებულია:
Pipeline([("scaler", ...), ("regressor", Ridge(random_state=42))])
GridSearchCV

KFold(n_splits=5, shuffle=True, random_state=42)

scoring="neg_root_mean_squared_error"` (log-space RMSE)

ძებნის სივრცე:
  scaler: StandardScaler, MinMaxScaler, passthrough
  regressor__alpha: [0.01, 0.1, 1, 10, 100, 1000]

Decision Tree-ზე იტესტებოდა depth/split/leaf/pruning დიაპაზონები, მაგრამ საბოლოო ხარისხი მაინც არასაკმარისი დარჩა.

---

## საბოლოო მოდელის შერჩევის დასაბუთება
საბოლოოდ არჩეულია **SECOND TRAIN REGRESSION (Ridge)**, რადგან:
CV შედეგი იყო სტაბილური და ძლიერი (best_cv_rmse_log ~0.1418).
OLS-თან შედარებით უკეთ ამცირებს variance-ს high-dimensional one-hot სივრცეში.
Decision Tree მოდელებთან შედარებით უკეთესი და უფრო პროგნოზირებადი გენერალიზაცია აჩვენა.

---

## MLflow Tracking
ექსპერიმენტები ტრეკდება MLflow-ზე (DagsHub Tracking URI) და ინახება:
მოდელის ტიპი და hyperparameter-ები
CV/train/holdout მეტრიკები
საბოლოო მოდელი შენახული

---

## MLflow ექსპერიმენტების ბმული
Experiment: [https://dagshub.com/ntsuk22/House_Prices.mlflow/#/experiments/3](https://dagshub.com/ntsuk22/House_Prices.mlflow/#/experiments/3)
Final run: [https://dagshub.com/ntsuk22/House_Prices.mlflow/#/experiments/3/runs/1186134cd8764888bc759cd9d115d97b](https://dagshub.com/ntsuk22/House_Prices.mlflow/#/experiments/3/runs/1186134cd8764888bc759cd9d115d97b)

---

## ჩაწერილი მეტრიკების აღწერა
best_cv_rmse_log — 5-fold CV-ის საუკეთესო RMSE ლოგ-სივრცეში (მთავარი გენერალიზაციის ინდიკატორი).
rmse_log_train — train RMSE ლოგ-სივრცეში (fit-ის ხარისხი train-ზე).
rmse_log_holdout — holdout RMSE ლოგ-სივრცეში.
rmse_holdout_dollars — holdout RMSE დოლარში (პრაქტიკული ბიზნეს-ინტერპრეტაცია).
დამატებით შედარებებში გამოყენებული იყო rmse_saleprice_train_dollars და rmse_saleprice_holdout_dollars.

---

## საუკეთესო მოდელის შედეგები
**Final Model:** Ridge  
**Best Params:** regressor__alpha=10.0, scaler=passthrough

### ძირითადი შედეგები (MLflow final run)
| მეტრიკა | მნიშვნელობა |
|---|---:|
| best_cv_rmse_log | **0.141831** |
| rmse_log_train | **0.11016** |
| rmse_log_holdout | **0.135571** |
| rmse_holdout_dollars | **22999.54** |
