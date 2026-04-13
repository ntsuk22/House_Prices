# House Prices - Kaggle Regression Project

---

## პროექტის სათაური
House Prices: Advanced Regression Techniques (Ames Housing)

---

## Kaggle-ის კონკურსის მოკლე მიმოხილვა
ამ კონკურსის მიზანია Ames, Iowa-ს საცხოვრებელი სახლების `SalePrice`-ის პროგნოზირება 79 განმარტებითი ცვლადის გამოყენებით.  
ამოცანა არის რეგრესია, სადაც შეფასების მთავარი იდეაა რაც შეიძლება ზუსტად დავიჭიროთ ფასის განაწილება როგორც ლოგ-სივრცეში, ისე რეალურ დოლარში.

---

## თქვენი მიდგომა პრობლემის გადასაჭრელად
პროექტში გამოყენებულია ეტაპობრივი, ექსპერიმენტული მიდგომა:
- ჯერ აშენდა საბაზისო ხაზოვანი მოდელი (`LinearRegression`) მარტივი preprocessing-ით.
- შემდეგ შეიცვალა preprocessing სქემა (ყველა სვეტის შენარჩუნება, განსხვავებული იმპუტაცია), რათა მეტი სიგნალი შემოსულიყო.
- ბოლოს დაემატა რეგულარიზაცია (`Ridge`) + `Pipeline` + `GridSearchCV`, რაც გახდა საბოლოო არჩევანი.
- პარალელურად ტესტირდა `DecisionTreeRegressor` სხვადასხვა კონფიგურაციით, მაგრამ არასტაბილური გენერალიზაციის გამო საბოლოო მოდელად არ დარჩა.

> **TL;DR:** საბოლოო მოდელი არის `Ridge` (`SECOND TRAIN REGRESSION`), რადგან ამ კონფიგურაციამ აჩვენა საუკეთესო სტაბილურობა CV + holdout შეფასებებში.

---

## რეპოზიტორიის სტრუქტურა
```text
House_Prices/
├── model_experiment.ipynb   # ტრენინგი, ექსპერიმენტები, მოდელების შედარება, MLflow logging
├── model_inference.ipynb    # საუკეთესო მოდელის ჩატვირთვა და test პროგნოზები
├── submission.csv           # Kaggle-ისთვის გენერირებული prediction ფაილი
├── README.md                # პროექტის საბოლოო დოკუმენტაცია
└── README_example.md        # სტრუქტურული რეფერენსი (სხვა პროექტიდან)
```

---

## ყველა ფაილის განმარტება
- `model_experiment.ipynb` — მონაცემთა მომზადება, feature processing, მოდელების ტრენინგი (Linear/Ridge/Decision Tree), მეტრიკების შედარება და MLflow tracking.
- `model_inference.ipynb` — შენახული საბოლოო მოდელის გამოყენებით test set-ზე პროგნოზირება და `submission.csv` გენერაცია.
- `submission.csv` — `Id, SalePrice` ფორმატის საბოლოო პასუხები Kaggle-ის შესაფასებლად.
- `README.md` — კონკურსის, მიდგომის, ექსპერიმენტების და საბოლოო არჩევანის სრულად აღწერა.
- `README_example.md` — მაგალითის სტრუქტურა, რომელიც გამოყენებულია დოკუმენტაციის ერთგვაროვანი ჩარჩოსთვის.

---

## Feature Engineering
### რეგრესიის ტრენინგების ევოლუცია
| ტრენინგი | მთავარი ცვლილება | ეფექტი | მინუსი |
|---|---|---|---|
| Train 1 | თითქმის მხოლოდ ორიგინალური ფიჩერები + `log1p` target | მარტივი baseline | ნაკლები დომენური სიგნალი |
| Train 2 (**Final**) | სრული სვეტები + Ridge-ready preprocessing | საუკეთესო სტაბილურობა | high-dimensional სივრცე |
| Train 3 | ratio + frequency encoding (ნაწილობრივი) | დამატებითი ექსპერიმენტი | CV გაუარესდა, noise გაიზარდა |

### Decision Tree მხარე
Decision Tree ექსპერიმენტებში გამოიყენებოდა:
- `TotalSF`, `TotalBath`, ასაკობრივი ნიშნები, interaction ტერმები (`OverallQual × TotalSF`), skewed ფიჩერებზე `log1p`.
- მიუხედავად სწორი საინჟინრო მიმართულებისა, out-of-sample შედეგები არასტაბილური დარჩა (ზოგჯერ log მეტრიკა უმჯობესდებოდა, მაგრამ დოლარში holdout უარესდებოდა).

---

## კატეგორიული ცვლადების რიცხვითში გადაყვანა
- ძირითადი მიდგომა: `pandas.get_dummies(..., drop_first=False)` (one-hot encoding).
- train/test თანხვედრა: `X_test.reindex(columns=X_train.columns, fill_value=0)`.
- დამატებით ერთ-ერთ ტრენინგში დაიტესტა ნაწილობრივი frequency encoding (`complete_cols`-ის კატეგორიებზე).

**შეფასება:**
- პლუსი: ლინეარულ მოდელებთან სრულად თავსებადია; უცნობი კატეგორიები test-ში კორექტულად ნულდება.
- მინუსი: ფართო sparse მატრიცა და კოლინეარული dummy ჯგუფები ზრდის OLS-ის variance-ს.

---

## Nan მნიშვნელობების დამუშავება
რეგრესიის სხვადასხვა ტრენინგში ორი სქემა გამოიყენებოდა:

| სქემა | რიცხვითი სვეტები | კატეგორიული სვეტები | შენიშვნა |
|---|---|---|---|
| Train 1 | train მედიანა | `"None"` | outlier-მდგრადი |
| Train 2/3 | `0` | `"None"` | მარტივი და სწრაფი, მაგრამ სემანტიკური რისკით |

**შეფასება:**
- მედიანა უფრო robust-ია outlier-ებთან.
- `0` იმპუტაცია მარტივია, მაგრამ ზოგ სვეტში შეიძლება შინაარსობრივ მნიშვნელობას დაემთხვეს და სიგნალი აურიოს.

---

## Cleaning მიდგომები
- `Id` ყოველთვის მოიხსნება feature-ებიდან.
- `SalePrice` target-ად გამოიყოფა.
- მონაცემი იყოფა `train_test_split(test_size=0.2, random_state=42)` სქემით.
- Train 1-ში მაღალი null (>80%) სვეტები იშლებოდა (`Alley`, `PoolQC`, `Fence`, `MiscFeature`).
- Train 2-ში ეს შეზღუდვა მოიხსნა და ყველა სვეტი დარჩა.

**რისკები:**
- ერთი holdout split შედეგს მგრძნობიარეს ხდის კონკრეტულ გაყოფაზე.
- მაღალი-null სვეტების წაშლა ზოგჯერ კარგ სიგნალსაც შლის; არწაშლა კი noise-ის ზრდას იწვევს.

---

## Feature Selection
პირდაპირი feature selection (`RFE`, correlation filter, `L1`, `PCA`) პრაქტიკულად არ გამოყენებულა.

- Train 1-ში ირიბი selection ხდებოდა მხოლოდ მაღალი null სვეტების წაშლით.
- Train 2/3-ში თითქმის ყველა სვეტი რჩებოდა.
- ამიტომ OLS-ზე overfitting-ის რისკი მაღალი იყო; Ridge-მა ეს რისკი L2 რეგულარიზაციით ნაწილობრივ დააბალანსა.

---

## გამოყენებული მიდგომები და მათი შეფასება
| მიმართულება | გამოყენებული მიდგომა | ძლიერი მხარე | სუსტი მხარე / რისკი |
|---|---|---|---|
| Target transform | `log1p(SalePrice)` | ამცირებს ასიმეტრიას, უკეთ მუშაობს regression-ზე | დოლარში ინტერპრეტაციას უკუტრანსფორმაცია სჭირდება |
| Categorical encoding | one-hot (`get_dummies`) | მარტივი და სტაბილური ლინეარულ მოდელზე | მაღალი განზომილება, მულტიკოლინეარობა |
| Missing values | median ან `0`/`"None"` | სწრაფი და reproducible | ზოგ სვეტში სემანტიკური დანაკარგი |
| Baseline model | `LinearRegression` | სწრაფი benchmark | რეგულარიზაცია არ აქვს, variance მაღალია |
| Regularized model | `Ridge` | კოლინეარულობაზე უკეთესი სტაბილურობა | alpha-ზე სენსიტიურია |
| Tree model | `DecisionTreeRegressor` | ინტერპრეტირებადი გაყოფები | მაღალი variance / ზოგჯერ underfitting |

---

## Training
### რეგრესია — 3 ტრენინგი
| ტრენინგი | რა შეიცვალა | უარყოფითი მხარე |
|---|---|---|
| First Train Regression | baseline preprocessing + Linear/Ridge შედარება | OLS-ზე variance მაღალი |
| Second Train Regression (**Best**) | სრული სვეტები + Ridge CV tuning | feature space კვლავ დიდი და კოლინეარულია |
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
- `LinearRegression` (OLS baseline)
- `Ridge` (`Pipeline`-ში, რეგულარიზებული ხაზი)
- `DecisionTreeRegressor` (სხვადასხვა depth/leaf/pruning კონფიგურაციით)

---

## Hyperparameter ოპტიმიზაციის მიდგომა
`Ridge`-ისთვის გამოყენებულია:
- `Pipeline([("scaler", ...), ("regressor", Ridge(random_state=42))])`
- `GridSearchCV`
- `KFold(n_splits=5, shuffle=True, random_state=42)`
- `scoring="neg_root_mean_squared_error"` (log-space RMSE)
- ძებნის სივრცე:
  - `scaler`: `StandardScaler`, `MinMaxScaler`, `passthrough`
  - `regressor__alpha`: `[0.01, 0.1, 1, 10, 100, 1000]`

Decision Tree-ზე ტესტირდებოდა depth/split/leaf/pruning დიაპაზონები, მაგრამ საბოლოო ხარისხი მაინც არასაკმარისი დარჩა.

---

## საბოლოო მოდელის შერჩევის დასაბუთება
საბოლოოდ არჩეულია **SECOND TRAIN REGRESSION (Ridge)**, რადგან:
- CV შედეგი იყო სტაბილური და ძლიერი (`best_cv_rmse_log` ~0.1418).
- OLS-თან შედარებით უკეთ ამცირებს variance-ს high-dimensional one-hot სივრცეში.
- Decision Tree მოდელებთან შედარებით უკეთესი და უფრო პროგნოზირებადი გენერალიზაცია აჩვენა.
- მოდელი რეალურად შენახულია და გამოყენებულია inference workflow-ში.

---

## MLflow Tracking
ექსპერიმენტები ტრეკდება MLflow-ზე (DagsHub Tracking URI) და ინახება:
- მოდელის ტიპი და hyperparameter-ები
- CV/train/holdout მეტრიკები
- საბოლოო არტეფაქტები (შენახული მოდელი)

---

## MLflow ექსპერიმენტების ბმული
- Experiment: [https://dagshub.com/ntsuk22/House_Prices.mlflow/#/experiments/3](https://dagshub.com/ntsuk22/House_Prices.mlflow/#/experiments/3)
- Final run: [https://dagshub.com/ntsuk22/House_Prices.mlflow/#/experiments/3/runs/1186134cd8764888bc759cd9d115d97b](https://dagshub.com/ntsuk22/House_Prices.mlflow/#/experiments/3/runs/1186134cd8764888bc759cd9d115d97b)

---

## ჩაწერილი მეტრიკების აღწერა
- `best_cv_rmse_log` — 5-fold CV-ის საუკეთესო RMSE ლოგ-სივრცეში (მთავარი გენერალიზაციის ინდიკატორი).
- `rmse_log_train` — train RMSE ლოგ-სივრცეში (fit-ის ხარისხი train-ზე).
- `rmse_log_holdout` — holdout RMSE ლოგ-სივრცეში.
- `rmse_holdout_dollars` — holdout RMSE დოლარში (პრაქტიკული ბიზნეს-ინტერპრეტაცია).
- დამატებით შედარებებში გამოყენებული იყო `rmse_saleprice_train_dollars` და `rmse_saleprice_holdout_dollars`.

---

## საუკეთესო მოდელის შედეგები
**Final Model:** `Ridge`  
**Best Params:** `regressor__alpha=10.0`, `scaler=passthrough`

### ძირითადი შედეგები (MLflow final run)
| მეტრიკა | მნიშვნელობა |
|---|---:|
| `best_cv_rmse_log` | **0.141831** |
| `rmse_log_train` | **0.11016** |
| `rmse_log_holdout` | **0.135571** |
| `rmse_holdout_dollars` | **22999.54** |

დამატებითი შედარებითი ჩანაწერებითაც (`~22.9k–23.0k` holdout $ დიაპაზონი) ეს კონფიგურაცია დარჩა ყველაზე სტაბილურ და პრაქტიკულ არჩევანად.