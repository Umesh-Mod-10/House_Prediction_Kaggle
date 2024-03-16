# %% Importing the dataset:

from itertools import product
import scipy.stats as ss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xg

# %% Importing the dataset:

train = pd.read_csv("C:/Users/umesh/OneDrive/Desktop/Umesh/Data Analysis/train.csv")
test = pd.read_csv("C:/Users/umesh/OneDrive/Desktop/Umesh/Data Analysis/test.csv")

# %% Getting the basic details:

print("Train Details")

print(train.info())
stats_train = train.describe()
print(train.shape)
print(train.isna().sum()[train.isna().sum() != 0].sort_values(ascending=False))
print(train.duplicated().sum())

# %% Getting the exact columns with nan values:

null = train.isna().sum()
Nan = (null[null != 0].sort_values(ascending=False))
print(Nan)
rows = train.shape[0]
print((Nan / rows) * 100)

# %% Creating a categorical column for Target:

print(pd.cut(x=train['SalePrice'], bins=5))
train['SalePrice_Level'] = pd.qcut(x=train['SalePrice'], q=4, labels=['cheap', 'average', 'expensive', 'luxury'])
print(train['SalePrice_Level'].value_counts())

# %% Creating new columns by the excited:

train['IsFence'] = train['Fence'].apply(lambda x: 0 if pd.isna(x) else 1)
train['IsPool'] = train['PoolQC'].apply(lambda x: 0 if pd.isna(x) else 1)
train['IsFirePlace'] = train['FireplaceQu'].apply(lambda x: 0 if pd.isna(x) else 1)
train['IsShed'] = train['MiscFeature'].apply(lambda x: 0 if pd.isna(x) else 1)
train['MasVnrType'] = train['MasVnrType'].fillna("None")
train['Current_Age'] = (2024 - train['YearBuilt'])
train['IsRenovated'] = train.apply(lambda x: 0 if (x['YearBuilt'] == x['YearRemodAdd']) else 1, axis=1)
train['IsGarage'] = train['GarageType'].apply(lambda x: 0 if pd.isna(x) else 1)
train['IsBasement'] = train['BsmtQual'].apply(lambda x: 0 if pd.isna(x) else 1)
train['IsDeck'] = train['WoodDeckSF'].apply(lambda x: 0 if x == 0 else 1)
train['Total_Bathrooms'] = (train['BsmtFullBath'] + train['BsmtHalfBath']*0.5 + train['FullBath'] + train['HalfBath']*0.5)
train['Is2Floor'] = train['2ndFlrSF'].apply(lambda x: 0 if x == 0 else 1)
train['TotalSA'] = train['BsmtFinSF1'] + train['BsmtFinSF2'] + train['1stFlrSF'] + train['2ndFlrSF']
train['Total_Porch'] = train['OpenPorchSF'] + train['EnclosedPorch'] + train['3SsnPorch'] + train['ScreenPorch']
train['IsScreenPorch'] = train['ScreenPorch'].apply(lambda x: 0 if x == 0 else 1)
train['Is3SsnPorch'] = train['3SsnPorch'].apply(lambda x: 0 if x == 0 else 1)
train['IsEnclosedPorch'] = train['EnclosedPorch'].apply(lambda x: 0 if x == 0 else 1)

# %% Dealing with Nan values:
# Dropping Alley, MasVnrType, PoolQC, Fence, MiscFeature column as it has a lot of Nan values:

train.drop(['Alley', 'MiscFeature', 'Fence', 'PoolQC', 'FireplaceQu'], axis=1, inplace=True)
print(train.isna().sum()[train.isna().sum() != 0].sort_values(ascending=False))

print(len(train.isna().sum()[train.isna().sum() != 0]))

# %%  Getting the Numerical data and Categorical data for dealing Nan values:

categorical_df = train.select_dtypes(include=['object'])
numerical_df = train.select_dtypes(exclude=['object'])

print(categorical_df.info())
print(numerical_df.info())

categorical_null = categorical_df.isna().sum()
categorical_df = categorical_null[categorical_null != 0]
categorical_df = categorical_df.sort_values(ascending=False)
print(categorical_df.sort_values(ascending=False))

numerical_null = numerical_df.isna().sum()
numerical_df = numerical_null[numerical_null != 0]
numerical_df = numerical_df.sort_values(ascending=False)
print(numerical_df.sort_values(ascending=False))

# %% Analysis of the Garage Nan Values:

Garage_Null = (train[train['GarageFinish'].isna()])
print(train['GarageType'].value_counts())

'''Since all the values of the Garage of are Nan as there is no garage for the home
Thus all those values are replaced by having a new value of No'''

index = ['GarageFinish', 'GarageCond', 'GarageQual', 'GarageType']
for i in index:
    train[i] = train[i].fillna("None")

print(numerical_df.sort_values(ascending=False))

categorical_df = train.select_dtypes(include=['object'])
categorical_null = categorical_df.isna().sum()
categorical_df = categorical_null[categorical_null != 0]
categorical_df = categorical_df.sort_values(ascending=False)
print(categorical_df.sort_values(ascending=False))

# %% Analysis of the Basement Nan Values:

Basement_Null = train[train['BsmtCond'].isna()]
print(train['BsmtFinType1'].value_counts())

index_unf = ['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 'BsmtExposure']
for i in index_unf:
    data = train[i].value_counts()
    sn.countplot(x=train[i], color='blue', alpha=0.4, edgecolor='black')
    plt.xlabel("The Categories of the column " + str(i) + " ---->")
    plt.ylabel("The occurrence of each category ---->")
    plt.title("The Category " + str(i))
    plt.grid()
    plt.show()

print(categorical_df.sort_values(ascending=False))
bas_null = train[train['BsmtExposure'].isna()]

index_bas = ['BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']
for i in index_bas:
    train[i] = train[i].fillna("None")

categorical_df = train.select_dtypes(include=['object'])
categorical_null = categorical_df.isna().sum()
categorical_df = categorical_null[categorical_null != 0]
categorical_df = categorical_df.sort_values(ascending=False)

print(categorical_df.sort_values(ascending=False))

# %% Analysis of the Electrical Nan Values:

eletric_null = train[train['Electrical'].isna()]
print(train['Electrical'].value_counts())

train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])

categorical_df = train.select_dtypes(include=['object'])
categorical_null = categorical_df.isna().sum()
categorical_df = categorical_null[categorical_null != 0]
categorical_df = categorical_df.sort_values(ascending=False)

print(categorical_df.sort_values(ascending=False))

# %% Imputing both columns with median and Garage with None:

train['GarageYrBlt'] = train['GarageYrBlt'].fillna(0)

train['LotFrontage'] = train['LotFrontage'].fillna(train.groupby('SalePrice_Level')['LotFrontage'].transform('median'))
train['MasVnrArea'] = train['MasVnrArea'].fillna(train.groupby('SalePrice_Level')['MasVnrArea'].transform('median'))

print(train.isna().sum()[train.isna().sum() != 0])
numerical_null = numerical_df.isna().sum()
numerical_df = numerical_null[numerical_null != 0]
print(numerical_df)

# %% Outlier detection of the dataset:

'''There is outlier in both GrLivArea and TotalBsmtSF'''

sn.scatterplot(data=train, x='SalePrice', y='GrLivArea', color="blue", alpha=0.3, edgecolor="black")
plt.grid(True)
plt.show()

sn.scatterplot(data=train, x='SalePrice', y='TotalBsmtSF', color="red", alpha=0.3, edgecolor="black")
plt.grid(True)
plt.show()

train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index, inplace=True)
train.drop(train[(train['GrLivArea']>4500) & (train['SalePrice']<300000)].index, inplace=True)
train.reset_index(drop=True, inplace=True)

# %% Checking of the correlation:

numeric_train = train.select_dtypes(include=['int64', 'float64'])
corr = numeric_train.corr()
plt.subplots(figsize=(15, 12))
sn.heatmap(corr, cmap='Greens', vmax=0.9, linecolor="black", square=True)
plt.show()

# %% Separating the Discrete and Continuous:

categorical_df = train.select_dtypes(include=['object'])
print(categorical_df.info())

numerical_df = train.select_dtypes(include=['int64'])
print(numerical_df.info())

# %% Getting the unique values from categorical columns:

col = categorical_df.columns
fig, ax = plt.subplots(8, 5, figsize=(20, 25))
ax = ax.flatten()
k = 0

for i in range(0, 9):
    for j in range(0, 4):
        if k < len(col):  # Make sure not to exceed the number of columns
            sn.countplot(data=train, x=col[k], color="blue", alpha=0.5, edgecolor="black", ax=ax[k])
            ax[k].set_title("The counts of " + col[k])
            ax[k].grid(True)
            v = col[k]
            cat = train[v].value_counts()
            for label, counts in cat.items():
                ax[k].axhline(y=counts, color='black', alpha=0.3, linestyle='--', linewidth=1)
            k += 1
        else:
            ax[i, j].axis('off')

plt.tight_layout()
plt.show()

# %% Getting the unique values from categorical columns:

col = categorical_df.columns
for i in col:
    cat = train[i].value_counts()
    print(cat)

# %% Finding the correlation of each categorical column:

print(col)
col1 = col.copy()
col2 = col.copy()
final = list(product(col1, col2, repeat=1))

corr_categorical = []
for i in final:
    if i[0] != i[1]:
        corr_categorical.append((i[0], i[1], list(ss.chi2_contingency(pd.crosstab(
            categorical_df[i[0]], categorical_df[i[1]])))[1]))

corr_categorical_df = pd.DataFrame(corr_categorical, columns=['var1', 'var2', 'coeff'])
corr_categorical = corr_categorical_df.pivot(index='var1', columns='var2', values='coeff')
corr_categorical_df = corr_categorical_df.sort_values(by='coeff')

# %% Visualization of this corr of categorical:

plt.figure(figsize=(8, 8))
sn.heatmap(data=corr_categorical, cmap="Greens", vmax=0.9, linecolor="black", square=True)
plt.show()

# %% Skewness of the target:

print(train['SalePrice'].skew())
print(train['SalePrice'].kurt())
sn.displot(train['SalePrice'], color='lightgreen', edgecolor="black")
plt.grid(False)
plt.show()

# %% Transforming into neutral skewness:

train["SalePrice"] = np.log1p(train["SalePrice"])

# %% Checking the plots:

print(train['SalePrice'].skew())
print(train['SalePrice'].kurt())
sn.displot(train['SalePrice'], edgecolor="black", color="lightgreen")
plt.grid(True)
plt.show()

# %% Test Dataset details:

print("Test Details")

print(test.info())
stats_test = test.describe()
print(test.shape)
print(test.isna().sum()[test.isna().sum() != 0].sort_values(ascending=False))
print(test.duplicated().sum())

# %% Getting the exact columns with nan values:

null = test.isna().sum()
Nan = (null[null != 0].sort_values(ascending=False))
print(Nan)
rows = test.shape[0]
print((Nan / rows) * 100)

# %% Creating new columns by the excited:

test['IsFence'] = test['Fence'].apply(lambda x: 0 if pd.isna(x) else 1)
test['IsPool'] = test['PoolQC'].apply(lambda x: 0 if pd.isna(x) else 1)
test['IsFirePlace'] = test['FireplaceQu'].apply(lambda x: 0 if pd.isna(x) else 1)
test['IsShed'] = test['MiscFeature'].apply(lambda x: 0 if pd.isna(x) else 1)
test['MasVnrType'] = test['MasVnrType'].fillna("None")
test['Current_Age'] = (2024 - test['YearBuilt'])
test['IsRenovated'] = test.apply(lambda x: 0 if (x['YearBuilt'] == x['YearRemodAdd']) else 1, axis=1)
test['IsGarage'] = test['GarageType'].apply(lambda x: 0 if pd.isna(x) else 1)
test['IsBasement'] = test['BsmtQual'].apply(lambda x: 0 if pd.isna(x) else 1)
test['IsDeck'] = test['WoodDeckSF'].apply(lambda x: 0 if x == 0 else 1)
test['Total_Bathrooms'] = (test['BsmtFullBath'] + test['BsmtHalfBath']*0.5 + test['FullBath'] + test['HalfBath']*0.5)
test['Is2Floor'] = test['2ndFlrSF'].apply(lambda x: 0 if x == 0 else 1)
test['TotalSA'] = test['BsmtFinSF1'] + test['BsmtFinSF2'] + test['1stFlrSF'] + test['2ndFlrSF']
test['Total_Porch'] = test['OpenPorchSF'] + test['EnclosedPorch'] + test['3SsnPorch'] + test['ScreenPorch']
test['IsScreenPorch'] = test['ScreenPorch'].apply(lambda x: 0 if x == 0 else 1)
test['Is3SsnPorch'] = test['3SsnPorch'].apply(lambda x: 0 if x == 0 else 1)
test['IsEnclosedPorch'] = test['EnclosedPorch'].apply(lambda x: 0 if x == 0 else 1)

# %% Dealing with Nan values:
# Dropping Alley, MasVnrType, PoolQC, Fence, MiscFeature column as it has a lot of Nan values:

test.drop(['Alley', 'MiscFeature', 'Fence', 'PoolQC', 'FireplaceQu', 'MasVnrType'], axis=1, inplace=True)
print(test.isna().sum()[test.isna().sum() != 0].sort_values(ascending=False))

print(len(test.isna().sum()[test.isna().sum() != 0]))

# %% Getting the Numerical data and Categorical data for dealing Nan values: [Test]

categorical_test = test.select_dtypes(include=['object'])
numerical_test = test.select_dtypes(exclude=['object'])

print(categorical_test.info())
print(numerical_test.info())

print(categorical_test.isna().sum()[categorical_test.isna().sum() != 0].sort_values(ascending=False))
print(numerical_test.isna().sum()[numerical_test.isna().sum() != 0].sort_values(ascending=False))

# %% Analysis of the Garage Nan Values:

Garage_null_t = test[test['GarageType'].isna()]
index = ['GarageFinish', 'GarageCond', 'GarageQual', 'GarageType']
for i in index:
    test[i] = test[i].fillna("None")

categorical_test = test.select_dtypes(include=['object'])
categorical_Nan = categorical_test.isna().sum()
categorical_test = categorical_Nan[categorical_Nan != 0]
categorical_test = categorical_test.sort_values(ascending=False)
print(categorical_test)

# %% Analysis of the Basement Nan Values:

print(test.isna().sum()[test.isna().sum() != 0].sort_values(ascending=False))
index_unf = ['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 'BsmtExposure']

for i in index_bas:
    test[i] = test[i].fillna("None")

categorical_test = test.select_dtypes(include=['object'])
categorical_Nan = categorical_test.isna().sum()
categorical_test = categorical_Nan[categorical_Nan != 0]
categorical_test = categorical_test.sort_values(ascending=False)
print(categorical_test)

# %% Checking the MS Zoning correlation and Imputation:

sample = (test[test['MSZoning'].isna()])
plt.scatter(train['MSZoning'], train['MSSubClass'], color='blue', alpha=0.4, edgecolors='black')
plt.show()

'''There is no relation between MS Zones and MS SubClasses. Thus we replace by the mode values'''

test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])

categorical_test = test.select_dtypes(include=['object'])
print(categorical_test.isna().sum()[categorical_test.isna().sum() != 0].sort_values(ascending=False))

# %% Dealing the Nan values of Utilities:

sample = (test[test['Utilities'].isna()])
print(test['Utilities'].value_counts())
test['Utilities'] = test['Utilities'].fillna(test['Utilities'].mode()[0])

categorical_test = test.select_dtypes(include=['object'])
print(categorical_test.isna().sum()[categorical_test.isna().sum() != 0].sort_values(ascending=False))

# %% Dealing the Nan values of Functional:

sn.scatterplot(data=test, x='Functional', y='Electrical', color='red', edgecolor='black')
plt.show()

mode = lambda x: x.mode().iloc[0] if not x.mode().empty else None
sample = test[test['Functional'].isna()]
print(test['Functional'].value_counts())
test['Functional'] = test['Functional'].fillna(test.groupby('Electrical')['Functional'].transform(mode))

categorical_test = test.select_dtypes(include=['object'])
print(categorical_test.isna().sum()[categorical_test.isna().sum() != 0].sort_values(ascending=False))

# %% Dealing the Nan values of Exterior Values:

sample = test[test['Exterior1st'].isna()]
print(test['Exterior1st'].value_counts())
print(test['Exterior2nd'].value_counts())

test['Exterior1st'] = test['Exterior1st'].fillna(test.groupby('ExterCond')['Exterior1st'].transform(mode))
test['Exterior2nd'] = test['Exterior2nd'].fillna(test.groupby('ExterCond')['Exterior2nd'].transform(mode))

categorical_test = test.select_dtypes(include=['object'])
print(categorical_test.isna().sum()[categorical_test.isna().sum() != 0].sort_values(ascending=False))

# %% Filling the rest with the mode of that column:

test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])
test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])

categorical_test = test.select_dtypes(include=['object'])
print(categorical_test.isna().sum()[categorical_test.isna().sum() != 0].sort_values(ascending=False))

# %% Filling the Garage Null values of the year:

test['GarageYrBlt'] = test['GarageYrBlt'].fillna(0)

numerical_test = test.select_dtypes(['int64', 'float64'])
print(numerical_test.isna().sum()[numerical_test.isna().sum()!=0].sort_values(ascending=False))

# %% Filling the basement null values:

base = ['BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']

sample = (test[test['BsmtHalfBath'].isna()])
for i in base:
    test[i] = test[i].fillna(0)

numerical_test = test.select_dtypes(['int64', 'float64'])
print(numerical_test.isna().sum()[numerical_test.isna().sum()!=0].sort_values(ascending=False))

# %% Filling the LotFrontage Nan values:

'''The LotFrontage value has a correlation with the GrLivArea, thus it can be used to fill the 
Nan values of LotFrontage'''

imputer_median = SimpleImputer(strategy="median")

test['LotFrontage'] = test['LotFrontage'].fillna(test.groupby('GrLivArea')['LotFrontage'].transform('median'))
test['LotFrontage'] = imputer_median.fit_transform(test[['LotFrontage']])

test['MasVnrArea'] = test['MasVnrArea'].fillna(test.groupby('TotalBsmtSF')['MasVnrArea'].transform('median'))
test['MasVnrArea'] = imputer_median.fit_transform(test[['MasVnrArea']])

print(test.isna().sum()[test.isna().sum() != 0].sort_values(ascending=False))

# %% Filling the Garage Values:

sample = test[test['GarageCars'].isna()]
t = test[test['GarageType'] == 'Detchd']
rt = (t.groupby('GarageType')[['GarageCars', 'GarageArea', 'GarageYrBlt']].median())

test['GarageCars'] = test['GarageCars'].fillna(test.groupby('GarageType')['GarageCars'].transform('median'))
test['GarageArea'] = test['GarageArea'].fillna(test.groupby('GarageType')['GarageArea'].transform('median'))

print(test.isna().sum()[test.isna().sum() != 0].sort_values(ascending=False))

# %% Checking the skewness of the numerical variables:

spread = train.select_dtypes(exclude=['object', 'category'])
skewed = []
for i in spread:
    a = (train[i].skew())
    b = (train[i].kurt())
    if (a > 3 or a < -3):
        print(i)
        print("Skewness:", a)
        print("Kurtosis:", b)
        print()
        skewed.append(i)

# %% Plotting these values by box plot:

plt.figure(figsize=(10, 5))
sn.boxplot(data=train[skewed], orient='h', palette='Paired')
plt.show()

# %% Splitting the Train and the test data:

y_train = train[['SalePrice']]
train.drop(columns=['SalePrice', 'SalePrice_Level'], inplace=True, axis=1)
all_features = pd.concat([train, test]).reset_index(drop=True)
train_index = y_train.shape[0]

# %% Scaling the Continuous data:

cont = train.select_dtypes(exclude=['object'])
cont.drop(columns=['Id'], axis=1, inplace=True)

for i in cont.columns:
    all_features[i] = MinMaxScaler().fit_transform(all_features[[i]])

# %% Categorical data Encoding:

cate = all_features.select_dtypes(include=['object'])
label = LabelEncoder()
print(cate.nunique().sort_values(ascending=False))
all_features = pd.get_dummies(all_features)

# %% Splitting the Train and the test data:

#X = train.drop(columns=['SalePrice'], axis=1)
#y = train['SalePrice']


X_train = all_features.iloc[:(train_index), :]
X_test = all_features.iloc[train_index:, :]

#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.20)

# %% Developing a Machine Learning Model:

Rf = RandomForestRegressor(n_jobs=-1, n_estimators=500, random_state=100)
Rf.fit(X_train, y_train)
Y_predict = Rf.predict(X_test)


#rf1 = Rf.score(X_train, y_train)
#print(r2_score(y_test, Y_predict))
#print(rf1)

# %%

xgb_r = xg.XGBRegressor(n_estimators=1000, verbosity=3, n_jobs=-1)
xgb_r.fit(X_train, y_train)
Y_predict = xgb_r.predict(X_test)



#xbr1 = xgb_r.score(X_train, y_train)
#print(r2_score(y_test, Y_predict))
#print(xbr1)

# %%

predictions = np.exp(Y_predict)

# %%

predictions_df = pd.DataFrame({'Id': test['Id'], 'SalePrice': predictions})
predictions_df.to_csv("C:/Users/umesh/OneDrive/Desktop/Umesh/Sub.csv", index_label=True)

