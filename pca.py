import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import warnings


from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, r2_score, roc_curve
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split


MY_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(MY_DIR, 'data')


warnings.filterwarnings('ignore')


def plt_show_sec(duration: float = 2):
    plt.show(block=False)
    plt.pause(duration)
    plt.close()

def formatting():
    print(f'\n{"*"*100}\n')

def plot_roc_(false_positive_rate, true_positive_rate, roc_auc):
    plt.figure(figsize=(5, 5))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, color='red', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt_show_sec()

def plot_feature_importances(gbm):
    n_features = X_train.shape[1]
    plt.barh(range(n_features), gbm.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X_train.columns)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    plt.ylim(-1, n_features)

data = pd.read_csv(os.path.join(DATA_DIR, 'Cleveland_data.csv'))

data = data.rename(columns={
    'age': 'Age', 'sex': 'Sex', 'cp': 'Cp', 
    'trestbps': 'TrestBPS', 'chol': 'Chol','fbs': 'FBS', 
    'restecg': 'RestECG', 'thalach': 'Thalach', 'exang': 'Exang',
    'oldpeak': 'Oldpeak', 'slope': 'Slope', 'ca': 'Ca',
    'thal': 'Thal', 'target': 'Target'})

print('Data\n', data)
formatting()

# Now, we will check null on all data and if data has null, we will sum of null data's.
# In this way, we find how many missing data is in the data.
print('Data Sum of Null Values\n')
print(data.isnull().sum())
formatting()

plt.figure(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True, fmt='.1f')
plt.tight_layout()
plt_show_sec()

plt.figure(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={'shrink': .5})
plt.tight_layout()
plt_show_sec()

sns.pairplot(data)
plt_show_sec()

sns.barplot(x=data.Age.value_counts()[:10].index, y=data.Age.value_counts()[:10].values)
plt.xlabel('Age')
plt.ylabel('Age Counter')
plt.title('Age Analysis System')
plt_show_sec()

# Firstly, find min and max ages
print('Age related data\n')
minAge = min(data.Age)
maxAge = max(data.Age)
meanAge = data.Age.mean()
print('Min Age :', minAge)
print('Max Age :', maxAge)
print('Mean Age :', meanAge)

young_ages = data[(data.Age >= 29) & (data.Age < 40)]
middle_ages = data[(data.Age >= 40) & (data.Age < 55)]
elderly_ages = data[(data.Age > 55)]
print('\nYoung Ages :', len(young_ages))
print('Middle Ages :', len(middle_ages))
print('Elderly Ages :', len(elderly_ages))
formatting()

sns.barplot(x=['Young ages', 'Middle ages', 'Elderly ages'], 
y=[len(young_ages), len(middle_ages), len(elderly_ages)])
plt.xlabel('Age Range')
plt.ylabel('Age Counts')
plt.title('Age State in Dataset')
plt_show_sec()

data['AgeRange'] = 0
youngAge_index = data[(data.Age >= 29) & (data.Age < 40)].index
middleAge_index = data[(data.Age >= 40) & (data.Age < 55)].index
elderlyAge_index = data[(data.Age > 55)].index
for index in elderlyAge_index:
    data.loc[index, 'AgeRange'] = 2
for index in middleAge_index:
    data.loc[index, 'AgeRange'] = 1
for index in youngAge_index:
    data.loc[index, 'AgeRange'] = 0

# Draw a categorical scatter-plot to show each observation
sns.swarmplot(data=data, hue='Sex', palette=['r', 'c', 'y'], x='AgeRange', y='Age')
plt_show_sec()

# Plot the total crashes
sns.set_color_codes('pastel')
sns.barplot(y='AgeRange', x='Sex', data=data, label='Total', color='b')
plt_show_sec()

sns.countplot(data=elderly_ages, x='Sex', hue='Sex')
plt.title("Elderly Sex Operations")
plt_show_sec()

elderly_ages.groupby(elderly_ages['Sex'])['Thalach'].agg('sum')

sns.barplot(x=elderly_ages.groupby(elderly_ages['Sex'])['Thalach'].agg('sum').index,
            y=elderly_ages.groupby(elderly_ages['Sex'])['Thalach'].agg('sum').values)
plt.title('Gender Group Thalach Show Sum Time')
plt_show_sec()

sns.violinplot(data.Age, palette='Set3', bw=.2, cut=1, linewidth=1)
plt.xticks(rotation=90)
plt.title('Age Rates')
plt_show_sec()

plt.figure(figsize=(15, 7))
sns.violinplot(x=data.Age, y=data.Target)
plt.xticks(rotation=90)
plt.legend()
plt.title('Age & Target System')
plt_show_sec()

colors = ['blue', 'green', 'yellow']
explode = [0, 0, 0.1]
plt.figure(figsize=(5, 5))
plt.pie([len(young_ages), len(middle_ages), len(elderly_ages)],
        labels=['Young ages', 'Middle ages', 'Elderly ages'],
        explode=explode, colors=colors, autopct='%1.1f%%')
plt.title('Age States', color='black', fontsize=15)
plt_show_sec()

# Sex (1 = male; 0 = female)
sns.countplot(data=data, x='Sex', hue='Sex')
plt_show_sec()

sns.countplot(data=data, x='Sex', hue=data.Slope)
plt.title('Slope & Sex Rates Show')
plt_show_sec()

total_genders_count = len(data.Sex)
male_count = len(data[data['Sex'] == 1])
female_count = len(data[data['Sex'] == 0])
print('Gender Categorization\n')
print('Total People :', total_genders_count)
print('Male Count :', male_count)
print('Female Count :', female_count)

# Percentage ratios
print('\nMale Percentage: {:.2f}%'.format((male_count / total_genders_count * 100)))
print('Female Percentage: {:.2f}%'.format((female_count / total_genders_count * 100)))
formatting()

# Male State & target 1 & 0
male_andtarget_on = len(data[(data.Sex == 1) & (data['Target'] == 1)])
male_andtarget_off = len(data[(data.Sex == 1) & (data['Target'] == 0)])
sns.barplot(x=['Male Target On', 'Male Target Off'],
            y=[male_andtarget_on, male_andtarget_off])
plt.xlabel('Male and Target State')
plt.ylabel('Count')
plt.title('State of the Gender')
plt_show_sec()

# Female State & target 1 & 0
female_andtarget_on = len(data[(data.Sex == 0) & (data['Target'] == 1)])
female_andtarget_off = len(data[(data.Sex == 0) & (data['Target'] == 0)])
sns.barplot(x=['Female Target On', 'Female Target Off'],
            y=[female_andtarget_on, female_andtarget_off])
plt.xlabel('Female and Target State')
plt.ylabel('Count')
plt.title('State of the Gender')
plt_show_sec()

# Plot miles per gallon against horsepower with other semantics
sns.relplot(x='TrestBPS', y='Age', sizes=(40, 400), alpha=.5, palette='muted', height=6, data=data)
plt_show_sec()

# As seen, there are 4 types of chest pain.
# 0 status at least
# 1 condition slightly distressed
# 2 condition medium problem
# 3 condition too bad

sns.countplot(data=data, x=data.Cp, hue=data.Cp)
plt.xlabel('Chest Type')
plt.ylabel('Count')
plt.title('Chest Type vs Count State')
plt_show_sec()

cp_one_target_zero = len(data[(data.Cp == 1) & (data.Target == 0)])
cp_one_target_one = len(data[(data.Cp == 1) & (data.Target == 1)])
sns.barplot(x=['cp_one_target_zero', 'cp_one_target_one'],
            y=[cp_one_target_zero, cp_one_target_one])
plt_show_sec()

cp_two_target_zero = len(data[(data.Cp == 2) & (data.Target == 0)])
cp_two_target_one = len(data[(data.Cp == 2) & (data.Target == 1)])
sns.barplot(x=['cp_two_target_zero', 'cp_two_target_one'],
            y=[cp_two_target_zero, cp_two_target_one])
plt_show_sec()

cp_three_target_zero = len(data[(data.Cp == 3) & (data.Target == 0)])
cp_three_target_one = len(data[(data.Cp == 3) & (data.Target == 1)])
sns.barplot(x=['cp_three_target_zero', 'cp_three_target_one'],
            y=[cp_three_target_zero, cp_three_target_one])
plt_show_sec()

# Show the results of a linear regression within each dataset
sns.lmplot(x='TrestBPS', y='Chol', data=data, hue='Cp')
plt_show_sec()

target_0_agerang_0 = len(data[(data.Target == 0) & (data.AgeRange == 0)])
target_1_agerang_0 = len(data[(data.Target == 1) & (data.AgeRange == 0)])
explode = [0, 0.1]
plt.figure(figsize=(5, 5))
plt.pie([target_0_agerang_0, target_1_agerang_0], explode=explode, 
        labels = ['Target 0, Age Range 0','Target 1, Age Range 0'], autopct='%1.1f%%')
plt.title('Target vs Age Range: Young Age ', fontsize=15)
plt_show_sec()

target_0_agerang_1 = len(data[(data.Target == 0) & (data.AgeRange == 1)])
target_1_agerang_1 = len(data[(data.Target == 1) & (data.AgeRange == 1)])
explode = [0, 0.1]
plt.figure(figsize=(5, 5))
plt.pie([target_0_agerang_1, target_1_agerang_1], explode=explode, 
        labels = ['Target 0, Age Range 1', 'Target 1, Age Range 1'], autopct='%1.1f%%')
plt.title('Target vs Age Range: Middle Age', fontsize=15)
plt_show_sec()

target_0_agerang_2 = len(data[(data.Target == 0) & (data.AgeRange == 2)])
target_1_agerang_2 = len(data[(data.Target == 1) & (data.AgeRange == 2)])
explode = [0, 0.1]
plt.figure(figsize=(5, 5))
plt.pie([target_0_agerang_2, target_1_agerang_2], explode=explode, 
        labels = ['Target 0, Age Range 2', 'Target 1, Age Range 2'], autopct='%1.1f%%')
plt.title('Target vs Age Range: Elderly Age ', fontsize=15)
plt_show_sec()

sns.barplot(x=data.Thalach.value_counts()[:20].index,
            y=data.Thalach.value_counts()[:20].values)
plt.xlabel('Thalach')
plt.ylabel('Count')
plt.title('Thalach Counts')
plt.xticks(rotation=45)
plt_show_sec()

age_unique = sorted(data.Age.unique())
age_thalach_values = data.groupby('Age')['Thalach'].count().values
mean_thalach = []
for i, age in enumerate(age_unique):
    mean_thalach.append(sum(data[data['Age'] == age].Thalach)/age_thalach_values[i])
plt.figure(figsize=(10, 5))
sns.pointplot(x=age_unique, y=mean_thalach, color='red', alpha=0.8)
plt.xlabel('Age', fontsize=15)
plt.xticks(rotation=45)
plt.ylabel('Thalach', fontsize=15)
plt.title('Age vs Thalach', fontsize=15)
plt.grid()
plt_show_sec()

age_range_thalach = data.groupby('AgeRange')['Thalach'].mean()
sns.barplot(x=age_range_thalach.index, y=age_range_thalach.values)
plt.xlabel('Age Range Values')
plt.ylabel('Maximum Thalach By Age Range')
plt.title('Illustration of the thalach to the Age Range')
plt_show_sec()

# As shown in this graph, this rate decreases as the heart rate is faster and in old age areas.
cp_thalach = data.groupby('Cp')['Thalach'].mean()
sns.barplot(x=cp_thalach.index, y=cp_thalach.values)
plt.xlabel('Degree of Chest Pain (Cp)')
plt.ylabel('Maximum Thalach By Cp Values')
plt.title('Illustration of Thalach to degree of chest pain')
plt_show_sec()

# As seen in this graph, the heart rate is less when the chest pain is low. But in cases where chest
# pain is 1, it is observed that the area is more. 2 and 3 were found to be of the same degree.
data.Thal.value_counts()
sns.countplot(data=data, x=data.Thal, hue=data.Thal)
plt_show_sec()

# Target 1
target_one_thal_zero = len(data[(data['Target'] == 1) & (data['Thal'] == 0)])
target_one_thal_one = len(data[(data['Target'] == 1) & (data['Thal'] == 1)])
target_one_thal_two = len(data[(data['Target'] == 1) & (data['Thal'] == 2)])
target_one_thal_three = len(data[(data['Target'] == 1) & (data['Thal'] == 3)])
print('Target vs Thal\n')
print('Target 1 Thal 0: ', target_one_thal_zero)
print('Target 1 Thal 1: ', target_one_thal_one)
print('Target 1 Thal 2: ', target_one_thal_two)
print('Target 1 Thal 3: ', target_one_thal_three)

# Target 0
target_zero_thal_zero = len(data[(data['Target'] == 0) & (data['Thal'] == 0)])
target_zero_thal_one = len(data[(data['Target'] == 0) & (data['Thal'] == 1)])
target_zero_thal_two = len(data[(data['Target'] == 0) & (data['Thal'] == 2)])
target_zero_thal_three = len(data[(data['Target'] == 0) & (data['Thal'] == 3)])
print('\nTarget 0 Thal 0: ', target_zero_thal_zero)
print('Target 0 Thal 1: ', target_zero_thal_one)
print('Target 0 Thal 2: ', target_zero_thal_two)
print('Target 0 Thal 3: ', target_zero_thal_three)
formatting()

f, ax = plt.subplots(figsize=(7, 7))
sns.barplot(y=['T 1 & 0, Th 0', 'T 1 & 0, Th 1', 'T 1 & 0, Th 2', 'T 1 & 0, Th 3'], 
            x=[1, 6, 130, 28], color='green', alpha=0.5,
            label='Target 1 Thal State')
sns.barplot(y=['T 1 & 0, Th 0', 'T 1 & 0, Th 1', 'T 1 & 0, Th 2', 'T 1 & 0, Th 3'],
            x=[1, 12, 36, 89], color='red', alpha=0.7,
            label='Target 0 Thal State')
ax.legend(loc='lower right', frameon=True)
ax.set(xlabel='Target State and Thal Counter',
       ylabel='Target State and Thal State',
       title='Target VS Thal')
plt.xticks(rotation=90)
plt_show_sec()

sns.countplot(data=data, x=data.Target, hue=data.Target)
plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Target Counter 1 & 0')
plt_show_sec()

sns.countplot(data=data, x=data.Target, hue=data.Sex)
plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Target & Sex Counter 1 & 0')
plt_show_sec()

# Determine the Age Ranges of patients with and without sickness and analyze the values
age_counter_target_1 = []
age_counter_target_0 = []
for age in data.Age.unique():
    age_counter_target_1.append(len(data[(data['Age'] == age) & (data.Target == 1)]))
    age_counter_target_0.append(len(data[(data['Age'] == age) & (data.Target == 0)]))

plt.scatter(x=data.Age.unique(),
            y=age_counter_target_1, color='blue',
            label='Target 1')
plt.scatter(x=data.Age.unique(),
            y=age_counter_target_0,
            color='red', 
label='Target 0')
plt.legend(loc='upper right', frameon=True)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Target 0 & Target 1 State')
plt_show_sec()

sns.lineplot(x='Sex', y='Oldpeak', hue='Target', data=data)
plt_show_sec()

sns.catplot(x='AgeRange', y='Chol', hue='Sex', data=data, kind='bar')
plt_show_sec()

sns.barplot(x='Sex', y='Chol', data=data, linewidth=2.5, facecolor=(1, 1, 1, 0), errcolor='.2', edgecolor='.2')
plt_show_sec()

male_young_t_1 = data[(data['Sex'] == 1) & (data['AgeRange'] == 0) & (data['Target'] == 1)]
male_middle_t_1 = data[(data['Sex'] == 1) & (data['AgeRange'] == 1) & (data['Target'] == 1)]
male_elderly_t_1 = data[(data['Sex'] == 1) & (data['AgeRange'] == 2) & (data['Target'] == 1)]
f, ax1 = plt.subplots(figsize=(20, 10))
sns.pointplot(x=np.arange(len(male_young_t_1)), y=male_young_t_1.TrestBPS, color='lime', alpha=0.8, label='Young')
sns.pointplot(x=np.arange(len(male_middle_t_1)), y=male_middle_t_1.TrestBPS, color='black', alpha=0.8, label='Middle')
sns.pointplot(x=np.arange(len(male_elderly_t_1)), y=male_elderly_t_1.TrestBPS, color='red', alpha=0.8, label='Elderly')
plt.xlabel('Range', fontsize=15)
plt.xticks(rotation=90)
plt.legend(loc='upper right', frameon=True)
plt.ylabel('TrestBPS', fontsize=15)
plt.title('Age Range Values vs TrestBPS', fontsize=20)
plt.grid()
plt_show_sec()

data_filter_mean = data[(data['Target'] == 1) & (data['Age'] > 50)].groupby('Sex')[['TrestBPS', 'Chol', 'Thalach']].mean()
data_filter_mean.unstack()
print('Mean of symptoms for Target 1\n\n', data_filter_mean)
formatting()

for i, col in enumerate(data.columns.values):
    plt.subplot(5, 3, i+1)
    plt.scatter([i for i in range(303)], data[col].values.tolist())
    plt.title(col)
    fig, ax = plt.gcf(), plt.gca()
    fig.set_size_inches(10, 10)
    plt.tight_layout()
plt_show_sec()

# Let's see the correlation values between the features
print('Correlation Matrix\n\n', data.corr())
formatting()

dataX = data.drop('Target', axis=1)
dataY = data['Target']
X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=42)
print('Train and test subsets\n')
print('X_train:', X_train.shape)
print('X_test:', X_test.shape)
print('y_train:', y_train.shape)
print('y_test:', y_test.shape)
formatting()

# Normalization as the first process
X_train = (X_train-np.min(X_train))/(np.max(X_train)-np.min(X_train))
X_test = (X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test))


"""Gradient Boosting Classifier"""

clff = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2, max_depth=1, random_state=23)
clff.fit(X_train, y_train)
y_pred=clff.predict(X_test)
print('Gradient Boosting Classifier:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
formatting()


"""Random Forest Classifier"""

clf=RandomForestClassifier(n_estimators=124,min_samples_split= 2,
                           min_samples_leaf= 1,max_features='sqrt',max_depth=None, bootstrap=False)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print('Random Forest Classifier:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
formatting()


"""PCA"""

pca = PCA().fit(X_train)
print('PCA Explained Variance Ratio:\n', pca.explained_variance_ratio_)
print('\nX-Train subsets Columns:\n', X_train.columns.values.tolist())
print('\nPCA Components:\n', pca.components_)
formatting()

cumulative = np.cumsum(pca.explained_variance_ratio_)
plt.step([i for i in range(len(cumulative))], cumulative)
plt_show_sec()

# PCA scatter plot for X Train subset
pca = PCA(n_components=8)
pca.fit(X_train)
reduced_data_train = pca.transform(X_train)
plt.scatter(reduced_data_train[:, 0], reduced_data_train[:, 1], label='reduced')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt_show_sec()

# PCA scatter plot for X Test subset
pca = PCA(n_components=8)
pca.fit(X_test)
reduced_data_test = pca.transform(X_test)
plt.scatter(reduced_data_test[:, 0], reduced_data_test[:, 1], label='reduced')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt_show_sec()

reduced_data_train = pd.DataFrame(reduced_data_train,
                                  columns=['Dim1', 'Dim2', 'Dim3', 'Dim4', 'Dim5', 'Dim6', 'Dim7', 'Dim8'])
reduced_data_test = pd.DataFrame(reduced_data_test,
                                 columns=['Dim1', 'Dim2', 'Dim3', 'Dim4', 'Dim5', 'Dim6', 'Dim7', 'Dim8'])
X_train = reduced_data_train
X_test = reduced_data_test

combine_features_list = [
    ('Dim1', 'Dim2', 'Dim3'),
    ('Dim4', 'Dim5', 'Dim5', 'Dim6'),
    ('Dim7', 'Dim8', 'Dim1'),
    ('Dim4', 'Dim8', 'Dim5')
]

parameters = [
    {
        'penalty': ['l1', 'l2'], 'C': [0.1, 0.4, 0.5], 'random_state': [0]
        },
]

for features in combine_features_list:
    X_train_set = X_train.loc[:, features]
    X_test_set = X_test.loc[:, features]
    gslog = GridSearchCV(LogisticRegression(), parameters, scoring='accuracy')
    gslog.fit(X_train_set, y_train)
    print('Best parameters set:')
    print(gslog.best_params_)

    predictions = [
    (gslog.predict(X_train_set), y_train, 'Train'),
    (gslog.predict(X_test_set), y_test, 'Test'),
    ]
    for pred in predictions:
        print('\n' + pred[2] + ' Classification Report:')
        print(classification_report(pred[1], pred[0]))
        print('\n' + pred[2] + ' Confusion Matrix:')
        print(confusion_matrix(pred[1], pred[0]))
    basari = cross_val_score(estimator=LogisticRegression(), X=X_train, y=y_train, cv=12)
    print('\nMean: ', basari.mean())
    print('Standard Deviation: ', basari.std())
    formatting()


"""Logistic Regression to classify the observations for different clinical features resulting in heart problems."""

lr=LogisticRegression(C=0.1,penalty='l1',random_state=0, solver='liblinear')
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba[:,1], pos_label='')
roc_auc = auc(false_positive_rate, true_positive_rate)
plot_roc_(false_positive_rate,true_positive_rate,roc_auc)

print('Hata Oranı:', r2_score(y_test,y_pred))
print('\nAccurancy Oranı:', accuracy_score(y_test, y_pred))
print('\nLogistic TRAIN score with', format(lr.score(X_train, y_train)))
print('\nLogistic TEST score with', format(lr.score(X_test, y_test)))

cm = confusion_matrix(y_test,y_pred)
print('\nConfusion Matrix:\n', cm)
print('\nCoEf Matrix:')
print(lr.coef_)
print('\nIntercept:')
print(lr.intercept_)
print('\nProba:')
print(lr.predict_log_proba)
formatting()

sns.heatmap(cm, annot=True)
plt_show_sec()