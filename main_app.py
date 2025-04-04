import os, joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from joblib import Memory

# save dir
save_graph = "/Users/jayanshrestha/Downloads/Python_Scripts/MachineLearningAgain/LoanApproval/graph_storage"
model_dir = "/Users/jayanshrestha/Downloads/Python_Scripts/MachineLearningAgain/LoanApproval/model_dir"

# cache memory for fast imputation
memory = Memory(location=os.path.join(model_dir, 'cache_dir'), verbose=1)

# set dataframe display tweaks
pd.set_option('display.max_column', 15)
pd.set_option('display.width', None)
# read csv
df = pd.read_csv('loan_approval_dataset.csv')
print(df.head())

# EDA
print(df.info())

# drop unnecessary column
df.drop(['loan_id'], inplace = True, axis = 1)

print(df.describe())
print(df.describe(include = 'object'))

# remove leading and trailing whitespaces in column names
df.columns = df.columns.str.strip()

# NaN values
print(df.isna().sum())

# unique values
def unique_values(df):
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        print(df[col].unique())

unique_values(df)

# pair plot
# plt.figure(figsize= (20, 20))
# sns.pairplot(data = df, hue ='loan_status', palette= 'rainbow')
# plt.title('Pair Plot')
# plt.savefig(os.path.join(save_graph, 'pairplot.png'))
# plt.close()

# count plot
def count_plot(cols):
    plt.figure(figsize= (10, 10))
    counter = sns.countplot(data = df, x= cols , hue ='loan_status', palette= 'rainbow')
    for containers in counter.containers:
        counter.bar_label(containers, fontsize = 12, rotation = 90)
    plt.title(f'CountPlot of {cols}')
    plt.savefig(os.path.join(save_graph, f'countplot_{cols}.png'))
    plt.close()

# count_plot('education')
# count_plot('self_employed')
# count_plot('loan_term')
# count_plot('no_of_dependents')

#histogram
def hist_plot(cols, color):
    plt.figure(figsize= (10, 10))
    sns.histplot(data = df, x= cols, kde = True, color= color)
    plt.title(f'HistPlot of {cols}')
    plt.savefig(os.path.join(save_graph, f'histplot_{cols}.png'))
    plt.close()

# hist_plot('income_annum', 'lime')
# hist_plot('loan_amount', 'mediumpurple')
# hist_plot('cibil_score', 'lightsalmon')
# hist_plot('residential_assets_value', 'darkgray')
# hist_plot('commercial_assets_value', 'khaki')
# hist_plot('luxury_assets_value', 'deepskyblue')
# hist_plot('bank_asset_value', 'lightcoral')

# box plot
def box_plot(cols, color):
    plt.figure(figsize= (12, 12))
    sns.boxplot(data = df, y= cols, color= color)
    plt.title(f'Box Plot of {cols}')
    plt.savefig(os.path.join(save_graph, f'boxplot_{cols}.png'))
    plt.close()

# box_plot('income_annum', 'lime')
# box_plot('loan_amount', 'mediumpurple')
# box_plot('cibil_score', 'lightsalmon')
# box_plot('residential_assets_value', 'darkgray')
# box_plot('commercial_assets_value', 'khaki')
# box_plot('luxury_assets_value', 'deepskyblue')
# box_plot('bank_asset_value', 'lightcoral')

# Encoding
# for education label
label_encoder = LabelEncoder()
df['education'] = label_encoder.fit_transform(df['education'])
df['education'] = df['education'].map({0:1, 1:0})

# for self_employed_label
df['self_employed'] = label_encoder.fit_transform(df['self_employed'])

# for loan_status
df['loan_status'] = label_encoder.fit_transform(df['loan_status'])
df['loan_status'] = df['loan_status'].map({0:1, 1:0})

# heat map
plt.figure(figsize= (10, 10))
sns.heatmap(data = df.corr(numeric_only= True), cmap= 'rainbow', annot = True)
plt.title('Heatmap')
plt.xticks(rotation = 270)
plt.savefig(os.path.join(save_graph, 'heatmap.png'))
plt.close()

# feature selection
X= df.drop(['loan_status'], axis = 1)
y= df['loan_status']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

# pipeline with scaling and model selection
model = [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(), xgb.XGBClassifier(), SVC()]
def train_evaluate_model(X_train, y_train, X_test, y_test, model, model_name, model_dir, save_graph):
    model_pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])

    if isinstance(model, DecisionTreeClassifier):
        param_grid = {
            'model__max_depth': [10, 20, 30], # depth of the tree, higher depth may lead to overfitting
            'model__min_samples_split': [2, 3, 5],
            'model__min_samples_leaf': [7, 10, 12],
            'model__max_features': [10, 11, 15],
        }
    elif isinstance(model, RandomForestClassifier):
        param_grid ={
            'model__max_depth': [5, 8, 10], # depth of the tree, higher depth may lead to overfitting
            'model__min_samples_split': [10, 12, 15],
            'model__min_samples_leaf': [2 ,3, 4]
        }
    elif isinstance(model, xgb.XGBClassifier):
        param_grid = {
            'model__n_estimators': [10, 15, 20], # number of times gradient boosting will occur, not the number of trees it will grow
            'model__max_depth': [10, 12, 15], # depth of the tree, higher depth may lead to overfitting
            'model__learning_rate': [0.1, 0.3, 0.5], # rate of convergence to local minima, higher rate may overshoot and miss local minima, lower rate may never reach local minima
            'model__subsample': [1], # percent of training data to use
            'model__colsample_bytree': [1] # percent of feature columns to use
        }
    elif isinstance(model, SVC):
        param_grid ={
            'model__kernel': ['rbf'], # Kernel type
            'model__C': [0.1, 1, 3, 5], # Regularization parameter, higher C leads to overfitting
            # 'model__degree': [1, 2, 3], # Degree for poly kernel
            'model__gamma': ['scale', 'auto', 0.1, 1] # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
        }
    else:
        param_grid = {}

    if param_grid:
        @memory.cache
        def cached_grid_search_fit(pipeline, param_grid, X_train, y_train):
            grid_search = GridSearchCV(pipeline, param_grid, n_jobs= -1, cv= 3, scoring= 'f1')
            grid_search.fit(X_train, y_train)
            return grid_search

        grid_search = cached_grid_search_fit(model_pipeline, param_grid, X_train, y_train)
        print(f'Best parameter for {model_name} is: {grid_search.best_params_}\n' )
        best_pipeline = grid_search.best_estimator_


    else:
        model_pipeline.fit(X_train, y_train)
        best_pipeline = model_pipeline


    y_pred = best_pipeline.predict(X_test)
    f1 = f1_score(y_true= y_test, y_pred= y_pred)
    accuracy = accuracy_score(y_true= y_test, y_pred= y_pred)

    print(f"Results for {model_name}:")
    print(f"f1_score: {f1}")
    print(f"accuracy_score: {accuracy}")

    # confusion matrix
    cm = confusion_matrix(y_true= y_test, y_pred= y_pred)
    plt.figure(figsize= (8, 8))
    sns.heatmap(data= cm, annot= True, cmap = "Set2", fmt=".0f")
    plt.title(f'Confusion Matrix of {model_name}')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(save_graph, f'confusionmatrix_{model_name}.png'))
    plt.close()

    joblib.dump(best_pipeline, os.path.join(model_dir, f'{model_name}_model.joblib'))
    print(f"Model saved to {os.path.join(model_dir, f'{model_name}_model.joblib')}\n")
    joblib.dump(f1, os.path.join(model_dir, f'{model_name}_f1.joblib'))
    # return f1


# model calls
train_evaluate_model(X_train, y_train, X_test, y_test, LogisticRegression(), 'Logistic Regression', model_dir, save_graph)
train_evaluate_model(X_train, y_train, X_test, y_test, DecisionTreeClassifier() , 'Decision Tree Classifier', model_dir, save_graph)
train_evaluate_model(X_train, y_train, X_test, y_test, xgb.XGBClassifier(), 'XGB Classifier', model_dir, save_graph)
train_evaluate_model(X_train, y_train, X_test, y_test, RandomForestClassifier(), 'Random Forest Classifier', model_dir, save_graph)
train_evaluate_model(X_train, y_train, X_test, y_test, SVC(), 'Support Vector Classifier', model_dir, save_graph)