# importing libraries
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, Binarizer

warnings.filterwarnings("ignore")

# Load the dataset
path = r'C:\Users\DELL\Desktop\Crop_recommendation.csv'
# Load the CSV file data into
data = pd.read_csv(path)
data.head()
print(data.to_string())
data.shape
data.info()
print ('columns',data.columns)
data.describe()
data.isnull().sum()
# Checking for duplicates
print(data.duplicated().sum())


#                 Spot-Check Normalized Models
def NormalizedModel(nameOfScaler):
    if nameOfScaler == 'standard':
        scaler = StandardScaler()
    elif nameOfScaler == 'minmax':
        scaler = MinMaxScaler()
    elif nameOfScaler == 'normalizer':
        scaler = Normalizer()
    elif nameOfScaler == 'binarizer':
        scaler = Binarizer()

        pipelines = []
        pipelines.append((nameOfScaler + 'RF', Pipeline([('Scaler', scaler), ('RF', RandomForestClassifier())])))

        return pipelines

#               Train Model

def fit_model(X_train, y_train, models):
    # Test options and evaluation metric
    num_folds = 10
    scoring = 'accuracy'

    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=0)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    return names, results


#                          Save Trained Model

def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))


#                          Performance Measure

def classification_metrics(model, conf_matrix):
    print(f"Training Accuracy Score: {model.score(X_train, y_train) * 100:.1f}%")
    print(f"Validation Accuracy Score: {model.score(X_test, y_test) * 100:.1f}%")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap='YlGnBu', fmt='g')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title('Confusion Matrix', fontsize=20, y=1.1)
    plt.ylabel('Actual label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.show()
    print(classification_report(y_test, y_pred))

# separating features and target label
X = data.drop('label' ,axis =1)
print(X.head())

# Put all the output into labels array
X = data.drop('label', axis=1)
y = data['label']

# Split the data into Training and Test sets
labels = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                    shuffle=True, random_state=42)

# Pass the training set into the
#  RandomForestClassifier model from Sklearn
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
prediction = rf.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
print("Random Forest Accuracy score:", accuracy)

#      Train model

pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())
model = pipeline.fit(X_train, y_train)
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_metrics(pipeline, conf_matrix)

# save model
save_model(model, 'RandomForest.pkl')
