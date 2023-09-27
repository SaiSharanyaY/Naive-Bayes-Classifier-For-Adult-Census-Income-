# Import the required libraries
import numpy as np
import pandas as pd
import matplotlib
import matplotlib_inline
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix

# import the dataset into a dataframe 

df = pd.read_csv('https://raw.githubusercontent.com/SaiSharanyaY/Naive-Bayes-Classifier-For-Adult-Census-Income-/main/adult_dataset.csv')

# get basic information on the attributes and datatypes of datasset
print("Describing the dataset : ")
df.info()

# get description of all numerical attributes
print("Describing numerical data")
desc_numerical = df.describe()
print(desc_numerical)

# description of all categorical values
print("Describing categorical data")
desc_categorical = df.describe(include = 'O')
print(desc_categorical)

# dropping redundant data column education num
df.drop(['education.num'], axis = 1)

# check for duplicate rows and if any
duplicates = df.duplicated()
df[duplicates]

# dropping duplicates
df = df.drop_duplicates()

# now check the size of df
print("Shape of dataframe after dropping duplicates")
print(df.shape)

# checking for missing values
df.isnull().sum()

# since the tableau visualization analysis shows ? entries, we will imputate them with mode and with new category.
m_val = ['workclass', 'occupation']
for i in m_val:
    df[i].replace('?', np.nan, inplace=True)
for i in m_val:
    df[i].fillna(df[i].mode()[0], inplace=True)
df['native.country'].replace(np.nan, 'unknown', inplace=True)

# we will check the value counts of income column classes and convert em into binary 1 or 0 format.
df['income'].value_counts()
df['income'].replace({"<=50K":0, ">50K": 1}, inplace=True)

# checking correlation statistically between target variable and the continuous variables using point biserial correlation
a=['age','capital.loss','capital.gain','hours.per.week','fnlwgt']
print("Correlation between continuous features and the target feature")
for i in a:
    print(i)
    print('correlation:',stats.pointbiserialr(df['income'],df[i])[0])


# performing chi-square test - used to determine level of association between categorical variables 
b = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
print("Chi Square test between categorical features and target feature")
for i in b:
    CrosstabResult=pd.crosstab(index=df[i],columns=df['income'])
    # Performing Chi-sq test
    ChiSqResult = chi2_contingency(CrosstabResult)
    print('The P-Value of the ChiSq Test for ',i ,':', ChiSqResult[1])
# the pvalue is very very less and <0.05 so, the correlation is negligible
# here we can observe that all the continuous features are independent itself as there is no high correlation between any including the target variable

# normal distribution graphs for all numerical data

def norm_dist_plot(data):

    mean = np.mean(data)
    std = np.std(data)
    s = np.random.normal(mean, std, len(data))

    count, bins, ignored = plt.hist(s, 100, density=True)

    plt.plot(bins, (1/(std*np.sqrt(2*np.pi)))*np.exp(-(bins-mean)**2/(2*std**2)), linewidth=2, color = 'r')
    plt.xlim([0,max(data)])
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.show()


# code to display the normal distribution plots, to run it just uncomment****************
'''
print("PLOT FOR AGE DISTRIBUTION")
norm_dist_plot(df['age'])
print("PLOT FOR FNLWGT DISTRIBUTION")
norm_dist_plot(df['fnlwgt'])
print("PLOT FOR CAPITAL.GAIN DISTRIBUTION")
norm_dist_plot(df['capital.gain'])
print("PLOT FOR CAPITAL.LOSS DISTRIBUTION")
norm_dist_plot(df['capital.loss'])
print("PLOT FOR HOURS.PER.WEEK DISTRIBUTION")
norm_dist_plot(df['hours.per.week'])

'''
# created a function called z_score which can categorize the column into multiple classes by using gaussian dist formula
# problem statement says to assume gausssian distribution, so according to the gaussian distribution we have data distribution around a wide range of intervals 
# i.e follows range of (mean-3*std_dev, mean+3*std_dev) with a bin width of one std_dev
# z_score 
def zscore(data, bin_width):
    # Calculate mean and standard deviation
    mean = np.mean(data)
    std_dev = np.std(data)
    # Apply Z-score transformation using formula
    z_scores = (data - mean) / std_dev
    bin_edges = np.arange(mean-3*std_dev, mean+3*std_dev + bin_width, bin_width)
    labels = [f'Category {i+1}' for i in range(len(bin_edges)-1)]
    binned_variable = pd.cut(data, bins=bin_edges, labels=labels, include_lowest=True)
    return binned_variable


# creation of equiwidth binning categorization
def bin_ctoc(continuous_variable, num_bins, bin_width):
    # Calculate bin edges
    bin_edges = np.arange(continuous_variable.min(), continuous_variable.max() + bin_width, bin_width)
    # Assign labels to bins
    labels = [f'Category {i+1}' for i in range(len(bin_edges)-1)]
    # Bin the continuous variable
    binned_variable = pd.cut(continuous_variable, bins=bin_edges, labels=labels, include_lowest=True)
    return binned_variable


df['age_cat'] = bin_ctoc(df['age'], num_bins=8, bin_width=10)
df['fnlwgt_cat'] = bin_ctoc(df['fnlwgt'], num_bins=66, bin_width=22501)
df['capital_gain_cat'] = bin_ctoc(df['capital.gain'], num_bins=18, bin_width=5777)
df['capital_loss_cat'] = bin_ctoc(df['capital.loss'], num_bins=16, bin_width=276)
df['hours_per_week_cat'] = bin_ctoc(df['hours.per.week'], num_bins=14, bin_width=7)

# dataframe used for equiwidth binning type
df_new = df[['age_cat','workclass','fnlwgt_cat','education','marital.status','occupation','relationship','race','sex','capital_gain_cat','capital_loss_cat','hours_per_week_cat','native.country','income']]

df['age'] = zscore(df['age'], np.std(df['age']))
df['fnlwgt'] = zscore(df['fnlwgt'], np.std(df['fnlwgt']))
df['capital.gain'] = zscore(df['capital.gain'], np.std(df['capital.gain']))
df['capital.loss'] = zscore(df['capital.loss'], np.std(df['capital.loss']))
df['hours.per.week'] = zscore(df['hours.per.week'], np.std(df['hours.per.week']))


# feature splitting 
X = df_new.drop(['income'], axis = 1)
y = df_new['income']


#Naive Bayes
class NaiveBayes:
    # initializing prior and prosterior dictionaries
    def __init__(self):
        self.priors = {}
        self.posteriors = {}
     # method to fit the training data   
    def fit(self, X, y):
        # Compute priors - prior is the probability of different classes of target variable
        #p(y)
        classes, counts = np.unique(y, return_counts=True)
        total = len(y)
        self.priors = dict(zip(classes, counts / total))
        
        # Compute posteriors - it is teh probability of a class in the considered feature.
        # P(C/X) = PRODUCT OF P(X/C)
        # P(X/C) = P(X1/C)*P(X2/C)*....P(X13/C)
        
        for attribute in X.columns:
            self.posteriors[attribute] = {}
            for value in X[attribute].unique():
                self.posteriors[attribute][value] = {}
                for c in classes:
                    subset = X[y == c][attribute]
                    if len(subset) == 0:
                        self.posteriors[attribute][value][c] = 0
                    else:
                        self.posteriors[attribute][value][c] = len(subset[subset == value]) / counts[c]

    # prediction method to predict the class label of target variable 
    def predict(self, X):
        predictions = []
        for i, row in X.iterrows():
            # here, we initialize probabilities for each class of the target variable
            probs = {c: self.priors[c] for c in self.priors}
            for attribute in X.columns:
                val = row[attribute]
                for c in self.priors:
                    if val in self.posteriors[attribute]:
                    # Multiply by P(feature=value|class)
                        probs[c] *= self.posteriors[attribute][val][c]
                    else:
                        probs[c] = 0
            #predict the class with maximum probability
            predictions.append(max(probs, key=probs.get))
        return predictions

# train and test data splitting manually using split_data function
def split_data(df, target_column, train_ratio=0.7, random_seed=12):
    # Set a random seed for reproducibility, by default its 12
    np.random.seed(random_seed)

    # Shuffle the dataset
    shuffling_data = df.sample(frac=1).reset_index(drop=True)

    # Calculate the number of samples for training and testing
    train_size = int(train_ratio * len(shuffling_data))

    # Split the data into training and testing sets
    train_data = shuffling_data.iloc[:train_size]
    test_data = shuffling_data.iloc[train_size:]

    # Separate features and target variable for the training and testing sets
    X_train = train_data.drop(target_column, axis=1)  # Features for training
    y_train = train_data[target_column]  # Target for training

    X_test = test_data.drop(target_column, axis=1)  # Features for testing
    y_test = test_data[target_column]  # Target for testing

    return X_train, X_test, y_train, y_test


print('**************************************************')

print("Naive Bayes Implementation for the equiwidth binning dataframe")

# Usage
X_train, X_test, y_train, y_test = split_data(df_new, 'income')
print('Train data shape for equiwidth binning data:', X_train.shape, y_train.shape)
print('Test data shape for equiwidth binning data:', X_test.shape, y_test.shape)

#naive bayes fitting and prediction on the equiwidth binning data
nb1 = NaiveBayes()
nb1.fit(X_train, y_train) #data fitting
predictions = nb1.predict(X_test) #data prediction
accuracy1 = np.mean(predictions == y_test) #accuracy calculation using the formula
confusion_matrix1 = confusion_matrix(y_test, predictions)
tn1, fp1, fn1, tp1 = confusion_matrix1.ravel() # ravel is used to flatten muultidimensional array into single array

# Calculate precision, recall, and f1 using the tp, tn, fp, fn values
precision1 = tp1 / (tp1 + fp1) if (tp1 + fp1) != 0 else 0
recall1 = tp1 / (tp1 + fn1) if (tp1 + fn1) != 0 else 0
f1score = 2 * (precision1 * recall1) / (precision1 + recall1) if (precision1 + recall1) != 0 else 0
    
print("Accuracy for equiwidth binning : ", accuracy1)
print("Precision for equiwidth binning",   precision1)
print("Recall for equiwidth binning " ,   recall1)
print("f1 for equiwidth binning",   f1score)


print('**************************************************')

print("Naive Bayes Implementation for the zscore binned dataframe")

X_train1, X_test1, y_train1, y_test1 = split_data(df, 'income')
print('Train data shape for the z_score binning :', X_train1.shape, y_train1.shape)
print('Test data shape for the z_score binning :', X_test1.shape, y_test1.shape)

nb2 = NaiveBayes()
nb2.fit(X_train1, y_train1)
prediction = nb2.predict(X_test1)
accuracy2 = np.mean(prediction == y_test1) 
confusion_matrix2 = confusion_matrix(y_test, prediction)
tn2, fp2, fn2, tp2 = confusion_matrix2.ravel() # ravel is used to flatten muultidimensional array into single array

# Calculate precision, recall, and f1 using the tp, fp, fn, tn values.
precision2 = tp2 / (tp2 + fp2) if (tp2 + fp2) != 0 else 0
recall2 = tp2 / (tp2 + fn2) if (tp2 + fn2) != 0 else 0
f1scoree = 2 * (precision2 * recall2) / (precision2 + recall2) if (precision2 + recall2) != 0 else 0
    
print("Accuracy for z_score binning : ", accuracy2)
print("Precision for z_score binning",   precision2)
print("Recall for z_score binning " ,   recall2)
print("f1 for z_score binning",   f1scoree)



print('***********************************************************************************************************')

# Implemention 10 fold cross validation to improvise the model performance and evaluate again.


class KFoldCrossValidation:
    def __init__(self, df, count_folds=10, random_seed=12):
        np.random.seed(random_seed)
        self.data = df.sample(frac=1).reset_index(drop=True)
        self.count_folds = count_folds

    def kfold(self):
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        #calculating the fold size
        fold_size = len(self.data) // self.count_folds

        for i in range(self.count_folds):
            start = i * fold_size
            end = start + fold_size if i < self.count_folds - 1 else len(self.data)
            val_indices = list(range(start, end))

            train_data = self.data.drop(val_indices)
            val_data = self.data.loc[val_indices]

            X_train = train_data.drop('income', axis=1)
            y_train = train_data['income']

            X_testing = val_data.drop('income', axis=1)
            y_testing = val_data['income']

            # Train your model (NaiveBayes or any other) using X_train, y_train
            nb_classifier = NaiveBayes()
            nb_classifier.fit(X_train, y_train)
            y_pred = nb_classifier.predict(X_testing)


            confusion_matrix_kfold = confusion_matrix(y_testing, y_pred)
            tn, fp, fn, tp = confusion_matrix_kfold.ravel()

            accuracy = np.mean(y_testing == y_pred)
            precision = tp / (tp + fp) if (tp + fp) != 0 else 0
            recall = tp / (tp + fn) if (tp + fn) != 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

            print("--------------------------------------------------------------------------------------")
            print("Accuracy for round ", i, " ", accuracy)
            print("Precision for round ", i, " ", precision)
            print("Recall for round ", i, " ", recall)
            print("f1 for round ", i, " ", f1)

            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        mean_accuracy = np.mean(accuracy_list)
        mean_precision = np.mean(precision_list)
        mean_recall = np.mean(recall_list)
        mean_f1 = np.mean(f1_list)

        print('___________________________________________________________________________________')
        print('Mean Accuracy:', mean_accuracy)
        print('Mean Precision:', mean_precision)
        print('Mean Recall:', mean_recall)
        print('Mean F1 Score:', mean_f1)
        print('___________________________________________________________________________________')


print('**************************************************************************')
print("Performing 10 Fold Cross Validation for equiwidth binning data")
kf_cv1 = KFoldCrossValidation(df_new, count_folds=10, random_seed=12)
kf_cv1.kfold()


print('**************************************************************************')
print("Performing 10 Fold Cross Validation for z_score binning data")
kf_cv2 = KFoldCrossValidation(df, count_folds=10, random_seed=12)
kf_cv2.kfold()



