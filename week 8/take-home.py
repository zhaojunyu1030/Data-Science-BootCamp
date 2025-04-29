# Logistic Regression:

# 1. Try different thresholds for computing predictions. By default it is 0.5. Use predict_proba function to compute probabilities and then try custom thresholds and see their impact on Accuracy, Precision and Recall
Threshold: 0.3
Confusion Matrix:
 [[149  14]
 [ 15  36]]
Accuracy: 0.8644859813084113
Precision: 0.72
Recall: 0.7058823529411765

Threshold: 0.5
Confusion Matrix:
 [[160   3]
 [ 25  26]]
Accuracy: 0.8691588785046729
Precision: 0.896551724137931
Recall: 0.5098039215686274

Threshold: 0.7
Confusion Matrix:
 [[163   0]
 [ 35  16]]
Accuracy: 0.8364485981308412
Precision: 1.0
Recall: 0.3137254901960784


# 2. Do the same analysis for other columns
=== Feature: RI ===

Threshold: 0.3
Confusion Matrix:
 [[163   0]
 [ 51   0]]
Accuracy: 0.7616822429906542
Precision: 0.0
Recall: 0.0

Threshold: 0.5
Confusion Matrix:
 [[163   0]
 [ 51   0]]
Accuracy: 0.7616822429906542
Precision: 0.0
Recall: 0.0

Threshold: 0.7
Confusion Matrix:
 [[163   0]
 [ 51   0]]
Accuracy: 0.7616822429906542
Precision: 0.0
Recall: 0.0

=== Feature: Na ===

Threshold: 0.3
Confusion Matrix:
 [[142  21]
 [ 16  35]]
Accuracy: 0.8271028037383178
Precision: 0.625
Recall: 0.6862745098039216

Threshold: 0.5
Confusion Matrix:
 [[159   4]
 [ 27  24]]
Accuracy: 0.8551401869158879
Precision: 0.8571428571428571
Recall: 0.47058823529411764

Threshold: 0.7
Confusion Matrix:
 [[163   0]
 [ 42   9]]
Accuracy: 0.8037383177570093
Precision: 1.0
Recall: 0.17647058823529413

=== Feature: Mg ===

Threshold: 0.3
Confusion Matrix:
 [[152  11]
 [  9  42]]
Accuracy: 0.9065420560747663
Precision: 0.7924528301886793
Recall: 0.8235294117647058

Threshold: 0.5
Confusion Matrix:
 [[153  10]
 [ 16  35]]
Accuracy: 0.8785046728971962
Precision: 0.7777777777777778
Recall: 0.6862745098039216

Threshold: 0.7
Confusion Matrix:
 [[154   9]
 [ 17  34]]
Accuracy: 0.8785046728971962
Precision: 0.7906976744186046
Recall: 0.6666666666666666

=== Feature: Al ===

Threshold: 0.3
Confusion Matrix:
 [[149  14]
 [ 15  36]]
Accuracy: 0.8644859813084113
Precision: 0.72
Recall: 0.7058823529411765

Threshold: 0.5
Confusion Matrix:
 [[160   3]
 [ 25  26]]
Accuracy: 0.8691588785046729
Precision: 0.896551724137931
Recall: 0.5098039215686274

Threshold: 0.7
Confusion Matrix:
 [[163   0]
 [ 35  16]]
Accuracy: 0.8364485981308412
Precision: 1.0
Recall: 0.3137254901960784

=== Feature: Si ===

Threshold: 0.3
Confusion Matrix:
 [[158   5]
 [ 37  14]]
Accuracy: 0.8037383177570093
Precision: 0.7368421052631579
Recall: 0.27450980392156865

Threshold: 0.5
Confusion Matrix:
 [[163   0]
 [ 49   2]]
Accuracy: 0.7710280373831776
Precision: 1.0
Recall: 0.0392156862745098

Threshold: 0.7
Confusion Matrix:
 [[163   0]
 [ 51   0]]
Accuracy: 0.7616822429906542
Precision: 0.0
Recall: 0.0


=== Feature: K ===

Threshold: 0.3
Confusion Matrix:
 [[163   0]
 [ 48   3]]
Accuracy: 0.7757009345794392
Precision: 1.0
Recall: 0.058823529411764705

Threshold: 0.5
Confusion Matrix:
 [[163   0]
 [ 51   0]]
Accuracy: 0.7616822429906542
Precision: 0.0
Recall: 0.0

Threshold: 0.7
Confusion Matrix:
 [[163   0]
 [ 51   0]]
Accuracy: 0.7616822429906542
Precision: 0.0
Recall: 0.0


=== Feature: Ca ===

Threshold: 0.3
Confusion Matrix:
 [[159   4]
 [ 51   0]]
Accuracy: 0.7429906542056075
Precision: 0.0
Recall: 0.0

Threshold: 0.5
Confusion Matrix:
 [[163   0]
 [ 51   0]]
Accuracy: 0.7616822429906542
Precision: 0.0
Recall: 0.0

Threshold: 0.7
Confusion Matrix:
 [[163   0]
 [ 51   0]]
Accuracy: 0.7616822429906542
Precision: 0.0
Recall: 0.0


=== Feature: Ba ===

Threshold: 0.3
Confusion Matrix:
 [[161   2]
 [ 24  27]]
Accuracy: 0.8785046728971962
Precision: 0.9310344827586207
Recall: 0.5294117647058824

Threshold: 0.5
Confusion Matrix:
 [[161   2]
 [ 33  18]]
Accuracy: 0.8364485981308412
Precision: 0.9
Recall: 0.35294117647058826

Threshold: 0.7
Confusion Matrix:
 [[162   1]
 [ 36  15]]
Accuracy: 0.8271028037383178
Precision: 0.9375
Recall: 0.29411764705882354


=== Feature: Fe ===

Threshold: 0.3
Confusion Matrix:
 [[163   0]
 [ 51   0]]
Accuracy: 0.7616822429906542
Precision: 0.0
Recall: 0.0

Threshold: 0.5
Confusion Matrix:
 [[163   0]
 [ 51   0]]
Accuracy: 0.7616822429906542
Precision: 0.0
Recall: 0.0

Threshold: 0.7
Confusion Matrix:
 [[163   0]
 [ 51   0]]
Accuracy: 0.7616822429906542
Precision: 0.0
Recall: 0.0

# 3. Fit a Logistic Regression Model on all features. Remember to preprocess data(eg. normalization and one hot encoding)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score

glass = pd.read_csv('glass.csv')
print(glass.head())
print(glass.Type.value_counts().sort_index())

glass['household'] = glass.Type.map({1:0, 2:0, 3:0, 5:1, 6:1, 7:1})
print(glass.household.value_counts())

glass.drop(columns=['Type'], inplace=True)

X = glass.drop(columns=['household'])
y = glass['household']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets (optional but better practice)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Fit logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Predict probabilities and classes
y_pred_prob = logreg.predict_proba(X_test)[:, 1]
y_pred = (y_pred_prob >= 0.5).astype(int)

# Evaluation
cm = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)
print('Confusion Matrix:\n', cm)

accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
precision = precision_score(y_true=y_test, y_pred=y_pred)
recall = recall_score(y_true=y_test, y_pred=y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)

# Plot predicted probabilities
plt.hist(y_pred_prob, bins=20)
plt.xlabel('Predicted Probability of household')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted Probabilities')
plt.show()

# Plot ROC Curves for each model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

glass = pd.read_csv('glass.csv')
print(glass.head())
print(glass.Type.value_counts().sort_index())

glass['household'] = glass.Type.map({1:0, 2:0, 3:0, 5:1, 6:1, 7:1})
print(glass.household.value_counts())

glass.drop(columns=['Type'], inplace=True)

# Features and target
X_full = glass.drop(columns=['household'])
y = glass['household']

# Standardize all features
scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full)

# Train/test split
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full_scaled, y, test_size=0.3, random_state=42)

# Logistic Regression on ALL features
logreg_full = LogisticRegression()
logreg_full.fit(X_train_full, y_train_full)

y_pred_prob_full = logreg_full.predict_proba(X_test_full)[:,1]
fpr_full, tpr_full, _ = roc_curve(y_test_full, y_pred_prob_full)
roc_auc_full = auc(fpr_full, tpr_full)

# List of single features
features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']

# Plot ROC Curves
plt.figure(figsize=(10, 8))

# Single feature models
for feature in features:
    X_single = glass[[feature]].values
    X_single_scaled = scaler.fit_transform(X_single)

    X_train, X_test, y_train, y_test = train_test_split(X_single_scaled, y, test_size=0.3, random_state=42)

    logreg_single = LogisticRegression()
    logreg_single.fit(X_train, y_train)

    y_pred_prob_single = logreg_single.predict_proba(X_test)[:,1]
    fpr_single, tpr_single, _ = roc_curve(y_test, y_pred_prob_single)
    roc_auc_single = auc(fpr_single, tpr_single)

    plt.plot(fpr_single, tpr_single, label=f'{feature} (AUC = {roc_auc_single:.2f})')

# Add the model trained on all features
plt.plot(fpr_full, tpr_full, label=f'All Features (AUC = {roc_auc_full:.2f})', linewidth=3, color='black')

# Plot settings
plt.plot([0, 1], [0, 1], 'k--')  # random guess line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Model')
plt.legend(loc="lower right")
plt.grid()
plt.show()


# Clustering:

# Repeat the above exercise for different values of k
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

# Load data
glass = pd.read_csv('glass.csv')

# Map Type to household
glass['household'] = glass.Type.map({1:0, 2:0, 3:0, 5:1, 6:1, 7:1})
glass.drop(columns=['Type'], inplace=True)

# Features and target
X = glass.drop(columns=['household'])
y = glass['household']

# Calculate correlation with target
correlations = X.corrwith(y).abs().sort_values(ascending=False)
print("Feature Correlations with 'household':\n", correlations)

# Values of k to try
k_values = [1, 2, 3, 5, 7, 9]  # 9 = use all features

# Plot ROC Curves
plt.figure(figsize=(10, 8))

for k in k_values:
    top_k_features = correlations.index[:k]
    print(f"\nUsing top {k} features: {list(top_k_features)}")

    X_k = X[top_k_features]
    
    # Standardize
    scaler = StandardScaler()
    X_k_scaled = scaler.fit_transform(X_k)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_k_scaled, y, test_size=0.3, random_state=42)

    # Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    y_pred_prob = logreg.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'k={k} (AUC = {roc_auc:.2f})')

# Plot random guess line
plt.plot([0, 1], [0, 1], 'k--')

# Plot settings
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Values of k')
plt.legend(loc="lower right")
plt.grid()
plt.show()

# How do the inertia and silhouette scores change?
    Inertia always decreases as k increases.
    Silhouette score increases at first, peaks at a good k, then decreases.
      
# What if you don't scale your features?
    Distance is biased; Clusters are distorted; Bad inertia; Wrong silhouette scores; and Poor clustering quality;

# Is there a 'right' k? Why or why not?
    No, because the possible k is not unique for each cluster seperation.

# Repeat the following exercise for food nutrients dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Load the food nutrients dataset
food = pd.read_csv('food_nutrients.csv')

# 2. Preprocessing

# Drop non-numeric columns (e.g., food names) if any
# Keep only numeric nutrient features
numeric_cols = food.select_dtypes(include=[np.number]).columns
X = food[numeric_cols]

# Handle missing values (drop or impute)
X = X.dropna()  # simplest; or you can use imputation

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Try different k values
k_values = range(2, 11)  # Try k from 2 to 10
inertias = []
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X_scaled)
    
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(sil_score)

# 4. Plot Inertia and Silhouette

plt.figure(figsize=(12,5))

# Inertia (Elbow method)
plt.subplot(1,2,1)
plt.plot(k_values, inertias, marker='o')
plt.title('Inertia vs k (Elbow Method)')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.grid()

# Silhouette Score
plt.subplot(1,2,2)
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Score vs k')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.grid()

plt.tight_layout()
plt.show()
