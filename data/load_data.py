import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
import xgboost as xgb
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# Pan Cancer dataframe
pan_cancer_df = pd.DataFrame()

# Cancer types
cancers = [
    "KICH", "ACC", "BLCA", "BRCA", "CESC", "ESCA", "KIRC", "LAML", "LGG",
    "LIHC", "OV", "PAAD", "PRAD", "READ", "TGCT", "THCA", "UCEC", "LUAD", "COAD", "SKCM"
]
cancers_ = ['luad', 'ucec', 'coad', 'skcm']

# Import data
# all data
for cancer in cancers:
  data = pd.read_csv(f'./content/{cancer}.csv')
  pan_cancer_df = pd.concat([pan_cancer_df, data], ignore_index=True)

# rename column names
# Define new column names
new_column_names = [
    "chrom", "chromStart", "chromEnd", "name", "score", "strand", "thickStart", "thickEnd", "reserved",
    "blockCount", "blockSizes", "chromStarts", "sampleCount", "freq", "Hugo_Symbol", "Entrez_Gene_Id",
    "Variant_Classification", "Variant_Type", "Reference_Allele", "Tumor_Seq_Allele1", "Tumor_Seq_Allele2",
    "dbSNP_RS", "dbSNP_Val_Status", "days_to_death", "cigarettes_per_day", "weight", "alcohol_history",
    "alcohol_intensity", "bmi", "years_smoked", "height", "gender", "project_id", "ethnicity",
    "Tumor_Sample_Barcode", "Matched_Norm_Sample_Barcode", "case_id"
]

# Assign new column names
pan_cancer_df.columns = new_column_names

"""
PREPROCESSING
"""
""" 
DATA CLEANING
"""
def swap_cancer_value(row):
    project_val = row["project_id"]
    for col in row.index:
        if col == "project_id":
            continue
        cell_val = str(row[col]).upper()  # Convert to uppercase for consistency
        # Check if the value matches "TCGA-" + a known cancer name
        for cancer in cancers:
            tcga_label = f"TCGA-{cancer}"
            if tcga_label in cell_val:
                # Swap project_id and the column's value
                row["project_id"], row[col] = cell_val, project_val
                return row  # Stop after the first swap
    return row

# Apply the function to every row
pan_cancer_df = pan_cancer_df.apply(swap_cancer_value, axis=1)

# Delete all rows that don't have an associated cancer name
# Get the initial row count
initial_row_count = len(pan_cancer_df)
# Filter out rows where 'project_id' is not in the correct format
valid_project_ids = [f"TCGA-{cancer}" for cancer in cancers]
pan_cancer_df = pan_cancer_df[pan_cancer_df["project_id"].isin(valid_project_ids)]
final_row_count = len(pan_cancer_df)
# Calculate the number of dropped rows
dropped_rows = initial_row_count - final_row_count
# Print how many rows were dropped
print(f"Dropped {dropped_rows} rows where 'project_id' did not match 'TCGA-<CANCER_NAME>'.")
# Reset index after filtering
pan_cancer_df = pan_cancer_df.reset_index(drop=True)

"""
DATA TRANSFORMATIONS/TRUNCATION
"""

# Typecast columns
numeric_columns = pan_cancer_df.select_dtypes(exclude=[object, float]).columns
pan_cancer_df[numeric_columns] = pan_cancer_df[numeric_columns].astype('int32')

# Columns to lowercase
pan_cancer_df.columns = pan_cancer_df.columns.str.lower()

# Rename columns
pan_cancer_df.rename(columns={'#"chrom"':'chrom',
                               'project_id':'cancer_type', 'variant_classification':'variant', 'matched_norm_sample_barcode':'barcode'}, inplace=True)

# TODO - maybe get rid of this?
# Specify the columns to split
columns_to_split = ['days_to_death', 'cigarettes_per_day', 'weight', 'alcohol_history', 'alcohol_intensity', 'bmi', 'years_smoked', 'height', 'gender', 'ethnicity', 'tumor_sample_barcode', 'barcode', 'case_id']

# Convert columns to strings and then split values
pan_cancer_df.loc[:, columns_to_split] = pan_cancer_df[columns_to_split].astype(str).apply(lambda x: x.str.split(','))

# Explode the specified columns and reset the index
pan_cancer_df = pan_cancer_df.explode(columns_to_split).reset_index(drop=True)

# visual
# Replace '--' with nan
pan_cancer_df.replace('--', np.nan, inplace=True)

# Check missing values
pan_cancer_df.isnull().sum()

# Drop columns based on null values
pan_cancer_df.drop(columns=['dbsnp_rs',
                            'dbsnp_val_status',
                            'days_to_death',
                            'cigarettes_per_day', 
                            'weight',
                            'alcohol_history', 
                            'alcohol_intensity', 
                            'years_smoked', 
                            'height', 
                            'ethnicity', 
                            'bmi'], 
                            inplace=True)

# ADDED: drop colors with no significant variance in data
pan_cancer_df.drop(columns=['blocksizes',
                            'freq',
                            'hugo_symbol'], 
                            inplace=True)


# Drop columns based on insignificance
pan_cancer_df.drop(columns=['case_id',
                            'reserved', 
                            'blockcount', 
                            'score', 
                            'strand', 
                            'chromstarts', 
                            'samplecount', 
                            'tumor_sample_barcode',
                            'entrez_gene_id'], 
                            inplace=True)

# Imputation
# Calculate the distribution of existing gender values
gender_distribution = pan_cancer_df['gender'].value_counts(normalize=True)

# Create a mask for null gender values
null_mask = pan_cancer_df['gender'].isnull()

# Fill null gender values with random genders based on the distribution
random_genders = np.random.choice(gender_distribution.index, size=null_mask.sum(), p=gender_distribution.values)
pan_cancer_df.loc[null_mask, 'gender'] = random_genders

"""
N: data is still noisy with mislabele data!
"""

"""
(MORE) DATA PROPROCESSING
"""

# categorical -> numerical data
# Label encoding
le = LabelEncoder()

class_labels = {}

# Iterate through columns and apply label encoding
for column in pan_cancer_df.columns:
    if pan_cancer_df[column].dtype == 'object':
        print(column)
        pan_cancer_df[column] = le.fit_transform(pan_cancer_df[column])
    
    if column == 'cancer_type':
            class_labels = {i: label for i, label in enumerate(le.classes_)}

print("CLASS LABELS")
print(class_labels.values())

"""
DATA ANALYSIS
"""

print(pan_cancer_df.corr()['cancer_type'])
# correlation matrix
# Create the heatmap using the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(pan_cancer_df.corr(), linewidths=0.5)
plt.title('Correlation Heatmap of Pan Cancer Features')

# plt.show()

"""
FEATURE SELECTION
"""

# Select required features
pan_cancer_df = pan_cancer_df[['chrom',
                               'chromstart',
                               'chromend', 
                               'name', 
                               'thickstart', 
                               'thickend',
                               'variant', 
                               'variant_type', 
                               'reference_allele', 
                               'tumor_seq_allele1', 
                               'tumor_seq_allele2', 
                               'gender', 
                               'cancer_type']]

# Splitting Data

# Split data into dependent/independent variables
X = pan_cancer_df.iloc[:, :-1].values
y = pan_cancer_df.iloc[:, -1].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


""" 
TRAINING
"""
# Cross validation
def cross_val(model, X_train, X_test, y_train, n_splits=5):
  oofs = np.zeros(len(X_train))
  preds = np.zeros(len(X_test))

  target_col = pd.DataFrame(data=y_train)

  folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
  stratified_target = pd.qcut(y_train, 10, labels=False, duplicates='drop')

  for index, (trn_idx, val_idx) in enumerate(folds.split(X_train, stratified_target)):
    print(f'\n================================ Fold {index + 1} ===================================')

    cv_X_train, cv_y_train = X_train[trn_idx], target_col.iloc[trn_idx]
    cv_X_val, cv_y_val = X_train[val_idx], target_col.iloc[val_idx]

    model.fit(cv_X_train, cv_y_train)

    val_preds = model.predict(cv_X_val)
    test_preds = model.predict(X_test)

    error = precision_score(cv_y_val, val_preds, average='macro')
    print(f'Precision is : {error}')

    oofs[val_idx] = val_preds
    preds += test_preds/n_splits

  total_error = precision_score(target_col, oofs, average='macro')
  print(f'\n Precision is {total_error}')

  return oofs, preds

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=10, random_state=42)
print("--- TRAINING RANDOM FOREST ---")
rf_classifier.fit(X_train, y_train)
rf_oofs, rf_pred = cross_val(rf_classifier, X_train, X_test, y_train, 5)
rf_pred = rf_pred.astype(int)

# XGBoost Classifier
xgb_classifier = xgb.XGBClassifier(objective="multi:softmax", num_class=20, n_estimators=10, random_state=42)

# Cross validation
print("--- CROSS VALIDATING XGBOOST ---")
xgb_oofs, xgb_pred = cross_val(xgb_classifier, X_train, X_test, y_train, 5)


"""
METRICS
"""

# Calculate the confusion matrix
print(len(y_test))
print(y_test[:20])
print(len(xgb_pred))
xgb_pred = xgb_pred.astype(int)
print(xgb_pred[:20])

cm = confusion_matrix(y_test, xgb_pred)

# Create a Seaborn heatmap for the confusion matrix
plt.figure(figsize=(12, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels.values(), yticklabels=class_labels.values())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ACCURACY 
# Random Forest
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest Classifier Accuracy: {rf_accuracy:.4f}")

# XGBoost
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print(f"XGBoost Classifier Accuracy: {xgb_accuracy:.4f}")

# PRECISION
# Random Forest
rf_precision = precision_score(rf_pred, y_test, average='macro')
print(f"Random Forest Classifier Precision: {rf_precision:.4f}")

# XGBoost
xgb_precision = precision_score(xgb_pred, y_test, average='macro')
print(f"XGBoost Classifier Precision: {xgb_precision:.4f}")

# Precision-Recall Curve
# Binarize labels
y_test_bin = label_binarize(y_test, classes=np.arange(20))

# Classes
class_names = list(class_labels.values())

# Plot curve
fig, axs = plt.subplots(4, 5, figsize=(12, 9))
fig.suptitle("Precision-Recall Curves", fontsize=16)
axs = axs.flatten()

for class_label in range(20):
  # Calculate precision/recall for current class
  xgb_precision, xgb_recall, _ = precision_recall_curve(y_test_bin[:, class_label], xgb_classifier.predict_proba(X_test)[:, class_label])

  # Plot current curve
  axs[class_label].plot(xgb_recall, xgb_precision) #, label=f"{class_names[class_label]}", color='r')
  axs[class_label].set_xlabel("Recall")
  axs[class_label].set_ylabel("Precision")
  axs[class_label].set_title(f"{class_names[class_label]}")
  #axs[class_label].legend(loc='best')
  axs[class_label].grid()

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# AUC/ROC Curve
# Plot curve
fig, axs = plt.subplots(4, 5, figsize=(18, 14))
fig.suptitle("Precision-Recall and AUC-ROC Curves", fontsize=16)
axs = axs.flatten()

for class_label in range(20):
    # Calculate ROC curve for current class
    fpr, tpr, _ = roc_curve(y_test_bin[:, class_label], xgb_classifier.predict_proba(X_test)[:, class_label])

    # Calculate AUC for ROC curve
    roc_auc = auc(fpr, tpr)

    # Plot Precision-Recall curve for the current class
    axs[class_label].plot(xgb_recall, xgb_precision) #, label=f"PR Curve ({class_names[class_label]})", color='r')
    axs[class_label].plot(fpr, tpr, label=f"AUC-ROC ({class_names[class_label]}) = {roc_auc:.2f}", color='b', linestyle='--')
    axs[class_label].set_xlabel("False Positive Rate")
    axs[class_label].set_ylabel("True Positive Rate")
    axs[class_label].set_title(f"{class_names[class_label]}")
    #axs[class_label].legend(loc='best')
    axs[class_label].grid()

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

"""
IMPROVEMENTS: 
- clean data and fix arrangement of columns
"""