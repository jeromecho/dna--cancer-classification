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
from xgboost import XGBClassifier
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

# Iterate through columns and apply label encoding
for column in pan_cancer_df.columns:
    if pan_cancer_df[column].dtype == 'object':
        pan_cancer_df[column] = le.fit_transform(pan_cancer_df[column])

class_labels = {i: label for i, label in enumerate(le.classes_)}


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


"""
IMPROVEMENTS: 
- clean data and fix arrangement of columns
"""