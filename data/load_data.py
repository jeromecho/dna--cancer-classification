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

# Display updated dataframe info
pan_cancer_df.info()

"""
PREPROCESSING
"""
# Typecast columns
numeric_columns = pan_cancer_df.select_dtypes(exclude=[object, float]).columns
pan_cancer_df[numeric_columns] = pan_cancer_df[numeric_columns].astype('int32')

# Columns to lowercase
pan_cancer_df.columns = pan_cancer_df.columns.str.lower()

# Rename columns
pan_cancer_df.rename(columns={'#"chrom"':'chrom',
                               'project_id':'cancer_type', 'variant_classification':'variant', 'matched_norm_sample_barcode':'barcode'}, inplace=True)

# Split records
# Replacing ‘ — ‘ with nulls
# Check missing values
# Drop columns
# Handle missing values