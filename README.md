# dna->cancer-classification

First bioinformatics project. Multi-class cancer classification using TCGA (The Cancer Genome Atlas) data.

### PICTURES

### TECH:
- Python 
- BigBed

### Metrics

Trained on 80,000 rows of noisy data with lots of mislabeling.

- Random Forest Classifier Accuracy: 0.2196
- XGBoost Classifier Accuracy: 0.4288
- Random Forest Classifier Precision: 0.0787
- XGBoost Classifier Precision: 0.1105

Since there are 20 types of cancer, so a baseline would achieve an expected accuracy of 
**0.05**, so the model seems to demonstrate significant learning

Credit where credit is due: https://medium.com/@shuv.sdr/cancer-prediction-from-genomic-analysis-with-machine-learning-2c957b579f05