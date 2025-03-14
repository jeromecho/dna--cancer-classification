# dna->cancer-classification

First bioinformatics project. Multi-class cancer classification using pan-cancer somatic data from TCGA (The Cancer Genome Atlas) data.

### PICTURES

<img width="500" alt="image" src="https://github.com/user-attachments/assets/3d2fd584-9449-4dce-b4b3-a91ba74bfe7c" />
<img width="500" alt="image" src="https://github.com/user-attachments/assets/2e3cb6a8-77eb-4a4e-b65c-41ff5759a933" />
<img width="500" alt="image" src="https://github.com/user-attachments/assets/814d3c19-8818-46e8-a621-c25c83d0b642" />
<img width="500" alt="image" src="https://github.com/user-attachments/assets/6f7febe0-234d-4eda-bf8c-5f7d76333e49" />


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

### Improvement Areas (ongoing) 

- Try training on the entirety of the dataset (2,000,000+ rows of data) 
- Try applying techniques for mislabeled and noisy data, such as a robust loss function

### Data/Credits

Data: https://genome.ucsc.edu/cgi-bin/hgTables?db=hg38&hgta_group=phenDis&hgta_track=gdcCancer&hgta_table=allCancer&hgta_doSchema=describe+table+schema
Credit where credit is due: https://medium.com/@shuv.sdr/cancer-prediction-from-genomic-analysis-with-machine-learning-2c957b579f05
