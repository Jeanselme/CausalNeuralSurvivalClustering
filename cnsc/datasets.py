from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from pycox import datasets
import pandas as pd
import numpy as np
import pickle

EPS = 1e-8

def load_dataset(dataset='SUPPORT', standardisation = True, **kwargs):
    """
        Preprocess dataset 
        Returns x, mask, t, e, names
    """
    if dataset == 'GBSG':
        df = datasets.gbsg.read_df()
        df = df.rename(columns = {'x0': 'Hormone', 'x1': 'Age', 'x2': 'Menopause', 'x3': 'Tumor Size', 
                                  'x4': 'Positive Nodes', 'x5': 'Progesterone Receptor', 'x6': 'Estrogen Receptor'})
        df = df.rename(columns = {"Hormone": 'treatment'})
    elif dataset == 'METABRIC':
        df = datasets.metabric.read_df()
        df = df.rename(columns = {'x0': 'MKI67', 'x1': 'EGFR', 'x2': 'PGR', 'x3': 'ERBB2', 
                                  'x4': 'Hormone', 'x5': 'Radiotherapy', 'x6': 'Chemotherapy', 'x7': 'ER-positive', 
                                  'x8': 'Age at diagnosis'})
        df['treatment'] = df['Radiotherapy'].astype(bool)
        df['duration'] += EPS
    elif dataset == 'SEER':
        subset = kwargs.pop('subset', False)
        path = kwargs.pop('path', '~/Desktop/Thesis/NeuralTreatment/data/') 
        df = pd.read_csv(path + 'export.csv')
        df = process_seer(df, subset = subset)
        df['duration'] += EPS # Avoid problem of the minimum value 0

        # Answer the question: Does patients under chemotherapy can benefit from radiation ?
        df['treatment'] = df['Radiation recode'].astype(bool)
        df = df.drop(columns = ['Radiation recode'])
    elif dataset == 'FRAMINGHAM':
        path = kwargs.pop('path', '~/Desktop/Thesis/NeuralTreatment/auton-survival/dsm/datasets/')
        df = pd.read_csv(path + 'framingham.csv')
        df = process_framingham(df)
        df['duration'] += EPS   
        df = df.rename(columns = {"BPMEDS": 'treatment'})

    covariates = df.drop(['duration', 'event', 'treatment'], axis = 'columns')
    return (StandardScaler().fit_transform(covariates.values) if standardisation else covariates.values).astype(float),\
           df['treatment'].values.astype(int),\
           df['duration'].values.astype(float),\
           df['event'].values.astype(int),\
           covariates.columns

def process_seer(df, subset = False):
    # Remove multiple visits
    df = df.groupby('Patient ID').first().drop(columns= ['Site recode ICD-O-3/WHO 2008'])

    # Encode using dictionary to remove missing data
    df["RX Summ--Surg Prim Site (1998+)"].replace('126', np.nan, inplace = True)

    # Remove not grades
    grades = ['Well differentiated; Grade I', 'Moderately differentiated; Grade II',
       'Poorly differentiated; Grade III', 'Undifferentiated; anaplastic; Grade IV']
    df = df[df["Grade (thru 2017)"].isin(grades)]

    categorical_col = ["Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)", "Laterality", 
        "Diagnostic Confirmation", "Histology recode - broad groupings", 
        "Radiation recode", "ER Status Recode Breast Cancer (1990+)", "PR Status Recode Breast Cancer (1990+)",
        "Histologic Type ICD-O-3", "ICD-O-3 Hist/behav, malignant", "Sequence number", "Derived HER2 Recode (2010+)",
        "CS extension (2004-2015)", "CS lymph nodes (2004-2015)", "CS mets at dx (2004-2015)", "Origin recode NHIA (Hispanic, Non-Hisp)"]

    # Remove patients without surgery
    df = df[df["RX Summ--Surg Prim Site (1998+)"] != '00']

    # Remove patients without chemo
    df = df[df["Chemotherapy recode (yes, no/unk)"] == 'Yes']

    if subset:
        # Select patients with likely hormonal therapy
        df = df[(df["ER Status Recode Breast Cancer (1990+)"] == 'Positive') |  (df["PR Status Recode Breast Cancer (1990+)"] == 'Positive') |  (df["Derived HER2 Recode (2010+)"] == 'Positive')]        

        categorical_col = ["Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)", "Laterality", 
            "Diagnostic Confirmation", "Histology recode - broad groupings", 
            "Radiation recode", "Histologic Type ICD-O-3", "ICD-O-3 Hist/behav, malignant", "Sequence number",
            "CS extension (2004-2015)", "CS lymph nodes (2004-2015)", "CS mets at dx (2004-2015)", "Origin recode NHIA (Hispanic, Non-Hisp)"]

    df["Sequence number"].replace(['88', '99'], np.nan, inplace = True)
    df["Regional nodes positive (1988+)"].replace(['95', '96', '97', '98', '99', '126'], np.nan, inplace = True)
    df["Regional nodes examined (1988+)"].replace(['95', '96', '97', '98', '99', '126'], np.nan, inplace = True)
    df = df.replace(['Blank(s)', 'Unknown'], np.nan).rename(columns = {"Survival months": "duration"})

    # Remove patients without survival time
    df = df[~df.duration.isna()]

    # Outcome 
    df['duration'] = df['duration'].astype(float)
    df['event'] = df["SEER cause-specific death classification"] == "Dead (attributable to this cancer dx)" # Death 

    df = df.drop(columns = ["COD to site recode"])
    df['Radiation recode'] = ~((df['Radiation recode'] == 'None/Unknown') | (df['Radiation recode'] == 'Refused (1988+)'))

    # Imput and encode categorical
    ordinal_col = ["Age recode with <1 year olds", "Grade (thru 2017)", "Year of diagnosis"]

    imputer = SimpleImputer(strategy='most_frequent')
    enc = OrdinalEncoder()
    df_cat = pd.DataFrame(enc.fit_transform(imputer.fit_transform(df[categorical_col])), columns = categorical_col, index = df.index)
    df_ord = pd.DataFrame(imputer.fit_transform(df[ordinal_col]), columns = ordinal_col, index = df.index)
    df_ord = df_ord.replace(
      {age: number
        for number, age in enumerate(['00 years', '01-04 years', '05-09 years', '10-14 years', '15-19 years', '20-24 years', '25-29 years',
        '30-34 years', '35-39 years', '40-44 years', '45-49 years', '50-54 years', '55-59 years', 
        '60-64 years', '65-69 years', '70-74 years', '75-79 years', '80-84 years', '85+ years'])
      }).replace({
        grade: number
        for number, grade in enumerate(grades)
      })

    ## Numerical
    numerical_col = ["Total number of in situ/malignant tumors for patient", "Total number of benign/borderline tumors for patient",
          "CS tumor size (2004-2015)", "Regional nodes examined (1988+)", "Regional nodes positive (1988+)"]
    imputer = SimpleImputer(strategy='mean')
    df_num = pd.DataFrame(imputer.fit_transform(df[numerical_col].astype(float)), columns = numerical_col, index = df.index)

    concat = pd.concat([df_cat, df_ord, df_num, df[['duration', 'event']]], axis = 1)
    return concat


def process_framingham(df):
    """
        Helper function to preprocess the Framingham dataset in the treatment context.
        Define time 0 as first point with treatment or first point observed
    """
    # Ignore time points with unsure time
    df = df[~df.BPMEDS.isna()]

    # Take first time with treatment with other
    # Or before CVD
    treated = df[['BPMEDS', 'RANDID']].groupby('RANDID').sum().BPMEDS > 0
    treated = df.groupby('RANDID').apply(lambda x: x.loc[x.BPMEDS.idxmax()])[treated]
    # Only keep patients as treated if treatment starts before event
    treated = treated[(treated.TIMECVD - treated.TIME) > 0]  

    # Keep all patients with no treatment
    not_treated = df.groupby('RANDID').first()
    not_treated = not_treated.loc[not_treated.index.difference(treated.index)]
    
    # Reassemble
    df = pd.concat([treated, not_treated], axis = 0, sort = True)

    dat_cat = df[['SEX', 'CURSMOKE', 'DIABETES', 'BPMEDS',
                    'educ', 'PREVCHD', 'PREVAP', 'PREVMI',
                    'PREVSTRK', 'PREVHYP']]
    dat_num = df[['TOTCHOL', 'AGE', 'SYSBP', 'DIABP',
                    'CIGPDAY', 'BMI', 'HEARTRTE', 'GLUCOSE']]

    x1 = pd.get_dummies(dat_cat)
    x2 = dat_num
    x = np.hstack([x1.values, x2.values])

    time = (df['TIMEDTH'] - df['TIME']).values
    event = df['DEATH'].values
    
    df = pd.DataFrame(SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(x), columns = np.concatenate([x1.columns, x2.columns]), index = df.index)
    df['duration'] = time / 365.
    df['event'] = event

    return df[df.duration > 0]