# -*- coding: utf-8 -*-
#!/usr/bin/env python
# coding: utf-8

# In[176]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 76)
sns.set(style="whitegrid")
import warnings


# In[177]:


#amina path
#df = pd.read_csv("/Users/aminarabehi/Downloads/Project 2/Patient_Characteristics_Survey__PCS___2019.csv", sep=";")


# In[178]:


#nirusa path
#df = pd.read_csv("C:/Users/nirus/Documents/DSTI/Ml_project_MI/Patient_Characteristics_Survey__PCS___2019.csv", sep=";")


# In[179]:


#arjun path
file_path = "D:/dstiprojects/Project 2/Project 2/Patient_Characteristics_Survey__PCS___2019.csv"
df = pd.read_csv(file_path, sep=';')


# In[180]:


df.shape


# In[181]:


df.head()


# In[182]:


df.tail()


# # I-DATA CLEANING

# In[183]:


df = df.drop(columns=['Survey Year', 'Three Digit Residence Zip Code'])

print("\nColumns after dropping:")
print(df.columns.tolist())


# In[184]:


df.dtypes.value_counts()


# In[185]:


# List of placeholder values that actually mean "missing"
placeholders = [
    'UNKNOWN', 'UNKNOWN RACE', 'UNKNOWN EMPLOYMENT STATUS',
    'UNKNOWN EMPLOYMENT HOURS', 'UNKNOWN INSURANCE COVERAGE',
    'UNKNOWN CHRONIC MED CONDITION', 'DATA NOT AVAILABLE',
    'CLIENT DID NOT ANSWER', "CLIENT DIDN'T ANSWER"
]
df.replace(placeholders, np.nan, inplace=True)
#on les remplace par NaN


# # II- EDA

# Target vizualisation

# In[186]:


df['Mental Illness'] = df['Mental Illness'].where(
    df['Mental Illness'].isna(),
    df['Mental Illness'].str.upper().str.strip()
)


# In[187]:


df["Mental Illness"] = df["Mental Illness"].map({
    'YES': 1,
    'NO': 0,
    '1': 1,
    '0': 0
})


# In[188]:


df["Mental Illness"] = df["Mental Illness"].astype('Int64')


# In[189]:


#Correction de potentiel incoherence 
mask = (
    (df['Mental Illness'] == 1) &
    (df['Serious Mental Illness'] != 'YES') &
    (df['Principal Diagnosis Class'] != 'MENTAL ILLNESS') &
    (df['Additional Diagnosis Class'] != 'MENTAL ILLNESS')
)

df.loc[mask, 'Mental Illness'] = 0
print(f"{mask.sum()} valeurs corrigées à 0 dans 'Mental Illness' car non justifiées")


# In[190]:


df['Mental Illness'].value_counts(dropna=False)


# In[ ]:





# Vizualization de nos données qualitatifs/ catégorielles (object)

# In[191]:


#rendre en MAJ sans toucher les NaN
for col in df.select_dtypes('object'):
    df[col] = df[col].where(df[col].isna(), df[col].str.upper())


# In[192]:


for col in df.select_dtypes('object') :
  print(f'{col:-<50}, {df[col].unique()}')


# In[193]:


for col in df.select_dtypes('object') :
    plt.figure(figsize=(5, 5))
    df[col].value_counts(dropna=False).plot.pie()
    plt.show()


# In[194]:


for col in df.select_dtypes('bool') :
  print(f'{col:-<50}, {df[col].unique()}')


# In[195]:


for col in df.select_dtypes('bool') :
    plt.figure(figsize=(5, 5))
    df[col].value_counts(dropna=False).plot.pie()
    plt.show()


# In[196]:


def afficher_distribution(colonnes, titre_bloc=None):
    """Affiche les valeurs absolues, les pourcentages et un graphique"""

    if titre_bloc:
        print("\n" + titre_bloc)
        print("-" * len(titre_bloc))

    for col in colonnes:
        print(f"\n{col} — Valeurs absolues")
        print(df[col].value_counts(dropna=False))

        print(f"\n{col} — Pourcentages")
        print(round(df[col].value_counts(normalize=True, dropna=False) * 100, 2))

        # Graphique
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col, order=df[col].value_counts(dropna=False).index, hue=col, palette="pastel", legend=False)
        plt.title(col)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

# Liste des colonnes qu'on veut visualiser
colonnes_objet = df.select_dtypes(include='object').columns

# Appel de la fonction
afficher_distribution(colonnes_objet, "Distributions après imputation")


# Creation of thematic sets:

# In[197]:


#ensemble socio-demographique
socio_cols = [
    'Age Group', 'Sex', 'Transgender', 'Sexual Orientation',
    'Hispanic Ethnicity', 'Race', 'Preferred Language',
    'Religious Preference', 'Region Served'
    ]


# In[198]:


#ensemble travail/etude
stuwork_cols = [ 'Education Status', 'Special Education Services',
    'Employment Status', 'Number Of Hours Worked Each Week' ]


# In[199]:


#Ensemble troubles neuro
neurodev_cols =['Intellectual Disability', 'Autism Spectrum',
    'Other Developmental Disability', 'Neurological Condition',
    'Speech Impairment', 'Hearing Impairment', 'Visual Impairment',
    'Mobility Impairment Disorder', 'Traumatic Brain Injury']


# In[200]:


#ensemble addictions
addiction_cols =['Alcohol Related Disorder', 'Drug Substance Disorder',
    'Opioid Related Disorder' ,'Cannabis Medicinal Use', 'Smokes']


# In[201]:


#ensemble pathologie
chronic_disease_cols = [
    'Obesity', 'Diabetes', 'Cancer'
]


# In[202]:


#statut judiciaire
justice_cols = ['Criminal Justice Status']


# In[203]:


#Ensemble sociodémographique


# In[204]:


for col in socio_cols:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=col, hue='Mental Illness', data=df, palette='bright')
    plt.title(f'{col} vs Mental Illness')
    plt.xticks(rotation=30, ha='right')  # éviter l'écrasement du texte
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Mental Illness')
    plt.tight_layout()
    plt.show()


# In[205]:


#Ensemble Situation (etude/travail)


# In[206]:


for col in stuwork_cols:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=col, hue='Mental Illness', data=df, palette='bright')
    plt.title(f'{col} vs Mental Illness')
    plt.xticks(rotation=30, ha='right')  # éviter l'écrasement du texte
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Mental Illness')
    plt.tight_layout()
    plt.show()


# In[207]:


for col in neurodev_cols :
    plt.figure(figsize=(8, 5))
    sns.countplot(x=col, hue='Mental Illness', data=df, palette='bright')
    plt.title(f'{col} vs Mental Illness')
    plt.xticks(rotation=30, ha='right')  # éviter l'écrasement du texte
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Mental Illness')
    plt.tight_layout()
    plt.show()


# In[208]:


for col in addiction_cols :
    plt.figure(figsize=(8, 5))
    sns.countplot(x=col, hue='Mental Illness', data=df, palette='bright')
    plt.title(f'{col} vs Mental Illness')
    plt.xticks(rotation=30, ha='right')  # éviter l'écrasement du texte
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Mental Illness')
    plt.tight_layout()
    plt.show()


# In[209]:


for col in chronic_disease_cols :
    plt.figure(figsize=(8, 5))
    sns.countplot(x=col, hue='Mental Illness', data=df, palette='bright')
    plt.title(f'{col} vs Mental Illness')
    plt.xticks(rotation=30, ha='right')  # éviter l'écrasement du texte
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Mental Illness')
    plt.tight_layout()
    plt.show()


# In[210]:


for col in justice_cols :
    plt.figure(figsize=(8, 5))
    sns.countplot(x=col, hue='Mental Illness', data=df, palette='bright')
    plt.title(f'{col} vs Mental Illness')
    plt.xticks(rotation=30, ha='right')  # éviter l'écrasement du texte
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Mental Illness')
    plt.tight_layout()
    plt.show()


# # III- HANDLING MISSING VALUES

# In[211]:


#Eviter les bug
df.replace(['NAN', 'NaN', 'nan', 'None', 'NONE', '[nan]'], np.nan, inplace=True)


# STEP 1: Simple imputation by mode (very few NaNs)

# In[212]:


cols_faible_nan = ['Sex', 'Age Group', 'Preferred Language']
for col in cols_faible_nan:
    mode_val = df[col].mode(dropna=True)[0]
    df[col] = df[col].fillna(mode_val)
    print(f"{col} imputé par mode : {mode_val}")


# STEP 2: Logical grouping imputation function
# 

# In[213]:


def imputer_par_groupe(df, col, group_cols):
    df_copy = df.copy()
    # Créer une table avec les valeurs les plus fréquentes par groupe
    group_df = df_copy.dropna(subset=[col]).groupby(group_cols)[col].agg(lambda x: x.value_counts().index[0]).reset_index()
    group_df.rename(columns={col: 'valeur_majoritaire'}, inplace=True)

    # Joindre pour récupérer la valeur imputée
    df_copy = df_copy.merge(group_df, on=group_cols, how='left')

    # Remplacer les NaN avec la valeur du groupe
    mask = df_copy[col].isna() & df_copy['valeur_majoritaire'].notna()
    df_copy.loc[mask, col] = df_copy.loc[mask, 'valeur_majoritaire']

    # Nettoyage
    df_copy.drop(columns='valeur_majoritaire', inplace=True)
    print(f"{col} imputé par groupe {group_cols} — {mask.sum()} valeurs remplies, reste {df_copy[col].isna().sum()} NaN")
    return df_copy


# STEP 3: Columns imputed by logical group

# In[214]:


grouped_imputations = {
    'Sexual Orientation': ['Age Group', 'Sex', 'Region Served'],
    'Employment Status': ['Education Status', 'Age Group'],
    'Obesity': ['Age Group', 'Sex', 'Diabetes'],
    'Transgender': ['Sex'],
    'Race': ['Region Served', 'Hispanic Ethnicity'],
    'Cannabis Medicinal Use': ['Age Group', 'Alcohol Related Disorder'],
    'Religious Preference': ['Age Group', 'Sex'],
    'Living Situation': ['Region Served'],
    'Household Composition': ['Living Situation', 'Age Group'],
}

for col, grp in grouped_imputations.items():
    df = imputer_par_groupe(df, col, grp)


# STEP 4: Smart Cascade Imputation

# In[215]:


def imputation_en_cascade(df, col, groupings):
    df_copy = df.copy()
    total_filled = 0
    for group_cols in groupings:
        table = df_copy.dropna(subset=[col]).groupby(group_cols)[col].agg(lambda x: x.value_counts().index[0]).reset_index()
        table.rename(columns={col: 'valeur_majoritaire'}, inplace=True)
        df_copy = df_copy.merge(table, on=group_cols, how='left')
        mask = df_copy[col].isna() & df_copy['valeur_majoritaire'].notna()
        df_copy.loc[mask, col] = df_copy.loc[mask, 'valeur_majoritaire']
        filled_now = mask.sum()
        total_filled += filled_now
        df_copy.drop(columns='valeur_majoritaire', inplace=True)
        print(f"{col} imputé par groupe {group_cols} — {filled_now} valeurs remplies, reste {df_copy[col].isna().sum()} NaN")
        if df_copy[col].isna().sum() == 0:
            break
    return df_copy


# STEP 5: Columns to be processed by expanded cascade

# In[216]:


cascading_imputations = {
    'Employment Status': [['Education Status', 'Age Group'], ['Age Group']],
    'Obesity': [['Age Group', 'Sex', 'Diabetes'], ['Age Group', 'Sex'], ['Age Group']],
    'Race': [['Region Served', 'Hispanic Ethnicity'], ['Region Served']],
    'Cannabis Medicinal Use': [['Age Group', 'Alcohol Related Disorder'], ['Age Group']]
}

for col, cascades in cascading_imputations.items():
    df = imputation_en_cascade(df, col, cascades)


# In[217]:


def imputation_en_cascade(df, col, groupings):
    df_copy = df.copy()
    total_filled = 0
    for group_cols in groupings:
        grouped = df_copy[df_copy[col].notna()].groupby(group_cols)[col].agg(pd.Series.mode).reset_index()
        grouped.rename(columns={col: 'valeur_majoritaire'}, inplace=True)
        df_copy = df_copy.merge(grouped, on=group_cols, how='left')
        mask = df_copy[col].isna() & df_copy['valeur_majoritaire'].notna()
        df_copy.loc[mask, col] = df_copy.loc[mask, 'valeur_majoritaire']
        filled_now = mask.sum()
        total_filled += filled_now
        df_copy.drop(columns='valeur_majoritaire', inplace=True)
        print(f"{col} imputé par groupe {group_cols} — {filled_now} valeurs remplies, reste {df_copy[col].isna().sum()} NaN")
        if df_copy[col].isna().sum() == 0:
            break
    return df_copy


# In[218]:


cascading_imputations = {
    'Hispanic Ethnicity': [['Race', 'Region Served'], ['Region Served']],
    'Education Status': [['Age Group', 'Employment Status'], ['Age Group']],
    'Veteran Status': [['Age Group', 'Sex'], ['Age Group']],
    'Special Education Services': [['Education Status', 'Age Group'], ['Age Group']],
    'Number Of Hours Worked Each Week': [['Employment Status'], ['Age Group']],
    'Intellectual Disability': [['Age Group', 'Sex']],
    'Autism Spectrum': [['Age Group', 'Sex']],
    'Other Developmental Disability': [['Age Group', 'Sex']],
    'Alcohol Related Disorder': [['Age Group', 'Sex']],
    'Drug Substance Disorder': [['Age Group', 'Sex']],
    'Opioid Related Disorder': [['Age Group', 'Sex']],
    'Mobility Impairment Disorder': [['Age Group', 'Sex']],
    'Hearing Impairment': [['Age Group', 'Sex']],
    'Visual Impairment': [['Age Group', 'Sex']],
    'Speech Impairment': [['Age Group', 'Sex']],
    'Hyperlipidemia': [['Age Group', 'Sex']],
    'High Blood Pressure': [['Age Group', 'Sex']],
    'Diabetes': [['Age Group', 'Sex']],
    'Heart Attack': [['Age Group', 'Sex']],
    'Stroke': [['Age Group', 'Sex']],
    'Other Cardiac': [['Age Group', 'Sex']],
    'Pulmonary Asthma': [['Age Group', 'Sex']],
    'Alzheimer or Dementia': [['Age Group', 'Sex']],
    'Kidney Disease': [['Age Group', 'Sex']],
    'Liver Disease': [['Age Group', 'Sex']],
    'Endocrine Condition': [['Age Group', 'Sex']],
    'Neurological Condition': [['Age Group', 'Sex']],
    'Traumatic Brain Injury': [['Age Group', 'Sex']],
    'Joint Disease': [['Age Group', 'Sex']],
    'Cancer': [['Age Group', 'Sex']],
    'Other Chronic Med Condition': [['Age Group', 'Sex']],
    'No Chronic Med Condition': [['Age Group', 'Sex']],
    'Cannabis Recreational Use': [['Age Group', 'Drug Substance Disorder'], ['Age Group']],
    'Smokes': [['Age Group', 'Drug Substance Disorder'], ['Age Group']],
    'Received Smoking Medication': [['Age Group', 'Smokes']],
    'Received Smoking Counseling': [['Age Group', 'Smokes']],
    'Alcohol 12m Service': [['Age Group', 'Alcohol Related Disorder']],
    'Opioid 12m Service': [['Age Group', 'Opioid Related Disorder']],
    'Drug/Substance 12m Service': [['Age Group', 'Drug Substance Disorder']],
    'SSI Cash Assistance': [['Age Group', 'Employment Status']],
    'SSDI Cash Assistance': [['Age Group', 'Employment Status']],
    'Veterans Disability Benefits': [['Age Group', 'Veteran Status']],
    'Veterans Cash Assistance': [['Age Group', 'Veteran Status']],
    'Public Assistance Cash Program': [['Age Group', 'Employment Status']],
    'Other Cash Benefits': [['Age Group', 'Employment Status']],
    'Medicaid and Medicare Insurance': [['Age Group', 'Employment Status']],
    'No Insurance': [['Age Group', 'Employment Status']],
    'Medicaid Insurance': [['Age Group', 'Employment Status']],
    'Medicaid Managed Insurance': [['Age Group', 'Employment Status']],
    'Medicare Insurance': [['Age Group', 'Employment Status']],
    'Private Insurance': [['Age Group', 'Employment Status']],
    'Child Health Plus Insurance': [['Age Group', 'Employment Status']],
    'Other Insurance': [['Age Group', 'Employment Status']],
    'Criminal Justice Status': [['Age Group', 'Sex']]
}

# Application des imputations
for col, cascades in cascading_imputations.items():
    df = imputation_en_cascade(df, col, cascades)


# STEP 6:Fix columns logicaly

# In[219]:


#Addictions for Children
for col in addiction_cols :
    df.loc[(df['Age Group'] == 'CHILD') & (df[col].isna()), col] = 'NO'
#Work
df.loc[(df['Age Group'] == 'CHILD') &
(df['Employment Status'].isna()), 'Employment Status'] = 'NOT APPLICABLE'
df.loc[(df['Age Group'] == 'CHILD') &
(df['Number Of Hours Worked Each Week'].isna()), 'Number Of Hours Worked Each Week'] = 'NOT APPLICABLE'

#Social help
cols_aides = [
    'SSI Cash Assistance', 'SSDI Cash Assistance', 'Veterans Disability Benefits',
    'Veterans Cash Assistance', 'Public Assistance Cash Program', 'Other Cash Benefits'
]
for col in cols_aides:
    df.loc[(df['Age Group'] == 'CHILD') &
    (df[col].isna()), col] = 'NO'

#Assurances
df.loc[(df['Age Group'] == 'CHILD') &
(df['Child Health Plus Insurance'].isna()), 'Child Health Plus Insurance'] = 'YES'
assurances = [
    'Medicaid and Medicare Insurance', 'No Insurance', 'Medicaid Insurance',
    'Medicaid Managed Insurance', 'Medicare Insurance', 'Private Insurance', 'Other Insurance'
]

for col in assurances:
    df.loc[(df['Age Group'] == 'CHILD') &
     (df[col].isna()), col] = 'NO'

#identity

df.loc[(df['Age Group'] == 'CHILD') &
  (df['Transgender'].isna()), 'Transgender'] = 'NO, NOT TRANSGENDER'

df.loc[(df['Age Group'] == 'CHILD') &
  (df['Sexual Orientation'].isna()), 'Sexual Orientation'] = 'OTHER'

df.loc[(df['Age Group'] == 'CHILD') &
  (df['Religious Preference'].isna()), 'Religious Preference'] = 'I DO NOT HAVE A FORMAL RELIGION, NOR AM I A SPIRITUAL PERSON'


# In[220]:


for col in df.select_dtypes(include='bool').columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(False, inplace=True)
        print(f"'{col}' (bool) filled with False")


# In[221]:


#use other columns

df['Serious Mental Illness'] = df['Serious Mental Illness'].astype(str).str.upper().str.strip()
df['Principal Diagnosis Class'] = df['Principal Diagnosis Class'].astype(str).str.upper().str.strip()
df['Additional Diagnosis Class'] = df['Additional Diagnosis Class'].astype(str).str.upper().str.strip()

# Imputer par 1
df.loc[
    df['Mental Illness'].isna() & (
        (df['Principal Diagnosis Class'] == 'MENTAL ILLNESS') |
        (df['Additional Diagnosis Class'] == 'MENTAL ILLNESS') |
        (df['Serious Mental Illness'] == 'YES')
    ),
    'Mental Illness'
] = 1

# Imputer par NO
df.loc[
    df['Mental Illness'].isna() & (
        df['Principal Diagnosis Class'].str.startswith('NOT MI') |
        df['Additional Diagnosis Class'].str.startswith('NOT MI') |
        (df['Serious Mental Illness'] == 'NO')
    ),
    'Mental Illness'
] = 0 

# Pour les derniers cas ambigus, on choisit par défaut NO
df['Mental Illness'] = df['Mental Illness'].fillna(0)


# In[222]:


print(df['Mental Illness'].value_counts(dropna=False))


# In[223]:


#Drop 'Prefered Langage' Columns
if 'Preferred Language' in df.columns:
    df.drop(columns=['Preferred Language'], inplace=True)
    print("Preferred Language Deleted")


# In[224]:


# List of columns affected by incorrectly encoded multiple values (e.g. '[NO, YES]')
colonnes_multiples = [
    'SSI Cash Assistance', 'SSDI Cash Assistance', 'Veterans Disability Benefits',
    'Veterans Cash Assistance', 'Public Assistance Cash Program', 'Other Cash Benefits'
]

# Cleaning function
for col in colonnes_multiples:
    df[col] = df[col].apply(lambda x: x if isinstance(x, str) and x in ['YES', 'NO'] else np.nan)
    print(f"{col} — Valeurs corrigées")

# Simple imputation by mode 
for col in colonnes_multiples:
    mode_val = df[col].mode(dropna=True)[0]
    df[col] = df[col].fillna(mode_val)  # Pas de inplace ici
    print(f"{col} rempli par le mode : {mode_val}")


# In[225]:


#Avoid bugs
df.replace(['NAN', 'NaN', 'nan', 'None', 'NONE', '[nan]'], np.nan, inplace=True)


# In[226]:


# Final alignment of consistency between the 3 diagnostic columns

# If Primary = Mental Illness or Additional = Mental Illness → Mental = 1
df.loc[
    (df['Mental Illness'].isna()) &
    (
        (df['Principal Diagnosis Class'] == 'MENTAL ILLNESS') |
        (df['Additional Diagnosis Class'] == 'MENTAL ILLNESS')
    ),
    'Mental Illness'
] = 1

# If Principal starts with NOT MI or Additional starts with NOT MI → Mental = 0
df.loc[
    (df['Mental Illness'].isna()) &
    (
        df['Principal Diagnosis Class'].str.startswith('NOT MI', na=False) |
        df['Additional Diagnosis Class'].str.startswith('NOT MI', na=False)
    ),
    'Mental Illness'
] = 0

# If Mental = 1 and Serious is NaN → Serious = YES
df.loc[(df['Mental Illness'] == 1) & (df['Serious Mental Illness'].isna()), 'Serious Mental Illness'] = 'YES'
# Si Mental = 0 et Serious est NaN → Serious = NO
df.loc[(df['Mental Illness'] == 0) & (df['Serious Mental Illness'].isna()), 'Serious Mental Illness'] = 'NO'

# If Mental = 1 and Principal is NaN → Principal = MENTAL ILLNESS
df.loc[(df['Mental Illness'] == 1) & (df['Principal Diagnosis Class'].isna()), 'Principal Diagnosis Class'] = 'MENTAL ILLNESS'
# If Mental = 0 and Principal is NaN → Principal = NOT MI - OTHER
df.loc[(df['Mental Illness'] == 0) & (df['Principal Diagnosis Class'].isna()), 'Principal Diagnosis Class'] = 'NOT MI - OTHER'

# If Mental = 1 and Additional is NaN → Additional = MENTAL ILLNESS
df.loc[(df['Mental Illness'] == 1) & (df['Additional Diagnosis Class'].isna()), 'Additional Diagnosis Class'] = 'MENTAL ILLNESS'
# If Mental = 0 and Additional is NaN → Additional = NO ADDITIONAL DIAGNOSIS
df.loc[(df['Mental Illness'] == 0) & (df['Additional Diagnosis Class'].isna()), 'Additional Diagnosis Class'] = 'NO ADDITIONAL DIAGNOSIS'


# In[227]:


#Check missing values
print(df.isna().sum())


# In[228]:


print(df['Mental Illness'].value_counts(dropna=False))


# In[229]:


for col in df.columns:
    print(f"\n{col}:\n", df[col].value_counts(dropna=False))


# # Save Cleaned Dataset

# In[230]:


df_cleaned = df


# In[231]:


df_cleaned.shape


# In[232]:


#arjun path
df_cleaned.to_csv("D:/dstiprojects/Project 2/Project 2/data_clean.csv", sep=";", index=False)
df_cleaned.to_excel("D:/dstiprojects/Project 2/Project 2/data_clean.xlsx", index=False)


# In[233]:


df_cleaned


# # IV - FEATURE ENGINEERING

# One hot encoding
# - mental ilness a déjà été encoder précédemment.
# - preferedlanguage ( la colonne n'existe plus )
# 

# In[234]:


import pandas as pd

one_hot = [
    "Region Served",
    "Sex",
    "Hispanic Ethnicity",
    "Race",
    "Transgender",
    "Special Education Services",
    "Household Composition",
    "Sexual Orientation",
    "Religious Preference",
    "Additional Diagnosis Class",
    "Principal Diagnosis Class",
    "Program Category",
]

# One-Hot Encoding
df_OneHot = pd.get_dummies(df_cleaned, columns=one_hot, drop_first=True)
df_OneHot


# In[235]:


print(df_OneHot.isna().sum())


# In[236]:


print(df_OneHot.columns)


# Ordinal encoding
# 
# The columns to be encoded in ordinal: Age group, living situation, special, education, number of hours worked each week, education status, employment status.
# 
# For education status: there are "other" and "NO FORMAL EDUCATION" that we try to understand in order to classify them.

# In[237]:


# Filter rows where Education Status is 'OTHER'

df_other = df[df["Education Status"] == "OTHER"]

# List of columns to view
cols_to_plot = ["Age Group", "Sex", "Region Served", "Employment Status", "Special Education Services"]

# Create a subplot for each column
plt.figure(figsize=(16, 14))
for i, col in enumerate(cols_to_plot, 1):
    plt.subplot(3, 2, i)
    sns.countplot(data=df_other, x=col, order=df_other[col].value_counts().index, palette="Set2")
    plt.title(f"Distribution de {col} pour Education Status = 'OTHER'")
    plt.xticks(rotation=45)
    plt.tight_layout()
plt.show()



# people who answered "other" in "education status" are mostly adults, not working, living in the NY region for whom special education services are not applicable.


# In[238]:


import matplotlib.pyplot as plt
import seaborn as sns

# Filter only people "NOT IN LABOR FORCE: UNEMPLOYED AND NOT LOOKING FOR WORK"
df_unemployed = df[
    df["Employment Status"].str.upper() == "NOT IN LABOR FORCE:UNEMPLOYED AND NOT LOOKING FOR WORK"
]

# Create a bar chart with Seaborn
plt.figure(figsize=(12, 6))
sns.countplot(
    data=df_unemployed,
    x="Education Status",
    order=df_unemployed["Education Status"].value_counts().index,
    palette="Set3"
)

# Customize the chart
plt.title("Nombre de personnes 'NOT IN LABOR FORCE' par niveau d'éducation", fontsize=14)
plt.xlabel("Niveau d'éducation")
plt.ylabel("Nombre de personnes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# for comparison, non-active people have a middle school to high school education level. so other will be just before this value


# In[239]:


# Filter rows where Education Status is 'OTHER'
df_other = df[df["Education Status"] == "NO FORMAL EDUCATION"]

# List of columns to view
cols_to_plot = ["Age Group", "Sex", "Region Served", "Employment Status", "Special Education Services"]

# Create a subplot for each column
plt.figure(figsize=(16, 14))
for i, col in enumerate(cols_to_plot, 1):
    plt.subplot(3, 2, i)
    sns.countplot(data=df_other, x=col, order=df_other[col].value_counts().index, palette="Set2")
    plt.title(f"Distribution de {col} pour Education Status = 'NO FORMAL EDUCATION'")
    plt.xticks(rotation=45)
    plt.tight_layout()
plt.show()


# In[240]:


import matplotlib.pyplot as plt
import seaborn as sns

# Filter only people with "NO FORMAL EDUCATION"
df_no_education = df[
    df["Education Status"].str.upper() == "NO FORMAL EDUCATION"
]

# Create a bar chart with Seaborn
plt.figure(figsize=(12, 6))
sns.countplot(
    data=df_no_education,
    x="Employment Status",
    order=df_no_education["Employment Status"].value_counts().index,
    palette="Set2"
)

# Customize the chart
plt.title("Distribution of Employment Status among people without formal education", fontsize=14)
plt.xlabel("Employment Status")
plt.ylabel("Nombre de personnes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[241]:


# Copy the previous DataFrame
df_Ordinal1 = df_OneHot.copy()

# --- Ordinal Encoding ---
ordinal_cols = {
    "Living Situation": [
        "INSTITUTIONAL SETTING",  # less autonomous
        "OTHER LIVING SITUATION",  # unknown / temporary
        "PRIVATE RESIDENCE"   # stable and autonomous
    ],
    "Education Status": [
        "NO FORMAL EDUCATION", #jamais scolarisé
        "PRE-K TO FIFTH GRADE",# primaire
        "OTHER", # inconnu / atypique => regarder graphique précédent
        "MIDDLE SCHOOL TO HIGH SCHOOL", # secondaire
        "SOME COLLEGE", # post secondaire
        "COLLEGE OR GRADUATE DEGREE",  # diplôme universitaire

    ],
    "Employment Status": [
        "NOT IN LABOR FORCE:UNEMPLOYED AND NOT LOOKING FOR WORK",# non actif
        "UNEMPLOYED, LOOKING FOR WORK",# non employé et cherche a être actif
        "NON-PAID/VOLUNTEER", #actif non payé
        "EMPLOYED" #actif payé
    ],
    "Age Group": [
        "CHILD",
        "ADULT"
    ],
    "Number Of Hours Worked Each Week": [
        "NOT APPLICABLE", #0
        "01-14 HOURS",
        "15-34 HOURS",
        "35 HOURS OR MORE"
    ]
}

# Apply ordinal encoding in the new DataFrame
for col, categories in ordinal_cols.items():
    mapping = {cat: i for i, cat in enumerate(categories)}
    df_Ordinal1[col] = df_Ordinal1[col].map(mapping)

df_Ordinal1


# In[242]:


print(df_Ordinal1.isna().sum())


# In[243]:


df_Ordinal1["Education Status"].unique()


# In[244]:


print(df_Ordinal1["Medicaid Managed Insurance"].unique())


# In[245]:


#we group the "not applicable" with "no" in the "Medicaid Managed Insurance" column

df_Ordinal1["Medicaid Managed Insurance"] = df_Ordinal1["Medicaid Managed Insurance"].replace("NOT APPLICABLE", "NO")


# In[246]:


#convert TRUE FALSE to 0 and 1
df_TrueFalse = df_Ordinal1.copy() 

for col in df_TrueFalse.columns:
    col_str = df_TrueFalse[col].astype(str).str.strip().str.lower()
    unique_vals = set(col_str.dropna().unique())
    if unique_vals.issubset({"true", "false"}):
        df_TrueFalse[col] = col_str.map({"true": 1, "false": 0})

df_TrueFalse


# In[247]:


#convert yes no to 0 and 1

dfYesNo = df_TrueFalse.copy() 

for col in dfYesNo.columns:
    col_str = dfYesNo[col].astype(str).str.strip().str.upper()
    unique_vals = set(col_str.dropna().unique())
    if unique_vals.issubset({"YES", "NO"}):
        dfYesNo[col] = col_str.map({"YES": 1, "NO": 0})

dfYesNo


# In[248]:


# View columns in string type (object or string)
colonnes_string = dfYesNo.select_dtypes(include=["object", "string"]).columns
print(colonnes_string)


# In[249]:


print(df['Mental Illness'].value_counts())


# Matrice de correlation

# In[250]:


correlation_matrix = dfYesNo.corr(numeric_only=True)


# In[251]:


#arjun Path
correlation_matrix.to_excel("D:/dstiprojects/Project 2/Project 2/correlation_matrix.xlsx")


# In[252]:


# Option to display up to 100 lines in the console
pd.set_option('display.max_rows', 100)

# Calculation of the correlation matrix on all numeric columns
correlation_matrix = dfYesNo.corr()
# Extracting correlations with the 'Mental Illness' column
correlations_with_mental = correlation_matrix["Mental Illness"]

# Sort in descending order
correlations_with_mental_sorted = correlations_with_mental.sort_values(ascending=False)

# Full display
print(correlations_with_mental_sorted)


# Drop column 
# => We keep all the column

# # V- MACHINE LEARNING PHASE

# # Regression logistique

# In[253]:


from sklearn.model_selection import train_test_split


# In[254]:


df_train, df_test = train_test_split(dfYesNo,test_size = 0.2) # 20% go into testing and 80% on training


# In[255]:


len(dfYesNo) #the size of the data frame


# In[256]:


len(df_train) #the size of the data frame of training


# In[257]:


len(df_test) #the size of the data frame of training


# In[258]:


print(dfYesNo.Sex_MALE.mean())
print(df_train.Sex_MALE.mean())
print(df_test.Sex_MALE.mean())


# In[259]:


print(dfYesNo["Mental Illness"].mean())
print(df_train["Mental Illness"].mean())
print(df_test["Mental Illness"].mean())


# In[260]:


df_train.columns


# In[261]:


# Select the desired columns
X_train = df_train.loc[:, [
    'Age Group', 'Living Situation', 'Employment Status',
       'Number Of Hours Worked Each Week', 'Education Status', 'Intellectual Disability',
       'Other Developmental Disability', 'Alcohol Related Disorder',
       'Drug Substance Disorder', 'Mobility Impairment Disorder',
       'Hearing Impairment', 'Visual Impairment', 'Hyperlipidemia',
       'High Blood Pressure', 'Diabetes', 'Obesity', 'Heart Attack', 'Stroke',
       'Other Cardiac', 'Pulmonary Asthma', 'Alzheimer or Dementia',
       'Kidney Disease', 'Liver Disease', 'Endocrine Condition',
       'Neurological Condition', 'Traumatic Brain Injury', 'Joint Disease',
       'Cancer', 'Other Chronic Med Condition', 'No Chronic Med Condition',
       'Unknown Chronic Med Condition', 'Cannabis Recreational Use',
       'Cannabis Medicinal Use', 'Smokes', 'Received Smoking Medication',
       'Received Smoking Counseling', 'Serious Mental Illness',
       'Alcohol 12m Service', 'Drug/Substance 12m Service',
       'SSI Cash Assistance', 'SSDI Cash Assistance',
       'Veterans Cash Assistance', 'Public Assistance Cash Program',
       'Other Cash Benefits', 'Medicaid and Medicare Insurance',
       'No Insurance', 'Unknown Insurance Coverage', 'Medicaid Insurance',
       'Medicaid Managed Insurance', 'Medicare Insurance', 'Private Insurance',
       'Child Health Plus Insurance', 'Region Served_LONG ISLAND REGION',
       'Region Served_NEW YORK CITY REGION', 'Region Served_WESTERN REGION',
       'Sex_MALE', 'Hispanic Ethnicity_YES, HISPANIC/LATINO',
       'Race_WHITE ONLY', 'Transgender_YES, TRANSGENDER',
       'Special Education Services_NOT APPLICABLE',
       'Household Composition_LIVES ALONE',
       'Household Composition_NOT APPLICABLE',
       'Sexual Orientation_LESBIAN OR GAY', 'Sexual Orientation_OTHER',
       'Sexual Orientation_STRAIGHT OR HETEROSEXUAL',
       'Religious Preference_I CONSIDER MYSELF SPIRITUAL, BUT NOT RELIGIOUS',
       'Religious Preference_I DO NOT HAVE A FORMAL RELIGION, NOR AM I A SPIRITUAL PERSON',
       'Additional Diagnosis Class_NO ADDITIONAL DIAGNOSIS',
       'Additional Diagnosis Class_NOT MI - DEVELOPMENTAL DISORDERS',
       'Additional Diagnosis Class_NOT MI - ORGANIC MENTAL DISORDER',
       'Additional Diagnosis Class_SUBSTANCE-RELATED AND ADDICTIVE DISORDERS',
       'Principal Diagnosis Class_NOT MI - DEVELOPMENTAL DISORDERS',
       'Principal Diagnosis Class_NOT MI - ORGANIC MENTAL DISORDER',
       'Principal Diagnosis Class_NOT MI - OTHER',
       'Principal Diagnosis Class_SUBSTANCE-RELATED AND ADDICTIVE DISORDERS',
       'Program Category_INPATIENT', 'Program Category_OUTPATIENT',
       'Program Category_RESIDENTIAL', 'Program Category_SUPPORT'
]].values

# Target
y_train = df_train['Mental Illness'].values


# In[262]:


# get the values of the columns for the test data
X_test = df_test.loc[:, [
        'Age Group', 'Living Situation', 'Employment Status',
       'Number Of Hours Worked Each Week', 'Education Status', 'Intellectual Disability',
       'Other Developmental Disability', 'Alcohol Related Disorder',
       'Drug Substance Disorder', 'Mobility Impairment Disorder',
       'Hearing Impairment', 'Visual Impairment', 'Hyperlipidemia',
       'High Blood Pressure', 'Diabetes', 'Obesity', 'Heart Attack', 'Stroke',
       'Other Cardiac', 'Pulmonary Asthma', 'Alzheimer or Dementia',
       'Kidney Disease', 'Liver Disease', 'Endocrine Condition',
       'Neurological Condition', 'Traumatic Brain Injury', 'Joint Disease',
       'Cancer', 'Other Chronic Med Condition', 'No Chronic Med Condition',
       'Unknown Chronic Med Condition', 'Cannabis Recreational Use',
       'Cannabis Medicinal Use', 'Smokes', 'Received Smoking Medication',
       'Received Smoking Counseling', 'Serious Mental Illness',
       'Alcohol 12m Service', 'Drug/Substance 12m Service',
       'SSI Cash Assistance', 'SSDI Cash Assistance',
       'Veterans Cash Assistance', 'Public Assistance Cash Program',
       'Other Cash Benefits', 'Medicaid and Medicare Insurance',
       'No Insurance', 'Unknown Insurance Coverage', 'Medicaid Insurance',
       'Medicaid Managed Insurance', 'Medicare Insurance', 'Private Insurance',
       'Child Health Plus Insurance', 'Region Served_LONG ISLAND REGION',
       'Region Served_NEW YORK CITY REGION', 'Region Served_WESTERN REGION',
       'Sex_MALE', 'Hispanic Ethnicity_YES, HISPANIC/LATINO',
       'Race_WHITE ONLY', 'Transgender_YES, TRANSGENDER',
       'Special Education Services_NOT APPLICABLE',
       'Household Composition_LIVES ALONE',
       'Household Composition_NOT APPLICABLE',
       'Sexual Orientation_LESBIAN OR GAY', 'Sexual Orientation_OTHER',
       'Sexual Orientation_STRAIGHT OR HETEROSEXUAL',
       'Religious Preference_I CONSIDER MYSELF SPIRITUAL, BUT NOT RELIGIOUS',
       'Religious Preference_I DO NOT HAVE A FORMAL RELIGION, NOR AM I A SPIRITUAL PERSON',
       'Additional Diagnosis Class_NO ADDITIONAL DIAGNOSIS',
       'Additional Diagnosis Class_NOT MI - DEVELOPMENTAL DISORDERS',
       'Additional Diagnosis Class_NOT MI - ORGANIC MENTAL DISORDER',
       'Additional Diagnosis Class_SUBSTANCE-RELATED AND ADDICTIVE DISORDERS',
       'Principal Diagnosis Class_NOT MI - DEVELOPMENTAL DISORDERS',
       'Principal Diagnosis Class_NOT MI - ORGANIC MENTAL DISORDER',
       'Principal Diagnosis Class_NOT MI - OTHER',
       'Principal Diagnosis Class_SUBSTANCE-RELATED AND ADDICTIVE DISORDERS',
       'Program Category_INPATIENT', 'Program Category_OUTPATIENT',
       'Program Category_RESIDENTIAL', 'Program Category_SUPPORT']].values

#Target
y_test = df_test['Mental Illness'].values


# In[263]:


from sklearn.linear_model import LogisticRegression


# In[264]:


lr_model = LogisticRegression(random_state=300, max_iter=1000, class_weight='balanced') # Attribuer un poids plus élevé à la classe minoritaire lors de l'entraînement du modèle


# In[265]:


import joblib
joblib.dump(lr_model, 'logistic_regression_model.joblib')


# In[266]:


# here we train the model on the training data
lr_model.fit(X=X_train, y=y_train)


# In[267]:


y_test_predicted = lr_model.predict(X_test) #want model to give predictions


# In[268]:


y_test_predicted  # what the model predict


# In[269]:


y_test # the reality


# In[270]:


(y_test_predicted == y_test).sum()/len(y_test) # calculate the accuracy by hand


# In[271]:


from sklearn.metrics import confusion_matrix # or confusion matrix (true positif ...)


# In[272]:


cf = pd.DataFrame(
    columns=["y_test_0","y_test_1"],index=["y_pred_0","y_pred_1"]
)


# In[273]:


cf.loc[:,:] = confusion_matrix(y_true= y_test,y_pred= y_test_predicted)


# In[274]:


cf


# In[275]:


cf/len(y_test)


# In[276]:


from sklearn.metrics import recall_score, precision_score # precision score, accuracy score, f1 score


# In[277]:


recall_score(y_true=y_test, y_pred=y_test_predicted)


# In[278]:


from sklearn.metrics import classification_report


# In[279]:


report =classification_report(y_true=y_test, y_pred=y_test_predicted)


# In[280]:


print(report) # support mean how many row


# Logistic regression with SMOTE MODE (data imbalance management)

# In[281]:


#install the packages for data imbalance
#!pip install pandas scikit-learn imbalanced-learn
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


# In[282]:


from sklearn.model_selection import train_test_split
import pandas as pd


# In[283]:


# Split the dataset
df_train, df_test = train_test_split(dfYesNo, test_size=0.2)


# In[284]:


# Dataset size
len(dfYesNo)
len(df_train)
len(df_test)


# In[285]:


# Average of the variable Sex_MALE
print(dfYesNo.Sex_MALE.mean())
print(df_train.Sex_MALE.mean())
print(df_test.Sex_MALE.mean())


# In[286]:


# Average of the Mental Illness variable
print(dfYesNo["Mental Illness"].mean())
print(df_train["Mental Illness"].mean())
print(df_test["Mental Illness"].mean())


# In[287]:


df_train.columns


# In[288]:


# Feature selection for X_train
X_train = df_train.loc[:, [
    'Age Group', 'Living Situation', 'Employment Status',
    'Number Of Hours Worked Each Week', 'Education Status', 'Intellectual Disability',
    'Other Developmental Disability', 'Alcohol Related Disorder',
    'Drug Substance Disorder', 'Mobility Impairment Disorder',
    'Hearing Impairment', 'Visual Impairment', 'Hyperlipidemia',
    'High Blood Pressure', 'Diabetes', 'Obesity', 'Heart Attack', 'Stroke',
    'Other Cardiac', 'Pulmonary Asthma', 'Alzheimer or Dementia',
    'Kidney Disease', 'Liver Disease', 'Endocrine Condition',
    'Neurological Condition', 'Traumatic Brain Injury', 'Joint Disease',
    'Cancer', 'Other Chronic Med Condition', 'No Chronic Med Condition',
    'Unknown Chronic Med Condition', 'Cannabis Recreational Use',
    'Cannabis Medicinal Use', 'Smokes', 'Received Smoking Medication',
    'Received Smoking Counseling', 'Serious Mental Illness',
    'Alcohol 12m Service', 'Drug/Substance 12m Service',
    'SSI Cash Assistance', 'SSDI Cash Assistance',
    'Veterans Cash Assistance', 'Public Assistance Cash Program',
    'Other Cash Benefits', 'Medicaid and Medicare Insurance',
    'No Insurance', 'Unknown Insurance Coverage', 'Medicaid Insurance',
    'Medicaid Managed Insurance', 'Medicare Insurance', 'Private Insurance',
    'Child Health Plus Insurance', 'Region Served_LONG ISLAND REGION',
    'Region Served_NEW YORK CITY REGION', 'Region Served_WESTERN REGION',
    'Sex_MALE', 'Hispanic Ethnicity_YES, HISPANIC/LATINO',
    'Race_WHITE ONLY', 'Transgender_YES, TRANSGENDER',
    'Special Education Services_NOT APPLICABLE',
    'Household Composition_LIVES ALONE',
    'Household Composition_NOT APPLICABLE',
    'Sexual Orientation_LESBIAN OR GAY', 'Sexual Orientation_OTHER',
    'Sexual Orientation_STRAIGHT OR HETEROSEXUAL',
    'Religious Preference_I CONSIDER MYSELF SPIRITUAL, BUT NOT RELIGIOUS',
    'Religious Preference_I DO NOT HAVE A FORMAL RELIGION, NOR AM I A SPIRITUAL PERSON',
    'Additional Diagnosis Class_NO ADDITIONAL DIAGNOSIS',
    'Additional Diagnosis Class_NOT MI - DEVELOPMENTAL DISORDERS',
    'Additional Diagnosis Class_NOT MI - ORGANIC MENTAL DISORDER',
    'Additional Diagnosis Class_SUBSTANCE-RELATED AND ADDICTIVE DISORDERS',
    'Principal Diagnosis Class_NOT MI - DEVELOPMENTAL DISORDERS',
    'Principal Diagnosis Class_NOT MI - ORGANIC MENTAL DISORDER',
    'Principal Diagnosis Class_NOT MI - OTHER',
    'Principal Diagnosis Class_SUBSTANCE-RELATED AND ADDICTIVE DISORDERS',
    'Program Category_INPATIENT', 'Program Category_OUTPATIENT',
    'Program Category_RESIDENTIAL', 'Program Category_SUPPORT'
]].values

# Target
y_train = df_train['Mental Illness'].values


# In[289]:


# Test set
X_test = df_test.loc[:, [
    'Age Group', 'Living Situation', 'Employment Status',
    'Number Of Hours Worked Each Week', 'Education Status', 'Intellectual Disability',
    'Other Developmental Disability', 'Alcohol Related Disorder',
    'Drug Substance Disorder', 'Mobility Impairment Disorder',
    'Hearing Impairment', 'Visual Impairment', 'Hyperlipidemia',
    'High Blood Pressure', 'Diabetes', 'Obesity', 'Heart Attack', 'Stroke',
    'Other Cardiac', 'Pulmonary Asthma', 'Alzheimer or Dementia',
    'Kidney Disease', 'Liver Disease', 'Endocrine Condition',
    'Neurological Condition', 'Traumatic Brain Injury', 'Joint Disease',
    'Cancer', 'Other Chronic Med Condition', 'No Chronic Med Condition',
    'Unknown Chronic Med Condition', 'Cannabis Recreational Use',
    'Cannabis Medicinal Use', 'Smokes', 'Received Smoking Medication',
    'Received Smoking Counseling', 'Serious Mental Illness',
    'Alcohol 12m Service', 'Drug/Substance 12m Service',
    'SSI Cash Assistance', 'SSDI Cash Assistance',
    'Veterans Cash Assistance', 'Public Assistance Cash Program',
    'Other Cash Benefits', 'Medicaid and Medicare Insurance',
    'No Insurance', 'Unknown Insurance Coverage', 'Medicaid Insurance',
    'Medicaid Managed Insurance', 'Medicare Insurance', 'Private Insurance',
    'Child Health Plus Insurance', 'Region Served_LONG ISLAND REGION',
    'Region Served_NEW YORK CITY REGION', 'Region Served_WESTERN REGION',
    'Sex_MALE', 'Hispanic Ethnicity_YES, HISPANIC/LATINO',
    'Race_WHITE ONLY', 'Transgender_YES, TRANSGENDER',
    'Special Education Services_NOT APPLICABLE',
    'Household Composition_LIVES ALONE',
    'Household Composition_NOT APPLICABLE',
    'Sexual Orientation_LESBIAN OR GAY', 'Sexual Orientation_OTHER',
    'Sexual Orientation_STRAIGHT OR HETEROSEXUAL',
    'Religious Preference_I CONSIDER MYSELF SPIRITUAL, BUT NOT RELIGIOUS',
    'Religious Preference_I DO NOT HAVE A FORMAL RELIGION, NOR AM I A SPIRITUAL PERSON',
    'Additional Diagnosis Class_NO ADDITIONAL DIAGNOSIS',
    'Additional Diagnosis Class_NOT MI - DEVELOPMENTAL DISORDERS',
    'Additional Diagnosis Class_NOT MI - ORGANIC MENTAL DISORDER',
    'Additional Diagnosis Class_SUBSTANCE-RELATED AND ADDICTIVE DISORDERS',
    'Principal Diagnosis Class_NOT MI - DEVELOPMENTAL DISORDERS',
    'Principal Diagnosis Class_NOT MI - ORGANIC MENTAL DISORDER',
    'Principal Diagnosis Class_NOT MI - OTHER',
    'Principal Diagnosis Class_SUBSTANCE-RELATED AND ADDICTIVE DISORDERS',
    'Program Category_INPATIENT', 'Program Category_OUTPATIENT',
    'Program Category_RESIDENTIAL', 'Program Category_SUPPORT'
]].values

# target test
y_test = df_test['Mental Illness'].values


# In[290]:


# === ADDED SMOTE (with normalization) ===
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sm = SMOTE(sampling_strategy=0.30, random_state=42)
X_train_smote, y_train_smote = sm.fit_resample(X_train_scaled, y_train)

print("Avant SMOTE :", pd.Series(y_train).value_counts())
print("Après SMOTE :", pd.Series(y_train_smote).value_counts())


# In[291]:


# === Modèle ===
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(random_state=300, max_iter=1000, class_weight='balanced')  # pondération automatique


# In[292]:


import joblib
joblib.dump(lr_model, 'logistic_regression_model_SMOTE2.joblib')


# In[293]:


# Training
lr_model.fit(X=X_train_smote, y=y_train_smote)


# In[294]:


# Prédiction
y_test_predicted = lr_model.predict(X_test_scaled)


# In[295]:


# Predictions and metrics
y_test_predicted
y_test
(y_test_predicted == y_test).sum() / len(y_test)


# In[296]:


# Confusion matrix
from sklearn.metrics import confusion_matrix
cf = pd.DataFrame(
    columns=["y_test_0", "y_test_1"], index=["y_pred_0", "y_pred_1"]
)


# In[297]:


cf.loc[:, :] = confusion_matrix(y_true=y_test, y_pred=y_test_predicted)


# In[298]:


cf


# In[299]:


cf / len(y_test)


# In[300]:


# Scores
from sklearn.metrics import recall_score, precision_score
recall_score(y_true=y_test, y_pred=y_test_predicted)


# In[301]:


from sklearn.metrics import classification_report
report_LR_SOMTE = classification_report(y_true=y_test, y_pred=y_test_predicted)
print(report_LR_SOMTE)


# # Random Forest

# In[302]:


from sklearn.ensemble import RandomForestClassifier


# In[303]:


rf_model = RandomForestClassifier()


# In[304]:


rf_model.fit(X=X_train,y=y_train)


# In[305]:


y_test_predicted_rf = rf_model.predict(X_test)


# In[306]:


report_rf = classification_report(y_pred=y_test_predicted_rf,y_true=y_test)


# Random forest with SMOTE MODE (data imbalance management)

# In[307]:


from sklearn.model_selection import train_test_split
import pandas as pd


# In[308]:


# Split the data Set
df_train, df_test = train_test_split(dfYesNo, test_size=0.2)


# In[309]:


# Dataset size
len(dfYesNo)
len(df_train)
len(df_test)


# In[310]:


X_train = df_train.loc[:, [
    'Age Group', 'Living Situation', 'Employment Status',
    'Number Of Hours Worked Each Week', 'Education Status', 'Intellectual Disability',
    'Other Developmental Disability', 'Alcohol Related Disorder',
    'Drug Substance Disorder', 'Mobility Impairment Disorder',
    'Hearing Impairment', 'Visual Impairment', 'Hyperlipidemia',
    'High Blood Pressure', 'Diabetes', 'Obesity', 'Heart Attack', 'Stroke',
    'Other Cardiac', 'Pulmonary Asthma', 'Alzheimer or Dementia',
    'Kidney Disease', 'Liver Disease', 'Endocrine Condition',
    'Neurological Condition', 'Traumatic Brain Injury', 'Joint Disease',
    'Cancer', 'Other Chronic Med Condition', 'No Chronic Med Condition',
    'Unknown Chronic Med Condition', 'Cannabis Recreational Use',
    'Cannabis Medicinal Use', 'Smokes', 'Received Smoking Medication',
    'Received Smoking Counseling', 'Serious Mental Illness',
    'Alcohol 12m Service', 'Drug/Substance 12m Service',
    'SSI Cash Assistance', 'SSDI Cash Assistance',
    'Veterans Cash Assistance', 'Public Assistance Cash Program',
    'Other Cash Benefits', 'Medicaid and Medicare Insurance',
    'No Insurance', 'Unknown Insurance Coverage', 'Medicaid Insurance',
    'Medicaid Managed Insurance', 'Medicare Insurance', 'Private Insurance',
    'Child Health Plus Insurance', 'Region Served_LONG ISLAND REGION',
    'Region Served_NEW YORK CITY REGION', 'Region Served_WESTERN REGION',
    'Sex_MALE', 'Hispanic Ethnicity_YES, HISPANIC/LATINO',
    'Race_WHITE ONLY', 'Transgender_YES, TRANSGENDER',
    'Special Education Services_NOT APPLICABLE',
    'Household Composition_LIVES ALONE',
    'Household Composition_NOT APPLICABLE',
    'Sexual Orientation_LESBIAN OR GAY', 'Sexual Orientation_OTHER',
    'Sexual Orientation_STRAIGHT OR HETEROSEXUAL',
    'Religious Preference_I CONSIDER MYSELF SPIRITUAL, BUT NOT RELIGIOUS',
    'Religious Preference_I DO NOT HAVE A FORMAL RELIGION, NOR AM I A SPIRITUAL PERSON',
    'Additional Diagnosis Class_NO ADDITIONAL DIAGNOSIS',
    'Additional Diagnosis Class_NOT MI - DEVELOPMENTAL DISORDERS',
    'Additional Diagnosis Class_NOT MI - ORGANIC MENTAL DISORDER',
    'Additional Diagnosis Class_SUBSTANCE-RELATED AND ADDICTIVE DISORDERS',
    'Principal Diagnosis Class_NOT MI - DEVELOPMENTAL DISORDERS',
    'Principal Diagnosis Class_NOT MI - ORGANIC MENTAL DISORDER',
    'Principal Diagnosis Class_NOT MI - OTHER',
    'Principal Diagnosis Class_SUBSTANCE-RELATED AND ADDICTIVE DISORDERS',
    'Program Category_INPATIENT', 'Program Category_OUTPATIENT',
    'Program Category_RESIDENTIAL', 'Program Category_SUPPORT'
]].values

# Target
y_train = df_train['Mental Illness'].values


# In[311]:


# Test set
X_test = df_test.loc[:, [
    'Age Group', 'Living Situation', 'Employment Status',
    'Number Of Hours Worked Each Week', 'Education Status', 'Intellectual Disability',
    'Other Developmental Disability', 'Alcohol Related Disorder',
    'Drug Substance Disorder', 'Mobility Impairment Disorder',
    'Hearing Impairment', 'Visual Impairment', 'Hyperlipidemia',
    'High Blood Pressure', 'Diabetes', 'Obesity', 'Heart Attack', 'Stroke',
    'Other Cardiac', 'Pulmonary Asthma', 'Alzheimer or Dementia',
    'Kidney Disease', 'Liver Disease', 'Endocrine Condition',
    'Neurological Condition', 'Traumatic Brain Injury', 'Joint Disease',
    'Cancer', 'Other Chronic Med Condition', 'No Chronic Med Condition',
    'Unknown Chronic Med Condition', 'Cannabis Recreational Use',
    'Cannabis Medicinal Use', 'Smokes', 'Received Smoking Medication',
    'Received Smoking Counseling', 'Serious Mental Illness',
    'Alcohol 12m Service', 'Drug/Substance 12m Service',
    'SSI Cash Assistance', 'SSDI Cash Assistance',
    'Veterans Cash Assistance', 'Public Assistance Cash Program',
    'Other Cash Benefits', 'Medicaid and Medicare Insurance',
    'No Insurance', 'Unknown Insurance Coverage', 'Medicaid Insurance',
    'Medicaid Managed Insurance', 'Medicare Insurance', 'Private Insurance',
    'Child Health Plus Insurance', 'Region Served_LONG ISLAND REGION',
    'Region Served_NEW YORK CITY REGION', 'Region Served_WESTERN REGION',
    'Sex_MALE', 'Hispanic Ethnicity_YES, HISPANIC/LATINO',
    'Race_WHITE ONLY', 'Transgender_YES, TRANSGENDER',
    'Special Education Services_NOT APPLICABLE',
    'Household Composition_LIVES ALONE',
    'Household Composition_NOT APPLICABLE',
    'Sexual Orientation_LESBIAN OR GAY', 'Sexual Orientation_OTHER',
    'Sexual Orientation_STRAIGHT OR HETEROSEXUAL',
    'Religious Preference_I CONSIDER MYSELF SPIRITUAL, BUT NOT RELIGIOUS',
    'Religious Preference_I DO NOT HAVE A FORMAL RELIGION, NOR AM I A SPIRITUAL PERSON',
    'Additional Diagnosis Class_NO ADDITIONAL DIAGNOSIS',
    'Additional Diagnosis Class_NOT MI - DEVELOPMENTAL DISORDERS',
    'Additional Diagnosis Class_NOT MI - ORGANIC MENTAL DISORDER',
    'Additional Diagnosis Class_SUBSTANCE-RELATED AND ADDICTIVE DISORDERS',
    'Principal Diagnosis Class_NOT MI - DEVELOPMENTAL DISORDERS',
    'Principal Diagnosis Class_NOT MI - ORGANIC MENTAL DISORDER',
    'Principal Diagnosis Class_NOT MI - OTHER',
    'Principal Diagnosis Class_SUBSTANCE-RELATED AND ADDICTIVE DISORDERS',
    'Program Category_INPATIENT', 'Program Category_OUTPATIENT',
    'Program Category_RESIDENTIAL', 'Program Category_SUPPORT'
]].values

# target test
y_test = df_test['Mental Illness'].values


# In[312]:


# === ADDED SMOTE (with normalization) ===
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# In[313]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[314]:


# sampling_strategy = 0.11 → la classe minoritaire atteindra 11 % de la majoritaire
sm = SMOTE(sampling_strategy=0.11, random_state=42)
X_train_smote, y_train_smote = sm.fit_resample(X_train_scaled, y_train)


# In[315]:


print("Avant SMOTE :", pd.Series(y_train).value_counts())
print("Après SMOTE :", pd.Series(y_train_smote).value_counts())


# In[316]:


# === Modèle Random Forest ===
from sklearn.ensemble import RandomForestClassifier


# In[317]:


rf_model = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    n_estimators=100,
)


# In[318]:


import joblib
joblib.dump(rf_model, 'random_forest_model.joblib')


# In[319]:


# Entraînement
rf_model.fit(X_train_smote, y_train_smote)


# In[320]:


# Prédiction avec seuil personnalisé
y_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
y_test_predicted = (y_probs > 0.4).astype(int)  # on tester 0.4 ou 0.6 aussi


# In[321]:


# Prédictions et métriques
y_test_predicted
y_test
(y_test_predicted == y_test).sum() / len(y_test)


# In[322]:


# Matrice de confusion
from sklearn.metrics import confusion_matrix
cf = pd.DataFrame(
    columns=["y_test_0", "y_test_1"], index=["y_pred_0", "y_pred_1"]
)


# In[323]:


cf.loc[:, :] = confusion_matrix(y_true=y_test, y_pred=y_test_predicted)
print(cf)


# In[324]:


print(cf / len(y_test))


# In[325]:


# Scores
from sklearn.metrics import recall_score, precision_score, classification_report


# In[326]:


print("Recall (classe 0) :", recall_score(y_test, y_test_predicted, pos_label=0))
print("Précision (classe 0) :", precision_score(y_test, y_test_predicted, pos_label=0))


# In[327]:


report_RFSmote = classification_report(y_true=y_test, y_pred=y_test_predicted)
print(report_RFSmote)


# # Decision tree Classifer

# In[328]:


from sklearn.tree import DecisionTreeClassifier


# In[329]:


# Try a Decision Tree classifier
dt_model = DecisionTreeClassifier()


# In[330]:


dt_model.fit(X=X_train,y=y_train)


# In[331]:


y_test_predicted_dt = dt_model.predict(X_test)


# In[332]:


report_dt = classification_report(y_pred=y_test_predicted_dt,y_true=y_test)


# In[333]:


from sklearn.ensemble import RandomForestClassifier # random forest model


# # Comparaison between Regression logistic, Decision Tree classifer and RandomForest

# In[334]:


print("Report of logistic regression")

print(report)


# In[335]:


print("Report of logistic regression with Smote mode")

print(report_LR_SOMTE)


# In[336]:


print("Report of Random Forest model")
print(report_rf)


# In[337]:


print("Report of Random forest With SMOTE mode")

print(report_RFSmote)


# In[338]:


print("Report of Decision Tree classifier model")
print(report_dt)


# In[ ]:




