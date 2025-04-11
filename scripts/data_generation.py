# Import libraries
import numpy as np
import pandas as pd

# Generate n random numbers that add to one
def generate_weights(n,scaling=None): 
    w = np.random.uniform(0,1,n)
    if scaling is not None:
        w = w * scaling
    return w/np.sum(w)

def generate_age(n_samples:int=5000):
    age_df = pd.read_csv('data/emperical_data/CDC_age_data.csv',index_col='Single-Year Ages Code')
    age_df = age_df[age_df.index >= 18] # Don't include people less than 21 years old
    age_df = age_df/age_df.sum() # Normalize age distribution into probabilities
    return list(age_df.sample(n_samples,replace=True,weights='Population').index)

def generate_income(n_samples:int=5000):
    return (1 - np.random.power(3,n_samples)) * 526.2

def generate_education(n_samples:int=5000):
    # Import and Clean Data
    education_df = pd.read_csv('data/emperical_data/education.csv').T
    education_df.columns = ['education']
    education_df['education'] = education_df['education'].str.replace(',','')
    education_df = education_df.astype('int')

    # Normalize distribution and sample n from the distribution
    education_df = education_df/education_df.sum()
    education_df = education_df.reset_index(drop=True) # Convert education to likert scale
    return list(education_df.sample(n_samples,replace=True,weights='education').index)

def generate_demographics(n:int=5000):
    # Initialize an empty dataframe with 5000 rows
    demographics_df = pd.DataFrame(index=range(n))

    # Generate demographic data
    demographics_df['demographic_age'] = generate_age(n)
    demographics_df['demographic_income'] = generate_income(n)
    demographics_df['demographic_education'] = generate_education(n)
    demographics_df['demographic_unobs_grp'] = np.random.choice(['A','B','C','D','E'],n)
    return demographics_df

def generate_attitudes(df,demographic_cont_vars,demographic_cat_vars):
    """
    This method will generate a dataframe of vaccination attitudes based on demographic data.
    Arguments:
    ===========
    df: pd.Dataframe
        Denotes the demographic attributes to generate a dataframe on. Can have as many columns as needed
    demographic_cont_vars: List of Strings
        A list denoting the column names of continuous variables
    demographic_cat_vars: List of String
        A list denoting the column names of categorical variables
    """
    # Normalize continuous demographic variables
    this_df = df.copy(deep=True)
    this_df[demographic_cont_vars] = this_df[demographic_cont_vars]/this_df[demographic_cont_vars].std()
    this_df[demographic_cont_vars] = this_df[demographic_cont_vars]/this_df[demographic_cont_vars].max()

    # Dummy categorical demographic variables
    this_df = pd.get_dummies(this_df,columns=[demographic_cat_vars],drop_first=False)

    # Normalize demographic variables by z-score and maximum value
    this_df = this_df/this_df.std()
    this_df = this_df/this_df.max()
    this_df = this_df * 2 # Scale by 2 to increase weight

    # Generate attitudes
    attitudes = ['att_covid','att_vaccine','att_safety','att_unobserved']
    attitudes_df = pd.DataFrame(index=range(this_df.shape[0]))
    demographics = list(this_df.columns)

    for attitude in attitudes:
        attitudes_df[attitude] = np.random.uniform(1,5,this_df.shape[0]) + np.random.normal(0,0.5,this_df.shape[0]) # Intrinsic views plus random error
        demographic_weights = generate_weights(len(demographics),scaling=[1,1,1,1/4,1/4,1/4,1/4,1/4]) # Generate weight of each demographic's effect
        for i,demographic in enumerate(demographics): # Add influence of demographics
            attitudes_df[attitude] = attitudes_df[attitude] + demographic_weights[i] * this_df[demographic]

        # Handle values that are outside of max range (1-10)
        attitudes_df[attitude] = attitudes_df[attitude].clip(1,10)

        # Round to nearest integer
        attitudes_df[attitude] = np.round(attitudes_df[attitude]).astype('int')
    return attitudes_df