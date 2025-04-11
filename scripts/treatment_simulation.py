# Importing Libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def assign_block_treatment(this_network):
    # Get network community structure
    treatment_assignment_df = this_network.node_community.copy(deep=True).to_frame()

    # Assign treatments to each community randomly and with equal probability
    community_to_treatment = np.random.choice([None,'reason','emotion'],size=this_network.n_communities) 

    # Assign treatment status to each unit within the communities
    treatment_assignment_df['treatment'] = community_to_treatment[treatment_assignment_df['community']]
    treatment_assignment_df = pd.get_dummies(treatment_assignment_df,columns=['treatment'])
    return treatment_assignment_df


def apply_treatment(attitudes_df,treatment_assignment_df,this_network):
    attitudes_df = attitudes_df.copy(deep=True)

    # Plot pre-treatment attitudes
    plt.title('Pre-treatment distribution of vaccination attitudes')
    plt.hist(attitudes_df,bins=3)
    plt.legend(attitudes_df.columns)
    plt.savefig('figures/pre_treat_att.png')
    plt.show()

    # Calculate treatment effect for each attitude
    for attitude in attitudes_df.columns:
        # Add base treatment effect
        att_values = (treatment_assignment_df['treatment_emotion'] | treatment_assignment_df['treatment_reason']).astype('int') * 3

        # Apply heterogenous treatment effects for each type of attitude, for each type of treatment
        att_values = att_values + treatment_assignment_df['treatment_emotion'].astype('int') * np.random.uniform(0,1,size=1)
        att_values = att_values + treatment_assignment_df['treatment_reason'].astype('int') * np.random.uniform(0,1,size=1)

        # Add random error term
        att_values = att_values + np.random.normal(0,0.5,5000)

        # Add treatment effect to dataframe
        attitudes_df[attitude] = attitudes_df[attitude] + att_values

        # Run network diffusion
        attitudes_df[attitude] = this_network.fj_diffusion(attitudes_df[attitude])
    
    # Round and Truncate Values
    attitudes_df = np.round(attitudes_df).astype('int')
    attitudes_df = attitudes_df.clip(1,10)

    # Plot post-treatment attitudes
    plt.title('Post-treatment distribution of vaccination attitudes')
    plt.hist(attitudes_df,bins=7)
    plt.legend(attitudes_df.columns)
    plt.savefig('figures/post_treat_att.png')
    plt.show()

    return attitudes_df