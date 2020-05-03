"""
TechMiner.transform
==================================================================================================


This module contains functions that can be applied to each element of a pandas.Series 
object using the map function. 


"""
import pandas as pd
import numpy as np

def nan2none(df):
    """Replace np.nan by None in a pandas.DataFrame.
    
    Args:
        df (pandas.DataFrame)
        
    Returns:
        pandas.DataFrame
    """
    
    for x in df.columns:
        df[x] = df[x].map(lambda x: None if isinstance(x, float) and np.isnan(x) else x)
        
    return df