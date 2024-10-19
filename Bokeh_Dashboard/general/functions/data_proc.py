import pandas as pd
from typing import Any
from bsoid_umap_prelim.Bokeh_Dashboard.general.CONFIG_func.config_func import *
from bsoid_umap_prelim.Bokeh_Dashboard.general.functions.function_config import *


##############
# TEST CASES #
##############

def test_check_dataframe_index(df: pd.DataFrame, expected_index: List[str]) -> None:
    """
    Runs tests to verify that a DataFrame has the expected multi-index.

    Parameters:
        df (pd.DataFrame): The DataFrame to test.
        expected_index (List[str]): The list of expected index level names.
    """
    result = check_dataframe_index(df, expected_index)

    if result:
        print(f"Test passed: DataFrame is indexed by {expected_index}")
    else:
        print(f"Test failed: DataFrame is not indexed by {expected_index}")
        
def check_dataframe_index(df: pd.DataFrame, expected_index: List[str]) -> bool:
    """
    Checks if the DataFrame has a multi-index matching the expected index names.

    Parameters:
        df (pd.DataFrame): The DataFrame to check.
        expected_index (List[str]): The list of expected index level names.

    Returns:
        bool: True if the DataFrame's index matches the expected index, otherwise False.
    """
    # Check if the DataFrame has a multi-index
    if not isinstance(df.index, pd.MultiIndex):
        return False
    
    # Get the levels of the multi-index
    index_levels = df.index.names
    
    # Check if the index levels match the expected levels
    return index_levels == expected_index

def assert_type(data: pd.DataFrame):
    """
    Asserts that the provided data is a pandas DataFrame.
    """
    assert isinstance(data, pd.DataFrame), f"Data must be a pandas DataFrame, got {type(data)} instead."

def index_check(expected_index: List[str], data: pd.DataFrame):
    """
    Checks if the DataFrame has the expected multi-level index.
    """
    assert isinstance(data.index, pd.MultiIndex), "DataFrame must have a MultiIndex."
    
    # Get the actual index levels and compare with expected
    actual_index = list(data.index.names)
    assert actual_index == expected_index, (
        f"Expected index levels {expected_index}, but got {actual_index}."
    )

def data_check(expected_columns: List[str], data: pd.DataFrame):
    """
    Checks if the DataFrame has the expected column names.
    """
    actual_columns = list(data.columns)
    assert actual_columns == expected_columns, (
        f"Expected columns {expected_columns}, but got {actual_columns}."
    )

#################
# DATA CLEANING #
#################

def dataframe_clean(df: pd.DataFrame):
    pass

def data_cleaning(expected_columns, expected_index, data): # turn inputs into *args and **kwargs
    
    cleaned_dataframe = None
    
    
    ##############
    # RUN CHECKS #
    ##############
    
    data_check()
    
    index_check()
    
    assert_type()
    
    ###########################################
    # MAIN CASE, BREAK IF FAIL, CLEAN IF OKAY #
    ###########################################
    
    if True:
        
        ################################
        # RUN STANDARDIZATION PIPELINE #
        ################################
        
        dataframe_clean()
        
    return cleaned_dataframe

#########################################
# FEED CLEANED DATAFRAME INTO DASHBOARD #
#########################################

