import sys
import os
import shutil
import json
import pickle
import pandas as pd
import tqdm
import glob
from scipy.io import savemat, loadmat
# import pyreadstat
# import statsmodels.api as sm
# from savReaderWriter import SavReader

class directories:
    def __init__(self):
        """
        Initialize directories class.

        Attributes:
        - cwd (str): Current working directory.
        - parent_dir (str): Parent directory of the current working directory.
        - files (list): List of files in the current working directory.
        - home_dir (str): Home directory path.
        """
        self.cwd = os.getcwd()
        self.parent_dir = os.path.dirname(self.cwd)
        self.grandparent_dir = os.path.dirname(self.parent_dir)
        self.great_grandparent_dir = os.path.dirname(self.grandparent_dir)
        self.gg_gp_dir = os.path.dirname(self.great_grandparent_dir)
        self.files = os.listdir()
        self.home_dir = os.getenv('HOME')
        
    def list_files(self, folder_path):
        """
        List files in a specified directory.

        Parameters:
        - folder_path (str): Path to the directory.

        Returns:
        - list: List of files in the directory.
        """
        return os.listdir(folder_path)
    
    def count_items(self, folder_path, item_type='directories'):
        """
        Count the number of files or directories in a specified directory.

        Parameters:
        - folder_path (str): Path to the directory.
        - item_type (str): Type of item to count ('files' or 'directories').

        Returns:
        - int: Number of specified items in the directory.
        """
        if item_type not in ['files', 'directories']:
            raise ValueError("item_type must be either 'files' or 'directories'")
        
        items = os.listdir(folder_path)
        if item_type == 'files':
            count = sum(os.path.isfile(os.path.join(folder_path, item)) for item in items)
        elif item_type == 'directories':
            count = sum(os.path.isdir(os.path.join(folder_path, item)) for item in items)
        
        return count
        
    def join_path_with_keyword(self, *keywords):
        """
        Join a base path with keywords to form a complete path.

        If any joined path does not exist, create a new directory.

        Parameters:
        - keywords (str): Keywords to append to the base path.

        Returns:
        - str: Complete path formed by joining base_path and keywords.
        """
        current_path = self.cwd
        for keyword in keywords:
            current_path = os.path.join(current_path, keyword)
            if not os.path.exists(current_path):
                self.construct_dir(keyword)
        return current_path
    
    def find_closest_data_folder(self):
        """
        Find the closest "data" folder starting from the current working directory and moving up the directory tree.

        Returns:
        - str or None: Path to the closest "data" folder, or None if no such folder is found.
        """
        current_path = self.cwd
        while current_path != self.parent_dir:
            data_path = os.path.join(current_path, 'data')
            if os.path.exists(data_path):
                return data_path
            current_path = os.path.dirname(current_path)
        return None
    
    def find_closest_kw_folder_up(self, kw='data'):
        """
        Find the closest "data" folder starting from the current working directory and moving up the directory tree.

        Parameters:
        - kw (str): The keyword to search for in folder names.

        Returns:
        - str or None: Path to the closest "data" folder, or None if no such folder is found.
        """
        current_path = self.cwd
        while True:
            # Check if the current directory contains the target folder
            kw_path = os.path.join(current_path, kw)
            if os.path.exists(kw_path) and os.path.isdir(kw_path):
                return kw_path
            
            # Move up to the parent directory
            if current_path == self.parent_dir:
                break
            current_path = os.path.dirname(current_path)
            tqdm.tqdm.write(f'Searching for "{kw}" folder upwards...')
        
        return None
    
    def find_closest_kw_folder_down(self, kw='data'):
        """
        Find the closest folder containing the given keyword starting from the current working directory and moving down the directory tree.

        Parameters:
        - kw (str): The keyword to search for in folder names.

        Returns:
        - str or None: Path to the closest folder containing the keyword, or None if no such folder is found.
        """
        for root, dirs, files in os.walk(self.cwd):
            for dir_name in dirs:
                if kw in dir_name:
                    return os.path.join(root, dir_name)
                tqdm.tqdm.write(f'Searching for "{kw}" folder downwards...')
        
        return None

    
    def shift_dir(self, new_dir):
        """
        Change current working directory to the specified directory.

        Parameters:
        - new_dir (str): Path to the new directory.
        """
        os.chdir(new_dir)
        self.cwd = os.getcwd()
        self.files = os.listdir()
        
        
    #######
    # FIX #
    #######
    
    # def parent_shift(self, iterations):
    #     for i in range(1, iterations + 1):
    #         try:
    #             self.shift_dir(self.parent_dir)
    #             print(f"Successfully traversed to parent directory for iteration {i}.")
    #         except Exception as e:
    #             print(f"Failed to traverse to parent directory on iteration {i}: {e}")
    #             break
        
    def construct_dir(self, new_dir_name):
        """
        Create a new directory.

        Parameters:
        - new_dir_name (str): Name of the new directory to be created.
        """
        os.makedirs(new_dir_name)
    
    def give_file_info(self, filename):
        """
        Get information about a file.

        Parameters:
        - filename (str): Name of the file.

        Returns:
        - dict or None: Dictionary with file information including filename, size, last modified time, and permissions,
                        or None if the file does not exist.
        """
        file_path = os.path.join(self.cwd, filename)
        if os.path.exists(file_path):
            file_stat = os.stat(file_path)
            return {
                'filename' : filename,
                'size' : file_stat.st_size,
                'last_modified' : file_stat.st_mtime,
                'permissions' : oct(file_stat.st_mode)[-3:]
            }
        else:
            return None

    def search_files(self, pattern):
        """
        Search for files matching a pattern in the current directory and its subdirectories.

        Parameters:
        - pattern (str): Pattern to search for in filenames.

        Returns:
        - list: List of file paths matching the pattern.
        """
        matches = []
        for root, _, files in os.walk(self.cwd):
            for file in files:
                if pattern in file:
                    matches.append(os.path.join(root, file))
        return matches
    
    def move_file(self, filename, dest):
        """
        Move a file to a new destination.

        Parameters:
        - filename (str): Name of the file to move.
        - dest (str): Destination path to move the file.
        """
        file_path = os.path.join(self.cwd, filename)
        
        if not os.path.exists(file_path):
            print(f"Error: File {filename} does not exist.")
            return
        
        if not os.path.exists(dest):
            os.makedirs(dest)  # Create destination directory if it doesn't exist
        
        try:
            shutil.move(file_path, dest)
            self.files = os.listdir()  # Update the list of files in the current directory
            print(f"File {filename} moved to {dest} successfully.")
        except Exception as e:
            print(f"Error moving file {filename} to {dest}: {str(e)}")
            
    def del_file(self, filename):
        """
        Delete a file.

        Parameters:
        - filename (str): Name of the file to delete.
        """
        file_path = os.path.join(self.cwd, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            self.files = os.listdir()
            
    def save_json(self, data, filename):
        """
        Save data as JSON to a file.

        Parameters:
        - data (object): Data to save as JSON.
        - filename (str): Name of the JSON file to save.
        """
        file_path = os.path.join(self.cwd, filename)
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file)
            
    def load_json(self, filename):
        """
        Load JSON data from a file.

        Parameters:
        - filename (str): Name of the JSON file to load.

        Returns:
        - object or None: Loaded JSON data, or None if the file does not exist.
        """
        file_path = os.path.join(self.cwd, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r') as json_file:
                return json.load(json_file)
        else:
            return None

    def save_pickle(self, data, filename):
        """
        Save data as a pickle object to a file.

        Parameters:
        - data (object): Data to save as a pickle object.
        - filename (str): Name of the pickle file to save.
        """
        file_path = os.path.join(self.cwd, filename)
        with open(file_path, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)
    
    def load_pickle(self, filename):
        """
        Load data from a pickle file.

        Parameters:
        - filename (str): Name of the pickle file to load.

        Returns:
        - object or None: Loaded data from the pickle file, or None if the file does not exist.
        """
        file_path = os.path.join(self.cwd, filename)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as pickle_file:
                return pickle.load(pickle_file)
        else:
            return None
    
    def save_dict_of_dfs_pickle(self, dfs, filename):
        """
        Save a dictionary of pandas DataFrames to a pickle file.

        Parameters:
        - dfs (dict): Dictionary where keys are DataFrame names and values are DataFrames.
        - filename (str): Filepath to save the pickle file.
        """
        file_path = os.path.join(self.cwd, filename)
        with open(file_path, 'wb') as f:
            pickle.dump(dfs, f)
        print(f"DataFrames saved as pickle object to {filename} successfully.")

    def load_dict_of_dfs_pickle(self, filename):
        """
        Load a dictionary of pandas DataFrames from a pickle file.

        Parameters:
        - filename (str): Filepath to the pickle file.

        Returns:
        - dfs (dict): Dictionary where keys are DataFrame names and values are DataFrames.
        """
        file_path = os.path.join(self.cwd, filename)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                dfs = pickle.load(f)
            print(f"DataFrames loaded from pickle object {filename} successfully.")
            return dfs
        else:
            return None

    def save_dict_of_dfs_hdf5(self, dfs, filename):
        """
        Save a dictionary of pandas DataFrames to an HDF5 file.

        Parameters:
        - dfs (dict): Dictionary where keys are DataFrame names and values are DataFrames.
        - filename (str): Filepath to save the HDF5 file.
        """
        file_path = os.path.join(self.cwd, filename)
        with pd.HDFStore(file_path, mode='w') as store:
            for key, df in dfs.items():
                store[key] = df
        print(f"DataFrames saved to HDF5 file {filename} successfully.")

    def load_dict_of_dfs_hdf5(self, filename):
        """
        Load a dictionary of pandas DataFrames from an HDF5 file.

        Parameters:
        - filename (str): Filepath to the HDF5 file.

        Returns:
        - dfs (dict): Dictionary where keys are DataFrame names and values are DataFrames.
        """
        dfs = {}
        file_path = os.path.join(self.cwd, filename)
        with pd.HDFStore(file_path, mode='r') as store:
            for key in store.keys():
                dfs[key.strip('/')] = store[key]
        print(f"DataFrames loaded from HDF5 file {filename} successfully.")
        return dfs

    def save_dict_of_dfs_csv(self, dfs, folder):
        """
        Save each DataFrame in a dictionary of pandas DataFrames to CSV files.

        Parameters:
        - dfs (dict): Dictionary where keys are DataFrame names and values are DataFrames.
        - folder (str): Folder path to save the CSV files. Each CSV file will be named after the DataFrame key.
        """
        for key, df in dfs.items():
            df.to_csv(f'{folder}/{key}.csv', index=False)
        print(f"DataFrames saved as CSV files to {folder} successfully.")

    def load_dict_of_dfs_csv(self, folder):
        """
        Load a dictionary of pandas DataFrames from a folder containing CSV files.

        Parameters:
        - folder (str): Folder path containing CSV files.

        Returns:
        - dfs (dict): Dictionary where keys are DataFrame names and values are DataFrames.
        """
        dfs = {}
        for filename in os.listdir(folder):
            if filename.endswith('.csv'):
                key = filename.split('.')[0]
                dfs[key] = pd.read_csv(f'{folder}/{filename}')
        print(f"DataFrames loaded from CSV files in {folder} successfully.")
        return dfs
    
    def save_dfs_to_csv(self, dfs, filename):
        """
        Save a dictionary of pandas DataFrames to CSV files.

        Parameters:
        - dfs (dict): Dictionary of DataFrames to save.
        - filename (str): Base name for the CSV files.
        """
        for key, df in dfs.items():
            df.to_csv(f"{filename}_{key}.csv", index=False)
    
    def load_dfs_from_csv(self, filename):
        """
        Load a dictionary of pandas DataFrames from CSV files.

        Parameters:
        - filename (str): Base name for the CSV files.

        Returns:
        - dict: Dictionary of DataFrames loaded from the CSV files.
        """
        dfs = {}
        for file in glob.glob(f"{filename}_*.csv"):
            key = file.split(f"{filename}_")[1].replace('.csv', '')
            dfs[key] = pd.read_csv(file)
        return dfs

    def save_dfs_to_sav(self, dfs, filename):
        """
        Save a dictionary of pandas DataFrames to a .sav file.

        Parameters:
        - dfs (dict): Dictionary of DataFrames to save.
        - filename (str): Name of the .sav file to save.
        """
        savemat(filename, {key: df.values for key, df in dfs.items()})
    
    def load_dfs_from_sav(self, filename):
        """
        Load a dictionary of pandas DataFrames from a .sav file.

        Parameters:
        - filename (str): Name of the .sav file to load.

        Returns:
        - dict: Dictionary of DataFrames loaded from the .sav file.
        """
        data = loadmat(filename)
        return {key: pd.DataFrame(value) for key, value in data.items() if isinstance(value, (list, np.ndarray))}
    
    @staticmethod
    def get_most_recent_file(directory):
        # Use glob to list all files in the directory
        list_of_files = glob.glob(os.path.join(directory, '*'))
        
        if not list_of_files:
            return None

        # Get the file with the latest modification time
        most_recent_file = max(list_of_files, key=os.path.getctime)

        return most_recent_file
