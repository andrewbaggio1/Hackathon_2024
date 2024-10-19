import logging
from bot.utils.save import *
from CONFIG import * 
import os 
import psutil
import time

from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)  # 5MB log file, 5 backups
logging.getLogger().addHandler(handler)

class debugger:
    
    def __init__(self, log_file="debugger.log"):
        # Set up the logging configuration
        logging.basicConfig(
            filename=log_file,  # Log file name
            level=logging.DEBUG,  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
            filemode='w'  # Write mode (use 'a' for append)
        )
        self.logger = logging.getLogger(__name__)  # Create a logger instance
        
        self.dir_obj = directories()
        self.logger.info("Initialized debugger class and directories object.")
        
        self.vals = None  # WILL CHANGE
    
    @time_execution
    def unpack_json(self):
        vals = {}
        for file in sorted(self.dir_obj.files):
            self.logger.info(f'Unpacking {file}, with a type of {type(file)}')
            try:
                vals[f'{file[:-5]}'] = self.dir_obj.load_pickle(file)  # Use load_json if necessary
                self.logger.debug(f'Successfully unpacked {file}.')
            except Exception as e:
                self.logger.error(f'Error unpacking {file}: {e}')
        self.vals = vals
        self.logger.info('Finished unpacking all files.')
        
    def type_checker(self):
        if not self.vals:
            self.logger.warning("No data in vals to check types. Have you called unpack_json?")
            return
        
        for key in self.vals.keys():
            self.logger.info(f'Checking {key} type: {type(self.vals[key])}')
            
    def file_logger(self):
        cwd = self.dir_obj.cwd
        current_files = self.dir_obj.files  # List of files in the current directory
        parent_dir = self.dir_obj.parent_dir  # Get parent directory path

        # Log current files
        if current_files:
            self.logger.info(f'Current files in directory ({self.dir_obj.directory}): {current_files}')
        else:
            self.logger.warning(f'No files found in the current directory: {self.dir_obj.directory}')
        
        # Log parent files (files in the parent directory)
        try:
            if parent_dir:
                parent_files = self.dir_obj.shift_dir(parent_dir).files
                self.logger.info(f'Files in parent directory ({parent_dir}): {parent_files}')
                self.dir_obj.shift_dir(cwd)  # Return to original directory
            else:
                self.logger.error(f'Parent directory does not exist: {parent_dir}')
        except Exception as e:
            self.logger.error(f'Error accessing parent directory: {e}')
            
    def access_config(self):
        base_dir = self.dir_obj.great_grandparent_dir  # SUBJECT TO BE GENERALIZED
        cwd = self.dir_obj.cwd
        
        try:
            # Log shifts between directories for tracking
            self.logger.info(f"Accessing base directory: {base_dir}")
            self.dir_obj.shift_dir(base_dir)
            
            closest_folder_mega = save.find_closest_kw_folder_down(kw='outputs_mega')
            self.logger.info(f"Shifting to closest 'outputs_mega' folder: {closest_folder_mega}")
            self.dir_obj.shift_dir(closest_folder_mega)
            
            closest_folder_end = save.find_closest_kw_folder_down(kw='outputs_end')
            self.logger.info(f"Shifting to closest 'outputs_end' folder: {closest_folder_end}")
            self.dir_obj.shift_dir(closest_folder_end)
            
            recent_job_dir = sorted(save.files)[-2] + '/bokeh'  # MOST RECENT JOB
            self.logger.info(f"Shifting to most recent job directory: {recent_job_dir}")
            self.dir_obj.shift_dir(recent_job_dir)

        except Exception as e:
            self.logger.error(f"Error during access_config: {e}")
        finally:
            # Ensure we always return to the original directory
            self.dir_obj.shift_dir(cwd)
            self.logger.info(f"Returned to original directory: {cwd}")

    def memory_usage_logger(self):
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        self.logger.info(f"Memory Usage: RSS={memory_info.rss / (1024 ** 2):.2f} MB, VMS={memory_info.vms / (1024 ** 2):.2f} MB")
        
    def time_execution(func):
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            elapsed_time = time.time() - start_time
            self.logger.info(f"{func.__name__} executed in {elapsed_time:.4f} seconds")
            return result
        return wrapper
    
    def compare_directories(self, dir1, dir2):
        files_dir1 = set(self.dir_obj.shift_dir(dir1).files)
        files_dir2 = set(self.dir_obj.shift_dir(dir2).files)
        
        added_files = files_dir2 - files_dir1
        removed_files = files_dir1 - files_dir2
        
        if added_files:
            self.logger.info(f"Files added in {dir2}: {added_files}")
        if removed_files:
            self.logger.info(f"Files removed from {dir1}: {removed_files}")
        
        self.dir_obj.shift_dir(self.dir_obj.cwd)  # Return to original directory

    def log_file_sizes(self):
        for file in self.dir_obj.files:
            file_path = os.path.join(self.dir_obj.directory, file)
            file_size = os.path.getsize(file_path) / (1024 ** 2)  # Convert to MB
            self.logger.info(f"File: {file}, Size: {file_size:.2f} MB")
            
    def config_logger(self, config):
        for key, value in config.items():
            self.logger.info(f"Config {key}: {value}")
            
    def exception_logger(self, func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Exception in {func.__name__}: {e}")
                raise e  # Optionally re-raise the exception
        return wrapper
    
    def dir_size_logger(self):
        total_size = 0
        for file in self.dir_obj.files:
            file_path = os.path.join(self.dir_obj.directory, file)
            total_size += os.path.getsize(file_path)
        total_size_mb = total_size / (1024 ** 2)  # Convert to MB
        self.logger.info(f"Total size of directory ({self.dir_obj.directory}): {total_size_mb:.2f} MB")
        
    def validate_files(self, validation_func):
        for file in self.dir_obj.files:
            file_path = os.path.join(self.dir_obj.directory, file)
            if not validation_func(file_path):
                self.logger.warning(f"Validation failed for file: {file}")
            else:
                self.logger.info(f"Validation succeeded for file: {file}")
    
    @staticmethod
    def val_func(file_path):
        pass
    
    # LIKE CONFIG'ing #
    def log_environment_variables(self):
        env_vars = ["PYTHONPATH", "HOME", "PATH"]  # Add more environment variables as needed
        for var in env_vars:
            self.logger.info(f"{var}: {os.getenv(var)}")
    
    def cleanup_old_logs(self, days=7):
        log_dir = os.path.dirname(self.logger.handlers[0].baseFilename)
        cutoff_time = time.time() - (days * 86400)  # Days to seconds
        for log_file in os.listdir(log_dir):
            log_file_path = os.path.join(log_dir, log_file)
            if os.path.getmtime(log_file_path) < cutoff_time:
                os.remove(log_file_path)
                self.logger.info(f"Removed old log file: {log_file}")