import sys
import os

# List of directories to be added to sys.path
directories = ['components',
               'data',  
               'features',  
               'notebooks',
               'output',
               'train',
               'tools',
               'utils',
               'xai']

# Add each directory to the sys.path variable
for dir_name in directories:
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir, dir_name)))