import re 
import os
import numpy as np

def find_paths_to_files(folder_path):
        regex_file = re.compile(r'.*F\d+.*\.dat')

        files_path = os.listdir(str(folder_path))
        files_path_np = np.array(files_path)

        files_path_mask = np.array([ bool(re.match(regex_file, path)) for path in files_path])

        files_paths = list(files_path_np[files_path_mask])
        print(files_paths)

        return files_paths