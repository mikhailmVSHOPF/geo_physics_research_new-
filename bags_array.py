import bag_df
import find_paths
import read_config

def bags_array(folder_path):
    path_list = find_paths.find_paths_to_files(folder_path)
    info = read_config.read_config_file(folder_path + "\\config_file.txt")
    dx, fps, _, rho, sigma = info
    bags_list = []
    for path in path_list:
        bag = bag_df.read_file(folder_path + "\\" + path, dx, fps, rho, sigma)
        bags_list.append(bag) 
    return bags_list