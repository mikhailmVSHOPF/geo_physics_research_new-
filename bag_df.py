import re
import os 
import pandas as pd 
import numpy as np

regex =  re.compile(r'(point|line)\t(\w+|\d+|)\t(\d+,?\d+)\t(\d+,?\d+)\t(\d+,?\d+)\t?(\d+,?\d+)?\t?(\d+,?\d+)?\t?(\d+,?\d+)?')


def read_file(file_source, dx, fps, rho, sigma):
    type_line_0 = []; line_name_0 = []; frame_start_0 = []; frame_end_0 = []; x_start_0 = []; y_start_0 = []; x_end_0 = []; y_end_0 = [];

    type_line_1 = []; line_name_1 = []; frame_start_1 = []; frame_end_1 = []; x_start_1 = []; y_start_1 = []; x_end_1 = []; y_end_1 = [];

    with open(file_source) as file:
        for line in file:
            if re.search(regex, line):
                data = re.findall(regex, line)[0]
                type = data[0]
                if type == 'line':
                    name = data[1]
                    frame_start = int(data[2])
                    frame_end = int(data[3])
                    x_start = float(data[4].replace(',', '.'))
                    y_start = float(data[5].replace(',', '.'))
                    x_end = float(data[4].replace(',', '.'))
                    y_end = float(data[5].replace(',', '.'))
                    type_line_0.append(type)
                    line_name_0.append(name)
                    frame_start_0.append(frame_start)
                    frame_end_0.append(frame_end)
                    x_start_0.append(x_start)
                    y_start_0.append(y_start)
                    x_end_0.append(x_end)
                    y_end_0.append(y_end)
                
                if type == 'point':
                    name = data[1]
                    frame_start = int(data[2])
                    frame_end = 0.0
                    x_start = float(data[3].replace(',', '.'))
                    y_start = float(data[4].replace(',', '.'))
                    x_end = 0.0
                    y_end = 0.0
                    type_line_1.append(type)
                    line_name_1.append(name)
                    frame_start_1.append(frame_start)
                    frame_end_1.append(frame_end)
                    x_start_1.append(x_start)
                    y_start_1.append(y_start)
                    x_end_1.append(x_end)
                    y_end_1.append(y_end)

    type_line = type_line_0 + type_line_1; line_name = line_name_0 + line_name_1; frame_start = frame_start_0 + frame_start_1; 
    frame_end = frame_end_0 + frame_end_1; x_start = x_start_0 + x_start_1; y_start = y_start_0 + y_start_1; x_end = x_end_0 + x_end_1; y_end = y_end_0 + y_end_1;

    frame_start = np.array(frame_start)/fps
    frame_end = np.array(frame_end)/fps

    # frame_start = frame_start - min(frame_start)

    x_start = np.array(x_start)
    x_start = x_start*dx

    y_start = np.array(y_start)
    y_start = y_start*dx

    x_end = np.array(x_end)
    x_end = x_end*dx

    y_end = np.array(y_end)
    y_end = y_end*dx

    df = pd.DataFrame({"type": type_line,
                        "name": line_name,
                        "frame_start": frame_start,
                        "frame_end": frame_end,
                        "x_start": x_start,
                        "y_start": y_start,
                        "x_end": x_end,
                        "y_end": y_end,
                        "source": file_source,
                        "dx": dx, 
                        "fps": int(fps),
                        "rho": rho,
                        "sigma": sigma})
    return df;
    




                