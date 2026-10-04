import re 

def read_config_file(file_source):
        try:
            with open(file_source) as file:
                regex_config_file_info = re.compile(r'^(\w+)=(\d+.*\d*|\w+)')
                data = [re.findall(regex_config_file_info, line) for line in file]
                dx = float(data[0][0][1])/100#м/кол-во пикселей
                fps = float(data[1][0][1])#кадров/секунду
                type = str(data[2][0][1])
                rho = float(data[4][0][1])
                sigma = float(data[3][0][1]) 
                F = float(data[5][0][1])
                return dx, fps, type, rho, sigma
        except Exception as e:
            print(e)