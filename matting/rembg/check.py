import platform
import os

file_list = [
        ('~/.u2net/u2net.onnx','https://drive.google.com/uc?id=1tCU5MM1LhRgGou5OpmpjBQbSrYIUoYab'),
        ('~/.u2net/u2net_human_seg.onnx','https://drive.google.com/uc?id=1ZfqwVxu-1XWC1xU1GHIP-FM_Knd_AX5j'),
        ('~/.u2net/u2net_cloth_seg.onnx','https://drive.google.com/uc?id=15rKbQSXQzrKCQurUjZFg8HqzZad8bcyz'),
        ('~/.u2net/u2netp.onnx','https://drive.google.com/uc?id=1tNuFmLv0TSNDjYIkjEdeH1IWKQdUA4HR')
        ]
print(f'Python version:{platform.python_version()}')

for (file,download_url) in file_list:
    if not os.path.isfile(file):
        print(f"{file} doesn't exist,please download it from {download_url} and put it into ~/.u2net/")
    else:
        print(f'{file} exists')
