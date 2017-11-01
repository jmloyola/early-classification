import os

current_dir = os.getcwd()
dataset_dir = os.path.join(current_dir, 'Conjuntos de Datos')
file_name = 'test_dataset.txt'
file_path = os.path.join(dataset_dir, file_name)

with open(file_path, 'r') as f:
    read_data = f.read()
    print(read_data)
f.closed
