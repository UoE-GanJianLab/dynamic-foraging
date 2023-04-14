import numpy as np

def one_to_zero_cell(cell_name: str) -> str:
    cell_type, index = cell_name.split('_')
    index = int(index)
    converted_index = index - 1
    return '_'.join([cell_type, str(converted_index)])

def zero_to_one_cell(cell_name: str) -> str:
    cell_type, index = cell_name.split('_')
    index = int(index)
    converted_index = index + 1
    return '_'.join([cell_type, str(converted_index)])


    