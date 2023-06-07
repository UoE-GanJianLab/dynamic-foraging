import numpy as np

# convert cell index from 1-based to 0-based
def one_to_zero_cell(cell_name: str) -> str:
    cell_type, index = cell_name.split('_')
    index = int(index)
    converted_index = index - 1
    return '_'.join([cell_type, str(converted_index)])

# convert cell index from 0-based to 1-based
def zero_to_one_cell(cell_name: str) -> str:
    cell_type, index = cell_name.split('_')
    index = int(index)
    converted_index = index + 1
    return '_'.join([cell_type, str(converted_index)])


    