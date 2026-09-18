import csv

from lib.utils import *

def convert_data_R():
    sample = get_indexed_sample_from_data("data/FHI.npz", 1)
    print(sample)

    log_data = [["sample", "accel", "axis"]]
    for i in range(sample.shape[0]):
        log_data.append([i, sample[i, 0], "x"])
        log_data.append([i, sample[i, 1], "y"])
        log_data.append([i, sample[i, 2], "z"])

    with open("data.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(log_data)