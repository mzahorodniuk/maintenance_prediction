import csv
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from lib.model import *
from lib.utils import *
from lib.dataset import *

WINDOW_SIZE = 208
STEP_SIZE = 104

def deep_ae_param(): # fault, smaller window, more params, model
    samples = []
    for n in range(1, 32):
        sample = get_indexed_sample_from_data("data/GTZH.npz", n)
        samples.append(sample)
    
    slices = []
    for sample in samples:
        slices += get_window_slices_from_sample(sample, WINDOW_SIZE, STEP_SIZE)

    dataset = ElevatorSliceParams(slices)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
    )
    
    print_traindata_parameter(loader)

    ae = DAEParam(21)
    print(get_n_params(ae))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)

    losses = train(ae, 100, criterion, optimizer, loader)

    # TEST
    plt.plot(losses, linewidth=1.0)
    plt.show()

    print("Test GTZH")
    ###################
    for n in range(1, 47):
        sample = get_indexed_sample_from_data("data/GTZH.npz", n)
        sample_slices = get_window_slices_from_sample(sample, WINDOW_SIZE, 1)

        dataset = ElevatorSliceParams(sample_slices)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )

        print(n, end=" ")
        get_sample_error(ae, loader)

    print("Test OVGU")
    ###################
    for n in range(1, 6):
        sample = get_indexed_sample_from_data("data/OVGU.npz", n)
        sample_slices = get_window_slices_from_sample(sample, WINDOW_SIZE, 1)

        dataset = ElevatorSliceParams(sample_slices)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )

        print(n, end=" ")
        get_sample_error(ae, loader)


def deep_ae_param_fault_test():
    log_data = [["model", "elevator_status", "error"]]

    #train_indices = [1, 6, 2, 7, 3, 8, 4]

    samples = []
    for n in range(1, 8):
        sample = get_indexed_sample_from_data("data/FHI.npz", n)
        samples.append(sample)
    
    slices = []
    for sample in samples:
        slices += get_window_slices_from_sample(sample, WINDOW_SIZE, STEP_SIZE)

    dataset = ElevatorSliceParams(slices)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
    )

    print_traindata_parameter(loader)

    ae = DAEParam(21)
    print(get_n_params(ae))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)

    losses = train(ae, 100, criterion, optimizer, loader)

    # TEST
    plt.plot(losses, linewidth=1.0)
    #plt.show()

    print("Test FHI")
    ###################
    for n in range(1, 11):
        sample = get_indexed_sample_from_data("data/FHI.npz", n)
        sample_slices = get_window_slices_from_sample(sample, WINDOW_SIZE, 1)

        dataset = ElevatorSliceParams(sample_slices)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )

        print(n, end=" ")
        error = get_sample_error(ae, loader)
        log_data.append(["DAEParam", "FHI", error])


    print("Test FHI-BREAK")
    ###################
    for n in range(1, 11):
        sample = get_indexed_sample_from_data("data/FHI-BREAK.npz", n)
        sample_slices = get_window_slices_from_sample(sample, WINDOW_SIZE, 1)

        dataset = ElevatorSliceParams(sample_slices)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )

        print(n, end=" ")
        error = get_sample_error(ae, loader)
        log_data.append(["DAEParam", "FHI-BREAK", error])

    with open("deep_ae_param_fault_test.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(log_data)