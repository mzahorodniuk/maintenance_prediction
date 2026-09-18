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

def conv_ae_amp_signals_test():
    sample = get_indexed_sample_from_data("data/GTZH.npz", 1)
    slices = get_window_slices_from_sample(sample, WINDOW_SIZE, STEP_SIZE)

    for slice in slices:
        slice_env = get_slice_envelope(slice)
        plot_slice_reconstruction(slice, slice_env)

def conv_ae_amp_signals():
    samples = []
    for n in range(1, 39):
        sample = get_indexed_sample_from_data("data/GTZH.npz", n)
        samples.append(sample)
    
    slices = []
    for sample in samples:
        sample_slices = get_window_slices_from_sample(sample, WINDOW_SIZE, STEP_SIZE)
        for sample_slice in sample_slices:
            slices.append(get_slice_envelope(sample_slice))

    dataset = ElevatorSlicesChannels(slices)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
    )

    print_traindata_parameter(loader)

    ae = CAETest()
    print(get_n_params(ae))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
    #optimizer = torch.optim.SGD(ae.parameters(), lr=0.1, momentum=0.9)

    losses = train(ae, 100, criterion, optimizer, loader)
    

    # TEST
    plt.plot(losses, linewidth=1.0)
    plt.show()

    print("Test GTZH")
    ###################
    for n in range(1, 47):
        sample = get_indexed_sample_from_data("data/GTZH.npz", n)

        slices = []
        sample_slices = get_window_slices_from_sample(sample, WINDOW_SIZE, 10)
        for sample_slice in sample_slices:
                slices.append(get_slice_envelope(sample_slice))

        dataset = ElevatorSlicesChannels(slices)
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
        
        slices = []
        sample_slices = get_window_slices_from_sample(sample, WINDOW_SIZE, 10)
        for sample_slice in sample_slices:
                slices.append(get_slice_envelope(sample_slice))

        dataset = ElevatorSlicesChannels(slices)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )

        print(n, end=" ")
        get_sample_error(ae, loader)


    ###################
    sample = get_indexed_sample_from_data("data/GTZH.npz", 45)
    
    slices = []
    sample_slices = get_window_slices_from_sample(sample, WINDOW_SIZE, 10)
    for sample_slice in sample_slices:
            slices.append(get_slice_envelope(sample_slice))

    dataset = ElevatorSlicesChannels(slices)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    #get_sample_error(ae, loader)

    plot_slice_reconstruction_continuous(5.0, ae, loader)


def conv_ae_amp_signals_fault_test():
    log_data = [["model", "elevator_status", "error"]]

    samples = []
    for n in range(1, 8):
        sample = get_indexed_sample_from_data("data/FHI.npz", n)
        samples.append(sample)
    
    slices = []
    for sample in samples:
        sample_slices = get_window_slices_from_sample(sample, WINDOW_SIZE, STEP_SIZE)
        for sample_slice in sample_slices:
            slices.append(get_slice_envelope(sample_slice))

    dataset = ElevatorSlicesChannels(slices)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
    )

    print_traindata_parameter(loader)

    ae = CAETest()
    print(get_n_params(ae))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)

    losses = train(ae, 100, criterion, optimizer, loader)

    # TEST
    plt.plot(losses, linewidth=1.0)
    plt.show()

    print("Test FHI")
    ###################
    for n in range(1, 11):
        sample = get_indexed_sample_from_data("data/FHI.npz", n)

        slices = []
        sample_slices = get_window_slices_from_sample(sample, WINDOW_SIZE, 1)
        for sample_slice in sample_slices:
                slices.append(get_slice_envelope(sample_slice))

        dataset = ElevatorSlicesChannels(slices)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )

        print(n, end=" ")
        error = get_sample_error(ae, loader)
        log_data.append(["CAETest-AMP", "FHI", error])


    print("Test FHI-BREAK")
    ###################
    for n in range(1, 11):
        sample = get_indexed_sample_from_data("data/FHI-BREAK.npz", n)

        slices = []
        sample_slices = get_window_slices_from_sample(sample, WINDOW_SIZE, 1)
        for sample_slice in sample_slices:
                slices.append(get_slice_envelope(sample_slice))

        dataset = ElevatorSlicesChannels(slices)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )

        print(n, end=" ")
        error = get_sample_error(ae, loader)
        log_data.append(["CAETest-AMP", "FHI-BREAK", error])

    with open("conv_ae_amp_signals_fault_test.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(log_data)