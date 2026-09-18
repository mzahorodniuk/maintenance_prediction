import csv
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from lib.model import CAE, CAETest, CAEDenseDecoder
from lib.utils import *
from lib.dataset import ElevatorSlicesChannels

def conv_ae():
    sample = get_indexed_sample_from_data("data/GTZH.npz", 24)
    slices = get_window_slices_from_sample(sample, 208, 104)

    norm_slices = z_score_slices(slices)

    dataset = ElevatorSlicesChannels(norm_slices)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
    )

    print_traindata_parameter(loader)

    ae = CAE()
    criterion = nn.MSELoss()
    #criterion = nn.L1Loss()
    #optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(ae.parameters(), lr=0.01, momentum=0.9)

    losses = train(ae, 5000, criterion, optimizer, loader)
    

    # TEST
    plt.plot(losses, linewidth=1.0)
    plt.show()

    sample = get_indexed_sample_from_data("data/GTZH.npz", 24)
    slices = get_window_slices_from_sample(sample, 208, 10)

    norm_slices = z_score_slices(slices)

    dataset = ElevatorSlicesChannels(norm_slices)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    plot_slice_reconstruction_continuous(2.0, ae, loader)


WINDOW_SIZE = 208
STEP_SIZE = 104

def conv_ae_full_sample():
    samples = []
    for n in range(1, 39):
        sample = get_indexed_sample_from_data("data/GTZH.npz", n)
        samples.append(sample)
    
    slices = []
    for sample in samples:
        slices += get_window_slices_from_sample(sample, WINDOW_SIZE, STEP_SIZE)
    #slices = get_window_slices_from_data("data/GTZH.npz", 256, 64) # TODO Problem -> keine komplette fahrt zum testen (nur samples)

    norm_slices = slices#z_score_slices(slices)

    dataset = ElevatorSlicesChannels(norm_slices)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
    )

    print_traindata_parameter(loader)

    ae = CAETest()
    print(get_n_params(ae))
    criterion = nn.MSELoss()
    #criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
    #optimizer = torch.optim.SGD(ae.parameters(), lr=0.1, momentum=0.9)

    losses = train(ae, 100, criterion, optimizer, loader)
    

    # TEST
    plt.plot(losses, linewidth=1.0)
    plt.show()

    print("Test Empty")
    ###################
    dataset = ElevatorSlicesChannels([np.zeros((WINDOW_SIZE, 3))])
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    get_sample_error(ae, loader)


    print("Test GTZH")
    ###################
    for n in range(1, 47):
        sample = get_indexed_sample_from_data("data/GTZH.npz", n)
        slices = get_window_slices_from_sample(sample, WINDOW_SIZE, 1)

        norm_slices = slices#z_score_slices(slices)

        dataset = ElevatorSlicesChannels(norm_slices)
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
        slices = get_window_slices_from_sample(sample, WINDOW_SIZE, 1)

        norm_slices = slices#z_score_slices(slices)

        dataset = ElevatorSlicesChannels(norm_slices)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )

        print(n, end=" ")
        get_sample_error(ae, loader)

    print("Test HS")
    ###################
    for n in range(1, 11):
        sample = get_indexed_sample_from_data("data/HS.npz", n)
        slices = get_window_slices_from_sample(sample, WINDOW_SIZE, 1)

        norm_slices = slices#z_score_slices(slices)

        dataset = ElevatorSlicesChannels(norm_slices)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )

        print(n, end=" ")
        get_sample_error(ae, loader)

    
    ###################
    sample = get_indexed_sample_from_data("data/GTZH.npz", 45)
    slices = get_window_slices_from_sample(sample, WINDOW_SIZE, 26)

    norm_slices = slices#z_score_slices(slices)

    dataset = ElevatorSlicesChannels(norm_slices)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    #get_sample_error(ae, loader)

    plot_slice_reconstruction_continuous(5.0, ae, loader)
    

def conv_ae_full_sample_fault_test():
    log_data = [["model", "elevator_status", "error"]]

    samples = []
    for n in range(1, 8):
        sample = get_indexed_sample_from_data("data/FHI.npz", n)
        samples.append(sample)
    
    slices = []
    for sample in samples:
        slices += get_window_slices_from_sample(sample, WINDOW_SIZE, STEP_SIZE)

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

        slices = get_window_slices_from_sample(sample, WINDOW_SIZE, 1)

        dataset = ElevatorSlicesChannels(slices)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )

        print(n, end=" ")
        error = get_sample_error(ae, loader)
        log_data.append(["CAETest", "FHI", error])


    print("Test FHI-BREAK")
    ###################
    for n in range(1, 11):
        sample = get_indexed_sample_from_data("data/FHI-BREAK.npz", n)

        slices = get_window_slices_from_sample(sample, WINDOW_SIZE, 1)

        dataset = ElevatorSlicesChannels(slices)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )

        print(n, end=" ")
        error = get_sample_error(ae, loader)
        log_data.append(["CAETest", "FHI-BREAK", error])

    with open("conv_ae_fault_test.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(log_data)
