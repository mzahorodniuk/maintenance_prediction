import csv
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from lib.model import DAE
from lib.utils import *
from lib.dataset import ElevatorSlices

def deep_ae():
    sample = get_indexed_sample_from_data("data/GTZH.npz", 24)
    slices = get_window_slices_from_sample(sample, 208, 104)

    norm_slices = slices#z_score_slices(slices)

    dataset = ElevatorSlices(norm_slices)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
    )

    print_traindata_parameter(loader)

    ae = DAE()
    criterion = nn.MSELoss()
    #criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
    #optimizer = torch.optim.SGD(ae.parameters(), lr=0.1, momentum=0.9)

    losses = train(ae, 5000, criterion, optimizer, loader)
    

    # TEST
    plt.plot(losses, linewidth=1.0)
    plt.show()

    sample = get_indexed_sample_from_data("data/GTZH.npz", 24)
    slices = get_window_slices_from_sample(sample, 208, 1)

    norm_slices = slices#z_score_slices(slices)

    dataset = ElevatorSlices(norm_slices)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    ae.eval()
    for x, _ in loader:
        with torch.no_grad():
            recon = ae(x)
            x_slice = vector_to_slice(x.numpy().squeeze())
            recon_slice = vector_to_slice(recon.numpy().squeeze())
            plot_slice_reconstruction(x_slice, recon_slice)


    """
    lr_list = np.linspace(0.15-0.05, 0.15+0.05, 10)
    for lr in lr_list:
        ae = DAE()
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(ae.parameters(), lr=lr, momentum=0.97)
        losses = train(ae, 250, criterion, optimizer, loader)
        plt.plot(losses, linewidth=0.5, label="lr={0}".format(np.round(lr, 3)))
    plt.legend()
    plt.ylim(0, 1.5)
    plt.show()
    """

WINDOW_SIZE = 208
STEP_SIZE = 104

def deep_ae_fault_test():
    log_data = [["model", "elevator_status", "error"]]

    samples = []
    for n in range(1, 8):
        sample = get_indexed_sample_from_data("data/FHI.npz", n)
        samples.append(sample)
    
    slices = []
    for sample in samples:
        slices += get_window_slices_from_sample(sample, WINDOW_SIZE, STEP_SIZE)

    dataset = ElevatorSlices(slices)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
    )

    print_traindata_parameter(loader)

    ae = DAE()
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

        dataset = ElevatorSlices(slices)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )

        print(n, end=" ")
        error = get_sample_error(ae, loader)
        log_data.append(["DAE", "FHI", error])


    print("Test FHI-BREAK")
    ###################
    for n in range(1, 11):
        sample = get_indexed_sample_from_data("data/FHI-BREAK.npz", n)

        slices = get_window_slices_from_sample(sample, WINDOW_SIZE, 1)

        dataset = ElevatorSlices(slices)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )

        print(n, end=" ")
        error = get_sample_error(ae, loader)
        log_data.append(["DAE", "FHI-BREAK", error])

    with open("deep_ae_fault_test.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(log_data)