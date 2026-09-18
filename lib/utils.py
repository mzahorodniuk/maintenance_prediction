import numpy as np
import torch
from tqdm import tqdm as _tqdm
from functools import partial
import matplotlib.pyplot as plt
import scipy.signal as signal

# Globale Standard-Optionen für tqdm #
tqdm = partial(
    _tqdm,
    #ascii=" ░▒▓█",
    leave=True,
    bar_format='{l_bar}{bar:25}{r_bar}{bar:-10b}'
)
######################################

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def z_score_norm_1d(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

def z_score_slices(slices):
    norm_slices = []
    for slice in slices:
        x = z_score_norm_1d(slice[:,0])
        y = z_score_norm_1d(slice[:,1])
        z = z_score_norm_1d(slice[:,2])
        norm_slices.append(np.column_stack((x, y, z)))
    return norm_slices
        
def get_indexed_sample_from_data(file, number):
    data = np.load(file)
    return data['arr_{0}'.format(number - 1)]

def get_window_slices_from_sample(data, size, step):
    slices = []
    max_index = data.shape[0] - 1
    for x_s in range(0, max_index+1, step):
        x_e = x_s + size - 1
        if x_e <= max_index:
            slices.append(data[x_s:x_e+1,:])
    return slices

def get_window_slices_from_data(file, w_size, w_step): #TODO anzahl an fahrten zum samplen (zum samplen immer ganze fahrten nehmen)
    data = np.load(file)
    samples = []
    for n in range(len(data)):
        sample = data['arr_{0}'.format(n)]
        samples.append(sample)
    
    slices = []
    for sample in samples:
        slices += get_window_slices_from_sample(sample, w_size, w_step)
    return slices

def print_traindata_parameter(dataset_loader):
    print("-----------------------")
    print(" Trainingdata:")
    print("   {0} batches per epoch".format(len(dataset_loader)))
    print("   {0} samples per batch".format(dataset_loader.batch_size))
    print("-----------------------\n")

def train(model, epochs, criterion, optimizer, data_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    running_epoch_losses = []
    with tqdm(range(epochs)) as pbar:
        for epoch in pbar:
            running_epoch_loss = 0
            running_epoch_count = 0
            for x_train, y_train in data_loader:
                x_train, y_train = x_train.to(device), y_train.to(device)
                optimizer.zero_grad()
                outputs = model(x_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
                running_epoch_loss += loss.to("cpu").detach().numpy()
                running_epoch_count += 1
            avg_epoch_loss = running_epoch_loss / running_epoch_count
            running_epoch_losses.append(avg_epoch_loss)
            pbar.set_postfix_str("Epoch: %u/%u Loss: %f" % (epoch + 1, epochs, avg_epoch_loss))
    return running_epoch_losses

def vector_to_slice(vec):
    length = vec.shape[0] // 3
    x = vec[0:length]
    y = vec[length:2*length]
    z = vec[2*length:3*length]
    slice = np.column_stack((x, y, z))
    return slice

def plot_slice_reconstruction(input_slice, output_slice):
    x_in, x_out = input_slice[:,0], output_slice[:,0]
    y_in, y_out = input_slice[:,1], output_slice[:,1]
    z_in, z_out = input_slice[:,2], output_slice[:,2]
    x_e = x_out - x_in
    y_e = y_out - y_in
    z_e = z_out - z_in

    plt.figure(figsize=(16, 8))

    plt.subplot(3,1,1)
    plt.plot(x_in, label="input", linewidth=0.75)
    plt.plot(x_out, label="output", linewidth=0.75)
    plt.plot(x_e, label="error", linewidth=0.75)
    plt.legend()
    plt.grid(True)
    plt.title("x")

    plt.subplot(3,1,2)
    plt.plot(y_in, label="input", linewidth=0.75)
    plt.plot(y_out, label="output", linewidth=0.75)
    plt.plot(y_e, label="error", linewidth=0.75)
    plt.legend()
    plt.grid(True)
    plt.title("y")

    plt.subplot(3,1,3)
    plt.plot(z_in, label="input", linewidth=0.75)
    plt.plot(z_out, label="output", linewidth=0.75)
    plt.plot(z_e, label="error", linewidth=0.75)
    plt.legend()
    plt.grid(True)
    plt.title("z")

    plt.tight_layout()
    plt.show()

def plot_slice_reconstruction_continuous(fps, ae, loader):
    plt.figure(figsize=(16, 8))
    ae.eval()
    sample_counter = 0
    for x, _ in loader:
        with torch.no_grad():
            recon = ae(x)
            x_slice = x.squeeze().T.numpy()
            recon_slice = recon.squeeze().T.numpy()
            
            input_slice = x_slice
            output_slice = recon_slice
            x_in, x_out = input_slice[:,0], output_slice[:,0]
            y_in, y_out = input_slice[:,1], output_slice[:,1]
            z_in, z_out = input_slice[:,2], output_slice[:,2]
            x_e = x_out - x_in
            y_e = y_out - y_in
            z_e = z_out - z_in

            plt.clf()

            plt.subplot(3,1,1)
            plt.plot(x_in, label="input", linewidth=0.75)
            plt.plot(x_out, label="output", linewidth=0.75)
            plt.plot(x_e, label="error", linewidth=0.75)
            plt.legend()
            plt.grid(True)
            plt.title("x")

            plt.subplot(3,1,2)
            plt.plot(y_in, label="input", linewidth=0.75)
            plt.plot(y_out, label="output", linewidth=0.75)
            plt.plot(y_e, label="error", linewidth=0.75)
            plt.legend()
            plt.grid(True)
            plt.title("y")

            plt.subplot(3,1,3)
            plt.plot(z_in, label="input", linewidth=0.75)
            plt.plot(z_out, label="output", linewidth=0.75)
            plt.plot(z_e, label="error", linewidth=0.75)
            plt.legend()
            plt.grid(True)
            plt.title("z")

            plt.tight_layout()

            plt.draw()
            plt.pause(1.0 / fps)
            print("Sample: %u" % sample_counter)
            sample_counter += 10

def get_sample_error(ae, data_loader):
    sum_error = 0
    window_count = 0
    std_mean = 0

    ae.eval()
    for x, _ in data_loader:
        with torch.no_grad():
            recon = ae(x)
            error = (recon - x).squeeze()
            std_mean += error.std().item()
            sum_error += error.abs().sum().item()
            window_count += 1
    print("Average window error: %f" % (sum_error / window_count))
    #print("Average window std error: %f" % (std_mean / window_count))
    score_text = "Average window score: %f" % (100 / (1 + (sum_error / window_count)))
    print(bold_text(score_text))
    return sum_error / window_count

def get_slice_params(slice): # mehr? nur für x,y? extra ae der z mean rekonstruiert?
    params = []

    x = slice[:,0]
    y = slice[:,1]
    z = slice[:,2]

    # peak
    params.append(np.max(np.abs(x)))
    params.append(np.max(np.abs(y)))
    params.append(np.max(np.abs(z)))

    # min
    params.append(np.min(x))
    params.append(np.min(y))
    params.append(np.min(z))

    # max
    params.append(np.max(x))
    params.append(np.max(y))
    params.append(np.max(z))

    # rms
    params.append(np.sqrt(np.sum(x**2) / x.shape[0]))
    params.append(np.sqrt(np.sum(y**2) / y.shape[0]))
    params.append(np.sqrt(np.sum(z**2) / z.shape[0]))

    # mean
    params.append(np.mean(x))
    params.append(np.mean(y))
    params.append(np.mean(z))

    # std
    params.append(np.std(x))
    params.append(np.std(y))
    params.append(np.std(z))

    # x,y correlation   --- deutliche abgrenzung von anderem verhalten ---
    params.append(np.corrcoef(x, y)[0,1])
    # x,z correlation
    params.append(np.corrcoef(x, z)[0,1])
    # y,z correlation
    params.append(np.corrcoef(y, z)[0,1])

    return np.array(params)

def get_slice_envelope(slice):
    x = slice[:,0]
    y = slice[:,1]
    z = slice[:,2]

    x_abs = np.abs(x)
    y_abs = np.abs(y)
    z_abs = np.abs(z) # Problem: keine null durchgänge mehr

    fs = 416
    fc = 104
    n = 2
    b, a = signal.butter(n, fc / (fs * 0.5), btype="lowpass")

    x_filt = signal.lfilter(b, a, x_abs)
    y_filt = signal.lfilter(b, a, y_abs)
    z_filt = signal.lfilter(b, a, z)

    return np.column_stack((x_filt, y_filt, z_filt))

def bold_text(text):
    return "\033[1m" + text + "\033[0m"