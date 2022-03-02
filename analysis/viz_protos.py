import os
import torch
import argparse
from models.vqvae import VQVAE
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision.utils import make_grid
import numpy as np
import utils


def load_model(model_filename):
    path = os.getcwd() + '/results/'

    if torch.cuda.is_available():
        data = torch.load(path + model_filename)
    else:
        data = torch.load(path + model_filename, map_location=lambda storage, loc: storage)

    params = data["hyperparameters"]

    model = VQVAE(params['n_hiddens'], params['n_residual_hiddens'],
                  params['n_residual_layers'], params['n_embeddings'],
                  params['embedding_dim'], params['beta']).to(device)

    model.load_state_dict(data['model'])

    return model, data


def plot_metrics(data):
    results = data["results"]
    recon_errors = savgol_filter(results["recon_errors"], 19, 5)
    perplexities = savgol_filter(results["perplexities"], 19, 5)
    loss_vals = savgol_filter(results["loss_vals"], 19, 5)

    f = plt.figure(figsize=(16, 4))
    ax = f.add_subplot(1, 3, 2)
    ax.plot(recon_errors)
    ax.set_yscale('log')
    ax.set_title('Reconstruction Error')
    ax.set_xlabel('iteration')

    ax = f.add_subplot(1, 3, 3)
    ax.plot(perplexities)
    ax.set_title('Average codebook usage (perplexity).')
    ax.set_xlabel('iteration')

    ax = f.add_subplot(1, 3, 1)
    ax.plot(loss_vals)
    ax.set_yscale('log')
    ax.set_title('Overall Loss')
    ax.set_xlabel('iteration')


def display_image_grid(x):
    x = make_grid(x.cpu().detach() + 0.5)
    x = x.numpy()
    fig = plt.imshow(np.transpose(x, (1, 2, 0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()


def reconstruct(data_loader, model):
    (x, _) = next(iter(data_loader))
    x = x.to(device)
    vq_encoder_output = model.pre_quantization_conv(model.encoder(x))
    _, z_q, _, _, e_indices = model.vector_quantization(vq_encoder_output)

    x_recon = model.decoder(z_q)
    return x, x_recon, z_q, e_indices


if __name__=='__main__':
    print("Hello world")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_filename = 'vqvae_data_wed_mar_2_13_59_22_2022.pth'

    model, vqvae_data = load_model(model_filename)

    training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(
        'CIFAR10', 32)

    x_val, x_val_recon, z_q, e_indices = reconstruct(validation_loader, model)
    print(x_val.shape)
    # Plot the inputs and the reconstructions.
    display_image_grid(x_val)
    display_image_grid(x_val_recon)
