import argparse

import numpy as np
import rioxarray
import tiffile
import torch
from matplotlib import pyplot as plt

from Network import S2S2Net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints/deeplabv3plus_6band_dice.ckpt", help="Path to the pretrained model checkpoint")
    parser.add_argument('--save_dir', type=str, default='tb_logs', help="Path to save the checkpoints")
    parser.add_argument('--train_data_path', type=str, help="Path to the training dataset")
    parser.add_argument('--model_type', type=str, default="dice", help="dice, bce, or dice_noSR")
    parser.add_argument('--num_epoch', type=int, help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=1, help="Training batch size")
    parser.add_argument('--learning_rate', type=float, default=0.00006, help="Training learning rate")
    parser.add_argument('--n_gpus', type=int, default=1, help="Number of GPU to use")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    args = parser.parse_args()

    model = S2S2Net(args=args)
    model.load_state_dict(
            torch.load(args.checkpoint_dir)["state_dict"]
        )
    model.eval()
    model.to("cuda")

    # data must be in int1. Used bands indices are RGB+NIR: 3, 2, 1, 7, 10, 11 (0-indexed !!!!!)
    x = tiffile.imread("test.tiff")
    x = torch.Tensor(x).permute(0, 3, 1, 2).type(torch.float32)
    x = x.to("cuda")
    print(x.shape)

    with torch.no_grad():
        # x = torch.rand((1, 6, 512, 512), device="cuda")
        pred = model(x)
        # pred = torch.sigmoid(pred)
        print(pred.shape)
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        # divide by 2**16 due to int16 and * 7 for nicer visuals
        axs[0].imshow(x.permute(0, 2, 3, 1).squeeze()[..., :3].cpu() / (2**16) * 7)
        axs[1].imshow(pred.squeeze().cpu().numpy())
        axs[2].imshow(torch.sigmoid(pred).squeeze().cpu().numpy(), vmin=0, vmax=1)
        plt.savefig("test.svg")
        plt.show()

if __name__ == '__main__':
    main()