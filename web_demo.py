import os
import torch
from option import get_option
from model import Generator
from PIL import Image
import numpy as np
import torchvision
import torchvision.transforms as transforms

import gradio as gr


class Inference():
    def __init__(self, opt):
        self.opt = opt
        self.img_size = opt.input_size
        self.dev = torch.device("cuda:{}".format(
            opt.gpu) if torch.cuda.is_available() else "cpu")

        self.model = Generator(in_channels=3, features=64).to(self.dev)

        ft_path = os.path.join(opt.ckpt_root, "Gen.pt")
        # self.model.load_state_dict(torch.load(ft_path))

        self.model.eval()

    @torch.no_grad()
    def image_colorization(self, img):

        img = np.repeat(img[..., np.newaxis], 3, -1)
        img = Image.fromarray(img)

        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        image = transform(img)
        image = image.unsqueeze(dim=0)
        image = image.to(self.dev)

        output = self.model(image)
        torchvision.utils.save_image(output, f"output.png")

        img_path = os.path.join("output.png")
        output = Image.open(img_path)

        return output

    def demo(self):
        gr.Interface(fn=self.image_colorization,
                     inputs=gr.Sketchpad(label="Draw Here", brush_radius=5, shape=(120, 120)), outputs="image").launch()


def main():
    opt = get_option()
    torch.manual_seed(opt.seed)
    inference = Inference(opt)
    inference.demo()


if __name__ == '__main__':
    main()
