import os
import torch
import torch.nn as nn
import torch.optim as optim
from data import generate_loader
from option import get_option
from model import Generator, Discriminator
from tqdm import tqdm
from torchmetrics.image import FrechetInceptionDistance


class Solver():
    def __init__(self, opt):
        self.opt = opt
        self.img_size = opt.input_size
        self.dev = torch.device("cuda:{}".format(
            opt.gpu) if torch.cuda.is_available() else "cpu")
        print("device: ", self.dev)

        self.generator = Generator(in_channels=3, features=64).to(self.dev)
        self.discriminator = Discriminator(in_channels=3).to(self.dev)

        if opt.pretrained:
            load_path = os.path.join(opt.chpt_root, "Gen.pt")
            self.generator.load_state_dict(torch.load(load_path))

            load_path = os.path.join(opt.chpt_root, "Disc.pt")
            self.discriminator.load_state_dict(torch.load(load_path))

        if opt.multigpu:
            self.generator = nn.DataParallel(
                self.generator, device_ids=self.opt.device_ids).to(self.dev)
            self.discriminator = nn.DataParallel(
                self.discriminator, device_ids=self.opt.device_ids).to(self.dev)

        print("# Generator params:", sum(
            map(lambda x: x.numel(), self.generator.parameters())))
        print("# Discriminator params:", sum(
            map(lambda x: x.numel(), self.discriminator.parameters())))

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        self.fid = FrechetInceptionDistance(
            reset_real_features=False, normalize=True)

        self.optimizer_G = optim.Adam(
            self.generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        self.train_loader = generate_loader(opt)
        print("train set ready")

        self.fid_minimum = 1000

    def fit(self):
        opt = self.opt
        print("start training")

        for epoch in range(opt.n_epoch):
            loop = tqdm(self.train_loader)

            for i, (input, target) in enumerate(loop):
                input = input.to(self.dev)
                target = target.to(self.dev)

                # train Discriminator
                fake_image = Generator(input)
                D_real = Discriminator(input, target)
                D_real_loss = self.bce_loss(D_real, torch.ones_like(D_real))
                D_fake = Discriminator(input, fake_image.detach())
                D_fake_loss = self.bce_loss(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2

                self.optimizer_D.zero_grad()
                D_loss.backward()
                self.optimizer_D.step()

                # train Generator
                D_fake = Discriminator(input, fake_image)
                G_fake_loss = self.bce_loss(D_fake, torch.ones_like(D_fake))
                G_recon_loss = self.l1_loss(fake_image, target) * opt.l1_lambda
                G_loss = G_fake_loss + G_recon_loss

                self.optimizer_G.zero_grad()
                G_loss.backward()
                self.optimizer_G.step()

                # fid score update
                self.fid.update(target, real=True)
                self.fid.update(fake_image, real=False)
                self.fid.compute()

            if self.fid < self.fid_minimum:
                self.fid_minimum = self.fid
                self.save()

            print(
                f"[Epoch {epoch+1}/{opt.n_epoch}] [D loss: {D_loss.item():.6f}] [G loss: {G_loss.item():.6f}] [FID minimum: {self.fid_minimum:.6f}]")

    def save(self):
        os.makedirs(os.path.join(self.opt.ckpt_root), exist_ok=True)
        G_save_path = os.path.join(self.opt.ckpt_root, "Gen.pt")
        D_save_path = os.path.join(self.opt.ckpt_root, "Disc.pt")
        torch.save(self.generator.state_dict(), G_save_path)
        torch.save(self.discriminator.state_dict(), D_save_path)

    def main():
        opt = get_option()
        torch.manual_seed(opt.seed)
        solver = Solver(opt)
        solver.fit()

    if __name__ == "__main__":
        main()
