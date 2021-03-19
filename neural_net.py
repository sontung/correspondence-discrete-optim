import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import cv2
import numpy as np
import sys
from PIL import Image
from pathlib import Path


class Generator(nn.Module):
    def __init__(self, nb_pixels, device):
        super(Generator, self).__init__()
        self.max = 0.25
        self.min = -1
        self.vec = torch.rand((nb_pixels, 2), requires_grad=True, device=device)

    def transform(self, x, theta):
        theta = theta.view(-1, 2, 3)
        x = x[:, :3, :, :]
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def find_theta(self, x):
        xs = self.main(x)
        if xs.size(0) == 1:
            xs = xs.squeeze().unsqueeze(0)
        else:
            xs = xs.squeeze()
        trans_vec = self.final_layer(xs)
        trans_vec = torch.sigmoid(trans_vec)
        theta = torch.tensor([1, 0, -1.6, 0, 1, 1.6], dtype=torch.float, requires_grad=True).repeat(x.size(0), 1).to(x.device)
        theta[:, 2] = -trans_vec[:, 0]*1.6
        theta[:, 5] = trans_vec[:, 1]*1.6
        return theta.view(-1, 2, 3)

    def infer(self, x, x_default):
        theta = self.find_theta(x)
        x_pred = self.transform(x_default, theta)
        return x_pred

    def forward(self, x, target):
        trans_vec = torch.sigmoid(self.vec)
        trans_vec = trans_vec*(self.max-self.min) + self.min
        theta = torch.hstack([torch.ones(x.size(0), 1).to(x.device),
                              torch.zeros(x.size(0), 1).to(x.device),
                              trans_vec[:, 0].reshape(-1, 1),
                              torch.zeros(x.size(0), 1).to(x.device),
                              torch.ones(x.size(0), 1).to(x.device),
                              trans_vec[:, 1].reshape(-1, 1)])
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        pred = F.grid_sample(x, grid)

        # pred = torch.ge(pred, 0).float()
        # target = torch.ge(target, 0).float()

        loss = nn.functional.mse_loss(torch.sum(pred, dim=0), target)
        loss += nn.functional.mse_loss(pred, target.unsqueeze(0).repeat(pred.size(0), 1, 1, 1))

        return x, torch.sum(pred, dim=0), loss


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    device = "cuda"
    to_tensor = torchvision.transforms.ToTensor()
    to_pil = torchvision.transforms.ToPILImage()

    img = cv2.imread("data/im1.png") * cv2.imread("data/im1masked.png")
    img = cv2.resize(img, (img.shape[1]//6, img.shape[0]//6))
    img = Image.fromarray(img)
    img_tensor = to_tensor(img).to(device)

    img2 = cv2.imread("data/im1.png") * cv2.imread("data/im1masked.png")
    img2 = cv2.resize(img2, (img2.shape[1]//6, img2.shape[0]//6))
    img2 = Image.fromarray(img2)
    img_target_tensor = to_tensor(img2).to(device)

    my_file = Path("saved/tensor_small.pt")
    if my_file.is_file():
        img_tensor2 = torch.load("saved/tensor_small.pt")
    else:
        img_tensor2 = []
        for i in range(img_tensor.size(1)):
            for j in range(img_tensor.size(2)):
                if torch.sum(img_tensor[:, i, j]).item() > 0.0:
                    img_tensor_small = torch.zeros_like(img_tensor)
                    img_tensor_small[:, i, j] = img_tensor[:, i, j]
                    img_tensor2.append(img_tensor_small)
        img_tensor2 = torch.stack(img_tensor2, dim=0)
        torch.save(img_tensor2, "saved/tensor_small.pt")

    model = Generator(img_tensor2.size(0), device).to(device)
    img_tensor2 = img_tensor2.to(device)
    optimizer = torch.optim.Adam([{"params": model.vec}], lr=1e-3, weight_decay=1e-4)
    img_tensor2 = torch.gt(img_tensor2, 0).float()
    img_target_tensor = torch.gt(img_target_tensor, 0).float()

    model.train()
    for _ in range(100):
        optimizer.zero_grad()
        img_tensor, img_tensor2_p, loss = model(img_tensor2, img_target_tensor)
        loss.backward()

        optimizer.step()
        print(loss.item())

    print(torch.sum(img_target_tensor), torch.sum(img_tensor2_p))
    img2 = to_pil(img_tensor2_p)
    img = to_pil(img_target_tensor)
    img3 = np.array(img)*128 + np.array(img2)*128

    Image.fromarray(img3).save("saved/test.png")



