import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

def r1_penalty(discriminator, real_samples):
    real_samples = real_samples.detach().requires_grad_(True)
    pred_real = discriminator(real_samples) # (B,1)
    grad_real = torch.autograd.grad(outputs=pred_real.sum(), inputs=real_samples, create_graph=True)[0] # 배치 내 각 샘플에 대한 gradient 계산
    grad_norm2 = grad_real.pow(2).view(grad_real.size(0), -1).sum(1) # (B, C, H, W) -> (B, C*H*W) -> (B,)
    return grad_norm2.mean() # Expectation

def r2_penalty(discriminator, fake_samples):
    fake_samples = fake_samples.detach().requires_grad_(True)
    pred_fake = discriminator(fake_samples) # (B,1)
    grad_fake = torch.autograd.grad(outputs=pred_fake.sum(), inputs=fake_samples, create_graph=True)[0] # 배치 내 각 샘플에 대한 gradient 계산
    grad_norm2 = grad_fake.pow(2).view(grad_fake.size(0), -1).sum(1) # (B, C, H, W) -> (B, C*H*W) -> (B,)
    return grad_norm2.mean() # Expectation

# ResidualBlock
class ResidualBlock(nn.Module):
    def __init__(self, main: nn.Sequential, shortcut: nn.Module):
        super().__init__()
        self.main = main
        self.shortcut = shortcut

    def forward(self, x):
        return self.main(x) + self.shortcut(x)

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim=128, img_channels=3, featuremap_channels=64):
        super().__init__()
        self.project = nn.Linear(z_dim, 4 * 4 * featuremap_channels * 8)
        self.blocks = nn.ModuleList([
            self._residualblock(featuremap_channels * 8, featuremap_channels * 4), # (B,512,4,4)->(B,512,8,8)->(B,256,8,8)
            self._residualblock(featuremap_channels * 4, featuremap_channels * 2), # (B,256,8,8)->(B,256,16,16)->(B,128,16,16)
            self._residualblock(featuremap_channels * 2, featuremap_channels)      # (B,128,16,16)->(B,128,32,32)->(B,64,32,32)
        ])
        self.to_img = nn.Conv2d(featuremap_channels, img_channels, 3, 1, 1) # (B,64,32,32)->(B,3,32,32)

    def _residualblock(self, in_ch, out_ch):
        main = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )

        if in_ch != out_ch:
            shortcut = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
            )
        else:
            shortcut = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        return ResidualBlock(main, shortcut)

    def forward(self, z):
        x = self.project(z).view(z.size(0), -1, 4, 4) # (B,128)->(B,8192)->(B,512,4,4)
        for block in self.blocks:
            x = block(x)
        return self.to_img(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_channels=3, featuremap_channels=64):
        super().__init__()
        self.blocks = nn.ModuleList([
            self._residualblock(img_channels, featuremap_channels), # (B,3,32,32)->(B,64,16,16)
            self._residualblock(featuremap_channels, featuremap_channels * 2),  # (B,64,16,16)->(B,128,8,8)
            self._residualblock(featuremap_channels * 2, featuremap_channels * 4) # (B,128,8,8)->(B,256,4,4)
        ])
        self.final = nn.Linear(featuremap_channels * 4 * 4 * 4, 1)

    def _residualblock(self, in_ch, out_ch):
        main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        # Shortcut path: downsample via 1x1 conv or pooling
        if in_ch != out_ch:
            shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2, padding=0)
        else:
            shortcut = nn.MaxPool2d(kernel_size=2, stride=2)
        return ResidualBlock(main, shortcut)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.final(x.view(x.size(0), -1)) # (B, featuremap_channels*4*4*4)->(B,1)

# Discriminator Loss - The relativistic discriminator 논문 참고
def loss_D_rsgan(D_real, D_fake):
    return -F.logsigmoid(D_real - D_fake).mean() # D_real, D_fake는 Discriminator의 logit

# Generator Loss - The relativistic discriminator 논문 참고
def loss_G_rsgan(D_real, D_fake):
    return -F.logsigmoid(D_fake - D_real).mean()

# StackedMNIST Dataset
class StackedMNIST(Dataset):
    def __init__(self, root='./data', train=True, transform=None):
        self.transform = transform or transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=self.transform
        )

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        indices = np.random.choice(len(self.mnist), 3, replace=False)
        imgs, labels = [], []
        for i in indices:
            img, lbl = self.mnist[i]
            imgs.append(img)
            labels.append(lbl)
        stacked_img = torch.cat(imgs, dim=0)  # (3,32,32)
        mode_label = labels[0] * 100 + labels[1] * 10 + labels[2]
        return stacked_img, mode_label


# Evaludate Mode Coverage & KL Divergence
def mode_coverage_and_KL(gen, clf, device, z_dim=128, num_samples=10000):
    gen.eval()
    clf.eval()
    counts = np.zeros(1000, dtype=int) # 각 모드가 생성된 횟수를 기록
    seen = set() # unique한 모드 개수를 구하기 위해 사용
    batch = 100 # 한 번에 생성할 샘플 수
    with torch.no_grad():
        for _ in range(num_samples // batch):
            z = torch.randn(batch, z_dim, device=device) # latent vector
            fake = gen(z)
            digits = []
            for c in range(3): # R, G, B 각 채널 별로
                out = clf(fake[:, c:c+1]) # 배치에 대해 채널 하나만 clf에 넣는다.
                preds = torch.argmax(out, dim=1).cpu().numpy() # 각 샘플 당 0~9 예측
                digits.append(preds)
            modes = digits[0] * 100 + digits[1] * 10 + digits[2]
            for m in modes:
                seen.add(m)
                counts[m] += 1
    p_fake = counts / counts.sum()
    p_real = np.ones(1000) / 1000  # uniform distribution
    kl = np.sum(p_fake * (np.log(p_fake + 1e-8) - np.log(p_real))) # reverse KL divergence
    return len(seen), kl

# Classifier
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,1,1), # (B, 32, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2), # (B, 32, 16, 16)
            nn.Conv2d(32,64,3,1,1), # (B, 64, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(2), # (B, 64, 8, 8)
            nn.Flatten(),
            nn.Linear(64*8*8,10)
        )

    def forward(self, x):
        return self.net(x)


def train_classifier(path='mnist_clf.pth'):
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
    ])
    loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True, num_workers=4
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = Classifier().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for e in range(5):
        total=0
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = classifier(x)
            loss = loss_fn(out,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()*x.size(0)
        print(f"[Classifier Epoch {e+1}] Loss:{total/len(loader.dataset):.4f}")
    torch.save(classifier.state_dict(), path)
    return classifier


def train_rpgan(
    G: torch.nn.Module,
    D: torch.nn.Module,
    clf: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    z_dim: int = 128,
    batch_size: int = 256,
    epochs: int = 30,
    lrG: float = 5e-5,
    lrD: float = 1e-5,
    beta1: float = 0.5,
    beta2: float = 0.9,
    lambda_r1: float = 5.0,
    lambda_r2: float = 5.0,
):

    G.to(device)
    D.to(device)
    clf.to(device)

    optG = optim.Adam(G.parameters(), lr=lrG, betas=(beta1, beta2))
    optD = optim.Adam(D.parameters(), lr=lrD, betas=(beta1, beta2))

    # DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    G.train()
    D.train()
    clf.eval()

    gen_losses = []

    for epoch in range(1, epochs + 1):
        for i, (real, _) in enumerate(loader):
            real = real.to(device)

            # Discriminator step
            z = torch.randn(batch_size, z_dim, device=device)
            fake = G(z).detach()
            D_real = D(real)
            D_fake = D(fake)
            loss_D = loss_D_rsgan(D_real, D_fake)

            # gradient penalty
            r1 = r1_penalty(D, real)
            r2 = r2_penalty(D, fake)
            final_loss_D = loss_D + (lambda_r1 / 2) * r1 + (lambda_r2 / 2) * r2

            optD.zero_grad()
            final_loss_D.backward()
            optD.step()

            # Generator step
            z = torch.randn(batch_size, z_dim, device=device)
            fake = G(z)
            loss_G = loss_G_rsgan(D(real.detach()), D(fake))

            optG.zero_grad()
            loss_G.backward()
            optG.step()

            gen_losses.append(loss_G.item())

            if i % 10 == 0:
                print(f"[Epoch {epoch:02d} | Iter {i:04d}] "
                      f"D_loss: {final_loss_D.item():.4f}, G_loss: {loss_G.item():.4f}, "
                      f"R1: {r1.item():.4f}, R2: {r2.item():.4f}")

        # Evaluate Mode Coverage & KL Divergence
        modes, kl = mode_coverage_and_KL(G, clf, device, z_dim)
        print(f"→ [Eval Epoch {epoch:02d}] Modes: {modes}, KL: {kl:.4f}")

    return G, D, gen_losses



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialization
    G = Generator(z_dim=128)
    D = Discriminator()

    # Clf 없으면 학습시키고, 있으면 로드
    clf_path = 'mnist_clf.pth'
    if not os.path.exists(clf_path):
        print("Training classifier...")
        clf = train_classifier(path=clf_path)
    else:
        clf = Classifier()
        clf.load_state_dict(torch.load(clf_path, map_location=device))

    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    dataset = StackedMNIST(root='./data', train=True, transform=transform)

    # Training
    G, D, gen_losses = train_rpgan(
        G, D, clf, dataset, device,
        z_dim=128,
        batch_size=256,
        epochs=30,
        lrG=5e-5,
        lrD=1e-5,
        lambda_r1=5.0,
        lambda_r2=5.0
    )

    # Generator loss 기록
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, 'gen_losses_rpgan_r1_and_r2.npy')
    np.save(save_path, np.array(gen_losses)) 

    print("Training complete.")