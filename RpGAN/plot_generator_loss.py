import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

path_rpgan_r1_and_r2 = os.path.join(OUTPUT_DIR, 'gen_losses_rpgan_r1_and_r2.npy')
path_rpgan_r1_only   = os.path.join(OUTPUT_DIR, 'gen_losses_rpgan_r1_only.npy')

gen_losses_r1_and_r2    = np.load(path_rpgan_r1_and_r2)
gen_losses_r1 = np.load(path_rpgan_r1_only)

plt.figure(figsize=(10, 6))
plt.plot(gen_losses_r1_and_r2, label='Generator Loss (RpGAN + R1 + R2)', color = 'blue', alpha=0.8)
plt.plot(gen_losses_r1, label='Generator Loss (RpGAN + R1)', color = 'green', alpha=0.8)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Generator Loss Over Training')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()