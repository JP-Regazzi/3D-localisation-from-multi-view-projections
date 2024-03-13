import matplotlib.pyplot as plt
import torch

from diffdrr.drr import DRR
from diffdrr.data import load_example_ct
from diffdrr.visualization import plot_drr

# Read in the volume and get the isocenter
volume, spacing = load_example_ct()
bx, by, bz = torch.tensor(volume.shape, dtype=torch.double) * \
    torch.tensor(spacing, dtype=torch.double) / 2

# Initialize the DRR module for generating synthetic X-rays
device = "cuda" if torch.cuda.is_available() else "cpu"
drr = DRR(
    volume,      # The CT volume as a numpy array
    spacing,     # Voxel dimensions of the CT
    # Source-to-detector radius (half of the source-to-detector distance)
    sdr=300.0,
    # Height of the DRR (if width is not seperately provided, the generated image is square)
    height=200,
    delx=4.0,    # Pixel spacing (in mm)
).to(device)

# Set the camera pose with rotation (yaw, pitch, roll) and translation (x, y, z)
# rotation = torch.tensor([[torch.pi, 0.0, torch.pi / 2]], device=device)
# translation = torch.tensor([[bx, by, bz]], device=device)

rotation = torch.tensor([[torch.pi, 0.0, torch.pi / 2]],
                        dtype=torch.double, device=device)
translation = torch.tensor(
    [[bx.item(), by.item(), bz.item()]], dtype=torch.double, device=device)

# 📸 Also note that DiffDRR can take many representations of SO(3) 📸
# For example, quaternions, rotation matrix, axis-angle, etc...
img = drr(rotation, translation,
          parameterization="euler_angles", convention="ZYX")
plot_drr(img, ticks=False)
plt.show()
