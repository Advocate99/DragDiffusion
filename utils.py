from PIL import Image, ImageDraw
from typing import Optional, Tuple
import numpy as np
import torch

def save_image(image, save_path):

    img = (image / 2 + 0.5).clamp(0, 1)
    img = img.cpu().permute(0, 2, 3, 1).float().detach().numpy()
    img = Image.fromarray((img[0] * 255).astype(np.uint8))
    img.save("images/"+save_path+".png")

def save_image_with_points(image, x,y, save_path):

    img = (image / 2 + 0.5).clamp(0, 1)
    img = img.cpu().permute(0, 2, 3, 1).float().detach().numpy()
    img = Image.fromarray((img[0] * 255).astype(np.uint8))

    draw = ImageDraw.Draw(img)
    x, y = y, x
    draw.ellipse((x-2,y-2,x+2,y+2), fill=(255, 0, 0), width=1)

    img.save("images/"+save_path+".png")

def create_circular_mask(
    h: int,
    w: int,
    center: Optional[Tuple[int, int]] = None,
    radius: Optional[int] = None,
) -> torch.Tensor:
    """
    Create a circular mask tensor.

    Args:
        h (int): The height of the mask tensor.
        w (int): The width of the mask tensor.
        center (Optional[Tuple[int, int]]): The center of the circle as a tuple (y, x). If None, the middle of the image is used.
        radius (Optional[int]): The radius of the circle. If None, the smallest distance between the center and image walls is used.

    Returns:
        A boolean tensor of shape [h, w] representing the circular mask.
    """
    if center is None:  # use the middle of the image
        center = (int(h / 2), int(w / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], h - center[0], w - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((Y - center[0]) ** 2 + (X - center[1]) ** 2)

    mask = dist_from_center <= radius
    mask = torch.from_numpy(mask).bool()
    return mask

def motion_supervison(handle_points, target_points, F, r1):
    loss = 0
    n = len(handle_points)
    for i in range(n):
        pi, ti = handle_points[i], target_points[i]
        pi = torch.tensor(pi).float()
        ti = torch.tensor(ti).float()
        target2handle = ti-pi
        d_i = target2handle / (torch.norm(target2handle) + 1e-7)
        if torch.norm(d_i) > torch.norm(target2handle):
            d_i = target2handle

        mask = create_circular_mask(
            F.shape[2], F.shape[3], center=pi.tolist(), radius=r1
        ).to(F.device)

        coordinates = torch.nonzero(mask).float()  # shape [num_points, 2]

        # Shift the coordinates in the direction d_i
        shifted_coordinates = coordinates + d_i[None].cuda()

        h, w = F.shape[2], F.shape[3]

        # Extract features in the mask region and compute the loss
        F_qi = F[:, :, mask]  # shape: [C, H*W]

        # Sample shifted patch from F
        normalized_shifted_coordinates = shifted_coordinates.clone()
        normalized_shifted_coordinates[:, 0] = (
            2.0 * shifted_coordinates[:, 0] / (h - 1)
        ) - 1  # for height
        normalized_shifted_coordinates[:, 1] = (
            2.0 * shifted_coordinates[:, 1] / (w - 1)
        ) - 1  # for width
        # Add extra dimensions for batch and channels (required by grid_sample)
        normalized_shifted_coordinates = normalized_shifted_coordinates.unsqueeze(
            0
        ).unsqueeze(
            0
        )  # shape [1, 1, num_points, 2]
        normalized_shifted_coordinates = normalized_shifted_coordinates.flip(
            -1
        )  # grid_sample expects [x, y] instead of [y, x]
        normalized_shifted_coordinates = normalized_shifted_coordinates.clamp(-1, 1)

        # Use grid_sample to interpolate the feature map F at the shifted patch coordinates
        F_qi_plus_di = torch.nn.functional.grid_sample(
            F, normalized_shifted_coordinates, mode="bilinear", align_corners=True
        )
        # Output has shape [1, C, 1, num_points] so squeeze it
        F_qi_plus_di = F_qi_plus_di.squeeze(2)  # shape [1, C, num_points]

        loss += torch.nn.functional.l1_loss(F_qi.detach(), F_qi_plus_di)
    return loss
