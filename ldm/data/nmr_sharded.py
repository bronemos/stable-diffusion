import numpy as np
import imageio
import yaml
import webdataset as wds
from io import BytesIO

from torch.utils.data import IterableDataset

import os

from srt.utils.nerf import transform_points


class NMRShardedDataset(IterableDataset):
    def __init__(
        self,
        path,
        mode,
        points_per_item=2048,
        max_len=None,
        canonical_view=True,
        full_scale=False,
    ):
        """Loads the NMR dataset as found at
        https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/NMR_Dataset.zip
        Hosted by Niemeyer et al. (https://github.com/autonomousvision/differentiable_volumetric_rendering)
        Args:
            path (str): Path to dataset.
            mode (str): 'train', 'val', or 'test'.
            points_per_item (int): Number of target points per scene.
            max_len (int): Limit to the number of entries in the dataset.
            canonical_view (bool): Return data in canonical camera coordinates (like in SRT), as opposed
                to world coordinates.
            full_scale (bool): Return all available target points, instead of sampling.
        """

        self.path = path
        self.mode = mode
        self.points_per_item = points_per_item
        self.max_len = max_len
        self.canonical = canonical_view
        self.full_scale = full_scale
        self.dataset = (
            wds.WebDataset(
                os.path.join(path, f"NMR-{mode}-{{00..12}}.tar"), shardshuffle=True
            )
            .shuffle(100)
            .decode("rgb")
        )
        self.dataset_iter = iter(self.dataset)
        self.idx = 0

        self.render_kwargs = {"min_dist": 2.0, "max_dist": 4.0}

        # Rotation matrix making z=0 is the ground plane.
        # Ensures that the scenes are layed out in the same way as the other datasets,
        # which is convenient for visualization.
        self.rot_mat = np.array(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        )

    def __iter__(self):
        return self

    def __next__(self):
        sample = next(self.dataset_iter)

        view_idx = self.idx % 24
        target_views = np.array(list(set(range(24)) - set([view_idx])))

        images = [sample[f"{i:04d}.png"] for i in range(24)]
        images = np.stack(images, 0).astype(np.float32) / 255.0
        input_image = np.transpose(images[view_idx], (2, 0, 1))

        cameras = np.load(BytesIO(sample["cameras"]))
        cameras = {k: v for k, v in cameras.items()}  # Load all matrices into memory

        for i in range(24):  # Apply rotation matrix to rotate coordinate system
            cameras[f"world_mat_inv_{i}"] = self.rot_mat @ cameras[f"world_mat_inv_{i}"]
            # The transpose here is not technically necessary, since the rotation matrix is symmetric
            cameras[f"world_mat_{i}"] = cameras[f"world_mat_{i}"] @ np.transpose(
                self.rot_mat
            )

        rays = []
        height = width = 64

        xmap = np.linspace(-1, 1, width)
        ymap = np.linspace(-1, 1, height)
        xmap, ymap = np.meshgrid(xmap, ymap)

        for i in range(24):
            cur_rays = np.stack((xmap, ymap, np.ones_like(xmap)), -1)
            cur_rays = transform_points(
                cur_rays,
                cameras[f"world_mat_inv_{i}"] @ cameras[f"camera_mat_inv_{i}"],
                translate=False,
            )
            cur_rays = cur_rays[..., :3]
            cur_rays = cur_rays / np.linalg.norm(cur_rays, axis=-1, keepdims=True)
            rays.append(cur_rays)

        rays = np.stack(rays, axis=0).astype(np.float32)
        camera_pos = [cameras[f"world_mat_inv_{i}"][:3, -1] for i in range(24)]
        camera_pos = np.stack(camera_pos, axis=0).astype(np.float32)
        # camera_pos and rays are now in world coordinates.

        if self.canonical:  # Transform to canonical camera coordinates
            canonical_extrinsic = cameras[f"world_mat_{view_idx}"].astype(np.float32)
            camera_pos = transform_points(camera_pos, canonical_extrinsic)
            rays = transform_points(rays, canonical_extrinsic, translate=False)

        rays_flat = np.reshape(rays[target_views], (-1, 3))
        pixels_flat = np.reshape(images[target_views], (-1, 3))
        cpos_flat = np.tile(
            np.expand_dims(camera_pos[target_views], 1), (1, height * width, 1)
        )
        cpos_flat = np.reshape(cpos_flat, (len(target_views) * height * width, 3))
        num_points = rays_flat.shape[0]

        if not self.full_scale:
            replace = num_points < self.points_per_item
            sampled_idxs = np.random.choice(
                np.arange(num_points), size=(self.points_per_item,), replace=replace
            )

            rays_sel = rays_flat[sampled_idxs]
            pixels_sel = pixels_flat[sampled_idxs]
            cpos_sel = cpos_flat[sampled_idxs]
        else:
            rays_sel = rays_flat
            pixels_sel = pixels_flat
            cpos_sel = cpos_flat

        result = {
            "input_images": np.expand_dims(input_image, 0),  # [1, 3, h, w]
            "input_camera_pos": np.expand_dims(camera_pos[view_idx], 0),  # [1, 3]
            "input_rays": np.expand_dims(rays[view_idx], 0),  # [1, h, w, 3]
            "target_pixels": pixels_sel,  # [p, 3]
            "target_camera_pos": cpos_sel,  # [p, 3]
            "target_rays": rays_sel,  # [p, 3]
            "sceneid": self.idx,  # int
            "scene_hash": sample["__key__"],
        }

        if self.canonical:
            result["transform"] = canonical_extrinsic  # [3, 4] (optional)

        self.idx += 1

        return result
