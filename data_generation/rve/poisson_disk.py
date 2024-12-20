r"""Poisson Disk Sampling in 2D periodic space."""
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Callable, Optional

class PeriodicPoissonDiskSampler2D:
    r"""
    Poisson Disk Sampling in 2D periodic space. The radius of each disk is
    not fixed, but is randomly sampled from a distribution given by the user.

    Args:
        width (int): Width of the 2D space.
        height (int): Height of the 2D space.
        r_min (float): Minimum radius of the disks.
        r_max (float): Maximum radius of the disks.
        r_distribution (Optional[Callable[[], float]]): Function that returns a
            random radius between r_min and r_max. If None, a uniform
            distribution is used.
        rng (Optional[np.random.Generator]): Random number generator. If None,
            use the default random number generator.
        print_fn (Optional[Callable[[str], None]]): Function that prints debug
            information. If None, do nothing.
    """
    def __init__(self, width: int, height: int, r_min: float, r_max: float,
                 r_distribution: Optional[Callable[[], float]] = None,
                 rng: Optional[np.random.Generator] = None,
                 print_fn: Optional[Callable[[str], None]] = print):
        self.width = width
        self.height = height
        self.r_min = r_min
        self.r_max = r_max
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        if r_distribution is None:
            r_distribution = lambda: self.rng.uniform(r_min, r_max)
        self.r_distribution = r_distribution
        if print_fn is None:
            print_fn = lambda x: None
        self.print_fn = print_fn

        cell_size = self.r_min * np.sqrt(2)
        self.num_grid_x = int(np.ceil(self.width / cell_size))
        self.num_grid_y = int(np.ceil(self.height / cell_size))
        # Adjust cell size for width and height directions, so that the grid
        # uniformly covers the space.
        self.cell_size_x = self.width / self.num_grid_x
        self.cell_size_y = self.height / self.num_grid_y

        def coord_to_grid_index(coord: NDArray) -> Tuple[int, int]:
            # coord is (2,) array
            coord = np.mod(coord, [self.width, self.height])
            return int(coord[0] / self.cell_size_x), int(coord[1] / self.cell_size_y)
        self.get_idx = coord_to_grid_index

        def periodic_dist(coord1: NDArray, coord2: NDArray) -> NDArray:
            # coord1 and coord2 are N x 2 or 2, M x 2 or 2 arrays, respectively
            coord1 = coord1.reshape(-1, 2)
            coord2 = coord2.reshape(-1, 2)
            dx = coord1[:, 0][:, None] - coord2[:, 0][None, :]  # N x M
            dx = np.mod(dx, self.width)  # Periodic boundary
            dx = np.minimum(dx, self.width - dx)
            dy = coord1[:, 1][:, None] - coord2[:, 1][None, :]  # N x M
            dy = np.mod(dy, self.height)  # Periodic boundary
            dy = np.minimum(dy, self.height - dy)
            return np.sqrt(dx ** 2 + dy ** 2)
        self.dist_func = periodic_dist

    def sample(self,
               target_area_frac: float,
               keep_last: bool = True) -> Tuple[NDArray, bool, float]:
        r"""
        Perform Poisson Disk Sampling.

        Args:
            target_area_frac (float): Fraction of the space that should be covered
                by the disks.
            keep_last (bool): If True, the last sample that exceeds the target
                area fraction is kept. If False, it is discarded.

        Returns:
            result (NDArray): Array of shape (N, 3) containing the coordinates
                of the samples and their radii.
            success (bool): Boolean value indicating whether the algorithm
                terminated successfully.
            area_frac (float): Fraction of the space that is covered by the disks.
        """
        all_samples = self._bridson_sample()
        return self._subsample(all_samples, target_area_frac, keep_last)

    def _bridson_sample(self) -> NDArray:
        r"""
        Perform Poisson Disk Sampling using Bridson's algorithm. The algorithm
        terminates when no more samples can be generated.

        Returns:
            result (NDArray): Array of shape (M, 3) containing the coordinates
                of the samples and their radii.
        """
        grid = np.full((self.num_grid_x, self.num_grid_y, 3), -1., dtype=np.float32)
        samples = []
        active_list = []
        first_sample = np.array([self.rng.uniform(0, self.width),
                                 self.rng.uniform(0, self.height),
                                 self.r_distribution()])
        samples.append(first_sample)
        active_list.append(first_sample)
        idx = self.get_idx(first_sample[:2])
        grid[idx[0], idx[1]] = first_sample

        n_candidates = 30  # fixed
        while active_list:
            # Randomly select an active sample
            current_idx = self.rng.integers(len(active_list))
            current_sample = active_list[current_idx]
            cx, cy, cr = current_sample
            exist_valid_candidate = False

            # Generate candidate points around the current sample
            for _ in range(n_candidates):
                candidate_r = self.r_distribution()
                angle = self.rng.uniform(0, 2 * np.pi)
                dist = self.rng.uniform(candidate_r + cr, 2 * (candidate_r + cr))
                candidate_x = cx + dist * np.cos(angle)
                candidate_y = cy + dist * np.sin(angle)
                candidate = np.array([candidate_x, candidate_y, candidate_r])

                # Validate the candidate
                valid = True
                candidate_idx = self.get_idx(candidate[:2])
                n_dx = int(np.ceil((self.r_max + candidate_r) / self.cell_size_x))
                n_dy = int(np.ceil((self.r_max + candidate_r) / self.cell_size_y))
                for dx in range(-n_dx, n_dx + 1):
                    for dy in range(-n_dy, n_dy + 1):
                        neighbor_idx = ((candidate_idx[0] + dx) % self.num_grid_x,
                                        (candidate_idx[1] + dy) % self.num_grid_y)
                        neighbor = grid[neighbor_idx[0], neighbor_idx[1]]
                        if neighbor[-1] > 0:  # radius > 0, not empty
                            dist = self.dist_func(candidate[:2], neighbor[:2])
                            if dist < neighbor[-1] + candidate_r:
                                valid = False
                                break
                    if not valid:
                        break

                if valid:
                    # Accept the candidate
                    samples.append(candidate)
                    active_list.append(candidate)
                    grid[candidate_idx[0], candidate_idx[1]] = candidate
                    exist_valid_candidate = True
                    break

            # Remove the current sample from the active list if no candidate is accepted
            if not exist_valid_candidate:
                active_list.pop(current_idx)

        return np.array(samples)

    def _subsample(self,
                   all_samples: NDArray,
                   target_area_frac: float,
                   keep_last: bool) -> Tuple[NDArray, bool, float]:
        r"""
        Subsample the samples generated by Bridson's algorithm to achieve the
        target area fraction.

        Args:
            all_samples (NDArray): Array of shape (M, 3) containing the coordinates
                of the samples and their radii.
            target_area_frac (float): Fraction of the space that should be covered
                by the disks.
            keep_last (bool): If True, the last sample that exceeds the target
                area fraction is kept. If False, it is discarded.

        Returns:
            result (NDArray): Array of shape (N, 3) containing the coordinates
                of the samples and their radii.
            success (bool): Boolean value indicating whether the algorithm
                terminated successfully.
            area_frac (float): Fraction of the space that is covered by the disks.
        """
        total_area = self.width * self.height
        current_area = 0.
        selected_idx = []
        if keep_last:
            first_idx = self.rng.integers(len(all_samples))
            first_sample = all_samples[first_idx]
            selected_idx.append(first_idx)
            current_area += np.pi * first_sample[-1] ** 2
        else:
            # select the first sample such that the target area fraction is not
            # exceeded
            threshold_radius = np.sqrt(target_area_frac * total_area / np.pi)
            valid_idx = np.where(all_samples[:, -1] < threshold_radius)[0]
            if len(valid_idx) == 0:
                self.print_fn("Subsampling failed: each disk is too large. "
                              "Try to adjust the parameters or set keep_last=True.")
                return np.array(), False, current_area / total_area
            first_idx = self.rng.choice(valid_idx)
            first_sample = all_samples[first_idx]
            selected_idx.append(first_idx)
            current_area += np.pi * first_sample[-1] ** 2

        dist_table = self.dist_func(all_samples[:, :2], all_samples[:, :2])
        while current_area < target_area_frac * total_area:
            if len(selected_idx) == len(all_samples):
                self.print_fn("Subsampling failed: all samples are selected. "
                              "But the target area fraction is not reached. "
                              "Try to adjust the parameters.")
                return np.array(all_samples[selected_idx]), False, current_area / total_area
            # Find the sample that is farthest from the selected samples
            min_dist_to_selected = np.min(dist_table[:, selected_idx], axis=1)
            min_dist_to_selected[selected_idx] = -np.inf  # exclude selected samples
            idx = np.argmax(min_dist_to_selected)
            sample = all_samples[idx]
            selected_idx.append(idx)
            current_area += np.pi * sample[-1] ** 2

        if current_area > target_area_frac * total_area and not keep_last:
            # Remove the last sample
            last_idx = selected_idx.pop()
            last_sample = all_samples[last_idx]
            current_area -= np.pi * last_sample[-1] ** 2

        return np.array(all_samples[selected_idx]), True, current_area / total_area

    def get_2d_bitmap(self, samples: NDArray, resolution: Tuple[int, int]) -> NDArray:
        r"""
        Generate a 2D bitmap from the samples. The bit value is stored as float.
        Due to periodic boundary, the values on right and top edges are not stored.

        Args:
            samples (NDArray): Array of shape (N, 3) containing the coordinates
                of the samples and their radii.
            resolution (Tuple[int, int]): Resolution of the bitmap.

        Returns:
            bitmap (NDArray): 2D bitmap of shape (resolution[0], resolution[1]).
                The bit value is 0.0 for empty pixels and 1.0 for pixels covered by
                the disks.
        """
        bitmap = np.zeros(resolution, dtype=np.float32)
        pixel_width = self.width / resolution[0]
        pixel_height = self.height / resolution[1]
        def coord_to_pixel_idx(coord: NDArray) -> Tuple[int, int]:
            # coord is (2,) array
            coord = np.mod(coord, [self.width, self.height])
            return int(coord[0] / pixel_width), int(coord[1] / pixel_height)

        for sample in samples:
            coord = sample[:2]
            r = sample[-1]
            idx = coord_to_pixel_idx(coord)
            # Compute the bounding box of the disk
            n_dx = int(np.ceil(r / pixel_width))
            n_dy = int(np.ceil(r / pixel_height))
            for dx in range(-n_dx, n_dx + 1):
                for dy in range(-n_dy, n_dy + 1):
                    px = (idx[0] + dx) % resolution[0]
                    py = (idx[1] + dy) % resolution[1]
                    p_coord = np.array([px * pixel_width, py * pixel_height])
                    if self.dist_func(p_coord, coord) < r:
                        bitmap[px, py] = 1.0

        return bitmap


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser(description="Poisson Disk Sampling in 2D periodic space.")
    parser.add_argument("--width", type=float, default=1.0, help="Width of the 2D space.")
    parser.add_argument("--height", type=float, default=1.0, help="Height of the 2D space.")
    parser.add_argument("--r_min", type=float, default=0.05, help="Minimum radius of the disks.")
    parser.add_argument("--r_max", type=float, default=0.1, help="Maximum radius of the disks.")
    parser.add_argument("--target_area_frac", type=float, default=0.5,
                        help="Fraction of the space that should be covered by the disks.")
    parser.add_argument("--keep_last", action="store_true", help="Keep the last sample.")
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 512),
                        help="Resolution of the bitmap.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--image_path", type=str, default="poisson_disk.png",
                        help="Path to save the image.")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    sampler = PeriodicPoissonDiskSampler2D(args.width, args.height, args.r_min, args.r_max, rng=rng)
    samples, success, area_frac = sampler.sample(args.target_area_frac, args.keep_last)
    print(f"Success: {success}")
    print(f"Area fraction: {area_frac}")
    bitmap = sampler.get_2d_bitmap(samples, args.resolution)
    plt.imshow(bitmap, cmap="gray")
    plt.axis("off")
    plt.savefig(args.image_path)
    print(f"Image saved to {args.image_path}")
    plt.close()
