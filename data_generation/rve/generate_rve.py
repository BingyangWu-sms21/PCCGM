r"""Generate RVEs and their corresponding effective properties."""
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch
import time
from fem_solver import PeriodicBCRVE2D
from poisson_disk import PeriodicPoissonDiskSampler2D


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate RVEs and their corresponding effective properties')
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of samples to generate')
    parser.add_argument('--resolution', type=int, nargs=2, default=[64, 64],
                        help='Resolution of RVEs')
    parser.add_argument('--output', type=str, help='Path to output folder for data')
    parser.add_argument("--width", type=float, default=1.0,
                        help="Width of the 2D space.")
    parser.add_argument("--height", type=float, default=1.0,
                        help="Height of the 2D space.")
    parser.add_argument("--r_min", type=float, default=0.05,
                        help="Minimum radius of the disks.")
    parser.add_argument("--r_max", type=float, default=0.1,
                        help="Maximum radius of the disks.")
    parser.add_argument("--target_area_frac_range", type=float, nargs=2,
                        default=[0.1, 0.9], help="Range of the target area fraction.")
    parser.add_argument("--keep_last", action="store_true",
                        help="Keep the last sample.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for "
                        "the Poisson disk sampler.")
    parser.add_argument("--properties", type=float, nargs=4, default=[1., 0., 2., 0.],
                        help="List of Young's moduli and Poisson's ratios of matrix "
                        "and inclusion. The order is E_matrix, nu_matrix, E_inclusion, "
                        "nu_inclusion.")
    parser.add_argument("--save_grad", action="store_true",
                        help="Save the gradient of the effective properties with "
                        "respect to the input RVE.")
    return parser.parse_args()


def create_output_dirs(args):
    if os.path.exists(args.output):
        print("Output directory already exists. Generated data will be overwritten.")
    sub_dir = (f"domain_{args.width:.2f}x{args.height:.2f}_res_{args.resolution[0]}x{args.resolution[1]}"
               f"_r_{args.r_min:.3f}-{args.r_max:.3f}_frac_"
               f"{args.target_area_frac_range[0]:.2f}-{args.target_area_frac_range[1]:.2f}"
               f"_keep_last_{args.keep_last}"
               f"_Em_{args.properties[0]:.2f}_nu_m_{args.properties[1]:.2f}"
               f"_Ei_{args.properties[2]:.2f}_nu_i_{args.properties[3]:.2f}"
               f"_seed_{args.seed}")
    os.makedirs(args.output, exist_ok=True)
    out_dir = os.path.join(args.output, sub_dir)
    os.makedirs(os.path.join(out_dir, "rves"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "moduli"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "imgs"), exist_ok=True)
    if args.save_grad:
        os.makedirs(os.path.join(out_dir, "grads"), exist_ok=True)
    return out_dir


def compute_effective_properties(i, out_dir, properties, solver, save_grad=False):
    rve_filename = os.path.join(out_dir, "rves", f"rve_{i}.npy")
    moduli_filename = os.path.join(out_dir, "moduli", f"moduli_{i}.npy")
    if not os.path.exists(rve_filename):
        raise ValueError(f"RVE file {rve_filename} does not exist.")
    rve = np.load(rve_filename)
    rve = torch.tensor(rve, dtype=torch.float64, requires_grad=True)
    lmbda_m = properties[0] * properties[1] / ((1. + properties[1]) * (1. - 2. * properties[1]))
    mu_m = properties[0] / (2. * (1. + properties[1]))
    lmbda_i = properties[2] * properties[3] / ((1. + properties[3]) * (1. - 2. * properties[3]))
    mu_i = properties[2] / (2. * (1. + properties[3]))
    lmbda = (1. - rve) * lmbda_m + rve * lmbda_i
    lmbda = lmbda.unsqueeze(0)
    mu = (1. - rve) * mu_m + rve * mu_i
    mu = mu.unsqueeze(0)
    try:
        _, moduli = solver.run_and_process(lmbda, mu)
    except Exception as e:
        print(f"Failed to compute effective properties for RVE {i}. Error: {e}")
    moduli = moduli.squeeze(0)
    np.save(moduli_filename, moduli.detach().numpy())
    if save_grad:
        grad_filename = os.path.join(out_dir, "grads", f"grad_{i}.npy")
        grad_tensor = torch.zeros(moduli.shape + rve.shape)  # (3, 3, *resolution)
        for i in range(moduli.shape[0]):
            for j in range(moduli.shape[1]):
                moduli[i, j].backward(retain_graph=True)
                grad_tensor[i, j] = rve.grad
                rve.grad.zero_()
        np.save(grad_filename, grad_tensor.detach().numpy())
        if i < 100:
            plt.imshow(grad_tensor[0, 0], cmap="gray")
            plt.axis("off")
            grad_img_filename = os.path.join(out_dir, "imgs", f"grad_{i}.png")
            plt.savefig(grad_img_filename, bbox_inches="tight", pad_inches=0)
            plt.close()


def main():
    args = parse_args()
    out_dir = create_output_dirs(args)
    start_time = time.time()
    print(f"Output directory: {out_dir}")
    rng = np.random.default_rng(seed=args.seed)
    sampler = PeriodicPoissonDiskSampler2D(
        args.width, args.height, args.r_min, args.r_max, rng=rng, print_fn=None)
    solver = PeriodicBCRVE2D(
        corner=(args.width, args.height),
        n_cells=args.resolution,
        global_strain_list = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)])
    current_samples = 0
    current_trails = 0
    while current_samples < args.samples:
        target_area_frac = rng.uniform(*args.target_area_frac_range)
        samples, success, area_frac = sampler.sample(target_area_frac, args.keep_last)
        if success:
            bitmap = sampler.get_2d_bitmap(samples, args.resolution)
            if current_samples < 100:
                plt.imshow(bitmap, cmap="gray")
                plt.axis("off")
                img_filename = os.path.join(out_dir, "imgs", f"rve_{current_samples}.png")
                plt.savefig(img_filename, bbox_inches="tight", pad_inches=0)
                plt.close()
            npy_filename = os.path.join(out_dir, "rves", f"rve_{current_samples}.npy")
            np.save(npy_filename, bitmap)
            try:
                compute_effective_properties(current_samples, out_dir, args.properties, solver,
                                             save_grad=args.save_grad)
                current_samples += 1
                if current_samples % max(10, (args.samples // 100)) == 0:
                    print(f"Generated {current_samples} / {args.samples} samples")
            except Exception as e:
                print(f"Failed to compute effective properties for RVE {current_samples}. Error: {e}")
        current_trails += 1
        if current_trails % max(20, (args.samples // 50)) == 0:
            print(f"Generated {current_samples} / {args.samples} samples after "
                  f"{current_trails} trails")

        if current_trails > 10 * args.samples:
            raise ValueError("Failed to generate enough samples. The success rate "
                             "of the Poisson disk sampler and FEM solver is too low. "
                             "Please adjust the parameters.")

    end_time = time.time()
    print(f"Finished generation in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
