#!/usr/bin/env python3
"""Batch evaluation of SVD Best Rank-k image compression on USC-SIPI TIFF images.

Expected input layout:
    USC_SIPI_Dataset/
        A/              # 20 grayscale TIFF images
        B/              # 20 grayscale TIFF images
        C/              # 10 RGB TIFF images

Folder names Group_A / Group_B / Group_C are also accepted.

The script:
- validates TIFF files, image dimensions and grayscale/RGB channel type;
- subtracts each channel mean, computes one SVD per centered image channel, and stores the mean;
- finds the minimum k for centered-energy thresholds from 99% down to 90% in 1% steps;
- reconstructs images and computes MSE, PSNR and SSIM;
- saves float32 SVD factors in uncompressed NPZ archives;
- reports theoretical and actual factor-storage ratios;
- exports CSV files, TIFF reconstructions, comparison images and plots.

Install dependencies:
    pip install -r requirements.txt

Run:
    python run_svd_experiment.py --input "./USC_SIPI_Dataset" --output "./SVD_Results"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Allows saving figures without opening GUI windows.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from skimage.metrics import structural_similarity


# ----------------------------- Experiment settings -----------------------------

ENERGY_LEVELS = tuple(percent / 100 for percent in range(99, 89, -1))  # 99%, 98%, ..., 90%
EXPECTED_COUNTS = {"A": 20, "B": 20, "C": 10}
GROUP_TYPES = {"A": "gray", "B": "gray", "C": "rgb"}
TIFF_EXTENSIONS = {".tif", ".tiff"}
BYTES_PER_FACTOR = 4  # float32


@dataclass
class ImageRecord:
    group: str
    path: Path
    relative_path: str
    image_id: str
    image: np.ndarray
    original_mode: str
    normalized_mode: str
    original_file_bytes: int
    sha256: str


def safe_name(value: str) -> str:
    """Return a filesystem-safe name while retaining useful filename text."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("._")
    return value or "image"


def resolve_group_dirs(input_dir: Path) -> dict[str, Path]:
    """Find A/B/C folders, accepting either A or Group_A naming."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {input_dir}")

    children = {p.name.lower(): p for p in input_dir.iterdir() if p.is_dir()}
    result: dict[str, Path] = {}
    aliases = {
        "A": ("a", "group_a", "group a"),
        "B": ("b", "group_b", "group b"),
        "C": ("c", "group_c", "group c"),
    }
    for group, names in aliases.items():
        found = next((children[name] for name in names if name in children), None)
        if found is None:
            raise FileNotFoundError(
                f"Không tìm thấy thư mục nhóm {group}. Hãy tạo {group}/ hoặc Group_{group}/ "
                f"trong {input_dir}."
            )
        result[group] = found
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tiff(path: Path, expected_kind: str, expected_size: tuple[int, int]) -> tuple[np.ndarray, str, str]:
    """Load and normalize a TIFF without resizing it.

    Grayscale groups accept L/1/I/I;16/F modes and convert them to 8-bit L.
    RGB group accepts common color modes and converts them to 8-bit RGB.
    The original mode is retained in the validation report.
    """
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("TIFF nhiều trang; chương trình chỉ xử lý ảnh TIFF một trang")

        actual_size = image.size  # (width, height)
        if actual_size != expected_size:
            raise ValueError(
                f"Sai kích thước: nhận {actual_size[0]}x{actual_size[1]}, "
                f"yêu cầu {expected_size[0]}x{expected_size[1]}; không tự resize"
            )

        original_mode = image.mode
        if expected_kind == "gray":
            grayscale_modes = {"1", "L", "I", "I;16", "I;16B", "I;16L", "F"}
            if original_mode not in grayscale_modes:
                raise ValueError(
                    f"Nhóm yêu cầu ảnh xám nhưng TIFF có mode {original_mode}; "
                    "hãy kiểm tra lại cách phân nhóm"
                )
            normalized = image.convert("L")
            array = np.asarray(normalized, dtype=np.uint8)
            normalized_mode = "L"
        else:
            color_modes = {"RGB", "RGBA", "CMYK", "YCbCr", "LAB", "HSV", "P"}
            if original_mode not in color_modes:
                raise ValueError(
                    f"Nhóm yêu cầu ảnh màu RGB nhưng TIFF có mode {original_mode}; "
                    "hãy kiểm tra lại cách phân nhóm"
                )
            normalized = image.convert("RGB")
            array = np.asarray(normalized, dtype=np.uint8)
            normalized_mode = "RGB"

    expected_shape = (expected_size[1], expected_size[0])
    if expected_kind == "gray" and array.shape != expected_shape:
        raise ValueError(f"Ma trận ảnh xám không đúng kích thước: {array.shape}")
    if expected_kind == "rgb" and array.shape != expected_shape + (3,):
        raise ValueError(f"Ảnh màu không có đúng 3 kênh RGB: {array.shape}")

    return array, original_mode, normalized_mode


def discover_and_validate(
    input_dir: Path,
    expected_size: tuple[int, int],
    validation_dir: Path,
) -> tuple[list[ImageRecord], list[dict[str, Any]], list[str]]:
    group_dirs = resolve_group_dirs(input_dir)
    records: list[ImageRecord] = []
    validation_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_hashes: dict[str, str] = {}

    for group in ("A", "B", "C"):
        group_dir = group_dirs[group]
        paths = sorted(
            p for p in group_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in TIFF_EXTENSIONS
        )
        if len(paths) != EXPECTED_COUNTS[group]:
            warnings.append(
                f"Nhóm {group}: tìm thấy {len(paths)} TIFF, trong khi kế hoạch là "
                f"{EXPECTED_COUNTS[group]} ảnh. Chương trình sẽ xử lý các ảnh hợp lệ tìm được."
            )

        for path in paths:
            relative = path.relative_to(group_dir).as_posix()
            relative_stem = str(Path(relative).with_suffix(""))
            image_id = safe_name(relative_stem.replace("/", "_"))
            row: dict[str, Any] = {
                "group": group,
                "file": str(path),
                "relative_file": relative,
                "image_id": image_id,
                "status": "REJECTED",
                "reason": "",
                "original_mode": "",
                "normalized_mode": "",
                "width": None,
                "height": None,
                "file_bytes": path.stat().st_size,
                "sha256": "",
                "duplicate_of": "",
            }
            try:
                array, original_mode, normalized_mode = load_tiff(
                    path, GROUP_TYPES[group], expected_size
                )
                file_hash = sha256_file(path)
                duplicate_of = seen_hashes.get(file_hash, "")
                if not duplicate_of:
                    seen_hashes[file_hash] = f"{group}/{relative}"
                row.update({
                    "status": "OK",
                    "reason": "",
                    "original_mode": original_mode,
                    "normalized_mode": normalized_mode,
                    "width": expected_size[0],
                    "height": expected_size[1],
                    "sha256": file_hash,
                    "duplicate_of": duplicate_of,
                })
                if duplicate_of:
                    warnings.append(
                        f"Ảnh {group}/{relative} có nội dung TIFF trùng SHA-256 với {duplicate_of}."
                    )
                if original_mode != normalized_mode:
                    warnings.append(
                        f"Ảnh {group}/{relative}: chuyển mode {original_mode} sang {normalized_mode}."
                    )
                records.append(ImageRecord(
                    group=group,
                    path=path,
                    relative_path=relative,
                    image_id=image_id,
                    image=array,
                    original_mode=original_mode,
                    normalized_mode=normalized_mode,
                    original_file_bytes=path.stat().st_size,
                    sha256=file_hash,
                ))
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                row["reason"] = str(exc)
            validation_rows.append(row)

    validation_dir.mkdir(parents=True, exist_ok=True)
    # Only rejected files are written to validation.csv; accepted files are omitted.
    rejected_rows = [row for row in validation_rows if row["status"] != "OK"]
    validation_columns = [
        "group", "file", "relative_file", "image_id", "status", "reason",
        "original_mode", "normalized_mode", "width", "height",
        "file_bytes", "sha256", "duplicate_of",
    ]
    validation_path = validation_dir / "validation.csv"
    validation_df = pd.DataFrame(rejected_rows, columns=validation_columns)
    try:
        validation_df.to_csv(validation_path, index=False, encoding="utf-8-sig")
    except PermissionError:
        # If validation.csv is open in Excel, preserve the report under a new name.
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        validation_path = validation_dir / f"validation_rejected_{timestamp}.csv"
        validation_df.to_csv(validation_path, index=False, encoding="utf-8-sig")
        warnings.append(
            "Không ghi đè được validation.csv (có thể đang mở trong Excel); "
            f"đã lưu danh sách ảnh bị loại tại {validation_path.name}."
        )
    return records, validation_rows, warnings


def decompose_channels(
    image: np.ndarray,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]], float]:
    """Compute reduced SVD after subtracting the mean intensity of each channel.

    The channel mean is stored separately and added back during reconstruction.
    Retained energy therefore measures variation around the mean, rather than
    allowing the overall brightness (DC component) to dominate the percentage.
    """
    if image.ndim == 2:
        channel_arrays = {"GRAY": image}
    else:
        channel_arrays = {name: image[:, :, index] for index, name in enumerate(("R", "G", "B"))}

    factors: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    start = time.perf_counter()
    for name, channel in channel_arrays.items():
        matrix = channel.astype(np.float64, copy=False)
        channel_mean = float(np.mean(matrix))
        centered_matrix = matrix - channel_mean
        U, singular_values, Vt = np.linalg.svd(centered_matrix, full_matrices=False)
        factors[name] = (U, singular_values, Vt, channel_mean)
    elapsed = time.perf_counter() - start
    return factors, elapsed


def choose_rank(singular_values: np.ndarray, eta: float) -> int:
    total_energy = float(np.sum(singular_values * singular_values))
    if total_energy <= 0.0:
        return 0
    cumulative = np.cumsum(singular_values * singular_values)
    return int(np.searchsorted(cumulative, eta * total_energy, side="left") + 1)


def reconstruct_from_float32_factors(
    factors: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    eta: float,
) -> tuple[np.ndarray, dict[str, int], dict[str, np.ndarray], float, int]:
    """Select ranks using mean-centered energy and reconstruct with float32 factors.

    The mean of each channel is stored as float32 and added back after the low-rank
    reconstruction. Metrics therefore evaluate the actual saved representation.
    """
    reconstructed_channels: dict[str, np.ndarray] = {}
    ranks: dict[str, int] = {}
    saved_arrays: dict[str, np.ndarray] = {}
    total_energy = 0.0
    retained_energy = 0.0
    theoretical_bytes = 0

    for name, (U, singular_values, Vt, channel_mean) in factors.items():
        squared = singular_values * singular_values
        total_energy += float(np.sum(squared))
        k = choose_rank(singular_values, eta)
        ranks[name] = k

        U_k = U[:, :k].astype(np.float32)
        s_k = singular_values[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)
        mean_k = np.asarray(channel_mean, dtype=np.float32)
        saved_arrays[f"U_{name}"] = U_k
        saved_arrays[f"s_{name}"] = s_k
        saved_arrays[f"Vt_{name}"] = Vt_k
        saved_arrays[f"mean_{name}"] = mean_k

        if k > 0:
            retained_energy += float(np.sum(squared[:k]))
            centered_reconstruction = (
                U_k.astype(np.float64) * s_k.astype(np.float64)
            ) @ Vt_k.astype(np.float64)
        else:
            centered_reconstruction = np.zeros((U.shape[0], Vt.shape[1]), dtype=np.float64)

        # Add back the channel mean that was removed before SVD.
        reconstructed_channels[name] = centered_reconstruction + float(mean_k)
        theoretical_bytes += BYTES_PER_FACTOR * (
            U.shape[0] * k + k + k * Vt.shape[1] + 1
        )

    achieved_energy = retained_energy / total_energy if total_energy > 0 else 1.0
    if "GRAY" in reconstructed_channels:
        reconstructed = reconstructed_channels["GRAY"]
    else:
        reconstructed = np.stack(
            [reconstructed_channels[name] for name in ("R", "G", "B")], axis=-1
        )

    reconstructed_u8 = np.rint(np.clip(reconstructed, 0, 255)).astype(np.uint8)
    return reconstructed_u8, ranks, saved_arrays, achieved_energy, theoretical_bytes


def calculate_metrics(original: np.ndarray, reconstructed: np.ndarray) -> tuple[float, float, float]:
    original_f = original.astype(np.float64)
    reconstructed_f = reconstructed.astype(np.float64)
    mse = float(np.mean((original_f - reconstructed_f) ** 2))
    psnr = math.inf if mse == 0 else float(10.0 * math.log10((255.0 ** 2) / mse))

    if original.ndim == 2:
        ssim = float(structural_similarity(original, reconstructed, data_range=255))
    else:
        ssim = float(structural_similarity(
            original, reconstructed, data_range=255, channel_axis=-1
        ))
    return mse, psnr, ssim


def save_reconstruction(path: Path, image_array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_array).save(path, format="TIFF")


def format_rank_label(ranks: dict[str, int]) -> str:
    """Create a compact k annotation for a grayscale or RGB reconstruction."""
    if "GRAY" in ranks:
        return f"k = {ranks['GRAY']}"

    # RGB is decomposed channel-by-channel, so each channel has its own rank.
    # Show the three ranks and their sum to avoid hiding this distinction.
    k_r = ranks.get("R", 0)
    k_g = ranks.get("G", 0)
    k_b = ranks.get("B", 0)
    return f"kR={k_r}, kG={k_g}, kB={k_b}\nΣk={k_r + k_g + k_b}"


def rank_filename_suffix(ranks: dict[str, int]) -> str:
    """Return k values for the reconstruction filename."""
    if "GRAY" in ranks:
        return f"k{ranks['GRAY']}"
    return f"kR{ranks.get('R', 0)}_kG{ranks.get('G', 0)}_kB{ranks.get('B', 0)}"


def plot_comparison(
    original: np.ndarray,
    reconstructions: dict[float, np.ndarray],
    reconstruction_ranks: dict[float, dict[str, int]],
    output_path: Path,
    title: str,
) -> None:
    """Save a montage with target energy and selected k shown above each image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = [("Original", original)] + [
        (
            f"Energy {eta:.0%}\n{format_rank_label(reconstruction_ranks[eta])}",
            reconstructions[eta],
        )
        for eta in ENERGY_LEVELS
    ]
    columns = 4
    rows = math.ceil(len(images) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 4.0 * rows), squeeze=False)
    flat_axes = axes.ravel()
    for axis, (label, array) in zip(flat_axes, images):
        if array.ndim == 2:
            axis.imshow(array, cmap="gray", vmin=0, vmax=255)
        else:
            axis.imshow(array)
        axis.set_title(label, fontsize=9, pad=7)
        axis.axis("off")
    for axis in flat_axes[len(images):]:
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spectrum(
    factors: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, (_, singular_values, _, _) in factors.items():
        indices = np.arange(1, len(singular_values) + 1)
        axes[0].semilogy(indices, np.maximum(singular_values, np.finfo(float).tiny), label=name)
        squared = singular_values * singular_values
        total = float(np.sum(squared))
        cumulative = np.cumsum(squared) / total if total > 0 else np.zeros_like(squared)
        axes[1].plot(indices, cumulative, label=name)

    axes[0].set_title("Singular-value spectrum")
    axes[0].set_xlabel("Index i")
    axes[0].set_ylabel("Singular value (log scale)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].set_title("Cumulative mean-centered energy")
    axes[1].set_xlabel("Rank k")
    axes[1].set_ylabel("Retained mean-centered energy")
    axes[1].set_ylim(0, 1.02)
    for eta in ENERGY_LEVELS:
        axes[1].axhline(eta, linestyle="--", linewidth=0.8, label=f"{eta:.0%}")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_summary_plots(summary: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("mean_k_per_channel", "Mean retained rank per channel", "Mean k"),
        ("mean_rank_fraction", "Mean retained rank fraction", "k / 512"),
        ("mse", "Mean MSE by group", "MSE"),
        ("psnr", "Mean PSNR by group", "PSNR (dB)"),
        ("ssim", "Mean SSIM by group", "SSIM"),
        ("theoretical_ratio_vs_raw", "Theoretical factor size / raw image size", "Ratio"),
        ("actual_ratio_vs_tiff", "Actual factor archive / source TIFF size", "Ratio"),
    ]

    for metric, title, ylabel in plot_specs:
        if metric not in summary.columns:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for group in ("A", "B", "C"):
            subset = summary[summary["group"] == group].sort_values("eta_target")
            if not subset.empty:
                ax.plot(
                    subset["eta_target"] * 100,
                    subset[metric],
                    marker="o",
                    linewidth=1.5,
                    label=f"Group {group}",
                )
        ax.set_title(title)
        ax.set_xlabel("Target retained energy (%)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f"{metric}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def run_experiment(
    input_dir: Path,
    output_dir: Path,
    expected_size: tuple[int, int],
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_dir = output_dir / "validation"
    records, validation_rows, warnings = discover_and_validate(
        input_dir, expected_size, validation_dir
    )

    rejected_rows = [row for row in validation_rows if row["status"] != "OK"]
    print(
        f"Đã kiểm tra {len(validation_rows)} tệp TIFF: "
        f"{len(records)} ảnh hợp lệ, {len(rejected_rows)} ảnh bị loại."
    )
    if rejected_rows:
        print("Danh sách ảnh bị loại:")
        for row in rejected_rows:
            print(
                f"  - Nhóm {row['group']} | {row['relative_file']} | "
                f"Lý do: {row['reason']}"
            )
    for warning in warnings:
        print(f"CẢNH BÁO: {warning}")

    if not records:
        print("Không có ảnh hợp lệ để xử lý. Xem validation/validation.csv để biết lỗi.")
        return 2

    recon_dir = output_dir / "reconstructions"
    factor_dir = output_dir / "svd_factors"
    comparison_dir = output_dir / "comparisons"
    spectrum_dir = output_dir / "spectra"
    plots_dir = output_dir / "plots"
    for folder in (recon_dir, factor_dir, comparison_dir, spectrum_dir, plots_dir):
        folder.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, str]] = []

    for image_index, record in enumerate(records, start=1):
        print(f"[{image_index}/{len(records)}] {record.group}/{record.relative_path}")
        try:
            factors, decomposition_time = decompose_channels(record.image)
            reconstructions: dict[float, np.ndarray] = {}
            reconstruction_ranks: dict[float, dict[str, int]] = {}
            image_slug = safe_name(record.image_id)
            channel_count = 1 if record.image.ndim == 2 else 3
            height, width = record.image.shape[:2]
            raw_pixel_bytes = int(record.image.size)  # uint8 -> one byte per value

            # Save one spectrum plot per original image.
            plot_spectrum(
                factors,
                spectrum_dir / record.group / f"{image_slug}_spectrum.png",
                f"Group {record.group} – {record.relative_path}",
            )

            for eta in ENERGY_LEVELS:
                reconstruction_start = time.perf_counter()
                reconstructed, ranks, saved_arrays, achieved_energy, theoretical_bytes = (
                    reconstruct_from_float32_factors(factors, eta)
                )
                reconstruction_time = time.perf_counter() - reconstruction_start
                reconstructions[eta] = reconstructed
                reconstruction_ranks[eta] = ranks.copy()

                eta_label = f"{int(round(eta * 100)):02d}"
                k_suffix = rank_filename_suffix(ranks)
                relative_output = Path(record.group) / (
                    f"{image_slug}_eta{eta_label}_{k_suffix}.tif"
                )
                save_reconstruction(recon_dir / relative_output, reconstructed)

                factor_path = factor_dir / record.group / f"{image_slug}_eta{eta_label}.npz"
                factor_path.parent.mkdir(parents=True, exist_ok=True)
                saved_arrays["source_shape"] = np.asarray(record.image.shape, dtype=np.int32)
                saved_arrays["eta_target"] = np.asarray(eta, dtype=np.float32)
                np.savez(factor_path, **saved_arrays)  # Uncompressed archive; factors are float32.
                factor_file_bytes = factor_path.stat().st_size

                mse, psnr, ssim = calculate_metrics(record.image, reconstructed)
                k_values = list(ranks.values())
                k_total = int(sum(k_values))
                mean_k_per_channel = float(k_total / channel_count)
                rank_fraction = float(k_total / (min(height, width) * channel_count))
                theoretical_ratio = theoretical_bytes / raw_pixel_bytes if raw_pixel_bytes else math.nan
                actual_ratio_vs_tiff = (
                    factor_file_bytes / record.original_file_bytes
                    if record.original_file_bytes else math.nan
                )
                actual_ratio_vs_raw = factor_file_bytes / raw_pixel_bytes if raw_pixel_bytes else math.nan

                row: dict[str, Any] = {
                    "group": record.group,
                    "image": record.path.name,
                    "image_id": record.image_id,
                    "source_relative_path": record.relative_path,
                    "channels": channel_count,
                    "width": width,
                    "height": height,
                    "eta_target": eta,
                    "eta_achieved": achieved_energy,
                    "energy_basis": "mean_centered_per_channel",
                    "mean_GRAY": factors["GRAY"][3] if "GRAY" in factors else np.nan,
                    "mean_R": factors["R"][3] if "R" in factors else np.nan,
                    "mean_G": factors["G"][3] if "G" in factors else np.nan,
                    "mean_B": factors["B"][3] if "B" in factors else np.nan,
                    "k": ranks.get("GRAY", np.nan),
                    "k_R": ranks.get("R", np.nan),
                    "k_G": ranks.get("G", np.nan),
                    "k_B": ranks.get("B", np.nan),
                    "k_total": k_total,
                    "mean_k_per_channel": mean_k_per_channel,
                    "rank_fraction": rank_fraction,
                    "mse": mse,
                    "psnr_db": psnr,
                    "ssim": ssim,
                    "raw_pixel_bytes": raw_pixel_bytes,
                    "source_tiff_bytes": record.original_file_bytes,
                    "theoretical_factor_bytes_float32": theoretical_bytes,
                    "theoretical_ratio_vs_raw": theoretical_ratio,
                    "theoretical_savings_pct_vs_raw": (1.0 - theoretical_ratio) * 100.0,
                    "factor_archive_bytes": factor_file_bytes,
                    "actual_ratio_vs_tiff": actual_ratio_vs_tiff,
                    "actual_ratio_vs_raw": actual_ratio_vs_raw,
                    "decomposition_time_s": decomposition_time,
                    "reconstruction_time_s": reconstruction_time,
                    "factor_file": str(factor_path.relative_to(output_dir)),
                    "reconstruction_file": str((recon_dir / relative_output).relative_to(output_dir)),
                }
                result_rows.append(row)

            plot_comparison(
                record.image,
                reconstructions,
                reconstruction_ranks,
                comparison_dir / record.group / f"{image_slug}_comparison.png",
                f"Group {record.group} – {record.relative_path}",
            )

        except Exception as exc:  # Keep processing the remaining images after a failure.
            error_rows.append({
                "group": record.group,
                "file": str(record.path),
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"  LỖI: {type(exc).__name__}: {exc}")

    results = pd.DataFrame(result_rows)
    if not results.empty:
        results = results.sort_values(
            ["group", "image", "eta_target"], ascending=[True, True, False]
        )
        results.to_csv(output_dir / "results.csv", index=False, encoding="utf-8-sig")

        summary = (
            results.groupby(["group", "eta_target"], as_index=False)
            .agg(
                image_count=("image_id", "nunique"),
                mean_k_per_channel=("mean_k_per_channel", "mean"),
                std_k_per_channel=("mean_k_per_channel", "std"),
                mean_rank_fraction=("rank_fraction", "mean"),
                std_rank_fraction=("rank_fraction", "std"),
                mse=("mse", "mean"),
                mse_std=("mse", "std"),
                psnr=("psnr_db", "mean"),
                psnr_std=("psnr_db", "std"),
                ssim=("ssim", "mean"),
                ssim_std=("ssim", "std"),
                theoretical_ratio_vs_raw=("theoretical_ratio_vs_raw", "mean"),
                actual_ratio_vs_tiff=("actual_ratio_vs_tiff", "mean"),
                actual_ratio_vs_raw=("actual_ratio_vs_raw", "mean"),
                eta_achieved=("eta_achieved", "mean"),
            )
        )
        summary = summary.sort_values(
            ["group", "eta_target"], ascending=[True, False]
        )
        summary.to_csv(output_dir / "group_summary.csv", index=False, encoding="utf-8-sig")
        make_summary_plots(summary, plots_dir)
    else:
        summary = pd.DataFrame()

    pd.DataFrame(error_rows, columns=["group", "file", "error"]).to_csv(
        output_dir / "errors.csv", index=False, encoding="utf-8-sig"
    )

    actual_counts = {group: sum(1 for record in records if record.group == group) for group in ("A", "B", "C")}
    successful_counts = {
        group: int(results[results["group"] == group]["image_id"].nunique()) if not results.empty else 0
        for group in ("A", "B", "C")
    }
    summary_text = [
        "SVD BEST RANK-k – EXPERIMENT SUMMARY",
        f"Input directory: {input_dir.resolve()}",
        f"Expected image size: {expected_size[0]}x{expected_size[1]}",
        f"Energy thresholds: {', '.join(f'{x:.0%}' for x in ENERGY_LEVELS)}",
        f"Valid input images: {len(records)}",
        f"Successful images: {sum(successful_counts.values())}",
        f"Failed images during SVD: {len(error_rows)}",
        "",
        "Counts by group (valid input / successfully processed):",
    ]
    for group in ("A", "B", "C"):
        summary_text.append(
            f"  Group {group}: {actual_counts[group]} / {successful_counts[group]} "
            f"(planned {EXPECTED_COUNTS[group]})"
        )
    summary_text += ["", "Warnings:"] + ([f"  - {w}" for w in warnings] or ["  None"])
    summary_text += [
        "",
        "Important interpretation notes:",
        "  - SVD is applied to each channel after subtracting its mean intensity.",
        "  - The stored channel mean is added back after low-rank reconstruction.",
        "  - eta_target and eta_achieved refer to mean-centered energy (contrast variation), not total raw pixel energy.",
        "  - k is selected independently per channel at the requested mean-centered energy threshold.",
        "  - RGB MSE is averaged over all pixels and all three channels.",
        "  - PSNR uses MAX=255; SSIM uses data_range=255.",
        "  - Theoretical factor size assumes float32 coefficients and excludes file headers.",
        "  - Actual factor archive size is an uncompressed NPZ containing float32 factors.",
        "  - actual_ratio_vs_tiff compares the factor archive with the source TIFF file size.",
        "  - Reconstruction TIFF files are for visual assessment, not the compressed payload.",
        "  - validation/validation.csv contains only rejected files, with group and reason.",
        "  - Retained-energy thresholds are tested from 99% down to 90% in 1% steps.",
    ]
    (output_dir / "run_summary.txt").write_text("\n".join(summary_text), encoding="utf-8")

    print("\nHoàn tất.")
    print(f"Kết quả: {output_dir.resolve()}")
    print(f"- results.csv: {len(results)} dòng kết quả")
    print("- group_summary.csv: thống kê trung bình và độ lệch chuẩn")
    print("- validation/validation.csv: chỉ liệt kê ảnh bị loại, kèm nhóm và lý do")
    print("- reconstructions/: ảnh TIFF khôi phục")
    print("- svd_factors/: bộ hệ số SVD float32")
    print("- comparisons/: ảnh so sánh gốc và khôi phục")
    print("- spectra/: phổ giá trị kỳ dị và năng lượng tích lũy")
    print("- plots/: biểu đồ so sánh giữa các nhóm")
    print("- errors.csv và run_summary.txt: lỗi và tổng kết")

    return 0 if not error_rows else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Đánh giá SVD Best Rank-k trên bộ ảnh TIFF USC-SIPI."
    )
    parser.add_argument(
        "--input", type=Path, default=Path("USC_SIPI_Dataset"),
        help="Thư mục chứa A/, B/, C/ (mặc định: USC_SIPI_Dataset).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("SVD_Results"),
        help="Thư mục lưu kết quả (mặc định: SVD_Results).",
    )
    parser.add_argument("--width", type=int, default=512, help="Chiều rộng ảnh yêu cầu (mặc định 512).")
    parser.add_argument("--height", type=int, default=512, help="Chiều cao ảnh yêu cầu (mặc định 512).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        print("Chiều rộng và chiều cao phải là số nguyên dương.", file=sys.stderr)
        return 2
    return run_experiment(args.input, args.output, (args.width, args.height))


if __name__ == "__main__":
    raise SystemExit(main())
