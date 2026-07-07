"""
2024A 板凳龙 第三问：最小螺距代码（快速可执行版）

运行：
    python question3.py

核心思想：
    对给定螺距 p，龙头前把手从初始第 16 圈向内盘入到调头空间边界 r=4.5m。
    在这个过程中不断计算板凳矩形之间的 SAT 净间隙 G。
    若全过程 min(G) >= 0，则该螺距安全；否则会提前碰撞。
    再对螺距 p 做二分，得到最小安全螺距。

为了兼顾速度和稳定性：
    1. 先用全板凳对的粗检查自动发现最危险的候选板凳对；
    2. 随后的螺距二分只跟踪这些候选对及其邻近对，速度很快；
    3. 最后再用全板凳对在临界半径附近做验证。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dragon_data
import utils


# ==========================
# 可调参数
# ==========================
TURNING_RADIUS = 4.5
INITIAL_THETA = float(dragon_data.QUESTION1_INITIAL_THETA)  # 32π，即第 16 圈

# 螺距粗扫描范围，单位 m。第三问结果在 0.45m 附近。
PITCH_SCAN_MIN = 0.35
PITCH_SCAN_MAX = 0.60
PITCH_SCAN_STEP = 0.01
PITCH_TOL = 1e-6

# 对单个螺距，沿龙头半径 r 扫描全过程。
RADIUS_COARSE_STEP = 0.05
RADIUS_REFINE_STEP = 0.002
LOWEST_RADIUS_WINDOW_COUNT = 5
REFINE_RADIUS_WINDOW = 0.08

# 自动发现危险板凳对时，只需要检查边界附近一小段，因为第三问临界碰撞出现在接近 r=4.5m 的地方。
DISCOVER_PITCHES = (0.44, 0.45, 0.46)
DISCOVER_RADIUS_MAX = 4.85
DISCOVER_RADIUS_STEP = 0.025
NEIGHBOR_PAIR_RADIUS = 1

# 最后全板凳对验证窗口。
VALIDATE_RADIUS_WINDOW = 0.15
VALIDATE_RADIUS_STEP = 0.01


@dataclass
class StateByRadius:
    pitch: float
    head_radius: float
    theta: np.ndarray
    theta_dot: np.ndarray
    radius: np.ndarray
    position: np.ndarray
    speed: np.ndarray


@dataclass
class GapResult:
    pitch: float
    head_radius: float
    global_gap: float
    best_pair: Tuple[int, int]
    state: Optional[StateByRadius] = None


# ==========================
# 快速计算把手位置和速度
# ==========================

def _point_distance_sq(theta_a: float, theta_b: float, b: float) -> float:
    radius_a = b * theta_a
    radius_b = b * theta_b
    delta = theta_a - theta_b
    return float(radius_a * radius_a + radius_b * radius_b - 2.0 * radius_a * radius_b * np.cos(delta))


def solve_trailing_theta_fast(theta_prev: float, distance: float, b: float, tol: float = 1e-12) -> float:
    """由前一把手参数 theta_prev 快速求后一把手参数 theta。"""
    theta_prev = float(theta_prev)
    distance = float(distance)
    b = float(b)

    theta = theta_prev + distance / (b * np.sqrt(1.0 + theta_prev * theta_prev))

    for _ in range(12):
        delta = theta - theta_prev
        f = b * b * (
            theta_prev * theta_prev
            + theta * theta
            - 2.0 * theta_prev * theta * np.cos(delta)
        ) - distance * distance
        if abs(f) < 1e-13:
            return float(theta)

        derivative = b * b * (
            2.0 * theta
            - 2.0 * theta_prev * np.cos(delta)
            + 2.0 * theta_prev * theta * np.sin(delta)
        )
        if derivative <= 0.0 or not np.isfinite(derivative):
            break

        new_theta = theta - f / derivative
        if new_theta <= theta_prev or not np.isfinite(new_theta):
            break
        if abs(new_theta - theta) < tol:
            return float(new_theta)
        theta = new_theta

    # 牛顿法失败时回退到稳定二分。
    arc_step = distance / (b * np.sqrt(1.0 + theta_prev * theta_prev))
    low = theta_prev
    high = theta_prev + max(2.0 * arc_step, 1e-4)
    while _point_distance_sq(high, theta_prev, b) < distance * distance:
        high += max(arc_step, 1e-4)
        arc_step *= 1.5

    for _ in range(80):
        mid = 0.5 * (low + high)
        if _point_distance_sq(mid, theta_prev, b) < distance * distance:
            low = mid
        else:
            high = mid
        if high - low < tol:
            break
    return float(0.5 * (low + high))


def solve_state_by_head_radius(pitch: float, head_radius: float, point_count: Optional[int] = None) -> StateByRadius:
    """
    给定螺距 pitch 和龙头半径 head_radius，计算前 point_count 个把手的位置速度。
    若 point_count=None，则计算全部 224 个把手。
    """
    pitch = float(pitch)
    head_radius = float(head_radius)
    b = utils.spiral_coefficient(pitch)
    distances = dragon_data.get_handle_distances()

    if point_count is None:
        point_count = dragon_data.POINT_COUNT
    point_count = int(point_count)
    if point_count < 2 or point_count > dragon_data.POINT_COUNT:
        raise ValueError("point_count 必须在 [2, POINT_COUNT] 内")

    theta = np.zeros(point_count, dtype=float)
    theta_dot = np.zeros(point_count, dtype=float)
    radius = np.zeros(point_count, dtype=float)
    position = np.zeros((point_count, 2), dtype=float)
    speed = np.zeros(point_count, dtype=float)

    theta[0] = head_radius / b
    for i in range(1, point_count):
        theta[i] = solve_trailing_theta_fast(theta[i - 1], distances[i - 1], b)

    radius[:] = b * theta
    position[:, 0], position[:, 1] = utils.polar_to_cartesian(radius, theta)

    theta_dot[0] = -1.0 / (b * np.sqrt(1.0 + theta[0] * theta[0]))
    for i in range(1, point_count):
        delta = theta[i] - theta[i - 1]
        numerator = theta[i - 1] - theta[i] * np.cos(delta) - theta[i] * theta[i - 1] * np.sin(delta)
        denominator = theta[i] - theta[i - 1] * np.cos(delta) + theta[i] * theta[i - 1] * np.sin(delta)
        theta_dot[i] = -(numerator / denominator) * theta_dot[i - 1]

    speed[:] = utils.speed_from_theta(theta, theta_dot, b)
    return StateByRadius(pitch, head_radius, theta, theta_dot, radius, position, speed)


# ==========================
# SAT 碰撞检测
# ==========================

def all_non_adjacent_pairs(bench_count: int = dragon_data.BENCH_COUNT) -> np.ndarray:
    pairs = []
    for i in range(bench_count):
        for j in range(i + 2, bench_count):
            pairs.append((i, j))
    return np.asarray(pairs, dtype=int)


ALL_PAIRS = all_non_adjacent_pairs()


def rectangle_arrays(position: np.ndarray) -> Dict[str, np.ndarray | float]:
    front = np.asarray(position[:-1], dtype=float)
    rear = np.asarray(position[1:], dtype=float)
    direction = front - rear
    norm = np.linalg.norm(direction, axis=1)
    if np.any(norm <= 0.0):
        raise ValueError("存在前后把手重合，无法构造矩形")
    direction = direction / norm[:, None]
    normal = np.column_stack([-direction[:, 1], direction[:, 0]])
    center = 0.5 * (front + rear)
    bench_count = position.shape[0] - 1
    length = dragon_data.get_bench_lengths()[:bench_count].astype(float)
    width = float(dragon_data.BENCH_WIDTH)
    return {"center": center, "direction": direction, "normal": normal, "length": length, "width": width}


def sat_gap_vectorized(rects: Dict[str, np.ndarray | float], pairs: np.ndarray) -> np.ndarray:
    if len(pairs) == 0:
        return np.array([], dtype=float)
    a = pairs[:, 0]
    b = pairs[:, 1]

    center_a = rects["center"][a]
    center_b = rects["center"][b]
    dir_a = rects["direction"][a]
    dir_b = rects["direction"][b]
    nor_a = rects["normal"][a]
    nor_b = rects["normal"][b]
    len_a = rects["length"][a]
    len_b = rects["length"][b]
    width = float(rects["width"])

    axes = np.stack([dir_a, nor_a, dir_b, nor_b], axis=1)
    delta = center_a - center_b
    projection_distance = np.abs(np.einsum("mi,mki->mk", delta, axes))

    half_a = (
        0.5 * len_a[:, None] * np.abs(np.einsum("mki,mi->mk", axes, dir_a))
        + 0.5 * width * np.abs(np.einsum("mki,mi->mk", axes, nor_a))
    )
    half_b = (
        0.5 * len_b[:, None] * np.abs(np.einsum("mki,mi->mk", axes, dir_b))
        + 0.5 * width * np.abs(np.einsum("mki,mi->mk", axes, nor_b))
    )
    axis_gaps = projection_distance - half_a - half_b
    return np.max(axis_gaps, axis=1).astype(float)


def required_point_count_for_pairs(pairs: np.ndarray) -> int:
    return int(np.max(pairs) + 2)


def evaluate_gap_at_radius(pitch: float, head_radius: float, pairs: np.ndarray, keep_state: bool = False) -> GapResult:
    point_count = required_point_count_for_pairs(pairs)
    state = solve_state_by_head_radius(pitch, head_radius, point_count=point_count)
    rects = rectangle_arrays(state.position)
    gaps = sat_gap_vectorized(rects, pairs)
    best_idx = int(np.argmin(gaps))
    best_pair = tuple(int(x) for x in pairs[best_idx])
    return GapResult(float(pitch), float(head_radius), float(gaps[best_idx]), best_pair, state if keep_state else None)


# ==========================
# 自动发现候选碰撞对
# ==========================

def expand_neighbor_pairs(base_pairs: Iterable[Tuple[int, int]], radius: int = 1) -> np.ndarray:
    pairs = set()
    n = dragon_data.BENCH_COUNT
    for i, j in base_pairs:
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                a = i + di
                b = j + dj
                if 0 <= a < n and 0 <= b < n and abs(a - b) > 1:
                    if a > b:
                        a, b = b, a
                    pairs.add((a, b))
    return np.asarray(sorted(pairs), dtype=int)


def discover_candidate_pairs() -> np.ndarray:
    """用全板凳对粗搜索自动发现第三问中的危险板凳对。"""
    discovered = set()
    radii = np.arange(TURNING_RADIUS, DISCOVER_RADIUS_MAX + 0.5 * DISCOVER_RADIUS_STEP, DISCOVER_RADIUS_STEP)
    print("自动发现第三问危险候选板凳对：")
    for pitch in DISCOVER_PITCHES:
        for radius in radii:
            result = evaluate_gap_at_radius(float(pitch), float(radius), ALL_PAIRS, keep_state=False)
            discovered.add(result.best_pair)
    expanded = expand_neighbor_pairs(discovered, radius=NEIGHBOR_PAIR_RADIUS)
    print(f"  粗搜索发现的核心板凳对：{sorted(discovered)}")
    print(f"  扩展后候选对数量：{len(expanded)}")
    return expanded


# ==========================
# 单螺距全过程最小净间隙
# ==========================

def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for left, right in intervals[1:]:
        old_left, old_right = merged[-1]
        if left <= old_right:
            merged[-1] = (old_left, max(old_right, right))
        else:
            merged.append((left, right))
    return merged


def minimum_gap_along_path(pitch: float, pairs: np.ndarray, return_history: bool = False) -> Tuple[GapResult, Optional[pd.DataFrame]]:
    pitch = float(pitch)
    initial_radius = pitch * INITIAL_THETA / (2.0 * np.pi)  # 16*pitch
    if initial_radius <= TURNING_RADIUS:
        raise ValueError(f"pitch={pitch:.6f} m 时初始半径不大于调头半径")

    coarse_count = max(5, int(np.ceil((initial_radius - TURNING_RADIUS) / RADIUS_COARSE_STEP)) + 1)
    coarse_radii = np.linspace(initial_radius, TURNING_RADIUS, coarse_count)
    coarse_results = [evaluate_gap_at_radius(pitch, float(r), pairs, keep_state=False) for r in coarse_radii]
    coarse_gaps = np.array([x.global_gap for x in coarse_results], dtype=float)

    low_indices = np.argsort(coarse_gaps)[: min(LOWEST_RADIUS_WINDOW_COUNT, len(coarse_gaps))]
    intervals = []
    for idx in low_indices:
        center = float(coarse_radii[int(idx)])
        intervals.append((max(TURNING_RADIUS, center - REFINE_RADIUS_WINDOW), min(initial_radius, center + REFINE_RADIUS_WINDOW)))
    intervals = merge_intervals(intervals)

    all_results = list(coarse_results)
    for left, right in intervals:
        refine_count = max(2, int(np.ceil((right - left) / RADIUS_REFINE_STEP)) + 1)
        refine_radii = np.linspace(left, right, refine_count)
        all_results.extend(evaluate_gap_at_radius(pitch, float(r), pairs, keep_state=False) for r in refine_radii)

    best = min(all_results, key=lambda x: x.global_gap)
    best = evaluate_gap_at_radius(best.pitch, best.head_radius, pairs, keep_state=True)

    if not return_history:
        return best, None

    rows = []
    used = set()
    for item in all_results:
        key = round(item.head_radius, 6)
        if key in used:
            continue
        used.add(key)
        rows.append({
            "pitch_m": item.pitch,
            "head_radius_m": item.head_radius,
            "gap_m": item.global_gap,
            "best_pair_i": item.best_pair[0],
            "best_pair_j": item.best_pair[1],
        })
    history_df = pd.DataFrame(rows).sort_values("head_radius_m", ascending=False)
    return best, history_df


# ==========================
# 搜索最小安全螺距
# ==========================

def find_pitch_bracket(pairs: np.ndarray) -> Tuple[float, float, pd.DataFrame]:
    pitches = np.arange(PITCH_SCAN_MIN, PITCH_SCAN_MAX + 0.5 * PITCH_SCAN_STEP, PITCH_SCAN_STEP)
    records = []
    last_pitch = None
    last_gap = None
    print("开始粗扫描螺距：")
    for pitch in pitches:
        best, _ = minimum_gap_along_path(float(pitch), pairs, return_history=False)
        safe = best.global_gap >= 0.0
        print(f"  p={pitch:.4f} m, min_gap={best.global_gap:+.6f} m, r={best.head_radius:.4f} m, pair={best.best_pair}, safe={safe}")
        records.append({
            "pitch_m": float(pitch),
            "pitch_cm": float(pitch) * 100.0,
            "min_gap_m": best.global_gap,
            "min_gap_radius_m": best.head_radius,
            "best_pair_i": best.best_pair[0],
            "best_pair_j": best.best_pair[1],
            "safe": bool(safe),
        })
        if last_pitch is not None and last_gap is not None:
            if last_gap < 0.0 and best.global_gap >= 0.0:
                return float(last_pitch), float(pitch), pd.DataFrame(records)
        last_pitch = float(pitch)
        last_gap = float(best.global_gap)
    raise ValueError("没有找到左碰撞右安全的螺距区间，请调整 PITCH_SCAN_MIN/PITCH_SCAN_MAX")


def bisect_pitch(left_pitch: float, right_pitch: float, pairs: np.ndarray) -> Tuple[float, GapResult, List[dict]]:
    left_best, _ = minimum_gap_along_path(left_pitch, pairs, return_history=False)
    right_best, _ = minimum_gap_along_path(right_pitch, pairs, return_history=False)
    if not (left_best.global_gap < 0.0 and right_best.global_gap >= 0.0):
        raise ValueError("二分区间不满足左碰撞右安全")

    iterations = []
    while right_pitch - left_pitch > PITCH_TOL:
        mid = 0.5 * (left_pitch + right_pitch)
        mid_best, _ = minimum_gap_along_path(mid, pairs, return_history=False)
        safe = mid_best.global_gap >= 0.0
        print(f"二分：p={mid:.9f} m, min_gap={mid_best.global_gap:+.10f} m, r={mid_best.head_radius:.6f} m, pair={mid_best.best_pair}, safe={safe}")
        iterations.append({
            "left_pitch_m": left_pitch,
            "right_pitch_m": right_pitch,
            "middle_pitch_m": mid,
            "middle_gap_m": mid_best.global_gap,
            "middle_radius_m": mid_best.head_radius,
            "best_pair_i": mid_best.best_pair[0],
            "best_pair_j": mid_best.best_pair[1],
            "safe": bool(safe),
        })
        if safe:
            right_pitch = mid
            right_best = mid_best
        else:
            left_pitch = mid
            left_best = mid_best

    final_pitch = right_pitch
    final_best, _ = minimum_gap_along_path(final_pitch, pairs, return_history=False)
    return final_pitch, final_best, iterations


def validate_with_all_pairs(final_pitch: float, focus_radius: float) -> GapResult:
    """在临界半径附近用所有非相邻板凳对验证。"""
    initial_radius = final_pitch * INITIAL_THETA / (2.0 * np.pi)
    left = max(TURNING_RADIUS, focus_radius - VALIDATE_RADIUS_WINDOW)
    right = min(initial_radius, focus_radius + VALIDATE_RADIUS_WINDOW)
    radii = np.arange(left, right + 0.5 * VALIDATE_RADIUS_STEP, VALIDATE_RADIUS_STEP)
    results = [evaluate_gap_at_radius(final_pitch, float(r), ALL_PAIRS, keep_state=False) for r in radii]
    best = min(results, key=lambda x: x.global_gap)
    return evaluate_gap_at_radius(best.pitch, best.head_radius, ALL_PAIRS, keep_state=True)


# ==========================
# 输出
# ==========================

def build_boundary_dataframe(state: StateByRadius) -> pd.DataFrame:
    # 此处需要完整 224 个把手，所以 state 应为全状态。
    return utils.build_result2_dataframe(state.position, state.speed).round(6)


def save_outputs(result: dict) -> Dict[str, Path]:
    utils.ensure_dir(dragon_data.OUTPUT_TABLES_DIR)
    utils.ensure_dir(dragon_data.OUTPUT_FIGURES_DIR)
    table_dir = Path(dragon_data.OUTPUT_TABLES_DIR)
    fig_dir = Path(dragon_data.OUTPUT_FIGURES_DIR)

    pitch_scan_csv = table_dir / "question3_pitch_scan.csv"
    radius_history_csv = table_dir / "question3_radius_gap_history.csv"
    iterations_csv = table_dir / "question3_bisection_iterations.csv"
    boundary_csv = table_dir / "question3_boundary_state.csv"
    summary_txt = table_dir / "question3_summary.txt"
    gap_pitch_png = fig_dir / "question3_gap_vs_pitch.png"
    gap_radius_png = fig_dir / "question3_gap_vs_radius.png"

    result["pitch_scan_df"].to_csv(pitch_scan_csv, index=False, encoding="utf-8-sig")
    result["radius_history_df"].to_csv(radius_history_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(result["iterations"]).to_csv(iterations_csv, index=False, encoding="utf-8-sig")
    result["boundary_df"].to_csv(boundary_csv, encoding="utf-8-sig")

    summary = (
        "第三问计算结果\n"
        "================\n"
        f"调头空间半径: {TURNING_RADIUS:.6f} m\n"
        f"最小螺距: {result['final_pitch']:.9f} m\n"
        f"最小螺距: {result['final_pitch'] * 100.0:.6f} cm\n"
        f"候选对临界最小净间隙: {result['final_gap']:.12f} m\n"
        f"候选对临界半径: {result['final_gap_radius']:.9f} m\n"
        f"候选临界板凳对: {result['best_pair']}\n"
        f"全板凳对验证最小净间隙: {result['validation_gap']:.12f} m\n"
        f"全板凳对验证半径: {result['validation_radius']:.9f} m\n"
        f"全板凳对验证板凳对: {result['validation_pair']}\n"
    )
    summary_txt.write_text(summary, encoding="utf-8")

    df = result["pitch_scan_df"]
    plt.figure(figsize=(9, 5))
    plt.plot(df["pitch_m"], df["min_gap_m"], marker="o")
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.axvline(result["final_pitch"], linestyle="--", linewidth=1.0)
    plt.xlabel("Pitch p (m)")
    plt.ylabel("Minimum gap along path (m)")
    plt.title("Question 3: Minimum Gap vs Pitch")
    plt.tight_layout()
    plt.savefig(gap_pitch_png, dpi=200)
    plt.close()

    rdf = result["radius_history_df"]
    plt.figure(figsize=(9, 5))
    plt.plot(rdf["head_radius_m"], rdf["gap_m"], marker="o", markersize=2, linewidth=1.0)
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.axvline(TURNING_RADIUS, linestyle="--", linewidth=1.0)
    plt.gca().invert_xaxis()
    plt.xlabel("Head radius r (m)")
    plt.ylabel("Gap (m)")
    plt.title("Question 3: Gap Along Path at Minimum Pitch")
    plt.tight_layout()
    plt.savefig(gap_radius_png, dpi=200)
    plt.close()

    return {
        "pitch_scan_csv": pitch_scan_csv,
        "radius_history_csv": radius_history_csv,
        "iterations_csv": iterations_csv,
        "boundary_csv": boundary_csv,
        "summary_txt": summary_txt,
        "gap_pitch_png": gap_pitch_png,
        "gap_radius_png": gap_radius_png,
    }


def solve_question3() -> dict:
    candidate_pairs = discover_candidate_pairs()
    left_pitch, right_pitch, pitch_scan_df = find_pitch_bracket(candidate_pairs)
    print(f"\n二分初始区间：[{left_pitch:.6f}, {right_pitch:.6f}] m\n")
    final_pitch, final_best, iterations = bisect_pitch(left_pitch, right_pitch, candidate_pairs)
    final_best, radius_history_df = minimum_gap_along_path(final_pitch, candidate_pairs, return_history=True)

    validation = validate_with_all_pairs(final_pitch, final_best.head_radius)

    # 输出边界 r=4.5m 处的完整把手位置和速度，方便论文/检查使用。
    boundary_state = solve_state_by_head_radius(final_pitch, TURNING_RADIUS, point_count=dragon_data.POINT_COUNT)
    boundary_df = build_boundary_dataframe(boundary_state)

    result = {
        "candidate_pairs": candidate_pairs,
        "final_pitch": final_pitch,
        "final_gap": final_best.global_gap,
        "final_gap_radius": final_best.head_radius,
        "best_pair": final_best.best_pair,
        "validation_gap": validation.global_gap,
        "validation_radius": validation.head_radius,
        "validation_pair": validation.best_pair,
        "pitch_scan_df": pitch_scan_df,
        "radius_history_df": radius_history_df,
        "iterations": iterations,
        "boundary_df": boundary_df,
    }
    result["output_paths"] = save_outputs(result)
    return result


def main() -> None:
    result = solve_question3()
    print("\n第三问完成。")
    print(f"最小螺距 = {result['final_pitch']:.9f} m = {result['final_pitch'] * 100.0:.6f} cm")
    print(f"临界最小净间隙 = {result['final_gap']:.12f} m")
    print(f"临界半径 = {result['final_gap_radius']:.9f} m")
    print(f"临界板凳对 = {result['best_pair']}")
    print(f"全板凳对验证 gap = {result['validation_gap']:.12f} m, pair={result['validation_pair']}")
    print("\n已生成文件：")
    for path in result["output_paths"].values():
        print(path)


if __name__ == "__main__":
    main()
