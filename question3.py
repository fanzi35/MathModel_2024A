
"""
2024A 板凳龙 第三问：最小螺距（30cm 下界 + 龙头-内两层碰撞版）

运行：
    python question3.py

建模前提：
    沿用第二问的碰撞规律：限制继续盘入的主要约束来自龙头板凳
    与其附近内两层螺线上的板凳之间的碰撞。因此第三问只检测：
        龙头板凳 0 与“以内两层”范围内的非相邻板凳 j 的碰撞。

严格性说明：
    1. 螺距二分下界固定为板凳宽度 0.30 m；
    2. 不经验指定某一组核心板凳对，例如不预设 (0, 19)；
    3. 每个半径状态下，根据当前所有板凳的极角 theta_mid 自动确定内两层板凳集合；
    4. 对该集合内全部龙头-非相邻板凳对做向量化 SAT 精确净间隙计算；
    5. 对龙头半径区间采用粗扫 + 危险区间加密 + 最优点再加密的确定性数值搜索。

注意：
    这里的“严格”是指在所采用的建模前提内没有经验指定碰撞对；
    连续半径区间仍然采用数值加密搜索近似求全过程最小净间隙。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dragon_data
import utils


# ==========================
# 第三问参数
# ==========================
TURNING_RADIUS = 4.5
INITIAL_THETA = float(dragon_data.QUESTION1_INITIAL_THETA)  # 32π，第 16 圈
THEORETICAL_PITCH_LOW = TURNING_RADIUS / 16.0

# 用户指定：初始二分下界取板凳宽度 30 cm。
PITCH_SEARCH_LOW = float(dragon_data.BENCH_WIDTH)  # 0.30 m
PITCH_TOL = 1e-6
INITIAL_SAFE_PITCH_GUESS = float(dragon_data.QUESTION1_PITCH)

# 内两层定义：以龙头极角 theta_head 为起点，向外取两圈，即 theta_mid < theta_head + 4π。
# 这里用板凳中点极角 theta_mid 判断板凳属于哪一层，比只用前把手更平滑。
LAYER_COUNT_FOR_HEAD_CHECK = 2
LAYER_THETA_SPAN = LAYER_COUNT_FOR_HEAD_CHECK * 2.0 * np.pi
HEAD_BENCH_INDEX = 0

# 半径扫描参数：先粗扫，再在危险区域加密。
RADIUS_SCAN_STEP = 0.04
LOWEST_RADIUS_WINDOW_COUNT = 10
REFINE_RADIUS_WINDOW = 0.06
RADIUS_REFINE_STEP = 0.004
FINAL_REFINE_WINDOW = 0.012
FINAL_REFINE_STEP = 0.0008

# 安全判定数值余量。
GAP_EPS = 1e-10


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
    pair_count: int
    state: Optional[StateByRadius] = None

    @property
    def safe(self) -> bool:
        return self.global_gap >= -GAP_EPS


# ==========================
# 给定螺距、给定龙头半径，求整条龙状态
# ==========================

def _point_distance_sq(theta_a: float, theta_b: float, b: float) -> float:
    radius_a = b * theta_a
    radius_b = b * theta_b
    delta = theta_a - theta_b
    return float(radius_a * radius_a + radius_b * radius_b - 2.0 * radius_a * radius_b * np.cos(delta))


def solve_trailing_theta_fast(theta_prev: float, distance: float, b: float, tol: float = 1e-12) -> float:
    """由前一把手极角 theta_prev 和把手距离 distance 求后一把手极角。"""
    theta_prev = float(theta_prev)
    distance = float(distance)
    b = float(b)

    # 先用局部弧长近似给出初值，再用牛顿法加速。
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

    # 牛顿法失败则回退到二分法，保证稳定。
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


def solve_state_by_head_radius(pitch: float, head_radius: float) -> StateByRadius:
    """给定螺距 pitch 和龙头半径 head_radius，求所有把手位置和速度。"""
    pitch = float(pitch)
    head_radius = float(head_radius)
    if pitch <= 0.0:
        raise ValueError("pitch 必须为正数")
    if head_radius <= 0.0:
        raise ValueError("head_radius 必须为正数")

    b = utils.spiral_coefficient(pitch)
    distances = dragon_data.get_handle_distances()
    point_count = dragon_data.POINT_COUNT

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

    # 速度递推与第一问一致，第三问主要使用位置碰撞，速度用于输出边界状态。
    theta_dot[0] = -1.0 / (b * np.sqrt(1.0 + theta[0] * theta[0]))
    for i in range(1, point_count):
        delta = theta[i] - theta[i - 1]
        numerator = theta[i - 1] - theta[i] * np.cos(delta) - theta[i] * theta[i - 1] * np.sin(delta)
        denominator = theta[i] - theta[i - 1] * np.cos(delta) + theta[i] * theta[i - 1] * np.sin(delta)
        theta_dot[i] = -(numerator / denominator) * theta_dot[i - 1]

    speed[:] = utils.speed_from_theta(theta, theta_dot, b)
    return StateByRadius(pitch, head_radius, theta, theta_dot, radius, position, speed)


# ==========================
# 内两层龙头碰撞对 + 向量化 SAT
# ==========================

def head_two_layer_pairs(theta: np.ndarray) -> np.ndarray:
    """根据当前极角自动构造“龙头 vs 内两层非相邻板凳”的检测对。

    板凳 i 的层位置用两把手极角中点 theta_mid[i] 表示。
    内两层定义为：theta_head <= theta_mid < theta_head + 4π。
    板凳 0 是龙头本身；板凳 1 与龙头相邻，二者通过把手连接，不纳入碰撞检测。
    """
    theta = np.asarray(theta, dtype=float)
    theta_head = float(theta[0])
    theta_front = theta[:-1]
    theta_rear = theta[1:]
    theta_mid = 0.5 * (theta_front + theta_rear)

    lower = theta_head
    upper = theta_head + LAYER_THETA_SPAN
    indices = np.where((theta_mid >= lower) & (theta_mid < upper))[0]
    indices = indices[indices >= HEAD_BENCH_INDEX + 2]  # 排除龙头和相邻板凳 1

    if len(indices) == 0:
        return np.empty((0, 2), dtype=np.int32)
    pairs = np.column_stack([
        np.full(len(indices), HEAD_BENCH_INDEX, dtype=np.int32),
        indices.astype(np.int32),
    ])
    return pairs


def rectangle_arrays(position: np.ndarray) -> Dict[str, np.ndarray | float]:
    """把所有板凳转换为矩形数组。"""
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
    return {
        "center": center,
        "direction": direction,
        "normal": normal,
        "length": length,
        "width": width,
    }


def sat_gap_vectorized(rects: Dict[str, np.ndarray | float], pairs: np.ndarray) -> np.ndarray:
    """对给定板凳对批量计算 SAT 净间隙。"""
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


def evaluate_gap_at_radius(pitch: float, head_radius: float, keep_state: bool = False) -> GapResult:
    """给定螺距和龙头半径，计算龙头与内两层板凳的最小 SAT 净间隙。"""
    state = solve_state_by_head_radius(pitch, head_radius)
    pairs = head_two_layer_pairs(state.theta)
    if len(pairs) == 0:
        return GapResult(float(pitch), float(head_radius), float("inf"), (-1, -1), 0, state if keep_state else None)

    rects = rectangle_arrays(state.position)
    sat_gaps = sat_gap_vectorized(rects, pairs)
    best_idx = int(np.argmin(sat_gaps))
    best_gap = float(sat_gaps[best_idx])
    best_pair = tuple(int(x) for x in pairs[best_idx])
    return GapResult(float(pitch), float(head_radius), best_gap, best_pair, len(pairs), state if keep_state else None)


# ==========================
# 单个螺距下，从初始半径到调头边界的数值全过程检查
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


def make_radius_grid(left: float, right: float, step: float) -> np.ndarray:
    """生成从 right 到 left 的半径网格，包含端点。"""
    left = float(left)
    right = float(right)
    step = float(step)
    if right < left:
        raise ValueError("right 必须不小于 left")
    count = max(2, int(np.ceil((right - left) / step)) + 1)
    return np.linspace(right, left, count)


def _scan_radius_intervals(pitch: float, intervals: List[Tuple[float, float]], step: float) -> List[GapResult]:
    results: List[GapResult] = []
    for left, right in intervals:
        radii = make_radius_grid(left, right, step)
        results.extend(evaluate_gap_at_radius(pitch, float(r), keep_state=False) for r in radii)
    return results


def minimum_gap_along_path(pitch: float, return_history: bool = False) -> Tuple[GapResult, Optional[pd.DataFrame]]:
    """计算某螺距下盘入到调头边界前，龙头与内两层板凳的全过程最小净间隙。"""
    pitch = float(pitch)
    initial_radius = pitch * INITIAL_THETA / (2.0 * np.pi)  # =16p

    if initial_radius <= TURNING_RADIUS:
        bad = GapResult(pitch, initial_radius, -np.inf, (-1, -1), 0, None)
        if return_history:
            return bad, pd.DataFrame([{
                "pitch_m": pitch,
                "head_radius_m": initial_radius,
                "gap_m": -np.inf,
                "best_pair_i": -1,
                "best_pair_j": -1,
                "pair_count": 0,
                "stage": "invalid_initial_radius",
            }])
        return bad, None

    # 第一层：全路径粗扫。
    coarse_radii = make_radius_grid(TURNING_RADIUS, initial_radius, RADIUS_SCAN_STEP)
    coarse_results = [evaluate_gap_at_radius(pitch, float(r), keep_state=False) for r in coarse_radii]
    coarse_gaps = np.array([x.global_gap for x in coarse_results], dtype=float)

    # 第二层：对粗扫中最危险的若干半径附近加密。
    low_indices = np.argsort(coarse_gaps)[: min(LOWEST_RADIUS_WINDOW_COUNT, len(coarse_gaps))]
    intervals: List[Tuple[float, float]] = []
    for idx in low_indices:
        center = float(coarse_radii[int(idx)])
        intervals.append((
            max(TURNING_RADIUS, center - REFINE_RADIUS_WINDOW),
            min(initial_radius, center + REFINE_RADIUS_WINDOW),
        ))
    negative_indices = np.where(coarse_gaps <= 0.0)[0]
    for idx in negative_indices:
        center = float(coarse_radii[int(idx)])
        intervals.append((
            max(TURNING_RADIUS, center - REFINE_RADIUS_WINDOW),
            min(initial_radius, center + REFINE_RADIUS_WINDOW),
        ))
    intervals = merge_intervals(intervals)
    refine_results = _scan_radius_intervals(pitch, intervals, RADIUS_REFINE_STEP)

    # 第三层：围绕当前最小值再加密一次。
    temp_best = min(coarse_results + refine_results, key=lambda x: x.global_gap)
    final_left = max(TURNING_RADIUS, temp_best.head_radius - FINAL_REFINE_WINDOW)
    final_right = min(initial_radius, temp_best.head_radius + FINAL_REFINE_WINDOW)
    final_results = _scan_radius_intervals(pitch, [(final_left, final_right)], FINAL_REFINE_STEP)

    all_results = coarse_results + refine_results + final_results
    best = min(all_results, key=lambda x: x.global_gap)
    best = evaluate_gap_at_radius(best.pitch, best.head_radius, keep_state=True)

    if not return_history:
        return best, None

    rows = []
    used = set()
    for stage, items in [("coarse", coarse_results), ("refine", refine_results), ("final_refine", final_results)]:
        for item in items:
            key = (stage, round(item.head_radius, 8))
            if key in used:
                continue
            used.add(key)
            rows.append({
                "pitch_m": item.pitch,
                "head_radius_m": item.head_radius,
                "gap_m": item.global_gap,
                "best_pair_i": item.best_pair[0],
                "best_pair_j": item.best_pair[1],
                "pair_count": item.pair_count,
                "stage": stage,
            })
    history_df = pd.DataFrame(rows).sort_values(["stage", "head_radius_m"], ascending=[True, False])
    return best, history_df


# ==========================
# 自动括区 + 螺距二分
# ==========================

def is_safe_pitch(pitch: float) -> Tuple[bool, GapResult]:
    best, _ = minimum_gap_along_path(pitch, return_history=False)
    return best.safe, best


def find_pitch_bracket() -> Tuple[float, float, pd.DataFrame]:
    """寻找左不安全、右安全的螺距二分初始区间。"""
    records = []
    left_pitch = PITCH_SEARCH_LOW
    left_safe, left_best = is_safe_pitch(left_pitch)
    records.append({
        "pitch_m": left_pitch,
        "pitch_cm": left_pitch * 100.0,
        "min_gap_m": left_best.global_gap,
        "min_gap_radius_m": left_best.head_radius,
        "best_pair_i": left_best.best_pair[0],
        "best_pair_j": left_best.best_pair[1],
        "pair_count": left_best.pair_count,
        "safe": bool(left_safe),
        "stage": "fixed_lower_bound_30cm",
    })

    if left_safe:
        # 在“螺距不小于板凳宽度”的约束下，若 0.30m 已安全，则最小值即为 0.30m。
        return left_pitch, left_pitch, pd.DataFrame(records)

    right_pitch = max(INITIAL_SAFE_PITCH_GUESS, left_pitch * 1.2)
    for _ in range(20):
        right_safe, right_best = is_safe_pitch(right_pitch)
        records.append({
            "pitch_m": right_pitch,
            "pitch_cm": right_pitch * 100.0,
            "min_gap_m": right_best.global_gap,
            "min_gap_radius_m": right_best.head_radius,
            "best_pair_i": right_best.best_pair[0],
            "best_pair_j": right_best.best_pair[1],
            "pair_count": right_best.pair_count,
            "safe": bool(right_safe),
            "stage": "right_bound_search",
        })
        if right_safe:
            return left_pitch, right_pitch, pd.DataFrame(records)
        right_pitch *= 1.15

    raise ValueError("自动括区失败：没有找到安全上界，请检查模型或增大搜索上限")


def bisect_pitch(left_pitch: float, right_pitch: float) -> Tuple[float, GapResult, List[dict]]:
    left_safe, left_best = is_safe_pitch(left_pitch)
    right_safe, right_best = is_safe_pitch(right_pitch)

    if abs(right_pitch - left_pitch) < 1e-15 and left_safe:
        return right_pitch, left_best, []
    if left_safe or not right_safe:
        raise ValueError("二分区间必须满足左端不安全、右端安全")

    iterations: List[dict] = []
    while right_pitch - left_pitch > PITCH_TOL:
        mid = 0.5 * (left_pitch + right_pitch)
        mid_safe, mid_best = is_safe_pitch(mid)
        print(
            f"二分：p={mid:.9f} m, min_gap={mid_best.global_gap:+.10f} m, "
            f"r={mid_best.head_radius:.6f} m, pair={mid_best.best_pair}, "
            f"pairs={mid_best.pair_count}, safe={mid_safe}"
        )
        iterations.append({
            "left_pitch_m": left_pitch,
            "right_pitch_m": right_pitch,
            "middle_pitch_m": mid,
            "middle_gap_m": mid_best.global_gap,
            "middle_radius_m": mid_best.head_radius,
            "best_pair_i": mid_best.best_pair[0],
            "best_pair_j": mid_best.best_pair[1],
            "pair_count": mid_best.pair_count,
            "safe": bool(mid_safe),
        })

        if mid_safe:
            right_pitch = mid
            right_best = mid_best
        else:
            left_pitch = mid
            left_best = mid_best

    final_pitch = right_pitch
    final_best, _ = minimum_gap_along_path(final_pitch, return_history=False)
    return final_pitch, final_best, iterations


# ==========================
# 输出结果
# ==========================

def build_boundary_dataframe(state: StateByRadius) -> pd.DataFrame:
    return utils.build_result2_dataframe(state.position, state.speed).round(6)


def save_outputs(result: dict) -> Dict[str, Path]:
    utils.ensure_dir(dragon_data.OUTPUT_TABLES_DIR)
    utils.ensure_dir(dragon_data.OUTPUT_FIGURES_DIR)
    table_dir = Path(dragon_data.OUTPUT_TABLES_DIR)
    fig_dir = Path(dragon_data.OUTPUT_FIGURES_DIR)

    bracket_csv = table_dir / "question3_bracket.csv"
    iterations_csv = table_dir / "question3_bisection_iterations.csv"
    radius_history_csv = table_dir / "question3_radius_gap_history.csv"
    boundary_csv = table_dir / "question3_boundary_state.csv"
    summary_txt = table_dir / "question3_summary.txt"
    gap_radius_png = fig_dir / "question3_gap_vs_radius.png"
    gap_pitch_png = fig_dir / "question3_bisection_gap.png"

    result["bracket_df"].to_csv(bracket_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(result["iterations"]).to_csv(iterations_csv, index=False, encoding="utf-8-sig")
    result["radius_history_df"].to_csv(radius_history_csv, index=False, encoding="utf-8-sig")
    result["boundary_df"].to_csv(boundary_csv, encoding="utf-8-sig")

    summary = (
        f"调头空间半径: {TURNING_RADIUS:.6f} m\n"
        f"理论半径约束下界: {THEORETICAL_PITCH_LOW:.9f} m\n"
        f"本文采用的二分下界: {PITCH_SEARCH_LOW:.9f} m（板凳宽度）\n"
        f"内两层极角范围: theta_head 到 theta_head + {LAYER_THETA_SPAN:.9f}\n"
        f"最小安全螺距: {result['final_pitch']:.9f} m\n"
        f"最小安全螺距: {result['final_pitch'] * 100.0:.6f} cm\n"
        f"临界最小净间隙: {result['final_gap']:.12f} m\n"
        f"临界半径: {result['final_gap_radius']:.9f} m\n"
        f"临界板凳对: {result['best_pair']}\n"
        f"临界状态检测板凳对数量: {result['final_pair_count']}\n"
    )
    summary_txt.write_text(summary, encoding="utf-8")

    rdf = result["radius_history_df"]
    plt.figure(figsize=(9, 5))
    for stage, group in rdf.groupby("stage"):
        plt.plot(group["head_radius_m"], group["gap_m"], marker="o", markersize=2, linewidth=1.0, label=stage)
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.axvline(TURNING_RADIUS, linestyle="--", linewidth=1.0)
    plt.gca().invert_xaxis()
    plt.xlabel("Head radius r (m)")
    plt.ylabel("SAT gap (m)")
    plt.title("Question 3: Head-vs-Two-Layer Gap Along Path")
    plt.legend()
    plt.tight_layout()
    plt.savefig(gap_radius_png, dpi=200)
    plt.close()

    itdf = pd.DataFrame(result["iterations"])
    plt.figure(figsize=(9, 5))
    if len(itdf) > 0:
        plt.plot(itdf["middle_pitch_m"], itdf["middle_gap_m"], marker="o", linewidth=1.0)
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.axvline(result["final_pitch"], linestyle="--", linewidth=1.0)
    plt.xlabel("Pitch p (m)")
    plt.ylabel("Minimum SAT gap along path (m)")
    plt.title("Question 3: Pitch Bisection History")
    plt.tight_layout()
    plt.savefig(gap_pitch_png, dpi=200)
    plt.close()

    return {
        "bracket_csv": bracket_csv,
        "iterations_csv": iterations_csv,
        "radius_history_csv": radius_history_csv,
        "boundary_csv": boundary_csv,
        "summary_txt": summary_txt,
        "gap_radius_png": gap_radius_png,
        "gap_pitch_png": gap_pitch_png,
    }


def solve_question3() -> dict:
    print("寻找螺距二分初始区间：下界固定 0.30m，只检测龙头与内两层非相邻板凳")
    left_pitch, right_pitch, bracket_df = find_pitch_bracket()
    print(f"二分初始区间: [{left_pitch:.9f}, {right_pitch:.9f}] m")

    final_pitch, final_best, iterations = bisect_pitch(left_pitch, right_pitch)
    final_best, radius_history_df = minimum_gap_along_path(final_pitch, return_history=True)

    # 输出龙头到达调头空间边界 r=4.5m 时的完整状态。
    boundary_state = solve_state_by_head_radius(final_pitch, TURNING_RADIUS)
    boundary_df = build_boundary_dataframe(boundary_state)

    result = {
        "final_pitch": final_pitch,
        "final_gap": final_best.global_gap,
        "final_gap_radius": final_best.head_radius,
        "best_pair": final_best.best_pair,
        "final_pair_count": final_best.pair_count,
        "bracket_df": bracket_df,
        "iterations": iterations,
        "radius_history_df": radius_history_df,
        "boundary_df": boundary_df,
    }
    result["output_paths"] = save_outputs(result)
    return result


def main() -> None:
    result = solve_question3()
    print("\n第三问完成。")
    print(f"最小安全螺距 = {result['final_pitch']:.9f} m = {result['final_pitch'] * 100.0:.6f} cm")
    print(f"临界最小净间隙 = {result['final_gap']:.12f} m")
    print(f"临界半径 = {result['final_gap_radius']:.9f} m")
    print(f"临界板凳对 = {result['best_pair']}")
    print(f"临界状态检测对数量 = {result['final_pair_count']}")
    print("\n已生成文件：")
    for path in result["output_paths"].values():
        print(path)


if __name__ == "__main__":
    main()
    # 防止少数本地 matplotlib 后台线程导致程序不退出。
    import sys
    import os
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
