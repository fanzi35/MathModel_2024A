from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "docs"
REFERENCE_DIR = DOCS_DIR / "reference_formats"
OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

HEAD_BENCH_LENGTH = 3.41
BODY_BENCH_LENGTH = 2.20
TAIL_BENCH_LENGTH = 2.20
HOLE_OFFSET = 0.275
BENCH_WIDTH = 0.30

DRAGON_BODY_COUNT = 221
POINT_COUNT = 224

QUESTION1_PITCH = 0.55
QUESTION1_HEAD_SPEED = 1.0
QUESTION1_INITIAL_THETA = 32.0 * np.pi
QUESTION1_TIME_POINTS = np.arange(0.0, 301.0, 1.0)
QUESTION1_KEY_TIMES = np.array([0, 60, 120, 180, 240, 300], dtype=float)
QUESTION1_KEY_POINT_INDICES = [0, 1, 51, 101, 151, 201, 223]


def get_handle_distances():
    """返回相邻把手中心的固定距离序列。"""
    head_distance = HEAD_BENCH_LENGTH - 2.0 * HOLE_OFFSET
    body_distance = BODY_BENCH_LENGTH - 2.0 * HOLE_OFFSET
    return np.array([head_distance] + [body_distance] * 222, dtype=float)


def get_point_names():
    """返回题目结果表对应的点名称。"""
    body_names = [f"第{i}节龙身" for i in range(1, DRAGON_BODY_COUNT + 1)]
    return ["龙头"] + body_names + ["龙尾", "龙尾（后）"]


def get_position_row_labels():
    """返回位置表的行标签。"""
    labels = []
    for name in get_point_names():
        labels.append(f"{name}x (m)")
        labels.append(f"{name}y (m)")
    return labels


def get_speed_row_labels():
    """返回速度表的行标签。"""
    return [f"{name} (m/s)" for name in get_point_names()]


def get_time_labels(time_points):
    """返回题目要求的时间列标签。"""
    return [f"{int(t)} s" for t in np.asarray(time_points, dtype=float)]
