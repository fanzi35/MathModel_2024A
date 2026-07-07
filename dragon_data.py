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
BENCH_WIDTH = 0.30
HOLE_OFFSET = 0.275

DRAGON_BODY_COUNT = 221
POINT_COUNT = 224
BENCH_COUNT = 223

QUESTION1_PITCH = 0.55
QUESTION1_HEAD_SPEED = 1.0
QUESTION1_INITIAL_THETA = 32.0 * np.pi
QUESTION1_TIME_POINTS = np.arange(0.0, 301.0, 1.0)
QUESTION1_KEY_TIMES = np.array([0, 60, 120, 180, 240, 300], dtype=float)

QUESTION2_SCAN_STEPS = (1.0, 0.1, 0.01)
QUESTION2_TIME_TOL = 1e-6
QUESTION2_MAX_TIME = 600.0
QUESTION2_KEY_POINT_INDICES = [0, 1, 51, 101, 151, 201, 223]
QUESTION2_CHINESE_FONTS = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Arial Unicode MS"]

QUESTION4_PITCH = 1.7
QUESTION4_TURN_RADIUS = 4.5
QUESTION4_RADIUS_RATIO = 2.0
QUESTION4_HEAD_SPEED = 1.0
QUESTION4_TIME_POINTS = np.arange(-100.0, 101.0, 1.0)
QUESTION4_KEY_TIMES = np.array([-100, -50, 0, 50, 100], dtype=float)
QUESTION4_KEY_POINT_INDICES = [0, 1, 51, 101, 151, 201, 223]


def get_handle_distances():
    """返回相邻把手中心的固定距离序列。"""
    head_distance = HEAD_BENCH_LENGTH - 2.0 * HOLE_OFFSET
    body_distance = BODY_BENCH_LENGTH - 2.0 * HOLE_OFFSET
    return np.array([head_distance] + [body_distance] * 222, dtype=float)


def get_bench_lengths():
    """返回每条板凳的长度。"""
    return np.array([HEAD_BENCH_LENGTH] + [BODY_BENCH_LENGTH] * 222, dtype=float)


def get_point_names():
    """返回题目结果表对应的点名称。"""
    body_names = [f"第{i}节龙身" for i in range(1, DRAGON_BODY_COUNT + 1)]
    return ["龙头"] + body_names + ["龙尾", "龙尾（后）"]


def get_result2_row_labels():
    """返回第二问结果表的行标签。"""
    return get_point_names()


def get_position_row_labels():
    """返回第一问位置表的行标签。"""
    labels = []
    for name in get_point_names():
        labels.append(f"{name}x (m)")
        labels.append(f"{name}y (m)")
    return labels


def get_speed_row_labels():
    """返回第一问速度表的行标签。"""
    return [f"{name} (m/s)" for name in get_point_names()]


def get_question2_summary_labels():
    """返回第二问摘要表的点名称。"""
    names = get_point_names()
    return [names[index] for index in QUESTION2_KEY_POINT_INDICES]


def get_question4_summary_labels():
    """返回第四问摘要表的点名称。"""
    names = get_point_names()
    return [names[index] for index in QUESTION4_KEY_POINT_INDICES]


def get_time_labels(time_points):
    """返回题目要求的时间列标签。"""
    labels = []
    for value in np.asarray(time_points, dtype=float):
        if abs(value - round(value)) < 1e-9:
            labels.append(f"{int(round(value))} s")
        else:
            labels.append(f"{value:.2f} s")
    return labels


def get_question2_plot_labels():
    """返回第二问图像文字配置。"""
    return {
        "gap_title": "第二问碰撞间隙变化曲线",
        "gap_xlabel": "时间 (s)",
        "gap_ylabel": "G(t)",
        "safe_text": "安全侧",
        "collision_text": "碰撞侧",
        "local_title": "第二问临界时刻局部几何示意图",
        "sat_left_title": "SAT 分离轴示意图",
        "sat_right_title": "SAT 投影区间示意图",
        "x_label": "横坐标 x (m)",
        "y_label": "纵坐标 y (m)",
    }


def get_question4_plot_labels():
    """返回第四问图像文字配置。"""
    return {
        "path_title": "第四问调头复合路径示意图",
        "state_title": "第四问板凳龙状态示意图",
        "speed_title": "第四问代表节点速度曲线",
        "x_label": "横坐标 x (m)",
        "y_label": "纵坐标 y (m)",
        "time_label": "时间 (s)",
        "speed_label": "速度 (m/s)",
    }
