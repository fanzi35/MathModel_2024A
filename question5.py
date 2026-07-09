"""
文件名：question5.py
用于存放文件五解决代码
"""
from pathlib import Path

import numpy as np
import pandas as pd

import dragon_data
import question4
import utils


def _parse_time_labels(columns):
    """将时间列标签解析为浮点时间。"""
    time_points = []
    for column in columns:
        label = str(column).replace(" s", "")
        time_points.append(float(label))
    return np.asarray(time_points, dtype=float)


def load_cached_question4_result(position_csv_path, speed_csv_path):
    """从第四问导出的 CSV 结果重建位置与速度矩阵。"""
    position_df = pd.read_csv(position_csv_path, index_col=0)
    speed_df = pd.read_csv(speed_csv_path, index_col=0)
    time_points = _parse_time_labels(position_df.columns)

    position_values = position_df.to_numpy(dtype=float)
    speed_values = speed_df.to_numpy(dtype=float)
    point_count = speed_values.shape[0]
    time_count = speed_values.shape[1]
    position = np.zeros((point_count, 2, time_count), dtype=float)
    for point_index in range(point_count):
        position[point_index, 0, :] = position_values[2 * point_index, :]
        position[point_index, 1, :] = position_values[2 * point_index + 1, :]

    return {
        "time_points": time_points,
        "position": position,
        "speed": speed_values,
    }


def extract_peak_metrics(speed, time_points):
    """从速度矩阵中提取最大速度及其位置。"""
    speed = np.asarray(speed, dtype=float)
    time_points = np.asarray(time_points, dtype=float)
    risk_point_index, risk_time_index = np.unravel_index(np.argmax(speed), speed.shape)
    return {
        "peak_speed": float(speed[risk_point_index, risk_time_index]),
        "risk_point_index": int(risk_point_index),
        "risk_time_index": int(risk_time_index),
        "risk_time": float(time_points[risk_time_index]),
    }


def build_refined_time_grid(center_time, half_width, step, lower_bound, upper_bound):
    """围绕中心时刻构造细化搜索时间网格。"""
    start = max(float(lower_bound), float(center_time) - float(half_width))
    end = min(float(upper_bound), float(center_time) + float(half_width))
    point_count = int(np.floor((end - start) / step + 1e-9)) + 1
    grid = start + np.arange(point_count + 1, dtype=float) * step
    grid = grid[grid <= end + 1e-9]
    if grid.size == 0 or abs(grid[0] - start) > 1e-9:
        grid = np.insert(grid, 0, start)
    if abs(grid[-1] - end) > 1e-9:
        grid = np.append(grid, end)
    return np.unique(np.round(grid, 10))


def merge_sample_results(sample_results):
    """合并多轮采样结果。"""
    time_map = {}
    for result in sample_results:
        current_times = np.asarray(result["time_points"], dtype=float)
        current_speed = np.asarray(result["speed"], dtype=float)
        current_position = np.asarray(result["position"], dtype=float)
        for time_index, current_time in enumerate(current_times):
            key = round(float(current_time), 10)
            time_map[key] = {
                "time": float(current_time),
                "speed": current_speed[:, time_index].copy(),
                "position": current_position[:, :, time_index].copy(),
            }

    ordered_keys = sorted(time_map.keys())
    merged_times = np.array([time_map[key]["time"] for key in ordered_keys], dtype=float)
    point_count = len(time_map[ordered_keys[0]]["speed"])
    merged_speed = np.zeros((point_count, len(ordered_keys)), dtype=float)
    merged_position = np.zeros((point_count, 2, len(ordered_keys)), dtype=float)
    for merged_index, key in enumerate(ordered_keys):
        merged_speed[:, merged_index] = time_map[key]["speed"]
        merged_position[:, :, merged_index] = time_map[key]["position"]
    return {
        "time_points": merged_times,
        "speed": merged_speed,
        "position": merged_position,
    }


def local_maximize_handle_speed(evaluator, risk_point_index, center_time, half_width, lower_bound, upper_bound, tol=None):
    """对单个风险把手速度做局部连续极大值搜索。"""
    if tol is None:
        tol = dragon_data.QUESTION5_LOCAL_TOL
    left = max(float(lower_bound), float(center_time) - float(half_width))
    right = min(float(upper_bound), float(center_time) + float(half_width))
    cache = {}

    def evaluate_at(time_value):
        key = round(float(time_value), 12)
        if key not in cache:
            state = evaluator(np.array([time_value], dtype=float))
            cache[key] = {
                "speed": float(state["speed"][risk_point_index, 0]),
                "position": state["position"][:, :, 0].copy(),
            }
        return cache[key]

    while right - left > tol:
        third = (right - left) / 3.0
        mid_left = left + third
        mid_right = right - third
        if evaluate_at(mid_left)["speed"] < evaluate_at(mid_right)["speed"]:
            left = mid_left
        else:
            right = mid_right

    candidate_times = [left, 0.5 * (left + right), right, center_time]
    best_time = candidate_times[0]
    best_state = evaluate_at(best_time)
    for current_time in candidate_times[1:]:
        current_state = evaluate_at(current_time)
        if current_state["speed"] > best_state["speed"]:
            best_time = current_time
            best_state = current_state

    return {
        "peak_time": float(best_time),
        "peak_speed": float(best_state["speed"]),
        "peak_position": best_state["position"],
    }


def layered_search_max_speed(coarse_result, evaluator):
    """用 1s / 0.1s / 0.01s 分层搜索最大速度，再做局部连续优化。"""
    coarse_times = np.asarray(coarse_result["time_points"], dtype=float)
    coarse_metrics = extract_peak_metrics(coarse_result["speed"], coarse_times)
    lower_bound = float(np.min(coarse_times))
    upper_bound = float(np.max(coarse_times))

    sampled_results = [coarse_result]
    current_center = coarse_metrics["risk_time"]
    current_metrics = coarse_metrics
    steps = dragon_data.QUESTION5_LAYER_STEPS

    for level_index in range(1, len(steps)):
        step = float(steps[level_index])
        half_width = float(steps[level_index - 1])
        refined_times = build_refined_time_grid(current_center, half_width, step, lower_bound, upper_bound)
        refined_result = evaluator(refined_times)
        sampled_results.append(refined_result)
        current_metrics = extract_peak_metrics(refined_result["speed"], refined_result["time_points"])
        current_center = current_metrics["risk_time"]

    local_result = local_maximize_handle_speed(
        evaluator=evaluator,
        risk_point_index=current_metrics["risk_point_index"],
        center_time=current_metrics["risk_time"],
        half_width=float(steps[-1]),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        tol=dragon_data.QUESTION5_LOCAL_TOL,
    )

    peak_state = evaluator(np.array([local_result["peak_time"]], dtype=float))
    sampled_results.append(peak_state)
    merged = merge_sample_results(sampled_results)

    return {
        "peak_speed": float(local_result["peak_speed"]),
        "peak_time": float(local_result["peak_time"]),
        "peak_position": local_result["peak_position"],
        "risk_point_index": int(current_metrics["risk_point_index"]),
        "merged_time_points": merged["time_points"],
        "merged_speed": merged["speed"],
        "merged_position": merged["position"],
    }


def _build_question4_evaluator():
    """构造基于第四问求解器的时间评估函数。"""
    def evaluator(time_points):
        return question4.solve_question4(time_points=np.asarray(time_points, dtype=float))

    return evaluator


def analyze_question5(question4_result, evaluator=None, speed_limit=None):
    """分析第五问速度约束。"""
    if speed_limit is None:
        speed_limit = dragon_data.QUESTION5_SPEED_LIMIT

    if evaluator is None:
        search_result = {
            "peak_speed": extract_peak_metrics(question4_result["speed"], question4_result["time_points"])["peak_speed"],
            "peak_time": extract_peak_metrics(question4_result["speed"], question4_result["time_points"])["risk_time"],
            "peak_position": question4_result["position"][:, :, extract_peak_metrics(question4_result["speed"], question4_result["time_points"])["risk_time_index"]],
            "risk_point_index": extract_peak_metrics(question4_result["speed"], question4_result["time_points"])["risk_point_index"],
            "merged_time_points": np.asarray(question4_result["time_points"], dtype=float),
            "merged_speed": np.asarray(question4_result["speed"], dtype=float),
            "merged_position": np.asarray(question4_result["position"], dtype=float),
        }
    else:
        search_result = layered_search_max_speed(question4_result, evaluator)

    point_names = dragon_data.get_point_names()
    vmax = float(speed_limit / search_result["peak_speed"])
    key_speed_df = utils.build_question5_key_speed_table(
        speed=search_result["merged_speed"],
        time_points=search_result["merged_time_points"],
        point_indices=dragon_data.QUESTION5_KEY_POINT_INDICES,
    ).round(6)
    risk_df = utils.build_question5_risk_table(
        point_name=point_names[search_result["risk_point_index"]],
        point_index=search_result["risk_point_index"],
        risk_time=float(search_result["peak_time"]),
        peak_speed=float(search_result["peak_speed"]),
    ).round(6)
    conclusion_df = utils.build_question5_conclusion_table(search_result["peak_speed"], vmax).round(6)

    return {
        "question4_result": question4_result,
        "speed_limit": float(speed_limit),
        "speed": search_result["merged_speed"],
        "time_points": search_result["merged_time_points"],
        "position": search_result["merged_position"],
        "peak_position": search_result["peak_position"],
        "M": float(search_result["peak_speed"]),
        "V_max": vmax,
        "risk_point_index": int(search_result["risk_point_index"]),
        "risk_time": float(search_result["peak_time"]),
        "key_speed_df": key_speed_df,
        "risk_df": risk_df,
        "conclusion_df": conclusion_df,
        "search_result": search_result,
    }


def solve_question5(question4_result=None):
    """求解第五问。"""
    if question4_result is None:
        position_csv = dragon_data.OUTPUT_TABLES_DIR / "result4_position.csv"
        speed_csv = dragon_data.OUTPUT_TABLES_DIR / "result4_speed.csv"
        if position_csv.exists() and speed_csv.exists():
            question4_result = load_cached_question4_result(position_csv, speed_csv)
        else:
            question4_result = question4.solve_question4()
    return analyze_question5(
        question4_result,
        evaluator=_build_question4_evaluator(),
        speed_limit=dragon_data.QUESTION5_SPEED_LIMIT,
    )


def save_question5_outputs(analysis, table_dir=None, figure_dir=None):
    """保存第五问输出。"""
    if table_dir is None:
        table_dir = dragon_data.OUTPUT_TABLES_DIR
    if figure_dir is None:
        figure_dir = dragon_data.OUTPUT_FIGURES_DIR
    table_dir = Path(table_dir)
    figure_dir = Path(figure_dir)

    utils.ensure_dir(table_dir)
    utils.ensure_dir(figure_dir)

    excel_path = table_dir / "result5.xlsx"
    key_csv = table_dir / "question5_key_speed.csv"
    key_md = table_dir / "question5_key_speed.md"
    risk_csv = table_dir / "question5_risk_handle.csv"
    risk_md = table_dir / "question5_risk_handle.md"
    conclusion_csv = table_dir / "question5_conclusion.csv"
    conclusion_md = table_dir / "question5_conclusion.md"

    utils.write_result5_excel(
        {
            "关键节点": analysis["key_speed_df"],
            "风险把手": analysis["risk_df"],
            "最终结论": analysis["conclusion_df"],
        },
        excel_path,
    )
    utils.save_dataframe_exports(analysis["key_speed_df"], key_csv, key_md)
    utils.save_dataframe_exports(analysis["risk_df"], risk_csv, risk_md)
    utils.save_dataframe_exports(analysis["conclusion_df"], conclusion_csv, conclusion_md)
    figure_paths = utils.save_question5_figures(
        position=analysis["position"],
        speed=analysis["speed"],
        time_points=analysis["time_points"],
        output_dir=figure_dir,
        speed_limit=analysis["speed_limit"],
        vmax=analysis["V_max"],
        peak_time=analysis["risk_time"],
        peak_speed=analysis["M"],
    )

    return {
        "excel": excel_path,
        "key_csv": key_csv,
        "risk_csv": risk_csv,
        "conclusion_csv": conclusion_csv,
        "figures": figure_paths,
    }


def main():
    analysis = solve_question5()
    output_paths = save_question5_outputs(analysis)
    print("已生成文件：")
    print(output_paths["excel"])
    print(output_paths["key_csv"])
    print(output_paths["risk_csv"])
    print(output_paths["conclusion_csv"])
    for figure_path in output_paths["figures"]:
        print(figure_path)
    print(f"M = {analysis['M']:.6f}")
    print(f"Vmax = {analysis['V_max']:.6f} m/s")


if __name__ == "__main__":
    main()
