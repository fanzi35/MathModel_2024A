import math

import numpy as np

import dragon_data
import question1
import utils


def _time_key(time_value):
    """统一时间缓存键。"""
    return round(float(time_value), 8)


def _build_cache():
    """构建第二问缓存。"""
    return {
        "state": {},
        "collision": {},
        "gap_history": {},
    }


def _extract_state_slice(result, column_index):
    """从第一问结果中提取单时刻状态。"""
    return {
        "time": float(result["time_points"][column_index]),
        "theta": result["theta"][:, column_index].copy(),
        "theta_dot": result["theta_dot"][:, column_index].copy(),
        "radius": result["radius"][:, column_index].copy(),
        "position": result["position"][:, :, column_index].copy(),
        "speed": result["speed"][:, column_index].copy(),
    }


def get_states_for_times(time_points, cache=None):
    """批量获取给定时刻的第一问状态。"""
    if cache is None:
        cache = _build_cache()

    ordered_times = [float(value) for value in time_points]
    missing_times = []
    for time_value in ordered_times:
        if _time_key(time_value) not in cache["state"]:
            missing_times.append(time_value)

    if missing_times:
        unique_times = np.array(sorted(set(missing_times)), dtype=float)
        result = question1.solve_question1(unique_times)
        for column_index, time_value in enumerate(unique_times):
            cache["state"][_time_key(time_value)] = _extract_state_slice(result, column_index)

    return [cache["state"][_time_key(time_value)] for time_value in ordered_times]


def get_state_at_time(time_value, cache=None):
    """获取单个时刻的第一问状态。"""
    return get_states_for_times([time_value], cache=cache)[0]


def get_head_collision_candidate_indices(theta):
    """按与龙头前把手极角差小于 4π 选取候选龙身板凳。"""
    theta_head = theta[0]
    theta_front = theta[:-1]
    candidate_indices = [
        bench_index
        for bench_index in range(2, len(theta_front))
        if 0.0 < theta_front[bench_index] - theta_head < 4.0 * np.pi
    ]
    return candidate_indices, theta_front


def build_bench_rectangles(position):
    """由把手点位置构造全部板凳矩形。"""
    bench_lengths = dragon_data.get_bench_lengths()
    rectangles = []
    for bench_index in range(dragon_data.BENCH_COUNT):
        rectangles.append(
            utils.build_bench_rectangle(
                front_point=position[bench_index],
                rear_point=position[bench_index + 1],
                bench_length=bench_lengths[bench_index],
                bench_width=dragon_data.BENCH_WIDTH,
                bench_index=bench_index,
            )
        )
    return rectangles


def build_candidate_pairs(candidate_indices):
    """只构造龙头板凳与候选龙身板凳的比较对。"""
    return [(0, bench_index) for bench_index in candidate_indices]


def evaluate_collision_from_state(state):
    """在给定时刻状态下计算最小净间隙。"""
    position = state["position"]
    theta = state["theta"]

    candidate_indices, _ = get_head_collision_candidate_indices(theta)
    rectangles = build_bench_rectangles(position)
    candidate_pairs = build_candidate_pairs(candidate_indices)

    min_gap = math.inf
    best_pair = None
    best_sat = None
    for head_index, bench_index in candidate_pairs:
        sat_result = utils.rectangle_sat_gap(rectangles[head_index], rectangles[bench_index])
        if sat_result["gap"] < min_gap:
            min_gap = sat_result["gap"]
            best_pair = (head_index, bench_index)
            best_sat = sat_result

    if best_pair is None:
        raise ValueError("未找到龙头板凳与内两层候选龙身板凳的比较对")

    return {
        "time": state["time"],
        "global_gap": float(min_gap),
        "best_pair": best_pair,
        "best_sat": best_sat,
        "rectangles": rectangles,
        "candidate_indices": candidate_indices,
        "candidate_pairs": candidate_pairs,
        "state": state,
    }


def evaluate_collision_state(time_value, cache=None):
    """计算某个时刻的碰撞状态。"""
    if cache is None:
        cache = _build_cache()

    key = _time_key(time_value)
    if key not in cache["collision"]:
        state = get_state_at_time(time_value, cache=cache)
        collision_state = evaluate_collision_from_state(state)
        cache["collision"][key] = collision_state
        cache["gap_history"][round(collision_state["time"], 6)] = collision_state["global_gap"]
    return cache["collision"][key]


def evaluate_collision_times(time_points, cache=None):
    """批量计算多个时刻的碰撞状态。"""
    if cache is None:
        cache = _build_cache()

    states = get_states_for_times(time_points, cache=cache)
    results = []
    for state in states:
        key = _time_key(state["time"])
        if key not in cache["collision"]:
            collision_state = evaluate_collision_from_state(state)
            cache["collision"][key] = collision_state
            cache["gap_history"][round(collision_state["time"], 6)] = collision_state["global_gap"]
        results.append(cache["collision"][key])
    return results


def _make_time_grid(start_time, end_time, step):
    """构造包含端点的时间网格。"""
    steps = int(round((end_time - start_time) / step))
    return np.array([round(start_time + index * step, 8) for index in range(steps + 1)], dtype=float)


def layered_refine_collision_interval(
    cache=None,
    scan_steps=None,
    max_time=None,
):
    """按 1s/0.1s/0.01s 分层细化碰撞区间。"""
    if cache is None:
        cache = _build_cache()
    if scan_steps is None:
        scan_steps = dragon_data.QUESTION2_SCAN_STEPS
    if max_time is None:
        max_time = dragon_data.QUESTION2_MAX_TIME
    b = utils.spiral_coefficient(dragon_data.QUESTION1_PITCH)
    arc_time_limit = float(utils.spiral_arc_length(dragon_data.QUESTION1_INITIAL_THETA, b))
    max_time = min(float(max_time), arc_time_limit - 1e-6)

    initial_state = evaluate_collision_state(0.0, cache=cache)
    if initial_state["global_gap"] <= 0:
        raise ValueError("初始时刻已经碰撞，无法继续搜索")

    level_records = []
    left_time = 0.0
    right_time = None
    for level_index, step in enumerate(scan_steps):
        if level_index == 0:
            time_grid = _make_time_grid(0.0, max_time, step)
        else:
            time_grid = _make_time_grid(left_time, right_time, step)
        collision_states = []
        for time_value in time_grid:
            collision_states.append(evaluate_collision_state(time_value, cache=cache))
            if len(collision_states) >= 2:
                previous_state = collision_states[-2]
                current_state = collision_states[-1]
                if previous_state["global_gap"] > 0.0 and current_state["global_gap"] <= 0.0:
                    break
        new_left = None
        new_right = None
        for index in range(1, len(collision_states)):
            previous_state = collision_states[index - 1]
            current_state = collision_states[index]
            if previous_state["global_gap"] > 0.0 and current_state["global_gap"] <= 0.0:
                new_left = float(previous_state["time"])
                new_right = float(current_state["time"])
                level_records.append(
                    {
                        "step": step,
                        "left_time": new_left,
                        "right_time": new_right,
                        "left_gap": previous_state["global_gap"],
                        "right_gap": current_state["global_gap"],
                    }
                )
                break
        if new_right is None:
            raise ValueError(f"在步长 {step} s 下未找到碰撞区间，最大搜索时间 {max_time} s 不足")
        left_time, right_time = new_left, new_right

    return {
        "left_time": left_time,
        "right_time": right_time,
        "levels": level_records,
        "cache": cache,
    }


def bisect_collision_time(left_time, right_time, cache=None, tolerance=None):
    """二分法求碰撞临界时刻。"""
    if cache is None:
        cache = _build_cache()
    if tolerance is None:
        tolerance = dragon_data.QUESTION2_TIME_TOL

    left_state = evaluate_collision_state(left_time, cache=cache)
    right_state = evaluate_collision_state(right_time, cache=cache)
    if not (left_state["global_gap"] > 0.0 and right_state["global_gap"] <= 0.0):
        raise ValueError("二分区间不满足左安全右碰撞")

    iterations = []
    while right_time - left_time > tolerance:
        middle_time = 0.5 * (left_time + right_time)
        middle_state = evaluate_collision_state(middle_time, cache=cache)
        iterations.append(
            {
                "left_time": left_time,
                "middle_time": middle_time,
                "right_time": right_time,
                "middle_gap": middle_state["global_gap"],
            }
        )
        if middle_state["global_gap"] > 0.0:
            left_time = middle_time
            left_state = middle_state
        else:
            right_time = middle_time
            right_state = middle_state

    return {
        "safe_time": left_time,
        "collision_time": right_time,
        "safe_state": left_state,
        "collision_state": right_state,
        "iterations": iterations,
        "cache": cache,
    }


def solve_question2(
    scan_steps=None,
    max_time=None,
    tolerance=None,
):
    """求解第二问。"""
    cache = _build_cache()
    refined_interval = layered_refine_collision_interval(
        cache=cache,
        scan_steps=scan_steps,
        max_time=max_time,
    )
    bisection_result = bisect_collision_time(
        left_time=refined_interval["left_time"],
        right_time=refined_interval["right_time"],
        cache=cache,
        tolerance=tolerance,
    )

    final_state = bisection_result["safe_state"]["state"]
    result2_df = utils.build_result2_dataframe(final_state["position"], final_state["speed"]).round(6)
    summary_df = utils.build_question2_summary_dataframe(result2_df).round(6)

    return {
        "safe_time": bisection_result["safe_time"],
        "collision_time": bisection_result["collision_time"],
        "safe_gap": bisection_result["safe_state"]["global_gap"],
        "collision_gap": bisection_result["collision_state"]["global_gap"],
        "final_collision_state": bisection_result["safe_state"],
        "result2_df": result2_df,
        "summary_df": summary_df,
        "refined_interval": refined_interval,
        "bisection": bisection_result,
        "gap_history": dict(sorted(cache["gap_history"].items())),
    }


def save_question2_outputs(result):
    """保存第二问表格和图像。"""
    utils.ensure_dir(dragon_data.OUTPUT_TABLES_DIR)
    utils.ensure_dir(dragon_data.OUTPUT_FIGURES_DIR)

    result2_excel = dragon_data.OUTPUT_TABLES_DIR / "result2.xlsx"
    result2_csv = dragon_data.OUTPUT_TABLES_DIR / "result2_full.csv"
    result2_md = dragon_data.OUTPUT_TABLES_DIR / "result2_full.md"
    summary_csv = dragon_data.OUTPUT_TABLES_DIR / "question2_summary.csv"
    summary_md = dragon_data.OUTPUT_TABLES_DIR / "question2_summary.md"

    utils.write_result2_excel(result["result2_df"], result2_excel)
    utils.save_dataframe_exports(result["result2_df"], result2_csv, result2_md)
    utils.save_dataframe_exports(result["summary_df"], summary_csv, summary_md)

    gap_figure = dragon_data.OUTPUT_FIGURES_DIR / "question2_gap_curve.png"
    local_figure = dragon_data.OUTPUT_FIGURES_DIR / "question2_local_geometry.png"
    sat_figure = dragon_data.OUTPUT_FIGURES_DIR / "question2_sat_projection.png"

    utils.save_question2_gap_curve(
        result["gap_history"],
        gap_figure,
        safe_time=result["safe_time"],
        collision_time=result["collision_time"],
    )
    collision_state = result["final_collision_state"]
    utils.save_question2_local_geometry(
        rectangles=collision_state["rectangles"],
        focus_pair=collision_state["best_pair"],
        layer_indices=([0], collision_state["candidate_indices"]),
        output_path=local_figure,
        theta=collision_state["state"]["theta"],
    )
    inner_index, outer_index = collision_state["best_pair"]
    utils.save_question2_sat_projection(
        rectangle_a=collision_state["rectangles"][inner_index],
        rectangle_b=collision_state["rectangles"][outer_index],
        sat_result=collision_state["best_sat"],
        output_path=sat_figure,
    )

    return {
        "excel": result2_excel,
        "full_csv": result2_csv,
        "summary_csv": summary_csv,
        "gap_figure": gap_figure,
        "local_figure": local_figure,
        "sat_figure": sat_figure,
    }


def main():
    result = solve_question2()
    output_paths = save_question2_outputs(result)
    print("已生成文件：")
    print(output_paths["excel"])
    print(output_paths["full_csv"])
    print(output_paths["summary_csv"])
    print(output_paths["gap_figure"])
    print(output_paths["local_figure"])
    print(output_paths["sat_figure"])
    print(f"终止时刻(安全侧) = {result['safe_time']:.6f} s")
    print(f"碰撞时刻(碰撞侧) = {result['collision_time']:.6f} s")


if __name__ == "__main__":
    main()
