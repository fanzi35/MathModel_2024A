from pathlib import Path

import numpy as np

import dragon_data
import utils


def solve_question1(time_points=None):
    """求解第一问的位置和速度。"""
    if time_points is None:
        time_points = dragon_data.QUESTION1_TIME_POINTS
    time_points = np.asarray(time_points, dtype=float)

    b = utils.spiral_coefficient(dragon_data.QUESTION1_PITCH)
    distances = dragon_data.get_handle_distances()
    point_count = dragon_data.POINT_COUNT
    time_count = len(time_points)

    theta = np.zeros((point_count, time_count), dtype=float)
    theta_dot = np.zeros((point_count, time_count), dtype=float)
    radius = np.zeros((point_count, time_count), dtype=float)
    position = np.zeros((point_count, 2, time_count), dtype=float)
    speed = np.zeros((point_count, time_count), dtype=float)

    initial_arc_length = utils.spiral_arc_length(dragon_data.QUESTION1_INITIAL_THETA, b)

    previous_head_theta = dragon_data.QUESTION1_INITIAL_THETA
    for time_index, current_time in enumerate(time_points):
        target_arc_length = initial_arc_length - dragon_data.QUESTION1_HEAD_SPEED * current_time
        theta[0, time_index] = utils.solve_theta_from_arc_length(
            target_arc_length=target_arc_length,
            b=b,
            upper_bound=previous_head_theta,
        )
        previous_head_theta = theta[0, time_index]

        for point_index in range(1, point_count):
            theta[point_index, time_index] = utils.solve_trailing_theta(
                theta_prev=theta[point_index - 1, time_index],
                distance=distances[point_index - 1],
                b=b,
            )

        radius[:, time_index] = b * theta[:, time_index]
        x, y = utils.polar_to_cartesian(radius[:, time_index], theta[:, time_index])
        position[:, 0, time_index] = x
        position[:, 1, time_index] = y

        theta_dot[0, time_index] = -1.0 / (b * np.sqrt(1.0 + theta[0, time_index] ** 2))
        for point_index in range(1, point_count):
            delta = theta[point_index, time_index] - theta[point_index - 1, time_index]
            numerator = (
                theta[point_index - 1, time_index]
                - theta[point_index, time_index] * np.cos(delta)
                - theta[point_index, time_index] * theta[point_index - 1, time_index] * np.sin(delta)
            )
            denominator = (
                theta[point_index, time_index]
                - theta[point_index - 1, time_index] * np.cos(delta)
                + theta[point_index, time_index] * theta[point_index - 1, time_index] * np.sin(delta)
            )
            theta_dot[point_index, time_index] = -(numerator / denominator) * theta_dot[point_index - 1, time_index]

        speed[:, time_index] = utils.speed_from_theta(theta[:, time_index], theta_dot[:, time_index], b)

    position_df = utils.build_position_dataframe(position, time_points).round(6)
    speed_df = utils.build_speed_dataframe(speed, time_points).round(6)
    summary_position_df, summary_speed_df = utils.build_summary_tables(position_df, speed_df)

    return {
        "time_points": time_points,
        "theta": theta,
        "theta_dot": theta_dot,
        "radius": radius,
        "position": position,
        "speed": speed,
        "position_df": position_df,
        "speed_df": speed_df,
        "summary_position_df": summary_position_df.round(6),
        "summary_speed_df": summary_speed_df.round(6),
    }


def save_question1_outputs(result):
    """保存第一问的表格和图像。"""
    utils.ensure_dir(dragon_data.OUTPUT_TABLES_DIR)
    utils.ensure_dir(dragon_data.OUTPUT_FIGURES_DIR)

    excel_path = dragon_data.OUTPUT_TABLES_DIR / "result1.xlsx"
    position_csv = dragon_data.OUTPUT_TABLES_DIR / "result1_position.csv"
    speed_csv = dragon_data.OUTPUT_TABLES_DIR / "result1_speed.csv"
    position_md = dragon_data.OUTPUT_TABLES_DIR / "result1_position.md"
    speed_md = dragon_data.OUTPUT_TABLES_DIR / "result1_speed.md"
    summary_position_csv = dragon_data.OUTPUT_TABLES_DIR / "question1_summary_position.csv"
    summary_speed_csv = dragon_data.OUTPUT_TABLES_DIR / "question1_summary_speed.csv"
    summary_position_md = dragon_data.OUTPUT_TABLES_DIR / "question1_summary_position.md"
    summary_speed_md = dragon_data.OUTPUT_TABLES_DIR / "question1_summary_speed.md"

    utils.write_result1_excel(
        position_df=result["position_df"],
        speed_df=result["speed_df"],
        output_path=excel_path,
    )
    utils.save_dataframe_exports(result["position_df"], position_csv, position_md)
    utils.save_dataframe_exports(result["speed_df"], speed_csv, speed_md)
    utils.save_dataframe_exports(result["summary_position_df"], summary_position_csv, summary_position_md)
    utils.save_dataframe_exports(result["summary_speed_df"], summary_speed_csv, summary_speed_md)
    figure_paths = utils.save_question1_figures(
        position=result["position"],
        speed=result["speed"],
        time_points=result["time_points"],
    )

    return {
        "excel": excel_path,
        "position_csv": position_csv,
        "speed_csv": speed_csv,
        "summary_position_csv": summary_position_csv,
        "summary_speed_csv": summary_speed_csv,
        "figures": figure_paths,
    }


def main():
    result = solve_question1()
    output_paths = save_question1_outputs(result)
    print("已生成文件：")
    print(output_paths["excel"])
    print(output_paths["position_csv"])
    print(output_paths["speed_csv"])
    for figure_path in output_paths["figures"]:
        print(figure_path)


if __name__ == "__main__":
    main()
