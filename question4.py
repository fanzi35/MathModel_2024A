"""
文件名：question4.py
用于存放问题四解决代码
"""
import numpy as np

import dragon_data
import utils


def solve_question4(time_points=None):
    """求解第四问的全局位置与速度。"""
    if time_points is None:
        time_points = dragon_data.QUESTION4_TIME_POINTS
    time_points = np.asarray(time_points, dtype=float)

    geometry = utils.solve_question4_turn_geometry(
        pitch=dragon_data.QUESTION4_PITCH,
        turn_radius=dragon_data.QUESTION4_TURN_RADIUS,
        radius_ratio=dragon_data.QUESTION4_RADIUS_RATIO,
    )

    point_count = dragon_data.POINT_COUNT
    time_count = len(time_points)
    distances = dragon_data.get_handle_distances()

    s_values = np.zeros((point_count, time_count), dtype=float)
    s_dot = np.zeros((point_count, time_count), dtype=float)
    position = np.zeros((point_count, 2, time_count), dtype=float)
    speed = np.zeros((point_count, time_count), dtype=float)

    for time_index, current_time in enumerate(time_points):
        s_values[0, time_index] = dragon_data.QUESTION4_HEAD_SPEED * current_time
        tangents = []
        for point_index in range(point_count):
            if point_index > 0:
                s_values[point_index, time_index] = utils.solve_trailing_path_parameter(
                    s_prev=s_values[point_index - 1, time_index],
                    distance=distances[point_index - 1],
                    geometry=geometry,
                )
            position[point_index, :, time_index] = utils.question4_path_position(s_values[point_index, time_index], geometry)
            tangents.append(utils.question4_path_tangent(s_values[point_index, time_index], geometry))

        s_dot[0, time_index] = dragon_data.QUESTION4_HEAD_SPEED
        for point_index in range(1, point_count):
            delta = position[point_index, :, time_index] - position[point_index - 1, :, time_index]
            numerator = float(np.dot(delta, tangents[point_index - 1]))
            denominator = float(np.dot(delta, tangents[point_index]))
            s_dot[point_index, time_index] = (numerator / denominator) * s_dot[point_index - 1, time_index]

        speed[:, time_index] = np.abs(s_dot[:, time_index])

    position_df = utils.build_position_dataframe(position, time_points).round(6)
    speed_df = utils.build_speed_dataframe(speed, time_points).round(6)
    summary_position_df, summary_speed_df = utils.build_question4_summary_tables(position_df, speed_df)

    return {
        "geometry": geometry,
        "time_points": time_points,
        "s": s_values,
        "s_dot": s_dot,
        "position": position,
        "speed": speed,
        "position_df": position_df,
        "speed_df": speed_df,
        "summary_position_df": summary_position_df.round(6),
        "summary_speed_df": summary_speed_df.round(6),
    }


def save_question4_outputs(result):
    """保存第四问的表格和图像。"""
    utils.ensure_dir(dragon_data.OUTPUT_TABLES_DIR)
    utils.ensure_dir(dragon_data.OUTPUT_FIGURES_DIR)

    excel_path = dragon_data.OUTPUT_TABLES_DIR / "result4.xlsx"
    position_csv = dragon_data.OUTPUT_TABLES_DIR / "result4_position.csv"
    speed_csv = dragon_data.OUTPUT_TABLES_DIR / "result4_speed.csv"
    position_md = dragon_data.OUTPUT_TABLES_DIR / "result4_position.md"
    speed_md = dragon_data.OUTPUT_TABLES_DIR / "result4_speed.md"
    summary_position_csv = dragon_data.OUTPUT_TABLES_DIR / "question4_summary_position.csv"
    summary_speed_csv = dragon_data.OUTPUT_TABLES_DIR / "question4_summary_speed.csv"
    summary_position_md = dragon_data.OUTPUT_TABLES_DIR / "question4_summary_position.md"
    summary_speed_md = dragon_data.OUTPUT_TABLES_DIR / "question4_summary_speed.md"

    utils.write_result4_excel(result["position_df"], result["speed_df"], excel_path)
    utils.save_dataframe_exports(result["position_df"], position_csv, position_md)
    utils.save_dataframe_exports(result["speed_df"], speed_csv, speed_md)
    utils.save_dataframe_exports(result["summary_position_df"], summary_position_csv, summary_position_md)
    utils.save_dataframe_exports(result["summary_speed_df"], summary_speed_csv, summary_speed_md)
    schematic_path = utils.save_question4_turn_schematic(
        geometry=result["geometry"],
        output_path=dragon_data.OUTPUT_FIGURES_DIR / "question4_turn_schematic.png",
    )
    figure_paths = utils.save_question4_figures(
        geometry=result["geometry"],
        s_values=result["s"],
        position=result["position"],
        speed=result["speed"],
        time_points=result["time_points"],
        output_dir=dragon_data.OUTPUT_FIGURES_DIR,
    )

    return {
        "excel": excel_path,
        "position_csv": position_csv,
        "speed_csv": speed_csv,
        "summary_position_csv": summary_position_csv,
        "summary_speed_csv": summary_speed_csv,
        "schematic": schematic_path,
        "figures": (schematic_path,) + tuple(figure_paths),
    }


def main():
    result = solve_question4()
    output_paths = save_question4_outputs(result)
    print("已生成文件：")
    print(output_paths["excel"])
    print(output_paths["position_csv"])
    print(output_paths["speed_csv"])
    print(output_paths["summary_position_csv"])
    print(output_paths["summary_speed_csv"])
    for figure_path in output_paths["figures"]:
        print(figure_path)


if __name__ == "__main__":
    main()
