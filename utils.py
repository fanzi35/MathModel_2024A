import shutil
import uuid
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dragon_data


SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
ET.register_namespace("", SHEET_NS)


def ensure_dir(path):
    """创建目录。"""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_question2_plot_labels():
    """返回第二问图像标签。"""
    return dragon_data.get_question2_plot_labels()


def configure_chinese_plotting():
    """设置中文绘图字体。"""
    plt.rcParams["font.sans-serif"] = dragon_data.QUESTION2_CHINESE_FONTS
    plt.rcParams["axes.unicode_minus"] = False


def spiral_coefficient(pitch):
    """根据螺距返回阿基米德螺线系数。"""
    return pitch / (2.0 * np.pi)


def spiral_arc_length(theta, b):
    """返回从原点到参数 theta 处的螺线弧长。"""
    theta = np.asarray(theta, dtype=float)
    return 0.5 * b * (theta * np.sqrt(1.0 + theta ** 2) + np.arcsinh(theta))


def solve_theta_from_arc_length(target_arc_length, b, upper_bound, tol=1e-12, max_iter=100):
    """由弧长反求龙头参数。"""
    low = 0.0
    high = float(upper_bound)
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        if spiral_arc_length(mid, b) < target_arc_length:
            low = mid
        else:
            high = mid
        if high - low < tol:
            break
    return 0.5 * (low + high)


def polar_to_cartesian(radius, theta):
    """极坐标转直角坐标。"""
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y


def point_distance_sq(theta_a, theta_b, b):
    """返回螺线两点间的平方距离。"""
    radius_a = b * theta_a
    radius_b = b * theta_b
    delta = theta_a - theta_b
    return radius_a ** 2 + radius_b ** 2 - 2.0 * radius_a * radius_b * np.cos(delta)


def solve_trailing_theta(theta_prev, distance, b, tol=1e-12, max_iter=100):
    """由前一点参数递推求后一点参数。"""
    arc_step = distance / (b * np.sqrt(1.0 + theta_prev ** 2))
    low = theta_prev
    high = theta_prev + max(arc_step * 2.0, 1e-4)

    while point_distance_sq(high, theta_prev, b) < distance ** 2:
        high += max(arc_step, 1e-4)
        arc_step *= 1.5

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        if point_distance_sq(mid, theta_prev, b) < distance ** 2:
            low = mid
        else:
            high = mid
        if high - low < tol:
            break
    return 0.5 * (low + high)


def speed_from_theta(theta, theta_dot, b):
    """由参数速度换算实际速度大小。"""
    return np.abs(b * np.sqrt(1.0 + theta ** 2) * theta_dot)


def spiral_tangent(theta, b):
    """返回阿基米德螺线对参数 theta 的导数向量。"""
    return np.array(
        [
            b * (np.cos(theta) - theta * np.sin(theta)),
            b * (np.sin(theta) + theta * np.cos(theta)),
        ],
        dtype=float,
    )


def circle_tangent(center, point, orientation):
    """返回圆上某点沿给定方向的单位切向量。"""
    radial = normalize_vector(np.asarray(point, dtype=float) - np.asarray(center, dtype=float))
    return orientation * perpendicular_vector(radial)


def choose_circle_orientation(center, point, desired_tangent):
    """根据期望切向选择圆弧正反方向。"""
    tangent_ccw = circle_tangent(center, point, 1.0)
    if np.dot(tangent_ccw, desired_tangent) >= 0.0:
        return 1.0
    return -1.0


def oriented_arc_angle(start_vector, end_vector, orientation):
    """计算给定方向下的圆弧圆心角。"""
    start_angle = np.arctan2(start_vector[1], start_vector[0])
    end_angle = np.arctan2(end_vector[1], end_vector[0])
    if orientation > 0:
        delta = end_angle - start_angle
    else:
        delta = start_angle - end_angle
    while delta < 0.0:
        delta += 2.0 * np.pi
    return float(delta)


def _solve_positive_quadratic_roots(a, b, c):
    """返回二次方程的所有正实根。"""
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return []
        root = -c / b
        return [float(root)] if root > 0 else []
    delta = b ** 2 - 4.0 * a * c
    if delta < 0:
        return []
    sqrt_delta = np.sqrt(max(delta, 0.0))
    roots = [(-b + sqrt_delta) / (2.0 * a), (-b - sqrt_delta) / (2.0 * a)]
    return sorted(float(root) for root in roots if root > 0)


def _arc_points_within_turn_circle(center, radius, start_point, end_point, orientation, turn_radius):
    """采样检查圆弧是否位于调头圆内。"""
    start_vector = np.asarray(start_point, dtype=float) - np.asarray(center, dtype=float)
    end_vector = np.asarray(end_point, dtype=float) - np.asarray(center, dtype=float)
    total_angle = oriented_arc_angle(start_vector, end_vector, orientation)
    start_angle = np.arctan2(start_vector[1], start_vector[0])
    sample_values = np.linspace(0.0, total_angle, 80)
    angles = start_angle + orientation * sample_values
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    radius_values = np.sqrt(x ** 2 + y ** 2)
    return bool(np.all(radius_values <= turn_radius + 1e-7))


def solve_question4_turn_geometry(pitch, turn_radius, radius_ratio=2.0):
    """求解第四问调头几何。"""
    b = spiral_coefficient(pitch)
    theta_a = turn_radius / b
    A = np.array(polar_to_cartesian(turn_radius, theta_a), dtype=float)
    B = -A

    incoming_tangent = -normalize_vector(spiral_tangent(theta_a, b))
    outgoing_tangent = normalize_vector(-spiral_tangent(theta_a, b))
    incoming_normal = perpendicular_vector(incoming_tangent)
    outgoing_normal = perpendicular_vector(outgoing_tangent)

    best_geometry = None
    u = A - B
    for sigma_1 in (1.0, -1.0):
        for sigma_2 in (1.0, -1.0):
            v = radius_ratio * sigma_1 * incoming_normal - sigma_2 * outgoing_normal
            roots = _solve_positive_quadratic_roots(
                np.dot(v, v) - (radius_ratio + 1.0) ** 2,
                2.0 * np.dot(u, v),
                np.dot(u, u),
            )
            for radius_2 in roots:
                radius_1 = radius_ratio * radius_2
                C1 = A + radius_1 * sigma_1 * incoming_normal
                C2 = B + radius_2 * sigma_2 * outgoing_normal
                D = C1 + (radius_1 / (radius_1 + radius_2)) * (C2 - C1)

                eps1 = choose_circle_orientation(C1, A, incoming_tangent)
                eps2 = choose_circle_orientation(C2, B, outgoing_tangent)
                tangent_d1 = circle_tangent(C1, D, eps1)
                tangent_d2 = circle_tangent(C2, D, eps2)
                if np.dot(tangent_d1, tangent_d2) < 0.99:
                    continue

                phi1 = oriented_arc_angle(A - C1, D - C1, eps1)
                phi2 = oriented_arc_angle(D - C2, B - C2, eps2)
                if phi1 <= 0.0 or phi2 <= 0.0:
                    continue
                if not _arc_points_within_turn_circle(C1, radius_1, A, D, eps1, turn_radius):
                    continue
                if not _arc_points_within_turn_circle(C2, radius_2, D, B, eps2, turn_radius):
                    continue

                length_1 = radius_1 * phi1
                length_2 = radius_2 * phi2
                total_length = length_1 + length_2
                candidate = {
                    "b": b,
                    "theta_A": float(theta_a),
                    "A": A,
                    "B": B,
                    "C1": C1,
                    "C2": C2,
                    "D": D,
                    "R1": float(radius_1),
                    "R2": float(radius_2),
                    "phi1": float(phi1),
                    "phi2": float(phi2),
                    "L1": float(length_1),
                    "L2": float(length_2),
                    "eps1": float(eps1),
                    "eps2": float(eps2),
                    "incoming_tangent": incoming_tangent,
                    "outgoing_tangent": outgoing_tangent,
                }
                if best_geometry is None or total_length < best_geometry["total_length"]:
                    candidate["total_length"] = float(total_length)
                    best_geometry = candidate

    if best_geometry is None:
        raise ValueError("未找到第四问调头几何的可行解")

    best_geometry["spiral_arc_at_A"] = float(spiral_arc_length(best_geometry["theta_A"], best_geometry["b"]))
    return best_geometry


def solve_question4_spiral_theta_from_s(s_value, geometry, outgoing=False):
    """由第四问全局弧长求盘入或盘出螺线参数。"""
    target_arc = geometry["spiral_arc_at_A"] - s_value if not outgoing else geometry["spiral_arc_at_A"] + s_value
    upper_bound = max(geometry["theta_A"] + abs(s_value) / geometry["b"] + 2.0, geometry["theta_A"] + 1.0)
    return solve_theta_from_arc_length(target_arc, geometry["b"], upper_bound=upper_bound)


def question4_path_position(s_value, geometry):
    """返回第四问全局路径上弧长 s 对应的位置。"""
    s_value = float(s_value)
    if s_value < 0.0:
        theta = solve_question4_spiral_theta_from_s(s_value, geometry, outgoing=False)
        radius = geometry["b"] * theta
        return np.array(polar_to_cartesian(radius, theta), dtype=float)
    if s_value <= geometry["L1"]:
        angle_start = np.arctan2(geometry["A"][1] - geometry["C1"][1], geometry["A"][0] - geometry["C1"][0])
        angle = angle_start + geometry["eps1"] * (s_value / geometry["R1"])
        return np.array(
            [geometry["C1"][0] + geometry["R1"] * np.cos(angle), geometry["C1"][1] + geometry["R1"] * np.sin(angle)],
            dtype=float,
        )
    if s_value <= geometry["L1"] + geometry["L2"]:
        angle_start = np.arctan2(geometry["D"][1] - geometry["C2"][1], geometry["D"][0] - geometry["C2"][0])
        angle = angle_start + geometry["eps2"] * ((s_value - geometry["L1"]) / geometry["R2"])
        return np.array(
            [geometry["C2"][0] + geometry["R2"] * np.cos(angle), geometry["C2"][1] + geometry["R2"] * np.sin(angle)],
            dtype=float,
        )
    theta = solve_question4_spiral_theta_from_s(s_value - geometry["L1"] - geometry["L2"], geometry, outgoing=True)
    radius = geometry["b"] * theta
    return -np.array(polar_to_cartesian(radius, theta), dtype=float)


def question4_path_tangent(s_value, geometry):
    """返回第四问全局路径上弧长 s 对应的单位切向量。"""
    s_value = float(s_value)
    if s_value < 0.0:
        theta = solve_question4_spiral_theta_from_s(s_value, geometry, outgoing=False)
        return -normalize_vector(spiral_tangent(theta, geometry["b"]))
    if s_value <= geometry["L1"]:
        point = question4_path_position(s_value, geometry)
        return circle_tangent(geometry["C1"], point, geometry["eps1"])
    if s_value <= geometry["L1"] + geometry["L2"]:
        point = question4_path_position(s_value, geometry)
        return circle_tangent(geometry["C2"], point, geometry["eps2"])
    theta = solve_question4_spiral_theta_from_s(s_value - geometry["L1"] - geometry["L2"], geometry, outgoing=True)
    return normalize_vector(-spiral_tangent(theta, geometry["b"]))


def solve_trailing_path_parameter(s_prev, distance, geometry, tol=1e-10, max_iter=100):
    """沿第四问全局路径递推求解后一个把手的弧长参数。"""
    high = float(s_prev)
    reference_point = question4_path_position(high, geometry)
    low = high - max(distance * 1.2, 1e-3)

    def gap(s_value):
        delta = question4_path_position(s_value, geometry) - reference_point
        return float(np.dot(delta, delta) - distance ** 2)

    while gap(low) < 0.0:
        low -= max(distance, 1e-3)

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        if gap(mid) < 0.0:
            high = mid
        else:
            low = mid
        if high - low < tol:
            break
    return 0.5 * (low + high)


def build_position_dataframe(position, time_points):
    """将第一问位置结果整理为表格。"""
    columns = dragon_data.get_time_labels(time_points)
    rows = []
    for point_index in range(position.shape[0]):
        rows.append(position[point_index, 0, :])
        rows.append(position[point_index, 1, :])
    data = np.vstack(rows)
    return pd.DataFrame(data, index=dragon_data.get_position_row_labels(), columns=columns)


def build_speed_dataframe(speed, time_points):
    """将第一问速度结果整理为表格。"""
    columns = dragon_data.get_time_labels(time_points)
    return pd.DataFrame(speed, index=dragon_data.get_speed_row_labels(), columns=columns)


def build_summary_tables(position_df, speed_df):
    """整理论文需要的第一问摘要表。"""
    key_columns = [column for column in dragon_data.get_time_labels(dragon_data.QUESTION1_KEY_TIMES) if column in position_df.columns]

    position_rows = [
        "龙头x (m)",
        "龙头y (m)",
        "第1节龙身x (m)",
        "第1节龙身y (m)",
        "第51节龙身x (m)",
        "第51节龙身y (m)",
        "第101节龙身x (m)",
        "第101节龙身y (m)",
        "第151节龙身x (m)",
        "第151节龙身y (m)",
        "第201节龙身x (m)",
        "第201节龙身y (m)",
        "龙尾（后）x (m)",
        "龙尾（后）y (m)",
    ]
    speed_rows = [
        "龙头 (m/s)",
        "第1节龙身 (m/s)",
        "第51节龙身 (m/s)",
        "第101节龙身 (m/s)",
        "第151节龙身 (m/s)",
        "第201节龙身 (m/s)",
        "龙尾（后） (m/s)",
    ]
    return position_df.loc[position_rows, key_columns], speed_df.loc[speed_rows, key_columns]


def build_result2_dataframe(position_column, speed_column):
    """将第二问单时刻结果整理为表格。"""
    data = np.column_stack([position_column[:, 0], position_column[:, 1], speed_column])
    return pd.DataFrame(
        data,
        index=dragon_data.get_result2_row_labels(),
        columns=["横坐标x (m)", "纵坐标y (m)", "速度 (m/s)"],
    )


def build_question2_summary_dataframe(result2_df):
    """提取第二问要求的关键节点表。"""
    return result2_df.loc[dragon_data.get_question2_summary_labels()]


def build_question4_summary_tables(position_df, speed_df):
    """整理第四问论文可用摘要表。"""
    key_columns = [
        column for column in dragon_data.get_time_labels(dragon_data.QUESTION4_KEY_TIMES)
        if column in position_df.columns
    ]
    point_names = dragon_data.get_point_names()
    position_rows = []
    for point_index in dragon_data.QUESTION4_KEY_POINT_INDICES:
        position_rows.append(f"{point_names[point_index]}x (m)")
        position_rows.append(f"{point_names[point_index]}y (m)")
    speed_rows = [f"{point_names[point_index]} (m/s)" for point_index in dragon_data.QUESTION4_KEY_POINT_INDICES]
    return position_df.loc[position_rows, key_columns], speed_df.loc[speed_rows, key_columns]


def save_dataframe_exports(dataframe, csv_path, markdown_path):
    """同时保存 csv 与 markdown 表格。"""
    dataframe.to_csv(csv_path, encoding="utf-8-sig")
    try:
        content = dataframe.to_markdown()
    except Exception:
        content = dataframe.to_string()
    markdown_path.write_text(content, encoding="utf-8")


def excel_column_name(index):
    """数字列号转 Excel 列名。"""
    name = []
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        name.append(chr(ord("A") + remainder))
    return "".join(reversed(name))


def create_sheet_data(dataframe):
    """将 DataFrame 转为 sheetData 节点。"""
    sheet_data = ET.Element(f"{{{SHEET_NS}}}sheetData")

    header_row = ET.SubElement(sheet_data, f"{{{SHEET_NS}}}row", {"r": "1"})
    for column_index, column_name in enumerate(dataframe.columns, start=2):
        cell = ET.SubElement(
            header_row,
            f"{{{SHEET_NS}}}c",
            {"r": f"{excel_column_name(column_index)}1", "t": "inlineStr"},
        )
        text_node = ET.SubElement(ET.SubElement(cell, f"{{{SHEET_NS}}}is"), f"{{{SHEET_NS}}}t")
        text_node.text = str(column_name)

    for row_index, (label, values) in enumerate(dataframe.iterrows(), start=2):
        row = ET.SubElement(sheet_data, f"{{{SHEET_NS}}}row", {"r": str(row_index)})
        label_cell = ET.SubElement(
            row,
            f"{{{SHEET_NS}}}c",
            {"r": f"A{row_index}", "t": "inlineStr"},
        )
        text_node = ET.SubElement(ET.SubElement(label_cell, f"{{{SHEET_NS}}}is"), f"{{{SHEET_NS}}}t")
        text_node.text = str(label)

        for column_index, value in enumerate(values, start=2):
            if pd.isna(value):
                continue
            cell_ref = f"{excel_column_name(column_index)}{row_index}"
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                cell = ET.SubElement(row, f"{{{SHEET_NS}}}c", {"r": cell_ref})
                ET.SubElement(cell, f"{{{SHEET_NS}}}v").text = f"{float(value):.6f}"
            else:
                cell = ET.SubElement(row, f"{{{SHEET_NS}}}c", {"r": cell_ref, "t": "inlineStr"})
                text_node = ET.SubElement(ET.SubElement(cell, f"{{{SHEET_NS}}}is"), f"{{{SHEET_NS}}}t")
                text_node.text = str(value)

    return sheet_data


def load_sheet_targets(template_path):
    """读取模板文件中的 sheet 名与 xml 路径映射。"""
    ns = {"a": SHEET_NS, "r": REL_NS}
    with zipfile.ZipFile(template_path) as zf:
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        sheet_targets = {}
        for sheet in workbook.find("a:sheets", ns):
            rid = sheet.attrib[f"{{{REL_NS}}}id"]
            sheet_targets[sheet.attrib["name"]] = "xl/" + rel_map[rid]
        return sheet_targets


def _write_template_excel(output_path, template_path, replacements):
    """按模板替换指定 sheet 的数据。"""
    ensure_dir(Path(output_path).parent)
    output_path = Path(output_path)
    source_copy_path = output_path.with_name(f"{output_path.stem}_{uuid.uuid4().hex}.source.xlsx")
    temp_path = output_path.with_name(f"{output_path.stem}_{uuid.uuid4().hex}.tmp.xlsx")
    shutil.copyfile(template_path, source_copy_path)

    with zipfile.ZipFile(source_copy_path, "r") as source_zip, zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as dest_zip:
        for item in source_zip.infolist():
            content = source_zip.read(item.filename)
            if item.filename in replacements:
                root = ET.fromstring(content)
                dimension = root.find(f"{{{SHEET_NS}}}dimension")
                if dimension is not None:
                    end_col = excel_column_name(replacements[item.filename].shape[1] + 1)
                    end_row = replacements[item.filename].shape[0] + 1
                    dimension.set("ref", f"A1:{end_col}{end_row}")
                old_sheet_data = root.find(f"{{{SHEET_NS}}}sheetData")
                if old_sheet_data is not None:
                    root.remove(old_sheet_data)
                root.append(create_sheet_data(replacements[item.filename]))
                content = ET.tostring(root, encoding="utf-8", xml_declaration=True)
            dest_zip.writestr(item, content)
    source_copy_path.unlink(missing_ok=True)
    output_path.unlink(missing_ok=True)
    temp_path.replace(output_path)


def write_result1_excel(position_df, speed_df, output_path):
    """按模板写出第一问 Excel。"""
    template_path = dragon_data.REFERENCE_DIR / "result1.xlsx"
    sheet_targets = load_sheet_targets(template_path)
    replacements = {
        sheet_targets["位置"]: position_df,
        sheet_targets["速度"]: speed_df,
    }
    _write_template_excel(output_path, template_path, replacements)


def write_result2_excel(result2_df, output_path):
    """按模板写出第二问 Excel。"""
    template_path = dragon_data.REFERENCE_DIR / "result2.xlsx"
    sheet_targets = load_sheet_targets(template_path)
    replacements = {
        sheet_targets["Sheet1"]: result2_df,
    }
    _write_template_excel(output_path, template_path, replacements)


def write_result4_excel(position_df, speed_df, output_path):
    """按模板写出第四问 Excel。"""
    template_path = dragon_data.REFERENCE_DIR / "result4.xlsx"
    sheet_targets = load_sheet_targets(template_path)
    replacements = {
        sheet_targets["位置"]: position_df,
        sheet_targets["速度"]: speed_df,
    }
    _write_template_excel(output_path, template_path, replacements)


def save_question1_figures(position, speed, time_points):
    """保存第一问图像。"""
    ensure_dir(dragon_data.OUTPUT_FIGURES_DIR)
    configure_chinese_plotting()

    figure_path_1 = dragon_data.OUTPUT_FIGURES_DIR / "question1_trajectory.png"
    figure_path_2 = dragon_data.OUTPUT_FIGURES_DIR / "question1_speed.png"

    key_indices = [0, 60, 120, 180, 240, 300]
    valid_indices = [index for index in key_indices if index < len(time_points)]

    plt.figure(figsize=(8, 8))
    for time_index in valid_indices:
        plt.plot(position[:, 0, time_index], position[:, 1, time_index], label=f"{int(time_points[time_index])} s")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("问题一轨迹关键时刻示意图")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path_1, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    labels = {
        0: "Head",
        1: "Body 1",
        51: "Body 51",
        101: "Body 101",
        151: "Body 151",
        201: "Body 201",
        223: "Tail Rear",
    }
    for point_index in [0, 1, 51, 101, 151, 201, 223]:
        plt.plot(time_points, speed[point_index], label=labels[point_index])
    plt.xlabel("时间 (s)")
    plt.ylabel("速度 (m/s)")
    plt.title("问题一代表节点速度曲线")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path_2, dpi=200)
    plt.close()

    return figure_path_1, figure_path_2


def normalize_vector(vector):
    """返回单位向量。"""
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("零向量不能归一化")
    return vector / norm


def perpendicular_vector(direction):
    """返回二维法向量。"""
    direction = np.asarray(direction, dtype=float)
    return np.array([-direction[1], direction[0]], dtype=float)


def build_rectangle_from_center(center, direction, length, width):
    """由中心点和方向构造矩形。"""
    direction = normalize_vector(direction)
    normal = perpendicular_vector(direction)
    center = np.asarray(center, dtype=float)
    half_long = 0.5 * length
    half_wide = 0.5 * width
    corners = np.array(
        [
            center + half_long * direction + half_wide * normal,
            center + half_long * direction - half_wide * normal,
            center - half_long * direction - half_wide * normal,
            center - half_long * direction + half_wide * normal,
        ]
    )
    return {
        "center": center,
        "direction": direction,
        "normal": normal,
        "length": float(length),
        "width": float(width),
        "corners": corners,
    }


def build_bench_rectangle(front_point, rear_point, bench_length, bench_width, bench_index):
    """由板凳前后把手点构造矩形。"""
    front_point = np.asarray(front_point, dtype=float)
    rear_point = np.asarray(rear_point, dtype=float)
    center = 0.5 * (front_point + rear_point)
    direction = front_point - rear_point
    rectangle = build_rectangle_from_center(center, direction, bench_length, bench_width)
    rectangle["bench_index"] = bench_index
    rectangle["front_point"] = front_point
    rectangle["rear_point"] = rear_point
    return rectangle


def rectangle_projection_half_extent(rectangle, axis):
    """返回矩形在某轴上的投影半宽。"""
    axis = normalize_vector(axis)
    return (
        0.5 * rectangle["length"] * abs(np.dot(axis, rectangle["direction"]))
        + 0.5 * rectangle["width"] * abs(np.dot(axis, rectangle["normal"]))
    )


def rectangle_sat_gap(rectangle_a, rectangle_b):
    """用 SAT 返回两矩形的净间隙。"""
    axes = [
        rectangle_a["direction"],
        rectangle_a["normal"],
        rectangle_b["direction"],
        rectangle_b["normal"],
    ]
    center_delta = rectangle_a["center"] - rectangle_b["center"]
    axis_gaps = []
    for axis in axes:
        axis = normalize_vector(axis)
        projection_distance = abs(np.dot(center_delta, axis))
        gap = projection_distance - rectangle_projection_half_extent(rectangle_a, axis) - rectangle_projection_half_extent(rectangle_b, axis)
        axis_gaps.append((gap, axis))

    gap, axis = max(axis_gaps, key=lambda item: item[0])
    return {
        "gap": float(gap),
        "axis": axis,
        "axis_gaps": axis_gaps,
    }


def plot_rectangle(ax, rectangle, color, label=None, linewidth=1.5, alpha=0.2):
    """在坐标轴上绘制矩形。"""
    corners = rectangle["corners"]
    polygon = np.vstack([corners, corners[0]])
    ax.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=alpha)
    ax.plot(polygon[:, 0], polygon[:, 1], color=color, linewidth=linewidth, label=label)


def build_local_geometry_display_indices(primary_indices, secondary_indices, focus_pair):
    """为局部几何图扩展出连续显示的板凳索引。"""
    display_set = set(primary_indices) | set(secondary_indices) | set(focus_pair)
    if not display_set:
        return []
    start_index = min(display_set)
    end_index = max(display_set)
    return list(range(start_index, end_index + 1))


def build_local_geometry_style_groups(display_indices, focus_pair):
    """返回局部几何图中的普通板凳与高亮板凳分组。"""
    focus_inner, focus_outer = focus_pair
    base_indices = [index for index in display_indices if index not in (focus_inner, focus_outer)]
    return base_indices, focus_inner, focus_outer


def build_local_geometry_handle_chain(rectangles, display_indices):
    """按连续板凳链提取把手点序列。"""
    if not display_indices:
        return np.zeros((0, 2), dtype=float)
    bench_map = {rectangle["bench_index"]: rectangle for rectangle in rectangles}
    handle_points = [bench_map[display_indices[0]]["front_point"]]
    for bench_index in display_indices:
        handle_points.append(bench_map[bench_index]["rear_point"])
    return np.asarray(handle_points, dtype=float)


def build_local_spiral_curve(theta, display_indices, pitch, point_count=400):
    """根据显示范围生成局部阿基米德螺线。"""
    if not display_indices:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    start_index = min(display_indices)
    end_index = max(display_indices) + 1
    theta_curve = np.linspace(float(theta[start_index]), float(theta[end_index]), int(point_count))
    radius_curve = spiral_coefficient(pitch) * theta_curve
    x_curve, y_curve = polar_to_cartesian(radius_curve, theta_curve)
    return theta_curve, x_curve, y_curve


def save_question2_gap_curve(gap_history, output_path, safe_time=None, collision_time=None):
    """保存 G(t) 曲线图。"""
    ensure_dir(Path(output_path).parent)
    configure_chinese_plotting()
    labels = get_question2_plot_labels()
    ordered = sorted(gap_history.items())
    times = np.array([item[0] for item in ordered], dtype=float)
    gaps = np.array([item[1] for item in ordered], dtype=float)

    plt.figure(figsize=(10, 6))
    plt.plot(times, gaps, marker="o", markersize=2, linewidth=1.0)
    plt.axhline(0.0, color="red", linestyle="--", linewidth=1.0)
    y_min = float(np.min(gaps))
    y_max = float(np.max(gaps))
    y_span = y_max - y_min if y_max > y_min else 1.0

    if safe_time is not None:
        plt.axvline(safe_time, color="#2ca02c", linestyle="--", linewidth=1.2)
        plt.text(
            safe_time,
            y_min + 0.75 * y_span,
            f"{labels['safe_text']} = {safe_time:.6f} s",
            rotation=90,
            color="#2ca02c",
            va="center",
            ha="right",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#2ca02c", "alpha": 0.8},
        )
    if collision_time is not None:
        plt.axvline(collision_time, color="#d62728", linestyle="--", linewidth=1.2)
        plt.text(
            collision_time,
            y_min + 0.25 * y_span,
            f"{labels['collision_text']} = {collision_time:.6f} s",
            rotation=90,
            color="#d62728",
            va="center",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#d62728", "alpha": 0.8},
        )
    plt.xlabel(labels["gap_xlabel"])
    plt.ylabel(labels["gap_ylabel"])
    plt.title(labels["gap_title"])
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_question2_local_geometry(rectangles, focus_pair, layer_indices, output_path, theta=None):
    """保存临界时刻局部几何图。"""
    ensure_dir(Path(output_path).parent)
    configure_chinese_plotting()
    labels = get_question2_plot_labels()
    inner_indices, outer_indices = layer_indices
    display_indices = build_local_geometry_display_indices(inner_indices, outer_indices, focus_pair)
    base_indices, focus_inner, focus_outer = build_local_geometry_style_groups(display_indices, focus_pair)
    bench_map = {rectangle["bench_index"]: rectangle for rectangle in rectangles}
    handle_points = build_local_geometry_handle_chain(rectangles, display_indices)

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    if theta is not None and display_indices:
        _, x_curve, y_curve = build_local_spiral_curve(theta, display_indices, dragon_data.QUESTION1_PITCH)
        ax.plot(x_curve, y_curve, color="black", linestyle="--", linewidth=1.0, alpha=0.7)

    for bench_index in base_indices:
        rectangle = bench_map[bench_index]
        plot_rectangle(ax, rectangle, "#ff7f0e", linewidth=1.0, alpha=0.12)

    for bench_index in display_indices:
        rectangle = bench_map[bench_index]
        ax.plot(
            [rectangle["front_point"][0], rectangle["rear_point"][0]],
            [rectangle["front_point"][1], rectangle["rear_point"][1]],
            color="black",
            linewidth=1.2,
        )

    if handle_points.size > 0:
        ax.scatter(handle_points[:, 0], handle_points[:, 1], color="black", s=18, zorder=5)

    plot_rectangle(ax, bench_map[focus_inner], "#d62728", label=f"Bench {focus_inner}", linewidth=2.0, alpha=0.30)
    plot_rectangle(ax, bench_map[focus_outer], "#2ca02c", label=f"Bench {focus_outer}", linewidth=2.0, alpha=0.30)

    plt.xlabel(labels["x_label"])
    plt.ylabel(labels["y_label"])
    plt.title(labels["local_title"])
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_question2_sat_projection(rectangle_a, rectangle_b, sat_result, output_path):
    """保存 SAT 投影示意图。"""
    ensure_dir(Path(output_path).parent)
    configure_chinese_plotting()
    labels = get_question2_plot_labels()
    axis = normalize_vector(sat_result["axis"])

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))
    plot_rectangle(ax_left, rectangle_a, "#d62728", label=f"Bench {rectangle_a['bench_index']}", linewidth=2.0, alpha=0.25)
    plot_rectangle(ax_left, rectangle_b, "#2ca02c", label=f"Bench {rectangle_b['bench_index']}", linewidth=2.0, alpha=0.25)
    center = 0.5 * (rectangle_a["center"] + rectangle_b["center"])
    axis_scale = max(rectangle_a["length"], rectangle_b["length"])
    ax_left.arrow(center[0], center[1], axis[0] * axis_scale, axis[1] * axis_scale, width=0.02, color="black")
    ax_left.set_title(labels["sat_left_title"])
    ax_left.set_aspect("equal")
    ax_left.legend()

    def interval(rectangle):
        projection_center = np.dot(rectangle["center"], axis)
        half_extent = rectangle_projection_half_extent(rectangle, axis)
        return projection_center - half_extent, projection_center + half_extent

    left_a, right_a = interval(rectangle_a)
    left_b, right_b = interval(rectangle_b)
    ax_right.plot([left_a, right_a], [1, 1], color="#d62728", linewidth=6)
    ax_right.plot([left_b, right_b], [0, 0], color="#2ca02c", linewidth=6)
    ax_right.set_yticks([0, 1])
    ax_right.set_yticklabels([f"Bench {rectangle_b['bench_index']}", f"Bench {rectangle_a['bench_index']}"])
    ax_right.set_title(f"{labels['sat_right_title']}\n间隙 = {sat_result['gap']:.6f}")
    ax_right.grid(True, axis="x", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_question2_bench_geometry_schematic(output_path, bench_length=2.20, bench_width=0.30, handle_offset=0.275):
    """保存单块板凳由两把手确定四顶点的示意图。"""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    configure_chinese_plotting()

    E1 = np.array([0.0, 0.0], dtype=float)
    E2 = np.array([bench_length, 0.0], dtype=float)
    A1 = np.array([0.0, bench_width / 2.0], dtype=float)
    A2 = np.array([0.0, -bench_width / 2.0], dtype=float)
    A3 = np.array([bench_length, -bench_width / 2.0], dtype=float)
    A4 = np.array([bench_length, bench_width / 2.0], dtype=float)

    M1 = np.array([handle_offset, 0.0], dtype=float)
    M2 = np.array([bench_length - handle_offset, 0.0], dtype=float)
    center = 0.5 * (M1 + M2)
    u = normalize_vector(M2 - M1)
    n = np.array([-u[1], u[0]], dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    rectangle_x = [A1[0], A4[0], A3[0], A2[0], A1[0]]
    rectangle_y = [A1[1], A4[1], A3[1], A2[1], A1[1]]
    ax.fill(rectangle_x, rectangle_y, color="#f5d7a1", alpha=0.55, zorder=1)
    ax.plot(rectangle_x, rectangle_y, color="#8c564b", linewidth=2.0, zorder=2)

    ax.plot([E1[0], E2[0]], [E1[1], E2[1]], color="#666666", linewidth=1.4, linestyle="--", zorder=2)
    ax.plot([M1[0], M2[0]], [M1[1], M2[1]], color="black", linewidth=1.8, zorder=3)

    u_length = 0.45 * np.linalg.norm(M2 - M1)
    n_length = 0.60
    ax.arrow(
        M1[0],
        M1[1],
        u[0] * u_length,
        u[1] * u_length,
        width=0.01,
        head_width=0.08,
        head_length=0.12,
        length_includes_head=True,
        color="#d62728",
        zorder=4,
    )
    ax.arrow(
        center[0],
        center[1],
        n[0] * n_length,
        n[1] * n_length,
        width=0.01,
        head_width=0.08,
        head_length=0.12,
        length_includes_head=True,
        color="#1f77b4",
        zorder=4,
    )

    label_points = {
        "M1": (M1, (-0.10, 0.10)),
        "M2": (M2, (0.06, 0.10)),
        "E1": (E1, (-0.10, -0.16)),
        "E2": (E2, (0.06, -0.16)),
        "A1": (A1, (-0.12, 0.10)),
        "A2": (A2, (-0.12, -0.16)),
        "A3": (A3, (0.06, -0.16)),
        "A4": (A4, (0.06, 0.10)),
    }
    for label, (point, offset) in label_points.items():
        ax.scatter(point[0], point[1], color="black", s=24, zorder=5)
        ax.text(point[0] + offset[0], point[1] + offset[1], label, fontsize=11, color="black")

    ax.text(
        M1[0] + 0.55 * u_length,
        M1[1] + 0.08,
        r"$\mathbf{u}$",
        color="#d62728",
        fontsize=12,
        ha="center",
    )
    ax.text(
        center[0] + 0.10,
        center[1] + 0.55 * n_length,
        r"$\mathbf{n}$",
        color="#1f77b4",
        fontsize=12,
        ha="left",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_xlim(-0.45, bench_length + 0.45)
    ax.set_ylim(-0.65, 0.75)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _find_question4_snapshot_index(s_values, geometry):
    """选择一个同时包含盘入、调头和盘出的时刻。"""
    for time_index in range(s_values.shape[1]):
        column = s_values[:, time_index]
        has_in = bool(np.any(column < 0.0))
        has_turn = bool(np.any((column >= 0.0) & (column <= geometry["L1"] + geometry["L2"])))
        has_out = bool(np.any(column > geometry["L1"] + geometry["L2"]))
        if has_in and has_turn and has_out:
            return time_index
    return s_values.shape[1] // 2


def save_question4_turn_schematic(geometry, output_path):
    """保存第四问调头路径论文示意图。"""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    configure_chinese_plotting()

    A = np.asarray(geometry["A"], dtype=float)
    B = np.asarray(geometry["B"], dtype=float)
    C = np.asarray(geometry["D"], dtype=float)
    O1 = np.asarray(geometry["C1"], dtype=float)
    O2 = np.asarray(geometry["C2"], dtype=float)
    O = np.zeros(2, dtype=float)
    turn_radius = np.linalg.norm(A)
    b = float(geometry["b"])
    theta_a = float(geometry["theta_A"])

    def spiral_point(theta_value):
        radius = b * theta_value
        return np.array(polar_to_cartesian(radius, theta_value), dtype=float)

    def polar_angle(point, center):
        delta = np.asarray(point, dtype=float) - np.asarray(center, dtype=float)
        return np.arctan2(delta[1], delta[0])

    def oriented_angles(start_point, end_point, center, orientation, count=240):
        start_angle = polar_angle(start_point, center)
        total_angle = oriented_arc_angle(
            np.asarray(start_point, dtype=float) - np.asarray(center, dtype=float),
            np.asarray(end_point, dtype=float) - np.asarray(center, dtype=float),
            orientation,
        )
        return start_angle + orientation * np.linspace(0.0, total_angle, count)

    def shortest_arc_angles(start_angle, end_angle, count=100):
        delta = (float(end_angle) - float(start_angle) + np.pi) % (2.0 * np.pi) - np.pi
        return float(start_angle) + np.linspace(0.0, delta, count)

    inner_span = 2.4 * np.pi
    outer_turns = 2.0
    theta_in_main = np.linspace(theta_a + inner_span, theta_a, 360)
    in_main = np.array([spiral_point(theta_value) for theta_value in theta_in_main])
    theta_out_main = np.linspace(theta_a, theta_a + inner_span, 360)
    out_main = -np.array([spiral_point(theta_value) for theta_value in theta_out_main])

    theta_in_outer = np.linspace(theta_a + inner_span, theta_a + inner_span + outer_turns * 2.0 * np.pi, 320)
    in_outer = np.array([spiral_point(theta_value) for theta_value in theta_in_outer])
    theta_out_outer = np.linspace(theta_a + inner_span, theta_a + inner_span + outer_turns * 2.0 * np.pi, 320)
    out_outer = -np.array([spiral_point(theta_value) for theta_value in theta_out_outer])

    arc1_angles = oriented_angles(A, C, O1, geometry["eps1"])
    arc1_points = np.column_stack(
        [O1[0] + geometry["R1"] * np.cos(arc1_angles), O1[1] + geometry["R1"] * np.sin(arc1_angles)]
    )
    arc2_angles = oriented_angles(C, B, O2, geometry["eps2"])
    arc2_points = np.column_stack(
        [O2[0] + geometry["R2"] * np.cos(arc2_angles), O2[1] + geometry["R2"] * np.sin(arc2_angles)]
    )

    boundary_angles = np.linspace(0.0, 2.0 * np.pi, 500)
    boundary = np.column_stack([turn_radius * np.cos(boundary_angles), turn_radius * np.sin(boundary_angles)])

    fig, ax = plt.subplots(figsize=(8.2, 8.2))
    ax.plot(in_outer[:, 0], in_outer[:, 1], color="#4c72b0", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.plot(out_outer[:, 0], out_outer[:, 1], color="#4c72b0", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.plot(in_main[:, 0], in_main[:, 1], color="#4c72b0", linewidth=2.0)
    ax.plot(out_main[:, 0], out_main[:, 1], color="#4c72b0", linewidth=2.0)
    ax.plot(boundary[:, 0], boundary[:, 1], color="#dd8452", linewidth=2.0)
    ax.plot(arc1_points[:, 0], arc1_points[:, 1], color="#55a868", linewidth=2.4)
    ax.plot(arc2_points[:, 0], arc2_points[:, 1], color="#55a868", linewidth=2.4)

    radius_color_1 = "#c44e52"
    radius_color_2 = "#8172b3"
    ax.plot([A[0], O1[0]], [A[1], O1[1]], color=radius_color_1, linewidth=1.6)
    ax.plot([C[0], O1[0]], [C[1], O1[1]], color=radius_color_1, linewidth=1.6)
    ax.plot([B[0], O2[0]], [B[1], O2[1]], color=radius_color_2, linewidth=1.6)
    ax.plot([C[0], O2[0]], [C[1], O2[1]], color=radius_color_2, linewidth=1.6)

    centerline_color = "#4a4a4a"
    ax.plot(
        [A[0], O[0], B[0]],
        [A[1], O[1], B[1]],
        color=centerline_color,
        linewidth=1.4,
        linestyle="--",
        zorder=1,
    )

    alpha_radius = max(0.28, 0.11 * turn_radius)
    alpha_angles = shortest_arc_angles(polar_angle(O1, C), polar_angle(A, C), 100)
    ax.plot(
        C[0] + alpha_radius * np.cos(alpha_angles),
        C[1] + alpha_radius * np.sin(alpha_angles),
        color="black",
        linewidth=1.2,
        zorder=4,
    )
    alpha_mid = alpha_angles[len(alpha_angles) // 2]
    alpha_label_radius = alpha_radius + 0.16
    ax.text(
        C[0] + alpha_label_radius * np.cos(alpha_mid),
        C[1] + alpha_label_radius * np.sin(alpha_mid),
        r"$\alpha$",
        fontsize=12,
        color="black",
        ha="center",
        va="center",
        zorder=5,
    )

    point_style = {
        "A": (A, (0.10, -0.12)),
        "B": (B, (0.10, -0.12)),
        "C": (C, (0.10, 0.10)),
        "O1": (O1, (0.10, 0.10)),
        "O2": (O2, (0.10, -0.15)),
        "O": (O, (0.10, 0.10)),
    }
    for label, (point, offset) in point_style.items():
        ax.scatter(point[0], point[1], color="black", s=22, zorder=4)
        ax.text(point[0] + offset[0], point[1] + offset[1], label, fontsize=11, color="black")

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    all_points = np.vstack([in_outer, out_outer, boundary, arc1_points, arc2_points, A, B, C, O1, O2, O])
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)
    x_pad = 0.08 * (x_max - x_min)
    y_pad = 0.08 * (y_max - y_min)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_question4_figures(geometry, s_values, position, speed, time_points, output_dir):
    """保存第四问图像。"""
    ensure_dir(output_dir)
    configure_chinese_plotting()
    labels = dragon_data.get_question4_plot_labels()

    path_figure = Path(output_dir) / "question4_path.png"
    state_figure = Path(output_dir) / "question4_state.png"
    speed_figure = Path(output_dir) / "question4_speed.png"

    path_s = np.linspace(-100.0, geometry["L1"] + geometry["L2"] + 100.0, 1200)
    path_points = np.array([question4_path_position(s_value, geometry) for s_value in path_s])

    plt.figure(figsize=(8, 8))
    plt.plot(path_points[:, 0], path_points[:, 1], color="#1f77b4", linewidth=1.5)
    plt.scatter(
        [geometry["A"][0], geometry["D"][0], geometry["B"][0]],
        [geometry["A"][1], geometry["D"][1], geometry["B"][1]],
        color=["#d62728", "#2ca02c", "#9467bd"],
        s=30,
    )
    plt.xlabel(labels["x_label"])
    plt.ylabel(labels["y_label"])
    plt.title(labels["path_title"])
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(path_figure, dpi=200)
    plt.close()

    snapshot_index = _find_question4_snapshot_index(s_values, geometry)
    plt.figure(figsize=(8, 8))
    plt.plot(path_points[:, 0], path_points[:, 1], color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    plt.plot(position[:, 0, snapshot_index], position[:, 1, snapshot_index], color="#d62728", linewidth=1.0)
    plt.scatter(position[:, 0, snapshot_index], position[:, 1, snapshot_index], color="black", s=12)
    plt.xlabel(labels["x_label"])
    plt.ylabel(labels["y_label"])
    plt.title(f"{labels['state_title']} (t={time_points[snapshot_index]:.0f} s)")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(state_figure, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    representative_indices = dragon_data.QUESTION4_KEY_POINT_INDICES
    for point_index in representative_indices:
        plt.plot(time_points, speed[point_index], label=dragon_data.get_point_names()[point_index])
    plt.xlabel(labels["time_label"])
    plt.ylabel(labels["speed_label"])
    plt.title(labels["speed_title"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(speed_figure, dpi=200)
    plt.close()

    return path_figure, state_figure, speed_figure


def get_question5_plot_labels():
    """返回第五问图像文字配置。"""
    if hasattr(dragon_data, "get_question5_plot_labels"):
        return dragon_data.get_question5_plot_labels()
    return {
        "speed_curve_title": "第五问代表节点速度曲线",
        "envelope_title": "第五问全体把手最大速度包络",
        "state_title": "第五问最大速度发生时刻状态图",
        "scale_title": "第五问速度缩放关系示意图",
        "x_label": "横坐标 x (m)",
        "y_label": "纵坐标 y (m)",
        "time_label": "时间 (s)",
        "speed_label": "速度 (m/s)",
        "head_speed_label": "龙头速度 (m/s)",
        "scaled_speed_label": "把手速度 (m/s)",
    }


def build_question5_key_speed_table(speed, time_points, point_indices=None):
    """构造第五问关键节点最大速度统计表。"""
    speed = np.asarray(speed, dtype=float)
    time_points = np.asarray(time_points, dtype=float)
    if point_indices is None:
        point_indices = dragon_data.QUESTION5_KEY_POINT_INDICES
    point_names = dragon_data.get_point_names()

    rows = []
    index = []
    for point_index in point_indices:
        peak_index = int(np.argmax(speed[point_index]))
        rows.append([float(speed[point_index, peak_index]), float(time_points[peak_index])])
        index.append(point_names[point_index])
    return pd.DataFrame(rows, index=index, columns=["最大速度 (m/s)", "出现时刻 (s)"])


def build_question5_risk_table(point_name, point_index, risk_time, peak_speed):
    """构造第五问最大风险把手表。"""
    return pd.DataFrame(
        [[point_name, int(point_index), float(risk_time), float(peak_speed)]],
        index=["最大风险把手"],
        columns=["把手名称", "把手编号", "发生时刻 (s)", "速度峰值 (m/s)"],
    )


def build_question5_conclusion_table(max_amplification, vmax):
    """构造第五问最终结论表。"""
    return pd.DataFrame(
        {"数值": [float(max_amplification), float(vmax)]},
        index=["速度放大系数 M", "龙头最大允许速度 Vmax (m/s)"],
    )


def _worksheet_xml_for_dataframe(dataframe):
    """构造单个工作表 XML。"""
    root = ET.Element(
        f"{{{SHEET_NS}}}worksheet",
        {"xmlns:r": REL_NS},
    )
    end_col = excel_column_name(dataframe.shape[1] + 1)
    end_row = dataframe.shape[0] + 1
    ET.SubElement(root, f"{{{SHEET_NS}}}dimension", {"ref": f"A1:{end_col}{end_row}"})
    root.append(create_sheet_data(dataframe))
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def write_result5_excel(sheet_map, output_path):
    """写出第五问 Excel 工作簿。"""
    ensure_dir(Path(output_path).parent)
    output_path = Path(output_path)
    sheets = list(sheet_map.items())

    workbook_xml = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
        "<sheets>",
    ]
    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]

    for sheet_index, (sheet_name, _) in enumerate(sheets, start=1):
        workbook_xml.append(f'<sheet name="{sheet_name}" sheetId="{sheet_index}" r:id="rId{sheet_index}"/>')
        workbook_rels.append(
            f'<Relationship Id="rId{sheet_index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{sheet_index}.xml"/>'
        )
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{sheet_index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )

    workbook_xml.extend(["</sheets>", "</workbook>"])
    workbook_rels.append("</Relationships>")
    content_types.append("</Types>")

    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""
    app_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>OpenAI Codex</Application>
</Properties>
"""
    core_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:creator>OpenAI Codex</dc:creator>
  <cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy>
</cp:coreProperties>
"""

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "\n".join(content_types).encode("utf-8"))
        zf.writestr("_rels/.rels", root_rels.encode("utf-8"))
        zf.writestr("docProps/app.xml", app_xml.encode("utf-8"))
        zf.writestr("docProps/core.xml", core_xml.encode("utf-8"))
        zf.writestr("xl/workbook.xml", "\n".join(workbook_xml).encode("utf-8"))
        zf.writestr("xl/_rels/workbook.xml.rels", "\n".join(workbook_rels).encode("utf-8"))
        for sheet_index, (_, dataframe) in enumerate(sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{sheet_index}.xml", _worksheet_xml_for_dataframe(dataframe))


def _question5_state_snapshot_index(speed):
    """返回第五问最大风险时刻索引。"""
    return int(np.unravel_index(np.argmax(speed), speed.shape)[1])


def save_question5_figures(position, speed, time_points, output_dir, speed_limit, vmax, peak_time, peak_speed):
    """保存第五问图像。"""
    ensure_dir(output_dir)
    configure_chinese_plotting()
    labels = get_question5_plot_labels()
    point_names = dragon_data.get_point_names()
    representative_indices = dragon_data.QUESTION5_KEY_POINT_INDICES

    curve_path = Path(output_dir) / "question5_speed_curves.png"
    envelope_path = Path(output_dir) / "question5_speed_envelope.png"
    state_path = Path(output_dir) / "question5_max_state.png"
    scale_path = Path(output_dir) / "question5_speed_scaling.png"

    plt.figure(figsize=(10, 6))
    for point_index in representative_indices:
        plt.plot(time_points, speed[point_index], label=point_names[point_index])
    plt.xlabel(labels["time_label"])
    plt.ylabel(labels["speed_label"])
    plt.title(labels["speed_curve_title"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=200)
    plt.close()

    envelope = np.max(speed, axis=0)
    envelope_peak_index = int(np.argmin(np.abs(np.asarray(time_points, dtype=float) - float(peak_time))))
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, envelope, color="#d62728", linewidth=1.5)
    plt.axhline(float(peak_speed), color="#2ca02c", linestyle="--", linewidth=1.0)
    plt.scatter([peak_time], [peak_speed], color="black", s=24, zorder=3)
    plt.text(
        peak_time,
        peak_speed,
        f"  M={peak_speed:.6f}\n  Vmax={vmax:.6f} m/s",
        va="bottom",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.8},
    )
    plt.xlabel(labels["time_label"])
    plt.ylabel(labels["speed_label"])
    plt.title(labels["envelope_title"])
    plt.tight_layout()
    plt.savefig(envelope_path, dpi=200)
    plt.close()

    risk_time_index = envelope_peak_index
    plt.figure(figsize=(8, 8))
    plt.plot(position[:, 0, risk_time_index], position[:, 1, risk_time_index], color="#d62728", linewidth=1.0)
    plt.scatter(position[:, 0, risk_time_index], position[:, 1, risk_time_index], color="black", s=12)
    plt.xlabel(labels["x_label"])
    plt.ylabel(labels["y_label"])
    plt.title(f"{labels['state_title']} (t={peak_time:.3f} s)")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(state_path, dpi=200)
    plt.close()

    scale_factors = dragon_data.QUESTION5_SCALE_FACTORS
    head_speeds = scale_factors * speed_limit
    selected_indices = [0, representative_indices[1], int(np.unravel_index(np.argmax(speed), speed.shape)[0])]
    plt.figure(figsize=(10, 6))
    for point_index in selected_indices:
        unit_speed = speed[point_index, risk_time_index]
        plt.plot(head_speeds, head_speeds * unit_speed, label=point_names[point_index])
    plt.axvline(vmax, color="#d62728", linestyle="--", linewidth=1.0)
    plt.axhline(speed_limit, color="#2ca02c", linestyle="--", linewidth=1.0)
    plt.text(
        vmax,
        speed_limit,
        f"  Vmax={vmax:.6f} m/s",
        va="bottom",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.8},
    )
    plt.xlabel(labels["head_speed_label"])
    plt.ylabel(labels["scaled_speed_label"])
    plt.title(labels["scale_title"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(scale_path, dpi=200)
    plt.close()

    return curve_path, envelope_path, state_path, scale_path
