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
            cell = ET.SubElement(row, f"{{{SHEET_NS}}}c", {"r": f"{excel_column_name(column_index)}{row_index}"})
            ET.SubElement(cell, f"{{{SHEET_NS}}}v").text = f"{float(value):.6f}"

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


def save_question1_figures(position, speed, time_points):
    """保存第一问图像。"""
    ensure_dir(dragon_data.OUTPUT_FIGURES_DIR)

    figure_path_1 = dragon_data.OUTPUT_FIGURES_DIR / "question1_trajectory.png"
    figure_path_2 = dragon_data.OUTPUT_FIGURES_DIR / "question1_speed.png"

    key_indices = [0, 60, 120, 180, 240, 300]
    valid_indices = [index for index in key_indices if index < len(time_points)]

    plt.figure(figsize=(8, 8))
    for time_index in valid_indices:
        plt.plot(position[:, 0, time_index], position[:, 1, time_index], label=f"{int(time_points[time_index])} s")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Question 1 Trajectory Snapshots")
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
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("Question 1 Representative Speeds")
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
