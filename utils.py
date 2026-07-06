import shutil
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


def spiral_coefficient(pitch):
    """根据螺距返回阿基米德螺线系数 b。"""
    return pitch / (2.0 * np.pi)


def spiral_arc_length(theta, b):
    """返回从原点到参数 theta 处的螺线弧长。"""
    theta = np.asarray(theta, dtype=float)
    return 0.5 * b * (theta * np.sqrt(1.0 + theta ** 2) + np.arcsinh(theta))


def solve_theta_from_arc_length(target_arc_length, b, upper_bound, tol=1e-12, max_iter=100):
    """由弧长反求龙头参数 theta。"""
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
    """将位置结果整理为题目表格。"""
    columns = dragon_data.get_time_labels(time_points)
    rows = []
    for point_index in range(position.shape[0]):
        rows.append(position[point_index, 0, :])
        rows.append(position[point_index, 1, :])
    data = np.vstack(rows)
    return pd.DataFrame(data, index=dragon_data.get_position_row_labels(), columns=columns)


def build_speed_dataframe(speed, time_points):
    """将速度结果整理为题目表格。"""
    columns = dragon_data.get_time_labels(time_points)
    return pd.DataFrame(speed, index=dragon_data.get_speed_row_labels(), columns=columns)


def build_summary_tables(position_df, speed_df):
    """整理论文需要的关键时刻表。"""
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

    summary_position = position_df.loc[position_rows, key_columns]
    summary_speed = speed_df.loc[speed_rows, key_columns]
    return summary_position, summary_speed


def save_dataframe_exports(dataframe, csv_path, markdown_path):
    """同时保存 csv 与 markdown 表格。"""
    dataframe.to_csv(csv_path, encoding="utf-8-sig")
    try:
        content = dataframe.to_markdown()
    except Exception:
        content = dataframe.to_string()
    markdown_path.write_text(content, encoding="utf-8")


def excel_column_name(index):
    """数字列号转 Excel 列名，1 -> A。"""
    name = []
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        name.append(chr(ord("A") + remainder))
    return "".join(reversed(name))


def create_sheet_data(dataframe):
    """将 DataFrame 转为 worksheet 的 sheetData 节点。"""
    sheet_data = ET.Element(f"{{{SHEET_NS}}}sheetData")

    header_row = ET.SubElement(sheet_data, f"{{{SHEET_NS}}}row", {"r": "1"})
    for column_index, column_name in enumerate(dataframe.columns, start=2):
        cell = ET.SubElement(
            header_row,
            f"{{{SHEET_NS}}}c",
            {"r": f"{excel_column_name(column_index)}1", "t": "inlineStr"},
        )
        is_node = ET.SubElement(cell, f"{{{SHEET_NS}}}is")
        ET.SubElement(is_node, f"{{{SHEET_NS}}}t").text = str(column_name)

    for row_index, (label, values) in enumerate(dataframe.iterrows(), start=2):
        row = ET.SubElement(sheet_data, f"{{{SHEET_NS}}}row", {"r": str(row_index)})
        label_cell = ET.SubElement(
            row,
            f"{{{SHEET_NS}}}c",
            {"r": f"A{row_index}", "t": "inlineStr"},
        )
        label_is = ET.SubElement(label_cell, f"{{{SHEET_NS}}}is")
        ET.SubElement(label_is, f"{{{SHEET_NS}}}t").text = str(label)

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


def write_result1_excel(position_df, speed_df, output_path):
    """按题目模板写出 result1.xlsx。"""
    template_path = dragon_data.REFERENCE_DIR / "result1.xlsx"
    ensure_dir(Path(output_path).parent)
    shutil.copyfile(template_path, output_path)

    sheet_targets = load_sheet_targets(template_path)
    replacements = {
        sheet_targets["位置"]: position_df,
        sheet_targets["速度"]: speed_df,
    }

    temp_path = Path(output_path).with_suffix(".tmp")
    with zipfile.ZipFile(output_path, "r") as source_zip, zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as dest_zip:
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

    temp_path.replace(output_path)


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
