#! /usr/bin/env python3
from typing import Any, Sequence, Tuple
from solid.objects import (
    circle,
    cylinder,
    import_scad,
    intersection,
    linear_extrude,
    multmatrix,
    polygon,
    union,
)
from math import ceil, cos, floor, sin, pi
from collections import deque

from euclid3 import Point2, Point3

from solid import (
    scad_render_to_file,
    translate,
    cube,
    color,
    rotate,
    scale,
    offset,
)
from solid.utils import (
    down,
    up,
)
from solid.splines import catmull_rom_points, bezier_points
from euclid3 import Point3, Point2
from subprocess import run
from svgpathtools import parse_path, Path
import json
from pathlib import Path as pathlib_Path
import numpy as np
from more_itertools import pairwise
from svgpathtools.path import Line
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from matplotlib.axes import SubplotBase
from math import sqrt
from sklearn.decomposition import PCA
import pdb
from copy import deepcopy
from deepmerge import Merger

root = pathlib_Path(__file__).parent.resolve()
line_module = import_scad(root.joinpath("line.scad"))
rod_module = import_scad(root.joinpath("rod.scad"))

config_merger = Merger(
    [(list, "override"), (dict, "merge"), (set, "union")], ["override"], ["override"]
)


def spt_char_point_to_tuple_point(p):
    # move from weird spt box to 0-1024, then from 0-1024 to -512-512
    return (np.real(p) - 512, -(np.imag(p) - 900) - 512)


def tuple_point_to_spt_point(p):
    return p[0] + 1j * p[1]


def spt_point_to_tuple_point(p):
    return (np.real(p), np.imag(p))


def normalize_medians(medians: Sequence[Sequence[int]]):
    return [spt_char_point_to_tuple_point(m[0] + 1j * m[1]) for m in medians]


def interpolate_equidistant_points(
    points: Sequence[Tuple[int, int]], parts_count: int
) -> Sequence[Tuple[int, int]]:
    points_lines = [
        Line(tuple_point_to_spt_point(p1), tuple_point_to_spt_point(p2))
        for p1, p2 in pairwise(points)
    ]
    points_path = Path(*points_lines)

    interpolated_points = [points_path.point(i) for i in np.linspace(0, 1, parts_count)]

    return [spt_point_to_tuple_point(p) for p in interpolated_points]


def calculate_stroke_length(medians: Sequence[Tuple[int, int]]):
    sum = 0
    for (m1, m2) in pairwise(medians):
        sum += sqrt((m1[0] - m2[0]) ** 2 + (m1[1] - m2[1]) ** 2)
    return sum

def smoothen_curve_special(points: Sequence[Tuple[float, float]], **kwargs):
    # initialize result points with first point
    result_ps = [points[0]]
    mag_rolling_average_count = 1
    prev_mags = deque([], mag_rolling_average_count)

    debug_plot_ax = kwargs.get('debug_plot_ax')

    for idx in range(1, len(points) - 1):
        p0 = points[idx-1]
        p1 = points[idx] # middle point
        p2 = points[idx+1]
        angle1 = ((2*pi) - np.arctan2(p0[1] - p1[1], p0[0] - p1[0])) % (2*pi)
        angle2 = ((2*pi) - np.arctan2(p2[1] - p1[1], p2[0] - p1[0])) % (2*pi)
        angle = angle2 - angle1 if angle1 <= angle2 else 2 * pi - (angle1 - angle2)
        angle_deg = (angle % (2 * pi))*180/pi
        # length needs to be taken into account to ensure algorithm always converges,
        # otherwise resulting point might be placed too far from p1
        total_dist = sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2) + sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        # empirically found: when (magnitude *= distance/125) it always converges
        multiplier = total_dist/125
        new_mag = multiplier * (1 - 1 * sqrt(1 + abs(abs(angle_deg) - 180)))

        prev_mags.append(new_mag)
        mag = np.average(list(prev_mags))

        if angle1 > pi:
            angle1 -= 2*pi
        if angle2 > pi:
            angle2 -= 2*pi
        if debug_plot_ax:
            circle1 = plt.Circle(p1, mag, edgecolor="orange", linewidth=1, fill=False)
            debug_plot_ax.add_patch(circle1)
            # debug_plot_ax.text(p1[0], p1[1], '%d' % (mag))

        # calculate the bisecting angle
        middleangle = angle1 + angle/2

        # translate points so p1 is the origin
        v1 = ((p0[0] - p1[0]), (p0[1] - p1[1]))
        v2 = ((p2[0] - p1[0]), (p2[1] - p1[1]))

        d = v1[0] * v2[1] - v2[0] * v1[1] # cross product to determine CW or CCW

        if d < 0:
            middleangle += pi
        dx = np.cos(middleangle) * mag
        dy = -np.sin(middleangle) * mag
        result_p = (p1[0] + dx, p1[1] + dy)

        if debug_plot_ax:
            circle2 = plt.Circle(result_p, 3, color="magenta")
            debug_plot_ax.add_patch(circle2)

        result_ps += [result_p]

    # add last point to results
    result_ps += [points[-1]]
    return result_ps

def generate_stroke(
    stroke_path: Path,
    part_medians: Sequence[Tuple[int, int]],
    part_z: lambda x, l: int,
    thickness: float,
    extrude_thickness: float,
    part_offset: float,
    smoothen_curve: bool,
    smoothen_curve_iterations: int,
    smoothen_surface: bool,
    smoothen_surface_amount: int,
    stroke_extra_width: float,
    parts_per_stroke_unit: int,
    **kwargs,
):
    debug_enable_plot: bool = kwargs.get("debug_enable_plot")
    debug_plot_voronoi: bool = kwargs.get("debug_plot_voronoi")
    debug_plot_stroke: bool = kwargs.get("debug_plot_stroke")
    debug_plot_medians: bool = kwargs.get("debug_plot_medians")
    debug_plot_ax: SubplotBase = kwargs.get("debug_plot_ax")
    debug_plot_zoom: int = kwargs.get("debug_plot_zoom")
    stroke_index: int = kwargs.get("stroke_index")
    t: float = kwargs.get("t")
    t_purpose: Sequence[str] = kwargs.get("t_purpose")

    ps: Sequence[Point2] = []

    for segment in stroke_path:
        segment_length = segment.length()
        sample_count = round(segment_length/5)
        # take samples from each segment
        for i in np.linspace(0, 1, sample_count, endpoint=False):
            p = spt_char_point_to_tuple_point(segment.point(i))
            ps.append(p)
    char_polygon = offset(stroke_extra_width)(polygon(ps))

    obj = union()

    org_voronoi_ps = part_medians
    if smoothen_curve:
        # TODO: if parts_per_stroke_unit is very high, it takes O(n^1.7) time, which is bad.
        # we could interpolate with an intermediate number of points and then reinterpolate at the end
        # but this doesn't really work with parts_per_stroke_unit so we first need to fix that
        interpolate_num_points = len(part_medians)
        smoothen_curve_t = t if 'smoothen_curve' in t_purpose else 1
        # higher density means more iterations needed to achieve same curvature
        # empirically found that iterations *= (density ^ 1.7) ensures similar curvature
        parts_per_stroke_unit_correction = (parts_per_stroke_unit ** 1.7) / 100
        iterations_count = ceil(smoothen_curve_t * smoothen_curve_iterations * parts_per_stroke_unit_correction)
        for i in range(iterations_count):
            is_end = i==iterations_count-1
            org_voronoi_ps = smoothen_curve_special(org_voronoi_ps, plot=(is_end and debug_enable_plot))
            # interpolate at every step to avoid crossing points after multiple iterations
            # also there are just better results when doing it after every step
            org_voronoi_ps = interpolate_equidistant_points(org_voronoi_ps, interpolate_num_points)

    # create boundaries for voronoi regions (ensure all regions within the 1024x1024 square are finite)
    voronoi_ps = [
        (-1536, -1536),
        (-1536, 1536),
        (1536, 1536),
        (1536, -1536),
        *org_voronoi_ps,
    ]
    vor = Voronoi(voronoi_ps)

    if debug_enable_plot:
        if debug_plot_voronoi:
            voronoi_plot_2d(vor, ax=debug_plot_ax)
        if debug_plot_stroke:
            ps2 = ps.copy()
            ps2.append(ps2[0])
            xs_stroke, ys_stroke = zip(*ps2)
            debug_plot_ax.plot(xs_stroke, ys_stroke, "g-")
        if debug_plot_medians:
            xs_medians, ys_medians = zip(*part_medians)
            debug_plot_ax.plot(xs_medians, ys_medians, "bo", markersize=2)
        lim = 512 * debug_plot_zoom
        # character data is y-inverted, so invert the y-axis as well
        debug_plot_ax.set_xlim([-lim, lim])
        debug_plot_ax.set_ylim([lim, -lim])
        debug_plot_ax.title.set_text(f"Stroke {stroke_index}")

    # start from 3 to skip the boundary-ensuring points
    regions = {
        idx: vor.regions[vor.point_region[idx]] for idx in range(3, len(voronoi_ps))
    }
    # part_height = (height_per_stroke - thickness)/parts_count
    # part_height = parts_count/50
    regions = {
        k: region
        for (k, region) in regions.items()
        if not (-1 in region or len(region) == 0)
    }

    smoothen_surface_amount = smoothen_surface_amount if smoothen_surface else 0
    smoothen_surface_t = t if 'smoothen_surface' in t_purpose else 1
    moving_average_num_parts = 1 + ceil(smoothen_surface_amount * smoothen_surface_t)
    mat_data: deque[Tuple[np.ndarray, float]] = deque([], moving_average_num_parts)
    for (region_idx, (voronoi_idx, region)) in enumerate(regions.items()):
        # if (region_idx % 2 == 0):
        #     continue

        z = part_z(region_idx, len(regions))
        z_next = part_z(region_idx + 1, len(regions))
        shear_t = t if 'shear' in t_purpose else 1
        z_next = (z_next * shear_t) + (z * (1 - shear_t))

        delta_z = z_next - z

        # print('r_i: {}, v_i: {}, z: {}, rs: {}'.format(region_idx, voronoi_idx, z, len(regions)))
        # keep_angle = i == len(voronoi_ps) - 1
        middle_p = Point2(voronoi_ps[voronoi_idx][0], voronoi_ps[voronoi_idx][1])
        if voronoi_idx >= len(voronoi_ps) - 1:
            voronoi_idx = voronoi_idx - 1
            # i2 = len(org_voronoi_ps) - 2
        ps = [vor.vertices[idx] for idx in region]
        # offset polygons with 1 unit to ensure overlap
        offset_polygon = offset(part_offset)(polygon(ps))
        part_obj = up(-thickness / 2)(
            linear_extrude(extrude_thickness)(intersection()(offset_polygon, char_polygon))
        )
        p_src = Point3(
            voronoi_ps[voronoi_idx][0], voronoi_ps[voronoi_idx][1], -delta_z / 2
        )
        p_dst = Point3(
            voronoi_ps[voronoi_idx + 1][0],
            voronoi_ps[voronoi_idx + 1][1],
            delta_z / 2,
        )

        translate_mat = np.matrix(
            (
                (1, 0, 0, -middle_p.x),
                (0, 1, 0, -middle_p.y),
                (0, 0, 1, 0),
                (0, 0, 0, 1),
            )
        ).reshape((4, 4))

        angle_z = -np.arctan2(p_dst.y - p_src.y, p_dst.x - p_src.x)
        rot_mat = np.matrix(
            (
                (cos(angle_z), -sin(angle_z), 0, 0),
                (sin(angle_z), cos(angle_z), 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1),
            )
        ).reshape((4, 4))

        dist_xy: float = sqrt((p_dst.x - p_src.x) ** 2 + (p_dst.y - p_src.y) ** 2)
        tangent_xy: float = (p_dst.z - p_src.z) / dist_xy

        mat_data.append((rot_mat, tangent_xy))
        mat_data_list = list(mat_data)
        len_mat_data = len(mat_data_list)

        mat = np.identity(4)
        mat = np.matmul(translate_mat, mat)

        for (rot_mat, saved_tangent_xy) in mat_data_list:
            tangent_xy = saved_tangent_xy/len_mat_data
            shear_mat = np.matrix(
                ((1, 0, 0, 0), (0, 1, 0, 0), (tangent_xy, 0, 1, 0), (0, 0, 0, 1))
            ).reshape((4, 4))
            mat = np.matmul(rot_mat, mat)
            mat = np.matmul(shear_mat, mat)
            mat = np.matmul(np.linalg.inv(rot_mat), mat)

        mat = np.matmul(np.linalg.inv(translate_mat), mat)

        prog = region_idx / len(regions)
        col = (prog, 1 - prog / 2, 1 - prog)
        slope_z_t = t if 'slope_z' in t_purpose else 1
        obj += up(z * slope_z_t)(color(col)(multmatrix(np.asarray(mat))(part_obj)))

    return obj


def flat(arr: Sequence[Sequence[Any]]):
    def ensure_iterable(thing):
       return thing if hasattr(thing, '__iter__') else [thing]

    return [item for sublist in ensure_iterable(arr) for item in ensure_iterable(sublist)]


def smoothen_points_curve(points: Sequence[Tuple[float, float]]):
    points_3d = catmull_rom_points(points)
    points = [(p.x, p.y) for p in points_3d]
    return points


class Config:
    def __init__(self, config: dict):
        self.character: str = config["character"]
        self.parts: Sequence[str] = config["parts"]
        self.thickness: float = config["general_options"]["thickness"]
        self.stretch: float = config["general_options"]["stretch"]
        self.parts_per_stroke_unit: float = config["general_options"][
            "parts_per_stroke_unit"
        ]
        self.smoothen_curve: bool = config["general_options"]["smoothen_curve"]
        self.smoothen_curve_iterations: int = config["general_options"][
            "smoothen_curve_iterations"
        ]
        self.smoothen_surface: bool = config["general_options"]["smoothen_surface"]
        self.smoothen_surface_amount: int = config["general_options"][
            "smoothen_surface_amount"
        ]
        self.part_offset: float = config["general_options"]["part_offset"]
        self.stroke_extra_width: float = config["general_options"]["stroke_extra_width"]
        self.flat_mode: bool = config["flat_mode"]
        self.flat_mode_spacing: float = config["flat_mode_options"]["spacing"]
        self.distance_between_strokes: float = config["general_options"][
            "distance_between_strokes"
        ]
        self.enable_connectors: bool = config["enable_connectors"]
        self.connector_end_distance: float = config["connector_options"]["end_distance"]
        self.force_horizontal_connectors: bool = config["connector_options"][
            "force_horizontal"
        ]
        self.connector_thickness: float = config["connector_options"]["thickness"]
        self.connector_n_segments: int = config["connector_options"]["n_segments"]
        self.untilted_mode: bool = config["untilted_mode"]
        self.enable_untilted_axis: bool = config["untilted_options"]["debug_axis"]
        self.centering_method: str = config["general_options"]["centering_method"]
        self.to_bottom_mode: bool = config["to_bottom_mode"]
        self.plate_overlap: float = config["plate_options"]["overlap"]
        self.enable_pillars: bool = config["enable_pillars"]
        self.plate_height: float = config["plate_options"]["height"]
        self.enable_plate: bool = config["enable_plate"]
        self.pillar_thickness: float = config["pillar_options"]["thickness"]
        self.pillar_insert_margin: float = config["pillar_options"]["insert_margin"]
        self.pillar_insert_n_segments: float = config["pillar_options"][
            "insert_n_segments"
        ]
        self.pillar_insert_angle: float = config["pillar_options"]["insert_angle"]
        self.pillar_insert_multiplier: float = config["pillar_options"][
            "insert_multiplier"
        ]
        self.pillar_end_distance: float = config["pillar_options"][
            "pillar_end_distance"
        ]
        self.scale: float = config["scale"]
        self.debug_enable_plot: bool = config["debug_options"]["enable_plot"]
        self.debug_show_plot: bool = config["debug_options"]["show_plot"]
        self.debug_plot_window_zoom: bool = config["debug_options"]["plot_window_zoom"]
        self.debug_plot_medians: bool = config["debug_options"]["plot_medians"]
        self.debug_plot_stroke: bool = config["debug_options"]["plot_stroke"]
        self.debug_plot_voronoi: bool = config["debug_options"]["plot_voronoi"]
        self.debug_plot_orig_medians: bool = config["debug_options"][
            "plot_orig_medians"
        ]
        self.debug_plot_zoom: bool = config["debug_options"]["plot_zoom"]
        self.untilted_mode_bottom_margin: float = config["untilted_options"][
            "bottom_margin"
        ]
        self.t: float = config.get("t", 1)
        self.t_purpose: float = config.get("t_purpose")

def generate(config_dict: dict):
    root_config = Config(config_dict)

    graphics_file = open(root.joinpath("../res/graphics.txt"), "r")
    character_data = {}

    for line in graphics_file.readlines():
        c = line.strip()
        if len(c) == 0:
            continue
        character_data = json.loads(line)
        if character_data["character"] == root_config.character:
            break

    number_of_strokes = len(character_data["medians"])
    stroke_configs = [
        (
            Config(
                config_merger.merge(
                    deepcopy(config_dict), config_dict["per_stroke_options"][i]
                )
            )
            if i in config_dict["per_stroke_options"]
            else root_config
        )
        for i in range(number_of_strokes)
    ]

    strokes = cube(0)
    plate = cube(0)
    pillars = cube(0)
    pillars_cutouts = cube(0)
    connectors = cube(0)
    debug = cube(0)

    orig_stroke_medians = [
        normalize_medians(medians) for medians in character_data["medians"]
    ]
    stroke_medians = orig_stroke_medians

    stroke_lengths = [calculate_stroke_length(medians) for medians in stroke_medians]

    parts_per_stroke_unit_t = root_config.t if 'parts_per_stroke_unit' in root_config.t_purpose else 1
    root_config.parts_per_stroke_unit = ceil(root_config.parts_per_stroke_unit * parts_per_stroke_unit_t)

    stroke_part_counts = [
        ceil(root_config.parts_per_stroke_unit * stroke_length / (0.5 * 1024))
        for stroke_length in stroke_lengths
    ]

    stroke_medians = [
        interpolate_equidistant_points(
            stroke_medians[i],
            stroke_part_counts[i],
        )
        for i in range(len(stroke_medians))
    ]

    # print(stroke_lengths, stroke_part_counts)
    stroke_paths_medians_lengths_counts = [
        (
            parse_path(character_data["strokes"][i]),
            stroke_medians[i],
            stroke_lengths[i],
            stroke_part_counts[i],
        )
        for i in range(len(character_data["strokes"]))
    ]

    # pdb.set_trace()
    # height_multiplier = height_per_stroke + distance_between_strokes - thickness
    medians_3d: Sequence[Sequence[Point3]] = []
    avg_part_stretch = root_config.stretch / root_config.parts_per_stroke_unit
    stroke_z_t = root_config.t if 'stroke_z' in root_config.t_purpose else 1
    stroke_zs = np.cumsum(
        [0]
        + [
            (
                i * root_config.flat_mode_spacing
                if root_config.flat_mode
                else parts_count * avg_part_stretch
                + root_config.distance_between_strokes
            ) * stroke_z_t
            for (
                i,
                (
                    paths,
                    medians,
                    lengths,
                    parts_count,
                ),
            ) in enumerate(stroke_paths_medians_lengths_counts)
        ]
    )

    plot_axes: list[SubplotBase] = None
    if root_config.debug_enable_plot:
        n_strokes = len(stroke_paths_medians_lengths_counts)
        nrows = 2 if n_strokes <= 6 else 3
        ncols = ceil(n_strokes / nrows)
        # nrows = 1
        # ncols = 1
        plt.rcParams["font.family"] = "Noto Sans SC"
        fig, axes_grid = plt.subplots(
            nrows=nrows, ncols=ncols, subplot_kw={"aspect": "equal"}
        )
        fig.set_size_inches(root_config.debug_plot_window_zoom * 750/80, root_config.debug_plot_window_zoom * 500/80)
        fig.canvas.manager.set_window_title(
            f"3D Hanzi Generator - Plot {root_config.character}"
        )
        plt.suptitle(f"Character: {root_config.character}").set_size(
            20
        )  # , fontproperties=font_properties)
        plot_axes = flat(axes_grid)

    # plot_axes = [fig.add_subplot() for i in enumerate(stroke_paths_medians_lengths_counts)]

    for i, (stroke_path, stroke_medians, _, parts_count) in enumerate(
        stroke_paths_medians_lengths_counts
    ):
        stroke_config = stroke_configs[i]
        part_z_fn = lambda i, l: i * avg_part_stretch
        # plot_ax = plot_axes[0]
        # debug_enable_plot = i == 1
        plot_ax = plot_axes[i] if stroke_config.debug_enable_plot else None
        extrude_thickness = (
            5000 if root_config.to_bottom_mode else stroke_config.thickness
        )
        stroke_obj = generate_stroke(
            stroke_path,
            stroke_medians,
            part_z_fn,
            stroke_config.thickness,
            extrude_thickness,
            stroke_config.part_offset,
            stroke_config.smoothen_curve,
            stroke_config.smoothen_curve_iterations,
            stroke_config.smoothen_surface,
            stroke_config.smoothen_surface_amount,
            stroke_config.stroke_extra_width,
            stroke_config.parts_per_stroke_unit,
            debug_plot_ax=plot_ax,
            debug_enable_plot=stroke_config.debug_enable_plot,
            debug_plot_voronoi=stroke_config.debug_plot_voronoi,
            debug_plot_stroke=stroke_config.debug_plot_stroke,
            debug_plot_medians=stroke_config.debug_plot_medians,
            debug_plot_zoom=stroke_config.debug_plot_zoom,
            stroke_index=i,
            t=stroke_config.t,
            t_purpose=stroke_config.t_purpose
        )
        if stroke_config.debug_plot_orig_medians:
            xs, ys = zip(*orig_stroke_medians[i])
            plot_ax.plot(xs, ys, "ro", markersize=2)
        stroke_z = stroke_zs[i] + stroke_config.thickness / 2
        medians_3d.append(
            list(
                map(
                    lambda i: Point3(
                        stroke_medians[i][0],
                        stroke_medians[i][1],
                        part_z_fn(i, parts_count) + stroke_z,
                    ),
                    range(parts_count),
                )
            )
        )
        strokes += up(stroke_z)(stroke_obj)

    if root_config.debug_enable_plot:
        plt.tight_layout()
        if root_config.debug_show_plot:
            plt.show()

    if root_config.enable_connectors:
        for i, (medians1, medians2) in enumerate(pairwise(medians_3d)):
            p1_inset = floor(root_config.connector_end_distance * len(medians1))
            p2_inset = floor(root_config.connector_end_distance * len(medians2))
            p1 = medians1[-(p1_inset + 1)]
            p2 = medians2[p2_inset]
            if root_config.force_horizontal_connectors:
                avg_z = (p1.z + p2.z) / 2
                p1 = p1.copy()
                p1.z = avg_z
                p2 = p2.copy()
                p2.z = avg_z
            connectors += line_module.line(
                p1,
                p2,
                root_config.connector_thickness,
                segments=root_config.connector_n_segments,
            )

    if root_config.untilted_mode:
        arr = np.array(flat(medians_3d))
        pca = PCA()
        pca.fit(arr)
        eigenvectors = pca.components_
        # make sure z doesnt have an effect on x and y
        eigenvectors[0][2] = 0
        eigenvectors[1][2] = 0

        # make sure all eigenvectors are in the same direction as the current axis
        for i in range(3):
            # e.g. for Z, if 0, 0, 1 would map to *, *, < 0, then invert it
            if eigenvectors[i][i] < 0:
                print(f"inverting {i} eigenvectors")
                eigenvectors[i] = -eigenvectors[i]

        # make sure there's no rotation around z
        eigenvectors[0][0] = 1
        eigenvectors[0][1] = 0
        eigenvectors[1][0] = 0
        eigenvectors[1][1] = 1

        mat = [(*values, 0) for values in eigenvectors]

        if root_config.enable_untilted_axis:
            for i, eigenvector in enumerate(eigenvectors):
                c = ("red", "green", "blue")[i]
                debug += color(c)(
                    line_module.line((0, 0, 0), Point3(*eigenvector) * 200, 20)
                )
            for i, eigenvector in enumerate(eigenvectors):
                c = ("pink", "lightgreen", "lightblue")[i]
                debug += multmatrix(mat)(
                    color(c)(
                        line_module.line((0, 0, 0), Point3(*eigenvector) * 200, 20)
                    )
                )

        strokes = multmatrix(mat)(strokes)

        medians_3d: Sequence[Sequence[Point3]] = [
            [Point3(*np.matmul(eigenvectors, np.array(median))) for median in medians]
            for medians in medians_3d
        ]

    medians_max_z = max(*[p[2] for p in flat(medians_3d)])
    if root_config.untilted_mode:
        medians_max_z = medians_max_z + root_config.untilted_mode_bottom_margin
    bottom = medians_max_z + root_config.thickness / 2

    strokes = up(-bottom)(strokes)
    connectors = up(-bottom)(connectors)

    medians_3d: Sequence[Sequence[Point3]] = [
        [median + Point3(0, 0, -bottom) for median in medians] for medians in medians_3d
    ]

    if root_config.centering_method == "average_medians":
        center = Point3(*tuple(map(np.mean, zip(*flat(medians_3d)))))
        strokes = translate((-center[0], -center[1], 0))(strokes)

    if root_config.to_bottom_mode:
        cube_height = 3000
        strokes = intersection()(
            strokes,
            down(cube_height / 2 - 1)(cube((2048, 2048, cube_height), center=True)),
        )

    plate_z = -root_config.plate_overlap
    for i, medians in enumerate(medians_3d):
        stroke_config = stroke_configs[i]
        if not stroke_config.enable_pillars:
            continue
        medians_index = floor(stroke_config.pillar_end_distance * len(medians))
        p = medians[medians_index]
        p_prev = medians[max(0, medians_index - 1)]
        p_next = medians[min(len(medians) - 1, medians_index + 1)]
        direction = p_next - p_prev
        # pdb.set_trace()
        # debug += color('red')(line_module.line(p, p + rico * 10, 20))
        angle = np.arctan2(direction.y, direction.x) * 180 / pi
        insert_height = stroke_config.thickness * 3
        insert_insertion = stroke_config.thickness * 0.4
        pillar_insert_end_p = p + (
            0,
            0,
            (stroke_config.thickness / 2) - insert_insertion,
        )
        pillar_cone_start_p = pillar_insert_end_p + (0, 0, insert_insertion + 5)
        pillar = rod_module.line(
            pillar_cone_start_p,
            (p.x, p.y, plate_z + root_config.plate_height / 2),
            stroke_config.pillar_thickness,
        )
        pillar += rod_module.cone(
            pillar_cone_start_p,
            pillar_insert_end_p,
            stroke_config.pillar_thickness,
            stroke_config.pillar_thickness / 2,
        )
        insert_segment_count = stroke_config.pillar_insert_n_segments
        insert_angle = (
            angle + stroke_config.pillar_insert_angle - 30
            if insert_segment_count == 6
            else angle + stroke_config.pillar_insert_angle - 45
        )

        def extrude_insert(surface):
            return translate(pillar_insert_end_p)(
                rotate((0, 0, insert_angle))(linear_extrude(insert_height)(surface))
            )

        insert = extrude_insert(
            circle(
                stroke_config.pillar_thickness * stroke_config.pillar_insert_multiplier,
                segments=insert_segment_count,
            )
        )
        insert_cutout = extrude_insert(
            circle(
                stroke_config.pillar_thickness * stroke_config.pillar_insert_multiplier
                + stroke_config.pillar_insert_margin,
                segments=insert_segment_count,
            )
        )
        insert = intersection()(insert, strokes)
        pillars += pillar + insert
        pillars_cutouts += pillar + insert_cutout

    strokes -= pillars_cutouts

    if root_config.enable_plate:
        r1 = 512
        r2 = 80
        plate += up(plate_z + root_config.plate_height / 2)(
            cylinder(r=r1 + r2, h=root_config.plate_height, center=True, segments=100)
        )

    obj = cube(0)
    for part in root_config.parts:
        part_obj = {
            "strokes": strokes,
            "plate": plate,
            "pillars": pillars,
            "connectors": connectors,
            "debug": debug,
        }[part]
        obj += part_obj

    return scale(root_config.scale * 60 / 1024)(rotate((-180, 0, 0))(obj))


def find_openscad():
    # from https://github.com/TheJKM/OpenSCAD-Parallel-Build/blob/master/openscad-parallel-build.py
    import platform
    import os

    p = ""
    # Check if we find OpenSCAD
    plat = platform.system()
    if plat == "Darwin":
        p = "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD"
    elif plat == "Linux":
        p = "/usr/bin/openscad"
    elif plat == "Windows":
        p = ""
    while not os.path.exists(p):
        print("Unable to find OpenSCAD. You can manually provide a path.")
        p = input("OpenSCAD executable: ")
        if os.path.exists(p):
            break

    return p


if __name__ == "__main__":
    from argparse import ArgumentParser
    import yaml
    from time import time

    parser = ArgumentParser("character_generator")
    parser.add_argument("--character", help="Hanzi", type=str)
    parser.add_argument("--out-dir", help="Out dir", type=str)
    parser.add_argument("--out-scad", help="Out .scad file", type=str)
    parser.add_argument("--out-stl", help="Out .stl file", type=str)
    parser.add_argument("--out-debug-plot", help="Out matplotlib (.png or .pdf) file", type=str)
    parser.add_argument("--stl", help="Stl or not", type=bool)
    parser.add_argument(
        "--parts",
        help="Comma-separated parts (strokes, plate, pillars, connectors, debug)",
        type=str,
    )
    parser.add_argument("--settings", help="Settings preset (.yaml file)", type=str)
    parser.add_argument("--scale", help="Scale the model", type=float)
    parser.add_argument("--t", help="For animations", type=float)
    parser.add_argument("--t-purpose", help="For animations (stroke_z,slope_z,shear,smoothen_surface,smoothen_curve)", type=str)
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir is not None else "."

    config: Any = {}
    if args.settings is not None:
        with open(args.settings, "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

    base_config: Any = None
    base_config_path = root.joinpath("base_settings.yml")
    with open(base_config_path, "r") as file:
        base_config = yaml.load(file, Loader=yaml.FullLoader)

    config = config_merger.merge(base_config, config)

    if args.parts is not None:
        config["parts"] = args.parts.split(",")

    if args.character is not None:
        config["character"] = args.character

    if args.scale is not None:
        config["scale"] = args.scale

    config["t"] = args.t if args.t != None else 1
    config["t_purpose"] = args.t_purpose.split(",") if args.t_purpose != None else ()

    header = "$fn = 40;"
    base_filename_parts = (
        config["character"],
        str(round(time())),
        "-".join(config["parts"]),
    )
    base_filename = "-".join(base_filename_parts)
    scad_filepath = (
        args.out_scad
        if args.out_scad is not None
        else out_dir + "/" + base_filename + ".scad"
    )
    stl_filepath = (
        args.out_stl
        if args.out_stl is not None
        else out_dir + "/" + base_filename + ".stl"
    )
    debug_plot_filepath = args.out_debug_plot

    if debug_plot_filepath is not None:
        config["debug_options"]["enable_plot"] = True
        config["debug_options"]["show_plot"] = False

    obj = generate(config)

    file_out = scad_render_to_file(
        obj, filepath=scad_filepath, file_header=header, include_orig_code=False
    )
    print(f"SCAD file written to: \n{file_out}")

    if plt is not None and debug_plot_filepath is not None:
        plt.ioff()
        plt.savefig(debug_plot_filepath, dpi=100)
        print(f"Debug plot file written to: \n{debug_plot_filepath}")

    if args.stl:
        print("Generating stl (this might take a while)")
        run([find_openscad(), "-o", stl_filepath, scad_filepath])
