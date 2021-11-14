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

from euclid3 import Point2, Point3

from solid import (
    scad_render_to_file,
    translate,
    cube,
    color,
    rotate,
    scale,
)
from solid.utils import (
    down,
    up,
)
from solid.splines import catmull_rom_points
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

root = pathlib_Path(__file__).parent.resolve()
line_module = import_scad(root.joinpath("line.scad"))
rod_module = import_scad(root.joinpath("rod.scad"))


def spt_char_point_to_tuple_point(p):
    # move from weird spt box to 0-1024, then from 0-1024 to -512-512
    return (np.real(p) - 512, -(np.imag(p) - 900) - 512)


def tuple_point_to_spt_point(p):
    return p[0] + 1j * p[1]


def spt_point_to_tuple_point(p):
    return (np.real(p), np.imag(p))


def normalize_medians(medians: Sequence[Sequence[int]]):
    return [spt_char_point_to_tuple_point(m[0] + 1j * m[1]) for m in medians]


def interpolate_equidistant_medians(
    medians: Sequence[Tuple[int, int]], parts_count: int
) -> Sequence[Tuple[int, int]]:
    medians_lines = [
        Line(tuple_point_to_spt_point(p1), tuple_point_to_spt_point(p2))
        for p1, p2 in pairwise(medians)
    ]
    medians_path = Path(*medians_lines)

    median_ps = [medians_path.point(i) for i in np.linspace(0, 1, parts_count)]

    return [spt_point_to_tuple_point(p) for p in median_ps]


def calculate_stroke_length(medians: Sequence[Tuple[int, int]]):
    sum = 0
    for (m1, m2) in pairwise(medians):
        sum += sqrt((m1[0] - m2[0]) ** 2 + (m1[1] - m2[1]) ** 2)
    return sum


def generate_stroke(
    stroke_path: Path,
    part_medians: Sequence[Tuple[int, int]],
    part_z: lambda x, l: int,
    thickness: float,
    extrude_thickness: float,
    debug_voronoi: bool,
    plot_ax: SubplotBase,
):
    ps: Sequence[Point2] = []

    for segment in stroke_path:
        for i in [0, 0.25, 0.5, 0.75]:
            p = spt_char_point_to_tuple_point(segment.point(i))
            ps.append(p)
    poly = polygon(ps)

    obj = union()
    char_obj = color([0, 0, 1])(
        up(-thickness / 2)(linear_extrude(extrude_thickness)(poly))
    )

    org_voronoi_ps = part_medians
    # create boundaries for voronoi regions (ensure all regions within the 1024x1024 square are finite)
    voronoi_ps = [
        (-1536, -1536),
        (-1536, 1536),
        (1536, 1536),
        (1536, -1536),
        *org_voronoi_ps,
    ]
    vor = Voronoi(voronoi_ps)

    if debug_voronoi:
        voronoi_plot_2d(vor, ax=plot_ax)
        ps2 = ps.copy()
        ps2.append(ps2[0])
        xs_stroke, ys_stroke = zip(*ps2)
        plot_ax.plot(xs_stroke, ys_stroke, "g-")
        xs_medians, ys_medians = zip(*part_medians)
        plot_ax.plot(xs_medians, ys_medians, "b-")
        plot_ax.set_xlim([-512, 512])
        plot_ax.set_ylim([-512, 512])

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
    for (region_idx, (voronoi_idx, region)) in enumerate(regions.items()):
        # if (region_idx % 2 == 0):
        #     continue

        z = part_z(region_idx, len(regions))
        z_next = part_z(region_idx + 1, len(regions))

        # print('r_i: {}, v_i: {}, z: {}, rs: {}'.format(region_idx, voronoi_idx, z, len(regions)))
        # keep_angle = i == len(voronoi_ps) - 1
        middle_p = Point2(voronoi_ps[voronoi_idx][0], voronoi_ps[voronoi_idx][1])
        if voronoi_idx >= len(voronoi_ps) - 1:
            voronoi_idx = voronoi_idx - 1
            # i2 = len(org_voronoi_ps) - 2
        ps = [vor.vertices[idx] for idx in region]
        vor_obj = up(-thickness / 2)(linear_extrude(extrude_thickness)(polygon(ps)))
        p_src = Point3(
            voronoi_ps[voronoi_idx][0], voronoi_ps[voronoi_idx][1], -(z_next - z) / 2
        )
        p_dst = Point3(
            voronoi_ps[voronoi_idx + 1][0],
            voronoi_ps[voronoi_idx + 1][1],
            (z_next - z) / 2,
        )
        dist = sqrt((p_src.x - p_dst.x) ** 2 + (p_src.y - p_dst.y) ** 2)
        # print(dist)
        mat = np.matrix(
            ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
        ).reshape((4, 4))
        angle_z = np.arctan2(p_src.x - p_dst.x, p_src.y - p_dst.y) + pi
        # print(angle_z)
        tangens_xy = (p_dst.z - p_src.z) / dist
        # print(np.arctan2(height, dist))
        rot_mat = np.matrix(
            (
                (cos(angle_z), -sin(angle_z), 0, 0),
                (sin(angle_z), cos(angle_z), 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1),
            )
        ).reshape((4, 4))
        shear_mat = np.matrix(
            ((1, 0, 0, 0), (0, 1, 0, 0), (0, tangens_xy, 1, 0), (0, 0, 0, 1))
        ).reshape((4, 4))
        mat = np.matmul(rot_mat, mat)
        mat = np.matmul(shear_mat, mat)
        mat = np.matmul(np.linalg.inv(rot_mat), mat)
        mmat = tuple(map(tuple, np.array(mat)))

        def tf(x):
            return translate(middle_p)(multmatrix(mmat)(translate(-middle_p)(x)))

        # if i > 4:
        #     continue
        # pdb.set_trace()
        # if i == 2:
        #     obj += color([1, 0, 0])(multmatrix(mmat)(cube(100, center=True)))
        # l_obj = line_module.line(p_src, p_dst, 4)
        # obj += color([1, 0, 0])(multmatrix(line_mat)(translate(-middle_p)(l_obj)) + l_obj)
        part_obj = intersection()(char_obj, vor_obj)
        prog = region_idx / len(regions)
        col = (prog, 1 - prog / 2, 1 - prog)
        obj += up(z)(color(col)(tf(part_obj)))
    # lines = [
    #     vor.vertices[line] for line in vor.ridge_vertices if -1 not in line
    # ]
    #
    # for line in lines:
    #     obj += color([1, 0, 0])(up(30)(line_module.line(line[0], line[1], 4)))
    # plt.figure(fig)
    # vor.regions
    # for i in :
    #     spt_p =
    #     spt_tg =
    #     r = (0, 0, np.angle(spt_tg) * 180/np.pi)
    #     print(r)
    #     # pdb.set_trace()
    #     obj += translate(spt_point_to_tuple_point(spt_p))(up(-10)(color([1, 0, 0])(rotate(r)(cube(5)))))

    # s = [Point2(-1, -1), Point2(-1, 1), Point2(1, 1), Point2(1, -1)]
    # # s = square(2)
    # ps = [Point3(0, 0, 0), Point3(5, 5, 5), Point3(5, 10, 5)]
    # # control_ps = []
    # # for p in ps:
    # #   control_ps.append(p)
    # spline_ps = catmull_rom_points(ps)
    # obj = sphere(0)
    # # obj = extrude_along_path(shape_pts=s, path_pts=spline_ps)
    # for p in spline_ps:
    #   obj += translate((p.x, p.y, p.z))(sphere(0.2))

    return obj


def flat(arr: Sequence[Sequence[Any]]):
    return [item for sublist in arr for item in sublist]


def smoothen_curve(points: Sequence[Tuple[float, float]]):
    points_3d = catmull_rom_points(points)
    points = [(p.x, p.y) for p in points_3d]
    return points


def generate(config: dict):
    character: str = config["character"]
    parts: Sequence[str] = config["parts"]
    thickness: float = config["general_options"]["thickness"]
    stretch: float = config["general_options"]["stretch"]
    parts_per_stroke_unit: float = config["general_options"]["parts_per_stroke_unit"]
    config_smoothen_curve: bool = config["general_options"]["smoothen_curve"]
    smoothen_curve_smoothness: float = config["general_options"]["smoothen_curve_smoothness"]
    flat_mode: bool = config["flat_mode"]
    flat_mode_spacing: float = config["flat_mode_options"]["spacing"]
    distance_between_strokes: float = config["general_options"][
        "distance_between_strokes"
    ]
    enable_connectors: bool = config["enable_connectors"]
    connector_end_distance: float = config["connector_options"]["end_distance"]
    force_horizontal_connectors: bool = config["connector_options"]["force_horizontal"]
    connector_thickness: float = config["connector_options"]["thickness"]
    connector_n_segments: int = config["connector_options"]["n_segments"]
    untilted_mode: bool = config["untilted_mode"]
    enable_untilted_axis: bool = config["untilted_options"]["debug_axis"]
    centering_method: str = config["general_options"]["centering_method"]
    to_bottom_mode: bool = config["to_bottom_mode"]
    plate_overlap: float = config["plate_options"]["overlap"]
    enable_pillars: bool = config["enable_pillars"]
    plate_height: float = config["plate_options"]["height"]
    enable_plate: bool = config["enable_plate"]
    pillar_thickness: float = config["pillar_options"]["thickness"]
    pillar_insert_margin: float = config["pillar_options"]["insert_margin"]
    pillar_insert_n_segments: float = config["pillar_options"]["insert_n_segments"]
    config_scale: float = config["scale"]
    debug_voronoi: bool = config["debug_options"]["plot_voronoi"]
    untilted_mode_bottom_margin: float = config["untilted_options"]["bottom_margin"]

    extrude_thickness = 5000 if config["to_bottom_mode"] else thickness

    graphics_file = open(root.joinpath("../res/graphics.txt"), "r")
    character_data = {}

    for line in graphics_file.readlines():
        c = line.strip()
        if len(c) == 0:
            continue
        character_data = json.loads(line)
        if character_data["character"] == character:
            break

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

    stroke_part_counts = [
        ceil(parts_per_stroke_unit * stroke_length / (0.5 * 1024))
        for stroke_length in stroke_lengths
    ]

    # simplify curve by removing points, then smoothen resulting curve
    if config_smoothen_curve:
        stroke_medians = [
            smoothen_curve(
                interpolate_equidistant_medians(
                    stroke_medians[i],
                    max(ceil(stroke_lengths[i] / (smoothen_curve_smoothness * 50)), 3)
                )
            )
            for i in range(len(stroke_medians))
        ]

    stroke_medians = [
        interpolate_equidistant_medians(
            stroke_medians[i],
            stroke_part_counts[i],
        ) for i in range(len(stroke_medians))
    ]

    print(stroke_lengths, stroke_part_counts)
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
    avg_part_stretch = stretch / parts_per_stroke_unit
    stroke_zs = np.cumsum(
        [0]
        + [
            (
                i * flat_mode_spacing
                if flat_mode
                else parts_count * avg_part_stretch + distance_between_strokes
            )
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
    if debug_voronoi:
        n_strokes = len(stroke_paths_medians_lengths_counts)
        nrows = 2 if n_strokes <= 6 else 3
        ncols = ceil(n_strokes / nrows)
        _fig, axes_grid = plt.subplots(nrows=nrows, ncols=ncols)
        plot_axes = flat(axes_grid)

    # plot_axes = [fig.add_subplot() for i in enumerate(stroke_paths_medians_lengths_counts)]

    for i, (stroke_path, stroke_medians, _, parts_count) in enumerate(
        stroke_paths_medians_lengths_counts
    ):
        part_z_fn = lambda i, l: i * avg_part_stretch
        plot_ax = plot_axes[i] if debug_voronoi else None
        stroke_obj = generate_stroke(
            stroke_path,
            stroke_medians,
            part_z_fn,
            thickness,
            extrude_thickness,
            debug_voronoi,
            plot_ax,
        )
        if debug_voronoi:
            xs, ys = zip(*orig_stroke_medians[i])
            plot_ax.plot(xs, ys, 'ro', markersize=2)
        stroke_z = stroke_zs[i] + thickness / 2
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

    if debug_voronoi:
        plt.tight_layout()
        plt.show()

    if enable_connectors:
        for i, (medians1, medians2) in enumerate(pairwise(medians_3d)):
            p1_inset = floor(connector_end_distance * len(medians1))
            p2_inset = floor(connector_end_distance * len(medians2))
            p1 = medians1[-(p1_inset + 1)]
            p2 = medians2[p2_inset]
            if force_horizontal_connectors:
                avg_z = (p1.z + p2.z) / 2
                p1 = p1.copy()
                p1.z = avg_z
                p2 = p2.copy()
                p2.z = avg_z
            connectors += line_module.line(
                p1, p2, connector_thickness, segments=connector_n_segments
            )

    if untilted_mode:
        arr = np.array(flat(medians_3d))
        pca = PCA()
        pca.fit(arr)
        eigenvectors = pca.components_
        # make sure z doesnt have an effect on x and y
        eigenvectors[0][2] = 0
        eigenvectors[1][2] = 0
        print(eigenvectors)
        # eigenvectors[2] = [0, 0, 1]

        print(np.matmul(np.array(eigenvectors), (0, 0, 1)))

        # make sure all eigenvectors are in the same direction as the current axis
        for i in range(3):
            # for Z, if 0, 0, 1 would map to *, *, < 0, then invert it
            if eigenvectors[i][i] < 0:
                print(f"inverting {i} eigenvectors")
                eigenvectors[i] = -eigenvectors[i]
        mat = [(*values, 0) for values in eigenvectors]
        print(mat)

        if enable_untilted_axis:
            for i, eigenvector in enumerate(eigenvectors):
                c = "red" if i == 0 else ("green" if i == 1 else "blue")
                debug += multmatrix(mat)(
                    color(c)(
                        line_module.line((0, 0, 0), Point3(*eigenvector) * 200, 20)
                    )
                )

        strokes = multmatrix(mat)(strokes)

        # obj += color('blue')(sphere(30))
        # mat = np.matmul(mat, [[1, 0, 0, 0], [0, 1, 0, 0], [0.3, 0, 1, 0], [0, 0, 0, 1]])

        if enable_untilted_axis:
            for i, eigenvector in enumerate(eigenvectors):
                c = "pink" if i == 0 else ("lightgreen" if i == 1 else "lightblue")
                debug += multmatrix(mat)(
                    color(c)(
                        line_module.line((0, 0, 0), Point3(*eigenvector) * 200, 20)
                    )
                )

        medians_3d: Sequence[Sequence[Point3]] = [
            [Point3(*np.matmul(eigenvectors, np.array(median))) for median in medians]
            for medians in medians_3d
        ]

    medians_max_z = max(*[p[2] for p in flat(medians_3d)])
    if untilted_mode:
        medians_max_z = medians_max_z + untilted_mode_bottom_margin
    bottom = medians_max_z + thickness / 2

    strokes = up(-bottom)(strokes)
    connectors = up(-bottom)(connectors)

    medians_3d: Sequence[Sequence[Point3]] = [
        [median + Point3(0, 0, -bottom) for median in medians] for medians in medians_3d
    ]

    if centering_method == "average_medians":
        center = Point3(*tuple(map(np.mean, zip(*flat(medians_3d)))))
        strokes = translate((-center[0], -center[1], 0))(strokes)

    if to_bottom_mode:
        cube_height = 3000
        strokes = intersection()(
            strokes,
            down(cube_height / 2 - 1)(cube((1024, 1024, cube_height), center=True)),
        )

    plate_z = -plate_overlap
    if enable_pillars:
        for i, medians in enumerate(medians_3d):
            p = medians[floor(len(medians) / 2)]
            p_prev = medians[max(0, floor(len(medians) / 2) - 1)]
            p_next = medians[min(len(medians) - 1, floor(len(medians) / 2) + 1)]
            direction = p_next - p_prev
            # pdb.set_trace()
            # debug += color('red')(line_module.line(p, p + rico * 10, 20))
            angle = np.arctan2(direction.y, direction.x) * 180 / pi
            insert_height = thickness * 3
            insert_insertion = thickness * 0.4
            pillar_insert_end_p = p + (0, 0, (thickness / 2) - insert_insertion)
            pillar_cone_start_p = pillar_insert_end_p + (0, 0, insert_insertion + 5)
            pillar = rod_module.line(
                pillar_cone_start_p,
                (p.x, p.y, plate_z + plate_height / 2),
                pillar_thickness,
            )
            pillar += rod_module.cone(
                pillar_cone_start_p,
                pillar_insert_end_p,
                pillar_thickness,
                pillar_thickness / 2,
            )
            insert_segment_count = pillar_insert_n_segments
            insert_multiplier = 1.2  # sqrt(2)
            insert_angle = angle + 45 - 30 if insert_segment_count == 6 else angle

            def extrude_insert(surface):
                return translate(pillar_insert_end_p)(
                    rotate((0, 0, insert_angle))(linear_extrude(insert_height)(surface))
                )

            insert = extrude_insert(
                circle(
                    pillar_thickness * insert_multiplier, segments=insert_segment_count
                )
            )
            insert_cutout = extrude_insert(
                circle(
                    pillar_thickness * insert_multiplier + pillar_insert_margin,
                    segments=insert_segment_count,
                )
            )
            insert = intersection()(insert, strokes)
            pillars += pillar + insert
            pillars_cutouts += pillar + insert_cutout

    strokes -= pillars_cutouts

    if enable_plate:
        r1 = 512
        r2 = 80
        plate += up(plate_z + plate_height / 2)(
            cylinder(r=r1 + r2, h=plate_height, center=True)
        )

    obj = cube(0)
    for part in parts:
        part_obj = {
            "strokes": strokes,
            "plate": plate,
            "pillars": pillars,
            "connectors": connectors,
            "debug": debug,
        }[part]
        obj += part_obj

    return scale(config_scale * 60 / 1024)(rotate((-180, 0, 0))(obj))


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
    from deepmerge import always_merger

    parser = ArgumentParser("character_generator")
    parser.add_argument("--character", help="Hanzi", type=str)
    parser.add_argument("--out-dir", help="Out dir", type=str)
    parser.add_argument("--out-scad", help="Out .scad file", type=str)
    parser.add_argument("--out-stl", help="Out .stl file", type=str)
    parser.add_argument("--stl", help="Stl or not", type=bool)
    parser.add_argument(
        "--parts",
        help="Comma-separated parts (strokes, plate, pillars, connectors, debug)",
        type=str,
    )
    parser.add_argument("--settings", help="Settings preset (.yaml file)", type=str)
    parser.add_argument("--scale", help="Scale the model", type=float)
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

    config = always_merger.merge(base_config, config)

    if args.parts is not None:
        config["parts"] = args.parts.split(",")

    if args.character is not None:
        config["character"] = args.character

    if args.scale is not None:
        config["scale"] = args.scale

    a = generate(config)

    header = "$fn = 40;"
    scad_filepath = (
        args.out_scad
        if args.out_scad is not None
        else out_dir + "/" + args.character + ".scad"
    )
    stl_filepath = (
        args.out_stl
        if args.out_stl is not None
        else out_dir + "/" + args.character + ".stl"
    )
    file_out = scad_render_to_file(
        a, filepath=scad_filepath, file_header=header, include_orig_code=False
    )
    print(f"SCAD file written to: \n{file_out}")

    if args.stl:
        print("Generating stl (this might take a while)")
        run([find_openscad(), "-o", stl_filepath, scad_filepath])
