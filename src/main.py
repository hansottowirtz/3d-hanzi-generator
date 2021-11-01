#! /usr/bin/env python3
from typing import Sequence, Tuple
from solid.objects import (
    cylinder,
    import_scad,
    intersection,
    linear_extrude,
    multmatrix,
    polygon,
    sphere,
    union,
)
from solid.solidpython import OpenSCADObject
import sys
from math import ceil, cos, floor, radians, sin, pi, tan, tau

from euclid3 import Point2, Point3, Vector3

from solid import (
    scad_render_to_file,
    text,
    translate,
    cube,
    color,
    rotate,
    square,
    scale,
)
from solid.utils import (
    UP_VEC,
    Vector23,
    distribute_in_grid,
    down,
    extrude_along_path,
    up,
)
from solid.splines import catmull_rom_polygon, bezier_points, catmull_rom_points
from euclid3 import Point3, Point2
from subprocess import run
from svgpathtools import parse_path, Path
import json
from pathlib import Path as pathlib_Path
import pdb
import numpy as np
from more_itertools import pairwise
from svgpathtools.path import Line
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from math import sqrt

root = pathlib_Path(__file__).parent.resolve()
line_module = import_scad(root.joinpath("line.scad"))

enable_connectors = False
force_straight_connectors = False
flat_mode = False
to_bottom_mode = True

height_per_stroke = 130
parts_per_stroke_unit = 10
distance_between_parts = 1
distance_between_strokes = 0
thickness = 100
stretch = 100
connector_thickness = 25
connector_end_distance = 0.1
flat_mode_spacing = 5
to_bottom_mode_margin = 20
to_bottom_mode_enable_plate = True
to_bottom_mode_plate_height = 40

extrude_thickness = 2000 if to_bottom_mode else thickness


def spt_char_point_to_tuple_point(p):
    return (np.real(p), -(np.imag(p) - 900))


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
        (-1024, -1024),
        (-1024, 2048),
        (2048, 2048),
        (2048, -1024),
        *org_voronoi_ps,
    ]
    vor = Voronoi(voronoi_ps)

    # fig = voronoi_plot_2d(vor)
    # plt.show()

    # skip the boundary-ensuring points
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


def generate(character):
    graphics_file = open(root.joinpath("../res/graphics.txt"), "r")
    character_data = {}

    for line in graphics_file.readlines():
        c = line.strip()
        if len(c) == 0:
            continue
        character_data = json.loads(line)
        if character_data["character"] == character:
            break

    obj = cube(0)

    stroke_medians = [
        normalize_medians(medians) for medians in character_data["medians"]
    ]

    stroke_lengths = [calculate_stroke_length(medians) for medians in stroke_medians]

    stroke_part_counts = [
        ceil(parts_per_stroke_unit * stroke_length / (0.5 * 1024))
        for stroke_length in stroke_lengths
    ]

    print(stroke_lengths, stroke_part_counts)
    stroke_paths_medians_lengths_counts = [
        (
            parse_path(character_data["strokes"][i]),
            interpolate_equidistant_medians(stroke_medians[i], stroke_part_counts[i]),
            stroke_lengths[i],
            stroke_part_counts[i],
        )
        for i in range(len(character_data["strokes"]))
    ]

    # pdb.set_trace()
    height_multiplier = height_per_stroke + distance_between_strokes - thickness
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
    print(stroke_zs)
    for i, (stroke_path, stroke_medians, _, parts_count) in enumerate(
        stroke_paths_medians_lengths_counts
    ):
        part_z_fn = lambda i, l: i * avg_part_stretch
        stroke_obj = generate_stroke(stroke_path, stroke_medians, part_z_fn)
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
        obj += up(stroke_z)(stroke_obj)

    if enable_connectors:
        for i, (medians1, medians2) in enumerate(pairwise(medians_3d)):
            p1_inset = floor(connector_end_distance * len(medians1))
            p2_inset = floor(connector_end_distance * len(medians2))
            p1 = medians1[-(p1_inset + 1)]
            p2 = medians2[p2_inset]
            if force_straight_connectors:
                avg_z = (p1.z + p2.z) / 2
                p1 = p1.copy()
                p1.z = avg_z
                p2 = p2.copy()
                p2.z = avg_z
            obj += line_module.line(p1, p2, connector_thickness)

    if to_bottom_mode:
        top_margin = 5
        bottom = stroke_zs[-1] + to_bottom_mode_margin
        obj = intersection()(
            obj,
            down(top_margin)(cube((1024, 1024, bottom + top_margin))),
        )
        # obj += up(bottom)(cube((1024, 1024, 20)))
        if to_bottom_mode_enable_plate:
            r1 = 512
            r2 = 100
            obj += translate((1024 - r1, 1024 - r1))(
                up(bottom)(cylinder(r=r1 + r2, h=to_bottom_mode_plate_height))
            )

    return scale(60 / 1024)(obj)


# ===============
# = ENTRY POINT =
# ===============
if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser("character_generator")
    parser.add_argument("--character", help="Hanzi", type=str)
    parser.add_argument("--out", help="Out dir", type=str)
    parser.add_argument("--stl", help="Stl or not", type=bool)
    args = parser.parse_args()

    out_dir = args.out

    a = generate(args.character)

    # header = '$fa = 20;\n$fs = 20;'
    header = ""
    filepath = out_dir + "/" + args.character + ".scad"
    file_out = scad_render_to_file(
        a, filepath=filepath, file_header=header, include_orig_code=True
    )
    print(f"{__file__}: SCAD file written to: \n{file_out}")

    if args.stl:
        run(["openscad", "-o", out_dir + "/" + args.character + ".stl", filepath])
