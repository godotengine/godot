# -*- coding: utf-8 -*-

# Basic exporter for svg icons (requires Inkscape)

import os.path
from os import listdir
from os.path import isfile, join
import subprocess

SVGS_PATH = 'source/'
OUT_DIR = '2x/'
DPI = 180


def export_all(svgs_path=SVGS_PATH, out_dir=OUT_DIR, dpi=DPI):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    file_names = [f for f in listdir(svgs_path) if isfile(join(svgs_path, f))]

    for file_name in file_names:
        # name without extensions
        name_only = file_name.replace('.svg', '')

        icon_from_name = name_only
        out_icon_names = [name_only] # export to a png with the same file name
        rotations = []
        transforms = []

        # special cases
        if special_icons.has_key(name_only):
            special_icon = special_icons[name_only]
            if type(special_icon) is dict:
                if special_icon.has_key('output_names'):
                    out_icon_names += special_icon['output_names']

        svg_file_path = '%s%s.svg' % (svgs_path, icon_from_name)

        for index, out_icon_name in enumerate(out_icon_names):
            subprocess.call(
                'inkscape -z -f {input} -d {dpi} -e{output}'.format(
                input=svg_file_path,
                dpi=dpi,
                output='%s%s.png' % (out_dir, out_icon_name)
            ), shell=True)


# special cases for icons that will be exported to multiple target pngs or that require transforms.
special_icons = {
    'icon_add_track': dict( output_names=['icon_add'] ),
    'icon_new': dict( output_names=['icon_file'] ),
    'icon_animation_tree_player': dict( output_names=['icon_animation_tree'] ),
    'icon_tool_rotate': dict( output_names=['icon_reload'] ),
    'icon_multi_edit': dict( output_names=['icon_multi_node_edit'] ),
    'icon_folder': dict( output_names=['icon_load'] ),
    'icon_file_list': dict( output_names=['icon_enum'] ),
    'icon_collision_2d': dict( output_names=['icon_collision_polygon_2d', 'icon_polygon_2d'] ),
    'icon_class_list': dict( output_names=['icon_filesystem'] ),
    'icon_color_ramp': dict( output_names=['icon_graph_color_ramp'] ),
    'icon_translation': dict( output_names=['icon_p_hash_translation'] ),
    'icon_shader': dict( output_names=['icon_shader_material', 'icon_material_shader'] ),
    'icon_canvas_item_shader_graph': dict( output_names=['icon_material_shader_graph'] ),

}

export_all()
