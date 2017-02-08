# -*- coding: utf-8 -*-

# Basic exporter for svg icons

from os import listdir
from os.path import isfile, join, dirname, realpath
import subprocess
import sys

import rsvg
import cairo

last_svg_path = None
last_svg_data = None

SCRIPT_FOLDER = dirname(realpath(__file__)) + '/'
theme_dir_base = SCRIPT_FOLDER + '../../scene/resources/default_theme/'
theme_dir_source = theme_dir_base + 'source/'
icons_dir_base = SCRIPT_FOLDER + '../editor/icons/'
icons_dir_2x = icons_dir_base + '2x/'
icons_dir_source = icons_dir_base + 'source/'


def svg_to_png(svg_path, png_path, dpi):
    global last_svg_path, last_svg_data

    zoom = int(dpi / 90)
    if last_svg_path != svg_path:
        last_svg_data = open(svg_path, 'r').read()
        last_svg_path = svg_path
    svg = rsvg.Handle(data=last_svg_data)
    img = cairo.ImageSurface(
        cairo.FORMAT_ARGB32,
        svg.props.width * zoom,
        svg.props.height * zoom
    )
    ctx = cairo.Context(img)
    ctx.set_antialias(cairo.ANTIALIAS_DEFAULT)
    ctx.scale(zoom, zoom)
    svg.render_cairo(ctx)
    img.write_to_png('%s.png' % png_path)
    svg.close()


def export_icons():
    svgs_path = icons_dir_source

    file_names = [f for f in listdir(svgs_path) if isfile(join(svgs_path, f))]

    for file_name in file_names:
        # name without extensions
        name_only = file_name.replace('.svg', '')

        out_icon_names = [name_only]  # export to a png with the same file name
        theme_out_icon_names = []
        # special cases
        if special_icons.has_key(name_only):
            special_icon = special_icons[name_only]
            if type(special_icon) is dict:
                if special_icon.get('avoid_self'):
                    out_icon_names = []
                if special_icon.has_key('output_names'):
                    out_icon_names += special_icon['output_names']
                if special_icon.has_key('theme_output_names'):
                    theme_out_icon_names += special_icon['theme_output_names']

        source_path = '%s%s.svg' % (svgs_path, name_only)

        for out_icon_name in out_icon_names:
            svg_to_png(source_path, icons_dir_base + out_icon_name, 90)
            svg_to_png(source_path, icons_dir_2x + out_icon_name, 180)
        for theme_out_icon_name in theme_out_icon_names:
            svg_to_png(source_path, theme_dir_base + theme_out_icon_name, 90)


def export_theme():
    svgs_path = theme_dir_source
    file_names = [f for f in listdir(svgs_path) if isfile(join(svgs_path, f))]

    for file_name in file_names:
        # name without extensions
        name_only = file_name.replace('.svg', '')

        out_icon_names = [name_only]  # export to a png with the same file name
        # special cases
        if theme_icons.has_key(name_only):
            special_icon = theme_icons[name_only]
            if type(special_icon) is dict:
                if special_icon.has_key('output_names'):
                    out_icon_names += special_icon['output_names']

        source_path = '%s%s.svg' % (svgs_path, name_only)

        for out_icon_name in out_icon_names:
            svg_to_png(source_path, theme_dir_base + out_icon_name, 90)


# special cases for icons that will be exported to multiple target pngs or that require transforms.
special_icons = {
    'icon_add_track': dict(
        output_names=['icon_add'],
        theme_output_names=['icon_add', 'icon_zoom_more']
    ),
    'icon_new': dict(output_names=['icon_file']),
    'icon_animation_tree_player': dict(output_names=['icon_animation_tree']),
    'icon_tool_rotate': dict(
        output_names=['icon_reload'],
        theme_output_names=['icon_reload']
    ),
    'icon_multi_edit': dict(output_names=['icon_multi_node_edit']),
    'icon_folder': dict(
        output_names=['icon_load', 'icon_open'],
        theme_output_names=['icon_folder']
    ),
    'icon_file_list': dict(output_names=['icon_enum']),
    'icon_collision_2d': dict(output_names=['icon_collision_polygon_2d', 'icon_polygon_2d']),
    'icon_class_list': dict(output_names=['icon_filesystem']),
    'icon_color_ramp': dict(output_names=['icon_graph_color_ramp']),
    'icon_translation': dict(output_names=['icon_p_hash_translation']),
    'icon_shader': dict(output_names=['icon_shader_material', 'icon_material_shader']),
    'icon_canvas_item_shader_graph': dict(output_names=['icon_material_shader_graph']),

    'icon_color_pick': dict(theme_output_names=['icon_color_pick'], avoid_self=True),
    'icon_play': dict(theme_output_names=['icon_play']),
    'icon_stop': dict(theme_output_names=['icon_stop']),
    'icon_zoom_less': dict(theme_output_names=['icon_zoom_less'], avoid_self=True),
    'icon_zoom_reset': dict(theme_output_names=['icon_zoom_reset'], avoid_self=True),
    'icon_snap': dict(theme_output_names=['icon_snap'])
}

theme_icons = {
    'icon_close': dict(output_names=['close', 'close_hl']),
    'tab_menu': dict(output_names=['tab_menu_hl'])
}

export_icons()
export_theme()
