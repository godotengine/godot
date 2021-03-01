/*************************************************************************/
/*  snappers.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "snappers.h"
#include "core/project_settings.h"

void Snappers::snap_read_item(Vector2 &r_pos) const {
	if (snapper_canvas_item_read.is_enabled()) {
		snapper_canvas_item_read.snap(r_pos);
	} else if (_gpu_snap_enabled) {
		r_pos = r_pos.floor();
	}
}

void Snappers::initialize(bool p_gpu_snap, bool p_snap_transforms, bool p_snap_viewports, bool p_stretch_mode_viewport) {

	_gpu_snap_enabled = p_gpu_snap;

	const char *sz_mode_selection = "Default,Disabled,Floor,Ceiling,Round";

#define GODOT_SNAP_DEFINE(MODE_STRING) \
	GLOBAL_DEF(MODE_STRING, 0);        \
	ProjectSettings::get_singleton()->set_custom_property_info(MODE_STRING, PropertyInfo(Variant::INT, MODE_STRING, PROPERTY_HINT_ENUM, sz_mode_selection))

	int item_pre_x = GODOT_SNAP_DEFINE("rendering/2d/snapping_modes/item_pre_x");
	int item_pre_y = GODOT_SNAP_DEFINE("rendering/2d/snapping_modes/item_pre_y");
	int item_post_x = GODOT_SNAP_DEFINE("rendering/2d/snapping_modes/item_post_x");
	int item_post_y = GODOT_SNAP_DEFINE("rendering/2d/snapping_modes/item_post_y");
	int item_read_x = GODOT_SNAP_DEFINE("rendering/2d/snapping_modes/item_read_x");
	int item_read_y = GODOT_SNAP_DEFINE("rendering/2d/snapping_modes/item_read_y");

	int camera_pre_x = GODOT_SNAP_DEFINE("rendering/2d/snapping_modes/camera_pre_x");
	int camera_pre_y = GODOT_SNAP_DEFINE("rendering/2d/snapping_modes/camera_pre_y");
	int camera_post_x = GODOT_SNAP_DEFINE("rendering/2d/snapping_modes/camera_post_x");
	int camera_post_y = GODOT_SNAP_DEFINE("rendering/2d/snapping_modes/camera_post_y");
	int camera_parent_pre_x = GODOT_SNAP_DEFINE("rendering/2d/snapping_modes/camera_parent_pre_x");
	int camera_parent_pre_y = GODOT_SNAP_DEFINE("rendering/2d/snapping_modes/camera_parent_pre_y");

	// defaults
	if (p_snap_transforms) {
		if (p_stretch_mode_viewport) {
			snapper_canvas_item_pre.set_snap_modes(Snapper2D::SNAP_ROUND, Snapper2D::SNAP_CEILING);
		} else {
			snapper_canvas_item_post.set_snap_modes(Snapper2D::SNAP_ROUND, Snapper2D::SNAP_ROUND);
		}
	}

	if (p_snap_viewports) {
		if (p_stretch_mode_viewport) {
			snapper_viewport_pre.set_snap_modes(Snapper2D::SNAP_ROUND, Snapper2D::SNAP_ROUND);
		} else {
			snapper_viewport_post.set_snap_modes(Snapper2D::SNAP_ROUND, Snapper2D::SNAP_ROUND);
		}
	}

	// default actions for these derive from earlier types
	snapper_canvas_item_read = snapper_canvas_item_pre;
	snapper_viewport_parent_pre = snapper_viewport_pre;

	// custom user overrides
	if (p_snap_transforms) {
		snapper_canvas_item_pre.set_custom_snap_modes((Snapper2D::SnapMode)item_pre_x, (Snapper2D::SnapMode)item_pre_y);
		snapper_canvas_item_post.set_custom_snap_modes((Snapper2D::SnapMode)item_post_x, (Snapper2D::SnapMode)item_post_y);
		snapper_canvas_item_read.set_custom_snap_modes((Snapper2D::SnapMode)item_read_x, (Snapper2D::SnapMode)item_read_y);
	}

	if (p_snap_viewports) {
		snapper_viewport_pre.set_custom_snap_modes((Snapper2D::SnapMode)camera_pre_x, (Snapper2D::SnapMode)camera_pre_y);
		snapper_viewport_post.set_custom_snap_modes((Snapper2D::SnapMode)camera_post_x, (Snapper2D::SnapMode)camera_post_y);
		snapper_viewport_parent_pre.set_custom_snap_modes((Snapper2D::SnapMode)camera_parent_pre_x, (Snapper2D::SnapMode)camera_parent_pre_y);
	}

#undef GODOT_SNAP_DEFINE
}
