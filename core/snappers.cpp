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

void Snappers::set_stretch_mode(String p_mode) {
	_stretch_mode_viewport = p_mode == "viewport";
}

void Snappers::set_transform_snap_2d(AdvancedSettings::Snap2DType p_type, AdvancedSettings::RoundMode p_mode_x, AdvancedSettings::RoundMode p_mode_y) {
	switch (p_type) {
		case AdvancedSettings::SNAP2D_TYPE_ITEM_PRE: {
			snapper_canvas_item_pre.set_snap_modes(p_mode_x, p_mode_y);
		} break;
		case AdvancedSettings::SNAP2D_TYPE_ITEM_POST: {
			snapper_canvas_item_post.set_snap_modes(p_mode_x, p_mode_y);
		} break;
		case AdvancedSettings::SNAP2D_TYPE_ITEM_READ: {
			snapper_canvas_item_read.set_snap_modes(p_mode_x, p_mode_y);
		} break;
		case AdvancedSettings::SNAP2D_TYPE_VIEWPORT_PRE: {
			snapper_viewport_pre.set_snap_modes(p_mode_x, p_mode_y);
		} break;
		case AdvancedSettings::SNAP2D_TYPE_VIEWPORT_POST: {
			snapper_viewport_post.set_snap_modes(p_mode_x, p_mode_y);
		} break;
		case AdvancedSettings::SNAP2D_TYPE_VIEWPORT_PARENT_PRE: {
			snapper_viewport_parent_pre.set_snap_modes(p_mode_x, p_mode_y);
		} break;
		default: {
		} break;
	}
}

void Snappers::initialize(bool p_gpu_snap) {
	_gpu_snap_enabled = p_gpu_snap;
	_snap_transforms_enabled = false;
}
