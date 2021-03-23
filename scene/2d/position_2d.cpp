/*************************************************************************/
/*  position_2d.cpp                                                      */
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

#include "position_2d.h"

#include "core/config/engine.h"
#include "scene/resources/texture.h"

const real_t DEFAULT_GIZMO_EXTENTS = 10.0;

void Position2D::_draw_cross() {
	real_t extents = get_gizmo_extents();
	// Colors taken from `axis_x_color` and `axis_y_color` (defined in `editor/editor_themes.cpp`)
	draw_line(Point2(-extents, 0), Point2(+extents, 0), Color(0.96, 0.20, 0.32));
	draw_line(Point2(0, -extents), Point2(0, +extents), Color(0.53, 0.84, 0.01));
}

#ifdef TOOLS_ENABLED
Rect2 Position2D::_edit_get_rect() const {
	real_t extents = get_gizmo_extents();
	return Rect2(Point2(-extents, -extents), Size2(extents * 2, extents * 2));
}

bool Position2D::_edit_use_rect() const {
	return false;
}
#endif

void Position2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			update();
		} break;
		case NOTIFICATION_DRAW: {
			if (!is_inside_tree()) {
				break;
			}
			if (Engine::get_singleton()->is_editor_hint()) {
				_draw_cross();
			}

		} break;
	}
}

void Position2D::set_gizmo_extents(real_t p_extents) {
	if (p_extents == DEFAULT_GIZMO_EXTENTS) {
		set_meta("_gizmo_extents_", Variant());
	} else {
		set_meta("_gizmo_extents_", p_extents);
	}

	update();
}

real_t Position2D::get_gizmo_extents() const {
	if (has_meta("_gizmo_extents_")) {
		return get_meta("_gizmo_extents_");
	} else {
		return DEFAULT_GIZMO_EXTENTS;
	}
}

void Position2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_set_gizmo_extents", "extents"), &Position2D::set_gizmo_extents);
	ClassDB::bind_method(D_METHOD("_get_gizmo_extents"), &Position2D::get_gizmo_extents);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "gizmo_extents", PROPERTY_HINT_RANGE, "0,1000,0.1,or_greater", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_INTERNAL), "_set_gizmo_extents", "_get_gizmo_extents");
}

Position2D::Position2D() {
}
