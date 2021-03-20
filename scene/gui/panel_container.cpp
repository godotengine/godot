/*************************************************************************/
/*  panel_container.cpp                                                  */
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

#include "panel_container.h"

Size2 PanelContainer::get_minimum_size() const {
	Ref<StyleBox> style;

	if (has_theme_stylebox("panel")) {
		style = get_theme_stylebox("panel");
	} else {
		style = get_theme_stylebox("panel", "PanelContainer");
	}

	Size2 ms;
	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
		if (!c || !c->is_visible()) {
			continue;
		}
		if (c->is_set_as_top_level()) {
			continue;
		}

		Size2 minsize = c->get_combined_minimum_size();
		ms.width = MAX(ms.width, minsize.width);
		ms.height = MAX(ms.height, minsize.height);
	}

	if (style.is_valid()) {
		ms += style->get_minimum_size();
	}
	return ms;
}

void PanelContainer::_notification(int p_what) {
	if (p_what == NOTIFICATION_DRAW) {
		RID ci = get_canvas_item();
		Ref<StyleBox> style;

		if (has_theme_stylebox("panel")) {
			style = get_theme_stylebox("panel");
		} else {
			style = get_theme_stylebox("panel", "PanelContainer");
		}

		style->draw(ci, Rect2(Point2(), get_size()));
	}

	if (p_what == NOTIFICATION_SORT_CHILDREN) {
		Ref<StyleBox> style;

		if (has_theme_stylebox("panel")) {
			style = get_theme_stylebox("panel");
		} else {
			style = get_theme_stylebox("panel", "PanelContainer");
		}

		Size2 size = get_size();
		Point2 ofs;
		if (style.is_valid()) {
			size -= style->get_minimum_size();
			ofs += style->get_offset();
		}

		for (int i = 0; i < get_child_count(); i++) {
			Control *c = Object::cast_to<Control>(get_child(i));
			if (!c || !c->is_visible_in_tree()) {
				continue;
			}
			if (c->is_set_as_top_level()) {
				continue;
			}

			fit_child_in_rect(c, Rect2(ofs, size));
		}
	}
}

PanelContainer::PanelContainer() {
	// Has visible stylebox, so stop by default.
	set_mouse_filter(MOUSE_FILTER_STOP);
}
