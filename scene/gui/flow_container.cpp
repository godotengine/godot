/*************************************************************************/
/*  flow_container.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "scene/gui/container.h"

#include "flow_container.h"

struct _LineData {
	int child_count = 0;
	int min_line_height = 0;
	int min_line_length = 0;
	int stretch_avail = 0;
	float stretch_ratio_total = 0;
};

void FlowContainer::_resort() {
	int separation_horizontal = get_theme_constant(SNAME("hseparation"));
	int separation_vertical = get_theme_constant(SNAME("vseparation"));

	bool rtl = is_layout_rtl();

	Map<Control *, Size2i> children_minsize_cache;

	Vector<_LineData> lines_data;

	Vector2i ofs;
	int line_height = 0;
	int line_length = 0;
	float line_stretch_ratio_total = 0;
	int current_container_size = vertical ? get_rect().size.y : get_rect().size.x;
	int children_in_current_line = 0;

	// First pass for line wrapping and minimum size calculation.
	for (int i = 0; i < get_child_count(); i++) {
		Control *child = Object::cast_to<Control>(get_child(i));
		if (!child || !child->is_visible_in_tree()) {
			continue;
		}
		if (child->is_set_as_top_level()) {
			continue;
		}

		Size2i child_msc = child->get_combined_minimum_size();

		if (vertical) { /* VERTICAL */
			if (children_in_current_line > 0) {
				ofs.y += separation_vertical;
			}
			if (ofs.y + child_msc.y > current_container_size) {
				line_length = ofs.y - separation_vertical;
				lines_data.push_back(_LineData{ children_in_current_line, line_height, line_length, current_container_size - line_length, line_stretch_ratio_total });

				// Move in new column (vertical line).
				ofs.x += line_height + separation_horizontal;
				ofs.y = 0;
				line_height = 0;
				line_stretch_ratio_total = 0;
				children_in_current_line = 0;
			}

			line_height = MAX(line_height, child_msc.x);
			if (child->get_v_size_flags() & SIZE_EXPAND) {
				line_stretch_ratio_total += child->get_stretch_ratio();
			}
			ofs.y += child_msc.y;

		} else { /* HORIZONTAL */
			if (children_in_current_line > 0) {
				ofs.x += separation_horizontal;
			}
			if (ofs.x + child_msc.x > current_container_size) {
				line_length = ofs.x - separation_horizontal;
				lines_data.push_back(_LineData{ children_in_current_line, line_height, line_length, current_container_size - line_length, line_stretch_ratio_total });

				// Move in new line.
				ofs.y += line_height + separation_vertical;
				ofs.x = 0;
				line_height = 0;
				line_stretch_ratio_total = 0;
				children_in_current_line = 0;
			}

			line_height = MAX(line_height, child_msc.y);
			if (child->get_h_size_flags() & SIZE_EXPAND) {
				line_stretch_ratio_total += child->get_stretch_ratio();
			}
			ofs.x += child_msc.x;
		}

		children_minsize_cache[child] = child_msc;
		children_in_current_line++;
	}
	line_length = vertical ? (ofs.y) : (ofs.x);
	lines_data.push_back(_LineData{ children_in_current_line, line_height, line_length, current_container_size - line_length, line_stretch_ratio_total });

	// Second pass for in-line expansion and alignment.

	int current_line_idx = 0;
	int child_idx_in_line = 0;

	ofs.x = 0;
	ofs.y = 0;

	for (int i = 0; i < get_child_count(); i++) {
		Control *child = Object::cast_to<Control>(get_child(i));
		if (!child || !child->is_visible_in_tree()) {
			continue;
		}
		if (child->is_set_as_top_level()) {
			continue;
		}
		Size2i child_size = children_minsize_cache[child];

		_LineData line_data = lines_data[current_line_idx];
		if (child_idx_in_line >= lines_data[current_line_idx].child_count) {
			current_line_idx++;
			child_idx_in_line = 0;
			if (vertical) {
				ofs.x += line_data.min_line_height + separation_horizontal;
				ofs.y = 0;
			} else {
				ofs.x = 0;
				ofs.y += line_data.min_line_height + separation_vertical;
			}
			line_data = lines_data[current_line_idx];
		}

		if (vertical) { /* VERTICAL */
			if (child->get_h_size_flags() & (SIZE_FILL | SIZE_SHRINK_CENTER | SIZE_SHRINK_END)) {
				child_size.width = line_data.min_line_height;
			}

			if (child->get_v_size_flags() & SIZE_EXPAND) {
				int stretch = line_data.stretch_avail * child->get_stretch_ratio() / line_data.stretch_ratio_total;
				child_size.height += stretch;
			}

		} else { /* HORIZONTAL */
			if (child->get_v_size_flags() & (SIZE_FILL | SIZE_SHRINK_CENTER | SIZE_SHRINK_END)) {
				child_size.height = line_data.min_line_height;
			}

			if (child->get_h_size_flags() & SIZE_EXPAND) {
				int stretch = line_data.stretch_avail * child->get_stretch_ratio() / line_data.stretch_ratio_total;
				child_size.width += stretch;
			}
		}

		Rect2 child_rect = Rect2(ofs, child_size);
		if (rtl) {
			child_rect.position.x = get_rect().size.x - child_rect.position.x - child_rect.size.width;
		}

		fit_child_in_rect(child, child_rect);

		if (vertical) { /* VERTICAL */
			ofs.y += child_size.height + separation_vertical;
		} else { /* HORIZONTAL */
			ofs.x += child_size.width + separation_horizontal;
		}

		child_idx_in_line++;
	}
	cached_size = (vertical ? ofs.x : ofs.y) + line_height;
	cached_line_count = lines_data.size();
}

Size2 FlowContainer::get_minimum_size() const {
	Size2i minimum;

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
		if (!c) {
			continue;
		}
		if (c->is_set_as_top_level()) {
			continue;
		}

		if (!c->is_visible()) {
			continue;
		}

		Size2i size = c->get_combined_minimum_size();

		if (vertical) { /* VERTICAL */
			minimum.height = MAX(minimum.height, size.height);
			minimum.width = cached_size;

		} else { /* HORIZONTAL */
			minimum.width = MAX(minimum.width, size.width);
			minimum.height = cached_size;
		}
	}

	return minimum;
}

void FlowContainer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_SORT_CHILDREN: {
			_resort();
			update_minimum_size();
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			update_minimum_size();
		} break;
		case NOTIFICATION_TRANSLATION_CHANGED:
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED: {
			queue_sort();
		} break;
	}
}

int FlowContainer::get_line_count() const {
	return cached_line_count;
}

FlowContainer::FlowContainer(bool p_vertical) {
	vertical = p_vertical;
}

void FlowContainer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_line_count"), &FlowContainer::get_line_count);
}
