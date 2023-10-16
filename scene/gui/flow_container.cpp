/**************************************************************************/
/*  flow_container.cpp                                                    */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "flow_container.h"

#include "scene/theme/theme_db.h"

struct _LineData {
	int child_count = 0;
	int min_line_height = 0;
	int min_line_length = 0;
	int stretch_avail = 0;
	float stretch_ratio_total = 0;
};

void FlowContainer::_resort() {
	// Avoid resorting if invisible.
	if (!is_visible_in_tree()) {
		return;
	}

	bool rtl = is_layout_rtl();

	HashMap<Control *, Size2i> children_minsize_cache;

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
		if (!child || !child->is_visible()) {
			continue;
		}
		if (child->is_set_as_top_level()) {
			continue;
		}

		Size2i child_msc = child->get_combined_minimum_size();

		if (vertical) { /* VERTICAL */
			if (children_in_current_line > 0) {
				ofs.y += theme_cache.v_separation;
			}
			if (ofs.y + child_msc.y > current_container_size) {
				line_length = ofs.y - theme_cache.v_separation;
				lines_data.push_back(_LineData{ children_in_current_line, line_height, line_length, current_container_size - line_length, line_stretch_ratio_total });

				// Move in new column (vertical line).
				ofs.x += line_height + theme_cache.h_separation;
				ofs.y = 0;
				line_height = 0;
				line_stretch_ratio_total = 0;
				children_in_current_line = 0;
			}

			line_height = MAX(line_height, child_msc.x);
			if (child->get_v_size_flags().has_flag(SIZE_EXPAND)) {
				line_stretch_ratio_total += child->get_stretch_ratio();
			}
			ofs.y += child_msc.y;

		} else { /* HORIZONTAL */
			if (children_in_current_line > 0) {
				ofs.x += theme_cache.h_separation;
			}
			if (ofs.x + child_msc.x > current_container_size) {
				line_length = ofs.x - theme_cache.h_separation;
				lines_data.push_back(_LineData{ children_in_current_line, line_height, line_length, current_container_size - line_length, line_stretch_ratio_total });

				// Move in new line.
				ofs.y += line_height + theme_cache.v_separation;
				ofs.x = 0;
				line_height = 0;
				line_stretch_ratio_total = 0;
				children_in_current_line = 0;
			}

			line_height = MAX(line_height, child_msc.y);
			if (child->get_h_size_flags().has_flag(SIZE_EXPAND)) {
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
		if (!child || !child->is_visible()) {
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
				ofs.x += line_data.min_line_height + theme_cache.h_separation;
				ofs.y = 0;
			} else {
				ofs.x = 0;
				ofs.y += line_data.min_line_height + theme_cache.v_separation;
			}
			line_data = lines_data[current_line_idx];
		}

		// The first child of each line adds the offset caused by the alignment,
		// but only if the line doesn't contain a child that expands.
		if (child_idx_in_line == 0 && Math::is_equal_approx(line_data.stretch_ratio_total, 0)) {
			int alignment_ofs = 0;
			switch (alignment) {
				case ALIGNMENT_CENTER:
					alignment_ofs = line_data.stretch_avail / 2;
					break;
				case ALIGNMENT_END:
					alignment_ofs = line_data.stretch_avail;
					break;
				default:
					break;
			}

			if (vertical) { /* VERTICAL */
				ofs.y += alignment_ofs;
			} else { /* HORIZONTAL */
				ofs.x += alignment_ofs;
			}
		}

		if (vertical) { /* VERTICAL */
			if (child->get_h_size_flags().has_flag(SIZE_FILL) || child->get_h_size_flags().has_flag(SIZE_SHRINK_CENTER) || child->get_h_size_flags().has_flag(SIZE_SHRINK_END)) {
				child_size.width = line_data.min_line_height;
			}

			if (child->get_v_size_flags().has_flag(SIZE_EXPAND)) {
				int stretch = line_data.stretch_avail * child->get_stretch_ratio() / line_data.stretch_ratio_total;
				child_size.height += stretch;
			}

		} else { /* HORIZONTAL */
			if (child->get_v_size_flags().has_flag(SIZE_FILL) || child->get_v_size_flags().has_flag(SIZE_SHRINK_CENTER) || child->get_v_size_flags().has_flag(SIZE_SHRINK_END)) {
				child_size.height = line_data.min_line_height;
			}

			if (child->get_h_size_flags().has_flag(SIZE_EXPAND)) {
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
			ofs.y += child_size.height + theme_cache.v_separation;
		} else { /* HORIZONTAL */
			ofs.x += child_size.width + theme_cache.h_separation;
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

Vector<int> FlowContainer::get_allowed_size_flags_horizontal() const {
	Vector<int> flags;
	flags.append(SIZE_FILL);
	if (!vertical) {
		flags.append(SIZE_EXPAND);
	}
	flags.append(SIZE_SHRINK_BEGIN);
	flags.append(SIZE_SHRINK_CENTER);
	flags.append(SIZE_SHRINK_END);
	return flags;
}

Vector<int> FlowContainer::get_allowed_size_flags_vertical() const {
	Vector<int> flags;
	flags.append(SIZE_FILL);
	if (vertical) {
		flags.append(SIZE_EXPAND);
	}
	flags.append(SIZE_SHRINK_BEGIN);
	flags.append(SIZE_SHRINK_CENTER);
	flags.append(SIZE_SHRINK_END);
	return flags;
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

void FlowContainer::_validate_property(PropertyInfo &p_property) const {
	if (is_fixed && p_property.name == "vertical") {
		p_property.usage = PROPERTY_USAGE_NONE;
	}
}

int FlowContainer::get_line_count() const {
	return cached_line_count;
}

void FlowContainer::set_alignment(AlignmentMode p_alignment) {
	if (alignment == p_alignment) {
		return;
	}
	alignment = p_alignment;
	_resort();
}

FlowContainer::AlignmentMode FlowContainer::get_alignment() const {
	return alignment;
}

void FlowContainer::set_vertical(bool p_vertical) {
	ERR_FAIL_COND_MSG(is_fixed, "Can't change orientation of " + get_class() + ".");
	vertical = p_vertical;
	update_minimum_size();
	_resort();
}

bool FlowContainer::is_vertical() const {
	return vertical;
}

FlowContainer::FlowContainer(bool p_vertical) {
	vertical = p_vertical;
}

void FlowContainer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_line_count"), &FlowContainer::get_line_count);

	ClassDB::bind_method(D_METHOD("set_alignment", "alignment"), &FlowContainer::set_alignment);
	ClassDB::bind_method(D_METHOD("get_alignment"), &FlowContainer::get_alignment);
	ClassDB::bind_method(D_METHOD("set_vertical", "vertical"), &FlowContainer::set_vertical);
	ClassDB::bind_method(D_METHOD("is_vertical"), &FlowContainer::is_vertical);

	BIND_ENUM_CONSTANT(ALIGNMENT_BEGIN);
	BIND_ENUM_CONSTANT(ALIGNMENT_CENTER);
	BIND_ENUM_CONSTANT(ALIGNMENT_END);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "alignment", PROPERTY_HINT_ENUM, "Begin,Center,End"), "set_alignment", "get_alignment");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "vertical"), "set_vertical", "is_vertical");

	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, FlowContainer, h_separation);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, FlowContainer, v_separation);
}
