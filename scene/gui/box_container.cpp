/**************************************************************************/
/*  box_container.cpp                                                     */
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

#include "box_container.h"

#include "scene/gui/label.h"
#include "scene/gui/margin_container.h"
#include "scene/theme/theme_db.h"

struct StretchData {
	real_t min_size = 0;
	real_t stretch_ratio = 0.0;
	bool will_stretch = false;
	real_t final_size = 0;
};

void BoxContainer::_resort() {
	const Size2 new_size = get_size();
	const bool rtl = is_layout_rtl();

	int children_count = 0;
	real_t stretch_min = 0;
	real_t stretch_avail = 0;
	real_t stretch_ratio_total = 0.0;
	LocalVector<StretchData> stretch_data;

	// First pass, determine minimum sizes and available stretch space.
	for (int i = 0; i < get_child_count(); i++) {
		Control *c = as_sortable_control(get_child(i));
		if (!c) {
			continue;
		}

		const Size2 size = c->get_combined_minimum_size();
		StretchData sdata;

		if (vertical) {
			stretch_min += size.height;
			sdata.min_size = size.height;
			sdata.will_stretch = c->get_v_size_flags().has_flag(SIZE_EXPAND);
		} else {
			stretch_min += size.width;
			sdata.min_size = size.width;
			sdata.will_stretch = c->get_h_size_flags().has_flag(SIZE_EXPAND);
		}

		if (sdata.will_stretch) {
			stretch_avail += sdata.min_size;
			stretch_ratio_total += c->get_stretch_ratio();
			sdata.stretch_ratio = c->get_stretch_ratio();
		}
		sdata.final_size = sdata.min_size;
		stretch_data.push_back(sdata);
		children_count++;
	}

	if (children_count == 0) {
		return;
	}

	const real_t stretch_max = (vertical ? new_size.height : new_size.width) - (children_count - 1) * theme_cache.separation;
	// Avoid negative stretch space.
	const real_t stretch_diff = MAX(0, stretch_max - stretch_min);

	stretch_avail += stretch_diff; // Available stretch space.

	// Second pass, determine final sizes for stretchable elements.
	// Go through all elements that want to stretch and remove ones without enough space.
	bool has_stretched = false;
	while (stretch_ratio_total > 0) {
		has_stretched = true;
		bool refit_successful = true;

		for (int i = 0; i < (int)stretch_data.size(); i++) {
			if (!stretch_data[i].will_stretch) {
				continue;
			}
			// Wants to stretch.
			const real_t desired_size = stretch_avail * stretch_data[i].stretch_ratio / stretch_ratio_total;

			// Check if it really can stretch.
			if (desired_size < stretch_data[i].min_size) {
				// Available stretching area is too small, remove it and retry.
				stretch_data[i].will_stretch = false;
				stretch_ratio_total -= stretch_data[i].stretch_ratio;
				stretch_avail -= stretch_data[i].min_size;
				stretch_data[i].final_size = stretch_data[i].min_size;
				refit_successful = false;
				break;
			}

			// Can stretch.
			stretch_data[i].final_size = desired_size;
		}

		if (refit_successful) { // Refit went well, break.
			break;
		}
	}

	real_t offset = 0;
	if (!has_stretched) {
		switch (alignment) {
			case ALIGNMENT_BEGIN:
				break;
			case ALIGNMENT_CENTER:
				offset = stretch_diff / 2;
				break;
			case ALIGNMENT_END:
				offset = stretch_diff;
				break;
		}
	}

	// Final pass, fit children to final sizes.
	int idx = 0;
	for (int i = 0; i < get_child_count(); i++) {
		Control *c = as_sortable_control(get_child(i));
		if (!c) {
			continue;
		}

		if (idx != 0) {
			offset += theme_cache.separation;
		}

		const real_t size = stretch_data[idx].final_size;

		Rect2 rect;
		if (vertical) {
			rect = Rect2(0, offset, new_size.width, size);
		} else {
			if (rtl) {
				rect = Rect2(new_size.width - offset - size, 0, size, new_size.height);
			} else {
				rect = Rect2(offset, 0, size, new_size.height);
			}
		}

		fit_child_in_rect(c, rect);

		offset += size;
		idx++;
	}
}

Size2 BoxContainer::get_minimum_size() const {
	Size2 minimum;
	bool first = true;

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = as_sortable_control(get_child(i), SortableVisibilityMode::VISIBLE);
		if (!c) {
			continue;
		}

		const Size2 size = c->get_combined_minimum_size();

		if (vertical) {
			if (size.width > minimum.width) {
				minimum.width = size.width;
			}
			minimum.height += size.height + (first ? 0 : theme_cache.separation);
		} else {
			if (size.height > minimum.height) {
				minimum.height = size.height;
			}
			minimum.width += size.width + (first ? 0 : theme_cache.separation);
		}

		first = false;
	}

	return minimum;
}

void BoxContainer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_SORT_CHILDREN: {
			_resort();
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

void BoxContainer::_validate_property(PropertyInfo &p_property) const {
	if (is_fixed && p_property.name == "vertical") {
		p_property.usage = PROPERTY_USAGE_NONE;
	}
}

void BoxContainer::set_alignment(AlignmentMode p_alignment) {
	if (alignment == p_alignment) {
		return;
	}
	alignment = p_alignment;
	_resort();
}

BoxContainer::AlignmentMode BoxContainer::get_alignment() const {
	return alignment;
}

void BoxContainer::set_vertical(bool p_vertical) {
	ERR_FAIL_COND_MSG(is_fixed, "Can't change orientation of " + get_class() + ".");
	vertical = p_vertical;
	update_minimum_size();
	_resort();
}

bool BoxContainer::is_vertical() const {
	return vertical;
}

Control *BoxContainer::add_spacer(bool p_begin) {
	Control *c = memnew(Control);
	c->set_mouse_filter(MOUSE_FILTER_PASS); //allow spacer to pass mouse events

	if (vertical) {
		c->set_v_size_flags(SIZE_EXPAND_FILL);
	} else {
		c->set_h_size_flags(SIZE_EXPAND_FILL);
	}

	add_child(c);
	if (p_begin) {
		move_child(c, 0);
	}

	return c;
}

Vector<int> BoxContainer::get_allowed_size_flags_horizontal() const {
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

Vector<int> BoxContainer::get_allowed_size_flags_vertical() const {
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

BoxContainer::BoxContainer(bool p_vertical) {
	vertical = p_vertical;
}

void BoxContainer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_spacer", "begin"), &BoxContainer::add_spacer);
	ClassDB::bind_method(D_METHOD("set_alignment", "alignment"), &BoxContainer::set_alignment);
	ClassDB::bind_method(D_METHOD("get_alignment"), &BoxContainer::get_alignment);
	ClassDB::bind_method(D_METHOD("set_vertical", "vertical"), &BoxContainer::set_vertical);
	ClassDB::bind_method(D_METHOD("is_vertical"), &BoxContainer::is_vertical);

	BIND_ENUM_CONSTANT(ALIGNMENT_BEGIN);
	BIND_ENUM_CONSTANT(ALIGNMENT_CENTER);
	BIND_ENUM_CONSTANT(ALIGNMENT_END);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "alignment", PROPERTY_HINT_ENUM, "Begin,Center,End"), "set_alignment", "get_alignment");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "vertical"), "set_vertical", "is_vertical");

	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, BoxContainer, separation);
}

MarginContainer *VBoxContainer::add_margin_child(const String &p_label, Control *p_control, bool p_expand) {
	Label *l = memnew(Label);
	l->set_theme_type_variation("HeaderSmall");
	l->set_text(p_label);
	add_child(l);
	MarginContainer *mc = memnew(MarginContainer);
	mc->add_child(p_control, true);
	add_child(mc);
	if (p_expand) {
		mc->set_v_size_flags(SIZE_EXPAND_FILL);
	}
	p_control->set_accessibility_name(p_label);

	return mc;
}
