/**************************************************************************/
/*  size_container.cpp                                                    */
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

#include "size_container.h"

Size2 SizeContainer::get_minimum_size() const {
	Size2 ms;
	Size2 maximum_size = get_custom_maximum_size();
	bool expand_horizontal = false;
	bool expand_vertical = false;
	for (int i = 0; i < get_child_count(); i++) {
		Control *c = as_sortable_control(get_child(i), SortableVisibilityMode::VISIBLE);
		if (!c) {
			continue;
		}
		// Make sure the children are up to date
		c->set_custom_maximum_size(get_custom_maximum_size());

		Size2 minsize = c->get_combined_minimum_size();
		Size2 maxsize = c->get_combined_maximum_size();
		ms = ms.max(minsize).max(maxsize);
		expand_horizontal = expand_horizontal || (c->get_h_size_flags() & SIZE_EXPAND);
		expand_vertical = expand_vertical || (c->get_v_size_flags() & SIZE_EXPAND);
	}

	if (maximum_size.x > 0 && expand_horizontal) {
		ms.x = MAX(ms.x, maximum_size.x);
	}
	if (maximum_size.y > 0 && expand_vertical) {
		ms.y = MAX(ms.y, maximum_size.y);
	}

	return ms;
}

void SizeContainer::add_child_notify(Node *p_child) {
	Container::add_child_notify(p_child);

	Control *control = Object::cast_to<Control>(p_child);
	if (!control) {
		return;
	}

	control->set_custom_maximum_size(get_custom_maximum_size());
	control->connect("size_flags_changed", Callable(this, "_child_size_bounds_changed"));
}

void SizeContainer::remove_child_notify(Node *p_child) {
	Container::remove_child_notify(p_child);

	Control *control = Object::cast_to<Control>(p_child);
	if (!control) {
		return;
	}

	control->set_custom_maximum_size(Size2());
	control->disconnect("size_flags_changed", Callable(this, "_child_size_bounds_changed"));
}

void SizeContainer::_validate_property(PropertyInfo &p_property) const {
	// Remove FULL_RECT and other wide presets from the anchors_preset options
	// since they conflict with the maximum_size constraint.
	if (p_property.name == "anchors_preset") {
		Vector<String> options = p_property.hint_string.split(",");
		String new_hint_string;

		for (int i = 0; i < options.size(); i++) {
			Vector<String> parts = options[i].split(":");
			if (parts.size() == 2) {
				int preset_value = parts[1].to_int();
				if (preset_value != PRESET_LEFT_WIDE &&
						preset_value != PRESET_TOP_WIDE &&
						preset_value != PRESET_RIGHT_WIDE &&
						preset_value != PRESET_BOTTOM_WIDE &&
						preset_value != PRESET_VCENTER_WIDE &&
						preset_value != PRESET_HCENTER_WIDE &&
						preset_value != PRESET_FULL_RECT) {
					if (!new_hint_string.is_empty()) {
						new_hint_string += ",";
					}
					new_hint_string += options[i];
				}
			}
		}

		p_property.hint_string = new_hint_string;
	} else if (p_property.name == "custom_maximum_size") {
		p_property.usage = PROPERTY_USAGE_DEFAULT;
	}
}

void SizeContainer::set_anchors_preset(LayoutPreset p_preset, bool p_keep_offsets) {
	// Prevent setting wide presets that would conflict with maximum_size
	if (p_preset == PRESET_LEFT_WIDE || p_preset == PRESET_TOP_WIDE ||
			p_preset == PRESET_RIGHT_WIDE || p_preset == PRESET_BOTTOM_WIDE ||
			p_preset == PRESET_VCENTER_WIDE || p_preset == PRESET_HCENTER_WIDE ||
			p_preset == PRESET_FULL_RECT) {
		WARN_PRINT("SizeContainer does not support wide anchor presets (including FULL_RECT) as they conflict with maximum_size constraints.");
		return;
	}

	Control::set_anchors_preset(p_preset, p_keep_offsets);
}

void SizeContainer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_SORT_CHILDREN: {
			update_size_bounds();
			Size2 size = get_size();
			Size2 maximum_size = get_custom_maximum_size();

			// Clamp container size to max_size if set
			if (maximum_size.x > 0) {
				size.x = MIN(size.x, maximum_size.x);
			}
			if (maximum_size.y > 0) {
				size.y = MIN(size.y, maximum_size.y);
			}

			for (int i = 0; i < get_child_count(); i++) {
				Control *c = as_sortable_control(get_child(i));
				if (!c) {
					continue;
				}

				// Fit child within the constrained size
				fit_child_in_rect(c, Rect2(Point2(), size));
			}
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			update_size_bounds();
		} break;
	}
}

void SizeContainer::_bind_methods() {
}

SizeContainer::SizeContainer() {
	set_clip_contents(true);
}
