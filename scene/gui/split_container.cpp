/**************************************************************************/
/*  split_container.cpp                                                   */
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

#include "split_container.h"

#include "scene/gui/label.h"
#include "scene/gui/margin_container.h"
#include "scene/theme/theme_db.h"

void SplitContainerDragger::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	SplitContainer *sc = Object::cast_to<SplitContainer>(get_parent());

	bool at_least_two_children = sc->_get_sortable_child(1);
	if (sc->collapsed || !at_least_two_children || sc->dragger_visibility != SplitContainer::DRAGGER_VISIBLE) {
		return;
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		if (mb->get_button_index() == MouseButton::LEFT) {
			if (mb->is_pressed()) {
				sc->dragging_index = dragger_index;
				sc->_update_dragger_positions(true);
				dragging = true;
				drag_ofs = sc->split_offsets[dragger_index];
				if (sc->vertical) {
					drag_from = get_transform().xform(mb->get_position()).y;
				} else {
					drag_from = get_transform().xform(mb->get_position()).x;
				}
			} else {
				dragging = false;
				sc->dragging_index = -1;
				queue_redraw();
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {
		if (!dragging) {
			return;
		}

		Vector2i in_parent_pos = get_transform().xform(mm->get_position());
		if (!sc->vertical && is_layout_rtl()) {
			sc->split_offsets.write[dragger_index] = drag_ofs - (in_parent_pos.x - drag_from);
		} else {
			sc->split_offsets.write[dragger_index] = drag_ofs + ((sc->vertical ? in_parent_pos.y : in_parent_pos.x) - drag_from);
		}
		sc->_update_dragger_positions(true);
		sc->queue_sort();
		sc->emit_signal(SNAME("dragged"), sc->split_offsets[dragger_index]);
	}
}

Control::CursorShape SplitContainerDragger::get_cursor_shape(const Point2 &p_pos) const {
	SplitContainer *sc = Object::cast_to<SplitContainer>(get_parent());

	if (!sc->collapsed && sc->dragger_visibility == SplitContainer::DRAGGER_VISIBLE) {
		return (sc->vertical ? CURSOR_VSPLIT : CURSOR_HSPLIT);
	}

	return Control::get_cursor_shape(p_pos);
}

void SplitContainerDragger::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_MOUSE_ENTER: {
			mouse_inside = true;
			SplitContainer *sc = Object::cast_to<SplitContainer>(get_parent());
			if (sc->theme_cache.autohide) {
				queue_redraw();
			}
		} break;

		case NOTIFICATION_MOUSE_EXIT: {
			mouse_inside = false;
			SplitContainer *sc = Object::cast_to<SplitContainer>(get_parent());
			if (sc->theme_cache.autohide) {
				queue_redraw();
			}
		} break;

		case NOTIFICATION_FOCUS_EXIT: {
			if (dragging) {
				dragging = false;
				queue_redraw();
			}
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (dragging && !is_visible_in_tree()) {
				dragging = false;
			}
		} break;

		case NOTIFICATION_DRAW: {
			SplitContainer *sc = Object::cast_to<SplitContainer>(get_parent());
			if (!dragging && !mouse_inside && sc->theme_cache.autohide) {
				return;
			}

			Ref<Texture2D> tex = sc->_get_grabber_icon();
			draw_texture(tex, (get_size() - tex->get_size()) / 2);
		} break;
	}
}

Control *SplitContainer::_get_sortable_child(int p_idx) const {
	int idx = 0;

	for (int i = 0; i < get_child_count(false); i++) {
		Control *child = as_sortable_control(get_child(i, false), SortableVisbilityMode::VISIBLE);
		if (!child) {
			continue;
		}

		if (idx == p_idx) {
			return child;
		}

		idx++;
		if (idx > p_idx) {
			break;
		}
	}

	return nullptr;
}

Ref<Texture2D> SplitContainer::_get_grabber_icon() const {
	if (is_fixed) {
		return theme_cache.grabber_icon;
	} else {
		if (vertical) {
			return theme_cache.grabber_icon_v;
		} else {
			return theme_cache.grabber_icon_h;
		}
	}
}

Point2i SplitContainer::_get_valid_range(int p_dragger_index) {
	ERR_FAIL_INDEX_V(p_dragger_index, (int)dragger_positions.size(), Point2i());
	int axis = vertical ? 1 : 0;
	Ref<Texture2D> g = _get_grabber_icon();
	int sep = (dragger_visibility != DRAGGER_HIDDEN_COLLAPSED) ? MAX(theme_cache.separation, vertical ? g->get_height() : g->get_width()) : 0;

	// Sum the minimum sizes on the left and right sides of the dragger.
	Point2i position_range = Point2i(0, get_size()[axis]);
	position_range.x += sep * p_dragger_index;
	position_range.y -= sep * (dragger_positions.size() - p_dragger_index);

	for (int i = 0; i < (int)dragger_positions.size() + 1; i++) {
		Control *child = _get_sortable_child(i);
		ERR_FAIL_NULL_V(child, Point2i());
		if (i <= p_dragger_index) {
			position_range.x += child->get_combined_minimum_size()[axis];
		} else if (i > p_dragger_index) {
			position_range.y -= child->get_combined_minimum_size()[axis];
		}
	}
	return position_range;
}

struct _StretchData {
	int min_size = 0;
	float stretch_ratio = 0.0;
	int final_size = 0;
	bool expand_flag = false;
};

void SplitContainer::_update_default_dragger_positions() {
	Ref<Texture2D> g = _get_grabber_icon();
	int sep = (dragger_visibility != DRAGGER_HIDDEN_COLLAPSED) ? MAX(theme_cache.separation, vertical ? g->get_height() : g->get_width()) : 0;
	int axis = vertical ? 1 : 0;
	float size = get_size()[axis];

	float total_min_size = get_minimum_size()[axis];
	float free_space = MAX(0.0, size - total_min_size);
	float stretchable_space = free_space;

	// First pass, determine the total stretch amount.
	int expand_count = 0;
	float stretch_total = 0;
	LocalVector<_StretchData> stretch_data;
	for (int i = 0; i < get_child_count(false); i++) {
		Control *child = as_sortable_control(get_child(i, false), SortableVisbilityMode::VISIBLE);
		if (!child) {
			continue;
		}
		_StretchData sdata;
		sdata.min_size = child->get_combined_minimum_size()[axis];
		sdata.final_size = sdata.min_size;
		if ((vertical ? child->get_v_size_flags() : child->get_h_size_flags()).has_flag(SIZE_EXPAND)) {
			sdata.stretch_ratio = child->get_stretch_ratio();
			stretch_total += sdata.stretch_ratio;
			stretchable_space += sdata.min_size;
			sdata.expand_flag = true;
			expand_count += 1;
		}
		stretch_data.push_back(sdata);
	}

	if ((int)stretch_data.size() <= 1) {
		default_dragger_positions.clear();
		return;
	}
	default_dragger_positions.resize((int)stretch_data.size() - 1);

	if (expand_count == 2 && (int)default_dragger_positions.size() == 1) {
		// Special case when there are 2 expanded children for compatibility.
		float ratio = stretch_data[0].stretch_ratio / stretch_total;
		default_dragger_positions[0] = size * ratio - sep / 2;
		return;
	}

	// Second pass, determine final sizes if stretching.
	bool stretch_complete = stretch_total <= 0.0;
	while (!stretch_complete) {
		// Keep track of accumulated error in pixels.
		float error = 0.0;
		for (int i = 0; i < (int)stretch_data.size(); i++) {
			if (stretch_data[i].stretch_ratio <= 0.0) {
				stretch_data[i].final_size = stretch_data[i].min_size;
				continue;
			}
			// Check if it reaches its minimum size.
			float desired_size = stretch_data[i].stretch_ratio / stretch_total * stretchable_space;
			error += desired_size - (int)desired_size;
			if (desired_size < stretch_data[i].min_size) {
				// Will not be stretched, remove and retry.
				stretch_total -= stretch_data[i].stretch_ratio;
				stretchable_space -= stretch_data[i].min_size;
				stretch_data[i].stretch_ratio = 0.0;
				break;
			} else {
				stretch_data[i].final_size = desired_size;
				// Dump accumulated error if one pixel or more.
				if (error >= 1.0) {
					stretch_data[i].final_size += 1;
					error -= 1;
				}
			}
		}
		stretch_complete = true;
	}

	// Final pass, set the default positions.
	int pos = 0;
	int expands_seen = 0;
	for (int i = 0; i < (int)default_dragger_positions.size(); i++) {
		// Do not add a dragger for the last child.
		pos += stretch_data[i].final_size;
		if (stretch_data[i].expand_flag) {
			expands_seen += 1;
		}
		if (expands_seen == 0) {
			// Before all expand flags.
			default_dragger_positions[i] = 0;
		} else if (expands_seen >= expand_count) {
			// After all expand flags.
			default_dragger_positions[i] = size;
		} else {
			default_dragger_positions[i] = pos;
		}
		pos += sep;
	}
}

void SplitContainer::_update_dragger_positions(bool p_clamp) {
	Ref<Texture2D> g = _get_grabber_icon();
	int sep = (dragger_visibility != DRAGGER_HIDDEN_COLLAPSED) ? MAX(theme_cache.separation, vertical ? g->get_height() : g->get_width()) : 0;
	int axis = vertical ? 1 : 0;
	int size = get_size()[axis];

	dragger_positions.resize(default_dragger_positions.size());
	if (split_offsets.size() < default_dragger_positions.size() || (int)split_offsets.size() < 1) {
		split_offsets.resize_zeroed(MAX(1, (int)default_dragger_positions.size()));
	}

	if (collapsed) {
		for (int i = 0; i < (int)dragger_positions.size(); i++) {
			dragger_positions[i] = default_dragger_positions[i];
			Point2i valid_range = _get_valid_range(i);
			dragger_positions[i] = CLAMP(dragger_positions[i], valid_range.x, valid_range.y);
			if (p_clamp) {
				split_offsets.write[i] = dragger_positions[i] - default_dragger_positions[i];
			}
			if (!vertical && is_layout_rtl()) {
				dragger_positions[i] = size - dragger_positions[i] - sep;
			}
		}
		return;
	}

	// Use split_offsets to find the desired dragger positions.
	for (int i = 0; i < (int)dragger_positions.size(); i++) {
		dragger_positions[i] = default_dragger_positions[i] + split_offsets[i];

		// Clamp the desired position to acceptatble values.
		Point2i valid_range = _get_valid_range(i);
		dragger_positions[i] = CLAMP(dragger_positions[i], valid_range.x, valid_range.y);
	}

	// Prevent overlaps.
	if (dragging_index == -1) {
		// Check each dragger with the one to the right of it.
		for (int i = 0; i < (int)dragger_positions.size() - 1; i++) {
			int check_min_size = _get_sortable_child(i + 1)->get_combined_minimum_size()[axis];
			int push_pos = dragger_positions[i] + sep + check_min_size;
			if (dragger_positions[i + 1] < push_pos) {
				dragger_positions[i + 1] = push_pos;
				Point2i valid_range = _get_valid_range(i);
				dragger_positions[i] = CLAMP(dragger_positions[i], valid_range.x, valid_range.y);
			}
		}
	} else {
		// Prioritize the active dragger.
		int dragging_position = dragger_positions[dragging_index];

		// Push overlapping draggers to the left.
		int accumulated_min_size = _get_sortable_child(dragging_index)->get_combined_minimum_size()[axis];
		for (int i = dragging_index - 1; i >= 0; i--) {
			int push_pos = dragging_position - sep * (dragging_index - i) - accumulated_min_size;
			if (dragger_positions[i] > push_pos) {
				dragger_positions[i] = push_pos;
			}
			accumulated_min_size += _get_sortable_child(i)->get_combined_minimum_size()[axis];
		}

		// Push overlapping draggers to the right.
		accumulated_min_size = 0;
		for (int i = dragging_index + 1; i < (int)dragger_positions.size(); i++) {
			accumulated_min_size += _get_sortable_child(i)->get_combined_minimum_size()[axis];
			int push_pos = dragging_position + sep * (i - dragging_index) + accumulated_min_size;
			if (dragger_positions[i] < push_pos) {
				dragger_positions[i] = push_pos;
			}
		}
	}

	// Clamp the split_offset if requested.
	if (p_clamp) {
		for (int i = 0; i < (int)dragger_positions.size(); i++) {
			split_offsets.write[i] -= default_dragger_positions[i] + split_offsets[i] - dragger_positions[i];
		}
	}

	// Invert if rtl.
	if (!vertical && is_layout_rtl()) {
		for (int i = 0; i < (int)dragger_positions.size(); i++) {
			dragger_positions[i] = size - dragger_positions[i] - sep;
		}
	}
}

void SplitContainer::_resort() {
	if (!is_visible_in_tree()) {
		return;
	}
	if (!_get_sortable_child(1)) {
		Control *child = _get_sortable_child(0);
		if (child) {
			// Only one valid child.
			fit_child_in_rect(child, Rect2(Point2(), get_size()));
		}
		return;
	}

	_update_default_dragger_positions();
	_update_dragger_positions(false);

	Ref<Texture2D> g = _get_grabber_icon();
	int sep = (dragger_visibility != DRAGGER_HIDDEN_COLLAPSED) ? MAX(theme_cache.separation, vertical ? g->get_height() : g->get_width()) : 0;

	int axis = vertical ? 1 : 0;
	Size2i new_size = get_size();
	bool rtl = is_layout_rtl();

	// Move the children.
	for (int i = 0; i < (int)dragger_positions.size() + 1; i++) {
		Control *child = _get_sortable_child(i);
		if (!child) {
			return;
		}
		int start_pos;
		int end_pos;
		if (!vertical && rtl) {
			start_pos = i >= (int)dragger_positions.size() ? 0 : dragger_positions[i] + sep;
			end_pos = i == 0 ? new_size[axis] : dragger_positions[i - 1];
		} else {
			start_pos = i == 0 ? 0 : dragger_positions[i - 1] + sep;
			end_pos = i >= (int)dragger_positions.size() ? new_size[axis] : dragger_positions[i];
		}
		int size = end_pos - start_pos;

		if (vertical) {
			fit_child_in_rect(child, Rect2(Point2(0, start_pos), Size2(new_size.width, size)));
		} else {
			fit_child_in_rect(child, Rect2(Point2(start_pos, 0), Size2(size, new_size.height)));
		}
	}

	_update_draggers();

	// Handle the dragger visibility and position.
	if (dragger_visibility == DRAGGER_VISIBLE && !collapsed) {
		int dragger_ctrl_size = MAX(sep, theme_cache.minimum_grab_thickness);
		ERR_FAIL_COND(dragging_area_controls.size() != dragger_positions.size());
		for (int i = 0; i < (int)dragger_positions.size(); i++) {
			dragging_area_controls[i]->show();

			int drag_pos = dragger_positions[i] - (dragger_ctrl_size - sep) / 2;
			if (vertical) {
				dragging_area_controls[i]->set_rect(Rect2(Point2(0, drag_pos), Size2(new_size.width, dragger_ctrl_size)));
			} else {
				dragging_area_controls[i]->set_rect(Rect2(Point2(drag_pos, 0), Size2(dragger_ctrl_size, new_size.height)));
			}

			dragging_area_controls[i]->queue_redraw();
		}
	} else {
		for (int i = 0; i < (int)dragging_area_controls.size(); i++) {
			dragging_area_controls[i]->hide();
		}
	}
}

void SplitContainer::_update_draggers() {
	int valid_child_count = 0;
	for (int i = 0; i < get_child_count(false); i++) {
		Control *child = as_sortable_control(get_child(i, false), SortableVisbilityMode::VISIBLE);
		if (!child) {
			continue;
		}
		valid_child_count++;
	}

	int dragger_count = valid_child_count - 1;
	int draggers_size_diff = dragger_count - (int)dragging_area_controls.size();
	if (draggers_size_diff > 0) {
		// Add new draggers.
		for (int i = 0; i < draggers_size_diff; i++) {
			SplitContainerDragger *dragger = memnew(SplitContainerDragger);
			dragging_area_controls.push_back(dragger);
			add_child(dragger, false, Node::INTERNAL_MODE_BACK);
			dragger->force_parent_owned();
		}
	} else if (draggers_size_diff < 0) {
		// Remove some draggers.
		for (int i = 0; i < -draggers_size_diff; i++) {
			int remove_at = (int)dragging_area_controls.size() - 1;
			SplitContainerDragger *dragger = dragging_area_controls[remove_at];
			dragging_area_controls.remove_at(remove_at);
			memdelete(dragger);
		}
	}

	// Make sure draggers have the correct index.
	for (int i = 0; i < (int)dragging_area_controls.size(); i++) {
		dragging_area_controls[i]->dragger_index = i;
	}
}

Size2 SplitContainer::get_minimum_size() const {
	Size2i minimum;
	Ref<Texture2D> g = _get_grabber_icon();
	int sep = (dragger_visibility != DRAGGER_HIDDEN_COLLAPSED) ? MAX(theme_cache.separation, vertical ? g->get_height() : g->get_width()) : 0;

	for (int i = 0; i < get_child_count(false); i++) {
		Control *child = _get_sortable_child(i);
		if (!child) {
			break;
		}

		if (i != 0) {
			// Add separator size.
			if (vertical) {
				minimum.height += sep;
			} else {
				minimum.width += sep;
			}
		}

		Size2 ms = child->get_combined_minimum_size();

		if (vertical) {
			minimum.height += ms.height;
			minimum.width = MAX(minimum.width, ms.width);
		} else {
			minimum.width += ms.width;
			minimum.height = MAX(minimum.height, ms.height);
		}
	}

	return minimum;
}

void SplitContainer::_validate_property(PropertyInfo &p_property) const {
	if (is_fixed && p_property.name == "vertical") {
		p_property.usage = PROPERTY_USAGE_NONE;
	}
}

void SplitContainer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_TRANSLATION_CHANGED:
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED: {
			queue_sort();
		} break;

		case NOTIFICATION_SORT_CHILDREN: {
			_resort();
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			update_minimum_size();
		} break;
	}
}

void SplitContainer::add_child_notify(Node *p_child) {
	Container::add_child_notify(p_child);
	if (p_child->get_internal_mode() == INTERNAL_MODE_DISABLED && as_sortable_control(p_child, SortableVisbilityMode::VISIBLE)) {
		// Add the dragger immediately, don't wait until sorting.
		_update_draggers();
	}
}

int SplitContainer::get_split_offset() const {
	return split_offsets[0];
}

void SplitContainer::set_split_offset(int p_offset) {
	if (split_offsets[0] == p_offset) {
		return;
	}

	split_offsets.write[0] = p_offset;
	queue_sort();
}

void SplitContainer::set_split_offsets(const Vector<int> &p_offsets) {
	split_offsets = p_offsets;
	queue_sort();
}

Vector<int> SplitContainer::get_split_offsets() const {
	return split_offsets;
}

void SplitContainer::clamp_split_offset() {
	if (!_get_sortable_child(1)) {
		// Needs at least two children.
		return;
	}

	_update_dragger_positions(true);
	queue_sort();
}

void SplitContainer::set_collapsed(bool p_collapsed) {
	if (collapsed == p_collapsed) {
		return;
	}

	collapsed = p_collapsed;
	queue_sort();
}

void SplitContainer::set_dragger_visibility(DraggerVisibility p_visibility) {
	if (dragger_visibility == p_visibility) {
		return;
	}

	dragger_visibility = p_visibility;
	queue_sort();
}

SplitContainer::DraggerVisibility SplitContainer::get_dragger_visibility() const {
	return dragger_visibility;
}

bool SplitContainer::is_collapsed() const {
	return collapsed;
}

void SplitContainer::set_vertical(bool p_vertical) {
	ERR_FAIL_COND_MSG(is_fixed, "Can't change orientation of " + get_class() + ".");
	vertical = p_vertical;
	update_minimum_size();
	_resort();
}

bool SplitContainer::is_vertical() const {
	return vertical;
}

Vector<int> SplitContainer::get_allowed_size_flags_horizontal() const {
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

Vector<int> SplitContainer::get_allowed_size_flags_vertical() const {
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

void SplitContainer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_split_offset", "offset"), &SplitContainer::set_split_offset);
	ClassDB::bind_method(D_METHOD("get_split_offset"), &SplitContainer::get_split_offset);
	ClassDB::bind_method(D_METHOD("set_split_offsets", "offsets"), &SplitContainer::set_split_offsets);
	ClassDB::bind_method(D_METHOD("get_split_offsets"), &SplitContainer::get_split_offsets);
	ClassDB::bind_method(D_METHOD("clamp_split_offset"), &SplitContainer::clamp_split_offset);

	ClassDB::bind_method(D_METHOD("set_collapsed", "collapsed"), &SplitContainer::set_collapsed);
	ClassDB::bind_method(D_METHOD("is_collapsed"), &SplitContainer::is_collapsed);

	ClassDB::bind_method(D_METHOD("set_dragger_visibility", "mode"), &SplitContainer::set_dragger_visibility);
	ClassDB::bind_method(D_METHOD("get_dragger_visibility"), &SplitContainer::get_dragger_visibility);

	ClassDB::bind_method(D_METHOD("set_vertical", "vertical"), &SplitContainer::set_vertical);
	ClassDB::bind_method(D_METHOD("is_vertical"), &SplitContainer::is_vertical);

	ADD_SIGNAL(MethodInfo("dragged", PropertyInfo(Variant::INT, "offset")));

	ADD_PROPERTY(PropertyInfo(Variant::INT, "split_offset", PROPERTY_HINT_NONE, String(), PROPERTY_USAGE_NO_EDITOR), "set_split_offset", "get_split_offset");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT32_ARRAY, "split_offsets", PROPERTY_HINT_NONE, "suffix:px"), "set_split_offsets", "get_split_offsets");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collapsed"), "set_collapsed", "is_collapsed");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "dragger_visibility", PROPERTY_HINT_ENUM, "Visible,Hidden,Hidden and Collapsed"), "set_dragger_visibility", "get_dragger_visibility");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "vertical"), "set_vertical", "is_vertical");

	BIND_ENUM_CONSTANT(DRAGGER_VISIBLE);
	BIND_ENUM_CONSTANT(DRAGGER_HIDDEN);
	BIND_ENUM_CONSTANT(DRAGGER_HIDDEN_COLLAPSED);

	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, SplitContainer, separation);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, SplitContainer, minimum_grab_thickness);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, SplitContainer, autohide);
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, SplitContainer, grabber_icon, "grabber");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, SplitContainer, grabber_icon_h, "h_grabber");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, SplitContainer, grabber_icon_v, "v_grabber");
}

SplitContainer::SplitContainer(bool p_vertical) {
	vertical = p_vertical;
	split_offsets.push_back(0);

	SplitContainerDragger *dragger = memnew(SplitContainerDragger);
	dragging_area_controls.push_back(dragger);
	add_child(dragger, false, Node::INTERNAL_MODE_BACK);
}
