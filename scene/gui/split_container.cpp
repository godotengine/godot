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

	if (sc->collapsed || !sc->get_containable_child(0) || !sc->get_containable_child(1) || sc->dragger_visibility != SplitContainer::DRAGGER_VISIBLE) {
		return;
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		if (mb->get_button_index() == MouseButton::LEFT) {
			if (mb->is_pressed()) {
				Control *first = sc->get_containable_child(0);
				Control *second = sc->get_containable_child(1);
				if (!first || !second) {
					return;
				}
				bool first_expanded = (sc->vertical ? first->get_v_size_flags() : first->get_h_size_flags()) & SIZE_EXPAND;
				bool second_expanded = (sc->vertical ? second->get_v_size_flags() : second->get_h_size_flags()) & SIZE_EXPAND;
				float ratio = first->get_stretch_ratio() / (first->get_stretch_ratio() + second->get_stretch_ratio());
				if (first_expanded && second_expanded) {
					expand_multiplier = ratio;
				} else if (first_expanded) {
					expand_multiplier = 1;
				} else {
					expand_multiplier = 0;
				}
				if (!sc->vertical && is_layout_rtl()) {
					expand_multiplier = 1 - expand_multiplier;
				}

				sc->_on_drag_start();
				dragging = true;
				drag_ofs = sc->split_offset;
				drag_from = get_transform().xform(mb->get_position())[sc->vertical ? 1 : 0];
				drag_from -= sc->get_size()[sc->vertical ? 1 : 0] * expand_multiplier;
			} else {
				dragging = false;
				queue_redraw();
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {
		if (!dragging) {
			return;
		}

		int drag_to = get_transform().xform(mm->get_position())[sc->vertical ? 1 : 0];
		drag_to -= sc->get_size()[sc->vertical ? 1 : 0] * expand_multiplier;
		int drag_delta = drag_to - drag_from;
		if (!sc->vertical && is_layout_rtl()) {
			drag_delta *= -1;
		}
		sc->split_offset = drag_ofs + drag_delta;
		sc->_compute_middle_sep(true, true);
		sc->queue_sort();
		sc->emit_signal(SNAME("dragged"), sc->get_split_offset());
	}
}

Control::CursorShape SplitContainerDragger::get_cursor_shape(const Point2 &p_pos) const {
	SplitContainer *sc = Object::cast_to<SplitContainer>(get_parent());

	if (!sc->collapsed && sc->dragger_visibility == SplitContainer::DRAGGER_VISIBLE) {
		return (sc->vertical ? CURSOR_VSPLIT : CURSOR_HSPLIT);
	}

	return Control::get_cursor_shape(p_pos);
}

bool SplitContainerDragger::is_dragging() const {
	return dragging;
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

Control *SplitContainer::get_containable_child(int p_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_idx, 2, nullptr, "SplitContainer can only have 2 splits.");
	int idx = 0;

	for (int i = 0; i < get_child_count(false); i++) {
		Control *c = Object::cast_to<Control>(get_child(i, false));
		if (!c || !c->is_visible()) {
			continue;
		}
		if (c->is_set_as_top_level()) {
			continue;
		}

		if (idx == p_idx) {
			return c;
		}

		idx++;
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

void SplitContainer::_on_drag_start() {
	if (get_containable_child(0) && get_containable_child(1)) {
		_compute_middle_sep(true);
	}
	if (!resize_separately) {
		return;
	}
	// Recursively clamp all children SplitContainers so they can be resized properly.
	for (int i = 0; i < 2; i++) {
		Control *split = get_containable_child(i);
		if (!split) {
			continue;
		}
		SplitContainer *sc = Object::cast_to<SplitContainer>(split);
		if (!sc) {
			continue;
		}
		sc->_on_drag_start();
	}
}

int SplitContainer::_get_separate_combined_minimum_size(bool p_minimize_first_side, int p_axis) const {
	if (!resize_separately) {
		return get_combined_minimum_size()[p_axis];
	}
	int sep = (dragger_visibility != DRAGGER_HIDDEN_COLLAPSED) ? MAX(theme_cache.separation, vertical ? _get_grabber_icon()->get_height() : _get_grabber_icon()->get_width()) : 0;
	int min_size_sum = sep;
	Control *near_split = get_containable_child(p_minimize_first_side ? 0 : 1);
	Control *far_split = get_containable_child(p_minimize_first_side ? 1 : 0);
	if (near_split) {
		// Use the minimum size for the closest split.
		SplitContainer *split_sc = Object::cast_to<SplitContainer>(near_split);
		if (split_sc) {
			min_size_sum += split_sc->_get_separate_combined_minimum_size(p_minimize_first_side, p_axis);
		} else {
			min_size_sum += near_split->get_combined_minimum_size()[p_axis];
		}
	}
	if (far_split) {
		if (vertical == (p_axis == 1)) {
			// Use the full size of the other split to prevent dragging past that point.
			min_size_sum += far_split->get_size()[p_axis];
		} else {
			// If we are not aligned in the same direction, use our children's minimum size.
			SplitContainer *split_sc = Object::cast_to<SplitContainer>(far_split);
			if (split_sc) {
				min_size_sum = MAX(min_size_sum - sep, split_sc->_get_separate_combined_minimum_size(p_minimize_first_side, p_axis));
			} else {
				min_size_sum = MAX(min_size_sum - sep, near_split->get_combined_minimum_size()[p_axis]);
			}
		}
	}
	return min_size_sum;
}

void SplitContainer::_push_parent(int p_delta) {
	if (p_delta == 0) {
		return;
	}
	// Change the split offset of the first valid parent SplitContainer.
	Control *last_parent = this;
	Control *parent;
	while (last_parent) {
		parent = last_parent->get_parent_control();
		if (!parent || !parent->is_inside_tree() || parent->is_set_as_top_level()) {
			break;
		}
		SplitContainer *sc = Object::cast_to<SplitContainer>(parent);
		if (!sc || !sc->is_visible()) {
			break;
		}
		if (sc->get_containable_child(0) && sc->get_containable_child(1) && sc->is_vertical() == vertical) {
			// If we are moving towards the divisor.
			if (sc->get_containable_child(p_delta < 0 ? 1 : 0) == last_parent) {
				// Clamp the parent before trying to push it.
				sc->_compute_middle_sep(true, false);
				sc->split_offset += p_delta;
				sc->_compute_middle_sep(true, true);
				sc->queue_sort();
				break;
			}
		}
		last_parent = parent;
	}
}

void SplitContainer::_adjust_child_split(bool p_adjust_first, bool p_vertical, int p_delta) {
	if (!resize_separately || collapsed || !is_visible()) {
		return;
	}
	Control *first = get_containable_child(0);
	Control *second = get_containable_child(1);
	int first_delta = p_delta;
	int second_delta = p_delta;
	if (vertical == p_vertical && first && second) {
		// Move our split_offset before resizing so our far split stays the same size.
		bool first_expanded = (vertical ? first->get_v_size_flags() : first->get_h_size_flags()) & SIZE_EXPAND;
		bool second_expanded = (vertical ? second->get_v_size_flags() : second->get_h_size_flags()) & SIZE_EXPAND;
		int size = get_size()[vertical ? 1 : 0];
		int targ_split_offset = split_offset;

		if (first_expanded && second_expanded) {
			Ref<Texture2D> g = _get_grabber_icon();
			int sep = (dragger_visibility != DRAGGER_HIDDEN_COLLAPSED) ? MAX(theme_cache.separation, vertical ? g->get_height() : g->get_width()) : 0;
			float ratio = first->get_stretch_ratio() / (first->get_stretch_ratio() + second->get_stretch_ratio());
			if (p_adjust_first) {
				int split_size = Math::round(size * (1 - ratio) - sep / 2);
				targ_split_offset += Math::round((size + p_delta) * (1 - ratio) - sep / 2) - split_size;
				targ_split_offset -= p_delta;
			} else {
				int split_size = Math::round(size * ratio - sep / 2);
				targ_split_offset += Math::round((size + p_delta) * ratio - sep / 2) - split_size;
			}
		} else if (first_expanded) {
			if (!p_adjust_first) {
				targ_split_offset += p_delta;
			}
		} else {
			if (p_adjust_first) {
				targ_split_offset -= p_delta;
			}
		}

		int target_mid_step = middle_sep - p_delta;
		split_offset = targ_split_offset;
		// Clamp the split offset with updated size.
		_compute_middle_sep(true, false, size - p_delta);
		queue_sort();
		int moved_delta = middle_sep - target_mid_step;
		if (moved_delta == 0) {
			// Finished adjusting.
			return;
		}
		if (!vertical && is_layout_rtl()) {
			moved_delta *= -1;
		}
		// Adjustment was clamped, continue to children SplitContainers.
		first_delta = -moved_delta;
		second_delta = -first_delta;
	}

	// Propagate to children SplitContainers.
	if (first) {
		SplitContainer *first_sc = Object::cast_to<SplitContainer>(first);
		if (first_sc) {
			first_sc->_adjust_child_split(p_adjust_first, p_vertical, first_delta);
		}
	}
	if (second) {
		SplitContainer *second_sc = Object::cast_to<SplitContainer>(second);
		if (second_sc) {
			second_sc->_adjust_child_split(p_adjust_first, p_vertical, second_delta);
		}
	}
}

void SplitContainer::_compute_middle_sep(bool p_clamp, bool p_affect_nested, int p_override_size) {
	Control *first = get_containable_child(0);
	Control *second = get_containable_child(1);

	// Determine expanded children.
	bool first_expanded = (vertical ? first->get_v_size_flags() : first->get_h_size_flags()) & SIZE_EXPAND;
	bool second_expanded = (vertical ? second->get_v_size_flags() : second->get_h_size_flags()) & SIZE_EXPAND;

	// Compute the minimum size.
	int axis = vertical ? 1 : 0;
	int size = p_override_size > 0 ? p_override_size : get_size()[axis];
	int ms_first = first->get_combined_minimum_size()[axis];
	int ms_second = second->get_combined_minimum_size()[axis];

	// Determine the separation between items.
	Ref<Texture2D> g = _get_grabber_icon();
	int sep = (dragger_visibility != DRAGGER_HIDDEN_COLLAPSED) ? MAX(theme_cache.separation, vertical ? g->get_height() : g->get_width()) : 0;

	// Compute the wished separation_point.
	int wished_middle_sep = 0;
	int split_offset_with_collapse = 0;
	if (!collapsed) {
		split_offset_with_collapse = split_offset;
	}
	if (first_expanded && second_expanded) {
		float ratio = first->get_stretch_ratio() / (first->get_stretch_ratio() + second->get_stretch_ratio());
		wished_middle_sep = size * ratio - sep / 2 + split_offset_with_collapse;
	} else if (first_expanded) {
		wished_middle_sep = size - sep + split_offset_with_collapse;
	} else {
		wished_middle_sep = split_offset_with_collapse;
	}

	if (resize_separately && !push_nested && dragging_area_control->is_dragging()) {
		// Prevent dragging past child's split.
		bool moving_towards_first = wished_middle_sep < middle_sep;
		if (!vertical && is_layout_rtl()) {
			moving_towards_first = wished_middle_sep < size - middle_sep - sep;
		}
		if (moving_towards_first) {
			SplitContainer *first_sc = Object::cast_to<SplitContainer>(first);
			if (first_sc) {
				ms_first = MAX(ms_first, first_sc->_get_separate_combined_minimum_size(false, axis));
			}
		} else {
			SplitContainer *second_sc = Object::cast_to<SplitContainer>(second);
			if (second_sc) {
				ms_second = MAX(ms_second, second_sc->_get_separate_combined_minimum_size(true, axis));
			}
		}
	}
	int last_sep = middle_sep;

	// Clamp the middle sep to acceptatble values.
	int clamped_middle_sep = CLAMP(wished_middle_sep, ms_first, size - sep - ms_second);
	middle_sep = clamped_middle_sep;
	if (!vertical && is_layout_rtl()) {
		middle_sep = size - clamped_middle_sep - sep;
	}

	// Clamp the split_offset if requested.
	if (p_clamp) {
		int sep_delta = wished_middle_sep - clamped_middle_sep;
		split_offset -= sep_delta;

		if (resize_separately && p_affect_nested) {
			_resort();
			// Adjust child split offsets so that only the nearest splits change size.
			int moved_delta = middle_sep - last_sep;
			if (!vertical && is_layout_rtl()) {
				moved_delta *= -1;
			}
			if (moved_delta != 0) {
				SplitContainer *first_sc = Object::cast_to<SplitContainer>(first);
				if (first_sc) {
					first_sc->_adjust_child_split(false, vertical, -moved_delta);
				}
				SplitContainer *second_sc = Object::cast_to<SplitContainer>(second);
				if (second_sc) {
					second_sc->_adjust_child_split(true, vertical, moved_delta);
				}
			}
		}
		if (push_nested && p_affect_nested) {
			_push_parent(sep_delta);
		}
	}
}

void SplitContainer::_resort() {
	Control *first = get_containable_child(0);
	Control *second = get_containable_child(1);

	// If we have only one element.
	if (!first || !second) {
		if (first) {
			fit_child_in_rect(first, Rect2(Point2(), get_size()));
		} else if (second) {
			fit_child_in_rect(second, Rect2(Point2(), get_size()));
		}
		dragging_area_control->hide();
		return;
	}

	// If we have more that one.
	_compute_middle_sep(false);

	// Determine the separation between items.
	Ref<Texture2D> g = _get_grabber_icon();
	int sep = (dragger_visibility != DRAGGER_HIDDEN_COLLAPSED) ? MAX(theme_cache.separation, vertical ? g->get_height() : g->get_width()) : 0;

	// Move the children, including the dragger.
	if (vertical) {
		fit_child_in_rect(first, Rect2(Point2(0, 0), Size2(get_size().width, middle_sep)));
		int sofs = middle_sep + sep;
		fit_child_in_rect(second, Rect2(Point2(0, sofs), Size2(get_size().width, get_size().height - sofs)));
	} else {
		if (is_layout_rtl()) {
			fit_child_in_rect(second, Rect2(Point2(0, 0), Size2(middle_sep, get_size().height)));
			int sofs = middle_sep + sep;
			fit_child_in_rect(first, Rect2(Point2(sofs, 0), Size2(get_size().width - sofs, get_size().height)));
		} else {
			fit_child_in_rect(first, Rect2(Point2(0, 0), Size2(middle_sep, get_size().height)));
			int sofs = middle_sep + sep;
			fit_child_in_rect(second, Rect2(Point2(sofs, 0), Size2(get_size().width - sofs, get_size().height)));
		}
	}

	// Handle the dragger visibility and position.
	if (dragger_visibility == DRAGGER_VISIBLE && !collapsed) {
		dragging_area_control->show();

		int dragger_ctrl_size = MAX(sep, theme_cache.minimum_grab_thickness);
		if (vertical) {
			dragging_area_control->set_rect(Rect2(Point2(0, middle_sep - (dragger_ctrl_size - sep) / 2), Size2(get_size().width, dragger_ctrl_size)));
		} else {
			dragging_area_control->set_rect(Rect2(Point2(middle_sep - (dragger_ctrl_size - sep) / 2, 0), Size2(dragger_ctrl_size, get_size().height)));
		}

		dragging_area_control->queue_redraw();
	} else {
		dragging_area_control->hide();
	}
}

Size2 SplitContainer::get_minimum_size() const {
	Size2i minimum;
	Ref<Texture2D> g = _get_grabber_icon();
	int sep = (dragger_visibility != DRAGGER_HIDDEN_COLLAPSED) ? MAX(theme_cache.separation, vertical ? g->get_height() : g->get_width()) : 0;

	for (int i = 0; i < 2; i++) {
		if (!get_containable_child(i)) {
			break;
		}

		if (i == 1) {
			if (vertical) {
				minimum.height += sep;
			} else {
				minimum.width += sep;
			}
		}

		Size2 ms = get_containable_child(i)->get_combined_minimum_size();

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

void SplitContainer::set_split_offset(int p_offset) {
	if (split_offset == p_offset) {
		return;
	}

	split_offset = p_offset;

	queue_sort();
}

int SplitContainer::get_split_offset() const {
	return split_offset;
}

void SplitContainer::clamp_split_offset() {
	if (!get_containable_child(0) || !get_containable_child(1)) {
		return;
	}

	_compute_middle_sep(true);
	queue_sort();
}

void SplitContainer::set_collapsed(bool p_collapsed) {
	if (collapsed == p_collapsed) {
		return;
	}

	collapsed = p_collapsed;
	queue_sort();
}

bool SplitContainer::is_collapsed() const {
	return collapsed;
}

void SplitContainer::set_push_nested(bool p_push_nested) {
	if (push_nested == p_push_nested) {
		return;
	}

	push_nested = p_push_nested;
	queue_sort();
}

bool SplitContainer::is_pushing_nested() const {
	return push_nested;
}

void SplitContainer::set_resize_separately(bool p_resize_separately) {
	if (resize_separately == p_resize_separately) {
		return;
	}

	resize_separately = p_resize_separately;
	queue_sort();
}

bool SplitContainer::is_resizing_separately() const {
	return resize_separately;
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
	ClassDB::bind_method(D_METHOD("clamp_split_offset"), &SplitContainer::clamp_split_offset);

	ClassDB::bind_method(D_METHOD("set_collapsed", "collapsed"), &SplitContainer::set_collapsed);
	ClassDB::bind_method(D_METHOD("is_collapsed"), &SplitContainer::is_collapsed);

	ClassDB::bind_method(D_METHOD("set_dragger_visibility", "mode"), &SplitContainer::set_dragger_visibility);
	ClassDB::bind_method(D_METHOD("get_dragger_visibility"), &SplitContainer::get_dragger_visibility);

	ClassDB::bind_method(D_METHOD("set_vertical", "vertical"), &SplitContainer::set_vertical);
	ClassDB::bind_method(D_METHOD("is_vertical"), &SplitContainer::is_vertical);

	ClassDB::bind_method(D_METHOD("is_pushing_nested"), &SplitContainer::is_pushing_nested);
	ClassDB::bind_method(D_METHOD("set_push_nested", "push_nested"), &SplitContainer::set_push_nested);
	ClassDB::bind_method(D_METHOD("is_resizing_separately"), &SplitContainer::is_resizing_separately);
	ClassDB::bind_method(D_METHOD("set_resize_separately", "resize_separately"), &SplitContainer::set_resize_separately);

	ADD_SIGNAL(MethodInfo("dragged", PropertyInfo(Variant::INT, "offset")));

	ADD_PROPERTY(PropertyInfo(Variant::INT, "split_offset", PROPERTY_HINT_NONE, "suffix:px"), "set_split_offset", "get_split_offset");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collapsed"), "set_collapsed", "is_collapsed");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "dragger_visibility", PROPERTY_HINT_ENUM, "Visible,Hidden,Hidden and Collapsed"), "set_dragger_visibility", "get_dragger_visibility");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "vertical"), "set_vertical", "is_vertical");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "push_nested"), "set_push_nested", "is_pushing_nested");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "resize_separately"), "set_resize_separately", "is_resizing_separately");

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

	dragging_area_control = memnew(SplitContainerDragger);
	add_child(dragging_area_control, false, Node::INTERNAL_MODE_BACK);
}
