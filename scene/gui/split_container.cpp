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
#include "split_container.compat.inc"

#include "scene/gui/texture_rect.h"
#include "scene/main/viewport.h"
#include "scene/theme/theme_db.h"

void SplitContainerDragger::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	SplitContainer *sc = Object::cast_to<SplitContainer>(get_parent());

	if (sc->collapsed || sc->valid_children.size() < 2u || !sc->dragging_enabled) {
		return;
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		if (mb->get_button_index() == MouseButton::LEFT) {
			if (mb->is_pressed()) {
				// To match the visual position, clamp on the first split.
				sc->_update_dragger_positions(0);
				dragging = true;
				sc->emit_signal(SNAME("drag_started"));
				start_drag_split_offset = sc->get_split_offset(dragger_index);
				if (sc->vertical) {
					drag_from = (int)get_transform().xform(mb->get_position()).y;
				} else {
					drag_from = (int)get_transform().xform(mb->get_position()).x;
				}
			} else {
				dragging = false;
				queue_redraw();
				sc->emit_signal(SNAME("drag_ended"));
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {
		if (!dragging) {
			return;
		}

		Vector2i in_parent_pos = get_transform().xform(mm->get_position());
		int new_drag_offset;
		if (!sc->vertical && is_layout_rtl()) {
			new_drag_offset = start_drag_split_offset - (in_parent_pos.x - drag_from);
		} else {
			new_drag_offset = start_drag_split_offset + ((sc->vertical ? in_parent_pos.y : in_parent_pos.x) - drag_from);
		}
		sc->set_split_offset(new_drag_offset, dragger_index);
		sc->_update_dragger_positions(dragger_index);
		sc->queue_sort();
		sc->emit_signal(SNAME("dragged"), sc->get_split_offset(dragger_index));
	}
}

Control::CursorShape SplitContainerDragger::get_cursor_shape(const Point2 &p_pos) const {
	SplitContainer *sc = Object::cast_to<SplitContainer>(get_parent());
	if (!sc->collapsed && sc->dragging_enabled) {
		return (sc->vertical ? CURSOR_VSPLIT : CURSOR_HSPLIT);
	}
	return Control::get_cursor_shape(p_pos);
}

void SplitContainerDragger::_accessibility_action_inc(const Variant &p_data) {
	SplitContainer *sc = Object::cast_to<SplitContainer>(get_parent());

	if (sc->collapsed || sc->valid_children.size() < 2u || !sc->dragging_enabled) {
		return;
	}
	sc->set_split_offset(sc->get_split_offset(dragger_index) - 10, dragger_index);
	sc->clamp_split_offset(dragger_index);
}

void SplitContainerDragger::_accessibility_action_dec(const Variant &p_data) {
	SplitContainer *sc = Object::cast_to<SplitContainer>(get_parent());

	if (sc->collapsed || sc->valid_children.size() < 2u || !sc->dragging_enabled) {
		return;
	}
	sc->set_split_offset(sc->get_split_offset(dragger_index) + 10, dragger_index);
	sc->clamp_split_offset(dragger_index);
}

void SplitContainerDragger::_accessibility_action_set_value(const Variant &p_data) {
	SplitContainer *sc = Object::cast_to<SplitContainer>(get_parent());

	if (sc->collapsed || sc->valid_children.size() < 2u || !sc->dragging_enabled) {
		return;
	}
	sc->set_split_offset(p_data, dragger_index);
	sc->clamp_split_offset(dragger_index);
}

void SplitContainerDragger::_touch_dragger_mouse_exited() {
	if (!dragging) {
		SplitContainer *sc = Object::cast_to<SplitContainer>(get_parent());
		touch_dragger->set_modulate(sc->theme_cache.touch_dragger_color);
	}
}

void SplitContainerDragger::_touch_dragger_gui_input(const Ref<InputEvent> &p_event) {
	if (!touch_dragger) {
		return;
	}
	Ref<InputEventMouseMotion> mm = p_event;
	Ref<InputEventMouseButton> mb = p_event;
	SplitContainer *sc = Object::cast_to<SplitContainer>(get_parent());

	if (mb.is_valid() && mb->get_button_index() == MouseButton::LEFT) {
		if (mb->is_pressed()) {
			touch_dragger->set_modulate(sc->theme_cache.touch_dragger_pressed_color);
		} else {
			touch_dragger->set_modulate(sc->theme_cache.touch_dragger_color);
		}
	}

	if (mm.is_valid() && !dragging) {
		touch_dragger->set_modulate(sc->theme_cache.touch_dragger_hover_color);
	}
}

void SplitContainerDragger::set_touch_dragger_enabled(bool p_enabled) {
	if (p_enabled) {
		touch_dragger = memnew(TextureRect);
		update_touch_dragger();
		SplitContainer *sc = Object::cast_to<SplitContainer>(get_parent());
		touch_dragger->set_modulate(sc->theme_cache.touch_dragger_color);
		touch_dragger->connect(SceneStringName(gui_input), callable_mp(this, &SplitContainerDragger::_touch_dragger_gui_input));
		touch_dragger->connect(SceneStringName(mouse_exited), callable_mp(this, &SplitContainerDragger::_touch_dragger_mouse_exited));
		add_child(touch_dragger, false, Node::INTERNAL_MODE_FRONT);
	} else {
		if (touch_dragger) {
			touch_dragger->queue_free();
			touch_dragger = nullptr;
		}
	}
	queue_redraw();
}

void SplitContainerDragger::update_touch_dragger() {
	if (!touch_dragger) {
		return;
	}
	SplitContainer *sc = Object::cast_to<SplitContainer>(get_parent());
	touch_dragger->set_texture(sc->_get_touch_dragger_icon());
	touch_dragger->set_anchors_and_offsets_preset(Control::PRESET_CENTER);
	touch_dragger->set_default_cursor_shape(sc->vertical ? CURSOR_VSPLIT : CURSOR_HSPLIT);
}

void SplitContainerDragger::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ACCESSIBILITY_UPDATE: {
			RID ae = get_accessibility_element();
			ERR_FAIL_COND(ae.is_null());

			DisplayServer::get_singleton()->accessibility_update_set_role(ae, DisplayServer::AccessibilityRole::ROLE_SPLITTER);
			DisplayServer::get_singleton()->accessibility_update_set_name(ae, RTR("Drag to resize"));

			SplitContainer *sc = Object::cast_to<SplitContainer>(get_parent());
			if (sc->collapsed || sc->valid_children.size() < 2u || !sc->dragging_enabled) {
				return;
			}
			sc->clamp_split_offset(dragger_index);
			DisplayServer::get_singleton()->accessibility_update_set_num_value(ae, sc->get_split_offset(dragger_index));

			DisplayServer::get_singleton()->accessibility_update_add_action(ae, DisplayServer::AccessibilityAction::ACTION_DECREMENT, callable_mp(this, &SplitContainerDragger::_accessibility_action_dec));
			DisplayServer::get_singleton()->accessibility_update_add_action(ae, DisplayServer::AccessibilityAction::ACTION_INCREMENT, callable_mp(this, &SplitContainerDragger::_accessibility_action_inc));
			DisplayServer::get_singleton()->accessibility_update_add_action(ae, DisplayServer::AccessibilityAction::ACTION_SET_VALUE, callable_mp(this, &SplitContainerDragger::_accessibility_action_set_value));
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			if (touch_dragger) {
				SplitContainer *sc = Object::cast_to<SplitContainer>(get_parent());
				touch_dragger->set_modulate(sc->theme_cache.touch_dragger_color);
				touch_dragger->set_texture(sc->_get_touch_dragger_icon());
			}
		} break;

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
			draw_style_box(sc->theme_cache.split_bar_background, split_bar_rect);
			if (sc->dragger_visibility == SplitContainer::DRAGGER_VISIBLE && (dragging || mouse_inside || !sc->theme_cache.autohide) && !sc->touch_dragger_enabled) {
				Ref<Texture2D> tex = sc->_get_grabber_icon();
				float available_size = sc->vertical ? (sc->get_size().x - tex->get_size().x) : (sc->get_size().y - tex->get_size().y);
				if (available_size - sc->drag_area_margin_begin - sc->drag_area_margin_end > 0) { // Draw the grabber only if it fits.
					draw_texture(tex, (split_bar_rect.get_position() + (split_bar_rect.get_size() - tex->get_size()) * 0.5));
				}
			}
			if (sc->show_drag_area && Engine::get_singleton()->is_editor_hint()) {
				draw_rect(Rect2(Vector2(0, 0), get_size()), sc->dragging_enabled ? Color(1, 1, 0, 0.3) : Color(1, 0, 0, 0.3));
			}
		} break;
	}
}

SplitContainerDragger::SplitContainerDragger() {
	set_focus_mode(FOCUS_ACCESSIBILITY);
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

Ref<Texture2D> SplitContainer::_get_touch_dragger_icon() const {
	if (is_fixed) {
		return theme_cache.touch_dragger_icon;
	} else {
		if (vertical) {
			return theme_cache.touch_dragger_icon_v;
		} else {
			return theme_cache.touch_dragger_icon_h;
		}
	}
}

int SplitContainer::_get_separation() const {
	if (dragger_visibility == DRAGGER_HIDDEN_COLLAPSED) {
		return 0;
	}

	if (touch_dragger_enabled) {
		return theme_cache.separation;
	}
	// DRAGGER_VISIBLE or DRAGGER_HIDDEN.
	Ref<Texture2D> g = _get_grabber_icon();
	return MAX(theme_cache.separation, vertical ? g->get_height() : g->get_width());
}

Point2i SplitContainer::_get_valid_range(int p_dragger_index) const {
	ERR_FAIL_INDEX_V(p_dragger_index, (int)dragger_positions.size(), Point2i());
	const int axis = vertical ? 1 : 0;
	const int sep = _get_separation();

	// Sum the minimum sizes on the left and right sides of the dragger.
	Point2i position_range = Point2i(0, (int)get_size()[axis]);
	position_range.x += sep * p_dragger_index;
	position_range.y -= sep * ((int)dragger_positions.size() - p_dragger_index);

	for (int i = 0; i < (int)valid_children.size(); i++) {
		Control *child = valid_children[i];
		ERR_FAIL_NULL_V(child, Point2i());
		if (i <= p_dragger_index) {
			position_range.x += (int)child->get_combined_minimum_size()[axis];
		} else if (i > p_dragger_index) {
			position_range.y -= (int)child->get_combined_minimum_size()[axis];
		}
	}
	return position_range;
}

PackedInt32Array SplitContainer::_get_desired_sizes() const {
	ERR_FAIL_COND_V((int)default_dragger_positions.size() != split_offsets.size() || (int)valid_children.size() - 1 != split_offsets.size(), PackedInt32Array());
	PackedInt32Array desired_sizes;
	desired_sizes.resize_uninitialized((int)valid_children.size());

	const int sep = _get_separation();
	const int axis = vertical ? 1 : 0;

	int desired_start_pos = 0;
	for (int i = 0; i < (int)valid_children.size() - 1; i++) {
		const int desired_end_pos = default_dragger_positions[i] + split_offsets[i];
		desired_sizes.write[i] = desired_end_pos - desired_start_pos;
		desired_start_pos = desired_end_pos + sep;
	}
	desired_sizes.write[(int)valid_children.size() - 1] = (int)get_size()[axis] - desired_start_pos;

	return desired_sizes;
}

void SplitContainer::_set_desired_sizes(const PackedInt32Array &p_desired_sizes, int p_priority_index) {
	const int sep = _get_separation();
	const int axis = vertical ? 1 : 0;
	const real_t size = get_size()[axis];

	real_t total_desired_size = 0;
	if (!p_desired_sizes.is_empty()) {
		ERR_FAIL_COND((int)valid_children.size() != p_desired_sizes.size());
		total_desired_size += sep * (p_desired_sizes.size() - 1);
	}

	struct StretchData {
		real_t min_size = 0;
		real_t stretch_ratio = 0.0;
		real_t final_size = 0;
		bool priority = false;
	};

	// First pass, determine the total stretch amount.
	real_t stretch_total = 0;
	LocalVector<StretchData> stretch_data;
	for (int i = 0; i < (int)valid_children.size(); i++) {
		Control *child = valid_children[i];
		StretchData sdata;
		sdata.min_size = child->get_combined_minimum_size()[axis];
		sdata.final_size = MAX(sdata.min_size, p_desired_sizes.is_empty() ? 0 : p_desired_sizes[i]);
		total_desired_size += sdata.final_size;
		sdata.priority = i == p_priority_index;
		// Treat the priority child as not expanded, so it doesn't shrink with other expanded children.
		if (i != p_priority_index && child->get_stretch_ratio() > 0 && (vertical ? child->get_v_size_flags() : child->get_h_size_flags()).has_flag(SIZE_EXPAND)) {
			sdata.stretch_ratio = child->get_stretch_ratio();
			stretch_total += sdata.stretch_ratio;
		}
		stretch_data.push_back(sdata);
	}

	real_t available_space = size - total_desired_size;

	// Grow expanding children.
	if (available_space > 0) {
		const real_t grow_amount = available_space / stretch_total;
		for (StretchData &sdata : stretch_data) {
			if (sdata.stretch_ratio <= 0) {
				continue;
			}
			const real_t prev_size = sdata.final_size;
			sdata.final_size = prev_size + grow_amount * sdata.stretch_ratio;
			const real_t size_diff = prev_size - sdata.final_size;
			available_space += size_diff;
		}
	}

	// Shrink expanding children.
	while (available_space < 0) {
		real_t shrinkable_stretch_ratio = 0.0;
		real_t shrinkable_amount = 0.0;
		for (const StretchData &sdata : stretch_data) {
			if (sdata.stretch_ratio <= 0 || sdata.final_size <= sdata.min_size) {
				continue;
			}
			shrinkable_stretch_ratio += sdata.stretch_ratio;
			shrinkable_amount += sdata.final_size - sdata.min_size;
		}
		if (shrinkable_stretch_ratio == 0) {
			break;
		}

		const real_t shrink_amount = MIN(-available_space, shrinkable_amount) / shrinkable_stretch_ratio;
		if (Math::is_zero_approx(shrink_amount)) {
			break;
		}
		const real_t prev_available_space = available_space;
		for (StretchData &sdata : stretch_data) {
			if (sdata.stretch_ratio <= 0 || sdata.final_size <= sdata.min_size) {
				continue;
			}
			const real_t prev_size = sdata.final_size;
			sdata.final_size = CLAMP(prev_size - shrink_amount * sdata.stretch_ratio, sdata.min_size, sdata.final_size);
			const real_t size_diff = prev_size - sdata.final_size;
			available_space += size_diff;
		}
		if (Math::is_equal_approx(available_space, prev_available_space)) {
			// Shrinking can fail due to values being too small to have an effect but too large for `is_zero_approx`.
			break;
		}
	}

	// Shrink non-expanding children.
	bool skip_priority_child = true;
	while (available_space < 0) {
		// Get largest and target sizes. The target size is the second largest size.
		real_t largest_size = 0;
		real_t target_size = 0;
		int largest_count = 0;
		for (const StretchData &sdata : stretch_data) {
			if (sdata.final_size <= sdata.min_size || (skip_priority_child && sdata.priority)) {
				continue;
			}
			if (sdata.final_size > largest_size) {
				target_size = largest_size;
				largest_size = sdata.final_size;
				largest_count = 1;
			} else if (sdata.final_size == largest_size) {
				largest_count++;
			} else if (sdata.final_size < largest_size && sdata.final_size > target_size) {
				target_size = sdata.final_size;
			}
		}
		if (largest_size <= 0) {
			if (skip_priority_child) {
				// Retry with priority child.
				skip_priority_child = false;
				continue;
			} else {
				// No more children to shrink.
				break;
			}
		}
		// Don't shrink smaller than needed.
		target_size = MAX(target_size, largest_size + available_space / largest_count);
		const real_t prev_available_space = available_space;
		for (StretchData &sdata : stretch_data) {
			if (sdata.final_size <= sdata.min_size || (skip_priority_child && sdata.priority)) {
				continue;
			}
			// Shrink all largest elements.
			if (sdata.final_size == largest_size) {
				sdata.final_size = CLAMP(target_size, sdata.min_size, sdata.final_size);
				const real_t size_diff = largest_size - sdata.final_size;
				available_space += size_diff;
			}
		}
		if (Math::is_zero_approx(available_space) || Math::is_equal_approx(available_space, prev_available_space)) {
			break;
		}
	}

	ERR_FAIL_COND((int)default_dragger_positions.size() != (int)stretch_data.size() - 1);

	// Update the split offsets to match the desired sizes.
	split_offsets.resize(MAX(1, (int)default_dragger_positions.size()));
	int pos = 0;
	real_t error_accumulator = 0.0;
	for (int i = 0; i < (int)default_dragger_positions.size(); i++) {
		int final_size = (int)stretch_data[i].final_size;
		if (final_size == stretch_data[i].final_size) {
			error_accumulator += stretch_data[i].final_size - final_size;
			if (error_accumulator > 1.0) {
				error_accumulator -= 1.0;
				final_size += 1;
			}
		}
		pos += final_size;
		split_offsets.write[i] = pos - default_dragger_positions[i];
		pos += sep;
	}
}

void SplitContainer::_update_default_dragger_positions() {
	if (valid_children.size() <= 1u) {
		default_dragger_positions.clear();
		return;
	}
	default_dragger_positions.resize((int)valid_children.size() - 1);

	const int sep = _get_separation();
	const int axis = vertical ? 1 : 0;
	const int size = (int)get_size()[axis];

	struct StretchData {
		int min_size = 0;
		real_t stretch_ratio = 0.0;
		int final_size = 0;
		bool expand_flag = false;
		bool will_stretch = false;
	};

	// First pass, determine the total stretch amount.
	real_t stretchable_space = size - sep * ((int)valid_children.size() - 1);
	real_t stretch_total = 0;
	int expand_count = 0;
	LocalVector<StretchData> stretch_data;
	for (const Control *child : valid_children) {
		StretchData sdata;
		sdata.min_size = (int)child->get_combined_minimum_size()[axis];
		sdata.final_size = sdata.min_size;
		if ((vertical ? child->get_v_size_flags() : child->get_h_size_flags()).has_flag(SIZE_EXPAND) && child->get_stretch_ratio() > 0) {
			sdata.stretch_ratio = child->get_stretch_ratio();
			stretch_total += sdata.stretch_ratio;
			sdata.expand_flag = true;
			sdata.will_stretch = true;
			expand_count++;
		} else {
			stretchable_space -= sdata.min_size;
		}
		stretch_data.push_back(sdata);
	}

#ifndef DISABLE_DEPRECATED
	if (expand_count == 2 && valid_children.size() == 2u) {
		// Special case when there are 2 expanded children, ignore minimum sizes.
		const real_t ratio = stretch_data[0].stretch_ratio / (stretch_data[0].stretch_ratio + stretch_data[1].stretch_ratio);
		default_dragger_positions[0] = (int)(size * ratio - sep * 0.5);
		return;
	}
#endif // DISABLE_DEPRECATED

	// Determine final sizes if stretching.
	while (stretch_total > 0.0 && stretchable_space > 0.0) {
		bool refit_successful = true;
		// Keep track of accumulated error in pixels.
		float error = 0.0;
		for (StretchData &sdata : stretch_data) {
			if (!sdata.will_stretch) {
				continue;
			}
			// Check if it reaches its minimum size.
			const float desired_stretch_size = sdata.stretch_ratio / stretch_total * stretchable_space;
			error += desired_stretch_size - (int)desired_stretch_size;
			if (desired_stretch_size < sdata.min_size) {
				// Will not be stretched, remove and retry.
				stretch_total -= sdata.stretch_ratio;
				stretchable_space -= sdata.min_size;
				sdata.will_stretch = false;
				sdata.final_size = sdata.min_size;
				refit_successful = false;
				break;
			} else {
				sdata.final_size = (int)desired_stretch_size;
				// Dump accumulated error if one pixel or more.
				if (error >= 1.0) {
					sdata.final_size += 1;
					error -= 1;
				}
			}
		}
		if (refit_successful) {
			break;
		}
	}

	// Set the default positions.
	int pos = 0;
	int expands_seen = 0;
	for (int i = 0; i < (int)default_dragger_positions.size(); i++) {
		pos += stretch_data[i].final_size;
		if (stretch_data[i].expand_flag) {
			expands_seen += 1;
		}
		if (expands_seen == 0) {
			// Before all expand flags.
			default_dragger_positions[i] = 0;
		} else if (expands_seen >= expand_count) {
			// After all expand flags.
			default_dragger_positions[i] = size - sep;
		} else {
			default_dragger_positions[i] = pos;
		}
		pos += sep;
	}
}

void SplitContainer::_update_dragger_positions(int p_clamp_index) {
	if (p_clamp_index != -1) {
		ERR_FAIL_INDEX(p_clamp_index, (int)dragger_positions.size());
	}

	const int sep = _get_separation();
	const int axis = vertical ? 1 : 0;
	const int size = (int)get_size()[axis];

	dragger_positions.resize(default_dragger_positions.size());

	if (split_offsets.size() < (int)default_dragger_positions.size() || split_offsets.is_empty()) {
		split_offsets.resize_initialized(MAX(1, (int)default_dragger_positions.size()));
	}

	if (collapsed) {
		for (int i = 0; i < (int)dragger_positions.size(); i++) {
			dragger_positions[i] = default_dragger_positions[i];
			const Point2i valid_range = _get_valid_range(i);
			dragger_positions[i] = CLAMP(dragger_positions[i], valid_range.x, valid_range.y);
			if (p_clamp_index != -1) {
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
		// Clamp the desired position to acceptable values.
		const Point2i valid_range = _get_valid_range(i);
		dragger_positions[i] = CLAMP(default_dragger_positions[i] + split_offsets[i], valid_range.x, valid_range.y);
	}

	// Prevent overlaps.
	if (p_clamp_index == -1) {
		// Check each dragger with the one to the right of it.
		for (int i = 0; i < (int)dragger_positions.size() - 1; i++) {
			const int check_min_size = (int)valid_children[i + 1]->get_combined_minimum_size()[axis];
			const int push_pos = dragger_positions[i] + sep + check_min_size;
			if (dragger_positions[i + 1] < push_pos) {
				dragger_positions[i + 1] = push_pos;
				const Point2i valid_range = _get_valid_range(i);
				dragger_positions[i] = CLAMP(dragger_positions[i], valid_range.x, valid_range.y);
			}
		}
	} else {
		// Prioritize the active dragger.
		const int dragging_position = dragger_positions[p_clamp_index];

		// Push overlapping draggers to the left.
		int accumulated_min_size = (int)valid_children[p_clamp_index]->get_combined_minimum_size()[axis];
		for (int i = p_clamp_index - 1; i >= 0; i--) {
			const int push_pos = dragging_position - sep * (p_clamp_index - i) - accumulated_min_size;
			if (dragger_positions[i] > push_pos) {
				dragger_positions[i] = push_pos;
			}
			accumulated_min_size += (int)valid_children[i]->get_combined_minimum_size()[axis];
		}

		// Push overlapping draggers to the right.
		accumulated_min_size = 0;
		for (int i = p_clamp_index + 1; i < (int)dragger_positions.size(); i++) {
			accumulated_min_size += (int)valid_children[i]->get_combined_minimum_size()[axis];
			const int push_pos = dragging_position + sep * (i - p_clamp_index) + accumulated_min_size;
			if (dragger_positions[i] < push_pos) {
				dragger_positions[i] = push_pos;
			}
		}
	}

	// Clamp the split_offset if requested.
	if (p_clamp_index != -1) {
		for (int i = 0; i < (int)dragger_positions.size(); i++) {
			split_offsets.write[i] = dragger_positions[i] - default_dragger_positions[i];
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
	if (valid_children.size() < 2u) {
		if (valid_children.size() == 1u) {
			// Only one valid child.
			Control *child = valid_children[0];
			fit_child_in_rect(child, Rect2(Point2(), get_size()));
		}
		for (SplitContainerDragger *dragger : dragging_area_controls) {
			dragger->hide();
		}
		return;
	}
	for (SplitContainerDragger *dragger : dragging_area_controls) {
		dragger->set_visible(!collapsed);
		if (touch_dragger_enabled) {
			dragger->touch_dragger->set_visible(dragging_enabled);
		}
	}

	_update_default_dragger_positions();
	_update_dragger_positions();

	const int sep = _get_separation();
	const int axis = vertical ? 1 : 0;
	const Size2i new_size = get_size();
	const bool rtl = is_layout_rtl();

	// Move the children.
	for (int i = 0; i < (int)valid_children.size(); i++) {
		Control *child = valid_children[i];
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

	// Update dragger positions.
	const int dragger_ctrl_size = MAX(sep, theme_cache.minimum_grab_thickness);
	const float split_bar_offset = (dragger_ctrl_size - sep) * 0.5;

	ERR_FAIL_COND(dragging_area_controls.size() != dragger_positions.size());
	for (int i = 0; i < (int)dragger_positions.size(); i++) {
		dragging_area_controls[i]->set_mouse_filter(dragging_enabled ? MOUSE_FILTER_STOP : MOUSE_FILTER_IGNORE);
		if (vertical) {
			const Rect2 split_bar_rect = Rect2(rtl ? drag_area_margin_end : drag_area_margin_begin, dragger_positions[i], new_size.width - drag_area_margin_begin - drag_area_margin_end, sep);
			dragging_area_controls[i]->set_rect(Rect2(split_bar_rect.position.x, split_bar_rect.position.y - split_bar_offset + drag_area_offset, split_bar_rect.size.x, dragger_ctrl_size));
			dragging_area_controls[i]->split_bar_rect = Rect2(Vector2(0.0, int(split_bar_offset) - drag_area_offset), split_bar_rect.size);
		} else {
			const Rect2 split_bar_rect = Rect2(dragger_positions[i], drag_area_margin_begin, sep, new_size.height - drag_area_margin_begin - drag_area_margin_end);
			dragging_area_controls[i]->set_rect(Rect2(split_bar_rect.position.x - split_bar_offset + drag_area_offset * (rtl ? -1 : 1), split_bar_rect.position.y, dragger_ctrl_size, split_bar_rect.size.y));
			dragging_area_controls[i]->split_bar_rect = Rect2(Vector2(int(split_bar_offset) - drag_area_offset * (rtl ? -1 : 1), 0.0), split_bar_rect.size);
		}
		dragging_area_controls[i]->queue_redraw();
	}
	queue_redraw();
}

void SplitContainer::_update_draggers() {
	const int valid_child_count = (int)valid_children.size();
	const int dragger_count = MAX(valid_child_count - 1, 1);
	const int draggers_size_diff = dragger_count - (int)dragging_area_controls.size();

	// Add new draggers.
	for (int i = 0; i < draggers_size_diff; i++) {
		SplitContainerDragger *dragger = memnew(SplitContainerDragger);
		dragging_area_controls.push_back(dragger);
		add_child(dragger, false, Node::INTERNAL_MODE_BACK);
		if (touch_dragger_enabled) {
			dragger->set_touch_dragger_enabled(true);
		}
	}

	// Remove extra draggers.
	for (int i = 0; i < -draggers_size_diff; i++) {
		const int remove_at = (int)dragging_area_controls.size() - 1;
		SplitContainerDragger *dragger = dragging_area_controls[remove_at];
		dragging_area_controls.remove_at(remove_at);
		// replace_by removes all children, so make sure it is a child before removing.
		if (dragger->get_parent() == this) {
			remove_child(dragger);
		}
		memdelete(dragger);
	}

	// Make sure draggers have the correct index.
	for (int i = 0; i < (int)dragging_area_controls.size(); i++) {
		dragging_area_controls[i]->dragger_index = i;
	}
}

Size2 SplitContainer::get_minimum_size() const {
	const int sep = _get_separation();
	const int axis = vertical ? 1 : 0;
	const int other_axis = vertical ? 0 : 1;

	Size2i minimum;

	if (valid_children.size() >= 2u) {
		minimum[axis] += sep * ((int)valid_children.size() - 1);
	}

	for (const Control *child : valid_children) {
		const Size2 min_size = child->get_combined_minimum_size();
		minimum[axis] += (int)min_size[axis];
		minimum[other_axis] = (int)MAX(minimum[other_axis], min_size[other_axis]);
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
		case NOTIFICATION_POSTINITIALIZE: {
			initialized = true;
		} break;
		case NOTIFICATION_SORT_CHILDREN: {
			_resort();
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			update_minimum_size();
		} break;
		case NOTIFICATION_PREDELETE: {
			valid_children.clear();
			dragging_area_controls.clear();
		} break;
	}
}

void SplitContainer::add_child_notify(Node *p_child) {
	Container::add_child_notify(p_child);

	if (p_child->is_internal()) {
		return;
	}
	Control *child = as_sortable_control(p_child, SortableVisibilityMode::IGNORE);
	if (!child) {
		return;
	}

	child->connect(SceneStringName(visibility_changed), callable_mp(this, &SplitContainer::_on_child_visibility_changed).bind(child));
	if (child->is_visible()) {
		_add_valid_child(child);
	}
}

void SplitContainer::remove_child_notify(Node *p_child) {
	Container::remove_child_notify(p_child);

	if (p_child->is_internal()) {
		return;
	}
	Control *child = as_sortable_control(p_child, SortableVisibilityMode::IGNORE);
	if (!child) {
		return;
	}

	child->disconnect(SceneStringName(visibility_changed), callable_mp(this, &SplitContainer::_on_child_visibility_changed));
	if (child->is_visible()) {
		_remove_valid_child(child);
	}
}

void SplitContainer::move_child_notify(Node *p_child) {
	Container::move_child_notify(p_child);

	Control *moved_child = as_sortable_control(p_child, SortableVisibilityMode::IGNORE);
	const int prev_index = valid_children.find(moved_child);
	if (prev_index == -1) {
		return;
	}

	PackedInt32Array desired_sizes;
	if (initialized && !split_offset_pending && valid_children.size() > 2u && split_offsets.size() == (int)default_dragger_positions.size()) {
		desired_sizes = _get_desired_sizes();
	}

	valid_children.remove_at(prev_index);

	// Get new index.
	int index = 0;
	for (int i = 0; i < get_child_count(false); i++) {
		Control *child = as_sortable_control(get_child(i, false), SortableVisibilityMode::IGNORE);
		if (!child) {
			continue;
		}
		if (child == moved_child) {
			break;
		}
		if (valid_children.has(child)) {
			index++;
		}
	}

	valid_children.insert(index, moved_child);

	if (desired_sizes.is_empty()) {
		return;
	}

	const int prev_desired_size = desired_sizes[prev_index];
	desired_sizes.remove_at(prev_index);
	desired_sizes.insert(index, prev_desired_size);
	_set_desired_sizes(desired_sizes, index);
}

void SplitContainer::_on_child_visibility_changed(Control *p_control) {
	if (p_control->is_visible()) {
		_add_valid_child(p_control);
	} else {
		_remove_valid_child(p_control);
	}
}

void SplitContainer::_add_valid_child(Control *p_control) {
	if (valid_children.has(p_control)) {
		return;
	}

	// Get index to insert.
	bool child_is_valid = false;
	int index = 0;
	for (int i = 0; i < get_child_count(false); i++) {
		Control *child = as_sortable_control(get_child(i, false), SortableVisibilityMode::IGNORE);
		if (!child) {
			continue;
		}
		if (child == p_control) {
			if (child->is_visible()) {
				child_is_valid = true;
			}
			break;
		}
		if (valid_children.has(child)) {
			index++;
		}
	}
	if (!child_is_valid) {
		return;
	}

	PackedInt32Array desired_sizes;
	if (initialized && can_use_desired_sizes && !split_offset_pending && valid_children.size() >= 2u && split_offsets.size() == (int)default_dragger_positions.size()) {
		desired_sizes = _get_desired_sizes();
	}

	valid_children.insert(index, p_control);
	if (!initialized) {
		// If not initialized, the theme cache isn't ready yet so return early.
		return;
	}

	_update_default_dragger_positions();
	queue_sort();

	if (valid_children.size() <= 2u) {
		// Already have first dragger.
		return;
	}

	// Call deferred in case already adding or removing children.
	callable_mp(this, &SplitContainer::_update_draggers).call_deferred();

	if (split_offset_pending && split_offsets.size() == (int)valid_children.size() - 1) {
		split_offset_pending = false;
	}
	if (desired_sizes.is_empty()) {
		return;
	}

	// Use the child's existing size as it's desired size.
	const int axis = vertical ? 1 : 0;
	desired_sizes.insert(index, (int)p_control->get_size()[axis]);
	_set_desired_sizes(desired_sizes, index);
}

void SplitContainer::_remove_valid_child(Control *p_control) {
	const int index = valid_children.find(p_control);
	if (index == -1) {
		return;
	}

	PackedInt32Array desired_sizes;
	if (initialized && !split_offset_pending && valid_children.size() > 2u && split_offsets.size() == (int)default_dragger_positions.size()) {
		desired_sizes = _get_desired_sizes();
	}

	valid_children.remove_at(index);
	if (!initialized) {
		return;
	}
	// Only use desired sizes to change the split offset after the first time a child is removed.
	// This allows adding children to not affect the split offsets when creating.
	can_use_desired_sizes = true;

	_update_default_dragger_positions();
	queue_sort();

	if (valid_children.size() <= 1u) {
		// Don't remove last dragger.
		return;
	}

	// Call deferred in case already adding or removing children.
	callable_mp(this, &SplitContainer::_update_draggers).call_deferred();

	if (split_offset_pending && split_offsets.size() == (int)valid_children.size() - 2) {
		split_offset_pending = false;
	}
	if (desired_sizes.is_empty()) {
		return;
	}

	desired_sizes.remove_at(index);
	_set_desired_sizes(desired_sizes);
}

void SplitContainer::set_split_offset(int p_offset, int p_index) {
	ERR_FAIL_INDEX(p_index, split_offsets.size());
	if (split_offsets[p_index] == p_offset) {
		return;
	}

	split_offsets.write[p_index] = p_offset;
	queue_sort();
}

int SplitContainer::get_split_offset(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, split_offsets.size(), 0);
	return split_offsets[p_index];
}

void SplitContainer::set_split_offsets(const PackedInt32Array &p_offsets) {
	if (split_offsets == p_offsets) {
		return;
	}
	split_offsets = p_offsets;
	split_offset_pending = split_offsets.size() > 1 && (int)valid_children.size() - 1 != split_offsets.size();
	queue_sort();
}

PackedInt32Array SplitContainer::get_split_offsets() const {
	return split_offsets;
}

void SplitContainer::clamp_split_offset(int p_priority_index) {
	ERR_FAIL_INDEX(p_priority_index, split_offsets.size());
	if (valid_children.size() < 2u) {
		// Needs at least two children.
		return;
	}

	_update_dragger_positions(p_priority_index);
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
	if (vertical == p_vertical) {
		return;
	}
	vertical = p_vertical;
	for (SplitContainerDragger *dragger : dragging_area_controls) {
		dragger->update_touch_dragger();
	}

	update_minimum_size();
	_resort();
}

bool SplitContainer::is_vertical() const {
	return vertical;
}

void SplitContainer::set_dragging_enabled(bool p_enabled) {
	if (dragging_enabled == p_enabled) {
		return;
	}
	dragging_enabled = p_enabled;
	if (!dragging_enabled) {
		bool was_dragging = false;
		for (SplitContainerDragger *dragger : dragging_area_controls) {
			was_dragging |= dragger->dragging;
			dragger->dragging = false;
		}
		if (was_dragging) {
			emit_signal(SNAME("drag_ended"));
		}
	}
	if (get_viewport()) {
		get_viewport()->update_mouse_cursor_state();
	}
	_resort();
}

bool SplitContainer::is_dragging_enabled() const {
	return dragging_enabled;
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

void SplitContainer::set_drag_area_margin_begin(int p_margin) {
	if (drag_area_margin_begin == p_margin) {
		return;
	}
	drag_area_margin_begin = p_margin;
	queue_sort();
}

int SplitContainer::get_drag_area_margin_begin() const {
	return drag_area_margin_begin;
}

void SplitContainer::set_drag_area_margin_end(int p_margin) {
	if (drag_area_margin_end == p_margin) {
		return;
	}
	drag_area_margin_end = p_margin;
	queue_sort();
}

int SplitContainer::get_drag_area_margin_end() const {
	return drag_area_margin_end;
}

void SplitContainer::set_drag_area_offset(int p_offset) {
	if (drag_area_offset == p_offset) {
		return;
	}
	drag_area_offset = p_offset;
	queue_sort();
}

int SplitContainer::get_drag_area_offset() const {
	return drag_area_offset;
}

void SplitContainer::set_show_drag_area_enabled(bool p_enabled) {
	show_drag_area = p_enabled;
	for (SplitContainerDragger *dragger : dragging_area_controls) {
		dragger->queue_redraw();
	}
}

bool SplitContainer::is_show_drag_area_enabled() const {
	return show_drag_area;
}

TypedArray<Control> SplitContainer::get_drag_area_controls() {
	TypedArray<Control> controls;
	controls.resize((int)dragging_area_controls.size());
	for (int i = 0; i < (int)dragging_area_controls.size(); i++) {
		controls[i] = dragging_area_controls[i];
	}
	return controls;
}

void SplitContainer::set_touch_dragger_enabled(bool p_enabled) {
	if (touch_dragger_enabled == p_enabled) {
		return;
	}
	touch_dragger_enabled = p_enabled;
	for (SplitContainerDragger *dragger : dragging_area_controls) {
		dragger->set_touch_dragger_enabled(p_enabled);
	}
}

bool SplitContainer::is_touch_dragger_enabled() const {
	return touch_dragger_enabled;
}

void SplitContainer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_split_offsets", "offsets"), &SplitContainer::set_split_offsets);
	ClassDB::bind_method(D_METHOD("get_split_offsets"), &SplitContainer::get_split_offsets);

	ClassDB::bind_method(D_METHOD("clamp_split_offset", "priority_index"), &SplitContainer::clamp_split_offset, DEFVAL(0));

	ClassDB::bind_method(D_METHOD("set_collapsed", "collapsed"), &SplitContainer::set_collapsed);
	ClassDB::bind_method(D_METHOD("is_collapsed"), &SplitContainer::is_collapsed);

	ClassDB::bind_method(D_METHOD("set_dragger_visibility", "mode"), &SplitContainer::set_dragger_visibility);
	ClassDB::bind_method(D_METHOD("get_dragger_visibility"), &SplitContainer::get_dragger_visibility);

	ClassDB::bind_method(D_METHOD("set_vertical", "vertical"), &SplitContainer::set_vertical);
	ClassDB::bind_method(D_METHOD("is_vertical"), &SplitContainer::is_vertical);

	ClassDB::bind_method(D_METHOD("set_dragging_enabled", "dragging_enabled"), &SplitContainer::set_dragging_enabled);
	ClassDB::bind_method(D_METHOD("is_dragging_enabled"), &SplitContainer::is_dragging_enabled);

	ClassDB::bind_method(D_METHOD("set_drag_area_margin_begin", "margin"), &SplitContainer::set_drag_area_margin_begin);
	ClassDB::bind_method(D_METHOD("get_drag_area_margin_begin"), &SplitContainer::get_drag_area_margin_begin);

	ClassDB::bind_method(D_METHOD("set_drag_area_margin_end", "margin"), &SplitContainer::set_drag_area_margin_end);
	ClassDB::bind_method(D_METHOD("get_drag_area_margin_end"), &SplitContainer::get_drag_area_margin_end);

	ClassDB::bind_method(D_METHOD("set_drag_area_offset", "offset"), &SplitContainer::set_drag_area_offset);
	ClassDB::bind_method(D_METHOD("get_drag_area_offset"), &SplitContainer::get_drag_area_offset);

	ClassDB::bind_method(D_METHOD("set_drag_area_highlight_in_editor", "drag_area_highlight_in_editor"), &SplitContainer::set_show_drag_area_enabled);
	ClassDB::bind_method(D_METHOD("is_drag_area_highlight_in_editor_enabled"), &SplitContainer::is_show_drag_area_enabled);

	ClassDB::bind_method(D_METHOD("get_drag_area_controls"), &SplitContainer::get_drag_area_controls);

	ClassDB::bind_method(D_METHOD("set_touch_dragger_enabled", "enabled"), &SplitContainer::set_touch_dragger_enabled);
	ClassDB::bind_method(D_METHOD("is_touch_dragger_enabled"), &SplitContainer::is_touch_dragger_enabled);

	ADD_SIGNAL(MethodInfo("dragged", PropertyInfo(Variant::INT, "offset")));
	ADD_SIGNAL(MethodInfo("drag_started"));
	ADD_SIGNAL(MethodInfo("drag_ended"));

	ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT32_ARRAY, "split_offsets", PROPERTY_HINT_NONE, "suffix:px"), "set_split_offsets", "get_split_offsets");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collapsed"), "set_collapsed", "is_collapsed");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "dragging_enabled"), "set_dragging_enabled", "is_dragging_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "dragger_visibility", PROPERTY_HINT_ENUM, "Visible,Hidden,Hidden and Collapsed"), "set_dragger_visibility", "get_dragger_visibility");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "vertical"), "set_vertical", "is_vertical");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "touch_dragger_enabled"), "set_touch_dragger_enabled", "is_touch_dragger_enabled");

	ADD_GROUP("Drag Area", "drag_area_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "drag_area_margin_begin", PROPERTY_HINT_NONE, "suffix:px"), "set_drag_area_margin_begin", "get_drag_area_margin_begin");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "drag_area_margin_end", PROPERTY_HINT_NONE, "suffix:px"), "set_drag_area_margin_end", "get_drag_area_margin_end");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "drag_area_offset", PROPERTY_HINT_NONE, "suffix:px"), "set_drag_area_offset", "get_drag_area_offset");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "drag_area_highlight_in_editor"), "set_drag_area_highlight_in_editor", "is_drag_area_highlight_in_editor_enabled");

	BIND_ENUM_CONSTANT(DRAGGER_VISIBLE);
	BIND_ENUM_CONSTANT(DRAGGER_HIDDEN);
	BIND_ENUM_CONSTANT(DRAGGER_HIDDEN_COLLAPSED);

	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, SplitContainer, touch_dragger_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, SplitContainer, touch_dragger_pressed_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, SplitContainer, touch_dragger_hover_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, SplitContainer, separation);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, SplitContainer, minimum_grab_thickness);
	BIND_THEME_ITEM_CONSTANT_WITH_HINT(SplitContainer, autohide, PROPERTY_HINT_ENUM, "Disabled,Enabled");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, SplitContainer, touch_dragger_icon, "touch_dragger");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, SplitContainer, touch_dragger_icon_h, "h_touch_dragger");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, SplitContainer, touch_dragger_icon_v, "v_touch_dragger");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, SplitContainer, grabber_icon, "grabber");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, SplitContainer, grabber_icon_h, "h_grabber");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, SplitContainer, grabber_icon_v, "v_grabber");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, SplitContainer, split_bar_background, "split_bar_background");

#ifndef DISABLE_DEPRECATED
	ClassDB::bind_method(D_METHOD("get_drag_area_control"), &SplitContainer::get_drag_area_control);
	ClassDB::bind_method(D_METHOD("set_split_offset", "offset"), &SplitContainer::_set_split_offset_first);
	ClassDB::bind_method(D_METHOD("get_split_offset"), &SplitContainer::_get_split_offset_first);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "split_offset", PROPERTY_HINT_NONE, String(), PROPERTY_USAGE_NO_EDITOR), "set_split_offset", "get_split_offset");
#endif // DISABLE_DEPRECATED
}

SplitContainer::SplitContainer(bool p_vertical) {
	vertical = p_vertical;
	split_offsets.push_back(0);

	SplitContainerDragger *dragger = memnew(SplitContainerDragger);
	dragging_area_controls.push_back(dragger);
	add_child(dragger, false, Node::INTERNAL_MODE_BACK);
}
