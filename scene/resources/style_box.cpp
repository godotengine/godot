/**************************************************************************/
/*  style_box.cpp                                                         */
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

#include "style_box.h"

#include "animation.h"
#include "scene/gui/control.h"
#include "scene/main/canvas_item.h"

Size2 StyleBox::get_minimum_size() const {
	Size2 min_size = Size2(get_margin(SIDE_LEFT) + get_margin(SIDE_RIGHT), get_margin(SIDE_TOP) + get_margin(SIDE_BOTTOM));
	Size2 custom_size;
	GDVIRTUAL_CALL(_get_minimum_size, custom_size);

	if (min_size.x < custom_size.x) {
		min_size.x = custom_size.x;
	}
	if (min_size.y < custom_size.y) {
		min_size.y = custom_size.y;
	}

	return min_size;
}

void StyleBox::set_content_margin(Side p_side, float p_value) {
	ERR_FAIL_INDEX((int)p_side, 4);

	content_margin[p_side] = p_value;
	emit_changed();
}

void StyleBox::set_content_margin_all(float p_value) {
	for (int i = 0; i < 4; i++) {
		content_margin[i] = p_value;
	}
	emit_changed();
}

void StyleBox::set_content_margin_individual(float p_left, float p_top, float p_right, float p_bottom) {
	content_margin[SIDE_LEFT] = p_left;
	content_margin[SIDE_TOP] = p_top;
	content_margin[SIDE_RIGHT] = p_right;
	content_margin[SIDE_BOTTOM] = p_bottom;
	emit_changed();
}

float StyleBox::get_content_margin(Side p_side) const {
	ERR_FAIL_INDEX_V((int)p_side, 4, 0.0);

	return content_margin[p_side];
}

float StyleBox::get_margin(Side p_side) const {
	ERR_FAIL_INDEX_V((int)p_side, 4, 0.0);

	if (content_margin[p_side] < 0) {
		return get_style_margin(p_side);
	} else {
		return content_margin[p_side];
	}
}

Point2 StyleBox::get_offset() const {
	return Point2(get_margin(SIDE_LEFT), get_margin(SIDE_TOP));
}

void StyleBox::begin_animation_group(StringName p_id) {
	animation_id = p_id;
	cached_state = nullptr;
}

void StyleBox::end_animation_group(StringName p_id) {
	Control *node = cast_to<Control>(CanvasItem::get_current_item_drawn());
	if (!node) {
		return;
	}

	if (!p_id.is_empty()) {
		animation_id = p_id;
	}
	ObjectID nid = node->get_instance_id();
	AnimKey key = AnimKey(nid, animation_id);

	if (transition_groups.has(key)) {
		AnimationState &current_state = transition_groups[key];
		uint64_t current_time = OS::get_singleton()->get_ticks_usec();
		current_state.current_time = current_time;
		StyleBox *sb = cast_to<StyleBox>(ObjectDB::get_instance(current_state.box_id));
		if (sb && !current_state.has_drawn_box) {
			bool exit_finished = current_state.is_exiting && current_state.current_time >= current_state.end_time;
			bool exitable = sb->exit_info.duration > 0;

			// If its the first frame of the exit animation,
			// increase the end time by exit duration to trigger the animation.
			if (!current_state.is_exiting) {
				current_state.end_time = current_time + sb->exit_info.duration;
				current_state.is_exiting = true;
			}

			if (!exit_finished && exitable) {
				// This flag is required to skip any draw logic that interferes with detecting exit animations.
				is_drawing_exit = true;
				sb->draw(node->get_canvas_item(), current_state.rect);
				is_drawing_exit = false;
			}
		}
	}

	animation_id = "";
	cached_state = nullptr;
}

void StyleBox::begin_draw(RID p_canvas_item, const Rect2 &p_rect) const {
	current_draw_rect = Rect2(p_rect);

	if (!custom_id.is_empty()) {
		animation_id = custom_id;
	}

	if (animation_id.is_empty()) {
		return;
	}

	Control *node = cast_to<Control>(get_current_item_drawn());
	if (!node) {
		return;
	}

	AnimKey key = AnimKey(node->get_instance_id(), animation_id);
	AnimationState &current_state = transition_groups[key];
	cached_state = &current_state;
	current_state.current_time = OS::get_singleton()->get_ticks_usec();

	// Only set up for normally drawn stylboxes, not boxes drawn during an exit animation,
	// since the exit state depends on these.
	if (!is_drawing_exit) {
		current_state.box_id = get_instance_id();
		current_state.rect = current_draw_rect;
		current_state.has_drawn_box = true;

		// If any stylebox draws to this group, its no longer exitng.
		if (current_state.is_exiting) {
			current_state.end_time = current_state.current_time + normal_info.duration;
			current_state.is_exiting = false;
		}
	}

	// Skip styleboxes with no animation duration
	if (current_state.is_exiting && exit_info.duration <= 0) {
		return;
	}
	if (!current_state.is_exiting && normal_info.duration <= 0) {
		return;
	}

	// Detect if its the first time this animation group is being drawn.
	if (exit_info.duration > 0 && cached_state->values[SNAME("transform_offset")].to == Variant()) {
		// Provide the starting points for the enter animation.
		get_animated_value("transform_offset", exit_transform.offset);
		get_animated_value("transform_scale", exit_transform.scale);
		get_animated_value("transform_rotation", exit_transform.rotation);
		get_animated_value("transform_modulate", exit_transform.modulate);
	}

	// Use default transform for normal state.
	DrawTransform dt;
	if (is_drawing_exit) {
		dt = exit_transform;
	}

	// Animate Transform2D parts individually to avoid weird interpolation.
	dt.offset = get_animated_value("transform_offset", dt.offset);
	dt.scale = get_animated_value("transform_scale", dt.scale);
	dt.rotation = get_animated_value("transform_rotation", dt.rotation);
	dt.pivot = p_rect.position + exit_transform.pivot * p_rect.size;

	Transform2D tf = Transform2D().translated(dt.offset) * Transform2D().translated(dt.pivot) * Transform2D().rotated(dt.rotation) * Transform2D().scaled(dt.scale) * Transform2D().translated(-dt.pivot);
	Color md = get_animated_value("transform_modulate", dt.modulate);

	RenderingServer *vs = RenderingServer::get_singleton();
	if (tf != Transform2D()) {
		vs->canvas_item_add_set_transform(p_canvas_item, tf);
	}
	if (md != Color(1, 1, 1, 1)) {
		vs->canvas_item_add_set_modulate(p_canvas_item, md);
	}
}

void StyleBox::end_draw(RID p_canvas_item, const Rect2 &p_rect) const {
	// Reset any draw transforms or modulates used during an exit or entrance animation.
	if (cached_state) {
		if (cached_state->is_exiting && exit_info.duration <= 0) {
			return;
		}
		if (!cached_state->is_exiting && normal_info.duration <= 0) {
			return;
		}
		RenderingServer *vs = RenderingServer::get_singleton();
		vs->canvas_item_add_set_modulate(p_canvas_item, Color(1, 1, 1, 1));
		vs->canvas_item_add_set_transform(p_canvas_item, Transform2D());
	}
}

Variant StyleBox::get_animated_value(StringName p_name, Variant p_target_value, StringName p_group) {
	Control *node = cast_to<Control>(CanvasItem::get_current_item_drawn());
	if (!node) {
		return p_target_value;
	}

	AnimationState *current_state = cached_state;

	// For external animated values, like font color
	if (!p_group.is_empty()) {
		AnimKey key = AnimKey(node->get_instance_id(), p_group);
		current_state = &transition_groups[key];
	}

	if (current_state == nullptr) {
		return p_target_value;
	}

	AnimationValues &values = current_state->values[p_name];
	current_state->has_drawn_group = true;

	if (p_target_value == Variant()) {
		return values.current;
	}

	// If its the first time, skip interpolation.
	StyleBox *sb = cast_to<StyleBox>(ObjectDB::get_instance(current_state->box_id));
	if (!sb || values.to == Variant()) {
		values.from = p_target_value;
		values.to = p_target_value;
		values.current = p_target_value;
		return p_target_value;
	}

	// For disabling animted rects.
	// This is desirable for nodes that scroll,
	// so styleboxes don't just slide around as you scroll.
	if (!sb->should_animate_rect && p_name == SNAME("rect")) {
		values.to = p_target_value;
		values.current = p_target_value;
		return p_target_value;
	}

	// Skip interpolation if there is not duration.
	AnimationInfo info = current_state->is_exiting ? sb->exit_info : sb->normal_info;
	if (info.duration <= 0) {
		values.to = p_target_value;
		values.current = p_target_value;
		return p_target_value;
	}

	uint64_t current_time = current_state->current_time;

	// Animations are only triggered when the target value has changed.
	if (values.to != p_target_value) {
		values.from = values.current;
		values.to = p_target_value;
		values.start_time = current_time;
		current_state->end_time = MAX(current_state->end_time, current_time + info.duration * 1000000);
		// One time connection to process_frame,
		// so queue_redraw() can be triggered on nodes that can animate.
		if (!redrawer_connected) {
			node->get_tree()->connect("process_frame", callable_mp_static(StyleBox::redraw_animating_nodes));
			redrawer_connected = true;
		}
		// One time connection per node,
		// for setting up and resetting variables each draw frame.
		if (!node->is_stylebox_animator_connected) {
			node->connect("draw", callable_mp_static(StyleBox::setup_animation_frame).bind(node));
			node->is_stylebox_animator_connected = true;
		}
	}
	if (values.from == Variant()) {
		values.from = p_target_value;
	}
	if (values.current == Variant()) {
		values.current = p_target_value;
	}

	double elapsed = (current_time - values.start_time) / 1000000.0;
	if (elapsed < info.duration) {
		Variant result = Animation::interpolate_variant(values.from, values.to, Tween::run_equation(info.transition, info.ease, elapsed, 0.0, 1.0, info.duration));
		values.current = result;
		return result;
	} else {
		values.current = values.to;
		values.from = values.to;
	}

	return p_target_value;
}

void StyleBox::redraw_animating_nodes() {
	for (HashMap<AnimKey, AnimationState>::Iterator it = transition_groups.begin(); it;) {
		Control *node = cast_to<Control>(ObjectDB::get_instance(it->key.first));
		if (node) {
			if (it->value.current_time < it->value.end_time) {
				node->queue_redraw();
			} else if (it->value.end_time > 0) {
				// Draw one extra frame for finished exit animations.
				// Ensures the final state does not draw the final frame of the stylebox at all.
				if (it->value.is_exiting) {
					transition_groups.erase(it->key);
					node->queue_redraw();
				}
			} else {
				// Just a way to detect if an animation JUST finished or any frame after.
				it->value.end_time = 0;
			}
		} else {
			transition_groups.erase(it->key);
		}
		++it;
	}
}

void StyleBox::setup_animation_frame(Control *p_node) {
	for (HashMap<AnimKey, AnimationState>::Iterator it = transition_groups.begin(); it;) {
		Control *node = cast_to<Control>(ObjectDB::get_instance(it->key.first));
		if (node == p_node) {
			// Erase transition groups if they were entirely skipped this frame.
			// It happens when get_animated_value is not called at all for a given animation id.
			// Its for clean-up in nodes like ItemList or TabBar, which may delete or scroll visible items.
			if (!it->value.has_drawn_group) {
				transition_groups.erase(it->key);
			} else {
				it->value.has_drawn_box = false;
				it->value.has_drawn_group = false;
			}
		}
		++it;
	}
}

void StyleBox::draw(RID p_canvas_item, const Rect2 &p_rect) const {
	GDVIRTUAL_CALL(_draw, p_canvas_item, p_rect);
}

Rect2 StyleBox::get_draw_rect(const Rect2 &p_rect) const {
	Rect2 ret;
	if (GDVIRTUAL_CALL(_get_draw_rect, p_rect, ret)) {
		return ret;
	}
	return p_rect;
}

CanvasItem *StyleBox::get_current_item_drawn() const {
	return CanvasItem::get_current_item_drawn();
}

bool StyleBox::test_mask(const Point2 &p_point, const Rect2 &p_rect) const {
	bool ret = true;
	GDVIRTUAL_CALL(_test_mask, p_point, p_rect, ret);
	return ret;
}

void StyleBox::set_transform_pivot(Point2 p_pivot) {
	exit_transform.pivot = p_pivot;
	emit_changed();
}

Point2 StyleBox::get_transform_pivot() {
	return exit_transform.pivot;
}

void StyleBox::set_transform_offset(Point2 p_value) {
	exit_transform.offset = p_value;
	emit_changed();
}

Point2 StyleBox::get_transform_offset() {
	return exit_transform.offset;
}

void StyleBox::set_transform_scale(Size2 p_value) {
	exit_transform.scale = p_value;
	emit_changed();
}

Size2 StyleBox::get_transform_scale() {
	return exit_transform.scale;
}

void StyleBox::set_transform_rotation(real_t p_value) {
	exit_transform.rotation = p_value;
	emit_changed();
}

real_t StyleBox::get_transform_rotation() {
	return exit_transform.rotation;
}

void StyleBox::set_transform_modulate(Color p_value) {
	exit_transform.modulate = p_value;
	emit_changed();
}

Color StyleBox::get_transform_modulate() {
	return exit_transform.modulate;
}

void StyleBox::set_animate_rect(bool p_value) {
	should_animate_rect = p_value;
	emit_changed();
}

bool StyleBox::get_animate_rect() {
	return should_animate_rect;
}

void StyleBox::set_custom_id(StringName p_value) {
	custom_id = p_value;
	emit_changed();
}

StringName StyleBox::get_custom_id() {
	return custom_id;
}

void StyleBox::set_animation_duration(AnimationPhase p_phase, real_t p_value) {
	if (p_phase == PHASE_EXIT) {
		exit_info.duration = p_value;
	} else {
		normal_info.duration = p_value;
	}
	emit_changed();
}

real_t StyleBox::get_animation_duration(AnimationPhase p_phase) {
	if (p_phase == PHASE_EXIT) {
		return exit_info.duration;
	} else {
		return normal_info.duration;
	}
}

void StyleBox::set_animation_ease(AnimationPhase p_phase, Tween::EaseType p_value) {
	if (p_phase == PHASE_EXIT) {
		exit_info.ease = p_value;
	} else {
		normal_info.ease = p_value;
	}
	emit_changed();
}

Tween::EaseType StyleBox::get_animation_ease(AnimationPhase p_phase) {
	if (p_phase == PHASE_EXIT) {
		return exit_info.ease;
	} else {
		return normal_info.ease;
	}
}

void StyleBox::set_animation_transition(AnimationPhase p_phase, Tween::TransitionType p_value) {
	if (p_phase == PHASE_EXIT) {
		exit_info.transition = p_value;
	} else {
		normal_info.transition = p_value;
	}
	emit_changed();
}

Tween::TransitionType StyleBox::get_animation_transition(AnimationPhase p_phase) {
	if (p_phase == PHASE_EXIT) {
		return exit_info.transition;
	} else {
		return normal_info.transition;
	}
}

void StyleBox::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_minimum_size"), &StyleBox::get_minimum_size);

	ClassDB::bind_method(D_METHOD("set_content_margin", "margin", "offset"), &StyleBox::set_content_margin);
	ClassDB::bind_method(D_METHOD("set_content_margin_all", "offset"), &StyleBox::set_content_margin_all);
	ClassDB::bind_method(D_METHOD("get_content_margin", "margin"), &StyleBox::get_content_margin);

	ClassDB::bind_method(D_METHOD("get_margin", "margin"), &StyleBox::get_margin);
	ClassDB::bind_method(D_METHOD("get_offset"), &StyleBox::get_offset);

	ClassDB::bind_method(D_METHOD("set_transform_pivot", "pivot"), &StyleBox::set_transform_pivot);
	ClassDB::bind_method(D_METHOD("get_transform_pivot"), &StyleBox::get_transform_pivot);

	ClassDB::bind_method(D_METHOD("set_transform_offset", "offset"), &StyleBox::set_transform_offset);
	ClassDB::bind_method(D_METHOD("get_transform_offset"), &StyleBox::get_transform_offset);

	ClassDB::bind_method(D_METHOD("set_transform_scale", "scale"), &StyleBox::set_transform_scale);
	ClassDB::bind_method(D_METHOD("get_transform_scale"), &StyleBox::get_transform_scale);

	ClassDB::bind_method(D_METHOD("set_transform_rotation", "rotation"), &StyleBox::set_transform_rotation);
	ClassDB::bind_method(D_METHOD("get_transform_rotation"), &StyleBox::get_transform_rotation);

	ClassDB::bind_method(D_METHOD("set_transform_modulate", "modulate"), &StyleBox::set_transform_modulate);
	ClassDB::bind_method(D_METHOD("get_transform_modulate"), &StyleBox::get_transform_modulate);

	ClassDB::bind_method(D_METHOD("set_animate_rect", "enabled"), &StyleBox::set_animate_rect);
	ClassDB::bind_method(D_METHOD("get_animate_rect"), &StyleBox::get_animate_rect);

	ClassDB::bind_method(D_METHOD("set_custom_id", "id"), &StyleBox::set_custom_id);
	ClassDB::bind_method(D_METHOD("get_custom_id"), &StyleBox::get_custom_id);

	ClassDB::bind_method(D_METHOD("set_animation_duration", "type", "duration"), &StyleBox::set_animation_duration);
	ClassDB::bind_method(D_METHOD("get_animation_duration", "type"), &StyleBox::get_animation_duration);

	ClassDB::bind_method(D_METHOD("set_animation_ease", "type", "ease"), &StyleBox::set_animation_ease);
	ClassDB::bind_method(D_METHOD("get_animation_ease", "type"), &StyleBox::get_animation_ease);

	ClassDB::bind_method(D_METHOD("set_animation_transition", "type", "transition"), &StyleBox::set_animation_transition);
	ClassDB::bind_method(D_METHOD("get_animation_transition", "type"), &StyleBox::get_animation_transition);

	ClassDB::bind_method(D_METHOD("draw", "canvas_item", "rect"), &StyleBox::draw);
	ClassDB::bind_method(D_METHOD("get_current_item_drawn"), &StyleBox::get_current_item_drawn);

	ClassDB::bind_method(D_METHOD("test_mask", "point", "rect"), &StyleBox::test_mask);

	BIND_ENUM_CONSTANT(PHASE_NORMAL);
	BIND_ENUM_CONSTANT(PHASE_EXIT);

	ADD_GROUP("Content Margins", "content_margin_");
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "content_margin_left", PROPERTY_HINT_RANGE, "-1,2048,1,suffix:px"), "set_content_margin", "get_content_margin", SIDE_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "content_margin_top", PROPERTY_HINT_RANGE, "-1,2048,1,suffix:px"), "set_content_margin", "get_content_margin", SIDE_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "content_margin_right", PROPERTY_HINT_RANGE, "-1,2048,1,suffix:px"), "set_content_margin", "get_content_margin", SIDE_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "content_margin_bottom", PROPERTY_HINT_RANGE, "-1,2048,1,suffix:px"), "set_content_margin", "get_content_margin", SIDE_BOTTOM);

	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "animation/duration", PROPERTY_HINT_RANGE, "0,1,.05,or_less,or_greater"), "set_animation_duration", "get_animation_duration", PHASE_NORMAL);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "animation/ease", PROPERTY_HINT_ENUM, "EaseIn,EaseOut,EaseInOut,EaseOutIn,Unset"), "set_animation_ease", "get_animation_ease", PHASE_NORMAL);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "animation/transition", PROPERTY_HINT_ENUM, "Linear,Sine,Quint,Quart,Quad,Expo,Elastic,Cubic,Circ,Bounce,Back,Spring,Unset"), "set_animation_transition", "get_animation_transition", PHASE_NORMAL);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "animation/animate_rect"), "set_animate_rect", "get_animate_rect");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "animation/custom_id"), "set_custom_id", "get_custom_id");

	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "animation/exiting/duration", PROPERTY_HINT_RANGE, "0,1,.05,or_less,or_greater"), "set_animation_duration", "get_animation_duration", PHASE_EXIT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "animation/exiting/ease", PROPERTY_HINT_ENUM, "EaseIn,EaseOut,EaseInOut,EaseOutIn,Unset"), "set_animation_ease", "get_animation_ease", PHASE_EXIT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "animation/exiting/transition", PROPERTY_HINT_ENUM, "Linear,Sine,Quint,Quart,Quad,Expo,Elastic,Cubic,Circ,Bounce,Back,Spring,Unset"), "set_animation_transition", "get_animation_transition", PHASE_EXIT);

	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "animation/exiting/modulate"), "set_transform_modulate", "get_transform_modulate");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "animation/exiting/offset", PROPERTY_HINT_NONE, "px"), "set_transform_offset", "get_transform_offset");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "animation/exiting/scale", PROPERTY_HINT_NONE), "set_transform_scale", "get_transform_scale");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "animation/exiting/rotation", PROPERTY_HINT_RANGE, "-180,180,1,radians_as_degrees"), "set_transform_rotation", "get_transform_rotation");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "animation/exiting/pivot", PROPERTY_HINT_NONE), "set_transform_pivot", "get_transform_pivot");

	GDVIRTUAL_BIND(_draw, "to_canvas_item", "rect")
	GDVIRTUAL_BIND(_get_draw_rect, "rect")
	GDVIRTUAL_BIND(_get_minimum_size)
	GDVIRTUAL_BIND(_test_mask, "point", "rect")
}

StyleBox::StyleBox() {
	for (int i = 0; i < 4; i++) {
		content_margin[i] = -1;
	}
}

void StyleBoxEmpty::_validate_property(PropertyInfo &p_property) const {
	// StyleBoxEmpty can't animate so there's no reason to show animation properties.
	if (p_property.name.begins_with("animation/")) {
		p_property.usage = PROPERTY_USAGE_NONE;
	}
}
