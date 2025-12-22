/**************************************************************************/
/*  style_box.h                                                           */
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

#pragma once

#include "core/io/resource.h"
#include "core/object/gdvirtual.gen.inc"
#include "scene/animation/tween.h"

class CanvasItem;
class Control;

class StyleBox : public Resource {
	GDCLASS(StyleBox, Resource);
	RES_BASE_EXTENSION("stylebox");
	OBJ_SAVE_TYPE(StyleBox);

	float content_margin[4];

	using AnimKey = Pair<ObjectID, StringName>;

	struct AnimationInfo {
		double duration = 0;
		Tween::EaseType ease = Tween::EASE_OUT;
		Tween::TransitionType transition = Tween::TRANS_SINE;
	};

	struct DrawTransform {
		Point2 pivot = Point2(.5, .5);
		Point2 offset = Point2();
		Size2 scale = Size2(1, 1);
		real_t rotation = 0.0;
		Color modulate = Color(1, 1, 1, 1);
	};

	struct AnimationValues {
		Variant from;
		Variant to;
		Variant current;
		real_t start_time = 0;
	};

	struct AnimationState {
		//Ref<StyleBox> box = nullptr;
		ObjectID box_id;
		Rect2 rect;
		DrawTransform draw_transform;
		bool has_drawn_group = false;
		bool has_drawn_box = false;
		bool is_exiting = false;

		uint64_t end_time = 0;
		uint64_t current_time = 0;

		HashMap<StringName, AnimationValues> values = {};
	};

	inline static HashMap<AnimKey, AnimationState> transition_groups;
	inline static StringName animation_id;
	inline static Rect2 current_draw_rect;
	inline static bool redrawer_connected = false;
	inline static AnimationState *cached_state = nullptr;
	inline static bool is_drawing_exit = false;

	AnimationInfo normal_info;
	AnimationInfo exit_info;
	DrawTransform exit_transform;
	bool should_animate_rect = true;
	StringName custom_id;

	static void redraw_animating_nodes();
	static void setup_animation_frame(Control *p_node);

protected:
	static void _bind_methods();
	virtual float get_style_margin(Side p_side) const { return 0; }

	void begin_draw(RID p_canvas_item, const Rect2 &p_rect) const;
	void end_draw(RID p_canvas_item, const Rect2 &p_rect) const;

	GDVIRTUAL2C_REQUIRED(_draw, RID, Rect2)
	GDVIRTUAL1RC(Rect2, _get_draw_rect, Rect2)
	GDVIRTUAL0RC(Size2, _get_minimum_size)
	GDVIRTUAL2RC(bool, _test_mask, Point2, Rect2)

public:
	enum AnimationPhase {
		PHASE_NORMAL,
		PHASE_EXIT,
		PHASE_MAX
	};

	static void begin_animation_group(StringName p_id);
	static void end_animation_group(StringName p_id = StringName());
	static Variant get_animated_value(StringName p_name, Variant p_target_value, StringName p_group = StringName());

	void set_transform_pivot(Point2 p_value);
	Point2 get_transform_pivot();

	void set_transform_offset(Point2 p_value);
	Point2 get_transform_offset();

	void set_transform_scale(Size2 p_value);
	Size2 get_transform_scale();

	void set_transform_rotation(real_t p_value);
	real_t get_transform_rotation();

	void set_transform_modulate(Color p_value);
	Color get_transform_modulate();

	void set_animation_duration(AnimationPhase p_type, real_t p_value);
	real_t get_animation_duration(AnimationPhase p_type);

	void set_animation_ease(AnimationPhase p_type, Tween::EaseType p_value);
	Tween::EaseType get_animation_ease(AnimationPhase p_type);

	void set_animation_transition(AnimationPhase p_type, Tween::TransitionType p_value);
	Tween::TransitionType get_animation_transition(AnimationPhase p_type);

	void set_animate_rect(bool p_value);
	bool get_animate_rect();

	void set_custom_id(StringName p_value);
	StringName get_custom_id();

	virtual Size2 get_minimum_size() const;

	void set_content_margin(Side p_side, float p_value);
	void set_content_margin_all(float p_value);
	void set_content_margin_individual(float p_left, float p_top, float p_right, float p_bottom);
	float get_content_margin(Side p_side) const;

	float get_margin(Side p_side) const;
	Point2 get_offset() const;

	virtual void draw(RID p_canvas_item, const Rect2 &p_rect) const;
	virtual Rect2 get_draw_rect(const Rect2 &p_rect) const;

	CanvasItem *get_current_item_drawn() const;

	virtual bool test_mask(const Point2 &p_point, const Rect2 &p_rect) const;

	StyleBox();
};

class StyleBoxEmpty : public StyleBox {
	GDCLASS(StyleBoxEmpty, StyleBox);
	virtual float get_style_margin(Side p_side) const override { return 0; }

public:
	virtual void draw(RID p_canvas_item, const Rect2 &p_rect) const override {}
	void _validate_property(PropertyInfo &p_property) const;
};

VARIANT_ENUM_CAST(StyleBox::AnimationPhase);
