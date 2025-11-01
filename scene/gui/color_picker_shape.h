/**************************************************************************/
/*  color_picker_shape.h                                                  */
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

#include "scene/gui/color_picker.h"

class ColorPickerShape : public Object {
	GDCLASS(ColorPickerShape, Object);

	void _emit_color_changed();

protected:
	static inline Ref<Shader> wheel_shader;
	static inline Ref<Shader> circle_shader;
	static inline Ref<Shader> circle_ok_color_shader;
	static inline Ref<Shader> rectangle_ok_color_hs_shader;
	static inline Ref<Shader> rectangle_ok_color_hl_shader;

	ColorPicker *color_picker = nullptr;
	bool is_dragging = false;

	virtual void _initialize_controls() = 0;
	virtual void _update_cursor(const Vector2 &p_color_change_vector, bool p_is_echo) = 0;

	bool can_handle(const Ref<InputEvent> &p_event, Vector2 &r_position, bool *r_is_click = nullptr);
	void apply_color();
	void cancel_event();

	void draw_focus_rect(Control *p_control, const Rect2 &p_rect = Rect2());
	void draw_focus_circle(Control *p_control);
	void draw_sv_square(Control *p_control, const Rect2 &p_square, bool p_draw_focus = true);
	void draw_cursor(Control *p_control, const Vector2 &p_center, bool p_draw_bg = true);
	void draw_circle_cursor(Control *p_control, float p_hue, float p_saturation);

	void connect_shape_focus(Control *p_shape);
	void shape_focus_entered();
	void shape_focus_exited();

	void handle_cursor_editing(const Ref<InputEvent> &p_event, Control *p_control);
	int get_edge_h_change(const Vector2 &p_color_change_vector);
	float get_h_on_circle_edge(const Vector2 &p_color_change_vector);

public:
	Vector<Control *> controls;
	bool is_initialized = false;
	bool cursor_editing = false;
	float echo_multiplier = 1;

	static void init_shaders();
	static void finish_shaders();

	virtual String get_name() const = 0;
	virtual Ref<Texture2D> get_icon() const = 0;
	virtual bool is_ok_hsl() const { return false; }

	void initialize_controls();
	virtual void update_theme() = 0;
	void update_cursor(const Vector2 &p_color_change_vector, bool p_is_echo);
	virtual void grab_focus() = 0;

	ColorPickerShape(ColorPicker *p_color_picker);
};

class ColorPickerShapeRectangle : public ColorPickerShape {
	Control *sv_square = nullptr;
	Control *hue_slider = nullptr;

	void _sv_square_input(const Ref<InputEvent> &p_event);
	void _hue_slider_input(const Ref<InputEvent> &p_event);

	void _sv_square_draw();
	void _hue_slider_draw();

protected:
	virtual void _initialize_controls() override;
	virtual void _update_cursor(const Vector2 &p_color_change_vector, bool p_is_echo) override;

public:
	virtual String get_name() const override { return ETR("HSV Rectangle"); }
	virtual Ref<Texture2D> get_icon() const override { return color_picker->theme_cache.shape_rect; }
	virtual void update_theme() override;
	virtual void grab_focus() override;

	ColorPickerShapeRectangle(ColorPicker *p_color_picker) :
			ColorPickerShape(p_color_picker) {}
};

class ColorPickerShapeOKHSRectangle : public ColorPickerShape {
	GDCLASS(ColorPickerShapeOKHSRectangle, ColorPickerShape);

	MarginContainer *rectangle_margin = nullptr;

protected:
	Control *square = nullptr;
	Control *square_overlay = nullptr;
	Control *value_slider = nullptr;
	virtual Ref<Shader> _get_shader() const { return ColorPickerShape::rectangle_ok_color_hs_shader; }
	virtual void _initialize_controls() override;
	virtual void _update_cursor(const Vector2 &p_color_change_vector, bool p_is_echo) override;

	virtual void _square_draw();
	virtual void _square_overlay_input(const Ref<InputEvent> &p_event);
	virtual void _square_overlay_draw();

	virtual void _value_slider_input(const Ref<InputEvent> &p_event);
	virtual void _value_slider_draw();

public:
	virtual String get_name() const override { return ETR("OK HS Rectangle"); }
	virtual bool is_ok_hsl() const override { return true; }
	virtual Ref<Texture2D> get_icon() const override { return color_picker->theme_cache.shape_rect; }
	virtual void update_theme() override;
	virtual void grab_focus() override;

	ColorPickerShapeOKHSRectangle(ColorPicker *p_color_picker) :
			ColorPickerShape(p_color_picker) {}
};

class ColorPickerShapeOKHLRectangle : public ColorPickerShapeOKHSRectangle {
	GDCLASS(ColorPickerShapeOKHLRectangle, ColorPickerShapeOKHSRectangle);

protected:
	virtual Ref<Shader> _get_shader() const override { return ColorPickerShape::rectangle_ok_color_hl_shader; }
	virtual void _update_cursor(const Vector2 &p_color_change_vector, bool p_is_echo) override;

	virtual void _square_draw() override;
	virtual void _square_overlay_input(const Ref<InputEvent> &p_event) override;
	virtual void _square_overlay_draw() override;

	virtual void _value_slider_input(const Ref<InputEvent> &p_event) override;
	virtual void _value_slider_draw() override;

public:
	virtual String get_name() const override { return ETR("OK HL Rectangle"); }
	virtual bool is_ok_hsl() const override { return true; }
	virtual Ref<Texture2D> get_icon() const override { return color_picker->theme_cache.shape_rect; }

	ColorPickerShapeOKHLRectangle(ColorPicker *p_color_picker) :
			ColorPickerShapeOKHSRectangle(p_color_picker) {}
};

class ColorPickerShapeWheel : public ColorPickerShape {
	GDCLASS(ColorPickerShapeWheel, ColorPickerShape);

	static constexpr float WHEEL_RADIUS = 0.42;

	Control *wheel = nullptr;
	Control *wheel_uv = nullptr;

	bool wheel_focused = true;
	bool spinning = false;
	bool rotate_next_echo_event = false;

	float _get_h_on_wheel(const Vector2 &p_color_change_vector);
	void _reset_wheel_focus();

	void _wheel_input(const Ref<InputEvent> &p_event);

	void _wheel_draw();
	void _wheel_uv_draw();

protected:
	virtual void _initialize_controls() override;
	virtual void _update_cursor(const Vector2 &p_color_change_vector, bool p_is_echo) override;

public:
	virtual String get_name() const override { return ETR("HSV Wheel"); }
	virtual Ref<Texture2D> get_icon() const override { return color_picker->theme_cache.shape_rect_wheel; }
	virtual void update_theme() override;
	virtual void grab_focus() override;

	ColorPickerShapeWheel(ColorPicker *p_color_picker) :
			ColorPickerShape(p_color_picker) {}
};

class ColorPickerShapeCircle : public ColorPickerShape {
	GDCLASS(ColorPickerShapeCircle, ColorPickerShape);

protected:
	Control *circle = nullptr;
	Control *circle_overlay = nullptr;
	Control *value_slider = nullptr;

	Vector2i circle_keyboard_joypad_picker_cursor_position;

	virtual void _circle_input(const Ref<InputEvent> &p_event) = 0;
	virtual void _value_slider_input(const Ref<InputEvent> &p_event) = 0;

	virtual void _circle_draw() = 0;
	virtual void _circle_overlay_draw() = 0;
	virtual void _value_slider_draw() = 0;

	void update_circle_cursor(const Vector2 &p_color_change_vector, const Vector2 &p_center, const Vector2 &p_hue_offset);

public:
	virtual Ref<Shader> _get_shader() const = 0;
	virtual void _initialize_controls() override;

	virtual Ref<Texture2D> get_icon() const override { return color_picker->theme_cache.shape_circle; }
	virtual void update_theme() override;
	virtual void grab_focus() override;

	ColorPickerShapeCircle(ColorPicker *p_color_picker) :
			ColorPickerShape(p_color_picker) {}
};

class ColorPickerShapeVHSCircle : public ColorPickerShapeCircle {
	GDCLASS(ColorPickerShapeVHSCircle, ColorPickerShapeCircle);

protected:
	virtual void _update_cursor(const Vector2 &p_color_change_vector, bool p_is_echo) override;
	virtual Ref<Shader> _get_shader() const override { return ColorPickerShape::circle_shader; }

	virtual void _circle_input(const Ref<InputEvent> &p_event) override;
	virtual void _value_slider_input(const Ref<InputEvent> &p_event) override;

	virtual void _circle_draw() override;
	virtual void _circle_overlay_draw() override;
	virtual void _value_slider_draw() override;

public:
	virtual String get_name() const override { return ETR("VHS Circle"); }

	ColorPickerShapeVHSCircle(ColorPicker *p_color_picker) :
			ColorPickerShapeCircle(p_color_picker) {}
};

class ColorPickerShapeOKHSLCircle : public ColorPickerShapeCircle {
	GDCLASS(ColorPickerShapeOKHSLCircle, ColorPickerShapeCircle);

protected:
	virtual void _update_cursor(const Vector2 &p_color_change_vector, bool p_is_echo) override;
	virtual Ref<Shader> _get_shader() const override { return ColorPickerShape::circle_ok_color_shader; }

	virtual void _circle_input(const Ref<InputEvent> &p_event) override;
	virtual void _value_slider_input(const Ref<InputEvent> &p_event) override;

	virtual void _circle_draw() override;
	virtual void _circle_overlay_draw() override;
	virtual void _value_slider_draw() override;

public:
	virtual String get_name() const override { return ETR("OKHSL Circle"); }
	virtual bool is_ok_hsl() const override { return true; }

	ColorPickerShapeOKHSLCircle(ColorPicker *p_color_picker) :
			ColorPickerShapeCircle(p_color_picker) {}
};
