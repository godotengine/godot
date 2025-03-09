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

#ifndef COLOR_PICKER_SHAPE_H
#define COLOR_PICKER_SHAPE_H

#include "scene/gui/color_picker.h"

class ColorPickerShape : public Object {
	GDCLASS(ColorPickerShape, Object);

	void _emit_color_changed();

protected:
	ColorPicker *color_picker = nullptr;
	bool is_dragging = false;

	virtual void _initialize_controls() = 0;

	bool can_handle(const Ref<InputEvent> &p_event, Vector2 &r_position, bool *r_is_click = nullptr);
	void handle_event();
	void cancel_event();

	void draw_sv_square(Control *p_control, const Rect2 &p_square);
	void draw_cursor(Control *p_control, const Vector2 &p_center, bool p_draw_bg = true);
	void draw_circle_cursor(Control *p_control, float p_hue);

public:
	Vector<Control *> controls;
	bool is_initialized = false;

	virtual String get_name() const = 0;
	virtual Ref<Texture2D> get_icon() const = 0;
	virtual bool is_ok_hsl() const { return false; }

	void initialize_controls();
	virtual void update_theme() = 0;

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

public:
	virtual String get_name() const override { return ETR("HSV Rectangle"); }
	virtual Ref<Texture2D> get_icon() const override { return color_picker->theme_cache.shape_rect; }
	virtual void update_theme() override;

	ColorPickerShapeRectangle(ColorPicker *p_color_picker) :
			ColorPickerShape(p_color_picker) {}
};

class ColorPickerShapeWheel : public ColorPickerShape {
	GDCLASS(ColorPickerShapeWheel, ColorPickerShape);

	inline static constexpr float WHEEL_RADIUS = 0.42;

	MarginContainer *wheel_margin = nullptr;
	Control *wheel = nullptr;
	Control *wheel_uv = nullptr;

	bool spinning = false;

	void _wheel_input(const Ref<InputEvent> &p_event);

	void _wheel_draw();
	void _wheel_uv_draw();

protected:
	virtual void _initialize_controls() override;

public:
	virtual String get_name() const override { return ETR("HSV Wheel"); }
	virtual Ref<Texture2D> get_icon() const override { return color_picker->theme_cache.shape_rect_wheel; }
	virtual void update_theme() override;

	ColorPickerShapeWheel(ColorPicker *p_color_picker) :
			ColorPickerShape(p_color_picker) {}
};

class ColorPickerShapeVHSCircle : public ColorPickerShape {
	GDCLASS(ColorPickerShapeVHSCircle, ColorPickerShape);

	MarginContainer *circle_margin = nullptr;
	Control *circle = nullptr;
	Control *circle_overlay = nullptr;
	Control *value_slider = nullptr;

	void _circle_input(const Ref<InputEvent> &p_event);
	void _value_slider_input(const Ref<InputEvent> &p_event);

	void _circle_draw();
	void _circle_overlay_draw();
	void _value_slider_draw();

protected:
	virtual void _initialize_controls() override;

public:
	virtual String get_name() const override { return ETR("VHS Circle"); }
	virtual Ref<Texture2D> get_icon() const override { return color_picker->theme_cache.shape_circle; }
	virtual void update_theme() override;

	ColorPickerShapeVHSCircle(ColorPicker *p_color_picker) :
			ColorPickerShape(p_color_picker) {}
};

class ColorPickerShapeOKHSLCircle : public ColorPickerShape {
	GDCLASS(ColorPickerShapeOKHSLCircle, ColorPickerShape);

	MarginContainer *circle_margin = nullptr;
	Control *circle = nullptr;
	Control *circle_overlay = nullptr;
	Control *value_slider = nullptr;

	void _circle_input(const Ref<InputEvent> &p_event);
	void _value_slider_input(const Ref<InputEvent> &p_event);

	void _circle_draw();
	void _circle_overlay_draw();
	void _value_slider_draw();

protected:
	virtual void _initialize_controls() override;

public:
	virtual String get_name() const override { return ETR("OKHSL Circle"); }
	virtual Ref<Texture2D> get_icon() const override { return color_picker->theme_cache.shape_circle; }
	virtual bool is_ok_hsl() const override { return true; }
	virtual void update_theme() override;

	ColorPickerShapeOKHSLCircle(ColorPicker *p_color_picker) :
			ColorPickerShape(p_color_picker) {}
};

#endif // COLOR_PICKER_SHAPE_H
