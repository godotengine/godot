/*************************************************************************/
/*  color_picker.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef COLOR_PICKER_H
#define COLOR_PICKER_H

#include "scene/gui/aspect_ratio_container.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/check_button.h"
#include "scene/gui/grid_container.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/popup.h"
#include "scene/gui/separator.h"
#include "scene/gui/slider.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/texture_rect.h"

class ColorPresetButton : public BaseButton {
	GDCLASS(ColorPresetButton, BaseButton);

	Color preset_color;

protected:
	void _notification(int);

public:
	void set_preset_color(const Color &p_color);
	Color get_preset_color() const;

	ColorPresetButton(Color p_color);
	~ColorPresetButton();
};

class ColorPicker : public BoxContainer {
	GDCLASS(ColorPicker, BoxContainer);

public:
	enum PickerShapeType {
		SHAPE_HSV_RECTANGLE,
		SHAPE_HSV_WHEEL,
		SHAPE_VHS_CIRCLE,

		SHAPE_MAX
	};

private:
	static Ref<Shader> wheel_shader;
	static Ref<Shader> circle_shader;
	static List<Color> preset_cache;

	Control *screen = nullptr;
	Control *uv_edit = memnew(Control);
	Control *w_edit = memnew(Control);
	AspectRatioContainer *wheel_edit = memnew(AspectRatioContainer);
	MarginContainer *wheel_margin = memnew(MarginContainer);
	Ref<ShaderMaterial> wheel_mat;
	Ref<ShaderMaterial> circle_mat;
	Control *wheel = memnew(Control);
	Control *wheel_uv = memnew(Control);
	TextureRect *sample = memnew(TextureRect);
	GridContainer *preset_container = memnew(GridContainer);
	HSeparator *preset_separator = memnew(HSeparator);
	Button *btn_add_preset = memnew(Button);
	Button *btn_pick = memnew(Button);
	CheckButton *btn_hsv = memnew(CheckButton);
	CheckButton *btn_raw = memnew(CheckButton);
	HSlider *scroll[4];
	SpinBox *values[4];
	Label *labels[4];
	Button *text_type = memnew(Button);
	LineEdit *c_text = memnew(LineEdit);

	bool edit_alpha = true;
	Size2i ms;
	bool text_is_constructor = false;
	PickerShapeType picker_type = SHAPE_HSV_WHEEL;

	const int preset_column_count = 9;
	int prev_preset_size = 0;
	List<Color> presets;

	Color color;
	Color old_color;

	bool display_old_color = false;
	bool raw_mode_enabled = false;
	bool hsv_mode_enabled = false;
	bool deferred_mode_enabled = false;
	bool updating = true;
	bool changing_color = false;
	bool spinning = false;
	bool presets_enabled = true;
	bool presets_visible = true;

	float h = 0.0;
	float s = 0.0;
	float v = 0.0;
	Color last_hsv;

	void _html_submitted(const String &p_html);
	void _value_changed(double);
	void _update_controls();
	void _update_color(bool p_update_sliders = true);
	void _update_text_value();
	void _text_type_toggled();
	void _sample_input(const Ref<InputEvent> &p_event);
	void _sample_draw();
	void _hsv_draw(int p_which, Control *c);
	void _slider_draw(int p_which);

	void _uv_input(const Ref<InputEvent> &p_event, Control *c);
	void _w_input(const Ref<InputEvent> &p_event);
	void _preset_input(const Ref<InputEvent> &p_event, const Color &p_color);
	void _screen_input(const Ref<InputEvent> &p_event);
	void _add_preset_pressed();
	void _screen_pick_pressed();
	void _focus_enter();
	void _focus_exit();
	void _html_focus_exit();

	inline int _get_preset_size();
	void _add_preset_button(int p_size, const Color &p_color);

protected:
	void _notification(int);
	static void _bind_methods();

public:
	static void init_shaders();
	static void finish_shaders();

	void set_edit_alpha(bool p_show);
	bool is_editing_alpha() const;

	void _set_pick_color(const Color &p_color, bool p_update_sliders);
	void set_pick_color(const Color &p_color);
	Color get_pick_color() const;
	void set_old_color(const Color &p_color);

	void set_display_old_color(bool p_enabled);
	bool is_displaying_old_color() const;

	void set_picker_shape(PickerShapeType p_picker_type);
	PickerShapeType get_picker_shape() const;

	void add_preset(const Color &p_color);
	void erase_preset(const Color &p_color);
	PackedColorArray get_presets() const;
	void _update_presets();

	void set_hsv_mode(bool p_enabled);
	bool is_hsv_mode() const;

	void set_raw_mode(bool p_enabled);
	bool is_raw_mode() const;

	void set_deferred_mode(bool p_enabled);
	bool is_deferred_mode() const;

	void set_presets_enabled(bool p_enabled);
	bool are_presets_enabled() const;

	void set_presets_visible(bool p_visible);
	bool are_presets_visible() const;

	void set_focus_on_line_edit();

	ColorPicker();
};

class ColorPickerButton : public Button {
	GDCLASS(ColorPickerButton, Button);

	// Initialization is now done deferred,
	// this improves performance in the inspector as the color picker
	// can be expensive to initialize.

	PopupPanel *popup = nullptr;
	ColorPicker *picker = nullptr;
	Color color;
	bool edit_alpha = true;

	void _about_to_popup();
	void _color_changed(const Color &p_color);
	void _modal_closed();

	virtual void pressed() override;

	void _update_picker();

protected:
	void _notification(int);
	static void _bind_methods();

public:
	void set_pick_color(const Color &p_color);
	Color get_pick_color() const;

	void set_edit_alpha(bool p_show);
	bool is_editing_alpha() const;

	ColorPicker *get_picker();
	PopupPanel *get_popup();

	ColorPickerButton();
};

VARIANT_ENUM_CAST(ColorPicker::PickerShapeType);
#endif // COLOR_PICKER_H
