/*************************************************************************/
/*  color_picker.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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

#include "scene/gui/slider.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/label.h"
#include "scene/gui/button.h"
#include "scene/gui/popup.h"
#include "scene/gui/box_container.h"
#include "scene/gui/option_button.h"

class ColorPicker : public HBoxContainer {

	OBJ_TYPE(ColorPicker,HBoxContainer);
public:

	enum Mode {
		MODE_RGB,
		MODE_HSV,
		MODE_RAW
	};
private:

	Mode mode;

	OptionButton *mode_box;

	Control *color_box;
	HSlider *scroll[4];
	SpinBox *values[4];
	Label *labels[4];
	Label *html_num;
	LineEdit *html;
	bool edit_alpha;
	Size2i ms;

	Color color;
	bool updating;

	void _html_entered(const String& p_html);
	void _value_changed(double);
	void _update_controls();
	void _update_color();
	void _color_box_draw();
protected:

	void _notification(int);
	static void _bind_methods();
public:

	void set_edit_alpha(bool p_show);
	bool is_editing_alpha() const;

	void set_color(const Color& p_color);
	Color get_color() const;

	void set_mode(Mode p_mode);
	Mode get_mode() const;


	ColorPicker();
};

VARIANT_ENUM_CAST( ColorPicker::Mode );

class ColorPickerButton : public Button {

	OBJ_TYPE(ColorPickerButton,Button);

	PopupPanel *popup;
	ColorPicker *picker;

	void _color_changed(const Color& p_color);
	virtual void pressed();

protected:

	void _notification(int);
	static void _bind_methods();
public:

	void set_color(const Color& p_color);
	Color get_color() const;

	void set_edit_alpha(bool p_show);
	bool is_editing_alpha() const;

	ColorPickerButton();
};

#endif // COLOR_PICKER_H
