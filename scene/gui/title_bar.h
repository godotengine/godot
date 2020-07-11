/*************************************************************************/
/*  title_bar.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef TITLE_BAR_H
#define TITLE_BAR_H

#include "scene/gui/texture_button.h"
#include "scene/main/window.h"

class TitleBar : public Control {
	GDCLASS(TitleBar, Control);

public:
	enum TitleButton {
		BUTTON_CLOSE = 0b001,
		BUTTON_MAXIMIZE = 0b010,
		BUTTON_MINIMIZE = 0b100,
	};

private:
	Window *window = nullptr;
	TextureButton *close_btn;
	TextureButton *maximize_btn;
	TextureButton *minimize_btn;
	// Dragging position relative to top left of the window, including native decorations
	Point2i initial_drag_pos = { -1, -1 };
	bool force_custom_buttons = false;

	void _update_button_rects();
	void _update_button_textures();

	// Signal handlers for the custom buttons.
	void _close_pressed();
	void _maximize_pressed();
	void _minimize_pressed();

protected:
	static void _bind_methods();
	virtual void _gui_input(Ref<InputEvent> p_event);
	void _notification(int p_what);

public:
	bool is_forcing_custom_buttons() const;
	void set_force_custom_buttons(bool p_force);

	bool is_button_enabled(TitleButton p_button);
	void set_buttons_enabled(int p_flags, bool p_enabled);

	void close_window();
	void maximize_window();
	void restore_window();
	void minimize_window();

	virtual Size2 get_minimum_size() const override;

	void bind_window(Window *p_window);

	TitleBar();
	~TitleBar();
};

VARIANT_ENUM_CAST(TitleBar::TitleButton);

#endif // TITLE_BAR_H
