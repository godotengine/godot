/**************************************************************************/
/*  color_mipmap_selector.h                                              */
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

#ifndef MIPMAP_SELECTOR_H
#define MIPMAP_SELECTOR_H

#include "scene/gui/base_button.h"
#include "scene/gui/box_container.h"

class PanelContainer;
class Button;

class MipmapSelector : public HBoxContainer {
	GDCLASS(MipmapSelector, HBoxContainer);

public:
	MipmapSelector();

	void set_mipmap_count(int count);
	int get_selected_mipmap() const;

private:
	void _notification(int p_what);

	void on_mipmap_button_toggled(BaseButton *button);
	void create_button(const String &p_text, Control *p_parent);
	void on_toggled(bool p_pressed);

	static void _bind_methods();

	Ref<ButtonGroup> mipmap_buttons;
	PanelContainer *panel = nullptr;
	HBoxContainer *container = nullptr;
	Button *toggle_button = nullptr;
};

#endif // MIPMAP_SELECTOR_H
