/**************************************************************************/
/*  popup_button.h                                                        */
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

#include "core/object/gdvirtual.gen.inc"
#include "scene/gui/button.h"
#include "scene/gui/popup.h"

class PopupButton : public Button {
	GDCLASS(PopupButton, Button);

	bool switch_on_hover = false;

	Popup *popup = nullptr;

	void _setup_popup();

	void _popup_visibility_changed();

protected:
	void _notification(int p_notification);
	static void _bind_methods();

	virtual void add_child_notify(Node *p_child) override;
	virtual void switched_on_hover(PopupButton *p_to) {}

	virtual Popup *create_popup();
	virtual void about_to_popup();
	virtual void setup_popup_position();
	virtual void post_popup();

	void ensure_popup();

	GDVIRTUAL0R(Popup *, _create_popup)
	GDVIRTUAL0(_about_to_popup)
	GDVIRTUAL0(_setup_popup_position)
	GDVIRTUAL0(_post_popup)

public:
	virtual void pressed() override;

	Popup *get_generic_popup();
	void show_popup();

	void set_switch_on_hover(bool p_enabled) { switch_on_hover = p_enabled; }
	bool is_switch_on_hover() { return switch_on_hover; }

	PopupButton();
};
