/**************************************************************************/
/*  popup.h                                                               */
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

#ifndef POPUP_H
#define POPUP_H

#include "scene/gui/control.h"

class Popup : public Control {
	GDCLASS(Popup, Control);

	bool exclusive;
	bool popped_up;

private:
	void _popup(const Rect2 &p_bounds = Rect2(), const bool p_centered = false);

protected:
	virtual void _post_popup() {}

	void _gui_input(Ref<InputEvent> p_event);
	void _notification(int p_what);
	virtual void _fix_size();
	static void _bind_methods();

public:
	enum {
		NOTIFICATION_POST_POPUP = 80,
		NOTIFICATION_POPUP_HIDE = 81
	};

	void set_exclusive(bool p_exclusive);
	bool is_exclusive() const;

	void popup_centered_ratio(float p_screen_ratio = 0.75);
	void popup_centered(const Size2 &p_size = Size2());
	void popup_centered_minsize(const Size2 &p_minsize = Size2());
	void set_as_minsize();
	void popup_centered_clamped(const Size2 &p_size = Size2(), float p_fallback_ratio = 0.75);
	virtual void popup(const Rect2 &p_bounds = Rect2());

	virtual String get_configuration_warning() const;

	Popup();
	~Popup();
};

class PopupPanel : public Popup {
	GDCLASS(PopupPanel, Popup);

protected:
	void _update_child_rects();
	void _notification(int p_what);

public:
	virtual Size2 get_minimum_size() const;
	PopupPanel();
};

#endif // POPUP_H
