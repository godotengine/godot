/*************************************************************************/
/*  popup.h                                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef POPUP_H
#define POPUP_H

#include "scene/gui/control.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class Popup : public Control {

	GDCLASS(Popup, Control);

	bool exclusive;
	bool popped_up;

protected:
	virtual void _post_popup() {}

	void _gui_input(InputEvent p_event);
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
	virtual void popup(const Rect2 &p_bounds = Rect2());

	virtual String get_configuration_warning() const;

	Popup();
	~Popup();
};

class PopupPanel : public Popup {

	GDCLASS(PopupPanel, Popup);

protected:
	void _notification(int p_what);

public:
	void set_child_rect(Control *p_child);
	PopupPanel();
};

#endif
