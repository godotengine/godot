/*************************************************************************/
/*  popup.h                                                              */
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

#ifndef POPUP_H
#define POPUP_H

#include "scene/main/window.h"

#include "core/local_vector.h"

class Popup : public Window {
	GDCLASS(Popup, Window);

	LocalVector<Window *> visible_parents;
	bool popped_up = false;

	void _input_from_window(const Ref<InputEvent> &p_event);

	void _initialize_visible_parents();
	void _deinitialize_visible_parents();

	void _parent_focused();

protected:
	void _close_pressed();
	virtual Rect2i _popup_adjust_rect() const override;

	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_as_minsize();
	Popup();
	~Popup();
};

class PopupPanel : public Popup {
	GDCLASS(PopupPanel, Popup);

	Panel *panel;

protected:
	void _update_child_rects();
	void _notification(int p_what);

	virtual Size2 _get_contents_minimum_size() const override;

public:
	void set_child_rect(Control *p_child);
	PopupPanel();
};

#endif
