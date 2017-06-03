/*************************************************************************/
/*  button_group.h                                                       */
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
#ifndef BUTTON_GROUP_H
#define BUTTON_GROUP_H

#include "scene/gui/box_container.h"

#if 0
class BaseButton;

class ButtonGroup : public BoxContainer {

	GDCLASS(ButtonGroup,BoxContainer);


	Set<BaseButton*> buttons;


	Array _get_button_list() const;
	void _pressed(Object *p_button);

protected:
friend class BaseButton;

	void _add_button(BaseButton *p_button);
	void _remove_button(BaseButton *p_button);

	static void _bind_methods();
public:

	void get_button_list(List<BaseButton*> *p_buttons) const;
	BaseButton *get_pressed_button() const;
	BaseButton *get_focused_button() const;
	void set_pressed_button(BaseButton *p_button);
	int get_pressed_button_index() const;

	ButtonGroup();
};

#endif
#endif // BUTTON_GROUP_H
