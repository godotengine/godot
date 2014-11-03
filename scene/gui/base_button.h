/*************************************************************************/
/*  base_button.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef BASE_BUTTON_H
#define BASE_BUTTON_H

#include "scene/gui/control.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/


class ButtonGroup;

class BaseButton : public Control {

	OBJ_TYPE( BaseButton, Control );

	bool toggle_mode;
	
	struct Status {
		
		bool pressed;
		bool hovering;
		bool press_attempt;
		bool pressing_inside;

		bool disabled;
		bool click_on_press;
		int pressing_button;

	} status;
	
	ButtonGroup *group;

		
protected:



	
	virtual void pressed();
	virtual void toggled(bool p_pressed);
	static void _bind_methods();
	virtual void _input_event(InputEvent p_event);
	void _notification(int p_what);

public:

	enum DrawMode {
		DRAW_NORMAL,
		DRAW_PRESSED,
		DRAW_HOVER,
		DRAW_DISABLED,
	};

	DrawMode get_draw_mode() const;

	/* Signals */
	
	bool is_pressed() const; ///< return wether button is pressed (toggled in)
	bool is_pressing() const; ///< return wether button is pressed (toggled in)
	bool is_hovered() const;

	void set_pressed(bool p_pressed); ///only works in toggle mode
	void set_toggle_mode(bool p_on);
	bool is_toggle_mode() const;
	
	void set_disabled(bool p_disabled);
	bool is_disabled() const;

	void set_click_on_press(bool p_click_on_press);
	bool get_click_on_press() const;


	BaseButton();	
	~BaseButton();

};

VARIANT_ENUM_CAST( BaseButton::DrawMode );

#endif
