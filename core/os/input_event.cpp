/*************************************************************************/
/*  input_event.cpp                                                      */
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
#include "input_event.h"
#include "input_map.h"
#include "os/keyboard.h"
/**
 * 
 */

bool InputEvent::operator==(const InputEvent &p_event) const {

	return true;
}
InputEvent::operator String() const {

	String str="Device "+itos(device)+" ID "+itos(ID)+" ";
	
	switch(type) {
	
		case NONE: {
		
			return "Event: None";
		} break;
		case KEY: {
			
			str+= "Event: Key ";
			str=str+"Unicode: "+String::chr(key.unicode)+" Scan: "+itos( key.scancode )+" Echo: "+String(key.echo?"True":"False")+" Pressed"+String(key.pressed?"True":"False")+" Mod: ";
			if (key.mod.shift)
				str+="S";
			if (key.mod.control)
				str+="C";
			if (key.mod.alt)
				str+="A";
			if (key.mod.meta)
				str+="M";
				
			return str;
		} break;
		case MOUSE_MOTION: {
		
			str+= "Event: Motion ";
			str=str+" Pos: " +itos(mouse_motion.x)+","+itos(mouse_motion.y)+" Rel: "+itos(mouse_motion.relative_x)+","+itos(mouse_motion.relative_y)+" Mask: ";
			for (int i=0;i<8;i++) {
			
				if ((1<<i)&mouse_motion.button_mask)
					str+=itos(i+1);
			}
			str+=" Mod: ";
			if (key.mod.shift)
				str+="S";
			if (key.mod.control)
				str+="C";
			if (key.mod.alt)
				str+="A";
			if (key.mod.meta)
				str+="M";

			return str;
		} break;
		case MOUSE_BUTTON: {
			str+= "Event: Button ";
			str=str+"Pressed: "+itos(mouse_button.pressed)+" Pos: " +itos(mouse_button.x)+","+itos(mouse_button.y)+" Button: "+itos(mouse_button.button_index)+" Mask: ";
			for (int i=0;i<8;i++) {
			
				if ((1<<i)&mouse_button.button_mask)
					str+=itos(i+1);
			}
			str+=" Mod: ";
			if (key.mod.shift)
				str+="S";
			if (key.mod.control)
				str+="C";
			if (key.mod.alt)
				str+="A";
			if (key.mod.meta)
				str+="M";

			str+=String(" DoubleClick: ")+(mouse_button.doubleclick?"Yes":"No");
			
			return str;
		
		} break;
		case JOYSTICK_MOTION: {
			str+= "Event: JoyMotion ";
			str=str+"Axis: "+itos(joy_motion.axis)+" Value: " +rtos(joy_motion.axis_value);
			return str;

		} break;
		case JOYSTICK_BUTTON: {
			str+= "Event: JoyButton ";
			str=str+"Pressed: "+itos(joy_button.pressed)+" Index: " +itos(joy_button.button_index)+" pressure "+rtos(joy_button.pressure);
			return str;

		} break;
		case SCREEN_TOUCH: {
			str+= "Event: ScreenTouch ";
			str=str+"Pressed: "+itos(screen_touch.pressed)+" Index: " +itos(screen_touch.index)+" pos "+rtos(screen_touch.x)+","+rtos(screen_touch.y);
			return str;

		} break;
		case SCREEN_DRAG: {
			str+= "Event: ScreenDrag ";
			str=str+" Index: " +itos(screen_drag.index)+" pos "+rtos(screen_drag.x)+","+rtos(screen_drag.y);
			return str;

		} break;
		case ACTION: {
			str+= "Event: Action: "+InputMap::get_singleton()->get_action_from_id(action.action)+" Pressed: "+itos(action.pressed);
			return str;

		} break;

	}
	
	return "";
}

void InputEvent::set_as_action(const String& p_action, bool p_pressed) {

	type=ACTION;
	action.action=InputMap::get_singleton()->get_action_id(p_action);
	action.pressed=p_pressed;
}

bool InputEvent::is_pressed() const {

	switch(type) {

		case KEY: return key.pressed;
		case MOUSE_BUTTON: return mouse_button.pressed;
		case JOYSTICK_BUTTON: return joy_button.pressed;
		case SCREEN_TOUCH: return screen_touch.pressed;
		case ACTION: return action.pressed;
		default: {}
	}

	return false;
}

bool InputEvent::is_echo() const {

	return (type==KEY && key.echo);
}

bool InputEvent::is_action(const String& p_action) const {

	return InputMap::get_singleton()->event_is_action(*this,p_action);
}

uint32_t InputEventKey::get_scancode_with_modifiers() const {

	uint32_t sc=scancode;
	if (mod.control)
		sc|=KEY_MASK_CTRL;
	if (mod.alt)
		sc|=KEY_MASK_ALT;
	if (mod.shift)
		sc|=KEY_MASK_SHIFT;
	if (mod.meta)
		sc|=KEY_MASK_META;

	return sc;

}
