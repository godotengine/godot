/*************************************************************************/
/*  base_button.cpp                                                      */
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
#include "base_button.h"
#include "os/keyboard.h"
#include "print_string.h"
#include "button_group.h"
#include "scene/scene_string_names.h"
#include "scene/main/viewport.h"

void BaseButton::_input_event(InputEvent p_event) {


	if (status.disabled) // no interaction with disabled button
		return;

	switch(p_event.type) {

		case InputEvent::MOUSE_BUTTON: {

			const InputEventMouseButton &b=p_event.mouse_button;

			if ( status.disabled || b.button_index!=1 )
				return;

			if (status.pressing_button)
				break;

			if (status.click_on_press) {

				if (b.pressed) {

					if (!toggle_mode) { //mouse press attempt

						status.press_attempt=true;
						status.pressing_inside=true;

						pressed();
						if (get_script_instance()) {
							Variant::CallError ce;
							get_script_instance()->call(SceneStringNames::get_singleton()->_pressed,NULL,0,ce);
						}

						emit_signal("pressed");

					} else {

						status.pressed=!status.pressed;
						pressed();
						if (get_script_instance()) {
							Variant::CallError ce;
							get_script_instance()->call(SceneStringNames::get_singleton()->_pressed,NULL,0,ce);
						}
						emit_signal("pressed");

						toggled(status.pressed);
						emit_signal("toggled",status.pressed);

					}


				} else {

					if (status.press_attempt && status.pressing_inside) {
//						released();
						emit_signal("released");
					}
					status.press_attempt=false;
				}
				update();
				break;
			}

			if (b.pressed) {

				status.press_attempt=true;
				status.pressing_inside=true;

			} else {


				if (status.press_attempt &&status.pressing_inside) {

					if (!toggle_mode) { //mouse press attempt

						pressed();
						if (get_script_instance()) {
							Variant::CallError ce;
							get_script_instance()->call(SceneStringNames::get_singleton()->_pressed,NULL,0,ce);
						}

						emit_signal("pressed");

					} else {

						status.pressed=!status.pressed;

						pressed();
						emit_signal("pressed");

						toggled(status.pressed);
						emit_signal("toggled",status.pressed);
						if (get_script_instance()) {
							get_script_instance()->call(SceneStringNames::get_singleton()->_toggled,status.pressed);
						}


					}

				}

				status.press_attempt=false;

			}

			update();
		} break;
		case InputEvent::MOUSE_MOTION: {

			if (status.press_attempt && status.pressing_button==0) {
				bool last_press_inside=status.pressing_inside;
				status.pressing_inside=has_point(Point2(p_event.mouse_motion.x,p_event.mouse_motion.y));
				if (last_press_inside!=status.pressing_inside)
					update();
			}
		} break;
		case InputEvent::ACTION:
		case InputEvent::JOYSTICK_BUTTON:
		case InputEvent::KEY: {


			if (p_event.is_echo()) {
				break;
			}

			if (status.disabled) {
				break;
			}

			if (status.press_attempt && status.pressing_button==0) {
				break;
			}

			if (p_event.is_action("ui_accept")) {

				if (p_event.is_pressed()) {

					status.pressing_button++;
					status.press_attempt=true;
					status.pressing_inside=true;

				} else if (status.press_attempt) {

					if (status.pressing_button)
						status.pressing_button--;

					if (status.pressing_button)
						break;

					status.press_attempt=false;
					status.pressing_inside=false;

					if (!toggle_mode) { //mouse press attempt

						pressed();
						emit_signal("pressed");
					} else {

						status.pressed=!status.pressed;

						pressed();
						emit_signal("pressed");

						toggled(status.pressed);
						if (get_script_instance()) {
							get_script_instance()->call(SceneStringNames::get_singleton()->_toggled,status.pressed);
						}
						emit_signal("toggled",status.pressed);
					}
				}

				accept_event();
				update();

			}
		}

	}
}

void BaseButton::_notification(int p_what) {


	if (p_what==NOTIFICATION_MOUSE_ENTER) {

		status.hovering=true;
		update();
	}

	if (p_what==NOTIFICATION_MOUSE_EXIT) {
		status.hovering=false;
		update();
	}
	if (p_what==NOTIFICATION_DRAG_BEGIN) {

		if (status.press_attempt) {
			status.press_attempt=false;
			status.pressing_button=0;
			update();
		}
	}

	if (p_what==NOTIFICATION_FOCUS_EXIT) {

		if (status.pressing_button && status.press_attempt) {
			status.press_attempt=false;
			status.pressing_button=0;
			update();
		}
	}

	if (p_what==NOTIFICATION_ENTER_TREE) {

		CanvasItem *ci=this;
		while(ci) {

			ButtonGroup *bg = ci->cast_to<ButtonGroup>();
			if (bg) {

				group=bg;
				group->_add_button(this);
			}

			ci=ci->get_parent_item();
		}
	}

	if (p_what==NOTIFICATION_EXIT_TREE) {

		if (group)
			group->_remove_button(this);
	}

	if (p_what==NOTIFICATION_VISIBILITY_CHANGED && !is_visible()) {

		if (!toggle_mode) {
			status.pressed = false;
		}
		status.hovering = false;
		status.press_attempt = false;
		status.pressing_inside = false;
		status.pressing_button = 0;
	}
}

void BaseButton::pressed() {

	if (get_script_instance())
		get_script_instance()->call("pressed");
}

void BaseButton::toggled(bool p_pressed) {

	if (get_script_instance())
		get_script_instance()->call("toggled",p_pressed);

}


void BaseButton::set_disabled(bool p_disabled) {

	status.disabled = p_disabled;
	update();
	_change_notify("disabled");
	if (p_disabled)
		set_focus_mode(FOCUS_NONE);
	else
		set_focus_mode(enabled_focus_mode);
}

bool BaseButton::is_disabled() const {

	return status.disabled;
}

void BaseButton::set_pressed(bool p_pressed) {

	if (!toggle_mode)
		return;
	if (status.pressed==p_pressed)
		return;
	_change_notify("pressed");
	status.pressed=p_pressed;
	update();
}

bool BaseButton::is_pressing() const{

	return status.press_attempt;
}

bool BaseButton::is_pressed() const {

	return toggle_mode?status.pressed:status.press_attempt;
}

bool BaseButton::is_hovered() const {

	return status.hovering;
}

BaseButton::DrawMode BaseButton::get_draw_mode() const {

	if (status.disabled) {
		return DRAW_DISABLED;
	};

	//print_line("press attempt: "+itos(status.press_attempt)+" hover: "+itos(status.hovering)+" pressed: "+itos(status.pressed));
	if (status.press_attempt==false && status.hovering && !status.pressed) {


		return DRAW_HOVER;
	} else {
		/* determine if pressed or not */

		bool pressing;
		if (status.press_attempt) {

			pressing=status.pressing_inside;
			if (status.pressed)
				pressing=!pressing;
		} else {

			pressing=status.pressed;
		}

		if (pressing)
			return DRAW_PRESSED;
		else
			return DRAW_NORMAL;
	}

	return DRAW_NORMAL;
}

void BaseButton::set_toggle_mode(bool p_on) {

	toggle_mode=p_on;
}

bool BaseButton::is_toggle_mode() const {

	return toggle_mode;
}

void BaseButton::set_click_on_press(bool p_click_on_press) {

	status.click_on_press=p_click_on_press;
}

bool BaseButton::get_click_on_press() const {

	return status.click_on_press;
}

void BaseButton::set_enabled_focus_mode(FocusMode p_mode) {

	enabled_focus_mode = p_mode;
	if (!status.disabled) {
		set_focus_mode( p_mode );
	}
}

Control::FocusMode BaseButton::get_enabled_focus_mode() const {

	return enabled_focus_mode;
}

void BaseButton::set_shortcut(const Ref<ShortCut>& p_shortcut) {

	if (shortcut.is_null() == p_shortcut.is_null())
		return;

	shortcut=p_shortcut;
	set_process_unhandled_input(shortcut.is_valid());
}

Ref<ShortCut> BaseButton:: get_shortcut() const {
	return shortcut;
}

void BaseButton::_unhandled_input(InputEvent p_event) {

	if (!is_disabled() && is_visible() && p_event.is_pressed() && !p_event.is_echo() && shortcut.is_valid() && shortcut->is_shortcut(p_event)) {

		if (get_viewport()->get_modal_stack_top() && !get_viewport()->get_modal_stack_top()->is_a_parent_of(this))
			return; //ignore because of modal window

		if (is_toggle_mode()) {
			set_pressed(!is_pressed());
			emit_signal("toggled",is_pressed());
		}

		emit_signal("pressed");
	}
}

String BaseButton::get_tooltip(const Point2& p_pos) const {

	String tooltip=Control::get_tooltip(p_pos);
	if (shortcut.is_valid() && shortcut->is_valid()) {
		if (tooltip.find("$sc")!=-1) {
			tooltip=tooltip.replace_first("$sc","("+shortcut->get_as_text()+")");
		} else {
			tooltip+=" ("+shortcut->get_as_text()+")";
		}
	}
	return tooltip;
}

void BaseButton::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_input_event"),&BaseButton::_input_event);
	ObjectTypeDB::bind_method(_MD("_unhandled_input"),&BaseButton::_unhandled_input);
	ObjectTypeDB::bind_method(_MD("set_pressed","pressed"),&BaseButton::set_pressed);
	ObjectTypeDB::bind_method(_MD("is_pressed"),&BaseButton::is_pressed);
	ObjectTypeDB::bind_method(_MD("is_hovered"),&BaseButton::is_hovered);
	ObjectTypeDB::bind_method(_MD("set_toggle_mode","enabled"),&BaseButton::set_toggle_mode);
	ObjectTypeDB::bind_method(_MD("is_toggle_mode"),&BaseButton::is_toggle_mode);
	ObjectTypeDB::bind_method(_MD("set_disabled","disabled"),&BaseButton::set_disabled);
	ObjectTypeDB::bind_method(_MD("is_disabled"),&BaseButton::is_disabled);
	ObjectTypeDB::bind_method(_MD("set_click_on_press","enable"),&BaseButton::set_click_on_press);
	ObjectTypeDB::bind_method(_MD("get_click_on_press"),&BaseButton::get_click_on_press);
	ObjectTypeDB::bind_method(_MD("get_draw_mode"),&BaseButton::get_draw_mode);
	ObjectTypeDB::bind_method(_MD("set_enabled_focus_mode","mode"),&BaseButton::set_enabled_focus_mode);
	ObjectTypeDB::bind_method(_MD("get_enabled_focus_mode"),&BaseButton::get_enabled_focus_mode);
	ObjectTypeDB::bind_method(_MD("set_shortcut","shortcut"),&BaseButton::set_shortcut);
	ObjectTypeDB::bind_method(_MD("get_shortcut"),&BaseButton::get_shortcut);

	BIND_VMETHOD(MethodInfo("_pressed"));
	BIND_VMETHOD(MethodInfo("_toggled",PropertyInfo(Variant::BOOL,"pressed")));

	ADD_SIGNAL( MethodInfo("pressed" ) );
	ADD_SIGNAL( MethodInfo("released" ) );
	ADD_SIGNAL( MethodInfo("toggled", PropertyInfo( Variant::BOOL,"pressed") ) );
	ADD_PROPERTYNZ( PropertyInfo( Variant::BOOL, "disabled"), _SCS("set_disabled"), _SCS("is_disabled"));
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "toggle_mode"), _SCS("set_toggle_mode"), _SCS("is_toggle_mode"));
	ADD_PROPERTYNZ( PropertyInfo( Variant::BOOL, "is_pressed"), _SCS("set_pressed"), _SCS("is_pressed"));
	ADD_PROPERTYNZ( PropertyInfo( Variant::BOOL, "click_on_press"), _SCS("set_click_on_press"), _SCS("get_click_on_press"));
	ADD_PROPERTY( PropertyInfo( Variant::INT,"enabled_focus_mode", PROPERTY_HINT_ENUM, "None,Click,All" ), _SCS("set_enabled_focus_mode"), _SCS("get_enabled_focus_mode") );
	ADD_PROPERTY( PropertyInfo( Variant::OBJECT, "shortcut",PROPERTY_HINT_RESOURCE_TYPE,"ShortCut"), _SCS("set_shortcut"), _SCS("get_shortcut"));


	BIND_CONSTANT( DRAW_NORMAL );
	BIND_CONSTANT( DRAW_PRESSED );
	BIND_CONSTANT( DRAW_HOVER );
	BIND_CONSTANT( DRAW_DISABLED );

}

BaseButton::BaseButton() {

	toggle_mode=false;
	status.pressed=false;
	status.press_attempt=false;
	status.hovering=false;
	status.pressing_inside=false;
	status.disabled = false;
	status.click_on_press=false;
	status.pressing_button=0;
	set_focus_mode( FOCUS_ALL );
	enabled_focus_mode = FOCUS_ALL;
	group=NULL;


}

BaseButton::~BaseButton()
{
}


