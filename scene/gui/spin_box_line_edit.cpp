/**************************************************************************/
/*  spin_box_line_edit.cpp                                                */
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

#include "spin_box_line_edit.h"

#include "core/object/callable_mp.h"
#include "scene/gui/spin_box.h"
#include "servers/display/accessibility_server.h"

void SpinBoxLineEdit::_accessibility_action_inc(const Variant &p_data) {
	SpinBox *parent_sb = Object::cast_to<SpinBox>(get_parent());
	if (parent_sb) {
		double step = ((parent_sb->get_step() > 0) ? parent_sb->get_step() : 1);
		parent_sb->set_value(parent_sb->get_value() + step);
	}
}

void SpinBoxLineEdit::_accessibility_action_dec(const Variant &p_data) {
	SpinBox *parent_sb = Object::cast_to<SpinBox>(get_parent());
	if (parent_sb) {
		double step = ((parent_sb->get_step() > 0) ? parent_sb->get_step() : 1);
		parent_sb->set_value(parent_sb->get_value() - step);
	}
}

void SpinBoxLineEdit::_notification(int p_what) {
	ERR_MAIN_THREAD_GUARD;
	switch (p_what) {
		case NOTIFICATION_ACCESSIBILITY_UPDATE: {
			RID ae = get_accessibility_element();
			ERR_FAIL_COND(ae.is_null());

			SpinBox *parent_sb = Object::cast_to<SpinBox>(get_parent());
			if (parent_sb) {
				AccessibilityServer::get_singleton()->update_set_role(ae, AccessibilityServerEnums::AccessibilityRole::ROLE_SPIN_BUTTON);
				AccessibilityServer::get_singleton()->update_set_name(ae, parent_sb->get_accessibility_name());
				AccessibilityServer::get_singleton()->update_set_description(ae, parent_sb->get_accessibility_description());
				AccessibilityServer::get_singleton()->update_set_live(ae, parent_sb->get_accessibility_live());
				AccessibilityServer::get_singleton()->update_set_num_value(ae, parent_sb->get_value());
				AccessibilityServer::get_singleton()->update_set_num_range(ae, parent_sb->get_min(), parent_sb->get_max());
				if (parent_sb->get_step() > 0) {
					AccessibilityServer::get_singleton()->update_set_num_step(ae, parent_sb->get_step());
				} else {
					AccessibilityServer::get_singleton()->update_set_num_step(ae, 1);
				}
				//AccessibilityServer::get_singleton()->update_set_num_jump(ae, ???);
				AccessibilityServer::get_singleton()->update_add_action(ae, AccessibilityServerEnums::AccessibilityAction::ACTION_DECREMENT, callable_mp(this, &SpinBoxLineEdit::_accessibility_action_dec));
				AccessibilityServer::get_singleton()->update_add_action(ae, AccessibilityServerEnums::AccessibilityAction::ACTION_INCREMENT, callable_mp(this, &SpinBoxLineEdit::_accessibility_action_inc));
			}
		} break;
	}
}
