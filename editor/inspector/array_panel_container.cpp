/**************************************************************************/
/*  array_panel_container.cpp                                             */
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

#include "array_panel_container.h"

#include "core/object/callable_mp.h"
#include "editor/inspector/editor_inspector.h"
#include "servers/display/accessibility_server.h"

void ArrayPanelContainer::_accessibility_action_menu(const Variant &p_data) {
	EditorInspectorArray *el = Object::cast_to<EditorInspectorArray>(get_meta("element"));
	if (el) {
		int index = get_meta("index");
		el->show_menu(index, Vector2());
	}
}

void ArrayPanelContainer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ACCESSIBILITY_UPDATE: {
			RID ae = get_accessibility_element();
			ERR_FAIL_COND(ae.is_null());

			AccessibilityServer::get_singleton()->update_set_role(ae, AccessibilityServerEnums::AccessibilityRole::ROLE_BUTTON);

			AccessibilityServer::get_singleton()->update_set_name(ae, get_meta("text"));
			AccessibilityServer::get_singleton()->update_set_value(ae, get_meta("text"));

			AccessibilityServer::get_singleton()->update_set_popup_type(ae, AccessibilityServerEnums::AccessibilityPopupType::POPUP_MENU);
			AccessibilityServer::get_singleton()->update_add_action(ae, AccessibilityServerEnums::AccessibilityAction::ACTION_SHOW_CONTEXT_MENU, callable_mp(this, &ArrayPanelContainer::_accessibility_action_menu));
		} break;
	}
}

ArrayPanelContainer::ArrayPanelContainer() {
	set_focus_mode(FOCUS_ACCESSIBILITY);
}
