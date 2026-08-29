/**************************************************************************/
/*  vertex_mobile_editor_panel.cpp                                      */
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

#ifdef TOOLS_ENABLED

#include "vertex_mobile_editor_panel.h"

#include "scene/gui/button.h"
#include "scene/gui/check_box.h"
#include "scene/gui/label.h"
#include "scene/gui/spin_box.h"

#include "core/object/callable_mp.h"

void VertexMobileEditorPanel::_apply_touch_sizes() {
	if (settings.is_null()) {
		return;
	}
	float ts = settings->get_touch_target_size();
	// Enlarge buttons so they meet minimum touch-target guidance on phones.
	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
		if (c) {
			c->set_custom_minimum_size(Size2(0, ts));
		}
	}
}

void VertexMobileEditorPanel::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			set_v_size_flags(SIZE_EXPAND_FILL);
			set_h_size_flags(SIZE_EXPAND_FILL);

			Label *title = memnew(Label);
			title->set_text("Vertex Mobile Editor");
			add_child(title);

			compact_button = memnew(Button);
			compact_button->set_text("Compact Mode");
			compact_button->set_toggle_mode(true);
			compact_button->set_pressed(settings->get_compact_mode());
			compact_button->connect("toggled", callable_mp(this, &VertexMobileEditorPanel::_on_compact_toggled));
			add_child(compact_button);

			collapse_button = memnew(Button);
			collapse_button->set_text("Collapse Panels");
			collapse_button->set_toggle_mode(true);
			collapse_button->set_pressed(settings->get_panels_collapsed());
			collapse_button->connect("toggled", callable_mp(this, &VertexMobileEditorPanel::_on_collapse_toggled));
			add_child(collapse_button);

			keyboard_assist_button = memnew(Button);
			keyboard_assist_button->set_text("Virtual Keyboard Assist");
			keyboard_assist_button->set_toggle_mode(true);
			keyboard_assist_button->set_pressed(settings->get_virtual_keyboard_assist());
			keyboard_assist_button->connect("toggled", callable_mp(this, &VertexMobileEditorPanel::_on_keyboard_assist_toggled));
			add_child(keyboard_assist_button);

			HBoxContainer *row1 = memnew(HBoxContainer);
			Label *l1 = memnew(Label);
			l1->set_text("Touch target (dp):");
			row1->add_child(l1);
			touch_target = memnew(SpinBox);
			touch_target->set_min(32);
			touch_target->set_max(96);
			touch_target->set_value(settings->get_touch_target_size());
			touch_target->connect("value_changed", callable_mp(this, &VertexMobileEditorPanel::_on_touch_target_changed));
			row1->add_child(touch_target);
			add_child(row1);

			HBoxContainer *row2 = memnew(HBoxContainer);
			Label *l2 = memnew(Label);
			l2->set_text("Toolbar height (dp):");
			row2->add_child(l2);
			toolbar_height = memnew(SpinBox);
			toolbar_height->set_min(32);
			toolbar_height->set_max(72);
			toolbar_height->set_value(settings->get_compact_toolbar_height());
			toolbar_height->connect("value_changed", callable_mp(this, &VertexMobileEditorPanel::_on_toolbar_height_changed));
			row2->add_child(toolbar_height);
			add_child(row2);

			_apply_touch_sizes();
		} break;
	}
}

void VertexMobileEditorPanel::_on_compact_toggled(bool p_on) {
	if (settings.is_valid()) {
		settings->set_compact_mode(p_on);
	}
}

void VertexMobileEditorPanel::_on_collapse_toggled(bool p_on) {
	if (settings.is_valid()) {
		settings->set_panels_collapsed(p_on);
	}
}

void VertexMobileEditorPanel::_on_keyboard_assist_toggled(bool p_on) {
	if (settings.is_valid()) {
		settings->set_virtual_keyboard_assist(p_on);
	}
}

void VertexMobileEditorPanel::_on_touch_target_changed(double p_v) {
	if (settings.is_valid()) {
		settings->set_touch_target_size((float)p_v);
		_apply_touch_sizes();
	}
}

void VertexMobileEditorPanel::_on_toolbar_height_changed(double p_v) {
	if (settings.is_valid()) {
		settings->set_compact_toolbar_height((float)p_v);
	}
}

VertexMobileEditorPanel::VertexMobileEditorPanel() {
	if (settings.is_null()) {
		settings.instantiate();
	}
}

void VertexMobileEditorPanel::_bind_methods() {
}

#endif // TOOLS_ENABLED
