/*************************************************************************/
/*  openxr_action_set_editor.cpp                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "openxr_action_set_editor.h"
#include "openxr_action_editor.h"

void OpenXRActionSetEditor::_bind_methods() {
	ADD_SIGNAL(MethodInfo("remove", PropertyInfo(Variant::OBJECT, "action_set_editor")));
	ADD_SIGNAL(MethodInfo("action_removed"));
}

void OpenXRActionSetEditor::_set_fold_icon() {
	if (is_expanded) {
		fold_btn->set_icon(get_theme_icon(SNAME("GuiTreeArrowDown"), SNAME("EditorIcons")));
	} else {
		fold_btn->set_icon(get_theme_icon(SNAME("GuiTreeArrowRight"), SNAME("EditorIcons")));
	}
}

void OpenXRActionSetEditor::_theme_changed() {
	_set_fold_icon();
	add_action->set_icon(get_theme_icon(SNAME("Add"), SNAME("EditorIcons")));
	rem_action_set->set_icon(get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")));
}

void OpenXRActionSetEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			_theme_changed();
			panel->add_theme_style_override("panel", get_theme_stylebox(SNAME("panel"), SNAME("TabContainer")));
		} break;
	}
}

OpenXRActionEditor *OpenXRActionSetEditor::_add_action_editor(Ref<OpenXRAction> p_action) {
	OpenXRActionEditor *action_editor = memnew(OpenXRActionEditor(p_action));
	action_editor->connect("remove", callable_mp(this, &OpenXRActionSetEditor::_on_remove_action));
	actions_vb->add_child(action_editor);

	return action_editor;
}

void OpenXRActionSetEditor::_update_actions() {
	// out with the old...
	while (actions_vb->get_child_count() > 0) {
		memdelete(actions_vb->get_child(0));
	}

	// in with the new...
	Array actions = action_set->get_actions();
	for (int i = 0; i < actions.size(); i++) {
		Ref<OpenXRAction> action = actions[i];
		_add_action_editor(action);
	}
}

void OpenXRActionSetEditor::_on_toggle_expand() {
	is_expanded = !is_expanded;
	actions_vb->set_visible(is_expanded);
	_set_fold_icon();
}

void OpenXRActionSetEditor::_on_action_set_name_changed(const String p_new_text) {
	// TODO validate if entry is allowed

	// If our localized name matches our action set name, set this too
	if (action_set->get_name() == action_set->get_localized_name()) {
		action_set->set_localized_name(p_new_text);
		action_set_localized_name->set_text(p_new_text);
	}
	action_set->set_name(p_new_text);
}

void OpenXRActionSetEditor::_on_action_set_localized_name_changed(const String p_new_text) {
	action_set->set_localized_name(p_new_text);
}

void OpenXRActionSetEditor::_on_action_set_priority_changed(const String p_new_text) {
	int64_t value = p_new_text.to_int();

	action_set->set_priority(value);
}

void OpenXRActionSetEditor::_on_add_action() {
	Ref<OpenXRAction> new_action;

	new_action.instantiate();
	new_action->set_name("New");
	new_action->set_localized_name("New");
	action_set->add_action(new_action);

	_add_action_editor(new_action);

	// TODO handle focus
}

void OpenXRActionSetEditor::_on_remove_action_set() {
	emit_signal("remove", this);
}

void OpenXRActionSetEditor::_on_remove_action(Object *p_action_editor) {
	OpenXRActionEditor *action_editor = Object::cast_to<OpenXRActionEditor>(p_action_editor);
	ERR_FAIL_NULL(action_editor);
	ERR_FAIL_COND(action_editor->get_parent() != actions_vb);
	Ref<OpenXRAction> action = action_editor->get_action();
	ERR_FAIL_COND(action.is_null());

	// TODO add undo/redo action

	// TODO find where this action is used by our interaction profiles and remove it there

	// And remove it....
	action_map->remove_action(action->get_name_with_set()); // remove it from the set and any interaction profile it relates to
	actions_vb->remove_child(action_editor);
	action_editor->queue_free();

	// Let action map editor know so we can update our interaction profiles
	emit_signal("action_removed");
}

void OpenXRActionSetEditor::set_focus_on_entry() {
	ERR_FAIL_NULL(action_set_name);
	action_set_name->grab_focus();
}

OpenXRActionSetEditor::OpenXRActionSetEditor(Ref<OpenXRActionMap> p_action_map, Ref<OpenXRActionSet> p_action_set) {
	action_map = p_action_map;
	action_set = p_action_set;

	set_h_size_flags(Control::SIZE_EXPAND_FILL);

	panel = memnew(PanelContainer);
	panel->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	add_child(panel);

	HBoxContainer *panel_hb = memnew(HBoxContainer);
	panel_hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	panel->add_child(panel_hb);

	fold_btn = memnew(Button);
	fold_btn->set_v_size_flags(Control::SIZE_SHRINK_BEGIN);
	fold_btn->connect("pressed", callable_mp(this, &OpenXRActionSetEditor::_on_toggle_expand));
	fold_btn->set_flat(true);
	panel_hb->add_child(fold_btn);

	main_vb = memnew(VBoxContainer);
	main_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	panel_hb->add_child(main_vb);

	action_set_hb = memnew(HBoxContainer);
	action_set_hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	main_vb->add_child(action_set_hb);

	action_set_name = memnew(LineEdit);
	action_set_name->set_text(action_set->get_name());
	action_set_name->set_custom_minimum_size(Size2(150.0, 0.0));
	action_set_name->connect("text_changed", callable_mp(this, &OpenXRActionSetEditor::_on_action_set_name_changed));
	action_set_hb->add_child(action_set_name);

	action_set_localized_name = memnew(LineEdit);
	action_set_localized_name->set_text(action_set->get_localized_name());
	action_set_localized_name->set_custom_minimum_size(Size2(150.0, 0.0));
	action_set_localized_name->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	action_set_localized_name->connect("text_changed", callable_mp(this, &OpenXRActionSetEditor::_on_action_set_localized_name_changed));
	action_set_hb->add_child(action_set_localized_name);

	action_set_priority = memnew(TextEdit);
	action_set_priority->set_text(itos(action_set->get_priority()));
	action_set_priority->set_custom_minimum_size(Size2(50.0, 0.0));
	action_set_priority->connect("text_changed", callable_mp(this, &OpenXRActionSetEditor::_on_action_set_priority_changed));
	action_set_hb->add_child(action_set_priority);

	add_action = memnew(Button);
	add_action->set_tooltip_text("Add Action.");
	add_action->connect("pressed", callable_mp(this, &OpenXRActionSetEditor::_on_add_action));
	add_action->set_flat(true);
	action_set_hb->add_child(add_action);

	rem_action_set = memnew(Button);
	rem_action_set->set_tooltip_text("Remove Action Set.");
	rem_action_set->connect("pressed", callable_mp(this, &OpenXRActionSetEditor::_on_remove_action_set));
	rem_action_set->set_flat(true);
	action_set_hb->add_child(rem_action_set);

	actions_vb = memnew(VBoxContainer);
	actions_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	main_vb->add_child(actions_vb);

	_update_actions();
}
