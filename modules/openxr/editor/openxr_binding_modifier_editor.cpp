/**************************************************************************/
/*  openxr_binding_modifier_editor.cpp                                    */
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

#include "openxr_binding_modifier_editor.h"

#include "editor/editor_string_names.h"

void OpenXRBindingModifierEditor::_bind_methods() {
	ClassDB::bind_method("_on_remove_binding_modifier", &OpenXRBindingModifierEditor::_on_remove_binding_modifier);

	ADD_SIGNAL(MethodInfo("remove", PropertyInfo(Variant::OBJECT, "binding_modifier_editor")));
}

void OpenXRBindingModifierEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			rem_binding_modifier_btn->set_icon(get_theme_icon(SNAME("Remove"), EditorStringName(EditorIcons)));
		} break;
	}
}

void OpenXRBindingModifierEditor::_on_remove_binding_modifier() {
	// Tell parent to remove us
	emit_signal("remove", this);
}

void OpenXRBindingModifierEditor::add_property_editor(const String &p_property, EditorProperty *p_editor) {
	p_editor->set_label(p_property.capitalize());
	p_editor->connect("property_changed", callable_mp(this, &OpenXRBindingModifierEditor::_on_property_changed));
	main_vb->add_child(p_editor);
	property_editors[StringName(p_property)] = p_editor;
}

void OpenXRBindingModifierEditor::_on_property_changed(const String &p_property, const Variant &p_value, const String &p_name, bool p_changing) {
	ERR_FAIL_NULL(undo_redo);
	ERR_FAIL_COND(binding_modifier.is_null());

	undo_redo->create_action(vformat(TTR("Modify '%s' for binding modifier '%s'"), p_property, binding_modifier->get_description()));
	undo_redo->add_do_property(binding_modifier.ptr(), p_property, p_value);
	undo_redo->add_do_method(property_editors[p_property], "update_property");
	undo_redo->add_undo_property(binding_modifier.ptr(), p_property, binding_modifier->get(p_property));
	undo_redo->add_undo_method(property_editors[p_property], "update_property");
	undo_redo->commit_action();
}

OpenXRBindingModifierEditor::OpenXRBindingModifierEditor() {
	undo_redo = EditorUndoRedoManager::get_singleton();

	set_h_size_flags(Control::SIZE_EXPAND_FILL);

	main_vb = memnew(VBoxContainer);
	main_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	add_child(main_vb);

	header_hb = memnew(HBoxContainer);
	header_hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	main_vb->add_child(header_hb);

	binding_modifier_title = memnew(Label);
	binding_modifier_title->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	header_hb->add_child(binding_modifier_title);

	rem_binding_modifier_btn = memnew(Button);
	rem_binding_modifier_btn->set_tooltip_text(TTR("Remove binding modifier."));
	rem_binding_modifier_btn->connect(SceneStringName(pressed), callable_mp(this, &OpenXRBindingModifierEditor::_on_remove_binding_modifier));
	rem_binding_modifier_btn->set_flat(true);
	header_hb->add_child(rem_binding_modifier_btn);
}

void OpenXRBindingModifierEditor::set_binding_modifier(Ref<OpenXRActionMap> p_action_map, Ref<OpenXRBindingModifier> p_binding_modifier) {
	action_map = p_action_map;
	binding_modifier = p_binding_modifier;

	binding_modifier_title->set_text(p_binding_modifier->get_description());

	for (const KeyValue<StringName, EditorProperty *> &editor : property_editors) {
		editor.value->set_object_and_property(binding_modifier.ptr(), editor.key);
		editor.value->update_property();
	}
}
