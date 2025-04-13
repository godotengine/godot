/**************************************************************************/
/*  add_component_dialog.cpp                                              */
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

#include "add_component_dialog.h"

#include "editor/themes/editor_scale.h"

AddComponentDialog::AddComponentDialog() {
	VBoxContainer *vbc = memnew(VBoxContainer);
	add_child(vbc);

	add_component_type = memnew(OptionButton);
	add_component_type->set_accessibility_name(TTRC("Type:"));

	vbc->add_child(add_component_type);

	Control *spacing = memnew(Control);
	vbc->add_child(spacing);
	spacing->set_custom_minimum_size(Size2(0, 10 * EDSCALE));

	set_ok_button_text(TTR("Add"));

	validation_panel = memnew(EditorValidationPanel);
	vbc->add_child(validation_panel);
	validation_panel->add_line(EditorValidationPanel::MSG_ID_DEFAULT, TTR("Component is valid."));
	validation_panel->set_update_callback(callable_mp(this, &AddComponentDialog::_check_component));
	validation_panel->set_accept_button(get_ok_button());
}

void AddComponentDialog::_complete_init(const StringName &p_title) {
	validation_panel->update();

	set_title(vformat(TTR("Add Component to \"%s\""), p_title));

	// Skip if we already completed the initialization.
	if (add_component_type->get_item_count()) {
		return;
	}

	// Theme icons can be retrieved only the Window has been initialized.
	for (int i = 0; i < Variant::VARIANT_MAX; i++) {
		if (i == Variant::NIL || i == Variant::RID || i == Variant::CALLABLE || i == Variant::SIGNAL) {
			continue; //not editable by inspector.
		}
		String type = i == Variant::OBJECT ? String("Resource") : Variant::get_type_name(Variant::Type(i));

		add_component_type->add_icon_item(get_editor_theme_icon(type), type, i);
	}
}

void AddComponentDialog::open(const StringName p_title, List<StringName> &p_existing_components) {
	this->_existing_components = p_existing_components;
	_complete_init(p_title);
	popup_centered();
}

StringName AddComponentDialog::get_component_name() {
	return "FIXME:: component name.";// add_meta_name->get_text();
}

void AddComponentDialog::_check_component() {
//	const String meta_name = add_meta_name->get_text();
//
//	if (meta_name.is_empty()) {
//		validation_panel->set_message(EditorValidationPanel::MSG_ID_DEFAULT, TTR("Metadata name can't be empty."), EditorValidationPanel::MSG_ERROR);
//	} else if (!meta_name.is_valid_ascii_identifier()) {
//		validation_panel->set_message(EditorValidationPanel::MSG_ID_DEFAULT, TTR("Metadata name must be a valid identifier."), EditorValidationPanel::MSG_ERROR);
//	} else if (_existing_metas.find(meta_name)) {
//		validation_panel->set_message(EditorValidationPanel::MSG_ID_DEFAULT, vformat(TTR("Metadata with name \"%s\" already exists."), meta_name), EditorValidationPanel::MSG_ERROR);
//	} else if (meta_name[0] == '_') {
//		validation_panel->set_message(EditorValidationPanel::MSG_ID_DEFAULT, TTR("Names starting with _ are reserved for editor-only metadata."), EditorValidationPanel::MSG_ERROR);
//	}
}
