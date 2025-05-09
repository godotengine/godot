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
	vbc->set_custom_minimum_size(Size2(400 * EDSCALE, 80 * EDSCALE));

	component_picker = memnew(EditorResourcePicker);
	component_picker->set_base_type("Component");
	component_picker->set_can_load(false);

	vbc->add_child(component_picker);

	Control *spacing = memnew(Control);
	vbc->add_child(spacing);
	spacing->set_custom_minimum_size(Size2(0, 10 * EDSCALE));

	set_ok_button_text(TTR("Add"));

	validation_panel = memnew(EditorValidationPanel);
	vbc->add_child(validation_panel);
	validation_panel->add_line(EditorValidationPanel::MSG_ID_DEFAULT, TTR("Component is valid."));
	validation_panel->set_update_callback(callable_mp(this, &AddComponentDialog::_check_component));
	validation_panel->set_accept_button(get_ok_button());

	component_picker->connect("resource_changed", callable_mp(validation_panel, &EditorValidationPanel::update).unbind(1));
}

void AddComponentDialog::_complete_init(const StringName &p_title) {
	validation_panel->update();

	set_title(vformat(TTR("Add Component to \"%s\""), p_title));

	component_picker->set_edited_resource(nullptr);
}

void AddComponentDialog::open(const StringName p_title, List<StringName> &p_existing_components) {
	this->_existing_components = p_existing_components;
	_complete_init(p_title);
	popup_centered();
}

Ref<Component> AddComponentDialog::get_component() {
	return Object::cast_to<Component>(*component_picker->get_edited_resource());
}

void AddComponentDialog::_check_component() {
	Ref<Component> c = Object::cast_to<Component>(*component_picker->get_edited_resource());

	StringName n;
	if (c.is_valid()) {
		n = c->get_component_class();
	}

	if (c.is_null()) {
		validation_panel->set_message(EditorValidationPanel::MSG_ID_DEFAULT, TTR("Choose a component."), EditorValidationPanel::MSG_ERROR);
	} else if (_existing_components.find(n)) {
		validation_panel->set_message(EditorValidationPanel::MSG_ID_DEFAULT, vformat(TTR("Component of type \"%s\" already exists on this Actor."), n), EditorValidationPanel::MSG_ERROR);
	}
}
