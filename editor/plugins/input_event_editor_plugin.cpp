/*************************************************************************/
/*  input_event_editor_plugin.cpp                                        */
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

#include "input_event_editor_plugin.h"

void InputEventConfigContainer::_bind_methods() {
}

void InputEventConfigContainer::_configure_pressed() {
	config_dialog->popup_and_configure(input_event);
}

void InputEventConfigContainer::_event_changed() {
	input_event_text->set_text(input_event->as_text());
}

void InputEventConfigContainer::_config_dialog_confirmed() {
	Ref<InputEvent> ie = config_dialog->get_event();
	input_event->copy_from(ie);
	_event_changed();
}

Size2 InputEventConfigContainer::get_minimum_size() const {
	// Don't bother with a minimum x size for the control - we don't want the inspector
	// to jump in size if a long text is placed in the label (e.g. Joypad Axis description)
	return Size2(0, HBoxContainer::get_minimum_size().y);
}

void InputEventConfigContainer::set_event(const Ref<InputEvent> &p_event) {
	Ref<InputEventKey> k = p_event;
	Ref<InputEventMouseButton> m = p_event;
	Ref<InputEventJoypadButton> jb = p_event;
	Ref<InputEventJoypadMotion> jm = p_event;

	if (k.is_valid()) {
		config_dialog->set_allowed_input_types(InputEventConfigurationDialog::InputType::INPUT_KEY);
	} else if (m.is_valid()) {
		config_dialog->set_allowed_input_types(InputEventConfigurationDialog::InputType::INPUT_MOUSE_BUTTON);
	} else if (jb.is_valid()) {
		config_dialog->set_allowed_input_types(InputEventConfigurationDialog::InputType::INPUT_JOY_BUTTON);
	} else if (jm.is_valid()) {
		config_dialog->set_allowed_input_types(InputEventConfigurationDialog::InputType::INPUT_JOY_MOTION);
	}

	input_event = p_event;
	_event_changed();
	input_event->connect("changed", callable_mp(this, &InputEventConfigContainer::_event_changed));
}

InputEventConfigContainer::InputEventConfigContainer() {
	MarginContainer *mc = memnew(MarginContainer);
	mc->add_theme_constant_override("margin_left", 10);
	mc->add_theme_constant_override("margin_right", 10);
	mc->add_theme_constant_override("margin_top", 10);
	mc->add_theme_constant_override("margin_bottom", 10);
	add_child(mc);

	HBoxContainer *hb = memnew(HBoxContainer);
	mc->add_child(hb);

	open_config_button = memnew(Button);
	open_config_button->set_text(TTR("Configure"));
	open_config_button->connect("pressed", callable_mp(this, &InputEventConfigContainer::_configure_pressed));
	hb->add_child(open_config_button);

	input_event_text = memnew(Label);
	hb->add_child(input_event_text);

	config_dialog = memnew(InputEventConfigurationDialog);
	config_dialog->connect("confirmed", callable_mp(this, &InputEventConfigContainer::_config_dialog_confirmed));
	add_child(config_dialog);
}

bool EditorInspectorPluginInputEvent::can_handle(Object *p_object) {
	Ref<InputEventKey> k = Ref<InputEventKey>(p_object);
	Ref<InputEventMouseButton> m = Ref<InputEventMouseButton>(p_object);
	Ref<InputEventJoypadButton> jb = Ref<InputEventJoypadButton>(p_object);
	Ref<InputEventJoypadMotion> jm = Ref<InputEventJoypadMotion>(p_object);

	return k.is_valid() || m.is_valid() || jb.is_valid() || jm.is_valid();
}

void EditorInspectorPluginInputEvent::parse_begin(Object *p_object) {
	Ref<InputEvent> ie = Ref<InputEvent>(p_object);

	InputEventConfigContainer *picker_controls = memnew(InputEventConfigContainer);
	picker_controls->set_event(ie);
	add_custom_control(picker_controls);
}

InputEventEditorPlugin::InputEventEditorPlugin(EditorNode *p_node) {
	Ref<EditorInspectorPluginInputEvent> plugin;
	plugin.instantiate();
	add_inspector_plugin(plugin);
}
