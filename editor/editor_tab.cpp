/**************************************************************************/
/*  editor_tab.cpp                                                        */
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

#include "editor_tab.h"

String EditorTab::get_name() const {
	return _name;
}

void EditorTab::set_name(String p_name) {
	_name = p_name;
}

String EditorTab::get_resource_path() const {
	return _resource_path;
}

void EditorTab::set_resource_path(String p_resource_path) {
	_resource_path = p_resource_path;
}

Ref<Texture2D> EditorTab::get_icon() const {
	return _icon;
}

void EditorTab::set_icon(Ref<Texture2D> p_icon) {
	_icon = p_icon;
}

Ref<Texture2D> EditorTab::get_tab_button_icon() const {
	return _tab_button_icon;
}

void EditorTab::set_tab_button_icon(Ref<Texture2D> p_tab_button_icon) {
	_tab_button_icon = p_tab_button_icon;
}

Variant EditorTab::get_state() const {
	return _state;
}

void EditorTab::set_state(Variant p_state) {
	_state = p_state;
}

bool EditorTab::get_closing() const {
	return _closing;
}

void EditorTab::set_closing(bool p_closing) {
	_closing = p_closing;
}

bool EditorTab::get_cancel() const {
	return _cancel;
}

void EditorTab::set_cancel(bool p_cancel) {
	_cancel = p_cancel;
}

uint64_t EditorTab::get_last_used() const {
	return _last_used;
}

void EditorTab::set_last_used(uint64_t p_last_used) {
	_last_used = p_last_used;
}

void EditorTab::update_last_used() {
	_last_used = OS::get_singleton()->get_ticks_msec();
}

void EditorTab::_bind_methods() {
	ADD_SIGNAL(MethodInfo("update_needed", PropertyInfo(Variant::OBJECT, "tab", PROPERTY_HINT_RESOURCE_TYPE, "EditorTab")));
	ADD_SIGNAL(MethodInfo("selected", PropertyInfo(Variant::OBJECT, "tab", PROPERTY_HINT_RESOURCE_TYPE, "EditorTab")));
	ADD_SIGNAL(MethodInfo("closing", PropertyInfo(Variant::OBJECT, "tab", PROPERTY_HINT_RESOURCE_TYPE, "EditorTab")));
	ADD_SIGNAL(MethodInfo("tab_button_pressed", PropertyInfo(Variant::OBJECT, "tab", PROPERTY_HINT_RESOURCE_TYPE, "EditorTab")));
	ADD_SIGNAL(MethodInfo("context_menu_needed", PropertyInfo(Variant::OBJECT, "tab", PROPERTY_HINT_RESOURCE_TYPE, "EditorTab"), PropertyInfo(Variant::OBJECT, "context_menu", PROPERTY_HINT_RESOURCE_TYPE, "PopupMenu")));
	ADD_SIGNAL(MethodInfo("context_menu_pressed", PropertyInfo(Variant::OBJECT, "tab", PROPERTY_HINT_RESOURCE_TYPE, "EditorTab"), PropertyInfo(Variant::INT, "option")));
}
