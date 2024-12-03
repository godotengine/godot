/**************************************************************************/
/*  editor_create_dialog.cpp                                              */
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

#include "editor_create_dialog.h"

EditorCreateDialog *EditorCreateDialog::singleton = nullptr;

bool EditorCreateDialog::_is_type_valid(const StringName &p_type_name) const {
	return ClassDB::class_exists(p_type_name) || ScriptServer::is_global_class(p_type_name);
}

void EditorCreateDialog::set_create_dialog(CreateDialog *p_create_dialog) {
	create_dialog = p_create_dialog;
}

CreateDialog *EditorCreateDialog::get_create_dialog() const {
	return create_dialog;
}

ConfirmationDialog *EditorCreateDialog::get_dialog_window() const {
	return Object::cast_to<ConfirmationDialog>(create_dialog);
}

void EditorCreateDialog::add_type_to_blacklist(const StringName &p_type_name) {
	CHECK_IF_TYPE_IS_VALID(p_type_name);
	if (custom_type_blacklist.has(p_type_name)) {
		return;
	}
	custom_type_blacklist.insert(p_type_name);
}

HashSet<String> &EditorCreateDialog::get_type_blacklist() {
	return custom_type_blacklist;
}

bool EditorCreateDialog::is_type_in_blacklist(const StringName &p_type_name) const {
	return custom_type_blacklist.has(p_type_name);
}

void EditorCreateDialog::remove_type_from_blacklist(const StringName &p_type_name) {
	CHECK_IF_TYPE_IS_VALID(p_type_name);
	if (!custom_type_blacklist.has(p_type_name)) {
		return;
	}
	custom_type_blacklist.erase(p_type_name);
}

void EditorCreateDialog::clear_type_blacklist() {
	custom_type_blacklist.clear();
}

void EditorCreateDialog::set_type_custom_suffix(const StringName &p_type_name, const String &p_custom_suffix) {
	CHECK_IF_TYPE_IS_VALID(p_type_name);
	if (custom_type_suffixes.has(p_type_name)) {
		custom_type_suffixes[p_type_name] = p_custom_suffix;
	} else {
		custom_type_suffixes.insert(p_type_name, p_custom_suffix);
	}
}

bool EditorCreateDialog::has_type_custom_suffix(const StringName &p_type_name) const {
	return custom_type_suffixes.has(p_type_name);
}

String EditorCreateDialog::get_type_custom_suffix(const StringName &p_type_name) const {
	CHECK_IF_TYPE_IS_VALID_V(p_type_name, "<invalid suffix>");
	return custom_type_suffixes.has(p_type_name) ? custom_type_suffixes.get(p_type_name) : "<suffix not found>";
}

void EditorCreateDialog::clear_type_custom_suffixes() {
	custom_type_suffixes.clear();
}

Tree *EditorCreateDialog::get_search_options() const {
	CHECK_IF_NO_BOUND_CREATE_DIALOG_V(nullptr);
	return create_dialog->search_options;
}

EditorCreateDialog *EditorCreateDialog::get_singleton() {
	if (singleton == nullptr) {
		singleton = memnew(EditorCreateDialog);
	}
	return singleton;
}

void EditorCreateDialog::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_type_to_blacklist", "type_name"), &EditorCreateDialog::add_type_to_blacklist);
	ClassDB::bind_method(D_METHOD("is_type_in_blacklist", "type_name"), &EditorCreateDialog::is_type_in_blacklist);
	ClassDB::bind_method(D_METHOD("remove_type_from_blacklist", "type_name"), &EditorCreateDialog::remove_type_from_blacklist);
	ClassDB::bind_method(D_METHOD("clear_type_blacklist"), &EditorCreateDialog::clear_type_blacklist);
	ClassDB::bind_method(D_METHOD("set_type_custom_suffix", "type_name", "custom_suffix"), &EditorCreateDialog::set_type_custom_suffix);
	ClassDB::bind_method(D_METHOD("has_type_custom_suffix", "type_name"), &EditorCreateDialog::has_type_custom_suffix);
	ClassDB::bind_method(D_METHOD("get_type_custom_suffix", "type_name"), &EditorCreateDialog::get_type_custom_suffix);
	ClassDB::bind_method(D_METHOD("clear_type_custom_suffixes"), &EditorCreateDialog::clear_type_custom_suffixes);
	ClassDB::bind_method(D_METHOD("get_search_options"), &EditorCreateDialog::get_search_options);

	ADD_SIGNAL(MethodInfo("dialog_poped"));
	ADD_SIGNAL(MethodInfo("dialog_closed"));
	ADD_SIGNAL(MethodInfo("created"));
	ADD_SIGNAL(MethodInfo("favorites_updated"));
}

EditorCreateDialog::EditorCreateDialog() {
}
