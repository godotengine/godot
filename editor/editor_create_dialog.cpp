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

#include "editor/editor_interface.h"

void EditorCreateDialog::_notify_visibility_changed() {
	CHECK_IF_NO_BOUND_CREATE_DIALOG;
	EditorInterface::get_singleton()->emit_signal(create_dialog->is_visible() ? "create_dialog_showed" : "create_dialog_hid", this);
}

void EditorCreateDialog::_notify_favourites_updated() {
	emit_signal("favorites_updated");
}

void EditorCreateDialog::_notify_created() {
	emit_signal("created");
}

void EditorCreateDialog::add_type_to_blacklist(const StringName &p_type_name) {
	CHECK_IF_NO_BOUND_CREATE_DIALOG;
	ERR_FAIL_COND_MSG(!ClassDB::is_parent_class(p_type_name, "Object"), vformat("Inexist type %s.", p_type_name));

	if (create_dialog->type_blacklist.has(p_type_name)) {
		return;
	}

	create_dialog->type_blacklist.insert(p_type_name);
	create_dialog->_update_search();
}

void EditorCreateDialog::remove_type_from_blacklist(const StringName &p_type_name) {
	CHECK_IF_NO_BOUND_CREATE_DIALOG;
	ERR_FAIL_COND_MSG(!ClassDB::is_parent_class(p_type_name, "Object"), vformat("Inexist type %s.", p_type_name));

	if (!create_dialog->type_blacklist.has(p_type_name)) {
		return;
	}

	create_dialog->type_blacklist.erase(p_type_name);
	create_dialog->_update_search();
}

void EditorCreateDialog::set_type_custom_suffix(const StringName &p_type_name, const String &p_custom_suffix) {
	CHECK_IF_NO_BOUND_CREATE_DIALOG;
	ERR_FAIL_COND_MSG(!ClassDB::is_parent_class(p_type_name, "Object"), vformat("Inexist type %s.", p_type_name));

	if (create_dialog->custom_type_suffixes.has(p_type_name)) {
		create_dialog->custom_type_suffixes[p_type_name] = p_custom_suffix;
	} else {
		create_dialog->custom_type_suffixes.insert(p_type_name, p_custom_suffix);
	}
	create_dialog->_update_search();
}

String EditorCreateDialog::get_type_custom_suffix(const StringName &p_type_name) const {
	CHECK_IF_NO_BOUND_CREATE_DIALOG_V("");
	ERR_FAIL_COND_V_MSG(!ClassDB::is_parent_class(p_type_name, "Object"), "", vformat("Inexist type %s.", p_type_name));
	return create_dialog->custom_type_suffixes.has(p_type_name) ? create_dialog->custom_type_suffixes.get(p_type_name) : "";
}

void EditorCreateDialog::clear_all_type_custom_suffixes() {
	CHECK_IF_NO_BOUND_CREATE_DIALOG;
	create_dialog->custom_type_suffixes.clear();
	create_dialog->_update_search();
}

Tree *EditorCreateDialog::get_search_options() const {
	CHECK_IF_NO_BOUND_CREATE_DIALOG_V(nullptr);
	return create_dialog->search_options;
}

void EditorCreateDialog::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_type_to_blacklist", "type_name"), &EditorCreateDialog::add_type_to_blacklist);
	ClassDB::bind_method(D_METHOD("remove_type_from_blacklist", "type_name"), &EditorCreateDialog::remove_type_from_blacklist);
	ClassDB::bind_method(D_METHOD("set_type_custom_suffix", "type_name", "custom_suffix"), &EditorCreateDialog::set_type_custom_suffix);
	ClassDB::bind_method(D_METHOD("get_type_custom_suffix", "type_name"), &EditorCreateDialog::get_type_custom_suffix);
	ClassDB::bind_method(D_METHOD("clear_all_type_custom_suffixes"), &EditorCreateDialog::clear_all_type_custom_suffixes);
	ClassDB::bind_method(D_METHOD("get_search_options"), &EditorCreateDialog::get_search_options);

	ADD_SIGNAL(MethodInfo("create"));
	ADD_SIGNAL(MethodInfo("favorites_updated"));
}

EditorCreateDialog::EditorCreateDialog() {
	ERR_PRINT("Cannot instantiate an editor create dialog without binding create dialog. The editor create dialog was deleted.");
	memdelete(this);
}

EditorCreateDialog::EditorCreateDialog(CreateDialog *p_create_dialog) {
	create_dialog = p_create_dialog;
	if (create_dialog != nullptr) {
		print_line(vformat(R"(Connecting signals from %s to %s)", create_dialog->get_name(), get_name()));
		create_dialog->connect("visibility_changed", callable_mp(this, &EditorCreateDialog::_notify_visibility_changed));
		create_dialog->connect("create", callable_mp(this, &EditorCreateDialog::_notify_created));
		create_dialog->connect("favorites_updated", callable_mp(this, &EditorCreateDialog::_notify_favourites_updated));
	} else {
		ERR_PRINT("Cannot instantiate an editor create dialog without binding create dialog. The editor create dialog was deleted.");
		memdelete(this);
	}
}
