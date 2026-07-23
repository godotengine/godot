/**************************************************************************/
/*  resource_bundle_editor_plugin.cpp                                     */
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

#include "resource_bundle_editor_plugin.h"

#include "core/io/dir_access.h"
#include "core/io/resource_saver.h"
#include "core/object/callable_mp.h"
#include "editor/docks/editor_dock_manager.h"
#include "editor/editor_node.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/script/script_editor_plugin.h"
#include "scene/gui/label.h"
#include "scene/gui/tab_container.h"
#include "scene/resources/resource_bundle.h"

#include "modules/gdscript/gdscript.h"

// ResourceBundleEditor

void ResourceBundleEditor::_update_info_label() {
	info_label->set_visible(bundle_container->get_tab_count() == 0);
}

void ResourceBundleEditor::_bundle_tab_close_pressed(int p_tab) {
	ResourceBundleTab *tab = Object::cast_to<ResourceBundleTab>(bundle_container->get_tab_control(p_tab));
	_remove_tab(tab);
}

int ResourceBundleEditor::_find_tab_index(const String &p_path) const {
	for (int i = 0; i < bundle_tabs.size(); i++) {
		Ref<ResourceBundle> bundle = bundle_tabs[i]->get_bundle();
		if (bundle.is_valid() && bundle->is_owned(p_path)) {
			return i;
		}
	}
	return -1;
}

void ResourceBundleEditor::_add_tab(const Ref<ResourceBundle> &p_bundle) {
	ERR_FAIL_COND_MSG(p_bundle->get_owned_path().is_empty(), "Bundle owned path is empty");
	String tab_name = p_bundle->get_owned_path().get_base_dir().get_file().to_pascal_case();
	ResourceBundleTab *new_tab = memnew(ResourceBundleTab);
	new_tab->set_name(tab_name);
	new_tab->set_tab_name(tab_name);
	new_tab->set_bundle(p_bundle);
	bundle_container->add_child(new_tab);
	bundle_tabs.push_back(new_tab);
	int idx = bundle_container->get_tab_count() - 1;
	bundle_container->set_current_tab(idx);
	_update_info_label();
}

void ResourceBundleEditor::_add_or_focus_tab(const Ref<ResourceBundle> &p_bundle) {
	if (int idx = _find_tab_index(p_bundle->get_owned_path()); idx != -1) {
		bundle_container->set_current_tab(idx);
	} else {
		_add_tab(p_bundle);
	}
}

void ResourceBundleEditor::_remove_tab(const String &p_path) {
	for (int i = 0; i < bundle_tabs.size(); i++) {
		Ref<ResourceBundle> bundle = bundle_tabs[i]->get_bundle();
		if (bundle.is_valid() && bundle->is_owned(p_path)) {
			ResourceBundleTab *tab = bundle_tabs[i];
			bundle_container->remove_child(tab);
			bundle_tabs.remove_at(i);
			_remove_tab(tab);
			break;
		}
	}
}

void ResourceBundleEditor::_remove_tab(ResourceBundleTab *p_tab) {
	if (p_tab && p_tab->is_inside_tree()) {
		bundle_container->remove_child(p_tab);
		p_tab->queue_free();
	}
	_update_info_label();
}

void ResourceBundleEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_VISIBILITY_CHANGED: {
			_update_info_label();
		}
	}
}

bool ResourceBundleEditor::can_drop_data(const Point2 &p_point, const Variant &p_data) const {
	Dictionary d = p_data;
	if (!d.has("type")) {
		return false;
	}

	String type = d["type"];
	if (type != "files_and_dirs") {
		return false;
	}

	return true;
}

void ResourceBundleEditor::drop_data(const Point2 &p_point, const Variant &p_data) {
	Dictionary d = p_data;
	if (!d.has("files")) {
		return;
	}

	const PackedStringArray paths = d["files"];
	for (const String &path : paths) {
		Ref<ResourceBundle> bundle = ResourceBundle::load(path);
		if (bundle.is_valid() && bundle->is_owned(path)) {
			_add_or_focus_tab(bundle);
		} else {
			make_bundle(path);
		}
	}
}

bool ResourceBundleEditor::make_bundle(const String &p_path) {
	int steps = 2;
	Ref<GDScript> script;
	Ref<ResourceBundle> bundle;
	{ // Step 1
		String path = p_path.path_join("schema.gd");
		script.instantiate();
		script->set_source_code("extends ResourceBundleSchema\n");
		script->set_path(path, true);
		Error err = ResourceSaver::save(script, path, ResourceSaver::FLAG_CHANGE_PATH);
		if (err == OK) {
			ScriptEditor::get_singleton()->edit(script.ptr());
			steps--;
		}
	}
	{ // Step 2
		bundle.instantiate();
		bundle->set_owned_path(p_path);
		bundle->set_schema_path(ResourceUID::path_to_uid(script->get_path()));
		Error err = ResourceSaver::save(bundle, p_path.path_join(".bundle"));
		if (err == OK) {
			EditorNode::get_singleton()->edit_resource(bundle);
			steps--;
		}
	}
	return steps == 0;
}

bool ResourceBundleEditor::remove_bundle(const String &p_path) {
	int steps = 0;
	{ // Step 1
		String path = p_path.path_join(".bundle");
		if (FileAccess::exists(path)) {
			steps++;
			Ref<DirAccess> dir = DirAccess::create(DirAccess::ACCESS_RESOURCES);
			Error err = dir->remove(path);
			if (err == OK) {
				_remove_tab(p_path);
				EditorFileSystem::get_singleton()->update_file(path);
				steps--;
			}
		}
	}
	{ // Step 2
		String path = p_path.path_join("schema.gd");
		if (FileAccess::exists(path)) {
			steps++;
			ScriptEditor::get_singleton()->close_file(path);
			Ref<DirAccess> dir = DirAccess::create(DirAccess::ACCESS_RESOURCES);
			Error err = dir->remove(path);
			if (err == OK) {
				EditorFileSystem::get_singleton()->update_file(path);
				steps--;
			}
		}
	}
	return steps == 0;
}

void ResourceBundleEditor::edit(Object *p_object) {
	Ref<ResourceBundle> bundle = Object::cast_to<ResourceBundle>(p_object);
	if (bundle.is_valid()) {
		_add_or_focus_tab(bundle);
	}
	_update_info_label();
}

ResourceBundleEditor::ResourceBundleEditor() {
	singleton = this;

	set_name(TTR("Bundle"));
	set_icon_name("ResourceBundle");
	set_default_slot(EditorDock::DOCK_SLOT_BOTTOM);
	set_available_layouts(EditorDock::DOCK_LAYOUT_HORIZONTAL | EditorDock::DOCK_LAYOUT_FLOATING);

	set_global(false);
	set_transient(true);
	set_closable(true);

	set_focus_mode(FOCUS_ALL);
	set_process_shortcut_input(true);

	file_dialog = memnew(EditorFileDialog);
	add_child(file_dialog);

	bundle_container = memnew(TabContainer);
	bundle_container->get_tab_bar()->set_tab_close_display_policy(TabBar::CLOSE_BUTTON_SHOW_ACTIVE_ONLY);
	bundle_container->get_tab_bar()->connect("tab_close_pressed", callable_mp(this, &ResourceBundleEditor::_bundle_tab_close_pressed));
	add_child(bundle_container);

	info_label = memnew(Label);
	info_label->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	info_label->set_text(TTRC("To get started, drag a directory here to convert it into a bundle, or open an existing bundle."));
	info_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	info_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	info_label->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	info_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	info_label->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	add_child(info_label);
}

// ResourceBundleTab

void ResourceBundleTab::set_tab_name(const String &p_name) {
	tab_name = p_name;
	label->set_text(tab_name);
}

String ResourceBundleTab::get_tab_name() const {
	return tab_name;
}

void ResourceBundleTab::set_bundle(const Ref<ResourceBundle> &p_bundle) {
	bundle = p_bundle;
}

Ref<ResourceBundle> ResourceBundleTab::get_bundle() const {
	return bundle;
}

ResourceBundleTab::ResourceBundleTab() {
	table = memnew(ResourceBundleTable);
	table->set_theme_type_variation("ScrollContainerSecondary");
	add_child(table);

	label = memnew(Label);
	add_child(label);
}

// ResourceBundleTable

ResourceBundleTable::ResourceBundleTable() {
}

// ResourceBundleEditorPlugin

void ResourceBundleEditorPlugin::edit(Object *p_object) {
	bundle_editor->edit(p_object);
}

bool ResourceBundleEditorPlugin::handles(Object *p_object) const {
	Ref<GDScript> script = Object::cast_to<GDScript>(p_object);
	Ref<ResourceBundle> bundle = Object::cast_to<ResourceBundle>(p_object);
	if (script.is_valid() && script->get_instance_base_type() == "ResourceBundleSchema") {
		return true;
	} else if (bundle.is_valid()) {
		return true;
	}
	return false;
}

void ResourceBundleEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		bundle_editor->make_visible();
	}
}

ResourceBundleEditorPlugin::ResourceBundleEditorPlugin() {
	bundle_editor = memnew(ResourceBundleEditor);
	EditorDockManager::get_singleton()->add_dock(bundle_editor);
}
