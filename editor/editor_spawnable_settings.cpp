/*************************************************************************/
/*  editor_spawnable_settings.cpp                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "editor_spawnable_settings.h"

#include "core/config/project_settings.h"
#include "editor/editor_file_system.h"
#include "editor/editor_node.h"
#include "scene/gui/tree.h"

void EditorSpawnableSettings::_item_edited() {
	TreeItem *ti = tree->get_selected();
	if (!ti || !ti->is_editable(0)) {
		return;
	}
	updating = true;

	ResourceUID::ID uid = ti->get_metadata(0);
	if (ProjectSettings::get_singleton()->has_setting(_get_prop(uid))) {
		spawnable_remove(uid);
	} else {
		spawnable_add(uid, MultiplayerAPI::SPAWN_MODE_SERVER);
	}
	updating = false;
}

void EditorSpawnableSettings::update_spawnables() {
	if (updating) {
		return;
	}
	updating = true;
	tree->clear();
	TreeItem *root = tree->create_item();
	_fill_tree(EditorFileSystem::get_singleton()->get_filesystem(), root);
	updating = false;
}

bool EditorSpawnableSettings::spawnable_add(const ResourceUID::ID p_id, MultiplayerAPI::SpawnMode p_mode) {
	const String name = _get_prop(p_id);

	UndoRedo *undo_redo = EditorNode::get_undo_redo();

	undo_redo->create_action(TTR("Add Spawnable"));
	undo_redo->add_do_property(ProjectSettings::get_singleton(), name, p_mode);

	if (ProjectSettings::get_singleton()->has_setting(name)) {
		undo_redo->add_undo_property(ProjectSettings::get_singleton(), name, ProjectSettings::get_singleton()->get(name));
	} else {
		undo_redo->add_undo_property(ProjectSettings::get_singleton(), name, Variant());
	}

	undo_redo->add_do_method(this, "update_spawnables");
	undo_redo->add_undo_method(this, "update_spawnables");

	undo_redo->add_do_method(this, "emit_signal", spawnable_changed);
	undo_redo->add_undo_method(this, "emit_signal", spawnable_changed);

	undo_redo->commit_action();
	return true;
}

void EditorSpawnableSettings::spawnable_remove(const ResourceUID::ID p_id) {
	const String name = _get_prop(p_id);

	UndoRedo *undo_redo = EditorNode::get_undo_redo();

	undo_redo->create_action(TTR("Remove Spawnable"));

	undo_redo->add_do_property(ProjectSettings::get_singleton(), name, Variant());
	undo_redo->add_undo_property(ProjectSettings::get_singleton(), name, ProjectSettings::get_singleton()->get(name));

	undo_redo->add_do_method(this, "update_spawnables");
	undo_redo->add_undo_method(this, "update_spawnables");

	undo_redo->add_do_method(this, "emit_signal", spawnable_changed);
	undo_redo->add_undo_method(this, "emit_signal", spawnable_changed);

	undo_redo->commit_action();
}

bool EditorSpawnableSettings::_fill_tree(EditorFileSystemDirectory *p_dir, TreeItem *p_item) {
	p_item->set_cell_mode(0, TreeItem::CELL_MODE_CUSTOM);
	p_item->set_icon(0, get_theme_icon(SNAME("folder"), SNAME("FileDialog")));
	p_item->set_text(0, p_dir->get_name().is_empty() ? "res://" : p_dir->get_name() + "/");
	p_item->set_editable(0, false);
	p_item->set_metadata(0, p_dir->get_path());

	bool used = false;
	for (int i = 0; i < p_dir->get_subdir_count(); i++) {
		TreeItem *subdir = tree->create_item(p_item);
		if (_fill_tree(p_dir->get_subdir(i), subdir)) {
			used = true;
		} else {
			memdelete(subdir);
		}
	}

	for (int i = 0; i < p_dir->get_file_count(); i++) {
		String type = p_dir->get_file_type(i);
		if (type != "PackedScene") {
			continue;
		}

		TreeItem *file = tree->create_item(p_item);
		file->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
		file->set_text(0, p_dir->get_file(i));

		ResourceUID::ID uid = p_dir->get_file_uid(i);

		file->set_icon(0, EditorNode::get_singleton()->get_class_icon(type));
		file->set_editable(0, true);
		file->set_checked(0, ProjectSettings::get_singleton()->has_setting(_get_prop(uid)));
		file->set_metadata(0, uid);

		used = true;
	}

	return used;
}

void EditorSpawnableSettings::_bind_methods() {
	ClassDB::bind_method("update_spawnables", &EditorSpawnableSettings::update_spawnables);
	ClassDB::bind_method("spawnable_add", &EditorSpawnableSettings::spawnable_add);
	ClassDB::bind_method("spawnable_remove", &EditorSpawnableSettings::spawnable_remove);

	ADD_SIGNAL(MethodInfo("spawnable_changed"));
}

EditorSpawnableSettings::EditorSpawnableSettings() {
	spawnable_changed = "spawnable_changed";

	Label *label = memnew(Label);
	label->set_text(TTR("Select the scenes you wish to be automatically replicated by the server over the network.\nSee the MultiplayerAPI documentation for more information."));
	add_child(label, true);

	tree = memnew(Tree);
	tree->set_v_size_flags(SIZE_EXPAND_FILL);
	tree->connect("item_edited", callable_mp(this, &EditorSpawnableSettings::_item_edited));
	add_child(tree, true);
}
