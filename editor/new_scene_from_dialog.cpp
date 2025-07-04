/**************************************************************************/
/*  new_scene_from_dialog.cpp                                             */
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

#include "editor/new_scene_from_dialog.h"

#include "core/config/project_settings.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "scene/2d/node_2d.h"
#include "scene/gui/grid_container.h"
#include "scene/gui/item_list.h"
#include "scene/resources/packed_scene.h"

NewSceneFromDialog::NewSceneFromDialog() {
	// Configure self
	set_ok_button_text(TTR("Create"));
	// Main Container
	VBoxContainer *vb = memnew(VBoxContainer);
	GridContainer *gc = memnew(GridContainer);
	gc->set_columns(2);
	vb->add_child(gc);
	add_child(vb);

	// Root Name Text Edit
	root_name_edit = memnew(LineEdit);
	root_name_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	register_text_enter(root_name_edit);
	gc->add_child(memnew(Label(TTR("Root Name:"))));
	gc->add_child(root_name_edit);

	// Inheritance Dropdown
	ancestor_options = memnew(OptionButton);
	ancestor_options->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	ancestor_options->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	gc->add_child(memnew(Label(TTR("Inherits:"))));
	gc->add_child(ancestor_options);

	// File Path Edit and Button
	HBoxContainer *hb = memnew(HBoxContainer);
	file_path_edit = memnew(LineEdit);
	file_path_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	path_button = memnew(Button);
	register_text_enter(file_path_edit);
	path_button->connect(SceneStringName(pressed), callable_mp(this, &NewSceneFromDialog::_browse_file));
	hb->add_child(file_path_edit);
	hb->add_child(path_button);
	gc->add_child(memnew(Label(TTR("Path:"))));
	gc->add_child(hb);

	// Set up File Browser:
	file_browser = memnew(EditorFileDialog);
	add_child(file_browser);
	file_browser->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
	Ref<PackedScene> sd = memnew(PackedScene);
	ResourceSaver::get_recognized_extensions(sd, &extensions);
	for (const String &extension : extensions) {
		file_browser->add_filter("*." + extension, extension.to_upper());
	}
	file_browser->set_title(TTR("Save New Scene As..."));

	file_browser->connect("file_selected", callable_mp(this, &NewSceneFromDialog::_file_selected));

	// TODO: Add reset options
	GridContainer *checkbox_gc = memnew(GridContainer);
	checkbox_gc->set_columns(2);
	reset_position_cb = memnew(CheckBox(TTR("Reset Position")));
	reset_rotation_cb = memnew(CheckBox(TTR("Reset Rotation")));
	reset_scale_cb = memnew(CheckBox(TTR("Reset Scale")));
	remove_script_cb = memnew(CheckBox(TTR("Remove Script")));
	checkbox_gc->add_child(reset_position_cb);
	checkbox_gc->add_child(reset_rotation_cb);
	checkbox_gc->add_child(reset_scale_cb);
	checkbox_gc->add_child(remove_script_cb);
	gc->add_child(memnew(Label(TTR("Configs:"))));
	gc->add_child(checkbox_gc);

	// Accept dialog
	accept = memnew(AcceptDialog);
	add_child(accept);

	set_title(TTR("Create New Scene From..."));
}

void NewSceneFromDialog::config(Node *p_selected_node) {
	selected_node = p_selected_node;
	// Set Root Name
	String root_name = p_selected_node->get_name();
	root_name_edit->set_text(root_name);

	String path_name = p_selected_node->get_owner()->get_scene_file_path().get_base_dir().path_join(root_name.to_snake_case() + ".tscn");
	// Set Path Name
	// String existing;
	// if (extensions.size()) {
	// 	String root_name(p_selected_node->get_name());
	// 	root_name = EditorNode::adjust_scene_name_casing(root_name);
	// 	existing = root_name + "." + extensions.begin()->to_lower();
	// }
	// TODO - The correct default_path
	file_path_edit->set_text(path_name);
	// TODO - Change the path when the base name changed

	//ANCHOR - set option buttons
	//NOTE - Need testing
	ancestor_options->clear();
	ancestor_options->add_item(p_selected_node->get_class_name(), 0);
	ancestor_options->set_item_tooltip(0, "New");

	int item_count = 1;
	if (p_selected_node->get_scene_instance_state().is_valid()) {
		Vector<Node *> instances;
		Ref<SceneState> scene_state = p_selected_node->get_scene_instance_state();
		while (scene_state.is_valid()) {
			Ref<PackedScene> pack_data = ResourceLoader::load(scene_state->get_path());
			if (!pack_data.is_valid()) {
				break;
			}
			// QUESTION - GEN_EDIT_STATE_INSTANCE?
			Node *current_node = pack_data->instantiate(PackedScene::GEN_EDIT_STATE_INSTANCE);
			String name = current_node->get_name();
			String ancestor_path_name = current_node->get_scene_file_path();
			String class_name = current_node->get_class_name();
			instances.push_back(current_node);
			ancestor_options->add_item(name, item_count);
			ancestor_options->set_item_tooltip(item_count, ancestor_path_name);
			ancestor_options->set_item_metadata(item_count, scene_state);

			scene_state = current_node->get_scene_inherited_state();
			item_count++;
		};
		for (Node *instance : instances) {
			memdelete(instance);
		}
	}
	ancestor_options->select(0);
}

Ref<SceneState> NewSceneFromDialog::get_selected_scene_state() const {
	return ancestor_options->get_selected_metadata();
}

String NewSceneFromDialog::get_new_node_name() const {
	return root_name_edit->get_text();
}

void NewSceneFromDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			path_button->set_button_icon(get_editor_theme_icon(SNAME("Folder")));
		} break;
	}
}

void NewSceneFromDialog::_bind_methods() {
	ADD_SIGNAL(MethodInfo(SNAME("create_branch_scene"),
			PropertyInfo(Variant::STRING, "file_path")));
}

void NewSceneFromDialog::_browse_file() {
	// TODO: set a default path
	file_browser->popup_file_dialog();
}

void NewSceneFromDialog::_file_selected(const String &p_file) {
	String path = ProjectSettings::get_singleton()->localize_path(p_file);
	file_path_edit->set_text(path);
	// _path_changed(path);

	String filename = path.get_file().get_basename();
	int select_start = path.rfind(filename);
	file_path_edit->select(select_start, select_start + filename.length());
	file_path_edit->set_caret_column(select_start + filename.length());
	file_path_edit->grab_focus();
}

void NewSceneFromDialog::_create_new_node() {
	String lpath = ProjectSettings::get_singleton()->localize_path(file_path_edit->get_text());
	if (EditorNode::get_singleton()->is_scene_open(lpath)) {
		accept->set_text(TTR("Can't overwrite scene that is still open!"));
		accept->popup_centered();
		return;
	}

	// TODO - There's some problem with the selected node. The selected node actually shouldn't be here?
	Node *base = selected_node;

	HashMap<const Node *, Node *> duplimap;
	HashMap<const Node *, Node *> inverse_duplimap;
	Node *copy = base->duplicate_from_editor(duplimap);

	for (const KeyValue<const Node *, Node *> &item : duplimap) {
		inverse_duplimap[item.value] = const_cast<Node *>(item.key);
	}

	if (copy) {
		// Handle Unique Nodes.
		for (int i = 0; i < copy->get_child_count(false); i++) {
			_set_node_owner_recursive(copy->get_child(i, false), copy, inverse_duplimap);
		}
		// Root node cannot ever be unique name in its own Scene!
		copy->set_unique_name_in_owner(false);

		bool reset_position = reset_scale_cb->is_pressed();
		bool reset_scale = reset_scale_cb->is_pressed();
		bool reset_rotation = reset_rotation_cb->is_pressed();
		// TODO - implementing remove_script
		bool remove_script = remove_script_cb->is_pressed();

		Node2D *copy_2d = Object::cast_to<Node2D>(copy);
		if (copy_2d != nullptr) {
			if (reset_position) {
				copy_2d->set_position(Vector2(0, 0));
			}
			if (reset_rotation) {
				copy_2d->set_rotation(0);
			}
			if (reset_scale) {
				copy_2d->set_scale(Size2(1, 1));
			}
		}
		Node3D *copy_3d = Object::cast_to<Node3D>(copy);
		if (copy_3d != nullptr) {
			if (reset_position) {
				copy_3d->set_position(Vector3(0, 0, 0));
			}
			if (reset_rotation) {
				copy_3d->set_rotation(Vector3(0, 0, 0));
			}
			if (reset_scale) {
				copy_3d->set_scale(Vector3(1, 1, 1));
			}
		}
		Ref<SceneState> inherited_state = ancestor_options->get_selected_metadata();
		if (inherited_state.is_valid()) {
			copy->set_scene_inherited_state(inherited_state);
		}

		copy->set_name(root_name_edit->get_text());

		Ref<PackedScene> sdata = memnew(PackedScene);
		Error err = sdata->pack(copy);
		memdelete(copy);

		if (err != OK) {
			accept->set_text(TTR("Couldn't save new scene. Likely dependencies (instances) couldn't be satisfied."));
			accept->popup_centered();
			return;
		}

		int flg = 0;
		if (EDITOR_GET("filesystem/on_save/compress_binary_resources")) {
			flg |= ResourceSaver::FLAG_COMPRESS;
		}

		// TODO: check file duplication!

		err = ResourceSaver::save(sdata, lpath, flg);
		if (err != OK) {
			accept->set_text(TTR("Error saving scene."));
			accept->popup_centered();
			return;
		}

		emit_signal(SNAME("create_branch_scene"), lpath);

	} else {
		// TODO: change to early return;
		accept->set_text(TTR("Error duplicating scene to save it."));
		accept->popup_centered();
		return;
	}
}

void NewSceneFromDialog::_set_node_owner_recursive(Node *p_node, Node *p_owner, const HashMap<const Node *, Node *> &p_inverse_duplimap) {
	HashMap<const Node *, Node *>::ConstIterator E = p_inverse_duplimap.find(p_node);

	if (E) {
		const Node *original = E->value;
		if (original->get_owner()) {
			p_node->set_owner(p_owner);
		}
	}

	for (int i = 0; i < p_node->get_child_count(false); i++) {
		_set_node_owner_recursive(p_node->get_child(i, false), p_owner, p_inverse_duplimap);
	}
}

void NewSceneFromDialog::ok_pressed() {
	_create_new_node();
}
