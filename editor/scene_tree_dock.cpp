/*************************************************************************/
/*  scene_tree_dock.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "scene_tree_dock.h"

#include "core/io/resource_saver.h"
#include "core/os/keyboard.h"
#include "core/project_settings.h"
#include "editor/animation_editor.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/multi_node_edit.h"
#include "editor/plugins/animation_player_editor_plugin.h"
#include "editor/plugins/canvas_item_editor_plugin.h"
#include "editor/plugins/script_editor_plugin.h"
#include "editor/plugins/spatial_editor_plugin.h"
#include "editor/script_editor_debugger.h"
#include "scene/main/viewport.h"
#include "scene/resources/packed_scene.h"

void SceneTreeDock::_nodes_drag_begin() {

	if (restore_script_editor_on_drag) {
		EditorNode::get_singleton()->set_visible_editor(EditorNode::EDITOR_SCRIPT);
		restore_script_editor_on_drag = false;
	}
}

void SceneTreeDock::_input(Ref<InputEvent> p_event) {

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid() && !mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {
		restore_script_editor_on_drag = false; //lost chance
	}
}

void SceneTreeDock::_unhandled_key_input(Ref<InputEvent> p_event) {

	if (get_viewport()->get_modal_stack_top())
		return; //ignore because of modal window

	if (!p_event->is_pressed() || p_event->is_echo())
		return;

	if (ED_IS_SHORTCUT("scene_tree/add_child_node", p_event)) {
		_tool_selected(TOOL_NEW);
	} else if (ED_IS_SHORTCUT("scene_tree/instance_scene", p_event)) {
		_tool_selected(TOOL_INSTANCE);
	} else if (ED_IS_SHORTCUT("scene_tree/change_node_type", p_event)) {
		_tool_selected(TOOL_REPLACE);
	} else if (ED_IS_SHORTCUT("scene_tree/duplicate", p_event)) {
		_tool_selected(TOOL_DUPLICATE);
	} else if (ED_IS_SHORTCUT("scene_tree/attach_script", p_event)) {
		_tool_selected(TOOL_ATTACH_SCRIPT);
	} else if (ED_IS_SHORTCUT("scene_tree/clear_script", p_event)) {
		_tool_selected(TOOL_CLEAR_SCRIPT);
	} else if (ED_IS_SHORTCUT("scene_tree/move_up", p_event)) {
		_tool_selected(TOOL_MOVE_UP);
	} else if (ED_IS_SHORTCUT("scene_tree/move_down", p_event)) {
		_tool_selected(TOOL_MOVE_DOWN);
	} else if (ED_IS_SHORTCUT("scene_tree/reparent", p_event)) {
		_tool_selected(TOOL_REPARENT);
	} else if (ED_IS_SHORTCUT("scene_tree/merge_from_scene", p_event)) {
		_tool_selected(TOOL_MERGE_FROM_SCENE);
	} else if (ED_IS_SHORTCUT("scene_tree/save_branch_as_scene", p_event)) {
		_tool_selected(TOOL_NEW_SCENE_FROM);
	} else if (ED_IS_SHORTCUT("scene_tree/delete_no_confirm", p_event)) {
		_tool_selected(TOOL_ERASE, true);
	} else if (ED_IS_SHORTCUT("scene_tree/copy_node_path", p_event)) {
		_tool_selected(TOOL_COPY_NODE_PATH);
	} else if (ED_IS_SHORTCUT("scene_tree/delete", p_event)) {
		_tool_selected(TOOL_ERASE);
	}
}

void SceneTreeDock::instance(const String &p_file) {

	Node *parent = scene_tree->get_selected();
	if (!parent || !edited_scene) {

		current_option = -1;
		accept->get_ok()->set_text(TTR("OK :("));
		accept->set_text(TTR("No parent to instance a child at."));
		accept->popup_centered_minsize();
		return;
	};

	ERR_FAIL_COND(!parent);

	Vector<String> scenes;
	scenes.push_back(p_file);
	_perform_instance_scenes(scenes, parent, -1);
}

void SceneTreeDock::instance_scenes(const Vector<String> &p_files, Node *p_parent) {

	Node *parent = p_parent;

	if (!parent) {
		parent = scene_tree->get_selected();
	}

	if (!parent || !edited_scene) {

		accept->get_ok()->set_text(TTR("OK"));
		accept->set_text(TTR("No parent to instance the scenes at."));
		accept->popup_centered_minsize();
		return;
	};

	_perform_instance_scenes(p_files, parent, -1);
}

void SceneTreeDock::_perform_instance_scenes(const Vector<String> &p_files, Node *parent, int p_pos) {

	ERR_FAIL_COND(!parent);

	Vector<Node *> instances;

	bool error = false;

	for (int i = 0; i < p_files.size(); i++) {

		Ref<PackedScene> sdata = ResourceLoader::load(p_files[i]);
		if (!sdata.is_valid()) {
			current_option = -1;
			accept->get_ok()->set_text(TTR("Ugh"));
			accept->set_text(vformat(TTR("Error loading scene from %s"), p_files[i]));
			accept->popup_centered_minsize();
			error = true;
			break;
		}

		Node *instanced_scene = sdata->instance(PackedScene::GEN_EDIT_STATE_INSTANCE);
		if (!instanced_scene) {
			current_option = -1;
			accept->get_ok()->set_text(TTR("Ugh"));
			accept->set_text(vformat(TTR("Error instancing scene from %s"), p_files[i]));
			accept->popup_centered_minsize();
			error = true;
			break;
		}

		if (edited_scene->get_filename() != "") {

			if (_cyclical_dependency_exists(edited_scene->get_filename(), instanced_scene)) {

				accept->get_ok()->set_text(TTR("Ok"));
				accept->set_text(vformat(TTR("Cannot instance the scene '%s' because the current scene exists within one of its nodes."), p_files[i]));
				accept->popup_centered_minsize();
				error = true;
				break;
			}
		}

		instanced_scene->set_filename(ProjectSettings::get_singleton()->localize_path(p_files[i]));

		instances.push_back(instanced_scene);
	}

	if (error) {
		for (int i = 0; i < instances.size(); i++) {
			memdelete(instances[i]);
		}
		return;
	}

	editor_data->get_undo_redo().create_action(TTR("Instance Scene(s)"));

	for (int i = 0; i < instances.size(); i++) {

		Node *instanced_scene = instances[i];

		editor_data->get_undo_redo().add_do_method(parent, "add_child", instanced_scene);
		if (p_pos >= 0) {
			editor_data->get_undo_redo().add_do_method(parent, "move_child", instanced_scene, p_pos + i);
		}
		editor_data->get_undo_redo().add_do_method(instanced_scene, "set_owner", edited_scene);
		editor_data->get_undo_redo().add_do_method(editor_selection, "clear");
		editor_data->get_undo_redo().add_do_method(editor_selection, "add_node", instanced_scene);
		editor_data->get_undo_redo().add_do_reference(instanced_scene);
		editor_data->get_undo_redo().add_undo_method(parent, "remove_child", instanced_scene);

		String new_name = parent->validate_child_name(instanced_scene);
		ScriptEditorDebugger *sed = ScriptEditor::get_singleton()->get_debugger();
		editor_data->get_undo_redo().add_do_method(sed, "live_debug_instance_node", edited_scene->get_path_to(parent), p_files[i], new_name);
		editor_data->get_undo_redo().add_undo_method(sed, "live_debug_remove_node", NodePath(String(edited_scene->get_path_to(parent)) + "/" + new_name));
	}

	editor_data->get_undo_redo().commit_action();
}

void SceneTreeDock::_replace_with_branch_scene(const String &p_file, Node *base) {
	Ref<PackedScene> sdata = ResourceLoader::load(p_file);
	if (!sdata.is_valid()) {
		accept->get_ok()->set_text(TTR("Ugh"));
		accept->set_text(vformat(TTR("Error loading scene from %s"), p_file));
		accept->popup_centered_minsize();
		return;
	}

	Node *instanced_scene = sdata->instance(PackedScene::GEN_EDIT_STATE_INSTANCE);
	if (!instanced_scene) {
		accept->get_ok()->set_text(TTR("Ugh"));
		accept->set_text(vformat(TTR("Error instancing scene from %s"), p_file));
		accept->popup_centered_minsize();
		return;
	}

	Node *parent = base->get_parent();
	int pos = base->get_index();
	memdelete(base);
	parent->add_child(instanced_scene);
	parent->move_child(instanced_scene, pos);
	instanced_scene->set_owner(edited_scene);
	editor_selection->clear();
	editor_selection->add_node(instanced_scene);
	scene_tree->set_selected(instanced_scene);
}

bool SceneTreeDock::_cyclical_dependency_exists(const String &p_target_scene_path, Node *p_desired_node) {
	int childCount = p_desired_node->get_child_count();

	if (p_desired_node->get_filename() == p_target_scene_path) {
		return true;
	}

	for (int i = 0; i < childCount; i++) {
		Node *child = p_desired_node->get_child(i);

		if (_cyclical_dependency_exists(p_target_scene_path, child)) {
			return true;
		}
	}

	return false;
}

void SceneTreeDock::_tool_selected(int p_tool, bool p_confirm_override) {

	current_option = p_tool;

	switch (p_tool) {

		case TOOL_NEW: {

			String preferred = "";
			Node *current_edited_scene_root = EditorNode::get_singleton()->get_edited_scene();

			if (current_edited_scene_root) {

				if (ClassDB::is_parent_class(current_edited_scene_root->get_class_name(), "Node2D"))
					preferred = "Node2D";
				else if (ClassDB::is_parent_class(current_edited_scene_root->get_class_name(), "Spatial"))
					preferred = "Spatial";
			}
			create_dialog->set_preferred_search_result_type(preferred);
			create_dialog->popup_create(true);
		} break;
		case TOOL_INSTANCE: {

			Node *scene = edited_scene;

			if (!scene) {
				EditorNode::get_singleton()->new_inherited_scene();
				break;
			}

			file->set_mode(EditorFileDialog::MODE_OPEN_FILE);
			List<String> extensions;
			ResourceLoader::get_recognized_extensions_for_type("PackedScene", &extensions);
			file->clear_filters();
			for (int i = 0; i < extensions.size(); i++) {

				file->add_filter("*." + extensions[i] + " ; " + extensions[i].to_upper());
			}

			file->popup_centered_ratio();

		} break;
		case TOOL_REPLACE: {

			create_dialog->popup_create(false);
		} break;
		case TOOL_ATTACH_SCRIPT: {

			Node *selected = scene_tree->get_selected();
			if (!selected)
				break;

			Ref<Script> existing = selected->get_script();
			if (existing.is_valid())
				editor->push_item(existing.ptr());
			else {
				String path = selected->get_filename();
				if (path == "") {
					String root_path = editor_data->get_edited_scene_root()->get_filename();
					if (root_path == "") {
						path = "res://" + selected->get_name();
					} else {
						path = root_path.get_base_dir() + "/" + selected->get_name();
					}
				}
				script_create_dialog->config(selected->get_class(), path);
				script_create_dialog->popup_centered();
			}

		} break;
		case TOOL_CLEAR_SCRIPT: {
			Node *selected = scene_tree->get_selected();
			if (!selected)
				break;

			Ref<Script> existing = selected->get_script();
			if (existing.is_valid()) {
				const RefPtr empty;
				selected->set_script(empty);
				button_create_script->show();
				button_clear_script->hide();
			}

		} break;
		case TOOL_MOVE_UP:
		case TOOL_MOVE_DOWN: {

			if (!scene_tree->get_selected())
				break;

			if (scene_tree->get_selected() == edited_scene) {

				current_option = -1;
				accept->get_ok()->set_text(TTR("I see.."));
				accept->set_text(TTR("This operation can't be done on the tree root."));
				accept->popup_centered_minsize();
				break;
			}

			if (!_validate_no_foreign())
				break;

			bool MOVING_DOWN = (p_tool == TOOL_MOVE_DOWN);
			bool MOVING_UP = !MOVING_DOWN;

			Node *common_parent = scene_tree->get_selected()->get_parent();
			List<Node *> selection = editor_selection->get_selected_node_list();
			selection.sort_custom<Node::Comparator>(); // sort by index
			if (MOVING_DOWN)
				selection.invert();

			int lowest_id = common_parent->get_child_count() - 1;
			int highest_id = 0;
			for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {
				int index = E->get()->get_index();

				if (index > highest_id) highest_id = index;
				if (index < lowest_id) lowest_id = index;

				if (E->get()->get_parent() != common_parent)
					common_parent = NULL;
			}

			if (!common_parent || (MOVING_DOWN && highest_id >= common_parent->get_child_count() - MOVING_DOWN) || (MOVING_UP && lowest_id == 0))
				break; // one or more nodes can not be moved

			if (selection.size() == 1) editor_data->get_undo_redo().create_action(TTR("Move Node In Parent"));
			if (selection.size() > 1) editor_data->get_undo_redo().create_action(TTR("Move Nodes In Parent"));

			for (int i = 0; i < selection.size(); i++) {
				Node *top_node = selection[i];
				Node *bottom_node = selection[selection.size() - 1 - i];

				ERR_FAIL_COND(!top_node->get_parent());
				ERR_FAIL_COND(!bottom_node->get_parent());

				int bottom_node_pos = bottom_node->get_index();
				int top_node_pos_next = top_node->get_index() + (MOVING_DOWN ? 1 : -1);

				editor_data->get_undo_redo().add_do_method(top_node->get_parent(), "move_child", top_node, top_node_pos_next);
				editor_data->get_undo_redo().add_undo_method(bottom_node->get_parent(), "move_child", bottom_node, bottom_node_pos);
			}

			editor_data->get_undo_redo().commit_action();

		} break;
		case TOOL_DUPLICATE: {

			if (!edited_scene)
				break;

			if (editor_selection->is_selected(edited_scene)) {

				current_option = -1;
				//accept->get_cancel()->hide();
				accept->get_ok()->set_text(TTR("I see.."));
				accept->set_text(TTR("This operation can't be done on the tree root."));
				accept->popup_centered_minsize();
				break;
			}

			if (!_validate_no_foreign())
				break;

			List<Node *> selection = editor_selection->get_selected_node_list();
			if (selection.size() == 0)
				break;

			editor_data->get_undo_redo().create_action(TTR("Duplicate Node(s)"));
			editor_data->get_undo_redo().add_do_method(editor_selection, "clear");

			Node *dupsingle = NULL;

			for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

				Node *node = E->get();
				Node *parent = node->get_parent();

				List<Node *> owned;
				node->get_owned_by(node->get_owner(), &owned);

				Map<const Node *, Node *> duplimap;
				Node *dup = node->duplicate_from_editor(duplimap);

				ERR_CONTINUE(!dup);

				if (selection.size() == 1)
					dupsingle = dup;

				dup->set_name(parent->validate_child_name(dup));

				editor_data->get_undo_redo().add_do_method(parent, "add_child_below_node", node, dup);
				for (List<Node *>::Element *F = owned.front(); F; F = F->next()) {

					if (!duplimap.has(F->get())) {

						continue;
					}
					Node *d = duplimap[F->get()];
					editor_data->get_undo_redo().add_do_method(d, "set_owner", node->get_owner());
				}
				editor_data->get_undo_redo().add_do_method(editor_selection, "add_node", dup);
				editor_data->get_undo_redo().add_undo_method(parent, "remove_child", dup);
				editor_data->get_undo_redo().add_do_reference(dup);

				ScriptEditorDebugger *sed = ScriptEditor::get_singleton()->get_debugger();

				editor_data->get_undo_redo().add_do_method(sed, "live_debug_duplicate_node", edited_scene->get_path_to(node), dup->get_name());
				editor_data->get_undo_redo().add_undo_method(sed, "live_debug_remove_node", NodePath(String(edited_scene->get_path_to(parent)) + "/" + dup->get_name()));
			}

			editor_data->get_undo_redo().commit_action();

			if (dupsingle)
				editor->push_item(dupsingle);

		} break;
		case TOOL_REPARENT: {

			if (!scene_tree->get_selected())
				break;

			if (editor_selection->is_selected(edited_scene)) {

				current_option = -1;
				accept->get_ok()->set_text(TTR("I see.."));
				accept->set_text(TTR("This operation can't be done on the tree root."));
				accept->popup_centered_minsize();
				break;
			}

			if (!_validate_no_foreign())
				break;

			List<Node *> nodes = editor_selection->get_selected_node_list();
			Set<Node *> nodeset;
			for (List<Node *>::Element *E = nodes.front(); E; E = E->next()) {

				nodeset.insert(E->get());
			}
			reparent_dialog->popup_centered_ratio();
			reparent_dialog->set_current(nodeset);

		} break;
		case TOOL_MULTI_EDIT: {

			Node *root = EditorNode::get_singleton()->get_edited_scene();
			if (!root)
				break;
			Ref<MultiNodeEdit> mne = memnew(MultiNodeEdit);
			for (const Map<Node *, Object *>::Element *E = EditorNode::get_singleton()->get_editor_selection()->get_selection().front(); E; E = E->next()) {
				mne->add_node(root->get_path_to(E->key()));
			}

			EditorNode::get_singleton()->push_item(mne.ptr());

		} break;

		case TOOL_ERASE: {

			List<Node *> remove_list = editor_selection->get_selected_node_list();

			if (remove_list.empty())
				return;

			if (!_validate_no_foreign())
				break;

			if (p_confirm_override) {
				_delete_confirm();

			} else {
				delete_dialog->set_text(TTR("Delete Node(s)?"));
				delete_dialog->popup_centered_minsize();
			}

		} break;
		case TOOL_MERGE_FROM_SCENE: {

			EditorNode::get_singleton()->merge_from_scene();
		} break;
		case TOOL_NEW_SCENE_FROM: {

			Node *scene = editor_data->get_edited_scene_root();

			if (!scene) {
				accept->get_ok()->set_text(TTR("I see.."));
				accept->set_text(TTR("This operation can't be done without a scene."));
				accept->popup_centered_minsize();
				break;
			}

			List<Node *> selection = editor_selection->get_selected_node_list();

			if (selection.size() != 1) {
				accept->get_ok()->set_text(TTR("I see.."));
				accept->set_text(TTR("This operation requires a single selected node."));
				accept->popup_centered_minsize();
				break;
			}

			Node *tocopy = selection.front()->get();

			if (tocopy == scene) {
				accept->get_ok()->set_text(TTR("I see.."));
				accept->set_text(TTR("Can not perform with the root node."));
				accept->popup_centered_minsize();
				break;
			}

			if (tocopy != editor_data->get_edited_scene_root() && tocopy->get_filename() != "") {
				accept->get_ok()->set_text(TTR("I see.."));
				accept->set_text(TTR("This operation can't be done on instanced scenes."));
				accept->popup_centered_minsize();
				break;
			}

			new_scene_from_dialog->set_mode(EditorFileDialog::MODE_SAVE_FILE);

			List<String> extensions;
			Ref<PackedScene> sd = memnew(PackedScene);
			ResourceSaver::get_recognized_extensions(sd, &extensions);
			new_scene_from_dialog->clear_filters();
			for (int i = 0; i < extensions.size(); i++) {
				new_scene_from_dialog->add_filter("*." + extensions[i] + " ; " + extensions[i].to_upper());
			}

			String existing;
			if (extensions.size()) {
				String root_name(tocopy->get_name());
				existing = root_name + "." + extensions.front()->get().to_lower();
			}
			new_scene_from_dialog->set_current_path(existing);

			new_scene_from_dialog->popup_centered_ratio();
			new_scene_from_dialog->set_title(TTR("Save New Scene As.."));
		} break;
		case TOOL_COPY_NODE_PATH: {
			List<Node *> selection = editor_selection->get_selected_node_list();
			List<Node *>::Element *e = selection.front();
			if (e) {
				Node *node = e->get();
				if (node) {
					Node *root = EditorNode::get_singleton()->get_edited_scene();
					NodePath path = root->get_path().rel_path_to(node->get_path());
					OS::get_singleton()->set_clipboard(path);
				}
			}
		} break;
		case TOOL_SCENE_EDITABLE_CHILDREN: {
			List<Node *> selection = editor_selection->get_selected_node_list();
			List<Node *>::Element *e = selection.front();
			if (e) {
				Node *node = e->get();
				if (node) {
					bool editable = EditorNode::get_singleton()->get_edited_scene()->is_editable_instance(node);
					int editable_item_idx = menu->get_item_idx_from_text(TTR("Editable Children"));
					int placeholder_item_idx = menu->get_item_idx_from_text(TTR("Load As Placeholder"));
					editable = !editable;

					EditorNode::get_singleton()->get_edited_scene()->set_editable_instance(node, editable);

					menu->set_item_checked(editable_item_idx, editable);
					if (editable) {
						node->set_scene_instance_load_placeholder(false);
						menu->set_item_checked(placeholder_item_idx, false);
					}
					scene_tree->update_tree();
				}
			}
		} break;
		case TOOL_SCENE_USE_PLACEHOLDER: {
			List<Node *> selection = editor_selection->get_selected_node_list();
			List<Node *>::Element *e = selection.front();
			if (e) {
				Node *node = e->get();
				if (node) {
					bool placeholder = node->get_scene_instance_load_placeholder();
					placeholder = !placeholder;
					int editable_item_idx = menu->get_item_idx_from_text(TTR("Editable Children"));
					int placeholder_item_idx = menu->get_item_idx_from_text(TTR("Load As Placeholder"));
					if (placeholder)
						EditorNode::get_singleton()->get_edited_scene()->set_editable_instance(node, false);

					node->set_scene_instance_load_placeholder(placeholder);
					menu->set_item_checked(editable_item_idx, false);
					menu->set_item_checked(placeholder_item_idx, placeholder);
					scene_tree->update_tree();
				}
			}
		} break;
		case TOOL_SCENE_CLEAR_INSTANCING: {
			List<Node *> selection = editor_selection->get_selected_node_list();
			List<Node *>::Element *e = selection.front();
			if (e) {
				Node *node = e->get();
				if (node) {
					Node *root = EditorNode::get_singleton()->get_edited_scene();
					UndoRedo *undo_redo = &editor_data->get_undo_redo();
					if (!root)
						break;

					ERR_FAIL_COND(node->get_filename() == String());
					undo_redo->create_action(TTR("Discard Instancing"));
					undo_redo->add_do_method(node, "set_filename", "");
					undo_redo->add_undo_method(node, "set_filename", node->get_filename());
					_node_replace_owner(node, node, root);
					undo_redo->add_do_method(scene_tree, "update_tree");
					undo_redo->add_undo_method(scene_tree, "update_tree");
					undo_redo->commit_action();
				}
			}
		} break;
		case TOOL_SCENE_OPEN: {
			List<Node *> selection = editor_selection->get_selected_node_list();
			List<Node *>::Element *e = selection.front();
			if (e) {
				Node *node = e->get();
				if (node) {
					scene_tree->emit_signal("open", node->get_filename());
				}
			}
		} break;
		case TOOL_SCENE_CLEAR_INHERITANCE: {
			clear_inherit_confirm->popup_centered_minsize();
		} break;
		case TOOL_SCENE_CLEAR_INHERITANCE_CONFIRM: {
			List<Node *> selection = editor_selection->get_selected_node_list();
			List<Node *>::Element *e = selection.front();
			if (e) {
				Node *node = e->get();
				if (node) {
					node->set_scene_inherited_state(Ref<SceneState>());
					scene_tree->update_tree();
					EditorNode::get_singleton()->get_property_editor()->update_tree();
				}
			}
		} break;
		case TOOL_SCENE_OPEN_INHERITED: {
			List<Node *> selection = editor_selection->get_selected_node_list();
			List<Node *>::Element *e = selection.front();
			if (e) {
				Node *node = e->get();
				if (node) {
					if (node && node->get_scene_inherited_state().is_valid()) {
						scene_tree->emit_signal("open", node->get_scene_inherited_state()->get_path());
					}
				}
			}
		} break;
		default: {

			if (p_tool >= EDIT_SUBRESOURCE_BASE) {

				int idx = p_tool - EDIT_SUBRESOURCE_BASE;

				ERR_FAIL_INDEX(idx, subresources.size());

				Object *obj = ObjectDB::get_instance(subresources[idx]);
				ERR_FAIL_COND(!obj);

				editor->push_item(obj);
			}
		}
	}
}

void SceneTreeDock::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_READY: {

			if (!first_enter)
				break;
			first_enter = false;

			CanvasItemEditorPlugin *canvas_item_plugin = Object::cast_to<CanvasItemEditorPlugin>(editor_data->get_editor("2D"));
			if (canvas_item_plugin) {
				canvas_item_plugin->get_canvas_item_editor()->connect("item_lock_status_changed", scene_tree, "_update_tree");
				canvas_item_plugin->get_canvas_item_editor()->connect("item_group_status_changed", scene_tree, "_update_tree");
				scene_tree->connect("node_changed", canvas_item_plugin->get_canvas_item_editor()->get_viewport_control(), "update");
			}

			SpatialEditorPlugin *spatial_editor_plugin = Object::cast_to<SpatialEditorPlugin>(editor_data->get_editor("3D"));
			spatial_editor_plugin->get_spatial_editor()->connect("item_lock_status_changed", scene_tree, "_update_tree");

			button_add->set_icon(get_icon("Add", "EditorIcons"));
			button_instance->set_icon(get_icon("Instance", "EditorIcons"));
			button_create_script->set_icon(get_icon("ScriptCreate", "EditorIcons"));
			button_clear_script->set_icon(get_icon("ScriptRemove", "EditorIcons"));

			filter->add_icon_override("right_icon", get_icon("Search", "EditorIcons"));

			EditorNode::get_singleton()->get_editor_selection()->connect("selection_changed", this, "_selection_changed");

		} break;

		case NOTIFICATION_ENTER_TREE: {
			clear_inherit_confirm->connect("confirmed", this, "_tool_selected", varray(TOOL_SCENE_CLEAR_INHERITANCE_CONFIRM));
		} break;

		case NOTIFICATION_EXIT_TREE: {
			clear_inherit_confirm->disconnect("confirmed", this, "_tool_selected");
		} break;
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			button_add->set_icon(get_icon("Add", "EditorIcons"));
			button_instance->set_icon(get_icon("Instance", "EditorIcons"));
			button_create_script->set_icon(get_icon("ScriptCreate", "EditorIcons"));
			button_clear_script->set_icon(get_icon("ScriptRemove", "EditorIcons"));

			filter->add_icon_override("right_icon", get_icon("Search", "EditorIcons"));
		} break;
	}
}

void SceneTreeDock::_node_replace_owner(Node *p_base, Node *p_node, Node *p_root) {

	if (p_base != p_node) {
		if (p_node->get_owner() == p_base) {
			UndoRedo *undo_redo = &editor_data->get_undo_redo();
			undo_redo->add_do_method(p_node, "set_owner", p_root);
			undo_redo->add_undo_method(p_node, "set_owner", p_base);
		}
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		_node_replace_owner(p_base, p_node->get_child(i), p_root);
	}
}

void SceneTreeDock::_load_request(const String &p_path) {

	editor->open_request(p_path);
}

void SceneTreeDock::_script_open_request(const Ref<Script> &p_script) {

	editor->edit_resource(p_script);
}

void SceneTreeDock::_node_selected() {

	Node *node = scene_tree->get_selected();

	if (!node) {

		editor->push_item(NULL);
		return;
	}

	if (ScriptEditor::get_singleton()->is_visible_in_tree()) {
		restore_script_editor_on_drag = true;
	}

	editor->push_item(node);
}

void SceneTreeDock::_node_renamed() {

	_node_selected();
}

void SceneTreeDock::_set_owners(Node *p_owner, const Array &p_nodes) {

	for (int i = 0; i < p_nodes.size(); i++) {

		Node *n = Object::cast_to<Node>(p_nodes[i]);
		if (!n)
			continue;
		n->set_owner(p_owner);
	}
}

void SceneTreeDock::_fill_path_renames(Vector<StringName> base_path, Vector<StringName> new_base_path, Node *p_node, List<Pair<NodePath, NodePath> > *p_renames) {

	base_path.push_back(p_node->get_name());
	if (new_base_path.size())
		new_base_path.push_back(p_node->get_name());

	NodePath from(base_path, true);
	NodePath to;
	if (new_base_path.size())
		to = NodePath(new_base_path, true);

	Pair<NodePath, NodePath> npp;
	npp.first = from;
	npp.second = to;

	p_renames->push_back(npp);

	for (int i = 0; i < p_node->get_child_count(); i++) {

		_fill_path_renames(base_path, new_base_path, p_node->get_child(i), p_renames);
	}
}

void SceneTreeDock::fill_path_renames(Node *p_node, Node *p_new_parent, List<Pair<NodePath, NodePath> > *p_renames) {

	if (!bool(EDITOR_DEF("editors/animation/autorename_animation_tracks", true)))
		return;

	Vector<StringName> base_path;
	Node *n = p_node->get_parent();
	while (n) {
		base_path.push_back(n->get_name());
		n = n->get_parent();
	}
	base_path.invert();

	Vector<StringName> new_base_path;
	if (p_new_parent) {
		n = p_new_parent;
		while (n) {
			new_base_path.push_back(n->get_name());
			n = n->get_parent();
		}

		new_base_path.invert();
	}

	_fill_path_renames(base_path, new_base_path, p_node, p_renames);
}

void SceneTreeDock::perform_node_renames(Node *p_base, List<Pair<NodePath, NodePath> > *p_renames, Map<Ref<Animation>, Set<int> > *r_rem_anims) {

	Map<Ref<Animation>, Set<int> > rem_anims;

	if (!r_rem_anims)
		r_rem_anims = &rem_anims;

	if (!bool(EDITOR_DEF("editors/animation/autorename_animation_tracks", true)))
		return;

	if (!p_base) {

		p_base = edited_scene;
	}

	if (!p_base)
		return;

	if (Object::cast_to<AnimationPlayer>(p_base)) {

		AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(p_base);
		List<StringName> anims;
		ap->get_animation_list(&anims);
		Node *root = ap->get_node(ap->get_root());

		if (root) {

			NodePath root_path = root->get_path();
			NodePath new_root_path = root_path;

			for (List<Pair<NodePath, NodePath> >::Element *E = p_renames->front(); E; E = E->next()) {

				if (E->get().first == root_path) {
					new_root_path = E->get().second;
					break;
				}
			}

			if (new_root_path != NodePath()) {
				//will not be erased

				for (List<StringName>::Element *E = anims.front(); E; E = E->next()) {

					Ref<Animation> anim = ap->get_animation(E->get());
					if (!r_rem_anims->has(anim)) {
						r_rem_anims->insert(anim, Set<int>());
						Set<int> &ran = r_rem_anims->find(anim)->get();
						for (int i = 0; i < anim->get_track_count(); i++)
							ran.insert(i);
					}

					Set<int> &ran = r_rem_anims->find(anim)->get();

					if (anim.is_null())
						continue;

					for (int i = 0; i < anim->get_track_count(); i++) {

						NodePath track_np = anim->track_get_path(i);
						Node *n = root->get_node(track_np);
						if (!n) {
							continue;
						}

						NodePath old_np = n->get_path();

						if (!ran.has(i))
							continue; //channel was removed

						for (List<Pair<NodePath, NodePath> >::Element *E = p_renames->front(); E; E = E->next()) {

							if (E->get().first == old_np) {

								if (E->get().second == NodePath()) {
									//will be erased

									int idx = 0;
									Set<int>::Element *EI = ran.front();
									ERR_FAIL_COND(!EI); //bug
									while (EI->get() != i) {
										idx++;
										EI = EI->next();
										ERR_FAIL_COND(!EI); //another bug
									}

									editor_data->get_undo_redo().add_do_method(anim.ptr(), "remove_track", idx);
									editor_data->get_undo_redo().add_undo_method(anim.ptr(), "add_track", anim->track_get_type(i), idx);
									editor_data->get_undo_redo().add_undo_method(anim.ptr(), "track_set_path", idx, track_np);
									editor_data->get_undo_redo().add_undo_method(anim.ptr(), "track_set_interpolation_type", idx, anim->track_get_interpolation_type(i));
									for (int j = 0; j < anim->track_get_key_count(i); j++) {

										editor_data->get_undo_redo().add_undo_method(anim.ptr(), "track_insert_key", idx, anim->track_get_key_time(i, j), anim->track_get_key_value(i, j), anim->track_get_key_transition(i, j));
									}

									ran.erase(i); //byebye channel

								} else {
									//will be renamed
									NodePath rel_path = new_root_path.rel_path_to(E->get().second);

									NodePath new_path = NodePath(rel_path.get_names(), track_np.get_subnames(), false);
									if (new_path == track_np)
										continue; //bleh
									editor_data->get_undo_redo().add_do_method(anim.ptr(), "track_set_path", i, new_path);
									editor_data->get_undo_redo().add_undo_method(anim.ptr(), "track_set_path", i, track_np);
								}
							}
						}
					}
				}
			}
		}
	}

	for (int i = 0; i < p_base->get_child_count(); i++)
		perform_node_renames(p_base->get_child(i), p_renames, r_rem_anims);
}

void SceneTreeDock::_node_prerenamed(Node *p_node, const String &p_new_name) {

	List<Pair<NodePath, NodePath> > path_renames;

	Vector<StringName> base_path;
	Node *n = p_node->get_parent();
	while (n) {
		base_path.push_back(n->get_name());
		n = n->get_parent();
	}
	base_path.invert();

	Vector<StringName> new_base_path = base_path;
	base_path.push_back(p_node->get_name());

	new_base_path.push_back(p_new_name);

	Pair<NodePath, NodePath> npp;
	npp.first = NodePath(base_path, true);
	npp.second = NodePath(new_base_path, true);
	path_renames.push_back(npp);

	for (int i = 0; i < p_node->get_child_count(); i++)
		_fill_path_renames(base_path, new_base_path, p_node->get_child(i), &path_renames);

	perform_node_renames(NULL, &path_renames);
}

bool SceneTreeDock::_validate_no_foreign() {

	List<Node *> selection = editor_selection->get_selected_node_list();

	for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

		if (E->get() != edited_scene && E->get()->get_owner() != edited_scene) {

			accept->get_ok()->set_text(TTR("Makes Sense!"));
			accept->set_text(TTR("Can't operate on nodes from a foreign scene!"));
			accept->popup_centered_minsize();
			return false;
		}

		if (edited_scene->get_scene_inherited_state().is_valid() && edited_scene->get_scene_inherited_state()->find_node_by_path(edited_scene->get_path_to(E->get())) >= 0) {

			accept->get_ok()->set_text(TTR("Makes Sense!"));
			accept->set_text(TTR("Can't operate on nodes the current scene inherits from!"));
			accept->popup_centered_minsize();
			return false;
		}
	}

	return true;
}

void SceneTreeDock::_node_reparent(NodePath p_path, bool p_keep_global_xform) {

	Node *new_parent = scene_root->get_node(p_path);
	ERR_FAIL_COND(!new_parent);

	List<Node *> selection = editor_selection->get_selected_node_list();

	if (selection.empty())
		return; //nothing to reparent

	Vector<Node *> nodes;

	for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {
		nodes.push_back(E->get());
	}

	_do_reparent(new_parent, -1, nodes, p_keep_global_xform);
}

void SceneTreeDock::_do_reparent(Node *p_new_parent, int p_position_in_parent, Vector<Node *> p_nodes, bool p_keep_global_xform) {

	Node *new_parent = p_new_parent;
	ERR_FAIL_COND(!new_parent);

	Node *validate = new_parent;
	while (validate) {

		if (p_nodes.find(validate) != -1) {
			ERR_EXPLAIN("Selection changed at some point.. can't reparent");
			ERR_FAIL();
			return;
		}
		validate = validate->get_parent();
	}
	//ok all valid

	if (p_nodes.size() == 0)
		return; //nothing to reparent

	//sort by tree order, so re-adding is easy
	p_nodes.sort_custom<Node::Comparator>();

	editor_data->get_undo_redo().create_action(TTR("Reparent Node"));

	List<Pair<NodePath, NodePath> > path_renames;
	Vector<StringName> former_names;

	int inc = 0;

	for (int ni = 0; ni < p_nodes.size(); ni++) {

		//no undo for now, sorry
		Node *node = p_nodes[ni];

		fill_path_renames(node, new_parent, &path_renames);
		former_names.push_back(node->get_name());

		List<Node *> owned;
		node->get_owned_by(node->get_owner(), &owned);
		Array owners;
		for (List<Node *>::Element *E = owned.front(); E; E = E->next()) {

			owners.push_back(E->get());
		}

		if (new_parent == node->get_parent() && node->get_index() < p_position_in_parent + ni) {
			//if child will generate a gap when moved, adjust
			inc--;
		}

		editor_data->get_undo_redo().add_do_method(node->get_parent(), "remove_child", node);
		editor_data->get_undo_redo().add_do_method(new_parent, "add_child", node);

		if (p_position_in_parent >= 0)
			editor_data->get_undo_redo().add_do_method(new_parent, "move_child", node, p_position_in_parent + inc);

		ScriptEditorDebugger *sed = ScriptEditor::get_singleton()->get_debugger();
		String new_name = new_parent->validate_child_name(node);
		editor_data->get_undo_redo().add_do_method(sed, "live_debug_reparent_node", edited_scene->get_path_to(node), edited_scene->get_path_to(new_parent), new_name, -1);
		editor_data->get_undo_redo().add_undo_method(sed, "live_debug_reparent_node", NodePath(String(edited_scene->get_path_to(new_parent)) + "/" + new_name), edited_scene->get_path_to(node->get_parent()), node->get_name(), node->get_index());

		if (p_keep_global_xform) {
			if (Object::cast_to<Node2D>(node))
				editor_data->get_undo_redo().add_do_method(node, "set_global_transform", Object::cast_to<Node2D>(node)->get_global_transform());
			if (Object::cast_to<Spatial>(node))
				editor_data->get_undo_redo().add_do_method(node, "set_global_transform", Object::cast_to<Spatial>(node)->get_global_transform());
			if (Object::cast_to<Control>(node))
				editor_data->get_undo_redo().add_do_method(node, "set_global_position", Object::cast_to<Control>(node)->get_global_position());
		}

		editor_data->get_undo_redo().add_do_method(this, "_set_owners", edited_scene, owners);

		if (AnimationPlayerEditor::singleton->get_key_editor()->get_root() == node)
			editor_data->get_undo_redo().add_do_method(AnimationPlayerEditor::singleton->get_key_editor(), "set_root", node);

		editor_data->get_undo_redo().add_undo_method(new_parent, "remove_child", node);
		editor_data->get_undo_redo().add_undo_method(node, "set_name", former_names[ni]);

		inc++;
	}

	//add and move in a second step.. (so old order is preserved)

	for (int ni = 0; ni < p_nodes.size(); ni++) {

		Node *node = p_nodes[ni];

		List<Node *> owned;
		node->get_owned_by(node->get_owner(), &owned);
		Array owners;
		for (List<Node *>::Element *E = owned.front(); E; E = E->next()) {

			owners.push_back(E->get());
		}

		int child_pos = node->get_position_in_parent();

		editor_data->get_undo_redo().add_undo_method(node->get_parent(), "add_child", node);
		editor_data->get_undo_redo().add_undo_method(node->get_parent(), "move_child", node, child_pos);
		editor_data->get_undo_redo().add_undo_method(this, "_set_owners", edited_scene, owners);
		if (AnimationPlayerEditor::singleton->get_key_editor()->get_root() == node)
			editor_data->get_undo_redo().add_undo_method(AnimationPlayerEditor::singleton->get_key_editor(), "set_root", node);

		if (p_keep_global_xform) {
			if (Object::cast_to<Node2D>(node))
				editor_data->get_undo_redo().add_undo_method(node, "set_transform", Object::cast_to<Node2D>(node)->get_transform());
			if (Object::cast_to<Spatial>(node))
				editor_data->get_undo_redo().add_undo_method(node, "set_transform", Object::cast_to<Spatial>(node)->get_transform());
			if (Object::cast_to<Control>(node))
				editor_data->get_undo_redo().add_undo_method(node, "set_position", Object::cast_to<Control>(node)->get_position());
		}
	}

	perform_node_renames(NULL, &path_renames);

	editor_data->get_undo_redo().commit_action();
}

void SceneTreeDock::_script_created(Ref<Script> p_script) {

	Node *selected = scene_tree->get_selected();
	if (!selected)
		return;
	selected->set_script(p_script.get_ref_ptr());
	editor->push_item(p_script.operator->());
	button_create_script->hide();
	button_clear_script->show();
}

void SceneTreeDock::_delete_confirm() {

	List<Node *> remove_list = editor_selection->get_selected_node_list();

	if (remove_list.empty())
		return;

	editor->get_editor_plugins_over()->make_visible(false);

	editor_data->get_undo_redo().create_action(TTR("Remove Node(s)"));

	bool entire_scene = false;

	for (List<Node *>::Element *E = remove_list.front(); E; E = E->next()) {

		if (E->get() == edited_scene) {
			entire_scene = true;
		}
	}

	if (entire_scene) {

		editor_data->get_undo_redo().add_do_method(editor, "set_edited_scene", (Object *)NULL);
		editor_data->get_undo_redo().add_undo_method(editor, "set_edited_scene", edited_scene);
		editor_data->get_undo_redo().add_undo_method(edited_scene, "set_owner", edited_scene->get_owner());
		editor_data->get_undo_redo().add_undo_method(scene_tree, "update_tree");
		editor_data->get_undo_redo().add_undo_reference(edited_scene);

	} else {

		remove_list.sort_custom<Node::Comparator>(); //sort nodes to keep positions
		List<Pair<NodePath, NodePath> > path_renames;

		//delete from animation
		for (List<Node *>::Element *E = remove_list.front(); E; E = E->next()) {
			Node *n = E->get();
			if (!n->is_inside_tree() || !n->get_parent())
				continue;

			fill_path_renames(n, NULL, &path_renames);
		}

		perform_node_renames(NULL, &path_renames);
		//delete for read
		for (List<Node *>::Element *E = remove_list.front(); E; E = E->next()) {
			Node *n = E->get();
			if (!n->is_inside_tree() || !n->get_parent())
				continue;

			List<Node *> owned;
			n->get_owned_by(n->get_owner(), &owned);
			Array owners;
			for (List<Node *>::Element *E = owned.front(); E; E = E->next()) {

				owners.push_back(E->get());
			}

			editor_data->get_undo_redo().add_do_method(n->get_parent(), "remove_child", n);
			editor_data->get_undo_redo().add_undo_method(n->get_parent(), "add_child", n);
			editor_data->get_undo_redo().add_undo_method(n->get_parent(), "move_child", n, n->get_index());
			if (AnimationPlayerEditor::singleton->get_key_editor()->get_root() == n)
				editor_data->get_undo_redo().add_undo_method(AnimationPlayerEditor::singleton->get_key_editor(), "set_root", n);
			editor_data->get_undo_redo().add_undo_method(this, "_set_owners", edited_scene, owners);
			editor_data->get_undo_redo().add_undo_reference(n);

			ScriptEditorDebugger *sed = ScriptEditor::get_singleton()->get_debugger();
			editor_data->get_undo_redo().add_do_method(sed, "live_debug_remove_and_keep_node", edited_scene->get_path_to(n), n->get_instance_id());
			editor_data->get_undo_redo().add_undo_method(sed, "live_debug_restore_node", n->get_instance_id(), edited_scene->get_path_to(n->get_parent()), n->get_index());
		}
	}
	editor_data->get_undo_redo().commit_action();

	// hack, force 2d editor viewport to refresh after deletion
	if (CanvasItemEditor *editor = CanvasItemEditor::get_singleton())
		editor->get_viewport_control()->update();

	editor->push_item(NULL);
}

void SceneTreeDock::_selection_changed() {

	int selection_size = EditorNode::get_singleton()->get_editor_selection()->get_selection().size();
	if (selection_size > 1) {
		//automatically turn on multi-edit
		_tool_selected(TOOL_MULTI_EDIT);
	}

	if (selection_size == 1) {
		if (EditorNode::get_singleton()->get_editor_selection()->get_selection().front()->key()->get_script().is_null()) {
			button_create_script->show();
			button_clear_script->hide();
		} else {
			button_create_script->hide();
			button_clear_script->show();
		}
	} else {
		button_create_script->hide();
		button_clear_script->hide();
	}
}

void SceneTreeDock::_create() {

	if (current_option == TOOL_NEW) {

		Node *parent = NULL;

		if (edited_scene) {
			// If root exists in edited scene
			parent = scene_tree->get_selected();
			if (!parent)
				parent = edited_scene;

		} else {
			// If no root exist in edited scene
			parent = scene_root;
			ERR_FAIL_COND(!parent);
		}

		Object *c = create_dialog->instance_selected();

		ERR_FAIL_COND(!c);
		Node *child = Object::cast_to<Node>(c);
		ERR_FAIL_COND(!child);

		editor_data->get_undo_redo().create_action(TTR("Create Node"));

		if (edited_scene) {

			editor_data->get_undo_redo().add_do_method(parent, "add_child", child);
			editor_data->get_undo_redo().add_do_method(child, "set_owner", edited_scene);
			editor_data->get_undo_redo().add_do_method(editor_selection, "clear");
			editor_data->get_undo_redo().add_do_method(editor_selection, "add_node", child);
			editor_data->get_undo_redo().add_do_reference(child);
			editor_data->get_undo_redo().add_undo_method(parent, "remove_child", child);

			String new_name = parent->validate_child_name(child);
			ScriptEditorDebugger *sed = ScriptEditor::get_singleton()->get_debugger();
			editor_data->get_undo_redo().add_do_method(sed, "live_debug_create_node", edited_scene->get_path_to(parent), child->get_class(), new_name);
			editor_data->get_undo_redo().add_undo_method(sed, "live_debug_remove_node", NodePath(String(edited_scene->get_path_to(parent)) + "/" + new_name));

		} else {

			editor_data->get_undo_redo().add_do_method(editor, "set_edited_scene", child);
			editor_data->get_undo_redo().add_do_method(scene_tree, "update_tree");
			editor_data->get_undo_redo().add_do_reference(child);
			editor_data->get_undo_redo().add_undo_method(editor, "set_edited_scene", (Object *)NULL);
		}

		editor_data->get_undo_redo().commit_action();
		editor->push_item(c);
		editor_selection->clear();
		if (Object::cast_to<Control>(c)) {
			//make editor more comfortable, so some controls don't appear super shrunk
			Control *ct = Object::cast_to<Control>(c);

			Size2 ms = ct->get_minimum_size();
			if (ms.width < 4)
				ms.width = 40;
			if (ms.height < 4)
				ms.height = 40;
			ct->set_size(ms);
		}

	} else if (current_option == TOOL_REPLACE) {
		List<Node *> selection = editor_selection->get_selected_node_list();
		ERR_FAIL_COND(selection.size() <= 0);
		for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {
			Node *n = E->get();
			ERR_FAIL_COND(!n);

			Object *c = create_dialog->instance_selected();

			ERR_FAIL_COND(!c);
			Node *newnode = Object::cast_to<Node>(c);
			ERR_FAIL_COND(!newnode);

			List<PropertyInfo> pinfo;
			n->get_property_list(&pinfo);

			for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {
				if (!(E->get().usage & PROPERTY_USAGE_STORAGE))
					continue;
				if (E->get().name == "__meta__")
					continue;
				newnode->set(E->get().name, n->get(E->get().name));
			}

			editor->push_item(NULL);

			//reconnect signals
			List<MethodInfo> sl;

			n->get_signal_list(&sl);
			for (List<MethodInfo>::Element *E = sl.front(); E; E = E->next()) {

				List<Object::Connection> cl;
				n->get_signal_connection_list(E->get().name, &cl);

				for (List<Object::Connection>::Element *F = cl.front(); F; F = F->next()) {

					Object::Connection &c = F->get();
					if (!(c.flags & Object::CONNECT_PERSIST))
						continue;
					newnode->connect(c.signal, c.target, c.method, varray(), Object::CONNECT_PERSIST);
				}
			}

			String newname = n->get_name();

			List<Node *> to_erase;
			for (int i = 0; i < n->get_child_count(); i++) {
				if (n->get_child(i)->get_owner() == NULL && n->is_owned_by_parent()) {
					to_erase.push_back(n->get_child(i));
				}
			}
			n->replace_by(newnode, true);

			if (n == edited_scene) {
				edited_scene = newnode;
				editor->set_edited_scene(newnode);
				newnode->set_editable_instances(n->get_editable_instances());
			}

			//small hack to make collisionshapes and other kind of nodes to work
			for (int i = 0; i < newnode->get_child_count(); i++) {
				Node *c = newnode->get_child(i);
				c->call("set_transform", c->call("get_transform"));
			}
			editor_data->get_undo_redo().clear_history();
			newnode->set_name(newname);

			editor->push_item(newnode);

			memdelete(n);

			while (to_erase.front()) {
				memdelete(to_erase.front()->get());
				to_erase.pop_front();
			}
		}
	}
}

void SceneTreeDock::set_edited_scene(Node *p_scene) {

	edited_scene = p_scene;
}

void SceneTreeDock::set_selected(Node *p_node, bool p_emit_selected) {

	scene_tree->set_selected(p_node, p_emit_selected);
}

void SceneTreeDock::import_subscene() {

	import_subscene_dialog->popup_centered_ratio();
}

void SceneTreeDock::_import_subscene() {

	Node *parent = scene_tree->get_selected();
	if (!parent) {
		parent = editor_data->get_edited_scene_root();
		ERR_FAIL_COND(!parent);
	}

	import_subscene_dialog->move(parent, edited_scene);
	editor_data->get_undo_redo().clear_history(); //no undo for now..
}

void SceneTreeDock::_new_scene_from(String p_file) {

	List<Node *> selection = editor_selection->get_selected_node_list();

	if (selection.size() != 1) {
		accept->get_ok()->set_text(TTR("I see.."));
		accept->set_text(TTR("This operation requires a single selected node."));
		accept->popup_centered_minsize();
		return;
	}

	Node *base = selection.front()->get();

	Map<Node *, Node *> reown;
	reown[editor_data->get_edited_scene_root()] = base;
	Node *copy = base->duplicate_and_reown(reown);
	if (copy) {

		Ref<PackedScene> sdata = memnew(PackedScene);
		Error err = sdata->pack(copy);
		memdelete(copy);

		if (err != OK) {
			accept->get_ok()->set_text(TTR("I see.."));
			accept->set_text(TTR("Couldn't save new scene. Likely dependencies (instances) couldn't be satisfied."));
			accept->popup_centered_minsize();
			return;
		}

		int flg = 0;
		if (EditorSettings::get_singleton()->get("filesystem/on_save/compress_binary_resources"))
			flg |= ResourceSaver::FLAG_COMPRESS;

		err = ResourceSaver::save(p_file, sdata, flg);
		if (err != OK) {
			accept->get_ok()->set_text(TTR("I see.."));
			accept->set_text(TTR("Error saving scene."));
			accept->popup_centered_minsize();
			return;
		}
		_replace_with_branch_scene(p_file, base);
	} else {
		accept->get_ok()->set_text(TTR("I see.."));
		accept->set_text(TTR("Error duplicating scene to save it."));
		accept->popup_centered_minsize();
		return;
	}
}

static bool _is_node_visible(Node *p_node) {

	if (!p_node->get_owner())
		return false;
	if (p_node->get_owner() != EditorNode::get_singleton()->get_edited_scene() && !EditorNode::get_singleton()->get_edited_scene()->is_editable_instance(p_node->get_owner()))
		return false;

	return true;
}

static bool _has_visible_children(Node *p_node) {

	bool collapsed = p_node->is_displayed_folded();
	if (collapsed)
		return false;

	for (int i = 0; i < p_node->get_child_count(); i++) {

		Node *child = p_node->get_child(i);
		if (!_is_node_visible(child))
			continue;

		return true;
	}

	return false;
}

static Node *_find_last_visible(Node *p_node) {

	Node *last = NULL;

	bool collapsed = p_node->is_displayed_folded();

	if (!collapsed) {
		for (int i = 0; i < p_node->get_child_count(); i++) {
			if (_is_node_visible(p_node->get_child(i))) {
				last = p_node->get_child(i);
			}
		}
	}

	if (last) {
		Node *lastc = _find_last_visible(last);
		if (lastc)
			last = lastc;

	} else {
		last = p_node;
	}

	return last;
}

void SceneTreeDock::_normalize_drop(Node *&to_node, int &to_pos, int p_type) {

	to_pos = -1;

	if (p_type == -1) {
		//drop at above selected node
		if (to_node == EditorNode::get_singleton()->get_edited_scene()) {
			to_node = NULL;
			ERR_EXPLAIN("Cannot perform drop above the root node!");
			ERR_FAIL();
		}

		to_pos = to_node->get_index();
		to_node = to_node->get_parent();

	} else if (p_type == 1) {
		//drop at below selected node
		if (to_node == EditorNode::get_singleton()->get_edited_scene()) {
			//if at lower sibling of root node
			to_pos = 0; //just insert at beginning of root node
			return;
		}

		Node *lower_sibling = NULL;

		if (_has_visible_children(to_node)) {
			to_pos = 0;
		} else {
			for (int i = to_node->get_index() + 1; i < to_node->get_parent()->get_child_count(); i++) {
				Node *c = to_node->get_parent()->get_child(i);
				if (_is_node_visible(c)) {
					lower_sibling = c;
					break;
				}
			}
			if (lower_sibling) {
				to_pos = lower_sibling->get_index();
			}

			to_node = to_node->get_parent();
		}
	}
}

void SceneTreeDock::_files_dropped(Vector<String> p_files, NodePath p_to, int p_type) {

	Node *node = get_node(p_to);
	ERR_FAIL_COND(!node);

	int to_pos = -1;
	_normalize_drop(node, to_pos, p_type);
	_perform_instance_scenes(p_files, node, to_pos);
}

void SceneTreeDock::_script_dropped(String p_file, NodePath p_to) {
	Ref<Script> scr = ResourceLoader::load(p_file);
	ERR_FAIL_COND(!scr.is_valid());
	Node *n = get_node(p_to);
	if (n) {
		n->set_script(scr.get_ref_ptr());
	}
}

void SceneTreeDock::_nodes_dragged(Array p_nodes, NodePath p_to, int p_type) {

	Vector<Node *> nodes;
	Node *to_node;

	for (int i = 0; i < p_nodes.size(); i++) {
		Node *n = get_node((p_nodes[i]));
		if (n) {
			nodes.push_back(n);
		}
	}

	if (nodes.size() == 0)
		return;

	to_node = get_node(p_to);
	if (!to_node)
		return;

	int to_pos = -1;

	_normalize_drop(to_node, to_pos, p_type);
	_do_reparent(to_node, to_pos, nodes, true);
}

void SceneTreeDock::_add_children_to_popup(Object *p_obj, int p_depth) {

	if (p_depth > 8)
		return;

	List<PropertyInfo> pinfo;
	p_obj->get_property_list(&pinfo);
	for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {

		if (!(E->get().usage & PROPERTY_USAGE_EDITOR))
			continue;
		if (E->get().hint != PROPERTY_HINT_RESOURCE_TYPE)
			continue;

		Variant value = p_obj->get(E->get().name);
		if (value.get_type() != Variant::OBJECT)
			continue;
		Object *obj = value;
		if (!obj)
			continue;

		Ref<Texture> icon;

		if (has_icon(obj->get_class(), "EditorIcons"))
			icon = get_icon(obj->get_class(), "EditorIcons");
		else
			icon = get_icon("Object", "EditorIcons");

		if (menu->get_item_count() == 0) {
			menu->add_submenu_item(TTR("Sub-Resources:"), "Sub-Resources");
		}
		int index = menu_subresources->get_item_count();
		menu_subresources->add_icon_item(icon, E->get().name.capitalize(), EDIT_SUBRESOURCE_BASE + subresources.size());
		menu_subresources->set_item_h_offset(index, p_depth * 10 * EDSCALE);
		subresources.push_back(obj->get_instance_id());

		_add_children_to_popup(obj, p_depth + 1);
	}
}

void SceneTreeDock::_tree_rmb(const Vector2 &p_menu_pos) {
	if (!EditorNode::get_singleton()->get_edited_scene()) {

		menu->clear();
		menu->add_icon_shortcut(get_icon("Add", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/add_child_node"), TOOL_NEW);
		menu->add_icon_shortcut(get_icon("Instance", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/instance_scene"), TOOL_INSTANCE);

		menu->set_size(Size2(1, 1));
		menu->set_position(p_menu_pos);
		menu->popup();
		return;
	}

	List<Node *> selection = editor_selection->get_selected_node_list();

	if (selection.size() == 0)
		return;

	menu->clear();

	if (selection.size() == 1) {

		subresources.clear();
		_add_children_to_popup(selection.front()->get(), 0);
		if (menu->get_item_count() > 0)
			menu->add_separator();

		menu->add_icon_shortcut(get_icon("Add", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/add_child_node"), TOOL_NEW);
		menu->add_icon_shortcut(get_icon("Instance", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/instance_scene"), TOOL_INSTANCE);
		menu->add_separator();
		menu->add_icon_shortcut(get_icon("ScriptCreate", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/attach_script"), TOOL_ATTACH_SCRIPT);
		menu->add_icon_shortcut(get_icon("ScriptRemove", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/clear_script"), TOOL_CLEAR_SCRIPT);
		menu->add_separator();
	}
	menu->add_icon_shortcut(get_icon("Reload", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/change_node_type"), TOOL_REPLACE);
	menu->add_separator();
	menu->add_icon_shortcut(get_icon("MoveUp", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/move_up"), TOOL_MOVE_UP);
	menu->add_icon_shortcut(get_icon("MoveDown", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/move_down"), TOOL_MOVE_DOWN);
	menu->add_icon_shortcut(get_icon("Duplicate", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/duplicate"), TOOL_DUPLICATE);
	menu->add_icon_shortcut(get_icon("Reparent", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/reparent"), TOOL_REPARENT);

	if (selection.size() == 1) {
		menu->add_separator();
		menu->add_icon_shortcut(get_icon("Blend", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/merge_from_scene"), TOOL_MERGE_FROM_SCENE);
		menu->add_icon_shortcut(get_icon("CreateNewSceneFrom", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/save_branch_as_scene"), TOOL_NEW_SCENE_FROM);
		menu->add_separator();
		menu->add_icon_shortcut(get_icon("CopyNodePath", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/copy_node_path"), TOOL_COPY_NODE_PATH);
		bool is_external = (selection[0]->get_filename() != "");
		if (is_external) {
			bool is_inherited = selection[0]->get_scene_inherited_state() != NULL;
			bool is_top_level = selection[0]->get_owner() == NULL;
			if (is_inherited && is_top_level) {
				menu->add_separator();
				menu->add_item(TTR("Clear Inheritance"), TOOL_SCENE_CLEAR_INHERITANCE);
				menu->add_icon_item(get_icon("Load", "EditorIcons"), TTR("Open in Editor"), TOOL_SCENE_OPEN_INHERITED);
			} else if (!is_top_level) {
				menu->add_separator();
				bool editable = EditorNode::get_singleton()->get_edited_scene()->is_editable_instance(selection[0]);
				bool placeholder = selection[0]->get_scene_instance_load_placeholder();
				menu->add_check_item(TTR("Editable Children"), TOOL_SCENE_EDITABLE_CHILDREN);
				menu->add_check_item(TTR("Load As Placeholder"), TOOL_SCENE_USE_PLACEHOLDER);
				menu->add_item(TTR("Discard Instancing"), TOOL_SCENE_CLEAR_INSTANCING);
				menu->add_icon_item(get_icon("Load", "EditorIcons"), TTR("Open in Editor"), TOOL_SCENE_OPEN);
				menu->set_item_checked(menu->get_item_idx_from_text(TTR("Editable Children")), editable);
				menu->set_item_checked(menu->get_item_idx_from_text(TTR("Load As Placeholder")), placeholder);
			}
		}
	}
	menu->add_separator();
	menu->add_icon_shortcut(get_icon("Remove", "EditorIcons"), ED_SHORTCUT("scene_tree/delete", TTR("Delete Node(s)"), KEY_DELETE), TOOL_ERASE);
	menu->set_size(Size2(1, 1));
	menu->set_position(p_menu_pos);
	menu->popup();
}

void SceneTreeDock::_filter_changed(const String &p_filter) {

	scene_tree->set_filter(p_filter);
}

String SceneTreeDock::get_filter() {

	return filter->get_text();
}

void SceneTreeDock::set_filter(const String &p_filter) {

	filter->set_text(p_filter);
	scene_tree->set_filter(p_filter);
}

void SceneTreeDock::_focus_node() {

	Node *node = scene_tree->get_selected();
	ERR_FAIL_COND(!node);

	if (node->is_class("CanvasItem")) {
		CanvasItemEditorPlugin *editor = Object::cast_to<CanvasItemEditorPlugin>(editor_data->get_editor("2D"));
		editor->get_canvas_item_editor()->focus_selection();
	} else {
		SpatialEditorPlugin *editor = Object::cast_to<SpatialEditorPlugin>(editor_data->get_editor("3D"));
		editor->get_spatial_editor()->get_editor_viewport(0)->focus_selection();
	}
}

void SceneTreeDock::open_script_dialog(Node *p_for_node) {

	scene_tree->set_selected(p_for_node, false);
	_tool_selected(TOOL_ATTACH_SCRIPT);
}

void SceneTreeDock::add_remote_tree_editor(Control *p_remote) {
	ERR_FAIL_COND(remote_tree != NULL);
	add_child(p_remote);
	remote_tree = p_remote;
	remote_tree->hide();
}

void SceneTreeDock::show_remote_tree() {

	_remote_tree_selected();
}

void SceneTreeDock::hide_remote_tree() {

	_local_tree_selected();
}

void SceneTreeDock::show_tab_buttons() {

	button_hb->show();
}

void SceneTreeDock::hide_tab_buttons() {

	button_hb->hide();
}

void SceneTreeDock::_remote_tree_selected() {

	scene_tree->hide();
	if (remote_tree)
		remote_tree->show();
	edit_remote->set_pressed(true);
	edit_local->set_pressed(false);

	emit_signal("remote_tree_selected");
}

void SceneTreeDock::_local_tree_selected() {

	scene_tree->show();
	if (remote_tree)
		remote_tree->hide();
	edit_remote->set_pressed(false);
	edit_local->set_pressed(true);

	_node_selected();
}

void SceneTreeDock::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_tool_selected"), &SceneTreeDock::_tool_selected, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("_create"), &SceneTreeDock::_create);
	ClassDB::bind_method(D_METHOD("_node_reparent"), &SceneTreeDock::_node_reparent);
	ClassDB::bind_method(D_METHOD("_set_owners"), &SceneTreeDock::_set_owners);
	ClassDB::bind_method(D_METHOD("_node_selected"), &SceneTreeDock::_node_selected);
	ClassDB::bind_method(D_METHOD("_node_renamed"), &SceneTreeDock::_node_renamed);
	ClassDB::bind_method(D_METHOD("_script_created"), &SceneTreeDock::_script_created);
	ClassDB::bind_method(D_METHOD("_load_request"), &SceneTreeDock::_load_request);
	ClassDB::bind_method(D_METHOD("_script_open_request"), &SceneTreeDock::_script_open_request);
	ClassDB::bind_method(D_METHOD("_unhandled_key_input"), &SceneTreeDock::_unhandled_key_input);
	ClassDB::bind_method(D_METHOD("_input"), &SceneTreeDock::_input);
	ClassDB::bind_method(D_METHOD("_nodes_drag_begin"), &SceneTreeDock::_nodes_drag_begin);
	ClassDB::bind_method(D_METHOD("_delete_confirm"), &SceneTreeDock::_delete_confirm);
	ClassDB::bind_method(D_METHOD("_node_prerenamed"), &SceneTreeDock::_node_prerenamed);
	ClassDB::bind_method(D_METHOD("_import_subscene"), &SceneTreeDock::_import_subscene);
	ClassDB::bind_method(D_METHOD("_selection_changed"), &SceneTreeDock::_selection_changed);
	ClassDB::bind_method(D_METHOD("_new_scene_from"), &SceneTreeDock::_new_scene_from);
	ClassDB::bind_method(D_METHOD("_nodes_dragged"), &SceneTreeDock::_nodes_dragged);
	ClassDB::bind_method(D_METHOD("_files_dropped"), &SceneTreeDock::_files_dropped);
	ClassDB::bind_method(D_METHOD("_script_dropped"), &SceneTreeDock::_script_dropped);
	ClassDB::bind_method(D_METHOD("_tree_rmb"), &SceneTreeDock::_tree_rmb);
	ClassDB::bind_method(D_METHOD("_filter_changed"), &SceneTreeDock::_filter_changed);
	ClassDB::bind_method(D_METHOD("_focus_node"), &SceneTreeDock::_focus_node);
	ClassDB::bind_method(D_METHOD("_remote_tree_selected"), &SceneTreeDock::_remote_tree_selected);
	ClassDB::bind_method(D_METHOD("_local_tree_selected"), &SceneTreeDock::_local_tree_selected);

	ClassDB::bind_method(D_METHOD("instance"), &SceneTreeDock::instance);

	ADD_SIGNAL(MethodInfo("remote_tree_selected"));
}

SceneTreeDock::SceneTreeDock(EditorNode *p_editor, Node *p_scene_root, EditorSelection *p_editor_selection, EditorData &p_editor_data) {

	set_name("Scene");
	editor = p_editor;
	edited_scene = NULL;
	editor_data = &p_editor_data;
	editor_selection = p_editor_selection;
	scene_root = p_scene_root;

	VBoxContainer *vbc = this;

	HBoxContainer *filter_hbc = memnew(HBoxContainer);
	filter_hbc->add_constant_override("separate", 0);
	ToolButton *tb;

	ED_SHORTCUT("scene_tree/add_child_node", TTR("Add Child Node"), KEY_MASK_CMD | KEY_A);
	ED_SHORTCUT("scene_tree/instance_scene", TTR("Instance Child Scene"));
	ED_SHORTCUT("scene_tree/change_node_type", TTR("Change Type"));
	ED_SHORTCUT("scene_tree/attach_script", TTR("Attach Script"));
	ED_SHORTCUT("scene_tree/clear_script", TTR("Clear Script"));
	ED_SHORTCUT("scene_tree/move_up", TTR("Move Up"), KEY_MASK_CMD | KEY_UP);
	ED_SHORTCUT("scene_tree/move_down", TTR("Move Down"), KEY_MASK_CMD | KEY_DOWN);
	ED_SHORTCUT("scene_tree/duplicate", TTR("Duplicate"), KEY_MASK_CMD | KEY_D);
	ED_SHORTCUT("scene_tree/reparent", TTR("Reparent"));
	ED_SHORTCUT("scene_tree/merge_from_scene", TTR("Merge From Scene"));
	ED_SHORTCUT("scene_tree/save_branch_as_scene", TTR("Save Branch as Scene"));
	ED_SHORTCUT("scene_tree/copy_node_path", TTR("Copy Node Path"), KEY_MASK_CMD | KEY_C);
	ED_SHORTCUT("scene_tree/delete_no_confirm", TTR("Delete (No Confirm)"), KEY_MASK_SHIFT | KEY_DELETE);
	ED_SHORTCUT("scene_tree/delete", TTR("Delete"), KEY_DELETE);

	tb = memnew(ToolButton);
	tb->connect("pressed", this, "_tool_selected", make_binds(TOOL_NEW, false));
	tb->set_tooltip(TTR("Add/Create a New Node"));
	tb->set_shortcut(ED_GET_SHORTCUT("scene_tree/add_child_node"));
	filter_hbc->add_child(tb);
	button_add = tb;

	tb = memnew(ToolButton);
	tb->connect("pressed", this, "_tool_selected", make_binds(TOOL_INSTANCE, false));
	tb->set_tooltip(TTR("Instance a scene file as a Node. Creates an inherited scene if no root node exists."));
	tb->set_shortcut(ED_GET_SHORTCUT("scene_tree/instance_scene"));
	filter_hbc->add_child(tb);
	button_instance = tb;

	vbc->add_child(filter_hbc);
	filter = memnew(LineEdit);
	filter->set_h_size_flags(SIZE_EXPAND_FILL);
	filter->set_placeholder(TTR("Filter nodes"));
	filter_hbc->add_child(filter);
	filter->connect("text_changed", this, "_filter_changed");

	tb = memnew(ToolButton);
	tb->connect("pressed", this, "_tool_selected", make_binds(TOOL_ATTACH_SCRIPT, false));
	tb->set_tooltip(TTR("Attach a new or existing script for the selected node."));
	tb->set_shortcut(ED_GET_SHORTCUT("scene_tree/attach_script"));
	filter_hbc->add_child(tb);
	tb->hide();
	button_create_script = tb;

	tb = memnew(ToolButton);
	tb->connect("pressed", this, "_tool_selected", make_binds(TOOL_CLEAR_SCRIPT, false));
	tb->set_tooltip(TTR("Clear a script for the selected node."));
	tb->set_shortcut(ED_GET_SHORTCUT("scene_tree/clear_script"));
	filter_hbc->add_child(tb);
	button_clear_script = tb;
	tb->hide();

	button_hb = memnew(HBoxContainer);
	vbc->add_child(button_hb);

	edit_remote = memnew(ToolButton);
	button_hb->add_child(edit_remote);
	edit_remote->set_h_size_flags(SIZE_EXPAND_FILL);
	edit_remote->set_text(TTR("Remote"));
	edit_remote->set_toggle_mode(true);
	edit_remote->connect("pressed", this, "_remote_tree_selected");

	edit_local = memnew(ToolButton);
	button_hb->add_child(edit_local);
	edit_local->set_h_size_flags(SIZE_EXPAND_FILL);
	edit_local->set_text(TTR("Local"));
	edit_local->set_toggle_mode(true);
	edit_local->connect("pressed", this, "_local_tree_selected");

	remote_tree = NULL;
	button_hb->hide();

	scene_tree = memnew(SceneTreeEditor(false, true, true));

	vbc->add_child(scene_tree);
	scene_tree->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
	scene_tree->connect("rmb_pressed", this, "_tree_rmb");

	scene_tree->connect("node_selected", this, "_node_selected", varray(), CONNECT_DEFERRED);
	scene_tree->connect("node_renamed", this, "_node_renamed", varray(), CONNECT_DEFERRED);
	scene_tree->connect("node_prerename", this, "_node_prerenamed");
	scene_tree->connect("open", this, "_load_request");
	scene_tree->connect("open_script", this, "_script_open_request");
	scene_tree->connect("nodes_rearranged", this, "_nodes_dragged");
	scene_tree->connect("files_dropped", this, "_files_dropped");
	scene_tree->connect("script_dropped", this, "_script_dropped");
	scene_tree->connect("nodes_dragged", this, "_nodes_drag_begin");

	scene_tree->get_scene_tree()->connect("item_double_clicked", this, "_focus_node");

	scene_tree->set_undo_redo(&editor_data->get_undo_redo());
	scene_tree->set_editor_selection(editor_selection);

	create_dialog = memnew(CreateDialog);
	create_dialog->set_base_type("Node");
	add_child(create_dialog);
	create_dialog->connect("create", this, "_create");

	script_create_dialog = memnew(ScriptCreateDialog);
	add_child(script_create_dialog);
	script_create_dialog->connect("script_created", this, "_script_created");

	reparent_dialog = memnew(ReparentDialog);
	add_child(reparent_dialog);
	reparent_dialog->connect("reparent", this, "_node_reparent");

	accept = memnew(AcceptDialog);
	add_child(accept);

	file = memnew(EditorFileDialog);
	add_child(file);
	file->connect("file_selected", this, "instance");
	set_process_unhandled_key_input(true);

	delete_dialog = memnew(ConfirmationDialog);
	add_child(delete_dialog);
	delete_dialog->connect("confirmed", this, "_delete_confirm");

	import_subscene_dialog = memnew(EditorSubScene);
	add_child(import_subscene_dialog);
	import_subscene_dialog->connect("subscene_selected", this, "_import_subscene");

	new_scene_from_dialog = memnew(EditorFileDialog);
	new_scene_from_dialog->set_mode(EditorFileDialog::MODE_SAVE_FILE);
	add_child(new_scene_from_dialog);
	new_scene_from_dialog->connect("file_selected", this, "_new_scene_from");

	menu = memnew(PopupMenu);
	add_child(menu);
	menu->connect("id_pressed", this, "_tool_selected");
	menu_subresources = memnew(PopupMenu);
	menu_subresources->set_name("Sub-Resources");
	menu->add_child(menu_subresources);
	first_enter = true;
	restore_script_editor_on_drag = false;

	clear_inherit_confirm = memnew(ConfirmationDialog);
	clear_inherit_confirm->set_text(TTR("Clear Inheritance? (No Undo!)"));
	clear_inherit_confirm->get_ok()->set_text(TTR("Clear!"));
	add_child(clear_inherit_confirm);

	set_process_input(true);
}
