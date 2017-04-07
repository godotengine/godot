/*************************************************************************/
/*  scene_tree_dock.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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

#include "animation_editor.h"
#include "core/io/resource_saver.h"
#include "editor/plugins/animation_player_editor_plugin.h"
#include "editor/plugins/canvas_item_editor_plugin.h"
#include "editor/plugins/script_editor_plugin.h"
#include "editor/plugins/spatial_editor_plugin.h"
#include "editor_node.h"
#include "editor_settings.h"
#include "global_config.h"
#include "multi_node_edit.h"
#include "os/keyboard.h"
#include "scene/main/viewport.h"
#include "scene/resources/packed_scene.h"
#include "script_editor_debugger.h"

void SceneTreeDock::_nodes_drag_begin() {

	if (restore_script_editor_on_drag) {
		EditorNode::get_singleton()->set_visible_editor(EditorNode::EDITOR_SCRIPT);
		restore_script_editor_on_drag = false;
	}
}

void SceneTreeDock::_input(InputEvent p_event) {

	if (p_event.type == InputEvent::MOUSE_BUTTON && !p_event.mouse_button.pressed && p_event.mouse_button.button_index == BUTTON_LEFT) {
		restore_script_editor_on_drag = false; //lost chance
	}
}

void SceneTreeDock::_unhandled_key_input(InputEvent p_event) {

	if (get_viewport()->get_modal_stack_top())
		return; //ignore because of modal window

	if (!p_event.key.pressed || p_event.key.echo)
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
		//accept->get_cancel()->hide();
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
			//accept->get_cancel()->hide();
			accept->get_ok()->set_text(TTR("Ugh"));
			accept->set_text(vformat(TTR("Error loading scene from %s"), p_files[i]));
			accept->popup_centered_minsize();
			error = true;
			break;
		}

		Node *instanced_scene = sdata->instance(PackedScene::GEN_EDIT_STATE_INSTANCE);
		if (!instanced_scene) {
			current_option = -1;
			//accept->get_cancel()->hide();
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

		instanced_scene->set_filename(GlobalConfig::get_singleton()->localize_path(p_files[i]));

		instances.push_back(instanced_scene);
	}

	if (error) {
		for (int i = 0; i < instances.size(); i++) {
			memdelete(instances[i]);
		}
		return;
	}

	//instanced_scene->generate_instance_state();

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
			/*
			if (!_validate_no_foreign())
				break;
			*/
			create_dialog->popup_create(true);
		} break;
		case TOOL_INSTANCE: {

			Node *scene = edited_scene;

			if (!scene) {

				EditorNode::get_singleton()->new_inherited_scene();

				/* should be legal now
				current_option=-1;
				//confirmation->get_cancel()->hide();
				accept->get_ok()->set_text("I see..");
				accept->set_text("This operation can't be done without a tree root.");
				accept->popup_centered_minsize();
				*/
				break;
			}

			/*
			if (!_validate_no_foreign())
				break;
			*/

			file->set_mode(EditorFileDialog::MODE_OPEN_FILE);
			List<String> extensions;
			ResourceLoader::get_recognized_extensions_for_type("PackedScene", &extensions);
			file->clear_filters();
			for (int i = 0; i < extensions.size(); i++) {

				file->add_filter("*." + extensions[i] + " ; " + extensions[i].to_upper());
			}

			//file->set_current_path(current_path);
			file->popup_centered_ratio();

		} break;
		case TOOL_REPLACE: {

			create_dialog->popup_create(false);
		} break;
		case TOOL_CONNECT: {

			Node *current = scene_tree->get_selected();
			if (!current)
				break;

			/*
			if (!_validate_no_foreign())
				break;
			connect_dialog->popup_centered_ratio();
			connect_dialog->set_node(current);
			*/

		} break;
		case TOOL_GROUP: {

			Node *current = scene_tree->get_selected();
			if (!current)
				break;
			/*
			if (!_validate_no_foreign())
				break;
			groups_editor->set_current(current);
			groups_editor->popup_centered_ratio();
			*/
		} break;
		case TOOL_ATTACH_SCRIPT: {

			Node *selected = scene_tree->get_selected();
			if (!selected)
				break;

			/*
			if (!_validate_no_foreign())
				break;
			*/

			Ref<Script> existing = selected->get_script();
			if (existing.is_valid())
				editor->push_item(existing.ptr());
			else {
				String path = selected->get_filename();
				script_create_dialog->config(selected->get_class(), path);
				script_create_dialog->popup_centered(Size2(300, 290));
				//script_create_dialog->popup_centered_minsize();
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
				//accept->get_cancel()->hide();
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

			List<Node *> reselect;

			editor_data->get_undo_redo().create_action(TTR("Duplicate Node(s)"));
			editor_data->get_undo_redo().add_do_method(editor_selection, "clear");

			Node *dupsingle = NULL;

			for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

				Node *node = E->get();
				Node *parent = node->get_parent();

				List<Node *> owned;
				node->get_owned_by(node->get_owner(), &owned);

				Map<Node *, Node *> duplimap;
				Node *dup = _duplicate(node, duplimap);

				ERR_CONTINUE(!dup);

				if (selection.size() == 1)
					dupsingle = dup;

				dup->set_name(parent->validate_child_name(dup));

				editor_data->get_undo_redo().add_do_method(parent, "_add_child_below_node", node, dup);
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

				//parent->add_child(dup);
				//reselect.push_back(dup);
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
				//confirmation->get_cancel()->hide();
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

				// hack, force 2d editor viewport to refresh after deletion
				if (CanvasItemEditor *editor = CanvasItemEditor::get_singleton())
					editor->get_viewport_control()->update();

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

			if (List<Node *>::Element *e = selection.front()) {
				if (Node *node = e->get()) {
					Node *root = EditorNode::get_singleton()->get_edited_scene();
					NodePath path = root->get_path().rel_path_to(node->get_path());
					OS::get_singleton()->set_clipboard(path);
				}
			}
		} break;
	}
}

void SceneTreeDock::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_READY: {

			if (!first_enter)
				break;
			first_enter = false;

			CanvasItemEditorPlugin *canvas_item_plugin = editor_data->get_editor("2D")->cast_to<CanvasItemEditorPlugin>();
			if (canvas_item_plugin) {
				canvas_item_plugin->get_canvas_item_editor()->connect("item_lock_status_changed", scene_tree, "_update_tree");
				canvas_item_plugin->get_canvas_item_editor()->connect("item_group_status_changed", scene_tree, "_update_tree");
				scene_tree->connect("node_changed", canvas_item_plugin->get_canvas_item_editor()->get_viewport_control(), "update");
			}
			button_add->set_icon(get_icon("Add", "EditorIcons"));
			button_instance->set_icon(get_icon("Instance", "EditorIcons"));
			button_create_script->set_icon(get_icon("ScriptCreate", "EditorIcons"));
			button_clear_script->set_icon(get_icon("ScriptRemove", "EditorIcons"));

			filter_icon->set_texture(get_icon("Zoom", "EditorIcons"));

			EditorNode::get_singleton()->get_editor_selection()->connect("selection_changed", this, "_selection_changed");

		} break;
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

Node *SceneTreeDock::_duplicate(Node *p_node, Map<Node *, Node *> &duplimap) {

	Node *node = NULL;

	if (p_node->get_filename() != "") { //an instance

		Ref<PackedScene> sd = ResourceLoader::load(p_node->get_filename());
		ERR_FAIL_COND_V(!sd.is_valid(), NULL);
		node = sd->instance(PackedScene::GEN_EDIT_STATE_INSTANCE);
		ERR_FAIL_COND_V(!node, NULL);
		node->set_scene_instance_load_placeholder(p_node->get_scene_instance_load_placeholder());
		//node->generate_instance_state();
	} else {
		Object *obj = ClassDB::instance(p_node->get_class());
		ERR_FAIL_COND_V(!obj, NULL);
		node = obj->cast_to<Node>();
		if (!node)
			memdelete(obj);
		ERR_FAIL_COND_V(!node, NULL);
	}

	List<PropertyInfo> plist;

	p_node->get_property_list(&plist);

	for (List<PropertyInfo>::Element *E = plist.front(); E; E = E->next()) {

		if (!(E->get().usage & PROPERTY_USAGE_STORAGE))
			continue;
		String name = E->get().name;
		node->set(name, p_node->get(name));
	}

	List<Node::GroupInfo> group_info;
	p_node->get_groups(&group_info);
	for (List<Node::GroupInfo>::Element *E = group_info.front(); E; E = E->next()) {

		if (E->get().persistent)
			node->add_to_group(E->get().name, true);
	}

	node->set_name(p_node->get_name());
	duplimap[p_node] = node;

	for (int i = 0; i < p_node->get_child_count(); i++) {

		Node *child = p_node->get_child(i);
		if (p_node->get_owner() != child->get_owner())
			continue; //don't bother with not in-scene nodes.

		Node *dup = _duplicate(child, duplimap);
		if (!dup) {
			memdelete(node);
			return NULL;
		}

		node->add_child(dup);
	}

	return node;
}

void SceneTreeDock::_set_owners(Node *p_owner, const Array &p_nodes) {

	for (int i = 0; i < p_nodes.size(); i++) {

		Object *obj = p_nodes[i];
		if (!obj)
			continue;

		Node *n = obj->cast_to<Node>();
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

	if (p_base->cast_to<AnimationPlayer>()) {

		AnimationPlayer *ap = p_base->cast_to<AnimationPlayer>();
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

									NodePath new_path = NodePath(rel_path.get_names(), track_np.get_subnames(), false, track_np.get_property());
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

	//ok all valid

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

	List<Node *> selection = editor_selection->get_selected_node_list();

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
			if (node->cast_to<Node2D>())
				editor_data->get_undo_redo().add_do_method(node, "set_global_transform", node->cast_to<Node2D>()->get_global_transform());
			if (node->cast_to<Spatial>())
				editor_data->get_undo_redo().add_do_method(node, "set_global_transform", node->cast_to<Spatial>()->get_global_transform());
			if (node->cast_to<Control>())
				editor_data->get_undo_redo().add_do_method(node, "set_global_pos", node->cast_to<Control>()->get_global_pos());
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
			if (node->cast_to<Node2D>())
				editor_data->get_undo_redo().add_undo_method(node, "set_transform", node->cast_to<Node2D>()->get_transform());
			if (node->cast_to<Spatial>())
				editor_data->get_undo_redo().add_undo_method(node, "set_transform", node->cast_to<Spatial>()->get_transform());
			if (node->cast_to<Control>())
				editor_data->get_undo_redo().add_undo_method(node, "set_pos", node->cast_to<Control>()->get_pos());
		}
	}

	perform_node_renames(NULL, &path_renames);

	editor_data->get_undo_redo().commit_action();
	//node->set_owner(owner);
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
			//editor_data->get_undo_redo().add_undo_method(n,"set_owner",n->get_owner());
			editor_data->get_undo_redo().add_undo_reference(n);

			ScriptEditorDebugger *sed = ScriptEditor::get_singleton()->get_debugger();
			editor_data->get_undo_redo().add_do_method(sed, "live_debug_remove_and_keep_node", edited_scene->get_path_to(n), n->get_instance_ID());
			editor_data->get_undo_redo().add_undo_method(sed, "live_debug_restore_node", n->get_instance_ID(), edited_scene->get_path_to(n->get_parent()), n->get_index());
		}
	}
	editor_data->get_undo_redo().commit_action();
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

	//tool_buttons[TOOL_MULTI_EDIT]->set_disabled(EditorNode::get_singleton()->get_editor_selection()->get_selection().size()<2);
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
		Node *child = c->cast_to<Node>();
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
			editor_data->get_undo_redo().add_do_reference(child);
			editor_data->get_undo_redo().add_undo_method(editor, "set_edited_scene", (Object *)NULL);
		}

		editor_data->get_undo_redo().commit_action();
		editor->push_item(c);

		if (c->cast_to<Control>()) {
			//make editor more comfortable, so some controls don't appear super shrunk
			Control *ct = c->cast_to<Control>();

			Size2 ms = ct->get_minimum_size();
			if (ms.width < 4)
				ms.width = 40;
			if (ms.height < 4)
				ms.height = 40;
			ct->set_size(ms);
		}

	} else if (current_option == TOOL_REPLACE) {
		Node *n = scene_tree->get_selected();
		ERR_FAIL_COND(!n);

		Object *c = create_dialog->instance_selected();

		ERR_FAIL_COND(!c);
		Node *newnode = c->cast_to<Node>();
		ERR_FAIL_COND(!newnode);

		List<PropertyInfo> pinfo;
		n->get_property_list(&pinfo);

		for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {
			if (!(E->get().usage & PROPERTY_USAGE_STORAGE))
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

	/*
	editor_data->get_undo_redo().create_action("Import Subscene");
	editor_data->get_undo_redo().add_do_method(parent,"add_child",ss);
	//editor_data->get_undo_redo().add_do_method(editor_selection,"clear");
	//editor_data->get_undo_redo().add_do_method(editor_selection,"add_node",child);
	editor_data->get_undo_redo().add_do_reference(ss);
	editor_data->get_undo_redo().add_undo_method(parent,"remove_child",ss);
	editor_data->get_undo_redo().commit_action();
*/
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
		/*
		if (EditorSettings::get_singleton()->get("filesystem/on_save/save_paths_as_relative"))
			flg|=ResourceSaver::FLAG_RELATIVE_PATHS;
		*/

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
#if 0
				//quite complicated, look for next visible in tree
				upper_sibling=_find_last_visible(upper_sibling);

				if (upper_sibling->get_parent()==to_node->get_parent()) {
					//just insert over this node because nothing is above at an upper level
					to_pos=to_node->get_index();
					to_node=to_node->get_parent();
				} else {
					to_pos=-1; //insert last in whathever is up
					to_node=upper_sibling->get_parent(); //insert at a parent of whathever is up
				}


			} else {
				//just insert over this node because nothing is above at the same level
				to_pos=to_node->get_index();
				to_node=to_node->get_parent();
			}
#endif
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

void SceneTreeDock::_tree_rmb(const Vector2 &p_menu_pos) {
	if (!EditorNode::get_singleton()->get_edited_scene()) {

		menu->clear();
		menu->add_icon_shortcut(get_icon("Add", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/add_child_node"), TOOL_NEW);
		menu->add_icon_shortcut(get_icon("Instance", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/instance_scene"), TOOL_INSTANCE);

		menu->set_size(Size2(1, 1));
		menu->set_pos(p_menu_pos);
		menu->popup();
		return;
	}

	List<Node *> selection = editor_selection->get_selected_node_list();

	if (selection.size() == 0)
		return;

	menu->clear();

	if (selection.size() == 1) {
		menu->add_icon_shortcut(get_icon("Add", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/add_child_node"), TOOL_NEW);
		menu->add_icon_shortcut(get_icon("Instance", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/instance_scene"), TOOL_INSTANCE);
		menu->add_separator();
		menu->add_icon_shortcut(get_icon("Reload", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/change_node_type"), TOOL_REPLACE);
		//menu->add_separator(); moved to their own dock
		//menu->add_icon_item(get_icon("Groups","EditorIcons"),TTR("Edit Groups"),TOOL_GROUP);
		//menu->add_icon_item(get_icon("Connect","EditorIcons"),TTR("Edit Connections"),TOOL_CONNECT);
		menu->add_separator();
		menu->add_icon_shortcut(get_icon("ScriptCreate", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/attach_script"), TOOL_ATTACH_SCRIPT);
		menu->add_icon_shortcut(get_icon("ScriptRemove", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/clear_script"), TOOL_CLEAR_SCRIPT);
		menu->add_separator();
	}

	menu->add_icon_shortcut(get_icon("Up", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/move_up"), TOOL_MOVE_UP);
	menu->add_icon_shortcut(get_icon("Down", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/move_down"), TOOL_MOVE_DOWN);
	menu->add_icon_shortcut(get_icon("Duplicate", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/duplicate"), TOOL_DUPLICATE);
	menu->add_icon_shortcut(get_icon("Reparent", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/reparent"), TOOL_REPARENT);

	if (selection.size() == 1) {
		menu->add_separator();
		menu->add_icon_shortcut(get_icon("Blend", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/merge_from_scene"), TOOL_MERGE_FROM_SCENE);
		menu->add_icon_shortcut(get_icon("CreateNewSceneFrom", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/save_branch_as_scene"), TOOL_NEW_SCENE_FROM);
		menu->add_separator();
		menu->add_icon_shortcut(get_icon("CopyNodePath", "EditorIcons"), ED_GET_SHORTCUT("scene_tree/copy_node_path"), TOOL_COPY_NODE_PATH);
	}
	menu->add_separator();
	menu->add_icon_shortcut(get_icon("Remove", "EditorIcons"), ED_SHORTCUT("scene_tree/delete", TTR("Delete Node(s)"), KEY_DELETE), TOOL_ERASE);
	menu->set_size(Size2(1, 1));
	menu->set_pos(p_menu_pos);
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
		CanvasItemEditorPlugin *editor = editor_data->get_editor("2D")->cast_to<CanvasItemEditorPlugin>();
		editor->get_canvas_item_editor()->focus_selection();
	} else {
		SpatialEditorPlugin *editor = editor_data->get_editor("3D")->cast_to<SpatialEditorPlugin>();
		editor->get_spatial_editor()->get_editor_viewport(0)->focus_selection();
	}
}

void SceneTreeDock::open_script_dialog(Node *p_for_node) {

	scene_tree->set_selected(p_for_node, false);
	_tool_selected(TOOL_ATTACH_SCRIPT);
}

void SceneTreeDock::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_tool_selected"), &SceneTreeDock::_tool_selected, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("_create"), &SceneTreeDock::_create);
	//ClassDB::bind_method(D_METHOD("_script_created"),&SceneTreeDock::_script_created);
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

	ClassDB::bind_method(D_METHOD("instance"), &SceneTreeDock::instance);
}

SceneTreeDock::SceneTreeDock(EditorNode *p_editor, Node *p_scene_root, EditorSelection *p_editor_selection, EditorData &p_editor_data) {

	editor = p_editor;
	edited_scene = NULL;
	editor_data = &p_editor_data;
	editor_selection = p_editor_selection;
	scene_root = p_scene_root;

	VBoxContainer *vbc = this;

	HBoxContainer *filter_hbc = memnew(HBoxContainer);
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
	filter_hbc->add_child(filter);
	filter_icon = memnew(TextureRect);
	filter_hbc->add_child(filter_icon);
	filter_icon->set_stretch_mode(TextureRect::STRETCH_KEEP_CENTERED);
	filter->connect("text_changed", this, "_filter_changed");

	tb = memnew(ToolButton);
	tb->connect("pressed", this, "_tool_selected", make_binds(TOOL_ATTACH_SCRIPT, false));
	tb->set_tooltip(TTR("Attach a new or existing script for the selected node."));
	tb->set_shortcut(ED_GET_SHORTCUT("scene_tree/attach_script"));
	filter_hbc->add_child(tb);
	button_create_script = tb;

	tb = memnew(ToolButton);
	tb->connect("pressed", this, "_tool_selected", make_binds(TOOL_CLEAR_SCRIPT, false));
	tb->set_tooltip(TTR("Clear a script for the selected node."));
	tb->set_shortcut(ED_GET_SHORTCUT("scene_tree/clear_script"));
	filter_hbc->add_child(tb);
	button_clear_script = tb;

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

	//groups_editor = memnew( GroupsEditor );
	//add_child(groups_editor);
	//groups_editor->set_undo_redo(&editor_data->get_undo_redo());

	//connect_dialog = memnew( ConnectionsDialog(p_editor) );
	//add_child(connect_dialog);
	//connect_dialog->set_undoredo(&editor_data->get_undo_redo());

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
	first_enter = true;
	restore_script_editor_on_drag = false;

	vbc->add_constant_override("separation", 4);
	set_process_input(true);
}
