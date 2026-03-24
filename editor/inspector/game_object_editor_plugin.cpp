/**************************************************************************/
/*  game_object_editor_plugin.cpp                                         */
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

#include "game_object_editor_plugin.h"

#include "core/io/resource_loader.h"
#include "core/object/script_language.h"
#include "editor/editor_node.h"
#include "editor/editor_undo_redo_manager.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/separator.h"

// GameObjectComponentList

void GameObjectComponentList::_refresh_list() {
	component_list->clear();
	if (!game_object) {
		return;
	}

	for (int i = 0; i < game_object->get_child_count(); i++) {
		Node *child = game_object->get_child(i);
		Ref<Texture2D> icon = EditorNode::get_singleton()->get_object_icon(child);
		component_list->add_item(child->get_name(), icon);
	}

	// Resize to fit contents, with a minimum and maximum height.
	int item_count = component_list->get_item_count();
	if (item_count == 0) {
		component_list->set_custom_minimum_size(Size2(0, 0));
		component_list->hide();
	} else {
		component_list->show();
		int height = CLAMP(item_count * 26 + 4, 30, 200);
		component_list->set_custom_minimum_size(Size2(0, height));
	}
}

void GameObjectComponentList::_on_add_component_pressed() {
	if (!create_dialog) {
		create_dialog = memnew(CreateDialog);
		create_dialog->set_base_type("Node");
		EditorNode::get_singleton()->get_gui_base()->add_child(create_dialog);
		create_dialog->connect("create", callable_mp(this, &GameObjectComponentList::_on_create_confirmed));
	}
	create_dialog->popup_create(true);
}

void GameObjectComponentList::_on_component_selected(int p_index) {
	if (!game_object || p_index < 0 || p_index >= game_object->get_child_count()) {
		return;
	}
	Node *child = game_object->get_child(p_index);
	EditorNode::get_singleton()->push_item(child);
}

void GameObjectComponentList::_on_create_confirmed() {
	if (!game_object || !create_dialog) {
		return;
	}

	Node *child = Object::cast_to<Node>(create_dialog->instantiate_selected());
	ERR_FAIL_NULL(child);

	Node *edited_scene = EditorNode::get_singleton()->get_edited_scene();
	ERR_FAIL_NULL(edited_scene);

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Add Component"));
	undo_redo->add_do_method(game_object, "add_child", child, true);
	undo_redo->add_do_method(child, "set_owner", edited_scene);
	undo_redo->add_do_reference(child);
	undo_redo->add_undo_method(game_object, "remove_child", child);
	undo_redo->commit_action();
}

void GameObjectComponentList::_on_add_script_pressed() {
	if (!script_file_dialog) {
		script_file_dialog = memnew(EditorFileDialog);
		script_file_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
		script_file_dialog->set_title(TTR("Add Script Component"));
		script_file_dialog->clear_filters();
		for (int i = 0; i < ScriptServer::get_language_count(); i++) {
			ScriptLanguage *lang = ScriptServer::get_language(i);
			List<String> extensions;
			lang->get_recognized_extensions(&extensions);
			for (const String &ext : extensions) {
				script_file_dialog->add_filter("*." + ext, lang->get_name());
			}
		}
		EditorNode::get_singleton()->get_gui_base()->add_child(script_file_dialog);
		script_file_dialog->connect("file_selected", callable_mp(this, &GameObjectComponentList::_on_script_file_selected));
	}
	script_file_dialog->popup_file_dialog();
}

void GameObjectComponentList::_on_script_file_selected(const String &p_path) {
	if (!game_object) {
		return;
	}

	Ref<Script> scr = ResourceLoader::load(p_path, "Script");
	ERR_FAIL_COND_MSG(scr.is_null(), "Failed to load script: " + p_path);

	StringName base_type = scr->get_instance_base_type();
	ERR_FAIL_COND_MSG(base_type == StringName(), "Script does not have a valid base type.");

	Object *obj = ClassDB::instantiate(base_type);
	ERR_FAIL_NULL_MSG(obj, "Failed to instantiate base type: " + String(base_type));

	Node *child = Object::cast_to<Node>(obj);
	if (!child) {
		if (!obj->is_ref_counted()) {
			memdelete(obj);
		}
		ERR_FAIL_MSG("Script base type is not a Node-derived type.");
	}

	child->set_name(Node::adjust_name_casing(p_path.get_file().get_basename()));
	child->set_script(scr);

	Node *edited_scene = EditorNode::get_singleton()->get_edited_scene();
	ERR_FAIL_NULL(edited_scene);

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Add Script Component"));
	undo_redo->add_do_method(game_object, "add_child", child, true);
	undo_redo->add_do_method(child, "set_owner", edited_scene);
	undo_redo->add_do_reference(child);
	undo_redo->add_undo_method(game_object, "remove_child", child);
	undo_redo->commit_action();
}

void GameObjectComponentList::_on_child_order_changed() {
	_refresh_list();
}

Variant GameObjectComponentList::_get_drag_data_fw(const Point2 &p_point, Control *p_from) {
	int idx = component_list->get_item_at_position(p_point);
	if (idx < 0) {
		return Variant();
	}

	Dictionary drag_data;
	drag_data["type"] = "game_object_component";
	drag_data["index"] = idx;

	// Create preview label.
	Label *label = memnew(Label);
	label->set_text(component_list->get_item_text(idx));
	set_drag_preview(label);

	return drag_data;
}

bool GameObjectComponentList::_can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {
	if (p_data.get_type() != Variant::DICTIONARY) {
		return false;
	}
	Dictionary d = p_data;
	if (!d.has("type") || String(d["type"]) != "game_object_component") {
		return false;
	}
	return true;
}

void GameObjectComponentList::_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
	if (!game_object) {
		return;
	}

	Dictionary d = p_data;
	int from_idx = d["index"];
	int to_idx = component_list->get_item_at_position(p_point);
	if (to_idx < 0) {
		to_idx = component_list->get_item_count() - 1;
	}
	if (from_idx == to_idx) {
		return;
	}

	Node *child = game_object->get_child(from_idx);
	ERR_FAIL_NULL(child);

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Reorder Component"));
	undo_redo->add_do_method(game_object, "move_child", child, to_idx);
	undo_redo->add_undo_method(game_object, "move_child", child, from_idx);
	undo_redo->commit_action();
}

void GameObjectComponentList::set_game_object(Node *p_game_object) {
	if (game_object && game_object->is_connected("child_order_changed", callable_mp(this, &GameObjectComponentList::_on_child_order_changed))) {
		game_object->disconnect("child_order_changed", callable_mp(this, &GameObjectComponentList::_on_child_order_changed));
	}
	game_object = p_game_object;
	if (game_object) {
		game_object->connect("child_order_changed", callable_mp(this, &GameObjectComponentList::_on_child_order_changed));
	}
	_refresh_list();
}

GameObjectComponentList::GameObjectComponentList() {
	// Header.
	Label *header = memnew(Label);
	header->set_text(TTR("Components"));
	header->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_LEFT);
	add_child(header);

	// Component list.
	component_list = memnew(ItemList);
	component_list->set_select_mode(ItemList::SELECT_SINGLE);
	component_list->set_allow_reselect(true);
	component_list->set_v_size_flags(SIZE_EXPAND_FILL);
	SET_DRAG_FORWARDING_GCDU(component_list, GameObjectComponentList);
	component_list->connect("item_selected", callable_mp(this, &GameObjectComponentList::_on_component_selected));
	add_child(component_list);

	// Separator.
	add_child(memnew(HSeparator));

	// Add Component button.
	Button *add_btn = memnew(Button);
	add_btn->set_text(TTR("Add Component"));
	add_btn->connect(SceneStringName(pressed), callable_mp(this, &GameObjectComponentList::_on_add_component_pressed));
	add_child(add_btn);

	// Add Script button.
	Button *add_script_btn = memnew(Button);
	add_script_btn->set_text(TTR("Add Script"));
	add_script_btn->connect(SceneStringName(pressed), callable_mp(this, &GameObjectComponentList::_on_add_script_pressed));
	add_child(add_script_btn);
}

// GameObjectInspectorPlugin

bool GameObjectInspectorPlugin::can_handle(Object *p_object) {
	return Object::cast_to<GameObject>(p_object) != nullptr ||
			Object::cast_to<GameObject2D>(p_object) != nullptr;
}

void GameObjectInspectorPlugin::parse_begin(Object *p_object) {
	Node *go = Object::cast_to<GameObject>(p_object);
	if (!go) {
		go = Object::cast_to<GameObject2D>(p_object);
	}
	if (!go) {
		return;
	}

	GameObjectComponentList *comp_list = memnew(GameObjectComponentList);
	comp_list->set_game_object(go);
	add_custom_control(comp_list);
}

// GameObjectEditorPlugin

GameObjectEditorPlugin::GameObjectEditorPlugin() {
	Ref<GameObjectInspectorPlugin> plugin;
	plugin.instantiate();
	add_inspector_plugin(plugin);
}
