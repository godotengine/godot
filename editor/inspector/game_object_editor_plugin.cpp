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
#include "editor/themes/editor_scale.h"
#include "scene/2d/physics/collision_object_2d.h"
#include "scene/2d/physics/collision_polygon_2d.h"
#include "scene/2d/physics/collision_shape_2d.h"
#include "scene/2d/physics/physics_body_2d.h"
#include "scene/2d/sprite_2d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/physics/collision_object_3d.h"
#include "scene/3d/physics/collision_polygon_3d.h"
#include "scene/3d/physics/collision_shape_3d.h"
#include "scene/3d/physics/physics_body_3d.h"
#include "scene/3d/physics/vehicle_body_3d.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/separator.h"
#include "scene/gui/texture_rect.h"

// GameObjectComponentList

void GameObjectComponentList::_rebuild_list() {
	// Disconnect child_order_changed from all previously tracked nodes.
	for (Node *node : displayed_nodes) {
		if (node && node->is_connected("child_order_changed", callable_mp(this, &GameObjectComponentList::_on_child_order_changed))) {
			node->disconnect("child_order_changed", callable_mp(this, &GameObjectComponentList::_on_child_order_changed));
		}
	}

	// Remove all existing component entries.
	while (component_container->get_child_count() > 0) {
		Node *child = component_container->get_child(0);
		component_container->remove_child(child);
		child->queue_free();
	}

	if (!game_object) {
		displayed_nodes.clear();
		return;
	}

	displayed_nodes.clear();
	for (int i = 0; i < game_object->get_child_count(); i++) {
		_rebuild_list_recursive(game_object->get_child(i), 0);
	}

	_update_warnings();
}

void GameObjectComponentList::_rebuild_list_recursive(Node *p_node, int p_depth) {
	// Track this node and listen for its children changing.
	displayed_nodes.push_back(p_node);
	if (!p_node->is_connected("child_order_changed", callable_mp(this, &GameObjectComponentList::_on_child_order_changed))) {
		p_node->connect("child_order_changed", callable_mp(this, &GameObjectComponentList::_on_child_order_changed));
	}

	// Outer container for this component entry.
	VBoxContainer *entry = memnew(VBoxContainer);
	component_container->add_child(entry);

	int entry_index = component_container->get_child_count() - 1;

	// Header bar: [fold toggle] [icon] [name] [delete button].
	HBoxContainer *header = memnew(HBoxContainer);
	entry->add_child(header);

	// Indentation spacer for nested items.
	if (p_depth > 0) {
		Control *spacer = memnew(Control);
		spacer->set_custom_minimum_size(Size2(16 * p_depth * EDSCALE, 0));
		header->add_child(spacer);
	}

	Button *fold_btn = memnew(Button);
	fold_btn->set_toggle_mode(true);
	fold_btn->set_pressed_no_signal(false);
	fold_btn->set_text(String::utf8("\u25BC")); // Down arrow (expanded).
	fold_btn->set_tooltip_text(TTR("Toggle component properties"));
	fold_btn->set_flat(true);
	fold_btn->set_custom_minimum_size(Size2(24 * EDSCALE, 0));
	fold_btn->connect(SceneStringName(pressed), callable_mp(this, &GameObjectComponentList::_toggle_component).bind(entry_index));
	header->add_child(fold_btn);

	// Icon.
	TextureRect *icon_rect = memnew(TextureRect);
	icon_rect->set_texture(EditorNode::get_singleton()->get_object_icon(p_node));
	icon_rect->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	icon_rect->set_custom_minimum_size(Size2(16 * EDSCALE, 16 * EDSCALE));
	icon_rect->set_v_size_flags(SIZE_SHRINK_CENTER);
	header->add_child(icon_rect);

	// Name label.
	Label *name_label = memnew(Label);
	name_label->set_text(p_node->get_name());
	name_label->set_h_size_flags(SIZE_EXPAND_FILL);
	header->add_child(name_label);

	// Delete button.
	Button *delete_btn = memnew(Button);
	delete_btn->set_text(String::utf8("\u2716")); // X mark.
	delete_btn->set_tooltip_text(TTR("Delete component"));
	delete_btn->set_flat(true);
	delete_btn->set_custom_minimum_size(Size2(24 * EDSCALE, 0));
	delete_btn->connect(SceneStringName(pressed), callable_mp(this, &GameObjectComponentList::_delete_component).bind(entry_index));
	header->add_child(delete_btn);

	// Separator under header.
	HSeparator *sep = memnew(HSeparator);
	entry->add_child(sep);

	// Sub-inspector for this component's properties.
	EditorInspector *sub_inspector = memnew(EditorInspector);
	sub_inspector->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	sub_inspector->set_use_doc_hints(true);
	sub_inspector->set_use_folding(true);
	sub_inspector->set_draw_focus_border(false);
	sub_inspector->set_focus_mode(Control::FOCUS_NONE);
	sub_inspector->edit(p_node);
	entry->add_child(sub_inspector);

	// Recurse into children (nested components).
	for (int i = 0; i < p_node->get_child_count(); i++) {
		_rebuild_list_recursive(p_node->get_child(i), p_depth + 1);
	}
}

void GameObjectComponentList::_toggle_component(int p_index) {
	if (p_index < 0 || p_index >= component_container->get_child_count()) {
		return;
	}

	VBoxContainer *entry = Object::cast_to<VBoxContainer>(component_container->get_child(p_index));
	ERR_FAIL_NULL(entry);

	// The sub-inspector is the last child of the entry (after header and separator).
	EditorInspector *sub_inspector = Object::cast_to<EditorInspector>(entry->get_child(entry->get_child_count() - 1));
	ERR_FAIL_NULL(sub_inspector);

	// Find the fold button in the header (may be preceded by a spacer).
	HBoxContainer *header = Object::cast_to<HBoxContainer>(entry->get_child(0));
	ERR_FAIL_NULL(header);
	Button *fold_btn = nullptr;
	for (int i = 0; i < header->get_child_count(); i++) {
		fold_btn = Object::cast_to<Button>(header->get_child(i));
		if (fold_btn && fold_btn->is_toggle_mode()) {
			break;
		}
		fold_btn = nullptr;
	}
	ERR_FAIL_NULL(fold_btn);

	bool collapsed = fold_btn->is_pressed();
	sub_inspector->set_visible(!collapsed);
	fold_btn->set_text(collapsed ? String::utf8("\u25B6") : String::utf8("\u25BC")); // Right arrow / Down arrow.
}

void GameObjectComponentList::_delete_component(int p_index) {
	if (!game_object || p_index < 0 || p_index >= displayed_nodes.size()) {
		return;
	}

	Node *child = displayed_nodes[p_index];
	ERR_FAIL_NULL(child);

	Node *parent = child->get_parent();
	ERR_FAIL_NULL(parent);

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Delete Component"));
	undo_redo->add_do_method(parent, "remove_child", child);
	undo_redo->add_undo_method(parent, "add_child", child, true);
	undo_redo->add_undo_method(child, "set_owner", EditorNode::get_singleton()->get_edited_scene());
	undo_redo->add_undo_reference(child);
	undo_redo->commit_action();

	// Force rebuild since nested deletions won't trigger game_object's child_order_changed.
	_rebuild_list();
}

Node *GameObjectComponentList::_find_target_parent(Node *p_child) {
	if (!game_object) {
		return nullptr;
	}

	// 3D: CollisionShape3D / CollisionPolygon3D -> nest under CollisionObject3D.
	if (Object::cast_to<CollisionShape3D>(p_child) || Object::cast_to<CollisionPolygon3D>(p_child)) {
		for (int i = 0; i < game_object->get_child_count(); i++) {
			Node *sibling = game_object->get_child(i);
			if (Object::cast_to<CollisionObject3D>(sibling)) {
				return sibling;
			}
		}
	}

	// 3D: MeshInstance3D -> nest under PhysicsBody3D.
	if (Object::cast_to<MeshInstance3D>(p_child)) {
		for (int i = 0; i < game_object->get_child_count(); i++) {
			Node *sibling = game_object->get_child(i);
			if (Object::cast_to<PhysicsBody3D>(sibling)) {
				return sibling;
			}
		}
	}

	// 3D: VehicleWheel3D -> nest under VehicleBody3D.
	if (Object::cast_to<VehicleWheel3D>(p_child)) {
		for (int i = 0; i < game_object->get_child_count(); i++) {
			Node *sibling = game_object->get_child(i);
			if (Object::cast_to<VehicleBody3D>(sibling)) {
				return sibling;
			}
		}
	}

	// 2D: CollisionShape2D / CollisionPolygon2D -> nest under CollisionObject2D.
	if (Object::cast_to<CollisionShape2D>(p_child) || Object::cast_to<CollisionPolygon2D>(p_child)) {
		for (int i = 0; i < game_object->get_child_count(); i++) {
			Node *sibling = game_object->get_child(i);
			if (Object::cast_to<CollisionObject2D>(sibling)) {
				return sibling;
			}
		}
	}

	// 2D: Sprite2D -> nest under PhysicsBody2D.
	if (Object::cast_to<Sprite2D>(p_child)) {
		for (int i = 0; i < game_object->get_child_count(); i++) {
			Node *sibling = game_object->get_child(i);
			if (Object::cast_to<PhysicsBody2D>(sibling)) {
				return sibling;
			}
		}
	}

	// Default: add as direct child of the GameObject.
	return game_object;
}

void GameObjectComponentList::_reparent_existing_children(Node *p_new_body) {
	if (!game_object) {
		return;
	}

	Node *edited_scene = EditorNode::get_singleton()->get_edited_scene();
	ERR_FAIL_NULL(edited_scene);

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();

	// Collect children to reparent (can't modify while iterating).
	Vector<Node *> to_reparent;

	if (Object::cast_to<CollisionObject3D>(p_new_body)) {
		for (int i = 0; i < game_object->get_child_count(); i++) {
			Node *child = game_object->get_child(i);
			if (child == p_new_body) {
				continue;
			}
			if (Object::cast_to<CollisionShape3D>(child) ||
					Object::cast_to<CollisionPolygon3D>(child) ||
					Object::cast_to<MeshInstance3D>(child)) {
				to_reparent.push_back(child);
			}
		}
	}

	if (Object::cast_to<CollisionObject2D>(p_new_body)) {
		for (int i = 0; i < game_object->get_child_count(); i++) {
			Node *child = game_object->get_child(i);
			if (child == p_new_body) {
				continue;
			}
			if (Object::cast_to<CollisionShape2D>(child) ||
					Object::cast_to<CollisionPolygon2D>(child) ||
					Object::cast_to<Sprite2D>(child)) {
				to_reparent.push_back(child);
			}
		}
	}

	for (Node *child : to_reparent) {
		undo_redo->add_do_method(game_object, "remove_child", child);
		undo_redo->add_do_method(p_new_body, "add_child", child, true);
		undo_redo->add_do_method(child, "set_owner", edited_scene);
		// Undo: move back to game_object.
		undo_redo->add_undo_method(p_new_body, "remove_child", child);
		undo_redo->add_undo_method(game_object, "add_child", child, true);
		undo_redo->add_undo_method(child, "set_owner", edited_scene);
	}
}

void GameObjectComponentList::_update_warnings() {
	if (!warning_label || !game_object) {
		return;
	}

	Vector<String> warnings;

	// Check all descendants for hierarchy issues.
	bool has_collision_object_3d = false;
	bool has_collision_shape_3d = false;
	bool has_collision_object_2d = false;
	bool has_collision_shape_2d = false;
	bool has_vehicle_body = false;
	bool has_vehicle_wheel = false;
	bool loose_collision_shape_3d = false;
	bool loose_collision_shape_2d = false;
	bool loose_vehicle_wheel = false;

	for (int i = 0; i < game_object->get_child_count(); i++) {
		Node *child = game_object->get_child(i);

		if (Object::cast_to<CollisionObject3D>(child)) {
			has_collision_object_3d = true;
			// Check if this body has collision shapes.
			bool body_has_shape = false;
			for (int j = 0; j < child->get_child_count(); j++) {
				if (Object::cast_to<CollisionShape3D>(child->get_child(j)) ||
						Object::cast_to<CollisionPolygon3D>(child->get_child(j))) {
					body_has_shape = true;
					break;
				}
			}
			if (!body_has_shape) {
				warnings.push_back(TTR("Physics body \"") + child->get_name() + TTR("\" has no collision shape. Add a CollisionShape component."));
			}
		}

		if (Object::cast_to<CollisionObject2D>(child)) {
			has_collision_object_2d = true;
			bool body_has_shape = false;
			for (int j = 0; j < child->get_child_count(); j++) {
				if (Object::cast_to<CollisionShape2D>(child->get_child(j)) ||
						Object::cast_to<CollisionPolygon2D>(child->get_child(j))) {
					body_has_shape = true;
					break;
				}
			}
			if (!body_has_shape) {
				warnings.push_back(TTR("Physics body \"") + child->get_name() + TTR("\" has no collision shape. Add a CollisionShape2D component."));
			}
		}

		if (Object::cast_to<CollisionShape3D>(child) || Object::cast_to<CollisionPolygon3D>(child)) {
			has_collision_shape_3d = true;
			loose_collision_shape_3d = true;
		}

		if (Object::cast_to<CollisionShape2D>(child) || Object::cast_to<CollisionPolygon2D>(child)) {
			has_collision_shape_2d = true;
			loose_collision_shape_2d = true;
		}

		if (Object::cast_to<VehicleBody3D>(child)) {
			has_vehicle_body = true;
		}

		if (Object::cast_to<VehicleWheel3D>(child)) {
			has_vehicle_wheel = true;
			loose_vehicle_wheel = true;
		}
	}

	// Loose collision shapes (direct children, not under a body).
	if (loose_collision_shape_3d) {
		warnings.push_back(TTR("CollisionShape needs a physics body parent. Add a RigidBody3D, StaticBody3D, or CharacterBody3D component."));
	}
	if (loose_collision_shape_2d) {
		warnings.push_back(TTR("CollisionShape2D needs a physics body parent. Add a RigidBody2D, StaticBody2D, or CharacterBody2D component."));
	}
	if (loose_vehicle_wheel) {
		warnings.push_back(TTR("VehicleWheel3D needs a VehicleBody3D parent."));
	}

	if (warnings.is_empty()) {
		warning_label->set_visible(false);
	} else {
		warning_label->set_visible(true);
		warning_label->set_text(String("\n").join(warnings));
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

void GameObjectComponentList::_on_create_confirmed() {
	if (!game_object || !create_dialog) {
		return;
	}

	Node *child = Object::cast_to<Node>(create_dialog->instantiate_selected());
	ERR_FAIL_NULL(child);

	Node *edited_scene = EditorNode::get_singleton()->get_edited_scene();
	ERR_FAIL_NULL(edited_scene);

	// Determine correct parent for auto-organization.
	Node *target_parent = _find_target_parent(child);
	if (!target_parent) {
		target_parent = game_object;
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Add Component"));
	undo_redo->add_do_method(target_parent, "add_child", child, true);
	undo_redo->add_do_method(child, "set_owner", edited_scene);
	undo_redo->add_do_reference(child);
	undo_redo->add_undo_method(target_parent, "remove_child", child);

	// If adding a physics body, reparent existing loose children under it.
	if (Object::cast_to<CollisionObject3D>(child) || Object::cast_to<CollisionObject2D>(child)) {
		_reparent_existing_children(child);
	}

	undo_redo->commit_action();

	// Force rebuild since nested additions won't trigger game_object's child_order_changed.
	_rebuild_list();
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

	// Determine correct parent for auto-organization.
	Node *target_parent = _find_target_parent(child);
	if (!target_parent) {
		target_parent = game_object;
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Add Script Component"));
	undo_redo->add_do_method(target_parent, "add_child", child, true);
	undo_redo->add_do_method(child, "set_owner", edited_scene);
	undo_redo->add_do_reference(child);
	undo_redo->add_undo_method(target_parent, "remove_child", child);

	// If adding a physics body, reparent existing loose children under it.
	if (Object::cast_to<CollisionObject3D>(child) || Object::cast_to<CollisionObject2D>(child)) {
		_reparent_existing_children(child);
	}

	undo_redo->commit_action();

	// Force rebuild since nested additions won't trigger game_object's child_order_changed.
	_rebuild_list();
}

void GameObjectComponentList::_on_child_order_changed() {
	_rebuild_list();
}

void GameObjectComponentList::set_game_object(Node *p_game_object) {
	if (game_object && game_object->is_connected("child_order_changed", callable_mp(this, &GameObjectComponentList::_on_child_order_changed))) {
		game_object->disconnect("child_order_changed", callable_mp(this, &GameObjectComponentList::_on_child_order_changed));
	}
	game_object = p_game_object;
	if (game_object) {
		game_object->connect("child_order_changed", callable_mp(this, &GameObjectComponentList::_on_child_order_changed));
	}
	_rebuild_list();
}

GameObjectComponentList::GameObjectComponentList() {
	// Header.
	Label *header = memnew(Label);
	header->set_text(TTR("Components"));
	header->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_LEFT);
	add_child(header);

	// Component container (rebuilt dynamically).
	component_container = memnew(VBoxContainer);
	add_child(component_container);

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

	// Warning label.
	warning_label = memnew(Label);
	warning_label->set_visible(false);
	warning_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	warning_label->add_theme_color_override("font_color", Color(1.0, 0.8, 0.2)); // Yellow/orange.
	add_child(warning_label);
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
