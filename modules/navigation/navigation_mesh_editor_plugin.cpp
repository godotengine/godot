/*************************************************************************/
/*  navigation_mesh_editor_plugin.cpp                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifdef TOOLS_ENABLED
#include "navigation_mesh_editor_plugin.h"

#include "core/io/marshalls.h"
#include "core/io/resource_saver.h"
#include "navigation_mesh_generator.h"
#include "scene/3d/mesh_instance.h"
#include "scene/gui/box_container.h"

void NavigationMeshEditor::_node_removed(Node *p_node) {
	if (p_node == node) {
		node = nullptr;

		hide();
	}
}

void NavigationMeshEditor::_notification(int p_option) {
	if (p_option == NOTIFICATION_ENTER_TREE) {
		button_bake->set_icon(get_icon("Bake", "EditorIcons"));
		button_reset->set_icon(get_icon("Reload", "EditorIcons"));
	}
}

void NavigationMeshEditor::_bake_pressed() {
	button_bake->set_pressed(false);

	ERR_FAIL_COND(!node);
	if (!node->get_navigation_mesh().is_valid()) {
		err_dialog->set_text(TTR("A NavigationMesh resource must be set or created for this node to work."));
		err_dialog->popup_centered_minsize();
		return;
	}

	NavigationMeshGenerator::get_singleton()->clear(node->get_navigation_mesh());
	NavigationMeshGenerator::get_singleton()->bake(node->get_navigation_mesh(), node);

	node->update_gizmo();
}

void NavigationMeshEditor::_clear_pressed() {
	if (node) {
		NavigationMeshGenerator::get_singleton()->clear(node->get_navigation_mesh());
	}

	button_bake->set_pressed(false);
	bake_info->set_text("");

	if (node) {
		node->update_gizmo();
	}
}

void NavigationMeshEditor::edit(NavigationMeshInstance *p_nav_mesh_instance) {
	if (p_nav_mesh_instance == nullptr || node == p_nav_mesh_instance) {
		return;
	}

	node = p_nav_mesh_instance;
}

void NavigationMeshEditor::_bind_methods() {
	ClassDB::bind_method("_bake_pressed", &NavigationMeshEditor::_bake_pressed);
	ClassDB::bind_method("_clear_pressed", &NavigationMeshEditor::_clear_pressed);
}

NavigationMeshEditor::NavigationMeshEditor() {
	bake_hbox = memnew(HBoxContainer);

	button_bake = memnew(ToolButton);
	bake_hbox->add_child(button_bake);
	button_bake->set_toggle_mode(true);
	button_bake->set_text(TTR("Bake NavMesh"));
	button_bake->connect("pressed", this, "_bake_pressed");

	button_reset = memnew(ToolButton);
	bake_hbox->add_child(button_reset);
	// No button text, we only use a revert icon which is set when entering the tree.
	button_reset->set_tooltip(TTR("Clear the navigation mesh."));
	button_reset->connect("pressed", this, "_clear_pressed");

	bake_info = memnew(Label);
	bake_hbox->add_child(bake_info);

	err_dialog = memnew(AcceptDialog);
	add_child(err_dialog);
	node = nullptr;
}

NavigationMeshEditor::~NavigationMeshEditor() {
}

void NavigationMeshEditorPlugin::edit(Object *p_object) {
	navigation_mesh_editor->edit(Object::cast_to<NavigationMeshInstance>(p_object));
}

bool NavigationMeshEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("NavigationMeshInstance");
}

void NavigationMeshEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		navigation_mesh_editor->show();
		navigation_mesh_editor->bake_hbox->show();
	} else {
		navigation_mesh_editor->hide();
		navigation_mesh_editor->bake_hbox->hide();
		navigation_mesh_editor->edit(nullptr);
	}
}

NavigationMeshEditorPlugin::NavigationMeshEditorPlugin(EditorNode *p_node) {
	editor = p_node;
	navigation_mesh_editor = memnew(NavigationMeshEditor);
	editor->get_viewport()->add_child(navigation_mesh_editor);
	add_control_to_container(CONTAINER_SPATIAL_EDITOR_MENU, navigation_mesh_editor->bake_hbox);
	navigation_mesh_editor->hide();
	navigation_mesh_editor->bake_hbox->hide();
}

NavigationMeshEditorPlugin::~NavigationMeshEditorPlugin() {
}

#endif
