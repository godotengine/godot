/*************************************************************************/
/*  mesh_instance_editor_plugin.cpp                                      */
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
#include "mesh_instance_editor_plugin.h"

#include "scene/3d/body_shape.h"
#include "scene/3d/navigation_mesh.h"
#include "scene/3d/physics_body.h"
#include "scene/gui/box_container.h"
#include "spatial_editor_plugin.h"

void MeshInstanceEditor::_node_removed(Node *p_node) {

	if (p_node == node) {
		node = NULL;
		options->hide();
	}
}

void MeshInstanceEditor::edit(MeshInstance *p_mesh) {

	node = p_mesh;
}

void MeshInstanceEditor::_menu_option(int p_option) {

	Ref<Mesh> mesh = node->get_mesh();
	if (mesh.is_null()) {
		err_dialog->set_text(TTR("Mesh is empty!"));
		err_dialog->popup_centered_minsize();
		return;
	}

	switch (p_option) {
		case MENU_OPTION_CREATE_STATIC_TRIMESH_BODY:
		case MENU_OPTION_CREATE_STATIC_CONVEX_BODY: {

			bool trimesh_shape = (p_option == MENU_OPTION_CREATE_STATIC_TRIMESH_BODY);

			EditorSelection *editor_selection = EditorNode::get_singleton()->get_editor_selection();
			UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();

			List<Node *> selection = editor_selection->get_selected_node_list();

			if (selection.empty()) {
				Ref<Shape> shape = trimesh_shape ? mesh->create_trimesh_shape() : mesh->create_convex_shape();
				if (shape.is_null())
					return;

				CollisionShape *cshape = memnew(CollisionShape);
				cshape->set_shape(shape);
				StaticBody *body = memnew(StaticBody);
				body->add_child(cshape);

				Node *owner = node == get_tree()->get_edited_scene_root() ? node : node->get_owner();

				if (trimesh_shape)
					ur->create_action(TTR("Create Static Trimesh Body"));
				else
					ur->create_action(TTR("Create Static Convex Body"));

				ur->add_do_method(node, "add_child", body);
				ur->add_do_method(body, "set_owner", owner);
				ur->add_do_method(cshape, "set_owner", owner);
				ur->add_do_reference(body);
				ur->add_undo_method(node, "remove_child", body);
				ur->commit_action();
				return;
			}

			if (trimesh_shape)
				ur->create_action(TTR("Create Static Trimesh Body"));
			else
				ur->create_action(TTR("Create Static Convex Body"));

			for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

				MeshInstance *instance = E->get()->cast_to<MeshInstance>();
				if (!instance)
					continue;

				Ref<Mesh> m = instance->get_mesh();
				if (m.is_null())
					continue;

				Ref<Shape> shape = trimesh_shape ? m->create_trimesh_shape() : m->create_convex_shape();
				if (shape.is_null())
					continue;

				CollisionShape *cshape = memnew(CollisionShape);
				cshape->set_shape(shape);
				StaticBody *body = memnew(StaticBody);
				body->add_child(cshape);

				Node *owner = instance == get_tree()->get_edited_scene_root() ? instance : instance->get_owner();

				ur->add_do_method(instance, "add_child", body);
				ur->add_do_method(body, "set_owner", owner);
				ur->add_do_method(cshape, "set_owner", owner);
				ur->add_do_reference(body);
				ur->add_undo_method(instance, "remove_child", body);
			}

			ur->commit_action();

		} break;

		case MENU_OPTION_CREATE_TRIMESH_COLLISION_SHAPE:
		case MENU_OPTION_CREATE_CONVEX_COLLISION_SHAPE: {

			if (node == get_tree()->get_edited_scene_root()) {
				err_dialog->set_text(TTR("This doesn't work on scene root!"));
				err_dialog->popup_centered_minsize();
				return;
			}

			bool trimesh_shape = (p_option == MENU_OPTION_CREATE_TRIMESH_COLLISION_SHAPE);

			Ref<Shape> shape = trimesh_shape ? mesh->create_trimesh_shape() : mesh->create_convex_shape();
			if (shape.is_null())
				return;

			CollisionShape *cshape = memnew(CollisionShape);
			cshape->set_shape(shape);

			Node *owner = node->get_owner();

			UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();

			if (trimesh_shape)
				ur->create_action(TTR("Create Trimesh Shape"));
			else
				ur->create_action(TTR("Create Convex Shape"));

			ur->add_do_method(node->get_parent(), "add_child", cshape);
			ur->add_do_method(node->get_parent(), "move_child", cshape, node->get_index() + 1);
			ur->add_do_method(cshape, "set_owner", owner);
			ur->add_do_reference(cshape);
			ur->add_undo_method(node->get_parent(), "remove_child", cshape);
			ur->commit_action();

		} break;

		case MENU_OPTION_CREATE_NAVMESH: {

			Ref<NavigationMesh> nmesh = memnew(NavigationMesh);

			if (nmesh.is_null())
				return;

			nmesh->create_from_mesh(mesh);
			NavigationMeshInstance *nmi = memnew(NavigationMeshInstance);
			nmi->set_navigation_mesh(nmesh);

			Node *owner = node == get_tree()->get_edited_scene_root() ? node : node->get_owner();

			UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
			ur->create_action(TTR("Create Navigation Mesh"));

			ur->add_do_method(node, "add_child", nmi);
			ur->add_do_method(nmi, "set_owner", owner);

			ur->add_do_reference(nmi);
			ur->add_undo_method(node, "remove_child", nmi);
			ur->commit_action();
		} break;

		case MENU_OPTION_CREATE_OUTLINE_MESH: {

			outline_dialog->popup_centered(Vector2(200, 90));
		} break;
	}
}

void MeshInstanceEditor::_create_outline_mesh() {

	Ref<Mesh> mesh = node->get_mesh();
	if (mesh.is_null()) {
		err_dialog->set_text(TTR("MeshInstance lacks a Mesh!"));
		err_dialog->popup_centered_minsize();
		return;
	}

	if (mesh->get_surface_count() == 0) {
		err_dialog->set_text(TTR("Mesh has not surface to create outlines from!"));
		err_dialog->popup_centered_minsize();
		return;
	}

	Ref<Mesh> mesho = mesh->create_outline(outline_size->get_value());

	if (mesho.is_null()) {
		err_dialog->set_text(TTR("Could not create outline!"));
		err_dialog->popup_centered_minsize();
		return;
	}

	MeshInstance *mi = memnew(MeshInstance);
	mi->set_mesh(mesho);
	Node *owner = node->get_owner();
	if (get_tree()->get_edited_scene_root() == node) {
		owner = node;
	}

	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();

	ur->create_action(TTR("Create Outline"));

	ur->add_do_method(node, "add_child", mi);
	ur->add_do_method(mi, "set_owner", owner);

	ur->add_do_reference(mi);
	ur->add_undo_method(node, "remove_child", mi);
	ur->commit_action();
}

void MeshInstanceEditor::_bind_methods() {

	ClassDB::bind_method("_menu_option", &MeshInstanceEditor::_menu_option);
	ClassDB::bind_method("_create_outline_mesh", &MeshInstanceEditor::_create_outline_mesh);
}

MeshInstanceEditor::MeshInstanceEditor() {

	options = memnew(MenuButton);
	SpatialEditor::get_singleton()->add_control_to_menu_panel(options);

	options->set_text(TTR("Mesh"));
	options->set_icon(EditorNode::get_singleton()->get_gui_base()->get_icon("MeshInstance", "EditorIcons"));

	options->get_popup()->add_item(TTR("Create Trimesh Static Body"), MENU_OPTION_CREATE_STATIC_TRIMESH_BODY);
	options->get_popup()->add_item(TTR("Create Convex Static Body"), MENU_OPTION_CREATE_STATIC_CONVEX_BODY);
	options->get_popup()->add_separator();
	options->get_popup()->add_item(TTR("Create Trimesh Collision Sibling"), MENU_OPTION_CREATE_TRIMESH_COLLISION_SHAPE);
	options->get_popup()->add_item(TTR("Create Convex Collision Sibling"), MENU_OPTION_CREATE_CONVEX_COLLISION_SHAPE);
	options->get_popup()->add_separator();
	options->get_popup()->add_item(TTR("Create Navigation Mesh"), MENU_OPTION_CREATE_NAVMESH);
	options->get_popup()->add_separator();
	options->get_popup()->add_item(TTR("Create Outline Mesh.."), MENU_OPTION_CREATE_OUTLINE_MESH);

	options->get_popup()->connect("id_pressed", this, "_menu_option");

	outline_dialog = memnew(ConfirmationDialog);
	outline_dialog->set_title(TTR("Create Outline Mesh"));
	outline_dialog->get_ok()->set_text(TTR("Create"));

	VBoxContainer *outline_dialog_vbc = memnew(VBoxContainer);
	outline_dialog->add_child(outline_dialog_vbc);
	//outline_dialog->set_child_rect(outline_dialog_vbc);

	outline_size = memnew(SpinBox);
	outline_size->set_min(0.001);
	outline_size->set_max(1024);
	outline_size->set_step(0.001);
	outline_size->set_value(0.05);
	outline_dialog_vbc->add_margin_child(TTR("Outline Size:"), outline_size);

	add_child(outline_dialog);
	outline_dialog->connect("confirmed", this, "_create_outline_mesh");

	err_dialog = memnew(AcceptDialog);
	add_child(err_dialog);
}

void MeshInstanceEditorPlugin::edit(Object *p_object) {

	mesh_editor->edit(p_object->cast_to<MeshInstance>());
}

bool MeshInstanceEditorPlugin::handles(Object *p_object) const {

	return p_object->is_class("MeshInstance");
}

void MeshInstanceEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		mesh_editor->options->show();
	} else {

		mesh_editor->options->hide();
		mesh_editor->edit(NULL);
	}
}

MeshInstanceEditorPlugin::MeshInstanceEditorPlugin(EditorNode *p_node) {

	editor = p_node;
	mesh_editor = memnew(MeshInstanceEditor);
	editor->get_viewport()->add_child(mesh_editor);

	mesh_editor->options->hide();
}

MeshInstanceEditorPlugin::~MeshInstanceEditorPlugin() {
}
