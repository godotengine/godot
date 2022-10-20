/*************************************************************************/
/*  mesh_instance_editor_plugin.cpp                                      */
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

#include "mesh_instance_editor_plugin.h"

#include "editor/editor_scale.h"
#include "scene/3d/collision_shape.h"
#include "scene/3d/navigation_mesh_instance.h"
#include "scene/3d/physics_body.h"
#include "scene/gui/box_container.h"
#include "spatial_editor_plugin.h"

void MeshInstanceEditor::_node_removed(Node *p_node) {
	if (p_node == node) {
		node = nullptr;
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
		case MENU_OPTION_CREATE_STATIC_TRIMESH_BODY: {
			EditorSelection *editor_selection = EditorNode::get_singleton()->get_editor_selection();
			UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();

			List<Node *> selection = editor_selection->get_selected_node_list();

			if (selection.empty()) {
				Ref<Shape> shape = mesh->create_trimesh_shape();
				if (shape.is_null()) {
					err_dialog->set_text(TTR("Couldn't create a Trimesh collision shape."));
					err_dialog->popup_centered_minsize();
					return;
				}

				CollisionShape *cshape = memnew(CollisionShape);
				cshape->set_shape(shape);
				StaticBody *body = memnew(StaticBody);
				body->add_child(cshape);

				Node *owner = get_tree()->get_edited_scene_root();

				ur->create_action(TTR("Create Static Trimesh Body"));
				ur->add_do_method(node, "add_child", body);
				ur->add_do_method(body, "set_owner", owner);
				ur->add_do_method(cshape, "set_owner", owner);
				ur->add_do_reference(body);
				ur->add_undo_method(node, "remove_child", body);
				ur->commit_action();
				return;
			}

			ur->create_action(TTR("Create Static Trimesh Body"));

			for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {
				MeshInstance *instance = Object::cast_to<MeshInstance>(E->get());
				if (!instance) {
					continue;
				}

				Ref<Mesh> m = instance->get_mesh();
				if (m.is_null()) {
					continue;
				}

				Ref<Shape> shape = m->create_trimesh_shape();
				if (shape.is_null()) {
					continue;
				}

				CollisionShape *cshape = memnew(CollisionShape);
				cshape->set_shape(shape);
				StaticBody *body = memnew(StaticBody);
				body->add_child(cshape);

				Node *owner = get_tree()->get_edited_scene_root();

				ur->add_do_method(instance, "add_child", body);
				ur->add_do_method(body, "set_owner", owner);
				ur->add_do_method(cshape, "set_owner", owner);
				ur->add_do_reference(body);
				ur->add_undo_method(instance, "remove_child", body);
			}

			ur->commit_action();

		} break;

		case MENU_OPTION_CREATE_TRIMESH_COLLISION_SHAPE: {
			if (node == get_tree()->get_edited_scene_root()) {
				err_dialog->set_text(TTR("This doesn't work on scene root!"));
				err_dialog->popup_centered_minsize();
				return;
			}

			Ref<Shape> shape = mesh->create_trimesh_shape();
			if (shape.is_null()) {
				return;
			}

			CollisionShape *cshape = memnew(CollisionShape);
			cshape->set_shape(shape);
			cshape->set_transform(node->get_transform());

			Node *owner = get_tree()->get_edited_scene_root();

			UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();

			ur->create_action(TTR("Create Trimesh Static Shape"));

			ur->add_do_method(node->get_parent(), "add_child", cshape);
			ur->add_do_method(node->get_parent(), "move_child", cshape, node->get_index() + 1);
			ur->add_do_method(cshape, "set_owner", owner);
			ur->add_do_reference(cshape);
			ur->add_undo_method(node->get_parent(), "remove_child", cshape);
			ur->commit_action();
		} break;

		case MENU_OPTION_CREATE_SINGLE_CONVEX_COLLISION_SHAPE:
		case MENU_OPTION_CREATE_SIMPLIFIED_CONVEX_COLLISION_SHAPE: {
			if (node == get_tree()->get_edited_scene_root()) {
				err_dialog->set_text(TTR("Can't create a single convex collision shape for the scene root."));
				err_dialog->popup_centered_minsize();
				return;
			}

			bool simplify = (p_option == MENU_OPTION_CREATE_SIMPLIFIED_CONVEX_COLLISION_SHAPE);

			Ref<Shape> shape = mesh->create_convex_shape(true, simplify);

			if (shape.is_null()) {
				err_dialog->set_text(TTR("Couldn't create a single convex collision shape."));
				err_dialog->popup_centered_minsize();
				return;
			}
			UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();

			if (simplify) {
				ur->create_action(TTR("Create Simplified Convex Shape"));
			} else {
				ur->create_action(TTR("Create Single Convex Shape"));
			}

			CollisionShape *cshape = memnew(CollisionShape);
			cshape->set_shape(shape);
			cshape->set_transform(node->get_transform());

			Node *owner = get_tree()->get_edited_scene_root();

			ur->add_do_method(node->get_parent(), "add_child", cshape);
			ur->add_do_method(node->get_parent(), "move_child", cshape, node->get_index() + 1);
			ur->add_do_method(cshape, "set_owner", owner);
			ur->add_do_reference(cshape);
			ur->add_undo_method(node->get_parent(), "remove_child", cshape);

			ur->commit_action();

		} break;

		case MENU_OPTION_CREATE_MULTIPLE_CONVEX_COLLISION_SHAPES: {
			if (node == get_tree()->get_edited_scene_root()) {
				err_dialog->set_text(TTR("Can't create multiple convex collision shapes for the scene root."));
				err_dialog->popup_centered_minsize();
				return;
			}

			Vector<Ref<Shape>> shapes = mesh->convex_decompose();

			if (!shapes.size()) {
				err_dialog->set_text(TTR("Couldn't create any collision shapes."));
				err_dialog->popup_centered_minsize();
				return;
			}
			UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();

			ur->create_action(TTR("Create Multiple Convex Shapes"));

			for (int i = 0; i < shapes.size(); i++) {
				CollisionShape *cshape = memnew(CollisionShape);
				cshape->set_shape(shapes[i]);
				cshape->set_transform(node->get_transform());

				Node *owner = get_tree()->get_edited_scene_root();

				ur->add_do_method(node->get_parent(), "add_child", cshape);
				ur->add_do_method(node->get_parent(), "move_child", cshape, node->get_index() + 1);
				ur->add_do_method(cshape, "set_owner", owner);
				ur->add_do_reference(cshape);
				ur->add_undo_method(node->get_parent(), "remove_child", cshape);
			}
			ur->commit_action();

		} break;

		case MENU_OPTION_CREATE_NAVMESH: {
			Ref<NavigationMesh> nmesh = memnew(NavigationMesh);

			if (nmesh.is_null()) {
				return;
			}

			nmesh->create_from_mesh(mesh);
			NavigationMeshInstance *nmi = memnew(NavigationMeshInstance);
			nmi->set_navigation_mesh(nmesh);

			Node *owner = get_tree()->get_edited_scene_root();

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
		case MENU_OPTION_CREATE_UV2: {
			Ref<ArrayMesh> mesh2 = node->get_mesh();
			if (!mesh2.is_valid()) {
				err_dialog->set_text(TTR("Contained Mesh is not of type ArrayMesh."));
				err_dialog->popup_centered_minsize();
				return;
			}

			Error err = mesh2->lightmap_unwrap(node->get_global_transform());
			if (err != OK) {
				err_dialog->set_text(TTR("UV Unwrap failed, mesh may not be manifold?"));
				err_dialog->popup_centered_minsize();
				return;
			}

		} break;
		case MENU_OPTION_DEBUG_UV1: {
			Ref<Mesh> mesh2 = node->get_mesh();
			if (!mesh2.is_valid()) {
				err_dialog->set_text(TTR("No mesh to debug."));
				err_dialog->popup_centered_minsize();
				return;
			}
			_create_uv_lines(0);
		} break;
		case MENU_OPTION_DEBUG_UV2: {
			Ref<Mesh> mesh2 = node->get_mesh();
			if (!mesh2.is_valid()) {
				err_dialog->set_text(TTR("No mesh to debug."));
				err_dialog->popup_centered_minsize();
				return;
			}
			_create_uv_lines(1);
		} break;
	}
}

struct MeshInstanceEditorEdgeSort {
	Vector2 a;
	Vector2 b;

	bool operator<(const MeshInstanceEditorEdgeSort &p_b) const {
		if (a == p_b.a) {
			return b < p_b.b;
		} else {
			return a < p_b.a;
		}
	}

	MeshInstanceEditorEdgeSort() {}
	MeshInstanceEditorEdgeSort(const Vector2 &p_a, const Vector2 &p_b) {
		if (p_a < p_b) {
			a = p_a;
			b = p_b;
		} else {
			b = p_a;
			a = p_b;
		}
	}
};

void MeshInstanceEditor::_create_uv_lines(int p_layer) {
	Ref<Mesh> mesh = node->get_mesh();
	ERR_FAIL_COND(!mesh.is_valid());

	Set<MeshInstanceEditorEdgeSort> edges;
	uv_lines.clear();
	for (int i = 0; i < mesh->get_surface_count(); i++) {
		if (mesh->surface_get_primitive_type(i) != Mesh::PRIMITIVE_TRIANGLES) {
			continue;
		}
		Array a = mesh->surface_get_arrays(i);

		PoolVector<Vector2> uv = a[p_layer == 0 ? Mesh::ARRAY_TEX_UV : Mesh::ARRAY_TEX_UV2];
		if (uv.size() == 0) {
			err_dialog->set_text(vformat(TTR("Mesh has no UV in layer %d."), p_layer + 1));
			err_dialog->popup_centered_minsize();
			return;
		}

		PoolVector<Vector2>::Read r = uv.read();

		PoolVector<int> indices = a[Mesh::ARRAY_INDEX];
		PoolVector<int>::Read ri;

		int ic;
		bool use_indices;

		if (indices.size()) {
			ic = indices.size();
			ri = indices.read();
			use_indices = true;
		} else {
			ic = uv.size();
			use_indices = false;
		}

		for (int j = 0; j < ic; j += 3) {
			for (int k = 0; k < 3; k++) {
				MeshInstanceEditorEdgeSort edge;
				if (use_indices) {
					edge.a = r[ri[j + k]];
					edge.b = r[ri[j + ((k + 1) % 3)]];
				} else {
					edge.a = r[j + k];
					edge.b = r[j + ((k + 1) % 3)];
				}

				if (edges.has(edge)) {
					continue;
				}

				uv_lines.push_back(edge.a);
				uv_lines.push_back(edge.b);
				edges.insert(edge);
			}
		}
	}

	debug_uv_dialog->popup_centered_minsize();
}

void MeshInstanceEditor::_debug_uv_draw() {
	if (uv_lines.size() == 0) {
		return;
	}

	debug_uv->set_clip_contents(true);
	debug_uv->draw_rect(Rect2(Vector2(), debug_uv->get_size()), get_color("dark_color_3", "Editor"));
	debug_uv->draw_set_transform(Vector2(), 0, debug_uv->get_size());
	// Use a translucent color to allow overlapping triangles to be visible.
	debug_uv->draw_multiline(uv_lines, get_color("mono_color", "Editor") * Color(1, 1, 1, 0.5), Math::round(EDSCALE));
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
	} else if (mesh->get_surface_count() == 1 && mesh->surface_get_primitive_type(0) != Mesh::PRIMITIVE_TRIANGLES) {
		err_dialog->set_text(TTR("Mesh primitive type is not PRIMITIVE_TRIANGLES!"));
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
	Node *owner = get_tree()->get_edited_scene_root();

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
	ClassDB::bind_method("_debug_uv_draw", &MeshInstanceEditor::_debug_uv_draw);
}

MeshInstanceEditor::MeshInstanceEditor() {
	options = memnew(MenuButton);
	options->set_switch_on_hover(true);
	SpatialEditor::get_singleton()->add_control_to_menu_panel(options);

	options->set_text(TTR("Mesh"));
	options->set_icon(EditorNode::get_singleton()->get_gui_base()->get_icon("MeshInstance", "EditorIcons"));

	options->get_popup()->add_item(TTR("Create Trimesh Static Body"), MENU_OPTION_CREATE_STATIC_TRIMESH_BODY);
	options->get_popup()->set_item_tooltip(options->get_popup()->get_item_count() - 1, TTR("Creates a StaticBody and assigns a polygon-based collision shape to it automatically.\nThis is the most accurate (but slowest) option for collision detection."));
	options->get_popup()->add_separator();
	options->get_popup()->add_item(TTR("Create Trimesh Collision Sibling"), MENU_OPTION_CREATE_TRIMESH_COLLISION_SHAPE);
	options->get_popup()->set_item_tooltip(options->get_popup()->get_item_count() - 1, TTR("Creates a polygon-based collision shape.\nThis is the most accurate (but slowest) option for collision detection."));
	options->get_popup()->add_item(TTR("Create Single Convex Collision Sibling"), MENU_OPTION_CREATE_SINGLE_CONVEX_COLLISION_SHAPE);
	options->get_popup()->set_item_tooltip(options->get_popup()->get_item_count() - 1, TTR("Creates a single convex collision shape.\nThis is the fastest (but least accurate) option for collision detection."));
	options->get_popup()->add_item(TTR("Create Simplified Convex Collision Sibling"), MENU_OPTION_CREATE_SIMPLIFIED_CONVEX_COLLISION_SHAPE);
	options->get_popup()->set_item_tooltip(options->get_popup()->get_item_count() - 1, TTR("Creates a simplified convex collision shape.\nThis is similar to single collision shape, but can result in a simpler geometry in some cases, at the cost of accuracy."));
	options->get_popup()->add_item(TTR("Create Multiple Convex Collision Siblings"), MENU_OPTION_CREATE_MULTIPLE_CONVEX_COLLISION_SHAPES);
	options->get_popup()->set_item_tooltip(options->get_popup()->get_item_count() - 1, TTR("Creates a polygon-based collision shape.\nThis is a performance middle-ground between a single convex collision and a polygon-based collision."));
	options->get_popup()->add_separator();
	options->get_popup()->add_item(TTR("Create Navigation Mesh"), MENU_OPTION_CREATE_NAVMESH);
	options->get_popup()->add_separator();
	options->get_popup()->add_item(TTR("Create Outline Mesh..."), MENU_OPTION_CREATE_OUTLINE_MESH);
	options->get_popup()->set_item_tooltip(options->get_popup()->get_item_count() - 1, TTR("Creates a static outline mesh. The outline mesh will have its normals flipped automatically.\nThis can be used instead of the SpatialMaterial Grow property when using that property isn't possible."));
	options->get_popup()->add_separator();
	options->get_popup()->add_item(TTR("View UV1"), MENU_OPTION_DEBUG_UV1);
	options->get_popup()->add_item(TTR("View UV2"), MENU_OPTION_DEBUG_UV2);
	options->get_popup()->add_item(TTR("Unwrap UV2 for Lightmap/AO"), MENU_OPTION_CREATE_UV2);

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

	debug_uv_dialog = memnew(AcceptDialog);
	debug_uv_dialog->set_title(TTR("UV Channel Debug"));
	add_child(debug_uv_dialog);
	debug_uv = memnew(Control);
	debug_uv->set_custom_minimum_size(Size2(600, 600) * EDSCALE);
	debug_uv->connect("draw", this, "_debug_uv_draw");
	debug_uv_dialog->add_child(debug_uv);
}

void MeshInstanceEditorPlugin::edit(Object *p_object) {
	mesh_editor->edit(Object::cast_to<MeshInstance>(p_object));
}

bool MeshInstanceEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("MeshInstance");
}

void MeshInstanceEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		mesh_editor->options->show();
	} else {
		mesh_editor->options->hide();
		mesh_editor->edit(nullptr);
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
