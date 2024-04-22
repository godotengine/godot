/**************************************************************************/
/*  visual_shape_2d_editor_plugin.cpp                                     */
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

#include "visual_shape_2d_editor_plugin.h"

#include "canvas_item_editor_plugin.h"
#include "core/math/geometry_2d.h"
#include "editor/editor_node.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/gui/editor_toaster.h"
#include "editor/scene_tree_dock.h"
#include "scene/2d/light_occluder_2d.h"
#include "scene/2d/mesh_instance_2d.h"
#include "scene/2d/physics/collision_shape_2d.h"
#include "scene/2d/polygon_2d.h"
#include "scene/2d/visual_shape_2d.h"
#include "scene/gui/menu_button.h"
#include "scene/resources/2d/capsule_shape_2d.h"
#include "scene/resources/2d/circle_shape_2d.h"
#include "scene/resources/2d/convex_polygon_shape_2d.h"
#include "scene/resources/2d/rectangle_shape_2d.h"

void VisualShape2DEditor::edit(VisualShape2D *p_visual_shape_2d) {
	visual_shape_2d = p_visual_shape_2d;
}

void VisualShape2DEditor::_menu_option(int p_option) {
	if (!visual_shape_2d) {
		return;
	}

	if ((p_option == MENU_OPTION_CONVERT_TO_MESH_2D || p_option == MENU_OPTION_CONVERT_TO_POLYGON_2D) && visual_shape_2d != get_tree()->get_edited_scene_root() && visual_shape_2d->get_owner() != get_tree()->get_edited_scene_root()) {
		EditorToaster::get_singleton()->popup_str(TTR("Can't convert a VisualShape from a foreign scene."), EditorToaster::SEVERITY_ERROR);
		return;
	}

	switch (p_option) {
		case MENU_OPTION_CONVERT_TO_MESH_2D: {
			_convert_to_mesh_2d_node();
		} break;
		case MENU_OPTION_CONVERT_TO_POLYGON_2D: {
			_convert_to_polygon_2d_node();
		} break;
		case MENU_OPTION_CREATE_COLLISION_SHAPE_2D: {
			_create_collision_shape_2d_node();
		} break;
		case MENU_OPTION_CREATE_LIGHT_OCCLUDER_2D: {
			_create_light_occluder_2d_node();
		} break;
	}
}

void VisualShape2DEditor::_convert_to_mesh_2d_node() {
	PackedVector2Array points = visual_shape_2d->get_points();
	if (points.size() < 3) {
		EditorToaster::get_singleton()->popup_str(TTR("Invalid geometry, can't replace by mesh."), EditorToaster::SEVERITY_ERROR);
		return;
	}

	Vector<int> poly = Geometry2D::triangulate_polygon(points);

	Ref<ArrayMesh> mesh;
	mesh.instantiate();

	Array a;
	a.resize(Mesh::ARRAY_MAX);
	a[Mesh::ARRAY_VERTEX] = points;
	a[Mesh::ARRAY_TEX_UV] = visual_shape_2d->get_uvs();
	a[Mesh::ARRAY_INDEX] = poly;

	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, a, Array(), Dictionary(), Mesh::ARRAY_FLAG_USE_2D_VERTICES);

	MeshInstance2D *mesh_instance = memnew(MeshInstance2D);
	mesh_instance->set_mesh(mesh);

	EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
	ur->create_action(TTR("Convert to MeshInstance2D"), UndoRedo::MERGE_DISABLE, visual_shape_2d);
	SceneTreeDock::get_singleton()->replace_node(visual_shape_2d, mesh_instance);
	ur->commit_action(false);
}

void VisualShape2DEditor::_convert_to_polygon_2d_node() {
	PackedVector2Array points = visual_shape_2d->get_points();
	if (points.is_empty()) {
		EditorToaster::get_singleton()->popup_str(TTR("Invalid geometry, can't create polygon."), EditorToaster::SEVERITY_ERROR);
		return;
	}

	int total_point_count = points.size();
	Point2 offset = visual_shape_2d->get_offset();

	Polygon2D *polygon_2d_instance = memnew(Polygon2D);

	polygon_2d_instance->set_color(visual_shape_2d->get_color());
	polygon_2d_instance->set_offset(offset);
	polygon_2d_instance->set_antialiased(visual_shape_2d->is_antialiased());

	PackedVector2Array vertices;
	vertices.resize(total_point_count);
	Vector2 *vertices_write = vertices.ptrw();

	PackedInt32Array index_array;
	index_array.resize(total_point_count);
	int *index_write = index_array.ptrw();

	for (int i = 0; i < total_point_count; i++) {
		vertices_write[i] = points[i] - offset;
		index_write[i] = i;
	}

	Array polys;
	polys.push_back(index_array);

	PackedVector2Array uvs = Transform2D(0, visual_shape_2d->get_size(), 0, Point2(0, 0)).xform(visual_shape_2d->get_uvs());

	polygon_2d_instance->set_polygon(vertices);
	polygon_2d_instance->set_uv(uvs);
	polygon_2d_instance->set_polygons(polys);

	EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
	ur->create_action(TTR("Convert to Polygon2D"), UndoRedo::MERGE_DISABLE, visual_shape_2d);
	SceneTreeDock::get_singleton()->replace_node(visual_shape_2d, polygon_2d_instance);
	ur->commit_action(false);
}

void VisualShape2DEditor::_create_collision_shape_2d_node() {
	CollisionShape2D *collision_shape_2d_instance = memnew(CollisionShape2D);
	Size2 size = visual_shape_2d->get_size();

	switch (visual_shape_2d->get_shape_type()) {
		case VisualShape2D::SHAPE_RECTANGLE: {
			Ref<RectangleShape2D> shape;
			shape.instantiate();
			shape->set_size(size);
			collision_shape_2d_instance->set_shape(shape);
			collision_shape_2d_instance->translate(visual_shape_2d->get_offset());
		} break;
		case VisualShape2D::SHAPE_CIRCLE: {
			float semi_major = MAX(size.x, size.y) / 2.0;
			float semi_minor = MIN(size.x, size.y) / 2.0;
			float difference = (semi_major - semi_minor) / semi_minor;
			// If there is more than a 10% difference, treat as an oval.
			if (difference > 0.1) {
				// Oval.
				Ref<ConvexPolygonShape2D> shape;
				shape.instantiate();
				shape->set_points(visual_shape_2d->get_points());
				collision_shape_2d_instance->set_shape(shape);
			} else {
				// Circle.
				Ref<CircleShape2D> shape;
				shape.instantiate();
				shape->set_radius(semi_major);
				collision_shape_2d_instance->set_shape(shape);
				collision_shape_2d_instance->translate(visual_shape_2d->get_offset());
			}
		} break;
		case VisualShape2D::SHAPE_CAPSULE: {
			Ref<CapsuleShape2D> shape;
			shape.instantiate();
			shape->set_radius(MIN(size.x, size.y) / 2);
			shape->set_height(MAX(size.x, size.y));
			collision_shape_2d_instance->set_shape(shape);
			if (size.x > size.y) {
				collision_shape_2d_instance->rotate(Math_PI / 2.0);
			}
			collision_shape_2d_instance->translate(visual_shape_2d->get_offset());
		} break;
		case VisualShape2D::SHAPE_EQUILATERAL_TRIANGLE:
		case VisualShape2D::SHAPE_RIGHT_TRIANGLE: {
			Ref<ConvexPolygonShape2D> shape;
			shape.instantiate();
			shape->set_points(visual_shape_2d->get_points());
			collision_shape_2d_instance->set_shape(shape);
		} break;
	}

	EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
	ur->create_action(TTR("Create CollisionShape2D Sibling"), UndoRedo::MERGE_DISABLE, visual_shape_2d);
	ur->add_do_method(this, "_add_as_sibling_or_child", visual_shape_2d, collision_shape_2d_instance);
	ur->add_do_reference(collision_shape_2d_instance);
	ur->add_undo_method(visual_shape_2d != get_tree()->get_edited_scene_root() ? visual_shape_2d->get_parent() : visual_shape_2d, "remove_child", collision_shape_2d_instance);
	ur->commit_action();
}

void VisualShape2DEditor::_create_light_occluder_2d_node() {
	PackedVector2Array points = visual_shape_2d->get_points();
	if (points.is_empty()) {
		EditorToaster::get_singleton()->popup_str(TTR("Invalid geometry, can't create light occluder."), EditorToaster::SEVERITY_ERROR);
		return;
	}

	Ref<OccluderPolygon2D> polygon;
	polygon.instantiate();
	polygon->set_polygon(points);

	LightOccluder2D *light_occluder_2d_instance = memnew(LightOccluder2D);
	light_occluder_2d_instance->set_occluder_polygon(polygon);

	EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
	ur->create_action(TTR("Create LightOccluder2D Sibling"), UndoRedo::MERGE_DISABLE, visual_shape_2d);
	ur->add_do_method(this, "_add_as_sibling_or_child", visual_shape_2d, light_occluder_2d_instance);
	ur->add_do_reference(light_occluder_2d_instance);
	ur->add_undo_method(visual_shape_2d != get_tree()->get_edited_scene_root() ? visual_shape_2d->get_parent() : visual_shape_2d, "remove_child", light_occluder_2d_instance);
	ur->commit_action();
}

void VisualShape2DEditor::_add_as_sibling_or_child(Node *p_own_node, Node *p_new_node) {
	// Can't make sibling if own node is scene root.
	if (p_own_node != get_tree()->get_edited_scene_root()) {
		p_own_node->get_parent()->add_child(p_new_node, true);
		Object::cast_to<Node2D>(p_new_node)->set_transform(Object::cast_to<Node2D>(p_own_node)->get_transform() * Object::cast_to<Node2D>(p_new_node)->get_transform());
	} else {
		p_own_node->add_child(p_new_node, true);
	}

	p_new_node->set_owner(get_tree()->get_edited_scene_root());
}

void VisualShape2DEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			options->set_button_icon(get_editor_theme_icon(SNAME("VisualShape2D")));

			options->get_popup()->set_item_icon(MENU_OPTION_CONVERT_TO_MESH_2D, get_editor_theme_icon(SNAME("MeshInstance2D")));
			options->get_popup()->set_item_icon(MENU_OPTION_CONVERT_TO_POLYGON_2D, get_editor_theme_icon(SNAME("Polygon2D")));
			options->get_popup()->set_item_icon(MENU_OPTION_CREATE_COLLISION_SHAPE_2D, get_editor_theme_icon(SNAME("CollisionShape2D")));
			options->get_popup()->set_item_icon(MENU_OPTION_CREATE_LIGHT_OCCLUDER_2D, get_editor_theme_icon(SNAME("LightOccluder2D")));
		} break;
	}
}

void VisualShape2DEditor::_bind_methods() {
	ClassDB::bind_method("_add_as_sibling_or_child", &VisualShape2DEditor::_add_as_sibling_or_child);
}

VisualShape2DEditor::VisualShape2DEditor() {
	options = memnew(MenuButton);

	CanvasItemEditor::get_singleton()->add_control_to_menu_panel(options);

	options->set_text(TTR("VisualShape2D"));

	options->get_popup()->add_item(TTR("Convert to MeshInstance2D"), MENU_OPTION_CONVERT_TO_MESH_2D);
	options->get_popup()->add_item(TTR("Convert to Polygon2D"), MENU_OPTION_CONVERT_TO_POLYGON_2D);
	options->get_popup()->add_item(TTR("Create CollisionShape2D Sibling"), MENU_OPTION_CREATE_COLLISION_SHAPE_2D);
	options->get_popup()->add_item(TTR("Create LightOccluder2D Sibling"), MENU_OPTION_CREATE_LIGHT_OCCLUDER_2D);
	options->set_switch_on_hover(true);

	options->get_popup()->connect(SceneStringName(id_pressed), callable_mp(this, &VisualShape2DEditor::_menu_option));
}

void VisualShape2DEditorPlugin::edit(Object *p_object) {
	visual_shape_editor->edit(Object::cast_to<VisualShape2D>(p_object));
}

bool VisualShape2DEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("VisualShape2D");
}

void VisualShape2DEditorPlugin::make_visible(bool p_visible) {
	visual_shape_editor->options->set_visible(p_visible);
}

VisualShape2DEditorPlugin::VisualShape2DEditorPlugin() {
	visual_shape_editor = memnew(VisualShape2DEditor);
	EditorNode::get_singleton()->get_gui_base()->add_child(visual_shape_editor);
	make_visible(false);
}
