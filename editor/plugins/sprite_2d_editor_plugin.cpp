/**************************************************************************/
/*  sprite_2d_editor_plugin.cpp                                           */
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

#include "sprite_2d_editor_plugin.h"

#include "canvas_item_editor_plugin.h"
#include "core/math/geometry_2d.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/scene_tree_dock.h"
#include "scene/2d/collision_polygon_2d.h"
#include "scene/2d/light_occluder_2d.h"
#include "scene/2d/mesh_instance_2d.h"
#include "scene/2d/polygon_2d.h"
#include "scene/gui/box_container.h"
#include "scene/gui/menu_button.h"
#include "thirdparty/misc/clipper.hpp"

void Sprite2DEditor::_node_removed(Node *p_node) {
	if (p_node == node) {
		node = nullptr;
		options->hide();
	}
}

void Sprite2DEditor::edit(Sprite2D *p_sprite) {
	node = p_sprite;
}

#define PRECISION 10.0

Vector<Vector2> expand(const Vector<Vector2> &points, const Rect2i &rect, float epsilon = 2.0) {
	int size = points.size();
	ERR_FAIL_COND_V(size < 2, Vector<Vector2>());

	ClipperLib::Path subj;
	ClipperLib::PolyTree solution;
	ClipperLib::PolyTree out;

	for (int i = 0; i < points.size(); i++) {
		subj << ClipperLib::IntPoint(points[i].x * PRECISION, points[i].y * PRECISION);
	}
	ClipperLib::ClipperOffset co;
	co.AddPath(subj, ClipperLib::jtMiter, ClipperLib::etClosedPolygon);
	co.Execute(solution, epsilon * PRECISION);

	ClipperLib::PolyNode *p = solution.GetFirst();

	ERR_FAIL_NULL_V(p, points);

	while (p->IsHole()) {
		p = p->GetNext();
	}

	//turn the result into simply polygon (AKA, fix overlap)

	//clamp into the specified rect
	ClipperLib::Clipper cl;
	cl.StrictlySimple(true);
	cl.AddPath(p->Contour, ClipperLib::ptSubject, true);
	//create the clipping rect
	ClipperLib::Path clamp;
	clamp.push_back(ClipperLib::IntPoint(0, 0));
	clamp.push_back(ClipperLib::IntPoint(rect.size.width * PRECISION, 0));
	clamp.push_back(ClipperLib::IntPoint(rect.size.width * PRECISION, rect.size.height * PRECISION));
	clamp.push_back(ClipperLib::IntPoint(0, rect.size.height * PRECISION));
	cl.AddPath(clamp, ClipperLib::ptClip, true);
	cl.Execute(ClipperLib::ctIntersection, out);

	Vector<Vector2> outPoints;
	ClipperLib::PolyNode *p2 = out.GetFirst();
	ERR_FAIL_NULL_V(p2, points);

	while (p2->IsHole()) {
		p2 = p2->GetNext();
	}

	int lasti = p2->Contour.size() - 1;
	Vector2 prev = Vector2(p2->Contour[lasti].X / PRECISION, p2->Contour[lasti].Y / PRECISION);
	for (uint64_t i = 0; i < p2->Contour.size(); i++) {
		Vector2 cur = Vector2(p2->Contour[i].X / PRECISION, p2->Contour[i].Y / PRECISION);
		if (cur.distance_to(prev) > 0.5) {
			outPoints.push_back(cur);
			prev = cur;
		}
	}
	return outPoints;
}

void Sprite2DEditor::_menu_option(int p_option) {
	if (!node) {
		return;
	}

	selected_menu_item = (Menu)p_option;

	switch (p_option) {
		case MENU_OPTION_CONVERT_TO_MESH_2D: {
			debug_uv_dialog->set_ok_button_text(TTR("Create MeshInstance2D"));
			debug_uv_dialog->set_title(TTR("MeshInstance2D Preview"));

			_popup_debug_uv_dialog();
		} break;
		case MENU_OPTION_CONVERT_TO_POLYGON_2D: {
			debug_uv_dialog->set_ok_button_text(TTR("Create Polygon2D"));
			debug_uv_dialog->set_title(TTR("Polygon2D Preview"));

			_popup_debug_uv_dialog();
		} break;
		case MENU_OPTION_CREATE_COLLISION_POLY_2D: {
			debug_uv_dialog->set_ok_button_text(TTR("Create CollisionPolygon2D"));
			debug_uv_dialog->set_title(TTR("CollisionPolygon2D Preview"));

			_popup_debug_uv_dialog();
		} break;
		case MENU_OPTION_CREATE_LIGHT_OCCLUDER_2D: {
			debug_uv_dialog->set_ok_button_text(TTR("Create LightOccluder2D"));
			debug_uv_dialog->set_title(TTR("LightOccluder2D Preview"));

			_popup_debug_uv_dialog();
		} break;
	}
}

void Sprite2DEditor::_popup_debug_uv_dialog() {
	String error_message;
	if (node->get_owner() != get_tree()->get_edited_scene_root()) {
		error_message = TTR("Can't convert a sprite from a foreign scene.");
	}
	Ref<Texture2D> texture = node->get_texture();
	if (texture.is_null()) {
		error_message = TTR("Can't convert an empty sprite to mesh.");
	}
	if (node->get_hframes() > 1 || node->get_vframes() > 1) {
		error_message = TTR("Can't convert a sprite using animation frames to mesh.");
	}

	if (!error_message.is_empty()) {
		err_dialog->set_text(error_message);
		err_dialog->popup_centered();
		return;
	}

	_update_mesh_data();
	debug_uv_dialog->popup_centered();
	debug_uv->queue_redraw();
}

void Sprite2DEditor::_update_mesh_data() {
	ERR_FAIL_NULL(node);
	Ref<Texture2D> texture = node->get_texture();
	ERR_FAIL_COND(texture.is_null());
	Ref<Image> image = texture->get_image();
	ERR_FAIL_COND(image.is_null());

	if (image->is_compressed()) {
		image->decompress();
	}

	// TODO: Add support for Sprite2D's region.
	Rect2 rect;
	rect.size = image->get_size();

	Ref<BitMap> bm;
	bm.instantiate();
	bm->create_from_image_alpha(image);

	int shrink = shrink_pixels->get_value();
	if (shrink > 0) {
		bm->shrink_mask(shrink, rect);
	}

	int grow = grow_pixels->get_value();
	if (grow > 0) {
		bm->grow_mask(grow, rect);
	}

	float epsilon = simplification->get_value();

	Vector<Vector<Vector2>> lines = bm->clip_opaque_to_polygons(rect, epsilon);

	uv_lines.clear();

	computed_vertices.clear();
	computed_uv.clear();
	computed_indices.clear();

	Size2 img_size = image->get_size();
	for (int i = 0; i < lines.size(); i++) {
		lines.write[i] = expand(lines[i], rect, epsilon);
	}

	if (selected_menu_item == MENU_OPTION_CONVERT_TO_MESH_2D) {
		for (int j = 0; j < lines.size(); j++) {
			int index_ofs = computed_vertices.size();

			for (int i = 0; i < lines[j].size(); i++) {
				Vector2 vtx = lines[j][i];
				computed_uv.push_back(vtx / img_size);

				vtx -= rect.position; //offset by rect position

				//flip if flipped
				if (node->is_flipped_h()) {
					vtx.x = rect.size.x - vtx.x - 1.0;
				}
				if (node->is_flipped_v()) {
					vtx.y = rect.size.y - vtx.y - 1.0;
				}

				if (node->is_centered()) {
					vtx -= rect.size / 2.0;
				}

				computed_vertices.push_back(vtx);
			}

			Vector<int> poly = Geometry2D::triangulate_polygon(lines[j]);

			for (int i = 0; i < poly.size(); i += 3) {
				for (int k = 0; k < 3; k++) {
					int idx = i + k;
					int idxn = i + (k + 1) % 3;
					uv_lines.push_back(lines[j][poly[idx]]);
					uv_lines.push_back(lines[j][poly[idxn]]);

					computed_indices.push_back(poly[idx] + index_ofs);
				}
			}
		}
	}

	outline_lines.clear();
	computed_outline_lines.clear();

	if (selected_menu_item == MENU_OPTION_CONVERT_TO_POLYGON_2D || selected_menu_item == MENU_OPTION_CREATE_COLLISION_POLY_2D || selected_menu_item == MENU_OPTION_CREATE_LIGHT_OCCLUDER_2D) {
		outline_lines.resize(lines.size());
		computed_outline_lines.resize(lines.size());
		for (int pi = 0; pi < lines.size(); pi++) {
			Vector<Vector2> ol;
			Vector<Vector2> col;

			ol.resize(lines[pi].size());
			col.resize(lines[pi].size());

			for (int i = 0; i < lines[pi].size(); i++) {
				Vector2 vtx = lines[pi][i];

				ol.write[i] = vtx;

				vtx -= rect.position; //offset by rect position

				//flip if flipped
				if (node->is_flipped_h()) {
					vtx.x = rect.size.x - vtx.x - 1.0;
				}
				if (node->is_flipped_v()) {
					vtx.y = rect.size.y - vtx.y - 1.0;
				}

				if (node->is_centered()) {
					vtx -= rect.size / 2.0;
				}

				col.write[i] = vtx;
			}

			outline_lines.write[pi] = ol;
			computed_outline_lines.write[pi] = col;
		}
	}

	debug_uv->queue_redraw();
}

void Sprite2DEditor::_create_node() {
	switch (selected_menu_item) {
		case MENU_OPTION_CONVERT_TO_MESH_2D: {
			_convert_to_mesh_2d_node();
		} break;
		case MENU_OPTION_CONVERT_TO_POLYGON_2D: {
			_convert_to_polygon_2d_node();
		} break;
		case MENU_OPTION_CREATE_COLLISION_POLY_2D: {
			_create_collision_polygon_2d_node();
		} break;
		case MENU_OPTION_CREATE_LIGHT_OCCLUDER_2D: {
			_create_light_occluder_2d_node();
		} break;
	}
}

void Sprite2DEditor::_convert_to_mesh_2d_node() {
	if (computed_vertices.size() < 3) {
		err_dialog->set_text(TTR("Invalid geometry, can't replace by mesh."));
		err_dialog->popup_centered();
		return;
	}

	Ref<ArrayMesh> mesh;
	mesh.instantiate();

	Array a;
	a.resize(Mesh::ARRAY_MAX);
	a[Mesh::ARRAY_VERTEX] = computed_vertices;
	a[Mesh::ARRAY_TEX_UV] = computed_uv;
	a[Mesh::ARRAY_INDEX] = computed_indices;

	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, a, Array(), Dictionary(), Mesh::ARRAY_FLAG_USE_2D_VERTICES);

	MeshInstance2D *mesh_instance = memnew(MeshInstance2D);
	mesh_instance->set_mesh(mesh);

	EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
	ur->create_action(TTR("Convert to MeshInstance2D"));
	SceneTreeDock::get_singleton()->replace_node(node, mesh_instance);
	ur->commit_action(false);
}

void Sprite2DEditor::_convert_to_polygon_2d_node() {
	if (computed_outline_lines.is_empty()) {
		err_dialog->set_text(TTR("Invalid geometry, can't create polygon."));
		err_dialog->popup_centered();
		return;
	}

	Polygon2D *polygon_2d_instance = memnew(Polygon2D);

	int total_point_count = 0;
	for (int i = 0; i < computed_outline_lines.size(); i++) {
		total_point_count += computed_outline_lines[i].size();
	}

	PackedVector2Array polygon;
	polygon.resize(total_point_count);
	Vector2 *polygon_write = polygon.ptrw();

	PackedVector2Array uvs;
	uvs.resize(total_point_count);
	Vector2 *uvs_write = uvs.ptrw();

	int current_point_index = 0;

	Array polys;
	polys.resize(computed_outline_lines.size());

	for (int i = 0; i < computed_outline_lines.size(); i++) {
		Vector<Vector2> outline = computed_outline_lines[i];
		Vector<Vector2> uv_outline = outline_lines[i];

		PackedInt32Array pia;
		pia.resize(outline.size());
		int *pia_write = pia.ptrw();

		for (int pi = 0; pi < outline.size(); pi++) {
			polygon_write[current_point_index] = outline[pi];
			uvs_write[current_point_index] = uv_outline[pi];
			pia_write[pi] = current_point_index;
			current_point_index++;
		}

		polys[i] = pia;
	}

	polygon_2d_instance->set_uv(uvs);
	polygon_2d_instance->set_polygon(polygon);
	polygon_2d_instance->set_polygons(polys);

	EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
	ur->create_action(TTR("Convert to Polygon2D"));
	SceneTreeDock::get_singleton()->replace_node(node, polygon_2d_instance);
	ur->commit_action(false);
}

void Sprite2DEditor::_create_collision_polygon_2d_node() {
	if (computed_outline_lines.is_empty()) {
		err_dialog->set_text(TTR("Invalid geometry, can't create collision polygon."));
		err_dialog->popup_centered();
		return;
	}

	for (int i = 0; i < computed_outline_lines.size(); i++) {
		Vector<Vector2> outline = computed_outline_lines[i];

		CollisionPolygon2D *collision_polygon_2d_instance = memnew(CollisionPolygon2D);
		collision_polygon_2d_instance->set_polygon(outline);

		EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
		ur->create_action(TTR("Create CollisionPolygon2D Sibling"));
		ur->add_do_method(this, "_add_as_sibling_or_child", node, collision_polygon_2d_instance);
		ur->add_do_reference(collision_polygon_2d_instance);
		ur->add_undo_method(node != this->get_tree()->get_edited_scene_root() ? node->get_parent() : this->get_tree()->get_edited_scene_root(), "remove_child", collision_polygon_2d_instance);
		ur->commit_action();
	}
}

void Sprite2DEditor::_create_light_occluder_2d_node() {
	if (computed_outline_lines.is_empty()) {
		err_dialog->set_text(TTR("Invalid geometry, can't create light occluder."));
		err_dialog->popup_centered();
		return;
	}

	for (int i = 0; i < computed_outline_lines.size(); i++) {
		Vector<Vector2> outline = computed_outline_lines[i];

		Ref<OccluderPolygon2D> polygon;
		polygon.instantiate();

		PackedVector2Array a;
		a.resize(outline.size());
		Vector2 *aw = a.ptrw();
		for (int io = 0; io < outline.size(); io++) {
			aw[io] = outline[io];
		}
		polygon->set_polygon(a);

		LightOccluder2D *light_occluder_2d_instance = memnew(LightOccluder2D);
		light_occluder_2d_instance->set_occluder_polygon(polygon);

		EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
		ur->create_action(TTR("Create LightOccluder2D Sibling"));
		ur->add_do_method(this, "_add_as_sibling_or_child", node, light_occluder_2d_instance);
		ur->add_do_reference(light_occluder_2d_instance);
		ur->add_undo_method(node != this->get_tree()->get_edited_scene_root() ? node->get_parent() : this->get_tree()->get_edited_scene_root(), "remove_child", light_occluder_2d_instance);
		ur->commit_action();
	}
}

void Sprite2DEditor::_add_as_sibling_or_child(Node *p_own_node, Node *p_new_node) {
	// Can't make sibling if own node is scene root
	if (p_own_node != this->get_tree()->get_edited_scene_root()) {
		p_own_node->get_parent()->add_child(p_new_node, true);
		Object::cast_to<Node2D>(p_new_node)->set_transform(Object::cast_to<Node2D>(p_own_node)->get_transform());
	} else {
		p_own_node->add_child(p_new_node, true);
	}

	p_new_node->set_owner(this->get_tree()->get_edited_scene_root());
}

void Sprite2DEditor::_debug_uv_draw() {
	Ref<Texture2D> tex = node->get_texture();
	ERR_FAIL_COND(!tex.is_valid());

	Point2 draw_pos_offset = Point2(1.0, 1.0);
	Size2 draw_size_offset = Size2(2.0, 2.0);

	debug_uv->set_clip_contents(true);
	debug_uv->draw_texture(tex, draw_pos_offset);
	debug_uv->set_custom_minimum_size(tex->get_size() + draw_size_offset);
	debug_uv->draw_set_transform(draw_pos_offset, 0, Size2(1.0, 1.0));

	Color color = Color(1.0, 0.8, 0.7);

	if (selected_menu_item == MENU_OPTION_CONVERT_TO_MESH_2D && uv_lines.size() > 0) {
		debug_uv->draw_multiline(uv_lines, color);

	} else if ((selected_menu_item == MENU_OPTION_CONVERT_TO_POLYGON_2D || selected_menu_item == MENU_OPTION_CREATE_COLLISION_POLY_2D || selected_menu_item == MENU_OPTION_CREATE_LIGHT_OCCLUDER_2D) && outline_lines.size() > 0) {
		for (int i = 0; i < outline_lines.size(); i++) {
			Vector<Vector2> outline = outline_lines[i];

			debug_uv->draw_polyline(outline, color);
			debug_uv->draw_line(outline[0], outline[outline.size() - 1], color);
		}
	}
}

void Sprite2DEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			options->set_icon(get_editor_theme_icon(SNAME("Sprite2D")));

			options->get_popup()->set_item_icon(MENU_OPTION_CONVERT_TO_MESH_2D, get_editor_theme_icon(SNAME("MeshInstance2D")));
			options->get_popup()->set_item_icon(MENU_OPTION_CONVERT_TO_POLYGON_2D, get_editor_theme_icon(SNAME("Polygon2D")));
			options->get_popup()->set_item_icon(MENU_OPTION_CREATE_COLLISION_POLY_2D, get_editor_theme_icon(SNAME("CollisionPolygon2D")));
			options->get_popup()->set_item_icon(MENU_OPTION_CREATE_LIGHT_OCCLUDER_2D, get_editor_theme_icon(SNAME("LightOccluder2D")));
		} break;
	}
}

void Sprite2DEditor::_bind_methods() {
	ClassDB::bind_method("_add_as_sibling_or_child", &Sprite2DEditor::_add_as_sibling_or_child);
}

Sprite2DEditor::Sprite2DEditor() {
	options = memnew(MenuButton);

	CanvasItemEditor::get_singleton()->add_control_to_menu_panel(options);

	options->set_text(TTR("Sprite2D"));

	options->get_popup()->add_item(TTR("Convert to MeshInstance2D"), MENU_OPTION_CONVERT_TO_MESH_2D);
	options->get_popup()->add_item(TTR("Convert to Polygon2D"), MENU_OPTION_CONVERT_TO_POLYGON_2D);
	options->get_popup()->add_item(TTR("Create CollisionPolygon2D Sibling"), MENU_OPTION_CREATE_COLLISION_POLY_2D);
	options->get_popup()->add_item(TTR("Create LightOccluder2D Sibling"), MENU_OPTION_CREATE_LIGHT_OCCLUDER_2D);
	options->set_switch_on_hover(true);

	options->get_popup()->connect("id_pressed", callable_mp(this, &Sprite2DEditor::_menu_option));

	err_dialog = memnew(AcceptDialog);
	add_child(err_dialog);

	debug_uv_dialog = memnew(ConfirmationDialog);
	VBoxContainer *vb = memnew(VBoxContainer);
	debug_uv_dialog->add_child(vb);
	ScrollContainer *scroll = memnew(ScrollContainer);
	scroll->set_custom_minimum_size(Size2(800, 500) * EDSCALE);
	vb->add_margin_child(TTR("Preview:"), scroll, true);
	debug_uv = memnew(Control);
	debug_uv->connect("draw", callable_mp(this, &Sprite2DEditor::_debug_uv_draw));
	scroll->add_child(debug_uv);
	debug_uv_dialog->connect("confirmed", callable_mp(this, &Sprite2DEditor::_create_node));

	HBoxContainer *hb = memnew(HBoxContainer);
	hb->add_child(memnew(Label(TTR("Simplification:"))));
	simplification = memnew(SpinBox);
	simplification->set_min(0.01);
	simplification->set_max(10.00);
	simplification->set_step(0.01);
	simplification->set_value(2);
	hb->add_child(simplification);
	hb->add_spacer();
	hb->add_child(memnew(Label(TTR("Shrink (Pixels):"))));
	shrink_pixels = memnew(SpinBox);
	shrink_pixels->set_min(0);
	shrink_pixels->set_max(10);
	shrink_pixels->set_step(1);
	shrink_pixels->set_value(0);
	hb->add_child(shrink_pixels);
	hb->add_spacer();
	hb->add_child(memnew(Label(TTR("Grow (Pixels):"))));
	grow_pixels = memnew(SpinBox);
	grow_pixels->set_min(0);
	grow_pixels->set_max(10);
	grow_pixels->set_step(1);
	grow_pixels->set_value(2);
	hb->add_child(grow_pixels);
	hb->add_spacer();
	update_preview = memnew(Button);
	update_preview->set_text(TTR("Update Preview"));
	update_preview->connect("pressed", callable_mp(this, &Sprite2DEditor::_update_mesh_data));
	hb->add_child(update_preview);
	vb->add_margin_child(TTR("Settings:"), hb);

	add_child(debug_uv_dialog);
}

void Sprite2DEditorPlugin::edit(Object *p_object) {
	sprite_editor->edit(Object::cast_to<Sprite2D>(p_object));
}

bool Sprite2DEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("Sprite2D");
}

void Sprite2DEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		sprite_editor->options->show();
	} else {
		sprite_editor->options->hide();
		sprite_editor->edit(nullptr);
	}
}

Sprite2DEditorPlugin::Sprite2DEditorPlugin() {
	sprite_editor = memnew(Sprite2DEditor);
	EditorNode::get_singleton()->get_main_screen_control()->add_child(sprite_editor);
	make_visible(false);

	//sprite_editor->options->hide();
}

Sprite2DEditorPlugin::~Sprite2DEditorPlugin() {
}
