/*************************************************************************/
/*  sprite_editor_plugin.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "sprite_editor_plugin.h"

#include "canvas_item_editor_plugin.h"
#include "scene/2d/mesh_instance_2d.h"
#include "scene/gui/box_container.h"
#include "thirdparty/misc/clipper.hpp"

void SpriteEditor::_node_removed(Node *p_node) {

	if (p_node == node) {
		node = NULL;
		options->hide();
	}
}

void SpriteEditor::edit(Sprite *p_sprite) {

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

	ERR_FAIL_COND_V(!p, points);

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
	while (p2->IsHole()) {
		p2 = p2->GetNext();
	}

	int lasti = p2->Contour.size() - 1;
	Vector2 prev = Vector2(p2->Contour[lasti].X / PRECISION, p2->Contour[lasti].Y / PRECISION);
	for (unsigned int i = 0; i < p2->Contour.size(); i++) {

		Vector2 cur = Vector2(p2->Contour[i].X / PRECISION, p2->Contour[i].Y / PRECISION);
		if (cur.distance_to(prev) > 0.5) {
			outPoints.push_back(cur);
			prev = cur;
		}
	}
	return outPoints;
}

void SpriteEditor::_menu_option(int p_option) {

	if (!node) {
		return;
	}

	switch (p_option) {
		case MENU_OPTION_CREATE_MESH_2D: {

			_update_mesh_data();
			debug_uv_dialog->popup_centered();
			debug_uv->update();

		} break;
	}
}

void SpriteEditor::_update_mesh_data() {

	Ref<Texture> texture = node->get_texture();
	if (texture.is_null()) {
		err_dialog->set_text(TTR("Sprite is empty!"));
		err_dialog->popup_centered_minsize();
		return;
	}

	if (node->get_hframes() > 1 || node->get_vframes() > 1) {
		err_dialog->set_text(TTR("Can't convert a sprite using animation frames to mesh."));
		err_dialog->popup_centered_minsize();
		return;
	}
	Ref<Image> image = texture->get_data();
	ERR_FAIL_COND(image.is_null());
	Rect2 rect;
	if (node->is_region())
		rect = node->get_region_rect();
	else
		rect.size = Size2(image->get_width(), image->get_height());

	Ref<BitMap> bm;
	bm.instance();
	bm->create_from_image_alpha(image);

	int grow = island_merging->get_value();
	if (grow > 0) {
		bm->grow_mask(grow, rect);
	}

	float epsilon = simplification->get_value();

	Vector<Vector<Vector2> > lines = bm->clip_opaque_to_polygons(rect, epsilon);

	uv_lines.clear();

	computed_vertices.clear();
	computed_uv.clear();
	computed_indices.clear();

	Size2 img_size = Vector2(image->get_width(), image->get_height());
	for (int j = 0; j < lines.size(); j++) {
		lines.write[j] = expand(lines[j], rect, epsilon);

		int index_ofs = computed_vertices.size();

		for (int i = 0; i < lines[j].size(); i++) {
			Vector2 vtx = lines[j][i];
			computed_uv.push_back(vtx / img_size);

			vtx -= rect.position; //offset by rect position

			//flip if flipped
			if (node->is_flipped_h())
				vtx.x = rect.size.x - vtx.x - 1.0;
			if (node->is_flipped_v())
				vtx.y = rect.size.y - vtx.y - 1.0;

			if (node->is_centered())
				vtx -= rect.size / 2.0;

			computed_vertices.push_back(vtx);
		}

		Vector<int> poly = Geometry::triangulate_polygon(lines[j]);

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

	debug_uv->update();
}

void SpriteEditor::_create_mesh_node() {

	if (computed_vertices.size() < 3) {
		err_dialog->set_text(TTR("Invalid geometry, can't replace by mesh."));
		err_dialog->popup_centered_minsize();
		return;
	}

	Ref<ArrayMesh> mesh;
	mesh.instance();

	Array a;
	a.resize(Mesh::ARRAY_MAX);
	a[Mesh::ARRAY_VERTEX] = computed_vertices;
	a[Mesh::ARRAY_TEX_UV] = computed_uv;
	a[Mesh::ARRAY_INDEX] = computed_indices;

	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, a, Array(), Mesh::ARRAY_FLAG_USE_2D_VERTICES);

	MeshInstance2D *mesh_instance = memnew(MeshInstance2D);
	mesh_instance->set_mesh(mesh);
	EditorNode::get_singleton()->get_scene_tree_dock()->replace_node(node, mesh_instance);
}

#if 0
void SpriteEditor::_create_uv_lines() {

	Ref<Mesh> sprite = node->get_sprite();
	ERR_FAIL_COND(!sprite.is_valid());

	Set<SpriteEditorEdgeSort> edges;
	uv_lines.clear();
	for (int i = 0; i < sprite->get_surface_count(); i++) {
		if (sprite->surface_get_primitive_type(i) != Mesh::PRIMITIVE_TRIANGLES)
			continue;
		Array a = sprite->surface_get_arrays(i);

		PoolVector<Vector2> uv = a[p_layer == 0 ? Mesh::ARRAY_TEX_UV : Mesh::ARRAY_TEX_UV2];
		if (uv.size() == 0) {
			err_dialog->set_text(TTR("Model has no UV in this layer"));
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

				SpriteEditorEdgeSort edge;
				if (use_indices) {
					edge.a = r[ri[j + k]];
					edge.b = r[ri[j + ((k + 1) % 3)]];
				} else {
					edge.a = r[j + k];
					edge.b = r[j + ((k + 1) % 3)];
				}

				if (edges.has(edge))
					continue;

				uv_lines.push_back(edge.a);
				uv_lines.push_back(edge.b);
				edges.insert(edge);
			}
		}
	}

	debug_uv_dialog->popup_centered_minsize();
}
#endif
void SpriteEditor::_debug_uv_draw() {

	if (uv_lines.size() == 0)
		return;

	Ref<Texture> tex = node->get_texture();
	ERR_FAIL_COND(!tex.is_valid());
	debug_uv->set_clip_contents(true);
	debug_uv->draw_texture(tex, Point2());
	debug_uv->set_custom_minimum_size(tex->get_size());
	//debug_uv->draw_set_transform(Vector2(), 0, debug_uv->get_size());
	debug_uv->draw_multiline(uv_lines, Color(1.0, 0.8, 0.7));
}

void SpriteEditor::_bind_methods() {

	ClassDB::bind_method("_menu_option", &SpriteEditor::_menu_option);
	ClassDB::bind_method("_debug_uv_draw", &SpriteEditor::_debug_uv_draw);
	ClassDB::bind_method("_update_mesh_data", &SpriteEditor::_update_mesh_data);
	ClassDB::bind_method("_create_mesh_node", &SpriteEditor::_create_mesh_node);
}

SpriteEditor::SpriteEditor() {

	options = memnew(MenuButton);

	CanvasItemEditor::get_singleton()->add_control_to_menu_panel(options);

	options->set_text(TTR("Sprite"));
	options->set_icon(EditorNode::get_singleton()->get_gui_base()->get_icon("Sprite", "EditorIcons"));

	options->get_popup()->add_item(TTR("Convert to 2D Mesh"), MENU_OPTION_CREATE_MESH_2D);

	options->get_popup()->connect("id_pressed", this, "_menu_option");

	err_dialog = memnew(AcceptDialog);
	add_child(err_dialog);

	debug_uv_dialog = memnew(ConfirmationDialog);
	debug_uv_dialog->get_ok()->set_text(TTR("Create 2D Mesh"));
	debug_uv_dialog->set_title("Mesh 2D Preview");
	VBoxContainer *vb = memnew(VBoxContainer);
	debug_uv_dialog->add_child(vb);
	ScrollContainer *scroll = memnew(ScrollContainer);
	scroll->set_custom_minimum_size(Size2(800, 500) * EDSCALE);
	scroll->set_enable_h_scroll(true);
	scroll->set_enable_v_scroll(true);
	vb->add_margin_child(TTR("Preview:"), scroll, true);
	debug_uv = memnew(Control);
	debug_uv->connect("draw", this, "_debug_uv_draw");
	scroll->add_child(debug_uv);
	debug_uv_dialog->connect("confirmed", this, "_create_mesh_node");

	HBoxContainer *hb = memnew(HBoxContainer);
	hb->add_child(memnew(Label(TTR("Simplification: "))));
	simplification = memnew(SpinBox);
	simplification->set_min(0.01);
	simplification->set_max(10.00);
	simplification->set_step(0.01);
	simplification->set_value(2);
	hb->add_child(simplification);
	hb->add_spacer();
	hb->add_child(memnew(Label(TTR("Grow (Pixels): "))));
	island_merging = memnew(SpinBox);
	island_merging->set_min(0);
	island_merging->set_max(10);
	island_merging->set_step(1);
	island_merging->set_value(2);
	hb->add_child(island_merging);
	hb->add_spacer();
	update_preview = memnew(Button);
	update_preview->set_text(TTR("Update Preview"));
	update_preview->connect("pressed", this, "_update_mesh_data");
	hb->add_child(update_preview);
	vb->add_margin_child(TTR("Settings:"), hb);

	add_child(debug_uv_dialog);
}

void SpriteEditorPlugin::edit(Object *p_object) {

	sprite_editor->edit(Object::cast_to<Sprite>(p_object));
}

bool SpriteEditorPlugin::handles(Object *p_object) const {

	return p_object->is_class("Sprite");
}

void SpriteEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		sprite_editor->options->show();
	} else {

		sprite_editor->options->hide();
		sprite_editor->edit(NULL);
	}
}

SpriteEditorPlugin::SpriteEditorPlugin(EditorNode *p_node) {

	editor = p_node;
	sprite_editor = memnew(SpriteEditor);
	editor->get_viewport()->add_child(sprite_editor);
	make_visible(false);

	//sprite_editor->options->hide();
}

SpriteEditorPlugin::~SpriteEditorPlugin() {
}
