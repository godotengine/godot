/*************************************************************************/
/*  tile_set_editor_plugin.cpp                                           */
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
#include "tile_set_editor_plugin.h"

#include "scene/2d/physics_body_2d.h"
#include "scene/2d/sprite.h"

void TileSetEditor::edit(const Ref<TileSet> &p_tileset) {

	tileset = p_tileset;
}

void TileSetEditor::_import_node(Node *p_node, Ref<TileSet> p_library) {

	for (int i = 0; i < p_node->get_child_count(); i++) {

		Node *child = p_node->get_child(i);

		if (!Object::cast_to<Sprite>(child)) {
			if (child->get_child_count() > 0) {
				_import_node(child, p_library);
			}

			continue;
		}

		Sprite *mi = Object::cast_to<Sprite>(child);
		Ref<Texture> texture = mi->get_texture();
		Ref<Texture> normal_map = mi->get_normal_map();
		Ref<ShaderMaterial> material = mi->get_material();

		if (texture.is_null())
			continue;

		int id = p_library->find_tile_by_name(mi->get_name());
		if (id < 0) {

			id = p_library->get_last_unused_tile_id();
			p_library->create_tile(id);
			p_library->tile_set_name(id, mi->get_name());
		}

		p_library->tile_set_texture(id, texture);
		p_library->tile_set_normal_map(id, normal_map);
		p_library->tile_set_material(id, material);

		p_library->tile_set_modulate(id, mi->get_modulate());

		Vector2 phys_offset;
		Size2 s;

		if (mi->is_region()) {
			s = mi->get_region_rect().size;
			p_library->tile_set_region(id, mi->get_region_rect());
		} else {
			const int frame = mi->get_frame();
			const int hframes = mi->get_hframes();
			s = texture->get_size() / Size2(hframes, mi->get_vframes());
			p_library->tile_set_region(id, Rect2(Vector2(frame % hframes, frame / hframes) * s, s));
		}

		if (mi->is_centered()) {
			phys_offset += -s / 2;
		}

		Vector<TileSet::ShapeData> collisions;
		Ref<NavigationPolygon> nav_poly;
		Ref<OccluderPolygon2D> occluder;
		bool found_collisions = false;

		for (int j = 0; j < mi->get_child_count(); j++) {

			Node *child2 = mi->get_child(j);

			if (Object::cast_to<NavigationPolygonInstance>(child2))
				nav_poly = Object::cast_to<NavigationPolygonInstance>(child2)->get_navigation_polygon();

			if (Object::cast_to<LightOccluder2D>(child2))
				occluder = Object::cast_to<LightOccluder2D>(child2)->get_occluder_polygon();

			if (!Object::cast_to<StaticBody2D>(child2))
				continue;

			found_collisions = true;

			StaticBody2D *sb = Object::cast_to<StaticBody2D>(child2);

			List<uint32_t> shapes;
			sb->get_shape_owners(&shapes);

			for (List<uint32_t>::Element *E = shapes.front(); E; E = E->next()) {
				if (sb->is_shape_owner_disabled(E->get())) continue;

				Transform2D shape_transform = sb->shape_owner_get_transform(E->get());
				bool one_way = sb->is_shape_owner_one_way_collision_enabled(E->get());

				shape_transform.set_origin(shape_transform.get_origin() - phys_offset);

				for (int k = 0; k < sb->shape_owner_get_shape_count(E->get()); k++) {

					Ref<Shape2D> shape = sb->shape_owner_get_shape(E->get(), k);
					TileSet::ShapeData shape_data;
					shape_data.shape = shape;
					shape_data.shape_transform = shape_transform;
					shape_data.one_way_collision = one_way;
					collisions.push_back(shape_data);
				}
			}
		}

		if (found_collisions) {
			p_library->tile_set_shapes(id, collisions);
		}

		p_library->tile_set_texture_offset(id, mi->get_offset());
		p_library->tile_set_navigation_polygon(id, nav_poly);
		p_library->tile_set_light_occluder(id, occluder);
		p_library->tile_set_occluder_offset(id, -phys_offset);
		p_library->tile_set_navigation_polygon_offset(id, -phys_offset);
	}
}

void TileSetEditor::_import_scene(Node *p_scene, Ref<TileSet> p_library, bool p_merge) {

	if (!p_merge)
		p_library->clear();

	_import_node(p_scene, p_library);
}

void TileSetEditor::_menu_confirm() {

	switch (option) {

		case MENU_OPTION_MERGE_FROM_SCENE:
		case MENU_OPTION_CREATE_FROM_SCENE: {

			EditorNode *en = editor;
			Node *scene = en->get_edited_scene();
			if (!scene)
				break;

			_import_scene(scene, tileset, option == MENU_OPTION_MERGE_FROM_SCENE);

		} break;
	}
}

void TileSetEditor::_name_dialog_confirm(const String &name) {

	switch (option) {

		case MENU_OPTION_REMOVE_ITEM: {

			int id = tileset->find_tile_by_name(name);

			if (id < 0 && name.is_valid_integer())
				id = name.to_int();

			if (tileset->has_tile(id)) {
				tileset->remove_tile(id);
			} else {
				err_dialog->set_text(TTR("Could not find tile:") + " " + name);
				err_dialog->popup_centered(Size2(300, 60));
			}
		} break;
	}
}

void TileSetEditor::_menu_cbk(int p_option) {

	option = p_option;
	switch (p_option) {

		case MENU_OPTION_ADD_ITEM: {

			tileset->create_tile(tileset->get_last_unused_tile_id());
		} break;
		case MENU_OPTION_REMOVE_ITEM: {

			nd->set_title(TTR("Remove Item"));
			nd->set_text(TTR("Item name or ID:"));
			nd->popup_centered(Size2(300, 95));
		} break;
		case MENU_OPTION_CREATE_FROM_SCENE: {

			cd->set_text(TTR("Create from scene?"));
			cd->popup_centered(Size2(300, 60));
		} break;
		case MENU_OPTION_MERGE_FROM_SCENE: {

			cd->set_text(TTR("Merge from scene?"));
			cd->popup_centered(Size2(300, 60));
		} break;
	}
}

Error TileSetEditor::update_library_file(Node *p_base_scene, Ref<TileSet> ml, bool p_merge) {

	_import_scene(p_base_scene, ml, p_merge);
	return OK;
}

void TileSetEditor::_bind_methods() {

	ClassDB::bind_method("_menu_cbk", &TileSetEditor::_menu_cbk);
	ClassDB::bind_method("_menu_confirm", &TileSetEditor::_menu_confirm);
	ClassDB::bind_method("_name_dialog_confirm", &TileSetEditor::_name_dialog_confirm);
}

TileSetEditor::TileSetEditor(EditorNode *p_editor) {

	Panel *panel = memnew(Panel);
	panel->set_anchors_and_margins_preset(Control::PRESET_WIDE);
	add_child(panel);
	MenuButton *options = memnew(MenuButton);
	panel->add_child(options);
	options->set_position(Point2(1, 1));
	options->set_text("Theme");
	options->get_popup()->add_item(TTR("Add Item"), MENU_OPTION_ADD_ITEM);
	options->get_popup()->add_item(TTR("Remove Item"), MENU_OPTION_REMOVE_ITEM);
	options->get_popup()->add_separator();
	options->get_popup()->add_item(TTR("Create from Scene"), MENU_OPTION_CREATE_FROM_SCENE);
	options->get_popup()->add_item(TTR("Merge from Scene"), MENU_OPTION_MERGE_FROM_SCENE);
	options->get_popup()->connect("id_pressed", this, "_menu_cbk");
	editor = p_editor;
	cd = memnew(ConfirmationDialog);
	add_child(cd);
	cd->get_ok()->connect("pressed", this, "_menu_confirm");

	nd = memnew(EditorNameDialog);
	add_child(nd);
	nd->set_hide_on_ok(true);
	nd->get_line_edit()->set_margin(MARGIN_TOP, 28);
	nd->connect("name_confirmed", this, "_name_dialog_confirm");

	err_dialog = memnew(AcceptDialog);
	add_child(err_dialog);
	err_dialog->set_title(TTR("Error"));
}

void TileSetEditorPlugin::edit(Object *p_node) {

	if (Object::cast_to<TileSet>(p_node)) {
		tileset_editor->edit(Object::cast_to<TileSet>(p_node));
		tileset_editor->show();
	} else
		tileset_editor->hide();
}

bool TileSetEditorPlugin::handles(Object *p_node) const {

	return p_node->is_class("TileSet");
}

void TileSetEditorPlugin::make_visible(bool p_visible) {

	if (p_visible)
		tileset_editor->show();
	else
		tileset_editor->hide();
}

TileSetEditorPlugin::TileSetEditorPlugin(EditorNode *p_node) {

	tileset_editor = memnew(TileSetEditor(p_node));

	p_node->get_viewport()->add_child(tileset_editor);
	tileset_editor->set_anchors_and_margins_preset(Control::PRESET_WIDE);
	tileset_editor->set_anchor(MARGIN_BOTTOM, Control::ANCHOR_BEGIN);
	tileset_editor->set_end(Point2(0, 22));
	tileset_editor->hide();
}
