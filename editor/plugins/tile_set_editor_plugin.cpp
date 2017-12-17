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

#include "editor/plugins/canvas_item_editor_plugin.h"
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
	options->set_text(TTR("Tile Set"));
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
		autotile_editor->edit(p_node);
	} else
		tileset_editor->hide();
}

bool TileSetEditorPlugin::handles(Object *p_node) const {

	return p_node->is_class("TileSet");
}

void TileSetEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		tileset_editor->show();
		autotile_button->show();
		autotile_editor->side_panel->show();
		if (autotile_button->is_pressed()) {
			autotile_editor->show();
		}
	} else {
		tileset_editor->hide();
		autotile_editor->side_panel->hide();
		autotile_editor->hide();
		autotile_button->hide();
	}
}

TileSetEditorPlugin::TileSetEditorPlugin(EditorNode *p_node) {

	tileset_editor = memnew(TileSetEditor(p_node));

	add_control_to_container(CONTAINER_CANVAS_EDITOR_MENU, tileset_editor);
	tileset_editor->set_anchors_and_margins_preset(Control::PRESET_WIDE);
	tileset_editor->set_anchor(MARGIN_BOTTOM, Control::ANCHOR_BEGIN);
	tileset_editor->set_end(Point2(0, 22));
	tileset_editor->hide();

	autotile_editor = memnew(AutotileEditor(p_node));
	add_control_to_container(CONTAINER_CANVAS_EDITOR_SIDE, autotile_editor->side_panel);
	autotile_editor->side_panel->set_anchors_and_margins_preset(Control::PRESET_WIDE);
	autotile_editor->side_panel->set_custom_minimum_size(Size2(200, 0));
	autotile_editor->side_panel->hide();
	autotile_button = p_node->add_bottom_panel_item("Autotiles", autotile_editor);
	autotile_button->hide();
}

AutotileEditor::AutotileEditor(EditorNode *p_editor) {

	editor = p_editor;

	//Side Panel
	side_panel = memnew(Control);
	side_panel->set_name("Autotiles");

	VSplitContainer *split = memnew(VSplitContainer);
	side_panel->add_child(split);
	split->set_anchors_and_margins_preset(Control::PRESET_WIDE);

	autotile_list = memnew(ItemList);
	autotile_list->set_v_size_flags(SIZE_EXPAND_FILL);
	autotile_list->set_h_size_flags(SIZE_EXPAND_FILL);
	autotile_list->set_custom_minimum_size(Size2(02, 200));
	autotile_list->connect("item_selected", this, "_on_autotile_selected");
	split->add_child(autotile_list);

	property_editor = memnew(PropertyEditor);
	property_editor->set_v_size_flags(SIZE_EXPAND_FILL);
	property_editor->set_h_size_flags(SIZE_EXPAND_FILL);
	split->add_child(property_editor);

	helper = memnew(AutotileEditorHelper(this));
	property_editor->call_deferred("edit", helper);

	// Editor

	dragging_point = -1;
	creating_shape = false;

	set_custom_minimum_size(Size2(0, 150));

	VBoxContainer *main_vb = memnew(VBoxContainer);
	add_child(main_vb);
	main_vb->set_anchors_and_margins_preset(Control::PRESET_WIDE);

	HBoxContainer *tool_hb = memnew(HBoxContainer);
	Ref<ButtonGroup> g(memnew(ButtonGroup));

	String label[EDITMODE_MAX] = { "Icon", "Bitmask", "Collision", "Occlusion", "Navigation", "Priority" };

	for (int i = 0; i < (int)EDITMODE_MAX; i++) {
		tool_editmode[i] = memnew(Button);
		tool_editmode[i]->set_text(label[i]);
		tool_editmode[i]->set_toggle_mode(true);
		tool_editmode[i]->set_button_group(g);
		Vector<Variant> args;
		args.push_back(i);
		tool_editmode[i]->connect("pressed", this, "_on_edit_mode_changed", args);
		tool_hb->add_child(tool_editmode[i]);
	}
	tool_editmode[EDITMODE_ICON]->set_pressed(true);

	main_vb->add_child(tool_hb);
	main_vb->add_child(memnew(HSeparator));

	toolbar = memnew(HBoxContainer);
	for (int i = 0; i < (int)TOOLBAR_MAX; i++) {
		tool_containers[i] = memnew(HBoxContainer);
		toolbar->add_child(tool_containers[i]);
		tool_containers[i]->hide();
	}

	Ref<ButtonGroup> tg(memnew(ButtonGroup));

	tools[TOOL_SELECT] = memnew(ToolButton);
	tool_containers[TOOLBAR_DUMMY]->add_child(tools[TOOL_SELECT]);
	tools[TOOL_SELECT]->set_tooltip("Select sub-tile to use as icon, this will be also used on invalid autotile bindings.");
	tools[TOOL_SELECT]->set_toggle_mode(true);
	tools[TOOL_SELECT]->set_button_group(tg);
	tools[TOOL_SELECT]->set_pressed(true);
	tool_containers[TOOLBAR_DUMMY]->show();

	Vector<Variant> p;
	tools[BITMASK_COPY] = memnew(ToolButton);
	p.push_back((int)BITMASK_COPY);
	tools[BITMASK_COPY]->connect("pressed", this, "_on_tool_clicked", p);
	tool_containers[TOOLBAR_BITMASK]->add_child(tools[BITMASK_COPY]);
	tools[BITMASK_PASTE] = memnew(ToolButton);
	p = Vector<Variant>();
	p.push_back((int)BITMASK_PASTE);
	tools[BITMASK_PASTE]->connect("pressed", this, "_on_tool_clicked", p);
	tool_containers[TOOLBAR_BITMASK]->add_child(tools[BITMASK_PASTE]);
	tools[BITMASK_CLEAR] = memnew(ToolButton);
	p = Vector<Variant>();
	p.push_back((int)BITMASK_CLEAR);
	tools[BITMASK_CLEAR]->connect("pressed", this, "_on_tool_clicked", p);
	tool_containers[TOOLBAR_BITMASK]->add_child(tools[BITMASK_CLEAR]);

	tools[SHAPE_NEW_POLYGON] = memnew(ToolButton);
	tool_containers[TOOLBAR_SHAPE]->add_child(tools[SHAPE_NEW_POLYGON]);
	tools[SHAPE_NEW_POLYGON]->set_toggle_mode(true);
	tools[SHAPE_NEW_POLYGON]->set_button_group(tg);
	tool_containers[TOOLBAR_SHAPE]->add_child(memnew(VSeparator));
	tools[SHAPE_DELETE] = memnew(ToolButton);
	p = Vector<Variant>();
	p.push_back((int)SHAPE_DELETE);
	tools[SHAPE_DELETE]->connect("pressed", this, "_on_tool_clicked", p);
	tool_containers[TOOLBAR_SHAPE]->add_child(tools[SHAPE_DELETE]);
	//tools[SHAPE_CREATE_FROM_NOT_BITMASKED] = memnew(ToolButton);
	//tool_containers[TOOLBAR_SHAPE]->add_child(tools[SHAPE_CREATE_FROM_NOT_BITMASKED]);
	tool_containers[TOOLBAR_SHAPE]->add_change_receptor(memnew(VSeparator));
	tools[SHAPE_KEEP_INSIDE_TILE] = memnew(ToolButton);
	tools[SHAPE_KEEP_INSIDE_TILE]->set_toggle_mode(true);
	tools[SHAPE_KEEP_INSIDE_TILE]->set_pressed(true);
	tool_containers[TOOLBAR_SHAPE]->add_child(tools[SHAPE_KEEP_INSIDE_TILE]);
	tools[SHAPE_SNAP_TO_BITMASK_GRID] = memnew(ToolButton);
	tools[SHAPE_SNAP_TO_BITMASK_GRID]->set_toggle_mode(true);
	tools[SHAPE_SNAP_TO_BITMASK_GRID]->set_pressed(true);
	tool_containers[TOOLBAR_SHAPE]->add_child(tools[SHAPE_SNAP_TO_BITMASK_GRID]);

	spin_priority = memnew(SpinBox);
	spin_priority->set_min(1);
	spin_priority->set_max(255);
	spin_priority->set_step(1);
	spin_priority->set_custom_minimum_size(Size2(100, 0));
	spin_priority->connect("value_changed", this, "_on_priority_changed");
	spin_priority->hide();
	toolbar->add_child(spin_priority);

	Control *separator = memnew(Control);
	separator->set_h_size_flags(SIZE_EXPAND_FILL);
	toolbar->add_child(separator);

	tools[ZOOM_OUT] = memnew(ToolButton);
	p = Vector<Variant>();
	p.push_back((int)ZOOM_OUT);
	tools[ZOOM_OUT]->connect("pressed", this, "_on_tool_clicked", p);
	toolbar->add_child(tools[ZOOM_OUT]);
	tools[ZOOM_1] = memnew(ToolButton);
	p = Vector<Variant>();
	p.push_back((int)ZOOM_1);
	tools[ZOOM_1]->connect("pressed", this, "_on_tool_clicked", p);
	toolbar->add_child(tools[ZOOM_1]);
	tools[ZOOM_IN] = memnew(ToolButton);
	p = Vector<Variant>();
	p.push_back((int)ZOOM_IN);
	tools[ZOOM_IN]->connect("pressed", this, "_on_tool_clicked", p);
	toolbar->add_child(tools[ZOOM_IN]);

	main_vb->add_child(toolbar);

	ScrollContainer *scroll = memnew(ScrollContainer);
	main_vb->add_child(scroll);
	scroll->set_v_size_flags(SIZE_EXPAND_FILL);

	workspace_container = memnew(Control);
	scroll->add_child(workspace_container);

	workspace = memnew(Control);
	workspace->connect("draw", this, "_on_workspace_draw");
	workspace->connect("gui_input", this, "_on_workspace_input");
	workspace_container->add_child(workspace);

	preview = memnew(Sprite);
	workspace->add_child(preview);
	preview->set_centered(false);
	preview->set_draw_behind_parent(true);
	preview->set_region(true);
}

void AutotileEditor::_bind_methods() {

	ClassDB::bind_method("_on_autotile_selected", &AutotileEditor::_on_autotile_selected);
	ClassDB::bind_method("_on_edit_mode_changed", &AutotileEditor::_on_edit_mode_changed);
	ClassDB::bind_method("_on_workspace_draw", &AutotileEditor::_on_workspace_draw);
	ClassDB::bind_method("_on_workspace_input", &AutotileEditor::_on_workspace_input);
	ClassDB::bind_method("_on_tool_clicked", &AutotileEditor::_on_tool_clicked);
	ClassDB::bind_method("_on_priority_changed", &AutotileEditor::_on_priority_changed);
}

void AutotileEditor::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {
		tools[TOOL_SELECT]->set_icon(get_icon("ToolSelect", "EditorIcons"));
		tools[BITMASK_COPY]->set_icon(get_icon("Duplicate", "EditorIcons"));
		tools[BITMASK_PASTE]->set_icon(get_icon("Override", "EditorIcons"));
		tools[BITMASK_CLEAR]->set_icon(get_icon("Clear", "EditorIcons"));
		tools[SHAPE_NEW_POLYGON]->set_icon(get_icon("CollisionPolygon2D", "EditorIcons"));
		tools[SHAPE_DELETE]->set_icon(get_icon("Remove", "EditorIcons"));
		tools[SHAPE_KEEP_INSIDE_TILE]->set_icon(get_icon("Snap", "EditorIcons"));
		tools[SHAPE_SNAP_TO_BITMASK_GRID]->set_icon(get_icon("SnapGrid", "EditorIcons"));
		tools[ZOOM_OUT]->set_icon(get_icon("ZoomLess", "EditorIcons"));
		tools[ZOOM_1]->set_icon(get_icon("ZoomReset", "EditorIcons"));
		tools[ZOOM_IN]->set_icon(get_icon("ZoomMore", "EditorIcons"));
	}
}

void AutotileEditor::_on_autotile_selected(int p_index) {

	if (get_current_tile() >= 0) {
		current_item_index = p_index;
		preview->set_texture(tile_set->tile_get_texture(get_current_tile()));
		preview->set_region_rect(tile_set->tile_get_region(get_current_tile()));
		workspace->set_custom_minimum_size(tile_set->tile_get_region(get_current_tile()).size);
	} else {
		current_item_index = -1;
		preview->set_texture(NULL);
		workspace->set_custom_minimum_size(Size2i());
	}
	helper->_change_notify("");
	workspace->update();
}

void AutotileEditor::_on_edit_mode_changed(int p_edit_mode) {

	edit_mode = (EditMode)p_edit_mode;
	switch (edit_mode) {
		case EDITMODE_BITMASK: {
			tool_containers[TOOLBAR_DUMMY]->show();
			tool_containers[TOOLBAR_BITMASK]->show();
			tool_containers[TOOLBAR_SHAPE]->hide();
			tools[TOOL_SELECT]->set_pressed(true);
			tools[TOOL_SELECT]->set_tooltip("LMB: set bit on.\nRMB: set bit off.");
			spin_priority->hide();
		} break;
		case EDITMODE_COLLISION:
		case EDITMODE_NAVIGATION:
		case EDITMODE_OCCLUSION: {
			tool_containers[TOOLBAR_DUMMY]->show();
			tool_containers[TOOLBAR_BITMASK]->hide();
			tool_containers[TOOLBAR_SHAPE]->show();
			tools[TOOL_SELECT]->set_tooltip("Select current edited sub-tile.");
			spin_priority->hide();
		} break;
		default: {
			tool_containers[TOOLBAR_DUMMY]->show();
			tool_containers[TOOLBAR_BITMASK]->hide();
			tool_containers[TOOLBAR_SHAPE]->hide();
			if (edit_mode == EDITMODE_ICON) {
				tools[TOOL_SELECT]->set_tooltip("Select sub-tile to use as icon, this will be also used on invalid autotile bindings.");
				spin_priority->hide();
			} else {
				tools[TOOL_SELECT]->set_tooltip("Select sub-tile to change it's priority.");
				spin_priority->show();
			}
		} break;
	}
	workspace->update();
}

void AutotileEditor::_on_workspace_draw() {

	if (get_current_tile() >= 0 && !tile_set.is_null()) {
		int spacing = tile_set->autotile_get_spacing(get_current_tile());
		Vector2 size = tile_set->autotile_get_size(get_current_tile());
		Rect2i region = tile_set->tile_get_region(get_current_tile());
		Color c(0.347214, 0.722656, 0.617063);

		switch (edit_mode) {
			case EDITMODE_ICON: {
				Vector2 coord = tile_set->autotile_get_icon_coordinate(get_current_tile());
				draw_highlight_tile(coord);
			} break;
			case EDITMODE_BITMASK: {
				c = Color(1, 0, 0, 0.5);
				for (float x = 0; x < region.size.x / (spacing + size.x); x++) {
					for (float y = 0; y < region.size.y / (spacing + size.y); y++) {
						Vector2 coord(x, y);
						Point2 anchor(coord.x * (spacing + size.x), coord.y * (spacing + size.y));
						uint16_t mask = tile_set->autotile_get_bitmask(get_current_tile(), coord);
						if (tile_set->autotile_get_bitmask_mode(get_current_tile()) == TileSet::BITMASK_2X2) {
							if (mask & TileSet::BIND_TOPLEFT) {
								workspace->draw_rect(Rect2(anchor, size / 2), c);
							}
							if (mask & TileSet::BIND_TOPRIGHT) {
								workspace->draw_rect(Rect2(anchor + Vector2(size.x / 2, 0), size / 2), c);
							}
							if (mask & TileSet::BIND_BOTTOMLEFT) {
								workspace->draw_rect(Rect2(anchor + Vector2(0, size.y / 2), size / 2), c);
							}
							if (mask & TileSet::BIND_BOTTOMRIGHT) {
								workspace->draw_rect(Rect2(anchor + size / 2, size / 2), c);
							}
						} else if (tile_set->autotile_get_bitmask_mode(get_current_tile()) == TileSet::BITMASK_3X3) {
							if (mask & TileSet::BIND_TOPLEFT) {
								workspace->draw_rect(Rect2(anchor, size / 3), c);
							}
							if (mask & TileSet::BIND_TOP) {
								workspace->draw_rect(Rect2(anchor + Vector2(size.x / 3, 0), size / 3), c);
							}
							if (mask & TileSet::BIND_TOPRIGHT) {
								workspace->draw_rect(Rect2(anchor + Vector2((size.x / 3) * 2, 0), size / 3), c);
							}
							if (mask & TileSet::BIND_LEFT) {
								workspace->draw_rect(Rect2(anchor + Vector2(0, size.y / 3), size / 3), c);
							}
							if (mask & TileSet::BIND_CENTER) {
								workspace->draw_rect(Rect2(anchor + Vector2(size.x / 3, size.y / 3), size / 3), c);
							}
							if (mask & TileSet::BIND_RIGHT) {
								workspace->draw_rect(Rect2(anchor + Vector2((size.x / 3) * 2, size.y / 3), size / 3), c);
							}
							if (mask & TileSet::BIND_BOTTOMLEFT) {
								workspace->draw_rect(Rect2(anchor + Vector2(0, (size.y / 3) * 2), size / 3), c);
							}
							if (mask & TileSet::BIND_BOTTOM) {
								workspace->draw_rect(Rect2(anchor + Vector2(size.x / 3, (size.y / 3) * 2), size / 3), c);
							}
							if (mask & TileSet::BIND_BOTTOMRIGHT) {
								workspace->draw_rect(Rect2(anchor + (size / 3) * 2, size / 3), c);
							}
						}
					}
				}
			} break;
			case EDITMODE_COLLISION:
			case EDITMODE_OCCLUSION:
			case EDITMODE_NAVIGATION: {
				Vector2 coord = edited_shape_coord;
				draw_highlight_tile(coord);
				draw_polygon_shapes();
			} break;
			case EDITMODE_PRIORITY: {
				spin_priority->set_value(tile_set->autotile_get_subtile_priority(get_current_tile(), edited_shape_coord));
				uint16_t mask = tile_set->autotile_get_bitmask(get_current_tile(), edited_shape_coord);
				Vector<Vector2> queue_others;
				int total = 0;
				for (Map<Vector2, uint16_t>::Element *E = tile_set->autotile_get_bitmask_map(get_current_tile()).front(); E; E = E->next()) {
					if (E->value() == mask) {
						total += tile_set->autotile_get_subtile_priority(get_current_tile(), E->key());
						if (E->key() != edited_shape_coord) {
							queue_others.push_back(E->key());
						}
					}
				}
				spin_priority->set_suffix(" / " + String::num(total, 0));
				draw_highlight_tile(edited_shape_coord, queue_others);
			} break;
		}

		float j = -size.x; //make sure to draw at 0
		while (j < region.size.x) {
			j += size.x;
			if (spacing <= 0) {
				workspace->draw_line(Point2(j, 0), Point2(j, region.size.y), c);
			} else {
				workspace->draw_rect(Rect2(Point2(j, 0), Size2(spacing, region.size.y)), c);
			}
			j += spacing;
		}
		j = -size.y; //make sure to draw at 0
		while (j < region.size.y) {
			j += size.y;
			if (spacing <= 0) {
				workspace->draw_line(Point2(0, j), Point2(region.size.x, j), c);
			} else {
				workspace->draw_rect(Rect2(Point2(0, j), Size2(region.size.x, spacing)), c);
			}
			j += spacing;
		}
	}
}

#define MIN_DISTANCE_SQUARED 10
void AutotileEditor::_on_workspace_input(const Ref<InputEvent> &p_ie) {

	if (get_current_tile() >= 0 && !tile_set.is_null()) {
		Ref<InputEventMouseButton> mb = p_ie;
		Ref<InputEventMouseMotion> mm = p_ie;

		static bool dragging;
		static bool erasing;

		int spacing = tile_set->autotile_get_spacing(get_current_tile());
		Vector2 size = tile_set->autotile_get_size(get_current_tile());
		switch (edit_mode) {
			case EDITMODE_ICON: {
				if (mb.is_valid()) {
					if (mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {
						Vector2 coord((int)(mb->get_position().x / (spacing + size.x)), (int)(mb->get_position().y / (spacing + size.y)));
						tile_set->autotile_set_icon_coordinate(get_current_tile(), coord);
						Rect2 region = tile_set->tile_get_region(get_current_tile());
						region.size = size;
						coord.x *= (spacing + size.x);
						coord.y *= (spacing + size.y);
						region.position += coord;
						autotile_list->set_item_icon_region(current_item_index, region);
						workspace->update();
					}
				}
			} break;
			case EDITMODE_BITMASK: {
				if (mb.is_valid()) {
					if (mb->is_pressed()) {
						if (dragging) {
							return;
						}
						if (mb->get_button_index() == BUTTON_RIGHT || mb->get_button_index() == BUTTON_LEFT) {
							dragging = true;
							erasing = (mb->get_button_index() == BUTTON_RIGHT);
							Vector2 coord((int)(mb->get_position().x / (spacing + size.x)), (int)(mb->get_position().y / (spacing + size.y)));
							Vector2 pos(coord.x * (spacing + size.x), coord.y * (spacing + size.y));
							pos = mb->get_position() - pos;
							uint16_t bit;
							if (tile_set->autotile_get_bitmask_mode(get_current_tile()) == TileSet::BITMASK_2X2) {
								if (pos.x < size.x / 2) {
									if (pos.y < size.y / 2) {
										bit = TileSet::BIND_TOPLEFT;
									} else {
										bit = TileSet::BIND_BOTTOMLEFT;
									}
								} else {
									if (pos.y < size.y / 2) {
										bit = TileSet::BIND_TOPRIGHT;
									} else {
										bit = TileSet::BIND_BOTTOMRIGHT;
									}
								}
							} else if (tile_set->autotile_get_bitmask_mode(get_current_tile()) == TileSet::BITMASK_3X3) {
								if (pos.x < size.x / 3) {
									if (pos.y < size.y / 3) {
										bit = TileSet::BIND_TOPLEFT;
									} else if (pos.y > (size.y / 3) * 2) {
										bit = TileSet::BIND_BOTTOMLEFT;
									} else {
										bit = TileSet::BIND_LEFT;
									}
								} else if (pos.x > (size.x / 3) * 2) {
									if (pos.y < size.y / 3) {
										bit = TileSet::BIND_TOPRIGHT;
									} else if (pos.y > (size.y / 3) * 2) {
										bit = TileSet::BIND_BOTTOMRIGHT;
									} else {
										bit = TileSet::BIND_RIGHT;
									}
								} else {
									if (pos.y < size.y / 3) {
										bit = TileSet::BIND_TOP;
									} else if (pos.y > (size.y / 3) * 2) {
										bit = TileSet::BIND_BOTTOM;
									} else {
										bit = TileSet::BIND_CENTER;
									}
								}
							}
							uint16_t mask = tile_set->autotile_get_bitmask(get_current_tile(), coord);
							if (erasing) {
								mask &= ~bit;
							} else {
								mask |= bit;
							}
							tile_set->autotile_set_bitmask(get_current_tile(), coord, mask);
							workspace->update();
						}
					} else {
						if ((erasing && mb->get_button_index() == BUTTON_RIGHT) || (!erasing && mb->get_button_index() == BUTTON_LEFT)) {
							dragging = false;
							erasing = false;
						}
					}
				}
				if (mm.is_valid()) {
					if (dragging) {
						Vector2 coord((int)(mm->get_position().x / (spacing + size.x)), (int)(mm->get_position().y / (spacing + size.y)));
						Vector2 pos(coord.x * (spacing + size.x), coord.y * (spacing + size.y));
						pos = mm->get_position() - pos;
						uint16_t bit;
						if (tile_set->autotile_get_bitmask_mode(get_current_tile()) == TileSet::BITMASK_2X2) {
							if (pos.x < size.x / 2) {
								if (pos.y < size.y / 2) {
									bit = TileSet::BIND_TOPLEFT;
								} else {
									bit = TileSet::BIND_BOTTOMLEFT;
								}
							} else {
								if (pos.y < size.y / 2) {
									bit = TileSet::BIND_TOPRIGHT;
								} else {
									bit = TileSet::BIND_BOTTOMRIGHT;
								}
							}
						} else if (tile_set->autotile_get_bitmask_mode(get_current_tile()) == TileSet::BITMASK_3X3) {
							if (pos.x < size.x / 3) {
								if (pos.y < size.y / 3) {
									bit = TileSet::BIND_TOPLEFT;
								} else if (pos.y > (size.y / 3) * 2) {
									bit = TileSet::BIND_BOTTOMLEFT;
								} else {
									bit = TileSet::BIND_LEFT;
								}
							} else if (pos.x > (size.x / 3) * 2) {
								if (pos.y < size.y / 3) {
									bit = TileSet::BIND_TOPRIGHT;
								} else if (pos.y > (size.y / 3) * 2) {
									bit = TileSet::BIND_BOTTOMRIGHT;
								} else {
									bit = TileSet::BIND_RIGHT;
								}
							} else {
								if (pos.y < size.y / 3) {
									bit = TileSet::BIND_TOP;
								} else if (pos.y > (size.y / 3) * 2) {
									bit = TileSet::BIND_BOTTOM;
								} else {
									bit = TileSet::BIND_CENTER;
								}
							}
						}
						uint16_t mask = tile_set->autotile_get_bitmask(get_current_tile(), coord);
						if (erasing) {
							mask &= ~bit;
						} else {
							mask |= bit;
						}
						tile_set->autotile_set_bitmask(get_current_tile(), coord, mask);
						workspace->update();
					}
				}
			} break;
			case EDITMODE_COLLISION:
			case EDITMODE_OCCLUSION:
			case EDITMODE_NAVIGATION:
			case EDITMODE_PRIORITY: {
				Vector2 shape_anchor = edited_shape_coord;
				shape_anchor.x *= (size.x + spacing);
				shape_anchor.y *= (size.y + spacing);
				if (tools[TOOL_SELECT]->is_pressed()) {
					if (mb.is_valid()) {
						if (mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {
							if (edit_mode != EDITMODE_PRIORITY && current_shape.size() > 0) {
								for (int i = 0; i < current_shape.size(); i++) {
									if ((current_shape[i] - mb->get_position()).length_squared() <= MIN_DISTANCE_SQUARED) {
										dragging_point = i;
										workspace->update();
										return;
									}
								}
							}
							Vector2 coord((int)(mb->get_position().x / (spacing + size.x)), (int)(mb->get_position().y / (spacing + size.y)));
							if (edited_shape_coord != coord) {
								edited_shape_coord = coord;
								edited_occlusion_shape = tile_set->autotile_get_light_occluder(get_current_tile(), edited_shape_coord);
								edited_navigation_shape = tile_set->autotile_get_navigation_polygon(get_current_tile(), edited_shape_coord);
								shape_anchor = edited_shape_coord;
								shape_anchor.x *= (size.x + spacing);
								shape_anchor.y *= (size.y + spacing);
								if (edit_mode == EDITMODE_OCCLUSION) {
									current_shape.resize(0);
									if (edited_occlusion_shape.is_valid()) {
										for (int i = 0; i < edited_occlusion_shape->get_polygon().size(); i++) {
											current_shape.push_back(edited_occlusion_shape->get_polygon()[i] + shape_anchor);
										}
									}
								} else if (edit_mode == EDITMODE_NAVIGATION) {
									current_shape.resize(0);
									if (edited_navigation_shape.is_valid()) {
										if (edited_navigation_shape->get_polygon_count() > 0) {
											PoolVector<Vector2> vertices = edited_navigation_shape->get_vertices();
											for (int i = 0; i < edited_navigation_shape->get_polygon(0).size(); i++) {
												current_shape.push_back(vertices[edited_navigation_shape->get_polygon(0)[i]] + shape_anchor);
											}
										}
									}
								}
							} else {
								if (edit_mode == EDITMODE_COLLISION) {
									Vector<TileSet::ShapeData> sd = tile_set->tile_get_shapes(get_current_tile());
									for (int i = 0; i < sd.size(); i++) {
										if (sd[i].autotile_coord == coord) {
											Ref<ConcavePolygonShape2D> shape = sd[i].shape;
											if (shape.is_valid()) {
												//FIXME: i need a way to know if the point is countained on the polygon instead of the rect
												Rect2 bounding_rect;
												PoolVector2Array polygon;
												bounding_rect.position = shape->get_segments()[0];
												for (int j = 0; j < shape->get_segments().size(); j += 2) {
													polygon.push_back(shape->get_segments()[j] + shape_anchor);
													bounding_rect.expand_to(shape->get_segments()[j] + shape_anchor);
												}
												if (bounding_rect.has_point(mb->get_position())) {
													current_shape = polygon;
													edited_collision_shape = shape;
												}
											}
										}
									}
								}
							}
							workspace->update();
						} else if (!mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {
							if (edit_mode == EDITMODE_COLLISION) {
								if (dragging_point >= 0) {
									dragging_point = -1;

									PoolVector<Vector2> segments;
									segments.resize(current_shape.size() * 2);
									PoolVector<Vector2>::Write w = segments.write();

									for (int i = 0; i < current_shape.size(); i++) {
										w[(i << 1) + 0] = current_shape[i] - shape_anchor;
										w[(i << 1) + 1] = current_shape[(i + 1) % current_shape.size()] - shape_anchor;
									}

									w = PoolVector<Vector2>::Write();
									edited_collision_shape->set_segments(segments);

									workspace->update();
								}
							} else if (edit_mode == EDITMODE_OCCLUSION) {
								if (dragging_point >= 0) {
									dragging_point = -1;

									PoolVector<Vector2> polygon;
									polygon.resize(current_shape.size());
									PoolVector<Vector2>::Write w = polygon.write();

									for (int i = 0; i < current_shape.size(); i++) {
										w[i] = current_shape[i] - shape_anchor;
									}

									w = PoolVector<Vector2>::Write();
									edited_occlusion_shape->set_polygon(polygon);

									workspace->update();
								}
							} else if (edit_mode == EDITMODE_NAVIGATION) {
								if (dragging_point >= 0) {
									dragging_point = -1;

									PoolVector<Vector2> polygon;
									Vector<int> indices;
									polygon.resize(current_shape.size());
									PoolVector<Vector2>::Write w = polygon.write();

									for (int i = 0; i < current_shape.size(); i++) {
										w[i] = current_shape[i] - shape_anchor;
										indices.push_back(i);
									}

									w = PoolVector<Vector2>::Write();
									edited_navigation_shape->set_vertices(polygon);
									edited_navigation_shape->add_polygon(indices);

									workspace->update();
								}
							}
						}
					} else if (mm.is_valid()) {
						if (dragging_point >= 0) {
							current_shape.set(dragging_point, snap_point(mm->get_position()));
							workspace->update();
						}
					}
				} else if (tools[SHAPE_NEW_POLYGON]->is_pressed()) {
					if (mb.is_valid()) {
						if (mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {
							Vector2 pos = mb->get_position();
							pos = snap_point(pos);
							if (creating_shape) {
								if (current_shape.size() > 0) {
									if ((pos - current_shape[0]).length_squared() <= MIN_DISTANCE_SQUARED) {
										if (current_shape.size() > 2) {
											close_shape(shape_anchor);
											workspace->update();
											return;
										}
									}
								}
								current_shape.push_back(pos);
								workspace->update();
							} else {
								creating_shape = true;
								current_shape.resize(0);
								current_shape.push_back(snap_point(pos));
							}
						} else if (mb->is_pressed() && mb->get_button_index() == BUTTON_RIGHT) {
							if (creating_shape) {
								close_shape(shape_anchor);
							}
						}
					} else if (mm.is_valid()) {
						if (creating_shape) {
							workspace->update();
						}
					}
				}
			} break;
		}
	}
}

void AutotileEditor::_on_tool_clicked(int p_tool) {
	if (p_tool == BITMASK_COPY) {
		bitmask_map_copy = tile_set->autotile_get_bitmask_map(get_current_tile());
	} else if (p_tool == BITMASK_PASTE) {
		tile_set->autotile_clear_bitmask_map(get_current_tile());
		for (Map<Vector2, uint16_t>::Element *E = bitmask_map_copy.front(); E; E = E->next()) {
			tile_set->autotile_set_bitmask(get_current_tile(), E->key(), E->value());
		}
		workspace->update();
	} else if (p_tool == BITMASK_CLEAR) {
		tile_set->autotile_clear_bitmask_map(get_current_tile());
		workspace->update();
	} else if (p_tool == SHAPE_DELETE) {
		if (creating_shape) {
			creating_shape = false;
			current_shape.resize(0);
			workspace->update();
		} else {
			switch (edit_mode) {
				case EDITMODE_COLLISION: {
					if (!edited_collision_shape.is_null()) {
						Vector<TileSet::ShapeData> sd = tile_set->tile_get_shapes(get_current_tile());
						int index;
						for (int i = 0; i < sd.size(); i++) {
							if (sd[i].shape == edited_collision_shape) {
								index = i;
								break;
							}
						}
						if (index >= 0) {
							sd.remove(index);
							tile_set->tile_set_shapes(get_current_tile(), sd);
							edited_collision_shape = Ref<ConcavePolygonShape2D>();
							current_shape.resize(0);
							workspace->update();
						}
					}
				} break;
				case EDITMODE_NAVIGATION: {
					if (!edited_navigation_shape.is_null()) {
						tile_set->autotile_set_navigation_polygon(get_current_tile(), Ref<NavigationPolygon>(), edited_shape_coord);
						edited_navigation_shape = Ref<NavigationPolygon>();
						current_shape.resize(0);
						workspace->update();
					}
				} break;
				case EDITMODE_OCCLUSION: {
					if (!edited_occlusion_shape.is_null()) {
						tile_set->autotile_set_light_occluder(get_current_tile(), Ref<OccluderPolygon2D>(), edited_shape_coord);
						edited_occlusion_shape = Ref<OccluderPolygon2D>();
						current_shape.resize(0);
						workspace->update();
					}
				} break;
			}
		}
	} else if (p_tool == ZOOM_OUT) {
		float scale = workspace->get_scale().x;
		if (scale > 0.1) {
			scale /= 2;
			workspace->set_scale(Vector2(scale, scale));
			workspace_container->set_custom_minimum_size(preview->get_region_rect().size * scale);
		}
	} else if (p_tool == ZOOM_1) {
		workspace->set_scale(Vector2(1, 1));
		workspace_container->set_custom_minimum_size(preview->get_region_rect().size);
	} else if (p_tool == ZOOM_IN) {
		float scale = workspace->get_scale().x;
		scale *= 2;
		workspace->set_scale(Vector2(scale, scale));
		workspace_container->set_custom_minimum_size(preview->get_region_rect().size * scale);
	}
}

void AutotileEditor::_on_priority_changed(float val) {
	tile_set->autotile_set_subtile_priority(get_current_tile(), edited_shape_coord, (int)val);
	workspace->update();
}

void AutotileEditor::draw_highlight_tile(Vector2 coord, const Vector<Vector2> &other_highlighted) {

	Vector2 size = tile_set->autotile_get_size(get_current_tile());
	int spacing = tile_set->autotile_get_spacing(get_current_tile());
	Rect2 region = tile_set->tile_get_region(get_current_tile());
	coord.x *= (size.x + spacing);
	coord.y *= (size.y + spacing);
	workspace->draw_rect(Rect2(0, 0, region.size.x, coord.y), Color(0.5, 0.5, 0.5, 0.5));
	workspace->draw_rect(Rect2(0, coord.y, coord.x, size.y), Color(0.5, 0.5, 0.5, 0.5));
	workspace->draw_rect(Rect2(coord.x + size.x, coord.y, region.size.x - coord.x - size.x, size.y), Color(0.5, 0.5, 0.5, 0.5));
	workspace->draw_rect(Rect2(0, coord.y + size.y, region.size.x, region.size.y - size.y - coord.y), Color(0.5, 0.5, 0.5, 0.5));
	coord += Vector2(1, 1);
	workspace->draw_rect(Rect2(coord, size - Vector2(2, 2)), Color(1, 0, 0), false);
	for (int i = 0; i < other_highlighted.size(); i++) {
		coord = other_highlighted[i];
		coord.x *= (size.x + spacing);
		coord.y *= (size.y + spacing);
		coord += Vector2(1, 1);
		workspace->draw_rect(Rect2(coord, size - Vector2(2, 2)), Color(1, 0, 0), false);
	}
}

void AutotileEditor::draw_polygon_shapes() {

	int t_id = get_current_tile();
	if (t_id < 0)
		return;

	switch (edit_mode) {
		case EDITMODE_COLLISION: {
			Vector<TileSet::ShapeData> sd = tile_set->tile_get_shapes(t_id);
			for (int i = 0; i < sd.size(); i++) {
				Vector2 coord = sd[i].autotile_coord;
				Vector2 anchor = tile_set->autotile_get_size(t_id);
				anchor.x += tile_set->autotile_get_spacing(t_id);
				anchor.y += tile_set->autotile_get_spacing(t_id);
				anchor.x *= coord.x;
				anchor.y *= coord.y;
				Ref<ConcavePolygonShape2D> shape = sd[i].shape;
				if (shape.is_valid()) {
					Color c_bg;
					Color c_border;
					if (coord == edited_shape_coord && sd[i].shape == edited_collision_shape) {
						c_bg = Color(0, 1, 1, 0.5);
						c_border = Color(0, 1, 1);
					} else {
						c_bg = Color(0.9, 0.7, 0.07, 0.5);
						c_border = Color(0.9, 0.7, 0.07, 1);
					}
					Vector<Vector2> polygon;
					Vector<Color> colors;
					if (shape == edited_collision_shape) {
						for (int j = 0; j < current_shape.size(); j++) {
							polygon.push_back(current_shape[j]);
							colors.push_back(c_bg);
						}
					} else {
						for (int j = 0; j < shape->get_segments().size(); j += 2) {
							polygon.push_back(shape->get_segments()[j] + anchor);
							colors.push_back(c_bg);
						}
					}
					workspace->draw_polygon(polygon, colors);
					if (coord == edited_shape_coord) {
						for (int j = 0; j < shape->get_segments().size(); j += 2) {
							workspace->draw_line(shape->get_segments()[j] + anchor, shape->get_segments()[j + 1] + anchor, c_border, 1, true);
						}
						if (shape == edited_collision_shape) {
							for (int j = 0; j < current_shape.size(); j++) {
								workspace->draw_circle(current_shape[j], 5, Color(1, 0, 0));
							}
						}
					}
				}
			}
		} break;
		case EDITMODE_OCCLUSION: {
			Map<Vector2, Ref<OccluderPolygon2D> > map = tile_set->autotile_get_light_oclusion_map(t_id);
			for (Map<Vector2, Ref<OccluderPolygon2D> >::Element *E = map.front(); E; E = E->next()) {
				Vector2 coord = E->key();
				Vector2 anchor = tile_set->autotile_get_size(t_id);
				anchor.x += tile_set->autotile_get_spacing(t_id);
				anchor.y += tile_set->autotile_get_spacing(t_id);
				anchor.x *= coord.x;
				anchor.y *= coord.y;
				Ref<OccluderPolygon2D> shape = E->value();
				if (shape.is_valid()) {
					Color c_bg;
					Color c_border;
					if (coord == edited_shape_coord && shape == edited_occlusion_shape) {
						c_bg = Color(0, 1, 1, 0.5);
						c_border = Color(0, 1, 1);
					} else {
						c_bg = Color(0.9, 0.7, 0.07, 0.5);
						c_border = Color(0.9, 0.7, 0.07, 1);
					}
					Vector<Vector2> polygon;
					Vector<Color> colors;
					if (shape == edited_occlusion_shape) {
						for (int j = 0; j < current_shape.size(); j++) {
							polygon.push_back(current_shape[j]);
							colors.push_back(c_bg);
						}
					} else {
						for (int j = 0; j < shape->get_polygon().size(); j++) {
							polygon.push_back(shape->get_polygon()[j] + anchor);
							colors.push_back(c_bg);
						}
					}
					workspace->draw_polygon(polygon, colors);
					if (coord == edited_shape_coord) {
						for (int j = 0; j < shape->get_polygon().size() - 1; j++) {
							workspace->draw_line(shape->get_polygon()[j] + anchor, shape->get_polygon()[j + 1] + anchor, c_border, 1, true);
						}
						workspace->draw_line(shape->get_polygon()[shape->get_polygon().size() - 1] + anchor, shape->get_polygon()[0] + anchor, c_border, 1, true);
						if (shape == edited_occlusion_shape) {
							for (int j = 0; j < current_shape.size(); j++) {
								workspace->draw_circle(current_shape[j], 5, Color(1, 0, 0));
							}
						}
					}
				}
			}
		} break;
		case EDITMODE_NAVIGATION: {
			Map<Vector2, Ref<NavigationPolygon> > map = tile_set->autotile_get_navigation_map(t_id);
			for (Map<Vector2, Ref<NavigationPolygon> >::Element *E = map.front(); E; E = E->next()) {
				Vector2 coord = E->key();
				Vector2 anchor = tile_set->autotile_get_size(t_id);
				anchor.x += tile_set->autotile_get_spacing(t_id);
				anchor.y += tile_set->autotile_get_spacing(t_id);
				anchor.x *= coord.x;
				anchor.y *= coord.y;
				Ref<NavigationPolygon> shape = E->value();
				if (shape.is_valid()) {
					Color c_bg;
					Color c_border;
					if (coord == edited_shape_coord && shape == edited_navigation_shape) {
						c_bg = Color(0, 1, 1, 0.5);
						c_border = Color(0, 1, 1);
					} else {
						c_bg = Color(0.9, 0.7, 0.07, 0.5);
						c_border = Color(0.9, 0.7, 0.07, 1);
					}
					Vector<Vector2> polygon;
					Vector<Color> colors;
					if (shape == edited_navigation_shape) {
						for (int j = 0; j < current_shape.size(); j++) {
							polygon.push_back(current_shape[j]);
							colors.push_back(c_bg);
						}
					} else if (shape->get_polygon_count() > 0) {
						PoolVector<Vector2> vertices = shape->get_vertices();
						for (int j = 0; j < shape->get_polygon(0).size(); j++) {
							polygon.push_back(vertices[shape->get_polygon(0)[j]] + anchor);
							colors.push_back(c_bg);
						}
					}
					workspace->draw_polygon(polygon, colors);
					if (coord == edited_shape_coord) {
						if (shape->get_polygon_count() > 0) {
							PoolVector<Vector2> vertices = shape->get_vertices();
							for (int j = 0; j < shape->get_polygon(0).size() - 1; j++) {
								workspace->draw_line(vertices[shape->get_polygon(0)[j]] + anchor, vertices[shape->get_polygon(0)[j + 1]] + anchor, c_border, 1, true);
							}
							if (shape == edited_navigation_shape) {
								for (int j = 0; j < current_shape.size(); j++) {
									workspace->draw_circle(current_shape[j], 5, Color(1, 0, 0));
								}
							}
						}
					}
				}
			}
		} break;
	}
	if (creating_shape) {
		for (int j = 0; j < current_shape.size() - 1; j++) {
			workspace->draw_line(current_shape[j], current_shape[j + 1], Color(0, 1, 1), 1, true);
		}
		workspace->draw_line(current_shape[current_shape.size() - 1], snap_point(workspace->get_local_mouse_position()), Color(0, 1, 1), 1, true);
	}
}

void AutotileEditor::close_shape(const Vector2 &shape_anchor) {

	creating_shape = false;

	if (edit_mode == EDITMODE_COLLISION) {
		Ref<ConcavePolygonShape2D> shape = memnew(ConcavePolygonShape2D);

		PoolVector<Vector2> segments;
		segments.resize(current_shape.size() * 2);
		PoolVector<Vector2>::Write w = segments.write();

		for (int i = 0; i < current_shape.size(); i++) {
			w[(i << 1) + 0] = current_shape[i] - shape_anchor;
			w[(i << 1) + 1] = current_shape[(i + 1) % current_shape.size()] - shape_anchor;
		}

		w = PoolVector<Vector2>::Write();
		shape->set_segments(segments);

		tile_set->tile_add_shape(get_current_tile(), shape, Transform2D(), false, edited_shape_coord);
		edited_collision_shape = shape;
		tools[TOOL_SELECT]->set_pressed(true);
		workspace->update();
	} else if (edit_mode == EDITMODE_OCCLUSION) {
		Ref<OccluderPolygon2D> shape = memnew(OccluderPolygon2D);

		PoolVector<Vector2> polygon;
		polygon.resize(current_shape.size());
		PoolVector<Vector2>::Write w = polygon.write();

		for (int i = 0; i < current_shape.size(); i++) {
			w[i] = current_shape[i] - shape_anchor;
		}

		w = PoolVector<Vector2>::Write();
		shape->set_polygon(polygon);

		tile_set->autotile_set_light_occluder(get_current_tile(), shape, edited_shape_coord);
		edited_occlusion_shape = shape;
		tools[TOOL_SELECT]->set_pressed(true);
		workspace->update();
	} else if (edit_mode == EDITMODE_NAVIGATION) {
		Ref<NavigationPolygon> shape = memnew(NavigationPolygon);

		PoolVector<Vector2> polygon;
		Vector<int> indices;
		polygon.resize(current_shape.size());
		PoolVector<Vector2>::Write w = polygon.write();

		for (int i = 0; i < current_shape.size(); i++) {
			w[i] = current_shape[i] - shape_anchor;
			indices.push_back(i);
		}

		w = PoolVector<Vector2>::Write();
		shape->set_vertices(polygon);
		shape->add_polygon(indices);
		tile_set->autotile_set_navigation_polygon(get_current_tile(), shape, edited_shape_coord);
		edited_navigation_shape = shape;
		tools[TOOL_SELECT]->set_pressed(true);
		workspace->update();
	}
}

Vector2 AutotileEditor::snap_point(const Vector2 &point) {
	Vector2 p = point;
	Vector2 coord = edited_shape_coord;
	Vector2 tile_size = tile_set->autotile_get_size(get_current_tile());
	int spacing = tile_set->autotile_get_spacing(get_current_tile());
	Vector2 anchor = coord;
	anchor.x *= (tile_size.x + spacing);
	anchor.y *= (tile_size.y + spacing);
	Rect2 region(anchor, tile_size);
	if (tools[SHAPE_KEEP_INSIDE_TILE]->is_pressed()) {
		if (p.x < region.position.x)
			p.x = region.position.x;
		if (p.y < region.position.y)
			p.y = region.position.y;
		if (p.x > region.position.x + region.size.x)
			p.x = region.position.x + region.size.x;
		if (p.y > region.position.y + region.size.y)
			p.y = region.position.y + region.size.y;
	}
	if (tools[SHAPE_SNAP_TO_BITMASK_GRID]->is_pressed()) {
		Vector2 p2 = p;
		if (tile_set->autotile_get_bitmask_mode(get_current_tile()) == TileSet::BITMASK_2X2) {
			p2.x = Math::stepify(p2.x, tile_size.x / 2);
			p2.y = Math::stepify(p2.y, tile_size.y / 2);
			if ((p2 - p).length_squared() <= MAX(tile_size.y / 4, MIN_DISTANCE_SQUARED)) {
				p = p2;
			}
		} else if (tile_set->autotile_get_bitmask_mode(get_current_tile()) == TileSet::BITMASK_3X3) {
			p2.x = Math::stepify(p2.x, tile_size.x / 3);
			p2.y = Math::stepify(p2.y, tile_size.y / 3);
			if ((p2 - p).length_squared() <= MAX(tile_size.y / 6, MIN_DISTANCE_SQUARED)) {
				p = p2;
			}
		}
	}
	p.floor();
	return p;
}

void AutotileEditor::edit(Object *p_node) {

	tile_set = Ref<TileSet>(Object::cast_to<TileSet>(p_node));
	helper->set_tileset(tile_set);

	autotile_list->clear();
	List<int> ids;
	tile_set->get_tile_list(&ids);
	for (List<int>::Element *E = ids.front(); E; E = E->next()) {
		if (tile_set->tile_get_is_autotile(E->get())) {
			autotile_list->add_item(tile_set->tile_get_name(E->get()));
			autotile_list->set_item_metadata(autotile_list->get_item_count() - 1, E->get());
			autotile_list->set_item_icon(autotile_list->get_item_count() - 1, tile_set->tile_get_texture(E->get()));
			Rect2 region = tile_set->tile_get_region(E->get());
			region.size = tile_set->autotile_get_size(E->get());
			Vector2 pos = tile_set->autotile_get_icon_coordinate(E->get());
			pos.x *= (tile_set->autotile_get_spacing(E->get()) + region.size.x);
			pos.y *= (tile_set->autotile_get_spacing(E->get()) + region.size.y);
			region.position += pos;
			autotile_list->set_item_icon_region(autotile_list->get_item_count() - 1, region);
		}
	}
	if (autotile_list->get_item_count() > 0) {
		autotile_list->select(0);
		_on_autotile_selected(0);
	}
	helper->_change_notify("");
}

int AutotileEditor::get_current_tile() {

	if (autotile_list->get_selected_items().size() == 0)
		return -1;
	else
		return autotile_list->get_item_metadata(autotile_list->get_selected_items()[0]);
}

void AutotileEditorHelper::set_tileset(const Ref<TileSet> &p_tileset) {

	tile_set = p_tileset;
}

bool AutotileEditorHelper::_set(const StringName &p_name, const Variant &p_value) {

	if (autotile_editor->get_current_tile() < 0 || tile_set.is_null())
		return false;

	String name = p_name.operator String();
	bool v = false;
	if (name == "bitmask_mode") {
		tile_set->set(String::num(autotile_editor->get_current_tile(), 0) + "/autotile/bitmask_mode", p_value, &v);
	} else if (name.left(7) == "layout/") {
		tile_set->set(String::num(autotile_editor->get_current_tile(), 0) + "/autotile" + name.right(6), p_value, &v);
	}
	if (v) {
		tile_set->_change_notify("");
		autotile_editor->workspace->update();
	}
	return v;
}

bool AutotileEditorHelper::_get(const StringName &p_name, Variant &r_ret) const {

	if (autotile_editor->get_current_tile() < 0 || tile_set.is_null())
		return false;

	String name = p_name.operator String();
	bool v = false;
	if (name == "bitmask_mode") {
		r_ret = tile_set->get(String::num(autotile_editor->get_current_tile(), 0) + "/autotile/bitmask_mode", &v);
	} else if (name.left(7) == "layout/") {
		r_ret = tile_set->get(String::num(autotile_editor->get_current_tile(), 0) + "/autotile" + name.right(6), &v);
	}
	return v;
}

void AutotileEditorHelper::_get_property_list(List<PropertyInfo> *p_list) const {

	if (autotile_editor->get_current_tile() < 0 || tile_set.is_null())
		return;

	p_list->push_back(PropertyInfo(Variant::INT, "bitmask_mode", PROPERTY_HINT_ENUM, "2x2,3x3"));
	p_list->push_back(PropertyInfo(Variant::VECTOR2, "layout/tile_size"));
	p_list->push_back(PropertyInfo(Variant::INT, "layout/spacing", PROPERTY_HINT_RANGE, "0,256,1"));
}

AutotileEditorHelper::AutotileEditorHelper(AutotileEditor *p_autotile_editor) {

	autotile_editor = p_autotile_editor;
}
