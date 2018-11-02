/*************************************************************************/
/*  tile_set_editor_plugin.cpp                                           */
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

#include "tile_set_editor_plugin.h"

#include "core/os/input.h"
#include "core/os/keyboard.h"
#include "editor/plugins/canvas_item_editor_plugin.h"
#include "scene/2d/physics_body_2d.h"
#include "scene/2d/sprite.h"

void TileSetEditor::edit(const Ref<TileSet> &p_tileset) {

	tileset = p_tileset;
	tileset->add_change_receptor(this);

	texture_list->clear();
	texture_map.clear();
	update_texture_list();
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

				Transform2D shape_transform = sb->get_transform() * sb->shape_owner_get_transform(E->get());
				bool one_way = sb->is_shape_owner_one_way_collision_enabled(E->get());

				shape_transform[2] -= phys_offset;

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
		p_library->tile_set_z_index(id, mi->get_z_index());
	}
}

void TileSetEditor::_import_scene(Node *p_scene, Ref<TileSet> p_library, bool p_merge) {

	if (!p_merge)
		p_library->clear();

	_import_node(p_scene, p_library);
}

Error TileSetEditor::update_library_file(Node *p_base_scene, Ref<TileSet> ml, bool p_merge) {

	_import_scene(p_base_scene, ml, p_merge);
	return OK;
}

void TileSetEditor::_bind_methods() {
	ClassDB::bind_method("_on_tileset_toolbar_button_pressed", &TileSetEditor::_on_tileset_toolbar_button_pressed);
	ClassDB::bind_method("_on_textures_added", &TileSetEditor::_on_textures_added);
	ClassDB::bind_method("_on_tileset_toolbar_confirm", &TileSetEditor::_on_tileset_toolbar_confirm);
	ClassDB::bind_method("_on_texture_list_selected", &TileSetEditor::_on_texture_list_selected);
	ClassDB::bind_method("_on_edit_mode_changed", &TileSetEditor::_on_edit_mode_changed);
	ClassDB::bind_method("_on_workspace_mode_changed", &TileSetEditor::_on_workspace_mode_changed);
	ClassDB::bind_method("_on_workspace_overlay_draw", &TileSetEditor::_on_workspace_overlay_draw);
	ClassDB::bind_method("_on_workspace_process", &TileSetEditor::_on_workspace_process);
	ClassDB::bind_method("_on_workspace_draw", &TileSetEditor::_on_workspace_draw);
	ClassDB::bind_method("_on_workspace_input", &TileSetEditor::_on_workspace_input);
	ClassDB::bind_method("_on_tool_clicked", &TileSetEditor::_on_tool_clicked);
	ClassDB::bind_method("_on_priority_changed", &TileSetEditor::_on_priority_changed);
	ClassDB::bind_method("_on_grid_snap_toggled", &TileSetEditor::_on_grid_snap_toggled);
	ClassDB::bind_method("_set_snap_step", &TileSetEditor::_set_snap_step);
	ClassDB::bind_method("_set_snap_off", &TileSetEditor::_set_snap_off);
	ClassDB::bind_method("_set_snap_sep", &TileSetEditor::_set_snap_sep);
}

void TileSetEditor::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {

		tileset_toolbar_buttons[TOOL_TILESET_ADD_TEXTURE]->set_icon(get_icon("ToolAddNode", "EditorIcons"));
		tileset_toolbar_buttons[TOOL_TILESET_REMOVE_TEXTURE]->set_icon(get_icon("Remove", "EditorIcons"));
		tileset_toolbar_tools->set_icon(get_icon("Tools", "EditorIcons"));

		tool_workspacemode[WORKSPACE_EDIT]->set_icon(get_icon("Edit", "EditorIcons"));
		tool_workspacemode[WORKSPACE_CREATE_SINGLE]->set_icon(get_icon("AddSingleTile", "EditorIcons"));
		tool_workspacemode[WORKSPACE_CREATE_AUTOTILE]->set_icon(get_icon("AddAutotile", "EditorIcons"));
		tool_workspacemode[WORKSPACE_CREATE_ATLAS]->set_icon(get_icon("AddAtlasTile", "EditorIcons"));

		tools[TOOL_SELECT]->set_icon(get_icon("ToolSelect", "EditorIcons"));
		tools[BITMASK_COPY]->set_icon(get_icon("Duplicate", "EditorIcons"));
		tools[BITMASK_PASTE]->set_icon(get_icon("Override", "EditorIcons"));
		tools[BITMASK_CLEAR]->set_icon(get_icon("Clear", "EditorIcons"));
		tools[SHAPE_NEW_POLYGON]->set_icon(get_icon("CollisionPolygon2D", "EditorIcons"));
		tools[SHAPE_DELETE]->set_icon(get_icon("Remove", "EditorIcons"));
		tools[SHAPE_KEEP_INSIDE_TILE]->set_icon(get_icon("Snap", "EditorIcons"));
		tools[TOOL_GRID_SNAP]->set_icon(get_icon("SnapGrid", "EditorIcons"));
		tools[ZOOM_OUT]->set_icon(get_icon("ZoomLess", "EditorIcons"));
		tools[ZOOM_1]->set_icon(get_icon("ZoomReset", "EditorIcons"));
		tools[ZOOM_IN]->set_icon(get_icon("ZoomMore", "EditorIcons"));
		tools[VISIBLE_INFO]->set_icon(get_icon("InformationSign", "EditorIcons"));

		tool_editmode[EDITMODE_REGION]->set_icon(get_icon("RegionEdit", "EditorIcons"));
		tool_editmode[EDITMODE_COLLISION]->set_icon(get_icon("StaticBody2D", "EditorIcons"));
		tool_editmode[EDITMODE_OCCLUSION]->set_icon(get_icon("LightOccluder2D", "EditorIcons"));
		tool_editmode[EDITMODE_NAVIGATION]->set_icon(get_icon("Navigation2D", "EditorIcons"));
		tool_editmode[EDITMODE_BITMASK]->set_icon(get_icon("PackedDataContainer", "EditorIcons"));
		tool_editmode[EDITMODE_PRIORITY]->set_icon(get_icon("MaterialPreviewLight1", "EditorIcons"));
		tool_editmode[EDITMODE_ICON]->set_icon(get_icon("LargeTexture", "EditorIcons"));
	}
}

TileSetEditor::TileSetEditor(EditorNode *p_editor) {

	editor = p_editor;
	set_name("Tile Set Bottom Editor");

	HSplitContainer *split = memnew(HSplitContainer);
	split->set_anchors_and_margins_preset(PRESET_WIDE, PRESET_MODE_MINSIZE, 10);
	add_child(split);

	VBoxContainer *left_container = memnew(VBoxContainer);
	split->add_child(left_container);

	texture_list = memnew(ItemList);
	left_container->add_child(texture_list);
	texture_list->set_v_size_flags(SIZE_EXPAND_FILL);
	texture_list->set_custom_minimum_size(Size2(200, 0));
	texture_list->connect("item_selected", this, "_on_texture_list_selected");

	HBoxContainer *tileset_toolbar_container = memnew(HBoxContainer);
	left_container->add_child(tileset_toolbar_container);

	tileset_toolbar_buttons[TOOL_TILESET_ADD_TEXTURE] = memnew(ToolButton);
	Vector<Variant> p;
	p.push_back((int)TOOL_TILESET_ADD_TEXTURE);
	tileset_toolbar_buttons[TOOL_TILESET_ADD_TEXTURE]->connect("pressed", this, "_on_tileset_toolbar_button_pressed", p);
	tileset_toolbar_container->add_child(tileset_toolbar_buttons[TOOL_TILESET_ADD_TEXTURE]);
	tileset_toolbar_buttons[TOOL_TILESET_ADD_TEXTURE]->set_tooltip(TTR("Add Texture(s) to TileSet"));

	tileset_toolbar_buttons[TOOL_TILESET_REMOVE_TEXTURE] = memnew(ToolButton);
	p = Vector<Variant>();
	p.push_back((int)TOOL_TILESET_REMOVE_TEXTURE);
	tileset_toolbar_buttons[TOOL_TILESET_REMOVE_TEXTURE]->connect("pressed", this, "_on_tileset_toolbar_button_pressed", p);
	tileset_toolbar_container->add_child(tileset_toolbar_buttons[TOOL_TILESET_REMOVE_TEXTURE]);
	tileset_toolbar_buttons[TOOL_TILESET_REMOVE_TEXTURE]->set_tooltip(TTR("Remove current Texture from TileSet"));

	Control *toolbar_separator = memnew(Control);
	toolbar_separator->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	tileset_toolbar_container->add_child(toolbar_separator);

	tileset_toolbar_tools = memnew(MenuButton);
	tileset_toolbar_tools->set_text("Tools");
	p = Vector<Variant>();
	p.push_back((int)TOOL_TILESET_CREATE_SCENE);
	tileset_toolbar_tools->get_popup()->add_item(TTR("Create from Scene"), TOOL_TILESET_CREATE_SCENE);
	p = Vector<Variant>();
	p.push_back((int)TOOL_TILESET_MERGE_SCENE);
	tileset_toolbar_tools->get_popup()->add_item(TTR("Merge from Scene"), TOOL_TILESET_MERGE_SCENE);

	tileset_toolbar_tools->get_popup()->connect("id_pressed", this, "_on_tileset_toolbar_button_pressed");
	tileset_toolbar_container->add_child(tileset_toolbar_tools);

	//---------------
	VBoxContainer *right_container = memnew(VBoxContainer);
	right_container->set_v_size_flags(SIZE_EXPAND_FILL);
	split->add_child(right_container);

	dragging_point = -1;
	creating_shape = false;
	snap_step = Vector2(32, 32);
	snap_offset = WORKSPACE_MARGIN;

	set_custom_minimum_size(Size2(0, 150));

	VBoxContainer *main_vb = memnew(VBoxContainer);
	right_container->add_child(main_vb);
	main_vb->set_v_size_flags(SIZE_EXPAND_FILL);

	HBoxContainer *tool_hb = memnew(HBoxContainer);
	Ref<ButtonGroup> g(memnew(ButtonGroup));

	String workspace_label[WORKSPACE_MODE_MAX] = { "Edit", "New Single Tile", "New Autotile", "New Atlas" };

	for (int i = 0; i < (int)WORKSPACE_MODE_MAX; i++) {
		tool_workspacemode[i] = memnew(Button);
		tool_workspacemode[i]->set_text(workspace_label[i]);
		tool_workspacemode[i]->set_toggle_mode(true);
		tool_workspacemode[i]->set_button_group(g);
		Vector<Variant> p;
		p.push_back(i);
		tool_workspacemode[i]->connect("pressed", this, "_on_workspace_mode_changed", p);
		tool_hb->add_child(tool_workspacemode[i]);
	}
	Control *spacer = memnew(Control);
	spacer->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	tool_hb->add_child(spacer);
	tool_hb->move_child(spacer, (int)WORKSPACE_CREATE_SINGLE);

	tool_workspacemode[WORKSPACE_EDIT]->set_pressed(true);
	workspace_mode = WORKSPACE_EDIT;

	main_vb->add_child(tool_hb);
	main_vb->add_child(memnew(HSeparator));

	tool_hb = memnew(HBoxContainer);

	g = Ref<ButtonGroup>(memnew(ButtonGroup));
	String label[EDITMODE_MAX] = { "Region", "Collision", "Occlusion", "Navigation", "Bitmask", "Priority", "Icon" };

	for (int i = 0; i < (int)EDITMODE_MAX; i++) {
		tool_editmode[i] = memnew(Button);
		tool_editmode[i]->set_text(label[i]);
		tool_editmode[i]->set_toggle_mode(true);
		tool_editmode[i]->set_button_group(g);
		Vector<Variant> p;
		p.push_back(i);
		tool_editmode[i]->connect("pressed", this, "_on_edit_mode_changed", p);
		tool_hb->add_child(tool_editmode[i]);
	}
	tool_editmode[EDITMODE_COLLISION]->set_pressed(true);
	edit_mode = EDITMODE_COLLISION;

	main_vb->add_child(tool_hb);
	separator_editmode = memnew(HSeparator);
	main_vb->add_child(separator_editmode);

	toolbar = memnew(HBoxContainer);
	Ref<ButtonGroup> tg(memnew(ButtonGroup));

	p = Vector<Variant>();
	tools[TOOL_SELECT] = memnew(ToolButton);
	toolbar->add_child(tools[TOOL_SELECT]);
	tools[TOOL_SELECT]->set_tooltip(TTR("Select sub-tile to use as icon, this will be also used on invalid autotile bindings."));
	tools[TOOL_SELECT]->set_toggle_mode(true);
	tools[TOOL_SELECT]->set_button_group(tg);
	tools[TOOL_SELECT]->set_pressed(true);
	p.push_back((int)TOOL_SELECT);
	tools[TOOL_SELECT]->connect("pressed", this, "_on_tool_clicked", p);

	tools[BITMASK_COPY] = memnew(ToolButton);
	p.push_back((int)BITMASK_COPY);
	tools[BITMASK_COPY]->connect("pressed", this, "_on_tool_clicked", p);
	toolbar->add_child(tools[BITMASK_COPY]);
	tools[BITMASK_PASTE] = memnew(ToolButton);
	p = Vector<Variant>();
	p.push_back((int)BITMASK_PASTE);
	tools[BITMASK_PASTE]->connect("pressed", this, "_on_tool_clicked", p);
	toolbar->add_child(tools[BITMASK_PASTE]);
	tools[BITMASK_CLEAR] = memnew(ToolButton);
	p = Vector<Variant>();
	p.push_back((int)BITMASK_CLEAR);
	tools[BITMASK_CLEAR]->connect("pressed", this, "_on_tool_clicked", p);
	toolbar->add_child(tools[BITMASK_CLEAR]);

	tools[SHAPE_NEW_POLYGON] = memnew(ToolButton);
	toolbar->add_child(tools[SHAPE_NEW_POLYGON]);
	tools[SHAPE_NEW_POLYGON]->set_toggle_mode(true);
	tools[SHAPE_NEW_POLYGON]->set_button_group(tg);

	separator_delete = memnew(VSeparator);
	toolbar->add_child(separator_delete);
	tools[SHAPE_DELETE] = memnew(ToolButton);
	p = Vector<Variant>();
	p.push_back((int)SHAPE_DELETE);
	tools[SHAPE_DELETE]->connect("pressed", this, "_on_tool_clicked", p);
	toolbar->add_child(tools[SHAPE_DELETE]);

	separator_grid = memnew(VSeparator);
	toolbar->add_child(separator_grid);
	tools[SHAPE_KEEP_INSIDE_TILE] = memnew(ToolButton);
	tools[SHAPE_KEEP_INSIDE_TILE]->set_toggle_mode(true);
	tools[SHAPE_KEEP_INSIDE_TILE]->set_pressed(true);
	toolbar->add_child(tools[SHAPE_KEEP_INSIDE_TILE]);
	tools[TOOL_GRID_SNAP] = memnew(ToolButton);
	tools[TOOL_GRID_SNAP]->set_toggle_mode(true);
	tools[TOOL_GRID_SNAP]->connect("toggled", this, "_on_grid_snap_toggled");
	toolbar->add_child(tools[TOOL_GRID_SNAP]);

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
	tools[ZOOM_OUT]->set_tooltip(TTR("Zoom Out"));
	tools[ZOOM_1] = memnew(ToolButton);
	p = Vector<Variant>();
	p.push_back((int)ZOOM_1);
	tools[ZOOM_1]->connect("pressed", this, "_on_tool_clicked", p);
	toolbar->add_child(tools[ZOOM_1]);
	tools[ZOOM_1]->set_tooltip(TTR("Reset Zoom"));
	tools[ZOOM_IN] = memnew(ToolButton);
	p = Vector<Variant>();
	p.push_back((int)ZOOM_IN);
	tools[ZOOM_IN]->connect("pressed", this, "_on_tool_clicked", p);
	toolbar->add_child(tools[ZOOM_IN]);
	tools[ZOOM_IN]->set_tooltip(TTR("Zoom In"));

	tools[VISIBLE_INFO] = memnew(ToolButton);
	tools[VISIBLE_INFO]->set_toggle_mode(true);
	tools[VISIBLE_INFO]->set_tooltip(TTR("Display tile's names (hold Alt Key)"));
	toolbar->add_child(tools[VISIBLE_INFO]);

	main_vb->add_child(toolbar);

	scroll = memnew(ScrollContainer);
	main_vb->add_child(scroll);
	scroll->set_v_size_flags(SIZE_EXPAND_FILL);
	scroll->set_clip_contents(true);

	workspace_container = memnew(Control);
	scroll->add_child(workspace_container);

	workspace_overlay = memnew(Control);
	workspace_overlay->connect("draw", this, "_on_workspace_overlay_draw");
	workspace_container->add_child(workspace_overlay);

	workspace = memnew(Control);
	workspace->set_focus_mode(FOCUS_ALL);
	workspace->connect("draw", this, "_on_workspace_draw");
	workspace->connect("gui_input", this, "_on_workspace_input");
	workspace->set_draw_behind_parent(true);
	workspace_overlay->add_child(workspace);

	preview = memnew(Sprite);
	workspace->add_child(preview);
	preview->set_centered(false);
	preview->set_draw_behind_parent(true);
	preview->set_position(WORKSPACE_MARGIN);

	//---------------
	cd = memnew(ConfirmationDialog);
	add_child(cd);
	cd->connect("confirmed", this, "_on_tileset_toolbar_confirm");

	//---------------
	err_dialog = memnew(AcceptDialog);
	add_child(err_dialog);

	//---------------
	texture_dialog = memnew(EditorFileDialog);
	texture_dialog->set_access(EditorFileDialog::ACCESS_RESOURCES);
	texture_dialog->set_mode(EditorFileDialog::MODE_OPEN_FILES);
	texture_dialog->clear_filters();
	List<String> extensions;

	ResourceLoader::get_recognized_extensions_for_type("Texture", &extensions);
	for (List<String>::Element *E = extensions.front(); E; E = E->next()) {

		texture_dialog->add_filter("*." + E->get() + " ; " + E->get().to_upper());
	}
	add_child(texture_dialog);
	texture_dialog->connect("files_selected", this, "_on_textures_added");

	//---------------
	helper = memnew(TilesetEditorContext(this));
	tile_names_opacity = 0;

	// config scale
	max_scale = 10.0f;
	min_scale = 0.1f;
	scale_ratio = 1.2f;
}

TileSetEditor::~TileSetEditor() {
	if (helper)
		memdelete(helper);
}

void TileSetEditor::_on_tileset_toolbar_button_pressed(int p_index) {
	option = p_index;
	switch (option) {
		case TOOL_TILESET_ADD_TEXTURE: {
			texture_dialog->popup_centered_ratio();
		} break;
		case TOOL_TILESET_REMOVE_TEXTURE: {
			if (get_current_texture().is_valid()) {
				cd->set_text(TTR("Remove selected texture and ALL TILES which use it?"));
				cd->popup_centered(Size2(300, 60));
			} else {
				err_dialog->set_text(TTR("You haven't selected a texture to remove."));
				err_dialog->popup_centered(Size2(300, 60));
			}
		} break;
		case TOOL_TILESET_CREATE_SCENE: {

			cd->set_text(TTR("Create from scene?"));
			cd->popup_centered(Size2(300, 60));
		} break;
		case TOOL_TILESET_MERGE_SCENE: {

			cd->set_text(TTR("Merge from scene?"));
			cd->popup_centered(Size2(300, 60));
		} break;
	}
}

void TileSetEditor::_on_tileset_toolbar_confirm() {
	switch (option) {
		case TOOL_TILESET_REMOVE_TEXTURE: {
			RID current_rid = get_current_texture()->get_rid();
			List<int> ids;
			tileset->get_tile_list(&ids);
			for (List<int>::Element *E = ids.front(); E; E = E->next()) {
				if (tileset->tile_get_texture(E->get())->get_rid() == current_rid) {
					tileset->remove_tile(E->get());
				}
			}
			texture_list->remove_item(texture_list->find_metadata(current_rid));
			texture_map.erase(current_rid);
			_on_texture_list_selected(-1);
		} break;
		case TOOL_TILESET_MERGE_SCENE:
		case TOOL_TILESET_CREATE_SCENE: {

			EditorNode *en = editor;
			Node *scene = en->get_edited_scene();
			if (!scene)
				break;
			_import_scene(scene, tileset, option == TOOL_TILESET_MERGE_SCENE);

			edit(tileset);
		} break;
	}
}

void TileSetEditor::_on_texture_list_selected(int p_index) {
	if (get_current_texture().is_valid()) {
		current_item_index = p_index;
		preview->set_texture(get_current_texture());
		workspace->set_custom_minimum_size(get_current_texture()->get_size() + WORKSPACE_MARGIN * 2);
		workspace_container->set_custom_minimum_size(get_current_texture()->get_size() + WORKSPACE_MARGIN * 2);
		workspace_overlay->set_custom_minimum_size(get_current_texture()->get_size() + WORKSPACE_MARGIN * 2);
		update_workspace_tile_mode();
	} else {
		current_item_index = -1;
		preview->set_texture(NULL);
		workspace->set_custom_minimum_size(Size2i());
		update_workspace_tile_mode();
	}
	set_current_tile(-1);
	workspace->update();
}

void TileSetEditor::_on_textures_added(const PoolStringArray &p_paths) {
	int invalid_count = 0;
	for (int i = 0; i < p_paths.size(); i++) {
		Ref<Texture> t = Ref<Texture>(ResourceLoader::load(p_paths[i]));

		ERR_EXPLAIN("'" + p_paths[i] + "' is not a valid texture.");
		ERR_CONTINUE(!t.is_valid());

		if (texture_map.has(t->get_rid())) {
			invalid_count++;
		} else {
			texture_list->add_item(t->get_path().get_file());
			texture_map.insert(t->get_rid(), t);
			texture_list->set_item_metadata(texture_list->get_item_count() - 1, t->get_rid());
		}
	}

	if (texture_list->get_item_count() > 0) {
		update_texture_list_icon();
		texture_list->select(texture_list->get_item_count() - 1);
		_on_texture_list_selected(texture_list->get_item_count() - 1);
	}

	if (invalid_count > 0) {
		err_dialog->set_text(vformat(TTR("%s file(s) were not added because was already on the list."), String::num(invalid_count, 0)));
		err_dialog->popup_centered(Size2(300, 60));
	}
}

void TileSetEditor::_on_edit_mode_changed(int p_edit_mode) {
	edit_mode = (EditMode)p_edit_mode;
	switch (edit_mode) {
		case EDITMODE_REGION: {
			tools[TOOL_SELECT]->show();
			tools[BITMASK_COPY]->hide();
			tools[BITMASK_PASTE]->hide();
			tools[BITMASK_CLEAR]->hide();
			tools[SHAPE_NEW_POLYGON]->hide();

			if (workspace_mode == WORKSPACE_EDIT) {
				separator_delete->show();
				tools[SHAPE_DELETE]->show();
			} else {
				separator_delete->hide();
				tools[SHAPE_DELETE]->hide();
			}

			separator_grid->show();
			tools[SHAPE_KEEP_INSIDE_TILE]->hide();
			tools[TOOL_GRID_SNAP]->show();

			tools[TOOL_SELECT]->set_pressed(true);
			tools[TOOL_SELECT]->set_tooltip(TTR("Drag handles to edit Rect.\nClick on another Tile to edit it."));
			spin_priority->hide();
		} break;
		case EDITMODE_BITMASK: {
			tools[TOOL_SELECT]->show();
			tools[BITMASK_COPY]->show();
			tools[BITMASK_PASTE]->show();
			tools[BITMASK_CLEAR]->show();
			tools[SHAPE_NEW_POLYGON]->hide();

			separator_delete->hide();
			tools[SHAPE_DELETE]->hide();

			separator_grid->hide();
			tools[SHAPE_KEEP_INSIDE_TILE]->hide();
			tools[TOOL_GRID_SNAP]->hide();

			tools[TOOL_SELECT]->set_pressed(true);
			tools[TOOL_SELECT]->set_tooltip(TTR("LMB: set bit on.\nRMB: set bit off.\nClick on another Tile to edit it."));
			spin_priority->hide();
		} break;
		case EDITMODE_COLLISION:
		case EDITMODE_NAVIGATION:
		case EDITMODE_OCCLUSION: {
			tools[TOOL_SELECT]->show();
			tools[BITMASK_COPY]->hide();
			tools[BITMASK_PASTE]->hide();
			tools[BITMASK_CLEAR]->hide();
			tools[SHAPE_NEW_POLYGON]->show();

			separator_delete->show();
			tools[SHAPE_DELETE]->show();

			separator_grid->show();
			tools[SHAPE_KEEP_INSIDE_TILE]->show();
			tools[TOOL_GRID_SNAP]->show();

			tools[TOOL_SELECT]->set_tooltip(TTR("Select current edited sub-tile.\nClick on another Tile to edit it."));
			spin_priority->hide();
			select_coord(edited_shape_coord);
		} break;
		default: {
			tools[TOOL_SELECT]->show();
			tools[BITMASK_COPY]->hide();
			tools[BITMASK_PASTE]->hide();
			tools[BITMASK_CLEAR]->hide();
			tools[SHAPE_NEW_POLYGON]->hide();

			separator_delete->hide();
			tools[SHAPE_DELETE]->hide();

			separator_grid->show();
			tools[SHAPE_KEEP_INSIDE_TILE]->hide();
			tools[TOOL_GRID_SNAP]->show();

			if (edit_mode == EDITMODE_ICON) {
				tools[TOOL_SELECT]->set_tooltip(TTR("Select sub-tile to use as icon, this will be also used on invalid autotile bindings.\nClick on another Tile to edit it."));
				spin_priority->hide();
			} else {
				tools[TOOL_SELECT]->set_tooltip(TTR("Select sub-tile to change its priority.\nClick on another Tile to edit it."));
				spin_priority->show();
			}
		} break;
	}
	workspace->update();
}

void TileSetEditor::_on_workspace_mode_changed(int p_workspace_mode) {
	workspace_mode = (WorkspaceMode)p_workspace_mode;
	if (p_workspace_mode == WORKSPACE_EDIT) {
		update_workspace_tile_mode();
	} else {
		for (int i = 0; i < EDITMODE_MAX; i++) {
			tool_editmode[i]->hide();
		}
		tool_editmode[EDITMODE_REGION]->show();
		tool_editmode[EDITMODE_REGION]->set_pressed(true);
		_on_edit_mode_changed(EDITMODE_REGION);
		separator_editmode->show();
	}
}

void TileSetEditor::_on_workspace_draw() {

	const Color COLOR_AUTOTILE = Color(0.266373, 0.565288, 0.988281);
	const Color COLOR_SINGLE = Color(0.988281, 0.909323, 0.266373);
	const Color COLOR_ATLAS = Color(0.78653, 0.812835, 0.832031);

	if (tileset.is_null())
		return;
	if (!get_current_texture().is_valid())
		return;

	draw_highlight_current_tile();

	draw_grid_snap();
	if (get_current_tile() >= 0) {
		int spacing = tileset->autotile_get_spacing(get_current_tile());
		Vector2 size = tileset->autotile_get_size(get_current_tile());
		Rect2i region = tileset->tile_get_region(get_current_tile());

		switch (edit_mode) {
			case EDITMODE_ICON: {
				Vector2 coord = tileset->autotile_get_icon_coordinate(get_current_tile());
				draw_highlight_subtile(coord);
			} break;
			case EDITMODE_BITMASK: {
				Color c(1, 0, 0, 0.5);
				for (float x = 0; x < region.size.x / (spacing + size.x); x++) {
					for (float y = 0; y < region.size.y / (spacing + size.y); y++) {
						Vector2 coord(x, y);
						Point2 anchor(coord.x * (spacing + size.x), coord.y * (spacing + size.y));
						anchor += WORKSPACE_MARGIN;
						anchor += region.position;
						uint16_t mask = tileset->autotile_get_bitmask(get_current_tile(), coord);
						if (tileset->autotile_get_bitmask_mode(get_current_tile()) == TileSet::BITMASK_2X2) {
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
						} else {
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
				if (tileset->tile_get_tile_mode(get_current_tile()) == TileSet::AUTO_TILE || tileset->tile_get_tile_mode(get_current_tile()) == TileSet::ATLAS_TILE) {
					Vector2 coord = edited_shape_coord;
					draw_highlight_subtile(coord);
				}
				draw_polygon_shapes();
				draw_grid_snap();
			} break;
			case EDITMODE_PRIORITY: {
				spin_priority->set_value(tileset->autotile_get_subtile_priority(get_current_tile(), edited_shape_coord));
				uint16_t mask = tileset->autotile_get_bitmask(get_current_tile(), edited_shape_coord);
				Vector<Vector2> queue_others;
				int total = 0;
				for (Map<Vector2, uint16_t>::Element *E = tileset->autotile_get_bitmask_map(get_current_tile()).front(); E; E = E->next()) {
					if (E->value() == mask) {
						total += tileset->autotile_get_subtile_priority(get_current_tile(), E->key());
						if (E->key() != edited_shape_coord) {
							queue_others.push_back(E->key());
						}
					}
				}
				spin_priority->set_suffix(" / " + String::num(total, 0));
				draw_highlight_subtile(edited_shape_coord, queue_others);
			} break;
			default: {}
		}

		draw_tile_subdivision(get_current_tile(), Color(0.347214, 0.722656, 0.617063));
	}

	RID current_texture_rid = get_current_texture()->get_rid();
	List<int> *tiles = new List<int>();
	tileset->get_tile_list(tiles);
	for (List<int>::Element *E = tiles->front(); E; E = E->next()) {
		int t_id = E->get();
		if (tileset->tile_get_texture(t_id)->get_rid() == current_texture_rid && (t_id != get_current_tile() || edit_mode != EDITMODE_REGION)) {
			Rect2i region = tileset->tile_get_region(t_id);
			region.position += WORKSPACE_MARGIN;
			Color c;
			if (tileset->tile_get_tile_mode(t_id) == TileSet::SINGLE_TILE)
				c = COLOR_SINGLE;
			else if (tileset->tile_get_tile_mode(t_id) == TileSet::AUTO_TILE)
				c = COLOR_AUTOTILE;
			else if (tileset->tile_get_tile_mode(t_id) == TileSet::ATLAS_TILE)
				c = COLOR_ATLAS;
			draw_tile_subdivision(t_id, Color(0.347214, 0.722656, 0.617063, 0.5));
			workspace->draw_rect(region, c, false);
		}
	}
	if (edit_mode == EDITMODE_REGION) {
		if (workspace_mode != WORKSPACE_EDIT) {
			Rect2i region = edited_region;
			Color c;
			if (workspace_mode == WORKSPACE_CREATE_SINGLE)
				c = COLOR_SINGLE;
			else if (workspace_mode == WORKSPACE_CREATE_AUTOTILE)
				c = COLOR_AUTOTILE;
			else if (workspace_mode == WORKSPACE_CREATE_ATLAS)
				c = COLOR_ATLAS;
			workspace->draw_rect(region, c, false);
			draw_edited_region_subdivision();
		} else {
			int t_id = get_current_tile();
			Rect2i region;
			if (draw_edited_region)
				region = edited_region;
			else {
				region = tileset->tile_get_region(t_id);
				region.position += WORKSPACE_MARGIN;
			}
			Color c;
			if (tileset->tile_get_tile_mode(t_id) == TileSet::SINGLE_TILE)
				c = COLOR_SINGLE;
			else if (tileset->tile_get_tile_mode(t_id) == TileSet::AUTO_TILE)
				c = COLOR_AUTOTILE;
			else if (tileset->tile_get_tile_mode(t_id) == TileSet::ATLAS_TILE)
				c = COLOR_ATLAS;
			if (draw_edited_region)
				draw_edited_region_subdivision();
			else
				draw_tile_subdivision(t_id, Color(0.347214, 0.722656, 0.617063, 1));
			workspace->draw_rect(region, c, false);
		}
	}
	workspace_overlay->update();
}

void TileSetEditor::_on_workspace_process() {
	float a = tile_names_opacity;
	if (Input::get_singleton()->is_key_pressed(KEY_ALT) || tools[VISIBLE_INFO]->is_pressed()) {
		a += get_tree()->get_idle_process_time() * 2;
	} else {
		a -= get_tree()->get_idle_process_time() * 2;
	}

	a = CLAMP(a, 0, 1);
	if (a != tile_names_opacity)
		workspace_overlay->update();
	tile_names_opacity = a;
}

void TileSetEditor::_on_workspace_overlay_draw() {

	if (!tileset.is_valid())
		return;
	if (!get_current_texture().is_valid())
		return;

	const Color COLOR_AUTOTILE = Color(0.266373, 0.565288, 0.988281);
	const Color COLOR_SINGLE = Color(0.988281, 0.909323, 0.266373);
	const Color COLOR_ATLAS = Color(0.78653, 0.812835, 0.832031);

	if (tile_names_opacity > 0) {
		RID current_texture_rid = get_current_texture()->get_rid();
		List<int> *tiles = new List<int>();
		tileset->get_tile_list(tiles);
		for (List<int>::Element *E = tiles->front(); E; E = E->next()) {
			int t_id = E->get();
			if (tileset->tile_get_texture(t_id)->get_rid() == current_texture_rid) {
				Rect2i region = tileset->tile_get_region(t_id);
				region.position += WORKSPACE_MARGIN;
				region.position *= workspace->get_scale().x;
				Color c;
				if (tileset->tile_get_tile_mode(t_id) == TileSet::SINGLE_TILE)
					c = COLOR_SINGLE;
				else if (tileset->tile_get_tile_mode(t_id) == TileSet::AUTO_TILE)
					c = COLOR_AUTOTILE;
				else if (tileset->tile_get_tile_mode(t_id) == TileSet::ATLAS_TILE)
					c = COLOR_ATLAS;
				c.a = tile_names_opacity;
				String tile_id_name = String::num(t_id, 0) + ": " + tileset->tile_get_name(t_id);
				Ref<Font> font = get_font("font", "Label");
				region.set_size(font->get_string_size(tile_id_name));
				workspace_overlay->draw_rect(region, c);
				region.position.y += region.size.y - 2;
				c = Color(0.1, 0.1, 0.1, tile_names_opacity);
				workspace_overlay->draw_string(font, region.position, tile_id_name, c);
			}
		}
	}

	int t_id = get_current_tile();
	if (t_id < 0)
		return;

	Ref<Texture> handle = get_icon("EditorHandle", "EditorIcons");
	if (draw_handles) {
		for (int i = 0; i < current_shape.size(); i++) {
			workspace_overlay->draw_texture(handle, current_shape[i] * workspace->get_scale().x - handle->get_size() * 0.5);
		}
	}
}

#define MIN_DISTANCE_SQUARED 6
void TileSetEditor::_on_workspace_input(const Ref<InputEvent> &p_ie) {
	if (tileset.is_null())
		return;
	if (!get_current_texture().is_valid())
		return;

	static bool dragging;
	static bool erasing;
	draw_edited_region = false;

	Rect2 current_tile_region = Rect2();
	if (get_current_tile() >= 0) {
		current_tile_region = tileset->tile_get_region(get_current_tile());
	}
	current_tile_region.position += WORKSPACE_MARGIN;

	Ref<InputEventMouseButton> mb = p_ie;
	Ref<InputEventMouseMotion> mm = p_ie;

	if (mb.is_valid()) {
		if (mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {
			if (!current_tile_region.has_point(mb->get_position())) {
				List<int> *tiles = new List<int>();
				tileset->get_tile_list(tiles);
				for (List<int>::Element *E = tiles->front(); E; E = E->next()) {
					int t_id = E->get();
					if (get_current_texture()->get_rid() == tileset->tile_get_texture(t_id)->get_rid()) {
						Rect2 r = tileset->tile_get_region(t_id);
						r.position += WORKSPACE_MARGIN;
						if (r.has_point(mb->get_position())) {
							set_current_tile(t_id);
							workspace->update();
							workspace_overlay->update();
							return;
						}
					}
				}
			}
		}

		// Mouse Wheel Event
		const int _mouse_button_index = mb->get_button_index();
		if (_mouse_button_index == BUTTON_WHEEL_UP && mb->get_control()) {
			_zoom_in();

		} else if (_mouse_button_index == BUTTON_WHEEL_DOWN && mb->get_control()) {
			_zoom_out();
		}
	}
	// Drag Middle Mouse
	if (mm.is_valid()) {
		if (mm->get_button_mask() & BUTTON_MASK_MIDDLE) {
			Vector2 dragged(mm->get_relative().x, mm->get_relative().y);
			scroll->set_h_scroll(scroll->get_h_scroll() - dragged.x * workspace->get_scale().x);
			scroll->set_v_scroll(scroll->get_v_scroll() - dragged.y * workspace->get_scale().x);
		}
	}

	if (edit_mode == EDITMODE_REGION) {
		if (mb.is_valid()) {
			if (mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {
				if (get_current_tile() >= 0 || workspace_mode != WORKSPACE_EDIT) {
					dragging = true;
					region_from = mb->get_position();
					edited_region = Rect2(region_from, Size2());
					workspace->update();
					workspace_overlay->update();
					return;
				}
			} else if (dragging && mb->is_pressed() && mb->get_button_index() == BUTTON_RIGHT) {
				dragging = false;
				edited_region = Rect2();
				workspace->update();
				workspace_overlay->update();
				return;
			} else if (dragging && !mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {
				dragging = false;
				update_edited_region(mb->get_position());
				edited_region.position -= WORKSPACE_MARGIN;
				if (!edited_region.has_no_area()) {
					if (get_current_tile() >= 0 && workspace_mode == WORKSPACE_EDIT) {
						tileset->tile_set_region(get_current_tile(), edited_region);
					} else {
						int t_id = tileset->get_last_unused_tile_id();
						tileset->create_tile(t_id);
						tileset->tile_set_texture(t_id, get_current_texture());
						tileset->tile_set_region(t_id, edited_region);
						tileset->tile_set_name(t_id, get_current_texture()->get_path().get_file() + " " + String::num(t_id, 0));
						if (workspace_mode != WORKSPACE_CREATE_SINGLE) {
							tileset->autotile_set_size(t_id, snap_step);
							tileset->autotile_set_spacing(t_id, snap_separation.x);
							tileset->tile_set_tile_mode(t_id, workspace_mode == WORKSPACE_CREATE_AUTOTILE ? TileSet::AUTO_TILE : TileSet::ATLAS_TILE);
						}
						set_current_tile(t_id);
						tool_workspacemode[WORKSPACE_EDIT]->set_pressed(true);
						_on_workspace_mode_changed(WORKSPACE_EDIT);
					}
				}
				workspace->update();
				workspace_overlay->update();
				return;
			}
		} else if (mm.is_valid()) {
			if (dragging) {
				update_edited_region(mm->get_position());
				draw_edited_region = true;
				workspace->update();
				workspace_overlay->update();
				return;
			}
		}
	}
	if (workspace_mode == WORKSPACE_EDIT) {

		if (get_current_tile() >= 0) {
			int spacing = tileset->autotile_get_spacing(get_current_tile());
			Vector2 size = tileset->autotile_get_size(get_current_tile());
			switch (edit_mode) {
				case EDITMODE_ICON: {
					if (mb.is_valid()) {
						if (mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT && current_tile_region.has_point(mb->get_position())) {
							Vector2 coord((int)((mb->get_position().x - current_tile_region.position.x) / (spacing + size.x)), (int)((mb->get_position().y - current_tile_region.position.y) / (spacing + size.y)));
							tileset->autotile_set_icon_coordinate(get_current_tile(), coord);
							Rect2 region = tileset->tile_get_region(get_current_tile());
							region.size = size;
							coord.x *= (spacing + size.x);
							coord.y *= (spacing + size.y);
							region.position += coord;
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
							if ((mb->get_button_index() == BUTTON_RIGHT || mb->get_button_index() == BUTTON_LEFT) && current_tile_region.has_point(mb->get_position())) {
								dragging = true;
								erasing = (mb->get_button_index() == BUTTON_RIGHT);
								Vector2 coord((int)((mb->get_position().x - current_tile_region.position.x) / (spacing + size.x)), (int)((mb->get_position().y - current_tile_region.position.y) / (spacing + size.y)));
								Vector2 pos(coord.x * (spacing + size.x), coord.y * (spacing + size.y));
								pos = mb->get_position() - (pos + current_tile_region.position);
								uint16_t bit = 0;
								if (tileset->autotile_get_bitmask_mode(get_current_tile()) == TileSet::BITMASK_2X2) {
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
								} else {
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
								uint16_t mask = tileset->autotile_get_bitmask(get_current_tile(), coord);
								if (erasing) {
									mask &= ~bit;
								} else {
									mask |= bit;
								}
								tileset->autotile_set_bitmask(get_current_tile(), coord, mask);
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
						if (dragging && current_tile_region.has_point(mm->get_position())) {
							Vector2 coord((int)((mm->get_position().x - current_tile_region.position.x) / (spacing + size.x)), (int)((mm->get_position().y - current_tile_region.position.y) / (spacing + size.y)));
							Vector2 pos(coord.x * (spacing + size.x), coord.y * (spacing + size.y));
							pos = mm->get_position() - (pos + current_tile_region.position);
							uint16_t bit = 0;
							if (tileset->autotile_get_bitmask_mode(get_current_tile()) == TileSet::BITMASK_2X2) {
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
							} else {
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
							uint16_t mask = tileset->autotile_get_bitmask(get_current_tile(), coord);
							if (erasing) {
								mask &= ~bit;
							} else {
								mask |= bit;
							}
							tileset->autotile_set_bitmask(get_current_tile(), coord, mask);
							workspace->update();
						}
					}
				} break;
				case EDITMODE_COLLISION:
				case EDITMODE_OCCLUSION:
				case EDITMODE_NAVIGATION:
				case EDITMODE_PRIORITY: {
					Vector2 shape_anchor = Vector2(0, 0);
					if (tileset->tile_get_tile_mode(get_current_tile()) == TileSet::AUTO_TILE || tileset->tile_get_tile_mode(get_current_tile()) == TileSet::ATLAS_TILE) {
						shape_anchor = edited_shape_coord;
						shape_anchor.x *= (size.x + spacing);
						shape_anchor.y *= (size.y + spacing);
					}
					shape_anchor += current_tile_region.position;
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
								if ((tileset->tile_get_tile_mode(get_current_tile()) == TileSet::AUTO_TILE || tileset->tile_get_tile_mode(get_current_tile()) == TileSet::ATLAS_TILE) && current_tile_region.has_point(mb->get_position())) {
									Vector2 coord((int)((mb->get_position().x - current_tile_region.position.x) / (spacing + size.x)), (int)((mb->get_position().y - current_tile_region.position.y) / (spacing + size.y)));
									if (edited_shape_coord != coord) {
										edited_shape_coord = coord;
										edited_occlusion_shape = tileset->autotile_get_light_occluder(get_current_tile(), edited_shape_coord);
										edited_navigation_shape = tileset->autotile_get_navigation_polygon(get_current_tile(), edited_shape_coord);
										Vector<TileSet::ShapeData> sd = tileset->tile_get_shapes(get_current_tile());
										bool found_collision_shape = false;
										for (int i = 0; i < sd.size(); i++) {
											if (sd[i].autotile_coord == coord) {
												edited_collision_shape = sd[i].shape;
												found_collision_shape = true;
												break;
											}
										}
										if (!found_collision_shape)
											edited_collision_shape = Ref<ConvexPolygonShape2D>(NULL);
										select_coord(edited_shape_coord);
									}
								}
								workspace->update();
							} else if (!mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {
								if (edit_mode == EDITMODE_COLLISION) {
									if (dragging_point >= 0) {
										dragging_point = -1;

										Vector<Vector2> points;

										for (int i = 0; i < current_shape.size(); i++) {
											Vector2 p = current_shape[i];
											if (tools[TOOL_GRID_SNAP]->is_pressed() || tools[SHAPE_KEEP_INSIDE_TILE]->is_pressed()) {
												p = snap_point(p);
											}
											points.push_back(p - shape_anchor);
										}

										edited_collision_shape->set_points(points);

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
									int t_id = get_current_tile();
									if (t_id >= 0) {
										if (edit_mode == EDITMODE_COLLISION) {
											Vector<TileSet::ShapeData> sd = tileset->tile_get_shapes(t_id);
											for (int i = 0; i < sd.size(); i++) {
												if (tileset->tile_get_tile_mode(get_current_tile()) == TileSet::SINGLE_TILE || sd[i].autotile_coord == edited_shape_coord) {
													Ref<ConvexPolygonShape2D> shape = sd[i].shape;

													if (!shape.is_null()) {
														sd.remove(i);
														tileset->tile_set_shapes(get_current_tile(), sd);
														edited_collision_shape = Ref<Shape2D>();
														workspace->update();
													}
													break;
												}
											}
										} else if (edit_mode == EDITMODE_OCCLUSION) {
											if (tileset->tile_get_tile_mode(get_current_tile()) == TileSet::AUTO_TILE || tileset->tile_get_tile_mode(get_current_tile()) == TileSet::ATLAS_TILE) {
												Map<Vector2, Ref<OccluderPolygon2D> > map = tileset->autotile_get_light_oclusion_map(t_id);
												for (Map<Vector2, Ref<OccluderPolygon2D> >::Element *E = map.front(); E; E = E->next()) {
													if (E->key() == edited_shape_coord) {
														tileset->autotile_set_light_occluder(get_current_tile(), Ref<OccluderPolygon2D>(), edited_shape_coord);
														break;
													}
												}
											} else
												tileset->tile_set_light_occluder(t_id, Ref<OccluderPolygon2D>());

											edited_occlusion_shape = Ref<OccluderPolygon2D>();
											workspace->update();
										} else if (edit_mode == EDITMODE_NAVIGATION) {
											if (tileset->tile_get_tile_mode(get_current_tile()) == TileSet::AUTO_TILE || tileset->tile_get_tile_mode(get_current_tile()) == TileSet::ATLAS_TILE) {
												Map<Vector2, Ref<NavigationPolygon> > map = tileset->autotile_get_navigation_map(t_id);
												for (Map<Vector2, Ref<NavigationPolygon> >::Element *E = map.front(); E; E = E->next()) {
													if (E->key() == edited_shape_coord) {
														tileset->autotile_set_navigation_polygon(t_id, Ref<NavigationPolygon>(), edited_shape_coord);
														break;
													}
												}
											} else
												tileset->tile_set_navigation_polygon(t_id, Ref<NavigationPolygon>());
											edited_navigation_shape = Ref<NavigationPolygon>();
											workspace->update();
										}
									}

									creating_shape = true;
									current_shape.resize(0);
									current_shape.push_back(snap_point(pos));
								}
							} else if (mb->is_pressed() && mb->get_button_index() == BUTTON_RIGHT && current_shape.size() > 2) {
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
				default: {}
			}
		}
	}
}

void TileSetEditor::_on_tool_clicked(int p_tool) {
	if (p_tool == BITMASK_COPY) {
		bitmask_map_copy = tileset->autotile_get_bitmask_map(get_current_tile());
	} else if (p_tool == BITMASK_PASTE) {
		tileset->autotile_clear_bitmask_map(get_current_tile());
		for (Map<Vector2, uint16_t>::Element *E = bitmask_map_copy.front(); E; E = E->next()) {
			tileset->autotile_set_bitmask(get_current_tile(), E->key(), E->value());
		}
		workspace->update();
	} else if (p_tool == BITMASK_CLEAR) {
		tileset->autotile_clear_bitmask_map(get_current_tile());
		workspace->update();
	} else if (p_tool == SHAPE_DELETE) {
		if (creating_shape) {
			creating_shape = false;
			current_shape.resize(0);
			workspace->update();
		} else {
			switch (edit_mode) {
				case EDITMODE_REGION: {
					if (workspace_mode == WORKSPACE_EDIT && get_current_tile() >= 0) {
						tileset->remove_tile(get_current_tile());
						workspace->update();
						workspace_overlay->update();
					}
					tool_workspacemode[WORKSPACE_EDIT]->set_pressed(true);
					workspace_mode = WORKSPACE_EDIT;
					update_workspace_tile_mode();
				} break;
				case EDITMODE_COLLISION: {
					if (!edited_collision_shape.is_null()) {
						Vector<TileSet::ShapeData> sd = tileset->tile_get_shapes(get_current_tile());
						int index = -1;
						for (int i = 0; i < sd.size(); i++) {
							if (sd[i].shape == edited_collision_shape) {
								index = i;
								break;
							}
						}
						if (index >= 0) {
							sd.remove(index);
							tileset->tile_set_shapes(get_current_tile(), sd);
							edited_collision_shape = Ref<Shape2D>();
							current_shape.resize(0);
							workspace->update();
						}
					}
				} break;
				case EDITMODE_NAVIGATION: {
					if (!edited_navigation_shape.is_null()) {
						tileset->autotile_set_navigation_polygon(get_current_tile(), Ref<NavigationPolygon>(), edited_shape_coord);
						edited_navigation_shape = Ref<NavigationPolygon>();
						current_shape.resize(0);
						workspace->update();
					}
				} break;
				case EDITMODE_OCCLUSION: {
					if (!edited_occlusion_shape.is_null()) {
						tileset->autotile_set_light_occluder(get_current_tile(), Ref<OccluderPolygon2D>(), edited_shape_coord);
						edited_occlusion_shape = Ref<OccluderPolygon2D>();
						current_shape.resize(0);
						workspace->update();
					}
				} break;
				default: {}
			}
		}
	} else if (p_tool == ZOOM_OUT) {
		_zoom_out();
	} else if (p_tool == ZOOM_1) {
		_reset_zoom();
	} else if (p_tool == ZOOM_IN) {
		_zoom_in();
	} else if (p_tool == TOOL_SELECT) {
		if (creating_shape) {
			// Cancel Creation
			creating_shape = false;
			current_shape.resize(0);
			workspace->update();
		}
	}
}

void TileSetEditor::_on_priority_changed(float val) {
	tileset->autotile_set_subtile_priority(get_current_tile(), edited_shape_coord, (int)val);
	workspace->update();
}

void TileSetEditor::_on_grid_snap_toggled(bool p_val) {
	helper->set_snap_options_visible(p_val);
	workspace->update();
}

void TileSetEditor::_set_snap_step(Vector2 p_val) {
	snap_step.x = CLAMP(p_val.x, 0, 256);
	snap_step.y = CLAMP(p_val.y, 0, 256);
	workspace->update();
}

void TileSetEditor::_set_snap_off(Vector2 p_val) {
	snap_offset.x = CLAMP(p_val.x, 0, 256 + WORKSPACE_MARGIN.x);
	snap_offset.y = CLAMP(p_val.y, 0, 256 + WORKSPACE_MARGIN.y);
	workspace->update();
}

void TileSetEditor::_set_snap_sep(Vector2 p_val) {
	snap_separation.x = CLAMP(p_val.x, 0, 256);
	snap_separation.y = CLAMP(p_val.y, 0, 256);
	workspace->update();
}

void TileSetEditor::_zoom_in() {
	float scale = workspace->get_scale().x;
	if (scale < max_scale) {
		scale *= scale_ratio;
		workspace->set_scale(Vector2(scale, scale));
		workspace_container->set_custom_minimum_size(workspace->get_rect().size * scale);
		workspace_overlay->set_custom_minimum_size(workspace->get_rect().size * scale);
	}
}
void TileSetEditor::_zoom_out() {

	float scale = workspace->get_scale().x;
	if (scale > min_scale) {
		scale /= scale_ratio;
		workspace->set_scale(Vector2(scale, scale));
		workspace_container->set_custom_minimum_size(workspace->get_rect().size * scale);
		workspace_overlay->set_custom_minimum_size(workspace->get_rect().size * scale);
	}
}
void TileSetEditor::_reset_zoom() {
	workspace->set_scale(Vector2(1, 1));
	workspace_container->set_custom_minimum_size(workspace->get_rect().size);
	workspace_overlay->set_custom_minimum_size(workspace->get_rect().size);
}

void TileSetEditor::draw_highlight_current_tile() {

	if (get_current_tile() >= 0) {
		Rect2 region = tileset->tile_get_region(get_current_tile());
		region.position += WORKSPACE_MARGIN;
		workspace->draw_rect(Rect2(0, 0, workspace->get_rect().size.x, region.position.y), Color(0.3, 0.3, 0.3, 0.3));
		workspace->draw_rect(Rect2(0, region.position.y, region.position.x, region.size.y), Color(0.3, 0.3, 0.3, 0.3));
		workspace->draw_rect(Rect2(region.position.x + region.size.x, region.position.y, workspace->get_rect().size.x - region.position.x - region.size.x, region.size.y), Color(0.3, 0.3, 0.3, 0.3));
		workspace->draw_rect(Rect2(0, region.position.y + region.size.y, workspace->get_rect().size.x, workspace->get_rect().size.y - region.size.y - region.position.y), Color(0.3, 0.3, 0.3, 0.3));
	} else {
		workspace->draw_rect(Rect2(Point2(0, 0), workspace->get_rect().size), Color(0.3, 0.3, 0.3, 0.3));
	}
}

void TileSetEditor::draw_highlight_subtile(Vector2 coord, const Vector<Vector2> &other_highlighted) {

	Vector2 size = tileset->autotile_get_size(get_current_tile());
	int spacing = tileset->autotile_get_spacing(get_current_tile());
	Rect2 region = tileset->tile_get_region(get_current_tile());
	coord.x *= (size.x + spacing);
	coord.y *= (size.y + spacing);
	coord += region.position;
	coord += WORKSPACE_MARGIN;
	workspace->draw_rect(Rect2(0, 0, workspace->get_rect().size.x, coord.y), Color(0.3, 0.3, 0.3, 0.3));
	workspace->draw_rect(Rect2(0, coord.y, coord.x, size.y), Color(0.3, 0.3, 0.3, 0.3));
	workspace->draw_rect(Rect2(coord.x + size.x, coord.y, workspace->get_rect().size.x - coord.x - size.x, size.y), Color(0.3, 0.3, 0.3, 0.3));
	workspace->draw_rect(Rect2(0, coord.y + size.y, workspace->get_rect().size.x, workspace->get_rect().size.y - size.y - coord.y), Color(0.3, 0.3, 0.3, 0.3));
	coord += Vector2(1, 1) / workspace->get_scale().x;
	workspace->draw_rect(Rect2(coord, size - Vector2(2, 2) / workspace->get_scale().x), Color(1, 0, 0), false);
	for (int i = 0; i < other_highlighted.size(); i++) {
		coord = other_highlighted[i];
		coord.x *= (size.x + spacing);
		coord.y *= (size.y + spacing);
		coord += region.position;
		coord += WORKSPACE_MARGIN;
		coord += Vector2(1, 1) / workspace->get_scale().x;
		workspace->draw_rect(Rect2(coord, size - Vector2(2, 2) / workspace->get_scale().x), Color(1, 0.5, 0.5), false);
	}
}

void TileSetEditor::draw_tile_subdivision(int p_id, Color p_color) const {
	Color c = p_color;
	if (tileset->tile_get_tile_mode(p_id) == TileSet::AUTO_TILE || tileset->tile_get_tile_mode(p_id) == TileSet::ATLAS_TILE) {
		Rect2 region = tileset->tile_get_region(p_id);
		Size2 size = tileset->autotile_get_size(p_id);
		int spacing = tileset->autotile_get_spacing(p_id);
		float j = 0;
		while (j < region.size.x) {
			j += size.x;
			if (spacing <= 0) {
				workspace->draw_line(region.position + WORKSPACE_MARGIN + Point2(j, 0), region.position + WORKSPACE_MARGIN + Point2(j, region.size.y), c);
			} else {
				workspace->draw_rect(Rect2(region.position + WORKSPACE_MARGIN + Point2(j, 0), Size2(spacing, region.size.y)), c);
			}
			j += spacing;
		}
		j = 0;
		while (j < region.size.y) {
			j += size.y;
			if (spacing <= 0) {
				workspace->draw_line(region.position + WORKSPACE_MARGIN + Point2(0, j), region.position + WORKSPACE_MARGIN + Point2(region.size.x, j), c);
			} else {
				workspace->draw_rect(Rect2(region.position + WORKSPACE_MARGIN + Point2(0, j), Size2(region.size.x, spacing)), c);
			}
			j += spacing;
		}
	}
}

void TileSetEditor::draw_edited_region_subdivision() const {
	Color c = Color(0.347214, 0.722656, 0.617063, 1);
	Rect2 region = edited_region;
	Size2 size;
	int spacing;
	bool draw;
	if (workspace_mode == WORKSPACE_EDIT) {
		int p_id = get_current_tile();
		size = tileset->autotile_get_size(p_id);
		spacing = tileset->autotile_get_spacing(p_id);
		draw = tileset->tile_get_tile_mode(p_id) == TileSet::AUTO_TILE || tileset->tile_get_tile_mode(p_id) == TileSet::ATLAS_TILE;
	} else {
		size = snap_step;
		spacing = snap_separation.x;
		draw = workspace_mode != WORKSPACE_CREATE_SINGLE;
	}
	if (draw) {

		float j = 0;
		while (j < region.size.x) {
			j += size.x;
			if (spacing <= 0) {
				workspace->draw_line(region.position + Point2(j, 0), region.position + Point2(j, region.size.y), c);
			} else {
				workspace->draw_rect(Rect2(region.position + Point2(j, 0), Size2(spacing, region.size.y)), c);
			}
			j += spacing;
		}
		j = 0;
		while (j < region.size.y) {
			j += size.y;
			if (spacing <= 0) {
				workspace->draw_line(region.position + Point2(0, j), region.position + Point2(region.size.x, j), c);
			} else {
				workspace->draw_rect(Rect2(region.position + Point2(0, j), Size2(region.size.x, spacing)), c);
			}
			j += spacing;
		}
	}
}

void TileSetEditor::draw_grid_snap() {
	if (tools[TOOL_GRID_SNAP]->is_pressed()) {
		Color grid_color = Color(0.39, 0, 1, 0.2f);
		Size2 s = workspace->get_size();

		int width_count = (int)(s.width / (snap_step.x + snap_separation.x));
		int height_count = (int)(s.height / (snap_step.y + snap_separation.y));

		if (snap_step.x != 0) {
			int last_p = 0;
			for (int i = 0; i <= width_count; i++) {
				if (i == 0 && snap_offset.x != 0) {
					last_p = snap_offset.x;
				}
				if (snap_separation.x != 0 && i != 0) {
					workspace->draw_rect(Rect2(last_p, 0, snap_separation.x, s.height), grid_color);
					last_p += snap_separation.x;
				} else
					workspace->draw_line(Point2(last_p, 0), Point2(last_p, s.height), grid_color);

				last_p += snap_step.x;
			}
		}

		if (snap_step.y != 0) {
			int last_p = 0;
			for (int i = 0; i <= height_count; i++) {
				if (i == 0 && snap_offset.y != 0) {
					last_p = snap_offset.y;
				}
				if (snap_separation.x != 0 && i != 0) {
					workspace->draw_rect(Rect2(0, last_p, s.width, snap_separation.y), grid_color);
					last_p += snap_separation.y;
				} else
					workspace->draw_line(Point2(0, last_p), Point2(s.width, last_p), grid_color);
				last_p += snap_step.y;
			}
		}
	}
}

void TileSetEditor::draw_polygon_shapes() {

	int t_id = get_current_tile();
	if (t_id < 0)
		return;

	draw_handles = false;

	switch (edit_mode) {
		case EDITMODE_COLLISION: {
			Vector<TileSet::ShapeData> sd = tileset->tile_get_shapes(t_id);
			for (int i = 0; i < sd.size(); i++) {
				Vector2 coord = Vector2(0, 0);
				Vector2 anchor = Vector2(0, 0);
				if (tileset->tile_get_tile_mode(get_current_tile()) == TileSet::AUTO_TILE || tileset->tile_get_tile_mode(get_current_tile()) == TileSet::ATLAS_TILE) {
					coord = sd[i].autotile_coord;
					anchor = tileset->autotile_get_size(t_id);
					anchor.x += tileset->autotile_get_spacing(t_id);
					anchor.y += tileset->autotile_get_spacing(t_id);
					anchor.x *= coord.x;
					anchor.y *= coord.y;
				}
				anchor += WORKSPACE_MARGIN;
				anchor += tileset->tile_get_region(t_id).position;
				Ref<ConvexPolygonShape2D> shape = sd[i].shape;
				if (shape.is_valid()) {
					Color c_bg;
					Color c_border;
					if ((tileset->tile_get_tile_mode(get_current_tile()) == TileSet::SINGLE_TILE || coord == edited_shape_coord) && sd[i].shape == edited_collision_shape) {
						c_bg = Color(0, 1, 1, 0.5);
						c_border = Color(0, 1, 1);
					} else {
						c_bg = Color(0.9, 0.7, 0.07, 0.5);
						c_border = Color(0.9, 0.7, 0.07, 1);
					}
					Vector<Vector2> polygon;
					Vector<Color> colors;
					if (shape == edited_collision_shape && current_shape.size() > 2) {
						for (int j = 0; j < current_shape.size(); j++) {
							polygon.push_back(current_shape[j]);
							colors.push_back(c_bg);
						}
					} else {
						for (int j = 0; j < shape->get_points().size(); j++) {
							polygon.push_back(shape->get_points()[j] + anchor);
							colors.push_back(c_bg);
						}
					}
					if (polygon.size() > 2) {
						workspace->draw_polygon(polygon, colors);
					}
					if (coord == edited_shape_coord || tileset->tile_get_tile_mode(get_current_tile()) == TileSet::SINGLE_TILE) {
						for (int j = 0; j < shape->get_points().size() - 1; j++) {
							workspace->draw_line(shape->get_points()[j] + anchor, shape->get_points()[j + 1] + anchor, c_border, 1, true);
						}

						if (shape == edited_collision_shape) {
							draw_handles = true;
						}
					}
				}
			}
		} break;
		case EDITMODE_OCCLUSION: {
			if (tileset->tile_get_tile_mode(get_current_tile()) == TileSet::SINGLE_TILE) {
				Ref<OccluderPolygon2D> shape = edited_occlusion_shape;
				if (shape.is_valid()) {
					Color c_bg = Color(0, 1, 1, 0.5);
					Color c_border = Color(0, 1, 1);

					Vector<Vector2> polygon;
					Vector<Color> colors;
					Vector2 anchor = WORKSPACE_MARGIN;
					anchor += tileset->tile_get_region(get_current_tile()).position;
					for (int j = 0; j < shape->get_polygon().size(); j++) {
						polygon.push_back(shape->get_polygon()[j] + anchor);
						colors.push_back(c_bg);
					}
					workspace->draw_polygon(polygon, colors);

					for (int j = 0; j < shape->get_polygon().size() - 1; j++) {
						workspace->draw_line(shape->get_polygon()[j] + anchor, shape->get_polygon()[j + 1] + anchor, c_border, 1, true);
					}
					workspace->draw_line(shape->get_polygon()[shape->get_polygon().size() - 1] + anchor, shape->get_polygon()[0] + anchor, c_border, 1, true);
					if (shape == edited_occlusion_shape) {
						draw_handles = true;
					}
				}
			} else {
				Map<Vector2, Ref<OccluderPolygon2D> > map = tileset->autotile_get_light_oclusion_map(t_id);
				for (Map<Vector2, Ref<OccluderPolygon2D> >::Element *E = map.front(); E; E = E->next()) {
					Vector2 coord = E->key();
					Vector2 anchor = tileset->autotile_get_size(t_id);
					anchor.x += tileset->autotile_get_spacing(t_id);
					anchor.y += tileset->autotile_get_spacing(t_id);
					anchor.x *= coord.x;
					anchor.y *= coord.y;
					anchor += WORKSPACE_MARGIN;
					anchor += tileset->tile_get_region(t_id).position;
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
						if (shape == edited_occlusion_shape && current_shape.size() > 2) {
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
								draw_handles = true;
							}
						}
					}
				}
			}
		} break;
		case EDITMODE_NAVIGATION: {
			if (tileset->tile_get_tile_mode(get_current_tile()) == TileSet::SINGLE_TILE) {
				Ref<NavigationPolygon> shape = edited_navigation_shape;

				if (shape.is_valid()) {
					Color c_bg = Color(0, 1, 1, 0.5);
					Color c_border = Color(0, 1, 1);

					Vector<Vector2> polygon;
					Vector<Color> colors;
					Vector2 anchor = WORKSPACE_MARGIN;
					anchor += tileset->tile_get_region(get_current_tile()).position;
					PoolVector<Vector2> vertices = shape->get_vertices();
					for (int j = 0; j < shape->get_polygon(0).size(); j++) {
						polygon.push_back(vertices[shape->get_polygon(0)[j]] + anchor);
						colors.push_back(c_bg);
					}
					workspace->draw_polygon(polygon, colors);

					if (shape->get_polygon_count() > 0) {
						PoolVector<Vector2> vertices = shape->get_vertices();
						for (int j = 0; j < shape->get_polygon(0).size() - 1; j++) {
							workspace->draw_line(vertices[shape->get_polygon(0)[j]] + anchor, vertices[shape->get_polygon(0)[j + 1]] + anchor, c_border, 1, true);
						}
						if (shape == edited_navigation_shape) {
							draw_handles = true;
						}
					}
				}

			} else {
				Map<Vector2, Ref<NavigationPolygon> > map = tileset->autotile_get_navigation_map(t_id);
				for (Map<Vector2, Ref<NavigationPolygon> >::Element *E = map.front(); E; E = E->next()) {
					Vector2 coord = E->key();
					Vector2 anchor = tileset->autotile_get_size(t_id);
					anchor.x += tileset->autotile_get_spacing(t_id);
					anchor.y += tileset->autotile_get_spacing(t_id);
					anchor.x *= coord.x;
					anchor.y *= coord.y;
					anchor += WORKSPACE_MARGIN;
					anchor += tileset->tile_get_region(t_id).position;
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
						if (shape == edited_navigation_shape && current_shape.size() > 2) {
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
									draw_handles = true;
								}
							}
						}
					}
				}
			}
		} break;
		default: {}
	}
	if (creating_shape) {
		for (int j = 0; j < current_shape.size() - 1; j++) {
			workspace->draw_line(current_shape[j], current_shape[j + 1], Color(0, 1, 1), 1, true);
		}
		workspace->draw_line(current_shape[current_shape.size() - 1], snap_point(workspace->get_local_mouse_position()), Color(0, 1, 1), 1, true);
	}
}

void TileSetEditor::close_shape(const Vector2 &shape_anchor) {

	creating_shape = false;

	if (edit_mode == EDITMODE_COLLISION) {
		if (current_shape.size() >= 3) {
			Ref<ConvexPolygonShape2D> shape = memnew(ConvexPolygonShape2D);

			Vector<Vector2> segments;
			float p_total = 0;

			for (int i = 0; i < current_shape.size(); i++) {
				segments.push_back(current_shape[i] - shape_anchor);

				if (i != current_shape.size() - 1)
					p_total += ((current_shape[i + 1].x - current_shape[i].x) * (-current_shape[i + 1].y + (-current_shape[i].y)));
				else
					p_total += ((current_shape[0].x - current_shape[i].x) * (-current_shape[0].y + (-current_shape[i].y)));
			}

			if (p_total < 0)
				segments.invert();

			shape->set_points(segments);

			if (tileset->tile_get_tile_mode(get_current_tile()) == TileSet::AUTO_TILE || tileset->tile_get_tile_mode(get_current_tile()) == TileSet::ATLAS_TILE)
				tileset->tile_add_shape(get_current_tile(), shape, Transform2D(), false, edited_shape_coord);
			else
				tileset->tile_add_shape(get_current_tile(), shape, Transform2D());

			edited_collision_shape = shape;
		}

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

		if (tileset->tile_get_tile_mode(get_current_tile()) == TileSet::AUTO_TILE || tileset->tile_get_tile_mode(get_current_tile()) == TileSet::ATLAS_TILE)
			tileset->autotile_set_light_occluder(get_current_tile(), shape, edited_shape_coord);
		else
			tileset->tile_set_light_occluder(get_current_tile(), shape);
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

		if (tileset->tile_get_tile_mode(get_current_tile()) == TileSet::AUTO_TILE || tileset->tile_get_tile_mode(get_current_tile()) == TileSet::ATLAS_TILE)
			tileset->autotile_set_navigation_polygon(get_current_tile(), shape, edited_shape_coord);
		else
			tileset->tile_set_navigation_polygon(get_current_tile(), shape);
		edited_navigation_shape = shape;
		tools[TOOL_SELECT]->set_pressed(true);
		workspace->update();
	}
	tileset->_change_notify("");
}

void TileSetEditor::select_coord(const Vector2 &coord) {
	current_shape = PoolVector2Array();
	if (get_current_tile() == -1)
		return;
	Rect2 current_tile_region = tileset->tile_get_region(get_current_tile());
	current_tile_region.position += WORKSPACE_MARGIN;
	if (tileset->tile_get_tile_mode(get_current_tile()) == TileSet::SINGLE_TILE) {
		if (edited_collision_shape != tileset->tile_get_shape(get_current_tile(), 0))
			edited_collision_shape = tileset->tile_get_shape(get_current_tile(), 0);
		if (edited_occlusion_shape != tileset->tile_get_light_occluder(get_current_tile()))
			edited_occlusion_shape = tileset->tile_get_light_occluder(get_current_tile());
		if (edited_navigation_shape != tileset->tile_get_navigation_polygon(get_current_tile()))
			edited_navigation_shape = tileset->tile_get_navigation_polygon(get_current_tile());

		if (edit_mode == EDITMODE_COLLISION) {
			current_shape.resize(0);
			if (edited_collision_shape.is_valid()) {
				for (int i = 0; i < edited_collision_shape->get_points().size(); i++) {
					current_shape.push_back(edited_collision_shape->get_points()[i] + current_tile_region.position);
				}
			}
		} else if (edit_mode == EDITMODE_OCCLUSION) {
			current_shape.resize(0);
			if (edited_occlusion_shape.is_valid()) {
				for (int i = 0; i < edited_occlusion_shape->get_polygon().size(); i++) {
					current_shape.push_back(edited_occlusion_shape->get_polygon()[i] + current_tile_region.position);
				}
			}
		} else if (edit_mode == EDITMODE_NAVIGATION) {
			current_shape.resize(0);
			if (edited_navigation_shape.is_valid()) {
				if (edited_navigation_shape->get_polygon_count() > 0) {
					PoolVector<Vector2> vertices = edited_navigation_shape->get_vertices();
					for (int i = 0; i < edited_navigation_shape->get_polygon(0).size(); i++) {
						current_shape.push_back(vertices[edited_navigation_shape->get_polygon(0)[i]] + current_tile_region.position);
					}
				}
			}
		}
	} else {
		int spacing = tileset->autotile_get_spacing(get_current_tile());
		Vector2 size = tileset->autotile_get_size(get_current_tile());
		Vector2 shape_anchor = coord;
		shape_anchor.x *= (size.x + spacing);
		shape_anchor.y *= (size.y + spacing);
		shape_anchor += current_tile_region.position;
		if (edit_mode == EDITMODE_COLLISION) {
			current_shape.resize(0);
			if (edited_collision_shape.is_valid()) {
				for (int j = 0; j < edited_collision_shape->get_points().size(); j++) {
					current_shape.push_back(edited_collision_shape->get_points()[j] + shape_anchor);
				}
			}
		} else if (edit_mode == EDITMODE_OCCLUSION) {
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
	}
	workspace->update();
	workspace_container->update();
	helper->_change_notify("");
}

Vector2 TileSetEditor::snap_point(const Vector2 &point) {
	Vector2 p = point;
	Vector2 coord = edited_shape_coord;
	Vector2 tile_size = tileset->autotile_get_size(get_current_tile());
	int spacing = tileset->autotile_get_spacing(get_current_tile());
	Vector2 anchor = coord;
	anchor.x *= (tile_size.x + spacing);
	anchor.y *= (tile_size.y + spacing);
	anchor += tileset->tile_get_region(get_current_tile()).position;
	anchor += WORKSPACE_MARGIN;
	Rect2 region(anchor, tile_size);
	if (tileset->tile_get_tile_mode(get_current_tile()) == TileSet::SINGLE_TILE) {
		region.position = tileset->tile_get_region(get_current_tile()).position + WORKSPACE_MARGIN;
		region.size = tileset->tile_get_region(get_current_tile()).size;
	}

	if (tools[TOOL_GRID_SNAP]->is_pressed()) {
		p.x = Math::snap_scalar_seperation(snap_offset.x, snap_step.x, p.x, snap_separation.x);
		p.y = Math::snap_scalar_seperation(snap_offset.y, snap_step.y, p.y, snap_separation.y);
	}
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
	return p;
}

void TileSetEditor::update_texture_list() {
	Ref<Texture> selected_texture = get_current_texture();

	helper->set_tileset(tileset);

	List<int> ids;
	tileset->get_tile_list(&ids);
	Vector<int> ids_to_remove;
	for (List<int>::Element *E = ids.front(); E; E = E->next()) {
		// Clear tiles referencing gone textures (user has been already given the chance to fix broken deps)
		if (!tileset->tile_get_texture(E->get()).is_valid()) {
			ids_to_remove.push_back(E->get());
			ERR_CONTINUE(!tileset->tile_get_texture(E->get()).is_valid());
		}

		if (!texture_map.has(tileset->tile_get_texture(E->get())->get_rid())) {
			texture_list->add_item(tileset->tile_get_texture(E->get())->get_path().get_file());
			texture_map.insert(tileset->tile_get_texture(E->get())->get_rid(), tileset->tile_get_texture(E->get()));
			texture_list->set_item_metadata(texture_list->get_item_count() - 1, tileset->tile_get_texture(E->get())->get_rid());
		}
	}
	for (int i = 0; i < ids_to_remove.size(); i++) {
		tileset->remove_tile(ids_to_remove[i]);
	}

	if (texture_list->get_item_count() > 0 && selected_texture.is_valid()) {
		texture_list->select(texture_list->find_metadata(selected_texture->get_rid()));
		if (texture_list->get_selected_items().size() > 0)
			_on_texture_list_selected(texture_list->get_selected_items()[0]);
	} else if (get_current_texture().is_valid()) {
		_on_texture_list_selected(texture_list->find_metadata(get_current_texture()->get_rid()));
	} else {
		_on_texture_list_selected(-1);
	}
	update_texture_list_icon();
	helper->_change_notify("");
}

void TileSetEditor::update_texture_list_icon() {

	for (int current_idx = 0; current_idx < texture_list->get_item_count(); current_idx++) {
		RID rid = texture_list->get_item_metadata(current_idx);
		texture_list->set_item_icon(current_idx, texture_map[rid]);
		texture_list->set_item_icon_region(current_idx, Rect2(0, 0, 150, 100));
	}
	texture_list->update();
}

void TileSetEditor::update_workspace_tile_mode() {

	if (workspace_mode != WORKSPACE_EDIT) {
		for (int i = 0; i < EDITMODE_MAX; i++) {
			tool_editmode[i]->hide();
		}
		tool_editmode[EDITMODE_REGION]->show();
		tool_editmode[EDITMODE_REGION]->set_pressed(true);
		_on_edit_mode_changed(EDITMODE_REGION);
		separator_editmode->show();
		return;
	}

	if (get_current_tile() < 0) {
		for (int i = 0; i < EDITMODE_MAX; i++) {
			tool_editmode[i]->hide();
		}
		for (int i = 0; i < ZOOM_OUT; i++) {
			tools[i]->hide();
		}
		separator_editmode->hide();
		separator_delete->hide();
		separator_grid->hide();
		return;
	}

	for (int i = 0; i < EDITMODE_MAX; i++) {
		tool_editmode[i]->show();
	}
	separator_editmode->show();

	if (tileset->tile_get_tile_mode(get_current_tile()) == TileSet::SINGLE_TILE) {
		if (tool_editmode[EDITMODE_ICON]->is_pressed() || tool_editmode[EDITMODE_PRIORITY]->is_pressed() || tool_editmode[EDITMODE_BITMASK]->is_pressed()) {
			tool_editmode[EDITMODE_COLLISION]->set_pressed(true);
			edit_mode = EDITMODE_COLLISION;
		}
		select_coord(Vector2(0, 0));

		tool_editmode[EDITMODE_ICON]->hide();
		tool_editmode[EDITMODE_BITMASK]->hide();
		tool_editmode[EDITMODE_PRIORITY]->hide();
	} else if (tileset->tile_get_tile_mode(get_current_tile()) == TileSet::AUTO_TILE || tileset->tile_get_tile_mode(get_current_tile()) == TileSet::ATLAS_TILE) {
		if (edit_mode == EDITMODE_ICON)
			select_coord(tileset->autotile_get_icon_coordinate(get_current_tile()));
		else
			select_coord(edited_shape_coord);
	} else if (tileset->tile_get_tile_mode(get_current_tile()) == TileSet::ATLAS_TILE) {
		if (tool_editmode[EDITMODE_PRIORITY]->is_pressed() || tool_editmode[EDITMODE_BITMASK]->is_pressed()) {
			tool_editmode[EDITMODE_COLLISION]->set_pressed(true);
			edit_mode = EDITMODE_COLLISION;
		}
		if (edit_mode == EDITMODE_ICON)
			select_coord(tileset->autotile_get_icon_coordinate(get_current_tile()));
		else
			select_coord(edited_shape_coord);

		tool_editmode[EDITMODE_BITMASK]->hide();
		tool_editmode[EDITMODE_PRIORITY]->hide();
	}
	_on_edit_mode_changed(edit_mode);
}

void TileSetEditor::update_edited_region(const Vector2 &end_point) {
	edited_region = Rect2(region_from, Size2());
	if (tools[TOOL_GRID_SNAP]->is_pressed()) {
		Vector2 grid_coord;
		grid_coord.x = Math::floor((region_from.x - snap_offset.x) / (snap_step.x + snap_separation.x));
		grid_coord.y = Math::floor((region_from.y - snap_offset.y) / (snap_step.y + snap_separation.y));
		grid_coord.x *= (snap_step.x + snap_separation.x);
		grid_coord.y *= (snap_step.y + snap_separation.y);
		grid_coord += snap_offset;
		edited_region.expand_to(grid_coord);
		grid_coord += snap_step;
		edited_region.expand_to(grid_coord);
		grid_coord.x = Math::floor((end_point.x - snap_offset.x) / (snap_step.x + snap_separation.x));
		grid_coord.y = Math::floor((end_point.y - snap_offset.y) / (snap_step.y + snap_separation.y));
		grid_coord.x *= (snap_step.x + snap_separation.x);
		grid_coord.y *= (snap_step.y + snap_separation.y);
		grid_coord += snap_offset;
		edited_region.expand_to(grid_coord);
		grid_coord += snap_step;
		if (grid_coord.x < end_point.x)
			grid_coord.x += snap_separation.x;
		if (grid_coord.y < end_point.y)
			grid_coord.y += snap_separation.y;
		edited_region.expand_to(grid_coord);
	} else {
		edited_region.expand_to(end_point);
	}
}

int TileSetEditor::get_current_tile() const {
	return current_tile;
}

void TileSetEditor::set_current_tile(int p_id) {
	if (current_tile != p_id) {
		current_tile = p_id;
		helper->_change_notify("");
		select_coord(Vector2(0, 0));
		update_workspace_tile_mode();
	}
}

Ref<Texture> TileSetEditor::get_current_texture() {
	if (texture_list->get_selected_items().size() == 0)
		return Ref<Texture>();
	else
		return texture_map[texture_list->get_item_metadata(texture_list->get_selected_items()[0])];
}

void TilesetEditorContext::set_tileset(const Ref<TileSet> &p_tileset) {

	tileset = p_tileset;
}

void TilesetEditorContext::set_snap_options_visible(bool p_visible) {
	snap_options_visible = p_visible;
	_change_notify("");
}

bool TilesetEditorContext::_set(const StringName &p_name, const Variant &p_value) {

	String name = p_name.operator String();

	if (name == "options_offset") {
		Vector2 snap = p_value;
		tileset_editor->_set_snap_off(snap + WORKSPACE_MARGIN);
		return true;
	} else if (name == "options_step") {
		Vector2 snap = p_value;
		tileset_editor->_set_snap_step(snap);
		return true;
	} else if (name == "options_separation") {
		Vector2 snap = p_value;
		tileset_editor->_set_snap_sep(snap);
		return true;
	} else if (p_name.operator String().left(5) == "tile_") {
		String name = p_name.operator String().right(5);
		bool v = false;

		if (tileset_editor->get_current_tile() < 0 || tileset.is_null())
			return false;

		if (name == "autotile_bitmask_mode") {
			tileset->set(String::num(tileset_editor->get_current_tile(), 0) + "/autotile/bitmask_mode", p_value, &v);
		} else if (name == "subtile_size") {
			tileset->set(String::num(tileset_editor->get_current_tile(), 0) + "/autotile/tile_size", p_value, &v);
		} else if (name == "subtile_spacing") {
			tileset->set(String::num(tileset_editor->get_current_tile(), 0) + "/autotile/spacing", p_value, &v);
		} else {
			tileset->set(String::num(tileset_editor->get_current_tile(), 0) + "/" + name, p_value, &v);
		}
		if (v) {
			tileset->_change_notify("");
			tileset_editor->workspace->update();
			tileset_editor->workspace_overlay->update();
		}
		return v;
	} else if (name == "tileset_script") {
		tileset->set_script(p_value);
		return true;
	}

	tileset_editor->err_dialog->set_text(TTR("This property can't be changed."));
	tileset_editor->err_dialog->popup_centered(Size2(300, 60));
	return false;
}

bool TilesetEditorContext::_get(const StringName &p_name, Variant &r_ret) const {

	String name = p_name.operator String();
	bool v = false;

	if (name == "options_offset") {
		r_ret = tileset_editor->snap_offset - WORKSPACE_MARGIN;
		v = true;
	} else if (name == "options_step") {
		r_ret = tileset_editor->snap_step;
		v = true;
	} else if (name == "options_separation") {
		r_ret = tileset_editor->snap_separation;
		v = true;
	} else if (name.left(5) == "tile_") {
		name = name.right(5);

		if (tileset_editor->get_current_tile() < 0 || tileset.is_null())
			return false;
		if (!tileset->has_tile(tileset_editor->get_current_tile()))
			return false;

		if (name == "autotile_bitmask_mode") {
			r_ret = tileset->get(String::num(tileset_editor->get_current_tile(), 0) + "/autotile/bitmask_mode", &v);
		} else if (name == "subtile_size") {
			r_ret = tileset->get(String::num(tileset_editor->get_current_tile(), 0) + "/autotile/tile_size", &v);
		} else if (name == "subtile_spacing") {
			r_ret = tileset->get(String::num(tileset_editor->get_current_tile(), 0) + "/autotile/spacing", &v);
		} else {
			r_ret = tileset->get(String::num(tileset_editor->get_current_tile(), 0) + "/" + name, &v);
		}
		return v;
	} else if (name == "selected_collision") {
		r_ret = tileset_editor->edited_collision_shape;
		v = true;
	} else if (name == "selected_navigation") {
		r_ret = tileset_editor->edited_navigation_shape;
		v = true;
	} else if (name == "selected_occlusion") {
		r_ret = tileset_editor->edited_occlusion_shape;
		v = true;
	} else if (name == "tileset_script") {
		r_ret = tileset->get_script();
		v = true;
	}
	return v;
}

void TilesetEditorContext::_get_property_list(List<PropertyInfo> *p_list) const {

	if (snap_options_visible) {
		p_list->push_back(PropertyInfo(Variant::NIL, "Snap Options", PROPERTY_HINT_NONE, "options_", PROPERTY_USAGE_GROUP));
		p_list->push_back(PropertyInfo(Variant::VECTOR2, "options_offset"));
		p_list->push_back(PropertyInfo(Variant::VECTOR2, "options_step"));
		p_list->push_back(PropertyInfo(Variant::VECTOR2, "options_separation"));
	}
	if (tileset_editor->get_current_tile() >= 0 && !tileset.is_null()) {
		int id = tileset_editor->get_current_tile();
		p_list->push_back(PropertyInfo(Variant::NIL, "Selected Tile", PROPERTY_HINT_NONE, "tile_", PROPERTY_USAGE_GROUP));
		p_list->push_back(PropertyInfo(Variant::STRING, "tile_name"));
		p_list->push_back(PropertyInfo(Variant::OBJECT, "tile_normal_map", PROPERTY_HINT_RESOURCE_TYPE, "Texture"));
		p_list->push_back(PropertyInfo(Variant::VECTOR2, "tile_tex_offset"));
		p_list->push_back(PropertyInfo(Variant::OBJECT, "tile_material", PROPERTY_HINT_RESOURCE_TYPE, "ShaderMaterial"));
		p_list->push_back(PropertyInfo(Variant::COLOR, "tile_modulate"));
		p_list->push_back(PropertyInfo(Variant::INT, "tile_tile_mode", PROPERTY_HINT_ENUM, "SINGLE_TILE,AUTO_TILE,ATLAS_TILE"));
		if (tileset->tile_get_tile_mode(id) == TileSet::AUTO_TILE) {
			p_list->push_back(PropertyInfo(Variant::INT, "tile_autotile_bitmask_mode", PROPERTY_HINT_ENUM, "2X2,3X3 (minimal),3X3"));
			p_list->push_back(PropertyInfo(Variant::VECTOR2, "tile_subtile_size"));
			p_list->push_back(PropertyInfo(Variant::INT, "tile_subtile_spacing", PROPERTY_HINT_RANGE, "0, 256, 1"));
		} else if (tileset->tile_get_tile_mode(id) == TileSet::ATLAS_TILE) {
			p_list->push_back(PropertyInfo(Variant::VECTOR2, "tile_subtile_size"));
			p_list->push_back(PropertyInfo(Variant::INT, "tile_subtile_spacing", PROPERTY_HINT_RANGE, "0, 256, 1"));
		}
		p_list->push_back(PropertyInfo(Variant::VECTOR2, "tile_occluder_offset"));
		p_list->push_back(PropertyInfo(Variant::VECTOR2, "tile_navigation_offset"));
		p_list->push_back(PropertyInfo(Variant::VECTOR2, "tile_shape_offset", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));
		p_list->push_back(PropertyInfo(Variant::VECTOR2, "tile_shape_transform", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));
		p_list->push_back(PropertyInfo(Variant::INT, "tile_z_index", PROPERTY_HINT_RANGE, itos(VS::CANVAS_ITEM_Z_MIN) + "," + itos(VS::CANVAS_ITEM_Z_MAX) + ",1"));
	}
	if (tileset_editor->edit_mode == TileSetEditor::EDITMODE_COLLISION && tileset_editor->edited_collision_shape.is_valid()) {
		p_list->push_back(PropertyInfo(Variant::OBJECT, "selected_collision", PROPERTY_HINT_RESOURCE_TYPE, tileset_editor->edited_collision_shape->get_class()));
	}
	if (tileset_editor->edit_mode == TileSetEditor::EDITMODE_NAVIGATION && tileset_editor->edited_navigation_shape.is_valid()) {
		p_list->push_back(PropertyInfo(Variant::OBJECT, "selected_navigation", PROPERTY_HINT_RESOURCE_TYPE, tileset_editor->edited_navigation_shape->get_class()));
	}
	if (tileset_editor->edit_mode == TileSetEditor::EDITMODE_OCCLUSION && tileset_editor->edited_occlusion_shape.is_valid()) {
		p_list->push_back(PropertyInfo(Variant::OBJECT, "selected_occlusion", PROPERTY_HINT_RESOURCE_TYPE, tileset_editor->edited_occlusion_shape->get_class()));
	}
	if (!tileset.is_null()) {
		p_list->push_back(PropertyInfo(Variant::OBJECT, "tileset_script", PROPERTY_HINT_RESOURCE_TYPE, "Script"));
	}
}

TilesetEditorContext::TilesetEditorContext(TileSetEditor *p_tileset_editor) {
	tileset_editor = p_tileset_editor;
}

void TileSetEditorPlugin::edit(Object *p_node) {

	if (Object::cast_to<TileSet>(p_node)) {
		tileset_editor->edit(Object::cast_to<TileSet>(p_node));
		editor->get_inspector()->edit(tileset_editor->helper);
	}
}

bool TileSetEditorPlugin::handles(Object *p_node) const {

	return p_node->is_class("TileSet") ||
		   p_node->is_class("TilesetEditorContext");
}

void TileSetEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		tileset_editor_button->show();
		editor->make_bottom_panel_item_visible(tileset_editor);
		get_tree()->connect("idle_frame", tileset_editor, "_on_workspace_process");
	} else {
		editor->hide_bottom_panel();
		tileset_editor_button->hide();
		get_tree()->disconnect("idle_frame", tileset_editor, "_on_workspace_process");
	}
}

TileSetEditorPlugin::TileSetEditorPlugin(EditorNode *p_node) {
	editor = p_node;
	tileset_editor = memnew(TileSetEditor(p_node));

	tileset_editor->set_custom_minimum_size(Size2(0, 200) * EDSCALE);
	tileset_editor->hide();

	tileset_editor_button = p_node->add_bottom_panel_item(TTR("Tile Set"), tileset_editor);
	tileset_editor_button->hide();
}
