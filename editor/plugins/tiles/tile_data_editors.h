/*************************************************************************/
/*  tile_data_editors.h                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef TILE_DATA_EDITORS_H
#define TILE_DATA_EDITORS_H

#include "tile_atlas_view.h"

#include "editor/editor_node.h"
#include "editor/editor_properties.h"

#include "scene/2d/tile_map.h"
#include "scene/gui/box_container.h"
#include "scene/gui/control.h"
#include "scene/gui/label.h"

class TileDataEditor : public VBoxContainer {
	GDCLASS(TileDataEditor, VBoxContainer);

private:
	bool _tile_set_changed_update_needed = false;
	void _tile_set_changed_plan_update();
	void _tile_set_changed_deferred_update();

protected:
	Ref<TileSet> tile_set;
	TileData *_get_tile_data(TileMapCell p_cell);
	virtual void _tile_set_changed(){};

	static void _bind_methods();

public:
	void set_tile_set(Ref<TileSet> p_tile_set);

	// Input to handle painting.
	virtual Control *get_toolbar() { return nullptr; };
	virtual void forward_draw_over_atlas(TileAtlasView *p_tile_atlas_view, TileSetAtlasSource *p_tile_atlas_source, CanvasItem *p_canvas_item, Transform2D p_transform){};
	virtual void forward_draw_over_alternatives(TileAtlasView *p_tile_atlas_view, TileSetAtlasSource *p_tile_atlas_source, CanvasItem *p_canvas_item, Transform2D p_transform){};
	virtual void forward_painting_atlas_gui_input(TileAtlasView *p_tile_atlas_view, TileSetAtlasSource *p_tile_atlas_source, const Ref<InputEvent> &p_event){};
	virtual void forward_painting_alternatives_gui_input(TileAtlasView *p_tile_atlas_view, TileSetAtlasSource *p_tile_atlas_source, const Ref<InputEvent> &p_event){};

	// Used to draw the tile data property value over a tile.
	virtual void draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileMapCell p_cell, bool p_selected = false){};
};

class DummyObject : public Object {
	GDCLASS(DummyObject, Object)
private:
	Map<String, Variant> properties;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;

public:
	bool has_dummy_property(StringName p_name);
	void add_dummy_property(StringName p_name);
	void remove_dummy_property(StringName p_name);
	void clear_dummy_properties();
};

class GenericTilePolygonEditor : public VBoxContainer {
	GDCLASS(GenericTilePolygonEditor, VBoxContainer);

private:
	Ref<TileSet> tile_set;
	LocalVector<Vector<Point2>> polygons;
	bool multiple_polygon_mode = false;

	bool use_undo_redo = true;
	UndoRedo *editor_undo_redo = EditorNode::get_undo_redo();

	// UI
	int hovered_polygon_index = -1;
	int hovered_point_index = -1;
	int hovered_segment_index = -1;
	Vector2 hovered_segment_point;

	enum DragType {
		DRAG_TYPE_NONE,
		DRAG_TYPE_DRAG_POINT,
		DRAG_TYPE_CREATE_POINT,
		DRAG_TYPE_PAN,
	};
	DragType drag_type = DRAG_TYPE_NONE;
	int drag_polygon_index;
	int drag_point_index;
	Vector2 drag_last_pos;
	PackedVector2Array drag_old_polygon;

	HBoxContainer *toolbar;
	Ref<ButtonGroup> tools_button_group;
	Button *button_create;
	Button *button_edit;
	Button *button_delete;
	Button *button_pixel_snap;
	MenuButton *button_advanced_menu;

	Vector<Point2> in_creation_polygon;

	Panel *panel;
	Control *base_control;
	EditorZoomWidget *editor_zoom_widget;
	Button *button_center_view;
	Vector2 panning;

	Ref<Texture2D> background_texture;
	Rect2 background_region;
	Vector2 background_offset;
	bool background_h_flip;
	bool background_v_flip;
	bool background_transpose;
	Color background_modulate;

	Color polygon_color = Color(1.0, 0.0, 0.0);

	enum AdvancedMenuOption {
		RESET_TO_DEFAULT_TILE,
		CLEAR_TILE,
		ROTATE_RIGHT,
		ROTATE_LEFT,
		FLIP_HORIZONTALLY,
		FLIP_VERTICALLY,
	};

	void _base_control_draw();
	void _zoom_changed();
	void _advanced_menu_item_pressed(int p_item_pressed);
	void _center_view();
	void _base_control_gui_input(Ref<InputEvent> p_event);

	void _snap_to_tile_shape(Point2 &r_point, float &r_current_snapped_dist, float p_snap_dist);
	void _snap_to_half_pixel(Point2 &r_point);
	void _grab_polygon_point(Vector2 p_pos, const Transform2D &p_polygon_xform, int &r_polygon_index, int &r_point_index);
	void _grab_polygon_segment_point(Vector2 p_pos, const Transform2D &p_polygon_xform, int &r_polygon_index, int &r_segment_index, Vector2 &r_point);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_use_undo_redo(bool p_use_undo_redo);

	void set_tile_set(Ref<TileSet> p_tile_set);
	void set_background(Ref<Texture2D> p_texture, Rect2 p_region = Rect2(), Vector2 p_offset = Vector2(), bool p_flip_h = false, bool p_flip_v = false, bool p_transpose = false, Color p_modulate = Color(1.0, 1.0, 1.0, 0.0));

	int get_polygon_count();
	int add_polygon(Vector<Point2> p_polygon, int p_index = -1);
	void remove_polygon(int p_index);
	void clear_polygons();
	void set_polygon(int p_polygon_index, Vector<Point2> p_polygon);
	Vector<Point2> get_polygon(int p_polygon_index);

	void set_polygons_color(Color p_color);
	void set_multiple_polygon_mode(bool p_multiple_polygon_mode);

	GenericTilePolygonEditor();
};

class TileDataDefaultEditor : public TileDataEditor {
	GDCLASS(TileDataDefaultEditor, TileDataEditor);

private:
	// Toolbar
	HBoxContainer *toolbar = memnew(HBoxContainer);
	Button *picker_button;

	// UI
	Ref<Texture2D> tile_bool_checked;
	Ref<Texture2D> tile_bool_unchecked;
	Label *label;

	EditorProperty *property_editor = nullptr;

	// Painting state.
	enum DragType {
		DRAG_TYPE_NONE = 0,
		DRAG_TYPE_PAINT,
		DRAG_TYPE_PAINT_RECT,
	};
	DragType drag_type = DRAG_TYPE_NONE;
	Vector2 drag_start_pos;
	Vector2 drag_last_pos;
	Map<TileMapCell, Variant> drag_modified;
	Variant drag_painted_value;

	void _property_value_changed(StringName p_property, Variant p_value, StringName p_field);

protected:
	DummyObject *dummy_object = memnew(DummyObject);

	UndoRedo *undo_redo = EditorNode::get_undo_redo();

	StringName type;
	String property;
	void _notification(int p_what);

	virtual Variant _get_painted_value();
	virtual void _set_painted_value(TileSetAtlasSource *p_tile_set_atlas_source, Vector2 p_coords, int p_alternative_tile);
	virtual void _set_value(TileSetAtlasSource *p_tile_set_atlas_source, Vector2 p_coords, int p_alternative_tile, Variant p_value);
	virtual Variant _get_value(TileSetAtlasSource *p_tile_set_atlas_source, Vector2 p_coords, int p_alternative_tile);
	virtual void _setup_undo_redo_action(TileSetAtlasSource *p_tile_set_atlas_source, Map<TileMapCell, Variant> p_previous_values, Variant p_new_value);

public:
	virtual Control *get_toolbar() override { return toolbar; };
	virtual void forward_draw_over_atlas(TileAtlasView *p_tile_atlas_view, TileSetAtlasSource *p_tile_atlas_source, CanvasItem *p_canvas_item, Transform2D p_transform) override;
	virtual void forward_draw_over_alternatives(TileAtlasView *p_tile_atlas_view, TileSetAtlasSource *p_tile_atlas_source, CanvasItem *p_canvas_item, Transform2D p_transform) override;
	virtual void forward_painting_atlas_gui_input(TileAtlasView *p_tile_atlas_view, TileSetAtlasSource *p_tile_atlas_source, const Ref<InputEvent> &p_event) override;
	virtual void forward_painting_alternatives_gui_input(TileAtlasView *p_tile_atlas_view, TileSetAtlasSource *p_tile_atlas_source, const Ref<InputEvent> &p_event) override;
	virtual void draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileMapCell p_cell, bool p_selected = false) override;

	void setup_property_editor(Variant::Type p_type, String p_property, String p_label = "", Variant p_default_value = Variant());

	TileDataDefaultEditor();
	~TileDataDefaultEditor();
};

class TileDataTextureOffsetEditor : public TileDataDefaultEditor {
	GDCLASS(TileDataTextureOffsetEditor, TileDataDefaultEditor);

public:
	virtual void draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileMapCell p_cell, bool p_selected = false) override;
};

class TileDataPositionEditor : public TileDataDefaultEditor {
	GDCLASS(TileDataPositionEditor, TileDataDefaultEditor);

public:
	virtual void draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileMapCell p_cell, bool p_selected = false) override;
};

class TileDataYSortEditor : public TileDataDefaultEditor {
	GDCLASS(TileDataYSortEditor, TileDataDefaultEditor);

public:
	virtual void draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileMapCell p_cell, bool p_selected = false) override;
};

class TileDataOcclusionShapeEditor : public TileDataDefaultEditor {
	GDCLASS(TileDataOcclusionShapeEditor, TileDataDefaultEditor);

private:
	int occlusion_layer = -1;

	// UI
	GenericTilePolygonEditor *polygon_editor;

	void _polygon_changed(PackedVector2Array p_polygon);

	virtual Variant _get_painted_value() override;
	virtual void _set_painted_value(TileSetAtlasSource *p_tile_set_atlas_source, Vector2 p_coords, int p_alternative_tile) override;
	virtual void _set_value(TileSetAtlasSource *p_tile_set_atlas_source, Vector2 p_coords, int p_alternative_tile, Variant p_value) override;
	virtual Variant _get_value(TileSetAtlasSource *p_tile_set_atlas_source, Vector2 p_coords, int p_alternative_tile) override;
	virtual void _setup_undo_redo_action(TileSetAtlasSource *p_tile_set_atlas_source, Map<TileMapCell, Variant> p_previous_values, Variant p_new_value) override;

protected:
	UndoRedo *undo_redo = EditorNode::get_undo_redo();

	virtual void _tile_set_changed() override;

	void _notification(int p_what);

public:
	virtual void draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileMapCell p_cell, bool p_selected = false) override;

	void set_occlusion_layer(int p_occlusion_layer) { occlusion_layer = p_occlusion_layer; }

	TileDataOcclusionShapeEditor();
};

class TileDataCollisionEditor : public TileDataDefaultEditor {
	GDCLASS(TileDataCollisionEditor, TileDataDefaultEditor);

	int physics_layer = -1;

	// UI
	GenericTilePolygonEditor *polygon_editor;
	DummyObject *dummy_object = memnew(DummyObject);
	Map<StringName, EditorProperty *> property_editors;

	void _property_value_changed(StringName p_property, Variant p_value, StringName p_field);
	void _property_selected(StringName p_path, int p_focusable);
	void _polygons_changed();

	virtual Variant _get_painted_value() override;
	virtual void _set_painted_value(TileSetAtlasSource *p_tile_set_atlas_source, Vector2 p_coords, int p_alternative_tile) override;
	virtual void _set_value(TileSetAtlasSource *p_tile_set_atlas_source, Vector2 p_coords, int p_alternative_tile, Variant p_value) override;
	virtual Variant _get_value(TileSetAtlasSource *p_tile_set_atlas_source, Vector2 p_coords, int p_alternative_tile) override;
	virtual void _setup_undo_redo_action(TileSetAtlasSource *p_tile_set_atlas_source, Map<TileMapCell, Variant> p_previous_values, Variant p_new_value) override;

protected:
	UndoRedo *undo_redo = EditorNode::get_undo_redo();

	virtual void _tile_set_changed() override;

	void _notification(int p_what);

public:
	virtual void draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileMapCell p_cell, bool p_selected = false) override;

	void set_physics_layer(int p_physics_layer) { physics_layer = p_physics_layer; }

	TileDataCollisionEditor();
	~TileDataCollisionEditor();
};

class TileDataTerrainsEditor : public TileDataEditor {
	GDCLASS(TileDataTerrainsEditor, TileDataEditor);

private:
	// Toolbar
	HBoxContainer *toolbar = memnew(HBoxContainer);
	Button *picker_button;

	// Painting state.
	enum DragType {
		DRAG_TYPE_NONE = 0,
		DRAG_TYPE_PAINT_TERRAIN_SET,
		DRAG_TYPE_PAINT_TERRAIN_SET_RECT,
		DRAG_TYPE_PAINT_TERRAIN_BITS,
		DRAG_TYPE_PAINT_TERRAIN_BITS_RECT,
	};
	DragType drag_type = DRAG_TYPE_NONE;
	Vector2 drag_start_pos;
	Vector2 drag_last_pos;
	Map<TileMapCell, Variant> drag_modified;
	Variant drag_painted_value;

	// UI
	Label *label;
	DummyObject *dummy_object = memnew(DummyObject);
	EditorPropertyEnum *terrain_set_property_editor = nullptr;
	EditorPropertyEnum *terrain_property_editor = nullptr;

	void _property_value_changed(StringName p_property, Variant p_value, StringName p_field);

	void _update_terrain_selector();

protected:
	virtual void _tile_set_changed() override;

	void _notification(int p_what);

	UndoRedo *undo_redo = EditorNode::get_undo_redo();

public:
	virtual Control *get_toolbar() override { return toolbar; };
	virtual void forward_draw_over_atlas(TileAtlasView *p_tile_atlas_view, TileSetAtlasSource *p_tile_atlas_source, CanvasItem *p_canvas_item, Transform2D p_transform) override;
	virtual void forward_draw_over_alternatives(TileAtlasView *p_tile_atlas_view, TileSetAtlasSource *p_tile_atlas_source, CanvasItem *p_canvas_item, Transform2D p_transform) override;
	virtual void forward_painting_atlas_gui_input(TileAtlasView *p_tile_atlas_view, TileSetAtlasSource *p_tile_atlas_source, const Ref<InputEvent> &p_event) override;
	virtual void forward_painting_alternatives_gui_input(TileAtlasView *p_tile_atlas_view, TileSetAtlasSource *p_tile_atlas_source, const Ref<InputEvent> &p_event) override;
	virtual void draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileMapCell p_cell, bool p_selected = false) override;

	TileDataTerrainsEditor();
	~TileDataTerrainsEditor();
};

class TileDataNavigationEditor : public TileDataDefaultEditor {
	GDCLASS(TileDataNavigationEditor, TileDataDefaultEditor);

private:
	int navigation_layer = -1;
	PackedVector2Array navigation_polygon;

	// UI
	GenericTilePolygonEditor *polygon_editor;

	void _polygon_changed(PackedVector2Array p_polygon);

	virtual Variant _get_painted_value() override;
	virtual void _set_painted_value(TileSetAtlasSource *p_tile_set_atlas_source, Vector2 p_coords, int p_alternative_tile) override;
	virtual void _set_value(TileSetAtlasSource *p_tile_set_atlas_source, Vector2 p_coords, int p_alternative_tile, Variant p_value) override;
	virtual Variant _get_value(TileSetAtlasSource *p_tile_set_atlas_source, Vector2 p_coords, int p_alternative_tile) override;
	virtual void _setup_undo_redo_action(TileSetAtlasSource *p_tile_set_atlas_source, Map<TileMapCell, Variant> p_previous_values, Variant p_new_value) override;

protected:
	UndoRedo *undo_redo = EditorNode::get_undo_redo();

	virtual void _tile_set_changed() override;

	void _notification(int p_what);

public:
	virtual void draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileMapCell p_cell, bool p_selected = false) override;

	void set_navigation_layer(int p_navigation_layer) { navigation_layer = p_navigation_layer; }

	TileDataNavigationEditor();
};

#endif // TILE_DATA_EDITORS_H
