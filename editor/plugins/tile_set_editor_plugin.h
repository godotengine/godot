/**************************************************************************/
/*  tile_set_editor_plugin.h                                              */
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

#ifndef TILE_SET_EDITOR_PLUGIN_H
#define TILE_SET_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "scene/2d/sprite.h"
#include "scene/resources/concave_polygon_shape_2d.h"
#include "scene/resources/convex_polygon_shape_2d.h"
#include "scene/resources/tile_set.h"

#define WORKSPACE_MARGIN Vector2(10, 10)
class TilesetEditorContext;

class TileSetEditor : public HSplitContainer {
	friend class TileSetEditorPlugin;
	friend class TilesetEditorContext;

	GDCLASS(TileSetEditor, HSplitContainer);

	enum TextureToolButtons {
		TOOL_TILESET_ADD_TEXTURE,
		TOOL_TILESET_REMOVE_TEXTURE,
		TOOL_TILESET_CREATE_SCENE,
		TOOL_TILESET_MERGE_SCENE,
		TOOL_TILESET_MAX
	};

	enum WorkspaceMode {
		WORKSPACE_EDIT,
		WORKSPACE_CREATE_SINGLE,
		WORKSPACE_CREATE_AUTOTILE,
		WORKSPACE_CREATE_ATLAS,
		WORKSPACE_MODE_MAX
	};

	enum EditMode {
		EDITMODE_REGION,
		EDITMODE_COLLISION,
		EDITMODE_OCCLUSION,
		EDITMODE_NAVIGATION,
		EDITMODE_BITMASK,
		EDITMODE_PRIORITY,
		EDITMODE_ICON,
		EDITMODE_Z_INDEX,
		EDITMODE_MAX
	};

	enum TileSetTools {
		SELECT_PREVIOUS,
		SELECT_NEXT,
		TOOL_SELECT,
		BITMASK_COPY,
		BITMASK_PASTE,
		BITMASK_CLEAR,
		SHAPE_NEW_POLYGON,
		SHAPE_NEW_RECTANGLE,
		SHAPE_TOGGLE_TYPE,
		SHAPE_DELETE,
		SHAPE_KEEP_INSIDE_TILE,
		TOOL_GRID_SNAP,
		ZOOM_OUT,
		ZOOM_1,
		ZOOM_IN,
		VISIBLE_INFO,
		TOOL_MAX
	};

	struct SubtileData {
		Array collisions;
		Ref<OccluderPolygon2D> occlusion_shape;
		Ref<NavigationPolygon> navigation_shape;
	};

	Ref<TileSet> tileset;
	TilesetEditorContext *helper;
	EditorNode *editor;
	UndoRedo *undo_redo;

	ConfirmationDialog *cd;
	AcceptDialog *err_dialog;
	EditorFileDialog *texture_dialog;

	ItemList *texture_list;
	int option;
	ToolButton *tileset_toolbar_buttons[TOOL_TILESET_MAX];
	MenuButton *tileset_toolbar_tools;
	Map<String, Ref<Texture>> texture_map;

	bool creating_shape;
	int dragging_point;
	bool tile_names_visible;
	Vector2 region_from;
	Rect2 edited_region;
	bool draw_edited_region;
	Vector2 edited_shape_coord;
	PoolVector2Array current_shape;
	Map<Vector2i, SubtileData> current_tile_data;
	Map<Vector2, uint32_t> bitmask_map_copy;

	Vector2 snap_step;
	Vector2 snap_offset;
	Vector2 snap_separation;

	Ref<Shape2D> edited_collision_shape;
	Ref<OccluderPolygon2D> edited_occlusion_shape;
	Ref<NavigationPolygon> edited_navigation_shape;

	int current_item_index;
	Sprite *preview;
	ScrollContainer *scroll;
	Label *empty_message;
	Control *workspace_container;
	bool draw_handles;
	Control *workspace_overlay;
	Control *workspace;
	Button *tool_workspacemode[WORKSPACE_MODE_MAX];
	Button *tool_editmode[EDITMODE_MAX];
	HSeparator *separator_editmode;
	HBoxContainer *toolbar;
	ToolButton *tools[TOOL_MAX];
	VSeparator *separator_shape_toggle;
	VSeparator *separator_bitmask;
	VSeparator *separator_delete;
	VSeparator *separator_grid;
	SpinBox *spin_priority;
	SpinBox *spin_z_index;
	WorkspaceMode workspace_mode;
	EditMode edit_mode;
	int current_tile;

	float max_scale;
	float min_scale;
	float scale_ratio;

	void update_texture_list();
	void update_texture_list_icon();

	void add_texture(Ref<Texture> p_texture);
	void remove_texture(Ref<Texture> p_texture);

	Ref<Texture> get_current_texture();

	static void _import_node(Node *p_node, Ref<TileSet> p_library);
	static void _import_scene(Node *p_scene, Ref<TileSet> p_library, bool p_merge);
	void _undo_redo_import_scene(Node *p_scene, bool p_merge);

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_from);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	void edit(const Ref<TileSet> &p_tileset);
	static Error update_library_file(Node *p_base_scene, Ref<TileSet> ml, bool p_merge = true);

	TileSetEditor(EditorNode *p_editor);
	~TileSetEditor();

private:
	void _on_tileset_toolbar_button_pressed(int p_index);
	void _on_tileset_toolbar_confirm();
	void _on_texture_list_selected(int p_index);
	void _on_textures_added(const PoolStringArray &p_paths);
	void _on_edit_mode_changed(int p_edit_mode);
	void _on_workspace_mode_changed(int p_workspace_mode);
	void _on_workspace_overlay_draw();
	void _on_workspace_draw();
	void _on_workspace_process();
	void _on_scroll_container_input(const Ref<InputEvent> &p_event);
	void _on_workspace_input(const Ref<InputEvent> &p_ie);
	void _on_tool_clicked(int p_tool);
	void _on_priority_changed(float val);
	void _on_z_index_changed(float val);
	void _on_grid_snap_toggled(bool p_val);
	Vector<Vector2> _get_collision_shape_points(const Ref<Shape2D> &p_shape);
	Vector<Vector2> _get_edited_shape_points();
	void _set_edited_shape_points(const Vector<Vector2> &points);
	void _update_tile_data();
	void _update_toggle_shape_button();
	void _select_next_tile();
	void _select_previous_tile();
	Array _get_tiles_in_current_texture(bool sorted = false);
	bool _sort_tiles(Variant p_a, Variant p_b);
	Vector2 _get_subtiles_count(int p_tile_id);
	void _select_next_subtile();
	void _select_previous_subtile();
	void _select_next_shape();
	void _select_previous_shape();
	void _set_edited_collision_shape(const Ref<Shape2D> &p_shape);
	void _set_snap_step(Vector2 p_val);
	void _set_snap_off(Vector2 p_val);
	void _set_snap_sep(Vector2 p_val);

	void _validate_current_tile_id();
	void _select_edited_shape_coord();
	void _undo_tile_removal(int p_id);

	void _zoom_in();
	void _zoom_out();
	void _zoom_reset();
	void _zoom_on_position(float p_zoom, const Vector2 &p_position);

	void draw_highlight_current_tile();
	void draw_highlight_subtile(Vector2 coord, const Vector<Vector2> &other_highlighted = Vector<Vector2>());
	void draw_tile_subdivision(int p_id, Color p_color) const;
	void draw_edited_region_subdivision() const;
	void draw_grid_snap();
	void draw_polygon_shapes();
	void close_shape(const Vector2 &shape_anchor);
	void select_coord(const Vector2 &coord);
	Vector2 snap_point(const Vector2 &point);
	void update_workspace_tile_mode();
	void update_workspace_minsize();
	void update_edited_region(const Vector2 &end_point);
	int get_grabbed_point(const Vector2 &p_mouse_pos, real_t grab_threshold);
	bool is_within_grabbing_distance_of_first_point(const Vector2 &p_pos, real_t p_grab_threshold);

	int get_current_tile() const;
	void set_current_tile(int p_id);
};

class TilesetEditorContext : public Object {
	friend class TileSetEditor;
	GDCLASS(TilesetEditorContext, Object);

	Ref<TileSet> tileset;
	TileSetEditor *tileset_editor;
	bool snap_options_visible;

public:
	bool _hide_script_from_inspector() { return true; }
	void set_tileset(const Ref<TileSet> &p_tileset);

private:
	void set_snap_options_visible(bool p_visible);

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	static void _bind_methods();

public:
	TilesetEditorContext(TileSetEditor *p_tileset_editor);
};

class TileSetEditorPlugin : public EditorPlugin {
	GDCLASS(TileSetEditorPlugin, EditorPlugin);

	TileSetEditor *tileset_editor;
	Button *tileset_editor_button;
	EditorNode *editor;

public:
	virtual String get_name() const { return "TileSet"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);
	void set_state(const Dictionary &p_state);
	Dictionary get_state() const;

	TileSetEditorPlugin(EditorNode *p_node);
};

#endif // TILE_SET_EDITOR_PLUGIN_H
