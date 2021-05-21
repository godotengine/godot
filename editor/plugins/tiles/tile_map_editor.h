/*************************************************************************/
/*  tile_map_editor.h                                                    */
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

#ifndef TILE_MAP_EDITOR_H
#define TILE_MAP_EDITOR_H

#include "tile_atlas_view.h"

#include "core/typedefs.h"
#include "editor/editor_node.h"
#include "scene/2d/tile_map.h"
#include "scene/gui/box_container.h"
#include "scene/gui/tabs.h"

class TileMapEditorPlugin : public VBoxContainer {
public:
	virtual Control *get_toolbar() const {
		return memnew(Control);
	};
	virtual bool forward_canvas_gui_input(const Ref<InputEvent> &p_event) { return false; };
	virtual void forward_canvas_draw_over_viewport(Control *p_overlay){};
	virtual void tile_set_changed(){};
	virtual void edit(ObjectID p_tile_map_id){};
};

class TileMapEditorTilesPlugin : public TileMapEditorPlugin {
	GDCLASS(TileMapEditorTilesPlugin, TileMapEditorPlugin);

private:
	UndoRedo *undo_redo = EditorNode::get_undo_redo();
	ObjectID tile_map_id;
	virtual void edit(ObjectID p_tile_map_id) override;

	///// Toolbar /////
	HBoxContainer *toolbar;

	Ref<ButtonGroup> tool_buttons_group;
	Button *select_tool_button;
	Button *paint_tool_button;
	Button *line_tool_button;
	Button *rect_tool_button;
	Button *bucket_tool_button;
	Button *picker_button;

	HBoxContainer *tools_settings;
	VSeparator *tools_settings_vsep;
	Button *erase_button;
	CheckBox *bucket_continuous_checkbox;

	VSeparator *tools_settings_vsep_2;
	CheckBox *random_tile_checkbox;
	float scattering = 0.0;
	Label *scatter_label;
	SpinBox *scatter_spinbox;
	void _on_random_tile_checkbox_toggled(bool p_pressed);
	void _on_scattering_spinbox_changed(double p_value);

	Button *toggle_grid_button;
	void _on_grid_toggled(bool p_pressed);

	void _update_toolbar();

	///// Tilemap editing. /////
	bool has_mouse = false;
	void _mouse_exited_viewport();

	enum DragType {
		DRAG_TYPE_NONE = 0,
		DRAG_TYPE_SELECT,
		DRAG_TYPE_MOVE,
		DRAG_TYPE_PAINT,
		DRAG_TYPE_LINE,
		DRAG_TYPE_RECT,
		DRAG_TYPE_BUCKET,
		DRAG_TYPE_PICK,
		DRAG_TYPE_CLIPBOARD_PASTE,
	};
	DragType drag_type = DRAG_TYPE_NONE;
	Vector2 drag_start_mouse_pos;
	Vector2 drag_last_mouse_pos;
	Map<Vector2i, TileMapCell> drag_modified;

	TileMapCell _pick_random_tile(const TileMapPattern *p_pattern);
	Map<Vector2i, TileMapCell> _draw_line(Vector2 p_start_drag_mouse_pos, Vector2 p_from_mouse_pos, Vector2 p_to_mouse_pos);
	Map<Vector2i, TileMapCell> _draw_rect(Vector2i p_start_cell, Vector2i p_end_cell);
	Map<Vector2i, TileMapCell> _draw_bucket_fill(Vector2i p_coords, bool p_contiguous);
	void _stop_dragging();

	///// Selection system. /////
	Set<Vector2i> tile_map_selection;
	TileMapPattern *tile_map_clipboard = memnew(TileMapPattern);
	TileMapPattern *selection_pattern = memnew(TileMapPattern);
	void _set_tile_map_selection(const TypedArray<Vector2i> &p_selection);
	TypedArray<Vector2i> _get_tile_map_selection() const;

	Set<TileMapCell> tile_set_selection;

	void _update_selection_pattern_from_tilemap_selection();
	void _update_selection_pattern_from_tileset_selection();
	void _update_tileset_selection_from_selection_pattern();
	void _update_fix_selected_and_hovered();

	///// Bottom panel. ////.
	Label *missing_source_label;
	Label *invalid_source_label;

	ItemList *sources_list;

	Ref<Texture2D> missing_atlas_texture_icon;
	void _update_tile_set_sources_list();

	void _update_bottom_panel();

	// Atlas sources.
	TileMapCell hovered_tile;
	TileAtlasView *tile_atlas_view;
	HSplitContainer *atlas_sources_split_container;

	bool tile_set_dragging_selection = false;
	Vector2i tile_set_drag_start_mouse_pos;

	Control *tile_atlas_control;
	void _tile_atlas_control_mouse_exited();
	void _tile_atlas_control_gui_input(const Ref<InputEvent> &p_event);
	void _tile_atlas_control_draw();

	Control *alternative_tiles_control;
	void _tile_alternatives_control_draw();
	void _tile_alternatives_control_mouse_exited();
	void _tile_alternatives_control_gui_input(const Ref<InputEvent> &p_event);

	void _update_atlas_view();

	// Scenes collection sources.
	ItemList *scene_tiles_list;

	void _update_scenes_collection_view();
	void _scene_thumbnail_done(const String &p_path, const Ref<Texture2D> &p_preview, const Ref<Texture2D> &p_small_preview, Variant p_ud);
	void _scenes_list_multi_selected(int p_index, bool p_selected);
	void _scenes_list_nothing_selected();

	// Update callback
	virtual void tile_set_changed() override;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual Control *get_toolbar() const override;
	virtual bool forward_canvas_gui_input(const Ref<InputEvent> &p_event) override;
	virtual void forward_canvas_draw_over_viewport(Control *p_overlay) override;

	TileMapEditorTilesPlugin();
	~TileMapEditorTilesPlugin();
};

class TileMapEditorTerrainsPlugin : public TileMapEditorPlugin {
	GDCLASS(TileMapEditorTerrainsPlugin, TileMapEditorPlugin);

private:
	UndoRedo *undo_redo = EditorNode::get_undo_redo();
	ObjectID tile_map_id;
	virtual void edit(ObjectID p_tile_map_id) override;

	// Toolbar.
	HBoxContainer *toolbar;

	Ref<ButtonGroup> tool_buttons_group;
	Button *paint_tool_button;

	HBoxContainer *tools_settings;
	VSeparator *tools_settings_vsep;
	Button *picker_button;
	Button *erase_button;

	void _update_toolbar();

	// TileMap editing.
	enum DragType {
		DRAG_TYPE_NONE = 0,
		DRAG_TYPE_PAINT,
		DRAG_TYPE_PICK,
	};
	DragType drag_type = DRAG_TYPE_NONE;
	Vector2 drag_start_mouse_pos;
	Vector2 drag_last_mouse_pos;
	Map<Vector2i, TileMapCell> drag_modified;

	// Painting
	class Constraint {
	private:
		const TileMap *tile_map;
		Vector2i base_cell_coords = Vector2i();
		int bit = -1;
		int terrain = -1;

	public:
		// TODO implement difference operator.
		bool operator<(const Constraint &p_other) const {
			if (base_cell_coords == p_other.base_cell_coords) {
				return bit < p_other.bit;
			}
			return base_cell_coords < p_other.base_cell_coords;
		}

		String to_string() const {
			return vformat("Constraint {pos:%s, bit:%d, terrain:%d}", base_cell_coords, bit, terrain);
		}

		Vector2i get_base_cell_coords() const {
			return base_cell_coords;
		}

		Map<Vector2i, TileSet::CellNeighbor> get_overlapping_coords_and_peering_bits() const;

		void set_terrain(int p_terrain) {
			terrain = p_terrain;
		}

		int get_terrain() const {
			return terrain;
		}

		Constraint(const TileMap *p_tile_map, const Vector2i &p_position, const TileSet::CellNeighbor &p_bit, int p_terrain);
		Constraint() {}
	};

	typedef Array TerrainsTilePattern;

	Set<TerrainsTilePattern> _get_valid_terrains_tile_patterns_for_constraints(int p_terrain_set, const Vector2i &p_position, Set<TileMapEditorTerrainsPlugin::Constraint> p_constraints) const;
	Set<TileMapEditorTerrainsPlugin::Constraint> _get_constraints_from_removed_cells_list(const Set<Vector2i> &p_to_replace, int p_terrain_set) const;
	Set<TileMapEditorTerrainsPlugin::Constraint> _get_constraints_from_added_tile(Vector2i p_position, int p_terrain_set, TerrainsTilePattern p_terrains_tile_pattern) const;
	Map<Vector2i, TerrainsTilePattern> _wave_function_collapse(const Set<Vector2i> &p_to_replace, int p_terrain_set, const Set<TileMapEditorTerrainsPlugin::Constraint> p_constraints) const;
	TileMapCell _get_random_tile_from_pattern(int p_terrain_set, TerrainsTilePattern p_terrain_tile_pattern) const;
	Map<Vector2i, TileMapCell> _draw_terrains(const Map<Vector2i, TerrainsTilePattern> &p_to_paint, int p_terrain_set) const;

	// Cached data.

	TerrainsTilePattern _build_terrains_tile_pattern(TileData *p_tile_data);
	LocalVector<Map<TerrainsTilePattern, Set<TileMapCell>>> per_terrain_terrains_tile_patterns_tiles;
	LocalVector<LocalVector<Set<TerrainsTilePattern>>> per_terrain_terrains_tile_patterns;

	Map<TileMapCell, TileData *> terrain_tiles;
	LocalVector<TileSet::CellNeighbor> tile_sides;

	// Bottom panel.
	Tree *terrains_tree;
	ItemList *terrains_tile_list;

	// Update functions.
	void _update_terrains_cache();
	void _update_terrains_tree();
	void _update_tiles_list();

	// Update callback
	virtual void tile_set_changed() override;

protected:
	void _notification(int p_what);
	//	static void _bind_methods();

public:
	virtual Control *get_toolbar() const override;
	virtual bool forward_canvas_gui_input(const Ref<InputEvent> &p_event) override;
	//virtual void forward_canvas_draw_over_viewport(Control *p_overlay) override;

	TileMapEditorTerrainsPlugin();
	~TileMapEditorTerrainsPlugin();
};

class TileMapEditor : public VBoxContainer {
	GDCLASS(TileMapEditor, VBoxContainer);

private:
	bool tileset_changed_needs_update = false;
	ObjectID tile_map_id;

	// Vector to keep plugins.
	Vector<TileMapEditorPlugin *> tile_map_editor_plugins;

	// Toolbar.
	HBoxContainer *tilemap_toolbar;

	// Bottom panel
	Label *missing_tileset_label;
	Tabs *tabs;
	void _update_bottom_panel();

	// TileMap
	Ref<Texture2D> missing_tile_texture;
	Ref<Texture2D> warning_pattern_texture;

	// CallBack
	void _tile_map_changed();
	void _tab_changed(int p_tab_changed);

protected:
	void _notification(int p_what);
	void _draw_shape(Control *p_control, Rect2 p_region, TileSet::TileShape p_shape, TileSet::TileOffsetAxis p_offset_axis, Color p_color);

public:
	bool forward_canvas_gui_input(const Ref<InputEvent> &p_event);
	void forward_canvas_draw_over_viewport(Control *p_overlay);

	void edit(TileMap *p_tile_map);
	Control *get_toolbar() { return tilemap_toolbar; };

	TileMapEditor();
	~TileMapEditor();

	// Static functions.
	static Vector<Vector2i> get_line(TileMap *p_tile_map, Vector2i p_from_cell, Vector2i p_to_cell);
};

#endif // TILE_MAP_EDITOR_PLUGIN_H
