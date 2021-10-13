/*************************************************************************/
/*  tile_set_atlas_source_editor.h                                       */
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

#ifndef TILE_SET_ATLAS_SOURCE_EDITOR_H
#define TILE_SET_ATLAS_SOURCE_EDITOR_H

#include "tile_atlas_view.h"
#include "tile_data_editors.h"

#include "editor/editor_node.h"
#include "scene/gui/split_container.h"
#include "scene/resources/tile_set.h"

class TileSet;

class TileSetAtlasSourceEditor : public HBoxContainer {
	GDCLASS(TileSetAtlasSourceEditor, HBoxContainer);

private:
	// A class to store which tiles are selected.
	struct TileSelection {
		Vector2i tile = TileSetSource::INVALID_ATLAS_COORDS;
		int alternative = TileSetSource::INVALID_TILE_ALTERNATIVE;

		bool operator<(const TileSelection &p_other) const {
			if (tile == p_other.tile) {
				return alternative < p_other.alternative;
			} else {
				return tile < p_other.tile;
			}
		}
	};

	// -- Proxy object for an atlas source, needed by the inspector --
	class TileSetAtlasSourceProxyObject : public Object {
		GDCLASS(TileSetAtlasSourceProxyObject, Object);

	private:
		Ref<TileSet> tile_set;
		TileSetAtlasSource *tile_set_atlas_source = nullptr;
		int source_id = TileSet::INVALID_SOURCE;

	protected:
		bool _set(const StringName &p_name, const Variant &p_value);
		bool _get(const StringName &p_name, Variant &r_ret) const;
		void _get_property_list(List<PropertyInfo> *p_list) const;
		static void _bind_methods();

	public:
		void set_id(int p_id);
		int get_id();

		void edit(Ref<TileSet> p_tile_set, TileSetAtlasSource *p_tile_set_atlas_source, int p_source_id);
	};

	// -- Proxy object for a tile, needed by the inspector --
	class AtlasTileProxyObject : public Object {
		GDCLASS(AtlasTileProxyObject, Object);

	private:
		TileSetAtlasSourceEditor *tiles_set_atlas_source_editor;

		TileSetAtlasSource *tile_set_atlas_source = nullptr;
		Set<TileSelection> tiles = Set<TileSelection>();

	protected:
		bool _set(const StringName &p_name, const Variant &p_value);
		bool _get(const StringName &p_name, Variant &r_ret) const;
		void _get_property_list(List<PropertyInfo> *p_list) const;

		static void _bind_methods();

	public:
		// Update the proxyed object.
		void edit(TileSetAtlasSource *p_tile_set_atlas_source, Set<TileSelection> p_tiles = Set<TileSelection>());

		AtlasTileProxyObject(TileSetAtlasSourceEditor *p_tiles_set_atlas_source_editor) {
			tiles_set_atlas_source_editor = p_tiles_set_atlas_source_editor;
		}
	};

	Ref<TileSet> tile_set;
	TileSetAtlasSource *tile_set_atlas_source = nullptr;
	int tile_set_atlas_source_id = TileSet::INVALID_SOURCE;

	UndoRedo *undo_redo = EditorNode::get_undo_redo();

	bool tile_set_changed_needs_update = false;

	// -- Properties painting --
	VBoxContainer *tile_data_painting_editor_container;
	Label *tile_data_editors_label;
	Button *tile_data_editor_dropdown_button;
	Popup *tile_data_editors_popup;
	Tree *tile_data_editors_tree;
	void _tile_data_editor_dropdown_button_draw();
	void _tile_data_editor_dropdown_button_pressed();

	// -- Tile data editors --
	String current_property;
	Control *current_tile_data_editor_toolbar = nullptr;
	Map<String, TileDataEditor *> tile_data_editors;
	TileDataEditor *current_tile_data_editor = nullptr;
	void _tile_data_editors_tree_selected();

	// -- Inspector --
	AtlasTileProxyObject *tile_proxy_object;
	Label *tile_inspector_label;
	EditorInspector *tile_inspector;
	Label *tile_inspector_no_tile_selected_label;
	String selected_property;
	void _inspector_property_selected(String p_property);

	TileSetAtlasSourceProxyObject *atlas_source_proxy_object;
	Label *atlas_source_inspector_label;
	EditorInspector *atlas_source_inspector;

	// -- Atlas view --
	HBoxContainer *toolbox;
	Label *tile_atlas_view_missing_source_label;
	TileAtlasView *tile_atlas_view;

	// Dragging
	enum DragType {
		DRAG_TYPE_NONE = 0,
		DRAG_TYPE_CREATE_TILES,
		DRAG_TYPE_CREATE_TILES_USING_RECT,
		DRAG_TYPE_CREATE_BIG_TILE,
		DRAG_TYPE_REMOVE_TILES,
		DRAG_TYPE_REMOVE_TILES_USING_RECT,

		DRAG_TYPE_MOVE_TILE,

		DRAG_TYPE_RECT_SELECT,

		DRAG_TYPE_MAY_POPUP_MENU,

		// Warning: keep in this order.
		DRAG_TYPE_RESIZE_TOP_LEFT,
		DRAG_TYPE_RESIZE_TOP,
		DRAG_TYPE_RESIZE_TOP_RIGHT,
		DRAG_TYPE_RESIZE_RIGHT,
		DRAG_TYPE_RESIZE_BOTTOM_RIGHT,
		DRAG_TYPE_RESIZE_BOTTOM,
		DRAG_TYPE_RESIZE_BOTTOM_LEFT,
		DRAG_TYPE_RESIZE_LEFT,
	};
	DragType drag_type = DRAG_TYPE_NONE;
	Vector2i drag_start_mouse_pos;
	Vector2i drag_last_mouse_pos;
	Vector2i drag_current_tile;

	Rect2i drag_start_tile_shape;
	Set<Vector2i> drag_modified_tiles;
	void _end_dragging();

	Map<Vector2i, List<const PropertyInfo *>> _group_properties_per_tiles(const List<PropertyInfo> &r_list, const TileSetAtlasSource *p_atlas);

	// Popup functions.
	enum MenuOptions {
		TILE_CREATE,
		TILE_CREATE_ALTERNATIVE,
		TILE_DELETE,

		ADVANCED_CLEANUP_TILES_OUTSIDE_TEXTURE,
		ADVANCED_AUTO_CREATE_TILES,
		ADVANCED_AUTO_REMOVE_TILES,
	};
	Vector2i menu_option_coords;
	int menu_option_alternative = TileSetSource::INVALID_TILE_ALTERNATIVE;
	void _menu_option(int p_option);

	// Tool buttons.
	Ref<ButtonGroup> tools_button_group;
	Button *tool_setup_atlas_source_button;
	Button *tool_select_button;
	Button *tool_paint_button;
	Label *tool_tile_id_label;

	// Tool settings.
	HBoxContainer *tool_settings;
	VSeparator *tool_settings_vsep;
	HBoxContainer *tool_settings_tile_data_toolbar_container;
	Button *tools_settings_erase_button;
	MenuButton *tool_advanced_menu_buttom;

	// Selection.
	Set<TileSelection> selection;

	void _set_selection_from_array(Array p_selection);
	Array _get_selection_as_array();

	// A control on the tile atlas to draw and handle input events.
	Vector2i hovered_base_tile_coords = TileSetSource::INVALID_ATLAS_COORDS;

	PopupMenu *base_tile_popup_menu;
	PopupMenu *empty_base_tile_popup_menu;
	Ref<Texture2D> resize_handle;
	Ref<Texture2D> resize_handle_disabled;
	Control *tile_atlas_control;
	Control *tile_atlas_control_unscaled;
	void _tile_atlas_control_draw();
	void _tile_atlas_control_unscaled_draw();
	void _tile_atlas_control_mouse_exited();
	void _tile_atlas_control_gui_input(const Ref<InputEvent> &p_event);
	void _tile_atlas_view_transform_changed();

	// A control over the alternative tiles.
	Vector3i hovered_alternative_tile_coords = Vector3i(TileSetSource::INVALID_ATLAS_COORDS.x, TileSetSource::INVALID_ATLAS_COORDS.y, TileSetSource::INVALID_TILE_ALTERNATIVE);

	PopupMenu *alternative_tile_popup_menu;
	Control *alternative_tiles_control;
	Control *alternative_tiles_control_unscaled;
	void _tile_alternatives_control_draw();
	void _tile_alternatives_control_unscaled_draw();
	void _tile_alternatives_control_mouse_exited();
	void _tile_alternatives_control_gui_input(const Ref<InputEvent> &p_event);

	// -- Update functions --
	void _update_tile_id_label();
	void _update_source_inspector();
	void _update_fix_selected_and_hovered_tiles();
	void _update_atlas_source_inspector();
	void _update_tile_inspector();
	void _update_tile_data_editors();
	void _update_current_tile_data_editor();
	void _update_manage_tile_properties_button();
	void _update_atlas_view();
	void _update_toolbar();

	// -- input events --
	void _unhandled_key_input(const Ref<InputEvent> &p_event);

	// -- Misc --
	void _auto_create_tiles();
	void _auto_remove_tiles();
	AcceptDialog *confirm_auto_create_tiles;

	void _tile_set_changed();
	void _tile_proxy_object_changed(String p_what);
	void _atlas_source_proxy_object_changed(String p_what);

	void _undo_redo_inspector_callback(Object *p_undo_redo, Object *p_edited, String p_property, Variant p_new_value);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void edit(Ref<TileSet> p_tile_set, TileSetAtlasSource *p_tile_set_source, int p_source_id);
	void init_source();

	TileSetAtlasSourceEditor();
	~TileSetAtlasSourceEditor();
};

#endif // TILE_SET_ATLAS_SOURCE_EDITOR_H
