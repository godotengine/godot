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

#include "scene/gui/split_container.h"
#include "scene/resources/tile_set/tile_set.h"

#include "editor/editor_node.h"

class TileSet;

class TileSetAtlasSourceEditor : public HBoxContainer {
	GDCLASS(TileSetAtlasSourceEditor, HBoxContainer);

private:
	// -- Proxy object for an atlas source, needed by the inspector --
	class TileSetAtlasSourceProxyObject : public Object {
		GDCLASS(TileSetAtlasSourceProxyObject, Object);

	private:
		Ref<TileSet> tile_set;
		TileSetAtlasSource *tile_set_atlas_source = nullptr;
		int source_id;

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
	class TileProxyObject : public Object {
		GDCLASS(TileProxyObject, Object);

	private:
		TileSetAtlasSourceEditor *tiles_editor_source_tab;

		TileSetAtlasSource *tile_set_atlas_source = nullptr;
		int source_id;
		Vector2i coords;

	protected:
		static void _bind_methods();

	public:
		// Accessors.
		Vector2i get_atlas_coords() const;
		void set_atlas_coords(Vector2i p_atlas_coords);
		Vector2i get_size_in_atlas() const;
		void set_size_in_atlas(Vector2i p_size_in_atlas);

		// Update the proxyed object.
		void edit(TileSetAtlasSource *p_tile_set_atlas_source, int p_source_id = -1, Vector2i p_coords = TileSetAtlasSource::INVALID_ATLAS_COORDS);

		TileProxyObject(TileSetAtlasSourceEditor *p_tiles_editor_source_tab) {
			tiles_editor_source_tab = p_tiles_editor_source_tab;
		}
	};

	// -- Proxy object for a tile, needed by the inspector --
	class AlternativeTileProxyObject : public Object {
		GDCLASS(AlternativeTileProxyObject, Object);

	private:
		TileSetAtlasSourceEditor *tiles_editor_source_tab;

		TileSetAtlasSource *tile_set_atlas_source = nullptr;
		int source_id;
		Vector2i coords;
		int alternative_tile;

	protected:
		static void _bind_methods();

	public:
		// Accessors.
		void set_id(int p_id);
		int get_id();

		// Update the proxyed object.
		void edit(TileSetAtlasSource *p_tile_set_atlas_source, int p_source_id = -1, Vector2i p_coords = TileSetAtlasSource::INVALID_ATLAS_COORDS, int p_alternative_tile = TileSetAtlasSource::INVALID_TILE_ALTERNATIVE);

		AlternativeTileProxyObject(TileSetAtlasSourceEditor *p_tiles_editor_source_tab) {
			tiles_editor_source_tab = p_tiles_editor_source_tab;
		}
	};

	Ref<TileSet> tile_set;
	TileSetAtlasSource *tile_set_atlas_source = nullptr;
	int tile_set_atlas_source_id = -1;

	UndoRedo *undo_redo = EditorNode::get_undo_redo();

	bool tileset_changed_needs_update = false;

	// -- Inspector --
	TileProxyObject *tile_proxy_object;
	Label *atlas_tile_inspector_label;
	EditorInspector *atlas_tile_inspector;

	AlternativeTileProxyObject *alternative_tile_proxy_object;
	Label *alternative_tile_inspector_label;
	EditorInspector *alternative_tile_inspector;

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
	int menu_option_alternative;
	void _menu_option(int p_option);

	// Tool buttons.
	Ref<ButtonGroup> tools_button_group;
	Button *tool_select_button;
	Button *tool_add_remove_button;
	Button *tool_add_remove_rect_button;
	Label *tool_tile_id_label;

	HBoxContainer *tool_settings;
	VSeparator *tool_settings_vsep;
	Button *tools_settings_erase_button;

	MenuButton *tool_advanced_menu_buttom;

	// Selection.
	Vector2i selected_tile = TileSetAtlasSource::INVALID_ATLAS_COORDS;
	int selected_alternative = TileSetAtlasSource::INVALID_TILE_ALTERNATIVE;

	// A control on the tile atlas to draw and handle input events.
	Vector2i hovered_base_tile_coords = TileSetAtlasSource::INVALID_ATLAS_COORDS;

	PopupMenu *base_tile_popup_menu;
	PopupMenu *empty_base_tile_popup_menu;
	Ref<Texture2D> resize_handle;
	Ref<Texture2D> resize_handle_disabled;
	Control *tile_atlas_control;
	void _tile_atlas_control_draw();
	void _tile_atlas_control_mouse_exited();
	void _tile_atlas_control_gui_input(const Ref<InputEvent> &p_event);
	void _tile_atlas_view_transform_changed();

	// A control over the alternative tiles.
	Vector3i hovered_alternative_tile_coords = Vector3i(TileSetAtlasSource::INVALID_ATLAS_COORDS.x, TileSetAtlasSource::INVALID_ATLAS_COORDS.y, TileSetAtlasSource::INVALID_TILE_ALTERNATIVE);

	PopupMenu *alternative_tile_popup_menu;
	Control *alternative_tiles_control;
	void _tile_alternatives_control_draw();
	void _tile_alternatives_control_mouse_exited();
	void _tile_alternatives_control_gui_input(const Ref<InputEvent> &p_event);

	// -- Update functions --
	void _update_tile_id_label();
	void _update_source_inspector();
	void _update_fix_selected_and_hovered_tiles();
	void _update_tile_inspector();
	void _update_alternative_tile_inspector();
	void _update_atlas_view();
	void _update_toolbar();

	// -- input events --
	void _unhandled_key_input(const Ref<InputEvent> &p_event);

	// -- Misc --
	void _auto_create_tiles();
	void _auto_remove_tiles();
	AcceptDialog *confirm_auto_create_tiles;

	void _tile_set_changed();
	void _atlas_source_proxy_object_changed(String p_what);

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
