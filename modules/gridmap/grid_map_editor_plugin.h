/*************************************************************************/
/*  grid_map_editor_plugin.h                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef GRID_MAP_EDITOR_PLUGIN_H
#define GRID_MAP_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "editor/pane_drag.h"
#include "grid_map.h"

class SpatialEditorPlugin;

class GridMapEditor : public VBoxContainer {
	GDCLASS(GridMapEditor, VBoxContainer);

	enum {

		GRID_CURSOR_SIZE = 50
	};

	enum InputAction {

		INPUT_NONE,
		INPUT_PAINT,
		INPUT_ERASE,
		INPUT_PICK,
		INPUT_SELECT,
		INPUT_PASTE,
	};

	enum ClipMode {

		CLIP_DISABLED,
		CLIP_ABOVE,
		CLIP_BELOW
	};

	enum DisplayMode {
		DISPLAY_THUMBNAIL,
		DISPLAY_LIST
	};

	UndoRedo *undo_redo;
	InputAction input_action;
	Panel *panel;
	MenuButton *options;
	SpinBox *floor;
	double accumulated_floor_delta;
	ToolButton *mode_thumbnail;
	ToolButton *mode_list;
	LineEdit *search_box;
	HSlider *size_slider;
	HBoxContainer *spatial_editor_hb;
	ConfirmationDialog *settings_dialog;
	VBoxContainer *settings_vbc;
	SpinBox *settings_pick_distance;
	Label *spin_box_label;

	struct SetItem {
		Vector3 pos;
		int new_value;
		int new_orientation;
		int old_value;
		int old_orientation;
	};

	List<SetItem> set_items;

	GridMap *node;
	MeshLibrary *last_mesh_library;
	ClipMode clip_mode;

	bool lock_view;
	Transform grid_xform;
	Transform edit_grid_xform;
	Vector3::Axis edit_axis;
	int edit_floor[3];
	Vector3 grid_ofs;

	RID grid[3];
	RID grid_instance[3];
	RID cursor_instance;
	RID selection_mesh;
	RID selection_instance;
	RID selection_level_mesh[3];
	RID selection_level_instance[3];
	RID paste_mesh;
	RID paste_instance;

	struct ClipboardItem {
		int cell_item;
		Vector3 grid_offset;
		int orientation;
		RID instance;
	};

	List<ClipboardItem> clipboard_items;

	Ref<SpatialMaterial> indicator_mat;
	Ref<SpatialMaterial> inner_mat;
	Ref<SpatialMaterial> outer_mat;
	Ref<SpatialMaterial> selection_floor_mat;

	bool updating;

	struct Selection {
		Vector3 click;
		Vector3 current;
		Vector3 begin;
		Vector3 end;
		bool active;
	} selection;
	Selection last_selection;

	struct PasteIndicator {
		Vector3 click;
		Vector3 current;
		Vector3 begin;
		Vector3 end;
		int orientation;
	};
	PasteIndicator paste_indicator;

	bool cursor_visible;
	Transform cursor_transform;

	Vector3 cursor_origin;

	int display_mode;
	int selected_palette;
	int cursor_rot;

	enum Menu {

		MENU_OPTION_NEXT_LEVEL,
		MENU_OPTION_PREV_LEVEL,
		MENU_OPTION_LOCK_VIEW,
		MENU_OPTION_CLIP_DISABLED,
		MENU_OPTION_CLIP_ABOVE,
		MENU_OPTION_CLIP_BELOW,
		MENU_OPTION_X_AXIS,
		MENU_OPTION_Y_AXIS,
		MENU_OPTION_Z_AXIS,
		MENU_OPTION_CURSOR_ROTATE_Y,
		MENU_OPTION_CURSOR_ROTATE_X,
		MENU_OPTION_CURSOR_ROTATE_Z,
		MENU_OPTION_CURSOR_BACK_ROTATE_Y,
		MENU_OPTION_CURSOR_BACK_ROTATE_X,
		MENU_OPTION_CURSOR_BACK_ROTATE_Z,
		MENU_OPTION_CURSOR_CLEAR_ROTATION,
		MENU_OPTION_PASTE_SELECTS,
		MENU_OPTION_SELECTION_DUPLICATE,
		MENU_OPTION_SELECTION_CUT,
		MENU_OPTION_SELECTION_CLEAR,
		MENU_OPTION_SELECTION_FILL,
		MENU_OPTION_GRIDMAP_SETTINGS

	};

	SpatialEditorPlugin *spatial_editor;

	struct AreaDisplay {
		RID mesh;
		RID instance;
	};

	ItemList *mesh_library_palette;
	Label *info_message;

	EditorNode *editor;

	void update_grid(); // Change which and where the grid is displayed
	void _draw_grids(const Vector3 &cell_size);
	void _configure();
	void _menu_option(int);
	void update_palette();
	void _set_display_mode(int p_mode);
	void _item_selected_cbk(int idx);
	void _update_cursor_transform();
	void _update_cursor_instance();
	void _update_clip();

	void _text_changed(const String &p_text);
	void _sbox_input(const Ref<InputEvent> &p_ie);
	void _mesh_library_palette_input(const Ref<InputEvent> &p_ie);

	void _icon_size_changed(float p_value);

	void _clear_clipboard_data();
	void _set_clipboard_data();
	void _update_paste_indicator();
	void _do_paste();
	void _update_selection_transform();
	void _validate_selection();
	void _set_selection(bool p_active, const Vector3 &p_begin = Vector3(), const Vector3 &p_end = Vector3());

	void _floor_changed(float p_value);
	void _floor_mouse_exited();

	void _delete_selection();
	void _fill_selection();

	bool do_input_action(Camera *p_camera, const Point2 &p_point, bool p_click);

	friend class GridMapEditorPlugin;

protected:
	void _notification(int p_what);
	void _node_removed(Node *p_node);
	static void _bind_methods();

public:
	bool forward_spatial_input_event(Camera *p_camera, const Ref<InputEvent> &p_event);

	void edit(GridMap *p_gridmap);
	GridMapEditor() {}
	GridMapEditor(EditorNode *p_editor);
	~GridMapEditor();
};

class GridMapEditorPlugin : public EditorPlugin {
	GDCLASS(GridMapEditorPlugin, EditorPlugin);

	GridMapEditor *grid_map_editor;
	EditorNode *editor;

protected:
	void _notification(int p_what);

public:
	virtual bool forward_spatial_gui_input(Camera *p_camera, const Ref<InputEvent> &p_event) { return grid_map_editor->forward_spatial_input_event(p_camera, p_event); }
	virtual String get_name() const { return "GridMap"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual void make_visible(bool p_visible);

	GridMapEditorPlugin(EditorNode *p_node);
	~GridMapEditorPlugin();
};

#endif // CUBE_GRID_MAP_EDITOR_PLUGIN_H
