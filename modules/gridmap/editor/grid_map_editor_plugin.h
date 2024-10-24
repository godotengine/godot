/**************************************************************************/
/*  grid_map_editor_plugin.h                                              */
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

#ifndef GRID_MAP_EDITOR_PLUGIN_H
#define GRID_MAP_EDITOR_PLUGIN_H

#ifdef TOOLS_ENABLED

#include "../grid_map.h"

#include "editor/plugins/editor_plugin.h"
#include "scene/gui/box_container.h"
#include "scene/gui/item_list.h"
#include "scene/gui/slider.h"
#include "scene/gui/spin_box.h"

class ConfirmationDialog;
class MenuButton;
class Node3DEditorPlugin;

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

	enum DisplayMode {
		DISPLAY_THUMBNAIL,
		DISPLAY_LIST
	};

	InputAction input_action = INPUT_NONE;
	Panel *panel = nullptr;
	MenuButton *options = nullptr;
	SpinBox *floor = nullptr;
	double accumulated_floor_delta = 0.0;
	Button *mode_thumbnail = nullptr;
	Button *mode_list = nullptr;
	LineEdit *search_box = nullptr;
	HSlider *size_slider = nullptr;
	HBoxContainer *spatial_editor_hb = nullptr;
	ConfirmationDialog *settings_dialog = nullptr;
	VBoxContainer *settings_vbc = nullptr;
	SpinBox *settings_pick_distance = nullptr;
	Label *spin_box_label = nullptr;

	struct SetItem {
		Vector3i position;
		int new_value = 0;
		int new_orientation = 0;
		int old_value = 0;
		int old_orientation = 0;
	};

	List<SetItem> set_items;

	GridMap *node = nullptr;
	// caching the node global transform to detect when the node has been
	// moved/scaled/rotated.
	Transform3D node_global_transform;
	Ref<MeshLibrary> mesh_library = nullptr;

	// plane we're editing cells on; depth comes from edit_floor
	Plane edit_plane;

	enum EditAxis {
		AXIS_X = 0,
		AXIS_Y,
		AXIS_Z,
		AXIS_Q, // axial hex coordinates northwest/southeast
		AXIS_R, // axial hex coordinates east/west
		AXIS_S, // axial hex coordinates northeast/southwest
		AXIS_MAX,
	};
	EditAxis edit_axis;
	int edit_floor[AXIS_MAX];

	RID active_grid_instance;
	RID grid_mesh[3];
	RID grid_instance[3];
	RID cursor_instance;

	struct ClipboardItem {
		int cell_item = 0;
		Vector3 grid_offset;
		int orientation = 0;
		RID instance;
	};

	List<ClipboardItem> clipboard_items;

	Ref<StandardMaterial3D> indicator_mat;
	Ref<StandardMaterial3D> inner_mat;
	Ref<StandardMaterial3D> outer_mat;

	struct Selection {
		Vector3 begin;
		Vector3 end;
		bool active = false;
	} selection;
	Selection last_selection;
	RID selection_tile_mesh;
	RID selection_multimesh;
	RID selection_instance;

	struct PasteIndicator {
		Vector3i current_cell;
		int orientation = 0;
	};
	PasteIndicator paste_indicator;

	bool cursor_visible = false;
	Transform3D cursor_transform;

	Vector3i cursor_cell;

	int display_mode = DISPLAY_THUMBNAIL;
	int selected_palette = -1;
	int cursor_rot = 0;

	enum Menu {
		MENU_OPTION_NEXT_LEVEL,
		MENU_OPTION_PREV_LEVEL,
		MENU_OPTION_LOCK_VIEW,
		MENU_OPTION_X_AXIS,
		MENU_OPTION_Y_AXIS,
		MENU_OPTION_Z_AXIS,
		MENU_OPTION_Q_AXIS,
		MENU_OPTION_R_AXIS,
		MENU_OPTION_S_AXIS,
		MENU_OPTION_ROTATE_AXIS_CW,
		MENU_OPTION_ROTATE_AXIS_CCW,
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

	Node3DEditorPlugin *spatial_editor = nullptr;

	struct AreaDisplay {
		RID mesh;
		RID instance;
	};

	ItemList *mesh_library_palette = nullptr;
	Label *info_message = nullptr;

	void update_grid(); // Change which and where the grid is displayed
	void _draw_hex_grid(RID p_grid, const Vector3 &p_cell_size);
	void _draw_hex_x_axis_grid(RID p_grid, const Vector3 &p_cell_size);
	void _draw_plane_grid(RID p_grid, const Vector3 &p_axis_n1, const Vector3 &p_axis_n2, const Vector3 &p_cell_size);
	void _draw_grids(const Vector3 &p_cell_size);
	void _update_cell_shape(const GridMap::CellShape cell_shape);
	void _update_options_menu();
	void _build_selection_meshes();
	void _configure();
	void _menu_option(int);
	void update_palette();
	void _update_mesh_library();
	void _set_display_mode(int p_mode);
	void _item_selected_cbk(int idx);
	void _update_cursor_transform();
	void _update_cursor_instance();
	void _update_theme();

	void _text_changed(const String &p_text);
	void _sbox_input(const Ref<InputEvent> &p_event);
	void _mesh_library_palette_input(const Ref<InputEvent> &p_ie);

	void _icon_size_changed(float p_value);

	void _clear_clipboard_data();
	void _set_clipboard_data();
	void _update_paste_indicator();
	void _do_paste();
	void _update_selection();
	void _set_selection(bool p_active, const Vector3 &p_begin = Vector3(), const Vector3 &p_end = Vector3());

	void _floor_changed(float p_value);
	void _floor_mouse_exited();

	void _delete_selection();
	void _fill_selection();

	bool do_input_action(Camera3D *p_camera, const Point2 &p_point, bool p_click);

	friend class GridMapEditorPlugin;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	EditorPlugin::AfterGUIInput forward_spatial_input_event(Camera3D *p_camera, const Ref<InputEvent> &p_event);

	void edit(GridMap *p_gridmap);
	GridMapEditor();
	~GridMapEditor();
};

class GridMapEditorPlugin : public EditorPlugin {
	GDCLASS(GridMapEditorPlugin, EditorPlugin);

	GridMapEditor *grid_map_editor = nullptr;

protected:
	void _notification(int p_what);

public:
	virtual EditorPlugin::AfterGUIInput forward_3d_gui_input(Camera3D *p_camera, const Ref<InputEvent> &p_event) override { return grid_map_editor->forward_spatial_input_event(p_camera, p_event); }
	virtual String get_name() const override { return "GridMap"; }
	bool has_main_screen() const override { return false; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;

	GridMapEditorPlugin();
	~GridMapEditorPlugin();
};

#endif // TOOLS_ENABLED

#endif // GRID_MAP_EDITOR_PLUGIN_H
