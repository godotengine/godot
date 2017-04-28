/*************************************************************************/
/*  grid_map_editor_plugin.h                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#ifndef GRID_MAP_EDITOR_PLUGIN_H
#define GRID_MAP_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "editor/pane_drag.h"
#include "grid_map.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

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
		INPUT_COPY,
		INPUT_SELECT,
		INPUT_DUPLICATE,
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
	OptionButton *edit_mode;
	ToolButton *mode_thumbnail;
	ToolButton *mode_list;
	HBoxContainer *spatial_editor_hb;
	ConfirmationDialog *settings_dialog;
	VBoxContainer *settings_vbc;
	SpinBox *settings_pick_distance;

	struct SetItem {

		Vector3 pos;
		int new_value;
		int new_orientation;
		int old_value;
		int old_orientation;
	};

	List<SetItem> set_items;

	GridMap *node;
	MeshLibrary *last_theme;
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
	RID duplicate_mesh;
	RID duplicate_instance;

	Ref<SpatialMaterial> indicator_mat;
	Ref<SpatialMaterial> inner_mat;
	Ref<SpatialMaterial> outer_mat;

	bool updating;

	struct Selection {

		Vector3 click;
		Vector3 current;
		Vector3 begin;
		Vector3 end;
		int duplicate_rot;
		bool active;
	} selection;

	bool cursor_visible;
	Transform cursor_transform;

	Vector3 cursor_origin;
	Vector3 last_mouseover;

	int display_mode;
	int selected_pallete;
	int selected_area;
	int cursor_rot;

	enum Menu {

		MENU_OPTION_CONFIGURE,
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
		MENU_OPTION_DUPLICATE_SELECTS,
		MENU_OPTION_SELECTION_MAKE_AREA,
		MENU_OPTION_SELECTION_MAKE_EXTERIOR_CONNECTOR,
		MENU_OPTION_SELECTION_DUPLICATE,
		MENU_OPTION_SELECTION_CLEAR,
		MENU_OPTION_REMOVE_AREA,
		MENU_OPTION_GRIDMAP_SETTINGS

	};

	SpatialEditorPlugin *spatial_editor;

	struct AreaDisplay {

		RID mesh;
		RID instance;
	};

	Vector<AreaDisplay> areas;

	void _update_areas_display();
	void _clear_areas();

	void update_grid();
	void _configure();
	void _menu_option(int);
	void update_pallete();
	void _set_display_mode(int p_mode);
	ItemList *theme_pallete;
	Tree *area_list;
	void _item_selected_cbk(int idx);
	void _update_cursor_transform();
	void _update_cursor_instance();
	void _update_clip();

	void _update_duplicate_indicator();
	void _duplicate_paste();
	void _update_selection_transform();
	void _validate_selection();

	void _edit_mode_changed(int p_what);
	void _area_renamed();
	void _area_selected();

	void _floor_changed(float p_value);

	void _delete_selection();
	void update_areas();

	EditorNode *editor;
	bool do_input_action(Camera *p_camera, const Point2 &p_point, bool p_click);

	friend class GridMapEditorPlugin;
	Panel *theme_panel;

protected:
	void _notification(int p_what);
	void _node_removed(Node *p_node);
	static void _bind_methods();

public:
	bool forward_spatial_input_event(Camera *p_camera, const InputEvent &p_event);

	void edit(GridMap *p_gridmap);
	GridMapEditor() {}
	GridMapEditor(EditorNode *p_editor);
	~GridMapEditor();
};

class GridMapEditorPlugin : public EditorPlugin {

	GDCLASS(GridMapEditorPlugin, EditorPlugin);

	GridMapEditor *gridmap_editor;
	EditorNode *editor;

public:
	virtual bool forward_spatial_input_event(Camera *p_camera, const InputEvent &p_event) { return gridmap_editor->forward_spatial_input_event(p_camera, p_event); }
	virtual String get_name() const { return "GridMap"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	GridMapEditorPlugin(EditorNode *p_node);
	~GridMapEditorPlugin();
};

#endif // CUBE_GRID_MAP_EDITOR_PLUGIN_H
