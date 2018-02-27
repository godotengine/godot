/*************************************************************************/
/*  tile_set_editor_plugin.h                                             */
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

#ifndef TILE_SET_EDITOR_PLUGIN_H
#define TILE_SET_EDITOR_PLUGIN_H

#include "editor/editor_name_dialog.h"
#include "editor/editor_node.h"
#include "editor/plugins/texture_region_editor_plugin.h"
#include "scene/2d/sprite.h"
#include "scene/resources/convex_polygon_shape_2d.h"
#include "scene/resources/tile_set.h"

class TileSetEditorHelper;

class TileSetEditor : public Control {

	friend class TileSetEditorPlugin;
	friend class TextureRegionEditor;

	GDCLASS(TileSetEditor, Control);

	enum EditMode {
		EDITMODE_COLLISION,
		EDITMODE_OCCLUSION,
		EDITMODE_NAVIGATION,
		EDITMODE_BITMASK,
		EDITMODE_PRIORITY,
		EDITMODE_ICON,
		EDITMODE_MAX
	};

	enum TileSetToolbar {
		TOOLBAR_DUMMY,
		TOOLBAR_BITMASK,
		TOOLBAR_SHAPE,
		TOOLBAR_MAX
	};

	enum TileSetTools {
		TOOL_SELECT,
		BITMASK_COPY,
		BITMASK_PASTE,
		BITMASK_CLEAR,
		SHAPE_NEW_POLYGON,
		SHAPE_DELETE,
		SHAPE_CREATE_FROM_BITMASK,
		SHAPE_CREATE_FROM_NOT_BITMASK,
		SHAPE_KEEP_INSIDE_TILE,
		SHAPE_GRID_SNAP,
		ZOOM_OUT,
		ZOOM_1,
		ZOOM_IN,
		TOOL_MAX
	};

	Ref<TileSet> tileset;

	Ref<ConvexPolygonShape2D> edited_collision_shape;
	Ref<OccluderPolygon2D> edited_occlusion_shape;
	Ref<NavigationPolygon> edited_navigation_shape;

	int current_item_index;
	Sprite *preview;
	ScrollContainer *scroll;
	Control *workspace_container;
	bool draw_handles;
	Control *workspace_overlay;
	Control *workspace;
	Button *tool_editmode[EDITMODE_MAX];
	HBoxContainer *tool_containers[TOOLBAR_MAX];
	HBoxContainer *toolbar;
	HBoxContainer *hb_grid;
	ToolButton *tools[TOOL_MAX];
	SpinBox *spin_priority;
	SpinBox *sb_step_y;
	SpinBox *sb_step_x;
	SpinBox *sb_off_y;
	SpinBox *sb_off_x;
	SpinBox *sb_sep_y;
	SpinBox *sb_sep_x;
	EditMode edit_mode;

	Vector2 snap_step;
	Vector2 snap_offset;
	Vector2 snap_separation;

	bool creating_shape;
	int dragging_point;
	Vector2 edited_shape_coord;
	PoolVector2Array current_shape;
	Map<Vector2, uint16_t> bitmask_map_copy;

	EditorNode *editor;
	TextureRegionEditor *texture_region_editor;
	Control *bottom_panel;
	Control *side_panel;
	ItemList *tile_list;
	PropertyEditor *property_editor;
	TileSetEditorHelper *helper;

	MenuButton *menu;
	ConfirmationDialog *cd;
	EditorNameDialog *nd;
	AcceptDialog *err_dialog;

	enum {

		MENU_OPTION_ADD_ITEM,
		MENU_OPTION_REMOVE_ITEM,
		MENU_OPTION_CREATE_FROM_SCENE,
		MENU_OPTION_MERGE_FROM_SCENE
	};

	int option;
	void _menu_cbk(int p_option);
	void _menu_confirm();
	void _name_dialog_confirm(const String &name);

	static void _import_node(Node *p_node, Ref<TileSet> p_library);
	static void _import_scene(Node *p_scene, Ref<TileSet> p_library, bool p_merge);

protected:
	static void _bind_methods();
	void _notification(int p_what);
	virtual void _changed_callback(Object *p_changed, const char *p_prop);

public:
	void edit(const Ref<TileSet> &p_tileset);
	static Error update_library_file(Node *p_base_scene, Ref<TileSet> ml, bool p_merge = true);

	TileSetEditor(EditorNode *p_editor);
	~TileSetEditor();

private:
	void _on_tile_list_selected(int p_index);
	void _on_edit_mode_changed(int p_edit_mode);
	void _on_workspace_overlay_draw();
	void _on_workspace_draw();
	void _on_workspace_input(const Ref<InputEvent> &p_ie);
	void _on_tool_clicked(int p_tool);
	void _on_priority_changed(float val);
	void _on_grid_snap_toggled(bool p_val);
	void _set_snap_step_x(float p_val);
	void _set_snap_step_y(float p_val);
	void _set_snap_off_x(float p_val);
	void _set_snap_off_y(float p_val);
	void _set_snap_sep_x(float p_val);
	void _set_snap_sep_y(float p_val);

	void initialize_bottom_editor();
	void draw_highlight_tile(Vector2 coord, const Vector<Vector2> &other_highlighted = Vector<Vector2>());
	void draw_grid_snap();
	void draw_polygon_shapes();
	void close_shape(const Vector2 &shape_anchor);
	void select_coord(const Vector2 &coord);
	Vector2 snap_point(const Vector2 &point);
	void update_tile_list();
	void update_tile_list_icon();
	void update_workspace_tile_mode();

	int get_current_tile();
};

class TileSetEditorHelper : public Object {

	friend class TileSetEditor;
	GDCLASS(TileSetEditorHelper, Object);

	Ref<TileSet> tileset;
	TileSetEditor *tileset_editor;
	int selected_tile;

public:
	void set_tileset(const Ref<TileSet> &p_tileset);

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	TileSetEditorHelper(TileSetEditor *p_tileset_editor);
};

class TileSetEditorPlugin : public EditorPlugin {

	GDCLASS(TileSetEditorPlugin, EditorPlugin);

	TileSetEditor *tileset_editor;
	EditorNode *editor;

	ToolButton *tileset_editor_button;
	ToolButton *texture_region_button;

public:
	virtual String get_name() const { return "TileSet"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	TileSetEditorPlugin(EditorNode *p_node);
};

#endif // TILE_SET_EDITOR_PLUGIN_H
