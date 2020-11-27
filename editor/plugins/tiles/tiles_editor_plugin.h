/*************************************************************************/
/*  tiles_editor_plugin.h                                                */
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

#ifndef TILES_EDITOR_PLUGIN_H
#define TILES_EDITOR_PLUGIN_H

#include "editor/editor_plugin.h"
#include "scene/gui/box_container.h"

#include "tile_atlas_view.h"
#include "tile_map_editor.h"
#include "tile_set_editor.h"

class TilesEditor : public VBoxContainer {
	GDCLASS(TilesEditor, VBoxContainer);

	static TilesEditor *singleton;

private:
	TileMap *tile_map;
	Ref<TileSet> tile_set;

	Button *tileset_tilemap_switch_button;

	Control *tilemap_toolbar;
	TileMapEditor *tilemap_editor;

	TileSetEditor *tileset_editor;

	void _update_switch_button();
	void _update_editors();

	// For synchronization.
	int sources_lists_current = -1;
	Set<ItemList *> sources_lists;
	void _source_list_changed();
	Set<TileAtlasView *> atlases_views;
	void _synchronize_atlas_views(Variant p_unused, Object *p_emitter); // TODO: remove the unused argument when connecting supports unbinding arguments with connect(...).

	void _tile_map_changed();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	_FORCE_INLINE_ static TilesEditor *get_singleton() { return singleton; }

	bool forward_canvas_gui_input(const Ref<InputEvent> &p_event) { return tilemap_editor->forward_canvas_gui_input(p_event); }
	void forward_canvas_draw_over_viewport(Control *p_overlay) { tilemap_editor->forward_canvas_draw_over_viewport(p_overlay); }

	void synchronize_sources_lists(int p_current = -1);
	void register_atlas_view_for_synchronization(TileAtlasView *p_atlas_view);
	void register_atlas_source_list_for_synchronization(ItemList *p_source_list);

	void edit(Object *p_object);

	TilesEditor(EditorNode *p_editor);
	~TilesEditor();
};

class TilesEditorPlugin : public EditorPlugin {
	GDCLASS(TilesEditorPlugin, EditorPlugin);

private:
	EditorNode *editor_node;
	TilesEditor *tiles_editor;
	Button *tiles_editor_button;

protected:
	void _notification(int p_what);

public:
	virtual bool forward_canvas_gui_input(const Ref<InputEvent> &p_event) override { return tiles_editor->forward_canvas_gui_input(p_event); }
	virtual void forward_canvas_draw_over_viewport(Control *p_overlay) override { tiles_editor->forward_canvas_draw_over_viewport(p_overlay); }

	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;

	TilesEditorPlugin(EditorNode *p_node);
	~TilesEditorPlugin();
};

#endif // TILES_EDITOR_PLUGIN_H
