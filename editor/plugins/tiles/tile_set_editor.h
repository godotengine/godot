/*************************************************************************/
/*  tile_set_editor.h                                                    */
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

#ifndef TILE_SET_EDITOR_H
#define TILE_SET_EDITOR_H

#include "scene/gui/box_container.h"
#include "scene/resources/tile_set.h"
#include "tile_data_editors.h"
#include "tile_set_atlas_source_editor.h"
#include "tile_set_scenes_collection_source_editor.h"

class TileSetEditor : public VBoxContainer {
	GDCLASS(TileSetEditor, VBoxContainer);

	static TileSetEditor *singleton;

private:
	Ref<TileSet> tile_set;
	bool tile_set_changed_needs_update = false;

	Label *no_source_selected_label;
	TileSetAtlasSourceEditor *tile_set_atlas_source_editor;
	TileSetScenesCollectionSourceEditor *tile_set_scenes_collection_source_editor;

	UndoRedo *undo_redo = EditorNode::get_undo_redo();

	void _update_atlas_sources_list(int force_selected_id = -1);

	// List of tile data editors.
	TileDataTextureOffsetEditor *tile_data_texture_offset_editor = memnew(TileDataTextureOffsetEditor);
	TileDataYSortEditor *tile_data_y_sort_editor = memnew(TileDataYSortEditor);
	TileDataIntegerEditor *tile_data_integer_editor = memnew(TileDataIntegerEditor);
	TileDataFloatEditor *tile_data_float_editor = memnew(TileDataFloatEditor);
	TileDataOcclusionShapeEditor *tile_data_occlusion_shape_editor = memnew(TileDataOcclusionShapeEditor);
	TileDataCollisionShapeEditor *tile_data_collision_shape_editor = memnew(TileDataCollisionShapeEditor);
	TileDataTerrainsEditor *tile_data_terrains_editor = memnew(TileDataTerrainsEditor);
	TileDataNavigationPolygonEditor *tile_data_navigation_polygon_editor = memnew(TileDataNavigationPolygonEditor);

	// -- Sources management --
	Button *sources_delete_button;
	MenuButton *sources_add_button;
	ItemList *sources_list;
	Ref<Texture2D> missing_texture_texture;
	void _source_selected(int p_source_index);
	void _source_add_id_pressed(int p_id_pressed);
	void _source_delete_pressed();

	void _tile_set_changed();

	void _undo_redo_inspector_callback(Object *p_undo_redo, Object *p_edited, String p_property, Variant p_new_value);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	_FORCE_INLINE_ static TileSetEditor *get_singleton() { return singleton; }

	TileDataEditor *get_tile_data_editor(String property);
	void edit(Ref<TileSet> p_tile_set);
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;

	TileSetEditor();
	~TileSetEditor();
};

#endif // TILE_SET_EDITOR_PLUGIN_H
