/*************************************************************************/
/*  tile_data_editors.h                                                  */
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

#ifndef TILE_DATA_EDITORS_H
#define TILE_DATA_EDITORS_H

#include "scene/gui/control.h"
#include "scene/resources/tile_set.h"

class TileDataEditor : public Control {
	GDCLASS(TileDataEditor, Control);

protected:
	TileData *tile_data;
	String property;

	TileData *_get_tile_data(TileSet *p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile);

public:
	// Edits a TileData property.
	void edit(TileSet *p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile, String p_property);

	// Used to draw the value over a tile.
	virtual void draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileSet *p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile, String p_property){};
};

class TileDataTextureOffsetEditor : public TileDataEditor {
	GDCLASS(TileDataTextureOffsetEditor, TileDataEditor);

public:
	virtual void draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileSet *p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile, String p_property) override;
};

class TileDataIntegerEditor : public TileDataEditor {
	GDCLASS(TileDataIntegerEditor, TileDataEditor);

public:
	virtual void draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileSet *p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile, String p_property) override;
};

class TileDataFloatEditor : public TileDataEditor {
	GDCLASS(TileDataFloatEditor, TileDataEditor);

public:
	virtual void draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileSet *p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile, String p_property) override;
};

class TileDataPositionEditor : public TileDataEditor {
	GDCLASS(TileDataPositionEditor, TileDataEditor);

public:
	virtual void draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileSet *p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile, String p_property) override;
};

class TileDataYSortEditor : public TileDataEditor {
	GDCLASS(TileDataYSortEditor, TileDataEditor);

public:
	virtual void draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileSet *p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile, String p_property) override;
};

class TileDataOcclusionShapeEditor : public TileDataEditor {
	GDCLASS(TileDataOcclusionShapeEditor, TileDataEditor);

public:
	virtual void draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileSet *p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile, String p_property) override;
};

class TileDataCollisionShapeEditor : public TileDataEditor {
	GDCLASS(TileDataCollisionShapeEditor, TileDataEditor);

public:
	virtual void draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileSet *p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile, String p_property) override;
};

class TileDataTerrainsEditor : public TileDataEditor {
	GDCLASS(TileDataTerrainsEditor, TileDataEditor);

public:
	virtual void draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileSet *p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile, String p_property) override;
};

class TileDataNavigationPolygonEditor : public TileDataEditor {
	GDCLASS(TileDataNavigationPolygonEditor, TileDataEditor);

public:
	virtual void draw_over_tile(CanvasItem *p_canvas_item, Transform2D p_transform, TileSet *p_tile_set, int p_atlas_source_id, Vector2i p_atlas_coords, int p_alternative_tile, String p_property) override;
};

#endif // TILE_DATA_EDITORS_H
