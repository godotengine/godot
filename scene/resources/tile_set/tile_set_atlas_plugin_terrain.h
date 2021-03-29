/*************************************************************************/
/*  tile_set_atlas_plugin_terrain.h                                      */
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

#ifndef TILE_SET_ATLAS_PLUGIN_TERRAIN_H
#define TILE_SET_ATLAS_PLUGIN_TERRAIN_H

#include "tile_set_atlas_plugin.h"

#include "scene/resources/tile_set/tile_set.h"

class TileMap;
class TileData;

class TileSetAtlasPluginTerrain : public TileSetPlugin {
	GDCLASS(TileSetAtlasPluginTerrain, TileSetPlugin);

private:
	static void _draw_square_corner_or_side_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileSet::CellNeighbor p_bit);
	static void _draw_square_corner_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileSet::CellNeighbor p_bit);
	static void _draw_square_side_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileSet::CellNeighbor p_bit);

	static void _draw_isometric_corner_or_side_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileSet::CellNeighbor p_bit);
	static void _draw_isometric_corner_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileSet::CellNeighbor p_bit);
	static void _draw_isometric_side_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileSet::CellNeighbor p_bit);

	static void _draw_half_offset_corner_or_side_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileSet::CellNeighbor p_bit, float p_overlap, TileSet::TileOffsetAxis p_offset_axis);
	static void _draw_half_offset_corner_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileSet::CellNeighbor p_bit, float p_overlap, TileSet::TileOffsetAxis p_offset_axis);
	static void _draw_half_offset_side_terrain_bit(CanvasItem *p_canvas_item, Color p_color, Vector2i p_size, TileSet::CellNeighbor p_bit, float p_overlap, TileSet::TileOffsetAxis p_offset_axis);

public:
	//virtual void tilemap_notification(const TileMap * p_tile_map, int p_what);

	static void draw_terrains(CanvasItem *p_canvas_item, Transform2D p_transform, TileSet *p_tile_set, const TileData *p_tile_data);
};

#endif // TILE_SET_ATLAS_PLUGIN_TERRAIN_H
