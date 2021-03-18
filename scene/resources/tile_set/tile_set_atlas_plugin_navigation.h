/*************************************************************************/
/*  tile_set_atlas_plugin_navigation.h                                   */
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

#ifndef TILE_SET_ATLAS_PLUGIN_NAVIGATION_H
#define TILE_SET_ATLAS_PLUGIN_NAVIGATION_H

#include "tile_set_atlas_plugin.h"

#include "scene/2d/navigation_region_2d.h"

class TileSetAtlasPluginNavigation : public TileSetPlugin {
	GDCLASS(TileSetAtlasPluginNavigation, TileSetPlugin);

public:
	// Tilemap updates
	virtual void tilemap_notification(TileMap *p_tile_map, int p_what) override;
	virtual void update_dirty_quadrants(TileMap *p_tile_map, SelfList<TileMapQuadrant>::List &r_dirty_quadrant_list) override;
	//virtual void create_quadrant(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) override;
	virtual void cleanup_quadrant(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) override;
	virtual void draw_quadrant_debug(TileMap *p_tile_map, TileMapQuadrant *p_quadrant) override;
};

#endif // TILE_SET_ATLAS_PLUGIN_NAVIGATION_H
