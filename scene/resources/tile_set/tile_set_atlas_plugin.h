/*************************************************************************/
/*  tile_set_atlas_plugin.h                                              */
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

#ifndef TILE_SET_PLUGIN_H
#define TILE_SET_PLUGIN_H

#include "core/object/class_db.h"
#include "core/object/object.h"
#include "core/templates/self_list.h"

class TileSet;
class TileMap;
struct TileMapQuadrant;

class TileSetAtlasPluginTileSetData {
public:
	// TileSet properties.
	virtual bool set(const StringName &p_name, const Variant &p_value) = 0;
	virtual bool get(const StringName &p_name, Variant &r_ret) const = 0;
	virtual void get_property_list(List<PropertyInfo> *p_list) const = 0;

	virtual ~TileSetAtlasPluginTileSetData(){};
};

class TileSetAtlasPluginTileData {
public:
	// TileSet tile properties.
	virtual bool set(const StringName &p_name, const Variant &p_value) = 0;
	virtual bool get(const StringName &p_name, Variant &r_ret) const = 0;
	virtual void get_property_list(List<PropertyInfo> *p_list) const = 0;

	virtual ~TileSetAtlasPluginTileData(){};
};

class TileSetAtlasPlugin : public Object {
	GDCLASS(TileSetAtlasPlugin, Object);

public:
	/*
    // Get the list of tileset layers supported by a tileset plugin
    virtual int get_layer_type_count() { return 0; };
    virtual StringName get_layer_type(int p_id) { return ""; };
    virtual String get_layer_type_icon(int p_id) { return ""; };
    virtual bool get_layer_type_multiple_mode(int p_id) { return true; };
*/

	// Name.
	virtual String get_name() const = 0;
	virtual String get_id() const = 0;

	// Tilemap updates.
	virtual void tilemap_notification(TileMap *p_tile_map, int p_what){};
	virtual void update_dirty_quadrants(TileMap *p_tile_map, SelfList<TileMapQuadrant>::List &r_dirty_quadrant_list){};
	virtual void initialize_quadrant(TileMap *p_tile_map, TileMapQuadrant *p_quadrant){};
	virtual void create_quadrant(TileMap *p_tile_map, const Vector2i &p_quadrant_coords, TileMapQuadrant *p_quadrant){};
	virtual void cleanup_quadrant(TileMap *p_tile_map, TileMapQuadrant *p_quadrant){};
};

#endif // TILE_SET_PLUGIN_H
