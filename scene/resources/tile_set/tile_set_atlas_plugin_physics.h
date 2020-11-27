/*************************************************************************/
/*  tile_set_atlas_plugin_physics.h                                      */
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

#ifndef TILE_SET_ATLAS_PLUGIN_PHYSICS_H
#define TILE_SET_ATLAS_PLUGIN_PHYSICS_H

#include "scene/resources/shape_2d.h"

class TileMap;

// Tile data
struct ShapeTileData {
	bool one_way;
	float one_way_margin;
	Ref<Shape2D> shape;
	Transform2D transform;
};

struct PhysicsLayerTileData {
	Vector<ShapeTileData> shapes;
};

struct PhysicsTileData {
	Vector<PhysicsLayerTileData> collisions;
};

// TileSet data
struct PhysicsLayerTileSetData {
	uint32_t collision_layer;
	uint32_t collision_mask;
	bool use_kinematic;
	float friction;
	float bounce;
};

struct PhysicsTileSetData {
	Vector<PhysicsLayerTileSetData> layers;
};

/*
class TileSetAtlasPluginPhysics : public Resource {
	GDCLASS(TileSetAtlasPluginPhysics, Resource);

private:
    void _update_state(const TileMap * p_tile_map);
    void _add_shape(int &shape_idx, const Quadrant &p_q, const Ref<Shape2D> &p_shape, const TileSet::ShapeData &p_shape_data, const Transform2D &p_xform, const Vector2 &p_metadata);

public:
    // Get the list of tileset layers supported by a tileset plugin
    virtual int get_layer_type_count() { return 0; };
    virtual String get_layer_type_name(int p_id) { return ""; };
    virtual String get_layer_type_icon(int p_id) { return ""; };
    virtual bool get_layer_type_multiple_mode(int p_id) { return true; };

    // Tilemap updates
    virtual void tilemap_notification(TileMap * p_tile_map, int p_what);
    virtual void update_dirty_quadrants(TileMap * p_tile_map);
    virtual void initialize_quadrant(TileMap * p_tile_map, Quadrant * p_quadrant);
    virtual void cleanup_quadrant(TileMap * p_tile_map, Quadrant * p_quadrant);
};
*/
#endif // TILE_SET_ATLAS_PLUGIN_PHYSICS_H
