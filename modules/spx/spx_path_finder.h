/**************************************************************************/
/*  spx_path_finer.h                                                      */
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

#ifndef SPX_PATH_FINER_H 
#define SPX_PATH_FINER_H

#include "core/object/object.h"
#include "scene/2d/node_2d.h"
#include "core/math/a_star_grid_2d.h"

class TileMapLayer;
class Node;
class PathDebugDrawer;
class CollisionShape2D;

class SpxPathFinder : public RefCounted {
    GDCLASS(SpxPathFinder, RefCounted);

private:
    Ref<AStarGrid2D> astar;
    PathDebugDrawer *drawer = nullptr;

    Vector2i _world_to_cell(const Vector2 &pos) const;
    Vector2 _cell_to_world(const Vector2i &cell) const;

    void _process_rectangle_shape(Node2D *owner, CollisionShape2D *shape);
    void _process_static_obstacles(Node2D *body);
    void _process_tilemap_obstacles(TileMapLayer *tilemap, int p_layer_id = 0);

protected:
    static void _bind_methods();

public:
    SpxPathFinder();
    ~SpxPathFinder();

    void setup_grid_spx(GdVec2 size, GdVec2 cell_size, GdBool with_debug);
	void setup_grid(Vector2i size, Vector2i cell_size, bool with_debug = false);
	void set_jumping_enabled(bool p_enabled);
	void add_all_obstacles(Node *root);

    GdArray find_path_spx(GdVec2 p_from, GdVec2 p_to);
	PackedVector2Array find_path(Vector2 start, Vector2 end);

    Vector2i get_size() const;
    Vector2 get_cell_size() const;
    bool is_cell_solid(Vector2i cell) const;
    Vector2 cell_to_world_gd(Vector2i cell) const;

    void clear_drawer(){
        drawer = nullptr;
    };
};

class PathDebugDrawer : public Node2D {
    GDCLASS(PathDebugDrawer, Node2D);

private:
    Ref<SpxPathFinder> path_finder;
    PackedVector2Array path;

    Vector2 start;
    Vector2 end;
    bool start_set = false;
    bool end_set = false;

    enum DragState { NONE, DRAG_START, DRAG_END };
    DragState dragging = NONE;

    void _update_path();
    bool _is_near(const Vector2 &p1, const Vector2 &p2, float threshold = 10.0) const;

protected:
    PathDebugDrawer() = default;
    ~PathDebugDrawer() = default;

    static void _bind_methods();
    void _notification(int p_what);
    void _ready();
	void _draw();
    void _exit_tree();
	void input(const Ref<InputEvent> &p_event) override;

public:
    explicit PathDebugDrawer(const Ref<SpxPathFinder> &p_path_finder) {
        path_finder = p_path_finder;
    }

    void set_path_finder(const Ref<SpxPathFinder> &p_path_finder);
    void set_path(const PackedVector2Array &p_path);
};


#endif // SPX_PATH_FINER_H 