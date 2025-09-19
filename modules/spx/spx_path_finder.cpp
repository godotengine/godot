/**************************************************************************/
/*  spx_path_finer.cpp                                                    */
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

#include "scene/2d/tile_map_layer.h"
#include "scene/2d/physics/static_body_2d.h"
#include "scene/2d/physics/collision_shape_2d.h"
#include "scene/resources/2d/rectangle_shape_2d.h"
#include "core/math/geometry_2d.h"
#include "core/variant/variant.h"
#include "spx_sprite.h"
#include "spx_base_mgr.h"
#include "spx_engine.h"
#include "spx_path_finder.h"


void SpxPathFinder::_bind_methods() {
    ClassDB::bind_method(D_METHOD("setup_grid", "size", "cell_size", "with_debug"), &SpxPathFinder::setup_grid);
    ClassDB::bind_method(D_METHOD("add_all_obstacles", "root"), &SpxPathFinder::add_all_obstacles);
    ClassDB::bind_method(D_METHOD("find_path", "start", "end"), &SpxPathFinder::find_path);

    ClassDB::bind_method(D_METHOD("get_size"), &SpxPathFinder::get_size);
    ClassDB::bind_method(D_METHOD("get_cell_size"), &SpxPathFinder::get_cell_size);
    ClassDB::bind_method(D_METHOD("is_cell_solid", "cell"), &SpxPathFinder::is_cell_solid);
    ClassDB::bind_method(D_METHOD("cell_to_world_gd", "cell"), &SpxPathFinder::cell_to_world_gd);
}

Vector2i SpxPathFinder::_world_to_cell(const Vector2 &pos) const {
    Vector2 cell_size = astar->get_cell_size();
    return Vector2i(
        (int)Math::floor(pos.x / cell_size.x),
        (int)Math::floor(pos.y / cell_size.y)
    );
}

Vector2 SpxPathFinder::_cell_to_world(const Vector2i &cell) const {
    Vector2 cell_size = astar->get_cell_size();
    return Vector2(
        cell.x * cell_size.x + cell_size.x * 0.5,
        cell.y * cell_size.y + cell_size.y * 0.5
    );
}

void SpxPathFinder::_process_rectangle_shape(Node2D *owner, CollisionShape2D *shape) {
    if (!shape || !shape->get_shape().is_valid()) return;

    RectangleShape2D *rect = Object::cast_to<RectangleShape2D>(shape->get_shape().ptr());
    if (!rect) return;

    Vector2 half_size = rect->get_size() * 0.5f;

    PackedVector2Array local_points;
    local_points.push_back(Vector2(-half_size.x, -half_size.y));
    local_points.push_back(Vector2(half_size.x, -half_size.y));
    local_points.push_back(Vector2(half_size.x, half_size.y));
    local_points.push_back(Vector2(-half_size.x, half_size.y));

    Transform2D global_xform = owner->get_global_transform() * shape->get_transform();
    PackedVector2Array world_points;
    for (int i = 0; i < local_points.size(); i++) {
        world_points.push_back(global_xform.xform(local_points[i]));
    }

    Vector2 min_p = world_points[0];
    Vector2 max_p = world_points[0];
    for (int i = 1; i < world_points.size(); i++) {
        min_p = min_p.min(world_points[i]);
        max_p = max_p.max(world_points[i]);
    }

    Vector2i start = _world_to_cell(min_p);
    Vector2i end   = _world_to_cell(max_p);

    for (int x = start.x; x <= end.x; x++) {
        for (int y = start.y; y <= end.y; y++) {
            Vector2 cell_center = _cell_to_world(Vector2i(x, y));
            if (Geometry2D::is_point_in_polygon(cell_center, world_points)) {
                astar->set_point_solid(Vector2i(x, y), true);
            }
        }
    }
}

void SpxPathFinder::_process_static_obstacles(Node2D *body) {

    for (int j = 0; j < body->get_child_count(); j++) {
        Node *child = body->get_child(j);

        if (CollisionShape2D *shape = Object::cast_to<CollisionShape2D>(child)) {
            _process_rectangle_shape(body, shape);
        }
    }
}

void SpxPathFinder::_process_tilemap_obstacles(TileMapLayer *layer, int p_layer_id) {
    if (!layer) 
        return;

    Array used_cells = layer->get_used_cells();
    Vector2 tile_size = layer->get_tile_set()->get_tile_size();

    for (int i = 0; i < used_cells.size(); ++i) {
        Vector2i cell = used_cells[i];

        TileData *td = layer->get_cell_tile_data(cell);
        if (!td) {
            continue;
        }

        int poly_count = td->get_collision_polygons_count(p_layer_id);
        if (poly_count <= 0) {
            continue;
        }

        Vector2 cell_local = layer->map_to_local(cell);
        Vector2 cell_global = layer->to_global(cell_local); 
             
        Vector2i start = _world_to_cell(cell_global - tile_size / 2);
        Vector2i end   = _world_to_cell(cell_global + tile_size / 2);

        Rect2i rect(start, end - start + Vector2i(1, 1));
        astar->fill_solid_region(rect, true);
    } 
}

SpxPathFinder::SpxPathFinder() {
    astar.instantiate();
}

SpxPathFinder::~SpxPathFinder() {
    if(drawer){
        drawer->queue_free();
        drawer = nullptr;
    }
}

void SpxPathFinder::setup_grid_spx(GdVec2 size, GdVec2 cell_size, GdBool with_debug) {
    setup_grid(size, cell_size, with_debug);
}

void SpxPathFinder::setup_grid(Vector2i size, Vector2i cell_size, bool with_debug) {
	astar->set_region({-size / 2, size});
    astar->set_cell_size(cell_size);
    astar->set_diagonal_mode(AStarGrid2D::DIAGONAL_MODE_NEVER);
    astar->set_jumping_enabled(true);
    astar->update();

    Node* root = nullptr;
    if (SpxEngine::get_singleton()) {
        root = SpxEngine::get_singleton()->get_spx_root();
    }

    if (!root){
        root = SceneTree::get_singleton()->get_current_scene();
    }

    if (root) {
        add_all_obstacles(root);

        if (with_debug && !drawer) {
            drawer = memnew(PathDebugDrawer(this));
            drawer->set_path_finder(this);
            root->add_child(drawer);
        }
    }
}

void SpxPathFinder::add_all_obstacles(Node *root) {
    if (!root) 
        return;

    Vector<Node*> stack;
    stack.push_back(root);

    while(!stack.is_empty()){
        Node *node = stack[stack.size() - 1];
        stack.resize(stack.size() - 1);

        for (int i = 0; i < node->get_child_count(); i++) {
            Node *child = node->get_child(i);

            if (StaticBody2D *body = Object::cast_to<StaticBody2D>(child)) {
                _process_static_obstacles(body);
            } else if (SpxSprite *sprite = Object::cast_to<SpxSprite>(child)) {
                if(sprite->get_physics_mode() == SpxSprite::PhysicsMode::STATIC)
                    _process_static_obstacles(sprite);
            } else if (TileMapLayer *layer = Object::cast_to<TileMapLayer>(child)) {
                _process_tilemap_obstacles(layer);
            }

            stack.push_back(child);
        }
    }
}

GdArray SpxPathFinder::find_path_spx(GdVec2 p_from, GdVec2 p_to) {
    auto path_points = find_path(p_from * Vector2(1, -1), p_to * Vector2(1, -1));
    auto count = path_points.size();
	GdArray result = SpxBaseMgr::create_array(GD_ARRAY_TYPE_FLOAT, count * 2);

	for(auto i = 0; i < count; i ++){
        auto idx = i * 2;
		SpxBaseMgr::set_array(result, idx, path_points[i].x);
		SpxBaseMgr::set_array(result, idx + 1, -path_points[i].y);
	}

	return result;
}

PackedVector2Array SpxPathFinder::find_path(Vector2 start, Vector2 end) {
	PackedVector2Array path;

    Vector2i from = _world_to_cell(start);
    Vector2i to   = _world_to_cell(end);

    auto cell_path = astar->get_id_path(from, to);
    for (int i = 0; i < cell_path.size(); i++) {
        path.push_back(_cell_to_world(cell_path[i]));
    }

    return path;
}

Vector2i SpxPathFinder::get_size() const {
    return astar->get_size();
}

Vector2 SpxPathFinder::get_cell_size() const {
    return astar->get_cell_size();
}

bool SpxPathFinder::is_cell_solid(Vector2i cell) const {
    return astar->is_point_solid(cell);
}

Vector2 SpxPathFinder::cell_to_world_gd(Vector2i cell) const {
    return _cell_to_world(cell);
}


void PathDebugDrawer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_path_finder", "path_finder"), &PathDebugDrawer::set_path_finder);
    ClassDB::bind_method(D_METHOD("set_path", "path"), &PathDebugDrawer::set_path);
}

void PathDebugDrawer::_update_path() {
    if(start_set || end_set) queue_redraw();
    if (!start_set || !end_set) return;
    path = path_finder->find_path(start, end);
    queue_redraw();
}

bool PathDebugDrawer::_is_near(const Vector2 &p1, const Vector2 &p2, float threshold) const {
    return p1.distance_to(p2) <= threshold;
}

void PathDebugDrawer::_notification(int p_what) {
    if (p_what == NOTIFICATION_READY) {
        _ready();
    }

    if (p_what == NOTIFICATION_DRAW) {
        _draw();
    }

    if (p_what == NOTIFICATION_EXIT_TREE) {
        _exit_tree();
    }
}

void PathDebugDrawer::_ready(){
    set_process_input(true);
}

void PathDebugDrawer::_draw() {
    if (path_finder.is_null()) 
        return;

    Vector2i grid_size = path_finder->get_size();
    Vector2 cell_size = path_finder->get_cell_size();

    for (int x = 0; x < grid_size.x; x++) {
        for (int y = 0; y < grid_size.y; y++) {
            Vector2i cell(x, y);
            cell -= grid_size /2;
            Vector2 world_pos = path_finder->cell_to_world_gd(cell);
            Vector2 top_left = world_pos - cell_size * 0.5;

            Color c = path_finder->is_cell_solid(cell) ? Color(1,0,0,0.5) : Color(0,1,0,0.1);
            draw_rect(Rect2(top_left, cell_size), c, true);
            draw_rect(Rect2(top_left, cell_size), Color(0,0,0), false);
        }
    }

    if (start_set) 
        draw_circle(start, 8, Color(0,1,1));
    if (end_set) 
        draw_circle(end, 8, Color(1,1,0));

    if (path.size() > 1) 
        draw_polyline(path, Color(0,0,1), 3.0, true);
}

void PathDebugDrawer::_exit_tree() {
    if(path_finder.is_valid()){
        path_finder->clear_drawer();
    }
}

void PathDebugDrawer::input(const Ref<InputEvent> &p_event) {
    if (path_finder.is_null()) return;

    Ref<InputEventMouseButton> mb = p_event;
    Ref<InputEventMouseMotion> mm = p_event;

    if (mb.is_valid()) {
        Vector2 mouse_pos = get_global_mouse_position();

        if (mb->is_pressed()) {
            if (mb->get_button_index() == MouseButton::LEFT) {
                if (!start_set) {
                    start = mouse_pos;
                    start_set = true;
                } else if (_is_near(mouse_pos, start)) {
                    dragging = DRAG_START;
                }
            } else if (mb->get_button_index() == MouseButton::RIGHT) {
                if (!end_set) {
                    end = mouse_pos;
                    end_set = true;
                } else if (_is_near(mouse_pos, end)) {
                    dragging = DRAG_END;
                }
            }

        } else {
            dragging = NONE;
        }

        _update_path();
    }

    if (mm.is_valid() && dragging != NONE) {
        Vector2 mouse_pos = get_global_mouse_position();
        if (dragging == DRAG_START) start = mouse_pos;
        else if (dragging == DRAG_END) end = mouse_pos;

        _update_path();
    }
}

void PathDebugDrawer::set_path_finder(const Ref<SpxPathFinder> &p_path_finder) {
    path_finder = p_path_finder;
    queue_redraw();
}

void PathDebugDrawer::set_path(const PackedVector2Array &p_path) {
    path = p_path;
    queue_redraw();
}
