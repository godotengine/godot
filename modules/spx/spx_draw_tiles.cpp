/**************************************************************************/
/*  spx_draw_tiles.cpp                                                    */
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

#include "core/os/keyboard.h"
#include "servers/physics_server_2d.h"
#include "scene/2d/sprite_2d.h"
#include "scene/2d/physics/animatable_body_2d.h"
#include "scene/2d/physics/collision_shape_2d.h"
#include "scene/resources/2d/rectangle_shape_2d.h"
#include "core/io/resource_loader.h"
#include "scene/resources/world_2d.h"
#include "scene/gui/color_rect.h"
#include "spx_draw_tiles.h"
#include "spx_engine.h"
#include "spx_ext_mgr.h"
#include "spx_res_mgr.h"
#include "spx_base_mgr.h"

void LayerRenderer::_draw_axis(Node2D *parent_node, const DrawContext &ctx) {
	Vector2 origin = ctx.layer_pos;

	float axis_len = 100.0;
    float flipped = ctx.axis_flipped ? -1 : 1;
	parent_node->draw_line(origin, origin + Vector2(axis_len, 0), Color(1, 0, 0), ctx.axis_width);
	parent_node->draw_line(origin, origin + Vector2(0, axis_len * flipped), Color(0, 1, 0), ctx.axis_width);

	Color origin_color = ctx.axis_dragging ? Color(1, 0, 0) : Color(1, 1, 0);
    float axis_width = 3 * ctx.axis_width;

	if (ctx.axis_dragging) {
		float cross_size = 6.0;
		Color cross_color(1, 1, 0);
		parent_node->draw_line(origin - Vector2(cross_size, 0), origin + Vector2(cross_size, 0), cross_color, axis_width);
		parent_node->draw_line(origin - Vector2(0, cross_size), origin + Vector2(0, cross_size), cross_color, axis_width);
	} 
    else {
		parent_node->draw_circle(origin, axis_width, origin_color);
	}
}

void LayerRenderer::_draw_grid(Node2D *parent_node, const DrawContext &ctx, Vector2 hover_pos) {
    _draw_used_rect(parent_node, ctx);
    _draw_guide_rect(parent_node, ctx, hover_pos);
}

void LayerRenderer::_draw_used_rect(Node2D *parent_node, const DrawContext &ctx) {
    Rect2 used_rect = ctx.map_layer->get_used_rect();
    int x_start = used_rect.position.x - ctx.guide_rect_radius;
    int y_start = used_rect.position.y - ctx.guide_rect_radius;
    int x_end = used_rect.position.x + used_rect.size.x + ctx.guide_rect_radius;
    int y_end = used_rect.position.y + used_rect.size.y + ctx.guide_rect_radius;

	for (int x = x_start; x < x_end; x++) {
		for (int y = y_start; y < y_end; y++) {
			Vector2 local_pos = ctx.map_layer->map_to_local(Vector2i(x, y)) - ctx.cell_size / 2.0;
			parent_node->draw_rect(Rect2(local_pos + ctx.layer_pos, ctx.cell_size), ctx.grid_color, false);
		}
	}
}

void LayerRenderer::_draw_guide_rect(Node2D *parent_node, const DrawContext &ctx, Vector2 hover_pos) {
    int radius = ctx.guide_rect_radius;
    for(int x = -radius; x <= radius; x++){
        for(int y = -radius; y <= radius; y++) {
            auto pos = hover_pos + Vector2(x * ctx.cell_size.x, y * ctx.cell_size.y);
            parent_node->draw_rect(Rect2(pos + ctx.layer_pos, ctx.cell_size), ctx.grid_color, false); 
        }
    }
}

void LayerRenderer::_draw_preview_texture(Node2D *parent_node, const DrawContext &ctx, Vector2 hover_pos) {
	parent_node->draw_rect(Rect2(hover_pos + ctx.layer_pos, ctx.cell_size), Color(1.0, 1.0, 0.0, 0.3), false);

	if (!ctx.axis_dragging && ctx.current_texture.is_valid()) {
		Rect2 preview_rect(hover_pos + ctx.layer_pos, ctx.cell_size);
		Color modulate(1.0, 1.0, 1.0, 0.5);
		parent_node->draw_texture_rect(ctx.current_texture, preview_rect, false, modulate);
	}
}

void LayerRenderer::draw(Node2D *parent_node, const DrawContext &ctx) {
    Vector2 local_mouse = ctx.map_layer->to_local(ctx.mouse_pos);
	Vector2i hover_coords = ctx.map_layer->local_to_map(local_mouse);
	Vector2 hover_pos = ctx.map_layer->map_to_local(hover_coords) - ctx.cell_size / 2.0;

    _draw_axis(parent_node, ctx);
	_draw_grid(parent_node, ctx, hover_pos);
	_draw_preview_texture(parent_node, ctx, hover_pos);
}

void SpxDrawTiles::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_layer_index", "index"), &SpxDrawTiles::set_layer_index);
    ClassDB::bind_method(D_METHOD("set_texture", "texture", "with_collision"), &SpxDrawTiles::set_texture);
    ClassDB::bind_method(D_METHOD("set_tile_size", "size"), &SpxDrawTiles::set_tile_size);
    ClassDB::bind_method(D_METHOD("undo"), &SpxDrawTiles::undo);
    ClassDB::bind_method(D_METHOD("redo"), &SpxDrawTiles::redo);
    ClassDB::bind_method(D_METHOD("place_or_erase_tile", "pos", "erase"), &SpxDrawTiles::place_or_erase_tile);
    ClassDB::bind_method(D_METHOD("clear_all_layers"), &SpxDrawTiles::clear_all_layers);
    ClassDB::bind_method(D_METHOD("enter_editor_mode"), &SpxDrawTiles::enter_editor_mode);
    ClassDB::bind_method(D_METHOD("exit_editor_mode"), &SpxDrawTiles::exit_editor_mode);
}

void SpxDrawTiles::_notification(int p_what) {
    if (p_what == NOTIFICATION_READY) {
        _ready();
    }

    if (p_what == NOTIFICATION_DRAW) {
        _draw();
    }
}

void SpxDrawTiles::_ready() {
    set_process(true); 
    set_process_input(true);
    set_z_index(1000);

    shared_tile_set.instantiate();
    shared_tile_set->set_tile_size(default_cell_size);
}

void SpxDrawTiles::_draw() {
    if(exit_editor)
        return;
    
    TileMapLayer *layer = _get_layer(current_layer_index);
    if (!layer){
        return;
    }

    DrawContext ctx;
    ctx.set_layer(layer)
       .set_cell_size(default_cell_size)
       .set_grid_color(GRID_COLOR)
       .set_axis_width(AXIS_WIDTH)
       .set_guide_rect_radius(GUIDE_RECT_RADIUS)
       .set_layer_pos(layer->get_position())
       .set_mouse_pos(get_global_mouse_position())
       .set_current_texture(current_texture)
       .set_flipped_axis(axis_flipped)
       .set_axis_dragging(axis_flipped);

    renderer.draw(this, ctx);
}

void SpxDrawTiles::input(const Ref<InputEvent> &p_event) {
    if(exit_editor)
        return;

    TileMapLayer *layer = _get_layer(current_layer_index);
    if (!layer) 
        return;
    
    Ref<InputEventMouseButton> mb = p_event;
    Ref<InputEventMouseMotion> mm = p_event;

    if (mb.is_valid()) {
        if (mb->get_button_index() == MouseButton::LEFT) {
            if (mb->is_pressed()) {
                auto mouse_pos = get_global_mouse_position();
                if(mouse_pos.distance_to(layer->get_global_position()) < drag_threshold){
                    axis_dragging = true;
                    drag_start = mouse_pos;
                    layer_start_pos = layer->get_global_position();
                }else {
                    tile_placing = true;
                }
            }else {
                axis_dragging = false;
                tile_placing = false;
            }

            queue_redraw();
        }
    }

    if (mm.is_valid()) {
        if(axis_dragging) {
            Vector2 delta = get_global_mouse_position() - drag_start;
            layer->set_global_position(layer_start_pos + delta);
        }else if(tile_placing) {
            place_or_erase_tile(get_global_mouse_position(), Input::get_singleton()->is_key_pressed(Key::SHIFT));
        }

        queue_redraw();
    }
}

// spx interface
void SpxDrawTiles::set_layer_index_spx(GdInt index) {
    axis_flipped = true;
    set_layer_index(index);
}

void SpxDrawTiles::place_tiles_spx(GdArray positions, GdString texture_path) {
    place_tiles_spx(positions, texture_path, current_layer_index);
}

void SpxDrawTiles::place_tiles_spx(GdArray positions, GdString texture_path, GdInt index) {
    set_layer_index(index);
    set_tile_texture_spx(texture_path, true);
    _place_tiles_bulk_spx(positions);
}

void SpxDrawTiles::place_tile_spx(GdVec2 pos, GdString texture_path) {
    place_tile_spx(pos, texture_path, current_layer_index);
}

void SpxDrawTiles::place_tile_spx(GdVec2 pos, GdString texture_path, GdInt index) {
    set_layer_index(index);
    set_tile_texture_spx(texture_path, true);
    _place_tile_spx(pos);
}

void SpxDrawTiles::erase_tile_spx(GdVec2 pos, GdInt layer_index) {
    auto flipped_pos = flip_y(pos);

    auto erase_at_layer = [&](TileMapLayer* layer) {
        if (!layer) return;

        Vector2 local_pos = layer->to_local(flipped_pos);
        Vector2i coords = layer->local_to_map(local_pos);
        layer->erase_cell(coords);
    };

    if(layer_index != -1){
        if(index_layer_map.has(layer_index)){
            erase_at_layer(index_layer_map[layer_index]);
        }
        return;
    }

    for (auto& item : index_layer_map) {
        erase_at_layer(item.value);
    }
}

GdString SpxDrawTiles::get_tile_spx(GdVec2 pos, GdInt layer_index) {
    if(layer_index == -1){
        layer_index = max_layer_index;
    }

    if(index_layer_map.has(layer_index)){
        auto layer = index_layer_map[layer_index];
        Vector2 local_pos = layer->to_local(flip_y(pos));
        Vector2i coords = layer->local_to_map(local_pos);
        return SpxReturnStr(_get_tile_texture_path(layer, coords));
    }

    return SpxReturnStr("");
    
}

GdString SpxDrawTiles::get_tile_spx(GdVec2 pos) {
	return get_tile_spx(pos, current_layer_index);
}

void SpxDrawTiles::set_tile_texture_spx(GdString texture_path, GdBool with_collision) {
	String path = SpxStr(texture_path);
	Ref<Texture2D> tex;
    if(path_cached_textures_bimap.has_key(path)){
        tex = path_cached_textures_bimap.get_value(path);
    } else {
        tex = resMgr->load_texture(path, true);
        if(!tex.is_null() && tex.is_valid())
            path_cached_textures_bimap.insert(path, tex);
    }                   

    set_texture(tex, with_collision);
}

void SpxDrawTiles::erase_tile_spx(GdVec2 pos) {
    place_or_erase_tile(flip_y(pos), true);
}

void SpxDrawTiles::_place_tiles_bulk_spx(GdArray positions) {
    if (!positions){
        return;
    }

    TileMapLayer *layer = _get_layer(current_layer_index);
    if (!layer){
        return;
    }

    int source_id = _get_or_create_source_id(current_texture);
    Vector2i atlas_coord(0,0);

    auto len = positions->size;
    for (int i = 0; i + 1 < len; i += 2) {
        auto x = *(SpxBaseMgr::get_array<float>(positions, i));
        auto y = *(SpxBaseMgr::get_array<float>(positions, i + 1));

        Vector2 pos = {x, y};
        Vector2 local_pos = layer->to_local(flip_y(pos));
        Vector2i coords = layer->local_to_map(local_pos);

        layer->set_cell(coords, source_id, atlas_coord, 0);
    }
}

void SpxDrawTiles::_place_tile_spx(GdVec2 pos) {
    place_or_erase_tile(flip_y(pos), false);
}

void SpxDrawTiles::set_layer_index(int index) {
    current_layer_index = index;
    _get_or_create_layer(index);
    queue_redraw();
}

void SpxDrawTiles::set_layer_offset_spx(int layer_index, Vector2 offset) {
    TileMapLayer *layer = _get_layer(layer_index);
    if (!layer) {
        return;
    }

    layer->set_position(flip_y(offset));
}

Vector2 SpxDrawTiles::get_layer_offset_spx(int layer_index) {
    TileMapLayer *layer = _get_layer(layer_index);
    if (!layer) {
        return Vector2();
    }

    auto pos = layer->get_position();
    return flip_y(pos);
}

void SpxDrawTiles::set_texture(Ref<Texture2D> texture, bool with_collision) {
    if (texture.is_null()) {
        print_error("Tile texture is null!");
        return;
    }

    current_texture = _get_or_create_scaled_texture(texture);
    _get_or_create_source_id(current_texture, with_collision);
    queue_redraw();
}

void SpxDrawTiles::place_tile(TileMapLayer* layer, Vector2i coords) {
    if (!current_texture.is_valid()){
        return;
    }

    int source_id = _get_or_create_source_id(current_texture);
    if (source_id != TileSet::INVALID_SOURCE) {
        layer->set_cell(coords, source_id, default_atlas_coord, 0);
        TileAction action{current_layer_index, coords, true, source_id, default_atlas_coord, 0};
        undo_stack.push_back(action);
        redo_stack.clear();
        queue_redraw();
    }
}

void SpxDrawTiles::erase_tile(TileMapLayer* layer, Vector2i coords) {
    int source_id = layer->get_cell_source_id(coords);
    if (source_id != TileSet::INVALID_SOURCE) {
        TileAction action{current_layer_index, coords, false, source_id, default_atlas_coord, 0};
        layer->erase_cell(coords);
        undo_stack.push_back(action);
        redo_stack.clear();
        queue_redraw();
    }
}

void SpxDrawTiles::place_or_erase_tile(Vector2 pos, bool erase) {
    TileMapLayer *layer = _get_layer(current_layer_index);
    if(!layer){
        return;
    }

    Vector2 local_pos = layer->to_local(pos);
    Vector2i coords = layer->local_to_map(local_pos);

    if (erase) {
        erase_tile(layer, coords);
    } else {
        place_tile(layer, coords);
    }
}

void SpxDrawTiles::undo() {
    if(exit_editor)
        return;

    if (undo_stack.is_empty()) 
        return;

    TileAction action = undo_stack[undo_stack.size() - 1];
    if(!_apply_action(action, true)) {
        return;
    }
        
    undo_stack.remove_at(undo_stack.size() - 1);
    redo_stack.push_back(action);
}

void SpxDrawTiles::redo() {
    if(exit_editor)
        return;

    if (redo_stack.is_empty()) 
        return;

    TileAction action = redo_stack[redo_stack.size() - 1];
    if(!_apply_action(action, false)) {
        return;
    }

    redo_stack.remove_at(redo_stack.size() - 1);
    undo_stack.push_back(action);
}

void SpxDrawTiles::clear_all_layers() {
    _destroy_layers();
    _clear_cache();
    queue_redraw();
}

TileMapLayer* SpxDrawTiles::_get_or_create_layer(int layer_index) {
    TileMapLayer *layer = _get_layer(layer_index);
    if (!layer) {
        layer = _create_layer(layer_index);
    }

    return layer;
}

TileMapLayer* SpxDrawTiles::_get_layer(int layer_index) {
    String layer_name = UNIQUE_LAYER_PREFIX + itos(layer_index);
    auto layer_node = get_node_or_null(layer_name);                 

    return layer_node ? Object::cast_to<TileMapLayer>(layer_node) : nullptr;
}

TileMapLayer *SpxDrawTiles::_create_layer(int layer_index) {
    TileMapLayer *layer = memnew(TileMapLayer);
    layer->set_tile_set(shared_tile_set);
    layer->set_name(UNIQUE_LAYER_PREFIX + itos(layer_index));
    layer->set_z_index(layer_index);
    layer->set_z_as_relative(false);
    layer->set_navigation_enabled(false);
    index_layer_map[layer_index] = layer;
    max_layer_index = MAX(layer_index, max_layer_index);
    add_child(layer);
    return layer;
}

int SpxDrawTiles::_get_or_create_source_id(Ref<Texture2D> scaled_texture, bool with_collision) {
    if (!scaled_texture.is_valid()) {
        print_error("Invalid texture!");
        return TileSet::INVALID_SOURCE;
    }

    if (scaled_texture_source_ids_bimap.has_key(scaled_texture)) {
        return scaled_texture_source_ids_bimap.get_value(scaled_texture);
    }

    if (default_cell_size.x > scaled_texture->get_size().x || default_cell_size.y > scaled_texture->get_size().y) {
        print_error("Tile size exceeds texture size!");
        return TileSet::INVALID_SOURCE;
    }

    int id = next_source_id++;
    scaled_texture_source_ids_bimap.insert(scaled_texture, id);
   
    shared_tile_set->add_physics_layer(0);
    shared_tile_set->set_physics_layer_collision_layer(0, 0xFFFF);
    shared_tile_set->set_physics_layer_collision_mask(0, 0xFFFF);
 
    Ref<TileSetAtlasSource> atlas_source;
    atlas_source.instantiate();
    atlas_source->set_texture(scaled_texture);
    atlas_source->set_texture_region_size(default_cell_size);
    if(!_create_tile(atlas_source, default_atlas_coord, with_collision)){
        print_error("Failed to create tile in atlas source!");
        return TileSet::INVALID_SOURCE;
    }

    shared_tile_set->add_source(atlas_source, id);
    source_id_collision_map[id] = with_collision;
    
    return id;
}

bool SpxDrawTiles::_create_tile(Ref<TileSetAtlasSource> atlas_source, const Vector2i &tile_coords, bool with_collision) {
    atlas_source->create_tile(tile_coords);
    auto tile_data = atlas_source->get_tile_data(tile_coords, 0);
    if (!tile_data) 
        return false;

    if (!with_collision) 
        return true;

    atlas_source->add_physics_layer(0);
    auto halfSize = default_cell_size.x / 2;
    Vector<Vector2> collision_rect = {
        Vector2(-halfSize, -halfSize),
        Vector2(halfSize, -halfSize),
        Vector2(halfSize,halfSize),
        Vector2(-halfSize, halfSize)
    };

    tile_data->add_collision_polygon(0);

    tile_data->set_collision_polygon_points(0, 0, collision_rect);

    return true;
}
Ref<ImageTexture> SpxDrawTiles::_get_or_create_scaled_texture(Ref<Texture2D> texture) {
	if (texture.is_null()) {
		return Ref<ImageTexture>();
	}

	if (texture_scaled_cache_map.has(texture)) {
        return texture_scaled_cache_map[texture];
    }

    Ref<Image> img = texture->get_image();
    //img->convert(Image::FORMAT_RGBA8);
    img->resize(default_cell_size.x, default_cell_size.y, Image::INTERPOLATE_LANCZOS);
       
    Ref<ImageTexture> scaled_tex;
    scaled_tex.instantiate();
    scaled_tex = scaled_tex->create_from_image(img);
    texture_scaled_cache_map[texture] = scaled_tex;

    if(path_cached_textures_bimap.has_value(texture)){
        scaled_texture_path_map[scaled_tex] = path_cached_textures_bimap.get_key(texture);
    }

    return scaled_tex;
}

String SpxDrawTiles::_get_tile_texture_path(TileMapLayer *layer, const Vector2i &pos) {
	if (!layer) return "";

    int source_id = layer->get_cell_source_id(pos);

    if(source_id != TileSet::INVALID_SOURCE){
        if(scaled_texture_source_ids_bimap.has_value(source_id)){
            auto tex = scaled_texture_source_ids_bimap.get_key(source_id);
            return scaled_texture_path_map[tex];
        }
    }

    return "";
}

void SpxDrawTiles::_destroy_layers(){
    for (int i = get_child_count() - 1; i >= 0; i--) {
        Node* child = get_child(i);
        TileMapLayer* layer = Object::cast_to<TileMapLayer>(child);
        if (layer) {
            remove_child(layer);
            layer->queue_free();
            layer = nullptr;
        }
    }
}

void SpxDrawTiles::_clear_cache() {
    undo_stack.clear();
    redo_stack.clear();

    path_cached_textures_bimap.clear();
    scaled_texture_source_ids_bimap.clear();
    texture_scaled_cache_map.clear();
    scaled_texture_path_map.clear();
    index_layer_map.clear();

    current_texture = nullptr;
    next_source_id = 1;
    current_layer_index = 0;
    max_layer_index = -1;
}

bool SpxDrawTiles::_apply_action(const TileAction &action, bool inverse) {
    TileMapLayer *layer = _get_layer(action.layer_index);
    if (!layer) 
        return false;

    bool place = inverse ? !action.placed : action.placed;

    if (place) {
        layer->set_cell(action.coords, action.source_id, action.atlas_coord, action.alternative_tile);
    } else {
        layer->erase_cell(action.coords);
    }

    queue_redraw();
    return true;
}