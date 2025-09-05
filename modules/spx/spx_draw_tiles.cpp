#include "spx_draw_tiles.h"
#include "core/os/keyboard.h"
#include "servers/physics_server_2d.h"
#include "scene/2d/sprite_2d.h"
#include "scene/2d/physics/animatable_body_2d.h"
#include "scene/2d/physics/collision_shape_2d.h"
#include "scene/resources/2d/rectangle_shape_2d.h"
#include "core/io/resource_loader.h"
#include "scene/resources/world_2d.h"
#include "scene/gui/color_rect.h"


void LayerRenderer::_draw_axis(Node2D *parent_node, const DrawContext &ctx) {
	Vector2 origin = ctx.layer_pos;

	float axis_len = 100.0;
	parent_node->draw_line(origin, origin + Vector2(axis_len, 0), Color(1, 0, 0), ctx.axis_width);
	parent_node->draw_line(origin, origin + Vector2(0, axis_len), Color(0, 1, 0), ctx.axis_width);

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
    ClassDB::bind_method(D_METHOD("set_texture", "texture", "with_collision"), &SpxDrawTiles::set_texture);
    ClassDB::bind_method(D_METHOD("set_layer_index", "index"), &SpxDrawTiles::set_layer_index);
    ClassDB::bind_method(D_METHOD("place_tile", "coords"), &SpxDrawTiles::place_tile);
    ClassDB::bind_method(D_METHOD("erase_tile", "coords"), &SpxDrawTiles::erase_tile);
    ClassDB::bind_method(D_METHOD("undo"), &SpxDrawTiles::undo);
    ClassDB::bind_method(D_METHOD("redo"), &SpxDrawTiles::redo);
    ClassDB::bind_method(D_METHOD("handle_mouse_click", "pos", "erase"), &SpxDrawTiles::handle_mouse_click);
    ClassDB::bind_method(D_METHOD("clear_all_layers"), &SpxDrawTiles::clear_all_layers);
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
    shared_tile_set.instantiate();
    shared_tile_set->set_tile_size(CELL_SIZE);
    set_z_index(1000);
}

void SpxDrawTiles::_draw() {
    TileMapLayer *layer = _get_layer(current_layer_index);
    if (!layer) 
        return;

    DrawContext ctx;
    ctx.set_layer(layer)
       .set_cell_size(CELL_SIZE)
       .set_grid_color(GRID_COLOR)
       .set_axis_width(AXIS_WIDTH)
       .set_guide_rect_radius(GUIDE_RECT_RADIUS)
       .set_layer_pos(layer->get_position())
       .set_mouse_pos(get_global_mouse_position())
       .set_current_texture(current_texture)
       .set_axis_dragging(axis_dragging);

    renderer.draw(this, ctx);
}

void SpxDrawTiles::input(const Ref<InputEvent> &event) {
    TileMapLayer *layer = _get_layer(current_layer_index);
    if (!layer) 
        return;

    if (const InputEventMouseButton *mb = Object::cast_to<InputEventMouseButton>(event.ptr())) {
        if (mb->get_button_index() == MouseButton::LEFT) {
            if (mb->is_pressed()) {
                auto mouse_pos = get_global_mouse_position();
                if(mouse_pos.distance_to(layer->get_global_position()) < drag_threshold){
                    axis_dragging = true;
                    drag_start = mouse_pos;
                    layer_start_pos = layer->get_global_position();
                }else{
                    tile_placing = true;
                }
            }
             else {
                axis_dragging = false;
                tile_placing = false;
            }

            queue_redraw();
        }
    }

    if (event->is_class("InputEventMouseMotion")) {
        if(axis_dragging) {
            Vector2 delta = get_global_mouse_position() - drag_start;
            layer->set_global_position(layer_start_pos + delta);
        }else if(tile_placing) {
            handle_mouse_click(get_global_mouse_position(), Input::get_singleton()->is_key_pressed(Key::SHIFT));
        }
        queue_redraw();
    }
}

void SpxDrawTiles::set_layer_index(int index) {
    current_layer_index = index;
    _get_or_create_layer(index);
    queue_redraw();
}

void SpxDrawTiles::set_texture(Ref<Texture2D> texture, bool with_collision) {
    if (texture.is_null()) 
        return;

    current_texture = _get_scaled_texture(texture);
    _get_or_create_source_id(current_texture, with_collision);
    queue_redraw();
}

void SpxDrawTiles::place_tile(Vector2i coords) {
    if (!current_texture.is_valid()) 
        return;

    TileMapLayer *layer = _get_or_create_layer(current_layer_index);

    int source_id = _get_or_create_source_id(current_texture);
    Vector2i atlas_coord(0,0);
    layer->set_cell(coords, source_id, atlas_coord, 0);

    TileAction action{current_layer_index, coords, true, source_id, atlas_coord, 0};
    undo_stack.push_back(action);
    redo_stack.clear();
    queue_redraw();
}

void SpxDrawTiles::erase_tile(Vector2i coords) {
    TileMapLayer *layer = _get_layer(current_layer_index);
    if (!layer) 
        return;

    int source_id = layer->get_cell_source_id(coords);
    if ( source_id != TileSet::INVALID_SOURCE) {
        TileAction action{current_layer_index, coords, false, source_id, Vector2i(0,0),0};
        layer->erase_cell(coords);
        undo_stack.push_back(action);
        redo_stack.clear();
        queue_redraw();
    }
}

void SpxDrawTiles::undo() {
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
    if (redo_stack.is_empty()) 
        return;

    TileAction action = redo_stack[redo_stack.size() - 1];
    if(!_apply_action(action, false)) {
        return;
    }

    redo_stack.remove_at(redo_stack.size() - 1);
    undo_stack.push_back(action);
}

void SpxDrawTiles::handle_mouse_click(Vector2 pos, bool erase) {
    TileMapLayer *layer = _get_layer(current_layer_index);
    if(layer == nullptr) 
        return;

    Vector2 local_pos = layer->to_local(pos);
    Vector2i coords = layer->local_to_map(local_pos);

    if (erase) erase_tile(coords);
    else place_tile(coords);
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
    add_child(layer);
	return layer;
}

int SpxDrawTiles::_get_or_create_source_id(Ref<Texture2D> texture, bool with_collision) {
    if (!texture.is_valid()) {
        print_error("Invalid texture!");
        return TileSet::INVALID_SOURCE;
    }

    if (texture_source_ids.has(texture)) {
        return texture_source_ids[texture];
    }

    if (CELL_SIZE.x > texture->get_size().x || CELL_SIZE.y > texture->get_size().y) {
        print_error("Tile size exceeds texture size!");
        return TileSet::INVALID_SOURCE;
    }

    int id = next_source_id++;
    texture_source_ids[texture] = id;
   
    shared_tile_set->add_physics_layer(0);
    shared_tile_set->set_physics_layer_collision_layer(0, 0xFFFF);
    shared_tile_set->set_physics_layer_collision_mask(0, 0xFFFF);
 
    Ref<TileSetAtlasSource> atlas_source;
    atlas_source.instantiate();
    atlas_source->set_texture(texture);
    atlas_source->set_texture_region_size(CELL_SIZE);
    if(!_create_tile(atlas_source, Vector2i(0,0), with_collision)){
        print_error("Failed to create tile in atlas source!");
        return TileSet::INVALID_SOURCE;
    }

    shared_tile_set->add_source(atlas_source, id);
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
    Vector<Vector2> collision_rect = {
        Vector2(0, 0),
        Vector2(CELL_SIZE.x, 0),
        CELL_SIZE,
        Vector2(0, CELL_SIZE.y)
    };

    tile_data->add_collision_polygon(0);
    tile_data->set_collision_polygon_points(0, 0, collision_rect);

    return true;
}
Ref<ImageTexture> SpxDrawTiles::_get_scaled_texture(Ref<Texture2D> texture) {
	if (texture.is_null()) {
		return Ref<ImageTexture>();
	}

	if (texture_scaled_cache.has(texture)) {
        return texture_scaled_cache[texture];
    }

    Ref<Image> img = texture->get_image();
    //img->convert(Image::FORMAT_RGBA8);
    img->resize(CELL_SIZE.x, CELL_SIZE.y, Image::INTERPOLATE_LANCZOS);
       
    Ref<ImageTexture> scaled_tex;
    scaled_tex.instantiate();
    scaled_tex = scaled_tex->create_from_image(img);
    texture_scaled_cache[texture] = scaled_tex;

    return scaled_tex;
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
    texture_source_ids.clear();
    texture_scaled_cache.clear();

    current_texture = nullptr;
    next_source_id = 1;
    current_layer_index = 0;
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