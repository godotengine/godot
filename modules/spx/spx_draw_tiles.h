/**************************************************************************/
/*  spx_draw_tiles.h                                                      */
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

#ifndef SPX_DRAW_TILES_H
#define SPX_DRAW_TILES_H

#include "scene/2d/node_2d.h"
#include "scene/2d/sprite_2d.h"
#include "scene/2d/camera_2d.h"
#include "scene/2d/tile_map.h"
#include "scene/resources/2d/tile_set.h"
#include "core/templates/hash_map.h"
#include "core/math/a_star_grid_2d.h"
#include "spx_sprite.h"


struct DrawContext {
    TileMapLayer *map_layer;

    Vector2i cell_size;
    Color grid_color;
    float axis_width;
    int guide_rect_radius;

    Vector2 layer_pos;
    Vector2 mouse_pos;

    Rect2 used_rect;
    Ref<Texture2D> current_texture;

    bool axis_flipped;
    bool axis_dragging;

    DrawContext& set_layer(TileMapLayer *layer) { map_layer = layer; return *this; }
    DrawContext& set_cell_size(Vector2i size) { cell_size = size; return *this; }
    DrawContext& set_grid_color(Color c) { grid_color = c; return *this; }
    DrawContext& set_axis_width(float w) { axis_width = w; return *this; }
    DrawContext& set_guide_rect_radius(int r) { guide_rect_radius = r; return *this; }

    DrawContext& set_layer_pos(Vector2 pos) { layer_pos = pos; return *this; }
    DrawContext& set_mouse_pos(Vector2 pos) { mouse_pos = pos; return *this; }

    DrawContext& set_current_texture(Ref<Texture2D> tex) { current_texture = tex; return *this; }
    DrawContext& set_flipped_axis(bool flipped) { axis_flipped = flipped; return *this; }
    DrawContext& set_axis_dragging(bool dragging) { axis_dragging = dragging; return *this; }
};


class LayerRenderer{
private:
    void _draw_axis(Node2D *parent_node, const DrawContext &ctx); 
    void _draw_grid(Node2D *parent_node, const DrawContext &ctx, Vector2 hover_pos); 
    void _draw_used_rect(Node2D *parent_node, const DrawContext &ctx); 
    void _draw_guide_rect(Node2D *parent_node, const DrawContext &ctx, Vector2 hover_pos); 
    void _draw_preview_texture(Node2D *parent_node, const DrawContext &ctx, Vector2 hover_pos); 
public:
    LayerRenderer() = default;
    ~LayerRenderer() = default;

    void draw(Node2D *parent_node, const DrawContext & ctx);
};


template<typename K, typename V>
class BiMap {
    HashMap<K, V> forward;
    HashMap<V, K> backward;
public:
    void insert(const K &k, const V &v) {
        forward[k] = v;
        backward[v] = k;
    }

    bool has_key(const K &k) const { return forward.has(k); }
    bool has_value(const V &v) const { return backward.has(v); }

    V get_value(const K &k) const { return forward[k]; }
    K get_key(const V &v) const { return backward[v]; }

    void erase_by_key(const K &k) {
        if (!forward.has(k)) return;
        auto v = forward[k];
        forward.erase(k);
        backward.erase(v);
    }

    void erase_by_value(const V &v) {
        if (!backward.has(v)) return;
        auto k = backward[v];
        backward.erase(v);
        forward.erase(k);
    }

    void clear() {
        forward.clear();
        backward.clear();
    }

    int size() const { return forward.size(); }
    bool empty() const { return forward.is_empty(); }
};


struct TileAction {
    int layer_index;
    Vector2i coords;
    bool placed;
    int source_id;
    Vector2i atlas_coord;
    int alternative_tile;
};


class SpxDrawTiles : public Node2D {
    GDCLASS(SpxDrawTiles, Node2D);

private:
    Ref<TileSet> shared_tile_set;
    Ref<Texture2D> current_texture;

    Vector<TileAction> undo_stack;
    Vector<TileAction> redo_stack;

    BiMap<String, Ref<Texture2D>> path_cached_textures_bimap;
    BiMap<Ref<Texture2D>, int> scaled_texture_source_ids_bimap;
    HashMap<int, bool> source_id_collision_map;
    HashMap<Ref<Texture2D>, Ref<ImageTexture>> texture_scaled_cache_map;
    HashMap<Ref<ImageTexture>, String> scaled_texture_path_map;
    HashMap<int, TileMapLayer*> index_layer_map;
    int max_layer_index = -1;
    int next_source_id = 1;

    const String UNIQUE_LAYER_PREFIX = "spx_draw_tiles_layer_";
    Vector2i default_cell_size{16, 16};
    Vector2i default_atlas_coord{0, 0};
    int current_layer_index = 0;

    bool exit_editor = true;
    bool axis_flipped = false;
    bool axis_dragging = false;
    bool tile_placing = false;
    static constexpr float drag_threshold = 10.0; 
    Vector2 drag_start;
    Vector2 layer_start_pos;

    const Color GRID_COLOR{1.0, 1.0, 0.0, 0.5};
    static constexpr int GUIDE_RECT_RADIUS = 5;
    static constexpr float AXIS_WIDTH = 5;
    LayerRenderer renderer;

protected:
    static void _bind_methods();
    void _notification(int p_what);
    void _ready();
	void _draw();
	void input(const Ref<InputEvent> &p_event) override;

public:
    SpxDrawTiles() = default;
    ~SpxDrawTiles() = default;

    // spx interface
    void set_layer_index_spx(GdInt index);
    void set_tile_texture_spx(GdString texture_path, GdBool with_collision);
    void place_tiles_spx(GdArray positions, GdString texture_path);
    void place_tiles_spx(GdArray positions, GdString texture_path, GdInt layer_index);
    void place_tile_spx(GdVec2 pos, GdString texture_path);
    void place_tile_spx(GdVec2 pos, GdString texture_path, GdInt layer_index);
    void erase_tile_spx(GdVec2 pos, GdInt layer_index);
    void erase_tile_spx(GdVec2 pos);
    GdString get_tile_spx(GdVec2 pos, GdInt layer_index);
    GdString get_tile_spx(GdVec2 pos);

    _FORCE_INLINE_ void set_tile_size(int size = 16){default_cell_size = Vector2(size, size);};
    void set_layer_index(int index);
    void set_layer_offset_spx(int index, Vector2 offset);
    Vector2 get_layer_offset_spx(int index);
    void set_texture(Ref<Texture2D> texture, bool with_collision = true);

    void place_tile(TileMapLayer* layer, Vector2i coords);
    void erase_tile(TileMapLayer* layer, Vector2i coords);
    void place_or_erase_tile(Vector2 pos, bool erase);

    _FORCE_INLINE_ bool has_collision(int source_id){
        if(source_id_collision_map.has(source_id)){       
            return source_id_collision_map[source_id];
        }

        return false;
    }

    void undo();
    void redo();

    void clear_all_layers();

    _FORCE_INLINE_ void enter_editor_mode(){exit_editor = false;}
    _FORCE_INLINE_ void exit_editor_mode(){exit_editor = true; queue_redraw();};

private:
    TileMapLayer* _get_or_create_layer(int layer_index);
    TileMapLayer* _get_layer(int layer_index);
    TileMapLayer* _create_layer(int layer_index);
    int _get_or_create_source_id(Ref<Texture2D> scaled_texture, bool with_collision = true);
    bool _create_tile(Ref<TileSetAtlasSource> atlas_source, const Vector2i &tile_coords, bool with_collision = true);
    Ref<ImageTexture> _get_or_create_scaled_texture(Ref<Texture2D> texture);
    String _get_tile_texture_path(TileMapLayer* layer, const Vector2i& pos);

    void _place_tiles_bulk_spx(GdArray positions);
    void _place_tile_spx(GdVec2 pos);
    _FORCE_INLINE_ Vector2 flip_y(const Vector2 &pos) { return pos * Vector2(1, -1); }

    void _destroy_layers();
    void _clear_cache();

    bool _apply_action(const TileAction &action, bool inverse);
};

#endif // SPX_DRAW_TILES_H
