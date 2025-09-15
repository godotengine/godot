/**************************************************************************/
/*  spx_ext_mgr.h                                                      */
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

#ifndef SPX_EXT_MGR_H
#define SPX_EXT_MGR_H

#include "gdextension_spx_ext.h"
#include "scene/2d/node_2d.h"
#include "spx_base_mgr.h"

class SpxPen;
class SpxDrawTiles;

struct DebugShape {
	enum Type {
		CIRCLE,
		RECT,
		LINE
	};
	Type type;
	GdVec2 position;
	GdVec2 size;
	GdFloat radius;
	GdVec2 to_position;
	GdColor color;
	Node2D *node;
};

class SpxExtMgr : SpxBaseMgr {
	SPXCLASS(SpxExtMgr, SpxBaseMgr)
public:
	virtual ~SpxExtMgr() = default;

private:
	RBMap<GdObj, SpxPen *> id_pens;
	Node *pen_root;
	
	Vector<DebugShape> debug_shapes;
	Node2D *debug_root;

	SpxDrawTiles* draw_tiles = nullptr;

	Node *pure_sprite_root;

	static Mutex lock;
private:
	SpxPen *_get_pen(GdObj id);
	void _clear_debug_shapes();

public:
	void on_awake() override;
	void on_start() override;
	void on_destroy() override;
	void on_update(float delta) override;

public:
	// engine API
	void request_exit(GdInt exit_code);
	void on_runtime_panic(GdString msg);
	
	// pause API
	void pause();
	void resume();
	GdBool is_paused();
	void next_frame();

	// obj APIs
	void destroy_all_pens();
	GdObj create_pen();
	void destroy_pen(GdObj obj);
	void pen_stamp(GdObj obj);
	void move_pen_to(GdObj obj, GdVec2 position);
	void pen_down(GdObj obj, GdBool move_by_mouse);
	void pen_up(GdObj obj);
	void set_pen_color_to(GdObj obj, GdColor color);
	void change_pen_by(GdObj obj, GdInt property, GdFloat amount);
	void set_pen_to(GdObj obj, GdInt property, GdFloat value);
	void change_pen_size_by(GdObj obj, GdFloat amount);
	void set_pen_size_to(GdObj obj, GdFloat size);
	void set_pen_stamp_texture(GdObj obj, GdString texture_path);

	// debug
	void debug_draw_circle(GdVec2 pos, GdFloat radius, GdColor color);
	void debug_draw_rect(GdVec2 pos, GdVec2 size, GdColor color);
	void debug_draw_line(GdVec2 from, GdVec2 to, GdColor color);

	// draw tiles 
	void open_draw_tiles_with_size(GdInt tile_size);
	void open_draw_tiles();
	void set_layer_index(GdInt index);
	void set_tile(GdString texture_path, GdBool with_collision);
    void set_layer_offset(GdInt index, GdVec2 offset);
    GdVec2 get_layer_offset(GdInt index);
	void place_tiles(GdArray positions, GdString texture_path);
	void place_tiles_with_layer(GdArray positions, GdString texture_path, GdInt layer_index);
    void place_tile(GdVec2 pos, GdString texture_path);
    void place_tile_with_layer(GdVec2 pos, GdString texture_path, GdInt layer_index);
    void erase_tile(GdVec2 pos);
	void close_draw_tiles();
	GdArray get_layer_point_path(GdVec2 p_from, GdVec2 p_to);
	void exit_tilemap_editor_mode();
	template<typename Func>
    void with_draw_tiles(Func f, const String error_msg = "The draw tiles node is null, first open it!!!") {
        if (draw_tiles == nullptr) {
            print_error(error_msg);
            return;
        }
        f();
    }
	template<typename Func>
	void without_draw_tiles(Func f) {
		if (draw_tiles == nullptr) {
			open_draw_tiles();
		}
		f();
	}

	// create sprites
	void clear_pure_sprites();
	void create_pure_sprite(GdString texture_path, GdVec2 pos, GdInt zindex);
};

#endif // SPX_EXT_MGR_H
