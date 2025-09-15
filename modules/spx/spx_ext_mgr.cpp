/**************************************************************************/
/*  spx_ext_mgr.cpp                                                    */
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

#include "spx_ext_mgr.h"
#include "core/input/input_event.h"
#include "core/math/color.h"
#include "gdextension_spx_ext.h"
#include "scene/2d/line_2d.h"
#include "scene/2d/sprite_2d.h"
#include "scene/2d/polygon_2d.h"
#include "spx.h"
#include "spx_engine.h"
#include "spx_pen.h"
#include "spx_res_mgr.h"
#include "spx_sprite.h"
#include "spx_draw_tiles.h"
#include <cmath>

#define resMgr SpxEngine::get_singleton()->get_res()

#define check_and_get_pen_v()                                              \
	auto pen = _get_pen(obj);                                              \
	if (pen == nullptr) {                                                  \
		print_error("try to get property of a null pen gid=" + itos(obj)); \
		return;                                                            \
	}

Mutex SpxExtMgr::lock;

void SpxExtMgr::on_awake() {
	SpxBaseMgr::on_awake();
	pen_root = memnew(Node2D);
	pen_root->set_name("pen_root");
	get_spx_root()->add_child(pen_root);
	
	debug_root = memnew(Node2D);
	debug_root->set_name("debug_root");
	get_spx_root()->add_child(debug_root);

	pure_sprite_root = memnew(Node2D);
	pure_sprite_root->set_name("pure_sprite_root");
	get_spx_root()->add_child(pure_sprite_root);
	
}

void SpxExtMgr::on_start() {
	SpxBaseMgr::on_start();
}

void SpxExtMgr::on_update(float delta) {
	SpxBaseMgr::on_update(delta);
	
	_clear_debug_shapes();
	
	lock.lock();
	for (auto mgr : id_pens) {
		mgr.value->on_update(delta);
	}
	lock.unlock();
}

void SpxExtMgr::on_destroy() {
	lock.lock();
	for (auto mgr : id_pens) {
		mgr.value->on_destroy();
	}
	id_pens.clear();
	if (pen_root) {
		pen_root->queue_free();
		pen_root = nullptr;
	}
	
	_clear_debug_shapes();
	if (debug_root) {
		debug_root->queue_free();
		debug_root = nullptr;
	}
	lock.unlock();
	clear_pure_sprites();
	pure_sprite_root = nullptr;
	SpxBaseMgr::on_destroy();
}

void SpxExtMgr::request_exit(GdInt exit_code) {
	auto callback = SpxEngine::get_singleton()->get_on_runtime_exit();
	if (callback != nullptr) {
		callback(exit_code);
	}	
	
	SpxEngine::get_singleton()->on_exit(exit_code);
	get_tree()->quit(exit_code);
}

void SpxExtMgr::on_runtime_panic(GdString msg) {
	auto msg_str = SpxStr(msg);
	auto callback = SpxEngine::get_singleton()->get_on_runtime_panic();
	if (callback != nullptr) {
		auto str = SpxReturnStr(msg_str);
		callback(str);
	}
}


SpxPen *SpxExtMgr::_get_pen(GdObj obj) {
	if (id_pens.has(obj)) {
		return id_pens[obj];
	}
	return nullptr;
}

void SpxExtMgr::destroy_all_pens() {
	lock.lock();
	for (auto mgr : id_pens) {
		mgr.value->erase_all();
	}
	lock.unlock();
}

GdObj SpxExtMgr::create_pen() {
	auto id = get_unique_id();
	lock.lock();
	SpxPen *node = memnew(SpxPen);
	node->on_create(id, pen_root);
	id_pens[id] = node;
	lock.unlock();
	return id;
}

void SpxExtMgr::destroy_pen(GdObj obj) {
	lock.lock();
	auto pen = _get_pen(obj);                                              \
	if (pen != nullptr) {
		id_pens.erase(obj);
		pen->on_destroy();
	}
	lock.unlock();
}

void SpxExtMgr::move_pen_to(GdObj obj, GdVec2 position) {
	check_and_get_pen_v()
	pen->move_to(position);
}
void SpxExtMgr::pen_stamp(GdObj obj) {
	check_and_get_pen_v()
	pen->stamp();
}
void SpxExtMgr::pen_down(GdObj obj, GdBool move_by_mouse) {
	check_and_get_pen_v()
	pen->on_down(move_by_mouse);
}
void SpxExtMgr::pen_up(GdObj obj) {
	check_and_get_pen_v()
	pen->on_up();
}
void SpxExtMgr::set_pen_color_to(GdObj obj, GdColor color) {
	check_and_get_pen_v()
	pen->set_color_to(color);
}
void SpxExtMgr::change_pen_by(GdObj obj, GdInt property, GdFloat amount) {
	check_and_get_pen_v()
	pen->change_by(property, amount);
}
void SpxExtMgr::set_pen_to(GdObj obj, GdInt property, GdFloat value) {
	check_and_get_pen_v()
	pen->set_to(property, value);
}
void SpxExtMgr::change_pen_size_by(GdObj obj, GdFloat amount) {
	check_and_get_pen_v()
	pen->change_size_by(amount);
}
void SpxExtMgr::set_pen_size_to(GdObj obj, GdFloat size) {
	check_and_get_pen_v()
	pen->set_size_to(size);
}
void SpxExtMgr::set_pen_stamp_texture(GdObj obj, GdString texture_path) {
	check_and_get_pen_v()
	pen->set_stamp_texture(texture_path);
}

// Pause API implementations - delegate to Spx layer
void SpxExtMgr::pause() {
	Spx::pause();
}

void SpxExtMgr::resume() {
	Spx::resume();
}

GdBool SpxExtMgr::is_paused() {
	return Spx::is_paused();
}

void SpxExtMgr::next_frame() {
	Spx::next_frame();
}

void SpxExtMgr::_clear_debug_shapes() {
	for (const DebugShape& shape : debug_shapes) {
		if (shape.node && shape.node->is_inside_tree()) {
			shape.node->queue_free();
		}
	}
	debug_shapes.clear();
}

void SpxExtMgr::debug_draw_circle(GdVec2 pos, GdFloat radius, GdColor color) {
	if (!debug_root) {
		return;
	}

	pos.y = -pos.y;
	Line2D* circle = memnew(Line2D);
	circle->set_default_color(color);
	circle->set_width(2.0f);
	
	PackedVector2Array points;
	int segments = MAX(16, (int)(radius * 0.8f));
	for (int i = 0; i <= segments; i++) {
		float angle = i * 6.283185f / segments;
		points.append(Vector2(std::cos(angle) * radius, std::sin(angle) * radius));
	}
	circle->set_points(points);
	circle->set_position(pos);
	
	debug_root->add_child(circle);
	
	DebugShape shape;
	shape.type = DebugShape::CIRCLE;
	shape.position = pos;
	shape.radius = radius;
	shape.color = color;
	shape.node = circle;
	debug_shapes.push_back(shape);
}

void SpxExtMgr::debug_draw_rect(GdVec2 pos, GdVec2 size, GdColor color) {
	if (!debug_root) {
		return;
	}

	Line2D* rect = memnew(Line2D);
	rect->set_default_color(color);
	rect->set_width(2.0f);
	
	pos.y = -pos.y;
	size = size * 0.5;
	PackedVector2Array points;
	points.append(Vector2(-size.x, -size.y));
	points.append(Vector2(size.x, -size.y));
	points.append(Vector2(size.x, size.y));
	points.append(Vector2(-size.x, size.y));
	points.append(Vector2(-size.x, -size.y));
	rect->set_points(points);
	rect->set_position(pos);
	
	debug_root->add_child(rect);
	
	DebugShape shape;
	shape.type = DebugShape::RECT;
	shape.position = pos;
	shape.size = size;
	shape.color = color;
	shape.node = rect;
	debug_shapes.push_back(shape);
}


void SpxExtMgr::open_draw_tiles() {
	open_draw_tiles_with_size(16);// default tile_size = 16
}

void SpxExtMgr::set_layer_offset(GdInt index, GdVec2 offset){
	if (draw_tiles == nullptr) {
        print_error("The draw tiles not exist");
        return;
    }
	draw_tiles->set_layer_offset(index, offset);

}
GdVec2 SpxExtMgr::get_layer_offset(GdInt index){
	if (draw_tiles == nullptr) {
        print_error("The draw tiles not exist");
        return GdVec2();
    }
	return draw_tiles->get_layer_offset(index);
}
void SpxExtMgr::open_draw_tiles_with_size(GdInt tile_size) {
    if (draw_tiles != nullptr) {
        print_error("The draw tiles node already created");
        return;
    }
    draw_tiles = memnew(SpxDrawTiles);
	draw_tiles->set_tile_size(tile_size);
    get_spx_root()->add_child(draw_tiles);
}

void SpxExtMgr::set_layer_index(GdInt index) {
	with_draw_tiles([&](){
		draw_tiles->set_sprite_index(index);
	});
}
void SpxExtMgr::set_tile(GdString texture_path, GdBool with_collision) {
	with_draw_tiles([&](){
		draw_tiles->set_sprite_texture(texture_path, with_collision);
	});
}

void SpxExtMgr::place_tiles(GdArray positions, GdString texture_path) {
	with_draw_tiles([&](){
		draw_tiles->place_sprites(positions, texture_path);
	});
}
void SpxExtMgr::place_tiles_with_layer(GdArray positions, GdString texture_path, GdInt layer_index) {
	with_draw_tiles([&](){
		draw_tiles->place_sprites(positions, texture_path, layer_index);
	});
}

void SpxExtMgr::place_tile(GdVec2 pos, GdString texture_path) {
	with_draw_tiles([&](){
		draw_tiles->place_sprite(pos, texture_path);
	});
}

void SpxExtMgr::place_tile_with_layer(GdVec2 pos, GdString texture_path, GdInt layer_index) {
	with_draw_tiles([&](){
		draw_tiles->place_sprite(pos, texture_path, layer_index);
	});
}

void SpxExtMgr::erase_tile(GdVec2 pos) {
	with_draw_tiles([&](){
		draw_tiles->erase_sprite(pos);
	});
}

GdArray SpxExtMgr::get_layer_point_path(GdVec2 p_from, GdVec2 p_to){
	if(draw_tiles != nullptr){
		return draw_tiles->get_layer_point_path(p_from, p_to);
	}

	return nullptr;
}

void SpxExtMgr::close_draw_tiles() {
	if (draw_tiles != nullptr) {
		draw_tiles->queue_free();
		draw_tiles = nullptr;
    }
}

void SpxExtMgr::exit_tilemap_editor_mode() {
	if (draw_tiles != nullptr) {
		draw_tiles->exit_editor_mode();
		draw_tiles = nullptr;
    }
}
void SpxExtMgr::clear_pure_sprites(){
	pure_sprite_root->queue_free();
}

void SpxExtMgr::create_pure_sprite(GdString texture_path, GdVec2 pos, GdInt zindex){
	if (pure_sprite_root == nullptr) {
		return;
	}
	Sprite2D* sprite = memnew(Sprite2D);
	auto path_str = SpxStr(texture_path);

	Ref<Texture2D> texture = nullptr;
	auto is_svg_mode = svgMgr->is_svg_file(path_str);
	if (is_svg_mode){
		int target_scale = 1;
		texture = svgMgr->get_svg_image(path_str, target_scale);
	}else{
		texture = resMgr->load_texture(path_str, true);
	}
	sprite->set_texture(texture);
	sprite->set_position(Vector2(pos.x,-pos.y));
	sprite->set_name(path_str.get_file());
	pure_sprite_root->add_child(sprite);
	sprite->set_z_index(zindex);
}

