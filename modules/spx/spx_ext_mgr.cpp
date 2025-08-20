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
#include "scene/2d/line_2d.h"
#include "scene/2d/sprite_2d.h"
#include "spx.h"
#include "spx_engine.h"
#include "spx_pen.h"
#include "spx_res_mgr.h"
#include "spx_sprite.h"

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
}

void SpxExtMgr::on_start() {
	SpxBaseMgr::on_start();
}

void SpxExtMgr::on_update(float delta) {
	SpxBaseMgr::on_update(delta);
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
	lock.unlock();
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
