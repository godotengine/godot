/*************************************************************************/
/*  engine.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "engine.h"

#include "version.h"
#include "version_hash.gen.h"

void Engine::set_iterations_per_second(int p_ips) {

	ips = p_ips;
}
int Engine::get_iterations_per_second() const {

	return ips;
}

void Engine::set_target_fps(int p_fps) {
	_target_fps = p_fps > 0 ? p_fps : 0;
}

float Engine::get_target_fps() const {
	return _target_fps;
}

uint64_t Engine::get_frames_drawn() {

	return frames_drawn;
}

void Engine::set_frame_delay(uint32_t p_msec) {

	_frame_delay = p_msec;
}

uint32_t Engine::get_frame_delay() const {

	return _frame_delay;
}

void Engine::set_time_scale(float p_scale) {

	_time_scale = p_scale;
}

float Engine::get_time_scale() const {

	return _time_scale;
}

Dictionary Engine::get_version_info() const {

	Dictionary dict;
	dict["major"] = VERSION_MAJOR;
	dict["minor"] = VERSION_MINOR;
#ifdef VERSION_PATCH
	dict["patch"] = VERSION_PATCH;
#else
	dict["patch"] = 0;
#endif
	dict["status"] = VERSION_STATUS;
	dict["build"] = VERSION_BUILD;
	dict["year"] = VERSION_YEAR;

	String hash = VERSION_HASH;
	dict["hash"] = hash.length() == 0 ? String("unknown") : hash;

	String stringver = String(dict["major"]) + "." + String(dict["minor"]);
	if ((int)dict["patch"] != 0)
		stringver += "." + String(dict["patch"]);
	stringver += "-" + String(dict["status"]) + " (" + String(dict["build"]) + ")";
	dict["string"] = stringver;

	return dict;
}

void Engine::add_singleton(const Singleton &p_singleton) {

	singletons.push_back(p_singleton);
	singleton_ptrs[p_singleton.name] = p_singleton.ptr;
}

Object *Engine::get_singleton_object(const String &p_name) const {

	const Map<StringName, Object *>::Element *E = singleton_ptrs.find(p_name);
	ERR_EXPLAIN("Failed to retrieve non-existent singleton '" + p_name + "'");
	ERR_FAIL_COND_V(!E, NULL);
	return E->get();
};

bool Engine::has_singleton(const String &p_name) const {

	return singleton_ptrs.has(p_name);
};

void Engine::get_singletons(List<Singleton> *p_singletons) {

	for (List<Singleton>::Element *E = singletons.front(); E; E = E->next())
		p_singletons->push_back(E->get());
}

Engine *Engine::singleton = NULL;

Engine *Engine::get_singleton() {
	return singleton;
}

Engine::Engine() {

	singleton = this;
	frames_drawn = 0;
	ips = 60;
	_frame_delay = 0;
	_fps = 1;
	_target_fps = 0;
	_time_scale = 1.0;
	_pixel_snap = false;
	_physics_frames = 0;
	_idle_frames = 0;
	_in_physics = false;
	_frame_ticks = 0;
	_frame_step = 0;
	editor_hint = false;
}
