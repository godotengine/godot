/**************************************************************************/
/*  spx_platform_mgr.cpp                                                     */
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

#include "spx_platform_mgr.h"
#include "core/config/engine.h"
#include "scene/main/window.h"
#include "spx.h"

void SpxPlatformMgr::on_awake() {
	SpxBaseMgr::on_awake();
	persistant_data_dir = ::OS::get_singleton()->get_user_data_dir();
}

void SpxPlatformMgr::set_stretch_mode(GdBool enable) {
    if (auto root = get_root()) {
        auto target_mode = enable
            ? Window::ContentScaleMode::CONTENT_SCALE_MODE_CANVAS_ITEMS
            : Window::ContentScaleMode::CONTENT_SCALE_MODE_DISABLED;

        if (root->get_content_scale_mode() != target_mode) {
            root->set_content_scale_mode(target_mode);
        }
    }
}

void SpxPlatformMgr::set_window_position(GdVec2 pos) {
	DisplayServer::get_singleton()->window_set_position(Size2i(pos.x, pos.y));
}
GdVec2 SpxPlatformMgr::get_window_position() {
	auto pos = DisplayServer::get_singleton()->window_get_position();
	return GdVec2(pos.x, pos.y);
}
void SpxPlatformMgr::set_window_size(GdInt width, GdInt height) {
	DisplayServer::get_singleton()->window_set_size(Size2i(width, height));
}

GdVec2 SpxPlatformMgr::get_window_size() {
	auto size = DisplayServer::get_singleton()->window_get_size_ext();
	return GdVec2(size.x, size.y);
}

void SpxPlatformMgr::set_window_title(GdString title) {
	DisplayServer::get_singleton()->window_set_title(SpxStr(title));
}

GdString SpxPlatformMgr::get_window_title() {
	String title = "";
	return SpxReturnStr(title);
}

void SpxPlatformMgr::set_window_fullscreen(GdBool enable) {
	auto mode = enable ? DisplayServer::WINDOW_MODE_FULLSCREEN : DisplayServer::WINDOW_MODE_WINDOWED;
	DisplayServer::get_singleton()->window_set_mode(mode);
}

GdBool SpxPlatformMgr::is_window_fullscreen() {
	return get_root()->get_mode() == Window::MODE_FULLSCREEN;
}

void SpxPlatformMgr::set_debug_mode(GdBool enable) {
	Spx::set_debug_mode(enable);
}

GdBool SpxPlatformMgr::is_debug_mode() {
	return Spx::debug_mode;
}

void SpxPlatformMgr::set_time_scale(GdFloat time_scale) {
	Engine::get_singleton()->set_time_scale(time_scale);
}

GdFloat SpxPlatformMgr::get_time_scale() {
	return Engine::get_singleton()->get_time_scale();
}

GdString SpxPlatformMgr::get_persistant_data_dir(){
	auto value = _get_persistant_data_dir();
	return SpxReturnStr(value);
}

String SpxPlatformMgr::_get_persistant_data_dir(){
	return persistant_data_dir;
}

void SpxPlatformMgr::_set_persistant_data_dir(String path){
	persistant_data_dir = path;
}
void SpxPlatformMgr::set_persistant_data_dir(GdString path){
	auto path_str = SpxStr(path);
	_set_persistant_data_dir(path_str);
}

GdBool SpxPlatformMgr::is_in_persistant_data_dir(GdString path){
	auto path_str = SpxStr(path);
	return path_str.begins_with(persistant_data_dir);
}
