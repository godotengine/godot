/**************************************************************************/
/*  resource_loader_zig.cpp                                               */
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

#include "resource_loader_zig.h"
#include "script_zig.h"
#include <godot_cpp/classes/file_access.hpp>

static Ref<ResourceFormatLoaderZig> zig_loader;

void ResourceFormatLoaderZig::init() {
	zig_loader.instantiate();
	ResourceLoader::get_singleton()->add_resource_format_loader(zig_loader);
}

void ResourceFormatLoaderZig::deinit() {
	ResourceLoader::get_singleton()->remove_resource_format_loader(zig_loader);
	zig_loader.unref();
}

Variant ResourceFormatLoaderZig::_load(const String &p_path, const String &original_path, bool use_sub_threads, int32_t cache_mode) const {
	Ref<ZigScript> model = memnew(ZigScript);
	model->_set_source_code(FileAccess::get_file_as_string(p_path));
	return model;
}
PackedStringArray ResourceFormatLoaderZig::_get_recognized_extensions() const {
	PackedStringArray array;
	array.push_back("zig");
	return array;
}
bool ResourceFormatLoaderZig::_handles_type(const StringName &type) const {
	String type_str = type;
	return type_str == "ZigScript" || type_str == "Script";
}
String ResourceFormatLoaderZig::_get_resource_type(const String &p_path) const {
	if (p_path.get_extension() == "zig") {
		return "ZigScript";
	}
	return "";
}
