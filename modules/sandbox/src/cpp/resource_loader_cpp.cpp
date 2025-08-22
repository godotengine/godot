/**************************************************************************/
/*  resource_loader_cpp.cpp                                               */
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

#include "resource_loader_cpp.h"
#include "core/io/file_access.h"
#include "script_cpp.h"

static Ref<ResourceFormatLoaderCPP> cpp_loader;

void ResourceFormatLoaderCPP::init() {
	cpp_loader.instantiate();
	ResourceLoader::get_singleton()->add_resource_format_loader(cpp_loader);
}

void ResourceFormatLoaderCPP::deinit() {
	ResourceLoader::get_singleton()->remove_resource_format_loader(cpp_loader);
	cpp_loader.unref();
}

Variant ResourceFormatLoaderCPP::_load(const String &p_path, const String &original_path, bool use_sub_threads, int32_t cache_mode) const {
	Ref<CPPScript> cpp_model;
	cpp_model.instantiate();
	cpp_model->set_file(p_path);
	return cpp_model;
}
PackedStringArray ResourceFormatLoaderCPP::get_recognized_extensions() const {
	PackedStringArray array;
	array.push_back("cpp");
	array.push_back("cc");
	array.push_back("hh");
	array.push_back("h");
	array.push_back("hpp");
	return array;
}
bool ResourceFormatLoaderCPP::handles_type(const StringName &type) const {
	String type_str = type;
	return type_str == "CPPScript" || type_str == "Script";
}
String ResourceFormatLoaderCPP::get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	if (el == "hpp" || el == "cpp" || el == "h" || el == "cc" || el == "hh") {
		return "CPPScript";
	}
	return "";
}
