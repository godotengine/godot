/**************************************************************************/
/*  resource_bundle_format.cpp                                           */
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

#include "resource_bundle_format.h"

#include "scene/resources/resource_bundle.h"

void ResourceBundleFormatLoader::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("bundle");
}

bool ResourceBundleFormatLoader::handles_type(const String &p_type) const {
	return p_type == "ResourceBundle";
}

String ResourceBundleFormatLoader::get_resource_type(const String &p_path) const {
	return "ResourceBundle";
}
bool ResourceBundleFormatLoader::recognize_path(const String &p_path, const String &p_for_type) const {
	if (!p_for_type.is_empty() && p_for_type != "ResourceBundle") {
		return false;
	}
	return p_path.get_extension().to_lower() == "bundle";
}

Ref<Resource> ResourceBundleFormatLoader::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	return ResourceFormatLoaderBinary::load(p_path, p_original_path, r_error, p_use_sub_threads, r_progress, p_cache_mode);
}

Error ResourceBundleFormatSaver::save(const Ref<Resource> &p_resource, const String &p_path, uint32_t p_flags) {
	return ResourceFormatSaverBinary::save(p_resource, p_path, p_flags);
}

bool ResourceBundleFormatSaver::recognize(const Ref<Resource> &p_resource) const {
	return p_resource->is_class("ResourceBundle");
}

void ResourceBundleFormatSaver::get_recognized_extensions(const Ref<Resource> &p_resource, List<String> *p_extensions) const {
	p_extensions->push_back("bundle");
}
