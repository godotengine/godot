/**************************************************************************/
/*  resource_bundle.cpp                                                   */
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

#include "resource_bundle.h"

#include "core/io/resource_loader.h"
#include "core/object/class_db.h"

void ResourceBundle::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_open", "value"), &ResourceBundle::set_open);
	ClassDB::bind_method(D_METHOD("is_open"), &ResourceBundle::is_open);
	ClassDB::bind_method(D_METHOD("set_owned_path", "path"), &ResourceBundle::set_owned_path);
	ClassDB::bind_method(D_METHOD("get_owned_path"), &ResourceBundle::get_owned_path);
	ClassDB::bind_method(D_METHOD("is_owned"), &ResourceBundle::is_owned);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "open"), "set_open", "is_open");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "owned_path"), "set_owned_path", "get_owned_path");
}

Ref<ResourceBundle> ResourceBundle::load(const String &p_path) {
	const String bundle_file = p_path.path_join(".bundle");
	if (ResourceLoader::exists(bundle_file, "ResourceBundle")) {
		return ResourceLoader::load(bundle_file, "ResourceBundle");
	}
	return Ref<Resource>();
}

void ResourceBundle::set_open(bool p_value) {
	open = p_value;
}

bool ResourceBundle::is_open() const {
	return open;
}

void ResourceBundle::set_owned_path(const String &p_path) {
	owned_path = p_path;
}

String ResourceBundle::get_owned_path() const {
	return owned_path;
}

bool ResourceBundle::is_owned(const String &p_path) const {
	return owned_path == p_path;
}

ResourceBundle::ResourceBundle() {
}
