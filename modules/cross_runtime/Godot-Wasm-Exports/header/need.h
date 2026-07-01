/**************************************************************************/
/*  need.h                                                                */
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

#pragma once

// need.h
// Aggregated include header for all generated C++ exports.
// Pulls in the full set of Godot core headers required across
// variant_accessors, variant_constructors, and every generated class file.
// Also declares the resource_cache (HashMap<ObjectID, Ref<Resource>>) used
// by RefCounted return paths, and the cache_loaded_resource() helper that
// registers a Resource with the cache to prevent premature collection.
// The resource_cache definition lives in the ResourceLoader class file;
// a release method must be called from the JS side to evict entries.

#include "core/error/error_macros.h"
#include "core/io/resource.h"
#include "core/object/class_db.h"
#include "core/object/method_bind.h"
#include "core/object/object.h"
#include "core/object/object_id.h"
#include "core/object/ref_counted.h"
#include "core/string/node_path.h"
#include "core/string/string_name.h"
#include "core/string/ustring.h"
#include "core/templates/hash_map.h"
#include "core/templates/rid.h"
#include "core/variant/array.h"
#include "core/variant/callable.h"
#include "core/variant/dictionary.h"
#include "core/variant/variant.h"

#include <emscripten.h>

#include <cstdint>
#include <string>

// Caches Refcounted resources
extern HashMap<ObjectID, Ref<Resource>> resource_cache;

static inline void cache_loaded_resource(Object *obj) {
	if (!obj) {
		return;
	}

	Resource *res = Object::cast_to<Resource>(obj);
	if (!res) {
		return;
	}

	ObjectID id = res->get_instance_id();
	resource_cache[id] = Ref<Resource>(res);
}
