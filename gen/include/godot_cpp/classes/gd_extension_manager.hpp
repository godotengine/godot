/**************************************************************************/
/*  gd_extension_manager.hpp                                              */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class GDExtension;
class String;

class GDExtensionManager : public Object {
	GDEXTENSION_CLASS(GDExtensionManager, Object)

	static GDExtensionManager *singleton;

public:
	enum LoadStatus {
		LOAD_STATUS_OK = 0,
		LOAD_STATUS_FAILED = 1,
		LOAD_STATUS_ALREADY_LOADED = 2,
		LOAD_STATUS_NOT_LOADED = 3,
		LOAD_STATUS_NEEDS_RESTART = 4,
	};

	static GDExtensionManager *get_singleton();

	GDExtensionManager::LoadStatus load_extension(const String &p_path);
	GDExtensionManager::LoadStatus load_extension_from_function(const String &p_path, const GDExtensionInitializationFunction *p_init_func);
	GDExtensionManager::LoadStatus reload_extension(const String &p_path);
	GDExtensionManager::LoadStatus unload_extension(const String &p_path);
	bool is_extension_loaded(const String &p_path) const;
	PackedStringArray get_loaded_extensions() const;
	Ref<GDExtension> get_extension(const String &p_path);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
	}

	~GDExtensionManager();

public:
};

} // namespace godot

VARIANT_ENUM_CAST(GDExtensionManager::LoadStatus);

