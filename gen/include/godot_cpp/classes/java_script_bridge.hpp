/**************************************************************************/
/*  java_script_bridge.hpp                                                */
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

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Callable;
class JavaScriptObject;

class JavaScriptBridge : public Object {
	GDEXTENSION_CLASS(JavaScriptBridge, Object)

	static JavaScriptBridge *singleton;

public:
	static JavaScriptBridge *get_singleton();

	Variant eval(const String &p_code, bool p_use_global_execution_context = false);
	Ref<JavaScriptObject> get_interface(const String &p_interface);
	Ref<JavaScriptObject> create_callback(const Callable &p_callable);
	bool is_js_buffer(const Ref<JavaScriptObject> &p_javascript_object);
	PackedByteArray js_buffer_to_packed_byte_array(const Ref<JavaScriptObject> &p_javascript_buffer);

private:
	Variant create_object_internal(const Variant **p_args, GDExtensionInt p_arg_count);

public:
	template <typename... Args>
	Variant create_object(const String &p_object, const Args &...p_args) {
		std::array<Variant, 1 + sizeof...(Args)> variant_args{{ Variant(p_object), Variant(p_args)... }};
		std::array<const Variant *, 1 + sizeof...(Args)> call_args;
		for (size_t i = 0; i < variant_args.size(); i++) {
			call_args[i] = &variant_args[i];
		}
		return create_object_internal(call_args.data(), variant_args.size());
	}
	void download_buffer(const PackedByteArray &p_buffer, const String &p_name, const String &p_mime = "application/octet-stream");
	bool pwa_needs_update() const;
	Error pwa_update();
	void force_fs_sync();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
	}

	~JavaScriptBridge();

public:
};

} // namespace godot

