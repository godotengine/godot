/*************************************************************************/
/*  gdnative.h                                                           */
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
#ifndef GDNATIVE_H
#define GDNATIVE_H

#include "io/resource_loader.h"
#include "io/resource_saver.h"
#include "os/thread_safe.h"
#include "resource.h"

#include "gdnative/gdnative.h"
#include "gdnative_api_struct.h"

class GDNativeLibrary : public Resource {
	GDCLASS(GDNativeLibrary, Resource)

	enum Platform {
		X11_32BIT,
		X11_64BIT,
		WINDOWS_32BIT,
		WINDOWS_64BIT,
		// NOTE(karroffel): I heard OSX 32 bit is dead, so 64 only
		OSX,

		// Android .so files must be located in directories corresponding to Android ABI names:
		// https://developer.android.com/ndk/guides/abis.html
		// Android runtime will select the matching library depending on the device.
		// The value here must simply point to the .so name, for example:
		// "res://libmy_gdnative.so" or "libmy_gdnative.so",
		// while in the project the actual paths can be "lib/android/armeabi-v7a/libmy_gdnative.so",
		// "lib/android/arm64-v8a/libmy_gdnative.so".
		ANDROID,

		IOS_32BIT,
		IOS_64BIT,

		// TODO(karroffel): figure out how to deal with web stuff at all...
		WASM,

		// TODO(karroffel): does UWP have different libs??
		// UWP,

		NUM_PLATFORMS

	};

	static String platform_names[NUM_PLATFORMS + 1];
	static String platform_lib_ext[NUM_PLATFORMS + 1];

	static Platform current_platform;

	String library_paths[NUM_PLATFORMS];

	bool singleton_gdnative;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	GDNativeLibrary();
	~GDNativeLibrary();

	static void _bind_methods();

	void set_library_path(StringName p_platform, String p_path);
	String get_library_path(StringName p_platform) const;

	String get_active_library_path() const;

	_FORCE_INLINE_ bool is_singleton_gdnative() const { return singleton_gdnative; }
	_FORCE_INLINE_ void set_singleton_gdnative(bool p_singleton) { singleton_gdnative = p_singleton; }
};

typedef godot_variant (*native_call_cb)(void *, godot_string *, godot_array *);
typedef void (*native_raw_call_cb)(void *, godot_string *, void *, int, void **, void *);

struct GDNativeCallRegistry {
	static GDNativeCallRegistry *singleton;

	inline GDNativeCallRegistry *get_singleton() {
		return singleton;
	}

	inline GDNativeCallRegistry()
		: native_calls(),
		  native_raw_calls() {}

	Map<StringName, native_call_cb> native_calls;
	Map<StringName, native_raw_call_cb> native_raw_calls;

	void register_native_call_type(StringName p_call_type, native_call_cb p_callback);
	void register_native_raw_call_type(StringName p_raw_call_type, native_raw_call_cb p_callback);

	Vector<StringName> get_native_call_types();
	Vector<StringName> get_native_raw_call_types();
};

class GDNative : public Reference {
	GDCLASS(GDNative, Reference)

	Ref<GDNativeLibrary> library;

	// TODO(karroffel): different platforms? WASM????
	void *native_handle;

	void _compile_dummy_for_api();

public:
	GDNative();
	~GDNative();

	static void _bind_methods();

	void set_library(Ref<GDNativeLibrary> p_library);
	Ref<GDNativeLibrary> get_library();

	bool is_initialized();

	bool initialize();
	bool terminate();

	Variant call_native(StringName p_native_call_type, StringName p_procedure_name, Array p_arguments = Array());
	void call_native_raw(StringName p_raw_call_type, StringName p_procedure_name, void *data, int num_args, void **args, void *r_return);
};

#endif // GDNATIVE_H
