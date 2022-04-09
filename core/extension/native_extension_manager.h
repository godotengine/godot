/*************************************************************************/
/*  native_extension_manager.h                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef NATIVE_EXTENSION_MANAGER_H
#define NATIVE_EXTENSION_MANAGER_H

#include "core/extension/native_extension.h"

class NativeExtensionManager : public Object {
	GDCLASS(NativeExtensionManager, Object);

	int32_t level = -1;
	Map<String, Ref<NativeExtension>> native_extension_map;

	static void _bind_methods();

	static NativeExtensionManager *singleton;

public:
	enum LoadStatus {
		LOAD_STATUS_OK,
		LOAD_STATUS_FAILED,
		LOAD_STATUS_ALREADY_LOADED,
		LOAD_STATUS_NOT_LOADED,
		LOAD_STATUS_NEEDS_RESTART,
	};

	LoadStatus load_extension(const String &p_path);
	LoadStatus reload_extension(const String &p_path);
	LoadStatus unload_extension(const String &p_path);
	bool is_extension_loaded(const String &p_path) const;
	Vector<String> get_loaded_extensions() const;
	Ref<NativeExtension> get_extension(const String &p_path);

	void initialize_extensions(NativeExtension::InitializationLevel p_level);
	void deinitialize_extensions(NativeExtension::InitializationLevel p_level);

	static NativeExtensionManager *get_singleton();

	void load_extensions();

	NativeExtensionManager();
};

VARIANT_ENUM_CAST(NativeExtensionManager::LoadStatus)

#endif // NATIVEEXTENSIONMANAGER_H
