/**************************************************************************/
/*  extension_source_code_manager.h                                       */
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

#include "core/variant/variant.h"
#include "editor/extension/editor_extension_source_code_plugin.h"

class ExtensionSourceCodeManager : public Object {
	GDCLASS(ExtensionSourceCodeManager, Object);

	static inline ExtensionSourceCodeManager *singleton = nullptr;

	LocalVector<Ref<EditorExtensionSourceCodePlugin>> plugins;

public:
	static ExtensionSourceCodeManager *get_singleton() { return singleton; }

	int get_plugin_count() const;
	const Ref<EditorExtensionSourceCodePlugin> &get_plugin_at_index(int p_index) const;

	void add_plugin(const Ref<EditorExtensionSourceCodePlugin> &p_plugin);
	void remove_plugin(const Ref<EditorExtensionSourceCodePlugin> &p_plugin);

	/*
	 * Attempt to get a registered plugin that can handle the given object.
	 * If the object is a source-available GDExtension class, and there is a registered plugin
	 * that can handle it, it will return that plugin; otherwise, it will return null.
	 */
	const Ref<EditorExtensionSourceCodePlugin> get_plugin_for_object(const Object *p_object) const;

	/*
	 * Attempt to get a registered plugin that can handle the extension class declared
	 * at the given file. If the file declares a GDExtension class, and there is a registered plugin
	 * that can handle it, it will return that plugin; otherwise, it will return null.
	 */
	const Ref<EditorExtensionSourceCodePlugin> get_plugin_for_file(const String &p_source_path) const;

	bool has_plugins_that_can_create_class_source();

	static void create();
	static void cleanup();
};
