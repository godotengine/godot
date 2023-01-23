/**************************************************************************/
/*  import_pipeline_plugin.h                                              */
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

#ifndef IMPORT_PIPELINE_PLUGIN_H
#define IMPORT_PIPELINE_PLUGIN_H

#include "core/error/error_macros.h"
#include "core/io/resource_importer.h"
#include "core/variant/dictionary.h"
#include "editor/import/import_pipeline_step.h"
#include "scene/resources/packed_scene.h"

class ImportPipelinePlugin : public RefCounted {
	GDCLASS(ImportPipelinePlugin, RefCounted);

protected:
	GDVIRTUAL0R(String, _get_category);
	GDVIRTUAL0R(PackedStringArray, _get_avaible_steps);
	GDVIRTUAL1R(Ref<ImportPipelineStep>, _get_step, String);

	static void _bind_methods();

public:
	virtual String get_category();
	virtual PackedStringArray get_avaible_steps();
	virtual Ref<ImportPipelineStep> get_step(const String &p_name);
};

class ImportPipelinePlugins : public Node {
	GDCLASS(ImportPipelinePlugins, Node);

	static ImportPipelinePlugins *singleton;

	Vector<Ref<ImportPipelinePlugin>> plugins;

protected:
	static void _bind_methods();

public:
	static ImportPipelinePlugins *get_singleton() { return singleton; }

	void add_plugin(Ref<ImportPipelinePlugin> p_plugin);
	void remove_plugin(Ref<ImportPipelinePlugin> p_plugin);
	int get_plugin_count() { return plugins.size(); }
	Ref<ImportPipelinePlugin> get_plugin(int p_index) { return plugins[p_index]; }
	Ref<ImportPipelineStep> create_step(const String &p_category, const String &p_name);

	ImportPipelinePlugins() { singleton = this; };
	~ImportPipelinePlugins() { singleton = nullptr; };
};

#endif // IMPORT_PIPELINE_PLUGIN_H
