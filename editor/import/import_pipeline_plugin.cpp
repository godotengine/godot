/**************************************************************************/
/*  import_pipeline_plugin.cpp                                            */
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

#include "import_pipeline_plugin.h"

#include "core/error/error_macros.h"
#include "editor/editor_node.h"
#include "editor/import/import_pipeline_step.h"

void ImportPipelinePlugin::_bind_methods() {
	GDVIRTUAL_BIND(_get_category);
	GDVIRTUAL_BIND(_get_avaible_steps);
	GDVIRTUAL_BIND(_get_step, "step");
}

String ImportPipelinePlugin::get_category() {
	String ret;
	if (GDVIRTUAL_CALL(_get_category, ret)) {
		return ret;
	}
	return "Others";
}

PackedStringArray ImportPipelinePlugin::get_avaible_steps() {
	PackedStringArray ret;
	if (GDVIRTUAL_CALL(_get_avaible_steps, ret)) {
		return ret;
	}
	return PackedStringArray();
}

Ref<ImportPipelineStep> ImportPipelinePlugin::get_step(const String &p_name) {
	Ref<ImportPipelineStep> ret;
	if (GDVIRTUAL_CALL(_get_step, p_name, ret)) {
		return ret;
	}
	return Ref<ImportPipelineStep>();
}

///////////////////////////////////////

ImportPipelinePlugins *ImportPipelinePlugins::singleton = nullptr;

void ImportPipelinePlugins::_bind_methods() {
	ADD_SIGNAL(MethodInfo("plugins_changed"));
}

void ImportPipelinePlugins::add_plugin(Ref<ImportPipelinePlugin> p_plugin) {
	plugins.push_back(p_plugin);
	emit_signal("plugins_changed");
}

void ImportPipelinePlugins::remove_plugin(Ref<ImportPipelinePlugin> p_plugin) {
	plugins.erase(p_plugin);
	emit_signal("plugins_changed");
}

Ref<ImportPipelineStep> ImportPipelinePlugins::create_step(const String &p_category, const String &p_name) {
	if (p_category.is_empty() || p_name.is_empty()) {
		return memnew(ImportPipelineStep);
	}
	for (Ref<ImportPipelinePlugin> plugin : plugins) {
		if (plugin->get_category() != p_category) {
			continue;
		}
		Ref<ImportPipelineStep> step = plugin->get_step(p_name);
		if (!step.is_valid()) {
			continue;
		}
		step->set_category_name(p_category);
		step->set_step_name(p_name);
		return step;
	}

	ERR_FAIL_V_MSG(Ref<ImportPipelineStep>(), vformat("Invalid category '%s' or name '%s' for pipeline step", p_category, p_name));
}
