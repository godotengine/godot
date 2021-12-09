/*************************************************************************/
/*  register_types.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "register_types.h"

#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/os/os.h"
#include "scene/main/scene_tree.h"

#include "pluginscript_language.h"
#include "pluginscript_script.h"
#include <pluginscript/godot_pluginscript.h>

static List<PluginScriptLanguage *> pluginscript_languages;

static Error _check_language_desc(const godot_pluginscript_language_desc *desc) {
	ERR_FAIL_COND_V(!desc->name, ERR_BUG);
	ERR_FAIL_COND_V(!desc->type, ERR_BUG);
	ERR_FAIL_COND_V(!desc->extension, ERR_BUG);
	ERR_FAIL_COND_V(!desc->recognized_extensions || !desc->recognized_extensions[0], ERR_BUG);
	ERR_FAIL_COND_V(!desc->init, ERR_BUG);
	ERR_FAIL_COND_V(!desc->finish, ERR_BUG);

	// desc->reserved_words is not mandatory
	// desc->comment_delimiters is not mandatory
	// desc->string_delimiters is not mandatory

	// desc->get_template_source_code is not mandatory
	// desc->validate is not mandatory

	// desc->get_template_source_code is not mandatory
	// desc->validate is not mandatory
	// desc->find_function is not mandatory
	// desc->make_function is not mandatory
	// desc->complete_code is not mandatory
	// desc->auto_indent_code is not mandatory
	ERR_FAIL_COND_V(!desc->add_global_constant, ERR_BUG);
	// desc->debug_get_error is not mandatory
	// desc->debug_get_stack_level_count is not mandatory
	// desc->debug_get_stack_level_line is not mandatory
	// desc->debug_get_stack_level_function is not mandatory
	// desc->debug_get_stack_level_source is not mandatory
	// desc->debug_get_stack_level_locals is not mandatory
	// desc->debug_get_stack_level_members is not mandatory
	// desc->debug_get_globals is not mandatory
	// desc->debug_parse_stack_level_expression is not mandatory
	// desc->profiling_start is not mandatory
	// desc->profiling_stop is not mandatory
	// desc->profiling_get_accumulated_data is not mandatory
	// desc->profiling_get_frame_data is not mandatory
	// desc->profiling_frame is not mandatory

	ERR_FAIL_COND_V(!desc->script_desc.init, ERR_BUG);
	ERR_FAIL_COND_V(!desc->script_desc.finish, ERR_BUG);

	ERR_FAIL_COND_V(!desc->script_desc.instance_desc.init, ERR_BUG);
	ERR_FAIL_COND_V(!desc->script_desc.instance_desc.finish, ERR_BUG);
	ERR_FAIL_COND_V(!desc->script_desc.instance_desc.set_prop, ERR_BUG);
	ERR_FAIL_COND_V(!desc->script_desc.instance_desc.get_prop, ERR_BUG);
	ERR_FAIL_COND_V(!desc->script_desc.instance_desc.call_method, ERR_BUG);
	ERR_FAIL_COND_V(!desc->script_desc.instance_desc.notification, ERR_BUG);
	// desc->script_desc.instance_desc.refcount_incremented is not mandatory
	// desc->script_desc.instance_desc.refcount_decremented is not mandatory
	return OK;
}

void GDAPI godot_pluginscript_register_language(const godot_pluginscript_language_desc *language_desc) {
	Error ret = _check_language_desc(language_desc);
	if (ret) {
		ERR_FAIL();
	}
	PluginScriptLanguage *language = memnew(PluginScriptLanguage(language_desc));
	ScriptServer::register_language(language);
	ResourceLoader::add_resource_format_loader(language->get_resource_loader());
	ResourceSaver::add_resource_format_saver(language->get_resource_saver());
	pluginscript_languages.push_back(language);
}

void register_pluginscript_types() {
	GDREGISTER_CLASS(PluginScript);
}

void unregister_pluginscript_types() {
	for (List<PluginScriptLanguage *>::Element *e = pluginscript_languages.front(); e; e = e->next()) {
		PluginScriptLanguage *language = e->get();
		ScriptServer::unregister_language(language);
		ResourceLoader::remove_resource_format_loader(language->get_resource_loader());
		ResourceSaver::remove_resource_format_saver(language->get_resource_saver());
		memdelete(language);
	}
}
