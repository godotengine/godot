/**************************************************************************/
/*  pluginscript_loader.cpp                                               */
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

// Godot imports
#include "core/os/file_access.h"
// Pythonscript imports
#include "pluginscript_language.h"
#include "pluginscript_loader.h"
#include "pluginscript_script.h"

ResourceFormatLoaderPluginScript::ResourceFormatLoaderPluginScript(PluginScriptLanguage *language) {
	_language = language;
}

RES ResourceFormatLoaderPluginScript::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_no_subresource_cache) {
	if (r_error) {
		*r_error = ERR_FILE_CANT_OPEN;
	}

	PluginScript *script = memnew(PluginScript);
	script->init(_language);

	Ref<PluginScript> scriptres(script);

	Error err = script->load_source_code(p_path);
	ERR_FAIL_COND_V(err != OK, RES());

	script->set_path(p_original_path);

	script->reload();

	if (r_error) {
		*r_error = OK;
	}

	return scriptres;
}

void ResourceFormatLoaderPluginScript::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back(_language->get_extension());
}

bool ResourceFormatLoaderPluginScript::handles_type(const String &p_type) const {
	return p_type == "Script" || p_type == _language->get_type();
}

String ResourceFormatLoaderPluginScript::get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	if (el == _language->get_extension()) {
		return _language->get_type();
	}
	return "";
}

ResourceFormatSaverPluginScript::ResourceFormatSaverPluginScript(PluginScriptLanguage *language) {
	_language = language;
}

Error ResourceFormatSaverPluginScript::save(const String &p_path, const RES &p_resource, uint32_t p_flags) {
	Ref<PluginScript> sqscr = p_resource;
	ERR_FAIL_COND_V(sqscr.is_null(), ERR_INVALID_PARAMETER);

	String source = sqscr->get_source_code();

	Error err;
	FileAccess *file = FileAccess::open(p_path, FileAccess::WRITE, &err);
	ERR_FAIL_COND_V(err, err);

	file->store_string(source);
	if (file->get_error() != OK && file->get_error() != ERR_FILE_EOF) {
		memdelete(file);
		return ERR_CANT_CREATE;
	}
	file->close();
	memdelete(file);
	return OK;
}

void ResourceFormatSaverPluginScript::get_recognized_extensions(const RES &p_resource, List<String> *p_extensions) const {
	if (Object::cast_to<PluginScript>(*p_resource)) {
		p_extensions->push_back(_language->get_extension());
	}
}

bool ResourceFormatSaverPluginScript::recognize(const RES &p_resource) const {
	return Object::cast_to<PluginScript>(*p_resource) != nullptr;
}
