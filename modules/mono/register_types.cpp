/*************************************************************************/
/*  register_types.cpp                                                   */
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
#include "register_types.h"

#include "project_settings.h"

#include "csharp_script.h"

CSharpLanguage *script_language_cs = NULL;
ResourceFormatLoaderCSharpScript *resource_loader_cs = NULL;
ResourceFormatSaverCSharpScript *resource_saver_cs = NULL;

_GodotSharp *_godotsharp = NULL;

void register_mono_types() {
	ClassDB::register_class<CSharpScript>();

	_godotsharp = memnew(_GodotSharp);

	ClassDB::register_class<_GodotSharp>();
	ProjectSettings::get_singleton()->add_singleton(ProjectSettings::Singleton("GodotSharp", _GodotSharp::get_singleton()));

	script_language_cs = memnew(CSharpLanguage);
	script_language_cs->set_language_index(ScriptServer::get_language_count());
	ScriptServer::register_language(script_language_cs);

	resource_loader_cs = memnew(ResourceFormatLoaderCSharpScript);
	ResourceLoader::add_resource_format_loader(resource_loader_cs);
	resource_saver_cs = memnew(ResourceFormatSaverCSharpScript);
	ResourceSaver::add_resource_format_saver(resource_saver_cs);
}

void unregister_mono_types() {
	ScriptServer::unregister_language(script_language_cs);

	if (script_language_cs)
		memdelete(script_language_cs);
	if (resource_loader_cs)
		memdelete(resource_loader_cs);
	if (resource_saver_cs)
		memdelete(resource_saver_cs);

	if (_godotsharp)
		memdelete(_godotsharp);
}
