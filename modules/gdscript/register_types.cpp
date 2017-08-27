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

#include "gd_script.h"
#include "io/file_access_encrypted.h"
#include "io/resource_loader.h"
#include "os/file_access.h"

GDScriptLanguage *script_language_gd = NULL;
ResourceFormatLoaderGDScript *resource_loader_gd = NULL;
ResourceFormatSaverGDScript *resource_saver_gd = NULL;

void register_gdscript_types() {

	ClassDB::register_class<GDScript>();
	ClassDB::register_virtual_class<GDFunctionState>();

	script_language_gd = memnew(GDScriptLanguage);
	//script_language_gd->init();
	ScriptServer::register_language(script_language_gd);
	resource_loader_gd = memnew(ResourceFormatLoaderGDScript);
	ResourceLoader::add_resource_format_loader(resource_loader_gd);
	resource_saver_gd = memnew(ResourceFormatSaverGDScript);
	ResourceSaver::add_resource_format_saver(resource_saver_gd);
}

void unregister_gdscript_types() {

	ScriptServer::unregister_language(script_language_gd);

	if (script_language_gd)
		memdelete(script_language_gd);
	if (resource_loader_gd)
		memdelete(resource_loader_gd);
	if (resource_saver_gd)
		memdelete(resource_saver_gd);
}
