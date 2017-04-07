/*************************************************************************/
/*  register_types.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "dl_script.h"

#include "io/resource_loader.h"
#include "io/resource_saver.h"

DLScriptLanguage *script_language_dl = NULL;
ResourceFormatLoaderDLScript *resource_loader_dl = NULL;
ResourceFormatSaverDLScript *resource_saver_dl = NULL;
//ResourceFormatLoaderDLLibrary *resource_loader_dllib=NULL;

void register_dlscript_types() {

	ClassDB::register_class<DLLibrary>();
	ClassDB::register_class<DLScript>();

	script_language_dl = memnew(DLScriptLanguage);
	//script_language_gd->init();
	ScriptServer::register_language(script_language_dl);
	resource_loader_dl = memnew(ResourceFormatLoaderDLScript);
	ResourceLoader::add_resource_format_loader(resource_loader_dl);
	resource_saver_dl = memnew(ResourceFormatSaverDLScript);
	ResourceSaver::add_resource_format_saver(resource_saver_dl);

	// resource_loader_dllib=memnew( ResourceFormatLoaderDLLibrary );
	// ResourceLoader::add_resource_format_loader(resource_loader_gd);
}

void unregister_dlscript_types() {

	ScriptServer::unregister_language(script_language_dl);

	if (script_language_dl)
		memdelete(script_language_dl);

	if (resource_loader_dl)
		memdelete(resource_loader_dl);

	if (resource_saver_dl)
		memdelete(resource_saver_dl);
}
