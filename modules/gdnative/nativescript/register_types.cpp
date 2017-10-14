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

#include "io/resource_loader.h"
#include "io/resource_saver.h"

#include "nativescript.h"

#include "core/os/os.h"

NativeScriptLanguage *native_script_language;

ResourceFormatLoaderNativeScript *resource_loader_gdns = NULL;
ResourceFormatSaverNativeScript *resource_saver_gdns = NULL;

void register_nativescript_types() {
	native_script_language = memnew(NativeScriptLanguage);

	ClassDB::register_class<NativeScript>();

	ScriptServer::register_language(native_script_language);

	resource_saver_gdns = memnew(ResourceFormatSaverNativeScript);
	ResourceSaver::add_resource_format_saver(resource_saver_gdns);

	resource_loader_gdns = memnew(ResourceFormatLoaderNativeScript);
	ResourceLoader::add_resource_format_loader(resource_loader_gdns);
}

void unregister_nativescript_types() {

	memdelete(resource_loader_gdns);

	memdelete(resource_saver_gdns);

	if (native_script_language) {
		ScriptServer::unregister_language(native_script_language);
		memdelete(native_script_language);
	}
}
