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

#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"

#include "nativescript.h"

#include "core/os/os.h"

NativeScriptLanguage *native_script_language;

Ref<ResourceFormatLoaderNativeScript> resource_loader_gdns;
Ref<ResourceFormatSaverNativeScript> resource_saver_gdns;

void register_nativescript_types() {
	native_script_language = memnew(NativeScriptLanguage);

	ClassDB::register_class<NativeScript>();

	native_script_language->set_language_index(ScriptServer::get_language_count());
	ScriptServer::register_language(native_script_language);

	resource_saver_gdns.instance();
	ResourceSaver::add_resource_format_saver(resource_saver_gdns);

	resource_loader_gdns.instance();
	ResourceLoader::add_resource_format_loader(resource_loader_gdns);
}

void unregister_nativescript_types() {
	ResourceLoader::remove_resource_format_loader(resource_loader_gdns);
	resource_loader_gdns.unref();

	ResourceSaver::remove_resource_format_saver(resource_saver_gdns);
	resource_saver_gdns.unref();

	if (native_script_language) {
		ScriptServer::unregister_language(native_script_language);
		memdelete(native_script_language);
	}
}
