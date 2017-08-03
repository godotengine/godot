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

#include "io/resource_loader.h"
#include "io/resource_saver.h"

#include "nativescript.h"

#include "core/os/os.h"

NativeScriptLanguage *native_script_language;

typedef void (*native_script_init_fn)(void *);

void init_call_cb(void *p_handle, godot_string *p_proc_name, void *p_data, int p_num_args, void **args, void *r_ret) {
	if (p_handle == NULL) {
		ERR_PRINT("No valid library handle, can't call nativescript init procedure");
		return;
	}

	void *library_proc;
	Error err = OS::get_singleton()->get_dynamic_library_symbol_handle(
			p_handle,
			*(String *)p_proc_name,
			library_proc,
			true); // we print our own message
	if (err != OK) {
		ERR_PRINT((String("GDNative procedure \"" + *(String *)p_proc_name) + "\" does not exists and can't be called").utf8().get_data());
		return;
	}

	native_script_init_fn fn = (native_script_init_fn)library_proc;

	fn(args[0]);
}

typedef void (*native_script_empty_callback)();

void noarg_call_cb(void *p_handle, godot_string *p_proc_name, void *p_data, int p_num_args, void **args, void *r_ret) {
	if (p_handle == NULL) {
		ERR_PRINT("No valid library handle, can't call nativescript callback");
		return;
	}

	void *library_proc;
	Error err = OS::get_singleton()->get_dynamic_library_symbol_handle(
			p_handle,
			*(String *)p_proc_name,
			library_proc,
			true);
	if (err != OK) {
		// it's fine if thread callbacks are not present in the library.
		return;
	}

	native_script_empty_callback fn = (native_script_empty_callback)library_proc;
	fn();
}

ResourceFormatLoaderNativeScript *resource_loader_gdns = NULL;
ResourceFormatSaverNativeScript *resource_saver_gdns = NULL;

void register_nativescript_types() {
	native_script_language = memnew(NativeScriptLanguage);

	ClassDB::register_class<NativeScript>();

	ScriptServer::register_language(native_script_language);

	GDNativeCallRegistry::singleton->register_native_raw_call_type(native_script_language->_init_call_type, init_call_cb);
	GDNativeCallRegistry::singleton->register_native_raw_call_type(native_script_language->_noarg_call_type, noarg_call_cb);

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
