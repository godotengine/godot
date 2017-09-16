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
#include "gdnative/gdnative.h"

#include "gdnative.h"

#include "io/resource_loader.h"
#include "io/resource_saver.h"

#include "nativescript/register_types.h"

#include "core/engine.h"
#include "core/os/os.h"
#include "core/project_settings.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_node.h"
#include "gd_native_library_editor.h"
// Class used to discover singleton gdnative files

void actual_discoverer_handler();

class GDNativeSingletonDiscover : public Object {
	// GDCLASS(GDNativeSingletonDiscover, Object)

	virtual String get_class() const {
		// okay, this is a really dirty hack.
		// We're overriding get_class so we can connect it to a signal
		// This works because get_class is a virtual method, so we don't
		// need to register a new class to ClassDB just for this one
		// little signal.

		actual_discoverer_handler();

		return "Object";
	}
};

Set<String> get_gdnative_singletons(EditorFileSystemDirectory *p_dir) {

	Set<String> file_paths;

	// check children

	for (int i = 0; i < p_dir->get_file_count(); i++) {
		String file_name = p_dir->get_file(i);
		String file_type = p_dir->get_file_type(i);

		if (file_type != "GDNativeLibrary") {
			continue;
		}

		Ref<GDNativeLibrary> lib = ResourceLoader::load(p_dir->get_file_path(i));
		if (lib.is_valid() && lib->is_singleton_gdnative()) {
			file_paths.insert(p_dir->get_file_path(i));
		}
	}

	// check subdirectories
	for (int i = 0; i < p_dir->get_subdir_count(); i++) {
		Set<String> paths = get_gdnative_singletons(p_dir->get_subdir(i));

		for (Set<String>::Element *E = paths.front(); E; E = E->next()) {
			file_paths.insert(E->get());
		}
	}

	return file_paths;
}

void actual_discoverer_handler() {
	EditorFileSystemDirectory *dir = EditorFileSystem::get_singleton()->get_filesystem();

	Set<String> file_paths = get_gdnative_singletons(dir);

	Array files;
	files.resize(file_paths.size());
	int i = 0;
	for (Set<String>::Element *E = file_paths.front(); E; i++, E = E->next()) {
		files.set(i, E->get());
	}

	ProjectSettings::get_singleton()->set("gdnative/singletons", files);

	ProjectSettings::get_singleton()->save();
}

GDNativeSingletonDiscover *discoverer = NULL;

static void editor_init_callback() {

	GDNativeLibraryEditor *library_editor = memnew(GDNativeLibraryEditor);
	library_editor->set_name(TTR("GDNative"));
	ProjectSettingsEditor::get_singleton()->get_tabs()->add_child(library_editor);

	discoverer = memnew(GDNativeSingletonDiscover);
	EditorFileSystem::get_singleton()->connect("filesystem_changed", discoverer, "get_class");
}

#endif

godot_variant cb_standard_varcall(void *handle, godot_string *p_procedure, godot_array *p_args) {
	if (handle == NULL) {
		ERR_PRINT("No valid library handle, can't call standard varcall procedure");
		godot_variant ret;
		godot_variant_new_nil(&ret);
		return ret;
	}

	void *library_proc;
	Error err = OS::get_singleton()->get_dynamic_library_symbol_handle(
			handle,
			*(String *)p_procedure,
			library_proc,
			true); // we roll our own message
	if (err != OK) {
		ERR_PRINT((String("GDNative procedure \"" + *(String *)p_procedure) + "\" does not exists and can't be called").utf8().get_data());
		godot_variant ret;
		godot_variant_new_nil(&ret);
		return ret;
	}

	godot_gdnative_procedure_fn proc;
	proc = (godot_gdnative_procedure_fn)library_proc;

	return proc(NULL, p_args);
}

void cb_singleton_call(
		void *p_handle,
		godot_string *p_proc_name,
		void *p_data,
		int p_num_args,
		void **p_args,
		void *r_return) {
	if (p_handle == NULL) {
		ERR_PRINT("No valid library handle, can't call singleton procedure");
		return;
	}

	void *singleton_proc;
	Error err = OS::get_singleton()->get_dynamic_library_symbol_handle(
			p_handle,
			*(String *)p_proc_name,
			singleton_proc);

	if (err != OK) {
		return;
	}

	void (*singleton_procedure_ptr)() = (void (*)())singleton_proc;
	singleton_procedure_ptr();
}

GDNativeCallRegistry *GDNativeCallRegistry::singleton;

Vector<Ref<GDNative> > singleton_gdnatives;

void register_gdnative_types() {

#ifdef TOOLS_ENABLED

	if (Engine::get_singleton()->is_editor_hint()) {
		EditorNode::add_init_callback(editor_init_callback);
	}
#endif

	ClassDB::register_class<GDNativeLibrary>();
	ClassDB::register_class<GDNative>();

	GDNativeCallRegistry::singleton = memnew(GDNativeCallRegistry);

	GDNativeCallRegistry::singleton->register_native_call_type("standard_varcall", cb_standard_varcall);

	GDNativeCallRegistry::singleton->register_native_raw_call_type("gdnative_singleton_call", cb_singleton_call);

	register_nativescript_types();

	// run singletons

	Array singletons = ProjectSettings::get_singleton()->get("gdnative/singletons");

	singleton_gdnatives.resize(singletons.size());

	for (int i = 0; i < singletons.size(); i++) {
		String path = singletons[i];

		Ref<GDNativeLibrary> lib = ResourceLoader::load(path);

		singleton_gdnatives[i].instance();
		singleton_gdnatives[i]->set_library(lib);

		if (!singleton_gdnatives[i]->initialize()) {
			// Can't initialize. Don't make a native_call then
			continue;
		}

		singleton_gdnatives[i]->call_native_raw(
				"gdnative_singleton_call",
				"godot_gdnative_singleton",
				NULL,
				0,
				NULL,
				NULL);
	}
}

void unregister_gdnative_types() {

	for (int i = 0; i < singleton_gdnatives.size(); i++) {

		if (singleton_gdnatives[i].is_null()) {
			continue;
		}

		if (!singleton_gdnatives[i]->is_initialized()) {
			continue;
		}

		singleton_gdnatives[i]->terminate();
	}

	unregister_nativescript_types();

	memdelete(GDNativeCallRegistry::singleton);

#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint() && discoverer != NULL) {
		memdelete(discoverer);
	}
#endif

	// This is for printing out the sizes of the core types

	/*
	print_line(String("array:\t")     + itos(sizeof(Array)));
	print_line(String("basis:\t")     + itos(sizeof(Basis)));
	print_line(String("color:\t")     + itos(sizeof(Color)));
	print_line(String("dict:\t" )     + itos(sizeof(Dictionary)));
	print_line(String("node_path:\t") + itos(sizeof(NodePath)));
	print_line(String("plane:\t")     + itos(sizeof(Plane)));
	print_line(String("poolarray:\t") + itos(sizeof(PoolByteArray)));
	print_line(String("quat:\t")      + itos(sizeof(Quat)));
	print_line(String("rect2:\t")     + itos(sizeof(Rect2)));
	print_line(String("rect3:\t")     + itos(sizeof(Rect3)));
	print_line(String("rid:\t")       + itos(sizeof(RID)));
	print_line(String("string:\t")    + itos(sizeof(String)));
	print_line(String("transform:\t") + itos(sizeof(Transform)));
	print_line(String("transfo2D:\t") + itos(sizeof(Transform2D)));
	print_line(String("variant:\t")   + itos(sizeof(Variant)));
	print_line(String("vector2:\t")   + itos(sizeof(Vector2)));
	print_line(String("vector3:\t")   + itos(sizeof(Vector3)));
	*/
}
