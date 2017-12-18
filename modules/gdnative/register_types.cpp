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

#include "arvr/register_types.h"
#include "nativescript/register_types.h"
#include "pluginscript/register_types.h"

#include "core/engine.h"
#include "core/os/os.h"
#include "core/project_settings.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_node.h"
#include "gdnative_library_editor_plugin.h"
#include "gdnative_library_singleton_editor.h"
// Class used to discover singleton gdnative files

static void actual_discoverer_handler();

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

static Set<String> get_gdnative_singletons(EditorFileSystemDirectory *p_dir) {

	Set<String> file_paths;

	// check children

	for (int i = 0; i < p_dir->get_file_count(); i++) {
		String file_name = p_dir->get_file(i);
		String file_type = p_dir->get_file_type(i);

		if (file_type != "GDNativeLibrary") {
			continue;
		}

		Ref<GDNativeLibrary> lib = ResourceLoader::load(p_dir->get_file_path(i));
		if (lib.is_valid() && lib->is_singleton()) {
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

static void actual_discoverer_handler() {

	EditorFileSystemDirectory *dir = EditorFileSystem::get_singleton()->get_filesystem();

	Set<String> file_paths = get_gdnative_singletons(dir);

	bool changed = false;
	Array current_files;
	if (ProjectSettings::get_singleton()->has_setting("gdnative/singletons")) {
		current_files = ProjectSettings::get_singleton()->get("gdnative/singletons");
	}
	Array files;
	files.resize(file_paths.size());
	int i = 0;
	for (Set<String>::Element *E = file_paths.front(); E; i++, E = E->next()) {
		if (!current_files.has(E->get())) {
			changed = true;
		}
		files.set(i, E->get());
	}

	// Check for removed files
	if (!changed) {
		for (int i = 0; i < current_files.size(); i++) {
			if (!file_paths.has(current_files[i])) {
				changed = true;
				break;
			}
		}
	}

	if (changed) {

		ProjectSettings::get_singleton()->set("gdnative/singletons", files);
		ProjectSettings::get_singleton()->save();
	}
}

static GDNativeSingletonDiscover *discoverer = NULL;

class GDNativeExportPlugin : public EditorExportPlugin {

protected:
	virtual void _export_file(const String &p_path, const String &p_type, const Set<String> &p_features);
};

struct LibrarySymbol {
	char *name;
	bool is_required;
};

void GDNativeExportPlugin::_export_file(const String &p_path, const String &p_type, const Set<String> &p_features) {
	if (p_type != "GDNativeLibrary") {
		return;
	}

	Ref<GDNativeLibrary> lib = ResourceLoader::load(p_path);

	if (lib.is_null()) {
		return;
	}

	Ref<ConfigFile> config = lib->get_config_file();

	{

		List<String> entry_keys;
		config->get_section_keys("entry", &entry_keys);

		for (List<String>::Element *E = entry_keys.front(); E; E = E->next()) {
			String key = E->get();

			Vector<String> tags = key.split(".");

			bool skip = false;
			for (int i = 0; i < tags.size(); i++) {
				bool has_feature = p_features.has(tags[i]);

				if (!has_feature) {
					skip = true;
					break;
				}
			}

			if (skip) {
				continue;
			}

			String entry_lib_path = config->get_value("entry", key);
			add_shared_object(entry_lib_path, tags);
		}
	}

	{
		List<String> dependency_keys;
		config->get_section_keys("dependencies", &dependency_keys);

		for (List<String>::Element *E = dependency_keys.front(); E; E = E->next()) {
			String key = E->get();

			Vector<String> tags = key.split(".");

			bool skip = false;
			for (int i = 0; i < tags.size(); i++) {
				bool has_feature = p_features.has(tags[i]);

				if (!has_feature) {
					skip = true;
					break;
				}
			}

			if (skip) {
				continue;
			}

			Vector<String> dependency_paths = config->get_value("dependencies", key);
			for (int i = 0; i < dependency_paths.size(); i++) {
				add_shared_object(dependency_paths[i], tags);
			}
		}
	}

	if (p_features.has("iOS")) {
		// Register symbols in the "fake" dynamic lookup table, because dlsym does not work well on iOS.
		LibrarySymbol expected_symbols[] = {
			{ "gdnative_init", true },
			{ "gdnative_terminate", false },
			{ "nativescript_init", false },
			{ "nativescript_frame", false },
			{ "nativescript_thread_enter", false },
			{ "nativescript_thread_exit", false },
			{ "gdnative_singleton", false }
		};
		String declare_pattern = "extern \"C\" void $name(void)$weak;\n";
		String additional_code = "extern void register_dynamic_symbol(char *name, void *address);\n"
								 "extern void add_ios_init_callback(void (*cb)());\n";
		String linker_flags = "";
		for (int i = 0; i < sizeof(expected_symbols) / sizeof(expected_symbols[0]); ++i) {
			String full_name = lib->get_symbol_prefix() + expected_symbols[i].name;
			String code = declare_pattern.replace("$name", full_name);
			code = code.replace("$weak", expected_symbols[i].is_required ? "" : " __attribute__((weak))");
			additional_code += code;

			if (!expected_symbols[i].is_required) {
				if (linker_flags.length() > 0) {
					linker_flags += " ";
				}
				linker_flags += "-Wl,-U,_" + full_name;
			}
		}

		additional_code += String("void $prefixinit() {\n").replace("$prefix", lib->get_symbol_prefix());
		String register_pattern = "  if (&$name) register_dynamic_symbol((char *)\"$name\", (void *)$name);\n";
		for (int i = 0; i < sizeof(expected_symbols) / sizeof(expected_symbols[0]); ++i) {
			String full_name = lib->get_symbol_prefix() + expected_symbols[i].name;
			additional_code += register_pattern.replace("$name", full_name);
		}
		additional_code += "}\n";
		additional_code += String("struct $prefixstruct {$prefixstruct() {add_ios_init_callback($prefixinit);}};\n").replace("$prefix", lib->get_symbol_prefix());
		additional_code += String("$prefixstruct $prefixstruct_instance;\n").replace("$prefix", lib->get_symbol_prefix());

		add_ios_cpp_code(additional_code);
		add_ios_linker_flags(linker_flags);
	}
}

static void editor_init_callback() {

	GDNativeLibrarySingletonEditor *library_editor = memnew(GDNativeLibrarySingletonEditor);
	library_editor->set_name(TTR("GDNative"));
	ProjectSettingsEditor::get_singleton()->get_tabs()->add_child(library_editor);

	discoverer = memnew(GDNativeSingletonDiscover);
	EditorFileSystem::get_singleton()->connect("filesystem_changed", discoverer, "get_class");

	Ref<GDNativeExportPlugin> export_plugin;
	export_plugin.instance();

	EditorExport::get_singleton()->add_export_plugin(export_plugin);

	EditorNode::get_singleton()->add_editor_plugin(memnew(GDNativeLibraryEditorPlugin(EditorNode::get_singleton())));
}

#endif

static godot_variant cb_standard_varcall(void *p_procedure_handle, godot_array *p_args) {

	godot_gdnative_procedure_fn proc;
	proc = (godot_gdnative_procedure_fn)p_procedure_handle;

	return proc(p_args);
}

GDNativeCallRegistry *GDNativeCallRegistry::singleton;

Vector<Ref<GDNative> > singleton_gdnatives;

GDNativeLibraryResourceLoader *resource_loader_gdnlib = NULL;
GDNativeLibraryResourceSaver *resource_saver_gdnlib = NULL;

void register_gdnative_types() {

#ifdef TOOLS_ENABLED

	EditorNode::add_init_callback(editor_init_callback);
#endif

	ClassDB::register_class<GDNativeLibrary>();
	ClassDB::register_class<GDNative>();

	resource_loader_gdnlib = memnew(GDNativeLibraryResourceLoader);
	resource_saver_gdnlib = memnew(GDNativeLibraryResourceSaver);

	ResourceLoader::add_resource_format_loader(resource_loader_gdnlib);
	ResourceSaver::add_resource_format_saver(resource_saver_gdnlib);

	GDNativeCallRegistry::singleton = memnew(GDNativeCallRegistry);

	GDNativeCallRegistry::singleton->register_native_call_type("standard_varcall", cb_standard_varcall);

	register_arvr_types();
	register_nativescript_types();
	register_pluginscript_types();

	// run singletons

	Array singletons = Array();
	if (ProjectSettings::get_singleton()->has_setting("gdnative/singletons")) {
		singletons = ProjectSettings::get_singleton()->get("gdnative/singletons");
	}

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

		void *proc_ptr;
		Error err = singleton_gdnatives[i]->get_symbol(
				lib->get_symbol_prefix() + "gdnative_singleton",
				proc_ptr);

		if (err != OK) {
			ERR_PRINT((String("No godot_gdnative_singleton in \"" + singleton_gdnatives[i]->get_library()->get_current_library_path()) + "\" found").utf8().get_data());
		} else {
			((void (*)())proc_ptr)();
		}
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
	singleton_gdnatives.clear();

	unregister_pluginscript_types();
	unregister_nativescript_types();
	unregister_arvr_types();

	memdelete(GDNativeCallRegistry::singleton);

#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint() && discoverer != NULL) {
		memdelete(discoverer);
	}
#endif

	memdelete(resource_loader_gdnlib);
	memdelete(resource_saver_gdnlib);

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
	print_line(String("aabb:\t")     + itos(sizeof(AABB)));
	print_line(String("rid:\t")       + itos(sizeof(RID)));
	print_line(String("string:\t")    + itos(sizeof(String)));
	print_line(String("transform:\t") + itos(sizeof(Transform)));
	print_line(String("transfo2D:\t") + itos(sizeof(Transform2D)));
	print_line(String("variant:\t")   + itos(sizeof(Variant)));
	print_line(String("vector2:\t")   + itos(sizeof(Vector2)));
	print_line(String("vector3:\t")   + itos(sizeof(Vector3)));
	*/
}
