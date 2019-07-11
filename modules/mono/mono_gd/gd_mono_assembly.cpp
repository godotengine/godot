/*************************************************************************/
/*  gd_mono_assembly.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "gd_mono_assembly.h"

#include <mono/metadata/mono-debug.h>
#include <mono/metadata/tokentype.h>

#include "core/list.h"
#include "core/os/file_access.h"
#include "core/os/os.h"
#include "core/project_settings.h"

#include "../godotsharp_dirs.h"
#include "gd_mono_class.h"

bool GDMonoAssembly::no_search = false;
bool GDMonoAssembly::in_preload = false;

Vector<String> GDMonoAssembly::search_dirs;

void GDMonoAssembly::fill_search_dirs(Vector<String> &r_search_dirs, const String &p_custom_config, const String &p_custom_bcl_dir) {

	String framework_dir;

	if (!p_custom_bcl_dir.empty()) {
		framework_dir = p_custom_bcl_dir;
	} else if (mono_assembly_getrootdir()) {
		framework_dir = String::utf8(mono_assembly_getrootdir()).plus_file("mono").plus_file("4.5");
	}

	if (!framework_dir.empty()) {
		r_search_dirs.push_back(framework_dir);
		r_search_dirs.push_back(framework_dir.plus_file("Facades"));
	}

	if (p_custom_config.length()) {
		r_search_dirs.push_back(GodotSharpDirs::get_res_temp_assemblies_base_dir().plus_file(p_custom_config));
	} else {
		r_search_dirs.push_back(GodotSharpDirs::get_res_temp_assemblies_dir());
	}

	if (p_custom_config.empty()) {
		r_search_dirs.push_back(GodotSharpDirs::get_res_assemblies_dir());
	} else {
		String api_config = p_custom_config == "Release" ? "Release" : "Debug";
		r_search_dirs.push_back(GodotSharpDirs::get_res_assemblies_base_dir().plus_file(api_config));
	}

	r_search_dirs.push_back(GodotSharpDirs::get_res_assemblies_base_dir());
	r_search_dirs.push_back(OS::get_singleton()->get_resource_dir());
	r_search_dirs.push_back(OS::get_singleton()->get_executable_path().get_base_dir());

#ifdef TOOLS_ENABLED
	r_search_dirs.push_back(GodotSharpDirs::get_data_editor_tools_dir());

	// For GodotTools to find the api assemblies
	r_search_dirs.push_back(GodotSharpDirs::get_data_editor_prebuilt_api_dir().plus_file("Debug"));
#endif
}

void GDMonoAssembly::assembly_load_hook(MonoAssembly *assembly, void *user_data) {

	if (no_search)
		return;

	// If our search and preload hooks fail to load the assembly themselves, the mono runtime still might.
	// Just do Assembly.LoadFrom("/Full/Path/On/Disk.dll");
	// In this case, we wouldn't have the assembly known in GDMono, which causes crashes
	// if any class inside the assembly is looked up by Godot.
	// And causing a lookup like that is as easy as throwing an exception defined in it...
	// No, we can't make the assembly load hooks smart enough because they get passed a MonoAssemblyName* only,
	// not the disk path passed to say Assembly.LoadFrom().
	_wrap_mono_assembly(assembly);
}

MonoAssembly *GDMonoAssembly::assembly_search_hook(MonoAssemblyName *aname, void *user_data) {
	return GDMonoAssembly::_search_hook(aname, user_data, false);
}

MonoAssembly *GDMonoAssembly::assembly_refonly_search_hook(MonoAssemblyName *aname, void *user_data) {
	return GDMonoAssembly::_search_hook(aname, user_data, true);
}

MonoAssembly *GDMonoAssembly::assembly_preload_hook(MonoAssemblyName *aname, char **assemblies_path, void *user_data) {
	return GDMonoAssembly::_preload_hook(aname, assemblies_path, user_data, false);
}

MonoAssembly *GDMonoAssembly::assembly_refonly_preload_hook(MonoAssemblyName *aname, char **assemblies_path, void *user_data) {
	return GDMonoAssembly::_preload_hook(aname, assemblies_path, user_data, true);
}

MonoAssembly *GDMonoAssembly::_search_hook(MonoAssemblyName *aname, void *user_data, bool refonly) {

	(void)user_data; // UNUSED

	String name = mono_assembly_name_get_name(aname);
	bool has_extension = name.ends_with(".dll") || name.ends_with(".exe");

	if (no_search)
		return NULL;

	GDMonoAssembly **loaded_asm = GDMono::get_singleton()->get_loaded_assembly(has_extension ? name.get_basename() : name);
	if (loaded_asm)
		return (*loaded_asm)->get_assembly();

	no_search = true; // Avoid the recursion madness

	GDMonoAssembly *res = _load_assembly_search(name, search_dirs, refonly);

	no_search = false;

	return res ? res->get_assembly() : NULL;
}

static _THREAD_LOCAL_(MonoImage *) image_corlib_loading = NULL;

MonoAssembly *GDMonoAssembly::_preload_hook(MonoAssemblyName *aname, char **, void *user_data, bool refonly) {

	(void)user_data; // UNUSED

	if (search_dirs.empty()) {
		fill_search_dirs(search_dirs);
	}

	{
		// If we find the assembly here, we load it with `mono_assembly_load_from_full`,
		// which in turn invokes load hooks before returning the MonoAssembly to us.
		// One of the load hooks is `load_aot_module`. This hook can end up calling preload hooks
		// again for the same assembly in certain in certain circumstances (the `do_load_image` part).
		// If this is the case and we return NULL due to the no_search condition below,
		// it will result in an internal crash later on. Therefore we need to return the assembly we didn't
		// get yet from `mono_assembly_load_from_full`. Luckily we have the image, which already got it.
		// This must be done here. If done in search hooks, it would cause `mono_assembly_load_from_full`
		// to think another MonoAssembly for this assembly was already loaded, making it delete its own,
		// when in fact both pointers were the same... This hooks thing is confusing.
		if (image_corlib_loading) {
			return mono_image_get_assembly(image_corlib_loading);
		}
	}

	if (no_search)
		return NULL;

	no_search = true;
	in_preload = true;

	String name = mono_assembly_name_get_name(aname);
	bool has_extension = name.ends_with(".dll");

	GDMonoAssembly *res = NULL;
	if (has_extension ? name == "mscorlib.dll" : name == "mscorlib") {
		GDMonoAssembly **stored_assembly = GDMono::get_singleton()->get_loaded_assembly(has_extension ? name.get_basename() : name);
		if (stored_assembly)
			return (*stored_assembly)->get_assembly();

		res = _load_assembly_search("mscorlib.dll", search_dirs, refonly);
	}

	no_search = false;
	in_preload = false;

	return res ? res->get_assembly() : NULL;
}

GDMonoAssembly *GDMonoAssembly::_load_assembly_search(const String &p_name, const Vector<String> &p_search_dirs, bool p_refonly) {

	GDMonoAssembly *res = NULL;
	String path;

	bool has_extension = p_name.ends_with(".dll") || p_name.ends_with(".exe");

	for (int i = 0; i < p_search_dirs.size(); i++) {
		const String &search_dir = p_search_dirs[i];

		if (has_extension) {
			path = search_dir.plus_file(p_name);
			if (FileAccess::exists(path)) {
				res = _load_assembly_from(p_name.get_basename(), path, p_refonly);
				if (res != NULL)
					return res;
			}
		} else {
			path = search_dir.plus_file(p_name + ".dll");
			if (FileAccess::exists(path)) {
				res = _load_assembly_from(p_name, path, p_refonly);
				if (res != NULL)
					return res;
			}

			path = search_dir.plus_file(p_name + ".exe");
			if (FileAccess::exists(path)) {
				res = _load_assembly_from(p_name, path, p_refonly);
				if (res != NULL)
					return res;
			}
		}
	}

	return NULL;
}

GDMonoAssembly *GDMonoAssembly::_load_assembly_from(const String &p_name, const String &p_path, bool p_refonly) {

	GDMonoAssembly *assembly = memnew(GDMonoAssembly(p_name, p_path));

	Error err = assembly->load(p_refonly);

	if (err != OK) {
		memdelete(assembly);
		ERR_FAIL_V(NULL);
	}

	MonoDomain *domain = mono_domain_get();
	GDMono::get_singleton()->add_assembly(domain ? mono_domain_get_id(domain) : 0, assembly);

	return assembly;
}

void GDMonoAssembly::_wrap_mono_assembly(MonoAssembly *assembly) {
	String name = mono_assembly_name_get_name(mono_assembly_get_name(assembly));

	MonoImage *image = mono_assembly_get_image(assembly);

	GDMonoAssembly *gdassembly = memnew(GDMonoAssembly(name, mono_image_get_filename(image)));
	Error err = gdassembly->wrapper_for_image(image);

	if (err != OK) {
		memdelete(gdassembly);
		ERR_FAIL();
	}

	MonoDomain *domain = mono_domain_get();
	GDMono::get_singleton()->add_assembly(domain ? mono_domain_get_id(domain) : 0, gdassembly);
}

void GDMonoAssembly::initialize() {

	mono_install_assembly_search_hook(&assembly_search_hook, NULL);
	mono_install_assembly_refonly_search_hook(&assembly_refonly_search_hook, NULL);
	mono_install_assembly_preload_hook(&assembly_preload_hook, NULL);
	mono_install_assembly_refonly_preload_hook(&assembly_refonly_preload_hook, NULL);
	mono_install_assembly_load_hook(&assembly_load_hook, NULL);
}

Error GDMonoAssembly::load(bool p_refonly) {

	ERR_FAIL_COND_V(loaded, ERR_FILE_ALREADY_IN_USE);

	refonly = p_refonly;

	uint64_t last_modified_time = FileAccess::get_modified_time(path);

	Vector<uint8_t> data = FileAccess::get_file_as_array(path);
	ERR_FAIL_COND_V(data.empty(), ERR_FILE_CANT_READ);

	String image_filename;

#ifdef ANDROID_ENABLED
	if (path.begins_with("res://")) {
		image_filename = path.substr(6, path.length());
	} else {
		image_filename = ProjectSettings::get_singleton()->globalize_path(path);
	}
#else
	// FIXME: globalize_path does not work on exported games
	image_filename = ProjectSettings::get_singleton()->globalize_path(path);
#endif

	MonoImageOpenStatus status = MONO_IMAGE_OK;

	image = mono_image_open_from_data_with_name(
			(char *)&data[0], data.size(),
			true, &status, refonly,
			image_filename.utf8().get_data());

	ERR_FAIL_COND_V(status != MONO_IMAGE_OK, ERR_FILE_CANT_OPEN);
	ERR_FAIL_NULL_V(image, ERR_FILE_CANT_OPEN);

#ifdef DEBUG_ENABLED
	Vector<uint8_t> pdb_data;
	String pdb_path(path + ".pdb");

	if (!FileAccess::exists(pdb_path)) {
		pdb_path = path.get_basename() + ".pdb"; // without .dll

		if (!FileAccess::exists(pdb_path))
			goto no_pdb;
	}

	pdb_data = FileAccess::get_file_as_array(pdb_path);

	// mono_debug_close_image doesn't seem to be needed
	mono_debug_open_image_from_memory(image, &pdb_data[0], pdb_data.size());

no_pdb:

#endif

	bool is_corlib_preload = in_preload && name == "mscorlib";

	if (is_corlib_preload)
		image_corlib_loading = image;

	assembly = mono_assembly_load_from_full(image, image_filename.utf8().get_data(), &status, refonly);

	if (is_corlib_preload)
		image_corlib_loading = NULL;

	ERR_FAIL_COND_V(status != MONO_IMAGE_OK || assembly == NULL, ERR_FILE_CANT_OPEN);

	// Decrement refcount which was previously incremented by mono_image_open_from_data_with_name
	mono_image_close(image);

	loaded = true;
	modified_time = last_modified_time;

	return OK;
}

Error GDMonoAssembly::wrapper_for_image(MonoImage *p_image) {

	ERR_FAIL_COND_V(loaded, ERR_FILE_ALREADY_IN_USE);

	assembly = mono_image_get_assembly(p_image);
	ERR_FAIL_NULL_V(assembly, FAILED);

	image = p_image;

	loaded = true;

	return OK;
}

void GDMonoAssembly::unload() {

	ERR_FAIL_COND(!loaded);

	for (Map<MonoClass *, GDMonoClass *>::Element *E = cached_raw.front(); E; E = E->next()) {
		memdelete(E->value());
	}

	cached_classes.clear();
	cached_raw.clear();

	assembly = NULL;
	image = NULL;
	loaded = false;
}

GDMonoClass *GDMonoAssembly::get_class(const StringName &p_namespace, const StringName &p_name) {

	ERR_FAIL_COND_V(!loaded, NULL);

	ClassKey key(p_namespace, p_name);

	GDMonoClass **match = cached_classes.getptr(key);

	if (match)
		return *match;

	MonoClass *mono_class = mono_class_from_name(image, String(p_namespace).utf8(), String(p_name).utf8());

	if (!mono_class)
		return NULL;

	GDMonoClass *wrapped_class = memnew(GDMonoClass(p_namespace, p_name, mono_class, this));

	cached_classes[key] = wrapped_class;
	cached_raw[mono_class] = wrapped_class;

	return wrapped_class;
}

GDMonoClass *GDMonoAssembly::get_class(MonoClass *p_mono_class) {

	ERR_FAIL_COND_V(!loaded, NULL);

	Map<MonoClass *, GDMonoClass *>::Element *match = cached_raw.find(p_mono_class);

	if (match)
		return match->value();

	StringName namespace_name = mono_class_get_namespace(p_mono_class);
	StringName class_name = mono_class_get_name(p_mono_class);

	GDMonoClass *wrapped_class = memnew(GDMonoClass(namespace_name, class_name, p_mono_class, this));

	cached_classes[ClassKey(namespace_name, class_name)] = wrapped_class;
	cached_raw[p_mono_class] = wrapped_class;

	return wrapped_class;
}

GDMonoClass *GDMonoAssembly::get_object_derived_class(const StringName &p_class) {

	GDMonoClass *match = NULL;

	if (gdobject_class_cache_updated) {
		Map<StringName, GDMonoClass *>::Element *result = gdobject_class_cache.find(p_class);

		if (result)
			match = result->get();
	} else {
		List<GDMonoClass *> nested_classes;

		int rows = mono_image_get_table_rows(image, MONO_TABLE_TYPEDEF);

		for (int i = 1; i < rows; i++) {
			MonoClass *mono_class = mono_class_get(image, (i + 1) | MONO_TOKEN_TYPE_DEF);

			if (!mono_class_is_assignable_from(CACHED_CLASS_RAW(GodotObject), mono_class))
				continue;

			GDMonoClass *current = get_class(mono_class);

			if (!current)
				continue;

			nested_classes.push_back(current);

			if (!match && current->get_name() == p_class)
				match = current;

			while (!nested_classes.empty()) {
				GDMonoClass *current_nested = nested_classes.front()->get();
				nested_classes.pop_back();

				void *iter = NULL;

				while (true) {
					MonoClass *raw_nested = mono_class_get_nested_types(current_nested->get_mono_ptr(), &iter);

					if (!raw_nested)
						break;

					GDMonoClass *nested_class = get_class(raw_nested);

					if (nested_class) {
						gdobject_class_cache.insert(nested_class->get_name(), nested_class);
						nested_classes.push_back(nested_class);
					}
				}
			}

			gdobject_class_cache.insert(current->get_name(), current);
		}

		gdobject_class_cache_updated = true;
	}

	return match;
}

GDMonoAssembly *GDMonoAssembly::load_from(const String &p_name, const String &p_path, bool p_refonly) {

	GDMonoAssembly **loaded_asm = GDMono::get_singleton()->get_loaded_assembly(p_name);
	if (loaded_asm)
		return *loaded_asm;
#ifdef DEBUG_ENABLED
	CRASH_COND(!FileAccess::exists(p_path));
#endif
	no_search = true;
	GDMonoAssembly *res = _load_assembly_from(p_name, p_path, p_refonly);
	no_search = false;
	return res;
}

GDMonoAssembly::GDMonoAssembly(const String &p_name, const String &p_path) {

	loaded = false;
	gdobject_class_cache_updated = false;
	name = p_name;
	path = p_path;
	refonly = false;
	modified_time = 0;
	assembly = NULL;
	image = NULL;
}

GDMonoAssembly::~GDMonoAssembly() {

	if (loaded)
		unload();
}
