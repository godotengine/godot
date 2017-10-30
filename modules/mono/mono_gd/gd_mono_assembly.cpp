/*************************************************************************/
/*  gd_mono_assembly.cpp                                                 */
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
#include "gd_mono_assembly.h"

#include <mono/metadata/mono-debug.h>
#include <mono/metadata/tokentype.h>

#include "list.h"
#include "os/file_access.h"
#include "os/os.h"

#include "../godotsharp_dirs.h"
#include "gd_mono_class.h"

bool GDMonoAssembly::no_search = false;
Vector<String> GDMonoAssembly::search_dirs;

MonoAssembly *GDMonoAssembly::_search_hook(MonoAssemblyName *aname, void *user_data) {

	(void)user_data; // UNUSED

	String name = mono_assembly_name_get_name(aname);
	bool has_extension = name.ends_with(".dll") || name.ends_with(".exe");

	if (no_search)
		return NULL;

	GDMonoAssembly **loaded_asm = GDMono::get_singleton()->get_loaded_assembly(has_extension ? name.get_basename() : name);
	if (loaded_asm)
		return (*loaded_asm)->get_assembly();

	no_search = true; // Avoid the recursion madness

	String path;
	MonoAssembly *res = NULL;

	for (int i = 0; i < search_dirs.size(); i++) {
		const String &search_dir = search_dirs[i];

		if (has_extension) {
			path = search_dir.plus_file(name);
			if (FileAccess::exists(path)) {
				res = _load_assembly_from(name.get_basename(), path);
				break;
			}
		} else {
			path = search_dir.plus_file(name + ".dll");
			if (FileAccess::exists(path)) {
				res = _load_assembly_from(name, path);
				break;
			}

			path = search_dir.plus_file(name + ".exe");
			if (FileAccess::exists(path)) {
				res = _load_assembly_from(name, path);
				break;
			}
		}
	}

	no_search = false;

	return res;
}

MonoAssembly *GDMonoAssembly::_preload_hook(MonoAssemblyName *aname, char **assemblies_path, void *user_data) {

	(void)user_data; // UNUSED

	if (search_dirs.empty()) {
#ifdef TOOLS_DOMAIN
		search_dirs.push_back(GodotSharpDirs::get_res_temp_assemblies_dir());
#endif
		search_dirs.push_back(GodotSharpDirs::get_res_assemblies_dir());
		search_dirs.push_back(OS::get_singleton()->get_resource_dir());
		search_dirs.push_back(OS::get_singleton()->get_executable_path().get_base_dir());

		const char *rootdir = mono_assembly_getrootdir();
		if (rootdir) {
			search_dirs.push_back(String(rootdir).plus_file("mono").plus_file("4.5"));
		}

		if (assemblies_path) {
			while (*assemblies_path) {
				search_dirs.push_back(*assemblies_path);
				++assemblies_path;
			}
		}
	}

	return NULL;
}

MonoAssembly *GDMonoAssembly::_load_assembly_from(const String &p_name, const String &p_path) {

	GDMonoAssembly *assembly = memnew(GDMonoAssembly(p_name, p_path));

	MonoDomain *domain = mono_domain_get();

	Error err = assembly->load(domain);

	if (err != OK) {
		memdelete(assembly);
		ERR_FAIL_V(NULL);
	}

	GDMono::get_singleton()->add_assembly(domain ? mono_domain_get_id(domain) : 0, assembly);

	return assembly->get_assembly();
}

void GDMonoAssembly::initialize() {

	// TODO refonly as well?
	mono_install_assembly_preload_hook(&GDMonoAssembly::_preload_hook, NULL);
	mono_install_assembly_search_hook(&GDMonoAssembly::_search_hook, NULL);
}

Error GDMonoAssembly::load(MonoDomain *p_domain) {

	ERR_FAIL_COND_V(loaded, ERR_FILE_ALREADY_IN_USE);

	uint64_t last_modified_time = FileAccess::get_modified_time(path);

	Vector<uint8_t> data = FileAccess::get_file_as_array(path);
	ERR_FAIL_COND_V(data.empty(), ERR_FILE_CANT_READ);

	String image_filename(path);

	MonoImageOpenStatus status;

	image = mono_image_open_from_data_with_name(
			(char *)&data[0], data.size(),
			true, &status, false,
			image_filename.utf8().get_data());

	ERR_FAIL_COND_V(status != MONO_IMAGE_OK || image == NULL, ERR_FILE_CANT_OPEN);

#ifdef DEBUG_ENABLED
	String pdb_path(path + ".pdb");

	if (!FileAccess::exists(pdb_path)) {
		pdb_path = path.get_basename() + ".pdb"; // without .dll

		if (!FileAccess::exists(pdb_path))
			goto no_pdb;
	}

	pdb_data.clear();
	pdb_data = FileAccess::get_file_as_array(pdb_path);
	mono_debug_open_image_from_memory(image, &pdb_data[0], pdb_data.size());

no_pdb:

#endif

	assembly = mono_assembly_load_from_full(image, image_filename.utf8().get_data(), &status, false);

	ERR_FAIL_COND_V(status != MONO_IMAGE_OK || assembly == NULL, ERR_FILE_CANT_OPEN);

	if (p_domain && mono_image_get_entry_point(image)) {
		// TODO should this be removed? do we want to call main? what other effects does this have?
		mono_jit_exec(p_domain, assembly, 0, NULL);
	}

	loaded = true;
	modified_time = last_modified_time;

	return OK;
}

Error GDMonoAssembly::wrapper_for_image(MonoImage *p_image) {

	ERR_FAIL_COND_V(loaded, ERR_FILE_ALREADY_IN_USE);

	assembly = mono_image_get_assembly(p_image);
	ERR_FAIL_NULL_V(assembly, FAILED);

	image = p_image;

	mono_image_addref(image);

	loaded = true;

	return OK;
}

void GDMonoAssembly::unload() {

	ERR_FAIL_COND(!loaded);

#ifdef DEBUG_ENABLED
	if (pdb_data.size()) {
		mono_debug_close_image(image);
		pdb_data.clear();
	}
#endif

	for (Map<MonoClass *, GDMonoClass *>::Element *E = cached_raw.front(); E; E = E->next()) {
		memdelete(E->value());
	}

	cached_classes.clear();
	cached_raw.clear();

	mono_image_close(image);

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
					MonoClass *raw_nested = mono_class_get_nested_types(current_nested->get_raw(), &iter);

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

GDMonoAssembly::GDMonoAssembly(const String &p_name, const String &p_path) {

	loaded = false;
	gdobject_class_cache_updated = false;
	name = p_name;
	path = p_path;
	modified_time = 0;
	assembly = NULL;
	image = NULL;
}

GDMonoAssembly::~GDMonoAssembly() {

	if (loaded)
		unload();
}
