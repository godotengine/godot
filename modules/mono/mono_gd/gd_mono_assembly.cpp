/**************************************************************************/
/*  gd_mono_assembly.cpp                                                  */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "gd_mono_assembly.h"

#include <mono/metadata/mono-debug.h>
#include <mono/metadata/tokentype.h>

#include "core/io/file_access_pack.h"
#include "core/list.h"
#include "core/os/file_access.h"
#include "core/os/os.h"
#include "core/project_settings.h"

#include "../godotsharp_dirs.h"
#include "gd_mono_cache.h"
#include "gd_mono_class.h"

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

#if !defined(TOOLS_ENABLED)
	String data_game_assemblies_dir = GodotSharpDirs::get_data_game_assemblies_dir();
	if (!data_game_assemblies_dir.empty()) {
		r_search_dirs.push_back(data_game_assemblies_dir);
	}
#endif

	if (p_custom_config.length()) {
		r_search_dirs.push_back(GodotSharpDirs::get_res_temp_assemblies_base_dir().plus_file(p_custom_config));
	} else {
		r_search_dirs.push_back(GodotSharpDirs::get_res_temp_assemblies_dir());
	}

	if (p_custom_config.empty()) {
		r_search_dirs.push_back(GodotSharpDirs::get_res_assemblies_dir());
	} else {
		String api_config = p_custom_config == "ExportRelease" ? "Release" : "Debug";
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

// This is how these assembly loading hooks work:
//
// - The 'search' hook checks if the assembly has already been loaded, to avoid loading again.
// - The 'preload' hook does the actual loading and is only called if the
//   'search' hook didn't find the assembly in the list of loaded assemblies.
// - The 'load' hook is called after the assembly has been loaded. Its job is to add the
//   assembly to the list of loaded assemblies so that the 'search' hook can look it up.

void GDMonoAssembly::assembly_load_hook(MonoAssembly *assembly, void *user_data) {
	String name = String::utf8(mono_assembly_name_get_name(mono_assembly_get_name(assembly)));

	MonoImage *image = mono_assembly_get_image(assembly);

	GDMonoAssembly *gdassembly = memnew(GDMonoAssembly(name, image, assembly));

#ifdef GD_MONO_HOT_RELOAD
	String path = String::utf8(mono_image_get_filename(image));
	if (FileAccess::exists(path))
		gdassembly->modified_time = FileAccess::get_modified_time(path);
#endif

	MonoDomain *domain = mono_domain_get();
	GDMono::get_singleton()->add_assembly(domain ? mono_domain_get_id(domain) : 0, gdassembly);
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
	String name = String::utf8(mono_assembly_name_get_name(aname));
	bool has_extension = name.ends_with(".dll") || name.ends_with(".exe");

	GDMonoAssembly *loaded_asm = GDMono::get_singleton()->get_loaded_assembly(has_extension ? name.get_basename() : name);
	if (loaded_asm)
		return loaded_asm->get_assembly();

	return NULL;
}

MonoAssembly *GDMonoAssembly::_preload_hook(MonoAssemblyName *aname, char **, void *user_data, bool refonly) {
	String name = String::utf8(mono_assembly_name_get_name(aname));
	return _load_assembly_search(name, aname, refonly, search_dirs);
}

MonoAssembly *GDMonoAssembly::_load_assembly_search(const String &p_name, MonoAssemblyName *p_aname, bool p_refonly, const Vector<String> &p_search_dirs) {
	MonoAssembly *res = NULL;
	String path;

	bool has_extension = p_name.ends_with(".dll") || p_name.ends_with(".exe");

	for (int i = 0; i < p_search_dirs.size(); i++) {
		const String &search_dir = p_search_dirs[i];

		if (has_extension) {
			path = search_dir.plus_file(p_name);
			if (FileAccess::exists(path)) {
				res = _real_load_assembly_from(path, p_refonly, p_aname);
				if (res != NULL)
					return res;
			}
		} else {
			path = search_dir.plus_file(p_name + ".dll");
			if (FileAccess::exists(path)) {
				res = _real_load_assembly_from(path, p_refonly, p_aname);
				if (res != NULL)
					return res;
			}

			path = search_dir.plus_file(p_name + ".exe");
			if (FileAccess::exists(path)) {
				res = _real_load_assembly_from(path, p_refonly, p_aname);
				if (res != NULL)
					return res;
			}
		}
	}

	return NULL;
}

String GDMonoAssembly::find_assembly(const String &p_name) {
	String path;

	bool has_extension = p_name.ends_with(".dll") || p_name.ends_with(".exe");

	for (int i = 0; i < search_dirs.size(); i++) {
		const String &search_dir = search_dirs[i];

		if (has_extension) {
			path = search_dir.plus_file(p_name);
			if (FileAccess::exists(path))
				return path;
		} else {
			path = search_dir.plus_file(p_name + ".dll");
			if (FileAccess::exists(path))
				return path;

			path = search_dir.plus_file(p_name + ".exe");
			if (FileAccess::exists(path))
				return path;
		}
	}

	return String();
}

void GDMonoAssembly::initialize() {
	fill_search_dirs(search_dirs);

	mono_install_assembly_search_hook(&assembly_search_hook, NULL);
	mono_install_assembly_refonly_search_hook(&assembly_refonly_search_hook, NULL);
	mono_install_assembly_preload_hook(&assembly_preload_hook, NULL);
	mono_install_assembly_refonly_preload_hook(&assembly_refonly_preload_hook, NULL);
	mono_install_assembly_load_hook(&assembly_load_hook, NULL);
}

MonoAssembly *GDMonoAssembly::_real_load_assembly_from(const String &p_path, bool p_refonly, MonoAssemblyName *p_aname) {
	Vector<uint8_t> data = FileAccess::get_file_as_array(p_path);
	ERR_FAIL_COND_V_MSG(data.empty(), NULL, "Could read the assembly in the specified location");

	String image_filename;

#ifdef ANDROID_ENABLED
	if (p_path.begins_with("res://")) {
		image_filename = p_path.substr(6, p_path.length());
	} else {
		image_filename = ProjectSettings::get_singleton()->globalize_path(p_path);
	}
#else
	// FIXME: globalize_path does not work on exported games
	image_filename = ProjectSettings::get_singleton()->globalize_path(p_path);
#endif

	MonoImageOpenStatus status = MONO_IMAGE_OK;

	MonoImage *image = mono_image_open_from_data_with_name(
			(char *)&data[0], data.size(),
			true, &status, p_refonly,
			image_filename.utf8());

	ERR_FAIL_COND_V_MSG(status != MONO_IMAGE_OK || !image, NULL, "Failed to open assembly image from memory: '" + p_path + "'.");

	if (p_aname != nullptr) {
		// Check assembly version
		const MonoTableInfo *table = mono_image_get_table_info(image, MONO_TABLE_ASSEMBLY);

		ERR_FAIL_NULL_V(table, nullptr);

		if (mono_table_info_get_rows(table)) {
			uint32_t cols[MONO_ASSEMBLY_SIZE];
			mono_metadata_decode_row(table, 0, cols, MONO_ASSEMBLY_SIZE);

			// Not sure about .NET's policy. We will only ensure major and minor are equal, and ignore build and revision.
			uint16_t major = cols[MONO_ASSEMBLY_MAJOR_VERSION];
			uint16_t minor = cols[MONO_ASSEMBLY_MINOR_VERSION];

			uint16_t required_minor;
			uint16_t required_major = mono_assembly_name_get_version(p_aname, &required_minor, nullptr, nullptr);

			if (required_major != 0) {
				if (major != required_major && minor != required_minor) {
					mono_image_close(image);
					return nullptr;
				}
			}
		}
	}

#ifdef DEBUG_ENABLED
	Vector<uint8_t> pdb_data;
	String pdb_path(p_path + ".pdb");

	if (!FileAccess::exists(pdb_path)) {
		pdb_path = p_path.get_basename() + ".pdb"; // without .dll

		if (!FileAccess::exists(pdb_path))
			goto no_pdb;
	}

	pdb_data = FileAccess::get_file_as_array(pdb_path);

	// mono_debug_close_image doesn't seem to be needed
	mono_debug_open_image_from_memory(image, &pdb_data[0], pdb_data.size());

no_pdb:

#endif

	bool need_manual_load_hook = mono_image_get_assembly(image) != nullptr; // Re-using an existing image with an assembly loaded

	status = MONO_IMAGE_OK;

	MonoAssembly *assembly = mono_assembly_load_from_full(image, image_filename.utf8().get_data(), &status, p_refonly);

	ERR_FAIL_COND_V_MSG(status != MONO_IMAGE_OK || !assembly, nullptr, "Failed to load assembly for image");

	if (need_manual_load_hook) {
		// For some reason if an assembly survived domain reloading (maybe because it's referenced somewhere else),
		// the mono internal search hook don't detect it, yet mono_image_open_from_data_with_name re-uses the image
		// and assembly, and mono_assembly_load_from_full doesn't call the load hook. We need to call it manually.
		String name = String::utf8(mono_assembly_name_get_name(mono_assembly_get_name(assembly)));
		bool has_extension = name.ends_with(".dll") || name.ends_with(".exe");
		GDMonoAssembly *loaded_asm = GDMono::get_singleton()->get_loaded_assembly(has_extension ? name.get_basename() : name);
		if (!loaded_asm)
			assembly_load_hook(assembly, nullptr);
	}

	// Decrement refcount which was previously incremented by mono_image_open_from_data_with_name
	mono_image_close(image);

	return assembly;
}

void GDMonoAssembly::unload() {
	ERR_FAIL_NULL(image); // Should not be called if already unloaded

	for (Map<MonoClass *, GDMonoClass *>::Element *E = cached_raw.front(); E; E = E->next()) {
		memdelete(E->value());
	}

	cached_classes.clear();
	cached_raw.clear();

	assembly = NULL;
	image = NULL;
}

String GDMonoAssembly::get_path() const {
	return String::utf8(mono_image_get_filename(image));
}

GDMonoClass *GDMonoAssembly::get_class(const StringName &p_namespace, const StringName &p_name) {
	ERR_FAIL_NULL_V(image, NULL);

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
	ERR_FAIL_NULL_V(image, NULL);

	Map<MonoClass *, GDMonoClass *>::Element *match = cached_raw.find(p_mono_class);

	if (match)
		return match->value();

	StringName namespace_name = String::utf8(mono_class_get_namespace(p_mono_class));
	StringName class_name = String::utf8(mono_class_get_name(p_mono_class));

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
				nested_classes.pop_front();

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

GDMonoAssembly *GDMonoAssembly::load(const String &p_name, MonoAssemblyName *p_aname, bool p_refonly, const Vector<String> &p_search_dirs) {
	if (GDMono::get_singleton()->get_corlib_assembly() && (p_name == "mscorlib" || p_name == "mscorlib.dll"))
		return GDMono::get_singleton()->get_corlib_assembly();

	// We need to manually call the search hook in this case, as it won't be called in the next step
	MonoAssembly *assembly = mono_assembly_invoke_search_hook(p_aname);

	if (!assembly) {
		assembly = _load_assembly_search(p_name, p_aname, p_refonly, p_search_dirs);
		if (!assembly) {
			return nullptr;
		}
	}

	GDMonoAssembly *loaded_asm = GDMono::get_singleton()->get_loaded_assembly(p_name);
	ERR_FAIL_NULL_V_MSG(loaded_asm, nullptr, "Loaded assembly missing from table. Did we not receive the load hook?");
	ERR_FAIL_COND_V(loaded_asm->get_assembly() != assembly, nullptr);

	return loaded_asm;
}

GDMonoAssembly *GDMonoAssembly::load_from(const String &p_name, const String &p_path, bool p_refonly) {
	if (p_name == "mscorlib" || p_name == "mscorlib.dll")
		return GDMono::get_singleton()->get_corlib_assembly();

	// We need to manually call the search hook in this case, as it won't be called in the next step
	MonoAssemblyName *aname = mono_assembly_name_new(p_name.utf8());
	MonoAssembly *assembly = mono_assembly_invoke_search_hook(aname);
	mono_assembly_name_free(aname);
	mono_free(aname);

	if (!assembly) {
		assembly = _real_load_assembly_from(p_path, p_refonly);
		if (!assembly) {
			return nullptr;
		}
	}

	GDMonoAssembly *loaded_asm = GDMono::get_singleton()->get_loaded_assembly(p_name);
	ERR_FAIL_NULL_V_MSG(loaded_asm, NULL, "Loaded assembly missing from table. Did we not receive the load hook?");

	return loaded_asm;
}

GDMonoAssembly::GDMonoAssembly(const String &p_name, MonoImage *p_image, MonoAssembly *p_assembly) :
		name(p_name),
		image(p_image),
		assembly(p_assembly),
#ifdef GD_MONO_HOT_RELOAD
		modified_time(0),
#endif
		gdobject_class_cache_updated(false) {
}

GDMonoAssembly::~GDMonoAssembly() {
	if (image)
		unload();
}
