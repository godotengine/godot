/*************************************************************************/
/*  gd_mono_assembly.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef GD_MONO_ASSEMBLY_H
#define GD_MONO_ASSEMBLY_H

#include <mono/jit/jit.h>
#include <mono/metadata/assembly.h>

#include "core/hash_map.h"
#include "core/map.h"
#include "core/ustring.h"
#include "gd_mono_utils.h"

class GDMonoAssembly {

	struct ClassKey {
		struct Hasher {
			static _FORCE_INLINE_ uint32_t hash(const ClassKey &p_key) {
				uint32_t hash = 0;

				GDMonoUtils::hash_combine(hash, p_key.namespace_name.hash());
				GDMonoUtils::hash_combine(hash, p_key.class_name.hash());

				return hash;
			}
		};

		_FORCE_INLINE_ bool operator==(const ClassKey &p_a) const {
			return p_a.class_name == class_name && p_a.namespace_name == namespace_name;
		}

		ClassKey() {}

		ClassKey(const StringName &p_namespace_name, const StringName &p_class_name) {
			namespace_name = p_namespace_name;
			class_name = p_class_name;
		}

		StringName namespace_name;
		StringName class_name;
	};

	MonoAssembly *assembly;
	MonoImage *image;

	bool refonly;
	bool loaded;

	String name;
	String path;
	uint64_t modified_time;

	HashMap<ClassKey, GDMonoClass *, ClassKey::Hasher> cached_classes;
	Map<MonoClass *, GDMonoClass *> cached_raw;

	bool gdobject_class_cache_updated;
	Map<StringName, GDMonoClass *> gdobject_class_cache;

#ifdef DEBUG_ENABLED
	Vector<uint8_t> pdb_data;
#endif

	static bool no_search;
	static bool in_preload;
	static Vector<String> search_dirs;

	static void assembly_load_hook(MonoAssembly *assembly, void *user_data);
	static MonoAssembly *assembly_search_hook(MonoAssemblyName *aname, void *user_data);
	static MonoAssembly *assembly_refonly_search_hook(MonoAssemblyName *aname, void *user_data);
	static MonoAssembly *assembly_preload_hook(MonoAssemblyName *aname, char **assemblies_path, void *user_data);
	static MonoAssembly *assembly_refonly_preload_hook(MonoAssemblyName *aname, char **assemblies_path, void *user_data);

	static MonoAssembly *_search_hook(MonoAssemblyName *aname, void *user_data, bool refonly);
	static MonoAssembly *_preload_hook(MonoAssemblyName *aname, char **assemblies_path, void *user_data, bool refonly);

	static GDMonoAssembly *_load_assembly_from(const String &p_name, const String &p_path, bool p_refonly);
	static GDMonoAssembly *_load_assembly_search(const String &p_name, const Vector<String> &p_search_dirs, bool p_refonly);
	static void _wrap_mono_assembly(MonoAssembly *assembly);

	friend class GDMono;
	static void initialize();

public:
	Error load(bool p_refonly);
	Error wrapper_for_image(MonoImage *p_image);
	void unload();

	_FORCE_INLINE_ bool is_refonly() const { return refonly; }
	_FORCE_INLINE_ bool is_loaded() const { return loaded; }
	_FORCE_INLINE_ MonoImage *get_image() const { return image; }
	_FORCE_INLINE_ MonoAssembly *get_assembly() const { return assembly; }
	_FORCE_INLINE_ String get_name() const { return name; }
	_FORCE_INLINE_ String get_path() const { return path; }
	_FORCE_INLINE_ uint64_t get_modified_time() const { return modified_time; }

	GDMonoClass *get_class(const StringName &p_namespace, const StringName &p_name);
	GDMonoClass *get_class(MonoClass *p_mono_class);

	GDMonoClass *get_object_derived_class(const StringName &p_class);

	static void fill_search_dirs(Vector<String> &r_search_dirs, const String &p_custom_config = String());

	static GDMonoAssembly *load_from(const String &p_name, const String &p_path, bool p_refonly);

	GDMonoAssembly(const String &p_name, const String &p_path = String());
	~GDMonoAssembly();
};

#endif // GD_MONO_ASSEMBLY_H
