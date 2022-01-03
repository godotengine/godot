/*************************************************************************/
/*  gd_mono_assembly.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/string/ustring.h"
#include "core/templates/hash_map.h"
#include "core/templates/map.h"
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

	String name;
	MonoImage *image;
	MonoAssembly *assembly;

	bool attrs_fetched = false;
	MonoCustomAttrInfo *attributes = nullptr;

#ifdef GD_MONO_HOT_RELOAD
	uint64_t modified_time = 0;
#endif

	HashMap<ClassKey, GDMonoClass *, ClassKey::Hasher> cached_classes;
	Map<MonoClass *, GDMonoClass *> cached_raw;

	static Vector<String> search_dirs;

	static void assembly_load_hook(MonoAssembly *assembly, void *user_data);
	static MonoAssembly *assembly_search_hook(MonoAssemblyName *aname, void *user_data);
	static MonoAssembly *assembly_refonly_search_hook(MonoAssemblyName *aname, void *user_data);
	static MonoAssembly *assembly_preload_hook(MonoAssemblyName *aname, char **assemblies_path, void *user_data);
	static MonoAssembly *assembly_refonly_preload_hook(MonoAssemblyName *aname, char **assemblies_path, void *user_data);

	static MonoAssembly *_search_hook(MonoAssemblyName *aname, void *user_data, bool refonly);
	static MonoAssembly *_preload_hook(MonoAssemblyName *aname, char **assemblies_path, void *user_data, bool refonly);

	static MonoAssembly *_real_load_assembly_from(const String &p_path, bool p_refonly, MonoAssemblyName *p_aname = nullptr);
	static MonoAssembly *_load_assembly_search(const String &p_name, MonoAssemblyName *p_aname, bool p_refonly, const Vector<String> &p_search_dirs);

	friend class GDMono;
	static void initialize();

public:
	void unload();

	_FORCE_INLINE_ MonoImage *get_image() const { return image; }
	_FORCE_INLINE_ MonoAssembly *get_assembly() const { return assembly; }
	_FORCE_INLINE_ String get_name() const { return name; }

#ifdef GD_MONO_HOT_RELOAD
	_FORCE_INLINE_ uint64_t get_modified_time() const { return modified_time; }
#endif

	String get_path() const;

	bool has_attribute(GDMonoClass *p_attr_class);
	MonoObject *get_attribute(GDMonoClass *p_attr_class);

	void fetch_attributes();

	GDMonoClass *get_class(const StringName &p_namespace, const StringName &p_name);
	GDMonoClass *get_class(MonoClass *p_mono_class);

	static String find_assembly(const String &p_name);

	static void fill_search_dirs(Vector<String> &r_search_dirs, const String &p_custom_config = String(), const String &p_custom_bcl_dir = String());
	static const Vector<String> &get_default_search_dirs() { return search_dirs; }

	static GDMonoAssembly *load(const String &p_name, MonoAssemblyName *p_aname, bool p_refonly, const Vector<String> &p_search_dirs);
	static GDMonoAssembly *load_from(const String &p_name, const String &p_path, bool p_refonly);

	GDMonoAssembly(const String &p_name, MonoImage *p_image, MonoAssembly *p_assembly) :
			name(p_name),
			image(p_image),
			assembly(p_assembly) {
	}
	~GDMonoAssembly();
};

#endif // GD_MONO_ASSEMBLY_H
