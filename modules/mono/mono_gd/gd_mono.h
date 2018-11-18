/*************************************************************************/
/*  gd_mono.h                                                            */
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

#ifndef GD_MONO_H
#define GD_MONO_H

#include "core/io/config_file.h"

#include "../godotsharp_defs.h"
#include "gd_mono_assembly.h"
#include "gd_mono_log.h"

#ifdef WINDOWS_ENABLED
#include "../utils/mono_reg_utils.h"
#endif

namespace APIAssembly {
enum Type {
	API_CORE,
	API_EDITOR
};

struct Version {
	uint64_t godot_api_hash;
	uint32_t bindings_version;
	uint32_t cs_glue_version;

	bool operator==(const Version &p_other) const {
		return godot_api_hash == p_other.godot_api_hash &&
			   bindings_version == p_other.bindings_version &&
			   cs_glue_version == p_other.cs_glue_version;
	}

	Version() :
			godot_api_hash(0),
			bindings_version(0),
			cs_glue_version(0) {
	}

	Version(uint64_t p_godot_api_hash,
			uint32_t p_bindings_version,
			uint32_t p_cs_glue_version) :
			godot_api_hash(p_godot_api_hash),
			bindings_version(p_bindings_version),
			cs_glue_version(p_cs_glue_version) {
	}

	static Version get_from_loaded_assembly(GDMonoAssembly *p_api_assembly, Type p_api_type);
};

String to_string(Type p_type);
} // namespace APIAssembly

#define SCRIPTS_DOMAIN GDMono::get_singleton()->get_scripts_domain()
#ifdef TOOLS_ENABLED
#define TOOLS_DOMAIN GDMono::get_singleton()->get_tools_domain()
#endif

class GDMono {

	bool runtime_initialized;
	bool finalizing_scripts_domain;

	MonoDomain *root_domain;
	MonoDomain *scripts_domain;
#ifdef TOOLS_ENABLED
	MonoDomain *tools_domain;
#endif

	bool core_api_assembly_out_of_sync;
	bool editor_api_assembly_out_of_sync;

	GDMonoAssembly *corlib_assembly;
	GDMonoAssembly *core_api_assembly;
	GDMonoAssembly *project_assembly;
#ifdef TOOLS_ENABLED
	GDMonoAssembly *editor_api_assembly;
	GDMonoAssembly *editor_tools_assembly;
#endif

	HashMap<uint32_t, HashMap<String, GDMonoAssembly *> > assemblies;

	void _domain_assemblies_cleanup(uint32_t p_domain_id);

	bool _load_corlib_assembly();
	bool _load_core_api_assembly();
#ifdef TOOLS_ENABLED
	bool _load_editor_api_assembly();
	bool _load_editor_tools_assembly();
#endif
	bool _load_project_assembly();

	bool _load_api_assemblies();

#ifdef TOOLS_ENABLED
	String _get_api_assembly_metadata_path();
#endif

	void _register_internal_calls();

	Error _load_scripts_domain();
	Error _unload_scripts_domain();

#ifdef TOOLS_ENABLED
	Error _load_tools_domain();
#endif

#ifdef DEBUG_METHODS_ENABLED
	uint64_t api_core_hash;
#ifdef TOOLS_ENABLED
	uint64_t api_editor_hash;
#endif
	void _initialize_and_check_api_hashes();
#endif

	GDMonoLog *gdmono_log;

#if defined(WINDOWS_ENABLED) && defined(TOOLS_ENABLED)
	MonoRegInfo mono_reg_info;
#endif

protected:
	static GDMono *singleton;

public:
#ifdef DEBUG_METHODS_ENABLED
	uint64_t get_api_core_hash() { return api_core_hash; }
#ifdef TOOLS_ENABLED
	uint64_t get_api_editor_hash() { return api_editor_hash; }
#endif
#endif

#ifdef TOOLS_ENABLED
	void metadata_set_api_assembly_invalidated(APIAssembly::Type p_api_type, bool p_invalidated);
	bool metadata_is_api_assembly_invalidated(APIAssembly::Type p_api_type);
#endif

	static GDMono *get_singleton() { return singleton; }

	static void unhandled_exception_hook(MonoObject *p_exc, void *p_user_data);

	// Do not use these, unless you know what you're doing
	void add_assembly(uint32_t p_domain_id, GDMonoAssembly *p_assembly);
	GDMonoAssembly **get_loaded_assembly(const String &p_name);

	_FORCE_INLINE_ bool is_runtime_initialized() const { return runtime_initialized && !mono_runtime_is_shutting_down() /* stays true after shutdown finished */; }

	_FORCE_INLINE_ bool is_finalizing_scripts_domain() { return finalizing_scripts_domain; }

	_FORCE_INLINE_ MonoDomain *get_scripts_domain() { return scripts_domain; }
#ifdef TOOLS_ENABLED
	_FORCE_INLINE_ MonoDomain *get_tools_domain() { return tools_domain; }
#endif

	_FORCE_INLINE_ GDMonoAssembly *get_corlib_assembly() const { return corlib_assembly; }
	_FORCE_INLINE_ GDMonoAssembly *get_core_api_assembly() const { return core_api_assembly; }
	_FORCE_INLINE_ GDMonoAssembly *get_project_assembly() const { return project_assembly; }
#ifdef TOOLS_ENABLED
	_FORCE_INLINE_ GDMonoAssembly *get_editor_api_assembly() const { return editor_api_assembly; }
	_FORCE_INLINE_ GDMonoAssembly *get_editor_tools_assembly() const { return editor_tools_assembly; }
#endif

#if defined(WINDOWS_ENABLED) && defined(TOOLS_ENABLED)
	const MonoRegInfo &get_mono_reg_info() { return mono_reg_info; }
#endif

	GDMonoClass *get_class(MonoClass *p_raw_class);

#ifdef TOOLS_ENABLED
	Error reload_scripts_domain();
#endif

	bool load_assembly(const String &p_name, GDMonoAssembly **r_assembly, bool p_refonly = false);
	bool load_assembly(const String &p_name, MonoAssemblyName *p_aname, GDMonoAssembly **r_assembly, bool p_refonly = false);
	bool load_assembly_from(const String &p_name, const String &p_path, GDMonoAssembly **r_assembly, bool p_refonly = false);

	Error finalize_and_unload_domain(MonoDomain *p_domain);

	void initialize();

	GDMono();
	~GDMono();
};

namespace gdmono {

class ScopeDomain {

	MonoDomain *prev_domain;

public:
	ScopeDomain(MonoDomain *p_domain) {
		MonoDomain *prev_domain = mono_domain_get();
		if (prev_domain != p_domain) {
			this->prev_domain = prev_domain;
			mono_domain_set(p_domain, false);
		} else {
			this->prev_domain = NULL;
		}
	}

	~ScopeDomain() {
		if (prev_domain)
			mono_domain_set(prev_domain, false);
	}
};

class ScopeExitDomainUnload {
	MonoDomain *domain;

public:
	ScopeExitDomainUnload(MonoDomain *p_domain) :
			domain(p_domain) {
	}

	~ScopeExitDomainUnload() {
		if (domain)
			GDMono::get_singleton()->finalize_and_unload_domain(domain);
	}
};

} // namespace gdmono

#define _GDMONO_SCOPE_DOMAIN_(m_mono_domain)                      \
	gdmono::ScopeDomain __gdmono__scope__domain__(m_mono_domain); \
	(void)__gdmono__scope__domain__;

#define _GDMONO_SCOPE_EXIT_DOMAIN_UNLOAD_(m_mono_domain)                                  \
	gdmono::ScopeExitDomainUnload __gdmono__scope__exit__domain__unload__(m_mono_domain); \
	(void)__gdmono__scope__exit__domain__unload__;

class _GodotSharp : public Object {
	GDCLASS(_GodotSharp, Object)

	friend class GDMono;

	bool _is_domain_finalizing_for_unload(int32_t p_domain_id);

	List<NodePath *> np_delete_queue;
	List<RID *> rid_delete_queue;

	bool queue_empty;

#ifndef NO_THREADS
	Mutex *queue_mutex;
#endif

protected:
	static _GodotSharp *singleton;
	static void _bind_methods();

public:
	static _GodotSharp *get_singleton() { return singleton; }

	void attach_thread();
	void detach_thread();

	int32_t get_domain_id();
	int32_t get_scripts_domain_id();

	bool is_scripts_domain_loaded();

	bool is_domain_finalizing_for_unload();
	bool is_domain_finalizing_for_unload(int32_t p_domain_id);
	bool is_domain_finalizing_for_unload(MonoDomain *p_domain);

	bool is_runtime_shutting_down();
	bool is_runtime_initialized();

	_GodotSharp();
	~_GodotSharp();
};

#endif // GD_MONO_H
