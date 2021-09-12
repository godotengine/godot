/*************************************************************************/
/*  gd_mono.h                                                            */
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

#ifndef GD_MONO_H
#define GD_MONO_H

#include "core/io/config_file.h"

#include "../godotsharp_defs.h"
#include "gd_mono_assembly.h"
#include "gd_mono_log.h"

#ifdef WINDOWS_ENABLED
#include "../utils/mono_reg_utils.h"
#endif

class GDMono {
public:
	enum UnhandledExceptionPolicy {
		POLICY_TERMINATE_APP,
		POLICY_LOG_ERROR
	};

private:
	bool runtime_initialized;
	bool finalizing_scripts_domain;

	UnhandledExceptionPolicy unhandled_exception_policy;

	MonoDomain *root_domain = nullptr;
	MonoDomain *scripts_domain = nullptr;

	HashMap<int32_t, HashMap<String, GDMonoAssembly *>> assemblies;

	GDMonoAssembly *corlib_assembly = nullptr;
	GDMonoAssembly *project_assembly = nullptr;
#ifdef TOOLS_ENABLED
	GDMonoAssembly *tools_assembly = nullptr;
	GDMonoAssembly *tools_project_editor_assembly = nullptr;
#endif

	GDMonoAssembly *core_api_assembly;
	GDMonoAssembly *editor_api_assembly;

	bool _load_core_api_assembly(GDMonoAssembly **r_loaded_api_assembly, const String &p_config);
#ifdef TOOLS_ENABLED
	bool _load_editor_api_assembly(GDMonoAssembly **r_loaded_api_assembly, const String &p_config);
#endif

	static bool _on_core_api_assembly_loaded();

	bool _load_corlib_assembly();
#ifdef TOOLS_ENABLED
	bool _load_tools_assemblies();
#endif
	bool _load_project_assembly();

	bool _try_load_api_assemblies();

	void _install_trace_listener();

	void _register_internal_calls();

#ifndef GD_MONO_SINGLE_APPDOMAIN
	Error _load_scripts_domain();
	Error _unload_scripts_domain();
#endif

	void _domain_assemblies_cleanup(int32_t p_domain_id);

	uint64_t api_core_hash;
#ifdef TOOLS_ENABLED
	uint64_t api_editor_hash;
#endif
	void _init_godot_api_hashes();
	void _init_exception_policy();

	GDMonoLog *gdmono_log = nullptr;

#if defined(WINDOWS_ENABLED) && defined(TOOLS_ENABLED)
	MonoRegInfo mono_reg_info;
#endif

	void add_mono_shared_libs_dir_to_path();
	void determine_mono_dirs(String &r_assembly_rootdir, String &r_config_dir);

protected:
	static GDMono *singleton;

public:
#ifdef DEBUG_METHODS_ENABLED
	uint64_t get_api_core_hash() {
		if (api_core_hash == 0) {
			api_core_hash = ClassDB::get_api_hash(ClassDB::API_CORE);
		}
		return api_core_hash;
	}
#ifdef TOOLS_ENABLED
	uint64_t get_api_editor_hash() {
		if (api_editor_hash == 0) {
			api_editor_hash = ClassDB::get_api_hash(ClassDB::API_EDITOR);
		}
		return api_editor_hash;
	}
#endif // TOOLS_ENABLED
#endif // DEBUG_METHODS_ENABLED

	_FORCE_INLINE_ static String get_expected_api_build_config() {
#ifdef TOOLS_ENABLED
		return "Debug";
#else
#ifdef DEBUG_ENABLED
		return "Debug";
#else
		return "Release";
#endif
#endif
	}

	static GDMono *get_singleton() { return singleton; }

	[[noreturn]] static void unhandled_exception_hook(MonoObject *p_exc, void *p_user_data);

	UnhandledExceptionPolicy get_unhandled_exception_policy() const { return unhandled_exception_policy; }

	// Do not use these, unless you know what you're doing
	void add_assembly(int32_t p_domain_id, GDMonoAssembly *p_assembly);
	GDMonoAssembly *get_loaded_assembly(const String &p_name);

	_FORCE_INLINE_ bool is_runtime_initialized() const { return runtime_initialized && !mono_runtime_is_shutting_down() /* stays true after shutdown finished */; }

	_FORCE_INLINE_ bool is_finalizing_scripts_domain() { return finalizing_scripts_domain; }

	_FORCE_INLINE_ MonoDomain *get_scripts_domain() { return scripts_domain; }

	_FORCE_INLINE_ GDMonoAssembly *get_corlib_assembly() const { return corlib_assembly; }
	_FORCE_INLINE_ GDMonoAssembly *get_core_api_assembly() const { return core_api_assembly; }
	_FORCE_INLINE_ GDMonoAssembly *get_project_assembly() const { return project_assembly; }
#ifdef TOOLS_ENABLED
	_FORCE_INLINE_ GDMonoAssembly *get_tools_assembly() const { return tools_assembly; }
#endif

#if defined(WINDOWS_ENABLED) && defined(TOOLS_ENABLED)
	const MonoRegInfo &get_mono_reg_info() { return mono_reg_info; }
#endif

#ifdef GD_MONO_HOT_RELOAD
	Error reload_scripts_domain();
#endif

	bool load_assembly(const String &p_name, GDMonoAssembly **r_assembly);
	bool load_assembly(const String &p_name, MonoAssemblyName *p_aname, GDMonoAssembly **r_assembly);
	bool load_assembly(const String &p_name, MonoAssemblyName *p_aname, GDMonoAssembly **r_assembly, const Vector<String> &p_search_dirs);
	bool load_assembly_from(const String &p_name, const String &p_path, GDMonoAssembly **r_assembly);

	Error finalize_and_unload_domain(MonoDomain *p_domain);

	void initialize();
	void initialize_load_assemblies();

	GDMono();
	~GDMono();
};

namespace gdmono {

class ScopeDomain {
	MonoDomain *prev_domain = nullptr;

public:
	ScopeDomain(MonoDomain *p_domain) {
		prev_domain = mono_domain_get();
		if (prev_domain != p_domain) {
			mono_domain_set(p_domain, false);
		} else {
			prev_domain = nullptr;
		}
	}

	~ScopeDomain() {
		if (prev_domain) {
			mono_domain_set(prev_domain, false);
		}
	}
};

class ScopeExitDomainUnload {
	MonoDomain *domain = nullptr;

public:
	ScopeExitDomainUnload(MonoDomain *p_domain) :
			domain(p_domain) {
	}

	~ScopeExitDomainUnload() {
		if (domain) {
			GDMono::get_singleton()->finalize_and_unload_domain(domain);
		}
	}
};
} // namespace gdmono

#define _GDMONO_SCOPE_DOMAIN_(m_mono_domain)                      \
	gdmono::ScopeDomain __gdmono__scope__domain__(m_mono_domain); \
	(void)__gdmono__scope__domain__;

#define _GDMONO_SCOPE_EXIT_DOMAIN_UNLOAD_(m_mono_domain)                                  \
	gdmono::ScopeExitDomainUnload __gdmono__scope__exit__domain__unload__(m_mono_domain); \
	(void)__gdmono__scope__exit__domain__unload__;

namespace mono_bind {

class GodotSharp : public Object {
	GDCLASS(GodotSharp, Object);

	friend class GDMono;

	bool _is_domain_finalizing_for_unload(int32_t p_domain_id);

	void _reload_assemblies(bool p_soft_reload);

protected:
	static GodotSharp *singleton;
	static void _bind_methods();

public:
	static GodotSharp *get_singleton() { return singleton; }

	void attach_thread();
	void detach_thread();

	int32_t get_domain_id();
	int32_t get_scripts_domain_id();

	bool is_scripts_domain_loaded();

	bool is_domain_finalizing_for_unload(int32_t p_domain_id);
	bool is_domain_finalizing_for_unload(MonoDomain *p_domain);

	bool is_runtime_shutting_down();
	bool is_runtime_initialized();

	GodotSharp();
	~GodotSharp();
};

} // namespace mono_bind

#endif // GD_MONO_H
