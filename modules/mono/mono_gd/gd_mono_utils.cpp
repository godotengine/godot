/*************************************************************************/
/*  gd_mono_utils.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "gd_mono_utils.h"

#include <mono/metadata/debug-helpers.h>
#include <mono/metadata/exception.h>

#include "core/os/dir_access.h"
#include "core/os/os.h"
#include "core/project_settings.h"
#include "core/reference.h"

#ifdef TOOLS_ENABLED
#include "editor/script_editor_debugger.h"
#endif

#include "../csharp_script.h"
#include "../utils/macros.h"
#include "gd_mono.h"
#include "gd_mono_cache.h"
#include "gd_mono_class.h"
#include "gd_mono_marshal.h"
#include "gd_mono_method_thunk.h"

namespace GDMonoUtils {

MonoObject *unmanaged_get_managed(Object *unmanaged) {
	if (!unmanaged)
		return NULL;

	if (unmanaged->get_script_instance()) {
		CSharpInstance *cs_instance = CAST_CSHARP_INSTANCE(unmanaged->get_script_instance());

		if (cs_instance) {
			return cs_instance->get_mono_object();
		}
	}

	// If the owner does not have a CSharpInstance...

	void *data = unmanaged->get_script_instance_binding(CSharpLanguage::get_singleton()->get_language_index());

	ERR_FAIL_NULL_V(data, NULL);

	CSharpScriptBinding &script_binding = ((Map<Object *, CSharpScriptBinding>::Element *)data)->value();

	if (!script_binding.inited) {
		MutexLock lock(CSharpLanguage::get_singleton()->get_language_bind_mutex());

		if (!script_binding.inited) { // Other thread may have set it up
			// Already had a binding that needs to be setup
			CSharpLanguage::get_singleton()->setup_csharp_script_binding(script_binding, unmanaged);

			ERR_FAIL_COND_V(!script_binding.inited, NULL);
		}
	}

	Ref<MonoGCHandle> &gchandle = script_binding.gchandle;
	ERR_FAIL_COND_V(gchandle.is_null(), NULL);

	MonoObject *target = gchandle->get_target();

	if (target)
		return target;

	CSharpLanguage::get_singleton()->release_script_gchandle(gchandle);

	// Create a new one

#ifdef DEBUG_ENABLED
	CRASH_COND(script_binding.type_name == StringName());
	CRASH_COND(script_binding.wrapper_class == NULL);
#endif

	MonoObject *mono_object = GDMonoUtils::create_managed_for_godot_object(script_binding.wrapper_class, script_binding.type_name, unmanaged);
	ERR_FAIL_NULL_V(mono_object, NULL);

	gchandle->set_handle(MonoGCHandle::new_strong_handle(mono_object), MonoGCHandle::STRONG_HANDLE);

	// Tie managed to unmanaged
	Reference *ref = Object::cast_to<Reference>(unmanaged);

	if (ref) {
		// Unsafe refcount increment. The managed instance also counts as a reference.
		// This way if the unmanaged world has no references to our owner
		// but the managed instance is alive, the refcount will be 1 instead of 0.
		// See: godot_icall_Reference_Dtor(MonoObject *p_obj, Object *p_ptr)
		ref->reference();
		CSharpLanguage::get_singleton()->post_unsafe_reference(ref);
	}

	return mono_object;
}

void set_main_thread(MonoThread *p_thread) {
	mono_thread_set_main(p_thread);
}

MonoThread *attach_current_thread() {
	ERR_FAIL_COND_V(!GDMono::get_singleton()->is_runtime_initialized(), NULL);
	MonoDomain *scripts_domain = GDMono::get_singleton()->get_scripts_domain();
#ifndef GD_MONO_SINGLE_APPDOMAIN
	MonoThread *mono_thread = mono_thread_attach(scripts_domain ? scripts_domain : mono_get_root_domain());
#else
	// The scripts domain is the root domain
	MonoThread *mono_thread = mono_thread_attach(scripts_domain);
#endif
	ERR_FAIL_NULL_V(mono_thread, NULL);
	return mono_thread;
}

void detach_current_thread() {
	ERR_FAIL_COND(!GDMono::get_singleton()->is_runtime_initialized());
	MonoThread *mono_thread = mono_thread_current();
	ERR_FAIL_NULL(mono_thread);
	mono_thread_detach(mono_thread);
}

void detach_current_thread(MonoThread *p_mono_thread) {
	ERR_FAIL_COND(!GDMono::get_singleton()->is_runtime_initialized());
	ERR_FAIL_NULL(p_mono_thread);
	mono_thread_detach(p_mono_thread);
}

MonoThread *get_current_thread() {
	return mono_thread_current();
}

bool is_thread_attached() {
	return mono_domain_get() != NULL;
}

void runtime_object_init(MonoObject *p_this_obj, GDMonoClass *p_class, MonoException **r_exc) {
	GDMonoMethod *ctor = p_class->get_method(".ctor", 0);
	ERR_FAIL_NULL(ctor);
	ctor->invoke_raw(p_this_obj, NULL, r_exc);
}

GDMonoClass *get_object_class(MonoObject *p_object) {
	return GDMono::get_singleton()->get_class(mono_object_get_class(p_object));
}

GDMonoClass *type_get_proxy_class(const StringName &p_type) {
	String class_name = p_type;

	if (class_name[0] == '_')
		class_name = class_name.substr(1, class_name.length());

	GDMonoClass *klass = GDMono::get_singleton()->get_core_api_assembly()->get_class(BINDINGS_NAMESPACE, class_name);

	if (klass && klass->is_static()) {
		// A static class means this is a Godot singleton class. If an instance is needed we use Godot.Object.
		return GDMonoCache::cached_data.class_GodotObject;
	}

#ifdef TOOLS_ENABLED
	if (!klass) {
		return GDMono::get_singleton()->get_editor_api_assembly()->get_class(BINDINGS_NAMESPACE, class_name);
	}
#endif

	return klass;
}

GDMonoClass *get_class_native_base(GDMonoClass *p_class) {
	GDMonoClass *klass = p_class;

	do {
		const GDMonoAssembly *assembly = klass->get_assembly();
		if (assembly == GDMono::get_singleton()->get_core_api_assembly())
			return klass;
#ifdef TOOLS_ENABLED
		if (assembly == GDMono::get_singleton()->get_editor_api_assembly())
			return klass;
#endif
	} while ((klass = klass->get_parent_class()) != NULL);

	return NULL;
}

MonoObject *create_managed_for_godot_object(GDMonoClass *p_class, const StringName &p_native, Object *p_object) {
	bool parent_is_object_class = ClassDB::is_parent_class(p_object->get_class_name(), p_native);
	ERR_FAIL_COND_V_MSG(!parent_is_object_class, NULL,
			"Type inherits from native type '" + p_native + "', so it can't be instanced in object of type: '" + p_object->get_class() + "'.");

	MonoObject *mono_object = mono_object_new(mono_domain_get(), p_class->get_mono_ptr());
	ERR_FAIL_NULL_V(mono_object, NULL);

	CACHED_FIELD(GodotObject, ptr)->set_value_raw(mono_object, p_object);

	// Construct
	GDMonoUtils::runtime_object_init(mono_object, p_class);

	return mono_object;
}

MonoObject *create_managed_from(const NodePath &p_from) {
	MonoObject *mono_object = mono_object_new(mono_domain_get(), CACHED_CLASS_RAW(NodePath));
	ERR_FAIL_NULL_V(mono_object, NULL);

	// Construct
	GDMonoUtils::runtime_object_init(mono_object, CACHED_CLASS(NodePath));

	CACHED_FIELD(NodePath, ptr)->set_value_raw(mono_object, memnew(NodePath(p_from)));

	return mono_object;
}

MonoObject *create_managed_from(const RID &p_from) {
	MonoObject *mono_object = mono_object_new(mono_domain_get(), CACHED_CLASS_RAW(RID));
	ERR_FAIL_NULL_V(mono_object, NULL);

	// Construct
	GDMonoUtils::runtime_object_init(mono_object, CACHED_CLASS(RID));

	CACHED_FIELD(RID, ptr)->set_value_raw(mono_object, memnew(RID(p_from)));

	return mono_object;
}

MonoObject *create_managed_from(const Array &p_from, GDMonoClass *p_class) {
	MonoObject *mono_object = mono_object_new(mono_domain_get(), p_class->get_mono_ptr());
	ERR_FAIL_NULL_V(mono_object, NULL);

	// Search constructor that takes a pointer as parameter
	MonoMethod *m;
	void *iter = NULL;
	while ((m = mono_class_get_methods(p_class->get_mono_ptr(), &iter))) {
		if (strcmp(mono_method_get_name(m), ".ctor") == 0) {
			MonoMethodSignature *sig = mono_method_signature(m);
			void *front = NULL;
			if (mono_signature_get_param_count(sig) == 1 &&
					mono_class_from_mono_type(mono_signature_get_params(sig, &front)) == CACHED_CLASS(IntPtr)->get_mono_ptr()) {
				break;
			}
		}
	}

	CRASH_COND(m == NULL);

	Array *new_array = memnew(Array(p_from));
	void *args[1] = { &new_array };

	MonoException *exc = NULL;
	GDMonoUtils::runtime_invoke(m, mono_object, args, &exc);
	UNHANDLED_EXCEPTION(exc);

	return mono_object;
}

MonoObject *create_managed_from(const Dictionary &p_from, GDMonoClass *p_class) {
	MonoObject *mono_object = mono_object_new(mono_domain_get(), p_class->get_mono_ptr());
	ERR_FAIL_NULL_V(mono_object, NULL);

	// Search constructor that takes a pointer as parameter
	MonoMethod *m;
	void *iter = NULL;
	while ((m = mono_class_get_methods(p_class->get_mono_ptr(), &iter))) {
		if (strcmp(mono_method_get_name(m), ".ctor") == 0) {
			MonoMethodSignature *sig = mono_method_signature(m);
			void *front = NULL;
			if (mono_signature_get_param_count(sig) == 1 &&
					mono_class_from_mono_type(mono_signature_get_params(sig, &front)) == CACHED_CLASS(IntPtr)->get_mono_ptr()) {
				break;
			}
		}
	}

	CRASH_COND(m == NULL);

	Dictionary *new_dict = memnew(Dictionary(p_from));
	void *args[1] = { &new_dict };

	MonoException *exc = NULL;
	GDMonoUtils::runtime_invoke(m, mono_object, args, &exc);
	UNHANDLED_EXCEPTION(exc);

	return mono_object;
}

MonoDomain *create_domain(const String &p_friendly_name) {
	print_verbose("Mono: Creating domain '" + p_friendly_name + "'...");

	MonoDomain *domain = mono_domain_create_appdomain((char *)p_friendly_name.utf8().get_data(), NULL);

	if (domain) {
		// Workaround to avoid this exception:
		// System.Configuration.ConfigurationErrorsException: Error Initializing the configuration system.
		// ---> System.ArgumentException: The 'ExeConfigFilename' argument cannot be null.
		mono_domain_set_config(domain, ".", "");
	}

	return domain;
}

String get_type_desc(MonoType *p_type) {
	return mono_type_full_name(p_type);
}

String get_type_desc(MonoReflectionType *p_reftype) {
	return get_type_desc(mono_reflection_type_get_type(p_reftype));
}

String get_exception_name_and_message(MonoException *p_exc) {
	String res;

	MonoClass *klass = mono_object_get_class((MonoObject *)p_exc);
	MonoType *type = mono_class_get_type(klass);

	char *full_name = mono_type_full_name(type);
	res += full_name;
	mono_free(full_name);

	res += ": ";

	MonoProperty *prop = mono_class_get_property_from_name(klass, "Message");
	MonoString *msg = (MonoString *)property_get_value(prop, (MonoObject *)p_exc, NULL, NULL);
	res += GDMonoMarshal::mono_string_to_godot(msg);

	return res;
}

void set_exception_message(MonoException *p_exc, String message) {
	MonoClass *klass = mono_object_get_class((MonoObject *)p_exc);
	MonoProperty *prop = mono_class_get_property_from_name(klass, "Message");
	MonoString *msg = GDMonoMarshal::mono_string_from_godot(message);
	void *params[1] = { msg };
	property_set_value(prop, (MonoObject *)p_exc, params, NULL);
}

void debug_print_unhandled_exception(MonoException *p_exc) {
	print_unhandled_exception(p_exc);
	debug_send_unhandled_exception_error(p_exc);
}

void debug_send_unhandled_exception_error(MonoException *p_exc) {
#ifdef DEBUG_ENABLED
	if (!ScriptDebugger::get_singleton()) {
#ifdef TOOLS_ENABLED
		if (Engine::get_singleton()->is_editor_hint()) {
			ERR_PRINT(GDMonoUtils::get_exception_name_and_message(p_exc));
		}
#endif
		return;
	}

	_TLS_RECURSION_GUARD_;

	ScriptLanguage::StackInfo separator;
	separator.file = String();
	separator.func = "--- " + RTR("End of inner exception stack trace") + " ---";
	separator.line = 0;

	Vector<ScriptLanguage::StackInfo> si;
	String exc_msg;

	while (p_exc != NULL) {
		GDMonoClass *st_klass = CACHED_CLASS(System_Diagnostics_StackTrace);
		MonoObject *stack_trace = mono_object_new(mono_domain_get(), st_klass->get_mono_ptr());

		MonoBoolean need_file_info = true;
		void *ctor_args[2] = { p_exc, &need_file_info };

		MonoException *unexpected_exc = NULL;
		CACHED_METHOD(System_Diagnostics_StackTrace, ctor_Exception_bool)->invoke_raw(stack_trace, ctor_args, &unexpected_exc);

		if (unexpected_exc) {
			GDMonoInternals::unhandled_exception(unexpected_exc);
			return;
		}

		Vector<ScriptLanguage::StackInfo> _si;
		if (stack_trace != NULL) {
			_si = CSharpLanguage::get_singleton()->stack_trace_get_info(stack_trace);
			for (int i = _si.size() - 1; i >= 0; i--)
				si.insert(0, _si[i]);
		}

		exc_msg += (exc_msg.length() > 0 ? " ---> " : "") + GDMonoUtils::get_exception_name_and_message(p_exc);

		GDMonoClass *exc_class = GDMono::get_singleton()->get_class(mono_get_exception_class());
		GDMonoProperty *inner_exc_prop = exc_class->get_property("InnerException");
		CRASH_COND(inner_exc_prop == NULL);

		MonoObject *inner_exc = inner_exc_prop->get_value((MonoObject *)p_exc);
		if (inner_exc != NULL)
			si.insert(0, separator);

		p_exc = (MonoException *)inner_exc;
	}

	String file = si.size() ? si[0].file : __FILE__;
	String func = si.size() ? si[0].func : FUNCTION_STR;
	int line = si.size() ? si[0].line : __LINE__;
	String error_msg = "Unhandled exception";

	ScriptDebugger::get_singleton()->send_error(func, file, line, error_msg, exc_msg, ERR_HANDLER_ERROR, si);
#endif
}

void debug_unhandled_exception(MonoException *p_exc) {
	GDMonoInternals::unhandled_exception(p_exc); // prints the exception as well
}

void print_unhandled_exception(MonoException *p_exc) {
	mono_print_unhandled_exception((MonoObject *)p_exc);
}

void set_pending_exception(MonoException *p_exc) {
#ifdef NO_PENDING_EXCEPTIONS
	debug_unhandled_exception(p_exc);
#else
	if (get_runtime_invoke_count() == 0) {
		debug_unhandled_exception(p_exc);
		return;
	}

	if (!mono_runtime_set_pending_exception(p_exc, false)) {
		ERR_PRINT("Exception thrown from managed code, but it could not be set as pending:");
		GDMonoUtils::debug_print_unhandled_exception(p_exc);
	}
#endif
}

_THREAD_LOCAL_(int)
current_invoke_count = 0;

MonoObject *runtime_invoke(MonoMethod *p_method, void *p_obj, void **p_params, MonoException **r_exc) {
	GD_MONO_BEGIN_RUNTIME_INVOKE;
	MonoObject *ret = mono_runtime_invoke(p_method, p_obj, p_params, (MonoObject **)r_exc);
	GD_MONO_END_RUNTIME_INVOKE;
	return ret;
}

MonoString *object_to_string(MonoObject *p_obj, MonoException **r_exc) {
	GD_MONO_BEGIN_RUNTIME_INVOKE;
	MonoString *ret = mono_object_to_string(p_obj, (MonoObject **)r_exc);
	GD_MONO_END_RUNTIME_INVOKE;
	return ret;
}

void property_set_value(MonoProperty *p_prop, void *p_obj, void **p_params, MonoException **r_exc) {
	GD_MONO_BEGIN_RUNTIME_INVOKE;
	mono_property_set_value(p_prop, p_obj, p_params, (MonoObject **)r_exc);
	GD_MONO_END_RUNTIME_INVOKE;
}

MonoObject *property_get_value(MonoProperty *p_prop, void *p_obj, void **p_params, MonoException **r_exc) {
	GD_MONO_BEGIN_RUNTIME_INVOKE;
	MonoObject *ret = mono_property_get_value(p_prop, p_obj, p_params, (MonoObject **)r_exc);
	GD_MONO_END_RUNTIME_INVOKE;
	return ret;
}

uint64_t unbox_enum_value(MonoObject *p_boxed, MonoType *p_enum_basetype, bool &r_error) {
	r_error = false;
	switch (mono_type_get_type(p_enum_basetype)) {
		case MONO_TYPE_BOOLEAN:
			return (bool)GDMonoMarshal::unbox<MonoBoolean>(p_boxed) ? 1 : 0;
		case MONO_TYPE_CHAR:
			return GDMonoMarshal::unbox<uint16_t>(p_boxed);
		case MONO_TYPE_U1:
			return GDMonoMarshal::unbox<uint8_t>(p_boxed);
		case MONO_TYPE_U2:
			return GDMonoMarshal::unbox<uint16_t>(p_boxed);
		case MONO_TYPE_U4:
			return GDMonoMarshal::unbox<uint32_t>(p_boxed);
		case MONO_TYPE_U8:
			return GDMonoMarshal::unbox<uint64_t>(p_boxed);
		case MONO_TYPE_I1:
			return GDMonoMarshal::unbox<int8_t>(p_boxed);
		case MONO_TYPE_I2:
			return GDMonoMarshal::unbox<int16_t>(p_boxed);
		case MONO_TYPE_I4:
			return GDMonoMarshal::unbox<int32_t>(p_boxed);
		case MONO_TYPE_I8:
			return GDMonoMarshal::unbox<int64_t>(p_boxed);
		default:
			r_error = true;
			return 0;
	}
}

void dispose(MonoObject *p_mono_object, MonoException **r_exc) {
	CACHED_METHOD_THUNK(GodotObject, Dispose).invoke(p_mono_object, r_exc);
}

namespace Marshal {

#ifdef MONO_GLUE_ENABLED
#ifdef TOOLS_ENABLED
#define NO_GLUE_RET(m_ret)                                     \
	{                                                          \
		if (!GDMonoCache::cached_data.godot_api_cache_updated) \
			return m_ret;                                      \
	}
#else
#define NO_GLUE_RET(m_ret) \
	{}
#endif
#else
#define NO_GLUE_RET(m_ret) \
	{ return m_ret; }
#endif

bool type_is_generic_array(MonoReflectionType *p_reftype) {
	NO_GLUE_RET(false);
	MonoException *exc = NULL;
	MonoBoolean res = CACHED_METHOD_THUNK(MarshalUtils, TypeIsGenericArray).invoke(p_reftype, &exc);
	UNHANDLED_EXCEPTION(exc);
	return (bool)res;
}

bool type_is_generic_dictionary(MonoReflectionType *p_reftype) {
	NO_GLUE_RET(false);
	MonoException *exc = NULL;
	MonoBoolean res = CACHED_METHOD_THUNK(MarshalUtils, TypeIsGenericDictionary).invoke(p_reftype, &exc);
	UNHANDLED_EXCEPTION(exc);
	return (bool)res;
}

bool type_is_system_generic_list(MonoReflectionType *p_reftype) {
	NO_GLUE_RET(false);
	MonoException *exc = NULL;
	MonoBoolean res = CACHED_METHOD_THUNK(MarshalUtils, TypeIsSystemGenericList).invoke(p_reftype, &exc);
	UNHANDLED_EXCEPTION(exc);
	return (bool)res;
}

bool type_is_system_generic_dictionary(MonoReflectionType *p_reftype) {
	NO_GLUE_RET(false);
	MonoException *exc = NULL;
	MonoBoolean res = CACHED_METHOD_THUNK(MarshalUtils, TypeIsSystemGenericDictionary).invoke(p_reftype, &exc);
	UNHANDLED_EXCEPTION(exc);
	return (bool)res;
}

bool type_is_generic_ienumerable(MonoReflectionType *p_reftype) {
	NO_GLUE_RET(false);
	MonoException *exc = NULL;
	MonoBoolean res = CACHED_METHOD_THUNK(MarshalUtils, TypeIsGenericIEnumerable).invoke(p_reftype, &exc);
	UNHANDLED_EXCEPTION(exc);
	return (bool)res;
}

bool type_is_generic_icollection(MonoReflectionType *p_reftype) {
	NO_GLUE_RET(false);
	MonoException *exc = NULL;
	MonoBoolean res = CACHED_METHOD_THUNK(MarshalUtils, TypeIsGenericICollection).invoke(p_reftype, &exc);
	UNHANDLED_EXCEPTION(exc);
	return (bool)res;
}

bool type_is_generic_idictionary(MonoReflectionType *p_reftype) {
	NO_GLUE_RET(false);
	MonoException *exc = NULL;
	MonoBoolean res = CACHED_METHOD_THUNK(MarshalUtils, TypeIsGenericIDictionary).invoke(p_reftype, &exc);
	UNHANDLED_EXCEPTION(exc);
	return (bool)res;
}

void array_get_element_type(MonoReflectionType *p_array_reftype, MonoReflectionType **r_elem_reftype) {
	MonoException *exc = NULL;
	CACHED_METHOD_THUNK(MarshalUtils, ArrayGetElementType).invoke(p_array_reftype, r_elem_reftype, &exc);
	UNHANDLED_EXCEPTION(exc);
}

void dictionary_get_key_value_types(MonoReflectionType *p_dict_reftype, MonoReflectionType **r_key_reftype, MonoReflectionType **r_value_reftype) {
	MonoException *exc = NULL;
	CACHED_METHOD_THUNK(MarshalUtils, DictionaryGetKeyValueTypes).invoke(p_dict_reftype, r_key_reftype, r_value_reftype, &exc);
	UNHANDLED_EXCEPTION(exc);
}

GDMonoClass *make_generic_array_type(MonoReflectionType *p_elem_reftype) {
	NO_GLUE_RET(NULL);
	MonoException *exc = NULL;
	MonoReflectionType *reftype = CACHED_METHOD_THUNK(MarshalUtils, MakeGenericArrayType).invoke(p_elem_reftype, &exc);
	UNHANDLED_EXCEPTION(exc);
	return GDMono::get_singleton()->get_class(mono_class_from_mono_type(mono_reflection_type_get_type(reftype)));
}

GDMonoClass *make_generic_dictionary_type(MonoReflectionType *p_key_reftype, MonoReflectionType *p_value_reftype) {
	NO_GLUE_RET(NULL);
	MonoException *exc = NULL;
	MonoReflectionType *reftype = CACHED_METHOD_THUNK(MarshalUtils, MakeGenericDictionaryType).invoke(p_key_reftype, p_value_reftype, &exc);
	UNHANDLED_EXCEPTION(exc);
	return GDMono::get_singleton()->get_class(mono_class_from_mono_type(mono_reflection_type_get_type(reftype)));
}

} // namespace Marshal

ScopeThreadAttach::ScopeThreadAttach() :
		mono_thread(NULL) {
	if (likely(GDMono::get_singleton()->is_runtime_initialized()) && unlikely(!mono_domain_get())) {
		mono_thread = GDMonoUtils::attach_current_thread();
	}
}

ScopeThreadAttach::~ScopeThreadAttach() {
	if (unlikely(mono_thread)) {
		GDMonoUtils::detach_current_thread(mono_thread);
	}
}

// namespace Marshal

} // namespace GDMonoUtils
