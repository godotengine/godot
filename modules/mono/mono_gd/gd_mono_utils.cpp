/*************************************************************************/
/*  gd_mono_utils.cpp                                                    */
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
#include "gd_mono_utils.h"

#include "os/dir_access.h"
#include "project_settings.h"
#include "reference.h"

#include "../csharp_script.h"
#include "gd_mono.h"
#include "gd_mono_class.h"
#include "gd_mono_marshal.h"

namespace GDMonoUtils {

MonoCache mono_cache;

#define CACHE_AND_CHECK(m_var, m_val)                                                        \
	{                                                                                        \
		m_var = m_val;                                                                       \
		if (!m_var) ERR_PRINT("Mono Cache: Member " #m_var " is null. This is really bad!"); \
	}

#define CACHE_CLASS_AND_CHECK(m_class, m_val) CACHE_AND_CHECK(GDMonoUtils::mono_cache.class_##m_class, m_val)
#define CACHE_NS_CLASS_AND_CHECK(m_ns, m_class, m_val) CACHE_AND_CHECK(GDMonoUtils::mono_cache.class_##m_ns##_##m_class, m_val)
#define CACHE_RAW_MONO_CLASS_AND_CHECK(m_class, m_val) CACHE_AND_CHECK(GDMonoUtils::mono_cache.rawclass_##m_class, m_val)
#define CACHE_FIELD_AND_CHECK(m_class, m_field, m_val) CACHE_AND_CHECK(GDMonoUtils::mono_cache.field_##m_class##_##m_field, m_val)
#define CACHE_METHOD_THUNK_AND_CHECK(m_class, m_method, m_val) CACHE_AND_CHECK(GDMonoUtils::mono_cache.methodthunk_##m_class##_##m_method, m_val)

void MonoCache::clear_members() {

	class_MonoObject = NULL;
	class_bool = NULL;
	class_int8_t = NULL;
	class_int16_t = NULL;
	class_int32_t = NULL;
	class_int64_t = NULL;
	class_uint8_t = NULL;
	class_uint16_t = NULL;
	class_uint32_t = NULL;
	class_uint64_t = NULL;
	class_float = NULL;
	class_double = NULL;
	class_String = NULL;
	class_IntPtr = NULL;

	rawclass_Dictionary = NULL;

	class_Vector2 = NULL;
	class_Rect2 = NULL;
	class_Transform2D = NULL;
	class_Vector3 = NULL;
	class_Basis = NULL;
	class_Quat = NULL;
	class_Transform = NULL;
	class_AABB = NULL;
	class_Color = NULL;
	class_Plane = NULL;
	class_NodePath = NULL;
	class_RID = NULL;
	class_GodotObject = NULL;
	class_GodotReference = NULL;
	class_Node = NULL;
	class_Control = NULL;
	class_Spatial = NULL;
	class_WeakRef = NULL;
	class_MarshalUtils = NULL;

	class_ExportAttribute = NULL;
	field_ExportAttribute_hint = NULL;
	field_ExportAttribute_hint_string = NULL;
	class_ToolAttribute = NULL;
	class_RemoteAttribute = NULL;
	class_SyncAttribute = NULL;
	class_MasterAttribute = NULL;
	class_SlaveAttribute = NULL;
	class_GodotMethodAttribute = NULL;
	field_GodotMethodAttribute_methodName = NULL;

	field_GodotObject_ptr = NULL;
	field_NodePath_ptr = NULL;
	field_Image_ptr = NULL;
	field_RID_ptr = NULL;

	methodthunk_MarshalUtils_DictionaryToArrays = NULL;
	methodthunk_MarshalUtils_ArraysToDictionary = NULL;
	methodthunk_SignalAwaiter_SignalCallback = NULL;
	methodthunk_SignalAwaiter_FailureCallback = NULL;
	methodthunk_GodotTaskScheduler_Activate = NULL;

	task_scheduler_handle = Ref<MonoGCHandle>();
}

#define GODOT_API_CLASS(m_class) (GDMono::get_singleton()->get_api_assembly()->get_class(BINDINGS_NAMESPACE, #m_class))

void update_corlib_cache() {

	CACHE_CLASS_AND_CHECK(MonoObject, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_object_class()));
	CACHE_CLASS_AND_CHECK(bool, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_boolean_class()));
	CACHE_CLASS_AND_CHECK(int8_t, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_sbyte_class()));
	CACHE_CLASS_AND_CHECK(int16_t, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_int16_class()));
	CACHE_CLASS_AND_CHECK(int32_t, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_int32_class()));
	CACHE_CLASS_AND_CHECK(int64_t, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_int64_class()));
	CACHE_CLASS_AND_CHECK(uint8_t, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_byte_class()));
	CACHE_CLASS_AND_CHECK(uint16_t, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_uint16_class()));
	CACHE_CLASS_AND_CHECK(uint32_t, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_uint32_class()));
	CACHE_CLASS_AND_CHECK(uint64_t, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_uint64_class()));
	CACHE_CLASS_AND_CHECK(float, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_single_class()));
	CACHE_CLASS_AND_CHECK(double, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_double_class()));
	CACHE_CLASS_AND_CHECK(String, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_string_class()));
	CACHE_CLASS_AND_CHECK(IntPtr, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_intptr_class()));
}

void update_godot_api_cache() {

	CACHE_CLASS_AND_CHECK(Vector2, GODOT_API_CLASS(Vector2));
	CACHE_CLASS_AND_CHECK(Rect2, GODOT_API_CLASS(Rect2));
	CACHE_CLASS_AND_CHECK(Transform2D, GODOT_API_CLASS(Transform2D));
	CACHE_CLASS_AND_CHECK(Vector3, GODOT_API_CLASS(Vector3));
	CACHE_CLASS_AND_CHECK(Basis, GODOT_API_CLASS(Basis));
	CACHE_CLASS_AND_CHECK(Quat, GODOT_API_CLASS(Quat));
	CACHE_CLASS_AND_CHECK(Transform, GODOT_API_CLASS(Transform));
	CACHE_CLASS_AND_CHECK(AABB, GODOT_API_CLASS(AABB));
	CACHE_CLASS_AND_CHECK(Color, GODOT_API_CLASS(Color));
	CACHE_CLASS_AND_CHECK(Plane, GODOT_API_CLASS(Plane));
	CACHE_CLASS_AND_CHECK(NodePath, GODOT_API_CLASS(NodePath));
	CACHE_CLASS_AND_CHECK(RID, GODOT_API_CLASS(NodePath));
	CACHE_CLASS_AND_CHECK(GodotObject, GODOT_API_CLASS(Object));
	CACHE_CLASS_AND_CHECK(GodotReference, GODOT_API_CLASS(Reference));
	CACHE_CLASS_AND_CHECK(Node, GODOT_API_CLASS(Node));
	CACHE_CLASS_AND_CHECK(Control, GODOT_API_CLASS(Control));
	CACHE_CLASS_AND_CHECK(Spatial, GODOT_API_CLASS(Spatial));
	CACHE_CLASS_AND_CHECK(WeakRef, GODOT_API_CLASS(WeakRef));
	CACHE_CLASS_AND_CHECK(MarshalUtils, GODOT_API_CLASS(MarshalUtils));

	// Attributes
	CACHE_CLASS_AND_CHECK(ExportAttribute, GODOT_API_CLASS(ExportAttribute));
	CACHE_FIELD_AND_CHECK(ExportAttribute, hint, CACHED_CLASS(ExportAttribute)->get_field("hint"));
	CACHE_FIELD_AND_CHECK(ExportAttribute, hint_string, CACHED_CLASS(ExportAttribute)->get_field("hint_string"));
	CACHE_CLASS_AND_CHECK(ToolAttribute, GODOT_API_CLASS(ToolAttribute));
	CACHE_CLASS_AND_CHECK(RemoteAttribute, GODOT_API_CLASS(RemoteAttribute));
	CACHE_CLASS_AND_CHECK(SyncAttribute, GODOT_API_CLASS(SyncAttribute));
	CACHE_CLASS_AND_CHECK(MasterAttribute, GODOT_API_CLASS(MasterAttribute));
	CACHE_CLASS_AND_CHECK(SlaveAttribute, GODOT_API_CLASS(SlaveAttribute));
	CACHE_CLASS_AND_CHECK(GodotMethodAttribute, GODOT_API_CLASS(GodotMethodAttribute));
	CACHE_FIELD_AND_CHECK(GodotMethodAttribute, methodName, CACHED_CLASS(GodotMethodAttribute)->get_field("methodName"));

	CACHE_FIELD_AND_CHECK(GodotObject, ptr, CACHED_CLASS(GodotObject)->get_field(BINDINGS_PTR_FIELD));
	CACHE_FIELD_AND_CHECK(NodePath, ptr, CACHED_CLASS(NodePath)->get_field(BINDINGS_PTR_FIELD));
	CACHE_FIELD_AND_CHECK(RID, ptr, CACHED_CLASS(RID)->get_field(BINDINGS_PTR_FIELD));

	CACHE_METHOD_THUNK_AND_CHECK(MarshalUtils, DictionaryToArrays, (MarshalUtils_DictToArrays)CACHED_CLASS(MarshalUtils)->get_method("DictionaryToArrays", 3)->get_thunk());
	CACHE_METHOD_THUNK_AND_CHECK(MarshalUtils, ArraysToDictionary, (MarshalUtils_ArraysToDict)CACHED_CLASS(MarshalUtils)->get_method("ArraysToDictionary", 2)->get_thunk());
	CACHE_METHOD_THUNK_AND_CHECK(SignalAwaiter, SignalCallback, (SignalAwaiter_SignalCallback)GODOT_API_CLASS(SignalAwaiter)->get_method("SignalCallback", 1)->get_thunk());
	CACHE_METHOD_THUNK_AND_CHECK(SignalAwaiter, FailureCallback, (SignalAwaiter_FailureCallback)GODOT_API_CLASS(SignalAwaiter)->get_method("FailureCallback", 0)->get_thunk());
	CACHE_METHOD_THUNK_AND_CHECK(GodotTaskScheduler, Activate, (GodotTaskScheduler_Activate)GODOT_API_CLASS(GodotTaskScheduler)->get_method("Activate", 0)->get_thunk());

	{
		/*
		 * TODO Right now we only support Dictionary<object, object>.
		 * It would be great if we could support other key/value types
		 * without forcing the user to copy the entries.
		 */
		GDMonoMethod *method_get_dict_type = CACHED_CLASS(MarshalUtils)->get_method("GetDictionaryType", 0);
		ERR_FAIL_NULL(method_get_dict_type);
		MonoReflectionType *dict_refl_type = (MonoReflectionType *)method_get_dict_type->invoke(NULL);
		ERR_FAIL_NULL(dict_refl_type);
		MonoType *dict_type = mono_reflection_type_get_type(dict_refl_type);
		ERR_FAIL_NULL(dict_type);

		CACHE_RAW_MONO_CLASS_AND_CHECK(Dictionary, mono_class_from_mono_type(dict_type));
	}

	MonoObject *task_scheduler = mono_object_new(SCRIPTS_DOMAIN, GODOT_API_CLASS(GodotTaskScheduler)->get_raw());
	mono_runtime_object_init(task_scheduler);
	mono_cache.task_scheduler_handle = MonoGCHandle::create_strong(task_scheduler);
}

void clear_cache() {
	mono_cache.cleanup();
	mono_cache.clear_members();
}

MonoObject *unmanaged_get_managed(Object *unmanaged) {
	if (unmanaged) {
		if (unmanaged->get_script_instance()) {
			CSharpInstance *cs_instance = CAST_CSHARP_INSTANCE(unmanaged->get_script_instance());

			if (cs_instance) {
				return cs_instance->get_mono_object();
			}
		}

		// Only called if the owner does not have a CSharpInstance
		void *data = unmanaged->get_script_instance_binding(CSharpLanguage::get_singleton()->get_language_index());

		if (data) {
			return ((Map<Object *, Ref<MonoGCHandle> >::Element *)data)->value()->get_target();
		}
	}

	return NULL;
}

void set_main_thread(MonoThread *p_thread) {
	mono_thread_set_main(p_thread);
}

void attach_current_thread() {
	ERR_FAIL_COND(!GDMono::get_singleton()->is_runtime_initialized());
	MonoThread *mono_thread = mono_thread_attach(SCRIPTS_DOMAIN);
	ERR_FAIL_NULL(mono_thread);
}

void detach_current_thread() {
	ERR_FAIL_COND(!GDMono::get_singleton()->is_runtime_initialized());
	MonoThread *mono_thread = mono_thread_current();
	ERR_FAIL_NULL(mono_thread);
	mono_thread_detach(mono_thread);
}

MonoThread *get_current_thread() {
	return mono_thread_current();
}

GDMonoClass *get_object_class(MonoObject *p_object) {
	return GDMono::get_singleton()->get_class(mono_object_get_class(p_object));
}

GDMonoClass *type_get_proxy_class(const StringName &p_type) {
	String class_name = p_type;

	if (class_name[0] == '_')
		class_name = class_name.substr(1, class_name.length());

	GDMonoClass *klass = GDMono::get_singleton()->get_api_assembly()->get_class(BINDINGS_NAMESPACE, class_name);

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
		if (assembly == GDMono::get_singleton()->get_api_assembly())
			return klass;
#ifdef TOOLS_ENABLED
		if (assembly == GDMono::get_singleton()->get_editor_api_assembly())
			return klass;
#endif
	} while ((klass = klass->get_parent_class()) != NULL);

	return NULL;
}

MonoObject *create_managed_for_godot_object(GDMonoClass *p_class, const StringName &p_native, Object *p_object) {
	String object_type = p_object->get_class_name();

	if (object_type[0] == '_')
		object_type = object_type.substr(1, object_type.length());

	if (!ClassDB::is_parent_class(object_type, p_native)) {
		ERR_EXPLAIN("Type inherits from native type '" + p_native + "', so it can't be instanced in object of type: '" + p_object->get_class() + "'");
		ERR_FAIL_V(NULL);
	}

	MonoObject *mono_object = mono_object_new(SCRIPTS_DOMAIN, p_class->get_raw());
	ERR_FAIL_NULL_V(mono_object, NULL);

	CACHED_FIELD(GodotObject, ptr)->set_value_raw(mono_object, p_object);

	// Construct
	mono_runtime_object_init(mono_object);

	return mono_object;
}

MonoObject *create_managed_from(const NodePath &p_from) {
	MonoObject *mono_object = mono_object_new(SCRIPTS_DOMAIN, CACHED_CLASS_RAW(NodePath));
	ERR_FAIL_NULL_V(mono_object, NULL);

	// Construct
	mono_runtime_object_init(mono_object);

	CACHED_FIELD(NodePath, ptr)->set_value_raw(mono_object, memnew(NodePath(p_from)));

	return mono_object;
}

MonoObject *create_managed_from(const RID &p_from) {
	MonoObject *mono_object = mono_object_new(SCRIPTS_DOMAIN, CACHED_CLASS_RAW(RID));
	ERR_FAIL_NULL_V(mono_object, NULL);

	// Construct
	mono_runtime_object_init(mono_object);

	CACHED_FIELD(RID, ptr)->set_value_raw(mono_object, memnew(RID(p_from)));

	return mono_object;
}

MonoDomain *create_domain(const String &p_friendly_name) {
	MonoDomain *domain = mono_domain_create_appdomain((char *)p_friendly_name.utf8().get_data(), NULL);

	if (domain) {
		// Workaround to avoid this exception:
		// System.Configuration.ConfigurationErrorsException: Error Initializing the configuration system.
		// ---> System.ArgumentException: The 'ExeConfigFilename' argument cannot be null.
		mono_domain_set_config(domain, ".", "");
	}

	return domain;
}

String get_exception_name_and_message(MonoObject *p_ex) {
	String res;

	MonoClass *klass = mono_object_get_class(p_ex);
	MonoType *type = mono_class_get_type(klass);

	char *full_name = mono_type_full_name(type);
	res += full_name;
	mono_free(full_name);

	res += ": ";

	MonoProperty *prop = mono_class_get_property_from_name(klass, "Message");
	MonoString *msg = (MonoString *)mono_property_get_value(prop, p_ex, NULL, NULL);
	res += GDMonoMarshal::mono_string_to_godot(msg);

	return res;
}
} // namespace GDMonoUtils
