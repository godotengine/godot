/**************************************************************************/
/*  godot.hpp                                                             */
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

#pragma once

#include <gdextension_interface.h>

namespace godot {

namespace internal {

extern "C" GDExtensionInterfaceGetProcAddress gdextension_interface_get_proc_address;
extern "C" GDExtensionClassLibraryPtr library;
extern "C" void *token;

extern "C" GDExtensionGodotVersion2 godot_version;

// All of the GDExtension interface functions.
extern "C" GDExtensionInterfaceGetGodotVersion2 gdextension_interface_get_godot_version2;
extern "C" GDExtensionInterfaceMemAlloc gdextension_interface_mem_alloc;
extern "C" GDExtensionInterfaceMemRealloc gdextension_interface_mem_realloc;
extern "C" GDExtensionInterfaceMemFree gdextension_interface_mem_free;
extern "C" GDExtensionInterfacePrintError gdextension_interface_print_error;
extern "C" GDExtensionInterfacePrintErrorWithMessage gdextension_interface_print_error_with_message;
extern "C" GDExtensionInterfacePrintWarning gdextension_interface_print_warning;
extern "C" GDExtensionInterfacePrintWarningWithMessage gdextension_interface_print_warning_with_message;
extern "C" GDExtensionInterfacePrintScriptError gdextension_interface_print_script_error;
extern "C" GDExtensionInterfacePrintScriptErrorWithMessage gdextension_interface_print_script_error_with_message;
extern "C" GDExtensionInterfaceGetNativeStructSize gdextension_interface_get_native_struct_size;
extern "C" GDExtensionInterfaceVariantNewCopy gdextension_interface_variant_new_copy;
extern "C" GDExtensionInterfaceVariantNewNil gdextension_interface_variant_new_nil;
extern "C" GDExtensionInterfaceVariantDestroy gdextension_interface_variant_destroy;
extern "C" GDExtensionInterfaceVariantCall gdextension_interface_variant_call;
extern "C" GDExtensionInterfaceVariantCallStatic gdextension_interface_variant_call_static;
extern "C" GDExtensionInterfaceVariantEvaluate gdextension_interface_variant_evaluate;
extern "C" GDExtensionInterfaceVariantSet gdextension_interface_variant_set;
extern "C" GDExtensionInterfaceVariantSetNamed gdextension_interface_variant_set_named;
extern "C" GDExtensionInterfaceVariantSetKeyed gdextension_interface_variant_set_keyed;
extern "C" GDExtensionInterfaceVariantSetIndexed gdextension_interface_variant_set_indexed;
extern "C" GDExtensionInterfaceVariantGet gdextension_interface_variant_get;
extern "C" GDExtensionInterfaceVariantGetNamed gdextension_interface_variant_get_named;
extern "C" GDExtensionInterfaceVariantGetKeyed gdextension_interface_variant_get_keyed;
extern "C" GDExtensionInterfaceVariantGetIndexed gdextension_interface_variant_get_indexed;
extern "C" GDExtensionInterfaceVariantIterInit gdextension_interface_variant_iter_init;
extern "C" GDExtensionInterfaceVariantIterNext gdextension_interface_variant_iter_next;
extern "C" GDExtensionInterfaceVariantIterGet gdextension_interface_variant_iter_get;
extern "C" GDExtensionInterfaceVariantHash gdextension_interface_variant_hash;
extern "C" GDExtensionInterfaceVariantRecursiveHash gdextension_interface_variant_recursive_hash;
extern "C" GDExtensionInterfaceVariantHashCompare gdextension_interface_variant_hash_compare;
extern "C" GDExtensionInterfaceVariantBooleanize gdextension_interface_variant_booleanize;
extern "C" GDExtensionInterfaceVariantDuplicate gdextension_interface_variant_duplicate;
extern "C" GDExtensionInterfaceVariantStringify gdextension_interface_variant_stringify;
extern "C" GDExtensionInterfaceVariantGetType gdextension_interface_variant_get_type;
extern "C" GDExtensionInterfaceVariantHasMethod gdextension_interface_variant_has_method;
extern "C" GDExtensionInterfaceVariantHasMember gdextension_interface_variant_has_member;
extern "C" GDExtensionInterfaceVariantHasKey gdextension_interface_variant_has_key;
extern "C" GDExtensionInterfaceVariantGetObjectInstanceId gdextension_interface_variant_get_object_instance_id;
extern "C" GDExtensionInterfaceVariantGetTypeName gdextension_interface_variant_get_type_name;
extern "C" GDExtensionInterfaceVariantCanConvert gdextension_interface_variant_can_convert;
extern "C" GDExtensionInterfaceVariantCanConvertStrict gdextension_interface_variant_can_convert_strict;
extern "C" GDExtensionInterfaceGetVariantFromTypeConstructor gdextension_interface_get_variant_from_type_constructor;
extern "C" GDExtensionInterfaceGetVariantToTypeConstructor gdextension_interface_get_variant_to_type_constructor;
extern "C" GDExtensionInterfaceGetVariantGetInternalPtrFunc gdextension_interface_variant_get_ptr_internal_getter;
extern "C" GDExtensionInterfaceVariantGetPtrOperatorEvaluator gdextension_interface_variant_get_ptr_operator_evaluator;
extern "C" GDExtensionInterfaceVariantGetPtrBuiltinMethod gdextension_interface_variant_get_ptr_builtin_method;
extern "C" GDExtensionInterfaceVariantGetPtrConstructor gdextension_interface_variant_get_ptr_constructor;
extern "C" GDExtensionInterfaceVariantGetPtrDestructor gdextension_interface_variant_get_ptr_destructor;
extern "C" GDExtensionInterfaceVariantConstruct gdextension_interface_variant_construct;
extern "C" GDExtensionInterfaceVariantGetPtrSetter gdextension_interface_variant_get_ptr_setter;
extern "C" GDExtensionInterfaceVariantGetPtrGetter gdextension_interface_variant_get_ptr_getter;
extern "C" GDExtensionInterfaceVariantGetPtrIndexedSetter gdextension_interface_variant_get_ptr_indexed_setter;
extern "C" GDExtensionInterfaceVariantGetPtrIndexedGetter gdextension_interface_variant_get_ptr_indexed_getter;
extern "C" GDExtensionInterfaceVariantGetPtrKeyedSetter gdextension_interface_variant_get_ptr_keyed_setter;
extern "C" GDExtensionInterfaceVariantGetPtrKeyedGetter gdextension_interface_variant_get_ptr_keyed_getter;
extern "C" GDExtensionInterfaceVariantGetPtrKeyedChecker gdextension_interface_variant_get_ptr_keyed_checker;
extern "C" GDExtensionInterfaceVariantGetConstantValue gdextension_interface_variant_get_constant_value;
extern "C" GDExtensionInterfaceVariantGetPtrUtilityFunction gdextension_interface_variant_get_ptr_utility_function;
extern "C" GDExtensionInterfaceStringNewWithLatin1Chars gdextension_interface_string_new_with_latin1_chars;
extern "C" GDExtensionInterfaceStringNewWithUtf8Chars gdextension_interface_string_new_with_utf8_chars;
extern "C" GDExtensionInterfaceStringNewWithUtf16Chars gdextension_interface_string_new_with_utf16_chars;
extern "C" GDExtensionInterfaceStringNewWithUtf32Chars gdextension_interface_string_new_with_utf32_chars;
extern "C" GDExtensionInterfaceStringNewWithWideChars gdextension_interface_string_new_with_wide_chars;
extern "C" GDExtensionInterfaceStringNewWithLatin1CharsAndLen gdextension_interface_string_new_with_latin1_chars_and_len;
extern "C" GDExtensionInterfaceStringNewWithUtf8CharsAndLen gdextension_interface_string_new_with_utf8_chars_and_len;
extern "C" GDExtensionInterfaceStringNewWithUtf8CharsAndLen2 gdextension_interface_string_new_with_utf8_chars_and_len2;
extern "C" GDExtensionInterfaceStringNewWithUtf16CharsAndLen gdextension_interface_string_new_with_utf16_chars_and_len;
extern "C" GDExtensionInterfaceStringNewWithUtf16CharsAndLen2 gdextension_interface_string_new_with_utf16_chars_and_len2;
extern "C" GDExtensionInterfaceStringNewWithUtf32CharsAndLen gdextension_interface_string_new_with_utf32_chars_and_len;
extern "C" GDExtensionInterfaceStringNewWithWideCharsAndLen gdextension_interface_string_new_with_wide_chars_and_len;
extern "C" GDExtensionInterfaceStringToLatin1Chars gdextension_interface_string_to_latin1_chars;
extern "C" GDExtensionInterfaceStringToUtf8Chars gdextension_interface_string_to_utf8_chars;
extern "C" GDExtensionInterfaceStringToUtf16Chars gdextension_interface_string_to_utf16_chars;
extern "C" GDExtensionInterfaceStringToUtf32Chars gdextension_interface_string_to_utf32_chars;
extern "C" GDExtensionInterfaceStringToWideChars gdextension_interface_string_to_wide_chars;
extern "C" GDExtensionInterfaceStringOperatorIndex gdextension_interface_string_operator_index;
extern "C" GDExtensionInterfaceStringOperatorIndexConst gdextension_interface_string_operator_index_const;
extern "C" GDExtensionInterfaceStringOperatorPlusEqString gdextension_interface_string_operator_plus_eq_string;
extern "C" GDExtensionInterfaceStringOperatorPlusEqChar gdextension_interface_string_operator_plus_eq_char;
extern "C" GDExtensionInterfaceStringOperatorPlusEqCstr gdextension_interface_string_operator_plus_eq_cstr;
extern "C" GDExtensionInterfaceStringOperatorPlusEqWcstr gdextension_interface_string_operator_plus_eq_wcstr;
extern "C" GDExtensionInterfaceStringOperatorPlusEqC32str gdextension_interface_string_operator_plus_eq_c32str;
extern "C" GDExtensionInterfaceStringResize gdextension_interface_string_resize;
extern "C" GDExtensionInterfaceStringNameNewWithLatin1Chars gdextension_interface_string_name_new_with_latin1_chars;
extern "C" GDExtensionInterfaceXmlParserOpenBuffer gdextension_interface_xml_parser_open_buffer;
extern "C" GDExtensionInterfaceFileAccessStoreBuffer gdextension_interface_file_access_store_buffer;
extern "C" GDExtensionInterfaceFileAccessGetBuffer gdextension_interface_file_access_get_buffer;
extern "C" GDExtensionInterfaceWorkerThreadPoolAddNativeGroupTask gdextension_interface_worker_thread_pool_add_native_group_task;
extern "C" GDExtensionInterfaceWorkerThreadPoolAddNativeTask gdextension_interface_worker_thread_pool_add_native_task;
extern "C" GDExtensionInterfacePackedByteArrayOperatorIndex gdextension_interface_packed_byte_array_operator_index;
extern "C" GDExtensionInterfacePackedByteArrayOperatorIndexConst gdextension_interface_packed_byte_array_operator_index_const;
extern "C" GDExtensionInterfacePackedColorArrayOperatorIndex gdextension_interface_packed_color_array_operator_index;
extern "C" GDExtensionInterfacePackedColorArrayOperatorIndexConst gdextension_interface_packed_color_array_operator_index_const;
extern "C" GDExtensionInterfacePackedFloat32ArrayOperatorIndex gdextension_interface_packed_float32_array_operator_index;
extern "C" GDExtensionInterfacePackedFloat32ArrayOperatorIndexConst gdextension_interface_packed_float32_array_operator_index_const;
extern "C" GDExtensionInterfacePackedFloat64ArrayOperatorIndex gdextension_interface_packed_float64_array_operator_index;
extern "C" GDExtensionInterfacePackedFloat64ArrayOperatorIndexConst gdextension_interface_packed_float64_array_operator_index_const;
extern "C" GDExtensionInterfacePackedInt32ArrayOperatorIndex gdextension_interface_packed_int32_array_operator_index;
extern "C" GDExtensionInterfacePackedInt32ArrayOperatorIndexConst gdextension_interface_packed_int32_array_operator_index_const;
extern "C" GDExtensionInterfacePackedInt64ArrayOperatorIndex gdextension_interface_packed_int64_array_operator_index;
extern "C" GDExtensionInterfacePackedInt64ArrayOperatorIndexConst gdextension_interface_packed_int64_array_operator_index_const;
extern "C" GDExtensionInterfacePackedStringArrayOperatorIndex gdextension_interface_packed_string_array_operator_index;
extern "C" GDExtensionInterfacePackedStringArrayOperatorIndexConst gdextension_interface_packed_string_array_operator_index_const;
extern "C" GDExtensionInterfacePackedVector2ArrayOperatorIndex gdextension_interface_packed_vector2_array_operator_index;
extern "C" GDExtensionInterfacePackedVector2ArrayOperatorIndexConst gdextension_interface_packed_vector2_array_operator_index_const;
extern "C" GDExtensionInterfacePackedVector3ArrayOperatorIndex gdextension_interface_packed_vector3_array_operator_index;
extern "C" GDExtensionInterfacePackedVector3ArrayOperatorIndexConst gdextension_interface_packed_vector3_array_operator_index_const;
extern "C" GDExtensionInterfacePackedVector4ArrayOperatorIndex gdextension_interface_packed_vector4_array_operator_index;
extern "C" GDExtensionInterfacePackedVector4ArrayOperatorIndexConst gdextension_interface_packed_vector4_array_operator_index_const;
extern "C" GDExtensionInterfaceArrayOperatorIndex gdextension_interface_array_operator_index;
extern "C" GDExtensionInterfaceArrayOperatorIndexConst gdextension_interface_array_operator_index_const;
extern "C" GDExtensionInterfaceArraySetTyped gdextension_interface_array_set_typed;
extern "C" GDExtensionInterfaceDictionaryOperatorIndex gdextension_interface_dictionary_operator_index;
extern "C" GDExtensionInterfaceDictionaryOperatorIndexConst gdextension_interface_dictionary_operator_index_const;
extern "C" GDExtensionInterfaceDictionarySetTyped gdextension_interface_dictionary_set_typed;
extern "C" GDExtensionInterfaceObjectMethodBindCall gdextension_interface_object_method_bind_call;
extern "C" GDExtensionInterfaceObjectMethodBindPtrcall gdextension_interface_object_method_bind_ptrcall;
extern "C" GDExtensionInterfaceObjectDestroy gdextension_interface_object_destroy;
extern "C" GDExtensionInterfaceGlobalGetSingleton gdextension_interface_global_get_singleton;
extern "C" GDExtensionInterfaceObjectGetInstanceBinding gdextension_interface_object_get_instance_binding;
extern "C" GDExtensionInterfaceObjectSetInstanceBinding gdextension_interface_object_set_instance_binding;
extern "C" GDExtensionInterfaceObjectFreeInstanceBinding gdextension_interface_object_free_instance_binding;
extern "C" GDExtensionInterfaceObjectSetInstance gdextension_interface_object_set_instance;
extern "C" GDExtensionInterfaceObjectGetClassName gdextension_interface_object_get_class_name;
extern "C" GDExtensionInterfaceObjectCastTo gdextension_interface_object_cast_to;
extern "C" GDExtensionInterfaceObjectGetInstanceFromId gdextension_interface_object_get_instance_from_id;
extern "C" GDExtensionInterfaceObjectGetInstanceId gdextension_interface_object_get_instance_id;
extern "C" GDExtensionInterfaceObjectHasScriptMethod gdextension_interface_object_has_script_method;
extern "C" GDExtensionInterfaceObjectCallScriptMethod gdextension_interface_object_call_script_method;
extern "C" GDExtensionInterfaceCallableCustomCreate2 gdextension_interface_callable_custom_create2;
extern "C" GDExtensionInterfaceCallableCustomGetUserData gdextension_interface_callable_custom_get_userdata;
extern "C" GDExtensionInterfaceRefGetObject gdextension_interface_ref_get_object;
extern "C" GDExtensionInterfaceRefSetObject gdextension_interface_ref_set_object;
extern "C" GDExtensionInterfaceScriptInstanceCreate3 gdextension_interface_script_instance_create3;
extern "C" GDExtensionInterfacePlaceHolderScriptInstanceCreate gdextension_interface_placeholder_script_instance_create;
extern "C" GDExtensionInterfacePlaceHolderScriptInstanceUpdate gdextension_interface_placeholder_script_instance_update;
extern "C" GDExtensionInterfaceObjectGetScriptInstance gdextension_interface_object_get_script_instance;
extern "C" GDExtensionInterfaceObjectSetScriptInstance gdextension_interface_object_set_script_instance;
extern "C" GDExtensionInterfaceClassdbConstructObject2 gdextension_interface_classdb_construct_object2;
extern "C" GDExtensionInterfaceClassdbGetMethodBind gdextension_interface_classdb_get_method_bind;
extern "C" GDExtensionInterfaceClassdbGetClassTag gdextension_interface_classdb_get_class_tag;
extern "C" GDExtensionInterfaceClassdbRegisterExtensionClass5 gdextension_interface_classdb_register_extension_class5;
extern "C" GDExtensionInterfaceClassdbRegisterExtensionClassMethod gdextension_interface_classdb_register_extension_class_method;
extern "C" GDExtensionInterfaceClassdbRegisterExtensionClassVirtualMethod gdextension_interface_classdb_register_extension_class_virtual_method;
extern "C" GDExtensionInterfaceClassdbRegisterExtensionClassIntegerConstant gdextension_interface_classdb_register_extension_class_integer_constant;
extern "C" GDExtensionInterfaceClassdbRegisterExtensionClassProperty gdextension_interface_classdb_register_extension_class_property;
extern "C" GDExtensionInterfaceClassdbRegisterExtensionClassPropertyIndexed gdextension_interface_classdb_register_extension_class_property_indexed;
extern "C" GDExtensionInterfaceClassdbRegisterExtensionClassPropertyGroup gdextension_interface_classdb_register_extension_class_property_group;
extern "C" GDExtensionInterfaceClassdbRegisterExtensionClassPropertySubgroup gdextension_interface_classdb_register_extension_class_property_subgroup;
extern "C" GDExtensionInterfaceClassdbRegisterExtensionClassSignal gdextension_interface_classdb_register_extension_class_signal;
extern "C" GDExtensionInterfaceClassdbUnregisterExtensionClass gdextension_interface_classdb_unregister_extension_class;
extern "C" GDExtensionInterfaceGetLibraryPath gdextension_interface_get_library_path;
extern "C" GDExtensionInterfaceEditorAddPlugin gdextension_interface_editor_add_plugin;
extern "C" GDExtensionInterfaceEditorRemovePlugin gdextension_interface_editor_remove_plugin;
extern "C" GDExtensionInterfaceEditorRegisterGetClassesUsedCallback gdextension_interface_editor_register_get_classes_used_callback;
extern "C" GDExtensionsInterfaceEditorHelpLoadXmlFromUtf8Chars gdextension_interface_editor_help_load_xml_from_utf8_chars;
extern "C" GDExtensionsInterfaceEditorHelpLoadXmlFromUtf8CharsAndLen gdextension_interface_editor_help_load_xml_from_utf8_chars_and_len;
extern "C" GDExtensionInterfaceImagePtrw gdextension_interface_image_ptrw;
extern "C" GDExtensionInterfaceImagePtr gdextension_interface_image_ptr;
extern "C" GDExtensionInterfaceRegisterMainLoopCallbacks gdextension_interface_register_main_loop_callbacks;

class DocDataRegistration {
public:
	DocDataRegistration(const char *p_hash, int p_uncompressed_size, int p_compressed_size, const unsigned char *p_data);
};

} // namespace internal

enum ModuleInitializationLevel {
	MODULE_INITIALIZATION_LEVEL_CORE = GDEXTENSION_INITIALIZATION_CORE,
	MODULE_INITIALIZATION_LEVEL_SERVERS = GDEXTENSION_INITIALIZATION_SERVERS,
	MODULE_INITIALIZATION_LEVEL_SCENE = GDEXTENSION_INITIALIZATION_SCENE,
	MODULE_INITIALIZATION_LEVEL_EDITOR = GDEXTENSION_INITIALIZATION_EDITOR,
	MODULE_INITIALIZATION_LEVEL_MAX
};

class GDExtensionBinding {
public:
	using Callback = void (*)(ModuleInitializationLevel p_level);

	struct InitData {
		GDExtensionInitializationLevel minimum_initialization_level = GDEXTENSION_INITIALIZATION_CORE;
		Callback init_callback = nullptr;
		Callback terminate_callback = nullptr;
		GDExtensionMainLoopCallbacks main_loop_callbacks = {};

		inline bool has_main_loop_callbacks() const {
			return main_loop_callbacks.frame_func || main_loop_callbacks.startup_func || main_loop_callbacks.shutdown_func;
		}
	};

	class InitDataList {
		int data_count = 0;
		int data_capacity = 0;
		InitData **data = nullptr;

	public:
		void add(InitData *p_cb);
		~InitDataList();
	};

	static bool api_initialized;
	static int level_initialized[MODULE_INITIALIZATION_LEVEL_MAX];
	static InitDataList initdata;
	static GDExtensionBool init(GDExtensionInterfaceGetProcAddress p_get_proc_address, GDExtensionClassLibraryPtr p_library, InitData *p_init_data, GDExtensionInitialization *r_initialization);

public:
	static void initialize_level(void *p_userdata, GDExtensionInitializationLevel p_level);
	static void deinitialize_level(void *p_userdata, GDExtensionInitializationLevel p_level);

	class InitObject {
		GDExtensionInterfaceGetProcAddress get_proc_address;
		GDExtensionClassLibraryPtr library;
		GDExtensionInitialization *initialization;
		mutable InitData *init_data = nullptr;

	public:
		InitObject(GDExtensionInterfaceGetProcAddress p_get_proc_address, GDExtensionClassLibraryPtr p_library, GDExtensionInitialization *r_initialization);

		void register_initializer(Callback p_init) const;
		void register_terminator(Callback p_init) const;
		void set_minimum_library_initialization_level(ModuleInitializationLevel p_level) const;

		// Register a callback that is called after all initialization levels when Godot is fully initialized.
		void register_startup_callback(GDExtensionMainLoopStartupCallback p_callback) const;
		// Register a callback that is called for every process frame. This will run after all `_process()` methods on Node, and before `ScriptServer::frame()`.
		void register_frame_callback(GDExtensionMainLoopFrameCallback p_callback) const;
		// Register a callback that is called before Godot is shutdown when it is still fully initialized.
		void register_shutdown_callback(GDExtensionMainLoopShutdownCallback p_callback) const;

		GDExtensionBool init() const;
	};
};

} // namespace godot
