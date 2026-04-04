/**************************************************************************/
/*  gdextension_interface_loader.cpp                                      */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#include <godot_cpp/core/error_macros.hpp>
#include <godot_cpp/core/gdextension_interface_loader.hpp>
#include <godot_cpp/core/load_proc_address.inc>

namespace godot {

namespace gdextension_interface {

// Godot 4.1 or newer.
#if GODOT_VERSION_MINOR >= 1
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 5
GDExtensionInterfaceGetGodotVersion get_godot_version = nullptr;
#endif
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 6
GDExtensionInterfaceMemAlloc mem_alloc = nullptr;
#endif
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 6
GDExtensionInterfaceMemRealloc mem_realloc = nullptr;
#endif
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 6
GDExtensionInterfaceMemFree mem_free = nullptr;
#endif
GDExtensionInterfacePrintError print_error = nullptr;
GDExtensionInterfacePrintErrorWithMessage print_error_with_message = nullptr;
GDExtensionInterfacePrintWarning print_warning = nullptr;
GDExtensionInterfacePrintWarningWithMessage print_warning_with_message = nullptr;
GDExtensionInterfacePrintScriptError print_script_error = nullptr;
GDExtensionInterfacePrintScriptErrorWithMessage print_script_error_with_message = nullptr;
GDExtensionInterfaceGetNativeStructSize get_native_struct_size = nullptr;
GDExtensionInterfaceVariantNewCopy variant_new_copy = nullptr;
GDExtensionInterfaceVariantNewNil variant_new_nil = nullptr;
GDExtensionInterfaceVariantDestroy variant_destroy = nullptr;
GDExtensionInterfaceVariantCall variant_call = nullptr;
GDExtensionInterfaceVariantCallStatic variant_call_static = nullptr;
GDExtensionInterfaceVariantEvaluate variant_evaluate = nullptr;
GDExtensionInterfaceVariantSet variant_set = nullptr;
GDExtensionInterfaceVariantSetNamed variant_set_named = nullptr;
GDExtensionInterfaceVariantSetKeyed variant_set_keyed = nullptr;
GDExtensionInterfaceVariantSetIndexed variant_set_indexed = nullptr;
GDExtensionInterfaceVariantGet variant_get = nullptr;
GDExtensionInterfaceVariantGetNamed variant_get_named = nullptr;
GDExtensionInterfaceVariantGetKeyed variant_get_keyed = nullptr;
GDExtensionInterfaceVariantGetIndexed variant_get_indexed = nullptr;
GDExtensionInterfaceVariantIterInit variant_iter_init = nullptr;
GDExtensionInterfaceVariantIterNext variant_iter_next = nullptr;
GDExtensionInterfaceVariantIterGet variant_iter_get = nullptr;
GDExtensionInterfaceVariantHash variant_hash = nullptr;
GDExtensionInterfaceVariantRecursiveHash variant_recursive_hash = nullptr;
GDExtensionInterfaceVariantHashCompare variant_hash_compare = nullptr;
GDExtensionInterfaceVariantBooleanize variant_booleanize = nullptr;
GDExtensionInterfaceVariantDuplicate variant_duplicate = nullptr;
GDExtensionInterfaceVariantStringify variant_stringify = nullptr;
GDExtensionInterfaceVariantGetType variant_get_type = nullptr;
GDExtensionInterfaceVariantHasMethod variant_has_method = nullptr;
GDExtensionInterfaceVariantHasMember variant_has_member = nullptr;
GDExtensionInterfaceVariantHasKey variant_has_key = nullptr;
GDExtensionInterfaceVariantGetTypeName variant_get_type_name = nullptr;
GDExtensionInterfaceVariantCanConvert variant_can_convert = nullptr;
GDExtensionInterfaceVariantCanConvertStrict variant_can_convert_strict = nullptr;
GDExtensionInterfaceGetVariantFromTypeConstructor get_variant_from_type_constructor = nullptr;
GDExtensionInterfaceGetVariantToTypeConstructor get_variant_to_type_constructor = nullptr;
GDExtensionInterfaceVariantGetPtrOperatorEvaluator variant_get_ptr_operator_evaluator = nullptr;
GDExtensionInterfaceVariantGetPtrBuiltinMethod variant_get_ptr_builtin_method = nullptr;
GDExtensionInterfaceVariantGetPtrConstructor variant_get_ptr_constructor = nullptr;
GDExtensionInterfaceVariantGetPtrDestructor variant_get_ptr_destructor = nullptr;
GDExtensionInterfaceVariantConstruct variant_construct = nullptr;
GDExtensionInterfaceVariantGetPtrSetter variant_get_ptr_setter = nullptr;
GDExtensionInterfaceVariantGetPtrGetter variant_get_ptr_getter = nullptr;
GDExtensionInterfaceVariantGetPtrIndexedSetter variant_get_ptr_indexed_setter = nullptr;
GDExtensionInterfaceVariantGetPtrIndexedGetter variant_get_ptr_indexed_getter = nullptr;
GDExtensionInterfaceVariantGetPtrKeyedSetter variant_get_ptr_keyed_setter = nullptr;
GDExtensionInterfaceVariantGetPtrKeyedGetter variant_get_ptr_keyed_getter = nullptr;
GDExtensionInterfaceVariantGetPtrKeyedChecker variant_get_ptr_keyed_checker = nullptr;
GDExtensionInterfaceVariantGetConstantValue variant_get_constant_value = nullptr;
GDExtensionInterfaceVariantGetPtrUtilityFunction variant_get_ptr_utility_function = nullptr;
GDExtensionInterfaceStringNewWithLatin1Chars string_new_with_latin1_chars = nullptr;
GDExtensionInterfaceStringNewWithUtf8Chars string_new_with_utf8_chars = nullptr;
GDExtensionInterfaceStringNewWithUtf16Chars string_new_with_utf16_chars = nullptr;
GDExtensionInterfaceStringNewWithUtf32Chars string_new_with_utf32_chars = nullptr;
GDExtensionInterfaceStringNewWithWideChars string_new_with_wide_chars = nullptr;
GDExtensionInterfaceStringNewWithLatin1CharsAndLen string_new_with_latin1_chars_and_len = nullptr;
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 3
GDExtensionInterfaceStringNewWithUtf8CharsAndLen string_new_with_utf8_chars_and_len = nullptr;
#endif
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 3
GDExtensionInterfaceStringNewWithUtf16CharsAndLen string_new_with_utf16_chars_and_len = nullptr;
#endif
GDExtensionInterfaceStringNewWithUtf32CharsAndLen string_new_with_utf32_chars_and_len = nullptr;
GDExtensionInterfaceStringNewWithWideCharsAndLen string_new_with_wide_chars_and_len = nullptr;
GDExtensionInterfaceStringToLatin1Chars string_to_latin1_chars = nullptr;
GDExtensionInterfaceStringToUtf8Chars string_to_utf8_chars = nullptr;
GDExtensionInterfaceStringToUtf16Chars string_to_utf16_chars = nullptr;
GDExtensionInterfaceStringToUtf32Chars string_to_utf32_chars = nullptr;
GDExtensionInterfaceStringToWideChars string_to_wide_chars = nullptr;
GDExtensionInterfaceStringOperatorIndex string_operator_index = nullptr;
GDExtensionInterfaceStringOperatorIndexConst string_operator_index_const = nullptr;
GDExtensionInterfaceStringOperatorPlusEqString string_operator_plus_eq_string = nullptr;
GDExtensionInterfaceStringOperatorPlusEqChar string_operator_plus_eq_char = nullptr;
GDExtensionInterfaceStringOperatorPlusEqCstr string_operator_plus_eq_cstr = nullptr;
GDExtensionInterfaceStringOperatorPlusEqWcstr string_operator_plus_eq_wcstr = nullptr;
GDExtensionInterfaceStringOperatorPlusEqC32str string_operator_plus_eq_c32str = nullptr;
GDExtensionInterfaceXmlParserOpenBuffer xml_parser_open_buffer = nullptr;
GDExtensionInterfaceFileAccessStoreBuffer file_access_store_buffer = nullptr;
GDExtensionInterfaceFileAccessGetBuffer file_access_get_buffer = nullptr;
GDExtensionInterfaceWorkerThreadPoolAddNativeGroupTask worker_thread_pool_add_native_group_task = nullptr;
GDExtensionInterfaceWorkerThreadPoolAddNativeTask worker_thread_pool_add_native_task = nullptr;
GDExtensionInterfacePackedByteArrayOperatorIndex packed_byte_array_operator_index = nullptr;
GDExtensionInterfacePackedByteArrayOperatorIndexConst packed_byte_array_operator_index_const = nullptr;
GDExtensionInterfacePackedFloat32ArrayOperatorIndex packed_float32_array_operator_index = nullptr;
GDExtensionInterfacePackedFloat32ArrayOperatorIndexConst packed_float32_array_operator_index_const = nullptr;
GDExtensionInterfacePackedFloat64ArrayOperatorIndex packed_float64_array_operator_index = nullptr;
GDExtensionInterfacePackedFloat64ArrayOperatorIndexConst packed_float64_array_operator_index_const = nullptr;
GDExtensionInterfacePackedInt32ArrayOperatorIndex packed_int32_array_operator_index = nullptr;
GDExtensionInterfacePackedInt32ArrayOperatorIndexConst packed_int32_array_operator_index_const = nullptr;
GDExtensionInterfacePackedInt64ArrayOperatorIndex packed_int64_array_operator_index = nullptr;
GDExtensionInterfacePackedInt64ArrayOperatorIndexConst packed_int64_array_operator_index_const = nullptr;
GDExtensionInterfacePackedStringArrayOperatorIndex packed_string_array_operator_index = nullptr;
GDExtensionInterfacePackedStringArrayOperatorIndexConst packed_string_array_operator_index_const = nullptr;
GDExtensionInterfacePackedVector2ArrayOperatorIndex packed_vector2_array_operator_index = nullptr;
GDExtensionInterfacePackedVector2ArrayOperatorIndexConst packed_vector2_array_operator_index_const = nullptr;
GDExtensionInterfacePackedVector3ArrayOperatorIndex packed_vector3_array_operator_index = nullptr;
GDExtensionInterfacePackedVector3ArrayOperatorIndexConst packed_vector3_array_operator_index_const = nullptr;
GDExtensionInterfacePackedColorArrayOperatorIndex packed_color_array_operator_index = nullptr;
GDExtensionInterfacePackedColorArrayOperatorIndexConst packed_color_array_operator_index_const = nullptr;
GDExtensionInterfaceArrayOperatorIndex array_operator_index = nullptr;
GDExtensionInterfaceArrayOperatorIndexConst array_operator_index_const = nullptr;
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 5
GDExtensionInterfaceArrayRef array_ref = nullptr;
#endif
GDExtensionInterfaceArraySetTyped array_set_typed = nullptr;
GDExtensionInterfaceDictionaryOperatorIndex dictionary_operator_index = nullptr;
GDExtensionInterfaceDictionaryOperatorIndexConst dictionary_operator_index_const = nullptr;
GDExtensionInterfaceObjectMethodBindCall object_method_bind_call = nullptr;
GDExtensionInterfaceObjectMethodBindPtrcall object_method_bind_ptrcall = nullptr;
GDExtensionInterfaceObjectDestroy object_destroy = nullptr;
GDExtensionInterfaceGlobalGetSingleton global_get_singleton = nullptr;
GDExtensionInterfaceObjectGetInstanceBinding object_get_instance_binding = nullptr;
GDExtensionInterfaceObjectSetInstanceBinding object_set_instance_binding = nullptr;
GDExtensionInterfaceObjectSetInstance object_set_instance = nullptr;
GDExtensionInterfaceObjectGetClassName object_get_class_name = nullptr;
GDExtensionInterfaceObjectCastTo object_cast_to = nullptr;
GDExtensionInterfaceObjectGetInstanceFromId object_get_instance_from_id = nullptr;
GDExtensionInterfaceObjectGetInstanceId object_get_instance_id = nullptr;
GDExtensionInterfaceRefGetObject ref_get_object = nullptr;
GDExtensionInterfaceRefSetObject ref_set_object = nullptr;
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 2
GDExtensionInterfaceScriptInstanceCreate script_instance_create = nullptr;
#endif
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 4
GDExtensionInterfaceClassdbConstructObject classdb_construct_object = nullptr;
#endif
GDExtensionInterfaceClassdbGetMethodBind classdb_get_method_bind = nullptr;
GDExtensionInterfaceClassdbGetClassTag classdb_get_class_tag = nullptr;
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 2
GDExtensionInterfaceClassdbRegisterExtensionClass classdb_register_extension_class = nullptr;
#endif
GDExtensionInterfaceClassdbRegisterExtensionClassMethod classdb_register_extension_class_method = nullptr;
GDExtensionInterfaceClassdbRegisterExtensionClassIntegerConstant classdb_register_extension_class_integer_constant = nullptr;
GDExtensionInterfaceClassdbRegisterExtensionClassProperty classdb_register_extension_class_property = nullptr;
GDExtensionInterfaceClassdbRegisterExtensionClassPropertyGroup classdb_register_extension_class_property_group = nullptr;
GDExtensionInterfaceClassdbRegisterExtensionClassPropertySubgroup classdb_register_extension_class_property_subgroup = nullptr;
GDExtensionInterfaceClassdbRegisterExtensionClassSignal classdb_register_extension_class_signal = nullptr;
GDExtensionInterfaceClassdbUnregisterExtensionClass classdb_unregister_extension_class = nullptr;
GDExtensionInterfaceGetLibraryPath get_library_path = nullptr;
GDExtensionInterfaceEditorAddPlugin editor_add_plugin = nullptr;
GDExtensionInterfaceEditorRemovePlugin editor_remove_plugin = nullptr;
#endif // GODOT_VERSION_MINOR >= 1

// Godot 4.2 or newer.
#if GODOT_VERSION_MINOR >= 2
GDExtensionInterfaceStringResize string_resize = nullptr;
GDExtensionInterfaceStringNameNewWithLatin1Chars string_name_new_with_latin1_chars = nullptr;
GDExtensionInterfaceStringNameNewWithUtf8Chars string_name_new_with_utf8_chars = nullptr;
GDExtensionInterfaceStringNameNewWithUtf8CharsAndLen string_name_new_with_utf8_chars_and_len = nullptr;
GDExtensionInterfaceObjectFreeInstanceBinding object_free_instance_binding = nullptr;
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 3
GDExtensionInterfaceScriptInstanceCreate2 script_instance_create2 = nullptr;
#endif
GDExtensionInterfacePlaceholderScriptInstanceCreate placeholder_script_instance_create = nullptr;
GDExtensionInterfacePlaceholderScriptInstanceUpdate placeholder_script_instance_update = nullptr;
GDExtensionInterfaceObjectGetScriptInstance object_get_script_instance = nullptr;
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 3
GDExtensionInterfaceCallableCustomCreate callable_custom_create = nullptr;
#endif
GDExtensionInterfaceCallableCustomGetUserdata callable_custom_get_userdata = nullptr;
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 3
GDExtensionInterfaceClassdbRegisterExtensionClass2 classdb_register_extension_class2 = nullptr;
#endif
GDExtensionInterfaceClassdbRegisterExtensionClassPropertyIndexed classdb_register_extension_class_property_indexed = nullptr;
#endif // GODOT_VERSION_MINOR >= 2

// Godot 4.3 or newer.
#if GODOT_VERSION_MINOR >= 3
GDExtensionInterfaceStringNewWithUtf8CharsAndLen2 string_new_with_utf8_chars_and_len2 = nullptr;
GDExtensionInterfaceStringNewWithUtf16CharsAndLen2 string_new_with_utf16_chars_and_len2 = nullptr;
GDExtensionInterfaceImagePtrw image_ptrw = nullptr;
GDExtensionInterfaceImagePtr image_ptr = nullptr;
GDExtensionInterfacePackedVector4ArrayOperatorIndex packed_vector4_array_operator_index = nullptr;
GDExtensionInterfacePackedVector4ArrayOperatorIndexConst packed_vector4_array_operator_index_const = nullptr;
GDExtensionInterfaceObjectHasScriptMethod object_has_script_method = nullptr;
GDExtensionInterfaceObjectCallScriptMethod object_call_script_method = nullptr;
GDExtensionInterfaceScriptInstanceCreate3 script_instance_create3 = nullptr;
GDExtensionInterfaceCallableCustomCreate2 callable_custom_create2 = nullptr;
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 4
GDExtensionInterfaceClassdbRegisterExtensionClass3 classdb_register_extension_class3 = nullptr;
#endif
GDExtensionInterfaceClassdbRegisterExtensionClassVirtualMethod classdb_register_extension_class_virtual_method = nullptr;
GDExtensionInterfaceEditorHelpLoadXmlFromUtf8Chars editor_help_load_xml_from_utf8_chars = nullptr;
GDExtensionInterfaceEditorHelpLoadXmlFromUtf8CharsAndLen editor_help_load_xml_from_utf8_chars_and_len = nullptr;
#endif // GODOT_VERSION_MINOR >= 3

// Godot 4.4 or newer.
#if GODOT_VERSION_MINOR >= 4
GDExtensionInterfaceVariantGetObjectInstanceId variant_get_object_instance_id = nullptr;
GDExtensionInterfaceVariantGetPtrInternalGetter variant_get_ptr_internal_getter = nullptr;
GDExtensionInterfaceDictionarySetTyped dictionary_set_typed = nullptr;
GDExtensionInterfaceClassdbConstructObject2 classdb_construct_object2 = nullptr;
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 5
GDExtensionInterfaceClassdbRegisterExtensionClass4 classdb_register_extension_class4 = nullptr;
#endif
#endif // GODOT_VERSION_MINOR >= 4

// Godot 4.5 or newer.
#if GODOT_VERSION_MINOR >= 5
GDExtensionInterfaceGetGodotVersion2 get_godot_version2 = nullptr;
GDExtensionInterfaceObjectSetScriptInstance object_set_script_instance = nullptr;
GDExtensionInterfaceClassdbRegisterExtensionClass5 classdb_register_extension_class5 = nullptr;
GDExtensionInterfaceEditorRegisterGetClassesUsedCallback editor_register_get_classes_used_callback = nullptr;
GDExtensionInterfaceRegisterMainLoopCallbacks register_main_loop_callbacks = nullptr;
#endif // GODOT_VERSION_MINOR >= 5

// Godot 4.6 or newer.
#if GODOT_VERSION_MINOR >= 6
GDExtensionInterfaceMemAlloc2 mem_alloc2 = nullptr;
GDExtensionInterfaceMemRealloc2 mem_realloc2 = nullptr;
GDExtensionInterfaceMemFree2 mem_free2 = nullptr;
#endif // GODOT_VERSION_MINOR >= 6

} // namespace gdextension_interface

namespace internal {

bool load_gdextension_interface(GDExtensionInterfaceGetProcAddress p_get_proc_address) {
	// Godot 4.1 or newer.
#if GODOT_VERSION_MINOR >= 1
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 5
	LOAD_PROC_ADDRESS(get_godot_version, GDExtensionInterfaceGetGodotVersion);
#endif
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 6
	LOAD_PROC_ADDRESS(mem_alloc, GDExtensionInterfaceMemAlloc);
#endif
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 6
	LOAD_PROC_ADDRESS(mem_realloc, GDExtensionInterfaceMemRealloc);
#endif
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 6
	LOAD_PROC_ADDRESS(mem_free, GDExtensionInterfaceMemFree);
#endif
	LOAD_PROC_ADDRESS(print_error_with_message, GDExtensionInterfacePrintErrorWithMessage);
	LOAD_PROC_ADDRESS(print_warning, GDExtensionInterfacePrintWarning);
	LOAD_PROC_ADDRESS(print_warning_with_message, GDExtensionInterfacePrintWarningWithMessage);
	LOAD_PROC_ADDRESS(print_script_error, GDExtensionInterfacePrintScriptError);
	LOAD_PROC_ADDRESS(print_script_error_with_message, GDExtensionInterfacePrintScriptErrorWithMessage);
	LOAD_PROC_ADDRESS(get_native_struct_size, GDExtensionInterfaceGetNativeStructSize);
	LOAD_PROC_ADDRESS(variant_new_copy, GDExtensionInterfaceVariantNewCopy);
	LOAD_PROC_ADDRESS(variant_new_nil, GDExtensionInterfaceVariantNewNil);
	LOAD_PROC_ADDRESS(variant_destroy, GDExtensionInterfaceVariantDestroy);
	LOAD_PROC_ADDRESS(variant_call, GDExtensionInterfaceVariantCall);
	LOAD_PROC_ADDRESS(variant_call_static, GDExtensionInterfaceVariantCallStatic);
	LOAD_PROC_ADDRESS(variant_evaluate, GDExtensionInterfaceVariantEvaluate);
	LOAD_PROC_ADDRESS(variant_set, GDExtensionInterfaceVariantSet);
	LOAD_PROC_ADDRESS(variant_set_named, GDExtensionInterfaceVariantSetNamed);
	LOAD_PROC_ADDRESS(variant_set_keyed, GDExtensionInterfaceVariantSetKeyed);
	LOAD_PROC_ADDRESS(variant_set_indexed, GDExtensionInterfaceVariantSetIndexed);
	LOAD_PROC_ADDRESS(variant_get, GDExtensionInterfaceVariantGet);
	LOAD_PROC_ADDRESS(variant_get_named, GDExtensionInterfaceVariantGetNamed);
	LOAD_PROC_ADDRESS(variant_get_keyed, GDExtensionInterfaceVariantGetKeyed);
	LOAD_PROC_ADDRESS(variant_get_indexed, GDExtensionInterfaceVariantGetIndexed);
	LOAD_PROC_ADDRESS(variant_iter_init, GDExtensionInterfaceVariantIterInit);
	LOAD_PROC_ADDRESS(variant_iter_next, GDExtensionInterfaceVariantIterNext);
	LOAD_PROC_ADDRESS(variant_iter_get, GDExtensionInterfaceVariantIterGet);
	LOAD_PROC_ADDRESS(variant_hash, GDExtensionInterfaceVariantHash);
	LOAD_PROC_ADDRESS(variant_recursive_hash, GDExtensionInterfaceVariantRecursiveHash);
	LOAD_PROC_ADDRESS(variant_hash_compare, GDExtensionInterfaceVariantHashCompare);
	LOAD_PROC_ADDRESS(variant_booleanize, GDExtensionInterfaceVariantBooleanize);
	LOAD_PROC_ADDRESS(variant_duplicate, GDExtensionInterfaceVariantDuplicate);
	LOAD_PROC_ADDRESS(variant_stringify, GDExtensionInterfaceVariantStringify);
	LOAD_PROC_ADDRESS(variant_get_type, GDExtensionInterfaceVariantGetType);
	LOAD_PROC_ADDRESS(variant_has_method, GDExtensionInterfaceVariantHasMethod);
	LOAD_PROC_ADDRESS(variant_has_member, GDExtensionInterfaceVariantHasMember);
	LOAD_PROC_ADDRESS(variant_has_key, GDExtensionInterfaceVariantHasKey);
	LOAD_PROC_ADDRESS(variant_get_type_name, GDExtensionInterfaceVariantGetTypeName);
	LOAD_PROC_ADDRESS(variant_can_convert, GDExtensionInterfaceVariantCanConvert);
	LOAD_PROC_ADDRESS(variant_can_convert_strict, GDExtensionInterfaceVariantCanConvertStrict);
	LOAD_PROC_ADDRESS(get_variant_from_type_constructor, GDExtensionInterfaceGetVariantFromTypeConstructor);
	LOAD_PROC_ADDRESS(get_variant_to_type_constructor, GDExtensionInterfaceGetVariantToTypeConstructor);
	LOAD_PROC_ADDRESS(variant_get_ptr_operator_evaluator, GDExtensionInterfaceVariantGetPtrOperatorEvaluator);
	LOAD_PROC_ADDRESS(variant_get_ptr_builtin_method, GDExtensionInterfaceVariantGetPtrBuiltinMethod);
	LOAD_PROC_ADDRESS(variant_get_ptr_constructor, GDExtensionInterfaceVariantGetPtrConstructor);
	LOAD_PROC_ADDRESS(variant_get_ptr_destructor, GDExtensionInterfaceVariantGetPtrDestructor);
	LOAD_PROC_ADDRESS(variant_construct, GDExtensionInterfaceVariantConstruct);
	LOAD_PROC_ADDRESS(variant_get_ptr_setter, GDExtensionInterfaceVariantGetPtrSetter);
	LOAD_PROC_ADDRESS(variant_get_ptr_getter, GDExtensionInterfaceVariantGetPtrGetter);
	LOAD_PROC_ADDRESS(variant_get_ptr_indexed_setter, GDExtensionInterfaceVariantGetPtrIndexedSetter);
	LOAD_PROC_ADDRESS(variant_get_ptr_indexed_getter, GDExtensionInterfaceVariantGetPtrIndexedGetter);
	LOAD_PROC_ADDRESS(variant_get_ptr_keyed_setter, GDExtensionInterfaceVariantGetPtrKeyedSetter);
	LOAD_PROC_ADDRESS(variant_get_ptr_keyed_getter, GDExtensionInterfaceVariantGetPtrKeyedGetter);
	LOAD_PROC_ADDRESS(variant_get_ptr_keyed_checker, GDExtensionInterfaceVariantGetPtrKeyedChecker);
	LOAD_PROC_ADDRESS(variant_get_constant_value, GDExtensionInterfaceVariantGetConstantValue);
	LOAD_PROC_ADDRESS(variant_get_ptr_utility_function, GDExtensionInterfaceVariantGetPtrUtilityFunction);
	LOAD_PROC_ADDRESS(string_new_with_latin1_chars, GDExtensionInterfaceStringNewWithLatin1Chars);
	LOAD_PROC_ADDRESS(string_new_with_utf8_chars, GDExtensionInterfaceStringNewWithUtf8Chars);
	LOAD_PROC_ADDRESS(string_new_with_utf16_chars, GDExtensionInterfaceStringNewWithUtf16Chars);
	LOAD_PROC_ADDRESS(string_new_with_utf32_chars, GDExtensionInterfaceStringNewWithUtf32Chars);
	LOAD_PROC_ADDRESS(string_new_with_wide_chars, GDExtensionInterfaceStringNewWithWideChars);
	LOAD_PROC_ADDRESS(string_new_with_latin1_chars_and_len, GDExtensionInterfaceStringNewWithLatin1CharsAndLen);
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 3
	LOAD_PROC_ADDRESS(string_new_with_utf8_chars_and_len, GDExtensionInterfaceStringNewWithUtf8CharsAndLen);
#endif
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 3
	LOAD_PROC_ADDRESS(string_new_with_utf16_chars_and_len, GDExtensionInterfaceStringNewWithUtf16CharsAndLen);
#endif
	LOAD_PROC_ADDRESS(string_new_with_utf32_chars_and_len, GDExtensionInterfaceStringNewWithUtf32CharsAndLen);
	LOAD_PROC_ADDRESS(string_new_with_wide_chars_and_len, GDExtensionInterfaceStringNewWithWideCharsAndLen);
	LOAD_PROC_ADDRESS(string_to_latin1_chars, GDExtensionInterfaceStringToLatin1Chars);
	LOAD_PROC_ADDRESS(string_to_utf8_chars, GDExtensionInterfaceStringToUtf8Chars);
	LOAD_PROC_ADDRESS(string_to_utf16_chars, GDExtensionInterfaceStringToUtf16Chars);
	LOAD_PROC_ADDRESS(string_to_utf32_chars, GDExtensionInterfaceStringToUtf32Chars);
	LOAD_PROC_ADDRESS(string_to_wide_chars, GDExtensionInterfaceStringToWideChars);
	LOAD_PROC_ADDRESS(string_operator_index, GDExtensionInterfaceStringOperatorIndex);
	LOAD_PROC_ADDRESS(string_operator_index_const, GDExtensionInterfaceStringOperatorIndexConst);
	LOAD_PROC_ADDRESS(string_operator_plus_eq_string, GDExtensionInterfaceStringOperatorPlusEqString);
	LOAD_PROC_ADDRESS(string_operator_plus_eq_char, GDExtensionInterfaceStringOperatorPlusEqChar);
	LOAD_PROC_ADDRESS(string_operator_plus_eq_cstr, GDExtensionInterfaceStringOperatorPlusEqCstr);
	LOAD_PROC_ADDRESS(string_operator_plus_eq_wcstr, GDExtensionInterfaceStringOperatorPlusEqWcstr);
	LOAD_PROC_ADDRESS(string_operator_plus_eq_c32str, GDExtensionInterfaceStringOperatorPlusEqC32str);
	LOAD_PROC_ADDRESS(xml_parser_open_buffer, GDExtensionInterfaceXmlParserOpenBuffer);
	LOAD_PROC_ADDRESS(file_access_store_buffer, GDExtensionInterfaceFileAccessStoreBuffer);
	LOAD_PROC_ADDRESS(file_access_get_buffer, GDExtensionInterfaceFileAccessGetBuffer);
	LOAD_PROC_ADDRESS(worker_thread_pool_add_native_group_task, GDExtensionInterfaceWorkerThreadPoolAddNativeGroupTask);
	LOAD_PROC_ADDRESS(worker_thread_pool_add_native_task, GDExtensionInterfaceWorkerThreadPoolAddNativeTask);
	LOAD_PROC_ADDRESS(packed_byte_array_operator_index, GDExtensionInterfacePackedByteArrayOperatorIndex);
	LOAD_PROC_ADDRESS(packed_byte_array_operator_index_const, GDExtensionInterfacePackedByteArrayOperatorIndexConst);
	LOAD_PROC_ADDRESS(packed_float32_array_operator_index, GDExtensionInterfacePackedFloat32ArrayOperatorIndex);
	LOAD_PROC_ADDRESS(packed_float32_array_operator_index_const, GDExtensionInterfacePackedFloat32ArrayOperatorIndexConst);
	LOAD_PROC_ADDRESS(packed_float64_array_operator_index, GDExtensionInterfacePackedFloat64ArrayOperatorIndex);
	LOAD_PROC_ADDRESS(packed_float64_array_operator_index_const, GDExtensionInterfacePackedFloat64ArrayOperatorIndexConst);
	LOAD_PROC_ADDRESS(packed_int32_array_operator_index, GDExtensionInterfacePackedInt32ArrayOperatorIndex);
	LOAD_PROC_ADDRESS(packed_int32_array_operator_index_const, GDExtensionInterfacePackedInt32ArrayOperatorIndexConst);
	LOAD_PROC_ADDRESS(packed_int64_array_operator_index, GDExtensionInterfacePackedInt64ArrayOperatorIndex);
	LOAD_PROC_ADDRESS(packed_int64_array_operator_index_const, GDExtensionInterfacePackedInt64ArrayOperatorIndexConst);
	LOAD_PROC_ADDRESS(packed_string_array_operator_index, GDExtensionInterfacePackedStringArrayOperatorIndex);
	LOAD_PROC_ADDRESS(packed_string_array_operator_index_const, GDExtensionInterfacePackedStringArrayOperatorIndexConst);
	LOAD_PROC_ADDRESS(packed_vector2_array_operator_index, GDExtensionInterfacePackedVector2ArrayOperatorIndex);
	LOAD_PROC_ADDRESS(packed_vector2_array_operator_index_const, GDExtensionInterfacePackedVector2ArrayOperatorIndexConst);
	LOAD_PROC_ADDRESS(packed_vector3_array_operator_index, GDExtensionInterfacePackedVector3ArrayOperatorIndex);
	LOAD_PROC_ADDRESS(packed_vector3_array_operator_index_const, GDExtensionInterfacePackedVector3ArrayOperatorIndexConst);
	LOAD_PROC_ADDRESS(packed_color_array_operator_index, GDExtensionInterfacePackedColorArrayOperatorIndex);
	LOAD_PROC_ADDRESS(packed_color_array_operator_index_const, GDExtensionInterfacePackedColorArrayOperatorIndexConst);
	LOAD_PROC_ADDRESS(array_operator_index, GDExtensionInterfaceArrayOperatorIndex);
	LOAD_PROC_ADDRESS(array_operator_index_const, GDExtensionInterfaceArrayOperatorIndexConst);
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 5
	LOAD_PROC_ADDRESS(array_ref, GDExtensionInterfaceArrayRef);
#endif
	LOAD_PROC_ADDRESS(array_set_typed, GDExtensionInterfaceArraySetTyped);
	LOAD_PROC_ADDRESS(dictionary_operator_index, GDExtensionInterfaceDictionaryOperatorIndex);
	LOAD_PROC_ADDRESS(dictionary_operator_index_const, GDExtensionInterfaceDictionaryOperatorIndexConst);
	LOAD_PROC_ADDRESS(object_method_bind_call, GDExtensionInterfaceObjectMethodBindCall);
	LOAD_PROC_ADDRESS(object_method_bind_ptrcall, GDExtensionInterfaceObjectMethodBindPtrcall);
	LOAD_PROC_ADDRESS(object_destroy, GDExtensionInterfaceObjectDestroy);
	LOAD_PROC_ADDRESS(global_get_singleton, GDExtensionInterfaceGlobalGetSingleton);
	LOAD_PROC_ADDRESS(object_get_instance_binding, GDExtensionInterfaceObjectGetInstanceBinding);
	LOAD_PROC_ADDRESS(object_set_instance_binding, GDExtensionInterfaceObjectSetInstanceBinding);
	LOAD_PROC_ADDRESS(object_set_instance, GDExtensionInterfaceObjectSetInstance);
	LOAD_PROC_ADDRESS(object_get_class_name, GDExtensionInterfaceObjectGetClassName);
	LOAD_PROC_ADDRESS(object_cast_to, GDExtensionInterfaceObjectCastTo);
	LOAD_PROC_ADDRESS(object_get_instance_from_id, GDExtensionInterfaceObjectGetInstanceFromId);
	LOAD_PROC_ADDRESS(object_get_instance_id, GDExtensionInterfaceObjectGetInstanceId);
	LOAD_PROC_ADDRESS(ref_get_object, GDExtensionInterfaceRefGetObject);
	LOAD_PROC_ADDRESS(ref_set_object, GDExtensionInterfaceRefSetObject);
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 2
	LOAD_PROC_ADDRESS(script_instance_create, GDExtensionInterfaceScriptInstanceCreate);
#endif
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 4
	LOAD_PROC_ADDRESS(classdb_construct_object, GDExtensionInterfaceClassdbConstructObject);
#endif
	LOAD_PROC_ADDRESS(classdb_get_method_bind, GDExtensionInterfaceClassdbGetMethodBind);
	LOAD_PROC_ADDRESS(classdb_get_class_tag, GDExtensionInterfaceClassdbGetClassTag);
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 2
	LOAD_PROC_ADDRESS(classdb_register_extension_class, GDExtensionInterfaceClassdbRegisterExtensionClass);
#endif
	LOAD_PROC_ADDRESS(classdb_register_extension_class_method, GDExtensionInterfaceClassdbRegisterExtensionClassMethod);
	LOAD_PROC_ADDRESS(classdb_register_extension_class_integer_constant, GDExtensionInterfaceClassdbRegisterExtensionClassIntegerConstant);
	LOAD_PROC_ADDRESS(classdb_register_extension_class_property, GDExtensionInterfaceClassdbRegisterExtensionClassProperty);
	LOAD_PROC_ADDRESS(classdb_register_extension_class_property_group, GDExtensionInterfaceClassdbRegisterExtensionClassPropertyGroup);
	LOAD_PROC_ADDRESS(classdb_register_extension_class_property_subgroup, GDExtensionInterfaceClassdbRegisterExtensionClassPropertySubgroup);
	LOAD_PROC_ADDRESS(classdb_register_extension_class_signal, GDExtensionInterfaceClassdbRegisterExtensionClassSignal);
	LOAD_PROC_ADDRESS(classdb_unregister_extension_class, GDExtensionInterfaceClassdbUnregisterExtensionClass);
	LOAD_PROC_ADDRESS(get_library_path, GDExtensionInterfaceGetLibraryPath);
	LOAD_PROC_ADDRESS(editor_add_plugin, GDExtensionInterfaceEditorAddPlugin);
	LOAD_PROC_ADDRESS(editor_remove_plugin, GDExtensionInterfaceEditorRemovePlugin);
#endif // GODOT_VERSION_MINOR >= 1

	// Godot 4.2 or newer.
#if GODOT_VERSION_MINOR >= 2
	LOAD_PROC_ADDRESS(string_resize, GDExtensionInterfaceStringResize);
	LOAD_PROC_ADDRESS(string_name_new_with_latin1_chars, GDExtensionInterfaceStringNameNewWithLatin1Chars);
	LOAD_PROC_ADDRESS(string_name_new_with_utf8_chars, GDExtensionInterfaceStringNameNewWithUtf8Chars);
	LOAD_PROC_ADDRESS(string_name_new_with_utf8_chars_and_len, GDExtensionInterfaceStringNameNewWithUtf8CharsAndLen);
	LOAD_PROC_ADDRESS(object_free_instance_binding, GDExtensionInterfaceObjectFreeInstanceBinding);
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 3
	LOAD_PROC_ADDRESS(script_instance_create2, GDExtensionInterfaceScriptInstanceCreate2);
#endif
	LOAD_PROC_ADDRESS(placeholder_script_instance_create, GDExtensionInterfacePlaceholderScriptInstanceCreate);
	LOAD_PROC_ADDRESS(placeholder_script_instance_update, GDExtensionInterfacePlaceholderScriptInstanceUpdate);
	LOAD_PROC_ADDRESS(object_get_script_instance, GDExtensionInterfaceObjectGetScriptInstance);
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 3
	LOAD_PROC_ADDRESS(callable_custom_create, GDExtensionInterfaceCallableCustomCreate);
#endif
	LOAD_PROC_ADDRESS(callable_custom_get_userdata, GDExtensionInterfaceCallableCustomGetUserdata);
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 3
	LOAD_PROC_ADDRESS(classdb_register_extension_class2, GDExtensionInterfaceClassdbRegisterExtensionClass2);
#endif
	LOAD_PROC_ADDRESS(classdb_register_extension_class_property_indexed, GDExtensionInterfaceClassdbRegisterExtensionClassPropertyIndexed);
#endif // GODOT_VERSION_MINOR >= 2

	// Godot 4.3 or newer.
#if GODOT_VERSION_MINOR >= 3
	LOAD_PROC_ADDRESS(string_new_with_utf8_chars_and_len2, GDExtensionInterfaceStringNewWithUtf8CharsAndLen2);
	LOAD_PROC_ADDRESS(string_new_with_utf16_chars_and_len2, GDExtensionInterfaceStringNewWithUtf16CharsAndLen2);
	LOAD_PROC_ADDRESS(image_ptrw, GDExtensionInterfaceImagePtrw);
	LOAD_PROC_ADDRESS(image_ptr, GDExtensionInterfaceImagePtr);
	LOAD_PROC_ADDRESS(packed_vector4_array_operator_index, GDExtensionInterfacePackedVector4ArrayOperatorIndex);
	LOAD_PROC_ADDRESS(packed_vector4_array_operator_index_const, GDExtensionInterfacePackedVector4ArrayOperatorIndexConst);
	LOAD_PROC_ADDRESS(object_has_script_method, GDExtensionInterfaceObjectHasScriptMethod);
	LOAD_PROC_ADDRESS(object_call_script_method, GDExtensionInterfaceObjectCallScriptMethod);
	LOAD_PROC_ADDRESS(script_instance_create3, GDExtensionInterfaceScriptInstanceCreate3);
	LOAD_PROC_ADDRESS(callable_custom_create2, GDExtensionInterfaceCallableCustomCreate2);
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 4
	LOAD_PROC_ADDRESS(classdb_register_extension_class3, GDExtensionInterfaceClassdbRegisterExtensionClass3);
#endif
	LOAD_PROC_ADDRESS(classdb_register_extension_class_virtual_method, GDExtensionInterfaceClassdbRegisterExtensionClassVirtualMethod);
	LOAD_PROC_ADDRESS(editor_help_load_xml_from_utf8_chars, GDExtensionInterfaceEditorHelpLoadXmlFromUtf8Chars);
	LOAD_PROC_ADDRESS(editor_help_load_xml_from_utf8_chars_and_len, GDExtensionInterfaceEditorHelpLoadXmlFromUtf8CharsAndLen);
#endif // GODOT_VERSION_MINOR >= 3

	// Godot 4.4 or newer.
#if GODOT_VERSION_MINOR >= 4
	LOAD_PROC_ADDRESS(variant_get_object_instance_id, GDExtensionInterfaceVariantGetObjectInstanceId);
	LOAD_PROC_ADDRESS(variant_get_ptr_internal_getter, GDExtensionInterfaceVariantGetPtrInternalGetter);
	LOAD_PROC_ADDRESS(dictionary_set_typed, GDExtensionInterfaceDictionarySetTyped);
	LOAD_PROC_ADDRESS(classdb_construct_object2, GDExtensionInterfaceClassdbConstructObject2);
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 5
	LOAD_PROC_ADDRESS(classdb_register_extension_class4, GDExtensionInterfaceClassdbRegisterExtensionClass4);
#endif
#endif // GODOT_VERSION_MINOR >= 4

	// Godot 4.5 or newer.
#if GODOT_VERSION_MINOR >= 5
	LOAD_PROC_ADDRESS(get_godot_version2, GDExtensionInterfaceGetGodotVersion2);
	LOAD_PROC_ADDRESS(object_set_script_instance, GDExtensionInterfaceObjectSetScriptInstance);
	LOAD_PROC_ADDRESS(classdb_register_extension_class5, GDExtensionInterfaceClassdbRegisterExtensionClass5);
	LOAD_PROC_ADDRESS(editor_register_get_classes_used_callback, GDExtensionInterfaceEditorRegisterGetClassesUsedCallback);
	LOAD_PROC_ADDRESS(register_main_loop_callbacks, GDExtensionInterfaceRegisterMainLoopCallbacks);
#endif // GODOT_VERSION_MINOR >= 5

	// Godot 4.6 or newer.
#if GODOT_VERSION_MINOR >= 6
	LOAD_PROC_ADDRESS(mem_alloc2, GDExtensionInterfaceMemAlloc2);
	LOAD_PROC_ADDRESS(mem_realloc2, GDExtensionInterfaceMemRealloc2);
	LOAD_PROC_ADDRESS(mem_free2, GDExtensionInterfaceMemFree2);
#endif // GODOT_VERSION_MINOR >= 6

	return true;
}

} // namespace internal

} // namespace godot