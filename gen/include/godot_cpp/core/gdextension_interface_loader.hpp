/**************************************************************************/
/*  gdextension_interface_loader.hpp                                      */
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

#pragma once

#include <gdextension_interface.h>
#include <godot_cpp/core/version.hpp>

namespace godot {

namespace gdextension_interface {

// Godot 4.1 or newer.
#if GODOT_VERSION_MINOR >= 1
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 5
extern "C" GDExtensionInterfaceGetGodotVersion get_godot_version;
#endif
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 6
extern "C" GDExtensionInterfaceMemAlloc mem_alloc;
#endif
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 6
extern "C" GDExtensionInterfaceMemRealloc mem_realloc;
#endif
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 6
extern "C" GDExtensionInterfaceMemFree mem_free;
#endif
extern "C" GDExtensionInterfacePrintError print_error;
extern "C" GDExtensionInterfacePrintErrorWithMessage print_error_with_message;
extern "C" GDExtensionInterfacePrintWarning print_warning;
extern "C" GDExtensionInterfacePrintWarningWithMessage print_warning_with_message;
extern "C" GDExtensionInterfacePrintScriptError print_script_error;
extern "C" GDExtensionInterfacePrintScriptErrorWithMessage print_script_error_with_message;
extern "C" GDExtensionInterfaceGetNativeStructSize get_native_struct_size;
extern "C" GDExtensionInterfaceVariantNewCopy variant_new_copy;
extern "C" GDExtensionInterfaceVariantNewNil variant_new_nil;
extern "C" GDExtensionInterfaceVariantDestroy variant_destroy;
extern "C" GDExtensionInterfaceVariantCall variant_call;
extern "C" GDExtensionInterfaceVariantCallStatic variant_call_static;
extern "C" GDExtensionInterfaceVariantEvaluate variant_evaluate;
extern "C" GDExtensionInterfaceVariantSet variant_set;
extern "C" GDExtensionInterfaceVariantSetNamed variant_set_named;
extern "C" GDExtensionInterfaceVariantSetKeyed variant_set_keyed;
extern "C" GDExtensionInterfaceVariantSetIndexed variant_set_indexed;
extern "C" GDExtensionInterfaceVariantGet variant_get;
extern "C" GDExtensionInterfaceVariantGetNamed variant_get_named;
extern "C" GDExtensionInterfaceVariantGetKeyed variant_get_keyed;
extern "C" GDExtensionInterfaceVariantGetIndexed variant_get_indexed;
extern "C" GDExtensionInterfaceVariantIterInit variant_iter_init;
extern "C" GDExtensionInterfaceVariantIterNext variant_iter_next;
extern "C" GDExtensionInterfaceVariantIterGet variant_iter_get;
extern "C" GDExtensionInterfaceVariantHash variant_hash;
extern "C" GDExtensionInterfaceVariantRecursiveHash variant_recursive_hash;
extern "C" GDExtensionInterfaceVariantHashCompare variant_hash_compare;
extern "C" GDExtensionInterfaceVariantBooleanize variant_booleanize;
extern "C" GDExtensionInterfaceVariantDuplicate variant_duplicate;
extern "C" GDExtensionInterfaceVariantStringify variant_stringify;
extern "C" GDExtensionInterfaceVariantGetType variant_get_type;
extern "C" GDExtensionInterfaceVariantHasMethod variant_has_method;
extern "C" GDExtensionInterfaceVariantHasMember variant_has_member;
extern "C" GDExtensionInterfaceVariantHasKey variant_has_key;
extern "C" GDExtensionInterfaceVariantGetTypeName variant_get_type_name;
extern "C" GDExtensionInterfaceVariantCanConvert variant_can_convert;
extern "C" GDExtensionInterfaceVariantCanConvertStrict variant_can_convert_strict;
extern "C" GDExtensionInterfaceGetVariantFromTypeConstructor get_variant_from_type_constructor;
extern "C" GDExtensionInterfaceGetVariantToTypeConstructor get_variant_to_type_constructor;
extern "C" GDExtensionInterfaceVariantGetPtrOperatorEvaluator variant_get_ptr_operator_evaluator;
extern "C" GDExtensionInterfaceVariantGetPtrBuiltinMethod variant_get_ptr_builtin_method;
extern "C" GDExtensionInterfaceVariantGetPtrConstructor variant_get_ptr_constructor;
extern "C" GDExtensionInterfaceVariantGetPtrDestructor variant_get_ptr_destructor;
extern "C" GDExtensionInterfaceVariantConstruct variant_construct;
extern "C" GDExtensionInterfaceVariantGetPtrSetter variant_get_ptr_setter;
extern "C" GDExtensionInterfaceVariantGetPtrGetter variant_get_ptr_getter;
extern "C" GDExtensionInterfaceVariantGetPtrIndexedSetter variant_get_ptr_indexed_setter;
extern "C" GDExtensionInterfaceVariantGetPtrIndexedGetter variant_get_ptr_indexed_getter;
extern "C" GDExtensionInterfaceVariantGetPtrKeyedSetter variant_get_ptr_keyed_setter;
extern "C" GDExtensionInterfaceVariantGetPtrKeyedGetter variant_get_ptr_keyed_getter;
extern "C" GDExtensionInterfaceVariantGetPtrKeyedChecker variant_get_ptr_keyed_checker;
extern "C" GDExtensionInterfaceVariantGetConstantValue variant_get_constant_value;
extern "C" GDExtensionInterfaceVariantGetPtrUtilityFunction variant_get_ptr_utility_function;
extern "C" GDExtensionInterfaceStringNewWithLatin1Chars string_new_with_latin1_chars;
extern "C" GDExtensionInterfaceStringNewWithUtf8Chars string_new_with_utf8_chars;
extern "C" GDExtensionInterfaceStringNewWithUtf16Chars string_new_with_utf16_chars;
extern "C" GDExtensionInterfaceStringNewWithUtf32Chars string_new_with_utf32_chars;
extern "C" GDExtensionInterfaceStringNewWithWideChars string_new_with_wide_chars;
extern "C" GDExtensionInterfaceStringNewWithLatin1CharsAndLen string_new_with_latin1_chars_and_len;
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 3
extern "C" GDExtensionInterfaceStringNewWithUtf8CharsAndLen string_new_with_utf8_chars_and_len;
#endif
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 3
extern "C" GDExtensionInterfaceStringNewWithUtf16CharsAndLen string_new_with_utf16_chars_and_len;
#endif
extern "C" GDExtensionInterfaceStringNewWithUtf32CharsAndLen string_new_with_utf32_chars_and_len;
extern "C" GDExtensionInterfaceStringNewWithWideCharsAndLen string_new_with_wide_chars_and_len;
extern "C" GDExtensionInterfaceStringToLatin1Chars string_to_latin1_chars;
extern "C" GDExtensionInterfaceStringToUtf8Chars string_to_utf8_chars;
extern "C" GDExtensionInterfaceStringToUtf16Chars string_to_utf16_chars;
extern "C" GDExtensionInterfaceStringToUtf32Chars string_to_utf32_chars;
extern "C" GDExtensionInterfaceStringToWideChars string_to_wide_chars;
extern "C" GDExtensionInterfaceStringOperatorIndex string_operator_index;
extern "C" GDExtensionInterfaceStringOperatorIndexConst string_operator_index_const;
extern "C" GDExtensionInterfaceStringOperatorPlusEqString string_operator_plus_eq_string;
extern "C" GDExtensionInterfaceStringOperatorPlusEqChar string_operator_plus_eq_char;
extern "C" GDExtensionInterfaceStringOperatorPlusEqCstr string_operator_plus_eq_cstr;
extern "C" GDExtensionInterfaceStringOperatorPlusEqWcstr string_operator_plus_eq_wcstr;
extern "C" GDExtensionInterfaceStringOperatorPlusEqC32str string_operator_plus_eq_c32str;
extern "C" GDExtensionInterfaceXmlParserOpenBuffer xml_parser_open_buffer;
extern "C" GDExtensionInterfaceFileAccessStoreBuffer file_access_store_buffer;
extern "C" GDExtensionInterfaceFileAccessGetBuffer file_access_get_buffer;
extern "C" GDExtensionInterfaceWorkerThreadPoolAddNativeGroupTask worker_thread_pool_add_native_group_task;
extern "C" GDExtensionInterfaceWorkerThreadPoolAddNativeTask worker_thread_pool_add_native_task;
extern "C" GDExtensionInterfacePackedByteArrayOperatorIndex packed_byte_array_operator_index;
extern "C" GDExtensionInterfacePackedByteArrayOperatorIndexConst packed_byte_array_operator_index_const;
extern "C" GDExtensionInterfacePackedFloat32ArrayOperatorIndex packed_float32_array_operator_index;
extern "C" GDExtensionInterfacePackedFloat32ArrayOperatorIndexConst packed_float32_array_operator_index_const;
extern "C" GDExtensionInterfacePackedFloat64ArrayOperatorIndex packed_float64_array_operator_index;
extern "C" GDExtensionInterfacePackedFloat64ArrayOperatorIndexConst packed_float64_array_operator_index_const;
extern "C" GDExtensionInterfacePackedInt32ArrayOperatorIndex packed_int32_array_operator_index;
extern "C" GDExtensionInterfacePackedInt32ArrayOperatorIndexConst packed_int32_array_operator_index_const;
extern "C" GDExtensionInterfacePackedInt64ArrayOperatorIndex packed_int64_array_operator_index;
extern "C" GDExtensionInterfacePackedInt64ArrayOperatorIndexConst packed_int64_array_operator_index_const;
extern "C" GDExtensionInterfacePackedStringArrayOperatorIndex packed_string_array_operator_index;
extern "C" GDExtensionInterfacePackedStringArrayOperatorIndexConst packed_string_array_operator_index_const;
extern "C" GDExtensionInterfacePackedVector2ArrayOperatorIndex packed_vector2_array_operator_index;
extern "C" GDExtensionInterfacePackedVector2ArrayOperatorIndexConst packed_vector2_array_operator_index_const;
extern "C" GDExtensionInterfacePackedVector3ArrayOperatorIndex packed_vector3_array_operator_index;
extern "C" GDExtensionInterfacePackedVector3ArrayOperatorIndexConst packed_vector3_array_operator_index_const;
extern "C" GDExtensionInterfacePackedColorArrayOperatorIndex packed_color_array_operator_index;
extern "C" GDExtensionInterfacePackedColorArrayOperatorIndexConst packed_color_array_operator_index_const;
extern "C" GDExtensionInterfaceArrayOperatorIndex array_operator_index;
extern "C" GDExtensionInterfaceArrayOperatorIndexConst array_operator_index_const;
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 5
extern "C" GDExtensionInterfaceArrayRef array_ref;
#endif
extern "C" GDExtensionInterfaceArraySetTyped array_set_typed;
extern "C" GDExtensionInterfaceDictionaryOperatorIndex dictionary_operator_index;
extern "C" GDExtensionInterfaceDictionaryOperatorIndexConst dictionary_operator_index_const;
extern "C" GDExtensionInterfaceObjectMethodBindCall object_method_bind_call;
extern "C" GDExtensionInterfaceObjectMethodBindPtrcall object_method_bind_ptrcall;
extern "C" GDExtensionInterfaceObjectDestroy object_destroy;
extern "C" GDExtensionInterfaceGlobalGetSingleton global_get_singleton;
extern "C" GDExtensionInterfaceObjectGetInstanceBinding object_get_instance_binding;
extern "C" GDExtensionInterfaceObjectSetInstanceBinding object_set_instance_binding;
extern "C" GDExtensionInterfaceObjectSetInstance object_set_instance;
extern "C" GDExtensionInterfaceObjectGetClassName object_get_class_name;
extern "C" GDExtensionInterfaceObjectCastTo object_cast_to;
extern "C" GDExtensionInterfaceObjectGetInstanceFromId object_get_instance_from_id;
extern "C" GDExtensionInterfaceObjectGetInstanceId object_get_instance_id;
extern "C" GDExtensionInterfaceRefGetObject ref_get_object;
extern "C" GDExtensionInterfaceRefSetObject ref_set_object;
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 2
extern "C" GDExtensionInterfaceScriptInstanceCreate script_instance_create;
#endif
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 4
extern "C" GDExtensionInterfaceClassdbConstructObject classdb_construct_object;
#endif
extern "C" GDExtensionInterfaceClassdbGetMethodBind classdb_get_method_bind;
extern "C" GDExtensionInterfaceClassdbGetClassTag classdb_get_class_tag;
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 2
extern "C" GDExtensionInterfaceClassdbRegisterExtensionClass classdb_register_extension_class;
#endif
extern "C" GDExtensionInterfaceClassdbRegisterExtensionClassMethod classdb_register_extension_class_method;
extern "C" GDExtensionInterfaceClassdbRegisterExtensionClassIntegerConstant classdb_register_extension_class_integer_constant;
extern "C" GDExtensionInterfaceClassdbRegisterExtensionClassProperty classdb_register_extension_class_property;
extern "C" GDExtensionInterfaceClassdbRegisterExtensionClassPropertyGroup classdb_register_extension_class_property_group;
extern "C" GDExtensionInterfaceClassdbRegisterExtensionClassPropertySubgroup classdb_register_extension_class_property_subgroup;
extern "C" GDExtensionInterfaceClassdbRegisterExtensionClassSignal classdb_register_extension_class_signal;
extern "C" GDExtensionInterfaceClassdbUnregisterExtensionClass classdb_unregister_extension_class;
extern "C" GDExtensionInterfaceGetLibraryPath get_library_path;
extern "C" GDExtensionInterfaceEditorAddPlugin editor_add_plugin;
extern "C" GDExtensionInterfaceEditorRemovePlugin editor_remove_plugin;
#endif // GODOT_VERSION_MINOR >= 1

// Godot 4.2 or newer.
#if GODOT_VERSION_MINOR >= 2
extern "C" GDExtensionInterfaceStringResize string_resize;
extern "C" GDExtensionInterfaceStringNameNewWithLatin1Chars string_name_new_with_latin1_chars;
extern "C" GDExtensionInterfaceStringNameNewWithUtf8Chars string_name_new_with_utf8_chars;
extern "C" GDExtensionInterfaceStringNameNewWithUtf8CharsAndLen string_name_new_with_utf8_chars_and_len;
extern "C" GDExtensionInterfaceObjectFreeInstanceBinding object_free_instance_binding;
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 3
extern "C" GDExtensionInterfaceScriptInstanceCreate2 script_instance_create2;
#endif
extern "C" GDExtensionInterfacePlaceholderScriptInstanceCreate placeholder_script_instance_create;
extern "C" GDExtensionInterfacePlaceholderScriptInstanceUpdate placeholder_script_instance_update;
extern "C" GDExtensionInterfaceObjectGetScriptInstance object_get_script_instance;
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 3
extern "C" GDExtensionInterfaceCallableCustomCreate callable_custom_create;
#endif
extern "C" GDExtensionInterfaceCallableCustomGetUserdata callable_custom_get_userdata;
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 3
extern "C" GDExtensionInterfaceClassdbRegisterExtensionClass2 classdb_register_extension_class2;
#endif
extern "C" GDExtensionInterfaceClassdbRegisterExtensionClassPropertyIndexed classdb_register_extension_class_property_indexed;
#endif // GODOT_VERSION_MINOR >= 2

// Godot 4.3 or newer.
#if GODOT_VERSION_MINOR >= 3
extern "C" GDExtensionInterfaceStringNewWithUtf8CharsAndLen2 string_new_with_utf8_chars_and_len2;
extern "C" GDExtensionInterfaceStringNewWithUtf16CharsAndLen2 string_new_with_utf16_chars_and_len2;
extern "C" GDExtensionInterfaceImagePtrw image_ptrw;
extern "C" GDExtensionInterfaceImagePtr image_ptr;
extern "C" GDExtensionInterfacePackedVector4ArrayOperatorIndex packed_vector4_array_operator_index;
extern "C" GDExtensionInterfacePackedVector4ArrayOperatorIndexConst packed_vector4_array_operator_index_const;
extern "C" GDExtensionInterfaceObjectHasScriptMethod object_has_script_method;
extern "C" GDExtensionInterfaceObjectCallScriptMethod object_call_script_method;
extern "C" GDExtensionInterfaceScriptInstanceCreate3 script_instance_create3;
extern "C" GDExtensionInterfaceCallableCustomCreate2 callable_custom_create2;
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 4
extern "C" GDExtensionInterfaceClassdbRegisterExtensionClass3 classdb_register_extension_class3;
#endif
extern "C" GDExtensionInterfaceClassdbRegisterExtensionClassVirtualMethod classdb_register_extension_class_virtual_method;
extern "C" GDExtensionInterfaceEditorHelpLoadXmlFromUtf8Chars editor_help_load_xml_from_utf8_chars;
extern "C" GDExtensionInterfaceEditorHelpLoadXmlFromUtf8CharsAndLen editor_help_load_xml_from_utf8_chars_and_len;
#endif // GODOT_VERSION_MINOR >= 3

// Godot 4.4 or newer.
#if GODOT_VERSION_MINOR >= 4
extern "C" GDExtensionInterfaceVariantGetObjectInstanceId variant_get_object_instance_id;
extern "C" GDExtensionInterfaceVariantGetPtrInternalGetter variant_get_ptr_internal_getter;
extern "C" GDExtensionInterfaceDictionarySetTyped dictionary_set_typed;
extern "C" GDExtensionInterfaceClassdbConstructObject2 classdb_construct_object2;
#if !defined(DISABLE_DEPRECATED) || GODOT_VERSION_MINOR < 5
extern "C" GDExtensionInterfaceClassdbRegisterExtensionClass4 classdb_register_extension_class4;
#endif
#endif // GODOT_VERSION_MINOR >= 4

// Godot 4.5 or newer.
#if GODOT_VERSION_MINOR >= 5
extern "C" GDExtensionInterfaceGetGodotVersion2 get_godot_version2;
extern "C" GDExtensionInterfaceObjectSetScriptInstance object_set_script_instance;
extern "C" GDExtensionInterfaceClassdbRegisterExtensionClass5 classdb_register_extension_class5;
extern "C" GDExtensionInterfaceEditorRegisterGetClassesUsedCallback editor_register_get_classes_used_callback;
extern "C" GDExtensionInterfaceRegisterMainLoopCallbacks register_main_loop_callbacks;
#endif // GODOT_VERSION_MINOR >= 5

// Godot 4.6 or newer.
#if GODOT_VERSION_MINOR >= 6
extern "C" GDExtensionInterfaceMemAlloc2 mem_alloc2;
extern "C" GDExtensionInterfaceMemRealloc2 mem_realloc2;
extern "C" GDExtensionInterfaceMemFree2 mem_free2;
#endif // GODOT_VERSION_MINOR >= 6

} // namespace gdextension_interface

namespace internal {

bool load_gdextension_interface(GDExtensionInterfaceGetProcAddress p_get_proc_address);

} // namespace internal

} // namespace godot