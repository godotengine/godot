/**************************************************************************/
/*  godot.cpp                                                             */
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

#include <godot_cpp/godot.hpp>

#include <godot_cpp/classes/editor_plugin_registration.hpp>
#include <godot_cpp/classes/wrapped.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/memory.hpp>
#include <godot_cpp/core/version.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/error_macros.hpp>

namespace godot {

namespace internal {

GDExtensionInterfaceGetProcAddress gdextension_interface_get_proc_address = nullptr;
GDExtensionClassLibraryPtr library = nullptr;
void *token = nullptr;

GDExtensionGodotVersion2 godot_version = {};

// All of the GDExtension interface functions.
GDExtensionInterfaceGetGodotVersion2 gdextension_interface_get_godot_version2 = nullptr;
GDExtensionInterfaceMemAlloc gdextension_interface_mem_alloc = nullptr;
GDExtensionInterfaceMemRealloc gdextension_interface_mem_realloc = nullptr;
GDExtensionInterfaceMemFree gdextension_interface_mem_free = nullptr;
GDExtensionInterfacePrintError gdextension_interface_print_error = nullptr;
GDExtensionInterfacePrintErrorWithMessage gdextension_interface_print_error_with_message = nullptr;
GDExtensionInterfacePrintWarning gdextension_interface_print_warning = nullptr;
GDExtensionInterfacePrintWarningWithMessage gdextension_interface_print_warning_with_message = nullptr;
GDExtensionInterfacePrintScriptError gdextension_interface_print_script_error = nullptr;
GDExtensionInterfacePrintScriptErrorWithMessage gdextension_interface_print_script_error_with_message = nullptr;
GDExtensionInterfaceGetNativeStructSize gdextension_interface_get_native_struct_size = nullptr;
GDExtensionInterfaceVariantNewCopy gdextension_interface_variant_new_copy = nullptr;
GDExtensionInterfaceVariantNewNil gdextension_interface_variant_new_nil = nullptr;
GDExtensionInterfaceVariantDestroy gdextension_interface_variant_destroy = nullptr;
GDExtensionInterfaceVariantCall gdextension_interface_variant_call = nullptr;
GDExtensionInterfaceVariantCallStatic gdextension_interface_variant_call_static = nullptr;
GDExtensionInterfaceVariantEvaluate gdextension_interface_variant_evaluate = nullptr;
GDExtensionInterfaceVariantSet gdextension_interface_variant_set = nullptr;
GDExtensionInterfaceVariantSetNamed gdextension_interface_variant_set_named = nullptr;
GDExtensionInterfaceVariantSetKeyed gdextension_interface_variant_set_keyed = nullptr;
GDExtensionInterfaceVariantSetIndexed gdextension_interface_variant_set_indexed = nullptr;
GDExtensionInterfaceVariantGet gdextension_interface_variant_get = nullptr;
GDExtensionInterfaceVariantGetNamed gdextension_interface_variant_get_named = nullptr;
GDExtensionInterfaceVariantGetKeyed gdextension_interface_variant_get_keyed = nullptr;
GDExtensionInterfaceVariantGetIndexed gdextension_interface_variant_get_indexed = nullptr;
GDExtensionInterfaceVariantIterInit gdextension_interface_variant_iter_init = nullptr;
GDExtensionInterfaceVariantIterNext gdextension_interface_variant_iter_next = nullptr;
GDExtensionInterfaceVariantIterGet gdextension_interface_variant_iter_get = nullptr;
GDExtensionInterfaceVariantHash gdextension_interface_variant_hash = nullptr;
GDExtensionInterfaceVariantRecursiveHash gdextension_interface_variant_recursive_hash = nullptr;
GDExtensionInterfaceVariantHashCompare gdextension_interface_variant_hash_compare = nullptr;
GDExtensionInterfaceVariantBooleanize gdextension_interface_variant_booleanize = nullptr;
GDExtensionInterfaceVariantDuplicate gdextension_interface_variant_duplicate = nullptr;
GDExtensionInterfaceVariantStringify gdextension_interface_variant_stringify = nullptr;
GDExtensionInterfaceVariantGetType gdextension_interface_variant_get_type = nullptr;
GDExtensionInterfaceVariantHasMethod gdextension_interface_variant_has_method = nullptr;
GDExtensionInterfaceVariantHasMember gdextension_interface_variant_has_member = nullptr;
GDExtensionInterfaceVariantHasKey gdextension_interface_variant_has_key = nullptr;
GDExtensionInterfaceVariantGetObjectInstanceId gdextension_interface_variant_get_object_instance_id = nullptr;
GDExtensionInterfaceVariantGetTypeName gdextension_interface_variant_get_type_name = nullptr;
GDExtensionInterfaceVariantCanConvert gdextension_interface_variant_can_convert = nullptr;
GDExtensionInterfaceVariantCanConvertStrict gdextension_interface_variant_can_convert_strict = nullptr;
GDExtensionInterfaceGetVariantFromTypeConstructor gdextension_interface_get_variant_from_type_constructor = nullptr;
GDExtensionInterfaceGetVariantToTypeConstructor gdextension_interface_get_variant_to_type_constructor = nullptr;
GDExtensionInterfaceGetVariantGetInternalPtrFunc gdextension_interface_variant_get_ptr_internal_getter = nullptr;
GDExtensionInterfaceVariantGetPtrOperatorEvaluator gdextension_interface_variant_get_ptr_operator_evaluator = nullptr;
GDExtensionInterfaceVariantGetPtrBuiltinMethod gdextension_interface_variant_get_ptr_builtin_method = nullptr;
GDExtensionInterfaceVariantGetPtrConstructor gdextension_interface_variant_get_ptr_constructor = nullptr;
GDExtensionInterfaceVariantGetPtrDestructor gdextension_interface_variant_get_ptr_destructor = nullptr;
GDExtensionInterfaceVariantConstruct gdextension_interface_variant_construct = nullptr;
GDExtensionInterfaceVariantGetPtrSetter gdextension_interface_variant_get_ptr_setter = nullptr;
GDExtensionInterfaceVariantGetPtrGetter gdextension_interface_variant_get_ptr_getter = nullptr;
GDExtensionInterfaceVariantGetPtrIndexedSetter gdextension_interface_variant_get_ptr_indexed_setter = nullptr;
GDExtensionInterfaceVariantGetPtrIndexedGetter gdextension_interface_variant_get_ptr_indexed_getter = nullptr;
GDExtensionInterfaceVariantGetPtrKeyedSetter gdextension_interface_variant_get_ptr_keyed_setter = nullptr;
GDExtensionInterfaceVariantGetPtrKeyedGetter gdextension_interface_variant_get_ptr_keyed_getter = nullptr;
GDExtensionInterfaceVariantGetPtrKeyedChecker gdextension_interface_variant_get_ptr_keyed_checker = nullptr;
GDExtensionInterfaceVariantGetConstantValue gdextension_interface_variant_get_constant_value = nullptr;
GDExtensionInterfaceVariantGetPtrUtilityFunction gdextension_interface_variant_get_ptr_utility_function = nullptr;
GDExtensionInterfaceStringNewWithLatin1Chars gdextension_interface_string_new_with_latin1_chars = nullptr;
GDExtensionInterfaceStringNewWithUtf8Chars gdextension_interface_string_new_with_utf8_chars = nullptr;
GDExtensionInterfaceStringNewWithUtf16Chars gdextension_interface_string_new_with_utf16_chars = nullptr;
GDExtensionInterfaceStringNewWithUtf32Chars gdextension_interface_string_new_with_utf32_chars = nullptr;
GDExtensionInterfaceStringNewWithWideChars gdextension_interface_string_new_with_wide_chars = nullptr;
GDExtensionInterfaceStringNewWithLatin1CharsAndLen gdextension_interface_string_new_with_latin1_chars_and_len = nullptr;
GDExtensionInterfaceStringNewWithUtf8CharsAndLen gdextension_interface_string_new_with_utf8_chars_and_len = nullptr;
GDExtensionInterfaceStringNewWithUtf8CharsAndLen2 gdextension_interface_string_new_with_utf8_chars_and_len2 = nullptr;
GDExtensionInterfaceStringNewWithUtf16CharsAndLen gdextension_interface_string_new_with_utf16_chars_and_len = nullptr;
GDExtensionInterfaceStringNewWithUtf16CharsAndLen2 gdextension_interface_string_new_with_utf16_chars_and_len2 = nullptr;
GDExtensionInterfaceStringNewWithUtf32CharsAndLen gdextension_interface_string_new_with_utf32_chars_and_len = nullptr;
GDExtensionInterfaceStringNewWithWideCharsAndLen gdextension_interface_string_new_with_wide_chars_and_len = nullptr;
GDExtensionInterfaceStringToLatin1Chars gdextension_interface_string_to_latin1_chars = nullptr;
GDExtensionInterfaceStringToUtf8Chars gdextension_interface_string_to_utf8_chars = nullptr;
GDExtensionInterfaceStringToUtf16Chars gdextension_interface_string_to_utf16_chars = nullptr;
GDExtensionInterfaceStringToUtf32Chars gdextension_interface_string_to_utf32_chars = nullptr;
GDExtensionInterfaceStringToWideChars gdextension_interface_string_to_wide_chars = nullptr;
GDExtensionInterfaceStringOperatorIndex gdextension_interface_string_operator_index = nullptr;
GDExtensionInterfaceStringOperatorIndexConst gdextension_interface_string_operator_index_const = nullptr;
GDExtensionInterfaceStringOperatorPlusEqString gdextension_interface_string_operator_plus_eq_string = nullptr;
GDExtensionInterfaceStringOperatorPlusEqChar gdextension_interface_string_operator_plus_eq_char = nullptr;
GDExtensionInterfaceStringOperatorPlusEqCstr gdextension_interface_string_operator_plus_eq_cstr = nullptr;
GDExtensionInterfaceStringOperatorPlusEqWcstr gdextension_interface_string_operator_plus_eq_wcstr = nullptr;
GDExtensionInterfaceStringOperatorPlusEqC32str gdextension_interface_string_operator_plus_eq_c32str = nullptr;
GDExtensionInterfaceStringResize gdextension_interface_string_resize = nullptr;
GDExtensionInterfaceStringNameNewWithLatin1Chars gdextension_interface_string_name_new_with_latin1_chars = nullptr;
GDExtensionInterfaceXmlParserOpenBuffer gdextension_interface_xml_parser_open_buffer = nullptr;
GDExtensionInterfaceFileAccessStoreBuffer gdextension_interface_file_access_store_buffer = nullptr;
GDExtensionInterfaceFileAccessGetBuffer gdextension_interface_file_access_get_buffer = nullptr;
GDExtensionInterfaceWorkerThreadPoolAddNativeGroupTask gdextension_interface_worker_thread_pool_add_native_group_task = nullptr;
GDExtensionInterfaceWorkerThreadPoolAddNativeTask gdextension_interface_worker_thread_pool_add_native_task = nullptr;
GDExtensionInterfacePackedByteArrayOperatorIndex gdextension_interface_packed_byte_array_operator_index = nullptr;
GDExtensionInterfacePackedByteArrayOperatorIndexConst gdextension_interface_packed_byte_array_operator_index_const = nullptr;
GDExtensionInterfacePackedColorArrayOperatorIndex gdextension_interface_packed_color_array_operator_index = nullptr;
GDExtensionInterfacePackedColorArrayOperatorIndexConst gdextension_interface_packed_color_array_operator_index_const = nullptr;
GDExtensionInterfacePackedFloat32ArrayOperatorIndex gdextension_interface_packed_float32_array_operator_index = nullptr;
GDExtensionInterfacePackedFloat32ArrayOperatorIndexConst gdextension_interface_packed_float32_array_operator_index_const = nullptr;
GDExtensionInterfacePackedFloat64ArrayOperatorIndex gdextension_interface_packed_float64_array_operator_index = nullptr;
GDExtensionInterfacePackedFloat64ArrayOperatorIndexConst gdextension_interface_packed_float64_array_operator_index_const = nullptr;
GDExtensionInterfacePackedInt32ArrayOperatorIndex gdextension_interface_packed_int32_array_operator_index = nullptr;
GDExtensionInterfacePackedInt32ArrayOperatorIndexConst gdextension_interface_packed_int32_array_operator_index_const = nullptr;
GDExtensionInterfacePackedInt64ArrayOperatorIndex gdextension_interface_packed_int64_array_operator_index = nullptr;
GDExtensionInterfacePackedInt64ArrayOperatorIndexConst gdextension_interface_packed_int64_array_operator_index_const = nullptr;
GDExtensionInterfacePackedStringArrayOperatorIndex gdextension_interface_packed_string_array_operator_index = nullptr;
GDExtensionInterfacePackedStringArrayOperatorIndexConst gdextension_interface_packed_string_array_operator_index_const = nullptr;
GDExtensionInterfacePackedVector2ArrayOperatorIndex gdextension_interface_packed_vector2_array_operator_index = nullptr;
GDExtensionInterfacePackedVector2ArrayOperatorIndexConst gdextension_interface_packed_vector2_array_operator_index_const = nullptr;
GDExtensionInterfacePackedVector3ArrayOperatorIndex gdextension_interface_packed_vector3_array_operator_index = nullptr;
GDExtensionInterfacePackedVector3ArrayOperatorIndexConst gdextension_interface_packed_vector3_array_operator_index_const = nullptr;
GDExtensionInterfacePackedVector4ArrayOperatorIndex gdextension_interface_packed_vector4_array_operator_index = nullptr;
GDExtensionInterfacePackedVector4ArrayOperatorIndexConst gdextension_interface_packed_vector4_array_operator_index_const = nullptr;
GDExtensionInterfaceArrayOperatorIndex gdextension_interface_array_operator_index = nullptr;
GDExtensionInterfaceArrayOperatorIndexConst gdextension_interface_array_operator_index_const = nullptr;
GDExtensionInterfaceArraySetTyped gdextension_interface_array_set_typed = nullptr;
GDExtensionInterfaceDictionaryOperatorIndex gdextension_interface_dictionary_operator_index = nullptr;
GDExtensionInterfaceDictionaryOperatorIndexConst gdextension_interface_dictionary_operator_index_const = nullptr;
GDExtensionInterfaceDictionarySetTyped gdextension_interface_dictionary_set_typed = nullptr;
GDExtensionInterfaceObjectMethodBindCall gdextension_interface_object_method_bind_call = nullptr;
GDExtensionInterfaceObjectMethodBindPtrcall gdextension_interface_object_method_bind_ptrcall = nullptr;
GDExtensionInterfaceObjectDestroy gdextension_interface_object_destroy = nullptr;
GDExtensionInterfaceGlobalGetSingleton gdextension_interface_global_get_singleton = nullptr;
GDExtensionInterfaceObjectGetInstanceBinding gdextension_interface_object_get_instance_binding = nullptr;
GDExtensionInterfaceObjectSetInstanceBinding gdextension_interface_object_set_instance_binding = nullptr;
GDExtensionInterfaceObjectFreeInstanceBinding gdextension_interface_object_free_instance_binding = nullptr;
GDExtensionInterfaceObjectSetInstance gdextension_interface_object_set_instance = nullptr;
GDExtensionInterfaceObjectGetClassName gdextension_interface_object_get_class_name = nullptr;
GDExtensionInterfaceObjectCastTo gdextension_interface_object_cast_to = nullptr;
GDExtensionInterfaceObjectGetInstanceFromId gdextension_interface_object_get_instance_from_id = nullptr;
GDExtensionInterfaceObjectGetInstanceId gdextension_interface_object_get_instance_id = nullptr;
GDExtensionInterfaceObjectHasScriptMethod gdextension_interface_object_has_script_method = nullptr;
GDExtensionInterfaceObjectCallScriptMethod gdextension_interface_object_call_script_method = nullptr;
GDExtensionInterfaceCallableCustomCreate2 gdextension_interface_callable_custom_create2 = nullptr;
GDExtensionInterfaceCallableCustomGetUserData gdextension_interface_callable_custom_get_userdata = nullptr;
GDExtensionInterfaceRefGetObject gdextension_interface_ref_get_object = nullptr;
GDExtensionInterfaceRefSetObject gdextension_interface_ref_set_object = nullptr;
GDExtensionInterfaceScriptInstanceCreate3 gdextension_interface_script_instance_create3 = nullptr;
GDExtensionInterfacePlaceHolderScriptInstanceCreate gdextension_interface_placeholder_script_instance_create = nullptr;
GDExtensionInterfacePlaceHolderScriptInstanceUpdate gdextension_interface_placeholder_script_instance_update = nullptr;
GDExtensionInterfaceObjectGetScriptInstance gdextension_interface_object_get_script_instance = nullptr;
GDExtensionInterfaceObjectSetScriptInstance gdextension_interface_object_set_script_instance = nullptr;
GDExtensionInterfaceClassdbConstructObject2 gdextension_interface_classdb_construct_object2 = nullptr;
GDExtensionInterfaceClassdbGetMethodBind gdextension_interface_classdb_get_method_bind = nullptr;
GDExtensionInterfaceClassdbGetClassTag gdextension_interface_classdb_get_class_tag = nullptr;
GDExtensionInterfaceClassdbRegisterExtensionClass5 gdextension_interface_classdb_register_extension_class5 = nullptr;
GDExtensionInterfaceClassdbRegisterExtensionClassMethod gdextension_interface_classdb_register_extension_class_method = nullptr;
GDExtensionInterfaceClassdbRegisterExtensionClassVirtualMethod gdextension_interface_classdb_register_extension_class_virtual_method = nullptr;
GDExtensionInterfaceClassdbRegisterExtensionClassIntegerConstant gdextension_interface_classdb_register_extension_class_integer_constant = nullptr;
GDExtensionInterfaceClassdbRegisterExtensionClassProperty gdextension_interface_classdb_register_extension_class_property = nullptr;
GDExtensionInterfaceClassdbRegisterExtensionClassPropertyIndexed gdextension_interface_classdb_register_extension_class_property_indexed = nullptr;
GDExtensionInterfaceClassdbRegisterExtensionClassPropertyGroup gdextension_interface_classdb_register_extension_class_property_group = nullptr;
GDExtensionInterfaceClassdbRegisterExtensionClassPropertySubgroup gdextension_interface_classdb_register_extension_class_property_subgroup = nullptr;
GDExtensionInterfaceClassdbRegisterExtensionClassSignal gdextension_interface_classdb_register_extension_class_signal = nullptr;
GDExtensionInterfaceClassdbUnregisterExtensionClass gdextension_interface_classdb_unregister_extension_class = nullptr;
GDExtensionInterfaceGetLibraryPath gdextension_interface_get_library_path = nullptr;
GDExtensionInterfaceEditorAddPlugin gdextension_interface_editor_add_plugin = nullptr;
GDExtensionInterfaceEditorRemovePlugin gdextension_interface_editor_remove_plugin = nullptr;
GDExtensionInterfaceEditorRegisterGetClassesUsedCallback gdextension_interface_editor_register_get_classes_used_callback = nullptr;
GDExtensionsInterfaceEditorHelpLoadXmlFromUtf8Chars gdextension_interface_editor_help_load_xml_from_utf8_chars = nullptr;
GDExtensionsInterfaceEditorHelpLoadXmlFromUtf8CharsAndLen gdextension_interface_editor_help_load_xml_from_utf8_chars_and_len = nullptr;
GDExtensionInterfaceImagePtrw gdextension_interface_image_ptrw = nullptr;
GDExtensionInterfaceImagePtr gdextension_interface_image_ptr = nullptr;
GDExtensionInterfaceRegisterMainLoopCallbacks gdextension_interface_register_main_loop_callbacks = nullptr;

struct DocData {
	const char *hash = nullptr;
	int uncompressed_size = 0;
	int compressed_size = 0;
	const unsigned char *data = nullptr;

	inline bool is_valid() const {
		return hash != nullptr && uncompressed_size > 0 && compressed_size > 0 && data != nullptr;
	}

	void load_data() const;
};

static DocData &get_doc_data() {
	static DocData doc_data;
	return doc_data;
}

DocDataRegistration::DocDataRegistration(const char *p_hash, int p_uncompressed_size, int p_compressed_size, const unsigned char *p_data) {
	DocData &doc_data = get_doc_data();
	if (doc_data.is_valid()) {
		printf("ERROR: Attempting to register documentation data when we already have some - discarding.\n");
		return;
	}
	doc_data.hash = p_hash;
	doc_data.uncompressed_size = p_uncompressed_size;
	doc_data.compressed_size = p_compressed_size;
	doc_data.data = p_data;
}

} // namespace internal

bool GDExtensionBinding::api_initialized = false;
int GDExtensionBinding::level_initialized[MODULE_INITIALIZATION_LEVEL_MAX] = { 0 };
GDExtensionBinding::InitDataList GDExtensionBinding::initdata;

#define ERR_PRINT_EARLY(m_msg) \
	internal::gdextension_interface_print_error(m_msg, FUNCTION_STR, __FILE__, __LINE__, false)

#define LOAD_PROC_ADDRESS(m_name, m_type)                                               \
	internal::gdextension_interface_##m_name = (m_type)p_get_proc_address(#m_name);     \
	if (!internal::gdextension_interface_##m_name) {                                    \
		ERR_PRINT_EARLY("Unable to load GDExtension interface function " #m_name "()"); \
		return false;                                                                   \
	}

// Partial definition of the legacy interface so we can detect it and show an error.
typedef struct {
	uint32_t version_major;
	uint32_t version_minor;
	uint32_t version_patch;
	const char *version_string;

	GDExtensionInterfaceFunctionPtr unused1;
	GDExtensionInterfaceFunctionPtr unused2;
	GDExtensionInterfaceFunctionPtr unused3;

	GDExtensionInterfacePrintError print_error;
	GDExtensionInterfacePrintErrorWithMessage print_error_with_message;
} LegacyGDExtensionInterface;

GDExtensionBool GDExtensionBinding::init(GDExtensionInterfaceGetProcAddress p_get_proc_address, GDExtensionClassLibraryPtr p_library, InitData *p_init_data, GDExtensionInitialization *r_initialization) {
	if (!p_init_data || !p_init_data->init_callback) {
		ERR_FAIL_V_MSG(false, "Initialization callback must be defined.");
	}

	if (api_initialized) {
		r_initialization->initialize = initialize_level;
		r_initialization->deinitialize = deinitialize_level;
		r_initialization->userdata = p_init_data;
		r_initialization->minimum_initialization_level = p_init_data->minimum_initialization_level;

		return true;
	}

	// Make sure we weren't passed the legacy struct.
	uint32_t *raw_interface = (uint32_t *)(void *)p_get_proc_address;
	if (uintptr_t(p_get_proc_address) % alignof(LegacyGDExtensionInterface) == 0 && raw_interface[0] == 4 && raw_interface[1] == 0) {
		// Use the legacy interface only to give a nice error.
		LegacyGDExtensionInterface *legacy_interface = (LegacyGDExtensionInterface *)p_get_proc_address;
		internal::gdextension_interface_print_error = (GDExtensionInterfacePrintError)legacy_interface->print_error;
		ERR_PRINT_EARLY("Cannot load a GDExtension built for Godot 4.1+ in Godot 4.0.");
		return false;
	}

	// Load the "print_error" function first (needed by the ERR_PRINT_EARLY() macro).
	internal::gdextension_interface_print_error = (GDExtensionInterfacePrintError)p_get_proc_address("print_error");
	if (!internal::gdextension_interface_print_error) {
		printf("ERROR: Unable to load GDExtension interface function print_error().\n");
		return false;
	}

	internal::gdextension_interface_get_proc_address = p_get_proc_address;
	internal::library = p_library;
	internal::token = p_library;

	LOAD_PROC_ADDRESS(get_godot_version2, GDExtensionInterfaceGetGodotVersion2);
	internal::gdextension_interface_get_godot_version2(&internal::godot_version);

	// Check that godot-cpp was compiled using an extension_api.json older or at the
	// same version as the Godot that is loading it.
	bool compatible;
	if (internal::godot_version.major != GODOT_VERSION_MAJOR) {
		compatible = internal::godot_version.major > GODOT_VERSION_MAJOR;
	} else if (internal::godot_version.minor != GODOT_VERSION_MINOR) {
		compatible = internal::godot_version.minor > GODOT_VERSION_MINOR;
	} else {
#if GODOT_VERSION_PATCH > 0
		compatible = internal::godot_version.patch >= GODOT_VERSION_PATCH;
#else
		// Prevent -Wtype-limits warning due to unsigned comparison.
		compatible = true;
#endif
	}
	if (!compatible) {
		// We need to use snprintf() here because vformat() uses Variant, and we haven't loaded
		// the GDExtension interface far enough to use Variants yet.
		char msg[128];
		snprintf(msg, 128, "Cannot load a GDExtension built for Godot %d.%d.%d using an older version of Godot (%d.%d.%d).",
				GODOT_VERSION_MAJOR, GODOT_VERSION_MINOR, GODOT_VERSION_PATCH,
				internal::godot_version.major, internal::godot_version.minor, internal::godot_version.patch);
		ERR_PRINT_EARLY(msg);
		return false;
	}

	LOAD_PROC_ADDRESS(mem_alloc, GDExtensionInterfaceMemAlloc);
	LOAD_PROC_ADDRESS(mem_realloc, GDExtensionInterfaceMemRealloc);
	LOAD_PROC_ADDRESS(mem_free, GDExtensionInterfaceMemFree);
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
	LOAD_PROC_ADDRESS(variant_get_object_instance_id, GDExtensionInterfaceVariantGetObjectInstanceId);
	LOAD_PROC_ADDRESS(variant_get_type_name, GDExtensionInterfaceVariantGetTypeName);
	LOAD_PROC_ADDRESS(variant_can_convert, GDExtensionInterfaceVariantCanConvert);
	LOAD_PROC_ADDRESS(variant_can_convert_strict, GDExtensionInterfaceVariantCanConvertStrict);
	LOAD_PROC_ADDRESS(get_variant_from_type_constructor, GDExtensionInterfaceGetVariantFromTypeConstructor);
	LOAD_PROC_ADDRESS(get_variant_to_type_constructor, GDExtensionInterfaceGetVariantToTypeConstructor);
	LOAD_PROC_ADDRESS(variant_get_ptr_internal_getter, GDExtensionInterfaceGetVariantGetInternalPtrFunc);
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
	LOAD_PROC_ADDRESS(string_new_with_utf8_chars_and_len, GDExtensionInterfaceStringNewWithUtf8CharsAndLen);
	LOAD_PROC_ADDRESS(string_new_with_utf8_chars_and_len2, GDExtensionInterfaceStringNewWithUtf8CharsAndLen2);
	LOAD_PROC_ADDRESS(string_new_with_utf16_chars_and_len, GDExtensionInterfaceStringNewWithUtf16CharsAndLen);
	LOAD_PROC_ADDRESS(string_new_with_utf16_chars_and_len2, GDExtensionInterfaceStringNewWithUtf16CharsAndLen2);
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
	LOAD_PROC_ADDRESS(string_resize, GDExtensionInterfaceStringResize);
	LOAD_PROC_ADDRESS(string_name_new_with_latin1_chars, GDExtensionInterfaceStringNameNewWithLatin1Chars);
	LOAD_PROC_ADDRESS(xml_parser_open_buffer, GDExtensionInterfaceXmlParserOpenBuffer);
	LOAD_PROC_ADDRESS(file_access_store_buffer, GDExtensionInterfaceFileAccessStoreBuffer);
	LOAD_PROC_ADDRESS(file_access_get_buffer, GDExtensionInterfaceFileAccessGetBuffer);
	LOAD_PROC_ADDRESS(worker_thread_pool_add_native_group_task, GDExtensionInterfaceWorkerThreadPoolAddNativeGroupTask);
	LOAD_PROC_ADDRESS(worker_thread_pool_add_native_task, GDExtensionInterfaceWorkerThreadPoolAddNativeTask);
	LOAD_PROC_ADDRESS(packed_byte_array_operator_index, GDExtensionInterfacePackedByteArrayOperatorIndex);
	LOAD_PROC_ADDRESS(packed_byte_array_operator_index_const, GDExtensionInterfacePackedByteArrayOperatorIndexConst);
	LOAD_PROC_ADDRESS(packed_color_array_operator_index, GDExtensionInterfacePackedColorArrayOperatorIndex);
	LOAD_PROC_ADDRESS(packed_color_array_operator_index_const, GDExtensionInterfacePackedColorArrayOperatorIndexConst);
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
	LOAD_PROC_ADDRESS(packed_vector4_array_operator_index, GDExtensionInterfacePackedVector4ArrayOperatorIndex);
	LOAD_PROC_ADDRESS(packed_vector4_array_operator_index_const, GDExtensionInterfacePackedVector4ArrayOperatorIndexConst);
	LOAD_PROC_ADDRESS(array_operator_index, GDExtensionInterfaceArrayOperatorIndex);
	LOAD_PROC_ADDRESS(array_operator_index_const, GDExtensionInterfaceArrayOperatorIndexConst);
	LOAD_PROC_ADDRESS(array_set_typed, GDExtensionInterfaceArraySetTyped);
	LOAD_PROC_ADDRESS(dictionary_operator_index, GDExtensionInterfaceDictionaryOperatorIndex);
	LOAD_PROC_ADDRESS(dictionary_operator_index_const, GDExtensionInterfaceDictionaryOperatorIndexConst);
	LOAD_PROC_ADDRESS(dictionary_set_typed, GDExtensionInterfaceDictionarySetTyped);
	LOAD_PROC_ADDRESS(object_method_bind_call, GDExtensionInterfaceObjectMethodBindCall);
	LOAD_PROC_ADDRESS(object_method_bind_ptrcall, GDExtensionInterfaceObjectMethodBindPtrcall);
	LOAD_PROC_ADDRESS(object_destroy, GDExtensionInterfaceObjectDestroy);
	LOAD_PROC_ADDRESS(global_get_singleton, GDExtensionInterfaceGlobalGetSingleton);
	LOAD_PROC_ADDRESS(object_get_instance_binding, GDExtensionInterfaceObjectGetInstanceBinding);
	LOAD_PROC_ADDRESS(object_set_instance_binding, GDExtensionInterfaceObjectSetInstanceBinding);
	LOAD_PROC_ADDRESS(object_free_instance_binding, GDExtensionInterfaceObjectFreeInstanceBinding);
	LOAD_PROC_ADDRESS(object_set_instance, GDExtensionInterfaceObjectSetInstance);
	LOAD_PROC_ADDRESS(object_get_class_name, GDExtensionInterfaceObjectGetClassName);
	LOAD_PROC_ADDRESS(object_cast_to, GDExtensionInterfaceObjectCastTo);
	LOAD_PROC_ADDRESS(object_get_instance_from_id, GDExtensionInterfaceObjectGetInstanceFromId);
	LOAD_PROC_ADDRESS(object_get_instance_id, GDExtensionInterfaceObjectGetInstanceId);
	LOAD_PROC_ADDRESS(object_has_script_method, GDExtensionInterfaceObjectHasScriptMethod);
	LOAD_PROC_ADDRESS(object_call_script_method, GDExtensionInterfaceObjectCallScriptMethod);
	LOAD_PROC_ADDRESS(callable_custom_create2, GDExtensionInterfaceCallableCustomCreate2);
	LOAD_PROC_ADDRESS(callable_custom_get_userdata, GDExtensionInterfaceCallableCustomGetUserData);
	LOAD_PROC_ADDRESS(ref_get_object, GDExtensionInterfaceRefGetObject);
	LOAD_PROC_ADDRESS(ref_set_object, GDExtensionInterfaceRefSetObject);
	LOAD_PROC_ADDRESS(script_instance_create3, GDExtensionInterfaceScriptInstanceCreate3);
	LOAD_PROC_ADDRESS(placeholder_script_instance_create, GDExtensionInterfacePlaceHolderScriptInstanceCreate);
	LOAD_PROC_ADDRESS(placeholder_script_instance_update, GDExtensionInterfacePlaceHolderScriptInstanceUpdate);
	LOAD_PROC_ADDRESS(object_get_script_instance, GDExtensionInterfaceObjectGetScriptInstance);
	LOAD_PROC_ADDRESS(object_set_script_instance, GDExtensionInterfaceObjectSetScriptInstance);
	LOAD_PROC_ADDRESS(classdb_construct_object2, GDExtensionInterfaceClassdbConstructObject2);
	LOAD_PROC_ADDRESS(classdb_get_method_bind, GDExtensionInterfaceClassdbGetMethodBind);
	LOAD_PROC_ADDRESS(classdb_get_class_tag, GDExtensionInterfaceClassdbGetClassTag);
	LOAD_PROC_ADDRESS(classdb_register_extension_class5, GDExtensionInterfaceClassdbRegisterExtensionClass5);
	LOAD_PROC_ADDRESS(classdb_register_extension_class_method, GDExtensionInterfaceClassdbRegisterExtensionClassMethod);
	LOAD_PROC_ADDRESS(classdb_register_extension_class_virtual_method, GDExtensionInterfaceClassdbRegisterExtensionClassVirtualMethod);
	LOAD_PROC_ADDRESS(classdb_register_extension_class_integer_constant, GDExtensionInterfaceClassdbRegisterExtensionClassIntegerConstant);
	LOAD_PROC_ADDRESS(classdb_register_extension_class_property, GDExtensionInterfaceClassdbRegisterExtensionClassProperty);
	LOAD_PROC_ADDRESS(classdb_register_extension_class_property_indexed, GDExtensionInterfaceClassdbRegisterExtensionClassPropertyIndexed);
	LOAD_PROC_ADDRESS(classdb_register_extension_class_property_group, GDExtensionInterfaceClassdbRegisterExtensionClassPropertyGroup);
	LOAD_PROC_ADDRESS(classdb_register_extension_class_property_subgroup, GDExtensionInterfaceClassdbRegisterExtensionClassPropertySubgroup);
	LOAD_PROC_ADDRESS(classdb_register_extension_class_signal, GDExtensionInterfaceClassdbRegisterExtensionClassSignal);
	LOAD_PROC_ADDRESS(classdb_unregister_extension_class, GDExtensionInterfaceClassdbUnregisterExtensionClass);
	LOAD_PROC_ADDRESS(get_library_path, GDExtensionInterfaceGetLibraryPath);
	LOAD_PROC_ADDRESS(editor_add_plugin, GDExtensionInterfaceEditorAddPlugin);
	LOAD_PROC_ADDRESS(editor_remove_plugin, GDExtensionInterfaceEditorRemovePlugin);
	LOAD_PROC_ADDRESS(editor_register_get_classes_used_callback, GDExtensionInterfaceEditorRegisterGetClassesUsedCallback);
	LOAD_PROC_ADDRESS(editor_help_load_xml_from_utf8_chars, GDExtensionsInterfaceEditorHelpLoadXmlFromUtf8Chars);
	LOAD_PROC_ADDRESS(editor_help_load_xml_from_utf8_chars_and_len, GDExtensionsInterfaceEditorHelpLoadXmlFromUtf8CharsAndLen);
	LOAD_PROC_ADDRESS(image_ptrw, GDExtensionInterfaceImagePtrw);
	LOAD_PROC_ADDRESS(image_ptr, GDExtensionInterfaceImagePtr);
	LOAD_PROC_ADDRESS(register_main_loop_callbacks, GDExtensionInterfaceRegisterMainLoopCallbacks);

	r_initialization->initialize = initialize_level;
	r_initialization->deinitialize = deinitialize_level;
	r_initialization->userdata = p_init_data;
	r_initialization->minimum_initialization_level = p_init_data->minimum_initialization_level;

	Variant::init_bindings();
	godot::internal::register_engine_classes();

	api_initialized = true;
	return true;
}

#undef LOAD_PROC_ADDRESS
#undef ERR_PRINT_EARLY

void GDExtensionBinding::initialize_level(void *p_userdata, GDExtensionInitializationLevel p_level) {
	ERR_FAIL_COND(static_cast<ModuleInitializationLevel>(p_level) >= MODULE_INITIALIZATION_LEVEL_MAX);
	ClassDB::current_level = p_level;

	InitData *init_data = static_cast<InitData *>(p_userdata);
	if (init_data && init_data->init_callback) {
		init_data->init_callback(static_cast<ModuleInitializationLevel>(p_level));
	}

	if (level_initialized[p_level] == 0) {
		ClassDB::initialize(p_level);
	}
	level_initialized[p_level]++;

	if ((ModuleInitializationLevel)p_level == MODULE_INITIALIZATION_LEVEL_CORE && init_data && init_data->has_main_loop_callbacks()) {
		internal::gdextension_interface_register_main_loop_callbacks(internal::library, &init_data->main_loop_callbacks);
	}

	if ((ModuleInitializationLevel)p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
		internal::gdextension_interface_editor_register_get_classes_used_callback(internal::library, &ClassDB::_editor_get_classes_used_callback);

		const internal::DocData &doc_data = internal::get_doc_data();
		if (doc_data.is_valid()) {
			doc_data.load_data();
		}
	}
}

void GDExtensionBinding::deinitialize_level(void *p_userdata, GDExtensionInitializationLevel p_level) {
	ERR_FAIL_COND(static_cast<ModuleInitializationLevel>(p_level) >= MODULE_INITIALIZATION_LEVEL_MAX);
	ClassDB::current_level = p_level;

	InitData *init_data = static_cast<InitData *>(p_userdata);
	if (init_data && init_data->terminate_callback) {
		init_data->terminate_callback(static_cast<ModuleInitializationLevel>(p_level));
	}

	level_initialized[p_level]--;
	if (level_initialized[p_level] == 0) {
		EditorPlugins::deinitialize(p_level);
		ClassDB::deinitialize(p_level);
	}
}

void GDExtensionBinding::InitDataList::add(InitData *p_data) {
	if (data_count == data_capacity) {
		void *new_ptr = realloc(data, sizeof(InitData *) * (data_capacity + 32));
		if (new_ptr) {
			data = (InitData **)(new_ptr);
			data_capacity += 32;
		} else {
			ERR_FAIL_MSG("Unable to allocate memory for extension callbacks.");
		}
	}
	data[data_count++] = p_data;
}

GDExtensionBinding::InitDataList::~InitDataList() {
	for (int i = 0; i < data_count; i++) {
		if (data[i]) {
			delete data[i];
		}
	}
	if (data) {
		free(data);
	}
}

GDExtensionBinding::InitObject::InitObject(GDExtensionInterfaceGetProcAddress p_get_proc_address, GDExtensionClassLibraryPtr p_library, GDExtensionInitialization *r_initialization) {
	get_proc_address = p_get_proc_address;
	library = p_library;
	initialization = r_initialization;
	init_data = new InitData();
	GDExtensionBinding::initdata.add(init_data);
}

void GDExtensionBinding::InitObject::register_initializer(Callback p_init) const {
	init_data->init_callback = p_init;
}

void GDExtensionBinding::InitObject::register_terminator(Callback p_terminate) const {
	init_data->terminate_callback = p_terminate;
}

void GDExtensionBinding::InitObject::set_minimum_library_initialization_level(ModuleInitializationLevel p_level) const {
	init_data->minimum_initialization_level = static_cast<GDExtensionInitializationLevel>(p_level);
}

void GDExtensionBinding::InitObject::register_startup_callback(GDExtensionMainLoopStartupCallback p_callback) const {
	init_data->main_loop_callbacks.startup_func = p_callback;
}

void GDExtensionBinding::InitObject::register_frame_callback(GDExtensionMainLoopFrameCallback p_callback) const {
	init_data->main_loop_callbacks.frame_func = p_callback;
}

void GDExtensionBinding::InitObject::register_shutdown_callback(GDExtensionMainLoopShutdownCallback p_callback) const {
	init_data->main_loop_callbacks.shutdown_func = p_callback;
}

GDExtensionBool GDExtensionBinding::InitObject::init() const {
	return GDExtensionBinding::init(get_proc_address, library, init_data, initialization);
}

void internal::DocData::load_data() const {
	PackedByteArray compressed;
	compressed.resize(compressed_size);
	memcpy(compressed.ptrw(), data, compressed_size);

	// FileAccess::COMPRESSION_DEFLATE = 1
	PackedByteArray decompressed = compressed.decompress(uncompressed_size, 1);

	internal::gdextension_interface_editor_help_load_xml_from_utf8_chars_and_len(reinterpret_cast<const char *>(decompressed.ptr()), uncompressed_size);
}

} // namespace godot
