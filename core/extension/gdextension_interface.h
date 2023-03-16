/**************************************************************************/
/*  gdextension_interface.h                                               */
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

#ifndef GDEXTENSION_INTERFACE_H
#define GDEXTENSION_INTERFACE_H

/* This is a C class header, you can copy it and use it directly in your own binders.
 * Together with the JSON file, you should be able to generate any binder.
 */

#include <stddef.h>
#include <stdint.h>

#ifndef __cplusplus
typedef uint32_t char32_t;
typedef uint16_t char16_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* VARIANT TYPES */

typedef enum {
	GDEXTENSION_VARIANT_TYPE_NIL,

	/*  atomic types */
	GDEXTENSION_VARIANT_TYPE_BOOL,
	GDEXTENSION_VARIANT_TYPE_INT,
	GDEXTENSION_VARIANT_TYPE_FLOAT,
	GDEXTENSION_VARIANT_TYPE_STRING,

	/* math types */
	GDEXTENSION_VARIANT_TYPE_VECTOR2,
	GDEXTENSION_VARIANT_TYPE_VECTOR2I,
	GDEXTENSION_VARIANT_TYPE_RECT2,
	GDEXTENSION_VARIANT_TYPE_RECT2I,
	GDEXTENSION_VARIANT_TYPE_VECTOR3,
	GDEXTENSION_VARIANT_TYPE_VECTOR3I,
	GDEXTENSION_VARIANT_TYPE_TRANSFORM2D,
	GDEXTENSION_VARIANT_TYPE_VECTOR4,
	GDEXTENSION_VARIANT_TYPE_VECTOR4I,
	GDEXTENSION_VARIANT_TYPE_PLANE,
	GDEXTENSION_VARIANT_TYPE_QUATERNION,
	GDEXTENSION_VARIANT_TYPE_AABB,
	GDEXTENSION_VARIANT_TYPE_BASIS,
	GDEXTENSION_VARIANT_TYPE_TRANSFORM3D,
	GDEXTENSION_VARIANT_TYPE_PROJECTION,

	/* misc types */
	GDEXTENSION_VARIANT_TYPE_COLOR,
	GDEXTENSION_VARIANT_TYPE_STRING_NAME,
	GDEXTENSION_VARIANT_TYPE_NODE_PATH,
	GDEXTENSION_VARIANT_TYPE_RID,
	GDEXTENSION_VARIANT_TYPE_OBJECT,
	GDEXTENSION_VARIANT_TYPE_CALLABLE,
	GDEXTENSION_VARIANT_TYPE_SIGNAL,
	GDEXTENSION_VARIANT_TYPE_DICTIONARY,
	GDEXTENSION_VARIANT_TYPE_ARRAY,

	/* typed arrays */
	GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY,
	GDEXTENSION_VARIANT_TYPE_PACKED_INT32_ARRAY,
	GDEXTENSION_VARIANT_TYPE_PACKED_INT64_ARRAY,
	GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY,
	GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT64_ARRAY,
	GDEXTENSION_VARIANT_TYPE_PACKED_STRING_ARRAY,
	GDEXTENSION_VARIANT_TYPE_PACKED_VECTOR2_ARRAY,
	GDEXTENSION_VARIANT_TYPE_PACKED_VECTOR3_ARRAY,
	GDEXTENSION_VARIANT_TYPE_PACKED_COLOR_ARRAY,

	GDEXTENSION_VARIANT_TYPE_VARIANT_MAX
} GDExtensionVariantType;

typedef enum {
	/* comparison */
	GDEXTENSION_VARIANT_OP_EQUAL,
	GDEXTENSION_VARIANT_OP_NOT_EQUAL,
	GDEXTENSION_VARIANT_OP_LESS,
	GDEXTENSION_VARIANT_OP_LESS_EQUAL,
	GDEXTENSION_VARIANT_OP_GREATER,
	GDEXTENSION_VARIANT_OP_GREATER_EQUAL,

	/* mathematic */
	GDEXTENSION_VARIANT_OP_ADD,
	GDEXTENSION_VARIANT_OP_SUBTRACT,
	GDEXTENSION_VARIANT_OP_MULTIPLY,
	GDEXTENSION_VARIANT_OP_DIVIDE,
	GDEXTENSION_VARIANT_OP_NEGATE,
	GDEXTENSION_VARIANT_OP_POSITIVE,
	GDEXTENSION_VARIANT_OP_MODULE,
	GDEXTENSION_VARIANT_OP_POWER,

	/* bitwise */
	GDEXTENSION_VARIANT_OP_SHIFT_LEFT,
	GDEXTENSION_VARIANT_OP_SHIFT_RIGHT,
	GDEXTENSION_VARIANT_OP_BIT_AND,
	GDEXTENSION_VARIANT_OP_BIT_OR,
	GDEXTENSION_VARIANT_OP_BIT_XOR,
	GDEXTENSION_VARIANT_OP_BIT_NEGATE,

	/* logic */
	GDEXTENSION_VARIANT_OP_AND,
	GDEXTENSION_VARIANT_OP_OR,
	GDEXTENSION_VARIANT_OP_XOR,
	GDEXTENSION_VARIANT_OP_NOT,

	/* containment */
	GDEXTENSION_VARIANT_OP_IN,
	GDEXTENSION_VARIANT_OP_MAX

} GDExtensionVariantOperator;

typedef void *GDExtensionVariantPtr;
typedef const void *GDExtensionConstVariantPtr;
typedef void *GDExtensionStringNamePtr;
typedef const void *GDExtensionConstStringNamePtr;
typedef void *GDExtensionStringPtr;
typedef const void *GDExtensionConstStringPtr;
typedef void *GDExtensionObjectPtr;
typedef const void *GDExtensionConstObjectPtr;
typedef void *GDExtensionTypePtr;
typedef const void *GDExtensionConstTypePtr;
typedef const void *GDExtensionMethodBindPtr;
typedef int64_t GDExtensionInt;
typedef uint8_t GDExtensionBool;
typedef uint64_t GDObjectInstanceID;
typedef void *GDExtensionRefPtr;
typedef const void *GDExtensionConstRefPtr;

/* VARIANT DATA I/O */

typedef enum {
	GDEXTENSION_CALL_OK,
	GDEXTENSION_CALL_ERROR_INVALID_METHOD,
	GDEXTENSION_CALL_ERROR_INVALID_ARGUMENT, // Expected a different variant type.
	GDEXTENSION_CALL_ERROR_TOO_MANY_ARGUMENTS, // Expected lower number of arguments.
	GDEXTENSION_CALL_ERROR_TOO_FEW_ARGUMENTS, // Expected higher number of arguments.
	GDEXTENSION_CALL_ERROR_INSTANCE_IS_NULL,
	GDEXTENSION_CALL_ERROR_METHOD_NOT_CONST, // Used for const call.
} GDExtensionCallErrorType;

typedef struct {
	GDExtensionCallErrorType error;
	int32_t argument;
	int32_t expected;
} GDExtensionCallError;

typedef void (*GDExtensionVariantFromTypeConstructorFunc)(GDExtensionVariantPtr, GDExtensionTypePtr);
typedef void (*GDExtensionTypeFromVariantConstructorFunc)(GDExtensionTypePtr, GDExtensionVariantPtr);
typedef void (*GDExtensionPtrOperatorEvaluator)(GDExtensionConstTypePtr p_left, GDExtensionConstTypePtr p_right, GDExtensionTypePtr r_result);
typedef void (*GDExtensionPtrBuiltInMethod)(GDExtensionTypePtr p_base, const GDExtensionConstTypePtr *p_args, GDExtensionTypePtr r_return, int p_argument_count);
typedef void (*GDExtensionPtrConstructor)(GDExtensionTypePtr p_base, const GDExtensionConstTypePtr *p_args);
typedef void (*GDExtensionPtrDestructor)(GDExtensionTypePtr p_base);
typedef void (*GDExtensionPtrSetter)(GDExtensionTypePtr p_base, GDExtensionConstTypePtr p_value);
typedef void (*GDExtensionPtrGetter)(GDExtensionConstTypePtr p_base, GDExtensionTypePtr r_value);
typedef void (*GDExtensionPtrIndexedSetter)(GDExtensionTypePtr p_base, GDExtensionInt p_index, GDExtensionConstTypePtr p_value);
typedef void (*GDExtensionPtrIndexedGetter)(GDExtensionConstTypePtr p_base, GDExtensionInt p_index, GDExtensionTypePtr r_value);
typedef void (*GDExtensionPtrKeyedSetter)(GDExtensionTypePtr p_base, GDExtensionConstTypePtr p_key, GDExtensionConstTypePtr p_value);
typedef void (*GDExtensionPtrKeyedGetter)(GDExtensionConstTypePtr p_base, GDExtensionConstTypePtr p_key, GDExtensionTypePtr r_value);
typedef uint32_t (*GDExtensionPtrKeyedChecker)(GDExtensionConstVariantPtr p_base, GDExtensionConstVariantPtr p_key);
typedef void (*GDExtensionPtrUtilityFunction)(GDExtensionTypePtr r_return, const GDExtensionConstTypePtr *p_args, int p_argument_count);

typedef GDExtensionObjectPtr (*GDExtensionClassConstructor)();

typedef void *(*GDExtensionInstanceBindingCreateCallback)(void *p_token, void *p_instance);
typedef void (*GDExtensionInstanceBindingFreeCallback)(void *p_token, void *p_instance, void *p_binding);
typedef GDExtensionBool (*GDExtensionInstanceBindingReferenceCallback)(void *p_token, void *p_binding, GDExtensionBool p_reference);

typedef struct {
	GDExtensionInstanceBindingCreateCallback create_callback;
	GDExtensionInstanceBindingFreeCallback free_callback;
	GDExtensionInstanceBindingReferenceCallback reference_callback;
} GDExtensionInstanceBindingCallbacks;

/* EXTENSION CLASSES */

typedef void *GDExtensionClassInstancePtr;

typedef GDExtensionBool (*GDExtensionClassSet)(GDExtensionClassInstancePtr p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionConstVariantPtr p_value);
typedef GDExtensionBool (*GDExtensionClassGet)(GDExtensionClassInstancePtr p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionVariantPtr r_ret);
typedef uint64_t (*GDExtensionClassGetRID)(GDExtensionClassInstancePtr p_instance);

typedef struct {
	GDExtensionVariantType type;
	GDExtensionStringNamePtr name;
	GDExtensionStringNamePtr class_name;
	uint32_t hint; // Bitfield of `PropertyHint` (defined in `extension_api.json`).
	GDExtensionStringPtr hint_string;
	uint32_t usage; // Bitfield of `PropertyUsageFlags` (defined in `extension_api.json`).
} GDExtensionPropertyInfo;

typedef struct {
	GDExtensionStringNamePtr name;
	GDExtensionPropertyInfo return_value;
	uint32_t flags; // Bitfield of `GDExtensionClassMethodFlags`.
	int32_t id;

	/* Arguments: `default_arguments` is an array of size `argument_count`. */
	uint32_t argument_count;
	GDExtensionPropertyInfo *arguments;

	/* Default arguments: `default_arguments` is an array of size `default_argument_count`. */
	uint32_t default_argument_count;
	GDExtensionVariantPtr *default_arguments;
} GDExtensionMethodInfo;

typedef const GDExtensionPropertyInfo *(*GDExtensionClassGetPropertyList)(GDExtensionClassInstancePtr p_instance, uint32_t *r_count);
typedef void (*GDExtensionClassFreePropertyList)(GDExtensionClassInstancePtr p_instance, const GDExtensionPropertyInfo *p_list);
typedef GDExtensionBool (*GDExtensionClassPropertyCanRevert)(GDExtensionClassInstancePtr p_instance, GDExtensionConstStringNamePtr p_name);
typedef GDExtensionBool (*GDExtensionClassPropertyGetRevert)(GDExtensionClassInstancePtr p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionVariantPtr r_ret);
typedef void (*GDExtensionClassNotification)(GDExtensionClassInstancePtr p_instance, int32_t p_what);
typedef void (*GDExtensionClassToString)(GDExtensionClassInstancePtr p_instance, GDExtensionBool *r_is_valid, GDExtensionStringPtr p_out);
typedef void (*GDExtensionClassReference)(GDExtensionClassInstancePtr p_instance);
typedef void (*GDExtensionClassUnreference)(GDExtensionClassInstancePtr p_instance);
typedef void (*GDExtensionClassCallVirtual)(GDExtensionClassInstancePtr p_instance, const GDExtensionConstTypePtr *p_args, GDExtensionTypePtr r_ret);
typedef GDExtensionObjectPtr (*GDExtensionClassCreateInstance)(void *p_userdata);
typedef void (*GDExtensionClassFreeInstance)(void *p_userdata, GDExtensionClassInstancePtr p_instance);
typedef GDExtensionClassCallVirtual (*GDExtensionClassGetVirtual)(void *p_userdata, GDExtensionConstStringNamePtr p_name);

typedef struct {
	GDExtensionBool is_virtual;
	GDExtensionBool is_abstract;
	GDExtensionClassSet set_func;
	GDExtensionClassGet get_func;
	GDExtensionClassGetPropertyList get_property_list_func;
	GDExtensionClassFreePropertyList free_property_list_func;
	GDExtensionClassPropertyCanRevert property_can_revert_func;
	GDExtensionClassPropertyGetRevert property_get_revert_func;
	GDExtensionClassNotification notification_func;
	GDExtensionClassToString to_string_func;
	GDExtensionClassReference reference_func;
	GDExtensionClassUnreference unreference_func;
	GDExtensionClassCreateInstance create_instance_func; // (Default) constructor; mandatory. If the class is not instantiable, consider making it virtual or abstract.
	GDExtensionClassFreeInstance free_instance_func; // Destructor; mandatory.
	GDExtensionClassGetVirtual get_virtual_func; // Queries a virtual function by name and returns a callback to invoke the requested virtual function.
	GDExtensionClassGetRID get_rid_func;
	void *class_userdata; // Per-class user data, later accessible in instance bindings.
} GDExtensionClassCreationInfo;

typedef void *GDExtensionClassLibraryPtr;

/* Method */

typedef enum {
	GDEXTENSION_METHOD_FLAG_NORMAL = 1,
	GDEXTENSION_METHOD_FLAG_EDITOR = 2,
	GDEXTENSION_METHOD_FLAG_CONST = 4,
	GDEXTENSION_METHOD_FLAG_VIRTUAL = 8,
	GDEXTENSION_METHOD_FLAG_VARARG = 16,
	GDEXTENSION_METHOD_FLAG_STATIC = 32,
	GDEXTENSION_METHOD_FLAGS_DEFAULT = GDEXTENSION_METHOD_FLAG_NORMAL,
} GDExtensionClassMethodFlags;

typedef enum {
	GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE,
	GDEXTENSION_METHOD_ARGUMENT_METADATA_INT_IS_INT8,
	GDEXTENSION_METHOD_ARGUMENT_METADATA_INT_IS_INT16,
	GDEXTENSION_METHOD_ARGUMENT_METADATA_INT_IS_INT32,
	GDEXTENSION_METHOD_ARGUMENT_METADATA_INT_IS_INT64,
	GDEXTENSION_METHOD_ARGUMENT_METADATA_INT_IS_UINT8,
	GDEXTENSION_METHOD_ARGUMENT_METADATA_INT_IS_UINT16,
	GDEXTENSION_METHOD_ARGUMENT_METADATA_INT_IS_UINT32,
	GDEXTENSION_METHOD_ARGUMENT_METADATA_INT_IS_UINT64,
	GDEXTENSION_METHOD_ARGUMENT_METADATA_REAL_IS_FLOAT,
	GDEXTENSION_METHOD_ARGUMENT_METADATA_REAL_IS_DOUBLE
} GDExtensionClassMethodArgumentMetadata;

typedef void (*GDExtensionClassMethodCall)(void *method_userdata, GDExtensionClassInstancePtr p_instance, const GDExtensionConstVariantPtr *p_args, GDExtensionInt p_argument_count, GDExtensionVariantPtr r_return, GDExtensionCallError *r_error);
typedef void (*GDExtensionClassMethodPtrCall)(void *method_userdata, GDExtensionClassInstancePtr p_instance, const GDExtensionConstTypePtr *p_args, GDExtensionTypePtr r_ret);

typedef struct {
	GDExtensionStringNamePtr name;
	void *method_userdata;
	GDExtensionClassMethodCall call_func;
	GDExtensionClassMethodPtrCall ptrcall_func;
	uint32_t method_flags; // Bitfield of `GDExtensionClassMethodFlags`.

	/* If `has_return_value` is false, `return_value_info` and `return_value_metadata` are ignored. */
	GDExtensionBool has_return_value;
	GDExtensionPropertyInfo *return_value_info;
	GDExtensionClassMethodArgumentMetadata return_value_metadata;

	/* Arguments: `arguments_info` and `arguments_metadata` are array of size `argument_count`.
	 * Name and hint information for the argument can be omitted in release builds. Class name should always be present if it applies.
	 */
	uint32_t argument_count;
	GDExtensionPropertyInfo *arguments_info;
	GDExtensionClassMethodArgumentMetadata *arguments_metadata;

	/* Default arguments: `default_arguments` is an array of size `default_argument_count`. */
	uint32_t default_argument_count;
	GDExtensionVariantPtr *default_arguments;
} GDExtensionClassMethodInfo;

/* SCRIPT INSTANCE EXTENSION */

typedef void *GDExtensionScriptInstanceDataPtr; // Pointer to custom ScriptInstance native implementation.

typedef GDExtensionBool (*GDExtensionScriptInstanceSet)(GDExtensionScriptInstanceDataPtr p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionConstVariantPtr p_value);
typedef GDExtensionBool (*GDExtensionScriptInstanceGet)(GDExtensionScriptInstanceDataPtr p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionVariantPtr r_ret);
typedef const GDExtensionPropertyInfo *(*GDExtensionScriptInstanceGetPropertyList)(GDExtensionScriptInstanceDataPtr p_instance, uint32_t *r_count);
typedef void (*GDExtensionScriptInstanceFreePropertyList)(GDExtensionScriptInstanceDataPtr p_instance, const GDExtensionPropertyInfo *p_list);
typedef GDExtensionVariantType (*GDExtensionScriptInstanceGetPropertyType)(GDExtensionScriptInstanceDataPtr p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionBool *r_is_valid);

typedef GDExtensionBool (*GDExtensionScriptInstancePropertyCanRevert)(GDExtensionScriptInstanceDataPtr p_instance, GDExtensionConstStringNamePtr p_name);
typedef GDExtensionBool (*GDExtensionScriptInstancePropertyGetRevert)(GDExtensionScriptInstanceDataPtr p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionVariantPtr r_ret);

typedef GDExtensionObjectPtr (*GDExtensionScriptInstanceGetOwner)(GDExtensionScriptInstanceDataPtr p_instance);
typedef void (*GDExtensionScriptInstancePropertyStateAdd)(GDExtensionConstStringNamePtr p_name, GDExtensionConstVariantPtr p_value, void *p_userdata);
typedef void (*GDExtensionScriptInstanceGetPropertyState)(GDExtensionScriptInstanceDataPtr p_instance, GDExtensionScriptInstancePropertyStateAdd p_add_func, void *p_userdata);

typedef const GDExtensionMethodInfo *(*GDExtensionScriptInstanceGetMethodList)(GDExtensionScriptInstanceDataPtr p_instance, uint32_t *r_count);
typedef void (*GDExtensionScriptInstanceFreeMethodList)(GDExtensionScriptInstanceDataPtr p_instance, const GDExtensionMethodInfo *p_list);

typedef GDExtensionBool (*GDExtensionScriptInstanceHasMethod)(GDExtensionScriptInstanceDataPtr p_instance, GDExtensionConstStringNamePtr p_name);

typedef void (*GDExtensionScriptInstanceCall)(GDExtensionScriptInstanceDataPtr p_self, GDExtensionConstStringNamePtr p_method, const GDExtensionConstVariantPtr *p_args, GDExtensionInt p_argument_count, GDExtensionVariantPtr r_return, GDExtensionCallError *r_error);
typedef void (*GDExtensionScriptInstanceNotification)(GDExtensionScriptInstanceDataPtr p_instance, int32_t p_what);
typedef void (*GDExtensionScriptInstanceToString)(GDExtensionScriptInstanceDataPtr p_instance, GDExtensionBool *r_is_valid, GDExtensionStringPtr r_out);

typedef void (*GDExtensionScriptInstanceRefCountIncremented)(GDExtensionScriptInstanceDataPtr p_instance);
typedef GDExtensionBool (*GDExtensionScriptInstanceRefCountDecremented)(GDExtensionScriptInstanceDataPtr p_instance);

typedef GDExtensionObjectPtr (*GDExtensionScriptInstanceGetScript)(GDExtensionScriptInstanceDataPtr p_instance);
typedef GDExtensionBool (*GDExtensionScriptInstanceIsPlaceholder)(GDExtensionScriptInstanceDataPtr p_instance);

typedef void *GDExtensionScriptLanguagePtr;

typedef GDExtensionScriptLanguagePtr (*GDExtensionScriptInstanceGetLanguage)(GDExtensionScriptInstanceDataPtr p_instance);

typedef void (*GDExtensionScriptInstanceFree)(GDExtensionScriptInstanceDataPtr p_instance);

typedef void *GDExtensionScriptInstancePtr; // Pointer to ScriptInstance.

typedef struct {
	GDExtensionScriptInstanceSet set_func;
	GDExtensionScriptInstanceGet get_func;
	GDExtensionScriptInstanceGetPropertyList get_property_list_func;
	GDExtensionScriptInstanceFreePropertyList free_property_list_func;

	GDExtensionScriptInstancePropertyCanRevert property_can_revert_func;
	GDExtensionScriptInstancePropertyGetRevert property_get_revert_func;

	GDExtensionScriptInstanceGetOwner get_owner_func;
	GDExtensionScriptInstanceGetPropertyState get_property_state_func;

	GDExtensionScriptInstanceGetMethodList get_method_list_func;
	GDExtensionScriptInstanceFreeMethodList free_method_list_func;
	GDExtensionScriptInstanceGetPropertyType get_property_type_func;

	GDExtensionScriptInstanceHasMethod has_method_func;

	GDExtensionScriptInstanceCall call_func;
	GDExtensionScriptInstanceNotification notification_func;

	GDExtensionScriptInstanceToString to_string_func;

	GDExtensionScriptInstanceRefCountIncremented refcount_incremented_func;
	GDExtensionScriptInstanceRefCountDecremented refcount_decremented_func;

	GDExtensionScriptInstanceGetScript get_script_func;

	GDExtensionScriptInstanceIsPlaceholder is_placeholder_func;

	GDExtensionScriptInstanceSet set_fallback_func;
	GDExtensionScriptInstanceGet get_fallback_func;

	GDExtensionScriptInstanceGetLanguage get_language_func;

	GDExtensionScriptInstanceFree free_func;

} GDExtensionScriptInstanceInfo;

/* INTERFACE */

typedef struct {
	uint32_t version_major;
	uint32_t version_minor;
	uint32_t version_patch;
	const char *version_string;

	/* GODOT CORE */

	void *(*mem_alloc)(size_t p_bytes);
	void *(*mem_realloc)(void *p_ptr, size_t p_bytes);
	void (*mem_free)(void *p_ptr);

	void (*print_error)(const char *p_description, const char *p_function, const char *p_file, int32_t p_line, GDExtensionBool p_editor_notify);
	void (*print_error_with_message)(const char *p_description, const char *p_message, const char *p_function, const char *p_file, int32_t p_line, GDExtensionBool p_editor_notify);
	void (*print_warning)(const char *p_description, const char *p_function, const char *p_file, int32_t p_line, GDExtensionBool p_editor_notify);
	void (*print_warning_with_message)(const char *p_description, const char *p_message, const char *p_function, const char *p_file, int32_t p_line, GDExtensionBool p_editor_notify);
	void (*print_script_error)(const char *p_description, const char *p_function, const char *p_file, int32_t p_line, GDExtensionBool p_editor_notify);
	void (*print_script_error_with_message)(const char *p_description, const char *p_message, const char *p_function, const char *p_file, int32_t p_line, GDExtensionBool p_editor_notify);

	uint64_t (*get_native_struct_size)(GDExtensionConstStringNamePtr p_name);

	/* GODOT VARIANT */

	/* variant general */
	void (*variant_new_copy)(GDExtensionVariantPtr r_dest, GDExtensionConstVariantPtr p_src);
	void (*variant_new_nil)(GDExtensionVariantPtr r_dest);
	void (*variant_destroy)(GDExtensionVariantPtr p_self);

	/* variant type */
	void (*variant_call)(GDExtensionVariantPtr p_self, GDExtensionConstStringNamePtr p_method, const GDExtensionConstVariantPtr *p_args, GDExtensionInt p_argument_count, GDExtensionVariantPtr r_return, GDExtensionCallError *r_error);
	void (*variant_call_static)(GDExtensionVariantType p_type, GDExtensionConstStringNamePtr p_method, const GDExtensionConstVariantPtr *p_args, GDExtensionInt p_argument_count, GDExtensionVariantPtr r_return, GDExtensionCallError *r_error);
	void (*variant_evaluate)(GDExtensionVariantOperator p_op, GDExtensionConstVariantPtr p_a, GDExtensionConstVariantPtr p_b, GDExtensionVariantPtr r_return, GDExtensionBool *r_valid);
	void (*variant_set)(GDExtensionVariantPtr p_self, GDExtensionConstVariantPtr p_key, GDExtensionConstVariantPtr p_value, GDExtensionBool *r_valid);
	void (*variant_set_named)(GDExtensionVariantPtr p_self, GDExtensionConstStringNamePtr p_key, GDExtensionConstVariantPtr p_value, GDExtensionBool *r_valid);
	void (*variant_set_keyed)(GDExtensionVariantPtr p_self, GDExtensionConstVariantPtr p_key, GDExtensionConstVariantPtr p_value, GDExtensionBool *r_valid);
	void (*variant_set_indexed)(GDExtensionVariantPtr p_self, GDExtensionInt p_index, GDExtensionConstVariantPtr p_value, GDExtensionBool *r_valid, GDExtensionBool *r_oob);
	void (*variant_get)(GDExtensionConstVariantPtr p_self, GDExtensionConstVariantPtr p_key, GDExtensionVariantPtr r_ret, GDExtensionBool *r_valid);
	void (*variant_get_named)(GDExtensionConstVariantPtr p_self, GDExtensionConstStringNamePtr p_key, GDExtensionVariantPtr r_ret, GDExtensionBool *r_valid);
	void (*variant_get_keyed)(GDExtensionConstVariantPtr p_self, GDExtensionConstVariantPtr p_key, GDExtensionVariantPtr r_ret, GDExtensionBool *r_valid);
	void (*variant_get_indexed)(GDExtensionConstVariantPtr p_self, GDExtensionInt p_index, GDExtensionVariantPtr r_ret, GDExtensionBool *r_valid, GDExtensionBool *r_oob);
	GDExtensionBool (*variant_iter_init)(GDExtensionConstVariantPtr p_self, GDExtensionVariantPtr r_iter, GDExtensionBool *r_valid);
	GDExtensionBool (*variant_iter_next)(GDExtensionConstVariantPtr p_self, GDExtensionVariantPtr r_iter, GDExtensionBool *r_valid);
	void (*variant_iter_get)(GDExtensionConstVariantPtr p_self, GDExtensionVariantPtr r_iter, GDExtensionVariantPtr r_ret, GDExtensionBool *r_valid);
	GDExtensionInt (*variant_hash)(GDExtensionConstVariantPtr p_self);
	GDExtensionInt (*variant_recursive_hash)(GDExtensionConstVariantPtr p_self, GDExtensionInt p_recursion_count);
	GDExtensionBool (*variant_hash_compare)(GDExtensionConstVariantPtr p_self, GDExtensionConstVariantPtr p_other);
	GDExtensionBool (*variant_booleanize)(GDExtensionConstVariantPtr p_self);
	void (*variant_duplicate)(GDExtensionConstVariantPtr p_self, GDExtensionVariantPtr r_ret, GDExtensionBool p_deep);
	void (*variant_stringify)(GDExtensionConstVariantPtr p_self, GDExtensionStringPtr r_ret);

	GDExtensionVariantType (*variant_get_type)(GDExtensionConstVariantPtr p_self);
	GDExtensionBool (*variant_has_method)(GDExtensionConstVariantPtr p_self, GDExtensionConstStringNamePtr p_method);
	GDExtensionBool (*variant_has_member)(GDExtensionVariantType p_type, GDExtensionConstStringNamePtr p_member);
	GDExtensionBool (*variant_has_key)(GDExtensionConstVariantPtr p_self, GDExtensionConstVariantPtr p_key, GDExtensionBool *r_valid);
	void (*variant_get_type_name)(GDExtensionVariantType p_type, GDExtensionStringPtr r_name);
	GDExtensionBool (*variant_can_convert)(GDExtensionVariantType p_from, GDExtensionVariantType p_to);
	GDExtensionBool (*variant_can_convert_strict)(GDExtensionVariantType p_from, GDExtensionVariantType p_to);

	/* ptrcalls */
	GDExtensionVariantFromTypeConstructorFunc (*get_variant_from_type_constructor)(GDExtensionVariantType p_type);
	GDExtensionTypeFromVariantConstructorFunc (*get_variant_to_type_constructor)(GDExtensionVariantType p_type);
	GDExtensionPtrOperatorEvaluator (*variant_get_ptr_operator_evaluator)(GDExtensionVariantOperator p_operator, GDExtensionVariantType p_type_a, GDExtensionVariantType p_type_b);
	GDExtensionPtrBuiltInMethod (*variant_get_ptr_builtin_method)(GDExtensionVariantType p_type, GDExtensionConstStringNamePtr p_method, GDExtensionInt p_hash);
	GDExtensionPtrConstructor (*variant_get_ptr_constructor)(GDExtensionVariantType p_type, int32_t p_constructor);
	GDExtensionPtrDestructor (*variant_get_ptr_destructor)(GDExtensionVariantType p_type);
	void (*variant_construct)(GDExtensionVariantType p_type, GDExtensionVariantPtr p_base, const GDExtensionConstVariantPtr *p_args, int32_t p_argument_count, GDExtensionCallError *r_error);
	GDExtensionPtrSetter (*variant_get_ptr_setter)(GDExtensionVariantType p_type, GDExtensionConstStringNamePtr p_member);
	GDExtensionPtrGetter (*variant_get_ptr_getter)(GDExtensionVariantType p_type, GDExtensionConstStringNamePtr p_member);
	GDExtensionPtrIndexedSetter (*variant_get_ptr_indexed_setter)(GDExtensionVariantType p_type);
	GDExtensionPtrIndexedGetter (*variant_get_ptr_indexed_getter)(GDExtensionVariantType p_type);
	GDExtensionPtrKeyedSetter (*variant_get_ptr_keyed_setter)(GDExtensionVariantType p_type);
	GDExtensionPtrKeyedGetter (*variant_get_ptr_keyed_getter)(GDExtensionVariantType p_type);
	GDExtensionPtrKeyedChecker (*variant_get_ptr_keyed_checker)(GDExtensionVariantType p_type);
	void (*variant_get_constant_value)(GDExtensionVariantType p_type, GDExtensionConstStringNamePtr p_constant, GDExtensionVariantPtr r_ret);
	GDExtensionPtrUtilityFunction (*variant_get_ptr_utility_function)(GDExtensionConstStringNamePtr p_function, GDExtensionInt p_hash);

	/*  extra utilities */
	void (*string_new_with_latin1_chars)(GDExtensionStringPtr r_dest, const char *p_contents);
	void (*string_new_with_utf8_chars)(GDExtensionStringPtr r_dest, const char *p_contents);
	void (*string_new_with_utf16_chars)(GDExtensionStringPtr r_dest, const char16_t *p_contents);
	void (*string_new_with_utf32_chars)(GDExtensionStringPtr r_dest, const char32_t *p_contents);
	void (*string_new_with_wide_chars)(GDExtensionStringPtr r_dest, const wchar_t *p_contents);
	void (*string_new_with_latin1_chars_and_len)(GDExtensionStringPtr r_dest, const char *p_contents, GDExtensionInt p_size);
	void (*string_new_with_utf8_chars_and_len)(GDExtensionStringPtr r_dest, const char *p_contents, GDExtensionInt p_size);
	void (*string_new_with_utf16_chars_and_len)(GDExtensionStringPtr r_dest, const char16_t *p_contents, GDExtensionInt p_size);
	void (*string_new_with_utf32_chars_and_len)(GDExtensionStringPtr r_dest, const char32_t *p_contents, GDExtensionInt p_size);
	void (*string_new_with_wide_chars_and_len)(GDExtensionStringPtr r_dest, const wchar_t *p_contents, GDExtensionInt p_size);
	/* Information about the following functions:
	 * - The return value is the resulting encoded string length.
	 * - The length returned is in characters, not in bytes. It also does not include a trailing zero.
	 * - These functions also do not write trailing zero, If you need it, write it yourself at the position indicated by the length (and make sure to allocate it).
	 * - Passing NULL in r_text means only the length is computed (again, without including trailing zero).
	 * - p_max_write_length argument is in characters, not bytes. It will be ignored if r_text is NULL.
	 * - p_max_write_length argument does not affect the return value, it's only to cap write length.
	 */
	GDExtensionInt (*string_to_latin1_chars)(GDExtensionConstStringPtr p_self, char *r_text, GDExtensionInt p_max_write_length);
	GDExtensionInt (*string_to_utf8_chars)(GDExtensionConstStringPtr p_self, char *r_text, GDExtensionInt p_max_write_length);
	GDExtensionInt (*string_to_utf16_chars)(GDExtensionConstStringPtr p_self, char16_t *r_text, GDExtensionInt p_max_write_length);
	GDExtensionInt (*string_to_utf32_chars)(GDExtensionConstStringPtr p_self, char32_t *r_text, GDExtensionInt p_max_write_length);
	GDExtensionInt (*string_to_wide_chars)(GDExtensionConstStringPtr p_self, wchar_t *r_text, GDExtensionInt p_max_write_length);
	char32_t *(*string_operator_index)(GDExtensionStringPtr p_self, GDExtensionInt p_index);
	const char32_t *(*string_operator_index_const)(GDExtensionConstStringPtr p_self, GDExtensionInt p_index);

	void (*string_operator_plus_eq_string)(GDExtensionStringPtr p_self, GDExtensionConstStringPtr p_b);
	void (*string_operator_plus_eq_char)(GDExtensionStringPtr p_self, char32_t p_b);
	void (*string_operator_plus_eq_cstr)(GDExtensionStringPtr p_self, const char *p_b);
	void (*string_operator_plus_eq_wcstr)(GDExtensionStringPtr p_self, const wchar_t *p_b);
	void (*string_operator_plus_eq_c32str)(GDExtensionStringPtr p_self, const char32_t *p_b);

	/*  XMLParser extra utilities */

	GDExtensionInt (*xml_parser_open_buffer)(GDExtensionObjectPtr p_instance, const uint8_t *p_buffer, size_t p_size);

	/*  FileAccess extra utilities */

	void (*file_access_store_buffer)(GDExtensionObjectPtr p_instance, const uint8_t *p_src, uint64_t p_length);
	uint64_t (*file_access_get_buffer)(GDExtensionConstObjectPtr p_instance, uint8_t *p_dst, uint64_t p_length);

	/*  WorkerThreadPool extra utilities */

	int64_t (*worker_thread_pool_add_native_group_task)(GDExtensionObjectPtr p_instance, void (*p_func)(void *, uint32_t), void *p_userdata, int p_elements, int p_tasks, GDExtensionBool p_high_priority, GDExtensionConstStringPtr p_description);
	int64_t (*worker_thread_pool_add_native_task)(GDExtensionObjectPtr p_instance, void (*p_func)(void *), void *p_userdata, GDExtensionBool p_high_priority, GDExtensionConstStringPtr p_description);

	/* Packed array functions */

	uint8_t *(*packed_byte_array_operator_index)(GDExtensionTypePtr p_self, GDExtensionInt p_index); // p_self should be a PackedByteArray
	const uint8_t *(*packed_byte_array_operator_index_const)(GDExtensionConstTypePtr p_self, GDExtensionInt p_index); // p_self should be a PackedByteArray

	GDExtensionTypePtr (*packed_color_array_operator_index)(GDExtensionTypePtr p_self, GDExtensionInt p_index); // p_self should be a PackedColorArray, returns Color ptr
	GDExtensionTypePtr (*packed_color_array_operator_index_const)(GDExtensionConstTypePtr p_self, GDExtensionInt p_index); // p_self should be a PackedColorArray, returns Color ptr

	float *(*packed_float32_array_operator_index)(GDExtensionTypePtr p_self, GDExtensionInt p_index); // p_self should be a PackedFloat32Array
	const float *(*packed_float32_array_operator_index_const)(GDExtensionConstTypePtr p_self, GDExtensionInt p_index); // p_self should be a PackedFloat32Array
	double *(*packed_float64_array_operator_index)(GDExtensionTypePtr p_self, GDExtensionInt p_index); // p_self should be a PackedFloat64Array
	const double *(*packed_float64_array_operator_index_const)(GDExtensionConstTypePtr p_self, GDExtensionInt p_index); // p_self should be a PackedFloat64Array

	int32_t *(*packed_int32_array_operator_index)(GDExtensionTypePtr p_self, GDExtensionInt p_index); // p_self should be a PackedInt32Array
	const int32_t *(*packed_int32_array_operator_index_const)(GDExtensionConstTypePtr p_self, GDExtensionInt p_index); // p_self should be a PackedInt32Array
	int64_t *(*packed_int64_array_operator_index)(GDExtensionTypePtr p_self, GDExtensionInt p_index); // p_self should be a PackedInt32Array
	const int64_t *(*packed_int64_array_operator_index_const)(GDExtensionConstTypePtr p_self, GDExtensionInt p_index); // p_self should be a PackedInt32Array

	GDExtensionStringPtr (*packed_string_array_operator_index)(GDExtensionTypePtr p_self, GDExtensionInt p_index); // p_self should be a PackedStringArray
	GDExtensionStringPtr (*packed_string_array_operator_index_const)(GDExtensionConstTypePtr p_self, GDExtensionInt p_index); // p_self should be a PackedStringArray

	GDExtensionTypePtr (*packed_vector2_array_operator_index)(GDExtensionTypePtr p_self, GDExtensionInt p_index); // p_self should be a PackedVector2Array, returns Vector2 ptr
	GDExtensionTypePtr (*packed_vector2_array_operator_index_const)(GDExtensionConstTypePtr p_self, GDExtensionInt p_index); // p_self should be a PackedVector2Array, returns Vector2 ptr
	GDExtensionTypePtr (*packed_vector3_array_operator_index)(GDExtensionTypePtr p_self, GDExtensionInt p_index); // p_self should be a PackedVector3Array, returns Vector3 ptr
	GDExtensionTypePtr (*packed_vector3_array_operator_index_const)(GDExtensionConstTypePtr p_self, GDExtensionInt p_index); // p_self should be a PackedVector3Array, returns Vector3 ptr

	GDExtensionVariantPtr (*array_operator_index)(GDExtensionTypePtr p_self, GDExtensionInt p_index); // p_self should be an Array ptr
	GDExtensionVariantPtr (*array_operator_index_const)(GDExtensionConstTypePtr p_self, GDExtensionInt p_index); // p_self should be an Array ptr
	void (*array_ref)(GDExtensionTypePtr p_self, GDExtensionConstTypePtr p_from); // p_self should be an Array ptr
	void (*array_set_typed)(GDExtensionTypePtr p_self, GDExtensionVariantType p_type, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstVariantPtr p_script); // p_self should be an Array ptr

	/* Dictionary functions */

	GDExtensionVariantPtr (*dictionary_operator_index)(GDExtensionTypePtr p_self, GDExtensionConstVariantPtr p_key); // p_self should be an Dictionary ptr
	GDExtensionVariantPtr (*dictionary_operator_index_const)(GDExtensionConstTypePtr p_self, GDExtensionConstVariantPtr p_key); // p_self should be an Dictionary ptr

	/* OBJECT */

	void (*object_method_bind_call)(GDExtensionMethodBindPtr p_method_bind, GDExtensionObjectPtr p_instance, const GDExtensionConstVariantPtr *p_args, GDExtensionInt p_arg_count, GDExtensionVariantPtr r_ret, GDExtensionCallError *r_error);
	void (*object_method_bind_ptrcall)(GDExtensionMethodBindPtr p_method_bind, GDExtensionObjectPtr p_instance, const GDExtensionConstTypePtr *p_args, GDExtensionTypePtr r_ret);
	void (*object_destroy)(GDExtensionObjectPtr p_o);
	GDExtensionObjectPtr (*global_get_singleton)(GDExtensionConstStringNamePtr p_name);

	void *(*object_get_instance_binding)(GDExtensionObjectPtr p_o, void *p_token, const GDExtensionInstanceBindingCallbacks *p_callbacks);
	void (*object_set_instance_binding)(GDExtensionObjectPtr p_o, void *p_token, void *p_binding, const GDExtensionInstanceBindingCallbacks *p_callbacks);

	void (*object_set_instance)(GDExtensionObjectPtr p_o, GDExtensionConstStringNamePtr p_classname, GDExtensionClassInstancePtr p_instance); /* p_classname should be a registered extension class and should extend the p_o object's class. */

	GDExtensionObjectPtr (*object_cast_to)(GDExtensionConstObjectPtr p_object, void *p_class_tag);
	GDExtensionObjectPtr (*object_get_instance_from_id)(GDObjectInstanceID p_instance_id);
	GDObjectInstanceID (*object_get_instance_id)(GDExtensionConstObjectPtr p_object);

	/* REFERENCE */

	GDExtensionObjectPtr (*ref_get_object)(GDExtensionConstRefPtr p_ref);
	void (*ref_set_object)(GDExtensionRefPtr p_ref, GDExtensionObjectPtr p_object);

	/* SCRIPT INSTANCE */

	GDExtensionScriptInstancePtr (*script_instance_create)(const GDExtensionScriptInstanceInfo *p_info, GDExtensionScriptInstanceDataPtr p_instance_data);

	/* CLASSDB */

	GDExtensionObjectPtr (*classdb_construct_object)(GDExtensionConstStringNamePtr p_classname); /* The passed class must be a built-in godot class, or an already-registered extension class. In both case, object_set_instance should be called to fully initialize the object. */
	GDExtensionMethodBindPtr (*classdb_get_method_bind)(GDExtensionConstStringNamePtr p_classname, GDExtensionConstStringNamePtr p_methodname, GDExtensionInt p_hash);
	void *(*classdb_get_class_tag)(GDExtensionConstStringNamePtr p_classname);

	/* CLASSDB EXTENSION */

	/* Provided parameters for `classdb_register_extension_*` can be safely freed once the function returns. */
	void (*classdb_register_extension_class)(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringNamePtr p_parent_class_name, const GDExtensionClassCreationInfo *p_extension_funcs);
	void (*classdb_register_extension_class_method)(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, const GDExtensionClassMethodInfo *p_method_info);
	void (*classdb_register_extension_class_integer_constant)(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringNamePtr p_enum_name, GDExtensionConstStringNamePtr p_constant_name, GDExtensionInt p_constant_value, GDExtensionBool p_is_bitfield);
	void (*classdb_register_extension_class_property)(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, const GDExtensionPropertyInfo *p_info, GDExtensionConstStringNamePtr p_setter, GDExtensionConstStringNamePtr p_getter);
	void (*classdb_register_extension_class_property_group)(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringPtr p_group_name, GDExtensionConstStringPtr p_prefix);
	void (*classdb_register_extension_class_property_subgroup)(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringPtr p_subgroup_name, GDExtensionConstStringPtr p_prefix);
	void (*classdb_register_extension_class_signal)(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringNamePtr p_signal_name, const GDExtensionPropertyInfo *p_argument_info, GDExtensionInt p_argument_count);
	void (*classdb_unregister_extension_class)(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name); /* Unregistering a parent class before a class that inherits it will result in failure. Inheritors must be unregistered first. */

	void (*get_library_path)(GDExtensionClassLibraryPtr p_library, GDExtensionStringPtr r_path);

} GDExtensionInterface;

/* INITIALIZATION */

typedef enum {
	GDEXTENSION_INITIALIZATION_CORE,
	GDEXTENSION_INITIALIZATION_SERVERS,
	GDEXTENSION_INITIALIZATION_SCENE,
	GDEXTENSION_INITIALIZATION_EDITOR,
	GDEXTENSION_MAX_INITIALIZATION_LEVEL,
} GDExtensionInitializationLevel;

typedef struct {
	/* Minimum initialization level required.
	 * If Core or Servers, the extension needs editor or game restart to take effect */
	GDExtensionInitializationLevel minimum_initialization_level;
	/* Up to the user to supply when initializing */
	void *userdata;
	/* This function will be called multiple times for each initialization level. */
	void (*initialize)(void *userdata, GDExtensionInitializationLevel p_level);
	void (*deinitialize)(void *userdata, GDExtensionInitializationLevel p_level);
} GDExtensionInitialization;

/* Define a C function prototype that implements the function below and expose it to dlopen() (or similar).
 * This is the entry point of the GDExtension library and will be called on initialization.
 * It can be used to set up different init levels, which are called during various stages of initialization/shutdown.
 * The function name must be a unique one specified in the .gdextension config file.
 */
typedef GDExtensionBool (*GDExtensionInitializationFunction)(const GDExtensionInterface *p_interface, GDExtensionClassLibraryPtr p_library, GDExtensionInitialization *r_initialization);

#ifdef __cplusplus
}
#endif

#endif // GDEXTENSION_INTERFACE_H
