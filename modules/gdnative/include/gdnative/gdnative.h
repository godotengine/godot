/*************************************************************************/
/*  gdnative.h                                                           */
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

#ifndef GODOT_GDNATIVE_H
#define GODOT_GDNATIVE_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__ANDROID__)
#define GDCALLINGCONV
#define GDAPI GDCALLINGCONV
#elif defined(__APPLE__)
#include "TargetConditionals.h"
#if TARGET_OS_IPHONE
#define GDCALLINGCONV __attribute__((visibility("default")))
#define GDAPI GDCALLINGCONV
#elif TARGET_OS_MAC
#define GDCALLINGCONV __attribute__((sysv_abi))
#define GDAPI GDCALLINGCONV
#endif
#else // !_WIN32 && !__APPLE__
#define GDCALLINGCONV __attribute__((sysv_abi))
#define GDAPI GDCALLINGCONV
#endif

// This is for libraries *using* the header, NOT GODOT EXPOSING STUFF!!
#ifdef __GNUC__
#define GDN_EXPORT __attribute__((visibility("default")))
#elif defined(_WIN32)
#define GDN_EXPORT __declspec(dllexport)
#else
#define GDN_EXPORT
#endif

#include <stdbool.h>
#include <stdint.h>

////// Error

typedef enum {
	GODOT_OK, // (0)
	GODOT_FAILED, ///< Generic fail error
	GODOT_ERR_UNAVAILABLE, ///< What is requested is unsupported/unavailable
	GODOT_ERR_UNCONFIGURED, ///< The object being used hasn't been properly set up yet
	GODOT_ERR_UNAUTHORIZED, ///< Missing credentials for requested resource
	GODOT_ERR_PARAMETER_RANGE_ERROR, ///< Parameter given out of range (5)
	GODOT_ERR_OUT_OF_MEMORY, ///< Out of memory
	GODOT_ERR_FILE_NOT_FOUND,
	GODOT_ERR_FILE_BAD_DRIVE,
	GODOT_ERR_FILE_BAD_PATH,
	GODOT_ERR_FILE_NO_PERMISSION, // (10)
	GODOT_ERR_FILE_ALREADY_IN_USE,
	GODOT_ERR_FILE_CANT_OPEN,
	GODOT_ERR_FILE_CANT_WRITE,
	GODOT_ERR_FILE_CANT_READ,
	GODOT_ERR_FILE_UNRECOGNIZED, // (15)
	GODOT_ERR_FILE_CORRUPT,
	GODOT_ERR_FILE_MISSING_DEPENDENCIES,
	GODOT_ERR_FILE_EOF,
	GODOT_ERR_CANT_OPEN, ///< Can't open a resource/socket/file
	GODOT_ERR_CANT_CREATE, // (20)
	GODOT_ERR_QUERY_FAILED,
	GODOT_ERR_ALREADY_IN_USE,
	GODOT_ERR_LOCKED, ///< resource is locked
	GODOT_ERR_TIMEOUT,
	GODOT_ERR_CANT_CONNECT, // (25)
	GODOT_ERR_CANT_RESOLVE,
	GODOT_ERR_CONNECTION_ERROR,
	GODOT_ERR_CANT_ACQUIRE_RESOURCE,
	GODOT_ERR_CANT_FORK,
	GODOT_ERR_INVALID_DATA, ///< Data passed is invalid (30)
	GODOT_ERR_INVALID_PARAMETER, ///< Parameter passed is invalid
	GODOT_ERR_ALREADY_EXISTS, ///< When adding, item already exists
	GODOT_ERR_DOES_NOT_EXIST, ///< When retrieving/erasing, it item does not exist
	GODOT_ERR_DATABASE_CANT_READ, ///< database is full
	GODOT_ERR_DATABASE_CANT_WRITE, ///< database is full (35)
	GODOT_ERR_COMPILATION_FAILED,
	GODOT_ERR_METHOD_NOT_FOUND,
	GODOT_ERR_LINK_FAILED,
	GODOT_ERR_SCRIPT_FAILED,
	GODOT_ERR_CYCLIC_LINK, // (40)
	GODOT_ERR_INVALID_DECLARATION,
	GODOT_ERR_DUPLICATE_SYMBOL,
	GODOT_ERR_PARSE_ERROR,
	GODOT_ERR_BUSY,
	GODOT_ERR_SKIP, // (45)
	GODOT_ERR_HELP, ///< user requested help!!
	GODOT_ERR_BUG, ///< a bug in the software certainly happened, due to a double check failing or unexpected behavior.
	GODOT_ERR_PRINTER_ON_FIRE, /// the parallel port printer is engulfed in flames
} godot_error;

/////// Object (forward declared)
typedef void godot_object;

/////// String

#include <gdnative/string.h>

/////// String name

#include <gdnative/string_name.h>

////// Vector2 & Vector2i

#include <gdnative/vector2.h>

////// Rect2 & Rect2i

#include <gdnative/rect2.h>

////// Vector3 & Vector3i

#include <gdnative/vector3.h>

////// Transform2D

#include <gdnative/transform2d.h>

/////// Plane

#include <gdnative/plane.h>

/////// Quaternion

#include <gdnative/quaternion.h>

/////// AABB

#include <gdnative/aabb.h>

/////// Basis

#include <gdnative/basis.h>

/////// Transform3D

#include <gdnative/transform_3d.h>

/////// Color

#include <gdnative/color.h>

/////// NodePath

#include <gdnative/node_path.h>

/////// RID

#include <gdnative/rid.h>

/////// Callable & Signal

#include <gdnative/callable.h>

/////// Dictionary

#include <gdnative/dictionary.h>

/////// Array

#include <gdnative/array.h>

// single API file for Packed*Array
#include <gdnative/packed_arrays.h>

void GDAPI godot_object_destroy(godot_object *p_o);

////// Variant

#include <gdnative/variant.h>

////// Singleton API

godot_object GDAPI *godot_global_get_singleton(char *p_name); // Result shouldn't be freed.

////// MethodBind API

typedef struct {
	uint8_t _dont_touch_that[1]; // TODO
} godot_method_bind;

godot_method_bind GDAPI *godot_method_bind_get_method(const char *p_classname, const char *p_methodname);
void GDAPI godot_method_bind_ptrcall(godot_method_bind *p_method_bind, godot_object *p_instance, const void **p_args, void *p_ret);
godot_variant GDAPI godot_method_bind_call(godot_method_bind *p_method_bind, godot_object *p_instance, const godot_variant **p_args, const int p_arg_count, godot_variant_call_error *p_call_error);
////// Script API

typedef struct godot_gdnative_api_version {
	unsigned int major;
	unsigned int minor;
} godot_gdnative_api_version;

typedef struct godot_gdnative_api_struct godot_gdnative_api_struct;

struct godot_gdnative_api_struct {
	unsigned int type;
	godot_gdnative_api_version version;
	const godot_gdnative_api_struct *next;
};

#define GDNATIVE_VERSION_COMPATIBLE(want, have) (want.major == have.major && want.minor <= have.minor)

typedef struct {
	godot_bool in_editor;
	uint64_t core_api_hash;
	uint64_t editor_api_hash;
	uint64_t no_api_hash;
	void (*report_version_mismatch)(const godot_object *p_library, const char *p_what, godot_gdnative_api_version p_want, godot_gdnative_api_version p_have);
	void (*report_loading_error)(const godot_object *p_library, const char *p_what);
	godot_object *gd_native_library; // pointer to GDNativeLibrary that is being initialized
	const struct godot_gdnative_core_api_struct *api_struct;
	const godot_string *active_library_path;
} godot_gdnative_init_options;

typedef struct {
	godot_bool in_editor;
} godot_gdnative_terminate_options;

// Calling convention?
typedef godot_object *(*godot_class_constructor)();

godot_class_constructor GDAPI godot_get_class_constructor(const char *p_classname);

godot_dictionary GDAPI godot_get_global_constants();

////// GDNative procedure types
typedef void (*godot_gdnative_init_fn)(godot_gdnative_init_options *);
typedef void (*godot_gdnative_terminate_fn)(godot_gdnative_terminate_options *);
typedef godot_variant (*godot_gdnative_procedure_fn)(godot_array *);

////// System Functions

typedef godot_variant (*native_call_cb)(void *, godot_array *);
void GDAPI godot_register_native_call_type(const char *p_call_type, native_call_cb p_callback);

//using these will help Godot track how much memory is in use in debug mode
void GDAPI *godot_alloc(int p_bytes);
void GDAPI *godot_realloc(void *p_ptr, int p_bytes);
void GDAPI godot_free(void *p_ptr);

// Helper print functions.
void GDAPI godot_print_error(const char *p_description, const char *p_function, const char *p_file, int p_line);
void GDAPI godot_print_warning(const char *p_description, const char *p_function, const char *p_file, int p_line);
void GDAPI godot_print_script_error(const char *p_description, const char *p_function, const char *p_file, int p_line);

//tags used for safe dynamic casting
void GDAPI *godot_get_class_tag(const godot_string_name *p_class);
godot_object GDAPI *godot_object_cast_to(const godot_object *p_object, void *p_class_tag);

// equivalent of GDScript's instance_from_id
godot_object GDAPI *godot_instance_from_id(uint64_t p_instance_id);

uint64_t GDAPI godot_object_get_instance_id(const godot_object *p_object);

#ifdef __cplusplus
}
#endif

#endif // GODOT_GDNATIVE_H
