/*************************************************************************/
/*  gdnative.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#ifndef GODOT_GDNATIVE_H
#define GODOT_GDNATIVE_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef GDAPI_BUILT_IN
#define GDAPI_EXPORT
#endif

#ifdef _WIN32
#if defined(GDAPI_EXPORT)
#define GDCALLINGCONV
#define GDAPI __declspec(dllexport) GDCALLINGCONV
#else
#define GDCALLINGCONV
#define GDAPI __declspec(dllimport) GDCALLINGCONV
#endif
#elif defined(__APPLE__)
#include "TargetConditionals.h"
#if TARGET_OS_IPHONE
#define GDCALLINGCONV
#define GDAPI
#elif TARGET_OS_MAC
#define GDCALLINGCONV __attribute__((sysv_abi))
#define GDAPI GDCALLINGCONV
#endif
#else
#define GDCALLINGCONV __attribute__((sysv_abi))
#define GDAPI GDCALLINGCONV
#endif

// This is for libraries *using* the header, NOT GODOT EXPOSING STUFF!!
#ifdef _WIN32
#define GDN_EXPORT __declspec(dllexport)
#else
#define GDN_EXPORT
#endif

#include <stdbool.h>
#include <stdint.h>

#define GODOT_API_VERSION 1

////// Error

typedef enum {
	GODOT_OK,
	GODOT_FAILED, ///< Generic fail error
	GODOT_ERR_UNAVAILABLE, ///< What is requested is unsupported/unavailable
	GODOT_ERR_UNCONFIGURED, ///< The object being used hasnt been properly set up yet
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
	GODOT_ERR_CANT_AQUIRE_RESOURCE,
	GODOT_ERR_CANT_FORK,
	GODOT_ERR_INVALID_DATA, ///< Data passed is invalid	(30)
	GODOT_ERR_INVALID_PARAMETER, ///< Parameter passed is invalid
	GODOT_ERR_ALREADY_EXISTS, ///< When adding, item already exists
	GODOT_ERR_DOES_NOT_EXIST, ///< When retrieving/erasing, it item does not exist
	GODOT_ERR_DATABASE_CANT_READ, ///< database is full
	GODOT_ERR_DATABASE_CANT_WRITE, ///< database is full	(35)
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
	GODOT_ERR_OMFG_THIS_IS_VERY_VERY_BAD, ///< shit happens, has never been used, though
	GODOT_ERR_WTF = GODOT_ERR_OMFG_THIS_IS_VERY_VERY_BAD ///< short version of the above
} godot_error;

////// bool

typedef bool godot_bool;

#define GODOT_TRUE 1
#define GODOT_FALSE 0

/////// int

typedef int godot_int;

/////// real

typedef float godot_real;

/////// Object (forward declared)
typedef void godot_object;

/////// Brute force forward declarations for the rest
/*
typedef struct godot_variant godot_variant;
typedef struct godot_string godot_string;
typedef struct godot_vector2 godot_vector2;
typedef struct godot_rect2 godot_rect2;
typedef struct godot_vector3 godot_vector3;
typedef struct godot_transform2d godot_transform2d;
typedef struct godot_plane godot_plane;
typedef struct godot_quat godot_quat;
typedef struct godot_rect3 godot_rect3;
typedef struct godot_basis godot_basis;
typedef struct godot_transform godot_transform;
typedef struct godot_color godot_color;
typedef struct godot_node_path godot_node_path;
typedef struct godot_rid godot_rid;
typedef struct godot_dictionary godot_dictionary;
typedef struct godot_array godot_array;
typedef struct godot_pool_byte_array godot_pool_byte_array;
typedef struct godot_pool_int_array godot_pool_int_array;
typedef struct godot_pool_real_array godot_pool_real_array;
typedef struct godot_pool_string_array godot_pool_string_array;
typedef struct godot_pool_vector2_array godot_pool_vector2_array;
typedef struct godot_pool_vector3_array godot_pool_vector3_array;
typedef struct godot_pool_color_array godot_pool_color_array;
*/
/////// String

#include "string.h"

////// Vector2

#include "vector2.h"

////// Rect2

#include "rect2.h"

////// Vector3

#include "vector3.h"

////// Transform2D

#include "transform2d.h"

/////// Plane

#include "plane.h"

/////// Quat

#include "quat.h"

/////// Rect3

#include "rect3.h"

/////// Basis

#include "basis.h"

/////// Transform

#include "transform.h"

/////// Color

#include "color.h"

/////// NodePath

#include "node_path.h"

/////// RID

#include "rid.h"

/////// Dictionary

#include "dictionary.h"

/////// Array

#include "array.h"

// single API file for Pool*Array
#include "pool_arrays.h"

void GDAPI godot_object_destroy(godot_object *p_o);

////// Variant

#include "variant.h"

////// Singleton API

godot_object GDAPI *godot_global_get_singleton(char *p_name); // result shouldn't be freed

////// MethodBind API

typedef struct {
	uint8_t _dont_touch_that[1]; // TODO
} godot_method_bind;

godot_method_bind GDAPI *godot_method_bind_get_method(const char *p_classname, const char *p_methodname);
void GDAPI godot_method_bind_ptrcall(godot_method_bind *p_method_bind, godot_object *p_instance, const void **p_args, void *p_ret);
godot_variant GDAPI godot_method_bind_call(godot_method_bind *p_method_bind, godot_object *p_instance, const godot_variant **p_args, const int p_arg_count, godot_variant_call_error *p_call_error);
////// Script API

typedef struct {
	godot_bool in_editor;
	uint64_t core_api_hash;
	uint64_t editor_api_hash;
	uint64_t no_api_hash;
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
typedef godot_variant (*godot_gdnative_procedure_fn)(void *, godot_array *);

////// System Functions

//using these will help Godot track how much memory is in use in debug mode
void GDAPI *godot_alloc(int p_bytes);
void GDAPI *godot_realloc(void *p_ptr, int p_bytes);
void GDAPI godot_free(void *p_ptr);

//print using Godot's error handler list
void GDAPI godot_print_error(const char *p_description, const char *p_function, const char *p_file, int p_line);
void GDAPI godot_print_warning(const char *p_description, const char *p_function, const char *p_file, int p_line);
void GDAPI godot_print(const godot_string *p_message);

#ifdef __cplusplus
}
#endif

#endif // GODOT_C_H
