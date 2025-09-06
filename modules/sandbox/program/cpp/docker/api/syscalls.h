/**************************************************************************/
/*  syscalls.h                                                            */
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
#define GAME_API_BASE 500

// System calls written in assembly
#define ECALL_PRINT (GAME_API_BASE + 0)
#define ECALL_VCALL (GAME_API_BASE + 1)
#define ECALL_VEVAL (GAME_API_BASE + 2)
#define ECALL_VASSIGN (GAME_API_BASE + 3)
#define ECALL_GET_OBJ (GAME_API_BASE + 4) // Get an object by name
#define ECALL_OBJ (GAME_API_BASE + 5) // All the Object functions
#define ECALL_OBJ_CALLP (GAME_API_BASE + 6) // Call a method on an object
#define ECALL_GET_NODE (GAME_API_BASE + 7) // Get a node by path
#define ECALL_NODE (GAME_API_BASE + 8) // All the Node functions
#define ECALL_NODE2D (GAME_API_BASE + 9) // All the Node2D functions
#define ECALL_NODE3D (GAME_API_BASE + 10) // All the Node3D functions

#define ECALL_THROW (GAME_API_BASE + 11)
#define ECALL_IS_EDITOR (GAME_API_BASE + 12)

#define ECALL_SINCOS (GAME_API_BASE + 13)
#define ECALL_VEC2_LENGTH (GAME_API_BASE + 14)
#define ECALL_VEC2_NORMALIZED (GAME_API_BASE + 15)
#define ECALL_VEC2_ROTATED (GAME_API_BASE + 16)

#define ECALL_VCREATE (GAME_API_BASE + 17)
#define ECALL_VCLONE (GAME_API_BASE + 18)
#define ECALL_VFETCH (GAME_API_BASE + 19)
#define ECALL_VSTORE (GAME_API_BASE + 20)

#define ECALL_ARRAY_OPS (GAME_API_BASE + 21)
#define ECALL_ARRAY_AT (GAME_API_BASE + 22)
#define ECALL_ARRAY_SIZE (GAME_API_BASE + 23)

#define ECALL_DICTIONARY_OPS (GAME_API_BASE + 24)

#define ECALL_STRING_CREATE (GAME_API_BASE + 25)
#define ECALL_STRING_OPS (GAME_API_BASE + 26)
#define ECALL_STRING_AT (GAME_API_BASE + 27)
#define ECALL_STRING_SIZE (GAME_API_BASE + 28)
#define ECALL_STRING_APPEND (GAME_API_BASE + 29)

#define ECALL_TIMER_PERIODIC (GAME_API_BASE + 30)
#define ECALL_TIMER_STOP (GAME_API_BASE + 31)

#define ECALL_NODE_CREATE (GAME_API_BASE + 32)

#define ECALL_MATH_OP32 (GAME_API_BASE + 33)
#define ECALL_MATH_OP64 (GAME_API_BASE + 34)
#define ECALL_LERP_OP32 (GAME_API_BASE + 35)
#define ECALL_LERP_OP64 (GAME_API_BASE + 36)

#define ECALL_VEC3_OPS (GAME_API_BASE + 37)

#define ECALL_CALLABLE_CREATE (GAME_API_BASE + 38)

#define ECALL_LOAD (GAME_API_BASE + 39)

#define ECALL_TRANSFORM_2D_OPS (GAME_API_BASE + 40)
#define ECALL_TRANSFORM_3D_OPS (GAME_API_BASE + 41)
#define ECALL_BASIS_OPS (GAME_API_BASE + 42)

#define ECALL_VEC2_OPS (GAME_API_BASE + 43)

#define ECALL_QUAT_OPS (GAME_API_BASE + 44)

#define ECALL_OBJ_PROP_GET (GAME_API_BASE + 45)
#define ECALL_OBJ_PROP_SET (GAME_API_BASE + 46)

#define ECALL_SANDBOX_ADD (GAME_API_BASE + 47)

#define ECALL_LAST (GAME_API_BASE + 48)

#define STRINGIFY_HELPER(x) #x
#define STRINGIFY(x) STRINGIFY_HELPER(x)

#define MAKE_SYSCALL(number, rval, name, ...)                      \
	__asm__(".pushsection .text\n"                                 \
			".global " #name "\n"                                  \
			".type " #name ", @function\n"                         \
			"" #name ":\n"                                         \
			"	li a7, " STRINGIFY(number) "\n"                    \
										   "   ecall\n"            \
										   "   ret\n"              \
										   ".popsection .text\n"); \
	extern "C" rval name(__VA_ARGS__);

#define EXTERN_SYSCALL(rval, name, ...) \
	extern "C" rval name(__VA_ARGS__);

enum class Object_Op {
	GET_METHOD_LIST,
	GET,
	SET,
	GET_PROPERTY_LIST,
	CONNECT,
	DISCONNECT,
	GET_SIGNAL_LIST,
};

enum class Node_Create_Shortlist {
	CREATE_CLASSDB = 0,
	CREATE_NODE,
	CREATE_NODE2D,
	CREATE_NODE3D,
};

enum class Node_Op {
	GET_NAME = 0,
	GET_PATH,
	GET_PARENT,
	QUEUE_FREE,
	DUPLICATE,
	GET_CHILD_COUNT,
	GET_CHILD,
	ADD_CHILD,
	ADD_CHILD_DEFERRED,
	ADD_SIBLING,
	ADD_SIBLING_DEFERRED,
	MOVE_CHILD,
	REMOVE_CHILD,
	REMOVE_CHILD_DEFERRED,
	GET_CHILDREN,
	SET_NAME,
	REPARENT,
	REPLACE_BY,
	ADD_TO_GROUP,
	REMOVE_FROM_GROUP,
	IS_IN_GROUP,
	IS_INSIDE_TREE,
};

enum class Node2D_Op {
	GET_POSITION = 0,
	SET_POSITION,
	GET_ROTATION,
	SET_ROTATION,
	GET_SCALE,
	SET_SCALE,
	GET_SKEW,
	SET_SKEW,
	GET_TRANSFORM,
	SET_TRANSFORM,
};

enum class Node3D_Op {
	GET_POSITION = 0,
	SET_POSITION,
	GET_ROTATION,
	SET_ROTATION,
	GET_SCALE,
	SET_SCALE,
	GET_TRANSFORM,
	SET_TRANSFORM,
	GET_QUATERNION,
	SET_QUATERNION,
};

enum class Array_Op {
	CREATE = 0,
	PUSH_BACK,
	PUSH_FRONT,
	POP_AT,
	POP_BACK,
	POP_FRONT,
	INSERT,
	ERASE,
	RESIZE,
	CLEAR,
	SORT,
	FETCH_TO_VECTOR,
	HAS,
};

enum class Dictionary_Op {
	GET = 0,
	SET,
	ERASE,
	HAS,
	GET_KEYS,
	GET_VALUES,
	GET_SIZE,
	CLEAR,
	MERGE,
	GET_OR_ADD,
};

enum class String_Op {
	COPY = 0,
	GET_LENGTH,
	GET_CHAR,
	APPEND,
	INSERT,
	FIND,
	ERASE,
	TO_STD_STRING,
	COMPARE,
	COMPARE_CSTR,
};

enum class Math_Op {
	SIN = 0,
	COS,
	TAN,
	ASIN,
	ACOS,
	ATAN,
	ATAN2,
	POW,
};

enum class Lerp_Op {
	LERP = 0,
	SMOOTHSTEP,
	CLAMP,
	SLERP,
};

enum class Vec2_Op {
	NORMALIZE = 0,
	LENGTH,
	LENGTH_SQ,
	ANGLE,
	ANGLE_TO,
	ANGLE_TO_POINT,
	PROJECT,
	DIRECTION_TO,
	SLIDE,
	BOUNCE,
	REFLECT,
	LIMIT_LENGTH,
	LERP,
	CUBIC_INTERPOLATE,
	SLERP,
	MOVE_TOWARD,
	ROTATED,
};

enum class Vec3_Op {
	HASH = 0,
	LENGTH,
	NORMALIZE,
	DOT,
	CROSS,
	DISTANCE_TO,
	DISTANCE_SQ_TO,
	ANGLE_TO,
	PROJECT,
	REFLECT,
	ROTATED,
	FLOOR,
};

enum class Transform2D_Op {
	IDENTITY = 0,
	CREATE,
	ASSIGN,
	GET_COLUMN,
	SET_COLUMN,
	ROTATED,
	SCALED,
	TRANSLATED,
	INVERTED,
	AFFINE_INVERTED,
	ORTHONORMALIZED,
	LOOKING_AT,
	INTERPOLATE_WITH,
	XFORM,
	XFORM_INV,
};

enum class Transform3D_Op {
	IDENTITY = 0,
	CREATE,
	ASSIGN,
	GET_BASIS,
	SET_BASIS,
	GET_ORIGIN,
	SET_ORIGIN,
	ROTATED,
	ROTATED_LOCAL,
	SCALED,
	SCALED_LOCAL,
	TRANSLATED,
	TRANSLATED_LOCAL,
	INVERTED,
	AFFINE_INVERTED,
	ORTHONORMALIZED,
	LOOKING_AT,
	INTERPOLATE_WITH,
	XFORM,
	XFORM_INV,
};

enum class Basis_Op {
	IDENTITY = 0,
	CREATE,
	ASSIGN,
	GET_ROW,
	SET_ROW,
	GET_COLUMN,
	SET_COLUMN,
	INVERTED,
	TRANSPOSED,
	DETERMINANT,
	ROTATED,
	LERP,
	SLERP,
};

enum class Quaternion_Op {
	CREATE = 0,
	ASSIGN,
	DOT,
	LENGTH_SQUARED,
	LENGTH,
	NORMALIZE,
	INVERSE,
	LOG,
	EXP,
	ANGLE_TO,
	SLERP,
	SLERPNI,
	CUBIC_INTERPOLATE,
	CUBIC_INTERPOLATE_IN_TIME,
	AT,
	GET_AXIS,
	GET_ANGLE,
	MUL,
};
