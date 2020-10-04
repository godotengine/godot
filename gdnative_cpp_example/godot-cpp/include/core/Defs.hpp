#ifndef DEFS_H
#define DEFS_H

namespace godot {

enum class Error {
	OK,
	FAILED, ///< Generic fail error
	ERR_UNAVAILABLE, ///< What is requested is unsupported/unavailable
	ERR_UNCONFIGURED, ///< The object being used hasnt been properly set up yet
	ERR_UNAUTHORIZED, ///< Missing credentials for requested resource
	ERR_PARAMETER_RANGE_ERROR, ///< Parameter given out of range (5)
	ERR_OUT_OF_MEMORY, ///< Out of memory
	ERR_FILE_NOT_FOUND,
	ERR_FILE_BAD_DRIVE,
	ERR_FILE_BAD_PATH,
	ERR_FILE_NO_PERMISSION, // (10)
	ERR_FILE_ALREADY_IN_USE,
	ERR_FILE_CANT_OPEN,
	ERR_FILE_CANT_WRITE,
	ERR_FILE_CANT_READ,
	ERR_FILE_UNRECOGNIZED, // (15)
	ERR_FILE_CORRUPT,
	ERR_FILE_MISSING_DEPENDENCIES,
	ERR_FILE_EOF,
	ERR_CANT_OPEN, ///< Can't open a resource/socket/file
	ERR_CANT_CREATE, // (20)
	ERR_QUERY_FAILED,
	ERR_ALREADY_IN_USE,
	ERR_LOCKED, ///< resource is locked
	ERR_TIMEOUT,
	ERR_CANT_CONNECT, // (25)
	ERR_CANT_RESOLVE,
	ERR_CONNECTION_ERROR,
	ERR_CANT_AQUIRE_RESOURCE,
	ERR_CANT_FORK,
	ERR_INVALID_DATA, ///< Data passed is invalid	(30)
	ERR_INVALID_PARAMETER, ///< Parameter passed is invalid
	ERR_ALREADY_EXISTS, ///< When adding, item already exists
	ERR_DOES_NOT_EXIST, ///< When retrieving/erasing, it item does not exist
	ERR_DATABASE_CANT_READ, ///< database is full
	ERR_DATABASE_CANT_WRITE, ///< database is full	(35)
	ERR_COMPILATION_FAILED,
	ERR_METHOD_NOT_FOUND,
	ERR_LINK_FAILED,
	ERR_SCRIPT_FAILED,
	ERR_CYCLIC_LINK, // (40)
	ERR_INVALID_DECLARATION,
	ERR_DUPLICATE_SYMBOL,
	ERR_PARSE_ERROR,
	ERR_BUSY,
	ERR_SKIP, // (45)
	ERR_HELP, ///< user requested help!!
	ERR_BUG, ///< a bug in the software certainly happened, due to a double check failing or unexpected behavior.
	ERR_PRINTER_ON_FIRE, /// the parallel port printer is engulfed in flames
	ERR_OMFG_THIS_IS_VERY_VERY_BAD, ///< shit happens, has never been used, though
	ERR_WTF = ERR_OMFG_THIS_IS_VERY_VERY_BAD ///< short version of the above
};

} // namespace godot

#include <GodotGlobal.hpp>

// alloca() is non-standard. When using MSVC, it's in malloc.h.
#if defined(__linux__) || defined(__APPLE__)
#include <alloca.h>
#else
#include <malloc.h>
#endif

typedef float real_t;

#define CMP_EPSILON 0.00001
#define CMP_EPSILON2 (CMP_EPSILON * CMP_EPSILON)
#define Math_PI 3.14159265358979323846
#define Math_TAU 6.2831853071795864769252867666

#define _PLANE_EQ_DOT_EPSILON 0.999
#define _PLANE_EQ_D_EPSILON 0.0001

#ifdef __GNUC__
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) x
#define unlikely(x) x
#endif

// Don't use this directly; instead, use any of the CRASH_* macros
#ifdef _MSC_VER
#define GENERATE_TRAP                       \
	__debugbreak();                         \
	/* Avoid warning about control paths */ \
	for (;;) {                              \
	}
#else
#define GENERATE_TRAP __builtin_trap();
#endif

// ERR/WARN macros
#ifndef WARN_PRINT
#define WARN_PRINT(msg) godot::Godot::print_warning(msg, __func__, __FILE__, __LINE__)
#endif

#ifndef WARN_PRINTS
#define WARN_PRINTS(msg) WARN_PRINT((msg).utf8().get_data())
#endif

#ifndef ERR_PRINT
#define ERR_PRINT(msg) godot::Godot::print_error(msg, __func__, __FILE__, __LINE__)
#endif

#ifndef ERR_PRINTS
#define ERR_PRINTS(msg) ERR_PRINT((msg).utf8().get_data())
#endif

#ifndef FATAL_PRINT
#define FATAL_PRINT(msg) ERR_PRINT(godot::String("FATAL: ") + (msg))
#endif

#ifndef ERR_MSG_INDEX
#define ERR_MSG_INDEX(index, size) (godot::String("Index ") + #index + "=" + godot::String::num_int64(index) + " out of size (" + #size + "=" + godot::String::num_int64(size) + ")")
#endif

#ifndef ERR_MSG_NULL
#define ERR_MSG_NULL(param) (godot::String("Parameter '") + #param + "' is null.")
#endif

#ifndef ERR_MSG_COND
#define ERR_MSG_COND(cond) (godot::String("Condition '") + #cond + "' is true.")
#endif

#ifndef ERR_FAIL_INDEX
#define ERR_FAIL_INDEX(index, size)                       \
	do {                                                  \
		if (unlikely((index) < 0 || (index) >= (size))) { \
			ERR_PRINT(ERR_MSG_INDEX(index, size));        \
			return;                                       \
		}                                                 \
	} while (0)
#endif

#ifndef ERR_FAIL_INDEX_V
#define ERR_FAIL_INDEX_V(index, size, ret)                \
	do {                                                  \
		if (unlikely((index) < 0 || (index) >= (size))) { \
			ERR_PRINT(ERR_MSG_INDEX(index, size));        \
			return ret;                                   \
		}                                                 \
	} while (0)
#endif

#ifndef ERR_FAIL_UNSIGNED_INDEX_V
#define ERR_FAIL_UNSIGNED_INDEX_V(index, size, ret) \
	do {                                            \
		if (unlikely((index) >= (size))) {          \
			ERR_PRINT(ERR_MSG_INDEX(index, size));  \
			return ret;                             \
		}                                           \
	} while (0)
#endif

#ifndef CRASH_BAD_INDEX
#define CRASH_BAD_INDEX(index, size)                      \
	do {                                                  \
		if (unlikely((index) < 0 || (index) >= (size))) { \
			FATAL_PRINT(ERR_MSG_INDEX(index, size));      \
			GENERATE_TRAP;                                \
		}                                                 \
	} while (0)
#endif

#ifndef ERR_FAIL_NULL
#define ERR_FAIL_NULL(param)                \
	do {                                    \
		if (unlikely(!param)) {             \
			ERR_PRINT(ERR_MSG_NULL(param)); \
			return;                         \
		}                                   \
	} while (0)
#endif

#ifndef ERR_FAIL_NULL_V
#define ERR_FAIL_NULL_V(param, ret)         \
	do {                                    \
		if (unlikely(!param)) {             \
			ERR_PRINT(ERR_MSG_NULL(param)); \
			return ret;                     \
		}                                   \
	} while (0)
#endif

#ifndef ERR_FAIL_COND
#define ERR_FAIL_COND(cond)                \
	do {                                   \
		if (unlikely(cond)) {              \
			ERR_PRINT(ERR_MSG_COND(cond)); \
			return;                        \
		}                                  \
	} while (0)
#endif

#ifndef CRASH_COND
#define CRASH_COND(cond)                     \
	do {                                     \
		if (unlikely(cond)) {                \
			FATAL_PRINT(ERR_MSG_COND(cond)); \
			return;                          \
		}                                    \
	} while (0)
#endif

#ifndef ERR_FAIL_COND_V
#define ERR_FAIL_COND_V(cond, ret)         \
	do {                                   \
		if (unlikely(cond)) {              \
			ERR_PRINT(ERR_MSG_COND(cond)); \
			return ret;                    \
		}                                  \
	} while (0)
#endif

#ifndef ERR_CONTINUE
#define ERR_CONTINUE(cond)                 \
	{                                      \
		if (unlikely(cond)) {              \
			ERR_PRINT(ERR_MSG_COND(cond)); \
			continue;                      \
		}                                  \
	}
#endif

#ifndef ERR_BREAK
#define ERR_BREAK(cond)                    \
	{                                      \
		if (unlikely(cond)) {              \
			ERR_PRINT(ERR_MSG_COND(cond)); \
			break;                         \
		}                                  \
	}
#endif

#ifndef ERR_FAIL
#define ERR_FAIL()                            \
	do {                                      \
		ERR_PRINT("Method/Function Failed."); \
		return;                               \
	} while (0)
#endif

#ifndef ERR_FAIL_V
#define ERR_FAIL_V(ret)                       \
	do {                                      \
		ERR_PRINT("Method/Function Failed."); \
		return ret;                           \
	} while (0)
#endif

#ifndef CRASH_NOW
#define CRASH_NOW()                             \
	do {                                        \
		FATAL_PRINT("Method/Function Failed."); \
		GENERATE_TRAP;                          \
	} while (0)
#endif

#endif // DEFS_H
