/*************************************************************************/
/*  error_macros.h                                                       */
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

#ifndef ERROR_MACROS_H
#define ERROR_MACROS_H

#include "core/safe_refcount.h"
#include "core/typedefs.h"

/**
 * Error macros. Unlike exceptions and asserts, these macros try to maintain consistency and stability
 * inside the code. It is recommended to always return processable data, so in case of an error,
 * the engine can keep working well.
 * In most cases, bugs and/or invalid data are not fatal and should never allow a perfectly running application
 * to fail or crash.
 */

/**
 * Pointer to the error macro printing function. Reassign to any function to have errors printed
 */

/** Function used by the error macros */

// function, file, line, error, explanation

enum ErrorHandlerType {
	ERR_HANDLER_ERROR,
	ERR_HANDLER_WARNING,
	ERR_HANDLER_SCRIPT,
	ERR_HANDLER_SHADER,
};

class String;
typedef void (*ErrorHandlerFunc)(void *, const char *, const char *, int p_line, const char *, const char *, ErrorHandlerType p_type);

struct ErrorHandlerList {
	ErrorHandlerFunc errfunc;
	void *userdata;

	ErrorHandlerList *next;

	ErrorHandlerList() {
		errfunc = nullptr;
		next = nullptr;
		userdata = nullptr;
	}
};

void add_error_handler(ErrorHandlerList *p_handler);
void remove_error_handler(ErrorHandlerList *p_handler);

void _err_print_error(const char *p_function, const char *p_file, int p_line, const char *p_error, ErrorHandlerType p_type = ERR_HANDLER_ERROR);
void _err_print_error(const char *p_function, const char *p_file, int p_line, const String &p_error, ErrorHandlerType p_type = ERR_HANDLER_ERROR);
void _err_print_error(const char *p_function, const char *p_file, int p_line, const char *p_error, const char *p_message, ErrorHandlerType p_type = ERR_HANDLER_ERROR);
void _err_print_error(const char *p_function, const char *p_file, int p_line, const String &p_error, const char *p_message, ErrorHandlerType p_type = ERR_HANDLER_ERROR);
void _err_print_error(const char *p_function, const char *p_file, int p_line, const char *p_error, const String &p_message, ErrorHandlerType p_type = ERR_HANDLER_ERROR);
void _err_print_error(const char *p_function, const char *p_file, int p_line, const String &p_error, const String &p_message, ErrorHandlerType p_type = ERR_HANDLER_ERROR);
void _err_print_index_error(const char *p_function, const char *p_file, int p_line, int64_t p_index, int64_t p_size, const char *p_index_str, const char *p_size_str, const char *p_message = "", bool fatal = false);
void _err_print_index_error(const char *p_function, const char *p_file, int p_line, int64_t p_index, int64_t p_size, const char *p_index_str, const char *p_size_str, const String &p_message, bool fatal = false);
void _err_flush_stdout();

#ifndef _STR
#define _STR(m_x) #m_x
#define _MKSTR(m_x) _STR(m_x)
#endif

#define _FNL __FILE__ ":"

/** An index has failed if m_index<0 or m_index >=m_size, the function exits */

#ifdef __GNUC__
//#define FUNCTION_STR __PRETTY_FUNCTION__ - too annoying
#define FUNCTION_STR __FUNCTION__
#else
#define FUNCTION_STR __FUNCTION__
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

// (*): See https://stackoverflow.com/questions/257418/do-while-0-what-is-it-good-for

/**
 * If `m_index` is less than 0 or greater than or equal to `m_size`, prints a generic
 * error message and returns from the function. This macro should be preferred to
 * `ERR_FAIL_COND` for bounds checking.
 */
#define ERR_FAIL_INDEX(m_index, m_size)                                                                         \
	if (unlikely((m_index) < 0 || (m_index) >= (m_size))) {                                                     \
		_err_print_index_error(FUNCTION_STR, __FILE__, __LINE__, m_index, m_size, _STR(m_index), _STR(m_size)); \
		return;                                                                                                 \
	} else                                                                                                      \
		((void)0)

/**
 * If `m_index` is less than 0 or greater than or equal to `m_size`, prints a custom
 * error message and returns from the function. This macro should be preferred to
 * `ERR_FAIL_COND_MSG` for bounds checking.
 */
#define ERR_FAIL_INDEX_MSG(m_index, m_size, m_msg)                                                                     \
	if (unlikely((m_index) < 0 || (m_index) >= (m_size))) {                                                            \
		_err_print_index_error(FUNCTION_STR, __FILE__, __LINE__, m_index, m_size, _STR(m_index), _STR(m_size), m_msg); \
		return;                                                                                                        \
	} else                                                                                                             \
		((void)0)

/**
 * If `m_index` is less than 0 or greater than or equal to `m_size`,
 * prints a generic error message and returns the value specified in `m_retval`.
 * This macro should be preferred to `ERR_FAIL_COND_V` for bounds checking.
 */
#define ERR_FAIL_INDEX_V(m_index, m_size, m_retval)                                                             \
	if (unlikely((m_index) < 0 || (m_index) >= (m_size))) {                                                     \
		_err_print_index_error(FUNCTION_STR, __FILE__, __LINE__, m_index, m_size, _STR(m_index), _STR(m_size)); \
		return m_retval;                                                                                        \
	} else                                                                                                      \
		((void)0)

/**
 * If `m_index` is less than 0 or greater than or equal to `m_size`,
 * prints a custom error message and returns the value specified in `m_retval`.
 * This macro should be preferred to `ERR_FAIL_COND_V_MSG` for bounds checking.
 */
#define ERR_FAIL_INDEX_V_MSG(m_index, m_size, m_retval, m_msg)                                                         \
	if (unlikely((m_index) < 0 || (m_index) >= (m_size))) {                                                            \
		_err_print_index_error(FUNCTION_STR, __FILE__, __LINE__, m_index, m_size, _STR(m_index), _STR(m_size), m_msg); \
		return m_retval;                                                                                               \
	} else                                                                                                             \
		((void)0)

/**
 * If `m_index` is greater than or equal to `m_size`,
 * prints a generic error message and returns the value specified in `m_retval`.
 * This macro should be preferred to `ERR_FAIL_COND_V` for unsigned bounds checking.
 */
#define ERR_FAIL_UNSIGNED_INDEX(m_index, m_size)                                                                \
	if (unlikely((m_index) >= (m_size))) {                                                                      \
		_err_print_index_error(FUNCTION_STR, __FILE__, __LINE__, m_index, m_size, _STR(m_index), _STR(m_size)); \
		return;                                                                                                 \
	} else                                                                                                      \
		((void)0)

/**
 * If `m_index` is greater than or equal to `m_size`,
 * prints a generic error message and returns the value specified in `m_retval`.
 * This macro should be preferred to `ERR_FAIL_COND_V` for unsigned bounds checking.
 */
#define ERR_FAIL_UNSIGNED_INDEX_V(m_index, m_size, m_retval)                                                    \
	if (unlikely((m_index) >= (m_size))) {                                                                      \
		_err_print_index_error(FUNCTION_STR, __FILE__, __LINE__, m_index, m_size, _STR(m_index), _STR(m_size)); \
		return m_retval;                                                                                        \
	} else                                                                                                      \
		((void)0)

/**
 * If `m_index` is greater than or equal to `m_size`,
 * prints a custom error message and returns the value specified in `m_retval`.
 * This macro should be preferred to `ERR_FAIL_COND_V_MSG` for unsigned bounds checking.
 */
#define ERR_FAIL_UNSIGNED_INDEX_V_MSG(m_index, m_size, m_retval, m_msg)                                                \
	if (unlikely((m_index) >= (m_size))) {                                                                             \
		_err_print_index_error(FUNCTION_STR, __FILE__, __LINE__, m_index, m_size, _STR(m_index), _STR(m_size), m_msg); \
		return m_retval;                                                                                               \
	} else                                                                                                             \
		((void)0)

/**
 * If `m_index` is less than 0 or greater than or equal to `m_size`,
 * crashes the engine immediately with a generic error message.
 * Only use this if there's no sensible fallback (i.e. the error is unrecoverable).
 * This macro should be preferred to `CRASH_COND` for bounds checking.
 */
#define CRASH_BAD_INDEX(m_index, m_size)                                                                                  \
	if (unlikely((m_index) < 0 || (m_index) >= (m_size))) {                                                               \
		_err_print_index_error(FUNCTION_STR, __FILE__, __LINE__, m_index, m_size, _STR(m_index), _STR(m_size), "", true); \
		GENERATE_TRAP                                                                                                     \
	} else                                                                                                                \
		((void)0)

/**
 * If `m_index` is less than 0 or greater than or equal to `m_size`,
 * crashes the engine immediately with a custom error message.
 * Only use this if there's no sensible fallback (i.e. the error is unrecoverable).
 * This macro should be preferred to `CRASH_COND` for bounds checking.
 */
#define CRASH_BAD_INDEX_MSG(m_index, m_size, m_msg)                                                                          \
	if (unlikely((m_index) < 0 || (m_index) >= (m_size))) {                                                                  \
		_err_print_index_error(FUNCTION_STR, __FILE__, __LINE__, m_index, m_size, _STR(m_index), _STR(m_size), m_msg, true); \
		GENERATE_TRAP                                                                                                        \
	} else                                                                                                                   \
		((void)0)

/**
 * If `m_index` is greater than or equal to `m_size`,
 * crashes the engine immediately with a generic error message.
 * Only use this if there's no sensible fallback (i.e. the error is unrecoverable).
 * This macro should be preferred to `CRASH_COND` for bounds checking.
 */
#define CRASH_BAD_UNSIGNED_INDEX(m_index, m_size)                                                                         \
	if (unlikely((m_index) >= (m_size))) {                                                                                \
		_err_print_index_error(FUNCTION_STR, __FILE__, __LINE__, m_index, m_size, _STR(m_index), _STR(m_size), "", true); \
		GENERATE_TRAP                                                                                                     \
	} else                                                                                                                \
		((void)0)

/**
 * If `m_param` is `null`, prints a generic error message and returns from the function.
 */
#define ERR_FAIL_NULL(m_param)                                                                          \
	if (unlikely(!m_param)) {                                                                           \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Parameter \"" _STR(m_param) "\" is null."); \
		return;                                                                                         \
	} else                                                                                              \
		((void)0)

/**
 * If `m_param` is `null`, prints a custom error message and returns from the function.
 */
#define ERR_FAIL_NULL_MSG(m_param, m_msg)                                                                      \
	if (unlikely(!m_param)) {                                                                                  \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Parameter \"" _STR(m_param) "\" is null.", m_msg); \
		return;                                                                                                \
	} else                                                                                                     \
		((void)0)

/**
 * If `m_param` is `null`, prints a generic error message and returns the value specified in `m_retval`.
 */
#define ERR_FAIL_NULL_V(m_param, m_retval)                                                              \
	if (unlikely(!m_param)) {                                                                           \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Parameter \"" _STR(m_param) "\" is null."); \
		return m_retval;                                                                                \
	} else                                                                                              \
		((void)0)

/**
 * If `m_param` is `null`, prints a custom error message and returns the value specified in `m_retval`.
 */
#define ERR_FAIL_NULL_V_MSG(m_param, m_retval, m_msg)                                                          \
	if (unlikely(!m_param)) {                                                                                  \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Parameter \"" _STR(m_param) "\" is null.", m_msg); \
		return m_retval;                                                                                       \
	} else                                                                                                     \
		((void)0)

/**
 * If `m_cond` evaluates to `true`, prints a generic error message and returns from the function.
 */
#define ERR_FAIL_COND(m_cond)                                                                          \
	if (unlikely(m_cond)) {                                                                            \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Condition \"" _STR(m_cond) "\" is true."); \
		return;                                                                                        \
	} else                                                                                             \
		((void)0)

/**
 * If `m_cond` evaluates to `true`, prints a custom error message and returns from the function.
 */
#define ERR_FAIL_COND_MSG(m_cond, m_msg)                                                                      \
	if (unlikely(m_cond)) {                                                                                   \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Condition \"" _STR(m_cond) "\" is true.", m_msg); \
		return;                                                                                               \
	} else                                                                                                    \
		((void)0)

/**
 * If `m_cond` evaluates to `true`, crashes the engine immediately with a generic error message.
 * Only use this if there's no sensible fallback (i.e. the error is unrecoverable).
 */
#define CRASH_COND(m_cond)                                                                                    \
	if (unlikely(m_cond)) {                                                                                   \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "FATAL: Condition \"" _STR(m_cond) "\" is true."); \
		GENERATE_TRAP                                                                                         \
	} else                                                                                                    \
		((void)0)

/**
 * If `m_cond` evaluates to `true`, crashes the engine immediately with a custom error message.
 * Only use this if there's no sensible fallback (i.e. the error is unrecoverable).
 */
#define CRASH_COND_MSG(m_cond, m_msg)                                                                                \
	if (unlikely(m_cond)) {                                                                                          \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "FATAL: Condition \"" _STR(m_cond) "\" is true.", m_msg); \
		GENERATE_TRAP                                                                                                \
	} else                                                                                                           \
		((void)0)

/**
 * If `m_cond` evaluates to `true`, prints a generic error message and returns the value specified in `m_retval`.
 */
#define ERR_FAIL_COND_V(m_cond, m_retval)                                                                                        \
	if (unlikely(m_cond)) {                                                                                                      \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Condition \"" _STR(m_cond) "\" is true. Returned: " _STR(m_retval)); \
		return m_retval;                                                                                                         \
	} else                                                                                                                       \
		((void)0)

/**
 * If `m_cond` evaluates to `true`, prints a custom error message and returns the value specified in `m_retval`.
 */
#define ERR_FAIL_COND_V_MSG(m_cond, m_retval, m_msg)                                                                                    \
	if (unlikely(m_cond)) {                                                                                                             \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Condition \"" _STR(m_cond) "\" is true. Returned: " _STR(m_retval), m_msg); \
		return m_retval;                                                                                                                \
	} else                                                                                                                              \
		((void)0)

/**
 * If `m_cond` evaluates to `true`, prints a custom error message and continues the loop the macro is located in.
 */
#define ERR_CONTINUE(m_cond)                                                                                       \
	if (unlikely(m_cond)) {                                                                                        \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Condition \"" _STR(m_cond) "\" is true. Continuing."); \
		continue;                                                                                                  \
	} else                                                                                                         \
		((void)0)

/**
 * If `m_cond` evaluates to `true`, prints a custom error message and continues the loop the macro is located in.
 */
#define ERR_CONTINUE_MSG(m_cond, m_msg)                                                                                   \
	if (unlikely(m_cond)) {                                                                                               \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Condition \"" _STR(m_cond) "\" is true. Continuing.", m_msg); \
		continue;                                                                                                         \
	} else                                                                                                                \
		((void)0)

/**
 * If `m_cond` evaluates to `true`, prints a generic error message and breaks from the loop the macro is located in.
 */
#define ERR_BREAK(m_cond)                                                                                        \
	if (unlikely(m_cond)) {                                                                                      \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Condition \"" _STR(m_cond) "\" is true. Breaking."); \
		break;                                                                                                   \
	} else                                                                                                       \
		((void)0)

/**
 * If `m_cond` evaluates to `true`, prints a custom error message and breaks from the loop the macro is located in.
 */
#define ERR_BREAK_MSG(m_cond, m_msg)                                                                                    \
	if (unlikely(m_cond)) {                                                                                             \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Condition \"" _STR(m_cond) "\" is true. Breaking.", m_msg); \
		break;                                                                                                          \
	} else                                                                                                              \
		((void)0)

/**
 * Prints a generic error message and returns from the function.
 */
#define ERR_FAIL()                                                            \
	if (true) {                                                               \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Method failed."); \
		return;                                                               \
	} else                                                                    \
		((void)0)

/**
 * Prints a custom error message and returns from the function.
 */
#define ERR_FAIL_MSG(m_msg)                                                          \
	if (true) {                                                                      \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Method failed.", m_msg); \
		return;                                                                      \
	} else                                                                           \
		((void)0)

/**
 * Prints a generic error message and returns the value specified in `m_retval`.
 */
#define ERR_FAIL_V(m_retval)                                                                              \
	if (true) {                                                                                           \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Method failed. Returning: " __STR(m_retval)); \
		return m_retval;                                                                                  \
	} else                                                                                                \
		((void)0)

/**
 * Prints a custom error message and returns the value specified in `m_retval`.
 */
#define ERR_FAIL_V_MSG(m_retval, m_msg)                                                                          \
	if (true) {                                                                                                  \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Method failed. Returning: " __STR(m_retval), m_msg); \
		return m_retval;                                                                                         \
	} else                                                                                                       \
		((void)0)

/**
 * Crashes the engine immediately with a generic error message.
 * Only use this if there's no sensible fallback (i.e. the error is unrecoverable).
 */
#define CRASH_NOW()                                                                  \
	if (true) {                                                                      \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "FATAL: Method failed."); \
		void _err_flush_stdout();                                                    \
		GENERATE_TRAP                                                                \
	} else                                                                           \
		((void)0)

/**
 * Crashes the engine immediately with a custom error message.
 * Only use this if there's no sensible fallback (i.e. the error is unrecoverable).
 */
#define CRASH_NOW_MSG(m_msg)                                                                \
	if (true) {                                                                             \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "FATAL: Method failed.", m_msg); \
		void _err_flush_stdout();                                                           \
		GENERATE_TRAP                                                                       \
	} else                                                                                  \
		((void)0)

/**
 * Prints an error message without returning.
 */
#define ERR_PRINT(m_string) \
	_err_print_error(FUNCTION_STR, __FILE__, __LINE__, m_string)

/**
 * Prints an error message without returning, but only do so once in the application lifecycle.
 * This can be used to avoid spamming the console with error messages.
 */
#define ERR_PRINT_ONCE(m_string)                                          \
	if (true) {                                                           \
		static bool first_print = true;                                   \
		if (first_print) {                                                \
			_err_print_error(FUNCTION_STR, __FILE__, __LINE__, m_string); \
			first_print = false;                                          \
		}                                                                 \
	} else                                                                \
		((void)0)

/**
 * Prints a warning message without returning. To warn about deprecated usage,
 * use `WARN_DEPRECATED` or `WARN_DEPRECATED_MSG` instead.
 */
#define WARN_PRINT(m_string) \
	_err_print_error(FUNCTION_STR, __FILE__, __LINE__, m_string, ERR_HANDLER_WARNING)

/**
 * Prints a warning message without returning, but only do so once in the application lifecycle.
 * This can be used to avoid spamming the console with warning messages.
 */
#define WARN_PRINT_ONCE(m_string)                                                              \
	if (true) {                                                                                \
		static bool first_print = true;                                                        \
		if (first_print) {                                                                     \
			_err_print_error(FUNCTION_STR, __FILE__, __LINE__, m_string, ERR_HANDLER_WARNING); \
			first_print = false;                                                               \
		}                                                                                      \
	} else                                                                                     \
		((void)0)

/**
 * Prints a generic deprecation warning message without returning.
 * This should be preferred to `WARN_PRINT` for deprecation warnings.
 */
#define WARN_DEPRECATED                                                                                                                                    \
	if (true) {                                                                                                                                            \
		static SafeFlag warning_shown;                                                                                                                     \
		if (!warning_shown.is_set()) {                                                                                                                     \
			_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "This method has been deprecated and will be removed in the future.", ERR_HANDLER_WARNING); \
			warning_shown.set();                                                                                                                           \
		}                                                                                                                                                  \
	} else                                                                                                                                                 \
		((void)0)

/**
 * Prints a custom deprecation warning message without returning.
 * This should be preferred to `WARN_PRINT` for deprecation warnings.
 */
#define WARN_DEPRECATED_MSG(m_msg)                                                                                                                                \
	if (true) {                                                                                                                                                   \
		static SafeFlag warning_shown;                                                                                                                            \
		if (!warning_shown.is_set()) {                                                                                                                            \
			_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "This method has been deprecated and will be removed in the future.", m_msg, ERR_HANDLER_WARNING); \
			warning_shown.set();                                                                                                                                  \
		}                                                                                                                                                         \
	} else                                                                                                                                                        \
		((void)0)

#endif

/**
 * This should be a 'free' assert for program flow and should not be needed in any releases,
 *  only used in dev builds.
 */
#ifdef DEV_ENABLED
#define DEV_ASSERT(m_cond)                                                                                              \
	if (unlikely(!(m_cond))) {                                                                                          \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "FATAL: DEV_ASSERT failed  \"" _STR(m_cond) "\" is false."); \
		void _err_flush_stdout();                                                                                       \
		GENERATE_TRAP                                                                                                   \
	} else                                                                                                              \
		((void)0)
#else
#define DEV_ASSERT(m_cond)
#endif

/**
 * These should be 'free' checks for program flow and should not be needed in any releases,
 *  only used in dev builds.
 */
#ifdef DEV_ENABLED
#define DEV_CHECK(m_cond)                                              \
	if (unlikely(!(m_cond))) {                                         \
		ERR_PRINT("DEV_CHECK failed  \"" _STR(m_cond) "\" is false."); \
	} else                                                             \
		((void)0)
#else
#define DEV_CHECK(m_cond)
#endif

#ifdef DEV_ENABLED
#define DEV_CHECK_ONCE(m_cond)                                                   \
	if (unlikely(!(m_cond))) {                                                   \
		ERR_PRINT_ONCE("DEV_CHECK_ONCE failed  \"" _STR(m_cond) "\" is false."); \
	} else                                                                       \
		((void)0)
#else
#define DEV_CHECK_ONCE(m_cond)
#endif
