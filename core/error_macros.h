/*************************************************************************/
/*  error_macros.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "core/typedefs.h"
/**
 * Error macros. Unlike exceptions and asserts, these macros try to maintain consistency and stability
 * inside the code. It is recommended to always return processable data, so in case of an error, the
 * engine can stay working well.
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
		errfunc = 0;
		next = 0;
		userdata = 0;
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

// Used to strip debug messages in release mode
#ifdef DEBUG_ENABLED
#define DEBUG_STR(m_msg) m_msg
#else
#define DEBUG_STR(m_msg) ""
#endif

// (*): See https://stackoverflow.com/questions/257418/do-while-0-what-is-it-good-for

#define ERR_FAIL_INDEX(m_index, m_size)                                                                             \
	do {                                                                                                            \
		if (unlikely((m_index) < 0 || (m_index) >= (m_size))) {                                                     \
			_err_print_index_error(FUNCTION_STR, __FILE__, __LINE__, m_index, m_size, _STR(m_index), _STR(m_size)); \
			return;                                                                                                 \
		}                                                                                                           \
	} while (0); // (*)

#define ERR_FAIL_INDEX_MSG(m_index, m_size, m_msg)                                                                                    \
	do {                                                                                                                              \
		if (unlikely((m_index) < 0 || (m_index) >= (m_size))) {                                                                       \
			_err_print_index_error(FUNCTION_STR, __FILE__, __LINE__, m_index, m_size, _STR(m_index), _STR(m_size), DEBUG_STR(m_msg)); \
			return;                                                                                                                   \
		}                                                                                                                             \
	} while (0); // (*)

/** An index has failed if m_index<0 or m_index >=m_size, the function exits.
* This function returns an error value, if returning Error, please select the most
* appropriate error condition from error_macros.h
*/

#define ERR_FAIL_INDEX_V(m_index, m_size, m_retval)                                                                 \
	do {                                                                                                            \
		if (unlikely((m_index) < 0 || (m_index) >= (m_size))) {                                                     \
			_err_print_index_error(FUNCTION_STR, __FILE__, __LINE__, m_index, m_size, _STR(m_index), _STR(m_size)); \
			return m_retval;                                                                                        \
		}                                                                                                           \
	} while (0); // (*)

#define ERR_FAIL_INDEX_V_MSG(m_index, m_size, m_retval, m_msg)                                                                        \
	do {                                                                                                                              \
		if (unlikely((m_index) < 0 || (m_index) >= (m_size))) {                                                                       \
			_err_print_index_error(FUNCTION_STR, __FILE__, __LINE__, m_index, m_size, _STR(m_index), _STR(m_size), DEBUG_STR(m_msg)); \
			return m_retval;                                                                                                          \
		}                                                                                                                             \
	} while (0); // (*)

/** An index has failed if m_index >=m_size, the function exits.
* This function returns an error value, if returning Error, please select the most
* appropriate error condition from error_macros.h
*/

#define ERR_FAIL_UNSIGNED_INDEX_V(m_index, m_size, m_retval)                                                        \
	do {                                                                                                            \
		if (unlikely((m_index) >= (m_size))) {                                                                      \
			_err_print_index_error(FUNCTION_STR, __FILE__, __LINE__, m_index, m_size, _STR(m_index), _STR(m_size)); \
			return m_retval;                                                                                        \
		}                                                                                                           \
	} while (0); // (*)

#define ERR_FAIL_UNSIGNED_INDEX_V_MSG(m_index, m_size, m_retval, m_msg)                                                               \
	do {                                                                                                                              \
		if (unlikely((m_index) >= (m_size))) {                                                                                        \
			_err_print_index_error(FUNCTION_STR, __FILE__, __LINE__, m_index, m_size, _STR(m_index), _STR(m_size), DEBUG_STR(m_msg)); \
			return m_retval;                                                                                                          \
		}                                                                                                                             \
	} while (0); // (*)

/** Use this one if there is no sensible fallback, that is, the error is unrecoverable.
*   We'll return a null reference and try to keep running.
*/
#define CRASH_BAD_INDEX(m_index, m_size)                                                                                      \
	do {                                                                                                                      \
		if (unlikely((m_index) < 0 || (m_index) >= (m_size))) {                                                               \
			_err_print_index_error(FUNCTION_STR, __FILE__, __LINE__, m_index, m_size, _STR(m_index), _STR(m_size), "", true); \
			GENERATE_TRAP                                                                                                     \
		}                                                                                                                     \
	} while (0); // (*)

#define CRASH_BAD_INDEX_MSG(m_index, m_size, m_msg)                                                                              \
	do {                                                                                                                         \
		if (unlikely((m_index) < 0 || (m_index) >= (m_size))) {                                                                  \
			_err_print_index_error(FUNCTION_STR, __FILE__, __LINE__, m_index, m_size, _STR(m_index), _STR(m_size), m_msg, true); \
			GENERATE_TRAP                                                                                                        \
		}                                                                                                                        \
	} while (0); // (*)

/** An error condition happened (m_cond tested true) (WARNING this is the opposite as assert().
  * the function will exit.
  */

#define ERR_FAIL_NULL(m_param)                                                                              \
	{                                                                                                       \
		if (unlikely(!m_param)) {                                                                           \
			_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Parameter ' " _STR(m_param) " ' is null."); \
			return;                                                                                         \
		}                                                                                                   \
	}

#define ERR_FAIL_NULL_MSG(m_param, m_msg)                                                                                     \
	{                                                                                                                         \
		if (unlikely(!m_param)) {                                                                                             \
			_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Parameter ' " _STR(m_param) " ' is null.", DEBUG_STR(m_msg)); \
			return;                                                                                                           \
		}                                                                                                                     \
	}

#define ERR_FAIL_NULL_V(m_param, m_retval)                                                                  \
	{                                                                                                       \
		if (unlikely(!m_param)) {                                                                           \
			_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Parameter ' " _STR(m_param) " ' is null."); \
			return m_retval;                                                                                \
		}                                                                                                   \
	}

#define ERR_FAIL_NULL_V_MSG(m_param, m_retval, m_msg)                                                                         \
	{                                                                                                                         \
		if (unlikely(!m_param)) {                                                                                             \
			_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Parameter ' " _STR(m_param) " ' is null.", DEBUG_STR(m_msg)); \
			return m_retval;                                                                                                  \
		}                                                                                                                     \
	}

/** An error condition happened (m_cond tested true) (WARNING this is the opposite as assert().
 * the function will exit.
 */

#define ERR_FAIL_COND(m_cond)                                                                              \
	{                                                                                                      \
		if (unlikely(m_cond)) {                                                                            \
			_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Condition ' " _STR(m_cond) " ' is true."); \
			return;                                                                                        \
		}                                                                                                  \
	}

#define ERR_FAIL_COND_MSG(m_cond, m_msg)                                                                                     \
	{                                                                                                                        \
		if (unlikely(m_cond)) {                                                                                              \
			_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Condition ' " _STR(m_cond) " ' is true.", DEBUG_STR(m_msg)); \
			return;                                                                                                          \
		}                                                                                                                    \
	}

/** Use this one if there is no sensible fallback, that is, the error is unrecoverable.
 */

#define CRASH_COND(m_cond)                                                                                        \
	{                                                                                                             \
		if (unlikely(m_cond)) {                                                                                   \
			_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "FATAL: Condition ' " _STR(m_cond) " ' is true."); \
			GENERATE_TRAP                                                                                         \
		}                                                                                                         \
	}

#define CRASH_COND_MSG(m_cond, m_msg)                                                                                               \
	{                                                                                                                               \
		if (unlikely(m_cond)) {                                                                                                     \
			_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "FATAL: Condition ' " _STR(m_cond) " ' is true.", DEBUG_STR(m_msg)); \
			GENERATE_TRAP                                                                                                           \
		}                                                                                                                           \
	}

/** An error condition happened (m_cond tested true) (WARNING this is the opposite as assert().
 * the function will exit.
 * This function returns an error value, if returning Error, please select the most
 * appropriate error condition from error_macros.h
 */

#define ERR_FAIL_COND_V(m_cond, m_retval)                                                                                            \
	{                                                                                                                                \
		if (unlikely(m_cond)) {                                                                                                      \
			_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Condition ' " _STR(m_cond) " ' is true. returned: " _STR(m_retval)); \
			return m_retval;                                                                                                         \
		}                                                                                                                            \
	}

#define ERR_FAIL_COND_V_MSG(m_cond, m_retval, m_msg)                                                                                                   \
	{                                                                                                                                                  \
		if (unlikely(m_cond)) {                                                                                                                        \
			_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Condition ' " _STR(m_cond) " ' is true. returned: " _STR(m_retval), DEBUG_STR(m_msg)); \
			return m_retval;                                                                                                                           \
		}                                                                                                                                              \
	}

/** An error condition happened (m_cond tested true) (WARNING this is the opposite as assert().
 * the loop will skip to the next iteration.
 */

#define ERR_CONTINUE(m_cond)                                                                                             \
	{                                                                                                                    \
		if (unlikely(m_cond)) {                                                                                          \
			_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Condition ' " _STR(m_cond) " ' is true. Continuing..:"); \
			continue;                                                                                                    \
		}                                                                                                                \
	}

#define ERR_CONTINUE_MSG(m_cond, m_msg)                                                                                                    \
	{                                                                                                                                      \
		if (unlikely(m_cond)) {                                                                                                            \
			_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Condition ' " _STR(m_cond) " ' is true. Continuing..:", DEBUG_STR(m_msg)); \
			continue;                                                                                                                      \
		}                                                                                                                                  \
	}

/** An error condition happened (m_cond tested true) (WARNING this is the opposite as assert().
 * the loop will break
 */

#define ERR_BREAK(m_cond)                                                                                              \
	{                                                                                                                  \
		if (unlikely(m_cond)) {                                                                                        \
			_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Condition ' " _STR(m_cond) " ' is true. Breaking..:"); \
			break;                                                                                                     \
		}                                                                                                              \
	}

#define ERR_BREAK_MSG(m_cond, m_msg)                                                                                                     \
	{                                                                                                                                    \
		if (unlikely(m_cond)) {                                                                                                          \
			_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Condition ' " _STR(m_cond) " ' is true. Breaking..:", DEBUG_STR(m_msg)); \
			break;                                                                                                                       \
		}                                                                                                                                \
	}

/** Print an error string and return
 */

#define ERR_FAIL()                                                                     \
	{                                                                                  \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Method/Function Failed."); \
		return;                                                                        \
	}

#define ERR_FAIL_MSG(m_msg)                                                                              \
	{                                                                                                    \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Method/Function Failed.", DEBUG_STR(m_msg)); \
		return;                                                                                          \
	}

/** Print an error string and return with value
 */

#define ERR_FAIL_V(m_value)                                                                                       \
	{                                                                                                             \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Method/Function Failed, returning: " __STR(m_value)); \
		return m_value;                                                                                           \
	}

#define ERR_FAIL_V_MSG(m_value, m_msg)                                                                                              \
	{                                                                                                                               \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Method/Function Failed, returning: " __STR(m_value), DEBUG_STR(m_msg)); \
		return m_value;                                                                                                             \
	}

/** Use this one if there is no sensible fallback, that is, the error is unrecoverable.
 */

#define CRASH_NOW()                                                                           \
	{                                                                                         \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "FATAL: Method/Function Failed."); \
		GENERATE_TRAP                                                                         \
	}

#define CRASH_NOW_MSG(m_msg)                                                                                    \
	{                                                                                                           \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "FATAL: Method/Function Failed.", DEBUG_STR(m_msg)); \
		GENERATE_TRAP                                                                                           \
	}

/** Print an error string.
 */

#define ERR_PRINT(m_string)                                           \
	{                                                                 \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, m_string); \
	}

#define ERR_PRINTS(m_string)                                          \
	{                                                                 \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, m_string); \
	}

#define ERR_PRINT_ONCE(m_string)                                          \
	{                                                                     \
		static bool first_print = true;                                   \
		if (first_print) {                                                \
			_err_print_error(FUNCTION_STR, __FILE__, __LINE__, m_string); \
			first_print = false;                                          \
		}                                                                 \
	}

/** Print a warning string.
 */

#define WARN_PRINT(m_string)                                                               \
	{                                                                                      \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, m_string, ERR_HANDLER_WARNING); \
	}

#define WARN_PRINTS(m_string)                                                              \
	{                                                                                      \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, m_string, ERR_HANDLER_WARNING); \
	}

#define WARN_PRINT_ONCE(m_string)                                                              \
	{                                                                                          \
		static bool first_print = true;                                                        \
		if (first_print) {                                                                     \
			_err_print_error(FUNCTION_STR, __FILE__, __LINE__, m_string, ERR_HANDLER_WARNING); \
			first_print = false;                                                               \
		}                                                                                      \
	}

#define WARN_DEPRECATED                                                                                                                                   \
	{                                                                                                                                                     \
		static volatile bool warning_shown = false;                                                                                                       \
		if (!warning_shown) {                                                                                                                             \
			_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "This method has been deprecated and will be removed in the future", ERR_HANDLER_WARNING); \
			warning_shown = true;                                                                                                                         \
		}                                                                                                                                                 \
	}

#define WARN_DEPRECATED_MSG(m_msg)                                                                                                                               \
	{                                                                                                                                                            \
		static volatile bool warning_shown = false;                                                                                                              \
		if (!warning_shown) {                                                                                                                                    \
			_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "This method has been deprecated and will be removed in the future", m_msg, ERR_HANDLER_WARNING); \
			warning_shown = true;                                                                                                                                \
		}                                                                                                                                                        \
	}

#endif
