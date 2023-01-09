/**************************************************************************
 *
 * Copyright 2008 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

/**
 * @file
 * Cross-platform debugging helpers.
 *
 * For now it just has assert and printf replacements, but it might be extended
 * with stack trace reports and more advanced logging in the near future.
 *
 * @author Jose Fonseca <jfonseca@vmware.com>
 */

#ifndef U_DEBUG_H_
#define U_DEBUG_H_

#include <stdarg.h>
#include <string.h>
#if !defined(_WIN32)
#include <sys/types.h>
#include <unistd.h>
#endif

#include "util/os_misc.h"
#include "util/u_atomic.h"
#include "util/detect_os.h"
#include "util/macros.h"

#if DETECT_OS_HAIKU
/* Haiku provides debug_printf in libroot with OS.h */
#include <OS.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum util_debug_type
{
   UTIL_DEBUG_TYPE_OUT_OF_MEMORY = 1,
   UTIL_DEBUG_TYPE_ERROR,
   UTIL_DEBUG_TYPE_SHADER_INFO,
   UTIL_DEBUG_TYPE_PERF_INFO,
   UTIL_DEBUG_TYPE_INFO,
   UTIL_DEBUG_TYPE_FALLBACK,
   UTIL_DEBUG_TYPE_CONFORMANCE,
};

/**
 * Structure that contains a callback for debug messages from the driver back
 * to the gallium frontend.
 */
struct util_debug_callback
{
   /**
    * When set to \c true, the callback may be called asynchronously from a
    * driver-created thread.
    */
   bool async;

   /**
    * Callback for the driver to report debug/performance/etc information back
    * to the gallium frontend.
    *
    * \param data       user-supplied data pointer
    * \param id         message type identifier, if pointed value is 0, then a
    *                   new id is assigned
    * \param type       UTIL_DEBUG_TYPE_*
    * \param format     printf-style format string
    * \param args       args for format string
    */
   void (*debug_message)(void *data,
                         unsigned *id,
                         enum util_debug_type type,
                         const char *fmt,
                         va_list args);
   void *data;
};

#define _util_printf_format(fmt, list) PRINTFLIKE(fmt, list)

void _debug_vprintf(const char *format, va_list ap);


static inline void
_debug_printf(const char *format, ...)
{
   va_list ap;
   va_start(ap, format);
   _debug_vprintf(format, ap);
   va_end(ap);
}


/**
 * Print debug messages.
 *
 * The actual channel used to output debug message is platform specific. To
 * avoid misformating or truncation, follow these rules of thumb:
 * - output whole lines
 * - avoid outputing large strings (512 bytes is the current maximum length
 * that is guaranteed to be printed in all platforms)
 */
#if !DETECT_OS_HAIKU
static inline void
debug_printf(const char *format, ...) _util_printf_format(1,2);

static inline void
debug_printf(const char *format, ...)
{
#ifdef DEBUG
   va_list ap;
   va_start(ap, format);
   _debug_vprintf(format, ap);
   va_end(ap);
#else
   (void) format; /* silence warning */
#endif
}
#endif


/*
 * ... isn't portable so we need to pass arguments in parentheses.
 *
 * usage:
 *    debug_printf_once(("answer: %i\n", 42));
 */
#define debug_printf_once(args) \
   do { \
      static bool once = true; \
      if (once) { \
         once = false; \
         debug_printf args; \
      } \
   } while (0)


#ifdef DEBUG
#define debug_vprintf(_format, _ap) _debug_vprintf(_format, _ap)
#else
#define debug_vprintf(_format, _ap) ((void)0)
#endif

#ifdef _WIN32
/**
 * Disable Win32 interactive error message boxes.
 *
 * Should be called as soon as possible for effectiveness.
 */
void
debug_disable_win32_error_dialogs(void);
#endif


/**
 * Hard-coded breakpoint.
 */
#ifdef DEBUG
#define debug_break() os_break()
#else /* !DEBUG */
#define debug_break() ((void)0)
#endif /* !DEBUG */


void
debug_get_version_option(const char *name, unsigned *major, unsigned *minor);


/**
 * Output the current function name.
 */
#ifdef DEBUG
#define debug_checkpoint() \
   _debug_printf("%s\n", __func__)
#else
#define debug_checkpoint() \
   ((void)0)
#endif


/**
 * Output the full source code position.
 */
#ifdef DEBUG
#define debug_checkpoint_full() \
   _debug_printf("%s:%u:%s\n", __FILE__, __LINE__, __func__)
#else
#define debug_checkpoint_full() \
   ((void)0)
#endif


/**
 * Output a warning message. Muted on release version.
 */
#ifdef DEBUG
#define debug_warning(__msg) \
   _debug_printf("%s:%u:%s: warning: %s\n", __FILE__, __LINE__, __func__, __msg)
#else
#define debug_warning(__msg) \
   ((void)0)
#endif


/**
 * Emit a warning message, but only once.
 */
#ifdef DEBUG
#define debug_warn_once(__msg) \
   do { \
      static bool warned = false; \
      if (!warned) { \
         _debug_printf("%s:%u:%s: one time warning: %s\n", \
                       __FILE__, __LINE__, __func__, __msg); \
         warned = true; \
      } \
   } while (0)
#else
#define debug_warn_once(__msg) \
   ((void)0)
#endif


/**
 * Output an error message. Not muted on release version.
 */
#ifdef DEBUG
#define debug_error(__msg) \
   _debug_printf("%s:%u:%s: error: %s\n", __FILE__, __LINE__, __func__, __msg)
#else
#define debug_error(__msg) \
   _debug_printf("error: %s\n", __msg)
#endif

/**
 * Output a debug log message to the debug info callback.
 */
#define util_debug_message(cb, type, fmt, ...) do { \
   static unsigned id = 0; \
   if ((cb) && (cb)->debug_message) { \
      _util_debug_message(cb, &id, \
                          UTIL_DEBUG_TYPE_ ## type, \
                          fmt, ##__VA_ARGS__); \
   } \
} while (0)

void
_util_debug_message(
   struct util_debug_callback *cb,
   unsigned *id,
   enum util_debug_type type,
   const char *fmt, ...) _util_printf_format(4, 5);


/**
 * Used by debug_dump_enum and debug_dump_flags to describe symbols.
 */
struct debug_named_value
{
   const char *name;
   uint64_t value;
   const char *desc;
};


/**
 * Some C pre-processor magic to simplify creating named values.
 *
 * Example:
 * @code
 * static const debug_named_value my_names[] = {
 *    DEBUG_NAMED_VALUE(MY_ENUM_VALUE_X),
 *    DEBUG_NAMED_VALUE(MY_ENUM_VALUE_Y),
 *    DEBUG_NAMED_VALUE(MY_ENUM_VALUE_Z),
 *    DEBUG_NAMED_VALUE_END
 * };
 *
 *    ...
 *    debug_printf("%s = %s\n",
 *                 name,
 *                 debug_dump_enum(my_names, my_value));
 *    ...
 * @endcode
 */
#define DEBUG_NAMED_VALUE(__symbol) {#__symbol, (uint64_t)__symbol, NULL}
#define DEBUG_NAMED_VALUE_WITH_DESCRIPTION(__symbol, __desc) {#__symbol, (uint64_t)__symbol, __desc}
#define DEBUG_NAMED_VALUE_END {NULL, 0, NULL}


/**
 * Convert a enum value to a string.
 */
const char *
debug_dump_enum(const struct debug_named_value *names,
                uint64_t value);

/**
 * Convert binary flags value to a string.
 */
const char *
debug_dump_flags(const struct debug_named_value *names,
                 uint64_t value);


struct debug_control {
    const char * string;
    uint64_t     flag;
};

uint64_t
parse_debug_string(const char *debug,
                   const struct debug_control *control);


uint64_t
parse_enable_string(const char *debug,
                    uint64_t default_value,
                    const struct debug_control *control);


bool
comma_separated_list_contains(const char *list, const char *s);

/**
 * Get option.
 *
 * It is an alias for getenv on Unix and Windows.
 *
 */
const char *
debug_get_option(const char *name, const char *dfault);

const char *
debug_get_option_cached(const char *name, const char *dfault);

bool
debug_parse_bool_option(const char *str, bool dfault);

bool
debug_get_bool_option(const char *name, bool dfault);

int64_t
debug_parse_num_option(const char *str, int64_t dfault);

int64_t
debug_get_num_option(const char *name, int64_t dfault);

uint64_t
debug_parse_flags_option(const char *name,
                         const char *str,
                         const struct debug_named_value *flags,
                         uint64_t dfault);

uint64_t
debug_get_flags_option(const char *name,
                       const struct debug_named_value *flags,
                       uint64_t dfault);

#define DEBUG_GET_ONCE_OPTION(suffix, name, dfault) \
static const char * \
debug_get_option_ ## suffix (void) \
{ \
   static bool initialized = false; \
   static const char * value; \
   if (unlikely(!p_atomic_read_relaxed(&initialized))) { \
      const char *str = debug_get_option_cached(name, dfault); \
      p_atomic_set(&value, str); \
      p_atomic_set(&initialized, true); \
   } \
   return value; \
}

static inline bool
__check_suid(void)
{
#if !defined(_WIN32)
   if (geteuid() != getuid())
      return true;
#endif
   return false;
}

#define DEBUG_GET_ONCE_BOOL_OPTION(sufix, name, dfault) \
static bool \
debug_get_option_ ## sufix (void) \
{ \
   static bool initialized = false; \
   static bool value; \
   if (unlikely(!p_atomic_read_relaxed(&initialized))) { \
      const char *str = debug_get_option_cached(name, NULL); \
      bool parsed_value = debug_parse_bool_option(str, dfault); \
      p_atomic_set(&value, parsed_value); \
      p_atomic_set(&initialized, true); \
   } \
   return value; \
}

#define DEBUG_GET_ONCE_NUM_OPTION(sufix, name, dfault) \
static int64_t \
debug_get_option_ ## sufix (void) \
{ \
   static bool initialized = false; \
   static int64_t value; \
   if (unlikely(!p_atomic_read_relaxed(&initialized))) { \
      const char *str = debug_get_option_cached(name, NULL); \
      int64_t parsed_value = debug_parse_num_option(str, dfault); \
      p_atomic_set(&value, parsed_value); \
      p_atomic_set(&initialized, true); \
   } \
   return value; \
}

#define DEBUG_GET_ONCE_FLAGS_OPTION(sufix, name, flags, dfault) \
static uint64_t \
debug_get_option_ ## sufix (void) \
{ \
   static bool initialized = false; \
   static uint64_t value; \
   if (unlikely(!p_atomic_read_relaxed(&initialized))) { \
      const char *str = debug_get_option_cached(name, NULL); \
      uint64_t parsed_value = debug_parse_flags_option(name, str, flags, dfault); \
      p_atomic_set(&value, parsed_value); \
      p_atomic_set(&initialized, true); \
   } \
   return value; \
}


#ifdef __cplusplus
}
#endif

#endif /* U_DEBUG_H_ */
