/**************************************************************************
 *
 * Copyright 2008-2010 VMware, Inc.
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


#include "hash_table.h"
#include "macros.h"
#include "os_misc.h"
#include "os_file.h"
#include "ralloc.h"
#include "simple_mtx.h"

#include <stdarg.h>


#if DETECT_OS_WINDOWS

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

#else

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#endif


#if DETECT_OS_ANDROID
#  define LOG_TAG "MESA"
#  include <unistd.h>
#  include <log/log.h>
#  include <cutils/properties.h>
#elif DETECT_OS_LINUX || DETECT_OS_CYGWIN || DETECT_OS_SOLARIS || DETECT_OS_HURD
#  include <unistd.h>
#elif DETECT_OS_OPENBSD || DETECT_OS_FREEBSD
#  include <sys/resource.h>
#  include <sys/sysctl.h>
#elif DETECT_OS_APPLE || DETECT_OS_BSD
#  include <sys/sysctl.h>
#elif DETECT_OS_HAIKU
#  include <kernel/OS.h>
#elif DETECT_OS_WINDOWS
#  include <windows.h>
#else
#error unexpected platform in os_sysinfo.c
#endif


void
os_log_message(const char *message)
{
   /* If the GALLIUM_LOG_FILE environment variable is set to a valid filename,
    * write all messages to that file.
    */
   static FILE *fout = NULL;

   if (!fout) {
#ifdef DEBUG
      /* one-time init */
      const char *filename = os_get_option("GALLIUM_LOG_FILE");
      if (filename) {
         if (strcmp(filename, "stdout") == 0) {
            fout = stdout;
         } else {
            const char *mode = "w";
            if (filename[0] == '+') {
               /* If the filename is prefixed with '+' then open the file for
                * appending instead of normal writing.
                */
               mode = "a";
            filename++; /* skip the '+' */
            }
            fout = fopen(filename, mode);
         }
      }
#endif
      if (!fout)
         fout = stderr;
   }

#if DETECT_OS_WINDOWS
   OutputDebugStringA(message);
#if !defined(_GAMING_XBOX)
   if(GetConsoleWindow() && !IsDebuggerPresent()) {
      fflush(stdout);
      fputs(message, fout);
      fflush(fout);
   }
   else if (fout != stderr) {
      fputs(message, fout);
      fflush(fout);
   }
#endif
#else /* !DETECT_OS_WINDOWS */
   fflush(stdout);
   fputs(message, fout);
   fflush(fout);
#  if DETECT_OS_ANDROID
   LOG_PRI(ANDROID_LOG_ERROR, LOG_TAG, "%s", message);
#  endif
#endif
}

#if DETECT_OS_ANDROID
#  include <ctype.h>
#  include "c11/threads.h"

/**
 * Get an option value from android's property system, as a fallback to
 * getenv() (which is generally less useful on android due to processes
 * typically being forked from the zygote.
 *
 * The option name used for getenv is translated into a property name
 * by:
 *
 *  1) convert to lowercase
 *  2) replace '_' with '.'
 *  3) if necessary, prepend "mesa."
 *
 * For example:
 *  - MESA_EXTENSION_OVERRIDE -> mesa.extension.override
 *  - GALLIUM_HUD -> mesa.gallium.hud
 *
 */
static char *
os_get_android_option(const char *name)
{
   static thread_local char os_android_option_value[PROPERTY_VALUE_MAX];
   char key[PROPERTY_KEY_MAX];
   char *p = key, *end = key + PROPERTY_KEY_MAX;
   /* add "mesa." prefix if necessary: */
   if (strstr(name, "MESA_") != name)
      p += strlcpy(p, "mesa.", end - p);
   p += strlcpy(p, name, end - p);
   for (int i = 0; key[i]; i++) {
      if (key[i] == '_') {
         key[i] = '.';
      } else {
         key[i] = tolower(key[i]);
      }
   }

   int len = property_get(key, os_android_option_value, NULL);
   if (len > 1) {
      return os_android_option_value;
   }
   return NULL;
}
#endif

const char *
os_get_option(const char *name)
{
   const char *opt = getenv(name);
#if DETECT_OS_ANDROID
   if (!opt) {
      opt = os_get_android_option(name);
   }
#endif
   return opt;
}

static struct hash_table *options_tbl;
static bool options_tbl_exited = false;
static simple_mtx_t options_tbl_mtx = SIMPLE_MTX_INITIALIZER;

/**
 * NOTE: The strings that allocated with ralloc_strdup(options_tbl, ...)
 * are freed by _mesa_hash_table_destroy automatically
 */
static void
options_tbl_fini(void)
{
   simple_mtx_lock(&options_tbl_mtx);
   _mesa_hash_table_destroy(options_tbl, NULL);
   options_tbl = NULL;
   options_tbl_exited = true;
   simple_mtx_unlock(&options_tbl_mtx);
}

const char *
os_get_option_cached(const char *name)
{
   const char *opt = NULL;
   simple_mtx_lock(&options_tbl_mtx);
   if (options_tbl_exited) {
      opt = os_get_option(name);
      goto exit_mutex;
   }

   if (!options_tbl) {
      options_tbl = _mesa_hash_table_create(NULL, _mesa_hash_string,
            _mesa_key_string_equal);
      if (options_tbl == NULL) {
         goto exit_mutex;
      }
      atexit(options_tbl_fini);
   }
   struct hash_entry *entry = _mesa_hash_table_search(options_tbl, name);
   if (entry) {
      opt = entry->data;
      goto exit_mutex;
   }

   char *name_dup = ralloc_strdup(options_tbl, name);
   if (name_dup == NULL) {
      goto exit_mutex;
   }
   opt = ralloc_strdup(options_tbl, os_get_option(name));
   _mesa_hash_table_insert(options_tbl, name_dup, (void *)opt);
exit_mutex:
   simple_mtx_unlock(&options_tbl_mtx);
   return opt;
}

/**
 * Return the size of the total physical memory.
 * \param size returns the size of the total physical memory
 * \return true for success, or false on failure
 */
bool
os_get_total_physical_memory(uint64_t *size)
{
#if DETECT_OS_LINUX || DETECT_OS_CYGWIN || DETECT_OS_SOLARIS || DETECT_OS_HURD
   const long phys_pages = sysconf(_SC_PHYS_PAGES);
   const long page_size = sysconf(_SC_PAGE_SIZE);

   if (phys_pages <= 0 || page_size <= 0)
      return false;

   *size = (uint64_t)phys_pages * (uint64_t)page_size;
   return true;
#elif DETECT_OS_APPLE || DETECT_OS_BSD
   size_t len = sizeof(*size);
   int mib[2];

   mib[0] = CTL_HW;
#if DETECT_OS_APPLE
   mib[1] = HW_MEMSIZE;
#elif DETECT_OS_NETBSD || DETECT_OS_OPENBSD
   mib[1] = HW_PHYSMEM64;
#elif DETECT_OS_FREEBSD
   mib[1] = HW_REALMEM;
#elif DETECT_OS_DRAGONFLY
   mib[1] = HW_PHYSMEM;
#else
#error Unsupported *BSD
#endif

   return (sysctl(mib, 2, size, &len, NULL, 0) == 0);
#elif DETECT_OS_HAIKU
   system_info info;
   status_t ret;

   ret = get_system_info(&info);
   if (ret != B_OK || info.max_pages <= 0)
      return false;

   *size = (uint64_t)info.max_pages * (uint64_t)B_PAGE_SIZE;
   return true;
#elif DETECT_OS_WINDOWS
   MEMORYSTATUSEX status;
   BOOL ret;

   status.dwLength = sizeof(status);
   ret = GlobalMemoryStatusEx(&status);
   *size = status.ullTotalPhys;
   return (ret == TRUE);
#else
#error unexpected platform in os_misc.c
   return false;
#endif
}

bool
os_get_available_system_memory(uint64_t *size)
{
#if DETECT_OS_LINUX
   char *meminfo = os_read_file("/proc/meminfo", NULL);
   if (!meminfo)
      return false;

   char *str = strstr(meminfo, "MemAvailable:");
   if (!str) {
      free(meminfo);
      return false;
   }

   uint64_t kb_mem_available;
   if (sscanf(str, "MemAvailable: %" PRIu64, &kb_mem_available) == 1) {
      free(meminfo);
      *size = kb_mem_available << 10;
      return true;
   }

   free(meminfo);
   return false;
#elif DETECT_OS_OPENBSD || DETECT_OS_FREEBSD
   struct rlimit rl;
#if DETECT_OS_OPENBSD
   int mib[] = { CTL_HW, HW_USERMEM64 };
#elif DETECT_OS_FREEBSD
   int mib[] = { CTL_HW, HW_USERMEM };
#endif
   int64_t mem_available;
   size_t len = sizeof(mem_available);

   /* physmem - wired */
   if (sysctl(mib, 2, &mem_available, &len, NULL, 0) == -1)
      return false;

   /* static login.conf limit */
   if (getrlimit(RLIMIT_DATA, &rl) == -1)
      return false;

   *size = MIN2(mem_available, rl.rlim_cur);
   return true;
#elif DETECT_OS_WINDOWS
   MEMORYSTATUSEX status;
   BOOL ret;

   status.dwLength = sizeof(status);
   ret = GlobalMemoryStatusEx(&status);
   *size = status.ullAvailPhys;
   return (ret == TRUE);
#else
   return false;
#endif
}

/**
 * Return the size of a page
 * \param size returns the size of a page
 * \return true for success, or false on failure
 */
bool
os_get_page_size(uint64_t *size)
{
#if DETECT_OS_UNIX && !DETECT_OS_APPLE && !DETECT_OS_HAIKU
   const long page_size = sysconf(_SC_PAGE_SIZE);

   if (page_size <= 0)
      return false;

   *size = (uint64_t)page_size;
   return true;
#elif DETECT_OS_HAIKU
   *size = (uint64_t)B_PAGE_SIZE;
   return true;
#elif DETECT_OS_WINDOWS
   SYSTEM_INFO SysInfo;

   GetSystemInfo(&SysInfo);
   *size = SysInfo.dwPageSize;
   return true;
#elif DETECT_OS_APPLE
   size_t len = sizeof(*size);
   int mib[2];

   mib[0] = CTL_HW;
   mib[1] = HW_PAGESIZE;
   return (sysctl(mib, 2, size, &len, NULL, 0) == 0);
#else
#error unexpected platform in os_sysinfo.c
   return false;
#endif
}
