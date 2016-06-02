
/* arm_init.c - NEON optimised filter functions
 *
 * Copyright (c) 2013 Glenn Randers-Pehrson
 * Written by Mans Rullgard, 2011.
 * Last changed in libpng 1.6.8 [December 19, 2013]
 *
 * This code is released under the libpng license.
 * For conditions of distribution and use, see the disclaimer
 * and license in png.h
 */
/* Below, after checking __linux__, various non-C90 POSIX 1003.1 functions are
 * called.
 */
#define _POSIX_SOURCE 1

#include "../pngpriv.h"

#ifdef PNG_READ_SUPPORTED
#if PNG_ARM_NEON_OPT > 0
#ifdef PNG_ARM_NEON_CHECK_SUPPORTED /* Do run-time checks */
#include <signal.h> /* for sig_atomic_t */

#ifdef __ANDROID__
/* Linux provides access to information about CPU capabilites via
 * /proc/self/auxv, however Android blocks this while still claiming to be
 * Linux.  The Andoid NDK, however, provides appropriate support.
 *
 * Documentation: http://www.kandroid.org/ndk/docs/CPU-ARM-NEON.html
 */
#include <cpu-features.h>

static int
png_have_neon(png_structp png_ptr)
{
   /* This is a whole lot easier than the mess below, however it is probably
    * implemented as below, therefore it is better to cache the result (these
    * function calls may be slow!)
    */
   PNG_UNUSED(png_ptr)
   return android_getCpuFamily() == ANDROID_CPU_FAMILY_ARM &&
      (android_getCpuFeatures() & ANDROID_CPU_ARM_FEATURE_NEON) != 0;
}
#elif defined(__linux__)
/* The generic __linux__ implementation requires reading /proc/self/auxv and
 * looking at each element for one that records NEON capabilities.
 */
#include <unistd.h> /* for POSIX 1003.1 */
#include <errno.h>  /* for EINTR */

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <elf.h>
#include <asm/hwcap.h>

/* A read call may be interrupted, in which case it returns -1 and sets errno to
 * EINTR if nothing was done, otherwise (if something was done) a partial read
 * may result.
 */
static size_t
safe_read(png_structp png_ptr, int fd, void *buffer_in, size_t nbytes)
{
   size_t ntotal = 0;
   char *buffer = png_voidcast(char*, buffer_in);

   while (nbytes > 0)
   {
      unsigned int nread;
      int iread;

      /* Passing nread > INT_MAX to read is implementation defined in POSIX
       * 1003.1, therefore despite the unsigned argument portable code must
       * limit the value to INT_MAX!
       */
      if (nbytes > INT_MAX)
         nread = INT_MAX;

      else
         nread = (unsigned int)/*SAFE*/nbytes;

      iread = read(fd, buffer, nread);

      if (iread == -1)
      {
         /* This is the devil in the details, a read can terminate early with 0
          * bytes read because of EINTR, yet it still returns -1 otherwise end
          * of file cannot be distinguished.
          */
         if (errno != EINTR)
         {
            png_warning(png_ptr, "/proc read failed");
            return 0; /* I.e., a permanent failure */
         }
      }

      else if (iread < 0)
      {
         /* Not a valid 'read' result: */
         png_warning(png_ptr, "OS /proc read bug");
         return 0;
      }

      else if (iread > 0)
      {
         /* Continue reading until a permanent failure, or EOF */
         buffer += iread;
         nbytes -= (unsigned int)/*SAFE*/iread;
         ntotal += (unsigned int)/*SAFE*/iread;
      }

      else
         return ntotal;
   }

   return ntotal; /* nbytes == 0 */
}

static int
png_have_neon(png_structp png_ptr)
{
   int fd = open("/proc/self/auxv", O_RDONLY);
   Elf32_auxv_t aux;

   /* Failsafe: failure to open means no NEON */
   if (fd == -1)
   {
      png_warning(png_ptr, "/proc/self/auxv open failed");
      return 0;
   }

   while (safe_read(png_ptr, fd, &aux, sizeof aux) == sizeof aux)
   {
      if (aux.a_type == AT_HWCAP && (aux.a_un.a_val & HWCAP_NEON) != 0)
      {
         close(fd);
         return 1;
      }
   }

   close(fd);
   return 0;
}
#else
   /* We don't know how to do a run-time check on this system */
#  error "no support for run-time ARM NEON checks"
#endif /* OS checks */
#endif /* PNG_ARM_NEON_CHECK_SUPPORTED */

#ifndef PNG_ALIGNED_MEMORY_SUPPORTED
#  error "ALIGNED_MEMORY is required; set: -DPNG_ALIGNED_MEMORY_SUPPORTED"
#endif

void
png_init_filter_functions_neon(png_structp pp, unsigned int bpp)
{
   /* The switch statement is compiled in for ARM_NEON_API, the call to
    * png_have_neon is compiled in for ARM_NEON_CHECK.  If both are defined
    * the check is only performed if the API has not set the NEON option on
    * or off explicitly.  In this case the check controls what happens.
    *
    * If the CHECK is not compiled in and the option is UNSET the behavior prior
    * to 1.6.7 was to use the NEON code - this was a bug caused by having the
    * wrong order of the 'ON' and 'default' cases.  UNSET now defaults to OFF,
    * as documented in png.h
    */
#ifdef PNG_ARM_NEON_API_SUPPORTED
   switch ((pp->options >> PNG_ARM_NEON) & 3)
   {
      case PNG_OPTION_UNSET:
         /* Allow the run-time check to execute if it has been enabled -
          * thus both API and CHECK can be turned on.  If it isn't supported
          * this case will fall through to the 'default' below, which just
          * returns.
          */
#endif /* PNG_ARM_NEON_API_SUPPORTED */
#ifdef PNG_ARM_NEON_CHECK_SUPPORTED
         {
            static volatile sig_atomic_t no_neon = -1; /* not checked */

            if (no_neon < 0)
               no_neon = !png_have_neon(pp);

            if (no_neon)
               return;
         }
#ifdef PNG_ARM_NEON_API_SUPPORTED
         break;
#endif
#endif /* PNG_ARM_NEON_CHECK_SUPPORTED */

#ifdef PNG_ARM_NEON_API_SUPPORTED
      default: /* OFF or INVALID */
         return;

      case PNG_OPTION_ON:
         /* Option turned on */
         break;
   }
#endif

   /* IMPORTANT: any new external functions used here must be declared using
    * PNG_INTERNAL_FUNCTION in ../pngpriv.h.  This is required so that the
    * 'prefix' option to configure works:
    *
    *    ./configure --with-libpng-prefix=foobar_
    *
    * Verify you have got this right by running the above command, doing a build
    * and examining pngprefix.h; it must contain a #define for every external
    * function you add.  (Notice that this happens automatically for the
    * initialization function.)
    */
   pp->read_filter[PNG_FILTER_VALUE_UP-1] = png_read_filter_row_up_neon;

   if (bpp == 3)
   {
      pp->read_filter[PNG_FILTER_VALUE_SUB-1] = png_read_filter_row_sub3_neon;
      pp->read_filter[PNG_FILTER_VALUE_AVG-1] = png_read_filter_row_avg3_neon;
      pp->read_filter[PNG_FILTER_VALUE_PAETH-1] =
         png_read_filter_row_paeth3_neon;
   }

   else if (bpp == 4)
   {
      pp->read_filter[PNG_FILTER_VALUE_SUB-1] = png_read_filter_row_sub4_neon;
      pp->read_filter[PNG_FILTER_VALUE_AVG-1] = png_read_filter_row_avg4_neon;
      pp->read_filter[PNG_FILTER_VALUE_PAETH-1] =
          png_read_filter_row_paeth4_neon;
   }
}
#endif /* PNG_ARM_NEON_OPT > 0 */
#endif /* PNG_READ_SUPPORTED */
