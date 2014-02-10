/* speex_types.h taken from libogg */
/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggVorbis SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE OggVorbis SOURCE CODE IS (C) COPYRIGHT 1994-2002             *
 * by the Xiph.Org Foundation http://www.xiph.org/                  *
 *                                                                  *
 ********************************************************************

 function: #ifdef jail to whip a few platforms into the UNIX ideal.
 last mod: $Id: os_types.h 7524 2004-08-11 04:20:36Z conrad $

 ********************************************************************/
/**
   @file speex_types.h
   @brief Speex types
*/
#ifndef _SPEEX_TYPES_H
#define _SPEEX_TYPES_H

#if defined(_WIN32) 

#  if defined(__CYGWIN__)
#    include <_G_config.h>
     typedef _G_int32_t spx_int32_t;
     typedef _G_uint32_t spx_uint32_t;
     typedef _G_int16_t spx_int16_t;
     typedef _G_uint16_t spx_uint16_t;
#  elif defined(__MINGW32__)
     typedef short spx_int16_t;
     typedef unsigned short spx_uint16_t;
     typedef int spx_int32_t;
     typedef unsigned int spx_uint32_t;
#  elif defined(__MWERKS__)
     typedef int spx_int32_t;
     typedef unsigned int spx_uint32_t;
     typedef short spx_int16_t;
     typedef unsigned short spx_uint16_t;
#  else
     /* MSVC/Borland */
     typedef __int32 spx_int32_t;
     typedef unsigned __int32 spx_uint32_t;
     typedef __int16 spx_int16_t;
     typedef unsigned __int16 spx_uint16_t;
#  endif

#elif defined(__MACOS__)

#  include <sys/types.h>
   typedef SInt16 spx_int16_t;
   typedef UInt16 spx_uint16_t;
   typedef SInt32 spx_int32_t;
   typedef UInt32 spx_uint32_t;

#elif (defined(__APPLE__) && defined(__MACH__)) /* MacOS X Framework build */

#  include <sys/types.h>
   typedef int16_t spx_int16_t;
   typedef u_int16_t spx_uint16_t;
   typedef int32_t spx_int32_t;
   typedef u_int32_t spx_uint32_t;

#elif defined(__BEOS__)

   /* Be */
#  include <inttypes.h>
   typedef int16_t spx_int16_t;
   typedef u_int16_t spx_uint16_t;
   typedef int32_t spx_int32_t;
   typedef u_int32_t spx_uint32_t;

#elif defined (__EMX__)

   /* OS/2 GCC */
   typedef short spx_int16_t;
   typedef unsigned short spx_uint16_t;
   typedef int spx_int32_t;
   typedef unsigned int spx_uint32_t;

#elif defined (DJGPP)

   /* DJGPP */
   typedef short spx_int16_t;
   typedef int spx_int32_t;
   typedef unsigned int spx_uint32_t;

#elif defined(R5900)

   /* PS2 EE */
   typedef int spx_int32_t;
   typedef unsigned spx_uint32_t;
   typedef short spx_int16_t;

#elif defined(__SYMBIAN32__)

   /* Symbian GCC */
   typedef signed short spx_int16_t;
   typedef unsigned short spx_uint16_t;
   typedef signed int spx_int32_t;
   typedef unsigned int spx_uint32_t;

#elif defined(CONFIG_TI_C54X) || defined (CONFIG_TI_C55X)

   typedef short spx_int16_t;
   typedef unsigned short spx_uint16_t;
   typedef long spx_int32_t;
   typedef unsigned long spx_uint32_t;

#elif defined(CONFIG_TI_C6X)

   typedef short spx_int16_t;
   typedef unsigned short spx_uint16_t;
   typedef int spx_int32_t;
   typedef unsigned int spx_uint32_t;

#else

#  include <speex/speex_config_types.h>

#endif

#endif  /* _SPEEX_TYPES_H */
