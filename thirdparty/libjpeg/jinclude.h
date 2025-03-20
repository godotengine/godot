/*
 * jinclude.h
 *
 * Copyright (C) 1991-1994, Thomas G. Lane.
 * Modified 2017-2022 by Guido Vollbeding.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This file exists to provide a single place to fix any problems with
 * including the wrong system include files.  (Common problems are taken
 * care of by the standard jconfig symbols, but on really weird systems
 * you may have to edit this file.)
 *
 * NOTE: this file is NOT intended to be included by applications using
 * the JPEG library.  Most applications need only include jpeglib.h.
 */


/* Include auto-config file to find out which system include files we need. */

#include "jconfig.h"		/* auto configuration options */
#define JCONFIG_INCLUDED	/* so that jpeglib.h doesn't do it again */

/*
 * We need the NULL macro and size_t typedef.
 * On an ANSI-conforming system it is sufficient to include <stddef.h>.
 * Otherwise, we get them from <stdlib.h> or <stdio.h>; we may have to
 * pull in <sys/types.h> as well.
 * Note that the core JPEG library does not require <stdio.h>;
 * only the default error handler and data source/destination modules do.
 * But we must pull it in because of the references to FILE in jpeglib.h.
 * You can remove those references if you want to compile without <stdio.h>.
 */

#ifdef HAVE_STDDEF_H
#include <stddef.h>
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#ifdef NEED_SYS_TYPES_H
#include <sys/types.h>
#endif

#include <stdio.h>

/*
 * We need memory copying and zeroing functions, plus strncpy().
 * ANSI and System V implementations declare these in <string.h>.
 * BSD doesn't have the mem() functions, but it does have bcopy()/bzero().
 * Some systems may declare memset and memcpy in <memory.h>.
 *
 * NOTE: we assume the size parameters to these functions are of type size_t.
 * Change the casts in these macros if not!
 */

#ifdef NEED_BSD_STRINGS

#include <strings.h>
#define MEMZERO(target,size)	bzero((void *)(target), (size_t)(size))
#define MEMCOPY(dest,src,size)	bcopy((const void *)(src), (void *)(dest), (size_t)(size))

#else /* not BSD, assume ANSI/SysV string lib */

#include <string.h>
#define MEMZERO(target,size)	memset((void *)(target), 0, (size_t)(size))
#define MEMCOPY(dest,src,size)	memcpy((void *)(dest), (const void *)(src), (size_t)(size))

#endif

/*
 * In ANSI C, and indeed any rational implementation, size_t is also the
 * type returned by sizeof().  However, it seems there are some irrational
 * implementations out there, in which sizeof() returns an int even though
 * size_t is defined as long or unsigned long.  To ensure consistent results
 * we always use this SIZEOF() macro in place of using sizeof() directly.
 */

#define SIZEOF(object)	((size_t) sizeof(object))

/*
 * The modules that use fread() and fwrite() always invoke them through
 * these macros.  On some systems you may need to twiddle the argument casts.
 * CAUTION: argument order is different from underlying functions!
 *
 * Furthermore, macros are provided for fflush() and ferror() in order
 * to facilitate adaption by applications using an own FILE class.
 *
 * You can define your own custom file I/O functions in jconfig.h and
 * #define JPEG_HAVE_FILE_IO_CUSTOM there to prevent redefinition here.
 *
 * You can #define JPEG_USE_FILE_IO_CUSTOM in jconfig.h to use custom file
 * I/O functions implemented in Delphi VCL (Visual Component Library)
 * in Vcl.Imaging.jpeg.pas for the TJPEGImage component utilizing
 * the Delphi RTL (Run-Time Library) TMemoryStream component:
 *
 *   procedure jpeg_stdio_src(var cinfo: jpeg_decompress_struct;
 *     input_file: TStream); external;
 *
 *   procedure jpeg_stdio_dest(var cinfo: jpeg_compress_struct;
 *     output_file: TStream); external;
 *
 *   function jfread(var buf; recsize, reccount: Integer; S: TStream): Integer;
 *   begin
 *     Result := S.Read(buf, recsize * reccount);
 *   end;
 *
 *   function jfwrite(const buf; recsize, reccount: Integer; S: TStream): Integer;
 *   begin
 *     Result := S.Write(buf, recsize * reccount);
 *   end;
 *
 *   function jfflush(S: TStream): Integer;
 *   begin
 *     Result := 0;
 *   end;
 *
 *   function jferror(S: TStream): Integer;
 *   begin
 *     Result := 0;
 *   end;
 *
 * TMemoryStream of Delphi RTL has the distinctive feature to provide dynamic
 * memory buffer management with a file/stream-based interface, particularly for
 * the write (output) operation, which is easier to apply compared with direct
 * implementations as given in jdatadst.c for memory destination.  Those direct
 * implementations of dynamic memory write tend to be more difficult to use,
 * so providing an option like TMemoryStream may be a useful alternative.
 *
 * The CFile/CMemFile classes of the Microsoft Foundation Class (MFC) Library
 * may be used in a similar fashion.
 */

#ifndef JPEG_HAVE_FILE_IO_CUSTOM
#ifdef JPEG_USE_FILE_IO_CUSTOM
extern size_t jfread(void * __ptr, size_t __size, size_t __n, FILE * __stream);
extern size_t jfwrite(const void * __ptr, size_t __size, size_t __n, FILE * __stream);
extern int    jfflush(FILE * __stream);
extern int    jferror(FILE * __fp);

#define JFREAD(file,buf,sizeofbuf)  \
  ((size_t) jfread((void *) (buf), (size_t) 1, (size_t) (sizeofbuf), (file)))
#define JFWRITE(file,buf,sizeofbuf)  \
  ((size_t) jfwrite((const void *) (buf), (size_t) 1, (size_t) (sizeofbuf), (file)))
#define JFFLUSH(file)	jfflush(file)
#define JFERROR(file)	jferror(file)
#else
#define JFREAD(file,buf,sizeofbuf)  \
  ((size_t) fread((void *) (buf), (size_t) 1, (size_t) (sizeofbuf), (file)))
#define JFWRITE(file,buf,sizeofbuf)  \
  ((size_t) fwrite((const void *) (buf), (size_t) 1, (size_t) (sizeofbuf), (file)))
#define JFFLUSH(file)	fflush(file)
#define JFERROR(file)	ferror(file)
#endif
#endif
