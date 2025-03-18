/*
 * jmemmac.c
 *
 * Copyright (C) 1992-1997, Thomas G. Lane.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * jmemmac.c provides an Apple Macintosh implementation of the system-
 * dependent portion of the JPEG memory manager.
 *
 * If you use jmemmac.c, then you must define USE_MAC_MEMMGR in the
 * JPEG_INTERNALS part of jconfig.h.
 *
 * jmemmac.c uses the Macintosh toolbox routines NewPtr and DisposePtr
 * instead of malloc and free.  It accurately determines the amount of
 * memory available by using CompactMem.  Notice that if left to its
 * own devices, this code can chew up all available space in the
 * application's zone, with the exception of the rather small "slop"
 * factor computed in jpeg_mem_available().  The application can ensure
 * that more space is left over by reducing max_memory_to_use.
 *
 * Large images are swapped to disk using temporary files and System 7.0+'s
 * temporary folder functionality.
 *
 * Note that jmemmac.c depends on two features of MacOS that were first
 * introduced in System 7: FindFolder and the FSSpec-based calls.
 * If your application uses jmemmac.c and is run under System 6 or earlier,
 * and the jpeg library decides it needs a temporary file, it will abort,
 * printing error messages about requiring System 7.  (If no temporary files
 * are created, it will run fine.)
 *
 * If you want to use jmemmac.c in an application that might be used with
 * System 6 or earlier, then you should remove dependencies on FindFolder
 * and the FSSpec calls.  You will need to replace FindFolder with some
 * other mechanism for finding a place to put temporary files, and you
 * should replace the FSSpec calls with their HFS equivalents:
 *
 *     FSpDelete     ->  HDelete
 *     FSpGetFInfo   ->  HGetFInfo
 *     FSpCreate     ->  HCreate
 *     FSpOpenDF     ->  HOpen      *** Note: not HOpenDF ***
 *     FSMakeFSSpec  ->  (fill in spec by hand.)
 *
 * (Use HOpen instead of HOpenDF.  HOpen is just a glue-interface to PBHOpen,
 * which is on all HFS macs.  HOpenDF is a System 7 addition which avoids the
 * ages-old problem of names starting with a period.)
 *
 * Contributed by Sam Bushell (jsam@iagu.on.net) and
 * Dan Gildor (gyld@in-touch.com).
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jmemsys.h"    /* import the system-dependent declarations */

#ifndef USE_MAC_MEMMGR	/* make sure user got configuration right */
  You forgot to define USE_MAC_MEMMGR in jconfig.h. /* deliberate syntax error */
#endif

#include <Memory.h>     /* we use the MacOS memory manager */
#include <Files.h>      /* we use the MacOS File stuff */
#include <Folders.h>    /* we use the MacOS HFS stuff */
#include <Script.h>     /* for smSystemScript */
#include <Gestalt.h>    /* we use Gestalt to test for specific functionality */

#ifndef TEMP_FILE_NAME		/* can override from jconfig.h or Makefile */
#define TEMP_FILE_NAME  "JPG%03d.TMP"
#endif

static int next_file_num;	/* to distinguish among several temp files */


/*
 * Memory allocation and freeing are controlled by the MacOS library
 * routines NewPtr() and DisposePtr(), which allocate fixed-address
 * storage.  Unfortunately, the IJG library isn't smart enough to cope
 * with relocatable storage.
 */

GLOBAL(void *)
jpeg_get_small (j_common_ptr cinfo, size_t sizeofobject)
{
  return (void *) NewPtr(sizeofobject);
}

GLOBAL(void)
jpeg_free_small (j_common_ptr cinfo, void * object, size_t sizeofobject)
{
  DisposePtr((Ptr) object);
}


/*
 * "Large" objects are treated the same as "small" ones.
 * NB: we include FAR keywords in the routine declarations simply for
 * consistency with the rest of the IJG code; FAR should expand to empty
 * on rational architectures like the Mac.
 */

GLOBAL(void FAR *)
jpeg_get_large (j_common_ptr cinfo, size_t sizeofobject)
{
  return (void FAR *) NewPtr(sizeofobject);
}

GLOBAL(void)
jpeg_free_large (j_common_ptr cinfo, void FAR * object, size_t sizeofobject)
{
  DisposePtr((Ptr) object);
}


/*
 * This routine computes the total memory space available for allocation.
 */

GLOBAL(long)
jpeg_mem_available (j_common_ptr cinfo, long min_bytes_needed,
		    long max_bytes_needed, long already_allocated)
{
  long limit = cinfo->mem->max_memory_to_use - already_allocated;
  long slop, mem;

  /* Don't ask for more than what application has told us we may use */
  if (max_bytes_needed > limit && limit > 0)
    max_bytes_needed = limit;
  /* Find whether there's a big enough free block in the heap.
   * CompactMem tries to create a contiguous block of the requested size,
   * and then returns the size of the largest free block (which could be
   * much more or much less than we asked for).
   * We add some slop to ensure we don't use up all available memory.
   */
  slop = max_bytes_needed / 16 + 32768L;
  mem = CompactMem(max_bytes_needed + slop) - slop;
  if (mem < 0)
    mem = 0;			/* sigh, couldn't even get the slop */
  /* Don't take more than the application says we can have */
  if (mem > limit && limit > 0)
    mem = limit;
  return mem;
}


/*
 * Backing store (temporary file) management.
 * Backing store objects are only used when the value returned by
 * jpeg_mem_available is less than the total space needed.  You can dispense
 * with these routines if you have plenty of virtual memory; see jmemnobs.c.
 */


METHODDEF(void)
read_backing_store (j_common_ptr cinfo, backing_store_ptr info,
		    void FAR * buffer_address,
		    long file_offset, long byte_count)
{
  long bytes = byte_count;
  long retVal;

  if ( SetFPos ( info->temp_file, fsFromStart, file_offset ) != noErr )
    ERREXIT(cinfo, JERR_TFILE_SEEK);

  retVal = FSRead ( info->temp_file, &bytes,
		    (unsigned char *) buffer_address );
  if ( retVal != noErr || bytes != byte_count )
    ERREXIT(cinfo, JERR_TFILE_READ);
}


METHODDEF(void)
write_backing_store (j_common_ptr cinfo, backing_store_ptr info,
		     void FAR * buffer_address,
		     long file_offset, long byte_count)
{
  long bytes = byte_count;
  long retVal;

  if ( SetFPos ( info->temp_file, fsFromStart, file_offset ) != noErr )
    ERREXIT(cinfo, JERR_TFILE_SEEK);

  retVal = FSWrite ( info->temp_file, &bytes,
		     (unsigned char *) buffer_address );
  if ( retVal != noErr || bytes != byte_count )
    ERREXIT(cinfo, JERR_TFILE_WRITE);
}


METHODDEF(void)
close_backing_store (j_common_ptr cinfo, backing_store_ptr info)
{
  FSClose ( info->temp_file );
  FSpDelete ( &(info->tempSpec) );
}


/*
 * Initial opening of a backing-store object.
 *
 * This version uses FindFolder to find the Temporary Items folder,
 * and puts the temporary file in there.
 */

GLOBAL(void)
jpeg_open_backing_store (j_common_ptr cinfo, backing_store_ptr info,
			 long total_bytes_needed)
{
  short         tmpRef, vRefNum;
  long          dirID;
  FInfo         finderInfo;
  FSSpec        theSpec;
  Str255        fName;
  OSErr         osErr;
  long          gestaltResponse = 0;

  /* Check that FSSpec calls are available. */
  osErr = Gestalt( gestaltFSAttr, &gestaltResponse );
  if ( ( osErr != noErr )
       || !( gestaltResponse & (1<<gestaltHasFSSpecCalls) ) )
    ERREXITS(cinfo, JERR_TFILE_CREATE, "- System 7.0 or later required");
  /* TO DO: add a proper error message to jerror.h. */

  /* Check that FindFolder is available. */
  osErr = Gestalt( gestaltFindFolderAttr, &gestaltResponse );
  if ( ( osErr != noErr )
       || !( gestaltResponse & (1<<gestaltFindFolderPresent) ) )
    ERREXITS(cinfo, JERR_TFILE_CREATE, "- System 7.0 or later required.");
  /* TO DO: add a proper error message to jerror.h. */

  osErr = FindFolder ( kOnSystemDisk, kTemporaryFolderType, kCreateFolder,
                       &vRefNum, &dirID );
  if ( osErr != noErr )
    ERREXITS(cinfo, JERR_TFILE_CREATE, "- temporary items folder unavailable");
  /* TO DO: Try putting the temp files somewhere else. */

  /* Keep generating file names till we find one that's not in use */
  for (;;) {
    next_file_num++;		/* advance counter */

    sprintf(info->temp_name, TEMP_FILE_NAME, next_file_num);
    strcpy ( (Ptr)fName+1, info->temp_name );
    *fName = strlen (info->temp_name);
    osErr = FSMakeFSSpec ( vRefNum, dirID, fName, &theSpec );

    if ( (osErr = FSpGetFInfo ( &theSpec, &finderInfo ) ) != noErr )
      break;
  }

  osErr = FSpCreate ( &theSpec, '????', '????', smSystemScript );
  if ( osErr != noErr )
    ERREXITS(cinfo, JERR_TFILE_CREATE, info->temp_name);

  osErr = FSpOpenDF ( &theSpec, fsRdWrPerm, &(info->temp_file) );
  if ( osErr != noErr )
    ERREXITS(cinfo, JERR_TFILE_CREATE, info->temp_name);

  info->tempSpec = theSpec;

  info->read_backing_store = read_backing_store;
  info->write_backing_store = write_backing_store;
  info->close_backing_store = close_backing_store;
  TRACEMSS(cinfo, 1, JTRC_TFILE_OPEN, info->temp_name);
}


/*
 * These routines take care of any system-dependent initialization and
 * cleanup required.
 */

GLOBAL(long)
jpeg_mem_init (j_common_ptr cinfo)
{
  next_file_num = 0;

  /* max_memory_to_use will be initialized to FreeMem()'s result;
   * the calling application might later reduce it, for example
   * to leave room to invoke multiple JPEG objects.
   * Note that FreeMem returns the total number of free bytes;
   * it may not be possible to allocate a single block of this size.
   */
  return FreeMem();
}

GLOBAL(void)
jpeg_mem_term (j_common_ptr cinfo)
{
  /* no work */
}
