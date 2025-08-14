/*
 * jmemsys.h
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1992-1997, Thomas G. Lane.
 * It was modified by The libjpeg-turbo Project to include only code and
 * information relevant to libjpeg-turbo.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This include file defines the interface between the system-independent
 * and system-dependent portions of the JPEG memory manager.  No other
 * modules need include it.  (The system-independent portion is jmemmgr.c;
 * there are several different versions of the system-dependent portion.)
 *
 * This file works as-is for the system-dependent memory managers supplied
 * in the IJG distribution.  You may need to modify it if you write a
 * custom memory manager.  If system-dependent changes are needed in
 * this file, the best method is to #ifdef them based on a configuration
 * symbol supplied in jconfig.h.
 */


/*
 * These two functions are used to allocate and release small chunks of
 * memory.  (Typically the total amount requested through jpeg_get_small is
 * no more than 20K or so; this will be requested in chunks of a few K each.)
 * Behavior should be the same as for the standard library functions malloc
 * and free; in particular, jpeg_get_small must return NULL on failure.
 * On most systems, these ARE malloc and free.  jpeg_free_small is passed the
 * size of the object being freed, just in case it's needed.
 */

EXTERN(void *) jpeg_get_small(j_common_ptr cinfo, size_t sizeofobject);
EXTERN(void) jpeg_free_small(j_common_ptr cinfo, void *object,
                             size_t sizeofobject);

/*
 * These two functions are used to allocate and release large chunks of
 * memory (up to the total free space designated by jpeg_mem_available).
 * These are identical to the jpeg_get/free_small routines; but we keep them
 * separate anyway, in case a different allocation strategy is desirable for
 * large chunks.
 */

EXTERN(void *) jpeg_get_large(j_common_ptr cinfo, size_t sizeofobject);
EXTERN(void) jpeg_free_large(j_common_ptr cinfo, void *object,
                             size_t sizeofobject);

/*
 * The macro MAX_ALLOC_CHUNK designates the maximum number of bytes that may
 * be requested in a single call to jpeg_get_large (and jpeg_get_small for that
 * matter, but that case should never come into play).  This macro was needed
 * to model the 64Kb-segment-size limit of far addressing on 80x86 machines.
 * On machines with flat address spaces, any large constant may be used.
 *
 * NB: jmemmgr.c expects that MAX_ALLOC_CHUNK will be representable as type
 * size_t and will be a multiple of sizeof(align_type).
 */

#ifndef MAX_ALLOC_CHUNK         /* may be overridden in jconfig.h */
#define MAX_ALLOC_CHUNK  1000000000L
#endif

/*
 * This routine computes the total space still available for allocation by
 * jpeg_get_large.  If more space than this is needed, backing store will be
 * used.  NOTE: any memory already allocated must not be counted.
 *
 * There is a minimum space requirement, corresponding to the minimum
 * feasible buffer sizes; jmemmgr.c will request that much space even if
 * jpeg_mem_available returns zero.  The maximum space needed, enough to hold
 * all working storage in memory, is also passed in case it is useful.
 * Finally, the total space already allocated is passed.  If no better
 * method is available, cinfo->mem->max_memory_to_use - already_allocated
 * is often a suitable calculation.
 *
 * It is OK for jpeg_mem_available to underestimate the space available
 * (that'll just lead to more backing-store access than is really necessary).
 * However, an overestimate will lead to failure.  Hence it's wise to subtract
 * a slop factor from the true available space.  5% should be enough.
 *
 * On machines with lots of virtual memory, any large constant may be returned.
 * Conversely, zero may be returned to always use the minimum amount of memory.
 */

EXTERN(size_t) jpeg_mem_available(j_common_ptr cinfo, size_t min_bytes_needed,
                                  size_t max_bytes_needed,
                                  size_t already_allocated);


/*
 * This structure holds whatever state is needed to access a single
 * backing-store object.  The read/write/close method pointers are called
 * by jmemmgr.c to manipulate the backing-store object; all other fields
 * are private to the system-dependent backing store routines.
 */

#define TEMP_NAME_LENGTH   64   /* max length of a temporary file's name */


typedef struct backing_store_struct *backing_store_ptr;

typedef struct backing_store_struct {
  /* Methods for reading/writing/closing this backing-store object */
  void (*read_backing_store) (j_common_ptr cinfo, backing_store_ptr info,
                              void *buffer_address, long file_offset,
                              long byte_count);
  void (*write_backing_store) (j_common_ptr cinfo, backing_store_ptr info,
                               void *buffer_address, long file_offset,
                               long byte_count);
  void (*close_backing_store) (j_common_ptr cinfo, backing_store_ptr info);

  /* Private fields for system-dependent backing-store management */
  /* For a typical implementation with temp files, we need: */
  FILE *temp_file;              /* stdio reference to temp file */
  char temp_name[TEMP_NAME_LENGTH]; /* name of temp file */
} backing_store_info;


/*
 * Initial opening of a backing-store object.  This must fill in the
 * read/write/close pointers in the object.  The read/write routines
 * may take an error exit if the specified maximum file size is exceeded.
 * (If jpeg_mem_available always returns a large value, this routine can
 * just take an error exit.)
 */

EXTERN(void) jpeg_open_backing_store(j_common_ptr cinfo,
                                     backing_store_ptr info,
                                     long total_bytes_needed);


/*
 * These routines take care of any system-dependent initialization and
 * cleanup required.  jpeg_mem_init will be called before anything is
 * allocated (and, therefore, nothing in cinfo is of use except the error
 * manager pointer).  It should return a suitable default value for
 * max_memory_to_use; this may subsequently be overridden by the surrounding
 * application.  (Note that max_memory_to_use is only important if
 * jpeg_mem_available chooses to consult it ... no one else will.)
 * jpeg_mem_term may assume that all requested memory has been freed and that
 * all opened backing-store objects have been closed.
 */

EXTERN(long) jpeg_mem_init(j_common_ptr cinfo);
EXTERN(void) jpeg_mem_term(j_common_ptr cinfo);
