/*
 * jdatadst-tj.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1994-1996, Thomas G. Lane.
 * Modified 2009-2012 by Guido Vollbeding.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2011, 2014, 2016, 2019, 2022-2023, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains compression data destination routines for the case of
 * emitting JPEG data to memory or to a file (or any stdio stream).
 * While these routines are sufficient for most applications,
 * some will want to use a different destination manager.
 * IMPORTANT: we assume that fwrite() will correctly transcribe an array of
 * JOCTETs into 8-bit-wide elements on external storage.  If char is wider
 * than 8 bits on your machine, you may need to do some tweaking.
 */

/* this is not a core library module, so it doesn't define JPEG_INTERNALS */
#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jerror.h"

void jpeg_mem_dest_tj(j_compress_ptr cinfo, unsigned char **outbuffer,
                      size_t *outsize, boolean alloc);


#define OUTPUT_BUF_SIZE  4096   /* choose an efficiently fwrite'able size */


/* Expanded data destination object for memory output */

typedef struct {
  struct jpeg_destination_mgr pub; /* public fields */

  unsigned char **outbuffer;    /* target buffer */
  size_t *outsize;
  unsigned char *newbuffer;     /* newly allocated buffer */
  JOCTET *buffer;               /* start of buffer */
  size_t bufsize;
  boolean alloc;
} my_mem_destination_mgr;

typedef my_mem_destination_mgr *my_mem_dest_ptr;


/*
 * Initialize destination --- called by jpeg_start_compress
 * before any data is actually written.
 */

METHODDEF(void)
init_mem_destination(j_compress_ptr cinfo)
{
  /* no work necessary here */
}


/*
 * Empty the output buffer --- called whenever buffer fills up.
 *
 * In typical applications, this should write the entire output buffer
 * (ignoring the current state of next_output_byte & free_in_buffer),
 * reset the pointer & count to the start of the buffer, and return TRUE
 * indicating that the buffer has been dumped.
 *
 * In applications that need to be able to suspend compression due to output
 * overrun, a FALSE return indicates that the buffer cannot be emptied now.
 * In this situation, the compressor will return to its caller (possibly with
 * an indication that it has not accepted all the supplied scanlines).  The
 * application should resume compression after it has made more room in the
 * output buffer.  Note that there are substantial restrictions on the use of
 * suspension --- see the documentation.
 *
 * When suspending, the compressor will back up to a convenient restart point
 * (typically the start of the current MCU). next_output_byte & free_in_buffer
 * indicate where the restart point will be if the current call returns FALSE.
 * Data beyond this point will be regenerated after resumption, so do not
 * write it out when emptying the buffer externally.
 */

METHODDEF(boolean)
empty_mem_output_buffer(j_compress_ptr cinfo)
{
  size_t nextsize;
  JOCTET *nextbuffer;
  my_mem_dest_ptr dest = (my_mem_dest_ptr)cinfo->dest;

  if (!dest->alloc) ERREXIT(cinfo, JERR_BUFFER_SIZE);

  /* Try to allocate new buffer with double size */
  nextsize = dest->bufsize * 2;
  nextbuffer = (JOCTET *)MALLOC(nextsize);

  if (nextbuffer == NULL)
    ERREXIT1(cinfo, JERR_OUT_OF_MEMORY, 10);

  memcpy(nextbuffer, dest->buffer, dest->bufsize);

  free(dest->newbuffer);

  dest->newbuffer = nextbuffer;

  dest->pub.next_output_byte = nextbuffer + dest->bufsize;
  dest->pub.free_in_buffer = dest->bufsize;

  dest->buffer = nextbuffer;
  dest->bufsize = nextsize;

  return TRUE;
}


/*
 * Terminate destination --- called by jpeg_finish_compress
 * after all data has been written.  Usually needs to flush buffer.
 *
 * NB: *not* called by jpeg_abort or jpeg_destroy; surrounding
 * application must deal with any cleanup that should happen even
 * for error exit.
 */

METHODDEF(void)
term_mem_destination(j_compress_ptr cinfo)
{
  my_mem_dest_ptr dest = (my_mem_dest_ptr)cinfo->dest;

  if (dest->alloc) *dest->outbuffer = dest->buffer;
  *dest->outsize = dest->bufsize - dest->pub.free_in_buffer;
}


/*
 * Prepare for output to a memory buffer.
 * The caller may supply an own initial buffer with appropriate size.
 * Otherwise, or when the actual data output exceeds the given size,
 * the library adapts the buffer size as necessary.
 * The standard library functions malloc/free are used for allocating
 * larger memory, so the buffer is available to the application after
 * finishing compression, and then the application is responsible for
 * freeing the requested memory.
 */

GLOBAL(void)
jpeg_mem_dest_tj(j_compress_ptr cinfo, unsigned char **outbuffer,
                 size_t *outsize, boolean alloc)
{
  boolean reused = FALSE;
  my_mem_dest_ptr dest;

  if (outbuffer == NULL || outsize == NULL)     /* sanity check */
    ERREXIT(cinfo, JERR_BUFFER_SIZE);

  /* The destination object is made permanent so that multiple JPEG images
   * can be written to the same buffer without re-executing jpeg_mem_dest.
   */
  if (cinfo->dest == NULL) {    /* first time for this JPEG object? */
    cinfo->dest = (struct jpeg_destination_mgr *)
      (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_PERMANENT,
                                  sizeof(my_mem_destination_mgr));
    dest = (my_mem_dest_ptr)cinfo->dest;
    dest->newbuffer = NULL;
    dest->buffer = NULL;
  } else if (cinfo->dest->init_destination != init_mem_destination) {
    /* It is unsafe to reuse the existing destination manager unless it was
     * created by this function.
     */
    ERREXIT(cinfo, JERR_BUFFER_SIZE);
  }

  dest = (my_mem_dest_ptr)cinfo->dest;
  dest->pub.init_destination = init_mem_destination;
  dest->pub.empty_output_buffer = empty_mem_output_buffer;
  dest->pub.term_destination = term_mem_destination;
  if (dest->buffer == *outbuffer && *outbuffer != NULL && alloc)
    reused = TRUE;
  dest->outbuffer = outbuffer;
  dest->outsize = outsize;
  dest->alloc = alloc;

  if (*outbuffer == NULL || *outsize == 0) {
    if (alloc) {
      /* Allocate initial buffer */
      dest->newbuffer = *outbuffer = (unsigned char *)MALLOC(OUTPUT_BUF_SIZE);
      if (dest->newbuffer == NULL)
        ERREXIT1(cinfo, JERR_OUT_OF_MEMORY, 10);
      *outsize = OUTPUT_BUF_SIZE;
    } else
      ERREXIT(cinfo, JERR_BUFFER_SIZE);
  }

  dest->pub.next_output_byte = dest->buffer = *outbuffer;
  if (!reused)
    dest->bufsize = *outsize;
  dest->pub.free_in_buffer = dest->bufsize;
}
