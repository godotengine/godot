/*
 * jdicc.c
 *
 * Copyright (C) 1997-1998, Thomas G. Lane, Todd Newman.
 * Copyright (C) 2017, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file provides code to read International Color Consortium (ICC) device
 * profiles embedded in JFIF JPEG image files.  The ICC has defined a standard
 * for including such data in JPEG "APP2" markers.  The code given here does
 * not know anything about the internal structure of the ICC profile data; it
 * just knows how to get the profile data from a JPEG file while reading it.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jerror.h"


#define ICC_MARKER  (JPEG_APP0 + 2)     /* JPEG marker code for ICC */
#define ICC_OVERHEAD_LEN  14            /* size of non-profile data in APP2 */


/*
 * Handy subroutine to test whether a saved marker is an ICC profile marker.
 */

LOCAL(boolean)
marker_is_icc(jpeg_saved_marker_ptr marker)
{
  return
    marker->marker == ICC_MARKER &&
    marker->data_length >= ICC_OVERHEAD_LEN &&
    /* verify the identifying string */
    marker->data[0] == 0x49 &&
    marker->data[1] == 0x43 &&
    marker->data[2] == 0x43 &&
    marker->data[3] == 0x5F &&
    marker->data[4] == 0x50 &&
    marker->data[5] == 0x52 &&
    marker->data[6] == 0x4F &&
    marker->data[7] == 0x46 &&
    marker->data[8] == 0x49 &&
    marker->data[9] == 0x4C &&
    marker->data[10] == 0x45 &&
    marker->data[11] == 0x0;
}


/*
 * See if there was an ICC profile in the JPEG file being read; if so,
 * reassemble and return the profile data.
 *
 * TRUE is returned if an ICC profile was found, FALSE if not.  If TRUE is
 * returned, *icc_data_ptr is set to point to the returned data, and
 * *icc_data_len is set to its length.
 *
 * IMPORTANT: the data at *icc_data_ptr is allocated with malloc() and must be
 * freed by the caller with free() when the caller no longer needs it.
 * (Alternatively, we could write this routine to use the IJG library's memory
 * allocator, so that the data would be freed implicitly when
 * jpeg_finish_decompress() is called.  But it seems likely that many
 * applications will prefer to have the data stick around after decompression
 * finishes.)
 */

GLOBAL(boolean)
jpeg_read_icc_profile(j_decompress_ptr cinfo, JOCTET **icc_data_ptr,
                      unsigned int *icc_data_len)
{
  jpeg_saved_marker_ptr marker;
  int num_markers = 0;
  int seq_no;
  JOCTET *icc_data;
  unsigned int total_length;
#define MAX_SEQ_NO  255         /* sufficient since marker numbers are bytes */
  char marker_present[MAX_SEQ_NO + 1];      /* 1 if marker found */
  unsigned int data_length[MAX_SEQ_NO + 1]; /* size of profile data in marker */
  unsigned int data_offset[MAX_SEQ_NO + 1]; /* offset for data in marker */

  if (icc_data_ptr == NULL || icc_data_len == NULL)
    ERREXIT(cinfo, JERR_BUFFER_SIZE);
  if (cinfo->global_state < DSTATE_READY)
    ERREXIT1(cinfo, JERR_BAD_STATE, cinfo->global_state);

  *icc_data_ptr = NULL;         /* avoid confusion if FALSE return */
  *icc_data_len = 0;

  /* This first pass over the saved markers discovers whether there are
   * any ICC markers and verifies the consistency of the marker numbering.
   */

  for (seq_no = 1; seq_no <= MAX_SEQ_NO; seq_no++)
    marker_present[seq_no] = 0;

  for (marker = cinfo->marker_list; marker != NULL; marker = marker->next) {
    if (marker_is_icc(marker)) {
      if (num_markers == 0)
        num_markers = marker->data[13];
      else if (num_markers != marker->data[13]) {
        WARNMS(cinfo, JWRN_BOGUS_ICC);  /* inconsistent num_markers fields */
        return FALSE;
      }
      seq_no = marker->data[12];
      if (seq_no <= 0 || seq_no > num_markers) {
        WARNMS(cinfo, JWRN_BOGUS_ICC);  /* bogus sequence number */
        return FALSE;
      }
      if (marker_present[seq_no]) {
        WARNMS(cinfo, JWRN_BOGUS_ICC);  /* duplicate sequence numbers */
        return FALSE;
      }
      marker_present[seq_no] = 1;
      data_length[seq_no] = marker->data_length - ICC_OVERHEAD_LEN;
    }
  }

  if (num_markers == 0)
    return FALSE;

  /* Check for missing markers, count total space needed,
   * compute offset of each marker's part of the data.
   */

  total_length = 0;
  for (seq_no = 1; seq_no <= num_markers; seq_no++) {
    if (marker_present[seq_no] == 0) {
      WARNMS(cinfo, JWRN_BOGUS_ICC);  /* missing sequence number */
      return FALSE;
    }
    data_offset[seq_no] = total_length;
    total_length += data_length[seq_no];
  }

  if (total_length == 0) {
    WARNMS(cinfo, JWRN_BOGUS_ICC);  /* found only empty markers? */
    return FALSE;
  }

  /* Allocate space for assembled data */
  icc_data = (JOCTET *)malloc(total_length * sizeof(JOCTET));
  if (icc_data == NULL)
    ERREXIT1(cinfo, JERR_OUT_OF_MEMORY, 11);  /* oops, out of memory */

  /* and fill it in */
  for (marker = cinfo->marker_list; marker != NULL; marker = marker->next) {
    if (marker_is_icc(marker)) {
      JOCTET FAR *src_ptr;
      JOCTET *dst_ptr;
      unsigned int length;
      seq_no = marker->data[12];
      dst_ptr = icc_data + data_offset[seq_no];
      src_ptr = marker->data + ICC_OVERHEAD_LEN;
      length = data_length[seq_no];
      while (length--) {
        *dst_ptr++ = *src_ptr++;
      }
    }
  }

  *icc_data_ptr = icc_data;
  *icc_data_len = total_length;

  return TRUE;
}
