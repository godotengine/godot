/*
 * Copyright © 2022  Google, Inc.
 *
 *  This is part of HarfBuzz, a text shaping library.
 *
 * Permission is hereby granted, without written agreement and without
 * license or royalty fees, to use, copy, modify, and distribute this
 * software and its documentation for any purpose, provided that the
 * above copyright notice and the following two paragraphs appear in
 * all copies of this software.
 *
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
 * ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN
 * IF THE COPYRIGHT HOLDER HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 *
 * THE COPYRIGHT HOLDER SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND THE COPYRIGHT HOLDER HAS NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 *
 */

#ifndef HB_SUBSET_REPACKER_H
#define HB_SUBSET_REPACKER_H

#include "hb.h"

HB_BEGIN_DECLS

#ifdef HB_EXPERIMENTAL_API
/*
 * struct hb_link_t
 * width:    offsetSize in bytes
 * position: position of the offset field in bytes
 * from beginning of subtable
 * objidx:   index of subtable
 */
struct hb_link_t
{
  unsigned width;
  unsigned position;
  unsigned objidx;
};

typedef struct hb_link_t hb_link_t;

/*
 * struct hb_object_t
 * head:    start of object data
 * tail:    end of object data
 * num_real_links:    num of offset field in the object
 * real_links:        pointer to array of offset info
 * num_virtual_links: num of objects that must be packed
 * after current object in the final serialized order
 * virtual_links:     array of virtual link info
 */
struct hb_object_t
{
  char *head;
  char *tail;
  unsigned num_real_links;
  hb_link_t *real_links;
  unsigned num_virtual_links;
  hb_link_t *virtual_links;
};

typedef struct hb_object_t hb_object_t;

HB_EXTERN hb_blob_t*
hb_subset_repack_or_fail (hb_tag_t table_tag,
                          hb_object_t* hb_objects,
                          unsigned num_hb_objs);

#endif

HB_END_DECLS

#endif /* HB_SUBSET_REPACKER_H */
