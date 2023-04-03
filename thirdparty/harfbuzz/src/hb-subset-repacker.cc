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
#include "hb-repacker.hh"

#ifdef HB_EXPERIMENTAL_API

/**
 * hb_subset_repack_or_fail:
 * @table_tag: tag of the table being packed, needed to allow table specific optimizations.
 * @hb_objects: raw array of struct hb_object_t, which provides
 * object graph info
 * @num_hb_objs: number of hb_object_t in the hb_objects array.
 *
 * Given the input object graph info, repack a table to eliminate
 * offset overflows. A nullptr is returned if the repacking attempt fails.
 * Table specific optimizations (eg. extension promotion in GSUB/GPOS) may be performed.
 * Passing HB_TAG_NONE will disable table specific optimizations.
 *
 * XSince: EXPERIMENTAL
 **/
hb_blob_t* hb_subset_repack_or_fail (hb_tag_t table_tag,
                                     hb_object_t* hb_objects,
                                     unsigned num_hb_objs)
{
  hb_vector_t<const hb_object_t *> packed;
  packed.alloc (num_hb_objs + 1);
  packed.push (nullptr);
  for (unsigned i = 0 ; i < num_hb_objs ; i++)
    packed.push (&(hb_objects[i]));

  return hb_resolve_overflows (packed,
                               table_tag,
                               20,
                               true);
}
#endif
