/*
 * Copyright © 2018 Adobe Inc.
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
 * Adobe Author(s): Michiharu Ariza
 */

#include "hb.hh"

#ifndef HB_NO_SUBSET_CFF

#include "hb-ot-cff-common.hh"
#include "hb-ot-cff2-table.hh"
#include "hb-subset-cff-common.hh"

/* Disable FDSelect format 0 for compatibility with fonttools which doesn't seem choose it.
 * Rarely any/much smaller than format 3 anyway. */
#define CFF_SERIALIZE_FDSELECT_0  0

using namespace CFF;

/**
 * hb_plan_subset_cff_fdselect
 * Determine an optimal FDSelect format according to a provided plan.
 *
 * Return value: FDSelect format, size, and ranges for the most compact subset FDSelect
 * along with a font index remapping table
 **/

bool
hb_plan_subset_cff_fdselect (const hb_subset_plan_t *plan,
			     unsigned int fdCount,
			     const FDSelect &src, /* IN */
			     unsigned int &subset_fd_count /* OUT */,
			     unsigned int &subset_fdselect_size /* OUT */,
			     unsigned int &subset_fdselect_format /* OUT */,
			     hb_vector_t<code_pair_t> &fdselect_ranges /* OUT */,
			     hb_inc_bimap_t &fdmap /* OUT */)
{
  subset_fd_count = 0;
  subset_fdselect_size = 0;
  subset_fdselect_format = 0;
  unsigned int num_ranges = 0;

  unsigned int subset_num_glyphs = plan->num_output_glyphs ();
  if (subset_num_glyphs == 0)
    return true;

  {
    /* use hb_set to determine the subset of font dicts */
    hb_set_t *set = hb_set_create ();
    if (unlikely (set == &Null (hb_set_t))) return false;
    hb_codepoint_t prev_fd = CFF_UNDEF_CODE;
    for (hb_codepoint_t i = 0; i < subset_num_glyphs; i++)
    {
      hb_codepoint_t glyph;
      hb_codepoint_t fd;
      if (!plan->old_gid_for_new_gid (i, &glyph))
      {
	/* fonttools retains FDSelect & font dicts for missing glyphs. do the same */
	glyph = i;
      }
      fd = src.get_fd (glyph);
      set->add (fd);

      if (fd != prev_fd)
      {
	num_ranges++;
	prev_fd = fd;
	code_pair_t pair = { fd, i };
	fdselect_ranges.push (pair);
      }
    }

    subset_fd_count = set->get_population ();
    if (subset_fd_count == fdCount)
    {
      /* all font dicts belong to the subset. no need to subset FDSelect & FDArray */
      fdmap.identity (fdCount);
      hb_set_destroy (set);
    }
    else
    {
      /* create a fdmap */
      fdmap.reset ();

      hb_codepoint_t fd = CFF_UNDEF_CODE;
      while (set->next (&fd))
	fdmap.add (fd);
      hb_set_destroy (set);
      if (unlikely (fdmap.get_population () != subset_fd_count))
	return false;
    }

    /* update each font dict index stored as "code" in fdselect_ranges */
    for (unsigned int i = 0; i < fdselect_ranges.length; i++)
      fdselect_ranges[i].code = fdmap[fdselect_ranges[i].code];
  }

  /* determine which FDSelect format is most compact */
  if (subset_fd_count > 0xFF)
  {
    if (unlikely (src.format != 4))
      return false;
    subset_fdselect_format = 4;
    subset_fdselect_size = FDSelect::min_size + FDSelect4::min_size + FDSelect4_Range::static_size * num_ranges + HBUINT32::static_size;
  }
  else
  {
#if CFF_SERIALIZE_FDSELECT_0
    unsigned int format0_size = FDSelect::min_size + FDSelect0::min_size + HBUINT8::static_size * subset_num_glyphs;
#endif
    unsigned int format3_size = FDSelect::min_size + FDSelect3::min_size + FDSelect3_Range::static_size * num_ranges + HBUINT16::static_size;

#if CFF_SERIALIZE_FDSELECT_0
    if (format0_size <= format3_size)
    {
      // subset_fdselect_format = 0;
      subset_fdselect_size = format0_size;
    }
    else
#endif
    {
      subset_fdselect_format = 3;
      subset_fdselect_size = format3_size;
    }
  }

  return true;
}

template <typename FDSELECT3_4>
static inline bool
serialize_fdselect_3_4 (hb_serialize_context_t *c,
			const unsigned int num_glyphs,
			const FDSelect &src,
			unsigned int size,
			const hb_vector_t<code_pair_t> &fdselect_ranges)
{
  TRACE_SERIALIZE (this);
  FDSELECT3_4 *p = c->allocate_size<FDSELECT3_4> (size);
  if (unlikely (!p)) return_trace (false);
  p->nRanges () = fdselect_ranges.length;
  for (unsigned int i = 0; i < fdselect_ranges.length; i++)
  {
    p->ranges[i].first = fdselect_ranges[i].glyph;
    p->ranges[i].fd = fdselect_ranges[i].code;
  }
  p->sentinel () = num_glyphs;
  return_trace (true);
}

/**
 * hb_serialize_cff_fdselect
 * Serialize a subset FDSelect format planned above.
 **/
bool
hb_serialize_cff_fdselect (hb_serialize_context_t *c,
			   const unsigned int num_glyphs,
			   const FDSelect &src,
			   unsigned int fd_count,
			   unsigned int fdselect_format,
			   unsigned int size,
			   const hb_vector_t<code_pair_t> &fdselect_ranges)
{
  TRACE_SERIALIZE (this);
  FDSelect *p = c->allocate_min<FDSelect> ();
  if (unlikely (!p)) return_trace (false);
  p->format = fdselect_format;
  size -= FDSelect::min_size;

  switch (fdselect_format)
  {
#if CFF_SERIALIZE_FDSELECT_0
  case 0:
  {
    FDSelect0 *p = c->allocate_size<FDSelect0> (size);
    if (unlikely (!p)) return_trace (false);
    unsigned int range_index = 0;
    unsigned int fd = fdselect_ranges[range_index++].code;
    for (unsigned int i = 0; i < num_glyphs; i++)
    {
      if ((range_index < fdselect_ranges.len) &&
	  (i >= fdselect_ranges[range_index].glyph))
      {
	fd = fdselect_ranges[range_index++].code;
      }
      p->fds[i] = fd;
    }
    return_trace (true);
  }
#endif /* CFF_SERIALIZE_FDSELECT_0 */

  case 3:
    return serialize_fdselect_3_4<FDSelect3> (c, num_glyphs, src,
					      size, fdselect_ranges);

  case 4:
    return serialize_fdselect_3_4<FDSelect4> (c, num_glyphs, src,
					      size, fdselect_ranges);

  default:
    return_trace (false);
  }
}


#endif
