/*
 * Copyright © 2017  Google, Inc.
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
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_STRING_ARRAY_HH
#if 0 /* Make checks happy. */
#define HB_STRING_ARRAY_HH
#endif

#include "hb-private.hh"

/* Based on Bruno Haible's code in Appendix B of Ulrich Drepper's dsohowto.pdf:
 * https://software.intel.com/sites/default/files/m/a/1/e/dsohowto.pdf */

#define HB_STRING_ARRAY_TYPE_NAME	HB_PASTE(HB_STRING_ARRAY_NAME, _msgstr_t)
#define HB_STRING_ARRAY_POOL_NAME	HB_PASTE(HB_STRING_ARRAY_NAME, _msgstr)
#define HB_STRING_ARRAY_OFFS_NAME	HB_PASTE(HB_STRING_ARRAY_NAME, _msgidx)

static const union HB_STRING_ARRAY_TYPE_NAME {
  struct {
/* I like to avoid storing the nul-termination byte since we don't need it,
 * but C++ does not allow that.
 * https://stackoverflow.com/q/28433862
 */
#define _S(s) char HB_PASTE (str, __LINE__)[sizeof (s)];
#include HB_STRING_ARRAY_LIST
#undef _S
  } st;
  char str[VAR];
}
HB_STRING_ARRAY_POOL_NAME =
{
  {
#define _S(s) s,
#include HB_STRING_ARRAY_LIST
#undef _S
  }
};
static const unsigned int HB_STRING_ARRAY_OFFS_NAME[] =
{
#define _S(s) offsetof (union HB_STRING_ARRAY_TYPE_NAME, st.HB_PASTE(str, __LINE__)),
#include HB_STRING_ARRAY_LIST
#undef _S
  sizeof (HB_STRING_ARRAY_TYPE_NAME)
};

static inline hb_bytes_t
HB_STRING_ARRAY_NAME (unsigned int i)
{
  assert (i < ARRAY_LENGTH (HB_STRING_ARRAY_OFFS_NAME) - 1);
  return hb_bytes_t (HB_STRING_ARRAY_POOL_NAME.str + HB_STRING_ARRAY_OFFS_NAME[i],
		     HB_STRING_ARRAY_OFFS_NAME[i + 1] - HB_STRING_ARRAY_OFFS_NAME[i] - 1);
}

#undef HB_STRING_ARRAY_TYPE_NAME
#undef HB_STRING_ARRAY_POOL_NAME
#undef HB_STRING_ARRAY_OFFS_NAME

#endif /* HB_STRING_ARRAY_HH */
