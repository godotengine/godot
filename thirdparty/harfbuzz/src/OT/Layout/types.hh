/*
 * Copyright © 2007,2008,2009  Red Hat, Inc.
 * Copyright © 2010,2012  Google, Inc.
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
 * Red Hat Author(s): Behdad Esfahbod
 * Google Author(s): Behdad Esfahbod, Garret Rieger
 */

#ifndef OT_LAYOUT_TYPES_HH
#define OT_LAYOUT_TYPES_HH

using hb_ot_layout_mapping_cache_t = hb_cache_t<16, 8, 8>;
static_assert (sizeof (hb_ot_layout_mapping_cache_t) == 512, "");

using hb_ot_layout_binary_cache_t = hb_cache_t<14, 1, 8>;
static_assert (sizeof (hb_ot_layout_binary_cache_t) == 256, "");

namespace OT {
namespace Layout {

struct SmallTypes {
  static constexpr unsigned size = 2;
  using large_int = uint32_t;
  using HBUINT = HBUINT16;
  using HBGlyphID = HBGlyphID16;
  using Offset = Offset16;
  template <typename Type, typename BaseType=void, bool has_null=true>
  using OffsetTo = OT::Offset16To<Type, BaseType, has_null>;
  template <typename Type>
  using ArrayOf = OT::Array16Of<Type>;
  template <typename Type>
  using SortedArrayOf = OT::SortedArray16Of<Type>;
};

struct MediumTypes {
  static constexpr unsigned size = 3;
  using large_int = uint64_t;
  using HBUINT = HBUINT24;
  using HBGlyphID = HBGlyphID24;
  using Offset = Offset24;
  template <typename Type, typename BaseType=void, bool has_null=true>
  using OffsetTo = OT::Offset24To<Type, BaseType, has_null>;
  template <typename Type>
  using ArrayOf = OT::Array24Of<Type>;
  template <typename Type>
  using SortedArrayOf = OT::SortedArray24Of<Type>;
};

}
}

#endif  /* OT_LAYOUT_TYPES_HH */
