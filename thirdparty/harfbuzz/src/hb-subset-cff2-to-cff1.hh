/*
 * Copyright © 2026 Behdad Esfahbod
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
 */

#ifndef HB_SUBSET_CFF2_TO_CFF1_HH
#define HB_SUBSET_CFF2_TO_CFF1_HH

#include "hb.hh"

#ifndef HB_NO_SUBSET_CFF

#include "hb-ot-cff1-table.hh"
#include "hb-ot-cff2-table.hh"
#include "hb-subset-cff-common.hh"

namespace OT {
  // Forward declarations - these are defined in hb-subset-cff2.cc
  struct cff2_subset_plan;
}

namespace CFF {

// Forward declaration
struct cff2_top_dict_values_t;

// Default font name for converted CFF1 fonts
static constexpr const char CFF1_DEFAULT_FONT_NAME[] = "CFF1Font";

/*
 * CFF2 to CFF1 Converter
 *
 * Converts an instantiated (pinned) CFF2 variable font to CFF1 format.
 * This is used when instantiating a variable font to a static instance.
 *
 * IMPLEMENTATION STATUS:
 * ✓ CFF1 structure (Header, Name INDEX, String INDEX, Top DICT INDEX)
 * ✓ ROS operator (makes font CID-keyed: "Adobe-Identity-0")
 * ✓ FDArray and FDSelect in Top DICT (required for CID fonts)
 * ✓ FDSelect3 format (compact range-based, 8 bytes for single-FD fonts)
 * ✓ CID Charset with identity mapping (format 2)
 * ✓ FontBBox from head table (xMin, yMin, xMax, yMax)
 * ✓ Width optimization (defaultWidthX/nominalWidthX with O(n) algorithm)
 * ✓ Width encoding in CharStrings (prepended if != defaultWidthX)
 * ✓ CharString specialization (h/v operators, combined when possible)
 * ✓ Stack depth control (generalize→specialize with maxstack=48)
 * ✓ CharStrings with endchar operators (CFF1 requires, CFF2 doesn't)
 * ✓ Private DICT instantiation (blend operators evaluated)
 * ✓ Desubroutinized path (CharStrings are flattened, no subroutines)
 * ✓ OTS validation passes
 * ✓ HarfBuzz rendering works
 *
 * FUTURE ENHANCEMENTS:
 * - Curve operator specialization (hhcurveto, vvcurveto, etc.)
 * - Peephole optimization (minor additional size savings)
 *
 * Key conversions:
 * - Version: 2 -> 1
 * - Add Name INDEX (required in CFF1)
 * - Wrap Top DICT in an INDEX (inline in CFF2, indexed in CFF1)
 * - Add String INDEX ("Adobe", "Identity" for ROS operator)
 * - Add ROS operator to Top DICT (makes it CID-keyed)
 * - Add FDSelect to Top DICT (required in CFF1 even with single FD)
 * - Add endchar to CharStrings (required in CFF1, optional in CFF2)
 */

struct cff1_subset_plan_from_cff2_t
{
  // Inherits most data from cff2_subset_plan
  const OT::cff2_subset_plan *cff2_plan;

  // CFF1-specific additions
  hb_vector_t<unsigned char> fontName;  // Single font name for Name INDEX

  bool create (const OT::cff2_subset_plan &cff2_plan_)
  {
    cff2_plan = &cff2_plan_;

    // Create a simple font name (CFF1 requires a Name INDEX)
    fontName.resize (strlen (CFF1_DEFAULT_FONT_NAME));
    if (fontName.in_error ()) return false;
    memcpy (fontName.arrayZ, CFF1_DEFAULT_FONT_NAME, strlen (CFF1_DEFAULT_FONT_NAME));

    return true;
  }
};

/* CFF1 Top DICT operator serializer that adds ROS and removes CFF2-specific ops */
struct cff1_from_cff2_top_dict_op_serializer_t : cff_top_dict_op_serializer_t<>
{
  bool serialize (hb_serialize_context_t *c,
                  const op_str_t &opstr,
                  const cff_sub_table_info_t &info) const
  {
    TRACE_SERIALIZE (this);

    switch (opstr.op)
    {
      case OpCode_vstore:
        // CFF2-only operator, skip it
        return_trace (true);

      case OpCode_CharStrings:
        return_trace (FontDict::serialize_link4_op(c, opstr.op, info.char_strings_link, whence_t::Absolute));

      case OpCode_FDArray:
      case OpCode_FDSelect:
        // These are explicitly serialized in the main function to ensure they're present
        // even if CFF2 doesn't have them. Skip them here to avoid duplication.
        return_trace (true);

      default:
        return_trace (copy_opstr (c, opstr));
    }
  }

  // Serialize ROS operator to make this a CID-keyed font
  bool serialize_ros (hb_serialize_context_t *c) const
  {
    TRACE_SERIALIZE (this);

    // ROS = Registry-Ordering-Supplement
    // We use "Adobe", "Identity", 0 for maximum compatibility

    // Allocate space and encode directly
    // Registry: SID for "Adobe" (custom string at index 0 = SID 391)
    // Ordering: SID for "Identity" (custom string at index 1 = SID 392)
    // Supplement: 0
    // Note: CFF standard strings end at SID 390, custom strings start at 391

    str_buff_t buff;
    str_encoder_t encoder (buff);

    encoder.encode_int (391);  // Registry SID ("Adobe" in our String INDEX)
    encoder.encode_int (392);  // Ordering SID ("Identity" in our String INDEX)
    encoder.encode_int (0);    // Supplement
    encoder.encode_op (OpCode_ROS);

    if (encoder.in_error ())
      return_trace (false);

    auto bytes = buff.as_bytes ();
    return_trace (c->embed (bytes.arrayZ, bytes.length));
  }
};

/* Main serialization function */
HB_INTERNAL bool
serialize_cff2_to_cff1 (hb_serialize_context_t *c,
                        OT::cff2_subset_plan &plan,
                        const cff2_top_dict_values_t &cff2_topDict,
                        const OT::cff2::accelerator_subset_t &acc);

} /* namespace CFF */

#endif /* HB_NO_SUBSET_CFF */

#endif /* HB_SUBSET_CFF2_TO_CFF1_HH */
