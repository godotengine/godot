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

#include "hb-open-type.hh"
#include "hb-ot-cff1-table.hh"
#include "hb-set.h"
#include "hb-bimap.hh"
#include "hb-subset-plan.hh"
#include "hb-subset-cff-common.hh"
#include "hb-cff1-interp-cs.hh"

using namespace CFF;

struct remap_sid_t
{
  unsigned get_population () const { return vector.length; }

  void alloc (unsigned size)
  {
    map.alloc (size);
    vector.alloc_exact (size);
  }

  bool in_error () const
  { return map.in_error () || vector.in_error (); }

  unsigned int add (unsigned int sid)
  {
    if (is_std_str (sid) || (sid == CFF_UNDEF_SID))
      return sid;

    sid = unoffset_sid (sid);
    unsigned v = next;
    if (map.set (sid, v, false))
    {
      vector.push (sid);
      next++;
    }
    else
      v = map.get (sid); // already exists
    return offset_sid (v);
  }

  unsigned int operator[] (unsigned int sid) const
  {
    if (is_std_str (sid) || (sid == CFF_UNDEF_SID))
      return sid;

    return offset_sid (map.get (unoffset_sid (sid)));
  }

  static const unsigned int num_std_strings = 391;

  static bool is_std_str (unsigned int sid) { return sid < num_std_strings; }
  static unsigned int offset_sid (unsigned int sid) { return sid + num_std_strings; }
  static unsigned int unoffset_sid (unsigned int sid) { return sid - num_std_strings; }
  unsigned next = 0;

  hb_map_t map;
  hb_vector_t<unsigned> vector;
};

struct cff1_sub_table_info_t : cff_sub_table_info_t
{
  cff1_sub_table_info_t ()
    : cff_sub_table_info_t (),
      encoding_link (0),
      charset_link (0)
   {
    privateDictInfo.init ();
  }

  objidx_t	encoding_link;
  objidx_t	charset_link;
  table_info_t	privateDictInfo;
};

/* a copy of a parsed out cff1_top_dict_values_t augmented with additional operators */
struct cff1_top_dict_values_mod_t : cff1_top_dict_values_t
{
  void init (const cff1_top_dict_values_t *base_= &Null (cff1_top_dict_values_t))
  {
    SUPER::init ();
    base = base_;
  }

  void fini () { SUPER::fini (); }

  unsigned get_count () const { return base->get_count () + SUPER::get_count (); }
  const cff1_top_dict_val_t &get_value (unsigned int i) const
  {
    if (i < base->get_count ())
      return (*base)[i];
    else
      return SUPER::values[i - base->get_count ()];
  }
  const cff1_top_dict_val_t &operator [] (unsigned int i) const { return get_value (i); }

  void reassignSIDs (const remap_sid_t& sidmap)
  {
    for (unsigned int i = 0; i < name_dict_values_t::ValCount; i++)
      nameSIDs[i] = sidmap[base->nameSIDs[i]];
  }

  protected:
  typedef cff1_top_dict_values_t SUPER;
  const cff1_top_dict_values_t *base;
};

struct top_dict_modifiers_t
{
  top_dict_modifiers_t (const cff1_sub_table_info_t &info_,
			const unsigned int (&nameSIDs_)[name_dict_values_t::ValCount])
    : info (info_),
      nameSIDs (nameSIDs_)
  {}

  const cff1_sub_table_info_t &info;
  const unsigned int	(&nameSIDs)[name_dict_values_t::ValCount];
};

struct cff1_top_dict_op_serializer_t : cff_top_dict_op_serializer_t<cff1_top_dict_val_t>
{
  bool serialize (hb_serialize_context_t *c,
		  const cff1_top_dict_val_t &opstr,
		  const top_dict_modifiers_t &mod) const
  {
    TRACE_SERIALIZE (this);

    op_code_t op = opstr.op;
    switch (op)
    {
      case OpCode_charset:
	if (mod.info.charset_link)
	  return_trace (FontDict::serialize_link4_op(c, op, mod.info.charset_link, whence_t::Absolute));
	else
	  goto fall_back;

      case OpCode_Encoding:
	if (mod.info.encoding_link)
	  return_trace (FontDict::serialize_link4_op(c, op, mod.info.encoding_link, whence_t::Absolute));
	else
	  goto fall_back;

      case OpCode_Private:
	return_trace (UnsizedByteStr::serialize_int2 (c, mod.info.privateDictInfo.size) &&
		      Dict::serialize_link4_op (c, op, mod.info.privateDictInfo.link, whence_t::Absolute));

      case OpCode_version:
      case OpCode_Notice:
      case OpCode_Copyright:
      case OpCode_FullName:
      case OpCode_FamilyName:
      case OpCode_Weight:
      case OpCode_PostScript:
      case OpCode_BaseFontName:
      case OpCode_FontName:
	return_trace (FontDict::serialize_int2_op (c, op, mod.nameSIDs[name_dict_values_t::name_op_to_index (op)]));

      case OpCode_ROS:
	{
	  /* for registry & ordering, reassigned SIDs are serialized
	   * for supplement, the original byte string is copied along with the op code */
	  op_str_t supp_op;
	  supp_op.op = op;
	  if ( unlikely (!(opstr.length >= opstr.last_arg_offset + 3)))
	    return_trace (false);
	  supp_op.ptr = opstr.ptr + opstr.last_arg_offset;
	  supp_op.length = opstr.length - opstr.last_arg_offset;
	  return_trace (UnsizedByteStr::serialize_int2 (c, mod.nameSIDs[name_dict_values_t::registry]) &&
			UnsizedByteStr::serialize_int2 (c, mod.nameSIDs[name_dict_values_t::ordering]) &&
			copy_opstr (c, supp_op));
	}
      fall_back:
      default:
	return_trace (cff_top_dict_op_serializer_t<cff1_top_dict_val_t>::serialize (c, opstr, mod.info));
    }
    return_trace (true);
  }

};

struct cff1_font_dict_op_serializer_t : cff_font_dict_op_serializer_t
{
  bool serialize (hb_serialize_context_t *c,
		  const op_str_t &opstr,
		  const cff1_font_dict_values_mod_t &mod) const
  {
    TRACE_SERIALIZE (this);

    if (opstr.op == OpCode_FontName)
      return_trace (FontDict::serialize_int2_op (c, opstr.op, mod.fontName));
    else
      return_trace (SUPER::serialize (c, opstr, mod.privateDictInfo));
  }

  private:
  typedef cff_font_dict_op_serializer_t SUPER;
};

struct cff1_cs_opset_flatten_t : cff1_cs_opset_t<cff1_cs_opset_flatten_t, flatten_param_t>
{
  static void flush_args_and_op (op_code_t op, cff1_cs_interp_env_t &env, flatten_param_t& param)
  {
    if (env.arg_start > 0)
      flush_width (env, param);

    switch (op)
    {
      case OpCode_hstem:
      case OpCode_hstemhm:
      case OpCode_vstem:
      case OpCode_vstemhm:
      case OpCode_hintmask:
      case OpCode_cntrmask:
      case OpCode_dotsection:
	if (param.drop_hints)
	{
	  env.clear_args ();
	  return;
	}
	HB_FALLTHROUGH;

      default:
	SUPER::flush_args_and_op (op, env, param);
	break;
    }
  }
  static void flush_args (cff1_cs_interp_env_t &env, flatten_param_t& param)
  {
    str_encoder_t  encoder (param.flatStr);
    for (unsigned int i = env.arg_start; i < env.argStack.get_count (); i++)
      encoder.encode_num_cs (env.eval_arg (i));
    SUPER::flush_args (env, param);
  }

  static void flush_op (op_code_t op, cff1_cs_interp_env_t &env, flatten_param_t& param)
  {
    str_encoder_t  encoder (param.flatStr);
    encoder.encode_op (op);
  }

  static void flush_width (cff1_cs_interp_env_t &env, flatten_param_t& param)
  {
    assert (env.has_width);
    str_encoder_t  encoder (param.flatStr);
    encoder.encode_num_cs (env.width);
  }

  static void flush_hintmask (op_code_t op, cff1_cs_interp_env_t &env, flatten_param_t& param)
  {
    SUPER::flush_hintmask (op, env, param);
    if (!param.drop_hints)
    {
      str_encoder_t  encoder (param.flatStr);
      for (unsigned int i = 0; i < env.hintmask_size; i++)
	encoder.encode_byte (env.str_ref[i]);
    }
  }

  private:
  typedef cff1_cs_opset_t<cff1_cs_opset_flatten_t, flatten_param_t> SUPER;
};

struct range_list_t : hb_vector_t<code_pair_t>
{
  /* replace the first glyph ID in the "glyph" field each range with a nLeft value */
  bool complete (unsigned int last_glyph)
  {
    hb_codepoint_t all_glyphs = 0;
    unsigned count = this->length;
    for (unsigned int i = count; i; i--)
    {
      code_pair_t &pair = arrayZ[i - 1];
      unsigned int nLeft = last_glyph - pair.glyph - 1;
      all_glyphs |= nLeft;
      last_glyph = pair.glyph;
      pair.glyph = nLeft;
    }
    bool two_byte = all_glyphs >= 0x100;
    return two_byte;
  }
};

struct cff1_cs_opset_subr_subset_t : cff1_cs_opset_t<cff1_cs_opset_subr_subset_t, subr_subset_param_t>
{
  static void process_op (op_code_t op, cff1_cs_interp_env_t &env, subr_subset_param_t& param)
  {
    switch (op) {

      case OpCode_return:
	param.current_parsed_str->add_op (op, env.str_ref);
	param.current_parsed_str->set_parsed ();
	env.return_from_subr ();
	param.set_current_str (env, false);
	break;

      case OpCode_endchar:
	param.current_parsed_str->add_op (op, env.str_ref);
	param.current_parsed_str->set_parsed ();
	SUPER::process_op (op, env, param);
	break;

      case OpCode_callsubr:
	process_call_subr (op, CSType_LocalSubr, env, param, env.localSubrs, param.local_closure);
	break;

      case OpCode_callgsubr:
	process_call_subr (op, CSType_GlobalSubr, env, param, env.globalSubrs, param.global_closure);
	break;

      default:
	SUPER::process_op (op, env, param);
	param.current_parsed_str->add_op (op, env.str_ref);
	break;
    }
  }

  protected:
  static void process_call_subr (op_code_t op, cs_type_t type,
				 cff1_cs_interp_env_t &env, subr_subset_param_t& param,
				 cff1_biased_subrs_t& subrs, hb_set_t *closure)
  {
    byte_str_ref_t    str_ref = env.str_ref;
    env.call_subr (subrs, type);
    param.current_parsed_str->add_call_op (op, str_ref, env.context.subr_num);
    closure->add (env.context.subr_num);
    param.set_current_str (env, true);
  }

  private:
  typedef cff1_cs_opset_t<cff1_cs_opset_subr_subset_t, subr_subset_param_t> SUPER;
};

struct cff1_private_dict_op_serializer_t : op_serializer_t
{
  cff1_private_dict_op_serializer_t (bool desubroutinize_, bool drop_hints_)
    : desubroutinize (desubroutinize_), drop_hints (drop_hints_) {}

  bool serialize (hb_serialize_context_t *c,
		  const op_str_t &opstr,
		  objidx_t subrs_link) const
  {
    TRACE_SERIALIZE (this);

    if (drop_hints && dict_opset_t::is_hint_op (opstr.op))
      return_trace (true);

    if (opstr.op == OpCode_Subrs)
    {
      if (desubroutinize || !subrs_link)
	return_trace (true);
      else
	return_trace (FontDict::serialize_link2_op (c, opstr.op, subrs_link));
    }

    return_trace (copy_opstr (c, opstr));
  }

  protected:
  const bool desubroutinize;
  const bool drop_hints;
};

struct cff1_subr_subsetter_t : subr_subsetter_t<cff1_subr_subsetter_t, CFF1Subrs, const OT::cff1::accelerator_subset_t, cff1_cs_interp_env_t, cff1_cs_opset_subr_subset_t, OpCode_endchar>
{
  cff1_subr_subsetter_t (const OT::cff1::accelerator_subset_t &acc_, const hb_subset_plan_t *plan_)
    : subr_subsetter_t (acc_, plan_) {}

  static void complete_parsed_str (cff1_cs_interp_env_t &env, subr_subset_param_t& param, parsed_cs_str_t &charstring)
  {
    /* insert width at the beginning of the charstring as necessary */
    if (env.has_width)
      charstring.set_prefix (env.width);

    /* subroutines/charstring left on the call stack are legally left unmarked
     * unmarked when a subroutine terminates with endchar. mark them.
     */
    param.current_parsed_str->set_parsed ();
    for (unsigned int i = 0; i < env.callStack.get_count (); i++)
    {
      parsed_cs_str_t *parsed_str = param.get_parsed_str_for_context (env.callStack[i]);
      if (likely (parsed_str))
	parsed_str->set_parsed ();
      else
	env.set_error ();
    }
  }
};

namespace OT {
struct cff1_subset_plan
{
  cff1_subset_plan ()
  {
    for (unsigned int i = 0; i < name_dict_values_t::ValCount; i++)
      topDictModSIDs[i] = CFF_UNDEF_SID;
  }

  void plan_subset_encoding (const OT::cff1::accelerator_subset_t &acc, hb_subset_plan_t *plan)
  {
    const Encoding *encoding = acc.encoding;
    unsigned int  size0, size1;
    unsigned code, last_code = CFF_UNDEF_CODE - 1;
    hb_vector_t<hb_codepoint_t> supp_codes;

    if (unlikely (!subset_enc_code_ranges.resize (0)))
    {
      plan->check_success (false);
      return;
    }

    supp_codes.init ();

    code_pair_t glyph_to_sid_cache {0, HB_CODEPOINT_INVALID};
    subset_enc_num_codes = plan->num_output_glyphs () - 1;
    unsigned int glyph;
    auto it = hb_iter (plan->new_to_old_gid_list);
    if (it->first == 0) it++;
    auto _ = *it;
    for (glyph = 1; glyph < num_glyphs; glyph++)
    {
      hb_codepoint_t old_glyph;
      if (glyph == _.first)
      {
	old_glyph = _.second;
	_ = *++it;
      }
      else
      {
	/* Retain the SID for the old missing glyph ID */
	old_glyph = glyph;
      }
      code = acc.glyph_to_code (old_glyph, &glyph_to_sid_cache);
      if (code == CFF_UNDEF_CODE)
      {
	subset_enc_num_codes = glyph - 1;
	break;
      }

      if (code != last_code + 1)
	subset_enc_code_ranges.push (code_pair_t {code, glyph});
      last_code = code;

      if (encoding != &Null (Encoding))
      {
	hb_codepoint_t  sid = acc.glyph_to_sid (old_glyph, &glyph_to_sid_cache);
	encoding->get_supplement_codes (sid, supp_codes);
	for (unsigned int i = 0; i < supp_codes.length; i++)
	  subset_enc_supp_codes.push (code_pair_t {supp_codes[i], sid});
      }
    }
    supp_codes.fini ();

    subset_enc_code_ranges.complete (glyph);

    assert (subset_enc_num_codes <= 0xFF);
    size0 = Encoding0::min_size + HBUINT8::static_size * subset_enc_num_codes;
    size1 = Encoding1::min_size + Encoding1_Range::static_size * subset_enc_code_ranges.length;

    if (size0 < size1)
      subset_enc_format = 0;
    else
      subset_enc_format = 1;
  }

  bool plan_subset_charset (const OT::cff1::accelerator_subset_t &acc, hb_subset_plan_t *plan)
  {
    unsigned int  size0, size_ranges;
    unsigned last_sid = CFF_UNDEF_CODE - 1;

    if (unlikely (!subset_charset_ranges.resize (0)))
    {
      plan->check_success (false);
      return false;
    }

    code_pair_t glyph_to_sid_cache {0, HB_CODEPOINT_INVALID};

    unsigned num_glyphs = plan->num_output_glyphs ();

    if (unlikely (!subset_charset_ranges.alloc (hb_min (num_glyphs,
							acc.num_charset_entries))))
    {
      plan->check_success (false);
      return false;
    }

    glyph_to_sid_map_t *glyph_to_sid_map = acc.cff_accelerator ?
					   acc.cff_accelerator->glyph_to_sid_map.get_acquire () :
					   nullptr;
    bool created_map = false;
    if (!glyph_to_sid_map && acc.cff_accelerator)
    {
      created_map = true;
      glyph_to_sid_map = acc.create_glyph_to_sid_map ();
    }

    auto it = hb_iter (plan->new_to_old_gid_list);
    if (it->first == 0) it++;
    auto _ = *it;
    bool not_is_cid = !acc.is_CID ();
    bool skip = !not_is_cid && glyph_to_sid_map;
    if (not_is_cid)
      sidmap.alloc (num_glyphs);
    for (hb_codepoint_t glyph = 1; glyph < num_glyphs; glyph++)
    {
      hb_codepoint_t old_glyph;
      if (glyph == _.first)
      {
	old_glyph = _.second;
	_ = *++it;
      }
      else
      {
	/* Retain the SID for the old missing glyph ID */
	old_glyph = glyph;
      }
      unsigned sid = glyph_to_sid_map ?
		     glyph_to_sid_map->arrayZ[old_glyph].code :
		     acc.glyph_to_sid (old_glyph, &glyph_to_sid_cache);

      if (not_is_cid)
	sid = sidmap.add (sid);

      if (sid != last_sid + 1)
	subset_charset_ranges.push (code_pair_t {sid, glyph});

      if (glyph == old_glyph && skip)
      {
	glyph = hb_min (_.first - 1, glyph_to_sid_map->arrayZ[old_glyph].glyph);
	sid += glyph - old_glyph;
      }
      last_sid = sid;
    }

    if (created_map)
    {
      if ((!plan->accelerator && acc.cff_accelerator) ||
	  !acc.cff_accelerator->glyph_to_sid_map.cmpexch (nullptr, glyph_to_sid_map))
      {
	glyph_to_sid_map->~glyph_to_sid_map_t ();
	hb_free (glyph_to_sid_map);
      }
    }

    bool two_byte = subset_charset_ranges.complete (num_glyphs);

    size0 = Charset0::get_size (plan->num_output_glyphs ());
    if (!two_byte)
      size_ranges = Charset1::get_size_for_ranges (subset_charset_ranges.length);
    else
      size_ranges = Charset2::get_size_for_ranges (subset_charset_ranges.length);

    if (size0 < size_ranges)
      subset_charset_format = 0;
    else if (!two_byte)
      subset_charset_format = 1;
    else
      subset_charset_format = 2;

    return true;
  }

  bool collect_sids_in_dicts (const OT::cff1::accelerator_subset_t &acc)
  {
    for (unsigned int i = 0; i < name_dict_values_t::ValCount; i++)
    {
      unsigned int sid = acc.topDict.nameSIDs[i];
      if (sid != CFF_UNDEF_SID)
      {
	topDictModSIDs[i] = sidmap.add (sid);
      }
    }

    if (acc.fdArray != &Null (CFF1FDArray))
      for (unsigned int i = 0; i < orig_fdcount; i++)
	if (fdmap.has (i))
	  (void)sidmap.add (acc.fontDicts[i].fontName);

    return true;
  }

  bool create (const OT::cff1::accelerator_subset_t &acc,
	       hb_subset_plan_t *plan)
  {
    /* make sure notdef is first */
    hb_codepoint_t old_glyph;
    if (!plan->old_gid_for_new_gid (0, &old_glyph) || (old_glyph != 0)) return false;

    num_glyphs = plan->num_output_glyphs ();
    orig_fdcount = acc.fdCount;
    drop_hints = plan->flags & HB_SUBSET_FLAGS_NO_HINTING;
    desubroutinize = plan->flags & HB_SUBSET_FLAGS_DESUBROUTINIZE;

 #ifdef HB_EXPERIMENTAL_API
    min_charstrings_off_size = (plan->flags & HB_SUBSET_FLAGS_IFTB_REQUIREMENTS) ? 4 : 0;
 #else
    min_charstrings_off_size = 0;
 #endif

    subset_charset = !acc.is_predef_charset ();
    if (!subset_charset)
      /* check whether the subset renumbers any glyph IDs */
      for (const auto &_ : plan->new_to_old_gid_list)
      {
	if (_.first != _.second)
	{
	  subset_charset = true;
	  break;
	}
      }

    subset_encoding = !acc.is_CID() && !acc.is_predef_encoding ();

    /* top dict INDEX */
    {
      /* Add encoding/charset to a (copy of) top dict as necessary */
      topdict_mod.init (&acc.topDict);
      bool need_to_add_enc = (subset_encoding && !acc.topDict.has_op (OpCode_Encoding));
      bool need_to_add_set = (subset_charset && !acc.topDict.has_op (OpCode_charset));
      if (need_to_add_enc || need_to_add_set)
      {
	if (need_to_add_enc)
	  topdict_mod.add_op (OpCode_Encoding);
	if (need_to_add_set)
	  topdict_mod.add_op (OpCode_charset);
      }
    }

    /* Determine re-mapping of font index as fdmap among other info */
    if (acc.fdSelect != &Null (CFF1FDSelect))
    {
	if (unlikely (!hb_plan_subset_cff_fdselect (plan,
				  orig_fdcount,
				  *acc.fdSelect,
				  subset_fdcount,
				  info.fd_select.size,
				  subset_fdselect_format,
				  subset_fdselect_ranges,
				  fdmap)))
	return false;
    }
    else
      fdmap.identity (1);

    /* remove unused SIDs & reassign SIDs */
    {
      /* SIDs for name strings in dicts are added before glyph names so they fit in 16-bit int range */
      if (unlikely (!collect_sids_in_dicts (acc)))
	return false;
      if (unlikely (sidmap.get_population () > 0x8000))	/* assumption: a dict won't reference that many strings */
	return false;

      if (subset_charset && !plan_subset_charset (acc, plan))
        return false;

      topdict_mod.reassignSIDs (sidmap);
    }

    if (desubroutinize)
    {
      /* Flatten global & local subrs */
      subr_flattener_t<const OT::cff1::accelerator_subset_t, cff1_cs_interp_env_t, cff1_cs_opset_flatten_t, OpCode_endchar>
		    flattener(acc, plan);
      if (!flattener.flatten (subset_charstrings))
	return false;
    }
    else
    {
      cff1_subr_subsetter_t       subr_subsetter (acc, plan);

      /* Subset subrs: collect used subroutines, leaving all unused ones behind */
      if (!subr_subsetter.subset ())
	return false;

      /* encode charstrings, global subrs, local subrs with new subroutine numbers */
      if (!subr_subsetter.encode_charstrings (subset_charstrings))
	return false;

      if (!subr_subsetter.encode_globalsubrs (subset_globalsubrs))
	return false;

      /* local subrs */
      if (!subset_localsubrs.resize (orig_fdcount))
	return false;
      for (unsigned int fd = 0; fd < orig_fdcount; fd++)
      {
	subset_localsubrs[fd].init ();
	if (fdmap.has (fd))
	{
	  if (!subr_subsetter.encode_localsubrs (fd, subset_localsubrs[fd]))
	    return false;
	}
      }
    }

    /* Encoding */
    if (subset_encoding)
      plan_subset_encoding (acc, plan);

    /* private dicts & local subrs */
    if (!acc.is_CID ())
      fontdicts_mod.push (cff1_font_dict_values_mod_t ());
    else
    {
      + hb_iter (acc.fontDicts)
      | hb_filter ([&] (const cff1_font_dict_values_t &_)
	{ return fdmap.has (&_ - &acc.fontDicts[0]); } )
      | hb_map ([&] (const cff1_font_dict_values_t &_)
	{
	  cff1_font_dict_values_mod_t mod;
	  mod.init (&_, sidmap[_.fontName]);
	  return mod;
	})
      | hb_sink (fontdicts_mod)
      ;
    }

    return !plan->in_error () &&
	   (subset_charstrings.length == plan->num_output_glyphs ()) &&
	   (fontdicts_mod.length == subset_fdcount);
  }

  cff1_top_dict_values_mod_t	topdict_mod;
  cff1_sub_table_info_t		info;

  unsigned int    num_glyphs;
  unsigned int    orig_fdcount = 0;
  unsigned int    subset_fdcount = 1;
  unsigned int    subset_fdselect_format = 0;
  hb_vector_t<code_pair_t>   subset_fdselect_ranges;

  /* font dict index remap table from fullset FDArray to subset FDArray.
   * set to CFF_UNDEF_CODE if excluded from subset */
  hb_inc_bimap_t   fdmap;

  str_buff_vec_t		subset_charstrings;
  str_buff_vec_t		subset_globalsubrs;
  hb_vector_t<str_buff_vec_t>	subset_localsubrs;
  hb_vector_t<cff1_font_dict_values_mod_t>  fontdicts_mod;

  bool		drop_hints = false;

  bool		gid_renum;
  bool		subset_encoding;
  uint8_t	subset_enc_format;
  unsigned int	subset_enc_num_codes;
  range_list_t	subset_enc_code_ranges;
  hb_vector_t<code_pair_t>  subset_enc_supp_codes;

  uint8_t	subset_charset_format;
  range_list_t	subset_charset_ranges;
  bool		subset_charset;

  remap_sid_t	sidmap;
  unsigned int	topDictModSIDs[name_dict_values_t::ValCount];

  bool		desubroutinize = false;

  unsigned	min_charstrings_off_size = 0;
};
} // namespace OT

static bool _serialize_cff1_charstrings (hb_serialize_context_t *c,
                                         struct OT::cff1_subset_plan &plan,
                                         const OT::cff1::accelerator_subset_t  &acc)
{
  c->push<CFF1CharStrings> ();

  unsigned data_size = 0;
  unsigned total_size = CFF1CharStrings::total_size (plan.subset_charstrings, &data_size, plan.min_charstrings_off_size);
  if (unlikely (!c->start_zerocopy (total_size)))
    return false;

  auto *cs = c->start_embed<CFF1CharStrings> ();
  if (unlikely (!cs->serialize (c, plan.subset_charstrings, &data_size, plan.min_charstrings_off_size))) {
    c->pop_discard ();
    return false;
  }

  plan.info.char_strings_link = c->pop_pack (false);
  return true;
}

bool
OT::cff1::accelerator_subset_t::serialize (hb_serialize_context_t *c,
					   struct OT::cff1_subset_plan &plan) const
{
  /* push charstrings onto the object stack first which will ensure it packs as the last
     object in the table. Keeping the chastrings last satisfies the requirements for patching
     via IFTB. If this ordering needs to be changed in the future, charstrings should be left
     at the end whenever HB_SUBSET_FLAGS_ITFB_REQUIREMENTS is enabled. */
  if (!_serialize_cff1_charstrings(c, plan, *this))
    return false;

  /* private dicts & local subrs */
  for (int i = (int) privateDicts.length; --i >= 0 ;)
  {
    if (plan.fdmap.has (i))
    {
      objidx_t	subrs_link = 0;
      if (plan.subset_localsubrs[i].length > 0)
      {
	auto *dest = c->push <CFF1Subrs> ();
	if (likely (dest->serialize (c, plan.subset_localsubrs[i])))
	  subrs_link = c->pop_pack ();
	else
	{
	  c->pop_discard ();
	  return false;
	}
      }

      auto *pd = c->push<PrivateDict> ();
      cff1_private_dict_op_serializer_t privSzr (plan.desubroutinize, plan.drop_hints);
      /* N.B. local subrs immediately follows its corresponding private dict. i.e., subr offset == private dict size */
      if (likely (pd->serialize (c, privateDicts[i], privSzr, subrs_link)))
      {
	unsigned fd = plan.fdmap[i];
	plan.fontdicts_mod[fd].privateDictInfo.size = c->length ();
	plan.fontdicts_mod[fd].privateDictInfo.link = c->pop_pack ();
      }
      else
      {
	c->pop_discard ();
	return false;
      }
    }
  }

  if (!is_CID ())
    plan.info.privateDictInfo = plan.fontdicts_mod[0].privateDictInfo;

  /* FDArray (FD Index) */
  if (fdArray != &Null (CFF1FDArray))
  {
    auto *fda = c->push<CFF1FDArray> ();
    cff1_font_dict_op_serializer_t  fontSzr;
    auto it = + hb_zip (+ hb_iter (plan.fontdicts_mod), + hb_iter (plan.fontdicts_mod));
    if (likely (fda->serialize (c, it, fontSzr)))
      plan.info.fd_array_link = c->pop_pack (false);
    else
    {
      c->pop_discard ();
      return false;
    }
  }

  /* FDSelect */
  if (fdSelect != &Null (CFF1FDSelect))
  {
    c->push ();
    if (likely (hb_serialize_cff_fdselect (c, plan.num_glyphs, *fdSelect, fdCount,
					   plan.subset_fdselect_format, plan.info.fd_select.size,
					   plan.subset_fdselect_ranges)))
      plan.info.fd_select.link = c->pop_pack ();
    else
    {
      c->pop_discard ();
      return false;
    }
  }

  /* Charset */
  if (plan.subset_charset)
  {
    auto *dest = c->push<Charset> ();
    if (likely (dest->serialize (c,
				 plan.subset_charset_format,
				 plan.num_glyphs,
				 plan.subset_charset_ranges)))
      plan.info.charset_link = c->pop_pack ();
    else
    {
      c->pop_discard ();
      return false;
    }
  }

  /* Encoding */
  if (plan.subset_encoding)
  {
    auto *dest = c->push<Encoding> ();
    if (likely (dest->serialize (c,
				 plan.subset_enc_format,
				 plan.subset_enc_num_codes,
				 plan.subset_enc_code_ranges,
				 plan.subset_enc_supp_codes)))
      plan.info.encoding_link = c->pop_pack ();
    else
    {
      c->pop_discard ();
      return false;
    }
  }

  /* global subrs */
  {
    auto *dest = c->push <CFF1Subrs> ();
    if (likely (dest->serialize (c, plan.subset_globalsubrs)))
      c->pop_pack (false);
    else
    {
      c->pop_discard ();
      return false;
    }
  }

  /* String INDEX */
  {
    auto *dest = c->push<CFF1StringIndex> ();
    if (likely (!plan.sidmap.in_error () &&
		dest->serialize (c, *stringIndex, plan.sidmap.vector)))
      c->pop_pack ();
    else
    {
      c->pop_discard ();
      return false;
    }
  }

  OT::cff1 *cff = c->allocate_min<OT::cff1> ();
  if (unlikely (!cff))
    return false;

  /* header */
  cff->version.major = 0x01;
  cff->version.minor = 0x00;
  cff->nameIndex = cff->min_size;
  cff->offSize = 4; /* unused? */

  /* name INDEX */
  if (unlikely (!c->embed (*nameIndex))) return false;

  /* top dict INDEX */
  {
    /* serialize singleton TopDict */
    auto *top = c->push<TopDict> ();
    cff1_top_dict_op_serializer_t topSzr;
    unsigned top_size = 0;
    top_dict_modifiers_t  modifier (plan.info, plan.topDictModSIDs);
    if (likely (top->serialize (c, plan.topdict_mod, topSzr, modifier)))
    {
      top_size = c->length ();
      c->pop_pack (false);
    }
    else
    {
      c->pop_discard ();
      return false;
    }
    /* serialize INDEX header for above */
    auto *dest = c->start_embed<CFF1Index> ();
    return dest->serialize_header (c, hb_iter (&top_size, 1), top_size);
  }
}

bool
OT::cff1::accelerator_subset_t::subset (hb_subset_context_t *c) const
{
  cff1_subset_plan cff_plan;

  if (unlikely (!cff_plan.create (*this, c->plan)))
  {
    DEBUG_MSG(SUBSET, nullptr, "Failed to generate a cff subsetting plan.");
    return false;
  }

  return serialize (c->serializer, cff_plan);
}


#endif
