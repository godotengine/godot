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
#include "hb-ot-cff2-table.hh"
#include "hb-set.h"
#include "hb-subset-cff2.hh"
#include "hb-subset-plan.hh"
#include "hb-subset-cff-common.hh"
#include "hb-cff2-interp-cs.hh"

using namespace CFF;

struct cff2_sub_table_info_t : cff_sub_table_info_t
{
  cff2_sub_table_info_t ()
    : cff_sub_table_info_t (),
      var_store_link (0)
  {}

  objidx_t  var_store_link;
};

struct cff2_top_dict_op_serializer_t : cff_top_dict_op_serializer_t<>
{
  bool serialize (hb_serialize_context_t *c,
		  const op_str_t &opstr,
		  const cff2_sub_table_info_t &info) const
  {
    TRACE_SERIALIZE (this);

    switch (opstr.op)
    {
      case OpCode_vstore:
	return_trace (FontDict::serialize_link4_op(c, opstr.op, info.var_store_link));

      default:
	return_trace (cff_top_dict_op_serializer_t<>::serialize (c, opstr, info));
    }
  }
};

struct cff2_cs_opset_flatten_t : cff2_cs_opset_t<cff2_cs_opset_flatten_t, flatten_param_t>
{
  static void flush_args_and_op (op_code_t op, cff2_cs_interp_env_t &env, flatten_param_t& param)
  {
    switch (op)
    {
      case OpCode_return:
      case OpCode_endchar:
	/* dummy opcodes in CFF2. ignore */
	break;

      case OpCode_hstem:
      case OpCode_hstemhm:
      case OpCode_vstem:
      case OpCode_vstemhm:
      case OpCode_hintmask:
      case OpCode_cntrmask:
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

  static void flush_args (cff2_cs_interp_env_t &env, flatten_param_t& param)
  {
    for (unsigned int i = 0; i < env.argStack.get_count ();)
    {
      const blend_arg_t &arg = env.argStack[i];
      if (arg.blending ())
      {
	if (unlikely (!((arg.numValues > 0) && (env.argStack.get_count () >= arg.numValues))))
	{
	  env.set_error ();
	  return;
	}
	flatten_blends (arg, i, env, param);
	i += arg.numValues;
      }
      else
      {
	str_encoder_t  encoder (param.flatStr);
	encoder.encode_num (arg);
	i++;
      }
    }
    SUPER::flush_args (env, param);
  }

  static void flatten_blends (const blend_arg_t &arg, unsigned int i, cff2_cs_interp_env_t &env, flatten_param_t& param)
  {
    /* flatten the default values */
    str_encoder_t  encoder (param.flatStr);
    for (unsigned int j = 0; j < arg.numValues; j++)
    {
      const blend_arg_t &arg1 = env.argStack[i + j];
      if (unlikely (!((arg1.blending () && (arg.numValues == arg1.numValues) && (arg1.valueIndex == j) &&
	      (arg1.deltas.length == env.get_region_count ())))))
      {
	env.set_error ();
	return;
      }
      encoder.encode_num (arg1);
    }
    /* flatten deltas for each value */
    for (unsigned int j = 0; j < arg.numValues; j++)
    {
      const blend_arg_t &arg1 = env.argStack[i + j];
      for (unsigned int k = 0; k < arg1.deltas.length; k++)
	encoder.encode_num (arg1.deltas[k]);
    }
    /* flatten the number of values followed by blend operator */
    encoder.encode_int (arg.numValues);
    encoder.encode_op (OpCode_blendcs);
  }

  static void flush_op (op_code_t op, cff2_cs_interp_env_t &env, flatten_param_t& param)
  {
    switch (op)
    {
      case OpCode_return:
      case OpCode_endchar:
	return;
      default:
	str_encoder_t  encoder (param.flatStr);
	encoder.encode_op (op);
    }
  }

  private:
  typedef cff2_cs_opset_t<cff2_cs_opset_flatten_t, flatten_param_t> SUPER;
  typedef cs_opset_t<blend_arg_t, cff2_cs_opset_flatten_t, cff2_cs_opset_flatten_t, cff2_cs_interp_env_t, flatten_param_t> CSOPSET;
};

struct cff2_cs_opset_subr_subset_t : cff2_cs_opset_t<cff2_cs_opset_subr_subset_t, subr_subset_param_t>
{
  static void process_op (op_code_t op, cff2_cs_interp_env_t &env, subr_subset_param_t& param)
  {
    switch (op) {

      case OpCode_return:
	param.current_parsed_str->set_parsed ();
	env.return_from_subr ();
	param.set_current_str (env, false);
	break;

      case OpCode_endchar:
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
				 cff2_cs_interp_env_t &env, subr_subset_param_t& param,
				 cff2_biased_subrs_t& subrs, hb_set_t *closure)
  {
    byte_str_ref_t    str_ref = env.str_ref;
    env.call_subr (subrs, type);
    param.current_parsed_str->add_call_op (op, str_ref, env.context.subr_num);
    closure->add (env.context.subr_num);
    param.set_current_str (env, true);
  }

  private:
  typedef cff2_cs_opset_t<cff2_cs_opset_subr_subset_t, subr_subset_param_t> SUPER;
};

struct cff2_subr_subsetter_t : subr_subsetter_t<cff2_subr_subsetter_t, CFF2Subrs, const OT::cff2::accelerator_subset_t, cff2_cs_interp_env_t, cff2_cs_opset_subr_subset_t>
{
  cff2_subr_subsetter_t (const OT::cff2::accelerator_subset_t &acc_, const hb_subset_plan_t *plan_)
    : subr_subsetter_t (acc_, plan_) {}

  static void complete_parsed_str (cff2_cs_interp_env_t &env, subr_subset_param_t& param, parsed_cs_str_t &charstring)
  {
    /* vsindex is inserted at the beginning of the charstring as necessary */
    if (env.seen_vsindex ())
    {
      number_t  ivs;
      ivs.set_int ((int)env.get_ivs ());
      charstring.set_prefix (ivs, OpCode_vsindexcs);
    }
  }
};

struct cff2_subset_plan {
  cff2_subset_plan ()
    : orig_fdcount (0),
      subset_fdcount(1),
      subset_fdselect_size (0),
      subset_fdselect_format (0),
      drop_hints (false),
      desubroutinize (false)
  {
    subset_fdselect_ranges.init ();
    fdmap.init ();
    subset_charstrings.init ();
    subset_globalsubrs.init ();
    subset_localsubrs.init ();
  }

  ~cff2_subset_plan ()
  {
    subset_fdselect_ranges.fini ();
    fdmap.fini ();
    subset_charstrings.fini_deep ();
    subset_globalsubrs.fini_deep ();
    subset_localsubrs.fini_deep ();
  }

  bool create (const OT::cff2::accelerator_subset_t &acc,
	      hb_subset_plan_t *plan)
  {
    orig_fdcount = acc.fdArray->count;

    drop_hints = plan->flags & HB_SUBSET_FLAGS_NO_HINTING;
    desubroutinize = plan->flags & HB_SUBSET_FLAGS_DESUBROUTINIZE;

    if (desubroutinize)
    {
      /* Flatten global & local subrs */
      subr_flattener_t<const OT::cff2::accelerator_subset_t, cff2_cs_interp_env_t, cff2_cs_opset_flatten_t>
		    flattener(acc, plan);
      if (!flattener.flatten (subset_charstrings))
	return false;
    }
    else
    {
      cff2_subr_subsetter_t	subr_subsetter (acc, plan);

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
	if (!subr_subsetter.encode_localsubrs (fd, subset_localsubrs[fd]))
	  return false;
      }
    }

    /* FDSelect */
    if (acc.fdSelect != &Null (CFF2FDSelect))
    {
      if (unlikely (!hb_plan_subset_cff_fdselect (plan,
						  orig_fdcount,
						  *(const FDSelect *)acc.fdSelect,
						  subset_fdcount,
						  subset_fdselect_size,
						  subset_fdselect_format,
						  subset_fdselect_ranges,
						  fdmap)))
	return false;
    }
    else
      fdmap.identity (1);

    return true;
  }

  cff2_sub_table_info_t info;

  unsigned int    orig_fdcount;
  unsigned int    subset_fdcount;
  unsigned int	  subset_fdselect_size;
  unsigned int    subset_fdselect_format;
  hb_vector_t<code_pair_t>   subset_fdselect_ranges;

  hb_inc_bimap_t   fdmap;

  str_buff_vec_t	    subset_charstrings;
  str_buff_vec_t	    subset_globalsubrs;
  hb_vector_t<str_buff_vec_t> subset_localsubrs;

  bool	    drop_hints;
  bool	    desubroutinize;
};

static bool _serialize_cff2 (hb_serialize_context_t *c,
			     cff2_subset_plan &plan,
			     const OT::cff2::accelerator_subset_t  &acc,
			     unsigned int num_glyphs)
{
  /* private dicts & local subrs */
  hb_vector_t<table_info_t>  private_dict_infos;
  if (unlikely (!private_dict_infos.resize (plan.subset_fdcount))) return false;

  for (int i = (int)acc.privateDicts.length; --i >= 0 ;)
  {
    if (plan.fdmap.has (i))
    {
      objidx_t	subrs_link = 0;

      if (plan.subset_localsubrs[i].length > 0)
      {
	CFF2Subrs *dest = c->start_embed <CFF2Subrs> ();
	if (unlikely (!dest)) return false;
	c->push ();
	if (likely (dest->serialize (c, plan.subset_localsubrs[i])))
	  subrs_link = c->pop_pack ();
	else
	{
	  c->pop_discard ();
	  return false;
	}
      }
      PrivateDict *pd = c->start_embed<PrivateDict> ();
      if (unlikely (!pd)) return false;
      c->push ();
      cff_private_dict_op_serializer_t privSzr (plan.desubroutinize, plan.drop_hints);
      if (likely (pd->serialize (c, acc.privateDicts[i], privSzr, subrs_link)))
      {
	unsigned fd = plan.fdmap[i];
	private_dict_infos[fd].size = c->length ();
	private_dict_infos[fd].link = c->pop_pack ();
      }
      else
      {
	c->pop_discard ();
	return false;
      }
    }
  }

  /* CharStrings */
  {
    CFF2CharStrings  *cs = c->start_embed<CFF2CharStrings> ();
    if (unlikely (!cs)) return false;
    c->push ();
    if (likely (cs->serialize (c, plan.subset_charstrings)))
      plan.info.char_strings_link = c->pop_pack ();
    else
    {
      c->pop_discard ();
      return false;
    }
  }

  /* FDSelect */
  if (acc.fdSelect != &Null (CFF2FDSelect))
  {
    c->push ();
    if (likely (hb_serialize_cff_fdselect (c, num_glyphs, *(const FDSelect *)acc.fdSelect, 					      plan.orig_fdcount,
					    plan.subset_fdselect_format, plan.subset_fdselect_size,
					    plan.subset_fdselect_ranges)))
      plan.info.fd_select.link = c->pop_pack ();
    else
    {
      c->pop_discard ();
      return false;
    }
  }

  /* FDArray (FD Index) */
  {
    c->push ();
    CFF2FDArray *fda = c->start_embed<CFF2FDArray> ();
    if (unlikely (!fda)) return false;
    cff_font_dict_op_serializer_t fontSzr;
    auto it =
    + hb_zip (+ hb_iter (acc.fontDicts)
	      | hb_filter ([&] (const cff2_font_dict_values_t &_)
		{ return plan.fdmap.has (&_ - &acc.fontDicts[0]); }),
	      hb_iter (private_dict_infos))
    ;
    if (unlikely (!fda->serialize (c, it, fontSzr))) return false;
    plan.info.fd_array_link = c->pop_pack ();
  }

  /* variation store */
  if (acc.varStore != &Null (CFF2VariationStore))
  {
    c->push ();
    CFF2VariationStore *dest = c->start_embed<CFF2VariationStore> ();
    if (unlikely (!dest || !dest->serialize (c, acc.varStore))) return false;
    plan.info.var_store_link = c->pop_pack ();
  }

  OT::cff2 *cff2 = c->allocate_min<OT::cff2> ();
  if (unlikely (!cff2)) return false;

  /* header */
  cff2->version.major = 0x02;
  cff2->version.minor = 0x00;
  cff2->topDict = OT::cff2::static_size;

  /* top dict */
  {
    TopDict &dict = cff2 + cff2->topDict;
    cff2_top_dict_op_serializer_t topSzr;
    if (unlikely (!dict.serialize (c, acc.topDict, topSzr, plan.info))) return false;
    cff2->topDictSize = c->head - (const char *)&dict;
  }

  /* global subrs */
  {
    CFF2Subrs *dest = c->start_embed <CFF2Subrs> ();
    if (unlikely (!dest)) return false;
    return dest->serialize (c, plan.subset_globalsubrs);
  }
}

static bool
_hb_subset_cff2 (const OT::cff2::accelerator_subset_t  &acc,
		 hb_subset_context_t	*c)
{
  cff2_subset_plan cff2_plan;

  if (unlikely (!cff2_plan.create (acc, c->plan))) return false;
  return _serialize_cff2 (c->serializer, cff2_plan, acc, c->plan->num_output_glyphs ());
}

bool
hb_subset_cff2 (hb_subset_context_t *c)
{
  OT::cff2::accelerator_subset_t acc;
  acc.init (c->plan->source);
  bool result = likely (acc.is_valid ()) && _hb_subset_cff2 (acc, c);
  acc.fini ();

  return result;
}

#endif
