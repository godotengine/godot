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
#ifndef HB_CFF2_INTERP_CS_HH
#define HB_CFF2_INTERP_CS_HH

#include "hb.hh"
#include "hb-cff-interp-cs-common.hh"

namespace CFF {

using namespace OT;

struct blend_arg_t : number_t
{
  void set_int (int v) { reset_blends (); number_t::set_int (v); }
  void set_fixed (int32_t v) { reset_blends (); number_t::set_fixed (v); }
  void set_real (double v) { reset_blends (); number_t::set_real (v); }

  void set_blends (unsigned int numValues_, unsigned int valueIndex_,
		   hb_array_t<const blend_arg_t> blends_)
  {
    numValues = numValues_;
    valueIndex = valueIndex_;
    unsigned numBlends = blends_.length;
    if (unlikely (!deltas.resize_exact (numBlends)))
      return;
    for (unsigned int i = 0; i < numBlends; i++)
      deltas.arrayZ[i] = blends_.arrayZ[i];
  }

  bool blending () const { return deltas.length > 0; }
  void reset_blends ()
  {
    numValues = valueIndex = 0;
    deltas.shrink (0);
  }

  unsigned int numValues;
  unsigned int valueIndex;
  hb_vector_t<number_t> deltas;
};

typedef biased_subrs_t<CFF2Subrs>   cff2_biased_subrs_t;

template <typename ELEM>
struct cff2_cs_interp_env_t : cs_interp_env_t<ELEM, CFF2Subrs>
{
  template <typename ACC>
  cff2_cs_interp_env_t (const hb_ubytes_t &str, ACC &acc, unsigned int fd,
			const int *coords_=nullptr, unsigned int num_coords_=0)
    : SUPER (str, acc.globalSubrs, acc.privateDicts[fd].localSubrs)
  {
    coords = coords_;
    num_coords = num_coords_;
    varStore = acc.varStore;
    do_blend = num_coords && coords && varStore->size;
    set_ivs (acc.privateDicts[fd].ivs);
  }

  void fini ()
  {
    SUPER::fini ();
  }

  op_code_t fetch_op ()
  {
    if (this->str_ref.avail ())
      return SUPER::fetch_op ();

    /* make up return or endchar op */
    if (this->callStack.is_empty ())
      return OpCode_endchar;
    else
      return OpCode_return;
  }

  const ELEM& eval_arg (unsigned int i)
  {
    return SUPER::argStack[i];
  }

  const ELEM& pop_arg ()
  {
    return SUPER::argStack.pop ();
  }

  void process_blend ()
  {
    if (!seen_blend)
    {
      region_count = varStore->varStore.get_region_index_count (get_ivs ());
      if (do_blend)
      {
	if (unlikely (!scalars.resize_exact (region_count)))
	  SUPER::set_error ();
	else
	  varStore->varStore.get_region_scalars (get_ivs (), coords, num_coords,
						 &scalars[0], region_count);
      }
      seen_blend = true;
    }
  }

  void process_vsindex ()
  {
    unsigned int  index = SUPER::argStack.pop_uint ();
    if (unlikely (seen_vsindex () || seen_blend))
    {
     SUPER::set_error ();
    }
    else
    {
      set_ivs (index);
    }
    seen_vsindex_ = true;
  }

  unsigned int get_region_count () const { return region_count; }
  void	 set_region_count (unsigned int region_count_) { region_count = region_count_; }
  unsigned int get_ivs () const { return ivs; }
  void	 set_ivs (unsigned int ivs_) { ivs = ivs_; }
  bool	 seen_vsindex () const { return seen_vsindex_; }

  double blend_deltas (hb_array_t<const ELEM> deltas) const
  {
    double v = 0;
    if (do_blend)
    {
      if (likely (scalars.length == deltas.length))
      {
        unsigned count = scalars.length;
	for (unsigned i = 0; i < count; i++)
	  v += (double) scalars.arrayZ[i] * deltas.arrayZ[i].to_real ();
      }
    }
    return v;
  }

  bool have_coords () const { return num_coords; }

  protected:
  const int     *coords;
  unsigned int  num_coords;
  const	 CFF2ItemVariationStore *varStore;
  unsigned int  region_count;
  unsigned int  ivs;
  hb_vector_t<float>  scalars;
  bool	  do_blend;
  bool	  seen_vsindex_ = false;
  bool	  seen_blend = false;

  typedef cs_interp_env_t<ELEM, CFF2Subrs> SUPER;
};
template <typename OPSET, typename PARAM, typename ELEM, typename PATH=path_procs_null_t<cff2_cs_interp_env_t<ELEM>, PARAM>>
struct cff2_cs_opset_t : cs_opset_t<ELEM, OPSET, cff2_cs_interp_env_t<ELEM>, PARAM, PATH>
{
  static void process_op (op_code_t op, cff2_cs_interp_env_t<ELEM> &env, PARAM& param)
  {
    switch (op) {
      case OpCode_callsubr:
      case OpCode_callgsubr:
	/* a subroutine number shouldn't be a blended value */
#if 0
	if (unlikely (env.argStack.peek ().blending ()))
	{
	  env.set_error ();
	  break;
	}
#endif
	SUPER::process_op (op, env, param);
	break;

      case OpCode_blendcs:
	OPSET::process_blend (env, param);
	break;

      case OpCode_vsindexcs:
#if 0
	if (unlikely (env.argStack.peek ().blending ()))
	{
	  env.set_error ();
	  break;
	}
#endif
	OPSET::process_vsindex (env, param);
	break;

      default:
	SUPER::process_op (op, env, param);
    }
  }

  template <typename T = ELEM,
	    hb_enable_if (hb_is_same (T, blend_arg_t))>
  static void process_arg_blend (cff2_cs_interp_env_t<ELEM> &env,
				 ELEM &arg,
				 const hb_array_t<const ELEM> blends,
				 unsigned n, unsigned i)
  {
    if (env.have_coords ())
      arg.set_int (round (arg.to_real () + env.blend_deltas (blends)));
    else
      arg.set_blends (n, i, blends);
  }
  template <typename T = ELEM,
	    hb_enable_if (!hb_is_same (T, blend_arg_t))>
  static void process_arg_blend (cff2_cs_interp_env_t<ELEM> &env,
				 ELEM &arg,
				 const hb_array_t<const ELEM> blends,
				 unsigned n, unsigned i)
  {
    arg.set_real (arg.to_real () + env.blend_deltas (blends));
  }

  static void process_blend (cff2_cs_interp_env_t<ELEM> &env, PARAM& param)
  {
    unsigned int n, k;

    env.process_blend ();
    k = env.get_region_count ();
    n = env.argStack.pop_uint ();
    /* copy the blend values into blend array of the default values */
    unsigned int start = env.argStack.get_count () - ((k+1) * n);
    /* let an obvious error case fail, but note CFF2 spec doesn't forbid n==0 */
    if (unlikely (start > env.argStack.get_count ()))
    {
      env.set_error ();
      return;
    }
    for (unsigned int i = 0; i < n; i++)
    {
      const hb_array_t<const ELEM> blends = env.argStack.sub_array (start + n + (i * k), k);
      process_arg_blend (env, env.argStack[start + i], blends, n, i);
    }

    /* pop off blend values leaving default values now adorned with blend values */
    env.argStack.pop (k * n);
  }

  static void process_vsindex (cff2_cs_interp_env_t<ELEM> &env, PARAM& param)
  {
    env.process_vsindex ();
    env.clear_args ();
  }

  private:
  typedef cs_opset_t<ELEM, OPSET, cff2_cs_interp_env_t<ELEM>, PARAM, PATH>  SUPER;
};

template <typename OPSET, typename PARAM, typename ELEM>
using cff2_cs_interpreter_t = cs_interpreter_t<cff2_cs_interp_env_t<ELEM>, OPSET, PARAM>;

} /* namespace CFF */

#endif /* HB_CFF2_INTERP_CS_HH */
