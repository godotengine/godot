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
#ifndef HB_CFF1_INTERP_CS_HH
#define HB_CFF1_INTERP_CS_HH

#include "hb.hh"
#include "hb-cff-interp-cs-common.hh"

namespace CFF {

using namespace OT;

typedef biased_subrs_t<CFF1Subrs>   cff1_biased_subrs_t;

struct cff1_cs_interp_env_t : cs_interp_env_t<number_t, CFF1Subrs>
{
  template <typename ACC>
  cff1_cs_interp_env_t (const hb_ubytes_t &str, ACC &acc, unsigned int fd,
			const int *coords_=nullptr, unsigned int num_coords_=0)
    : SUPER (str, acc.globalSubrs, acc.privateDicts[fd].localSubrs)
  {
    processed_width = false;
    has_width = false;
    arg_start = 0;
    in_seac = false;
  }

  void set_width (bool has_width_)
  {
    if (likely (!processed_width && (SUPER::argStack.get_count () > 0)))
    {
      if (has_width_)
      {
	width = SUPER::argStack[0];
	has_width = true;
	arg_start = 1;
      }
    }
    processed_width = true;
  }

  void clear_args ()
  {
    arg_start = 0;
    SUPER::clear_args ();
  }

  void set_in_seac (bool _in_seac) { in_seac = _in_seac; }

  bool	  processed_width;
  bool	  has_width;
  unsigned int  arg_start;
  number_t	width;
  bool	  in_seac;

  private:
  typedef cs_interp_env_t<number_t, CFF1Subrs> SUPER;
};

template <typename OPSET, typename PARAM, typename PATH=path_procs_null_t<cff1_cs_interp_env_t, PARAM>>
struct cff1_cs_opset_t : cs_opset_t<number_t, OPSET, cff1_cs_interp_env_t, PARAM, PATH>
{
  /* PostScript-originated legacy opcodes (OpCode_add etc) are unsupported */
  /* Type 1-originated deprecated opcodes, seac behavior of endchar and dotsection are supported */

  static void process_op (op_code_t op, cff1_cs_interp_env_t &env, PARAM& param)
  {
    switch (op) {
      case OpCode_dotsection:
	SUPER::flush_args_and_op (op, env, param);
	break;

      case OpCode_endchar:
	OPSET::check_width (op, env, param);
	if (env.argStack.get_count () >= 4)
	{
	  OPSET::process_seac (env, param);
	}
	OPSET::flush_args_and_op (op, env, param);
	env.set_endchar (true);
	break;

      default:
	SUPER::process_op (op, env, param);
    }
  }

  static void check_width (op_code_t op, cff1_cs_interp_env_t &env, PARAM& param)
  {
    if (!env.processed_width)
    {
      bool  has_width = false;
      switch (op)
      {
	case OpCode_endchar:
	case OpCode_hstem:
	case OpCode_hstemhm:
	case OpCode_vstem:
	case OpCode_vstemhm:
	case OpCode_hintmask:
	case OpCode_cntrmask:
	  has_width = ((env.argStack.get_count () & 1) != 0);
	  break;
	case OpCode_hmoveto:
	case OpCode_vmoveto:
	  has_width = (env.argStack.get_count () > 1);
	  break;
	case OpCode_rmoveto:
	  has_width = (env.argStack.get_count () > 2);
	  break;
	default:
	  return;
      }
      env.set_width (has_width);
    }
  }

  static void process_seac (cff1_cs_interp_env_t &env, PARAM& param)
  {
  }

  static void flush_args (cff1_cs_interp_env_t &env, PARAM& param)
  {
    SUPER::flush_args (env, param);
    env.clear_args ();  /* pop off width */
  }

  private:
  typedef cs_opset_t<number_t, OPSET, cff1_cs_interp_env_t, PARAM, PATH>  SUPER;
};

template <typename OPSET, typename PARAM>
using cff1_cs_interpreter_t = cs_interpreter_t<cff1_cs_interp_env_t, OPSET, PARAM>;

} /* namespace CFF */

#endif /* HB_CFF1_INTERP_CS_HH */
