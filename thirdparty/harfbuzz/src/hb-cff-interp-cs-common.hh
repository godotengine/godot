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
#ifndef HB_CFF_INTERP_CS_COMMON_HH
#define HB_CFF_INTERP_CS_COMMON_HH

#include "hb.hh"
#include "hb-cff-interp-common.hh"

namespace CFF {

using namespace OT;

enum cs_type_t {
  CSType_CharString,
  CSType_GlobalSubr,
  CSType_LocalSubr
};

struct call_context_t
{
  void init (const byte_str_ref_t substr_=byte_str_ref_t (), cs_type_t type_=CSType_CharString, unsigned int subr_num_=0)
  {
    str_ref = substr_;
    type = type_;
    subr_num = subr_num_;
  }

  void fini () {}

  byte_str_ref_t  str_ref;
  cs_type_t	  type;
  unsigned int    subr_num;
};

/* call stack */
const unsigned int kMaxCallLimit = 10;
struct call_stack_t : cff_stack_t<call_context_t, kMaxCallLimit> {};

template <typename SUBRS>
struct biased_subrs_t
{
  void init (const SUBRS *subrs_)
  {
    subrs = subrs_;
    unsigned int  nSubrs = get_count ();
    if (nSubrs < 1240)
      bias = 107;
    else if (nSubrs < 33900)
      bias = 1131;
    else
      bias = 32768;
  }

  void fini () {}

  unsigned int get_count () const { return subrs ? subrs->count : 0; }
  unsigned int get_bias () const  { return bias; }

  byte_str_t operator [] (unsigned int index) const
  {
    if (unlikely (!subrs || index >= subrs->count))
      return Null (byte_str_t);
    else
      return (*subrs)[index];
  }

  protected:
  unsigned int  bias;
  const SUBRS   *subrs;
};

struct point_t
{
  void init ()
  {
    x.init ();
    y.init ();
  }

  void set_int (int _x, int _y)
  {
    x.set_int (_x);
    y.set_int (_y);
  }

  void move_x (const number_t &dx) { x += dx; }
  void move_y (const number_t &dy) { y += dy; }
  void move (const number_t &dx, const number_t &dy) { move_x (dx); move_y (dy); }
  void move (const point_t &d) { move_x (d.x); move_y (d.y); }

  number_t  x;
  number_t  y;
};

template <typename ARG, typename SUBRS>
struct cs_interp_env_t : interp_env_t<ARG>
{
  void init (const byte_str_t &str, const SUBRS *globalSubrs_, const SUBRS *localSubrs_)
  {
    interp_env_t<ARG>::init (str);

    context.init (str, CSType_CharString);
    seen_moveto = true;
    seen_hintmask = false;
    hstem_count = 0;
    vstem_count = 0;
    hintmask_size = 0;
    pt.init ();
    callStack.init ();
    globalSubrs.init (globalSubrs_);
    localSubrs.init (localSubrs_);
  }
  void fini ()
  {
    interp_env_t<ARG>::fini ();

    callStack.fini ();
    globalSubrs.fini ();
    localSubrs.fini ();
  }

  bool in_error () const
  {
    return callStack.in_error () || SUPER::in_error ();
  }

  bool pop_subr_num (const biased_subrs_t<SUBRS>& biasedSubrs, unsigned int &subr_num)
  {
    subr_num = 0;
    int n = SUPER::argStack.pop_int ();
    n += biasedSubrs.get_bias ();
    if (unlikely ((n < 0) || ((unsigned int)n >= biasedSubrs.get_count ())))
      return false;

    subr_num = (unsigned int)n;
    return true;
  }

  void call_subr (const biased_subrs_t<SUBRS>& biasedSubrs, cs_type_t type)
  {
    unsigned int subr_num = 0;

    if (unlikely (!pop_subr_num (biasedSubrs, subr_num)
		 || callStack.get_count () >= kMaxCallLimit))
    {
      SUPER::set_error ();
      return;
    }
    context.str_ref = SUPER::str_ref;
    callStack.push (context);

    context.init ( biasedSubrs[subr_num], type, subr_num);
    SUPER::str_ref = context.str_ref;
  }

  void return_from_subr ()
  {
    if (unlikely (SUPER::str_ref.in_error ()))
      SUPER::set_error ();
    context = callStack.pop ();
    SUPER::str_ref = context.str_ref;
  }

  void determine_hintmask_size ()
  {
    if (!seen_hintmask)
    {
      vstem_count += SUPER::argStack.get_count() / 2;
      hintmask_size = (hstem_count + vstem_count + 7) >> 3;
      seen_hintmask = true;
    }
  }

  void set_endchar (bool endchar_flag_) { endchar_flag = endchar_flag_; }
  bool is_endchar () const { return endchar_flag; }

  const number_t &get_x () const { return pt.x; }
  const number_t &get_y () const { return pt.y; }
  const point_t &get_pt () const { return pt; }

  void moveto (const point_t &pt_ ) { pt = pt_; }

  public:
  call_context_t   context;
  bool	  endchar_flag;
  bool	  seen_moveto;
  bool	  seen_hintmask;

  unsigned int  hstem_count;
  unsigned int  vstem_count;
  unsigned int  hintmask_size;
  call_stack_t	callStack;
  biased_subrs_t<SUBRS>   globalSubrs;
  biased_subrs_t<SUBRS>   localSubrs;

  private:
  point_t	 pt;

  typedef interp_env_t<ARG> SUPER;
};

template <typename ENV, typename PARAM>
struct path_procs_null_t
{
  static void rmoveto (ENV &env, PARAM& param) {}
  static void hmoveto (ENV &env, PARAM& param) {}
  static void vmoveto (ENV &env, PARAM& param) {}
  static void rlineto (ENV &env, PARAM& param) {}
  static void hlineto (ENV &env, PARAM& param) {}
  static void vlineto (ENV &env, PARAM& param) {}
  static void rrcurveto (ENV &env, PARAM& param) {}
  static void rcurveline (ENV &env, PARAM& param) {}
  static void rlinecurve (ENV &env, PARAM& param) {}
  static void vvcurveto (ENV &env, PARAM& param) {}
  static void hhcurveto (ENV &env, PARAM& param) {}
  static void vhcurveto (ENV &env, PARAM& param) {}
  static void hvcurveto (ENV &env, PARAM& param) {}
  static void moveto (ENV &env, PARAM& param, const point_t &pt) {}
  static void line (ENV &env, PARAM& param, const point_t &pt1) {}
  static void curve (ENV &env, PARAM& param, const point_t &pt1, const point_t &pt2, const point_t &pt3) {}
  static void hflex (ENV &env, PARAM& param) {}
  static void flex (ENV &env, PARAM& param) {}
  static void hflex1 (ENV &env, PARAM& param) {}
  static void flex1 (ENV &env, PARAM& param) {}
};

template <typename ARG, typename OPSET, typename ENV, typename PARAM, typename PATH=path_procs_null_t<ENV, PARAM>>
struct cs_opset_t : opset_t<ARG>
{
  static void process_op (op_code_t op, ENV &env, PARAM& param)
  {
    switch (op) {

      case OpCode_return:
	env.return_from_subr ();
	break;
      case OpCode_endchar:
	OPSET::check_width (op, env, param);
	env.set_endchar (true);
	OPSET::flush_args_and_op (op, env, param);
	break;

      case OpCode_fixedcs:
	env.argStack.push_fixed_from_substr (env.str_ref);
	break;

      case OpCode_callsubr:
	env.call_subr (env.localSubrs, CSType_LocalSubr);
	break;

      case OpCode_callgsubr:
	env.call_subr (env.globalSubrs, CSType_GlobalSubr);
	break;

      case OpCode_hstem:
      case OpCode_hstemhm:
	OPSET::check_width (op, env, param);
	OPSET::process_hstem (op, env, param);
	break;
      case OpCode_vstem:
      case OpCode_vstemhm:
	OPSET::check_width (op, env, param);
	OPSET::process_vstem (op, env, param);
	break;
      case OpCode_hintmask:
      case OpCode_cntrmask:
	OPSET::check_width (op, env, param);
	OPSET::process_hintmask (op, env, param);
	break;
      case OpCode_rmoveto:
	OPSET::check_width (op, env, param);
	PATH::rmoveto (env, param);
	OPSET::process_post_move (op, env, param);
	break;
      case OpCode_hmoveto:
	OPSET::check_width (op, env, param);
	PATH::hmoveto (env, param);
	OPSET::process_post_move (op, env, param);
	break;
      case OpCode_vmoveto:
	OPSET::check_width (op, env, param);
	PATH::vmoveto (env, param);
	OPSET::process_post_move (op, env, param);
	break;
      case OpCode_rlineto:
	PATH::rlineto (env, param);
	process_post_path (op, env, param);
	break;
      case OpCode_hlineto:
	PATH::hlineto (env, param);
	process_post_path (op, env, param);
	break;
      case OpCode_vlineto:
	PATH::vlineto (env, param);
	process_post_path (op, env, param);
	break;
      case OpCode_rrcurveto:
	PATH::rrcurveto (env, param);
	process_post_path (op, env, param);
	break;
      case OpCode_rcurveline:
	PATH::rcurveline (env, param);
	process_post_path (op, env, param);
	break;
      case OpCode_rlinecurve:
	PATH::rlinecurve (env, param);
	process_post_path (op, env, param);
	break;
      case OpCode_vvcurveto:
	PATH::vvcurveto (env, param);
	process_post_path (op, env, param);
	break;
      case OpCode_hhcurveto:
	PATH::hhcurveto (env, param);
	process_post_path (op, env, param);
	break;
      case OpCode_vhcurveto:
	PATH::vhcurveto (env, param);
	process_post_path (op, env, param);
	break;
      case OpCode_hvcurveto:
	PATH::hvcurveto (env, param);
	process_post_path (op, env, param);
	break;

      case OpCode_hflex:
	PATH::hflex (env, param);
	OPSET::process_post_flex (op, env, param);
	break;

      case OpCode_flex:
	PATH::flex (env, param);
	OPSET::process_post_flex (op, env, param);
	break;

      case OpCode_hflex1:
	PATH::hflex1 (env, param);
	OPSET::process_post_flex (op, env, param);
	break;

      case OpCode_flex1:
	PATH::flex1 (env, param);
	OPSET::process_post_flex (op, env, param);
	break;

      default:
	SUPER::process_op (op, env);
	break;
    }
  }

  static void process_hstem (op_code_t op, ENV &env, PARAM& param)
  {
    env.hstem_count += env.argStack.get_count () / 2;
    OPSET::flush_args_and_op (op, env, param);
  }

  static void process_vstem (op_code_t op, ENV &env, PARAM& param)
  {
    env.vstem_count += env.argStack.get_count () / 2;
    OPSET::flush_args_and_op (op, env, param);
  }

  static void process_hintmask (op_code_t op, ENV &env, PARAM& param)
  {
    env.determine_hintmask_size ();
    if (likely (env.str_ref.avail (env.hintmask_size)))
    {
      OPSET::flush_hintmask (op, env, param);
      env.str_ref.inc (env.hintmask_size);
    }
  }

  static void process_post_flex (op_code_t op, ENV &env, PARAM& param)
  {
    OPSET::flush_args_and_op (op, env, param);
  }

  static void check_width (op_code_t op, ENV &env, PARAM& param)
  {}

  static void process_post_move (op_code_t op, ENV &env, PARAM& param)
  {
    if (!env.seen_moveto)
    {
      env.determine_hintmask_size ();
      env.seen_moveto = true;
    }
    OPSET::flush_args_and_op (op, env, param);
  }

  static void process_post_path (op_code_t op, ENV &env, PARAM& param)
  {
    OPSET::flush_args_and_op (op, env, param);
  }

  static void flush_args_and_op (op_code_t op, ENV &env, PARAM& param)
  {
    OPSET::flush_args (env, param);
    OPSET::flush_op (op, env, param);
  }

  static void flush_args (ENV &env, PARAM& param)
  {
    env.pop_n_args (env.argStack.get_count ());
  }

  static void flush_op (op_code_t op, ENV &env, PARAM& param)
  {
  }

  static void flush_hintmask (op_code_t op, ENV &env, PARAM& param)
  {
    OPSET::flush_args_and_op (op, env, param);
  }

  static bool is_number_op (op_code_t op)
  {
    switch (op)
    {
      case OpCode_shortint:
      case OpCode_fixedcs:
      case OpCode_TwoBytePosInt0: case OpCode_TwoBytePosInt1:
      case OpCode_TwoBytePosInt2: case OpCode_TwoBytePosInt3:
      case OpCode_TwoByteNegInt0: case OpCode_TwoByteNegInt1:
      case OpCode_TwoByteNegInt2: case OpCode_TwoByteNegInt3:
	return true;

      default:
	/* 1-byte integer */
	return (OpCode_OneByteIntFirst <= op) && (op <= OpCode_OneByteIntLast);
    }
  }

  protected:
  typedef opset_t<ARG>  SUPER;
};

template <typename PATH, typename ENV, typename PARAM>
struct path_procs_t
{
  static void rmoveto (ENV &env, PARAM& param)
  {
    point_t pt1 = env.get_pt ();
    const number_t &dy = env.pop_arg ();
    const number_t &dx = env.pop_arg ();
    pt1.move (dx, dy);
    PATH::moveto (env, param, pt1);
  }

  static void hmoveto (ENV &env, PARAM& param)
  {
    point_t pt1 = env.get_pt ();
    pt1.move_x (env.pop_arg ());
    PATH::moveto (env, param, pt1);
  }

  static void vmoveto (ENV &env, PARAM& param)
  {
    point_t pt1 = env.get_pt ();
    pt1.move_y (env.pop_arg ());
    PATH::moveto (env, param, pt1);
  }

  static void rlineto (ENV &env, PARAM& param)
  {
    for (unsigned int i = 0; i + 2 <= env.argStack.get_count (); i += 2)
    {
      point_t pt1 = env.get_pt ();
      pt1.move (env.eval_arg (i), env.eval_arg (i+1));
      PATH::line (env, param, pt1);
    }
  }

  static void hlineto (ENV &env, PARAM& param)
  {
    point_t pt1;
    unsigned int i = 0;
    for (; i + 2 <= env.argStack.get_count (); i += 2)
    {
      pt1 = env.get_pt ();
      pt1.move_x (env.eval_arg (i));
      PATH::line (env, param, pt1);
      pt1.move_y (env.eval_arg (i+1));
      PATH::line (env, param, pt1);
    }
    if (i < env.argStack.get_count ())
    {
      pt1 = env.get_pt ();
      pt1.move_x (env.eval_arg (i));
      PATH::line (env, param, pt1);
    }
  }

  static void vlineto (ENV &env, PARAM& param)
  {
    point_t pt1;
    unsigned int i = 0;
    for (; i + 2 <= env.argStack.get_count (); i += 2)
    {
      pt1 = env.get_pt ();
      pt1.move_y (env.eval_arg (i));
      PATH::line (env, param, pt1);
      pt1.move_x (env.eval_arg (i+1));
      PATH::line (env, param, pt1);
    }
    if (i < env.argStack.get_count ())
    {
      pt1 = env.get_pt ();
      pt1.move_y (env.eval_arg (i));
      PATH::line (env, param, pt1);
    }
  }

  static void rrcurveto (ENV &env, PARAM& param)
  {
    for (unsigned int i = 0; i + 6 <= env.argStack.get_count (); i += 6)
    {
      point_t pt1 = env.get_pt ();
      pt1.move (env.eval_arg (i), env.eval_arg (i+1));
      point_t pt2 = pt1;
      pt2.move (env.eval_arg (i+2), env.eval_arg (i+3));
      point_t pt3 = pt2;
      pt3.move (env.eval_arg (i+4), env.eval_arg (i+5));
      PATH::curve (env, param, pt1, pt2, pt3);
    }
  }

  static void rcurveline (ENV &env, PARAM& param)
  {
    unsigned int arg_count = env.argStack.get_count ();
    if (unlikely (arg_count < 8))
      return;

    unsigned int i = 0;
    unsigned int curve_limit = arg_count - 2;
    for (; i + 6 <= curve_limit; i += 6)
    {
      point_t pt1 = env.get_pt ();
      pt1.move (env.eval_arg (i), env.eval_arg (i+1));
      point_t pt2 = pt1;
      pt2.move (env.eval_arg (i+2), env.eval_arg (i+3));
      point_t pt3 = pt2;
      pt3.move (env.eval_arg (i+4), env.eval_arg (i+5));
      PATH::curve (env, param, pt1, pt2, pt3);
    }

    point_t pt1 = env.get_pt ();
    pt1.move (env.eval_arg (i), env.eval_arg (i+1));
    PATH::line (env, param, pt1);
  }

  static void rlinecurve (ENV &env, PARAM& param)
  {
    unsigned int arg_count = env.argStack.get_count ();
    if (unlikely (arg_count < 8))
      return;

    unsigned int i = 0;
    unsigned int line_limit = arg_count - 6;
    for (; i + 2 <= line_limit; i += 2)
    {
      point_t pt1 = env.get_pt ();
      pt1.move (env.eval_arg (i), env.eval_arg (i+1));
      PATH::line (env, param, pt1);
    }

    point_t pt1 = env.get_pt ();
    pt1.move (env.eval_arg (i), env.eval_arg (i+1));
    point_t pt2 = pt1;
    pt2.move (env.eval_arg (i+2), env.eval_arg (i+3));
    point_t pt3 = pt2;
    pt3.move (env.eval_arg (i+4), env.eval_arg (i+5));
    PATH::curve (env, param, pt1, pt2, pt3);
  }

  static void vvcurveto (ENV &env, PARAM& param)
  {
    unsigned int i = 0;
    point_t pt1 = env.get_pt ();
    if ((env.argStack.get_count () & 1) != 0)
      pt1.move_x (env.eval_arg (i++));
    for (; i + 4 <= env.argStack.get_count (); i += 4)
    {
      pt1.move_y (env.eval_arg (i));
      point_t pt2 = pt1;
      pt2.move (env.eval_arg (i+1), env.eval_arg (i+2));
      point_t pt3 = pt2;
      pt3.move_y (env.eval_arg (i+3));
      PATH::curve (env, param, pt1, pt2, pt3);
      pt1 = env.get_pt ();
    }
  }

  static void hhcurveto (ENV &env, PARAM& param)
  {
    unsigned int i = 0;
    point_t pt1 = env.get_pt ();
    if ((env.argStack.get_count () & 1) != 0)
      pt1.move_y (env.eval_arg (i++));
    for (; i + 4 <= env.argStack.get_count (); i += 4)
    {
      pt1.move_x (env.eval_arg (i));
      point_t pt2 = pt1;
      pt2.move (env.eval_arg (i+1), env.eval_arg (i+2));
      point_t pt3 = pt2;
      pt3.move_x (env.eval_arg (i+3));
      PATH::curve (env, param, pt1, pt2, pt3);
      pt1 = env.get_pt ();
    }
  }

  static void vhcurveto (ENV &env, PARAM& param)
  {
    point_t pt1, pt2, pt3;
    unsigned int i = 0;
    if ((env.argStack.get_count () % 8) >= 4)
    {
      point_t pt1 = env.get_pt ();
      pt1.move_y (env.eval_arg (i));
      point_t pt2 = pt1;
      pt2.move (env.eval_arg (i+1), env.eval_arg (i+2));
      point_t pt3 = pt2;
      pt3.move_x (env.eval_arg (i+3));
      i += 4;

      for (; i + 8 <= env.argStack.get_count (); i += 8)
      {
	PATH::curve (env, param, pt1, pt2, pt3);
	pt1 = env.get_pt ();
	pt1.move_x (env.eval_arg (i));
	pt2 = pt1;
	pt2.move (env.eval_arg (i+1), env.eval_arg (i+2));
	pt3 = pt2;
	pt3.move_y (env.eval_arg (i+3));
	PATH::curve (env, param, pt1, pt2, pt3);

	pt1 = pt3;
	pt1.move_y (env.eval_arg (i+4));
	pt2 = pt1;
	pt2.move (env.eval_arg (i+5), env.eval_arg (i+6));
	pt3 = pt2;
	pt3.move_x (env.eval_arg (i+7));
      }
      if (i < env.argStack.get_count ())
	pt3.move_y (env.eval_arg (i));
      PATH::curve (env, param, pt1, pt2, pt3);
    }
    else
    {
      for (; i + 8 <= env.argStack.get_count (); i += 8)
      {
	pt1 = env.get_pt ();
	pt1.move_y (env.eval_arg (i));
	pt2 = pt1;
	pt2.move (env.eval_arg (i+1), env.eval_arg (i+2));
	pt3 = pt2;
	pt3.move_x (env.eval_arg (i+3));
	PATH::curve (env, param, pt1, pt2, pt3);

	pt1 = pt3;
	pt1.move_x (env.eval_arg (i+4));
	pt2 = pt1;
	pt2.move (env.eval_arg (i+5), env.eval_arg (i+6));
	pt3 = pt2;
	pt3.move_y (env.eval_arg (i+7));
	if ((env.argStack.get_count () - i < 16) && ((env.argStack.get_count () & 1) != 0))
	  pt3.move_x (env.eval_arg (i+8));
	PATH::curve (env, param, pt1, pt2, pt3);
      }
    }
  }

  static void hvcurveto (ENV &env, PARAM& param)
  {
    point_t pt1, pt2, pt3;
    unsigned int i = 0;
    if ((env.argStack.get_count () % 8) >= 4)
    {
      point_t pt1 = env.get_pt ();
      pt1.move_x (env.eval_arg (i));
      point_t pt2 = pt1;
      pt2.move (env.eval_arg (i+1), env.eval_arg (i+2));
      point_t pt3 = pt2;
      pt3.move_y (env.eval_arg (i+3));
      i += 4;

      for (; i + 8 <= env.argStack.get_count (); i += 8)
      {
	PATH::curve (env, param, pt1, pt2, pt3);
	pt1 = env.get_pt ();
	pt1.move_y (env.eval_arg (i));
	pt2 = pt1;
	pt2.move (env.eval_arg (i+1), env.eval_arg (i+2));
	pt3 = pt2;
	pt3.move_x (env.eval_arg (i+3));
	PATH::curve (env, param, pt1, pt2, pt3);

	pt1 = pt3;
	pt1.move_x (env.eval_arg (i+4));
	pt2 = pt1;
	pt2.move (env.eval_arg (i+5), env.eval_arg (i+6));
	pt3 = pt2;
	pt3.move_y (env.eval_arg (i+7));
      }
      if (i < env.argStack.get_count ())
	pt3.move_x (env.eval_arg (i));
      PATH::curve (env, param, pt1, pt2, pt3);
    }
    else
    {
      for (; i + 8 <= env.argStack.get_count (); i += 8)
      {
	pt1 = env.get_pt ();
	pt1.move_x (env.eval_arg (i));
	pt2 = pt1;
	pt2.move (env.eval_arg (i+1), env.eval_arg (i+2));
	pt3 = pt2;
	pt3.move_y (env.eval_arg (i+3));
	PATH::curve (env, param, pt1, pt2, pt3);

	pt1 = pt3;
	pt1.move_y (env.eval_arg (i+4));
	pt2 = pt1;
	pt2.move (env.eval_arg (i+5), env.eval_arg (i+6));
	pt3 = pt2;
	pt3.move_x (env.eval_arg (i+7));
	if ((env.argStack.get_count () - i < 16) && ((env.argStack.get_count () & 1) != 0))
	  pt3.move_y (env.eval_arg (i+8));
	PATH::curve (env, param, pt1, pt2, pt3);
      }
    }
  }

  /* default actions to be overridden */
  static void moveto (ENV &env, PARAM& param, const point_t &pt)
  { env.moveto (pt); }

  static void line (ENV &env, PARAM& param, const point_t &pt1)
  { PATH::moveto (env, param, pt1); }

  static void curve (ENV &env, PARAM& param, const point_t &pt1, const point_t &pt2, const point_t &pt3)
  { PATH::moveto (env, param, pt3); }

  static void hflex (ENV &env, PARAM& param)
  {
    if (likely (env.argStack.get_count () == 7))
    {
      point_t pt1 = env.get_pt ();
      pt1.move_x (env.eval_arg (0));
      point_t pt2 = pt1;
      pt2.move (env.eval_arg (1), env.eval_arg (2));
      point_t pt3 = pt2;
      pt3.move_x (env.eval_arg (3));
      point_t pt4 = pt3;
      pt4.move_x (env.eval_arg (4));
      point_t pt5 = pt4;
      pt5.move_x (env.eval_arg (5));
      pt5.y = pt1.y;
      point_t pt6 = pt5;
      pt6.move_x (env.eval_arg (6));

      curve2 (env, param, pt1, pt2, pt3, pt4, pt5, pt6);
    }
    else
      env.set_error ();
  }

  static void flex (ENV &env, PARAM& param)
  {
    if (likely (env.argStack.get_count () == 13))
    {
      point_t pt1 = env.get_pt ();
      pt1.move (env.eval_arg (0), env.eval_arg (1));
      point_t pt2 = pt1;
      pt2.move (env.eval_arg (2), env.eval_arg (3));
      point_t pt3 = pt2;
      pt3.move (env.eval_arg (4), env.eval_arg (5));
      point_t pt4 = pt3;
      pt4.move (env.eval_arg (6), env.eval_arg (7));
      point_t pt5 = pt4;
      pt5.move (env.eval_arg (8), env.eval_arg (9));
      point_t pt6 = pt5;
      pt6.move (env.eval_arg (10), env.eval_arg (11));

      curve2 (env, param, pt1, pt2, pt3, pt4, pt5, pt6);
    }
    else
      env.set_error ();
  }

  static void hflex1 (ENV &env, PARAM& param)
  {
    if (likely (env.argStack.get_count () == 9))
    {
      point_t pt1 = env.get_pt ();
      pt1.move (env.eval_arg (0), env.eval_arg (1));
      point_t pt2 = pt1;
      pt2.move (env.eval_arg (2), env.eval_arg (3));
      point_t pt3 = pt2;
      pt3.move_x (env.eval_arg (4));
      point_t pt4 = pt3;
      pt4.move_x (env.eval_arg (5));
      point_t pt5 = pt4;
      pt5.move (env.eval_arg (6), env.eval_arg (7));
      point_t pt6 = pt5;
      pt6.move_x (env.eval_arg (8));
      pt6.y = env.get_pt ().y;

      curve2 (env, param, pt1, pt2, pt3, pt4, pt5, pt6);
    }
    else
      env.set_error ();
  }

  static void flex1 (ENV &env, PARAM& param)
  {
    if (likely (env.argStack.get_count () == 11))
    {
      point_t d;
      d.init ();
      for (unsigned int i = 0; i < 10; i += 2)
	d.move (env.eval_arg (i), env.eval_arg (i+1));

      point_t pt1 = env.get_pt ();
      pt1.move (env.eval_arg (0), env.eval_arg (1));
      point_t pt2 = pt1;
      pt2.move (env.eval_arg (2), env.eval_arg (3));
      point_t pt3 = pt2;
      pt3.move (env.eval_arg (4), env.eval_arg (5));
      point_t pt4 = pt3;
      pt4.move (env.eval_arg (6), env.eval_arg (7));
      point_t pt5 = pt4;
      pt5.move (env.eval_arg (8), env.eval_arg (9));
      point_t pt6 = pt5;

      if (fabs (d.x.to_real ()) > fabs (d.y.to_real ()))
      {
	pt6.move_x (env.eval_arg (10));
	pt6.y = env.get_pt ().y;
      }
      else
      {
	pt6.x = env.get_pt ().x;
	pt6.move_y (env.eval_arg (10));
      }

      curve2 (env, param, pt1, pt2, pt3, pt4, pt5, pt6);
    }
    else
      env.set_error ();
  }

  protected:
  static void curve2 (ENV &env, PARAM& param,
		      const point_t &pt1, const point_t &pt2, const point_t &pt3,
		      const point_t &pt4, const point_t &pt5, const point_t &pt6)
  {
    PATH::curve (env, param, pt1, pt2, pt3);
    PATH::curve (env, param, pt4, pt5, pt6);
  }
};

template <typename ENV, typename OPSET, typename PARAM>
struct cs_interpreter_t : interpreter_t<ENV>
{
  bool interpret (PARAM& param)
  {
    SUPER::env.set_endchar (false);

    for (;;) {
      OPSET::process_op (SUPER::env.fetch_op (), SUPER::env, param);
      if (unlikely (SUPER::env.in_error ()))
	return false;
      if (SUPER::env.is_endchar ())
	break;
    }

    return true;
  }

  private:
  typedef interpreter_t<ENV> SUPER;
};

} /* namespace CFF */

#endif /* HB_CFF_INTERP_CS_COMMON_HH */
