/*
 * CFF2 instancing: round absolute outline points and stem edges before
 * converting them back to relative operands (fonttools/fonttools#4338).
 */

#ifndef HB_SUBSET_CFF2_ROUND_HH
#define HB_SUBSET_CFF2_ROUND_HH

#include "hb-cff2-interp-cs.hh"

namespace CFF {

struct cff2_cs_rounding_t
{
  struct coordinate_t
  {
    double value = 0.;
    hb_vector_t<double> deltas;

    bool round (blend_arg_t &arg, unsigned k)
    {
      if (unlikely (!deltas.resize (k) || !arg.deltas.resize (k)))
	return false;

      double previous = value;
      value += arg.to_real ();
      /* OpenType rounding, including negative half-integers. */
      arg.number_t::set_real (floor (value + .5) - floor (previous + .5));
      bool variable = false;
      for (unsigned r = 0; r < k; r++)
      {
	previous = deltas.arrayZ[r];
	deltas.arrayZ[r] += arg.deltas.arrayZ[r].to_real ();
	double d = floor (deltas.arrayZ[r] + .5) - floor (previous + .5);
	arg.deltas.arrayZ[r].set_real (d);
	variable |= d != 0.;
      }
      if (!variable) arg.deltas.clear ();
      return true;
    }
  };

  bool round (op_code_t op, hb_array_t<blend_arg_t> args, unsigned k)
  {
    unsigned n = args.length;
    switch (op)
    {
      case OpCode_rmoveto:
	return n == 2 && alternating (args, k);
      case OpCode_hmoveto:
      case OpCode_vmoveto:
	return n == 1 && point[op == OpCode_vmoveto].round (args[0], k);
      case OpCode_rlineto:
	return n && !(n % 2) && alternating (args, k);
      case OpCode_rrcurveto:
	return n && !(n % 6) && alternating (args, k);
      case OpCode_rcurveline:
	return n >= 8 && !((n - 2) % 6) && alternating (args, k);
      case OpCode_rlinecurve:
	return n >= 8 && !(n % 2) && alternating (args, k);
      case OpCode_hlineto:
      case OpCode_vlineto:
	return n && alternating (args, k, op == OpCode_vlineto);

      case OpCode_hhcurveto:
      case OpCode_vvcurveto:
      {
	if (unlikely (n < 4 || (n % 4) > 1)) return false;
	unsigned axis = op == OpCode_vvcurveto;
	unsigned i = 0;
	if (n & 1)
	  if (!point[!axis].round (args[i++], k)) return false;
	for (; i < n; i += 4)
	{
	  if (!point[axis].round (args[i], k) ||
	      !point[0].round (args[i + 1], k) ||
	      !point[1].round (args[i + 2], k) ||
	      !point[axis].round (args[i + 3], k)) return false;
	}
	return true;
      }

      case OpCode_hvcurveto:
      case OpCode_vhcurveto:
      {
	if (unlikely (n < 4 || (n % 4) > 1)) return false;
	unsigned axis = op == OpCode_vhcurveto;
	unsigned i = 0;
	for (; i + 4 <= n; i += 4, axis = !axis)
	{
	  if (!point[axis].round (args[i], k) ||
	      !point[0].round (args[i + 1], k) ||
	      !point[1].round (args[i + 2], k) ||
	      !point[!axis].round (args[i + 3], k)) return false;
	  if (i + 5 == n)
	    return point[axis].round (args[i + 4], k);
	}
	return true;
      }

      case OpCode_flex:
	if (unlikely (n != 13 || !alternating (args.sub_array (0, 12), k)))
	  return false;
	args[12].number_t::set_real (floor (args[12].to_real () + .5));
	for (auto &d : args[12].deltas) d.set_real (floor (d.to_real () + .5));
	return true;

      case OpCode_hflex:
      case OpCode_hflex1:
      {
	if (unlikely (n != (op == OpCode_hflex ? 7u : 9u))) return false;
	/* These operators implicitly return to the initial y coordinate. */
	coordinate_t y = point[1];
	if (unlikely (y.deltas.in_error ())) return false;
	for (unsigned i = 0; i < n; i++)
	{
	  unsigned axis = op == OpCode_hflex ? i == 2 : i == 1 || i == 3 || i == 7;
	  if (!point[axis].round (args[i], k)) return false;
	}
	point[1] = std::move (y);
	return true;
      }

      case OpCode_hstem:
      case OpCode_hstemhm:
      case OpCode_vstem:
      case OpCode_vstemhm:
      case OpCode_hintmask:
      case OpCode_cntrmask:
      {
	/* Stem operands are successive edges, relative to zero per operator.
	 * A mask's operands declare implicit vertical stems. */
	coordinate_t stem;
	if (unlikely (n % 2)) return false;
	for (auto &arg : args)
	  if (!stem.round (arg, k)) return false;
	return true;
      }

      default:
	return true;
    }
  }

  private:
  bool alternating (hb_array_t<blend_arg_t> args, unsigned k, unsigned axis = 0)
  {
    for (auto &arg : args)
    {
      if (!point[axis].round (arg, k)) return false;
      axis = !axis;
    }
    return true;
  }

  coordinate_t point[2];
};

} /* namespace CFF */

#endif /* HB_SUBSET_CFF2_ROUND_HH */
