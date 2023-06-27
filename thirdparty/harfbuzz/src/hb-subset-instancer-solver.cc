/*
 * Copyright © 2023  Behdad Esfahbod
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

#include "hb.hh"

/* This file is a straight port of the following:
 *
 * https://github.com/fonttools/fonttools/blob/f73220816264fc383b8a75f2146e8d69e455d398/Lib/fontTools/varLib/instancer/solver.py
 *
 * Where that file returns None for a triple, we return Triple{}.
 * This should be safe.
 */

constexpr static float EPSILON = 1.f / (1 << 14);
constexpr static float MAX_F2DOT14 = float (0x7FFF) / (1 << 14);

struct Triple {

  Triple () :
    minimum (0.f), middle (0.f), maximum (0.f) {}

  Triple (float minimum_, float middle_, float maximum_) :
    minimum (minimum_), middle (middle_), maximum (maximum_) {}

  bool operator == (const Triple &o) const
  {
    return minimum == o.minimum &&
	   middle  == o.middle  &&
	   maximum == o.maximum;
  }

  float minimum;
  float middle;
  float maximum;
};

static inline Triple _reverse_negate(const Triple &v)
{ return {-v.maximum, -v.middle, -v.minimum}; }


static inline float supportScalar (float coord, const Triple &tent)
{
  /* Copied from VarRegionAxis::evaluate() */
  float start = tent.minimum, peak = tent.middle, end = tent.maximum;

  if (unlikely (start > peak || peak > end))
    return 1.;
  if (unlikely (start < 0 && end > 0 && peak != 0))
    return 1.;

  if (peak == 0 || coord == peak)
    return 1.;

  if (coord <= start || end <= coord)
    return 0.;

  /* Interpolate */
  if (coord < peak)
    return (coord - start) / (peak - start);
  else
    return  (end - coord) / (end - peak);
}


using result_item_t = hb_pair_t<float, Triple>;
using result_t = hb_vector_t<result_item_t>;

static inline result_t
_solve (Triple tent, Triple axisLimit, bool negative = false)
{
  float axisMin = axisLimit.minimum;
  float axisDef = axisLimit.middle;
  float axisMax = axisLimit.maximum;
  float lower = tent.minimum;
  float peak  = tent.middle;
  float upper = tent.maximum;

  // Mirror the problem such that axisDef <= peak
  if (axisDef > peak)
  {
    result_t vec = _solve (_reverse_negate (tent),
			   _reverse_negate (axisLimit),
			   !negative);

    for (auto &p : vec)
      p = hb_pair (p.first, _reverse_negate (p.second));

    return vec;
  }
  // axisDef <= peak

  /* case 1: The whole deltaset falls outside the new limit; we can drop it
   *
   *                                          peak
   *  1.........................................o..........
   *                                           / \
   *                                          /   \
   *                                         /     \
   *                                        /       \
   *  0---|-----------|----------|-------- o         o----1
   *    axisMin     axisDef    axisMax   lower     upper
   */
  if (axisMax <= lower && axisMax < peak)
      return result_t{};  // No overlap

  /* case 2: Only the peak and outermost bound fall outside the new limit;
   * we keep the deltaset, update peak and outermost bound and scale deltas
   * by the scalar value for the restricted axis at the new limit, and solve
   * recursively.
   *
   *                                  |peak
   *  1...............................|.o..........
   *                                  |/ \
   *                                  /   \
   *                                 /|    \
   *                                / |     \
   *  0--------------------------- o  |      o----1
   *                           lower  |      upper
   *                                  |
   *                                axisMax
   *
   * Convert to:
   *
   *  1............................................
   *                                  |
   *                                  o peak
   *                                 /|
   *                                /x|
   *  0--------------------------- o  o upper ----1
   *                           lower  |
   *                                  |
   *                                axisMax
   */
  if (axisMax < peak)
  {
    float mult = supportScalar (axisMax, tent);
    tent = Triple{lower, axisMax, axisMax};

    result_t vec = _solve (tent, axisLimit);

    for (auto &p : vec)
      p = hb_pair (p.first * mult, p.second);

    return vec;
  }

  // lower <= axisDef <= peak <= axisMax

  float gain = supportScalar (axisDef, tent);
  result_t out {hb_pair (gain, Triple{})};

  // First, the positive side

  // outGain is the scalar of axisMax at the tent.
  float outGain = supportScalar (axisMax, tent);

  /* Case 3a: Gain is more than outGain. The tent down-slope crosses
   * the axis into negative. We have to split it into multiples.
   *
   *                      | peak  |
   *  1...................|.o.....|..............
   *                      |/x\_   |
   *  gain................+....+_.|..............
   *                     /|    |y\|
   *  ................../.|....|..+_......outGain
   *                   /  |    |  | \
   *  0---|-----------o   |    |  |  o----------1
   *    axisMin    lower  |    |  |   upper
   *                      |    |  |
   *                axisDef    |  axisMax
   *                           |
   *                      crossing
   */
  if (gain > outGain)
  {
    // Crossing point on the axis.
    float crossing = peak + ((1 - gain) * (upper - peak) / (1 - outGain));

    Triple loc{peak, peak, crossing};
    float scalar = 1.f;

    // The part before the crossing point.
    out.push (hb_pair (scalar - gain, loc));

    /* The part after the crossing point may use one or two tents,
     * depending on whether upper is before axisMax or not, in one
     * case we need to keep it down to eternity.
     *
     * Case 3a1, similar to case 1neg; just one tent needed, as in
     * the drawing above.
     */
    if (upper >= axisMax)
    {
      Triple loc {crossing, axisMax, axisMax};
      float scalar = supportScalar (axisMax, tent);

      out.push (hb_pair (scalar - gain, loc));
    }

    /* Case 3a2: Similar to case 2neg; two tents needed, to keep
     * down to eternity.
     *
     *                      | peak             |
     *  1...................|.o................|...
     *                      |/ \_              |
     *  gain................+....+_............|...
     *                     /|    | \xxxxxxxxxxy|
     *                    / |    |  \_xxxxxyyyy|
     *                   /  |    |    \xxyyyyyy|
     *  0---|-----------o   |    |     o-------|--1
     *    axisMin    lower  |    |      upper  |
     *                      |    |             |
     *                axisDef    |             axisMax
     *                           |
     *                      crossing
     */
    else
    {
      // A tent's peak cannot fall on axis default. Nudge it.
      if (upper == axisDef)
	upper += EPSILON;

      // Downslope.
      Triple loc1 {crossing, upper, axisMax};
      float scalar1 = 0.f;

      // Eternity justify.
      Triple loc2 {upper, axisMax, axisMax};
      float scalar2 = 1.f; // supportScalar({"tag": axisMax}, {"tag": tent})

      out.push (hb_pair (scalar1 - gain, loc1));
      out.push (hb_pair (scalar2 - gain, loc2));
    }
  }

  /* Case 3: Outermost limit still fits within F2Dot14 bounds;
   * we keep deltas as is and only scale the axes bounds. Deltas beyond -1.0
   * or +1.0 will never be applied as implementations must clamp to that range.
   *
   * A second tent is needed for cases when gain is positive, though we add it
   * unconditionally and it will be dropped because scalar ends up 0.
   *
   * TODO: See if we can just move upper closer to adjust the slope, instead of
   * second tent.
   *
   *            |           peak |
   *  1.........|............o...|..................
   *            |           /x\  |
   *            |          /xxx\ |
   *            |         /xxxxx\|
   *            |        /xxxxxxx+
   *            |       /xxxxxxxx|\
   *  0---|-----|------oxxxxxxxxx|xo---------------1
   *    axisMin |  lower         | upper
   *            |                |
   *          axisDef          axisMax
   */
  else if (axisDef + (axisMax - axisDef) * 2 >= upper)
  {
    if (!negative && axisDef + (axisMax - axisDef) * MAX_F2DOT14 < upper)
    {
      // we clamp +2.0 to the max F2Dot14 (~1.99994) for convenience
      upper = axisDef + (axisMax - axisDef) * MAX_F2DOT14;
      assert (peak < upper);
    }

    // Special-case if peak is at axisMax.
    if (axisMax == peak)
	upper = peak;

    Triple loc1 {hb_max (axisDef, lower), peak, upper};
    float scalar1 = 1.f;

    Triple loc2 {peak, upper, upper};
    float scalar2 = 0.f;

    // Don't add a dirac delta!
    if (axisDef < upper)
	out.push (hb_pair (scalar1 - gain, loc1));
    if (peak < upper)
	out.push (hb_pair (scalar2 - gain, loc2));
  }

  /* Case 4: New limit doesn't fit; we need to chop into two tents,
   * because the shape of a triangle with part of one side cut off
   * cannot be represented as a triangle itself.
   *
   *            |   peak |
   *  1.........|......o.|...................
   *            |     /x\|
   *            |    |xxy|\_
   *            |   /xxxy|  \_
   *            |  |xxxxy|    \_
   *            |  /xxxxy|      \_
   *  0---|-----|-oxxxxxx|        o----------1
   *    axisMin | lower  |        upper
   *            |        |
   *          axisDef  axisMax
   */
  else
  {
    Triple loc1 {hb_max (axisDef, lower), peak, axisMax};
    float scalar1 = 1.f;

    Triple loc2 {peak, axisMax, axisMax};
    float scalar2 = supportScalar (axisMax, tent);

    out.push (hb_pair (scalar1 - gain, loc1));
    // Don't add a dirac delta!
    if (peak < axisMax)
      out.push (hb_pair (scalar2 - gain, loc2));
  }

  /* Now, the negative side
   *
   * Case 1neg: Lower extends beyond axisMin: we chop. Simple.
   *
   *                     |   |peak
   *  1..................|...|.o.................
   *                     |   |/ \
   *  gain...............|...+...\...............
   *                     |x_/|    \
   *                     |/  |     \
   *                   _/|   |      \
   *  0---------------o  |   |       o----------1
   *              lower  |   |       upper
   *                     |   |
   *               axisMin   axisDef
   */
  if (lower <= axisMin)
  {
    Triple loc {axisMin, axisMin, axisDef};
    float scalar = supportScalar (axisMin, tent);

    out.push (hb_pair (scalar - gain, loc));
  }

  /* Case 2neg: Lower is betwen axisMin and axisDef: we add two
   * tents to keep it down all the way to eternity.
   *
   *      |               |peak
   *  1...|...............|.o.................
   *      |               |/ \
   *  gain|...............+...\...............
   *      |yxxxxxxxxxxxxx/|    \
   *      |yyyyyyxxxxxxx/ |     \
   *      |yyyyyyyyyyyx/  |      \
   *  0---|-----------o   |       o----------1
   *    axisMin    lower  |       upper
   *                      |
   *                    axisDef
   */
  else
  {
    // A tent's peak cannot fall on axis default. Nudge it.
    if (lower == axisDef)
      lower -= EPSILON;

    // Downslope.
    Triple loc1 {axisMin, lower, axisDef};
    float scalar1 = 0.f;

    // Eternity justify.
    Triple loc2 {axisMin, axisMin, lower};
    float scalar2 = 0.f;

    out.push (hb_pair (scalar1 - gain, loc1));
    out.push (hb_pair (scalar2 - gain, loc2));
  }

  return out;
}

/* Normalizes value based on a min/default/max triple. */
static inline float normalizeValue (float v, const Triple &triple, bool extrapolate = false)
{
  /*
  >>> normalizeValue(400, (100, 400, 900))
  0.0
  >>> normalizeValue(100, (100, 400, 900))
  -1.0
  >>> normalizeValue(650, (100, 400, 900))
  0.5
  */
  float lower = triple.minimum, def = triple.middle, upper = triple.maximum;
  assert (lower <= def && def <= upper);

  if (!extrapolate)
      v = hb_max (hb_min (v, upper), lower);

  if ((v == def) || (lower == upper))
    return 0.f;

  if ((v < def && lower != def) || (v > def && upper == def))
    return (v - def) / (def - lower);
  else
  {
    assert ((v > def && upper != def) ||
	    (v < def && lower == def));
    return (v - def) / (upper - def);
  }
}

/* Given a tuple (lower,peak,upper) "tent" and new axis limits
 * (axisMin,axisDefault,axisMax), solves how to represent the tent
 * under the new axis configuration.  All values are in normalized
 * -1,0,+1 coordinate system. Tent values can be outside this range.
 *
 * Return value: a list of tuples. Each tuple is of the form
 * (scalar,tent), where scalar is a multipler to multiply any
 * delta-sets by, and tent is a new tent for that output delta-set.
 * If tent value is Triple{}, that is a special deltaset that should
 * be always-enabled (called "gain").
 */
HB_INTERNAL result_t rebase_tent (Triple tent, Triple axisLimit);

result_t
rebase_tent (Triple tent, Triple axisLimit)
{
  assert (-1.f <= axisLimit.minimum && axisLimit.minimum <= axisLimit.middle && axisLimit.middle <= axisLimit.maximum && axisLimit.maximum <= +1.f);
  assert (-2.f <= tent.minimum && tent.minimum <= tent.middle && tent.middle <= tent.maximum && tent.maximum <= +2.f);
  assert (tent.middle != 0.f);

  result_t sols = _solve (tent, axisLimit);

  auto n = [&axisLimit] (float v) { return normalizeValue (v, axisLimit, true); };

  result_t out;
  for (auto &p : sols)
  {
    if (!p.first) continue;
    if (p.second == Triple{})
    {
      out.push (p);
      continue;
    }
    Triple t = p.second;
    out.push (hb_pair (p.first,
		       Triple{n (t.minimum), n (t.middle), n (t.maximum)}));
  }

  return sols;
}
