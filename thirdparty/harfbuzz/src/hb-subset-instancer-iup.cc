/*
 * Copyright © 2024  Google, Inc.
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

#include "hb-subset-instancer-iup.hh"

#include "hb-bit-page.hh"

using hb_iup_set_t = hb_bit_page_t;

/* This file is a straight port of the following:
 *
 * https://github.com/fonttools/fonttools/blob/main/Lib/fontTools/varLib/iup.py
 *
 * Where that file returns optimzied deltas vector, we return optimized
 * referenced point indices.
 */

constexpr static unsigned MAX_LOOKBACK = 8;

static void _iup_contour_bound_forced_set (const hb_array_t<const contour_point_t> contour_points,
                                           const hb_array_t<const int> x_deltas,
                                           const hb_array_t<const int> y_deltas,
                                           hb_iup_set_t& forced_set, /* OUT */
                                           double tolerance = 0.0)
{
  unsigned len = contour_points.length;
  unsigned next_i = 0;
  for (int i = len - 1; i >= 0; i--)
  {
    unsigned last_i = (len + i -1) % len;
    for (unsigned j = 0; j < 2; j++)
    {
      double cj, lcj, ncj;
      int dj, ldj, ndj;
      if (j == 0)
      {
        cj = static_cast<double> (contour_points.arrayZ[i].x);
        dj = x_deltas.arrayZ[i];
        lcj = static_cast<double> (contour_points.arrayZ[last_i].x);
        ldj = x_deltas.arrayZ[last_i];
        ncj = static_cast<double> (contour_points.arrayZ[next_i].x);
        ndj = x_deltas.arrayZ[next_i];
      }
      else
      {
        cj = static_cast<double> (contour_points.arrayZ[i].y);
        dj = y_deltas.arrayZ[i];
        lcj = static_cast<double> (contour_points.arrayZ[last_i].y);
        ldj = y_deltas.arrayZ[last_i];
        ncj = static_cast<double> (contour_points.arrayZ[next_i].y);
        ndj = y_deltas.arrayZ[next_i];
      }

      double c1, c2;
      int d1, d2;
      if (lcj <= ncj)
      {
        c1 = lcj;
        c2 = ncj;
        d1 = ldj;
        d2 = ndj;
      }
      else
      {
        c1 = ncj;
        c2 = lcj;
        d1 = ndj;
        d2 = ldj;
      }

      bool force = false;
      if (c1 == c2)
      {
        if (abs (d1 - d2) > tolerance && abs (dj) > tolerance)
          force = true;
      }
      else if (c1 <= cj && cj <= c2)
      {
        if (!(hb_min (d1, d2) - tolerance <= dj &&
              dj <= hb_max (d1, d2) + tolerance))
          force = true;
      }
      else
      {
        if (d1 != d2)
        {
          if (cj < c1)
          {
            if (abs (dj) > tolerance &&
                abs (dj - d1) > tolerance &&
                ((dj - tolerance < d1) != (d1 < d2)))
                force = true;
          }
          else
          {
            if (abs (dj) > tolerance &&
                abs (dj - d2) > tolerance &&
                ((d2 < dj + tolerance) != (d1 < d2)))
              force = true;
          }
        }
      }

      if (force)
      {
        forced_set.add (i);
        break;
      }
    }
    next_i = i;
  }
}

template <typename T,
          hb_enable_if (hb_is_trivially_copyable (T))>
static bool rotate_array (const hb_array_t<const T>& org_array,
                          int k,
                          hb_vector_t<T>& out)
{
  unsigned n = org_array.length;
  if (!n) return true;
  if (unlikely (!out.resize_dirty  (n)))
    return false;

  unsigned item_size = hb_static_size (T);
  if (k < 0)
    k = n - (-k) % n;
  else
    k %= n;

  hb_memcpy ((void *) out.arrayZ, (const void *) (org_array.arrayZ + n - k), k * item_size);
  hb_memcpy ((void *) (out.arrayZ + k), (const void *) org_array.arrayZ, (n - k) * item_size);
  return true;
}

static bool rotate_set (const hb_iup_set_t& org_set,
                        int k,
                        unsigned n,
                        hb_iup_set_t& out)
{
  if (!n) return false;
  k %= n;
  if (k < 0)
    k = n + k;

  if (k == 0)
  {
    out = org_set;
  }
  else
  {
    for (unsigned v : org_set)
      out.add ((v + k) % n);
  }
  return true;
}

/* Given two reference coordinates (start and end of contour_points array),
 * output interpolated deltas for points in between */
static bool _iup_segment (const hb_array_t<const contour_point_t> contour_points,
                          const hb_array_t<const int> x_deltas,
                          const hb_array_t<const int> y_deltas,
                          const contour_point_t& p1, const contour_point_t& p2,
                          int p1_dx, int p2_dx,
                          int p1_dy, int p2_dy,
			  double tolerance_sq,
                          hb_vector_t<double>& interp_x_deltas, /* OUT */
                          hb_vector_t<double>& interp_y_deltas /* OUT */)
{
  unsigned n = contour_points.length;
  if (unlikely (!interp_x_deltas.resize_dirty  (n) ||
                !interp_y_deltas.resize_dirty  (n)))
    return false;

  for (unsigned j = 0; j < 2; j++)
  {
    float contour_point_t::* xp;
    double x1, x2, d1, d2;
    const int *in;
    double *out;
    if (j == 0)
    {
      xp = &contour_point_t::x;
      x1 = static_cast<double> (p1.x);
      x2 = static_cast<double> (p2.x);
      d1 = p1_dx;
      d2 = p2_dx;
      in = x_deltas.arrayZ;
      out = interp_x_deltas.arrayZ;
    }
    else
    {
      xp = &contour_point_t::y;
      x1 = static_cast<double> (p1.y);
      x2 = static_cast<double> (p2.y);
      d1 = p1_dy;
      d2 = p2_dy;
      in = y_deltas.arrayZ;
      out = interp_y_deltas.arrayZ;
    }

    if (x1 == x2)
    {
      if (d1 == d2)
      {
        for (unsigned i = 0; i < n; i++)
          out[i] = d1;
      }
      else
      {
        for (unsigned i = 0; i < n; i++)
          out[i] = 0.0;
      }
      continue;
    }

    if (x1 > x2)
    {
      hb_swap (x1, x2);
      hb_swap (d1, d2);
    }

    double scale = (d2 - d1) / (x2 - x1);
    for (unsigned i = 0; i < n; i++)
    {
      double x = (double) (contour_points.arrayZ[i].*xp);
      double d;
      if (x <= x1)
        d = d1;
      else if (x >= x2)
        d = d2;
      else
        d = d1 + (x - x1) * scale;

      out[i] = d;
      double err = d - in[i];
      if (err * err > tolerance_sq)
	return false;
    }
  }
  return true;
}

static bool _can_iup_in_between (const hb_array_t<const contour_point_t> contour_points,
                                 const hb_array_t<const int> x_deltas,
                                 const hb_array_t<const int> y_deltas,
                                 const contour_point_t& p1, const contour_point_t& p2,
                                 int p1_dx, int p2_dx,
                                 int p1_dy, int p2_dy,
                                 double tolerance_sq,
				 hb_vector_t<double> &interp_x_deltas, /* scratch */
				 hb_vector_t<double> &interp_y_deltas /* scratch */)
{
  if (!_iup_segment (contour_points, x_deltas, y_deltas,
                     p1, p2, p1_dx, p2_dx, p1_dy, p2_dy,
		     tolerance_sq,
                     interp_x_deltas, interp_y_deltas))
    return false;

  unsigned num = contour_points.length;

  for (unsigned i = 0; i < num; i++)
  {
    double dx = static_cast<double> (x_deltas.arrayZ[i]) - interp_x_deltas.arrayZ[i];
    double dy = static_cast<double> (y_deltas.arrayZ[i]) - interp_y_deltas.arrayZ[i];

    if (dx * dx + dy * dy > tolerance_sq)
      return false;
  }
  return true;
}

static bool _iup_contour_optimize_dp (const contour_point_vector_t& contour_points,
                                      const hb_vector_t<int>& x_deltas,
                                      const hb_vector_t<int>& y_deltas,
                                      const hb_iup_set_t& forced_set,
                                      double tolerance_sq,
                                      unsigned lookback,
                                      hb_vector_t<unsigned>& costs, /* OUT */
                                      hb_vector_t<int>& chain, /* OUT */
				      hb_vector_t<double> &interp_x_deltas_scratch,
				      hb_vector_t<double> &interp_y_deltas_scratch)
{
  unsigned n = contour_points.length;
  if (unlikely (!costs.resize_dirty  (n) ||
                !chain.resize_dirty  (n)))
    return false;

  lookback = hb_min (lookback, MAX_LOOKBACK);

  for (unsigned i = 0; i < n; i++)
  {
    unsigned best_cost = (i == 0 ? 1 : costs.arrayZ[i-1] + 1);

    costs.arrayZ[i] = best_cost;
    chain.arrayZ[i] = (i == 0 ? -1 : i - 1);

    if (i > 0 && forced_set.has (i - 1))
      continue;

    int lookback_index = hb_max ((int) i - (int) lookback + 1, -1);
    for (int j = i - 2; j >= lookback_index; j--)
    {
      unsigned cost = j == -1 ? 1 : costs.arrayZ[j] + 1;
      /* num points between i and j */
      unsigned num_points = i - j - 1;
      unsigned p1 = (j == -1 ? n - 1 : j);
      if (cost < best_cost &&
          _can_iup_in_between (contour_points.as_array ().sub_array (j + 1, num_points),
                               x_deltas.as_array ().sub_array (j + 1, num_points),
                               y_deltas.as_array ().sub_array (j + 1, num_points),
                               contour_points.arrayZ[p1], contour_points.arrayZ[i],
                               x_deltas.arrayZ[p1], x_deltas.arrayZ[i],
                               y_deltas.arrayZ[p1], y_deltas.arrayZ[i],
                               tolerance_sq,
			       interp_x_deltas_scratch, interp_y_deltas_scratch))
      {
        best_cost = cost;
        costs.arrayZ[i] = best_cost;
        chain.arrayZ[i] = j;
      }

      if (j > 0 && forced_set.has (j))
        break;
    }
  }
  return true;
}

static bool _iup_contour_optimize (const hb_array_t<const contour_point_t> contour_points,
                                   const hb_array_t<const int> x_deltas,
                                   const hb_array_t<const int> y_deltas,
                                   hb_array_t<bool> opt_indices, /* OUT */
                                   double tolerance,
				   iup_scratch_t &scratch)
{
  unsigned n = contour_points.length;
  if (opt_indices.length != n ||
      x_deltas.length != n ||
      y_deltas.length != n)
    return false;

  if (unlikely (n > hb_iup_set_t::PAGE_BITS))
    return true; // Refuse to work

  bool all_within_tolerance = true;
  double tolerance_sq = tolerance * tolerance;
  for (unsigned i = 0; i < n; i++)
  {
    int dx = x_deltas.arrayZ[i];
    int dy = y_deltas.arrayZ[i];
    if ((double) dx * dx + (double) dy * dy > tolerance_sq)
    {
      all_within_tolerance = false;
      break;
    }
  }

  /* If all are within tolerance distance, do nothing, opt_indices is
   * initilized to false */
  if (all_within_tolerance)
    return true;

  /* If there's exactly one point, return it */
  if (n == 1)
  {
    opt_indices.arrayZ[0] = true;
    return true;
  }

  /* If all deltas are exactly the same, return just one (the first one) */
  bool all_deltas_are_equal = true;
  for (unsigned i = 1; i < n; i++)
    if (x_deltas.arrayZ[i] != x_deltas.arrayZ[0] ||
        y_deltas.arrayZ[i] != y_deltas.arrayZ[0])
    {
      all_deltas_are_equal = false;
      break;
    }

  if (all_deltas_are_equal)
  {
    opt_indices.arrayZ[0] = true;
    return true;
  }

  /* else, solve the general problem using Dynamic Programming */
  hb_iup_set_t forced_set;
  _iup_contour_bound_forced_set (contour_points, x_deltas, y_deltas, forced_set, tolerance);

  hb_vector_t<unsigned> &costs = scratch.costs.reset ();
  hb_vector_t<int> &chain = scratch.chain.reset ();

  if (!forced_set.is_empty ())
  {
    int k = n - 1 - forced_set.get_max ();
    if (k < 0)
      return false;

    hb_vector_t<int> &rot_x_deltas = scratch.rot_x_deltas.reset ();
    hb_vector_t<int> &rot_y_deltas = scratch.rot_y_deltas.reset ();
    contour_point_vector_t &rot_points = scratch.rot_points;
    rot_points.reset ();
    hb_iup_set_t rot_forced_set;
    if (!rotate_array (contour_points, k, rot_points) ||
        !rotate_array (x_deltas, k, rot_x_deltas) ||
        !rotate_array (y_deltas, k, rot_y_deltas) ||
        !rotate_set (forced_set, k, n, rot_forced_set))
      return false;

    if (!_iup_contour_optimize_dp (rot_points, rot_x_deltas, rot_y_deltas,
                                   rot_forced_set, tolerance_sq, n,
                                   costs, chain,
				   scratch.interp_x_deltas, scratch.interp_y_deltas))
      return false;

    hb_iup_set_t solution;
    int index = n - 1;
    while (index != -1)
    {
      solution.add (index);
      index = chain.arrayZ[index];
    }

    if (solution.is_empty () ||
        forced_set.get_population () > solution.get_population ())
      return false;

    for (unsigned i : solution)
      opt_indices.arrayZ[i] = true;

    hb_vector_t<bool> &rot_indices = scratch.rot_indices.reset ();
    const hb_array_t<const bool> opt_indices_array (opt_indices.arrayZ, opt_indices.length);
    rotate_array (opt_indices_array, -k, rot_indices);

    for (unsigned i = 0; i < n; i++)
      opt_indices.arrayZ[i] = rot_indices.arrayZ[i];
  }
  else
  {
    hb_vector_t<int> repeat_x_deltas, repeat_y_deltas;
    contour_point_vector_t repeat_points;

    if (unlikely (!repeat_x_deltas.resize_dirty  (n * 2) ||
                  !repeat_y_deltas.resize_dirty  (n * 2) ||
                  !repeat_points.resize_dirty  (n * 2)))
      return false;

    unsigned contour_point_size = hb_static_size (contour_point_t);
    for (unsigned i = 0; i < n; i++)
    {
      hb_memcpy ((void *) repeat_x_deltas.arrayZ, (const void *) x_deltas.arrayZ, n * sizeof (repeat_x_deltas[0]));
      hb_memcpy ((void *) (repeat_x_deltas.arrayZ + n), (const void *) x_deltas.arrayZ, n * sizeof (repeat_x_deltas[0]));

      hb_memcpy ((void *) repeat_y_deltas.arrayZ, (const void *) y_deltas.arrayZ, n * sizeof (repeat_x_deltas[0]));
      hb_memcpy ((void *) (repeat_y_deltas.arrayZ + n), (const void *) y_deltas.arrayZ, n * sizeof (repeat_x_deltas[0]));

      hb_memcpy ((void *) repeat_points.arrayZ, (const void *) contour_points.arrayZ, n * contour_point_size);
      hb_memcpy ((void *) (repeat_points.arrayZ + n), (const void *) contour_points.arrayZ, n * contour_point_size);
    }

    if (!_iup_contour_optimize_dp (repeat_points, repeat_x_deltas, repeat_y_deltas,
                                   forced_set, tolerance_sq, n,
                                   costs, chain,
				   scratch.interp_x_deltas, scratch.interp_y_deltas))
      return false;

    unsigned best_cost = n + 1;
    int len = costs.length;
    hb_iup_set_t best_sol;
    for (int start = n - 1; start < len; start++)
    {
      hb_iup_set_t solution;
      int i = start;
      int lookback = start - (int) n;
      while (i > lookback)
      {
        solution.add (i % n);
        i = chain.arrayZ[i];
      }
      if (i == lookback)
      {
        unsigned cost_i = i < 0 ? 0 : costs.arrayZ[i];
        unsigned cost = costs.arrayZ[start] - cost_i;
        if (cost <= best_cost)
        {
          best_sol = solution;
          best_cost = cost;
        }
      }
    }

    for (unsigned i = 0; i < n; i++)
      if (best_sol.has (i))
        opt_indices.arrayZ[i] = true;
  }
  return true;
}

bool iup_delta_optimize (const contour_point_vector_t& contour_points,
                         const hb_vector_t<int>& x_deltas,
                         const hb_vector_t<int>& y_deltas,
                         hb_vector_t<bool>& opt_indices, /* OUT */
			 iup_scratch_t &scratch,
                         double tolerance)
{
  if (!opt_indices.resize (contour_points.length))
      return false;

  hb_vector_t<unsigned> &end_points = scratch.end_points.reset ();

  unsigned count = contour_points.length;
  if (unlikely (!end_points.alloc (count)))
    return false;

  for (unsigned i = 0; i < count - 4; i++)
    if (contour_points.arrayZ[i].is_end_point)
      end_points.push (i);

  /* phantom points */
  for (unsigned i = count - 4; i < count; i++)
    end_points.push (i);

  if (end_points.in_error ()) return false;

  unsigned start = 0;
  for (unsigned end : end_points)
  {
    unsigned len = end - start + 1;
    if (!_iup_contour_optimize (contour_points.as_array ().sub_array (start, len),
                                x_deltas.as_array ().sub_array (start, len),
                                y_deltas.as_array ().sub_array (start, len),
                                opt_indices.as_array ().sub_array (start, len),
                                tolerance,
				scratch))
      return false;
    start = end + 1;
  }
  return true;
}
