/*
 * Copyright © 2022 Behdad Esfahbod
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
#ifndef HB_GEOMETRY_HH
#define HB_GEOMETRY_HH

#include "hb.hh"

#include "hb-algs.hh"


template <typename Float = float>
struct hb_extents_t
{
  hb_extents_t () {}
  hb_extents_t (const hb_glyph_extents_t &extents) :
		xmin (hb_min (extents.x_bearing, extents.x_bearing + extents.width)),
		ymin (hb_min (extents.y_bearing, extents.y_bearing + extents.height)),
		xmax (hb_max (extents.x_bearing, extents.x_bearing + extents.width)),
		ymax (hb_max (extents.y_bearing, extents.y_bearing + extents.height)) {}
  hb_extents_t (Float xmin, Float ymin, Float xmax, Float ymax) :
    xmin (xmin), ymin (ymin), xmax (xmax), ymax (ymax) {}

  bool is_empty () const { return xmin >= xmax || ymin >= ymax; }
  bool is_void () const { return xmin > xmax; }

  void union_ (const hb_extents_t &o)
  {
    if (o.is_empty ()) return;
    if (is_empty ())
    {
      *this = o;
      return;
    }
    xmin = hb_min (xmin, o.xmin);
    ymin = hb_min (ymin, o.ymin);
    xmax = hb_max (xmax, o.xmax);
    ymax = hb_max (ymax, o.ymax);
  }

  void intersect (const hb_extents_t &o)
  {
    if (o.is_empty () || is_empty ())
    {
      *this = hb_extents_t {};
      return;
    }
    xmin = hb_max (xmin, o.xmin);
    ymin = hb_max (ymin, o.ymin);
    xmax = hb_min (xmax, o.xmax);
    ymax = hb_min (ymax, o.ymax);
  }

  void
  add_point (Float x, Float y)
  {
    if (unlikely (is_void ()))
    {
      xmin = xmax = x;
      ymin = ymax = y;
    }
    else
    {
      xmin = hb_min (xmin, x);
      ymin = hb_min (ymin, y);
      xmax = hb_max (xmax, x);
      ymax = hb_max (ymax, y);
    }
  }

  hb_glyph_extents_t to_glyph_extents (bool xneg = false, bool yneg = false) const
  {
    hb_position_t x0 = (hb_position_t) roundf (xmin);
    hb_position_t y0 = (hb_position_t) roundf (ymin);
    hb_position_t x1 = (hb_position_t) roundf (xmax);
    hb_position_t y1 = (hb_position_t) roundf (ymax);
    return hb_glyph_extents_t {xneg ? x1 : x0,
			       yneg ? y0 : y1,
			       xneg ? x0 - x1 : x1 - x0,
			       yneg ? y1 - y0 : y0 - y1};
  }

  Float xmin = 0;
  Float ymin = 0;
  Float xmax = -1;
  Float ymax = -1;
};

template <typename Float = float>
struct hb_transform_t
{
  hb_transform_t () {}
  hb_transform_t (Float xx, Float yx,
		  Float xy, Float yy,
		  Float x0, Float y0) :
    xx (xx), yx (yx), xy (xy), yy (yy), x0 (x0), y0 (y0) {}

  bool is_identity () const
  {
    return xx == 1 && yx == 0 &&
	   xy == 0 && yy == 1 &&
	   x0 == 0 && y0 == 0;
  }
  bool is_translation () const
  {
    return xx == 1 && yx == 0 &&
	   xy == 0 && yy == 1;
  }

  void multiply (const hb_transform_t &o, bool before=false)
  {
    // Copied from cairo-matrix.c
    const hb_transform_t &a = before ? o : *this;
    const hb_transform_t &b = before ? *this : o;
    *this = {
      a.xx * b.xx + a.xy * b.yx,
      a.yx * b.xx + a.yy * b.yx,
      a.xx * b.xy + a.xy * b.yy,
      a.yx * b.xy + a.yy * b.yy,
      a.xx * b.x0 + a.xy * b.y0 + a.x0,
      a.yx * b.x0 + a.yy * b.y0 + a.y0
    };
  }

  HB_ALWAYS_INLINE
  void transform_distance (Float &dx, Float &dy) const
  {
    Float new_x = xx * dx + xy * dy;
    Float new_y = yx * dx + yy * dy;
    dx = new_x;
    dy = new_y;
  }

  HB_ALWAYS_INLINE
  void transform_point (Float &x, Float &y) const
  {
    Float new_x = x0 + xx * x + xy * y;
    Float new_y = y0 + yx * x + yy * y;
    x = new_x;
    y = new_y;
  }

  void transform_extents (hb_extents_t<Float> &extents) const
  {
    Float quad_x[4], quad_y[4];

    quad_x[0] = extents.xmin;
    quad_y[0] = extents.ymin;
    quad_x[1] = extents.xmin;
    quad_y[1] = extents.ymax;
    quad_x[2] = extents.xmax;
    quad_y[2] = extents.ymin;
    quad_x[3] = extents.xmax;
    quad_y[3] = extents.ymax;

    extents = hb_extents_t<Float> {};
    for (unsigned i = 0; i < 4; i++)
    {
      transform_point (quad_x[i], quad_y[i]);
      extents.add_point (quad_x[i], quad_y[i]);
    }
  }

  void transform (const hb_transform_t &o, bool before=false) { multiply (o, before); }

  static hb_transform_t translation (Float x, Float y)
  {
    return {1, 0, 0, 1, x, y};
  }
  void translate (Float x, Float y, bool before=false)
  {
    if (before)
    {
      x0 += x;
      y0 += y;
    }
    else
    {
      if (x == 0 && y == 0)
	return;

      x0 += xx * x + xy * y;
      y0 += yx * x + yy * y;
    }
  }

  static hb_transform_t scaling (Float scaleX, Float scaleY)
  {
    return {scaleX, 0, 0, scaleY, 0, 0};
  }
  void scale (Float scaleX, Float scaleY)
  {
    if (scaleX == 1 && scaleY == 1)
      return;

    xx *= scaleX;
    yx *= scaleX;
    xy *= scaleY;
    yy *= scaleY;
  }
  static hb_transform_t scaling_around_center (Float scaleX, Float scaleY, Float center_x, Float center_y)
  {
    return {scaleX, 0, 0, scaleY,
	    center_x ? (1 - scaleX) * center_x : 0,
	    center_y ? (1 - scaleY) * center_y : 0};
  }
  void scale_around_center (Float scaleX, Float scaleY, Float center_x, Float center_y)
  {
    if (scaleX == 1 && scaleY == 1)
      return;

    transform (scaling_around_center (scaleX, scaleY, center_x, center_y));
  }

  static hb_transform_t rotation (Float radians)
  {
    // https://github.com/fonttools/fonttools/blob/f66ee05f71c8b57b5f519ee975e95edcd1466e14/Lib/fontTools/misc/transform.py#L240
    Float c;
    Float s;
    hb_sincos (radians, s, c);
    return {c, s, -s, c, 0, 0};
  }
  void rotate (Float radians, bool before=false)
  {
    if (radians == 0)
      return;

    transform (rotation (radians), before);
  }

  static hb_transform_t rotation_around_center (Float radians, Float center_x, Float center_y)
  {
    Float s, c;
    hb_sincos (radians, s, c);
    return {
      c, s, -s, c,
      (1 - c) * center_x + s * center_y,
      -s * center_x +  (1 - c) * center_y
    };
  }
  void rotate_around_center (Float radians, Float center_x, Float center_y, bool before=false)
  {
    if (radians == 0)
      return;

    transform (rotation_around_center (radians, center_x, center_y), before);
  }

  static hb_transform_t skewing (Float skewX, Float skewY)
  {
    return {1, skewY ? tanf (skewY) : 0, skewX ? tanf (skewX) : 0, 1, 0, 0};
  }
  void skew (Float skewX, Float skewY)
  {
    if (skewX == 0 && skewY == 0)
      return;

    transform (skewing (skewX, skewY));
  }
  static hb_transform_t skewing_around_center (Float skewX, Float skewY, Float center_x, Float center_y)
  {
    skewX = skewX ? tanf (skewX) : 0;
    skewY = skewY ? tanf (skewY) : 0;
    return {
	    1, skewY, skewX, 1,
	    center_y ? -skewX * center_y : 0,
	    center_x ? -skewY * center_x : 0
    };
  }
  void skew_around_center (Float skewX, Float skewY, Float center_x, Float center_y)
  {
    if (skewX == 0 && skewY == 0)
	    return;

    transform (skewing_around_center (skewX, skewY, center_x, center_y));
  }

  Float xx = 1;
  Float yx = 0;
  Float xy = 0;
  Float yy = 1;
  Float x0 = 0;
  Float y0 = 0;
};

#define HB_TRANSFORM_IDENTITY {1, 0, 0, 1, 0, 0}

template <typename Float = float>
struct hb_bounds_t
{
  enum status_t {
    UNBOUNDED,
    BOUNDED,
    EMPTY,
  };

  hb_bounds_t (status_t status = UNBOUNDED) : status (status) {}
  hb_bounds_t (const hb_extents_t<Float> &extents) :
    status (extents.is_empty () ? EMPTY : BOUNDED), extents (extents) {}

  void union_ (const hb_bounds_t &o)
  {
    if (o.status == UNBOUNDED)
      status = UNBOUNDED;
    else if (o.status == BOUNDED)
    {
      if (status == EMPTY)
	*this = o;
      else if (status == BOUNDED)
        extents.union_ (o.extents);
    }
  }

  void intersect (const hb_bounds_t &o)
  {
    if (o.status == EMPTY)
      status = EMPTY;
    else if (o.status == BOUNDED)
    {
      if (status == UNBOUNDED)
	*this = o;
      else if (status == BOUNDED)
      {
        extents.intersect (o.extents);
	if (extents.is_empty ())
	  status = EMPTY;
      }
    }
  }

  status_t status;
  hb_extents_t<Float> extents;
};

template <typename Float = float>
struct hb_transform_decomposed_t
{
  Float translateX = 0;
  Float translateY = 0;
  Float rotation = 0;  // in radians, counter-clockwise
  Float scaleX = 1;
  Float scaleY = 1;
  Float skewX = 0;  // in radians, counter-clockwise
  Float skewY = 0;  // in radians, counter-clockwise
  Float tCenterX = 0;
  Float tCenterY = 0;

  operator bool () const
  {
    return translateX || translateY ||
	   rotation ||
	   scaleX != 1 || scaleY != 1 ||
	   skewX || skewY ||
	   tCenterX || tCenterY;
  }

  hb_transform_t<Float> to_transform () const
  {
    hb_transform_t<Float> t;
    t.translate (translateX + tCenterX, translateY + tCenterY);
    t.rotate (rotation);
    t.scale (scaleX, scaleY);
    t.skew (-skewX, skewY);
    t.translate (-tCenterX, -tCenterY);
    return t;
  }
};


#endif /* HB_GEOMETRY_HH */
