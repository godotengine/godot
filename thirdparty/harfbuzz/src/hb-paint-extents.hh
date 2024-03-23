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

#ifndef HB_PAINT_EXTENTS_HH
#define HB_PAINT_EXTENTS_HH

#include "hb.hh"
#include "hb-paint.h"


typedef struct hb_extents_t
{
  hb_extents_t () {}
  hb_extents_t (float xmin, float ymin, float xmax, float ymax) :
    xmin (xmin), ymin (ymin), xmax (xmax), ymax (ymax) {}

  bool is_empty () const { return xmin >= xmax || ymin >= ymax; }
  bool is_void () const { return xmin > xmax; }

  void union_ (const hb_extents_t &o)
  {
    xmin = hb_min (xmin, o.xmin);
    ymin = hb_min (ymin, o.ymin);
    xmax = hb_max (xmax, o.xmax);
    ymax = hb_max (ymax, o.ymax);
  }

  void intersect (const hb_extents_t &o)
  {
    xmin = hb_max (xmin, o.xmin);
    ymin = hb_max (ymin, o.ymin);
    xmax = hb_min (xmax, o.xmax);
    ymax = hb_min (ymax, o.ymax);
  }

  void
  add_point (float x, float y)
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

  float xmin = 0.f;
  float ymin = 0.f;
  float xmax = -1.f;
  float ymax = -1.f;
} hb_extents_t;

typedef struct hb_transform_t
{
  hb_transform_t () {}
  hb_transform_t (float xx, float yx,
		  float xy, float yy,
		  float x0, float y0) :
    xx (xx), yx (yx), xy (xy), yy (yy), x0 (x0), y0 (y0) {}

  void multiply (const hb_transform_t &o)
  {
    /* Copied from cairo, with "o" being "a" there and "this" being "b" there. */
    hb_transform_t r;

    r.xx = o.xx * xx + o.yx * xy;
    r.yx = o.xx * yx + o.yx * yy;

    r.xy = o.xy * xx + o.yy * xy;
    r.yy = o.xy * yx + o.yy * yy;

    r.x0 = o.x0 * xx + o.y0 * xy + x0;
    r.y0 = o.x0 * yx + o.y0 * yy + y0;

    *this = r;
  }

  void transform_distance (float &dx, float &dy) const
  {
    float new_x = xx * dx + xy * dy;
    float new_y = yx * dx + yy * dy;
    dx = new_x;
    dy = new_y;
  }

  void transform_point (float &x, float &y) const
  {
    transform_distance (x, y);
    x += x0;
    y += y0;
  }

  void transform_extents (hb_extents_t &extents) const
  {
    float quad_x[4], quad_y[4];

    quad_x[0] = extents.xmin;
    quad_y[0] = extents.ymin;
    quad_x[1] = extents.xmin;
    quad_y[1] = extents.ymax;
    quad_x[2] = extents.xmax;
    quad_y[2] = extents.ymin;
    quad_x[3] = extents.xmax;
    quad_y[3] = extents.ymax;

    extents = hb_extents_t {};
    for (unsigned i = 0; i < 4; i++)
    {
      transform_point (quad_x[i], quad_y[i]);
      extents.add_point (quad_x[i], quad_y[i]);
    }
  }

  float xx = 1.f;
  float yx = 0.f;
  float xy = 0.f;
  float yy = 1.f;
  float x0 = 0.f;
  float y0 = 0.f;
} hb_transform_t;

typedef struct hb_bounds_t
{
  enum status_t {
    UNBOUNDED,
    BOUNDED,
    EMPTY,
  };

  hb_bounds_t (status_t status) : status (status) {}
  hb_bounds_t (const hb_extents_t &extents) :
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
  hb_extents_t extents;
} hb_bounds_t;

typedef struct  hb_paint_extents_context_t hb_paint_extents_context_t;

struct hb_paint_extents_context_t
{
  hb_paint_extents_context_t ()
  {
    transforms.push (hb_transform_t{});
    clips.push (hb_bounds_t{hb_bounds_t::UNBOUNDED});
    groups.push (hb_bounds_t{hb_bounds_t::EMPTY});
  }

  hb_extents_t get_extents ()
  {
    return groups.tail().extents;
  }

  bool is_bounded ()
  {
    return groups.tail().status != hb_bounds_t::UNBOUNDED;
  }

  void push_transform (const hb_transform_t &trans)
  {
    hb_transform_t t = transforms.tail ();
    t.multiply (trans);
    transforms.push (t);
  }

  void pop_transform ()
  {
    transforms.pop ();
  }

  void push_clip (hb_extents_t extents)
  {
    /* Transform extents and push a new clip. */
    const hb_transform_t &t = transforms.tail ();
    t.transform_extents (extents);

    clips.push (hb_bounds_t {extents});
  }

  void pop_clip ()
  {
    clips.pop ();
  }

  void push_group ()
  {
    groups.push (hb_bounds_t {hb_bounds_t::EMPTY});
  }

  void pop_group (hb_paint_composite_mode_t mode)
  {
    const hb_bounds_t src_bounds = groups.pop ();
    hb_bounds_t &backdrop_bounds = groups.tail ();

    // https://learn.microsoft.com/en-us/typography/opentype/spec/colr#format-32-paintcomposite
    switch ((int) mode)
    {
      case HB_PAINT_COMPOSITE_MODE_CLEAR:
	backdrop_bounds.status = hb_bounds_t::EMPTY;
	break;
      case HB_PAINT_COMPOSITE_MODE_SRC:
      case HB_PAINT_COMPOSITE_MODE_SRC_OUT:
	backdrop_bounds = src_bounds;
	break;
      case HB_PAINT_COMPOSITE_MODE_DEST:
      case HB_PAINT_COMPOSITE_MODE_DEST_OUT:
	break;
      case HB_PAINT_COMPOSITE_MODE_SRC_IN:
      case HB_PAINT_COMPOSITE_MODE_DEST_IN:
	backdrop_bounds.intersect (src_bounds);
	break;
      default:
	backdrop_bounds.union_ (src_bounds);
	break;
     }
  }

  void paint ()
  {
    const hb_bounds_t &clip = clips.tail ();
    hb_bounds_t &group = groups.tail ();

    group.union_ (clip);
  }

  protected:
  hb_vector_t<hb_transform_t> transforms;
  hb_vector_t<hb_bounds_t> clips;
  hb_vector_t<hb_bounds_t> groups;
};

HB_INTERNAL hb_paint_funcs_t *
hb_paint_extents_get_funcs ();


#endif /* HB_PAINT_EXTENTS_HH */
