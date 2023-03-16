#ifndef OT_GLYF_VARCOMPOSITEGLYPH_HH
#define OT_GLYF_VARCOMPOSITEGLYPH_HH


#include "../../hb-open-type.hh"
#include "coord-setter.hh"


namespace OT {
namespace glyf_impl {


struct VarCompositeGlyphRecord
{
  protected:
  enum var_composite_glyph_flag_t
  {
    USE_MY_METRICS		= 0x0001,
    AXIS_INDICES_ARE_SHORT	= 0x0002,
    UNIFORM_SCALE		= 0x0004,
    HAVE_TRANSLATE_X		= 0x0008,
    HAVE_TRANSLATE_Y		= 0x0010,
    HAVE_ROTATION		= 0x0020,
    HAVE_SCALE_X		= 0x0040,
    HAVE_SCALE_Y		= 0x0080,
    HAVE_SKEW_X			= 0x0100,
    HAVE_SKEW_Y			= 0x0200,
    HAVE_TCENTER_X		= 0x0400,
    HAVE_TCENTER_Y		= 0x0800,
    GID_IS_24			= 0x1000,
    AXES_HAVE_VARIATION		= 0x2000,
  };

  public:

  unsigned int get_size () const
  {
    unsigned int size = min_size;

    unsigned axis_width = (flags & AXIS_INDICES_ARE_SHORT) ? 4 : 3;
    size += numAxes * axis_width;

    // gid
    size += 2;
    if (flags & GID_IS_24)		size += 1;

    if (flags & HAVE_TRANSLATE_X)	size += 2;
    if (flags & HAVE_TRANSLATE_Y)	size += 2;
    if (flags & HAVE_ROTATION)		size += 2;
    if (flags & HAVE_SCALE_X)		size += 2;
    if (flags & HAVE_SCALE_Y)		size += 2;
    if (flags & HAVE_SKEW_X)		size += 2;
    if (flags & HAVE_SKEW_Y)		size += 2;
    if (flags & HAVE_TCENTER_X)		size += 2;
    if (flags & HAVE_TCENTER_Y)		size += 2;

    return size;
  }

  bool has_more () const { return true; }

  bool is_use_my_metrics () const { return flags & USE_MY_METRICS; }

  hb_codepoint_t get_gid () const
  {
    if (flags & GID_IS_24)
      return StructAfter<const HBGlyphID24> (numAxes);
    else
      return StructAfter<const HBGlyphID16> (numAxes);
  }

  unsigned get_numAxes () const
  {
    return numAxes;
  }

  unsigned get_num_points () const
  {
    unsigned num = 0;
    if (flags & AXES_HAVE_VARIATION)			num += numAxes;
    if (flags & (HAVE_TRANSLATE_X | HAVE_TRANSLATE_Y))	num++;
    if (flags & HAVE_ROTATION)				num++;
    if (flags & (HAVE_SCALE_X | HAVE_SCALE_Y))		num++;
    if (flags & (HAVE_SKEW_X | HAVE_SKEW_Y))		num++;
    if (flags & (HAVE_TCENTER_X | HAVE_TCENTER_Y))	num++;
    return num;
  }

  void transform_points (hb_array_t<contour_point_t> record_points,
			 contour_point_vector_t &points) const
  {
    float matrix[4];
    contour_point_t trans;

    get_transformation_from_points (record_points, matrix, trans);

    points.transform (matrix);
    points.translate (trans);
  }

  static inline void transform (float (&matrix)[4], contour_point_t &trans,
				float (other)[6])
  {
    // https://github.com/fonttools/fonttools/blob/f66ee05f71c8b57b5f519ee975e95edcd1466e14/Lib/fontTools/misc/transform.py#L268
    float xx1 = other[0];
    float xy1 = other[1];
    float yx1 = other[2];
    float yy1 = other[3];
    float dx1 = other[4];
    float dy1 = other[5];
    float xx2 = matrix[0];
    float xy2 = matrix[1];
    float yx2 = matrix[2];
    float yy2 = matrix[3];
    float dx2 = trans.x;
    float dy2 = trans.y;

    matrix[0] = xx1*xx2 + xy1*yx2;
    matrix[1] = xx1*xy2 + xy1*yy2;
    matrix[2] = yx1*xx2 + yy1*yx2;
    matrix[3] = yx1*xy2 + yy1*yy2;
    trans.x = xx2*dx1 + yx2*dy1 + dx2;
    trans.y = xy2*dx1 + yy2*dy1 + dy2;
  }

  static void translate (float (&matrix)[4], contour_point_t &trans,
			 float translateX, float translateY)
  {
    // https://github.com/fonttools/fonttools/blob/f66ee05f71c8b57b5f519ee975e95edcd1466e14/Lib/fontTools/misc/transform.py#L213
    float other[6] = {1.f, 0.f, 0.f, 1.f, translateX, translateY};
    transform (matrix, trans, other);
  }

  static void scale (float (&matrix)[4], contour_point_t &trans,
		     float scaleX, float scaleY)
  {
    // https://github.com/fonttools/fonttools/blob/f66ee05f71c8b57b5f519ee975e95edcd1466e14/Lib/fontTools/misc/transform.py#L224
    float other[6] = {scaleX, 0.f, 0.f, scaleY, 0.f, 0.f};
    transform (matrix, trans, other);
  }

  static void rotate (float (&matrix)[4], contour_point_t &trans,
		      float rotation)
  {
    // https://github.com/fonttools/fonttools/blob/f66ee05f71c8b57b5f519ee975e95edcd1466e14/Lib/fontTools/misc/transform.py#L240
    rotation = rotation * float (M_PI);
    float c = cosf (rotation);
    float s = sinf (rotation);
    float other[6] = {c, s, -s, c, 0.f, 0.f};
    transform (matrix, trans, other);
  }

  static void skew (float (&matrix)[4], contour_point_t &trans,
		    float skewX, float skewY)
  {
    // https://github.com/fonttools/fonttools/blob/f66ee05f71c8b57b5f519ee975e95edcd1466e14/Lib/fontTools/misc/transform.py#L255
    skewX = skewX * float (M_PI);
    skewY = skewY * float (M_PI);
    float other[6] = {1.f, tanf (skewY), tanf (skewX), 1.f, 0.f, 0.f};
    transform (matrix, trans, other);
  }

  bool get_points (contour_point_vector_t &points) const
  {
    float translateX = 0.f;
    float translateY = 0.f;
    float rotation = 0.f;
    float scaleX = 1.f * (1 << 12);
    float scaleY = 1.f * (1 << 12);
    float skewX = 0.f;
    float skewY = 0.f;
    float tCenterX = 0.f;
    float tCenterY = 0.f;

    if (unlikely (!points.resize (points.length + get_num_points ()))) return false;

    unsigned axis_width = (flags & AXIS_INDICES_ARE_SHORT) ? 2 : 1;
    unsigned axes_size = numAxes * axis_width;

    const F2DOT14 *q = (const F2DOT14 *) (axes_size +
					  (flags & GID_IS_24 ? 3 : 2) +
					  &StructAfter<const HBUINT8> (numAxes));

    hb_array_t<contour_point_t> rec_points = points.as_array ().sub_array (points.length - get_num_points ());

    unsigned count = numAxes;
    if (flags & AXES_HAVE_VARIATION)
    {
      for (unsigned i = 0; i < count; i++)
	rec_points[i].x = *q++;
      rec_points += count;
    }
    else
      q += count;

    const HBUINT16 *p = (const HBUINT16 *) q;

    if (flags & HAVE_TRANSLATE_X)	translateX = * (const FWORD *) p++;
    if (flags & HAVE_TRANSLATE_Y)	translateY = * (const FWORD *) p++;
    if (flags & HAVE_ROTATION)		rotation = * (const F2DOT14 *) p++;
    if (flags & HAVE_SCALE_X)		scaleX = * (const F4DOT12 *) p++;
    if (flags & HAVE_SCALE_Y)		scaleY = * (const F4DOT12 *) p++;
    if (flags & HAVE_SKEW_X)		skewX = * (const F2DOT14 *) p++;
    if (flags & HAVE_SKEW_Y)		skewY = * (const F2DOT14 *) p++;
    if (flags & HAVE_TCENTER_X)		tCenterX = * (const FWORD *) p++;
    if (flags & HAVE_TCENTER_Y)		tCenterY = * (const FWORD *) p++;

    if ((flags & UNIFORM_SCALE) && !(flags & HAVE_SCALE_Y))
      scaleY = scaleX;

    if (flags & (HAVE_TRANSLATE_X | HAVE_TRANSLATE_Y))
    {
      rec_points[0].x = translateX;
      rec_points[0].y = translateY;
      rec_points++;
    }
    if (flags & HAVE_ROTATION)
    {
      rec_points[0].x = rotation;
      rec_points++;
    }
    if (flags & (HAVE_SCALE_X | HAVE_SCALE_Y))
    {
      rec_points[0].x = scaleX;
      rec_points[0].y = scaleY;
      rec_points++;
    }
    if (flags & (HAVE_SKEW_X | HAVE_SKEW_Y))
    {
      rec_points[0].x = skewX;
      rec_points[0].y = skewY;
      rec_points++;
    }
    if (flags & (HAVE_TCENTER_X | HAVE_TCENTER_Y))
    {
      rec_points[0].x = tCenterX;
      rec_points[0].y = tCenterY;
      rec_points++;
    }
    assert (!rec_points);

    return true;
  }

  void get_transformation_from_points (hb_array_t<contour_point_t> rec_points,
				       float (&matrix)[4], contour_point_t &trans) const
  {
    if (flags & AXES_HAVE_VARIATION)
      rec_points += numAxes;

    matrix[0] = matrix[3] = 1.f;
    matrix[1] = matrix[2] = 0.f;
    trans.init (0.f, 0.f);

    float translateX = 0.f;
    float translateY = 0.f;
    float rotation = 0.f;
    float scaleX = 1.f;
    float scaleY = 1.f;
    float skewX = 0.f;
    float skewY = 0.f;
    float tCenterX = 0.f;
    float tCenterY = 0.f;

    if (flags & (HAVE_TRANSLATE_X | HAVE_TRANSLATE_Y))
    {
      translateX = rec_points[0].x;
      translateY = rec_points[0].y;
      rec_points++;
    }
    if (flags & HAVE_ROTATION)
    {
      rotation = rec_points[0].x / (1 << 14);
      rec_points++;
    }
    if (flags & (HAVE_SCALE_X | HAVE_SCALE_Y))
    {
      scaleX = rec_points[0].x / (1 << 12);
      scaleY = rec_points[0].y / (1 << 12);
      rec_points++;
    }
    if (flags & (HAVE_SKEW_X | HAVE_SKEW_Y))
    {
      skewX = rec_points[0].x / (1 << 14);
      skewY = rec_points[0].y / (1 << 14);
      rec_points++;
    }
    if (flags & (HAVE_TCENTER_X | HAVE_TCENTER_Y))
    {
      tCenterX = rec_points[0].x;
      tCenterY = rec_points[0].y;
      rec_points++;
    }
    assert (!rec_points);

    translate (matrix, trans, translateX + tCenterX, translateY + tCenterY);
    rotate (matrix, trans, rotation);
    scale (matrix, trans, scaleX, scaleY);
    skew (matrix, trans, -skewX, skewY);
    translate (matrix, trans, -tCenterX, -tCenterY);
  }

  void set_variations (coord_setter_t &setter,
		       hb_array_t<contour_point_t> rec_points) const
  {
    bool have_variations = flags & AXES_HAVE_VARIATION;
    unsigned axis_width = (flags & AXIS_INDICES_ARE_SHORT) ? 2 : 1;

    const HBUINT8  *p = (const HBUINT8 *)  (((HBUINT8 *) &numAxes) + numAxes.static_size + (flags & GID_IS_24 ? 3 : 2));
    const HBUINT16 *q = (const HBUINT16 *) (((HBUINT8 *) &numAxes) + numAxes.static_size + (flags & GID_IS_24 ? 3 : 2));

    const F2DOT14 *a = (const F2DOT14 *) ((HBUINT8 *) (axis_width == 1 ? (p + numAxes) : (HBUINT8 *) (q + numAxes)));

    unsigned count = numAxes;
    for (unsigned i = 0; i < count; i++)
    {
      unsigned axis_index = axis_width == 1 ? (unsigned) *p++ : (unsigned) *q++;

      signed v = have_variations ? rec_points[i].x : *a++;

      v += setter[axis_index];
      v = hb_clamp (v, -(1<<14), (1<<14));
      setter[axis_index] = v;
    }
  }

  protected:
  HBUINT16	flags;
  HBUINT8	numAxes;
  public:
  DEFINE_SIZE_MIN (3);
};

using var_composite_iter_t = composite_iter_tmpl<VarCompositeGlyphRecord>;

struct VarCompositeGlyph
{
  const GlyphHeader &header;
  hb_bytes_t bytes;
  VarCompositeGlyph (const GlyphHeader &header_, hb_bytes_t bytes_) :
    header (header_), bytes (bytes_) {}

  var_composite_iter_t iter () const
  { return var_composite_iter_t (bytes, &StructAfter<VarCompositeGlyphRecord, GlyphHeader> (header)); }

};


} /* namespace glyf_impl */
} /* namespace OT */


#endif /* OT_GLYF_VARCOMPOSITEGLYPH_HH */
