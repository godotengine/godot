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
    GID_IS_24BIT		= 0x1000,
    AXES_HAVE_VARIATION		= 0x2000,
    RESET_UNSPECIFIED_AXES	= 0x4000,
  };

  public:

  unsigned int get_size () const
  {
    unsigned fl = flags;
    unsigned int size = min_size;

    unsigned axis_width = (fl & AXIS_INDICES_ARE_SHORT) ? 4 : 3;
    size += numAxes * axis_width;

    if (fl & GID_IS_24BIT)	size += 1;

    // 2 bytes each for the following flags
    fl = fl & (HAVE_TRANSLATE_X | HAVE_TRANSLATE_Y |
	       HAVE_ROTATION |
	       HAVE_SCALE_X | HAVE_SCALE_Y |
	       HAVE_SKEW_X | HAVE_SKEW_Y |
	       HAVE_TCENTER_X | HAVE_TCENTER_Y);
    size += hb_popcount (fl) * 2;

    return size;
  }

  bool has_more () const { return true; }

  bool is_use_my_metrics () const { return flags & USE_MY_METRICS; }
  bool is_reset_unspecified_axes () const { return flags & RESET_UNSPECIFIED_AXES; }

  hb_codepoint_t get_gid () const
  {
    if (flags & GID_IS_24BIT)
      return * (const HBGlyphID24 *) &pad;
    else
      return * (const HBGlyphID16 *) &pad;
  }

  void set_gid (hb_codepoint_t gid)
  {
    if (flags & GID_IS_24BIT)
      * (HBGlyphID24 *) &pad = gid;
    else
      * (HBGlyphID16 *) &pad = gid;
  }

  unsigned get_numAxes () const
  {
    return numAxes;
  }

  unsigned get_num_points () const
  {
    unsigned fl = flags;
    unsigned num = 0;
    if (fl & AXES_HAVE_VARIATION)			num += numAxes;

    /* Hopefully faster code, relying on the value of the flags. */
    fl = (((fl & (HAVE_TRANSLATE_Y | HAVE_SCALE_Y | HAVE_SKEW_Y | HAVE_TCENTER_Y)) >> 1) | fl) &
         (HAVE_TRANSLATE_X | HAVE_ROTATION | HAVE_SCALE_X | HAVE_SKEW_X | HAVE_TCENTER_X);
    num += hb_popcount (fl);
    return num;

    /* Slower but more readable code. */
    if (fl & (HAVE_TRANSLATE_X | HAVE_TRANSLATE_Y))	num++;
    if (fl & HAVE_ROTATION)				num++;
    if (fl & (HAVE_SCALE_X | HAVE_SCALE_Y))		num++;
    if (fl & (HAVE_SKEW_X | HAVE_SKEW_Y))		num++;
    if (fl & (HAVE_TCENTER_X | HAVE_TCENTER_Y))		num++;
    return num;
  }

  void transform_points (hb_array_t<const contour_point_t> record_points,
			 hb_array_t<contour_point_t> points) const
  {
    float matrix[4];
    contour_point_t trans;

    get_transformation_from_points (record_points.arrayZ, matrix, trans);

    auto arrayZ = points.arrayZ;
    unsigned count = points.length;

    if (matrix[0] != 1.f || matrix[1] != 0.f ||
	matrix[2] != 0.f || matrix[3] != 1.f)
      for (unsigned i = 0; i < count; i++)
        arrayZ[i].transform (matrix);

    if (trans.x != 0.f || trans.y != 0.f)
      for (unsigned i = 0; i < count; i++)
        arrayZ[i].translate (trans);
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
    if (!translateX && !translateY)
      return;

    trans.x += matrix[0] * translateX + matrix[2] * translateY;
    trans.y += matrix[1] * translateX + matrix[3] * translateY;
  }

  static void scale (float (&matrix)[4], contour_point_t &trans,
		     float scaleX, float scaleY)
  {
    if (scaleX == 1.f && scaleY == 1.f)
      return;

    matrix[0] *= scaleX;
    matrix[1] *= scaleX;
    matrix[2] *= scaleY;
    matrix[3] *= scaleY;
  }

  static void rotate (float (&matrix)[4], contour_point_t &trans,
		      float rotation)
  {
    if (!rotation)
      return;

    // https://github.com/fonttools/fonttools/blob/f66ee05f71c8b57b5f519ee975e95edcd1466e14/Lib/fontTools/misc/transform.py#L240
    rotation = rotation * HB_PI;
    float c;
    float s;
#ifdef HAVE_SINCOSF
    sincosf (rotation, &s, &c);
#else
    c = cosf (rotation);
    s = sinf (rotation);
#endif
    float other[6] = {c, s, -s, c, 0.f, 0.f};
    transform (matrix, trans, other);
  }

  static void skew (float (&matrix)[4], contour_point_t &trans,
		    float skewX, float skewY)
  {
    if (!skewX && !skewY)
      return;

    // https://github.com/fonttools/fonttools/blob/f66ee05f71c8b57b5f519ee975e95edcd1466e14/Lib/fontTools/misc/transform.py#L255
    skewX = skewX * HB_PI;
    skewY = skewY * HB_PI;
    float other[6] = {1.f,
		      skewY ? tanf (skewY) : 0.f,
		      skewX ? tanf (skewX) : 0.f,
		      1.f,
		      0.f, 0.f};
    transform (matrix, trans, other);
  }

  bool get_points (contour_point_vector_t &points) const
  {
    unsigned num_points = get_num_points ();

    points.alloc (points.length + num_points + 4); // For phantom points
    if (unlikely (!points.resize (points.length + num_points, false))) return false;
    contour_point_t *rec_points = points.arrayZ + (points.length - num_points);
    hb_memset (rec_points, 0, num_points * sizeof (rec_points[0]));

    unsigned fl = flags;

    unsigned num_axes = numAxes;
    unsigned axis_width = (fl & AXIS_INDICES_ARE_SHORT) ? 2 : 1;
    unsigned axes_size = num_axes * axis_width;

    const F2DOT14 *q = (const F2DOT14 *) (axes_size +
					  (fl & GID_IS_24BIT ? 3 : 2) +
					  (const HBUINT8 *) &pad);

    unsigned count = num_axes;
    if (fl & AXES_HAVE_VARIATION)
    {
      for (unsigned i = 0; i < count; i++)
	rec_points++->x = q++->to_int ();
    }
    else
      q += count;

    const HBUINT16 *p = (const HBUINT16 *) q;

    if (fl & (HAVE_TRANSLATE_X | HAVE_TRANSLATE_Y))
    {
      int translateX = (fl & HAVE_TRANSLATE_X) ? * (const FWORD *) p++ : 0;
      int translateY = (fl & HAVE_TRANSLATE_Y) ? * (const FWORD *) p++ : 0;
      rec_points->x = translateX;
      rec_points->y = translateY;
      rec_points++;
    }
    if (fl & HAVE_ROTATION)
    {
      int rotation = (fl & HAVE_ROTATION) ? ((const F4DOT12 *) p++)->to_int () : 0;
      rec_points->x = rotation;
      rec_points++;
    }
    if (fl & (HAVE_SCALE_X | HAVE_SCALE_Y))
    {
      int scaleX = (fl & HAVE_SCALE_X) ? ((const F6DOT10 *) p++)->to_int () : 1 << 10;
      int scaleY = (fl & HAVE_SCALE_Y) ? ((const F6DOT10 *) p++)->to_int () : 1 << 10;
      if ((fl & UNIFORM_SCALE) && !(fl & HAVE_SCALE_Y))
	scaleY = scaleX;
      rec_points->x = scaleX;
      rec_points->y = scaleY;
      rec_points++;
    }
    if (fl & (HAVE_SKEW_X | HAVE_SKEW_Y))
    {
      int skewX = (fl & HAVE_SKEW_X) ? ((const F4DOT12 *) p++)->to_int () : 0;
      int skewY = (fl & HAVE_SKEW_Y) ? ((const F4DOT12 *) p++)->to_int () : 0;
      rec_points->x = skewX;
      rec_points->y = skewY;
      rec_points++;
    }
    if (fl & (HAVE_TCENTER_X | HAVE_TCENTER_Y))
    {
      int tCenterX = (fl & HAVE_TCENTER_X) ? * (const FWORD *) p++ : 0;
      int tCenterY = (fl & HAVE_TCENTER_Y) ? * (const FWORD *) p++ : 0;
      rec_points->x = tCenterX;
      rec_points->y = tCenterY;
      rec_points++;
    }

    return true;
  }

  void get_transformation_from_points (const contour_point_t *rec_points,
				       float (&matrix)[4], contour_point_t &trans) const
  {
    unsigned fl = flags;

    if (fl & AXES_HAVE_VARIATION)
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

    if (fl & (HAVE_TRANSLATE_X | HAVE_TRANSLATE_Y))
    {
      translateX = rec_points->x;
      translateY = rec_points->y;
      rec_points++;
    }
    if (fl & HAVE_ROTATION)
    {
      rotation = rec_points->x / (1 << 12);
      rec_points++;
    }
    if (fl & (HAVE_SCALE_X | HAVE_SCALE_Y))
    {
      scaleX = rec_points->x / (1 << 10);
      scaleY = rec_points->y / (1 << 10);
      rec_points++;
    }
    if (fl & (HAVE_SKEW_X | HAVE_SKEW_Y))
    {
      skewX = rec_points->x / (1 << 12);
      skewY = rec_points->y / (1 << 12);
      rec_points++;
    }
    if (fl & (HAVE_TCENTER_X | HAVE_TCENTER_Y))
    {
      tCenterX = rec_points->x;
      tCenterY = rec_points->y;
      rec_points++;
    }

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
    unsigned num_axes = numAxes;

    const HBUINT8  *p = (const HBUINT8 *)  (((HBUINT8 *) &numAxes) + numAxes.static_size + (flags & GID_IS_24BIT ? 3 : 2));
    const HBUINT16 *q = (const HBUINT16 *) (((HBUINT8 *) &numAxes) + numAxes.static_size + (flags & GID_IS_24BIT ? 3 : 2));

    const F2DOT14 *a = (const F2DOT14 *) ((HBUINT8 *) (axis_width == 1 ? (p + num_axes) : (HBUINT8 *) (q + num_axes)));

    unsigned count = num_axes;
    for (unsigned i = 0; i < count; i++)
    {
      unsigned axis_index = axis_width == 1 ? (unsigned) *p++ : (unsigned) *q++;

      signed v = have_variations ? rec_points.arrayZ[i].x : a++->to_int ();

      v = hb_clamp (v, -(1<<14), (1<<14));
      setter[axis_index] = v;
    }
  }

  protected:
  HBUINT16	flags;
  HBUINT8	numAxes;
  HBUINT16	pad;
  public:
  DEFINE_SIZE_MIN (5);
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

  const hb_bytes_t trim_padding () const
  {
    unsigned length = GlyphHeader::static_size;
    for (auto &comp : iter ())
      length += comp.get_size ();
    return bytes.sub_array (0, length);
  }
};


} /* namespace glyf_impl */
} /* namespace OT */


#endif /* OT_GLYF_VARCOMPOSITEGLYPH_HH */
