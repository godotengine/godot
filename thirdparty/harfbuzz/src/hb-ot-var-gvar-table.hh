/*
 * Copyright © 2019  Adobe Inc.
 * Copyright © 2019  Ebrahim Byagowi
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

#ifndef HB_OT_VAR_GVAR_TABLE_HH
#define HB_OT_VAR_GVAR_TABLE_HH

#include "hb-open-type.hh"

/*
 * gvar -- Glyph Variation Table
 * https://docs.microsoft.com/en-us/typography/opentype/spec/gvar
 */
#define HB_OT_TAG_gvar HB_TAG('g','v','a','r')

namespace OT {

struct contour_point_t
{
  void init (float x_ = 0.f, float y_ = 0.f, bool is_end_point_ = false)
  { flag = 0; x = x_; y = y_; is_end_point = is_end_point_; }

  void translate (const contour_point_t &p) { x += p.x; y += p.y; }

  uint8_t flag;
  float x, y;
  bool is_end_point;
};

struct contour_point_vector_t : hb_vector_t<contour_point_t>
{
  void extend (const hb_array_t<contour_point_t> &a)
  {
    unsigned int old_len = length;
    resize (old_len + a.length);
    for (unsigned int i = 0; i < a.length; i++)
      (*this)[old_len + i] = a[i];
  }

  void transform (const float (&matrix)[4])
  {
    for (unsigned int i = 0; i < length; i++)
    {
      contour_point_t &p = (*this)[i];
      float x_ = p.x * matrix[0] + p.y * matrix[2];
	   p.y = p.x * matrix[1] + p.y * matrix[3];
      p.x = x_;
    }
  }

  void translate (const contour_point_t& delta)
  {
    for (unsigned int i = 0; i < length; i++)
      (*this)[i].translate (delta);
  }
};

/* https://docs.microsoft.com/en-us/typography/opentype/spec/otvarcommonformats#tuplevariationheader */
struct TupleVariationHeader
{
  unsigned get_size (unsigned axis_count) const
  { return min_size + get_all_tuples (axis_count).get_size (); }

  unsigned get_data_size () const { return varDataSize; }

  const TupleVariationHeader &get_next (unsigned axis_count) const
  { return StructAtOffset<TupleVariationHeader> (this, get_size (axis_count)); }

  float calculate_scalar (const int *coords, unsigned int coord_count,
			  const hb_array_t<const F2DOT14> shared_tuples) const
  {
    hb_array_t<const F2DOT14> peak_tuple;

    if (has_peak ())
      peak_tuple = get_peak_tuple (coord_count);
    else
    {
      unsigned int index = get_index ();
      if (unlikely (index * coord_count >= shared_tuples.length))
	return 0.f;
      peak_tuple = shared_tuples.sub_array (coord_count * index, coord_count);
    }

    hb_array_t<const F2DOT14> start_tuple;
    hb_array_t<const F2DOT14> end_tuple;
    if (has_intermediate ())
    {
      start_tuple = get_start_tuple (coord_count);
      end_tuple = get_end_tuple (coord_count);
    }

    float scalar = 1.f;
    for (unsigned int i = 0; i < coord_count; i++)
    {
      int v = coords[i];
      int peak = peak_tuple[i];
      if (!peak || v == peak) continue;

      if (has_intermediate ())
      {
	int start = start_tuple[i];
	int end = end_tuple[i];
	if (unlikely (start > peak || peak > end ||
		      (start < 0 && end > 0 && peak))) continue;
	if (v < start || v > end) return 0.f;
	if (v < peak)
	{ if (peak != start) scalar *= (float) (v - start) / (peak - start); }
	else
	{ if (peak != end) scalar *= (float) (end - v) / (end - peak); }
      }
      else if (!v || v < hb_min (0, peak) || v > hb_max (0, peak)) return 0.f;
      else
	scalar *= (float) v / peak;
    }
    return scalar;
  }

  bool           has_peak () const { return tupleIndex & TuppleIndex::EmbeddedPeakTuple; }
  bool   has_intermediate () const { return tupleIndex & TuppleIndex::IntermediateRegion; }
  bool has_private_points () const { return tupleIndex & TuppleIndex::PrivatePointNumbers; }
  unsigned      get_index () const { return tupleIndex & TuppleIndex::TupleIndexMask; }

  protected:
  struct TuppleIndex : HBUINT16
  {
    enum Flags {
      EmbeddedPeakTuple   = 0x8000u,
      IntermediateRegion  = 0x4000u,
      PrivatePointNumbers = 0x2000u,
      TupleIndexMask      = 0x0FFFu
    };

    DEFINE_SIZE_STATIC (2);
  };

  hb_array_t<const F2DOT14> get_all_tuples (unsigned axis_count) const
  { return StructAfter<UnsizedArrayOf<F2DOT14>> (tupleIndex).as_array ((has_peak () + has_intermediate () * 2) * axis_count); }
  hb_array_t<const F2DOT14> get_peak_tuple (unsigned axis_count) const
  { return get_all_tuples (axis_count).sub_array (0, axis_count); }
  hb_array_t<const F2DOT14> get_start_tuple (unsigned axis_count) const
  { return get_all_tuples (axis_count).sub_array (has_peak () * axis_count, axis_count); }
  hb_array_t<const F2DOT14> get_end_tuple (unsigned axis_count) const
  { return get_all_tuples (axis_count).sub_array (has_peak () * axis_count + axis_count, axis_count); }

  HBUINT16	varDataSize;	/* The size in bytes of the serialized
				 * data for this tuple variation table. */
  TuppleIndex	tupleIndex;	/* A packed field. The high 4 bits are flags (see below).
				   The low 12 bits are an index into a shared tuple
				   records array. */
  /* UnsizedArrayOf<F2DOT14> peakTuple - optional */
				/* Peak tuple record for this tuple variation table — optional,
				 * determined by flags in the tupleIndex value.
				 *
				 * Note that this must always be included in the 'cvar' table. */
  /* UnsizedArrayOf<F2DOT14> intermediateStartTuple - optional */
				/* Intermediate start tuple record for this tuple variation table — optional,
				   determined by flags in the tupleIndex value. */
  /* UnsizedArrayOf<F2DOT14> intermediateEndTuple - optional */
				/* Intermediate end tuple record for this tuple variation table — optional,
				 * determined by flags in the tupleIndex value. */
  public:
  DEFINE_SIZE_MIN (4);
};

struct GlyphVariationData
{
  const TupleVariationHeader &get_tuple_var_header (void) const
  { return StructAfter<TupleVariationHeader> (data); }

  struct tuple_iterator_t
  {
    void init (hb_bytes_t var_data_bytes_, unsigned int axis_count_)
    {
      var_data_bytes = var_data_bytes_;
      var_data = var_data_bytes_.as<GlyphVariationData> ();
      index = 0;
      axis_count = axis_count_;
      current_tuple = &var_data->get_tuple_var_header ();
      data_offset = 0;
    }

    bool get_shared_indices (hb_vector_t<unsigned int> &shared_indices /* OUT */)
    {
      if (var_data->has_shared_point_numbers ())
      {
	const HBUINT8 *base = &(var_data+var_data->data);
	const HBUINT8 *p = base;
	if (!unpack_points (p, shared_indices, var_data_bytes)) return false;
	data_offset = p - base;
      }
      return true;
    }

    bool is_valid () const
    {
      return (index < var_data->tupleVarCount.get_count ()) &&
	     var_data_bytes.check_range (current_tuple, TupleVariationHeader::min_size) &&
	     var_data_bytes.check_range (current_tuple, hb_max (current_tuple->get_data_size (), current_tuple->get_size (axis_count))) &&
	     current_tuple->get_size (axis_count);
    }

    bool move_to_next ()
    {
      data_offset += current_tuple->get_data_size ();
      current_tuple = &current_tuple->get_next (axis_count);
      index++;
      return is_valid ();
    }

    const HBUINT8 *get_serialized_data () const
    { return &(var_data+var_data->data) + data_offset; }

    private:
    const GlyphVariationData *var_data;
    unsigned int index;
    unsigned int axis_count;
    unsigned int data_offset;

    public:
    hb_bytes_t var_data_bytes;
    const TupleVariationHeader *current_tuple;
  };

  static bool get_tuple_iterator (hb_bytes_t var_data_bytes, unsigned axis_count,
				  hb_vector_t<unsigned int> &shared_indices /* OUT */,
				  tuple_iterator_t *iterator /* OUT */)
  {
    iterator->init (var_data_bytes, axis_count);
    if (!iterator->get_shared_indices (shared_indices))
      return false;
    return iterator->is_valid ();
  }

  bool has_shared_point_numbers () const { return tupleVarCount.has_shared_point_numbers (); }

  static bool unpack_points (const HBUINT8 *&p /* IN/OUT */,
			     hb_vector_t<unsigned int> &points /* OUT */,
			     const hb_bytes_t &bytes)
  {
    enum packed_point_flag_t
    {
      POINTS_ARE_WORDS     = 0x80,
      POINT_RUN_COUNT_MASK = 0x7F
    };

    if (unlikely (!bytes.check_range (p))) return false;

    uint16_t count = *p++;
    if (count & POINTS_ARE_WORDS)
    {
      if (unlikely (!bytes.check_range (p))) return false;
      count = ((count & POINT_RUN_COUNT_MASK) << 8) | *p++;
    }
    points.resize (count);

    unsigned int n = 0;
    uint16_t i = 0;
    while (i < count)
    {
      if (unlikely (!bytes.check_range (p))) return false;
      uint16_t j;
      uint8_t control = *p++;
      uint16_t run_count = (control & POINT_RUN_COUNT_MASK) + 1;
      if (control & POINTS_ARE_WORDS)
      {
	for (j = 0; j < run_count && i < count; j++, i++)
	{
	  if (unlikely (!bytes.check_range ((const HBUINT16 *) p)))
	    return false;
	  n += *(const HBUINT16 *)p;
	  points[i] = n;
	  p += HBUINT16::static_size;
	}
      }
      else
      {
	for (j = 0; j < run_count && i < count; j++, i++)
	{
	  if (unlikely (!bytes.check_range (p))) return false;
	  n += *p++;
	  points[i] = n;
	}
      }
      if (j < run_count) return false;
    }
    return true;
  }

  static bool unpack_deltas (const HBUINT8 *&p /* IN/OUT */,
			     hb_vector_t<int> &deltas /* IN/OUT */,
			     const hb_bytes_t &bytes)
  {
    enum packed_delta_flag_t
    {
      DELTAS_ARE_ZERO      = 0x80,
      DELTAS_ARE_WORDS     = 0x40,
      DELTA_RUN_COUNT_MASK = 0x3F
    };

    unsigned int i = 0;
    unsigned int count = deltas.length;
    while (i < count)
    {
      if (unlikely (!bytes.check_range (p))) return false;
      uint8_t control = *p++;
      unsigned int run_count = (control & DELTA_RUN_COUNT_MASK) + 1;
      unsigned int j;
      if (control & DELTAS_ARE_ZERO)
	for (j = 0; j < run_count && i < count; j++, i++)
	  deltas[i] = 0;
      else if (control & DELTAS_ARE_WORDS)
	for (j = 0; j < run_count && i < count; j++, i++)
	{
	  if (unlikely (!bytes.check_range ((const HBUINT16 *) p)))
	    return false;
	  deltas[i] = *(const HBINT16 *) p;
	  p += HBUINT16::static_size;
	}
      else
	for (j = 0; j < run_count && i < count; j++, i++)
	{
	  if (unlikely (!bytes.check_range (p)))
	    return false;
	  deltas[i] = *(const HBINT8 *) p++;
	}
      if (j < run_count)
	return false;
    }
    return true;
  }

  bool has_data () const { return tupleVarCount; }

  protected:
  struct TupleVarCount : HBUINT16
  {
    bool has_shared_point_numbers () const { return ((*this) & SharedPointNumbers); }
    unsigned int get_count () const { return (*this) & CountMask; }

    protected:
    enum Flags
    {
      SharedPointNumbers= 0x8000u,
      CountMask		= 0x0FFFu
    };
    public:
    DEFINE_SIZE_STATIC (2);
  };

  TupleVarCount	tupleVarCount;  /* A packed field. The high 4 bits are flags, and the
				 * low 12 bits are the number of tuple variation tables
				 * for this glyph. The number of tuple variation tables
				 * can be any number between 1 and 4095. */
  Offset16To<HBUINT8>
		data;		/* Offset from the start of the GlyphVariationData table
				 * to the serialized data. */
  /* TupleVariationHeader tupleVariationHeaders[] *//* Array of tuple variation headers. */
  public:
  DEFINE_SIZE_MIN (4);
};

struct gvar
{
  static constexpr hb_tag_t tableTag = HB_OT_TAG_gvar;

  bool sanitize_shallow (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && (version.major == 1) &&
		  (glyphCount == c->get_num_glyphs ()) &&
		  sharedTuples.sanitize (c, this, axisCount * sharedTupleCount) &&
		  (is_long_offset () ?
		     c->check_array (get_long_offset_array (), glyphCount+1) :
		     c->check_array (get_short_offset_array (), glyphCount+1)) &&
		  c->check_array (((const HBUINT8*)&(this+dataZ)) + get_offset (0),
				  get_offset (glyphCount) - get_offset (0)));
  }

  /* GlyphVariationData not sanitized here; must be checked while accessing each glyph varation data */
  bool sanitize (hb_sanitize_context_t *c) const
  { return sanitize_shallow (c); }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);

    gvar *out = c->serializer->allocate_min<gvar> ();
    if (unlikely (!out)) return_trace (false);

    out->version.major = 1;
    out->version.minor = 0;
    out->axisCount = axisCount;
    out->sharedTupleCount = sharedTupleCount;

    unsigned int num_glyphs = c->plan->num_output_glyphs ();
    out->glyphCount = num_glyphs;

    unsigned int subset_data_size = 0;
    for (hb_codepoint_t gid = (c->plan->flags & HB_SUBSET_FLAGS_NOTDEF_OUTLINE) ? 0 : 1;
         gid < num_glyphs;
         gid++)
    {
      hb_codepoint_t old_gid;
      if (!c->plan->old_gid_for_new_gid (gid, &old_gid)) continue;
      subset_data_size += get_glyph_var_data_bytes (c->source_blob, old_gid).length;
    }

    bool long_offset = subset_data_size & ~0xFFFFu;
    out->flags = long_offset ? 1 : 0;

    HBUINT8 *subset_offsets = c->serializer->allocate_size<HBUINT8> ((long_offset ? 4 : 2) * (num_glyphs + 1));
    if (!subset_offsets) return_trace (false);

    /* shared tuples */
    if (!sharedTupleCount || !sharedTuples)
      out->sharedTuples = 0;
    else
    {
      unsigned int shared_tuple_size = F2DOT14::static_size * axisCount * sharedTupleCount;
      F2DOT14 *tuples = c->serializer->allocate_size<F2DOT14> (shared_tuple_size);
      if (!tuples) return_trace (false);
      out->sharedTuples = (char *) tuples - (char *) out;
      memcpy (tuples, this+sharedTuples, shared_tuple_size);
    }

    char *subset_data = c->serializer->allocate_size<char> (subset_data_size);
    if (!subset_data) return_trace (false);
    out->dataZ = subset_data - (char *) out;

    unsigned int glyph_offset = 0;
    for (hb_codepoint_t gid = (c->plan->flags & HB_SUBSET_FLAGS_NOTDEF_OUTLINE) ? 0 : 1;
         gid < num_glyphs;
         gid++)
    {
      hb_codepoint_t old_gid;
      hb_bytes_t var_data_bytes = c->plan->old_gid_for_new_gid (gid, &old_gid)
				? get_glyph_var_data_bytes (c->source_blob, old_gid)
				: hb_bytes_t ();

      if (long_offset)
	((HBUINT32 *) subset_offsets)[gid] = glyph_offset;
      else
	((HBUINT16 *) subset_offsets)[gid] = glyph_offset / 2;

      if (var_data_bytes.length > 0)
	memcpy (subset_data, var_data_bytes.arrayZ, var_data_bytes.length);
      subset_data += var_data_bytes.length;
      glyph_offset += var_data_bytes.length;
    }
    if (long_offset)
      ((HBUINT32 *) subset_offsets)[num_glyphs] = glyph_offset;
    else
      ((HBUINT16 *) subset_offsets)[num_glyphs] = glyph_offset / 2;

    return_trace (true);
  }

  protected:
  const hb_bytes_t get_glyph_var_data_bytes (hb_blob_t *blob, hb_codepoint_t glyph) const
  {
    unsigned start_offset = get_offset (glyph);
    unsigned length = get_offset (glyph+1) - start_offset;
    hb_bytes_t var_data = blob->as_bytes ().sub_array (((unsigned) dataZ) + start_offset, length);
    return likely (var_data.length >= GlyphVariationData::min_size) ? var_data : hb_bytes_t ();
  }

  bool is_long_offset () const { return flags & 1; }

  unsigned get_offset (unsigned i) const
  { return is_long_offset () ? get_long_offset_array ()[i] : get_short_offset_array ()[i] * 2; }

  const HBUINT32 * get_long_offset_array () const { return (const HBUINT32 *) &offsetZ; }
  const HBUINT16 *get_short_offset_array () const { return (const HBUINT16 *) &offsetZ; }

  public:
  struct accelerator_t
  {
    void init (hb_face_t *face)
    { table = hb_sanitize_context_t ().reference_table<gvar> (face); }
    void fini () { table.destroy (); }

    private:
    struct x_getter { static float get (const contour_point_t &p) { return p.x; } };
    struct y_getter { static float get (const contour_point_t &p) { return p.y; } };

    template <typename T>
    static float infer_delta (const hb_array_t<contour_point_t> points,
			      const hb_array_t<contour_point_t> deltas,
			      unsigned int target, unsigned int prev, unsigned int next)
    {
      float target_val = T::get (points[target]);
      float prev_val = T::get (points[prev]);
      float next_val = T::get (points[next]);
      float prev_delta = T::get (deltas[prev]);
      float next_delta = T::get (deltas[next]);

      if (prev_val == next_val)
	return (prev_delta == next_delta) ? prev_delta : 0.f;
      else if (target_val <= hb_min (prev_val, next_val))
	return (prev_val < next_val) ? prev_delta : next_delta;
      else if (target_val >= hb_max (prev_val, next_val))
	return (prev_val > next_val) ? prev_delta : next_delta;

      /* linear interpolation */
      float r = (target_val - prev_val) / (next_val - prev_val);
      return (1.f - r) * prev_delta + r * next_delta;
    }

    static unsigned int next_index (unsigned int i, unsigned int start, unsigned int end)
    { return (i >= end) ? start : (i + 1); }

    public:
    bool apply_deltas_to_points (hb_codepoint_t glyph, hb_font_t *font,
				 const hb_array_t<contour_point_t> points) const
    {
      /* num_coords should exactly match gvar's axisCount due to how GlyphVariationData tuples are aligned */
      if (!font->num_coords || font->num_coords != table->axisCount) return true;

      if (unlikely (glyph >= table->glyphCount)) return true;

      hb_bytes_t var_data_bytes = table->get_glyph_var_data_bytes (table.get_blob (), glyph);
      if (!var_data_bytes.as<GlyphVariationData> ()->has_data ()) return true;
      hb_vector_t<unsigned int> shared_indices;
      GlyphVariationData::tuple_iterator_t iterator;
      if (!GlyphVariationData::get_tuple_iterator (var_data_bytes, table->axisCount,
						   shared_indices, &iterator))
	return true; /* so isn't applied at all */

      /* Save original points for inferred delta calculation */
      contour_point_vector_t orig_points;
      orig_points.resize (points.length);
      for (unsigned int i = 0; i < orig_points.length; i++)
	orig_points[i] = points[i];

      contour_point_vector_t deltas; /* flag is used to indicate referenced point */
      deltas.resize (points.length);

      hb_vector_t<unsigned> end_points;
      for (unsigned i = 0; i < points.length; ++i)
	if (points[i].is_end_point)
	  end_points.push (i);

      int *coords = font->coords;
      unsigned num_coords = font->num_coords;
      hb_array_t<const F2DOT14> shared_tuples = (table+table->sharedTuples).as_array (table->sharedTupleCount * table->axisCount);
      do
      {
	float scalar = iterator.current_tuple->calculate_scalar (coords, num_coords, shared_tuples);
	if (scalar == 0.f) continue;
	const HBUINT8 *p = iterator.get_serialized_data ();
	unsigned int length = iterator.current_tuple->get_data_size ();
	if (unlikely (!iterator.var_data_bytes.check_range (p, length)))
	  return false;

	hb_bytes_t bytes ((const char *) p, length);
	hb_vector_t<unsigned int> private_indices;
	if (iterator.current_tuple->has_private_points () &&
	    !GlyphVariationData::unpack_points (p, private_indices, bytes))
	  return false;
	const hb_array_t<unsigned int> &indices = private_indices.length ? private_indices : shared_indices;

	bool apply_to_all = (indices.length == 0);
	unsigned int num_deltas = apply_to_all ? points.length : indices.length;
	hb_vector_t<int> x_deltas;
	x_deltas.resize (num_deltas);
	if (!GlyphVariationData::unpack_deltas (p, x_deltas, bytes))
	  return false;
	hb_vector_t<int> y_deltas;
	y_deltas.resize (num_deltas);
	if (!GlyphVariationData::unpack_deltas (p, y_deltas, bytes))
	  return false;

	for (unsigned int i = 0; i < deltas.length; i++)
	  deltas[i].init ();
	for (unsigned int i = 0; i < num_deltas; i++)
	{
	  unsigned int pt_index = apply_to_all ? i : indices[i];
	  deltas[pt_index].flag = 1;	/* this point is referenced, i.e., explicit deltas specified */
	  deltas[pt_index].x += x_deltas[i] * scalar;
	  deltas[pt_index].y += y_deltas[i] * scalar;
	}

	/* infer deltas for unreferenced points */
	unsigned start_point = 0;
	for (unsigned c = 0; c < end_points.length; c++)
	{
	  unsigned end_point = end_points[c];

	  /* Check the number of unreferenced points in a contour. If no unref points or no ref points, nothing to do. */
	  unsigned unref_count = 0;
	  for (unsigned i = start_point; i <= end_point; i++)
	    if (!deltas[i].flag) unref_count++;

	  unsigned j = start_point;
	  if (unref_count == 0 || unref_count > end_point - start_point)
	    goto no_more_gaps;

	  for (;;)
	  {
	    /* Locate the next gap of unreferenced points between two referenced points prev and next.
	     * Note that a gap may wrap around at left (start_point) and/or at right (end_point).
	     */
	    unsigned int prev, next, i;
	    for (;;)
	    {
	      i = j;
	      j = next_index (i, start_point, end_point);
	      if (deltas[i].flag && !deltas[j].flag) break;
	    }
	    prev = j = i;
	    for (;;)
	    {
	      i = j;
	      j = next_index (i, start_point, end_point);
	      if (!deltas[i].flag && deltas[j].flag) break;
	    }
	    next = j;
	    /* Infer deltas for all unref points in the gap between prev and next */
	    i = prev;
	    for (;;)
	    {
	      i = next_index (i, start_point, end_point);
	      if (i == next) break;
	      deltas[i].x = infer_delta<x_getter> (orig_points.as_array (), deltas.as_array (), i, prev, next);
	      deltas[i].y = infer_delta<y_getter> (orig_points.as_array (), deltas.as_array (), i, prev, next);
	      if (--unref_count == 0) goto no_more_gaps;
	    }
	  }
no_more_gaps:
	  start_point = end_point + 1;
	}

	/* apply specified / inferred deltas to points */
	for (unsigned int i = 0; i < points.length; i++)
	{
	  points[i].x += deltas[i].x;
	  points[i].y += deltas[i].y;
	}
      } while (iterator.move_to_next ());

      return true;
    }

    unsigned int get_axis_count () const { return table->axisCount; }

    private:
    hb_blob_ptr_t<gvar> table;
  };

  protected:
  FixedVersion<>version;	/* Version number of the glyph variations table
				 * Set to 0x00010000u. */
  HBUINT16	axisCount;	/* The number of variation axes for this font. This must be
				 * the same number as axisCount in the 'fvar' table. */
  HBUINT16	sharedTupleCount;
				/* The number of shared tuple records. Shared tuple records
				 * can be referenced within glyph variation data tables for
				 * multiple glyphs, as opposed to other tuple records stored
				 * directly within a glyph variation data table. */
  NNOffset32To<UnsizedArrayOf<F2DOT14>>
		sharedTuples;	/* Offset from the start of this table to the shared tuple records.
				 * Array of tuple records shared across all glyph variation data tables. */
  HBUINT16	glyphCount;	/* The number of glyphs in this font. This must match the number of
				 * glyphs stored elsewhere in the font. */
  HBUINT16	flags;		/* Bit-field that gives the format of the offset array that follows.
				 * If bit 0 is clear, the offsets are uint16; if bit 0 is set, the
				 * offsets are uint32. */
  Offset32To<GlyphVariationData>
		dataZ;		/* Offset from the start of this table to the array of
				 * GlyphVariationData tables. */
  UnsizedArrayOf<HBUINT8>
		offsetZ;	/* Offsets from the start of the GlyphVariationData array
				 * to each GlyphVariationData table. */
  public:
  DEFINE_SIZE_MIN (20);
};

struct gvar_accelerator_t : gvar::accelerator_t {};

} /* namespace OT */

#endif /* HB_OT_VAR_GVAR_TABLE_HH */
