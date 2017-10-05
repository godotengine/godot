/*
 * Copyright © 2017  Google, Inc.
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
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_AAT_LAYOUT_COMMON_PRIVATE_HH
#define HB_AAT_LAYOUT_COMMON_PRIVATE_HH

#include "hb-aat-layout-private.hh"


namespace AAT {

using namespace OT;


/*
 * Binary Searching Tables
 */

struct BinSearchHeader
{

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  HBUINT16	unitSize;	/* Size of a lookup unit for this search in bytes. */
  HBUINT16	nUnits;		/* Number of units of the preceding size to be searched. */
  HBUINT16	searchRange;	/* The value of unitSize times the largest power of 2
				 * that is less than or equal to the value of nUnits. */
  HBUINT16	entrySelector;	/* The log base 2 of the largest power of 2 less than
				 * or equal to the value of nUnits. */
  HBUINT16	rangeShift;	/* The value of unitSize times the difference of the
				 * value of nUnits minus the largest power of 2 less
				 * than or equal to the value of nUnits. */
  public:
  DEFINE_SIZE_STATIC (10);
};

template <typename Type>
struct BinSearchArrayOf
{
  inline const Type& operator [] (unsigned int i) const
  {
    if (unlikely (i >= header.nUnits)) return Null(Type);
    return StructAtOffset<Type> (bytesZ, i * header.unitSize);
  }
  inline Type& operator [] (unsigned int i)
  {
    return StructAtOffset<Type> (bytesZ, i * header.unitSize);
  }
  inline unsigned int get_size (void) const
  { return header.static_size + header.nUnits * header.unitSize; }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!sanitize_shallow (c))) return_trace (false);

    /* Note: for structs that do not reference other structs,
     * we do not need to call their sanitize() as we already did
     * a bound check on the aggregate array size.  We just include
     * a small unreachable expression to make sure the structs
     * pointed to do have a simple sanitize(), ie. they do not
     * reference other structs via offsets.
     */
    (void) (false && StructAtOffset<Type> (bytesZ, 0).sanitize (c));

    return_trace (true);
  }
  inline bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!sanitize_shallow (c))) return_trace (false);
    unsigned int count = header.nUnits;
    for (unsigned int i = 0; i < count; i++)
      if (unlikely (!(*this)[i].sanitize (c, base)))
        return_trace (false);
    return_trace (true);
  }

  template <typename T>
  inline const Type *bsearch (const T &key) const
  {
    unsigned int size = header.unitSize;
    int min = 0, max = (int) header.nUnits - 1;
    while (min <= max)
    {
      int mid = (min + max) / 2;
      const Type *p = (const Type *) (((const char *) bytesZ) + (mid * size));
      int c = p->cmp (key);
      if (c < 0)
	max = mid - 1;
      else if (c > 0)
	min = mid + 1;
      else
	return p;
    }
    return nullptr;
  }

  private:
  inline bool sanitize_shallow (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (header.sanitize (c) &&
		  Type::static_size >= header.unitSize &&
		  c->check_array (bytesZ, header.unitSize, header.nUnits));
  }

  protected:
  BinSearchHeader	header;
  HBUINT8		bytesZ[VAR];
  public:
  DEFINE_SIZE_ARRAY (10, bytesZ);
};


/*
 * Lookup Table
 */

template <typename T> struct Lookup;

template <typename T>
struct LookupFormat0
{
  friend struct Lookup<T>;

  private:
  inline const T* get_value (hb_codepoint_t glyph_id, unsigned int num_glyphs) const
  {
    if (unlikely (glyph_id >= num_glyphs)) return nullptr;
    return &arrayZ[glyph_id];
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (arrayZ.sanitize (c, c->get_num_glyphs ()));
  }

  protected:
  HBUINT16	format;		/* Format identifier--format = 0 */
  UnsizedArrayOf<T>
		arrayZ;		/* Array of lookup values, indexed by glyph index. */
  public:
  DEFINE_SIZE_ARRAY (2, arrayZ);
};


template <typename T>
struct LookupSegmentSingle
{
  inline int cmp (hb_codepoint_t g) const {
    return g < first ? -1 : g <= last ? 0 : +1 ;
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && value.sanitize (c));
  }

  GlyphID	last;		/* Last GlyphID in this segment */
  GlyphID	first;		/* First GlyphID in this segment */
  T		value;		/* The lookup value (only one) */
  public:
  DEFINE_SIZE_STATIC (4 + T::static_size);
};

template <typename T>
struct LookupFormat2
{
  friend struct Lookup<T>;

  private:
  inline const T* get_value (hb_codepoint_t glyph_id) const
  {
    const LookupSegmentSingle<T> *v = segments.bsearch (glyph_id);
    return v ? &v->value : nullptr;
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (segments.sanitize (c));
  }

  protected:
  HBUINT16	format;		/* Format identifier--format = 2 */
  BinSearchArrayOf<LookupSegmentSingle<T> >
		segments;	/* The actual segments. These must already be sorted,
				 * according to the first word in each one (the last
				 * glyph in each segment). */
  public:
  DEFINE_SIZE_ARRAY (8, segments);
};

template <typename T>
struct LookupSegmentArray
{
  inline const T* get_value (hb_codepoint_t glyph_id, const void *base) const
  {
    return first <= glyph_id && glyph_id <= last ? &(base+valuesZ)[glyph_id - first] : nullptr;
  }

  inline int cmp (hb_codepoint_t g) const {
    return g < first ? -1 : g <= last ? 0 : +1 ;
  }

  inline bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  first <= last &&
		  valuesZ.sanitize (c, base, last - first + 1));
  }

  GlyphID	last;		/* Last GlyphID in this segment */
  GlyphID	first;		/* First GlyphID in this segment */
  OffsetTo<UnsizedArrayOf<T> >
		valuesZ;	/* A 16-bit offset from the start of
				 * the table to the data. */
  public:
  DEFINE_SIZE_STATIC (6);
};

template <typename T>
struct LookupFormat4
{
  friend struct Lookup<T>;

  private:
  inline const T* get_value (hb_codepoint_t glyph_id) const
  {
    const LookupSegmentArray<T> *v = segments.bsearch (glyph_id);
    return v ? v->get_value (glyph_id, this) : nullptr;
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (segments.sanitize (c, this));
  }

  protected:
  HBUINT16	format;		/* Format identifier--format = 2 */
  BinSearchArrayOf<LookupSegmentArray<T> >
		segments;	/* The actual segments. These must already be sorted,
				 * according to the first word in each one (the last
				 * glyph in each segment). */
  public:
  DEFINE_SIZE_ARRAY (8, segments);
};

template <typename T>
struct LookupSingle
{
  inline int cmp (hb_codepoint_t g) const { return glyph.cmp (g); }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && value.sanitize (c));
  }

  GlyphID	glyph;		/* Last GlyphID */
  T		value;		/* The lookup value (only one) */
  public:
  DEFINE_SIZE_STATIC (4 + T::static_size);
};

template <typename T>
struct LookupFormat6
{
  friend struct Lookup<T>;

  private:
  inline const T* get_value (hb_codepoint_t glyph_id) const
  {
    const LookupSingle<T> *v = entries.bsearch (glyph_id);
    return v ? &v->value : nullptr;
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (entries.sanitize (c));
  }

  protected:
  HBUINT16	format;		/* Format identifier--format = 6 */
  BinSearchArrayOf<LookupSingle<T> >
		entries;	/* The actual entries, sorted by glyph index. */
  public:
  DEFINE_SIZE_ARRAY (8, entries);
};

template <typename T>
struct LookupFormat8
{
  friend struct Lookup<T>;

  private:
  inline const T* get_value (hb_codepoint_t glyph_id) const
  {
    return firstGlyph <= glyph_id && glyph_id - firstGlyph < glyphCount ? &valueArrayZ[glyph_id - firstGlyph] : nullptr;
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && valueArrayZ.sanitize (c, glyphCount));
  }

  protected:
  HBUINT16	format;		/* Format identifier--format = 6 */
  GlyphID	firstGlyph;	/* First glyph index included in the trimmed array. */
  HBUINT16	glyphCount;	/* Total number of glyphs (equivalent to the last
				 * glyph minus the value of firstGlyph plus 1). */
  UnsizedArrayOf<T>
		valueArrayZ;	/* The lookup values (indexed by the glyph index
				 * minus the value of firstGlyph). */
  public:
  DEFINE_SIZE_ARRAY (6, valueArrayZ);
};

template <typename T>
struct Lookup
{
  inline const T* get_value (hb_codepoint_t glyph_id, unsigned int num_glyphs) const
  {
    switch (u.format) {
    case 0: return u.format0.get_value (glyph_id, num_glyphs);
    case 2: return u.format2.get_value (glyph_id);
    case 4: return u.format4.get_value (glyph_id);
    case 6: return u.format6.get_value (glyph_id);
    case 8: return u.format8.get_value (glyph_id);
    default:return nullptr;
    }
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (!u.format.sanitize (c)) return_trace (false);
    switch (u.format) {
    case 0: return_trace (u.format0.sanitize (c));
    case 2: return_trace (u.format2.sanitize (c));
    case 4: return_trace (u.format4.sanitize (c));
    case 6: return_trace (u.format6.sanitize (c));
    case 8: return_trace (u.format8.sanitize (c));
    default:return_trace (true);
    }
  }

  protected:
  union {
  HBUINT16		format;		/* Format identifier */
  LookupFormat0<T>	format0;
  LookupFormat2<T>	format2;
  LookupFormat4<T>	format4;
  LookupFormat6<T>	format6;
  LookupFormat8<T>	format8;
  } u;
  public:
  DEFINE_SIZE_UNION (2, format);
};


/*
 * Extended State Table
 */

template <typename T>
struct Entry
{
  inline bool sanitize (hb_sanitize_context_t *c, unsigned int count) const
  {
    TRACE_SANITIZE (this);
    /* Note, we don't recurse-sanitize data because we don't access it.
     * That said, in our DEFINE_SIZE_STATIC we access T::static_size,
     * which ensures that data has a simple sanitize(). To be determined
     * if I need to remove that as well. */
    return_trace (c->check_struct (this));
  }

  public:
  HBUINT16	newState;	/* Byte offset from beginning of state table
				 * to the new state. Really?!?! Or just state
				 * number?  The latter in morx for sure. */
  HBUINT16	flags;		/* Table specific. */
  T		data;		/* Optional offsets to per-glyph tables. */
  public:
  DEFINE_SIZE_STATIC (4 + T::static_size);
};

template <>
struct Entry<void>
{
  inline bool sanitize (hb_sanitize_context_t *c, unsigned int count) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  public:
  HBUINT16	newState;	/* Byte offset from beginning of state table to the new state. */
  HBUINT16	flags;		/* Table specific. */
  public:
  DEFINE_SIZE_STATIC (4);
};

template <typename Extra>
struct StateTable
{
  inline unsigned int get_class (hb_codepoint_t glyph_id, unsigned int num_glyphs) const
  {
    const HBUINT16 *v = (this+classTable).get_value (glyph_id, num_glyphs);
    return v ? *v : 1;
  }

  inline const Entry<Extra> *get_entries () const
  {
    return (this+entryTable).arrayZ;
  }

  inline const Entry<Extra> *get_entryZ (unsigned int state, unsigned int klass) const
  {
    if (unlikely (klass >= nClasses)) return nullptr;

    const HBUINT16 *states = (this+stateArrayTable).arrayZ;
    const Entry<Extra> *entries = (this+entryTable).arrayZ;

    unsigned int entry = states[state * nClasses + klass];

    return &entries[entry];
  }

  inline bool sanitize (hb_sanitize_context_t *c,
			unsigned int *num_entries_out = nullptr) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!(c->check_struct (this) &&
		    classTable.sanitize (c, this)))) return_trace (false);

    const HBUINT16 *states = (this+stateArrayTable).arrayZ;
    const Entry<Extra> *entries = (this+entryTable).arrayZ;

    unsigned int num_states = 1;
    unsigned int num_entries = 0;

    unsigned int state = 0;
    unsigned int entry = 0;
    while (state < num_states)
    {
      if (unlikely (!c->check_array (states,
				     states[0].static_size * nClasses,
				     num_states)))
	return_trace (false);
      { /* Sweep new states. */
	const HBUINT16 *stop = &states[num_states * nClasses];
	for (const HBUINT16 *p = &states[state * nClasses]; p < stop; p++)
	  num_entries = MAX<unsigned int> (num_entries, *p + 1);
	state = num_states;
      }

      if (unlikely (!c->check_array (entries,
				     entries[0].static_size,
				     num_entries)))
	return_trace (false);
      { /* Sweep new entries. */
	const Entry<Extra> *stop = &entries[num_entries];
	for (const Entry<Extra> *p = &entries[entry]; p < stop; p++)
	  num_states = MAX<unsigned int> (num_states, p->newState + 1);
	entry = num_entries;
      }
    }

    if (num_entries_out)
      *num_entries_out = num_entries;

    return_trace (true);
  }

  protected:
  HBUINT32	nClasses;	/* Number of classes, which is the number of indices
				 * in a single line in the state array. */
  LOffsetTo<Lookup<HBUINT16> >
		classTable;	/* Offset to the class table. */
  LOffsetTo<UnsizedArrayOf<HBUINT16> >
		stateArrayTable;/* Offset to the state array. */
  LOffsetTo<UnsizedArrayOf<Entry<Extra> > >
		entryTable;	/* Offset to the entry array. */

  public:
  DEFINE_SIZE_STATIC (16);
};

template <typename EntryData>
struct StateTableDriver
{
  inline StateTableDriver (const StateTable<EntryData> &machine_,
			   hb_buffer_t *buffer_,
			   hb_face_t *face_) :
	      machine (machine_),
	      buffer (buffer_),
	      num_glyphs (face_->get_num_glyphs ()) {}

  template <typename context_t>
  inline void drive (context_t *c)
  {
    hb_glyph_info_t *info = buffer->info;

    if (!c->in_place)
      buffer->clear_output ();

    unsigned int state = 0;
    bool last_was_dont_advance = false;
    for (buffer->idx = 0;;)
    {
      unsigned int klass = buffer->idx < buffer->len ?
			   machine.get_class (info[buffer->idx].codepoint, num_glyphs) :
			   0 /* End of text */;
      const Entry<EntryData> *entry = machine.get_entryZ (state, klass);
      if (unlikely (!entry))
	break;

      /* Unsafe-to-break before this if not in state 0, as things might
       * go differently if we start from state 0 here. */
      if (state && buffer->idx)
      {
	/* If there's no action and we're just epsilon-transitioning to state 0,
	 * safe to break. */
	if (c->is_actionable (this, entry) ||
	    !(entry->newState == 0 && entry->flags == context_t::DontAdvance))
	  buffer->unsafe_to_break (buffer->idx - 1, buffer->idx + 1);
      }

      /* Unsafe-to-break if end-of-text would kick in here. */
      if (buffer->idx + 2 <= buffer->len)
      {
	const Entry<EntryData> *end_entry = machine.get_entryZ (state, 0);
	if (c->is_actionable (this, end_entry))
	  buffer->unsafe_to_break (buffer->idx, buffer->idx + 2);
      }

      if (unlikely (!c->transition (this, entry)))
        break;

      last_was_dont_advance = (entry->flags & context_t::DontAdvance) && buffer->max_ops-- > 0;

      state = entry->newState;

      if (buffer->idx == buffer->len)
        break;

      if (!last_was_dont_advance)
        buffer->next_glyph ();
    }

    if (!c->in_place)
    {
      for (; buffer->idx < buffer->len;)
        buffer->next_glyph ();
      buffer->swap_buffers ();
    }
  }

  public:
  const StateTable<EntryData> &machine;
  hb_buffer_t *buffer;
  unsigned int num_glyphs;
};



struct hb_aat_apply_context_t :
       hb_dispatch_context_t<hb_aat_apply_context_t, bool, HB_DEBUG_APPLY>
{
  inline const char *get_name (void) { return "APPLY"; }
  template <typename T>
  inline return_t dispatch (const T &obj) { return obj.apply (this); }
  static return_t default_return_value (void) { return false; }
  bool stop_sublookup_iteration (return_t r) const { return r; }

  hb_font_t *font;
  hb_face_t *face;
  hb_buffer_t *buffer;
  hb_sanitize_context_t sanitizer;

  /* Unused. For debug tracing only. */
  unsigned int lookup_index;
  unsigned int debug_depth;

  inline hb_aat_apply_context_t (hb_font_t *font_,
				 hb_buffer_t *buffer_,
				 hb_blob_t *table) :
		font (font_), face (font->face), buffer (buffer_),
		sanitizer (), lookup_index (0), debug_depth (0)
  {
    sanitizer.init (table);
    sanitizer.set_num_glyphs (face->get_num_glyphs ());
    sanitizer.start_processing ();
  }

  inline void set_lookup_index (unsigned int i) { lookup_index = i; }

  inline ~hb_aat_apply_context_t (void)
  {
    sanitizer.end_processing ();
  }
};


} /* namespace AAT */


#endif /* HB_AAT_LAYOUT_COMMON_PRIVATE_HH */
