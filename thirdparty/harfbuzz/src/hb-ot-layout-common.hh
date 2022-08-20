/*
 * Copyright © 2007,2008,2009  Red Hat, Inc.
 * Copyright © 2010,2012  Google, Inc.
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
 * Red Hat Author(s): Behdad Esfahbod
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_OT_LAYOUT_COMMON_HH
#define HB_OT_LAYOUT_COMMON_HH

#include "hb.hh"
#include "hb-ot-layout.hh"
#include "hb-open-type.hh"
#include "hb-set.hh"
#include "hb-bimap.hh"

#include "OT/Layout/Common/Coverage.hh"
#include "OT/Layout/types.hh"

// TODO(garretrieger): cleanup these after migration.
using OT::Layout::Common::Coverage;
using OT::Layout::Common::RangeRecord;
using OT::Layout::SmallTypes;
using OT::Layout::MediumTypes;

#ifndef HB_MAX_NESTING_LEVEL
#define HB_MAX_NESTING_LEVEL	64
#endif
#ifndef HB_MAX_CONTEXT_LENGTH
#define HB_MAX_CONTEXT_LENGTH	64
#endif
#ifndef HB_CLOSURE_MAX_STAGES
/*
 * The maximum number of times a lookup can be applied during shaping.
 * Used to limit the number of iterations of the closure algorithm.
 * This must be larger than the number of times add_gsub_pause() is
 * called in a collect_features call of any shaper.
 */
#define HB_CLOSURE_MAX_STAGES	12
#endif

#ifndef HB_MAX_SCRIPTS
#define HB_MAX_SCRIPTS	500
#endif

#ifndef HB_MAX_LANGSYS
#define HB_MAX_LANGSYS	2000
#endif

#ifndef HB_MAX_LANGSYS_FEATURE_COUNT
#define HB_MAX_LANGSYS_FEATURE_COUNT 50000
#endif

#ifndef HB_MAX_FEATURE_INDICES
#define HB_MAX_FEATURE_INDICES	1500
#endif

#ifndef HB_MAX_LOOKUP_VISIT_COUNT
#define HB_MAX_LOOKUP_VISIT_COUNT	35000
#endif


namespace OT {

template<typename Iterator>
static inline void ClassDef_serialize (hb_serialize_context_t *c,
				       Iterator it);

static void ClassDef_remap_and_serialize (
    hb_serialize_context_t *c,
    const hb_set_t &klasses,
    bool use_class_zero,
    hb_sorted_vector_t<hb_pair_t<hb_codepoint_t, hb_codepoint_t>> &glyph_and_klass, /* IN/OUT */
    hb_map_t *klass_map /*IN/OUT*/);


struct hb_prune_langsys_context_t
{
  hb_prune_langsys_context_t (const void         *table_,
                              hb_hashmap_t<unsigned, hb::unique_ptr<hb_set_t>> *script_langsys_map_,
                              const hb_map_t     *duplicate_feature_map_,
                              hb_set_t           *new_collected_feature_indexes_)
      :table (table_),
      script_langsys_map (script_langsys_map_),
      duplicate_feature_map (duplicate_feature_map_),
      new_feature_indexes (new_collected_feature_indexes_),
      script_count (0),langsys_feature_count (0) {}

  bool visitScript ()
  { return script_count++ < HB_MAX_SCRIPTS; }

  bool visitLangsys (unsigned feature_count)
  {
    langsys_feature_count += feature_count;
    return langsys_feature_count < HB_MAX_LANGSYS_FEATURE_COUNT;
  }

  public:
  const void *table;
  hb_hashmap_t<unsigned, hb::unique_ptr<hb_set_t>> *script_langsys_map;
  const hb_map_t     *duplicate_feature_map;
  hb_set_t           *new_feature_indexes;

  private:
  unsigned script_count;
  unsigned langsys_feature_count;
};

struct hb_subset_layout_context_t :
  hb_dispatch_context_t<hb_subset_layout_context_t, hb_empty_t, HB_DEBUG_SUBSET>
{
  const char *get_name () { return "SUBSET_LAYOUT"; }
  static return_t default_return_value () { return hb_empty_t (); }

  bool visitScript ()
  {
    return script_count++ < HB_MAX_SCRIPTS;
  }

  bool visitLangSys ()
  {
    return langsys_count++ < HB_MAX_LANGSYS;
  }

  bool visitFeatureIndex (int count)
  {
    feature_index_count += count;
    return feature_index_count < HB_MAX_FEATURE_INDICES;
  }

  bool visitLookupIndex()
  {
    lookup_index_count++;
    return lookup_index_count < HB_MAX_LOOKUP_VISIT_COUNT;
  }

  hb_subset_context_t *subset_context;
  const hb_tag_t table_tag;
  const hb_map_t *lookup_index_map;
  const hb_hashmap_t<unsigned, hb::unique_ptr<hb_set_t>> *script_langsys_map;
  const hb_map_t *feature_index_map;
  unsigned cur_script_index;

  hb_subset_layout_context_t (hb_subset_context_t *c_,
			      hb_tag_t tag_,
			      hb_map_t *lookup_map_,
			      hb_hashmap_t<unsigned, hb::unique_ptr<hb_set_t>> *script_langsys_map_,
			      hb_map_t *feature_index_map_) :
				subset_context (c_),
				table_tag (tag_),
				lookup_index_map (lookup_map_),
				script_langsys_map (script_langsys_map_),
				feature_index_map (feature_index_map_),
				cur_script_index (0xFFFFu),
				script_count (0),
				langsys_count (0),
				feature_index_count (0),
				lookup_index_count (0)
  {}

  private:
  unsigned script_count;
  unsigned langsys_count;
  unsigned feature_index_count;
  unsigned lookup_index_count;
};

struct hb_collect_variation_indices_context_t :
       hb_dispatch_context_t<hb_collect_variation_indices_context_t>
{
  template <typename T>
  return_t dispatch (const T &obj) { obj.collect_variation_indices (this); return hb_empty_t (); }
  static return_t default_return_value () { return hb_empty_t (); }

  hb_set_t *layout_variation_indices;
  const hb_set_t *glyph_set;
  const hb_map_t *gpos_lookups;

  hb_collect_variation_indices_context_t (hb_set_t *layout_variation_indices_,
					  const hb_set_t *glyph_set_,
					  const hb_map_t *gpos_lookups_) :
					layout_variation_indices (layout_variation_indices_),
					glyph_set (glyph_set_),
					gpos_lookups (gpos_lookups_) {}
};

template<typename OutputArray>
struct subset_offset_array_t
{
  subset_offset_array_t (hb_subset_context_t *subset_context_,
			 OutputArray& out_,
			 const void *base_) : subset_context (subset_context_),
					      out (out_), base (base_) {}

  template <typename T>
  bool operator () (T&& offset)
  {
    auto snap = subset_context->serializer->snapshot ();
    auto *o = out.serialize_append (subset_context->serializer);
    if (unlikely (!o)) return false;
    bool ret = o->serialize_subset (subset_context, offset, base);
    if (!ret)
    {
      out.pop ();
      subset_context->serializer->revert (snap);
    }
    return ret;
  }

  private:
  hb_subset_context_t *subset_context;
  OutputArray &out;
  const void *base;
};


template<typename OutputArray, typename Arg>
struct subset_offset_array_arg_t
{
  subset_offset_array_arg_t (hb_subset_context_t *subset_context_,
			     OutputArray& out_,
			     const void *base_,
			     Arg &&arg_) : subset_context (subset_context_), out (out_),
					  base (base_), arg (arg_) {}

  template <typename T>
  bool operator () (T&& offset)
  {
    auto snap = subset_context->serializer->snapshot ();
    auto *o = out.serialize_append (subset_context->serializer);
    if (unlikely (!o)) return false;
    bool ret = o->serialize_subset (subset_context, offset, base, arg);
    if (!ret)
    {
      out.pop ();
      subset_context->serializer->revert (snap);
    }
    return ret;
  }

  private:
  hb_subset_context_t *subset_context;
  OutputArray &out;
  const void *base;
  Arg &&arg;
};

/*
 * Helper to subset an array of offsets. Subsets the thing pointed to by each offset
 * and discards the offset in the array if the subset operation results in an empty
 * thing.
 */
struct
{
  template<typename OutputArray>
  subset_offset_array_t<OutputArray>
  operator () (hb_subset_context_t *subset_context, OutputArray& out,
	       const void *base) const
  { return subset_offset_array_t<OutputArray> (subset_context, out, base); }

  /* Variant with one extra argument passed to serialize_subset */
  template<typename OutputArray, typename Arg>
  subset_offset_array_arg_t<OutputArray, Arg>
  operator () (hb_subset_context_t *subset_context, OutputArray& out,
	       const void *base, Arg &&arg) const
  { return subset_offset_array_arg_t<OutputArray, Arg> (subset_context, out, base, arg); }
}
HB_FUNCOBJ (subset_offset_array);

template<typename OutputArray>
struct subset_record_array_t
{
  subset_record_array_t (hb_subset_layout_context_t *c_, OutputArray* out_,
			 const void *base_) : subset_layout_context (c_),
					      out (out_), base (base_) {}

  template <typename T>
  void
  operator () (T&& record)
  {
    auto snap = subset_layout_context->subset_context->serializer->snapshot ();
    bool ret = record.subset (subset_layout_context, base);
    if (!ret) subset_layout_context->subset_context->serializer->revert (snap);
    else out->len++;
  }

  private:
  hb_subset_layout_context_t *subset_layout_context;
  OutputArray *out;
  const void *base;
};

/*
 * Helper to subset a RecordList/record array. Subsets each Record in the array and
 * discards the record if the subset operation returns false.
 */
struct
{
  template<typename OutputArray>
  subset_record_array_t<OutputArray>
  operator () (hb_subset_layout_context_t *c, OutputArray* out,
	       const void *base) const
  { return subset_record_array_t<OutputArray> (c, out, base); }
}
HB_FUNCOBJ (subset_record_array);


template<typename OutputArray>
struct serialize_math_record_array_t
{
  serialize_math_record_array_t (hb_serialize_context_t *serialize_context_,
                         OutputArray& out_,
                         const void *base_) : serialize_context (serialize_context_),
                                              out (out_), base (base_) {}

  template <typename T>
  bool operator () (T&& record)
  {
    if (!serialize_context->copy (record, base)) return false;
    out.len++;
    return true;
  }

  private:
  hb_serialize_context_t *serialize_context;
  OutputArray &out;
  const void *base;
};

/*
 * Helper to serialize an array of MATH records.
 */
struct
{
  template<typename OutputArray>
  serialize_math_record_array_t<OutputArray>
  operator () (hb_serialize_context_t *serialize_context, OutputArray& out,
               const void *base) const
  { return serialize_math_record_array_t<OutputArray> (serialize_context, out, base); }

}
HB_FUNCOBJ (serialize_math_record_array);

/*
 *
 * OpenType Layout Common Table Formats
 *
 */


/*
 * Script, ScriptList, LangSys, Feature, FeatureList, Lookup, LookupList
 */

struct IndexArray : Array16Of<Index>
{
  bool intersects (const hb_map_t *indexes) const
  { return hb_any (*this, indexes); }

  template <typename Iterator,
	    hb_requires (hb_is_iterator (Iterator))>
  void serialize (hb_serialize_context_t *c,
		  hb_subset_layout_context_t *l,
		  Iterator it)
  {
    if (!it) return;
    if (unlikely (!c->extend_min ((*this)))) return;

    for (const auto _ : it)
    {
      if (!l->visitLookupIndex()) break;

      Index i;
      i = _;
      c->copy (i);
      this->len++;
    }
  }

  unsigned int get_indexes (unsigned int start_offset,
			    unsigned int *_count /* IN/OUT */,
			    unsigned int *_indexes /* OUT */) const
  {
    if (_count)
    {
      + this->sub_array (start_offset, _count)
      | hb_sink (hb_array (_indexes, *_count))
      ;
    }
    return this->len;
  }

  void add_indexes_to (hb_set_t* output /* OUT */) const
  {
    output->add_array (as_array ());
  }
};


struct Record_sanitize_closure_t {
  hb_tag_t tag;
  const void *list_base;
};

template <typename Type>
struct Record
{
  int cmp (hb_tag_t a) const { return tag.cmp (a); }

  bool subset (hb_subset_layout_context_t *c, const void *base) const
  {
    TRACE_SUBSET (this);
    auto *out = c->subset_context->serializer->embed (this);
    if (unlikely (!out)) return_trace (false);
    bool ret = out->offset.serialize_subset (c->subset_context, offset, base, c, &tag);
    return_trace (ret);
  }

  bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    const Record_sanitize_closure_t closure = {tag, base};
    return_trace (c->check_struct (this) && offset.sanitize (c, base, &closure));
  }

  Tag		tag;		/* 4-byte Tag identifier */
  Offset16To<Type>
		offset;		/* Offset from beginning of object holding
				 * the Record */
  public:
  DEFINE_SIZE_STATIC (6);
};

template <typename Type>
struct RecordArrayOf : SortedArray16Of<Record<Type>>
{
  const Offset16To<Type>& get_offset (unsigned int i) const
  { return (*this)[i].offset; }
  Offset16To<Type>& get_offset (unsigned int i)
  { return (*this)[i].offset; }
  const Tag& get_tag (unsigned int i) const
  { return (*this)[i].tag; }
  unsigned int get_tags (unsigned int start_offset,
			 unsigned int *record_count /* IN/OUT */,
			 hb_tag_t     *record_tags /* OUT */) const
  {
    if (record_count)
    {
      + this->sub_array (start_offset, record_count)
      | hb_map (&Record<Type>::tag)
      | hb_sink (hb_array (record_tags, *record_count))
      ;
    }
    return this->len;
  }
  bool find_index (hb_tag_t tag, unsigned int *index) const
  {
    return this->bfind (tag, index, HB_NOT_FOUND_STORE, Index::NOT_FOUND_INDEX);
  }
};

template <typename Type>
struct RecordListOf : RecordArrayOf<Type>
{
  const Type& operator [] (unsigned int i) const
  { return this+this->get_offset (i); }

  bool subset (hb_subset_context_t *c,
	       hb_subset_layout_context_t *l) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->start_embed (*this);
    if (unlikely (!c->serializer->extend_min (out))) return_trace (false);

    + this->iter ()
    | hb_apply (subset_record_array (l, out, this))
    ;
    return_trace (true);
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (RecordArrayOf<Type>::sanitize (c, this));
  }
};

/* https://docs.microsoft.com/en-us/typography/opentype/spec/features_pt#size */
struct FeatureParamsSize
{
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!c->check_struct (this))) return_trace (false);

    /* This subtable has some "history", if you will.  Some earlier versions of
     * Adobe tools calculated the offset of the FeatureParams subtable from the
     * beginning of the FeatureList table!  Now, that is dealt with in the
     * Feature implementation.  But we still need to be able to tell junk from
     * real data.  Note: We don't check that the nameID actually exists.
     *
     * Read Roberts wrote on 9/15/06 on opentype-list@indx.co.uk :
     *
     * Yes, it is correct that a new version of the AFDKO (version 2.0) will be
     * coming out soon, and that the makeotf program will build a font with a
     * 'size' feature that is correct by the specification.
     *
     * The specification for this feature tag is in the "OpenType Layout Tag
     * Registry". You can see a copy of this at:
     * https://docs.microsoft.com/en-us/typography/opentype/spec/features_pt#tag-size
     *
     * Here is one set of rules to determine if the 'size' feature is built
     * correctly, or as by the older versions of MakeOTF. You may be able to do
     * better.
     *
     * Assume that the offset to the size feature is according to specification,
     * and make the following value checks. If it fails, assume the size
     * feature is calculated as versions of MakeOTF before the AFDKO 2.0 built it.
     * If this fails, reject the 'size' feature. The older makeOTF's calculated the
     * offset from the beginning of the FeatureList table, rather than from the
     * beginning of the 'size' Feature table.
     *
     * If "design size" == 0:
     *     fails check
     *
     * Else if ("subfamily identifier" == 0 and
     *     "range start" == 0 and
     *     "range end" == 0 and
     *     "range start" == 0 and
     *     "menu name ID" == 0)
     *     passes check: this is the format used when there is a design size
     * specified, but there is no recommended size range.
     *
     * Else if ("design size" <  "range start" or
     *     "design size" >   "range end" or
     *     "range end" <= "range start" or
     *     "menu name ID"  < 256 or
     *     "menu name ID"  > 32767 or
     *     menu name ID is not a name ID which is actually in the name table)
     *     fails test
     * Else
     *     passes test.
     */

    if (!designSize)
      return_trace (false);
    else if (subfamilyID == 0 &&
	     subfamilyNameID == 0 &&
	     rangeStart == 0 &&
	     rangeEnd == 0)
      return_trace (true);
    else if (designSize < rangeStart ||
	     designSize > rangeEnd ||
	     subfamilyNameID < 256 ||
	     subfamilyNameID > 32767)
      return_trace (false);
    else
      return_trace (true);
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    return_trace ((bool) c->serializer->embed (*this));
  }

  HBUINT16	designSize;	/* Represents the design size in 720/inch
				 * units (decipoints).  The design size entry
				 * must be non-zero.  When there is a design
				 * size but no recommended size range, the
				 * rest of the array will consist of zeros. */
  HBUINT16	subfamilyID;	/* Has no independent meaning, but serves
				 * as an identifier that associates fonts
				 * in a subfamily. All fonts which share a
				 * Preferred or Font Family name and which
				 * differ only by size range shall have the
				 * same subfamily value, and no fonts which
				 * differ in weight or style shall have the
				 * same subfamily value. If this value is
				 * zero, the remaining fields in the array
				 * will be ignored. */
  NameID	subfamilyNameID;/* If the preceding value is non-zero, this
				 * value must be set in the range 256 - 32767
				 * (inclusive). It records the value of a
				 * field in the name table, which must
				 * contain English-language strings encoded
				 * in Windows Unicode and Macintosh Roman,
				 * and may contain additional strings
				 * localized to other scripts and languages.
				 * Each of these strings is the name an
				 * application should use, in combination
				 * with the family name, to represent the
				 * subfamily in a menu.  Applications will
				 * choose the appropriate version based on
				 * their selection criteria. */
  HBUINT16	rangeStart;	/* Large end of the recommended usage range
				 * (inclusive), stored in 720/inch units
				 * (decipoints). */
  HBUINT16	rangeEnd;	/* Small end of the recommended usage range
				   (exclusive), stored in 720/inch units
				 * (decipoints). */
  public:
  DEFINE_SIZE_STATIC (10);
};

/* https://docs.microsoft.com/en-us/typography/opentype/spec/features_pt#ssxx */
struct FeatureParamsStylisticSet
{
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    /* Right now minorVersion is at zero.  Which means, any table supports
     * the uiNameID field. */
    return_trace (c->check_struct (this));
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    return_trace ((bool) c->serializer->embed (*this));
  }

  HBUINT16	version;	/* (set to 0): This corresponds to a “minor”
				 * version number. Additional data may be
				 * added to the end of this Feature Parameters
				 * table in the future. */

  NameID	uiNameID;	/* The 'name' table name ID that specifies a
				 * string (or strings, for multiple languages)
				 * for a user-interface label for this
				 * feature.  The values of uiLabelNameId and
				 * sampleTextNameId are expected to be in the
				 * font-specific name ID range (256-32767),
				 * though that is not a requirement in this
				 * Feature Parameters specification. The
				 * user-interface label for the feature can
				 * be provided in multiple languages. An
				 * English string should be included as a
				 * fallback. The string should be kept to a
				 * minimal length to fit comfortably with
				 * different application interfaces. */
  public:
  DEFINE_SIZE_STATIC (4);
};

/* https://docs.microsoft.com/en-us/typography/opentype/spec/features_ae#cv01-cv99 */
struct FeatureParamsCharacterVariants
{
  unsigned
  get_characters (unsigned start_offset, unsigned *char_count, hb_codepoint_t *chars) const
  {
    if (char_count)
    {
      + characters.sub_array (start_offset, char_count)
      | hb_sink (hb_array (chars, *char_count))
      ;
    }
    return characters.len;
  }

  unsigned get_size () const
  { return min_size + characters.len * HBUINT24::static_size; }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    return_trace ((bool) c->serializer->embed (*this));
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  characters.sanitize (c));
  }

  HBUINT16	format;			/* Format number is set to 0. */
  NameID	featUILableNameID;	/* The ‘name’ table name ID that
					 * specifies a string (or strings,
					 * for multiple languages) for a
					 * user-interface label for this
					 * feature. (May be NULL.) */
  NameID	featUITooltipTextNameID;/* The ‘name’ table name ID that
					 * specifies a string (or strings,
					 * for multiple languages) that an
					 * application can use for tooltip
					 * text for this feature. (May be
					 * nullptr.) */
  NameID	sampleTextNameID;	/* The ‘name’ table name ID that
					 * specifies sample text that
					 * illustrates the effect of this
					 * feature. (May be NULL.) */
  HBUINT16	numNamedParameters;	/* Number of named parameters. (May
					 * be zero.) */
  NameID	firstParamUILabelNameID;/* The first ‘name’ table name ID
					 * used to specify strings for
					 * user-interface labels for the
					 * feature parameters. (Must be zero
					 * if numParameters is zero.) */
  Array16Of<HBUINT24>
		characters;		/* Array of the Unicode Scalar Value
					 * of the characters for which this
					 * feature provides glyph variants.
					 * (May be zero.) */
  public:
  DEFINE_SIZE_ARRAY (14, characters);
};

struct FeatureParams
{
  bool sanitize (hb_sanitize_context_t *c, hb_tag_t tag) const
  {
#ifdef HB_NO_LAYOUT_FEATURE_PARAMS
    return true;
#endif
    TRACE_SANITIZE (this);
    if (tag == HB_TAG ('s','i','z','e'))
      return_trace (u.size.sanitize (c));
    if ((tag & 0xFFFF0000u) == HB_TAG ('s','s','\0','\0')) /* ssXX */
      return_trace (u.stylisticSet.sanitize (c));
    if ((tag & 0xFFFF0000u) == HB_TAG ('c','v','\0','\0')) /* cvXX */
      return_trace (u.characterVariants.sanitize (c));
    return_trace (true);
  }

  bool subset (hb_subset_context_t *c, const Tag* tag) const
  {
    TRACE_SUBSET (this);
    if (!tag) return_trace (false);
    if (*tag == HB_TAG ('s','i','z','e'))
      return_trace (u.size.subset (c));
    if ((*tag & 0xFFFF0000u) == HB_TAG ('s','s','\0','\0')) /* ssXX */
      return_trace (u.stylisticSet.subset (c));
    if ((*tag & 0xFFFF0000u) == HB_TAG ('c','v','\0','\0')) /* cvXX */
      return_trace (u.characterVariants.subset (c));
    return_trace (false);
  }

#ifndef HB_NO_LAYOUT_FEATURE_PARAMS
  const FeatureParamsSize& get_size_params (hb_tag_t tag) const
  {
    if (tag == HB_TAG ('s','i','z','e'))
      return u.size;
    return Null (FeatureParamsSize);
  }
  const FeatureParamsStylisticSet& get_stylistic_set_params (hb_tag_t tag) const
  {
    if ((tag & 0xFFFF0000u) == HB_TAG ('s','s','\0','\0')) /* ssXX */
      return u.stylisticSet;
    return Null (FeatureParamsStylisticSet);
  }
  const FeatureParamsCharacterVariants& get_character_variants_params (hb_tag_t tag) const
  {
    if ((tag & 0xFFFF0000u) == HB_TAG ('c','v','\0','\0')) /* cvXX */
      return u.characterVariants;
    return Null (FeatureParamsCharacterVariants);
  }
#endif

  private:
  union {
  FeatureParamsSize			size;
  FeatureParamsStylisticSet		stylisticSet;
  FeatureParamsCharacterVariants	characterVariants;
  } u;
  public:
  DEFINE_SIZE_MIN (0);
};


struct Feature
{
  unsigned int get_lookup_count () const
  { return lookupIndex.len; }
  hb_tag_t get_lookup_index (unsigned int i) const
  { return lookupIndex[i]; }
  unsigned int get_lookup_indexes (unsigned int start_index,
				   unsigned int *lookup_count /* IN/OUT */,
				   unsigned int *lookup_tags /* OUT */) const
  { return lookupIndex.get_indexes (start_index, lookup_count, lookup_tags); }
  void add_lookup_indexes_to (hb_set_t *lookup_indexes) const
  { lookupIndex.add_indexes_to (lookup_indexes); }

  const FeatureParams &get_feature_params () const
  { return this+featureParams; }

  bool intersects_lookup_indexes (const hb_map_t *lookup_indexes) const
  { return lookupIndex.intersects (lookup_indexes); }

  bool subset (hb_subset_context_t         *c,
	       hb_subset_layout_context_t  *l,
	       const Tag                   *tag = nullptr) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->start_embed (*this);
    if (unlikely (!out || !c->serializer->extend_min (out))) return_trace (false);

    out->featureParams.serialize_subset (c, featureParams, this, tag);

    auto it =
    + hb_iter (lookupIndex)
    | hb_filter (l->lookup_index_map)
    | hb_map (l->lookup_index_map)
    ;

    out->lookupIndex.serialize (c->serializer, l, it);
    // The decision to keep or drop this feature is already made before we get here
    // so always retain it.
    return_trace (true);
  }

  bool sanitize (hb_sanitize_context_t *c,
		 const Record_sanitize_closure_t *closure = nullptr) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!(c->check_struct (this) && lookupIndex.sanitize (c))))
      return_trace (false);

    /* Some earlier versions of Adobe tools calculated the offset of the
     * FeatureParams subtable from the beginning of the FeatureList table!
     *
     * If sanitizing "failed" for the FeatureParams subtable, try it with the
     * alternative location.  We would know sanitize "failed" if old value
     * of the offset was non-zero, but it's zeroed now.
     *
     * Only do this for the 'size' feature, since at the time of the faulty
     * Adobe tools, only the 'size' feature had FeatureParams defined.
     */

    if (likely (featureParams.is_null ()))
      return_trace (true);

    unsigned int orig_offset = featureParams;
    if (unlikely (!featureParams.sanitize (c, this, closure ? closure->tag : HB_TAG_NONE)))
      return_trace (false);

    if (featureParams == 0 && closure &&
	closure->tag == HB_TAG ('s','i','z','e') &&
	closure->list_base && closure->list_base < this)
    {
      unsigned int new_offset_int = orig_offset -
				    (((char *) this) - ((char *) closure->list_base));

      Offset16To<FeatureParams> new_offset;
      /* Check that it would not overflow. */
      new_offset = new_offset_int;
      if (new_offset == new_offset_int &&
	  c->try_set (&featureParams, new_offset_int) &&
	  !featureParams.sanitize (c, this, closure ? closure->tag : HB_TAG_NONE))
	return_trace (false);
    }

    return_trace (true);
  }

  Offset16To<FeatureParams>
		 featureParams;	/* Offset to Feature Parameters table (if one
				 * has been defined for the feature), relative
				 * to the beginning of the Feature Table; = Null
				 * if not required */
  IndexArray	 lookupIndex;	/* Array of LookupList indices */
  public:
  DEFINE_SIZE_ARRAY_SIZED (4, lookupIndex);
};

struct RecordListOfFeature : RecordListOf<Feature>
{
  bool subset (hb_subset_context_t *c,
	       hb_subset_layout_context_t *l) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->start_embed (*this);
    if (unlikely (!out || !c->serializer->extend_min (out))) return_trace (false);

    unsigned count = this->len;
    + hb_zip (*this, hb_range (count))
    | hb_filter (l->feature_index_map, hb_second)
    | hb_map (hb_first)
    | hb_apply (subset_record_array (l, out, this))
    ;
    return_trace (true);
  }
};

typedef RecordListOf<Feature> FeatureList;


struct LangSys
{
  unsigned int get_feature_count () const
  { return featureIndex.len; }
  hb_tag_t get_feature_index (unsigned int i) const
  { return featureIndex[i]; }
  unsigned int get_feature_indexes (unsigned int start_offset,
				    unsigned int *feature_count /* IN/OUT */,
				    unsigned int *feature_indexes /* OUT */) const
  { return featureIndex.get_indexes (start_offset, feature_count, feature_indexes); }
  void add_feature_indexes_to (hb_set_t *feature_indexes) const
  { featureIndex.add_indexes_to (feature_indexes); }

  bool has_required_feature () const { return reqFeatureIndex != 0xFFFFu; }
  unsigned int get_required_feature_index () const
  {
    if (reqFeatureIndex == 0xFFFFu)
      return Index::NOT_FOUND_INDEX;
   return reqFeatureIndex;
  }

  LangSys* copy (hb_serialize_context_t *c) const
  {
    TRACE_SERIALIZE (this);
    return_trace (c->embed (*this));
  }

  bool compare (const LangSys& o, const hb_map_t *feature_index_map) const
  {
    if (reqFeatureIndex != o.reqFeatureIndex)
      return false;

    auto iter =
    + hb_iter (featureIndex)
    | hb_filter (feature_index_map)
    | hb_map (feature_index_map)
    ;

    auto o_iter =
    + hb_iter (o.featureIndex)
    | hb_filter (feature_index_map)
    | hb_map (feature_index_map)
    ;

    for (; iter && o_iter; iter++, o_iter++)
    {
      unsigned a = *iter;
      unsigned b = *o_iter;
      if (a != b) return false;
    }

    if (iter || o_iter) return false;

    return true;
  }

  void collect_features (hb_prune_langsys_context_t *c) const
  {
    if (!has_required_feature () && !get_feature_count ()) return;
    if (has_required_feature () &&
        c->duplicate_feature_map->has (reqFeatureIndex))
      c->new_feature_indexes->add (get_required_feature_index ());

    + hb_iter (featureIndex)
    | hb_filter (c->duplicate_feature_map)
    | hb_sink (c->new_feature_indexes)
    ;
  }

  bool subset (hb_subset_context_t        *c,
	       hb_subset_layout_context_t *l,
	       const Tag                  *tag = nullptr) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->start_embed (*this);
    if (unlikely (!out || !c->serializer->extend_min (out))) return_trace (false);

    const unsigned *v;
    out->reqFeatureIndex = l->feature_index_map->has (reqFeatureIndex, &v) ? *v : 0xFFFFu;

    if (!l->visitFeatureIndex (featureIndex.len))
      return_trace (false);

    auto it =
    + hb_iter (featureIndex)
    | hb_filter (l->feature_index_map)
    | hb_map (l->feature_index_map)
    ;

    bool ret = bool (it);
    out->featureIndex.serialize (c->serializer, l, it);
    return_trace (ret);
  }

  bool sanitize (hb_sanitize_context_t *c,
		 const Record_sanitize_closure_t * = nullptr) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && featureIndex.sanitize (c));
  }

  Offset16	lookupOrderZ;	/* = Null (reserved for an offset to a
				 * reordering table) */
  HBUINT16	reqFeatureIndex;/* Index of a feature required for this
				 * language system--if no required features
				 * = 0xFFFFu */
  IndexArray	featureIndex;	/* Array of indices into the FeatureList */
  public:
  DEFINE_SIZE_ARRAY_SIZED (6, featureIndex);
};
DECLARE_NULL_NAMESPACE_BYTES (OT, LangSys);

struct Script
{
  unsigned int get_lang_sys_count () const
  { return langSys.len; }
  const Tag& get_lang_sys_tag (unsigned int i) const
  { return langSys.get_tag (i); }
  unsigned int get_lang_sys_tags (unsigned int start_offset,
				  unsigned int *lang_sys_count /* IN/OUT */,
				  hb_tag_t     *lang_sys_tags /* OUT */) const
  { return langSys.get_tags (start_offset, lang_sys_count, lang_sys_tags); }
  const LangSys& get_lang_sys (unsigned int i) const
  {
    if (i == Index::NOT_FOUND_INDEX) return get_default_lang_sys ();
    return this+langSys[i].offset;
  }
  bool find_lang_sys_index (hb_tag_t tag, unsigned int *index) const
  { return langSys.find_index (tag, index); }

  bool has_default_lang_sys () const           { return defaultLangSys != 0; }
  const LangSys& get_default_lang_sys () const { return this+defaultLangSys; }

  void prune_langsys (hb_prune_langsys_context_t *c,
                      unsigned script_index) const
  {
    if (!has_default_lang_sys () && !get_lang_sys_count ()) return;
    if (!c->visitScript ()) return;

    if (!c->script_langsys_map->has (script_index))
    {
      if (unlikely (!c->script_langsys_map->set (script_index, hb::unique_ptr<hb_set_t> {hb_set_create ()})))
	return;
    }

    unsigned langsys_count = get_lang_sys_count ();
    if (has_default_lang_sys ())
    {
      //only collect features from non-redundant langsys
      const LangSys& d = get_default_lang_sys ();
      if (c->visitLangsys (d.get_feature_count ())) {
        d.collect_features (c);
      }

      for (auto _ : + hb_zip (langSys, hb_range (langsys_count)))
      {
        const LangSys& l = this+_.first.offset;
        if (!c->visitLangsys (l.get_feature_count ())) continue;
        if (l.compare (d, c->duplicate_feature_map)) continue;

        l.collect_features (c);
        c->script_langsys_map->get (script_index)->add (_.second);
      }
    }
    else
    {
      for (auto _ : + hb_zip (langSys, hb_range (langsys_count)))
      {
        const LangSys& l = this+_.first.offset;
        if (!c->visitLangsys (l.get_feature_count ())) continue;
        l.collect_features (c);
        c->script_langsys_map->get (script_index)->add (_.second);
      }
    }
  }

  bool subset (hb_subset_context_t         *c,
	       hb_subset_layout_context_t  *l,
	       const Tag                   *tag) const
  {
    TRACE_SUBSET (this);
    if (!l->visitScript ()) return_trace (false);
    if (tag && !c->plan->layout_scripts->has (*tag))
      return false;

    auto *out = c->serializer->start_embed (*this);
    if (unlikely (!out || !c->serializer->extend_min (out))) return_trace (false);

    bool defaultLang = false;
    if (has_default_lang_sys ())
    {
      c->serializer->push ();
      const LangSys& ls = this+defaultLangSys;
      bool ret = ls.subset (c, l);
      if (!ret && tag && *tag != HB_TAG ('D', 'F', 'L', 'T'))
      {
	c->serializer->pop_discard ();
	out->defaultLangSys = 0;
      }
      else
      {
	c->serializer->add_link (out->defaultLangSys, c->serializer->pop_pack ());
	defaultLang = true;
      }
    }

    const hb_set_t *active_langsys = l->script_langsys_map->get (l->cur_script_index);
    if (active_langsys)
    {
      unsigned count = langSys.len;
      + hb_zip (langSys, hb_range (count))
      | hb_filter (active_langsys, hb_second)
      | hb_map (hb_first)
      | hb_filter ([=] (const Record<LangSys>& record) {return l->visitLangSys (); })
      | hb_apply (subset_record_array (l, &(out->langSys), this))
      ;
    }

    return_trace (bool (out->langSys.len) || defaultLang || l->table_tag == HB_OT_TAG_GSUB);
  }

  bool sanitize (hb_sanitize_context_t *c,
		 const Record_sanitize_closure_t * = nullptr) const
  {
    TRACE_SANITIZE (this);
    return_trace (defaultLangSys.sanitize (c, this) && langSys.sanitize (c, this));
  }

  protected:
  Offset16To<LangSys>
		defaultLangSys;	/* Offset to DefaultLangSys table--from
				 * beginning of Script table--may be Null */
  RecordArrayOf<LangSys>
		langSys;	/* Array of LangSysRecords--listed
				 * alphabetically by LangSysTag */
  public:
  DEFINE_SIZE_ARRAY_SIZED (4, langSys);
};

struct RecordListOfScript : RecordListOf<Script>
{
  bool subset (hb_subset_context_t *c,
               hb_subset_layout_context_t *l) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->start_embed (*this);
    if (unlikely (!out || !c->serializer->extend_min (out))) return_trace (false);

    unsigned count = this->len;
    for (auto _ : + hb_zip (*this, hb_range (count)))
    {
      auto snap = c->serializer->snapshot ();
      l->cur_script_index = _.second;
      bool ret = _.first.subset (l, this);
      if (!ret) c->serializer->revert (snap);
      else out->len++;
    }

    return_trace (true);
  }
};

typedef RecordListOfScript ScriptList;



struct LookupFlag : HBUINT16
{
  enum Flags {
    RightToLeft		= 0x0001u,
    IgnoreBaseGlyphs	= 0x0002u,
    IgnoreLigatures	= 0x0004u,
    IgnoreMarks		= 0x0008u,
    IgnoreFlags		= 0x000Eu,
    UseMarkFilteringSet	= 0x0010u,
    Reserved		= 0x00E0u,
    MarkAttachmentType	= 0xFF00u
  };
  public:
  DEFINE_SIZE_STATIC (2);
};

} /* namespace OT */
/* This has to be outside the namespace. */
HB_MARK_AS_FLAG_T (OT::LookupFlag::Flags);
namespace OT {

struct Lookup
{
  unsigned int get_subtable_count () const { return subTable.len; }

  template <typename TSubTable>
  const Array16OfOffset16To<TSubTable>& get_subtables () const
  { return reinterpret_cast<const Array16OfOffset16To<TSubTable> &> (subTable); }
  template <typename TSubTable>
  Array16OfOffset16To<TSubTable>& get_subtables ()
  { return reinterpret_cast<Array16OfOffset16To<TSubTable> &> (subTable); }

  template <typename TSubTable>
  const TSubTable& get_subtable (unsigned int i) const
  { return this+get_subtables<TSubTable> ()[i]; }
  template <typename TSubTable>
  TSubTable& get_subtable (unsigned int i)
  { return this+get_subtables<TSubTable> ()[i]; }

  unsigned int get_size () const
  {
    const HBUINT16 &markFilteringSet = StructAfter<const HBUINT16> (subTable);
    if (lookupFlag & LookupFlag::UseMarkFilteringSet)
      return (const char *) &StructAfter<const char> (markFilteringSet) - (const char *) this;
    return (const char *) &markFilteringSet - (const char *) this;
  }

  unsigned int get_type () const { return lookupType; }

  /* lookup_props is a 32-bit integer where the lower 16-bit is LookupFlag and
   * higher 16-bit is mark-filtering-set if the lookup uses one.
   * Not to be confused with glyph_props which is very similar. */
  uint32_t get_props () const
  {
    unsigned int flag = lookupFlag;
    if (unlikely (flag & LookupFlag::UseMarkFilteringSet))
    {
      const HBUINT16 &markFilteringSet = StructAfter<HBUINT16> (subTable);
      flag += (markFilteringSet << 16);
    }
    return flag;
  }

  template <typename TSubTable, typename context_t, typename ...Ts>
  typename context_t::return_t dispatch (context_t *c, Ts&&... ds) const
  {
    unsigned int lookup_type = get_type ();
    TRACE_DISPATCH (this, lookup_type);
    unsigned int count = get_subtable_count ();
    for (unsigned int i = 0; i < count; i++) {
      typename context_t::return_t r = get_subtable<TSubTable> (i).dispatch (c, lookup_type, std::forward<Ts> (ds)...);
      if (c->stop_sublookup_iteration (r))
	return_trace (r);
    }
    return_trace (c->default_return_value ());
  }

  bool serialize (hb_serialize_context_t *c,
		  unsigned int lookup_type,
		  uint32_t lookup_props,
		  unsigned int num_subtables)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (this))) return_trace (false);
    lookupType = lookup_type;
    lookupFlag = lookup_props & 0xFFFFu;
    if (unlikely (!subTable.serialize (c, num_subtables))) return_trace (false);
    if (lookupFlag & LookupFlag::UseMarkFilteringSet)
    {
      if (unlikely (!c->extend (this))) return_trace (false);
      HBUINT16 &markFilteringSet = StructAfter<HBUINT16> (subTable);
      markFilteringSet = lookup_props >> 16;
    }
    return_trace (true);
  }

  template <typename TSubTable>
  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->start_embed (*this);
    if (unlikely (!out || !c->serializer->extend_min (out))) return_trace (false);
    out->lookupType = lookupType;
    out->lookupFlag = lookupFlag;

    const hb_set_t *glyphset = c->plan->glyphset_gsub ();
    unsigned int lookup_type = get_type ();
    + hb_iter (get_subtables <TSubTable> ())
    | hb_filter ([this, glyphset, lookup_type] (const Offset16To<TSubTable> &_) { return (this+_).intersects (glyphset, lookup_type); })
    | hb_apply (subset_offset_array (c, out->get_subtables<TSubTable> (), this, lookup_type))
    ;

    if (lookupFlag & LookupFlag::UseMarkFilteringSet)
    {
      if (unlikely (!c->serializer->extend (out))) return_trace (false);
      const HBUINT16 &markFilteringSet = StructAfter<HBUINT16> (subTable);
      HBUINT16 &outMarkFilteringSet = StructAfter<HBUINT16> (out->subTable);
      outMarkFilteringSet = markFilteringSet;
    }

    return_trace (out->subTable.len);
  }

  template <typename TSubTable>
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (!(c->check_struct (this) && subTable.sanitize (c))) return_trace (false);

    unsigned subtables = get_subtable_count ();
    if (unlikely (!c->visit_subtables (subtables))) return_trace (false);

    if (lookupFlag & LookupFlag::UseMarkFilteringSet)
    {
      const HBUINT16 &markFilteringSet = StructAfter<HBUINT16> (subTable);
      if (!markFilteringSet.sanitize (c)) return_trace (false);
    }

    if (unlikely (!get_subtables<TSubTable> ().sanitize (c, this, get_type ())))
      return_trace (false);

    if (unlikely (get_type () == TSubTable::Extension && !c->get_edit_count ()))
    {
      /* The spec says all subtables of an Extension lookup should
       * have the same type, which shall not be the Extension type
       * itself (but we already checked for that).
       * This is specially important if one has a reverse type!
       *
       * We only do this if sanitizer edit_count is zero.  Otherwise,
       * some of the subtables might have become insane after they
       * were sanity-checked by the edits of subsequent subtables.
       * https://bugs.chromium.org/p/chromium/issues/detail?id=960331
       */
      unsigned int type = get_subtable<TSubTable> (0).u.extension.get_type ();
      for (unsigned int i = 1; i < subtables; i++)
	if (get_subtable<TSubTable> (i).u.extension.get_type () != type)
	  return_trace (false);
    }
    return_trace (true);
  }

  protected:
  HBUINT16	lookupType;		/* Different enumerations for GSUB and GPOS */
  HBUINT16	lookupFlag;		/* Lookup qualifiers */
  Array16Of<Offset16>
		subTable;		/* Array of SubTables */
/*HBUINT16	markFilteringSetX[HB_VAR_ARRAY];*//* Index (base 0) into GDEF mark glyph sets
					 * structure. This field is only present if bit
					 * UseMarkFilteringSet of lookup flags is set. */
  public:
  DEFINE_SIZE_ARRAY (6, subTable);
};

template <typename Types>
using LookupList = List16OfOffsetTo<Lookup, typename Types::HBUINT>;

template <typename TLookup, typename OffsetType>
struct LookupOffsetList : List16OfOffsetTo<TLookup, OffsetType>
{
  bool subset (hb_subset_context_t        *c,
	       hb_subset_layout_context_t *l) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->start_embed (this);
    if (unlikely (!out || !c->serializer->extend_min (out))) return_trace (false);

    unsigned count = this->len;
    + hb_zip (*this, hb_range (count))
    | hb_filter (l->lookup_index_map, hb_second)
    | hb_map (hb_first)
    | hb_apply (subset_offset_array (c, *out, this))
    ;
    return_trace (true);
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (List16OfOffset16To<TLookup>::sanitize (c, this));
  }
};


/*
 * Coverage Table
 */


static void ClassDef_remap_and_serialize (hb_serialize_context_t *c,
					  const hb_set_t &klasses,
                                          bool use_class_zero,
                                          hb_sorted_vector_t<hb_pair_t<hb_codepoint_t, hb_codepoint_t>> &glyph_and_klass, /* IN/OUT */
					  hb_map_t *klass_map /*IN/OUT*/)
{
  if (!klass_map)
  {
    ClassDef_serialize (c, glyph_and_klass.iter ());
    return;
  }

  /* any glyph not assigned a class value falls into Class zero (0),
   * if any glyph assigned to class 0, remapping must start with 0->0*/
  if (!use_class_zero)
    klass_map->set (0, 0);

  unsigned idx = klass_map->has (0) ? 1 : 0;
  for (const unsigned k: klasses.iter ())
  {
    if (klass_map->has (k)) continue;
    klass_map->set (k, idx);
    idx++;
  }


  for (unsigned i = 0; i < glyph_and_klass.length; i++)
  {
    hb_codepoint_t klass = glyph_and_klass[i].second;
    glyph_and_klass[i].second = klass_map->get (klass);
  }

  c->propagate_error (glyph_and_klass, klasses);
  ClassDef_serialize (c, glyph_and_klass.iter ());
}

/*
 * Class Definition Table
 */

template <typename Types>
struct ClassDefFormat1_3
{
  friend struct ClassDef;

  private:
  unsigned int get_class (hb_codepoint_t glyph_id) const
  {
    return classValue[(unsigned int) (glyph_id - startGlyph)];
  }

  template<typename Iterator,
	   hb_requires (hb_is_sorted_source_of (Iterator, hb_codepoint_t))>
  bool serialize (hb_serialize_context_t *c,
		  Iterator it)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (this))) return_trace (false);

    if (unlikely (!it))
    {
      classFormat = 1;
      startGlyph = 0;
      classValue.len = 0;
      return_trace (true);
    }

    hb_codepoint_t glyph_min = (*it).first;
    hb_codepoint_t glyph_max = + it
			       | hb_map (hb_first)
			       | hb_reduce (hb_max, 0u);
    unsigned glyph_count = glyph_max - glyph_min + 1;

    startGlyph = glyph_min;
    if (unlikely (!classValue.serialize (c, glyph_count))) return_trace (false);
    for (const hb_pair_t<hb_codepoint_t, unsigned> gid_klass_pair : + it)
    {
      unsigned idx = gid_klass_pair.first - glyph_min;
      classValue[idx] = gid_klass_pair.second;
    }
    return_trace (true);
  }

  bool subset (hb_subset_context_t *c,
	       hb_map_t *klass_map = nullptr /*OUT*/,
               bool keep_empty_table = true,
               bool use_class_zero = true,
               const Coverage* glyph_filter = nullptr) const
  {
    TRACE_SUBSET (this);
    const hb_map_t &glyph_map = *c->plan->glyph_map_gsub;

    hb_sorted_vector_t<hb_pair_t<hb_codepoint_t, hb_codepoint_t>> glyph_and_klass;
    hb_set_t orig_klasses;

    hb_codepoint_t start = startGlyph;
    hb_codepoint_t end   = start + classValue.len;

    for (const hb_codepoint_t gid : + hb_range (start, end))
    {
      hb_codepoint_t new_gid = glyph_map[gid];
      if (new_gid == HB_MAP_VALUE_INVALID) continue;
      if (glyph_filter && !glyph_filter->has(gid)) continue;

      unsigned klass = classValue[gid - start];
      if (!klass) continue;

      glyph_and_klass.push (hb_pair (new_gid, klass));
      orig_klasses.add (klass);
    }

    unsigned glyph_count = glyph_filter
                           ? hb_len (hb_iter (glyph_map.keys()) | hb_filter (glyph_filter))
                           : glyph_map.get_population ();
    use_class_zero = use_class_zero && glyph_count <= glyph_and_klass.length;
    ClassDef_remap_and_serialize (c->serializer,
                                  orig_klasses,
                                  use_class_zero,
                                  glyph_and_klass,
                                  klass_map);
    return_trace (keep_empty_table || (bool) glyph_and_klass);
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && classValue.sanitize (c));
  }

  unsigned cost () const { return 1; }

  template <typename set_t>
  bool collect_coverage (set_t *glyphs) const
  {
    unsigned int start = 0;
    unsigned int count = classValue.len;
    for (unsigned int i = 0; i < count; i++)
    {
      if (classValue[i])
	continue;

      if (start != i)
	if (unlikely (!glyphs->add_range (startGlyph + start, startGlyph + i)))
	  return false;

      start = i + 1;
    }
    if (start != count)
      if (unlikely (!glyphs->add_range (startGlyph + start, startGlyph + count)))
	return false;

    return true;
  }

  template <typename set_t>
  bool collect_class (set_t *glyphs, unsigned klass) const
  {
    unsigned int count = classValue.len;
    for (unsigned int i = 0; i < count; i++)
      if (classValue[i] == klass) glyphs->add (startGlyph + i);
    return true;
  }

  bool intersects (const hb_set_t *glyphs) const
  {
    /* TODO Speed up, using hb_set_next()? */
    hb_codepoint_t start = startGlyph;
    hb_codepoint_t end = startGlyph + classValue.len;
    for (hb_codepoint_t iter = startGlyph - 1;
	 hb_set_next (glyphs, &iter) && iter < end;)
      if (classValue[iter - start]) return true;
    return false;
  }
  bool intersects_class (const hb_set_t *glyphs, uint16_t klass) const
  {
    unsigned int count = classValue.len;
    if (klass == 0)
    {
      /* Match if there's any glyph that is not listed! */
      hb_codepoint_t g = HB_SET_VALUE_INVALID;
      if (!hb_set_next (glyphs, &g)) return false;
      if (g < startGlyph) return true;
      g = startGlyph + count - 1;
      if (hb_set_next (glyphs, &g)) return true;
      /* Fall through. */
    }
    /* TODO Speed up, using set overlap first? */
    /* TODO(iter) Rewrite as dagger. */
    const HBUINT16 *arr = classValue.arrayZ;
    for (unsigned int i = 0; i < count; i++)
      if (arr[i] == klass && glyphs->has (startGlyph + i))
	return true;
    return false;
  }

  void intersected_class_glyphs (const hb_set_t *glyphs, unsigned klass, hb_set_t *intersect_glyphs) const
  {
    unsigned count = classValue.len;
    if (klass == 0)
    {
      unsigned start_glyph = startGlyph;
      for (unsigned g = HB_SET_VALUE_INVALID;
	   hb_set_next (glyphs, &g) && g < start_glyph;)
	intersect_glyphs->add (g);

      for (unsigned g = startGlyph + count - 1;
	   hb_set_next (glyphs, &g);)
	intersect_glyphs->add (g);

      return;
    }

    for (unsigned i = 0; i < count; i++)
      if (classValue[i] == klass && glyphs->has (startGlyph + i))
	intersect_glyphs->add (startGlyph + i);

#if 0
    /* The following implementation is faster asymptotically, but slower
     * in practice. */
    unsigned start_glyph = startGlyph;
    unsigned end_glyph = start_glyph + count;
    for (unsigned g = startGlyph - 1;
	 hb_set_next (glyphs, &g) && g < end_glyph;)
      if (classValue.arrayZ[g - start_glyph] == klass)
        intersect_glyphs->add (g);
#endif
  }

  void intersected_classes (const hb_set_t *glyphs, hb_set_t *intersect_classes) const
  {
    if (glyphs->is_empty ()) return;
    hb_codepoint_t end_glyph = startGlyph + classValue.len - 1;
    if (glyphs->get_min () < startGlyph ||
        glyphs->get_max () > end_glyph)
      intersect_classes->add (0);

    for (const auto& _ : + hb_enumerate (classValue))
    {
      hb_codepoint_t g = startGlyph + _.first;
      if (glyphs->has (g))
        intersect_classes->add (_.second);
    }
  }

  protected:
  HBUINT16	classFormat;	/* Format identifier--format = 1 */
  typename Types::HBGlyphID
		 startGlyph;	/* First GlyphID of the classValueArray */
  typename Types::template ArrayOf<HBUINT16>
		classValue;	/* Array of Class Values--one per GlyphID */
  public:
  DEFINE_SIZE_ARRAY (2 + 2 * Types::size, classValue);
};

template <typename Types>
struct ClassDefFormat2_4
{
  friend struct ClassDef;

  private:
  unsigned int get_class (hb_codepoint_t glyph_id) const
  {
    return rangeRecord.bsearch (glyph_id).value;
  }

  template<typename Iterator,
	   hb_requires (hb_is_sorted_source_of (Iterator, hb_codepoint_t))>
  bool serialize (hb_serialize_context_t *c,
		  Iterator it)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (this))) return_trace (false);

    if (unlikely (!it))
    {
      classFormat = 2;
      rangeRecord.len = 0;
      return_trace (true);
    }

    unsigned num_ranges = 1;
    hb_codepoint_t prev_gid = (*it).first;
    unsigned prev_klass = (*it).second;

    RangeRecord<Types> range_rec;
    range_rec.first = prev_gid;
    range_rec.last = prev_gid;
    range_rec.value = prev_klass;

    auto *record = c->copy (range_rec);
    if (unlikely (!record)) return_trace (false);

    for (const auto gid_klass_pair : + (++it))
    {
      hb_codepoint_t cur_gid = gid_klass_pair.first;
      unsigned cur_klass = gid_klass_pair.second;

      if (cur_gid != prev_gid + 1 ||
	  cur_klass != prev_klass)
      {
	if (unlikely (!record)) break;
	record->last = prev_gid;
	num_ranges++;

	range_rec.first = cur_gid;
	range_rec.last = cur_gid;
	range_rec.value = cur_klass;

	record = c->copy (range_rec);
      }

      prev_klass = cur_klass;
      prev_gid = cur_gid;
    }

    if (likely (record)) record->last = prev_gid;
    rangeRecord.len = num_ranges;
    return_trace (true);
  }

  bool subset (hb_subset_context_t *c,
	       hb_map_t *klass_map = nullptr /*OUT*/,
               bool keep_empty_table = true,
               bool use_class_zero = true,
               const Coverage* glyph_filter = nullptr) const
  {
    TRACE_SUBSET (this);
    const hb_map_t &glyph_map = *c->plan->glyph_map_gsub;

    hb_sorted_vector_t<hb_pair_t<hb_codepoint_t, hb_codepoint_t>> glyph_and_klass;
    hb_set_t orig_klasses;

    unsigned num_source_glyphs = c->plan->source->get_num_glyphs ();
    unsigned count = rangeRecord.len;
    for (unsigned i = 0; i < count; i++)
    {
      unsigned klass = rangeRecord[i].value;
      if (!klass) continue;
      hb_codepoint_t start = rangeRecord[i].first;
      hb_codepoint_t end   = hb_min (rangeRecord[i].last + 1, num_source_glyphs);
      for (hb_codepoint_t g = start; g < end; g++)
      {
        hb_codepoint_t new_gid = glyph_map[g];
	if (new_gid == HB_MAP_VALUE_INVALID) continue;
        if (glyph_filter && !glyph_filter->has (g)) continue;

	glyph_and_klass.push (hb_pair (new_gid, klass));
	orig_klasses.add (klass);
      }
    }

    const hb_set_t& glyphset = *c->plan->glyphset_gsub ();
    unsigned glyph_count = glyph_filter
                           ? hb_len (hb_iter (glyphset) | hb_filter (glyph_filter))
                           : glyph_map.get_population ();
    use_class_zero = use_class_zero && glyph_count <= glyph_and_klass.length;
    ClassDef_remap_and_serialize (c->serializer,
                                  orig_klasses,
                                  use_class_zero,
                                  glyph_and_klass,
                                  klass_map);
    return_trace (keep_empty_table || (bool) glyph_and_klass);
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (rangeRecord.sanitize (c));
  }

  unsigned cost () const { return hb_bit_storage ((unsigned) rangeRecord.len); /* bsearch cost */ }

  template <typename set_t>
  bool collect_coverage (set_t *glyphs) const
  {
    unsigned int count = rangeRecord.len;
    for (unsigned int i = 0; i < count; i++)
      if (rangeRecord[i].value)
	if (unlikely (!rangeRecord[i].collect_coverage (glyphs)))
	  return false;
    return true;
  }

  template <typename set_t>
  bool collect_class (set_t *glyphs, unsigned int klass) const
  {
    unsigned int count = rangeRecord.len;
    for (unsigned int i = 0; i < count; i++)
    {
      if (rangeRecord[i].value == klass)
	if (unlikely (!rangeRecord[i].collect_coverage (glyphs)))
	  return false;
    }
    return true;
  }

  bool intersects (const hb_set_t *glyphs) const
  {
    /* TODO Speed up, using hb_set_next() and bsearch()? */
    unsigned int count = rangeRecord.len;
    for (unsigned int i = 0; i < count; i++)
    {
      const auto& range = rangeRecord[i];
      if (range.intersects (*glyphs) && range.value)
	return true;
    }
    return false;
  }
  bool intersects_class (const hb_set_t *glyphs, uint16_t klass) const
  {
    unsigned int count = rangeRecord.len;
    if (klass == 0)
    {
      /* Match if there's any glyph that is not listed! */
      hb_codepoint_t g = HB_SET_VALUE_INVALID;
      for (unsigned int i = 0; i < count; i++)
      {
	if (!hb_set_next (glyphs, &g))
	  break;
	if (g < rangeRecord[i].first)
	  return true;
	g = rangeRecord[i].last;
      }
      if (g != HB_SET_VALUE_INVALID && hb_set_next (glyphs, &g))
	return true;
      /* Fall through. */
    }
    for (const auto &range : rangeRecord)
      if (range.value == klass && range.intersects (*glyphs))
	return true;
    return false;
  }

  void intersected_class_glyphs (const hb_set_t *glyphs, unsigned klass, hb_set_t *intersect_glyphs) const
  {
    unsigned count = rangeRecord.len;
    if (klass == 0)
    {
      hb_codepoint_t g = HB_SET_VALUE_INVALID;
      for (unsigned int i = 0; i < count; i++)
      {
	if (!hb_set_next (glyphs, &g))
	  goto done;
	while (g < rangeRecord[i].first)
	{
	  intersect_glyphs->add (g);
	  if (!hb_set_next (glyphs, &g))
	    goto done;
        }
        g = rangeRecord[i].last;
      }
      while (hb_set_next (glyphs, &g))
	intersect_glyphs->add (g);
      done:

      return;
    }

#if 0
    /* The following implementation is faster asymptotically, but slower
     * in practice. */
    if ((count >> 3) > glyphs->get_population ())
    {
      for (hb_codepoint_t g = HB_SET_VALUE_INVALID;
	   hb_set_next (glyphs, &g);)
        if (rangeRecord.as_array ().bfind (g))
	  intersect_glyphs->add (g);
      return;
    }
#endif

    for (unsigned int i = 0; i < count; i++)
    {
      if (rangeRecord[i].value != klass) continue;

      unsigned end = rangeRecord[i].last + 1;
      for (hb_codepoint_t g = rangeRecord[i].first - 1;
	   hb_set_next (glyphs, &g) && g < end;)
	intersect_glyphs->add (g);
    }
  }

  void intersected_classes (const hb_set_t *glyphs, hb_set_t *intersect_classes) const
  {
    if (glyphs->is_empty ()) return;

    unsigned count = rangeRecord.len;
    hb_codepoint_t g = HB_SET_VALUE_INVALID;
    for (unsigned int i = 0; i < count; i++)
    {
      if (!hb_set_next (glyphs, &g))
        break;
      if (g < rangeRecord[i].first)
      {
        intersect_classes->add (0);
        break;
      }
      g = rangeRecord[i].last;
    }
    if (g != HB_SET_VALUE_INVALID && hb_set_next (glyphs, &g))
      intersect_classes->add (0);

    for (const auto& record : rangeRecord.iter ())
      if (record.intersects (*glyphs))
        intersect_classes->add (record.value);
  }

  protected:
  HBUINT16	classFormat;	/* Format identifier--format = 2 */
  typename Types::template SortedArrayOf<RangeRecord<Types>>
		rangeRecord;	/* Array of glyph ranges--ordered by
				 * Start GlyphID */
  public:
  DEFINE_SIZE_ARRAY (2 + Types::size, rangeRecord);
};

struct ClassDef
{
  /* Has interface. */
  static constexpr unsigned SENTINEL = 0;
  typedef unsigned int value_t;
  value_t operator [] (hb_codepoint_t k) const { return get (k); }
  bool has (hb_codepoint_t k) const { return (*this)[k] != SENTINEL; }
  /* Projection. */
  hb_codepoint_t operator () (hb_codepoint_t k) const { return get (k); }

  unsigned int get (hb_codepoint_t k) const { return get_class (k); }
  unsigned int get_class (hb_codepoint_t glyph_id) const
  {
    switch (u.format) {
    case 1: return u.format1.get_class (glyph_id);
    case 2: return u.format2.get_class (glyph_id);
#ifndef HB_NO_BEYOND_64K
    case 3: return u.format3.get_class (glyph_id);
    case 4: return u.format4.get_class (glyph_id);
#endif
    default:return 0;
    }
  }

  template<typename Iterator,
	   hb_requires (hb_is_sorted_source_of (Iterator, hb_codepoint_t))>
  bool serialize (hb_serialize_context_t *c, Iterator it_with_class_zero)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (this))) return_trace (false);

    auto it = + it_with_class_zero | hb_filter (hb_second);

    unsigned format = 2;
    hb_codepoint_t glyph_max = 0;
    if (likely (it))
    {
      hb_codepoint_t glyph_min = (*it).first;
      glyph_max = glyph_min;

      unsigned num_glyphs = 0;
      unsigned num_ranges = 1;
      hb_codepoint_t prev_gid = glyph_min;
      unsigned prev_klass = (*it).second;

      for (const auto gid_klass_pair : it)
      {
	hb_codepoint_t cur_gid = gid_klass_pair.first;
	unsigned cur_klass = gid_klass_pair.second;
        num_glyphs++;
	if (cur_gid == glyph_min) continue;
        if (cur_gid > glyph_max) glyph_max = cur_gid;
	if (cur_gid != prev_gid + 1 ||
	    cur_klass != prev_klass)
	  num_ranges++;

	prev_gid = cur_gid;
	prev_klass = cur_klass;
      }

      if (num_glyphs && 1 + (glyph_max - glyph_min + 1) <= num_ranges * 3)
	format = 1;
    }

#ifndef HB_NO_BEYOND_64K
    if (glyph_max > 0xFFFFu)
      format += 2;
#endif

    u.format = format;

    switch (u.format)
    {
    case 1: return_trace (u.format1.serialize (c, it));
    case 2: return_trace (u.format2.serialize (c, it));
#ifndef HB_NO_BEYOND_64K
    case 3: return_trace (u.format3.serialize (c, it));
    case 4: return_trace (u.format4.serialize (c, it));
#endif
    default:return_trace (false);
    }
  }

  bool subset (hb_subset_context_t *c,
	       hb_map_t *klass_map = nullptr /*OUT*/,
               bool keep_empty_table = true,
               bool use_class_zero = true,
               const Coverage* glyph_filter = nullptr) const
  {
    TRACE_SUBSET (this);
    switch (u.format) {
    case 1: return_trace (u.format1.subset (c, klass_map, keep_empty_table, use_class_zero, glyph_filter));
    case 2: return_trace (u.format2.subset (c, klass_map, keep_empty_table, use_class_zero, glyph_filter));
#ifndef HB_NO_BEYOND_64K
    case 3: return_trace (u.format3.subset (c, klass_map, keep_empty_table, use_class_zero, glyph_filter));
    case 4: return_trace (u.format4.subset (c, klass_map, keep_empty_table, use_class_zero, glyph_filter));
#endif
    default:return_trace (false);
    }
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (!u.format.sanitize (c)) return_trace (false);
    switch (u.format) {
    case 1: return_trace (u.format1.sanitize (c));
    case 2: return_trace (u.format2.sanitize (c));
#ifndef HB_NO_BEYOND_64K
    case 3: return_trace (u.format3.sanitize (c));
    case 4: return_trace (u.format4.sanitize (c));
#endif
    default:return_trace (true);
    }
  }

  unsigned cost () const
  {
    switch (u.format) {
    case 1: return u.format1.cost ();
    case 2: return u.format2.cost ();
#ifndef HB_NO_BEYOND_64K
    case 3: return u.format3.cost ();
    case 4: return u.format4.cost ();
#endif
    default:return 0u;
    }
  }

  /* Might return false if array looks unsorted.
   * Used for faster rejection of corrupt data. */
  template <typename set_t>
  bool collect_coverage (set_t *glyphs) const
  {
    switch (u.format) {
    case 1: return u.format1.collect_coverage (glyphs);
    case 2: return u.format2.collect_coverage (glyphs);
#ifndef HB_NO_BEYOND_64K
    case 3: return u.format3.collect_coverage (glyphs);
    case 4: return u.format4.collect_coverage (glyphs);
#endif
    default:return false;
    }
  }

  /* Might return false if array looks unsorted.
   * Used for faster rejection of corrupt data. */
  template <typename set_t>
  bool collect_class (set_t *glyphs, unsigned int klass) const
  {
    switch (u.format) {
    case 1: return u.format1.collect_class (glyphs, klass);
    case 2: return u.format2.collect_class (glyphs, klass);
#ifndef HB_NO_BEYOND_64K
    case 3: return u.format3.collect_class (glyphs, klass);
    case 4: return u.format4.collect_class (glyphs, klass);
#endif
    default:return false;
    }
  }

  bool intersects (const hb_set_t *glyphs) const
  {
    switch (u.format) {
    case 1: return u.format1.intersects (glyphs);
    case 2: return u.format2.intersects (glyphs);
#ifndef HB_NO_BEYOND_64K
    case 3: return u.format3.intersects (glyphs);
    case 4: return u.format4.intersects (glyphs);
#endif
    default:return false;
    }
  }
  bool intersects_class (const hb_set_t *glyphs, unsigned int klass) const
  {
    switch (u.format) {
    case 1: return u.format1.intersects_class (glyphs, klass);
    case 2: return u.format2.intersects_class (glyphs, klass);
#ifndef HB_NO_BEYOND_64K
    case 3: return u.format3.intersects_class (glyphs, klass);
    case 4: return u.format4.intersects_class (glyphs, klass);
#endif
    default:return false;
    }
  }

  void intersected_class_glyphs (const hb_set_t *glyphs, unsigned klass, hb_set_t *intersect_glyphs) const
  {
    switch (u.format) {
    case 1: return u.format1.intersected_class_glyphs (glyphs, klass, intersect_glyphs);
    case 2: return u.format2.intersected_class_glyphs (glyphs, klass, intersect_glyphs);
#ifndef HB_NO_BEYOND_64K
    case 3: return u.format3.intersected_class_glyphs (glyphs, klass, intersect_glyphs);
    case 4: return u.format4.intersected_class_glyphs (glyphs, klass, intersect_glyphs);
#endif
    default:return;
    }
  }

  void intersected_classes (const hb_set_t *glyphs, hb_set_t *intersect_classes) const
  {
    switch (u.format) {
    case 1: return u.format1.intersected_classes (glyphs, intersect_classes);
    case 2: return u.format2.intersected_classes (glyphs, intersect_classes);
#ifndef HB_NO_BEYOND_64K
    case 3: return u.format3.intersected_classes (glyphs, intersect_classes);
    case 4: return u.format4.intersected_classes (glyphs, intersect_classes);
#endif
    default:return;
    }
  }


  protected:
  union {
  HBUINT16			format;		/* Format identifier */
  ClassDefFormat1_3<SmallTypes>	format1;
  ClassDefFormat2_4<SmallTypes>	format2;
#ifndef HB_NO_BEYOND_64K
  ClassDefFormat1_3<MediumTypes>format3;
  ClassDefFormat2_4<MediumTypes>format4;
#endif
  } u;
  public:
  DEFINE_SIZE_UNION (2, format);
};

template<typename Iterator>
static inline void ClassDef_serialize (hb_serialize_context_t *c,
				       Iterator it)
{ c->start_embed<ClassDef> ()->serialize (c, it); }


/*
 * Item Variation Store
 */

struct VarRegionAxis
{
  float evaluate (int coord) const
  {
    int start = startCoord, peak = peakCoord, end = endCoord;

    /* TODO Move these to sanitize(). */
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
      return float (coord - start) / (peak - start);
    else
      return float (end - coord) / (end - peak);
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
    /* TODO Handle invalid start/peak/end configs, so we don't
     * have to do that at runtime. */
  }

  public:
  F2DOT14	startCoord;
  F2DOT14	peakCoord;
  F2DOT14	endCoord;
  public:
  DEFINE_SIZE_STATIC (6);
};

#define REGION_CACHE_ITEM_CACHE_INVALID 2.f

struct VarRegionList
{
  using cache_t = float;

  float evaluate (unsigned int region_index,
		  const int *coords, unsigned int coord_len,
		  cache_t *cache = nullptr) const
  {
    if (unlikely (region_index >= regionCount))
      return 0.;

    float *cached_value = nullptr;
    if (cache)
    {
      cached_value = &(cache[region_index]);
      if (likely (*cached_value != REGION_CACHE_ITEM_CACHE_INVALID))
	return *cached_value;
    }

    const VarRegionAxis *axes = axesZ.arrayZ + (region_index * axisCount);

    float v = 1.;
    unsigned int count = axisCount;
    for (unsigned int i = 0; i < count; i++)
    {
      int coord = i < coord_len ? coords[i] : 0;
      float factor = axes[i].evaluate (coord);
      if (factor == 0.f)
      {
        if (cache)
	  *cached_value = 0.;
	return 0.;
      }
      v *= factor;
    }

    if (cache)
      *cached_value = v;
    return v;
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && axesZ.sanitize (c, axisCount * regionCount));
  }

  bool serialize (hb_serialize_context_t *c, const VarRegionList *src, const hb_bimap_t &region_map)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (this))) return_trace (false);
    axisCount = src->axisCount;
    regionCount = region_map.get_population ();
    if (unlikely (hb_unsigned_mul_overflows (axisCount * regionCount,
					     VarRegionAxis::static_size))) return_trace (false);
    if (unlikely (!c->extend (this))) return_trace (false);
    unsigned int region_count = src->regionCount;
    for (unsigned int r = 0; r < regionCount; r++)
    {
      unsigned int backward = region_map.backward (r);
      if (backward >= region_count) return_trace (false);
      memcpy (&axesZ[axisCount * r], &src->axesZ[axisCount * backward], VarRegionAxis::static_size * axisCount);
    }

    return_trace (true);
  }

  unsigned int get_size () const { return min_size + VarRegionAxis::static_size * axisCount * regionCount; }

  public:
  HBUINT16	axisCount;
  HBUINT15	regionCount;
  protected:
  UnsizedArrayOf<VarRegionAxis>
		axesZ;
  public:
  DEFINE_SIZE_ARRAY (4, axesZ);
};

struct VarData
{
  unsigned int get_region_index_count () const
  { return regionIndices.len; }

  unsigned int get_row_size () const
  { return (wordCount () + regionIndices.len) * (longWords () ? 2 : 1); }

  unsigned int get_size () const
  { return min_size
	 - regionIndices.min_size + regionIndices.get_size ()
	 + itemCount * get_row_size ();
  }

  float get_delta (unsigned int inner,
		   const int *coords, unsigned int coord_count,
		   const VarRegionList &regions,
		   VarRegionList::cache_t *cache = nullptr) const
  {
    if (unlikely (inner >= itemCount))
      return 0.;

   unsigned int count = regionIndices.len;
   bool is_long = longWords ();
   unsigned word_count = wordCount ();
   unsigned int scount = is_long ? count - word_count : word_count;
   unsigned int lcount = is_long ? word_count : 0;

   const HBUINT8 *bytes = get_delta_bytes ();
   const HBUINT8 *row = bytes + inner * (scount + count);

   float delta = 0.;
   unsigned int i = 0;

   const HBINT16 *lcursor = reinterpret_cast<const HBINT16 *> (row);
   for (; i < lcount; i++)
   {
     float scalar = regions.evaluate (regionIndices.arrayZ[i], coords, coord_count, cache);
     delta += scalar * *lcursor++;
   }
   const HBINT16 *scursor = reinterpret_cast<const HBINT16 *> (lcursor);
   for (; i < scount; i++)
   {
     float scalar = regions.evaluate (regionIndices.arrayZ[i], coords, coord_count, cache);
     delta += scalar * *scursor++;
   }
   const HBINT8 *bcursor = reinterpret_cast<const HBINT8 *> (scursor);
   for (; i < count; i++)
   {
     float scalar = regions.evaluate (regionIndices.arrayZ[i], coords, coord_count, cache);
     delta += scalar * *bcursor++;
   }

   return delta;
  }

  void get_region_scalars (const int *coords, unsigned int coord_count,
			   const VarRegionList &regions,
			   float *scalars /*OUT */,
			   unsigned int num_scalars) const
  {
    unsigned count = hb_min (num_scalars, regionIndices.len);
    for (unsigned int i = 0; i < count; i++)
      scalars[i] = regions.evaluate (regionIndices.arrayZ[i], coords, coord_count);
    for (unsigned int i = count; i < num_scalars; i++)
      scalars[i] = 0.f;
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  regionIndices.sanitize (c) &&
		  wordCount () <= regionIndices.len &&
		  c->check_range (get_delta_bytes (),
				  itemCount,
				  get_row_size ()));
  }

  bool serialize (hb_serialize_context_t *c,
		  const VarData *src,
		  const hb_inc_bimap_t &inner_map,
		  const hb_bimap_t &region_map)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (this))) return_trace (false);
    itemCount = inner_map.get_next_value ();

    /* Optimize word count */
    unsigned ri_count = src->regionIndices.len;
    enum delta_size_t { kZero=0, kNonWord, kWord };
    hb_vector_t<delta_size_t> delta_sz;
    hb_vector_t<unsigned int> ri_map;	/* maps old index to new index */
    delta_sz.resize (ri_count);
    ri_map.resize (ri_count);
    unsigned int new_word_count = 0;
    unsigned int r;

    bool has_long = false;
    if (src->longWords ())
    {
      for (r = 0; r < ri_count; r++)
      {
	for (unsigned int i = 0; i < inner_map.get_next_value (); i++)
	{
	  unsigned int old = inner_map.backward (i);
	  int32_t delta = src->get_item_delta (old, r);
	  if (delta < -65536 || 65535 < delta)
	  {
	    has_long = true;
	    break;
	  }
        }
      }
    }

    signed min_threshold = has_long ? -65536 : -128;
    signed max_threshold = has_long ? +65535 : +127;
    for (r = 0; r < ri_count; r++)
    {
      delta_sz[r] = kZero;
      for (unsigned int i = 0; i < inner_map.get_next_value (); i++)
      {
	unsigned int old = inner_map.backward (i);
	int32_t delta = src->get_item_delta (old, r);
	if (delta < min_threshold || max_threshold < delta)
	{
	  delta_sz[r] = kWord;
	  new_word_count++;
	  break;
	}
	else if (delta != 0)
	  delta_sz[r] = kNonWord;
      }
    }

    unsigned int word_index = 0;
    unsigned int non_word_index = new_word_count;
    unsigned int new_ri_count = 0;
    for (r = 0; r < ri_count; r++)
      if (delta_sz[r])
      {
	ri_map[r] = (delta_sz[r] == kWord)? word_index++ : non_word_index++;
	new_ri_count++;
      }

    wordSizeCount = new_word_count | (has_long ? 0x8000u /* LONG_WORDS */ : 0);

    regionIndices.len = new_ri_count;

    if (unlikely (!c->extend (this))) return_trace (false);

    for (r = 0; r < ri_count; r++)
      if (delta_sz[r]) regionIndices[ri_map[r]] = region_map[src->regionIndices[r]];

    for (unsigned int i = 0; i < itemCount; i++)
    {
      unsigned int	old = inner_map.backward (i);
      for (unsigned int r = 0; r < ri_count; r++)
	if (delta_sz[r]) set_item_delta (i, ri_map[r], src->get_item_delta (old, r));
    }

    return_trace (true);
  }

  void collect_region_refs (hb_set_t &region_indices, const hb_inc_bimap_t &inner_map) const
  {
    for (unsigned int r = 0; r < regionIndices.len; r++)
    {
      unsigned int region = regionIndices[r];
      if (region_indices.has (region)) continue;
      for (unsigned int i = 0; i < inner_map.get_next_value (); i++)
	if (get_item_delta (inner_map.backward (i), r) != 0)
	{
	  region_indices.add (region);
	  break;
	}
    }
  }

  protected:
  const HBUINT8 *get_delta_bytes () const
  { return &StructAfter<HBUINT8> (regionIndices); }

  HBUINT8 *get_delta_bytes ()
  { return &StructAfter<HBUINT8> (regionIndices); }

  int32_t get_item_delta (unsigned int item, unsigned int region) const
  {
    if ( item >= itemCount || unlikely (region >= regionIndices.len)) return 0;
    const HBINT8 *p = (const HBINT8 *) get_delta_bytes () + item * get_row_size ();
    unsigned word_count = wordCount ();
    bool is_long = longWords ();
    if (is_long)
    {
      if (region < word_count)
	return ((const HBINT32 *) p)[region];
      else
	return ((const HBINT16 *)(p + HBINT32::static_size * word_count))[region - word_count];
    }
    else
    {
      if (region < word_count)
	return ((const HBINT16 *) p)[region];
      else
	return (p + HBINT16::static_size * word_count)[region - word_count];
    }
  }

  void set_item_delta (unsigned int item, unsigned int region, int32_t delta)
  {
    HBINT8 *p = (HBINT8 *)get_delta_bytes () + item * get_row_size ();
    unsigned word_count = wordCount ();
    bool is_long = longWords ();
    if (is_long)
    {
      if (region < word_count)
	((HBINT32 *) p)[region] = delta;
      else
	((HBINT16 *)(p + HBINT32::static_size * word_count))[region - word_count] = delta;
    }
    else
    {
      if (region < word_count)
	((HBINT16 *) p)[region] = delta;
      else
	(p + HBINT16::static_size * word_count)[region - word_count] = delta;
    }
  }

  bool longWords () const { return wordSizeCount & 0x8000u /* LONG_WORDS */; }
  unsigned wordCount () const { return wordSizeCount & 0x7FFFu /* WORD_DELTA_COUNT_MASK */; }

  protected:
  HBUINT16		itemCount;
  HBUINT16		wordSizeCount;
  Array16Of<HBUINT16>	regionIndices;
/*UnsizedArrayOf<HBUINT8>bytesX;*/
  public:
  DEFINE_SIZE_ARRAY (6, regionIndices);
};

struct VariationStore
{
  using cache_t = VarRegionList::cache_t;

  cache_t *create_cache () const
  {
#ifdef HB_NO_VAR
    return nullptr;
#endif
    auto &r = this+regions;
    unsigned count = r.regionCount;

    float *cache = (float *) hb_malloc (sizeof (float) * count);
    if (unlikely (!cache)) return nullptr;

    for (unsigned i = 0; i < count; i++)
      cache[i] = REGION_CACHE_ITEM_CACHE_INVALID;

    return cache;
  }

  static void destroy_cache (cache_t *cache) { hb_free (cache); }

  private:
  float get_delta (unsigned int outer, unsigned int inner,
		   const int *coords, unsigned int coord_count,
		   VarRegionList::cache_t *cache = nullptr) const
  {
#ifdef HB_NO_VAR
    return 0.f;
#endif

    if (unlikely (outer >= dataSets.len))
      return 0.f;

    return (this+dataSets[outer]).get_delta (inner,
					     coords, coord_count,
					     this+regions,
					     cache);
  }

  public:
  float get_delta (unsigned int index,
		   const int *coords, unsigned int coord_count,
		   VarRegionList::cache_t *cache = nullptr) const
  {
    unsigned int outer = index >> 16;
    unsigned int inner = index & 0xFFFF;
    return get_delta (outer, inner, coords, coord_count, cache);
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
#ifdef HB_NO_VAR
    return true;
#endif

    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  format == 1 &&
		  regions.sanitize (c, this) &&
		  dataSets.sanitize (c, this));
  }

  bool serialize (hb_serialize_context_t *c,
		  const VariationStore *src,
		  const hb_array_t <hb_inc_bimap_t> &inner_maps)
  {
    TRACE_SERIALIZE (this);
#ifdef HB_NO_VAR
    return_trace (false);
#endif

    if (unlikely (!c->extend_min (this))) return_trace (false);

    unsigned int set_count = 0;
    for (unsigned int i = 0; i < inner_maps.length; i++)
      if (inner_maps[i].get_population ())
	set_count++;

    format = 1;

    const auto &src_regions = src+src->regions;

    hb_set_t region_indices;
    for (unsigned int i = 0; i < inner_maps.length; i++)
      (src+src->dataSets[i]).collect_region_refs (region_indices, inner_maps[i]);

    if (region_indices.in_error ())
      return_trace (false);

    region_indices.del_range ((src_regions).regionCount, hb_set_t::INVALID);

    /* TODO use constructor when our data-structures support that. */
    hb_inc_bimap_t region_map;
    + hb_iter (region_indices)
    | hb_apply ([&region_map] (unsigned _) { region_map.add(_); })
    ;
    if (region_map.in_error())
      return_trace (false);

    if (unlikely (!regions.serialize_serialize (c, &src_regions, region_map)))
      return_trace (false);

    dataSets.len = set_count;
    if (unlikely (!c->extend (dataSets))) return_trace (false);

    /* TODO: The following code could be simplified when
     * List16OfOffset16To::subset () can take a custom param to be passed to VarData::serialize () */
    unsigned int set_index = 0;
    for (unsigned int i = 0; i < inner_maps.length; i++)
    {
      if (!inner_maps[i].get_population ()) continue;
      if (unlikely (!dataSets[set_index++]
		     .serialize_serialize (c, &(src+src->dataSets[i]), inner_maps[i], region_map)))
	return_trace (false);
    }

    return_trace (true);
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
#ifdef HB_NO_VAR
    return_trace (false);
#endif

    VariationStore *varstore_prime = c->serializer->start_embed<VariationStore> ();
    if (unlikely (!varstore_prime)) return_trace (false);

    const hb_set_t *variation_indices = c->plan->layout_variation_indices;
    if (variation_indices->is_empty ()) return_trace (false);

    hb_vector_t<hb_inc_bimap_t> inner_maps;
    inner_maps.resize ((unsigned) dataSets.len);

    for (unsigned idx : c->plan->layout_variation_indices->iter ())
    {
      uint16_t major = idx >> 16;
      uint16_t minor = idx & 0xFFFF;

      if (major >= inner_maps.length)
	return_trace (false);
      inner_maps[major].add (minor);
    }
    varstore_prime->serialize (c->serializer, this, inner_maps.as_array ());

    return_trace (
        !c->serializer->in_error()
        && varstore_prime->dataSets);
  }

  unsigned int get_region_index_count (unsigned int major) const
  {
#ifdef HB_NO_VAR
    return 0;
#endif
    return (this+dataSets[major]).get_region_index_count ();
  }

  void get_region_scalars (unsigned int major,
			   const int *coords, unsigned int coord_count,
			   float *scalars /*OUT*/,
			   unsigned int num_scalars) const
  {
#ifdef HB_NO_VAR
    for (unsigned i = 0; i < num_scalars; i++)
      scalars[i] = 0.f;
    return;
#endif

    (this+dataSets[major]).get_region_scalars (coords, coord_count,
					       this+regions,
					       &scalars[0], num_scalars);
  }

  unsigned int get_sub_table_count () const
   {
#ifdef HB_NO_VAR
     return 0;
#endif
     return dataSets.len;
   }

  protected:
  HBUINT16				format;
  Offset32To<VarRegionList>		regions;
  Array16OfOffset32To<VarData>		dataSets;
  public:
  DEFINE_SIZE_ARRAY_SIZED (8, dataSets);
};

#undef REGION_CACHE_ITEM_CACHE_INVALID

/*
 * Feature Variations
 */

struct ConditionFormat1
{
  friend struct Condition;

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->embed (this);
    if (unlikely (!out)) return_trace (false);
    return_trace (true);
  }

  private:
  bool evaluate (const int *coords, unsigned int coord_len) const
  {
    int coord = axisIndex < coord_len ? coords[axisIndex] : 0;
    return filterRangeMinValue <= coord && coord <= filterRangeMaxValue;
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  protected:
  HBUINT16	format;		/* Format identifier--format = 1 */
  HBUINT16	axisIndex;
  F2DOT14	filterRangeMinValue;
  F2DOT14	filterRangeMaxValue;
  public:
  DEFINE_SIZE_STATIC (8);
};

struct Condition
{
  bool evaluate (const int *coords, unsigned int coord_len) const
  {
    switch (u.format) {
    case 1: return u.format1.evaluate (coords, coord_len);
    default:return false;
    }
  }

  template <typename context_t, typename ...Ts>
  typename context_t::return_t dispatch (context_t *c, Ts&&... ds) const
  {
    TRACE_DISPATCH (this, u.format);
    if (unlikely (!c->may_dispatch (this, &u.format))) return_trace (c->no_dispatch_return_value ());
    switch (u.format) {
    case 1: return_trace (c->dispatch (u.format1, std::forward<Ts> (ds)...));
    default:return_trace (c->default_return_value ());
    }
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (!u.format.sanitize (c)) return_trace (false);
    switch (u.format) {
    case 1: return_trace (u.format1.sanitize (c));
    default:return_trace (true);
    }
  }

  protected:
  union {
  HBUINT16		format;		/* Format identifier */
  ConditionFormat1	format1;
  } u;
  public:
  DEFINE_SIZE_UNION (2, format);
};

struct ConditionSet
{
  bool evaluate (const int *coords, unsigned int coord_len) const
  {
    unsigned int count = conditions.len;
    for (unsigned int i = 0; i < count; i++)
      if (!(this+conditions.arrayZ[i]).evaluate (coords, coord_len))
	return false;
    return true;
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->start_embed (this);
    if (unlikely (!out || !c->serializer->extend_min (out))) return_trace (false);

    + conditions.iter ()
    | hb_apply (subset_offset_array (c, out->conditions, this))
    ;

    return_trace (bool (out->conditions));
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (conditions.sanitize (c, this));
  }

  protected:
  Array16OfOffset32To<Condition>	conditions;
  public:
  DEFINE_SIZE_ARRAY (2, conditions);
};

struct FeatureTableSubstitutionRecord
{
  friend struct FeatureTableSubstitution;

  void collect_lookups (const void *base, hb_set_t *lookup_indexes /* OUT */) const
  {
    return (base+feature).add_lookup_indexes_to (lookup_indexes);
  }

  void closure_features (const void *base,
			 const hb_map_t *lookup_indexes,
			 hb_set_t       *feature_indexes /* OUT */) const
  {
    if ((base+feature).intersects_lookup_indexes (lookup_indexes))
      feature_indexes->add (featureIndex);
  }

  bool subset (hb_subset_layout_context_t *c, const void *base) const
  {
    TRACE_SUBSET (this);
    if (!c->feature_index_map->has (featureIndex)) {
      // Feature that is being substituted is not being retained, so we don't
      // need this.
      return_trace (false);
    }

    auto *out = c->subset_context->serializer->embed (this);
    if (unlikely (!out)) return_trace (false);

    out->featureIndex = c->feature_index_map->get (featureIndex);
    bool ret = out->feature.serialize_subset (c->subset_context, feature, base, c);
    return_trace (ret);
  }

  bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && feature.sanitize (c, base));
  }

  protected:
  HBUINT16		featureIndex;
  Offset32To<Feature>	feature;
  public:
  DEFINE_SIZE_STATIC (6);
};

struct FeatureTableSubstitution
{
  const Feature *find_substitute (unsigned int feature_index) const
  {
    unsigned int count = substitutions.len;
    for (unsigned int i = 0; i < count; i++)
    {
      const FeatureTableSubstitutionRecord &record = substitutions.arrayZ[i];
      if (record.featureIndex == feature_index)
	return &(this+record.feature);
    }
    return nullptr;
  }

  void collect_lookups (const hb_set_t *feature_indexes,
			hb_set_t       *lookup_indexes /* OUT */) const
  {
    + hb_iter (substitutions)
    | hb_filter (feature_indexes, &FeatureTableSubstitutionRecord::featureIndex)
    | hb_apply ([this, lookup_indexes] (const FeatureTableSubstitutionRecord& r)
		{ r.collect_lookups (this, lookup_indexes); })
    ;
  }

  void closure_features (const hb_map_t *lookup_indexes,
			 hb_set_t       *feature_indexes /* OUT */) const
  {
    for (const FeatureTableSubstitutionRecord& record : substitutions)
      record.closure_features (this, lookup_indexes, feature_indexes);
  }

  bool intersects_features (const hb_map_t *feature_index_map) const
  {
    for (const FeatureTableSubstitutionRecord& record : substitutions)
    {
      if (feature_index_map->has (record.featureIndex)) return true;
    }
    return false;
  }

  bool subset (hb_subset_context_t        *c,
	       hb_subset_layout_context_t *l) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->start_embed (*this);
    if (unlikely (!out || !c->serializer->extend_min (out))) return_trace (false);

    out->version.major = version.major;
    out->version.minor = version.minor;

    + substitutions.iter ()
    | hb_apply (subset_record_array (l, &(out->substitutions), this))
    ;

    return_trace (bool (out->substitutions));
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (version.sanitize (c) &&
		  likely (version.major == 1) &&
		  substitutions.sanitize (c, this));
  }

  protected:
  FixedVersion<>	version;	/* Version--0x00010000u */
  Array16Of<FeatureTableSubstitutionRecord>
			substitutions;
  public:
  DEFINE_SIZE_ARRAY (6, substitutions);
};

struct FeatureVariationRecord
{
  friend struct FeatureVariations;

  void collect_lookups (const void     *base,
			const hb_set_t *feature_indexes,
			hb_set_t       *lookup_indexes /* OUT */) const
  {
    return (base+substitutions).collect_lookups (feature_indexes, lookup_indexes);
  }

  void closure_features (const void     *base,
			 const hb_map_t *lookup_indexes,
			 hb_set_t       *feature_indexes /* OUT */) const
  {
    (base+substitutions).closure_features (lookup_indexes, feature_indexes);
  }

  bool intersects_features (const void *base, const hb_map_t *feature_index_map) const
  {
    return (base+substitutions).intersects_features (feature_index_map);
  }

  bool subset (hb_subset_layout_context_t *c, const void *base) const
  {
    TRACE_SUBSET (this);
    auto *out = c->subset_context->serializer->embed (this);
    if (unlikely (!out)) return_trace (false);

    out->conditions.serialize_subset (c->subset_context, conditions, base);
    out->substitutions.serialize_subset (c->subset_context, substitutions, base, c);

    return_trace (true);
  }

  bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    return_trace (conditions.sanitize (c, base) &&
		  substitutions.sanitize (c, base));
  }

  protected:
  Offset32To<ConditionSet>
			conditions;
  Offset32To<FeatureTableSubstitution>
			substitutions;
  public:
  DEFINE_SIZE_STATIC (8);
};

struct FeatureVariations
{
  static constexpr unsigned NOT_FOUND_INDEX = 0xFFFFFFFFu;

  bool find_index (const int *coords, unsigned int coord_len,
		   unsigned int *index) const
  {
    unsigned int count = varRecords.len;
    for (unsigned int i = 0; i < count; i++)
    {
      const FeatureVariationRecord &record = varRecords.arrayZ[i];
      if ((this+record.conditions).evaluate (coords, coord_len))
      {
	*index = i;
	return true;
      }
    }
    *index = NOT_FOUND_INDEX;
    return false;
  }

  const Feature *find_substitute (unsigned int variations_index,
				  unsigned int feature_index) const
  {
    const FeatureVariationRecord &record = varRecords[variations_index];
    return (this+record.substitutions).find_substitute (feature_index);
  }

  FeatureVariations* copy (hb_serialize_context_t *c) const
  {
    TRACE_SERIALIZE (this);
    return_trace (c->embed (*this));
  }

  void collect_lookups (const hb_set_t *feature_indexes,
			hb_set_t       *lookup_indexes /* OUT */) const
  {
    for (const FeatureVariationRecord& r : varRecords)
      r.collect_lookups (this, feature_indexes, lookup_indexes);
  }

  void closure_features (const hb_map_t *lookup_indexes,
			 hb_set_t       *feature_indexes /* OUT */) const
  {
    for (const FeatureVariationRecord& record : varRecords)
      record.closure_features (this, lookup_indexes, feature_indexes);
  }

  bool subset (hb_subset_context_t *c,
	       hb_subset_layout_context_t *l) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->start_embed (*this);
    if (unlikely (!out || !c->serializer->extend_min (out))) return_trace (false);

    out->version.major = version.major;
    out->version.minor = version.minor;

    int keep_up_to = -1;
    for (int i = varRecords.len - 1; i >= 0; i--) {
      if (varRecords[i].intersects_features (this, l->feature_index_map)) {
        keep_up_to = i;
        break;
      }
    }

    unsigned count = (unsigned) (keep_up_to + 1);
    for (unsigned i = 0; i < count; i++) {
      subset_record_array (l, &(out->varRecords), this) (varRecords[i]);
    }
    return_trace (bool (out->varRecords));
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (version.sanitize (c) &&
		  likely (version.major == 1) &&
		  varRecords.sanitize (c, this));
  }

  protected:
  FixedVersion<>	version;	/* Version--0x00010000u */
  Array32Of<FeatureVariationRecord>
			varRecords;
  public:
  DEFINE_SIZE_ARRAY_SIZED (8, varRecords);
};


/*
 * Device Tables
 */

struct HintingDevice
{
  friend struct Device;

  private:

  hb_position_t get_x_delta (hb_font_t *font) const
  { return get_delta (font->x_ppem, font->x_scale); }

  hb_position_t get_y_delta (hb_font_t *font) const
  { return get_delta (font->y_ppem, font->y_scale); }

  public:

  unsigned int get_size () const
  {
    unsigned int f = deltaFormat;
    if (unlikely (f < 1 || f > 3 || startSize > endSize)) return 3 * HBUINT16::static_size;
    return HBUINT16::static_size * (4 + ((endSize - startSize) >> (4 - f)));
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && c->check_range (this, this->get_size ()));
  }

  HintingDevice* copy (hb_serialize_context_t *c) const
  {
    TRACE_SERIALIZE (this);
    return_trace (c->embed<HintingDevice> (this));
  }

  private:

  int get_delta (unsigned int ppem, int scale) const
  {
    if (!ppem) return 0;

    int pixels = get_delta_pixels (ppem);

    if (!pixels) return 0;

    return (int) (pixels * (int64_t) scale / ppem);
  }
  int get_delta_pixels (unsigned int ppem_size) const
  {
    unsigned int f = deltaFormat;
    if (unlikely (f < 1 || f > 3))
      return 0;

    if (ppem_size < startSize || ppem_size > endSize)
      return 0;

    unsigned int s = ppem_size - startSize;

    unsigned int byte = deltaValueZ[s >> (4 - f)];
    unsigned int bits = (byte >> (16 - (((s & ((1 << (4 - f)) - 1)) + 1) << f)));
    unsigned int mask = (0xFFFFu >> (16 - (1 << f)));

    int delta = bits & mask;

    if ((unsigned int) delta >= ((mask + 1) >> 1))
      delta -= mask + 1;

    return delta;
  }

  protected:
  HBUINT16	startSize;		/* Smallest size to correct--in ppem */
  HBUINT16	endSize;		/* Largest size to correct--in ppem */
  HBUINT16	deltaFormat;		/* Format of DeltaValue array data: 1, 2, or 3
					 * 1	Signed 2-bit value, 8 values per uint16
					 * 2	Signed 4-bit value, 4 values per uint16
					 * 3	Signed 8-bit value, 2 values per uint16
					 */
  UnsizedArrayOf<HBUINT16>
		deltaValueZ;		/* Array of compressed data */
  public:
  DEFINE_SIZE_ARRAY (6, deltaValueZ);
};

struct VariationDevice
{
  friend struct Device;

  private:

  hb_position_t get_x_delta (hb_font_t *font,
			     const VariationStore &store,
			     VariationStore::cache_t *store_cache = nullptr) const
  { return font->em_scalef_x (get_delta (font, store, store_cache)); }

  hb_position_t get_y_delta (hb_font_t *font,
			     const VariationStore &store,
			     VariationStore::cache_t *store_cache = nullptr) const
  { return font->em_scalef_y (get_delta (font, store, store_cache)); }

  VariationDevice* copy (hb_serialize_context_t *c, const hb_map_t *layout_variation_idx_map) const
  {
    TRACE_SERIALIZE (this);
    auto snap = c->snapshot ();
    auto *out = c->embed (this);
    if (unlikely (!out)) return_trace (nullptr);
    if (!layout_variation_idx_map || layout_variation_idx_map->is_empty ()) return_trace (out);

    /* TODO Just get() and bail if NO_VARIATION. Needs to setup the map to return that. */
    if (!layout_variation_idx_map->has (varIdx))
    {
      c->revert (snap);
      return_trace (nullptr);
    }
    unsigned new_idx = layout_variation_idx_map->get (varIdx);
    out->varIdx = new_idx;
    return_trace (out);
  }

  void record_variation_index (hb_set_t *layout_variation_indices) const
  {
    layout_variation_indices->add (varIdx);
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  private:

  float get_delta (hb_font_t *font,
		   const VariationStore &store,
		   VariationStore::cache_t *store_cache = nullptr) const
  {
    return store.get_delta (varIdx, font->coords, font->num_coords, (VariationStore::cache_t *) store_cache);
  }

  protected:
  VarIdx	varIdx;
  HBUINT16	deltaFormat;	/* Format identifier for this table: 0x0x8000 */
  public:
  DEFINE_SIZE_STATIC (6);
};

struct DeviceHeader
{
  protected:
  HBUINT16		reserved1;
  HBUINT16		reserved2;
  public:
  HBUINT16		format;		/* Format identifier */
  public:
  DEFINE_SIZE_STATIC (6);
};

struct Device
{
  hb_position_t get_x_delta (hb_font_t *font,
			     const VariationStore &store=Null (VariationStore),
			     VariationStore::cache_t *store_cache = nullptr) const
  {
    switch (u.b.format)
    {
#ifndef HB_NO_HINTING
    case 1: case 2: case 3:
      return u.hinting.get_x_delta (font);
#endif
#ifndef HB_NO_VAR
    case 0x8000:
      return u.variation.get_x_delta (font, store, store_cache);
#endif
    default:
      return 0;
    }
  }
  hb_position_t get_y_delta (hb_font_t *font,
			     const VariationStore &store=Null (VariationStore),
			     VariationStore::cache_t *store_cache = nullptr) const
  {
    switch (u.b.format)
    {
    case 1: case 2: case 3:
#ifndef HB_NO_HINTING
      return u.hinting.get_y_delta (font);
#endif
#ifndef HB_NO_VAR
    case 0x8000:
      return u.variation.get_y_delta (font, store, store_cache);
#endif
    default:
      return 0;
    }
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (!u.b.format.sanitize (c)) return_trace (false);
    switch (u.b.format) {
#ifndef HB_NO_HINTING
    case 1: case 2: case 3:
      return_trace (u.hinting.sanitize (c));
#endif
#ifndef HB_NO_VAR
    case 0x8000:
      return_trace (u.variation.sanitize (c));
#endif
    default:
      return_trace (true);
    }
  }

  Device* copy (hb_serialize_context_t *c, const hb_map_t *layout_variation_idx_map=nullptr) const
  {
    TRACE_SERIALIZE (this);
    switch (u.b.format) {
#ifndef HB_NO_HINTING
    case 1:
    case 2:
    case 3:
      return_trace (reinterpret_cast<Device *> (u.hinting.copy (c)));
#endif
#ifndef HB_NO_VAR
    case 0x8000:
      return_trace (reinterpret_cast<Device *> (u.variation.copy (c, layout_variation_idx_map)));
#endif
    default:
      return_trace (nullptr);
    }
  }

  void collect_variation_indices (hb_set_t *layout_variation_indices) const
  {
    switch (u.b.format) {
#ifndef HB_NO_HINTING
    case 1:
    case 2:
    case 3:
      return;
#endif
#ifndef HB_NO_VAR
    case 0x8000:
      u.variation.record_variation_index (layout_variation_indices);
      return;
#endif
    default:
      return;
    }
  }

  protected:
  union {
  DeviceHeader		b;
  HintingDevice		hinting;
#ifndef HB_NO_VAR
  VariationDevice	variation;
#endif
  } u;
  public:
  DEFINE_SIZE_UNION (6, b);
};


} /* namespace OT */


#endif /* HB_OT_LAYOUT_COMMON_HH */
