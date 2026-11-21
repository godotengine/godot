/*
 * Copyright © 2013  Google, Inc.
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

#ifndef HB_OT_LAYOUT_JSTF_TABLE_HH
#define HB_OT_LAYOUT_JSTF_TABLE_HH

#include "hb-open-type.hh"
#include "hb-ot-layout-gpos-table.hh"


namespace OT {


/*
 * JstfModList -- Justification Modification List Tables
 */

typedef IndexArray JstfModList;


/*
 * JstfMax -- Justification Maximum Table
 */

typedef List16OfOffset16To<PosLookup> JstfMax;


/*
 * JstfPriority -- Justification Priority Table
 */

struct JstfPriority
{
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  shrinkageEnableGSUB.sanitize (c, this) &&
		  shrinkageDisableGSUB.sanitize (c, this) &&
		  shrinkageEnableGPOS.sanitize (c, this) &&
		  shrinkageDisableGPOS.sanitize (c, this) &&
		  shrinkageJstfMax.sanitize (c, this) &&
		  extensionEnableGSUB.sanitize (c, this) &&
		  extensionDisableGSUB.sanitize (c, this) &&
		  extensionEnableGPOS.sanitize (c, this) &&
		  extensionDisableGPOS.sanitize (c, this) &&
		  extensionJstfMax.sanitize (c, this));
  }

  protected:
  Offset16To<JstfModList>
		shrinkageEnableGSUB;	/* Offset to Shrinkage Enable GSUB
					 * JstfModList table--from beginning of
					 * JstfPriority table--may be NULL */
  Offset16To<JstfModList>
		shrinkageDisableGSUB;	/* Offset to Shrinkage Disable GSUB
					 * JstfModList table--from beginning of
					 * JstfPriority table--may be NULL */
  Offset16To<JstfModList>
		shrinkageEnableGPOS;	/* Offset to Shrinkage Enable GPOS
					 * JstfModList table--from beginning of
					 * JstfPriority table--may be NULL */
  Offset16To<JstfModList>
		shrinkageDisableGPOS;	/* Offset to Shrinkage Disable GPOS
					 * JstfModList table--from beginning of
					 * JstfPriority table--may be NULL */
  Offset16To<JstfMax>
		shrinkageJstfMax;	/* Offset to Shrinkage JstfMax table--
					 * from beginning of JstfPriority table
					 * --may be NULL */
  Offset16To<JstfModList>
		extensionEnableGSUB;	/* Offset to Extension Enable GSUB
					 * JstfModList table--from beginning of
					 * JstfPriority table--may be NULL */
  Offset16To<JstfModList>
		extensionDisableGSUB;	/* Offset to Extension Disable GSUB
					 * JstfModList table--from beginning of
					 * JstfPriority table--may be NULL */
  Offset16To<JstfModList>
		extensionEnableGPOS;	/* Offset to Extension Enable GPOS
					 * JstfModList table--from beginning of
					 * JstfPriority table--may be NULL */
  Offset16To<JstfModList>
		extensionDisableGPOS;	/* Offset to Extension Disable GPOS
					 * JstfModList table--from beginning of
					 * JstfPriority table--may be NULL */
  Offset16To<JstfMax>
		extensionJstfMax;	/* Offset to Extension JstfMax table--
					 * from beginning of JstfPriority table
					 * --may be NULL */

  public:
  DEFINE_SIZE_STATIC (20);
};


/*
 * JstfLangSys -- Justification Language System Table
 */

struct JstfLangSys : List16OfOffset16To<JstfPriority>
{
  bool sanitize (hb_sanitize_context_t *c,
		 const Record_sanitize_closure_t * = nullptr) const
  {
    TRACE_SANITIZE (this);
    return_trace (List16OfOffset16To<JstfPriority>::sanitize (c));
  }
};


/*
 * ExtenderGlyphs -- Extender Glyph Table
 */

template <typename Types>
using ExtenderGlyphs = SortedArray16Of<typename Types::HBGlyphID>;


/*
 * JstfScript -- The Justification Table
 */

template <typename Types>
struct JstfScript
{
  unsigned int get_lang_sys_count () const
  { return langSys.len; }
  const Tag& get_lang_sys_tag (unsigned int i) const
  { return langSys.get_tag (i); }
  unsigned int get_lang_sys_tags (unsigned int start_offset,
				  unsigned int *lang_sys_count /* IN/OUT */,
				  hb_tag_t     *lang_sys_tags /* OUT */) const
  { return langSys.get_tags (start_offset, lang_sys_count, lang_sys_tags); }
  const JstfLangSys& get_lang_sys (unsigned int i) const
  {
    if (i == Index::NOT_FOUND_INDEX) return get_default_lang_sys ();
    return this+langSys[i].offset;
  }
  bool find_lang_sys_index (hb_tag_t tag, unsigned int *index) const
  { return langSys.find_index (tag, index); }

  bool has_default_lang_sys () const               { return defaultLangSys != 0; }
  const JstfLangSys& get_default_lang_sys () const { return this+defaultLangSys; }

  bool sanitize (hb_sanitize_context_t *c,
		 const Record_sanitize_closure_t * = nullptr) const
  {
    TRACE_SANITIZE (this);
    return_trace (extenderGlyphs.sanitize (c, this) &&
		  defaultLangSys.sanitize (c, this) &&
		  langSys.sanitize (c, this));
  }

  protected:
  typename Types::template LOffsetTo<ExtenderGlyphs<Types>>
		extenderGlyphs;	/* Offset to ExtenderGlyph table--from beginning
				 * of JstfScript table-may be NULL */
  typename Types::template LOffsetTo<JstfLangSys>
		defaultLangSys;	/* Offset to DefaultJstfLangSys table--from
				 * beginning of JstfScript table--may be Null */
  RecordArrayOf<JstfLangSys>
		langSys;	/* Array of JstfLangSysRecords--listed
				 * alphabetically by LangSysTag */
  public:
  DEFINE_SIZE_ARRAY (2 * Types::LOffset::static_size + 2, langSys);
};

template <typename Types>
struct JstfScriptRecord
{
  int cmp (hb_tag_t a) const { return tag.cmp (a); }

  bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    const Record_sanitize_closure_t closure = {tag, base};
    return_trace (c->check_struct (this) &&
		  offset.sanitize (c, base, &closure));
  }

  Tag		tag;		/* 4-byte JstfScript identification tag */
  typename Types::template LOffsetTo<JstfScript<Types>>
		offset;		/* Offset from beginning of JSTF header */
  public:
  DEFINE_SIZE_STATIC (4 + Types::LOffset::static_size);
};

template <typename Types>
struct JstfScriptList : SortedArray16Of<JstfScriptRecord<Types>>
{
  const Tag& get_tag (unsigned int i) const
  { return (*this)[i].tag; }
  unsigned int get_tags (unsigned int start_offset,
			 unsigned int *script_count /* IN/OUT */,
			 hb_tag_t     *script_tags /* OUT */) const
  {
    if (script_count)
    {
      + this->as_array ().sub_array (start_offset, script_count)
      | hb_map (&JstfScriptRecord<Types>::tag)
      | hb_sink (hb_array (script_tags, *script_count))
      ;
    }
    return this->len;
  }
  const JstfScript<Types>& get_script (const void *base, unsigned int i) const
  { return base+(*this)[i].offset; }
  bool find_index (hb_tag_t tag, unsigned int *index) const
  {
    return this->bfind (tag, index, HB_NOT_FOUND_STORE, Index::NOT_FOUND_INDEX);
  }
};


/*
 * JSTF -- Justification
 * https://docs.microsoft.com/en-us/typography/opentype/spec/jstf
 */

struct JSTF
{
  static constexpr hb_tag_t tableTag = HB_OT_TAG_JSTF;

  unsigned int get_script_count () const
  {
#ifndef HB_NO_BEYOND_64K
    if (has_script_list2 ()) return get_script_list2 ().len;
#endif
    return scriptList.len;
  }
  const Tag& get_script_tag (unsigned int i) const
  {
#ifndef HB_NO_BEYOND_64K
    if (has_script_list2 ()) return get_script_list2 ().get_tag (i);
#endif
    return scriptList.get_tag (i);
  }
  unsigned int get_script_tags (unsigned int start_offset,
				unsigned int *script_count /* IN/OUT */,
				hb_tag_t     *script_tags /* OUT */) const
  {
#ifndef HB_NO_BEYOND_64K
    if (has_script_list2 ())
      return get_script_list2 ().get_tags (start_offset, script_count, script_tags);
#endif
    return scriptList.get_tags (start_offset, script_count, script_tags);
  }
  template <typename Types>
  const JstfScript<Types>& get_script (unsigned int i) const;
  bool find_script_index (hb_tag_t tag, unsigned int *index) const
  {
#ifndef HB_NO_BEYOND_64K
    if (has_script_list2 ()) return get_script_list2 ().find_index (tag, index);
#endif
    return scriptList.find_index (tag, index);
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!version.sanitize (c) ||
		  !hb_barrier () ||
		  version.major != 1 ||
		  !scriptList.sanitize (c, this)))
      return_trace (false);
#ifndef HB_NO_BEYOND_64K
    if (unlikely (version.minor >= 1 && !get_script_list2 ().sanitize (c, this)))
      return_trace (false);
#endif
    return_trace (true);
  }

  private:
#ifndef HB_NO_BEYOND_64K
  bool has_script_list2 () const
  { return version.minor >= 1 && get_script_list2 ().len; }
  const JstfScriptList<Layout::MediumTypes>& get_script_list2 () const
  { return StructAfter<JstfScriptList<Layout::MediumTypes>> (scriptList); }
#endif

  protected:
  FixedVersion<>version;	/* Version of the JSTF table--initially set
				 * to 0x00010000u */
  JstfScriptList<Layout::SmallTypes>
		scriptList;	/* Array of JstfScripts--listed
				 * alphabetically by ScriptTag */
#ifndef HB_NO_BEYOND_64K
/*JstfScriptList<Layout::MediumTypes>
		scriptList2;*//* Array of JstfScript2s--listed
				 * alphabetically by ScriptTag.
				 * Present only if minorVersion is 1 or greater. */
#endif
  public:
  DEFINE_SIZE_ARRAY (6, scriptList);
};

template <>
inline const JstfScript<Layout::SmallTypes>&
JSTF::get_script<Layout::SmallTypes> (unsigned int i) const
{ return scriptList.get_script (this, i); }

#ifndef HB_NO_BEYOND_64K
template <>
inline const JstfScript<Layout::MediumTypes>&
JSTF::get_script<Layout::MediumTypes> (unsigned int i) const
{ return get_script_list2 ().get_script (this, i); }
#endif


} /* namespace OT */


#endif /* HB_OT_LAYOUT_JSTF_TABLE_HH */
