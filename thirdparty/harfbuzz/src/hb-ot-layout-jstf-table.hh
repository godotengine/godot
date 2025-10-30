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

typedef SortedArray16Of<HBGlyphID16> ExtenderGlyphs;


/*
 * JstfScript -- The Justification Table
 */

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
  Offset16To<ExtenderGlyphs>
		extenderGlyphs;	/* Offset to ExtenderGlyph table--from beginning
				 * of JstfScript table-may be NULL */
  Offset16To<JstfLangSys>
		defaultLangSys;	/* Offset to DefaultJstfLangSys table--from
				 * beginning of JstfScript table--may be Null */
  RecordArrayOf<JstfLangSys>
		langSys;	/* Array of JstfLangSysRecords--listed
				 * alphabetically by LangSysTag */
  public:
  DEFINE_SIZE_ARRAY (6, langSys);
};


/*
 * JSTF -- Justification
 * https://docs.microsoft.com/en-us/typography/opentype/spec/jstf
 */

struct JSTF
{
  static constexpr hb_tag_t tableTag = HB_OT_TAG_JSTF;

  unsigned int get_script_count () const
  { return scriptList.len; }
  const Tag& get_script_tag (unsigned int i) const
  { return scriptList.get_tag (i); }
  unsigned int get_script_tags (unsigned int start_offset,
				unsigned int *script_count /* IN/OUT */,
				hb_tag_t     *script_tags /* OUT */) const
  { return scriptList.get_tags (start_offset, script_count, script_tags); }
  const JstfScript& get_script (unsigned int i) const
  { return this+scriptList[i].offset; }
  bool find_script_index (hb_tag_t tag, unsigned int *index) const
  { return scriptList.find_index (tag, index); }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (version.sanitize (c) &&
		  hb_barrier () &&
		  likely (version.major == 1) &&
		  scriptList.sanitize (c, this));
  }

  protected:
  FixedVersion<>version;	/* Version of the JSTF table--initially set
				 * to 0x00010000u */
  RecordArrayOf<JstfScript>
		scriptList;	/* Array of JstfScripts--listed
				 * alphabetically by ScriptTag */
  public:
  DEFINE_SIZE_ARRAY (6, scriptList);
};


} /* namespace OT */


#endif /* HB_OT_LAYOUT_JSTF_TABLE_HH */
