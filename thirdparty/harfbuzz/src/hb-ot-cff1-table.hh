/*
 * Copyright © 2018 Adobe Inc.
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

#ifndef HB_OT_CFF1_TABLE_HH
#define HB_OT_CFF1_TABLE_HH

#include "hb-ot-cff-common.hh"
#include "hb-subset-cff1.hh"
#include "hb-draw.hh"

#define HB_STRING_ARRAY_NAME cff1_std_strings
#define HB_STRING_ARRAY_LIST "hb-ot-cff1-std-str.hh"
#include "hb-string-array.hh"
#undef HB_STRING_ARRAY_LIST
#undef HB_STRING_ARRAY_NAME

namespace CFF {

/*
 * CFF -- Compact Font Format (CFF)
 * https://www.adobe.com/content/dam/acom/en/devnet/font/pdfs/5176.CFF.pdf
 */
#define HB_OT_TAG_cff1 HB_TAG('C','F','F',' ')

#define CFF_UNDEF_SID   CFF_UNDEF_CODE

enum EncodingID { StandardEncoding = 0, ExpertEncoding = 1 };
enum CharsetID { ISOAdobeCharset = 0, ExpertCharset = 1, ExpertSubsetCharset = 2 };

typedef CFFIndex<HBUINT16>  CFF1Index;
template <typename Type> struct CFF1IndexOf : CFFIndexOf<HBUINT16, Type> {};

typedef CFFIndex<HBUINT16> CFF1Index;
typedef CFF1Index          CFF1CharStrings;
typedef Subrs<HBUINT16>    CFF1Subrs;

struct CFF1FDSelect : FDSelect {};

/* Encoding */
struct Encoding0 {
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (codes.sanitize (c));
  }

  hb_codepoint_t get_code (hb_codepoint_t glyph) const
  {
    assert (glyph > 0);
    glyph--;
    if (glyph < nCodes ())
    {
      return (hb_codepoint_t)codes[glyph];
    }
    else
      return CFF_UNDEF_CODE;
  }

  HBUINT8 &nCodes () { return codes.len; }
  HBUINT8 nCodes () const { return codes.len; }

  ArrayOf<HBUINT8, HBUINT8> codes;

  DEFINE_SIZE_ARRAY_SIZED (1, codes);
};

struct Encoding1_Range {
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  HBUINT8   first;
  HBUINT8   nLeft;

  DEFINE_SIZE_STATIC (2);
};

struct Encoding1 {
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (ranges.sanitize (c));
  }

  hb_codepoint_t get_code (hb_codepoint_t glyph) const
  {
    assert (glyph > 0);
    glyph--;
    for (unsigned int i = 0; i < nRanges (); i++)
    {
      if (glyph <= ranges[i].nLeft)
      {
	hb_codepoint_t code = (hb_codepoint_t) ranges[i].first + glyph;
	return (likely (code < 0x100) ? code: CFF_UNDEF_CODE);
      }
      glyph -= (ranges[i].nLeft + 1);
    }
    return CFF_UNDEF_CODE;
  }

  HBUINT8 &nRanges () { return ranges.len; }
  HBUINT8 nRanges () const { return ranges.len; }

  ArrayOf<Encoding1_Range, HBUINT8> ranges;

  DEFINE_SIZE_ARRAY_SIZED (1, ranges);
};

struct SuppEncoding {
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  HBUINT8   code;
  HBUINT16  glyph;

  DEFINE_SIZE_STATIC (3);
};

struct CFF1SuppEncData {
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (supps.sanitize (c));
  }

  void get_codes (hb_codepoint_t sid, hb_vector_t<hb_codepoint_t> &codes) const
  {
    for (unsigned int i = 0; i < nSups (); i++)
      if (sid == supps[i].glyph)
	codes.push (supps[i].code);
  }

  HBUINT8 &nSups () { return supps.len; }
  HBUINT8 nSups () const { return supps.len; }

  ArrayOf<SuppEncoding, HBUINT8> supps;

  DEFINE_SIZE_ARRAY_SIZED (1, supps);
};

struct Encoding
{
  /* serialize a fullset Encoding */
  bool serialize (hb_serialize_context_t *c, const Encoding &src)
  {
    TRACE_SERIALIZE (this);
    unsigned int size = src.get_size ();
    Encoding *dest = c->allocate_size<Encoding> (size);
    if (unlikely (!dest)) return_trace (false);
    memcpy (dest, &src, size);
    return_trace (true);
  }

  /* serialize a subset Encoding */
  bool serialize (hb_serialize_context_t *c,
		  uint8_t format,
		  unsigned int enc_count,
		  const hb_vector_t<code_pair_t>& code_ranges,
		  const hb_vector_t<code_pair_t>& supp_codes)
  {
    TRACE_SERIALIZE (this);
    Encoding *dest = c->extend_min (*this);
    if (unlikely (!dest)) return_trace (false);
    dest->format = format | ((supp_codes.length > 0) ? 0x80 : 0);
    switch (format) {
    case 0:
    {
      Encoding0 *fmt0 = c->allocate_size<Encoding0> (Encoding0::min_size + HBUINT8::static_size * enc_count);
      if (unlikely (!fmt0)) return_trace (false);
      fmt0->nCodes () = enc_count;
      unsigned int glyph = 0;
      for (unsigned int i = 0; i < code_ranges.length; i++)
      {
	hb_codepoint_t code = code_ranges[i].code;
	for (int left = (int)code_ranges[i].glyph; left >= 0; left--)
	  fmt0->codes[glyph++] = code++;
	if (unlikely (!((glyph <= 0x100) && (code <= 0x100))))
	  return_trace (false);
      }
    }
    break;

    case 1:
    {
      Encoding1 *fmt1 = c->allocate_size<Encoding1> (Encoding1::min_size + Encoding1_Range::static_size * code_ranges.length);
      if (unlikely (!fmt1)) return_trace (false);
      fmt1->nRanges () = code_ranges.length;
      for (unsigned int i = 0; i < code_ranges.length; i++)
      {
	if (unlikely (!((code_ranges[i].code <= 0xFF) && (code_ranges[i].glyph <= 0xFF))))
	  return_trace (false);
	fmt1->ranges[i].first = code_ranges[i].code;
	fmt1->ranges[i].nLeft = code_ranges[i].glyph;
      }
    }
    break;

    }

    if (supp_codes.length)
    {
      CFF1SuppEncData *suppData = c->allocate_size<CFF1SuppEncData> (CFF1SuppEncData::min_size + SuppEncoding::static_size * supp_codes.length);
      if (unlikely (!suppData)) return_trace (false);
      suppData->nSups () = supp_codes.length;
      for (unsigned int i = 0; i < supp_codes.length; i++)
      {
	suppData->supps[i].code = supp_codes[i].code;
	suppData->supps[i].glyph = supp_codes[i].glyph; /* actually SID */
      }
    }

    return_trace (true);
  }

  unsigned int get_size () const
  {
    unsigned int size = min_size;
    switch (table_format ())
    {
    case 0: size += u.format0.get_size (); break;
    case 1: size += u.format1.get_size (); break;
    }
    if (has_supplement ())
      size += suppEncData ().get_size ();
    return size;
  }

  hb_codepoint_t get_code (hb_codepoint_t glyph) const
  {
    switch (table_format ())
    {
    case 0: return u.format0.get_code (glyph);
    case 1: return u.format1.get_code (glyph);
    default:return 0;
    }
  }

  uint8_t table_format () const { return format & 0x7F; }
  bool  has_supplement () const { return format & 0x80; }

  void get_supplement_codes (hb_codepoint_t sid, hb_vector_t<hb_codepoint_t> &codes) const
  {
    codes.resize (0);
    if (has_supplement ())
      suppEncData().get_codes (sid, codes);
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!c->check_struct (this)))
      return_trace (false);

    switch (table_format ())
    {
    case 0: if (unlikely (!u.format0.sanitize (c))) { return_trace (false); } break;
    case 1: if (unlikely (!u.format1.sanitize (c))) { return_trace (false); } break;
    default:return_trace (false);
    }
    return_trace (likely (!has_supplement () || suppEncData ().sanitize (c)));
  }

  protected:
  const CFF1SuppEncData &suppEncData () const
  {
    switch (table_format ())
    {
    case 0: return StructAfter<CFF1SuppEncData> (u.format0.codes[u.format0.nCodes ()-1]);
    case 1: return StructAfter<CFF1SuppEncData> (u.format1.ranges[u.format1.nRanges ()-1]);
    default:return Null (CFF1SuppEncData);
    }
  }

  public:
  HBUINT8	format;
  union {
  Encoding0	format0;
  Encoding1	format1;
  } u;
  /* CFF1SuppEncData  suppEncData; */

  DEFINE_SIZE_MIN (1);
};

/* Charset */
struct Charset0 {
  bool sanitize (hb_sanitize_context_t *c, unsigned int num_glyphs) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && sids[num_glyphs - 1].sanitize (c));
  }

  hb_codepoint_t get_sid (hb_codepoint_t glyph) const
  {
    if (glyph == 0)
      return 0;
    else
      return sids[glyph - 1];
  }

  hb_codepoint_t get_glyph (hb_codepoint_t sid, unsigned int num_glyphs) const
  {
    if (sid == 0)
      return 0;

    for (unsigned int glyph = 1; glyph < num_glyphs; glyph++)
    {
      if (sids[glyph-1] == sid)
	return glyph;
    }
    return 0;
  }

  unsigned int get_size (unsigned int num_glyphs) const
  {
    assert (num_glyphs > 0);
    return HBUINT16::static_size * (num_glyphs - 1);
  }

  HBUINT16  sids[HB_VAR_ARRAY];

  DEFINE_SIZE_ARRAY(0, sids);
};

template <typename TYPE>
struct Charset_Range {
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  HBUINT16  first;
  TYPE      nLeft;

  DEFINE_SIZE_STATIC (HBUINT16::static_size + TYPE::static_size);
};

template <typename TYPE>
struct Charset1_2 {
  bool sanitize (hb_sanitize_context_t *c, unsigned int num_glyphs) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!c->check_struct (this)))
      return_trace (false);
    num_glyphs--;
    for (unsigned int i = 0; num_glyphs > 0; i++)
    {
      if (unlikely (!ranges[i].sanitize (c) || (num_glyphs < ranges[i].nLeft + 1)))
	return_trace (false);
      num_glyphs -= (ranges[i].nLeft + 1);
    }
    return_trace (true);
  }

  hb_codepoint_t get_sid (hb_codepoint_t glyph) const
  {
    if (glyph == 0) return 0;
    glyph--;
    for (unsigned int i = 0;; i++)
    {
      if (glyph <= ranges[i].nLeft)
	return (hb_codepoint_t)ranges[i].first + glyph;
      glyph -= (ranges[i].nLeft + 1);
    }

    return 0;
  }

  hb_codepoint_t get_glyph (hb_codepoint_t sid, unsigned int num_glyphs) const
  {
    if (sid == 0) return 0;
    hb_codepoint_t  glyph = 1;
    for (unsigned int i = 0;; i++)
    {
      if (glyph >= num_glyphs)
	return 0;
      if ((ranges[i].first <= sid) && (sid <= ranges[i].first + ranges[i].nLeft))
	return glyph + (sid - ranges[i].first);
      glyph += (ranges[i].nLeft + 1);
    }

    return 0;
  }

  unsigned int get_size (unsigned int num_glyphs) const
  {
    unsigned int size = HBUINT8::static_size;
    int glyph = (int)num_glyphs;

    assert (glyph > 0);
    glyph--;
    for (unsigned int i = 0; glyph > 0; i++)
    {
      glyph -= (ranges[i].nLeft + 1);
      size += Charset_Range<TYPE>::static_size;
    }

    return size;
  }

  Charset_Range<TYPE>   ranges[HB_VAR_ARRAY];

  DEFINE_SIZE_ARRAY (0, ranges);
};

typedef Charset1_2<HBUINT8>     Charset1;
typedef Charset1_2<HBUINT16>    Charset2;
typedef Charset_Range<HBUINT8>  Charset1_Range;
typedef Charset_Range<HBUINT16> Charset2_Range;

struct Charset
{
  /* serialize a fullset Charset */
  bool serialize (hb_serialize_context_t *c, const Charset &src, unsigned int num_glyphs)
  {
    TRACE_SERIALIZE (this);
    unsigned int size = src.get_size (num_glyphs);
    Charset *dest = c->allocate_size<Charset> (size);
    if (unlikely (!dest)) return_trace (false);
    memcpy (dest, &src, size);
    return_trace (true);
  }

  /* serialize a subset Charset */
  bool serialize (hb_serialize_context_t *c,
		  uint8_t format,
		  unsigned int num_glyphs,
		  const hb_vector_t<code_pair_t>& sid_ranges)
  {
    TRACE_SERIALIZE (this);
    Charset *dest = c->extend_min (*this);
    if (unlikely (!dest)) return_trace (false);
    dest->format = format;
    switch (format)
    {
    case 0:
    {
      Charset0 *fmt0 = c->allocate_size<Charset0> (Charset0::min_size + HBUINT16::static_size * (num_glyphs - 1));
      if (unlikely (!fmt0)) return_trace (false);
      unsigned int glyph = 0;
      for (unsigned int i = 0; i < sid_ranges.length; i++)
      {
	hb_codepoint_t sid = sid_ranges[i].code;
	for (int left = (int)sid_ranges[i].glyph; left >= 0; left--)
	  fmt0->sids[glyph++] = sid++;
      }
    }
    break;

    case 1:
    {
      Charset1 *fmt1 = c->allocate_size<Charset1> (Charset1::min_size + Charset1_Range::static_size * sid_ranges.length);
      if (unlikely (!fmt1)) return_trace (false);
      for (unsigned int i = 0; i < sid_ranges.length; i++)
      {
	if (unlikely (!(sid_ranges[i].glyph <= 0xFF)))
	  return_trace (false);
	fmt1->ranges[i].first = sid_ranges[i].code;
	fmt1->ranges[i].nLeft = sid_ranges[i].glyph;
      }
    }
    break;

    case 2:
    {
      Charset2 *fmt2 = c->allocate_size<Charset2> (Charset2::min_size + Charset2_Range::static_size * sid_ranges.length);
      if (unlikely (!fmt2)) return_trace (false);
      for (unsigned int i = 0; i < sid_ranges.length; i++)
      {
	if (unlikely (!(sid_ranges[i].glyph <= 0xFFFF)))
	  return_trace (false);
	fmt2->ranges[i].first = sid_ranges[i].code;
	fmt2->ranges[i].nLeft = sid_ranges[i].glyph;
      }
    }
    break;

    }
    return_trace (true);
  }

  unsigned int get_size (unsigned int num_glyphs) const
  {
    switch (format)
    {
    case 0: return min_size + u.format0.get_size (num_glyphs);
    case 1: return min_size + u.format1.get_size (num_glyphs);
    case 2: return min_size + u.format2.get_size (num_glyphs);
    default:return 0;
    }
  }

  hb_codepoint_t get_sid (hb_codepoint_t glyph, unsigned int num_glyphs) const
  {
    if (unlikely (glyph >= num_glyphs)) return 0;
    switch (format)
    {
    case 0: return u.format0.get_sid (glyph);
    case 1: return u.format1.get_sid (glyph);
    case 2: return u.format2.get_sid (glyph);
    default:return 0;
    }
  }

  hb_codepoint_t get_glyph (hb_codepoint_t sid, unsigned int num_glyphs) const
  {
    switch (format)
    {
    case 0: return u.format0.get_glyph (sid, num_glyphs);
    case 1: return u.format1.get_glyph (sid, num_glyphs);
    case 2: return u.format2.get_glyph (sid, num_glyphs);
    default:return 0;
    }
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!c->check_struct (this)))
      return_trace (false);

    switch (format)
    {
    case 0: return_trace (u.format0.sanitize (c, c->get_num_glyphs ()));
    case 1: return_trace (u.format1.sanitize (c, c->get_num_glyphs ()));
    case 2: return_trace (u.format2.sanitize (c, c->get_num_glyphs ()));
    default:return_trace (false);
    }
  }

  HBUINT8       format;
  union {
    Charset0    format0;
    Charset1    format1;
    Charset2    format2;
  } u;

  DEFINE_SIZE_MIN (1);
};

struct CFF1StringIndex : CFF1Index
{
  bool serialize (hb_serialize_context_t *c, const CFF1StringIndex &strings,
		  const hb_inc_bimap_t &sidmap)
  {
    TRACE_SERIALIZE (this);
    if (unlikely ((strings.count == 0) || (sidmap.get_population () == 0)))
    {
      if (unlikely (!c->extend_min (this->count)))
	return_trace (false);
      count = 0;
      return_trace (true);
    }

    byte_str_array_t bytesArray;
    bytesArray.init ();
    if (!bytesArray.resize (sidmap.get_population ()))
      return_trace (false);
    for (unsigned int i = 0; i < strings.count; i++)
    {
      hb_codepoint_t  j = sidmap[i];
      if (j != HB_MAP_VALUE_INVALID)
	bytesArray[j] = strings[i];
    }

    bool result = CFF1Index::serialize (c, bytesArray);
    bytesArray.fini ();
    return_trace (result);
  }
};

struct cff1_top_dict_interp_env_t : num_interp_env_t
{
  cff1_top_dict_interp_env_t ()
    : num_interp_env_t(), prev_offset(0), last_offset(0) {}

  unsigned int prev_offset;
  unsigned int last_offset;
};

struct name_dict_values_t
{
  enum name_dict_val_index_t
  {
      version,
      notice,
      copyright,
      fullName,
      familyName,
      weight,
      postscript,
      fontName,
      baseFontName,
      registry,
      ordering,

      ValCount
  };

  void init ()
  {
    for (unsigned int i = 0; i < ValCount; i++)
      values[i] = CFF_UNDEF_SID;
  }

  unsigned int& operator[] (unsigned int i)
  { assert (i < ValCount); return values[i]; }

  unsigned int operator[] (unsigned int i) const
  { assert (i < ValCount); return values[i]; }

  static enum name_dict_val_index_t name_op_to_index (op_code_t op)
  {
    switch (op) {
      default: // can't happen - just make some compiler happy
      case OpCode_version:
	return version;
      case OpCode_Notice:
	return notice;
      case OpCode_Copyright:
	return copyright;
      case OpCode_FullName:
	return fullName;
      case OpCode_FamilyName:
	return familyName;
      case OpCode_Weight:
	return weight;
      case OpCode_PostScript:
	return postscript;
      case OpCode_FontName:
	return fontName;
      case OpCode_BaseFontName:
	return baseFontName;
    }
  }

  unsigned int  values[ValCount];
};

struct cff1_top_dict_val_t : op_str_t
{
  unsigned int  last_arg_offset;
};

struct cff1_top_dict_values_t : top_dict_values_t<cff1_top_dict_val_t>
{
  void init ()
  {
    top_dict_values_t<cff1_top_dict_val_t>::init ();

    nameSIDs.init ();
    ros_supplement = 0;
    cidCount = 8720;
    EncodingOffset = 0;
    CharsetOffset = 0;
    FDSelectOffset = 0;
    privateDictInfo.init ();
  }
  void fini () { top_dict_values_t<cff1_top_dict_val_t>::fini (); }

  bool is_CID () const
  { return nameSIDs[name_dict_values_t::registry] != CFF_UNDEF_SID; }

  name_dict_values_t  nameSIDs;
  unsigned int    ros_supplement_offset;
  unsigned int    ros_supplement;
  unsigned int    cidCount;

  unsigned int    EncodingOffset;
  unsigned int    CharsetOffset;
  unsigned int    FDSelectOffset;
  table_info_t    privateDictInfo;
};

struct cff1_top_dict_opset_t : top_dict_opset_t<cff1_top_dict_val_t>
{
  static void process_op (op_code_t op, cff1_top_dict_interp_env_t& env, cff1_top_dict_values_t& dictval)
  {
    cff1_top_dict_val_t  val;
    val.last_arg_offset = (env.last_offset-1) - dictval.opStart;  /* offset to the last argument */

    switch (op) {
      case OpCode_version:
      case OpCode_Notice:
      case OpCode_Copyright:
      case OpCode_FullName:
      case OpCode_FamilyName:
      case OpCode_Weight:
      case OpCode_PostScript:
      case OpCode_BaseFontName:
	dictval.nameSIDs[name_dict_values_t::name_op_to_index (op)] = env.argStack.pop_uint ();
	env.clear_args ();
	break;
      case OpCode_isFixedPitch:
      case OpCode_ItalicAngle:
      case OpCode_UnderlinePosition:
      case OpCode_UnderlineThickness:
      case OpCode_PaintType:
      case OpCode_CharstringType:
      case OpCode_UniqueID:
      case OpCode_StrokeWidth:
      case OpCode_SyntheticBase:
      case OpCode_CIDFontVersion:
      case OpCode_CIDFontRevision:
      case OpCode_CIDFontType:
      case OpCode_UIDBase:
      case OpCode_FontBBox:
      case OpCode_XUID:
      case OpCode_BaseFontBlend:
	env.clear_args ();
	break;

      case OpCode_CIDCount:
	dictval.cidCount = env.argStack.pop_uint ();
	env.clear_args ();
	break;

      case OpCode_ROS:
	dictval.ros_supplement = env.argStack.pop_uint ();
	dictval.nameSIDs[name_dict_values_t::ordering] = env.argStack.pop_uint ();
	dictval.nameSIDs[name_dict_values_t::registry] = env.argStack.pop_uint ();
	env.clear_args ();
	break;

      case OpCode_Encoding:
	dictval.EncodingOffset = env.argStack.pop_uint ();
	env.clear_args ();
	if (unlikely (dictval.EncodingOffset == 0)) return;
	break;

      case OpCode_charset:
	dictval.CharsetOffset = env.argStack.pop_uint ();
	env.clear_args ();
	if (unlikely (dictval.CharsetOffset == 0)) return;
	break;

      case OpCode_FDSelect:
	dictval.FDSelectOffset = env.argStack.pop_uint ();
	env.clear_args ();
	break;

      case OpCode_Private:
	dictval.privateDictInfo.offset = env.argStack.pop_uint ();
	dictval.privateDictInfo.size = env.argStack.pop_uint ();
	env.clear_args ();
	break;

      default:
	env.last_offset = env.str_ref.offset;
	top_dict_opset_t<cff1_top_dict_val_t>::process_op (op, env, dictval);
	/* Record this operand below if stack is empty, otherwise done */
	if (!env.argStack.is_empty ()) return;
	break;
    }

    if (unlikely (env.in_error ())) return;

    dictval.add_op (op, env.str_ref, val);
  }
};

struct cff1_font_dict_values_t : dict_values_t<op_str_t>
{
  void init ()
  {
    dict_values_t<op_str_t>::init ();
    privateDictInfo.init ();
    fontName = CFF_UNDEF_SID;
  }
  void fini () { dict_values_t<op_str_t>::fini (); }

  table_info_t       privateDictInfo;
  unsigned int    fontName;
};

struct cff1_font_dict_opset_t : dict_opset_t
{
  static void process_op (op_code_t op, num_interp_env_t& env, cff1_font_dict_values_t& dictval)
  {
    switch (op) {
      case OpCode_FontName:
	dictval.fontName = env.argStack.pop_uint ();
	env.clear_args ();
	break;
      case OpCode_FontMatrix:
      case OpCode_PaintType:
	env.clear_args ();
	break;
      case OpCode_Private:
	dictval.privateDictInfo.offset = env.argStack.pop_uint ();
	dictval.privateDictInfo.size = env.argStack.pop_uint ();
	env.clear_args ();
	break;

      default:
	dict_opset_t::process_op (op, env);
	if (!env.argStack.is_empty ()) return;
	break;
    }

    if (unlikely (env.in_error ())) return;

    dictval.add_op (op, env.str_ref);
  }
};

template <typename VAL>
struct cff1_private_dict_values_base_t : dict_values_t<VAL>
{
  void init ()
  {
    dict_values_t<VAL>::init ();
    subrsOffset = 0;
    localSubrs = &Null (CFF1Subrs);
  }
  void fini () { dict_values_t<VAL>::fini (); }

  unsigned int      subrsOffset;
  const CFF1Subrs    *localSubrs;
};

typedef cff1_private_dict_values_base_t<op_str_t> cff1_private_dict_values_subset_t;
typedef cff1_private_dict_values_base_t<num_dict_val_t> cff1_private_dict_values_t;

struct cff1_private_dict_opset_t : dict_opset_t
{
  static void process_op (op_code_t op, num_interp_env_t& env, cff1_private_dict_values_t& dictval)
  {
    num_dict_val_t val;
    val.init ();

    switch (op) {
      case OpCode_BlueValues:
      case OpCode_OtherBlues:
      case OpCode_FamilyBlues:
      case OpCode_FamilyOtherBlues:
      case OpCode_StemSnapH:
      case OpCode_StemSnapV:
	env.clear_args ();
	break;
      case OpCode_StdHW:
      case OpCode_StdVW:
      case OpCode_BlueScale:
      case OpCode_BlueShift:
      case OpCode_BlueFuzz:
      case OpCode_ForceBold:
      case OpCode_LanguageGroup:
      case OpCode_ExpansionFactor:
      case OpCode_initialRandomSeed:
      case OpCode_defaultWidthX:
      case OpCode_nominalWidthX:
	val.single_val = env.argStack.pop_num ();
	env.clear_args ();
	break;
      case OpCode_Subrs:
	dictval.subrsOffset = env.argStack.pop_uint ();
	env.clear_args ();
	break;

      default:
	dict_opset_t::process_op (op, env);
	if (!env.argStack.is_empty ()) return;
	break;
    }

    if (unlikely (env.in_error ())) return;

    dictval.add_op (op, env.str_ref, val);
  }
};

struct cff1_private_dict_opset_subset : dict_opset_t
{
  static void process_op (op_code_t op, num_interp_env_t& env, cff1_private_dict_values_subset_t& dictval)
  {
    switch (op) {
      case OpCode_BlueValues:
      case OpCode_OtherBlues:
      case OpCode_FamilyBlues:
      case OpCode_FamilyOtherBlues:
      case OpCode_StemSnapH:
      case OpCode_StemSnapV:
      case OpCode_StdHW:
      case OpCode_StdVW:
      case OpCode_BlueScale:
      case OpCode_BlueShift:
      case OpCode_BlueFuzz:
      case OpCode_ForceBold:
      case OpCode_LanguageGroup:
      case OpCode_ExpansionFactor:
      case OpCode_initialRandomSeed:
      case OpCode_defaultWidthX:
      case OpCode_nominalWidthX:
	env.clear_args ();
	break;

      case OpCode_Subrs:
	dictval.subrsOffset = env.argStack.pop_uint ();
	env.clear_args ();
	break;

      default:
	dict_opset_t::process_op (op, env);
	if (!env.argStack.is_empty ()) return;
	break;
    }

    if (unlikely (env.in_error ())) return;

    dictval.add_op (op, env.str_ref);
  }
};

typedef dict_interpreter_t<cff1_top_dict_opset_t, cff1_top_dict_values_t, cff1_top_dict_interp_env_t> cff1_top_dict_interpreter_t;
typedef dict_interpreter_t<cff1_font_dict_opset_t, cff1_font_dict_values_t> cff1_font_dict_interpreter_t;

typedef CFF1Index CFF1NameIndex;
typedef CFF1IndexOf<TopDict> CFF1TopDictIndex;

struct cff1_font_dict_values_mod_t
{
  cff1_font_dict_values_mod_t() { init (); }

  void init () { init ( &Null (cff1_font_dict_values_t), CFF_UNDEF_SID ); }

  void init (const cff1_font_dict_values_t *base_,
	     unsigned int fontName_)
  {
    base = base_;
    fontName = fontName_;
    privateDictInfo.init ();
  }

  unsigned get_count () const { return base->get_count (); }

  const op_str_t &operator [] (unsigned int i) const { return (*base)[i]; }

  const cff1_font_dict_values_t    *base;
  table_info_t		   privateDictInfo;
  unsigned int		fontName;
};

struct CFF1FDArray : FDArray<HBUINT16>
{
  /* FDArray::serialize() requires this partial specialization to compile */
  template <typename ITER, typename OP_SERIALIZER>
  bool serialize (hb_serialize_context_t *c, ITER it, OP_SERIALIZER& opszr)
  { return FDArray<HBUINT16>::serialize<cff1_font_dict_values_mod_t, cff1_font_dict_values_mod_t> (c, it, opszr); }
};

} /* namespace CFF */

namespace OT {

using namespace CFF;

struct cff1
{
  static constexpr hb_tag_t tableTag = HB_OT_TAG_cff1;

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  likely (version.major == 1));
  }

  template <typename PRIVOPSET, typename PRIVDICTVAL>
  struct accelerator_templ_t
  {
    void init (hb_face_t *face)
    {
      topDict.init ();
      fontDicts.init ();
      privateDicts.init ();

      this->blob = sc.reference_table<cff1> (face);

      /* setup for run-time santization */
      sc.init (this->blob);
      sc.start_processing ();

      const OT::cff1 *cff = this->blob->template as<OT::cff1> ();

      if (cff == &Null (OT::cff1))
      { fini (); return; }

      nameIndex = &cff->nameIndex (cff);
      if ((nameIndex == &Null (CFF1NameIndex)) || !nameIndex->sanitize (&sc))
      { fini (); return; }

      topDictIndex = &StructAtOffset<CFF1TopDictIndex> (nameIndex, nameIndex->get_size ());
      if ((topDictIndex == &Null (CFF1TopDictIndex)) || !topDictIndex->sanitize (&sc) || (topDictIndex->count == 0))
      { fini (); return; }

      { /* parse top dict */
	const byte_str_t topDictStr = (*topDictIndex)[0];
	if (unlikely (!topDictStr.sanitize (&sc))) { fini (); return; }
	cff1_top_dict_interpreter_t top_interp;
	top_interp.env.init (topDictStr);
	topDict.init ();
	if (unlikely (!top_interp.interpret (topDict))) { fini (); return; }
      }

      if (is_predef_charset ())
	charset = &Null (Charset);
      else
      {
	charset = &StructAtOffsetOrNull<Charset> (cff, topDict.CharsetOffset);
	if (unlikely ((charset == &Null (Charset)) || !charset->sanitize (&sc))) { fini (); return; }
      }

      fdCount = 1;
      if (is_CID ())
      {
	fdArray = &StructAtOffsetOrNull<CFF1FDArray> (cff, topDict.FDArrayOffset);
	fdSelect = &StructAtOffsetOrNull<CFF1FDSelect> (cff, topDict.FDSelectOffset);
	if (unlikely ((fdArray == &Null (CFF1FDArray)) || !fdArray->sanitize (&sc) ||
	    (fdSelect == &Null (CFF1FDSelect)) || !fdSelect->sanitize (&sc, fdArray->count)))
	{ fini (); return; }

	fdCount = fdArray->count;
      }
      else
      {
	fdArray = &Null (CFF1FDArray);
	fdSelect = &Null (CFF1FDSelect);
      }

      encoding = &Null (Encoding);
      if (is_CID ())
      {
	if (unlikely (charset == &Null (Charset))) { fini (); return; }
      }
      else
      {
	if (!is_predef_encoding ())
	{
	  encoding = &StructAtOffsetOrNull<Encoding> (cff, topDict.EncodingOffset);
	  if (unlikely ((encoding == &Null (Encoding)) || !encoding->sanitize (&sc))) { fini (); return; }
	}
      }

      stringIndex = &StructAtOffset<CFF1StringIndex> (topDictIndex, topDictIndex->get_size ());
      if ((stringIndex == &Null (CFF1StringIndex)) || !stringIndex->sanitize (&sc))
      { fini (); return; }

      globalSubrs = &StructAtOffset<CFF1Subrs> (stringIndex, stringIndex->get_size ());
      if ((globalSubrs != &Null (CFF1Subrs)) && !globalSubrs->sanitize (&sc))
      { fini (); return; }

      charStrings = &StructAtOffsetOrNull<CFF1CharStrings> (cff, topDict.charStringsOffset);

      if ((charStrings == &Null (CFF1CharStrings)) || unlikely (!charStrings->sanitize (&sc)))
      { fini (); return; }

      num_glyphs = charStrings->count;
      if (num_glyphs != sc.get_num_glyphs ())
      { fini (); return; }

      if (unlikely (!privateDicts.resize (fdCount)))
      { fini (); return; }
      for (unsigned int i = 0; i < fdCount; i++)
	privateDicts[i].init ();

      // parse CID font dicts and gather private dicts
      if (is_CID ())
      {
	for (unsigned int i = 0; i < fdCount; i++)
	{
	  byte_str_t fontDictStr = (*fdArray)[i];
	  if (unlikely (!fontDictStr.sanitize (&sc))) { fini (); return; }
	  cff1_font_dict_values_t *font;
	  cff1_font_dict_interpreter_t font_interp;
	  font_interp.env.init (fontDictStr);
	  font = fontDicts.push ();
	  if (unlikely (font == &Crap (cff1_font_dict_values_t))) { fini (); return; }
	  font->init ();
	  if (unlikely (!font_interp.interpret (*font))) { fini (); return; }
	  PRIVDICTVAL *priv = &privateDicts[i];
	  const byte_str_t privDictStr (StructAtOffset<UnsizedByteStr> (cff, font->privateDictInfo.offset), font->privateDictInfo.size);
	  if (unlikely (!privDictStr.sanitize (&sc))) { fini (); return; }
	  dict_interpreter_t<PRIVOPSET, PRIVDICTVAL> priv_interp;
	  priv_interp.env.init (privDictStr);
	  priv->init ();
	  if (unlikely (!priv_interp.interpret (*priv))) { fini (); return; }

	  priv->localSubrs = &StructAtOffsetOrNull<CFF1Subrs> (&privDictStr, priv->subrsOffset);
	  if (priv->localSubrs != &Null (CFF1Subrs) &&
	      unlikely (!priv->localSubrs->sanitize (&sc)))
	  { fini (); return; }
	}
      }
      else  /* non-CID */
      {
	cff1_top_dict_values_t *font = &topDict;
	PRIVDICTVAL *priv = &privateDicts[0];

	const byte_str_t privDictStr (StructAtOffset<UnsizedByteStr> (cff, font->privateDictInfo.offset), font->privateDictInfo.size);
	if (unlikely (!privDictStr.sanitize (&sc))) { fini (); return; }
	dict_interpreter_t<PRIVOPSET, PRIVDICTVAL> priv_interp;
	priv_interp.env.init (privDictStr);
	priv->init ();
	if (unlikely (!priv_interp.interpret (*priv))) { fini (); return; }

	priv->localSubrs = &StructAtOffsetOrNull<CFF1Subrs> (&privDictStr, priv->subrsOffset);
	if (priv->localSubrs != &Null (CFF1Subrs) &&
	    unlikely (!priv->localSubrs->sanitize (&sc)))
	{ fini (); return; }
      }
    }

    void fini ()
    {
      sc.end_processing ();
      topDict.fini ();
      fontDicts.fini_deep ();
      privateDicts.fini_deep ();
      hb_blob_destroy (blob);
      blob = nullptr;
    }

    bool is_valid () const { return blob; }
    bool   is_CID () const { return topDict.is_CID (); }

    bool is_predef_charset () const { return topDict.CharsetOffset <= ExpertSubsetCharset; }

    unsigned int std_code_to_glyph (hb_codepoint_t code) const
    {
      hb_codepoint_t sid = lookup_standard_encoding_for_sid (code);
      if (unlikely (sid == CFF_UNDEF_SID))
	return 0;

      if (charset != &Null (Charset))
	return charset->get_glyph (sid, num_glyphs);
      else if ((topDict.CharsetOffset == ISOAdobeCharset)
	      && (code <= 228 /*zcaron*/)) return sid;
      return 0;
    }

    bool is_predef_encoding () const { return topDict.EncodingOffset <= ExpertEncoding; }

    hb_codepoint_t glyph_to_code (hb_codepoint_t glyph) const
    {
      if (encoding != &Null (Encoding))
	return encoding->get_code (glyph);
      else
      {
	hb_codepoint_t sid = glyph_to_sid (glyph);
	if (sid == 0) return 0;
	hb_codepoint_t code = 0;
	switch (topDict.EncodingOffset)
	{
	case StandardEncoding:
	  code = lookup_standard_encoding_for_code (sid);
	  break;
	case ExpertEncoding:
	  code = lookup_expert_encoding_for_code (sid);
	  break;
	default:
	  break;
	}
	return code;
      }
    }

    hb_codepoint_t glyph_to_sid (hb_codepoint_t glyph) const
    {
      if (charset != &Null (Charset))
	return charset->get_sid (glyph, num_glyphs);
      else
      {
	hb_codepoint_t sid = 0;
	switch (topDict.CharsetOffset)
	{
	  case ISOAdobeCharset:
	    if (glyph <= 228 /*zcaron*/) sid = glyph;
	    break;
	  case ExpertCharset:
	    sid = lookup_expert_charset_for_sid (glyph);
	    break;
	  case ExpertSubsetCharset:
	      sid = lookup_expert_subset_charset_for_sid (glyph);
	    break;
	  default:
	    break;
	}
	return sid;
      }
    }

    hb_codepoint_t sid_to_glyph (hb_codepoint_t sid) const
    {
      if (charset != &Null (Charset))
	return charset->get_glyph (sid, num_glyphs);
      else
      {
	hb_codepoint_t glyph = 0;
	switch (topDict.CharsetOffset)
	{
	  case ISOAdobeCharset:
	    if (sid <= 228 /*zcaron*/) glyph = sid;
	    break;
	  case ExpertCharset:
	    glyph = lookup_expert_charset_for_glyph (sid);
	    break;
	  case ExpertSubsetCharset:
	    glyph = lookup_expert_subset_charset_for_glyph (sid);
	    break;
	  default:
	    break;
	}
	return glyph;
      }
    }

    protected:
    hb_blob_t	           *blob;
    hb_sanitize_context_t   sc;

    public:
    const Encoding	    *encoding;
    const Charset	    *charset;
    const CFF1NameIndex     *nameIndex;
    const CFF1TopDictIndex  *topDictIndex;
    const CFF1StringIndex   *stringIndex;
    const CFF1Subrs	    *globalSubrs;
    const CFF1CharStrings   *charStrings;
    const CFF1FDArray       *fdArray;
    const CFF1FDSelect      *fdSelect;
    unsigned int	     fdCount;

    cff1_top_dict_values_t   topDict;
    hb_vector_t<cff1_font_dict_values_t>
			     fontDicts;
    hb_vector_t<PRIVDICTVAL> privateDicts;

    unsigned int	     num_glyphs;
  };

  struct accelerator_t : accelerator_templ_t<cff1_private_dict_opset_t, cff1_private_dict_values_t>
  {
    void init (hb_face_t *face)
    {
      SUPER::init (face);

      if (!is_valid ()) return;
      if (is_CID ()) return;

      /* fill glyph_names */
      for (hb_codepoint_t gid = 0; gid < num_glyphs; gid++)
      {
	hb_codepoint_t	sid = glyph_to_sid (gid);
	gname_t	gname;
	gname.sid = sid;
	if (sid < cff1_std_strings_length)
	  gname.name = cff1_std_strings (sid);
	else
	{
	  byte_str_t	ustr = (*stringIndex)[sid - cff1_std_strings_length];
	  gname.name = hb_bytes_t ((const char*)ustr.arrayZ, ustr.length);
	}
	if (unlikely (!gname.name.arrayZ)) { fini (); return; }
	glyph_names.push (gname);
      }
      glyph_names.qsort ();
    }

    void fini ()
    {
      glyph_names.fini ();

      SUPER::fini ();
    }

    bool get_glyph_name (hb_codepoint_t glyph,
			 char *buf, unsigned int buf_len) const
    {
      if (!buf) return true;
      if (unlikely (!is_valid ())) return false;
      if (is_CID()) return false;
      hb_codepoint_t sid = glyph_to_sid (glyph);
      const char *str;
      size_t str_len;
      if (sid < cff1_std_strings_length)
      {
	hb_bytes_t byte_str = cff1_std_strings (sid);
	str = byte_str.arrayZ;
	str_len = byte_str.length;
      }
      else
      {
	byte_str_t ubyte_str = (*stringIndex)[sid - cff1_std_strings_length];
	str = (const char *)ubyte_str.arrayZ;
	str_len = ubyte_str.length;
      }
      if (!str_len) return false;
      unsigned int len = hb_min (buf_len - 1, str_len);
      strncpy (buf, (const char*)str, len);
      buf[len] = '\0';
      return true;
    }

    bool get_glyph_from_name (const char *name, int len,
			      hb_codepoint_t *glyph) const
    {
      if (len < 0) len = strlen (name);
      if (unlikely (!len)) return false;

      gname_t key = { hb_bytes_t (name, len), 0 };
      const gname_t *gname = glyph_names.bsearch (key);
      if (!gname) return false;
      hb_codepoint_t gid = sid_to_glyph (gname->sid);
      if (!gid && gname->sid) return false;
      *glyph = gid;
      return true;
    }

    HB_INTERNAL bool get_extents (hb_font_t *font, hb_codepoint_t glyph, hb_glyph_extents_t *extents) const;
    HB_INTERNAL bool get_seac_components (hb_codepoint_t glyph, hb_codepoint_t *base, hb_codepoint_t *accent) const;
#ifdef HB_EXPERIMENTAL_API
    HB_INTERNAL bool get_path (hb_font_t *font, hb_codepoint_t glyph, draw_helper_t &draw_helper) const;
#endif

    private:
    struct gname_t
    {
      hb_bytes_t	name;
      uint16_t		sid;

      static int cmp (const void *a_, const void *b_)
      {
	const gname_t *a = (const gname_t *)a_;
	const gname_t *b = (const gname_t *)b_;
	int minlen = hb_min (a->name.length, b->name.length);
	int ret = strncmp (a->name.arrayZ, b->name.arrayZ, minlen);
	if (ret) return ret;
	return a->name.length - b->name.length;
      }

      int cmp (const gname_t &a) const { return cmp (&a, this); }
    };

    hb_sorted_vector_t<gname_t>	glyph_names;

    typedef accelerator_templ_t<cff1_private_dict_opset_t, cff1_private_dict_values_t> SUPER;
  };

  struct accelerator_subset_t : accelerator_templ_t<cff1_private_dict_opset_subset, cff1_private_dict_values_subset_t> {};

  bool subset (hb_subset_context_t *c) const { return hb_subset_cff1 (c); }

  protected:
  HB_INTERNAL static hb_codepoint_t lookup_standard_encoding_for_code (hb_codepoint_t sid);
  HB_INTERNAL static hb_codepoint_t lookup_expert_encoding_for_code (hb_codepoint_t sid);
  HB_INTERNAL static hb_codepoint_t lookup_expert_charset_for_sid (hb_codepoint_t glyph);
  HB_INTERNAL static hb_codepoint_t lookup_expert_subset_charset_for_sid (hb_codepoint_t glyph);
  HB_INTERNAL static hb_codepoint_t lookup_expert_charset_for_glyph (hb_codepoint_t sid);
  HB_INTERNAL static hb_codepoint_t lookup_expert_subset_charset_for_glyph (hb_codepoint_t sid);
  HB_INTERNAL static hb_codepoint_t lookup_standard_encoding_for_sid (hb_codepoint_t code);

  public:
  FixedVersion<HBUINT8> version;	  /* Version of CFF table. set to 0x0100u */
  OffsetTo<CFF1NameIndex, HBUINT8> nameIndex; /* headerSize = Offset to Name INDEX. */
  HBUINT8	       offSize;	  /* offset size (unused?) */

  public:
  DEFINE_SIZE_STATIC (4);
};

struct cff1_accelerator_t : cff1::accelerator_t {};
} /* namespace OT */

#endif /* HB_OT_CFF1_TABLE_HH */
