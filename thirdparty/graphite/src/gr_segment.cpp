// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2010, SIL International, All rights reserved.

#include "graphite2/Segment.h"
#include "inc/UtfCodec.h"
#include "inc/Segment.h"

using namespace graphite2;

namespace
{

  gr_segment* makeAndInitialize(const Font *font, const Face *face, uint32 script, const Features* pFeats/*must not be NULL*/, gr_encform enc, const void* pStart, size_t nChars, int dir)
  {
      if (script == 0x20202020) script = 0;
      else if ((script & 0x00FFFFFF) == 0x00202020) script = script & 0xFF000000;
      else if ((script & 0x0000FFFF) == 0x00002020) script = script & 0xFFFF0000;
      else if ((script & 0x000000FF) == 0x00000020) script = script & 0xFFFFFF00;
      // if (!font) return NULL;
      Segment* pRes=new Segment(nChars, face, script, dir);


      if (!pRes->read_text(face, pFeats, enc, pStart, nChars) || !pRes->runGraphite())
      {
        delete pRes;
        return NULL;
      }
      pRes->finalise(font, true);

      return static_cast<gr_segment*>(pRes);
  }

  template <typename utf_iter>
  inline size_t count_unicode_chars(utf_iter first, const utf_iter last, const void **error)
  {
      size_t n_chars = 0;
      uint32 usv = 0;

      if (last)
      {
          if (!first.validate(last))
          {
              if (error)  *error = last - 1;
              return 0;
          }
          for (;first != last; ++first, ++n_chars)
              if ((usv = *first) == 0 || first.error()) break;
      }
      else
      {
          while ((usv = *first) != 0 && !first.error())
          {
              ++first;
              ++n_chars;
          }
      }

      if (error)  *error = first.error() ? first : 0;
      return n_chars;
  }
}


extern "C" {

size_t gr_count_unicode_characters(gr_encform enc, const void* buffer_begin, const void* buffer_end/*don't go on or past end, If NULL then ignored*/, const void** pError)   //Also stops on nul. Any nul is not in the count
{
    assert(buffer_begin);

    switch (enc)
    {
    case gr_utf8:   return count_unicode_chars<utf8::const_iterator>(buffer_begin, buffer_end, pError); break;
    case gr_utf16:  return count_unicode_chars<utf16::const_iterator>(buffer_begin, buffer_end, pError); break;
    case gr_utf32:  return count_unicode_chars<utf32::const_iterator>(buffer_begin, buffer_end, pError); break;
    default:        return 0;
    }
}


gr_segment* gr_make_seg(const gr_font *font, const gr_face *face, gr_uint32 script, const gr_feature_val* pFeats, gr_encform enc, const void* pStart, size_t nChars, int dir)
{
    if (!face) return nullptr;

    const gr_feature_val * tmp_feats = 0;
    if (pFeats == 0)
        pFeats = tmp_feats = static_cast<const gr_feature_val*>(face->theSill().cloneFeatures(0));
    gr_segment * seg = makeAndInitialize(font, face, script, pFeats, enc, pStart, nChars, dir);
    delete static_cast<const FeatureVal*>(tmp_feats);

    return seg;
}


void gr_seg_destroy(gr_segment* p)
{
    delete static_cast<Segment*>(p);
}


float gr_seg_advance_X(const gr_segment* pSeg/*not NULL*/)
{
    assert(pSeg);
    return pSeg->advance().x;
}


float gr_seg_advance_Y(const gr_segment* pSeg/*not NULL*/)
{
    assert(pSeg);
    return pSeg->advance().y;
}


unsigned int gr_seg_n_cinfo(const gr_segment* pSeg/*not NULL*/)
{
    assert(pSeg);
    return static_cast<unsigned int>(pSeg->charInfoCount());
}


const gr_char_info* gr_seg_cinfo(const gr_segment* pSeg/*not NULL*/, unsigned int index/*must be <number_of_CharInfo*/)
{
    assert(pSeg);
    return static_cast<const gr_char_info*>(pSeg->charinfo(index));
}

unsigned int gr_seg_n_slots(const gr_segment* pSeg/*not NULL*/)
{
    assert(pSeg);
    return static_cast<unsigned int>(pSeg->slotCount());
}

const gr_slot* gr_seg_first_slot(gr_segment* pSeg/*not NULL*/)
{
    assert(pSeg);
    return static_cast<const gr_slot*>(pSeg->first());
}

const gr_slot* gr_seg_last_slot(gr_segment* pSeg/*not NULL*/)
{
    assert(pSeg);
    return static_cast<const gr_slot*>(pSeg->last());
}

float gr_seg_justify(gr_segment* pSeg/*not NULL*/, const gr_slot* pSlot/*not NULL*/, const gr_font *pFont, double width, enum gr_justFlags flags, const gr_slot *pFirst, const gr_slot *pLast)
{
    assert(pSeg);
    assert(pSlot);
    return pSeg->justify(const_cast<gr_slot *>(pSlot), pFont, float(width), justFlags(flags), const_cast<gr_slot *>(pFirst), const_cast<gr_slot *>(pLast));
}

} // extern "C"
