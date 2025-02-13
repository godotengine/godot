// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2010, SIL International, All rights reserved.

#include <cstring>

#include "inc/Main.h"
#include "inc/bits.h"
#include "inc/Endian.h"
#include "inc/FeatureMap.h"
#include "inc/FeatureVal.h"
#include "graphite2/Font.h"
#include "inc/TtfUtil.h"
#include <cstdlib>
#include "inc/Face.h"


using namespace graphite2;

namespace
{
    static int cmpNameAndFeatures(const void *ap, const void *bp)
    {
        const NameAndFeatureRef & a = *static_cast<const NameAndFeatureRef *>(ap),
                                & b = *static_cast<const NameAndFeatureRef *>(bp);
        return (a < b ? -1 : (b < a ? 1 : 0));
    }

    const size_t    FEAT_HEADER     = sizeof(uint32) + 2*sizeof(uint16) + sizeof(uint32),
                    FEATURE_SIZE    = sizeof(uint32)
                                    + 2*sizeof(uint16)
                                    + sizeof(uint32)
                                    + 2*sizeof(uint16),
                    FEATURE_SETTING_SIZE = sizeof(int16) + sizeof(uint16);

    uint16 readFeatureSettings(const byte * p, FeatureSetting * s, size_t num_settings)
    {
        uint16 max_val = 0;
        for (FeatureSetting * const end = s + num_settings; s != end; ++s)
        {
            const int16 value = be::read<int16>(p);
            ::new (s) FeatureSetting(value, be::read<uint16>(p));
            if (uint16(value) > max_val)    max_val = value;
        }

        return max_val;
    }
}

FeatureRef::FeatureRef(const Face & face,
    unsigned short & bits_offset, uint32 max_val,
    uint32 name, uint16 uiName, flags_t flags,
    FeatureSetting *settings, uint16 num_set) throw()
: m_face(&face),
  m_nameValues(settings),
  m_mask(mask_over_val(max_val)),
  m_max(max_val),
  m_id(name),
  m_nameid(uiName),
  m_numSet(num_set),
  m_flags(flags)
{
    const uint8 need_bits = bit_set_count(m_mask);
    m_index = (bits_offset + need_bits) / SIZEOF_CHUNK;
    if (m_index > bits_offset / SIZEOF_CHUNK)
        bits_offset = m_index*SIZEOF_CHUNK;
    m_bits = bits_offset % SIZEOF_CHUNK;
    bits_offset += need_bits;
    m_mask <<= m_bits;
}

FeatureRef::~FeatureRef() throw()
{
    free(m_nameValues);
}

bool FeatureMap::readFeats(const Face & face)
{
    const Face::Table feat(face, TtfUtil::Tag::Feat);
    const byte * p = feat;
    if (!p) return true;
    if (feat.size() < FEAT_HEADER) return false;

    const byte *const feat_start = p,
               *const feat_end = p + feat.size();

    const uint32 version = be::read<uint32>(p);
    m_numFeats = be::read<uint16>(p);
    be::skip<uint16>(p);
    be::skip<uint32>(p);

    // Sanity checks
    if (m_numFeats == 0)    return true;
    if (version < 0x00010000 ||
        p + m_numFeats*FEATURE_SIZE > feat_end)
    {   //defensive
        m_numFeats = 0;
        return false;
    }

    m_feats = new FeatureRef [m_numFeats];
    uint16 * const  defVals = gralloc<uint16>(m_numFeats);
    if (!defVals || !m_feats) return false;
    unsigned short bits = 0;     //to cause overflow on first Feature

    for (int i = 0, ie = m_numFeats; i != ie; i++)
    {
        const uint32    label   = version < 0x00020000 ? be::read<uint16>(p) : be::read<uint32>(p);
        const uint16    num_settings = be::read<uint16>(p);
        if (version >= 0x00020000)
            be::skip<uint16>(p);
        const uint32    settings_offset = be::read<uint32>(p);
        const uint16    flags  = be::read<uint16>(p),
                        uiName = be::read<uint16>(p);

        if (settings_offset > size_t(feat_end - feat_start)
            || settings_offset + num_settings * FEATURE_SETTING_SIZE > size_t(feat_end - feat_start))
        {
            free(defVals);
            return false;
        }

        FeatureSetting *uiSet;
        uint32 maxVal;
        if (num_settings != 0)
        {
            uiSet = gralloc<FeatureSetting>(num_settings);
            if (!uiSet)
            {
                free(defVals);
                return false;
            }
            maxVal = readFeatureSettings(feat_start + settings_offset, uiSet, num_settings);
            defVals[i] = uiSet[0].value();
        }
        else
        {
            uiSet = 0;
            maxVal = 0xffffffff;
            defVals[i] = 0;
        }

        ::new (m_feats + i) FeatureRef (face, bits, maxVal,
                                       label, uiName, 
                                       FeatureRef::flags_t(flags),
                                       uiSet, num_settings);
    }
    new (&m_defaultFeatures) Features(bits/(sizeof(uint32)*8) + 1, *this);
    m_pNamedFeats = new NameAndFeatureRef[m_numFeats];
    if (!m_pNamedFeats)
    {
        free(defVals);
        return false;
    }
    for (int i = 0; i < m_numFeats; ++i)
    {
        m_feats[i].applyValToFeature(defVals[i], m_defaultFeatures);
        m_pNamedFeats[i] = m_feats[i];
    }

    free(defVals);

    qsort(m_pNamedFeats, m_numFeats, sizeof(NameAndFeatureRef), &cmpNameAndFeatures);

    return true;
}

bool SillMap::readFace(const Face & face)
{
    if (!m_FeatureMap.readFeats(face)) return false;
    if (!readSill(face)) return false;
    return true;
}


bool SillMap::readSill(const Face & face)
{
    const Face::Table sill(face, TtfUtil::Tag::Sill);
    const byte *p = sill;

    if (!p) return true;
    if (sill.size() < 12) return false;
    if (be::read<uint32>(p) != 0x00010000UL) return false;
    m_numLanguages = be::read<uint16>(p);
    m_langFeats = new LangFeaturePair[m_numLanguages];
    if (!m_langFeats || !m_FeatureMap.m_numFeats) { m_numLanguages = 0; return true; }        //defensive

    p += 6;     // skip the fast search
    if (sill.size() < m_numLanguages * 8U + 12) return false;

    for (int i = 0; i < m_numLanguages; i++)
    {
        uint32 langid = be::read<uint32>(p);
        uint16 numSettings = be::read<uint16>(p);
        uint16 offset = be::read<uint16>(p);
        if (offset + 8U * numSettings > sill.size() && numSettings > 0) return false;
        Features* feats = new Features(m_FeatureMap.m_defaultFeatures);
        if (!feats) return false;
        const byte *pLSet = sill + offset;

        // Apply langauge specific settings
        for (int j = 0; j < numSettings; j++)
        {
            uint32 name = be::read<uint32>(pLSet);
            uint16 val = be::read<uint16>(pLSet);
            pLSet += 2;
            const FeatureRef* pRef = m_FeatureMap.findFeatureRef(name);
            if (pRef)   pRef->applyValToFeature(val, *feats);
        }
        // Add the language id feature which is always feature id 1
        const FeatureRef* pRef = m_FeatureMap.findFeatureRef(1);
        if (pRef)   pRef->applyValToFeature(langid, *feats);

        m_langFeats[i].m_lang = langid;
        m_langFeats[i].m_pFeatures = feats;
    }
    return true;
}


Features* SillMap::cloneFeatures(uint32 langname/*0 means default*/) const
{
    if (langname)
    {
        // the number of languages in a font is usually small e.g. 8 in Doulos
        // so this loop is not very expensive
        for (uint16 i = 0; i < m_numLanguages; i++)
        {
            if (m_langFeats[i].m_lang == langname)
                return new Features(*m_langFeats[i].m_pFeatures);
        }
    }
    return new Features (m_FeatureMap.m_defaultFeatures);
}



const FeatureRef *FeatureMap::findFeatureRef(uint32 name) const
{
    NameAndFeatureRef *it;

    for (it = m_pNamedFeats; it < m_pNamedFeats + m_numFeats; ++it)
        if (it->m_name == name)
            return it->m_pFRef;
    return NULL;
}

bool FeatureRef::applyValToFeature(uint32 val, Features & pDest) const
{
    if (val>maxVal() || !m_face)
      return false;
    if (pDest.m_pMap==NULL)
      pDest.m_pMap = &m_face->theSill().theFeatureMap();
    else
      if (pDest.m_pMap!=&m_face->theSill().theFeatureMap())
        return false;       //incompatible
    if (m_index >= pDest.size())
        pDest.resize(m_index+1);
    pDest[m_index] &= ~m_mask;
    pDest[m_index] |= (uint32(val) << m_bits);
    return true;
}

uint32 FeatureRef::getFeatureVal(const Features& feats) const
{
  if (m_index < feats.size() && m_face
      && &m_face->theSill().theFeatureMap()==feats.m_pMap)
    return (feats[m_index] & m_mask) >> m_bits;
  else
    return 0;
}
