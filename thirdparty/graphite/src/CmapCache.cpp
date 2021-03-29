/*  GRAPHITE2 LICENSING

    Copyright 2010, SIL International
    All rights reserved.

    This library is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published
    by the Free Software Foundation; either version 2.1 of License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should also have received a copy of the GNU Lesser General Public
    License along with this library in the file named "LICENSE".
    If not, write to the Free Software Foundation, 51 Franklin Street,
    Suite 500, Boston, MA 02110-1335, USA or visit their web page on the
    internet at http://www.fsf.org/licenses/lgpl.html.

Alternatively, the contents of this file may be used under the terms of the
Mozilla Public License (http://mozilla.org/MPL) or the GNU General Public
License, as published by the Free Software Foundation, either version 2
of the License or (at your option) any later version.
*/

#include "inc/Main.h"
#include "inc/CmapCache.h"
#include "inc/Face.h"
#include "inc/TtfTypes.h"
#include "inc/TtfUtil.h"


using namespace graphite2;

const void * bmp_subtable(const Face::Table & cmap)
{
    const void * stbl;
    if (!cmap.size()) return 0;
    if (TtfUtil::CheckCmapSubtable4(stbl = TtfUtil::FindCmapSubtable(cmap, 3, 1, cmap.size()), cmap + cmap.size())
     || TtfUtil::CheckCmapSubtable4(stbl = TtfUtil::FindCmapSubtable(cmap, 0, 3, cmap.size()), cmap + cmap.size())
     || TtfUtil::CheckCmapSubtable4(stbl = TtfUtil::FindCmapSubtable(cmap, 0, 2, cmap.size()), cmap + cmap.size())
     || TtfUtil::CheckCmapSubtable4(stbl = TtfUtil::FindCmapSubtable(cmap, 0, 1, cmap.size()), cmap + cmap.size())
     || TtfUtil::CheckCmapSubtable4(stbl = TtfUtil::FindCmapSubtable(cmap, 0, 0, cmap.size()), cmap + cmap.size()))
        return stbl;
    return 0;
}

const void * smp_subtable(const Face::Table & cmap)
{
    const void * stbl;
    if (!cmap.size()) return 0;
    if (TtfUtil::CheckCmapSubtable12(stbl = TtfUtil::FindCmapSubtable(cmap, 3, 10, cmap.size()), cmap + cmap.size())
     || TtfUtil::CheckCmapSubtable12(stbl = TtfUtil::FindCmapSubtable(cmap, 0, 4, cmap.size()), cmap + cmap.size()))
        return stbl;
    return 0;
}

template <unsigned int (*NextCodePoint)(const void *, unsigned int, int *),
          uint16 (*LookupCodePoint)(const void *, unsigned int, int)>
bool cache_subtable(uint16 * blocks[], const void * cst, const unsigned int limit)
{
    int rangeKey = 0;
    uint32          codePoint = NextCodePoint(cst, 0, &rangeKey),
                    prevCodePoint = 0;
    while (codePoint < limit)
    {
        unsigned int block = codePoint >> 8;
        if (!blocks[block])
        {
            blocks[block] = grzeroalloc<uint16>(0x100);
            if (!blocks[block])
                return false;
        }
        blocks[block][codePoint & 0xFF] = LookupCodePoint(cst, codePoint, rangeKey);
        // prevent infinite loop
        if (codePoint <= prevCodePoint)
            codePoint = prevCodePoint + 1;
        prevCodePoint = codePoint;
        codePoint =  NextCodePoint(cst, codePoint, &rangeKey);
    }
    return true;
}


CachedCmap::CachedCmap(const Face & face)
: m_isBmpOnly(true),
  m_blocks(0)
{
    const Face::Table cmap(face, Tag::cmap);
    if (!cmap)  return;

    const void * bmp_cmap = bmp_subtable(cmap);
    const void * smp_cmap = smp_subtable(cmap);
    m_isBmpOnly = !smp_cmap;

    m_blocks = grzeroalloc<uint16 *>(m_isBmpOnly ? 0x100 : 0x1100);
    if (m_blocks && smp_cmap)
    {
        if (!cache_subtable<TtfUtil::CmapSubtable12NextCodepoint, TtfUtil::CmapSubtable12Lookup>(m_blocks, smp_cmap, 0x10FFFF))
            return;
    }

    if (m_blocks && bmp_cmap)
    {
        if (!cache_subtable<TtfUtil::CmapSubtable4NextCodepoint, TtfUtil::CmapSubtable4Lookup>(m_blocks, bmp_cmap, 0xFFFF))
            return;
    }
}

CachedCmap::~CachedCmap() throw()
{
    if (!m_blocks) return;
    unsigned int numBlocks = (m_isBmpOnly)? 0x100 : 0x1100;
    for (unsigned int i = 0; i < numBlocks; i++)
        free(m_blocks[i]);
    free(m_blocks);
}

uint16 CachedCmap::operator [] (const uint32 usv) const throw()
{
    if ((m_isBmpOnly && usv > 0xFFFF) || (usv > 0x10FFFF))
        return 0;
    const uint32 block = 0xFFFF & (usv >> 8);
    if (m_blocks[block])
        return m_blocks[block][usv & 0xFF];
    return 0;
};

CachedCmap::operator bool() const throw()
{
    return m_blocks != 0;
}


DirectCmap::DirectCmap(const Face & face)
: _cmap(face, Tag::cmap),
  _smp(smp_subtable(_cmap)),
  _bmp(bmp_subtable(_cmap))
{
}

uint16 DirectCmap::operator [] (const uint32 usv) const throw()
{
    return usv > 0xFFFF
            ? (_smp ? TtfUtil::CmapSubtable12Lookup(_smp, usv, 0) : 0)
            : TtfUtil::CmapSubtable4Lookup(_bmp, usv, 0);
}

DirectCmap::operator bool () const throw()
{
    return _cmap && _bmp;
}

