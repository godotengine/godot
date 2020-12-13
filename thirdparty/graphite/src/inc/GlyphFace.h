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
#pragma once

#include "inc/Main.h"
#include "inc/Position.h"
#include "inc/Sparse.h"

namespace graphite2 {

enum metrics {
    kgmetLsb = 0, kgmetRsb,
    kgmetBbTop, kgmetBbBottom, kgmetBbLeft, kgmetBbRight,
    kgmetBbHeight, kgmetBbWidth,
    kgmetAdvWidth, kgmetAdvHeight,
    kgmetAscent, kgmetDescent
};


class GlyphFace
{
public:
    GlyphFace();
    template<typename I>
    GlyphFace(const Rect & bbox, const Position & adv, I first, const I last);

    const Position    & theAdvance() const;
    const Rect        & theBBox() const { return m_bbox; }
    const sparse      & attrs() const { return m_attrs; }
    int32               getMetric(uint8 metric) const;

    CLASS_NEW_DELETE;
private:
    Rect     m_bbox;        // bounding box metrics in design units
    Position m_advance;     // Advance width and height in design units
    sparse   m_attrs;
};


// Inlines: class GlyphFace
//
inline
GlyphFace::GlyphFace()
{}

template<typename I>
GlyphFace::GlyphFace(const Rect & bbox, const Position & adv, I first, const I last)
: m_bbox(bbox),
  m_advance(adv),
  m_attrs(first, last)
{
}

inline
const Position & GlyphFace::theAdvance() const {
    return m_advance;
}

} // namespace graphite2
