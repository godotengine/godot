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
#include <cassert>
#include "graphite2/Font.h"
#include "inc/Main.h"
#include "inc/Face.h"

namespace graphite2 {

#define INVALID_ADVANCE -1e38f      // can't be a static const because non-integral

class Font
{
public:
    Font(float ppm, const Face & face, const void * appFontHandle=0, const gr_font_ops * ops=0);
    virtual ~Font();

    float advance(unsigned short glyphid) const;
    float scale() const;
    bool isHinted() const;
    const Face & face() const;
    operator bool () const throw()  { return m_advances; }

    CLASS_NEW_DELETE;
private:
    gr_font_ops         m_ops;
    const void  * const m_appFontHandle;
    float             * m_advances;  // One advance per glyph in pixels. Nan if not defined
    const Face        & m_face;
    float               m_scale;      // scales from design units to ppm
    bool                m_hinted;

    Font(const Font&);
    Font& operator=(const Font&);
};

inline
float Font::advance(unsigned short glyphid) const
{
    if (m_advances[glyphid] == INVALID_ADVANCE)
        m_advances[glyphid] = (*m_ops.glyph_advance_x)(m_appFontHandle, glyphid);
    return m_advances[glyphid];
}

inline
float Font::scale() const
{
    return m_scale;
}

inline
bool Font::isHinted() const
{
    return m_hinted;
}

inline
const Face & Font::face() const
{
    return m_face;
}

} // namespace graphite2

struct gr_font : public graphite2::Font {};
