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
#include "inc/Face.h"
#include "inc/Font.h"
#include "inc/GlyphCache.h"

using namespace graphite2;

Font::Font(float ppm, const Face & f, const void * appFontHandle, const gr_font_ops * ops)
: m_appFontHandle(appFontHandle ? appFontHandle : this),
  m_face(f),
  m_scale(ppm / f.glyphs().unitsPerEm()),
  m_hinted(appFontHandle && ops && (ops->glyph_advance_x || ops->glyph_advance_y))
{
    memset(&m_ops, 0, sizeof m_ops);
    if (m_hinted && ops)
        memcpy(&m_ops, ops, min(sizeof m_ops, ops->size));
    else
        m_ops.glyph_advance_x = &Face::default_glyph_advance;

    size_t nGlyphs = f.glyphs().numGlyphs();
    m_advances = gralloc<float>(nGlyphs);
    if (m_advances)
    {
        for (float *advp = m_advances; nGlyphs; --nGlyphs, ++advp)
            *advp = INVALID_ADVANCE;
    }
}


/*virtual*/ Font::~Font()
{
    free(m_advances);
}
