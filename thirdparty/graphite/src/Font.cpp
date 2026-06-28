// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2010, SIL International, All rights reserved.

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
