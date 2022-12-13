// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2012, SIL International, All rights reserved.


#include "inc/Segment.h"
#include "graphite2/Font.h"
#include "inc/debug.h"
#include "inc/CharInfo.h"
#include "inc/Slot.h"
#include "inc/Main.h"
#include <cmath>

using namespace graphite2;

class JustifyTotal {
public:
    JustifyTotal() : m_numGlyphs(0), m_tStretch(0), m_tShrink(0), m_tStep(0), m_tWeight(0) {}
    void accumulate(Slot *s, Segment *seg, int level);
    int weight() const { return m_tWeight; }

    CLASS_NEW_DELETE

private:
    int m_numGlyphs;
    int m_tStretch;
    int m_tShrink;
    int m_tStep;
    int m_tWeight;
};

void JustifyTotal::accumulate(Slot *s, Segment *seg, int level)
{
    ++m_numGlyphs;
    m_tStretch += s->getJustify(seg, level, 0);
    m_tShrink += s->getJustify(seg, level, 1);
    m_tStep += s->getJustify(seg, level, 2);
    m_tWeight += s->getJustify(seg, level, 3);
}

float Segment::justify(Slot *pSlot, const Font *font, float width, GR_MAYBE_UNUSED justFlags jflags, Slot *pFirst, Slot *pLast)
{
    Slot *end = last();
    float currWidth = 0.0;
    const float scale = font ? font->scale() : 1.0f;
    Position res;

    if (width < 0 && !(silf()->flags()))
        return width;

    if ((m_dir & 1) != m_silf->dir() && m_silf->bidiPass() != m_silf->numPasses())
    {
        reverseSlots();
        std::swap(pFirst, pLast);
    }
    if (!pFirst) pFirst = pSlot;
    while (!pFirst->isBase()) pFirst = pFirst->attachedTo();
    if (!pLast) pLast = last();
    while (!pLast->isBase()) pLast = pLast->attachedTo();
    const float base = pFirst->origin().x / scale;
    width = width / scale;
    if ((jflags & gr_justEndInline) == 0)
    {
        while (pLast != pFirst && pLast)
        {
            Rect bbox = theGlyphBBoxTemporary(pLast->glyph());
            if (bbox.bl.x != 0.f || bbox.bl.y != 0.f || bbox.tr.x != 0.f || bbox.tr.y == 0.f)
                break;
            pLast = pLast->prev();
        }
    }

    if (pLast)
        end = pLast->nextSibling();
    if (pFirst)
        pFirst = pFirst->nextSibling();

    int icount = 0;
    int numLevels = silf()->numJustLevels();
    if (!numLevels)
    {
        for (Slot *s = pSlot; s && s != end; s = s->nextSibling())
        {
            CharInfo *c = charinfo(s->before());
            if (isWhitespace(c->unicodeChar()))
            {
                s->setJustify(this, 0, 3, 1);
                s->setJustify(this, 0, 2, 1);
                s->setJustify(this, 0, 0, -1);
                ++icount;
            }
        }
        if (!icount)
        {
            for (Slot *s = pSlot; s && s != end; s = s->nextSibling())
            {
                s->setJustify(this, 0, 3, 1);
                s->setJustify(this, 0, 2, 1);
                s->setJustify(this, 0, 0, -1);
            }
        }
        ++numLevels;
    }

    Vector<JustifyTotal> stats(numLevels);
    for (Slot *s = pFirst; s && s != end; s = s->nextSibling())
    {
        float w = s->origin().x / scale + s->advance() - base;
        if (w > currWidth) currWidth = w;
        for (int j = 0; j < numLevels; ++j)
            stats[j].accumulate(s, this, j);
        s->just(0);
    }

    for (int i = (width < 0.0f) ? -1 : numLevels - 1; i >= 0; --i)
    {
        float diff;
        float error = 0.;
        float diffpw;
        int tWeight = stats[i].weight();
        if (tWeight == 0) continue;

        do {
            error = 0.;
            diff = width - currWidth;
            diffpw = diff / tWeight;
            tWeight = 0;
            for (Slot *s = pFirst; s && s != end; s = s->nextSibling()) // don't include final glyph
            {
                int w = s->getJustify(this, i, 3);
                float pref = diffpw * w + error;
                int step = s->getJustify(this, i, 2);
                if (!step) step = 1;        // handle lazy font developers
                if (pref > 0)
                {
                    float max = uint16(s->getJustify(this, i, 0));
                    if (i == 0) max -= s->just();
                    if (pref > max) pref = max;
                    else tWeight += w;
                }
                else
                {
                    float max = uint16(s->getJustify(this, i, 1));
                    if (i == 0) max += s->just();
                    if (-pref > max) pref = -max;
                    else tWeight += w;
                }
                int actual = int(pref / step) * step;

                if (actual)
                {
                    error += diffpw * w - actual;
                    if (i == 0)
                        s->just(s->just() + actual);
                    else
                        s->setJustify(this, i, 4, actual);
                }
            }
            currWidth += diff - error;
        } while (i == 0 && int(std::abs(error)) > 0 && tWeight);
    }

    Slot *oldFirst = m_first;
    Slot *oldLast = m_last;
    if (silf()->flags() & 1)
    {
        m_first = pSlot = addLineEnd(pSlot);
        m_last = pLast = addLineEnd(end);
        if (!m_first || !m_last) return -1.0;
    }
    else
    {
        m_first = pSlot;
        m_last = pLast;
    }

    // run justification passes here
#if !defined GRAPHITE2_NTRACING
    json * const dbgout = m_face->logger();
    if (dbgout)
        *dbgout << json::object
                    << "justifies"  << objectid(this)
                    << "passes"     << json::array;
#endif

    if (m_silf->justificationPass() != m_silf->positionPass() && (width >= 0.f || (silf()->flags() & 1)))
        m_silf->runGraphite(this, m_silf->justificationPass(), m_silf->positionPass());

#if !defined GRAPHITE2_NTRACING
    if (dbgout)
    {
        *dbgout     << json::item << json::close; // Close up the passes array
        positionSlots(NULL, pSlot, pLast, m_dir);
        Slot *lEnd = pLast->nextSibling();
        *dbgout << "output" << json::array;
        for(Slot * t = pSlot; t != lEnd; t = t->next())
            *dbgout     << dslot(this, t);
        *dbgout         << json::close << json::close;
    }
#endif

    res = positionSlots(font, pSlot, pLast, m_dir);

    if (silf()->flags() & 1)
    {
        if (m_first)
            delLineEnd(m_first);
        if (m_last)
            delLineEnd(m_last);
    }
    m_first = oldFirst;
    m_last = oldLast;

    if ((m_dir & 1) != m_silf->dir() && m_silf->bidiPass() != m_silf->numPasses())
        reverseSlots();
    return res.x;
}

Slot *Segment::addLineEnd(Slot *nSlot)
{
    Slot *eSlot = newSlot();
    if (!eSlot) return NULL;
    const uint16 gid = silf()->endLineGlyphid();
    const GlyphFace * theGlyph = m_face->glyphs().glyphSafe(gid);
    eSlot->setGlyph(this, gid, theGlyph);
    if (nSlot)
    {
        eSlot->next(nSlot);
        eSlot->prev(nSlot->prev());
        nSlot->prev(eSlot);
        eSlot->before(nSlot->before());
        if (eSlot->prev())
            eSlot->after(eSlot->prev()->after());
        else
            eSlot->after(nSlot->before());
    }
    else
    {
        nSlot = m_last;
        eSlot->prev(nSlot);
        nSlot->next(eSlot);
        eSlot->after(eSlot->prev()->after());
        eSlot->before(nSlot->after());
    }
    return eSlot;
}

void Segment::delLineEnd(Slot *s)
{
    Slot *nSlot = s->next();
    if (nSlot)
    {
        nSlot->prev(s->prev());
        if (s->prev())
            s->prev()->next(nSlot);
    }
    else
        s->prev()->next(NULL);
    freeSlot(s);
}
