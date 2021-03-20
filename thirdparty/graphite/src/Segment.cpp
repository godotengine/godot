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
#include "inc/UtfCodec.h"
#include <cstring>
#include <cstdlib>

#include "inc/bits.h"
#include "inc/Segment.h"
#include "graphite2/Font.h"
#include "inc/CharInfo.h"
#include "inc/debug.h"
#include "inc/Slot.h"
#include "inc/Main.h"
#include "inc/CmapCache.h"
#include "inc/Collider.h"
#include "graphite2/Segment.h"


using namespace graphite2;

Segment::Segment(size_t numchars, const Face* face, uint32 script, int textDir)
: m_freeSlots(NULL),
  m_freeJustifies(NULL),
  m_charinfo(new CharInfo[numchars]),
  m_collisions(NULL),
  m_face(face),
  m_silf(face->chooseSilf(script)),
  m_first(NULL),
  m_last(NULL),
  m_bufSize(numchars + 10),
  m_numGlyphs(numchars),
  m_numCharinfo(numchars),
  m_defaultOriginal(0),
  m_dir(textDir),
  m_flags(((m_silf->flags() & 0x20) != 0) << 1),
  m_passBits(m_silf->aPassBits() ? -1 : 0)
{
    freeSlot(newSlot());
    m_bufSize = log_binary(numchars)+1;
}

Segment::~Segment()
{
    for (SlotRope::iterator i = m_slots.begin(); i != m_slots.end(); ++i)
        free(*i);
    for (AttributeRope::iterator i = m_userAttrs.begin(); i != m_userAttrs.end(); ++i)
        free(*i);
    for (JustifyRope::iterator i = m_justifies.begin(); i != m_justifies.end(); ++i)
        free(*i);
    delete[] m_charinfo;
    free(m_collisions);
}

void Segment::appendSlot(int id, int cid, int gid, int iFeats, size_t coffset)
{
    Slot *aSlot = newSlot();

    if (!aSlot) return;
    m_charinfo[id].init(cid);
    m_charinfo[id].feats(iFeats);
    m_charinfo[id].base(coffset);
    const GlyphFace * theGlyph = m_face->glyphs().glyphSafe(gid);
    m_charinfo[id].breakWeight(theGlyph ? theGlyph->attrs()[m_silf->aBreak()] : 0);

    aSlot->child(NULL);
    aSlot->setGlyph(this, gid, theGlyph);
    aSlot->originate(id);
    aSlot->before(id);
    aSlot->after(id);
    if (m_last) m_last->next(aSlot);
    aSlot->prev(m_last);
    m_last = aSlot;
    if (!m_first) m_first = aSlot;
    if (theGlyph && m_silf->aPassBits())
        m_passBits &= theGlyph->attrs()[m_silf->aPassBits()]
                    | (m_silf->numPasses() > 16 ? (theGlyph->attrs()[m_silf->aPassBits() + 1] << 16) : 0);
}

Slot *Segment::newSlot()
{
    if (!m_freeSlots)
    {
        // check that the segment doesn't grow indefinintely
        if (m_numGlyphs > m_numCharinfo * MAX_SEG_GROWTH_FACTOR)
            return NULL;
        int numUser = m_silf->numUser();
#if !defined GRAPHITE2_NTRACING
        if (m_face->logger()) ++numUser;
#endif
        Slot *newSlots = grzeroalloc<Slot>(m_bufSize);
        int16 *newAttrs = grzeroalloc<int16>(m_bufSize * numUser);
        if (!newSlots || !newAttrs)
        {
            free(newSlots);
            free(newAttrs);
            return NULL;
        }
        for (size_t i = 0; i < m_bufSize; i++)
        {
            ::new (newSlots + i) Slot(newAttrs + i * numUser);
            newSlots[i].next(newSlots + i + 1);
        }
        newSlots[m_bufSize - 1].next(NULL);
        newSlots[0].next(NULL);
        m_slots.push_back(newSlots);
        m_userAttrs.push_back(newAttrs);
        m_freeSlots = (m_bufSize > 1)? newSlots + 1 : NULL;
        return newSlots;
    }
    Slot *res = m_freeSlots;
    m_freeSlots = m_freeSlots->next();
    res->next(NULL);
    return res;
}

void Segment::freeSlot(Slot *aSlot)
{
    if (aSlot == nullptr) return;
    if (m_last == aSlot) m_last = aSlot->prev();
    if (m_first == aSlot) m_first = aSlot->next();
    if (aSlot->attachedTo())
        aSlot->attachedTo()->removeChild(aSlot);
    while (aSlot->firstChild())
    {
        if (aSlot->firstChild()->attachedTo() == aSlot)
        {
            aSlot->firstChild()->attachTo(nullptr);
            aSlot->removeChild(aSlot->firstChild());
        }
        else
            aSlot->firstChild(nullptr);
    }
    // reset the slot incase it is reused
    ::new (aSlot) Slot(aSlot->userAttrs());
    memset(aSlot->userAttrs(), 0, m_silf->numUser() * sizeof(int16));
    // Update generation counter for debug
#if !defined GRAPHITE2_NTRACING
    if (m_face->logger())
        ++aSlot->userAttrs()[m_silf->numUser()];
#endif
    // update next pointer
    if (!m_freeSlots)
        aSlot->next(nullptr);
    else
        aSlot->next(m_freeSlots);
    m_freeSlots = aSlot;
}

SlotJustify *Segment::newJustify()
{
    if (!m_freeJustifies)
    {
        const size_t justSize = SlotJustify::size_of(m_silf->numJustLevels());
        byte *justs = grzeroalloc<byte>(justSize * m_bufSize);
        if (!justs) return NULL;
        for (ptrdiff_t i = m_bufSize - 2; i >= 0; --i)
        {
            SlotJustify *p = reinterpret_cast<SlotJustify *>(justs + justSize * i);
            SlotJustify *next = reinterpret_cast<SlotJustify *>(justs + justSize * (i + 1));
            p->next = next;
        }
        m_freeJustifies = (SlotJustify *)justs;
        m_justifies.push_back(m_freeJustifies);
    }
    SlotJustify *res = m_freeJustifies;
    m_freeJustifies = m_freeJustifies->next;
    res->next = NULL;
    return res;
}

void Segment::freeJustify(SlotJustify *aJustify)
{
    int numJust = m_silf->numJustLevels();
    if (m_silf->numJustLevels() <= 0) numJust = 1;
    aJustify->next = m_freeJustifies;
    memset(aJustify->values, 0, numJust*SlotJustify::NUMJUSTPARAMS*sizeof(int16));
    m_freeJustifies = aJustify;
}

// reverse the slots but keep diacritics in their same position after their bases
void Segment::reverseSlots()
{
    m_dir = m_dir ^ 64;                 // invert the reverse flag
    if (m_first == m_last) return;      // skip 0 or 1 glyph runs

    Slot *t = 0;
    Slot *curr = m_first;
    Slot *tlast;
    Slot *tfirst;
    Slot *out = 0;

    while (curr && getSlotBidiClass(curr) == 16)
        curr = curr->next();
    if (!curr) return;
    tfirst = curr->prev();
    tlast = curr;

    while (curr)
    {
        if (getSlotBidiClass(curr) == 16)
        {
            Slot *d = curr->next();
            while (d && getSlotBidiClass(d) == 16)
                d = d->next();

            d = d ? d->prev() : m_last;
            Slot *p = out->next();    // one after the diacritics. out can't be null
            if (p)
                p->prev(d);
            else
                tlast = d;
            t = d->next();
            d->next(p);
            curr->prev(out);
            out->next(curr);
        }
        else    // will always fire first time round the loop
        {
            if (out)
                out->prev(curr);
            t = curr->next();
            curr->next(out);
            out = curr;
        }
        curr = t;
    }
    out->prev(tfirst);
    if (tfirst)
        tfirst->next(out);
    else
        m_first = out;
    m_last = tlast;
}

void Segment::linkClusters(Slot *s, Slot * end)
{
    end = end->next();

    for (; s != end && !s->isBase(); s = s->next());
    Slot * ls = s;

    if (m_dir & 1)
    {
        for (; s != end; s = s->next())
        {
            if (!s->isBase())   continue;

            s->sibling(ls);
            ls = s;
        }
    }
    else
    {
        for (; s != end; s = s->next())
        {
            if (!s->isBase())   continue;

            ls->sibling(s);
            ls = s;
        }
    }
}

Position Segment::positionSlots(const Font *font, Slot * iStart, Slot * iEnd, bool isRtl, bool isFinal)
{
    Position currpos(0., 0.);
    float clusterMin = 0.;
    Rect bbox;
    bool reorder = (currdir() != isRtl);

    if (reorder)
    {
        Slot *temp;
        reverseSlots();
        temp = iStart;
        iStart = iEnd;
        iEnd = temp;
    }
    if (!iStart)    iStart = m_first;
    if (!iEnd)      iEnd   = m_last;

    if (!iStart || !iEnd)   // only true for empty segments
        return currpos;

    if (isRtl)
    {
        for (Slot * s = iEnd, * const end = iStart->prev(); s && s != end; s = s->prev())
        {
            if (s->isBase())
                currpos = s->finalise(this, font, currpos, bbox, 0, clusterMin = currpos.x, isRtl, isFinal);
        }
    }
    else
    {
        for (Slot * s = iStart, * const end = iEnd->next(); s && s != end; s = s->next())
        {
            if (s->isBase())
                currpos = s->finalise(this, font, currpos, bbox, 0, clusterMin = currpos.x, isRtl, isFinal);
        }
    }
    if (reorder)
        reverseSlots();
    return currpos;
}


void Segment::associateChars(int offset, size_t numChars)
{
    int i = 0, j = 0;
    CharInfo *c, *cend;
    for (c = m_charinfo + offset, cend = m_charinfo + offset + numChars; c != cend; ++c)
    {
        c->before(-1);
        c->after(-1);
    }
    for (Slot * s = m_first; s; s->index(i++), s = s->next())
    {
        j = s->before();
        if (j < 0)  continue;

        for (const int after = s->after(); j <= after; ++j)
        {
            c = charinfo(j);
            if (c->before() == -1 || i < c->before())   c->before(i);
            if (c->after() < i)                         c->after(i);
        }
    }
    for (Slot *s = m_first; s; s = s->next())
    {
        int a;
        for (a = s->after() + 1; a < offset + int(numChars) && charinfo(a)->after() < 0; ++a)
        { charinfo(a)->after(s->index()); }
        --a;
        s->after(a);

        for (a = s->before() - 1; a >= offset && charinfo(a)->before() < 0; --a)
        { charinfo(a)->before(s->index()); }
        ++a;
        s->before(a);
    }
}


template <typename utf_iter>
inline void process_utf_data(Segment & seg, const Face & face, const int fid, utf_iter c, size_t n_chars)
{
    const Cmap    & cmap = face.cmap();
    int slotid = 0;

    const typename utf_iter::codeunit_type * const base = c;
    for (; n_chars; --n_chars, ++c, ++slotid)
    {
        const uint32 usv = *c;
        uint16 gid = cmap[usv];
        if (!gid)   gid = face.findPseudo(usv);
        seg.appendSlot(slotid, usv, gid, fid, c - base);
    }
}


bool Segment::read_text(const Face *face, const Features* pFeats/*must not be NULL*/, gr_encform enc, const void* pStart, size_t nChars)
{
    assert(face);
    assert(pFeats);
    if (!m_charinfo) return false;

    // utf iterator is self recovering so we don't care about the error state of the iterator.
    switch (enc)
    {
    case gr_utf8:   process_utf_data(*this, *face, addFeatures(*pFeats), utf8::const_iterator(pStart), nChars); break;
    case gr_utf16:  process_utf_data(*this, *face, addFeatures(*pFeats), utf16::const_iterator(pStart), nChars); break;
    case gr_utf32:  process_utf_data(*this, *face, addFeatures(*pFeats), utf32::const_iterator(pStart), nChars); break;
    }
    return true;
}

void Segment::doMirror(uint16 aMirror)
{
    Slot * s;
    for (s = m_first; s; s = s->next())
    {
        unsigned short g = glyphAttr(s->gid(), aMirror);
        if (g && (!(dir() & 4) || !glyphAttr(s->gid(), aMirror + 1)))
            s->setGlyph(this, g);
    }
}

bool Segment::initCollisions()
{
    m_collisions = grzeroalloc<SlotCollision>(slotCount());
    if (!m_collisions) return false;

    for (Slot *p = m_first; p; p = p->next())
        if (p->index() < slotCount())
            ::new (collisionInfo(p)) SlotCollision(this, p);
        else
            return false;
    return true;
}
