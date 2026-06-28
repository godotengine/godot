// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2010, SIL International, All rights reserved.

#include "inc/Segment.h"
#include "inc/Slot.h"
#include "inc/Silf.h"
#include "inc/CharInfo.h"
#include "inc/Rule.h"
#include "inc/Collider.h"


using namespace graphite2;

Slot::Slot(int16 *user_attrs) :
    m_next(NULL), m_prev(NULL),
    m_glyphid(0), m_realglyphid(0), m_original(0), m_before(0), m_after(0),
    m_index(0), m_parent(NULL), m_child(NULL), m_sibling(NULL),
    m_position(0, 0), m_shift(0, 0), m_advance(0, 0),
    m_attach(0, 0), m_with(0, 0), m_just(0.),
    m_flags(0), m_attLevel(0), m_bidiCls(-1), m_bidiLevel(0),
    m_userAttr(user_attrs), m_justs(NULL)
{
}

// take care, this does not copy any of the GrSlot pointer fields
void Slot::set(const Slot & orig, int charOffset, size_t sizeAttr, size_t justLevels, size_t numChars)
{
    // leave m_next and m_prev unchanged
    m_glyphid = orig.m_glyphid;
    m_realglyphid = orig.m_realglyphid;
    m_original = orig.m_original + charOffset;
    if (charOffset + int(orig.m_before) < 0)
        m_before = 0;
    else
        m_before = orig.m_before + charOffset;
    if (charOffset <= 0 && orig.m_after + charOffset >= numChars)
        m_after = int(numChars) - 1;
    else
        m_after = orig.m_after + charOffset;
    m_parent = NULL;
    m_child = NULL;
    m_sibling = NULL;
    m_position = orig.m_position;
    m_shift = orig.m_shift;
    m_advance = orig.m_advance;
    m_attach = orig.m_attach;
    m_with = orig.m_with;
    m_flags = orig.m_flags;
    m_attLevel = orig.m_attLevel;
    m_bidiCls = orig.m_bidiCls;
    m_bidiLevel = orig.m_bidiLevel;
    if (m_userAttr && orig.m_userAttr)
        memcpy(m_userAttr, orig.m_userAttr, sizeAttr * sizeof(*m_userAttr));
    if (m_justs && orig.m_justs)
        memcpy(m_justs, orig.m_justs, SlotJustify::size_of(justLevels));
}

void Slot::update(int /*numGrSlots*/, int numCharInfo, Position &relpos)
{
    m_before += numCharInfo;
    m_after += numCharInfo;
    m_position = m_position + relpos;
}

Position Slot::finalise(const Segment *seg, const Font *font, Position & base, Rect & bbox, uint8 attrLevel, float & clusterMin, bool rtl, bool isFinal, int depth)
{
    SlotCollision *coll = NULL;
    if (depth > 100 || (attrLevel && m_attLevel > attrLevel)) return Position(0, 0);
    float scale = font ? font->scale() : 1.0f;
    Position shift(m_shift.x * (rtl * -2 + 1) + m_just, m_shift.y);
    float tAdvance = m_advance.x + m_just;
    if (isFinal && (coll = seg->collisionInfo(this)))
    {
        const Position &collshift = coll->offset();
        if (!(coll->flags() & SlotCollision::COLL_KERN) || rtl)
            shift = shift + collshift;
    }
    const GlyphFace * glyphFace = seg->getFace()->glyphs().glyphSafe(glyph());
    if (font)
    {
        scale = font->scale();
        shift *= scale;
        if (font->isHinted() && glyphFace)
            tAdvance = (m_advance.x - glyphFace->theAdvance().x + m_just) * scale + font->advance(glyph());
        else
            tAdvance *= scale;
    }
    Position res;

    m_position = base + shift;
    if (!m_parent)
    {
        res = base + Position(tAdvance, m_advance.y * scale);
        clusterMin = m_position.x;
    }
    else
    {
        float tAdv;
        m_position += (m_attach - m_with) * scale;
        tAdv = m_advance.x >= 0.5f ? m_position.x + tAdvance - shift.x : 0.f;
        res = Position(tAdv, 0);
        if ((m_advance.x >= 0.5f || m_position.x < 0) && m_position.x < clusterMin) clusterMin = m_position.x;
    }

    if (glyphFace)
    {
        Rect ourBbox = glyphFace->theBBox() * scale + m_position;
        bbox = bbox.widen(ourBbox);
    }

    if (m_child && m_child != this && m_child->attachedTo() == this)
    {
        Position tRes = m_child->finalise(seg, font, m_position, bbox, attrLevel, clusterMin, rtl, isFinal, depth + 1);
        if ((!m_parent || m_advance.x >= 0.5f) && tRes.x > res.x) res = tRes;
    }

    if (m_parent && m_sibling && m_sibling != this && m_sibling->attachedTo() == m_parent)
    {
        Position tRes = m_sibling->finalise(seg, font, base, bbox, attrLevel, clusterMin, rtl, isFinal, depth + 1);
        if (tRes.x > res.x) res = tRes;
    }

    if (!m_parent && clusterMin < base.x)
    {
        Position adj = Position(m_position.x - clusterMin, 0.);
        res += adj;
        m_position += adj;
        if (m_child) m_child->floodShift(adj);
    }
    return res;
}

int32 Slot::clusterMetric(const Segment *seg, uint8 metric, uint8 attrLevel, bool rtl)
{
    Position base;
    if (glyph() >= seg->getFace()->glyphs().numGlyphs())
        return 0;
    Rect bbox = seg->theGlyphBBoxTemporary(glyph());
    float clusterMin = 0.;
    Position res = finalise(seg, NULL, base, bbox, attrLevel, clusterMin, rtl, false);

    switch (metrics(metric))
    {
    case kgmetLsb :
        return int32(bbox.bl.x);
    case kgmetRsb :
        return int32(res.x - bbox.tr.x);
    case kgmetBbTop :
        return int32(bbox.tr.y);
    case kgmetBbBottom :
        return int32(bbox.bl.y);
    case kgmetBbLeft :
        return int32(bbox.bl.x);
    case kgmetBbRight :
        return int32(bbox.tr.x);
    case kgmetBbWidth :
        return int32(bbox.tr.x - bbox.bl.x);
    case kgmetBbHeight :
        return int32(bbox.tr.y - bbox.bl.y);
    case kgmetAdvWidth :
        return int32(res.x);
    case kgmetAdvHeight :
        return int32(res.y);
    default :
        return 0;
    }
}

#define SLOTGETCOLATTR(x) { SlotCollision *c = seg->collisionInfo(this); return c ? int(c-> x) : 0; }

int Slot::getAttr(const Segment *seg, attrCode ind, uint8 subindex) const
{
    if (ind >= gr_slatJStretch && ind < gr_slatJStretch + 20 && ind != gr_slatJWidth)
    {
        int indx = ind - gr_slatJStretch;
        return getJustify(seg, indx / 5, indx % 5);
    }

    switch (ind)
    {
    case gr_slatAdvX :      return int(m_advance.x);
    case gr_slatAdvY :      return int(m_advance.y);
    case gr_slatAttTo :     return m_parent ? 1 : 0;
    case gr_slatAttX :      return int(m_attach.x);
    case gr_slatAttY :      return int(m_attach.y);
    case gr_slatAttXOff :
    case gr_slatAttYOff :   return 0;
    case gr_slatAttWithX :  return int(m_with.x);
    case gr_slatAttWithY :  return int(m_with.y);
    case gr_slatAttWithXOff:
    case gr_slatAttWithYOff:return 0;
    case gr_slatAttLevel :  return m_attLevel;
    case gr_slatBreak :     return seg->charinfo(m_original)->breakWeight();
    case gr_slatCompRef :   return 0;
    case gr_slatDir :       return seg->dir() & 1;
    case gr_slatInsert :    return isInsertBefore();
    case gr_slatPosX :      return int(m_position.x); // but need to calculate it
    case gr_slatPosY :      return int(m_position.y);
    case gr_slatShiftX :    return int(m_shift.x);
    case gr_slatShiftY :    return int(m_shift.y);
    case gr_slatMeasureSol: return -1; // err what's this?
    case gr_slatMeasureEol: return -1;
    case gr_slatJWidth:     return int(m_just);
    case gr_slatUserDefnV1: subindex = 0; GR_FALLTHROUGH;
      // no break
    case gr_slatUserDefn :  return subindex < seg->numAttrs() ?  m_userAttr[subindex] : 0;
    case gr_slatSegSplit :  return seg->charinfo(m_original)->flags() & 3;
    case gr_slatBidiLevel:  return m_bidiLevel;
    case gr_slatColFlags :		{ SlotCollision *c = seg->collisionInfo(this); return c ? c->flags() : 0; }
    case gr_slatColLimitblx:SLOTGETCOLATTR(limit().bl.x)
    case gr_slatColLimitbly:SLOTGETCOLATTR(limit().bl.y)
    case gr_slatColLimittrx:SLOTGETCOLATTR(limit().tr.x)
    case gr_slatColLimittry:SLOTGETCOLATTR(limit().tr.y)
    case gr_slatColShiftx :	SLOTGETCOLATTR(offset().x)
    case gr_slatColShifty :	SLOTGETCOLATTR(offset().y)
    case gr_slatColMargin :	SLOTGETCOLATTR(margin())
    case gr_slatColMarginWt:SLOTGETCOLATTR(marginWt())
    case gr_slatColExclGlyph:SLOTGETCOLATTR(exclGlyph())
    case gr_slatColExclOffx:SLOTGETCOLATTR(exclOffset().x)
    case gr_slatColExclOffy:SLOTGETCOLATTR(exclOffset().y)
    case gr_slatSeqClass :	SLOTGETCOLATTR(seqClass())
    case gr_slatSeqProxClass:SLOTGETCOLATTR(seqProxClass())
    case gr_slatSeqOrder :	SLOTGETCOLATTR(seqOrder())
    case gr_slatSeqAboveXoff:SLOTGETCOLATTR(seqAboveXoff())
    case gr_slatSeqAboveWt: SLOTGETCOLATTR(seqAboveWt())
    case gr_slatSeqBelowXlim:SLOTGETCOLATTR(seqBelowXlim())
    case gr_slatSeqBelowWt:	SLOTGETCOLATTR(seqBelowWt())
    case gr_slatSeqValignHt:SLOTGETCOLATTR(seqValignHt())
    case gr_slatSeqValignWt:SLOTGETCOLATTR(seqValignWt())
    default : return 0;
    }
}

#define SLOTCOLSETATTR(x) { \
        SlotCollision *c = seg->collisionInfo(this); \
        if (c) { c-> x ; c->setFlags(c->flags() & ~SlotCollision::COLL_KNOWN); } \
        break; }
#define SLOTCOLSETCOMPLEXATTR(t, y, x) { \
        SlotCollision *c = seg->collisionInfo(this); \
        if (c) { \
        const t &s = c-> y; \
        c-> x ; c->setFlags(c->flags() & ~SlotCollision::COLL_KNOWN); } \
        break; }

void Slot::setAttr(Segment *seg, attrCode ind, uint8 subindex, int16 value, const SlotMap & map)
{
    if (ind == gr_slatUserDefnV1)
    {
        ind = gr_slatUserDefn;
        subindex = 0;
        if (seg->numAttrs() == 0)
            return;
    }
    else if (ind >= gr_slatJStretch && ind < gr_slatJStretch + 20 && ind != gr_slatJWidth)
    {
        int indx = ind - gr_slatJStretch;
        return setJustify(seg, indx / 5, indx % 5, value);
    }

    switch (ind)
    {
    case gr_slatAdvX :  m_advance.x = value; break;
    case gr_slatAdvY :  m_advance.y = value; break;
    case gr_slatAttTo :
    {
        const uint16 idx = uint16(value);
        if (idx < map.size() && map[idx])
        {
            Slot *other = map[idx];
            if (other == this || other == m_parent || other->isCopied()) break;
            if (m_parent) { m_parent->removeChild(this); attachTo(NULL); }
            Slot *pOther = other;
            int count = 0;
            bool foundOther = false;
            while (pOther)
            {
                ++count;
                if (pOther == this) foundOther = true;
                pOther = pOther->attachedTo();
            }
            for (pOther = m_child; pOther; pOther = pOther->m_child)
                ++count;
            for (pOther = m_sibling; pOther; pOther = pOther->m_sibling)
                ++count;
            if (count < 100 && !foundOther && other->child(this))
            {
                attachTo(other);
                if ((map.dir() != 0) ^ (idx > subindex))
                    m_with = Position(advance(), 0);
                else        // normal match to previous root
                    m_attach = Position(other->advance(), 0);
            }
        }
        break;
    }
    case gr_slatAttX :          m_attach.x = value; break;
    case gr_slatAttY :          m_attach.y = value; break;
    case gr_slatAttXOff :
    case gr_slatAttYOff :       break;
    case gr_slatAttWithX :      m_with.x = value; break;
    case gr_slatAttWithY :      m_with.y = value; break;
    case gr_slatAttWithXOff :
    case gr_slatAttWithYOff :   break;
    case gr_slatAttLevel :
        m_attLevel = byte(value);
        break;
    case gr_slatBreak :
        seg->charinfo(m_original)->breakWeight(value);
        break;
    case gr_slatCompRef :   break;      // not sure what to do here
    case gr_slatDir : break;
    case gr_slatInsert :
        markInsertBefore(value? true : false);
        break;
    case gr_slatPosX :      break; // can't set these here
    case gr_slatPosY :      break;
    case gr_slatShiftX :    m_shift.x = value; break;
    case gr_slatShiftY :    m_shift.y = value; break;
    case gr_slatMeasureSol :    break;
    case gr_slatMeasureEol :    break;
    case gr_slatJWidth :    just(value); break;
    case gr_slatSegSplit :  seg->charinfo(m_original)->addflags(value & 3); break;
    case gr_slatUserDefn :  m_userAttr[subindex] = value; break;
    case gr_slatColFlags :  {
        SlotCollision *c = seg->collisionInfo(this);
        if (c)
            c->setFlags(value);
        break; }
    case gr_slatColLimitblx :	SLOTCOLSETCOMPLEXATTR(Rect, limit(), setLimit(Rect(Position(value, s.bl.y), s.tr)))
    case gr_slatColLimitbly :	SLOTCOLSETCOMPLEXATTR(Rect, limit(), setLimit(Rect(Position(s.bl.x, value), s.tr)))
    case gr_slatColLimittrx :	SLOTCOLSETCOMPLEXATTR(Rect, limit(), setLimit(Rect(s.bl, Position(value, s.tr.y))))
    case gr_slatColLimittry :	SLOTCOLSETCOMPLEXATTR(Rect, limit(), setLimit(Rect(s.bl, Position(s.tr.x, value))))
    case gr_slatColMargin :		SLOTCOLSETATTR(setMargin(value))
    case gr_slatColMarginWt :	SLOTCOLSETATTR(setMarginWt(value))
    case gr_slatColExclGlyph :	SLOTCOLSETATTR(setExclGlyph(value))
    case gr_slatColExclOffx :	SLOTCOLSETCOMPLEXATTR(Position, exclOffset(), setExclOffset(Position(value, s.y)))
    case gr_slatColExclOffy :	SLOTCOLSETCOMPLEXATTR(Position, exclOffset(), setExclOffset(Position(s.x, value)))
    case gr_slatSeqClass :		SLOTCOLSETATTR(setSeqClass(value))
	case gr_slatSeqProxClass :	SLOTCOLSETATTR(setSeqProxClass(value))
    case gr_slatSeqOrder :		SLOTCOLSETATTR(setSeqOrder(value))
    case gr_slatSeqAboveXoff :	SLOTCOLSETATTR(setSeqAboveXoff(value))
    case gr_slatSeqAboveWt :	SLOTCOLSETATTR(setSeqAboveWt(value))
    case gr_slatSeqBelowXlim :	SLOTCOLSETATTR(setSeqBelowXlim(value))
    case gr_slatSeqBelowWt :	SLOTCOLSETATTR(setSeqBelowWt(value))
    case gr_slatSeqValignHt :	SLOTCOLSETATTR(setSeqValignHt(value))
    case gr_slatSeqValignWt :	SLOTCOLSETATTR(setSeqValignWt(value))
    default :
        break;
    }
}

int Slot::getJustify(const Segment *seg, uint8 level, uint8 subindex) const
{
    if (level && level >= seg->silf()->numJustLevels()) return 0;

    if (m_justs)
        return m_justs->values[level * SlotJustify::NUMJUSTPARAMS + subindex];

    if (level >= seg->silf()->numJustLevels()) return 0;
    Justinfo *jAttrs = seg->silf()->justAttrs() + level;

    switch (subindex) {
        case 0 : return seg->glyphAttr(gid(), jAttrs->attrStretch());
        case 1 : return seg->glyphAttr(gid(), jAttrs->attrShrink());
        case 2 : return seg->glyphAttr(gid(), jAttrs->attrStep());
        case 3 : return seg->glyphAttr(gid(), jAttrs->attrWeight());
        case 4 : return 0;      // not been set yet, so clearly 0
        default: return 0;
    }
}

void Slot::setJustify(Segment *seg, uint8 level, uint8 subindex, int16 value)
{
    if (level && level >= seg->silf()->numJustLevels()) return;
    if (!m_justs)
    {
        SlotJustify *j = seg->newJustify();
        if (!j) return;
        j->LoadSlot(this, seg);
        m_justs = j;
    }
    m_justs->values[level * SlotJustify::NUMJUSTPARAMS + subindex] = value;
}

bool Slot::child(Slot *ap)
{
    if (this == ap) return false;
    else if (ap == m_child) return true;
    else if (!m_child)
        m_child = ap;
    else
        return m_child->sibling(ap);
    return true;
}

bool Slot::sibling(Slot *ap)
{
    if (this == ap) return false;
    else if (ap == m_sibling) return true;
    else if (!m_sibling || !ap)
        m_sibling = ap;
    else
        return m_sibling->sibling(ap);
    return true;
}

bool Slot::removeChild(Slot *ap)
{
    if (this == ap || !m_child || !ap) return false;
    else if (ap == m_child)
    {
        Slot *nSibling = m_child->nextSibling();
        m_child->nextSibling(NULL);
        m_child = nSibling;
        return true;
    }
    for (Slot *p = m_child; p; p = p->m_sibling)
    {
        if (p->m_sibling && p->m_sibling == ap)
        {
            p->m_sibling = p->m_sibling->m_sibling;
            ap->nextSibling(NULL);
            return true;
        }
    }
    return false;
}

void Slot::setGlyph(Segment *seg, uint16 glyphid, const GlyphFace * theGlyph)
{
    m_glyphid = glyphid;
    m_bidiCls = -1;
    if (!theGlyph)
    {
        theGlyph = seg->getFace()->glyphs().glyphSafe(glyphid);
        if (!theGlyph)
        {
            m_realglyphid = 0;
            m_advance = Position(0.,0.);
            return;
        }
    }
    m_realglyphid = theGlyph->attrs()[seg->silf()->aPseudo()];
    if (m_realglyphid > seg->getFace()->glyphs().numGlyphs())
        m_realglyphid = 0;
    const GlyphFace *aGlyph = theGlyph;
    if (m_realglyphid)
    {
        aGlyph = seg->getFace()->glyphs().glyphSafe(m_realglyphid);
        if (!aGlyph) aGlyph = theGlyph;
    }
    m_advance = Position(aGlyph->theAdvance().x, 0.);
    if (seg->silf()->aPassBits())
    {
        seg->mergePassBits(uint8(theGlyph->attrs()[seg->silf()->aPassBits()]));
        if (seg->silf()->numPasses() > 16)
            seg->mergePassBits(theGlyph->attrs()[seg->silf()->aPassBits()+1] << 16);
    }
}

void Slot::floodShift(Position adj, int depth)
{
    if (depth > 100)
        return;
    m_position += adj;
    if (m_child) m_child->floodShift(adj, depth + 1);
    if (m_sibling) m_sibling->floodShift(adj, depth + 1);
}

void SlotJustify::LoadSlot(const Slot *s, const Segment *seg)
{
    for (int i = seg->silf()->numJustLevels() - 1; i >= 0; --i)
    {
        Justinfo *justs = seg->silf()->justAttrs() + i;
        int16 *v = values + i * NUMJUSTPARAMS;
        v[0] = seg->glyphAttr(s->gid(), justs->attrStretch());
        v[1] = seg->glyphAttr(s->gid(), justs->attrShrink());
        v[2] = seg->glyphAttr(s->gid(), justs->attrStep());
        v[3] = seg->glyphAttr(s->gid(), justs->attrWeight());
    }
}

Slot * Slot::nextInCluster(const Slot *s) const
{
    Slot *base;
    if (s->firstChild())
        return s->firstChild();
    else if (s->nextSibling())
        return s->nextSibling();
    while ((base = s->attachedTo()))
    {
        // if (base->firstChild() == s && base->nextSibling())
        if (base->nextSibling())
            return base->nextSibling();
        s = base;
    }
    return NULL;
}

bool Slot::isChildOf(const Slot *base) const
{
    for (Slot *p = m_parent; p; p = p->m_parent)
        if (p == base)
            return true;
    return false;
}
