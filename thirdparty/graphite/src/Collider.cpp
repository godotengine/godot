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
#include <algorithm>
#include <limits>
#include <cmath>
#include <string>
#include <functional>
#include "inc/Collider.h"
#include "inc/Segment.h"
#include "inc/Slot.h"
#include "inc/GlyphCache.h"
#include "inc/Sparse.h"

#define ISQRT2 0.707106781f

// Possible rounding error for subbox boundaries: 0.016 = 1/64 = 1/256 * 4
// (values in font range from 0..256)
// #define SUBBOX_RND_ERR 0.016

using namespace graphite2;

////    SHIFT-COLLIDER    ////

// Initialize the Collider to hold the basic movement limits for the
// target slot, the one we are focusing on fixing.
bool ShiftCollider::initSlot(Segment *seg, Slot *aSlot, const Rect &limit, float margin, float marginWeight,
    const Position &currShift, const Position &currOffset, int dir, GR_MAYBE_UNUSED json * const dbgout)
{
    int i;
    float mx, mn;
    float a, shift;
    const GlyphCache &gc = seg->getFace()->glyphs();
    unsigned short gid = aSlot->gid();
    if (!gc.check(gid))
        return false;
    const BBox &bb = gc.getBoundingBBox(gid);
    const SlantBox &sb = gc.getBoundingSlantBox(gid);
    //float sx = aSlot->origin().x + currShift.x;
    //float sy = aSlot->origin().y + currShift.y;
    if (currOffset.x != 0.f || currOffset.y != 0.f)
        _limit = Rect(limit.bl - currOffset, limit.tr - currOffset);
    else
        _limit = limit;
    // For a ShiftCollider, these indices indicate which vector we are moving by:
    // each _ranges represents absolute space with respect to the origin of the slot. Thus take into account true origins but subtract the vmin for the slot
    for (i = 0; i < 4; ++i)
    {
        switch (i) {
            case 0 :	// x direction
                mn = _limit.bl.x + currOffset.x;
                mx = _limit.tr.x + currOffset.x;
                _len[i] = bb.xa - bb.xi;
                a = currOffset.y + currShift.y;
                _ranges[i].initialise<XY>(mn, mx, margin, marginWeight, a);
                break;
            case 1 :	// y direction
                mn = _limit.bl.y + currOffset.y;
                mx = _limit.tr.y + currOffset.y;
                _len[i] = bb.ya - bb.yi;
                a = currOffset.x + currShift.x;
                _ranges[i].initialise<XY>(mn, mx, margin, marginWeight, a);
                break;
            case 2 :	// sum (negatively sloped diagonal boundaries)
                // pick closest x,y limit boundaries in s direction
                shift = currOffset.x + currOffset.y + currShift.x + currShift.y;
                mn = -2 * min(currShift.x - _limit.bl.x, currShift.y - _limit.bl.y) + shift;
                mx = 2 * min(_limit.tr.x - currShift.x, _limit.tr.y - currShift.y) + shift;
                _len[i] = sb.sa - sb.si;
                a = currOffset.x - currOffset.y + currShift.x - currShift.y;
                _ranges[i].initialise<SD>(mn, mx, margin / ISQRT2, marginWeight, a);
                break;
            case 3 :	// diff (positively sloped diagonal boundaries)
                // pick closest x,y limit boundaries in d direction
                shift = currOffset.x - currOffset.y + currShift.x - currShift.y;
                mn = -2 * min(currShift.x - _limit.bl.x, _limit.tr.y - currShift.y) + shift;
                mx = 2 * min(_limit.tr.x - currShift.x, currShift.y - _limit.bl.y) + shift;
                _len[i] = sb.da - sb.di;
                a = currOffset.x + currOffset.y + currShift.x + currShift.y;
                _ranges[i].initialise<SD>(mn, mx, margin / ISQRT2, marginWeight, a);
                break;
        }
    }

	_target = aSlot;
    if ((dir & 1) == 0)
    {
        // For LTR, switch and negate x limits.
        _limit.bl.x = -1 * limit.tr.x;
        //_limit.tr.x = -1 * limit.bl.x;
    }
    _currOffset = currOffset;
    _currShift = currShift;
    _origin = aSlot->origin() - currOffset;     // the original anchor position of the glyph

	_margin = margin;
	_marginWt = marginWeight;

    SlotCollision *c = seg->collisionInfo(aSlot);
    _seqClass = c->seqClass();
	_seqProxClass = c->seqProxClass();
    _seqOrder = c->seqOrder();
    return true;
}

template <class O>
float sdm(float vi, float va, float mx, float my, O op)
{
    float res = 2 * mx - vi;
    if (op(res, vi + 2 * my))
    {
        res = va + 2 * my;
        if (op(res, 2 * mx - va))
            res = mx + my;
    }
    return res;
}

// Mark an area with a cost that can vary along the x or y axis. The region is expressed in terms of the centre of the target glyph in each axis
void ShiftCollider::addBox_slope(bool isx, const Rect &box, const BBox &bb, const SlantBox &sb, const Position &org, float weight, float m, bool minright, int axis)
{
    float a, c;
    switch (axis) {
        case 0 :
             if (box.bl.y < org.y + bb.ya && box.tr.y > org.y + bb.yi && box.width() > 0)
            {
                a = org.y + 0.5f * (bb.yi + bb.ya);
                c = 0.5f * (bb.xi + bb.xa);
                if (isx)
                    _ranges[axis].weighted<XY>(box.bl.x - c, box.tr.x - c, weight, a, m,
                                                (minright ? box.tr.x : box.bl.x) - c, a, 0, false);
                else
                    _ranges[axis].weighted<XY>(box.bl.x - c, box.tr.x - c, weight, a, 0, 0, org.y,
                                                m * (a * a + sqr((minright ? box.tr.y : box.bl.y) - 0.5f * (bb.yi + bb.ya))), false);
            }
            break;
        case 1 :
            if (box.bl.x < org.x + bb.xa && box.tr.x > org.x + bb.xi && box.height() > 0)
            {
                a = org.x + 0.5f * (bb.xi + bb.xa);
                c = 0.5f * (bb.yi + bb.ya);
                if (isx)
                    _ranges[axis].weighted<XY>(box.bl.y - c, box.tr.y - c, weight, a, 0, 0, org.x,
                                                m * (a * a + sqr((minright ? box.tr.x : box.bl.x) - 0.5f * (bb.xi + bb.xa))), false);
                else
                    _ranges[axis].weighted<XY>(box.bl.y - c, box.tr.y - c, weight, a, m,
                                                (minright ? box.tr.y : box.bl.y) - c, a, 0, false);
            }
            break;
        case 2 :
            if (box.bl.x - box.tr.y < org.x - org.y + sb.da && box.tr.x - box.bl.y > org.x - org.y + sb.di)
            {
                float d = org.x - org.y + 0.5f * (sb.di + sb.da);
                c = 0.5f * (sb.si + sb.sa);
                float smax = min(2 * box.tr.x - d, 2 * box.tr.y + d);
                float smin = max(2 * box.bl.x - d, 2 * box.bl.y + d);
                if (smin > smax) return;
                float si;
                a = d;
                if (isx)
                    si = 2 * (minright ? box.tr.x : box.bl.x) - a;
                else
                    si = 2 * (minright ? box.tr.y : box.bl.y) + a;
                _ranges[axis].weighted<SD>(smin - c, smax - c, weight / 2, a, m / 2, si, 0, 0, isx);
            }
            break;
        case 3 :
            if (box.bl.x + box.bl.y < org.x + org.y + sb.sa && box.tr.x + box.tr.y > org.x + org.y + sb.si)
            {
                float s = org.x + org.y + 0.5f * (sb.si + sb.sa);
                c = 0.5f * (sb.di + sb.da);
                float dmax = min(2 * box.tr.x - s, s - 2 * box.bl.y);
                float dmin = max(2 * box.bl.x - s, s - 2 * box.tr.y);
                if (dmin > dmax) return;
                float di;
                a = s;
                if (isx)
                    di = 2 * (minright ? box.tr.x : box.bl.x) - a;
                else
                    di = 2 * (minright ? box.tr.y : box.bl.y) + a;
                _ranges[axis].weighted<SD>(dmin - c, dmax - c, weight / 2, a, m / 2, di, 0, 0, !isx);
            }
            break;
        default :
            break;
    }
    return;
}

// Mark an area with an absolute cost, making it completely inaccessible.
inline void ShiftCollider::removeBox(const Rect &box, const BBox &bb, const SlantBox &sb, const Position &org, int axis)
{
    float c;
    switch (axis) {
        case 0 :
            if (box.bl.y < org.y + bb.ya && box.tr.y > org.y + bb.yi && box.width() > 0)
            {
                c = 0.5f * (bb.xi + bb.xa);
                _ranges[axis].exclude(box.bl.x - c, box.tr.x - c);
            }
            break;
        case 1 :
            if (box.bl.x < org.x + bb.xa && box.tr.x > org.x + bb.xi && box.height() > 0)
            {
                c = 0.5f * (bb.yi + bb.ya);
                _ranges[axis].exclude(box.bl.y - c, box.tr.y - c);
            }
            break;
        case 2 :
            if (box.bl.x - box.tr.y < org.x - org.y + sb.da && box.tr.x - box.bl.y > org.x - org.y + sb.di
                && box.width() > 0 && box.height() > 0)
            {
                float di = org.x - org.y + sb.di;
                float da = org.x - org.y + sb.da;
                float smax = sdm(di, da, box.tr.x, box.tr.y, std::greater<float>());
                float smin = sdm(da, di, box.bl.x, box.bl.y, std::less<float>());
                c = 0.5f * (sb.si + sb.sa);
                _ranges[axis].exclude(smin - c, smax - c);
            }
            break;
        case 3 :
            if (box.bl.x + box.bl.y < org.x + org.y + sb.sa && box.tr.x + box.tr.y > org.x + org.y + sb.si
                && box.width() > 0 && box.height() > 0)
            {
                float si = org.x + org.y + sb.si;
                float sa = org.x + org.y + sb.sa;
                float dmax = sdm(si, sa, box.tr.x, -box.bl.y, std::greater<float>());
                float dmin = sdm(sa, si, box.bl.x, -box.tr.y, std::less<float>());
                c = 0.5f * (sb.di + sb.da);
                _ranges[axis].exclude(dmin - c, dmax - c);
            }
            break;
        default :
            break;
    }
    return;
}

// Adjust the movement limits for the target to avoid having it collide
// with the given neighbor slot. Also determine if there is in fact a collision
// between the target and the given slot.
bool ShiftCollider::mergeSlot(Segment *seg, Slot *slot, const SlotCollision *cslot, const Position &currShift,
		bool isAfter,  // slot is logically after _target
		bool sameCluster, bool &hasCol, bool isExclusion,
        GR_MAYBE_UNUSED json * const dbgout )
{
    bool isCol = false;
    const float sx = slot->origin().x - _origin.x + currShift.x;
    const float sy = slot->origin().y - _origin.y + currShift.y;
    const float sd = sx - sy;
    const float ss = sx + sy;
    float vmin, vmax;
    float omin, omax, otmin, otmax;
    float cmin, cmax;   // target limits
    float torg;
    const GlyphCache &gc = seg->getFace()->glyphs();
    const unsigned short gid = slot->gid();
    if (!gc.check(gid))
        return false;
    const BBox &bb = gc.getBoundingBBox(gid);

    // SlotCollision * cslot = seg->collisionInfo(slot);
    int orderFlags = 0;
    bool sameClass = _seqProxClass == 0 && cslot->seqClass() == _seqClass;
    if (sameCluster && _seqClass
        && (sameClass || (_seqProxClass != 0 && cslot->seqClass() == _seqProxClass)))
		// Force the target glyph to be in the specified direction from the slot we're testing.
        orderFlags = _seqOrder;

    // short circuit if only interested in direct collision and we are out of range
    if (orderFlags || (sx + bb.xa + _margin >= _limit.bl.x && sx + bb.xi - _margin <= _limit.tr.x)
                    || (sy + bb.ya + _margin >= _limit.bl.y && sy + bb.yi - _margin <= _limit.tr.y))

    {
        const float tx = _currOffset.x + _currShift.x;
        const float ty = _currOffset.y + _currShift.y;
        const float td = tx - ty;
        const float ts = tx + ty;
        const SlantBox &sb = gc.getBoundingSlantBox(gid);
        const unsigned short tgid = _target->gid();
        const BBox &tbb = gc.getBoundingBBox(tgid);
        const SlantBox &tsb = gc.getBoundingSlantBox(tgid);
        float seq_above_wt = cslot->seqAboveWt();
        float seq_below_wt = cslot->seqBelowWt();
        float seq_valign_wt = cslot->seqValignWt();
        float lmargin;
        // if isAfter, invert orderFlags for diagonal orders.
        if (isAfter)
        {
            // invert appropriate bits
            orderFlags ^= (sameClass ? 0x3F : 0x3);
            // consider 2 bits at a time, non overlapping. If both bits set, clear them
            orderFlags = orderFlags ^ ((((orderFlags >> 1) & orderFlags) & 0x15) * 3);
        }

#if !defined GRAPHITE2_NTRACING
        if (dbgout)
            dbgout->setenv(0, slot);
#endif

        // Process main bounding octabox.
        for (int i = 0; i < 4; ++i)
        {
            switch (i) {
                case 0 :	// x direction
                    vmin = max(max(bb.xi - tbb.xa + sx, sb.di - tsb.da + ty + sd), sb.si - tsb.sa - ty + ss);
                    vmax = min(min(bb.xa - tbb.xi + sx, sb.da - tsb.di + ty + sd), sb.sa - tsb.si - ty + ss);
                    otmin = tbb.yi + ty;
                    otmax = tbb.ya + ty;
                    omin = bb.yi + sy;
                    omax = bb.ya + sy;
                    torg = _currOffset.x;
                    cmin = _limit.bl.x + torg;
                    cmax = _limit.tr.x - tbb.xi + tbb.xa + torg;
                    lmargin = _margin;
                    break;
                case 1 :	// y direction
                    vmin = max(max(bb.yi - tbb.ya + sy, tsb.di - sb.da + tx - sd), sb.si - tsb.sa - tx + ss);
                    vmax = min(min(bb.ya - tbb.yi + sy, tsb.da - sb.di + tx - sd), sb.sa - tsb.si - tx + ss);
                    otmin = tbb.xi + tx;
                    otmax = tbb.xa + tx;
                    omin = bb.xi + sx;
                    omax = bb.xa + sx;
                    torg = _currOffset.y;
                    cmin = _limit.bl.y + torg;
                    cmax = _limit.tr.y - tbb.yi + tbb.ya + torg;
                    lmargin = _margin;
                    break;
                case 2 :    // sum - moving along the positively-sloped vector, so the boundaries are the
                            // negatively-sloped boundaries.
                    vmin = max(max(sb.si - tsb.sa + ss, 2 * (bb.yi - tbb.ya + sy) + td), 2 * (bb.xi - tbb.xa + sx) - td);
                    vmax = min(min(sb.sa - tsb.si + ss, 2 * (bb.ya - tbb.yi + sy) + td), 2 * (bb.xa - tbb.xi + sx) - td);
                    otmin = tsb.di + td;
                    otmax = tsb.da + td;
                    omin = sb.di + sd;
                    omax = sb.da + sd;
                    torg = _currOffset.x + _currOffset.y;
                    cmin = _limit.bl.x + _limit.bl.y + torg;
                    cmax = _limit.tr.x + _limit.tr.y - tsb.si + tsb.sa + torg;
                    lmargin = _margin / ISQRT2;
                    break;
                case 3 :    // diff - moving along the negatively-sloped vector, so the boundaries are the
                            // positively-sloped boundaries.
                    vmin = max(max(sb.di - tsb.da + sd, 2 * (bb.xi - tbb.xa + sx) - ts), -2 * (bb.ya - tbb.yi + sy) + ts);
                    vmax = min(min(sb.da - tsb.di + sd, 2 * (bb.xa - tbb.xi + sx) - ts), -2 * (bb.yi - tbb.ya + sy) + ts);
                    otmin = tsb.si + ts;
                    otmax = tsb.sa + ts;
                    omin = sb.si + ss;
                    omax = sb.sa + ss;
                    torg = _currOffset.x - _currOffset.y;
                    cmin = _limit.bl.x - _limit.tr.y + torg;
                    cmax = _limit.tr.x - _limit.bl.y - tsb.di + tsb.da + torg;
                    lmargin = _margin / ISQRT2;
                    break;
                default :
                    continue;
            }

#if !defined GRAPHITE2_NTRACING
            if (dbgout)
                dbgout->setenv(1, reinterpret_cast<void *>(-1));
#define DBGTAG(x) if (dbgout) dbgout->setenv(1, reinterpret_cast<void *>(-x));
#else
#define DBGTAG(x)
#endif

            if (orderFlags)
            {
                Position org(tx, ty);
                float xminf = _limit.bl.x + _currOffset.x + tbb.xi;
                float xpinf = _limit.tr.x + _currOffset.x + tbb.xa;
                float ypinf = _limit.tr.y + _currOffset.y + tbb.ya;
                float yminf = _limit.bl.y + _currOffset.y + tbb.yi;
                switch (orderFlags) {
                    case SlotCollision::SEQ_ORDER_RIGHTUP :
                    {
                        float r1Xedge = cslot->seqAboveXoff() + 0.5f * (bb.xi + bb.xa) + sx;
                        float r3Xedge = cslot->seqBelowXlim() + bb.xa + sx + 0.5f * (tbb.xa - tbb.xi);
                        float r2Yedge = 0.5f * (bb.yi + bb.ya) + sy;

                        // DBGTAG(1x) means the regions are up and right
                        // region 1
                        DBGTAG(11)
                        addBox_slope(true, Rect(Position(xminf, r2Yedge), Position(r1Xedge, ypinf)),
                                        tbb, tsb, org, 0, seq_above_wt, true, i);
                        // region 2
                        DBGTAG(12)
                        removeBox(Rect(Position(xminf, yminf), Position(r3Xedge, r2Yedge)), tbb, tsb, org, i);
                        // region 3, which end is zero is irrelevant since m weight is 0
                        DBGTAG(13)
                        addBox_slope(true, Rect(Position(r3Xedge, yminf), Position(xpinf, r2Yedge - cslot->seqValignHt())),
                                        tbb, tsb, org, seq_below_wt, 0, true, i);
                        // region 4
                        DBGTAG(14)
                        addBox_slope(false, Rect(Position(sx + bb.xi, r2Yedge), Position(xpinf, r2Yedge + cslot->seqValignHt())),
                                        tbb, tsb, org, 0, seq_valign_wt, true, i);
                        // region 5
                        DBGTAG(15)
                        addBox_slope(false, Rect(Position(sx + bb.xi, r2Yedge - cslot->seqValignHt()), Position(xpinf, r2Yedge)),
                                        tbb, tsb, org, seq_below_wt, seq_valign_wt, false, i);
                        break;
                    }
                    case SlotCollision::SEQ_ORDER_LEFTDOWN :
                    {
                        float r1Xedge = 0.5f * (bb.xi + bb.xa) + cslot->seqAboveXoff() + sx;
                        float r3Xedge = bb.xi - cslot->seqBelowXlim() + sx - 0.5f * (tbb.xa - tbb.xi);
                        float r2Yedge = 0.5f * (bb.yi + bb.ya) + sy;
                        // DBGTAG(2x) means the regions are up and right
                        // region 1
                        DBGTAG(21)
                        addBox_slope(true, Rect(Position(r1Xedge, yminf), Position(xpinf, r2Yedge)),
                                        tbb, tsb, org, 0, seq_above_wt, false, i);
                        // region 2
                        DBGTAG(22)
                        removeBox(Rect(Position(r3Xedge, r2Yedge), Position(xpinf, ypinf)), tbb, tsb, org, i);
                        // region 3
                        DBGTAG(23)
                        addBox_slope(true, Rect(Position(xminf, r2Yedge - cslot->seqValignHt()), Position(r3Xedge, ypinf)),
                                        tbb, tsb, org, seq_below_wt, 0, false, i);
                        // region 4
                        DBGTAG(24)
                        addBox_slope(false, Rect(Position(xminf, r2Yedge), Position(sx + bb.xa, r2Yedge + cslot->seqValignHt())),
                                        tbb, tsb, org, 0, seq_valign_wt, true, i);
                        // region 5
                        DBGTAG(25)
                        addBox_slope(false, Rect(Position(xminf, r2Yedge - cslot->seqValignHt()),
                                        Position(sx + bb.xa, r2Yedge)), tbb, tsb, org, seq_below_wt, seq_valign_wt, false, i);
                        break;
                    }
                    case SlotCollision::SEQ_ORDER_NOABOVE : // enforce neighboring glyph being above
                        DBGTAG(31);
                        removeBox(Rect(Position(bb.xi - tbb.xa + sx, sy + bb.ya),
                                        Position(bb.xa - tbb.xi + sx, ypinf)), tbb, tsb, org, i);
                        break;
                    case SlotCollision::SEQ_ORDER_NOBELOW :	// enforce neighboring glyph being below
                        DBGTAG(32);
                        removeBox(Rect(Position(bb.xi - tbb.xa + sx, yminf),
                                        Position(bb.xa - tbb.xi + sx, sy + bb.yi)), tbb, tsb, org, i);
                        break;
                    case SlotCollision::SEQ_ORDER_NOLEFT :  // enforce neighboring glyph being to the left
                        DBGTAG(33)
                        removeBox(Rect(Position(xminf, bb.yi - tbb.ya + sy),
                                        Position(bb.xi - tbb.xa + sx, bb.ya - tbb.yi + sy)), tbb, tsb, org, i);
                        break;
                    case SlotCollision::SEQ_ORDER_NORIGHT : // enforce neighboring glyph being to the right
                        DBGTAG(34)
                        removeBox(Rect(Position(bb.xa - tbb.xi + sx, bb.yi - tbb.ya + sy),
                                        Position(xpinf, bb.ya - tbb.yi + sy)), tbb, tsb, org, i);
                        break;
                    default :
                        break;
                }
            }

            if (vmax < cmin - lmargin || vmin > cmax + lmargin || omax < otmin - lmargin || omin > otmax + lmargin)
                continue;

            // Process sub-boxes that are defined for this glyph.
            // We only need to do this if there was in fact a collision with the main octabox.
            uint8 numsub = gc.numSubBounds(gid);
            if (numsub > 0)
            {
                bool anyhits = false;
                for (int j = 0; j < numsub; ++j)
                {
                    const BBox &sbb = gc.getSubBoundingBBox(gid, j);
                    const SlantBox &ssb = gc.getSubBoundingSlantBox(gid, j);
                    switch (i) {
                        case 0 :    // x
                            vmin = max(max(sbb.xi-tbb.xa+sx, ssb.di-tsb.da+sd+ty), ssb.si-tsb.sa+ss-ty);
                            vmax = min(min(sbb.xa-tbb.xi+sx, ssb.da-tsb.di+sd+ty), ssb.sa-tsb.si+ss-ty);
                            omin = sbb.yi + sy;
                            omax = sbb.ya + sy;
                            break;
                        case 1 :    // y
                            vmin = max(max(sbb.yi-tbb.ya+sy, tsb.di-ssb.da-sd+tx), ssb.si-tsb.sa+ss-tx);
                            vmax = min(min(sbb.ya-tbb.yi+sy, tsb.da-ssb.di-sd+tx), ssb.sa-tsb.si+ss-tx);
                            omin = sbb.xi + sx;
                            omax = sbb.xa + sx;
                            break;
                        case 2 :    // sum
                            vmin = max(max(ssb.si-tsb.sa+ss, 2*(sbb.yi-tbb.ya+sy)+td), 2*(sbb.xi-tbb.xa+sx)-td);
                            vmax = min(min(ssb.sa-tsb.si+ss, 2*(sbb.ya-tbb.yi+sy)+td), 2*(sbb.xa-tbb.xi+sx)-td);
                            omin = ssb.di + sd;
                            omax = ssb.da + sd;
                            break;
                        case 3 :    // diff
                            vmin = max(max(ssb.di-tsb.da+sd, 2*(sbb.xi-tbb.xa+sx)-ts), -2*(sbb.ya-tbb.yi+sy)+ts);
                            vmax = min(min(ssb.da-tsb.di+sd, 2*(sbb.xa-tbb.xi+sx)-ts), -2*(sbb.yi-tbb.ya+sy)+ts);
                            omin = ssb.si + ss;
                            omax = ssb.sa + ss;
                            break;
                    }
                    if (vmax < cmin - lmargin || vmin > cmax + lmargin || omax < otmin - lmargin || omin > otmax + lmargin)
                        continue;

#if !defined GRAPHITE2_NTRACING
                    if (dbgout)
                        dbgout->setenv(1, reinterpret_cast<void *>(j));
#endif
                    if (omin > otmax)
                        _ranges[i].weightedAxis(i, vmin - lmargin, vmax + lmargin, 0, 0, 0, 0, 0,
                                                sqr(lmargin - omin + otmax) * _marginWt, false);
                    else if (omax < otmin)
                        _ranges[i].weightedAxis(i, vmin - lmargin, vmax + lmargin, 0, 0, 0, 0, 0,
                                                sqr(lmargin - otmin + omax) * _marginWt, false);
                    else
                        _ranges[i].exclude_with_margins(vmin, vmax, i);
                    anyhits = true;
                }
                if (anyhits)
                    isCol = true;
            }
            else // no sub-boxes
            {
#if !defined GRAPHITE2_NTRACING
                    if (dbgout)
                        dbgout->setenv(1, reinterpret_cast<void *>(-1));
#endif
                isCol = true;
                if (omin > otmax)
                    _ranges[i].weightedAxis(i, vmin - lmargin, vmax + lmargin, 0, 0, 0, 0, 0,
                                            sqr(lmargin - omin + otmax) * _marginWt, false);
                else if (omax < otmin)
                    _ranges[i].weightedAxis(i, vmin - lmargin, vmax + lmargin, 0, 0, 0, 0, 0,
                                            sqr(lmargin - otmin + omax) * _marginWt, false);
                else
                    _ranges[i].exclude_with_margins(vmin, vmax, i);

            }
        }
    }
    bool res = true;
    if (cslot->exclGlyph() > 0 && gc.check(cslot->exclGlyph()) && !isExclusion)
    {
        // Set up the bogus slot representing the exclusion glyph.
        Slot *exclSlot = seg->newSlot();
        if (!exclSlot)
            return res;
        exclSlot->setGlyph(seg, cslot->exclGlyph());
        Position exclOrigin(slot->origin() + cslot->exclOffset());
        exclSlot->origin(exclOrigin);
        SlotCollision exclInfo(seg, exclSlot);
        res &= mergeSlot(seg, exclSlot, &exclInfo, currShift, isAfter, sameCluster, isCol, true, dbgout );
        seg->freeSlot(exclSlot);
    }
    hasCol |= isCol;
    return res;

}   // end of ShiftCollider::mergeSlot


// Figure out where to move the target glyph to, and return the amount to shift by.
Position ShiftCollider::resolve(GR_MAYBE_UNUSED Segment *seg, bool &isCol, GR_MAYBE_UNUSED json * const dbgout)
{
    float tbase;
    float totalCost = (float)(std::numeric_limits<float>::max() / 2);
    Position resultPos = Position(0, 0);
#if !defined GRAPHITE2_NTRACING
	int bestAxis = -1;
    if (dbgout)
    {
		outputJsonDbgStartSlot(dbgout, seg);
        *dbgout << "vectors" << json::array;
    }
#endif
    isCol = true;
    for (int i = 0; i < 4; ++i)
    {
        float bestCost = -1;
        float bestPos;
        // Calculate the margin depending on whether we are moving diagonally or not:
        switch (i) {
            case 0 :	// x direction
                tbase = _currOffset.x;
                break;
            case 1 :	// y direction
                tbase = _currOffset.y;
                break;
            case 2 :	// sum (negatively-sloped diagonals)
                tbase = _currOffset.x + _currOffset.y;
                break;
            case 3 :	// diff (positively-sloped diagonals)
                tbase = _currOffset.x - _currOffset.y;
                break;
        }
        Position testp;
        bestPos = _ranges[i].closest(0, bestCost) - tbase;     // Get the best relative position
#if !defined GRAPHITE2_NTRACING
        if (dbgout)
            outputJsonDbgOneVector(dbgout, seg, i, tbase, bestCost, bestPos) ;
#endif
        if (bestCost >= 0.0f)
        {
            isCol = false;
            switch (i) {
                case 0 : testp = Position(bestPos, _currShift.y); break;
                case 1 : testp = Position(_currShift.x, bestPos); break;
                case 2 : testp = Position(0.5f * (_currShift.x - _currShift.y + bestPos), 0.5f * (_currShift.y - _currShift.x + bestPos)); break;
                case 3 : testp = Position(0.5f * (_currShift.x + _currShift.y + bestPos), 0.5f * (_currShift.x + _currShift.y - bestPos)); break;
            }
            if (bestCost < totalCost - 0.01f)
            {
                totalCost = bestCost;
                resultPos = testp;
#if !defined GRAPHITE2_NTRACING
                bestAxis = i;
#endif
            }
        }
    }  // end of loop over 4 directions

#if !defined GRAPHITE2_NTRACING
    if (dbgout)
        outputJsonDbgEndSlot(dbgout, resultPos, bestAxis, isCol);
#endif

    return resultPos;

}   // end of ShiftCollider::resolve


#if !defined GRAPHITE2_NTRACING

void ShiftCollider::outputJsonDbg(json * const dbgout, Segment *seg, int axis)
{
    int axisMax = axis;
    if (axis < 0) // output all axes
    {
        *dbgout << "gid" << _target->gid()
            << "limit" << _limit
            << "target" << json::object
                << "origin" << _target->origin()
                << "margin" << _margin
                << "bbox" << seg->theGlyphBBoxTemporary(_target->gid())
                << "slantbox" << seg->getFace()->glyphs().slant(_target->gid())
                << json::close; // target object
        *dbgout << "ranges" << json::array;
        axis = 0;
        axisMax = 3;
    }
    for (int iAxis = axis; iAxis <= axisMax; ++iAxis)
    {
        *dbgout << json::flat << json::array << _ranges[iAxis].position();
        for (Zones::const_iterator s = _ranges[iAxis].begin(), e = _ranges[iAxis].end(); s != e; ++s)
            *dbgout << json::flat << json::array
                        << Position(s->x, s->xm) << s->sm << s->smx << s->c
                    << json::close;
        *dbgout << json::close;
    }
    if (axis < axisMax) // looped through the _ranges array for all axes
        *dbgout << json::close; // ranges array
}

void ShiftCollider::outputJsonDbgStartSlot(json * const dbgout, Segment *seg)
{
        *dbgout << json::object // slot - not closed till the end of the caller method
                << "slot" << objectid(dslot(seg, _target))
				<< "gid" << _target->gid()
                << "limit" << _limit
                << "target" << json::object
                    << "origin" << _origin
                    << "currShift" << _currShift
                    << "currOffset" << seg->collisionInfo(_target)->offset()
                    << "bbox" << seg->theGlyphBBoxTemporary(_target->gid())
                    << "slantBox" << seg->getFace()->glyphs().slant(_target->gid())
                    << "fix" << "shift";
        *dbgout     << json::close; // target object
}

void ShiftCollider::outputJsonDbgEndSlot(GR_MAYBE_UNUSED json * const dbgout,
	 Position resultPos, int bestAxis, bool isCol)
{
    *dbgout << json::close // vectors array
    << "result" << resultPos
	//<< "scraping" << _scraping[bestAxis]
	<< "bestAxis" << bestAxis
    << "stillBad" << isCol
    << json::close; // slot object
}

void ShiftCollider::outputJsonDbgOneVector(json * const dbgout, Segment *seg, int axis,
	float tleft, float bestCost, float bestVal)
{
	const char * label;
	switch (axis)
	{
		case 0:	label = "x";			break;
		case 1:	label = "y";			break;
		case 2:	label = "sum (NE-SW)";	break;
		case 3:	label = "diff (NW-SE)";	break;
		default: label = "???";			break;
	}

	*dbgout << json::object // vector
		<< "direction" << label
		<< "targetMin" << tleft;

	outputJsonDbgRemovals(dbgout, axis, seg);

    *dbgout << "ranges";
    outputJsonDbg(dbgout, seg, axis);

    *dbgout << "bestCost" << bestCost
        << "bestVal" << bestVal + tleft
        << json::close; // vectors object
}

void ShiftCollider::outputJsonDbgRemovals(json * const dbgout, int axis, Segment *seg)
{
    *dbgout << "removals" << json::array;
    _ranges[axis].jsonDbgOut(seg);
    *dbgout << json::close; // removals array
}

#endif // !defined GRAPHITE2_NTRACING


////    KERN-COLLIDER    ////

inline
static float localmax (float al, float au, float bl, float bu, float x)
{
    if (al < bl)
    { if (au < bu) return au < x ? au : x; }
    else if (au > bu) return bl < x ? bl : x;
    return x;
}

inline
static float localmin(float al, float au, float bl, float bu, float x)
{
    if (bl > al)
    { if (bu > au) return bl > x ? bl : x; }
    else if (au > bu) return al > x ? al : x;
    return x;
}

// Return the given edge of the glyph at height y, taking any slant box into account.
static float get_edge(Segment *seg, const Slot *s, const Position &shift, float y, float width, float margin, bool isRight)
{
    const GlyphCache &gc = seg->getFace()->glyphs();
    unsigned short gid = s->gid();
    float sx = s->origin().x + shift.x;
    float sy = s->origin().y + shift.y;
    uint8 numsub = gc.numSubBounds(gid);
    float res = isRight ? (float)-1e38 : (float)1e38;

    if (numsub > 0)
    {
        for (int i = 0; i < numsub; ++i)
        {
            const BBox &sbb = gc.getSubBoundingBBox(gid, i);
            const SlantBox &ssb = gc.getSubBoundingSlantBox(gid, i);
            if (sy + sbb.yi - margin > y + width / 2 || sy + sbb.ya + margin < y - width / 2)
                continue;
            if (isRight)
            {
                float x = sx + sbb.xa + margin;
                if (x > res)
                {
                    float td = sx - sy + ssb.da + margin + y;
                    float ts = sx + sy + ssb.sa + margin - y;
                    x = localmax(td - width / 2, td + width / 2,  ts - width / 2, ts + width / 2, x);
                    if (x > res)
                        res = x;
                }
            }
            else
            {
                float x = sx + sbb.xi - margin;
                if (x < res)
                {
                    float td = sx - sy + ssb.di - margin + y;
                    float ts = sx + sy + ssb.si - margin - y;
                    x = localmin(td - width / 2, td + width / 2, ts - width / 2, ts + width / 2, x);
                    if (x < res)
                        res = x;
                }
            }
        }
    }
    else
    {
        const BBox &bb = gc.getBoundingBBox(gid);
        const SlantBox &sb = gc.getBoundingSlantBox(gid);
        if (sy + bb.yi - margin > y + width / 2 || sy + bb.ya + margin < y - width / 2)
            return res;
        float td = sx - sy + y;
        float ts = sx + sy - y;
        if (isRight)
            res = localmax(td + sb.da - width / 2, td + sb.da + width / 2, ts + sb.sa - width / 2, ts + sb.sa + width / 2, sx + bb.xa) + margin;
        else
            res = localmin(td + sb.di - width / 2, td + sb.di + width / 2, ts + sb.si - width / 2, ts + sb.si + width / 2, sx + bb.xi) - margin;
    }
    return res;
}


bool KernCollider::initSlot(Segment *seg, Slot *aSlot, const Rect &limit, float margin,
    const Position &currShift, const Position &offsetPrev, int dir,
    float ymin, float ymax, GR_MAYBE_UNUSED json * const dbgout)
{
    const GlyphCache &gc = seg->getFace()->glyphs();
    const Slot *base = aSlot;
    // const Slot *last = aSlot;
    const Slot *s;
    int numSlices;
    while (base->attachedTo())
        base = base->attachedTo();
    if (margin < 10) margin = 10;

    _limit = limit;
    _offsetPrev = offsetPrev; // kern from a previous pass

    // Calculate the height of the glyph and how many horizontal slices to use.
    if (_maxy >= 1e37f)
    {
        _sliceWidth = margin / 1.5f;
        _maxy = ymax + margin;
        _miny = ymin - margin;
        numSlices = int((_maxy - _miny + 2) / (_sliceWidth / 1.5f) + 1.f);  // +2 helps with rounding errors
        _edges.clear();
        _edges.insert(_edges.begin(), numSlices, (dir & 1) ? 1e38f : -1e38f);
        _xbound = (dir & 1) ? (float)1e38f : (float)-1e38f;
    }
    else if (_maxy != ymax || _miny != ymin)
    {
        if (_miny != ymin)
        {
            numSlices = int((ymin - margin - _miny) / _sliceWidth - 1);
            _miny += numSlices * _sliceWidth;
            if (numSlices < 0)
                _edges.insert(_edges.begin(), -numSlices, (dir & 1) ? 1e38f : -1e38f);
            else if ((unsigned)numSlices < _edges.size())    // this shouldn't fire since we always grow the range
            {
                Vector<float>::iterator e = _edges.begin();
                while (numSlices--)
                    ++e;
                _edges.erase(_edges.begin(), e);
            }
        }
        if (_maxy != ymax)
        {
            numSlices = int((ymax + margin - _miny) / _sliceWidth + 1);
            _maxy = numSlices * _sliceWidth + _miny;
            if (numSlices > (int)_edges.size())
                _edges.insert(_edges.end(), numSlices - _edges.size(), (dir & 1) ? 1e38f : -1e38f);
            else if (numSlices < (int)_edges.size())   // this shouldn't fire since we always grow the range
            {
                while ((int)_edges.size() > numSlices)
                    _edges.pop_back();
            }
        }
        goto done;
    }
    numSlices = int(_edges.size());

#if !defined GRAPHITE2_NTRACING
    // Debugging
    _seg = seg;
    _slotNear.clear();
    _slotNear.insert(_slotNear.begin(), numSlices, NULL);
    _nearEdges.clear();
    _nearEdges.insert(_nearEdges.begin(), numSlices, (dir & 1) ? -1e38f : +1e38f);
#endif

    // Determine the trailing edge of each slice (ie, left edge for a RTL glyph).
    for (s = base; s; s = s->nextInCluster(s))
    {
        SlotCollision *c = seg->collisionInfo(s);
        if (!gc.check(s->gid()))
            return false;
        const BBox &bs = gc.getBoundingBBox(s->gid());
        float x = s->origin().x + c->shift().x + ((dir & 1) ? bs.xi : bs.xa);
        // Loop over slices.
        // Note smin might not be zero if glyph s is not at the bottom of the cluster; similarly for smax.
        float toffset = c->shift().y - _miny + 1 + s->origin().y;
        int smin = max(0, int((bs.yi + toffset) / _sliceWidth));
        int smax = min(numSlices - 1, int((bs.ya + toffset) / _sliceWidth + 1));
        for (int i = smin; i <= smax; ++i)
        {
            float t;
            float y = _miny - 1 + (i + .5f) * _sliceWidth; // vertical center of slice
            if ((dir & 1) && x < _edges[i])
            {
                t = get_edge(seg, s, c->shift(), y, _sliceWidth, margin, false);
                if (t < _edges[i])
                {
                    _edges[i] = t;
                    if (t < _xbound)
                        _xbound = t;
                }
            }
            else if (!(dir & 1) && x > _edges[i])
            {
                t = get_edge(seg, s, c->shift(), y, _sliceWidth, margin, true);
                if (t > _edges[i])
                {
                    _edges[i] = t;
                    if (t > _xbound)
                        _xbound = t;
                }
            }
        }
    }
    done:
    _mingap = (float)1e37;      // less than 1e38 s.t. 1e38-_mingap is really big
    _target = aSlot;
    _margin = margin;
    _currShift = currShift;
    return true;
}   // end of KernCollider::initSlot


// Determine how much the target slot needs to kern away from the given slot.
// In other words, merge information from given slot's position with what the target slot knows
// about how it can kern.
// Return false if we know there is no collision, true if we think there might be one.
bool KernCollider::mergeSlot(Segment *seg, Slot *slot, const Position &currShift, float currSpace, int dir, GR_MAYBE_UNUSED json * const dbgout)
{
    int rtl = (dir & 1) * 2 - 1;
    if (!seg->getFace()->glyphs().check(slot->gid()))
        return false;
    const Rect &bb = seg->theGlyphBBoxTemporary(slot->gid());
    const float sx = slot->origin().x + currShift.x;
    float x = (sx + (rtl > 0 ? bb.tr.x : bb.bl.x)) * rtl;
    // this isn't going to reduce _mingap so skip
    if (_hit && x < rtl * (_xbound - _mingap - currSpace))
        return false;

    const float sy = slot->origin().y + currShift.y;
    int smin = max(1, int((bb.bl.y + (1 - _miny + sy)) / _sliceWidth + 1)) - 1;
    int smax = min((int)_edges.size() - 2, int((bb.tr.y + (1 - _miny + sy)) / _sliceWidth + 1)) + 1;
    if (smin > smax)
        return false;
    bool collides = false;
    bool nooverlap = true;

    for (int i = smin; i <= smax; ++i)
    {
        float here = _edges[i] * rtl;
        if (here > (float)9e37)
            continue;
        if (!_hit || x > here - _mingap - currSpace)
        {
            float y = (float)(_miny - 1 + (i + .5f) * _sliceWidth);  // vertical center of slice
            // 2 * currSpace to account for the space that is already separating them and the space we want to add
            float m = get_edge(seg, slot, currShift, y, _sliceWidth, 0., rtl > 0) * rtl + 2 * currSpace;
            if (m < (float)-8e37)       // only true if the glyph has a gap in it
                continue;
            nooverlap = false;
            float t = here - m;
            // _mingap is positive to shrink
            if (t < _mingap || (!_hit && !collides))
            {
                _mingap = t;
                collides = true;
            }
#if !defined GRAPHITE2_NTRACING
            // Debugging - remember the closest neighboring edge for this slice.
            if (m > rtl * _nearEdges[i])
            {
                _slotNear[i] = slot;
                _nearEdges[i] = m * rtl;
            }
#endif
        }
        else
            nooverlap = false;
    }
    if (nooverlap)
        _mingap = max(_mingap, _xbound - rtl * (currSpace + _margin + x));
    if (collides && !nooverlap)
        _hit = true;
    return collides | nooverlap;   // note that true is not a necessarily reliable value

}   // end of KernCollider::mergeSlot


// Return the amount to kern by.
Position KernCollider::resolve(GR_MAYBE_UNUSED Segment *seg, GR_MAYBE_UNUSED Slot *slot,
        int dir, GR_MAYBE_UNUSED json * const dbgout)
{
    float resultNeeded = (1 - 2 * (dir & 1)) * _mingap;
    // float resultNeeded = (1 - 2 * (dir & 1)) * (_mingap - margin);
    float result = min(_limit.tr.x - _offsetPrev.x, max(resultNeeded, _limit.bl.x - _offsetPrev.x));

#if !defined GRAPHITE2_NTRACING
    if (dbgout)
    {
        *dbgout << json::object // slot
                << "slot" << objectid(dslot(seg, _target))
				<< "gid" << _target->gid()
                << "limit" << _limit
                << "miny" << _miny
                << "maxy" << _maxy
                << "slicewidth" << _sliceWidth
                << "target" << json::object
                    << "origin" << _target->origin()
                    //<< "currShift" << _currShift
                    << "offsetPrev" << _offsetPrev
                    << "bbox" << seg->theGlyphBBoxTemporary(_target->gid())
                    << "slantBox" << seg->getFace()->glyphs().slant(_target->gid())
                    << "fix" << "kern"
                    << json::close; // target object

        *dbgout << "slices" << json::array;
        for (int is = 0; is < (int)_edges.size(); is++)
        {
            *dbgout << json::flat << json::object
                << "i" << is
                << "targetEdge" << _edges[is]
                << "neighbor" << objectid(dslot(seg, _slotNear[is]))
                << "nearEdge" << _nearEdges[is]
                << json::close;
        }
        *dbgout << json::close; // slices array

        *dbgout
            << "xbound" << _xbound
            << "minGap" << _mingap
            << "needed" << resultNeeded
            << "result" << result
            << "stillBad" << (result != resultNeeded)
            << json::close; // slot object
    }
#endif

    return Position(result, 0.);

}   // end of KernCollider::resolve

void KernCollider::shift(const Position &mv, int dir)
{
    for (Vector<float>::iterator e = _edges.begin(); e != _edges.end(); ++e)
        *e += mv.x;
    _xbound += (1 - 2 * (dir & 1)) * mv.x;
}

////    SLOT-COLLISION    ////

// Initialize the collision attributes for the given slot.
SlotCollision::SlotCollision(Segment *seg, Slot *slot)
{
    initFromSlot(seg, slot);
}

void SlotCollision::initFromSlot(Segment *seg, Slot *slot)
{
    // Initialize slot attributes from glyph attributes.
	// The order here must match the order in the grcompiler code,
	// GrcSymbolTable::AssignInternalGlyphAttrIDs.
    uint16 gid = slot->gid();
    uint16 aCol = seg->silf()->aCollision(); // flags attr ID
    const GlyphFace * glyphFace = seg->getFace()->glyphs().glyphSafe(gid);
    if (!glyphFace)
        return;
    const sparse &p = glyphFace->attrs();
    _flags = p[aCol];
    _limit = Rect(Position(int16(p[aCol+1]), int16(p[aCol+2])),
                  Position(int16(p[aCol+3]), int16(p[aCol+4])));
    _margin = p[aCol+5];
    _marginWt = p[aCol+6];

    _seqClass = p[aCol+7];
	_seqProxClass = p[aCol+8];
    _seqOrder = p[aCol+9];
	_seqAboveXoff = p[aCol+10];
	_seqAboveWt = p[aCol+11];
	_seqBelowXlim = p[aCol+12];
	_seqBelowWt = p[aCol+13];
	_seqValignHt = p[aCol+14];
	_seqValignWt = p[aCol+15];

    // These attributes do not have corresponding glyph attribute:
    _exclGlyph = 0;
    _exclOffset = Position(0, 0);
}

float SlotCollision::getKern(int dir) const
{
    if ((_flags & SlotCollision::COLL_KERN) != 0)
        return float(_shift.x * ((dir & 1) ? -1 : 1));
    else
    	return 0;
}

bool SlotCollision::ignore() const
{
	return ((flags() & SlotCollision::COLL_IGNORE) || (flags() & SlotCollision::COLL_ISSPACE));
}
