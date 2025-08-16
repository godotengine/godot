// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2010, SIL International, All rights reserved.

#include <algorithm>
#include <cmath>
#include <limits>

#include "inc/Intervals.h"
#include "inc/Segment.h"
#include "inc/Slot.h"
#include "inc/debug.h"
#include "inc/bits.h"

using namespace graphite2;

#include <cmath>

inline
Zones::Exclusion  Zones::Exclusion::split_at(float p) {
    Exclusion r(*this);
    r.xm = x = p;
    return r;
}

inline
void Zones::Exclusion::left_trim(float p) {
    x = p;
}

inline
Zones::Exclusion & Zones::Exclusion::operator += (Exclusion const & rhs) {
    c += rhs.c; sm += rhs.sm; smx += rhs.smx; open = false;
    return *this;
}

inline
uint8 Zones::Exclusion::outcode(float val) const {
    float p = val;
    //float d = std::numeric_limits<float>::epsilon();
    float d = 0.;
    return ((p - xm >= d) << 1) | (x - p > d);
}

void Zones::exclude_with_margins(float xmin, float xmax, int axis) {
    remove(xmin, xmax);
    weightedAxis(axis, xmin-_margin_len, xmin, 0, 0, _margin_weight, xmin-_margin_len, 0, 0, false);
    weightedAxis(axis, xmax, xmax+_margin_len, 0, 0, _margin_weight, xmax+_margin_len, 0, 0, false);
}

namespace
{

inline
bool separated(float a, float b) {
    return a != b;
    //int exp;
    //float res = frexpf(fabs(a - b), &exp);
    //return (*(unsigned int *)(&res) > 4);
    //return std::fabs(a-b) > std::numeric_limits<float>::epsilon(); // std::epsilon may not work. but 0.5 fails exising 64 bit tests
    //return std::fabs(a-b) > 0.5f;
}

}

void Zones::insert(Exclusion e)
{
#if !defined GRAPHITE2_NTRACING
    addDebug(&e);
#endif
    e.x = max(e.x, _pos);
    e.xm = min(e.xm, _posm);
    if (e.x >= e.xm) return;

    for (iterator i = _exclusions.begin(), ie = _exclusions.end(); i != ie && e.x < e.xm; ++i)
    {
        const uint8 oca = e.outcode(i->x),
                    ocb = e.outcode(i->xm);
        if ((oca & ocb) != 0) continue;

        switch (oca ^ ocb)  // What kind of overlap?
        {
        case 0:     // e completely covers i
            // split e at i.x into e1,e2
            // split e2 at i.mx into e2,e3
            // drop e1 ,i+e2, e=e3
            *i += e;
            e.left_trim(i->xm);
            break;
        case 1:     // e overlaps on the rhs of i
            // split i at e->x into i1,i2
            // split e at i.mx into e1,e2
            // trim i1, insert i2+e1, e=e2
            if (!separated(i->xm, e.x)) break;
            if (separated(i->x,e.x))   { i = _exclusions.insert(i,i->split_at(e.x)); ++i; }
            *i += e;
            e.left_trim(i->xm);
            break;
        case 2:     // e overlaps on the lhs of i
            // split e at i->x into e1,e2
            // split i at e.mx into i1,i2
            // drop e1, insert e2+i1, trim i2
            if (!separated(e.xm, i->x)) return;
            if (separated(e.xm, i->xm)) i = _exclusions.insert(i,i->split_at(e.xm));
            *i += e;
            return;
        case 3:     // i completely covers e
            // split i at e.x into i1,i2
            // split i2 at e.mx into i2,i3
            // insert i1, insert e+i2
            if (separated(e.xm, i->xm)) i = _exclusions.insert(i,i->split_at(e.xm));
            i = _exclusions.insert(i, i->split_at(e.x));
            *++i += e;
            return;
        }

        ie = _exclusions.end();
    }
}


void Zones::remove(float x, float xm)
{
#if !defined GRAPHITE2_NTRACING
    removeDebug(x, xm);
#endif
    x = max(x, _pos);
    xm = min(xm, _posm);
    if (x >= xm) return;

    for (iterator i = _exclusions.begin(), ie = _exclusions.end(); i != ie; ++i)
    {
        const uint8 oca = i->outcode(x),
                    ocb = i->outcode(xm);
        if ((oca & ocb) != 0)   continue;

        switch (oca ^ ocb)  // What kind of overlap?
        {
        case 0:     // i completely covers e
            if (separated(i->x, x))  { i = _exclusions.insert(i,i->split_at(x)); ++i; }
            GR_FALLTHROUGH;
            // no break
        case 1:     // i overlaps on the rhs of e
            i->left_trim(xm);
            return;
        case 2:     // i overlaps on the lhs of e
            i->xm = x;
            if (separated(i->x, i->xm)) break;
            GR_FALLTHROUGH;
            // no break
        case 3:     // e completely covers i
            i = _exclusions.erase(i);
            --i;
            break;
        }

        ie = _exclusions.end();
    }
}


Zones::const_iterator Zones::find_exclusion_under(float x) const
{
    size_t l = 0, h = _exclusions.size();

    while (l < h)
    {
        size_t const p = (l+h) >> 1;
        switch (_exclusions[p].outcode(x))
        {
        case 0 : return _exclusions.begin()+p;
        case 1 : h = p; break;
        case 2 :
        case 3 : l = p+1; break;
        }
    }

    return _exclusions.begin()+l;
}


float Zones::closest(float origin, float & cost) const
{
    float best_c = std::numeric_limits<float>::max(),
          best_x = 0;

    const const_iterator start = find_exclusion_under(origin);

    // Forward scan looking for lowest cost
    for (const_iterator i = start, ie = _exclusions.end(); i != ie; ++i)
        if (i->track_cost(best_c, best_x, origin)) break;

    // Backward scan looking for lowest cost
    //  We start from the exclusion to the immediate left of start since we've
    //  already tested start with the right most scan above.
    for (const_iterator i = start-1, ie = _exclusions.begin()-1; i != ie; --i)
        if (i->track_cost(best_c, best_x, origin)) break;

    cost = (best_c == std::numeric_limits<float>::max() ? -1 : best_c);
    return best_x;
}


// Cost and test position functions

bool Zones::Exclusion::track_cost(float & best_cost, float & best_pos, float origin) const {
    const float p = test_position(origin),
                localc = cost(p - origin);
    if (open && localc > best_cost) return true;

    if (localc < best_cost)
    {
        best_cost = localc;
        best_pos = p;
    }
    return false;
}

inline
float Zones::Exclusion::cost(float p) const {
    return (sm * p - 2 * smx) * p + c;
}


float Zones::Exclusion::test_position(float origin) const {
    if (sm < 0)
    {
        // sigh, test both ends and perhaps the middle too!
        float res = x;
        float cl = cost(x);
        if (x < origin && xm > origin)
        {
            float co = cost(origin);
            if (co < cl)
            {
                cl = co;
                res = origin;
            }
        }
        float cr = cost(xm);
        return cl > cr ? xm : res;
    }
    else
    {
        float zerox = smx / sm + origin;
        if (zerox < x) return x;
        else if (zerox > xm) return xm;
        else return zerox;
    }
}


#if !defined GRAPHITE2_NTRACING

void Zones::jsonDbgOut(Segment *seg) const {

    if (_dbg)
    {
        for (Zones::idebugs s = dbgs_begin(), e = dbgs_end(); s != e; ++s)
        {
            *_dbg << json::flat << json::array
                << objectid(dslot(seg, (Slot *)(s->_env[0])))
                << reinterpret_cast<ptrdiff_t>(s->_env[1]);
            if (s->_isdel)
                *_dbg << "remove" << Position(s->_excl.x, s->_excl.xm);
            else
                *_dbg << "exclude" << json::flat << json::array
                    << s->_excl.x << s->_excl.xm
                    << s->_excl.sm << s->_excl.smx << s->_excl.c
                    << json::close;
            *_dbg << json::close;
        }
    }
}

#endif
