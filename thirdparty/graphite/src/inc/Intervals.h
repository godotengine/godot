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

#include <utility>

#include "inc/Main.h"
#include "inc/List.h"
#include "inc/json.h"
#include "inc/Position.h"

// An IntervalSet represents the possible movement of a given glyph in a given direction
// (horizontally, vertically, or diagonally).
// A vector is needed to represent disjoint ranges, eg, -300..-150, 20..200, 500..750.
// Each pair represents the min/max of a sub-range.

namespace graphite2 {

class Segment;

enum zones_t {SD, XY};

class Zones
{
    struct Exclusion
    {
        template<zones_t O>
        static Exclusion weighted(float xmin, float xmax, float f, float a0,
                float m, float xi, float ai, float c, bool nega);

        float   x,  // x position
                xm, // xmax position
                c,  // constant + sum(MiXi^2)
                sm, // sum(Mi)
                smx; // sum(MiXi)
        bool    open;

        Exclusion(float x, float w, float smi, float smxi, float c);
        Exclusion & operator += (Exclusion const & rhs);
        uint8 outcode(float p) const;

        Exclusion   split_at(float p);
        void        left_trim(float p);

        bool        track_cost(float & cost, float & x, float origin) const;

    private:
        float test_position(float origin) const;
        float cost(float x) const;
     };

    typedef Vector<Exclusion>                   exclusions;

    typedef exclusions::iterator                iterator;
    typedef Exclusion *                         pointer;
    typedef Exclusion &                         reference;
    typedef std::reverse_iterator<iterator>     reverse_iterator;

public:
    typedef exclusions::const_iterator              const_iterator;
    typedef Exclusion const *                       const_pointer;
    typedef Exclusion const &                       const_reference;
    typedef std::reverse_iterator<const_iterator>   const_reverse_iterator;

#if !defined GRAPHITE2_NTRACING
    struct Debug
    {
        Exclusion       _excl;
        bool            _isdel;
        Vector<void *>  _env;

        Debug(Exclusion *e, bool isdel, json *dbg) : _excl(*e), _isdel(isdel), _env(dbg->getenvs()) { };
    };

    typedef Vector<Debug>                       debugs;
    typedef debugs::const_iterator                    idebugs;
    void addDebug(Exclusion *e);
    void removeDebug(float pos, float posm);
    void setdebug(json *dbgout) { _dbg = dbgout; }
    idebugs dbgs_begin() const { return _dbgs.begin(); }
    idebugs dbgs_end() const { return _dbgs.end(); }
    void jsonDbgOut(Segment *seg) const;
    Position position() const { return Position(_pos, _posm); }
#endif

    Zones();
    template<zones_t O>
    void initialise(float xmin, float xmax, float margin_len, float margin_weight, float ao);

    void exclude(float xmin, float xmax);
    void exclude_with_margins(float xmin, float xmax, int axis);

    template<zones_t O>
    void weighted(float xmin, float xmax, float f, float a0, float mi, float xi, float ai, float c, bool nega);
    void weightedAxis(int axis, float xmin, float xmax, float f, float a0, float mi, float xi, float ai, float c, bool nega);

    float closest( float origin, float &cost) const;

    const_iterator begin() const { return _exclusions.begin(); }
    const_iterator end() const { return _exclusions.end(); }

private:
    exclusions  _exclusions;
#if !defined GRAPHITE2_NTRACING
    json      * _dbg;
    debugs      _dbgs;
#endif
    float       _margin_len,
                _margin_weight,
                _pos,
                _posm;

    void            insert(Exclusion e);
    void            remove(float x, float xm);
    const_iterator  find_exclusion_under(float x) const;
};


inline
Zones::Zones()
: _margin_len(0), _margin_weight(0), _pos(0), _posm(0)
{
#if !defined GRAPHITE2_NTRACING
    _dbg = 0;
#endif
    _exclusions.reserve(8);
}

inline
Zones::Exclusion::Exclusion(float x_, float xm_, float smi, float smxi, float c_)
: x(x_), xm(xm_), c(c_), sm(smi), smx(smxi), open(false)
{ }

template<zones_t O>
inline
void Zones::initialise(float xmin, float xmax, float margin_len,
        float margin_weight, float a0)
{
    _margin_len = margin_len;
    _margin_weight = margin_weight;
    _pos = xmin;
    _posm = xmax;
    _exclusions.clear();
    _exclusions.push_back(Exclusion::weighted<O>(xmin, xmax, 1, a0, 0, 0, 0, 0, false));
    _exclusions.front().open = true;
#if !defined GRAPHITE2_NTRACING
    _dbgs.clear();
#endif
}

inline
void Zones::exclude(float xmin, float xmax) {
    remove(xmin, xmax);
}

template<zones_t O>
inline
void Zones::weighted(float xmin, float xmax, float f, float a0,
        float m, float xi, float ai, float c, bool nega) {
    insert(Exclusion::weighted<O>(xmin, xmax, f, a0, m, xi, ai, c, nega));
}

inline
void Zones::weightedAxis(int axis, float xmin, float xmax, float f, float a0,
        float m, float xi, float ai, float c, bool nega) {
    if (axis < 2)
        weighted<XY>(xmin, xmax, f, a0, m, xi, ai, c, nega);
    else
        weighted<SD>(xmin, xmax, f, a0, m, xi, ai, c, nega);
}

#if !defined GRAPHITE2_NTRACING
inline
void Zones::addDebug(Exclusion *e) {
    if (_dbg)
        _dbgs.push_back(Debug(e, false, _dbg));
}

inline
void Zones::removeDebug(float pos, float posm) {
    if (_dbg)
    {
        Exclusion e(pos, posm, 0, 0, 0);
        _dbgs.push_back(Debug(&e, true, _dbg));
    }
}
#endif

template<>
inline
Zones::Exclusion Zones::Exclusion::weighted<XY>(float xmin, float xmax, float f, float a0,
        float m, float xi, GR_MAYBE_UNUSED float ai, float c, GR_MAYBE_UNUSED bool nega) {
    return Exclusion(xmin, xmax,
            m + f,
            m * xi,
            m * xi * xi + f * a0 * a0 + c);
}

template<>
inline
Zones::Exclusion Zones::Exclusion::weighted<SD>(float xmin, float xmax, float f, float a0,
        float m, float xi, float ai,float c, bool nega) {
    float xia = nega ? xi - ai : xi + ai;
    return Exclusion(xmin, xmax,
            0.25f * (m + 2.f * f),
            0.25f * m * xia,
            0.25f * (m * xia * xia + 2.f * f * a0 * a0) + c);
}

} // end of namespace graphite2
