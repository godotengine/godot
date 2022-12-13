// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2011, SIL International, All rights reserved.

//  debug.h
//
//  Created on: 22 Dec 2011
//      Author: tim

#pragma once

#if !defined GRAPHITE2_NTRACING

#include <utility>
#include "inc/json.h"
#include "inc/Position.h"

namespace graphite2
{

class CharInfo;
class Segment;
class Slot;

typedef std::pair<const Segment * const, const Slot * const>    dslot;
struct objectid
{
    char name[16];
    objectid(const dslot &) throw();
    objectid(const Segment * const p) throw();
};


json & operator << (json & j, const Position &) throw();
json & operator << (json & j, const Rect &) throw();
json & operator << (json & j, const CharInfo &) throw();
json & operator << (json & j, const dslot &) throw();
json & operator << (json & j, const objectid &) throw();
json & operator << (json & j, const telemetry &) throw();



inline
json & operator << (json & j, const Position & p) throw()
{
    return j << json::flat << json::array << p.x << p.y << json::close;
}


inline
json & operator << (json & j, const Rect & p) throw()
{
    return j << json::flat << json::array << p.bl.x << p.bl.y << p.tr.x << p.tr.y << json::close;
}


inline
json & operator << (json & j, const objectid & sid) throw()
{
    return j << sid.name;
}


} // namespace graphite2

#endif //!defined GRAPHITE2_NTRACING

