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
#include <cstring>
#include <cassert>
#include "inc/Main.h"
#include "inc/List.h"

namespace graphite2 {

class FeatureRef;
class FeatureMap;

class FeatureVal : public Vector<uint32>
{
public:
    FeatureVal() : m_pMap(0) { }
    FeatureVal(int num, const FeatureMap & pMap) : Vector<uint32>(num), m_pMap(&pMap) {}
    FeatureVal(const FeatureVal & rhs) : Vector<uint32>(rhs), m_pMap(rhs.m_pMap) {}

    FeatureVal & operator = (const FeatureVal & rhs) { Vector<uint32>::operator = (rhs); m_pMap = rhs.m_pMap; return *this; }

    bool operator ==(const FeatureVal & b) const
    {
        size_t n = size();
        if (n != b.size())      return false;

        for(const_iterator l = begin(), r = b.begin(); n && *l == *r; --n, ++l, ++r);

        return n == 0;
    }

    CLASS_NEW_DELETE
private:
    friend class FeatureRef;        //so that FeatureRefs can manipulate m_vec directly
    const FeatureMap* m_pMap;
};

typedef FeatureVal Features;

} // namespace graphite2


struct gr_feature_val : public graphite2::FeatureVal {};
