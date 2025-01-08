// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2010, SIL International, All rights reserved.

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
