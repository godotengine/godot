// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2010, SIL International, All rights reserved.

#include "inc/GlyphFace.h"


using namespace graphite2;

int32 GlyphFace::getMetric(uint8 metric) const
{
    switch (metrics(metric))
    {
        case kgmetLsb       : return int32(m_bbox.bl.x);
        case kgmetRsb       : return int32(m_advance.x - m_bbox.tr.x);
        case kgmetBbTop     : return int32(m_bbox.tr.y);
        case kgmetBbBottom  : return int32(m_bbox.bl.y);
        case kgmetBbLeft    : return int32(m_bbox.bl.x);
        case kgmetBbRight   : return int32(m_bbox.tr.x);
        case kgmetBbHeight  : return int32(m_bbox.tr.y - m_bbox.bl.y);
        case kgmetBbWidth   : return int32(m_bbox.tr.x - m_bbox.bl.x);
        case kgmetAdvWidth  : return int32(m_advance.x);
        case kgmetAdvHeight : return int32(m_advance.y);
        default : return 0;
    }
}
