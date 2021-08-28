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
