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
#include <cassert>
#include "graphite2/Segment.h"
#include "inc/CharInfo.h"

extern "C"
{

unsigned int gr_cinfo_unicode_char(const gr_char_info* p/*not NULL*/)
{
    assert(p);
    return p->unicodeChar();
}


int gr_cinfo_break_weight(const gr_char_info* p/*not NULL*/)
{
    assert(p);
    return p->breakWeight();
}

int gr_cinfo_after(const gr_char_info *p/*not NULL*/)
{
    assert(p);
    return p->after();
}

int gr_cinfo_before(const gr_char_info *p/*not NULL*/)
{
    assert(p);
    return p->before();
}

size_t gr_cinfo_base(const gr_char_info *p/*not NULL*/)
{
    assert(p);
    return p->base();
}

} // extern "C"
