// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2010, SIL International, All rights reserved.

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
