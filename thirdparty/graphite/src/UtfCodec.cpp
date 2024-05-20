// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2010, SIL International, All rights reserved.

#include "inc/UtfCodec.h"
//using namespace graphite2;

namespace graphite2 {

}

using namespace graphite2;

const int8 _utf_codec<8>::sz_lut[16] =
{
        1,1,1,1,1,1,1,1,    // 1 byte
        0,0,0,0,            // trailing byte
        2,2,                // 2 bytes
        3,                  // 3 bytes
        4                   // 4 bytes
};

const byte  _utf_codec<8>::mask_lut[5] = {0x7f, 0xff, 0x3f, 0x1f, 0x0f};
