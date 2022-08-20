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
