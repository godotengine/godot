/* -----------------------------------------------------------------------------

    Copyright (c) 2006 Simon Brown                          si@sjbrown.co.uk

    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to
    the following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

   -------------------------------------------------------------------------- */

#ifndef SQUISH_COLOURBLOCK_H
#define SQUISH_COLOURBLOCK_H

#include "squish.h"
#include "maths.h"

namespace squish {

void WriteColourBlock3( Vec3::Arg start, Vec3::Arg end, u8 const* indices, void* block );
void WriteColourBlock4( Vec3::Arg start, Vec3::Arg end, u8 const* indices, void* block );

void DecompressColour( u8* rgba, void const* block, bool isDxt1 );
// -- GODOT start --
void DecompressColourBc4( u8* rgba, void const* block );
void DecompressColourBc5( u8* rgba, void const* block );
// -- GODOT end --

} // namespace squish

#endif // ndef SQUISH_COLOURBLOCK_H
