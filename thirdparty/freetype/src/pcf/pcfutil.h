/*  pcfutil.h

    FreeType font driver for pcf fonts

  Copyright 2000, 2001, 2004 by
  Francesco Zappa Nardelli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/


#ifndef PCFUTIL_H_
#define PCFUTIL_H_


#include <ft2build.h>
#include FT_CONFIG_CONFIG_H
#include <freetype/internal/compiler-macros.h>

FT_BEGIN_HEADER

  FT_LOCAL( void )
  BitOrderInvert( unsigned char*  buf,
                  size_t          nbytes );

  FT_LOCAL( void )
  TwoByteSwap( unsigned char*  buf,
               size_t          nbytes );

  FT_LOCAL( void )
  FourByteSwap( unsigned char*  buf,
                size_t          nbytes );

FT_END_HEADER

#endif /* PCFUTIL_H_ */


/* END */
