                  FreeType font driver for PCF fonts

                       Francesco Zappa Nardelli
                  <francesco.zappa.nardelli@ens.fr>


Introduction
************

PCF (Portable Compiled Format) is a binary bitmap font format, largely used
in X world. This code implements a PCF driver for the FreeType library.
Glyph images are loaded into memory only on demand, thus leading to a small
memory footprint.

Information on the PCF font format can only be worked out from
`pcfread.c', and `pcfwrite.c', to be found, for instance, in the XFree86
(www.xfree86.org) source tree (xc/lib/font/bitmap/).

Many good bitmap fonts in bdf format come with XFree86: they can be
compiled into the pcf format using the `bdftopcf' utility.


Supported hardware
******************

The driver has been tested on linux/x86 and sunos5.5/sparc.  In both
cases the compiler was gcc.  When back in Paris, I will test it also
on linux/alpha.


Encodings
*********

Use `FT_Get_BDF_Charset_ID' to access the encoding and registry.

The driver always exports `ft_encoding_none' as face->charmap.encoding.
FT_Get_Char_Index() behavior is unmodified, that is, it converts the ULong
value given as argument into the corresponding glyph number.


Known problems
**************

- dealing explicitly with encodings breaks the uniformity of FreeType 2
  API.

- except for encodings properties, client applications have no
  visibility of the PCF_Face object.  This means that applications
  cannot directly access font tables and are obliged to trust
  FreeType.

- currently, glyph names and ink_metrics are ignored.

I plan to give full visibility of the PCF_Face object in the next
release of the driver, thus implementing also glyph names and
ink_metrics.

- height is defined as (ascent - descent).  Is this correct?

- if unable to read size information from the font, PCF_Init_Face
  sets available_size->width and available_size->height to 12.

- too many english grammar errors in the readme file :-(


License
*******

Copyright (C) 2000 by Francesco Zappa Nardelli

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


Credits
*******

Keith Packard wrote the pcf driver found in XFree86.  His work is at
the same time the specification and the sample implementation of the
PCF format.  Undoubtedly, this driver is inspired from his work.
