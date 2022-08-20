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
#include "graphite2/Font.h"
#include "inc/Font.h"


using namespace graphite2;

extern "C" {

void gr_engine_version(int *nMajor, int *nMinor, int *nBugFix)
{
    if (nMajor) *nMajor = GR2_VERSION_MAJOR;
    if (nMinor) *nMinor = GR2_VERSION_MINOR;
    if (nBugFix) *nBugFix = GR2_VERSION_BUGFIX;
}

gr_font* gr_make_font(float ppm/*pixels per em*/, const gr_face *face)
{
    return gr_make_font_with_advance_fn(ppm, 0, 0, face);
}


gr_font* gr_make_font_with_ops(float ppm/*pixels per em*/, const void* appFontHandle/*non-NULL*/, const gr_font_ops * font_ops, const gr_face * face/*needed for scaling*/)
{                 //the appFontHandle must stay alive all the time when the gr_font is alive. When finished with the gr_font, call destroy_gr_font
    if (face == 0 || ppm <= 0)  return 0;

    Font * const res = new Font(ppm, *face, appFontHandle, font_ops);
    if (*res)
        return static_cast<gr_font*>(res);
    else
    {
        delete res;
        return 0;
    }
}

gr_font* gr_make_font_with_advance_fn(float ppm/*pixels per em*/, const void* appFontHandle/*non-NULL*/, gr_advance_fn getAdvance, const gr_face * face/*needed for scaling*/)
{
    const gr_font_ops ops = {sizeof(gr_font_ops), getAdvance, NULL};
    return gr_make_font_with_ops(ppm, appFontHandle, &ops, face);
}

void gr_font_destroy(gr_font *font)
{
    delete static_cast<Font*>(font);
}


} // extern "C"
