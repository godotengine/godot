// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2010, SIL International, All rights reserved.

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
