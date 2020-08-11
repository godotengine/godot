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
#include "inc/Position.h"
#include <cmath>

using namespace graphite2;

bool Rect::hitTest(Rect &other)
{
    if (bl.x > other.tr.x) return false;
    if (tr.x < other.bl.x) return false;
    if (bl.y > other.tr.y) return false;
    if (tr.y < other.bl.y) return false;
    return true;
}

Position Rect::overlap(Position &offset, Rect &other, Position &othero)
{
    float ax = (bl.x + offset.x) - (other.tr.x + othero.x);
    float ay = (bl.y + offset.y) - (other.tr.y + othero.y);
    float bx = (other.bl.x + othero.x) - (tr.x + offset.x);
    float by = (other.bl.y + othero.y) - (tr.y + offset.y);
    return Position((ax > bx ? ax : bx), (ay > by ? ay : by));
}

float boundmin(float move, float lim1, float lim2, float &error)
{
    // error is always positive for easy comparison
    if (move < lim1 && move < lim2)
    { error = 0.; return move; }
    else if (lim1 < lim2)
    { error = std::fabs(move - lim1); return lim1; }
    else
    { error = std::fabs(move - lim2); return lim2; }
}

#if 0
Position Rect::constrainedAvoid(Position &offset, Rect &box, Rect &sdbox, Position &other, Rect &obox, Rect &osdbox)
{
    // a = max, i = min, s = sum, d = diff
    float eax, eay, eix, eiy, eas, eis, ead, eid;
    float beste = INF;
    Position res;
    // calculate the movements in each direction and the error (amount of remaining overlap)
    // first param is movement, second and third are movement over the constraining box
    float ax = boundmin(obox.tr.x + other.x - box.bl.x - offset.x + 1, tr.x - offset.x, INF, &eax);
    float ay = boundmin(obox.tr.y + other.y - box.bl.y - offset.y + 1, tr.y - offset.y, INF, &eay);
    float ix = boundmin(obox.bl.x + other.x - box.tr.x - offset.x + 1, bl.x - offset.x, INF, &eix);
    float iy = boundmin(obox.bl.y + other.y - box.tr.y - offset.y + 1, bl.y - offset.y, INF, &eiy);
    float as = boundmin(ISQRT2 * (osdbox.tr.x + other.x + other.y - sdbox.bl.x - offset.x - offset.y) + 1, tr.x - offset.x, tr.y - offset.y, &eas);
    float is = boundmin(ISQRT2 * (osdbox.bl.x + other.x + other.y - sdbox.tr.x - offset.x - offset.y) + 1, bl.x - offset.x, bl.y - offset.y, &eis);
    float ad = boundmin(ISQRT2 * (osdbox.tr.y + other.x - other.y - sdbox.bl.y - offset.x + offset.y) + 1, tr.y - offset.y, tr.x - offset.x, &ead);
    float id = boundmin(ISQRT2 * (osdbox.bl.y + other.x - other.y - sdbox.tr.y - offset.x + offset.y) + 1, bl.y - offset.y, bl.x - offset.x, &eid);

    if (eax < beste)
    { res = Position(ax, 0); beste = eax; }
    if (eay < beste)
    { res = Position(0, ay); beste = eay; }
    if (eix < beste)
    { res = Position(ix, 0); beste = eix; }
    if (eiy < beste)
    { res = Position(0, iy); beste = eiy; }
    if (SQRT2 * (eas) < beste)
    { res = Position(as, ad); beste = SQRT2 * (eas); }
    if (SQRT2 * (eis) < beste)
    { res = Position(is, is); beste = SQRT2 * (eis); }
    if (SQRT2 * (ead) < beste)
    { res = Position(ad, ad); beste = SQRT2 * (ead); }
    if (SQRT2 * (eid) < beste)
    { res = Position(id, id); beste = SQRT2 * (eid); }
    return res;
}
#endif
