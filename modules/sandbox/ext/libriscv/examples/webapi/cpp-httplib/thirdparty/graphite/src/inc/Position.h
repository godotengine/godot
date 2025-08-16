// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2010, SIL International, All rights reserved.

#pragma once

namespace graphite2 {

class Position
{
public:
    Position() : x(0), y(0) { }
    Position(const float inx, const float iny) : x(inx), y(iny) {}
    Position operator + (const Position& a) const { return Position(x + a.x, y + a.y); }
    Position operator - (const Position& a) const { return Position(x - a.x, y - a.y); }
    Position operator * (const float m) const { return Position(x * m, y * m); }
    Position &operator += (const Position &a) { x += a.x; y += a.y; return *this; }
    Position &operator *= (const float m) { x *= m; y *= m; return *this; }

    float x;
    float y;
};

class Rect
{
public :
    Rect() {}
    Rect(const Position& botLeft, const Position& topRight): bl(botLeft), tr(topRight) {}
    Rect widen(const Rect& other) { return Rect(Position(bl.x > other.bl.x ? other.bl.x : bl.x, bl.y > other.bl.y ? other.bl.y : bl.y), Position(tr.x > other.tr.x ? tr.x : other.tr.x, tr.y > other.tr.y ? tr.y : other.tr.y)); }
    Rect operator + (const Position &a) const { return Rect(Position(bl.x + a.x, bl.y + a.y), Position(tr.x + a.x, tr.y + a.y)); }
    Rect operator - (const Position &a) const { return Rect(Position(bl.x - a.x, bl.y - a.y), Position(tr.x - a.x, tr.y - a.y)); }
    Rect operator * (float m) const { return Rect(Position(bl.x, bl.y) * m, Position(tr.x, tr.y) * m); }
    float width() const { return tr.x - bl.x; }
    float height() const { return tr.y - bl.y; }

    bool hitTest(Rect &other);

    // returns Position(overlapx, overlapy) where overlap<0 if overlapping else positive)
    Position overlap(Position &offset, Rect &other, Position &otherOffset);
    //Position constrainedAvoid(Position &offset, Rect &box, Rect &sdbox, Position &other, Rect &obox, Rect &osdbox);

    Position bl;
    Position tr;
};

} // namespace graphite2
