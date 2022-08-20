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
#pragma once
#include "inc/Main.h"


namespace graphite2 {

class CharInfo
{

public:
    CharInfo() : m_char(0), m_before(-1), m_after(-1), m_base(0), m_featureid(0), m_break(0), m_flags(0) {}
    void init(int cid) { m_char = cid; }
    unsigned int unicodeChar() const { return m_char; }
    void feats(int offset) { m_featureid = offset; }
    int fid() const { return m_featureid; }
    int breakWeight() const { return m_break; }
    void breakWeight(int val) { m_break = val; }
    int after() const { return m_after; }
    void after(int val) { m_after = val; }
    int before() const { return m_before; }
    void before(int val) { m_before = val; }
    size_t base() const { return m_base; }
    void base(size_t offset) { m_base = offset; }
    void addflags(uint8 val) { m_flags |= val; }
    uint8 flags() const { return m_flags; }

    CLASS_NEW_DELETE
private:
    int m_char;     // Unicode character from character stream
    int m_before;   // slot index before us, comes before
    int m_after;    // slot index after us, comes after
    size_t  m_base; // offset into input string corresponding to this charinfo
    uint8 m_featureid;  // index into features list in the segment
    int8 m_break;   // breakweight coming from lb table
    uint8 m_flags;  // 0,1 segment split.
};

} // namespace graphite2

struct gr_char_info : public graphite2::CharInfo {};
