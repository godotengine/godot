/*  GRAPHITE2 LICENSING

    Copyright 2012, SIL International
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

//#include "inc/FeatureMap.h"
//#include "inc/GlyphsCache.h"
//#include "inc/Silf.h"

#ifndef GRAPHITE2_NFILEFACE

#include <cstdio>
#include <cassert>

#include "graphite2/Font.h"

#include "inc/Main.h"
#include "inc/TtfTypes.h"
#include "inc/TtfUtil.h"

namespace graphite2 {


class FileFace
{
    static const void * get_table_fn(const void* appFaceHandle, unsigned int name, size_t *len);
    static void         rel_table_fn(const void* appFaceHandle, const void *table_buffer);

public:
    static const gr_face_ops ops;

    FileFace(const char *filename);
    ~FileFace();

    operator bool () const throw();
    CLASS_NEW_DELETE;

private:        //defensive
    FILE          * _file;
    size_t          _file_len;

    TtfUtil::Sfnt::OffsetSubTable         * _header_tbl;
    TtfUtil::Sfnt::OffsetSubTable::Entry  * _table_dir;

    FileFace(const FileFace&);
    FileFace& operator=(const FileFace&);
};

inline
FileFace::operator bool() const throw()
{
    return _file && _header_tbl && _table_dir;
}

} // namespace graphite2

#endif      //!GRAPHITE2_NFILEFACE
