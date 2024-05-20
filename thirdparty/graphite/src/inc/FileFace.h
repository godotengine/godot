// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2012, SIL International, All rights reserved.

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
