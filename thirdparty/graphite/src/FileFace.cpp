// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2012, SIL International, All rights reserved.

#include <cstring>
#include "inc/FileFace.h"


#ifndef GRAPHITE2_NFILEFACE

using namespace graphite2;

FileFace::FileFace(const char *filename)
: _file(fopen(filename, "rb")),
  _file_len(0),
  _header_tbl(NULL),
  _table_dir(NULL)
{
    if (!_file) return;

    if (fseek(_file, 0, SEEK_END)) return;
    _file_len = ftell(_file);
    if (fseek(_file, 0, SEEK_SET)) return;

    size_t tbl_offset, tbl_len;

    // Get the header.
    if (!TtfUtil::GetHeaderInfo(tbl_offset, tbl_len)) return;
    if (fseek(_file, long(tbl_offset), SEEK_SET)) return;
    _header_tbl = (TtfUtil::Sfnt::OffsetSubTable*)gralloc<char>(tbl_len);
    if (_header_tbl)
    {
        if (fread(_header_tbl, 1, tbl_len, _file) != tbl_len) return;
        if (!TtfUtil::CheckHeader(_header_tbl)) return;
    }

    // Get the table directory
    if (!TtfUtil::GetTableDirInfo(_header_tbl, tbl_offset, tbl_len)) return;
    _table_dir = (TtfUtil::Sfnt::OffsetSubTable::Entry*)gralloc<char>(tbl_len);
    if (fseek(_file, long(tbl_offset), SEEK_SET)) return;
    if (_table_dir && fread(_table_dir, 1, tbl_len, _file) != tbl_len)
    {
        free(_table_dir);
        _table_dir = NULL;
    }
    return;
}

FileFace::~FileFace()
{
    free(_table_dir);
    free(_header_tbl);
    if (_file)
        fclose(_file);
}


const void *FileFace::get_table_fn(const void* appFaceHandle, unsigned int name, size_t *len)
{
    if (appFaceHandle == 0)     return 0;
    const FileFace & file_face = *static_cast<const FileFace *>(appFaceHandle);

    void *tbl;
    size_t tbl_offset, tbl_len;
    if (!TtfUtil::GetTableInfo(name, file_face._header_tbl, file_face._table_dir, tbl_offset, tbl_len))
        return 0;

    if (tbl_offset > file_face._file_len || tbl_len > file_face._file_len - tbl_offset
            || fseek(file_face._file, long(tbl_offset), SEEK_SET) != 0)
        return 0;

    tbl = malloc(tbl_len);
    if (!tbl || fread(tbl, 1, tbl_len, file_face._file) != tbl_len)
    {
        free(tbl);
        return 0;
    }

    if (len) *len = tbl_len;
    return tbl;
}

void FileFace::rel_table_fn(const void* appFaceHandle, const void *table_buffer)
{
    if (appFaceHandle == 0)     return;

    free(const_cast<void *>(table_buffer));
}

const gr_face_ops FileFace::ops = { sizeof FileFace::ops, &FileFace::get_table_fn, &FileFace::rel_table_fn };


#endif                  //!GRAPHITE2_NFILEFACE
