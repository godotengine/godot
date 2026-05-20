// ArchiveName.h

#ifndef ZIP7_INC_ARCHIVE_NAME_H
#define ZIP7_INC_ARCHIVE_NAME_H

#include "../../../Windows/FileFind.h"

/* (fi != NULL) only if (paths.Size() == 1) */

UString CreateArchiveName(
    const UStringVector &paths,
    bool isHash,
    const NWindows::NFile::NFind::CFileInfo *fi,
    UString &baseName);

#endif
