// Common/ListFileUtils.h

#ifndef ZIP7_INC_COMMON_LIST_FILE_UTILS_H
#define ZIP7_INC_COMMON_LIST_FILE_UTILS_H

#include "MyString.h"
#include "MyTypes.h"

#define Z7_WIN_CP_UTF16   1200
#define Z7_WIN_CP_UTF16BE 1201

// bool ReadNamesFromListFile(CFSTR fileName, UStringVector &strings, UINT codePage = CP_OEMCP);

 // = CP_OEMCP
bool ReadNamesFromListFile2(CFSTR fileName, UStringVector &strings, UINT codePage,
    DWORD &lastError);

#endif
