// EnumDirItems.h

#ifndef ZIP7_INC_ENUM_DIR_ITEMS_H
#define ZIP7_INC_ENUM_DIR_ITEMS_H

#include "../../../Common/Wildcard.h"

#include "DirItem.h"


HRESULT EnumerateItems(
    const NWildcard::CCensor &censor,
    NWildcard::ECensorPathMode pathMode,
    const UString &addPathPrefix,
    CDirItems &dirItems);


struct CMessagePathException: public UString
{
  CMessagePathException(const char *a, const wchar_t *u = NULL);
  CMessagePathException(const wchar_t *a, const wchar_t *u = NULL);
};


HRESULT EnumerateDirItemsAndSort(
    NWildcard::CCensor &censor,
    NWildcard::ECensorPathMode pathMode,
    const UString &addPathPrefix,
    UStringVector &sortedPaths,
    UStringVector &sortedFullPaths,
    CDirItemsStat &st,
    IDirItemsCallback *callback);

#ifdef _WIN32
void ConvertToLongNames(NWildcard::CCensor &censor);
#endif

#endif
