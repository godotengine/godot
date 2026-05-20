// List.h

#ifndef ZIP7_INC_LIST_H
#define ZIP7_INC_LIST_H

#include "../../../Common/Wildcard.h"

#include "../Common/LoadCodecs.h"

struct CListOptions
{
  bool ExcludeDirItems;
  bool ExcludeFileItems;
  bool DisablePercents;

  CListOptions():
    ExcludeDirItems(false),
    ExcludeFileItems(false),
    DisablePercents(false)
    {}
};

HRESULT ListArchives(
    const CListOptions &listOptions,
    CCodecs *codecs,
    const CObjectVector<COpenType> &types,
    const CIntVector &excludedFormats,
    bool stdInMode,
    UStringVector &archivePaths, UStringVector &archivePathsFull,
    bool processAltStreams, bool showAltStreams,
    const NWildcard::CCensorNode &wildcardCensor,
    bool enableHeaders, bool techMode,
  #ifndef Z7_NO_CRYPTO
    bool &passwordEnabled, UString &password,
  #endif
  #ifndef Z7_SFX
    const CObjectVector<CProperty> *props,
  #endif
    UInt64 &errors,
    UInt64 &numWarnings);

#endif
