// UpdatePair.h

#ifndef ZIP7_INC_UPDATE_PAIR_H
#define ZIP7_INC_UPDATE_PAIR_H

#include "DirItem.h"
#include "UpdateAction.h"

#include "../../Archive/IArchive.h"

struct CUpdatePair
{
  NUpdateArchive::NPairState::EEnum State;
  int ArcIndex;
  int DirIndex;
  int HostIndex; // >= 0 for alt streams only, contains index of host pair

  CUpdatePair(): ArcIndex(-1), DirIndex(-1), HostIndex(-1) {}
};

void GetUpdatePairInfoList(
    const CDirItems &dirItems,
    const CObjectVector<CArcItem> &arcItems,
    NFileTimeType::EEnum fileTimeType,
    CRecordVector<CUpdatePair> &updatePairs);

#endif
