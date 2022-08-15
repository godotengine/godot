///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilDiaTableSegmentMap.h                                                  //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// DIA API implementation for DXIL modules.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "dxc/Support/WinIncludes.h"

#include "dia2.h"

#include "dxc/Support/Global.h"
#include "dxc/Support/microcom.h"

#include "DxilDiaTable.h"

namespace dxil_dia {
class Session;

class SegmentMapTable : public impl::TableBase<IDiaEnumSegments, IDiaSegment> {
public:
  SegmentMapTable(IMalloc *pMalloc, Session *pSession);

  HRESULT GetItem(DWORD index, IDiaSegment **ppItem) override;
};

}  // namespace dxil_dia
