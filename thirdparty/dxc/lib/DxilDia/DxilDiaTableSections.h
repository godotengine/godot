///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilDiaTableSections.h                                                    //
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

#include "DxilDia.h"
#include "DxilDiaTable.h"

namespace dxil_dia {
class Session;

class SectionsTable : public impl::TableBase<IDiaEnumSectionContribs, IDiaSectionContrib> {
public:
  SectionsTable(IMalloc *pMalloc, Session *pSession);
  HRESULT GetItem(DWORD index, IDiaSectionContrib **ppItem) override;
};

}  // namespace dxil_dia
