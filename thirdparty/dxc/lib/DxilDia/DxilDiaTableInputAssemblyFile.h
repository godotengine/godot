///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilDiaTableInputAssemblyFile.h                                           //
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

class InputAssemblyFilesTable
  : public impl::TableBase<IDiaEnumInputAssemblyFiles, IDiaInputAssemblyFile> {
public:
  InputAssemblyFilesTable(IMalloc *pMalloc, Session *pSession);
  HRESULT GetItem(DWORD index, IDiaInputAssemblyFile **ppItem) override;
};
}  // namespace dxil_dia
