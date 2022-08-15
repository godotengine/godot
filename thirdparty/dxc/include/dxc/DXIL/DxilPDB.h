///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilPDB.h                                                                 //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Helpers to wrap debug information in a PDB container.                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/Support/WinIncludes.h"
#include "llvm/ADT/ArrayRef.h"

struct IDxcBlob;
struct IStream;
struct IMalloc;

namespace hlsl {
namespace pdb {

  HRESULT LoadDataFromStream(IMalloc *pMalloc, IStream *pIStream, IDxcBlob **ppHash, IDxcBlob **ppContainer);
  HRESULT LoadDataFromStream(IMalloc *pMalloc, IStream *pIStream, IDxcBlob **pOutContainer);
  HRESULT WriteDxilPDB(IMalloc *pMalloc, IDxcBlob *pContainer, llvm::ArrayRef<BYTE> HashData, IDxcBlob **ppOutBlob);
  HRESULT WriteDxilPDB(IMalloc *pMalloc, llvm::ArrayRef<BYTE> ContainerData, llvm::ArrayRef<BYTE> HashData, IDxcBlob **ppOutBlob);
}
}
