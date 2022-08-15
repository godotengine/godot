///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilContainerReader.h                                                     //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Helper class for reading from dxil container.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/Support/Global.h"
#include "dxc/Support/WinAdapter.h"
#include "dxc/DxilContainer/DxilContainer.h"
#include "dxc/DxilContainer/DxilContainerReader.h"

namespace hlsl {

HRESULT DxilContainerReader::Load(_In_ const void* pContainer, _In_ uint32_t containerSizeInBytes) {
  if (pContainer == nullptr) {
    return E_FAIL;
  }

  const DxilContainerHeader *pHeader = IsDxilContainerLike(pContainer, containerSizeInBytes);
  if (pHeader == nullptr) {
    return E_FAIL;
  }
  if (!IsValidDxilContainer(pHeader, containerSizeInBytes)) {
    return E_FAIL;
  }

  m_pContainer = pContainer;
  m_uContainerSize = containerSizeInBytes;
  m_pHeader = pHeader;
  
  return S_OK;
}

HRESULT DxilContainerReader::GetVersion(_Out_ DxilContainerVersion *pResult) {
  if (pResult == nullptr) return E_POINTER;
  if (!IsLoaded()) return E_NOT_VALID_STATE;
  *pResult = m_pHeader->Version;
  return S_OK;
}

HRESULT DxilContainerReader::GetPartCount(_Out_ uint32_t *pResult) {
  if (pResult == nullptr) return E_POINTER;
  if (!IsLoaded()) return E_NOT_VALID_STATE;
  *pResult = m_pHeader->PartCount;
  return S_OK;
}

HRESULT DxilContainerReader::GetPartContent(uint32_t idx, _Outptr_ const void **ppResult, _Out_ uint32_t *pResultSize) {
  if (ppResult == nullptr) return E_POINTER;
  *ppResult = nullptr;
  if (!IsLoaded()) return E_NOT_VALID_STATE;
  if (idx >= m_pHeader->PartCount) return E_BOUNDS;
  const DxilPartHeader *pPart = GetDxilContainerPart(m_pHeader, idx);
  *ppResult = GetDxilPartData(pPart);
  if (pResultSize != nullptr) {
    *pResultSize = pPart->PartSize;
  }
  return S_OK;
}

HRESULT DxilContainerReader::GetPartFourCC(uint32_t idx, _Out_ uint32_t *pResult) {
  if (pResult == nullptr) return E_POINTER;
  if (!IsLoaded()) return E_NOT_VALID_STATE;
  if (idx >= m_pHeader->PartCount) return E_BOUNDS;
  const DxilPartHeader *pPart = GetDxilContainerPart(m_pHeader, idx);
  *pResult = pPart->PartFourCC;
  return S_OK;
}

HRESULT DxilContainerReader::FindFirstPartKind(uint32_t kind, _Out_ uint32_t *pResult) {
  if (pResult == nullptr) return E_POINTER;
  *pResult = 0;
  if (!IsLoaded()) return E_NOT_VALID_STATE;
  DxilPartIterator it = std::find_if(begin(m_pHeader), end(m_pHeader), DxilPartIsType(kind));
  *pResult = (it == end(m_pHeader)) ? DXIL_CONTAINER_BLOB_NOT_FOUND : it.index;
  return S_OK;
}
  
} // namespace hlsl
