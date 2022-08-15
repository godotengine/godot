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

#pragma once

#include "llvm/ADT/STLExtras.h"
#include "dxc/Support/Global.h"
#include "dxc/Support/WinIncludes.h"
#include "dxc/DxilContainer/DxilContainer.h"

namespace hlsl {

#define DXIL_CONTAINER_BLOB_NOT_FOUND -1

  struct DxilContainerHeader;

  //=================================================================================================================================
  // DxilContainerReader
  //
  // Parse a DXIL or DXBC Container that you provide as input.
  //
  // Basic usage:
  // (1) Call Load()
  // (2) Call various Get*() commands to retrieve information about the container such as 
  //     how many blobs are in it, the hash of the container, the version #, and most importantly
  //     retrieve all of the Blobs.  You can retrieve blobs by searching for the FourCC, or
  //     enumerate through all of them.  Multiple blobs can even have the same FourCC, if you choose to 
  //     create the DXBC that way, and this parser will let you discover all of them.
  // (3) You can parse a new container by calling Load() again, or just get rid of the class.
  // 
  class DxilContainerReader
  {
  public:
    DxilContainerReader() {}

    // Sets the container to be parsed, and does some
    // basic integrity checking, making sure the blob FourCCs 
    // are all from the known list, and ensuring the version is:
    //     Major = DXBC_MAJOR_VERSION
    //     Minor = DXBC_MAJOR_VERSION
    // 
    // Returns S_OK or E_FAIL
    HRESULT Load(_In_ const void* pContainer, _In_ uint32_t containerSizeInBytes);

    HRESULT GetVersion(_Out_ DxilContainerVersion *pResult);
    HRESULT GetPartCount(_Out_ uint32_t *pResult);
    HRESULT GetPartContent(uint32_t idx, _Outptr_ const void **ppResult, _Out_ uint32_t *pResultSize = nullptr);
    HRESULT GetPartFourCC(uint32_t idx, _Out_ uint32_t *pResult);
    HRESULT FindFirstPartKind(uint32_t kind, _Out_ uint32_t *pResult);

  private:
    const void* m_pContainer = nullptr;
    uint32_t m_uContainerSize = 0;
    const DxilContainerHeader *m_pHeader = nullptr;

    bool IsLoaded() const { return m_pHeader != nullptr; }
  };
  
} // namespace hlsl