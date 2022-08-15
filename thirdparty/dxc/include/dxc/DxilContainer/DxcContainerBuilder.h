///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxcContainerBuilder.h                                                     //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Implements the Dxil Container Builder                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "dxc/dxcapi.h"
#include "dxc/Support/Global.h"
#include "dxc/Support/WinIncludes.h"
#include "dxc/DxilContainer/DxilContainer.h"
#include "dxc/Support/microcom.h"
#include "llvm/ADT/SmallVector.h"

using namespace hlsl;
namespace hlsl {
  class AbstractMemoryStream;
}

class DxcContainerBuilder : public IDxcContainerBuilder {
public:
  HRESULT STDMETHODCALLTYPE Load(_In_ IDxcBlob *pDxilContainerHeader) override; // Loads DxilContainer to the builder
  HRESULT STDMETHODCALLTYPE AddPart(_In_ UINT32 fourCC, _In_ IDxcBlob *pSource) override; // Add the given part with fourCC
  HRESULT STDMETHODCALLTYPE RemovePart(_In_ UINT32 fourCC) override;                // Remove the part with fourCC
  HRESULT STDMETHODCALLTYPE SerializeContainer(_Out_ IDxcOperationResult **ppResult) override; // Builds a container of the given container builder state

  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL()
  DXC_MICROCOM_TM_CTOR(DxcContainerBuilder)
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void **ppvObject) override {
    return DoBasicQueryInterface<IDxcContainerBuilder>(this, riid, ppvObject);
  }

  void Init(const char *warning = nullptr) {
    m_warning = warning;
    m_RequireValidation = false;
    m_HasPrivateData = false;
  }

protected:
  DXC_MICROCOM_TM_REF_FIELDS()

private:
  class DxilPart {
  public:
    UINT32 m_fourCC;
    CComPtr<IDxcBlob> m_Blob;
    DxilPart(UINT32 fourCC, IDxcBlob *pSource) : m_fourCC(fourCC), m_Blob(pSource) {}
  };
  typedef llvm::SmallVector<DxilPart, 8> PartList;

  PartList m_parts;
  CComPtr<IDxcBlob> m_pContainer; 
  const char *m_warning;
  bool m_RequireValidation;
  bool m_HasPrivateData;

  UINT32 ComputeContainerSize();
  HRESULT UpdateContainerHeader(AbstractMemoryStream *pStream, uint32_t containerSize);
  HRESULT UpdateOffsetTable(AbstractMemoryStream *pStream);
  HRESULT UpdateParts(AbstractMemoryStream *pStream);
  void AddPart(DxilPart&& part);
};