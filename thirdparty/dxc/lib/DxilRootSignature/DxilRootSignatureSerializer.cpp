///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilRootSignatureSerializer.cpp                                           //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Serializer for root signature structures.                                 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DXIL/DxilConstants.h"
#include "dxc/DxilRootSignature/DxilRootSignature.h"
#include "dxc/Support/Global.h"
#include "dxc/Support/WinIncludes.h"
#include "dxc/Support/WinFunctions.h"
#include "dxc/Support/FileIOHelper.h"
#include "dxc/dxcapi.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/DiagnosticPrinter.h"

#include <string>
#include <algorithm>
#include <utility>
#include <vector>
#include <set>

#include "DxilRootSignatureHelper.h"

using namespace llvm;
using std::string;

namespace hlsl {

using namespace root_sig_helper;

//////////////////////////////////////////////////////////////////////////////
// Simple serializer.

class SimpleSerializer {
  struct Segment {
    void *pData;
    unsigned cbSize;
    bool bOwner;
    unsigned Offset;
    Segment *pNext;
  };

public:
  SimpleSerializer();
  ~SimpleSerializer();

  HRESULT AddBlock(void *pData, unsigned cbSize, unsigned *pOffset);
  HRESULT ReserveBlock(void **ppData, unsigned cbSize, unsigned *pOffset);

  HRESULT Compact(_Out_writes_bytes_(cbSize) char *pData, unsigned cbSize);
  unsigned GetSize();

protected:
  unsigned m_cbSegments;
  Segment *m_pSegment;
  Segment **m_ppSegment;
};

SimpleSerializer::SimpleSerializer() {
  m_cbSegments = 0;
  m_pSegment = nullptr;
  m_ppSegment = &m_pSegment;
}

SimpleSerializer::~SimpleSerializer() {
  while (m_pSegment) {
    Segment *pSegment = m_pSegment;
    m_pSegment = pSegment->pNext;

    if (pSegment->bOwner) {
      delete[] (char*)pSegment->pData;
    }

    delete pSegment;
  }
}

HRESULT SimpleSerializer::AddBlock(void *pData, unsigned cbSize,
                                   unsigned *pOffset) {
  Segment *pSegment = nullptr;
  IFRBOOL(!(cbSize != 0 && pData == nullptr), E_FAIL);

  IFROOM(pSegment = new (std::nothrow) Segment);
  pSegment->pData = pData;

  m_cbSegments = (m_cbSegments + 3) & ~3;
  pSegment->Offset = m_cbSegments;
  pSegment->cbSize = cbSize;
  pSegment->bOwner = false;
  pSegment->pNext = nullptr;

  m_cbSegments += pSegment->cbSize;
  *m_ppSegment = pSegment;
  m_ppSegment = &pSegment->pNext;

  if (pOffset != nullptr) {
    *pOffset = pSegment->Offset;
  }

  return S_OK;
}

HRESULT SimpleSerializer::ReserveBlock(void **ppData, unsigned cbSize,
                                       unsigned *pOffset) {
  HRESULT hr = S_OK;
  Segment *pSegment = nullptr;
  void *pClonedData = nullptr;

  IFCOOM(pSegment = new (std::nothrow) Segment);
  pSegment->pData = nullptr;

  IFCOOM(pClonedData = new (std::nothrow) char[cbSize]);
  pSegment->pData = pClonedData;

  m_cbSegments = (m_cbSegments + 3) & ~3;
  pSegment->Offset = m_cbSegments;
  pSegment->cbSize = cbSize;
  pSegment->bOwner = true;
  pSegment->pNext = nullptr;

  m_cbSegments += pSegment->cbSize;
  *m_ppSegment = pSegment;
  m_ppSegment = &pSegment->pNext;

  *ppData = pClonedData;
  if (pOffset) {
    *pOffset = pSegment->Offset;
  }

Cleanup:
  if (FAILED(hr)) {
    delete[] (char*)pClonedData;
    delete pSegment;
  }
  return hr;
}

HRESULT SimpleSerializer::Compact(_Out_writes_bytes_(cbSize) char *pData,
                                  unsigned cbSize) {
  unsigned cb = GetSize();
  IFRBOOL(cb <= cbSize, E_FAIL);
  DXASSERT_NOMSG(cb <= UINT32_MAX / 2);

  char *p = (char *)pData;
  cb = 0;

  for (Segment *pSegment = m_pSegment; pSegment; pSegment = pSegment->pNext) {
    unsigned cbAlign = ((cb + 3) & ~3) - cb;

    _Analysis_assume_(p + cbAlign <= pData + cbSize);
    memset(p, 0xab, cbAlign);

    p += cbAlign;
    cb += cbAlign;

    _Analysis_assume_(p + pSegment->cbSize <= pData + cbSize);
    memcpy(p, pSegment->pData, pSegment->cbSize);

    p += pSegment->cbSize;
    cb += pSegment->cbSize;
  }

  // Trailing zeros
  _Analysis_assume_(p + cbSize - cb <= pData + cbSize);
  memset(p, 0xab, cbSize - cb);

  return S_OK;
}

unsigned SimpleSerializer::GetSize() {
  // Round up to 4==sizeof(unsigned).
  return ((m_cbSegments + 3) >> 2) * 4;
}

template<typename T_ROOT_SIGNATURE_DESC,
  typename T_ROOT_PARAMETER,
  typename T_ROOT_DESCRIPTOR_INTERNAL,
  typename T_DESCRIPTOR_RANGE_INTERNAL>
void SerializeRootSignatureTemplate(_In_ const T_ROOT_SIGNATURE_DESC* pRootSignature,
                                    DxilRootSignatureVersion DescVersion,
                                    _COM_Outptr_ IDxcBlob** ppBlob,
                                    DiagnosticPrinter &DiagPrinter,
                                    _In_ bool bAllowReservedRegisterSpace) {
  DxilContainerRootSignatureDesc RS;
  uint32_t Offset;
  SimpleSerializer Serializer;
  IFT(Serializer.AddBlock(&RS, sizeof(RS), &Offset));
  IFTBOOL(Offset == 0, E_FAIL);

  const T_ROOT_SIGNATURE_DESC *pRS = pRootSignature;
  RS.Version = (uint32_t)DescVersion;
  RS.Flags = (uint32_t)pRS->Flags;
  RS.NumParameters = pRS->NumParameters;
  RS.NumStaticSamplers = pRS->NumStaticSamplers;

  DxilContainerRootParameter *pRP;
  IFT(Serializer.ReserveBlock((void**)&pRP,
    sizeof(DxilContainerRootParameter)*RS.NumParameters, &RS.RootParametersOffset));

  for (uint32_t iRP = 0; iRP < RS.NumParameters; iRP++) {
    const T_ROOT_PARAMETER *pInRP = &pRS->pParameters[iRP];
    DxilContainerRootParameter *pOutRP = &pRP[iRP];
    pOutRP->ParameterType = (uint32_t)pInRP->ParameterType;
    pOutRP->ShaderVisibility = (uint32_t)pInRP->ShaderVisibility;
    switch (pInRP->ParameterType) {
    case DxilRootParameterType::DescriptorTable: {
      DxilContainerRootDescriptorTable *p1;
      IFT(Serializer.ReserveBlock((void**)&p1,
                                  sizeof(DxilContainerRootDescriptorTable),
                                  &pOutRP->PayloadOffset));
      p1->NumDescriptorRanges = pInRP->DescriptorTable.NumDescriptorRanges;

      T_DESCRIPTOR_RANGE_INTERNAL *p2;
      IFT(Serializer.ReserveBlock((void**)&p2,
                                  sizeof(T_DESCRIPTOR_RANGE_INTERNAL)*p1->NumDescriptorRanges,
                                  &p1->DescriptorRangesOffset));

      for (uint32_t i = 0; i < p1->NumDescriptorRanges; i++) {
        p2[i].RangeType = (uint32_t)pInRP->DescriptorTable.pDescriptorRanges[i].RangeType;
        p2[i].NumDescriptors = pInRP->DescriptorTable.pDescriptorRanges[i].NumDescriptors;
        p2[i].BaseShaderRegister = pInRP->DescriptorTable.pDescriptorRanges[i].BaseShaderRegister;
        p2[i].RegisterSpace = pInRP->DescriptorTable.pDescriptorRanges[i].RegisterSpace;
        p2[i].OffsetInDescriptorsFromTableStart = pInRP->DescriptorTable.pDescriptorRanges[i].OffsetInDescriptorsFromTableStart;
        DxilDescriptorRangeFlags Flags = GetFlags(pInRP->DescriptorTable.pDescriptorRanges[i]);
        SetFlags(p2[i], Flags);
      }
      break;
    }
    case DxilRootParameterType::Constants32Bit: {
      DxilRootConstants *p;
      IFT(Serializer.ReserveBlock((void**)&p, sizeof(DxilRootConstants), &pOutRP->PayloadOffset));
      p->Num32BitValues = pInRP->Constants.Num32BitValues;
      p->ShaderRegister = pInRP->Constants.ShaderRegister;
      p->RegisterSpace = pInRP->Constants.RegisterSpace;
      break;
    }
    case DxilRootParameterType::CBV:
    case DxilRootParameterType::SRV:
    case DxilRootParameterType::UAV: {
      T_ROOT_DESCRIPTOR_INTERNAL *p;
      IFT(Serializer.ReserveBlock((void**)&p, sizeof(T_ROOT_DESCRIPTOR_INTERNAL), &pOutRP->PayloadOffset));
      p->ShaderRegister = pInRP->Descriptor.ShaderRegister;
      p->RegisterSpace = pInRP->Descriptor.RegisterSpace;
      DxilRootDescriptorFlags Flags = GetFlags(pInRP->Descriptor);
      SetFlags(*p, Flags);
      break;
    }
    default:
      EAT(DiagPrinter << "D3DSerializeRootSignature: unknown root parameter type ("
                      << (uint32_t)pInRP->ParameterType << ")\n");
    }
  }

  DxilStaticSamplerDesc *pSS;
  unsigned StaticSamplerSize = sizeof(DxilStaticSamplerDesc)*RS.NumStaticSamplers;
  IFT(Serializer.ReserveBlock((void**)&pSS, StaticSamplerSize, &RS.StaticSamplersOffset));
  memcpy(pSS, pRS->pStaticSamplers, StaticSamplerSize);

  // Create the result blob.
  CDxcMallocHeapPtr<char> bytes(DxcGetThreadMallocNoRef());
  CComPtr<IDxcBlob> pBlob;
  unsigned cb = Serializer.GetSize();
  DXASSERT_NOMSG((cb & 0x3) == 0);
  IFTBOOL(bytes.Allocate(cb), E_OUTOFMEMORY);
  IFT(Serializer.Compact(bytes.m_pData, cb));
  IFT(DxcCreateBlobOnMalloc(bytes.m_pData, bytes.GetMallocNoRef(), cb, ppBlob));
  bytes.Detach(); // Ownership transfered to ppBlob.
}

_Use_decl_annotations_
void SerializeRootSignature(const DxilVersionedRootSignatureDesc *pRootSignature,
                            IDxcBlob **ppBlob, IDxcBlobEncoding **ppErrorBlob,
                            bool bAllowReservedRegisterSpace) {
  DXASSERT_NOMSG(pRootSignature != nullptr);
  DXASSERT_NOMSG(ppBlob != nullptr);
  DXASSERT_NOMSG(ppErrorBlob != nullptr);

  *ppBlob = nullptr;
  *ppErrorBlob = nullptr;

  // TODO: change SerializeRootSignature to take raw_ostream&
  string DiagString;
  raw_string_ostream DiagStream(DiagString);
  DiagnosticPrinterRawOStream DiagPrinter(DiagStream);

  // Verify root signature.
  if (!VerifyRootSignature(pRootSignature, DiagStream,
                           bAllowReservedRegisterSpace)) {
    DiagStream.flush();
    DxcCreateBlobWithEncodingOnHeapCopy(DiagString.c_str(), DiagString.size(), CP_UTF8, ppErrorBlob);
    return;
  }

  try {
    switch (pRootSignature->Version)
    {
    case DxilRootSignatureVersion::Version_1_0:
      SerializeRootSignatureTemplate<
        DxilRootSignatureDesc,
        DxilRootParameter,
        DxilRootDescriptor,
        DxilContainerDescriptorRange>(&pRootSignature->Desc_1_0,
          DxilRootSignatureVersion::Version_1_0,
          ppBlob, DiagPrinter,
          bAllowReservedRegisterSpace);
      break;

    case DxilRootSignatureVersion::Version_1_1:
    default:
      DXASSERT(pRootSignature->Version == DxilRootSignatureVersion::Version_1_1, "else VerifyRootSignature didn't validate");
      SerializeRootSignatureTemplate<
        DxilRootSignatureDesc1,
        DxilRootParameter1,
        DxilContainerRootDescriptor1,
        DxilContainerDescriptorRange1>(&pRootSignature->Desc_1_1,
          DxilRootSignatureVersion::Version_1_1,
          ppBlob, DiagPrinter,
          bAllowReservedRegisterSpace);
      break;
    }
  } catch (...) {
    DiagStream.flush();
    DxcCreateBlobWithEncodingOnHeapCopy(DiagString.c_str(), DiagString.size(), CP_UTF8, ppErrorBlob);
  }
}

//=============================================================================
//
// CVersionedRootSignatureDeserializer.
//
//=============================================================================
class CVersionedRootSignatureDeserializer {
protected:
  const DxilVersionedRootSignatureDesc *m_pRootSignature;
  const DxilVersionedRootSignatureDesc *m_pRootSignature10;
  const DxilVersionedRootSignatureDesc *m_pRootSignature11;

public:
  CVersionedRootSignatureDeserializer();
  ~CVersionedRootSignatureDeserializer();

  void Initialize(_In_reads_bytes_(SrcDataSizeInBytes) const void *pSrcData,
                  _In_ uint32_t SrcDataSizeInBytes);

  const DxilVersionedRootSignatureDesc *GetRootSignatureDescAtVersion(DxilRootSignatureVersion convertToVersion);

  const DxilVersionedRootSignatureDesc *GetUnconvertedRootSignatureDesc();
};

CVersionedRootSignatureDeserializer::CVersionedRootSignatureDeserializer()
  : m_pRootSignature(nullptr)
  , m_pRootSignature10(nullptr)
  , m_pRootSignature11(nullptr) {
}

CVersionedRootSignatureDeserializer::~CVersionedRootSignatureDeserializer() {
  DeleteRootSignature(m_pRootSignature10);
  DeleteRootSignature(m_pRootSignature11);
}

void CVersionedRootSignatureDeserializer::Initialize(_In_reads_bytes_(SrcDataSizeInBytes) const void *pSrcData,
                                                     _In_ uint32_t SrcDataSizeInBytes) {
  const DxilVersionedRootSignatureDesc *pRootSignature = nullptr;
  DeserializeRootSignature(pSrcData, SrcDataSizeInBytes, &pRootSignature);

  switch (pRootSignature->Version) {
  case DxilRootSignatureVersion::Version_1_0:
    m_pRootSignature10 = pRootSignature;
    break;

  case DxilRootSignatureVersion::Version_1_1:
    m_pRootSignature11 = pRootSignature;
    break;

  default:
    DeleteRootSignature(pRootSignature);
    return;
  }

  m_pRootSignature = pRootSignature;
}

const DxilVersionedRootSignatureDesc *
CVersionedRootSignatureDeserializer::GetUnconvertedRootSignatureDesc() {
  return m_pRootSignature;
}

const DxilVersionedRootSignatureDesc *
CVersionedRootSignatureDeserializer::GetRootSignatureDescAtVersion(DxilRootSignatureVersion ConvertToVersion) {
  switch (ConvertToVersion) {
  case DxilRootSignatureVersion::Version_1_0:
    if (m_pRootSignature10 == nullptr) {
      ConvertRootSignature(m_pRootSignature,
                           ConvertToVersion,
                           (const DxilVersionedRootSignatureDesc **)&m_pRootSignature10);
    }
    return m_pRootSignature10;

  case DxilRootSignatureVersion::Version_1_1:
    if (m_pRootSignature11 == nullptr) {
      ConvertRootSignature(m_pRootSignature,
                           ConvertToVersion,
                           (const DxilVersionedRootSignatureDesc **)&m_pRootSignature11);
    }
    return m_pRootSignature11;

  default:
    IFTBOOL(false, E_FAIL);
  }

  return nullptr;
}

template<typename T_ROOT_SIGNATURE_DESC,
         typename T_ROOT_PARAMETER,
         typename T_ROOT_DESCRIPTOR,
         typename T_ROOT_DESCRIPTOR_INTERNAL,
         typename T_DESCRIPTOR_RANGE,
         typename T_DESCRIPTOR_RANGE_INTERNAL>
void DeserializeRootSignatureTemplate(_In_reads_bytes_(SrcDataSizeInBytes) const void *pSrcData,
                                      _In_ uint32_t SrcDataSizeInBytes,
                                      DxilRootSignatureVersion DescVersion,
                                      T_ROOT_SIGNATURE_DESC &RootSignatureDesc) {
  // Note that in case of failure, outside code must deallocate memory.
  T_ROOT_SIGNATURE_DESC *pRootSignature = &RootSignatureDesc;
  const char *pData = (const char *)pSrcData;
  const char *pMaxPtr = pData + SrcDataSizeInBytes;
  UNREFERENCED_PARAMETER(DescVersion);
  DXASSERT_NOMSG(((const uint32_t*)pData)[0] == (uint32_t)DescVersion);

  // Root signature.
  IFTBOOL(pData + sizeof(DxilContainerRootSignatureDesc) <= pMaxPtr, E_FAIL);
  const DxilContainerRootSignatureDesc *pRS = (const DxilContainerRootSignatureDesc *)pData;
  pRootSignature->Flags = (DxilRootSignatureFlags)pRS->Flags;
  pRootSignature->NumParameters = pRS->NumParameters;
  pRootSignature->NumStaticSamplers = pRS->NumStaticSamplers;
  // Intialize all pointers early so that clean up works properly.
  pRootSignature->pParameters = nullptr;
  pRootSignature->pStaticSamplers = nullptr;

  size_t s = sizeof(DxilContainerRootParameter)*pRS->NumParameters;
  const DxilContainerRootParameter *pInRTS = (const DxilContainerRootParameter *)(pData + pRS->RootParametersOffset);
  IFTBOOL(((const char*)pInRTS) + s <= pMaxPtr, E_FAIL);
  if (pRootSignature->NumParameters) {
    pRootSignature->pParameters = new T_ROOT_PARAMETER[pRootSignature->NumParameters];
  }
  memset((void *)pRootSignature->pParameters, 0, s);

  for(unsigned iRP = 0; iRP < pRootSignature->NumParameters; iRP++) {
    DxilRootParameterType ParameterType = (DxilRootParameterType)pInRTS[iRP].ParameterType;
    T_ROOT_PARAMETER *pOutRTS = (T_ROOT_PARAMETER *)&pRootSignature->pParameters[iRP];
    pOutRTS->ParameterType = ParameterType;
    pOutRTS->ShaderVisibility = (DxilShaderVisibility)pInRTS[iRP].ShaderVisibility;
    switch(ParameterType) {
    case DxilRootParameterType::DescriptorTable: {
      const DxilContainerRootDescriptorTable *p1 = (const DxilContainerRootDescriptorTable*)(pData + pInRTS[iRP].PayloadOffset);
      IFTBOOL((const char*)p1 + sizeof(DxilContainerRootDescriptorTable) <= pMaxPtr, E_FAIL);
      pOutRTS->DescriptorTable.NumDescriptorRanges = p1->NumDescriptorRanges;
      pOutRTS->DescriptorTable.pDescriptorRanges = nullptr;
      const T_DESCRIPTOR_RANGE_INTERNAL *p2 = (const T_DESCRIPTOR_RANGE_INTERNAL*)(pData + p1->DescriptorRangesOffset);
      IFTBOOL((const char*)p2 + sizeof(T_DESCRIPTOR_RANGE_INTERNAL) <= pMaxPtr, E_FAIL);
      if (p1->NumDescriptorRanges) {
        pOutRTS->DescriptorTable.pDescriptorRanges = new T_DESCRIPTOR_RANGE[p1->NumDescriptorRanges];
      }
      for (unsigned i = 0; i < p1->NumDescriptorRanges; i++) {
        T_DESCRIPTOR_RANGE *p3 = (T_DESCRIPTOR_RANGE *)&pOutRTS->DescriptorTable.pDescriptorRanges[i];
        p3->RangeType                         = (DxilDescriptorRangeType)p2[i].RangeType;
        p3->NumDescriptors                    = p2[i].NumDescriptors;
        p3->BaseShaderRegister                = p2[i].BaseShaderRegister;
        p3->RegisterSpace                     = p2[i].RegisterSpace;
        p3->OffsetInDescriptorsFromTableStart = p2[i].OffsetInDescriptorsFromTableStart;
        DxilDescriptorRangeFlags Flags = GetFlags(p2[i]);
        SetFlags(*p3, Flags);
      }
      break;
    }
    case DxilRootParameterType::Constants32Bit: {
      const DxilRootConstants *p = (const DxilRootConstants*)(pData + pInRTS[iRP].PayloadOffset);
      IFTBOOL((const char*)p + sizeof(DxilRootConstants) <= pMaxPtr, E_FAIL);
      pOutRTS->Constants.Num32BitValues = p->Num32BitValues;
      pOutRTS->Constants.ShaderRegister = p->ShaderRegister;
      pOutRTS->Constants.RegisterSpace  = p->RegisterSpace;
      break;
    }
    case DxilRootParameterType::CBV:
    case DxilRootParameterType::SRV:
    case DxilRootParameterType::UAV: {
      const T_ROOT_DESCRIPTOR *p = (const T_ROOT_DESCRIPTOR *)(pData + pInRTS[iRP].PayloadOffset);
      IFTBOOL((const char*)p + sizeof(T_ROOT_DESCRIPTOR) <= pMaxPtr, E_FAIL);
      pOutRTS->Descriptor.ShaderRegister = p->ShaderRegister;
      pOutRTS->Descriptor.RegisterSpace  = p->RegisterSpace;
      DxilRootDescriptorFlags Flags = GetFlags(*p);
      SetFlags(pOutRTS->Descriptor, Flags);
      break;
    }
    default:
      IFT(E_FAIL);
    }
  }

  s = sizeof(DxilStaticSamplerDesc)*pRS->NumStaticSamplers;
  const DxilStaticSamplerDesc *pInSS = (const DxilStaticSamplerDesc *)(pData + pRS->StaticSamplersOffset);
  IFTBOOL(((const char*)pInSS) + s <= pMaxPtr, E_FAIL);
  if (pRootSignature->NumStaticSamplers) {
    pRootSignature->pStaticSamplers = new DxilStaticSamplerDesc[pRootSignature->NumStaticSamplers];
  }
  memcpy((void*)pRootSignature->pStaticSamplers, pInSS, s);
}

_Use_decl_annotations_
void DeserializeRootSignature(const void *pSrcData,
                              uint32_t SrcDataSizeInBytes,
                              const DxilVersionedRootSignatureDesc **ppRootSignature) {
  DxilVersionedRootSignatureDesc *pRootSignature = nullptr;
  IFTBOOL(pSrcData != nullptr && SrcDataSizeInBytes != 0 && ppRootSignature != nullptr, E_INVALIDARG);
  IFTBOOL(*ppRootSignature == nullptr, E_INVALIDARG);
  const char *pData = (const char *)pSrcData;
  IFTBOOL(pData + sizeof(uint32_t) < pData + SrcDataSizeInBytes, E_FAIL);

  DxilRootSignatureVersion Version = (const DxilRootSignatureVersion)((const uint32_t*)pData)[0];

  pRootSignature = new DxilVersionedRootSignatureDesc();

  try {
    switch (Version) {
    case DxilRootSignatureVersion::Version_1_0:
      pRootSignature->Version = DxilRootSignatureVersion::Version_1_0;
      DeserializeRootSignatureTemplate<
         DxilRootSignatureDesc,
         DxilRootParameter,
         DxilRootDescriptor,
         DxilRootDescriptor,
         DxilDescriptorRange,
         DxilContainerDescriptorRange>(pSrcData,
                                       SrcDataSizeInBytes,
                                       DxilRootSignatureVersion::Version_1_0,
                                       pRootSignature->Desc_1_0);
      break;

    case DxilRootSignatureVersion::Version_1_1:
      pRootSignature->Version = DxilRootSignatureVersion::Version_1_1;
      DeserializeRootSignatureTemplate<
         DxilRootSignatureDesc1,
         DxilRootParameter1,
         DxilRootDescriptor1,
         DxilContainerRootDescriptor1,
         DxilDescriptorRange1,
         DxilContainerDescriptorRange1>(pSrcData,
                                        SrcDataSizeInBytes,
                                        DxilRootSignatureVersion::Version_1_1,
                                        pRootSignature->Desc_1_1);
      break;

    default:
      IFT(E_FAIL);
      break;
    }
  } catch(...) {
    DeleteRootSignature(pRootSignature);
    throw;
  }

  *ppRootSignature = pRootSignature;
}

} // namespace hlsl
