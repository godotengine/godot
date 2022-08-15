///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilRootSignatureConvert.cpp                                              //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Convert root signature structures.                                        //
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


template<typename IN_DXIL_ROOT_SIGNATURE_DESC,
  typename OUT_DXIL_ROOT_SIGNATURE_DESC,
  typename OUT_DXIL_ROOT_PARAMETER,
  typename OUT_DXIL_ROOT_DESCRIPTOR,
  typename OUT_DXIL_DESCRIPTOR_RANGE>
void ConvertRootSignatureTemplate(const IN_DXIL_ROOT_SIGNATURE_DESC &DescIn,
                                  DxilRootSignatureVersion DescVersionOut,
                                  OUT_DXIL_ROOT_SIGNATURE_DESC &DescOut)
{
  const IN_DXIL_ROOT_SIGNATURE_DESC *pDescIn = &DescIn;
  OUT_DXIL_ROOT_SIGNATURE_DESC *pDescOut = &DescOut;

  // Root signature descriptor.
  pDescOut->Flags = pDescIn->Flags;
  pDescOut->NumParameters = 0;
  pDescOut->NumStaticSamplers = 0;
  // Intialize all pointers early so that clean up works properly.
  pDescOut->pParameters = nullptr;
  pDescOut->pStaticSamplers = nullptr;

  // Root signature parameters.
  if (pDescIn->NumParameters > 0) {
    pDescOut->pParameters = new OUT_DXIL_ROOT_PARAMETER[pDescIn->NumParameters];
    pDescOut->NumParameters = pDescIn->NumParameters;
    memset((void *)pDescOut->pParameters, 0, pDescOut->NumParameters*sizeof(OUT_DXIL_ROOT_PARAMETER));
  }

  for (unsigned iRP = 0; iRP < pDescIn->NumParameters; iRP++) {
    const auto &ParamIn = pDescIn->pParameters[iRP];
    OUT_DXIL_ROOT_PARAMETER &ParamOut = (OUT_DXIL_ROOT_PARAMETER &)pDescOut->pParameters[iRP];

    ParamOut.ParameterType = ParamIn.ParameterType;
    ParamOut.ShaderVisibility = ParamIn.ShaderVisibility;

    switch (ParamIn.ParameterType) {
    case DxilRootParameterType::DescriptorTable: {
      ParamOut.DescriptorTable.pDescriptorRanges = nullptr;
      unsigned NumRanges = ParamIn.DescriptorTable.NumDescriptorRanges;
      if (NumRanges > 0) {
        ParamOut.DescriptorTable.pDescriptorRanges = new OUT_DXIL_DESCRIPTOR_RANGE[NumRanges];
        ParamOut.DescriptorTable.NumDescriptorRanges = NumRanges;
      }

      for (unsigned i = 0; i < NumRanges; i++) {
        const auto &RangeIn = ParamIn.DescriptorTable.pDescriptorRanges[i];
        OUT_DXIL_DESCRIPTOR_RANGE &RangeOut = (OUT_DXIL_DESCRIPTOR_RANGE &)ParamOut.DescriptorTable.pDescriptorRanges[i];

        RangeOut.RangeType = RangeIn.RangeType;
        RangeOut.NumDescriptors = RangeIn.NumDescriptors;
        RangeOut.BaseShaderRegister = RangeIn.BaseShaderRegister;
        RangeOut.RegisterSpace = RangeIn.RegisterSpace;
        RangeOut.OffsetInDescriptorsFromTableStart = RangeIn.OffsetInDescriptorsFromTableStart;
        DxilDescriptorRangeFlags Flags = GetFlags(RangeIn);
        SetFlags(RangeOut, Flags);
      }
      break;
    }
    case DxilRootParameterType::Constants32Bit: {
      ParamOut.Constants.Num32BitValues = ParamIn.Constants.Num32BitValues;
      ParamOut.Constants.ShaderRegister = ParamIn.Constants.ShaderRegister;
      ParamOut.Constants.RegisterSpace = ParamIn.Constants.RegisterSpace;
      break;
    }
    case DxilRootParameterType::CBV:
    case DxilRootParameterType::SRV:
    case DxilRootParameterType::UAV: {
      ParamOut.Descriptor.ShaderRegister = ParamIn.Descriptor.ShaderRegister;
      ParamOut.Descriptor.RegisterSpace = ParamIn.Descriptor.RegisterSpace;
      DxilRootDescriptorFlags Flags = GetFlags(ParamIn.Descriptor);
      SetFlags(ParamOut.Descriptor, Flags);
      break;
    }
    default:
      IFT(E_FAIL);
    }
  }

  // Static samplers.
  if (pDescIn->NumStaticSamplers > 0) {
    pDescOut->pStaticSamplers = new DxilStaticSamplerDesc[pDescIn->NumStaticSamplers];
    pDescOut->NumStaticSamplers = pDescIn->NumStaticSamplers;
    memcpy((void*)pDescOut->pStaticSamplers, pDescIn->pStaticSamplers, pDescOut->NumStaticSamplers*sizeof(DxilStaticSamplerDesc));
  }
}

void ConvertRootSignature(const DxilVersionedRootSignatureDesc * pRootSignatureIn,
                          DxilRootSignatureVersion RootSignatureVersionOut,
                          const DxilVersionedRootSignatureDesc ** ppRootSignatureOut) {
  IFTBOOL(pRootSignatureIn != nullptr && ppRootSignatureOut != nullptr, E_INVALIDARG);
  *ppRootSignatureOut = nullptr;

  if (pRootSignatureIn->Version == RootSignatureVersionOut){
    // No conversion. Return the original root signature pointer; no cloning.
    *ppRootSignatureOut = pRootSignatureIn;
    return;
  }

  DxilVersionedRootSignatureDesc *pRootSignatureOut = nullptr;

  try {
    pRootSignatureOut = new DxilVersionedRootSignatureDesc();
    memset(pRootSignatureOut, 0, sizeof(*pRootSignatureOut));

    // Convert root signature.
    switch (RootSignatureVersionOut) {
    case DxilRootSignatureVersion::Version_1_0:
      switch (pRootSignatureIn->Version) {
      case DxilRootSignatureVersion::Version_1_1:
        pRootSignatureOut->Version = DxilRootSignatureVersion::Version_1_0;
        ConvertRootSignatureTemplate<
          DxilRootSignatureDesc1,
          DxilRootSignatureDesc,
          DxilRootParameter,
          DxilRootDescriptor,
          DxilDescriptorRange>(pRootSignatureIn->Desc_1_1,
            DxilRootSignatureVersion::Version_1_0,
            pRootSignatureOut->Desc_1_0);
        break;
      default:
        IFT(E_INVALIDARG);
      }
      break;

    case DxilRootSignatureVersion::Version_1_1:
      switch (pRootSignatureIn->Version) {
      case DxilRootSignatureVersion::Version_1_0:
        pRootSignatureOut->Version = DxilRootSignatureVersion::Version_1_1;
        ConvertRootSignatureTemplate<
          DxilRootSignatureDesc,
          DxilRootSignatureDesc1,
          DxilRootParameter1,
          DxilRootDescriptor1,
          DxilDescriptorRange1>(pRootSignatureIn->Desc_1_0,
            DxilRootSignatureVersion::Version_1_1,
            pRootSignatureOut->Desc_1_1);
        break;
      default:
        IFT(E_INVALIDARG);
      }
      break;

    default:
      IFT(E_INVALIDARG);
      break;
    }
  }
  catch (...) {
    DeleteRootSignature(pRootSignatureOut);
    throw;
  }

  *ppRootSignatureOut = pRootSignatureOut;
}


} // namespace hlsl
