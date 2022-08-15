///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilSignatureElement.h                                                    //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Class to pack HLSL signature element.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "llvm/ADT/StringRef.h"
#include "dxc/DXIL/DxilSemantic.h"
#include "dxc/DXIL/DxilInterpolationMode.h"
#include "dxc/DXIL/DxilCompType.h"
#include "dxc/HLSL/DxilSignatureAllocator.h"
#include <string>
#include <vector>
#include <limits.h>

namespace hlsl {

class ShaderModel;

class DxilPackElement : public DxilSignatureAllocator::PackElement {
  DxilSignatureElement *m_pSE;
  bool m_bUseMinPrecision;

public:
  DxilPackElement(DxilSignatureElement *pSE, bool useMinPrecision) : m_pSE(pSE), m_bUseMinPrecision(useMinPrecision) {}
  ~DxilPackElement() override {}
  uint32_t GetID() const override { return m_pSE->GetID(); }
  DXIL::SemanticKind GetKind() const override { return m_pSE->GetKind(); }
  DXIL::InterpolationMode GetInterpolationMode() const override { return m_pSE->GetInterpolationMode()->GetKind(); }
  DXIL::SemanticInterpretationKind GetInterpretation() const override { return m_pSE->GetInterpretation(); }
  DXIL::SignatureDataWidth GetDataBitWidth() const override {
    uint8_t size = m_pSE->GetCompType().GetSizeInBits();
    // bool, min precision, or 32 bit types map to 32 bit size.
    if (size == 16) {
      return m_bUseMinPrecision ? DXIL::SignatureDataWidth::Bits32 : DXIL::SignatureDataWidth::Bits16;
    }
    else if (size == 1 || size == 32) {
      return DXIL::SignatureDataWidth::Bits32;
    }
    return DXIL::SignatureDataWidth::Undefined;
  }
  uint32_t GetRows() const override { return m_pSE->GetRows(); }
  uint32_t GetCols() const override { return m_pSE->GetCols(); }
  bool IsAllocated() const override { return m_pSE->IsAllocated(); }
  uint32_t GetStartRow() const override { return m_pSE->GetStartRow(); }
  uint32_t GetStartCol() const override { return m_pSE->GetStartCol(); }

  void ClearLocation() override {
    m_pSE->SetStartRow(-1);
    m_pSE->SetStartCol(-1);
  }
  void SetLocation(uint32_t Row, uint32_t Col) override {
    m_pSE->SetStartRow(Row);
    m_pSE->SetStartCol(Col);
  }

  DxilSignatureElement *Get() { return m_pSE; }
};

class DxilSignature;
// Packs the signature elements per DXIL constraints and returns the number of rows used for the signature.
unsigned PackDxilSignature(DxilSignature &sig, DXIL::PackingStrategy packing);

} // namespace hlsl
