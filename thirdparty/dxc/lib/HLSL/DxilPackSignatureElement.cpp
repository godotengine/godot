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


#include "dxc/Support/Global.h"
#include "dxc/DXIL/DxilSignature.h"
#include "dxc/HLSL/DxilSignatureAllocator.h"
#include "dxc/DXIL/DxilSigPoint.h"
#include "dxc/HLSL/DxilPackSignatureElement.h"

using namespace hlsl;
using namespace llvm;

namespace hlsl {
unsigned PackDxilSignature(DxilSignature &sig, DXIL::PackingStrategy packing) {
  unsigned rowsUsed = 0;
  bool bUseMinPrecision = sig.UseMinPrecision();
  // Transfer to elements derived from DxilSignatureAllocator::PackElement
  std::vector<DxilPackElement> packElements;
  for (auto &SE : sig.GetElements()) {
    if (DxilSignature::ShouldBeAllocated(SE.get()->GetInterpretation()))
      packElements.emplace_back(SE.get(), bUseMinPrecision);
  }

  DXIL::SigPointKind Kind = sig.GetSigPointKind();
  if (Kind == DXIL::SigPointKind::GSOut) {
    // Special case due to support for multiple streams
    DxilSignatureAllocator alloc[4] = {{32, bUseMinPrecision},
                                       {32, bUseMinPrecision},
                                       {32, bUseMinPrecision},
                                       {32, bUseMinPrecision}};
    std::vector<DxilSignatureAllocator::PackElement*> elements[4];
    for (auto &SE : packElements) {
      elements[SE.Get()->GetOutputStream()].push_back(&SE);
    }
    for (unsigned i = 0; i < 4; ++i) {
      if (!elements[i].empty()) {
        unsigned streamRowsUsed = 0;
        switch (packing) {
        case DXIL::PackingStrategy::PrefixStable:
          streamRowsUsed = alloc[i].PackPrefixStable(elements[i], 0, 32);
          break;
        case DXIL::PackingStrategy::Optimized:
          streamRowsUsed = alloc[i].PackOptimized(elements[i], 0, 32);
          break;
        default:
          DXASSERT(false, "otherwise, invalid packing strategy supplied");
        }
        if (streamRowsUsed > rowsUsed)
          rowsUsed = streamRowsUsed;
      }
    }
    // rowsUsed isn't really meaningful in this case.
    return rowsUsed;
  }

  const SigPoint *SP = SigPoint::GetSigPoint(Kind);
  DXIL::PackingKind PK = SP->GetPackingKind();

  switch (PK) {
  case DXIL::PackingKind::None:
    // no packing.
    break;

  case DXIL::PackingKind::InputAssembler:
    // incrementally assign each element that belongs in the signature to the start of the next free row
    for (auto &SE : packElements) {
      SE.SetLocation(rowsUsed, 0);
      rowsUsed += SE.GetRows();
    }
    break;

  case DXIL::PackingKind::Vertex:
  case DXIL::PackingKind::PatchConstant: {
      DxilSignatureAllocator alloc(32, bUseMinPrecision);
      std::vector<DxilSignatureAllocator::PackElement*> elements;
      elements.reserve(packElements.size());
      for (auto &SE : packElements){
        elements.push_back(&SE);
      }
      switch (packing) {
      case DXIL::PackingStrategy::PrefixStable:
        rowsUsed = alloc.PackPrefixStable(elements, 0, 32);
        break;
      case DXIL::PackingStrategy::Optimized:
        rowsUsed = alloc.PackOptimized(elements, 0, 32);
        break;
      default:
        DXASSERT(false, "otherwise, invalid packing strategy supplied");
      }
    }
    break;

  case DXIL::PackingKind::Target:
    // for SV_Target, assign rows according to semantic index, the rest are unassigned (-1)
    // Note: Overlapping semantic indices should be checked elsewhere
    for (auto &SE : packElements) {
      if (SE.GetKind() != DXIL::SemanticKind::Target)
        continue;
      unsigned row = SE.Get()->GetSemanticStartIndex();
      SE.SetLocation(row, 0);
      DXASSERT(SE.GetRows() == 1, "otherwise, SV_Target output not broken into separate rows earlier");
      row += SE.GetRows();
      if (rowsUsed < row)
        rowsUsed = row;
    }
    break;

  case DXIL::PackingKind::Invalid:
  default:
    DXASSERT(false, "unexpected PackingKind.");
  }

  return rowsUsed;
}
}

#include <algorithm>
#include "dxc/HLSL/DxilSignatureAllocator.inl"