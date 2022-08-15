///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// RDATDxilSubobjects.cpp                                                    //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Implement LoadSubobjectsFromRDAT, depending on both DXIL and RDAT libs.   //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/Support/Global.h"
#include "dxc/Support/Unicode.h"
#include "dxc/Support/WinIncludes.h"
#include "dxc/DXIL/DxilSubobject.h"
#include "dxc/DxilContainer/DxilRuntimeReflection.h"

namespace hlsl {

bool LoadSubobjectsFromRDAT(DxilSubobjects &subobjects, const RDAT::DxilRuntimeData &rdat) {
  auto table = rdat.GetSubobjectTable();
  if (!table)
    return false;
  bool result = true;
  for (unsigned i = 0; i < table.Count(); ++i) {
    try {
      auto reader = table[i];
      DXIL::SubobjectKind kind = reader.getKind();
      bool bLocalRS = false;
      switch (kind) {
      case DXIL::SubobjectKind::StateObjectConfig:
        subobjects.CreateStateObjectConfig(reader.getName(),
          reader.getStateObjectConfig().getFlags());
        break;
      case DXIL::SubobjectKind::LocalRootSignature:
        bLocalRS = true;
      case DXIL::SubobjectKind::GlobalRootSignature:
        if (!reader.getRootSignature()) {
          result = false;
          continue;
        }
        subobjects.CreateRootSignature(reader.getName(), bLocalRS,
                                       reader.getRootSignature().getData(),
                                       reader.getRootSignature().sizeData());
        break;
      case DXIL::SubobjectKind::SubobjectToExportsAssociation: {
        auto association = reader.getSubobjectToExportsAssociation();
        auto exports = association.getExports();
        uint32_t NumExports = exports.Count();
        std::vector<llvm::StringRef> Exports;
        Exports.resize(NumExports);
        for (unsigned i = 0; i < NumExports; ++i) {
          Exports[i] = exports[i];
        }
        subobjects.CreateSubobjectToExportsAssociation(reader.getName(),
          association.getSubobject(), Exports.data(), NumExports);
        break;
      }
      case DXIL::SubobjectKind::RaytracingShaderConfig:
        subobjects.CreateRaytracingShaderConfig(reader.getName(),
          reader.getRaytracingShaderConfig().getMaxPayloadSizeInBytes(),
          reader.getRaytracingShaderConfig().getMaxAttributeSizeInBytes());
        break;
      case DXIL::SubobjectKind::RaytracingPipelineConfig:
        subobjects.CreateRaytracingPipelineConfig(reader.getName(),
          reader.getRaytracingPipelineConfig().getMaxTraceRecursionDepth());
        break;
      case DXIL::SubobjectKind::HitGroup:
        subobjects.CreateHitGroup(reader.getName(),
          reader.getHitGroup().getType(),
          reader.getHitGroup().getAnyHit(),
          reader.getHitGroup().getClosestHit(),
          reader.getHitGroup().getIntersection());
        break;
      case DXIL::SubobjectKind::RaytracingPipelineConfig1:
        subobjects.CreateRaytracingPipelineConfig1(
          reader.getName(),
          reader.getRaytracingPipelineConfig1().getMaxTraceRecursionDepth(),
          reader.getRaytracingPipelineConfig1().getFlags());
        break;
      }
    }
    catch (hlsl::Exception &) {
      result = false;
    }
  }
  return result;
}

} // namespace hlsl

