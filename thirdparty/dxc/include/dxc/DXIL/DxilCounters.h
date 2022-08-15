///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilCounters.h                                                            //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Counters for Dxil instructions types.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <stdint.h>

namespace llvm {
  class Module;
  class StringRef;
}

namespace hlsl {

struct DxilCounters {
  // <py::lines('OPCODE-COUNTERS')>['uint32_t %s = 0;' % c for c in hctdb_instrhelp.get_counters()]</py>
  // OPCODE-COUNTERS:BEGIN
  uint32_t array_local_bytes = 0;
  uint32_t array_local_ldst = 0;
  uint32_t array_static_bytes = 0;
  uint32_t array_static_ldst = 0;
  uint32_t array_tgsm_bytes = 0;
  uint32_t array_tgsm_ldst = 0;
  uint32_t atomic = 0;
  uint32_t barrier = 0;
  uint32_t branches = 0;
  uint32_t fence = 0;
  uint32_t floats = 0;
  uint32_t gs_cut = 0;
  uint32_t gs_emit = 0;
  uint32_t insts = 0;
  uint32_t ints = 0;
  uint32_t sig_ld = 0;
  uint32_t sig_st = 0;
  uint32_t tex_bias = 0;
  uint32_t tex_cmp = 0;
  uint32_t tex_grad = 0;
  uint32_t tex_load = 0;
  uint32_t tex_norm = 0;
  uint32_t tex_store = 0;
  uint32_t uints = 0;
  // OPCODE-COUNTERS:END

  uint32_t AllArrayBytes() {
    return array_local_bytes
      + array_static_bytes
      + array_tgsm_bytes;
  }
  uint32_t AllArrayAccesses() {
    return array_local_ldst
      + array_static_ldst
      + array_tgsm_ldst;
  }
};

void CountInstructions(llvm::Module &M, DxilCounters& counters);
uint32_t *LookupByName(llvm::StringRef name, DxilCounters& counters);

} // namespace hlsl
