// Copyright (c) 2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SOURCE_OPT_REFLECT_H_
#define SOURCE_OPT_REFLECT_H_

#include "source/latest_version_spirv_header.h"

namespace spvtools {
namespace opt {

// Note that as SPIR-V evolves over time, new opcodes may appear. So the
// following functions tend to be outdated and should be updated when SPIR-V
// version bumps.

inline bool IsDebug1Inst(SpvOp opcode) {
  return (opcode >= SpvOpSourceContinued && opcode <= SpvOpSourceExtension) ||
         opcode == SpvOpString;
}
inline bool IsDebug2Inst(SpvOp opcode) {
  return opcode == SpvOpName || opcode == SpvOpMemberName;
}
inline bool IsDebug3Inst(SpvOp opcode) {
  return opcode == SpvOpModuleProcessed;
}
inline bool IsDebugLineInst(SpvOp opcode) {
  return opcode == SpvOpLine || opcode == SpvOpNoLine;
}
inline bool IsAnnotationInst(SpvOp opcode) {
  return (opcode >= SpvOpDecorate && opcode <= SpvOpGroupMemberDecorate) ||
         opcode == SpvOpDecorateId || opcode == SpvOpDecorateStringGOOGLE ||
         opcode == SpvOpMemberDecorateStringGOOGLE;
}
inline bool IsTypeInst(SpvOp opcode) {
  return (opcode >= SpvOpTypeVoid && opcode <= SpvOpTypeForwardPointer) ||
         opcode == SpvOpTypePipeStorage || opcode == SpvOpTypeNamedBarrier ||
         opcode == SpvOpTypeAccelerationStructureNV ||
         opcode == SpvOpTypeCooperativeMatrixNV;
}
inline bool IsConstantInst(SpvOp opcode) {
  return opcode >= SpvOpConstantTrue && opcode <= SpvOpSpecConstantOp;
}
inline bool IsCompileTimeConstantInst(SpvOp opcode) {
  return opcode >= SpvOpConstantTrue && opcode <= SpvOpConstantNull;
}
inline bool IsSpecConstantInst(SpvOp opcode) {
  return opcode >= SpvOpSpecConstantTrue && opcode <= SpvOpSpecConstantOp;
}
inline bool IsTerminatorInst(SpvOp opcode) {
  return opcode >= SpvOpBranch && opcode <= SpvOpUnreachable;
}

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_REFLECT_H_
