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
#include "source/opcode.h"

namespace spvtools {
namespace opt {

// Note that as SPIR-V evolves over time, new opcodes may appear. So the
// following functions tend to be outdated and should be updated when SPIR-V
// version bumps.

inline bool IsDebug1Inst(spv::Op opcode) {
  return (opcode >= spv::Op::OpSourceContinued &&
          opcode <= spv::Op::OpSourceExtension) ||
         opcode == spv::Op::OpString;
}
inline bool IsDebug2Inst(spv::Op opcode) {
  return opcode == spv::Op::OpName || opcode == spv::Op::OpMemberName;
}
inline bool IsDebug3Inst(spv::Op opcode) {
  return opcode == spv::Op::OpModuleProcessed;
}
inline bool IsOpLineInst(spv::Op opcode) {
  return opcode == spv::Op::OpLine || opcode == spv::Op::OpNoLine;
}
inline bool IsAnnotationInst(spv::Op opcode) {
  return (opcode >= spv::Op::OpDecorate &&
          opcode <= spv::Op::OpGroupMemberDecorate) ||
         opcode == spv::Op::OpDecorateId ||
         opcode == spv::Op::OpDecorateStringGOOGLE ||
         opcode == spv::Op::OpMemberDecorateStringGOOGLE;
}
inline bool IsTypeInst(spv::Op opcode) {
  return spvOpcodeGeneratesType(opcode) ||
         opcode == spv::Op::OpTypeForwardPointer;
}
inline bool IsConstantInst(spv::Op opcode) {
  return spvOpcodeIsConstant(opcode);
}
inline bool IsSpecConstantInst(spv::Op opcode) {
  return spvOpcodeIsSpecConstant(opcode);
}

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_REFLECT_H_
