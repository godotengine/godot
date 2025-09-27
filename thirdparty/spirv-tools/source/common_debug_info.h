// Copyright (c) 2021 The Khronos Group Inc.
// Copyright (c) 2021 Valve Corporation
// Copyright (c) 2021 LunarG Inc.
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

#ifndef SOURCE_COMMON_DEBUG_INFO_HEADER_H_
#define SOURCE_COMMON_DEBUG_INFO_HEADER_H_

// This enum defines the known common set of instructions that are the same
// between OpenCL.DebugInfo.100 and NonSemantic.Shader.DebugInfo.100.
// Note that NonSemantic.Shader.* instructions can still have slightly
// different encoding, as it does not use literals anywhere and only constants.
enum CommonDebugInfoInstructions {
  CommonDebugInfoDebugInfoNone = 0,
  CommonDebugInfoDebugCompilationUnit = 1,
  CommonDebugInfoDebugTypeBasic = 2,
  CommonDebugInfoDebugTypePointer = 3,
  CommonDebugInfoDebugTypeQualifier = 4,
  CommonDebugInfoDebugTypeArray = 5,
  CommonDebugInfoDebugTypeVector = 6,
  CommonDebugInfoDebugTypedef = 7,
  CommonDebugInfoDebugTypeFunction = 8,
  CommonDebugInfoDebugTypeEnum = 9,
  CommonDebugInfoDebugTypeComposite = 10,
  CommonDebugInfoDebugTypeMember = 11,
  CommonDebugInfoDebugTypeInheritance = 12,
  CommonDebugInfoDebugTypePtrToMember = 13,
  CommonDebugInfoDebugTypeTemplate = 14,
  CommonDebugInfoDebugTypeTemplateParameter = 15,
  CommonDebugInfoDebugTypeTemplateTemplateParameter = 16,
  CommonDebugInfoDebugTypeTemplateParameterPack = 17,
  CommonDebugInfoDebugGlobalVariable = 18,
  CommonDebugInfoDebugFunctionDeclaration = 19,
  CommonDebugInfoDebugFunction = 20,
  CommonDebugInfoDebugLexicalBlock = 21,
  CommonDebugInfoDebugLexicalBlockDiscriminator = 22,
  CommonDebugInfoDebugScope = 23,
  CommonDebugInfoDebugNoScope = 24,
  CommonDebugInfoDebugInlinedAt = 25,
  CommonDebugInfoDebugLocalVariable = 26,
  CommonDebugInfoDebugInlinedVariable = 27,
  CommonDebugInfoDebugDeclare = 28,
  CommonDebugInfoDebugValue = 29,
  CommonDebugInfoDebugOperation = 30,
  CommonDebugInfoDebugExpression = 31,
  CommonDebugInfoDebugMacroDef = 32,
  CommonDebugInfoDebugMacroUndef = 33,
  CommonDebugInfoDebugImportedEntity = 34,
  CommonDebugInfoDebugSource = 35,
  CommonDebugInfoInstructionsMax = 0x7ffffff
};

#endif  // SOURCE_COMMON_DEBUG_INFO_HEADER_H_
