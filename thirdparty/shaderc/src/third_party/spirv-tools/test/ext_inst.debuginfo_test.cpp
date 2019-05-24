// Copyright (c) 2017 Google Inc.
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

#include <string>
#include <vector>

#include "DebugInfo.h"
#include "gmock/gmock.h"
#include "test/test_fixture.h"
#include "test/unit_spirv.h"

// This file tests the correctness of encoding and decoding of instructions
// involving the DebugInfo extended instruction set.
// Semantic correctness should be the responsibility of validator.
//
// See https://www.khronos.org/registry/spir-v/specs/1.0/DebugInfo.html

namespace spvtools {
namespace {

using spvtest::Concatenate;
using spvtest::MakeInstruction;
using spvtest::MakeVector;
using testing::Eq;

struct InstructionCase {
  uint32_t opcode;
  std::string name;
  std::string operands;
  std::vector<uint32_t> expected_operands;
};

using ExtInstDebugInfoRoundTripTest =
    spvtest::TextToBinaryTestBase<::testing::TestWithParam<InstructionCase>>;
using ExtInstDebugInfoRoundTripTestExplicit = spvtest::TextToBinaryTest;

TEST_P(ExtInstDebugInfoRoundTripTest, ParameterizedExtInst) {
  const std::string input =
      "%1 = OpExtInstImport \"DebugInfo\"\n"
      "%3 = OpExtInst %2 %1 " +
      GetParam().name + GetParam().operands + "\n";
  // First make sure it assembles correctly.
  EXPECT_THAT(
      CompiledInstructions(input),
      Eq(Concatenate(
          {MakeInstruction(SpvOpExtInstImport, {1}, MakeVector("DebugInfo")),
           MakeInstruction(SpvOpExtInst, {2, 3, 1, GetParam().opcode},
                           GetParam().expected_operands)})))
      << input;
  // Now check the round trip through the disassembler.
  EXPECT_THAT(EncodeAndDecodeSuccessfully(input), input) << input;
}

#define CASE_0(Enum)                                      \
  {                                                       \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, "", {} \
  }

#define CASE_ILL(Enum, L0, L1)                                           \
  {                                                                      \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, " %4 " #L0 " " #L1, { \
      4, L0, L1                                                          \
    }                                                                    \
  }

#define CASE_IL(Enum, L0)                                                \
  {                                                                      \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, " %4 " #L0, { 4, L0 } \
  }

#define CASE_I(Enum)                                            \
  {                                                             \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, " %4", { 4 } \
  }

#define CASE_II(Enum)                                                 \
  {                                                                   \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, " %4 %5", { 4, 5 } \
  }

#define CASE_III(Enum)                                                      \
  {                                                                         \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, " %4 %5 %6", { 4, 5, 6 } \
  }

#define CASE_IIII(Enum)                                              \
  {                                                                  \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, " %4 %5 %6 %7", { \
      4, 5, 6, 7                                                     \
    }                                                                \
  }

#define CASE_IIIII(Enum)                                                \
  {                                                                     \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, " %4 %5 %6 %7 %8", { \
      4, 5, 6, 7, 8                                                     \
    }                                                                   \
  }

#define CASE_IIIIII(Enum)                                                  \
  {                                                                        \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, " %4 %5 %6 %7 %8 %9", { \
      4, 5, 6, 7, 8, 9                                                     \
    }                                                                      \
  }

#define CASE_IIIIIII(Enum)                                                     \
  {                                                                            \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, " %4 %5 %6 %7 %8 %9 %10", { \
      4, 5, 6, 7, 8, 9, 10                                                     \
    }                                                                          \
  }

#define CASE_IIILLI(Enum, L0, L1)                  \
  {                                                \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, \
        " %4 %5 %6 " #L0 " " #L1 " %7", {          \
      4, 5, 6, L0, L1, 7                           \
    }                                              \
  }

#define CASE_IIILLIL(Enum, L0, L1, L2)             \
  {                                                \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, \
        " %4 %5 %6 " #L0 " " #L1 " %7 " #L2, {     \
      4, 5, 6, L0, L1, 7, L2                       \
    }                                              \
  }

#define CASE_IE(Enum, E0)                                        \
  {                                                              \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, " %4 " #E0, { \
      4, uint32_t(DebugInfo##E0)                                 \
    }                                                            \
  }

#define CASE_IIE(Enum, E0)                                          \
  {                                                                 \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, " %4 %5 " #E0, { \
      4, 5, uint32_t(DebugInfo##E0)                                 \
    }                                                               \
  }

#define CASE_ISF(Enum, S0, Fstr, Fnum)                                    \
  {                                                                       \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, " %4 " #S0 " " Fstr, { \
      4, uint32_t(SpvStorageClass##S0), Fnum                              \
    }                                                                     \
  }

#define CASE_LII(Enum, L0)                                             \
  {                                                                    \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, " " #L0 " %4 %5", { \
      L0, 4, 5                                                         \
    }                                                                  \
  }

#define CASE_ILI(Enum, L0)                                             \
  {                                                                    \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, " %4 " #L0 " %5", { \
      4, L0, 5                                                         \
    }                                                                  \
  }

#define CASE_ILII(Enum, L0)                                               \
  {                                                                       \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, " %4 " #L0 " %5 %6", { \
      4, L0, 5, 6                                                         \
    }                                                                     \
  }

#define CASE_ILLII(Enum, L0, L1)                   \
  {                                                \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, \
        " %4 " #L0 " " #L1 " %5 %6", {             \
      4, L0, L1, 5, 6                              \
    }                                              \
  }

#define CASE_IIILLIIF(Enum, L0, L1, Fstr, Fnum)    \
  {                                                \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, \
        " %4 %5 %6 " #L0 " " #L1 " %7 %8 " Fstr, { \
      4, 5, 6, L0, L1, 7, 8, Fnum                  \
    }                                              \
  }

#define CASE_IIILLIIFII(Enum, L0, L1, Fstr, Fnum)            \
  {                                                          \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum,           \
        " %4 %5 %6 " #L0 " " #L1 " %7 %8 " Fstr " %9 %10", { \
      4, 5, 6, L0, L1, 7, 8, Fnum, 9, 10                     \
    }                                                        \
  }

#define CASE_IIILLIIFIIII(Enum, L0, L1, Fstr, Fnum)                  \
  {                                                                  \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum,                   \
        " %4 %5 %6 " #L0 " " #L1 " %7 %8 " Fstr " %9 %10 %11 %12", { \
      4, 5, 6, L0, L1, 7, 8, Fnum, 9, 10, 11, 12                     \
    }                                                                \
  }

#define CASE_IIILLIIFIIIIII(Enum, L0, L1, Fstr, Fnum)                        \
  {                                                                          \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum,                           \
        " %4 %5 %6 " #L0 " " #L1 " %7 %8 " Fstr " %9 %10 %11 %12 %13 %14", { \
      4, 5, 6, L0, L1, 7, 8, Fnum, 9, 10, 11, 12, 13, 14                     \
    }                                                                        \
  }

#define CASE_IEILLIIF(Enum, E0, L0, L1, Fstr, Fnum)     \
  {                                                     \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum,      \
        " %4 " #E0 " %5 " #L0 " " #L1 " %6 %7 " Fstr, { \
      4, uint32_t(DebugInfo##E0), 5, L0, L1, 6, 7, Fnum \
    }                                                   \
  }

#define CASE_IEILLIIFI(Enum, E0, L0, L1, Fstr, Fnum)          \
  {                                                           \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum,            \
        " %4 " #E0 " %5 " #L0 " " #L1 " %6 %7 " Fstr " %8", { \
      4, uint32_t(DebugInfo##E0), 5, L0, L1, 6, 7, Fnum, 8    \
    }                                                         \
  }

#define CASE_IEILLIIFII(Enum, E0, L0, L1, Fstr, Fnum)            \
  {                                                              \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum,               \
        " %4 " #E0 " %5 " #L0 " " #L1 " %6 %7 " Fstr " %8 %9", { \
      4, uint32_t(DebugInfo##E0), 5, L0, L1, 6, 7, Fnum, 8, 9    \
    }                                                            \
  }

#define CASE_IEILLIIFIII(Enum, E0, L0, L1, Fstr, Fnum)               \
  {                                                                  \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum,                   \
        " %4 " #E0 " %5 " #L0 " " #L1 " %6 %7 " Fstr " %8 %9 %10", { \
      4, uint32_t(DebugInfo##E0), 5, L0, L1, 6, 7, Fnum, 8, 9, 10    \
    }                                                                \
  }

#define CASE_IEILLIIFIIII(Enum, E0, L0, L1, Fstr, Fnum)                  \
  {                                                                      \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum,                       \
        " %4 " #E0 " %5 " #L0 " " #L1 " %6 %7 " Fstr " %8 %9 %10 %11", { \
      4, uint32_t(DebugInfo##E0), 5, L0, L1, 6, 7, Fnum, 8, 9, 10, 11    \
    }                                                                    \
  }

#define CASE_IIILLIIIF(Enum, L0, L1, Fstr, Fnum)      \
  {                                                   \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum,    \
        " %4 %5 %6 " #L0 " " #L1 " %7 %8 %9 " Fstr, { \
      4, 5, 6, L0, L1, 7, 8, 9, Fnum                  \
    }                                                 \
  }

#define CASE_IIILLIIIFI(Enum, L0, L1, Fstr, Fnum)            \
  {                                                          \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum,           \
        " %4 %5 %6 " #L0 " " #L1 " %7 %8 %9 " Fstr " %10", { \
      4, 5, 6, L0, L1, 7, 8, 9, Fnum, 10                     \
    }                                                        \
  }

#define CASE_IIIIF(Enum, Fstr, Fnum)                                       \
  {                                                                        \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, " %4 %5 %6 %7 " Fstr, { \
      4, 5, 6, 7, Fnum                                                     \
    }                                                                      \
  }

#define CASE_IIILL(Enum, L0, L1)                                               \
  {                                                                            \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, " %4 %5 %6 " #L0 " " #L1, { \
      4, 5, 6, L0, L1                                                          \
    }                                                                          \
  }

#define CASE_IIIILL(Enum, L0, L1)                  \
  {                                                \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, \
        " %4 %5 %6 %7 " #L0 " " #L1, {             \
      4, 5, 6, 7, L0, L1                           \
    }                                              \
  }

#define CASE_IILLI(Enum, L0, L1)                   \
  {                                                \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, \
        " %4 %5 " #L0 " " #L1 " %6", {             \
      4, 5, L0, L1, 6                              \
    }                                              \
  }

#define CASE_IILLII(Enum, L0, L1)                  \
  {                                                \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, \
        " %4 %5 " #L0 " " #L1 " %6 %7", {          \
      4, 5, L0, L1, 6, 7                           \
    }                                              \
  }

#define CASE_IILLIII(Enum, L0, L1)                 \
  {                                                \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, \
        " %4 %5 " #L0 " " #L1 " %6 %7 %8", {       \
      4, 5, L0, L1, 6, 7, 8                        \
    }                                              \
  }

#define CASE_IILLIIII(Enum, L0, L1)                \
  {                                                \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, \
        " %4 %5 " #L0 " " #L1 " %6 %7 %8 %9", {    \
      4, 5, L0, L1, 6, 7, 8, 9                     \
    }                                              \
  }

#define CASE_IIILLIIFLI(Enum, L0, L1, Fstr, Fnum, L2)            \
  {                                                              \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum,               \
        " %4 %5 %6 " #L0 " " #L1 " %7 %8 " Fstr " " #L2 " %9", { \
      4, 5, 6, L0, L1, 7, 8, Fnum, L2, 9                         \
    }                                                            \
  }

#define CASE_IIILLIIFLII(Enum, L0, L1, Fstr, Fnum, L2)               \
  {                                                                  \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum,                   \
        " %4 %5 %6 " #L0 " " #L1 " %7 %8 " Fstr " " #L2 " %9 %10", { \
      4, 5, 6, L0, L1, 7, 8, Fnum, L2, 9, 10                         \
    }                                                                \
  }

#define CASE_E(Enum, E0)                                      \
  {                                                           \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, " " #E0, { \
      uint32_t(DebugInfo##E0)                                 \
    }                                                         \
  }

#define CASE_EL(Enum, E0, L0)                                         \
  {                                                                   \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, " " #E0 " " #L0, { \
      uint32_t(DebugInfo##E0), L0                                     \
    }                                                                 \
  }

#define CASE_ELL(Enum, E0, L0, L1)                                            \
  {                                                                           \
    uint32_t(DebugInfoDebug##Enum), "Debug" #Enum, " " #E0 " " #L0 " " #L1, { \
      uint32_t(DebugInfo##E0), L0, L1                                         \
    }                                                                         \
  }

// DebugInfo 4.1 Absent Debugging Information
INSTANTIATE_TEST_SUITE_P(DebugInfoDebugInfoNone, ExtInstDebugInfoRoundTripTest,
                         ::testing::ValuesIn(std::vector<InstructionCase>({
                             CASE_0(InfoNone),  // enum value 0
                         })));

// DebugInfo 4.2 Compilation Unit
INSTANTIATE_TEST_SUITE_P(DebugInfoDebugCompilationUnit,
                         ExtInstDebugInfoRoundTripTest,
                         ::testing::ValuesIn(std::vector<InstructionCase>({
                             CASE_ILL(CompilationUnit, 100, 42),
                         })));

// DebugInfo 4.3 Type instructions
INSTANTIATE_TEST_SUITE_P(DebugInfoDebugTypeBasic, ExtInstDebugInfoRoundTripTest,
                         ::testing::ValuesIn(std::vector<InstructionCase>({
                             CASE_IIE(TypeBasic, Unspecified),
                             CASE_IIE(TypeBasic, Address),
                             CASE_IIE(TypeBasic, Boolean),
                             CASE_IIE(TypeBasic, Float),
                             CASE_IIE(TypeBasic, Signed),
                             CASE_IIE(TypeBasic, SignedChar),
                             CASE_IIE(TypeBasic, Unsigned),
                             CASE_IIE(TypeBasic, UnsignedChar),
                         })));

// The FlagIsPublic is value is (1 << 0) | (1 << 2) which is the same
// as the bitwise-OR of FlagIsProtected and FlagIsPrivate.
// The disassembler will emit the compound expression instead.
// There is no simple fix for this.  This enum is not really a mask
// for the bottom two bits.
TEST_F(ExtInstDebugInfoRoundTripTestExplicit, FlagIsPublic) {
  const std::string prefix =
      "%1 = OpExtInstImport \"DebugInfo\"\n"
      "%3 = OpExtInst %2 %1 DebugTypePointer %4 Private ";
  const std::string input = prefix + "FlagIsPublic\n";
  const std::string expected = prefix + "FlagIsProtected|FlagIsPrivate\n";
  // First make sure it assembles correctly.
  EXPECT_THAT(
      CompiledInstructions(input),
      Eq(Concatenate(
          {MakeInstruction(SpvOpExtInstImport, {1}, MakeVector("DebugInfo")),
           MakeInstruction(SpvOpExtInst, {2, 3, 1, DebugInfoDebugTypePointer, 4,
                                          uint32_t(SpvStorageClassPrivate),
                                          DebugInfoFlagIsPublic})})))
      << input;
  // Now check the round trip through the disassembler.
  EXPECT_THAT(EncodeAndDecodeSuccessfully(input), Eq(expected)) << input;
}

INSTANTIATE_TEST_SUITE_P(
    DebugInfoDebugTypePointer, ExtInstDebugInfoRoundTripTest,
    ::testing::ValuesIn(std::vector<InstructionCase>({

        //// Use each flag independently.
        CASE_ISF(TypePointer, Private, "FlagIsProtected",
                 uint32_t(DebugInfoFlagIsProtected)),
        CASE_ISF(TypePointer, Private, "FlagIsPrivate",
                 uint32_t(DebugInfoFlagIsPrivate)),

        // FlagIsPublic is tested above.

        CASE_ISF(TypePointer, Private, "FlagIsLocal",
                 uint32_t(DebugInfoFlagIsLocal)),
        CASE_ISF(TypePointer, Private, "FlagIsDefinition",
                 uint32_t(DebugInfoFlagIsDefinition)),
        CASE_ISF(TypePointer, Private, "FlagFwdDecl",
                 uint32_t(DebugInfoFlagFwdDecl)),
        CASE_ISF(TypePointer, Private, "FlagArtificial",
                 uint32_t(DebugInfoFlagArtificial)),
        CASE_ISF(TypePointer, Private, "FlagExplicit",
                 uint32_t(DebugInfoFlagExplicit)),
        CASE_ISF(TypePointer, Private, "FlagPrototyped",
                 uint32_t(DebugInfoFlagPrototyped)),
        CASE_ISF(TypePointer, Private, "FlagObjectPointer",
                 uint32_t(DebugInfoFlagObjectPointer)),
        CASE_ISF(TypePointer, Private, "FlagStaticMember",
                 uint32_t(DebugInfoFlagStaticMember)),
        CASE_ISF(TypePointer, Private, "FlagIndirectVariable",
                 uint32_t(DebugInfoFlagIndirectVariable)),
        CASE_ISF(TypePointer, Private, "FlagLValueReference",
                 uint32_t(DebugInfoFlagLValueReference)),
        CASE_ISF(TypePointer, Private, "FlagIsOptimized",
                 uint32_t(DebugInfoFlagIsOptimized)),

        //// Use flags in combination, and try different storage classes.
        CASE_ISF(TypePointer, Function, "FlagIsProtected|FlagIsPrivate",
                 uint32_t(DebugInfoFlagIsProtected) |
                     uint32_t(DebugInfoFlagIsPrivate)),
        CASE_ISF(
            TypePointer, Workgroup,
            "FlagIsPrivate|FlagFwdDecl|FlagIndirectVariable|FlagIsOptimized",
            uint32_t(DebugInfoFlagIsPrivate) | uint32_t(DebugInfoFlagFwdDecl) |
                uint32_t(DebugInfoFlagIndirectVariable) |
                uint32_t(DebugInfoFlagIsOptimized)),

    })));

INSTANTIATE_TEST_SUITE_P(DebugInfoDebugTypeQualifier,
                         ExtInstDebugInfoRoundTripTest,
                         ::testing::ValuesIn(std::vector<InstructionCase>({
                             CASE_IE(TypeQualifier, ConstType),
                             CASE_IE(TypeQualifier, VolatileType),
                             CASE_IE(TypeQualifier, RestrictType),
                         })));

INSTANTIATE_TEST_SUITE_P(DebugInfoDebugTypeArray, ExtInstDebugInfoRoundTripTest,
                         ::testing::ValuesIn(std::vector<InstructionCase>({
                             CASE_II(TypeArray),
                             CASE_III(TypeArray),
                             CASE_IIII(TypeArray),
                             CASE_IIIII(TypeArray),
                         })));

INSTANTIATE_TEST_SUITE_P(DebugInfoDebugTypeVector,
                         ExtInstDebugInfoRoundTripTest,
                         ::testing::ValuesIn(std::vector<InstructionCase>({
                             CASE_IL(TypeVector, 2),
                             CASE_IL(TypeVector, 3),
                             CASE_IL(TypeVector, 4),
                             CASE_IL(TypeVector, 16),
                         })));

INSTANTIATE_TEST_SUITE_P(DebugInfoDebugTypedef, ExtInstDebugInfoRoundTripTest,
                         ::testing::ValuesIn(std::vector<InstructionCase>({
                             CASE_IIILLI(Typedef, 12, 13),
                             CASE_IIILLI(Typedef, 14, 99),
                         })));

INSTANTIATE_TEST_SUITE_P(DebugInfoDebugTypeFunction,
                         ExtInstDebugInfoRoundTripTest,
                         ::testing::ValuesIn(std::vector<InstructionCase>({
                             CASE_I(TypeFunction),
                             CASE_II(TypeFunction),
                             CASE_III(TypeFunction),
                             CASE_IIII(TypeFunction),
                             CASE_IIIII(TypeFunction),
                         })));

INSTANTIATE_TEST_SUITE_P(
    DebugInfoDebugTypeEnum, ExtInstDebugInfoRoundTripTest,
    ::testing::ValuesIn(std::vector<InstructionCase>({
        CASE_IIILLIIFII(
            TypeEnum, 12, 13,
            "FlagIsPrivate|FlagFwdDecl|FlagIndirectVariable|FlagIsOptimized",
            uint32_t(DebugInfoFlagIsPrivate) | uint32_t(DebugInfoFlagFwdDecl) |
                uint32_t(DebugInfoFlagIndirectVariable) |
                uint32_t(DebugInfoFlagIsOptimized)),
        CASE_IIILLIIFIIII(TypeEnum, 17, 18, "FlagStaticMember",
                          uint32_t(DebugInfoFlagStaticMember)),
        CASE_IIILLIIFIIIIII(TypeEnum, 99, 1, "FlagStaticMember",
                            uint32_t(DebugInfoFlagStaticMember)),
    })));

INSTANTIATE_TEST_SUITE_P(
    DebugInfoDebugTypeComposite, ExtInstDebugInfoRoundTripTest,
    ::testing::ValuesIn(std::vector<InstructionCase>({
        CASE_IEILLIIF(
            TypeComposite, Class, 12, 13,
            "FlagIsPrivate|FlagFwdDecl|FlagIndirectVariable|FlagIsOptimized",
            uint32_t(DebugInfoFlagIsPrivate) | uint32_t(DebugInfoFlagFwdDecl) |
                uint32_t(DebugInfoFlagIndirectVariable) |
                uint32_t(DebugInfoFlagIsOptimized)),
        // Cover all tag values: Class, Structure, Union
        CASE_IEILLIIF(TypeComposite, Class, 12, 13, "FlagIsPrivate",
                      uint32_t(DebugInfoFlagIsPrivate)),
        CASE_IEILLIIF(TypeComposite, Structure, 12, 13, "FlagIsPrivate",
                      uint32_t(DebugInfoFlagIsPrivate)),
        CASE_IEILLIIF(TypeComposite, Union, 12, 13, "FlagIsPrivate",
                      uint32_t(DebugInfoFlagIsPrivate)),
        // Now add members
        CASE_IEILLIIFI(TypeComposite, Class, 9, 10, "FlagIsPrivate",
                       uint32_t(DebugInfoFlagIsPrivate)),
        CASE_IEILLIIFII(TypeComposite, Class, 9, 10, "FlagIsPrivate",
                        uint32_t(DebugInfoFlagIsPrivate)),
        CASE_IEILLIIFIII(TypeComposite, Class, 9, 10, "FlagIsPrivate",
                         uint32_t(DebugInfoFlagIsPrivate)),
        CASE_IEILLIIFIIII(TypeComposite, Class, 9, 10, "FlagIsPrivate",
                          uint32_t(DebugInfoFlagIsPrivate)),
    })));

INSTANTIATE_TEST_SUITE_P(
    DebugInfoDebugTypeMember, ExtInstDebugInfoRoundTripTest,
    ::testing::ValuesIn(std::vector<InstructionCase>({
        CASE_IIILLIIIF(TypeMember, 12, 13, "FlagIsPrivate",
                       uint32_t(DebugInfoFlagIsPrivate)),
        CASE_IIILLIIIF(TypeMember, 99, 100, "FlagIsPrivate|FlagFwdDecl",
                       uint32_t(DebugInfoFlagIsPrivate) |
                           uint32_t(DebugInfoFlagFwdDecl)),
        // Add the optional Id argument.
        CASE_IIILLIIIFI(TypeMember, 12, 13, "FlagIsPrivate",
                        uint32_t(DebugInfoFlagIsPrivate)),
    })));

INSTANTIATE_TEST_SUITE_P(
    DebugInfoDebugTypeInheritance, ExtInstDebugInfoRoundTripTest,
    ::testing::ValuesIn(std::vector<InstructionCase>({
        CASE_IIIIF(TypeInheritance, "FlagIsPrivate",
                   uint32_t(DebugInfoFlagIsPrivate)),
        CASE_IIIIF(TypeInheritance, "FlagIsPrivate|FlagFwdDecl",
                   uint32_t(DebugInfoFlagIsPrivate) |
                       uint32_t(DebugInfoFlagFwdDecl)),
    })));

INSTANTIATE_TEST_SUITE_P(DebugInfoDebugTypePtrToMember,
                         ExtInstDebugInfoRoundTripTest,
                         ::testing::ValuesIn(std::vector<InstructionCase>({
                             CASE_II(TypePtrToMember),
                         })));

// DebugInfo 4.4 Templates

INSTANTIATE_TEST_SUITE_P(DebugInfoDebugTypeTemplate,
                         ExtInstDebugInfoRoundTripTest,
                         ::testing::ValuesIn(std::vector<InstructionCase>({
                             CASE_II(TypeTemplate),
                             CASE_III(TypeTemplate),
                             CASE_IIII(TypeTemplate),
                             CASE_IIIII(TypeTemplate),
                         })));

INSTANTIATE_TEST_SUITE_P(DebugInfoDebugTypeTemplateParameter,
                         ExtInstDebugInfoRoundTripTest,
                         ::testing::ValuesIn(std::vector<InstructionCase>({
                             CASE_IIIILL(TypeTemplateParameter, 1, 2),
                             CASE_IIIILL(TypeTemplateParameter, 99, 102),
                             CASE_IIIILL(TypeTemplateParameter, 10, 7),
                         })));

INSTANTIATE_TEST_SUITE_P(DebugInfoDebugTypeTemplateTemplateParameter,
                         ExtInstDebugInfoRoundTripTest,
                         ::testing::ValuesIn(std::vector<InstructionCase>({
                             CASE_IIILL(TypeTemplateTemplateParameter, 1, 2),
                             CASE_IIILL(TypeTemplateTemplateParameter, 99, 102),
                             CASE_IIILL(TypeTemplateTemplateParameter, 10, 7),
                         })));

INSTANTIATE_TEST_SUITE_P(DebugInfoDebugTypeTemplateParameterPack,
                         ExtInstDebugInfoRoundTripTest,
                         ::testing::ValuesIn(std::vector<InstructionCase>({
                             CASE_IILLI(TypeTemplateParameterPack, 1, 2),
                             CASE_IILLII(TypeTemplateParameterPack, 99, 102),
                             CASE_IILLIII(TypeTemplateParameterPack, 10, 7),
                             CASE_IILLIIII(TypeTemplateParameterPack, 10, 7),
                         })));

// DebugInfo 4.5 Global Variables

INSTANTIATE_TEST_SUITE_P(
    DebugInfoDebugGlobalVariable, ExtInstDebugInfoRoundTripTest,
    ::testing::ValuesIn(std::vector<InstructionCase>({
        CASE_IIILLIIIF(GlobalVariable, 1, 2, "FlagIsOptimized",
                       uint32_t(DebugInfoFlagIsOptimized)),
        CASE_IIILLIIIF(GlobalVariable, 42, 43, "FlagIsOptimized",
                       uint32_t(DebugInfoFlagIsOptimized)),
        CASE_IIILLIIIFI(GlobalVariable, 1, 2, "FlagIsOptimized",
                        uint32_t(DebugInfoFlagIsOptimized)),
        CASE_IIILLIIIFI(GlobalVariable, 42, 43, "FlagIsOptimized",
                        uint32_t(DebugInfoFlagIsOptimized)),
    })));

// DebugInfo 4.6 Functions

INSTANTIATE_TEST_SUITE_P(
    DebugInfoDebugFunctionDeclaration, ExtInstDebugInfoRoundTripTest,
    ::testing::ValuesIn(std::vector<InstructionCase>({
        CASE_IIILLIIF(FunctionDeclaration, 1, 2, "FlagIsOptimized",
                      uint32_t(DebugInfoFlagIsOptimized)),
        CASE_IIILLIIF(FunctionDeclaration, 42, 43, "FlagFwdDecl",
                      uint32_t(DebugInfoFlagFwdDecl)),
    })));

INSTANTIATE_TEST_SUITE_P(
    DebugInfoDebugFunction, ExtInstDebugInfoRoundTripTest,
    ::testing::ValuesIn(std::vector<InstructionCase>({
        CASE_IIILLIIFLI(Function, 1, 2, "FlagIsOptimized",
                        uint32_t(DebugInfoFlagIsOptimized), 3),
        CASE_IIILLIIFLI(Function, 42, 43, "FlagFwdDecl",
                        uint32_t(DebugInfoFlagFwdDecl), 44),
        // Add the optional declaration Id.
        CASE_IIILLIIFLII(Function, 1, 2, "FlagIsOptimized",
                         uint32_t(DebugInfoFlagIsOptimized), 3),
        CASE_IIILLIIFLII(Function, 42, 43, "FlagFwdDecl",
                         uint32_t(DebugInfoFlagFwdDecl), 44),
    })));

// DebugInfo 4.7 Local Information

INSTANTIATE_TEST_SUITE_P(DebugInfoDebugLexicalBlock,
                         ExtInstDebugInfoRoundTripTest,
                         ::testing::ValuesIn(std::vector<InstructionCase>({
                             CASE_ILLII(LexicalBlock, 1, 2),
                             CASE_ILLII(LexicalBlock, 42, 43),
                         })));

INSTANTIATE_TEST_SUITE_P(DebugInfoDebugLexicalBlockDiscriminator,
                         ExtInstDebugInfoRoundTripTest,
                         ::testing::ValuesIn(std::vector<InstructionCase>({
                             CASE_ILI(LexicalBlockDiscriminator, 1),
                             CASE_ILI(LexicalBlockDiscriminator, 42),
                         })));

INSTANTIATE_TEST_SUITE_P(DebugInfoDebugScope, ExtInstDebugInfoRoundTripTest,
                         ::testing::ValuesIn(std::vector<InstructionCase>({
                             CASE_I(Scope),
                             CASE_II(Scope),
                         })));

INSTANTIATE_TEST_SUITE_P(DebugInfoDebugNoScope, ExtInstDebugInfoRoundTripTest,
                         ::testing::ValuesIn(std::vector<InstructionCase>({
                             CASE_0(NoScope),
                         })));

INSTANTIATE_TEST_SUITE_P(DebugInfoDebugInlinedAt, ExtInstDebugInfoRoundTripTest,
                         ::testing::ValuesIn(std::vector<InstructionCase>({
                             CASE_LII(InlinedAt, 1),
                             CASE_LII(InlinedAt, 42),
                         })));

// DebugInfo 4.8 Local Variables

INSTANTIATE_TEST_SUITE_P(DebugInfoDebugLocalVariable,
                         ExtInstDebugInfoRoundTripTest,
                         ::testing::ValuesIn(std::vector<InstructionCase>({
                             CASE_IIILLI(LocalVariable, 1, 2),
                             CASE_IIILLI(LocalVariable, 42, 43),
                             CASE_IIILLIL(LocalVariable, 1, 2, 3),
                             CASE_IIILLIL(LocalVariable, 42, 43, 44),
                         })));

INSTANTIATE_TEST_SUITE_P(DebugInfoDebugInlinedVariable,
                         ExtInstDebugInfoRoundTripTest,
                         ::testing::ValuesIn(std::vector<InstructionCase>({
                             CASE_II(InlinedVariable),
                         })));

INSTANTIATE_TEST_SUITE_P(DebugInfoDebugDebugDeclare,
                         ExtInstDebugInfoRoundTripTest,
                         ::testing::ValuesIn(std::vector<InstructionCase>({
                             CASE_III(Declare),
                         })));

INSTANTIATE_TEST_SUITE_P(
    DebugInfoDebugDebugValue, ExtInstDebugInfoRoundTripTest,
    ::testing::ValuesIn(std::vector<InstructionCase>({
        CASE_III(Value),
        CASE_IIII(Value),
        CASE_IIIII(Value),
        CASE_IIIIII(Value),
        // Test up to 4 id parameters. We can always try more.
        CASE_IIIIIII(Value),
    })));

INSTANTIATE_TEST_SUITE_P(DebugInfoDebugDebugOperation,
                         ExtInstDebugInfoRoundTripTest,
                         ::testing::ValuesIn(std::vector<InstructionCase>({
                             CASE_E(Operation, Deref),
                             CASE_E(Operation, Plus),
                             CASE_E(Operation, Minus),
                             CASE_EL(Operation, PlusUconst, 1),
                             CASE_EL(Operation, PlusUconst, 42),
                             CASE_ELL(Operation, BitPiece, 1, 2),
                             CASE_ELL(Operation, BitPiece, 4, 5),
                             CASE_E(Operation, Swap),
                             CASE_E(Operation, Xderef),
                             CASE_E(Operation, StackValue),
                             CASE_EL(Operation, Constu, 1),
                             CASE_EL(Operation, Constu, 42),
                         })));

INSTANTIATE_TEST_SUITE_P(DebugInfoDebugDebugExpression,
                         ExtInstDebugInfoRoundTripTest,
                         ::testing::ValuesIn(std::vector<InstructionCase>({
                             CASE_0(Expression),
                             CASE_I(Expression),
                             CASE_II(Expression),
                             CASE_III(Expression),
                             CASE_IIII(Expression),
                             CASE_IIIII(Expression),
                             CASE_IIIIII(Expression),
                             CASE_IIIIIII(Expression),
                         })));

// DebugInfo 4.9 Macros

INSTANTIATE_TEST_SUITE_P(DebugInfoDebugMacroDef, ExtInstDebugInfoRoundTripTest,
                         ::testing::ValuesIn(std::vector<InstructionCase>({
                             CASE_ILI(MacroDef, 1),
                             CASE_ILI(MacroDef, 42),
                             CASE_ILII(MacroDef, 1),
                             CASE_ILII(MacroDef, 42),
                         })));

INSTANTIATE_TEST_SUITE_P(DebugInfoDebugMacroUndef,
                         ExtInstDebugInfoRoundTripTest,
                         ::testing::ValuesIn(std::vector<InstructionCase>({
                             CASE_ILI(MacroUndef, 1),
                             CASE_ILI(MacroUndef, 42),
                         })));

#undef CASE_0
#undef CASE_ILL
#undef CASE_IL
#undef CASE_I
#undef CASE_II
#undef CASE_III
#undef CASE_IIII
#undef CASE_IIIII
#undef CASE_IIIIII
#undef CASE_IIIIIII
#undef CASE_IIILLI
#undef CASE_IIILLIL
#undef CASE_IE
#undef CASE_IIE
#undef CASE_ISF
#undef CASE_LII
#undef CASE_ILI
#undef CASE_ILII
#undef CASE_ILLII
#undef CASE_IIILLIIF
#undef CASE_IIILLIIFII
#undef CASE_IIILLIIFIIII
#undef CASE_IIILLIIFIIIIII
#undef CASE_IEILLIIF
#undef CASE_IEILLIIFI
#undef CASE_IEILLIIFII
#undef CASE_IEILLIIFIII
#undef CASE_IEILLIIFIIII
#undef CASE_IIILLIIIF
#undef CASE_IIILLIIIFI
#undef CASE_IIIIF
#undef CASE_IIILL
#undef CASE_IIIILL
#undef CASE_IILLI
#undef CASE_IILLII
#undef CASE_IILLIII
#undef CASE_IILLIIII
#undef CASE_IIILLIIFLI
#undef CASE_IIILLIIFLII
#undef CASE_E
#undef CASE_EL
#undef CASE_ELL

}  // namespace
}  // namespace spvtools
