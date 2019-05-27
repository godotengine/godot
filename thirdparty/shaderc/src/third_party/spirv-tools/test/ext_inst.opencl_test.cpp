// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#include "gmock/gmock.h"
#include "source/latest_version_opencl_std_header.h"
#include "test/test_fixture.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace {

using spvtest::Concatenate;
using spvtest::MakeInstruction;
using spvtest::MakeVector;
using spvtest::TextToBinaryTest;
using testing::Eq;

struct InstructionCase {
  uint32_t opcode;
  std::string name;
  std::string operands;
  std::vector<uint32_t> expected_operands;
};

using ExtInstOpenCLStdRoundTripTest =
    spvtest::TextToBinaryTestBase<::testing::TestWithParam<InstructionCase>>;

TEST_P(ExtInstOpenCLStdRoundTripTest, ParameterizedExtInst) {
  // This example should not validate.
  const std::string input =
      "%1 = OpExtInstImport \"OpenCL.std\"\n"
      "%3 = OpExtInst %2 %1 " +
      GetParam().name + " " + GetParam().operands + "\n";
  // First make sure it assembles correctly.
  EXPECT_THAT(
      CompiledInstructions(input),
      Eq(Concatenate(
          {MakeInstruction(SpvOpExtInstImport, {1}, MakeVector("OpenCL.std")),
           MakeInstruction(SpvOpExtInst, {2, 3, 1, GetParam().opcode},
                           GetParam().expected_operands)})))
      << input;
  // Now check the round trip through the disassembler.
  EXPECT_THAT(EncodeAndDecodeSuccessfully(input), input) << input;
}

#define CASE1(Enum, Name)                                      \
  {                                                            \
    uint32_t(OpenCLLIB::Entrypoints::Enum), #Name, "%4", { 4 } \
  }
#define CASE2(Enum, Name)                                            \
  {                                                                  \
    uint32_t(OpenCLLIB::Entrypoints::Enum), #Name, "%4 %5", { 4, 5 } \
  }
#define CASE3(Enum, Name)                                                  \
  {                                                                        \
    uint32_t(OpenCLLIB::Entrypoints::Enum), #Name, "%4 %5 %6", { 4, 5, 6 } \
  }
#define CASE4(Enum, Name)                                           \
  {                                                                 \
    uint32_t(OpenCLLIB::Entrypoints::Enum), #Name, "%4 %5 %6 %7", { \
      4, 5, 6, 7                                                    \
    }                                                               \
  }
#define CASE2Lit(Enum, Name, LiteralNumber)                                   \
  {                                                                           \
    uint32_t(OpenCLLIB::Entrypoints::Enum), #Name, "%4 %5 " #LiteralNumber, { \
      4, 5, LiteralNumber                                                     \
    }                                                                         \
  }
#define CASE3Round(Enum, Name, Mode)                                    \
  {                                                                     \
    uint32_t(OpenCLLIB::Entrypoints::Enum), #Name, "%4 %5 %6 " #Mode, { \
      4, 5, 6, uint32_t(SpvFPRoundingMode##Mode)                        \
    }                                                                   \
  }

// clang-format off
// OpenCL.std: 2.1 Math extended instructions
INSTANTIATE_TEST_SUITE_P(
    OpenCLMath, ExtInstOpenCLStdRoundTripTest,
    ::testing::ValuesIn(std::vector<InstructionCase>({
        // We are only testing the correctness of encoding and decoding here.
        // Semantic correctness should be the responsibility of validator.
        CASE1(Acos, acos), // enum value 0
        CASE1(Acosh, acosh),
        CASE1(Acospi, acospi),
        CASE1(Asin, asin),
        CASE1(Asinh, asinh),
        CASE1(Asinh, asinh),
        CASE1(Asinpi, asinpi),
        CASE1(Atan, atan),
        CASE2(Atan2, atan2),
        CASE1(Atanh, atanh),
        CASE1(Atanpi, atanpi),
        CASE2(Atan2pi, atan2pi),
        CASE1(Cbrt, cbrt),
        CASE1(Ceil, ceil),
        CASE1(Ceil, ceil),
        CASE2(Copysign, copysign),
        CASE1(Cos, cos),
        CASE1(Cosh, cosh),
        CASE1(Cospi, cospi),
        CASE1(Erfc, erfc),
        CASE1(Erf, erf),
        CASE1(Exp, exp),
        CASE1(Exp2, exp2),
        CASE1(Exp10, exp10),
        CASE1(Expm1, expm1),
        CASE1(Fabs, fabs),
        CASE2(Fdim, fdim),
        CASE1(Floor, floor),
        CASE3(Fma, fma),
        CASE2(Fmax, fmax),
        CASE2(Fmin, fmin),
        CASE2(Fmod, fmod),
        CASE2(Fract, fract),
        CASE2(Frexp, frexp),
        CASE2(Hypot, hypot),
        CASE1(Ilogb, ilogb),
        CASE2(Ldexp, ldexp),
        CASE1(Lgamma, lgamma),
        CASE2(Lgamma_r, lgamma_r),
        CASE1(Log, log),
        CASE1(Log2, log2),
        CASE1(Log10, log10),
        CASE1(Log1p, log1p),
        CASE3(Mad, mad),
        CASE2(Maxmag, maxmag),
        CASE2(Minmag, minmag),
        CASE2(Modf, modf),
        CASE1(Nan, nan),
        CASE2(Nextafter, nextafter),
        CASE2(Pow, pow),
        CASE2(Pown, pown),
        CASE2(Powr, powr),
        CASE2(Remainder, remainder),
        CASE3(Remquo, remquo),
        CASE1(Rint, rint),
        CASE2(Rootn, rootn),
        CASE1(Round, round),
        CASE1(Rsqrt, rsqrt),
        CASE1(Sin, sin),
        CASE2(Sincos, sincos),
        CASE1(Sinh, sinh),
        CASE1(Sinpi, sinpi),
        CASE1(Sqrt, sqrt),
        CASE1(Tan, tan),
        CASE1(Tanh, tanh),
        CASE1(Tanpi, tanpi),
        CASE1(Tgamma, tgamma),
        CASE1(Trunc, trunc),
        CASE1(Half_cos, half_cos),
        CASE2(Half_divide, half_divide),
        CASE1(Half_exp, half_exp),
        CASE1(Half_exp2, half_exp2),
        CASE1(Half_exp10, half_exp10),
        CASE1(Half_log, half_log),
        CASE1(Half_log2, half_log2),
        CASE1(Half_log10, half_log10),
        CASE2(Half_powr, half_powr),
        CASE1(Half_recip, half_recip),
        CASE1(Half_rsqrt, half_rsqrt),
        CASE1(Half_sin, half_sin),
        CASE1(Half_sqrt, half_sqrt),
        CASE1(Half_tan, half_tan),
        CASE1(Native_cos, native_cos),
        CASE2(Native_divide, native_divide),
        CASE1(Native_exp, native_exp),
        CASE1(Native_exp2, native_exp2),
        CASE1(Native_exp10, native_exp10),
        CASE1(Native_log, native_log),
        CASE1(Native_log10, native_log10),
        CASE2(Native_powr, native_powr),
        CASE1(Native_recip, native_recip),
        CASE1(Native_rsqrt, native_rsqrt),
        CASE1(Native_sin, native_sin),
        CASE1(Native_sqrt, native_sqrt),
        CASE1(Native_tan, native_tan), // enum value 94
    })));

// OpenCL.std: 2.1 Integer instructions
INSTANTIATE_TEST_SUITE_P(
    OpenCLInteger, ExtInstOpenCLStdRoundTripTest,
    ::testing::ValuesIn(std::vector<InstructionCase>({
        CASE1(SAbs, s_abs), // enum value 141
        CASE2(SAbs_diff, s_abs_diff),
        CASE2(SAdd_sat, s_add_sat),
        CASE2(UAdd_sat, u_add_sat),
        CASE2(SHadd, s_hadd),
        CASE2(UHadd, u_hadd),
        CASE2(SRhadd, s_rhadd),
        CASE2(SRhadd, s_rhadd),
        CASE3(SClamp, s_clamp),
        CASE3(UClamp, u_clamp),
        CASE1(Clz, clz),
        CASE1(Ctz, ctz),
        CASE3(SMad_hi, s_mad_hi),
        CASE3(UMad_sat, u_mad_sat),
        CASE3(SMad_sat, s_mad_sat),
        CASE2(SMax, s_max),
        CASE2(UMax, u_max),
        CASE2(SMin, s_min),
        CASE2(UMin, u_min),
        CASE2(SMul_hi, s_mul_hi),
        CASE2(Rotate, rotate),
        CASE2(SSub_sat, s_sub_sat),
        CASE2(USub_sat, u_sub_sat),
        CASE2(U_Upsample, u_upsample),
        CASE2(S_Upsample, s_upsample),
        CASE1(Popcount, popcount),
        CASE3(SMad24, s_mad24),
        CASE3(UMad24, u_mad24),
        CASE2(SMul24, s_mul24),
        CASE2(UMul24, u_mul24), // enum value 170
        CASE1(UAbs, u_abs), // enum value 201
        CASE2(UAbs_diff, u_abs_diff),
        CASE2(UMul_hi, u_mul_hi),
        CASE3(UMad_hi, u_mad_hi), // enum value 204
    })));

// OpenCL.std: 2.3 Common instrucitons
INSTANTIATE_TEST_SUITE_P(
    OpenCLCommon, ExtInstOpenCLStdRoundTripTest,
    ::testing::ValuesIn(std::vector<InstructionCase>({
        CASE3(FClamp, fclamp), // enum value 95
        CASE1(Degrees, degrees),
        CASE2(FMax_common, fmax_common),
        CASE2(FMin_common, fmin_common),
        CASE3(Mix, mix),
        CASE1(Radians, radians),
        CASE2(Step, step),
        CASE3(Smoothstep, smoothstep),
        CASE1(Sign, sign), // enum value 103
    })));

// OpenCL.std: 2.4 Geometric instructions
INSTANTIATE_TEST_SUITE_P(
    OpenCLGeometric, ExtInstOpenCLStdRoundTripTest,
    ::testing::ValuesIn(std::vector<InstructionCase>({
        CASE2(Cross, cross), // enum value 104
        CASE2(Distance, distance),
        CASE1(Length, length),
        CASE1(Normalize, normalize),
        CASE2(Fast_distance, fast_distance),
        CASE1(Fast_length, fast_length),
        CASE1(Fast_normalize, fast_normalize), // enum value 110
    })));

// OpenCL.std: 2.5 Relational instructions
INSTANTIATE_TEST_SUITE_P(
    OpenCLRelational, ExtInstOpenCLStdRoundTripTest,
    ::testing::ValuesIn(std::vector<InstructionCase>({
        CASE3(Bitselect, bitselect), // enum value 186
        CASE3(Select, select), // enum value 187
    })));

// OpenCL.std: 2.6 Vector data load and store instructions
INSTANTIATE_TEST_SUITE_P(
    OpenCLVectorLoadStore, ExtInstOpenCLStdRoundTripTest,
    ::testing::ValuesIn(std::vector<InstructionCase>({
        // The last argument to Vloadn must be one of 2, 3, 4, 8, 16.
        CASE2Lit(Vloadn, vloadn, 2),
        CASE2Lit(Vloadn, vloadn, 3),
        CASE2Lit(Vloadn, vloadn, 4),
        CASE2Lit(Vloadn, vloadn, 8),
        CASE2Lit(Vloadn, vloadn, 16),
        CASE3(Vstoren, vstoren),
        CASE2(Vload_half, vload_half),
        CASE2Lit(Vload_halfn, vload_halfn, 2),
        CASE2Lit(Vload_halfn, vload_halfn, 3),
        CASE2Lit(Vload_halfn, vload_halfn, 4),
        CASE2Lit(Vload_halfn, vload_halfn, 8),
        CASE2Lit(Vload_halfn, vload_halfn, 16),
        CASE3(Vstore_half, vstore_half),
        // Try all the rounding modes.
        CASE3Round(Vstore_half_r, vstore_half_r, RTE),
        CASE3Round(Vstore_half_r, vstore_half_r, RTZ),
        CASE3Round(Vstore_half_r, vstore_half_r, RTP),
        CASE3Round(Vstore_half_r, vstore_half_r, RTN),
        CASE3(Vstore_halfn, vstore_halfn),
        CASE3Round(Vstore_halfn_r, vstore_halfn_r, RTE),
        CASE3Round(Vstore_halfn_r, vstore_halfn_r, RTZ),
        CASE3Round(Vstore_halfn_r, vstore_halfn_r, RTP),
        CASE3Round(Vstore_halfn_r, vstore_halfn_r, RTN),
        CASE2Lit(Vloada_halfn, vloada_halfn, 2),
        CASE2Lit(Vloada_halfn, vloada_halfn, 3),
        CASE2Lit(Vloada_halfn, vloada_halfn, 4),
        CASE2Lit(Vloada_halfn, vloada_halfn, 8),
        CASE2Lit(Vloada_halfn, vloada_halfn, 16),
        CASE3(Vstorea_halfn, vstorea_halfn),
        CASE3Round(Vstorea_halfn_r, vstorea_halfn_r, RTE),
        CASE3Round(Vstorea_halfn_r, vstorea_halfn_r, RTZ),
        CASE3Round(Vstorea_halfn_r, vstorea_halfn_r, RTP),
        CASE3Round(Vstorea_halfn_r, vstorea_halfn_r, RTN),
    })));

// OpenCL.std: 2.7 Miscellaneous vector instructions
INSTANTIATE_TEST_SUITE_P(
    OpenCLMiscellaneousVector, ExtInstOpenCLStdRoundTripTest,
    ::testing::ValuesIn(std::vector<InstructionCase>({
        CASE2(Shuffle, shuffle),
        CASE3(Shuffle2, shuffle2),
    })));

// OpenCL.std: 2.8 Miscellaneous instructions

#define PREFIX uint32_t(OpenCLLIB::Entrypoints::Printf), "printf"
INSTANTIATE_TEST_SUITE_P(
    OpenCLMiscPrintf, ExtInstOpenCLStdRoundTripTest,
    ::testing::ValuesIn(std::vector<InstructionCase>({
      // Printf is interesting because it takes a variable number of arguments.
      // Start with zero optional arguments.
      {PREFIX, "%4", {4}},
      {PREFIX, "%4 %5", {4, 5}},
      {PREFIX, "%4 %5 %6", {4, 5, 6}},
      {PREFIX, "%4 %5 %6 %7", {4, 5, 6, 7}},
      {PREFIX, "%4 %5 %6 %7 %8", {4, 5, 6, 7, 8}},
      {PREFIX, "%4 %5 %6 %7 %8 %9", {4, 5, 6, 7, 8, 9}},
      {PREFIX, "%4 %5 %6 %7 %8 %9 %10", {4, 5, 6, 7, 8, 9, 10}},
      {PREFIX, "%4 %5 %6 %7 %8 %9 %10 %11", {4, 5, 6, 7, 8, 9, 10, 11}},
      {PREFIX, "%4 %5 %6 %7 %8 %9 %10 %11 %12",
        {4, 5, 6, 7, 8, 9, 10, 11, 12}},
      {PREFIX, "%4 %5 %6 %7 %8 %9 %10 %11 %12 %13",
        {4, 5, 6, 7, 8, 9, 10, 11, 12, 13}},
      {PREFIX, "%4 %5 %6 %7 %8 %9 %10 %11 %12 %13 %14",
        {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}},
    })));
#undef PREFIX

INSTANTIATE_TEST_SUITE_P(
    OpenCLMiscPrefetch, ExtInstOpenCLStdRoundTripTest,
    ::testing::ValuesIn(std::vector<InstructionCase>({
        CASE2(Prefetch, prefetch),
    })));

// OpenCL.std: 2.9.1 Image encoding
// No new instructions defined in this section.

// OpenCL.std: 2.9.2 Sampler encoding
// No new instructions defined in this section.

// OpenCL.std: 2.9.3 Image read
// No new instructions defined in this section.
// Use core instruction OpImageSampleExplicitLod instead.

// OpenCL.std: 2.9.4 Image write
// No new instructions defined in this section.

// clang-format on

#undef CASE1
#undef CASE2
#undef CASE3
#undef CASE4
#undef CASE2Lit
#undef CASE3Round

}  // namespace
}  // namespace spvtools
