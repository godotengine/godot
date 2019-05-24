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

// Assembler tests for instructions in the "Image Instructions" section of
// the SPIR-V spec.

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "test/test_fixture.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace {

using spvtest::MakeInstruction;
using spvtest::TextToBinaryTest;
using ::testing::Eq;

// An example case for a mask value with operands.
struct ImageOperandsCase {
  std::string image_operands;
  // The expected mask, followed by its operands.
  std::vector<uint32_t> expected_mask_and_operands;
};

// Test all kinds of image operands.

using ImageOperandsTest =
    spvtest::TextToBinaryTestBase<::testing::TestWithParam<ImageOperandsCase>>;

TEST_P(ImageOperandsTest, Sample) {
  const std::string input =
      "%2 = OpImageFetch %1 %3 %4" + GetParam().image_operands + "\n";
  EXPECT_THAT(CompiledInstructions(input),
              Eq(MakeInstruction(SpvOpImageFetch, {1, 2, 3, 4},
                                 GetParam().expected_mask_and_operands)));
}

#define MASK(NAME) SpvImageOperands##NAME##Mask
INSTANTIATE_TEST_SUITE_P(
    TextToBinaryImageOperandsAny, ImageOperandsTest,
    ::testing::ValuesIn(std::vector<ImageOperandsCase>{
        // TODO(dneto): Rev32 adds many more values, and rearranges their
        // values.
        // Image operands are optional.
        {"", {}},
        // Test each kind, alone.
        {" Bias %5", {MASK(Bias), 5}},
        {" Lod %5", {MASK(Lod), 5}},
        {" Grad %5 %6", {MASK(Grad), 5, 6}},
        {" ConstOffset %5", {MASK(ConstOffset), 5}},
        {" Offset %5", {MASK(Offset), 5}},
        {" ConstOffsets %5", {MASK(ConstOffsets), 5}},
        {" Sample %5", {MASK(Sample), 5}},
        {" MinLod %5", {MASK(MinLod), 5}},
    }));
#undef MASK
#define MASK(NAME) static_cast<uint32_t>(SpvImageOperands##NAME##Mask)
INSTANTIATE_TEST_SUITE_P(
    TextToBinaryImageOperandsCombination, ImageOperandsTest,
    ::testing::ValuesIn(std::vector<ImageOperandsCase>{
        // TODO(dneto): Rev32 adds many more values, and rearranges their
        // values.
        // Test adjacent pairs, so we can easily debug the values when it fails.
        {" Bias|Lod %5 %6", {MASK(Bias) | MASK(Lod), 5, 6}},
        {" Lod|Grad %5 %6 %7", {MASK(Lod) | MASK(Grad), 5, 6, 7}},
        {" Grad|ConstOffset %5 %6 %7",
         {MASK(Grad) | MASK(ConstOffset), 5, 6, 7}},
        {" ConstOffset|Offset %5 %6", {MASK(ConstOffset) | MASK(Offset), 5, 6}},
        {" Offset|ConstOffsets %5 %6",
         {MASK(Offset) | MASK(ConstOffsets), 5, 6}},
        {" ConstOffsets|Sample %5 %6",
         {MASK(ConstOffsets) | MASK(Sample), 5, 6}},
        // Test all masks together.
        {" Bias|Lod|Grad|ConstOffset|Offset|ConstOffsets|Sample"
         " %5 %6 %7 %8 %9 %10 %11 %12",
         {MASK(Bias) | MASK(Lod) | MASK(Grad) | MASK(ConstOffset) |
              MASK(Offset) | MASK(ConstOffsets) | MASK(Sample),
          5, 6, 7, 8, 9, 10, 11, 12}},
        // The same, but with mask value names reversed.
        {" Sample|ConstOffsets|Offset|ConstOffset|Grad|Lod|Bias"
         " %5 %6 %7 %8 %9 %10 %11 %12",
         {MASK(Bias) | MASK(Lod) | MASK(Grad) | MASK(ConstOffset) |
              MASK(Offset) | MASK(ConstOffsets) | MASK(Sample),
          5, 6, 7, 8, 9, 10, 11, 12}}}));
#undef MASK

TEST_F(ImageOperandsTest, WrongOperand) {
  EXPECT_THAT(CompileFailure("%r = OpImageFetch %t %i %c xxyyzz"),
              Eq("Invalid image operand 'xxyyzz'."));
}

// Test OpImage

using OpImageTest = TextToBinaryTest;

TEST_F(OpImageTest, Valid) {
  const std::string input = "%2 = OpImage %1 %3\n";
  EXPECT_THAT(CompiledInstructions(input),
              Eq(MakeInstruction(SpvOpImage, {1, 2, 3})));

  // Test the disassembler.
  EXPECT_THAT(EncodeAndDecodeSuccessfully(input), input);
}

TEST_F(OpImageTest, InvalidTypeOperand) {
  EXPECT_THAT(CompileFailure("%2 = OpImage 42"),
              Eq("Expected id to start with %."));
}

TEST_F(OpImageTest, MissingSampledImageOperand) {
  EXPECT_THAT(CompileFailure("%2 = OpImage %1"),
              Eq("Expected operand, found end of stream."));
}

TEST_F(OpImageTest, InvalidSampledImageOperand) {
  EXPECT_THAT(CompileFailure("%2 = OpImage %1 1000"),
              Eq("Expected id to start with %."));
}

TEST_F(OpImageTest, TooManyOperands) {
  // We should improve this message, to say what instruction we're trying to
  // parse.
  EXPECT_THAT(CompileFailure("%2 = OpImage %1 %3 %4"),  // an Id
              Eq("Expected '=', found end of stream."));

  EXPECT_THAT(CompileFailure("%2 = OpImage %1 %3 99"),  // a number
              Eq("Expected <opcode> or <result-id> at the beginning of an "
                 "instruction, found '99'."));
  EXPECT_THAT(CompileFailure("%2 = OpImage %1 %3 \"abc\""),  // a string
              Eq("Expected <opcode> or <result-id> at the beginning of an "
                 "instruction, found '\"abc\"'."));
}

// Test OpImageSparseRead

using OpImageSparseReadTest = TextToBinaryTest;

TEST_F(OpImageSparseReadTest, OnlyRequiredOperands) {
  const std::string input = "%2 = OpImageSparseRead %1 %3 %4\n";
  EXPECT_THAT(CompiledInstructions(input),
              Eq(MakeInstruction(SpvOpImageSparseRead, {1, 2, 3, 4})));
  // Test the disassembler.
  EXPECT_THAT(EncodeAndDecodeSuccessfully(input), input);
}

// Test all kinds of image operands on OpImageSparseRead

using ImageSparseReadImageOperandsTest =
    spvtest::TextToBinaryTestBase<::testing::TestWithParam<ImageOperandsCase>>;

TEST_P(ImageSparseReadImageOperandsTest, Sample) {
  const std::string input =
      "%2 = OpImageSparseRead %1 %3 %4" + GetParam().image_operands + "\n";
  EXPECT_THAT(CompiledInstructions(input),
              Eq(MakeInstruction(SpvOpImageSparseRead, {1, 2, 3, 4},
                                 GetParam().expected_mask_and_operands)));
  // Test the disassembler.
  EXPECT_THAT(EncodeAndDecodeSuccessfully(input), input);
}

#define MASK(NAME) SpvImageOperands##NAME##Mask
INSTANTIATE_TEST_SUITE_P(ImageSparseReadImageOperandsAny,
                         ImageSparseReadImageOperandsTest,
                         ::testing::ValuesIn(std::vector<ImageOperandsCase>{
                             // Image operands are optional.
                             {"", {}},
                             // Test each kind, alone.
                             {" Bias %5", {MASK(Bias), 5}},
                             {" Lod %5", {MASK(Lod), 5}},
                             {" Grad %5 %6", {MASK(Grad), 5, 6}},
                             {" ConstOffset %5", {MASK(ConstOffset), 5}},
                             {" Offset %5", {MASK(Offset), 5}},
                             {" ConstOffsets %5", {MASK(ConstOffsets), 5}},
                             {" Sample %5", {MASK(Sample), 5}},
                             {" MinLod %5", {MASK(MinLod), 5}},
                         }));
#undef MASK
#define MASK(NAME) static_cast<uint32_t>(SpvImageOperands##NAME##Mask)
INSTANTIATE_TEST_SUITE_P(
    ImageSparseReadImageOperandsCombination, ImageSparseReadImageOperandsTest,
    ::testing::ValuesIn(std::vector<ImageOperandsCase>{
        // values.
        // Test adjacent pairs, so we can easily debug the values when it fails.
        {" Bias|Lod %5 %6", {MASK(Bias) | MASK(Lod), 5, 6}},
        {" Lod|Grad %5 %6 %7", {MASK(Lod) | MASK(Grad), 5, 6, 7}},
        {" Grad|ConstOffset %5 %6 %7",
         {MASK(Grad) | MASK(ConstOffset), 5, 6, 7}},
        {" ConstOffset|Offset %5 %6", {MASK(ConstOffset) | MASK(Offset), 5, 6}},
        {" Offset|ConstOffsets %5 %6",
         {MASK(Offset) | MASK(ConstOffsets), 5, 6}},
        {" ConstOffsets|Sample %5 %6",
         {MASK(ConstOffsets) | MASK(Sample), 5, 6}},
        // Test all masks together.
        {" Bias|Lod|Grad|ConstOffset|Offset|ConstOffsets|Sample"
         " %5 %6 %7 %8 %9 %10 %11 %12",
         {MASK(Bias) | MASK(Lod) | MASK(Grad) | MASK(ConstOffset) |
              MASK(Offset) | MASK(ConstOffsets) | MASK(Sample),
          5, 6, 7, 8, 9, 10, 11, 12}},
        // Don't try the masks reversed, since this is a round trip test,
        // and the disassembler will sort them.
    }));
#undef MASK

TEST_F(OpImageSparseReadTest, InvalidTypeOperand) {
  EXPECT_THAT(CompileFailure("%2 = OpImageSparseRead 42"),
              Eq("Expected id to start with %."));
}

TEST_F(OpImageSparseReadTest, MissingImageOperand) {
  EXPECT_THAT(CompileFailure("%2 = OpImageSparseRead %1"),
              Eq("Expected operand, found end of stream."));
}

TEST_F(OpImageSparseReadTest, InvalidImageOperand) {
  EXPECT_THAT(CompileFailure("%2 = OpImageSparseRead %1 1000"),
              Eq("Expected id to start with %."));
}

TEST_F(OpImageSparseReadTest, MissingCoordinateOperand) {
  EXPECT_THAT(CompileFailure("%2 = OpImageSparseRead %1 %2"),
              Eq("Expected operand, found end of stream."));
}

TEST_F(OpImageSparseReadTest, InvalidCoordinateOperand) {
  EXPECT_THAT(CompileFailure("%2 = OpImageSparseRead %1 %2 1000"),
              Eq("Expected id to start with %."));
}

// TODO(dneto): OpSampledImage
// TODO(dneto): OpImageSampleImplicitLod
// TODO(dneto): OpImageSampleExplicitLod
// TODO(dneto): OpImageSampleDrefImplicitLod
// TODO(dneto): OpImageSampleDrefExplicitLod
// TODO(dneto): OpImageSampleProjImplicitLod
// TODO(dneto): OpImageSampleProjExplicitLod
// TODO(dneto): OpImageSampleProjDrefImplicitLod
// TODO(dneto): OpImageSampleProjDrefExplicitLod
// TODO(dneto): OpImageGather
// TODO(dneto): OpImageDrefGather
// TODO(dneto): OpImageRead
// TODO(dneto): OpImageWrite
// TODO(dneto): OpImageQueryFormat
// TODO(dneto): OpImageQueryOrder
// TODO(dneto): OpImageQuerySizeLod
// TODO(dneto): OpImageQuerySize
// TODO(dneto): OpImageQueryLod
// TODO(dneto): OpImageQueryLevels
// TODO(dneto): OpImageQuerySamples
// TODO(dneto): OpImageSparseSampleImplicitLod
// TODO(dneto): OpImageSparseSampleExplicitLod
// TODO(dneto): OpImageSparseSampleDrefImplicitLod
// TODO(dneto): OpImageSparseSampleDrefExplicitLod
// TODO(dneto): OpImageSparseSampleProjImplicitLod
// TODO(dneto): OpImageSparseSampleProjExplicitLod
// TODO(dneto): OpImageSparseSampleProjDrefImplicitLod
// TODO(dneto): OpImageSparseSampleProjDrefExplicitLod
// TODO(dneto): OpImageSparseFetch
// TODO(dneto): OpImageSparseDrefGather
// TODO(dneto): OpImageSparseTexelsResident

}  // namespace
}  // namespace spvtools
