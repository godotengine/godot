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

// Assembler tests for instructions in the "Mode-Setting" section of the
// SPIR-V spec.

#include <string>
#include <tuple>
#include <vector>

#include "gmock/gmock.h"
#include "test/test_fixture.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace {

using spvtest::EnumCase;
using spvtest::MakeInstruction;
using spvtest::MakeVector;
using ::testing::Combine;
using ::testing::Eq;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

// Test OpMemoryModel

// An example case for OpMemoryModel
struct MemoryModelCase {
  uint32_t get_addressing_value() const {
    return static_cast<uint32_t>(addressing_value);
  }
  uint32_t get_memory_value() const {
    return static_cast<uint32_t>(memory_value);
  }
  SpvAddressingModel addressing_value;
  std::string addressing_name;
  SpvMemoryModel memory_value;
  std::string memory_name;
};

using OpMemoryModelTest =
    spvtest::TextToBinaryTestBase<TestWithParam<MemoryModelCase>>;

TEST_P(OpMemoryModelTest, AnyMemoryModelCase) {
  const std::string input = "OpMemoryModel " + GetParam().addressing_name +
                            " " + GetParam().memory_name;
  EXPECT_THAT(
      CompiledInstructions(input),
      Eq(MakeInstruction(SpvOpMemoryModel, {GetParam().get_addressing_value(),
                                            GetParam().get_memory_value()})));
}

#define CASE(ADDRESSING, MEMORY)                                         \
  {                                                                      \
    SpvAddressingModel##ADDRESSING, #ADDRESSING, SpvMemoryModel##MEMORY, \
        #MEMORY                                                          \
  }
// clang-format off
INSTANTIATE_TEST_SUITE_P(TextToBinaryMemoryModel, OpMemoryModelTest,
                        ValuesIn(std::vector<MemoryModelCase>{
                          // These cases exercise each addressing model, and
                          // each memory model, but not necessarily in
                          // combination.
                            CASE(Logical,Simple),
                            CASE(Logical,GLSL450),
                            CASE(Physical32,OpenCL),
                            CASE(Physical64,OpenCL),
                        }));
#undef CASE
// clang-format on

TEST_F(OpMemoryModelTest, WrongModel) {
  EXPECT_THAT(CompileFailure("OpMemoryModel xxyyzz Simple"),
              Eq("Invalid addressing model 'xxyyzz'."));
  EXPECT_THAT(CompileFailure("OpMemoryModel Logical xxyyzz"),
              Eq("Invalid memory model 'xxyyzz'."));
}

// Test OpEntryPoint

// An example case for OpEntryPoint
struct EntryPointCase {
  uint32_t get_execution_value() const {
    return static_cast<uint32_t>(execution_value);
  }
  SpvExecutionModel execution_value;
  std::string execution_name;
  std::string entry_point_name;
};

using OpEntryPointTest =
    spvtest::TextToBinaryTestBase<TestWithParam<EntryPointCase>>;

TEST_P(OpEntryPointTest, AnyEntryPointCase) {
  // TODO(dneto): utf-8, escaping, quoting cases for entry point name.
  const std::string input = "OpEntryPoint " + GetParam().execution_name +
                            " %1 \"" + GetParam().entry_point_name + "\"";
  EXPECT_THAT(
      CompiledInstructions(input),
      Eq(MakeInstruction(SpvOpEntryPoint, {GetParam().get_execution_value(), 1},
                         MakeVector(GetParam().entry_point_name))));
}

// clang-format off
#define CASE(NAME) SpvExecutionModel##NAME, #NAME
INSTANTIATE_TEST_SUITE_P(TextToBinaryEntryPoint, OpEntryPointTest,
                        ValuesIn(std::vector<EntryPointCase>{
                          { CASE(Vertex), "" },
                          { CASE(TessellationControl), "my tess" },
                          { CASE(TessellationEvaluation), "really fancy" },
                          { CASE(Geometry), "Euclid" },
                          { CASE(Fragment), "FAT32" },
                          { CASE(GLCompute), "cubic" },
                          { CASE(Kernel), "Sanders" },
                        }));
#undef CASE
// clang-format on

TEST_F(OpEntryPointTest, WrongModel) {
  EXPECT_THAT(CompileFailure("OpEntryPoint xxyyzz %1 \"fun\""),
              Eq("Invalid execution model 'xxyyzz'."));
}

// Test OpExecutionMode
using OpExecutionModeTest = spvtest::TextToBinaryTestBase<
    TestWithParam<std::tuple<spv_target_env, EnumCase<SpvExecutionMode>>>>;

TEST_P(OpExecutionModeTest, AnyExecutionMode) {
  // This string should assemble, but should not validate.
  std::stringstream input;
  input << "OpExecutionMode %1 " << std::get<1>(GetParam()).name();
  for (auto operand : std::get<1>(GetParam()).operands())
    input << " " << operand;
  EXPECT_THAT(CompiledInstructions(input.str(), std::get<0>(GetParam())),
              Eq(MakeInstruction(SpvOpExecutionMode,
                                 {1, std::get<1>(GetParam()).value()},
                                 std::get<1>(GetParam()).operands())));
}

#define CASE(NAME) SpvExecutionMode##NAME, #NAME
INSTANTIATE_TEST_SUITE_P(
    TextToBinaryExecutionMode, OpExecutionModeTest,
    Combine(Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1),
            ValuesIn(std::vector<EnumCase<SpvExecutionMode>>{
                // The operand literal values are arbitrarily chosen,
                // but there are the right number of them.
                {CASE(Invocations), {101}},
                {CASE(SpacingEqual), {}},
                {CASE(SpacingFractionalEven), {}},
                {CASE(SpacingFractionalOdd), {}},
                {CASE(VertexOrderCw), {}},
                {CASE(VertexOrderCcw), {}},
                {CASE(PixelCenterInteger), {}},
                {CASE(OriginUpperLeft), {}},
                {CASE(OriginLowerLeft), {}},
                {CASE(EarlyFragmentTests), {}},
                {CASE(PointMode), {}},
                {CASE(Xfb), {}},
                {CASE(DepthReplacing), {}},
                {CASE(DepthGreater), {}},
                {CASE(DepthLess), {}},
                {CASE(DepthUnchanged), {}},
                {CASE(LocalSize), {64, 1, 2}},
                {CASE(LocalSizeHint), {8, 2, 4}},
                {CASE(InputPoints), {}},
                {CASE(InputLines), {}},
                {CASE(InputLinesAdjacency), {}},
                {CASE(Triangles), {}},
                {CASE(InputTrianglesAdjacency), {}},
                {CASE(Quads), {}},
                {CASE(Isolines), {}},
                {CASE(OutputVertices), {21}},
                {CASE(OutputPoints), {}},
                {CASE(OutputLineStrip), {}},
                {CASE(OutputTriangleStrip), {}},
                {CASE(VecTypeHint), {96}},
                {CASE(ContractionOff), {}},
            })));

INSTANTIATE_TEST_SUITE_P(
    TextToBinaryExecutionModeV11, OpExecutionModeTest,
    Combine(Values(SPV_ENV_UNIVERSAL_1_1),
            ValuesIn(std::vector<EnumCase<SpvExecutionMode>>{
                {CASE(Initializer)},
                {CASE(Finalizer)},
                {CASE(SubgroupSize), {12}},
                {CASE(SubgroupsPerWorkgroup), {64}}})));
#undef CASE

TEST_F(OpExecutionModeTest, WrongMode) {
  EXPECT_THAT(CompileFailure("OpExecutionMode %1 xxyyzz"),
              Eq("Invalid execution mode 'xxyyzz'."));
}

TEST_F(OpExecutionModeTest, TooManyModes) {
  EXPECT_THAT(CompileFailure("OpExecutionMode %1 Xfb PointMode"),
              Eq("Expected <opcode> or <result-id> at the beginning of an "
                 "instruction, found 'PointMode'."));
}

// Test OpCapability

using OpCapabilityTest =
    spvtest::TextToBinaryTestBase<TestWithParam<EnumCase<SpvCapability>>>;

TEST_P(OpCapabilityTest, AnyCapability) {
  const std::string input = "OpCapability " + GetParam().name();
  EXPECT_THAT(CompiledInstructions(input),
              Eq(MakeInstruction(SpvOpCapability, {GetParam().value()})));
}

// clang-format off
#define CASE(NAME) { SpvCapability##NAME, #NAME }
INSTANTIATE_TEST_SUITE_P(TextToBinaryCapability, OpCapabilityTest,
                        ValuesIn(std::vector<EnumCase<SpvCapability>>{
                            CASE(Matrix),
                            CASE(Shader),
                            CASE(Geometry),
                            CASE(Tessellation),
                            CASE(Addresses),
                            CASE(Linkage),
                            CASE(Kernel),
                            CASE(Vector16),
                            CASE(Float16Buffer),
                            CASE(Float16),
                            CASE(Float64),
                            CASE(Int64),
                            CASE(Int64Atomics),
                            CASE(ImageBasic),
                            CASE(ImageReadWrite),
                            CASE(ImageMipmap),
                            // Value 16 intentionally missing
                            CASE(Pipes),
                            CASE(Groups),
                            CASE(DeviceEnqueue),
                            CASE(LiteralSampler),
                            CASE(AtomicStorage),
                            CASE(Int16),
                            CASE(TessellationPointSize),
                            CASE(GeometryPointSize),
                            CASE(ImageGatherExtended),
                            // Value 26 intentionally missing
                            CASE(StorageImageMultisample),
                            CASE(UniformBufferArrayDynamicIndexing),
                            CASE(SampledImageArrayDynamicIndexing),
                            CASE(StorageBufferArrayDynamicIndexing),
                            CASE(StorageImageArrayDynamicIndexing),
                            CASE(ClipDistance),
                            CASE(CullDistance),
                            CASE(ImageCubeArray),
                            CASE(SampleRateShading),
                            CASE(ImageRect),
                            CASE(SampledRect),
                            CASE(GenericPointer),
                            CASE(Int8),
                            CASE(InputAttachment),
                            CASE(SparseResidency),
                            CASE(MinLod),
                            CASE(Sampled1D),
                            CASE(Image1D),
                            CASE(SampledCubeArray),
                            CASE(SampledBuffer),
                            CASE(ImageBuffer),
                            CASE(ImageMSArray),
                            CASE(StorageImageExtendedFormats),
                            CASE(ImageQuery),
                            CASE(DerivativeControl),
                            CASE(InterpolationFunction),
                            CASE(TransformFeedback),
                        }));
#undef CASE
// clang-format on

using TextToBinaryCapability = spvtest::TextToBinaryTest;

TEST_F(TextToBinaryCapability, BadMissingCapability) {
  EXPECT_THAT(CompileFailure("OpCapability"),
              Eq("Expected operand, found end of stream."));
}

TEST_F(TextToBinaryCapability, BadInvalidCapability) {
  EXPECT_THAT(CompileFailure("OpCapability 123"),
              Eq("Invalid capability '123'."));
}

// TODO(dneto): OpExecutionMode

}  // namespace
}  // namespace spvtools
