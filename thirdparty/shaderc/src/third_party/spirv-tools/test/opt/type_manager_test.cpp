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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "effcee/effcee.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "source/opt/build_module.h"
#include "source/opt/instruction.h"
#include "source/opt/type_manager.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace opt {
namespace analysis {
namespace {

bool Validate(const std::vector<uint32_t>& bin) {
  spv_target_env target_env = SPV_ENV_UNIVERSAL_1_2;
  spv_context spvContext = spvContextCreate(target_env);
  spv_diagnostic diagnostic = nullptr;
  spv_const_binary_t binary = {bin.data(), bin.size()};
  spv_result_t error = spvValidate(spvContext, &binary, &diagnostic);
  if (error != 0) spvDiagnosticPrint(diagnostic);
  spvDiagnosticDestroy(diagnostic);
  spvContextDestroy(spvContext);
  return error == 0;
}

void Match(const std::string& original, IRContext* context,
           bool do_validation = true) {
  std::vector<uint32_t> bin;
  context->module()->ToBinary(&bin, true);
  if (do_validation) {
    EXPECT_TRUE(Validate(bin));
  }
  std::string assembly;
  SpirvTools tools(SPV_ENV_UNIVERSAL_1_2);
  EXPECT_TRUE(
      tools.Disassemble(bin, &assembly, SpirvTools::kDefaultDisassembleOption))
      << "Disassembling failed for shader:\n"
      << assembly << std::endl;
  auto match_result = effcee::Match(assembly, original);
  EXPECT_EQ(effcee::Result::Status::Ok, match_result.status())
      << match_result.message() << "\nChecking result:\n"
      << assembly;
}

std::vector<std::unique_ptr<Type>> GenerateAllTypes() {
  // Types in this test case are only equal to themselves, nothing else.
  std::vector<std::unique_ptr<Type>> types;

  // Void, Bool
  types.emplace_back(new Void());
  auto* voidt = types.back().get();
  types.emplace_back(new Bool());
  auto* boolt = types.back().get();

  // Integer
  types.emplace_back(new Integer(32, true));
  auto* s32 = types.back().get();
  types.emplace_back(new Integer(32, false));
  types.emplace_back(new Integer(64, true));
  types.emplace_back(new Integer(64, false));
  auto* u64 = types.back().get();

  // Float
  types.emplace_back(new Float(32));
  auto* f32 = types.back().get();
  types.emplace_back(new Float(64));

  // Vector
  types.emplace_back(new Vector(s32, 2));
  types.emplace_back(new Vector(s32, 3));
  auto* v3s32 = types.back().get();
  types.emplace_back(new Vector(u64, 4));
  types.emplace_back(new Vector(f32, 3));
  auto* v3f32 = types.back().get();

  // Matrix
  types.emplace_back(new Matrix(v3s32, 3));
  types.emplace_back(new Matrix(v3s32, 4));
  types.emplace_back(new Matrix(v3f32, 4));

  // Images
  types.emplace_back(new Image(s32, SpvDim2D, 0, 0, 0, 0, SpvImageFormatRg8,
                               SpvAccessQualifierReadOnly));
  auto* image1 = types.back().get();
  types.emplace_back(new Image(s32, SpvDim2D, 0, 1, 0, 0, SpvImageFormatRg8,
                               SpvAccessQualifierReadOnly));
  types.emplace_back(new Image(s32, SpvDim3D, 0, 1, 0, 0, SpvImageFormatRg8,
                               SpvAccessQualifierReadOnly));
  types.emplace_back(new Image(voidt, SpvDim3D, 0, 1, 0, 1, SpvImageFormatRg8,
                               SpvAccessQualifierReadWrite));
  auto* image2 = types.back().get();

  // Sampler
  types.emplace_back(new Sampler());

  // Sampled Image
  types.emplace_back(new SampledImage(image1));
  types.emplace_back(new SampledImage(image2));

  // Array
  types.emplace_back(new Array(f32, 100));
  types.emplace_back(new Array(f32, 42));
  auto* a42f32 = types.back().get();
  types.emplace_back(new Array(u64, 24));

  // RuntimeArray
  types.emplace_back(new RuntimeArray(v3f32));
  types.emplace_back(new RuntimeArray(v3s32));
  auto* rav3s32 = types.back().get();

  // Struct
  types.emplace_back(new Struct(std::vector<const Type*>{s32}));
  types.emplace_back(new Struct(std::vector<const Type*>{s32, f32}));
  auto* sts32f32 = types.back().get();
  types.emplace_back(
      new Struct(std::vector<const Type*>{u64, a42f32, rav3s32}));

  // Opaque
  types.emplace_back(new Opaque(""));
  types.emplace_back(new Opaque("hello"));
  types.emplace_back(new Opaque("world"));

  // Pointer
  types.emplace_back(new Pointer(f32, SpvStorageClassInput));
  types.emplace_back(new Pointer(sts32f32, SpvStorageClassFunction));
  types.emplace_back(new Pointer(a42f32, SpvStorageClassFunction));

  // Function
  types.emplace_back(new Function(voidt, {}));
  types.emplace_back(new Function(voidt, {boolt}));
  types.emplace_back(new Function(voidt, {boolt, s32}));
  types.emplace_back(new Function(s32, {boolt, s32}));

  // Event, Device Event, Reserve Id, Queue,
  types.emplace_back(new Event());
  types.emplace_back(new DeviceEvent());
  types.emplace_back(new ReserveId());
  types.emplace_back(new Queue());

  // Pipe, Forward Pointer, PipeStorage, NamedBarrier, AccelerationStructureNV
  types.emplace_back(new Pipe(SpvAccessQualifierReadWrite));
  types.emplace_back(new Pipe(SpvAccessQualifierReadOnly));
  types.emplace_back(new ForwardPointer(1, SpvStorageClassInput));
  types.emplace_back(new ForwardPointer(2, SpvStorageClassInput));
  types.emplace_back(new ForwardPointer(2, SpvStorageClassUniform));
  types.emplace_back(new PipeStorage());
  types.emplace_back(new NamedBarrier());
  types.emplace_back(new AccelerationStructureNV());

  return types;
}

TEST(TypeManager, TypeStrings) {
  const std::string text = R"(
    OpTypeForwardPointer !20 !2 ; id for %p is 20, Uniform is 2
    %void    = OpTypeVoid
    %bool    = OpTypeBool
    %u32     = OpTypeInt 32 0
    %id4     = OpConstant %u32 4
    %s32     = OpTypeInt 32 1
    %f64     = OpTypeFloat 64
    %v3u32   = OpTypeVector %u32 3
    %m3x3    = OpTypeMatrix %v3u32 3
    %img1    = OpTypeImage %s32 Cube 0 1 1 0 R32f ReadWrite
    %img2    = OpTypeImage %s32 Cube 0 1 1 0 R32f
    %sampler = OpTypeSampler
    %si1     = OpTypeSampledImage %img1
    %si2     = OpTypeSampledImage %img2
    %a5u32   = OpTypeArray %u32 %id4
    %af64    = OpTypeRuntimeArray %f64
    %st1     = OpTypeStruct %u32
    %st2     = OpTypeStruct %f64 %s32 %v3u32
    %opaque1 = OpTypeOpaque ""
    %opaque2 = OpTypeOpaque "opaque"
    %p       = OpTypePointer Uniform %st1
    %f       = OpTypeFunction %void %u32 %u32
    %event   = OpTypeEvent
    %de      = OpTypeDeviceEvent
    %ri      = OpTypeReserveId
    %queue   = OpTypeQueue
    %pipe    = OpTypePipe ReadOnly
    %ps      = OpTypePipeStorage
    %nb      = OpTypeNamedBarrier
    %rtacc   = OpTypeAccelerationStructureNV
  )";

  std::vector<std::pair<uint32_t, std::string>> type_id_strs = {
      {1, "void"},
      {2, "bool"},
      {3, "uint32"},
      // Id 4 is used by the constant.
      {5, "sint32"},
      {6, "float64"},
      {7, "<uint32, 3>"},
      {8, "<<uint32, 3>, 3>"},
      {9, "image(sint32, 3, 0, 1, 1, 0, 3, 2)"},
      {10, "image(sint32, 3, 0, 1, 1, 0, 3, 0)"},
      {11, "sampler"},
      {12, "sampled_image(image(sint32, 3, 0, 1, 1, 0, 3, 2))"},
      {13, "sampled_image(image(sint32, 3, 0, 1, 1, 0, 3, 0))"},
      {14, "[uint32, id(4)]"},
      {15, "[float64]"},
      {16, "{uint32}"},
      {17, "{float64, sint32, <uint32, 3>}"},
      {18, "opaque('')"},
      {19, "opaque('opaque')"},
      {20, "{uint32}*"},
      {21, "(uint32, uint32) -> void"},
      {22, "event"},
      {23, "device_event"},
      {24, "reserve_id"},
      {25, "queue"},
      {26, "pipe(0)"},
      {27, "pipe_storage"},
      {28, "named_barrier"},
      {29, "accelerationStructureNV"},
  };

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text);
  TypeManager manager(nullptr, context.get());

  EXPECT_EQ(type_id_strs.size(), manager.NumTypes());

  for (const auto& p : type_id_strs) {
    EXPECT_EQ(p.second, manager.GetType(p.first)->str());
    EXPECT_EQ(p.first, manager.GetId(manager.GetType(p.first)));
  }
}

TEST(TypeManager, StructWithFwdPtr) {
  const std::string text = R"(
               OpCapability Addresses
               OpCapability Kernel
          %1 = OpExtInstImport "OpenCL.std"
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %7 "test"
               OpSource OpenCL_C 102000
               OpDecorate %11 FuncParamAttr NoCapture
         %11 = OpDecorationGroup
               OpGroupDecorate %11 %8 %9
               OpTypeForwardPointer %100 CrossWorkgroup
       %void = OpTypeVoid
  %150 = OpTypeStruct %100
%100 = OpTypePointer CrossWorkgroup %150
          %6 = OpTypeFunction %void %100 %100
          %7 = OpFunction %void Pure %6
          %8 = OpFunctionParameter %100
          %9 = OpFunctionParameter %100
         %10 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  TypeManager manager(nullptr, context.get());

  Type* p100 = manager.GetType(100);
  Type* s150 = manager.GetType(150);

  EXPECT_TRUE(p100->AsPointer());
  EXPECT_EQ(p100->AsPointer()->pointee_type(), s150);

  EXPECT_TRUE(s150->AsStruct());
  EXPECT_EQ(s150->AsStruct()->element_types()[0], p100);
}

TEST(TypeManager, CircularFwdPtr) {
  const std::string text = R"(
               OpCapability Addresses
               OpCapability Kernel
          %1 = OpExtInstImport "OpenCL.std"
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %7 "test"
               OpSource OpenCL_C 102000
               OpDecorate %11 FuncParamAttr NoCapture
         %11 = OpDecorationGroup
               OpGroupDecorate %11 %8 %9
               OpTypeForwardPointer %100 CrossWorkgroup
               OpTypeForwardPointer %200 CrossWorkgroup
       %void = OpTypeVoid
        %int = OpTypeInt 32 0
      %float = OpTypeFloat 32
  %150 = OpTypeStruct %200 %int
  %250 = OpTypeStruct %100 %float
%100 = OpTypePointer CrossWorkgroup %150
%200 = OpTypePointer CrossWorkgroup %250
          %6 = OpTypeFunction %void %100 %200
          %7 = OpFunction %void Pure %6
          %8 = OpFunctionParameter %100
          %9 = OpFunctionParameter %200
         %10 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  TypeManager manager(nullptr, context.get());

  Type* p100 = manager.GetType(100);
  Type* s150 = manager.GetType(150);
  Type* p200 = manager.GetType(200);
  Type* s250 = manager.GetType(250);

  EXPECT_TRUE(p100->AsPointer());
  EXPECT_EQ(p100->AsPointer()->pointee_type(), s150);

  EXPECT_TRUE(p200->AsPointer());
  EXPECT_EQ(p200->AsPointer()->pointee_type(), s250);

  EXPECT_TRUE(s150->AsStruct());
  EXPECT_EQ(s150->AsStruct()->element_types()[0], p200);

  EXPECT_TRUE(s250->AsStruct());
  EXPECT_EQ(s250->AsStruct()->element_types()[0], p100);
}

TEST(TypeManager, IsomorphicStructWithFwdPtr) {
  const std::string text = R"(
               OpCapability Addresses
               OpCapability Kernel
          %1 = OpExtInstImport "OpenCL.std"
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %7 "test"
               OpSource OpenCL_C 102000
               OpDecorate %11 FuncParamAttr NoCapture
         %11 = OpDecorationGroup
               OpGroupDecorate %11 %8 %9
               OpTypeForwardPointer %100 CrossWorkgroup
               OpTypeForwardPointer %200 CrossWorkgroup
       %void = OpTypeVoid
  %_struct_1 = OpTypeStruct %100
  %_struct_2 = OpTypeStruct %200
%100 = OpTypePointer CrossWorkgroup %_struct_1
%200 = OpTypePointer CrossWorkgroup %_struct_2
          %6 = OpTypeFunction %void %100 %200
          %7 = OpFunction %void Pure %6
          %8 = OpFunctionParameter %100
          %9 = OpFunctionParameter %200
         %10 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  TypeManager manager(nullptr, context.get());

  EXPECT_EQ(manager.GetType(100), manager.GetType(200));
}

TEST(TypeManager, IsomorphicCircularFwdPtr) {
  const std::string text = R"(
               OpCapability Addresses
               OpCapability Kernel
          %1 = OpExtInstImport "OpenCL.std"
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %7 "test"
               OpSource OpenCL_C 102000
               OpDecorate %11 FuncParamAttr NoCapture
         %11 = OpDecorationGroup
               OpGroupDecorate %11 %8 %9
               OpTypeForwardPointer %100 CrossWorkgroup
               OpTypeForwardPointer %200 CrossWorkgroup
               OpTypeForwardPointer %300 CrossWorkgroup
               OpTypeForwardPointer %400 CrossWorkgroup
       %void = OpTypeVoid
        %int = OpTypeInt 32 0
      %float = OpTypeFloat 32
  %150 = OpTypeStruct %200 %int
  %250 = OpTypeStruct %100 %float
  %350 = OpTypeStruct %400 %int
  %450 = OpTypeStruct %300 %float
%100 = OpTypePointer CrossWorkgroup %150
%200 = OpTypePointer CrossWorkgroup %250
%300 = OpTypePointer CrossWorkgroup %350
%400 = OpTypePointer CrossWorkgroup %450
          %6 = OpTypeFunction %void %100 %200
          %7 = OpFunction %void Pure %6
          %8 = OpFunctionParameter %100
          %9 = OpFunctionParameter %200
         %10 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  TypeManager manager(nullptr, context.get());

  Type* p100 = manager.GetType(100);
  Type* p300 = manager.GetType(300);
  EXPECT_EQ(p100, p300);
  Type* p200 = manager.GetType(200);
  Type* p400 = manager.GetType(400);
  EXPECT_EQ(p200, p400);

  Type* p150 = manager.GetType(150);
  Type* p350 = manager.GetType(350);
  EXPECT_EQ(p150, p350);
  Type* p250 = manager.GetType(250);
  Type* p450 = manager.GetType(450);
  EXPECT_EQ(p250, p450);
}

TEST(TypeManager, PartialIsomorphicFwdPtr) {
  const std::string text = R"(
               OpCapability Addresses
               OpCapability Kernel
          %1 = OpExtInstImport "OpenCL.std"
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %7 "test"
               OpSource OpenCL_C 102000
               OpDecorate %11 FuncParamAttr NoCapture
         %11 = OpDecorationGroup
               OpGroupDecorate %11 %8 %9
               OpTypeForwardPointer %100 CrossWorkgroup
               OpTypeForwardPointer %200 CrossWorkgroup
       %void = OpTypeVoid
        %int = OpTypeInt 32 0
      %float = OpTypeFloat 32
  %150 = OpTypeStruct %200 %int
  %250 = OpTypeStruct %200 %int
%100 = OpTypePointer CrossWorkgroup %150
%200 = OpTypePointer CrossWorkgroup %250
          %6 = OpTypeFunction %void %100 %200
          %7 = OpFunction %void Pure %6
          %8 = OpFunctionParameter %100
          %9 = OpFunctionParameter %200
         %10 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  TypeManager manager(nullptr, context.get());

  Type* p100 = manager.GetType(100);
  Type* p200 = manager.GetType(200);
  EXPECT_EQ(p100->AsPointer()->pointee_type(),
            p200->AsPointer()->pointee_type());
}

TEST(TypeManager, DecorationOnStruct) {
  const std::string text = R"(
    OpDecorate %struct1 Block
    OpDecorate %struct2 Block
    OpDecorate %struct3 Block
    OpDecorate %struct4 Block

    %u32 = OpTypeInt 32 0             ; id: 5
    %f32 = OpTypeFloat 32             ; id: 6
    %struct1 = OpTypeStruct %u32 %f32 ; base
    %struct2 = OpTypeStruct %f32 %u32 ; different member order
    %struct3 = OpTypeStruct %f32      ; different member list
    %struct4 = OpTypeStruct %u32 %f32 ; the same
    %struct7 = OpTypeStruct %f32      ; no decoration
  )";
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text);
  TypeManager manager(nullptr, context.get());

  ASSERT_EQ(7u, manager.NumTypes());
  // Make sure we get ids correct.
  ASSERT_EQ("uint32", manager.GetType(5)->str());
  ASSERT_EQ("float32", manager.GetType(6)->str());

  // Try all combinations of pairs. Expect to be the same type only when the
  // same id or (1, 4).
  for (const auto id1 : {1, 2, 3, 4, 7}) {
    for (const auto id2 : {1, 2, 3, 4, 7}) {
      if (id1 == id2 || (id1 == 1 && id2 == 4) || (id1 == 4 && id2 == 1)) {
        EXPECT_TRUE(manager.GetType(id1)->IsSame(manager.GetType(id2)))
            << "%struct" << id1 << " is expected to be the same as %struct"
            << id2;
      } else {
        EXPECT_FALSE(manager.GetType(id1)->IsSame(manager.GetType(id2)))
            << "%struct" << id1 << " is expected to be different with %struct"
            << id2;
      }
    }
  }
}

TEST(TypeManager, DecorationOnMember) {
  const std::string text = R"(
    OpMemberDecorate %struct1  0 Offset 0
    OpMemberDecorate %struct2  0 Offset 0
    OpMemberDecorate %struct3  0 Offset 0
    OpMemberDecorate %struct4  0 Offset 0
    OpMemberDecorate %struct5  1 Offset 0
    OpMemberDecorate %struct6  0 Offset 4

    OpDecorate %struct7 Block
    OpMemberDecorate %struct7  0 Offset 0

    %u32 = OpTypeInt 32 0              ; id: 8
    %f32 = OpTypeFloat 32              ; id: 9
    %struct1  = OpTypeStruct %u32 %f32 ; base
    %struct2  = OpTypeStruct %f32 %u32 ; different member order
    %struct3  = OpTypeStruct %f32      ; different member list
    %struct4  = OpTypeStruct %u32 %f32 ; the same
    %struct5  = OpTypeStruct %u32 %f32 ; member decorate different field
    %struct6  = OpTypeStruct %u32 %f32 ; different member decoration parameter
    %struct7  = OpTypeStruct %u32 %f32 ; extra decoration on the struct
    %struct10 = OpTypeStruct %u32 %f32 ; no member decoration
  )";
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text);
  TypeManager manager(nullptr, context.get());

  ASSERT_EQ(10u, manager.NumTypes());
  // Make sure we get ids correct.
  ASSERT_EQ("uint32", manager.GetType(8)->str());
  ASSERT_EQ("float32", manager.GetType(9)->str());

  // Try all combinations of pairs. Expect to be the same type only when the
  // same id or (1, 4).
  for (const auto id1 : {1, 2, 3, 4, 5, 6, 7, 10}) {
    for (const auto id2 : {1, 2, 3, 4, 5, 6, 7, 10}) {
      if (id1 == id2 || (id1 == 1 && id2 == 4) || (id1 == 4 && id2 == 1)) {
        EXPECT_TRUE(manager.GetType(id1)->IsSame(manager.GetType(id2)))
            << "%struct" << id1 << " is expected to be the same as %struct"
            << id2;
      } else {
        EXPECT_FALSE(manager.GetType(id1)->IsSame(manager.GetType(id2)))
            << "%struct" << id1 << " is expected to be different with %struct"
            << id2;
      }
    }
  }
}

TEST(TypeManager, DecorationEmpty) {
  const std::string text = R"(
    OpDecorate %struct1 Block
    OpMemberDecorate %struct2  0 Offset 0

    %u32 = OpTypeInt 32 0 ; id: 3
    %f32 = OpTypeFloat 32 ; id: 4
    %struct1  = OpTypeStruct %u32 %f32
    %struct2  = OpTypeStruct %f32 %u32
    %struct5  = OpTypeStruct %f32
  )";
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text);
  TypeManager manager(nullptr, context.get());

  ASSERT_EQ(5u, manager.NumTypes());
  // Make sure we get ids correct.
  ASSERT_EQ("uint32", manager.GetType(3)->str());
  ASSERT_EQ("float32", manager.GetType(4)->str());

  // %struct1 with decoration on itself
  EXPECT_FALSE(manager.GetType(1)->decoration_empty());
  // %struct2 with decoration on its member
  EXPECT_FALSE(manager.GetType(2)->decoration_empty());
  EXPECT_TRUE(manager.GetType(3)->decoration_empty());
  EXPECT_TRUE(manager.GetType(4)->decoration_empty());
  // %struct5 has no decorations
  EXPECT_TRUE(manager.GetType(5)->decoration_empty());
}

TEST(TypeManager, BeginEndForEmptyModule) {
  const std::string text = "";
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text);
  TypeManager manager(nullptr, context.get());
  ASSERT_EQ(0u, manager.NumTypes());

  EXPECT_EQ(manager.begin(), manager.end());
}

TEST(TypeManager, BeginEnd) {
  const std::string text = R"(
    %void1   = OpTypeVoid
    %void2   = OpTypeVoid
    %bool    = OpTypeBool
    %u32     = OpTypeInt 32 0
    %f64     = OpTypeFloat 64
  )";
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text);
  TypeManager manager(nullptr, context.get());
  ASSERT_EQ(5u, manager.NumTypes());

  EXPECT_NE(manager.begin(), manager.end());
  for (const auto& t : manager) {
    switch (t.first) {
      case 1:
      case 2:
        EXPECT_EQ("void", t.second->str());
        break;
      case 3:
        EXPECT_EQ("bool", t.second->str());
        break;
      case 4:
        EXPECT_EQ("uint32", t.second->str());
        break;
      case 5:
        EXPECT_EQ("float64", t.second->str());
        break;
      default:
        EXPECT_TRUE(false && "unreachable");
        break;
    }
  }
}

TEST(TypeManager, LookupType) {
  const std::string text = R"(
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%int  = OpTypeInt 32 1
%vec2 = OpTypeVector %int 2
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  EXPECT_NE(context, nullptr);
  TypeManager manager(nullptr, context.get());

  Void voidTy;
  EXPECT_EQ(manager.GetId(&voidTy), 1u);

  Integer uintTy(32, false);
  EXPECT_EQ(manager.GetId(&uintTy), 2u);

  Integer intTy(32, true);
  EXPECT_EQ(manager.GetId(&intTy), 3u);

  Integer intTy2(32, true);
  Vector vecTy(&intTy2, 2u);
  EXPECT_EQ(manager.GetId(&vecTy), 4u);
}

TEST(TypeManager, RemoveId) {
  const std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%1 = OpTypeInt 32 0
%2 = OpTypeInt 32 1
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(context, nullptr);

  context->get_type_mgr()->RemoveId(1u);
  ASSERT_EQ(context->get_type_mgr()->GetType(1u), nullptr);
  ASSERT_NE(context->get_type_mgr()->GetType(2u), nullptr);

  context->get_type_mgr()->RemoveId(2u);
  ASSERT_EQ(context->get_type_mgr()->GetType(1u), nullptr);
  ASSERT_EQ(context->get_type_mgr()->GetType(2u), nullptr);
}

TEST(TypeManager, RemoveIdNonDuplicateAmbiguousType) {
  const std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%1 = OpTypeInt 32 0
%2 = OpTypeStruct %1
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(context, nullptr);

  Integer u32(32, false);
  Struct st({&u32});
  ASSERT_EQ(context->get_type_mgr()->GetId(&st), 2u);
  context->get_type_mgr()->RemoveId(2u);
  ASSERT_EQ(context->get_type_mgr()->GetType(2u), nullptr);
  ASSERT_EQ(context->get_type_mgr()->GetId(&st), 0u);
}

TEST(TypeManager, RemoveIdDuplicateAmbiguousType) {
  const std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%1 = OpTypeInt 32 0
%2 = OpTypeStruct %1
%3 = OpTypeStruct %1
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(context, nullptr);

  Integer u32(32, false);
  Struct st({&u32});
  uint32_t id = context->get_type_mgr()->GetId(&st);
  ASSERT_NE(id, 0u);
  uint32_t toRemove = id == 2u ? 2u : 3u;
  uint32_t toStay = id == 2u ? 3u : 2u;
  context->get_type_mgr()->RemoveId(toRemove);
  ASSERT_EQ(context->get_type_mgr()->GetType(toRemove), nullptr);
  ASSERT_EQ(context->get_type_mgr()->GetId(&st), toStay);
}

TEST(TypeManager, RemoveIdDoesntUnmapOtherTypes) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
%1 = OpTypeInt 32 0
%2 = OpTypeStruct %1
%3 = OpTypeStruct %1
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(context, nullptr);

  Integer u32(32, false);
  Struct st({&u32});

  EXPECT_EQ(1u, context->get_type_mgr()->GetId(&u32));
  uint32_t id = context->get_type_mgr()->GetId(&st);
  ASSERT_NE(id, 0u);
  uint32_t toRemove = id == 2u ? 3u : 2u;
  uint32_t toStay = id == 2u ? 2u : 3u;
  context->get_type_mgr()->RemoveId(toRemove);
  ASSERT_EQ(context->get_type_mgr()->GetType(toRemove), nullptr);
  ASSERT_EQ(context->get_type_mgr()->GetId(&st), toStay);
}

TEST(TypeManager, GetTypeAndPointerType) {
  const std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%1 = OpTypeInt 32 0
%2 = OpTypeStruct %1
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(context, nullptr);

  Integer u32(32, false);
  Pointer u32Ptr(&u32, SpvStorageClassFunction);
  Struct st({&u32});
  Pointer stPtr(&st, SpvStorageClassInput);

  auto pair = context->get_type_mgr()->GetTypeAndPointerType(
      3u, SpvStorageClassFunction);
  ASSERT_EQ(nullptr, pair.first);
  ASSERT_EQ(nullptr, pair.second);

  pair = context->get_type_mgr()->GetTypeAndPointerType(
      1u, SpvStorageClassFunction);
  ASSERT_TRUE(pair.first->IsSame(&u32));
  ASSERT_TRUE(pair.second->IsSame(&u32Ptr));

  pair =
      context->get_type_mgr()->GetTypeAndPointerType(2u, SpvStorageClassInput);
  ASSERT_TRUE(pair.first->IsSame(&st));
  ASSERT_TRUE(pair.second->IsSame(&stPtr));
}

TEST(TypeManager, DuplicateType) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
%1 = OpTypeInt 32 0
%2 = OpTypeInt 32 0
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(context, nullptr);

  const Type* type1 = context->get_type_mgr()->GetType(1u);
  const Type* type2 = context->get_type_mgr()->GetType(2u);
  EXPECT_NE(type1, nullptr);
  EXPECT_NE(type2, nullptr);
  EXPECT_EQ(*type1, *type2);
}

TEST(TypeManager, MultipleStructs) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpDecorate %3 Constant
%1 = OpTypeInt 32 0
%2 = OpTypeStruct %1
%3 = OpTypeStruct %1
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(context, nullptr);

  const Type* type1 = context->get_type_mgr()->GetType(2u);
  const Type* type2 = context->get_type_mgr()->GetType(3u);
  EXPECT_NE(type1, nullptr);
  EXPECT_NE(type2, nullptr);
  EXPECT_FALSE(type1->IsSame(type2));
}

TEST(TypeManager, RemovingIdAvoidsUseAfterFree) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
%1 = OpTypeInt 32 0
%2 = OpTypeStruct %1
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(context, nullptr);

  Integer u32(32, false);
  Struct st({&u32});
  const Type* type = context->get_type_mgr()->GetType(2u);
  EXPECT_NE(type, nullptr);
  context->get_type_mgr()->RemoveId(1u);
  EXPECT_TRUE(type->IsSame(&st));
}

TEST(TypeManager, RegisterAndRemoveId) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
%1 = OpTypeInt 32 0
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(context, nullptr);

  uint32_t id = 2u;
  {
    // Ensure that u32 goes out of scope.
    Integer u32(32, false);
    Struct st({&u32});
    context->get_type_mgr()->RegisterType(id, st);
  }

  context->get_type_mgr()->RemoveId(id);
  EXPECT_EQ(nullptr, context->get_type_mgr()->GetType(id));
}

TEST(TypeManager, RegisterAndRemoveIdAllTypes) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(context, nullptr);

  std::vector<std::unique_ptr<Type>> types = GenerateAllTypes();
  uint32_t id = 1u;
  for (auto& t : types) {
    context->get_type_mgr()->RegisterType(id, *t);
    EXPECT_EQ(*t, *context->get_type_mgr()->GetType(id));
  }
  types.clear();

  for (; id > 0; --id) {
    context->get_type_mgr()->RemoveId(id);
    EXPECT_EQ(nullptr, context->get_type_mgr()->GetType(id));
  }
}

TEST(TypeManager, RegisterAndRemoveIdWithDecorations) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
%1 = OpTypeInt 32 0
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(context, nullptr);

  uint32_t id = 2u;
  {
    Integer u32(32, false);
    Struct st({&u32, &u32});
    st.AddDecoration({10});
    st.AddDecoration({11});
    st.AddMemberDecoration(0, {{35, 4}});
    st.AddMemberDecoration(1, {{35, 4}});
    st.AddMemberDecoration(1, {{36, 5}});
    context->get_type_mgr()->RegisterType(id, st);
    EXPECT_EQ(st, *context->get_type_mgr()->GetType(id));
  }

  context->get_type_mgr()->RemoveId(id);
  EXPECT_EQ(nullptr, context->get_type_mgr()->GetType(id));
}

TEST(TypeManager, GetTypeInstructionInt) {
  const std::string text = R"(
; CHECK: OpTypeInt 32 0
; CHECK: OpTypeInt 16 1
OpCapability Shader
OpCapability Int16
OpCapability Linkage
OpMemoryModel Logical GLSL450
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  EXPECT_NE(context, nullptr);

  Integer uint_32(32, false);
  context->get_type_mgr()->GetTypeInstruction(&uint_32);

  Integer int_16(16, true);
  context->get_type_mgr()->GetTypeInstruction(&int_16);

  Match(text, context.get());
}

TEST(TypeManager, GetTypeInstructionDuplicateInts) {
  const std::string text = R"(
; CHECK: OpTypeInt 32 0
; CHECK-NOT: OpType
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  EXPECT_NE(context, nullptr);

  Integer uint_32(32, false);
  uint32_t id = context->get_type_mgr()->GetTypeInstruction(&uint_32);

  Integer other(32, false);
  EXPECT_EQ(context->get_type_mgr()->GetTypeInstruction(&other), id);

  Match(text, context.get());
}

TEST(TypeManager, GetTypeInstructionAllTypes) {
  const std::string text = R"(
; CHECK: [[uint:%\w+]] = OpTypeInt 32 0
; CHECK: [[input_ptr:%\w+]] = OpTypePointer Input [[uint]]
; CHECK: [[uniform_ptr:%\w+]] = OpTypePointer Uniform [[uint]]
; CHECK: [[uint24:%\w+]] = OpConstant [[uint]] 24
; CHECK: [[uint42:%\w+]] = OpConstant [[uint]] 42
; CHECK: [[uint100:%\w+]] = OpConstant [[uint]] 100
; CHECK: [[void:%\w+]] = OpTypeVoid
; CHECK: [[bool:%\w+]] = OpTypeBool
; CHECK: [[s32:%\w+]] = OpTypeInt 32 1
; CHECK: OpTypeInt 64 1
; CHECK: [[u64:%\w+]] = OpTypeInt 64 0
; CHECK: [[f32:%\w+]] = OpTypeFloat 32
; CHECK: OpTypeFloat 64
; CHECK: OpTypeVector [[s32]] 2
; CHECK: [[v3s32:%\w+]] = OpTypeVector [[s32]] 3
; CHECK: OpTypeVector [[u64]] 4
; CHECK: [[v3f32:%\w+]] = OpTypeVector [[f32]] 3
; CHECK: OpTypeMatrix [[v3s32]] 3
; CHECK: OpTypeMatrix [[v3s32]] 4
; CHECK: OpTypeMatrix [[v3f32]] 4
; CHECK: [[image1:%\w+]] = OpTypeImage [[s32]] 2D 0 0 0 0 Rg8 ReadOnly
; CHECK: OpTypeImage [[s32]] 2D 0 1 0 0 Rg8 ReadOnly
; CHECK: OpTypeImage [[s32]] 3D 0 1 0 0 Rg8 ReadOnly
; CHECK: [[image2:%\w+]] = OpTypeImage [[void]] 3D 0 1 0 1 Rg8 ReadWrite
; CHECK: OpTypeSampler
; CHECK: OpTypeSampledImage [[image1]]
; CHECK: OpTypeSampledImage [[image2]]
; CHECK: OpTypeArray [[f32]] [[uint100]]
; CHECK: [[a42f32:%\w+]] = OpTypeArray [[f32]] [[uint42]]
; CHECK: OpTypeArray [[u64]] [[uint24]]
; CHECK: OpTypeRuntimeArray [[v3f32]]
; CHECK: [[rav3s32:%\w+]] = OpTypeRuntimeArray [[v3s32]]
; CHECK: OpTypeStruct [[s32]]
; CHECK: [[sts32f32:%\w+]] = OpTypeStruct [[s32]] [[f32]]
; CHECK: OpTypeStruct [[u64]] [[a42f32]] [[rav3s32]]
; CHECK: OpTypeOpaque ""
; CHECK: OpTypeOpaque "hello"
; CHECK: OpTypeOpaque "world"
; CHECK: OpTypePointer Input [[f32]]
; CHECK: OpTypePointer Function [[sts32f32]]
; CHECK: OpTypePointer Function [[a42f32]]
; CHECK: OpTypeFunction [[void]]
; CHECK: OpTypeFunction [[void]] [[bool]]
; CHECK: OpTypeFunction [[void]] [[bool]] [[s32]]
; CHECK: OpTypeFunction [[s32]] [[bool]] [[s32]]
; CHECK: OpTypeEvent
; CHECK: OpTypeDeviceEvent
; CHECK: OpTypeReserveId
; CHECK: OpTypeQueue
; CHECK: OpTypePipe ReadWrite
; CHECK: OpTypePipe ReadOnly
; CHECK: OpTypeForwardPointer [[input_ptr]] Input
; CHECK: OpTypeForwardPointer [[uniform_ptr]] Input
; CHECK: OpTypeForwardPointer [[uniform_ptr]] Uniform
; CHECK: OpTypePipeStorage
; CHECK: OpTypeNamedBarrier
; CHECK: OpTypeAccelerationStructureNV
OpCapability Shader
OpCapability Int64
OpCapability Linkage
OpMemoryModel Logical GLSL450
%uint = OpTypeInt 32 0
%1 = OpTypePointer Input %uint
%2 = OpTypePointer Uniform %uint
%24 = OpConstant %uint 24
%42 = OpConstant %uint 42
%100 = OpConstant %uint 100
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(context, nullptr);

  std::vector<std::unique_ptr<Type>> types = GenerateAllTypes();
  for (auto& t : types) {
    context->get_type_mgr()->GetTypeInstruction(t.get());
  }

  Match(text, context.get(), false);
}

TEST(TypeManager, GetTypeInstructionWithDecorations) {
  const std::string text = R"(
; CHECK: OpDecorate [[struct:%\w+]] CPacked
; CHECK: OpMemberDecorate [[struct]] 1 Offset 4
; CHECK: [[uint:%\w+]] = OpTypeInt 32 0
; CHECK: [[struct]] = OpTypeStruct [[uint]] [[uint]]
OpCapability Shader
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical GLSL450
%uint = OpTypeInt 32 0
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(context, nullptr);

  Integer u32(32, false);
  Struct st({&u32, &u32});
  st.AddDecoration({10});
  st.AddMemberDecoration(1, {{35, 4}});
  (void)context->get_def_use_mgr();
  context->get_type_mgr()->GetTypeInstruction(&st);

  Match(text, context.get());
}

TEST(TypeManager, GetPointerToAmbiguousType1) {
  const std::string text = R"(
; CHECK: [[struct1:%\w+]] = OpTypeStruct
; CHECK: [[struct2:%\w+]] = OpTypeStruct
; CHECK: OpTypePointer Function [[struct2]]
; CHECK: OpTypePointer Function [[struct1]]
OpCapability Shader
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical GLSL450
%uint = OpTypeInt 32 0
%1 = OpTypeStruct %uint
%2 = OpTypeStruct %uint
%3 = OpTypePointer Function %2
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(context, nullptr);

  context->get_type_mgr()->FindPointerToType(1, SpvStorageClassFunction);
  Match(text, context.get());
}

TEST(TypeManager, GetPointerToAmbiguousType2) {
  const std::string text = R"(
; CHECK: [[struct1:%\w+]] = OpTypeStruct
; CHECK: [[struct2:%\w+]] = OpTypeStruct
; CHECK: OpTypePointer Function [[struct1]]
; CHECK: OpTypePointer Function [[struct2]]
OpCapability Shader
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical GLSL450
%uint = OpTypeInt 32 0
%1 = OpTypeStruct %uint
%2 = OpTypeStruct %uint
%3 = OpTypePointer Function %1
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(context, nullptr);

  context->get_type_mgr()->FindPointerToType(2, SpvStorageClassFunction);
  Match(text, context.get());
}

}  // namespace
}  // namespace analysis
}  // namespace opt
}  // namespace spvtools
