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
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "source/opt/types.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace opt {
namespace analysis {
namespace {

// Fixture class providing some element types.
class SameTypeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    void_t_ = MakeUnique<Void>();
    u32_t_ = MakeUnique<Integer>(32, false);
    f64_t_ = MakeUnique<Float>(64);
    v3u32_t_ = MakeUnique<Vector>(u32_t_.get(), 3);
    image_t_ =
        MakeUnique<Image>(f64_t_.get(), SpvDim2D, 1, 1, 0, 0, SpvImageFormatR16,
                          SpvAccessQualifierReadWrite);
  }

  // Element types to be used for constructing other types for testing.
  std::unique_ptr<Type> void_t_;
  std::unique_ptr<Type> u32_t_;
  std::unique_ptr<Type> f64_t_;
  std::unique_ptr<Type> v3u32_t_;
  std::unique_ptr<Type> image_t_;
};

#define TestMultipleInstancesOfTheSameType(ty, ...)                       \
  TEST_F(SameTypeTest, MultiSame##ty) {                                   \
    std::vector<std::unique_ptr<Type>> types;                             \
    for (int i = 0; i < 10; ++i) types.emplace_back(new ty(__VA_ARGS__)); \
    for (size_t i = 0; i < types.size(); ++i) {                           \
      for (size_t j = 0; j < types.size(); ++j) {                         \
        EXPECT_TRUE(types[i]->IsSame(types[j].get()))                     \
            << "expected '" << types[i]->str() << "' is the same as '"    \
            << types[j]->str() << "'";                                    \
        EXPECT_TRUE(*types[i] == *types[j])                               \
            << "expected '" << types[i]->str() << "' is the same as '"    \
            << types[j]->str() << "'";                                    \
      }                                                                   \
    }                                                                     \
  }
TestMultipleInstancesOfTheSameType(Void);
TestMultipleInstancesOfTheSameType(Bool);
TestMultipleInstancesOfTheSameType(Integer, 32, true);
TestMultipleInstancesOfTheSameType(Float, 64);
TestMultipleInstancesOfTheSameType(Vector, u32_t_.get(), 3);
TestMultipleInstancesOfTheSameType(Matrix, v3u32_t_.get(), 4);
TestMultipleInstancesOfTheSameType(Image, f64_t_.get(), SpvDimCube, 0, 0, 1, 1,
                                   SpvImageFormatRgb10A2,
                                   SpvAccessQualifierWriteOnly);
TestMultipleInstancesOfTheSameType(Sampler);
TestMultipleInstancesOfTheSameType(SampledImage, image_t_.get());
TestMultipleInstancesOfTheSameType(Array, u32_t_.get(), 10);
TestMultipleInstancesOfTheSameType(RuntimeArray, u32_t_.get());
TestMultipleInstancesOfTheSameType(Struct, std::vector<const Type*>{
                                               u32_t_.get(), f64_t_.get()});
TestMultipleInstancesOfTheSameType(Opaque, "testing rocks");
TestMultipleInstancesOfTheSameType(Pointer, u32_t_.get(), SpvStorageClassInput);
TestMultipleInstancesOfTheSameType(Function, u32_t_.get(),
                                   {f64_t_.get(), f64_t_.get()});
TestMultipleInstancesOfTheSameType(Event);
TestMultipleInstancesOfTheSameType(DeviceEvent);
TestMultipleInstancesOfTheSameType(ReserveId);
TestMultipleInstancesOfTheSameType(Queue);
TestMultipleInstancesOfTheSameType(Pipe, SpvAccessQualifierReadWrite);
TestMultipleInstancesOfTheSameType(ForwardPointer, 10, SpvStorageClassUniform);
TestMultipleInstancesOfTheSameType(PipeStorage);
TestMultipleInstancesOfTheSameType(NamedBarrier);
TestMultipleInstancesOfTheSameType(AccelerationStructureNV);
#undef TestMultipleInstanceOfTheSameType

std::vector<std::unique_ptr<Type>> GenerateAllTypes() {
  // Types in this test case are only equal to themselves, nothing else.
  std::vector<std::unique_ptr<Type>> types;

  // Forward Pointer
  types.emplace_back(new ForwardPointer(10000, SpvStorageClassInput));
  types.emplace_back(new ForwardPointer(20000, SpvStorageClassInput));

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
  types.emplace_back(new Pointer(voidt, SpvStorageClassFunction));

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

  // Pipe, Forward Pointer, PipeStorage, NamedBarrier
  types.emplace_back(new Pipe(SpvAccessQualifierReadWrite));
  types.emplace_back(new Pipe(SpvAccessQualifierReadOnly));
  types.emplace_back(new ForwardPointer(1, SpvStorageClassInput));
  types.emplace_back(new ForwardPointer(2, SpvStorageClassInput));
  types.emplace_back(new ForwardPointer(2, SpvStorageClassUniform));
  types.emplace_back(new PipeStorage());
  types.emplace_back(new NamedBarrier());

  return types;
}

TEST(Types, AllTypes) {
  // Types in this test case are only equal to themselves, nothing else.
  std::vector<std::unique_ptr<Type>> types = GenerateAllTypes();

  for (size_t i = 0; i < types.size(); ++i) {
    for (size_t j = 0; j < types.size(); ++j) {
      if (i == j) {
        EXPECT_TRUE(types[i]->IsSame(types[j].get()))
            << "expected '" << types[i]->str() << "' is the same as '"
            << types[j]->str() << "'";
      } else {
        EXPECT_FALSE(types[i]->IsSame(types[j].get()))
            << "expected '" << types[i]->str() << "' is different to '"
            << types[j]->str() << "'";
      }
    }
  }
}

TEST(Types, IntSignedness) {
  std::vector<bool> signednesses = {true, false, false, true};
  std::vector<std::unique_ptr<Integer>> types;
  for (bool s : signednesses) {
    types.emplace_back(new Integer(32, s));
  }
  for (size_t i = 0; i < signednesses.size(); i++) {
    EXPECT_EQ(signednesses[i], types[i]->IsSigned());
  }
}

TEST(Types, IntWidth) {
  std::vector<uint32_t> widths = {1, 2, 4, 8, 16, 32, 48, 64, 128};
  std::vector<std::unique_ptr<Integer>> types;
  for (uint32_t w : widths) {
    types.emplace_back(new Integer(w, true));
  }
  for (size_t i = 0; i < widths.size(); i++) {
    EXPECT_EQ(widths[i], types[i]->width());
  }
}

TEST(Types, FloatWidth) {
  std::vector<uint32_t> widths = {1, 2, 4, 8, 16, 32, 48, 64, 128};
  std::vector<std::unique_ptr<Float>> types;
  for (uint32_t w : widths) {
    types.emplace_back(new Float(w));
  }
  for (size_t i = 0; i < widths.size(); i++) {
    EXPECT_EQ(widths[i], types[i]->width());
  }
}

TEST(Types, VectorElementCount) {
  auto s32 = MakeUnique<Integer>(32, true);
  for (uint32_t c : {2, 3, 4}) {
    auto s32v = MakeUnique<Vector>(s32.get(), c);
    EXPECT_EQ(c, s32v->element_count());
  }
}

TEST(Types, MatrixElementCount) {
  auto s32 = MakeUnique<Integer>(32, true);
  auto s32v4 = MakeUnique<Vector>(s32.get(), 4);
  for (uint32_t c : {1, 2, 3, 4, 10, 100}) {
    auto s32m = MakeUnique<Matrix>(s32v4.get(), c);
    EXPECT_EQ(c, s32m->element_count());
  }
}

TEST(Types, IsUniqueType) {
  std::vector<std::unique_ptr<Type>> types = GenerateAllTypes();

  for (auto& t : types) {
    bool expectation = true;
    // Disallowing variable pointers.
    switch (t->kind()) {
      case Type::kArray:
      case Type::kRuntimeArray:
      case Type::kStruct:
        expectation = false;
        break;
      default:
        break;
    }
    EXPECT_EQ(t->IsUniqueType(false), expectation)
        << "expected '" << t->str() << "' to be a "
        << (expectation ? "" : "non-") << "unique type";

    // Allowing variables pointers.
    if (t->AsPointer()) expectation = false;
    EXPECT_EQ(t->IsUniqueType(true), expectation)
        << "expected '" << t->str() << "' to be a "
        << (expectation ? "" : "non-") << "unique type";
  }
}

std::vector<std::unique_ptr<Type>> GenerateAllTypesWithDecorations() {
  std::vector<std::unique_ptr<Type>> types = GenerateAllTypes();
  uint32_t elems = 1;
  uint32_t decs = 1;
  for (auto& t : types) {
    for (uint32_t i = 0; i < (decs % 10); ++i) {
      std::vector<uint32_t> decoration;
      for (uint32_t j = 0; j < (elems % 4) + 1; ++j) {
        decoration.push_back(j);
      }
      t->AddDecoration(std::move(decoration));
      ++elems;
      ++decs;
    }
  }

  return types;
}

TEST(Types, Clone) {
  std::vector<std::unique_ptr<Type>> types = GenerateAllTypesWithDecorations();
  for (auto& t : types) {
    auto clone = t->Clone();
    EXPECT_TRUE(*t == *clone);
    EXPECT_TRUE(t->HasSameDecorations(clone.get()));
    EXPECT_NE(clone.get(), t.get());
  }
}

TEST(Types, RemoveDecorations) {
  std::vector<std::unique_ptr<Type>> types = GenerateAllTypesWithDecorations();
  for (auto& t : types) {
    auto decorationless = t->RemoveDecorations();
    EXPECT_EQ(*t == *decorationless, t->decoration_empty());
    EXPECT_EQ(t->HasSameDecorations(decorationless.get()),
              t->decoration_empty());
    EXPECT_NE(t.get(), decorationless.get());
  }
}

}  // namespace
}  // namespace analysis
}  // namespace opt
}  // namespace spvtools
