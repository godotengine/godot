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

#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "source/util/make_unique.h"
#include "test/opt/module_utils.h"
#include "test/opt/pass_fixture.h"

namespace spvtools {
namespace opt {
namespace {

using spvtest::GetIdBound;
using ::testing::Eq;

// A null pass whose construtors accept arguments
class NullPassWithArgs : public NullPass {
 public:
  NullPassWithArgs(uint32_t) {}
  NullPassWithArgs(std::string) {}
  NullPassWithArgs(const std::vector<int>&) {}
  NullPassWithArgs(const std::vector<int>&, uint32_t) {}

  const char* name() const override { return "null-with-args"; }
};

TEST(PassManager, Interface) {
  PassManager manager;
  EXPECT_EQ(0u, manager.NumPasses());

  manager.AddPass<StripDebugInfoPass>();
  EXPECT_EQ(1u, manager.NumPasses());
  EXPECT_STREQ("strip-debug", manager.GetPass(0)->name());

  manager.AddPass(MakeUnique<NullPass>());
  EXPECT_EQ(2u, manager.NumPasses());
  EXPECT_STREQ("strip-debug", manager.GetPass(0)->name());
  EXPECT_STREQ("null", manager.GetPass(1)->name());

  manager.AddPass<StripDebugInfoPass>();
  EXPECT_EQ(3u, manager.NumPasses());
  EXPECT_STREQ("strip-debug", manager.GetPass(0)->name());
  EXPECT_STREQ("null", manager.GetPass(1)->name());
  EXPECT_STREQ("strip-debug", manager.GetPass(2)->name());

  manager.AddPass<NullPassWithArgs>(1u);
  manager.AddPass<NullPassWithArgs>("null pass args");
  manager.AddPass<NullPassWithArgs>(std::initializer_list<int>{1, 2});
  manager.AddPass<NullPassWithArgs>(std::initializer_list<int>{1, 2}, 3);
  EXPECT_EQ(7u, manager.NumPasses());
  EXPECT_STREQ("strip-debug", manager.GetPass(0)->name());
  EXPECT_STREQ("null", manager.GetPass(1)->name());
  EXPECT_STREQ("strip-debug", manager.GetPass(2)->name());
  EXPECT_STREQ("null-with-args", manager.GetPass(3)->name());
  EXPECT_STREQ("null-with-args", manager.GetPass(4)->name());
  EXPECT_STREQ("null-with-args", manager.GetPass(5)->name());
  EXPECT_STREQ("null-with-args", manager.GetPass(6)->name());
}

// A pass that appends an OpNop instruction to the debug1 section.
class AppendOpNopPass : public Pass {
 public:
  const char* name() const override { return "AppendOpNop"; }
  Status Process() override {
    context()->AddDebug1Inst(MakeUnique<Instruction>(context()));
    return Status::SuccessWithChange;
  }
};

// A pass that appends specified number of OpNop instructions to the debug1
// section.
class AppendMultipleOpNopPass : public Pass {
 public:
  explicit AppendMultipleOpNopPass(uint32_t num_nop) : num_nop_(num_nop) {}

  const char* name() const override { return "AppendOpNop"; }
  Status Process() override {
    for (uint32_t i = 0; i < num_nop_; i++) {
      context()->AddDebug1Inst(MakeUnique<Instruction>(context()));
    }
    return Status::SuccessWithChange;
  }

 private:
  uint32_t num_nop_;
};

// A pass that duplicates the last instruction in the debug1 section.
class DuplicateInstPass : public Pass {
 public:
  const char* name() const override { return "DuplicateInst"; }
  Status Process() override {
    auto inst = MakeUnique<Instruction>(*(--context()->debug1_end()));
    context()->AddDebug1Inst(std::move(inst));
    return Status::SuccessWithChange;
  }
};

using PassManagerTest = PassTest<::testing::Test>;

TEST_F(PassManagerTest, Run) {
  const std::string text = "OpMemoryModel Logical GLSL450\nOpSource ESSL 310\n";

  AddPass<AppendOpNopPass>();
  AddPass<AppendOpNopPass>();
  RunAndCheck(text, text + "OpNop\nOpNop\n");

  RenewPassManger();
  AddPass<AppendOpNopPass>();
  AddPass<DuplicateInstPass>();
  RunAndCheck(text, text + "OpNop\nOpNop\n");

  RenewPassManger();
  AddPass<DuplicateInstPass>();
  AddPass<AppendOpNopPass>();
  RunAndCheck(text, text + "OpSource ESSL 310\nOpNop\n");

  RenewPassManger();
  AddPass<AppendMultipleOpNopPass>(3);
  RunAndCheck(text, text + "OpNop\nOpNop\nOpNop\n");
}

// A pass that appends an OpTypeVoid instruction that uses a given id.
class AppendTypeVoidInstPass : public Pass {
 public:
  explicit AppendTypeVoidInstPass(uint32_t result_id) : result_id_(result_id) {}

  const char* name() const override { return "AppendTypeVoidInstPass"; }
  Status Process() override {
    auto inst = MakeUnique<Instruction>(context(), SpvOpTypeVoid, 0, result_id_,
                                        std::vector<Operand>{});
    context()->AddType(std::move(inst));
    return Status::SuccessWithChange;
  }

 private:
  uint32_t result_id_;
};

TEST(PassManager, RecomputeIdBoundAutomatically) {
  PassManager manager;
  std::unique_ptr<Module> module(new Module());
  IRContext context(SPV_ENV_UNIVERSAL_1_2, std::move(module),
                    manager.consumer());
  EXPECT_THAT(GetIdBound(*context.module()), Eq(0u));

  manager.Run(&context);
  manager.AddPass<AppendOpNopPass>();
  // With no ID changes, the ID bound does not change.
  EXPECT_THAT(GetIdBound(*context.module()), Eq(0u));

  // Now we force an Id of 100 to be used.
  manager.AddPass(MakeUnique<AppendTypeVoidInstPass>(100));
  EXPECT_THAT(GetIdBound(*context.module()), Eq(0u));
  manager.Run(&context);
  // The Id has been updated automatically, even though the pass
  // did not update it.
  EXPECT_THAT(GetIdBound(*context.module()), Eq(101u));

  // Try one more time!
  manager.AddPass(MakeUnique<AppendTypeVoidInstPass>(200));
  manager.Run(&context);
  EXPECT_THAT(GetIdBound(*context.module()), Eq(201u));

  // Add another pass, but which uses a lower Id.
  manager.AddPass(MakeUnique<AppendTypeVoidInstPass>(10));
  manager.Run(&context);
  // The Id stays high.
  EXPECT_THAT(GetIdBound(*context.module()), Eq(201u));
}

}  // anonymous namespace
}  // namespace opt
}  // namespace spvtools
