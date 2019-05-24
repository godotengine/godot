// Copyright (c) 2018 Google LLC
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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "source/opt/build_module.h"
#include "source/opt/constants.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {
namespace analysis {
namespace {

using ConstantManagerTest = ::testing::Test;

TEST_F(ConstantManagerTest, GetDefiningInstruction) {
  const std::string text = R"(
%int = OpTypeInt 32 0
%1 = OpTypeStruct %int
%2 = OpTypeStruct %int
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ASSERT_NE(context, nullptr);

  Type* struct_type_1 = context->get_type_mgr()->GetType(1);
  StructConstant struct_const_1(struct_type_1->AsStruct());
  Instruction* const_inst_1 =
      context->get_constant_mgr()->GetDefiningInstruction(&struct_const_1, 1);
  EXPECT_EQ(const_inst_1->type_id(), 1);

  Type* struct_type_2 = context->get_type_mgr()->GetType(2);
  StructConstant struct_const_2(struct_type_2->AsStruct());
  Instruction* const_inst_2 =
      context->get_constant_mgr()->GetDefiningInstruction(&struct_const_2, 2);
  EXPECT_EQ(const_inst_2->type_id(), 2);
}

TEST_F(ConstantManagerTest, GetDefiningInstruction2) {
  const std::string text = R"(
%int = OpTypeInt 32 0
%1 = OpTypeStruct %int
%2 = OpTypeStruct %int
%3 = OpConstantNull %1
%4 = OpConstantNull %2
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ASSERT_NE(context, nullptr);

  Type* struct_type_1 = context->get_type_mgr()->GetType(1);
  NullConstant struct_const_1(struct_type_1->AsStruct());
  Instruction* const_inst_1 =
      context->get_constant_mgr()->GetDefiningInstruction(&struct_const_1, 1);
  EXPECT_EQ(const_inst_1->type_id(), 1);
  EXPECT_EQ(const_inst_1->result_id(), 3);

  Type* struct_type_2 = context->get_type_mgr()->GetType(2);
  NullConstant struct_const_2(struct_type_2->AsStruct());
  Instruction* const_inst_2 =
      context->get_constant_mgr()->GetDefiningInstruction(&struct_const_2, 2);
  EXPECT_EQ(const_inst_2->type_id(), 2);
  EXPECT_EQ(const_inst_2->result_id(), 4);
}

}  // namespace
}  // namespace analysis
}  // namespace opt
}  // namespace spvtools
