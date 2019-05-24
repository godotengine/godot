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

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "source/opt/instruction.h"
#include "source/opt/instruction_list.h"

namespace spvtools {
namespace opt {
namespace {

using ::testing::ContainerEq;
using ::testing::ElementsAre;
using InstructionListTest = ::testing::Test;

// A class that overrides the destructor, so we can trace it.
class TestInstruction : public Instruction {
 public:
  TestInstruction() : Instruction() { created_instructions_.push_back(this); }

  ~TestInstruction() { deleted_instructions_.push_back(this); }

  static std::vector<TestInstruction*> created_instructions_;
  static std::vector<TestInstruction*> deleted_instructions_;
};

std::vector<TestInstruction*> TestInstruction::created_instructions_;
std::vector<TestInstruction*> TestInstruction::deleted_instructions_;

// Test that the destructor for InstructionList is calling the destructor
// for every element that is in the list.
TEST(InstructionListTest, Destructor) {
  InstructionList* list = new InstructionList();
  list->push_back(std::unique_ptr<Instruction>(new Instruction()));
  list->push_back(std::unique_ptr<Instruction>(new Instruction()));
  delete list;

  // Sorting because we do not care if the order of create and destruction is
  // the same.  Using generic sort just incase things are changed above.
  std::sort(TestInstruction::created_instructions_.begin(),
            TestInstruction::created_instructions_.end());
  std::sort(TestInstruction::deleted_instructions_.begin(),
            TestInstruction::deleted_instructions_.end());
  EXPECT_THAT(TestInstruction::created_instructions_,
              ContainerEq(TestInstruction::deleted_instructions_));
}

// Test the |InsertBefore| with a single instruction in the iterator class.
// Need to make sure the elements are inserted in the correct order, and the
// return value points to the correct location.
//
// Comparing addresses to make sure they remain stable, so other data structures
// can have pointers to instructions in InstructionList.
TEST(InstructionListTest, InsertBefore1) {
  InstructionList list;
  std::vector<Instruction*> inserted_instructions;
  for (int i = 0; i < 4; i++) {
    std::unique_ptr<Instruction> inst(new Instruction());
    inserted_instructions.push_back(inst.get());
    auto new_element = list.end().InsertBefore(std::move(inst));
    EXPECT_EQ(&*new_element, inserted_instructions.back());
  }

  std::vector<Instruction*> output;
  for (auto& i : list) {
    output.push_back(&i);
  }
  EXPECT_THAT(output, ContainerEq(inserted_instructions));
}

// Test inserting an entire vector of instructions using InsertBefore.  Checking
// the order of insertion and the return value.
//
// Comparing addresses to make sure they remain stable, so other data structures
// can have pointers to instructions in InstructionList.
TEST(InstructionListTest, InsertBefore2) {
  InstructionList list;
  std::vector<std::unique_ptr<Instruction>> new_instructions;
  std::vector<Instruction*> created_instructions;
  for (int i = 0; i < 4; i++) {
    std::unique_ptr<Instruction> inst(new Instruction());
    created_instructions.push_back(inst.get());
    new_instructions.push_back(std::move(inst));
  }
  auto new_element = list.begin().InsertBefore(std::move(new_instructions));
  EXPECT_TRUE(new_instructions.empty());
  EXPECT_EQ(&*new_element, created_instructions.front());

  std::vector<Instruction*> output;
  for (auto& i : list) {
    output.push_back(&i);
  }
  EXPECT_THAT(output, ContainerEq(created_instructions));
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
