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

#ifndef TEST_OPT_PASS_FIXTURE_H_
#define TEST_OPT_PASS_FIXTURE_H_

#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "effcee/effcee.h"
#include "gtest/gtest.h"
#include "source/opt/build_module.h"
#include "source/opt/pass_manager.h"
#include "source/opt/passes.h"
#include "source/spirv_validator_options.h"
#include "source/util/make_unique.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace opt {

// Template class for testing passes. It contains some handy utility methods for
// running passes and checking results.
//
// To write value-Parameterized tests:
//   using ValueParamTest = PassTest<::testing::TestWithParam<std::string>>;
// To use as normal fixture:
//   using FixtureTest = PassTest<::testing::Test>;
template <typename TestT>
class PassTest : public TestT {
 public:
  PassTest()
      : consumer_(
            [](spv_message_level_t, const char*, const spv_position_t&,
               const char* message) { std::cerr << message << std::endl; }),
        context_(nullptr),
        tools_(SPV_ENV_UNIVERSAL_1_3),
        manager_(new PassManager()),
        assemble_options_(SpirvTools::kDefaultAssembleOption),
        disassemble_options_(SpirvTools::kDefaultDisassembleOption) {}

  // Runs the given |pass| on the binary assembled from the |original|.
  // Returns a tuple of the optimized binary and the boolean value returned
  // from pass Process() function.
  std::tuple<std::vector<uint32_t>, Pass::Status> OptimizeToBinary(
      Pass* pass, const std::string& original, bool skip_nop) {
    context_ = std::move(BuildModule(SPV_ENV_UNIVERSAL_1_3, consumer_, original,
                                     assemble_options_));
    EXPECT_NE(nullptr, context()) << "Assembling failed for shader:\n"
                                  << original << std::endl;
    if (!context()) {
      return std::make_tuple(std::vector<uint32_t>(), Pass::Status::Failure);
    }

    const auto status = pass->Run(context());

    std::vector<uint32_t> binary;
    context()->module()->ToBinary(&binary, skip_nop);
    return std::make_tuple(binary, status);
  }

  // Runs a single pass of class |PassT| on the binary assembled from the
  // |assembly|. Returns a tuple of the optimized binary and the boolean value
  // from the pass Process() function.
  template <typename PassT, typename... Args>
  std::tuple<std::vector<uint32_t>, Pass::Status> SinglePassRunToBinary(
      const std::string& assembly, bool skip_nop, Args&&... args) {
    auto pass = MakeUnique<PassT>(std::forward<Args>(args)...);
    pass->SetMessageConsumer(consumer_);
    return OptimizeToBinary(pass.get(), assembly, skip_nop);
  }

  // Runs a single pass of class |PassT| on the binary assembled from the
  // |assembly|, disassembles the optimized binary. Returns a tuple of
  // disassembly string and the boolean value from the pass Process() function.
  template <typename PassT, typename... Args>
  std::tuple<std::string, Pass::Status> SinglePassRunAndDisassemble(
      const std::string& assembly, bool skip_nop, bool do_validation,
      Args&&... args) {
    std::vector<uint32_t> optimized_bin;
    auto status = Pass::Status::SuccessWithoutChange;
    std::tie(optimized_bin, status) = SinglePassRunToBinary<PassT>(
        assembly, skip_nop, std::forward<Args>(args)...);
    if (do_validation) {
      spv_target_env target_env = SPV_ENV_UNIVERSAL_1_3;
      spv_context spvContext = spvContextCreate(target_env);
      spv_diagnostic diagnostic = nullptr;
      spv_const_binary_t binary = {optimized_bin.data(), optimized_bin.size()};
      spv_result_t error = spvValidateWithOptions(
          spvContext, ValidatorOptions(), &binary, &diagnostic);
      EXPECT_EQ(error, 0);
      if (error != 0) spvDiagnosticPrint(diagnostic);
      spvDiagnosticDestroy(diagnostic);
      spvContextDestroy(spvContext);
    }
    std::string optimized_asm;
    EXPECT_TRUE(
        tools_.Disassemble(optimized_bin, &optimized_asm, disassemble_options_))
        << "Disassembling failed for shader:\n"
        << assembly << std::endl;
    return std::make_tuple(optimized_asm, status);
  }

  // Runs a single pass of class |PassT| on the binary assembled from the
  // |original| assembly, and checks whether the optimized binary can be
  // disassembled to the |expected| assembly. Optionally will also validate
  // the optimized binary. This does *not* involve pass manager. Callers
  // are suggested to use SCOPED_TRACE() for better messages.
  template <typename PassT, typename... Args>
  void SinglePassRunAndCheck(const std::string& original,
                             const std::string& expected, bool skip_nop,
                             bool do_validation, Args&&... args) {
    std::vector<uint32_t> optimized_bin;
    auto status = Pass::Status::SuccessWithoutChange;
    std::tie(optimized_bin, status) = SinglePassRunToBinary<PassT>(
        original, skip_nop, std::forward<Args>(args)...);
    // Check whether the pass returns the correct modification indication.
    EXPECT_NE(Pass::Status::Failure, status);
    EXPECT_EQ(original == expected,
              status == Pass::Status::SuccessWithoutChange);
    if (do_validation) {
      spv_target_env target_env = SPV_ENV_UNIVERSAL_1_3;
      spv_context spvContext = spvContextCreate(target_env);
      spv_diagnostic diagnostic = nullptr;
      spv_const_binary_t binary = {optimized_bin.data(), optimized_bin.size()};
      spv_result_t error = spvValidateWithOptions(
          spvContext, ValidatorOptions(), &binary, &diagnostic);
      EXPECT_EQ(error, 0);
      if (error != 0) spvDiagnosticPrint(diagnostic);
      spvDiagnosticDestroy(diagnostic);
      spvContextDestroy(spvContext);
    }
    std::string optimized_asm;
    EXPECT_TRUE(
        tools_.Disassemble(optimized_bin, &optimized_asm, disassemble_options_))
        << "Disassembling failed for shader:\n"
        << original << std::endl;
    EXPECT_EQ(expected, optimized_asm);
  }

  // Runs a single pass of class |PassT| on the binary assembled from the
  // |original| assembly, and checks whether the optimized binary can be
  // disassembled to the |expected| assembly. This does *not* involve pass
  // manager. Callers are suggested to use SCOPED_TRACE() for better messages.
  template <typename PassT, typename... Args>
  void SinglePassRunAndCheck(const std::string& original,
                             const std::string& expected, bool skip_nop,
                             Args&&... args) {
    SinglePassRunAndCheck<PassT>(original, expected, skip_nop, false,
                                 std::forward<Args>(args)...);
  }

  // Runs a single pass of class |PassT| on the binary assembled from the
  // |original| assembly, then runs an Effcee matcher over the disassembled
  // result, using checks parsed from |original|.  Always skips OpNop.
  // This does *not* involve pass manager.  Callers are suggested to use
  // SCOPED_TRACE() for better messages.
  template <typename PassT, typename... Args>
  void SinglePassRunAndMatch(const std::string& original, bool do_validation,
                             Args&&... args) {
    const bool skip_nop = true;
    auto pass_result = SinglePassRunAndDisassemble<PassT>(
        original, skip_nop, do_validation, std::forward<Args>(args)...);
    auto disassembly = std::get<0>(pass_result);
    auto match_result = effcee::Match(disassembly, original);
    EXPECT_EQ(effcee::Result::Status::Ok, match_result.status())
        << match_result.message() << "\nChecking result:\n"
        << disassembly;
  }

  // Adds a pass to be run.
  template <typename PassT, typename... Args>
  void AddPass(Args&&... args) {
    manager_->AddPass<PassT>(std::forward<Args>(args)...);
  }

  // Renews the pass manager, including clearing all previously added passes.
  void RenewPassManger() {
    manager_ = MakeUnique<PassManager>();
    manager_->SetMessageConsumer(consumer_);
  }

  // Runs the passes added thus far using a pass manager on the binary assembled
  // from the |original| assembly, and checks whether the optimized binary can
  // be disassembled to the |expected| assembly. Callers are suggested to use
  // SCOPED_TRACE() for better messages.
  void RunAndCheck(const std::string& original, const std::string& expected) {
    assert(manager_->NumPasses());

    context_ = std::move(BuildModule(SPV_ENV_UNIVERSAL_1_3, nullptr, original,
                                     assemble_options_));
    ASSERT_NE(nullptr, context());

    manager_->Run(context());

    std::vector<uint32_t> binary;
    context()->module()->ToBinary(&binary, /* skip_nop = */ false);

    std::string optimized;
    EXPECT_TRUE(tools_.Disassemble(binary, &optimized, disassemble_options_));
    EXPECT_EQ(expected, optimized);
  }

  void SetAssembleOptions(uint32_t assemble_options) {
    assemble_options_ = assemble_options;
  }

  void SetDisassembleOptions(uint32_t disassemble_options) {
    disassemble_options_ = disassemble_options;
  }

  MessageConsumer consumer() { return consumer_; }
  IRContext* context() { return context_.get(); }

  void SetMessageConsumer(MessageConsumer msg_consumer) {
    consumer_ = msg_consumer;
  }

  spv_validator_options ValidatorOptions() { return &validator_options_; }

 private:
  MessageConsumer consumer_;            // Message consumer.
  std::unique_ptr<IRContext> context_;  // IR context
  SpirvTools tools_;  // An instance for calling SPIRV-Tools functionalities.
  std::unique_ptr<PassManager> manager_;  // The pass manager.
  uint32_t assemble_options_;
  uint32_t disassemble_options_;
  spv_validator_options_t validator_options_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // TEST_OPT_PASS_FIXTURE_H_
