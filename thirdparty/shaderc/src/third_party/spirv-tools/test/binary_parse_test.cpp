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

#include <algorithm>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "source/latest_version_opencl_std_header.h"
#include "source/table.h"
#include "test/test_fixture.h"
#include "test/unit_spirv.h"

// Returns true if two spv_parsed_operand_t values are equal.
// To use this operator, this definition must appear in the same namespace
// as spv_parsed_operand_t.
static bool operator==(const spv_parsed_operand_t& a,
                       const spv_parsed_operand_t& b) {
  return a.offset == b.offset && a.num_words == b.num_words &&
         a.type == b.type && a.number_kind == b.number_kind &&
         a.number_bit_width == b.number_bit_width;
}

namespace spvtools {
namespace {

using ::spvtest::Concatenate;
using ::spvtest::MakeInstruction;
using ::spvtest::MakeVector;
using ::spvtest::ScopedContext;
using ::testing::_;
using ::testing::AnyOf;
using ::testing::Eq;
using ::testing::InSequence;
using ::testing::Return;

// An easily-constructible and comparable object for the contents of an
// spv_parsed_instruction_t.  Unlike spv_parsed_instruction_t, owns the memory
// of its components.
struct ParsedInstruction {
  explicit ParsedInstruction(const spv_parsed_instruction_t& inst)
      : words(inst.words, inst.words + inst.num_words),
        opcode(static_cast<SpvOp>(inst.opcode)),
        ext_inst_type(inst.ext_inst_type),
        type_id(inst.type_id),
        result_id(inst.result_id),
        operands(inst.operands, inst.operands + inst.num_operands) {}

  std::vector<uint32_t> words;
  SpvOp opcode;
  spv_ext_inst_type_t ext_inst_type;
  uint32_t type_id;
  uint32_t result_id;
  std::vector<spv_parsed_operand_t> operands;

  bool operator==(const ParsedInstruction& b) const {
    return words == b.words && opcode == b.opcode &&
           ext_inst_type == b.ext_inst_type && type_id == b.type_id &&
           result_id == b.result_id && operands == b.operands;
  }
};

// Prints a ParsedInstruction object to the given output stream, and returns
// the stream.
std::ostream& operator<<(std::ostream& os, const ParsedInstruction& inst) {
  os << "\nParsedInstruction( {";
  spvtest::PrintTo(spvtest::WordVector(inst.words), &os);
  os << "}, opcode: " << int(inst.opcode)
     << " ext_inst_type: " << int(inst.ext_inst_type)
     << " type_id: " << inst.type_id << " result_id: " << inst.result_id;
  for (const auto& operand : inst.operands) {
    os << " { offset: " << operand.offset << " num_words: " << operand.num_words
       << " type: " << int(operand.type)
       << " number_kind: " << int(operand.number_kind)
       << " number_bit_width: " << int(operand.number_bit_width) << "}";
  }
  os << ")";
  return os;
}

// Sanity check for the equality operator on ParsedInstruction.
TEST(ParsedInstruction, ZeroInitializedAreEqual) {
  spv_parsed_instruction_t pi = {};
  ParsedInstruction a(pi);
  ParsedInstruction b(pi);
  EXPECT_THAT(a, ::testing::TypedEq<ParsedInstruction>(b));
}

// Googlemock class receiving Header/Instruction calls from spvBinaryParse().
class MockParseClient {
 public:
  MOCK_METHOD6(Header, spv_result_t(spv_endianness_t endian, uint32_t magic,
                                    uint32_t version, uint32_t generator,
                                    uint32_t id_bound, uint32_t reserved));
  MOCK_METHOD1(Instruction, spv_result_t(const ParsedInstruction&));
};

// Casts user_data as MockParseClient and invokes its Header().
spv_result_t invoke_header(void* user_data, spv_endianness_t endian,
                           uint32_t magic, uint32_t version, uint32_t generator,
                           uint32_t id_bound, uint32_t reserved) {
  return static_cast<MockParseClient*>(user_data)->Header(
      endian, magic, version, generator, id_bound, reserved);
}

// Casts user_data as MockParseClient and invokes its Instruction().
spv_result_t invoke_instruction(
    void* user_data, const spv_parsed_instruction_t* parsed_instruction) {
  return static_cast<MockParseClient*>(user_data)->Instruction(
      ParsedInstruction(*parsed_instruction));
}

// The SPIR-V module header words for the Khronos Assembler generator,
// for a module with an ID bound of 1.
const uint32_t kHeaderForBound1[] = {
    SpvMagicNumber, SpvVersion,
    SPV_GENERATOR_WORD(SPV_GENERATOR_KHRONOS_ASSEMBLER, 0), 1 /*bound*/,
    0 /*schema*/};

// Returns the expected SPIR-V module header words for the Khronos
// Assembler generator, and with a given Id bound.
std::vector<uint32_t> ExpectedHeaderForBound(uint32_t bound) {
  return {SpvMagicNumber, 0x10000,
          SPV_GENERATOR_WORD(SPV_GENERATOR_KHRONOS_ASSEMBLER, 0), bound, 0};
}

// Returns a parsed operand for a non-number value at the given word offset
// within an instruction.
spv_parsed_operand_t MakeSimpleOperand(uint16_t offset,
                                       spv_operand_type_t type) {
  return {offset, 1, type, SPV_NUMBER_NONE, 0};
}

// Returns a parsed operand for a literal unsigned integer value at the given
// word offset within an instruction.
spv_parsed_operand_t MakeLiteralNumberOperand(uint16_t offset) {
  return {offset, 1, SPV_OPERAND_TYPE_LITERAL_INTEGER, SPV_NUMBER_UNSIGNED_INT,
          32};
}

// Returns a parsed operand for a literal string value at the given
// word offset within an instruction.
spv_parsed_operand_t MakeLiteralStringOperand(uint16_t offset,
                                              uint16_t length) {
  return {offset, length, SPV_OPERAND_TYPE_LITERAL_STRING, SPV_NUMBER_NONE, 0};
}

// Returns a ParsedInstruction for an OpTypeVoid instruction that would
// generate the given result Id.
ParsedInstruction MakeParsedVoidTypeInstruction(uint32_t result_id) {
  const auto void_inst = MakeInstruction(SpvOpTypeVoid, {result_id});
  const auto void_operands = std::vector<spv_parsed_operand_t>{
      MakeSimpleOperand(1, SPV_OPERAND_TYPE_RESULT_ID)};
  const spv_parsed_instruction_t parsed_void_inst = {
      void_inst.data(),
      static_cast<uint16_t>(void_inst.size()),
      SpvOpTypeVoid,
      SPV_EXT_INST_TYPE_NONE,
      0,  // type id
      result_id,
      void_operands.data(),
      static_cast<uint16_t>(void_operands.size())};
  return ParsedInstruction(parsed_void_inst);
}

// Returns a ParsedInstruction for an OpTypeInt instruction that generates
// the given result Id for a 32-bit signed integer scalar type.
ParsedInstruction MakeParsedInt32TypeInstruction(uint32_t result_id) {
  const auto i32_inst = MakeInstruction(SpvOpTypeInt, {result_id, 32, 1});
  const auto i32_operands = std::vector<spv_parsed_operand_t>{
      MakeSimpleOperand(1, SPV_OPERAND_TYPE_RESULT_ID),
      MakeLiteralNumberOperand(2), MakeLiteralNumberOperand(3)};
  spv_parsed_instruction_t parsed_i32_inst = {
      i32_inst.data(),
      static_cast<uint16_t>(i32_inst.size()),
      SpvOpTypeInt,
      SPV_EXT_INST_TYPE_NONE,
      0,  // type id
      result_id,
      i32_operands.data(),
      static_cast<uint16_t>(i32_operands.size())};
  return ParsedInstruction(parsed_i32_inst);
}

class BinaryParseTest : public spvtest::TextToBinaryTestBase<::testing::Test> {
 protected:
  ~BinaryParseTest() { spvDiagnosticDestroy(diagnostic_); }

  void Parse(const SpirvVector& words, spv_result_t expected_result,
             bool flip_words = false) {
    SpirvVector flipped_words(words);
    SCOPED_TRACE(flip_words ? "Flipped Endianness" : "Normal Endianness");
    if (flip_words) {
      std::transform(flipped_words.begin(), flipped_words.end(),
                     flipped_words.begin(), [](const uint32_t raw_word) {
                       return spvFixWord(raw_word,
                                         I32_ENDIAN_HOST == I32_ENDIAN_BIG
                                             ? SPV_ENDIANNESS_LITTLE
                                             : SPV_ENDIANNESS_BIG);
                     });
    }
    EXPECT_EQ(expected_result,
              spvBinaryParse(ScopedContext().context, &client_,
                             flipped_words.data(), flipped_words.size(),
                             invoke_header, invoke_instruction, &diagnostic_));
  }

  spv_diagnostic diagnostic_ = nullptr;
  MockParseClient client_;
};

// Adds an EXPECT_CALL to client_->Header() with appropriate parameters,
// including bound.  Returns the EXPECT_CALL result.
#define EXPECT_HEADER(bound)                                                   \
  EXPECT_CALL(                                                                 \
      client_,                                                                 \
      Header(AnyOf(SPV_ENDIANNESS_LITTLE, SPV_ENDIANNESS_BIG), SpvMagicNumber, \
             0x10000, SPV_GENERATOR_WORD(SPV_GENERATOR_KHRONOS_ASSEMBLER, 0),  \
             bound, 0 /*reserved*/))

static const bool kSwapEndians[] = {false, true};

TEST_F(BinaryParseTest, EmptyModuleHasValidHeaderAndNoInstructionCallbacks) {
  for (bool endian_swap : kSwapEndians) {
    const auto words = CompileSuccessfully("");
    EXPECT_HEADER(1).WillOnce(Return(SPV_SUCCESS));
    EXPECT_CALL(client_, Instruction(_)).Times(0);  // No instruction callback.
    Parse(words, SPV_SUCCESS, endian_swap);
    EXPECT_EQ(nullptr, diagnostic_);
  }
}

TEST_F(BinaryParseTest, NullDiagnosticsIsOkForGoodParse) {
  const auto words = CompileSuccessfully("");
  EXPECT_HEADER(1).WillOnce(Return(SPV_SUCCESS));
  EXPECT_CALL(client_, Instruction(_)).Times(0);  // No instruction callback.
  EXPECT_EQ(
      SPV_SUCCESS,
      spvBinaryParse(ScopedContext().context, &client_, words.data(),
                     words.size(), invoke_header, invoke_instruction, nullptr));
}

TEST_F(BinaryParseTest, NullDiagnosticsIsOkForBadParse) {
  auto words = CompileSuccessfully("");
  words.push_back(0xffffffff);  // Certainly invalid instruction header.
  EXPECT_HEADER(1).WillOnce(Return(SPV_SUCCESS));
  EXPECT_CALL(client_, Instruction(_)).Times(0);  // No instruction callback.
  EXPECT_EQ(
      SPV_ERROR_INVALID_BINARY,
      spvBinaryParse(ScopedContext().context, &client_, words.data(),
                     words.size(), invoke_header, invoke_instruction, nullptr));
}

// Make sure that we don't blow up when both the consumer and the diagnostic are
// null.
TEST_F(BinaryParseTest, NullConsumerNullDiagnosticsForBadParse) {
  auto words = CompileSuccessfully("");

  auto ctx = spvtools::Context(SPV_ENV_UNIVERSAL_1_1);
  ctx.SetMessageConsumer(nullptr);

  words.push_back(0xffffffff);  // Certainly invalid instruction header.
  EXPECT_HEADER(1).WillOnce(Return(SPV_SUCCESS));
  EXPECT_CALL(client_, Instruction(_)).Times(0);  // No instruction callback.
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY,
            spvBinaryParse(ctx.CContext(), &client_, words.data(), words.size(),
                           invoke_header, invoke_instruction, nullptr));
}

TEST_F(BinaryParseTest, SpecifyConsumerNullDiagnosticsForGoodParse) {
  const auto words = CompileSuccessfully("");

  auto ctx = spvtools::Context(SPV_ENV_UNIVERSAL_1_1);
  int invocation = 0;
  ctx.SetMessageConsumer([&invocation](spv_message_level_t, const char*,
                                       const spv_position_t&,
                                       const char*) { ++invocation; });

  EXPECT_HEADER(1).WillOnce(Return(SPV_SUCCESS));
  EXPECT_CALL(client_, Instruction(_)).Times(0);  // No instruction callback.
  EXPECT_EQ(SPV_SUCCESS,
            spvBinaryParse(ctx.CContext(), &client_, words.data(), words.size(),
                           invoke_header, invoke_instruction, nullptr));
  EXPECT_EQ(0, invocation);
}

TEST_F(BinaryParseTest, SpecifyConsumerNullDiagnosticsForBadParse) {
  auto words = CompileSuccessfully("");

  auto ctx = spvtools::Context(SPV_ENV_UNIVERSAL_1_1);
  int invocation = 0;
  ctx.SetMessageConsumer(
      [&invocation](spv_message_level_t level, const char* source,
                    const spv_position_t& position, const char* message) {
        ++invocation;
        EXPECT_EQ(SPV_MSG_ERROR, level);
        EXPECT_STREQ("input", source);
        EXPECT_EQ(0u, position.line);
        EXPECT_EQ(0u, position.column);
        EXPECT_EQ(1u, position.index);
        EXPECT_STREQ("Invalid opcode: 65535", message);
      });

  words.push_back(0xffffffff);  // Certainly invalid instruction header.
  EXPECT_HEADER(1).WillOnce(Return(SPV_SUCCESS));
  EXPECT_CALL(client_, Instruction(_)).Times(0);  // No instruction callback.
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY,
            spvBinaryParse(ctx.CContext(), &client_, words.data(), words.size(),
                           invoke_header, invoke_instruction, nullptr));
  EXPECT_EQ(1, invocation);
}

TEST_F(BinaryParseTest, SpecifyConsumerSpecifyDiagnosticsForGoodParse) {
  const auto words = CompileSuccessfully("");

  auto ctx = spvtools::Context(SPV_ENV_UNIVERSAL_1_1);
  int invocation = 0;
  ctx.SetMessageConsumer([&invocation](spv_message_level_t, const char*,
                                       const spv_position_t&,
                                       const char*) { ++invocation; });

  EXPECT_HEADER(1).WillOnce(Return(SPV_SUCCESS));
  EXPECT_CALL(client_, Instruction(_)).Times(0);  // No instruction callback.
  EXPECT_EQ(SPV_SUCCESS,
            spvBinaryParse(ctx.CContext(), &client_, words.data(), words.size(),
                           invoke_header, invoke_instruction, &diagnostic_));
  EXPECT_EQ(0, invocation);
  EXPECT_EQ(nullptr, diagnostic_);
}

TEST_F(BinaryParseTest, SpecifyConsumerSpecifyDiagnosticsForBadParse) {
  auto words = CompileSuccessfully("");

  auto ctx = spvtools::Context(SPV_ENV_UNIVERSAL_1_1);
  int invocation = 0;
  ctx.SetMessageConsumer([&invocation](spv_message_level_t, const char*,
                                       const spv_position_t&,
                                       const char*) { ++invocation; });

  words.push_back(0xffffffff);  // Certainly invalid instruction header.
  EXPECT_HEADER(1).WillOnce(Return(SPV_SUCCESS));
  EXPECT_CALL(client_, Instruction(_)).Times(0);  // No instruction callback.
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY,
            spvBinaryParse(ctx.CContext(), &client_, words.data(), words.size(),
                           invoke_header, invoke_instruction, &diagnostic_));
  EXPECT_EQ(0, invocation);
  EXPECT_STREQ("Invalid opcode: 65535", diagnostic_->error);
}

TEST_F(BinaryParseTest,
       ModuleWithSingleInstructionHasValidHeaderAndInstructionCallback) {
  for (bool endian_swap : kSwapEndians) {
    const auto words = CompileSuccessfully("%1 = OpTypeVoid");
    InSequence calls_expected_in_specific_order;
    EXPECT_HEADER(2).WillOnce(Return(SPV_SUCCESS));
    EXPECT_CALL(client_, Instruction(MakeParsedVoidTypeInstruction(1)))
        .WillOnce(Return(SPV_SUCCESS));
    Parse(words, SPV_SUCCESS, endian_swap);
    EXPECT_EQ(nullptr, diagnostic_);
  }
}

TEST_F(BinaryParseTest, NullHeaderCallbackIsIgnored) {
  const auto words = CompileSuccessfully("%1 = OpTypeVoid");
  EXPECT_CALL(client_, Header(_, _, _, _, _, _))
      .Times(0);  // No header callback.
  EXPECT_CALL(client_, Instruction(MakeParsedVoidTypeInstruction(1)))
      .WillOnce(Return(SPV_SUCCESS));
  EXPECT_EQ(SPV_SUCCESS, spvBinaryParse(ScopedContext().context, &client_,
                                        words.data(), words.size(), nullptr,
                                        invoke_instruction, &diagnostic_));
  EXPECT_EQ(nullptr, diagnostic_);
}

TEST_F(BinaryParseTest, NullInstructionCallbackIsIgnored) {
  const auto words = CompileSuccessfully("%1 = OpTypeVoid");
  EXPECT_HEADER((2)).WillOnce(Return(SPV_SUCCESS));
  EXPECT_CALL(client_, Instruction(_)).Times(0);  // No instruction callback.
  EXPECT_EQ(SPV_SUCCESS,
            spvBinaryParse(ScopedContext().context, &client_, words.data(),
                           words.size(), invoke_header, nullptr, &diagnostic_));
  EXPECT_EQ(nullptr, diagnostic_);
}

// Check the result of multiple instruction callbacks.
//
// This test exercises non-default values for the following members of the
// spv_parsed_instruction_t struct: words, num_words, opcode, result_id,
// operands, num_operands.
TEST_F(BinaryParseTest, TwoScalarTypesGenerateTwoInstructionCallbacks) {
  for (bool endian_swap : kSwapEndians) {
    const auto words = CompileSuccessfully(
        "%1 = OpTypeVoid "
        "%2 = OpTypeInt 32 1");
    InSequence calls_expected_in_specific_order;
    EXPECT_HEADER(3).WillOnce(Return(SPV_SUCCESS));
    EXPECT_CALL(client_, Instruction(MakeParsedVoidTypeInstruction(1)))
        .WillOnce(Return(SPV_SUCCESS));
    EXPECT_CALL(client_, Instruction(MakeParsedInt32TypeInstruction(2)))
        .WillOnce(Return(SPV_SUCCESS));
    Parse(words, SPV_SUCCESS, endian_swap);
    EXPECT_EQ(nullptr, diagnostic_);
  }
}

TEST_F(BinaryParseTest, EarlyReturnWithZeroPassingCallbacks) {
  for (bool endian_swap : kSwapEndians) {
    const auto words = CompileSuccessfully(
        "%1 = OpTypeVoid "
        "%2 = OpTypeInt 32 1");
    InSequence calls_expected_in_specific_order;
    EXPECT_HEADER(3).WillOnce(Return(SPV_ERROR_INVALID_BINARY));
    // Early exit means no calls to Instruction().
    EXPECT_CALL(client_, Instruction(_)).Times(0);
    Parse(words, SPV_ERROR_INVALID_BINARY, endian_swap);
    // On error, the binary parser doesn't generate its own diagnostics.
    EXPECT_EQ(nullptr, diagnostic_);
  }
}

TEST_F(BinaryParseTest,
       EarlyReturnWithZeroPassingCallbacksAndSpecifiedResultCode) {
  for (bool endian_swap : kSwapEndians) {
    const auto words = CompileSuccessfully(
        "%1 = OpTypeVoid "
        "%2 = OpTypeInt 32 1");
    InSequence calls_expected_in_specific_order;
    EXPECT_HEADER(3).WillOnce(Return(SPV_REQUESTED_TERMINATION));
    // Early exit means no calls to Instruction().
    EXPECT_CALL(client_, Instruction(_)).Times(0);
    Parse(words, SPV_REQUESTED_TERMINATION, endian_swap);
    // On early termination, the binary parser doesn't generate its own
    // diagnostics.
    EXPECT_EQ(nullptr, diagnostic_);
  }
}

TEST_F(BinaryParseTest, EarlyReturnWithOnePassingCallback) {
  for (bool endian_swap : kSwapEndians) {
    const auto words = CompileSuccessfully(
        "%1 = OpTypeVoid "
        "%2 = OpTypeInt 32 1 "
        "%3 = OpTypeFloat 32");
    InSequence calls_expected_in_specific_order;
    EXPECT_HEADER(4).WillOnce(Return(SPV_SUCCESS));
    EXPECT_CALL(client_, Instruction(MakeParsedVoidTypeInstruction(1)))
        .WillOnce(Return(SPV_REQUESTED_TERMINATION));
    Parse(words, SPV_REQUESTED_TERMINATION, endian_swap);
    // On early termination, the binary parser doesn't generate its own
    // diagnostics.
    EXPECT_EQ(nullptr, diagnostic_);
  }
}

TEST_F(BinaryParseTest, EarlyReturnWithTwoPassingCallbacks) {
  for (bool endian_swap : kSwapEndians) {
    const auto words = CompileSuccessfully(
        "%1 = OpTypeVoid "
        "%2 = OpTypeInt 32 1 "
        "%3 = OpTypeFloat 32");
    InSequence calls_expected_in_specific_order;
    EXPECT_HEADER(4).WillOnce(Return(SPV_SUCCESS));
    EXPECT_CALL(client_, Instruction(MakeParsedVoidTypeInstruction(1)))
        .WillOnce(Return(SPV_SUCCESS));
    EXPECT_CALL(client_, Instruction(MakeParsedInt32TypeInstruction(2)))
        .WillOnce(Return(SPV_REQUESTED_TERMINATION));
    Parse(words, SPV_REQUESTED_TERMINATION, endian_swap);
    // On early termination, the binary parser doesn't generate its own
    // diagnostics.
    EXPECT_EQ(nullptr, diagnostic_);
  }
}

TEST_F(BinaryParseTest, InstructionWithStringOperand) {
  const std::string str =
      "the future is already here, it's just not evenly distributed";
  const auto str_words = MakeVector(str);
  const auto instruction = MakeInstruction(SpvOpName, {99}, str_words);
  const auto words = Concatenate({ExpectedHeaderForBound(100), instruction});
  InSequence calls_expected_in_specific_order;
  EXPECT_HEADER(100).WillOnce(Return(SPV_SUCCESS));
  const auto operands = std::vector<spv_parsed_operand_t>{
      MakeSimpleOperand(1, SPV_OPERAND_TYPE_ID),
      MakeLiteralStringOperand(2, static_cast<uint16_t>(str_words.size()))};
  EXPECT_CALL(client_,
              Instruction(ParsedInstruction(spv_parsed_instruction_t{
                  instruction.data(), static_cast<uint16_t>(instruction.size()),
                  SpvOpName, SPV_EXT_INST_TYPE_NONE, 0 /*type id*/,
                  0 /* No result id for OpName*/, operands.data(),
                  static_cast<uint16_t>(operands.size())})))
      .WillOnce(Return(SPV_SUCCESS));
  // Since we are actually checking the output, don't test the
  // endian-swapped version.
  Parse(words, SPV_SUCCESS, false);
  EXPECT_EQ(nullptr, diagnostic_);
}

// Checks for non-zero values for the result_id and ext_inst_type members
// spv_parsed_instruction_t.
TEST_F(BinaryParseTest, ExtendedInstruction) {
  const auto words = CompileSuccessfully(
      "%extcl = OpExtInstImport \"OpenCL.std\" "
      "%result = OpExtInst %float %extcl sqrt %x");
  EXPECT_HEADER(5).WillOnce(Return(SPV_SUCCESS));
  EXPECT_CALL(client_, Instruction(_)).WillOnce(Return(SPV_SUCCESS));
  // We're only interested in the second call to Instruction():
  const auto operands = std::vector<spv_parsed_operand_t>{
      MakeSimpleOperand(1, SPV_OPERAND_TYPE_TYPE_ID),
      MakeSimpleOperand(2, SPV_OPERAND_TYPE_RESULT_ID),
      MakeSimpleOperand(3,
                        SPV_OPERAND_TYPE_ID),  // Extended instruction set Id
      MakeSimpleOperand(4, SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER),
      MakeSimpleOperand(5, SPV_OPERAND_TYPE_ID),  // Id of the argument
  };
  const auto instruction = MakeInstruction(
      SpvOpExtInst,
      {2, 3, 1, static_cast<uint32_t>(OpenCLLIB::Entrypoints::Sqrt), 4});
  EXPECT_CALL(client_,
              Instruction(ParsedInstruction(spv_parsed_instruction_t{
                  instruction.data(), static_cast<uint16_t>(instruction.size()),
                  SpvOpExtInst, SPV_EXT_INST_TYPE_OPENCL_STD, 2 /*type id*/,
                  3 /*result id*/, operands.data(),
                  static_cast<uint16_t>(operands.size())})))
      .WillOnce(Return(SPV_SUCCESS));
  // Since we are actually checking the output, don't test the
  // endian-swapped version.
  Parse(words, SPV_SUCCESS, false);
  EXPECT_EQ(nullptr, diagnostic_);
}

// A binary parser diagnostic test case where we provide the words array
// pointer and word count explicitly.
struct WordsAndCountDiagnosticCase {
  const uint32_t* words;
  size_t num_words;
  std::string expected_diagnostic;
};

using BinaryParseWordsAndCountDiagnosticTest = spvtest::TextToBinaryTestBase<
    ::testing::TestWithParam<WordsAndCountDiagnosticCase>>;

TEST_P(BinaryParseWordsAndCountDiagnosticTest, WordAndCountCases) {
  EXPECT_EQ(
      SPV_ERROR_INVALID_BINARY,
      spvBinaryParse(ScopedContext().context, nullptr, GetParam().words,
                     GetParam().num_words, nullptr, nullptr, &diagnostic));
  ASSERT_NE(nullptr, diagnostic);
  EXPECT_THAT(diagnostic->error, Eq(GetParam().expected_diagnostic));
}

INSTANTIATE_TEST_SUITE_P(
    BinaryParseDiagnostic, BinaryParseWordsAndCountDiagnosticTest,
    ::testing::ValuesIn(std::vector<WordsAndCountDiagnosticCase>{
        {nullptr, 0, "Missing module."},
        {kHeaderForBound1, 0,
         "Module has incomplete header: only 0 words instead of 5"},
        {kHeaderForBound1, 1,
         "Module has incomplete header: only 1 words instead of 5"},
        {kHeaderForBound1, 2,
         "Module has incomplete header: only 2 words instead of 5"},
        {kHeaderForBound1, 3,
         "Module has incomplete header: only 3 words instead of 5"},
        {kHeaderForBound1, 4,
         "Module has incomplete header: only 4 words instead of 5"},
    }));

// A binary parser diagnostic test case where a vector of words is
// provided.  We'll use this to express cases that can't be created
// via the assembler.  Either we want to make a malformed instruction,
// or an invalid case the assembler would reject.
struct WordVectorDiagnosticCase {
  std::vector<uint32_t> words;
  std::string expected_diagnostic;
};

using BinaryParseWordVectorDiagnosticTest = spvtest::TextToBinaryTestBase<
    ::testing::TestWithParam<WordVectorDiagnosticCase>>;

TEST_P(BinaryParseWordVectorDiagnosticTest, WordVectorCases) {
  const auto& words = GetParam().words;
  EXPECT_THAT(spvBinaryParse(ScopedContext().context, nullptr, words.data(),
                             words.size(), nullptr, nullptr, &diagnostic),
              AnyOf(SPV_ERROR_INVALID_BINARY, SPV_ERROR_INVALID_ID));
  ASSERT_NE(nullptr, diagnostic);
  EXPECT_THAT(diagnostic->error, Eq(GetParam().expected_diagnostic));
}

INSTANTIATE_TEST_SUITE_P(
    BinaryParseDiagnostic, BinaryParseWordVectorDiagnosticTest,
    ::testing::ValuesIn(std::vector<WordVectorDiagnosticCase>{
        {Concatenate({ExpectedHeaderForBound(1), {spvOpcodeMake(0, SpvOpNop)}}),
         "Invalid instruction word count: 0"},
        {Concatenate(
             {ExpectedHeaderForBound(1),
              {spvOpcodeMake(1, static_cast<SpvOp>(
                                    std::numeric_limits<uint16_t>::max()))}}),
         "Invalid opcode: 65535"},
        {Concatenate({ExpectedHeaderForBound(1),
                      MakeInstruction(SpvOpNop, {42})}),
         "Invalid instruction OpNop starting at word 5: expected "
         "no more operands after 1 words, but stated word count is 2."},
        // Supply several more unexpectd words.
        {Concatenate({ExpectedHeaderForBound(1),
                      MakeInstruction(SpvOpNop, {42, 43, 44, 45, 46, 47})}),
         "Invalid instruction OpNop starting at word 5: expected "
         "no more operands after 1 words, but stated word count is 7."},
        {Concatenate({ExpectedHeaderForBound(1),
                      MakeInstruction(SpvOpTypeVoid, {1, 2})}),
         "Invalid instruction OpTypeVoid starting at word 5: expected "
         "no more operands after 2 words, but stated word count is 3."},
        {Concatenate({ExpectedHeaderForBound(1),
                      MakeInstruction(SpvOpTypeVoid, {1, 2, 5, 9, 10})}),
         "Invalid instruction OpTypeVoid starting at word 5: expected "
         "no more operands after 2 words, but stated word count is 6."},
        {Concatenate({ExpectedHeaderForBound(1),
                      MakeInstruction(SpvOpTypeInt, {1, 32, 1, 9})}),
         "Invalid instruction OpTypeInt starting at word 5: expected "
         "no more operands after 4 words, but stated word count is 5."},
        {Concatenate({ExpectedHeaderForBound(1),
                      MakeInstruction(SpvOpTypeInt, {1})}),
         "End of input reached while decoding OpTypeInt starting at word 5:"
         " expected more operands after 2 words."},

        // Check several cases for running off the end of input.

        // Detect a missing single word operand.
        {Concatenate({ExpectedHeaderForBound(1),
                      {spvOpcodeMake(2, SpvOpTypeStruct)}}),
         "End of input reached while decoding OpTypeStruct starting at word"
         " 5: missing result ID operand at word offset 1."},
        // Detect this a missing a multi-word operand to OpConstant.
        // We also lie and say the OpConstant instruction has 5 words when
        // it only has 3.  Corresponds to something like this:
        //    %1 = OpTypeInt 64 0
        //    %2 = OpConstant %1 <missing>
        {Concatenate({ExpectedHeaderForBound(3),
                      {MakeInstruction(SpvOpTypeInt, {1, 64, 0})},
                      {spvOpcodeMake(5, SpvOpConstant), 1, 2}}),
         "End of input reached while decoding OpConstant starting at word"
         " 9: missing possibly multi-word literal number operand at word "
         "offset 3."},
        // Detect when we provide only one word from the 64-bit literal,
        // and again lie about the number of words in the instruction.
        {Concatenate({ExpectedHeaderForBound(3),
                      {MakeInstruction(SpvOpTypeInt, {1, 64, 0})},
                      {spvOpcodeMake(5, SpvOpConstant), 1, 2, 42}}),
         "End of input reached while decoding OpConstant starting at word"
         " 9: truncated possibly multi-word literal number operand at word "
         "offset 3."},
        // Detect when a required string operand is missing.
        // Also, lie about the length of the instruction.
        {Concatenate({ExpectedHeaderForBound(3),
                      {spvOpcodeMake(3, SpvOpString), 1}}),
         "End of input reached while decoding OpString starting at word"
         " 5: missing literal string operand at word offset 2."},
        // Detect when a required string operand is truncated: it's missing
        // a null terminator.  Catching the error avoids a buffer overrun.
        {Concatenate({ExpectedHeaderForBound(3),
                      {spvOpcodeMake(4, SpvOpString), 1, 0x41414141,
                       0x41414141}}),
         "End of input reached while decoding OpString starting at word"
         " 5: truncated literal string operand at word offset 2."},
        // Detect when an optional string operand is truncated: it's missing
        // a null terminator.  Catching the error avoids a buffer overrun.
        // (It is valid for an optional string operand to be absent.)
        {Concatenate({ExpectedHeaderForBound(3),
                      {spvOpcodeMake(6, SpvOpSource),
                       static_cast<uint32_t>(SpvSourceLanguageOpenCL_C), 210,
                       1 /* file id */,
                       /*start of string*/ 0x41414141, 0x41414141}}),
         "End of input reached while decoding OpSource starting at word"
         " 5: truncated literal string operand at word offset 4."},

        // (End of input exhaustion test cases.)

        // In this case the instruction word count is too small, where
        // it would truncate a multi-word operand to OpConstant.
        {Concatenate({ExpectedHeaderForBound(3),
                      {MakeInstruction(SpvOpTypeInt, {1, 64, 0})},
                      {spvOpcodeMake(4, SpvOpConstant), 1, 2, 44, 44}}),
         "Invalid word count: OpConstant starting at word 9 says it has 4"
         " words, but found 5 words instead."},
        // Word count is to small, where it would truncate a literal string.
        {Concatenate({ExpectedHeaderForBound(2),
                      {spvOpcodeMake(3, SpvOpString), 1, 0x41414141, 0}}),
         "Invalid word count: OpString starting at word 5 says it has 3"
         " words, but found 4 words instead."},
        // Word count is too large.  The string terminates before the last
        // word.
        {Concatenate({ExpectedHeaderForBound(2),
                      {spvOpcodeMake(4, SpvOpString), 1 /* result id */},
                      MakeVector("abc"),
                      {0 /* this word does not belong*/}}),
         "Invalid instruction OpString starting at word 5: expected no more"
         " operands after 3 words, but stated word count is 4."},
        // Word count is too large.  There are too many words after the string
        // literal.  A linkage attribute decoration is the only case in SPIR-V
        // where a string operand is followed by another operand.
        {Concatenate({ExpectedHeaderForBound(2),
                      {spvOpcodeMake(6, SpvOpDecorate), 1 /* target id */,
                       static_cast<uint32_t>(SpvDecorationLinkageAttributes)},
                      MakeVector("abc"),
                      {static_cast<uint32_t>(SpvLinkageTypeImport),
                       0 /* does not belong */}}),
         "Invalid instruction OpDecorate starting at word 5: expected no more"
         " operands after 5 words, but stated word count is 6."},
        // Like the previous case, but with 5 extra words.
        {Concatenate({ExpectedHeaderForBound(2),
                      {spvOpcodeMake(10, SpvOpDecorate), 1 /* target id */,
                       static_cast<uint32_t>(SpvDecorationLinkageAttributes)},
                      MakeVector("abc"),
                      {static_cast<uint32_t>(SpvLinkageTypeImport),
                       /* don't belong */ 0, 1, 2, 3, 4}}),
         "Invalid instruction OpDecorate starting at word 5: expected no more"
         " operands after 5 words, but stated word count is 10."},
        // Like the previous two cases, but with OpMemberDecorate.
        {Concatenate({ExpectedHeaderForBound(2),
                      {spvOpcodeMake(7, SpvOpMemberDecorate), 1 /* target id */,
                       42 /* member index */,
                       static_cast<uint32_t>(SpvDecorationLinkageAttributes)},
                      MakeVector("abc"),
                      {static_cast<uint32_t>(SpvLinkageTypeImport),
                       0 /* does not belong */}}),
         "Invalid instruction OpMemberDecorate starting at word 5: expected no"
         " more operands after 6 words, but stated word count is 7."},
        {Concatenate({ExpectedHeaderForBound(2),
                      {spvOpcodeMake(11, SpvOpMemberDecorate),
                       1 /* target id */, 42 /* member index */,
                       static_cast<uint32_t>(SpvDecorationLinkageAttributes)},
                      MakeVector("abc"),
                      {static_cast<uint32_t>(SpvLinkageTypeImport),
                       /* don't belong */ 0, 1, 2, 3, 4}}),
         "Invalid instruction OpMemberDecorate starting at word 5: expected no"
         " more operands after 6 words, but stated word count is 11."},
        // Word count is too large.  There should be no more words
        // after the RelaxedPrecision decoration.
        {Concatenate({ExpectedHeaderForBound(2),
                      {spvOpcodeMake(4, SpvOpDecorate), 1 /* target id */,
                       static_cast<uint32_t>(SpvDecorationRelaxedPrecision),
                       0 /* does not belong */}}),
         "Invalid instruction OpDecorate starting at word 5: expected no"
         " more operands after 3 words, but stated word count is 4."},
        // Word count is too large.  There should be only one word after
        // the SpecId decoration enum word.
        {Concatenate({ExpectedHeaderForBound(2),
                      {spvOpcodeMake(5, SpvOpDecorate), 1 /* target id */,
                       static_cast<uint32_t>(SpvDecorationSpecId),
                       42 /* the spec id */, 0 /* does not belong */}}),
         "Invalid instruction OpDecorate starting at word 5: expected no"
         " more operands after 4 words, but stated word count is 5."},
        {Concatenate({ExpectedHeaderForBound(2),
                      {spvOpcodeMake(2, SpvOpTypeVoid), 0}}),
         "Error: Result Id is 0"},
        {Concatenate({
             ExpectedHeaderForBound(2),
             {spvOpcodeMake(2, SpvOpTypeVoid), 1},
             {spvOpcodeMake(2, SpvOpTypeBool), 1},
         }),
         "Id 1 is defined more than once"},
        {Concatenate({ExpectedHeaderForBound(3),
                      MakeInstruction(SpvOpExtInst, {2, 3, 100, 4, 5})}),
         "OpExtInst set Id 100 does not reference an OpExtInstImport result "
         "Id"},
        {Concatenate({ExpectedHeaderForBound(101),
                      MakeInstruction(SpvOpExtInstImport, {100},
                                      MakeVector("OpenCL.std")),
                      // OpenCL cos is #14
                      MakeInstruction(SpvOpExtInst, {2, 3, 100, 14, 5, 999})}),
         "Invalid instruction OpExtInst starting at word 10: expected no "
         "more operands after 6 words, but stated word count is 7."},
        // In this case, the OpSwitch selector refers to an invalid ID.
        {Concatenate({ExpectedHeaderForBound(3),
                      MakeInstruction(SpvOpSwitch, {1, 2, 42, 3})}),
         "Invalid OpSwitch: selector id 1 has no type"},
        // In this case, the OpSwitch selector refers to an ID that has
        // no type.
        {Concatenate({ExpectedHeaderForBound(3),
                      MakeInstruction(SpvOpLabel, {1}),
                      MakeInstruction(SpvOpSwitch, {1, 2, 42, 3})}),
         "Invalid OpSwitch: selector id 1 has no type"},
        {Concatenate({ExpectedHeaderForBound(3),
                      MakeInstruction(SpvOpTypeInt, {1, 32, 0}),
                      MakeInstruction(SpvOpSwitch, {1, 3, 42, 3})}),
         "Invalid OpSwitch: selector id 1 is a type, not a value"},
        {Concatenate({ExpectedHeaderForBound(3),
                      MakeInstruction(SpvOpTypeFloat, {1, 32}),
                      MakeInstruction(SpvOpConstant, {1, 2, 0x78f00000}),
                      MakeInstruction(SpvOpSwitch, {2, 3, 42, 3})}),
         "Invalid OpSwitch: selector id 2 is not a scalar integer"},
        {Concatenate({ExpectedHeaderForBound(3),
                      MakeInstruction(SpvOpExtInstImport, {1},
                                      MakeVector("invalid-import"))}),
         "Invalid extended instruction import 'invalid-import'"},
        {Concatenate({
             ExpectedHeaderForBound(3),
             MakeInstruction(SpvOpTypeInt, {1, 32, 0}),
             MakeInstruction(SpvOpConstant, {2, 2, 42}),
         }),
         "Type Id 2 is not a type"},
        {Concatenate({
             ExpectedHeaderForBound(3),
             MakeInstruction(SpvOpTypeBool, {1}),
             MakeInstruction(SpvOpConstant, {1, 2, 42}),
         }),
         "Type Id 1 is not a scalar numeric type"},
    }));

// A binary parser diagnostic case generated from an assembly text input.
struct AssemblyDiagnosticCase {
  std::string assembly;
  std::string expected_diagnostic;
};

using BinaryParseAssemblyDiagnosticTest = spvtest::TextToBinaryTestBase<
    ::testing::TestWithParam<AssemblyDiagnosticCase>>;

TEST_P(BinaryParseAssemblyDiagnosticTest, AssemblyCases) {
  auto words = CompileSuccessfully(GetParam().assembly);
  EXPECT_THAT(spvBinaryParse(ScopedContext().context, nullptr, words.data(),
                             words.size(), nullptr, nullptr, &diagnostic),
              AnyOf(SPV_ERROR_INVALID_BINARY, SPV_ERROR_INVALID_ID));
  ASSERT_NE(nullptr, diagnostic);
  EXPECT_THAT(diagnostic->error, Eq(GetParam().expected_diagnostic));
}

INSTANTIATE_TEST_SUITE_P(
    BinaryParseDiagnostic, BinaryParseAssemblyDiagnosticTest,
    ::testing::ValuesIn(std::vector<AssemblyDiagnosticCase>{
        {"%1 = OpConstant !0 42", "Error: Type Id is 0"},
        // A required id is 0.
        {"OpName !0 \"foo\"", "Id is 0"},
        // An optional id is 0, in this case the optional
        // initializer.
        {"%2 = OpVariable %1 CrossWorkgroup !0", "Id is 0"},
        {"OpControlBarrier !0 %1 %2", "scope ID is 0"},
        {"OpControlBarrier %1 !0 %2", "scope ID is 0"},
        {"OpControlBarrier %1 %2 !0", "memory semantics ID is 0"},
        {"%import = OpExtInstImport \"GLSL.std.450\" "
         "%result = OpExtInst %type %import !999999 %x",
         "Invalid extended instruction number: 999999"},
        {"%2 = OpSpecConstantOp %1 !1000 %2",
         "Invalid OpSpecConstantOp opcode: 1000"},
        {"OpCapability !9999", "Invalid capability operand: 9999"},
        {"OpSource !9999 100", "Invalid source language operand: 9999"},
        {"OpEntryPoint !9999", "Invalid execution model operand: 9999"},
        {"OpMemoryModel !9999", "Invalid addressing model operand: 9999"},
        {"OpMemoryModel Logical !9999", "Invalid memory model operand: 9999"},
        {"OpExecutionMode %1 !9999", "Invalid execution mode operand: 9999"},
        {"OpTypeForwardPointer %1 !9999",
         "Invalid storage class operand: 9999"},
        {"%2 = OpTypeImage %1 !9999", "Invalid dimensionality operand: 9999"},
        {"%2 = OpTypeImage %1 1D 0 0 0 0 !9999",
         "Invalid image format operand: 9999"},
        {"OpDecorate %1 FPRoundingMode !9999",
         "Invalid floating-point rounding mode operand: 9999"},
        {"OpDecorate %1 LinkageAttributes \"C\" !9999",
         "Invalid linkage type operand: 9999"},
        {"%1 = OpTypePipe !9999", "Invalid access qualifier operand: 9999"},
        {"OpDecorate %1 FuncParamAttr !9999",
         "Invalid function parameter attribute operand: 9999"},
        {"OpDecorate %1 !9999", "Invalid decoration operand: 9999"},
        {"OpDecorate %1 BuiltIn !9999", "Invalid built-in operand: 9999"},
        {"%2 = OpGroupIAdd %1 %3 !9999",
         "Invalid group operation operand: 9999"},
        {"OpDecorate %1 FPFastMathMode !63",
         "Invalid floating-point fast math mode operand: 63 has invalid mask "
         "component 32"},
        {"%2 = OpFunction %2 !31",
         "Invalid function control operand: 31 has invalid mask component 16"},
        {"OpLoopMerge %1 %2 !1027",
         "Invalid loop control operand: 1027 has invalid mask component 1024"},
        {"%2 = OpImageFetch %1 %image %coord !32770",
         "Invalid image operand: 32770 has invalid mask component 32768"},
        {"OpSelectionMerge %1 !7",
         "Invalid selection control operand: 7 has invalid mask component 4"},
    }));

}  // namespace
}  // namespace spvtools
