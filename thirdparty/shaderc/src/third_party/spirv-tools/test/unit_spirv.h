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

#ifndef TEST_UNIT_SPIRV_H_
#define TEST_UNIT_SPIRV_H_

#include <stdint.h>

#include <iomanip>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "source/assembly_grammar.h"
#include "source/binary.h"
#include "source/diagnostic.h"
#include "source/enum_set.h"
#include "source/opcode.h"
#include "source/spirv_endian.h"
#include "source/text.h"
#include "source/text_handler.h"
#include "source/val/validate.h"
#include "spirv-tools/libspirv.h"

#ifdef __ANDROID__
#include <sstream>
namespace std {
template <typename T>
std::string to_string(const T& val) {
  std::ostringstream os;
  os << val;
  return os.str();
}
}  // namespace std
#endif

// Determine endianness & predicate tests on it
enum {
  I32_ENDIAN_LITTLE = 0x03020100ul,
  I32_ENDIAN_BIG = 0x00010203ul,
};

static const union {
  unsigned char bytes[4];
  uint32_t value;
} o32_host_order = {{0, 1, 2, 3}};
#define I32_ENDIAN_HOST (o32_host_order.value)

// A namespace for utilities used in SPIR-V Tools unit tests.
namespace spvtest {

class WordVector;

// Emits the given word vector to the given stream.
// This function can be used by the gtest value printer.
void PrintTo(const WordVector& words, ::std::ostream* os);

// A proxy class to allow us to easily write out vectors of SPIR-V words.
class WordVector {
 public:
  explicit WordVector(const std::vector<uint32_t>& val) : value_(val) {}
  explicit WordVector(const spv_binary_t& binary)
      : value_(binary.code, binary.code + binary.wordCount) {}

  // Returns the underlying vector.
  const std::vector<uint32_t>& value() const { return value_; }

  // Returns the string representation of this word vector.
  std::string str() const {
    std::ostringstream os;
    PrintTo(*this, &os);
    return os.str();
  }

 private:
  const std::vector<uint32_t> value_;
};

inline void PrintTo(const WordVector& words, ::std::ostream* os) {
  size_t count = 0;
  const auto saved_flags = os->flags();
  const auto saved_fill = os->fill();
  for (uint32_t value : words.value()) {
    *os << "0x" << std::setw(8) << std::setfill('0') << std::hex << value
        << " ";
    if (count++ % 8 == 7) {
      *os << std::endl;
    }
  }
  os->flags(saved_flags);
  os->fill(saved_fill);
}

// Returns a vector of words representing a single instruction with the
// given opcode and operand words as a vector.
inline std::vector<uint32_t> MakeInstruction(
    SpvOp opcode, const std::vector<uint32_t>& args) {
  std::vector<uint32_t> result{
      spvOpcodeMake(uint16_t(args.size() + 1), opcode)};
  result.insert(result.end(), args.begin(), args.end());
  return result;
}

// Returns a vector of words representing a single instruction with the
// given opcode and whose operands are the concatenation of the two given
// argument lists.
inline std::vector<uint32_t> MakeInstruction(
    SpvOp opcode, std::vector<uint32_t> args,
    const std::vector<uint32_t>& extra_args) {
  args.insert(args.end(), extra_args.begin(), extra_args.end());
  return MakeInstruction(opcode, args);
}

// Returns the vector of words representing the concatenation
// of all input vectors.
inline std::vector<uint32_t> Concatenate(
    const std::vector<std::vector<uint32_t>>& instructions) {
  std::vector<uint32_t> result;
  for (const auto& instruction : instructions) {
    result.insert(result.end(), instruction.begin(), instruction.end());
  }
  return result;
}

// Encodes a string as a sequence of words, using the SPIR-V encoding.
inline std::vector<uint32_t> MakeVector(std::string input) {
  std::vector<uint32_t> result;
  uint32_t word = 0;
  size_t num_bytes = input.size();
  // SPIR-V strings are null-terminated.  The byte_index == num_bytes
  // case is used to push the terminating null byte.
  for (size_t byte_index = 0; byte_index <= num_bytes; byte_index++) {
    const auto new_byte =
        (byte_index < num_bytes ? uint8_t(input[byte_index]) : uint8_t(0));
    word |= (new_byte << (8 * (byte_index % sizeof(uint32_t))));
    if (3 == (byte_index % sizeof(uint32_t))) {
      result.push_back(word);
      word = 0;
    }
  }
  // Emit a trailing partial word.
  if ((num_bytes + 1) % sizeof(uint32_t)) {
    result.push_back(word);
  }
  return result;
}

// A type for easily creating spv_text_t values, with an implicit conversion to
// spv_text.
struct AutoText {
  explicit AutoText(const std::string& value)
      : str(value), text({str.data(), str.size()}) {}
  operator spv_text() { return &text; }
  std::string str;
  spv_text_t text;
};

// An example case for an enumerated value, optionally with operands.
template <typename E>
class EnumCase {
 public:
  EnumCase() = default;  // Required by ::testing::Combine().
  EnumCase(E val, std::string enum_name, std::vector<uint32_t> ops = {})
      : enum_value_(val), name_(enum_name), operands_(ops) {}
  // Returns the enum value as a uint32_t.
  uint32_t value() const { return static_cast<uint32_t>(enum_value_); }
  // Returns the name of the enumerant.
  const std::string& name() const { return name_; }
  // Returns a reference to the operands.
  const std::vector<uint32_t>& operands() const { return operands_; }

 private:
  E enum_value_;
  std::string name_;
  std::vector<uint32_t> operands_;
};

// Returns a string with num_4_byte_chars Unicode characters,
// each of which has a 4-byte UTF-8 encoding.
inline std::string MakeLongUTF8String(size_t num_4_byte_chars) {
  // An example of a longest valid UTF-8 character.
  // Be explicit about the character type because Microsoft compilers can
  // otherwise interpret the character string as being over wide (16-bit)
  // characters.  Ideally, we would just use a C++11 UTF-8 string literal,
  // but we want to support older Microsoft compilers.
  const std::basic_string<char> earth_africa("\xF0\x9F\x8C\x8D");
  EXPECT_EQ(4u, earth_africa.size());

  std::string result;
  result.reserve(num_4_byte_chars * 4);
  for (size_t i = 0; i < num_4_byte_chars; i++) {
    result += earth_africa;
  }
  EXPECT_EQ(4 * num_4_byte_chars, result.size());
  return result;
}

// Returns a vector of all valid target environment enums.
inline std::vector<spv_target_env> AllTargetEnvironments() {
  return {
      SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
      SPV_ENV_OPENCL_1_2,    SPV_ENV_OPENCL_EMBEDDED_1_2,
      SPV_ENV_OPENCL_2_0,    SPV_ENV_OPENCL_EMBEDDED_2_0,
      SPV_ENV_OPENCL_2_1,    SPV_ENV_OPENCL_EMBEDDED_2_1,
      SPV_ENV_OPENCL_2_2,    SPV_ENV_OPENCL_EMBEDDED_2_2,
      SPV_ENV_VULKAN_1_0,    SPV_ENV_OPENGL_4_0,
      SPV_ENV_OPENGL_4_1,    SPV_ENV_OPENGL_4_2,
      SPV_ENV_OPENGL_4_3,    SPV_ENV_OPENGL_4_5,
      SPV_ENV_UNIVERSAL_1_2, SPV_ENV_UNIVERSAL_1_3,
      SPV_ENV_VULKAN_1_1,    SPV_ENV_WEBGPU_0,
  };
}

// Returns the capabilities in a CapabilitySet as an ordered vector.
inline std::vector<SpvCapability> ElementsIn(
    const spvtools::CapabilitySet& capabilities) {
  std::vector<SpvCapability> result;
  capabilities.ForEach([&result](SpvCapability c) { result.push_back(c); });
  return result;
}

}  // namespace spvtest
#endif  // TEST_UNIT_SPIRV_H_
