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

#ifndef SOURCE_COMP_MARKV_LOGGER_H_
#define SOURCE_COMP_MARKV_LOGGER_H_

#include "source/comp/markv.h"

namespace spvtools {
namespace comp {

class MarkvLogger {
 public:
  MarkvLogger(MarkvLogConsumer log_consumer, MarkvDebugConsumer debug_consumer)
      : log_consumer_(log_consumer), debug_consumer_(debug_consumer) {}

  void AppendText(const std::string& str) {
    Append(str);
    use_delimiter_ = false;
  }

  void AppendTextNewLine(const std::string& str) {
    Append(str);
    Append("\n");
    use_delimiter_ = false;
  }

  void AppendBitSequence(const std::string& str) {
    if (debug_consumer_) instruction_bits_ << str;
    if (use_delimiter_) Append("-");
    Append(str);
    use_delimiter_ = true;
  }

  void AppendWhitespaces(size_t num) {
    Append(std::string(num, ' '));
    use_delimiter_ = false;
  }

  void NewLine() {
    Append("\n");
    use_delimiter_ = false;
  }

  bool DebugInstruction(const spv_parsed_instruction_t& inst) {
    bool result = true;
    if (debug_consumer_) {
      result = debug_consumer_(
          std::vector<uint32_t>(inst.words, inst.words + inst.num_words),
          instruction_bits_.str(), instruction_comment_.str());
      instruction_bits_.str(std::string());
      instruction_comment_.str(std::string());
    }
    return result;
  }

 private:
  MarkvLogger(const MarkvLogger&) = delete;
  MarkvLogger(MarkvLogger&&) = delete;
  MarkvLogger& operator=(const MarkvLogger&) = delete;
  MarkvLogger& operator=(MarkvLogger&&) = delete;

  void Append(const std::string& str) {
    if (log_consumer_) log_consumer_(str);
    if (debug_consumer_) instruction_comment_ << str;
  }

  MarkvLogConsumer log_consumer_;
  MarkvDebugConsumer debug_consumer_;

  std::stringstream instruction_bits_;
  std::stringstream instruction_comment_;

  // If true a delimiter will be appended before the next bit sequence.
  // Used to generate outputs like: 1100-0 1110-1-1100-1-1111-0 110-0.
  bool use_delimiter_ = false;
};

}  // namespace comp
}  // namespace spvtools

#endif  // SOURCE_COMP_MARKV_LOGGER_H_
