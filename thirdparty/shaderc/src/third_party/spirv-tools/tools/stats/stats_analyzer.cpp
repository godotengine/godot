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

#include "tools/stats/stats_analyzer.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "source/comp/markv_model.h"
#include "source/enum_string_mapping.h"
#include "source/latest_version_spirv_header.h"
#include "source/opcode.h"
#include "source/operand.h"
#include "source/spirv_constant.h"

namespace spvtools {
namespace stats {
namespace {

// Signals that the value is not in the coding scheme and a fallback method
// needs to be used.
const uint64_t kMarkvNoneOfTheAbove =
    comp::MarkvModel::GetMarkvNoneOfTheAbove();

std::string GetVersionString(uint32_t word) {
  std::stringstream ss;
  ss << "Version " << SPV_SPIRV_VERSION_MAJOR_PART(word) << "."
     << SPV_SPIRV_VERSION_MINOR_PART(word);
  return ss.str();
}

std::string GetGeneratorString(uint32_t word) {
  return spvGeneratorStr(SPV_GENERATOR_TOOL_PART(word));
}

std::string GetOpcodeString(uint32_t word) {
  return spvOpcodeString(static_cast<SpvOp>(word));
}

std::string GetCapabilityString(uint32_t word) {
  return CapabilityToString(static_cast<SpvCapability>(word));
}

template <class T>
std::string KeyIsLabel(T key) {
  std::stringstream ss;
  ss << key;
  return ss.str();
}

template <class Key>
std::unordered_map<Key, double> GetRecall(
    const std::unordered_map<Key, uint32_t>& hist, uint64_t total) {
  std::unordered_map<Key, double> freq;
  for (const auto& pair : hist) {
    const double frequency =
        static_cast<double>(pair.second) / static_cast<double>(total);
    freq.emplace(pair.first, frequency);
  }
  return freq;
}

template <class Key>
std::unordered_map<Key, double> GetPrevalence(
    const std::unordered_map<Key, uint32_t>& hist) {
  uint64_t total = 0;
  for (const auto& pair : hist) {
    total += pair.second;
  }

  return GetRecall(hist, total);
}

// Writes |freq| to |out| sorted by frequency in the following format:
// LABEL3 70%
// LABEL1 20%
// LABEL2 10%
// |label_from_key| is used to convert |Key| to label.
template <class Key>
void WriteFreq(std::ostream& out, const std::unordered_map<Key, double>& freq,
               std::string (*label_from_key)(Key)) {
  std::vector<std::pair<Key, double>> sorted_freq(freq.begin(), freq.end());
  std::sort(sorted_freq.begin(), sorted_freq.end(),
            [](const std::pair<Key, double>& left,
               const std::pair<Key, double>& right) {
              return left.second > right.second;
            });

  for (const auto& pair : sorted_freq) {
    if (pair.second < 0.001) break;
    out << label_from_key(pair.first) << " " << pair.second * 100.0 << "%"
        << std::endl;
  }
}

}  // namespace

StatsAnalyzer::StatsAnalyzer(const SpirvStats& stats) : stats_(stats) {
  num_modules_ = 0;
  for (const auto& pair : stats_.version_hist) {
    num_modules_ += pair.second;
  }

  version_freq_ = GetRecall(stats_.version_hist, num_modules_);
  generator_freq_ = GetRecall(stats_.generator_hist, num_modules_);
  capability_freq_ = GetRecall(stats_.capability_hist, num_modules_);
  extension_freq_ = GetRecall(stats_.extension_hist, num_modules_);
  opcode_freq_ = GetPrevalence(stats_.opcode_hist);
}

void StatsAnalyzer::WriteVersion(std::ostream& out) {
  WriteFreq(out, version_freq_, GetVersionString);
}

void StatsAnalyzer::WriteGenerator(std::ostream& out) {
  WriteFreq(out, generator_freq_, GetGeneratorString);
}

void StatsAnalyzer::WriteCapability(std::ostream& out) {
  WriteFreq(out, capability_freq_, GetCapabilityString);
}

void StatsAnalyzer::WriteExtension(std::ostream& out) {
  WriteFreq(out, extension_freq_, KeyIsLabel);
}

void StatsAnalyzer::WriteOpcode(std::ostream& out) {
  out << "Total unique opcodes used: " << opcode_freq_.size() << std::endl;
  WriteFreq(out, opcode_freq_, GetOpcodeString);
}

void StatsAnalyzer::WriteConstantLiterals(std::ostream& out) {
  out << "Constant literals" << std::endl;

  out << "Float 32" << std::endl;
  WriteFreq(out, GetPrevalence(stats_.f32_constant_hist), KeyIsLabel);

  out << std::endl << "Float 64" << std::endl;
  WriteFreq(out, GetPrevalence(stats_.f64_constant_hist), KeyIsLabel);

  out << std::endl << "Unsigned int 16" << std::endl;
  WriteFreq(out, GetPrevalence(stats_.u16_constant_hist), KeyIsLabel);

  out << std::endl << "Signed int 16" << std::endl;
  WriteFreq(out, GetPrevalence(stats_.s16_constant_hist), KeyIsLabel);

  out << std::endl << "Unsigned int 32" << std::endl;
  WriteFreq(out, GetPrevalence(stats_.u32_constant_hist), KeyIsLabel);

  out << std::endl << "Signed int 32" << std::endl;
  WriteFreq(out, GetPrevalence(stats_.s32_constant_hist), KeyIsLabel);

  out << std::endl << "Unsigned int 64" << std::endl;
  WriteFreq(out, GetPrevalence(stats_.u64_constant_hist), KeyIsLabel);

  out << std::endl << "Signed int 64" << std::endl;
  WriteFreq(out, GetPrevalence(stats_.s64_constant_hist), KeyIsLabel);
}

void StatsAnalyzer::WriteOpcodeMarkov(std::ostream& out) {
  if (stats_.opcode_markov_hist.empty()) return;

  const std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint32_t>>&
      cue_to_hist = stats_.opcode_markov_hist[0];

  // Sort by prevalence of the opcodes in opcode_freq_ (descending).
  std::vector<std::pair<uint32_t, std::unordered_map<uint32_t, uint32_t>>>
      sorted_cue_to_hist(cue_to_hist.begin(), cue_to_hist.end());
  std::sort(
      sorted_cue_to_hist.begin(), sorted_cue_to_hist.end(),
      [this](const std::pair<uint32_t, std::unordered_map<uint32_t, uint32_t>>&
                 left,
             const std::pair<uint32_t, std::unordered_map<uint32_t, uint32_t>>&
                 right) {
        const double lf = opcode_freq_[left.first];
        const double rf = opcode_freq_[right.first];
        if (lf == rf) return right.first > left.first;
        return lf > rf;
      });

  for (const auto& kv : sorted_cue_to_hist) {
    const uint32_t cue = kv.first;
    const double kFrequentEnoughToAnalyze = 0.0001;
    if (opcode_freq_[cue] < kFrequentEnoughToAnalyze) continue;

    const std::unordered_map<uint32_t, uint32_t>& hist = kv.second;

    uint32_t total = 0;
    for (const auto& pair : hist) {
      total += pair.second;
    }

    std::vector<std::pair<uint32_t, uint32_t>> sorted_hist(hist.begin(),
                                                           hist.end());
    std::sort(sorted_hist.begin(), sorted_hist.end(),
              [](const std::pair<uint32_t, uint32_t>& left,
                 const std::pair<uint32_t, uint32_t>& right) {
                if (left.second == right.second)
                  return right.first > left.first;
                return left.second > right.second;
              });

    for (const auto& pair : sorted_hist) {
      const double prior = opcode_freq_[pair.first];
      const double posterior =
          static_cast<double>(pair.second) / static_cast<double>(total);
      out << GetOpcodeString(cue) << " -> " << GetOpcodeString(pair.first)
          << " " << posterior * 100 << "% (base rate " << prior * 100
          << "%, pair occurrences " << pair.second << ")" << std::endl;
    }
  }
}

}  // namespace stats
}  // namespace spvtools
