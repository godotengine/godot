// Copyright 2017 The Effcee Authors.
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

#include "check.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "cursor.h"
#include "effcee.h"
#include "make_unique.h"
#include "to_string.h"

using Status = effcee::Result::Status;
using StringPiece = effcee::StringPiece;
using Type = effcee::Check::Type;

namespace {

// Returns a table of suffix to type mappings.
const std::vector<std::pair<StringPiece, Type>>& TypeStringTable() {
  static std::vector<std::pair<StringPiece, Type>> type_str_table{
      {"", Type::Simple},  {"-NEXT", Type::Next},   {"-SAME", Type::Same},
      {"-DAG", Type::DAG}, {"-LABEL", Type::Label}, {"-NOT", Type::Not}};
  return type_str_table;
}

// Returns the Check::Type value matching the suffix part of a check rule
// prefix.  Assumes |suffix| is valid.
Type TypeForSuffix(StringPiece suffix) {
  const auto& type_str_table = TypeStringTable();
  const auto pair_iter =
      std::find_if(type_str_table.begin(), type_str_table.end(),
                   [suffix](const std::pair<StringPiece, Type>& elem) {
                     return suffix == elem.first;
                   });
  assert(pair_iter != type_str_table.end());
  return pair_iter->second;
}
}  // namespace

namespace effcee {

int Check::Part::CountCapturingGroups() {
  if (type_ == Type::Regex) return RE2(param_).NumberOfCapturingGroups();
  if (type_ == Type::VarDef) return RE2(expression_).NumberOfCapturingGroups();
  return 0;
}

Check::Check(Type type, StringPiece param) : type_(type), param_(param) {
  parts_.push_back(make_unique<Check::Part>(Part::Type::Fixed, param));
}

bool Check::Part::MightMatch(const VarMapping& vars) const {
  return type_ != Type::VarUse ||
         vars.find(ToString(VarUseName())) != vars.end();
}

std::string Check::Part::Regex(const VarMapping& vars) const {
  switch (type_) {
    case Type::Fixed:
      return RE2::QuoteMeta(param_);
    case Type::Regex:
      return ToString(param_);
    case Type::VarDef:
      return std::string("(") + ToString(expression_) + ")";
    case Type::VarUse: {
      auto where = vars.find(ToString(VarUseName()));
      if (where != vars.end()) {
        // Return the escaped form of the current value of the variable.
        return RE2::QuoteMeta((*where).second);
      } else {
        // The variable is not yet set.  Should not get here.
        return "";
      }
    }
  }
  return "";  // Unreachable.  But we need to satisfy GCC.
}

bool Check::Matches(StringPiece* input, StringPiece* captured,
                    VarMapping* vars) const {
  if (parts_.empty()) return false;
  for (auto& part : parts_) {
    if (!part->MightMatch(*vars)) return false;
  }

  std::unordered_map<int, std::string> var_def_indices;

  std::ostringstream consume_regex;
  int num_captures = 1;  // The outer capture.
  for (auto& part : parts_) {
    consume_regex << part->Regex(*vars);
    const auto var_def_name = part->VarDefName();
    if (!var_def_name.empty()) {
      var_def_indices[num_captures++] = ToString(var_def_name);
    }
    num_captures += part->NumCapturingGroups();
  }
  std::unique_ptr<StringPiece[]> captures(new StringPiece[num_captures]);
  const bool matched = RE2(consume_regex.str())
                           .Match(*input, 0, input->size(), RE2::UNANCHORED,
                                  captures.get(), num_captures);
  if (matched) {
    *captured = captures[0];
    input->remove_prefix(captured->end() - input->begin());
    // Update the variable mapping.
    for (auto& var_def_index : var_def_indices) {
      const int index = var_def_index.first;
      (*vars)[var_def_index.second] = ToString(captures[index]);
    }
  }

  return matched;
}

namespace {
// Returns a parts list for the given pattern.  This splits out regular
// expressions as delimited by {{ and }}, and also variable uses and
// definitions.
Check::Parts PartsForPattern(StringPiece pattern) {
  Check::Parts parts;
  StringPiece fixed, regex, var;

  using Type = Check::Part::Type;

  while (!pattern.empty()) {
    const auto regex_start = pattern.find("{{");
    const auto regex_end = pattern.find("}}");
    const auto var_start = pattern.find("[[");
    const auto var_end = pattern.find("]]");
    const bool regex_exists =
        regex_start < regex_end && regex_end < StringPiece::npos;
    const bool var_exists = var_start < var_end && var_end < StringPiece::npos;

    if (regex_exists && (!var_exists || regex_start < var_start)) {
      const auto consumed =
          RE2::Consume(&pattern, "(.*?){{(.*?)}}", &fixed, &regex);
      if (!consumed) {
        assert(consumed &&
               "Did not make forward progress for regex in check rule");
      }
      if (!fixed.empty()) {
        parts.emplace_back(make_unique<Check::Part>(Type::Fixed, fixed));
      }
      if (!regex.empty()) {
        parts.emplace_back(make_unique<Check::Part>(Type::Regex, regex));
      }
    } else if (var_exists && (!regex_exists || var_start < regex_start)) {
      const auto consumed =
          RE2::Consume(&pattern, "(.*?)\\[\\[(.*?)\\]\\]", &fixed, &var);
      if (!consumed) {
        assert(consumed &&
               "Did not make forward progress for var in check rule");
      }
      if (!fixed.empty()) {
        parts.emplace_back(make_unique<Check::Part>(Type::Fixed, fixed));
      }
      if (!var.empty()) {
        auto colon = var.find(":");
        // A colon at the end is useless anyway, so just make it a variable
        // use.
        if (colon == StringPiece::npos || colon == var.size() - 1) {
          parts.emplace_back(make_unique<Check::Part>(Type::VarUse, var));
        } else {
          StringPiece name = var.substr(0, colon);
          StringPiece expression = var.substr(colon + 1, StringPiece::npos);
          parts.emplace_back(
              make_unique<Check::Part>(Type::VarDef, var, name, expression));
        }
      }
    } else {
      // There is no regex, no var def, no var use.  Must be a fixed string.
      parts.push_back(make_unique<Check::Part>(Type::Fixed, pattern));
      break;
    }
  }

  return parts;
}

}  // namespace

std::pair<Result, CheckList> ParseChecks(StringPiece str,
                                         const Options& options) {
  // Returns a pair whose first member is a result constructed from the
  // given status and message, and the second member is an empy pattern.
  auto failure = [](Status status, StringPiece message) {
    return std::make_pair(Result(status, message), CheckList{});
  };

  if (options.prefix().size() == 0)
    return failure(Status::BadOption, "Rule prefix is empty");
  if (RE2::FullMatch(options.prefix(), "\\s+"))
    return failure(Status::BadOption,
                   "Rule prefix is whitespace.  That's silly.");

  CheckList check_list;

  const auto quoted_prefix = RE2::QuoteMeta(options.prefix());
  // Match the following parts:
  //    .*?               - Text that is not the rule prefix
  //    quoted_prefix     - A Simple Check prefix
  //    (-NEXT|-SAME)?    - An optional check type suffix. Two shown here.
  //    :                 - Colon
  //    \s*               - Whitespace
  //    (.*?)             - Captured parameter
  //    \s*               - Whitespace
  //    $                 - End of line

  const RE2 regexp(std::string(".*?") + quoted_prefix +
                   "(-NEXT|-SAME|-DAG|-LABEL|-NOT)?"
                   ":\\s*(.*?)\\s*$");
  Cursor cursor(str);
  while (!cursor.Exhausted()) {
    const auto line = cursor.RestOfLine();

    StringPiece matched_param;
    StringPiece suffix;
    if (RE2::PartialMatch(line, regexp, &suffix, &matched_param)) {
      const Type type = TypeForSuffix(suffix);
      auto parts(PartsForPattern(matched_param));
      check_list.push_back(Check(type, matched_param, std::move(parts)));
    }
    cursor.AdvanceLine();
  }

  if (check_list.empty()) {
    return failure(
        Status::NoRules,
        std::string("No check rules specified. Looking for prefix ") +
            options.prefix());
  }

  if (check_list[0].type() == Type::Same) {
    return failure(Status::BadRule, std::string(options.prefix()) +
                                        "-SAME can't be the first check rule");
  }

  return std::make_pair(Result(Result::Status::Ok), check_list);
}
}  // namespace effcee
