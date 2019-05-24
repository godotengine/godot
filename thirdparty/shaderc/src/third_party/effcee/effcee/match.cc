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

#include <algorithm>
#include <cassert>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "cursor.h"
#include "diagnostic.h"
#include "effcee.h"
#include "to_string.h"

using effcee::Check;
using Status = effcee::Result::Status;
using Type = effcee::Check::Type;

namespace effcee {

Result Match(StringPiece input, StringPiece checks, const Options& options) {
  const auto& parse_result = ParseChecks(checks, options);
  if (!parse_result.first) return parse_result.first;

  // A mapping from variable names to values.  This is updated when a check rule
  // matches a variable definition.
  VarMapping vars;

  // We think of the input string as a sequence of lines that can satisfy
  // the checks.  Walk through the rules until no unsatisfied checks are left.
  // We will erase a check when it has been satisifed.
  const CheckList& pattern = parse_result.second;
  assert(pattern.size() > 0);

  // What checks are resolved?  Entry |i| is true when check |i| in the
  // pattern is resolved.
  std::vector<bool> resolved(pattern.size(), false);

  // The matching algorithm scans both the input and the pattern from start
  // to finish.  At the start, all checks are unresolved.  We try to match
  // each line in the input against the unresolved checks in a sliding window
  // in the pattern.  When a positive check matches, we mark it as resolved.
  // When a negative check matches, the algorithm terminates with failure.
  // We mark a negative check as resolved when it is the earliest unresolved
  // check and the first positive check after it is resolved.

  // Initially the pattern window is just the first element.
  // |first_check| is the first unresolved check.
  size_t first_check = 0;
  const size_t num_checks = pattern.size();

  // The 1-based line number of the most recent successful match.
  int matched_line_num = 0;

  // Set up a cursor to scan the input, and helpers for generating diagnostics.
  Cursor cursor(input);
  // Points to the end of the previous positive match.
  StringPiece previous_match_end = input.substr(0, 0);

  // Returns a failure diagnostic without a message.;
  auto fail = []() { return Diagnostic(Status::Fail); };
  // Returns a string describing the filename, line, and column of a check rule,
  // including the text of the check rule and a caret pointing to the parameter
  // string.
  auto check_msg = [&checks, &options](StringPiece where, StringPiece message) {
    std::ostringstream out;
    out << options.checks_name() << LineMessage(checks, where, message);
    return out.str();
  };
  // Returns a string describing the filename, line, and column of an input
  // string position, including the full line containing the position, and a
  // caret pointing to the position.
  auto input_msg = [&input, &options](StringPiece where, StringPiece message) {
    std::ostringstream out;
    out << options.input_name() << LineMessage(input, where, message);
    return out.str();
  };
  // Returns a string describing the value of each variable use in the
  // given check, in the context of the |where| portion of the input line.
  auto var_notes = [&input_msg, &vars](StringPiece where, const Check& check) {
    std::ostringstream out;
    for (const auto& part : check.parts()) {
      const auto var_use = part->VarUseName();
      if (!var_use.empty()) {
        std::ostringstream phrase;
        std::string var_use_str(ToString(var_use));
        if (vars.find(var_use_str) != vars.end()) {
          phrase << "note: with variable \"" << var_use << "\" equal to \""
                 << vars[var_use_str] << "\"";
        } else {
          phrase << "note: uses undefined variable \"" << var_use << "\"";
        }
        out << input_msg(where, phrase.str());
      }
    }
    return out.str();
  };

  // For each line.
  for (; !cursor.Exhausted(); cursor.AdvanceLine()) {
    // Try to match the current line against the unresolved checks.

    // The number of characters the cursor should advance to accommodate a
    // recent DAG check match.
    size_t deferred_advance = 0;

    bool scan_this_line = true;
    while (scan_this_line) {
      // Skip the initial segment of resolved checks.  Slides the left end of
      // the pattern window toward the right.
      while (first_check < num_checks && resolved[first_check]) ++first_check;
      // We've reached the end of the pattern.  Declare success.
      if (first_check == num_checks) return Result(Result::Status::Ok);

      size_t first_unresolved_dag = num_checks;
      size_t first_unresolved_negative = num_checks;

      bool resolved_something = false;

      for (size_t i = first_check; i < num_checks; ++i) {
        if (resolved[i]) continue;

        const Check& check = pattern[i];

        if (check.type() != Type::DAG) {
          cursor.Advance(deferred_advance);
          deferred_advance = 0;
        }
        const StringPiece rest_of_line = cursor.RestOfLine();
        StringPiece unconsumed = rest_of_line;
        StringPiece captured;

        if (check.Matches(&unconsumed, &captured, &vars)) {
          if (check.type() == Type::Not) {
            return fail() << input_msg(captured,
                                       "error: CHECK-NOT: string occurred!")
                          << check_msg(
                                 check.param(),
                                 "note: CHECK-NOT: pattern specified here")
                          << var_notes(captured, check);
          }

          if (check.type() == Type::Same &&
              cursor.line_num() != matched_line_num) {
            return fail()
                   << check_msg(check.param(),
                                "error: CHECK-SAME: is not on the same line as "
                                "previous match")
                   << input_msg(captured, "note: 'next' match was here")
                   << input_msg(previous_match_end,
                                "note: previous match ended here");
          }

          if (check.type() == Type::Next) {
            if (cursor.line_num() == matched_line_num) {
              return fail()
                     << check_msg(check.param(),
                                  "error: CHECK-NEXT: is on the same line as "
                                  "previous match")
                     << input_msg(captured, "note: 'next' match was here")
                     << input_msg(previous_match_end,
                                  "note: previous match ended here")
                     << var_notes(previous_match_end, check);
            }
            if (cursor.line_num() > 1 + matched_line_num) {
              // This must be valid since there was an intervening line.
              const auto non_match =
                  Cursor(input)
                      .Advance(previous_match_end.begin() - input.begin())
                      .AdvanceLine()
                      .RestOfLine();

              return fail()
                     << check_msg(check.param(),
                                  "error: CHECK-NEXT: is not on the line after "
                                  "the previous match")
                     << input_msg(captured, "note: 'next' match was here")
                     << input_msg(previous_match_end,
                                  "note: previous match ended here")
                     << input_msg(non_match,
                                  "note: non-matching line after previous "
                                  "match is here")
                     << var_notes(previous_match_end, check);
            }
          }

          if (check.type() != Type::DAG && first_unresolved_dag < i) {
            return fail()
                   << check_msg(pattern[first_unresolved_dag].param(),
                                "error: expected string not found in input")
                   << input_msg(previous_match_end, "note: scanning from here")
                   << input_msg(captured, "note: next check matches here")
                   << var_notes(previous_match_end, check);
          }

          resolved[i] = true;
          matched_line_num = cursor.line_num();
          previous_match_end = unconsumed;
          resolved_something = true;

          // Resolve any prior negative checks that precede an unresolved DAG.
          for (auto j = first_unresolved_negative,
                    limit = std::min(first_unresolved_dag, i);
               j < limit; ++j) {
            resolved[j] = true;
          }

          // Normally advance past the matched text.  But DAG checks might need
          // to match out of order on the same line.  So only advance for
          // non-DAG cases.

          const size_t advance_proposal =
              rest_of_line.size() - unconsumed.size();
          if (check.type() == Type::DAG) {
            deferred_advance = std::max(deferred_advance, advance_proposal);
          } else {
            cursor.Advance(advance_proposal);
          }

        } else {
          // This line did not match the check.
          if (check.type() == Type::Not) {
            first_unresolved_negative = std::min(first_unresolved_negative, i);
            // An unresolved Not check stops the search for more DAG checks.
            if (first_unresolved_dag < num_checks) i = num_checks;
          } else if (check.type() == Type::DAG) {
            first_unresolved_dag = std::min(first_unresolved_dag, i);
          } else {
            // An unresolved non-DAG check check stops this pass over the
            // checks.
            i = num_checks;
          }
        }
      }
      scan_this_line = resolved_something;
    }
  }

  // Fail if there are any unresolved positive checks.
  for (auto i = first_check; i < num_checks; ++i) {
    if (resolved[i]) continue;
    const auto check = pattern[i];
    if (check.type() == Type::Not) continue;

    return fail() << check_msg(check.param(),
                               "error: expected string not found in input")
                  << input_msg(previous_match_end, "note: scanning from here")
                  << var_notes(previous_match_end, check);
  }

  return Result(Result::Status::Ok);
}
}  // namespace effcee
