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

#ifndef EFFCEE_CHECK_H
#define EFFCEE_CHECK_H

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "effcee.h"
#include "make_unique.h"

namespace effcee {

// A mapping from a name to a string value.
using VarMapping = std::unordered_map<std::string, std::string>;

// A single check indicating something to be matched.
//
// A _positive_ check is _resolved_ when its parameter is matches a part of the
// in the input text.  A _negative_ check is _resolved_ when its parameter does
// _not_ match a section of the input between context-dependent start and end
// points.
class Check {
 public:
  // The type Determines when the check is satisfied.  The Not type denotes
  // a negative check.  The other types denote positive checks.
  enum class Type {
    Simple,  // Matches a string.
    Next,    // Matches a string, on the line following previous match.
    Same,    // Matches a string, on the same line as the previous metch.
    DAG,     // Matches a string, unordered with respect to other
    Label,   // Like Simple, but resets local variables.
    Not,     // Given string is not found before next positive match.
  };

  // A Part is a contiguous segment of the check pattern.  A part is
  // distinguished by how it matches against input.
  class Part {
   public:
    enum class Type {
      Fixed,   // A fixed string: characters are matched exactly, in sequence.
      Regex,   // A regular expression
      VarDef,  // A variable definition
      VarUse,  // A variable use
    };

    Part(Type type, StringPiece param)
        : type_(type),
          param_(param),
          name_(),
          expression_(),
          num_capturing_groups_(CountCapturingGroups()) {}

    // A constructor for a VarDef variant.
    Part(Type type, StringPiece param, StringPiece name, StringPiece expr)
        : type_(type),
          param_(param),
          name_(name),
          expression_(expr),
          num_capturing_groups_(CountCapturingGroups()) {}

    // Returns true if this part might match a target string.  The only case where
    // this is false is for a VarUse part where the variable is not yet defined.
    bool MightMatch(const VarMapping& vars) const;

    // Returns a regular expression to match this part, given a mapping of
    // variable names to values.  If this part is a fixed string or variable use
    // then quoting has been applied.
    std::string Regex(const VarMapping& vars) const;

    // Returns number of capturing subgroups in the regex for a Regex or VarDef
    // part, and 0 for other parts.
    int NumCapturingGroups() const { return num_capturing_groups_; }

    // If this is a VarDef, then returns the name of the variable. Otherwise
    // returns an empty string.
    StringPiece VarDefName() const { return name_; }

    // If this is a VarUse, then returns the name of the variable. Otherwise
    // returns an empty string.
    StringPiece VarUseName() const {
      return type_ == Type::VarUse ? param_ : "";
    }

   private:
    // Computes the number of capturing groups in this part. This is zero
    // for Fixed and VarUse parts.
    int CountCapturingGroups();

    // The part type.
    Type type_;
    // The part parameter.  For a Regex, VarDef, and VarUse, this does not
    // have the delimiters.
    StringPiece param_;

    // For a VarDef, the name of the variable.
    StringPiece name_;
    // For a VarDef, the regex matching the new value for the variable.
    StringPiece expression_;
    // The number of capturing subgroups in the regex for a Regex or VarDef
    // part, and 0 for other kinds of parts.
    int num_capturing_groups_;
  };

  using Parts = std::vector<std::unique_ptr<Part>>;

  // MSVC needs a default constructor.  However, a default-constructed Check
  // instance can't be used for matching.
  Check() : type_(Type::Simple) {}

  // Construct a Check object of the given type and fixed parameter string.
  // In particular, this retains a StringPiece reference to the |param|
  // contents, so that string storage should remain valid for the duration
  // of this object.
  Check(Type type, StringPiece param);

  // Construct a Check object of the given type, with given parameter string
  // and specified parts.
  Check(Type type, StringPiece param, Parts&& parts)
      : type_(type), param_(param), parts_(std::move(parts)) {}

  // Move constructor.
  Check(Check&& other) : type_(other.type_), param_(other.param_) {
    parts_.swap(other.parts_);
  }
  // Copy constructor.
  Check(const Check& other) : type_(other.type_), param_(other.param_) {
    for (const auto& part : other.parts_) {
      parts_.push_back(make_unique<Part>(*part));
    }
  }
  // Copy and move assignment.
  Check& operator=(Check other) {
    type_ = other.type_;
    param_ = other.param_;
    std::swap(parts_, other.parts_);
    return *this;
  }

  // Accessors.
  Type type() const { return type_; }
  StringPiece param() const { return param_; }
  const Parts& parts() const { return parts_; }

  // Tries to match the given string, using |vars| as the variable mapping
  // context.  A variable use, e.g. '[[X]]', matches the current value for
  // that variable in vars, 'X' in this case.  A variable definition,
  // e.g. '[[XYZ:[0-9]+]]', will match against the regex provdided after the
  // colon.  If successful, returns true, advances |str| past the matched
  // portion, saves the captured substring in |captured|, and sets the value
  // of named variables in |vars| with the strings they matched. Otherwise
  // returns false and does not update |str| or |captured|.  Assumes this
  // instance is not default-constructed.
  bool Matches(StringPiece* str, StringPiece* captured, VarMapping* vars) const;

 private:
  // The type of check.
  Type type_;

  // The parameter as given in user input, if any.
  StringPiece param_;

  // The parameter, broken down into parts.
  Parts parts_;
};

// Equality operator for Check.
inline bool operator==(const Check& lhs, const Check& rhs) {
  return lhs.type() == rhs.type() && lhs.param() == rhs.param();
}

// Inequality operator for Check.
inline bool operator!=(const Check& lhs, const Check& rhs) {
  return !(lhs == rhs);
}

using CheckList = std::vector<Check>;

// Parses |checks_string|, returning a Result status object and the sequence
// of recognized checks, taking |options| into account.  The result status
// object indicates success, or failure with a message.
// TODO(dneto): Only matches simple checks for now.
std::pair<Result, CheckList> ParseChecks(StringPiece checks_string,
                                         const Options& options);

}  // namespace effcee

#endif
