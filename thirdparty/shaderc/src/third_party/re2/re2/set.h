// Copyright 2010 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef RE2_SET_H_
#define RE2_SET_H_

#include <string>
#include <utility>
#include <vector>

#include "re2/re2.h"

namespace re2 {
class Prog;
class Regexp;
}  // namespace re2

namespace re2 {

// An RE2::Set represents a collection of regexps that can
// be searched for simultaneously.
class RE2::Set {
 public:
  enum ErrorKind {
    kNoError = 0,
    kNotCompiled,   // The set is not compiled.
    kOutOfMemory,   // The DFA ran out of memory.
    kInconsistent,  // The result is inconsistent. This should never happen.
  };

  struct ErrorInfo {
    ErrorKind kind;
  };

  Set(const RE2::Options& options, RE2::Anchor anchor);
  ~Set();

  // Adds pattern to the set using the options passed to the constructor.
  // Returns the index that will identify the regexp in the output of Match(),
  // or -1 if the regexp cannot be parsed.
  // Indices are assigned in sequential order starting from 0.
  // Errors do not increment the index; if error is not NULL, *error will hold
  // the error message from the parser.
  int Add(const StringPiece& pattern, std::string* error);

  // Compiles the set in preparation for matching.
  // Returns false if the compiler runs out of memory.
  // Add() must not be called again after Compile().
  // Compile() must be called before Match().
  bool Compile();

  // Returns true if text matches at least one of the regexps in the set.
  // Fills v (if not NULL) with the indices of the matching regexps.
  // Callers must not expect v to be sorted.
  bool Match(const StringPiece& text, std::vector<int>* v) const;

  // As above, but populates error_info (if not NULL) when none of the regexps
  // in the set matched. This can inform callers when DFA execution fails, for
  // example, because they might wish to handle that case differently.
  bool Match(const StringPiece& text, std::vector<int>* v,
             ErrorInfo* error_info) const;

 private:
  typedef std::pair<std::string, re2::Regexp*> Elem;

  RE2::Options options_;
  RE2::Anchor anchor_;
  std::vector<Elem> elem_;
  re2::Prog* prog_;
  bool compiled_;
  int size_;

  Set(const Set&) = delete;
  Set& operator=(const Set&) = delete;
};

}  // namespace re2

#endif  // RE2_SET_H_
