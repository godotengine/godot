//===- StringMatcher.h - Generate a matcher for input strings ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the StringMatcher class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TABLEGEN_STRINGMATCHER_H
#define LLVM_TABLEGEN_STRINGMATCHER_H

#include "llvm/ADT/StringRef.h"
#include <string>
#include <utility>
#include <vector>

namespace llvm {
  class raw_ostream;

/// StringMatcher - Given a list of strings and code to execute when they match,
/// output a simple switch tree to classify the input string.
///
/// If a match is found, the code in Vals[i].second is executed; control must
/// not exit this code fragment.  If nothing matches, execution falls through.
///
class StringMatcher {
public:
  typedef std::pair<std::string, std::string> StringPair;

private:
  StringRef StrVariableName;
  const std::vector<StringPair> &Matches;
  raw_ostream &OS;

public:
  StringMatcher(StringRef strVariableName, 
                const std::vector<StringPair> &matches, raw_ostream &os)
    : StrVariableName(strVariableName), Matches(matches), OS(os) {}

  void Emit(unsigned Indent = 0) const;

private:
  bool EmitStringMatcherForChar(const std::vector<const StringPair*> &Matches,
                                unsigned CharNo, unsigned IndentCount) const;
};

} // end llvm namespace.

#endif
