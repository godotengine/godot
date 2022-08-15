//===-- SpecialCaseList.h - special case list for sanitizers ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===//
//
// This is a utility class used to parse user-provided text files with
// "special case lists" for code sanitizers. Such files are used to
// define "ABI list" for DataFlowSanitizer and blacklists for another sanitizers
// like AddressSanitizer or UndefinedBehaviorSanitizer.
//
// Empty lines and lines starting with "#" are ignored. All the rest lines
// should have the form:
//   section:wildcard_expression[=category]
// If category is not specified, it is assumed to be empty string.
// Definitions of "section" and "category" are sanitizer-specific. For example,
// sanitizer blacklists support sections "src", "fun" and "global".
// Wildcard expressions define, respectively, source files, functions or
// globals which shouldn't be instrumented.
// Examples of categories:
//   "functional": used in DFSan to list functions with pure functional
//                 semantics.
//   "init": used in ASan blacklist to disable initialization-order bugs
//           detection for certain globals or source files.
// Full special case list file example:
// ---
// # Blacklisted items:
// fun:*_ZN4base6subtle*
// global:*global_with_bad_access_or_initialization*
// global:*global_with_initialization_issues*=init
// type:*Namespace::ClassName*=init
// src:file_with_tricky_code.cc
// src:ignore-global-initializers-issues.cc=init
//
// # Functions with pure functional semantics:
// fun:cos=functional
// fun:sin=functional
// ---
// Note that the wild card is in fact an llvm::Regex, but * is automatically
// replaced with .*
// This is similar to the "ignore" feature of ThreadSanitizer.
// http://code.google.com/p/data-race-test/wiki/ThreadSanitizerIgnores
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_SPECIALCASELIST_H
#define LLVM_SUPPORT_SPECIALCASELIST_H

#include "llvm/ADT/StringMap.h"
#include <string>
#include <vector>

namespace llvm {
class MemoryBuffer;
class Regex;
class StringRef;

class SpecialCaseList {
public:
  /// Parses the special case list entries from files. On failure, returns
  /// 0 and writes an error message to string.
  static std::unique_ptr<SpecialCaseList>
  create(const std::vector<std::string> &Paths, std::string &Error);
  /// Parses the special case list from a memory buffer. On failure, returns
  /// 0 and writes an error message to string.
  static std::unique_ptr<SpecialCaseList> create(const MemoryBuffer *MB,
                                                 std::string &Error);
  /// Parses the special case list entries from files. On failure, reports a
  /// fatal error.
  static std::unique_ptr<SpecialCaseList>
  createOrDie(const std::vector<std::string> &Paths);

  ~SpecialCaseList();

  /// Returns true, if special case list contains a line
  /// \code
  ///   @Section:<E>=@Category
  /// \endcode
  /// and @Query satisfies a wildcard expression <E>.
  bool inSection(StringRef Section, StringRef Query,
                 StringRef Category = StringRef()) const;

private:
  SpecialCaseList(SpecialCaseList const &) = delete;
  SpecialCaseList &operator=(SpecialCaseList const &) = delete;

  struct Entry;
  StringMap<StringMap<Entry>> Entries;
  StringMap<StringMap<std::string>> Regexps;
  bool IsCompiled;

  SpecialCaseList();
  /// Parses just-constructed SpecialCaseList entries from a memory buffer.
  bool parse(const MemoryBuffer *MB, std::string &Error);
  /// compile() should be called once, after parsing all the memory buffers.
  void compile();
};

}  // namespace llvm

#endif  // LLVM_SUPPORT_SPECIALCASELIST_H

