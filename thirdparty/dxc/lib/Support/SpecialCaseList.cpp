//===-- SpecialCaseList.cpp - special case list for sanitizers ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a utility class for instrumentation passes (like AddressSanitizer
// or ThreadSanitizer) to avoid instrumenting some functions or global
// variables, or to instrument some functions or global variables in a specific
// way, based on a user-supplied list.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/SpecialCaseList.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Regex.h"
#include <string>
#include <system_error>
#include <utility>

namespace llvm {

/// Represents a set of regular expressions.  Regular expressions which are
/// "literal" (i.e. no regex metacharacters) are stored in Strings, while all
/// others are represented as a single pipe-separated regex in RegEx.  The
/// reason for doing so is efficiency; StringSet is much faster at matching
/// literal strings than Regex.
struct SpecialCaseList::Entry {
  Entry() {}
  Entry(Entry &&Other)
      : Strings(std::move(Other.Strings)), RegEx(std::move(Other.RegEx)) {}

  StringSet<> Strings;
  std::unique_ptr<Regex> RegEx;

  bool match(StringRef Query) const {
    return Strings.count(Query) || (RegEx && RegEx->match(Query));
  }
};

SpecialCaseList::SpecialCaseList() : Entries(), Regexps(), IsCompiled(false) {}

std::unique_ptr<SpecialCaseList>
SpecialCaseList::create(const std::vector<std::string> &Paths,
                        std::string &Error) {
  std::unique_ptr<SpecialCaseList> SCL(new SpecialCaseList());
  for (auto Path : Paths) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr =
        MemoryBuffer::getFile(Path);
    if (std::error_code EC = FileOrErr.getError()) {
      Error = (Twine("can't open file '") + Path + "': " + EC.message()).str();
      return nullptr;
    }
    std::string ParseError;
    if (!SCL->parse(FileOrErr.get().get(), ParseError)) {
      Error = (Twine("error parsing file '") + Path + "': " + ParseError).str();
      return nullptr;
    }
  }
  SCL->compile();
  return SCL;
}

std::unique_ptr<SpecialCaseList> SpecialCaseList::create(const MemoryBuffer *MB,
                                                         std::string &Error) {
  std::unique_ptr<SpecialCaseList> SCL(new SpecialCaseList());
  if (!SCL->parse(MB, Error))
    return nullptr;
  SCL->compile();
  return SCL;
}

std::unique_ptr<SpecialCaseList>
SpecialCaseList::createOrDie(const std::vector<std::string> &Paths) {
  std::string Error;
  if (auto SCL = create(Paths, Error))
    return SCL;
  report_fatal_error(Error);
}

bool SpecialCaseList::parse(const MemoryBuffer *MB, std::string &Error) {
  // Iterate through each line in the blacklist file.
  SmallVector<StringRef, 16> Lines;
  SplitString(MB->getBuffer(), Lines, "\n\r");
  int LineNo = 1;
  for (auto I = Lines.begin(), E = Lines.end(); I != E; ++I, ++LineNo) {
    // Ignore empty lines and lines starting with "#"
    if (I->empty() || I->startswith("#"))
      continue;
    // Get our prefix and unparsed regexp.
    std::pair<StringRef, StringRef> SplitLine = I->split(":");
    StringRef Prefix = SplitLine.first;
    if (SplitLine.second.empty()) {
      // Missing ':' in the line.
      Error = (Twine("malformed line ") + Twine(LineNo) + ": '" +
               SplitLine.first + "'").str();
      return false;
    }

    std::pair<StringRef, StringRef> SplitRegexp = SplitLine.second.split("=");
    std::string Regexp = SplitRegexp.first;
    StringRef Category = SplitRegexp.second;

    // See if we can store Regexp in Strings.
    if (Regex::isLiteralERE(Regexp)) {
      Entries[Prefix][Category].Strings.insert(Regexp);
      continue;
    }

    // Replace * with .*
    for (size_t pos = 0; (pos = Regexp.find("*", pos)) != std::string::npos;
         pos += strlen(".*")) {
      Regexp.replace(pos, strlen("*"), ".*");
    }

    // Check that the regexp is valid.
    Regex CheckRE(Regexp);
    std::string REError;
    if (!CheckRE.isValid(REError)) {
      Error = (Twine("malformed regex in line ") + Twine(LineNo) + ": '" +
               SplitLine.second + "': " + REError).str();
      return false;
    }

    // Add this regexp into the proper group by its prefix.
    if (!Regexps[Prefix][Category].empty())
      Regexps[Prefix][Category] += "|";
    Regexps[Prefix][Category] += "^" + Regexp + "$";
  }
  return true;
}

void SpecialCaseList::compile() {
  assert(!IsCompiled && "compile() should only be called once");
  // Iterate through each of the prefixes, and create Regexs for them.
  for (StringMap<StringMap<std::string>>::const_iterator I = Regexps.begin(),
                                                         E = Regexps.end();
       I != E; ++I) {
    for (StringMap<std::string>::const_iterator II = I->second.begin(),
                                                IE = I->second.end();
         II != IE; ++II) {
      Entries[I->getKey()][II->getKey()].RegEx.reset(new Regex(II->getValue()));
    }
  }
  Regexps.clear();
  IsCompiled = true;
}

SpecialCaseList::~SpecialCaseList() {}

bool SpecialCaseList::inSection(StringRef Section, StringRef Query,
                                StringRef Category) const {
  assert(IsCompiled && "SpecialCaseList::compile() was not called!");
  StringMap<StringMap<Entry> >::const_iterator I = Entries.find(Section);
  if (I == Entries.end()) return false;
  StringMap<Entry>::const_iterator II = I->second.find(Category);
  if (II == I->second.end()) return false;

  return II->getValue().match(Query);
}

}  // namespace llvm
