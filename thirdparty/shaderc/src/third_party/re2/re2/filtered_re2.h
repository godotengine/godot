// Copyright 2009 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef RE2_FILTERED_RE2_H_
#define RE2_FILTERED_RE2_H_

// The class FilteredRE2 is used as a wrapper to multiple RE2 regexps.
// It provides a prefilter mechanism that helps in cutting down the
// number of regexps that need to be actually searched.
//
// By design, it does not include a string matching engine. This is to
// allow the user of the class to use their favorite string match
// engine. The overall flow is: Add all the regexps using Add, then
// Compile the FilteredRE2. The compile returns strings that need to
// be matched. Note that all returned strings are lowercase. For
// applying regexps to a search text, the caller does the string
// matching using the strings returned. When doing the string match,
// note that the caller has to do that on lower cased version of the
// search text. Then call FirstMatch or AllMatches with a vector of
// indices of strings that were found in the text to get the actual
// regexp matches.

#include <string>
#include <vector>

#include "re2/re2.h"

namespace re2 {

class PrefilterTree;

class FilteredRE2 {
 public:
  FilteredRE2();
  explicit FilteredRE2(int min_atom_len);
  ~FilteredRE2();

  // Uses RE2 constructor to create a RE2 object (re). Returns
  // re->error_code(). If error_code is other than NoError, then re is
  // deleted and not added to re2_vec_.
  RE2::ErrorCode Add(const StringPiece& pattern,
                     const RE2::Options& options,
                     int *id);

  // Prepares the regexps added by Add for filtering.  Returns a set
  // of strings that the caller should check for in candidate texts.
  // The returned strings are lowercased. When doing string matching,
  // the search text should be lowercased first to find matching
  // strings from the set of strings returned by Compile.  Call after
  // all Add calls are done.
  void Compile(std::vector<std::string>* strings_to_match);

  // Returns the index of the first matching regexp.
  // Returns -1 on no match. Can be called prior to Compile.
  // Does not do any filtering: simply tries to Match the
  // regexps in a loop.
  int SlowFirstMatch(const StringPiece& text) const;

  // Returns the index of the first matching regexp.
  // Returns -1 on no match. Compile has to be called before
  // calling this.
  int FirstMatch(const StringPiece& text,
                 const std::vector<int>& atoms) const;

  // Returns the indices of all matching regexps, after first clearing
  // matched_regexps.
  bool AllMatches(const StringPiece& text,
                  const std::vector<int>& atoms,
                  std::vector<int>* matching_regexps) const;

  // Returns the indices of all potentially matching regexps after first
  // clearing potential_regexps.
  // A regexp is potentially matching if it passes the filter.
  // If a regexp passes the filter it may still not match.
  // A regexp that does not pass the filter is guaranteed to not match.
  void AllPotentials(const std::vector<int>& atoms,
                     std::vector<int>* potential_regexps) const;

  // The number of regexps added.
  int NumRegexps() const { return static_cast<int>(re2_vec_.size()); }

 private:

  // Get the individual RE2 objects. Useful for testing.
  RE2* GetRE2(int regexpid) const { return re2_vec_[regexpid]; }

  // Print prefilter.
  void PrintPrefilter(int regexpid);

  // Useful for testing and debugging.
  void RegexpsGivenStrings(const std::vector<int>& matched_atoms,
                           std::vector<int>* passed_regexps);

  // All the regexps in the FilteredRE2.
  std::vector<RE2*> re2_vec_;

  // Has the FilteredRE2 been compiled using Compile()
  bool compiled_;

  // An AND-OR tree of string atoms used for filtering regexps.
  PrefilterTree* prefilter_tree_;

  FilteredRE2(const FilteredRE2&) = delete;
  FilteredRE2& operator=(const FilteredRE2&) = delete;
};

}  // namespace re2

#endif  // RE2_FILTERED_RE2_H_
