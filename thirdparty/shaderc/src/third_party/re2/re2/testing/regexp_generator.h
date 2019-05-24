// Copyright 2008 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef RE2_TESTING_REGEXP_GENERATOR_H_
#define RE2_TESTING_REGEXP_GENERATOR_H_

// Regular expression generator: generates all possible
// regular expressions within given parameters (see below for details).

#include <stdint.h>
#include <random>
#include <string>
#include <vector>

#include "util/util.h"
#include "re2/stringpiece.h"

namespace re2 {

// Regular expression generator.
//
// Given a set of atom expressions like "a", "b", or "."
// and operators like "%s*", generates all possible regular expressions
// using at most maxbases base expressions and maxops operators.
// For each such expression re, calls HandleRegexp(re).
//
// Callers are expected to subclass RegexpGenerator and provide HandleRegexp.
//
class RegexpGenerator {
 public:
  RegexpGenerator(int maxatoms, int maxops,
                  const std::vector<std::string>& atoms,
                  const std::vector<std::string>& ops);
  virtual ~RegexpGenerator() {}

  // Generates all the regular expressions, calling HandleRegexp(re) for each.
  void Generate();

  // Generates n random regular expressions, calling HandleRegexp(re) for each.
  void GenerateRandom(int32_t seed, int n);

  // Handles a regular expression.  Must be provided by subclass.
  virtual void HandleRegexp(const std::string& regexp) = 0;

  // The egrep regexp operators: * + ? | and concatenation.
  static const std::vector<std::string>& EgrepOps();

 private:
  void RunPostfix(const std::vector<std::string>& post);
  void GeneratePostfix(std::vector<std::string>* post,
                       int nstk, int ops, int lits);
  bool GenerateRandomPostfix(std::vector<std::string>* post,
                             int nstk, int ops, int lits);

  int maxatoms_;                    // Maximum number of atoms allowed in expr.
  int maxops_;                      // Maximum number of ops allowed in expr.
  std::vector<std::string> atoms_;  // Possible atoms.
  std::vector<std::string> ops_;    // Possible ops.
  std::minstd_rand0 rng_;           // Random number generator.

  RegexpGenerator(const RegexpGenerator&) = delete;
  RegexpGenerator& operator=(const RegexpGenerator&) = delete;
};

// Helpers for preparing arguments to RegexpGenerator constructor.

// Returns one string for each character in s.
std::vector<std::string> Explode(const StringPiece& s);

// Splits string everywhere sep is found, returning
// vector of pieces.
std::vector<std::string> Split(const StringPiece& sep, const StringPiece& s);

}  // namespace re2

#endif  // RE2_TESTING_REGEXP_GENERATOR_H_
