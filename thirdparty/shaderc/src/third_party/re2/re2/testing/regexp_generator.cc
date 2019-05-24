// Copyright 2008 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Regular expression generator: generates all possible
// regular expressions within parameters (see regexp_generator.h for details).

// The regexp generator first generates a sequence of commands in a simple
// postfix language.  Each command in the language is a string,
// like "a" or "%s*" or "%s|%s".
//
// To evaluate a command, enough arguments are popped from the value stack to
// plug into the %s slots.  Then the result is pushed onto the stack.
// For example, the command sequence
//      a b %s%s c
// results in the stack
//      ab c
//
// GeneratePostfix generates all possible command sequences.
// Then RunPostfix turns each sequence into a regular expression
// and passes the regexp to HandleRegexp.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <memory>
#include <stack>
#include <string>
#include <vector>

#include "util/test.h"
#include "util/logging.h"
#include "util/strutil.h"
#include "util/utf.h"
#include "re2/testing/regexp_generator.h"

namespace re2 {

// Returns a vector of the egrep regexp operators.
const std::vector<std::string>& RegexpGenerator::EgrepOps() {
  static const char *ops[] = {
    "%s%s",
    "%s|%s",
    "%s*",
    "%s+",
    "%s?",
    "%s\\C*",
  };
  static std::vector<std::string> v(ops, ops + arraysize(ops));
  return v;
}

RegexpGenerator::RegexpGenerator(int maxatoms, int maxops,
                                 const std::vector<std::string>& atoms,
                                 const std::vector<std::string>& ops)
    : maxatoms_(maxatoms), maxops_(maxops), atoms_(atoms), ops_(ops) {
  // Degenerate case.
  if (atoms_.empty())
    maxatoms_ = 0;
  if (ops_.empty())
    maxops_ = 0;
}

// Generates all possible regular expressions (within the parameters),
// calling HandleRegexp for each one.
void RegexpGenerator::Generate() {
  std::vector<std::string> postfix;
  GeneratePostfix(&postfix, 0, 0, 0);
}

// Generates random regular expressions, calling HandleRegexp for each one.
void RegexpGenerator::GenerateRandom(int32_t seed, int n) {
  rng_.seed(seed);

  for (int i = 0; i < n; i++) {
    std::vector<std::string> postfix;
    GenerateRandomPostfix(&postfix, 0, 0, 0);
  }
}

// Counts and returns the number of occurrences of "%s" in s.
static int CountArgs(const std::string& s) {
  const char *p = s.c_str();
  int n = 0;
  while ((p = strstr(p, "%s")) != NULL) {
    p += 2;
    n++;
  }
  return n;
}

// Generates all possible postfix command sequences.
// Each sequence is handed off to RunPostfix to generate a regular expression.
// The arguments are:
//   post:  the current postfix sequence
//   nstk:  the number of elements that would be on the stack after executing
//          the sequence
//   ops:   the number of operators used in the sequence
//   atoms: the number of atoms used in the sequence
// For example, if post were ["a", "b", "%s%s", "c"],
// then nstk = 2, ops = 1, atoms = 3.
//
// The initial call should be GeneratePostfix([empty vector], 0, 0, 0).
//
void RegexpGenerator::GeneratePostfix(std::vector<std::string>* post,
                                      int nstk, int ops, int atoms) {
  if (nstk == 1)
    RunPostfix(*post);

  // Early out: if used too many operators or can't
  // get back down to a single expression on the stack
  // using binary operators, give up.
  if (ops + nstk - 1 > maxops_)
    return;

  // Add atoms if there is room.
  if (atoms < maxatoms_) {
    for (size_t i = 0; i < atoms_.size(); i++) {
      post->push_back(atoms_[i]);
      GeneratePostfix(post, nstk + 1, ops, atoms + 1);
      post->pop_back();
    }
  }

  // Add operators if there are enough arguments.
  if (ops < maxops_) {
    for (size_t i = 0; i < ops_.size(); i++) {
      const std::string& fmt = ops_[i];
      int nargs = CountArgs(fmt);
      if (nargs <= nstk) {
        post->push_back(fmt);
        GeneratePostfix(post, nstk - nargs + 1, ops + 1, atoms);
        post->pop_back();
      }
    }
  }
}

// Generates a random postfix command sequence.
// Stops and returns true once a single sequence has been generated.
bool RegexpGenerator::GenerateRandomPostfix(std::vector<std::string>* post,
                                            int nstk, int ops, int atoms) {
  std::uniform_int_distribution<int> random_stop(0, maxatoms_ - atoms);
  std::uniform_int_distribution<int> random_bit(0, 1);
  std::uniform_int_distribution<int> random_ops_index(
      0, static_cast<int>(ops_.size()) - 1);
  std::uniform_int_distribution<int> random_atoms_index(
      0, static_cast<int>(atoms_.size()) - 1);

  for (;;) {
    // Stop if we get to a single element, but only sometimes.
    if (nstk == 1 && random_stop(rng_) == 0) {
      RunPostfix(*post);
      return true;
    }

    // Early out: if used too many operators or can't
    // get back down to a single expression on the stack
    // using binary operators, give up.
    if (ops + nstk - 1 > maxops_)
      return false;

    // Add operators if there are enough arguments.
    if (ops < maxops_ && random_bit(rng_) == 0) {
      const std::string& fmt = ops_[random_ops_index(rng_)];
      int nargs = CountArgs(fmt);
      if (nargs <= nstk) {
        post->push_back(fmt);
        bool ret = GenerateRandomPostfix(post, nstk - nargs + 1,
                                         ops + 1, atoms);
        post->pop_back();
        if (ret)
          return true;
      }
    }

    // Add atoms if there is room.
    if (atoms < maxatoms_ && random_bit(rng_) == 0) {
      post->push_back(atoms_[random_atoms_index(rng_)]);
      bool ret = GenerateRandomPostfix(post, nstk + 1, ops, atoms + 1);
      post->pop_back();
      if (ret)
        return true;
    }
  }
}

// Interprets the postfix command sequence to create a regular expression
// passed to HandleRegexp.  The results of operators like %s|%s are wrapped
// in (?: ) to avoid needing to maintain a precedence table.
void RegexpGenerator::RunPostfix(const std::vector<std::string>& post) {
  std::stack<std::string> regexps;
  for (size_t i = 0; i < post.size(); i++) {
    switch (CountArgs(post[i])) {
      default:
        LOG(FATAL) << "Bad operator: " << post[i];
      case 0:
        regexps.push(post[i]);
        break;
      case 1: {
        std::string a = regexps.top();
        regexps.pop();
        regexps.push("(?:" + StringPrintf(post[i].c_str(), a.c_str()) + ")");
        break;
      }
      case 2: {
        std::string b = regexps.top();
        regexps.pop();
        std::string a = regexps.top();
        regexps.pop();
        regexps.push("(?:" +
                     StringPrintf(post[i].c_str(), a.c_str(), b.c_str()) +
                     ")");
        break;
      }
    }
  }

  if (regexps.size() != 1) {
    // Internal error - should never happen.
    printf("Bad regexp program:\n");
    for (size_t i = 0; i < post.size(); i++) {
      printf("  %s\n", CEscape(post[i]).c_str());
    }
    printf("Stack after running program:\n");
    while (!regexps.empty()) {
      printf("  %s\n", CEscape(regexps.top()).c_str());
      regexps.pop();
    }
    LOG(FATAL) << "Bad regexp program.";
  }

  HandleRegexp(regexps.top());
  HandleRegexp("^(?:" + regexps.top() + ")$");
  HandleRegexp("^(?:" + regexps.top() + ")");
  HandleRegexp("(?:" + regexps.top() + ")$");
}

// Split s into an vector of strings, one for each UTF-8 character.
std::vector<std::string> Explode(const StringPiece& s) {
  std::vector<std::string> v;

  for (const char *q = s.begin(); q < s.end(); ) {
    const char* p = q;
    Rune r;
    q += chartorune(&r, q);
    v.push_back(std::string(p, q - p));
  }

  return v;
}

// Split string everywhere a substring is found, returning
// vector of pieces.
std::vector<std::string> Split(const StringPiece& sep, const StringPiece& s) {
  std::vector<std::string> v;

  if (sep.size() == 0)
    return Explode(s);

  const char *p = s.begin();
  for (const char *q = s.begin(); q + sep.size() <= s.end(); q++) {
    if (StringPiece(q, sep.size()) == sep) {
      v.push_back(std::string(p, q - p));
      p = q + sep.size();
      q = p - 1;  // -1 for ++ in loop
      continue;
    }
  }
  if (p < s.end())
    v.push_back(std::string(p, s.end() - p));
  return v;
}

}  // namespace re2
