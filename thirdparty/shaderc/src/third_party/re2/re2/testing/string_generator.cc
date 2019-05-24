// Copyright 2008 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// String generator: generates all possible strings of up to
// maxlen letters using the set of letters in alpha.
// Fetch strings using a Java-like Next()/HasNext() interface.

#include <stddef.h>
#include <stdint.h>
#include <string>
#include <vector>

#include "util/test.h"
#include "util/logging.h"
#include "re2/testing/string_generator.h"

namespace re2 {

StringGenerator::StringGenerator(int maxlen,
                                 const std::vector<std::string>& alphabet)
    : maxlen_(maxlen), alphabet_(alphabet),
      generate_null_(false),
      random_(false), nrandom_(0) {

  // Degenerate case: no letters, no non-empty strings.
  if (alphabet_.empty())
    maxlen_ = 0;

  // Next() will return empty string (digits_ is empty).
  hasnext_ = true;
}

// Resets the string generator state to the beginning.
void StringGenerator::Reset() {
  digits_.clear();
  hasnext_ = true;
  random_ = false;
  nrandom_ = 0;
  generate_null_ = false;
}

// Increments the big number in digits_, returning true if successful.
// Returns false if all the numbers have been used.
bool StringGenerator::IncrementDigits() {
  // First try to increment the current number.
  for (int i = static_cast<int>(digits_.size()) - 1; i >= 0; i--) {
    if (++digits_[i] < static_cast<int>(alphabet_.size()))
      return true;
    digits_[i] = 0;
  }

  // If that failed, make a longer number.
  if (static_cast<int>(digits_.size()) < maxlen_) {
    digits_.push_back(0);
    return true;
  }

  return false;
}

// Generates random digits_, return true if successful.
// Returns false if the random sequence is over.
bool StringGenerator::RandomDigits() {
  if (--nrandom_ <= 0)
    return false;

  std::uniform_int_distribution<int> random_len(0, maxlen_);
  std::uniform_int_distribution<int> random_alphabet_index(
      0, static_cast<int>(alphabet_.size()) - 1);

  // Pick length.
  int len = random_len(rng_);
  digits_.resize(len);
  for (int i = 0; i < len; i++)
    digits_[i] = random_alphabet_index(rng_);
  return true;
}

// Returns the next string in the iteration, which is the one
// currently described by digits_.  Calls IncrementDigits
// after computing the string, so that it knows the answer
// for subsequent HasNext() calls.
const StringPiece& StringGenerator::Next() {
  CHECK(hasnext_);
  if (generate_null_) {
    generate_null_ = false;
    sp_ = StringPiece();
    return sp_;
  }
  s_.clear();
  for (size_t i = 0; i < digits_.size(); i++) {
    s_ += alphabet_[digits_[i]];
  }
  hasnext_ = random_ ? RandomDigits() : IncrementDigits();
  sp_ = s_;
  return sp_;
}

// Sets generator up to return n random strings.
void StringGenerator::Random(int32_t seed, int n) {
  rng_.seed(seed);

  random_ = true;
  nrandom_ = n;
  hasnext_ = nrandom_ > 0;
}

void StringGenerator::GenerateNULL() {
  generate_null_ = true;
  hasnext_ = true;
}

}  // namespace re2
