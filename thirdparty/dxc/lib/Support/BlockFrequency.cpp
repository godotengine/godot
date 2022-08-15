//====--------------- lib/Support/BlockFrequency.cpp -----------*- C++ -*-====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Block Frequency class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/BlockFrequency.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace llvm;

BlockFrequency &BlockFrequency::operator*=(const BranchProbability &Prob) {
  Frequency = Prob.scale(Frequency);
  return *this;
}

const BlockFrequency
BlockFrequency::operator*(const BranchProbability &Prob) const {
  BlockFrequency Freq(Frequency);
  Freq *= Prob;
  return Freq;
}

BlockFrequency &BlockFrequency::operator/=(const BranchProbability &Prob) {
  Frequency = Prob.scaleByInverse(Frequency);
  return *this;
}

BlockFrequency BlockFrequency::operator/(const BranchProbability &Prob) const {
  BlockFrequency Freq(Frequency);
  Freq /= Prob;
  return Freq;
}

BlockFrequency &BlockFrequency::operator+=(const BlockFrequency &Freq) {
  uint64_t Before = Freq.Frequency;
  Frequency += Freq.Frequency;

  // If overflow, set frequency to the maximum value.
  if (Frequency < Before)
    Frequency = UINT64_MAX;

  return *this;
}

const BlockFrequency
BlockFrequency::operator+(const BlockFrequency &Prob) const {
  BlockFrequency Freq(Frequency);
  Freq += Prob;
  return Freq;
}

BlockFrequency &BlockFrequency::operator>>=(const unsigned count) {
  // Frequency can never be 0 by design.
  assert(Frequency != 0);

  // Shift right by count.
  Frequency >>= count;

  // Saturate to 1 if we are 0.
  Frequency |= Frequency == 0;
  return *this;
}
