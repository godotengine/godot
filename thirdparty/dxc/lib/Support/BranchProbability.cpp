//===-------------- lib/Support/BranchProbability.cpp -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Branch Probability class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

raw_ostream &BranchProbability::print(raw_ostream &OS) const {
  return OS << N << " / " << D << " = "
            << format("%g%%", ((double)N / D) * 100.0);
}

void BranchProbability::dump() const { print(dbgs()) << '\n'; }

static uint64_t scale(uint64_t Num, uint32_t N, uint32_t D) {
  assert(D && "divide by 0");

  // Fast path for multiplying by 1.0.
  if (!Num || D == N)
    return Num;

  // Split Num into upper and lower parts to multiply, then recombine.
  uint64_t ProductHigh = (Num >> 32) * N;
  uint64_t ProductLow = (Num & UINT32_MAX) * N;

  // Split into 32-bit digits.
  uint32_t Upper32 = ProductHigh >> 32;
  uint32_t Lower32 = ProductLow & UINT32_MAX;
  uint32_t Mid32Partial = ProductHigh & UINT32_MAX;
  uint32_t Mid32 = Mid32Partial + (ProductLow >> 32);

  // Carry.
  Upper32 += Mid32 < Mid32Partial;

  // Check for overflow.
  if (Upper32 >= D)
    return UINT64_MAX;

  uint64_t Rem = (uint64_t(Upper32) << 32) | Mid32;
  uint64_t UpperQ = Rem / D;

  // Check for overflow.
  if (UpperQ > UINT32_MAX)
    return UINT64_MAX;

  Rem = ((Rem % D) << 32) | Lower32;
  uint64_t LowerQ = Rem / D;
  uint64_t Q = (UpperQ << 32) + LowerQ;

  // Check for overflow.
  return Q < LowerQ ? UINT64_MAX : Q;
}

uint64_t BranchProbability::scale(uint64_t Num) const {
  return ::scale(Num, N, D);
}

uint64_t BranchProbability::scaleByInverse(uint64_t Num) const {
  return ::scale(Num, D, N);
}
