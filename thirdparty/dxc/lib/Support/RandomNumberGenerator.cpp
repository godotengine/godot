//===-- RandomNumberGenerator.cpp - Implement RNG class -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements deterministic random number generation (RNG).
// The current implementation is NOT cryptographically secure as it uses
// the C++11 <random> facilities.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/RandomNumberGenerator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "rng"

// Tracking BUG: 19665
// http://llvm.org/bugs/show_bug.cgi?id=19665
//
// Do not change to cl::opt<uint64_t> since this silently breaks argument parsing.
#if 0 // HLSL Change Starts - option pending
static cl::opt<unsigned long long>
Seed("rng-seed", cl::value_desc("seed"),
     cl::desc("Seed for the random number generator"), cl::init(0));
#else
static const unsigned long long Seed = 0; // will go boom in the constructor, can't be set yet
#endif // HLSL Change Ends

RandomNumberGenerator::RandomNumberGenerator(StringRef Salt) {
  DEBUG(
    if (Seed == 0)
      dbgs() << "Warning! Using unseeded random number generator.\n"
  );

  // Combine seed and salts using std::seed_seq.
  // Data: Seed-low, Seed-high, Salt
  // Note: std::seed_seq can only store 32-bit values, even though we
  // are using a 64-bit RNG. This isn't a problem since the Mersenne
  // twister constructor copies these correctly into its initial state.
  std::vector<uint32_t> Data;
  Data.reserve(2 + Salt.size());
  Data.push_back(Seed);
  Data.push_back(Seed >> 32);

  std::copy(Salt.begin(), Salt.end(), Data.end());

  std::seed_seq SeedSeq(Data.begin(), Data.end());
  Generator.seed(SeedSeq);
}

uint_fast64_t RandomNumberGenerator::operator()() {
  return Generator();
}
