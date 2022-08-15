//==- llvm/Support/RandomNumberGenerator.h - RNG for diversity ---*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an abstraction for deterministic random number
// generation (RNG).  Note that the current implementation is not
// cryptographically secure as it uses the C++11 <random> facilities.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_RANDOMNUMBERGENERATOR_H_
#define LLVM_SUPPORT_RANDOMNUMBERGENERATOR_H_

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataTypes.h" // Needed for uint64_t on Windows.
#include <random>

namespace llvm {

/// A random number generator.
///
/// Instances of this class should not be shared across threads. The
/// seed should be set by passing the -rng-seed=<uint64> option. Use
/// Module::createRNG to create a new RNG instance for use with that
/// module.
class RandomNumberGenerator {
public:
  /// Returns a random number in the range [0, Max).
  uint_fast64_t operator()();

private:
  /// Seeds and salts the underlying RNG engine.
  ///
  /// This constructor should not be used directly. Instead use
  /// Module::createRNG to create a new RNG salted with the Module ID.
  RandomNumberGenerator(StringRef Salt);

  // 64-bit Mersenne Twister by Matsumoto and Nishimura, 2000
  // http://en.cppreference.com/w/cpp/numeric/random/mersenne_twister_engine
  // This RNG is deterministically portable across C++11
  // implementations.
  std::mt19937_64 Generator;

  // Noncopyable.
  RandomNumberGenerator(const RandomNumberGenerator &other) = delete;
  RandomNumberGenerator &operator=(const RandomNumberGenerator &other) = delete;

  friend class Module;
};
}

#endif
