//===-- llvm/ADT/Statistic.h - Easy way to expose stats ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the 'Statistic' class, which is designed to be an easy way
// to expose various metrics from passes.  These statistics are printed at the
// end of a run (from llvm_shutdown), when the -stats command line option is
// passed on the command line.
//
// This is useful for reporting information like the number of instructions
// simplified, optimized or removed by various transformations, like this:
//
// static Statistic NumInstsKilled("gcse", "Number of instructions killed");
//
// Later, in the code: ++NumInstsKilled;
//
// NOTE: Statistics *must* be declared as global variables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_STATISTIC_H
#define LLVM_ADT_STATISTIC_H

#include "llvm/Support/Atomic.h"
#include "llvm/Support/Valgrind.h"

namespace llvm {
class raw_ostream;

class Statistic {
public:
  const char *Name;
  const char *Desc;
  volatile llvm::sys::cas_flag Value;
  bool Initialized;

  llvm::sys::cas_flag getValue() const { return Value; }
  const char *getName() const { return Name; }
  const char *getDesc() const { return Desc; }

  /// construct - This should only be called for non-global statistics.
  void construct(const char *name, const char *desc) {
    Name = name; Desc = desc;
    Value = 0; Initialized = false;
  }

  // Allow use of this class as the value itself.
  operator unsigned() const { return Value; }

#if (!defined(NDEBUG) || defined(LLVM_ENABLE_STATS)) && 0 // HLSL Change - always disable, shouldn't do process-wide alloc in compile
   const Statistic &operator=(unsigned Val) {
    Value = Val;
    return init();
  }

  const Statistic &operator++() {
    // FIXME: This function and all those that follow carefully use an
    // atomic operation to update the value safely in the presence of
    // concurrent accesses, but not to read the return value, so the
    // return value is not thread safe.
    sys::AtomicIncrement(&Value);
    return init();
  }

  unsigned operator++(int) {
    init();
    unsigned OldValue = Value;
    sys::AtomicIncrement(&Value);
    return OldValue;
  }

  const Statistic &operator--() {
    sys::AtomicDecrement(&Value);
    return init();
  }

  unsigned operator--(int) {
    init();
    unsigned OldValue = Value;
    sys::AtomicDecrement(&Value);
    return OldValue;
  }

  const Statistic &operator+=(const unsigned &V) {
    if (!V) return *this;
    sys::AtomicAdd(&Value, V);
    return init();
  }

  const Statistic &operator-=(const unsigned &V) {
    if (!V) return *this;
    sys::AtomicAdd(&Value, -V);
    return init();
  }

  const Statistic &operator*=(const unsigned &V) {
    sys::AtomicMul(&Value, V);
    return init();
  }

  const Statistic &operator/=(const unsigned &V) {
    sys::AtomicDiv(&Value, V);
    return init();
  }

#else  // Statistics are disabled in release builds.

  const Statistic &operator=(unsigned Val) {
    return *this;
  }

  const Statistic &operator++() {
    return *this;
  }

  unsigned operator++(int) {
    return 0;
  }

  const Statistic &operator--() {
    return *this;
  }

  unsigned operator--(int) {
    return 0;
  }

  const Statistic &operator+=(const unsigned &V) {
    return *this;
  }

  const Statistic &operator-=(const unsigned &V) {
    return *this;
  }

  const Statistic &operator*=(const unsigned &V) {
    return *this;
  }

  const Statistic &operator/=(const unsigned &V) {
    return *this;
  }

#endif  // !defined(NDEBUG) || defined(LLVM_ENABLE_STATS)

protected:
  Statistic &init() {
    bool tmp = Initialized;
    sys::MemoryFence();
    if (!tmp) RegisterStatistic();
    TsanHappensAfter(this);
    return *this;
  }
  void RegisterStatistic();
};

// STATISTIC - A macro to make definition of statistics really simple.  This
// automatically passes the DEBUG_TYPE of the file into the statistic.
#define STATISTIC(VARNAME, DESC) \
  static llvm::Statistic VARNAME = { DEBUG_TYPE, DESC, 0, 0 }

/// \brief Enable the collection and printing of statistics.
void EnableStatistics();

/// \brief Check if statistics are enabled.
bool AreStatisticsEnabled();

/// \brief Print statistics to the file returned by CreateInfoOutputFile().
void PrintStatistics();

/// \brief Print statistics to the given output stream.
void PrintStatistics(raw_ostream &OS);

} // End llvm namespace

#endif
