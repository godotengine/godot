//===-- llvm/MC/SubtargetFeature.h - CPU characteristics --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines and manages user or tool specified CPU characteristics.
// The intent is to be able to package specific features that should or should
// not be used on a specific target processor.  A tool, such as llc, could, as
// as example, gather chip info from the command line, a long with features
// that should be used on that chip.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_SUBTARGETFEATURE_H
#define LLVM_MC_SUBTARGETFEATURE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/DataTypes.h"
#include <bitset>

namespace llvm {
  class raw_ostream;
  class StringRef;

// A container class for subtarget features.
// This is convenient because std::bitset does not have a constructor
// with an initializer list of set bits.
const unsigned MAX_SUBTARGET_FEATURES = 64;
class FeatureBitset : public std::bitset<MAX_SUBTARGET_FEATURES> {
public:
  // Cannot inherit constructors because it's not supported by VC++..
  FeatureBitset() : bitset() {}

  FeatureBitset(const bitset<MAX_SUBTARGET_FEATURES>& B) : bitset(B) {}

  FeatureBitset(std::initializer_list<unsigned> Init) : bitset() {
    for (auto I = Init.begin() , E = Init.end(); I != E; ++I)
      set(*I);
  }
};

//===----------------------------------------------------------------------===//
///
/// SubtargetFeatureKV - Used to provide key value pairs for feature and
/// CPU bit flags.
//
struct SubtargetFeatureKV {
  const char *Key;                      // K-V key string
  const char *Desc;                     // Help descriptor
  FeatureBitset Value;                  // K-V integer value
  FeatureBitset Implies;                // K-V bit mask

  // Compare routine for std::lower_bound
  bool operator<(StringRef S) const {
    return StringRef(Key) < S;
  }
};

//===----------------------------------------------------------------------===//
///
/// SubtargetInfoKV - Used to provide key value pairs for CPU and arbitrary
/// pointers.
//
struct SubtargetInfoKV {
  const char *Key;                      // K-V key string
  const void *Value;                    // K-V pointer value

  // Compare routine for std::lower_bound
  bool operator<(StringRef S) const {
    return StringRef(Key) < S;
  }
};
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
///
/// SubtargetFeatures - Manages the enabling and disabling of subtarget
/// specific features.  Features are encoded as a string of the form
///   "+attr1,+attr2,-attr3,...,+attrN"
/// A comma separates each feature from the next (all lowercase.)
/// Each of the remaining features is prefixed with + or - indicating whether
/// that feature should be enabled or disabled contrary to the cpu
/// specification.
///

class SubtargetFeatures {
  std::vector<std::string> Features;    // Subtarget features as a vector
public:
  explicit SubtargetFeatures(StringRef Initial = "");

  /// Features string accessors.
  std::string getString() const;

  /// Adding Features.
  void AddFeature(StringRef String, bool Enable = true);

  /// ToggleFeature - Toggle a feature and returns the newly updated feature
  /// bits.
  FeatureBitset ToggleFeature(FeatureBitset Bits, StringRef String,
                         ArrayRef<SubtargetFeatureKV> FeatureTable);

  /// Apply the feature flag and return the newly updated feature bits.
  FeatureBitset ApplyFeatureFlag(FeatureBitset Bits, StringRef Feature,
                                 ArrayRef<SubtargetFeatureKV> FeatureTable);

  /// Get feature bits of a CPU.
  FeatureBitset getFeatureBits(StringRef CPU,
                          ArrayRef<SubtargetFeatureKV> CPUTable,
                          ArrayRef<SubtargetFeatureKV> FeatureTable);

  /// Print feature string.
  void print(raw_ostream &OS) const;

  // Dump feature info.
  void dump() const;

  /// Adds the default features for the specified target triple.
  void getDefaultSubtargetFeatures(const Triple& Triple);
};

} // End namespace llvm

#endif
