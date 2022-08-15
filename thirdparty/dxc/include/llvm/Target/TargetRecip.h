//===--------------------- llvm/Target/TargetRecip.h ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class is used to customize machine-specific reciprocal estimate code
// generation in a target-independent way.
// If a target does not support operations in this specification, then code
// generation will default to using supported operations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETRECIP_H
#define LLVM_TARGET_TARGETRECIP_H

#include "llvm/ADT/StringRef.h"
#include <vector>
#include <string>
#include <map>

namespace llvm {

struct TargetRecip {
public:
  TargetRecip();

  /// Initialize all or part of the operations from command-line options or
  /// a front end.
  TargetRecip(const std::vector<std::string> &Args);
  
  /// Set whether a particular reciprocal operation is enabled and how many
  /// refinement steps are needed when using it. Use "all" to set enablement
  /// and refinement steps for all operations.
  void setDefaults(const StringRef &Key, bool Enable, unsigned RefSteps);

  /// Return true if the reciprocal operation has been enabled by default or
  /// from the command-line. Return false if the operation has been disabled
  /// by default or from the command-line.
  bool isEnabled(const StringRef &Key) const;

  /// Return the number of iterations necessary to refine the
  /// the result of a machine instruction for the given reciprocal operation.
  unsigned getRefinementSteps(const StringRef &Key) const;

  bool operator==(const TargetRecip &Other) const;

private:
  enum {
    Uninitialized = -1
  };
  
  struct RecipParams {
    int8_t Enabled;
    int8_t RefinementSteps;
    
    RecipParams() : Enabled(Uninitialized), RefinementSteps(Uninitialized) {}
  };
  
  std::map<StringRef, RecipParams> RecipMap;
  typedef std::map<StringRef, RecipParams>::iterator RecipIter;
  typedef std::map<StringRef, RecipParams>::const_iterator ConstRecipIter;

  bool parseGlobalParams(const std::string &Arg);
  void parseIndividualParams(const std::vector<std::string> &Args);
};

} // End llvm namespace

#endif
