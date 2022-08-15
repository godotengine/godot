//===-------------------------- TargetRecip.cpp ---------------------------===//
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

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetRecip.h"
#include <map>

using namespace llvm;

// These are the names of the individual reciprocal operations. These are
// the key strings for queries and command-line inputs.
// In addition, the command-line interface recognizes the global parameters
// "all", "none", and "default".
static const char *RecipOps[] = {
  "divd",
  "divf",
  "vec-divd",
  "vec-divf",
  "sqrtd",
  "sqrtf",
  "vec-sqrtd",
  "vec-sqrtf",
};

// The uninitialized state is needed for the enabled settings and refinement
// steps because custom settings may arrive via the command-line before target
// defaults are set.
TargetRecip::TargetRecip() {
  unsigned NumStrings = llvm::array_lengthof(RecipOps);
  for (unsigned i = 0; i < NumStrings; ++i)
    RecipMap.insert(std::make_pair(RecipOps[i], RecipParams()));
}

static bool parseRefinementStep(const StringRef &In, size_t &Position,
                                uint8_t &Value) {
  const char RefStepToken = ':';
  Position = In.find(RefStepToken);
  if (Position == StringRef::npos)
    return false;

  StringRef RefStepString = In.substr(Position + 1);
  // Allow exactly one numeric character for the additional refinement
  // step parameter.
  if (RefStepString.size() == 1) {
    char RefStepChar = RefStepString[0];
    if (RefStepChar >= '0' && RefStepChar <= '9') {
      Value = RefStepChar - '0';
      return true;
    }
  }
  report_fatal_error("Invalid refinement step for -recip.");
}

bool TargetRecip::parseGlobalParams(const std::string &Arg) {
  StringRef ArgSub = Arg;

  // Look for an optional setting of the number of refinement steps needed
  // for this type of reciprocal operation.
  size_t RefPos;
  uint8_t RefSteps;
  StringRef RefStepString;
  if (parseRefinementStep(ArgSub, RefPos, RefSteps)) {
    // Split the string for further processing.
    RefStepString = ArgSub.substr(RefPos + 1);
    ArgSub = ArgSub.substr(0, RefPos);
  }
  bool Enable;
  bool UseDefaults;
  if (ArgSub == "all") {
    UseDefaults = false;
    Enable = true;
  } else if (ArgSub == "none") {
    UseDefaults = false;
    Enable = false;
  } else if (ArgSub == "default") {
    UseDefaults = true;
  } else {
    // Any other string is invalid or an individual setting.
    return false;
  }

  // All enable values will be initialized to target defaults if 'default' was
  // specified.
  if (!UseDefaults)
    for (auto &KV : RecipMap)
      KV.second.Enabled = Enable;

  // Custom refinement count was specified with all, none, or default.
  if (!RefStepString.empty())
    for (auto &KV : RecipMap)
      KV.second.RefinementSteps = RefSteps;
  
  return true;
}

void TargetRecip::parseIndividualParams(const std::vector<std::string> &Args) {
  static const char DisabledPrefix = '!';
  unsigned NumArgs = Args.size();

  for (unsigned i = 0; i != NumArgs; ++i) {
    StringRef Val = Args[i];
    
    bool IsDisabled = Val[0] == DisabledPrefix;
    // Ignore the disablement token for string matching.
    if (IsDisabled)
      Val = Val.substr(1);
    
    size_t RefPos;
    uint8_t RefSteps;
    StringRef RefStepString;
    if (parseRefinementStep(Val, RefPos, RefSteps)) {
      // Split the string for further processing.
      RefStepString = Val.substr(RefPos + 1);
      Val = Val.substr(0, RefPos);
    }

    RecipIter Iter = RecipMap.find(Val);
    if (Iter == RecipMap.end()) {
      // Try again specifying float suffix.
      Iter = RecipMap.find(Val.str() + 'f');
      if (Iter == RecipMap.end()) {
        Iter = RecipMap.find(Val.str() + 'd');
        assert(Iter == RecipMap.end() && "Float entry missing from map");
        report_fatal_error("Invalid option for -recip.");
      }
      
      // The option was specified without a float or double suffix.
      if (RecipMap[Val.str() + 'd'].Enabled != Uninitialized) {
        // Make sure that the double entry was not already specified.
        // The float entry will be checked below.
        report_fatal_error("Duplicate option for -recip.");
      }
    }
    
    if (Iter->second.Enabled != Uninitialized)
      report_fatal_error("Duplicate option for -recip.");
    
    // Mark the matched option as found. Do not allow duplicate specifiers.
    Iter->second.Enabled = !IsDisabled;
    if (!RefStepString.empty())
      Iter->second.RefinementSteps = RefSteps;
    
    // If the precision was not specified, the double entry is also initialized.
    if (Val.back() != 'f' && Val.back() != 'd') {
      RecipMap[Val.str() + 'd'].Enabled = !IsDisabled;
      if (!RefStepString.empty())
        RecipMap[Val.str() + 'd'].RefinementSteps = RefSteps;
    }
  }
}

TargetRecip::TargetRecip(const std::vector<std::string> &Args) :
  TargetRecip() {
  unsigned NumArgs = Args.size();

  // Check if "all", "default", or "none" was specified.
  if (NumArgs == 1 && parseGlobalParams(Args[0]))
    return;
 
  parseIndividualParams(Args);
}

bool TargetRecip::isEnabled(const StringRef &Key) const {
  ConstRecipIter Iter = RecipMap.find(Key);
  assert(Iter != RecipMap.end() && "Unknown name for reciprocal map");
  assert(Iter->second.Enabled != Uninitialized &&
         "Enablement setting was not initialized");
  return Iter->second.Enabled;
}

unsigned TargetRecip::getRefinementSteps(const StringRef &Key) const {
  ConstRecipIter Iter = RecipMap.find(Key);
  assert(Iter != RecipMap.end() && "Unknown name for reciprocal map");
  assert(Iter->second.RefinementSteps != Uninitialized &&
         "Refinement step setting was not initialized");
  return Iter->second.RefinementSteps;
}

/// Custom settings (previously initialized values) override target defaults.
void TargetRecip::setDefaults(const StringRef &Key, bool Enable,
                              unsigned RefSteps) {
  if (Key == "all") {
    for (auto &KV : RecipMap) {
      RecipParams &RP = KV.second;
      if (RP.Enabled == Uninitialized)
        RP.Enabled = Enable;
      if (RP.RefinementSteps == Uninitialized)
        RP.RefinementSteps = RefSteps;
    }
  } else {
    RecipParams &RP = RecipMap[Key];
    if (RP.Enabled == Uninitialized)
      RP.Enabled = Enable;
    if (RP.RefinementSteps == Uninitialized)
      RP.RefinementSteps = RefSteps;
  }
}

bool TargetRecip::operator==(const TargetRecip &Other) const {
  for (const auto &KV : RecipMap) {
    const StringRef &Op = KV.first;
    const RecipParams &RP = KV.second;
    const RecipParams &OtherRP = Other.RecipMap.find(Op)->second;
    if (RP.RefinementSteps != OtherRP.RefinementSteps)
      return false;
    if (RP.Enabled != OtherRP.Enabled)
      return false;
  }
  return true;
}
