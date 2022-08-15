//===- LibCallSemantics.cpp - Describe library semantics ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements interfaces that can be used to describe language
// specific runtime library interfaces (e.g. libc, libm, etc) to LLVM
// optimizers.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LibCallSemantics.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Function.h"
using namespace llvm;

/// This impl pointer in ~LibCallInfo is actually a StringMap.  This
/// helper does the cast.
static StringMap<const LibCallFunctionInfo*> *getMap(void *Ptr) {
  return static_cast<StringMap<const LibCallFunctionInfo*> *>(Ptr);
}

LibCallInfo::~LibCallInfo() {
  delete getMap(Impl);
}

const LibCallLocationInfo &LibCallInfo::getLocationInfo(unsigned LocID) const {
  // Get location info on the first call.
  if (NumLocations == 0)
    NumLocations = getLocationInfo(Locations);
  
  assert(LocID < NumLocations && "Invalid location ID!");
  return Locations[LocID];
}


/// Return the LibCallFunctionInfo object corresponding to
/// the specified function if we have it.  If not, return null.
const LibCallFunctionInfo *
LibCallInfo::getFunctionInfo(const Function *F) const {
  StringMap<const LibCallFunctionInfo*> *Map = getMap(Impl);
  
  /// If this is the first time we are querying for this info, lazily construct
  /// the StringMap to index it.
  if (!Map) {
    Impl = Map = new StringMap<const LibCallFunctionInfo*>();
    
    const LibCallFunctionInfo *Array = getFunctionInfoArray();
    if (!Array) return nullptr;
    
    // We now have the array of entries.  Populate the StringMap.
    for (unsigned i = 0; Array[i].Name; ++i)
      (*Map)[Array[i].Name] = Array+i;
  }
  
  // Look up this function in the string map.
  return Map->lookup(F->getName());
}

/// See if the given exception handling personality function is one that we
/// understand.  If so, return a description of it; otherwise return Unknown.
EHPersonality llvm::classifyEHPersonality(const Value *Pers) {
  const Function *F = dyn_cast<Function>(Pers->stripPointerCasts());
  if (!F)
    return EHPersonality::Unknown;
  return StringSwitch<EHPersonality>(F->getName())
    .Case("__gnat_eh_personality", EHPersonality::GNU_Ada)
    .Case("__gxx_personality_v0",  EHPersonality::GNU_CXX)
    .Case("__gcc_personality_v0",  EHPersonality::GNU_C)
    .Case("__objc_personality_v0", EHPersonality::GNU_ObjC)
    .Case("_except_handler3",      EHPersonality::MSVC_X86SEH)
    .Case("_except_handler4",      EHPersonality::MSVC_X86SEH)
    .Case("__C_specific_handler",  EHPersonality::MSVC_Win64SEH)
    .Case("__CxxFrameHandler3",    EHPersonality::MSVC_CXX)
    .Default(EHPersonality::Unknown);
}

bool llvm::canSimplifyInvokeNoUnwind(const Function *F) {
  EHPersonality Personality = classifyEHPersonality(F->getPersonalityFn());
  // We can't simplify any invokes to nounwind functions if the personality
  // function wants to catch asynch exceptions.  The nounwind attribute only
  // implies that the function does not throw synchronous exceptions.
  return !isAsynchronousEHPersonality(Personality);
}
