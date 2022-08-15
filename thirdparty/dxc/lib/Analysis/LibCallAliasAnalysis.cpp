//===- LibCallAliasAnalysis.cpp - Implement AliasAnalysis for libcalls ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LibCallAliasAnalysis class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LibCallAliasAnalysis.h"
#include "llvm/Analysis/LibCallSemantics.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
using namespace llvm;
  
// Register this pass...
char LibCallAliasAnalysis::ID = 0;
INITIALIZE_AG_PASS(LibCallAliasAnalysis, AliasAnalysis, "libcall-aa",
                   "LibCall Alias Analysis", false, true, false)

FunctionPass *llvm::createLibCallAliasAnalysisPass(LibCallInfo *LCI) {
  return new LibCallAliasAnalysis(LCI);
}

LibCallAliasAnalysis::~LibCallAliasAnalysis() {
  delete LCI;
}

void LibCallAliasAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AliasAnalysis::getAnalysisUsage(AU);
  AU.setPreservesAll();                         // Does not transform code
}

bool LibCallAliasAnalysis::runOnFunction(Function &F) {
  // set up super class
  InitializeAliasAnalysis(this, &F.getParent()->getDataLayout());
  return false;
}

/// AnalyzeLibCallDetails - Given a call to a function with the specified
/// LibCallFunctionInfo, see if we can improve the mod/ref footprint of the call
/// vs the specified pointer/size.
AliasAnalysis::ModRefResult
LibCallAliasAnalysis::AnalyzeLibCallDetails(const LibCallFunctionInfo *FI,
                                            ImmutableCallSite CS,
                                            const MemoryLocation &Loc) {
  // If we have a function, check to see what kind of mod/ref effects it
  // has.  Start by including any info globally known about the function.
  AliasAnalysis::ModRefResult MRInfo = FI->UniversalBehavior;
  if (MRInfo == NoModRef) return MRInfo;
  
  // If that didn't tell us that the function is 'readnone', check to see
  // if we have detailed info and if 'P' is any of the locations we know
  // about.
  const LibCallFunctionInfo::LocationMRInfo *Details = FI->LocationDetails;
  if (Details == nullptr)
    return MRInfo;
  
  // If the details array is of the 'DoesNot' kind, we only know something if
  // the pointer is a match for one of the locations in 'Details'.  If we find a
  // match, we can prove some interactions cannot happen.
  // 
  if (FI->DetailsType == LibCallFunctionInfo::DoesNot) {
    // Find out if the pointer refers to a known location.
    for (unsigned i = 0; Details[i].LocationID != ~0U; ++i) {
      const LibCallLocationInfo &LocInfo =
      LCI->getLocationInfo(Details[i].LocationID);
      LibCallLocationInfo::LocResult Res = LocInfo.isLocation(CS, Loc);
      if (Res != LibCallLocationInfo::Yes) continue;
      
      // If we find a match against a location that we 'do not' interact with,
      // learn this info into MRInfo.
      return ModRefResult(MRInfo & ~Details[i].MRInfo);
    }
    return MRInfo;
  }
  
  // If the details are of the 'DoesOnly' sort, we know something if the pointer
  // is a match for one of the locations in 'Details'.  Also, if we can prove
  // that the pointers is *not* one of the locations in 'Details', we know that
  // the call is NoModRef.
  assert(FI->DetailsType == LibCallFunctionInfo::DoesOnly);
  
  // Find out if the pointer refers to a known location.
  bool NoneMatch = true;
  for (unsigned i = 0; Details[i].LocationID != ~0U; ++i) {
    const LibCallLocationInfo &LocInfo =
    LCI->getLocationInfo(Details[i].LocationID);
    LibCallLocationInfo::LocResult Res = LocInfo.isLocation(CS, Loc);
    if (Res == LibCallLocationInfo::No) continue;
    
    // If we don't know if this pointer points to the location, then we have to
    // assume it might alias in some case.
    if (Res == LibCallLocationInfo::Unknown) {
      NoneMatch = false;
      continue;
    }
    
    // If we know that this pointer definitely is pointing into the location,
    // merge in this information.
    return ModRefResult(MRInfo & Details[i].MRInfo);
  }
  
  // If we found that the pointer is guaranteed to not match any of the
  // locations in our 'DoesOnly' rule, then we know that the pointer must point
  // to some other location.  Since the libcall doesn't mod/ref any other
  // locations, return NoModRef.
  if (NoneMatch)
    return NoModRef;
  
  // Otherwise, return any other info gained so far.
  return MRInfo;
}

// getModRefInfo - Check to see if the specified callsite can clobber the
// specified memory object.
//
AliasAnalysis::ModRefResult
LibCallAliasAnalysis::getModRefInfo(ImmutableCallSite CS,
                                    const MemoryLocation &Loc) {
  ModRefResult MRInfo = ModRef;
  
  // If this is a direct call to a function that LCI knows about, get the
  // information about the runtime function.
  if (LCI) {
    if (const Function *F = CS.getCalledFunction()) {
      if (const LibCallFunctionInfo *FI = LCI->getFunctionInfo(F)) {
        MRInfo = ModRefResult(MRInfo & AnalyzeLibCallDetails(FI, CS, Loc));
        if (MRInfo == NoModRef) return NoModRef;
      }
    }
  }
  
  // The AliasAnalysis base class has some smarts, lets use them.
  return (ModRefResult)(MRInfo | AliasAnalysis::getModRefInfo(CS, Loc));
}
