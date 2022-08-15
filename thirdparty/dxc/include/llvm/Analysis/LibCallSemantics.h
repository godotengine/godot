//===- LibCallSemantics.h - Describe library semantics --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces that can be used to describe language specific
// runtime library interfaces (e.g. libc, libm, etc) to LLVM optimizers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LIBCALLSEMANTICS_H
#define LLVM_ANALYSIS_LIBCALLSEMANTICS_H

#include "llvm/Analysis/AliasAnalysis.h"

namespace llvm {
class InvokeInst;

  /// LibCallLocationInfo - This struct describes a set of memory locations that
  /// are accessed by libcalls.  Identification of a location is doing with a
  /// simple callback function.
  ///
  /// For example, the LibCallInfo may be set up to model the behavior of
  /// standard libm functions.  The location that they may be interested in is
  /// an abstract location that represents errno for the current target.  In
  /// this case, a location for errno is anything such that the predicate
  /// returns true.  On Mac OS X, this predicate would return true if the
  /// pointer is the result of a call to "__error()".
  ///
  /// Locations can also be defined in a constant-sensitive way.  For example,
  /// it is possible to define a location that returns true iff it is passed
  /// into the call as a specific argument.  This is useful for modeling things
  /// like "printf", which can store to memory, but only through pointers passed
  /// with a '%n' constraint.
  ///
  struct LibCallLocationInfo {
    // TODO: Flags: isContextSensitive etc.
    
    /// isLocation - Return a LocResult if the specified pointer refers to this
    /// location for the specified call site.  This returns "Yes" if we can tell
    /// that the pointer *does definitely* refer to the location, "No" if we can
    /// tell that the location *definitely does not* refer to the location, and
    /// returns "Unknown" if we cannot tell for certain.
    enum LocResult {
      Yes, No, Unknown
    };
    LocResult (*isLocation)(ImmutableCallSite CS, const MemoryLocation &Loc);
  };
  
  /// LibCallFunctionInfo - Each record in the array of FunctionInfo structs
  /// records the behavior of one libcall that is known by the optimizer.  This
  /// captures things like the side effects of the call.  Side effects are
  /// modeled both universally (in the readnone/readonly) sense, but also
  /// potentially against a set of abstract locations defined by the optimizer.
  /// This allows an optimizer to define that some libcall (e.g. sqrt) is
  /// side-effect free except that it might modify errno (thus, the call is
  /// *not* universally readonly).  Or it might say that the side effects
  /// are unknown other than to say that errno is not modified.
  ///
  struct LibCallFunctionInfo {
    /// Name - This is the name of the libcall this describes.
    const char *Name;
    
    /// TODO: Constant folding function: Constant* vector -> Constant*.
    
    /// UniversalBehavior - This captures the absolute mod/ref behavior without
    /// any specific context knowledge.  For example, if the function is known
    /// to be readonly, this would be set to 'ref'.  If known to be readnone,
    /// this is set to NoModRef.
    AliasAnalysis::ModRefResult UniversalBehavior;
    
    /// LocationMRInfo - This pair captures info about whether a specific
    /// location is modified or referenced by a libcall.
    struct LocationMRInfo {
      /// LocationID - ID # of the accessed location or ~0U for array end.
      unsigned LocationID;
      /// MRInfo - Mod/Ref info for this location.
      AliasAnalysis::ModRefResult MRInfo;
    };
    
    /// DetailsType - Indicate the sense of the LocationDetails array.  This
    /// controls how the LocationDetails array is interpreted.
    enum {
      /// DoesOnly - If DetailsType is set to DoesOnly, then we know that the
      /// *only* mod/ref behavior of this function is captured by the
      /// LocationDetails array.  If we are trying to say that 'sqrt' can only
      /// modify errno, we'd have the {errnoloc,mod} in the LocationDetails
      /// array and have DetailsType set to DoesOnly.
      DoesOnly,
      
      /// DoesNot - If DetailsType is set to DoesNot, then the sense of the
      /// LocationDetails array is completely inverted.  This means that we *do
      /// not* know everything about the side effects of this libcall, but we do
      /// know things that the libcall cannot do.  This is useful for complex
      /// functions like 'ctime' which have crazy mod/ref behavior, but are
      /// known to never read or write errno.  In this case, we'd have
      /// {errnoloc,modref} in the LocationDetails array and DetailsType would
      /// be set to DoesNot, indicating that ctime does not read or write the
      /// errno location.
      DoesNot
    } DetailsType;
    
    /// LocationDetails - This is a pointer to an array of LocationMRInfo
    /// structs which indicates the behavior of the libcall w.r.t. specific
    /// locations.  For example, if this libcall is known to only modify
    /// 'errno', it would have a LocationDetails array with the errno ID and
    /// 'mod' in it.  See the DetailsType field for how this is interpreted.
    ///
    /// In the "DoesOnly" case, this information is 'may' information for: there
    /// is no guarantee that the specified side effect actually does happen,
    /// just that it could.  In the "DoesNot" case, this is 'must not' info.
    ///
    /// If this pointer is null, no details are known.
    ///
    const LocationMRInfo *LocationDetails;
  };
  
  
  /// LibCallInfo - Abstract interface to query about library call information.
  /// Instances of this class return known information about some set of
  /// libcalls.
  /// 
  class LibCallInfo {
    // Implementation details of this object, private.
    mutable void *Impl;
    mutable const LibCallLocationInfo *Locations;
    mutable unsigned NumLocations;
  public:
    LibCallInfo() : Impl(nullptr), Locations(nullptr), NumLocations(0) {}
    virtual ~LibCallInfo();
    
    //===------------------------------------------------------------------===//
    //  Accessor Methods: Efficient access to contained data.
    //===------------------------------------------------------------------===//
    
    /// getLocationInfo - Return information about the specified LocationID.
    const LibCallLocationInfo &getLocationInfo(unsigned LocID) const;
    
    
    /// getFunctionInfo - Return the LibCallFunctionInfo object corresponding to
    /// the specified function if we have it.  If not, return null.
    const LibCallFunctionInfo *getFunctionInfo(const Function *F) const;
    
    
    //===------------------------------------------------------------------===//
    //  Implementation Methods: Subclasses should implement these.
    //===------------------------------------------------------------------===//
    
    /// getLocationInfo - Return descriptors for the locations referenced by
    /// this set of libcalls.
    virtual unsigned getLocationInfo(const LibCallLocationInfo *&Array) const {
      return 0;
    }
    
    /// getFunctionInfoArray - Return an array of descriptors that describe the
    /// set of libcalls represented by this LibCallInfo object.  This array is
    /// terminated by an entry with a NULL name.
    virtual const LibCallFunctionInfo *getFunctionInfoArray() const = 0;
  };

  enum class EHPersonality {
    Unknown,
    GNU_Ada,
    GNU_C,
    GNU_CXX,
    GNU_ObjC,
    MSVC_X86SEH,
    MSVC_Win64SEH,
    MSVC_CXX,
  };

  /// \brief See if the given exception handling personality function is one
  /// that we understand.  If so, return a description of it; otherwise return
  /// Unknown.
  EHPersonality classifyEHPersonality(const Value *Pers);

  /// \brief Returns true if this personality function catches asynchronous
  /// exceptions.
  inline bool isAsynchronousEHPersonality(EHPersonality Pers) {
    // The two SEH personality functions can catch asynch exceptions. We assume
    // unknown personalities don't catch asynch exceptions.
    switch (Pers) {
    case EHPersonality::MSVC_X86SEH:
    case EHPersonality::MSVC_Win64SEH:
      return true;
    default: return false;
    }
    llvm_unreachable("invalid enum");
  }

  /// \brief Returns true if this is an MSVC personality function.
  inline bool isMSVCEHPersonality(EHPersonality Pers) {
    // The two SEH personality functions can catch asynch exceptions. We assume
    // unknown personalities don't catch asynch exceptions.
    switch (Pers) {
    case EHPersonality::MSVC_CXX:
    case EHPersonality::MSVC_X86SEH:
    case EHPersonality::MSVC_Win64SEH:
      return true;
    default: return false;
    }
    llvm_unreachable("invalid enum");
  }

  /// \brief Return true if this personality may be safely removed if there
  /// are no invoke instructions remaining in the current function.
  inline bool isNoOpWithoutInvoke(EHPersonality Pers) {
    switch (Pers) {
    case EHPersonality::Unknown:
      return false;
    // All known personalities currently have this behavior
    default: return true;
    }
    llvm_unreachable("invalid enum");
  }

  bool canSimplifyInvokeNoUnwind(const Function *F);

} // end namespace llvm

#endif
