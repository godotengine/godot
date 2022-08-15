//===- JITEventListener.h - Exposes events from JIT compilation -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the JITEventListener interface, which lets users get
// callbacks when significant events happen during the JIT compilation process.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITEVENTLISTENER_H
#define LLVM_EXECUTIONENGINE_JITEVENTLISTENER_H

#include "RuntimeDyld.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Support/DataTypes.h"
#include <vector>

namespace llvm {
class Function;
class MachineFunction;
class OProfileWrapper;
class IntelJITEventsWrapper;

namespace object {
  class ObjectFile;
}

/// JITEvent_EmittedFunctionDetails - Helper struct for containing information
/// about a generated machine code function.
struct JITEvent_EmittedFunctionDetails {
  struct LineStart {
    /// The address at which the current line changes.
    uintptr_t Address;

    /// The new location information.  These can be translated to DebugLocTuples
    /// using MF->getDebugLocTuple().
    DebugLoc Loc;
  };

  /// The machine function the struct contains information for.
  const MachineFunction *MF;

  /// The list of line boundary information, sorted by address.
  std::vector<LineStart> LineStarts;
};

/// JITEventListener - Abstract interface for use by the JIT to notify clients
/// about significant events during compilation. For example, to notify
/// profilers and debuggers that need to know where functions have been emitted.
///
/// The default implementation of each method does nothing.
class JITEventListener {
public:
  typedef JITEvent_EmittedFunctionDetails EmittedFunctionDetails;

public:
  JITEventListener() {}
  virtual ~JITEventListener() {}

  /// NotifyObjectEmitted - Called after an object has been successfully
  /// emitted to memory.  NotifyFunctionEmitted will not be called for
  /// individual functions in the object.
  ///
  /// ELF-specific information
  /// The ObjectImage contains the generated object image
  /// with section headers updated to reflect the address at which sections
  /// were loaded and with relocations performed in-place on debug sections.
  virtual void NotifyObjectEmitted(const object::ObjectFile &Obj,
                                   const RuntimeDyld::LoadedObjectInfo &L) {}

  /// NotifyFreeingObject - Called just before the memory associated with
  /// a previously emitted object is released.
  virtual void NotifyFreeingObject(const object::ObjectFile &Obj) {}

  // Get a pointe to the GDB debugger registration listener.
  static JITEventListener *createGDBRegistrationListener();

#if LLVM_USE_INTEL_JITEVENTS
  // Construct an IntelJITEventListener
  static JITEventListener *createIntelJITEventListener();

  // Construct an IntelJITEventListener with a test Intel JIT API implementation
  static JITEventListener *createIntelJITEventListener(
                                      IntelJITEventsWrapper* AlternativeImpl);
#else
  static JITEventListener *createIntelJITEventListener() { return nullptr; }

  static JITEventListener *createIntelJITEventListener(
                                      IntelJITEventsWrapper* AlternativeImpl) {
    return nullptr;
  }
#endif // USE_INTEL_JITEVENTS

#if LLVM_USE_OPROFILE
  // Construct an OProfileJITEventListener
  static JITEventListener *createOProfileJITEventListener();

  // Construct an OProfileJITEventListener with a test opagent implementation
  static JITEventListener *createOProfileJITEventListener(
                                      OProfileWrapper* AlternativeImpl);
#else

  static JITEventListener *createOProfileJITEventListener() { return nullptr; }

  static JITEventListener *createOProfileJITEventListener(
                                      OProfileWrapper* AlternativeImpl) {
    return nullptr;
  }
#endif // USE_OPROFILE
private:
  virtual void anchor();
};

} // end namespace llvm.

#endif // defined LLVM_EXECUTIONENGINE_JITEVENTLISTENER_H
