//===- GVMaterializer.h - Interface for GV materializers --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides an abstract interface for loading a module from some
// place.  This interface allows incremental or random access loading of
// functions from the file.  This is useful for applications like JIT compilers
// or interprocedural optimizers that do not need the entire program in memory
// at the same time.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_GVMATERIALIZER_H
#define LLVM_IR_GVMATERIALIZER_H

#include <system_error>
#include <vector>
#include "llvm/ADT/ArrayRef.h" // HLSL Change
#include "llvm/ADT/StringRef.h" // HLSL Change

namespace llvm {
class Function;
class GlobalValue;
class Module;
class StructType;

class GVMaterializer {
protected:
  GVMaterializer() {}

public:
  virtual ~GVMaterializer();

  /// True if GV has been materialized and can be dematerialized back to
  /// whatever backing store this GVMaterializer uses.
  virtual bool isDematerializable(const GlobalValue *GV) const = 0;

  /// Make sure the given GlobalValue is fully read.
  ///
  virtual std::error_code materialize(GlobalValue *GV) = 0;

  /// If the given GlobalValue is read in, and if the GVMaterializer supports
  /// it, release the memory for the GV, and set it up to be materialized
  /// lazily. If the Materializer doesn't support this capability, this method
  /// is a noop.
  ///
  virtual void dematerialize(GlobalValue *) {}

  /// Make sure the entire Module has been completely read.
  ///
  virtual std::error_code materializeModule(Module *M) = 0;

  virtual std::error_code materializeMetadata() = 0;
  virtual std::error_code materializeSelectNamedMetadata(llvm::ArrayRef<llvm::StringRef>) = 0; // HLSL Change
  virtual void setStripDebugInfo() = 0;

  virtual std::vector<StructType *> getIdentifiedStructTypes() const = 0;
};

} // End llvm namespace

#endif
