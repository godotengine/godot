//===- ValueMapper.h - Remapping for constants and metadata -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MapValue interface which is used by various parts of
// the Transforms/Utils library to implement cloning and linking facilities.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_VALUEMAPPER_H
#define LLVM_TRANSFORMS_UTILS_VALUEMAPPER_H

#include "llvm/IR/ValueMap.h"

namespace llvm {
  class Value;
  class Instruction;
  typedef ValueMap<const Value *, WeakVH> ValueToValueMapTy;

  /// ValueMapTypeRemapper - This is a class that can be implemented by clients
  /// to remap types when cloning constants and instructions.
  class ValueMapTypeRemapper {
    virtual void anchor();  // Out of line method.
  public:
    virtual ~ValueMapTypeRemapper() {}

    /// remapType - The client should implement this method if they want to
    /// remap types while mapping values.
    virtual Type *remapType(Type *SrcTy) = 0;
  };

  /// ValueMaterializer - This is a class that can be implemented by clients
  /// to materialize Values on demand.
  class ValueMaterializer {
    virtual void anchor(); // Out of line method.
  public:
    virtual ~ValueMaterializer() {}

    /// materializeValueFor - The client should implement this method if they
    /// want to generate a mapped Value on demand. For example, if linking
    /// lazily.
    virtual Value *materializeValueFor(Value *V) = 0;
  };

  /// RemapFlags - These are flags that the value mapping APIs allow.
  enum RemapFlags {
    RF_None = 0,

    /// RF_NoModuleLevelChanges - If this flag is set, the remapper knows that
    /// only local values within a function (such as an instruction or argument)
    /// are mapped, not global values like functions and global metadata.
    RF_NoModuleLevelChanges = 1,

    /// RF_IgnoreMissingEntries - If this flag is set, the remapper ignores
    /// entries that are not in the value map.  If it is unset, it aborts if an
    /// operand is asked to be remapped which doesn't exist in the mapping.
    RF_IgnoreMissingEntries = 2
  };

  static inline RemapFlags operator|(RemapFlags LHS, RemapFlags RHS) {
    return RemapFlags(unsigned(LHS)|unsigned(RHS));
  }

  Value *MapValue(const Value *V, ValueToValueMapTy &VM,
                  RemapFlags Flags = RF_None,
                  ValueMapTypeRemapper *TypeMapper = nullptr,
                  ValueMaterializer *Materializer = nullptr);

  Metadata *MapMetadata(const Metadata *MD, ValueToValueMapTy &VM,
                        RemapFlags Flags = RF_None,
                        ValueMapTypeRemapper *TypeMapper = nullptr,
                        ValueMaterializer *Materializer = nullptr);

  /// MapMetadata - provide versions that preserve type safety for MDNodes.
  MDNode *MapMetadata(const MDNode *MD, ValueToValueMapTy &VM,
                      RemapFlags Flags = RF_None,
                      ValueMapTypeRemapper *TypeMapper = nullptr,
                      ValueMaterializer *Materializer = nullptr);

  void RemapInstruction(Instruction *I, ValueToValueMapTy &VM,
                        RemapFlags Flags = RF_None,
                        ValueMapTypeRemapper *TypeMapper = nullptr,
                        ValueMaterializer *Materializer = nullptr);

  /// MapValue - provide versions that preserve type safety for Constants.
  inline Constant *MapValue(const Constant *V, ValueToValueMapTy &VM,
                            RemapFlags Flags = RF_None,
                            ValueMapTypeRemapper *TypeMapper = nullptr,
                            ValueMaterializer *Materializer = nullptr) {
    return cast<Constant>(MapValue((const Value*)V, VM, Flags, TypeMapper,
                                   Materializer));
  }

} // End llvm namespace

#endif
