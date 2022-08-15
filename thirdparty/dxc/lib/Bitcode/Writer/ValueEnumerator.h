//===-- Bitcode/Writer/ValueEnumerator.h - Number values --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class gives values and types Unique ID's.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_BITCODE_WRITER_VALUEENUMERATOR_H
#define LLVM_LIB_BITCODE_WRITER_VALUEENUMERATOR_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/UniqueVector.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/UseListOrder.h"
#include <vector>

namespace llvm {

class Type;
class Value;
class Instruction;
class BasicBlock;
class Comdat;
class Function;
class Module;
class Metadata;
class LocalAsMetadata;
class MDNode;
class NamedMDNode;
class AttributeSet;
class ValueSymbolTable;
class MDSymbolTable;
class raw_ostream;

class ValueEnumerator {
public:
  typedef std::vector<Type*> TypeList;

  // For each value, we remember its Value* and occurrence frequency.
  typedef std::vector<std::pair<const Value*, unsigned> > ValueList;

  UseListOrderStack UseListOrders;

private:
  typedef DenseMap<Type*, unsigned> TypeMapType;
  TypeMapType TypeMap;
  TypeList Types;

  typedef DenseMap<const Value*, unsigned> ValueMapType;
  ValueMapType ValueMap;
  ValueList Values;

  typedef UniqueVector<const Comdat *> ComdatSetType;
  ComdatSetType Comdats;

  std::vector<const Metadata *> MDs;
  SmallVector<const LocalAsMetadata *, 8> FunctionLocalMDs;
  typedef DenseMap<const Metadata *, unsigned> MetadataMapType;
  MetadataMapType MDValueMap;
  bool HasMDString;
  bool HasDILocation;
  bool HasGenericDINode;
  bool ShouldPreserveUseListOrder;

  typedef DenseMap<AttributeSet, unsigned> AttributeGroupMapType;
  AttributeGroupMapType AttributeGroupMap;
  std::vector<AttributeSet> AttributeGroups;

  typedef DenseMap<AttributeSet, unsigned> AttributeMapType;
  AttributeMapType AttributeMap;
  std::vector<AttributeSet> Attribute;

  /// GlobalBasicBlockIDs - This map memoizes the basic block ID's referenced by
  /// the "getGlobalBasicBlockID" method.
  mutable DenseMap<const BasicBlock*, unsigned> GlobalBasicBlockIDs;

  typedef DenseMap<const Instruction*, unsigned> InstructionMapType;
  InstructionMapType InstructionMap;
  unsigned InstructionCount;

  /// BasicBlocks - This contains all the basic blocks for the currently
  /// incorporated function.  Their reverse mapping is stored in ValueMap.
  std::vector<const BasicBlock*> BasicBlocks;

  /// When a function is incorporated, this is the size of the Values list
  /// before incorporation.
  unsigned NumModuleValues;

  /// When a function is incorporated, this is the size of the MDValues list
  /// before incorporation.
  unsigned NumModuleMDs;

  unsigned FirstFuncConstantID;
  unsigned FirstInstID;

  ValueEnumerator(const ValueEnumerator &) = delete;
  void operator=(const ValueEnumerator &) = delete;
public:
  ValueEnumerator(const Module &M, bool ShouldPreserveUseListOrder);

  void dump() const;
  void print(raw_ostream &OS, const ValueMapType &Map, const char *Name) const;
  void print(raw_ostream &OS, const MetadataMapType &Map,
             const char *Name) const;

  unsigned getValueID(const Value *V) const;
  unsigned getMetadataID(const Metadata *MD) const {
    auto ID = getMetadataOrNullID(MD);
    assert(ID != 0 && "Metadata not in slotcalculator!");
    return ID - 1;
  }
  unsigned getMetadataOrNullID(const Metadata *MD) const {
    return MDValueMap.lookup(MD);
  }

  bool hasMDString() const { return HasMDString; }
  bool hasDILocation() const { return HasDILocation; }
  bool hasGenericDINode() const { return HasGenericDINode; }

  bool shouldPreserveUseListOrder() const { return ShouldPreserveUseListOrder; }

  unsigned getTypeID(Type *T) const {
    TypeMapType::const_iterator I = TypeMap.find(T);
    assert(I != TypeMap.end() && "Type not in ValueEnumerator!");
    return I->second-1;
  }

  unsigned getInstructionID(const Instruction *I) const;
  void setInstructionID(const Instruction *I);

  unsigned getAttributeID(AttributeSet PAL) const {
    if (PAL.isEmpty()) return 0;  // Null maps to zero.
    AttributeMapType::const_iterator I = AttributeMap.find(PAL);
    assert(I != AttributeMap.end() && "Attribute not in ValueEnumerator!");
    return I->second;
  }

  unsigned getAttributeGroupID(AttributeSet PAL) const {
    if (PAL.isEmpty()) return 0;  // Null maps to zero.
    AttributeGroupMapType::const_iterator I = AttributeGroupMap.find(PAL);
    assert(I != AttributeGroupMap.end() && "Attribute not in ValueEnumerator!");
    return I->second;
  }

  /// getFunctionConstantRange - Return the range of values that corresponds to
  /// function-local constants.
  void getFunctionConstantRange(unsigned &Start, unsigned &End) const {
    Start = FirstFuncConstantID;
    End = FirstInstID;
  }

  const ValueList &getValues() const { return Values; }
  const std::vector<const Metadata *> &getMDs() const { return MDs; }
  const SmallVectorImpl<const LocalAsMetadata *> &getFunctionLocalMDs() const {
    return FunctionLocalMDs;
  }
  const TypeList &getTypes() const { return Types; }
  const std::vector<const BasicBlock*> &getBasicBlocks() const {
    return BasicBlocks;
  }
  const std::vector<AttributeSet> &getAttributes() const {
    return Attribute;
  }
  const std::vector<AttributeSet> &getAttributeGroups() const {
    return AttributeGroups;
  }

  const ComdatSetType &getComdats() const { return Comdats; }
  unsigned getComdatID(const Comdat *C) const;

  /// getGlobalBasicBlockID - This returns the function-specific ID for the
  /// specified basic block.  This is relatively expensive information, so it
  /// should only be used by rare constructs such as address-of-label.
  unsigned getGlobalBasicBlockID(const BasicBlock *BB) const;

  /// incorporateFunction/purgeFunction - If you'd like to deal with a function,
  /// use these two methods to get its data into the ValueEnumerator!
  ///
  void incorporateFunction(const Function &F);
  void purgeFunction();
  uint64_t computeBitsRequiredForTypeIndicies() const;

private:
  void OptimizeConstants(unsigned CstStart, unsigned CstEnd);

  void EnumerateMDNodeOperands(const MDNode *N);
  void EnumerateMetadata(const Metadata *MD);
  void EnumerateFunctionLocalMetadata(const LocalAsMetadata *Local);
  void EnumerateNamedMDNode(const NamedMDNode *NMD);
  void EnumerateValue(const Value *V);
  void EnumerateType(Type *T);
  void EnumerateOperandType(const Value *V);
  void EnumerateAttributes(AttributeSet PAL);

  void EnumerateValueSymbolTable(const ValueSymbolTable &ST);
  void EnumerateNamedMetadata(const Module &M);
};

} // End llvm namespace

#endif
