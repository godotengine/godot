//===- DebugInfo.h - Debug Information Helpers ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a bunch of datatypes that are useful for creating and
// walking debug info in LLVM IR form. They essentially provide wrappers around
// the information in the global variables that's needed when constructing the
// DWARF information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_DEBUGINFO_H
#define LLVM_IR_DEBUGINFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
#include <iterator>

namespace llvm {
class Module;
class DbgDeclareInst;
class DbgValueInst;

/// \brief Maps from type identifier to the actual MDNode.
typedef DenseMap<const MDString *, DIType *> DITypeIdentifierMap;

/// \brief Find subprogram that is enclosing this scope.
DISubprogram *getDISubprogram(const MDNode *Scope);

/// \brief Find debug info for a given function.
///
/// \returns a valid subprogram, if found. Otherwise, return \c nullptr.
DISubprogram *getDISubprogram(const Function *F);

/// \brief Find underlying composite type.
DICompositeTypeBase *getDICompositeType(DIType *T);

/// \brief Generate map by visiting all retained types.
DITypeIdentifierMap generateDITypeIdentifierMap(const NamedMDNode *CU_Nodes);

/// \brief Strip debug info in the module if it exists.
///
/// To do this, we remove all calls to the debugger intrinsics and any named
/// metadata for debugging. We also remove debug locations for instructions.
/// Return true if module is modified.
bool StripDebugInfo(Module &M);
bool stripDebugInfo(Function &F);

/// \brief Return Debug Info Metadata Version by checking module flags.
unsigned getDebugMetadataVersionFromModule(const Module &M);
bool hasDebugInfo(const Module &M); // HLSL Change - Helper function to check if there's real debug info (variables, types)

/// \brief Utility to find all debug info in a module.
///
/// DebugInfoFinder tries to list all debug info MDNodes used in a module. To
/// list debug info MDNodes used by an instruction, DebugInfoFinder uses
/// processDeclare, processValue and processLocation to handle DbgDeclareInst,
/// DbgValueInst and DbgLoc attached to instructions. processModule will go
/// through all DICompileUnits in llvm.dbg.cu and list debug info MDNodes
/// used by the CUs.
class DebugInfoFinder {
public:
  DebugInfoFinder() : TypeMapInitialized(false) {}

  /// \brief Process entire module and collect debug info anchors.
  void processModule(const Module &M);

  /// \brief Process DbgDeclareInst.
  void processDeclare(const Module &M, const DbgDeclareInst *DDI);
  /// \brief Process DbgValueInst.
  void processValue(const Module &M, const DbgValueInst *DVI);
  /// \brief Process debug info location.
  void processLocation(const Module &M, const DILocation *Loc);

  /// \brief Clear all lists.
  void reset();

  // HLSL Change Begins.
  /// \brief Append new global variable.
  bool appendGlobalVariable(DIGlobalVariable *DIG);
  // HLSL Change Ends.
private:
  void InitializeTypeMap(const Module &M);

  void processType(DIType *DT);
  void processSubprogram(DISubprogram *SP);
  void processScope(DIScope *Scope);
  bool addCompileUnit(DICompileUnit *CU);
  bool addGlobalVariable(DIGlobalVariable *DIG);
  bool addSubprogram(DISubprogram *SP);
  bool addType(DIType *DT);
  bool addScope(DIScope *Scope);

public:
  typedef SmallVectorImpl<DICompileUnit *>::const_iterator
      compile_unit_iterator;
  typedef SmallVectorImpl<DISubprogram *>::const_iterator subprogram_iterator;
  typedef SmallVectorImpl<DIGlobalVariable *>::const_iterator
      global_variable_iterator;
  typedef SmallVectorImpl<DIType *>::const_iterator type_iterator;
  typedef SmallVectorImpl<DIScope *>::const_iterator scope_iterator;

  iterator_range<compile_unit_iterator> compile_units() const {
    return iterator_range<compile_unit_iterator>(CUs.begin(), CUs.end());
  }

  iterator_range<subprogram_iterator> subprograms() const {
    return iterator_range<subprogram_iterator>(SPs.begin(), SPs.end());
  }

  iterator_range<global_variable_iterator> global_variables() const {
    return iterator_range<global_variable_iterator>(GVs.begin(), GVs.end());
  }

  iterator_range<type_iterator> types() const {
    return iterator_range<type_iterator>(TYs.begin(), TYs.end());
  }

  iterator_range<scope_iterator> scopes() const {
    return iterator_range<scope_iterator>(Scopes.begin(), Scopes.end());
  }

  unsigned compile_unit_count() const { return CUs.size(); }
  unsigned global_variable_count() const { return GVs.size(); }
  unsigned subprogram_count() const { return SPs.size(); }
  unsigned type_count() const { return TYs.size(); }
  unsigned scope_count() const { return Scopes.size(); }

private:
  SmallVector<DICompileUnit *, 8> CUs;
  SmallVector<DISubprogram *, 8> SPs;
  SmallVector<DIGlobalVariable *, 8> GVs;
  SmallVector<DIType *, 8> TYs;
  SmallVector<DIScope *, 8> Scopes;
  SmallPtrSet<const MDNode *, 64> NodesSeen;
  DITypeIdentifierMap TypeIdentifierMap;

  /// \brief Specify if TypeIdentifierMap is initialized.
  bool TypeMapInitialized;
};

DenseMap<const Function *, DISubprogram *> makeSubprogramMap(const Module &M);

} // end namespace llvm

#endif
