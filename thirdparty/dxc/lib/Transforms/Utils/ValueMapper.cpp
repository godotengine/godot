//===- ValueMapper.cpp - Interface shared by lib/Transforms/Utils ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MapValue function, which is shared by various parts of
// the lib/Transforms/Utils library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
using namespace llvm;

// Out of line method to get vtable etc for class.
void ValueMapTypeRemapper::anchor() {}
void ValueMaterializer::anchor() {}

Value *llvm::MapValue(const Value *V, ValueToValueMapTy &VM, RemapFlags Flags,
                      ValueMapTypeRemapper *TypeMapper,
                      ValueMaterializer *Materializer) {
  ValueToValueMapTy::iterator I = VM.find(V);
  
  // If the value already exists in the map, use it.
  if (I != VM.end() && I->second) return I->second;
  
  // If we have a materializer and it can materialize a value, use that.
  if (Materializer) {
    if (Value *NewV = Materializer->materializeValueFor(const_cast<Value*>(V)))
      return VM[V] = NewV;
  }

  // Global values do not need to be seeded into the VM if they
  // are using the identity mapping.
  if (isa<GlobalValue>(V))
    return VM[V] = const_cast<Value*>(V);
  
  if (const InlineAsm *IA = dyn_cast<InlineAsm>(V)) {
    // Inline asm may need *type* remapping.
    FunctionType *NewTy = IA->getFunctionType();
    if (TypeMapper) {
      NewTy = cast<FunctionType>(TypeMapper->remapType(NewTy));

      if (NewTy != IA->getFunctionType())
        V = InlineAsm::get(NewTy, IA->getAsmString(), IA->getConstraintString(),
                           IA->hasSideEffects(), IA->isAlignStack());
    }
    
    return VM[V] = const_cast<Value*>(V);
  }

  if (const auto *MDV = dyn_cast<MetadataAsValue>(V)) {
    const Metadata *MD = MDV->getMetadata();
    // If this is a module-level metadata and we know that nothing at the module
    // level is changing, then use an identity mapping.
    if (!isa<LocalAsMetadata>(MD) && (Flags & RF_NoModuleLevelChanges))
      return VM[V] = const_cast<Value *>(V);

    auto *MappedMD = MapMetadata(MD, VM, Flags, TypeMapper, Materializer);
    if (MD == MappedMD || (!MappedMD && (Flags & RF_IgnoreMissingEntries)))
      return VM[V] = const_cast<Value *>(V);

    // FIXME: This assert crashes during bootstrap, but I think it should be
    // correct.  For now, just match behaviour from before the metadata/value
    // split.
    //
    //    assert(MappedMD && "Referenced metadata value not in value map");
    return VM[V] = MetadataAsValue::get(V->getContext(), MappedMD);
  }

  // Okay, this either must be a constant (which may or may not be mappable) or
  // is something that is not in the mapping table.
  Constant *C = const_cast<Constant*>(dyn_cast<Constant>(V));
  if (!C)
    return nullptr;
  
  if (BlockAddress *BA = dyn_cast<BlockAddress>(C)) {
    Function *F = 
      cast<Function>(MapValue(BA->getFunction(), VM, Flags, TypeMapper, Materializer));
    BasicBlock *BB = cast_or_null<BasicBlock>(MapValue(BA->getBasicBlock(), VM,
                                                       Flags, TypeMapper, Materializer));
    return VM[V] = BlockAddress::get(F, BB ? BB : BA->getBasicBlock());
  }
  
  // Otherwise, we have some other constant to remap.  Start by checking to see
  // if all operands have an identity remapping.
  unsigned OpNo = 0, NumOperands = C->getNumOperands();
  Value *Mapped = nullptr;
  for (; OpNo != NumOperands; ++OpNo) {
    Value *Op = C->getOperand(OpNo);
    Mapped = MapValue(Op, VM, Flags, TypeMapper, Materializer);
    if (Mapped != C) break;
  }
  
  // See if the type mapper wants to remap the type as well.
  Type *NewTy = C->getType();
  if (TypeMapper)
    NewTy = TypeMapper->remapType(NewTy);

  // If the result type and all operands match up, then just insert an identity
  // mapping.
  if (OpNo == NumOperands && NewTy == C->getType())
    return VM[V] = C;
  
  // Okay, we need to create a new constant.  We've already processed some or
  // all of the operands, set them all up now.
  SmallVector<Constant*, 8> Ops;
  Ops.reserve(NumOperands);
  for (unsigned j = 0; j != OpNo; ++j)
    Ops.push_back(cast<Constant>(C->getOperand(j)));
  
  // If one of the operands mismatch, push it and the other mapped operands.
  if (OpNo != NumOperands) {
    Ops.push_back(cast<Constant>(Mapped));
  
    // Map the rest of the operands that aren't processed yet.
    for (++OpNo; OpNo != NumOperands; ++OpNo)
      Ops.push_back(MapValue(cast<Constant>(C->getOperand(OpNo)), VM,
                             Flags, TypeMapper, Materializer));
  }
  
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C))
    return VM[V] = CE->getWithOperands(Ops, NewTy);
  if (isa<ConstantArray>(C))
    return VM[V] = ConstantArray::get(cast<ArrayType>(NewTy), Ops);
  if (isa<ConstantStruct>(C))
    return VM[V] = ConstantStruct::get(cast<StructType>(NewTy), Ops);
  if (isa<ConstantVector>(C))
    return VM[V] = ConstantVector::get(Ops);
  // If this is a no-operand constant, it must be because the type was remapped.
  if (isa<UndefValue>(C))
    return VM[V] = UndefValue::get(NewTy);
  if (isa<ConstantAggregateZero>(C))
    return VM[V] = ConstantAggregateZero::get(NewTy);
  assert(isa<ConstantPointerNull>(C));
  return VM[V] = ConstantPointerNull::get(cast<PointerType>(NewTy));
}

static Metadata *mapToMetadata(ValueToValueMapTy &VM, const Metadata *Key,
                     Metadata *Val) {
  VM.MD()[Key].reset(Val);
  return Val;
}

static Metadata *mapToSelf(ValueToValueMapTy &VM, const Metadata *MD) {
  return mapToMetadata(VM, MD, const_cast<Metadata *>(MD));
}

static Metadata *MapMetadataImpl(const Metadata *MD,
                                 SmallVectorImpl<MDNode *> &Cycles,
                                 ValueToValueMapTy &VM, RemapFlags Flags,
                                 ValueMapTypeRemapper *TypeMapper,
                                 ValueMaterializer *Materializer);

static Metadata *mapMetadataOp(Metadata *Op, SmallVectorImpl<MDNode *> &Cycles,
                               ValueToValueMapTy &VM, RemapFlags Flags,
                               ValueMapTypeRemapper *TypeMapper,
                               ValueMaterializer *Materializer) {
  if (!Op)
    return nullptr;
  if (Metadata *MappedOp =
          MapMetadataImpl(Op, Cycles, VM, Flags, TypeMapper, Materializer))
    return MappedOp;
  // Use identity map if MappedOp is null and we can ignore missing entries.
  if (Flags & RF_IgnoreMissingEntries)
    return Op;

  // FIXME: This assert crashes during bootstrap, but I think it should be
  // correct.  For now, just match behaviour from before the metadata/value
  // split.
  //
  //    llvm_unreachable("Referenced metadata not in value map!");
  return nullptr;
}

/// \brief Remap nodes.
///
/// Insert \c NewNode in the value map, and then remap \c OldNode's operands.
/// Assumes that \c NewNode is already a clone of \c OldNode.
///
/// \pre \c NewNode is a clone of \c OldNode.
static bool remap(const MDNode *OldNode, MDNode *NewNode,
                  SmallVectorImpl<MDNode *> &Cycles, ValueToValueMapTy &VM,
                  RemapFlags Flags, ValueMapTypeRemapper *TypeMapper,
                  ValueMaterializer *Materializer) {
  assert(OldNode->getNumOperands() == NewNode->getNumOperands() &&
         "Expected nodes to match");
  assert(OldNode->isResolved() && "Expected resolved node");
  assert(!NewNode->isUniqued() && "Expected non-uniqued node");

  // Map the node upfront so it's available for cyclic references.
  mapToMetadata(VM, OldNode, NewNode);
  bool AnyChanged = false;
  for (unsigned I = 0, E = OldNode->getNumOperands(); I != E; ++I) {
    Metadata *Old = OldNode->getOperand(I);
    assert(NewNode->getOperand(I) == Old &&
           "Expected old operands to already be in place");

    Metadata *New = mapMetadataOp(OldNode->getOperand(I), Cycles, VM, Flags,
                                  TypeMapper, Materializer);
    if (Old != New) {
      AnyChanged = true;
      NewNode->replaceOperandWith(I, New);
    }
  }

  return AnyChanged;
}

/// \brief Map a distinct MDNode.
///
/// Distinct nodes are not uniqued, so they must always recreated.
static Metadata *mapDistinctNode(const MDNode *Node,
                                 SmallVectorImpl<MDNode *> &Cycles,
                                 ValueToValueMapTy &VM, RemapFlags Flags,
                                 ValueMapTypeRemapper *TypeMapper,
                                 ValueMaterializer *Materializer) {
  assert(Node->isDistinct() && "Expected distinct node");

  MDNode *NewMD = MDNode::replaceWithDistinct(Node->clone());
  remap(Node, NewMD, Cycles, VM, Flags, TypeMapper, Materializer);

  // Track any cycles beneath this node.
  for (Metadata *Op : NewMD->operands())
    if (auto *Node = dyn_cast_or_null<MDNode>(Op))
      if (!Node->isResolved())
        Cycles.push_back(Node);

  return NewMD;
}

/// \brief Map a uniqued MDNode.
///
/// Uniqued nodes may not need to be recreated (they may map to themselves).
static Metadata *mapUniquedNode(const MDNode *Node,
                                SmallVectorImpl<MDNode *> &Cycles,
                                ValueToValueMapTy &VM, RemapFlags Flags,
                                ValueMapTypeRemapper *TypeMapper,
                                ValueMaterializer *Materializer) {
  assert(Node->isUniqued() && "Expected uniqued node");

  // Create a temporary node upfront in case we have a metadata cycle.
  auto ClonedMD = Node->clone();
  if (!remap(Node, ClonedMD.get(), Cycles, VM, Flags, TypeMapper, Materializer))
    // No operands changed, so use the identity mapping.
    return mapToSelf(VM, Node);

  // At least one operand has changed, so uniquify the cloned node.
  return mapToMetadata(VM, Node,
                       MDNode::replaceWithUniqued(std::move(ClonedMD)));
}

static Metadata *MapMetadataImpl(const Metadata *MD,
                                 SmallVectorImpl<MDNode *> &Cycles,
                                 ValueToValueMapTy &VM, RemapFlags Flags,
                                 ValueMapTypeRemapper *TypeMapper,
                                 ValueMaterializer *Materializer) {
  // If the value already exists in the map, use it.
  if (Metadata *NewMD = VM.MD().lookup(MD).get())
    return NewMD;

  if (isa<MDString>(MD))
    return mapToSelf(VM, MD);

  if (isa<ConstantAsMetadata>(MD))
    if ((Flags & RF_NoModuleLevelChanges))
      return mapToSelf(VM, MD);

  if (const auto *VMD = dyn_cast<ValueAsMetadata>(MD)) {
    Value *MappedV =
        MapValue(VMD->getValue(), VM, Flags, TypeMapper, Materializer);
    if (VMD->getValue() == MappedV ||
        (!MappedV && (Flags & RF_IgnoreMissingEntries)))
      return mapToSelf(VM, MD);

    // FIXME: This assert crashes during bootstrap, but I think it should be
    // correct.  For now, just match behaviour from before the metadata/value
    // split.
    //
    //    assert(MappedV && "Referenced metadata not in value map!");
    if (MappedV)
      return mapToMetadata(VM, MD, ValueAsMetadata::get(MappedV));
    return nullptr;
  }

  // Note: this cast precedes the Flags check so we always get its associated
  // assertion.
  const MDNode *Node = cast<MDNode>(MD);

  // If this is a module-level metadata and we know that nothing at the
  // module level is changing, then use an identity mapping.
  if (Flags & RF_NoModuleLevelChanges)
    return mapToSelf(VM, MD);

  // Require resolved nodes whenever metadata might be remapped.
  assert(Node->isResolved() && "Unexpected unresolved node");

  if (Node->isDistinct())
    return mapDistinctNode(Node, Cycles, VM, Flags, TypeMapper, Materializer);

  return mapUniquedNode(Node, Cycles, VM, Flags, TypeMapper, Materializer);
}

Metadata *llvm::MapMetadata(const Metadata *MD, ValueToValueMapTy &VM,
                            RemapFlags Flags, ValueMapTypeRemapper *TypeMapper,
                            ValueMaterializer *Materializer) {
  SmallVector<MDNode *, 8> Cycles;
  Metadata *NewMD =
      MapMetadataImpl(MD, Cycles, VM, Flags, TypeMapper, Materializer);

  // Resolve cycles underneath MD.
  if (NewMD && NewMD != MD) {
    if (auto *N = dyn_cast<MDNode>(NewMD))
      if (!N->isResolved())
        N->resolveCycles();

    for (MDNode *N : Cycles)
      if (!N->isResolved())
        N->resolveCycles();
  } else {
    // Shouldn't get unresolved cycles if nothing was remapped.
    assert(Cycles.empty() && "Expected no unresolved cycles");
  }

  return NewMD;
}

MDNode *llvm::MapMetadata(const MDNode *MD, ValueToValueMapTy &VM,
                          RemapFlags Flags, ValueMapTypeRemapper *TypeMapper,
                          ValueMaterializer *Materializer) {
  return cast<MDNode>(MapMetadata(static_cast<const Metadata *>(MD), VM, Flags,
                                  TypeMapper, Materializer));
}

/// RemapInstruction - Convert the instruction operands from referencing the
/// current values into those specified by VMap.
///
void llvm::RemapInstruction(Instruction *I, ValueToValueMapTy &VMap,
                            RemapFlags Flags, ValueMapTypeRemapper *TypeMapper,
                            ValueMaterializer *Materializer){
  // Remap operands.
  for (User::op_iterator op = I->op_begin(), E = I->op_end(); op != E; ++op) {
    Value *V = MapValue(*op, VMap, Flags, TypeMapper, Materializer);
    // If we aren't ignoring missing entries, assert that something happened.
    if (V)
      *op = V;
    else
      assert((Flags & RF_IgnoreMissingEntries) &&
             "Referenced value not in value map!");
  }

  // Remap phi nodes' incoming blocks.
  if (PHINode *PN = dyn_cast<PHINode>(I)) {
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
      Value *V = MapValue(PN->getIncomingBlock(i), VMap, Flags);
      // If we aren't ignoring missing entries, assert that something happened.
      if (V)
        PN->setIncomingBlock(i, cast<BasicBlock>(V));
      else
        assert((Flags & RF_IgnoreMissingEntries) &&
               "Referenced block not in value map!");
    }
  }

  // Remap attached metadata.
  SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
  I->getAllMetadata(MDs);
  for (SmallVectorImpl<std::pair<unsigned, MDNode *>>::iterator
           MI = MDs.begin(),
           ME = MDs.end();
       MI != ME; ++MI) {
    MDNode *Old = MI->second;
    MDNode *New = MapMetadata(Old, VMap, Flags, TypeMapper, Materializer);
    if (New != Old)
      I->setMetadata(MI->first, New);
  }
  
  if (!TypeMapper)
    return;

  // If the instruction's type is being remapped, do so now.
  if (auto CS = CallSite(I)) {
    SmallVector<Type *, 3> Tys;
    FunctionType *FTy = CS.getFunctionType();
    Tys.reserve(FTy->getNumParams());
    for (Type *Ty : FTy->params())
      Tys.push_back(TypeMapper->remapType(Ty));
    CS.mutateFunctionType(FunctionType::get(
        TypeMapper->remapType(I->getType()), Tys, FTy->isVarArg()));
    return;
  }
  if (auto *AI = dyn_cast<AllocaInst>(I))
    AI->setAllocatedType(TypeMapper->remapType(AI->getAllocatedType()));
  if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
    GEP->setSourceElementType(
        TypeMapper->remapType(GEP->getSourceElementType()));
    GEP->setResultElementType(
        TypeMapper->remapType(GEP->getResultElementType()));
  }
  I->mutateType(TypeMapper->remapType(I->getType()));
}
