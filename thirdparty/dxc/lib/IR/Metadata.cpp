//===- Metadata.cpp - Implement Metadata classes --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Metadata classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Metadata.h"
#include "LLVMContextImpl.h"
#include "MetadataImpl.h"
#include "SymbolTableListTraitsImpl.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"

using namespace llvm;

MetadataAsValue::MetadataAsValue(Type *Ty, Metadata *MD)
    : Value(Ty, MetadataAsValueVal), MD(MD) {
  track();
}

MetadataAsValue::~MetadataAsValue() {
  getType()->getContext().pImpl->MetadataAsValues.erase(MD);
  untrack();
}

/// \brief Canonicalize metadata arguments to intrinsics.
///
/// To support bitcode upgrades (and assembly semantic sugar) for \a
/// MetadataAsValue, we need to canonicalize certain metadata.
///
///   - nullptr is replaced by an empty MDNode.
///   - An MDNode with a single null operand is replaced by an empty MDNode.
///   - An MDNode whose only operand is a \a ConstantAsMetadata gets skipped.
///
/// This maintains readability of bitcode from when metadata was a type of
/// value, and these bridges were unnecessary.
static Metadata *canonicalizeMetadataForValue(LLVMContext &Context,
                                              Metadata *MD) {
  if (!MD)
    // !{}
    return MDNode::get(Context, None);

  // Return early if this isn't a single-operand MDNode.
  auto *N = dyn_cast<MDNode>(MD);
  if (!N || N->getNumOperands() != 1)
    return MD;

  if (!N->getOperand(0))
    // !{}
    return MDNode::get(Context, None);

  if (auto *C = dyn_cast<ConstantAsMetadata>(N->getOperand(0)))
    // Look through the MDNode.
    return C;

  return MD;
}

MetadataAsValue *MetadataAsValue::get(LLVMContext &Context, Metadata *MD) {
  MD = canonicalizeMetadataForValue(Context, MD);
  auto *&Entry = Context.pImpl->MetadataAsValues[MD];
  if (!Entry)
    Entry = new MetadataAsValue(Type::getMetadataTy(Context), MD);
  return Entry;
}

MetadataAsValue *MetadataAsValue::getIfExists(LLVMContext &Context,
                                              Metadata *MD) {
  MD = canonicalizeMetadataForValue(Context, MD);
  auto &Store = Context.pImpl->MetadataAsValues;
  return Store.lookup(MD);
}

void MetadataAsValue::handleChangedMetadata(Metadata *MD) {
  LLVMContext &Context = getContext();
  MD = canonicalizeMetadataForValue(Context, MD);
  auto &Store = Context.pImpl->MetadataAsValues;

  // Stop tracking the old metadata.
  Store.erase(this->MD);
  untrack();
  this->MD = nullptr;

  // Start tracking MD, or RAUW if necessary.
  auto *&Entry = Store[MD];
  if (Entry) {
    replaceAllUsesWith(Entry);
    delete this;
    return;
  }

  this->MD = MD;
  track();
  Entry = this;
}

void MetadataAsValue::track() {
  if (MD)
    MetadataTracking::track(&MD, *MD, *this);
}

void MetadataAsValue::untrack() {
  if (MD)
    MetadataTracking::untrack(MD);
}

void ReplaceableMetadataImpl::addRef(void *Ref, OwnerTy Owner) {
  bool WasInserted =
      UseMap.insert(std::make_pair(Ref, std::make_pair(Owner, NextIndex)))
          .second;
  (void)WasInserted;
  assert(WasInserted && "Expected to add a reference");

  ++NextIndex;
  assert(NextIndex != 0 && "Unexpected overflow");
}

void ReplaceableMetadataImpl::dropRef(void *Ref) {
  bool WasErased = UseMap.erase(Ref);
  (void)WasErased;
  // assert(WasErased && "Expected to drop a reference"); // HLSL Change - not while cleaning up OOM
}

void ReplaceableMetadataImpl::moveRef(void *Ref, void *New,
                                      const Metadata &MD) {
  auto I = UseMap.find(Ref);
  assert(I != UseMap.end() && "Expected to move a reference");
  auto OwnerAndIndex = I->second;
  UseMap.erase(I);
  bool WasInserted = UseMap.insert(std::make_pair(New, OwnerAndIndex)).second;
  (void)WasInserted;
  assert(WasInserted && "Expected to add a reference");

  // Check that the references are direct if there's no owner.
  (void)MD;
  assert((OwnerAndIndex.first || *static_cast<Metadata **>(Ref) == &MD) &&
         "Reference without owner must be direct");
  assert((OwnerAndIndex.first || *static_cast<Metadata **>(New) == &MD) &&
         "Reference without owner must be direct");
}

void ReplaceableMetadataImpl::replaceAllUsesWith(Metadata *MD) {
  assert(!(MD && isa<MDNode>(MD) && cast<MDNode>(MD)->isTemporary()) &&
         "Expected non-temp node");

  if (UseMap.empty())
    return;

  // Copy out uses since UseMap will get touched below.
  typedef std::pair<void *, std::pair<OwnerTy, uint64_t>> UseTy;
  SmallVector<UseTy, 8> Uses(UseMap.begin(), UseMap.end());
  std::sort(Uses.begin(), Uses.end(), [](const UseTy &L, const UseTy &R) {
    return L.second.second < R.second.second;
  });
  for (const auto &Pair : Uses) {
    // Check that this Ref hasn't disappeared after RAUW (when updating a
    // previous Ref).
    if (!UseMap.count(Pair.first))
      continue;

    OwnerTy Owner = Pair.second.first;
    if (!Owner) {
      // Update unowned tracking references directly.
      Metadata *&Ref = *static_cast<Metadata **>(Pair.first);
      Ref = MD;
      if (MD)
        MetadataTracking::track(Ref);
      UseMap.erase(Pair.first);
      continue;
    }

    // Check for MetadataAsValue.
    if (Owner.is<MetadataAsValue *>()) {
      Owner.get<MetadataAsValue *>()->handleChangedMetadata(MD);
      continue;
    }

    // There's a Metadata owner -- dispatch.
    Metadata *OwnerMD = Owner.get<Metadata *>();
    switch (OwnerMD->getMetadataID()) {
#define HANDLE_METADATA_LEAF(CLASS)                                            \
  case Metadata::CLASS##Kind:                                                  \
    cast<CLASS>(OwnerMD)->handleChangedOperand(Pair.first, MD);                \
    continue;
#include "llvm/IR/Metadata.def"
    default:
      llvm_unreachable("Invalid metadata subclass");
    }
  }
  assert(UseMap.empty() && "Expected all uses to be replaced");
}

void ReplaceableMetadataImpl::resolveAllUses(bool ResolveUsers) {
  if (UseMap.empty())
    return;

  if (!ResolveUsers) {
    UseMap.clear();
    return;
  }

  // Copy out uses since UseMap could get touched below.
  typedef std::pair<void *, std::pair<OwnerTy, uint64_t>> UseTy;
  SmallVector<UseTy, 8> Uses(UseMap.begin(), UseMap.end());
  std::sort(Uses.begin(), Uses.end(), [](const UseTy &L, const UseTy &R) {
    return L.second.second < R.second.second;
  });
  UseMap.clear();
  for (const auto &Pair : Uses) {
    auto Owner = Pair.second.first;
    if (!Owner)
      continue;
    if (Owner.is<MetadataAsValue *>())
      continue;

    // Resolve MDNodes that point at this.
    auto *OwnerMD = dyn_cast<MDNode>(Owner.get<Metadata *>());
    if (!OwnerMD)
      continue;
    if (OwnerMD->isResolved())
      continue;
    OwnerMD->decrementUnresolvedOperandCount();
  }
}

static Function *getLocalFunction(Value *V) {
  assert(V && "Expected value");
  if (auto *A = dyn_cast<Argument>(V))
    return A->getParent();
  if (BasicBlock *BB = cast<Instruction>(V)->getParent())
    return BB->getParent();
  return nullptr;
}

ValueAsMetadata *ValueAsMetadata::get(Value *V) {
  assert(V && "Unexpected null Value");

  auto &Context = V->getContext();
  auto *&Entry = Context.pImpl->ValuesAsMetadata[V];
  if (!Entry) {
    assert((isa<Constant>(V) || isa<Argument>(V) || isa<Instruction>(V)) &&
           "Expected constant or function-local value");
    assert(!V->IsUsedByMD &&
           "Expected this to be the only metadata use");
    V->IsUsedByMD = true;
    if (auto *C = dyn_cast<Constant>(V))
      Entry = new ConstantAsMetadata(C);
    else
      Entry = new LocalAsMetadata(V);
  }

  return Entry;
}

ValueAsMetadata *ValueAsMetadata::getIfExists(Value *V) {
  assert(V && "Unexpected null Value");
  return V->getContext().pImpl->ValuesAsMetadata.lookup(V);
}

void ValueAsMetadata::handleDeletion(Value *V) {
  assert(V && "Expected valid value");

  auto &Store = V->getType()->getContext().pImpl->ValuesAsMetadata;
  auto I = Store.find(V);
  if (I == Store.end())
    return;

  // Remove old entry from the map.
  ValueAsMetadata *MD = I->second;
  // assert(MD && "Expected valid metadata"); // HLSL Change - MD might be nullptr under OOM
  // assert(MD->getValue() == V && "Expected valid mapping"); // HLSL Change - MD might be nullptr under OOM
  Store.erase(I);

  // Delete the metadata.
  if (MD) MD->replaceAllUsesWith(nullptr); // HLSL Change - MD might be nullptr under OOM
  delete MD;
}

void ValueAsMetadata::handleRAUW(Value *From, Value *To) {
  assert(From && "Expected valid value");
  assert(To && "Expected valid value");
  assert(From != To && "Expected changed value");
  assert(From->getType() == To->getType() && "Unexpected type change");

  LLVMContext &Context = From->getType()->getContext();
  auto &Store = Context.pImpl->ValuesAsMetadata;
  auto I = Store.find(From);
  if (I == Store.end()) {
    assert(!From->IsUsedByMD &&
           "Expected From not to be used by metadata");
    return;
  }

  // Remove old entry from the map.
  assert(From->IsUsedByMD &&
         "Expected From to be used by metadata");
  From->IsUsedByMD = false;
  ValueAsMetadata *MD = I->second;
  assert(MD && "Expected valid metadata");
  assert(MD->getValue() == From && "Expected valid mapping");
  Store.erase(I);

  if (isa<LocalAsMetadata>(MD)) {
    if (auto *C = dyn_cast<Constant>(To)) {
      // Local became a constant.
      MD->replaceAllUsesWith(ConstantAsMetadata::get(C));
      delete MD;
      return;
    }
    if (getLocalFunction(From) && getLocalFunction(To) &&
        getLocalFunction(From) != getLocalFunction(To)) {
      // Function changed.
      MD->replaceAllUsesWith(nullptr);
      delete MD;
      return;
    }
  } else if (!isa<Constant>(To)) {
    // Changed to function-local value.
    MD->replaceAllUsesWith(nullptr);
    delete MD;
    return;
  }

  auto *&Entry = Store[To];
  if (Entry) {
    // The target already exists.
    MD->replaceAllUsesWith(Entry);
    delete MD;
    return;
  }

  // Update MD in place (and update the map entry).
  assert(!To->IsUsedByMD &&
         "Expected this to be the only metadata use");
  To->IsUsedByMD = true;
  MD->V = To;
  Entry = MD;
}

//===----------------------------------------------------------------------===//
// MDString implementation.
//

MDString *MDString::get(LLVMContext &Context, StringRef Str) {
  auto &Store = Context.pImpl->MDStringCache;
  auto I = Store.find(Str);
  if (I != Store.end())
    return &I->second;

  auto *Entry =
      StringMapEntry<MDString>::Create(Str, Store.getAllocator(), MDString());
  // HLSL Change Begin: Don't leak on insertion failure
  try {
  bool WasInserted = Store.insert(Entry);
  (void)WasInserted;
  assert(WasInserted && "Expected entry to be inserted");
  }
  catch (...) {
    Entry->Destroy();
    throw;
  }
  // HLSL Change End
  Entry->second.Entry = Entry;
  return &Entry->second;
}

StringRef MDString::getString() const {
  assert(Entry && "Expected to find string map entry");
  return Entry->first();
}

//===----------------------------------------------------------------------===//
// MDNode implementation.
//

// Assert that the MDNode types will not be unaligned by the objects
// prepended to them.
#define HANDLE_MDNODE_LEAF(CLASS)                                              \
  static_assert(                                                               \
      llvm::AlignOf<uint64_t>::Alignment >= llvm::AlignOf<CLASS>::Alignment,   \
      "Alignment is insufficient after objects prepended to " #CLASS);
#include "llvm/IR/Metadata.def"

void *MDNode::operator new(size_t Size, unsigned NumOps) {
  size_t OpSize = NumOps * sizeof(MDOperand);
  // uint64_t is the most aligned type we need support (ensured by static_assert
  // above)
  OpSize = RoundUpToAlignment(OpSize, llvm::alignOf<uint64_t>());
  void *Ptr = reinterpret_cast<char *>(::operator new(OpSize + Size)) + OpSize;
  MDOperand *O = static_cast<MDOperand *>(Ptr);
  for (MDOperand *E = O - NumOps; O != E; --O)
    (void)new (O - 1) MDOperand;
  return Ptr;
}

void MDNode::operator delete(void *Mem) {
  MDNode *N = static_cast<MDNode *>(Mem);
  size_t OpSize = N->NumOperands * sizeof(MDOperand);
  OpSize = RoundUpToAlignment(OpSize, llvm::alignOf<uint64_t>());

  MDOperand *O = static_cast<MDOperand *>(Mem);
  for (MDOperand *E = O - N->NumOperands; O != E; --O)
    (O - 1)->~MDOperand();
  ::operator delete(reinterpret_cast<char *>(Mem) - OpSize);
}

MDNode::MDNode(LLVMContext &Context, unsigned ID, StorageType Storage,
               ArrayRef<Metadata *> Ops1, ArrayRef<Metadata *> Ops2)
    : Metadata(ID, Storage), NumOperands(Ops1.size() + Ops2.size()),
      NumUnresolved(0), Context(Context) {
  unsigned Op = 0;
  for (Metadata *MD : Ops1)
    setOperand(Op++, MD);
  for (Metadata *MD : Ops2)
    setOperand(Op++, MD);

  if (isDistinct())
    return;

  if (isUniqued())
    // Check whether any operands are unresolved, requiring re-uniquing.  If
    // not, don't support RAUW.
    if (!countUnresolvedOperands())
      return;

  this->Context.makeReplaceable(make_unique<ReplaceableMetadataImpl>(Context));
}

TempMDNode MDNode::clone() const {
  switch (getMetadataID()) {
  default:
    llvm_unreachable("Invalid MDNode subclass");
#define HANDLE_MDNODE_LEAF(CLASS)                                              \
  case CLASS##Kind:                                                            \
    return cast<CLASS>(this)->cloneImpl();
#include "llvm/IR/Metadata.def"
  }
}

static bool isOperandUnresolved(Metadata *Op) {
  if (auto *N = dyn_cast_or_null<MDNode>(Op))
    return !N->isResolved();
  return false;
}

unsigned MDNode::countUnresolvedOperands() {
  assert(NumUnresolved == 0 && "Expected unresolved ops to be uncounted");
  NumUnresolved = std::count_if(op_begin(), op_end(), isOperandUnresolved);
  return NumUnresolved;
}

void MDNode::makeUniqued() {
  assert(isTemporary() && "Expected this to be temporary");
  assert(!isResolved() && "Expected this to be unresolved");

  // Enable uniquing callbacks.
  for (auto &Op : mutable_operands())
    Op.reset(Op.get(), this);

  // Make this 'uniqued'.
  Storage = Uniqued;
  if (!countUnresolvedOperands())
    resolve();

  assert(isUniqued() && "Expected this to be uniqued");
}

void MDNode::makeDistinct() {
  assert(isTemporary() && "Expected this to be temporary");
  assert(!isResolved() && "Expected this to be unresolved");

  // Pretend to be uniqued, resolve the node, and then store in distinct table.
  Storage = Uniqued;
  resolve();
  storeDistinctInContext();

  assert(isDistinct() && "Expected this to be distinct");
  assert(isResolved() && "Expected this to be resolved");
}

void MDNode::resolve() {
  assert(isUniqued() && "Expected this to be uniqued");
  assert(!isResolved() && "Expected this to be unresolved");

  // Move the map, so that this immediately looks resolved.
  auto Uses = Context.takeReplaceableUses();
  NumUnresolved = 0;
  assert(isResolved() && "Expected this to be resolved");

  // Drop RAUW support.
  Uses->resolveAllUses();
}

void MDNode::resolveAfterOperandChange(Metadata *Old, Metadata *New) {
  assert(NumUnresolved != 0 && "Expected unresolved operands");

  // Check if an operand was resolved.
  if (!isOperandUnresolved(Old)) {
    if (isOperandUnresolved(New))
      // An operand was un-resolved!
      ++NumUnresolved;
  } else if (!isOperandUnresolved(New))
    decrementUnresolvedOperandCount();
}

void MDNode::decrementUnresolvedOperandCount() {
  if (!--NumUnresolved)
    // Last unresolved operand has just been resolved.
    resolve();
}

void MDNode::resolveCycles() {
  if (isResolved())
    return;

  // Resolve this node immediately.
  resolve();

  // Resolve all operands.
  for (const auto &Op : operands()) {
    auto *N = dyn_cast_or_null<MDNode>(Op);
    if (!N)
      continue;

    assert(!N->isTemporary() &&
           "Expected all forward declarations to be resolved");
    if (!N->isResolved())
      N->resolveCycles();
  }
}

static bool hasSelfReference(MDNode *N) {
  for (Metadata *MD : N->operands())
    if (MD == N)
      return true;
  return false;
}

MDNode *MDNode::replaceWithPermanentImpl() {
  if (hasSelfReference(this))
    return replaceWithDistinctImpl();
  return replaceWithUniquedImpl();
}

MDNode *MDNode::replaceWithUniquedImpl() {
  // Try to uniquify in place.
  MDNode *UniquedNode = uniquify();

  if (UniquedNode == this) {
    makeUniqued();
    return this;
  }

  // Collision, so RAUW instead.
  replaceAllUsesWith(UniquedNode);
  deleteAsSubclass();
  return UniquedNode;
}

MDNode *MDNode::replaceWithDistinctImpl() {
  makeDistinct();
  return this;
}

void MDTuple::recalculateHash() {
  setHash(MDTupleInfo::KeyTy::calculateHash(this));
}

void MDNode::dropAllReferences() {
  for (unsigned I = 0, E = NumOperands; I != E; ++I)
    setOperand(I, nullptr);
  if (!isResolved()) {
    Context.getReplaceableUses()->resolveAllUses(/* ResolveUsers */ false);
    (void)Context.takeReplaceableUses();
  }
}

void MDNode::handleChangedOperand(void *Ref, Metadata *New) {
  unsigned Op = static_cast<MDOperand *>(Ref) - op_begin();
  assert(Op < getNumOperands() && "Expected valid operand");

  if (!isUniqued()) {
    // This node is not uniqued.  Just set the operand and be done with it.
    setOperand(Op, New);
    return;
  }

  // This node is uniqued.
  eraseFromStore();

  Metadata *Old = getOperand(Op);
  setOperand(Op, New);

  // Drop uniquing for self-reference cycles.
  if (New == this) {
    if (!isResolved())
      resolve();
    storeDistinctInContext();
    return;
  }

  // Re-unique the node.
  auto *Uniqued = uniquify();
  if (Uniqued == this) {
    if (!isResolved())
      resolveAfterOperandChange(Old, New);
    return;
  }

  // Collision.
  if (!isResolved()) {
    // Still unresolved, so RAUW.
    //
    // First, clear out all operands to prevent any recursion (similar to
    // dropAllReferences(), but we still need the use-list).
    for (unsigned O = 0, E = getNumOperands(); O != E; ++O)
      setOperand(O, nullptr);
    Context.getReplaceableUses()->replaceAllUsesWith(Uniqued);
    deleteAsSubclass();
    return;
  }

  // Store in non-uniqued form if RAUW isn't possible.
  storeDistinctInContext();
}

void MDNode::deleteAsSubclass() {
  switch (getMetadataID()) {
  default:
    llvm_unreachable("Invalid subclass of MDNode");
#define HANDLE_MDNODE_LEAF(CLASS)                                              \
  case CLASS##Kind:                                                            \
    delete cast<CLASS>(this);                                                  \
    break;
#include "llvm/IR/Metadata.def"
  }
}

template <class T, class InfoT>
static T *uniquifyImpl(T *N, DenseSet<T *, InfoT> &Store) {
  if (T *U = getUniqued(Store, N))
    return U;

  Store.insert(N);
  return N;
}

template <class NodeTy> struct MDNode::HasCachedHash {
  typedef char Yes[1];
  typedef char No[2];
  template <class U, U Val> struct SFINAE {};

  template <class U>
  static Yes &check(SFINAE<void (U::*)(unsigned), &U::setHash> *);
  template <class U> static No &check(...);

  static const bool value = sizeof(check<NodeTy>(nullptr)) == sizeof(Yes);
};

MDNode *MDNode::uniquify() {
  assert(!hasSelfReference(this) && "Cannot uniquify a self-referencing node");

  // Try to insert into uniquing store.
  switch (getMetadataID()) {
  default:
    llvm_unreachable("Invalid subclass of MDNode");
#define HANDLE_MDNODE_LEAF(CLASS)                                              \
  case CLASS##Kind: {                                                          \
    CLASS *SubclassThis = cast<CLASS>(this);                                   \
    std::integral_constant<bool, HasCachedHash<CLASS>::value>                  \
        ShouldRecalculateHash;                                                 \
    dispatchRecalculateHash(SubclassThis, ShouldRecalculateHash);              \
    return uniquifyImpl(SubclassThis, getContext().pImpl->CLASS##s);           \
  }
#include "llvm/IR/Metadata.def"
  }
}

void MDNode::eraseFromStore() {
  switch (getMetadataID()) {
  default:
    llvm_unreachable("Invalid subclass of MDNode");
#define HANDLE_MDNODE_LEAF(CLASS)                                              \
  case CLASS##Kind:                                                            \
    getContext().pImpl->CLASS##s.erase(cast<CLASS>(this));                     \
    break;
#include "llvm/IR/Metadata.def"
  }
}

MDTuple *MDTuple::getImpl(LLVMContext &Context, ArrayRef<Metadata *> MDs,
                          StorageType Storage, bool ShouldCreate) {
  unsigned Hash = 0;
  if (Storage == Uniqued) {
    MDTupleInfo::KeyTy Key(MDs);
    if (auto *N = getUniqued(Context.pImpl->MDTuples, Key))
      return N;
    if (!ShouldCreate)
      return nullptr;
    Hash = Key.getHash();
  } else {
    assert(ShouldCreate && "Expected non-uniqued nodes to always be created");
  }

  // HLSL Change - guard with try/catch
  MDTuple *MDTuplePtr(new (MDs.size()) MDTuple(Context, Storage, Hash, MDs));
  MDTuple *Result;
  try {
    Result = storeImpl(MDTuplePtr, Storage, Context.pImpl->MDTuples);
  } catch (...) {
    MDTuplePtr->deleteAsSubclass();
    throw;
  }
  return Result;
}

void MDNode::deleteTemporary(MDNode *N) {
  assert(N->isTemporary() && "Expected temporary node");
  N->replaceAllUsesWith(nullptr);
  N->deleteAsSubclass();
}

void MDNode::storeDistinctInContext() {
  assert(isResolved() && "Expected resolved nodes");
  Storage = Distinct;

  // Reset the hash.
  switch (getMetadataID()) {
  default:
    llvm_unreachable("Invalid subclass of MDNode");
#define HANDLE_MDNODE_LEAF(CLASS)                                              \
  case CLASS##Kind: {                                                          \
    std::integral_constant<bool, HasCachedHash<CLASS>::value> ShouldResetHash; \
    dispatchResetHash(cast<CLASS>(this), ShouldResetHash);                     \
    break;                                                                     \
  }
#include "llvm/IR/Metadata.def"
  }

  getContext().pImpl->DistinctMDNodes.insert(this);
}

void MDNode::replaceOperandWith(unsigned I, Metadata *New) {
  if (getOperand(I) == New)
    return;

  if (!isUniqued()) {
    setOperand(I, New);
    return;
  }

  handleChangedOperand(mutable_begin() + I, New);
}

void MDNode::setOperand(unsigned I, Metadata *New) {
  assert(I < NumOperands);
  mutable_begin()[I].reset(New, isUniqued() ? this : nullptr);
}

/// \brief Get a node, or a self-reference that looks like it.
///
/// Special handling for finding self-references, for use by \a
/// MDNode::concatenate() and \a MDNode::intersect() to maintain behaviour from
/// when self-referencing nodes were still uniqued.  If the first operand has
/// the same operands as \c Ops, return the first operand instead.
static MDNode *getOrSelfReference(LLVMContext &Context,
                                  ArrayRef<Metadata *> Ops) {
  if (!Ops.empty())
    if (MDNode *N = dyn_cast_or_null<MDNode>(Ops[0]))
      if (N->getNumOperands() == Ops.size() && N == N->getOperand(0)) {
        for (unsigned I = 1, E = Ops.size(); I != E; ++I)
          if (Ops[I] != N->getOperand(I))
            return MDNode::get(Context, Ops);
        return N;
      }

  return MDNode::get(Context, Ops);
}

MDNode *MDNode::concatenate(MDNode *A, MDNode *B) {
  if (!A)
    return B;
  if (!B)
    return A;

  SmallVector<Metadata *, 4> MDs;
  MDs.reserve(A->getNumOperands() + B->getNumOperands());
  MDs.append(A->op_begin(), A->op_end());
  MDs.append(B->op_begin(), B->op_end());

  // FIXME: This preserves long-standing behaviour, but is it really the right
  // behaviour?  Or was that an unintended side-effect of node uniquing?
  return getOrSelfReference(A->getContext(), MDs);
}

MDNode *MDNode::intersect(MDNode *A, MDNode *B) {
  if (!A || !B)
    return nullptr;

  SmallVector<Metadata *, 4> MDs;
  for (Metadata *MD : A->operands())
    if (std::find(B->op_begin(), B->op_end(), MD) != B->op_end())
      MDs.push_back(MD);

  // FIXME: This preserves long-standing behaviour, but is it really the right
  // behaviour?  Or was that an unintended side-effect of node uniquing?
  return getOrSelfReference(A->getContext(), MDs);
}

MDNode *MDNode::getMostGenericAliasScope(MDNode *A, MDNode *B) {
  if (!A || !B)
    return nullptr;

  SmallVector<Metadata *, 4> MDs(B->op_begin(), B->op_end());
  for (Metadata *MD : A->operands())
    if (std::find(B->op_begin(), B->op_end(), MD) == B->op_end())
      MDs.push_back(MD);

  // FIXME: This preserves long-standing behaviour, but is it really the right
  // behaviour?  Or was that an unintended side-effect of node uniquing?
  return getOrSelfReference(A->getContext(), MDs);
}

MDNode *MDNode::getMostGenericFPMath(MDNode *A, MDNode *B) {
  if (!A || !B)
    return nullptr;

  APFloat AVal = mdconst::extract<ConstantFP>(A->getOperand(0))->getValueAPF();
  APFloat BVal = mdconst::extract<ConstantFP>(B->getOperand(0))->getValueAPF();
  if (AVal.compare(BVal) == APFloat::cmpLessThan)
    return A;
  return B;
}

static bool isContiguous(const ConstantRange &A, const ConstantRange &B) {
  return A.getUpper() == B.getLower() || A.getLower() == B.getUpper();
}

static bool canBeMerged(const ConstantRange &A, const ConstantRange &B) {
  return !A.intersectWith(B).isEmptySet() || isContiguous(A, B);
}

static bool tryMergeRange(SmallVectorImpl<ConstantInt *> &EndPoints,
                          ConstantInt *Low, ConstantInt *High) {
  ConstantRange NewRange(Low->getValue(), High->getValue());
  unsigned Size = EndPoints.size();
  APInt LB = EndPoints[Size - 2]->getValue();
  APInt LE = EndPoints[Size - 1]->getValue();
  ConstantRange LastRange(LB, LE);
  if (canBeMerged(NewRange, LastRange)) {
    ConstantRange Union = LastRange.unionWith(NewRange);
    Type *Ty = High->getType();
    EndPoints[Size - 2] =
        cast<ConstantInt>(ConstantInt::get(Ty, Union.getLower()));
    EndPoints[Size - 1] =
        cast<ConstantInt>(ConstantInt::get(Ty, Union.getUpper()));
    return true;
  }
  return false;
}

static void addRange(SmallVectorImpl<ConstantInt *> &EndPoints,
                     ConstantInt *Low, ConstantInt *High) {
  if (!EndPoints.empty())
    if (tryMergeRange(EndPoints, Low, High))
      return;

  EndPoints.push_back(Low);
  EndPoints.push_back(High);
}

MDNode *MDNode::getMostGenericRange(MDNode *A, MDNode *B) {
  // Given two ranges, we want to compute the union of the ranges. This
  // is slightly complitade by having to combine the intervals and merge
  // the ones that overlap.

  if (!A || !B)
    return nullptr;

  if (A == B)
    return A;

  // First, walk both lists in older of the lower boundary of each interval.
  // At each step, try to merge the new interval to the last one we adedd.
  SmallVector<ConstantInt *, 4> EndPoints;
  int AI = 0;
  int BI = 0;
  int AN = A->getNumOperands() / 2;
  int BN = B->getNumOperands() / 2;
  while (AI < AN && BI < BN) {
    ConstantInt *ALow = mdconst::extract<ConstantInt>(A->getOperand(2 * AI));
    ConstantInt *BLow = mdconst::extract<ConstantInt>(B->getOperand(2 * BI));

    if (ALow->getValue().slt(BLow->getValue())) {
      addRange(EndPoints, ALow,
               mdconst::extract<ConstantInt>(A->getOperand(2 * AI + 1)));
      ++AI;
    } else {
      addRange(EndPoints, BLow,
               mdconst::extract<ConstantInt>(B->getOperand(2 * BI + 1)));
      ++BI;
    }
  }
  while (AI < AN) {
    addRange(EndPoints, mdconst::extract<ConstantInt>(A->getOperand(2 * AI)),
             mdconst::extract<ConstantInt>(A->getOperand(2 * AI + 1)));
    ++AI;
  }
  while (BI < BN) {
    addRange(EndPoints, mdconst::extract<ConstantInt>(B->getOperand(2 * BI)),
             mdconst::extract<ConstantInt>(B->getOperand(2 * BI + 1)));
    ++BI;
  }

  // If we have more than 2 ranges (4 endpoints) we have to try to merge
  // the last and first ones.
  unsigned Size = EndPoints.size();
  if (Size > 4) {
    ConstantInt *FB = EndPoints[0];
    ConstantInt *FE = EndPoints[1];
    if (tryMergeRange(EndPoints, FB, FE)) {
      for (unsigned i = 0; i < Size - 2; ++i) {
        EndPoints[i] = EndPoints[i + 2];
      }
      EndPoints.resize(Size - 2);
    }
  }

  // If in the end we have a single range, it is possible that it is now the
  // full range. Just drop the metadata in that case.
  if (EndPoints.size() == 2) {
    ConstantRange Range(EndPoints[0]->getValue(), EndPoints[1]->getValue());
    if (Range.isFullSet())
      return nullptr;
  }

  SmallVector<Metadata *, 4> MDs;
  MDs.reserve(EndPoints.size());
  for (auto *I : EndPoints)
    MDs.push_back(ConstantAsMetadata::get(I));
  return MDNode::get(A->getContext(), MDs);
}

//===----------------------------------------------------------------------===//
// NamedMDNode implementation.
//

static SmallVector<TrackingMDRef, 4> &getNMDOps(void *Operands) {
  return *(SmallVector<TrackingMDRef, 4> *)Operands;
}

NamedMDNode::NamedMDNode(const Twine &N)
    : Name(N.str()), Parent(nullptr),
      Operands(new SmallVector<TrackingMDRef, 4>()) {}

NamedMDNode::~NamedMDNode() {
  dropAllReferences();
  delete &getNMDOps(Operands);
}

unsigned NamedMDNode::getNumOperands() const {
  return (unsigned)getNMDOps(Operands).size();
}

MDNode *NamedMDNode::getOperand(unsigned i) const {
  assert(i < getNumOperands() && "Invalid Operand number!");
  auto *N = getNMDOps(Operands)[i].get();
  return cast_or_null<MDNode>(N);
}

void NamedMDNode::addOperand(MDNode *M) { getNMDOps(Operands).emplace_back(M); }

void NamedMDNode::setOperand(unsigned I, MDNode *New) {
  assert(I < getNumOperands() && "Invalid operand number");
  getNMDOps(Operands)[I].reset(New);
}

void NamedMDNode::eraseFromParent() {
  getParent()->eraseNamedMetadata(this);
}

void NamedMDNode::dropAllReferences() {
  getNMDOps(Operands).clear();
}

StringRef NamedMDNode::getName() const {
  return StringRef(Name);
}

//===----------------------------------------------------------------------===//
// Instruction Metadata method implementations.
//
void MDAttachmentMap::set(unsigned ID, MDNode &MD) {
  for (auto &I : Attachments)
    if (I.first == ID) {
      I.second.reset(&MD);
      return;
    }
  Attachments.emplace_back(std::piecewise_construct, std::make_tuple(ID),
                           std::make_tuple(&MD));
}

void MDAttachmentMap::erase(unsigned ID) {
  if (empty())
    return;

  // Common case is one/last value.
  if (Attachments.back().first == ID) {
    Attachments.pop_back();
    return;
  }

  for (auto I = Attachments.begin(), E = std::prev(Attachments.end()); I != E;
       ++I)
    if (I->first == ID) {
      *I = std::move(Attachments.back());
      Attachments.pop_back();
      return;
    }
}

MDNode *MDAttachmentMap::lookup(unsigned ID) const {
  for (const auto &I : Attachments)
    if (I.first == ID)
      return I.second;
  return nullptr;
}

void MDAttachmentMap::getAll(
    SmallVectorImpl<std::pair<unsigned, MDNode *>> &Result) const {
  Result.append(Attachments.begin(), Attachments.end());

  // Sort the resulting array so it is stable.
  if (Result.size() > 1)
    array_pod_sort(Result.begin(), Result.end());
}

void Instruction::setMetadata(StringRef Kind, MDNode *Node) {
  if (!Node && !hasMetadata())
    return;
  setMetadata(getContext().getMDKindID(Kind), Node);
}

MDNode *Instruction::getMetadataImpl(StringRef Kind) const {
  unsigned KindID = 0;
#if 0 // HLSL Change Starts
  return getMetadataImpl(getContext().getMDKindID(Kind))
#else
  // Calling special function to check for existence of string id,
  // so it doesn't get instantiated in the context.
  if (getContext().findMDKindID(Kind, &KindID))
    return getMetadataImpl(KindID);
  return nullptr;
#endif // HLSL Change Ends
}

void Instruction::dropUnknownMetadata(ArrayRef<unsigned> KnownIDs) {
  SmallSet<unsigned, 5> KnownSet;
  KnownSet.insert(KnownIDs.begin(), KnownIDs.end());

  // Drop debug if needed
  if (KnownSet.erase(LLVMContext::MD_dbg))
    DbgLoc = DebugLoc();

  if (!hasMetadataHashEntry())
    return; // Nothing to remove!

  auto &InstructionMetadata = getContext().pImpl->InstructionMetadata;

  if (KnownSet.empty()) {
    // Just drop our entry at the store.
    InstructionMetadata.erase(this);
    setHasMetadataHashEntry(false);
    return;
  }

  auto &Info = InstructionMetadata[this];
  Info.remove_if([&KnownSet](const std::pair<unsigned, TrackingMDNodeRef> &I) {
    return !KnownSet.count(I.first);
  });

  if (Info.empty()) {
    // Drop our entry at the store.
    InstructionMetadata.erase(this);
    setHasMetadataHashEntry(false);
  }
}

/// setMetadata - Set the metadata of of the specified kind to the specified
/// node.  This updates/replaces metadata if already present, or removes it if
/// Node is null.
void Instruction::setMetadata(unsigned KindID, MDNode *Node) {
  if (!Node && !hasMetadata())
    return;

  // Handle 'dbg' as a special case since it is not stored in the hash table.
  if (KindID == LLVMContext::MD_dbg) {
    DbgLoc = DebugLoc(Node);
    return;
  }
  
  // Handle the case when we're adding/updating metadata on an instruction.
  if (Node) {
    auto &Info = getContext().pImpl->InstructionMetadata[this];
    assert(!Info.empty() == hasMetadataHashEntry() &&
           "HasMetadata bit is wonked");
    if (Info.empty())
      setHasMetadataHashEntry(true);
    Info.set(KindID, *Node);
    return;
  }

  // Otherwise, we're removing metadata from an instruction.
  assert((hasMetadataHashEntry() ==
          (getContext().pImpl->InstructionMetadata.count(this) > 0)) &&
         "HasMetadata bit out of date!");
  if (!hasMetadataHashEntry())
    return;  // Nothing to remove!
  auto &Info = getContext().pImpl->InstructionMetadata[this];

  // Handle removal of an existing value.
  Info.erase(KindID);

  if (!Info.empty())
    return;

  getContext().pImpl->InstructionMetadata.erase(this);
  setHasMetadataHashEntry(false);
}

void Instruction::setAAMetadata(const AAMDNodes &N) {
  setMetadata(LLVMContext::MD_tbaa, N.TBAA);
  setMetadata(LLVMContext::MD_alias_scope, N.Scope);
  setMetadata(LLVMContext::MD_noalias, N.NoAlias);
}

MDNode *Instruction::getMetadataImpl(unsigned KindID) const {
  // Handle 'dbg' as a special case since it is not stored in the hash table.
  if (KindID == LLVMContext::MD_dbg)
    return DbgLoc.getAsMDNode();

  if (!hasMetadataHashEntry())
    return nullptr;
  auto &Info = getContext().pImpl->InstructionMetadata[this];
  assert(!Info.empty() && "bit out of sync with hash table");

  return Info.lookup(KindID);
}

void Instruction::getAllMetadataImpl(
    SmallVectorImpl<std::pair<unsigned, MDNode *>> &Result) const {
  Result.clear();
  
  // Handle 'dbg' as a special case since it is not stored in the hash table.
  if (DbgLoc) {
    Result.push_back(
        std::make_pair((unsigned)LLVMContext::MD_dbg, DbgLoc.getAsMDNode()));
    if (!hasMetadataHashEntry()) return;
  }

  assert(hasMetadataHashEntry() &&
         getContext().pImpl->InstructionMetadata.count(this) &&
         "Shouldn't have called this");
  const auto &Info = getContext().pImpl->InstructionMetadata.find(this)->second;
  assert(!Info.empty() && "Shouldn't have called this");
  Info.getAll(Result);
}

void Instruction::getAllMetadataOtherThanDebugLocImpl(
    SmallVectorImpl<std::pair<unsigned, MDNode *>> &Result) const {
  Result.clear();
  assert(hasMetadataHashEntry() &&
         getContext().pImpl->InstructionMetadata.count(this) &&
         "Shouldn't have called this");
  const auto &Info = getContext().pImpl->InstructionMetadata.find(this)->second;
  assert(!Info.empty() && "Shouldn't have called this");
  Info.getAll(Result);
}

/// clearMetadataHashEntries - Clear all hashtable-based metadata from
/// this instruction.
void Instruction::clearMetadataHashEntries() {
  assert(hasMetadataHashEntry() && "Caller should check");
  getContext().pImpl->InstructionMetadata.erase(this);
  setHasMetadataHashEntry(false);
}

MDNode *Function::getMetadata(unsigned KindID) const {
  if (!hasMetadata())
    return nullptr;
  return getContext().pImpl->FunctionMetadata[this].lookup(KindID);
}

MDNode *Function::getMetadata(StringRef Kind) const {
  if (!hasMetadata())
    return nullptr;
  return getMetadata(getContext().getMDKindID(Kind));
}

void Function::setMetadata(unsigned KindID, MDNode *MD) {
  if (MD) {
    if (!hasMetadata())
      setHasMetadataHashEntry(true);

    getContext().pImpl->FunctionMetadata[this].set(KindID, *MD);
    return;
  }

  // Nothing to unset.
  if (!hasMetadata())
    return;

  auto &Store = getContext().pImpl->FunctionMetadata[this];
  Store.erase(KindID);
  if (Store.empty())
    clearMetadata();
}

void Function::setMetadata(StringRef Kind, MDNode *MD) {
  if (!MD && !hasMetadata())
    return;
  setMetadata(getContext().getMDKindID(Kind), MD);
}

void Function::getAllMetadata(
    SmallVectorImpl<std::pair<unsigned, MDNode *>> &MDs) const {
  MDs.clear();

  if (!hasMetadata())
    return;

  getContext().pImpl->FunctionMetadata[this].getAll(MDs);
}

void Function::dropUnknownMetadata(ArrayRef<unsigned> KnownIDs) {
  if (!hasMetadata())
    return;
  if (KnownIDs.empty()) {
    clearMetadata();
    return;
  }

  SmallSet<unsigned, 5> KnownSet;
  KnownSet.insert(KnownIDs.begin(), KnownIDs.end());

  auto &Store = getContext().pImpl->FunctionMetadata[this];
  assert(!Store.empty());

  Store.remove_if([&KnownSet](const std::pair<unsigned, TrackingMDNodeRef> &I) {
    return !KnownSet.count(I.first);
  });

  if (Store.empty())
    clearMetadata();
}

void Function::clearMetadata() {
  if (!hasMetadata())
    return;
  getContext().pImpl->FunctionMetadata.erase(this);
  setHasMetadataHashEntry(false);
}
