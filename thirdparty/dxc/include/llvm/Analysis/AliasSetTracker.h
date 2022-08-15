//===- llvm/Analysis/AliasSetTracker.h - Build Alias Sets -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines two classes: AliasSetTracker and AliasSet.  These interface
// are used to classify a collection of pointer references into a maximal number
// of disjoint sets.  Each AliasSet object constructed by the AliasSetTracker
// object refers to memory disjoint from the other sets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_ALIASSETTRACKER_H
#define LLVM_ANALYSIS_ALIASSETTRACKER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/ValueHandle.h"
#include <vector>

namespace llvm {

class AliasAnalysis;
class LoadInst;
class StoreInst;
class VAArgInst;
class AliasSetTracker;
class AliasSet;

class AliasSet : public ilist_node<AliasSet> {
  friend class AliasSetTracker;

  class PointerRec {
    Value *Val;  // The pointer this record corresponds to.
    PointerRec **PrevInList, *NextInList;
    AliasSet *AS;
    uint64_t Size;
    AAMDNodes AAInfo;
  public:
    PointerRec(Value *V)
      : Val(V), PrevInList(nullptr), NextInList(nullptr), AS(nullptr), Size(0),
        AAInfo(DenseMapInfo<AAMDNodes>::getEmptyKey()) {}

    Value *getValue() const { return Val; }
    
    PointerRec *getNext() const { return NextInList; }
    bool hasAliasSet() const { return AS != nullptr; }

    PointerRec** setPrevInList(PointerRec **PIL) {
      PrevInList = PIL;
      return &NextInList;
    }

    void updateSizeAndAAInfo(uint64_t NewSize, const AAMDNodes &NewAAInfo) {
      if (NewSize > Size) Size = NewSize;

      if (AAInfo == DenseMapInfo<AAMDNodes>::getEmptyKey())
        // We don't have a AAInfo yet. Set it to NewAAInfo.
        AAInfo = NewAAInfo;
      else if (AAInfo != NewAAInfo)
        // NewAAInfo conflicts with AAInfo.
        AAInfo = DenseMapInfo<AAMDNodes>::getTombstoneKey();
    }

    uint64_t getSize() const { return Size; }

    /// getAAInfo - Return the AAInfo, or null if there is no
    /// information or conflicting information.
    AAMDNodes getAAInfo() const {
      // If we have missing or conflicting AAInfo, return null.
      if (AAInfo == DenseMapInfo<AAMDNodes>::getEmptyKey() ||
          AAInfo == DenseMapInfo<AAMDNodes>::getTombstoneKey())
        return AAMDNodes();
      return AAInfo;
    }

    AliasSet *getAliasSet(AliasSetTracker &AST) {
      assert(AS && "No AliasSet yet!");
      if (AS->Forward) {
        AliasSet *OldAS = AS;
        AS = OldAS->getForwardedTarget(AST);
        AS->addRef();
        OldAS->dropRef(AST);
      }
      return AS;
    }

    void setAliasSet(AliasSet *as) {
      assert(!AS && "Already have an alias set!");
      AS = as;
    }

    void eraseFromList() {
      if (NextInList) NextInList->PrevInList = PrevInList;
      *PrevInList = NextInList;
      if (AS->PtrListEnd == &NextInList) {
        AS->PtrListEnd = PrevInList;
        assert(*AS->PtrListEnd == nullptr && "List not terminated right!");
      }
      delete this;
    }
  };

  PointerRec *PtrList, **PtrListEnd;  // Doubly linked list of nodes.
  AliasSet *Forward;             // Forwarding pointer.

  // All instructions without a specific address in this alias set.
  std::vector<AssertingVH<Instruction> > UnknownInsts;

  // RefCount - Number of nodes pointing to this AliasSet plus the number of
  // AliasSets forwarding to it.
  unsigned RefCount : 28;

  /// The kinds of access this alias set models.
  ///
  /// We keep track of whether this alias set merely refers to the locations of
  /// memory (and not any particular access), whether it modifies or references
  /// the memory, or whether it does both. The lattice goes from "NoAccess" to
  /// either RefAccess or ModAccess, then to ModRefAccess as necessary.
  enum AccessLattice {
    NoAccess = 0,
    RefAccess = 1,
    ModAccess = 2,
    ModRefAccess = RefAccess | ModAccess
  };
  unsigned Access : 2;

  /// The kind of alias relationship between pointers of the set.
  ///
  /// These represent conservatively correct alias results between any members
  /// of the set. We represent these independently of the values of alias
  /// results in order to pack it into a single bit. Lattice goes from
  /// MustAlias to MayAlias.
  enum AliasLattice {
    SetMustAlias = 0, SetMayAlias = 1
  };
  unsigned Alias : 1;

  // Volatile - True if this alias set contains volatile loads or stores.
  bool Volatile : 1;

  void addRef() { ++RefCount; }
  void dropRef(AliasSetTracker &AST) {
    assert(RefCount >= 1 && "Invalid reference count detected!");
    if (--RefCount == 0)
      removeFromTracker(AST);
  }

  Instruction *getUnknownInst(unsigned i) const {
    assert(i < UnknownInsts.size());
    return UnknownInsts[i];
  }
  
public:
  /// Accessors...
  bool isRef() const { return Access & RefAccess; }
  bool isMod() const { return Access & ModAccess; }
  bool isMustAlias() const { return Alias == SetMustAlias; }
  bool isMayAlias()  const { return Alias == SetMayAlias; }

  // isVolatile - Return true if this alias set contains volatile loads or
  // stores.
  bool isVolatile() const { return Volatile; }

  /// isForwardingAliasSet - Return true if this alias set should be ignored as
  /// part of the AliasSetTracker object.
  bool isForwardingAliasSet() const { return Forward; }

  /// mergeSetIn - Merge the specified alias set into this alias set...
  ///
  void mergeSetIn(AliasSet &AS, AliasSetTracker &AST);

  // Alias Set iteration - Allow access to all of the pointer which are part of
  // this alias set...
  class iterator;
  iterator begin() const { return iterator(PtrList); }
  iterator end()   const { return iterator(); }
  bool empty() const { return PtrList == nullptr; }

  void print(raw_ostream &OS) const;
  void dump() const;

  /// Define an iterator for alias sets... this is just a forward iterator.
  class iterator : public std::iterator<std::forward_iterator_tag,
                                        PointerRec, ptrdiff_t> {
    PointerRec *CurNode;
  public:
    explicit iterator(PointerRec *CN = nullptr) : CurNode(CN) {}

    bool operator==(const iterator& x) const {
      return CurNode == x.CurNode;
    }
    bool operator!=(const iterator& x) const { return !operator==(x); }

    value_type &operator*() const {
      assert(CurNode && "Dereferencing AliasSet.end()!");
      return *CurNode;
    }
    value_type *operator->() const { return &operator*(); }

    Value *getPointer() const { return CurNode->getValue(); }
    uint64_t getSize() const { return CurNode->getSize(); }
    AAMDNodes getAAInfo() const { return CurNode->getAAInfo(); }

    iterator& operator++() {                // Preincrement
      assert(CurNode && "Advancing past AliasSet.end()!");
      CurNode = CurNode->getNext();
      return *this;
    }
    iterator operator++(int) { // Postincrement
      iterator tmp = *this; ++*this; return tmp;
    }
  };

private:
  // Can only be created by AliasSetTracker. Also, ilist creates one
  // to serve as a sentinel.
  friend struct ilist_sentinel_traits<AliasSet>;
  AliasSet()
    : PtrList(nullptr), PtrListEnd(&PtrList), Forward(nullptr), RefCount(0),
      Access(NoAccess), Alias(SetMustAlias), Volatile(false) {
  }

  AliasSet(const AliasSet &AS) = delete;
  void operator=(const AliasSet &AS) = delete;

  PointerRec *getSomePointer() const {
    return PtrList;
  }

  /// getForwardedTarget - Return the real alias set this represents.  If this
  /// has been merged with another set and is forwarding, return the ultimate
  /// destination set.  This also implements the union-find collapsing as well.
  AliasSet *getForwardedTarget(AliasSetTracker &AST) {
    if (!Forward) return this;

    AliasSet *Dest = Forward->getForwardedTarget(AST);
    if (Dest != Forward) {
      Dest->addRef();
      Forward->dropRef(AST);
      Forward = Dest;
    }
    return Dest;
  }

  void removeFromTracker(AliasSetTracker &AST);

  void addPointer(AliasSetTracker &AST, PointerRec &Entry, uint64_t Size,
                  const AAMDNodes &AAInfo,
                  bool KnownMustAlias = false);
  void addUnknownInst(Instruction *I, AliasAnalysis &AA);
  void removeUnknownInst(AliasSetTracker &AST, Instruction *I) {
    bool WasEmpty = UnknownInsts.empty();
    for (size_t i = 0, e = UnknownInsts.size(); i != e; ++i)
      if (UnknownInsts[i] == I) {
        UnknownInsts[i] = UnknownInsts.back();
        UnknownInsts.pop_back();
        --i; --e;  // Revisit the moved entry.
      }
    if (!WasEmpty && UnknownInsts.empty())
      dropRef(AST);
  }
  void setVolatile() { Volatile = true; }

public:
  /// aliasesPointer - Return true if the specified pointer "may" (or must)
  /// alias one of the members in the set.
  ///
  bool aliasesPointer(const Value *Ptr, uint64_t Size, const AAMDNodes &AAInfo,
                      AliasAnalysis &AA) const;
  bool aliasesUnknownInst(const Instruction *Inst, AliasAnalysis &AA) const;
};

inline raw_ostream& operator<<(raw_ostream &OS, const AliasSet &AS) {
  AS.print(OS);
  return OS;
}


class AliasSetTracker {
  /// CallbackVH - A CallbackVH to arrange for AliasSetTracker to be
  /// notified whenever a Value is deleted.
  class ASTCallbackVH : public CallbackVH {
    AliasSetTracker *AST;
    void deleted() override;
    void allUsesReplacedWith(Value *) override;
  public:
    ASTCallbackVH(Value *V, AliasSetTracker *AST = nullptr);
    ASTCallbackVH &operator=(Value *V);
  };
  /// ASTCallbackVHDenseMapInfo - Traits to tell DenseMap that tell us how to
  /// compare and hash the value handle.
  struct ASTCallbackVHDenseMapInfo : public DenseMapInfo<Value *> {};

  AliasAnalysis &AA;
  ilist<AliasSet> AliasSets;

  typedef DenseMap<ASTCallbackVH, AliasSet::PointerRec*,
                   ASTCallbackVHDenseMapInfo>
    PointerMapType;

  // Map from pointers to their node
  PointerMapType PointerMap;

public:
  /// AliasSetTracker ctor - Create an empty collection of AliasSets, and use
  /// the specified alias analysis object to disambiguate load and store
  /// addresses.
  explicit AliasSetTracker(AliasAnalysis &aa) : AA(aa) {}
  ~AliasSetTracker() { clear(); }

  /// add methods - These methods are used to add different types of
  /// instructions to the alias sets.  Adding a new instruction can result in
  /// one of three actions happening:
  ///
  ///   1. If the instruction doesn't alias any other sets, create a new set.
  ///   2. If the instruction aliases exactly one set, add it to the set
  ///   3. If the instruction aliases multiple sets, merge the sets, and add
  ///      the instruction to the result.
  ///
  /// These methods return true if inserting the instruction resulted in the
  /// addition of a new alias set (i.e., the pointer did not alias anything).
  ///
  bool add(Value *Ptr, uint64_t Size, const AAMDNodes &AAInfo); // Add a loc.
  bool add(LoadInst *LI);
  bool add(StoreInst *SI);
  bool add(VAArgInst *VAAI);
  bool add(Instruction *I);       // Dispatch to one of the other add methods...
  void add(BasicBlock &BB);       // Add all instructions in basic block
  void add(const AliasSetTracker &AST); // Add alias relations from another AST
  bool addUnknown(Instruction *I);

  /// remove methods - These methods are used to remove all entries that might
  /// be aliased by the specified instruction.  These methods return true if any
  /// alias sets were eliminated.
  // Remove a location
  bool remove(Value *Ptr, uint64_t Size, const AAMDNodes &AAInfo);
  bool remove(LoadInst *LI);
  bool remove(StoreInst *SI);
  bool remove(VAArgInst *VAAI);
  bool remove(Instruction *I);
  void remove(AliasSet &AS);
  bool removeUnknown(Instruction *I);
  
  void clear();

  /// getAliasSets - Return the alias sets that are active.
  ///
  const ilist<AliasSet> &getAliasSets() const { return AliasSets; }

  /// getAliasSetForPointer - Return the alias set that the specified pointer
  /// lives in.  If the New argument is non-null, this method sets the value to
  /// true if a new alias set is created to contain the pointer (because the
  /// pointer didn't alias anything).
  AliasSet &getAliasSetForPointer(Value *P, uint64_t Size,
                                  const AAMDNodes &AAInfo,
                                  bool *New = nullptr);

  /// getAliasSetForPointerIfExists - Return the alias set containing the
  /// location specified if one exists, otherwise return null.
  AliasSet *getAliasSetForPointerIfExists(const Value *P, uint64_t Size,
                                          const AAMDNodes &AAInfo) {
    return findAliasSetForPointer(P, Size, AAInfo);
  }

  /// containsPointer - Return true if the specified location is represented by
  /// this alias set, false otherwise.  This does not modify the AST object or
  /// alias sets.
  bool containsPointer(const Value *P, uint64_t Size,
                       const AAMDNodes &AAInfo) const;

  /// Return true if the specified instruction "may" (or must) alias one of the
  /// members in any of the sets.
  bool containsUnknown(const Instruction *I) const;

  /// getAliasAnalysis - Return the underlying alias analysis object used by
  /// this tracker.
  AliasAnalysis &getAliasAnalysis() const { return AA; }

  /// deleteValue method - This method is used to remove a pointer value from
  /// the AliasSetTracker entirely.  It should be used when an instruction is
  /// deleted from the program to update the AST.  If you don't use this, you
  /// would have dangling pointers to deleted instructions.
  ///
  void deleteValue(Value *PtrVal);

  /// copyValue - This method should be used whenever a preexisting value in the
  /// program is copied or cloned, introducing a new value.  Note that it is ok
  /// for clients that use this method to introduce the same value multiple
  /// times: if the tracker already knows about a value, it will ignore the
  /// request.
  ///
  void copyValue(Value *From, Value *To);


  typedef ilist<AliasSet>::iterator iterator;
  typedef ilist<AliasSet>::const_iterator const_iterator;

  const_iterator begin() const { return AliasSets.begin(); }
  const_iterator end()   const { return AliasSets.end(); }

  iterator begin() { return AliasSets.begin(); }
  iterator end()   { return AliasSets.end(); }

  void print(raw_ostream &OS) const;
  void dump() const;

private:
  friend class AliasSet;
  void removeAliasSet(AliasSet *AS);

  // getEntryFor - Just like operator[] on the map, except that it creates an
  // entry for the pointer if it doesn't already exist.
  AliasSet::PointerRec &getEntryFor(Value *V) {
    AliasSet::PointerRec *&Entry = PointerMap[ASTCallbackVH(V, this)];
    if (!Entry)
      Entry = new AliasSet::PointerRec(V);
    return *Entry;
  }

  AliasSet &addPointer(Value *P, uint64_t Size, const AAMDNodes &AAInfo,
                       AliasSet::AccessLattice E,
                       bool &NewSet) {
    NewSet = false;
    AliasSet &AS = getAliasSetForPointer(P, Size, AAInfo, &NewSet);
    AS.Access |= E;
    return AS;
  }
  AliasSet *findAliasSetForPointer(const Value *Ptr, uint64_t Size,
                                   const AAMDNodes &AAInfo);

  AliasSet *findAliasSetForUnknownInst(Instruction *Inst);
};

inline raw_ostream& operator<<(raw_ostream &OS, const AliasSetTracker &AST) {
  AST.print(OS);
  return OS;
}

} // End llvm namespace

#endif
