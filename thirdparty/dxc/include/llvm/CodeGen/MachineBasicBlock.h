//===-- llvm/CodeGen/MachineBasicBlock.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Collect the sequence of machine instructions for a basic block.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEBASICBLOCK_H
#define LLVM_CODEGEN_MACHINEBASICBLOCK_H

#include "llvm/ADT/GraphTraits.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Support/DataTypes.h"
#include <functional>

namespace llvm {

class Pass;
class BasicBlock;
class MachineFunction;
class MCSymbol;
class SlotIndexes;
class StringRef;
class raw_ostream;
class MachineBranchProbabilityInfo;

template <>
struct ilist_traits<MachineInstr> : public ilist_default_traits<MachineInstr> {
private:
  mutable ilist_half_node<MachineInstr> Sentinel;

  // this is only set by the MachineBasicBlock owning the LiveList
  friend class MachineBasicBlock;
  MachineBasicBlock* Parent;

public:
  MachineInstr *createSentinel() const {
    return static_cast<MachineInstr*>(&Sentinel);
  }
  void destroySentinel(MachineInstr *) const {}

  MachineInstr *provideInitialHead() const { return createSentinel(); }
  MachineInstr *ensureHead(MachineInstr*) const { return createSentinel(); }
  static void noteHead(MachineInstr*, MachineInstr*) {}

  void addNodeToList(MachineInstr* N);
  void removeNodeFromList(MachineInstr* N);
  void transferNodesFromList(ilist_traits &SrcTraits,
                             ilist_iterator<MachineInstr> first,
                             ilist_iterator<MachineInstr> last);
  void deleteNode(MachineInstr *N);
private:
  void createNode(const MachineInstr &);
};

class MachineBasicBlock : public ilist_node<MachineBasicBlock> {
  typedef ilist<MachineInstr> Instructions;
  Instructions Insts;
  const BasicBlock *BB;
  int Number;
  MachineFunction *xParent;

  /// Predecessors/Successors - Keep track of the predecessor / successor
  /// basicblocks.
  std::vector<MachineBasicBlock *> Predecessors;
  std::vector<MachineBasicBlock *> Successors;

  /// Weights - Keep track of the weights to the successors. This vector
  /// has the same order as Successors, or it is empty if we don't use it
  /// (disable optimization).
  std::vector<uint32_t> Weights;
  typedef std::vector<uint32_t>::iterator weight_iterator;
  typedef std::vector<uint32_t>::const_iterator const_weight_iterator;

  /// LiveIns - Keep track of the physical registers that are livein of
  /// the basicblock.
  std::vector<unsigned> LiveIns;

  /// Alignment - Alignment of the basic block. Zero if the basic block does
  /// not need to be aligned.
  /// The alignment is specified as log2(bytes).
  unsigned Alignment;

  /// IsLandingPad - Indicate that this basic block is entered via an
  /// exception handler.
  bool IsLandingPad;

  /// AddressTaken - Indicate that this basic block is potentially the
  /// target of an indirect branch.
  bool AddressTaken;

  /// \brief since getSymbol is a relatively heavy-weight operation, the symbol
  /// is only computed once and is cached.
  mutable MCSymbol *CachedMCSymbol;

  // Intrusive list support
  MachineBasicBlock() {}

  explicit MachineBasicBlock(MachineFunction &mf, const BasicBlock *bb);

  ~MachineBasicBlock();

  // MachineBasicBlocks are allocated and owned by MachineFunction.
  friend class MachineFunction;

public:
  /// getBasicBlock - Return the LLVM basic block that this instance
  /// corresponded to originally. Note that this may be NULL if this instance
  /// does not correspond directly to an LLVM basic block.
  ///
  const BasicBlock *getBasicBlock() const { return BB; }

  /// getName - Return the name of the corresponding LLVM basic block, or
  /// "(null)".
  StringRef getName() const;

  /// getFullName - Return a formatted string to identify this block and its
  /// parent function.
  std::string getFullName() const;

  /// hasAddressTaken - Test whether this block is potentially the target
  /// of an indirect branch.
  bool hasAddressTaken() const { return AddressTaken; }

  /// setHasAddressTaken - Set this block to reflect that it potentially
  /// is the target of an indirect branch.
  void setHasAddressTaken() { AddressTaken = true; }

  /// getParent - Return the MachineFunction containing this basic block.
  ///
  const MachineFunction *getParent() const { return xParent; }
  MachineFunction *getParent() { return xParent; }


  /// bundle_iterator - MachineBasicBlock iterator that automatically skips over
  /// MIs that are inside bundles (i.e. walk top level MIs only).
  template<typename Ty, typename IterTy>
  class bundle_iterator
    : public std::iterator<std::bidirectional_iterator_tag, Ty, ptrdiff_t> {
    IterTy MII;

  public:
    bundle_iterator(IterTy mii) : MII(mii) {}

    bundle_iterator(Ty &mi) : MII(mi) {
      assert(!mi.isBundledWithPred() &&
             "It's not legal to initialize bundle_iterator with a bundled MI");
    }
    bundle_iterator(Ty *mi) : MII(mi) {
      assert((!mi || !mi->isBundledWithPred()) &&
             "It's not legal to initialize bundle_iterator with a bundled MI");
    }
    // Template allows conversion from const to nonconst.
    template<class OtherTy, class OtherIterTy>
    bundle_iterator(const bundle_iterator<OtherTy, OtherIterTy> &I)
      : MII(I.getInstrIterator()) {}
    bundle_iterator() : MII(nullptr) {}

    Ty &operator*() const { return *MII; }
    Ty *operator->() const { return &operator*(); }

    operator Ty*() const { return MII; }

    bool operator==(const bundle_iterator &x) const {
      return MII == x.MII;
    }
    bool operator!=(const bundle_iterator &x) const {
      return !operator==(x);
    }

    // Increment and decrement operators...
    bundle_iterator &operator--() {      // predecrement - Back up
      do --MII;
      while (MII->isBundledWithPred());
      return *this;
    }
    bundle_iterator &operator++() {      // preincrement - Advance
      while (MII->isBundledWithSucc())
        ++MII;
      ++MII;
      return *this;
    }
    bundle_iterator operator--(int) {    // postdecrement operators...
      bundle_iterator tmp = *this;
      --*this;
      return tmp;
    }
    bundle_iterator operator++(int) {    // postincrement operators...
      bundle_iterator tmp = *this;
      ++*this;
      return tmp;
    }

    IterTy getInstrIterator() const {
      return MII;
    }
  };

  typedef Instructions::iterator                                 instr_iterator;
  typedef Instructions::const_iterator                     const_instr_iterator;
  typedef std::reverse_iterator<instr_iterator>          reverse_instr_iterator;
  typedef
  std::reverse_iterator<const_instr_iterator>      const_reverse_instr_iterator;

  typedef
  bundle_iterator<MachineInstr,instr_iterator>                         iterator;
  typedef
  bundle_iterator<const MachineInstr,const_instr_iterator>       const_iterator;
  typedef std::reverse_iterator<const_iterator>          const_reverse_iterator;
  typedef std::reverse_iterator<iterator>                      reverse_iterator;


  unsigned size() const { return (unsigned)Insts.size(); }
  bool empty() const { return Insts.empty(); }

  MachineInstr       &instr_front()       { return Insts.front(); }
  MachineInstr       &instr_back()        { return Insts.back();  }
  const MachineInstr &instr_front() const { return Insts.front(); }
  const MachineInstr &instr_back()  const { return Insts.back();  }

  MachineInstr       &front()             { return Insts.front(); }
  MachineInstr       &back()              { return *--end();      }
  const MachineInstr &front()       const { return Insts.front(); }
  const MachineInstr &back()        const { return *--end();      }

  instr_iterator                instr_begin()       { return Insts.begin();  }
  const_instr_iterator          instr_begin() const { return Insts.begin();  }
  instr_iterator                  instr_end()       { return Insts.end();    }
  const_instr_iterator            instr_end() const { return Insts.end();    }
  reverse_instr_iterator       instr_rbegin()       { return Insts.rbegin(); }
  const_reverse_instr_iterator instr_rbegin() const { return Insts.rbegin(); }
  reverse_instr_iterator       instr_rend  ()       { return Insts.rend();   }
  const_reverse_instr_iterator instr_rend  () const { return Insts.rend();   }

  iterator                begin()       { return instr_begin();  }
  const_iterator          begin() const { return instr_begin();  }
  iterator                end  ()       { return instr_end();    }
  const_iterator          end  () const { return instr_end();    }
  reverse_iterator       rbegin()       { return instr_rbegin(); }
  const_reverse_iterator rbegin() const { return instr_rbegin(); }
  reverse_iterator       rend  ()       { return instr_rend();   }
  const_reverse_iterator rend  () const { return instr_rend();   }

  inline iterator_range<iterator> terminators() {
    return iterator_range<iterator>(getFirstTerminator(), end());
  }
  inline iterator_range<const_iterator> terminators() const {
    return iterator_range<const_iterator>(getFirstTerminator(), end());
  }

  // Machine-CFG iterators
  typedef std::vector<MachineBasicBlock *>::iterator       pred_iterator;
  typedef std::vector<MachineBasicBlock *>::const_iterator const_pred_iterator;
  typedef std::vector<MachineBasicBlock *>::iterator       succ_iterator;
  typedef std::vector<MachineBasicBlock *>::const_iterator const_succ_iterator;
  typedef std::vector<MachineBasicBlock *>::reverse_iterator
                                                         pred_reverse_iterator;
  typedef std::vector<MachineBasicBlock *>::const_reverse_iterator
                                                   const_pred_reverse_iterator;
  typedef std::vector<MachineBasicBlock *>::reverse_iterator
                                                         succ_reverse_iterator;
  typedef std::vector<MachineBasicBlock *>::const_reverse_iterator
                                                   const_succ_reverse_iterator;
  pred_iterator        pred_begin()       { return Predecessors.begin(); }
  const_pred_iterator  pred_begin() const { return Predecessors.begin(); }
  pred_iterator        pred_end()         { return Predecessors.end();   }
  const_pred_iterator  pred_end()   const { return Predecessors.end();   }
  pred_reverse_iterator        pred_rbegin()
                                          { return Predecessors.rbegin();}
  const_pred_reverse_iterator  pred_rbegin() const
                                          { return Predecessors.rbegin();}
  pred_reverse_iterator        pred_rend()
                                          { return Predecessors.rend();  }
  const_pred_reverse_iterator  pred_rend()   const
                                          { return Predecessors.rend();  }
  unsigned             pred_size()  const {
    return (unsigned)Predecessors.size();
  }
  bool                 pred_empty() const { return Predecessors.empty(); }
  succ_iterator        succ_begin()       { return Successors.begin();   }
  const_succ_iterator  succ_begin() const { return Successors.begin();   }
  succ_iterator        succ_end()         { return Successors.end();     }
  const_succ_iterator  succ_end()   const { return Successors.end();     }
  succ_reverse_iterator        succ_rbegin()
                                          { return Successors.rbegin();  }
  const_succ_reverse_iterator  succ_rbegin() const
                                          { return Successors.rbegin();  }
  succ_reverse_iterator        succ_rend()
                                          { return Successors.rend();    }
  const_succ_reverse_iterator  succ_rend()   const
                                          { return Successors.rend();    }
  unsigned             succ_size()  const {
    return (unsigned)Successors.size();
  }
  bool                 succ_empty() const { return Successors.empty();   }

  inline iterator_range<pred_iterator> predecessors() {
    return iterator_range<pred_iterator>(pred_begin(), pred_end());
  }
  inline iterator_range<const_pred_iterator> predecessors() const {
    return iterator_range<const_pred_iterator>(pred_begin(), pred_end());
  }
  inline iterator_range<succ_iterator> successors() {
    return iterator_range<succ_iterator>(succ_begin(), succ_end());
  }
  inline iterator_range<const_succ_iterator> successors() const {
    return iterator_range<const_succ_iterator>(succ_begin(), succ_end());
  }

  // LiveIn management methods.

  /// Adds the specified register as a live in. Note that it is an error to add
  /// the same register to the same set more than once unless the intention is
  /// to call sortUniqueLiveIns after all registers are added.
  void addLiveIn(unsigned Reg) { LiveIns.push_back(Reg); }

  /// Sorts and uniques the LiveIns vector. It can be significantly faster to do
  /// this than repeatedly calling isLiveIn before calling addLiveIn for every
  /// LiveIn insertion.
  void sortUniqueLiveIns() {
    std::sort(LiveIns.begin(), LiveIns.end());
    LiveIns.erase(std::unique(LiveIns.begin(), LiveIns.end()), LiveIns.end());
  }

  /// Add PhysReg as live in to this block, and ensure that there is a copy of
  /// PhysReg to a virtual register of class RC. Return the virtual register
  /// that is a copy of the live in PhysReg.
  unsigned addLiveIn(unsigned PhysReg, const TargetRegisterClass *RC);

  /// removeLiveIn - Remove the specified register from the live in set.
  ///
  void removeLiveIn(unsigned Reg);

  /// isLiveIn - Return true if the specified register is in the live in set.
  ///
  bool isLiveIn(unsigned Reg) const;

  // Iteration support for live in sets.  These sets are kept in sorted
  // order by their register number.
  typedef std::vector<unsigned>::const_iterator livein_iterator;
  livein_iterator livein_begin() const { return LiveIns.begin(); }
  livein_iterator livein_end()   const { return LiveIns.end(); }
  bool            livein_empty() const { return LiveIns.empty(); }

  /// getAlignment - Return alignment of the basic block.
  /// The alignment is specified as log2(bytes).
  ///
  unsigned getAlignment() const { return Alignment; }

  /// setAlignment - Set alignment of the basic block.
  /// The alignment is specified as log2(bytes).
  ///
  void setAlignment(unsigned Align) { Alignment = Align; }

  /// isLandingPad - Returns true if the block is a landing pad. That is
  /// this basic block is entered via an exception handler.
  bool isLandingPad() const { return IsLandingPad; }

  /// setIsLandingPad - Indicates the block is a landing pad.  That is
  /// this basic block is entered via an exception handler.
  void setIsLandingPad(bool V = true) { IsLandingPad = V; }

  /// getLandingPadSuccessor - If this block has a successor that is a landing
  /// pad, return it. Otherwise return NULL.
  const MachineBasicBlock *getLandingPadSuccessor() const;

  // Code Layout methods.

  /// moveBefore/moveAfter - move 'this' block before or after the specified
  /// block.  This only moves the block, it does not modify the CFG or adjust
  /// potential fall-throughs at the end of the block.
  void moveBefore(MachineBasicBlock *NewAfter);
  void moveAfter(MachineBasicBlock *NewBefore);

  /// updateTerminator - Update the terminator instructions in block to account
  /// for changes to the layout. If the block previously used a fallthrough,
  /// it may now need a branch, and if it previously used branching it may now
  /// be able to use a fallthrough.
  void updateTerminator();

  // Machine-CFG mutators

  /// addSuccessor - Add succ as a successor of this MachineBasicBlock.
  /// The Predecessors list of succ is automatically updated. WEIGHT
  /// parameter is stored in Weights list and it may be used by
  /// MachineBranchProbabilityInfo analysis to calculate branch probability.
  ///
  /// Note that duplicate Machine CFG edges are not allowed.
  ///
  void addSuccessor(MachineBasicBlock *succ, uint32_t weight = 0);

  /// Set successor weight of a given iterator.
  void setSuccWeight(succ_iterator I, uint32_t weight);

  /// removeSuccessor - Remove successor from the successors list of this
  /// MachineBasicBlock. The Predecessors list of succ is automatically updated.
  ///
  void removeSuccessor(MachineBasicBlock *succ);

  /// removeSuccessor - Remove specified successor from the successors list of
  /// this MachineBasicBlock. The Predecessors list of succ is automatically
  /// updated.  Return the iterator to the element after the one removed.
  ///
  succ_iterator removeSuccessor(succ_iterator I);

  /// replaceSuccessor - Replace successor OLD with NEW and update weight info.
  ///
  void replaceSuccessor(MachineBasicBlock *Old, MachineBasicBlock *New);


  /// transferSuccessors - Transfers all the successors from MBB to this
  /// machine basic block (i.e., copies all the successors fromMBB and
  /// remove all the successors from fromMBB).
  void transferSuccessors(MachineBasicBlock *fromMBB);

  /// transferSuccessorsAndUpdatePHIs - Transfers all the successors, as
  /// in transferSuccessors, and update PHI operands in the successor blocks
  /// which refer to fromMBB to refer to this.
  void transferSuccessorsAndUpdatePHIs(MachineBasicBlock *fromMBB);

  /// isPredecessor - Return true if the specified MBB is a predecessor of this
  /// block.
  bool isPredecessor(const MachineBasicBlock *MBB) const;

  /// isSuccessor - Return true if the specified MBB is a successor of this
  /// block.
  bool isSuccessor(const MachineBasicBlock *MBB) const;

  /// isLayoutSuccessor - Return true if the specified MBB will be emitted
  /// immediately after this block, such that if this block exits by
  /// falling through, control will transfer to the specified MBB. Note
  /// that MBB need not be a successor at all, for example if this block
  /// ends with an unconditional branch to some other block.
  bool isLayoutSuccessor(const MachineBasicBlock *MBB) const;

  /// canFallThrough - Return true if the block can implicitly transfer
  /// control to the block after it by falling off the end of it.  This should
  /// return false if it can reach the block after it, but it uses an explicit
  /// branch to do so (e.g., a table jump).  True is a conservative answer.
  bool canFallThrough();

  /// Returns a pointer to the first instruction in this block that is not a
  /// PHINode instruction. When adding instructions to the beginning of the
  /// basic block, they should be added before the returned value, not before
  /// the first instruction, which might be PHI.
  /// Returns end() is there's no non-PHI instruction.
  iterator getFirstNonPHI();

  /// SkipPHIsAndLabels - Return the first instruction in MBB after I that is
  /// not a PHI or a label. This is the correct point to insert copies at the
  /// beginning of a basic block.
  iterator SkipPHIsAndLabels(iterator I);

  /// getFirstTerminator - returns an iterator to the first terminator
  /// instruction of this basic block. If a terminator does not exist,
  /// it returns end()
  iterator getFirstTerminator();
  const_iterator getFirstTerminator() const {
    return const_cast<MachineBasicBlock *>(this)->getFirstTerminator();
  }

  /// getFirstInstrTerminator - Same getFirstTerminator but it ignores bundles
  /// and return an instr_iterator instead.
  instr_iterator getFirstInstrTerminator();

  /// getFirstNonDebugInstr - returns an iterator to the first non-debug
  /// instruction in the basic block, or end()
  iterator getFirstNonDebugInstr();
  const_iterator getFirstNonDebugInstr() const {
    return const_cast<MachineBasicBlock *>(this)->getFirstNonDebugInstr();
  }

  /// getLastNonDebugInstr - returns an iterator to the last non-debug
  /// instruction in the basic block, or end()
  iterator getLastNonDebugInstr();
  const_iterator getLastNonDebugInstr() const {
    return const_cast<MachineBasicBlock *>(this)->getLastNonDebugInstr();
  }

  /// SplitCriticalEdge - Split the critical edge from this block to the
  /// given successor block, and return the newly created block, or null
  /// if splitting is not possible.
  ///
  /// This function updates LiveVariables, MachineDominatorTree, and
  /// MachineLoopInfo, as applicable.
  MachineBasicBlock *SplitCriticalEdge(MachineBasicBlock *Succ, Pass *P);

  void pop_front() { Insts.pop_front(); }
  void pop_back() { Insts.pop_back(); }
  void push_back(MachineInstr *MI) { Insts.push_back(MI); }

  /// Insert MI into the instruction list before I, possibly inside a bundle.
  ///
  /// If the insertion point is inside a bundle, MI will be added to the bundle,
  /// otherwise MI will not be added to any bundle. That means this function
  /// alone can't be used to prepend or append instructions to bundles. See
  /// MIBundleBuilder::insert() for a more reliable way of doing that.
  instr_iterator insert(instr_iterator I, MachineInstr *M);

  /// Insert a range of instructions into the instruction list before I.
  template<typename IT>
  void insert(iterator I, IT S, IT E) {
    assert((I == end() || I->getParent() == this) &&
           "iterator points outside of basic block");
    Insts.insert(I.getInstrIterator(), S, E);
  }

  /// Insert MI into the instruction list before I.
  iterator insert(iterator I, MachineInstr *MI) {
    assert((I == end() || I->getParent() == this) &&
           "iterator points outside of basic block");
    assert(!MI->isBundledWithPred() && !MI->isBundledWithSucc() &&
           "Cannot insert instruction with bundle flags");
    return Insts.insert(I.getInstrIterator(), MI);
  }

  /// Insert MI into the instruction list after I.
  iterator insertAfter(iterator I, MachineInstr *MI) {
    assert((I == end() || I->getParent() == this) &&
           "iterator points outside of basic block");
    assert(!MI->isBundledWithPred() && !MI->isBundledWithSucc() &&
           "Cannot insert instruction with bundle flags");
    return Insts.insertAfter(I.getInstrIterator(), MI);
  }

  /// Remove an instruction from the instruction list and delete it.
  ///
  /// If the instruction is part of a bundle, the other instructions in the
  /// bundle will still be bundled after removing the single instruction.
  instr_iterator erase(instr_iterator I);

  /// Remove an instruction from the instruction list and delete it.
  ///
  /// If the instruction is part of a bundle, the other instructions in the
  /// bundle will still be bundled after removing the single instruction.
  instr_iterator erase_instr(MachineInstr *I) {
    return erase(instr_iterator(I));
  }

  /// Remove a range of instructions from the instruction list and delete them.
  iterator erase(iterator I, iterator E) {
    return Insts.erase(I.getInstrIterator(), E.getInstrIterator());
  }

  /// Remove an instruction or bundle from the instruction list and delete it.
  ///
  /// If I points to a bundle of instructions, they are all erased.
  iterator erase(iterator I) {
    return erase(I, std::next(I));
  }

  /// Remove an instruction from the instruction list and delete it.
  ///
  /// If I is the head of a bundle of instructions, the whole bundle will be
  /// erased.
  iterator erase(MachineInstr *I) {
    return erase(iterator(I));
  }

  /// Remove the unbundled instruction from the instruction list without
  /// deleting it.
  ///
  /// This function can not be used to remove bundled instructions, use
  /// remove_instr to remove individual instructions from a bundle.
  MachineInstr *remove(MachineInstr *I) {
    assert(!I->isBundled() && "Cannot remove bundled instructions");
    return Insts.remove(I);
  }

  /// Remove the possibly bundled instruction from the instruction list
  /// without deleting it.
  ///
  /// If the instruction is part of a bundle, the other instructions in the
  /// bundle will still be bundled after removing the single instruction.
  MachineInstr *remove_instr(MachineInstr *I);

  void clear() {
    Insts.clear();
  }

  /// Take an instruction from MBB 'Other' at the position From, and insert it
  /// into this MBB right before 'Where'.
  ///
  /// If From points to a bundle of instructions, the whole bundle is moved.
  void splice(iterator Where, MachineBasicBlock *Other, iterator From) {
    // The range splice() doesn't allow noop moves, but this one does.
    if (Where != From)
      splice(Where, Other, From, std::next(From));
  }

  /// Take a block of instructions from MBB 'Other' in the range [From, To),
  /// and insert them into this MBB right before 'Where'.
  ///
  /// The instruction at 'Where' must not be included in the range of
  /// instructions to move.
  void splice(iterator Where, MachineBasicBlock *Other,
              iterator From, iterator To) {
    Insts.splice(Where.getInstrIterator(), Other->Insts,
                 From.getInstrIterator(), To.getInstrIterator());
  }

  /// removeFromParent - This method unlinks 'this' from the containing
  /// function, and returns it, but does not delete it.
  MachineBasicBlock *removeFromParent();

  /// eraseFromParent - This method unlinks 'this' from the containing
  /// function and deletes it.
  void eraseFromParent();

  /// ReplaceUsesOfBlockWith - Given a machine basic block that branched to
  /// 'Old', change the code and CFG so that it branches to 'New' instead.
  void ReplaceUsesOfBlockWith(MachineBasicBlock *Old, MachineBasicBlock *New);

  /// CorrectExtraCFGEdges - Various pieces of code can cause excess edges in
  /// the CFG to be inserted.  If we have proven that MBB can only branch to
  /// DestA and DestB, remove any other MBB successors from the CFG. DestA and
  /// DestB can be null. Besides DestA and DestB, retain other edges leading
  /// to LandingPads (currently there can be only one; we don't check or require
  /// that here). Note it is possible that DestA and/or DestB are LandingPads.
  bool CorrectExtraCFGEdges(MachineBasicBlock *DestA,
                            MachineBasicBlock *DestB,
                            bool isCond);

  /// findDebugLoc - find the next valid DebugLoc starting at MBBI, skipping
  /// any DBG_VALUE instructions.  Return UnknownLoc if there is none.
  DebugLoc findDebugLoc(instr_iterator MBBI);
  DebugLoc findDebugLoc(iterator MBBI) {
    return findDebugLoc(MBBI.getInstrIterator());
  }

  /// Possible outcome of a register liveness query to computeRegisterLiveness()
  enum LivenessQueryResult {
    LQR_Live,            ///< Register is known to be live.
    LQR_OverlappingLive, ///< Register itself is not live, but some overlapping
                         ///< register is.
    LQR_Dead,            ///< Register is known to be dead.
    LQR_Unknown          ///< Register liveness not decidable from local
                         ///< neighborhood.
  };

  /// Return whether (physical) register \p Reg has been <def>ined and not
  /// <kill>ed as of just before \p Before.
  ///
  /// Search is localised to a neighborhood of \p Neighborhood instructions
  /// before (searching for defs or kills) and \p Neighborhood instructions
  /// after (searching just for defs) \p Before.
  ///
  /// \p Reg must be a physical register.
  LivenessQueryResult computeRegisterLiveness(const TargetRegisterInfo *TRI,
                                              unsigned Reg,
                                              const_iterator Before,
                                              unsigned Neighborhood=10) const;

  // Debugging methods.
  void dump() const;
  void print(raw_ostream &OS, SlotIndexes* = nullptr) const;
  void print(raw_ostream &OS, ModuleSlotTracker &MST,
             SlotIndexes * = nullptr) const;

  // Printing method used by LoopInfo.
  void printAsOperand(raw_ostream &OS, bool PrintType = true) const;

  /// getNumber - MachineBasicBlocks are uniquely numbered at the function
  /// level, unless they're not in a MachineFunction yet, in which case this
  /// will return -1.
  ///
  int getNumber() const { return Number; }
  void setNumber(int N) { Number = N; }

  /// getSymbol - Return the MCSymbol for this basic block.
  ///
  MCSymbol *getSymbol() const;


private:
  /// getWeightIterator - Return weight iterator corresponding to the I
  /// successor iterator.
  weight_iterator getWeightIterator(succ_iterator I);
  const_weight_iterator getWeightIterator(const_succ_iterator I) const;

  friend class MachineBranchProbabilityInfo;

  /// getSuccWeight - Return weight of the edge from this block to MBB. This
  /// method should NOT be called directly, but by using getEdgeWeight method
  /// from MachineBranchProbabilityInfo class.
  uint32_t getSuccWeight(const_succ_iterator Succ) const;


  // Methods used to maintain doubly linked list of blocks...
  friend struct ilist_traits<MachineBasicBlock>;

  // Machine-CFG mutators

  /// addPredecessor - Remove pred as a predecessor of this MachineBasicBlock.
  /// Don't do this unless you know what you're doing, because it doesn't
  /// update pred's successors list. Use pred->addSuccessor instead.
  ///
  void addPredecessor(MachineBasicBlock *pred);

  /// removePredecessor - Remove pred as a predecessor of this
  /// MachineBasicBlock. Don't do this unless you know what you're
  /// doing, because it doesn't update pred's successors list. Use
  /// pred->removeSuccessor instead.
  ///
  void removePredecessor(MachineBasicBlock *pred);
};

raw_ostream& operator<<(raw_ostream &OS, const MachineBasicBlock &MBB);

// This is useful when building IndexedMaps keyed on basic block pointers.
struct MBB2NumberFunctor {
  unsigned operator()(const MachineBasicBlock *MBB) const {
    return MBB->getNumber();
  }
};

//===--------------------------------------------------------------------===//
// GraphTraits specializations for machine basic block graphs (machine-CFGs)
//===--------------------------------------------------------------------===//

// Provide specializations of GraphTraits to be able to treat a
// MachineFunction as a graph of MachineBasicBlocks...
//

template <> struct GraphTraits<MachineBasicBlock *> {
  typedef MachineBasicBlock NodeType;
  typedef MachineBasicBlock::succ_iterator ChildIteratorType;

  static NodeType *getEntryNode(MachineBasicBlock *BB) { return BB; }
  static inline ChildIteratorType child_begin(NodeType *N) {
    return N->succ_begin();
  }
  static inline ChildIteratorType child_end(NodeType *N) {
    return N->succ_end();
  }
};

template <> struct GraphTraits<const MachineBasicBlock *> {
  typedef const MachineBasicBlock NodeType;
  typedef MachineBasicBlock::const_succ_iterator ChildIteratorType;

  static NodeType *getEntryNode(const MachineBasicBlock *BB) { return BB; }
  static inline ChildIteratorType child_begin(NodeType *N) {
    return N->succ_begin();
  }
  static inline ChildIteratorType child_end(NodeType *N) {
    return N->succ_end();
  }
};

// Provide specializations of GraphTraits to be able to treat a
// MachineFunction as a graph of MachineBasicBlocks... and to walk it
// in inverse order.  Inverse order for a function is considered
// to be when traversing the predecessor edges of a MBB
// instead of the successor edges.
//
template <> struct GraphTraits<Inverse<MachineBasicBlock*> > {
  typedef MachineBasicBlock NodeType;
  typedef MachineBasicBlock::pred_iterator ChildIteratorType;
  static NodeType *getEntryNode(Inverse<MachineBasicBlock *> G) {
    return G.Graph;
  }
  static inline ChildIteratorType child_begin(NodeType *N) {
    return N->pred_begin();
  }
  static inline ChildIteratorType child_end(NodeType *N) {
    return N->pred_end();
  }
};

template <> struct GraphTraits<Inverse<const MachineBasicBlock*> > {
  typedef const MachineBasicBlock NodeType;
  typedef MachineBasicBlock::const_pred_iterator ChildIteratorType;
  static NodeType *getEntryNode(Inverse<const MachineBasicBlock*> G) {
    return G.Graph;
  }
  static inline ChildIteratorType child_begin(NodeType *N) {
    return N->pred_begin();
  }
  static inline ChildIteratorType child_end(NodeType *N) {
    return N->pred_end();
  }
};



/// MachineInstrSpan provides an interface to get an iteration range
/// containing the instruction it was initialized with, along with all
/// those instructions inserted prior to or following that instruction
/// at some point after the MachineInstrSpan is constructed.
class MachineInstrSpan {
  MachineBasicBlock &MBB;
  MachineBasicBlock::iterator I, B, E;
public:
  MachineInstrSpan(MachineBasicBlock::iterator I)
    : MBB(*I->getParent()),
      I(I),
      B(I == MBB.begin() ? MBB.end() : std::prev(I)),
      E(std::next(I)) {}

  MachineBasicBlock::iterator begin() {
    return B == MBB.end() ? MBB.begin() : std::next(B);
  }
  MachineBasicBlock::iterator end() { return E; }
  bool empty() { return begin() == end(); }

  MachineBasicBlock::iterator getInitial() { return I; }
};

} // End llvm namespace

#endif
