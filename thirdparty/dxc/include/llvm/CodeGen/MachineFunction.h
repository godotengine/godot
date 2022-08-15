//===-- llvm/CodeGen/MachineFunction.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Collect native machine code for a function.  This class contains a list of
// MachineBasicBlock instances that make up the current compiled function.
//
// This class also contains pointers to various classes which hold
// target-specific information about the generated code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEFUNCTION_H
#define LLVM_CODEGEN_MACHINEFUNCTION_H

#include "llvm/ADT/ilist.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ArrayRecycler.h"
#include "llvm/Support/Recycler.h"

namespace llvm {

class Value;
class Function;
class GCModuleInfo;
class MachineRegisterInfo;
class MachineFrameInfo;
class MachineConstantPool;
class MachineJumpTableInfo;
class MachineModuleInfo;
class MCContext;
class Pass;
class TargetMachine;
class TargetSubtargetInfo;
class TargetRegisterClass;
struct MachinePointerInfo;

template <>
struct ilist_traits<MachineBasicBlock>
    : public ilist_default_traits<MachineBasicBlock> {
  mutable ilist_half_node<MachineBasicBlock> Sentinel;
public:
  MachineBasicBlock *createSentinel() const {
    return static_cast<MachineBasicBlock*>(&Sentinel);
  }
  void destroySentinel(MachineBasicBlock *) const {}

  MachineBasicBlock *provideInitialHead() const { return createSentinel(); }
  MachineBasicBlock *ensureHead(MachineBasicBlock*) const {
    return createSentinel();
  }
  static void noteHead(MachineBasicBlock*, MachineBasicBlock*) {}

  void addNodeToList(MachineBasicBlock* MBB);
  void removeNodeFromList(MachineBasicBlock* MBB);
  void deleteNode(MachineBasicBlock *MBB);
private:
  void createNode(const MachineBasicBlock &);
};

/// MachineFunctionInfo - This class can be derived from and used by targets to
/// hold private target-specific information for each MachineFunction.  Objects
/// of type are accessed/created with MF::getInfo and destroyed when the
/// MachineFunction is destroyed.
struct MachineFunctionInfo {
  virtual ~MachineFunctionInfo();

  /// \brief Factory function: default behavior is to call new using the
  /// supplied allocator.
  ///
  /// This function can be overridden in a derive class.
  template<typename Ty>
  static Ty *create(BumpPtrAllocator &Allocator, MachineFunction &MF) {
    return new (Allocator.Allocate<Ty>()) Ty(MF);
  }
};

class MachineFunction {
  const Function *Fn;
  const TargetMachine &Target;
  const TargetSubtargetInfo *STI;
  MCContext &Ctx;
  MachineModuleInfo &MMI;

  // RegInfo - Information about each register in use in the function.
  MachineRegisterInfo *RegInfo;

  // Used to keep track of target-specific per-machine function information for
  // the target implementation.
  MachineFunctionInfo *MFInfo;

  // Keep track of objects allocated on the stack.
  MachineFrameInfo *FrameInfo;

  // Keep track of constants which are spilled to memory
  MachineConstantPool *ConstantPool;
  
  // Keep track of jump tables for switch instructions
  MachineJumpTableInfo *JumpTableInfo;

  // Function-level unique numbering for MachineBasicBlocks.  When a
  // MachineBasicBlock is inserted into a MachineFunction is it automatically
  // numbered and this vector keeps track of the mapping from ID's to MBB's.
  std::vector<MachineBasicBlock*> MBBNumbering;

  // Pool-allocate MachineFunction-lifetime and IR objects.
  BumpPtrAllocator Allocator;

  // Allocation management for instructions in function.
  Recycler<MachineInstr> InstructionRecycler;

  // Allocation management for operand arrays on instructions.
  ArrayRecycler<MachineOperand> OperandRecycler;

  // Allocation management for basic blocks in function.
  Recycler<MachineBasicBlock> BasicBlockRecycler;

  // List of machine basic blocks in function
  typedef ilist<MachineBasicBlock> BasicBlockListType;
  BasicBlockListType BasicBlocks;

  /// FunctionNumber - This provides a unique ID for each function emitted in
  /// this translation unit.
  ///
  unsigned FunctionNumber;
  
  /// Alignment - The alignment of the function.
  unsigned Alignment;

  /// ExposesReturnsTwice - True if the function calls setjmp or related
  /// functions with attribute "returns twice", but doesn't have
  /// the attribute itself.
  /// This is used to limit optimizations which cannot reason
  /// about the control flow of such functions.
  bool ExposesReturnsTwice;

  /// True if the function includes any inline assembly.
  bool HasInlineAsm;

  MachineFunction(const MachineFunction &) = delete;
  void operator=(const MachineFunction&) = delete;
public:
  MachineFunction(const Function *Fn, const TargetMachine &TM,
                  unsigned FunctionNum, MachineModuleInfo &MMI);
  ~MachineFunction();

  MachineModuleInfo &getMMI() const { return MMI; }
  MCContext &getContext() const { return Ctx; }

  /// Return the DataLayout attached to the Module associated to this MF.
  const DataLayout &getDataLayout() const;

  /// getFunction - Return the LLVM function that this machine code represents
  ///
  const Function *getFunction() const { return Fn; }

  /// getName - Return the name of the corresponding LLVM function.
  ///
  StringRef getName() const;

  /// getFunctionNumber - Return a unique ID for the current function.
  ///
  unsigned getFunctionNumber() const { return FunctionNumber; }

  /// getTarget - Return the target machine this machine code is compiled with
  ///
  const TargetMachine &getTarget() const { return Target; }

  /// getSubtarget - Return the subtarget for which this machine code is being
  /// compiled.
  const TargetSubtargetInfo &getSubtarget() const { return *STI; }
  void setSubtarget(const TargetSubtargetInfo *ST) { STI = ST; }

  /// getSubtarget - This method returns a pointer to the specified type of
  /// TargetSubtargetInfo.  In debug builds, it verifies that the object being
  /// returned is of the correct type.
  template<typename STC> const STC &getSubtarget() const {
    return *static_cast<const STC *>(STI);
  }

  /// getRegInfo - Return information about the registers currently in use.
  ///
  MachineRegisterInfo &getRegInfo() { return *RegInfo; }
  const MachineRegisterInfo &getRegInfo() const { return *RegInfo; }

  /// getFrameInfo - Return the frame info object for the current function.
  /// This object contains information about objects allocated on the stack
  /// frame of the current function in an abstract way.
  ///
  MachineFrameInfo *getFrameInfo() { return FrameInfo; }
  const MachineFrameInfo *getFrameInfo() const { return FrameInfo; }

  /// getJumpTableInfo - Return the jump table info object for the current 
  /// function.  This object contains information about jump tables in the
  /// current function.  If the current function has no jump tables, this will
  /// return null.
  const MachineJumpTableInfo *getJumpTableInfo() const { return JumpTableInfo; }
  MachineJumpTableInfo *getJumpTableInfo() { return JumpTableInfo; }

  /// getOrCreateJumpTableInfo - Get the JumpTableInfo for this function, if it
  /// does already exist, allocate one.
  MachineJumpTableInfo *getOrCreateJumpTableInfo(unsigned JTEntryKind);

  
  /// getConstantPool - Return the constant pool object for the current
  /// function.
  ///
  MachineConstantPool *getConstantPool() { return ConstantPool; }
  const MachineConstantPool *getConstantPool() const { return ConstantPool; }

  /// getAlignment - Return the alignment (log2, not bytes) of the function.
  ///
  unsigned getAlignment() const { return Alignment; }

  /// setAlignment - Set the alignment (log2, not bytes) of the function.
  ///
  void setAlignment(unsigned A) { Alignment = A; }

  /// ensureAlignment - Make sure the function is at least 1 << A bytes aligned.
  void ensureAlignment(unsigned A) {
    if (Alignment < A) Alignment = A;
  }

  /// exposesReturnsTwice - Returns true if the function calls setjmp or
  /// any other similar functions with attribute "returns twice" without
  /// having the attribute itself.
  bool exposesReturnsTwice() const {
    return ExposesReturnsTwice;
  }

  /// setCallsSetJmp - Set a flag that indicates if there's a call to
  /// a "returns twice" function.
  void setExposesReturnsTwice(bool B) {
    ExposesReturnsTwice = B;
  }

  /// Returns true if the function contains any inline assembly.
  bool hasInlineAsm() const {
    return HasInlineAsm;
  }

  /// Set a flag that indicates that the function contains inline assembly.
  void setHasInlineAsm(bool B) {
    HasInlineAsm = B;
  }

  /// getInfo - Keep track of various per-function pieces of information for
  /// backends that would like to do so.
  ///
  template<typename Ty>
  Ty *getInfo() {
    if (!MFInfo)
      MFInfo = Ty::template create<Ty>(Allocator, *this);
    return static_cast<Ty*>(MFInfo);
  }

  template<typename Ty>
  const Ty *getInfo() const {
     return const_cast<MachineFunction*>(this)->getInfo<Ty>();
  }

  /// getBlockNumbered - MachineBasicBlocks are automatically numbered when they
  /// are inserted into the machine function.  The block number for a machine
  /// basic block can be found by using the MBB::getBlockNumber method, this
  /// method provides the inverse mapping.
  ///
  MachineBasicBlock *getBlockNumbered(unsigned N) const {
    assert(N < MBBNumbering.size() && "Illegal block number");
    assert(MBBNumbering[N] && "Block was removed from the machine function!");
    return MBBNumbering[N];
  }

  /// Should we be emitting segmented stack stuff for the function
  bool shouldSplitStack();

  /// getNumBlockIDs - Return the number of MBB ID's allocated.
  ///
  unsigned getNumBlockIDs() const { return (unsigned)MBBNumbering.size(); }
  
  /// RenumberBlocks - This discards all of the MachineBasicBlock numbers and
  /// recomputes them.  This guarantees that the MBB numbers are sequential,
  /// dense, and match the ordering of the blocks within the function.  If a
  /// specific MachineBasicBlock is specified, only that block and those after
  /// it are renumbered.
  void RenumberBlocks(MachineBasicBlock *MBBFrom = nullptr);
  
  /// print - Print out the MachineFunction in a format suitable for debugging
  /// to the specified stream.
  ///
  void print(raw_ostream &OS, SlotIndexes* = nullptr) const;

  /// viewCFG - This function is meant for use from the debugger.  You can just
  /// say 'call F->viewCFG()' and a ghostview window should pop up from the
  /// program, displaying the CFG of the current function with the code for each
  /// basic block inside.  This depends on there being a 'dot' and 'gv' program
  /// in your path.
  ///
  void viewCFG() const;

  /// viewCFGOnly - This function is meant for use from the debugger.  It works
  /// just like viewCFG, but it does not include the contents of basic blocks
  /// into the nodes, just the label.  If you are only interested in the CFG
  /// this can make the graph smaller.
  ///
  void viewCFGOnly() const;

  /// dump - Print the current MachineFunction to cerr, useful for debugger use.
  ///
  void dump() const;

  /// verify - Run the current MachineFunction through the machine code
  /// verifier, useful for debugger use.
  void verify(Pass *p = nullptr, const char *Banner = nullptr) const;

  // Provide accessors for the MachineBasicBlock list...
  typedef BasicBlockListType::iterator iterator;
  typedef BasicBlockListType::const_iterator const_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef std::reverse_iterator<iterator>             reverse_iterator;

  /// addLiveIn - Add the specified physical register as a live-in value and
  /// create a corresponding virtual register for it.
  unsigned addLiveIn(unsigned PReg, const TargetRegisterClass *RC);

  //===--------------------------------------------------------------------===//
  // BasicBlock accessor functions.
  //
  iterator                 begin()       { return BasicBlocks.begin(); }
  const_iterator           begin() const { return BasicBlocks.begin(); }
  iterator                 end  ()       { return BasicBlocks.end();   }
  const_iterator           end  () const { return BasicBlocks.end();   }

  reverse_iterator        rbegin()       { return BasicBlocks.rbegin(); }
  const_reverse_iterator  rbegin() const { return BasicBlocks.rbegin(); }
  reverse_iterator        rend  ()       { return BasicBlocks.rend();   }
  const_reverse_iterator  rend  () const { return BasicBlocks.rend();   }

  unsigned                  size() const { return (unsigned)BasicBlocks.size();}
  bool                     empty() const { return BasicBlocks.empty(); }
  const MachineBasicBlock &front() const { return BasicBlocks.front(); }
        MachineBasicBlock &front()       { return BasicBlocks.front(); }
  const MachineBasicBlock & back() const { return BasicBlocks.back(); }
        MachineBasicBlock & back()       { return BasicBlocks.back(); }

  void push_back (MachineBasicBlock *MBB) { BasicBlocks.push_back (MBB); }
  void push_front(MachineBasicBlock *MBB) { BasicBlocks.push_front(MBB); }
  void insert(iterator MBBI, MachineBasicBlock *MBB) {
    BasicBlocks.insert(MBBI, MBB);
  }
  void splice(iterator InsertPt, iterator MBBI) {
    BasicBlocks.splice(InsertPt, BasicBlocks, MBBI);
  }
  void splice(iterator InsertPt, iterator MBBI, iterator MBBE) {
    BasicBlocks.splice(InsertPt, BasicBlocks, MBBI, MBBE);
  }

  void remove(iterator MBBI) {
    BasicBlocks.remove(MBBI);
  }
  void erase(iterator MBBI) {
    BasicBlocks.erase(MBBI);
  }

  //===--------------------------------------------------------------------===//
  // Internal functions used to automatically number MachineBasicBlocks
  //

  /// \brief Adds the MBB to the internal numbering. Returns the unique number
  /// assigned to the MBB.
  ///
  unsigned addToMBBNumbering(MachineBasicBlock *MBB) {
    MBBNumbering.push_back(MBB);
    return (unsigned)MBBNumbering.size()-1;
  }

  /// removeFromMBBNumbering - Remove the specific machine basic block from our
  /// tracker, this is only really to be used by the MachineBasicBlock
  /// implementation.
  void removeFromMBBNumbering(unsigned N) {
    assert(N < MBBNumbering.size() && "Illegal basic block #");
    MBBNumbering[N] = nullptr;
  }

  /// CreateMachineInstr - Allocate a new MachineInstr. Use this instead
  /// of `new MachineInstr'.
  ///
  MachineInstr *CreateMachineInstr(const MCInstrDesc &MCID,
                                   DebugLoc DL,
                                   bool NoImp = false);

  /// CloneMachineInstr - Create a new MachineInstr which is a copy of the
  /// 'Orig' instruction, identical in all ways except the instruction
  /// has no parent, prev, or next.
  ///
  /// See also TargetInstrInfo::duplicate() for target-specific fixes to cloned
  /// instructions.
  MachineInstr *CloneMachineInstr(const MachineInstr *Orig);

  /// DeleteMachineInstr - Delete the given MachineInstr.
  ///
  void DeleteMachineInstr(MachineInstr *MI);

  /// CreateMachineBasicBlock - Allocate a new MachineBasicBlock. Use this
  /// instead of `new MachineBasicBlock'.
  ///
  MachineBasicBlock *CreateMachineBasicBlock(const BasicBlock *bb = nullptr);

  /// DeleteMachineBasicBlock - Delete the given MachineBasicBlock.
  ///
  void DeleteMachineBasicBlock(MachineBasicBlock *MBB);

  /// getMachineMemOperand - Allocate a new MachineMemOperand.
  /// MachineMemOperands are owned by the MachineFunction and need not be
  /// explicitly deallocated.
  MachineMemOperand *getMachineMemOperand(MachinePointerInfo PtrInfo,
                                          unsigned f, uint64_t s,
                                          unsigned base_alignment,
                                          const AAMDNodes &AAInfo = AAMDNodes(),
                                          const MDNode *Ranges = nullptr);
  
  /// getMachineMemOperand - Allocate a new MachineMemOperand by copying
  /// an existing one, adjusting by an offset and using the given size.
  /// MachineMemOperands are owned by the MachineFunction and need not be
  /// explicitly deallocated.
  MachineMemOperand *getMachineMemOperand(const MachineMemOperand *MMO,
                                          int64_t Offset, uint64_t Size);

  typedef ArrayRecycler<MachineOperand>::Capacity OperandCapacity;

  /// Allocate an array of MachineOperands. This is only intended for use by
  /// internal MachineInstr functions.
  MachineOperand *allocateOperandArray(OperandCapacity Cap) {
    return OperandRecycler.allocate(Cap, Allocator);
  }

  /// Dellocate an array of MachineOperands and recycle the memory. This is
  /// only intended for use by internal MachineInstr functions.
  /// Cap must be the same capacity that was used to allocate the array.
  void deallocateOperandArray(OperandCapacity Cap, MachineOperand *Array) {
    OperandRecycler.deallocate(Cap, Array);
  }

  /// \brief Allocate and initialize a register mask with @p NumRegister bits.
  uint32_t *allocateRegisterMask(unsigned NumRegister) {
    unsigned Size = (NumRegister + 31) / 32;
    uint32_t *Mask = Allocator.Allocate<uint32_t>(Size);
    for (unsigned i = 0; i != Size; ++i)
      Mask[i] = 0;
    return Mask;
  }

  /// allocateMemRefsArray - Allocate an array to hold MachineMemOperand
  /// pointers.  This array is owned by the MachineFunction.
  MachineInstr::mmo_iterator allocateMemRefsArray(unsigned long Num);

  /// extractLoadMemRefs - Allocate an array and populate it with just the
  /// load information from the given MachineMemOperand sequence.
  std::pair<MachineInstr::mmo_iterator,
            MachineInstr::mmo_iterator>
    extractLoadMemRefs(MachineInstr::mmo_iterator Begin,
                       MachineInstr::mmo_iterator End);

  /// extractStoreMemRefs - Allocate an array and populate it with just the
  /// store information from the given MachineMemOperand sequence.
  std::pair<MachineInstr::mmo_iterator,
            MachineInstr::mmo_iterator>
    extractStoreMemRefs(MachineInstr::mmo_iterator Begin,
                        MachineInstr::mmo_iterator End);

  //===--------------------------------------------------------------------===//
  // Label Manipulation.
  //
  
  /// getJTISymbol - Return the MCSymbol for the specified non-empty jump table.
  /// If isLinkerPrivate is specified, an 'l' label is returned, otherwise a
  /// normal 'L' label is returned.
  MCSymbol *getJTISymbol(unsigned JTI, MCContext &Ctx, 
                         bool isLinkerPrivate = false) const;
  
  /// getPICBaseSymbol - Return a function-local symbol to represent the PIC
  /// base.
  MCSymbol *getPICBaseSymbol() const;
};

//===--------------------------------------------------------------------===//
// GraphTraits specializations for function basic block graphs (CFGs)
//===--------------------------------------------------------------------===//

// Provide specializations of GraphTraits to be able to treat a
// machine function as a graph of machine basic blocks... these are
// the same as the machine basic block iterators, except that the root
// node is implicitly the first node of the function.
//
template <> struct GraphTraits<MachineFunction*> :
  public GraphTraits<MachineBasicBlock*> {
  static NodeType *getEntryNode(MachineFunction *F) {
    return &F->front();
  }

  // nodes_iterator/begin/end - Allow iteration over all nodes in the graph
  typedef MachineFunction::iterator nodes_iterator;
  static nodes_iterator nodes_begin(MachineFunction *F) { return F->begin(); }
  static nodes_iterator nodes_end  (MachineFunction *F) { return F->end(); }
  static unsigned       size       (MachineFunction *F) { return F->size(); }
};
template <> struct GraphTraits<const MachineFunction*> :
  public GraphTraits<const MachineBasicBlock*> {
  static NodeType *getEntryNode(const MachineFunction *F) {
    return &F->front();
  }

  // nodes_iterator/begin/end - Allow iteration over all nodes in the graph
  typedef MachineFunction::const_iterator nodes_iterator;
  static nodes_iterator nodes_begin(const MachineFunction *F) {
    return F->begin();
  }
  static nodes_iterator nodes_end  (const MachineFunction *F) {
    return F->end();
  }
  static unsigned       size       (const MachineFunction *F)  {
    return F->size();
  }
};


// Provide specializations of GraphTraits to be able to treat a function as a
// graph of basic blocks... and to walk it in inverse order.  Inverse order for
// a function is considered to be when traversing the predecessor edges of a BB
// instead of the successor edges.
//
template <> struct GraphTraits<Inverse<MachineFunction*> > :
  public GraphTraits<Inverse<MachineBasicBlock*> > {
  static NodeType *getEntryNode(Inverse<MachineFunction*> G) {
    return &G.Graph->front();
  }
};
template <> struct GraphTraits<Inverse<const MachineFunction*> > :
  public GraphTraits<Inverse<const MachineBasicBlock*> > {
  static NodeType *getEntryNode(Inverse<const MachineFunction *> G) {
    return &G.Graph->front();
  }
};

} // End llvm namespace

#endif
