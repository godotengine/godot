//===------- llvm/CodeGen/ScheduleDAG.h - Common Base Class------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ScheduleDAG class, which is used as the common
// base class for instruction schedulers. This encapsulates the scheduling DAG,
// which is shared between SelectionDAG and MachineInstr scheduling.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SCHEDULEDAG_H
#define LLVM_CODEGEN_SCHEDULEDAG_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/TargetLowering.h"

namespace llvm {
  class AliasAnalysis;
  class SUnit;
  class MachineConstantPool;
  class MachineFunction;
  class MachineRegisterInfo;
  class MachineInstr;
  struct MCSchedClassDesc;
  class TargetRegisterInfo;
  class ScheduleDAG;
  class SDNode;
  class TargetInstrInfo;
  class MCInstrDesc;
  class TargetMachine;
  class TargetRegisterClass;
  template<class Graph> class GraphWriter;

  /// SDep - Scheduling dependency. This represents one direction of an
  /// edge in the scheduling DAG.
  class SDep {
  public:
    /// Kind - These are the different kinds of scheduling dependencies.
    enum Kind {
      Data,        ///< Regular data dependence (aka true-dependence).
      Anti,        ///< A register anti-dependedence (aka WAR).
      Output,      ///< A register output-dependence (aka WAW).
      Order        ///< Any other ordering dependency.
    };

    // Strong dependencies must be respected by the scheduler. Artificial
    // dependencies may be removed only if they are redundant with another
    // strong depedence.
    //
    // Weak dependencies may be violated by the scheduling strategy, but only if
    // the strategy can prove it is correct to do so.
    //
    // Strong OrderKinds must occur before "Weak".
    // Weak OrderKinds must occur after "Weak".
    enum OrderKind {
      Barrier,      ///< An unknown scheduling barrier.
      MayAliasMem,  ///< Nonvolatile load/Store instructions that may alias.
      MustAliasMem, ///< Nonvolatile load/Store instructions that must alias.
      Artificial,   ///< Arbitrary strong DAG edge (no real dependence).
      Weak,         ///< Arbitrary weak DAG edge.
      Cluster       ///< Weak DAG edge linking a chain of clustered instrs.
    };

  private:
    /// Dep - A pointer to the depending/depended-on SUnit, and an enum
    /// indicating the kind of the dependency.
    PointerIntPair<SUnit *, 2, Kind> Dep;

    /// Contents - A union discriminated by the dependence kind.
    union {
      /// Reg - For Data, Anti, and Output dependencies, the associated
      /// register. For Data dependencies that don't currently have a register
      /// assigned, this is set to zero.
      unsigned Reg;

      /// Order - Additional information about Order dependencies.
      unsigned OrdKind; // enum OrderKind
    } Contents;

    /// Latency - The time associated with this edge. Often this is just
    /// the value of the Latency field of the predecessor, however advanced
    /// models may provide additional information about specific edges.
    unsigned Latency;

  public:
    /// SDep - Construct a null SDep. This is only for use by container
    /// classes which require default constructors. SUnits may not
    /// have null SDep edges.
    SDep() : Dep(nullptr, Data) {}

    /// SDep - Construct an SDep with the specified values.
    SDep(SUnit *S, Kind kind, unsigned Reg)
      : Dep(S, kind), Contents() {
      switch (kind) {
      default:
        llvm_unreachable("Reg given for non-register dependence!");
      case Anti:
      case Output:
        assert(Reg != 0 &&
               "SDep::Anti and SDep::Output must use a non-zero Reg!");
        Contents.Reg = Reg;
        Latency = 0;
        break;
      case Data:
        Contents.Reg = Reg;
        Latency = 1;
        break;
      }
    }
    SDep(SUnit *S, OrderKind kind)
      : Dep(S, Order), Contents(), Latency(0) {
      Contents.OrdKind = kind;
    }

    /// Return true if the specified SDep is equivalent except for latency.
    bool overlaps(const SDep &Other) const {
      if (Dep != Other.Dep) return false;
      switch (Dep.getInt()) {
      case Data:
      case Anti:
      case Output:
        return Contents.Reg == Other.Contents.Reg;
      case Order:
        return Contents.OrdKind == Other.Contents.OrdKind;
      }
      llvm_unreachable("Invalid dependency kind!");
    }

    bool operator==(const SDep &Other) const {
      return overlaps(Other) && Latency == Other.Latency;
    }

    bool operator!=(const SDep &Other) const {
      return !operator==(Other);
    }

    /// getLatency - Return the latency value for this edge, which roughly
    /// means the minimum number of cycles that must elapse between the
    /// predecessor and the successor, given that they have this edge
    /// between them.
    unsigned getLatency() const {
      return Latency;
    }

    /// setLatency - Set the latency for this edge.
    void setLatency(unsigned Lat) {
      Latency = Lat;
    }

    //// getSUnit - Return the SUnit to which this edge points.
    SUnit *getSUnit() const {
      return Dep.getPointer();
    }

    //// setSUnit - Assign the SUnit to which this edge points.
    void setSUnit(SUnit *SU) {
      Dep.setPointer(SU);
    }

    /// getKind - Return an enum value representing the kind of the dependence.
    Kind getKind() const {
      return Dep.getInt();
    }

    /// isCtrl - Shorthand for getKind() != SDep::Data.
    bool isCtrl() const {
      return getKind() != Data;
    }

    /// isNormalMemory - Test if this is an Order dependence between two
    /// memory accesses where both sides of the dependence access memory
    /// in non-volatile and fully modeled ways.
    bool isNormalMemory() const {
      return getKind() == Order && (Contents.OrdKind == MayAliasMem
                                    || Contents.OrdKind == MustAliasMem);
    }

    /// isBarrier - Test if this is an Order dependence that is marked
    /// as a barrier.
    bool isBarrier() const {
      return getKind() == Order && Contents.OrdKind == Barrier;
    }

    /// isNormalMemoryOrBarrier - Test if this is could be any kind of memory
    /// dependence.
    bool isNormalMemoryOrBarrier() const {
      return (isNormalMemory() || isBarrier());
    }

    /// isMustAlias - Test if this is an Order dependence that is marked
    /// as "must alias", meaning that the SUnits at either end of the edge
    /// have a memory dependence on a known memory location.
    bool isMustAlias() const {
      return getKind() == Order && Contents.OrdKind == MustAliasMem;
    }

    /// isWeak - Test if this a weak dependence. Weak dependencies are
    /// considered DAG edges for height computation and other heuristics, but do
    /// not force ordering. Breaking a weak edge may require the scheduler to
    /// compensate, for example by inserting a copy.
    bool isWeak() const {
      return getKind() == Order && Contents.OrdKind >= Weak;
    }

    /// isArtificial - Test if this is an Order dependence that is marked
    /// as "artificial", meaning it isn't necessary for correctness.
    bool isArtificial() const {
      return getKind() == Order && Contents.OrdKind == Artificial;
    }

    /// isCluster - Test if this is an Order dependence that is marked
    /// as "cluster", meaning it is artificial and wants to be adjacent.
    bool isCluster() const {
      return getKind() == Order && Contents.OrdKind == Cluster;
    }

    /// isAssignedRegDep - Test if this is a Data dependence that is
    /// associated with a register.
    bool isAssignedRegDep() const {
      return getKind() == Data && Contents.Reg != 0;
    }

    /// getReg - Return the register associated with this edge. This is
    /// only valid on Data, Anti, and Output edges. On Data edges, this
    /// value may be zero, meaning there is no associated register.
    unsigned getReg() const {
      assert((getKind() == Data || getKind() == Anti || getKind() == Output) &&
             "getReg called on non-register dependence edge!");
      return Contents.Reg;
    }

    /// setReg - Assign the associated register for this edge. This is
    /// only valid on Data, Anti, and Output edges. On Anti and Output
    /// edges, this value must not be zero. On Data edges, the value may
    /// be zero, which would mean that no specific register is associated
    /// with this edge.
    void setReg(unsigned Reg) {
      assert((getKind() == Data || getKind() == Anti || getKind() == Output) &&
             "setReg called on non-register dependence edge!");
      assert((getKind() != Anti || Reg != 0) &&
             "SDep::Anti edge cannot use the zero register!");
      assert((getKind() != Output || Reg != 0) &&
             "SDep::Output edge cannot use the zero register!");
      Contents.Reg = Reg;
    }
  };

  template <>
  struct isPodLike<SDep> { static const bool value = true; };

  /// SUnit - Scheduling unit. This is a node in the scheduling DAG.
  class SUnit {
  private:
    enum : unsigned { BoundaryID = ~0u };

    SDNode *Node;                       // Representative node.
    MachineInstr *Instr;                // Alternatively, a MachineInstr.
  public:
    SUnit *OrigNode;                    // If not this, the node from which
                                        // this node was cloned.
                                        // (SD scheduling only)

    const MCSchedClassDesc *SchedClass; // NULL or resolved SchedClass.

    // Preds/Succs - The SUnits before/after us in the graph.
    SmallVector<SDep, 4> Preds;  // All sunit predecessors.
    SmallVector<SDep, 4> Succs;  // All sunit successors.

    typedef SmallVectorImpl<SDep>::iterator pred_iterator;
    typedef SmallVectorImpl<SDep>::iterator succ_iterator;
    typedef SmallVectorImpl<SDep>::const_iterator const_pred_iterator;
    typedef SmallVectorImpl<SDep>::const_iterator const_succ_iterator;

    unsigned NodeNum;                   // Entry # of node in the node vector.
    unsigned NodeQueueId;               // Queue id of node.
    unsigned NumPreds;                  // # of SDep::Data preds.
    unsigned NumSuccs;                  // # of SDep::Data sucss.
    unsigned NumPredsLeft;              // # of preds not scheduled.
    unsigned NumSuccsLeft;              // # of succs not scheduled.
    unsigned WeakPredsLeft;             // # of weak preds not scheduled.
    unsigned WeakSuccsLeft;             // # of weak succs not scheduled.
    unsigned short NumRegDefsLeft;      // # of reg defs with no scheduled use.
    unsigned short Latency;             // Node latency.
    bool isVRegCycle      : 1;          // May use and def the same vreg.
    bool isCall           : 1;          // Is a function call.
    bool isCallOp         : 1;          // Is a function call operand.
    bool isTwoAddress     : 1;          // Is a two-address instruction.
    bool isCommutable     : 1;          // Is a commutable instruction.
    bool hasPhysRegUses   : 1;          // Has physreg uses.
    bool hasPhysRegDefs   : 1;          // Has physreg defs that are being used.
    bool hasPhysRegClobbers : 1;        // Has any physreg defs, used or not.
    bool isPending        : 1;          // True once pending.
    bool isAvailable      : 1;          // True once available.
    bool isScheduled      : 1;          // True once scheduled.
    bool isScheduleHigh   : 1;          // True if preferable to schedule high.
    bool isScheduleLow    : 1;          // True if preferable to schedule low.
    bool isCloned         : 1;          // True if this node has been cloned.
    bool isUnbuffered     : 1;          // Uses an unbuffered resource.
    bool hasReservedResource : 1;       // Uses a reserved resource.
    Sched::Preference SchedulingPref;   // Scheduling preference.

  private:
    bool isDepthCurrent   : 1;          // True if Depth is current.
    bool isHeightCurrent  : 1;          // True if Height is current.
    unsigned Depth;                     // Node depth.
    unsigned Height;                    // Node height.
  public:
    unsigned TopReadyCycle; // Cycle relative to start when node is ready.
    unsigned BotReadyCycle; // Cycle relative to end when node is ready.

    const TargetRegisterClass *CopyDstRC; // Is a special copy node if not null.
    const TargetRegisterClass *CopySrcRC;

    /// SUnit - Construct an SUnit for pre-regalloc scheduling to represent
    /// an SDNode and any nodes flagged to it.
    SUnit(SDNode *node, unsigned nodenum)
      : Node(node), Instr(nullptr), OrigNode(nullptr), SchedClass(nullptr),
        NodeNum(nodenum), NodeQueueId(0), NumPreds(0), NumSuccs(0),
        NumPredsLeft(0), NumSuccsLeft(0), WeakPredsLeft(0), WeakSuccsLeft(0),
        NumRegDefsLeft(0), Latency(0), isVRegCycle(false), isCall(false),
        isCallOp(false), isTwoAddress(false), isCommutable(false),
        hasPhysRegUses(false), hasPhysRegDefs(false), hasPhysRegClobbers(false),
        isPending(false), isAvailable(false), isScheduled(false),
        isScheduleHigh(false), isScheduleLow(false), isCloned(false),
        isUnbuffered(false), hasReservedResource(false),
        SchedulingPref(Sched::None), isDepthCurrent(false),
        isHeightCurrent(false), Depth(0), Height(0), TopReadyCycle(0),
        BotReadyCycle(0), CopyDstRC(nullptr), CopySrcRC(nullptr) {}

    /// SUnit - Construct an SUnit for post-regalloc scheduling to represent
    /// a MachineInstr.
    SUnit(MachineInstr *instr, unsigned nodenum)
      : Node(nullptr), Instr(instr), OrigNode(nullptr), SchedClass(nullptr),
        NodeNum(nodenum), NodeQueueId(0), NumPreds(0), NumSuccs(0),
        NumPredsLeft(0), NumSuccsLeft(0), WeakPredsLeft(0), WeakSuccsLeft(0),
        NumRegDefsLeft(0), Latency(0), isVRegCycle(false), isCall(false),
        isCallOp(false), isTwoAddress(false), isCommutable(false),
        hasPhysRegUses(false), hasPhysRegDefs(false), hasPhysRegClobbers(false),
        isPending(false), isAvailable(false), isScheduled(false),
        isScheduleHigh(false), isScheduleLow(false), isCloned(false),
        isUnbuffered(false), hasReservedResource(false),
        SchedulingPref(Sched::None), isDepthCurrent(false),
        isHeightCurrent(false), Depth(0), Height(0), TopReadyCycle(0),
        BotReadyCycle(0), CopyDstRC(nullptr), CopySrcRC(nullptr) {}

    /// SUnit - Construct a placeholder SUnit.
    SUnit()
      : Node(nullptr), Instr(nullptr), OrigNode(nullptr), SchedClass(nullptr),
        NodeNum(BoundaryID), NodeQueueId(0), NumPreds(0), NumSuccs(0),
        NumPredsLeft(0), NumSuccsLeft(0), WeakPredsLeft(0), WeakSuccsLeft(0),
        NumRegDefsLeft(0), Latency(0), isVRegCycle(false), isCall(false),
        isCallOp(false), isTwoAddress(false), isCommutable(false),
        hasPhysRegUses(false), hasPhysRegDefs(false), hasPhysRegClobbers(false),
        isPending(false), isAvailable(false), isScheduled(false),
        isScheduleHigh(false), isScheduleLow(false), isCloned(false),
        isUnbuffered(false), hasReservedResource(false),
        SchedulingPref(Sched::None), isDepthCurrent(false),
        isHeightCurrent(false), Depth(0), Height(0), TopReadyCycle(0),
        BotReadyCycle(0), CopyDstRC(nullptr), CopySrcRC(nullptr) {}

    /// \brief Boundary nodes are placeholders for the boundary of the
    /// scheduling region.
    ///
    /// BoundaryNodes can have DAG edges, including Data edges, but they do not
    /// correspond to schedulable entities (e.g. instructions) and do not have a
    /// valid ID. Consequently, always check for boundary nodes before accessing
    /// an assoicative data structure keyed on node ID.
    bool isBoundaryNode() const { return NodeNum == BoundaryID; };

    /// setNode - Assign the representative SDNode for this SUnit.
    /// This may be used during pre-regalloc scheduling.
    void setNode(SDNode *N) {
      assert(!Instr && "Setting SDNode of SUnit with MachineInstr!");
      Node = N;
    }

    /// getNode - Return the representative SDNode for this SUnit.
    /// This may be used during pre-regalloc scheduling.
    SDNode *getNode() const {
      assert(!Instr && "Reading SDNode of SUnit with MachineInstr!");
      return Node;
    }

    /// isInstr - Return true if this SUnit refers to a machine instruction as
    /// opposed to an SDNode.
    bool isInstr() const { return Instr; }

    /// setInstr - Assign the instruction for the SUnit.
    /// This may be used during post-regalloc scheduling.
    void setInstr(MachineInstr *MI) {
      assert(!Node && "Setting MachineInstr of SUnit with SDNode!");
      Instr = MI;
    }

    /// getInstr - Return the representative MachineInstr for this SUnit.
    /// This may be used during post-regalloc scheduling.
    MachineInstr *getInstr() const {
      assert(!Node && "Reading MachineInstr of SUnit with SDNode!");
      return Instr;
    }

    /// addPred - This adds the specified edge as a pred of the current node if
    /// not already.  It also adds the current node as a successor of the
    /// specified node.
    bool addPred(const SDep &D, bool Required = true);

    /// removePred - This removes the specified edge as a pred of the current
    /// node if it exists.  It also removes the current node as a successor of
    /// the specified node.
    void removePred(const SDep &D);

    /// getDepth - Return the depth of this node, which is the length of the
    /// maximum path up to any node which has no predecessors.
    unsigned getDepth() const {
      if (!isDepthCurrent)
        const_cast<SUnit *>(this)->ComputeDepth();
      return Depth;
    }

    /// getHeight - Return the height of this node, which is the length of the
    /// maximum path down to any node which has no successors.
    unsigned getHeight() const {
      if (!isHeightCurrent)
        const_cast<SUnit *>(this)->ComputeHeight();
      return Height;
    }

    /// setDepthToAtLeast - If NewDepth is greater than this node's
    /// depth value, set it to be the new depth value. This also
    /// recursively marks successor nodes dirty.
    void setDepthToAtLeast(unsigned NewDepth);

    /// setDepthToAtLeast - If NewDepth is greater than this node's
    /// depth value, set it to be the new height value. This also
    /// recursively marks predecessor nodes dirty.
    void setHeightToAtLeast(unsigned NewHeight);

    /// setDepthDirty - Set a flag in this node to indicate that its
    /// stored Depth value will require recomputation the next time
    /// getDepth() is called.
    void setDepthDirty();

    /// setHeightDirty - Set a flag in this node to indicate that its
    /// stored Height value will require recomputation the next time
    /// getHeight() is called.
    void setHeightDirty();

    /// isPred - Test if node N is a predecessor of this node.
    bool isPred(SUnit *N) {
      for (unsigned i = 0, e = (unsigned)Preds.size(); i != e; ++i)
        if (Preds[i].getSUnit() == N)
          return true;
      return false;
    }

    /// isSucc - Test if node N is a successor of this node.
    bool isSucc(SUnit *N) {
      for (unsigned i = 0, e = (unsigned)Succs.size(); i != e; ++i)
        if (Succs[i].getSUnit() == N)
          return true;
      return false;
    }

    bool isTopReady() const {
      return NumPredsLeft == 0;
    }
    bool isBottomReady() const {
      return NumSuccsLeft == 0;
    }

    /// \brief Order this node's predecessor edges such that the critical path
    /// edge occurs first.
    void biasCriticalPath();

    void dump(const ScheduleDAG *G) const;
    void dumpAll(const ScheduleDAG *G) const;
    void print(raw_ostream &O, const ScheduleDAG *G) const;

  private:
    void ComputeDepth();
    void ComputeHeight();
  };

  //===--------------------------------------------------------------------===//
  /// SchedulingPriorityQueue - This interface is used to plug different
  /// priorities computation algorithms into the list scheduler. It implements
  /// the interface of a standard priority queue, where nodes are inserted in
  /// arbitrary order and returned in priority order.  The computation of the
  /// priority and the representation of the queue are totally up to the
  /// implementation to decide.
  ///
  class SchedulingPriorityQueue {
    virtual void anchor();
    unsigned CurCycle;
    bool HasReadyFilter;
  public:
    SchedulingPriorityQueue(bool rf = false):
      CurCycle(0), HasReadyFilter(rf) {}
    virtual ~SchedulingPriorityQueue() {}

    virtual bool isBottomUp() const = 0;

    virtual void initNodes(std::vector<SUnit> &SUnits) = 0;
    virtual void addNode(const SUnit *SU) = 0;
    virtual void updateNode(const SUnit *SU) = 0;
    virtual void releaseState() = 0;

    virtual bool empty() const = 0;

    bool hasReadyFilter() const { return HasReadyFilter; }

    virtual bool tracksRegPressure() const { return false; }

    virtual bool isReady(SUnit *) const {
      assert(!HasReadyFilter && "The ready filter must override isReady()");
      return true;
    }
    virtual void push(SUnit *U) = 0;

    void push_all(const std::vector<SUnit *> &Nodes) {
      for (std::vector<SUnit *>::const_iterator I = Nodes.begin(),
           E = Nodes.end(); I != E; ++I)
        push(*I);
    }

    virtual SUnit *pop() = 0;

    virtual void remove(SUnit *SU) = 0;

    virtual void dump(ScheduleDAG *) const {}

    /// scheduledNode - As each node is scheduled, this method is invoked.  This
    /// allows the priority function to adjust the priority of related
    /// unscheduled nodes, for example.
    ///
    virtual void scheduledNode(SUnit *) {}

    virtual void unscheduledNode(SUnit *) {}

    void setCurCycle(unsigned Cycle) {
      CurCycle = Cycle;
    }

    unsigned getCurCycle() const {
      return CurCycle;
    }
  };

  class ScheduleDAG {
  public:
    const TargetMachine &TM;              // Target processor
    const TargetInstrInfo *TII;           // Target instruction information
    const TargetRegisterInfo *TRI;        // Target processor register info
    MachineFunction &MF;                  // Machine function
    MachineRegisterInfo &MRI;             // Virtual/real register map
    std::vector<SUnit> SUnits;            // The scheduling units.
    SUnit EntrySU;                        // Special node for the region entry.
    SUnit ExitSU;                         // Special node for the region exit.

#ifdef NDEBUG
    static const bool StressSched = false;
#else
    bool StressSched;
#endif

    explicit ScheduleDAG(MachineFunction &mf);

    virtual ~ScheduleDAG();

    /// clearDAG - clear the DAG state (between regions).
    void clearDAG();

    /// getInstrDesc - Return the MCInstrDesc of this SUnit.
    /// Return NULL for SDNodes without a machine opcode.
    const MCInstrDesc *getInstrDesc(const SUnit *SU) const {
      if (SU->isInstr()) return &SU->getInstr()->getDesc();
      return getNodeDesc(SU->getNode());
    }

    /// viewGraph - Pop up a GraphViz/gv window with the ScheduleDAG rendered
    /// using 'dot'.
    ///
    virtual void viewGraph(const Twine &Name, const Twine &Title);
    virtual void viewGraph();

    virtual void dumpNode(const SUnit *SU) const = 0;

    /// getGraphNodeLabel - Return a label for an SUnit node in a visualization
    /// of the ScheduleDAG.
    virtual std::string getGraphNodeLabel(const SUnit *SU) const = 0;

    /// getDAGLabel - Return a label for the region of code covered by the DAG.
    virtual std::string getDAGName() const = 0;

    /// addCustomGraphFeatures - Add custom features for a visualization of
    /// the ScheduleDAG.
    virtual void addCustomGraphFeatures(GraphWriter<ScheduleDAG*> &) const {}

#ifndef NDEBUG
    /// VerifyScheduledDAG - Verify that all SUnits were scheduled and that
    /// their state is consistent. Return the number of scheduled SUnits.
    unsigned VerifyScheduledDAG(bool isBottomUp);
#endif

  private:
    // Return the MCInstrDesc of this SDNode or NULL.
    const MCInstrDesc *getNodeDesc(const SDNode *Node) const;
  };

  class SUnitIterator : public std::iterator<std::forward_iterator_tag,
                                             SUnit, ptrdiff_t> {
    SUnit *Node;
    unsigned Operand;

    SUnitIterator(SUnit *N, unsigned Op) : Node(N), Operand(Op) {}
  public:
    bool operator==(const SUnitIterator& x) const {
      return Operand == x.Operand;
    }
    bool operator!=(const SUnitIterator& x) const { return !operator==(x); }

    pointer operator*() const {
      return Node->Preds[Operand].getSUnit();
    }
    pointer operator->() const { return operator*(); }

    SUnitIterator& operator++() {                // Preincrement
      ++Operand;
      return *this;
    }
    SUnitIterator operator++(int) { // Postincrement
      SUnitIterator tmp = *this; ++*this; return tmp;
    }

    static SUnitIterator begin(SUnit *N) { return SUnitIterator(N, 0); }
    static SUnitIterator end  (SUnit *N) {
      return SUnitIterator(N, (unsigned)N->Preds.size());
    }

    unsigned getOperand() const { return Operand; }
    const SUnit *getNode() const { return Node; }
    /// isCtrlDep - Test if this is not an SDep::Data dependence.
    bool isCtrlDep() const {
      return getSDep().isCtrl();
    }
    bool isArtificialDep() const {
      return getSDep().isArtificial();
    }
    const SDep &getSDep() const {
      return Node->Preds[Operand];
    }
  };

  template <> struct GraphTraits<SUnit*> {
    typedef SUnit NodeType;
    typedef SUnitIterator ChildIteratorType;
    static inline NodeType *getEntryNode(SUnit *N) { return N; }
    static inline ChildIteratorType child_begin(NodeType *N) {
      return SUnitIterator::begin(N);
    }
    static inline ChildIteratorType child_end(NodeType *N) {
      return SUnitIterator::end(N);
    }
  };

  template <> struct GraphTraits<ScheduleDAG*> : public GraphTraits<SUnit*> {
    typedef std::vector<SUnit>::iterator nodes_iterator;
    static nodes_iterator nodes_begin(ScheduleDAG *G) {
      return G->SUnits.begin();
    }
    static nodes_iterator nodes_end(ScheduleDAG *G) {
      return G->SUnits.end();
    }
  };

  /// ScheduleDAGTopologicalSort is a class that computes a topological
  /// ordering for SUnits and provides methods for dynamically updating
  /// the ordering as new edges are added.
  ///
  /// This allows a very fast implementation of IsReachable, for example.
  ///
  class ScheduleDAGTopologicalSort {
    /// SUnits - A reference to the ScheduleDAG's SUnits.
    std::vector<SUnit> &SUnits;
    SUnit *ExitSU;

    /// Index2Node - Maps topological index to the node number.
    std::vector<int> Index2Node;
    /// Node2Index - Maps the node number to its topological index.
    std::vector<int> Node2Index;
    /// Visited - a set of nodes visited during a DFS traversal.
    BitVector Visited;

    /// DFS - make a DFS traversal and mark all nodes affected by the
    /// edge insertion. These nodes will later get new topological indexes
    /// by means of the Shift method.
    void DFS(const SUnit *SU, int UpperBound, bool& HasLoop);

    /// Shift - reassign topological indexes for the nodes in the DAG
    /// to preserve the topological ordering.
    void Shift(BitVector& Visited, int LowerBound, int UpperBound);

    /// Allocate - assign the topological index to the node n.
    void Allocate(int n, int index);

  public:
    ScheduleDAGTopologicalSort(std::vector<SUnit> &SUnits, SUnit *ExitSU);

    /// InitDAGTopologicalSorting - create the initial topological
    /// ordering from the DAG to be scheduled.
    void InitDAGTopologicalSorting();

    /// IsReachable - Checks if SU is reachable from TargetSU.
    bool IsReachable(const SUnit *SU, const SUnit *TargetSU);

    /// WillCreateCycle - Return true if addPred(TargetSU, SU) creates a cycle.
    bool WillCreateCycle(SUnit *TargetSU, SUnit *SU);

    /// AddPred - Updates the topological ordering to accommodate an edge
    /// to be added from SUnit X to SUnit Y.
    void AddPred(SUnit *Y, SUnit *X);

    /// RemovePred - Updates the topological ordering to accommodate an
    /// an edge to be removed from the specified node N from the predecessors
    /// of the current node M.
    void RemovePred(SUnit *M, SUnit *N);

    typedef std::vector<int>::iterator iterator;
    typedef std::vector<int>::const_iterator const_iterator;
    iterator begin() { return Index2Node.begin(); }
    const_iterator begin() const { return Index2Node.begin(); }
    iterator end() { return Index2Node.end(); }
    const_iterator end() const { return Index2Node.end(); }

    typedef std::vector<int>::reverse_iterator reverse_iterator;
    typedef std::vector<int>::const_reverse_iterator const_reverse_iterator;
    reverse_iterator rbegin() { return Index2Node.rbegin(); }
    const_reverse_iterator rbegin() const { return Index2Node.rbegin(); }
    reverse_iterator rend() { return Index2Node.rend(); }
    const_reverse_iterator rend() const { return Index2Node.rend(); }
  };
}

#endif
