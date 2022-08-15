//===-- llvm/CodeGen/SelectionDAGNodes.h - SelectionDAG Nodes ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the SDNode class and derived classes, which are used to
// represent the nodes and operations present in a SelectionDAG.  These nodes
// and operations are machine code level operations, with some similarities to
// the GCC RTL representation.
//
// Clients should include the SelectionDAG.h file instead of this file directly.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SELECTIONDAGNODES_H
#define LLVM_CODEGEN_SELECTIONDAGNODES_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>

namespace llvm {

class SelectionDAG;
class GlobalValue;
class MachineBasicBlock;
class MachineConstantPoolValue;
class SDNode;
class Value;
class MCSymbol;
template <typename T> struct DenseMapInfo;
template <typename T> struct simplify_type;
template <typename T> struct ilist_traits;

void checkForCycles(const SDNode *N, const SelectionDAG *DAG = nullptr,
                    bool force = false);

/// This represents a list of ValueType's that has been intern'd by
/// a SelectionDAG.  Instances of this simple value class are returned by
/// SelectionDAG::getVTList(...).
///
struct SDVTList {
  const EVT *VTs;
  unsigned int NumVTs;
};

namespace ISD {
  /// Node predicates

  /// Return true if the specified node is a
  /// BUILD_VECTOR where all of the elements are ~0 or undef.
  bool isBuildVectorAllOnes(const SDNode *N);

  /// Return true if the specified node is a
  /// BUILD_VECTOR where all of the elements are 0 or undef.
  bool isBuildVectorAllZeros(const SDNode *N);

  /// \brief Return true if the specified node is a BUILD_VECTOR node of
  /// all ConstantSDNode or undef.
  bool isBuildVectorOfConstantSDNodes(const SDNode *N);

  /// \brief Return true if the specified node is a BUILD_VECTOR node of
  /// all ConstantFPSDNode or undef.
  bool isBuildVectorOfConstantFPSDNodes(const SDNode *N);

  /// Return true if the specified node is a
  /// ISD::SCALAR_TO_VECTOR node or a BUILD_VECTOR node where only the low
  /// element is not an undef.
  bool isScalarToVector(const SDNode *N);

  /// Return true if the node has at least one operand
  /// and all operands of the specified node are ISD::UNDEF.
  bool allOperandsUndef(const SDNode *N);
}  // end llvm:ISD namespace
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
/// Unlike LLVM values, Selection DAG nodes may return multiple
/// values as the result of a computation.  Many nodes return multiple values,
/// from loads (which define a token and a return value) to ADDC (which returns
/// a result and a carry value), to calls (which may return an arbitrary number
/// of values).
///
/// As such, each use of a SelectionDAG computation must indicate the node that
/// computes it as well as which return value to use from that node.  This pair
/// of information is represented with the SDValue value type.
///
class SDValue {
  friend struct DenseMapInfo<SDValue>;

  SDNode *Node;       // The node defining the value we are using.
  unsigned ResNo;     // Which return value of the node we are using.
public:
  SDValue() : Node(nullptr), ResNo(0) {}
  SDValue(SDNode *node, unsigned resno);

  /// get the index which selects a specific result in the SDNode
  unsigned getResNo() const { return ResNo; }

  /// get the SDNode which holds the desired result
  SDNode *getNode() const { return Node; }

  /// set the SDNode
  void setNode(SDNode *N) { Node = N; }

  inline SDNode *operator->() const { return Node; }

  bool operator==(const SDValue &O) const {
    return Node == O.Node && ResNo == O.ResNo;
  }
  bool operator!=(const SDValue &O) const {
    return !operator==(O);
  }
  bool operator<(const SDValue &O) const {
    return std::tie(Node, ResNo) < std::tie(O.Node, O.ResNo);
  }
  explicit operator bool() const {
    return Node != nullptr;
  }

  SDValue getValue(unsigned R) const {
    return SDValue(Node, R);
  }

  // Return true if this node is an operand of N.
  bool isOperandOf(const SDNode *N) const;

  /// Return the ValueType of the referenced return value.
  inline EVT getValueType() const;

  /// Return the simple ValueType of the referenced return value.
  MVT getSimpleValueType() const {
    return getValueType().getSimpleVT();
  }

  /// Returns the size of the value in bits.
  unsigned getValueSizeInBits() const {
    return getValueType().getSizeInBits();
  }

  unsigned getScalarValueSizeInBits() const {
    return getValueType().getScalarType().getSizeInBits();
  }

  // Forwarding methods - These forward to the corresponding methods in SDNode.
  inline unsigned getOpcode() const;
  inline unsigned getNumOperands() const;
  inline const SDValue &getOperand(unsigned i) const;
  inline uint64_t getConstantOperandVal(unsigned i) const;
  inline bool isTargetMemoryOpcode() const;
  inline bool isTargetOpcode() const;
  inline bool isMachineOpcode() const;
  inline unsigned getMachineOpcode() const;
  inline const DebugLoc &getDebugLoc() const;
  inline void dump() const;
  inline void dumpr() const;

  /// Return true if this operand (which must be a chain) reaches the
  /// specified operand without crossing any side-effecting instructions.
  /// In practice, this looks through token factors and non-volatile loads.
  /// In order to remain efficient, this only
  /// looks a couple of nodes in, it does not do an exhaustive search.
  bool reachesChainWithoutSideEffects(SDValue Dest,
                                      unsigned Depth = 2) const;

  /// Return true if there are no nodes using value ResNo of Node.
  inline bool use_empty() const;

  /// Return true if there is exactly one node using value ResNo of Node.
  inline bool hasOneUse() const;
};


template<> struct DenseMapInfo<SDValue> {
  static inline SDValue getEmptyKey() {
    SDValue V;
    V.ResNo = -1U;
    return V;
  }
  static inline SDValue getTombstoneKey() {
    SDValue V;
    V.ResNo = -2U;
    return V;
  }
  static unsigned getHashValue(const SDValue &Val) {
    return ((unsigned)((uintptr_t)Val.getNode() >> 4) ^
            (unsigned)((uintptr_t)Val.getNode() >> 9)) + Val.getResNo();
  }
  static bool isEqual(const SDValue &LHS, const SDValue &RHS) {
    return LHS == RHS;
  }
};
template <> struct isPodLike<SDValue> { static const bool value = true; };


/// Allow casting operators to work directly on
/// SDValues as if they were SDNode*'s.
template<> struct simplify_type<SDValue> {
  typedef SDNode* SimpleType;
  static SimpleType getSimplifiedValue(SDValue &Val) {
    return Val.getNode();
  }
};
template<> struct simplify_type<const SDValue> {
  typedef /*const*/ SDNode* SimpleType;
  static SimpleType getSimplifiedValue(const SDValue &Val) {
    return Val.getNode();
  }
};

/// Represents a use of a SDNode. This class holds an SDValue,
/// which records the SDNode being used and the result number, a
/// pointer to the SDNode using the value, and Next and Prev pointers,
/// which link together all the uses of an SDNode.
///
class SDUse {
  /// Val - The value being used.
  SDValue Val;
  /// User - The user of this value.
  SDNode *User;
  /// Prev, Next - Pointers to the uses list of the SDNode referred by
  /// this operand.
  SDUse **Prev, *Next;

  SDUse(const SDUse &U) = delete;
  void operator=(const SDUse &U) = delete;

public:
  SDUse() : Val(), User(nullptr), Prev(nullptr), Next(nullptr) {}

  /// Normally SDUse will just implicitly convert to an SDValue that it holds.
  operator const SDValue&() const { return Val; }

  /// If implicit conversion to SDValue doesn't work, the get() method returns
  /// the SDValue.
  const SDValue &get() const { return Val; }

  /// This returns the SDNode that contains this Use.
  SDNode *getUser() { return User; }

  /// Get the next SDUse in the use list.
  SDUse *getNext() const { return Next; }

  /// Convenience function for get().getNode().
  SDNode *getNode() const { return Val.getNode(); }
  /// Convenience function for get().getResNo().
  unsigned getResNo() const { return Val.getResNo(); }
  /// Convenience function for get().getValueType().
  EVT getValueType() const { return Val.getValueType(); }

  /// Convenience function for get().operator==
  bool operator==(const SDValue &V) const {
    return Val == V;
  }

  /// Convenience function for get().operator!=
  bool operator!=(const SDValue &V) const {
    return Val != V;
  }

  /// Convenience function for get().operator<
  bool operator<(const SDValue &V) const {
    return Val < V;
  }

private:
  friend class SelectionDAG;
  friend class SDNode;

  void setUser(SDNode *p) { User = p; }

  /// Remove this use from its existing use list, assign it the
  /// given value, and add it to the new value's node's use list.
  inline void set(const SDValue &V);
  /// Like set, but only supports initializing a newly-allocated
  /// SDUse with a non-null value.
  inline void setInitial(const SDValue &V);
  /// Like set, but only sets the Node portion of the value,
  /// leaving the ResNo portion unmodified.
  inline void setNode(SDNode *N);

  void addToList(SDUse **List) {
    Next = *List;
    if (Next) Next->Prev = &Next;
    Prev = List;
    *List = this;
  }

  void removeFromList() {
    *Prev = Next;
    if (Next) Next->Prev = Prev;
  }
};

/// simplify_type specializations - Allow casting operators to work directly on
/// SDValues as if they were SDNode*'s.
template<> struct simplify_type<SDUse> {
  typedef SDNode* SimpleType;
  static SimpleType getSimplifiedValue(SDUse &Val) {
    return Val.getNode();
  }
};


/// Represents one node in the SelectionDAG.
///
class SDNode : public FoldingSetNode, public ilist_node<SDNode> {
private:
  /// The operation that this node performs.
  int16_t NodeType;

  /// This is true if OperandList was new[]'d.  If true,
  /// then they will be delete[]'d when the node is destroyed.
  uint16_t OperandsNeedDelete : 1;

  /// This tracks whether this node has one or more dbg_value
  /// nodes corresponding to it.
  uint16_t HasDebugValue : 1;

protected:
  /// This member is defined by this class, but is not used for
  /// anything.  Subclasses can use it to hold whatever state they find useful.
  /// This field is initialized to zero by the ctor.
  uint16_t SubclassData : 14;

private:
  /// Unique id per SDNode in the DAG.
  int NodeId;

  /// The values that are used by this operation.
  SDUse *OperandList;

  /// The types of the values this node defines.  SDNode's may
  /// define multiple values simultaneously.
  const EVT *ValueList;

  /// List of uses for this SDNode.
  SDUse *UseList;

  /// The number of entries in the Operand/Value list.
  unsigned short NumOperands, NumValues;

  // The ordering of the SDNodes. It roughly corresponds to the ordering of the
  // original LLVM instructions.
  // This is used for turning off scheduling, because we'll forgo
  // the normal scheduling algorithms and output the instructions according to
  // this ordering.
  unsigned IROrder;

  /// Source line information.
  DebugLoc debugLoc;

  /// Return a pointer to the specified value type.
  static const EVT *getValueTypeList(EVT VT);

  friend class SelectionDAG;
  friend struct ilist_traits<SDNode>;

public:
  //===--------------------------------------------------------------------===//
  //  Accessors
  //

  /// Return the SelectionDAG opcode value for this node. For
  /// pre-isel nodes (those for which isMachineOpcode returns false), these
  /// are the opcode values in the ISD and <target>ISD namespaces. For
  /// post-isel opcodes, see getMachineOpcode.
  unsigned getOpcode()  const { return (unsigned short)NodeType; }

  /// Test if this node has a target-specific opcode (in the
  /// \<target\>ISD namespace).
  bool isTargetOpcode() const { return NodeType >= ISD::BUILTIN_OP_END; }

  /// Test if this node has a target-specific
  /// memory-referencing opcode (in the \<target\>ISD namespace and
  /// greater than FIRST_TARGET_MEMORY_OPCODE).
  bool isTargetMemoryOpcode() const {
    return NodeType >= ISD::FIRST_TARGET_MEMORY_OPCODE;
  }

  /// Test if this node is a memory intrinsic (with valid pointer information).
  /// INTRINSIC_W_CHAIN and INTRINSIC_VOID nodes are sometimes created for
  /// non-memory intrinsics (with chains) that are not really instances of
  /// MemSDNode. For such nodes, we need some extra state to determine the
  /// proper classof relationship.
  bool isMemIntrinsic() const {
    return (NodeType == ISD::INTRINSIC_W_CHAIN ||
            NodeType == ISD::INTRINSIC_VOID) && ((SubclassData >> 13) & 1);
  }

  /// Test if this node has a post-isel opcode, directly
  /// corresponding to a MachineInstr opcode.
  bool isMachineOpcode() const { return NodeType < 0; }

  /// This may only be called if isMachineOpcode returns
  /// true. It returns the MachineInstr opcode value that the node's opcode
  /// corresponds to.
  unsigned getMachineOpcode() const {
    assert(isMachineOpcode() && "Not a MachineInstr opcode!");
    return ~NodeType;
  }

  /// Get this bit.
  bool getHasDebugValue() const { return HasDebugValue; }

  /// Set this bit.
  void setHasDebugValue(bool b) { HasDebugValue = b; }

  /// Return true if there are no uses of this node.
  bool use_empty() const { return UseList == nullptr; }

  /// Return true if there is exactly one use of this node.
  bool hasOneUse() const {
    return !use_empty() && std::next(use_begin()) == use_end();
  }

  /// Return the number of uses of this node. This method takes
  /// time proportional to the number of uses.
  size_t use_size() const { return std::distance(use_begin(), use_end()); }

  /// Return the unique node id.
  int getNodeId() const { return NodeId; }

  /// Set unique node id.
  void setNodeId(int Id) { NodeId = Id; }

  /// Return the node ordering.
  unsigned getIROrder() const { return IROrder; }

  /// Set the node ordering.
  void setIROrder(unsigned Order) { IROrder = Order; }

  /// Return the source location info.
  const DebugLoc &getDebugLoc() const { return debugLoc; }

  /// Set source location info.  Try to avoid this, putting
  /// it in the constructor is preferable.
  void setDebugLoc(DebugLoc dl) { debugLoc = std::move(dl); }

  /// This class provides iterator support for SDUse
  /// operands that use a specific SDNode.
  class use_iterator
    : public std::iterator<std::forward_iterator_tag, SDUse, ptrdiff_t> {
    SDUse *Op;
    explicit use_iterator(SDUse *op) : Op(op) {
    }
    friend class SDNode;
  public:
    typedef std::iterator<std::forward_iterator_tag,
                          SDUse, ptrdiff_t>::reference reference;
    typedef std::iterator<std::forward_iterator_tag,
                          SDUse, ptrdiff_t>::pointer pointer;

    use_iterator(const use_iterator &I) : Op(I.Op) {}
    use_iterator() : Op(nullptr) {}

    bool operator==(const use_iterator &x) const {
      return Op == x.Op;
    }
    bool operator!=(const use_iterator &x) const {
      return !operator==(x);
    }

    /// Return true if this iterator is at the end of uses list.
    bool atEnd() const { return Op == nullptr; }

    // Iterator traversal: forward iteration only.
    use_iterator &operator++() {          // Preincrement
      assert(Op && "Cannot increment end iterator!");
      Op = Op->getNext();
      return *this;
    }

    use_iterator operator++(int) {        // Postincrement
      use_iterator tmp = *this; ++*this; return tmp;
    }

    /// Retrieve a pointer to the current user node.
    SDNode *operator*() const {
      assert(Op && "Cannot dereference end iterator!");
      return Op->getUser();
    }

    SDNode *operator->() const { return operator*(); }

    SDUse &getUse() const { return *Op; }

    /// Retrieve the operand # of this use in its user.
    unsigned getOperandNo() const {
      assert(Op && "Cannot dereference end iterator!");
      return (unsigned)(Op - Op->getUser()->OperandList);
    }
  };

  /// Provide iteration support to walk over all uses of an SDNode.
  use_iterator use_begin() const {
    return use_iterator(UseList);
  }

  static use_iterator use_end() { return use_iterator(nullptr); }

  inline iterator_range<use_iterator> uses() {
    return iterator_range<use_iterator>(use_begin(), use_end());
  }
  inline iterator_range<use_iterator> uses() const {
    return iterator_range<use_iterator>(use_begin(), use_end());
  }

  /// Return true if there are exactly NUSES uses of the indicated value.
  /// This method ignores uses of other values defined by this operation.
  bool hasNUsesOfValue(unsigned NUses, unsigned Value) const;

  /// Return true if there are any use of the indicated value.
  /// This method ignores uses of other values defined by this operation.
  bool hasAnyUseOfValue(unsigned Value) const;

  /// Return true if this node is the only use of N.
  bool isOnlyUserOf(const SDNode *N) const;

  /// Return true if this node is an operand of N.
  bool isOperandOf(const SDNode *N) const;

  /// Return true if this node is a predecessor of N.
  /// NOTE: Implemented on top of hasPredecessor and every bit as
  /// expensive. Use carefully.
  bool isPredecessorOf(const SDNode *N) const {
    return N->hasPredecessor(this);
  }

  /// Return true if N is a predecessor of this node.
  /// N is either an operand of this node, or can be reached by recursively
  /// traversing up the operands.
  /// NOTE: This is an expensive method. Use it carefully.
  bool hasPredecessor(const SDNode *N) const;

  /// Return true if N is a predecessor of this node.
  /// N is either an operand of this node, or can be reached by recursively
  /// traversing up the operands.
  /// In this helper the Visited and worklist sets are held externally to
  /// cache predecessors over multiple invocations. If you want to test for
  /// multiple predecessors this method is preferable to multiple calls to
  /// hasPredecessor. Be sure to clear Visited and Worklist if the DAG
  /// changes.
  /// NOTE: This is still very expensive. Use carefully.
  bool hasPredecessorHelper(const SDNode *N,
                            SmallPtrSetImpl<const SDNode *> &Visited,
                            SmallVectorImpl<const SDNode *> &Worklist) const;

  /// Return the number of values used by this operation.
  unsigned getNumOperands() const { return NumOperands; }

  /// Helper method returns the integer value of a ConstantSDNode operand.
  uint64_t getConstantOperandVal(unsigned Num) const;

  const SDValue &getOperand(unsigned Num) const {
    assert(Num < NumOperands && "Invalid child # of SDNode!");
    return OperandList[Num];
  }

  typedef SDUse* op_iterator;
  op_iterator op_begin() const { return OperandList; }
  op_iterator op_end() const { return OperandList+NumOperands; }
  ArrayRef<SDUse> ops() const { return makeArrayRef(op_begin(), op_end()); }

  /// Iterator for directly iterating over the operand SDValue's.
  struct value_op_iterator
      : iterator_adaptor_base<value_op_iterator, op_iterator,
                              std::random_access_iterator_tag, SDValue,
                              ptrdiff_t, value_op_iterator *,
                              value_op_iterator *> {
    explicit value_op_iterator(SDUse *U = nullptr)
      : iterator_adaptor_base(U) {}

    const SDValue &operator*() const { return I->get(); }
  };

  iterator_range<value_op_iterator> op_values() const {
    return iterator_range<value_op_iterator>(value_op_iterator(op_begin()),
                                             value_op_iterator(op_end()));
  }

  SDVTList getVTList() const {
    SDVTList X = { ValueList, NumValues };
    return X;
  }

  /// If this node has a glue operand, return the node
  /// to which the glue operand points. Otherwise return NULL.
  SDNode *getGluedNode() const {
    if (getNumOperands() != 0 &&
      getOperand(getNumOperands()-1).getValueType() == MVT::Glue)
      return getOperand(getNumOperands()-1).getNode();
    return nullptr;
  }

  // If this is a pseudo op, like copyfromreg, look to see if there is a
  // real target node glued to it.  If so, return the target node.
  const SDNode *getGluedMachineNode() const {
    const SDNode *FoundNode = this;

    // Climb up glue edges until a machine-opcode node is found, or the
    // end of the chain is reached.
    while (!FoundNode->isMachineOpcode()) {
      const SDNode *N = FoundNode->getGluedNode();
      if (!N) break;
      FoundNode = N;
    }

    return FoundNode;
  }

  /// If this node has a glue value with a user, return
  /// the user (there is at most one). Otherwise return NULL.
  SDNode *getGluedUser() const {
    for (use_iterator UI = use_begin(), UE = use_end(); UI != UE; ++UI)
      if (UI.getUse().get().getValueType() == MVT::Glue)
        return *UI;
    return nullptr;
  }

  /// Return the number of values defined/returned by this operator.
  unsigned getNumValues() const { return NumValues; }

  /// Return the type of a specified result.
  EVT getValueType(unsigned ResNo) const {
    assert(ResNo < NumValues && "Illegal result number!");
    return ValueList[ResNo];
  }

  /// Return the type of a specified result as a simple type.
  MVT getSimpleValueType(unsigned ResNo) const {
    return getValueType(ResNo).getSimpleVT();
  }

  /// Returns MVT::getSizeInBits(getValueType(ResNo)).
  unsigned getValueSizeInBits(unsigned ResNo) const {
    return getValueType(ResNo).getSizeInBits();
  }

  typedef const EVT* value_iterator;
  value_iterator value_begin() const { return ValueList; }
  value_iterator value_end() const { return ValueList+NumValues; }

  /// Return the opcode of this operation for printing.
  std::string getOperationName(const SelectionDAG *G = nullptr) const;
  static const char* getIndexedModeName(ISD::MemIndexedMode AM);
  void print_types(raw_ostream &OS, const SelectionDAG *G) const;
  void print_details(raw_ostream &OS, const SelectionDAG *G) const;
  void print(raw_ostream &OS, const SelectionDAG *G = nullptr) const;
  void printr(raw_ostream &OS, const SelectionDAG *G = nullptr) const;

  /// Print a SelectionDAG node and all children down to
  /// the leaves.  The given SelectionDAG allows target-specific nodes
  /// to be printed in human-readable form.  Unlike printr, this will
  /// print the whole DAG, including children that appear multiple
  /// times.
  ///
  void printrFull(raw_ostream &O, const SelectionDAG *G = nullptr) const;

  /// Print a SelectionDAG node and children up to
  /// depth "depth."  The given SelectionDAG allows target-specific
  /// nodes to be printed in human-readable form.  Unlike printr, this
  /// will print children that appear multiple times wherever they are
  /// used.
  ///
  void printrWithDepth(raw_ostream &O, const SelectionDAG *G = nullptr,
                       unsigned depth = 100) const;


  /// Dump this node, for debugging.
  void dump() const;

  /// Dump (recursively) this node and its use-def subgraph.
  void dumpr() const;

  /// Dump this node, for debugging.
  /// The given SelectionDAG allows target-specific nodes to be printed
  /// in human-readable form.
  void dump(const SelectionDAG *G) const;

  /// Dump (recursively) this node and its use-def subgraph.
  /// The given SelectionDAG allows target-specific nodes to be printed
  /// in human-readable form.
  void dumpr(const SelectionDAG *G) const;

  /// printrFull to dbgs().  The given SelectionDAG allows
  /// target-specific nodes to be printed in human-readable form.
  /// Unlike dumpr, this will print the whole DAG, including children
  /// that appear multiple times.
  void dumprFull(const SelectionDAG *G = nullptr) const;

  /// printrWithDepth to dbgs().  The given
  /// SelectionDAG allows target-specific nodes to be printed in
  /// human-readable form.  Unlike dumpr, this will print children
  /// that appear multiple times wherever they are used.
  ///
  void dumprWithDepth(const SelectionDAG *G = nullptr,
                      unsigned depth = 100) const;

  /// Gather unique data for the node.
  void Profile(FoldingSetNodeID &ID) const;

  /// This method should only be used by the SDUse class.
  void addUse(SDUse &U) { U.addToList(&UseList); }

protected:
  static SDVTList getSDVTList(EVT VT) {
    SDVTList Ret = { getValueTypeList(VT), 1 };
    return Ret;
  }

  SDNode(unsigned Opc, unsigned Order, DebugLoc dl, SDVTList VTs,
         ArrayRef<SDValue> Ops)
      : NodeType(Opc), OperandsNeedDelete(true), HasDebugValue(false),
        SubclassData(0), NodeId(-1),
        OperandList(Ops.size() ? new SDUse[Ops.size()] : nullptr),
        ValueList(VTs.VTs), UseList(nullptr), NumOperands(Ops.size()),
        NumValues(VTs.NumVTs), IROrder(Order), debugLoc(std::move(dl)) {
    assert(debugLoc.hasTrivialDestructor() && "Expected trivial destructor");
    assert(NumOperands == Ops.size() &&
           "NumOperands wasn't wide enough for its operands!");
    assert(NumValues == VTs.NumVTs &&
           "NumValues wasn't wide enough for its operands!");
    for (unsigned i = 0; i != Ops.size(); ++i) {
      assert(OperandList && "no operands available");
      OperandList[i].setUser(this);
      OperandList[i].setInitial(Ops[i]);
    }
    checkForCycles(this);
  }

  /// This constructor adds no operands itself; operands can be
  /// set later with InitOperands.
  SDNode(unsigned Opc, unsigned Order, DebugLoc dl, SDVTList VTs)
      : NodeType(Opc), OperandsNeedDelete(false), HasDebugValue(false),
        SubclassData(0), NodeId(-1), OperandList(nullptr), ValueList(VTs.VTs),
        UseList(nullptr), NumOperands(0), NumValues(VTs.NumVTs),
        IROrder(Order), debugLoc(std::move(dl)) {
    assert(debugLoc.hasTrivialDestructor() && "Expected trivial destructor");
    assert(NumValues == VTs.NumVTs &&
           "NumValues wasn't wide enough for its operands!");
  }

  /// Initialize the operands list of this with 1 operand.
  void InitOperands(SDUse *Ops, const SDValue &Op0) {
    Ops[0].setUser(this);
    Ops[0].setInitial(Op0);
    NumOperands = 1;
    OperandList = Ops;
    checkForCycles(this);
  }

  /// Initialize the operands list of this with 2 operands.
  void InitOperands(SDUse *Ops, const SDValue &Op0, const SDValue &Op1) {
    Ops[0].setUser(this);
    Ops[0].setInitial(Op0);
    Ops[1].setUser(this);
    Ops[1].setInitial(Op1);
    NumOperands = 2;
    OperandList = Ops;
    checkForCycles(this);
  }

  /// Initialize the operands list of this with 3 operands.
  void InitOperands(SDUse *Ops, const SDValue &Op0, const SDValue &Op1,
                    const SDValue &Op2) {
    Ops[0].setUser(this);
    Ops[0].setInitial(Op0);
    Ops[1].setUser(this);
    Ops[1].setInitial(Op1);
    Ops[2].setUser(this);
    Ops[2].setInitial(Op2);
    NumOperands = 3;
    OperandList = Ops;
    checkForCycles(this);
  }

  /// Initialize the operands list of this with 4 operands.
  void InitOperands(SDUse *Ops, const SDValue &Op0, const SDValue &Op1,
                    const SDValue &Op2, const SDValue &Op3) {
    Ops[0].setUser(this);
    Ops[0].setInitial(Op0);
    Ops[1].setUser(this);
    Ops[1].setInitial(Op1);
    Ops[2].setUser(this);
    Ops[2].setInitial(Op2);
    Ops[3].setUser(this);
    Ops[3].setInitial(Op3);
    NumOperands = 4;
    OperandList = Ops;
    checkForCycles(this);
  }

  /// Initialize the operands list of this with N operands.
  void InitOperands(SDUse *Ops, const SDValue *Vals, unsigned N) {
    for (unsigned i = 0; i != N; ++i) {
      Ops[i].setUser(this);
      Ops[i].setInitial(Vals[i]);
    }
    NumOperands = N;
    assert(NumOperands == N &&
           "NumOperands wasn't wide enough for its operands!");
    OperandList = Ops;
    checkForCycles(this);
  }

  /// Release the operands and set this node to have zero operands.
  void DropOperands();
};

/// Wrapper class for IR location info (IR ordering and DebugLoc) to be passed
/// into SDNode creation functions.
/// When an SDNode is created from the DAGBuilder, the DebugLoc is extracted
/// from the original Instruction, and IROrder is the ordinal position of
/// the instruction.
/// When an SDNode is created after the DAG is being built, both DebugLoc and
/// the IROrder are propagated from the original SDNode.
/// So SDLoc class provides two constructors besides the default one, one to
/// be used by the DAGBuilder, the other to be used by others.
class SDLoc {
private:
  // Ptr could be used for either Instruction* or SDNode*. It is used for
  // Instruction* if IROrder is not -1.
  const void *Ptr;
  int IROrder;

public:
  SDLoc() : Ptr(nullptr), IROrder(0) {}
  SDLoc(const SDNode *N) : Ptr(N), IROrder(-1) {
    assert(N && "null SDNode");
  }
  SDLoc(const SDValue V) : Ptr(V.getNode()), IROrder(-1) {
    assert(Ptr && "null SDNode");
  }
  SDLoc(const Instruction *I, int Order) : Ptr(I), IROrder(Order) {
    assert(Order >= 0 && "bad IROrder");
  }
  unsigned getIROrder() {
    if (IROrder >= 0 || Ptr == nullptr) {
      return (unsigned)IROrder;
    }
    const SDNode *N = (const SDNode*)(Ptr);
    return N->getIROrder();
  }
  DebugLoc getDebugLoc() {
    if (!Ptr) {
      return DebugLoc();
    }
    if (IROrder >= 0) {
      const Instruction *I = (const Instruction*)(Ptr);
      return I->getDebugLoc();
    }
    const SDNode *N = (const SDNode*)(Ptr);
    return N->getDebugLoc();
  }
};


// Define inline functions from the SDValue class.

inline SDValue::SDValue(SDNode *node, unsigned resno)
    : Node(node), ResNo(resno) {
  assert((!Node || ResNo < Node->getNumValues()) &&
         "Invalid result number for the given node!");
  assert(ResNo < -2U && "Cannot use result numbers reserved for DenseMaps.");
}

inline unsigned SDValue::getOpcode() const {
  return Node->getOpcode();
}
inline EVT SDValue::getValueType() const {
  return Node->getValueType(ResNo);
}
inline unsigned SDValue::getNumOperands() const {
  return Node->getNumOperands();
}
inline const SDValue &SDValue::getOperand(unsigned i) const {
  return Node->getOperand(i);
}
inline uint64_t SDValue::getConstantOperandVal(unsigned i) const {
  return Node->getConstantOperandVal(i);
}
inline bool SDValue::isTargetOpcode() const {
  return Node->isTargetOpcode();
}
inline bool SDValue::isTargetMemoryOpcode() const {
  return Node->isTargetMemoryOpcode();
}
inline bool SDValue::isMachineOpcode() const {
  return Node->isMachineOpcode();
}
inline unsigned SDValue::getMachineOpcode() const {
  return Node->getMachineOpcode();
}
inline bool SDValue::use_empty() const {
  return !Node->hasAnyUseOfValue(ResNo);
}
inline bool SDValue::hasOneUse() const {
  return Node->hasNUsesOfValue(1, ResNo);
}
inline const DebugLoc &SDValue::getDebugLoc() const {
  return Node->getDebugLoc();
}
inline void SDValue::dump() const {
  return Node->dump();
}
inline void SDValue::dumpr() const {
  return Node->dumpr();
}
// Define inline functions from the SDUse class.

inline void SDUse::set(const SDValue &V) {
  if (Val.getNode()) removeFromList();
  Val = V;
  if (V.getNode()) V.getNode()->addUse(*this);
}

inline void SDUse::setInitial(const SDValue &V) {
  Val = V;
  V.getNode()->addUse(*this);
}

inline void SDUse::setNode(SDNode *N) {
  if (Val.getNode()) removeFromList();
  Val.setNode(N);
  if (N) N->addUse(*this);
}

/// These are IR-level optimization flags that may be propagated to SDNodes.
/// TODO: This data structure should be shared by the IR optimizer and the
/// the backend.
struct SDNodeFlags {
private:
  bool NoUnsignedWrap : 1;
  bool NoSignedWrap : 1;
  bool Exact : 1;
  bool UnsafeAlgebra : 1;
  bool NoNaNs : 1;
  bool NoInfs : 1;
  bool NoSignedZeros : 1;
  bool AllowReciprocal : 1;

public:
  /// Default constructor turns off all optimization flags.
  SDNodeFlags() {
    NoUnsignedWrap = false;
    NoSignedWrap = false;
    Exact = false;
    UnsafeAlgebra = false;
    NoNaNs = false;
    NoInfs = false;
    NoSignedZeros = false;
    AllowReciprocal = false;
  }

  // These are mutators for each flag.
  void setNoUnsignedWrap(bool b) { NoUnsignedWrap = b; }
  void setNoSignedWrap(bool b) { NoSignedWrap = b; }
  void setExact(bool b) { Exact = b; }
  void setUnsafeAlgebra(bool b) { UnsafeAlgebra = b; }
  void setNoNaNs(bool b) { NoNaNs = b; }
  void setNoInfs(bool b) { NoInfs = b; }
  void setNoSignedZeros(bool b) { NoSignedZeros = b; }
  void setAllowReciprocal(bool b) { AllowReciprocal = b; }

  // These are accessors for each flag.
  bool hasNoUnsignedWrap() const { return NoUnsignedWrap; }
  bool hasNoSignedWrap() const { return NoSignedWrap; }
  bool hasExact() const { return Exact; }
  bool hasUnsafeAlgebra() const { return UnsafeAlgebra; }
  bool hasNoNaNs() const { return NoNaNs; }
  bool hasNoInfs() const { return NoInfs; }
  bool hasNoSignedZeros() const { return NoSignedZeros; }
  bool hasAllowReciprocal() const { return AllowReciprocal; }

  /// Return a raw encoding of the flags.
  /// This function should only be used to add data to the NodeID value.
  unsigned getRawFlags() const {
    return (NoUnsignedWrap << 0) | (NoSignedWrap << 1) | (Exact << 2) |
           (UnsafeAlgebra << 3) | (NoNaNs << 4) | (NoInfs << 5) |
           (NoSignedZeros << 6) | (AllowReciprocal << 7);
  }
};

/// This class is used for single-operand SDNodes.  This is solely
/// to allow co-allocation of node operands with the node itself.
class UnarySDNode : public SDNode {
  SDUse Op;
public:
  UnarySDNode(unsigned Opc, unsigned Order, DebugLoc dl, SDVTList VTs,
              SDValue X)
    : SDNode(Opc, Order, dl, VTs) {
    InitOperands(&Op, X);
  }
};

/// This class is used for two-operand SDNodes.  This is solely
/// to allow co-allocation of node operands with the node itself.
class BinarySDNode : public SDNode {
  SDUse Ops[2];
public:
  BinarySDNode(unsigned Opc, unsigned Order, DebugLoc dl, SDVTList VTs,
               SDValue X, SDValue Y)
    : SDNode(Opc, Order, dl, VTs) {
    InitOperands(Ops, X, Y);
  }
};

/// Returns true if the opcode is a binary operation with flags.
static bool isBinOpWithFlags(unsigned Opcode) {
  switch (Opcode) {
  case ISD::SDIV:
  case ISD::UDIV:
  case ISD::SRA:
  case ISD::SRL:
  case ISD::MUL:
  case ISD::ADD:
  case ISD::SUB:
  case ISD::SHL:
  case ISD::FADD:
  case ISD::FDIV:
  case ISD::FMUL:
  case ISD::FREM:
  case ISD::FSUB:
    return true;
  default:
    return false;
  }
}

/// This class is an extension of BinarySDNode
/// used from those opcodes that have associated extra flags.
class BinaryWithFlagsSDNode : public BinarySDNode {
public:
  SDNodeFlags Flags;
  BinaryWithFlagsSDNode(unsigned Opc, unsigned Order, DebugLoc dl, SDVTList VTs,
                        SDValue X, SDValue Y, const SDNodeFlags &NodeFlags)
      : BinarySDNode(Opc, Order, dl, VTs, X, Y), Flags(NodeFlags) {}
  static bool classof(const SDNode *N) {
    return isBinOpWithFlags(N->getOpcode());
  }
};

/// This class is used for three-operand SDNodes. This is solely
/// to allow co-allocation of node operands with the node itself.
class TernarySDNode : public SDNode {
  SDUse Ops[3];
public:
  TernarySDNode(unsigned Opc, unsigned Order, DebugLoc dl, SDVTList VTs,
                SDValue X, SDValue Y, SDValue Z)
    : SDNode(Opc, Order, dl, VTs) {
    InitOperands(Ops, X, Y, Z);
  }
};


/// This class is used to form a handle around another node that
/// is persistent and is updated across invocations of replaceAllUsesWith on its
/// operand.  This node should be directly created by end-users and not added to
/// the AllNodes list.
class HandleSDNode : public SDNode {
  SDUse Op;
public:
  explicit HandleSDNode(SDValue X)
    : SDNode(ISD::HANDLENODE, 0, DebugLoc(), getSDVTList(MVT::Other)) {
    InitOperands(&Op, X);
  }
  ~HandleSDNode();
  const SDValue &getValue() const { return Op; }
};

class AddrSpaceCastSDNode : public UnarySDNode {
private:
  unsigned SrcAddrSpace;
  unsigned DestAddrSpace;

public:
  AddrSpaceCastSDNode(unsigned Order, DebugLoc dl, EVT VT, SDValue X,
                      unsigned SrcAS, unsigned DestAS);

  unsigned getSrcAddressSpace() const { return SrcAddrSpace; }
  unsigned getDestAddressSpace() const { return DestAddrSpace; }

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::ADDRSPACECAST;
  }
};

/// This is an abstract virtual class for memory operations.
class MemSDNode : public SDNode {
private:
  // VT of in-memory value.
  EVT MemoryVT;

protected:
  /// Memory reference information.
  MachineMemOperand *MMO;

public:
  MemSDNode(unsigned Opc, unsigned Order, DebugLoc dl, SDVTList VTs,
            EVT MemoryVT, MachineMemOperand *MMO);

  MemSDNode(unsigned Opc, unsigned Order, DebugLoc dl, SDVTList VTs,
            ArrayRef<SDValue> Ops, EVT MemoryVT, MachineMemOperand *MMO);

  bool readMem() const { return MMO->isLoad(); }
  bool writeMem() const { return MMO->isStore(); }

  /// Returns alignment and volatility of the memory access
  unsigned getOriginalAlignment() const {
    return MMO->getBaseAlignment();
  }
  unsigned getAlignment() const {
    return MMO->getAlignment();
  }

  /// Return the SubclassData value, which contains an
  /// encoding of the volatile flag, as well as bits used by subclasses. This
  /// function should only be used to compute a FoldingSetNodeID value.
  unsigned getRawSubclassData() const {
    return SubclassData;
  }

  // We access subclass data here so that we can check consistency
  // with MachineMemOperand information.
  bool isVolatile() const { return (SubclassData >> 5) & 1; }
  bool isNonTemporal() const { return (SubclassData >> 6) & 1; }
  bool isInvariant() const { return (SubclassData >> 7) & 1; }

  AtomicOrdering getOrdering() const {
    return AtomicOrdering((SubclassData >> 8) & 15);
  }
  SynchronizationScope getSynchScope() const {
    return SynchronizationScope((SubclassData >> 12) & 1);
  }

  // Returns the offset from the location of the access.
  int64_t getSrcValueOffset() const { return MMO->getOffset(); }

  /// Returns the AA info that describes the dereference.
  AAMDNodes getAAInfo() const { return MMO->getAAInfo(); }

  /// Returns the Ranges that describes the dereference.
  const MDNode *getRanges() const { return MMO->getRanges(); }

  /// Return the type of the in-memory value.
  EVT getMemoryVT() const { return MemoryVT; }

  /// Return a MachineMemOperand object describing the memory
  /// reference performed by operation.
  MachineMemOperand *getMemOperand() const { return MMO; }

  const MachinePointerInfo &getPointerInfo() const {
    return MMO->getPointerInfo();
  }

  /// Return the address space for the associated pointer
  unsigned getAddressSpace() const {
    return getPointerInfo().getAddrSpace();
  }

  /// Update this MemSDNode's MachineMemOperand information
  /// to reflect the alignment of NewMMO, if it has a greater alignment.
  /// This must only be used when the new alignment applies to all users of
  /// this MachineMemOperand.
  void refineAlignment(const MachineMemOperand *NewMMO) {
    MMO->refineAlignment(NewMMO);
  }

  const SDValue &getChain() const { return getOperand(0); }
  const SDValue &getBasePtr() const {
    return getOperand(getOpcode() == ISD::STORE ? 2 : 1);
  }

  // Methods to support isa and dyn_cast
  static bool classof(const SDNode *N) {
    // For some targets, we lower some target intrinsics to a MemIntrinsicNode
    // with either an intrinsic or a target opcode.
    return N->getOpcode() == ISD::LOAD                ||
           N->getOpcode() == ISD::STORE               ||
           N->getOpcode() == ISD::PREFETCH            ||
           N->getOpcode() == ISD::ATOMIC_CMP_SWAP     ||
           N->getOpcode() == ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS ||
           N->getOpcode() == ISD::ATOMIC_SWAP         ||
           N->getOpcode() == ISD::ATOMIC_LOAD_ADD     ||
           N->getOpcode() == ISD::ATOMIC_LOAD_SUB     ||
           N->getOpcode() == ISD::ATOMIC_LOAD_AND     ||
           N->getOpcode() == ISD::ATOMIC_LOAD_OR      ||
           N->getOpcode() == ISD::ATOMIC_LOAD_XOR     ||
           N->getOpcode() == ISD::ATOMIC_LOAD_NAND    ||
           N->getOpcode() == ISD::ATOMIC_LOAD_MIN     ||
           N->getOpcode() == ISD::ATOMIC_LOAD_MAX     ||
           N->getOpcode() == ISD::ATOMIC_LOAD_UMIN    ||
           N->getOpcode() == ISD::ATOMIC_LOAD_UMAX    ||
           N->getOpcode() == ISD::ATOMIC_LOAD         ||
           N->getOpcode() == ISD::ATOMIC_STORE        ||
           N->getOpcode() == ISD::MLOAD               ||
           N->getOpcode() == ISD::MSTORE              ||
           N->getOpcode() == ISD::MGATHER             ||
           N->getOpcode() == ISD::MSCATTER            ||
           N->isMemIntrinsic()                        ||
           N->isTargetMemoryOpcode();
  }
};

/// This is an SDNode representing atomic operations.
class AtomicSDNode : public MemSDNode {
  SDUse Ops[4];

  /// For cmpxchg instructions, the ordering requirements when a store does not
  /// occur.
  AtomicOrdering FailureOrdering;

  void InitAtomic(AtomicOrdering SuccessOrdering,
                  AtomicOrdering FailureOrdering,
                  SynchronizationScope SynchScope) {
    // This must match encodeMemSDNodeFlags() in SelectionDAG.cpp.
    assert((SuccessOrdering & 15) == SuccessOrdering &&
           "Ordering may not require more than 4 bits!");
    assert((FailureOrdering & 15) == FailureOrdering &&
           "Ordering may not require more than 4 bits!");
    assert((SynchScope & 1) == SynchScope &&
           "SynchScope may not require more than 1 bit!");
    SubclassData |= SuccessOrdering << 8;
    SubclassData |= SynchScope << 12;
    this->FailureOrdering = FailureOrdering;
    assert(getSuccessOrdering() == SuccessOrdering &&
           "Ordering encoding error!");
    assert(getFailureOrdering() == FailureOrdering &&
           "Ordering encoding error!");
    assert(getSynchScope() == SynchScope && "Synch-scope encoding error!");
  }

public:
  // Opc:   opcode for atomic
  // VTL:    value type list
  // Chain:  memory chain for operaand
  // Ptr:    address to update as a SDValue
  // Cmp:    compare value
  // Swp:    swap value
  // SrcVal: address to update as a Value (used for MemOperand)
  // Align:  alignment of memory
  AtomicSDNode(unsigned Opc, unsigned Order, DebugLoc dl, SDVTList VTL,
               EVT MemVT, SDValue Chain, SDValue Ptr, SDValue Cmp, SDValue Swp,
               MachineMemOperand *MMO, AtomicOrdering Ordering,
               SynchronizationScope SynchScope)
      : MemSDNode(Opc, Order, dl, VTL, MemVT, MMO) {
    InitAtomic(Ordering, Ordering, SynchScope);
    InitOperands(Ops, Chain, Ptr, Cmp, Swp);
  }
  AtomicSDNode(unsigned Opc, unsigned Order, DebugLoc dl, SDVTList VTL,
               EVT MemVT,
               SDValue Chain, SDValue Ptr,
               SDValue Val, MachineMemOperand *MMO,
               AtomicOrdering Ordering, SynchronizationScope SynchScope)
    : MemSDNode(Opc, Order, dl, VTL, MemVT, MMO) {
    InitAtomic(Ordering, Ordering, SynchScope);
    InitOperands(Ops, Chain, Ptr, Val);
  }
  AtomicSDNode(unsigned Opc, unsigned Order, DebugLoc dl, SDVTList VTL,
               EVT MemVT,
               SDValue Chain, SDValue Ptr,
               MachineMemOperand *MMO,
               AtomicOrdering Ordering, SynchronizationScope SynchScope)
    : MemSDNode(Opc, Order, dl, VTL, MemVT, MMO) {
    InitAtomic(Ordering, Ordering, SynchScope);
    InitOperands(Ops, Chain, Ptr);
  }
  AtomicSDNode(unsigned Opc, unsigned Order, DebugLoc dl, SDVTList VTL, EVT MemVT,
               const SDValue* AllOps, SDUse *DynOps, unsigned NumOps,
               MachineMemOperand *MMO,
               AtomicOrdering SuccessOrdering, AtomicOrdering FailureOrdering,
               SynchronizationScope SynchScope)
    : MemSDNode(Opc, Order, dl, VTL, MemVT, MMO) {
    InitAtomic(SuccessOrdering, FailureOrdering, SynchScope);
    assert((DynOps || NumOps <= array_lengthof(Ops)) &&
           "Too many ops for internal storage!");
    InitOperands(DynOps ? DynOps : Ops, AllOps, NumOps);
  }

  const SDValue &getBasePtr() const { return getOperand(1); }
  const SDValue &getVal() const { return getOperand(2); }

  AtomicOrdering getSuccessOrdering() const {
    return getOrdering();
  }

  // Not quite enough room in SubclassData for everything, so failure gets its
  // own field.
  AtomicOrdering getFailureOrdering() const {
    return FailureOrdering;
  }

  bool isCompareAndSwap() const {
    unsigned Op = getOpcode();
    return Op == ISD::ATOMIC_CMP_SWAP || Op == ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS;
  }

  // Methods to support isa and dyn_cast
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::ATOMIC_CMP_SWAP     ||
           N->getOpcode() == ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS ||
           N->getOpcode() == ISD::ATOMIC_SWAP         ||
           N->getOpcode() == ISD::ATOMIC_LOAD_ADD     ||
           N->getOpcode() == ISD::ATOMIC_LOAD_SUB     ||
           N->getOpcode() == ISD::ATOMIC_LOAD_AND     ||
           N->getOpcode() == ISD::ATOMIC_LOAD_OR      ||
           N->getOpcode() == ISD::ATOMIC_LOAD_XOR     ||
           N->getOpcode() == ISD::ATOMIC_LOAD_NAND    ||
           N->getOpcode() == ISD::ATOMIC_LOAD_MIN     ||
           N->getOpcode() == ISD::ATOMIC_LOAD_MAX     ||
           N->getOpcode() == ISD::ATOMIC_LOAD_UMIN    ||
           N->getOpcode() == ISD::ATOMIC_LOAD_UMAX    ||
           N->getOpcode() == ISD::ATOMIC_LOAD         ||
           N->getOpcode() == ISD::ATOMIC_STORE;
  }
};

/// This SDNode is used for target intrinsics that touch
/// memory and need an associated MachineMemOperand. Its opcode may be
/// INTRINSIC_VOID, INTRINSIC_W_CHAIN, PREFETCH, or a target-specific opcode
/// with a value not less than FIRST_TARGET_MEMORY_OPCODE.
class MemIntrinsicSDNode : public MemSDNode {
public:
  MemIntrinsicSDNode(unsigned Opc, unsigned Order, DebugLoc dl, SDVTList VTs,
                     ArrayRef<SDValue> Ops, EVT MemoryVT,
                     MachineMemOperand *MMO)
    : MemSDNode(Opc, Order, dl, VTs, Ops, MemoryVT, MMO) {
    SubclassData |= 1u << 13;
  }

  // Methods to support isa and dyn_cast
  static bool classof(const SDNode *N) {
    // We lower some target intrinsics to their target opcode
    // early a node with a target opcode can be of this class
    return N->isMemIntrinsic()             ||
           N->getOpcode() == ISD::PREFETCH ||
           N->isTargetMemoryOpcode();
  }
};

/// This SDNode is used to implement the code generator
/// support for the llvm IR shufflevector instruction.  It combines elements
/// from two input vectors into a new input vector, with the selection and
/// ordering of elements determined by an array of integers, referred to as
/// the shuffle mask.  For input vectors of width N, mask indices of 0..N-1
/// refer to elements from the LHS input, and indices from N to 2N-1 the RHS.
/// An index of -1 is treated as undef, such that the code generator may put
/// any value in the corresponding element of the result.
class ShuffleVectorSDNode : public SDNode {
  SDUse Ops[2];

  // The memory for Mask is owned by the SelectionDAG's OperandAllocator, and
  // is freed when the SelectionDAG object is destroyed.
  const int *Mask;
protected:
  friend class SelectionDAG;
  ShuffleVectorSDNode(EVT VT, unsigned Order, DebugLoc dl, SDValue N1,
                      SDValue N2, const int *M)
    : SDNode(ISD::VECTOR_SHUFFLE, Order, dl, getSDVTList(VT)), Mask(M) {
    InitOperands(Ops, N1, N2);
  }
public:

  ArrayRef<int> getMask() const {
    EVT VT = getValueType(0);
    return makeArrayRef(Mask, VT.getVectorNumElements());
  }
  int getMaskElt(unsigned Idx) const {
    assert(Idx < getValueType(0).getVectorNumElements() && "Idx out of range!");
    return Mask[Idx];
  }

  bool isSplat() const { return isSplatMask(Mask, getValueType(0)); }
  int  getSplatIndex() const {
    assert(isSplat() && "Cannot get splat index for non-splat!");
    EVT VT = getValueType(0);
    for (unsigned i = 0, e = VT.getVectorNumElements(); i != e; ++i) {
      if (Mask[i] >= 0)
        return Mask[i];
    }
    llvm_unreachable("Splat with all undef indices?");
  }
  static bool isSplatMask(const int *Mask, EVT VT);

  /// Change values in a shuffle permute mask assuming
  /// the two vector operands have swapped position.
  static void commuteMask(SmallVectorImpl<int> &Mask) {
    unsigned NumElems = Mask.size();
    for (unsigned i = 0; i != NumElems; ++i) {
      int idx = Mask[i];
      if (idx < 0)
        continue;
      else if (idx < (int)NumElems)
        Mask[i] = idx + NumElems;
      else
        Mask[i] = idx - NumElems;
    }
  }

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::VECTOR_SHUFFLE;
  }
};

class ConstantSDNode : public SDNode {
  const ConstantInt *Value;
  friend class SelectionDAG;
  ConstantSDNode(bool isTarget, bool isOpaque, const ConstantInt *val,
                 DebugLoc DL, EVT VT)
    : SDNode(isTarget ? ISD::TargetConstant : ISD::Constant,
             0, DL, getSDVTList(VT)), Value(val) {
    SubclassData |= (uint16_t)isOpaque;
  }
public:

  const ConstantInt *getConstantIntValue() const { return Value; }
  const APInt &getAPIntValue() const { return Value->getValue(); }
  uint64_t getZExtValue() const { return Value->getZExtValue(); }
  int64_t getSExtValue() const { return Value->getSExtValue(); }

  bool isOne() const { return Value->isOne(); }
  bool isNullValue() const { return Value->isNullValue(); }
  bool isAllOnesValue() const { return Value->isAllOnesValue(); }

  bool isOpaque() const { return SubclassData & 1; }

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::Constant ||
           N->getOpcode() == ISD::TargetConstant;
  }
};

class ConstantFPSDNode : public SDNode {
  const ConstantFP *Value;
  friend class SelectionDAG;
  ConstantFPSDNode(bool isTarget, const ConstantFP *val, DebugLoc DL, EVT VT)
    : SDNode(isTarget ? ISD::TargetConstantFP : ISD::ConstantFP,
             0, DL, getSDVTList(VT)), Value(val) {
  }
public:

  const APFloat& getValueAPF() const { return Value->getValueAPF(); }
  const ConstantFP *getConstantFPValue() const { return Value; }

  /// Return true if the value is positive or negative zero.
  bool isZero() const { return Value->isZero(); }

  /// Return true if the value is a NaN.
  bool isNaN() const { return Value->isNaN(); }

  /// Return true if the value is an infinity
  bool isInfinity() const { return Value->isInfinity(); }

  /// Return true if the value is negative.
  bool isNegative() const { return Value->isNegative(); }

  /// We don't rely on operator== working on double values, as
  /// it returns true for things that are clearly not equal, like -0.0 and 0.0.
  /// As such, this method can be used to do an exact bit-for-bit comparison of
  /// two floating point values.

  /// We leave the version with the double argument here because it's just so
  /// convenient to write "2.0" and the like.  Without this function we'd
  /// have to duplicate its logic everywhere it's called.
  bool isExactlyValue(double V) const {
    bool ignored;
    APFloat Tmp(V);
    Tmp.convert(Value->getValueAPF().getSemantics(),
                APFloat::rmNearestTiesToEven, &ignored);
    return isExactlyValue(Tmp);
  }
  bool isExactlyValue(const APFloat& V) const;

  static bool isValueValidForType(EVT VT, const APFloat& Val);

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::ConstantFP ||
           N->getOpcode() == ISD::TargetConstantFP;
  }
};

class GlobalAddressSDNode : public SDNode {
  const GlobalValue *TheGlobal;
  int64_t Offset;
  unsigned char TargetFlags;
  friend class SelectionDAG;
  GlobalAddressSDNode(unsigned Opc, unsigned Order, DebugLoc DL,
                      const GlobalValue *GA, EVT VT, int64_t o,
                      unsigned char TargetFlags);
public:

  const GlobalValue *getGlobal() const { return TheGlobal; }
  int64_t getOffset() const { return Offset; }
  unsigned char getTargetFlags() const { return TargetFlags; }
  // Return the address space this GlobalAddress belongs to.
  unsigned getAddressSpace() const;

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::GlobalAddress ||
           N->getOpcode() == ISD::TargetGlobalAddress ||
           N->getOpcode() == ISD::GlobalTLSAddress ||
           N->getOpcode() == ISD::TargetGlobalTLSAddress;
  }
};

class FrameIndexSDNode : public SDNode {
  int FI;
  friend class SelectionDAG;
  FrameIndexSDNode(int fi, EVT VT, bool isTarg)
    : SDNode(isTarg ? ISD::TargetFrameIndex : ISD::FrameIndex,
      0, DebugLoc(), getSDVTList(VT)), FI(fi) {
  }
public:

  int getIndex() const { return FI; }

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::FrameIndex ||
           N->getOpcode() == ISD::TargetFrameIndex;
  }
};

class JumpTableSDNode : public SDNode {
  int JTI;
  unsigned char TargetFlags;
  friend class SelectionDAG;
  JumpTableSDNode(int jti, EVT VT, bool isTarg, unsigned char TF)
    : SDNode(isTarg ? ISD::TargetJumpTable : ISD::JumpTable,
      0, DebugLoc(), getSDVTList(VT)), JTI(jti), TargetFlags(TF) {
  }
public:

  int getIndex() const { return JTI; }
  unsigned char getTargetFlags() const { return TargetFlags; }

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::JumpTable ||
           N->getOpcode() == ISD::TargetJumpTable;
  }
};

class ConstantPoolSDNode : public SDNode {
  union {
    const Constant *ConstVal;
    MachineConstantPoolValue *MachineCPVal;
  } Val;
  int Offset;  // It's a MachineConstantPoolValue if top bit is set.
  unsigned Alignment;  // Minimum alignment requirement of CP (not log2 value).
  unsigned char TargetFlags;
  friend class SelectionDAG;
  ConstantPoolSDNode(bool isTarget, const Constant *c, EVT VT, int o,
                     unsigned Align, unsigned char TF)
    : SDNode(isTarget ? ISD::TargetConstantPool : ISD::ConstantPool, 0,
             DebugLoc(), getSDVTList(VT)), Offset(o), Alignment(Align),
             TargetFlags(TF) {
    assert(Offset >= 0 && "Offset is too large");
    Val.ConstVal = c;
  }
  ConstantPoolSDNode(bool isTarget, MachineConstantPoolValue *v,
                     EVT VT, int o, unsigned Align, unsigned char TF)
    : SDNode(isTarget ? ISD::TargetConstantPool : ISD::ConstantPool, 0,
             DebugLoc(), getSDVTList(VT)), Offset(o), Alignment(Align),
             TargetFlags(TF) {
    assert(Offset >= 0 && "Offset is too large");
    Val.MachineCPVal = v;
    Offset |= 1 << (sizeof(unsigned)*CHAR_BIT-1);
  }
public:

  bool isMachineConstantPoolEntry() const {
    return Offset < 0;
  }

  const Constant *getConstVal() const {
    assert(!isMachineConstantPoolEntry() && "Wrong constantpool type");
    return Val.ConstVal;
  }

  MachineConstantPoolValue *getMachineCPVal() const {
    assert(isMachineConstantPoolEntry() && "Wrong constantpool type");
    return Val.MachineCPVal;
  }

  int getOffset() const {
    return Offset & ~(1 << (sizeof(unsigned)*CHAR_BIT-1));
  }

  // Return the alignment of this constant pool object, which is either 0 (for
  // default alignment) or the desired value.
  unsigned getAlignment() const { return Alignment; }
  unsigned char getTargetFlags() const { return TargetFlags; }

  Type *getType() const;

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::ConstantPool ||
           N->getOpcode() == ISD::TargetConstantPool;
  }
};

/// Completely target-dependent object reference.
class TargetIndexSDNode : public SDNode {
  unsigned char TargetFlags;
  int Index;
  int64_t Offset;
  friend class SelectionDAG;
public:

  TargetIndexSDNode(int Idx, EVT VT, int64_t Ofs, unsigned char TF)
    : SDNode(ISD::TargetIndex, 0, DebugLoc(), getSDVTList(VT)),
      TargetFlags(TF), Index(Idx), Offset(Ofs) {}
public:

  unsigned char getTargetFlags() const { return TargetFlags; }
  int getIndex() const { return Index; }
  int64_t getOffset() const { return Offset; }

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::TargetIndex;
  }
};

class BasicBlockSDNode : public SDNode {
  MachineBasicBlock *MBB;
  friend class SelectionDAG;
  /// Debug info is meaningful and potentially useful here, but we create
  /// blocks out of order when they're jumped to, which makes it a bit
  /// harder.  Let's see if we need it first.
  explicit BasicBlockSDNode(MachineBasicBlock *mbb)
    : SDNode(ISD::BasicBlock, 0, DebugLoc(), getSDVTList(MVT::Other)), MBB(mbb)
  {}
public:

  MachineBasicBlock *getBasicBlock() const { return MBB; }

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::BasicBlock;
  }
};

/// A "pseudo-class" with methods for operating on BUILD_VECTORs.
class BuildVectorSDNode : public SDNode {
  // These are constructed as SDNodes and then cast to BuildVectorSDNodes.
  explicit BuildVectorSDNode() = delete;
public:
  /// Check if this is a constant splat, and if so, find the
  /// smallest element size that splats the vector.  If MinSplatBits is
  /// nonzero, the element size must be at least that large.  Note that the
  /// splat element may be the entire vector (i.e., a one element vector).
  /// Returns the splat element value in SplatValue.  Any undefined bits in
  /// that value are zero, and the corresponding bits in the SplatUndef mask
  /// are set.  The SplatBitSize value is set to the splat element size in
  /// bits.  HasAnyUndefs is set to true if any bits in the vector are
  /// undefined.  isBigEndian describes the endianness of the target.
  bool isConstantSplat(APInt &SplatValue, APInt &SplatUndef,
                       unsigned &SplatBitSize, bool &HasAnyUndefs,
                       unsigned MinSplatBits = 0,
                       bool isBigEndian = false) const;

  /// \brief Returns the splatted value or a null value if this is not a splat.
  ///
  /// If passed a non-null UndefElements bitvector, it will resize it to match
  /// the vector width and set the bits where elements are undef.
  SDValue getSplatValue(BitVector *UndefElements = nullptr) const;

  /// \brief Returns the splatted constant or null if this is not a constant
  /// splat.
  ///
  /// If passed a non-null UndefElements bitvector, it will resize it to match
  /// the vector width and set the bits where elements are undef.
  ConstantSDNode *
  getConstantSplatNode(BitVector *UndefElements = nullptr) const;

  /// \brief Returns the splatted constant FP or null if this is not a constant
  /// FP splat.
  ///
  /// If passed a non-null UndefElements bitvector, it will resize it to match
  /// the vector width and set the bits where elements are undef.
  ConstantFPSDNode *
  getConstantFPSplatNode(BitVector *UndefElements = nullptr) const;

  bool isConstant() const;

  static inline bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::BUILD_VECTOR;
  }
};

/// An SDNode that holds an arbitrary LLVM IR Value. This is
/// used when the SelectionDAG needs to make a simple reference to something
/// in the LLVM IR representation.
///
class SrcValueSDNode : public SDNode {
  const Value *V;
  friend class SelectionDAG;
  /// Create a SrcValue for a general value.
  explicit SrcValueSDNode(const Value *v)
    : SDNode(ISD::SRCVALUE, 0, DebugLoc(), getSDVTList(MVT::Other)), V(v) {}

public:
  /// Return the contained Value.
  const Value *getValue() const { return V; }

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::SRCVALUE;
  }
};

class MDNodeSDNode : public SDNode {
  const MDNode *MD;
  friend class SelectionDAG;
  explicit MDNodeSDNode(const MDNode *md)
  : SDNode(ISD::MDNODE_SDNODE, 0, DebugLoc(), getSDVTList(MVT::Other)), MD(md)
  {}
public:

  const MDNode *getMD() const { return MD; }

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::MDNODE_SDNODE;
  }
};

class RegisterSDNode : public SDNode {
  unsigned Reg;
  friend class SelectionDAG;
  RegisterSDNode(unsigned reg, EVT VT)
    : SDNode(ISD::Register, 0, DebugLoc(), getSDVTList(VT)), Reg(reg) {
  }
public:

  unsigned getReg() const { return Reg; }

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::Register;
  }
};

class RegisterMaskSDNode : public SDNode {
  // The memory for RegMask is not owned by the node.
  const uint32_t *RegMask;
  friend class SelectionDAG;
  RegisterMaskSDNode(const uint32_t *mask)
    : SDNode(ISD::RegisterMask, 0, DebugLoc(), getSDVTList(MVT::Untyped)),
      RegMask(mask) {}
public:

  const uint32_t *getRegMask() const { return RegMask; }

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::RegisterMask;
  }
};

class BlockAddressSDNode : public SDNode {
  const BlockAddress *BA;
  int64_t Offset;
  unsigned char TargetFlags;
  friend class SelectionDAG;
  BlockAddressSDNode(unsigned NodeTy, EVT VT, const BlockAddress *ba,
                     int64_t o, unsigned char Flags)
    : SDNode(NodeTy, 0, DebugLoc(), getSDVTList(VT)),
             BA(ba), Offset(o), TargetFlags(Flags) {
  }
public:
  const BlockAddress *getBlockAddress() const { return BA; }
  int64_t getOffset() const { return Offset; }
  unsigned char getTargetFlags() const { return TargetFlags; }

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::BlockAddress ||
           N->getOpcode() == ISD::TargetBlockAddress;
  }
};

class EHLabelSDNode : public SDNode {
  SDUse Chain;
  MCSymbol *Label;
  friend class SelectionDAG;
  EHLabelSDNode(unsigned Order, DebugLoc dl, SDValue ch, MCSymbol *L)
    : SDNode(ISD::EH_LABEL, Order, dl, getSDVTList(MVT::Other)), Label(L) {
    InitOperands(&Chain, ch);
  }
public:
  MCSymbol *getLabel() const { return Label; }

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::EH_LABEL;
  }
};

class ExternalSymbolSDNode : public SDNode {
  const char *Symbol;
  unsigned char TargetFlags;

  friend class SelectionDAG;
  ExternalSymbolSDNode(bool isTarget, const char *Sym, unsigned char TF, EVT VT)
    : SDNode(isTarget ? ISD::TargetExternalSymbol : ISD::ExternalSymbol,
             0, DebugLoc(), getSDVTList(VT)), Symbol(Sym), TargetFlags(TF) {
  }
public:

  const char *getSymbol() const { return Symbol; }
  unsigned char getTargetFlags() const { return TargetFlags; }

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::ExternalSymbol ||
           N->getOpcode() == ISD::TargetExternalSymbol;
  }
};

class MCSymbolSDNode : public SDNode {
  MCSymbol *Symbol;

  friend class SelectionDAG;
  MCSymbolSDNode(MCSymbol *Symbol, EVT VT)
      : SDNode(ISD::MCSymbol, 0, DebugLoc(), getSDVTList(VT)), Symbol(Symbol) {}

public:
  MCSymbol *getMCSymbol() const { return Symbol; }

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::MCSymbol;
  }
};

class CondCodeSDNode : public SDNode {
  ISD::CondCode Condition;
  friend class SelectionDAG;
  explicit CondCodeSDNode(ISD::CondCode Cond)
    : SDNode(ISD::CONDCODE, 0, DebugLoc(), getSDVTList(MVT::Other)),
      Condition(Cond) {
  }
public:

  ISD::CondCode get() const { return Condition; }

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::CONDCODE;
  }
};

/// NOTE: avoid using this node as this may disappear in the
/// future and most targets don't support it.
class CvtRndSatSDNode : public SDNode {
  ISD::CvtCode CvtCode;
  friend class SelectionDAG;
  explicit CvtRndSatSDNode(EVT VT, unsigned Order, DebugLoc dl,
                           ArrayRef<SDValue> Ops, ISD::CvtCode Code)
    : SDNode(ISD::CONVERT_RNDSAT, Order, dl, getSDVTList(VT), Ops),
      CvtCode(Code) {
    assert(Ops.size() == 5 && "wrong number of operations");
  }
public:
  ISD::CvtCode getCvtCode() const { return CvtCode; }

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::CONVERT_RNDSAT;
  }
};

/// This class is used to represent EVT's, which are used
/// to parameterize some operations.
class VTSDNode : public SDNode {
  EVT ValueType;
  friend class SelectionDAG;
  explicit VTSDNode(EVT VT)
    : SDNode(ISD::VALUETYPE, 0, DebugLoc(), getSDVTList(MVT::Other)),
      ValueType(VT) {
  }
public:

  EVT getVT() const { return ValueType; }

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::VALUETYPE;
  }
};

/// Base class for LoadSDNode and StoreSDNode
class LSBaseSDNode : public MemSDNode {
  //! Operand array for load and store
  /*!
    \note Moving this array to the base class captures more
    common functionality shared between LoadSDNode and
    StoreSDNode
   */
  SDUse Ops[4];
public:
  LSBaseSDNode(ISD::NodeType NodeTy, unsigned Order, DebugLoc dl,
               SDValue *Operands, unsigned numOperands,
               SDVTList VTs, ISD::MemIndexedMode AM, EVT MemVT,
               MachineMemOperand *MMO)
    : MemSDNode(NodeTy, Order, dl, VTs, MemVT, MMO) {
    SubclassData |= AM << 2;
    assert(getAddressingMode() == AM && "MemIndexedMode encoding error!");
    InitOperands(Ops, Operands, numOperands);
    assert((getOffset().getOpcode() == ISD::UNDEF || isIndexed()) &&
           "Only indexed loads and stores have a non-undef offset operand");
  }

  const SDValue &getOffset() const {
    return getOperand(getOpcode() == ISD::LOAD ? 2 : 3);
  }

  /// Return the addressing mode for this load or store:
  /// unindexed, pre-inc, pre-dec, post-inc, or post-dec.
  ISD::MemIndexedMode getAddressingMode() const {
    return ISD::MemIndexedMode((SubclassData >> 2) & 7);
  }

  /// Return true if this is a pre/post inc/dec load/store.
  bool isIndexed() const { return getAddressingMode() != ISD::UNINDEXED; }

  /// Return true if this is NOT a pre/post inc/dec load/store.
  bool isUnindexed() const { return getAddressingMode() == ISD::UNINDEXED; }

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::LOAD ||
           N->getOpcode() == ISD::STORE;
  }
};

/// This class is used to represent ISD::LOAD nodes.
class LoadSDNode : public LSBaseSDNode {
  friend class SelectionDAG;
  LoadSDNode(SDValue *ChainPtrOff, unsigned Order, DebugLoc dl, SDVTList VTs,
             ISD::MemIndexedMode AM, ISD::LoadExtType ETy, EVT MemVT,
             MachineMemOperand *MMO)
    : LSBaseSDNode(ISD::LOAD, Order, dl, ChainPtrOff, 3, VTs, AM, MemVT, MMO) {
    SubclassData |= (unsigned short)ETy;
    assert(getExtensionType() == ETy && "LoadExtType encoding error!");
    assert(readMem() && "Load MachineMemOperand is not a load!");
    assert(!writeMem() && "Load MachineMemOperand is a store!");
  }
public:

  /// Return whether this is a plain node,
  /// or one of the varieties of value-extending loads.
  ISD::LoadExtType getExtensionType() const {
    return ISD::LoadExtType(SubclassData & 3);
  }

  const SDValue &getBasePtr() const { return getOperand(1); }
  const SDValue &getOffset() const { return getOperand(2); }

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::LOAD;
  }
};

/// This class is used to represent ISD::STORE nodes.
class StoreSDNode : public LSBaseSDNode {
  friend class SelectionDAG;
  StoreSDNode(SDValue *ChainValuePtrOff, unsigned Order, DebugLoc dl,
              SDVTList VTs, ISD::MemIndexedMode AM, bool isTrunc, EVT MemVT,
              MachineMemOperand *MMO)
    : LSBaseSDNode(ISD::STORE, Order, dl, ChainValuePtrOff, 4,
                   VTs, AM, MemVT, MMO) {
    SubclassData |= (unsigned short)isTrunc;
    assert(isTruncatingStore() == isTrunc && "isTrunc encoding error!");
    assert(!readMem() && "Store MachineMemOperand is a load!");
    assert(writeMem() && "Store MachineMemOperand is not a store!");
  }
public:

  /// Return true if the op does a truncation before store.
  /// For integers this is the same as doing a TRUNCATE and storing the result.
  /// For floats, it is the same as doing an FP_ROUND and storing the result.
  bool isTruncatingStore() const { return SubclassData & 1; }

  const SDValue &getValue() const { return getOperand(1); }
  const SDValue &getBasePtr() const { return getOperand(2); }
  const SDValue &getOffset() const { return getOperand(3); }

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::STORE;
  }
};

/// This base class is used to represent MLOAD and MSTORE nodes
class MaskedLoadStoreSDNode : public MemSDNode {
  // Operands
  SDUse Ops[4];
public:
  friend class SelectionDAG;
  MaskedLoadStoreSDNode(ISD::NodeType NodeTy, unsigned Order, DebugLoc dl,
                   SDValue *Operands, unsigned numOperands, 
                   SDVTList VTs, EVT MemVT, MachineMemOperand *MMO)
    : MemSDNode(NodeTy, Order, dl, VTs, MemVT, MMO) {
    InitOperands(Ops, Operands, numOperands);
  }

  // In the both nodes address is Op1, mask is Op2:
  // MaskedLoadSDNode (Chain, ptr, mask, src0), src0 is a passthru value
  // MaskedStoreSDNode (Chain, ptr, mask, data)
  // Mask is a vector of i1 elements
  const SDValue &getBasePtr() const { return getOperand(1); }
  const SDValue &getMask() const    { return getOperand(2); }

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::MLOAD ||
           N->getOpcode() == ISD::MSTORE;
  }
};

/// This class is used to represent an MLOAD node
class MaskedLoadSDNode : public MaskedLoadStoreSDNode {
public:
  friend class SelectionDAG;
  MaskedLoadSDNode(unsigned Order, DebugLoc dl, SDValue *Operands,
                   unsigned numOperands, SDVTList VTs, ISD::LoadExtType ETy,
                   EVT MemVT, MachineMemOperand *MMO)
    : MaskedLoadStoreSDNode(ISD::MLOAD, Order, dl, Operands, numOperands,
                            VTs, MemVT, MMO) {
    SubclassData |= (unsigned short)ETy;
  }

  ISD::LoadExtType getExtensionType() const {
    return ISD::LoadExtType(SubclassData & 3);
  } 
  const SDValue &getSrc0() const { return getOperand(3); }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::MLOAD;
  }
};

/// This class is used to represent an MSTORE node
class MaskedStoreSDNode : public MaskedLoadStoreSDNode {

public:
  friend class SelectionDAG;
  MaskedStoreSDNode(unsigned Order, DebugLoc dl, SDValue *Operands,
                    unsigned numOperands, SDVTList VTs, bool isTrunc, EVT MemVT,
                    MachineMemOperand *MMO)
    : MaskedLoadStoreSDNode(ISD::MSTORE, Order, dl, Operands, numOperands,
                            VTs, MemVT, MMO) {
      SubclassData |= (unsigned short)isTrunc;
  }
  /// Return true if the op does a truncation before store.
  /// For integers this is the same as doing a TRUNCATE and storing the result.
  /// For floats, it is the same as doing an FP_ROUND and storing the result.
  bool isTruncatingStore() const { return SubclassData & 1; }

  const SDValue &getValue() const { return getOperand(3); }

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::MSTORE;
  }
};

/// This is a base class used to represent
/// MGATHER and MSCATTER nodes
///
class MaskedGatherScatterSDNode : public MemSDNode {
  // Operands
  SDUse Ops[5];
public:
  friend class SelectionDAG;
  MaskedGatherScatterSDNode(ISD::NodeType NodeTy, unsigned Order, DebugLoc dl,
                            ArrayRef<SDValue> Operands, SDVTList VTs, EVT MemVT,
                            MachineMemOperand *MMO)
    : MemSDNode(NodeTy, Order, dl, VTs, MemVT, MMO) {
    assert(Operands.size() == 5 && "Incompatible number of operands");
    InitOperands(Ops, Operands.data(), Operands.size());
  }

  // In the both nodes address is Op1, mask is Op2:
  // MaskedGatherSDNode  (Chain, src0, mask, base, index), src0 is a passthru value
  // MaskedScatterSDNode (Chain, value, mask, base, index)
  // Mask is a vector of i1 elements
  const SDValue &getBasePtr() const { return getOperand(3); }
  const SDValue &getIndex()   const { return getOperand(4); }
  const SDValue &getMask()    const { return getOperand(2); }
  const SDValue &getValue()   const { return getOperand(1); }

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::MGATHER ||
           N->getOpcode() == ISD::MSCATTER;
  }
};

/// This class is used to represent an MGATHER node
///
class MaskedGatherSDNode : public MaskedGatherScatterSDNode {
public:
  friend class SelectionDAG;
  MaskedGatherSDNode(unsigned Order, DebugLoc dl, ArrayRef<SDValue> Operands, 
                     SDVTList VTs, EVT MemVT, MachineMemOperand *MMO)
    : MaskedGatherScatterSDNode(ISD::MGATHER, Order, dl, Operands, VTs, MemVT,
                                MMO) {
    assert(getValue().getValueType() == getValueType(0) &&
           "Incompatible type of the PathThru value in MaskedGatherSDNode");
    assert(getMask().getValueType().getVectorNumElements() == 
           getValueType(0).getVectorNumElements() && 
           "Vector width mismatch between mask and data");
    assert(getMask().getValueType().getScalarType() == MVT::i1 && 
           "Vector width mismatch between mask and data");
  }

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::MGATHER;
  }
};

/// This class is used to represent an MSCATTER node
///
class MaskedScatterSDNode : public MaskedGatherScatterSDNode {

public:
  friend class SelectionDAG;
  MaskedScatterSDNode(unsigned Order, DebugLoc dl,ArrayRef<SDValue> Operands,
                      SDVTList VTs, EVT MemVT, MachineMemOperand *MMO)
    : MaskedGatherScatterSDNode(ISD::MSCATTER, Order, dl, Operands, VTs, MemVT,
                                MMO) {
    assert(getMask().getValueType().getVectorNumElements() == 
           getValue().getValueType().getVectorNumElements() && 
           "Vector width mismatch between mask and data");
    assert(getMask().getValueType().getScalarType() == MVT::i1 && 
           "Vector width mismatch between mask and data");
  }

  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::MSCATTER;
  }
};

/// An SDNode that represents everything that will be needed
/// to construct a MachineInstr. These nodes are created during the
/// instruction selection proper phase.
class MachineSDNode : public SDNode {
public:
  typedef MachineMemOperand **mmo_iterator;

private:
  friend class SelectionDAG;
  MachineSDNode(unsigned Opc, unsigned Order, const DebugLoc DL, SDVTList VTs)
    : SDNode(Opc, Order, DL, VTs), MemRefs(nullptr), MemRefsEnd(nullptr) {}

  /// Operands for this instruction, if they fit here. If
  /// they don't, this field is unused.
  SDUse LocalOperands[4];

  /// Memory reference descriptions for this instruction.
  mmo_iterator MemRefs;
  mmo_iterator MemRefsEnd;

public:
  mmo_iterator memoperands_begin() const { return MemRefs; }
  mmo_iterator memoperands_end() const { return MemRefsEnd; }
  bool memoperands_empty() const { return MemRefsEnd == MemRefs; }

  /// Assign this MachineSDNodes's memory reference descriptor
  /// list. This does not transfer ownership.
  void setMemRefs(mmo_iterator NewMemRefs, mmo_iterator NewMemRefsEnd) {
    for (mmo_iterator MMI = NewMemRefs, MME = NewMemRefsEnd; MMI != MME; ++MMI)
      assert(*MMI && "Null mem ref detected!");
    MemRefs = NewMemRefs;
    MemRefsEnd = NewMemRefsEnd;
  }

  static bool classof(const SDNode *N) {
    return N->isMachineOpcode();
  }
};

class SDNodeIterator : public std::iterator<std::forward_iterator_tag,
                                            SDNode, ptrdiff_t> {
  const SDNode *Node;
  unsigned Operand;

  SDNodeIterator(const SDNode *N, unsigned Op) : Node(N), Operand(Op) {}
public:
  bool operator==(const SDNodeIterator& x) const {
    return Operand == x.Operand;
  }
  bool operator!=(const SDNodeIterator& x) const { return !operator==(x); }

  pointer operator*() const {
    return Node->getOperand(Operand).getNode();
  }
  pointer operator->() const { return operator*(); }

  SDNodeIterator& operator++() {                // Preincrement
    ++Operand;
    return *this;
  }
  SDNodeIterator operator++(int) { // Postincrement
    SDNodeIterator tmp = *this; ++*this; return tmp;
  }
  size_t operator-(SDNodeIterator Other) const {
    assert(Node == Other.Node &&
           "Cannot compare iterators of two different nodes!");
    return Operand - Other.Operand;
  }

  static SDNodeIterator begin(const SDNode *N) { return SDNodeIterator(N, 0); }
  static SDNodeIterator end  (const SDNode *N) {
    return SDNodeIterator(N, N->getNumOperands());
  }

  unsigned getOperand() const { return Operand; }
  const SDNode *getNode() const { return Node; }
};

template <> struct GraphTraits<SDNode*> {
  typedef SDNode NodeType;
  typedef SDNodeIterator ChildIteratorType;
  static inline NodeType *getEntryNode(SDNode *N) { return N; }
  static inline ChildIteratorType child_begin(NodeType *N) {
    return SDNodeIterator::begin(N);
  }
  static inline ChildIteratorType child_end(NodeType *N) {
    return SDNodeIterator::end(N);
  }
};

/// The largest SDNode class.
typedef MaskedGatherScatterSDNode LargestSDNode;

/// The SDNode class with the greatest alignment requirement.
typedef GlobalAddressSDNode MostAlignedSDNode;

namespace ISD {
  /// Returns true if the specified node is a non-extending and unindexed load.
  inline bool isNormalLoad(const SDNode *N) {
    const LoadSDNode *Ld = dyn_cast<LoadSDNode>(N);
    return Ld && Ld->getExtensionType() == ISD::NON_EXTLOAD &&
      Ld->getAddressingMode() == ISD::UNINDEXED;
  }

  /// Returns true if the specified node is a non-extending load.
  inline bool isNON_EXTLoad(const SDNode *N) {
    return isa<LoadSDNode>(N) &&
      cast<LoadSDNode>(N)->getExtensionType() == ISD::NON_EXTLOAD;
  }

  /// Returns true if the specified node is a EXTLOAD.
  inline bool isEXTLoad(const SDNode *N) {
    return isa<LoadSDNode>(N) &&
      cast<LoadSDNode>(N)->getExtensionType() == ISD::EXTLOAD;
  }

  /// Returns true if the specified node is a SEXTLOAD.
  inline bool isSEXTLoad(const SDNode *N) {
    return isa<LoadSDNode>(N) &&
      cast<LoadSDNode>(N)->getExtensionType() == ISD::SEXTLOAD;
  }

  /// Returns true if the specified node is a ZEXTLOAD.
  inline bool isZEXTLoad(const SDNode *N) {
    return isa<LoadSDNode>(N) &&
      cast<LoadSDNode>(N)->getExtensionType() == ISD::ZEXTLOAD;
  }

  /// Returns true if the specified node is an unindexed load.
  inline bool isUNINDEXEDLoad(const SDNode *N) {
    return isa<LoadSDNode>(N) &&
      cast<LoadSDNode>(N)->getAddressingMode() == ISD::UNINDEXED;
  }

  /// Returns true if the specified node is a non-truncating
  /// and unindexed store.
  inline bool isNormalStore(const SDNode *N) {
    const StoreSDNode *St = dyn_cast<StoreSDNode>(N);
    return St && !St->isTruncatingStore() &&
      St->getAddressingMode() == ISD::UNINDEXED;
  }

  /// Returns true if the specified node is a non-truncating store.
  inline bool isNON_TRUNCStore(const SDNode *N) {
    return isa<StoreSDNode>(N) && !cast<StoreSDNode>(N)->isTruncatingStore();
  }

  /// Returns true if the specified node is a truncating store.
  inline bool isTRUNCStore(const SDNode *N) {
    return isa<StoreSDNode>(N) && cast<StoreSDNode>(N)->isTruncatingStore();
  }

  /// Returns true if the specified node is an unindexed store.
  inline bool isUNINDEXEDStore(const SDNode *N) {
    return isa<StoreSDNode>(N) &&
      cast<StoreSDNode>(N)->getAddressingMode() == ISD::UNINDEXED;
  }
}

} // end llvm namespace

#endif
