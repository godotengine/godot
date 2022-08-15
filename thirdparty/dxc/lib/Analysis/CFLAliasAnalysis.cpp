//===- CFLAliasAnalysis.cpp - CFL-Based Alias Analysis Implementation ------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a CFL-based context-insensitive alias analysis
// algorithm. It does not depend on types. The algorithm is a mixture of the one
// described in "Demand-driven alias analysis for C" by Xin Zheng and Radu
// Rugina, and "Fast algorithms for Dyck-CFL-reachability with applications to
// Alias Analysis" by Zhang Q, Lyu M R, Yuan H, and Su Z. -- to summarize the
// papers, we build a graph of the uses of a variable, where each node is a
// memory location, and each edge is an action that happened on that memory
// location.  The "actions" can be one of Dereference, Reference, or Assign.
//
// Two variables are considered as aliasing iff you can reach one value's node
// from the other value's node and the language formed by concatenating all of
// the edge labels (actions) conforms to a context-free grammar.
//
// Because this algorithm requires a graph search on each query, we execute the
// algorithm outlined in "Fast algorithms..." (mentioned above)
// in order to transform the graph into sets of variables that may alias in
// ~nlogn time (n = number of variables.), which makes queries take constant
// time.
//===----------------------------------------------------------------------===//

#include "StratifiedSets.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <forward_list>
#include <memory>
#include <tuple>

using namespace llvm;

#define DEBUG_TYPE "cfl-aa"

// Try to go from a Value* to a Function*. Never returns nullptr.
static Optional<Function *> parentFunctionOfValue(Value *);

// Returns possible functions called by the Inst* into the given
// SmallVectorImpl. Returns true if targets found, false otherwise.
// This is templated because InvokeInst/CallInst give us the same
// set of functions that we care about, and I don't like repeating
// myself.
template <typename Inst>
static bool getPossibleTargets(Inst *, SmallVectorImpl<Function *> &);

// Some instructions need to have their users tracked. Instructions like
// `add` require you to get the users of the Instruction* itself, other
// instructions like `store` require you to get the users of the first
// operand. This function gets the "proper" value to track for each
// type of instruction we support.
static Optional<Value *> getTargetValue(Instruction *);

// There are certain instructions (i.e. FenceInst, etc.) that we ignore.
// This notes that we should ignore those.
static bool hasUsefulEdges(Instruction *);

const StratifiedIndex StratifiedLink::SetSentinel =
    std::numeric_limits<StratifiedIndex>::max();

namespace {
// StratifiedInfo Attribute things.
typedef unsigned StratifiedAttr;
LLVM_CONSTEXPR unsigned MaxStratifiedAttrIndex = NumStratifiedAttrs;
LLVM_CONSTEXPR unsigned AttrAllIndex = 0;
LLVM_CONSTEXPR unsigned AttrGlobalIndex = 1;
LLVM_CONSTEXPR unsigned AttrUnknownIndex = 2;
LLVM_CONSTEXPR unsigned AttrFirstArgIndex = 3;
LLVM_CONSTEXPR unsigned AttrLastArgIndex = MaxStratifiedAttrIndex;
LLVM_CONSTEXPR unsigned AttrMaxNumArgs = AttrLastArgIndex - AttrFirstArgIndex;

LLVM_CONSTEXPR StratifiedAttr AttrNone = 0;
LLVM_CONSTEXPR StratifiedAttr AttrUnknown = 1 << AttrUnknownIndex;
LLVM_CONSTEXPR StratifiedAttr AttrAll = ~AttrNone;

// \brief StratifiedSets call for knowledge of "direction", so this is how we
// represent that locally.
enum class Level { Same, Above, Below };

// \brief Edges can be one of four "weights" -- each weight must have an inverse
// weight (Assign has Assign; Reference has Dereference).
enum class EdgeType {
  // The weight assigned when assigning from or to a value. For example, in:
  // %b = getelementptr %a, 0
  // ...The relationships are %b assign %a, and %a assign %b. This used to be
  // two edges, but having a distinction bought us nothing.
  Assign,

  // The edge used when we have an edge going from some handle to a Value.
  // Examples of this include:
  // %b = load %a              (%b Dereference %a)
  // %b = extractelement %a, 0 (%a Dereference %b)
  Dereference,

  // The edge used when our edge goes from a value to a handle that may have
  // contained it at some point. Examples:
  // %b = load %a              (%a Reference %b)
  // %b = extractelement %a, 0 (%b Reference %a)
  Reference
};

// \brief Encodes the notion of a "use"
struct Edge {
  // \brief Which value the edge is coming from
  Value *From;

  // \brief Which value the edge is pointing to
  Value *To;

  // \brief Edge weight
  EdgeType Weight;

  // \brief Whether we aliased any external values along the way that may be
  // invisible to the analysis (i.e. landingpad for exceptions, calls for
  // interprocedural analysis, etc.)
  StratifiedAttrs AdditionalAttrs;

  Edge(Value *From, Value *To, EdgeType W, StratifiedAttrs A)
      : From(From), To(To), Weight(W), AdditionalAttrs(A) {}
};

// \brief Information we have about a function and would like to keep around
struct FunctionInfo {
  StratifiedSets<Value *> Sets;
  // Lots of functions have < 4 returns. Adjust as necessary.
  SmallVector<Value *, 4> ReturnedValues;

  FunctionInfo(StratifiedSets<Value *> &&S, SmallVector<Value *, 4> &&RV)
      : Sets(std::move(S)), ReturnedValues(std::move(RV)) {}
};

struct CFLAliasAnalysis;

struct FunctionHandle : public CallbackVH {
  FunctionHandle(Function *Fn, CFLAliasAnalysis *CFLAA)
      : CallbackVH(Fn), CFLAA(CFLAA) {
    assert(Fn != nullptr);
    assert(CFLAA != nullptr);
  }

  ~FunctionHandle() override {}

  void deleted() override { removeSelfFromCache(); }
  void allUsesReplacedWith(Value *) override { removeSelfFromCache(); }

private:
  CFLAliasAnalysis *CFLAA;

  void removeSelfFromCache();
};

struct CFLAliasAnalysis : public ImmutablePass, public AliasAnalysis {
private:
  /// \brief Cached mapping of Functions to their StratifiedSets.
  /// If a function's sets are currently being built, it is marked
  /// in the cache as an Optional without a value. This way, if we
  /// have any kind of recursion, it is discernable from a function
  /// that simply has empty sets.
  DenseMap<Function *, Optional<FunctionInfo>> Cache;
  std::forward_list<FunctionHandle> Handles;

public:
  static char ID;

  CFLAliasAnalysis() : ImmutablePass(ID) {
    initializeCFLAliasAnalysisPass(*PassRegistry::getPassRegistry());
  }

  ~CFLAliasAnalysis() override {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AliasAnalysis::getAnalysisUsage(AU);
  }

  void *getAdjustedAnalysisPointer(const void *ID) override {
    if (ID == &AliasAnalysis::ID)
      return (AliasAnalysis *)this;
    return this;
  }

  /// \brief Inserts the given Function into the cache.
  void scan(Function *Fn);

  void evict(Function *Fn) { Cache.erase(Fn); }

  /// \brief Ensures that the given function is available in the cache.
  /// Returns the appropriate entry from the cache.
  const Optional<FunctionInfo> &ensureCached(Function *Fn) {
    auto Iter = Cache.find(Fn);
    if (Iter == Cache.end()) {
      scan(Fn);
      Iter = Cache.find(Fn);
      assert(Iter != Cache.end());
      assert(Iter->second.hasValue());
    }
    return Iter->second;
  }

  AliasResult query(const MemoryLocation &LocA, const MemoryLocation &LocB);

  AliasResult alias(const MemoryLocation &LocA,
                    const MemoryLocation &LocB) override {
    if (LocA.Ptr == LocB.Ptr) {
      if (LocA.Size == LocB.Size) {
        return MustAlias;
      } else {
        return PartialAlias;
      }
    }

    // Comparisons between global variables and other constants should be
    // handled by BasicAA.
    // TODO: ConstantExpr handling -- CFLAA may report NoAlias when comparing
    // a GlobalValue and ConstantExpr, but every query needs to have at least
    // one Value tied to a Function, and neither GlobalValues nor ConstantExprs
    // are.
    if (isa<Constant>(LocA.Ptr) && isa<Constant>(LocB.Ptr)) {
      return AliasAnalysis::alias(LocA, LocB);
    }

    AliasResult QueryResult = query(LocA, LocB);
    if (QueryResult == MayAlias)
      return AliasAnalysis::alias(LocA, LocB);

    return QueryResult;
  }

  bool doInitialization(Module &M) override;
};

void FunctionHandle::removeSelfFromCache() {
  assert(CFLAA != nullptr);
  auto *Val = getValPtr();
  CFLAA->evict(cast<Function>(Val));
  setValPtr(nullptr);
}

// \brief Gets the edges our graph should have, based on an Instruction*
class GetEdgesVisitor : public InstVisitor<GetEdgesVisitor, void> {
  CFLAliasAnalysis &AA;
  SmallVectorImpl<Edge> &Output;

public:
  GetEdgesVisitor(CFLAliasAnalysis &AA, SmallVectorImpl<Edge> &Output)
      : AA(AA), Output(Output) {}

  void visitInstruction(Instruction &) {
    llvm_unreachable("Unsupported instruction encountered");
  }

  void visitPtrToIntInst(PtrToIntInst &Inst) {
    auto *Ptr = Inst.getOperand(0);
    Output.push_back(Edge(Ptr, Ptr, EdgeType::Assign, AttrUnknown));
  }

  void visitIntToPtrInst(IntToPtrInst &Inst) {
    auto *Ptr = &Inst;
    Output.push_back(Edge(Ptr, Ptr, EdgeType::Assign, AttrUnknown));
  }

  void visitCastInst(CastInst &Inst) {
    Output.push_back(
        Edge(&Inst, Inst.getOperand(0), EdgeType::Assign, AttrNone));
  }

  void visitBinaryOperator(BinaryOperator &Inst) {
    auto *Op1 = Inst.getOperand(0);
    auto *Op2 = Inst.getOperand(1);
    Output.push_back(Edge(&Inst, Op1, EdgeType::Assign, AttrNone));
    Output.push_back(Edge(&Inst, Op2, EdgeType::Assign, AttrNone));
  }

  void visitAtomicCmpXchgInst(AtomicCmpXchgInst &Inst) {
    auto *Ptr = Inst.getPointerOperand();
    auto *Val = Inst.getNewValOperand();
    Output.push_back(Edge(Ptr, Val, EdgeType::Dereference, AttrNone));
  }

  void visitAtomicRMWInst(AtomicRMWInst &Inst) {
    auto *Ptr = Inst.getPointerOperand();
    auto *Val = Inst.getValOperand();
    Output.push_back(Edge(Ptr, Val, EdgeType::Dereference, AttrNone));
  }

  void visitPHINode(PHINode &Inst) {
    for (Value *Val : Inst.incoming_values()) {
      Output.push_back(Edge(&Inst, Val, EdgeType::Assign, AttrNone));
    }
  }

  void visitGetElementPtrInst(GetElementPtrInst &Inst) {
    auto *Op = Inst.getPointerOperand();
    Output.push_back(Edge(&Inst, Op, EdgeType::Assign, AttrNone));
    for (auto I = Inst.idx_begin(), E = Inst.idx_end(); I != E; ++I)
      Output.push_back(Edge(&Inst, *I, EdgeType::Assign, AttrNone));
  }

  void visitSelectInst(SelectInst &Inst) {
    // Condition is not processed here (The actual statement producing
    // the condition result is processed elsewhere). For select, the
    // condition is evaluated, but not loaded, stored, or assigned
    // simply as a result of being the condition of a select.

    auto *TrueVal = Inst.getTrueValue();
    Output.push_back(Edge(&Inst, TrueVal, EdgeType::Assign, AttrNone));
    auto *FalseVal = Inst.getFalseValue();
    Output.push_back(Edge(&Inst, FalseVal, EdgeType::Assign, AttrNone));
  }

  void visitAllocaInst(AllocaInst &) {}

  void visitLoadInst(LoadInst &Inst) {
    auto *Ptr = Inst.getPointerOperand();
    auto *Val = &Inst;
    Output.push_back(Edge(Val, Ptr, EdgeType::Reference, AttrNone));
  }

  void visitStoreInst(StoreInst &Inst) {
    auto *Ptr = Inst.getPointerOperand();
    auto *Val = Inst.getValueOperand();
    Output.push_back(Edge(Ptr, Val, EdgeType::Dereference, AttrNone));
  }

  void visitVAArgInst(VAArgInst &Inst) {
    // We can't fully model va_arg here. For *Ptr = Inst.getOperand(0), it does
    // two things:
    //  1. Loads a value from *((T*)*Ptr).
    //  2. Increments (stores to) *Ptr by some target-specific amount.
    // For now, we'll handle this like a landingpad instruction (by placing the
    // result in its own group, and having that group alias externals).
    auto *Val = &Inst;
    Output.push_back(Edge(Val, Val, EdgeType::Assign, AttrAll));
  }

  static bool isFunctionExternal(Function *Fn) {
    return Fn->isDeclaration() || !Fn->hasLocalLinkage();
  }

  // Gets whether the sets at Index1 above, below, or equal to the sets at
  // Index2. Returns None if they are not in the same set chain.
  static Optional<Level> getIndexRelation(const StratifiedSets<Value *> &Sets,
                                          StratifiedIndex Index1,
                                          StratifiedIndex Index2) {
    if (Index1 == Index2)
      return Level::Same;

    const auto *Current = &Sets.getLink(Index1);
    while (Current->hasBelow()) {
      if (Current->Below == Index2)
        return Level::Below;
      Current = &Sets.getLink(Current->Below);
    }

    Current = &Sets.getLink(Index1);
    while (Current->hasAbove()) {
      if (Current->Above == Index2)
        return Level::Above;
      Current = &Sets.getLink(Current->Above);
    }

    return NoneType();
  }

  bool
  tryInterproceduralAnalysis(const SmallVectorImpl<Function *> &Fns,
                             Value *FuncValue,
                             const iterator_range<User::op_iterator> &Args) {
    const unsigned ExpectedMaxArgs = 8;
    const unsigned MaxSupportedArgs = 50;
    assert(Fns.size() > 0);

    // I put this here to give us an upper bound on time taken by IPA. Is it
    // really (realistically) needed? Keep in mind that we do have an n^2 algo.
    if (std::distance(Args.begin(), Args.end()) > (int)MaxSupportedArgs)
      return false;

    // Exit early if we'll fail anyway
    for (auto *Fn : Fns) {
      if (isFunctionExternal(Fn) || Fn->isVarArg())
        return false;
      auto &MaybeInfo = AA.ensureCached(Fn);
      if (!MaybeInfo.hasValue())
        return false;
    }

    SmallVector<Value *, ExpectedMaxArgs> Arguments(Args.begin(), Args.end());
    SmallVector<StratifiedInfo, ExpectedMaxArgs> Parameters;
    for (auto *Fn : Fns) {
      auto &Info = *AA.ensureCached(Fn);
      auto &Sets = Info.Sets;
      auto &RetVals = Info.ReturnedValues;

      Parameters.clear();
      for (auto &Param : Fn->args()) {
        auto MaybeInfo = Sets.find(&Param);
        // Did a new parameter somehow get added to the function/slip by?
        if (!MaybeInfo.hasValue())
          return false;
        Parameters.push_back(*MaybeInfo);
      }

      // Adding an edge from argument -> return value for each parameter that
      // may alias the return value
      for (unsigned I = 0, E = Parameters.size(); I != E; ++I) {
        auto &ParamInfo = Parameters[I];
        auto &ArgVal = Arguments[I];
        bool AddEdge = false;
        StratifiedAttrs Externals;
        for (unsigned X = 0, XE = RetVals.size(); X != XE; ++X) {
          auto MaybeInfo = Sets.find(RetVals[X]);
          if (!MaybeInfo.hasValue())
            return false;

          auto &RetInfo = *MaybeInfo;
          auto RetAttrs = Sets.getLink(RetInfo.Index).Attrs;
          auto ParamAttrs = Sets.getLink(ParamInfo.Index).Attrs;
          auto MaybeRelation =
              getIndexRelation(Sets, ParamInfo.Index, RetInfo.Index);
          if (MaybeRelation.hasValue()) {
            AddEdge = true;
            Externals |= RetAttrs | ParamAttrs;
          }
        }
        if (AddEdge)
          Output.push_back(Edge(FuncValue, ArgVal, EdgeType::Assign,
                                StratifiedAttrs().flip()));
      }

      if (Parameters.size() != Arguments.size())
        return false;

      // Adding edges between arguments for arguments that may end up aliasing
      // each other. This is necessary for functions such as
      // void foo(int** a, int** b) { *a = *b; }
      // (Technically, the proper sets for this would be those below
      // Arguments[I] and Arguments[X], but our algorithm will produce
      // extremely similar, and equally correct, results either way)
      for (unsigned I = 0, E = Arguments.size(); I != E; ++I) {
        auto &MainVal = Arguments[I];
        auto &MainInfo = Parameters[I];
        auto &MainAttrs = Sets.getLink(MainInfo.Index).Attrs;
        for (unsigned X = I + 1; X != E; ++X) {
          auto &SubInfo = Parameters[X];
          auto &SubVal = Arguments[X];
          auto &SubAttrs = Sets.getLink(SubInfo.Index).Attrs;
          auto MaybeRelation =
              getIndexRelation(Sets, MainInfo.Index, SubInfo.Index);

          if (!MaybeRelation.hasValue())
            continue;

          auto NewAttrs = SubAttrs | MainAttrs;
          Output.push_back(Edge(MainVal, SubVal, EdgeType::Assign, NewAttrs));
        }
      }
    }
    return true;
  }

  template <typename InstT> void visitCallLikeInst(InstT &Inst) {
    SmallVector<Function *, 4> Targets;
    if (getPossibleTargets(&Inst, Targets)) {
      if (tryInterproceduralAnalysis(Targets, &Inst, Inst.arg_operands()))
        return;
      // Cleanup from interprocedural analysis
      Output.clear();
    }

    for (Value *V : Inst.arg_operands())
      Output.push_back(Edge(&Inst, V, EdgeType::Assign, AttrAll));
  }

  void visitCallInst(CallInst &Inst) { visitCallLikeInst(Inst); }

  void visitInvokeInst(InvokeInst &Inst) { visitCallLikeInst(Inst); }

  // Because vectors/aggregates are immutable and unaddressable,
  // there's nothing we can do to coax a value out of them, other
  // than calling Extract{Element,Value}. We can effectively treat
  // them as pointers to arbitrary memory locations we can store in
  // and load from.
  void visitExtractElementInst(ExtractElementInst &Inst) {
    auto *Ptr = Inst.getVectorOperand();
    auto *Val = &Inst;
    Output.push_back(Edge(Val, Ptr, EdgeType::Reference, AttrNone));
  }

  void visitInsertElementInst(InsertElementInst &Inst) {
    auto *Vec = Inst.getOperand(0);
    auto *Val = Inst.getOperand(1);
    Output.push_back(Edge(&Inst, Vec, EdgeType::Assign, AttrNone));
    Output.push_back(Edge(&Inst, Val, EdgeType::Dereference, AttrNone));
  }

  void visitLandingPadInst(LandingPadInst &Inst) {
    // Exceptions come from "nowhere", from our analysis' perspective.
    // So we place the instruction its own group, noting that said group may
    // alias externals
    Output.push_back(Edge(&Inst, &Inst, EdgeType::Assign, AttrAll));
  }

  void visitInsertValueInst(InsertValueInst &Inst) {
    auto *Agg = Inst.getOperand(0);
    auto *Val = Inst.getOperand(1);
    Output.push_back(Edge(&Inst, Agg, EdgeType::Assign, AttrNone));
    Output.push_back(Edge(&Inst, Val, EdgeType::Dereference, AttrNone));
  }

  void visitExtractValueInst(ExtractValueInst &Inst) {
    auto *Ptr = Inst.getAggregateOperand();
    Output.push_back(Edge(&Inst, Ptr, EdgeType::Reference, AttrNone));
  }

  void visitShuffleVectorInst(ShuffleVectorInst &Inst) {
    auto *From1 = Inst.getOperand(0);
    auto *From2 = Inst.getOperand(1);
    Output.push_back(Edge(&Inst, From1, EdgeType::Assign, AttrNone));
    Output.push_back(Edge(&Inst, From2, EdgeType::Assign, AttrNone));
  }

  void visitConstantExpr(ConstantExpr *CE) {
    switch (CE->getOpcode()) {
    default:
      llvm_unreachable("Unknown instruction type encountered!");
// Build the switch statement using the Instruction.def file.
#define HANDLE_INST(NUM, OPCODE, CLASS)                                        \
  case Instruction::OPCODE:                                                    \
    visit##OPCODE(*(CLASS *)CE);                                               \
    break;
#include "llvm/IR/Instruction.def"
    }
  }
};

// For a given instruction, we need to know which Value* to get the
// users of in order to build our graph. In some cases (i.e. add),
// we simply need the Instruction*. In other cases (i.e. store),
// finding the users of the Instruction* is useless; we need to find
// the users of the first operand. This handles determining which
// value to follow for us.
//
// Note: we *need* to keep this in sync with GetEdgesVisitor. Add
// something to GetEdgesVisitor, add it here -- remove something from
// GetEdgesVisitor, remove it here.
class GetTargetValueVisitor
    : public InstVisitor<GetTargetValueVisitor, Value *> {
public:
  Value *visitInstruction(Instruction &Inst) { return &Inst; }

  Value *visitStoreInst(StoreInst &Inst) { return Inst.getPointerOperand(); }

  Value *visitAtomicCmpXchgInst(AtomicCmpXchgInst &Inst) {
    return Inst.getPointerOperand();
  }

  Value *visitAtomicRMWInst(AtomicRMWInst &Inst) {
    return Inst.getPointerOperand();
  }

  Value *visitInsertElementInst(InsertElementInst &Inst) {
    return Inst.getOperand(0);
  }

  Value *visitInsertValueInst(InsertValueInst &Inst) {
    return Inst.getAggregateOperand();
  }
};

// Set building requires a weighted bidirectional graph.
template <typename EdgeTypeT> class WeightedBidirectionalGraph {
public:
  typedef std::size_t Node;

private:
  const static Node StartNode = Node(0);

  struct Edge {
    EdgeTypeT Weight;
    Node Other;

    Edge(const EdgeTypeT &W, const Node &N) : Weight(W), Other(N) {}

    bool operator==(const Edge &E) const {
      return Weight == E.Weight && Other == E.Other;
    }

    bool operator!=(const Edge &E) const { return !operator==(E); }
  };

  struct NodeImpl {
    std::vector<Edge> Edges;
  };

  std::vector<NodeImpl> NodeImpls;

  bool inbounds(Node NodeIndex) const { return NodeIndex < NodeImpls.size(); }

  const NodeImpl &getNode(Node N) const { return NodeImpls[N]; }
  NodeImpl &getNode(Node N) { return NodeImpls[N]; }

public:
  // ----- Various Edge iterators for the graph ----- //

  // \brief Iterator for edges. Because this graph is bidirected, we don't
  // allow modificaiton of the edges using this iterator. Additionally, the
  // iterator becomes invalid if you add edges to or from the node you're
  // getting the edges of.
  struct EdgeIterator : public std::iterator<std::forward_iterator_tag,
                                             std::tuple<EdgeTypeT, Node *>> {
    EdgeIterator(const typename std::vector<Edge>::const_iterator &Iter)
        : Current(Iter) {}

    EdgeIterator(NodeImpl &Impl) : Current(Impl.begin()) {}

    EdgeIterator &operator++() {
      ++Current;
      return *this;
    }

    EdgeIterator operator++(int) {
      EdgeIterator Copy(Current);
      operator++();
      return Copy;
    }

    std::tuple<EdgeTypeT, Node> &operator*() {
      Store = std::make_tuple(Current->Weight, Current->Other);
      return Store;
    }

    bool operator==(const EdgeIterator &Other) const {
      return Current == Other.Current;
    }

    bool operator!=(const EdgeIterator &Other) const {
      return !operator==(Other);
    }

  private:
    typename std::vector<Edge>::const_iterator Current;
    std::tuple<EdgeTypeT, Node> Store;
  };

  // Wrapper for EdgeIterator with begin()/end() calls.
  struct EdgeIterable {
    EdgeIterable(const std::vector<Edge> &Edges)
        : BeginIter(Edges.begin()), EndIter(Edges.end()) {}

    EdgeIterator begin() { return EdgeIterator(BeginIter); }

    EdgeIterator end() { return EdgeIterator(EndIter); }

  private:
    typename std::vector<Edge>::const_iterator BeginIter;
    typename std::vector<Edge>::const_iterator EndIter;
  };

  // ----- Actual graph-related things ----- //

  WeightedBidirectionalGraph() {}

  WeightedBidirectionalGraph(WeightedBidirectionalGraph<EdgeTypeT> &&Other)
      : NodeImpls(std::move(Other.NodeImpls)) {}

  WeightedBidirectionalGraph<EdgeTypeT> &
  operator=(WeightedBidirectionalGraph<EdgeTypeT> &&Other) {
    NodeImpls = std::move(Other.NodeImpls);
    return *this;
  }

  Node addNode() {
    auto Index = NodeImpls.size();
    auto NewNode = Node(Index);
    NodeImpls.push_back(NodeImpl());
    return NewNode;
  }

  void addEdge(Node From, Node To, const EdgeTypeT &Weight,
               const EdgeTypeT &ReverseWeight) {
    assert(inbounds(From));
    assert(inbounds(To));
    auto &FromNode = getNode(From);
    auto &ToNode = getNode(To);
    FromNode.Edges.push_back(Edge(Weight, To));
    ToNode.Edges.push_back(Edge(ReverseWeight, From));
  }

  EdgeIterable edgesFor(const Node &N) const {
    const auto &Node = getNode(N);
    return EdgeIterable(Node.Edges);
  }

  bool empty() const { return NodeImpls.empty(); }
  std::size_t size() const { return NodeImpls.size(); }

  // \brief Gets an arbitrary node in the graph as a starting point for
  // traversal.
  Node getEntryNode() {
    assert(inbounds(StartNode));
    return StartNode;
  }
};

typedef WeightedBidirectionalGraph<std::pair<EdgeType, StratifiedAttrs>> GraphT;
typedef DenseMap<Value *, GraphT::Node> NodeMapT;
}

// -- Setting up/registering CFLAA pass -- //
char CFLAliasAnalysis::ID = 0;

INITIALIZE_AG_PASS(CFLAliasAnalysis, AliasAnalysis, "cfl-aa",
                   "CFL-Based AA implementation", false, true, false)

ImmutablePass *llvm::createCFLAliasAnalysisPass() {
  return new CFLAliasAnalysis();
}

//===----------------------------------------------------------------------===//
// Function declarations that require types defined in the namespace above
//===----------------------------------------------------------------------===//

// Given an argument number, returns the appropriate Attr index to set.
static StratifiedAttr argNumberToAttrIndex(StratifiedAttr);

// Given a Value, potentially return which AttrIndex it maps to.
static Optional<StratifiedAttr> valueToAttrIndex(Value *Val);

// Gets the inverse of a given EdgeType.
static EdgeType flipWeight(EdgeType);

// Gets edges of the given Instruction*, writing them to the SmallVector*.
static void argsToEdges(CFLAliasAnalysis &, Instruction *,
                        SmallVectorImpl<Edge> &);

// Gets edges of the given ConstantExpr*, writing them to the SmallVector*.
static void argsToEdges(CFLAliasAnalysis &, ConstantExpr *,
                        SmallVectorImpl<Edge> &);

// Gets the "Level" that one should travel in StratifiedSets
// given an EdgeType.
static Level directionOfEdgeType(EdgeType);

// Builds the graph needed for constructing the StratifiedSets for the
// given function
static void buildGraphFrom(CFLAliasAnalysis &, Function *,
                           SmallVectorImpl<Value *> &, NodeMapT &, GraphT &);

// Gets the edges of a ConstantExpr as if it was an Instruction. This
// function also acts on any nested ConstantExprs, adding the edges
// of those to the given SmallVector as well.
static void constexprToEdges(CFLAliasAnalysis &, ConstantExpr &,
                             SmallVectorImpl<Edge> &);

// Given an Instruction, this will add it to the graph, along with any
// Instructions that are potentially only available from said Instruction
// For example, given the following line:
//   %0 = load i16* getelementptr ([1 x i16]* @a, 0, 0), align 2
// addInstructionToGraph would add both the `load` and `getelementptr`
// instructions to the graph appropriately.
static void addInstructionToGraph(CFLAliasAnalysis &, Instruction &,
                                  SmallVectorImpl<Value *> &, NodeMapT &,
                                  GraphT &);

// Notes whether it would be pointless to add the given Value to our sets.
static bool canSkipAddingToSets(Value *Val);

// Builds the graph + StratifiedSets for a function.
static FunctionInfo buildSetsFrom(CFLAliasAnalysis &, Function *);

static Optional<Function *> parentFunctionOfValue(Value *Val) {
  if (auto *Inst = dyn_cast<Instruction>(Val)) {
    auto *Bb = Inst->getParent();
    return Bb->getParent();
  }

  if (auto *Arg = dyn_cast<Argument>(Val))
    return Arg->getParent();
  return NoneType();
}

template <typename Inst>
static bool getPossibleTargets(Inst *Call,
                               SmallVectorImpl<Function *> &Output) {
  if (auto *Fn = Call->getCalledFunction()) {
    Output.push_back(Fn);
    return true;
  }

  // TODO: If the call is indirect, we might be able to enumerate all potential
  // targets of the call and return them, rather than just failing.
  return false;
}

static Optional<Value *> getTargetValue(Instruction *Inst) {
  GetTargetValueVisitor V;
  return V.visit(Inst);
}

static bool hasUsefulEdges(Instruction *Inst) {
  bool IsNonInvokeTerminator =
      isa<TerminatorInst>(Inst) && !isa<InvokeInst>(Inst);
  return !isa<CmpInst>(Inst) && !isa<FenceInst>(Inst) && !IsNonInvokeTerminator;
}

static bool hasUsefulEdges(ConstantExpr *CE) {
  // ConstantExpr doens't have terminators, invokes, or fences, so only needs
  // to check for compares.
  return CE->getOpcode() != Instruction::ICmp &&
         CE->getOpcode() != Instruction::FCmp;
}

static Optional<StratifiedAttr> valueToAttrIndex(Value *Val) {
  if (isa<GlobalValue>(Val))
    return AttrGlobalIndex;

  if (auto *Arg = dyn_cast<Argument>(Val))
    // Only pointer arguments should have the argument attribute,
    // because things can't escape through scalars without us seeing a
    // cast, and thus, interaction with them doesn't matter.
    if (!Arg->hasNoAliasAttr() && Arg->getType()->isPointerTy())
      return argNumberToAttrIndex(Arg->getArgNo());
  return NoneType();
}

static StratifiedAttr argNumberToAttrIndex(unsigned ArgNum) {
  if (ArgNum >= AttrMaxNumArgs)
    return AttrAllIndex;
  return ArgNum + AttrFirstArgIndex;
}

static EdgeType flipWeight(EdgeType Initial) {
  switch (Initial) {
  case EdgeType::Assign:
    return EdgeType::Assign;
  case EdgeType::Dereference:
    return EdgeType::Reference;
  case EdgeType::Reference:
    return EdgeType::Dereference;
  }
  llvm_unreachable("Incomplete coverage of EdgeType enum");
}

static void argsToEdges(CFLAliasAnalysis &Analysis, Instruction *Inst,
                        SmallVectorImpl<Edge> &Output) {
  assert(hasUsefulEdges(Inst) &&
         "Expected instructions to have 'useful' edges");
  GetEdgesVisitor v(Analysis, Output);
  v.visit(Inst);
}

static void argsToEdges(CFLAliasAnalysis &Analysis, ConstantExpr *CE,
                        SmallVectorImpl<Edge> &Output) {
  assert(hasUsefulEdges(CE) && "Expected constant expr to have 'useful' edges");
  GetEdgesVisitor v(Analysis, Output);
  v.visitConstantExpr(CE);
}

static Level directionOfEdgeType(EdgeType Weight) {
  switch (Weight) {
  case EdgeType::Reference:
    return Level::Above;
  case EdgeType::Dereference:
    return Level::Below;
  case EdgeType::Assign:
    return Level::Same;
  }
  llvm_unreachable("Incomplete switch coverage");
}

static void constexprToEdges(CFLAliasAnalysis &Analysis,
                             ConstantExpr &CExprToCollapse,
                             SmallVectorImpl<Edge> &Results) {
  SmallVector<ConstantExpr *, 4> Worklist;
  Worklist.push_back(&CExprToCollapse);

  SmallVector<Edge, 8> ConstexprEdges;
  SmallPtrSet<ConstantExpr *, 4> Visited;
  while (!Worklist.empty()) {
    auto *CExpr = Worklist.pop_back_val();

    if (!hasUsefulEdges(CExpr))
      continue;

    ConstexprEdges.clear();
    argsToEdges(Analysis, CExpr, ConstexprEdges);
    for (auto &Edge : ConstexprEdges) {
      if (auto *Nested = dyn_cast<ConstantExpr>(Edge.From))
        if (Visited.insert(Nested).second)
          Worklist.push_back(Nested);

      if (auto *Nested = dyn_cast<ConstantExpr>(Edge.To))
        if (Visited.insert(Nested).second)
          Worklist.push_back(Nested);
    }

    Results.append(ConstexprEdges.begin(), ConstexprEdges.end());
  }
}

static void addInstructionToGraph(CFLAliasAnalysis &Analysis, Instruction &Inst,
                                  SmallVectorImpl<Value *> &ReturnedValues,
                                  NodeMapT &Map, GraphT &Graph) {
  const auto findOrInsertNode = [&Map, &Graph](Value *Val) {
    auto Pair = Map.insert(std::make_pair(Val, GraphT::Node()));
    auto &Iter = Pair.first;
    if (Pair.second) {
      auto NewNode = Graph.addNode();
      Iter->second = NewNode;
    }
    return Iter->second;
  };

  // We don't want the edges of most "return" instructions, but we *do* want
  // to know what can be returned.
  if (isa<ReturnInst>(&Inst))
    ReturnedValues.push_back(&Inst);

  if (!hasUsefulEdges(&Inst))
    return;

  SmallVector<Edge, 8> Edges;
  argsToEdges(Analysis, &Inst, Edges);

  // In the case of an unused alloca (or similar), edges may be empty. Note
  // that it exists so we can potentially answer NoAlias.
  if (Edges.empty()) {
    auto MaybeVal = getTargetValue(&Inst);
    assert(MaybeVal.hasValue());
    auto *Target = *MaybeVal;
    findOrInsertNode(Target);
    return;
  }

  const auto addEdgeToGraph = [&Graph, &findOrInsertNode](const Edge &E) {
    auto To = findOrInsertNode(E.To);
    auto From = findOrInsertNode(E.From);
    auto FlippedWeight = flipWeight(E.Weight);
    auto Attrs = E.AdditionalAttrs;
    Graph.addEdge(From, To, std::make_pair(E.Weight, Attrs),
                  std::make_pair(FlippedWeight, Attrs));
  };

  SmallVector<ConstantExpr *, 4> ConstantExprs;
  for (const Edge &E : Edges) {
    addEdgeToGraph(E);
    if (auto *Constexpr = dyn_cast<ConstantExpr>(E.To))
      ConstantExprs.push_back(Constexpr);
    if (auto *Constexpr = dyn_cast<ConstantExpr>(E.From))
      ConstantExprs.push_back(Constexpr);
  }

  for (ConstantExpr *CE : ConstantExprs) {
    Edges.clear();
    constexprToEdges(Analysis, *CE, Edges);
    std::for_each(Edges.begin(), Edges.end(), addEdgeToGraph);
  }
}

// Aside: We may remove graph construction entirely, because it doesn't really
// buy us much that we don't already have. I'd like to add interprocedural
// analysis prior to this however, in case that somehow requires the graph
// produced by this for efficient execution
static void buildGraphFrom(CFLAliasAnalysis &Analysis, Function *Fn,
                           SmallVectorImpl<Value *> &ReturnedValues,
                           NodeMapT &Map, GraphT &Graph) {
  for (auto &Bb : Fn->getBasicBlockList())
    for (auto &Inst : Bb.getInstList())
      addInstructionToGraph(Analysis, Inst, ReturnedValues, Map, Graph);
}

static bool canSkipAddingToSets(Value *Val) {
  // Constants can share instances, which may falsely unify multiple
  // sets, e.g. in
  // store i32* null, i32** %ptr1
  // store i32* null, i32** %ptr2
  // clearly ptr1 and ptr2 should not be unified into the same set, so
  // we should filter out the (potentially shared) instance to
  // i32* null.
  if (isa<Constant>(Val)) {
    bool Container = isa<ConstantVector>(Val) || isa<ConstantArray>(Val) ||
                     isa<ConstantStruct>(Val);
    // TODO: Because all of these things are constant, we can determine whether
    // the data is *actually* mutable at graph building time. This will probably
    // come for free/cheap with offset awareness.
    bool CanStoreMutableData =
        isa<GlobalValue>(Val) || isa<ConstantExpr>(Val) || Container;
    return !CanStoreMutableData;
  }

  return false;
}

static FunctionInfo buildSetsFrom(CFLAliasAnalysis &Analysis, Function *Fn) {
  NodeMapT Map;
  GraphT Graph;
  SmallVector<Value *, 4> ReturnedValues;

  buildGraphFrom(Analysis, Fn, ReturnedValues, Map, Graph);

  DenseMap<GraphT::Node, Value *> NodeValueMap;
  NodeValueMap.resize(Map.size());
  for (const auto &Pair : Map)
    NodeValueMap.insert(std::make_pair(Pair.second, Pair.first));

  const auto findValueOrDie = [&NodeValueMap](GraphT::Node Node) {
    auto ValIter = NodeValueMap.find(Node);
    assert(ValIter != NodeValueMap.end());
    return ValIter->second;
  };

  StratifiedSetsBuilder<Value *> Builder;

  SmallVector<GraphT::Node, 16> Worklist;
  for (auto &Pair : Map) {
    Worklist.clear();

    auto *Value = Pair.first;
    Builder.add(Value);
    auto InitialNode = Pair.second;
    Worklist.push_back(InitialNode);
    while (!Worklist.empty()) {
      auto Node = Worklist.pop_back_val();
      auto *CurValue = findValueOrDie(Node);
      if (canSkipAddingToSets(CurValue))
        continue;

      for (const auto &EdgeTuple : Graph.edgesFor(Node)) {
        auto Weight = std::get<0>(EdgeTuple);
        auto Label = Weight.first;
        auto &OtherNode = std::get<1>(EdgeTuple);
        auto *OtherValue = findValueOrDie(OtherNode);

        if (canSkipAddingToSets(OtherValue))
          continue;

        bool Added;
        switch (directionOfEdgeType(Label)) {
        case Level::Above:
          Added = Builder.addAbove(CurValue, OtherValue);
          break;
        case Level::Below:
          Added = Builder.addBelow(CurValue, OtherValue);
          break;
        case Level::Same:
          Added = Builder.addWith(CurValue, OtherValue);
          break;
        }

        auto Aliasing = Weight.second;
        if (auto MaybeCurIndex = valueToAttrIndex(CurValue))
          Aliasing.set(*MaybeCurIndex);
        if (auto MaybeOtherIndex = valueToAttrIndex(OtherValue))
          Aliasing.set(*MaybeOtherIndex);
        Builder.noteAttributes(CurValue, Aliasing);
        Builder.noteAttributes(OtherValue, Aliasing);

        if (Added)
          Worklist.push_back(OtherNode);
      }
    }
  }

  // There are times when we end up with parameters not in our graph (i.e. if
  // it's only used as the condition of a branch). Other bits of code depend on
  // things that were present during construction being present in the graph.
  // So, we add all present arguments here.
  for (auto &Arg : Fn->args()) {
    if (!Builder.add(&Arg))
      continue;

    auto Attrs = valueToAttrIndex(&Arg);
    if (Attrs.hasValue())
      Builder.noteAttributes(&Arg, *Attrs);
  }

  return FunctionInfo(Builder.build(), std::move(ReturnedValues));
}

void CFLAliasAnalysis::scan(Function *Fn) {
  auto InsertPair = Cache.insert(std::make_pair(Fn, Optional<FunctionInfo>()));
  (void)InsertPair;
  assert(InsertPair.second &&
         "Trying to scan a function that has already been cached");

  FunctionInfo Info(buildSetsFrom(*this, Fn));
  Cache[Fn] = std::move(Info);
  Handles.push_front(FunctionHandle(Fn, this));
}

AliasResult CFLAliasAnalysis::query(const MemoryLocation &LocA,
                                    const MemoryLocation &LocB) {
  auto *ValA = const_cast<Value *>(LocA.Ptr);
  auto *ValB = const_cast<Value *>(LocB.Ptr);

  Function *Fn = nullptr;
  auto MaybeFnA = parentFunctionOfValue(ValA);
  auto MaybeFnB = parentFunctionOfValue(ValB);
  if (!MaybeFnA.hasValue() && !MaybeFnB.hasValue()) {
    // The only times this is known to happen are when globals + InlineAsm
    // are involved
    DEBUG(dbgs() << "CFLAA: could not extract parent function information.\n");
    return MayAlias;
  }

  if (MaybeFnA.hasValue()) {
    Fn = *MaybeFnA;
    assert((!MaybeFnB.hasValue() || *MaybeFnB == *MaybeFnA) &&
           "Interprocedural queries not supported");
  } else {
    Fn = *MaybeFnB;
  }

  assert(Fn != nullptr);
  auto &MaybeInfo = ensureCached(Fn);
  assert(MaybeInfo.hasValue());

  auto &Sets = MaybeInfo->Sets;
  auto MaybeA = Sets.find(ValA);
  if (!MaybeA.hasValue())
    return MayAlias;

  auto MaybeB = Sets.find(ValB);
  if (!MaybeB.hasValue())
    return MayAlias;

  auto SetA = *MaybeA;
  auto SetB = *MaybeB;
  auto AttrsA = Sets.getLink(SetA.Index).Attrs;
  auto AttrsB = Sets.getLink(SetB.Index).Attrs;

  // Stratified set attributes are used as markets to signify whether a member
  // of a StratifiedSet (or a member of a set above the current set) has
  // interacted with either arguments or globals. "Interacted with" meaning
  // its value may be different depending on the value of an argument or
  // global. The thought behind this is that, because arguments and globals
  // may alias each other, if AttrsA and AttrsB have touched args/globals,
  // we must conservatively say that they alias. However, if at least one of
  // the sets has no values that could legally be altered by changing the value
  // of an argument or global, then we don't have to be as conservative.
  if (AttrsA.any() && AttrsB.any())
    return MayAlias;

  // We currently unify things even if the accesses to them may not be in
  // bounds, so we can't return partial alias here because we don't
  // know whether the pointer is really within the object or not.
  // IE Given an out of bounds GEP and an alloca'd pointer, we may
  // unify the two. We can't return partial alias for this case.
  // Since we do not currently track enough information to
  // differentiate

  if (SetA.Index == SetB.Index)
    return MayAlias;

  return NoAlias;
}

bool CFLAliasAnalysis::doInitialization(Module &M) {
  InitializeAliasAnalysis(this, &M.getDataLayout());
  return true;
}
