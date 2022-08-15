//===-- SpillPlacement.cpp - Optimal Spill Code Placement -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the spill code placement analysis.
//
// Each edge bundle corresponds to a node in a Hopfield network. Constraints on
// basic blocks are weighted by the block frequency and added to become the node
// bias.
//
// Transparent basic blocks have the variable live through, but don't care if it
// is spilled or in a register. These blocks become connections in the Hopfield
// network, again weighted by block frequency.
//
// The Hopfield network minimizes (possibly locally) its energy function:
//
//   E = -sum_n V_n * ( B_n + sum_{n, m linked by b} V_m * F_b )
//
// The energy function represents the expected spill code execution frequency,
// or the cost of spilling. This is a Lyapunov function which never increases
// when a node is updated. It is guaranteed to converge to a local minimum.
//
//===----------------------------------------------------------------------===//

#include "SpillPlacement.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/EdgeBundles.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"

using namespace llvm;

#define DEBUG_TYPE "spillplacement"

char SpillPlacement::ID = 0;
INITIALIZE_PASS_BEGIN(SpillPlacement, "spill-code-placement",
                      "Spill Code Placement Analysis", true, true)
INITIALIZE_PASS_DEPENDENCY(EdgeBundles)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_END(SpillPlacement, "spill-code-placement",
                    "Spill Code Placement Analysis", true, true)

char &llvm::SpillPlacementID = SpillPlacement::ID;

void SpillPlacement::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<MachineBlockFrequencyInfo>();
  AU.addRequiredTransitive<EdgeBundles>();
  AU.addRequiredTransitive<MachineLoopInfo>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

/// Node - Each edge bundle corresponds to a Hopfield node.
///
/// The node contains precomputed frequency data that only depends on the CFG,
/// but Bias and Links are computed each time placeSpills is called.
///
/// The node Value is positive when the variable should be in a register. The
/// value can change when linked nodes change, but convergence is very fast
/// because all weights are positive.
///
struct SpillPlacement::Node {
  /// BiasN - Sum of blocks that prefer a spill.
  BlockFrequency BiasN;
  /// BiasP - Sum of blocks that prefer a register.
  BlockFrequency BiasP;

  /// Value - Output value of this node computed from the Bias and links.
  /// This is always on of the values {-1, 0, 1}. A positive number means the
  /// variable should go in a register through this bundle.
  int Value;

  typedef SmallVector<std::pair<BlockFrequency, unsigned>, 4> LinkVector;

  /// Links - (Weight, BundleNo) for all transparent blocks connecting to other
  /// bundles. The weights are all positive block frequencies.
  LinkVector Links;

  /// SumLinkWeights - Cached sum of the weights of all links + ThresHold.
  BlockFrequency SumLinkWeights;

  /// preferReg - Return true when this node prefers to be in a register.
  bool preferReg() const {
    // Undecided nodes (Value==0) go on the stack.
    return Value > 0;
  }

  /// mustSpill - Return True if this node is so biased that it must spill.
  bool mustSpill() const {
    // We must spill if Bias < -sum(weights) or the MustSpill flag was set.
    // BiasN is saturated when MustSpill is set, make sure this still returns
    // true when the RHS saturates. Note that SumLinkWeights includes Threshold.
    return BiasN >= BiasP + SumLinkWeights;
  }

  /// clear - Reset per-query data, but preserve frequencies that only depend on
  // the CFG.
  void clear(const BlockFrequency &Threshold) {
    BiasN = BiasP = Value = 0;
    SumLinkWeights = Threshold;
    Links.clear();
  }

  /// addLink - Add a link to bundle b with weight w.
  void addLink(unsigned b, BlockFrequency w) {
    // Update cached sum.
    SumLinkWeights += w;

    // There can be multiple links to the same bundle, add them up.
    for (LinkVector::iterator I = Links.begin(), E = Links.end(); I != E; ++I)
      if (I->second == b) {
        I->first += w;
        return;
      }
    // This must be the first link to b.
    Links.push_back(std::make_pair(w, b));
  }

  /// addBias - Bias this node.
  void addBias(BlockFrequency freq, BorderConstraint direction) {
    switch (direction) {
    default:
      break;
    case PrefReg:
      BiasP += freq;
      break;
    case PrefSpill:
      BiasN += freq;
      break;
    case MustSpill:
      BiasN = BlockFrequency::getMaxFrequency();
      break;
    }
  }

  /// update - Recompute Value from Bias and Links. Return true when node
  /// preference changes.
  bool update(const Node nodes[], const BlockFrequency &Threshold) {
    // Compute the weighted sum of inputs.
    BlockFrequency SumN = BiasN;
    BlockFrequency SumP = BiasP;
    for (LinkVector::iterator I = Links.begin(), E = Links.end(); I != E; ++I) {
      if (nodes[I->second].Value == -1)
        SumN += I->first;
      else if (nodes[I->second].Value == 1)
        SumP += I->first;
    }

    // Each weighted sum is going to be less than the total frequency of the
    // bundle. Ideally, we should simply set Value = sign(SumP - SumN), but we
    // will add a dead zone around 0 for two reasons:
    //
    //  1. It avoids arbitrary bias when all links are 0 as is possible during
    //     initial iterations.
    //  2. It helps tame rounding errors when the links nominally sum to 0.
    //
    bool Before = preferReg();
    if (SumN >= SumP + Threshold)
      Value = -1;
    else if (SumP >= SumN + Threshold)
      Value = 1;
    else
      Value = 0;
    return Before != preferReg();
  }
};

bool SpillPlacement::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;
  bundles = &getAnalysis<EdgeBundles>();
  loops = &getAnalysis<MachineLoopInfo>();

  assert(!nodes && "Leaking node array");
  nodes = new Node[bundles->getNumBundles()];

  // Compute total ingoing and outgoing block frequencies for all bundles.
  BlockFrequencies.resize(mf.getNumBlockIDs());
  MBFI = &getAnalysis<MachineBlockFrequencyInfo>();
  setThreshold(MBFI->getEntryFreq());
  for (MachineFunction::iterator I = mf.begin(), E = mf.end(); I != E; ++I) {
    unsigned Num = I->getNumber();
    BlockFrequencies[Num] = MBFI->getBlockFreq(I);
  }

  // We never change the function.
  return false;
}

void SpillPlacement::releaseMemory() {
  delete[] nodes;
  nodes = nullptr;
}

/// activate - mark node n as active if it wasn't already.
void SpillPlacement::activate(unsigned n) {
  if (ActiveNodes->test(n))
    return;
  ActiveNodes->set(n);
  nodes[n].clear(Threshold);

  // Very large bundles usually come from big switches, indirect branches,
  // landing pads, or loops with many 'continue' statements. It is difficult to
  // allocate registers when so many different blocks are involved.
  //
  // Give a small negative bias to large bundles such that a substantial
  // fraction of the connected blocks need to be interested before we consider
  // expanding the region through the bundle. This helps compile time by
  // limiting the number of blocks visited and the number of links in the
  // Hopfield network.
  if (bundles->getBlocks(n).size() > 100) {
    nodes[n].BiasP = 0;
    nodes[n].BiasN = (MBFI->getEntryFreq() / 16);
  }
}

/// \brief Set the threshold for a given entry frequency.
///
/// Set the threshold relative to \c Entry.  Since the threshold is used as a
/// bound on the open interval (-Threshold;Threshold), 1 is the minimum
/// threshold.
void SpillPlacement::setThreshold(const BlockFrequency &Entry) {
  // Apparently 2 is a good threshold when Entry==2^14, but we need to scale
  // it.  Divide by 2^13, rounding as appropriate.
  uint64_t Freq = Entry.getFrequency();
  uint64_t Scaled = (Freq >> 13) + bool(Freq & (1 << 12));
  Threshold = std::max(UINT64_C(1), Scaled);
}

/// addConstraints - Compute node biases and weights from a set of constraints.
/// Set a bit in NodeMask for each active node.
void SpillPlacement::addConstraints(ArrayRef<BlockConstraint> LiveBlocks) {
  for (ArrayRef<BlockConstraint>::iterator I = LiveBlocks.begin(),
       E = LiveBlocks.end(); I != E; ++I) {
    BlockFrequency Freq = BlockFrequencies[I->Number];

    // Live-in to block?
    if (I->Entry != DontCare) {
      unsigned ib = bundles->getBundle(I->Number, 0);
      activate(ib);
      nodes[ib].addBias(Freq, I->Entry);
    }

    // Live-out from block?
    if (I->Exit != DontCare) {
      unsigned ob = bundles->getBundle(I->Number, 1);
      activate(ob);
      nodes[ob].addBias(Freq, I->Exit);
    }
  }
}

/// addPrefSpill - Same as addConstraints(PrefSpill)
void SpillPlacement::addPrefSpill(ArrayRef<unsigned> Blocks, bool Strong) {
  for (ArrayRef<unsigned>::iterator I = Blocks.begin(), E = Blocks.end();
       I != E; ++I) {
    BlockFrequency Freq = BlockFrequencies[*I];
    if (Strong)
      Freq += Freq;
    unsigned ib = bundles->getBundle(*I, 0);
    unsigned ob = bundles->getBundle(*I, 1);
    activate(ib);
    activate(ob);
    nodes[ib].addBias(Freq, PrefSpill);
    nodes[ob].addBias(Freq, PrefSpill);
  }
}

void SpillPlacement::addLinks(ArrayRef<unsigned> Links) {
  for (ArrayRef<unsigned>::iterator I = Links.begin(), E = Links.end(); I != E;
       ++I) {
    unsigned Number = *I;
    unsigned ib = bundles->getBundle(Number, 0);
    unsigned ob = bundles->getBundle(Number, 1);

    // Ignore self-loops.
    if (ib == ob)
      continue;
    activate(ib);
    activate(ob);
    if (nodes[ib].Links.empty() && !nodes[ib].mustSpill())
      Linked.push_back(ib);
    if (nodes[ob].Links.empty() && !nodes[ob].mustSpill())
      Linked.push_back(ob);
    BlockFrequency Freq = BlockFrequencies[Number];
    nodes[ib].addLink(ob, Freq);
    nodes[ob].addLink(ib, Freq);
  }
}

bool SpillPlacement::scanActiveBundles() {
  Linked.clear();
  RecentPositive.clear();
  for (int n = ActiveNodes->find_first(); n>=0; n = ActiveNodes->find_next(n)) {
    nodes[n].update(nodes, Threshold);
    // A node that must spill, or a node without any links is not going to
    // change its value ever again, so exclude it from iterations.
    if (nodes[n].mustSpill())
      continue;
    if (!nodes[n].Links.empty())
      Linked.push_back(n);
    if (nodes[n].preferReg())
      RecentPositive.push_back(n);
  }
  return !RecentPositive.empty();
}

/// iterate - Repeatedly update the Hopfield nodes until stability or the
/// maximum number of iterations is reached.
/// @param Linked - Numbers of linked nodes that need updating.
void SpillPlacement::iterate() {
  // First update the recently positive nodes. They have likely received new
  // negative bias that will turn them off.
  while (!RecentPositive.empty())
    nodes[RecentPositive.pop_back_val()].update(nodes, Threshold);

  if (Linked.empty())
    return;

  // Run up to 10 iterations. The edge bundle numbering is closely related to
  // basic block numbering, so there is a strong tendency towards chains of
  // linked nodes with sequential numbers. By scanning the linked nodes
  // backwards and forwards, we make it very likely that a single node can
  // affect the entire network in a single iteration. That means very fast
  // convergence, usually in a single iteration.
  for (unsigned iteration = 0; iteration != 10; ++iteration) {
    // Scan backwards, skipping the last node when iteration is not zero. When
    // iteration is not zero, the last node was just updated.
    bool Changed = false;
    for (SmallVectorImpl<unsigned>::const_reverse_iterator I =
           iteration == 0 ? Linked.rbegin() : std::next(Linked.rbegin()),
           E = Linked.rend(); I != E; ++I) {
      unsigned n = *I;
      if (nodes[n].update(nodes, Threshold)) {
        Changed = true;
        if (nodes[n].preferReg())
          RecentPositive.push_back(n);
      }
    }
    if (!Changed || !RecentPositive.empty())
      return;

    // Scan forwards, skipping the first node which was just updated.
    Changed = false;
    for (SmallVectorImpl<unsigned>::const_iterator I =
           std::next(Linked.begin()), E = Linked.end(); I != E; ++I) {
      unsigned n = *I;
      if (nodes[n].update(nodes, Threshold)) {
        Changed = true;
        if (nodes[n].preferReg())
          RecentPositive.push_back(n);
      }
    }
    if (!Changed || !RecentPositive.empty())
      return;
  }
}

void SpillPlacement::prepare(BitVector &RegBundles) {
  Linked.clear();
  RecentPositive.clear();
  // Reuse RegBundles as our ActiveNodes vector.
  ActiveNodes = &RegBundles;
  ActiveNodes->clear();
  ActiveNodes->resize(bundles->getNumBundles());
}

bool
SpillPlacement::finish() {
  assert(ActiveNodes && "Call prepare() first");

  // Write preferences back to ActiveNodes.
  bool Perfect = true;
  for (int n = ActiveNodes->find_first(); n>=0; n = ActiveNodes->find_next(n))
    if (!nodes[n].preferReg()) {
      ActiveNodes->reset(n);
      Perfect = false;
    }
  ActiveNodes = nullptr;
  return Perfect;
}
