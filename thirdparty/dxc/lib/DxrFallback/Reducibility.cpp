#include "Reducibility.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Scalar.h"

#include "LLVMUtils.h"

#include <fstream>
#include <vector>
#include <map>

#define DBGS errs
//#define DBGS dbgs

using namespace llvm;

struct Node
{
  SetVector<Node*> in;
  SetVector<Node*> out;
  SetVector<BasicBlock*> blocks; // block 0 dominates all others in this node
  size_t numInstructions = 0;

  Node() {}
  Node(BasicBlock* B) { insert(B); }

  void insert(BasicBlock* B)
  {
    numInstructions += B->size();
    blocks.insert(B);
  }
};


static void printDotGraph(const std::vector<Node*> nodes, const std::string& filename)
{
  DBGS() << "Writing " << filename << " ...";
  std::ofstream out(filename);
  if (!out)
  {
    DBGS() << "FAILED\n";
    return;
  }

  // Give nodes a numerical index to make the output cleaner
  std::map<Node*, int> idxMap;
  for (size_t i = 0; i < nodes.size(); ++i)
    idxMap[nodes[i]] = i;


  // Error check - make sure that all the out/in nodes are in the map
  for (Node* N : nodes)
  {
    for (Node* P : N->in)
    {
      if (idxMap.find(P) == idxMap.end())
        DBGS() << "MISSING INPUT NODE\n";
      if (P->out.count(N) == 0)
        DBGS() << "MISSING OUTGOING EDGE FROM PREDECESSOR.\n";
    }
    for (Node* S : N->out)
    {
      if (idxMap.find(S) == idxMap.end())
        DBGS() << "MISSING OUTPUT NODE\n";
      if (S->in.count(N) == 0)
        DBGS() << "MISSING INCOMING EDGE FROM SUCCESSOR.\n";
    }
  }


  // Print header
  out << "digraph g {\n";
  out << "node [\n";
  out << "  fontsize = \"12\"\n";
  out << "  labeljust = \"l\"\n";
  out << "]\n";

  for (unsigned i = 0; i < nodes.size(); ++i)
  {
    Node* N = nodes[i];

    // node
    out << "  N" << i << " [shape=record,label=\"";
    for (BasicBlock* B : N->blocks)
      out << B->getName().str() << "\\n";
    out << "\"];\n";

    // out edges
    for (Node* S : N->out)
      out << "  N" << i << " -> N" << idxMap[S] << ";\n";

    // in edges
    //for( Node* P : N->in )    
    //  out << "  N" << idxMap[P] << " -> N" << i << " [style=dotted];\n";
  }

  out << "}\n";

  DBGS() << "\n";
}

static void printDotGraph(const std::vector<Node*> nodes, Function* F, int step)
{
  printDotGraph(nodes, ("red." + F->getName() + "_" + std::to_string(step) + ".dot").str());
}


static Node* split(Node* N, std::map<BasicBlock*, Node*>& bbToNode, bool firstSplit)
{
  // Remove one predecessor P from N
  assert(N->in.size() > 1);
  Node* P = N->in.pop_back_val();
  P->out.remove(N);

  // Point P to the clone of N, Np
  Node* Np = new Node();
  P->out.insert(Np);
  Np->in.insert(P);

  // Copy successors of N to Np
  for (Node* S : N->out)
  {
    Np->out.insert(S);
    S->in.insert(Np);
  }

#if 1
  // Run reg2mem on the whole function so we don't have to deal with phis
  if (firstSplit)
  {
    runPasses(N->blocks[0]->getParent(), {
      createDemoteRegisterToMemoryPass()
    });
  }


  // Clone N into Np
  ValueToValueMapTy VMap;
  for (BasicBlock* B : N->blocks)
  {
    BasicBlock* Bp = CloneBasicBlock(B, VMap, ".c", B->getParent());
    Np->insert(Bp);
    VMap[B] = Bp;
  }
  for (BasicBlock* B : Np->blocks)
    for (Instruction& I : *B)
      RemapInstruction(&I, VMap, RF_NoModuleLevelChanges | RF_IgnoreMissingEntries);

  // Remap terminators of P from N to Np
  for (BasicBlock* B : P->blocks)
    RemapInstruction(B->getTerminator(), VMap, RF_NoModuleLevelChanges | RF_IgnoreMissingEntries);

#else
  // Clone N into Np
  ValueToValueMapTy VMap;
  for (BasicBlock* B : N->blocks)
  {
    BasicBlock* Bp = CloneBasicBlock(B, VMap, ".c", B->getParent());
    Np->insert(Bp);
    VMap[B] = Bp;
  }
  for (BasicBlock* B : Np->blocks)
    for (Instruction& I : *B)
      RemapInstruction(&I, VMap, RF_NoModuleLevelChanges | RF_IgnoreMissingEntries);


  // Remove incoming values from phis in Np that don't come from actual predecessors
  BasicBlock* NpEntry = Np->blocks[0];
  std::set<BasicBlock*> predSet(pred_begin(NpEntry), pred_end(NpEntry));
  auto I = NpEntry->begin();
  while (PHINode* phi = dyn_cast<PHINode>(I++))
  {
    if (phi->getNumIncomingValues() == predSet.size())
      continue;
    for (unsigned i = 0; i < phi->getNumIncomingValues(); )
    {
      BasicBlock* B = phi->getIncomingBlock(i);
      if (!predSet.count(B))
      {
        phi->removeIncomingValue(B);
        continue;
      }
      ++i;
    }
  }


  // Remove phi references to P in N. (Do this before remapping terminators.)
  BasicBlock* Nentry = N->blocks[0];
  for (BasicBlock* PB : predecessors(Nentry))
  {
    if (P->blocks.count(PB))
      Nentry->removePredecessor(PB);
  }

  // Remap terminators of P from N to Np
  for (BasicBlock* B : P->blocks)
    RemapInstruction(B->getTerminator(), VMap, RF_NoModuleLevelChanges | RF_IgnoreMissingEntries);


  // Update phis in successors of Np.
  // There are several cases for a value Vs reaching S. Vs may be defined in N and
  // a clone Vsp in Np or only passing through one or the other. Furthermore, Vs may 
  // either appear in a phi in the entry block of S or not.
  // 1) Vs defined in N (and clone Vsp in Np) and in phi:
  //    Add incoming value [Vsp, Bp] for cloned value Vsp from predecessor basic
  //    block Bp in Np wherever [Vs, B] appears
  // 2) Vs defined in N (and clone Vsp in Np) and not in phi:
  //    Add phi [Vs,B],[Vsp,Bp] if Vs reaches a use in or through S
  // 3) Vs passing through N or Np and in phi
  //    Change [Vs,B] to [Vs,Bp] in phis in S if Vs reached S through P
  // 4) Vs passing through N or Np and not in a phi
  //    Do nothing
  // 
  // TODO: Only 1) is implemented below and it isn't checking for definition in N
  for (Node* S : Np->out)
  {
    BasicBlock* Sentry = S->blocks[0];
    auto I = Sentry->begin();
    while (PHINode* phi = dyn_cast<PHINode>(I++))
    {
      for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i)
      {
        BasicBlock* B = phi->getIncomingBlock(i);
        if (N->blocks.count(B))
        {
          Value* V = phi->getIncomingValue(i);
          Value* Vp = VMap[V];
          if (!Vp)
            Vp = V; // Def not in N
          BasicBlock* Bp = dyn_cast<BasicBlock>(VMap[B]);
          phi->addIncoming(Vp, Bp);
        }
      }
    }
  }
#endif

  return Np;
}

// Returns the number of splits
int makeReducible(Function* F)
{
  // Break critical edges now in case we need to do mem2reg in split(). mem2reg
  // will break critical edges and the CFG needs to remain unchanged.
  runPasses(F, {
    createBreakCriticalEdgesPass()
  });

  // initialize nodes
  std::vector<Node*> nodes;
  std::map<BasicBlock*, Node*> bbToNode;
  for (BasicBlock& B : *F)
  {
    nodes.push_back(new Node(&B));
    bbToNode[&B] = nodes.back();
  }

  // initialize edges
  for (Node* N : nodes)
  {
    for (BasicBlock* B : successors(N->blocks[0]))
    {
      Node* BN = bbToNode[B];
      N->out.insert(BN);
      BN->in.insert(N);
    }
  }

  int step = 0;
  bool print = false;
  if (print) printDotGraph(nodes, F, step++);

  int numSplits = 0;
  while (!nodes.empty())
  {
    bool changed;
    do
    {
  // It might more efficient to use a worklist based implementation instead
  // of iterating over the vector.
      changed = false;
      for (size_t i = 0; i < nodes.size(); )
      {
        Node* N = nodes[i];

        // Remove self references
        if (N->in.count(N))
        {
          N->in.remove(N);
          N->out.remove(N);
          changed = true;
        }

        // Remove singletons
        if (N->in.size() == 0 && N->out.size() == 0)
        {
          nodes.erase(nodes.begin() + i);
          changed = true;
          if (print) printDotGraph(nodes, F, step++);
          continue;
        }

        // Remove nodes with only one incoming edge
        if (N->in.size() == 1)
        {
          // fold into predecessor
          Node* P = N->in.back();
          P->blocks.insert(N->blocks.begin(), N->blocks.end());
          P->out.remove(N);
          for (Node* S : N->out)
          {
            S->in.remove(N);
            P->out.insert(S);
            S->in.insert(P);
          }
          P->numInstructions += N->numInstructions;
          nodes.erase(nodes.begin() + i);
          changed = true;
          if (print) printDotGraph(nodes, F, step++);
          continue;
        }

        i++;
      }
    } while (changed);

    if (!nodes.empty())
    {
      // Duplicate the smallest node with more than one incoming edge. Better 
      // methods exist for picking the node to split, e.g. "Making Graphs Reducible
      // with Controlled Node Splitting" by Janssen and Corporaal.
      size_t idxMin = ~0;
      for (size_t i = 0; i < nodes.size(); ++i)
      {
        if (nodes[i]->in.size() <= 1)
          continue;

        if (idxMin == ~0u || nodes[i]->numInstructions < nodes[idxMin]->numInstructions)
          idxMin = i;
      }
      nodes.push_back(split(nodes[idxMin], bbToNode, numSplits == 0));
      numSplits++;
      if (print) printDotGraph(nodes, F, step++);
    }
  }
  return numSplits;
}
