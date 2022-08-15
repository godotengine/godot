#include "LiveValues.h"

#include "llvm/IR/CFG.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

static void applyMapping(InstructionSetVector& iset, llvm::DenseMap<llvm::Instruction *, llvm::Instruction *>& imap)
{
  // There will be probably be few entries in the imap, so apply them one at a time to the iset.
  for (auto& kv : imap)
  {
    if (iset.count(kv.first) != 0)
    {
      iset.remove(kv.first);
      iset.insert(kv.second);
    }
  }
}

// Compute liveness of a value at basic blocks. Roughly based on
// Algorithm 6 & 7 from the paper "Computing Liveness Sets for SSA-
// Form Programs" by Brander et al., 2011.

LiveValues::LiveValues(ArrayRef<Instruction*> computeLiveAt)
{
  m_liveSets.resize(computeLiveAt.size());

  // Build index and set of active blocks
  for (unsigned int i = 0; i < computeLiveAt.size(); i++)
  {
    Instruction* v = computeLiveAt[i];
    m_computeLiveAtIndex.insert(std::make_pair(v, i));

    m_activeBlocks.insert(v->getParent());
  }

  if (computeLiveAt.size() > 0)
  {
      m_function = computeLiveAt[0]->getParent()->getParent();
  }
}

// Go over all the instructions between begin (included) and end (excluded) and mark the given value
// live for code locations contained in the given range.
void LiveValues::markLiveRange(Instruction* value, BasicBlock::iterator begin, BasicBlock::iterator end)
{
  BasicBlock* B = begin->getParent();

  if (m_activeBlocks.count(B) == 0)
    return;  // Nothing to mark in this block

  for (BasicBlock::iterator I = begin; I != end; ++I)
  {
    if (m_computeLiveAtIndex.count(I))
    {
      // Mark this value
      unsigned int index = m_computeLiveAtIndex[I];
      m_liveSets[index].insert(value);
      m_allLiveSet.insert(value);
      // Also store for each value where it is live.
      m_liveAtIndices[value].insert(index);
    }
  }
}

void LiveValues::upAndMark(Instruction* def, Use& use, BlockSet& scanned)
{
  // Determine the starting point for the backwards search.
  // (Remember that Use represents an edge between the definition of a value and its use)
  // In the case in which the user of the use is a phi node we start the search from the terminator
  // of the preceding block.
  // This allows to avoid going through loop back-edges in cases like these:
  //                 |
  //                 | (y)
  //                 v
  //          -----------------
  //     (x)  | z = phi(x, y) |
  //    ----> | ...           |
  //    |     | x = z + 1     |
  //    |     -----------------
  //    |             |
  //    |             |
  //    |             |
  //    |             v
  //    |     -----------------
  //    |     |               |
  //    ------| INDIRECT CALL |
  //          |               |
  //          -----------------
  //                  | (Start the search for the definition of x (backwards) from here!)
  //                  v
  //
  // Notice that here x is live across the call. This case is tricky because the def comes 'after'
  // the use. The def still dominates the use because phi nodes logically use their input values on the
  // edges, i.e. on the terminator of the preceding blocks.
  //
  // This has the advantage of being able to traverse edges strictly backwards.

  Instruction* startingPoint = dyn_cast<Instruction>(use.getUser());
  if (PHINode* usePHI = dyn_cast<PHINode>(startingPoint))
  {
    BasicBlock* predecessor = usePHI->getIncomingBlock(use);
    startingPoint = predecessor->getTerminator();
  }

  BasicBlock* startingPointBB = startingPoint->getParent();
  BasicBlock* defBB = def->getParent();

  // Start a bottom-up recursive search from startingPoint to the definition of the current value.
  // Mark all the code ranges that we encounter on the way a having the current value 'live'.
  // 'scanned' contains the blocks that we have scanned to the bottom of the block and the we know
  // already having the current value 'live'.

  SmallVector<BasicBlock*, 16> worklist;
  worklist.push_back(startingPointBB);

  BlockSet visited;

  while (!worklist.empty())
  {
    BasicBlock* B = worklist.pop_back_val();

    if (scanned.count(B) != 0)
      continue;

    // We have reached the block that contains the definition of the value. We are done for this
    // branch of the search.
    if (B == defBB)
    {
      if (defBB == startingPointBB)
      {
        // If the first block that we visit is also the last mark only the range of instructions
        // between the def and the starting point.
        //    -----------------
        //    |               |
        //    | x = // def    |  <--
        //    |               |    !
        //    |               |    ! This is the range in which x is live.
        //    |               |    !
        //    | = x // use    |  <--
        //    |               |
        //    -----------------

        markLiveRange(def, ++BasicBlock::iterator(def), BasicBlock::iterator(startingPoint));
      }
      else
      {
        markLiveRange(def, ++BasicBlock::iterator(def), defBB->end());
        scanned.insert(B);
      }
    }
    else
    {
      if (B == startingPointBB)
      {
        // We are in the starting-point block.
        // This can mean two things:
        // 1. We are in the first iteration, mark the range between begin and starting point as
        // live.
        if (visited.count(B) == 0)
        {
          markLiveRange(def, B->begin(), BasicBlock::iterator(startingPoint));
        }
        // 2. We came back here because the starting point is in a loop.
        // In this case mark the whole block as live range and don't come back anymore.
        else
        {
          markLiveRange(def, B->begin(), B->end());
          scanned.insert(B);
        }

        // The if statement above allows to manage situations like this:
        //         BB0
        //        -----------------
        //        | x = ...       |
        //        -----------------
        //                |
        //                |
        //                |
        //         BB1    v
        //        -----------------<--                     <--
        //        |               |  !                       !
        //  ----->|               |  ! First range marked    !
        //  |     |               |  !                       !
        //  |     | ... = x       |<--                       ! Second and final range marked
        //  |     |               |                          !
        //  |     | INDIRECT CALL |                          !
        //  |     |               |                          !
        //  |     -----------------                        <--
        //  |              |
        //  ---------------
        // x is defined outside a loop and used inside a loop. This means that it is live inside the
        // whole loop.
        // So, we first mark the range from the use of x to the top of BB1 and, when we visit BB1
        // again (because BB1 is a predecessor of BB1) we mark the whole block as live range.
        // <rant>
        // This case could have been managed much more easily and efficiently if we had access to
        // LLVM LoopInfo analysis pass.
        // We could have done the following: x is uses in a loop and defined outside of it => mark
        // the whole loop body as live range.
        // </rant>
      }
      else
      {
        // We are in an intermediate block on the way to the definition mark it, all as live range.
        markLiveRange(def, B->begin(), B->end());
        scanned.insert(B);
      }

      visited.insert(B);

      for (pred_iterator P = pred_begin(B), PE = pred_end(B); P != PE; ++P)
      {
        worklist.push_back(*P);
      }
    }
  }
}

void LiveValues::run()
{
  if (m_computeLiveAtIndex.empty())
    return;

  // for each variable v do
  for (inst_iterator I = inst_begin(m_function), E = inst_end(m_function); I != E; ++I)
  {
    Instruction* v = &*I;
    assert(v->getParent()->getParent() == m_function);

    // for each block B where v is used do
    BlockSet scanned;
    for (Value::use_iterator U = v->use_begin(), UE = v->use_end(); U != UE; ++U)
    {
      Instruction* user = cast<Instruction>(U->getUser());
      assert(user->getParent()->getParent() == m_function);
      (void)user;

      upAndMark(v, *U, scanned);
    }
  }
}

void LiveValues::remapLiveValues(llvm::DenseMap<llvm::Instruction*, llvm::Instruction*>& imap)
{
  applyMapping(m_allLiveSet, imap);
  for (auto& liveSet : m_liveSets)
    applyMapping(liveSet, imap);
}

const LiveValues::Indices* LiveValues::getIndicesWhereLive(const Value* value) const
{
  const auto& iter = m_liveAtIndices.find(value);
  if (iter == m_liveAtIndices.end())
    return nullptr;
  return &iter->second;
}

void LiveValues::setIndicesWhereLive(Value* value, const Indices* indices)
{
  for (unsigned int idx : *indices)
    setLiveAtIndex(value, idx, true);
}

bool LiveValues::liveInDisjointRegions(const Value* valueA, const Value* valueB) const
{
  const Indices* indicesA = getIndicesWhereLive(valueA);
  if (!indicesA)
    return true;

  const Indices* indicesB = getIndicesWhereLive(valueB);
  if (!indicesB)
    return true;

  for (const unsigned int index : *indicesA)
  {
    if (indicesB->count(index))
      return false;
  }

  return true;
}

void LiveValues::setLiveAtIndex(Value* value, unsigned int index, bool live)
{
  assert(index <= m_computeLiveAtIndex.size());
  if (live)
  {
    m_liveAtIndices[value].insert(index);
    Instruction* inst = cast<Instruction>(value);
    m_liveSets[index].insert(inst);
    m_allLiveSet.insert(inst);
  }
  else
  {
    m_liveAtIndices[value].remove(index);
    Instruction* inst = cast<Instruction>(value);
    m_liveSets[index].remove(inst);
    if (m_liveAtIndices[value].empty())
      m_allLiveSet.remove(inst);
  }
}

void LiveValues::setLiveAtAllIndices(llvm::Value* value, bool live)
{
  Instruction* inst = cast<Instruction>(value);
  if (live)
  {
    for (unsigned int index = 0; index < m_computeLiveAtIndex.size(); ++index)
    {
      m_liveAtIndices[value].insert(index);
      m_liveSets[index].insert(inst);
    }
    m_allLiveSet.insert(inst);
  }
  else
  {
    for (unsigned int index = 0; index < m_computeLiveAtIndex.size(); ++index)
    {
      m_liveAtIndices[value].remove(index);
      m_liveSets[index].remove(inst);
    }
    if (m_liveAtIndices[value].empty())
      m_allLiveSet.remove(inst);
  }
}

bool LiveValues::getLiveAtIndex(const Value* value, unsigned int index) const
{
  assert(index <= m_computeLiveAtIndex.size());
  const auto& it = m_liveAtIndices.find(value);
  if (it == m_liveAtIndices.end())
    return false;
  return (it->second.count(index) != 0);
}
