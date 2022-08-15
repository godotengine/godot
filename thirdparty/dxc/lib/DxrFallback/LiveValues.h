#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/BasicBlock.h"

#include <map>
#include <set>
#include <vector>

namespace llvm
{
  class AllocaInst;
  class BasicBlock;
  class Function;
  class Instruction;
  class Use;
  class Value;
}

typedef std::set<llvm::BasicBlock*> BasicBlockSet;
typedef llvm::SetVector<llvm::Instruction*> InstructionSetVector;

// Compute live values at specified instructions.
class LiveValues
{
public:
  LiveValues(llvm::ArrayRef<llvm::Instruction*> computeLiveAt);

  // Compute live values at specified instructions (computeLiveAt)
  void run();

  // Returns all values that are live at the index.
  const InstructionSetVector& getLiveValues(unsigned int index) const { return m_liveSets[index]; }

  // Returns all live values, excluding allocas.
  const InstructionSetVector& getAllLiveValues() const { return m_allLiveSet; }

  // Update the live sets using the map
  void remapLiveValues(llvm::DenseMap<llvm::Instruction*, llvm::Instruction*>& imap);

  typedef llvm::SetVector<unsigned int> Indices;

  // Return all indices at which the given value is live.
  const Indices* getIndicesWhereLive(const llvm::Value* value) const;


  // For the two given values, check if they are both live at any of the
  // marker instructions. This does not perform a true "lifetime overlap"
  // test, it considers values to be disjoint if they have disjoint sets of
  // markers.
  // For example, value A is live at call sites 0, 1, 2, value B is live at
  // 3, 4, where A is used for the last time between 2 and 3 and B is defined
  // before that use. A and B will be considered "disjoint" in the sense of
  // this method, even though the lifetimes of their values overlap.
  bool liveInDisjointRegions(const llvm::Value* valueA, const llvm::Value* valueB) const;

  // Return true if the given value is live at the given index.
  bool getLiveAtIndex(const llvm::Value* value, unsigned int index) const;

  // Update the analysis manually. Use only if you know exactly what you are
  // doing and document the reason thoroughly.
  void setLiveAtIndex(llvm::Value* value, unsigned int index, bool live);
  void setLiveAtAllIndices(llvm::Value* value, bool live);
  void setIndicesWhereLive(llvm::Value* value, const Indices* indices);


private:
  llvm::Function*                   m_function = nullptr;
  std::vector<InstructionSetVector> m_liveSets;
  InstructionSetVector              m_allLiveSet;
  llvm::SmallSet<llvm::BasicBlock*, 8>             m_activeBlocks;
  llvm::DenseMap<llvm::Instruction*, unsigned int> m_computeLiveAtIndex;
  llvm::DenseMap<const llvm::Value*, Indices>      m_liveAtIndices;

  typedef llvm::SmallSet<llvm::BasicBlock*, 8> BlockSet;

  void markLiveRange(llvm::Instruction* value, llvm::BasicBlock::iterator begin, llvm::BasicBlock::iterator end);
  void upAndMark(llvm::Instruction* v, llvm::Use& use, BlockSet& scanned);
};
