//===-- BranchFolding.h - Fold machine code branch instructions -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_BRANCHFOLDING_H
#define LLVM_LIB_CODEGEN_BRANCHFOLDING_H

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/Support/BlockFrequency.h"
#include <vector>

namespace llvm {
  class MachineBlockFrequencyInfo;
  class MachineBranchProbabilityInfo;
  class MachineFunction;
  class MachineModuleInfo;
  class RegScavenger;
  class TargetInstrInfo;
  class TargetRegisterInfo;

  class LLVM_LIBRARY_VISIBILITY BranchFolder {
  public:
    explicit BranchFolder(bool defaultEnableTailMerge, bool CommonHoist,
                          const MachineBlockFrequencyInfo &MBFI,
                          const MachineBranchProbabilityInfo &MBPI);

    bool OptimizeFunction(MachineFunction &MF,
                          const TargetInstrInfo *tii,
                          const TargetRegisterInfo *tri,
                          MachineModuleInfo *mmi);
  private:
    class MergePotentialsElt {
      unsigned Hash;
      MachineBasicBlock *Block;
    public:
      MergePotentialsElt(unsigned h, MachineBasicBlock *b)
        : Hash(h), Block(b) {}

      unsigned getHash() const { return Hash; }
      MachineBasicBlock *getBlock() const { return Block; }

      void setBlock(MachineBasicBlock *MBB) {
        Block = MBB;
      }

      bool operator<(const MergePotentialsElt &) const;
    };
    typedef std::vector<MergePotentialsElt>::iterator MPIterator;
    std::vector<MergePotentialsElt> MergePotentials;
    SmallPtrSet<const MachineBasicBlock*, 2> TriedMerging;

    class SameTailElt {
      MPIterator MPIter;
      MachineBasicBlock::iterator TailStartPos;
    public:
      SameTailElt(MPIterator mp, MachineBasicBlock::iterator tsp)
        : MPIter(mp), TailStartPos(tsp) {}

      MPIterator getMPIter() const {
        return MPIter;
      }
      MergePotentialsElt &getMergePotentialsElt() const {
        return *getMPIter();
      }
      MachineBasicBlock::iterator getTailStartPos() const {
        return TailStartPos;
      }
      unsigned getHash() const {
        return getMergePotentialsElt().getHash();
      }
      MachineBasicBlock *getBlock() const {
        return getMergePotentialsElt().getBlock();
      }
      bool tailIsWholeBlock() const {
        return TailStartPos == getBlock()->begin();
      }

      void setBlock(MachineBasicBlock *MBB) {
        getMergePotentialsElt().setBlock(MBB);
      }
      void setTailStartPos(MachineBasicBlock::iterator Pos) {
        TailStartPos = Pos;
      }
    };
    std::vector<SameTailElt> SameTails;

    bool EnableTailMerge;
    bool EnableHoistCommonCode;
    const TargetInstrInfo *TII;
    const TargetRegisterInfo *TRI;
    MachineModuleInfo *MMI;
    RegScavenger *RS;

    /// \brief This class keeps track of branch frequencies of newly created
    /// blocks and tail-merged blocks.
    class MBFIWrapper {
    public:
      MBFIWrapper(const MachineBlockFrequencyInfo &I) : MBFI(I) {}
      BlockFrequency getBlockFreq(const MachineBasicBlock *MBB) const;
      void setBlockFreq(const MachineBasicBlock *MBB, BlockFrequency F);

    private:
      const MachineBlockFrequencyInfo &MBFI;
      DenseMap<const MachineBasicBlock *, BlockFrequency> MergedBBFreq;
    };

    MBFIWrapper MBBFreqInfo;
    const MachineBranchProbabilityInfo &MBPI;

    bool TailMergeBlocks(MachineFunction &MF);
    bool TryTailMergeBlocks(MachineBasicBlock* SuccBB,
                       MachineBasicBlock* PredBB);
    void setCommonTailEdgeWeights(MachineBasicBlock &TailMBB);
    void MaintainLiveIns(MachineBasicBlock *CurMBB,
                         MachineBasicBlock *NewMBB);
    void ReplaceTailWithBranchTo(MachineBasicBlock::iterator OldInst,
                                 MachineBasicBlock *NewDest);
    MachineBasicBlock *SplitMBBAt(MachineBasicBlock &CurMBB,
                                  MachineBasicBlock::iterator BBI1,
                                  const BasicBlock *BB);
    unsigned ComputeSameTails(unsigned CurHash, unsigned minCommonTailLength,
                              MachineBasicBlock *SuccBB,
                              MachineBasicBlock *PredBB);
    void RemoveBlocksWithHash(unsigned CurHash, MachineBasicBlock* SuccBB,
                                                MachineBasicBlock* PredBB);
    bool CreateCommonTailOnlyBlock(MachineBasicBlock *&PredBB,
                                   MachineBasicBlock *SuccBB,
                                   unsigned maxCommonTailLength,
                                   unsigned &commonTailIndex);

    bool OptimizeBranches(MachineFunction &MF);
    bool OptimizeBlock(MachineBasicBlock *MBB);
    void RemoveDeadBlock(MachineBasicBlock *MBB);
    bool OptimizeImpDefsBlock(MachineBasicBlock *MBB);

    bool HoistCommonCode(MachineFunction &MF);
    bool HoistCommonCodeInSuccs(MachineBasicBlock *MBB);
  };
}

#endif /* LLVM_CODEGEN_BRANCHFOLDING_HPP */
