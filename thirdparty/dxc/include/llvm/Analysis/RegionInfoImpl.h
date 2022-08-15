//===- RegionInfoImpl.h - SESE region detection analysis --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Detects single entry single exit regions in the control flow graph.
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_REGIONINFOIMPL_H
#define LLVM_ANALYSIS_REGIONINFOIMPL_H

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/RegionIterator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <iterator>
#include <set>

namespace llvm {

#define DEBUG_TYPE "region"

//===----------------------------------------------------------------------===//
/// RegionBase Implementation
template <class Tr>
RegionBase<Tr>::RegionBase(BlockT *Entry, BlockT *Exit,
                           typename Tr::RegionInfoT *RInfo, DomTreeT *dt,
                           RegionT *Parent)
    : RegionNodeBase<Tr>(Parent, Entry, 1), RI(RInfo), DT(dt), exit(Exit) {}

template <class Tr>
RegionBase<Tr>::~RegionBase() {
  // Free the cached nodes.
  for (typename BBNodeMapT::iterator it = BBNodeMap.begin(),
                                     ie = BBNodeMap.end();
       it != ie; ++it)
    delete it->second;

  // Only clean the cache for this Region. Caches of child Regions will be
  // cleaned when the child Regions are deleted.
  BBNodeMap.clear();
}

template <class Tr>
void RegionBase<Tr>::replaceEntry(BlockT *BB) {
  this->entry.setPointer(BB);
}

template <class Tr>
void RegionBase<Tr>::replaceExit(BlockT *BB) {
  assert(exit && "No exit to replace!");
  exit = BB;
}

template <class Tr>
void RegionBase<Tr>::replaceEntryRecursive(BlockT *NewEntry) {
  std::vector<RegionT *> RegionQueue;
  BlockT *OldEntry = getEntry();

  RegionQueue.push_back(static_cast<RegionT *>(this));
  while (!RegionQueue.empty()) {
    RegionT *R = RegionQueue.back();
    RegionQueue.pop_back();

    R->replaceEntry(NewEntry);
    for (typename RegionT::const_iterator RI = R->begin(), RE = R->end();
         RI != RE; ++RI) {
      if ((*RI)->getEntry() == OldEntry)
        RegionQueue.push_back(RI->get());
    }
  }
}

template <class Tr>
void RegionBase<Tr>::replaceExitRecursive(BlockT *NewExit) {
  std::vector<RegionT *> RegionQueue;
  BlockT *OldExit = getExit();

  RegionQueue.push_back(static_cast<RegionT *>(this));
  while (!RegionQueue.empty()) {
    RegionT *R = RegionQueue.back();
    RegionQueue.pop_back();

    R->replaceExit(NewExit);
    for (typename RegionT::const_iterator RI = R->begin(), RE = R->end();
         RI != RE; ++RI) {
      if ((*RI)->getExit() == OldExit)
        RegionQueue.push_back(RI->get());
    }
  }
}

template <class Tr>
bool RegionBase<Tr>::contains(const BlockT *B) const {
  BlockT *BB = const_cast<BlockT *>(B);

  if (!DT->getNode(BB))
    return false;

  BlockT *entry = getEntry(), *exit = getExit();

  // Toplevel region.
  if (!exit)
    return true;

  return (DT->dominates(entry, BB) &&
          !(DT->dominates(exit, BB) && DT->dominates(entry, exit)));
}

template <class Tr>
bool RegionBase<Tr>::contains(const LoopT *L) const {
  // BBs that are not part of any loop are element of the Loop
  // described by the NULL pointer. This loop is not part of any region,
  // except if the region describes the whole function.
  if (!L)
    return getExit() == nullptr;

  if (!contains(L->getHeader()))
    return false;

  SmallVector<BlockT *, 8> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);

  for (BlockT *BB : ExitingBlocks) {
    if (!contains(BB))
      return false;
  }

  return true;
}

template <class Tr>
typename Tr::LoopT *RegionBase<Tr>::outermostLoopInRegion(LoopT *L) const {
  if (!contains(L))
    return nullptr;

  while (L && contains(L->getParentLoop())) {
    L = L->getParentLoop();
  }

  return L;
}

template <class Tr>
typename Tr::LoopT *RegionBase<Tr>::outermostLoopInRegion(LoopInfoT *LI,
                                                          BlockT *BB) const {
  assert(LI && BB && "LI and BB cannot be null!");
  LoopT *L = LI->getLoopFor(BB);
  return outermostLoopInRegion(L);
}

template <class Tr>
typename RegionBase<Tr>::BlockT *RegionBase<Tr>::getEnteringBlock() const {
  BlockT *entry = getEntry();
  BlockT *Pred;
  BlockT *enteringBlock = nullptr;

  for (PredIterTy PI = InvBlockTraits::child_begin(entry),
                  PE = InvBlockTraits::child_end(entry);
       PI != PE; ++PI) {
    Pred = *PI;
    if (DT->getNode(Pred) && !contains(Pred)) {
      if (enteringBlock)
        return nullptr;

      enteringBlock = Pred;
    }
  }

  return enteringBlock;
}

template <class Tr>
typename RegionBase<Tr>::BlockT *RegionBase<Tr>::getExitingBlock() const {
  BlockT *exit = getExit();
  BlockT *Pred;
  BlockT *exitingBlock = nullptr;

  if (!exit)
    return nullptr;

  for (PredIterTy PI = InvBlockTraits::child_begin(exit),
                  PE = InvBlockTraits::child_end(exit);
       PI != PE; ++PI) {
    Pred = *PI;
    if (contains(Pred)) {
      if (exitingBlock)
        return nullptr;

      exitingBlock = Pred;
    }
  }

  return exitingBlock;
}

template <class Tr>
bool RegionBase<Tr>::isSimple() const {
  return !isTopLevelRegion() && getEnteringBlock() && getExitingBlock();
}

template <class Tr>
std::string RegionBase<Tr>::getNameStr() const {
  std::string exitName;
  std::string entryName;

  if (getEntry()->getName().empty()) {
    raw_string_ostream OS(entryName);

    getEntry()->printAsOperand(OS, false);
  } else
    entryName = getEntry()->getName();

  if (getExit()) {
    if (getExit()->getName().empty()) {
      raw_string_ostream OS(exitName);

      getExit()->printAsOperand(OS, false);
    } else
      exitName = getExit()->getName();
  } else
    exitName = "<Function Return>";

  return entryName + " => " + exitName;
}

template <class Tr>
void RegionBase<Tr>::verifyBBInRegion(BlockT *BB) const {
  if (!contains(BB))
    llvm_unreachable("Broken region found!");

  BlockT *entry = getEntry(), *exit = getExit();

  for (SuccIterTy SI = BlockTraits::child_begin(BB),
                  SE = BlockTraits::child_end(BB);
       SI != SE; ++SI) {
    if (!contains(*SI) && exit != *SI)
      llvm_unreachable("Broken region found!");
  }

  if (entry != BB) {
    for (PredIterTy SI = InvBlockTraits::child_begin(BB),
                    SE = InvBlockTraits::child_end(BB);
         SI != SE; ++SI) {
      if (!contains(*SI))
        llvm_unreachable("Broken region found!");
    }
  }
}

template <class Tr>
void RegionBase<Tr>::verifyWalk(BlockT *BB, std::set<BlockT *> *visited) const {
  BlockT *exit = getExit();

  visited->insert(BB);

  verifyBBInRegion(BB);

  for (SuccIterTy SI = BlockTraits::child_begin(BB),
                  SE = BlockTraits::child_end(BB);
       SI != SE; ++SI) {
    if (*SI != exit && visited->find(*SI) == visited->end())
      verifyWalk(*SI, visited);
  }
}

template <class Tr>
void RegionBase<Tr>::verifyRegion() const {
  // Only do verification when user wants to, otherwise this expensive check
  // will be invoked by PMDataManager::verifyPreservedAnalysis when
  // a regionpass (marked PreservedAll) finish.
  if (!RegionInfoBase<Tr>::VerifyRegionInfo)
    return;

  std::set<BlockT *> visited;
  verifyWalk(getEntry(), &visited);
}

template <class Tr>
void RegionBase<Tr>::verifyRegionNest() const {
  for (typename RegionT::const_iterator RI = begin(), RE = end(); RI != RE;
       ++RI)
    (*RI)->verifyRegionNest();

  verifyRegion();
}

template <class Tr>
typename RegionBase<Tr>::element_iterator RegionBase<Tr>::element_begin() {
  return GraphTraits<RegionT *>::nodes_begin(static_cast<RegionT *>(this));
}

template <class Tr>
typename RegionBase<Tr>::element_iterator RegionBase<Tr>::element_end() {
  return GraphTraits<RegionT *>::nodes_end(static_cast<RegionT *>(this));
}

template <class Tr>
typename RegionBase<Tr>::const_element_iterator
RegionBase<Tr>::element_begin() const {
  return GraphTraits<const RegionT *>::nodes_begin(
      static_cast<const RegionT *>(this));
}

template <class Tr>
typename RegionBase<Tr>::const_element_iterator
RegionBase<Tr>::element_end() const {
  return GraphTraits<const RegionT *>::nodes_end(
      static_cast<const RegionT *>(this));
}

template <class Tr>
typename Tr::RegionT *RegionBase<Tr>::getSubRegionNode(BlockT *BB) const {
  typedef typename Tr::RegionT RegionT;
  RegionT *R = RI->getRegionFor(BB);

  if (!R || R == this)
    return nullptr;

  // If we pass the BB out of this region, that means our code is broken.
  assert(contains(R) && "BB not in current region!");

  while (contains(R->getParent()) && R->getParent() != this)
    R = R->getParent();

  if (R->getEntry() != BB)
    return nullptr;

  return R;
}

template <class Tr>
typename Tr::RegionNodeT *RegionBase<Tr>::getBBNode(BlockT *BB) const {
  assert(contains(BB) && "Can get BB node out of this region!");

  typename BBNodeMapT::const_iterator at = BBNodeMap.find(BB);

  if (at != BBNodeMap.end())
    return at->second;

  auto Deconst = const_cast<RegionBase<Tr> *>(this);
  RegionNodeT *NewNode = new RegionNodeT(static_cast<RegionT *>(Deconst), BB);
  BBNodeMap.insert(std::make_pair(BB, NewNode));
  return NewNode;
}

template <class Tr>
typename Tr::RegionNodeT *RegionBase<Tr>::getNode(BlockT *BB) const {
  assert(contains(BB) && "Can get BB node out of this region!");
  if (RegionT *Child = getSubRegionNode(BB))
    return Child->getNode();

  return getBBNode(BB);
}

template <class Tr>
void RegionBase<Tr>::transferChildrenTo(RegionT *To) {
  for (iterator I = begin(), E = end(); I != E; ++I) {
    (*I)->parent = To;
    To->children.push_back(std::move(*I));
  }
  children.clear();
}

template <class Tr>
void RegionBase<Tr>::addSubRegion(RegionT *SubRegion, bool moveChildren) {
  assert(!SubRegion->parent && "SubRegion already has a parent!");
  assert(std::find_if(begin(), end(), [&](const std::unique_ptr<RegionT> &R) {
           return R.get() == SubRegion;
         }) == children.end() &&
         "Subregion already exists!");

  SubRegion->parent = static_cast<RegionT *>(this);
  children.push_back(std::unique_ptr<RegionT>(SubRegion));

  if (!moveChildren)
    return;

  assert(SubRegion->children.empty() &&
         "SubRegions that contain children are not supported");

  for (element_iterator I = element_begin(), E = element_end(); I != E; ++I) {
    if (!(*I)->isSubRegion()) {
      BlockT *BB = (*I)->template getNodeAs<BlockT>();

      if (SubRegion->contains(BB))
        RI->setRegionFor(BB, SubRegion);
    }
  }

  std::vector<std::unique_ptr<RegionT>> Keep;
  for (iterator I = begin(), E = end(); I != E; ++I) {
    if (SubRegion->contains(I->get()) && I->get() != SubRegion) {
      (*I)->parent = SubRegion;
      SubRegion->children.push_back(std::move(*I));
    } else
      Keep.push_back(std::move(*I));
  }

  children.clear();
  children.insert(
      children.begin(),
      std::move_iterator<typename RegionSet::iterator>(Keep.begin()),
      std::move_iterator<typename RegionSet::iterator>(Keep.end()));
}

template <class Tr>
typename Tr::RegionT *RegionBase<Tr>::removeSubRegion(RegionT *Child) {
  assert(Child->parent == this && "Child is not a child of this region!");
  Child->parent = nullptr;
  typename RegionSet::iterator I = std::find_if(
      children.begin(), children.end(),
      [&](const std::unique_ptr<RegionT> &R) { return R.get() == Child; });
  assert(I != children.end() && "Region does not exit. Unable to remove.");
  children.erase(children.begin() + (I - begin()));
  return Child;
}

template <class Tr>
unsigned RegionBase<Tr>::getDepth() const {
  unsigned Depth = 0;

  for (RegionT *R = getParent(); R != nullptr; R = R->getParent())
    ++Depth;

  return Depth;
}

template <class Tr>
typename Tr::RegionT *RegionBase<Tr>::getExpandedRegion() const {
  unsigned NumSuccessors = Tr::getNumSuccessors(exit);

  if (NumSuccessors == 0)
    return nullptr;

  for (PredIterTy PI = InvBlockTraits::child_begin(getExit()),
                  PE = InvBlockTraits::child_end(getExit());
       PI != PE; ++PI) {
    if (!DT->dominates(getEntry(), *PI))
      return nullptr;
  }

  RegionT *R = RI->getRegionFor(exit);

  if (R->getEntry() != exit) {
    if (Tr::getNumSuccessors(exit) == 1)
      return new RegionT(getEntry(), *BlockTraits::child_begin(exit), RI, DT);
    return nullptr;
  }

  while (R->getParent() && R->getParent()->getEntry() == exit)
    R = R->getParent();

  if (!DT->dominates(getEntry(), R->getExit())) {
    for (PredIterTy PI = InvBlockTraits::child_begin(getExit()),
                    PE = InvBlockTraits::child_end(getExit());
         PI != PE; ++PI) {
      if (!DT->dominates(R->getExit(), *PI))
        return nullptr;
    }
  }

  return new RegionT(getEntry(), R->getExit(), RI, DT);
}

template <class Tr>
void RegionBase<Tr>::print(raw_ostream &OS, bool print_tree, unsigned level,
                           PrintStyle Style) const {
  if (print_tree)
    OS.indent(level * 2) << '[' << level << "] " << getNameStr();
  else
    OS.indent(level * 2) << getNameStr();

  OS << '\n';

  if (Style != PrintNone) {
    OS.indent(level * 2) << "{\n";
    OS.indent(level * 2 + 2);

    if (Style == PrintBB) {
      for (const auto *BB : blocks())
        OS << BB->getName() << ", "; // TODO: remove the last ","
    } else if (Style == PrintRN) {
      for (const_element_iterator I = element_begin(), E = element_end();
           I != E; ++I) {
        OS << **I << ", "; // TODO: remove the last ",
      }
    }

    OS << '\n';
  }

  if (print_tree) {
    for (const_iterator RI = begin(), RE = end(); RI != RE; ++RI)
      (*RI)->print(OS, print_tree, level + 1, Style);
  }

  if (Style != PrintNone)
    OS.indent(level * 2) << "} \n";
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
template <class Tr>
void RegionBase<Tr>::dump() const {
  print(dbgs(), true, getDepth(), RegionInfoBase<Tr>::printStyle);
}
#endif

template <class Tr>
void RegionBase<Tr>::clearNodeCache() {
  // Free the cached nodes.
  for (typename BBNodeMapT::iterator I = BBNodeMap.begin(),
                                     IE = BBNodeMap.end();
       I != IE; ++I)
    delete I->second;

  BBNodeMap.clear();
  for (typename RegionT::iterator RI = begin(), RE = end(); RI != RE; ++RI)
    (*RI)->clearNodeCache();
}
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
// RegionInfoBase implementation
//

template <class Tr>
RegionInfoBase<Tr>::RegionInfoBase()
    : TopLevelRegion(nullptr) {}

template <class Tr>
RegionInfoBase<Tr>::~RegionInfoBase() {
  releaseMemory();
}

template <class Tr>
bool RegionInfoBase<Tr>::isCommonDomFrontier(BlockT *BB, BlockT *entry,
                                             BlockT *exit) const {
  for (PredIterTy PI = InvBlockTraits::child_begin(BB),
                  PE = InvBlockTraits::child_end(BB);
       PI != PE; ++PI) {
    BlockT *P = *PI;
    if (DT->dominates(entry, P) && !DT->dominates(exit, P))
      return false;
  }

  return true;
}

template <class Tr>
bool RegionInfoBase<Tr>::isRegion(BlockT *entry, BlockT *exit) const {
  assert(entry && exit && "entry and exit must not be null!");
  typedef typename DomFrontierT::DomSetType DST;

  DST *entrySuccs = &DF->find(entry)->second;

  // Exit is the header of a loop that contains the entry. In this case,
  // the dominance frontier must only contain the exit.
  if (!DT->dominates(entry, exit)) {
    for (typename DST::iterator SI = entrySuccs->begin(),
                                SE = entrySuccs->end();
         SI != SE; ++SI) {
      if (*SI != exit && *SI != entry)
        return false;
    }

    return true;
  }

  DST *exitSuccs = &DF->find(exit)->second;

  // Do not allow edges leaving the region.
  for (typename DST::iterator SI = entrySuccs->begin(), SE = entrySuccs->end();
       SI != SE; ++SI) {
    if (*SI == exit || *SI == entry)
      continue;
    if (exitSuccs->find(*SI) == exitSuccs->end())
      return false;
    if (!isCommonDomFrontier(*SI, entry, exit))
      return false;
  }

  // Do not allow edges pointing into the region.
  for (typename DST::iterator SI = exitSuccs->begin(), SE = exitSuccs->end();
       SI != SE; ++SI) {
    if (DT->properlyDominates(entry, *SI) && *SI != exit)
      return false;
  }

  return true;
}

template <class Tr>
void RegionInfoBase<Tr>::insertShortCut(BlockT *entry, BlockT *exit,
                                        BBtoBBMap *ShortCut) const {
  assert(entry && exit && "entry and exit must not be null!");

  typename BBtoBBMap::iterator e = ShortCut->find(exit);

  if (e == ShortCut->end())
    // No further region at exit available.
    (*ShortCut)[entry] = exit;
  else {
    // We found a region e that starts at exit. Therefore (entry, e->second)
    // is also a region, that is larger than (entry, exit). Insert the
    // larger one.
    BlockT *BB = e->second;
    (*ShortCut)[entry] = BB;
  }
}

template <class Tr>
typename Tr::DomTreeNodeT *
RegionInfoBase<Tr>::getNextPostDom(DomTreeNodeT *N, BBtoBBMap *ShortCut) const {
  typename BBtoBBMap::iterator e = ShortCut->find(N->getBlock());

  if (e == ShortCut->end())
    return N->getIDom();

  return PDT->getNode(e->second)->getIDom();
}

template <class Tr>
bool RegionInfoBase<Tr>::isTrivialRegion(BlockT *entry, BlockT *exit) const {
  assert(entry && exit && "entry and exit must not be null!");

  unsigned num_successors =
      BlockTraits::child_end(entry) - BlockTraits::child_begin(entry);

  if (num_successors <= 1 && exit == *(BlockTraits::child_begin(entry)))
    return true;

  return false;
}

template <class Tr>
typename Tr::RegionT *RegionInfoBase<Tr>::createRegion(BlockT *entry,
                                                       BlockT *exit) {
  assert(entry && exit && "entry and exit must not be null!");

  if (isTrivialRegion(entry, exit))
    return nullptr;

  RegionT *region =
      new RegionT(entry, exit, static_cast<RegionInfoT *>(this), DT);
  BBtoRegion.insert(std::make_pair(entry, region));

#ifdef XDEBUG
  region->verifyRegion();
#else
  DEBUG(region->verifyRegion());
#endif

  updateStatistics(region);
  return region;
}

template <class Tr>
void RegionInfoBase<Tr>::findRegionsWithEntry(BlockT *entry,
                                              BBtoBBMap *ShortCut) {
  assert(entry);

  DomTreeNodeT *N = PDT->getNode(entry);
  if (!N)
    return;

  RegionT *lastRegion = nullptr;
  BlockT *lastExit = entry;

  // As only a BasicBlock that postdominates entry can finish a region, walk the
  // post dominance tree upwards.
  while ((N = getNextPostDom(N, ShortCut))) {
    BlockT *exit = N->getBlock();

    if (!exit)
      break;

    if (isRegion(entry, exit)) {
      RegionT *newRegion = createRegion(entry, exit);

      if (lastRegion)
        newRegion->addSubRegion(lastRegion);

      lastRegion = newRegion;
      lastExit = exit;
    }

    // This can never be a region, so stop the search.
    if (!DT->dominates(entry, exit))
      break;
  }

  // Tried to create regions from entry to lastExit.  Next time take a
  // shortcut from entry to lastExit.
  if (lastExit != entry)
    insertShortCut(entry, lastExit, ShortCut);
}

template <class Tr>
void RegionInfoBase<Tr>::scanForRegions(FuncT &F, BBtoBBMap *ShortCut) {
  typedef typename std::add_pointer<FuncT>::type FuncPtrT;
  BlockT *entry = GraphTraits<FuncPtrT>::getEntryNode(&F);
  DomTreeNodeT *N = DT->getNode(entry);

  // Iterate over the dominance tree in post order to start with the small
  // regions from the bottom of the dominance tree.  If the small regions are
  // detected first, detection of bigger regions is faster, as we can jump
  // over the small regions.
  for (auto DomNode : post_order(N))
    findRegionsWithEntry(DomNode->getBlock(), ShortCut);
}

template <class Tr>
typename Tr::RegionT *RegionInfoBase<Tr>::getTopMostParent(RegionT *region) {
  while (region->getParent())
    region = region->getParent();

  return region;
}

template <class Tr>
void RegionInfoBase<Tr>::buildRegionsTree(DomTreeNodeT *N, RegionT *region) {
  BlockT *BB = N->getBlock();

  // Passed region exit
  while (BB == region->getExit())
    region = region->getParent();

  typename BBtoRegionMap::iterator it = BBtoRegion.find(BB);

  // This basic block is a start block of a region. It is already in the
  // BBtoRegion relation. Only the child basic blocks have to be updated.
  if (it != BBtoRegion.end()) {
    RegionT *newRegion = it->second;
    region->addSubRegion(getTopMostParent(newRegion));
    region = newRegion;
  } else {
    BBtoRegion[BB] = region;
  }

  for (typename DomTreeNodeT::iterator CI = N->begin(), CE = N->end(); CI != CE;
       ++CI) {
    buildRegionsTree(*CI, region);
  }
}

#ifdef XDEBUG
template <class Tr>
bool RegionInfoBase<Tr>::VerifyRegionInfo = true;
#else
template <class Tr>
bool RegionInfoBase<Tr>::VerifyRegionInfo = false;
#endif

template <class Tr>
typename Tr::RegionT::PrintStyle RegionInfoBase<Tr>::printStyle =
    RegionBase<Tr>::PrintNone;

template <class Tr>
void RegionInfoBase<Tr>::print(raw_ostream &OS) const {
  OS << "Region tree:\n";
  TopLevelRegion->print(OS, true, 0, printStyle);
  OS << "End region tree\n";
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
template <class Tr>
void RegionInfoBase<Tr>::dump() const { print(dbgs()); }
#endif

template <class Tr>
void RegionInfoBase<Tr>::releaseMemory() {
  BBtoRegion.clear();
  if (TopLevelRegion)
    delete TopLevelRegion;
  TopLevelRegion = nullptr;
}

template <class Tr>
void RegionInfoBase<Tr>::verifyAnalysis() const {
  TopLevelRegion->verifyRegionNest();
}

// Region pass manager support.
template <class Tr>
typename Tr::RegionT *RegionInfoBase<Tr>::getRegionFor(BlockT *BB) const {
  typename BBtoRegionMap::const_iterator I = BBtoRegion.find(BB);
  return I != BBtoRegion.end() ? I->second : nullptr;
}

template <class Tr>
void RegionInfoBase<Tr>::setRegionFor(BlockT *BB, RegionT *R) {
  BBtoRegion[BB] = R;
}

template <class Tr>
typename Tr::RegionT *RegionInfoBase<Tr>::operator[](BlockT *BB) const {
  return getRegionFor(BB);
}

template <class Tr>
typename RegionInfoBase<Tr>::BlockT *
RegionInfoBase<Tr>::getMaxRegionExit(BlockT *BB) const {
  BlockT *Exit = nullptr;

  while (true) {
    // Get largest region that starts at BB.
    RegionT *R = getRegionFor(BB);
    while (R && R->getParent() && R->getParent()->getEntry() == BB)
      R = R->getParent();

    // Get the single exit of BB.
    if (R && R->getEntry() == BB)
      Exit = R->getExit();
    else if (++BlockTraits::child_begin(BB) == BlockTraits::child_end(BB))
      Exit = *BlockTraits::child_begin(BB);
    else // No single exit exists.
      return Exit;

    // Get largest region that starts at Exit.
    RegionT *ExitR = getRegionFor(Exit);
    while (ExitR && ExitR->getParent() &&
           ExitR->getParent()->getEntry() == Exit)
      ExitR = ExitR->getParent();

    for (PredIterTy PI = InvBlockTraits::child_begin(Exit),
                    PE = InvBlockTraits::child_end(Exit);
         PI != PE; ++PI) {
      if (!R->contains(*PI) && !ExitR->contains(*PI))
        break;
    }

    // This stops infinite cycles.
    if (DT->dominates(Exit, BB))
      break;

    BB = Exit;
  }

  return Exit;
}

template <class Tr>
typename Tr::RegionT *RegionInfoBase<Tr>::getCommonRegion(RegionT *A,
                                                          RegionT *B) const {
  assert(A && B && "One of the Regions is NULL");

  if (A->contains(B))
    return A;

  while (!B->contains(A))
    B = B->getParent();

  return B;
}

template <class Tr>
typename Tr::RegionT *
RegionInfoBase<Tr>::getCommonRegion(SmallVectorImpl<RegionT *> &Regions) const {
  RegionT *ret = Regions.back();
  Regions.pop_back();

  for (RegionT *R : Regions)
    ret = getCommonRegion(ret, R);

  return ret;
}

template <class Tr>
typename Tr::RegionT *
RegionInfoBase<Tr>::getCommonRegion(SmallVectorImpl<BlockT *> &BBs) const {
  RegionT *ret = getRegionFor(BBs.back());
  BBs.pop_back();

  for (BlockT *BB : BBs)
    ret = getCommonRegion(ret, getRegionFor(BB));

  return ret;
}

template <class Tr>
void RegionInfoBase<Tr>::splitBlock(BlockT *NewBB, BlockT *OldBB) {
  RegionT *R = getRegionFor(OldBB);

  setRegionFor(NewBB, R);

  while (R->getEntry() == OldBB && !R->isTopLevelRegion()) {
    R->replaceEntry(NewBB);
    R = R->getParent();
  }

  setRegionFor(OldBB, R);
}

template <class Tr>
void RegionInfoBase<Tr>::calculate(FuncT &F) {
  typedef typename std::add_pointer<FuncT>::type FuncPtrT;

  // ShortCut a function where for every BB the exit of the largest region
  // starting with BB is stored. These regions can be threated as single BBS.
  // This improves performance on linear CFGs.
  BBtoBBMap ShortCut;

  scanForRegions(F, &ShortCut);
  BlockT *BB = GraphTraits<FuncPtrT>::getEntryNode(&F);
  buildRegionsTree(DT->getNode(BB), TopLevelRegion);
}

#undef DEBUG_TYPE

} // end namespace llvm

#endif
