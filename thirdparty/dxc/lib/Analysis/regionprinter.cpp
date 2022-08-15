//===- RegionPrinter.cpp - Print regions tree pass ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Print out the region tree of a function using dotty/graphviz.
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Passes.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/DOTGraphTraitsPass.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/RegionIterator.h"
#include "llvm/Analysis/RegionPrinter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
/// onlySimpleRegion - Show only the simple regions in the RegionViewer.
static cl::opt<bool>
onlySimpleRegions("only-simple-regions",
                  cl::desc("Show only simple regions in the graphviz viewer"),
                  cl::Hidden,
                  cl::init(false));

namespace llvm {
template<>
struct DOTGraphTraits<RegionNode*> : public DefaultDOTGraphTraits {

  DOTGraphTraits (bool isSimple=false)
    : DefaultDOTGraphTraits(isSimple) {}

  std::string getNodeLabel(RegionNode *Node, RegionNode *Graph) {

    if (!Node->isSubRegion()) {
      BasicBlock *BB = Node->getNodeAs<BasicBlock>();

      if (isSimple())
        return DOTGraphTraits<const Function*>
          ::getSimpleNodeLabel(BB, BB->getParent());
      else
        return DOTGraphTraits<const Function*>
          ::getCompleteNodeLabel(BB, BB->getParent());
    }

    return "Not implemented";
  }
};

template<>
struct DOTGraphTraits<RegionInfoPass*> : public DOTGraphTraits<RegionNode*> {

  DOTGraphTraits (bool isSimple = false)
    : DOTGraphTraits<RegionNode*>(isSimple) {}

  static std::string getGraphName(RegionInfoPass *DT) {
    return "Region Graph";
  }

  std::string getNodeLabel(RegionNode *Node, RegionInfoPass *G) {
    RegionInfo &RI = G->getRegionInfo();
    return DOTGraphTraits<RegionNode*>::getNodeLabel(Node,
                                                     reinterpret_cast<RegionNode*>(RI.getTopLevelRegion()));
  }

  std::string getEdgeAttributes(RegionNode *srcNode,
    GraphTraits<RegionInfo*>::ChildIteratorType CI, RegionInfoPass *G) {
    RegionInfo &RI = G->getRegionInfo();
    RegionNode *destNode = *CI;

    if (srcNode->isSubRegion() || destNode->isSubRegion())
      return "";

    // In case of a backedge, do not use it to define the layout of the nodes.
    BasicBlock *srcBB = srcNode->getNodeAs<BasicBlock>();
    BasicBlock *destBB = destNode->getNodeAs<BasicBlock>();

    Region *R = RI.getRegionFor(destBB);

    while (R && R->getParent())
      if (R->getParent()->getEntry() == destBB)
        R = R->getParent();
      else
        break;

    if (R && R->getEntry() == destBB && R->contains(srcBB)) // HLSL Change - add null check
      return "constraint=false";

    return "";
  }

  // Print the cluster of the subregions. This groups the single basic blocks
  // and adds a different background color for each group.
  static void printRegionCluster(const Region &R,
                                 GraphWriter<RegionInfoPass*> &GW,
                                 unsigned depth = 0) {
    raw_ostream &O = GW.getOStream();
    O.indent(2 * depth) << "subgraph cluster_" << static_cast<const void*>(&R)
      << " {\n";
    O.indent(2 * (depth + 1)) << "label = \"\";\n";

    if (!onlySimpleRegions || R.isSimple()) {
      O.indent(2 * (depth + 1)) << "style = filled;\n";
      O.indent(2 * (depth + 1)) << "color = "
        << ((R.getDepth() * 2 % 12) + 1) << "\n";

    } else {
      O.indent(2 * (depth + 1)) << "style = solid;\n";
      O.indent(2 * (depth + 1)) << "color = "
        << ((R.getDepth() * 2 % 12) + 2) << "\n";
    }

    for (Region::const_iterator RI = R.begin(), RE = R.end(); RI != RE; ++RI)
      printRegionCluster(**RI, GW, depth + 1);

    const RegionInfo &RI = *static_cast<const RegionInfo*>(R.getRegionInfo());

    for (auto *BB : R.blocks())
      if (RI.getRegionFor(BB) == &R)
        O.indent(2 * (depth + 1)) << "Node"
          << static_cast<const void*>(RI.getTopLevelRegion()->getBBNode(BB))
          << ";\n";

    O.indent(2 * depth) << "}\n";
  }

  static void addCustomGraphFeatures(const RegionInfoPass* RIP,
                                     GraphWriter<RegionInfoPass*> &GW) {
    const RegionInfo &RI = RIP->getRegionInfo();
    raw_ostream &O = GW.getOStream();
    O << "\tcolorscheme = \"paired12\"\n";
    printRegionCluster(*RI.getTopLevelRegion(), GW, 4);
  }
};
} //end namespace llvm

namespace {

struct RegionViewer
  : public DOTGraphTraitsViewer<RegionInfoPass, false> {
  static char ID;
  RegionViewer() : DOTGraphTraitsViewer<RegionInfoPass, false>("reg", ID){
    initializeRegionViewerPass(*PassRegistry::getPassRegistry());
  }
};
char RegionViewer::ID = 0;

struct RegionOnlyViewer
  : public DOTGraphTraitsViewer<RegionInfoPass, true> {
  static char ID;
  RegionOnlyViewer() : DOTGraphTraitsViewer<RegionInfoPass, true>("regonly", ID) {
    initializeRegionOnlyViewerPass(*PassRegistry::getPassRegistry());
  }
};
char RegionOnlyViewer::ID = 0;

struct RegionPrinter
  : public DOTGraphTraitsPrinter<RegionInfoPass, false> {
  static char ID;
  RegionPrinter() :
    DOTGraphTraitsPrinter<RegionInfoPass, false>("reg", ID) {
      initializeRegionPrinterPass(*PassRegistry::getPassRegistry());
    }
};
char RegionPrinter::ID = 0;
} //end anonymous namespace

INITIALIZE_PASS(RegionPrinter, "dot-regions",
                "Print regions of function to 'dot' file", true, true)

INITIALIZE_PASS(RegionViewer, "view-regions", "View regions of function",
                true, true)

INITIALIZE_PASS(RegionOnlyViewer, "view-regions-only",
                "View regions of function (with no function bodies)",
                true, true)

namespace {

struct RegionOnlyPrinter
  : public DOTGraphTraitsPrinter<RegionInfoPass, true> {
  static char ID;
  RegionOnlyPrinter() :
    DOTGraphTraitsPrinter<RegionInfoPass, true>("reg", ID) {
      initializeRegionOnlyPrinterPass(*PassRegistry::getPassRegistry());
    }
};

}

char RegionOnlyPrinter::ID = 0;
INITIALIZE_PASS(RegionOnlyPrinter, "dot-regions-only",
                "Print regions of function to 'dot' file "
                "(with no function bodies)",
                true, true)

FunctionPass* llvm::createRegionViewerPass() {
  return new RegionViewer();
}

FunctionPass* llvm::createRegionOnlyViewerPass() {
  return new RegionOnlyViewer();
}

FunctionPass* llvm::createRegionPrinterPass() {
  return new RegionPrinter();
}

FunctionPass* llvm::createRegionOnlyPrinterPass() {
  return new RegionOnlyPrinter();
}

