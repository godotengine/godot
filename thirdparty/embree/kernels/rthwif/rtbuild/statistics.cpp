// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "statistics.h"

namespace embree
{
  class RestoreStreamState 
  {
  public:
    RestoreStreamState(std::ostream& iostream)
      : iostream(iostream), flags(iostream.flags()), precision(iostream.precision()) {
    }

    ~RestoreStreamState() {
      iostream.flags(flags);
      iostream.precision(precision);
    }
    
  private:
    std::ostream& iostream;
    std::ios::fmtflags flags;
    std::streamsize precision;
  };
  
  double ratio(double a, double b)
  {
    if (b == 0.0) return 0.0f;
    else return a/b;
  }

  double percent(double a, double b) {
    return 100.0*ratio(a,b);
  }

  double ratio(size_t a, size_t b) {
    return ratio(double(a), double(b));
  }
  double percent(size_t a, size_t b) {
    return percent(double(a), double(b));
  }
  
  void BVHStatistics::NodeStat::print(std::ostream& cout, double totalSAH, size_t totalBytes, size_t numPrimitives) const
  {
    RestoreStreamState iostate(cout);
    cout << std::setw(7) << numNodes << " ";
    cout << std::setw(7) << std::setprecision(3) << sah();
    cout << std::setw(7) << std::setprecision(2) << percent(sah(),totalSAH) << "% ";
    cout << std::setw(8) << std::setprecision(2) << bytes()/1E6  << " MB ";
    cout << std::setw(7) << std::setprecision(2) << percent(numBytes,numBytes) << "% ";
    cout << std::setw(7) << std::setprecision(2) << percent(bytes(),totalBytes) << "% ";
    cout << std::setw(8) << std::setprecision(2) << ratio(bytes(),numNodes) << " ";
    cout << std::setw(8) << std::setprecision(2) << ratio(bytes(),numChildrenUsed) << " ";
    cout << std::setw(8) << std::setprecision(2) << ratio(bytes(),numPrimitives) << " ";
    cout << std::setw(7) << std::setprecision(2) << ratio(numChildrenUsed,numNodes) << " ";
    cout << std::setw(7) << std::setprecision(2) << 100.0*fillRate() << "% ";
    cout << std::endl;
  }
  
  void BVHStatistics::LeafStat::print(std::ostream& cout, double totalSAH, size_t totalBytes, size_t numPrimitives, bool blocks) const
  {
    RestoreStreamState iostate(cout);
    size_t N = blocks ? numBlocks : numLeaves;
    cout << std::setw(7) << N << " ";
    cout << std::setw(7) << std::setprecision(3) << sah();
    cout << std::setw(7) << std::setprecision(2) << percent(sah(),totalSAH) << "% ";
    cout << std::setw(8) << std::setprecision(2) << double(bytes())/1E6  << " MB ";
    cout << std::setw(7) << std::setprecision(2) << percent(numBytesUsed,numBytesTotal) << "% ";
    cout << std::setw(7) << std::setprecision(2) << percent(bytes(),totalBytes) << "% ";
    cout << std::setw(8) << std::setprecision(2) << ratio(bytes(),N) << " ";
    cout << std::setw(8) << std::setprecision(2) << ratio(bytes(),numPrimsUsed) << " ";
    cout << std::setw(8) << std::setprecision(2) << ratio(bytes(),numPrimitives) << " ";
    cout << std::setw(7) << std::setprecision(2) << ratio(numPrimsUsed,N) << " ";
    cout << std::setw(7) << std::setprecision(2) << 100.0*fillRate() << "% ";
    cout << std::endl;
  }
  
  void BVHStatistics::print (std::ostream& cout) const
  {
    RestoreStreamState iostate(cout);
    cout.setf(std::ios::fixed, std::ios::floatfield);
    cout.fill(' ');
    
    double totalSAH   = internalNode.nodeSAH + quadLeaf.leafSAH + proceduralLeaf.leafSAH + instanceLeaf.leafSAH;
    size_t totalBytes = internalNode.bytes() + quadLeaf.bytes() + proceduralLeaf.bytes() + instanceLeaf.bytes();
    size_t totalNodes = internalNode.numNodes + quadLeaf.numLeaves + proceduralLeaf.numLeaves + instanceLeaf.numLeaves;
    size_t totalPrimitives = quadLeaf.numPrimsUsed + proceduralLeaf.numPrimsUsed + instanceLeaf.numPrimsUsed;

    cout << std::endl;
    cout << "BVH statistics:" << std::endl;
    cout << "---------------" << std::endl;
    cout << "  numScenePrimitives          = " << numScenePrimitives << std::endl;
    cout << "  numBuildPrimitives          = " << numBuildPrimitives << std::endl;
    cout << "  numBuildPrimitivesPostSplit = " << numBuildPrimitivesPostSplit << std::endl;
    cout << "  primRefSplits               = " << std::setprecision(2) << percent(numBuildPrimitivesPostSplit,numBuildPrimitives) << "%" << std::endl;
    cout << "  numBVHPrimitives            = " << totalPrimitives << std::endl;
    cout << "  spatialSplits               = " << std::setprecision(2) << percent(totalPrimitives,numScenePrimitives) << "%" << std::endl;    
    cout << std::endl;
     
    cout << "                      #nodes     SAH   total       bytes     used    total   b/node  b/child   b/prim  #child     fill" << std::endl;
    cout << "----------------------------------------------------------------------------------------------------------------------" << std::endl;
       cout << "  total            : ";
    cout << std::setw(7) << totalNodes << " ";
    cout << std::setw(7) << std::setprecision(3) << totalSAH;
    cout << " 100.00% ";
    cout << std::setw(8) << std::setprecision(2) << totalBytes/1E6 << " MB ";
    cout << " 100.00% ";
    cout << " 100.00% ";
    cout << "         ";
    cout << "         ";
    cout << std::setw(8) << std::setprecision(2) << ratio(totalBytes,totalPrimitives) << std::endl;

    LeafStat leaf = quadLeaf + proceduralLeaf + instanceLeaf;
    cout << "  internalNode     : "; internalNode  .print(cout,totalSAH,totalBytes,totalPrimitives);
    cout << "  leaves           : "; leaf          .print(cout,totalSAH,totalBytes,totalPrimitives);
    cout << "    quadLeaf       : "; quadLeaf      .print(cout,totalSAH,totalBytes,totalPrimitives);
    cout << "    proceduralLeaf : "; proceduralLeaf.print(cout,totalSAH,totalBytes,totalPrimitives);
    cout << "    proceduralBlock: "; proceduralLeaf.print(cout,totalSAH,totalBytes,totalPrimitives,true);
    cout << "    instanceLeaf   : "; instanceLeaf  .print(cout,totalSAH,totalBytes,totalPrimitives);
  }
  
  void BVHStatistics::print_raw(std::ostream& cout) const
  {
    RestoreStreamState iostate(cout);
    size_t totalPrimitives = quadLeaf.numPrimsUsed + proceduralLeaf.numPrimsUsed + instanceLeaf.numPrimsUsed;
    cout << "bvh_spatial_split_factor = " << percent(totalPrimitives,numBuildPrimitives) << std::endl;
    
    cout << "bvh_internal_sah = " << internalNode.nodeSAH << std::endl;
    cout << "bvh_internal_num = " << internalNode.numNodes << std::endl;
    cout << "bvh_internal_num_children_used = " << internalNode.numChildrenUsed << std::endl;
    cout << "bvh_internal_num_children_total = " << internalNode.numChildrenTotal << std::endl;
    cout << "bvh_internal_num_bytes = " << internalNode.bytes() << std::endl;
    
    cout << "bvh_quad_leaf_sah = " << quadLeaf.leafSAH << std::endl;
    cout << "bvh_quad_leaf_num = " << quadLeaf.numLeaves << std::endl;
    cout << "bvh_quad_leaf_num_prims_used = " << quadLeaf.numPrimsUsed << std::endl;
    cout << "bvh_quad_leaf_num_prims_total = " << quadLeaf.numPrimsTotal << std::endl;
    cout << "bvh_quad_leaf_num_bytes_used = " << quadLeaf.numBytesUsed << std::endl;
    cout << "bvh_quad_leaf_num_bytes_total = " << quadLeaf.numBytesTotal << std::endl;

    cout << "bvh_procedural_leaf_sah = " << proceduralLeaf.leafSAH << std::endl;
    cout << "bvh_procedural_leaf_num = " << proceduralLeaf.numLeaves << std::endl;
    cout << "bvh_procedural_leaf_num_prims_used = " << proceduralLeaf.numPrimsUsed << std::endl;
    cout << "bvh_procedural_leaf_num_prims_total = " << proceduralLeaf.numPrimsTotal << std::endl;
    cout << "bvh_procedural_leaf_num_bytes_used = " << proceduralLeaf.numBytesUsed << std::endl;
    cout << "bvh_procedural_leaf_num_bytes_total = " << proceduralLeaf.numBytesTotal << std::endl;

    cout << "bvh_instance_leaf_sah = " << instanceLeaf.leafSAH << std::endl;
    cout << "bvh_instance_leaf_num = " << instanceLeaf.numLeaves << std::endl;
    cout << "bvh_instance_leaf_num_prims_used = " << instanceLeaf.numPrimsUsed << std::endl;
    cout << "bvh_instance_leaf_num_prims_total = " << instanceLeaf.numPrimsTotal << std::endl;
    cout << "bvh_instance_leaf_num_bytes_used = " << instanceLeaf.numBytesUsed << std::endl;
    cout << "bvh_instance_leaf_num_bytes_total = " << instanceLeaf.numBytesTotal << std::endl;
  }
}
