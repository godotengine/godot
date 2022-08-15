//===-- llvm/ADT/GraphTraits.h - Graph traits template ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the little GraphTraits<X> template class that should be
// specialized by classes that want to be iteratable by generic graph iterators.
//
// This file also defines the marker class Inverse that is used to iterate over
// graphs in a graph defined, inverse ordering...
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_GRAPHTRAITS_H
#define LLVM_ADT_GRAPHTRAITS_H

namespace llvm {

// GraphTraits - This class should be specialized by different graph types...
// which is why the default version is empty.
//
template<class GraphType>
struct GraphTraits {
  // Elements to provide:

  // typedef NodeType          - Type of Node in the graph
  // typedef ChildIteratorType - Type used to iterate over children in graph

  // static NodeType *getEntryNode(const GraphType &)
  //    Return the entry node of the graph

  // static ChildIteratorType child_begin(NodeType *)
  // static ChildIteratorType child_end  (NodeType *)
  //    Return iterators that point to the beginning and ending of the child
  //    node list for the specified node.
  //


  // typedef  ...iterator nodes_iterator;
  // static nodes_iterator nodes_begin(GraphType *G)
  // static nodes_iterator nodes_end  (GraphType *G)
  //    nodes_iterator/begin/end - Allow iteration over all nodes in the graph

  // static unsigned       size       (GraphType *G)
  //    Return total number of nodes in the graph
  //


  // If anyone tries to use this class without having an appropriate
  // specialization, make an error.  If you get this error, it's because you
  // need to include the appropriate specialization of GraphTraits<> for your
  // graph, or you need to define it for a new graph type. Either that or
  // your argument to XXX_begin(...) is unknown or needs to have the proper .h
  // file #include'd.
  //
  typedef typename GraphType::UnknownGraphTypeError NodeType;
};


// Inverse - This class is used as a little marker class to tell the graph
// iterator to iterate over the graph in a graph defined "Inverse" ordering.
// Not all graphs define an inverse ordering, and if they do, it depends on
// the graph exactly what that is.  Here's an example of usage with the
// df_iterator:
//
// idf_iterator<Method*> I = idf_begin(M), E = idf_end(M);
// for (; I != E; ++I) { ... }
//
// Which is equivalent to:
// df_iterator<Inverse<Method*> > I = idf_begin(M), E = idf_end(M);
// for (; I != E; ++I) { ... }
//
template <class GraphType>
struct Inverse {
  const GraphType &Graph;

  inline Inverse(const GraphType &G) : Graph(G) {}
};

// Provide a partial specialization of GraphTraits so that the inverse of an
// inverse falls back to the original graph.
template<class T>
struct GraphTraits<Inverse<Inverse<T> > > {
  typedef typename GraphTraits<T>::NodeType NodeType;
  typedef typename GraphTraits<T>::ChildIteratorType ChildIteratorType;

  static NodeType *getEntryNode(Inverse<Inverse<T> > *G) {
    return GraphTraits<T>::getEntryNode(G->Graph.Graph);
  }

  static ChildIteratorType child_begin(NodeType* N) {
    return GraphTraits<T>::child_begin(N);
  }

  static ChildIteratorType child_end(NodeType* N) {
    return GraphTraits<T>::child_end(N);
  }
};

} // End llvm namespace

#endif
