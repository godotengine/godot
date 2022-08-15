//===- llvm/ADT/DepthFirstIterator.h - Depth First iterator -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file builds on the ADT/GraphTraits.h file to build generic depth
// first graph iterator.  This file exposes the following functions/types:
//
// df_begin/df_end/df_iterator
//   * Normal depth-first iteration - visit a node and then all of its children.
//
// idf_begin/idf_end/idf_iterator
//   * Depth-first iteration on the 'inverse' graph.
//
// df_ext_begin/df_ext_end/df_ext_iterator
//   * Normal depth-first iteration - visit a node and then all of its children.
//     This iterator stores the 'visited' set in an external set, which allows
//     it to be more efficient, and allows external clients to use the set for
//     other purposes.
//
// idf_ext_begin/idf_ext_end/idf_ext_iterator
//   * Depth-first iteration on the 'inverse' graph.
//     This iterator stores the 'visited' set in an external set, which allows
//     it to be more efficient, and allows external clients to use the set for
//     other purposes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_DEPTHFIRSTITERATOR_H
#define LLVM_ADT_DEPTHFIRSTITERATOR_H

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/iterator_range.h"
#include <set>
#include <vector>

namespace llvm {

// df_iterator_storage - A private class which is used to figure out where to
// store the visited set.
template<class SetType, bool External>   // Non-external set
class df_iterator_storage {
public:
  SetType Visited;
};

template<class SetType>
class df_iterator_storage<SetType, true> {
public:
  df_iterator_storage(SetType &VSet) : Visited(VSet) {}
  df_iterator_storage(const df_iterator_storage &S) : Visited(S.Visited) {}
  SetType &Visited;
};


// Generic Depth First Iterator
template<class GraphT,
class SetType = llvm::SmallPtrSet<typename GraphTraits<GraphT>::NodeType*, 8>,
         bool ExtStorage = false, class GT = GraphTraits<GraphT> >
class df_iterator : public std::iterator<std::forward_iterator_tag,
                                         typename GT::NodeType, ptrdiff_t>,
                    public df_iterator_storage<SetType, ExtStorage> {
  typedef std::iterator<std::forward_iterator_tag,
                        typename GT::NodeType, ptrdiff_t> super;

  typedef typename GT::NodeType          NodeType;
  typedef typename GT::ChildIteratorType ChildItTy;
  typedef PointerIntPair<NodeType*, 1>   PointerIntTy;

  // VisitStack - Used to maintain the ordering.  Top = current block
  // First element is node pointer, second is the 'next child' to visit
  // if the int in PointerIntTy is 0, the 'next child' to visit is invalid
  std::vector<std::pair<PointerIntTy, ChildItTy> > VisitStack;
private:
  inline df_iterator(NodeType *Node) {
    this->Visited.insert(Node);
    VisitStack.push_back(std::make_pair(PointerIntTy(Node, 0), 
                                        GT::child_begin(Node)));
  }
  inline df_iterator() { 
    // End is when stack is empty 
  }
  inline df_iterator(NodeType *Node, SetType &S)
    : df_iterator_storage<SetType, ExtStorage>(S) {
    if (!S.count(Node)) {
      VisitStack.push_back(std::make_pair(PointerIntTy(Node, 0), 
                                          GT::child_begin(Node)));
      this->Visited.insert(Node);
    }
  }
  inline df_iterator(SetType &S)
    : df_iterator_storage<SetType, ExtStorage>(S) {
    // End is when stack is empty
  }

  inline void toNext() {
    do {
      std::pair<PointerIntTy, ChildItTy> &Top = VisitStack.back();
      NodeType *Node = Top.first.getPointer();
      ChildItTy &It  = Top.second;
      if (!Top.first.getInt()) {
        // now retrieve the real begin of the children before we dive in
        It = GT::child_begin(Node);
        Top.first.setInt(1);
      }

      while (It != GT::child_end(Node)) {
        NodeType *Next = *It++;
        // Has our next sibling been visited?
        if (Next && this->Visited.insert(Next).second) {
          // No, do it now.
          VisitStack.push_back(std::make_pair(PointerIntTy(Next, 0), 
                                              GT::child_begin(Next)));
          return;
        }
      }

      // Oops, ran out of successors... go up a level on the stack.
      VisitStack.pop_back();
    } while (!VisitStack.empty());
  }

public:
  typedef typename super::pointer pointer;

  // Provide static begin and end methods as our public "constructors"
  static df_iterator begin(const GraphT &G) {
    return df_iterator(GT::getEntryNode(G));
  }
  static df_iterator end(const GraphT &G) { return df_iterator(); }

  // Static begin and end methods as our public ctors for external iterators
  static df_iterator begin(const GraphT &G, SetType &S) {
    return df_iterator(GT::getEntryNode(G), S);
  }
  static df_iterator end(const GraphT &G, SetType &S) { return df_iterator(S); }

  bool operator==(const df_iterator &x) const {
    return VisitStack == x.VisitStack;
  }
  bool operator!=(const df_iterator &x) const { return !(*this == x); }

  pointer operator*() const { return VisitStack.back().first.getPointer(); }

  // This is a nonstandard operator-> that dereferences the pointer an extra
  // time... so that you can actually call methods ON the Node, because
  // the contained type is a pointer.  This allows BBIt->getTerminator() f.e.
  //
  NodeType *operator->() const { return **this; }

  df_iterator &operator++() { // Preincrement
    toNext();
    return *this;
  }

  /// \brief Skips all children of the current node and traverses to next node
  ///
  /// Note: This function takes care of incrementing the iterator. If you
  /// always increment and call this function, you risk walking off the end.
  df_iterator &skipChildren() {
    VisitStack.pop_back();
    if (!VisitStack.empty())
      toNext();
    return *this;
  }

  df_iterator operator++(int) { // Postincrement
    df_iterator tmp = *this;
    ++*this;
    return tmp;
  }

  // nodeVisited - return true if this iterator has already visited the
  // specified node.  This is public, and will probably be used to iterate over
  // nodes that a depth first iteration did not find: ie unreachable nodes.
  //
  bool nodeVisited(NodeType *Node) const {
    return this->Visited.count(Node) != 0;
  }

  /// getPathLength - Return the length of the path from the entry node to the
  /// current node, counting both nodes.
  unsigned getPathLength() const { return VisitStack.size(); }

  /// getPath - Return the n'th node in the path from the entry node to the
  /// current node.
  NodeType *getPath(unsigned n) const {
    return VisitStack[n].first.getPointer();
  }
};


// Provide global constructors that automatically figure out correct types...
//
template <class T>
df_iterator<T> df_begin(const T& G) {
  return df_iterator<T>::begin(G);
}

template <class T>
df_iterator<T> df_end(const T& G) {
  return df_iterator<T>::end(G);
}

// Provide an accessor method to use them in range-based patterns.
template <class T>
iterator_range<df_iterator<T>> depth_first(const T& G) {
  return make_range(df_begin(G), df_end(G));
}

// Provide global definitions of external depth first iterators...
template <class T, class SetTy = std::set<typename GraphTraits<T>::NodeType*> >
struct df_ext_iterator : public df_iterator<T, SetTy, true> {
  df_ext_iterator(const df_iterator<T, SetTy, true> &V)
    : df_iterator<T, SetTy, true>(V) {}
};

template <class T, class SetTy>
df_ext_iterator<T, SetTy> df_ext_begin(const T& G, SetTy &S) {
  return df_ext_iterator<T, SetTy>::begin(G, S);
}

template <class T, class SetTy>
df_ext_iterator<T, SetTy> df_ext_end(const T& G, SetTy &S) {
  return df_ext_iterator<T, SetTy>::end(G, S);
}

template <class T, class SetTy>
iterator_range<df_ext_iterator<T, SetTy>> depth_first_ext(const T& G,
                                                          SetTy &S) {
  return make_range(df_ext_begin(G, S), df_ext_end(G, S));
}


// Provide global definitions of inverse depth first iterators...
template <class T,
  class SetTy = llvm::SmallPtrSet<typename GraphTraits<T>::NodeType*, 8>,
          bool External = false>
struct idf_iterator : public df_iterator<Inverse<T>, SetTy, External> {
  idf_iterator(const df_iterator<Inverse<T>, SetTy, External> &V)
    : df_iterator<Inverse<T>, SetTy, External>(V) {}
};

template <class T>
idf_iterator<T> idf_begin(const T& G) {
  return idf_iterator<T>::begin(Inverse<T>(G));
}

template <class T>
idf_iterator<T> idf_end(const T& G){
  return idf_iterator<T>::end(Inverse<T>(G));
}

// Provide an accessor method to use them in range-based patterns.
template <class T>
iterator_range<idf_iterator<T>> inverse_depth_first(const T& G) {
  return make_range(idf_begin(G), idf_end(G));
}

// Provide global definitions of external inverse depth first iterators...
template <class T, class SetTy = std::set<typename GraphTraits<T>::NodeType*> >
struct idf_ext_iterator : public idf_iterator<T, SetTy, true> {
  idf_ext_iterator(const idf_iterator<T, SetTy, true> &V)
    : idf_iterator<T, SetTy, true>(V) {}
  idf_ext_iterator(const df_iterator<Inverse<T>, SetTy, true> &V)
    : idf_iterator<T, SetTy, true>(V) {}
};

template <class T, class SetTy>
idf_ext_iterator<T, SetTy> idf_ext_begin(const T& G, SetTy &S) {
  return idf_ext_iterator<T, SetTy>::begin(Inverse<T>(G), S);
}

template <class T, class SetTy>
idf_ext_iterator<T, SetTy> idf_ext_end(const T& G, SetTy &S) {
  return idf_ext_iterator<T, SetTy>::end(Inverse<T>(G), S);
}

template <class T, class SetTy>
iterator_range<idf_ext_iterator<T, SetTy>> inverse_depth_first_ext(const T& G,
                                                                   SetTy &S) {
  return make_range(idf_ext_begin(G, S), idf_ext_end(G, S));
}

} // End llvm namespace

#endif
