//==-- llvm/ADT/ilist.h - Intrusive Linked List Template ---------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines classes to implement an intrusive doubly linked list class
// (i.e. each node of the list must contain a next and previous field for the
// list.
//
// The ilist_traits trait class is used to gain access to the next and previous
// fields of the node type that the list is instantiated with.  If it is not
// specialized, the list defaults to using the getPrev(), getNext() method calls
// to get the next and previous pointers.
//
// The ilist class itself, should be a plug in replacement for list, assuming
// that the nodes contain next/prev pointers.  This list replacement does not
// provide a constant time size() method, so be careful to use empty() when you
// really want to know if it's empty.
//
// The ilist class is implemented by allocating a 'tail' node when the list is
// created (using ilist_traits<>::createSentinel()).  This tail node is
// absolutely required because the user must be able to compute end()-1. Because
// of this, users of the direct next/prev links will see an extra link on the
// end of the list, which should be ignored.
//
// Requirements for a user of this list:
//
//   1. The user must provide {g|s}et{Next|Prev} methods, or specialize
//      ilist_traits to provide an alternate way of getting and setting next and
//      prev links.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_ILIST_H
#define LLVM_ADT_ILIST_H

#include "llvm/Support/Compiler.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>

namespace llvm {

template<typename NodeTy, typename Traits> class iplist;
template<typename NodeTy> class ilist_iterator;

/// ilist_nextprev_traits - A fragment for template traits for intrusive list
/// that provides default next/prev implementations for common operations.
///
template<typename NodeTy>
struct ilist_nextprev_traits {
  static NodeTy *getPrev(NodeTy *N) { return N->getPrev(); }
  static NodeTy *getNext(NodeTy *N) { return N->getNext(); }
  static const NodeTy *getPrev(const NodeTy *N) { return N->getPrev(); }
  static const NodeTy *getNext(const NodeTy *N) { return N->getNext(); }

  static void setPrev(NodeTy *N, NodeTy *Prev) { N->setPrev(Prev); }
  static void setNext(NodeTy *N, NodeTy *Next) { N->setNext(Next); }
};

template<typename NodeTy>
struct ilist_traits;

/// ilist_sentinel_traits - A fragment for template traits for intrusive list
/// that provides default sentinel implementations for common operations.
///
/// ilist_sentinel_traits implements a lazy dynamic sentinel allocation
/// strategy. The sentinel is stored in the prev field of ilist's Head.
///
template<typename NodeTy>
struct ilist_sentinel_traits {
  /// createSentinel - create the dynamic sentinel
  static NodeTy *createSentinel() { return new NodeTy(); }

  /// destroySentinel - deallocate the dynamic sentinel
  static void destroySentinel(NodeTy *N) { delete N; }

  /// provideInitialHead - when constructing an ilist, provide a starting
  /// value for its Head
  /// @return null node to indicate that it needs to be allocated later
  static NodeTy *provideInitialHead() { return nullptr; }

  /// ensureHead - make sure that Head is either already
  /// initialized or assigned a fresh sentinel
  /// @return the sentinel
  static NodeTy *ensureHead(NodeTy *&Head) {
    if (!Head) {
      Head = ilist_traits<NodeTy>::createSentinel();
      ilist_traits<NodeTy>::noteHead(Head, Head);
      ilist_traits<NodeTy>::setNext(Head, nullptr);
      return Head;
    }
    return ilist_traits<NodeTy>::getPrev(Head);
  }

  /// noteHead - stash the sentinel into its default location
  static void noteHead(NodeTy *NewHead, NodeTy *Sentinel) {
    ilist_traits<NodeTy>::setPrev(NewHead, Sentinel);
  }
};

/// ilist_node_traits - A fragment for template traits for intrusive list
/// that provides default node related operations.
///
template<typename NodeTy>
struct ilist_node_traits {
  static NodeTy *createNode(const NodeTy &V) { return new NodeTy(V); }
  static void deleteNode(NodeTy *V) { delete V; }

  void addNodeToList(NodeTy *) {}
  void removeNodeFromList(NodeTy *) {}
  void transferNodesFromList(ilist_node_traits &    /*SrcTraits*/,
                             ilist_iterator<NodeTy> /*first*/,
                             ilist_iterator<NodeTy> /*last*/) {}
};

/// ilist_default_traits - Default template traits for intrusive list.
/// By inheriting from this, you can easily use default implementations
/// for all common operations.
///
template<typename NodeTy>
struct ilist_default_traits : public ilist_nextprev_traits<NodeTy>,
                              public ilist_sentinel_traits<NodeTy>,
                              public ilist_node_traits<NodeTy> {
};

// Template traits for intrusive list.  By specializing this template class, you
// can change what next/prev fields are used to store the links...
template<typename NodeTy>
struct ilist_traits : public ilist_default_traits<NodeTy> {};

// Const traits are the same as nonconst traits...
template<typename Ty>
struct ilist_traits<const Ty> : public ilist_traits<Ty> {};

//===----------------------------------------------------------------------===//
// ilist_iterator<Node> - Iterator for intrusive list.
//
template<typename NodeTy>
class ilist_iterator
  : public std::iterator<std::bidirectional_iterator_tag, NodeTy, ptrdiff_t> {

public:
  typedef ilist_traits<NodeTy> Traits;
  typedef std::iterator<std::bidirectional_iterator_tag,
                        NodeTy, ptrdiff_t> super;

  typedef typename super::value_type value_type;
  typedef typename super::difference_type difference_type;
  typedef typename super::pointer pointer;
  typedef typename super::reference reference;
private:
  pointer NodePtr;

  // ilist_iterator is not a random-access iterator, but it has an
  // implicit conversion to pointer-type, which is. Declare (but
  // don't define) these functions as private to help catch
  // accidental misuse.
  void operator[](difference_type) const;
  void operator+(difference_type) const;
  void operator-(difference_type) const;
  void operator+=(difference_type) const;
  void operator-=(difference_type) const;
  template<class T> void operator<(T) const;
  template<class T> void operator<=(T) const;
  template<class T> void operator>(T) const;
  template<class T> void operator>=(T) const;
  template<class T> void operator-(T) const;
public:

  ilist_iterator(pointer NP) : NodePtr(NP) {}
  ilist_iterator(reference NR) : NodePtr(&NR) {}
  ilist_iterator() : NodePtr(nullptr) {}

  // This is templated so that we can allow constructing a const iterator from
  // a nonconst iterator...
  template<class node_ty>
  ilist_iterator(const ilist_iterator<node_ty> &RHS)
    : NodePtr(RHS.getNodePtrUnchecked()) {}

  // This is templated so that we can allow assigning to a const iterator from
  // a nonconst iterator...
  template<class node_ty>
  const ilist_iterator &operator=(const ilist_iterator<node_ty> &RHS) {
    NodePtr = RHS.getNodePtrUnchecked();
    return *this;
  }

  // Accessors...
  operator pointer() const {
    return NodePtr;
  }

  reference operator*() const {
    return *NodePtr;
  }
  pointer operator->() const { return &operator*(); }

  // Comparison operators
  bool operator==(const ilist_iterator &RHS) const {
    return NodePtr == RHS.NodePtr;
  }
  bool operator!=(const ilist_iterator &RHS) const {
    return NodePtr != RHS.NodePtr;
  }

  // Increment and decrement operators...
  ilist_iterator &operator--() {      // predecrement - Back up
    NodePtr = Traits::getPrev(NodePtr);
    assert(NodePtr && "--'d off the beginning of an ilist!");
    return *this;
  }
  ilist_iterator &operator++() {      // preincrement - Advance
    NodePtr = Traits::getNext(NodePtr);
    return *this;
  }
  ilist_iterator operator--(int) {    // postdecrement operators...
    ilist_iterator tmp = *this;
    --*this;
    return tmp;
  }
  ilist_iterator operator++(int) {    // postincrement operators...
    ilist_iterator tmp = *this;
    ++*this;
    return tmp;
  }

  // Internal interface, do not use...
  pointer getNodePtrUnchecked() const { return NodePtr; }
};

// These are to catch errors when people try to use them as random access
// iterators.
template<typename T>
void operator-(int, ilist_iterator<T>) = delete;
template<typename T>
void operator-(ilist_iterator<T>,int) = delete;

template<typename T>
void operator+(int, ilist_iterator<T>) = delete;
template<typename T>
void operator+(ilist_iterator<T>,int) = delete;

// operator!=/operator== - Allow mixed comparisons without dereferencing
// the iterator, which could very likely be pointing to end().
template<typename T>
bool operator!=(const T* LHS, const ilist_iterator<const T> &RHS) {
  return LHS != RHS.getNodePtrUnchecked();
}
template<typename T>
bool operator==(const T* LHS, const ilist_iterator<const T> &RHS) {
  return LHS == RHS.getNodePtrUnchecked();
}
template<typename T>
bool operator!=(T* LHS, const ilist_iterator<T> &RHS) {
  return LHS != RHS.getNodePtrUnchecked();
}
template<typename T>
bool operator==(T* LHS, const ilist_iterator<T> &RHS) {
  return LHS == RHS.getNodePtrUnchecked();
}


// Allow ilist_iterators to convert into pointers to a node automatically when
// used by the dyn_cast, cast, isa mechanisms...

template<typename From> struct simplify_type;

template<typename NodeTy> struct simplify_type<ilist_iterator<NodeTy> > {
  typedef NodeTy* SimpleType;

  static SimpleType getSimplifiedValue(ilist_iterator<NodeTy> &Node) {
    return &*Node;
  }
};
template<typename NodeTy> struct simplify_type<const ilist_iterator<NodeTy> > {
  typedef /*const*/ NodeTy* SimpleType;

  static SimpleType getSimplifiedValue(const ilist_iterator<NodeTy> &Node) {
    return &*Node;
  }
};
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
//
/// iplist - The subset of list functionality that can safely be used on nodes
/// of polymorphic types, i.e. a heterogeneous list with a common base class that
/// holds the next/prev pointers.  The only state of the list itself is a single
/// pointer to the head of the list.
///
/// This list can be in one of three interesting states:
/// 1. The list may be completely unconstructed.  In this case, the head
///    pointer is null.  When in this form, any query for an iterator (e.g.
///    begin() or end()) causes the list to transparently change to state #2.
/// 2. The list may be empty, but contain a sentinel for the end iterator. This
///    sentinel is created by the Traits::createSentinel method and is a link
///    in the list.  When the list is empty, the pointer in the iplist points
///    to the sentinel.  Once the sentinel is constructed, it
///    is not destroyed until the list is.
/// 3. The list may contain actual objects in it, which are stored as a doubly
///    linked list of nodes.  One invariant of the list is that the predecessor
///    of the first node in the list always points to the last node in the list,
///    and the successor pointer for the sentinel (which always stays at the
///    end of the list) is always null.
///
template<typename NodeTy, typename Traits=ilist_traits<NodeTy> >
class iplist : public Traits {
  mutable NodeTy *Head;

  // Use the prev node pointer of 'head' as the tail pointer.  This is really a
  // circularly linked list where we snip the 'next' link from the sentinel node
  // back to the first node in the list (to preserve assertions about going off
  // the end of the list).
  NodeTy *getTail() { return this->ensureHead(Head); }
  const NodeTy *getTail() const { return this->ensureHead(Head); }
  void setTail(NodeTy *N) const { this->noteHead(Head, N); }

  /// CreateLazySentinel - This method verifies whether the sentinel for the
  /// list has been created and lazily makes it if not.
  void CreateLazySentinel() const {
    this->ensureHead(Head);
  }

  static bool op_less(NodeTy &L, NodeTy &R) { return L < R; }
  static bool op_equal(NodeTy &L, NodeTy &R) { return L == R; }

  // No fundamental reason why iplist can't be copyable, but the default
  // copy/copy-assign won't do.
  iplist(const iplist &) = delete;
  void operator=(const iplist &) = delete;

public:
  typedef NodeTy *pointer;
  typedef const NodeTy *const_pointer;
  typedef NodeTy &reference;
  typedef const NodeTy &const_reference;
  typedef NodeTy value_type;
  typedef ilist_iterator<NodeTy> iterator;
  typedef ilist_iterator<const NodeTy> const_iterator;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef std::reverse_iterator<const_iterator>  const_reverse_iterator;
  typedef std::reverse_iterator<iterator>  reverse_iterator;

  iplist() : Head(this->provideInitialHead()) {}
  ~iplist() {
    if (!Head) return;
    clear();
    Traits::destroySentinel(getTail());
  }

  // Iterator creation methods.
  iterator begin() {
    CreateLazySentinel();
    return iterator(Head);
  }
  const_iterator begin() const {
    CreateLazySentinel();
    return const_iterator(Head);
  }
  iterator end() {
    CreateLazySentinel();
    return iterator(getTail());
  }
  const_iterator end() const {
    CreateLazySentinel();
    return const_iterator(getTail());
  }

  // reverse iterator creation methods.
  reverse_iterator rbegin()            { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const{ return const_reverse_iterator(end()); }
  reverse_iterator rend()              { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const { return const_reverse_iterator(begin());}


  // Miscellaneous inspection routines.
  size_type max_size() const { return size_type(-1); }
  bool LLVM_ATTRIBUTE_UNUSED_RESULT empty() const {
    return !Head || Head == getTail();
  }

  // Front and back accessor functions...
  reference front() {
    assert(!empty() && "Called front() on empty list!");
    return *Head;
  }
  const_reference front() const {
    assert(!empty() && "Called front() on empty list!");
    return *Head;
  }
  reference back() {
    assert(!empty() && "Called back() on empty list!");
    return *this->getPrev(getTail());
  }
  const_reference back() const {
    assert(!empty() && "Called back() on empty list!");
    return *this->getPrev(getTail());
  }

  void swap(iplist &RHS) {
    assert(0 && "Swap does not use list traits callback correctly yet!");
    std::swap(Head, RHS.Head);
  }

  iterator insert(iterator where, NodeTy *New) {
    NodeTy *CurNode = where.getNodePtrUnchecked();
    NodeTy *PrevNode = this->getPrev(CurNode);
    this->setNext(New, CurNode);
    this->setPrev(New, PrevNode);

    if (CurNode != Head)  // Is PrevNode off the beginning of the list?
      this->setNext(PrevNode, New);
    else
      Head = New;
    this->setPrev(CurNode, New);

    // HLSL Change Begin: Undo insertion if exception
    try {
      this->addNodeToList(New);  // Notify traits that we added a node...
    } catch (...) {
      // Undo insertion
      if (New == Head)
        Head = CurNode;
      else
        this->setNext(PrevNode, CurNode);
      this->setPrev(CurNode, PrevNode);
      this->setPrev(New, nullptr);
      this->setNext(New, nullptr);
      throw;
    }
    // HLSL Change End
    return New;
  }

  iterator insertAfter(iterator where, NodeTy *New) {
    if (empty())
      return insert(begin(), New);
    else
      return insert(++where, New);
  }

  NodeTy *remove(iterator &IT) {
    assert(IT != end() && "Cannot remove end of list!");
    NodeTy *Node = &*IT;
    NodeTy *NextNode = this->getNext(Node);
    NodeTy *PrevNode = this->getPrev(Node);

    if (Node != Head)  // Is PrevNode off the beginning of the list?
      this->setNext(PrevNode, NextNode);
    else
      Head = NextNode;
    this->setPrev(NextNode, PrevNode);
    IT = NextNode;
    this->removeNodeFromList(Node);  // Notify traits that we removed a node...

    // Set the next/prev pointers of the current node to null.  This isn't
    // strictly required, but this catches errors where a node is removed from
    // an ilist (and potentially deleted) with iterators still pointing at it.
    // When those iterators are incremented or decremented, they will assert on
    // the null next/prev pointer instead of "usually working".
    this->setNext(Node, nullptr);
    this->setPrev(Node, nullptr);
    return Node;
  }

  NodeTy *remove(const iterator &IT) {
    iterator MutIt = IT;
    return remove(MutIt);
  }

  // erase - remove a node from the controlled sequence... and delete it.
  iterator erase(iterator where) {
    this->deleteNode(remove(where));
    return where;
  }

  /// Remove all nodes from the list like clear(), but do not call
  /// removeNodeFromList() or deleteNode().
  ///
  /// This should only be used immediately before freeing nodes in bulk to
  /// avoid traversing the list and bringing all the nodes into cache.
  void clearAndLeakNodesUnsafely() {
    if (Head) {
      Head = getTail();
      this->setPrev(Head, Head);
    }
  }

private:
  // transfer - The heart of the splice function.  Move linked list nodes from
  // [first, last) into position.
  //
  void transfer(iterator position, iplist &L2, iterator first, iterator last) {
    assert(first != last && "Should be checked by callers");
    // Position cannot be contained in the range to be transferred.
    // Check for the most common mistake.
    assert(position != first &&
           "Insertion point can't be one of the transferred nodes");

    if (position != last) {
      // Note: we have to be careful about the case when we move the first node
      // in the list.  This node is the list sentinel node and we can't move it.
      NodeTy *ThisSentinel = getTail();
      setTail(nullptr);
      NodeTy *L2Sentinel = L2.getTail();
      L2.setTail(nullptr);

      // Remove [first, last) from its old position.
      NodeTy *First = &*first, *Prev = this->getPrev(First);
      NodeTy *Next = last.getNodePtrUnchecked(), *Last = this->getPrev(Next);
      if (Prev)
        this->setNext(Prev, Next);
      else
        L2.Head = Next;
      this->setPrev(Next, Prev);

      // Splice [first, last) into its new position.
      NodeTy *PosNext = position.getNodePtrUnchecked();
      NodeTy *PosPrev = this->getPrev(PosNext);

      // Fix head of list...
      if (PosPrev)
        this->setNext(PosPrev, First);
      else
        Head = First;
      this->setPrev(First, PosPrev);

      // Fix end of list...
      this->setNext(Last, PosNext);
      this->setPrev(PosNext, Last);

      this->transferNodesFromList(L2, First, PosNext);

      // Now that everything is set, restore the pointers to the list sentinels.
      L2.setTail(L2Sentinel);
      setTail(ThisSentinel);
    }
  }

public:

  //===----------------------------------------------------------------------===
  // Functionality derived from other functions defined above...
  //

  size_type LLVM_ATTRIBUTE_UNUSED_RESULT size() const {
    if (!Head) return 0; // Don't require construction of sentinel if empty.
    return std::distance(begin(), end());
  }

  iterator erase(iterator first, iterator last) {
    while (first != last)
      first = erase(first);
    return last;
  }

  void clear() { if (Head) erase(begin(), end()); }

  // Front and back inserters...
  void push_front(NodeTy *val) { insert(begin(), val); }
  void push_back(NodeTy *val) { insert(end(), val); }
  void pop_front() {
    assert(!empty() && "pop_front() on empty list!");
    erase(begin());
  }
  void pop_back() {
    assert(!empty() && "pop_back() on empty list!");
    iterator t = end(); erase(--t);
  }

  // Special forms of insert...
  template<class InIt> void insert(iterator where, InIt first, InIt last) {
    for (; first != last; ++first) insert(where, *first);
  }

  // Splice members - defined in terms of transfer...
  void splice(iterator where, iplist &L2) {
    if (!L2.empty())
      transfer(where, L2, L2.begin(), L2.end());
  }
  void splice(iterator where, iplist &L2, iterator first) {
    iterator last = first; ++last;
    if (where == first || where == last) return; // No change
    transfer(where, L2, first, last);
  }
  void splice(iterator where, iplist &L2, iterator first, iterator last) {
    if (first != last) transfer(where, L2, first, last);
  }
};


template<typename NodeTy>
struct ilist : public iplist<NodeTy> {
  typedef typename iplist<NodeTy>::size_type size_type;
  typedef typename iplist<NodeTy>::iterator iterator;

  ilist() {}
  ilist(const ilist &right) {
    insert(this->begin(), right.begin(), right.end());
  }
  explicit ilist(size_type count) {
    insert(this->begin(), count, NodeTy());
  }
  ilist(size_type count, const NodeTy &val) {
    insert(this->begin(), count, val);
  }
  template<class InIt> ilist(InIt first, InIt last) {
    insert(this->begin(), first, last);
  }

  // bring hidden functions into scope
  using iplist<NodeTy>::insert;
  using iplist<NodeTy>::push_front;
  using iplist<NodeTy>::push_back;

  // Main implementation here - Insert for a node passed by value...
  iterator insert(iterator where, const NodeTy &val) {
    return insert(where, this->createNode(val));
  }


  // Front and back inserters...
  void push_front(const NodeTy &val) { insert(this->begin(), val); }
  void push_back(const NodeTy &val) { insert(this->end(), val); }

  void insert(iterator where, size_type count, const NodeTy &val) {
    for (; count != 0; --count) insert(where, val);
  }

  // Assign special forms...
  void assign(size_type count, const NodeTy &val) {
    iterator I = this->begin();
    for (; I != this->end() && count != 0; ++I, --count)
      *I = val;
    if (count != 0)
      insert(this->end(), val, val);
    else
      erase(I, this->end());
  }
  template<class InIt> void assign(InIt first1, InIt last1) {
    iterator first2 = this->begin(), last2 = this->end();
    for ( ; first1 != last1 && first2 != last2; ++first1, ++first2)
      *first1 = *first2;
    if (first2 == last2)
      erase(first1, last1);
    else
      insert(last1, first2, last2);
  }


  // Resize members...
  void resize(size_type newsize, NodeTy val) {
    iterator i = this->begin();
    size_type len = 0;
    for ( ; i != this->end() && len < newsize; ++i, ++len) /* empty*/ ;

    if (len == newsize)
      erase(i, this->end());
    else                                          // i == end()
      insert(this->end(), newsize - len, val);
  }
  void resize(size_type newsize) { resize(newsize, NodeTy()); }
};

} // End llvm namespace

namespace std {
  // Ensure that swap uses the fast list swap...
  template<class Ty>
  void swap(llvm::iplist<Ty> &Left, llvm::iplist<Ty> &Right) {
    Left.swap(Right);
  }
}  // End 'std' extensions...

#endif // LLVM_ADT_ILIST_H
