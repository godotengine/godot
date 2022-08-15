//===-- llvm/Value.h - Definition of the Value class ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Value class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_VALUE_H
#define LLVM_IR_VALUE_H

#include "llvm-c/Core.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/Use.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class APInt;
class Argument;
class AssemblyAnnotationWriter;
class BasicBlock;
class Constant;
class DataLayout;
class Function;
class GlobalAlias;
class GlobalObject;
class GlobalValue;
class GlobalVariable;
class InlineAsm;
class Instruction;
class LLVMContext;
class Module;
class ModuleSlotTracker;
class StringRef;
class Twine;
class Type;
class ValueHandleBase;
class ValueSymbolTable;
class raw_ostream;

template<typename ValueTy> class StringMapEntry;
typedef StringMapEntry<Value*> ValueName;

//===----------------------------------------------------------------------===//
//                                 Value Class
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

/// \brief LLVM Value Representation
///
/// This is a very important LLVM class. It is the base class of all values
/// computed by a program that may be used as operands to other values. Value is
/// the super class of other important classes such as Instruction and Function.
/// All Values have a Type. Type is not a subclass of Value. Some values can
/// have a name and they belong to some Module.  Setting the name on the Value
/// automatically updates the module's symbol table.
///
/// Every value has a "use list" that keeps track of which other Values are
/// using this Value.  A Value can also have an arbitrary number of ValueHandle
/// objects that watch it and listen to RAUW and Destroy events.  See
/// llvm/IR/ValueHandle.h for details.
class Value {
  Type *VTy;
  Use *UseList;

  friend class ValueAsMetadata; // Allow access to IsUsedByMD.
  friend class ValueHandleBase;

  const unsigned char SubclassID;   // Subclass identifier (for isa/dyn_cast)
  unsigned char HasValueHandle : 1; // Has a ValueHandle pointing to this?
protected:
  /// \brief Hold subclass data that can be dropped.
  ///
  /// This member is similar to SubclassData, however it is for holding
  /// information which may be used to aid optimization, but which may be
  /// cleared to zero without affecting conservative interpretation.
  unsigned char SubclassOptionalData : 7;

private:
  /// \brief Hold arbitrary subclass data.
  ///
  /// This member is defined by this class, but is not used for anything.
  /// Subclasses can use it to hold whatever state they find useful.  This
  /// field is initialized to zero by the ctor.
  unsigned short SubclassData;

protected:
  /// \brief The number of operands in the subclass.
  ///
  /// This member is defined by this class, but not used for anything.
  /// Subclasses can use it to store their number of operands, if they have
  /// any.
  ///
  /// This is stored here to save space in User on 64-bit hosts.  Since most
  /// instances of Value have operands, 32-bit hosts aren't significantly
  /// affected.
  ///
  /// Note, this should *NOT* be used directly by any class other than User.
  /// User uses this value to find the Use list.
  enum : unsigned { NumUserOperandsBits = 29 };
  unsigned NumUserOperands : NumUserOperandsBits;

  bool IsUsedByMD : 1;
  bool HasName : 1;
  bool HasHungOffUses : 1;

private:
  template <typename UseT> // UseT == 'Use' or 'const Use'
  class use_iterator_impl
      : public std::iterator<std::forward_iterator_tag, UseT *> {
    UseT *U;
    explicit use_iterator_impl(UseT *u) : U(u) {}
    friend class Value;

  public:
    use_iterator_impl() : U() {}

    bool operator==(const use_iterator_impl &x) const { return U == x.U; }
    bool operator!=(const use_iterator_impl &x) const { return !operator==(x); }

    use_iterator_impl &operator++() { // Preincrement
      assert(U && "Cannot increment end iterator!");
      U = U->getNext();
      return *this;
    }
    use_iterator_impl operator++(int) { // Postincrement
      auto tmp = *this;
      ++*this;
      return tmp;
    }

    UseT &operator*() const {
      assert(U && "Cannot dereference end iterator!");
      return *U;
    }

    UseT *operator->() const { return &operator*(); }

    operator use_iterator_impl<const UseT>() const {
      return use_iterator_impl<const UseT>(U);
    }
  };

  template <typename UserTy> // UserTy == 'User' or 'const User'
  class user_iterator_impl
      : public std::iterator<std::forward_iterator_tag, UserTy *> {
    use_iterator_impl<Use> UI;
    explicit user_iterator_impl(Use *U) : UI(U) {}
    friend class Value;

  public:
    user_iterator_impl() {}

    bool operator==(const user_iterator_impl &x) const { return UI == x.UI; }
    bool operator!=(const user_iterator_impl &x) const { return !operator==(x); }

    /// \brief Returns true if this iterator is equal to user_end() on the value.
    bool atEnd() const { return *this == user_iterator_impl(); }

    user_iterator_impl &operator++() { // Preincrement
      ++UI;
      return *this;
    }
    user_iterator_impl operator++(int) { // Postincrement
      auto tmp = *this;
      ++*this;
      return tmp;
    }

    // Retrieve a pointer to the current User.
    UserTy *operator*() const {
      return UI->getUser();
    }

    UserTy *operator->() const { return operator*(); }

    operator user_iterator_impl<const UserTy>() const {
      return user_iterator_impl<const UserTy>(*UI);
    }

    Use &getUse() const { return *UI; }
  };

  void operator=(const Value &) = delete;
  Value(const Value &) = delete;

protected:
  Value(Type *Ty, unsigned scid);
public:
  virtual ~Value();

  /// \brief Support for debugging, callable in GDB: V->dump()
  void dump() const;

  /// \brief Implement operator<< on Value.
  /// @{
  void print(raw_ostream &O) const;
  void print(raw_ostream &O, ModuleSlotTracker &MST) const;
  /// @}

  /// \brief Print the name of this Value out to the specified raw_ostream.
  ///
  /// This is useful when you just want to print 'int %reg126', not the
  /// instruction that generated it. If you specify a Module for context, then
  /// even constanst get pretty-printed; for example, the type of a null
  /// pointer is printed symbolically.
  /// @{
  void printAsOperand(raw_ostream &O, bool PrintType = true,
                      const Module *M = nullptr) const;
  void printAsOperand(raw_ostream &O, bool PrintType,
                      ModuleSlotTracker &MST) const;
  /// @}

  /// \brief All values are typed, get the type of this value.
  Type *getType() const { return VTy; }

  /// \brief All values hold a context through their type.
  LLVMContext &getContext() const;

  // \brief All values can potentially be named.
  bool hasName() const { return HasName; }
  ValueName *getValueName() const;
  void setValueName(ValueName *VN);

private:
  void destroyValueName();
  void setNameImpl(const Twine &Name);

public:
  /// \brief Return a constant reference to the value's name.
  ///
  /// This is cheap and guaranteed to return the same reference as long as the
  /// value is not modified.
  StringRef getName() const;

  /// \brief Change the name of the value.
  ///
  /// Choose a new unique name if the provided name is taken.
  ///
  /// \param Name The new name; or "" if the value's name should be removed.
  void setName(const Twine &Name);


  /// \brief Transfer the name from V to this value.
  ///
  /// After taking V's name, sets V's name to empty.
  ///
  /// \note It is an error to call V->takeName(V).
  void takeName(Value *V);

  /// \brief Change all uses of this to point to a new Value.
  ///
  /// Go through the uses list for this definition and make each use point to
  /// "V" instead of "this".  After this completes, 'this's use list is
  /// guaranteed to be empty.
  void replaceAllUsesWith(Value *V);

  /// replaceUsesOutsideBlock - Go through the uses list for this definition and
  /// make each use point to "V" instead of "this" when the use is outside the
  /// block. 'This's use list is expected to have at least one element.
  /// Unlike replaceAllUsesWith this function does not support basic block
  /// values or constant users.
  void replaceUsesOutsideBlock(Value *V, BasicBlock *BB);

  //----------------------------------------------------------------------
  // Methods for handling the chain of uses of this Value.
  //
  bool               use_empty() const { return UseList == nullptr; }

  typedef use_iterator_impl<Use>       use_iterator;
  typedef use_iterator_impl<const Use> const_use_iterator;
  use_iterator       use_begin()       { return use_iterator(UseList); }
  const_use_iterator use_begin() const { return const_use_iterator(UseList); }
  use_iterator       use_end()         { return use_iterator();   }
  const_use_iterator use_end()   const { return const_use_iterator();   }
  iterator_range<use_iterator> uses() {
    return iterator_range<use_iterator>(use_begin(), use_end());
  }
  iterator_range<const_use_iterator> uses() const {
    return iterator_range<const_use_iterator>(use_begin(), use_end());
  }

  bool               user_empty() const { return UseList == nullptr; }

  typedef user_iterator_impl<User>       user_iterator;
  typedef user_iterator_impl<const User> const_user_iterator;
  user_iterator       user_begin()       { return user_iterator(UseList); }
  const_user_iterator user_begin() const { return const_user_iterator(UseList); }
  user_iterator       user_end()         { return user_iterator();   }
  const_user_iterator user_end()   const { return const_user_iterator();   }
  User               *user_back()        { return *user_begin(); }
  const User         *user_back()  const { return *user_begin(); }
  iterator_range<user_iterator> users() {
    return iterator_range<user_iterator>(user_begin(), user_end());
  }
  iterator_range<const_user_iterator> users() const {
    return iterator_range<const_user_iterator>(user_begin(), user_end());
  }

  /// \brief Return true if there is exactly one user of this value.
  ///
  /// This is specialized because it is a common request and does not require
  /// traversing the whole use list.
  bool hasOneUse() const {
    const_use_iterator I = use_begin(), E = use_end();
    if (I == E) return false;
    return ++I == E;
  }

  /// \brief Return true if this Value has exactly N users.
  bool hasNUses(unsigned N) const;

  /// \brief Return true if this value has N users or more.
  ///
  /// This is logically equivalent to getNumUses() >= N.
  bool hasNUsesOrMore(unsigned N) const;

  /// \brief Check if this value is used in the specified basic block.
  bool isUsedInBasicBlock(const BasicBlock *BB) const;

  /// \brief This method computes the number of uses of this Value.
  ///
  /// This is a linear time operation.  Use hasOneUse, hasNUses, or
  /// hasNUsesOrMore to check for specific values.
  unsigned getNumUses() const;

  /// \brief This method should only be used by the Use class.
  void addUse(Use &U) { U.addToList(&UseList); }

  /// \brief Concrete subclass of this.
  ///
  /// An enumeration for keeping track of the concrete subclass of Value that
  /// is actually instantiated. Values of this enumeration are kept in the
  /// Value classes SubclassID field. They are used for concrete type
  /// identification.
  enum ValueTy {
#define HANDLE_VALUE(Name) Name##Val,
#include "llvm/IR/Value.def"

    // Markers:
#define HANDLE_CONSTANT_MARKER(Marker, Constant) Marker = Constant##Val,
#include "llvm/IR/Value.def"
  };

  /// \brief Return an ID for the concrete type of this object.
  ///
  /// This is used to implement the classof checks.  This should not be used
  /// for any other purpose, as the values may change as LLVM evolves.  Also,
  /// note that for instructions, the Instruction's opcode is added to
  /// InstructionVal. So this means three things:
  /// # there is no value with code InstructionVal (no opcode==0).
  /// # there are more possible values for the value type than in ValueTy enum.
  /// # the InstructionVal enumerator must be the highest valued enumerator in
  ///   the ValueTy enum.
  unsigned getValueID() const {
    return SubclassID;
  }

  /// \brief Return the raw optional flags value contained in this value.
  ///
  /// This should only be used when testing two Values for equivalence.
  unsigned getRawSubclassOptionalData() const {
    return SubclassOptionalData;
  }

  /// \brief Clear the optional flags contained in this value.
  void clearSubclassOptionalData() {
    SubclassOptionalData = 0;
  }

  /// \brief Check the optional flags for equality.
  bool hasSameSubclassOptionalData(const Value *V) const {
    return SubclassOptionalData == V->SubclassOptionalData;
  }

  /// \brief Clear any optional flags not set in the given Value.
  void intersectOptionalDataWith(const Value *V) {
    SubclassOptionalData &= V->SubclassOptionalData;
  }

  /// \brief Return true if there is a value handle associated with this value.
  bool hasValueHandle() const { return HasValueHandle; }

  /// \brief Return true if there is metadata referencing this value.
  bool isUsedByMetadata() const { return IsUsedByMD; }

  /// \brief Strip off pointer casts, all-zero GEPs, and aliases.
  ///
  /// Returns the original uncasted value.  If this is called on a non-pointer
  /// value, it returns 'this'.
  Value *stripPointerCasts();
  const Value *stripPointerCasts() const {
    return const_cast<Value*>(this)->stripPointerCasts();
  }

  /// \brief Strip off pointer casts and all-zero GEPs.
  ///
  /// Returns the original uncasted value.  If this is called on a non-pointer
  /// value, it returns 'this'.
  Value *stripPointerCastsNoFollowAliases();
  const Value *stripPointerCastsNoFollowAliases() const {
    return const_cast<Value*>(this)->stripPointerCastsNoFollowAliases();
  }

  /// \brief Strip off pointer casts and all-constant inbounds GEPs.
  ///
  /// Returns the original pointer value.  If this is called on a non-pointer
  /// value, it returns 'this'.
  Value *stripInBoundsConstantOffsets();
  const Value *stripInBoundsConstantOffsets() const {
    return const_cast<Value*>(this)->stripInBoundsConstantOffsets();
  }

  /// \brief Accumulate offsets from \a stripInBoundsConstantOffsets().
  ///
  /// Stores the resulting constant offset stripped into the APInt provided.
  /// The provided APInt will be extended or truncated as needed to be the
  /// correct bitwidth for an offset of this pointer type.
  ///
  /// If this is called on a non-pointer value, it returns 'this'.
  Value *stripAndAccumulateInBoundsConstantOffsets(const DataLayout &DL,
                                                   APInt &Offset);
  const Value *stripAndAccumulateInBoundsConstantOffsets(const DataLayout &DL,
                                                         APInt &Offset) const {
    return const_cast<Value *>(this)
        ->stripAndAccumulateInBoundsConstantOffsets(DL, Offset);
  }

  /// \brief Strip off pointer casts and inbounds GEPs.
  ///
  /// Returns the original pointer value.  If this is called on a non-pointer
  /// value, it returns 'this'.
  Value *stripInBoundsOffsets();
  const Value *stripInBoundsOffsets() const {
    return const_cast<Value*>(this)->stripInBoundsOffsets();
  }

  /// \brief Translate PHI node to its predecessor from the given basic block.
  ///
  /// If this value is a PHI node with CurBB as its parent, return the value in
  /// the PHI node corresponding to PredBB.  If not, return ourself.  This is
  /// useful if you want to know the value something has in a predecessor
  /// block.
  Value *DoPHITranslation(const BasicBlock *CurBB, const BasicBlock *PredBB);

  const Value *DoPHITranslation(const BasicBlock *CurBB,
                                const BasicBlock *PredBB) const{
    return const_cast<Value*>(this)->DoPHITranslation(CurBB, PredBB);
  }

  /// \brief The maximum alignment for instructions.
  ///
  /// This is the greatest alignment value supported by load, store, and alloca
  /// instructions, and global values.
  static const unsigned MaxAlignmentExponent = 29;
  static const unsigned MaximumAlignment = 1u << MaxAlignmentExponent;

  /// \brief Mutate the type of this Value to be of the specified type.
  ///
  /// Note that this is an extremely dangerous operation which can create
  /// completely invalid IR very easily.  It is strongly recommended that you
  /// recreate IR objects with the right types instead of mutating them in
  /// place.
  void mutateType(Type *Ty) {
    VTy = Ty;
  }

  /// \brief Sort the use-list.
  ///
  /// Sorts the Value's use-list by Cmp using a stable mergesort.  Cmp is
  /// expected to compare two \a Use references.
  template <class Compare> void sortUseList(Compare Cmp);

  /// \brief Reverse the use-list.
  void reverseUseList();

private:
  /// \brief Merge two lists together.
  ///
  /// Merges \c L and \c R using \c Cmp.  To enable stable sorts, always pushes
  /// "equal" items from L before items from R.
  ///
  /// \return the first element in the list.
  ///
  /// \note Completely ignores \a Use::Prev (doesn't read, doesn't update).
  template <class Compare>
  static Use *mergeUseLists(Use *L, Use *R, Compare Cmp) {
    Use *Merged;

    // HLSL Change Begins. Copied from LLVM Version 8.0.0.
    // MergeUseListsImpl(L, R, &Merged, Cmp);
    Use **Next = &Merged;

    while (true) {
      if (!L) {
        *Next = R;
        break;
      }
      if (!R) {
        *Next = L;
        break;
      }
      if (Cmp(*R, *L)) {
        *Next = R;
        Next = &R->Next;
        R = R->Next;
      } else {
        *Next = L;
        Next = &L->Next;
        L = L->Next;
      }
    }
    // HLSL Change Ends.

    return Merged;
  }
  
  /// \brief Tail-recursive helper for \a mergeUseLists().
  ///
  /// \param[out] Next the first element in the list.
  // HLSL Change Begins.
  //template <class Compare>
  //static void mergeUseListsImpl(Use *L, Use *R, Use **Next, Compare Cmp);
  // HLSL Change Ends.

protected:
  unsigned short getSubclassDataFromValue() const { return SubclassData; }
  void setValueSubclassData(unsigned short D) { SubclassData = D; }
};

inline raw_ostream &operator<<(raw_ostream &OS, const Value &V) {
  V.print(OS);
  return OS;
}

void Use::set(Value *V) {
  if (Val) removeFromList();
  Val = V;
  if (V) V->addUse(*this);
}

template <class Compare> void Value::sortUseList(Compare Cmp) {
  if (!UseList || !UseList->Next)
    // No need to sort 0 or 1 uses.
    return;

  // Note: this function completely ignores Prev pointers until the end when
  // they're fixed en masse.

  // Create a binomial vector of sorted lists, visiting uses one at a time and
  // merging lists as necessary.
  const unsigned MaxSlots = 32;
  Use *Slots[MaxSlots];

  // Collect the first use, turning it into a single-item list.
  Use *Next = UseList->Next;
  UseList->Next = nullptr;
  unsigned NumSlots = 1;
  Slots[0] = UseList;

  // Collect all but the last use.
  while (Next->Next) {
    Use *Current = Next;
    Next = Current->Next;

    // Turn Current into a single-item list.
    Current->Next = nullptr;

    // Save Current in the first available slot, merging on collisions.
    unsigned I;
    for (I = 0; I < NumSlots; ++I) {
      if (!Slots[I])
        break;

      // Merge two lists, doubling the size of Current and emptying slot I.
      //
      // Since the uses in Slots[I] originally preceded those in Current, send
      // Slots[I] in as the left parameter to maintain a stable sort.
      Current = mergeUseLists(Slots[I], Current, Cmp);
      Slots[I] = nullptr;
    }
    // Check if this is a new slot.
    if (I == NumSlots) {
      ++NumSlots;
      assert(NumSlots <= MaxSlots && "Use list bigger than 2^32");
    }

    // Found an open slot.
    Slots[I] = Current;
  }

  // Merge all the lists together.
  assert(Next && "Expected one more Use");
  assert(!Next->Next && "Expected only one Use");
  UseList = Next;
  for (unsigned I = 0; I < NumSlots; ++I)
    if (Slots[I])
      // Since the uses in Slots[I] originally preceded those in UseList, send
      // Slots[I] in as the left parameter to maintain a stable sort.
      UseList = mergeUseLists(Slots[I], UseList, Cmp);

  // Fix the Prev pointers.
  for (Use *I = UseList, **Prev = &UseList; I; I = I->Next) {
    I->setPrev(Prev);
    Prev = &I->Next;
  }
}

// HLSL Change Begins.
/*
template <class Compare>
void Value::mergeUseListsImpl(Use *L, Use *R, Use **Next, Compare Cmp) {
  if (!L) {
    *Next = R;
    return;
  }
  if (!R) {
    *Next = L;
    return;
  }
  if (Cmp(*R, *L)) {
    *Next = R;
    mergeUseListsImpl(L, R->Next, &R->Next, Cmp);
    return;
  }
  *Next = L;
  mergeUseListsImpl(L->Next, R, &L->Next, Cmp);
}
*/
// HLSL Change Ends.

// isa - Provide some specializations of isa so that we don't have to include
// the subtype header files to test to see if the value is a subclass...
//
template <> struct isa_impl<Constant, Value> {
  static inline bool doit(const Value &Val) {
    return Val.getValueID() >= Value::ConstantFirstVal &&
      Val.getValueID() <= Value::ConstantLastVal;
  }
};

template <> struct isa_impl<Argument, Value> {
  static inline bool doit (const Value &Val) {
    return Val.getValueID() == Value::ArgumentVal;
  }
};

template <> struct isa_impl<InlineAsm, Value> {
  static inline bool doit(const Value &Val) {
    return Val.getValueID() == Value::InlineAsmVal;
  }
};

template <> struct isa_impl<Instruction, Value> {
  static inline bool doit(const Value &Val) {
    return Val.getValueID() >= Value::InstructionVal;
  }
};

template <> struct isa_impl<BasicBlock, Value> {
  static inline bool doit(const Value &Val) {
    return Val.getValueID() == Value::BasicBlockVal;
  }
};

template <> struct isa_impl<Function, Value> {
  static inline bool doit(const Value &Val) {
    return Val.getValueID() == Value::FunctionVal;
  }
};

template <> struct isa_impl<GlobalVariable, Value> {
  static inline bool doit(const Value &Val) {
    return Val.getValueID() == Value::GlobalVariableVal;
  }
};

template <> struct isa_impl<GlobalAlias, Value> {
  static inline bool doit(const Value &Val) {
    return Val.getValueID() == Value::GlobalAliasVal;
  }
};

template <> struct isa_impl<GlobalValue, Value> {
  static inline bool doit(const Value &Val) {
    return isa<GlobalObject>(Val) || isa<GlobalAlias>(Val);
  }
};

template <> struct isa_impl<GlobalObject, Value> {
  static inline bool doit(const Value &Val) {
    return isa<GlobalVariable>(Val) || isa<Function>(Val);
  }
};

// Value* is only 4-byte aligned.
template<>
class PointerLikeTypeTraits<Value*> {
  typedef Value* PT;
public:
  static inline void *getAsVoidPointer(PT P) { return P; }
  static inline PT getFromVoidPointer(void *P) {
    return static_cast<PT>(P);
  }
  enum { NumLowBitsAvailable = 2 };
};

// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_ISA_CONVERSION_FUNCTIONS(Value, LLVMValueRef)

/* Specialized opaque value conversions.
 */
inline Value **unwrap(LLVMValueRef *Vals) {
  return reinterpret_cast<Value**>(Vals);
}

template<typename T>
inline T **unwrap(LLVMValueRef *Vals, unsigned Length) {
#ifdef DEBUG
  for (LLVMValueRef *I = Vals, *E = Vals + Length; I != E; ++I)
    cast<T>(*I);
#endif
  (void)Length;
  return reinterpret_cast<T**>(Vals);
}

inline LLVMValueRef *wrap(const Value **Vals) {
  return reinterpret_cast<LLVMValueRef*>(const_cast<Value**>(Vals));
}

} // End llvm namespace

#endif
