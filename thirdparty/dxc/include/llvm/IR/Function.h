//===-- llvm/Function.h - Class to represent a single function --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the Function class, which represents a
// single function/procedure in LLVM.
//
// A function basically consists of a list of basic blocks, a list of arguments,
// and a symbol table.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_FUNCTION_H
#define LLVM_IR_FUNCTION_H

#include "llvm/ADT/iterator_range.h"
#include "llvm/ADT/Optional.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/OperandTraits.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class FunctionType;
class LLVMContext;

template<> struct ilist_traits<Argument>
  : public SymbolTableListTraits<Argument, Function> {

  Argument *createSentinel() const {
    return static_cast<Argument*>(&Sentinel);
  }
  static void destroySentinel(Argument*) {}

  Argument *provideInitialHead() const { return createSentinel(); }
  Argument *ensureHead(Argument*) const { return createSentinel(); }
  static void noteHead(Argument*, Argument*) {}

  static ValueSymbolTable *getSymTab(Function *ItemParent);
private:
  mutable ilist_half_node<Argument> Sentinel;
};

class Function : public GlobalObject, public ilist_node<Function> {
public:
  typedef iplist<Argument> ArgumentListType;
  typedef iplist<BasicBlock> BasicBlockListType;

  // BasicBlock iterators...
  typedef BasicBlockListType::iterator iterator;
  typedef BasicBlockListType::const_iterator const_iterator;

  typedef ArgumentListType::iterator arg_iterator;
  typedef ArgumentListType::const_iterator const_arg_iterator;

private:
  // Important things that make up a function!
  BasicBlockListType  BasicBlocks;        ///< The basic blocks
  mutable ArgumentListType ArgumentList;  ///< The formal arguments
  std::unique_ptr<ValueSymbolTable> SymTab; ///< Symbol table of args/instructions // HLSL Change: use unique_ptr
  AttributeSet AttributeSets;             ///< Parameter attributes
  FunctionType *Ty;

  /*
   * Value::SubclassData
   *
   * bit 0  : HasLazyArguments
   * bit 1  : HasPrefixData
   * bit 2  : HasPrologueData
   * bit 3-6: CallingConvention
   */

  /// Bits from GlobalObject::GlobalObjectSubclassData.
  enum {
    /// Whether this function is materializable.
    IsMaterializableBit = 1 << 0,
    HasMetadataHashEntryBit = 1 << 1
  };
  void setGlobalObjectBit(unsigned Mask, bool Value) {
    setGlobalObjectSubClassData((~Mask & getGlobalObjectSubClassData()) |
                                (Value ? Mask : 0u));
  }

  friend class SymbolTableListTraits<Function, Module>;

  void setParent(Module *parent);

  /// hasLazyArguments/CheckLazyArguments - The argument list of a function is
  /// built on demand, so that the list isn't allocated until the first client
  /// needs it.  The hasLazyArguments predicate returns true if the arg list
  /// hasn't been set up yet.
  bool hasLazyArguments() const {
    return getSubclassDataFromValue() & (1<<0);
  }
  void CheckLazyArguments() const {
    if (hasLazyArguments())
      BuildLazyArguments();
  }
  void BuildLazyArguments() const;

  Function(const Function&) = delete;
  void operator=(const Function&) = delete;

  /// Function ctor - If the (optional) Module argument is specified, the
  /// function is automatically inserted into the end of the function list for
  /// the module.
  ///
  Function(FunctionType *Ty, LinkageTypes Linkage,
           const Twine &N = "", Module *M = nullptr);

public:
  static Function *Create(FunctionType *Ty, LinkageTypes Linkage,
                          const Twine &N = "", Module *M = nullptr) {
    return new(1) Function(Ty, Linkage, N, M);
  }

  ~Function() override;

  /// \brief Provide fast operand accessors
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  /// \brief Get the personality function associated with this function.
  bool hasPersonalityFn() const { return getNumOperands() != 0; }
  Constant *getPersonalityFn() const {
    assert(hasPersonalityFn());
    return cast<Constant>(Op<0>());
  }
  void setPersonalityFn(Constant *C);

  Type *getReturnType() const;           // Return the type of the ret val
  FunctionType *getFunctionType() const; // Return the FunctionType for me

  /// getContext - Return a reference to the LLVMContext associated with this
  /// function.
  LLVMContext &getContext() const;

  /// isVarArg - Return true if this function takes a variable number of
  /// arguments.
  bool isVarArg() const;

  bool isMaterializable() const;
  void setIsMaterializable(bool V);

  /// getIntrinsicID - This method returns the ID number of the specified
  /// function, or Intrinsic::not_intrinsic if the function is not an
  /// intrinsic, or if the pointer is null.  This value is always defined to be
  /// zero to allow easy checking for whether a function is intrinsic or not.
  /// The particular intrinsic functions which correspond to this value are
  /// defined in llvm/Intrinsics.h.
  Intrinsic::ID getIntrinsicID() const LLVM_READONLY { return IntID; }
  bool isIntrinsic() const { return getName().startswith("llvm."); }

  /// \brief Recalculate the ID for this function if it is an Intrinsic defined
  /// in llvm/Intrinsics.h.  Sets the intrinsic ID to Intrinsic::not_intrinsic
  /// if the name of this function does not match an intrinsic in that header.
  /// Note, this method does not need to be called directly, as it is called
  /// from Value::setName() whenever the name of this function changes.
  void recalculateIntrinsicID();

  /// getCallingConv()/setCallingConv(CC) - These method get and set the
  /// calling convention of this function.  The enum values for the known
  /// calling conventions are defined in CallingConv.h.
  CallingConv::ID getCallingConv() const {
    return static_cast<CallingConv::ID>(getSubclassDataFromValue() >> 3);
  }
  void setCallingConv(CallingConv::ID CC) {
    setValueSubclassData((getSubclassDataFromValue() & 7) |
                         (static_cast<unsigned>(CC) << 3));
  }

  /// @brief Return the attribute list for this Function.
  AttributeSet getAttributes() const { return AttributeSets; }

  /// @brief Set the attribute list for this Function.
  void setAttributes(AttributeSet attrs) { AttributeSets = attrs; }

  /// @brief Add function attributes to this function.
  void addFnAttr(Attribute::AttrKind N) {
    setAttributes(AttributeSets.addAttribute(getContext(),
                                             AttributeSet::FunctionIndex, N));
  }

  /// @brief Remove function attributes from this function.
  void removeFnAttr(Attribute::AttrKind N) {
    setAttributes(AttributeSets.removeAttribute(
        getContext(), AttributeSet::FunctionIndex, N));
  }

  /// @brief Add function attributes to this function.
  void addFnAttr(StringRef Kind) {
    setAttributes(
      AttributeSets.addAttribute(getContext(),
                                 AttributeSet::FunctionIndex, Kind));
  }
  void addFnAttr(StringRef Kind, StringRef Value) {
    setAttributes(
      AttributeSets.addAttribute(getContext(),
                                 AttributeSet::FunctionIndex, Kind, Value));
  }

  /// Set the entry count for this function.
  void setEntryCount(uint64_t Count);

  /// Get the entry count for this function.
  Optional<uint64_t> getEntryCount() const;

  /// @brief Return true if the function has the attribute.
  bool hasFnAttribute(Attribute::AttrKind Kind) const {
    return AttributeSets.hasAttribute(AttributeSet::FunctionIndex, Kind);
  }
  bool hasFnAttribute(StringRef Kind) const {
    return AttributeSets.hasAttribute(AttributeSet::FunctionIndex, Kind);
  }

  /// @brief Return the attribute for the given attribute kind.
  Attribute getFnAttribute(Attribute::AttrKind Kind) const {
    return AttributeSets.getAttribute(AttributeSet::FunctionIndex, Kind);
  }
  Attribute getFnAttribute(StringRef Kind) const {
    return AttributeSets.getAttribute(AttributeSet::FunctionIndex, Kind);
  }

  /// \brief Return the stack alignment for the function.
  unsigned getFnStackAlignment() const {
    return AttributeSets.getStackAlignment(AttributeSet::FunctionIndex);
  }

  /// hasGC/getGC/setGC/clearGC - The name of the garbage collection algorithm
  ///                             to use during code generation.
  bool hasGC() const;
  const char *getGC() const;
  void setGC(const char *Str);
  void clearGC();

  /// @brief adds the attribute to the list of attributes.
  void addAttribute(unsigned i, Attribute::AttrKind attr);

  /// @brief adds the attributes to the list of attributes.
  void addAttributes(unsigned i, AttributeSet attrs);

  /// @brief removes the attributes from the list of attributes.
  void removeAttributes(unsigned i, AttributeSet attr);

  /// @brief adds the dereferenceable attribute to the list of attributes.
  void addDereferenceableAttr(unsigned i, uint64_t Bytes);

  /// @brief adds the dereferenceable_or_null attribute to the list of
  /// attributes.
  void addDereferenceableOrNullAttr(unsigned i, uint64_t Bytes);

  /// @brief Extract the alignment for a call or parameter (0=unknown).
  unsigned getParamAlignment(unsigned i) const {
    return AttributeSets.getParamAlignment(i);
  }

  /// @brief Extract the number of dereferenceable bytes for a call or
  /// parameter (0=unknown).
  uint64_t getDereferenceableBytes(unsigned i) const {
    return AttributeSets.getDereferenceableBytes(i);
  }
  
  /// @brief Extract the number of dereferenceable_or_null bytes for a call or
  /// parameter (0=unknown).
  uint64_t getDereferenceableOrNullBytes(unsigned i) const {
    return AttributeSets.getDereferenceableOrNullBytes(i);
  }
  
  /// @brief Determine if the function does not access memory.
  bool doesNotAccessMemory() const {
    return AttributeSets.hasAttribute(AttributeSet::FunctionIndex,
                                      Attribute::ReadNone);
  }
  void setDoesNotAccessMemory() {
    addFnAttr(Attribute::ReadNone);
  }

  /// @brief Determine if the function does not access or only reads memory.
  bool onlyReadsMemory() const {
    return doesNotAccessMemory() ||
      AttributeSets.hasAttribute(AttributeSet::FunctionIndex,
                                 Attribute::ReadOnly);
  }
  void setOnlyReadsMemory() {
    addFnAttr(Attribute::ReadOnly);
  }

  /// @brief Determine if the call can access memmory only using pointers based
  /// on its arguments.
  bool onlyAccessesArgMemory() const {
    return AttributeSets.hasAttribute(AttributeSet::FunctionIndex,
                                      Attribute::ArgMemOnly);
  }
  void setOnlyAccessesArgMemory() {
    addFnAttr(Attribute::ArgMemOnly);
  }
  
  /// @brief Determine if the function cannot return.
  bool doesNotReturn() const {
    return AttributeSets.hasAttribute(AttributeSet::FunctionIndex,
                                      Attribute::NoReturn);
  }
  void setDoesNotReturn() {
    addFnAttr(Attribute::NoReturn);
  }

  /// @brief Determine if the function cannot unwind.
  bool doesNotThrow() const {
    return AttributeSets.hasAttribute(AttributeSet::FunctionIndex,
                                      Attribute::NoUnwind);
  }
  void setDoesNotThrow() {
    addFnAttr(Attribute::NoUnwind);
  }

  /// @brief Determine if the call cannot be duplicated.
  bool cannotDuplicate() const {
    return AttributeSets.hasAttribute(AttributeSet::FunctionIndex,
                                      Attribute::NoDuplicate);
  }
  void setCannotDuplicate() {
    addFnAttr(Attribute::NoDuplicate);
  }

  /// @brief Determine if the call is convergent.
  bool isConvergent() const {
    return AttributeSets.hasAttribute(AttributeSet::FunctionIndex,
                                      Attribute::Convergent);
  }
  void setConvergent() {
    addFnAttr(Attribute::Convergent);
  }


  /// @brief True if the ABI mandates (or the user requested) that this
  /// function be in a unwind table.
  bool hasUWTable() const {
    return AttributeSets.hasAttribute(AttributeSet::FunctionIndex,
                                      Attribute::UWTable);
  }
  void setHasUWTable() {
    addFnAttr(Attribute::UWTable);
  }

  /// @brief True if this function needs an unwind table.
  bool needsUnwindTableEntry() const {
    return hasUWTable() || !doesNotThrow();
  }

  /// @brief Determine if the function returns a structure through first
  /// pointer argument.
  bool hasStructRetAttr() const {
    return AttributeSets.hasAttribute(1, Attribute::StructRet) ||
           AttributeSets.hasAttribute(2, Attribute::StructRet);
  }

  /// @brief Determine if the parameter does not alias other parameters.
  /// @param n The parameter to check. 1 is the first parameter, 0 is the return
  bool doesNotAlias(unsigned n) const {
    return AttributeSets.hasAttribute(n, Attribute::NoAlias);
  }
  void setDoesNotAlias(unsigned n) {
    addAttribute(n, Attribute::NoAlias);
  }

  /// @brief Determine if the parameter can be captured.
  /// @param n The parameter to check. 1 is the first parameter, 0 is the return
  bool doesNotCapture(unsigned n) const {
    return AttributeSets.hasAttribute(n, Attribute::NoCapture);
  }
  void setDoesNotCapture(unsigned n) {
    addAttribute(n, Attribute::NoCapture);
  }

  bool doesNotAccessMemory(unsigned n) const {
    return AttributeSets.hasAttribute(n, Attribute::ReadNone);
  }
  void setDoesNotAccessMemory(unsigned n) {
    addAttribute(n, Attribute::ReadNone);
  }

  bool onlyReadsMemory(unsigned n) const {
    return doesNotAccessMemory(n) ||
      AttributeSets.hasAttribute(n, Attribute::ReadOnly);
  }
  void setOnlyReadsMemory(unsigned n) {
    addAttribute(n, Attribute::ReadOnly);
  }

  /// copyAttributesFrom - copy all additional attributes (those not needed to
  /// create a Function) from the Function Src to this one.
  void copyAttributesFrom(const GlobalValue *Src) override;

  /// deleteBody - This method deletes the body of the function, and converts
  /// the linkage to external.
  ///
  void deleteBody() {
    dropAllReferences();
    setLinkage(ExternalLinkage);
  }

  /// removeFromParent - This method unlinks 'this' from the containing module,
  /// but does not delete it.
  ///
  void removeFromParent() override;

  /// eraseFromParent - This method unlinks 'this' from the containing module
  /// and deletes it.
  ///
  void eraseFromParent() override;


  /// Get the underlying elements of the Function... the basic block list is
  /// empty for external functions.
  ///
  const ArgumentListType &getArgumentList() const {
    CheckLazyArguments();
    return ArgumentList;
  }
  ArgumentListType &getArgumentList() {
    CheckLazyArguments();
    return ArgumentList;
  }
  static iplist<Argument> Function::*getSublistAccess(Argument*) {
    return &Function::ArgumentList;
  }

  const BasicBlockListType &getBasicBlockList() const { return BasicBlocks; }
        BasicBlockListType &getBasicBlockList()       { return BasicBlocks; }
  static iplist<BasicBlock> Function::*getSublistAccess(BasicBlock*) {
    return &Function::BasicBlocks;
  }

  const BasicBlock       &getEntryBlock() const   { return front(); }
        BasicBlock       &getEntryBlock()         { return front(); }

  //===--------------------------------------------------------------------===//
  // Symbol Table Accessing functions...

  /// getSymbolTable() - Return the symbol table...
  ///
  inline       ValueSymbolTable &getValueSymbolTable()       { return *SymTab; }
  inline const ValueSymbolTable &getValueSymbolTable() const { return *SymTab; }


  //===--------------------------------------------------------------------===//
  // BasicBlock iterator forwarding functions
  //
  iterator                begin()       { return BasicBlocks.begin(); }
  const_iterator          begin() const { return BasicBlocks.begin(); }
  iterator                end  ()       { return BasicBlocks.end();   }
  const_iterator          end  () const { return BasicBlocks.end();   }

  size_t                   size() const { return BasicBlocks.size();  }
  bool                    empty() const { return BasicBlocks.empty(); }
  const BasicBlock       &front() const { return BasicBlocks.front(); }
        BasicBlock       &front()       { return BasicBlocks.front(); }
  const BasicBlock        &back() const { return BasicBlocks.back();  }
        BasicBlock        &back()       { return BasicBlocks.back();  }

/// @name Function Argument Iteration
/// @{

  arg_iterator arg_begin() {
    CheckLazyArguments();
    return ArgumentList.begin();
  }
  const_arg_iterator arg_begin() const {
    CheckLazyArguments();
    return ArgumentList.begin();
  }
  arg_iterator arg_end() {
    CheckLazyArguments();
    return ArgumentList.end();
  }
  const_arg_iterator arg_end() const {
    CheckLazyArguments();
    return ArgumentList.end();
  }

  iterator_range<arg_iterator> args() {
    return iterator_range<arg_iterator>(arg_begin(), arg_end());
  }

  iterator_range<const_arg_iterator> args() const {
    return iterator_range<const_arg_iterator>(arg_begin(), arg_end());
  }

/// @}

  size_t arg_size() const;
  bool arg_empty() const;

  bool hasPrefixData() const {
    return getSubclassDataFromValue() & (1<<1);
  }

  Constant *getPrefixData() const;
  void setPrefixData(Constant *PrefixData);

  bool hasPrologueData() const {
    return getSubclassDataFromValue() & (1<<2);
  }

  Constant *getPrologueData() const;
  void setPrologueData(Constant *PrologueData);

  /// Print the function to an output stream with an optional
  /// AssemblyAnnotationWriter.
  void print(raw_ostream &OS, AssemblyAnnotationWriter *AAW = nullptr) const;

  /// viewCFG - This function is meant for use from the debugger.  You can just
  /// say 'call F->viewCFG()' and a ghostview window should pop up from the
  /// program, displaying the CFG of the current function with the code for each
  /// basic block inside.  This depends on there being a 'dot' and 'gv' program
  /// in your path.
  ///
  void viewCFG() const;

  /// viewCFGOnly - This function is meant for use from the debugger.  It works
  /// just like viewCFG, but it does not include the contents of basic blocks
  /// into the nodes, just the label.  If you are only interested in the CFG
  /// this can make the graph smaller.
  ///
  void viewCFGOnly() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Value *V) {
    return V->getValueID() == Value::FunctionVal;
  }

  /// dropAllReferences() - This method causes all the subinstructions to "let
  /// go" of all references that they are maintaining.  This allows one to
  /// 'delete' a whole module at a time, even though there may be circular
  /// references... first all references are dropped, and all use counts go to
  /// zero.  Then everything is deleted for real.  Note that no operations are
  /// valid on an object that has "dropped all references", except operator
  /// delete.
  ///
  /// Since no other object in the module can have references into the body of a
  /// function, dropping all references deletes the entire body of the function,
  /// including any contained basic blocks.
  ///
  void dropAllReferences();

  /// hasAddressTaken - returns true if there are any uses of this function
  /// other than direct calls or invokes to it, or blockaddress expressions.
  /// Optionally passes back an offending user for diagnostic purposes.
  ///
  bool hasAddressTaken(const User** = nullptr) const;

  /// isDefTriviallyDead - Return true if it is trivially safe to remove
  /// this function definition from the module (because it isn't externally
  /// visible, does not have its address taken, and has no callers).  To make
  /// this more accurate, call removeDeadConstantUsers first.
  bool isDefTriviallyDead() const;

  /// callsFunctionThatReturnsTwice - Return true if the function has a call to
  /// setjmp or other function that gcc recognizes as "returning twice".
  bool callsFunctionThatReturnsTwice() const;

  /// \brief Check if this has any metadata.
  bool hasMetadata() const { return hasMetadataHashEntry(); }

  /// \brief Get the current metadata attachment, if any.
  ///
  /// Returns \c nullptr if such an attachment is missing.
  /// @{
  MDNode *getMetadata(unsigned KindID) const;
  MDNode *getMetadata(StringRef Kind) const;
  /// @}

  /// \brief Set a particular kind of metadata attachment.
  ///
  /// Sets the given attachment to \c MD, erasing it if \c MD is \c nullptr or
  /// replacing it if it already exists.
  /// @{
  void setMetadata(unsigned KindID, MDNode *MD);
  void setMetadata(StringRef Kind, MDNode *MD);
  /// @}

  /// \brief Get all current metadata attachments.
  void
  getAllMetadata(SmallVectorImpl<std::pair<unsigned, MDNode *>> &MDs) const;

  /// \brief Drop metadata not in the given list.
  ///
  /// Drop all metadata from \c this not included in \c KnownIDs.
  void dropUnknownMetadata(ArrayRef<unsigned> KnownIDs);

private:
  // Shadow Value::setValueSubclassData with a private forwarding method so that
  // subclasses cannot accidentally use it.
  void setValueSubclassData(unsigned short D) {
    Value::setValueSubclassData(D);
  }

  bool hasMetadataHashEntry() const {
    return getGlobalObjectSubClassData() & HasMetadataHashEntryBit;
  }
  void setHasMetadataHashEntry(bool HasEntry) {
    setGlobalObjectBit(HasMetadataHashEntryBit, HasEntry);
  }

  void clearMetadata();
};

inline ValueSymbolTable *
ilist_traits<BasicBlock>::getSymTab(Function *F) {
  return F ? &F->getValueSymbolTable() : nullptr;
}

inline ValueSymbolTable *
ilist_traits<Argument>::getSymTab(Function *F) {
  return F ? &F->getValueSymbolTable() : nullptr;
}

template <>
struct OperandTraits<Function> : public OptionalOperandTraits<Function> {};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(Function, Value)

} // End llvm namespace

#endif
