//===-- llvm/GlobalValue.h - Class to represent a global value --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a common base class of all globally definable objects.  As such,
// it is subclassed by GlobalVariable, GlobalAlias and by Function.  This is
// used because you can do certain things with these global objects that you
// can't do to anything else.  For example, use the address of one as a
// constant.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_GLOBALVALUE_H
#define LLVM_IR_GLOBALVALUE_H

#include "llvm/IR/Constant.h"
#include "llvm/IR/DerivedTypes.h"
#include <system_error>

namespace llvm {

class Comdat;
class PointerType;
class Module;

namespace Intrinsic {
  enum ID : unsigned;
}

class GlobalValue : public Constant {
  GlobalValue(const GlobalValue &) = delete;
public:
  /// @brief An enumeration for the kinds of linkage for global values.
  enum LinkageTypes {
    ExternalLinkage = 0,///< Externally visible function
    AvailableExternallyLinkage, ///< Available for inspection, not emission.
    LinkOnceAnyLinkage, ///< Keep one copy of function when linking (inline)
    LinkOnceODRLinkage, ///< Same, but only replaced by something equivalent.
    WeakAnyLinkage,     ///< Keep one copy of named function when linking (weak)
    WeakODRLinkage,     ///< Same, but only replaced by something equivalent.
    AppendingLinkage,   ///< Special purpose, only applies to global arrays
    InternalLinkage,    ///< Rename collisions when linking (static functions).
    PrivateLinkage,     ///< Like Internal, but omit from symbol table.
    ExternalWeakLinkage,///< ExternalWeak linkage description.
    CommonLinkage       ///< Tentative definitions.
  };

  /// @brief An enumeration for the kinds of visibility of global values.
  enum VisibilityTypes {
    DefaultVisibility = 0,  ///< The GV is visible
    HiddenVisibility,       ///< The GV is hidden
    ProtectedVisibility     ///< The GV is protected
  };

  /// @brief Storage classes of global values for PE targets.
  enum DLLStorageClassTypes {
    DefaultStorageClass   = 0,
    DLLImportStorageClass = 1, ///< Function to be imported from DLL
    DLLExportStorageClass = 2  ///< Function to be accessible from DLL.
  };

protected:
  GlobalValue(PointerType *Ty, ValueTy VTy, Use *Ops, unsigned NumOps,
              LinkageTypes Linkage, const Twine &Name)
      : Constant(Ty, VTy, Ops, NumOps), Linkage(Linkage),
        Visibility(DefaultVisibility), UnnamedAddr(0),
        DllStorageClass(DefaultStorageClass),
        ThreadLocal(NotThreadLocal), IntID((Intrinsic::ID)0U), Parent(nullptr) {
    setName(Name);
  }

  // Note: VC++ treats enums as signed, so an extra bit is required to prevent
  // Linkage and Visibility from turning into negative values.
  LinkageTypes Linkage : 5;   // The linkage of this global
  unsigned Visibility : 2;    // The visibility style of this global
  unsigned UnnamedAddr : 1;   // This value's address is not significant
  unsigned DllStorageClass : 2; // DLL storage class

  unsigned ThreadLocal : 3; // Is this symbol "Thread Local", if so, what is
                            // the desired model?
  static const unsigned GlobalValueSubClassDataBits = 19;

private:
  // Give subclasses access to what otherwise would be wasted padding.
  // (19 + 3 + 2 + 1 + 2 + 5) == 32.
  unsigned SubClassData : GlobalValueSubClassDataBits;

  friend class Constant;
  void destroyConstantImpl();
  Value *handleOperandChangeImpl(Value *From, Value *To, Use *U);

protected:
  /// \brief The intrinsic ID for this subclass (which must be a Function).
  ///
  /// This member is defined by this class, but not used for anything.
  /// Subclasses can use it to store their intrinsic ID, if they have one.
  ///
  /// This is stored here to save space in Function on 64-bit hosts.
  Intrinsic::ID IntID;

  unsigned getGlobalValueSubClassData() const {
    return SubClassData;
  }
  void setGlobalValueSubClassData(unsigned V) {
    assert(V < (1 << GlobalValueSubClassDataBits) && "It will not fit");
    SubClassData = V;
  }

  Module *Parent;             // The containing module.
public:
  enum ThreadLocalMode {
    NotThreadLocal = 0,
    GeneralDynamicTLSModel,
    LocalDynamicTLSModel,
    InitialExecTLSModel,
    LocalExecTLSModel
  };

  ~GlobalValue() override {
    removeDeadConstantUsers();   // remove any dead constants using this.
  }

  unsigned getAlignment() const;

  bool hasUnnamedAddr() const { return UnnamedAddr; }
  void setUnnamedAddr(bool Val) { UnnamedAddr = Val; }

  bool hasComdat() const { return getComdat() != nullptr; }
  Comdat *getComdat();
  const Comdat *getComdat() const {
    return const_cast<GlobalValue *>(this)->getComdat();
  }

  VisibilityTypes getVisibility() const { return VisibilityTypes(Visibility); }
  bool hasDefaultVisibility() const { return Visibility == DefaultVisibility; }
  bool hasHiddenVisibility() const { return Visibility == HiddenVisibility; }
  bool hasProtectedVisibility() const {
    return Visibility == ProtectedVisibility;
  }
  void setVisibility(VisibilityTypes V) {
    assert((!hasLocalLinkage() || V == DefaultVisibility) &&
           "local linkage requires default visibility");
    Visibility = V;
  }

  /// If the value is "Thread Local", its value isn't shared by the threads.
  bool isThreadLocal() const { return getThreadLocalMode() != NotThreadLocal; }
  void setThreadLocal(bool Val) {
    setThreadLocalMode(Val ? GeneralDynamicTLSModel : NotThreadLocal);
  }
  void setThreadLocalMode(ThreadLocalMode Val) {
    assert(Val == NotThreadLocal || getValueID() != Value::FunctionVal);
    ThreadLocal = Val;
  }
  ThreadLocalMode getThreadLocalMode() const {
    return static_cast<ThreadLocalMode>(ThreadLocal);
  }

  DLLStorageClassTypes getDLLStorageClass() const {
    return DLLStorageClassTypes(DllStorageClass);
  }
  bool hasDLLImportStorageClass() const {
    return DllStorageClass == DLLImportStorageClass;
  }
  bool hasDLLExportStorageClass() const {
    return DllStorageClass == DLLExportStorageClass;
  }
  void setDLLStorageClass(DLLStorageClassTypes C) { DllStorageClass = C; }

  bool hasSection() const { return !StringRef(getSection()).empty(); }
  // It is unfortunate that we have to use "char *" in here since this is
  // always non NULL, but:
  // * The C API expects a null terminated string, so we cannot use StringRef.
  // * The C API expects us to own it, so we cannot use a std:string.
  // * For GlobalAliases we can fail to find the section and we have to
  //   return "", so we cannot use a "const std::string &".
  const char *getSection() const;

  /// Global values are always pointers.
  PointerType *getType() const { return cast<PointerType>(User::getType()); }

  Type *getValueType() const { return getType()->getElementType(); }

  static LinkageTypes getLinkOnceLinkage(bool ODR) {
    return ODR ? LinkOnceODRLinkage : LinkOnceAnyLinkage;
  }
  static LinkageTypes getWeakLinkage(bool ODR) {
    return ODR ? WeakODRLinkage : WeakAnyLinkage;
  }

  static bool isExternalLinkage(LinkageTypes Linkage) {
    return Linkage == ExternalLinkage;
  }
  static bool isAvailableExternallyLinkage(LinkageTypes Linkage) {
    return Linkage == AvailableExternallyLinkage;
  }
  static bool isLinkOnceODRLinkage(LinkageTypes Linkage) {
    return Linkage == LinkOnceODRLinkage;
  }
  static bool isLinkOnceLinkage(LinkageTypes Linkage) {
    return Linkage == LinkOnceAnyLinkage || Linkage == LinkOnceODRLinkage;
  }
  static bool isWeakAnyLinkage(LinkageTypes Linkage) {
    return Linkage == WeakAnyLinkage;
  }
  static bool isWeakODRLinkage(LinkageTypes Linkage) {
    return Linkage == WeakODRLinkage;
  }
  static bool isWeakLinkage(LinkageTypes Linkage) {
    return isWeakAnyLinkage(Linkage) || isWeakODRLinkage(Linkage);
  }
  static bool isAppendingLinkage(LinkageTypes Linkage) {
    return Linkage == AppendingLinkage;
  }
  static bool isInternalLinkage(LinkageTypes Linkage) {
    return Linkage == InternalLinkage;
  }
  static bool isPrivateLinkage(LinkageTypes Linkage) {
    return Linkage == PrivateLinkage;
  }
  static bool isLocalLinkage(LinkageTypes Linkage) {
    return isInternalLinkage(Linkage) || isPrivateLinkage(Linkage);
  }
  static bool isExternalWeakLinkage(LinkageTypes Linkage) {
    return Linkage == ExternalWeakLinkage;
  }
  static bool isCommonLinkage(LinkageTypes Linkage) {
    return Linkage == CommonLinkage;
  }

  /// Whether the definition of this global may be discarded if it is not used
  /// in its compilation unit.
  static bool isDiscardableIfUnused(LinkageTypes Linkage) {
    return isLinkOnceLinkage(Linkage) || isLocalLinkage(Linkage);
  }

  /// Whether the definition of this global may be replaced by something
  /// non-equivalent at link time. For example, if a function has weak linkage
  /// then the code defining it may be replaced by different code.
  static bool mayBeOverridden(LinkageTypes Linkage) {
    return Linkage == WeakAnyLinkage || Linkage == LinkOnceAnyLinkage ||
           Linkage == CommonLinkage || Linkage == ExternalWeakLinkage;
  }

  /// Whether the definition of this global may be replaced at link time.  NB:
  /// Using this method outside of the code generators is almost always a
  /// mistake: when working at the IR level use mayBeOverridden instead as it
  /// knows about ODR semantics.
  static bool isWeakForLinker(LinkageTypes Linkage)  {
    return Linkage == WeakAnyLinkage || Linkage == WeakODRLinkage ||
           Linkage == LinkOnceAnyLinkage || Linkage == LinkOnceODRLinkage ||
           Linkage == CommonLinkage || Linkage == ExternalWeakLinkage;
  }

  bool hasExternalLinkage() const { return isExternalLinkage(Linkage); }
  bool hasAvailableExternallyLinkage() const {
    return isAvailableExternallyLinkage(Linkage);
  }
  bool hasLinkOnceLinkage() const {
    return isLinkOnceLinkage(Linkage);
  }
  bool hasLinkOnceODRLinkage() const { return isLinkOnceODRLinkage(Linkage); }
  bool hasWeakLinkage() const {
    return isWeakLinkage(Linkage);
  }
  bool hasWeakAnyLinkage() const {
    return isWeakAnyLinkage(Linkage);
  }
  bool hasWeakODRLinkage() const {
    return isWeakODRLinkage(Linkage);
  }
  bool hasAppendingLinkage() const { return isAppendingLinkage(Linkage); }
  bool hasInternalLinkage() const { return isInternalLinkage(Linkage); }
  bool hasPrivateLinkage() const { return isPrivateLinkage(Linkage); }
  bool hasLocalLinkage() const { return isLocalLinkage(Linkage); }
  bool hasExternalWeakLinkage() const { return isExternalWeakLinkage(Linkage); }
  bool hasCommonLinkage() const { return isCommonLinkage(Linkage); }

  void setLinkage(LinkageTypes LT) {
    if (isLocalLinkage(LT))
      Visibility = DefaultVisibility;
    Linkage = LT;
  }
  LinkageTypes getLinkage() const { return Linkage; }

  bool isDiscardableIfUnused() const {
    return isDiscardableIfUnused(Linkage);
  }

  bool mayBeOverridden() const { return mayBeOverridden(Linkage); }

  bool isWeakForLinker() const { return isWeakForLinker(Linkage); }

  /// Copy all additional attributes (those not needed to create a GlobalValue)
  /// from the GlobalValue Src to this one.
  virtual void copyAttributesFrom(const GlobalValue *Src);

  /// If special LLVM prefix that is used to inform the asm printer to not emit
  /// usual symbol prefix before the symbol name is used then return linkage
  /// name after skipping this special LLVM prefix.
  static StringRef getRealLinkageName(StringRef Name) {
    if (!Name.empty() && Name[0] == '\1')
      return Name.substr(1);
    return Name;
  }

/// @name Materialization
/// Materialization is used to construct functions only as they're needed. This
/// is useful to reduce memory usage in LLVM or parsing work done by the
/// BitcodeReader to load the Module.
/// @{

  /// If this function's Module is being lazily streamed in functions from disk
  /// or some other source, this method can be used to check to see if the
  /// function has been read in yet or not.
  bool isMaterializable() const;

  /// Returns true if this function was loaded from a GVMaterializer that's
  /// still attached to its Module and that knows how to dematerialize the
  /// function.
  bool isDematerializable() const;

  /// Make sure this GlobalValue is fully read. If the module is corrupt, this
  /// returns true and fills in the optional string with information about the
  /// problem.  If successful, this returns false.
  std::error_code materialize();

  /// If this GlobalValue is read in, and if the GVMaterializer supports it,
  /// release the memory for the function, and set it up to be materialized
  /// lazily. If !isDematerializable(), this method is a noop.
  void dematerialize();

/// @}

  /// Return true if the primary definition of this global value is outside of
  /// the current translation unit.
  bool isDeclaration() const;

  bool isDeclarationForLinker() const {
    if (hasAvailableExternallyLinkage())
      return true;

    return isDeclaration();
  }

  /// Returns true if this global's definition will be the one chosen by the
  /// linker.
  bool isStrongDefinitionForLinker() const {
    return !(isDeclarationForLinker() || isWeakForLinker());
  }

  /// This method unlinks 'this' from the containing module, but does not delete
  /// it.
  virtual void removeFromParent() = 0;

  /// This method unlinks 'this' from the containing module and deletes it.
  virtual void eraseFromParent() = 0;

  /// Get the module that this global value is contained inside of...
  Module *getParent() { return Parent; }
  const Module *getParent() const { return Parent; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const Value *V) {
    return V->getValueID() == Value::FunctionVal ||
           V->getValueID() == Value::GlobalVariableVal ||
           V->getValueID() == Value::GlobalAliasVal;
  }
};

} // End llvm namespace

#endif
