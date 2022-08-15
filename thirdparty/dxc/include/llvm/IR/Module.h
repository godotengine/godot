//===-- llvm/Module.h - C++ class to represent a VM module ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// @file
/// Module.h This file contains the declarations for the Module class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_MODULE_H
#define LLVM_IR_MODULE_H

#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/Comdat.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/DataTypes.h"
#include <system_error>

// HLSL Change start
namespace hlsl {
class DxilModule;
class HLModule;
}
// HLSL Change end

namespace llvm {
class FunctionType;
class GVMaterializer;
class LLVMContext;
class RandomNumberGenerator;
class StructType;

template<> struct ilist_traits<Function>
  : public SymbolTableListTraits<Function, Module> {

  // createSentinel is used to get hold of the node that marks the end of the
  // list... (same trick used here as in ilist_traits<Instruction>)
  Function *createSentinel() const {
    return static_cast<Function*>(&Sentinel);
  }
  static void destroySentinel(Function*) {}

  Function *provideInitialHead() const { return createSentinel(); }
  Function *ensureHead(Function*) const { return createSentinel(); }
  static void noteHead(Function*, Function*) {}

private:
  mutable ilist_node<Function> Sentinel;
};

template<> struct ilist_traits<GlobalVariable>
  : public SymbolTableListTraits<GlobalVariable, Module> {
  // createSentinel is used to create a node that marks the end of the list.
  GlobalVariable *createSentinel() const {
    return static_cast<GlobalVariable*>(&Sentinel);
  }
  static void destroySentinel(GlobalVariable*) {}

  GlobalVariable *provideInitialHead() const { return createSentinel(); }
  GlobalVariable *ensureHead(GlobalVariable*) const { return createSentinel(); }
  static void noteHead(GlobalVariable*, GlobalVariable*) {}
private:
  mutable ilist_node<GlobalVariable> Sentinel;
};

template<> struct ilist_traits<GlobalAlias>
  : public SymbolTableListTraits<GlobalAlias, Module> {
  // createSentinel is used to create a node that marks the end of the list.
  GlobalAlias *createSentinel() const {
    return static_cast<GlobalAlias*>(&Sentinel);
  }
  static void destroySentinel(GlobalAlias*) {}

  GlobalAlias *provideInitialHead() const { return createSentinel(); }
  GlobalAlias *ensureHead(GlobalAlias*) const { return createSentinel(); }
  static void noteHead(GlobalAlias*, GlobalAlias*) {}
private:
  mutable ilist_node<GlobalAlias> Sentinel;
};

template<> struct ilist_traits<NamedMDNode>
  : public ilist_default_traits<NamedMDNode> {
  // createSentinel is used to get hold of a node that marks the end of
  // the list...
  NamedMDNode *createSentinel() const {
    return static_cast<NamedMDNode*>(&Sentinel);
  }
  static void destroySentinel(NamedMDNode*) {}

  NamedMDNode *provideInitialHead() const { return createSentinel(); }
  NamedMDNode *ensureHead(NamedMDNode*) const { return createSentinel(); }
  static void noteHead(NamedMDNode*, NamedMDNode*) {}
  void addNodeToList(NamedMDNode *) {}
  void removeNodeFromList(NamedMDNode *) {}
private:
  mutable ilist_node<NamedMDNode> Sentinel;
};

/// A Module instance is used to store all the information related to an
/// LLVM module. Modules are the top level container of all other LLVM
/// Intermediate Representation (IR) objects. Each module directly contains a
/// list of globals variables, a list of functions, a list of libraries (or
/// other modules) this module depends on, a symbol table, and various data
/// about the target's characteristics.
///
/// A module maintains a GlobalValRefMap object that is used to hold all
/// constant references to global variables in the module.  When a global
/// variable is destroyed, it should have no entries in the GlobalValueRefMap.
/// @brief The main container class for the LLVM Intermediate Representation.
class Module {
/// @name Types And Enumerations
/// @{
public:
  /// The type for the list of global variables.
  typedef iplist<GlobalVariable> GlobalListType;
  /// The type for the list of functions.
  typedef iplist<Function> FunctionListType;
  /// The type for the list of aliases.
  typedef iplist<GlobalAlias> AliasListType;
  /// The type for the list of named metadata.
  typedef ilist<NamedMDNode> NamedMDListType;
  /// The type of the comdat "symbol" table.
  typedef StringMap<Comdat> ComdatSymTabType;

  /// The Global Variable iterator.
  typedef GlobalListType::iterator                      global_iterator;
  /// The Global Variable constant iterator.
  typedef GlobalListType::const_iterator          const_global_iterator;

  /// The Function iterators.
  typedef FunctionListType::iterator                           iterator;
  /// The Function constant iterator
  typedef FunctionListType::const_iterator               const_iterator;

  /// The Function reverse iterator.
  typedef FunctionListType::reverse_iterator             reverse_iterator;
  /// The Function constant reverse iterator.
  typedef FunctionListType::const_reverse_iterator const_reverse_iterator;

  /// The Global Alias iterators.
  typedef AliasListType::iterator                        alias_iterator;
  /// The Global Alias constant iterator
  typedef AliasListType::const_iterator            const_alias_iterator;

  /// The named metadata iterators.
  typedef NamedMDListType::iterator             named_metadata_iterator;
  /// The named metadata constant iterators.
  typedef NamedMDListType::const_iterator const_named_metadata_iterator;

  /// This enumeration defines the supported behaviors of module flags.
  enum ModFlagBehavior {
    /// Emits an error if two values disagree, otherwise the resulting value is
    /// that of the operands.
    Error = 1,

    /// Emits a warning if two values disagree. The result value will be the
    /// operand for the flag from the first module being linked.
    Warning = 2,

    /// Adds a requirement that another module flag be present and have a
    /// specified value after linking is performed. The value must be a metadata
    /// pair, where the first element of the pair is the ID of the module flag
    /// to be restricted, and the second element of the pair is the value the
    /// module flag should be restricted to. This behavior can be used to
    /// restrict the allowable results (via triggering of an error) of linking
    /// IDs with the **Override** behavior.
    Require = 3,

    /// Uses the specified value, regardless of the behavior or value of the
    /// other module. If both modules specify **Override**, but the values
    /// differ, an error will be emitted.
    Override = 4,

    /// Appends the two values, which are required to be metadata nodes.
    Append = 5,

    /// Appends the two values, which are required to be metadata
    /// nodes. However, duplicate entries in the second list are dropped
    /// during the append operation.
    AppendUnique = 6,

    // Markers:
    ModFlagBehaviorFirstVal = Error,
    ModFlagBehaviorLastVal = AppendUnique
  };

  /// Checks if Metadata represents a valid ModFlagBehavior, and stores the
  /// converted result in MFB.
  static bool isValidModFlagBehavior(Metadata *MD, ModFlagBehavior &MFB);

  struct ModuleFlagEntry {
    ModFlagBehavior Behavior;
    MDString *Key;
    Metadata *Val;
    ModuleFlagEntry(ModFlagBehavior B, MDString *K, Metadata *V)
        : Behavior(B), Key(K), Val(V) {}
  };

/// @}
/// @name Member Variables
/// @{
private:
  LLVMContext &Context;           ///< The LLVMContext from which types and
                                  ///< constants are allocated.
  GlobalListType GlobalList;      ///< The Global Variables in the module
  FunctionListType FunctionList;  ///< The Functions in the module
  AliasListType AliasList;        ///< The Aliases in the module
  NamedMDListType NamedMDList;    ///< The named metadata in the module
  std::string GlobalScopeAsm;     ///< Inline Asm at global scope.
  ValueSymbolTable *ValSymTab;    ///< Symbol table for values
  ComdatSymTabType ComdatSymTab;  ///< Symbol table for COMDATs
  std::unique_ptr<GVMaterializer>
  Materializer;                   ///< Used to materialize GlobalValues
  std::string ModuleID;           ///< Human readable identifier for the module
  std::string TargetTriple;       ///< Platform target triple Module compiled on
                                  ///< Format: (arch)(sub)-(vendor)-(sys0-(abi)
  void *NamedMDSymTab;            ///< NamedMDNode names.
  DataLayout DL;                  ///< DataLayout associated with the module

  friend class Constant;

  // HLSL Change start
  hlsl::HLModule *TheHLModule = nullptr;
  hlsl::DxilModule *TheDxilModule = nullptr;
  // HLSL Change end

/// @}
/// @name Constructors
/// @{
public:
  /// The Module constructor. Note that there is no default constructor. You
  /// must provide a name for the module upon construction.
  explicit Module(StringRef ModuleID, LLVMContext& C);
  /// The module destructor. This will dropAllReferences.
  ~Module();

/// @}
/// @name Module Level Accessors
/// @{

  /// Get the module identifier which is, essentially, the name of the module.
  /// @returns the module identifier as a string
  const std::string &getModuleIdentifier() const { return ModuleID; }

  /// \brief Get a short "name" for the module.
  ///
  /// This is useful for debugging or logging. It is essentially a convenience
  /// wrapper around getModuleIdentifier().
  StringRef getName() const { return ModuleID; }

  /// Get the data layout string for the module's target platform. This is
  /// equivalent to getDataLayout()->getStringRepresentation().
  const std::string &getDataLayoutStr() const {
    return DL.getStringRepresentation();
  }

  /// Get the data layout for the module's target platform.
  const DataLayout &getDataLayout() const;

  /// Get the target triple which is a string describing the target host.
  /// @returns a string containing the target triple.
  const std::string &getTargetTriple() const { return TargetTriple; }

  /// Get the global data context.
  /// @returns LLVMContext - a container for LLVM's global information
  LLVMContext &getContext() const { return Context; }

  /// Get any module-scope inline assembly blocks.
  /// @returns a string containing the module-scope inline assembly blocks.
  const std::string &getModuleInlineAsm() const { return GlobalScopeAsm; }

  /// Get a RandomNumberGenerator salted for use with this module. The
  /// RNG can be seeded via -rng-seed=<uint64> and is salted with the
  /// ModuleID and the provided pass salt. The returned RNG should not
  /// be shared across threads or passes.
  ///
  /// A unique RNG per pass ensures a reproducible random stream even
  /// when other randomness consuming passes are added or removed. In
  /// addition, the random stream will be reproducible across LLVM
  /// versions when the pass does not change.
  RandomNumberGenerator *createRNG(const Pass* P) const;

/// @}
/// @name Module Level Mutators
/// @{

  /// Set the module identifier.
  void setModuleIdentifier(StringRef ID) { ModuleID = ID; }

  /// Set the data layout
  void setDataLayout(StringRef Desc);
  void setDataLayout(const DataLayout &Other);

  /// Set the target triple.
  void setTargetTriple(StringRef T) { TargetTriple = T; }

  /// Set the module-scope inline assembly blocks.
  /// A trailing newline is added if the input doesn't have one.
  void setModuleInlineAsm(StringRef Asm) {
    GlobalScopeAsm = Asm;
    if (!GlobalScopeAsm.empty() &&
        GlobalScopeAsm[GlobalScopeAsm.size()-1] != '\n')
      GlobalScopeAsm += '\n';
  }

  /// Append to the module-scope inline assembly blocks.
  /// A trailing newline is added if the input doesn't have one.
  void appendModuleInlineAsm(StringRef Asm) {
    GlobalScopeAsm += Asm;
    if (!GlobalScopeAsm.empty() &&
        GlobalScopeAsm[GlobalScopeAsm.size()-1] != '\n')
      GlobalScopeAsm += '\n';
  }

/// @}
/// @name Generic Value Accessors
/// @{

  /// Return the global value in the module with the specified name, of
  /// arbitrary type. This method returns null if a global with the specified
  /// name is not found.
  GlobalValue *getNamedValue(StringRef Name) const;

  /// Return a unique non-zero ID for the specified metadata kind. This ID is
  /// uniqued across modules in the current LLVMContext.
  unsigned getMDKindID(StringRef Name) const;

  /// Populate client supplied SmallVector with the name for custom metadata IDs
  /// registered in this LLVMContext.
  void getMDKindNames(SmallVectorImpl<StringRef> &Result) const;

  /// Return the type with the specified name, or null if there is none by that
  /// name.
  StructType *getTypeByName(StringRef Name) const;

  std::vector<StructType *> getIdentifiedStructTypes() const;

/// @}
/// @name Function Accessors
/// @{

  /// Look up the specified function in the module symbol table. Four
  /// possibilities:
  ///   1. If it does not exist, add a prototype for the function and return it.
  ///   2. If it exists, and has a local linkage, the existing function is
  ///      renamed and a new one is inserted.
  ///   3. Otherwise, if the existing function has the correct prototype, return
  ///      the existing function.
  ///   4. Finally, the function exists but has the wrong prototype: return the
  ///      function with a constantexpr cast to the right prototype.
  Constant *getOrInsertFunction(StringRef Name, FunctionType *T,
                                AttributeSet AttributeList);

  Constant *getOrInsertFunction(StringRef Name, FunctionType *T);

  /// Look up the specified function in the module symbol table. If it does not
  /// exist, add a prototype for the function and return it. This function
  /// guarantees to return a constant of pointer to the specified function type
  /// or a ConstantExpr BitCast of that type if the named function has a
  /// different type. This version of the method takes a null terminated list of
  /// function arguments, which makes it easier for clients to use.
  Constant *getOrInsertFunction(StringRef Name,
                                AttributeSet AttributeList,
                                Type *RetTy, ...) LLVM_END_WITH_NULL;

  /// Same as above, but without the attributes.
  Constant *getOrInsertFunction(StringRef Name, Type *RetTy, ...)
    LLVM_END_WITH_NULL;

  /// Look up the specified function in the module symbol table. If it does not
  /// exist, return null.
  Function *getFunction(StringRef Name) const;

/// @}
/// @name Global Variable Accessors
/// @{

  /// Look up the specified global variable in the module symbol table. If it
  /// does not exist, return null. If AllowInternal is set to true, this
  /// function will return types that have InternalLinkage. By default, these
  /// types are not returned.
  GlobalVariable *getGlobalVariable(StringRef Name) const {
    return getGlobalVariable(Name, false);
  }

  GlobalVariable *getGlobalVariable(StringRef Name, bool AllowInternal) const {
    return const_cast<Module *>(this)->getGlobalVariable(Name, AllowInternal);
  }

  GlobalVariable *getGlobalVariable(StringRef Name, bool AllowInternal = false);

  /// Return the global variable in the module with the specified name, of
  /// arbitrary type. This method returns null if a global with the specified
  /// name is not found.
  GlobalVariable *getNamedGlobal(StringRef Name) {
    return getGlobalVariable(Name, true);
  }
  const GlobalVariable *getNamedGlobal(StringRef Name) const {
    return const_cast<Module *>(this)->getNamedGlobal(Name);
  }

  /// Look up the specified global in the module symbol table.
  ///   1. If it does not exist, add a declaration of the global and return it.
  ///   2. Else, the global exists but has the wrong type: return the function
  ///      with a constantexpr cast to the right type.
  ///   3. Finally, if the existing global is the correct declaration, return
  ///      the existing global.
  Constant *getOrInsertGlobal(StringRef Name, Type *Ty);

/// @}
/// @name Global Alias Accessors
/// @{

  /// Return the global alias in the module with the specified name, of
  /// arbitrary type. This method returns null if a global with the specified
  /// name is not found.
  GlobalAlias *getNamedAlias(StringRef Name) const;

/// @}
/// @name Named Metadata Accessors
/// @{

  /// Return the first NamedMDNode in the module with the specified name. This
  /// method returns null if a NamedMDNode with the specified name is not found.
  NamedMDNode *getNamedMetadata(const Twine &Name) const;

  /// Return the named MDNode in the module with the specified name. This method
  /// returns a new NamedMDNode if a NamedMDNode with the specified name is not
  /// found.
  NamedMDNode *getOrInsertNamedMetadata(StringRef Name);

  /// Remove the given NamedMDNode from this module and delete it.
  void eraseNamedMetadata(NamedMDNode *NMD);

/// @}
/// @name Comdat Accessors
/// @{

  /// Return the Comdat in the module with the specified name. It is created
  /// if it didn't already exist.
  Comdat *getOrInsertComdat(StringRef Name);

/// @}
/// @name Module Flags Accessors
/// @{

  /// Returns the module flags in the provided vector.
  void getModuleFlagsMetadata(SmallVectorImpl<ModuleFlagEntry> &Flags) const;

  /// Return the corresponding value if Key appears in module flags, otherwise
  /// return null.
  Metadata *getModuleFlag(StringRef Key) const;

  /// Returns the NamedMDNode in the module that represents module-level flags.
  /// This method returns null if there are no module-level flags.
  NamedMDNode *getModuleFlagsMetadata() const;

  /// Returns the NamedMDNode in the module that represents module-level flags.
  /// If module-level flags aren't found, it creates the named metadata that
  /// contains them.
  NamedMDNode *getOrInsertModuleFlagsMetadata();

  /// Add a module-level flag to the module-level flags metadata. It will create
  /// the module-level flags named metadata if it doesn't already exist.
  void addModuleFlag(ModFlagBehavior Behavior, StringRef Key, Metadata *Val);
  void addModuleFlag(ModFlagBehavior Behavior, StringRef Key, Constant *Val);
  void addModuleFlag(ModFlagBehavior Behavior, StringRef Key, uint32_t Val);
  void addModuleFlag(MDNode *Node);

/// @}
/// @name Materialization
/// @{

  /// Sets the GVMaterializer to GVM. This module must not yet have a
  /// Materializer. To reset the materializer for a module that already has one,
  /// call MaterializeAllPermanently first. Destroying this module will destroy
  /// its materializer without materializing any more GlobalValues. Without
  /// destroying the Module, there is no way to detach or destroy a materializer
  /// without materializing all the GVs it controls, to avoid leaving orphan
  /// unmaterialized GVs.
  void setMaterializer(GVMaterializer *GVM);
  /// Retrieves the GVMaterializer, if any, for this Module.
  GVMaterializer *getMaterializer() const { return Materializer.get(); }

  /// Returns true if this GV was loaded from this Module's GVMaterializer and
  /// the GVMaterializer knows how to dematerialize the GV.
  bool isDematerializable(const GlobalValue *GV) const;

  /// Make sure the GlobalValue is fully read. If the module is corrupt, this
  /// returns true and fills in the optional string with information about the
  /// problem. If successful, this returns false.
  std::error_code materialize(GlobalValue *GV);
  /// If the GlobalValue is read in, and if the GVMaterializer supports it,
  /// release the memory for the function, and set it up to be materialized
  /// lazily. If !isDematerializable(), this method is a no-op.
  void dematerialize(GlobalValue *GV);

  /// Make sure all GlobalValues in this Module are fully read.
  std::error_code materializeAll();

  /// Make sure all GlobalValues in this Module are fully read and clear the
  /// Materializer. If the module is corrupt, this DOES NOT clear the old
  /// Materializer.
  std::error_code materializeAllPermanently();

  std::error_code materializeMetadata();
  std::error_code materializeSelectNamedMetadata(ArrayRef<StringRef> NamedMetadata); // HLSL Change

/// @}
/// @name Direct access to the globals list, functions list, and symbol table
/// @{

  /// Get the Module's list of global variables (constant).
  const GlobalListType   &getGlobalList() const       { return GlobalList; }
  /// Get the Module's list of global variables.
  GlobalListType         &getGlobalList()             { return GlobalList; }
  static GlobalListType Module::*getSublistAccess(GlobalVariable*) {
    return &Module::GlobalList;
  }
  /// Get the Module's list of functions (constant).
  const FunctionListType &getFunctionList() const     { return FunctionList; }
  /// Get the Module's list of functions.
  FunctionListType       &getFunctionList()           { return FunctionList; }
  static FunctionListType Module::*getSublistAccess(Function*) {
    return &Module::FunctionList;
  }
  /// Get the Module's list of aliases (constant).
  const AliasListType    &getAliasList() const        { return AliasList; }
  /// Get the Module's list of aliases.
  AliasListType          &getAliasList()              { return AliasList; }
  static AliasListType Module::*getSublistAccess(GlobalAlias*) {
    return &Module::AliasList;
  }
  /// Get the Module's list of named metadata (constant).
  const NamedMDListType  &getNamedMDList() const      { return NamedMDList; }
  /// Get the Module's list of named metadata.
  NamedMDListType        &getNamedMDList()            { return NamedMDList; }
  static NamedMDListType Module::*getSublistAccess(NamedMDNode*) {
    return &Module::NamedMDList;
  }
  /// Get the symbol table of global variable and function identifiers
  const ValueSymbolTable &getValueSymbolTable() const { return *ValSymTab; }
  /// Get the Module's symbol table of global variable and function identifiers.
  ValueSymbolTable       &getValueSymbolTable()       { return *ValSymTab; }
  /// Get the Module's symbol table for COMDATs (constant).
  const ComdatSymTabType &getComdatSymbolTable() const { return ComdatSymTab; }
  /// Get the Module's symbol table for COMDATs.
  ComdatSymTabType &getComdatSymbolTable() { return ComdatSymTab; }

/// @}
/// @name Global Variable Iteration
/// @{

  global_iterator       global_begin()       { return GlobalList.begin(); }
  const_global_iterator global_begin() const { return GlobalList.begin(); }
  global_iterator       global_end  ()       { return GlobalList.end(); }
  const_global_iterator global_end  () const { return GlobalList.end(); }
  bool                  global_empty() const { return GlobalList.empty(); }

  iterator_range<global_iterator> globals() {
    return iterator_range<global_iterator>(global_begin(), global_end());
  }
  iterator_range<const_global_iterator> globals() const {
    return iterator_range<const_global_iterator>(global_begin(), global_end());
  }

/// @}
/// @name Function Iteration
/// @{

  iterator                begin()       { return FunctionList.begin(); }
  const_iterator          begin() const { return FunctionList.begin(); }
  iterator                end  ()       { return FunctionList.end();   }
  const_iterator          end  () const { return FunctionList.end();   }
  reverse_iterator        rbegin()      { return FunctionList.rbegin(); }
  const_reverse_iterator  rbegin() const{ return FunctionList.rbegin(); }
  reverse_iterator        rend()        { return FunctionList.rend(); }
  const_reverse_iterator  rend() const  { return FunctionList.rend(); }
  size_t                  size() const  { return FunctionList.size(); }
  bool                    empty() const { return FunctionList.empty(); }

  iterator_range<iterator> functions() {
    return iterator_range<iterator>(begin(), end());
  }
  iterator_range<const_iterator> functions() const {
    return iterator_range<const_iterator>(begin(), end());
  }

/// @}
/// @name Alias Iteration
/// @{

  alias_iterator       alias_begin()            { return AliasList.begin(); }
  const_alias_iterator alias_begin() const      { return AliasList.begin(); }
  alias_iterator       alias_end  ()            { return AliasList.end();   }
  const_alias_iterator alias_end  () const      { return AliasList.end();   }
  size_t               alias_size () const      { return AliasList.size();  }
  bool                 alias_empty() const      { return AliasList.empty(); }

  iterator_range<alias_iterator> aliases() {
    return iterator_range<alias_iterator>(alias_begin(), alias_end());
  }
  iterator_range<const_alias_iterator> aliases() const {
    return iterator_range<const_alias_iterator>(alias_begin(), alias_end());
  }

/// @}
/// @name Named Metadata Iteration
/// @{

  named_metadata_iterator named_metadata_begin() { return NamedMDList.begin(); }
  const_named_metadata_iterator named_metadata_begin() const {
    return NamedMDList.begin();
  }

  named_metadata_iterator named_metadata_end() { return NamedMDList.end(); }
  const_named_metadata_iterator named_metadata_end() const {
    return NamedMDList.end();
  }

  size_t named_metadata_size() const { return NamedMDList.size();  }
  bool named_metadata_empty() const { return NamedMDList.empty(); }

  iterator_range<named_metadata_iterator> named_metadata() {
    return iterator_range<named_metadata_iterator>(named_metadata_begin(),
                                                   named_metadata_end());
  }
  iterator_range<const_named_metadata_iterator> named_metadata() const {
    return iterator_range<const_named_metadata_iterator>(named_metadata_begin(),
                                                         named_metadata_end());
  }

  /// Destroy ConstantArrays in LLVMContext if they are not used.
  /// ConstantArrays constructed during linking can cause quadratic memory
  /// explosion. Releasing all unused constants can cause a 20% LTO compile-time
  /// slowdown for a large application.
  ///
  /// NOTE: Constants are currently owned by LLVMContext. This can then only
  /// be called where all uses of the LLVMContext are understood.
  void dropTriviallyDeadConstantArrays();

/// @}
/// @name Utility functions for printing and dumping Module objects
/// @{

  /// Print the module to an output stream with an optional
  /// AssemblyAnnotationWriter.  If \c ShouldPreserveUseListOrder, then include
  /// uselistorder directives so that use-lists can be recreated when reading
  /// the assembly.
  void print(raw_ostream &OS, AssemblyAnnotationWriter *AAW,
             bool ShouldPreserveUseListOrder = false) const;

  /// Dump the module to stderr (for debugging).
  void dump() const;
  
  /// This function causes all the subinstructions to "let go" of all references
  /// that they are maintaining.  This allows one to 'delete' a whole class at
  /// a time, even though there may be circular references... first all
  /// references are dropped, and all use counts go to zero.  Then everything
  /// is delete'd for real.  Note that no operations are valid on an object
  /// that has "dropped all references", except operator delete.
  void dropAllReferences();

/// @}
/// @name Utility functions for querying Debug information.
/// @{

  /// \brief Returns the Dwarf Version by checking module flags.
  unsigned getDwarfVersion() const;

/// @}
/// @name Utility functions for querying and setting PIC level
/// @{

  /// \brief Returns the PIC level (small or large model)
  PICLevel::Level getPICLevel() const;

  /// \brief Set the PIC level (small or large model)
  void setPICLevel(PICLevel::Level PL);
/// @}

  // HLSL Change start
  typedef void (*RemoveGlobalCallback)(llvm::Module*, llvm::GlobalObject*);
  typedef void(*ResetModuleCallback)(llvm::Module*);
  RemoveGlobalCallback pfnRemoveGlobal = nullptr;
  void CallRemoveGlobalHook(llvm::GlobalObject* G) {
    if (pfnRemoveGlobal) (*pfnRemoveGlobal)(this, G);
  }
  bool HasHLModule() const { return TheHLModule != nullptr; }
  void SetHLModule(hlsl::HLModule *pValue) { TheHLModule = pValue; }
  hlsl::HLModule &GetHLModule() const { return *TheHLModule; }
  hlsl::HLModule &GetOrCreateHLModule(bool skipInit = false);
  ResetModuleCallback pfnResetHLModule = nullptr;
  void ResetHLModule() { if (pfnResetHLModule) (*pfnResetHLModule)(this); }
  bool HasDxilModule() const { return TheDxilModule != nullptr; }
  void SetDxilModule(hlsl::DxilModule *pValue) { TheDxilModule = pValue; }
  hlsl::DxilModule &GetDxilModule() const { return *TheDxilModule; }
  hlsl::DxilModule &GetOrCreateDxilModule(bool skipInit = false);
  ResetModuleCallback pfnResetDxilModule = nullptr;
  void ResetDxilModule() { if (pfnResetDxilModule) (*pfnResetDxilModule)(this); }
  // HLSL Change end
};

/// An raw_ostream inserter for modules.
inline raw_ostream &operator<<(raw_ostream &O, const Module &M) {
  M.print(O, nullptr);
  return O;
}

// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(Module, LLVMModuleRef)

/* LLVMModuleProviderRef exists for historical reasons, but now just holds a
 * Module.
 */
inline Module *unwrap(LLVMModuleProviderRef MP) {
  return reinterpret_cast<Module*>(MP);
}
  
} // End llvm namespace

#endif
