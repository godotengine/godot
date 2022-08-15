//===- llvm/Pass.h - Base class for Passes ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a base class that indicates that a specified class is a
// transformation pass implementation.
//
// Passes are designed this way so that it is possible to run passes in a cache
// and organizationally optimal order without having to specify it at the front
// end.  This allows arbitrary passes to be strung together and have them
// executed as efficiently as possible.
//
// Passes should extend one of the classes below, depending on the guarantees
// that it can make about what will be modified as it is run.  For example, most
// global optimizations should derive from FunctionPass, because they do not add
// or delete functions, they operate on the internals of the function.
//
// Note that this file #includes PassSupport.h and PassAnalysisSupport.h (at the
// bottom), so the APIs exposed by these files are also automatically available
// to all users of this file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASS_H
#define LLVM_PASS_H

#include "llvm/Support/Compiler.h"
#include <string>
#include "llvm/ADT/ArrayRef.h" // HLSL Change
#include "llvm/ADT/StringRef.h" // HLSL Change

namespace llvm {

class BasicBlock;
class Function;
class Module;
class AnalysisUsage;
class PassInfo;
class ImmutablePass;
class PMStack;
class AnalysisResolver;
class PMDataManager;
class raw_ostream;
class StringRef;

// AnalysisID - Use the PassInfo to identify a pass...
typedef const void* AnalysisID;

/// Different types of internal pass managers. External pass managers
/// (PassManager and FunctionPassManager) are not represented here.
/// Ordering of pass manager types is important here.
enum PassManagerType {
  PMT_Unknown = 0,
  PMT_ModulePassManager = 1, ///< MPPassManager
  PMT_CallGraphPassManager,  ///< CGPassManager
  PMT_FunctionPassManager,   ///< FPPassManager
  PMT_LoopPassManager,       ///< LPPassManager
  PMT_RegionPassManager,     ///< RGPassManager
  PMT_BasicBlockPassManager, ///< BBPassManager
  PMT_Last
};

// Different types of passes.
enum PassKind {
  PT_BasicBlock,
  PT_Region,
  PT_Loop,
  PT_Function,
  PT_CallGraphSCC,
  PT_Module,
  PT_PassManager
};

// HLSL Change Starts
//
// LLVM passes typically support an empty constructor, but often they will
// have in addition to that constructors with arguments, as well as global
// options that can modify their behavior.
//
// This section allows a data-driven pipeline assembler to be built.
// The DxcOptimizer holds much of the needed information, but passes
// frequently are implemented in non-header files and as such this
// becomes a reasonable mechanism to pass flags around.
//
// Pass implementations should use the GetPassOption* functions to read
// arguments, setting fields in place of globals and updating state
// for constructor arguments.
//
typedef std::pair<llvm::StringRef, llvm::StringRef> PassOption;
typedef llvm::ArrayRef<PassOption> PassOptions;
struct PassOptionsCompare {
  bool operator()(const PassOption &a, const PassOption &b) {
    return a.first < b.first;
  }
};
bool GetPassOption(PassOptions &O, llvm::StringRef name, llvm::StringRef *pValue);
bool GetPassOptionBool(PassOptions &O, llvm::StringRef name, bool *pValue, bool defaultValue);
bool GetPassOptionUnsigned(PassOptions &O, llvm::StringRef name, unsigned *pValue, unsigned defaultValue);
bool GetPassOptionInt(PassOptions &O, llvm::StringRef name, int *pValue, int defaultValue);
bool GetPassOptionUInt32(PassOptions &O, llvm::StringRef name, uint32_t *pValue, uint32_t defaultValue);
bool GetPassOptionUInt64(PassOptions &O, llvm::StringRef name, uint64_t *pValue, uint64_t defaultValue);
bool GetPassOptionFloat(PassOptions &O, llvm::StringRef name, float *pValue, float defaultValue);
// HLSL Change Ends

//===----------------------------------------------------------------------===//
/// Pass interface - Implemented by all 'passes'.  Subclass this if you are an
/// interprocedural optimization or you do not fit into any of the more
/// constrained passes described below.
///
class Pass {
  AnalysisResolver *Resolver;  // Used to resolve analysis
  const void *PassID;
  PassKind Kind;
  void operator=(const Pass&) = delete;
  Pass(const Pass &) = delete;

protected: raw_ostream *OSOverride; // HLSL Change

public:
  explicit Pass(PassKind K, char &pid)
    : Resolver(nullptr), PassID(&pid), Kind(K), OSOverride(nullptr) { } // HLSL Change
  virtual ~Pass();

  void setOSOverride(raw_ostream *OS) { OSOverride = OS; } // HLSL Change
  virtual void applyOptions(PassOptions O) { return; } // HLSL Change
  virtual void dumpConfig(raw_ostream &OS); // HLSL Change

  PassKind getPassKind() const { return Kind; }

  /// getPassName - Return a nice clean name for a pass.  This usually
  /// implemented in terms of the name that is registered by one of the
  /// Registration templates, but can be overloaded directly.
  ///
  virtual StringRef getPassName() const;
  const char *getPassArgument() const; // HLSL Change

  /// getPassID - Return the PassID number that corresponds to this pass.
  AnalysisID getPassID() const {
    return PassID;
  }

  /// doInitialization - Virtual method overridden by subclasses to do
  /// any necessary initialization before any pass is run.
  ///
  virtual bool doInitialization(Module &)  { return false; }

  /// doFinalization - Virtual method overriden by subclasses to do any
  /// necessary clean up after all passes have run.
  ///
  virtual bool doFinalization(Module &) { return false; }

  /// print - Print out the internal state of the pass.  This is called by
  /// Analyze to print out the contents of an analysis.  Otherwise it is not
  /// necessary to implement this method.  Beware that the module pointer MAY be
  /// null.  This automatically forwards to a virtual function that does not
  /// provide the Module* in case the analysis doesn't need it it can just be
  /// ignored.
  ///
  virtual void print(raw_ostream &O, const Module *M) const;
  void dump() const; // dump - Print to stderr.

  /// createPrinterPass - Get a Pass appropriate to print the IR this
  /// pass operates on (Module, Function or MachineFunction).
  virtual Pass *createPrinterPass(raw_ostream &O,
                                  const std::string &Banner) const = 0;

  /// Each pass is responsible for assigning a pass manager to itself.
  /// PMS is the stack of available pass manager.
  virtual void assignPassManager(PMStack &,
                                 PassManagerType) {}
  /// Check if available pass managers are suitable for this pass or not.
  virtual void preparePassManager(PMStack &);

  ///  Return what kind of Pass Manager can manage this pass.
  virtual PassManagerType getPotentialPassManagerType() const;

  // Access AnalysisResolver
  void setResolver(AnalysisResolver *AR);
  AnalysisResolver *getResolver() const { return Resolver; }

  /// getAnalysisUsage - This function should be overriden by passes that need
  /// analysis information to do their job.  If a pass specifies that it uses a
  /// particular analysis result to this function, it can then use the
  /// getAnalysis<AnalysisType>() function, below.
  ///
  virtual void getAnalysisUsage(AnalysisUsage &) const;

  /// releaseMemory() - This member can be implemented by a pass if it wants to
  /// be able to release its memory when it is no longer needed.  The default
  /// behavior of passes is to hold onto memory for the entire duration of their
  /// lifetime (which is the entire compile time).  For pipelined passes, this
  /// is not a big deal because that memory gets recycled every time the pass is
  /// invoked on another program unit.  For IP passes, it is more important to
  /// free memory when it is unused.
  ///
  /// Optionally implement this function to release pass memory when it is no
  /// longer used.
  ///
  virtual void releaseMemory();

  /// getAdjustedAnalysisPointer - This method is used when a pass implements
  /// an analysis interface through multiple inheritance.  If needed, it should
  /// override this to adjust the this pointer as needed for the specified pass
  /// info.
  virtual void *getAdjustedAnalysisPointer(AnalysisID ID);
  virtual ImmutablePass *getAsImmutablePass();
  virtual PMDataManager *getAsPMDataManager();

  /// verifyAnalysis() - This member can be implemented by a analysis pass to
  /// check state of analysis information.
  virtual void verifyAnalysis() const;

  // dumpPassStructure - Implement the -debug-passes=PassStructure option
  virtual void dumpPassStructure(unsigned Offset = 0);

  // lookupPassInfo - Return the pass info object for the specified pass class,
  // or null if it is not known.
  static const PassInfo *lookupPassInfo(const void *TI);

  // lookupPassInfo - Return the pass info object for the pass with the given
  // argument string, or null if it is not known.
  static const PassInfo *lookupPassInfo(StringRef Arg);

  // createPass - Create a object for the specified pass class,
  // or null if it is not known.
  static Pass *createPass(AnalysisID ID);

  /// getAnalysisIfAvailable<AnalysisType>() - Subclasses use this function to
  /// get analysis information that might be around, for example to update it.
  /// This is different than getAnalysis in that it can fail (if the analysis
  /// results haven't been computed), so should only be used if you can handle
  /// the case when the analysis is not available.  This method is often used by
  /// transformation APIs to update analysis results for a pass automatically as
  /// the transform is performed.
  ///
  template<typename AnalysisType> AnalysisType *
    getAnalysisIfAvailable() const; // Defined in PassAnalysisSupport.h

  /// mustPreserveAnalysisID - This method serves the same function as
  /// getAnalysisIfAvailable, but works if you just have an AnalysisID.  This
  /// obviously cannot give you a properly typed instance of the class if you
  /// don't have the class name available (use getAnalysisIfAvailable if you
  /// do), but it can tell you if you need to preserve the pass at least.
  ///
  bool mustPreserveAnalysisID(char &AID) const;

  /// getAnalysis<AnalysisType>() - This function is used by subclasses to get
  /// to the analysis information that they claim to use by overriding the
  /// getAnalysisUsage function.
  ///
  template<typename AnalysisType>
  AnalysisType &getAnalysis() const; // Defined in PassAnalysisSupport.h

  template<typename AnalysisType>
  AnalysisType &getAnalysis(Function &F); // Defined in PassAnalysisSupport.h

  template<typename AnalysisType>
  AnalysisType &getAnalysisID(AnalysisID PI) const;

  template<typename AnalysisType>
  AnalysisType &getAnalysisID(AnalysisID PI, Function &F);
};


//===----------------------------------------------------------------------===//
/// ModulePass class - This class is used to implement unstructured
/// interprocedural optimizations and analyses.  ModulePasses may do anything
/// they want to the program.
///
class ModulePass : public Pass {
public:
  /// createPrinterPass - Get a module printer pass.
  Pass *createPrinterPass(raw_ostream &O,
                          const std::string &Banner) const override;

  /// runOnModule - Virtual method overriden by subclasses to process the module
  /// being operated on.
  virtual bool runOnModule(Module &M) = 0;

  void assignPassManager(PMStack &PMS, PassManagerType T) override;

  ///  Return what kind of Pass Manager can manage this pass.
  PassManagerType getPotentialPassManagerType() const override;

  explicit ModulePass(char &pid) : Pass(PT_Module, pid) {}
  // Force out-of-line virtual method.
  ~ModulePass() override;
};


//===----------------------------------------------------------------------===//
/// ImmutablePass class - This class is used to provide information that does
/// not need to be run.  This is useful for things like target information and
/// "basic" versions of AnalysisGroups.
///
class ImmutablePass : public ModulePass {
public:
  /// initializePass - This method may be overriden by immutable passes to allow
  /// them to perform various initialization actions they require.  This is
  /// primarily because an ImmutablePass can "require" another ImmutablePass,
  /// and if it does, the overloaded version of initializePass may get access to
  /// these passes with getAnalysis<>.
  ///
  virtual void initializePass();

  ImmutablePass *getAsImmutablePass() override { return this; }

  /// ImmutablePasses are never run.
  ///
  bool runOnModule(Module &) override { return false; }

  explicit ImmutablePass(char &pid)
  : ModulePass(pid) {}

  // Force out-of-line virtual method.
  ~ImmutablePass() override;
};

//===----------------------------------------------------------------------===//
/// FunctionPass class - This class is used to implement most global
/// optimizations.  Optimizations should subclass this class if they meet the
/// following constraints:
///
///  1. Optimizations are organized globally, i.e., a function at a time
///  2. Optimizing a function does not cause the addition or removal of any
///     functions in the module
///
class FunctionPass : public Pass {
public:
  explicit FunctionPass(char &pid) : Pass(PT_Function, pid) {}

  /// createPrinterPass - Get a function printer pass.
  Pass *createPrinterPass(raw_ostream &O,
                          const std::string &Banner) const override;

  /// runOnFunction - Virtual method overriden by subclasses to do the
  /// per-function processing of the pass.
  ///
  virtual bool runOnFunction(Function &F) = 0;

  void assignPassManager(PMStack &PMS, PassManagerType T) override;

  ///  Return what kind of Pass Manager can manage this pass.
  PassManagerType getPotentialPassManagerType() const override;

protected:
  /// skipOptnoneFunction - This function has Attribute::OptimizeNone
  /// and most transformation passes should skip it.
  bool skipOptnoneFunction(const Function &F) const;
};
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
/// BasicBlockPass class - This class is used to implement most local
/// optimizations.  Optimizations should subclass this class if they
/// meet the following constraints:
///   1. Optimizations are local, operating on either a basic block or
///      instruction at a time.
///   2. Optimizations do not modify the CFG of the contained function, or any
///      other basic block in the function.
///   3. Optimizations conform to all of the constraints of FunctionPasses.
///
class BasicBlockPass : public Pass {
public:
  explicit BasicBlockPass(char &pid) : Pass(PT_BasicBlock, pid) {}

  /// createPrinterPass - Get a basic block printer pass.
  Pass *createPrinterPass(raw_ostream &O,
                          const std::string &Banner) const override;

  using llvm::Pass::doInitialization;
  using llvm::Pass::doFinalization;

  /// doInitialization - Virtual method overridden by BasicBlockPass subclasses
  /// to do any necessary per-function initialization.
  ///
  virtual bool doInitialization(Function &);

  /// runOnBasicBlock - Virtual method overriden by subclasses to do the
  /// per-basicblock processing of the pass.
  ///
  virtual bool runOnBasicBlock(BasicBlock &BB) = 0;

  /// doFinalization - Virtual method overriden by BasicBlockPass subclasses to
  /// do any post processing needed after all passes have run.
  ///
  virtual bool doFinalization(Function &);

  void assignPassManager(PMStack &PMS, PassManagerType T) override;

  ///  Return what kind of Pass Manager can manage this pass.
  PassManagerType getPotentialPassManagerType() const override;

protected:
  /// skipOptnoneFunction - Containing function has Attribute::OptimizeNone
  /// and most transformation passes should skip it.
  bool skipOptnoneFunction(const BasicBlock &BB) const;
};

/// If the user specifies the -time-passes argument on an LLVM tool command line
/// then the value of this boolean will be true, otherwise false.
/// @brief This is the storage for the -time-passes option.
extern bool TimePassesIsEnabled;

} // End llvm namespace

// Include support files that contain important APIs commonly used by Passes,
// but that we want to separate out to make it easier to read the header files.
//
#include "llvm/PassSupport.h"
#include "llvm/PassAnalysisSupport.h"

#endif
