//===-- IndirectionUtils.h - Utilities for adding indirections --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Contains utilities for adding indirections and breaking up modules.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_INDIRECTIONUTILS_H
#define LLVM_EXECUTIONENGINE_ORC_INDIRECTIONUTILS_H

#include "JITSymbol.h"
#include "LambdaResolver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <sstream>

namespace llvm {
namespace orc {

/// @brief Base class for JITLayer independent aspects of
///        JITCompileCallbackManager.
class JITCompileCallbackManagerBase {
public:

  typedef std::function<TargetAddress()> CompileFtor;

  /// @brief Handle to a newly created compile callback. Can be used to get an
  ///        IR constant representing the address of the trampoline, and to set
  ///        the compile action for the callback.
  class CompileCallbackInfo {
  public:
    CompileCallbackInfo(TargetAddress Addr, CompileFtor &Compile)
      : Addr(Addr), Compile(Compile) {}

    TargetAddress getAddress() const { return Addr; }
    void setCompileAction(CompileFtor Compile) {
      this->Compile = std::move(Compile);
    }
  private:
    TargetAddress Addr;
    CompileFtor &Compile;
  };

  /// @brief Construct a JITCompileCallbackManagerBase.
  /// @param ErrorHandlerAddress The address of an error handler in the target
  ///                            process to be used if a compile callback fails.
  /// @param NumTrampolinesPerBlock Number of trampolines to emit if there is no
  ///                             available trampoline when getCompileCallback is
  ///                             called.
  JITCompileCallbackManagerBase(TargetAddress ErrorHandlerAddress,
                                unsigned NumTrampolinesPerBlock)
    : ErrorHandlerAddress(ErrorHandlerAddress),
      NumTrampolinesPerBlock(NumTrampolinesPerBlock) {}

  virtual ~JITCompileCallbackManagerBase() {}

  /// @brief Execute the callback for the given trampoline id. Called by the JIT
  ///        to compile functions on demand.
  TargetAddress executeCompileCallback(TargetAddress TrampolineAddr) {
    auto I = ActiveTrampolines.find(TrampolineAddr);
    // FIXME: Also raise an error in the Orc error-handler when we finally have
    //        one.
    if (I == ActiveTrampolines.end())
      return ErrorHandlerAddress;

    // Found a callback handler. Yank this trampoline out of the active list and
    // put it back in the available trampolines list, then try to run the
    // handler's compile and update actions.
    // Moving the trampoline ID back to the available list first means there's at
    // least one available trampoline if the compile action triggers a request for
    // a new one.
    auto Compile = std::move(I->second);
    ActiveTrampolines.erase(I);
    AvailableTrampolines.push_back(TrampolineAddr);

    if (auto Addr = Compile())
      return Addr;

    return ErrorHandlerAddress;
  }

  /// @brief Reserve a compile callback.
  virtual CompileCallbackInfo getCompileCallback(LLVMContext &Context) = 0;

  /// @brief Get a CompileCallbackInfo for an existing callback.
  CompileCallbackInfo getCompileCallbackInfo(TargetAddress TrampolineAddr) {
    auto I = ActiveTrampolines.find(TrampolineAddr);
    assert(I != ActiveTrampolines.end() && "Not an active trampoline.");
    return CompileCallbackInfo(I->first, I->second);
  }

  /// @brief Release a compile callback.
  ///
  ///   Note: Callbacks are auto-released after they execute. This method should
  /// only be called to manually release a callback that is not going to
  /// execute.
  void releaseCompileCallback(TargetAddress TrampolineAddr) {
    auto I = ActiveTrampolines.find(TrampolineAddr);
    assert(I != ActiveTrampolines.end() && "Not an active trampoline.");
    ActiveTrampolines.erase(I);
    AvailableTrampolines.push_back(TrampolineAddr);
  }

protected:
  TargetAddress ErrorHandlerAddress;
  unsigned NumTrampolinesPerBlock;

  typedef std::map<TargetAddress, CompileFtor> TrampolineMapT;
  TrampolineMapT ActiveTrampolines;
  std::vector<TargetAddress> AvailableTrampolines;
};

/// @brief Manage compile callbacks.
template <typename JITLayerT, typename TargetT>
class JITCompileCallbackManager : public JITCompileCallbackManagerBase {
public:

  /// @brief Construct a JITCompileCallbackManager.
  /// @param JIT JIT layer to emit callback trampolines, etc. into.
  /// @param Context LLVMContext to use for trampoline & resolve block modules.
  /// @param ErrorHandlerAddress The address of an error handler in the target
  ///                            process to be used if a compile callback fails.
  /// @param NumTrampolinesPerBlock Number of trampolines to allocate whenever
  ///                               there is no existing callback trampoline.
  ///                               (Trampolines are allocated in blocks for
  ///                               efficiency.)
  JITCompileCallbackManager(JITLayerT &JIT, RuntimeDyld::MemoryManager &MemMgr,
                            LLVMContext &Context,
                            TargetAddress ErrorHandlerAddress,
                            unsigned NumTrampolinesPerBlock)
    : JITCompileCallbackManagerBase(ErrorHandlerAddress,
                                    NumTrampolinesPerBlock),
      JIT(JIT), MemMgr(MemMgr) {
    emitResolverBlock(Context);
  }

  /// @brief Get/create a compile callback with the given signature.
  CompileCallbackInfo getCompileCallback(LLVMContext &Context) final {
    TargetAddress TrampolineAddr = getAvailableTrampolineAddr(Context);
    auto &Compile = this->ActiveTrampolines[TrampolineAddr];
    return CompileCallbackInfo(TrampolineAddr, Compile);
  }

private:

  std::vector<std::unique_ptr<Module>>
  SingletonSet(std::unique_ptr<Module> M) {
    std::vector<std::unique_ptr<Module>> Ms;
    Ms.push_back(std::move(M));
    return Ms;
  }

  void emitResolverBlock(LLVMContext &Context) {
    std::unique_ptr<Module> M(new Module("resolver_block_module",
                                         Context));
    TargetT::insertResolverBlock(*M, *this);
    auto NonResolver =
      createLambdaResolver(
          [](const std::string &Name) -> RuntimeDyld::SymbolInfo {
            llvm_unreachable("External symbols in resolver block?");
          },
          [](const std::string &Name) -> RuntimeDyld::SymbolInfo {
            llvm_unreachable("Dylib symbols in resolver block?");
          });
    auto H = JIT.addModuleSet(SingletonSet(std::move(M)), &MemMgr,
                              std::move(NonResolver));
    JIT.emitAndFinalize(H);
    auto ResolverBlockSymbol =
      JIT.findSymbolIn(H, TargetT::ResolverBlockName, false);
    assert(ResolverBlockSymbol && "Failed to insert resolver block");
    ResolverBlockAddr = ResolverBlockSymbol.getAddress();
  }

  TargetAddress getAvailableTrampolineAddr(LLVMContext &Context) {
    if (this->AvailableTrampolines.empty())
      grow(Context);
    assert(!this->AvailableTrampolines.empty() &&
           "Failed to grow available trampolines.");
    TargetAddress TrampolineAddr = this->AvailableTrampolines.back();
    this->AvailableTrampolines.pop_back();
    return TrampolineAddr;
  }

  void grow(LLVMContext &Context) {
    assert(this->AvailableTrampolines.empty() && "Growing prematurely?");
    std::unique_ptr<Module> M(new Module("trampoline_block", Context));
    auto GetLabelName =
      TargetT::insertCompileCallbackTrampolines(*M, ResolverBlockAddr,
                                                this->NumTrampolinesPerBlock,
                                                this->ActiveTrampolines.size());
    auto NonResolver =
      createLambdaResolver(
          [](const std::string &Name) -> RuntimeDyld::SymbolInfo {
            llvm_unreachable("External symbols in trampoline block?");
          },
          [](const std::string &Name) -> RuntimeDyld::SymbolInfo {
            llvm_unreachable("Dylib symbols in trampoline block?");
          });
    auto H = JIT.addModuleSet(SingletonSet(std::move(M)), &MemMgr,
                              std::move(NonResolver));
    JIT.emitAndFinalize(H);
    for (unsigned I = 0; I < this->NumTrampolinesPerBlock; ++I) {
      std::string Name = GetLabelName(I);
      auto TrampolineSymbol = JIT.findSymbolIn(H, Name, false);
      assert(TrampolineSymbol && "Failed to emit trampoline.");
      this->AvailableTrampolines.push_back(TrampolineSymbol.getAddress());
    }
  }

  JITLayerT &JIT;
  RuntimeDyld::MemoryManager &MemMgr;
  TargetAddress ResolverBlockAddr;
};

/// @brief Build a function pointer of FunctionType with the given constant
///        address.
///
///   Usage example: Turn a trampoline address into a function pointer constant
/// for use in a stub.
Constant* createIRTypedAddress(FunctionType &FT, TargetAddress Addr);

/// @brief Create a function pointer with the given type, name, and initializer
///        in the given Module.
GlobalVariable* createImplPointer(PointerType &PT, Module &M,
                                  const Twine &Name, Constant *Initializer);

/// @brief Turn a function declaration into a stub function that makes an
///        indirect call using the given function pointer.
void makeStub(Function &F, GlobalVariable &ImplPointer);

/// @brief Raise linkage types and rename as necessary to ensure that all
///        symbols are accessible for other modules.
///
///   This should be called before partitioning a module to ensure that the
/// partitions retain access to each other's symbols.
void makeAllSymbolsExternallyAccessible(Module &M);

/// @brief Clone a function declaration into a new module.
///
///   This function can be used as the first step towards creating a callback
/// stub (see makeStub), or moving a function body (see moveFunctionBody).
///
///   If the VMap argument is non-null, a mapping will be added between F and
/// the new declaration, and between each of F's arguments and the new
/// declaration's arguments. This map can then be passed in to moveFunction to
/// move the function body if required. Note: When moving functions between
/// modules with these utilities, all decls should be cloned (and added to a
/// single VMap) before any bodies are moved. This will ensure that references
/// between functions all refer to the versions in the new module.
Function* cloneFunctionDecl(Module &Dst, const Function &F,
                            ValueToValueMapTy *VMap = nullptr);

/// @brief Move the body of function 'F' to a cloned function declaration in a
///        different module (See related cloneFunctionDecl).
///
///   If the target function declaration is not supplied via the NewF parameter
/// then it will be looked up via the VMap.
///
///   This will delete the body of function 'F' from its original parent module,
/// but leave its declaration.
void moveFunctionBody(Function &OrigF, ValueToValueMapTy &VMap,
                      ValueMaterializer *Materializer = nullptr,
                      Function *NewF = nullptr);

/// @brief Clone a global variable declaration into a new module.
GlobalVariable* cloneGlobalVariableDecl(Module &Dst, const GlobalVariable &GV,
                                        ValueToValueMapTy *VMap = nullptr);

/// @brief Move global variable GV from its parent module to cloned global
///        declaration in a different module.
///
///   If the target global declaration is not supplied via the NewGV parameter
/// then it will be looked up via the VMap.
///
///   This will delete the initializer of GV from its original parent module,
/// but leave its declaration.
void moveGlobalVariableInitializer(GlobalVariable &OrigGV,
                                   ValueToValueMapTy &VMap,
                                   ValueMaterializer *Materializer = nullptr,
                                   GlobalVariable *NewGV = nullptr);

} // End namespace orc.
} // End namespace llvm.

#endif // LLVM_EXECUTIONENGINE_ORC_INDIRECTIONUTILS_H
