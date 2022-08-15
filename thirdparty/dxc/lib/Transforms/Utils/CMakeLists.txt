add_llvm_library(LLVMTransformUtils
  ASanStackFrameLayout.cpp
  AddDiscriminators.cpp
  BasicBlockUtils.cpp
  BreakCriticalEdges.cpp
  BuildLibCalls.cpp
  BypassSlowDivision.cpp
  CloneFunction.cpp
  CloneModule.cpp
  CmpInstAnalysis.cpp
  CodeExtractor.cpp
  CtorUtils.cpp
  DemoteRegToStack.cpp
  FlattenCFG.cpp
  GlobalStatus.cpp
  InlineFunction.cpp
  InstructionNamer.cpp
  IntegerDivision.cpp
  LCSSA.cpp
  Local.cpp
  LoopSimplify.cpp
  LoopSimplifyId.cpp
  LoopUnroll.cpp
  LoopUnrollRuntime.cpp
  LoopUtils.cpp
  LoopVersioning.cpp
  LowerInvoke.cpp
  LowerSwitch.cpp
  Mem2Reg.cpp
  MetaRenamer.cpp
  ModuleUtils.cpp
  PromoteMemoryToRegister.cpp
  SSAUpdater.cpp
  SimplifyCFG.cpp
  SimplifyIndVar.cpp
  SimplifyInstructions.cpp
  SimplifyLibCalls.cpp
  SymbolRewriter.cpp
  UnifyFunctionExitNodes.cpp
  Utils.cpp
  ValueMapper.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/Utils
  )

add_dependencies(LLVMTransformUtils intrinsics_gen)

target_link_libraries(LLVMTransformUtils PUBLIC LLVMDXIL) # HLSL Change