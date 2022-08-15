add_llvm_library(LLVMObjCARCOpts
  ObjCARC.cpp
  ObjCARCOpts.cpp
  ObjCARCExpand.cpp
  ObjCARCAPElim.cpp
  ObjCARCAliasAnalysis.cpp
  ARCInstKind.cpp
  ObjCARCContract.cpp
  DependencyAnalysis.cpp
  ProvenanceAnalysis.cpp
  ProvenanceAnalysisEvaluator.cpp
  PtrState.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  )

add_dependencies(LLVMObjCARCOpts intrinsics_gen)
