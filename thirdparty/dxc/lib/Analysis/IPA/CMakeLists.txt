add_llvm_library(LLVMipa
  CallGraph.cpp
  CallGraphSCCPass.cpp
  CallPrinter.cpp
  GlobalsModRef.cpp
  IPA.cpp
  InlineCost.cpp
  )

add_dependencies(LLVMipa intrinsics_gen)
