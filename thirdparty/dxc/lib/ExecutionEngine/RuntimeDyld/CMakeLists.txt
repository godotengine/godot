add_llvm_library(LLVMRuntimeDyld
  RTDyldMemoryManager.cpp
  RuntimeDyld.cpp
  RuntimeDyldChecker.cpp
  RuntimeDyldCOFF.cpp
  RuntimeDyldELF.cpp
  RuntimeDyldMachO.cpp

  DEPENDS
  intrinsics_gen
  )
