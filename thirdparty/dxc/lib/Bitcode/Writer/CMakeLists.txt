add_llvm_library(LLVMBitWriter
  BitWriter.cpp
  BitcodeWriter.cpp
  BitcodeWriterPass.cpp
  ValueEnumerator.cpp

  DEPENDS
  intrinsics_gen
  )
