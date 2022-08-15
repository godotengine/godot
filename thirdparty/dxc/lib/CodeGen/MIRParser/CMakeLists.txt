add_llvm_library(LLVMMIRParser
  MILexer.cpp
  MIParser.cpp
  MIRParser.cpp
  )

add_dependencies(LLVMMIRParser intrinsics_gen)
