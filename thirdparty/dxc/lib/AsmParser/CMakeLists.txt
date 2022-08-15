# AsmParser
add_llvm_library(LLVMAsmParser
  LLLexer.cpp
  LLParser.cpp
  Parser.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Analysis

  DEPENDS
  intrinsics_gen
  )
