add_llvm_library(LLVMMCParser
  AsmLexer.cpp
  AsmParser.cpp
  COFFAsmParser.cpp
  DarwinAsmParser.cpp
  ELFAsmParser.cpp
  MCAsmLexer.cpp
  MCAsmParser.cpp
  MCAsmParserExtension.cpp
  MCTargetAsmParser.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/MC/MCParser
  )
