add_llvm_library(LLVMObject
  Archive.cpp
  ArchiveWriter.cpp
  Binary.cpp
  COFFObjectFile.cpp
  COFFYAML.cpp
  ELF.cpp
  ELFObjectFile.cpp
  ELFYAML.cpp
  Error.cpp
  IRObjectFile.cpp
  MachOObjectFile.cpp
  MachOUniversal.cpp
  Object.cpp
  ObjectFile.cpp
  RecordStreamer.cpp
  SymbolicFile.cpp
  SymbolSize.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Object

  DEPENDS
  intrinsics_gen
  )
