macro(add_pdb_impl_folder group)
  list(APPEND PDB_IMPL_SOURCES ${ARGN})
  source_group(${group} FILES ${ARGN})
endmacro()

if(HAVE_DIA_SDK)
  include_directories(${MSVC_DIA_SDK_DIR}/include)
  set(LIBPDB_LINK_FOLDERS "${MSVC_DIA_SDK_DIR}\\lib")
  if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(LIBPDB_LINK_FOLDERS "${LIBPDB_LINK_FOLDERS}\\amd64")
  endif()
  file(TO_CMAKE_PATH "${LIBPDB_LINK_FOLDERS}\\diaguids.lib" LIBPDB_ADDITIONAL_LIBRARIES)

  add_pdb_impl_folder(DIA
    DIA/DIADataStream.cpp
    DIA/DIAEnumDebugStreams.cpp
    DIA/DIAEnumLineNumbers.cpp
    DIA/DIAEnumSourceFiles.cpp
    DIA/DIAEnumSymbols.cpp
    DIA/DIALineNumber.cpp
    DIA/DIARawSymbol.cpp
    DIA/DIASession.cpp
    DIA/DIASourceFile.cpp
    )

    set(LIBPDB_ADDITIONAL_HEADER_DIRS "${LLVM_MAIN_INCLUDE_DIR}/llvm/DebugInfo/PDB/DIA")

endif()

list(APPEND LIBPDB_ADDITIONAL_HEADER_DIRS "${LLVM_MAIN_INCLUDE_DIR}/llvm/DebugInfo/PDB")

add_llvm_library(LLVMDebugInfoPDB
  IPDBSourceFile.cpp
  PDB.cpp
  PDBContext.cpp
  PDBExtras.cpp
  PDBInterfaceAnchors.cpp
  PDBSymbol.cpp
  PDBSymbolAnnotation.cpp
  PDBSymbolBlock.cpp
  PDBSymbolCompiland.cpp
  PDBSymbolCompilandDetails.cpp
  PDBSymbolCompilandEnv.cpp
  PDBSymbolCustom.cpp
  PDBSymbolData.cpp
  PDBSymbolExe.cpp
  PDBSymbolFunc.cpp
  PDBSymbolFuncDebugEnd.cpp
  PDBSymbolFuncDebugStart.cpp
  PDBSymbolLabel.cpp
  PDBSymbolPublicSymbol.cpp
  PDBSymbolThunk.cpp
  PDBSymbolTypeArray.cpp
  PDBSymbolTypeBaseClass.cpp
  PDBSymbolTypeBuiltin.cpp
  PDBSymbolTypeCustom.cpp
  PDBSymbolTypeDimension.cpp
  PDBSymbolTypeEnum.cpp
  PDBSymbolTypeFriend.cpp
  PDBSymbolTypeFunctionArg.cpp
  PDBSymbolTypeFunctionSig.cpp
  PDBSymbolTypeManaged.cpp
  PDBSymbolTypePointer.cpp
  PDBSymbolTypeTypedef.cpp
  PDBSymbolTypeUDT.cpp
  PDBSymbolTypeVTable.cpp
  PDBSymbolTypeVTableShape.cpp
  PDBSymbolUnknown.cpp
  PDBSymbolUsingNamespace.cpp
  PDBSymDumper.cpp
  ${PDB_IMPL_SOURCES}

  ADDITIONAL_HEADER_DIRS
  ${LIBPDB_ADDITIONAL_HEADER_DIRS}
  )

target_link_libraries(LLVMDebugInfoPDB INTERFACE "${LIBPDB_ADDITIONAL_LIBRARIES}")
