# Make sure that the path to libffi headers is on the command
# line. That path can be a compiler's non-default path even when
# FFI_INCLUDE_DIR was not used, because cmake has its own paths for
# searching for headers (CMAKE_SYSTEM_INCLUDE_PATH, for instance):
if( FFI_INCLUDE_PATH )
  include_directories( ${FFI_INCLUDE_PATH} )
endif()

add_llvm_library(LLVMInterpreter
  Execution.cpp
  ExternalFunctions.cpp
  Interpreter.cpp
  )

if( LLVM_ENABLE_FFI )
  target_link_libraries( LLVMInterpreter PRIVATE ${FFI_LIBRARY_PATH} )
endif()

add_dependencies(LLVMInterpreter intrinsics_gen)
