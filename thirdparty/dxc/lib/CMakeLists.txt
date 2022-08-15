# `Support' and `TableGen' libraries are added on the top-level CMakeLists.txt

add_subdirectory(IR)
add_subdirectory(IRReader)
# add_subdirectory(CodeGen) # HLSL Change
add_subdirectory(Bitcode)
add_subdirectory(Transforms)
add_subdirectory(Linker)
add_subdirectory(Analysis)
# add_subdirectory(LTO) # HLSL Change
# add_subdirectory(MC) # HLSL Change
# add_subdirectory(Object) # HLSL Change
add_subdirectory(Option)
# add_subdirectory(DebugInfo) # HLSL Change
# add_subdirectory(ExecutionEngine) # HLSL Change
add_subdirectory(Target)
add_subdirectory(AsmParser)
# add_subdirectory(LineEditor) # HLSL Change
add_subdirectory(ProfileData)
# add_subdirectory(Fuzzer) # HLSL Change
add_subdirectory(Passes) # HLSL Change
add_subdirectory(PassPrinters) # HLSL Change
# add_subdirectory(LibDriver) # HLSL Change
add_subdirectory(DxcSupport) # HLSL Change
add_subdirectory(HLSL) # HLSL Change
add_subdirectory(DXIL) # HLSL Change
add_subdirectory(DxilContainer) # HLSL Change
add_subdirectory(DxilPIXPasses) # HLSL Change
if(WIN32) # HLSL Change
  add_subdirectory(DxilDia) # HLSL Change
endif(WIN32) # HLSL Change
add_subdirectory(DxilRootSignature) # HLSL Change
add_subdirectory(DxcBindingTable) # HLSL Change
add_subdirectory(DxrFallback) # HLSL Change
add_subdirectory(Miniz) # HLSL Change
