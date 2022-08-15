include(HCT)

add_hlsl_hctgen(HLSLIntrinsicOp OUTPUT HlslIntrinsicOp.h)

add_subdirectory(DXIL)
add_subdirectory(DxilContainer)
add_subdirectory(HLSL)
add_subdirectory(Support)
add_subdirectory(Tracing)
