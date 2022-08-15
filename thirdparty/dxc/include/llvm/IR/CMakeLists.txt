set(LLVM_TARGET_DEFINITIONS Intrinsics.td)

tablegen(LLVM Intrinsics.gen -gen-intrinsic)

add_public_tablegen_target(intrinsics_gen)
