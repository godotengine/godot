# Add godot-cpp library
option(GODOTCPP_DISABLE_EXCEPTIONS "" OFF)
set(GODOTCPP_TARGET "template_release" CACHE STRING "")
add_subdirectory(godot-cpp)

# Add sandboxing library
option(RISCV_32I "" OFF)
option(RISCV_BINARY_TRANSLATION "" ON)
option(RISCV_MEMORY_TRAPS "" OFF)
add_subdirectory(libriscv/lib libriscv)
target_compile_definitions(riscv PUBLIC
	RISCV_SYSCALLS_MAX=600
	RISCV_BRK_MEMORY_SIZE=0x100000  # 1MB (we don't need much)
)
