#ifndef LIBRISCV_SETTINGS_H
#define LIBRISCV_SETTINGS_H

/*
 * These values are automatically set according to their cmake variables.
 */
/* #undef RISCV_DEBUG */
#define RISCV_EXT_A
#define RISCV_EXT_C
/* #undef RISCV_EXT_V */
/* #undef RISCV_32I */
#define RISCV_64I
/* #undef RISCV_128I */
/* #undef RISCV_FCSR */
/* #undef RISCV_EXPERIMENTAL */
/* #undef RISCV_MEMORY_TRAPS */
/* #undef RISCV_MULTIPROCESS */
#define RISCV_BINARY_TRANSLATION
#define RISCV_FLAT_RW_ARENA
/* #undef RISCV_ENCOMPASSING_ARENA */
#define RISCV_THREADED
/* #undef RISCV_TAILCALL_DISPATCH */
/* #undef RISCV_LIBTCC */

#endif /* LIBRISCV_SETTINGS_H */
