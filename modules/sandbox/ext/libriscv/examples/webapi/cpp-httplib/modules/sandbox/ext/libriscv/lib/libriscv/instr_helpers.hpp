#pragma once

#define ATOMIC_INSTR(x, ...) INSTRUCTION(x, __VA_ARGS__)
#define FLOAT_INSTR(x, ...) INSTRUCTION(x, __VA_ARGS__)
#define VECTOR_INSTR(x, ...) INSTRUCTION(x, __VA_ARGS__)
#define COMPRESSED_INSTR(x, ...) INSTRUCTION(x, __VA_ARGS__)
#define RVINSTR_ATTR     RISCV_HOT_PATH()
#define RVINSTR_COLDATTR RISCV_COLD_PATH()
#define RVPRINTR_ATTR    RISCV_COLD_PATH()

#define DECODED_ATOMIC(x) DECODED_INSTR(x)
#define DECODED_FLOAT(x) DECODED_INSTR(x)
#define DECODED_VECTOR(x) DECODED_INSTR(x)
#define DECODED_COMPR(x) DECODED_INSTR(x)

#define CI_CODE(x, y) ((x << 13) | (y))
#define CIC2_CODE(x, y) ((x << 12) | (y))

#define RVREGTYPE(x) typename std::remove_reference_t<decltype(x)>::address_t
#define RVTOSIGNED(x) (static_cast<typename std::make_signed<decltype(x)>::type> (x))
#define RVSIGNTYPE(x) typename std::make_signed<RVREGTYPE(x)>::type
#define RVIMM(x, y)   y.signed_imm()
#define RVXLEN(x)     (8u*sizeof(RVREGTYPE(x)))
#define RVIS32BIT(x)  (sizeof(RVREGTYPE(x)) == 4)
#define RVIS64BIT(x)  (sizeof(RVREGTYPE(x)) == 8)
#define RVIS128BIT(x) (sizeof(RVREGTYPE(x)) == 16)
#define RVISGE64BIT(x)  (sizeof(RVREGTYPE(x)) >= 8)
