#include <string>

namespace riscv {
	extern const std::string bintr_code =
		R"123(
#if (defined(__TINYC__) && defined(__FreeBSD__))
#define int8_t   char
#define uint8_t  unsigned char
#define int16_t  short
#define uint16_t unsigned short
#define int32_t  int
#define uint32_t unsigned int
#define int64_t  long long
#define uint64_t unsigned long long
#define uintptr_t unsigned long long
#elif defined(__TINYC__) && defined(_WIN32)
#define int8_t   char
#define uint8_t  unsigned char
#define int16_t  short
#define uint16_t unsigned short
#define int32_t  int
#define uint32_t unsigned int
#define int64_t  long long
#define uint64_t unsigned long long
#define uintptr_t unsigned long long
#elif defined(__TINYC__) && defined(__ANDROID__)
#define uintptr_t unsigned long
#elif defined(__TINYC__)
#define int8_t   signed char
#define uint8_t  unsigned char
#define int16_t  short
#define uint16_t unsigned short
#define int32_t  int
#define uint32_t unsigned int
#define int64_t  long
#define uint64_t unsigned long
#define uintptr_t unsigned long
#else
#include <stdint.h>
#endif
#define LIKELY(x) __builtin_expect((x), 1)
#define UNLIKELY(x) __builtin_expect((x), 0)
#define ILLEGAL_OPCODE  0
#define MISALIGNED_INSTRUCTION 4
#define VISIBLE  __attribute__((visibility("default")))
#define INTERNAL __attribute__((visibility("hidden")))

#if RISCV_TRANSLATION_DYLIB == 4
	typedef uint32_t addr_t;
	typedef int32_t saddr_t;
#  define XLEN  32
#elif RISCV_TRANSLATION_DYLIB == 8
	typedef uint64_t addr_t;
	typedef int64_t saddr_t;
#  define XLEN  64
#elif RISCV_TRANSLATION_DYLIB == 16
	typedef __uint128_t addr_t;
	typedef __int128_t saddr_t;
#endif
#ifdef RISCV_EXT_C
#define RISCV_ALIGN_MASK 0x1
#else
#define RISCV_ALIGN_MASK 0x3
#endif

#define HOST_UNKNOWN 0
#define HOST_AMD64   1

#ifdef __TINYC__
#define UNREACHABLE() /**/
static inline float fminf(float x, float y) {
	return (x < y) ? x : y;
}
static inline double fmin(double x, double y) {
	return (x < y) ? x : y;
}
static inline float fmaxf(float x, float y) {
	return (x >= y) ? x : y;
}
static inline double fmax(double x, double y) {
	return (x >= y) ? x : y;
}
static inline uint32_t do_bswap32(uint32_t x) {
	return (x << 24 | (x & 0xFF00) << 8 | (x & 0xFF0000) >> 8 | x >> 24);
}
#define do_bswap64(x) (do_bswap32((x) >> 32) | ((uint64_t)do_bswap32((x) >> 32) << 32))
#define do_clz(x) api.clz(x)
#define do_clzl(x) api.clzl(x)
#define do_ctz(x) api.ctz(x)
#define do_ctzl(x) api.ctzl(x)
#define do_cpop(x) api.cpop(x)
#define do_cpopl(x) api.cpopl(x)
#else
#define UNREACHABLE() __builtin_unreachable()
#define do_bswap32(x) __builtin_bswap32(x)
#define do_bswap64(x) __builtin_bswap64(x)
#define do_clz(x) __builtin_clz(x)
#define do_clzl(x) __builtin_clzl(x)
#define do_ctz(x) __builtin_ctz(x)
#define do_ctzl(x) __builtin_ctzl(x)
#define do_cpop(x) __builtin_popcount(x)
#define do_cpopl(x) __builtin_popcountl(x)
#define fminf(x, y) __builtin_fminf(x, y)
#define fmin(x, y) __builtin_fmin(x, y)
#define fmaxf(x, y) __builtin_fmaxf(x, y)
#define fmax(x, y) __builtin_fmax(x, y)
#endif

#ifdef __HAVE_BUILTIN_SPECULATION_SAFE_VALUE
#define SPECSAFE(x) __builtin_speculation_safe_value(x)
#else
#define SPECSAFE(x) (x)
#endif

#ifdef RISCV_EXT_VECTOR
typedef union {
	float  f32[RISCV_EXT_VECTOR / 4];
	double f64[RISCV_EXT_VECTOR / 8];
} VectorLane __attribute__ ((aligned (RISCV_EXT_VECTOR)));

typedef struct {
	VectorLane  lane[32];
} RVV __attribute__ ((aligned (RISCV_EXT_VECTOR)));
#endif

typedef union {
	int32_t i32[2];
	float   f32[2];
	int64_t i64;
	double  f64;
	struct {
		uint32_t bits  : 31;
		uint32_t sign  : 1;
		uint32_t upper;
	} lsign;
	struct {
		uint64_t bits  : 63;
		uint64_t sign  : 1;
	} usign;
} fp64reg;

#ifdef RISCV_NANBOXING
#define load_fl(reg, iv) \
	(reg)->i32[0] = (iv);
	(reg)->i32[1] = 0;
#define set_fl(reg, fv) \
	(reg)->f32[0] = (fv);
	(reg)->i32[1] = 0;
#else
#define load_fl(reg, fv) (reg)->i32[0] = (fv)
#define set_fl(reg, fv)  (reg)->f32[0] = (fv)
#endif

#define load_dbl(reg, dv) (reg)->i64 = (dv)
#define set_dbl(reg, dv)  (reg)->f64 = (dv)

// Thin variant of CPU for higher compilation speed
__attribute__((aligned(RISCV_MACHINE_ALIGNMENT)))
typedef struct {
	addr_t  r[32];
	addr_t  pc;
	uint32_t fcsr;
	fp64reg fr[32];
#ifdef RISCV_EXT_VECTOR
	RVV rvv;
#endif
} CPU;

#define PAGENO(x) ((addr_t)(x) >> 12)
#define PAGEOFF(x) ((addr_t)(x) & 0xFFF)
typedef void (*syscall_t) (CPU*);
typedef void (*handler) (CPU*, uint32_t);

static struct CallbackTable {
	uint8_t (*mem_ld8) (const CPU*, addr_t);
	uint16_t (*mem_ld16) (const CPU*, addr_t);
	uint32_t (*mem_ld32) (const CPU*, addr_t);
	uint64_t (*mem_ld64) (const CPU*, addr_t);
	void (*mem_st8) (const CPU*, addr_t, uint8_t);
	void (*mem_st16) (const CPU*, addr_t, uint16_t);
	void (*mem_st32) (const CPU*, addr_t, uint32_t);
	void (*mem_st64) (const CPU*, addr_t, uint64_t);
	void (*vec_load)(const CPU*, int, addr_t);
	void (*vec_store)(const CPU*, addr_t, int);
	syscall_t* syscalls;
	uint64_t (*system_call)(CPU*, addr_t, uint64_t, uint64_t, int);
	void (*unknown_syscall)(CPU*, addr_t);
	int  (*system)(CPU*, uint32_t);
	unsigned (*execute)(CPU*, uint32_t);
	unsigned (*execute_handler)(CPU*, uint32_t, uintptr_t);
	handler* handlers;
	void (*exception) (CPU*, addr_t, int);
	void (*trace) (CPU*, const char*, addr_t, uint32_t);
	float  (*sqrtf32)(float);
	double (*sqrtf64)(double);
	int (*clz) (uint32_t);
	int (*clzl) (uint64_t);
	int (*ctz) (uint32_t);
	int (*ctzl) (uint64_t);
	int (*cpop) (uint32_t);
	int (*cpopl) (uint64_t);
} api;
#define ARENA_READ_BOUNDARY  (RISCV_ARENA_END - 0x1000)
#define ARENA_WRITE_BOUNDARY (RISCV_ARENA_END - RISCV_ARENA_ROEND)
#define ARENA_READABLE(x) ((x) - 0x1000 < ARENA_READ_BOUNDARY)
#define ARENA_WRITABLE(x) ((x) - RISCV_ARENA_ROEND < ARENA_WRITE_BOUNDARY)

INTERNAL static int32_t arena_offset;
//#define ARENA_AT(cpu, x)  (arena_ptr + (x))
#define ARENA_AT(cpu, x)  (*(char **)((uintptr_t)cpu + arena_offset) + (x))

INTERNAL static int32_t ic_offset;
#define INS_COUNTER(cpu) (*(uint64_t *)((uintptr_t)cpu + ic_offset))
#define MAX_COUNTER(cpu) (*(uint64_t *)((uintptr_t)cpu + ic_offset + 8))

typedef struct {
	addr_t pageno;
	uint8_t *data;
} CachedPage;
INTERNAL static int32_t rdcache_offset;
#define RD_CACHE(cpu) ((CachedPage *)((uintptr_t)cpu + rdcache_offset))
#define WR_CACHE(cpu) ((CachedPage *)((uintptr_t)cpu + rdcache_offset + sizeof(CachedPage)))

#ifdef __TINYC__
// Use the API directly as TCC doesn't optimize well
#define rd8  api.mem_ld8
#define rd16 api.mem_ld16
#define rd32 api.mem_ld32
#define rd64 api.mem_ld64
#define wr8  api.mem_st8
#define wr16 api.mem_st16
#define wr32 api.mem_st32
#define wr64 api.mem_st64
#else
static uint8_t rd8(CPU* cpu, addr_t addr) {
	const addr_t pageno = addr >> 12;
	if (RD_CACHE(cpu)->pageno != pageno) {
		return api.mem_ld8(cpu, addr);
	}
	return RD_CACHE(cpu)->data[PAGEOFF(addr)];
}
static uint16_t rd16(CPU* cpu, addr_t addr) {
	const addr_t pageno = addr >> 12;
	if (RD_CACHE(cpu)->pageno != pageno) {
		return api.mem_ld16(cpu, addr);
	}
	return *(uint16_t *)&RD_CACHE(cpu)->data[PAGEOFF(addr)];
}
static uint32_t rd32(CPU* cpu, addr_t addr) {
	const addr_t pageno = addr >> 12;
	if (RD_CACHE(cpu)->pageno != pageno) {
		return api.mem_ld32(cpu, addr);
	}
	return *(uint32_t *)&RD_CACHE(cpu)->data[PAGEOFF(addr)];
}
static uint64_t rd64(CPU* cpu, addr_t addr) {
	const addr_t pageno = addr >> 12;
	if (RD_CACHE(cpu)->pageno != pageno) {
		return api.mem_ld64(cpu, addr);
	}
	return *(uint64_t *)&RD_CACHE(cpu)->data[PAGEOFF(addr)];
}
static void wr8(CPU* cpu, addr_t addr, uint8_t value) {
	if (WR_CACHE(cpu)->pageno == addr >> 12) {
		WR_CACHE(cpu)->data[PAGEOFF(addr)] = value;
		return;
	}
	api.mem_st8(cpu, addr, value);
}
static void wr16(CPU* cpu, addr_t addr, uint16_t value) {
	if (WR_CACHE(cpu)->pageno == addr >> 12) {
		*(uint16_t *)&WR_CACHE(cpu)->data[PAGEOFF(addr)] = value;
		return;
	}
	api.mem_st16(cpu, addr, value);
}
static void wr32(CPU* cpu, addr_t addr, uint32_t value) {
	if (WR_CACHE(cpu)->pageno == addr >> 12) {
		*(uint32_t *)&WR_CACHE(cpu)->data[PAGEOFF(addr)] = value;
		return;
	}
	api.mem_st32(cpu, addr, value);
}
static void wr64(CPU* cpu, addr_t addr, uint64_t value) {
	if (WR_CACHE(cpu)->pageno == addr >> 12) {
		*(uint64_t *)&WR_CACHE(cpu)->data[PAGEOFF(addr)] = value;
		return;
	}
	api.mem_st64(cpu, addr, value);
}
#endif

static inline int do_syscall(CPU* cpu, uint64_t counter, uint64_t max_counter, addr_t sysno)
{
	INS_COUNTER(cpu) = counter; // Reveal instruction counters
	MAX_COUNTER(cpu) = max_counter;
	addr_t old_pc = cpu->pc;
	if (LIKELY(sysno < RISCV_MAX_SYSCALLS))
		api.syscalls[SPECSAFE(sysno)](cpu);
	else
		api.unknown_syscall(cpu, sysno);
	// Resume if the system call did not modify PC, or hit a limit
	return (cpu->pc != old_pc || counter >= MAX_COUNTER(cpu));
}

#define JUMP_TO(addr) \
	pc = addr & ~(addr_t)RISCV_ALIGN_MASK;

// https://stackoverflow.com/questions/28868367/getting-the-high-part-of-64-bit-integer-multiplication
// As written by catid
static inline uint64_t MUL128(
	uint64_t* r_hi,
	const uint64_t x,
	const uint64_t y)
{
	const uint64_t x0 = (uint32_t)x, x1 = x >> 32;
	const uint64_t y0 = (uint32_t)y, y1 = y >> 32;
	const uint64_t p11 = x1 * y1, p01 = x0 * y1;
	const uint64_t p10 = x1 * y0, p00 = x0 * y0;

	// 64-bit product + two 32-bit values
	const uint64_t middle = p10 + (p00 >> 32) + (uint32_t)p01;

	// 64-bit product + two 32-bit values
	*r_hi = p11 + (middle >> 32) + (p01 >> 32);

	// Add LOW PART and lower half of MIDDLE PART
	return (middle << 32) | (uint32_t)p00;
}

#ifdef EMBEDDABLE_CODE
static
#else
extern VISIBLE
#endif
void init(struct CallbackTable* table, int32_t arena_off, int32_t ins_counter_off, int32_t rdcache_off)
{
	api = *table;
	arena_offset = arena_off;
	ic_offset = ins_counter_off;
	//max_ic_offset = ins_counter_off + sizeof(uint64_t);
	rdcache_offset = rdcache_off;
}

typedef struct {
	uint64_t ic;
	uint64_t max_ic;
} ReturnValues;
)123";
}
