/*
** Common definitions for the JIT compiler.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
*/

#ifndef _LJ_JIT_H
#define _LJ_JIT_H

#include "lj_obj.h"
#if LJ_HASJIT
#include "lj_ir.h"

/* -- JIT engine flags ---------------------------------------------------- */

/* General JIT engine flags. 4 bits. */
#define JIT_F_ON		0x00000001

/* CPU-specific JIT engine flags. 12 bits. Flags and strings must match. */
#define JIT_F_CPU		0x00000010

#if LJ_TARGET_X86ORX64

#define JIT_F_SSE3		(JIT_F_CPU << 0)
#define JIT_F_SSE4_1		(JIT_F_CPU << 1)
#define JIT_F_BMI2		(JIT_F_CPU << 2)


#define JIT_F_CPUSTRING		"\4SSE3\6SSE4.1\4BMI2"

#elif LJ_TARGET_ARM

#define JIT_F_ARMV6_		(JIT_F_CPU << 0)
#define JIT_F_ARMV6T2_		(JIT_F_CPU << 1)
#define JIT_F_ARMV7		(JIT_F_CPU << 2)
#define JIT_F_ARMV8		(JIT_F_CPU << 3)
#define JIT_F_VFPV2		(JIT_F_CPU << 4)
#define JIT_F_VFPV3		(JIT_F_CPU << 5)

#define JIT_F_ARMV6		(JIT_F_ARMV6_|JIT_F_ARMV6T2_|JIT_F_ARMV7|JIT_F_ARMV8)
#define JIT_F_ARMV6T2		(JIT_F_ARMV6T2_|JIT_F_ARMV7|JIT_F_ARMV8)
#define JIT_F_VFP		(JIT_F_VFPV2|JIT_F_VFPV3)

#define JIT_F_CPUSTRING		"\5ARMv6\7ARMv6T2\5ARMv7\5ARMv8\5VFPv2\5VFPv3"

#elif LJ_TARGET_PPC

#define JIT_F_SQRT		(JIT_F_CPU << 0)
#define JIT_F_ROUND		(JIT_F_CPU << 1)

#define JIT_F_CPUSTRING		"\4SQRT\5ROUND"

#elif LJ_TARGET_MIPS

#define JIT_F_MIPSXXR2		(JIT_F_CPU << 0)

#if LJ_TARGET_MIPS32
#if LJ_TARGET_MIPSR6
#define JIT_F_CPUSTRING		"\010MIPS32R6"
#else
#define JIT_F_CPUSTRING		"\010MIPS32R2"
#endif
#else
#if LJ_TARGET_MIPSR6
#define JIT_F_CPUSTRING		"\010MIPS64R6"
#else
#define JIT_F_CPUSTRING		"\010MIPS64R2"
#endif
#endif

#elif LJ_TARGET_RISCV64

#define JIT_F_RVC		(JIT_F_CPU << 0)
#define JIT_F_RVZba		(JIT_F_CPU << 1)
#define JIT_F_RVZbb		(JIT_F_CPU << 2)
#define JIT_F_RVZicond		(JIT_F_CPU << 3)
#define JIT_F_RVXThead		(JIT_F_CPU << 4)

#define JIT_F_CPUSTRING		"\003RVC\003Zba\003Zbb\006Zicond\006XThead"

#if LJ_TARGET_LINUX
#include <sys/syscall.h>

#ifndef __NR_riscv_hwprobe
#ifndef __NR_arch_specific_syscall
#define __NR_arch_specific_syscall 244
#endif
#define __NR_riscv_hwprobe (__NR_arch_specific_syscall + 14)
#endif

struct riscv_hwprobe {
    int64_t key;
    uint64_t value;
};

#define RISCV_HWPROBE_KEY_MVENDORID     0
#define RISCV_HWPROBE_KEY_MARCHID       1
#define RISCV_HWPROBE_KEY_MIMPID        2
#define RISCV_HWPROBE_KEY_BASE_BEHAVIOR 3
#define RISCV_HWPROBE_KEY_IMA_EXT_0     4

#define RISCV_HWPROBE_IMA_C      (1 << 1)
#define RISCV_HWPROBE_EXT_ZBA    (1 << 3)
#define RISCV_HWPROBE_EXT_ZBB    (1 << 4)
#define RISCV_HWPROBE_EXT_ZICOND (1ULL << 35)

#endif

#else

#define JIT_F_CPUSTRING		""

#endif

/* Optimization flags. 12 bits. */
#define JIT_F_OPT		0x00010000
#define JIT_F_OPT_MASK		0x0fff0000

#define JIT_F_OPT_FOLD		(JIT_F_OPT << 0)
#define JIT_F_OPT_CSE		(JIT_F_OPT << 1)
#define JIT_F_OPT_DCE		(JIT_F_OPT << 2)
#define JIT_F_OPT_FWD		(JIT_F_OPT << 3)
#define JIT_F_OPT_DSE		(JIT_F_OPT << 4)
#define JIT_F_OPT_NARROW	(JIT_F_OPT << 5)
#define JIT_F_OPT_LOOP		(JIT_F_OPT << 6)
#define JIT_F_OPT_ABC		(JIT_F_OPT << 7)
#define JIT_F_OPT_SINK		(JIT_F_OPT << 8)
#define JIT_F_OPT_FUSE		(JIT_F_OPT << 9)
#define JIT_F_OPT_FMA		(JIT_F_OPT << 10)

/* Optimizations names for -O. Must match the order above. */
#define JIT_F_OPTSTRING	\
  "\4fold\3cse\3dce\3fwd\3dse\6narrow\4loop\3abc\4sink\4fuse\3fma"

/* Optimization levels set a fixed combination of flags. */
#define JIT_F_OPT_0	0
#define JIT_F_OPT_1	(JIT_F_OPT_FOLD|JIT_F_OPT_CSE|JIT_F_OPT_DCE)
#define JIT_F_OPT_2	(JIT_F_OPT_1|JIT_F_OPT_NARROW|JIT_F_OPT_LOOP)
#define JIT_F_OPT_3	(JIT_F_OPT_2|\
  JIT_F_OPT_FWD|JIT_F_OPT_DSE|JIT_F_OPT_ABC|JIT_F_OPT_SINK|JIT_F_OPT_FUSE)
#define JIT_F_OPT_DEFAULT	JIT_F_OPT_3
/* Note: FMA is not set by default. */

/* -- JIT engine parameters ----------------------------------------------- */

#if LJ_TARGET_WINDOWS || LJ_64
/* See: https://devblogs.microsoft.com/oldnewthing/20031008-00/?p=42223 */
#define JIT_P_sizemcode_DEFAULT		64
#else
/* Could go as low as 4K, but the mmap() overhead would be rather high. */
#define JIT_P_sizemcode_DEFAULT		32
#endif

/* Optimization parameters and their defaults. Length is a char in octal! */
#define JIT_PARAMDEF(_) \
  _(\010, maxtrace,	1000)	/* Max. # of traces in cache. */ \
  _(\011, maxrecord,	4000)	/* Max. # of recorded IR instructions. */ \
  _(\012, maxirconst,	500)	/* Max. # of IR constants of a trace. */ \
  _(\007, maxside,	100)	/* Max. # of side traces of a root trace. */ \
  _(\007, maxsnap,	500)	/* Max. # of snapshots for a trace. */ \
  _(\011, minstitch,	0)	/* Min. # of IR ins for a stitched trace. */ \
  \
  _(\007, hotloop,	56)	/* # of iter. to detect a hot loop/call. */ \
  _(\007, hotexit,	10)	/* # of taken exits to start a side trace. */ \
  _(\007, tryside,	4)	/* # of attempts to compile a side trace. */ \
  \
  _(\012, instunroll,	4)	/* Max. unroll for instable loops. */ \
  _(\012, loopunroll,	15)	/* Max. unroll for loop ops in side traces. */ \
  _(\012, callunroll,	3)	/* Max. unroll for recursive calls. */ \
  _(\011, recunroll,	2)	/* Min. unroll for true recursion. */ \
  \
  /* Size of each machine code area (in KBytes). */ \
  _(\011, sizemcode,	JIT_P_sizemcode_DEFAULT) \
  /* Max. total size of all machine code areas (in KBytes). */ \
  _(\010, maxmcode,	512) \
  /* End of list. */

enum {
#define JIT_PARAMENUM(len, name, value)	JIT_P_##name,
JIT_PARAMDEF(JIT_PARAMENUM)
#undef JIT_PARAMENUM
  JIT_P__MAX
};

#define JIT_PARAMSTR(len, name, value)	#len #name
#define JIT_P_STRING	JIT_PARAMDEF(JIT_PARAMSTR)

/* -- JIT engine data structures ------------------------------------------ */

/* Trace compiler state. */
typedef enum {
  LJ_TRACE_IDLE,	/* Trace compiler idle. */
  LJ_TRACE_ACTIVE = 0x10,
  LJ_TRACE_RECORD,	/* Bytecode recording active. */
  LJ_TRACE_RECORD_1ST,	/* Record 1st instruction, too. */
  LJ_TRACE_START,	/* New trace started. */
  LJ_TRACE_END,		/* End of trace. */
  LJ_TRACE_ASM,		/* Assemble trace. */
  LJ_TRACE_ERR		/* Trace aborted with error. */
} TraceState;

/* Post-processing action. */
typedef enum {
  LJ_POST_NONE,		/* No action. */
  LJ_POST_FIXCOMP,	/* Fixup comparison and emit pending guard. */
  LJ_POST_FIXGUARD,	/* Fixup and emit pending guard. */
  LJ_POST_FIXGUARDSNAP,	/* Fixup and emit pending guard and snapshot. */
  LJ_POST_FIXBOOL,	/* Fixup boolean result. */
  LJ_POST_FIXCONST,	/* Fixup constant results. */
  LJ_POST_FFRETRY	/* Suppress recording of retried fast functions. */
} PostProc;

/* Machine code type. */
#if LJ_TARGET_X86ORX64
typedef uint8_t MCode;
#else
typedef uint32_t MCode;
#endif

/* Linked list of MCode areas. */
typedef struct MCLink {
  MCode *next;		/* Next area. */
  size_t size;		/* Size of current area. */
} MCLink;

/* Stack snapshot header. */
typedef struct SnapShot {
  uint32_t mapofs;	/* Offset into snapshot map. */
  IRRef1 ref;		/* First IR ref for this snapshot. */
  uint16_t mcofs;	/* Offset into machine code in MCode units. */
  uint8_t nslots;	/* Number of valid slots. */
  uint8_t topslot;	/* Maximum frame extent. */
  uint8_t nent;		/* Number of compressed entries. */
  uint8_t count;	/* Count of taken exits for this snapshot. */
} SnapShot;

#define SNAPCOUNT_DONE	255	/* Already compiled and linked a side trace. */

/* Compressed snapshot entry. */
typedef uint32_t SnapEntry;

#define SNAP_FRAME		0x010000	/* Frame slot. */
#define SNAP_CONT		0x020000	/* Continuation slot. */
#define SNAP_NORESTORE		0x040000	/* No need to restore slot. */
#define SNAP_SOFTFPNUM		0x080000	/* Soft-float number. */
#define SNAP_KEYINDEX		0x100000	/* Traversal key index. */
LJ_STATIC_ASSERT(SNAP_FRAME == TREF_FRAME);
LJ_STATIC_ASSERT(SNAP_CONT == TREF_CONT);
LJ_STATIC_ASSERT(SNAP_KEYINDEX == TREF_KEYINDEX);

#define SNAP(slot, flags, ref)	(((SnapEntry)(slot) << 24) + (flags) + (ref))
#define SNAP_TR(slot, tr) \
  (((SnapEntry)(slot) << 24) + \
   ((tr) & (TREF_KEYINDEX|TREF_CONT|TREF_FRAME|TREF_REFMASK)))
#if !LJ_FR2
#define SNAP_MKPC(pc)		((SnapEntry)u32ptr(pc))
#endif
#define SNAP_MKFTSZ(ftsz)	((SnapEntry)(ftsz))
#define snap_ref(sn)		((sn) & 0xffff)
#define snap_slot(sn)		((BCReg)((sn) >> 24))
#define snap_isframe(sn)	((sn) & SNAP_FRAME)
#define snap_setref(sn, ref)	(((sn) & (0xffff0000&~SNAP_NORESTORE)) | (ref))

static LJ_AINLINE const BCIns *snap_pc(SnapEntry *sn)
{
#if LJ_FR2
  uint64_t pcbase;
  memcpy(&pcbase, sn, sizeof(uint64_t));
  return (const BCIns *)(pcbase >> 8);
#else
  return (const BCIns *)(uintptr_t)*sn;
#endif
}

/* Snapshot and exit numbers. */
typedef uint32_t SnapNo;
typedef uint32_t ExitNo;

/* Trace number. */
typedef uint32_t TraceNo;	/* Used to pass around trace numbers. */
typedef uint16_t TraceNo1;	/* Stored trace number. */

/* Type of link. ORDER LJ_TRLINK */
typedef enum {
  LJ_TRLINK_NONE,		/* Incomplete trace. No link, yet. */
  LJ_TRLINK_ROOT,		/* Link to other root trace. */
  LJ_TRLINK_LOOP,		/* Loop to same trace. */
  LJ_TRLINK_TAILREC,		/* Tail-recursion. */
  LJ_TRLINK_UPREC,		/* Up-recursion. */
  LJ_TRLINK_DOWNREC,		/* Down-recursion. */
  LJ_TRLINK_INTERP,		/* Fallback to interpreter. */
  LJ_TRLINK_RETURN,		/* Return to interpreter. */
  LJ_TRLINK_STITCH		/* Trace stitching. */
} TraceLink;

/* Trace object. */
typedef struct GCtrace {
  GCHeader;
  uint16_t nsnap;	/* Number of snapshots. */
  IRRef nins;		/* Next IR instruction. Biased with REF_BIAS. */
#if LJ_GC64
  uint32_t unused_gc64;
#endif
  GCRef gclist;
  IRIns *ir;		/* IR instructions/constants. Biased with REF_BIAS. */
  IRRef nk;		/* Lowest IR constant. Biased with REF_BIAS. */
  uint32_t nsnapmap;	/* Number of snapshot map elements. */
  SnapShot *snap;	/* Snapshot array. */
  SnapEntry *snapmap;	/* Snapshot map. */
  GCRef startpt;	/* Starting prototype. */
  MRef startpc;		/* Bytecode PC of starting instruction. */
  BCIns startins;	/* Original bytecode of starting instruction. */
  MSize szmcode;	/* Size of machine code. */
  MCode *mcode;		/* Start of machine code. */
#if LJ_ABI_PAUTH
  ASMFunction mcauth;	/* Start of machine code, with ptr auth applied. */
#endif
  MSize mcloop;		/* Offset of loop start in machine code. */
  uint16_t nchild;	/* Number of child traces (root trace only). */
  uint16_t spadjust;	/* Stack pointer adjustment (offset in bytes). */
  TraceNo1 traceno;	/* Trace number. */
  TraceNo1 link;	/* Linked trace (or self for loops). */
  TraceNo1 root;	/* Root trace of side trace (or 0 for root traces). */
  TraceNo1 nextroot;	/* Next root trace for same prototype. */
  TraceNo1 nextside;	/* Next side trace of same root trace. */
  uint8_t sinktags;	/* Trace has SINK tags. */
  uint8_t topslot;	/* Top stack slot already checked to be allocated. */
  uint8_t linktype;	/* Type of link. */
  uint8_t unused1;
#ifdef LUAJIT_USE_GDBJIT
  void *gdbjit_entry;	/* GDB JIT entry. */
#endif
} GCtrace;

#define gco2trace(o)	check_exp((o)->gch.gct == ~LJ_TTRACE, (GCtrace *)(o))
#define traceref(J, n) \
  check_exp((n)>0 && (MSize)(n)<J->sizetrace, (GCtrace *)gcref(J->trace[(n)]))

LJ_STATIC_ASSERT(offsetof(GChead, gclist) == offsetof(GCtrace, gclist));

static LJ_AINLINE MSize snap_nextofs(GCtrace *T, SnapShot *snap)
{
  if (snap+1 == &T->snap[T->nsnap])
    return T->nsnapmap;
  else
    return (snap+1)->mapofs;
}

/* Round-robin penalty cache for bytecodes leading to aborted traces. */
typedef struct HotPenalty {
  MRef pc;		/* Starting bytecode PC. */
  uint16_t val;		/* Penalty value, i.e. hotcount start. */
  uint16_t reason;	/* Abort reason (really TraceErr). */
} HotPenalty;

#define PENALTY_SLOTS	64	/* Penalty cache slot. Must be a power of 2. */
#define PENALTY_MIN	(36*2)	/* Minimum penalty value. */
#define PENALTY_MAX	60000	/* Maximum penalty value. */
#define PENALTY_RNDBITS	4	/* # of random bits to add to penalty value. */

/* Round-robin backpropagation cache for narrowing conversions. */
typedef struct BPropEntry {
  IRRef1 key;		/* Key: original reference. */
  IRRef1 val;		/* Value: reference after conversion. */
  IRRef mode;		/* Mode for this entry (currently IRCONV_*). */
} BPropEntry;

/* Number of slots for the backpropagation cache. Must be a power of 2. */
#define BPROP_SLOTS	16

/* Scalar evolution analysis cache. */
typedef struct ScEvEntry {
  MRef pc;		/* Bytecode PC of FORI. */
  IRRef1 idx;		/* Index reference. */
  IRRef1 start;		/* Constant start reference. */
  IRRef1 stop;		/* Constant stop reference. */
  IRRef1 step;		/* Constant step reference. */
  IRType1 t;		/* Scalar type. */
  uint8_t dir;		/* Direction. 1: +, 0: -. */
} ScEvEntry;

/* Reverse bytecode map (IRRef -> PC). Only for selected instructions. */
typedef struct RBCHashEntry {
  MRef pc;		/* Bytecode PC. */
  GCRef pt;		/* Prototype. */
  IRRef ref;		/* IR reference. */
} RBCHashEntry;

/* Number of slots in the reverse bytecode hash table. Must be a power of 2. */
#define RBCHASH_SLOTS	8

/* 128 bit SIMD constants. */
enum {
  LJ_KSIMD_ABS,
  LJ_KSIMD_NEG,
  LJ_KSIMD__MAX
};

enum {
#if LJ_TARGET_X86ORX64
  LJ_K64_TOBIT,		/* 2^52 + 2^51 */
  LJ_K64_2P64,		/* 2^64 */
  LJ_K64_M2P64,		/* -2^64 */
#if LJ_32
  LJ_K64_M2P64_31,	/* -2^64 or -2^31 */
#else
  LJ_K64_M2P64_31 = LJ_K64_M2P64,
#endif
#endif
#if LJ_TARGET_MIPS
  LJ_K64_2P31,		/* 2^31 */
#if LJ_64
  LJ_K64_2P63,		/* 2^63 */
  LJ_K64_M2P64,		/* -2^64 */
#endif
#endif
  LJ_K64__MAX,
};
#define LJ_K64__USED	(LJ_TARGET_X86ORX64 || LJ_TARGET_MIPS)

enum {
#if LJ_TARGET_X86ORX64
  LJ_K32_M2P64_31,	/* -2^64 or -2^31 */
#endif
#if LJ_TARGET_PPC
  LJ_K32_2P52_2P31,	/* 2^52 + 2^31 */
  LJ_K32_2P52,		/* 2^52 */
#endif
#if LJ_TARGET_PPC || LJ_TARGET_MIPS
  LJ_K32_2P31,		/* 2^31 */
#endif
#if LJ_TARGET_MIPS64
  LJ_K32_2P63,		/* 2^63 */
  LJ_K32_M2P64,		/* -2^64 */
#endif
  LJ_K32__MAX
};
#define LJ_K32__USED	(LJ_TARGET_X86ORX64 || LJ_TARGET_PPC || LJ_TARGET_MIPS)

/* Get 16 byte aligned pointer to SIMD constant. */
#define LJ_KSIMD(J, n) \
  ((TValue *)(((intptr_t)&J->ksimd[2*(n)] + 15) & ~(intptr_t)15))

/* Set/reset flag to activate the SPLIT pass for the current trace. */
#if LJ_SOFTFP32 || (LJ_32 && LJ_HASFFI)
#define lj_needsplit(J)		(J->needsplit = 1)
#define lj_resetsplit(J)	(J->needsplit = 0)
#else
#define lj_needsplit(J)		UNUSED(J)
#define lj_resetsplit(J)	UNUSED(J)
#endif

/* Fold state is used to fold instructions on-the-fly. */
typedef struct FoldState {
  IRIns ins;		/* Currently emitted instruction. */
  IRIns left[2];	/* Instruction referenced by left operand. */
  IRIns right[2];	/* Instruction referenced by right operand. */
} FoldState;

/* JIT compiler state. */
typedef struct jit_State {
  GCtrace cur;		/* Current trace. */
  GCtrace *curfinal;	/* Final address of current trace (set during asm). */

  lua_State *L;		/* Current Lua state. */
  const BCIns *pc;	/* Current PC. */
  GCfunc *fn;		/* Current function. */
  GCproto *pt;		/* Current prototype. */
  TRef *base;		/* Current frame base, points into J->slots. */

  uint32_t flags;	/* JIT engine flags. */
  BCReg maxslot;	/* Relative to baseslot. */
  BCReg baseslot;	/* Current frame base, offset into J->slots. */

  uint8_t mergesnap;	/* Allowed to merge with next snapshot. */
  uint8_t needsnap;	/* Need snapshot before recording next bytecode. */
  IRType1 guardemit;	/* Accumulated IRT_GUARD for emitted instructions. */
  uint8_t bcskip;	/* Number of bytecode instructions to skip. */

  FoldState fold;	/* Fold state. */

  const BCIns *bc_min;	/* Start of allowed bytecode range for root trace. */
  MSize bc_extent;	/* Extent of the range. */

  TraceState state;	/* Trace compiler state. */

  int32_t instunroll;	/* Unroll counter for instable loops. */
  int32_t loopunroll;	/* Unroll counter for loop ops in side traces. */
  int32_t tailcalled;	/* Number of successive tailcalls. */
  int32_t framedepth;	/* Current frame depth. */
  int32_t retdepth;	/* Return frame depth (count of RETF). */

#if LJ_K32__USED
  uint32_t k32[LJ_K32__MAX];  /* Common 4 byte constants used by backends. */
#endif
  TValue ksimd[LJ_KSIMD__MAX*2+1];  /* 16 byte aligned SIMD constants. */
#if LJ_K64__USED
  TValue k64[LJ_K64__MAX];  /* Common 8 byte constants. */
#endif

  IRIns *irbuf;		/* Temp. IR instruction buffer. Biased with REF_BIAS. */
  IRRef irtoplim;	/* Upper limit of instruction buffer (biased). */
  IRRef irbotlim;	/* Lower limit of instruction buffer (biased). */
  IRRef loopref;	/* Last loop reference or ref of final LOOP (or 0). */

  MSize sizesnap;	/* Size of temp. snapshot buffer. */
  SnapShot *snapbuf;	/* Temp. snapshot buffer. */
  SnapEntry *snapmapbuf;  /* Temp. snapshot map buffer. */
  MSize sizesnapmap;	/* Size of temp. snapshot map buffer. */

  PostProc postproc;	/* Required post-processing after execution. */
#if LJ_SOFTFP32 || (LJ_32 && LJ_HASFFI)
  uint8_t needsplit;	/* Need SPLIT pass. */
#endif
  uint8_t retryrec;	/* Retry recording. */

  GCRef *trace;		/* Array of traces. */
  TraceNo freetrace;	/* Start of scan for next free trace. */
  MSize sizetrace;	/* Size of trace array. */
  IRRef1 ktrace;	/* Reference to KGC with GCtrace. */

  IRRef1 chain[IR__MAX];  /* IR instruction skip-list chain anchors. */
  TRef slot[LJ_MAX_JSLOTS+LJ_STACK_EXTRA];  /* Stack slot map. */

  int32_t param[JIT_P__MAX];  /* JIT engine parameters. */

  MCode *exitstubgroup[LJ_MAX_EXITSTUBGR];  /* Exit stub group addresses. */

  HotPenalty penalty[PENALTY_SLOTS];  /* Penalty slots. */
  uint32_t penaltyslot;	/* Round-robin index into penalty slots. */

#ifdef LUAJIT_ENABLE_TABLE_BUMP
  RBCHashEntry rbchash[RBCHASH_SLOTS];  /* Reverse bytecode map. */
#endif

  BPropEntry bpropcache[BPROP_SLOTS];  /* Backpropagation cache slots. */
  uint32_t bpropslot;	/* Round-robin index into bpropcache slots. */

  ScEvEntry scev;	/* Scalar evolution analysis cache slots. */

  const BCIns *startpc;	/* Bytecode PC of starting instruction. */
  TraceNo parent;	/* Parent of current side trace (0 for root traces). */
  ExitNo exitno;	/* Exit number in parent of current side trace. */
  int exitcode;		/* Exit code from unwound trace. */

  BCIns *patchpc;	/* PC for pending re-patch. */
  BCIns patchins;	/* Instruction for pending re-patch. */

  int mcprot;		/* Protection of current mcode area. */
  MCode *mcarea;	/* Base of current mcode area. */
  MCode *mctop;		/* Top of current mcode area. */
  MCode *mcbot;		/* Bottom of current mcode area. */
  size_t szmcarea;	/* Size of current mcode area. */
  size_t szallmcarea;	/* Total size of all allocated mcode areas. */

  TValue errinfo;	/* Additional info element for trace errors. */

#if LJ_HASPROFILE
  GCproto *prev_pt;	/* Previous prototype. */
  BCLine prev_line;	/* Previous line. */
  int prof_mode;	/* Profiling mode: 0, 'f', 'l'. */
#endif
} jit_State;

#ifdef LUA_USE_ASSERT
#define lj_assertJ(c, ...)	lj_assertG_(J2G(J), (c), __VA_ARGS__)
#else
#define lj_assertJ(c, ...)	((void)J)
#endif
#endif

#endif
