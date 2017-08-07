/*************************************************
*      Perl-Compatible Regular Expressions       *
*************************************************/

/* PCRE is a library of functions to support regular expressions whose syntax
and semantics are as close as possible to those of the Perl 5 language.

                       Written by Philip Hazel
     Original API code Copyright (c) 1997-2012 University of Cambridge
         New API code Copyright (c) 2016 University of Cambridge

-----------------------------------------------------------------------------
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of the University of Cambridge nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
-----------------------------------------------------------------------------
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "pcre2_internal.h"

#ifdef SUPPORT_JIT

/* All-in-one: Since we use the JIT compiler only from here,
we just include it. This way we don't need to touch the build
system files. */

#define SLJIT_CONFIG_AUTO 1
#define SLJIT_CONFIG_STATIC 1
#define SLJIT_VERBOSE 0

#ifdef PCRE2_DEBUG
#define SLJIT_DEBUG 1
#else
#define SLJIT_DEBUG 0
#endif

#define SLJIT_MALLOC(size, allocator_data) pcre2_jit_malloc(size, allocator_data)
#define SLJIT_FREE(ptr, allocator_data) pcre2_jit_free(ptr, allocator_data)

static void * pcre2_jit_malloc(size_t size, void *allocator_data)
{
pcre2_memctl *allocator = ((pcre2_memctl*)allocator_data);
return allocator->malloc(size, allocator->memory_data);
}

static void pcre2_jit_free(void *ptr, void *allocator_data)
{
pcre2_memctl *allocator = ((pcre2_memctl*)allocator_data);
allocator->free(ptr, allocator->memory_data);
}

#include "sljit/sljitLir.c"

#if defined SLJIT_CONFIG_UNSUPPORTED && SLJIT_CONFIG_UNSUPPORTED
#error Unsupported architecture
#endif

/* Defines for debugging purposes. */

/* 1 - Use unoptimized capturing brackets.
   2 - Enable capture_last_ptr (includes option 1). */
/* #define DEBUG_FORCE_UNOPTIMIZED_CBRAS 2 */

/* 1 - Always have a control head. */
/* #define DEBUG_FORCE_CONTROL_HEAD 1 */

/* Allocate memory for the regex stack on the real machine stack.
Fast, but limited size. */
#define MACHINE_STACK_SIZE 32768

/* Growth rate for stack allocated by the OS. Should be the multiply
of page size. */
#define STACK_GROWTH_RATE 8192

/* Enable to check that the allocation could destroy temporaries. */
#if defined SLJIT_DEBUG && SLJIT_DEBUG
#define DESTROY_REGISTERS 1
#endif

/*
Short summary about the backtracking mechanism empolyed by the jit code generator:

The code generator follows the recursive nature of the PERL compatible regular
expressions. The basic blocks of regular expressions are condition checkers
whose execute different commands depending on the result of the condition check.
The relationship between the operators can be horizontal (concatenation) and
vertical (sub-expression) (See struct backtrack_common for more details).

  'ab' - 'a' and 'b' regexps are concatenated
  'a+' - 'a' is the sub-expression of the '+' operator

The condition checkers are boolean (true/false) checkers. Machine code is generated
for the checker itself and for the actions depending on the result of the checker.
The 'true' case is called as the matching path (expected path), and the other is called as
the 'backtrack' path. Branch instructions are expesive for all CPUs, so we avoid taken
branches on the matching path.

 Greedy star operator (*) :
   Matching path: match happens.
   Backtrack path: match failed.
 Non-greedy star operator (*?) :
   Matching path: no need to perform a match.
   Backtrack path: match is required.

The following example shows how the code generated for a capturing bracket
with two alternatives. Let A, B, C, D are arbirary regular expressions, and
we have the following regular expression:

   A(B|C)D

The generated code will be the following:

 A matching path
 '(' matching path (pushing arguments to the stack)
 B matching path
 ')' matching path (pushing arguments to the stack)
 D matching path
 return with successful match

 D backtrack path
 ')' backtrack path (If we arrived from "C" jump to the backtrack of "C")
 B backtrack path
 C expected path
 jump to D matching path
 C backtrack path
 A backtrack path

 Notice, that the order of backtrack code paths are the opposite of the fast
 code paths. In this way the topmost value on the stack is always belong
 to the current backtrack code path. The backtrack path must check
 whether there is a next alternative. If so, it needs to jump back to
 the matching path eventually. Otherwise it needs to clear out its own stack
 frame and continue the execution on the backtrack code paths.
*/

/*
Saved stack frames:

Atomic blocks and asserts require reloading the values of private data
when the backtrack mechanism performed. Because of OP_RECURSE, the data
are not necessarly known in compile time, thus we need a dynamic restore
mechanism.

The stack frames are stored in a chain list, and have the following format:
([ capturing bracket offset ][ start value ][ end value ])+ ... [ 0 ] [ previous head ]

Thus we can restore the private data to a particular point in the stack.
*/

typedef struct jit_arguments {
  /* Pointers first. */
  struct sljit_stack *stack;
  PCRE2_SPTR str;
  PCRE2_SPTR begin;
  PCRE2_SPTR end;
  pcre2_match_data *match_data;
  PCRE2_SPTR startchar_ptr;
  PCRE2_UCHAR *mark_ptr;
  int (*callout)(pcre2_callout_block *, void *);
  void *callout_data;
  /* Everything else after. */
  sljit_uw offset_limit;
  sljit_u32 limit_match;
  sljit_u32 oveccount;
  sljit_u32 options;
} jit_arguments;

#define JIT_NUMBER_OF_COMPILE_MODES 3

typedef struct executable_functions {
  void *executable_funcs[JIT_NUMBER_OF_COMPILE_MODES];
  void *read_only_data_heads[JIT_NUMBER_OF_COMPILE_MODES];
  sljit_uw executable_sizes[JIT_NUMBER_OF_COMPILE_MODES];
  sljit_u32 top_bracket;
  sljit_u32 limit_match;
} executable_functions;

typedef struct jump_list {
  struct sljit_jump *jump;
  struct jump_list *next;
} jump_list;

typedef struct stub_list {
  struct sljit_jump *start;
  struct sljit_label *quit;
  struct stub_list *next;
} stub_list;

typedef struct label_addr_list {
  struct sljit_label *label;
  sljit_uw *update_addr;
  struct label_addr_list *next;
} label_addr_list;

enum frame_types {
  no_frame = -1,
  no_stack = -2
};

enum control_types {
  type_mark = 0,
  type_then_trap = 1
};

typedef int (SLJIT_CALL *jit_function)(jit_arguments *args);

/* The following structure is the key data type for the recursive
code generator. It is allocated by compile_matchingpath, and contains
the arguments for compile_backtrackingpath. Must be the first member
of its descendants. */
typedef struct backtrack_common {
  /* Concatenation stack. */
  struct backtrack_common *prev;
  jump_list *nextbacktracks;
  /* Internal stack (for component operators). */
  struct backtrack_common *top;
  jump_list *topbacktracks;
  /* Opcode pointer. */
  PCRE2_SPTR cc;
} backtrack_common;

typedef struct assert_backtrack {
  backtrack_common common;
  jump_list *condfailed;
  /* Less than 0 if a frame is not needed. */
  int framesize;
  /* Points to our private memory word on the stack. */
  int private_data_ptr;
  /* For iterators. */
  struct sljit_label *matchingpath;
} assert_backtrack;

typedef struct bracket_backtrack {
  backtrack_common common;
  /* Where to coninue if an alternative is successfully matched. */
  struct sljit_label *alternative_matchingpath;
  /* For rmin and rmax iterators. */
  struct sljit_label *recursive_matchingpath;
  /* For greedy ? operator. */
  struct sljit_label *zero_matchingpath;
  /* Contains the branches of a failed condition. */
  union {
    /* Both for OP_COND, OP_SCOND. */
    jump_list *condfailed;
    assert_backtrack *assert;
    /* For OP_ONCE. Less than 0 if not needed. */
    int framesize;
  } u;
  /* Points to our private memory word on the stack. */
  int private_data_ptr;
} bracket_backtrack;

typedef struct bracketpos_backtrack {
  backtrack_common common;
  /* Points to our private memory word on the stack. */
  int private_data_ptr;
  /* Reverting stack is needed. */
  int framesize;
  /* Allocated stack size. */
  int stacksize;
} bracketpos_backtrack;

typedef struct braminzero_backtrack {
  backtrack_common common;
  struct sljit_label *matchingpath;
} braminzero_backtrack;

typedef struct char_iterator_backtrack {
  backtrack_common common;
  /* Next iteration. */
  struct sljit_label *matchingpath;
  union {
    jump_list *backtracks;
    struct {
      unsigned int othercasebit;
      PCRE2_UCHAR chr;
      BOOL enabled;
    } charpos;
  } u;
} char_iterator_backtrack;

typedef struct ref_iterator_backtrack {
  backtrack_common common;
  /* Next iteration. */
  struct sljit_label *matchingpath;
} ref_iterator_backtrack;

typedef struct recurse_entry {
  struct recurse_entry *next;
  /* Contains the function entry. */
  struct sljit_label *entry;
  /* Collects the calls until the function is not created. */
  jump_list *calls;
  /* Points to the starting opcode. */
  sljit_sw start;
} recurse_entry;

typedef struct recurse_backtrack {
  backtrack_common common;
  BOOL inlined_pattern;
} recurse_backtrack;

#define OP_THEN_TRAP OP_TABLE_LENGTH

typedef struct then_trap_backtrack {
  backtrack_common common;
  /* If then_trap is not NULL, this structure contains the real
  then_trap for the backtracking path. */
  struct then_trap_backtrack *then_trap;
  /* Points to the starting opcode. */
  sljit_sw start;
  /* Exit point for the then opcodes of this alternative. */
  jump_list *quit;
  /* Frame size of the current alternative. */
  int framesize;
} then_trap_backtrack;

#define MAX_RANGE_SIZE 4

typedef struct compiler_common {
  /* The sljit ceneric compiler. */
  struct sljit_compiler *compiler;
  /* First byte code. */
  PCRE2_SPTR start;
  /* Maps private data offset to each opcode. */
  sljit_s32 *private_data_ptrs;
  /* Chain list of read-only data ptrs. */
  void *read_only_data_head;
  /* Tells whether the capturing bracket is optimized. */
  sljit_u8 *optimized_cbracket;
  /* Tells whether the starting offset is a target of then. */
  sljit_u8 *then_offsets;
  /* Current position where a THEN must jump. */
  then_trap_backtrack *then_trap;
  /* Starting offset of private data for capturing brackets. */
  sljit_s32 cbra_ptr;
  /* Output vector starting point. Must be divisible by 2. */
  sljit_s32 ovector_start;
  /* Points to the starting character of the current match. */
  sljit_s32 start_ptr;
  /* Last known position of the requested byte. */
  sljit_s32 req_char_ptr;
  /* Head of the last recursion. */
  sljit_s32 recursive_head_ptr;
  /* First inspected character for partial matching.
     (Needed for avoiding zero length partial matches.) */
  sljit_s32 start_used_ptr;
  /* Starting pointer for partial soft matches. */
  sljit_s32 hit_start;
  /* Pointer of the match end position. */
  sljit_s32 match_end_ptr;
  /* Points to the marked string. */
  sljit_s32 mark_ptr;
  /* Recursive control verb management chain. */
  sljit_s32 control_head_ptr;
  /* Points to the last matched capture block index. */
  sljit_s32 capture_last_ptr;
  /* Fast forward skipping byte code pointer. */
  PCRE2_SPTR fast_forward_bc_ptr;
  /* Locals used by fast fail optimization. */
  sljit_s32 fast_fail_start_ptr;
  sljit_s32 fast_fail_end_ptr;

  /* Flipped and lower case tables. */
  const sljit_u8 *fcc;
  sljit_sw lcc;
  /* Mode can be PCRE2_JIT_COMPLETE and others. */
  int mode;
  /* TRUE, when minlength is greater than 0. */
  BOOL might_be_empty;
  /* \K is found in the pattern. */
  BOOL has_set_som;
  /* (*SKIP:arg) is found in the pattern. */
  BOOL has_skip_arg;
  /* (*THEN) is found in the pattern. */
  BOOL has_then;
  /* (*SKIP) or (*SKIP:arg) is found in lookbehind assertion. */
  BOOL has_skip_in_assert_back;
  /* Currently in recurse or negative assert. */
  BOOL local_exit;
  /* Currently in a positive assert. */
  BOOL positive_assert;
  /* Newline control. */
  int nltype;
  sljit_u32 nlmax;
  sljit_u32 nlmin;
  int newline;
  int bsr_nltype;
  sljit_u32 bsr_nlmax;
  sljit_u32 bsr_nlmin;
  /* Dollar endonly. */
  int endonly;
  /* Tables. */
  sljit_sw ctypes;
  /* Named capturing brackets. */
  PCRE2_SPTR name_table;
  sljit_sw name_count;
  sljit_sw name_entry_size;

  /* Labels and jump lists. */
  struct sljit_label *partialmatchlabel;
  struct sljit_label *quit_label;
  struct sljit_label *forced_quit_label;
  struct sljit_label *accept_label;
  struct sljit_label *ff_newline_shortcut;
  stub_list *stubs;
  label_addr_list *label_addrs;
  recurse_entry *entries;
  recurse_entry *currententry;
  jump_list *partialmatch;
  jump_list *quit;
  jump_list *positive_assert_quit;
  jump_list *forced_quit;
  jump_list *accept;
  jump_list *calllimit;
  jump_list *stackalloc;
  jump_list *revertframes;
  jump_list *wordboundary;
  jump_list *anynewline;
  jump_list *hspace;
  jump_list *vspace;
  jump_list *casefulcmp;
  jump_list *caselesscmp;
  jump_list *reset_match;
  BOOL unset_backref;
  BOOL alt_circumflex;
#ifdef SUPPORT_UNICODE
  BOOL utf;
  BOOL use_ucp;
  jump_list *getucd;
#if PCRE2_CODE_UNIT_WIDTH == 8
  jump_list *utfreadchar;
  jump_list *utfreadchar16;
  jump_list *utfreadtype8;
#endif
#endif /* SUPPORT_UNICODE */
} compiler_common;

/* For byte_sequence_compare. */

typedef struct compare_context {
  int length;
  int sourcereg;
#if defined SLJIT_UNALIGNED && SLJIT_UNALIGNED
  int ucharptr;
  union {
    sljit_s32 asint;
    sljit_u16 asushort;
#if PCRE2_CODE_UNIT_WIDTH == 8
    sljit_u8 asbyte;
    sljit_u8 asuchars[4];
#elif PCRE2_CODE_UNIT_WIDTH == 16
    sljit_u16 asuchars[2];
#elif PCRE2_CODE_UNIT_WIDTH == 32
    sljit_u32 asuchars[1];
#endif
  } c;
  union {
    sljit_s32 asint;
    sljit_u16 asushort;
#if PCRE2_CODE_UNIT_WIDTH == 8
    sljit_u8 asbyte;
    sljit_u8 asuchars[4];
#elif PCRE2_CODE_UNIT_WIDTH == 16
    sljit_u16 asuchars[2];
#elif PCRE2_CODE_UNIT_WIDTH == 32
    sljit_u32 asuchars[1];
#endif
  } oc;
#endif
} compare_context;

/* Undefine sljit macros. */
#undef CMP

/* Used for accessing the elements of the stack. */
#define STACK(i)      ((-(i) - 1) * (int)sizeof(sljit_sw))

#define TMP1          SLJIT_R0
#define TMP2          SLJIT_R2
#define TMP3          SLJIT_R3
#define STR_PTR       SLJIT_S0
#define STR_END       SLJIT_S1
#define STACK_TOP     SLJIT_R1
#define STACK_LIMIT   SLJIT_S2
#define COUNT_MATCH   SLJIT_S3
#define ARGUMENTS     SLJIT_S4
#define RETURN_ADDR   SLJIT_R4

/* Local space layout. */
/* These two locals can be used by the current opcode. */
#define LOCALS0          (0 * sizeof(sljit_sw))
#define LOCALS1          (1 * sizeof(sljit_sw))
/* Two local variables for possessive quantifiers (char1 cannot use them). */
#define POSSESSIVE0      (2 * sizeof(sljit_sw))
#define POSSESSIVE1      (3 * sizeof(sljit_sw))
/* Max limit of recursions. */
#define LIMIT_MATCH      (4 * sizeof(sljit_sw))
/* The output vector is stored on the stack, and contains pointers
to characters. The vector data is divided into two groups: the first
group contains the start / end character pointers, and the second is
the start pointers when the end of the capturing group has not yet reached. */
#define OVECTOR_START    (common->ovector_start)
#define OVECTOR(i)       (OVECTOR_START + (i) * (sljit_sw)sizeof(sljit_sw))
#define OVECTOR_PRIV(i)  (common->cbra_ptr + (i) * (sljit_sw)sizeof(sljit_sw))
#define PRIVATE_DATA(cc) (common->private_data_ptrs[(cc) - common->start])

#if PCRE2_CODE_UNIT_WIDTH == 8
#define MOV_UCHAR  SLJIT_MOV_U8
#define MOVU_UCHAR SLJIT_MOVU_U8
#define IN_UCHARS(x) (x)
#elif PCRE2_CODE_UNIT_WIDTH == 16
#define MOV_UCHAR  SLJIT_MOV_U16
#define MOVU_UCHAR SLJIT_MOVU_U16
#define UCHAR_SHIFT (1)
#define IN_UCHARS(x) ((x) * 2)
#elif PCRE2_CODE_UNIT_WIDTH == 32
#define MOV_UCHAR  SLJIT_MOV_U32
#define MOVU_UCHAR SLJIT_MOVU_U32
#define UCHAR_SHIFT (2)
#define IN_UCHARS(x) ((x) * 4)
#else
#error Unsupported compiling mode
#endif

/* Shortcuts. */
#define DEFINE_COMPILER \
  struct sljit_compiler *compiler = common->compiler
#define OP1(op, dst, dstw, src, srcw) \
  sljit_emit_op1(compiler, (op), (dst), (dstw), (src), (srcw))
#define OP2(op, dst, dstw, src1, src1w, src2, src2w) \
  sljit_emit_op2(compiler, (op), (dst), (dstw), (src1), (src1w), (src2), (src2w))
#define LABEL() \
  sljit_emit_label(compiler)
#define JUMP(type) \
  sljit_emit_jump(compiler, (type))
#define JUMPTO(type, label) \
  sljit_set_label(sljit_emit_jump(compiler, (type)), (label))
#define JUMPHERE(jump) \
  sljit_set_label((jump), sljit_emit_label(compiler))
#define SET_LABEL(jump, label) \
  sljit_set_label((jump), (label))
#define CMP(type, src1, src1w, src2, src2w) \
  sljit_emit_cmp(compiler, (type), (src1), (src1w), (src2), (src2w))
#define CMPTO(type, src1, src1w, src2, src2w, label) \
  sljit_set_label(sljit_emit_cmp(compiler, (type), (src1), (src1w), (src2), (src2w)), (label))
#define OP_FLAGS(op, dst, dstw, src, srcw, type) \
  sljit_emit_op_flags(compiler, (op), (dst), (dstw), (src), (srcw), (type))
#define GET_LOCAL_BASE(dst, dstw, offset) \
  sljit_get_local_base(compiler, (dst), (dstw), (offset))

#define READ_CHAR_MAX 0x7fffffff

static PCRE2_SPTR bracketend(PCRE2_SPTR cc)
{
SLJIT_ASSERT((*cc >= OP_ASSERT && *cc <= OP_ASSERTBACK_NOT) || (*cc >= OP_ONCE && *cc <= OP_SCOND));
do cc += GET(cc, 1); while (*cc == OP_ALT);
SLJIT_ASSERT(*cc >= OP_KET && *cc <= OP_KETRPOS);
cc += 1 + LINK_SIZE;
return cc;
}

static int no_alternatives(PCRE2_SPTR cc)
{
int count = 0;
SLJIT_ASSERT((*cc >= OP_ASSERT && *cc <= OP_ASSERTBACK_NOT) || (*cc >= OP_ONCE && *cc <= OP_SCOND));
do
  {
  cc += GET(cc, 1);
  count++;
  }
while (*cc == OP_ALT);
SLJIT_ASSERT(*cc >= OP_KET && *cc <= OP_KETRPOS);
return count;
}

/* Functions whose might need modification for all new supported opcodes:
 next_opcode
 check_opcode_types
 set_private_data_ptrs
 get_framesize
 init_frame
 get_private_data_copy_length
 copy_private_data
 compile_matchingpath
 compile_backtrackingpath
*/

static PCRE2_SPTR next_opcode(compiler_common *common, PCRE2_SPTR cc)
{
SLJIT_UNUSED_ARG(common);
switch(*cc)
  {
  case OP_SOD:
  case OP_SOM:
  case OP_SET_SOM:
  case OP_NOT_WORD_BOUNDARY:
  case OP_WORD_BOUNDARY:
  case OP_NOT_DIGIT:
  case OP_DIGIT:
  case OP_NOT_WHITESPACE:
  case OP_WHITESPACE:
  case OP_NOT_WORDCHAR:
  case OP_WORDCHAR:
  case OP_ANY:
  case OP_ALLANY:
  case OP_NOTPROP:
  case OP_PROP:
  case OP_ANYNL:
  case OP_NOT_HSPACE:
  case OP_HSPACE:
  case OP_NOT_VSPACE:
  case OP_VSPACE:
  case OP_EXTUNI:
  case OP_EODN:
  case OP_EOD:
  case OP_CIRC:
  case OP_CIRCM:
  case OP_DOLL:
  case OP_DOLLM:
  case OP_CRSTAR:
  case OP_CRMINSTAR:
  case OP_CRPLUS:
  case OP_CRMINPLUS:
  case OP_CRQUERY:
  case OP_CRMINQUERY:
  case OP_CRRANGE:
  case OP_CRMINRANGE:
  case OP_CRPOSSTAR:
  case OP_CRPOSPLUS:
  case OP_CRPOSQUERY:
  case OP_CRPOSRANGE:
  case OP_CLASS:
  case OP_NCLASS:
  case OP_REF:
  case OP_REFI:
  case OP_DNREF:
  case OP_DNREFI:
  case OP_RECURSE:
  case OP_CALLOUT:
  case OP_ALT:
  case OP_KET:
  case OP_KETRMAX:
  case OP_KETRMIN:
  case OP_KETRPOS:
  case OP_REVERSE:
  case OP_ASSERT:
  case OP_ASSERT_NOT:
  case OP_ASSERTBACK:
  case OP_ASSERTBACK_NOT:
  case OP_ONCE:
  case OP_ONCE_NC:
  case OP_BRA:
  case OP_BRAPOS:
  case OP_CBRA:
  case OP_CBRAPOS:
  case OP_COND:
  case OP_SBRA:
  case OP_SBRAPOS:
  case OP_SCBRA:
  case OP_SCBRAPOS:
  case OP_SCOND:
  case OP_CREF:
  case OP_DNCREF:
  case OP_RREF:
  case OP_DNRREF:
  case OP_FALSE:
  case OP_TRUE:
  case OP_BRAZERO:
  case OP_BRAMINZERO:
  case OP_BRAPOSZERO:
  case OP_PRUNE:
  case OP_SKIP:
  case OP_THEN:
  case OP_COMMIT:
  case OP_FAIL:
  case OP_ACCEPT:
  case OP_ASSERT_ACCEPT:
  case OP_CLOSE:
  case OP_SKIPZERO:
  return cc + PRIV(OP_lengths)[*cc];

  case OP_CHAR:
  case OP_CHARI:
  case OP_NOT:
  case OP_NOTI:
  case OP_STAR:
  case OP_MINSTAR:
  case OP_PLUS:
  case OP_MINPLUS:
  case OP_QUERY:
  case OP_MINQUERY:
  case OP_UPTO:
  case OP_MINUPTO:
  case OP_EXACT:
  case OP_POSSTAR:
  case OP_POSPLUS:
  case OP_POSQUERY:
  case OP_POSUPTO:
  case OP_STARI:
  case OP_MINSTARI:
  case OP_PLUSI:
  case OP_MINPLUSI:
  case OP_QUERYI:
  case OP_MINQUERYI:
  case OP_UPTOI:
  case OP_MINUPTOI:
  case OP_EXACTI:
  case OP_POSSTARI:
  case OP_POSPLUSI:
  case OP_POSQUERYI:
  case OP_POSUPTOI:
  case OP_NOTSTAR:
  case OP_NOTMINSTAR:
  case OP_NOTPLUS:
  case OP_NOTMINPLUS:
  case OP_NOTQUERY:
  case OP_NOTMINQUERY:
  case OP_NOTUPTO:
  case OP_NOTMINUPTO:
  case OP_NOTEXACT:
  case OP_NOTPOSSTAR:
  case OP_NOTPOSPLUS:
  case OP_NOTPOSQUERY:
  case OP_NOTPOSUPTO:
  case OP_NOTSTARI:
  case OP_NOTMINSTARI:
  case OP_NOTPLUSI:
  case OP_NOTMINPLUSI:
  case OP_NOTQUERYI:
  case OP_NOTMINQUERYI:
  case OP_NOTUPTOI:
  case OP_NOTMINUPTOI:
  case OP_NOTEXACTI:
  case OP_NOTPOSSTARI:
  case OP_NOTPOSPLUSI:
  case OP_NOTPOSQUERYI:
  case OP_NOTPOSUPTOI:
  cc += PRIV(OP_lengths)[*cc];
#ifdef SUPPORT_UNICODE
  if (common->utf && HAS_EXTRALEN(cc[-1])) cc += GET_EXTRALEN(cc[-1]);
#endif
  return cc;

  /* Special cases. */
  case OP_TYPESTAR:
  case OP_TYPEMINSTAR:
  case OP_TYPEPLUS:
  case OP_TYPEMINPLUS:
  case OP_TYPEQUERY:
  case OP_TYPEMINQUERY:
  case OP_TYPEUPTO:
  case OP_TYPEMINUPTO:
  case OP_TYPEEXACT:
  case OP_TYPEPOSSTAR:
  case OP_TYPEPOSPLUS:
  case OP_TYPEPOSQUERY:
  case OP_TYPEPOSUPTO:
  return cc + PRIV(OP_lengths)[*cc] - 1;

  case OP_ANYBYTE:
#ifdef SUPPORT_UNICODE
  if (common->utf) return NULL;
#endif
  return cc + 1;

  case OP_CALLOUT_STR:
  return cc + GET(cc, 1 + 2*LINK_SIZE);

#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH != 8
  case OP_XCLASS:
  return cc + GET(cc, 1);
#endif

  case OP_MARK:
  case OP_PRUNE_ARG:
  case OP_SKIP_ARG:
  case OP_THEN_ARG:
  return cc + 1 + 2 + cc[1];

  default:
  /* All opcodes are supported now! */
  SLJIT_ASSERT_STOP();
  return NULL;
  }
}

static BOOL check_opcode_types(compiler_common *common, PCRE2_SPTR cc, PCRE2_SPTR ccend)
{
int count;
PCRE2_SPTR slot;
PCRE2_SPTR assert_back_end = cc - 1;

/* Calculate important variables (like stack size) and checks whether all opcodes are supported. */
while (cc < ccend)
  {
  switch(*cc)
    {
    case OP_SET_SOM:
    common->has_set_som = TRUE;
    common->might_be_empty = TRUE;
    cc += 1;
    break;

    case OP_REF:
    case OP_REFI:
    common->optimized_cbracket[GET2(cc, 1)] = 0;
    cc += 1 + IMM2_SIZE;
    break;

    case OP_CBRAPOS:
    case OP_SCBRAPOS:
    common->optimized_cbracket[GET2(cc, 1 + LINK_SIZE)] = 0;
    cc += 1 + LINK_SIZE + IMM2_SIZE;
    break;

    case OP_COND:
    case OP_SCOND:
    /* Only AUTO_CALLOUT can insert this opcode. We do
       not intend to support this case. */
    if (cc[1 + LINK_SIZE] == OP_CALLOUT || cc[1 + LINK_SIZE] == OP_CALLOUT_STR)
      return FALSE;
    cc += 1 + LINK_SIZE;
    break;

    case OP_CREF:
    common->optimized_cbracket[GET2(cc, 1)] = 0;
    cc += 1 + IMM2_SIZE;
    break;

    case OP_DNREF:
    case OP_DNREFI:
    case OP_DNCREF:
    count = GET2(cc, 1 + IMM2_SIZE);
    slot = common->name_table + GET2(cc, 1) * common->name_entry_size;
    while (count-- > 0)
      {
      common->optimized_cbracket[GET2(slot, 0)] = 0;
      slot += common->name_entry_size;
      }
    cc += 1 + 2 * IMM2_SIZE;
    break;

    case OP_RECURSE:
    /* Set its value only once. */
    if (common->recursive_head_ptr == 0)
      {
      common->recursive_head_ptr = common->ovector_start;
      common->ovector_start += sizeof(sljit_sw);
      }
    cc += 1 + LINK_SIZE;
    break;

    case OP_CALLOUT:
    case OP_CALLOUT_STR:
    if (common->capture_last_ptr == 0)
      {
      common->capture_last_ptr = common->ovector_start;
      common->ovector_start += sizeof(sljit_sw);
      }
    cc += (*cc == OP_CALLOUT) ? PRIV(OP_lengths)[OP_CALLOUT] : GET(cc, 1 + 2*LINK_SIZE);
    break;

    case OP_ASSERTBACK:
    slot = bracketend(cc);
    if (slot > assert_back_end)
      assert_back_end = slot;
    cc += 1 + LINK_SIZE;
    break;

    case OP_THEN_ARG:
    common->has_then = TRUE;
    common->control_head_ptr = 1;
    /* Fall through. */

    case OP_PRUNE_ARG:
    case OP_MARK:
    if (common->mark_ptr == 0)
      {
      common->mark_ptr = common->ovector_start;
      common->ovector_start += sizeof(sljit_sw);
      }
    cc += 1 + 2 + cc[1];
    break;

    case OP_THEN:
    common->has_then = TRUE;
    common->control_head_ptr = 1;
    cc += 1;
    break;

    case OP_SKIP:
    if (cc < assert_back_end)
      common->has_skip_in_assert_back = TRUE;
    cc += 1;
    break;

    case OP_SKIP_ARG:
    common->control_head_ptr = 1;
    common->has_skip_arg = TRUE;
    if (cc < assert_back_end)
      common->has_skip_in_assert_back = TRUE;
    cc += 1 + 2 + cc[1];
    break;

    default:
    cc = next_opcode(common, cc);
    if (cc == NULL)
      return FALSE;
    break;
    }
  }
return TRUE;
}

static BOOL is_accelerated_repeat(PCRE2_SPTR cc)
{
switch(*cc)
  {
  case OP_TYPESTAR:
  case OP_TYPEMINSTAR:
  case OP_TYPEPLUS:
  case OP_TYPEMINPLUS:
  case OP_TYPEPOSSTAR:
  case OP_TYPEPOSPLUS:
  return (cc[1] != OP_ANYNL && cc[1] != OP_EXTUNI);

  case OP_STAR:
  case OP_MINSTAR:
  case OP_PLUS:
  case OP_MINPLUS:
  case OP_POSSTAR:
  case OP_POSPLUS:

  case OP_STARI:
  case OP_MINSTARI:
  case OP_PLUSI:
  case OP_MINPLUSI:
  case OP_POSSTARI:
  case OP_POSPLUSI:

  case OP_NOTSTAR:
  case OP_NOTMINSTAR:
  case OP_NOTPLUS:
  case OP_NOTMINPLUS:
  case OP_NOTPOSSTAR:
  case OP_NOTPOSPLUS:

  case OP_NOTSTARI:
  case OP_NOTMINSTARI:
  case OP_NOTPLUSI:
  case OP_NOTMINPLUSI:
  case OP_NOTPOSSTARI:
  case OP_NOTPOSPLUSI:
  return TRUE;

  case OP_CLASS:
  case OP_NCLASS:
#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH != 8
  case OP_XCLASS:
  cc += (*cc == OP_XCLASS) ? GET(cc, 1) : (int)(1 + (32 / sizeof(PCRE2_UCHAR)));
#else
  cc += (1 + (32 / sizeof(PCRE2_UCHAR)));
#endif

  switch(*cc)
    {
    case OP_CRSTAR:
    case OP_CRMINSTAR:
    case OP_CRPLUS:
    case OP_CRMINPLUS:
    case OP_CRPOSSTAR:
    case OP_CRPOSPLUS:
    return TRUE;
    }
  break;
  }
return FALSE;
}

static SLJIT_INLINE BOOL detect_fast_forward_skip(compiler_common *common, int *private_data_start)
{
PCRE2_SPTR cc = common->start;
PCRE2_SPTR end;

/* Skip not repeated brackets. */
while (TRUE)
  {
  switch(*cc)
    {
    case OP_SOD:
    case OP_SOM:
    case OP_SET_SOM:
    case OP_NOT_WORD_BOUNDARY:
    case OP_WORD_BOUNDARY:
    case OP_EODN:
    case OP_EOD:
    case OP_CIRC:
    case OP_CIRCM:
    case OP_DOLL:
    case OP_DOLLM:
    /* Zero width assertions. */
    cc++;
    continue;
    }

  if (*cc != OP_BRA && *cc != OP_CBRA)
    break;

  end = cc + GET(cc, 1);
  if (*end != OP_KET || PRIVATE_DATA(end) != 0)
    return FALSE;
  if (*cc == OP_CBRA)
    {
    if (common->optimized_cbracket[GET2(cc, 1 + LINK_SIZE)] == 0)
      return FALSE;
    cc += IMM2_SIZE;
    }
  cc += 1 + LINK_SIZE;
  }

if (is_accelerated_repeat(cc))
  {
  common->fast_forward_bc_ptr = cc;
  common->private_data_ptrs[(cc + 1) - common->start] = *private_data_start;
  *private_data_start += sizeof(sljit_sw);
  return TRUE;
  }
return FALSE;
}

static SLJIT_INLINE void detect_fast_fail(compiler_common *common, PCRE2_SPTR cc, int *private_data_start, sljit_s32 depth)
{
  PCRE2_SPTR next_alt;

  SLJIT_ASSERT(*cc == OP_BRA || *cc == OP_CBRA);

  if (*cc == OP_CBRA && common->optimized_cbracket[GET2(cc, 1 + LINK_SIZE)] == 0)
    return;

  next_alt = bracketend(cc) - (1 + LINK_SIZE);
  if (*next_alt != OP_KET || PRIVATE_DATA(next_alt) != 0)
    return;

  do
    {
    next_alt = cc + GET(cc, 1);

    cc += 1 + LINK_SIZE + ((*cc == OP_CBRA) ? IMM2_SIZE : 0);

    while (TRUE)
      {
      switch(*cc)
        {
        case OP_SOD:
        case OP_SOM:
        case OP_SET_SOM:
        case OP_NOT_WORD_BOUNDARY:
        case OP_WORD_BOUNDARY:
        case OP_EODN:
        case OP_EOD:
        case OP_CIRC:
        case OP_CIRCM:
        case OP_DOLL:
        case OP_DOLLM:
        /* Zero width assertions. */
        cc++;
        continue;
        }
      break;
      }

    if (depth > 0 && (*cc == OP_BRA || *cc == OP_CBRA))
      detect_fast_fail(common, cc, private_data_start, depth - 1);

    if (is_accelerated_repeat(cc))
      {
      common->private_data_ptrs[(cc + 1) - common->start] = *private_data_start;

      if (common->fast_fail_start_ptr == 0)
        common->fast_fail_start_ptr = *private_data_start;

      *private_data_start += sizeof(sljit_sw);
      common->fast_fail_end_ptr = *private_data_start;

      if (*private_data_start > SLJIT_MAX_LOCAL_SIZE)
        return;
      }

    cc = next_alt;
    }
  while (*cc == OP_ALT);
}

static int get_class_iterator_size(PCRE2_SPTR cc)
{
sljit_u32 min;
sljit_u32 max;
switch(*cc)
  {
  case OP_CRSTAR:
  case OP_CRPLUS:
  return 2;

  case OP_CRMINSTAR:
  case OP_CRMINPLUS:
  case OP_CRQUERY:
  case OP_CRMINQUERY:
  return 1;

  case OP_CRRANGE:
  case OP_CRMINRANGE:
  min = GET2(cc, 1);
  max = GET2(cc, 1 + IMM2_SIZE);
  if (max == 0)
    return (*cc == OP_CRRANGE) ? 2 : 1;
  max -= min;
  if (max > 2)
    max = 2;
  return max;

  default:
  return 0;
  }
}

static BOOL detect_repeat(compiler_common *common, PCRE2_SPTR begin)
{
PCRE2_SPTR end = bracketend(begin);
PCRE2_SPTR next;
PCRE2_SPTR next_end;
PCRE2_SPTR max_end;
PCRE2_UCHAR type;
sljit_sw length = end - begin;
sljit_s32 min, max, i;

/* Detect fixed iterations first. */
if (end[-(1 + LINK_SIZE)] != OP_KET)
  return FALSE;

/* Already detected repeat. */
if (common->private_data_ptrs[end - common->start - LINK_SIZE] != 0)
  return TRUE;

next = end;
min = 1;
while (1)
  {
  if (*next != *begin)
    break;
  next_end = bracketend(next);
  if (next_end - next != length || memcmp(begin, next, IN_UCHARS(length)) != 0)
    break;
  next = next_end;
  min++;
  }

if (min == 2)
  return FALSE;

max = 0;
max_end = next;
if (*next == OP_BRAZERO || *next == OP_BRAMINZERO)
  {
  type = *next;
  while (1)
    {
    if (next[0] != type || next[1] != OP_BRA || next[2 + LINK_SIZE] != *begin)
      break;
    next_end = bracketend(next + 2 + LINK_SIZE);
    if (next_end - next != (length + 2 + LINK_SIZE) || memcmp(begin, next + 2 + LINK_SIZE, IN_UCHARS(length)) != 0)
      break;
    next = next_end;
    max++;
    }

  if (next[0] == type && next[1] == *begin && max >= 1)
    {
    next_end = bracketend(next + 1);
    if (next_end - next == (length + 1) && memcmp(begin, next + 1, IN_UCHARS(length)) == 0)
      {
      for (i = 0; i < max; i++, next_end += 1 + LINK_SIZE)
        if (*next_end != OP_KET)
          break;

      if (i == max)
        {
        common->private_data_ptrs[max_end - common->start - LINK_SIZE] = next_end - max_end;
        common->private_data_ptrs[max_end - common->start - LINK_SIZE + 1] = (type == OP_BRAZERO) ? OP_UPTO : OP_MINUPTO;
        /* +2 the original and the last. */
        common->private_data_ptrs[max_end - common->start - LINK_SIZE + 2] = max + 2;
        if (min == 1)
          return TRUE;
        min--;
        max_end -= (1 + LINK_SIZE) + GET(max_end, -LINK_SIZE);
        }
      }
    }
  }

if (min >= 3)
  {
  common->private_data_ptrs[end - common->start - LINK_SIZE] = max_end - end;
  common->private_data_ptrs[end - common->start - LINK_SIZE + 1] = OP_EXACT;
  common->private_data_ptrs[end - common->start - LINK_SIZE + 2] = min;
  return TRUE;
  }

return FALSE;
}

#define CASE_ITERATOR_PRIVATE_DATA_1 \
    case OP_MINSTAR: \
    case OP_MINPLUS: \
    case OP_QUERY: \
    case OP_MINQUERY: \
    case OP_MINSTARI: \
    case OP_MINPLUSI: \
    case OP_QUERYI: \
    case OP_MINQUERYI: \
    case OP_NOTMINSTAR: \
    case OP_NOTMINPLUS: \
    case OP_NOTQUERY: \
    case OP_NOTMINQUERY: \
    case OP_NOTMINSTARI: \
    case OP_NOTMINPLUSI: \
    case OP_NOTQUERYI: \
    case OP_NOTMINQUERYI:

#define CASE_ITERATOR_PRIVATE_DATA_2A \
    case OP_STAR: \
    case OP_PLUS: \
    case OP_STARI: \
    case OP_PLUSI: \
    case OP_NOTSTAR: \
    case OP_NOTPLUS: \
    case OP_NOTSTARI: \
    case OP_NOTPLUSI:

#define CASE_ITERATOR_PRIVATE_DATA_2B \
    case OP_UPTO: \
    case OP_MINUPTO: \
    case OP_UPTOI: \
    case OP_MINUPTOI: \
    case OP_NOTUPTO: \
    case OP_NOTMINUPTO: \
    case OP_NOTUPTOI: \
    case OP_NOTMINUPTOI:

#define CASE_ITERATOR_TYPE_PRIVATE_DATA_1 \
    case OP_TYPEMINSTAR: \
    case OP_TYPEMINPLUS: \
    case OP_TYPEQUERY: \
    case OP_TYPEMINQUERY:

#define CASE_ITERATOR_TYPE_PRIVATE_DATA_2A \
    case OP_TYPESTAR: \
    case OP_TYPEPLUS:

#define CASE_ITERATOR_TYPE_PRIVATE_DATA_2B \
    case OP_TYPEUPTO: \
    case OP_TYPEMINUPTO:

static void set_private_data_ptrs(compiler_common *common, int *private_data_start, PCRE2_SPTR ccend)
{
PCRE2_SPTR cc = common->start;
PCRE2_SPTR alternative;
PCRE2_SPTR end = NULL;
int private_data_ptr = *private_data_start;
int space, size, bracketlen;
BOOL repeat_check = TRUE;

while (cc < ccend)
  {
  space = 0;
  size = 0;
  bracketlen = 0;
  if (private_data_ptr > SLJIT_MAX_LOCAL_SIZE)
    break;

  if (repeat_check && (*cc == OP_ONCE || *cc == OP_ONCE_NC || *cc == OP_BRA || *cc == OP_CBRA || *cc == OP_COND))
    {
    if (detect_repeat(common, cc))
      {
      /* These brackets are converted to repeats, so no global
      based single character repeat is allowed. */
      if (cc >= end)
        end = bracketend(cc);
      }
    }
  repeat_check = TRUE;

  switch(*cc)
    {
    case OP_KET:
    if (common->private_data_ptrs[cc + 1 - common->start] != 0)
      {
      common->private_data_ptrs[cc - common->start] = private_data_ptr;
      private_data_ptr += sizeof(sljit_sw);
      cc += common->private_data_ptrs[cc + 1 - common->start];
      }
    cc += 1 + LINK_SIZE;
    break;

    case OP_ASSERT:
    case OP_ASSERT_NOT:
    case OP_ASSERTBACK:
    case OP_ASSERTBACK_NOT:
    case OP_ONCE:
    case OP_ONCE_NC:
    case OP_BRAPOS:
    case OP_SBRA:
    case OP_SBRAPOS:
    case OP_SCOND:
    common->private_data_ptrs[cc - common->start] = private_data_ptr;
    private_data_ptr += sizeof(sljit_sw);
    bracketlen = 1 + LINK_SIZE;
    break;

    case OP_CBRAPOS:
    case OP_SCBRAPOS:
    common->private_data_ptrs[cc - common->start] = private_data_ptr;
    private_data_ptr += sizeof(sljit_sw);
    bracketlen = 1 + LINK_SIZE + IMM2_SIZE;
    break;

    case OP_COND:
    /* Might be a hidden SCOND. */
    alternative = cc + GET(cc, 1);
    if (*alternative == OP_KETRMAX || *alternative == OP_KETRMIN)
      {
      common->private_data_ptrs[cc - common->start] = private_data_ptr;
      private_data_ptr += sizeof(sljit_sw);
      }
    bracketlen = 1 + LINK_SIZE;
    break;

    case OP_BRA:
    bracketlen = 1 + LINK_SIZE;
    break;

    case OP_CBRA:
    case OP_SCBRA:
    bracketlen = 1 + LINK_SIZE + IMM2_SIZE;
    break;

    case OP_BRAZERO:
    case OP_BRAMINZERO:
    case OP_BRAPOSZERO:
    repeat_check = FALSE;
    size = 1;
    break;

    CASE_ITERATOR_PRIVATE_DATA_1
    space = 1;
    size = -2;
    break;

    CASE_ITERATOR_PRIVATE_DATA_2A
    space = 2;
    size = -2;
    break;

    CASE_ITERATOR_PRIVATE_DATA_2B
    space = 2;
    size = -(2 + IMM2_SIZE);
    break;

    CASE_ITERATOR_TYPE_PRIVATE_DATA_1
    space = 1;
    size = 1;
    break;

    CASE_ITERATOR_TYPE_PRIVATE_DATA_2A
    if (cc[1] != OP_ANYNL && cc[1] != OP_EXTUNI)
      space = 2;
    size = 1;
    break;

    case OP_TYPEUPTO:
    if (cc[1 + IMM2_SIZE] != OP_ANYNL && cc[1 + IMM2_SIZE] != OP_EXTUNI)
      space = 2;
    size = 1 + IMM2_SIZE;
    break;

    case OP_TYPEMINUPTO:
    space = 2;
    size = 1 + IMM2_SIZE;
    break;

    case OP_CLASS:
    case OP_NCLASS:
    space = get_class_iterator_size(cc + size);
    size = 1 + 32 / sizeof(PCRE2_UCHAR);
    break;

#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH != 8
    case OP_XCLASS:
    space = get_class_iterator_size(cc + size);
    size = GET(cc, 1);
    break;
#endif

    default:
    cc = next_opcode(common, cc);
    SLJIT_ASSERT(cc != NULL);
    break;
    }

  /* Character iterators, which are not inside a repeated bracket,
     gets a private slot instead of allocating it on the stack. */
  if (space > 0 && cc >= end)
    {
    common->private_data_ptrs[cc - common->start] = private_data_ptr;
    private_data_ptr += sizeof(sljit_sw) * space;
    }

  if (size != 0)
    {
    if (size < 0)
      {
      cc += -size;
#ifdef SUPPORT_UNICODE
      if (common->utf && HAS_EXTRALEN(cc[-1])) cc += GET_EXTRALEN(cc[-1]);
#endif
      }
    else
      cc += size;
    }

  if (bracketlen > 0)
    {
    if (cc >= end)
      {
      end = bracketend(cc);
      if (end[-1 - LINK_SIZE] == OP_KET)
        end = NULL;
      }
    cc += bracketlen;
    }
  }
*private_data_start = private_data_ptr;
}

/* Returns with a frame_types (always < 0) if no need for frame. */
static int get_framesize(compiler_common *common, PCRE2_SPTR cc, PCRE2_SPTR ccend, BOOL recursive, BOOL *needs_control_head)
{
int length = 0;
int possessive = 0;
BOOL stack_restore = FALSE;
BOOL setsom_found = recursive;
BOOL setmark_found = recursive;
/* The last capture is a local variable even for recursions. */
BOOL capture_last_found = FALSE;

#if defined DEBUG_FORCE_CONTROL_HEAD && DEBUG_FORCE_CONTROL_HEAD
SLJIT_ASSERT(common->control_head_ptr != 0);
*needs_control_head = TRUE;
#else
*needs_control_head = FALSE;
#endif

if (ccend == NULL)
  {
  ccend = bracketend(cc) - (1 + LINK_SIZE);
  if (!recursive && (*cc == OP_CBRAPOS || *cc == OP_SCBRAPOS))
    {
    possessive = length = (common->capture_last_ptr != 0) ? 5 : 3;
    /* This is correct regardless of common->capture_last_ptr. */
    capture_last_found = TRUE;
    }
  cc = next_opcode(common, cc);
  }

SLJIT_ASSERT(cc != NULL);
while (cc < ccend)
  switch(*cc)
    {
    case OP_SET_SOM:
    SLJIT_ASSERT(common->has_set_som);
    stack_restore = TRUE;
    if (!setsom_found)
      {
      length += 2;
      setsom_found = TRUE;
      }
    cc += 1;
    break;

    case OP_MARK:
    case OP_PRUNE_ARG:
    case OP_THEN_ARG:
    SLJIT_ASSERT(common->mark_ptr != 0);
    stack_restore = TRUE;
    if (!setmark_found)
      {
      length += 2;
      setmark_found = TRUE;
      }
    if (common->control_head_ptr != 0)
      *needs_control_head = TRUE;
    cc += 1 + 2 + cc[1];
    break;

    case OP_RECURSE:
    stack_restore = TRUE;
    if (common->has_set_som && !setsom_found)
      {
      length += 2;
      setsom_found = TRUE;
      }
    if (common->mark_ptr != 0 && !setmark_found)
      {
      length += 2;
      setmark_found = TRUE;
      }
    if (common->capture_last_ptr != 0 && !capture_last_found)
      {
      length += 2;
      capture_last_found = TRUE;
      }
    cc += 1 + LINK_SIZE;
    break;

    case OP_CBRA:
    case OP_CBRAPOS:
    case OP_SCBRA:
    case OP_SCBRAPOS:
    stack_restore = TRUE;
    if (common->capture_last_ptr != 0 && !capture_last_found)
      {
      length += 2;
      capture_last_found = TRUE;
      }
    length += 3;
    cc += 1 + LINK_SIZE + IMM2_SIZE;
    break;

    case OP_THEN:
    stack_restore = TRUE;
    if (common->control_head_ptr != 0)
      *needs_control_head = TRUE;
    cc ++;
    break;

    default:
    stack_restore = TRUE;
    /* Fall through. */

    case OP_NOT_WORD_BOUNDARY:
    case OP_WORD_BOUNDARY:
    case OP_NOT_DIGIT:
    case OP_DIGIT:
    case OP_NOT_WHITESPACE:
    case OP_WHITESPACE:
    case OP_NOT_WORDCHAR:
    case OP_WORDCHAR:
    case OP_ANY:
    case OP_ALLANY:
    case OP_ANYBYTE:
    case OP_NOTPROP:
    case OP_PROP:
    case OP_ANYNL:
    case OP_NOT_HSPACE:
    case OP_HSPACE:
    case OP_NOT_VSPACE:
    case OP_VSPACE:
    case OP_EXTUNI:
    case OP_EODN:
    case OP_EOD:
    case OP_CIRC:
    case OP_CIRCM:
    case OP_DOLL:
    case OP_DOLLM:
    case OP_CHAR:
    case OP_CHARI:
    case OP_NOT:
    case OP_NOTI:

    case OP_EXACT:
    case OP_POSSTAR:
    case OP_POSPLUS:
    case OP_POSQUERY:
    case OP_POSUPTO:

    case OP_EXACTI:
    case OP_POSSTARI:
    case OP_POSPLUSI:
    case OP_POSQUERYI:
    case OP_POSUPTOI:

    case OP_NOTEXACT:
    case OP_NOTPOSSTAR:
    case OP_NOTPOSPLUS:
    case OP_NOTPOSQUERY:
    case OP_NOTPOSUPTO:

    case OP_NOTEXACTI:
    case OP_NOTPOSSTARI:
    case OP_NOTPOSPLUSI:
    case OP_NOTPOSQUERYI:
    case OP_NOTPOSUPTOI:

    case OP_TYPEEXACT:
    case OP_TYPEPOSSTAR:
    case OP_TYPEPOSPLUS:
    case OP_TYPEPOSQUERY:
    case OP_TYPEPOSUPTO:

    case OP_CLASS:
    case OP_NCLASS:
    case OP_XCLASS:

    case OP_CALLOUT:
    case OP_CALLOUT_STR:

    cc = next_opcode(common, cc);
    SLJIT_ASSERT(cc != NULL);
    break;
    }

/* Possessive quantifiers can use a special case. */
if (SLJIT_UNLIKELY(possessive == length))
  return stack_restore ? no_frame : no_stack;

if (length > 0)
  return length + 1;
return stack_restore ? no_frame : no_stack;
}

static void init_frame(compiler_common *common, PCRE2_SPTR cc, PCRE2_SPTR ccend, int stackpos, int stacktop, BOOL recursive)
{
DEFINE_COMPILER;
BOOL setsom_found = recursive;
BOOL setmark_found = recursive;
/* The last capture is a local variable even for recursions. */
BOOL capture_last_found = FALSE;
int offset;

/* >= 1 + shortest item size (2) */
SLJIT_UNUSED_ARG(stacktop);
SLJIT_ASSERT(stackpos >= stacktop + 2);

stackpos = STACK(stackpos);
if (ccend == NULL)
  {
  ccend = bracketend(cc) - (1 + LINK_SIZE);
  if (recursive || (*cc != OP_CBRAPOS && *cc != OP_SCBRAPOS))
    cc = next_opcode(common, cc);
  }

SLJIT_ASSERT(cc != NULL);
while (cc < ccend)
  switch(*cc)
    {
    case OP_SET_SOM:
    SLJIT_ASSERT(common->has_set_som);
    if (!setsom_found)
      {
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(0));
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, SLJIT_IMM, -OVECTOR(0));
      stackpos += (int)sizeof(sljit_sw);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, TMP1, 0);
      stackpos += (int)sizeof(sljit_sw);
      setsom_found = TRUE;
      }
    cc += 1;
    break;

    case OP_MARK:
    case OP_PRUNE_ARG:
    case OP_THEN_ARG:
    SLJIT_ASSERT(common->mark_ptr != 0);
    if (!setmark_found)
      {
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->mark_ptr);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, SLJIT_IMM, -common->mark_ptr);
      stackpos += (int)sizeof(sljit_sw);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, TMP1, 0);
      stackpos += (int)sizeof(sljit_sw);
      setmark_found = TRUE;
      }
    cc += 1 + 2 + cc[1];
    break;

    case OP_RECURSE:
    if (common->has_set_som && !setsom_found)
      {
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(0));
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, SLJIT_IMM, -OVECTOR(0));
      stackpos += (int)sizeof(sljit_sw);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, TMP1, 0);
      stackpos += (int)sizeof(sljit_sw);
      setsom_found = TRUE;
      }
    if (common->mark_ptr != 0 && !setmark_found)
      {
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->mark_ptr);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, SLJIT_IMM, -common->mark_ptr);
      stackpos += (int)sizeof(sljit_sw);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, TMP1, 0);
      stackpos += (int)sizeof(sljit_sw);
      setmark_found = TRUE;
      }
    if (common->capture_last_ptr != 0 && !capture_last_found)
      {
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->capture_last_ptr);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, SLJIT_IMM, -common->capture_last_ptr);
      stackpos += (int)sizeof(sljit_sw);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, TMP1, 0);
      stackpos += (int)sizeof(sljit_sw);
      capture_last_found = TRUE;
      }
    cc += 1 + LINK_SIZE;
    break;

    case OP_CBRA:
    case OP_CBRAPOS:
    case OP_SCBRA:
    case OP_SCBRAPOS:
    if (common->capture_last_ptr != 0 && !capture_last_found)
      {
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->capture_last_ptr);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, SLJIT_IMM, -common->capture_last_ptr);
      stackpos += (int)sizeof(sljit_sw);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, TMP1, 0);
      stackpos += (int)sizeof(sljit_sw);
      capture_last_found = TRUE;
      }
    offset = (GET2(cc, 1 + LINK_SIZE)) << 1;
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, SLJIT_IMM, OVECTOR(offset));
    stackpos += (int)sizeof(sljit_sw);
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset));
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1));
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, TMP1, 0);
    stackpos += (int)sizeof(sljit_sw);
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, TMP2, 0);
    stackpos += (int)sizeof(sljit_sw);

    cc += 1 + LINK_SIZE + IMM2_SIZE;
    break;

    default:
    cc = next_opcode(common, cc);
    SLJIT_ASSERT(cc != NULL);
    break;
    }

OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, SLJIT_IMM, 0);
SLJIT_ASSERT(stackpos == STACK(stacktop));
}

static SLJIT_INLINE int get_private_data_copy_length(compiler_common *common, PCRE2_SPTR cc, PCRE2_SPTR ccend, BOOL needs_control_head)
{
int private_data_length = needs_control_head ? 3 : 2;
int size;
PCRE2_SPTR alternative;
/* Calculate the sum of the private machine words. */
while (cc < ccend)
  {
  size = 0;
  switch(*cc)
    {
    case OP_KET:
    if (PRIVATE_DATA(cc) != 0)
      {
      private_data_length++;
      SLJIT_ASSERT(PRIVATE_DATA(cc + 1) != 0);
      cc += PRIVATE_DATA(cc + 1);
      }
    cc += 1 + LINK_SIZE;
    break;

    case OP_ASSERT:
    case OP_ASSERT_NOT:
    case OP_ASSERTBACK:
    case OP_ASSERTBACK_NOT:
    case OP_ONCE:
    case OP_ONCE_NC:
    case OP_BRAPOS:
    case OP_SBRA:
    case OP_SBRAPOS:
    case OP_SCOND:
    private_data_length++;
    SLJIT_ASSERT(PRIVATE_DATA(cc) != 0);
    cc += 1 + LINK_SIZE;
    break;

    case OP_CBRA:
    case OP_SCBRA:
    if (common->optimized_cbracket[GET2(cc, 1 + LINK_SIZE)] == 0)
      private_data_length++;
    cc += 1 + LINK_SIZE + IMM2_SIZE;
    break;

    case OP_CBRAPOS:
    case OP_SCBRAPOS:
    private_data_length += 2;
    cc += 1 + LINK_SIZE + IMM2_SIZE;
    break;

    case OP_COND:
    /* Might be a hidden SCOND. */
    alternative = cc + GET(cc, 1);
    if (*alternative == OP_KETRMAX || *alternative == OP_KETRMIN)
      private_data_length++;
    cc += 1 + LINK_SIZE;
    break;

    CASE_ITERATOR_PRIVATE_DATA_1
    if (PRIVATE_DATA(cc))
      private_data_length++;
    cc += 2;
#ifdef SUPPORT_UNICODE
    if (common->utf && HAS_EXTRALEN(cc[-1])) cc += GET_EXTRALEN(cc[-1]);
#endif
    break;

    CASE_ITERATOR_PRIVATE_DATA_2A
    if (PRIVATE_DATA(cc))
      private_data_length += 2;
    cc += 2;
#ifdef SUPPORT_UNICODE
    if (common->utf && HAS_EXTRALEN(cc[-1])) cc += GET_EXTRALEN(cc[-1]);
#endif
    break;

    CASE_ITERATOR_PRIVATE_DATA_2B
    if (PRIVATE_DATA(cc))
      private_data_length += 2;
    cc += 2 + IMM2_SIZE;
#ifdef SUPPORT_UNICODE
    if (common->utf && HAS_EXTRALEN(cc[-1])) cc += GET_EXTRALEN(cc[-1]);
#endif
    break;

    CASE_ITERATOR_TYPE_PRIVATE_DATA_1
    if (PRIVATE_DATA(cc))
      private_data_length++;
    cc += 1;
    break;

    CASE_ITERATOR_TYPE_PRIVATE_DATA_2A
    if (PRIVATE_DATA(cc))
      private_data_length += 2;
    cc += 1;
    break;

    CASE_ITERATOR_TYPE_PRIVATE_DATA_2B
    if (PRIVATE_DATA(cc))
      private_data_length += 2;
    cc += 1 + IMM2_SIZE;
    break;

    case OP_CLASS:
    case OP_NCLASS:
#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH != 8
    case OP_XCLASS:
    size = (*cc == OP_XCLASS) ? GET(cc, 1) : 1 + 32 / (int)sizeof(PCRE2_UCHAR);
#else
    size = 1 + 32 / (int)sizeof(PCRE2_UCHAR);
#endif
    if (PRIVATE_DATA(cc))
      private_data_length += get_class_iterator_size(cc + size);
    cc += size;
    break;

    default:
    cc = next_opcode(common, cc);
    SLJIT_ASSERT(cc != NULL);
    break;
    }
  }
SLJIT_ASSERT(cc == ccend);
return private_data_length;
}

static void copy_private_data(compiler_common *common, PCRE2_SPTR cc, PCRE2_SPTR ccend,
  BOOL save, int stackptr, int stacktop, BOOL needs_control_head)
{
DEFINE_COMPILER;
int srcw[2];
int count, size;
BOOL tmp1next = TRUE;
BOOL tmp1empty = TRUE;
BOOL tmp2empty = TRUE;
PCRE2_SPTR alternative;
enum {
  start,
  loop,
  end
} status;

status = save ? start : loop;
stackptr = STACK(stackptr - 2);
stacktop = STACK(stacktop - 1);

if (!save)
  {
  stackptr += (needs_control_head ? 2 : 1) * sizeof(sljit_sw);
  if (stackptr < stacktop)
    {
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), stackptr);
    stackptr += sizeof(sljit_sw);
    tmp1empty = FALSE;
    }
  if (stackptr < stacktop)
    {
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(STACK_TOP), stackptr);
    stackptr += sizeof(sljit_sw);
    tmp2empty = FALSE;
    }
  /* The tmp1next must be TRUE in either way. */
  }

do
  {
  count = 0;
  switch(status)
    {
    case start:
    SLJIT_ASSERT(save && common->recursive_head_ptr != 0);
    count = 1;
    srcw[0] = common->recursive_head_ptr;
    if (needs_control_head)
      {
      SLJIT_ASSERT(common->control_head_ptr != 0);
      count = 2;
      srcw[1] = common->control_head_ptr;
      }
    status = loop;
    break;

    case loop:
    if (cc >= ccend)
      {
      status = end;
      break;
      }

    switch(*cc)
      {
      case OP_KET:
      if (PRIVATE_DATA(cc) != 0)
        {
        count = 1;
        srcw[0] = PRIVATE_DATA(cc);
        SLJIT_ASSERT(PRIVATE_DATA(cc + 1) != 0);
        cc += PRIVATE_DATA(cc + 1);
        }
      cc += 1 + LINK_SIZE;
      break;

      case OP_ASSERT:
      case OP_ASSERT_NOT:
      case OP_ASSERTBACK:
      case OP_ASSERTBACK_NOT:
      case OP_ONCE:
      case OP_ONCE_NC:
      case OP_BRAPOS:
      case OP_SBRA:
      case OP_SBRAPOS:
      case OP_SCOND:
      count = 1;
      srcw[0] = PRIVATE_DATA(cc);
      SLJIT_ASSERT(srcw[0] != 0);
      cc += 1 + LINK_SIZE;
      break;

      case OP_CBRA:
      case OP_SCBRA:
      if (common->optimized_cbracket[GET2(cc, 1 + LINK_SIZE)] == 0)
        {
        count = 1;
        srcw[0] = OVECTOR_PRIV(GET2(cc, 1 + LINK_SIZE));
        }
      cc += 1 + LINK_SIZE + IMM2_SIZE;
      break;

      case OP_CBRAPOS:
      case OP_SCBRAPOS:
      count = 2;
      srcw[0] = PRIVATE_DATA(cc);
      srcw[1] = OVECTOR_PRIV(GET2(cc, 1 + LINK_SIZE));
      SLJIT_ASSERT(srcw[0] != 0 && srcw[1] != 0);
      cc += 1 + LINK_SIZE + IMM2_SIZE;
      break;

      case OP_COND:
      /* Might be a hidden SCOND. */
      alternative = cc + GET(cc, 1);
      if (*alternative == OP_KETRMAX || *alternative == OP_KETRMIN)
        {
        count = 1;
        srcw[0] = PRIVATE_DATA(cc);
        SLJIT_ASSERT(srcw[0] != 0);
        }
      cc += 1 + LINK_SIZE;
      break;

      CASE_ITERATOR_PRIVATE_DATA_1
      if (PRIVATE_DATA(cc))
        {
        count = 1;
        srcw[0] = PRIVATE_DATA(cc);
        }
      cc += 2;
#ifdef SUPPORT_UNICODE
      if (common->utf && HAS_EXTRALEN(cc[-1])) cc += GET_EXTRALEN(cc[-1]);
#endif
      break;

      CASE_ITERATOR_PRIVATE_DATA_2A
      if (PRIVATE_DATA(cc))
        {
        count = 2;
        srcw[0] = PRIVATE_DATA(cc);
        srcw[1] = PRIVATE_DATA(cc) + sizeof(sljit_sw);
        }
      cc += 2;
#ifdef SUPPORT_UNICODE
      if (common->utf && HAS_EXTRALEN(cc[-1])) cc += GET_EXTRALEN(cc[-1]);
#endif
      break;

      CASE_ITERATOR_PRIVATE_DATA_2B
      if (PRIVATE_DATA(cc))
        {
        count = 2;
        srcw[0] = PRIVATE_DATA(cc);
        srcw[1] = PRIVATE_DATA(cc) + sizeof(sljit_sw);
        }
      cc += 2 + IMM2_SIZE;
#ifdef SUPPORT_UNICODE
      if (common->utf && HAS_EXTRALEN(cc[-1])) cc += GET_EXTRALEN(cc[-1]);
#endif
      break;

      CASE_ITERATOR_TYPE_PRIVATE_DATA_1
      if (PRIVATE_DATA(cc))
        {
        count = 1;
        srcw[0] = PRIVATE_DATA(cc);
        }
      cc += 1;
      break;

      CASE_ITERATOR_TYPE_PRIVATE_DATA_2A
      if (PRIVATE_DATA(cc))
        {
        count = 2;
        srcw[0] = PRIVATE_DATA(cc);
        srcw[1] = srcw[0] + sizeof(sljit_sw);
        }
      cc += 1;
      break;

      CASE_ITERATOR_TYPE_PRIVATE_DATA_2B
      if (PRIVATE_DATA(cc))
        {
        count = 2;
        srcw[0] = PRIVATE_DATA(cc);
        srcw[1] = srcw[0] + sizeof(sljit_sw);
        }
      cc += 1 + IMM2_SIZE;
      break;

      case OP_CLASS:
      case OP_NCLASS:
#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH != 8
      case OP_XCLASS:
      size = (*cc == OP_XCLASS) ? GET(cc, 1) : 1 + 32 / (int)sizeof(PCRE2_UCHAR);
#else
      size = 1 + 32 / (int)sizeof(PCRE2_UCHAR);
#endif
      if (PRIVATE_DATA(cc))
        switch(get_class_iterator_size(cc + size))
          {
          case 1:
          count = 1;
          srcw[0] = PRIVATE_DATA(cc);
          break;

          case 2:
          count = 2;
          srcw[0] = PRIVATE_DATA(cc);
          srcw[1] = srcw[0] + sizeof(sljit_sw);
          break;

          default:
          SLJIT_ASSERT_STOP();
          break;
          }
      cc += size;
      break;

      default:
      cc = next_opcode(common, cc);
      SLJIT_ASSERT(cc != NULL);
      break;
      }
    break;

    case end:
    SLJIT_ASSERT_STOP();
    break;
    }

  while (count > 0)
    {
    count--;
    if (save)
      {
      if (tmp1next)
        {
        if (!tmp1empty)
          {
          OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackptr, TMP1, 0);
          stackptr += sizeof(sljit_sw);
          }
        OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), srcw[count]);
        tmp1empty = FALSE;
        tmp1next = FALSE;
        }
      else
        {
        if (!tmp2empty)
          {
          OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackptr, TMP2, 0);
          stackptr += sizeof(sljit_sw);
          }
        OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), srcw[count]);
        tmp2empty = FALSE;
        tmp1next = TRUE;
        }
      }
    else
      {
      if (tmp1next)
        {
        SLJIT_ASSERT(!tmp1empty);
        OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), srcw[count], TMP1, 0);
        tmp1empty = stackptr >= stacktop;
        if (!tmp1empty)
          {
          OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), stackptr);
          stackptr += sizeof(sljit_sw);
          }
        tmp1next = FALSE;
        }
      else
        {
        SLJIT_ASSERT(!tmp2empty);
        OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), srcw[count], TMP2, 0);
        tmp2empty = stackptr >= stacktop;
        if (!tmp2empty)
          {
          OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(STACK_TOP), stackptr);
          stackptr += sizeof(sljit_sw);
          }
        tmp1next = TRUE;
        }
      }
    }
  }
while (status != end);

if (save)
  {
  if (tmp1next)
    {
    if (!tmp1empty)
      {
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackptr, TMP1, 0);
      stackptr += sizeof(sljit_sw);
      }
    if (!tmp2empty)
      {
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackptr, TMP2, 0);
      stackptr += sizeof(sljit_sw);
      }
    }
  else
    {
    if (!tmp2empty)
      {
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackptr, TMP2, 0);
      stackptr += sizeof(sljit_sw);
      }
    if (!tmp1empty)
      {
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackptr, TMP1, 0);
      stackptr += sizeof(sljit_sw);
      }
    }
  }
SLJIT_ASSERT(cc == ccend && stackptr == stacktop && (save || (tmp1empty && tmp2empty)));
}

static SLJIT_INLINE PCRE2_SPTR set_then_offsets(compiler_common *common, PCRE2_SPTR cc, sljit_u8 *current_offset)
{
PCRE2_SPTR end = bracketend(cc);
BOOL has_alternatives = cc[GET(cc, 1)] == OP_ALT;

/* Assert captures then. */
if (*cc >= OP_ASSERT && *cc <= OP_ASSERTBACK_NOT)
  current_offset = NULL;
/* Conditional block does not. */
if (*cc == OP_COND || *cc == OP_SCOND)
  has_alternatives = FALSE;

cc = next_opcode(common, cc);
if (has_alternatives)
  current_offset = common->then_offsets + (cc - common->start);

while (cc < end)
  {
  if ((*cc >= OP_ASSERT && *cc <= OP_ASSERTBACK_NOT) || (*cc >= OP_ONCE && *cc <= OP_SCOND))
    cc = set_then_offsets(common, cc, current_offset);
  else
    {
    if (*cc == OP_ALT && has_alternatives)
      current_offset = common->then_offsets + (cc + 1 + LINK_SIZE - common->start);
    if (*cc >= OP_THEN && *cc <= OP_THEN_ARG && current_offset != NULL)
      *current_offset = 1;
    cc = next_opcode(common, cc);
    }
  }

return end;
}

#undef CASE_ITERATOR_PRIVATE_DATA_1
#undef CASE_ITERATOR_PRIVATE_DATA_2A
#undef CASE_ITERATOR_PRIVATE_DATA_2B
#undef CASE_ITERATOR_TYPE_PRIVATE_DATA_1
#undef CASE_ITERATOR_TYPE_PRIVATE_DATA_2A
#undef CASE_ITERATOR_TYPE_PRIVATE_DATA_2B

static SLJIT_INLINE BOOL is_powerof2(unsigned int value)
{
return (value & (value - 1)) == 0;
}

static SLJIT_INLINE void set_jumps(jump_list *list, struct sljit_label *label)
{
while (list)
  {
  /* sljit_set_label is clever enough to do nothing
  if either the jump or the label is NULL. */
  SET_LABEL(list->jump, label);
  list = list->next;
  }
}

static SLJIT_INLINE void add_jump(struct sljit_compiler *compiler, jump_list **list, struct sljit_jump *jump)
{
jump_list *list_item = sljit_alloc_memory(compiler, sizeof(jump_list));
if (list_item)
  {
  list_item->next = *list;
  list_item->jump = jump;
  *list = list_item;
  }
}

static void add_stub(compiler_common *common, struct sljit_jump *start)
{
DEFINE_COMPILER;
stub_list *list_item = sljit_alloc_memory(compiler, sizeof(stub_list));

if (list_item)
  {
  list_item->start = start;
  list_item->quit = LABEL();
  list_item->next = common->stubs;
  common->stubs = list_item;
  }
}

static void flush_stubs(compiler_common *common)
{
DEFINE_COMPILER;
stub_list *list_item = common->stubs;

while (list_item)
  {
  JUMPHERE(list_item->start);
  add_jump(compiler, &common->stackalloc, JUMP(SLJIT_FAST_CALL));
  JUMPTO(SLJIT_JUMP, list_item->quit);
  list_item = list_item->next;
  }
common->stubs = NULL;
}

static void add_label_addr(compiler_common *common, sljit_uw *update_addr)
{
DEFINE_COMPILER;
label_addr_list *label_addr;

label_addr = sljit_alloc_memory(compiler, sizeof(label_addr_list));
if (label_addr == NULL)
  return;
label_addr->label = LABEL();
label_addr->update_addr = update_addr;
label_addr->next = common->label_addrs;
common->label_addrs = label_addr;
}

static SLJIT_INLINE void count_match(compiler_common *common)
{
DEFINE_COMPILER;

OP2(SLJIT_SUB | SLJIT_SET_E, COUNT_MATCH, 0, COUNT_MATCH, 0, SLJIT_IMM, 1);
add_jump(compiler, &common->calllimit, JUMP(SLJIT_ZERO));
}

static SLJIT_INLINE void allocate_stack(compiler_common *common, int size)
{
/* May destroy all locals and registers except TMP2. */
DEFINE_COMPILER;

SLJIT_ASSERT(size > 0);
OP2(SLJIT_ADD, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, size * sizeof(sljit_sw));
#ifdef DESTROY_REGISTERS
OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, 12345);
OP1(SLJIT_MOV, TMP3, 0, TMP1, 0);
OP1(SLJIT_MOV, RETURN_ADDR, 0, TMP1, 0);
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCALS0, TMP1, 0);
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCALS1, TMP1, 0);
#endif
add_stub(common, CMP(SLJIT_GREATER, STACK_TOP, 0, STACK_LIMIT, 0));
}

static SLJIT_INLINE void free_stack(compiler_common *common, int size)
{
DEFINE_COMPILER;

SLJIT_ASSERT(size > 0);
OP2(SLJIT_SUB, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, size * sizeof(sljit_sw));
}

static sljit_uw * allocate_read_only_data(compiler_common *common, sljit_uw size)
{
DEFINE_COMPILER;
sljit_uw *result;

if (SLJIT_UNLIKELY(sljit_get_compiler_error(compiler)))
  return NULL;

result = (sljit_uw *)SLJIT_MALLOC(size + sizeof(sljit_uw), compiler->allocator_data);
if (SLJIT_UNLIKELY(result == NULL))
  {
  sljit_set_compiler_memory_error(compiler);
  return NULL;
  }

*(void**)result = common->read_only_data_head;
common->read_only_data_head = (void *)result;
return result + 1;
}

static SLJIT_INLINE void reset_ovector(compiler_common *common, int length)
{
DEFINE_COMPILER;
struct sljit_label *loop;
sljit_s32 i;

/* At this point we can freely use all temporary registers. */
SLJIT_ASSERT(length > 1);
/* TMP1 returns with begin - 1. */
OP2(SLJIT_SUB, SLJIT_R0, 0, SLJIT_MEM1(SLJIT_S0), SLJIT_OFFSETOF(jit_arguments, begin), SLJIT_IMM, IN_UCHARS(1));
if (length < 8)
  {
  for (i = 1; i < length; i++)
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(i), SLJIT_R0, 0);
  }
else
  {
  GET_LOCAL_BASE(SLJIT_R1, 0, OVECTOR_START);
  OP1(SLJIT_MOV, SLJIT_R2, 0, SLJIT_IMM, length - 1);
  loop = LABEL();
  OP1(SLJIT_MOVU, SLJIT_MEM1(SLJIT_R1), sizeof(sljit_sw), SLJIT_R0, 0);
  OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_R2, 0, SLJIT_R2, 0, SLJIT_IMM, 1);
  JUMPTO(SLJIT_NOT_ZERO, loop);
  }
}

static SLJIT_INLINE void reset_fast_fail(compiler_common *common)
{
DEFINE_COMPILER;
sljit_s32 i;

SLJIT_ASSERT(common->fast_fail_start_ptr < common->fast_fail_end_ptr);

OP2(SLJIT_SUB, TMP1, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
for (i = common->fast_fail_start_ptr; i < common->fast_fail_end_ptr; i += sizeof(sljit_sw))
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), i, TMP1, 0);
}

static SLJIT_INLINE void do_reset_match(compiler_common *common, int length)
{
DEFINE_COMPILER;
struct sljit_label *loop;
int i;

SLJIT_ASSERT(length > 1);
/* OVECTOR(1) contains the "string begin - 1" constant. */
if (length > 2)
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(1));
if (length < 8)
  {
  for (i = 2; i < length; i++)
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(i), TMP1, 0);
  }
else
  {
  GET_LOCAL_BASE(TMP2, 0, OVECTOR_START + sizeof(sljit_sw));
  OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_IMM, length - 2);
  loop = LABEL();
  OP1(SLJIT_MOVU, SLJIT_MEM1(TMP2), sizeof(sljit_sw), TMP1, 0);
  OP2(SLJIT_SUB | SLJIT_SET_E, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, 1);
  JUMPTO(SLJIT_NOT_ZERO, loop);
  }

OP1(SLJIT_MOV, STACK_TOP, 0, ARGUMENTS, 0);
if (common->mark_ptr != 0)
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->mark_ptr, SLJIT_IMM, 0);
if (common->control_head_ptr != 0)
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, SLJIT_IMM, 0);
OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(STACK_TOP), SLJIT_OFFSETOF(jit_arguments, stack));
OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->start_ptr);
OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(STACK_TOP), SLJIT_OFFSETOF(struct sljit_stack, base));
}

static sljit_sw SLJIT_CALL do_search_mark(sljit_sw *current, PCRE2_SPTR skip_arg)
{
while (current != NULL)
  {
  switch (current[-2])
    {
    case type_then_trap:
    break;

    case type_mark:
    if (PRIV(strcmp)(skip_arg, (PCRE2_SPTR)current[-3]) == 0)
      return current[-4];
    break;

    default:
    SLJIT_ASSERT_STOP();
    break;
    }
  SLJIT_ASSERT(current > (sljit_sw*)current[-1]);
  current = (sljit_sw*)current[-1];
  }
return -1;
}

static SLJIT_INLINE void copy_ovector(compiler_common *common, int topbracket)
{
DEFINE_COMPILER;
struct sljit_label *loop;

/* At this point we can freely use all registers. */
OP1(SLJIT_MOV, SLJIT_S2, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(1));
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(1), STR_PTR, 0);

OP1(SLJIT_MOV, SLJIT_R0, 0, ARGUMENTS, 0);
OP1(SLJIT_MOV, SLJIT_S0, 0, SLJIT_MEM1(SLJIT_SP), common->start_ptr);
if (common->mark_ptr != 0)
  OP1(SLJIT_MOV, SLJIT_R2, 0, SLJIT_MEM1(SLJIT_SP), common->mark_ptr);
OP1(SLJIT_MOV_U32, SLJIT_R1, 0, SLJIT_MEM1(SLJIT_R0), SLJIT_OFFSETOF(jit_arguments, oveccount));
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_R0), SLJIT_OFFSETOF(jit_arguments, startchar_ptr), SLJIT_S0, 0);
if (common->mark_ptr != 0)
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_R0), SLJIT_OFFSETOF(jit_arguments, mark_ptr), SLJIT_R2, 0);
OP2(SLJIT_ADD, SLJIT_R2, 0, SLJIT_MEM1(SLJIT_R0), SLJIT_OFFSETOF(jit_arguments, match_data),
  SLJIT_IMM, SLJIT_OFFSETOF(pcre2_match_data, ovector) - sizeof(PCRE2_SIZE));

GET_LOCAL_BASE(SLJIT_S0, 0, OVECTOR_START);
OP1(SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(SLJIT_R0), SLJIT_OFFSETOF(jit_arguments, begin));

loop = LABEL();
OP2(SLJIT_SUB, SLJIT_S1, 0, SLJIT_MEM1(SLJIT_S0), 0, SLJIT_R0, 0);
OP2(SLJIT_ADD, SLJIT_S0, 0, SLJIT_S0, 0, SLJIT_IMM, sizeof(sljit_sw));
/* Copy the integer value to the output buffer */
#if PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32
OP2(SLJIT_ASHR, SLJIT_S1, 0, SLJIT_S1, 0, SLJIT_IMM, UCHAR_SHIFT);
#endif
SLJIT_ASSERT(sizeof(PCRE2_SIZE) == 4 || sizeof(PCRE2_SIZE) == 8);
if (sizeof(PCRE2_SIZE) == 4)
  OP1(SLJIT_MOVU_U32, SLJIT_MEM1(SLJIT_R2), sizeof(PCRE2_SIZE), SLJIT_S1, 0);
else
  OP1(SLJIT_MOVU, SLJIT_MEM1(SLJIT_R2), sizeof(PCRE2_SIZE), SLJIT_S1, 0);
OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_R1, 0, SLJIT_R1, 0, SLJIT_IMM, 1);
JUMPTO(SLJIT_NOT_ZERO, loop);

/* Calculate the return value, which is the maximum ovector value. */
if (topbracket > 1)
  {
  GET_LOCAL_BASE(SLJIT_R0, 0, OVECTOR_START + topbracket * 2 * sizeof(sljit_sw));
  OP1(SLJIT_MOV, SLJIT_R1, 0, SLJIT_IMM, topbracket + 1);

  /* OVECTOR(0) is never equal to SLJIT_S2. */
  loop = LABEL();
  OP1(SLJIT_MOVU, SLJIT_R2, 0, SLJIT_MEM1(SLJIT_R0), -(2 * (sljit_sw)sizeof(sljit_sw)));
  OP2(SLJIT_SUB, SLJIT_R1, 0, SLJIT_R1, 0, SLJIT_IMM, 1);
  CMPTO(SLJIT_EQUAL, SLJIT_R2, 0, SLJIT_S2, 0, loop);
  OP1(SLJIT_MOV, SLJIT_RETURN_REG, 0, SLJIT_R1, 0);
  }
else
  OP1(SLJIT_MOV, SLJIT_RETURN_REG, 0, SLJIT_IMM, 1);
}

static SLJIT_INLINE void return_with_partial_match(compiler_common *common, struct sljit_label *quit)
{
DEFINE_COMPILER;
sljit_s32 mov_opcode;

SLJIT_COMPILE_ASSERT(STR_END == SLJIT_S1, str_end_must_be_saved_reg2);
SLJIT_ASSERT(common->start_used_ptr != 0 && common->start_ptr != 0
  && (common->mode == PCRE2_JIT_PARTIAL_SOFT ? common->hit_start != 0 : common->hit_start == 0));

OP1(SLJIT_MOV, SLJIT_R1, 0, ARGUMENTS, 0);
OP1(SLJIT_MOV, SLJIT_R2, 0, SLJIT_MEM1(SLJIT_SP),
  common->mode == PCRE2_JIT_PARTIAL_SOFT ? common->hit_start : common->start_ptr);
OP1(SLJIT_MOV, SLJIT_RETURN_REG, 0, SLJIT_IMM, PCRE2_ERROR_PARTIAL);

/* Store match begin and end. */
OP1(SLJIT_MOV, SLJIT_S0, 0, SLJIT_MEM1(SLJIT_R1), SLJIT_OFFSETOF(jit_arguments, begin));
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_R1), SLJIT_OFFSETOF(jit_arguments, startchar_ptr), SLJIT_R2, 0);
OP1(SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(SLJIT_R1), SLJIT_OFFSETOF(jit_arguments, match_data));

mov_opcode = (sizeof(PCRE2_SIZE) == 4) ? SLJIT_MOV_U32 : SLJIT_MOV;

OP2(SLJIT_SUB, SLJIT_R2, 0, SLJIT_R2, 0, SLJIT_S0, 0);
#if PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32
OP2(SLJIT_ASHR, SLJIT_R2, 0, SLJIT_R2, 0, SLJIT_IMM, UCHAR_SHIFT);
#endif
OP1(mov_opcode, SLJIT_MEM1(SLJIT_R1), SLJIT_OFFSETOF(pcre2_match_data, ovector), SLJIT_R2, 0);

OP2(SLJIT_SUB, STR_END, 0, STR_END, 0, SLJIT_S0, 0);
#if PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32
OP2(SLJIT_ASHR, STR_END, 0, STR_END, 0, SLJIT_IMM, UCHAR_SHIFT);
#endif
OP1(mov_opcode, SLJIT_MEM1(SLJIT_R1), SLJIT_OFFSETOF(pcre2_match_data, ovector) + sizeof(PCRE2_SIZE), STR_END, 0);

JUMPTO(SLJIT_JUMP, quit);
}

static SLJIT_INLINE void check_start_used_ptr(compiler_common *common)
{
/* May destroy TMP1. */
DEFINE_COMPILER;
struct sljit_jump *jump;

if (common->mode == PCRE2_JIT_PARTIAL_SOFT)
  {
  /* The value of -1 must be kept for start_used_ptr! */
  OP2(SLJIT_ADD, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->start_used_ptr, SLJIT_IMM, 1);
  /* Jumps if start_used_ptr < STR_PTR, or start_used_ptr == -1. Although overwriting
  is not necessary if start_used_ptr == STR_PTR, it does not hurt as well. */
  jump = CMP(SLJIT_LESS_EQUAL, TMP1, 0, STR_PTR, 0);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->start_used_ptr, STR_PTR, 0);
  JUMPHERE(jump);
  }
else if (common->mode == PCRE2_JIT_PARTIAL_HARD)
  {
  jump = CMP(SLJIT_LESS_EQUAL, SLJIT_MEM1(SLJIT_SP), common->start_used_ptr, STR_PTR, 0);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->start_used_ptr, STR_PTR, 0);
  JUMPHERE(jump);
  }
}

static SLJIT_INLINE BOOL char_has_othercase(compiler_common *common, PCRE2_SPTR cc)
{
/* Detects if the character has an othercase. */
unsigned int c;

#ifdef SUPPORT_UNICODE
if (common->utf)
  {
  GETCHAR(c, cc);
  if (c > 127)
    {
    return c != UCD_OTHERCASE(c);
    }
#if PCRE2_CODE_UNIT_WIDTH != 8
  return common->fcc[c] != c;
#endif
  }
else
#endif
  c = *cc;
return MAX_255(c) ? common->fcc[c] != c : FALSE;
}

static SLJIT_INLINE unsigned int char_othercase(compiler_common *common, unsigned int c)
{
/* Returns with the othercase. */
#ifdef SUPPORT_UNICODE
if (common->utf && c > 127)
  {
  return UCD_OTHERCASE(c);
  }
#endif
return TABLE_GET(c, common->fcc, c);
}

static unsigned int char_get_othercase_bit(compiler_common *common, PCRE2_SPTR cc)
{
/* Detects if the character and its othercase has only 1 bit difference. */
unsigned int c, oc, bit;
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
int n;
#endif

#ifdef SUPPORT_UNICODE
if (common->utf)
  {
  GETCHAR(c, cc);
  if (c <= 127)
    oc = common->fcc[c];
  else
    {
    oc = UCD_OTHERCASE(c);
    }
  }
else
  {
  c = *cc;
  oc = TABLE_GET(c, common->fcc, c);
  }
#else
c = *cc;
oc = TABLE_GET(c, common->fcc, c);
#endif

SLJIT_ASSERT(c != oc);

bit = c ^ oc;
/* Optimized for English alphabet. */
if (c <= 127 && bit == 0x20)
  return (0 << 8) | 0x20;

/* Since c != oc, they must have at least 1 bit difference. */
if (!is_powerof2(bit))
  return 0;

#if PCRE2_CODE_UNIT_WIDTH == 8

#ifdef SUPPORT_UNICODE
if (common->utf && c > 127)
  {
  n = GET_EXTRALEN(*cc);
  while ((bit & 0x3f) == 0)
    {
    n--;
    bit >>= 6;
    }
  return (n << 8) | bit;
  }
#endif /* SUPPORT_UNICODE */
return (0 << 8) | bit;

#elif PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32

#ifdef SUPPORT_UNICODE
if (common->utf && c > 65535)
  {
  if (bit >= (1 << 10))
    bit >>= 10;
  else
    return (bit < 256) ? ((2 << 8) | bit) : ((3 << 8) | (bit >> 8));
  }
#endif /* SUPPORT_UNICODE */
return (bit < 256) ? ((0 << 8) | bit) : ((1 << 8) | (bit >> 8));

#endif /* PCRE2_CODE_UNIT_WIDTH == [8|16|32] */
}

static void check_partial(compiler_common *common, BOOL force)
{
/* Checks whether a partial matching is occurred. Does not modify registers. */
DEFINE_COMPILER;
struct sljit_jump *jump = NULL;

SLJIT_ASSERT(!force || common->mode != PCRE2_JIT_COMPLETE);

if (common->mode == PCRE2_JIT_COMPLETE)
  return;

if (!force)
  jump = CMP(SLJIT_GREATER_EQUAL, SLJIT_MEM1(SLJIT_SP), common->start_used_ptr, STR_PTR, 0);
else if (common->mode == PCRE2_JIT_PARTIAL_SOFT)
  jump = CMP(SLJIT_EQUAL, SLJIT_MEM1(SLJIT_SP), common->start_used_ptr, SLJIT_IMM, -1);

if (common->mode == PCRE2_JIT_PARTIAL_SOFT)
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->hit_start, SLJIT_IMM, 0);
else
  {
  if (common->partialmatchlabel != NULL)
    JUMPTO(SLJIT_JUMP, common->partialmatchlabel);
  else
    add_jump(compiler, &common->partialmatch, JUMP(SLJIT_JUMP));
  }

if (jump != NULL)
  JUMPHERE(jump);
}

static void check_str_end(compiler_common *common, jump_list **end_reached)
{
/* Does not affect registers. Usually used in a tight spot. */
DEFINE_COMPILER;
struct sljit_jump *jump;

if (common->mode == PCRE2_JIT_COMPLETE)
  {
  add_jump(compiler, end_reached, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));
  return;
  }

jump = CMP(SLJIT_LESS, STR_PTR, 0, STR_END, 0);
if (common->mode == PCRE2_JIT_PARTIAL_SOFT)
  {
  add_jump(compiler, end_reached, CMP(SLJIT_GREATER_EQUAL, SLJIT_MEM1(SLJIT_SP), common->start_used_ptr, STR_PTR, 0));
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->hit_start, SLJIT_IMM, 0);
  add_jump(compiler, end_reached, JUMP(SLJIT_JUMP));
  }
else
  {
  add_jump(compiler, end_reached, CMP(SLJIT_GREATER_EQUAL, SLJIT_MEM1(SLJIT_SP), common->start_used_ptr, STR_PTR, 0));
  if (common->partialmatchlabel != NULL)
    JUMPTO(SLJIT_JUMP, common->partialmatchlabel);
  else
    add_jump(compiler, &common->partialmatch, JUMP(SLJIT_JUMP));
  }
JUMPHERE(jump);
}

static void detect_partial_match(compiler_common *common, jump_list **backtracks)
{
DEFINE_COMPILER;
struct sljit_jump *jump;

if (common->mode == PCRE2_JIT_COMPLETE)
  {
  add_jump(compiler, backtracks, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));
  return;
  }

/* Partial matching mode. */
jump = CMP(SLJIT_LESS, STR_PTR, 0, STR_END, 0);
add_jump(compiler, backtracks, CMP(SLJIT_GREATER_EQUAL, SLJIT_MEM1(SLJIT_SP), common->start_used_ptr, STR_PTR, 0));
if (common->mode == PCRE2_JIT_PARTIAL_SOFT)
  {
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->hit_start, SLJIT_IMM, 0);
  add_jump(compiler, backtracks, JUMP(SLJIT_JUMP));
  }
else
  {
  if (common->partialmatchlabel != NULL)
    JUMPTO(SLJIT_JUMP, common->partialmatchlabel);
  else
    add_jump(compiler, &common->partialmatch, JUMP(SLJIT_JUMP));
  }
JUMPHERE(jump);
}

static void peek_char(compiler_common *common, sljit_u32 max)
{
/* Reads the character into TMP1, keeps STR_PTR.
Does not check STR_END. TMP2 Destroyed. */
DEFINE_COMPILER;
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
struct sljit_jump *jump;
#endif

SLJIT_UNUSED_ARG(max);

OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), 0);
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
if (common->utf)
  {
  if (max < 128) return;

  jump = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0xc0);
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  add_jump(compiler, &common->utfreadchar, JUMP(SLJIT_FAST_CALL));
  OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, TMP2, 0);
  JUMPHERE(jump);
  }
#endif /* SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32 */

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 16
if (common->utf)
  {
  if (max < 0xd800) return;

  OP2(SLJIT_SUB, TMP2, 0, TMP1, 0, SLJIT_IMM, 0xd800);
  jump = CMP(SLJIT_GREATER, TMP2, 0, SLJIT_IMM, 0xdc00 - 0xd800 - 1);
  /* TMP2 contains the high surrogate. */
  OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
  OP2(SLJIT_ADD, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x40);
  OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, 10);
  OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x3ff);
  OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);
  JUMPHERE(jump);
  }
#endif
}

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8

static BOOL is_char7_bitset(const sljit_u8 *bitset, BOOL nclass)
{
/* Tells whether the character codes below 128 are enough
to determine a match. */
const sljit_u8 value = nclass ? 0xff : 0;
const sljit_u8 *end = bitset + 32;

bitset += 16;
do
  {
  if (*bitset++ != value)
    return FALSE;
  }
while (bitset < end);
return TRUE;
}

static void read_char7_type(compiler_common *common, BOOL full_read)
{
/* Reads the precise character type of a character into TMP1, if the character
is less than 128. Otherwise it returns with zero. Does not check STR_END. The
full_read argument tells whether characters above max are accepted or not. */
DEFINE_COMPILER;
struct sljit_jump *jump;

SLJIT_ASSERT(common->utf);

OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), 0);
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(TMP2), common->ctypes);

if (full_read)
  {
  jump = CMP(SLJIT_LESS, TMP2, 0, SLJIT_IMM, 0xc0);
  OP1(SLJIT_MOV_U8, TMP2, 0, SLJIT_MEM1(TMP2), (sljit_sw)PRIV(utf8_table4) - 0xc0);
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP2, 0);
  JUMPHERE(jump);
  }
}

#endif /* SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8 */

static void read_char_range(compiler_common *common, sljit_u32 min, sljit_u32 max, BOOL update_str_ptr)
{
/* Reads the precise value of a character into TMP1, if the character is
between min and max (c >= min && c <= max). Otherwise it returns with a value
outside the range. Does not check STR_END. */
DEFINE_COMPILER;
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
struct sljit_jump *jump;
#endif
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
struct sljit_jump *jump2;
#endif

SLJIT_UNUSED_ARG(update_str_ptr);
SLJIT_UNUSED_ARG(min);
SLJIT_UNUSED_ARG(max);
SLJIT_ASSERT(min <= max);

OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
if (common->utf)
  {
  if (max < 128 && !update_str_ptr) return;

  jump = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0xc0);
  if (min >= 0x10000)
    {
    OP2(SLJIT_SUB, TMP2, 0, TMP1, 0, SLJIT_IMM, 0xf0);
    if (update_str_ptr)
      OP1(SLJIT_MOV_U8, RETURN_ADDR, 0, SLJIT_MEM1(TMP1), (sljit_sw)PRIV(utf8_table4) - 0xc0);
    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
    jump2 = CMP(SLJIT_GREATER, TMP2, 0, SLJIT_IMM, 0x7);
    OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, 6);
    OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x3f);
    OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);
    OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(1));
    OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 6);
    OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x3f);
    OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);
    OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(2));
    if (!update_str_ptr)
      OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(3));
    OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 6);
    OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x3f);
    OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);
    JUMPHERE(jump2);
    if (update_str_ptr)
      OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, RETURN_ADDR, 0);
    }
  else if (min >= 0x800 && max <= 0xffff)
    {
    OP2(SLJIT_SUB, TMP2, 0, TMP1, 0, SLJIT_IMM, 0xe0);
    if (update_str_ptr)
      OP1(SLJIT_MOV_U8, RETURN_ADDR, 0, SLJIT_MEM1(TMP1), (sljit_sw)PRIV(utf8_table4) - 0xc0);
    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
    jump2 = CMP(SLJIT_GREATER, TMP2, 0, SLJIT_IMM, 0xf);
    OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, 6);
    OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x3f);
    OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);
    OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(1));
    if (!update_str_ptr)
      OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(2));
    OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 6);
    OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x3f);
    OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);
    JUMPHERE(jump2);
    if (update_str_ptr)
      OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, RETURN_ADDR, 0);
    }
  else if (max >= 0x800)
    add_jump(compiler, (max < 0x10000) ? &common->utfreadchar16 : &common->utfreadchar, JUMP(SLJIT_FAST_CALL));
  else if (max < 128)
    {
    OP1(SLJIT_MOV_U8, TMP2, 0, SLJIT_MEM1(TMP1), (sljit_sw)PRIV(utf8_table4) - 0xc0);
    OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP2, 0);
    }
  else
    {
    OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
    if (!update_str_ptr)
      OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
    else
      OP1(SLJIT_MOV_U8, RETURN_ADDR, 0, SLJIT_MEM1(TMP1), (sljit_sw)PRIV(utf8_table4) - 0xc0);
    OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x3f);
    OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 6);
    OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x3f);
    OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);
    if (update_str_ptr)
      OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, RETURN_ADDR, 0);
    }
  JUMPHERE(jump);
  }
#endif

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 16
if (common->utf)
  {
  if (max >= 0x10000)
    {
    OP2(SLJIT_SUB, TMP2, 0, TMP1, 0, SLJIT_IMM, 0xd800);
    jump = CMP(SLJIT_GREATER, TMP2, 0, SLJIT_IMM, 0xdc00 - 0xd800 - 1);
    /* TMP2 contains the high surrogate. */
    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
    OP2(SLJIT_ADD, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x40);
    OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, 10);
    OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
    OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x3ff);
    OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);
    JUMPHERE(jump);
    return;
    }

  if (max < 0xd800 && !update_str_ptr) return;

  /* Skip low surrogate if necessary. */
  OP2(SLJIT_SUB, TMP2, 0, TMP1, 0, SLJIT_IMM, 0xd800);
  jump = CMP(SLJIT_GREATER, TMP2, 0, SLJIT_IMM, 0xdc00 - 0xd800 - 1);
  if (update_str_ptr)
    OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  if (max >= 0xd800)
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, 0x10000);
  JUMPHERE(jump);
  }
#endif
}

static SLJIT_INLINE void read_char(compiler_common *common)
{
read_char_range(common, 0, READ_CHAR_MAX, TRUE);
}

static void read_char8_type(compiler_common *common, BOOL update_str_ptr)
{
/* Reads the character type into TMP1, updates STR_PTR. Does not check STR_END. */
DEFINE_COMPILER;
#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH != 8
struct sljit_jump *jump;
#endif
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
struct sljit_jump *jump2;
#endif

SLJIT_UNUSED_ARG(update_str_ptr);

OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), 0);
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
if (common->utf)
  {
  /* This can be an extra read in some situations, but hopefully
  it is needed in most cases. */
  OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(TMP2), common->ctypes);
  jump = CMP(SLJIT_LESS, TMP2, 0, SLJIT_IMM, 0xc0);
  if (!update_str_ptr)
    {
    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
    OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
    OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x3f);
    OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, 6);
    OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x3f);
    OP2(SLJIT_OR, TMP2, 0, TMP2, 0, TMP1, 0);
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, 0);
    jump2 = CMP(SLJIT_GREATER, TMP2, 0, SLJIT_IMM, 255);
    OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(TMP2), common->ctypes);
    JUMPHERE(jump2);
    }
  else
    add_jump(compiler, &common->utfreadtype8, JUMP(SLJIT_FAST_CALL));
  JUMPHERE(jump);
  return;
  }
#endif /* SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8 */

#if PCRE2_CODE_UNIT_WIDTH != 8
/* The ctypes array contains only 256 values. */
OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, 0);
jump = CMP(SLJIT_GREATER, TMP2, 0, SLJIT_IMM, 255);
#endif
OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(TMP2), common->ctypes);
#if PCRE2_CODE_UNIT_WIDTH != 8
JUMPHERE(jump);
#endif

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 16
if (common->utf && update_str_ptr)
  {
  /* Skip low surrogate if necessary. */
  OP2(SLJIT_SUB, TMP2, 0, TMP2, 0, SLJIT_IMM, 0xd800);
  jump = CMP(SLJIT_GREATER, TMP2, 0, SLJIT_IMM, 0xdc00 - 0xd800 - 1);
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  JUMPHERE(jump);
  }
#endif /* SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 16 */
}

static void skip_char_back(compiler_common *common)
{
/* Goes one character back. Affects STR_PTR and TMP1. Does not check begin. */
DEFINE_COMPILER;
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
#if PCRE2_CODE_UNIT_WIDTH == 8
struct sljit_label *label;

if (common->utf)
  {
  label = LABEL();
  OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), -IN_UCHARS(1));
  OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xc0);
  CMPTO(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, 0x80, label);
  return;
  }
#elif PCRE2_CODE_UNIT_WIDTH == 16
if (common->utf)
  {
  OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), -IN_UCHARS(1));
  OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  /* Skip low surrogate if necessary. */
  OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xfc00);
  OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0xdc00);
  OP_FLAGS(SLJIT_MOV, TMP1, 0, SLJIT_UNUSED, 0, SLJIT_EQUAL);
  OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 1);
  OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, TMP1, 0);
  return;
  }
#endif /* PCRE2_CODE_UNIT_WIDTH == [8|16] */
#endif /* SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32 */
OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
}

static void check_newlinechar(compiler_common *common, int nltype, jump_list **backtracks, BOOL jumpifmatch)
{
/* Character comes in TMP1. Checks if it is a newline. TMP2 may be destroyed. */
DEFINE_COMPILER;
struct sljit_jump *jump;

if (nltype == NLTYPE_ANY)
  {
  add_jump(compiler, &common->anynewline, JUMP(SLJIT_FAST_CALL));
  add_jump(compiler, backtracks, JUMP(jumpifmatch ? SLJIT_NOT_ZERO : SLJIT_ZERO));
  }
else if (nltype == NLTYPE_ANYCRLF)
  {
  if (jumpifmatch)
    {
    add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, CHAR_CR));
    add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, CHAR_NL));
    }
  else
    {
    jump = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, CHAR_CR);
    add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, CHAR_NL));
    JUMPHERE(jump);
    }
  }
else
  {
  SLJIT_ASSERT(nltype == NLTYPE_FIXED && common->newline < 256);
  add_jump(compiler, backtracks, CMP(jumpifmatch ? SLJIT_EQUAL : SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, common->newline));
  }
}

#ifdef SUPPORT_UNICODE

#if PCRE2_CODE_UNIT_WIDTH == 8
static void do_utfreadchar(compiler_common *common)
{
/* Fast decoding a UTF-8 character. TMP1 contains the first byte
of the character (>= 0xc0). Return char value in TMP1, length in TMP2. */
DEFINE_COMPILER;
struct sljit_jump *jump;

sljit_emit_fast_enter(compiler, RETURN_ADDR, 0);
OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x3f);
OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 6);
OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x3f);
OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);

/* Searching for the first zero. */
OP2(SLJIT_AND | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0x800);
jump = JUMP(SLJIT_NOT_ZERO);
/* Two byte sequence. */
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, IN_UCHARS(2));
sljit_emit_fast_return(compiler, RETURN_ADDR, 0);

JUMPHERE(jump);
OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(1));
OP2(SLJIT_XOR, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x800);
OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 6);
OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x3f);
OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);

OP2(SLJIT_AND | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0x10000);
jump = JUMP(SLJIT_NOT_ZERO);
/* Three byte sequence. */
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(2));
OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, IN_UCHARS(3));
sljit_emit_fast_return(compiler, RETURN_ADDR, 0);

/* Four byte sequence. */
JUMPHERE(jump);
OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(2));
OP2(SLJIT_XOR, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x10000);
OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 6);
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(3));
OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x3f);
OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);
OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, IN_UCHARS(4));
sljit_emit_fast_return(compiler, RETURN_ADDR, 0);
}

static void do_utfreadchar16(compiler_common *common)
{
/* Fast decoding a UTF-8 character. TMP1 contains the first byte
of the character (>= 0xc0). Return value in TMP1. */
DEFINE_COMPILER;
struct sljit_jump *jump;

sljit_emit_fast_enter(compiler, RETURN_ADDR, 0);
OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x3f);
OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 6);
OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x3f);
OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);

/* Searching for the first zero. */
OP2(SLJIT_AND | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0x800);
jump = JUMP(SLJIT_NOT_ZERO);
/* Two byte sequence. */
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
sljit_emit_fast_return(compiler, RETURN_ADDR, 0);

JUMPHERE(jump);
OP2(SLJIT_AND | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0x400);
OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_UNUSED, 0, SLJIT_NOT_ZERO);
/* This code runs only in 8 bit mode. No need to shift the value. */
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP2, 0);
OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(1));
OP2(SLJIT_XOR, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x800);
OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 6);
OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x3f);
OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);
/* Three byte sequence. */
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(2));
sljit_emit_fast_return(compiler, RETURN_ADDR, 0);
}

static void do_utfreadtype8(compiler_common *common)
{
/* Fast decoding a UTF-8 character type. TMP2 contains the first byte
of the character (>= 0xc0). Return value in TMP1. */
DEFINE_COMPILER;
struct sljit_jump *jump;
struct sljit_jump *compare;

sljit_emit_fast_enter(compiler, RETURN_ADDR, 0);

OP2(SLJIT_AND | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP2, 0, SLJIT_IMM, 0x20);
jump = JUMP(SLJIT_NOT_ZERO);
/* Two byte sequence. */
OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x1f);
/* The upper 5 bits are known at this point. */
compare = CMP(SLJIT_GREATER, TMP2, 0, SLJIT_IMM, 0x3);
OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, 6);
OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x3f);
OP2(SLJIT_OR, TMP2, 0, TMP2, 0, TMP1, 0);
OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(TMP2), common->ctypes);
sljit_emit_fast_return(compiler, RETURN_ADDR, 0);

JUMPHERE(compare);
OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, 0);
sljit_emit_fast_return(compiler, RETURN_ADDR, 0);

/* We only have types for characters less than 256. */
JUMPHERE(jump);
OP1(SLJIT_MOV_U8, TMP2, 0, SLJIT_MEM1(TMP2), (sljit_sw)PRIV(utf8_table4) - 0xc0);
OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, 0);
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP2, 0);
sljit_emit_fast_return(compiler, RETURN_ADDR, 0);
}

#endif /* PCRE2_CODE_UNIT_WIDTH == 8 */

/* UCD_BLOCK_SIZE must be 128 (see the assert below). */
#define UCD_BLOCK_MASK 127
#define UCD_BLOCK_SHIFT 7

static void do_getucd(compiler_common *common)
{
/* Search the UCD record for the character comes in TMP1.
Returns chartype in TMP1 and UCD offset in TMP2. */
DEFINE_COMPILER;

SLJIT_ASSERT(UCD_BLOCK_SIZE == 128 && sizeof(ucd_record) == 8);

sljit_emit_fast_enter(compiler, RETURN_ADDR, 0);
OP2(SLJIT_LSHR, TMP2, 0, TMP1, 0, SLJIT_IMM, UCD_BLOCK_SHIFT);
OP1(SLJIT_MOV_U8, TMP2, 0, SLJIT_MEM1(TMP2), (sljit_sw)PRIV(ucd_stage1));
OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, UCD_BLOCK_MASK);
OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, UCD_BLOCK_SHIFT);
OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, TMP2, 0);
OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, (sljit_sw)PRIV(ucd_stage2));
OP1(SLJIT_MOV_U16, TMP2, 0, SLJIT_MEM2(TMP2, TMP1), 1);
OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, (sljit_sw)PRIV(ucd_records) + SLJIT_OFFSETOF(ucd_record, chartype));
OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM2(TMP1, TMP2), 3);
sljit_emit_fast_return(compiler, RETURN_ADDR, 0);
}

#endif /* SUPPORT_UNICODE */

static SLJIT_INLINE struct sljit_label *mainloop_entry(compiler_common *common, BOOL hascrorlf, sljit_u32 overall_options)
{
DEFINE_COMPILER;
struct sljit_label *mainloop;
struct sljit_label *newlinelabel = NULL;
struct sljit_jump *start;
struct sljit_jump *end = NULL;
struct sljit_jump *end2 = NULL;
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
struct sljit_jump *singlechar;
#endif
jump_list *newline = NULL;
BOOL newlinecheck = FALSE;
BOOL readuchar = FALSE;

if (!(hascrorlf || (overall_options & PCRE2_FIRSTLINE) != 0)
    && (common->nltype == NLTYPE_ANY || common->nltype == NLTYPE_ANYCRLF || common->newline > 255))
  newlinecheck = TRUE;

SLJIT_ASSERT(common->forced_quit_label == NULL);

if ((overall_options & PCRE2_FIRSTLINE) != 0)
  {
  /* Search for the end of the first line. */
  SLJIT_ASSERT(common->match_end_ptr != 0);
  OP1(SLJIT_MOV, TMP3, 0, STR_PTR, 0);

  if (common->nltype == NLTYPE_FIXED && common->newline > 255)
    {
    mainloop = LABEL();
    OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
    end = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);
    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-1));
    OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
    CMPTO(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, (common->newline >> 8) & 0xff, mainloop);
    CMPTO(SLJIT_NOT_EQUAL, TMP2, 0, SLJIT_IMM, common->newline & 0xff, mainloop);
    JUMPHERE(end);
    OP2(SLJIT_SUB, SLJIT_MEM1(SLJIT_SP), common->match_end_ptr, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
    }
  else
    {
    end = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);
    mainloop = LABEL();
    /* Continual stores does not cause data dependency. */
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->match_end_ptr, STR_PTR, 0);
    read_char_range(common, common->nlmin, common->nlmax, TRUE);
    check_newlinechar(common, common->nltype, &newline, TRUE);
    CMPTO(SLJIT_LESS, STR_PTR, 0, STR_END, 0, mainloop);
    JUMPHERE(end);
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->match_end_ptr, STR_PTR, 0);
    set_jumps(newline, LABEL());
    }

  OP1(SLJIT_MOV, STR_PTR, 0, TMP3, 0);
  }
else if ((overall_options & PCRE2_USE_OFFSET_LIMIT) != 0)
  {
  /* Check whether offset limit is set and valid. */
  SLJIT_ASSERT(common->match_end_ptr != 0);

  OP1(SLJIT_MOV, TMP1, 0, ARGUMENTS, 0);
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, offset_limit));
  OP1(SLJIT_MOV, TMP2, 0, STR_END, 0);
  end = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, (sljit_sw) PCRE2_UNSET);
  OP1(SLJIT_MOV, TMP2, 0, ARGUMENTS, 0);
#if PCRE2_CODE_UNIT_WIDTH == 16
  OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 1);
#elif PCRE2_CODE_UNIT_WIDTH == 32
  OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 2);
#endif
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(TMP2), SLJIT_OFFSETOF(jit_arguments, begin));
  OP2(SLJIT_ADD, TMP2, 0, TMP2, 0, TMP1, 0);
  end2 = CMP(SLJIT_LESS_EQUAL, TMP2, 0, STR_END, 0);
  OP1(SLJIT_MOV, TMP2, 0, STR_END, 0);
  JUMPHERE(end2);
  OP1(SLJIT_MOV, SLJIT_RETURN_REG, 0, SLJIT_IMM, PCRE2_ERROR_NOMATCH);
  add_jump(compiler, &common->forced_quit, CMP(SLJIT_LESS, TMP2, 0, STR_PTR, 0));
  JUMPHERE(end);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->match_end_ptr, TMP2, 0);
  }

start = JUMP(SLJIT_JUMP);

if (newlinecheck)
  {
  newlinelabel = LABEL();
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  end = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);
  OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), 0);
  OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, common->newline & 0xff);
  OP_FLAGS(SLJIT_MOV, TMP1, 0, SLJIT_UNUSED, 0, SLJIT_EQUAL);
#if PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32
  OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, UCHAR_SHIFT);
#endif
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);
  end2 = JUMP(SLJIT_JUMP);
  }

mainloop = LABEL();

/* Increasing the STR_PTR here requires one less jump in the most common case. */
#ifdef SUPPORT_UNICODE
if (common->utf) readuchar = TRUE;
#endif
if (newlinecheck) readuchar = TRUE;

if (readuchar)
  OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), 0);

if (newlinecheck)
  CMPTO(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, (common->newline >> 8) & 0xff, newlinelabel);

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
#if PCRE2_CODE_UNIT_WIDTH == 8
if (common->utf)
  {
  singlechar = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0xc0);
  OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(TMP1), (sljit_sw)PRIV(utf8_table4) - 0xc0);
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);
  JUMPHERE(singlechar);
  }
#elif PCRE2_CODE_UNIT_WIDTH == 16
if (common->utf)
  {
  singlechar = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0xd800);
  OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xfc00);
  OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0xd800);
  OP_FLAGS(SLJIT_MOV, TMP1, 0, SLJIT_UNUSED, 0, SLJIT_EQUAL);
  OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 1);
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);
  JUMPHERE(singlechar);
  }
#endif /* PCRE2_CODE_UNIT_WIDTH == [8|16] */
#endif /* SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32 */
JUMPHERE(start);

if (newlinecheck)
  {
  JUMPHERE(end);
  JUMPHERE(end2);
  }

return mainloop;
}

#define MAX_N_CHARS 16
#define MAX_DIFF_CHARS 6

static SLJIT_INLINE void add_prefix_char(PCRE2_UCHAR chr, PCRE2_UCHAR *chars)
{
PCRE2_UCHAR i, len;

len = chars[0];
if (len == 255)
  return;

if (len == 0)
  {
  chars[0] = 1;
  chars[1] = chr;
  return;
  }

for (i = len; i > 0; i--)
  if (chars[i] == chr)
    return;

if (len >= MAX_DIFF_CHARS - 1)
  {
  chars[0] = 255;
  return;
  }

len++;
chars[len] = chr;
chars[0] = len;
}

static int scan_prefix(compiler_common *common, PCRE2_SPTR cc, PCRE2_UCHAR *chars, int max_chars, sljit_u32 *rec_count)
{
/* Recursive function, which scans prefix literals. */
BOOL last, any, class, caseless;
int len, repeat, len_save, consumed = 0;
sljit_u32 chr; /* Any unicode character. */
sljit_u8 *bytes, *bytes_end, byte;
PCRE2_SPTR alternative, cc_save, oc;
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
PCRE2_UCHAR othercase[8];
#elif defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 16
PCRE2_UCHAR othercase[2];
#else
PCRE2_UCHAR othercase[1];
#endif

repeat = 1;
while (TRUE)
  {
  if (*rec_count == 0)
    return 0;
  (*rec_count)--;

  last = TRUE;
  any = FALSE;
  class = FALSE;
  caseless = FALSE;

  switch (*cc)
    {
    case OP_CHARI:
    caseless = TRUE;
    case OP_CHAR:
    last = FALSE;
    cc++;
    break;

    case OP_SOD:
    case OP_SOM:
    case OP_SET_SOM:
    case OP_NOT_WORD_BOUNDARY:
    case OP_WORD_BOUNDARY:
    case OP_EODN:
    case OP_EOD:
    case OP_CIRC:
    case OP_CIRCM:
    case OP_DOLL:
    case OP_DOLLM:
    /* Zero width assertions. */
    cc++;
    continue;

    case OP_ASSERT:
    case OP_ASSERT_NOT:
    case OP_ASSERTBACK:
    case OP_ASSERTBACK_NOT:
    cc = bracketend(cc);
    continue;

    case OP_PLUSI:
    case OP_MINPLUSI:
    case OP_POSPLUSI:
    caseless = TRUE;
    case OP_PLUS:
    case OP_MINPLUS:
    case OP_POSPLUS:
    cc++;
    break;

    case OP_EXACTI:
    caseless = TRUE;
    case OP_EXACT:
    repeat = GET2(cc, 1);
    last = FALSE;
    cc += 1 + IMM2_SIZE;
    break;

    case OP_QUERYI:
    case OP_MINQUERYI:
    case OP_POSQUERYI:
    caseless = TRUE;
    case OP_QUERY:
    case OP_MINQUERY:
    case OP_POSQUERY:
    len = 1;
    cc++;
#ifdef SUPPORT_UNICODE
    if (common->utf && HAS_EXTRALEN(*cc)) len += GET_EXTRALEN(*cc);
#endif
    max_chars = scan_prefix(common, cc + len, chars, max_chars, rec_count);
    if (max_chars == 0)
      return consumed;
    last = FALSE;
    break;

    case OP_KET:
    cc += 1 + LINK_SIZE;
    continue;

    case OP_ALT:
    cc += GET(cc, 1);
    continue;

    case OP_ONCE:
    case OP_ONCE_NC:
    case OP_BRA:
    case OP_BRAPOS:
    case OP_CBRA:
    case OP_CBRAPOS:
    alternative = cc + GET(cc, 1);
    while (*alternative == OP_ALT)
      {
      max_chars = scan_prefix(common, alternative + 1 + LINK_SIZE, chars, max_chars, rec_count);
      if (max_chars == 0)
        return consumed;
      alternative += GET(alternative, 1);
      }

    if (*cc == OP_CBRA || *cc == OP_CBRAPOS)
      cc += IMM2_SIZE;
    cc += 1 + LINK_SIZE;
    continue;

    case OP_CLASS:
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
    if (common->utf && !is_char7_bitset((const sljit_u8 *)(cc + 1), FALSE))
      return consumed;
#endif
    class = TRUE;
    break;

    case OP_NCLASS:
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
    if (common->utf) return consumed;
#endif
    class = TRUE;
    break;

#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH != 8
    case OP_XCLASS:
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
    if (common->utf) return consumed;
#endif
    any = TRUE;
    cc += GET(cc, 1);
    break;
#endif

    case OP_DIGIT:
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
    if (common->utf && !is_char7_bitset((const sljit_u8 *)common->ctypes - cbit_length + cbit_digit, FALSE))
      return consumed;
#endif
    any = TRUE;
    cc++;
    break;

    case OP_WHITESPACE:
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
    if (common->utf && !is_char7_bitset((const sljit_u8 *)common->ctypes - cbit_length + cbit_space, FALSE))
      return consumed;
#endif
    any = TRUE;
    cc++;
    break;

    case OP_WORDCHAR:
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
    if (common->utf && !is_char7_bitset((const sljit_u8 *)common->ctypes - cbit_length + cbit_word, FALSE))
      return consumed;
#endif
    any = TRUE;
    cc++;
    break;

    case OP_NOT:
    case OP_NOTI:
    cc++;
    /* Fall through. */
    case OP_NOT_DIGIT:
    case OP_NOT_WHITESPACE:
    case OP_NOT_WORDCHAR:
    case OP_ANY:
    case OP_ALLANY:
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
    if (common->utf) return consumed;
#endif
    any = TRUE;
    cc++;
    break;

#ifdef SUPPORT_UNICODE
    case OP_NOTPROP:
    case OP_PROP:
#if PCRE2_CODE_UNIT_WIDTH != 32
    if (common->utf) return consumed;
#endif
    any = TRUE;
    cc += 1 + 2;
    break;
#endif

    case OP_TYPEEXACT:
    repeat = GET2(cc, 1);
    cc += 1 + IMM2_SIZE;
    continue;

    case OP_NOTEXACT:
    case OP_NOTEXACTI:
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
    if (common->utf) return consumed;
#endif
    any = TRUE;
    repeat = GET2(cc, 1);
    cc += 1 + IMM2_SIZE + 1;
    break;

    default:
    return consumed;
    }

  if (any)
    {
    do
      {
      chars[0] = 255;

      consumed++;
      if (--max_chars == 0)
        return consumed;
      chars += MAX_DIFF_CHARS;
      }
    while (--repeat > 0);

    repeat = 1;
    continue;
    }

  if (class)
    {
    bytes = (sljit_u8*) (cc + 1);
    cc += 1 + 32 / sizeof(PCRE2_UCHAR);

    switch (*cc)
      {
      case OP_CRSTAR:
      case OP_CRMINSTAR:
      case OP_CRPOSSTAR:
      case OP_CRQUERY:
      case OP_CRMINQUERY:
      case OP_CRPOSQUERY:
      max_chars = scan_prefix(common, cc + 1, chars, max_chars, rec_count);
      if (max_chars == 0)
        return consumed;
      break;

      default:
      case OP_CRPLUS:
      case OP_CRMINPLUS:
      case OP_CRPOSPLUS:
      break;

      case OP_CRRANGE:
      case OP_CRMINRANGE:
      case OP_CRPOSRANGE:
      repeat = GET2(cc, 1);
      if (repeat <= 0)
        return consumed;
      break;
      }

    do
      {
      if (bytes[31] & 0x80)
        chars[0] = 255;
      else if (chars[0] != 255)
        {
        bytes_end = bytes + 32;
        chr = 0;
        do
          {
          byte = *bytes++;
          SLJIT_ASSERT((chr & 0x7) == 0);
          if (byte == 0)
            chr += 8;
          else
            {
            do
              {
              if ((byte & 0x1) != 0)
                add_prefix_char(chr, chars);
              byte >>= 1;
              chr++;
              }
            while (byte != 0);
            chr = (chr + 7) & ~7;
            }
          }
        while (chars[0] != 255 && bytes < bytes_end);
        bytes = bytes_end - 32;
        }

      consumed++;
      if (--max_chars == 0)
        return consumed;
      chars += MAX_DIFF_CHARS;
      }
    while (--repeat > 0);

    switch (*cc)
      {
      case OP_CRSTAR:
      case OP_CRMINSTAR:
      case OP_CRPOSSTAR:
      return consumed;

      case OP_CRQUERY:
      case OP_CRMINQUERY:
      case OP_CRPOSQUERY:
      cc++;
      break;

      case OP_CRRANGE:
      case OP_CRMINRANGE:
      case OP_CRPOSRANGE:
      if (GET2(cc, 1) != GET2(cc, 1 + IMM2_SIZE))
        return consumed;
      cc += 1 + 2 * IMM2_SIZE;
      break;
      }

    repeat = 1;
    continue;
    }

  len = 1;
#ifdef SUPPORT_UNICODE
  if (common->utf && HAS_EXTRALEN(*cc)) len += GET_EXTRALEN(*cc);
#endif

  if (caseless && char_has_othercase(common, cc))
    {
#ifdef SUPPORT_UNICODE
    if (common->utf)
      {
      GETCHAR(chr, cc);
      if ((int)PRIV(ord2utf)(char_othercase(common, chr), othercase) != len)
        return consumed;
      }
    else
#endif
      {
      chr = *cc;
      othercase[0] = TABLE_GET(chr, common->fcc, chr);
      }
    }
  else
    {
    caseless = FALSE;
    othercase[0] = 0; /* Stops compiler warning - PH */
    }

  len_save = len;
  cc_save = cc;
  while (TRUE)
    {
    oc = othercase;
    do
      {
      chr = *cc;
      add_prefix_char(*cc, chars);

      if (caseless)
        add_prefix_char(*oc, chars);

      len--;
      consumed++;
      if (--max_chars == 0)
        return consumed;
      chars += MAX_DIFF_CHARS;
      cc++;
      oc++;
      }
    while (len > 0);

    if (--repeat == 0)
      break;

    len = len_save;
    cc = cc_save;
    }

  repeat = 1;
  if (last)
    return consumed;
  }
}

#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86)

static sljit_s32 character_to_int32(PCRE2_UCHAR chr)
{
sljit_s32 value = (sljit_s32)chr;
#if PCRE2_CODE_UNIT_WIDTH == 8
#define SSE2_COMPARE_TYPE_INDEX 0
return (value << 24) | (value << 16) | (value << 8) | value;
#elif PCRE2_CODE_UNIT_WIDTH == 16
#define SSE2_COMPARE_TYPE_INDEX 1
return (value << 16) | value;
#elif PCRE2_CODE_UNIT_WIDTH == 32
#define SSE2_COMPARE_TYPE_INDEX 2
return value;
#else
#error "Unsupported unit width"
#endif
}

static SLJIT_INLINE void fast_forward_first_char2_sse2(compiler_common *common, PCRE2_UCHAR char1, PCRE2_UCHAR char2)
{
DEFINE_COMPILER;
struct sljit_label *start;
struct sljit_jump *quit[3];
struct sljit_jump *nomatch;
sljit_u8 instruction[8];
sljit_s32 tmp1_ind = sljit_get_register_index(TMP1);
sljit_s32 tmp2_ind = sljit_get_register_index(TMP2);
sljit_s32 str_ptr_ind = sljit_get_register_index(STR_PTR);
BOOL load_twice = FALSE;
PCRE2_UCHAR bit;

bit = char1 ^ char2;
if (!is_powerof2(bit))
  bit = 0;

if ((char1 != char2) && bit == 0)
  load_twice = TRUE;

quit[0] = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);

/* First part (unaligned start) */

OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, character_to_int32(char1 | bit));

SLJIT_ASSERT(tmp1_ind < 8 && tmp2_ind == 1);

/* MOVD xmm, r/m32 */
instruction[0] = 0x66;
instruction[1] = 0x0f;
instruction[2] = 0x6e;
instruction[3] = 0xc0 | (2 << 3) | tmp1_ind;
sljit_emit_op_custom(compiler, instruction, 4);

if (char1 != char2)
  {
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, character_to_int32(bit != 0 ? bit : char2));

  /* MOVD xmm, r/m32 */
  instruction[3] = 0xc0 | (3 << 3) | tmp1_ind;
  sljit_emit_op_custom(compiler, instruction, 4);
  }

/* PSHUFD xmm1, xmm2/m128, imm8 */
instruction[2] = 0x70;
instruction[3] = 0xc0 | (2 << 3) | 2;
instruction[4] = 0;
sljit_emit_op_custom(compiler, instruction, 5);

if (char1 != char2)
  {
  /* PSHUFD xmm1, xmm2/m128, imm8 */
  instruction[3] = 0xc0 | (3 << 3) | 3;
  instruction[4] = 0;
  sljit_emit_op_custom(compiler, instruction, 5);
  }

OP2(SLJIT_AND, TMP2, 0, STR_PTR, 0, SLJIT_IMM, 0xf);
OP2(SLJIT_AND, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, ~0xf);

/* MOVDQA xmm1, xmm2/m128 */
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)

if (str_ptr_ind < 8)
  {
  instruction[2] = 0x6f;
  instruction[3] = (0 << 3) | str_ptr_ind;
  sljit_emit_op_custom(compiler, instruction, 4);

  if (load_twice)
    {
    instruction[3] = (1 << 3) | str_ptr_ind;
    sljit_emit_op_custom(compiler, instruction, 4);
    }
  }
else
  {
  instruction[1] = 0x41;
  instruction[2] = 0x0f;
  instruction[3] = 0x6f;
  instruction[4] = (0 << 3) | (str_ptr_ind & 0x7);
  sljit_emit_op_custom(compiler, instruction, 5);

  if (load_twice)
    {
    instruction[4] = (1 << 3) | str_ptr_ind;
    sljit_emit_op_custom(compiler, instruction, 5);
    }
  instruction[1] = 0x0f;
  }

#else

instruction[2] = 0x6f;
instruction[3] = (0 << 3) | str_ptr_ind;
sljit_emit_op_custom(compiler, instruction, 4);

if (load_twice)
  {
  instruction[3] = (1 << 3) | str_ptr_ind;
  sljit_emit_op_custom(compiler, instruction, 4);
  }

#endif

if (bit != 0)
  {
  /* POR xmm1, xmm2/m128 */
  instruction[2] = 0xeb;
  instruction[3] = 0xc0 | (0 << 3) | 3;
  sljit_emit_op_custom(compiler, instruction, 4);
  }

/* PCMPEQB/W/D xmm1, xmm2/m128 */
instruction[2] = 0x74 + SSE2_COMPARE_TYPE_INDEX;
instruction[3] = 0xc0 | (0 << 3) | 2;
sljit_emit_op_custom(compiler, instruction, 4);

if (load_twice)
  {
  instruction[3] = 0xc0 | (1 << 3) | 3;
  sljit_emit_op_custom(compiler, instruction, 4);
  }

/* PMOVMSKB reg, xmm */
instruction[2] = 0xd7;
instruction[3] = 0xc0 | (tmp1_ind << 3) | 0;
sljit_emit_op_custom(compiler, instruction, 4);

if (load_twice)
  {
  OP1(SLJIT_MOV, RETURN_ADDR, 0, TMP2, 0);
  instruction[3] = 0xc0 | (tmp2_ind << 3) | 1;
  sljit_emit_op_custom(compiler, instruction, 4);

  OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);
  OP1(SLJIT_MOV, TMP2, 0, RETURN_ADDR, 0);
  }

OP2(SLJIT_ASHR, TMP1, 0, TMP1, 0, TMP2, 0);

/* BSF r32, r/m32 */
instruction[0] = 0x0f;
instruction[1] = 0xbc;
instruction[2] = 0xc0 | (tmp1_ind << 3) | tmp1_ind;
sljit_emit_op_custom(compiler, instruction, 3);

nomatch = JUMP(SLJIT_ZERO);

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP2, 0);
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);
quit[1] = JUMP(SLJIT_JUMP);

JUMPHERE(nomatch);

start = LABEL();
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, 16);
quit[2] = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);

/* Second part (aligned) */

instruction[0] = 0x66;
instruction[1] = 0x0f;

/* MOVDQA xmm1, xmm2/m128 */
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)

if (str_ptr_ind < 8)
  {
  instruction[2] = 0x6f;
  instruction[3] = (0 << 3) | str_ptr_ind;
  sljit_emit_op_custom(compiler, instruction, 4);

  if (load_twice)
    {
    instruction[3] = (1 << 3) | str_ptr_ind;
    sljit_emit_op_custom(compiler, instruction, 4);
    }
  }
else
  {
  instruction[1] = 0x41;
  instruction[2] = 0x0f;
  instruction[3] = 0x6f;
  instruction[4] = (0 << 3) | (str_ptr_ind & 0x7);
  sljit_emit_op_custom(compiler, instruction, 5);

  if (load_twice)
    {
    instruction[4] = (1 << 3) | str_ptr_ind;
    sljit_emit_op_custom(compiler, instruction, 5);
    }
  instruction[1] = 0x0f;
  }

#else

instruction[2] = 0x6f;
instruction[3] = (0 << 3) | str_ptr_ind;
sljit_emit_op_custom(compiler, instruction, 4);

if (load_twice)
  {
  instruction[3] = (1 << 3) | str_ptr_ind;
  sljit_emit_op_custom(compiler, instruction, 4);
  }

#endif

if (bit != 0)
  {
  /* POR xmm1, xmm2/m128 */
  instruction[2] = 0xeb;
  instruction[3] = 0xc0 | (0 << 3) | 3;
  sljit_emit_op_custom(compiler, instruction, 4);
  }

/* PCMPEQB/W/D xmm1, xmm2/m128 */
instruction[2] = 0x74 + SSE2_COMPARE_TYPE_INDEX;
instruction[3] = 0xc0 | (0 << 3) | 2;
sljit_emit_op_custom(compiler, instruction, 4);

if (load_twice)
  {
  instruction[3] = 0xc0 | (1 << 3) | 3;
  sljit_emit_op_custom(compiler, instruction, 4);
  }

/* PMOVMSKB reg, xmm */
instruction[2] = 0xd7;
instruction[3] = 0xc0 | (tmp1_ind << 3) | 0;
sljit_emit_op_custom(compiler, instruction, 4);

if (load_twice)
  {
  instruction[3] = 0xc0 | (tmp2_ind << 3) | 1;
  sljit_emit_op_custom(compiler, instruction, 4);

  OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);
  }

/* BSF r32, r/m32 */
instruction[0] = 0x0f;
instruction[1] = 0xbc;
instruction[2] = 0xc0 | (tmp1_ind << 3) | tmp1_ind;
sljit_emit_op_custom(compiler, instruction, 3);

JUMPTO(SLJIT_ZERO, start);

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);

start = LABEL();
SET_LABEL(quit[0], start);
SET_LABEL(quit[1], start);
SET_LABEL(quit[2], start);
}

#undef SSE2_COMPARE_TYPE_INDEX

#endif

static void fast_forward_first_char2(compiler_common *common, PCRE2_UCHAR char1, PCRE2_UCHAR char2, sljit_s32 offset)
{
DEFINE_COMPILER;
struct sljit_label *start;
struct sljit_jump *quit;
struct sljit_jump *found;
PCRE2_UCHAR mask;
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
struct sljit_label *utf_start = NULL;
struct sljit_jump *utf_quit = NULL;
#endif
BOOL has_match_end = (common->match_end_ptr != 0);

if (offset > 0)
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(offset));

if (has_match_end)
  {
  OP1(SLJIT_MOV, TMP3, 0, STR_END, 0);

  OP2(SLJIT_ADD, STR_END, 0, SLJIT_MEM1(SLJIT_SP), common->match_end_ptr, SLJIT_IMM, IN_UCHARS(offset + 1));
#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86)
  if (sljit_x86_is_cmov_available())
    {
    OP2(SLJIT_SUB | SLJIT_SET_U, SLJIT_UNUSED, 0, STR_END, 0, TMP3, 0);
    sljit_x86_emit_cmov(compiler, SLJIT_GREATER, STR_END, TMP3, 0);
    }
#endif
    {
    quit = CMP(SLJIT_LESS_EQUAL, STR_END, 0, TMP3, 0);
    OP1(SLJIT_MOV, STR_END, 0, TMP3, 0);
    JUMPHERE(quit);
    }
  }

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
if (common->utf && offset > 0)
  utf_start = LABEL();
#endif

#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86)

/* SSE2 accelerated first character search. */

if (sljit_x86_is_sse2_available())
  {
  fast_forward_first_char2_sse2(common, char1, char2);

  SLJIT_ASSERT(common->mode == PCRE2_JIT_COMPLETE || offset == 0);
  if (common->mode == PCRE2_JIT_COMPLETE)
    {
    /* In complete mode, we don't need to run a match when STR_PTR == STR_END. */
    SLJIT_ASSERT(common->forced_quit_label == NULL);
    OP1(SLJIT_MOV, SLJIT_RETURN_REG, 0, SLJIT_IMM, PCRE2_ERROR_NOMATCH);
    add_jump(compiler, &common->forced_quit, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
    if (common->utf && offset > 0)
      {
      SLJIT_ASSERT(common->mode == PCRE2_JIT_COMPLETE);

      OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-offset));
      OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
#if PCRE2_CODE_UNIT_WIDTH == 8
      OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xc0);
      CMPTO(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, 0x80, utf_start);
#elif PCRE2_CODE_UNIT_WIDTH == 16
      OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xfc00);
      CMPTO(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, 0xdc00, utf_start);
#else
#error "Unknown code width"
#endif
      OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
      }
#endif

    if (offset > 0)
      OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(offset));
    }
  else if (sljit_x86_is_cmov_available())
    {
    OP2(SLJIT_SUB | SLJIT_SET_U, SLJIT_UNUSED, 0, STR_PTR, 0, STR_END, 0);
    sljit_x86_emit_cmov(compiler, SLJIT_GREATER_EQUAL, STR_PTR, has_match_end ? SLJIT_MEM1(SLJIT_SP) : STR_END, has_match_end ? common->match_end_ptr : 0);
    }
  else
    {
    quit = CMP(SLJIT_LESS, STR_PTR, 0, STR_END, 0);
    OP1(SLJIT_MOV, STR_PTR, 0, has_match_end ? SLJIT_MEM1(SLJIT_SP) : STR_END, has_match_end ? common->match_end_ptr : 0);
    JUMPHERE(quit);
    }

  if (has_match_end)
    OP1(SLJIT_MOV, STR_END, 0, TMP3, 0);
  return;
  }

#endif

quit = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);

start = LABEL();
OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), 0);

if (char1 == char2)
  found = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, char1);
else
  {
  mask = char1 ^ char2;
  if (is_powerof2(mask))
    {
    OP2(SLJIT_OR, TMP1, 0, TMP1, 0, SLJIT_IMM, mask);
    found = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, char1 | mask);
    }
  else
    {
    OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, char1);
    OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_UNUSED, 0, SLJIT_EQUAL);
    OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, char2);
    OP_FLAGS(SLJIT_OR | SLJIT_SET_E, TMP2, 0, TMP2, 0, SLJIT_EQUAL);
    found = JUMP(SLJIT_NOT_ZERO);
    }
  }

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
CMPTO(SLJIT_LESS, STR_PTR, 0, STR_END, 0, start);

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
if (common->utf && offset > 0)
  utf_quit = JUMP(SLJIT_JUMP);
#endif

JUMPHERE(found);

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
if (common->utf && offset > 0)
  {
  OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-offset));
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
#if PCRE2_CODE_UNIT_WIDTH == 8
  OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xc0);
  CMPTO(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, 0x80, utf_start);
#elif PCRE2_CODE_UNIT_WIDTH == 16
  OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xfc00);
  CMPTO(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, 0xdc00, utf_start);
#else
#error "Unknown code width"
#endif
  OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  JUMPHERE(utf_quit);
  }
#endif

JUMPHERE(quit);

if (has_match_end)
  {
  quit = CMP(SLJIT_LESS, STR_PTR, 0, STR_END, 0);
  OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(SLJIT_SP), common->match_end_ptr);
  if (offset > 0)
    OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(offset));
  JUMPHERE(quit);
  OP1(SLJIT_MOV, STR_END, 0, TMP3, 0);
  }

if (offset > 0)
  OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(offset));
}

static SLJIT_INLINE BOOL fast_forward_first_n_chars(compiler_common *common)
{
DEFINE_COMPILER;
struct sljit_label *start;
struct sljit_jump *quit;
struct sljit_jump *match;
/* bytes[0] represent the number of characters between 0
and MAX_N_BYTES - 1, 255 represents any character. */
PCRE2_UCHAR chars[MAX_N_CHARS * MAX_DIFF_CHARS];
sljit_s32 offset;
PCRE2_UCHAR mask;
PCRE2_UCHAR *char_set, *char_set_end;
int i, max, from;
int range_right = -1, range_len;
sljit_u8 *update_table = NULL;
BOOL in_range;
sljit_u32 rec_count;

for (i = 0; i < MAX_N_CHARS; i++)
  chars[i * MAX_DIFF_CHARS] = 0;

rec_count = 10000;
max = scan_prefix(common, common->start, chars, MAX_N_CHARS, &rec_count);

if (max < 1)
  return FALSE;

in_range = FALSE;
/* Prevent compiler "uninitialized" warning */
from = 0;
range_len = 4 /* minimum length */ - 1;
for (i = 0; i <= max; i++)
  {
  if (in_range && (i - from) > range_len && (chars[(i - 1) * MAX_DIFF_CHARS] < 255))
    {
    range_len = i - from;
    range_right = i - 1;
    }

  if (i < max && chars[i * MAX_DIFF_CHARS] < 255)
    {
    SLJIT_ASSERT(chars[i * MAX_DIFF_CHARS] > 0);
    if (!in_range)
      {
      in_range = TRUE;
      from = i;
      }
    }
  else
    in_range = FALSE;
  }

if (range_right >= 0)
  {
  update_table = (sljit_u8 *)allocate_read_only_data(common, 256);
  if (update_table == NULL)
    return TRUE;
  memset(update_table, IN_UCHARS(range_len), 256);

  for (i = 0; i < range_len; i++)
    {
    char_set = chars + ((range_right - i) * MAX_DIFF_CHARS);
    SLJIT_ASSERT(char_set[0] > 0 && char_set[0] < 255);
    char_set_end = char_set + char_set[0];
    char_set++;
    while (char_set <= char_set_end)
      {
      if (update_table[(*char_set) & 0xff] > IN_UCHARS(i))
        update_table[(*char_set) & 0xff] = IN_UCHARS(i);
      char_set++;
      }
    }
  }

offset = -1;
/* Scan forward. */
for (i = 0; i < max; i++)
  {
  if (offset == -1)
    {
    if (chars[i * MAX_DIFF_CHARS] <= 2)
      offset = i;
    }
  else if (chars[offset * MAX_DIFF_CHARS] == 2 && chars[i * MAX_DIFF_CHARS] <= 2)
    {
    if (chars[i * MAX_DIFF_CHARS] == 1)
      offset = i;
    else
      {
      mask = chars[offset * MAX_DIFF_CHARS + 1] ^ chars[offset * MAX_DIFF_CHARS + 2];
      if (!is_powerof2(mask))
        {
        mask = chars[i * MAX_DIFF_CHARS + 1] ^ chars[i * MAX_DIFF_CHARS + 2];
        if (is_powerof2(mask))
          offset = i;
        }
      }
    }
  }

if (range_right < 0)
  {
  if (offset < 0)
    return FALSE;
  SLJIT_ASSERT(chars[offset * MAX_DIFF_CHARS] >= 1 && chars[offset * MAX_DIFF_CHARS] <= 2);
  /* Works regardless the value is 1 or 2. */
  mask = chars[offset * MAX_DIFF_CHARS + chars[offset * MAX_DIFF_CHARS]];
  fast_forward_first_char2(common, chars[offset * MAX_DIFF_CHARS + 1], mask, offset);
  return TRUE;
  }

if (range_right == offset)
  offset = -1;

SLJIT_ASSERT(offset == -1 || (chars[offset * MAX_DIFF_CHARS] >= 1 && chars[offset * MAX_DIFF_CHARS] <= 2));

max -= 1;
SLJIT_ASSERT(max > 0);
if (common->match_end_ptr != 0)
  {
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->match_end_ptr);
  OP1(SLJIT_MOV, TMP3, 0, STR_END, 0);
  OP2(SLJIT_SUB, STR_END, 0, STR_END, 0, SLJIT_IMM, IN_UCHARS(max));
  quit = CMP(SLJIT_LESS_EQUAL, STR_END, 0, TMP1, 0);
  OP1(SLJIT_MOV, STR_END, 0, TMP1, 0);
  JUMPHERE(quit);
  }
else
  OP2(SLJIT_SUB, STR_END, 0, STR_END, 0, SLJIT_IMM, IN_UCHARS(max));

SLJIT_ASSERT(range_right >= 0);

#if !(defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
OP1(SLJIT_MOV, RETURN_ADDR, 0, SLJIT_IMM, (sljit_sw)update_table);
#endif

start = LABEL();
quit = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);

#if PCRE2_CODE_UNIT_WIDTH == 8 || (defined SLJIT_LITTLE_ENDIAN && SLJIT_LITTLE_ENDIAN)
OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(range_right));
#else
OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(range_right + 1) - 1);
#endif

#if !(defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM2(RETURN_ADDR, TMP1), 0);
#else
OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(TMP1), (sljit_sw)update_table);
#endif
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);
CMPTO(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, 0, start);

if (offset >= 0)
  {
  OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(offset));
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

  if (chars[offset * MAX_DIFF_CHARS] == 1)
    CMPTO(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, chars[offset * MAX_DIFF_CHARS + 1], start);
  else
    {
    mask = chars[offset * MAX_DIFF_CHARS + 1] ^ chars[offset * MAX_DIFF_CHARS + 2];
    if (is_powerof2(mask))
      {
      OP2(SLJIT_OR, TMP1, 0, TMP1, 0, SLJIT_IMM, mask);
      CMPTO(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, chars[offset * MAX_DIFF_CHARS + 1] | mask, start);
      }
    else
      {
      match = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, chars[offset * MAX_DIFF_CHARS + 1]);
      CMPTO(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, chars[offset * MAX_DIFF_CHARS + 2], start);
      JUMPHERE(match);
      }
    }
  }

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
if (common->utf && offset != 0)
  {
  if (offset < 0)
    {
    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), 0);
    OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
    }
  else
    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-1));
#if PCRE2_CODE_UNIT_WIDTH == 8
  OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xc0);
  CMPTO(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, 0x80, start);
#elif PCRE2_CODE_UNIT_WIDTH == 16
  OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xfc00);
  CMPTO(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, 0xdc00, start);
#else
#error "Unknown code width"
#endif
  if (offset < 0)
    OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  }
#endif

if (offset >= 0)
  OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

JUMPHERE(quit);

if (common->match_end_ptr != 0)
  {
  if (range_right >= 0)
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->match_end_ptr);
  OP1(SLJIT_MOV, STR_END, 0, TMP3, 0);
  if (range_right >= 0)
    {
    quit = CMP(SLJIT_LESS_EQUAL, STR_PTR, 0, TMP1, 0);
    OP1(SLJIT_MOV, STR_PTR, 0, TMP1, 0);
    JUMPHERE(quit);
    }
  }
else
  OP2(SLJIT_ADD, STR_END, 0, STR_END, 0, SLJIT_IMM, IN_UCHARS(max));
return TRUE;
}

#undef MAX_N_CHARS

static SLJIT_INLINE void fast_forward_first_char(compiler_common *common, PCRE2_UCHAR first_char, BOOL caseless)
{
PCRE2_UCHAR oc;

oc = first_char;
if (caseless)
  {
  oc = TABLE_GET(first_char, common->fcc, first_char);
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 8
  if (first_char > 127 && common->utf)
    oc = UCD_OTHERCASE(first_char);
#endif
  }

fast_forward_first_char2(common, first_char, oc, 0);
}

static SLJIT_INLINE void fast_forward_newline(compiler_common *common)
{
DEFINE_COMPILER;
struct sljit_label *loop;
struct sljit_jump *lastchar;
struct sljit_jump *firstchar;
struct sljit_jump *quit;
struct sljit_jump *foundcr = NULL;
struct sljit_jump *notfoundnl;
jump_list *newline = NULL;

if (common->match_end_ptr != 0)
  {
  OP1(SLJIT_MOV, TMP3, 0, STR_END, 0);
  OP1(SLJIT_MOV, STR_END, 0, SLJIT_MEM1(SLJIT_SP), common->match_end_ptr);
  }

if (common->nltype == NLTYPE_FIXED && common->newline > 255)
  {
  lastchar = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);
  OP1(SLJIT_MOV, TMP1, 0, ARGUMENTS, 0);
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, str));
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, begin));
  firstchar = CMP(SLJIT_LESS_EQUAL, STR_PTR, 0, TMP2, 0);

  OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, IN_UCHARS(2));
  OP2(SLJIT_SUB | SLJIT_SET_U, SLJIT_UNUSED, 0, STR_PTR, 0, TMP1, 0);
  OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_UNUSED, 0, SLJIT_GREATER_EQUAL);
#if PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32
  OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, UCHAR_SHIFT);
#endif
  OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, TMP2, 0);

  loop = LABEL();
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  quit = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);
  OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-2));
  OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-1));
  CMPTO(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, (common->newline >> 8) & 0xff, loop);
  CMPTO(SLJIT_NOT_EQUAL, TMP2, 0, SLJIT_IMM, common->newline & 0xff, loop);

  JUMPHERE(quit);
  JUMPHERE(firstchar);
  JUMPHERE(lastchar);

  if (common->match_end_ptr != 0)
    OP1(SLJIT_MOV, STR_END, 0, TMP3, 0);
  return;
  }

OP1(SLJIT_MOV, TMP1, 0, ARGUMENTS, 0);
OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, str));
firstchar = CMP(SLJIT_LESS_EQUAL, STR_PTR, 0, TMP2, 0);
skip_char_back(common);

loop = LABEL();
common->ff_newline_shortcut = loop;

read_char_range(common, common->nlmin, common->nlmax, TRUE);
lastchar = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);
if (common->nltype == NLTYPE_ANY || common->nltype == NLTYPE_ANYCRLF)
  foundcr = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, CHAR_CR);
check_newlinechar(common, common->nltype, &newline, FALSE);
set_jumps(newline, loop);

if (common->nltype == NLTYPE_ANY || common->nltype == NLTYPE_ANYCRLF)
  {
  quit = JUMP(SLJIT_JUMP);
  JUMPHERE(foundcr);
  notfoundnl = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);
  OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), 0);
  OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, CHAR_NL);
  OP_FLAGS(SLJIT_MOV, TMP1, 0, SLJIT_UNUSED, 0, SLJIT_EQUAL);
#if PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32
  OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, UCHAR_SHIFT);
#endif
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);
  JUMPHERE(notfoundnl);
  JUMPHERE(quit);
  }
JUMPHERE(lastchar);
JUMPHERE(firstchar);

if (common->match_end_ptr != 0)
  OP1(SLJIT_MOV, STR_END, 0, TMP3, 0);
}

static BOOL check_class_ranges(compiler_common *common, const sljit_u8 *bits, BOOL nclass, BOOL invert, jump_list **backtracks);

static SLJIT_INLINE void fast_forward_start_bits(compiler_common *common, const sljit_u8 *start_bits)
{
DEFINE_COMPILER;
struct sljit_label *start;
struct sljit_jump *quit;
struct sljit_jump *found = NULL;
jump_list *matches = NULL;
#if PCRE2_CODE_UNIT_WIDTH != 8
struct sljit_jump *jump;
#endif

if (common->match_end_ptr != 0)
  {
  OP1(SLJIT_MOV, RETURN_ADDR, 0, STR_END, 0);
  OP1(SLJIT_MOV, STR_END, 0, SLJIT_MEM1(SLJIT_SP), common->match_end_ptr);
  }

start = LABEL();
quit = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);
OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), 0);
#ifdef SUPPORT_UNICODE
if (common->utf)
  OP1(SLJIT_MOV, TMP3, 0, TMP1, 0);
#endif

if (!check_class_ranges(common, start_bits, (start_bits[31] & 0x80) != 0, TRUE, &matches))
  {
#if PCRE2_CODE_UNIT_WIDTH != 8
  jump = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 255);
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, 255);
  JUMPHERE(jump);
#endif
  OP2(SLJIT_AND, TMP2, 0, TMP1, 0, SLJIT_IMM, 0x7);
  OP2(SLJIT_LSHR, TMP1, 0, TMP1, 0, SLJIT_IMM, 3);
  OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(TMP1), (sljit_sw)start_bits);
  OP2(SLJIT_SHL, TMP2, 0, SLJIT_IMM, 1, TMP2, 0);
  OP2(SLJIT_AND | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, TMP2, 0);
  found = JUMP(SLJIT_NOT_ZERO);
  }

#ifdef SUPPORT_UNICODE
if (common->utf)
  OP1(SLJIT_MOV, TMP1, 0, TMP3, 0);
#endif
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
#ifdef SUPPORT_UNICODE
#if PCRE2_CODE_UNIT_WIDTH == 8
if (common->utf)
  {
  CMPTO(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0xc0, start);
  OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(TMP1), (sljit_sw)PRIV(utf8_table4) - 0xc0);
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);
  }
#elif PCRE2_CODE_UNIT_WIDTH == 16
if (common->utf)
  {
  CMPTO(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0xd800, start);
  OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xfc00);
  OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0xd800);
  OP_FLAGS(SLJIT_MOV, TMP1, 0, SLJIT_UNUSED, 0, SLJIT_EQUAL);
  OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 1);
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);
  }
#endif /* PCRE2_CODE_UNIT_WIDTH == [8|16] */
#endif /* SUPPORT_UNICODE */
JUMPTO(SLJIT_JUMP, start);
if (found != NULL)
  JUMPHERE(found);
if (matches != NULL)
  set_jumps(matches, LABEL());
JUMPHERE(quit);

if (common->match_end_ptr != 0)
  OP1(SLJIT_MOV, STR_END, 0, RETURN_ADDR, 0);
}

static SLJIT_INLINE struct sljit_jump *search_requested_char(compiler_common *common, PCRE2_UCHAR req_char, BOOL caseless, BOOL has_firstchar)
{
DEFINE_COMPILER;
struct sljit_label *loop;
struct sljit_jump *toolong;
struct sljit_jump *alreadyfound;
struct sljit_jump *found;
struct sljit_jump *foundoc = NULL;
struct sljit_jump *notfound;
sljit_u32 oc, bit;

SLJIT_ASSERT(common->req_char_ptr != 0);
OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), common->req_char_ptr);
OP2(SLJIT_ADD, TMP1, 0, STR_PTR, 0, SLJIT_IMM, REQ_CU_MAX);
toolong = CMP(SLJIT_LESS, TMP1, 0, STR_END, 0);
alreadyfound = CMP(SLJIT_LESS, STR_PTR, 0, TMP2, 0);

if (has_firstchar)
  OP2(SLJIT_ADD, TMP1, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
else
  OP1(SLJIT_MOV, TMP1, 0, STR_PTR, 0);

loop = LABEL();
notfound = CMP(SLJIT_GREATER_EQUAL, TMP1, 0, STR_END, 0);

OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(TMP1), 0);
oc = req_char;
if (caseless)
  {
  oc = TABLE_GET(req_char, common->fcc, req_char);
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 8
  if (req_char > 127 && common->utf)
    oc = UCD_OTHERCASE(req_char);
#endif
  }
if (req_char == oc)
  found = CMP(SLJIT_EQUAL, TMP2, 0, SLJIT_IMM, req_char);
else
  {
  bit = req_char ^ oc;
  if (is_powerof2(bit))
    {
    OP2(SLJIT_OR, TMP2, 0, TMP2, 0, SLJIT_IMM, bit);
    found = CMP(SLJIT_EQUAL, TMP2, 0, SLJIT_IMM, req_char | bit);
    }
  else
    {
    found = CMP(SLJIT_EQUAL, TMP2, 0, SLJIT_IMM, req_char);
    foundoc = CMP(SLJIT_EQUAL, TMP2, 0, SLJIT_IMM, oc);
    }
  }
OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, IN_UCHARS(1));
JUMPTO(SLJIT_JUMP, loop);

JUMPHERE(found);
if (foundoc)
  JUMPHERE(foundoc);
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->req_char_ptr, TMP1, 0);
JUMPHERE(alreadyfound);
JUMPHERE(toolong);
return notfound;
}

static void do_revertframes(compiler_common *common)
{
DEFINE_COMPILER;
struct sljit_jump *jump;
struct sljit_label *mainloop;

sljit_emit_fast_enter(compiler, RETURN_ADDR, 0);
OP1(SLJIT_MOV, TMP1, 0, STACK_TOP, 0);
GET_LOCAL_BASE(TMP3, 0, 0);

/* Drop frames until we reach STACK_TOP. */
mainloop = LABEL();
OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(TMP1), 0);
OP2(SLJIT_SUB | SLJIT_SET_S, SLJIT_UNUSED, 0, TMP2, 0, SLJIT_IMM, 0);
jump = JUMP(SLJIT_SIG_LESS_EQUAL);

OP2(SLJIT_ADD, TMP2, 0, TMP2, 0, TMP3, 0);
OP1(SLJIT_MOV, SLJIT_MEM1(TMP2), 0, SLJIT_MEM1(TMP1), sizeof(sljit_sw));
OP1(SLJIT_MOV, SLJIT_MEM1(TMP2), sizeof(sljit_sw), SLJIT_MEM1(TMP1), 2 * sizeof(sljit_sw));
OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, 3 * sizeof(sljit_sw));
JUMPTO(SLJIT_JUMP, mainloop);

JUMPHERE(jump);
jump = JUMP(SLJIT_SIG_LESS);
/* End of dropping frames. */
sljit_emit_fast_return(compiler, RETURN_ADDR, 0);

JUMPHERE(jump);
OP1(SLJIT_NEG, TMP2, 0, TMP2, 0);
OP2(SLJIT_ADD, TMP2, 0, TMP2, 0, TMP3, 0);
OP1(SLJIT_MOV, SLJIT_MEM1(TMP2), 0, SLJIT_MEM1(TMP1), sizeof(sljit_sw));
OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, 2 * sizeof(sljit_sw));
JUMPTO(SLJIT_JUMP, mainloop);
}

static void check_wordboundary(compiler_common *common)
{
DEFINE_COMPILER;
struct sljit_jump *skipread;
jump_list *skipread_list = NULL;
#if PCRE2_CODE_UNIT_WIDTH != 8 || defined SUPPORT_UNICODE
struct sljit_jump *jump;
#endif

SLJIT_COMPILE_ASSERT(ctype_word == 0x10, ctype_word_must_be_16);

sljit_emit_fast_enter(compiler, SLJIT_MEM1(SLJIT_SP), LOCALS0);
/* Get type of the previous char, and put it to LOCALS1. */
OP1(SLJIT_MOV, TMP1, 0, ARGUMENTS, 0);
OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, begin));
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCALS1, SLJIT_IMM, 0);
skipread = CMP(SLJIT_LESS_EQUAL, STR_PTR, 0, TMP1, 0);
skip_char_back(common);
check_start_used_ptr(common);
read_char(common);

/* Testing char type. */
#ifdef SUPPORT_UNICODE
if (common->use_ucp)
  {
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, 1);
  jump = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, CHAR_UNDERSCORE);
  add_jump(compiler, &common->getucd, JUMP(SLJIT_FAST_CALL));
  OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, ucp_Ll);
  OP2(SLJIT_SUB | SLJIT_SET_U, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, ucp_Lu - ucp_Ll);
  OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_UNUSED, 0, SLJIT_LESS_EQUAL);
  OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, ucp_Nd - ucp_Ll);
  OP2(SLJIT_SUB | SLJIT_SET_U, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, ucp_No - ucp_Nd);
  OP_FLAGS(SLJIT_OR, TMP2, 0, TMP2, 0, SLJIT_LESS_EQUAL);
  JUMPHERE(jump);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCALS1, TMP2, 0);
  }
else
#endif
  {
#if PCRE2_CODE_UNIT_WIDTH != 8
  jump = CMP(SLJIT_GREATER, TMP1, 0, SLJIT_IMM, 255);
#elif defined SUPPORT_UNICODE
  /* Here LOCALS1 has already been zeroed. */
  jump = NULL;
  if (common->utf)
    jump = CMP(SLJIT_GREATER, TMP1, 0, SLJIT_IMM, 255);
#endif /* PCRE2_CODE_UNIT_WIDTH == 8 */
  OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(TMP1), common->ctypes);
  OP2(SLJIT_LSHR, TMP1, 0, TMP1, 0, SLJIT_IMM, 4 /* ctype_word */);
  OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, 1);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCALS1, TMP1, 0);
#if PCRE2_CODE_UNIT_WIDTH != 8
  JUMPHERE(jump);
#elif defined SUPPORT_UNICODE
  if (jump != NULL)
    JUMPHERE(jump);
#endif /* PCRE2_CODE_UNIT_WIDTH == 8 */
  }
JUMPHERE(skipread);

OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, 0);
check_str_end(common, &skipread_list);
peek_char(common, READ_CHAR_MAX);

/* Testing char type. This is a code duplication. */
#ifdef SUPPORT_UNICODE
if (common->use_ucp)
  {
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, 1);
  jump = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, CHAR_UNDERSCORE);
  add_jump(compiler, &common->getucd, JUMP(SLJIT_FAST_CALL));
  OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, ucp_Ll);
  OP2(SLJIT_SUB | SLJIT_SET_U, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, ucp_Lu - ucp_Ll);
  OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_UNUSED, 0, SLJIT_LESS_EQUAL);
  OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, ucp_Nd - ucp_Ll);
  OP2(SLJIT_SUB | SLJIT_SET_U, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, ucp_No - ucp_Nd);
  OP_FLAGS(SLJIT_OR, TMP2, 0, TMP2, 0, SLJIT_LESS_EQUAL);
  JUMPHERE(jump);
  }
else
#endif
  {
#if PCRE2_CODE_UNIT_WIDTH != 8
  /* TMP2 may be destroyed by peek_char. */
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, 0);
  jump = CMP(SLJIT_GREATER, TMP1, 0, SLJIT_IMM, 255);
#elif defined SUPPORT_UNICODE
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, 0);
  jump = NULL;
  if (common->utf)
    jump = CMP(SLJIT_GREATER, TMP1, 0, SLJIT_IMM, 255);
#endif
  OP1(SLJIT_MOV_U8, TMP2, 0, SLJIT_MEM1(TMP1), common->ctypes);
  OP2(SLJIT_LSHR, TMP2, 0, TMP2, 0, SLJIT_IMM, 4 /* ctype_word */);
  OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, 1);
#if PCRE2_CODE_UNIT_WIDTH != 8
  JUMPHERE(jump);
#elif defined SUPPORT_UNICODE
  if (jump != NULL)
    JUMPHERE(jump);
#endif /* PCRE2_CODE_UNIT_WIDTH == 8 */
  }
set_jumps(skipread_list, LABEL());

OP2(SLJIT_XOR | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP2, 0, SLJIT_MEM1(SLJIT_SP), LOCALS1);
sljit_emit_fast_return(compiler, SLJIT_MEM1(SLJIT_SP), LOCALS0);
}

static BOOL check_class_ranges(compiler_common *common, const sljit_u8 *bits, BOOL nclass, BOOL invert, jump_list **backtracks)
{
/* May destroy TMP1. */
DEFINE_COMPILER;
int ranges[MAX_RANGE_SIZE];
sljit_u8 bit, cbit, all;
int i, byte, length = 0;

bit = bits[0] & 0x1;
/* All bits will be zero or one (since bit is zero or one). */
all = -bit;

for (i = 0; i < 256; )
  {
  byte = i >> 3;
  if ((i & 0x7) == 0 && bits[byte] == all)
    i += 8;
  else
    {
    cbit = (bits[byte] >> (i & 0x7)) & 0x1;
    if (cbit != bit)
      {
      if (length >= MAX_RANGE_SIZE)
        return FALSE;
      ranges[length] = i;
      length++;
      bit = cbit;
      all = -cbit;
      }
    i++;
    }
  }

if (((bit == 0) && nclass) || ((bit == 1) && !nclass))
  {
  if (length >= MAX_RANGE_SIZE)
    return FALSE;
  ranges[length] = 256;
  length++;
  }

if (length < 0 || length > 4)
  return FALSE;

bit = bits[0] & 0x1;
if (invert) bit ^= 0x1;

/* No character is accepted. */
if (length == 0 && bit == 0)
  add_jump(compiler, backtracks, JUMP(SLJIT_JUMP));

switch(length)
  {
  case 0:
  /* When bit != 0, all characters are accepted. */
  return TRUE;

  case 1:
  add_jump(compiler, backtracks, CMP(bit == 0 ? SLJIT_LESS : SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, ranges[0]));
  return TRUE;

  case 2:
  if (ranges[0] + 1 != ranges[1])
    {
    OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, ranges[0]);
    add_jump(compiler, backtracks, CMP(bit != 0 ? SLJIT_LESS : SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, ranges[1] - ranges[0]));
    }
  else
    add_jump(compiler, backtracks, CMP(bit != 0 ? SLJIT_EQUAL : SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, ranges[0]));
  return TRUE;

  case 3:
  if (bit != 0)
    {
    add_jump(compiler, backtracks, CMP(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, ranges[2]));
    if (ranges[0] + 1 != ranges[1])
      {
      OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, ranges[0]);
      add_jump(compiler, backtracks, CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, ranges[1] - ranges[0]));
      }
    else
      add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, ranges[0]));
    return TRUE;
    }

  add_jump(compiler, backtracks, CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, ranges[0]));
  if (ranges[1] + 1 != ranges[2])
    {
    OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, ranges[1]);
    add_jump(compiler, backtracks, CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, ranges[2] - ranges[1]));
    }
  else
    add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, ranges[1]));
  return TRUE;

  case 4:
  if ((ranges[1] - ranges[0]) == (ranges[3] - ranges[2])
      && (ranges[0] | (ranges[2] - ranges[0])) == ranges[2]
      && (ranges[1] & (ranges[2] - ranges[0])) == 0
      && is_powerof2(ranges[2] - ranges[0]))
    {
    SLJIT_ASSERT((ranges[0] & (ranges[2] - ranges[0])) == 0 && (ranges[2] & ranges[3] & (ranges[2] - ranges[0])) != 0);
    OP2(SLJIT_OR, TMP1, 0, TMP1, 0, SLJIT_IMM, ranges[2] - ranges[0]);
    if (ranges[2] + 1 != ranges[3])
      {
      OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, ranges[2]);
      add_jump(compiler, backtracks, CMP(bit != 0 ? SLJIT_LESS : SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, ranges[3] - ranges[2]));
      }
    else
      add_jump(compiler, backtracks, CMP(bit != 0 ? SLJIT_EQUAL : SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, ranges[2]));
    return TRUE;
    }

  if (bit != 0)
    {
    i = 0;
    if (ranges[0] + 1 != ranges[1])
      {
      OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, ranges[0]);
      add_jump(compiler, backtracks, CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, ranges[1] - ranges[0]));
      i = ranges[0];
      }
    else
      add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, ranges[0]));

    if (ranges[2] + 1 != ranges[3])
      {
      OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, ranges[2] - i);
      add_jump(compiler, backtracks, CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, ranges[3] - ranges[2]));
      }
    else
      add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, ranges[2] - i));
    return TRUE;
    }

  OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, ranges[0]);
  add_jump(compiler, backtracks, CMP(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, ranges[3] - ranges[0]));
  if (ranges[1] + 1 != ranges[2])
    {
    OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, ranges[1] - ranges[0]);
    add_jump(compiler, backtracks, CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, ranges[2] - ranges[1]));
    }
  else
    add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, ranges[1] - ranges[0]));
  return TRUE;

  default:
  SLJIT_ASSERT_STOP();
  return FALSE;
  }
}

static void check_anynewline(compiler_common *common)
{
/* Check whether TMP1 contains a newline character. TMP2 destroyed. */
DEFINE_COMPILER;

sljit_emit_fast_enter(compiler, RETURN_ADDR, 0);

OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x0a);
OP2(SLJIT_SUB | SLJIT_SET_U, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0x0d - 0x0a);
OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_UNUSED, 0, SLJIT_LESS_EQUAL);
OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0x85 - 0x0a);
#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32
#if PCRE2_CODE_UNIT_WIDTH == 8
if (common->utf)
  {
#endif
  OP_FLAGS(SLJIT_OR, TMP2, 0, TMP2, 0, SLJIT_EQUAL);
  OP2(SLJIT_OR, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x1);
  OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0x2029 - 0x0a);
#if PCRE2_CODE_UNIT_WIDTH == 8
  }
#endif
#endif /* SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH == [16|32] */
OP_FLAGS(SLJIT_OR | SLJIT_SET_E, TMP2, 0, TMP2, 0, SLJIT_EQUAL);
sljit_emit_fast_return(compiler, RETURN_ADDR, 0);
}

static void check_hspace(compiler_common *common)
{
/* Check whether TMP1 contains a newline character. TMP2 destroyed. */
DEFINE_COMPILER;

sljit_emit_fast_enter(compiler, RETURN_ADDR, 0);

OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0x09);
OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_UNUSED, 0, SLJIT_EQUAL);
OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0x20);
OP_FLAGS(SLJIT_OR, TMP2, 0, TMP2, 0, SLJIT_EQUAL);
OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0xa0);
#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32
#if PCRE2_CODE_UNIT_WIDTH == 8
if (common->utf)
  {
#endif
  OP_FLAGS(SLJIT_OR, TMP2, 0, TMP2, 0, SLJIT_EQUAL);
  OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0x1680);
  OP_FLAGS(SLJIT_OR, TMP2, 0, TMP2, 0, SLJIT_EQUAL);
  OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0x180e);
  OP_FLAGS(SLJIT_OR, TMP2, 0, TMP2, 0, SLJIT_EQUAL);
  OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x2000);
  OP2(SLJIT_SUB | SLJIT_SET_U, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0x200A - 0x2000);
  OP_FLAGS(SLJIT_OR, TMP2, 0, TMP2, 0, SLJIT_LESS_EQUAL);
  OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0x202f - 0x2000);
  OP_FLAGS(SLJIT_OR, TMP2, 0, TMP2, 0, SLJIT_EQUAL);
  OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0x205f - 0x2000);
  OP_FLAGS(SLJIT_OR, TMP2, 0, TMP2, 0, SLJIT_EQUAL);
  OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0x3000 - 0x2000);
#if PCRE2_CODE_UNIT_WIDTH == 8
  }
#endif
#endif /* SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH == [16|32] */
OP_FLAGS(SLJIT_OR | SLJIT_SET_E, TMP2, 0, TMP2, 0, SLJIT_EQUAL);

sljit_emit_fast_return(compiler, RETURN_ADDR, 0);
}

static void check_vspace(compiler_common *common)
{
/* Check whether TMP1 contains a newline character. TMP2 destroyed. */
DEFINE_COMPILER;

sljit_emit_fast_enter(compiler, RETURN_ADDR, 0);

OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x0a);
OP2(SLJIT_SUB | SLJIT_SET_U, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0x0d - 0x0a);
OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_UNUSED, 0, SLJIT_LESS_EQUAL);
OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0x85 - 0x0a);
#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32
#if PCRE2_CODE_UNIT_WIDTH == 8
if (common->utf)
  {
#endif
  OP_FLAGS(SLJIT_OR | SLJIT_SET_E, TMP2, 0, TMP2, 0, SLJIT_EQUAL);
  OP2(SLJIT_OR, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x1);
  OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0x2029 - 0x0a);
#if PCRE2_CODE_UNIT_WIDTH == 8
  }
#endif
#endif /* SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH == [16|32] */
OP_FLAGS(SLJIT_OR | SLJIT_SET_E, TMP2, 0, TMP2, 0, SLJIT_EQUAL);

sljit_emit_fast_return(compiler, RETURN_ADDR, 0);
}

#define CHAR1 STR_END
#define CHAR2 STACK_TOP

static void do_casefulcmp(compiler_common *common)
{
DEFINE_COMPILER;
struct sljit_jump *jump;
struct sljit_label *label;

sljit_emit_fast_enter(compiler, RETURN_ADDR, 0);
OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, TMP2, 0);
OP1(SLJIT_MOV, TMP3, 0, CHAR1, 0);
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCALS0, CHAR2, 0);
OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, IN_UCHARS(1));
OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

label = LABEL();
OP1(MOVU_UCHAR, CHAR1, 0, SLJIT_MEM1(TMP1), IN_UCHARS(1));
OP1(MOVU_UCHAR, CHAR2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(1));
jump = CMP(SLJIT_NOT_EQUAL, CHAR1, 0, CHAR2, 0);
OP2(SLJIT_SUB | SLJIT_SET_E, TMP2, 0, TMP2, 0, SLJIT_IMM, IN_UCHARS(1));
JUMPTO(SLJIT_NOT_ZERO, label);

JUMPHERE(jump);
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
OP1(SLJIT_MOV, CHAR1, 0, TMP3, 0);
OP1(SLJIT_MOV, CHAR2, 0, SLJIT_MEM1(SLJIT_SP), LOCALS0);
sljit_emit_fast_return(compiler, RETURN_ADDR, 0);
}

#define LCC_TABLE STACK_LIMIT

static void do_caselesscmp(compiler_common *common)
{
DEFINE_COMPILER;
struct sljit_jump *jump;
struct sljit_label *label;

sljit_emit_fast_enter(compiler, RETURN_ADDR, 0);
OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, TMP2, 0);

OP1(SLJIT_MOV, TMP3, 0, LCC_TABLE, 0);
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCALS0, CHAR1, 0);
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCALS1, CHAR2, 0);
OP1(SLJIT_MOV, LCC_TABLE, 0, SLJIT_IMM, common->lcc);
OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, IN_UCHARS(1));
OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

label = LABEL();
OP1(MOVU_UCHAR, CHAR1, 0, SLJIT_MEM1(TMP1), IN_UCHARS(1));
OP1(MOVU_UCHAR, CHAR2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(1));
#if PCRE2_CODE_UNIT_WIDTH != 8
jump = CMP(SLJIT_GREATER, CHAR1, 0, SLJIT_IMM, 255);
#endif
OP1(SLJIT_MOV_U8, CHAR1, 0, SLJIT_MEM2(LCC_TABLE, CHAR1), 0);
#if PCRE2_CODE_UNIT_WIDTH != 8
JUMPHERE(jump);
jump = CMP(SLJIT_GREATER, CHAR2, 0, SLJIT_IMM, 255);
#endif
OP1(SLJIT_MOV_U8, CHAR2, 0, SLJIT_MEM2(LCC_TABLE, CHAR2), 0);
#if PCRE2_CODE_UNIT_WIDTH != 8
JUMPHERE(jump);
#endif
jump = CMP(SLJIT_NOT_EQUAL, CHAR1, 0, CHAR2, 0);
OP2(SLJIT_SUB | SLJIT_SET_E, TMP2, 0, TMP2, 0, SLJIT_IMM, IN_UCHARS(1));
JUMPTO(SLJIT_NOT_ZERO, label);

JUMPHERE(jump);
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
OP1(SLJIT_MOV, LCC_TABLE, 0, TMP3, 0);
OP1(SLJIT_MOV, CHAR1, 0, SLJIT_MEM1(SLJIT_SP), LOCALS0);
OP1(SLJIT_MOV, CHAR2, 0, SLJIT_MEM1(SLJIT_SP), LOCALS1);
sljit_emit_fast_return(compiler, RETURN_ADDR, 0);
}

#undef LCC_TABLE
#undef CHAR1
#undef CHAR2

#if defined SUPPORT_UNICODE

static PCRE2_SPTR SLJIT_CALL do_utf_caselesscmp(PCRE2_SPTR src1, jit_arguments *args, PCRE2_SPTR end1)
{
/* This function would be ineffective to do in JIT level. */
sljit_u32 c1, c2;
PCRE2_SPTR src2 = args->startchar_ptr;
PCRE2_SPTR end2 = args->end;
const ucd_record *ur;
const sljit_u32 *pp;

while (src1 < end1)
  {
  if (src2 >= end2)
    return (PCRE2_SPTR)1;
  GETCHARINC(c1, src1);
  GETCHARINC(c2, src2);
  ur = GET_UCD(c2);
  if (c1 != c2 && c1 != c2 + ur->other_case)
    {
    pp = PRIV(ucd_caseless_sets) + ur->caseset;
    for (;;)
      {
      if (c1 < *pp) return NULL;
      if (c1 == *pp++) break;
      }
    }
  }
return src2;
}

#endif /* SUPPORT_UNICODE */

static PCRE2_SPTR byte_sequence_compare(compiler_common *common, BOOL caseless, PCRE2_SPTR cc,
    compare_context *context, jump_list **backtracks)
{
DEFINE_COMPILER;
unsigned int othercasebit = 0;
PCRE2_SPTR othercasechar = NULL;
#ifdef SUPPORT_UNICODE
int utflength;
#endif

if (caseless && char_has_othercase(common, cc))
  {
  othercasebit = char_get_othercase_bit(common, cc);
  SLJIT_ASSERT(othercasebit);
  /* Extracting bit difference info. */
#if PCRE2_CODE_UNIT_WIDTH == 8
  othercasechar = cc + (othercasebit >> 8);
  othercasebit &= 0xff;
#elif PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32
  /* Note that this code only handles characters in the BMP. If there
  ever are characters outside the BMP whose othercase differs in only one
  bit from itself (there currently are none), this code will need to be
  revised for PCRE2_CODE_UNIT_WIDTH == 32. */
  othercasechar = cc + (othercasebit >> 9);
  if ((othercasebit & 0x100) != 0)
    othercasebit = (othercasebit & 0xff) << 8;
  else
    othercasebit &= 0xff;
#endif /* PCRE2_CODE_UNIT_WIDTH == [8|16|32] */
  }

if (context->sourcereg == -1)
  {
#if PCRE2_CODE_UNIT_WIDTH == 8
#if defined SLJIT_UNALIGNED && SLJIT_UNALIGNED
  if (context->length >= 4)
    OP1(SLJIT_MOV_S32, TMP1, 0, SLJIT_MEM1(STR_PTR), -context->length);
  else if (context->length >= 2)
    OP1(SLJIT_MOV_U16, TMP1, 0, SLJIT_MEM1(STR_PTR), -context->length);
  else
#endif
    OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(STR_PTR), -context->length);
#elif PCRE2_CODE_UNIT_WIDTH == 16
#if defined SLJIT_UNALIGNED && SLJIT_UNALIGNED
  if (context->length >= 4)
    OP1(SLJIT_MOV_S32, TMP1, 0, SLJIT_MEM1(STR_PTR), -context->length);
  else
#endif
    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), -context->length);
#elif PCRE2_CODE_UNIT_WIDTH == 32
  OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), -context->length);
#endif /* PCRE2_CODE_UNIT_WIDTH == [8|16|32] */
  context->sourcereg = TMP2;
  }

#ifdef SUPPORT_UNICODE
utflength = 1;
if (common->utf && HAS_EXTRALEN(*cc))
  utflength += GET_EXTRALEN(*cc);

do
  {
#endif

  context->length -= IN_UCHARS(1);
#if (defined SLJIT_UNALIGNED && SLJIT_UNALIGNED) && (PCRE2_CODE_UNIT_WIDTH == 8 || PCRE2_CODE_UNIT_WIDTH == 16)

  /* Unaligned read is supported. */
  if (othercasebit != 0 && othercasechar == cc)
    {
    context->c.asuchars[context->ucharptr] = *cc | othercasebit;
    context->oc.asuchars[context->ucharptr] = othercasebit;
    }
  else
    {
    context->c.asuchars[context->ucharptr] = *cc;
    context->oc.asuchars[context->ucharptr] = 0;
    }
  context->ucharptr++;

#if PCRE2_CODE_UNIT_WIDTH == 8
  if (context->ucharptr >= 4 || context->length == 0 || (context->ucharptr == 2 && context->length == 1))
#else
  if (context->ucharptr >= 2 || context->length == 0)
#endif
    {
    if (context->length >= 4)
      OP1(SLJIT_MOV_S32, context->sourcereg, 0, SLJIT_MEM1(STR_PTR), -context->length);
    else if (context->length >= 2)
      OP1(SLJIT_MOV_U16, context->sourcereg, 0, SLJIT_MEM1(STR_PTR), -context->length);
#if PCRE2_CODE_UNIT_WIDTH == 8
    else if (context->length >= 1)
      OP1(SLJIT_MOV_U8, context->sourcereg, 0, SLJIT_MEM1(STR_PTR), -context->length);
#endif /* PCRE2_CODE_UNIT_WIDTH == 8 */
    context->sourcereg = context->sourcereg == TMP1 ? TMP2 : TMP1;

    switch(context->ucharptr)
      {
      case 4 / sizeof(PCRE2_UCHAR):
      if (context->oc.asint != 0)
        OP2(SLJIT_OR, context->sourcereg, 0, context->sourcereg, 0, SLJIT_IMM, context->oc.asint);
      add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, context->sourcereg, 0, SLJIT_IMM, context->c.asint | context->oc.asint));
      break;

      case 2 / sizeof(PCRE2_UCHAR):
      if (context->oc.asushort != 0)
        OP2(SLJIT_OR, context->sourcereg, 0, context->sourcereg, 0, SLJIT_IMM, context->oc.asushort);
      add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, context->sourcereg, 0, SLJIT_IMM, context->c.asushort | context->oc.asushort));
      break;

#if PCRE2_CODE_UNIT_WIDTH == 8
      case 1:
      if (context->oc.asbyte != 0)
        OP2(SLJIT_OR, context->sourcereg, 0, context->sourcereg, 0, SLJIT_IMM, context->oc.asbyte);
      add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, context->sourcereg, 0, SLJIT_IMM, context->c.asbyte | context->oc.asbyte));
      break;
#endif

      default:
      SLJIT_ASSERT_STOP();
      break;
      }
    context->ucharptr = 0;
    }

#else

  /* Unaligned read is unsupported or in 32 bit mode. */
  if (context->length >= 1)
    OP1(MOV_UCHAR, context->sourcereg, 0, SLJIT_MEM1(STR_PTR), -context->length);

  context->sourcereg = context->sourcereg == TMP1 ? TMP2 : TMP1;

  if (othercasebit != 0 && othercasechar == cc)
    {
    OP2(SLJIT_OR, context->sourcereg, 0, context->sourcereg, 0, SLJIT_IMM, othercasebit);
    add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, context->sourcereg, 0, SLJIT_IMM, *cc | othercasebit));
    }
  else
    add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, context->sourcereg, 0, SLJIT_IMM, *cc));

#endif

  cc++;
#ifdef SUPPORT_UNICODE
  utflength--;
  }
while (utflength > 0);
#endif

return cc;
}

#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH != 8

#define SET_TYPE_OFFSET(value) \
  if ((value) != typeoffset) \
    { \
    if ((value) < typeoffset) \
      OP2(SLJIT_ADD, typereg, 0, typereg, 0, SLJIT_IMM, typeoffset - (value)); \
    else \
      OP2(SLJIT_SUB, typereg, 0, typereg, 0, SLJIT_IMM, (value) - typeoffset); \
    } \
  typeoffset = (value);

#define SET_CHAR_OFFSET(value) \
  if ((value) != charoffset) \
    { \
    if ((value) < charoffset) \
      OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, (sljit_sw)(charoffset - (value))); \
    else \
      OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, (sljit_sw)((value) - charoffset)); \
    } \
  charoffset = (value);

static PCRE2_SPTR compile_char1_matchingpath(compiler_common *common, PCRE2_UCHAR type, PCRE2_SPTR cc, jump_list **backtracks, BOOL check_str_ptr);

static void compile_xclass_matchingpath(compiler_common *common, PCRE2_SPTR cc, jump_list **backtracks)
{
DEFINE_COMPILER;
jump_list *found = NULL;
jump_list **list = (cc[0] & XCL_NOT) == 0 ? &found : backtracks;
sljit_uw c, charoffset, max = 256, min = READ_CHAR_MAX;
struct sljit_jump *jump = NULL;
PCRE2_SPTR ccbegin;
int compares, invertcmp, numberofcmps;
#if defined SUPPORT_UNICODE && (PCRE2_CODE_UNIT_WIDTH == 8 || PCRE2_CODE_UNIT_WIDTH == 16)
BOOL utf = common->utf;
#endif

#ifdef SUPPORT_UNICODE
BOOL needstype = FALSE, needsscript = FALSE, needschar = FALSE;
BOOL charsaved = FALSE;
int typereg = TMP1;
const sljit_u32 *other_cases;
sljit_uw typeoffset;
#endif

/* Scanning the necessary info. */
cc++;
ccbegin = cc;
compares = 0;

if (cc[-1] & XCL_MAP)
  {
  min = 0;
  cc += 32 / sizeof(PCRE2_UCHAR);
  }

while (*cc != XCL_END)
  {
  compares++;
  if (*cc == XCL_SINGLE)
    {
    cc ++;
    GETCHARINCTEST(c, cc);
    if (c > max) max = c;
    if (c < min) min = c;
#ifdef SUPPORT_UNICODE
    needschar = TRUE;
#endif
    }
  else if (*cc == XCL_RANGE)
    {
    cc ++;
    GETCHARINCTEST(c, cc);
    if (c < min) min = c;
    GETCHARINCTEST(c, cc);
    if (c > max) max = c;
#ifdef SUPPORT_UNICODE
    needschar = TRUE;
#endif
    }
#ifdef SUPPORT_UNICODE
  else
    {
    SLJIT_ASSERT(*cc == XCL_PROP || *cc == XCL_NOTPROP);
    cc++;
    if (*cc == PT_CLIST)
      {
      other_cases = PRIV(ucd_caseless_sets) + cc[1];
      while (*other_cases != NOTACHAR)
        {
        if (*other_cases > max) max = *other_cases;
        if (*other_cases < min) min = *other_cases;
        other_cases++;
        }
      }
    else
      {
      max = READ_CHAR_MAX;
      min = 0;
      }

    switch(*cc)
      {
      case PT_ANY:
      /* Any either accepts everything or ignored. */
      if (cc[-1] == XCL_PROP)
        {
        compile_char1_matchingpath(common, OP_ALLANY, cc, backtracks, FALSE);
        if (list == backtracks)
          add_jump(compiler, backtracks, JUMP(SLJIT_JUMP));
        return;
        }
      break;

      case PT_LAMP:
      case PT_GC:
      case PT_PC:
      case PT_ALNUM:
      needstype = TRUE;
      break;

      case PT_SC:
      needsscript = TRUE;
      break;

      case PT_SPACE:
      case PT_PXSPACE:
      case PT_WORD:
      case PT_PXGRAPH:
      case PT_PXPRINT:
      case PT_PXPUNCT:
      needstype = TRUE;
      needschar = TRUE;
      break;

      case PT_CLIST:
      case PT_UCNC:
      needschar = TRUE;
      break;

      default:
      SLJIT_ASSERT_STOP();
      break;
      }
    cc += 2;
    }
#endif
  }
SLJIT_ASSERT(compares > 0);

/* We are not necessary in utf mode even in 8 bit mode. */
cc = ccbegin;
read_char_range(common, min, max, (cc[-1] & XCL_NOT) != 0);

if ((cc[-1] & XCL_HASPROP) == 0)
  {
  if ((cc[-1] & XCL_MAP) != 0)
    {
    jump = CMP(SLJIT_GREATER, TMP1, 0, SLJIT_IMM, 255);
    if (!check_class_ranges(common, (const sljit_u8 *)cc, (((const sljit_u8 *)cc)[31] & 0x80) != 0, TRUE, &found))
      {
      OP2(SLJIT_AND, TMP2, 0, TMP1, 0, SLJIT_IMM, 0x7);
      OP2(SLJIT_LSHR, TMP1, 0, TMP1, 0, SLJIT_IMM, 3);
      OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(TMP1), (sljit_sw)cc);
      OP2(SLJIT_SHL, TMP2, 0, SLJIT_IMM, 1, TMP2, 0);
      OP2(SLJIT_AND | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, TMP2, 0);
      add_jump(compiler, &found, JUMP(SLJIT_NOT_ZERO));
      }

    add_jump(compiler, backtracks, JUMP(SLJIT_JUMP));
    JUMPHERE(jump);

    cc += 32 / sizeof(PCRE2_UCHAR);
    }
  else
    {
    OP2(SLJIT_SUB, TMP2, 0, TMP1, 0, SLJIT_IMM, min);
    add_jump(compiler, (cc[-1] & XCL_NOT) == 0 ? backtracks : &found, CMP(SLJIT_GREATER, TMP2, 0, SLJIT_IMM, max - min));
    }
  }
else if ((cc[-1] & XCL_MAP) != 0)
  {
  OP1(SLJIT_MOV, RETURN_ADDR, 0, TMP1, 0);
#ifdef SUPPORT_UNICODE
  charsaved = TRUE;
#endif
  if (!check_class_ranges(common, (const sljit_u8 *)cc, FALSE, TRUE, list))
    {
#if PCRE2_CODE_UNIT_WIDTH == 8
    jump = NULL;
    if (common->utf)
#endif
      jump = CMP(SLJIT_GREATER, TMP1, 0, SLJIT_IMM, 255);

    OP2(SLJIT_AND, TMP2, 0, TMP1, 0, SLJIT_IMM, 0x7);
    OP2(SLJIT_LSHR, TMP1, 0, TMP1, 0, SLJIT_IMM, 3);
    OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(TMP1), (sljit_sw)cc);
    OP2(SLJIT_SHL, TMP2, 0, SLJIT_IMM, 1, TMP2, 0);
    OP2(SLJIT_AND | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, TMP2, 0);
    add_jump(compiler, list, JUMP(SLJIT_NOT_ZERO));

#if PCRE2_CODE_UNIT_WIDTH == 8
    if (common->utf)
#endif
      JUMPHERE(jump);
    }

  OP1(SLJIT_MOV, TMP1, 0, RETURN_ADDR, 0);
  cc += 32 / sizeof(PCRE2_UCHAR);
  }

#ifdef SUPPORT_UNICODE
if (needstype || needsscript)
  {
  if (needschar && !charsaved)
    OP1(SLJIT_MOV, RETURN_ADDR, 0, TMP1, 0);

  OP2(SLJIT_LSHR, TMP2, 0, TMP1, 0, SLJIT_IMM, UCD_BLOCK_SHIFT);
  OP1(SLJIT_MOV_U8, TMP2, 0, SLJIT_MEM1(TMP2), (sljit_sw)PRIV(ucd_stage1));
  OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, UCD_BLOCK_MASK);
  OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, UCD_BLOCK_SHIFT);
  OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, TMP2, 0);
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, (sljit_sw)PRIV(ucd_stage2));
  OP1(SLJIT_MOV_U16, TMP2, 0, SLJIT_MEM2(TMP2, TMP1), 1);

  /* Before anything else, we deal with scripts. */
  if (needsscript)
    {
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, (sljit_sw)PRIV(ucd_records) + SLJIT_OFFSETOF(ucd_record, script));
    OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM2(TMP1, TMP2), 3);

    ccbegin = cc;

    while (*cc != XCL_END)
      {
      if (*cc == XCL_SINGLE)
        {
        cc ++;
        GETCHARINCTEST(c, cc);
        }
      else if (*cc == XCL_RANGE)
        {
        cc ++;
        GETCHARINCTEST(c, cc);
        GETCHARINCTEST(c, cc);
        }
      else
        {
        SLJIT_ASSERT(*cc == XCL_PROP || *cc == XCL_NOTPROP);
        cc++;
        if (*cc == PT_SC)
          {
          compares--;
          invertcmp = (compares == 0 && list != backtracks);
          if (cc[-1] == XCL_NOTPROP)
            invertcmp ^= 0x1;
          jump = CMP(SLJIT_EQUAL ^ invertcmp, TMP1, 0, SLJIT_IMM, (int)cc[1]);
          add_jump(compiler, compares > 0 ? list : backtracks, jump);
          }
        cc += 2;
        }
      }

    cc = ccbegin;
    }

  if (needschar)
    {
    OP1(SLJIT_MOV, TMP1, 0, RETURN_ADDR, 0);
    }

  if (needstype)
    {
    if (!needschar)
      {
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, (sljit_sw)PRIV(ucd_records) + SLJIT_OFFSETOF(ucd_record, chartype));
      OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM2(TMP1, TMP2), 3);
      }
    else
      {
      OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, 3);
      OP1(SLJIT_MOV_U8, RETURN_ADDR, 0, SLJIT_MEM1(TMP2), (sljit_sw)PRIV(ucd_records) + SLJIT_OFFSETOF(ucd_record, chartype));
      typereg = RETURN_ADDR;
      }
    }
  }
#endif

/* Generating code. */
charoffset = 0;
numberofcmps = 0;
#ifdef SUPPORT_UNICODE
typeoffset = 0;
#endif

while (*cc != XCL_END)
  {
  compares--;
  invertcmp = (compares == 0 && list != backtracks);
  jump = NULL;

  if (*cc == XCL_SINGLE)
    {
    cc ++;
    GETCHARINCTEST(c, cc);

    if (numberofcmps < 3 && (*cc == XCL_SINGLE || *cc == XCL_RANGE))
      {
      OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, (sljit_sw)(c - charoffset));
      OP_FLAGS(numberofcmps == 0 ? SLJIT_MOV : SLJIT_OR, TMP2, 0, numberofcmps == 0 ? SLJIT_UNUSED : TMP2, 0, SLJIT_EQUAL);
      numberofcmps++;
      }
    else if (numberofcmps > 0)
      {
      OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, (sljit_sw)(c - charoffset));
      OP_FLAGS(SLJIT_OR | SLJIT_SET_E, TMP2, 0, TMP2, 0, SLJIT_EQUAL);
      jump = JUMP(SLJIT_NOT_ZERO ^ invertcmp);
      numberofcmps = 0;
      }
    else
      {
      jump = CMP(SLJIT_EQUAL ^ invertcmp, TMP1, 0, SLJIT_IMM, (sljit_sw)(c - charoffset));
      numberofcmps = 0;
      }
    }
  else if (*cc == XCL_RANGE)
    {
    cc ++;
    GETCHARINCTEST(c, cc);
    SET_CHAR_OFFSET(c);
    GETCHARINCTEST(c, cc);

    if (numberofcmps < 3 && (*cc == XCL_SINGLE || *cc == XCL_RANGE))
      {
      OP2(SLJIT_SUB | SLJIT_SET_U, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, (sljit_sw)(c - charoffset));
      OP_FLAGS(numberofcmps == 0 ? SLJIT_MOV : SLJIT_OR, TMP2, 0, numberofcmps == 0 ? SLJIT_UNUSED : TMP2, 0, SLJIT_LESS_EQUAL);
      numberofcmps++;
      }
    else if (numberofcmps > 0)
      {
      OP2(SLJIT_SUB | SLJIT_SET_U, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, (sljit_sw)(c - charoffset));
      OP_FLAGS(SLJIT_OR | SLJIT_SET_E, TMP2, 0, TMP2, 0, SLJIT_LESS_EQUAL);
      jump = JUMP(SLJIT_NOT_ZERO ^ invertcmp);
      numberofcmps = 0;
      }
    else
      {
      jump = CMP(SLJIT_LESS_EQUAL ^ invertcmp, TMP1, 0, SLJIT_IMM, (sljit_sw)(c - charoffset));
      numberofcmps = 0;
      }
    }
#ifdef SUPPORT_UNICODE
  else
    {
    SLJIT_ASSERT(*cc == XCL_PROP || *cc == XCL_NOTPROP);
    if (*cc == XCL_NOTPROP)
      invertcmp ^= 0x1;
    cc++;
    switch(*cc)
      {
      case PT_ANY:
      if (!invertcmp)
        jump = JUMP(SLJIT_JUMP);
      break;

      case PT_LAMP:
      OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, typereg, 0, SLJIT_IMM, ucp_Lu - typeoffset);
      OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_UNUSED, 0, SLJIT_EQUAL);
      OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, typereg, 0, SLJIT_IMM, ucp_Ll - typeoffset);
      OP_FLAGS(SLJIT_OR, TMP2, 0, TMP2, 0, SLJIT_EQUAL);
      OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, typereg, 0, SLJIT_IMM, ucp_Lt - typeoffset);
      OP_FLAGS(SLJIT_OR | SLJIT_SET_E, TMP2, 0, TMP2, 0, SLJIT_EQUAL);
      jump = JUMP(SLJIT_NOT_ZERO ^ invertcmp);
      break;

      case PT_GC:
      c = PRIV(ucp_typerange)[(int)cc[1] * 2];
      SET_TYPE_OFFSET(c);
      jump = CMP(SLJIT_LESS_EQUAL ^ invertcmp, typereg, 0, SLJIT_IMM, PRIV(ucp_typerange)[(int)cc[1] * 2 + 1] - c);
      break;

      case PT_PC:
      jump = CMP(SLJIT_EQUAL ^ invertcmp, typereg, 0, SLJIT_IMM, (int)cc[1] - typeoffset);
      break;

      case PT_SC:
      compares++;
      /* Do nothing. */
      break;

      case PT_SPACE:
      case PT_PXSPACE:
      SET_CHAR_OFFSET(9);
      OP2(SLJIT_SUB | SLJIT_SET_U, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0xd - 0x9);
      OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_UNUSED, 0, SLJIT_LESS_EQUAL);

      OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0x85 - 0x9);
      OP_FLAGS(SLJIT_OR | SLJIT_SET_E, TMP2, 0, TMP2, 0, SLJIT_EQUAL);

      OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0x180e - 0x9);
      OP_FLAGS(SLJIT_OR | SLJIT_SET_E, TMP2, 0, TMP2, 0, SLJIT_EQUAL);

      SET_TYPE_OFFSET(ucp_Zl);
      OP2(SLJIT_SUB | SLJIT_SET_U, SLJIT_UNUSED, 0, typereg, 0, SLJIT_IMM, ucp_Zs - ucp_Zl);
      OP_FLAGS(SLJIT_OR | SLJIT_SET_E, TMP2, 0, TMP2, 0, SLJIT_LESS_EQUAL);
      jump = JUMP(SLJIT_NOT_ZERO ^ invertcmp);
      break;

      case PT_WORD:
      OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, (sljit_sw)(CHAR_UNDERSCORE - charoffset));
      OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_UNUSED, 0, SLJIT_EQUAL);
      /* Fall through. */

      case PT_ALNUM:
      SET_TYPE_OFFSET(ucp_Ll);
      OP2(SLJIT_SUB | SLJIT_SET_U, SLJIT_UNUSED, 0, typereg, 0, SLJIT_IMM, ucp_Lu - ucp_Ll);
      OP_FLAGS((*cc == PT_ALNUM) ? SLJIT_MOV : SLJIT_OR, TMP2, 0, (*cc == PT_ALNUM) ? SLJIT_UNUSED : TMP2, 0, SLJIT_LESS_EQUAL);
      SET_TYPE_OFFSET(ucp_Nd);
      OP2(SLJIT_SUB | SLJIT_SET_U, SLJIT_UNUSED, 0, typereg, 0, SLJIT_IMM, ucp_No - ucp_Nd);
      OP_FLAGS(SLJIT_OR | SLJIT_SET_E, TMP2, 0, TMP2, 0, SLJIT_LESS_EQUAL);
      jump = JUMP(SLJIT_NOT_ZERO ^ invertcmp);
      break;

      case PT_CLIST:
      other_cases = PRIV(ucd_caseless_sets) + cc[1];

      /* At least three characters are required.
         Otherwise this case would be handled by the normal code path. */
      SLJIT_ASSERT(other_cases[0] != NOTACHAR && other_cases[1] != NOTACHAR && other_cases[2] != NOTACHAR);
      SLJIT_ASSERT(other_cases[0] < other_cases[1] && other_cases[1] < other_cases[2]);

      /* Optimizing character pairs, if their difference is power of 2. */
      if (is_powerof2(other_cases[1] ^ other_cases[0]))
        {
        if (charoffset == 0)
          OP2(SLJIT_OR, TMP2, 0, TMP1, 0, SLJIT_IMM, other_cases[1] ^ other_cases[0]);
        else
          {
          OP2(SLJIT_ADD, TMP2, 0, TMP1, 0, SLJIT_IMM, (sljit_sw)charoffset);
          OP2(SLJIT_OR, TMP2, 0, TMP2, 0, SLJIT_IMM, other_cases[1] ^ other_cases[0]);
          }
        OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP2, 0, SLJIT_IMM, other_cases[1]);
        OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_UNUSED, 0, SLJIT_EQUAL);
        other_cases += 2;
        }
      else if (is_powerof2(other_cases[2] ^ other_cases[1]))
        {
        if (charoffset == 0)
          OP2(SLJIT_OR, TMP2, 0, TMP1, 0, SLJIT_IMM, other_cases[2] ^ other_cases[1]);
        else
          {
          OP2(SLJIT_ADD, TMP2, 0, TMP1, 0, SLJIT_IMM, (sljit_sw)charoffset);
          OP2(SLJIT_OR, TMP2, 0, TMP2, 0, SLJIT_IMM, other_cases[1] ^ other_cases[0]);
          }
        OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP2, 0, SLJIT_IMM, other_cases[2]);
        OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_UNUSED, 0, SLJIT_EQUAL);

        OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, (sljit_sw)(other_cases[0] - charoffset));
        OP_FLAGS(SLJIT_OR | ((other_cases[3] == NOTACHAR) ? SLJIT_SET_E : 0), TMP2, 0, TMP2, 0, SLJIT_EQUAL);

        other_cases += 3;
        }
      else
        {
        OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, (sljit_sw)(*other_cases++ - charoffset));
        OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_UNUSED, 0, SLJIT_EQUAL);
        }

      while (*other_cases != NOTACHAR)
        {
        OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, (sljit_sw)(*other_cases++ - charoffset));
        OP_FLAGS(SLJIT_OR | ((*other_cases == NOTACHAR) ? SLJIT_SET_E : 0), TMP2, 0, TMP2, 0, SLJIT_EQUAL);
        }
      jump = JUMP(SLJIT_NOT_ZERO ^ invertcmp);
      break;

      case PT_UCNC:
      OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, (sljit_sw)(CHAR_DOLLAR_SIGN - charoffset));
      OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_UNUSED, 0, SLJIT_EQUAL);
      OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, (sljit_sw)(CHAR_COMMERCIAL_AT - charoffset));
      OP_FLAGS(SLJIT_OR, TMP2, 0, TMP2, 0, SLJIT_EQUAL);
      OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, (sljit_sw)(CHAR_GRAVE_ACCENT - charoffset));
      OP_FLAGS(SLJIT_OR, TMP2, 0, TMP2, 0, SLJIT_EQUAL);

      SET_CHAR_OFFSET(0xa0);
      OP2(SLJIT_SUB | SLJIT_SET_U, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, (sljit_sw)(0xd7ff - charoffset));
      OP_FLAGS(SLJIT_OR, TMP2, 0, TMP2, 0, SLJIT_LESS_EQUAL);
      SET_CHAR_OFFSET(0);
      OP2(SLJIT_SUB | SLJIT_SET_U, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0xe000 - 0);
      OP_FLAGS(SLJIT_OR | SLJIT_SET_E, TMP2, 0, TMP2, 0, SLJIT_GREATER_EQUAL);
      jump = JUMP(SLJIT_NOT_ZERO ^ invertcmp);
      break;

      case PT_PXGRAPH:
      /* C and Z groups are the farthest two groups. */
      SET_TYPE_OFFSET(ucp_Ll);
      OP2(SLJIT_SUB | SLJIT_SET_U, SLJIT_UNUSED, 0, typereg, 0, SLJIT_IMM, ucp_So - ucp_Ll);
      OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_UNUSED, 0, SLJIT_GREATER);

      jump = CMP(SLJIT_NOT_EQUAL, typereg, 0, SLJIT_IMM, ucp_Cf - ucp_Ll);

      /* In case of ucp_Cf, we overwrite the result. */
      SET_CHAR_OFFSET(0x2066);
      OP2(SLJIT_SUB | SLJIT_SET_U, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0x2069 - 0x2066);
      OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_UNUSED, 0, SLJIT_LESS_EQUAL);

      OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0x061c - 0x2066);
      OP_FLAGS(SLJIT_OR, TMP2, 0, TMP2, 0, SLJIT_EQUAL);

      OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0x180e - 0x2066);
      OP_FLAGS(SLJIT_OR, TMP2, 0, TMP2, 0, SLJIT_EQUAL);

      JUMPHERE(jump);
      jump = CMP(SLJIT_ZERO ^ invertcmp, TMP2, 0, SLJIT_IMM, 0);
      break;

      case PT_PXPRINT:
      /* C and Z groups are the farthest two groups. */
      SET_TYPE_OFFSET(ucp_Ll);
      OP2(SLJIT_SUB | SLJIT_SET_U, SLJIT_UNUSED, 0, typereg, 0, SLJIT_IMM, ucp_So - ucp_Ll);
      OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_UNUSED, 0, SLJIT_GREATER);

      OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, typereg, 0, SLJIT_IMM, ucp_Zs - ucp_Ll);
      OP_FLAGS(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_NOT_EQUAL);

      jump = CMP(SLJIT_NOT_EQUAL, typereg, 0, SLJIT_IMM, ucp_Cf - ucp_Ll);

      /* In case of ucp_Cf, we overwrite the result. */
      SET_CHAR_OFFSET(0x2066);
      OP2(SLJIT_SUB | SLJIT_SET_U, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0x2069 - 0x2066);
      OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_UNUSED, 0, SLJIT_LESS_EQUAL);

      OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0x061c - 0x2066);
      OP_FLAGS(SLJIT_OR, TMP2, 0, TMP2, 0, SLJIT_EQUAL);

      JUMPHERE(jump);
      jump = CMP(SLJIT_ZERO ^ invertcmp, TMP2, 0, SLJIT_IMM, 0);
      break;

      case PT_PXPUNCT:
      SET_TYPE_OFFSET(ucp_Sc);
      OP2(SLJIT_SUB | SLJIT_SET_U, SLJIT_UNUSED, 0, typereg, 0, SLJIT_IMM, ucp_So - ucp_Sc);
      OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_UNUSED, 0, SLJIT_LESS_EQUAL);

      SET_CHAR_OFFSET(0);
      OP2(SLJIT_SUB | SLJIT_SET_U, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0x7f);
      OP_FLAGS(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_LESS_EQUAL);

      SET_TYPE_OFFSET(ucp_Pc);
      OP2(SLJIT_SUB | SLJIT_SET_U, SLJIT_UNUSED, 0, typereg, 0, SLJIT_IMM, ucp_Ps - ucp_Pc);
      OP_FLAGS(SLJIT_OR | SLJIT_SET_E, TMP2, 0, TMP2, 0, SLJIT_LESS_EQUAL);
      jump = JUMP(SLJIT_NOT_ZERO ^ invertcmp);
      break;

      default:
      SLJIT_ASSERT_STOP();
      break;
      }
    cc += 2;
    }
#endif

  if (jump != NULL)
    add_jump(compiler, compares > 0 ? list : backtracks, jump);
  }

if (found != NULL)
  set_jumps(found, LABEL());
}

#undef SET_TYPE_OFFSET
#undef SET_CHAR_OFFSET

#endif

static PCRE2_SPTR compile_simple_assertion_matchingpath(compiler_common *common, PCRE2_UCHAR type, PCRE2_SPTR cc, jump_list **backtracks)
{
DEFINE_COMPILER;
int length;
struct sljit_jump *jump[4];
#ifdef SUPPORT_UNICODE
struct sljit_label *label;
#endif /* SUPPORT_UNICODE */

switch(type)
  {
  case OP_SOD:
  OP1(SLJIT_MOV, TMP1, 0, ARGUMENTS, 0);
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, begin));
  add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, STR_PTR, 0, TMP1, 0));
  return cc;

  case OP_SOM:
  OP1(SLJIT_MOV, TMP1, 0, ARGUMENTS, 0);
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, str));
  add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, STR_PTR, 0, TMP1, 0));
  return cc;

  case OP_NOT_WORD_BOUNDARY:
  case OP_WORD_BOUNDARY:
  add_jump(compiler, &common->wordboundary, JUMP(SLJIT_FAST_CALL));
  add_jump(compiler, backtracks, JUMP(type == OP_NOT_WORD_BOUNDARY ? SLJIT_NOT_ZERO : SLJIT_ZERO));
  return cc;

  case OP_EODN:
  /* Requires rather complex checks. */
  jump[0] = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);
  if (common->nltype == NLTYPE_FIXED && common->newline > 255)
    {
    OP2(SLJIT_ADD, TMP2, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(2));
    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
    if (common->mode == PCRE2_JIT_COMPLETE)
      add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, TMP2, 0, STR_END, 0));
    else
      {
      jump[1] = CMP(SLJIT_EQUAL, TMP2, 0, STR_END, 0);
      OP2(SLJIT_SUB | SLJIT_SET_U, SLJIT_UNUSED, 0, TMP2, 0, STR_END, 0);
      OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_UNUSED, 0, SLJIT_LESS);
      OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, (common->newline >> 8) & 0xff);
      OP_FLAGS(SLJIT_OR | SLJIT_SET_E, TMP2, 0, TMP2, 0, SLJIT_NOT_EQUAL);
      add_jump(compiler, backtracks, JUMP(SLJIT_NOT_EQUAL));
      check_partial(common, TRUE);
      add_jump(compiler, backtracks, JUMP(SLJIT_JUMP));
      JUMPHERE(jump[1]);
      }
    OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(1));
    add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, (common->newline >> 8) & 0xff));
    add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, TMP2, 0, SLJIT_IMM, common->newline & 0xff));
    }
  else if (common->nltype == NLTYPE_FIXED)
    {
    OP2(SLJIT_ADD, TMP2, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
    add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, TMP2, 0, STR_END, 0));
    add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, common->newline));
    }
  else
    {
    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
    jump[1] = CMP(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, CHAR_CR);
    OP2(SLJIT_ADD, TMP2, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(2));
    OP2(SLJIT_SUB | SLJIT_SET_U, SLJIT_UNUSED, 0, TMP2, 0, STR_END, 0);
    jump[2] = JUMP(SLJIT_GREATER);
    add_jump(compiler, backtracks, JUMP(SLJIT_LESS));
    /* Equal. */
    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(1));
    jump[3] = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, CHAR_NL);
    add_jump(compiler, backtracks, JUMP(SLJIT_JUMP));

    JUMPHERE(jump[1]);
    if (common->nltype == NLTYPE_ANYCRLF)
      {
      OP2(SLJIT_ADD, TMP2, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
      add_jump(compiler, backtracks, CMP(SLJIT_LESS, TMP2, 0, STR_END, 0));
      add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, CHAR_NL));
      }
    else
      {
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCALS1, STR_PTR, 0);
      read_char_range(common, common->nlmin, common->nlmax, TRUE);
      add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, STR_PTR, 0, STR_END, 0));
      add_jump(compiler, &common->anynewline, JUMP(SLJIT_FAST_CALL));
      add_jump(compiler, backtracks, JUMP(SLJIT_ZERO));
      OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(SLJIT_SP), LOCALS1);
      }
    JUMPHERE(jump[2]);
    JUMPHERE(jump[3]);
    }
  JUMPHERE(jump[0]);
  check_partial(common, FALSE);
  return cc;

  case OP_EOD:
  add_jump(compiler, backtracks, CMP(SLJIT_LESS, STR_PTR, 0, STR_END, 0));
  check_partial(common, FALSE);
  return cc;

  case OP_DOLL:
  OP1(SLJIT_MOV, TMP2, 0, ARGUMENTS, 0);
  OP2(SLJIT_AND32 | SLJIT_SET_E, SLJIT_UNUSED, 0, SLJIT_MEM1(TMP2), SLJIT_OFFSETOF(jit_arguments, options), SLJIT_IMM, PCRE2_NOTEOL);
  add_jump(compiler, backtracks, JUMP(SLJIT_NOT_ZERO));

  if (!common->endonly)
    compile_simple_assertion_matchingpath(common, OP_EODN, cc, backtracks);
  else
    {
    add_jump(compiler, backtracks, CMP(SLJIT_LESS, STR_PTR, 0, STR_END, 0));
    check_partial(common, FALSE);
    }
  return cc;

  case OP_DOLLM:
  jump[1] = CMP(SLJIT_LESS, STR_PTR, 0, STR_END, 0);
  OP1(SLJIT_MOV, TMP2, 0, ARGUMENTS, 0);
  OP2(SLJIT_AND32 | SLJIT_SET_E, SLJIT_UNUSED, 0, SLJIT_MEM1(TMP2), SLJIT_OFFSETOF(jit_arguments, options), SLJIT_IMM, PCRE2_NOTEOL);
  add_jump(compiler, backtracks, JUMP(SLJIT_NOT_ZERO));
  check_partial(common, FALSE);
  jump[0] = JUMP(SLJIT_JUMP);
  JUMPHERE(jump[1]);

  if (common->nltype == NLTYPE_FIXED && common->newline > 255)
    {
    OP2(SLJIT_ADD, TMP2, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(2));
    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
    if (common->mode == PCRE2_JIT_COMPLETE)
      add_jump(compiler, backtracks, CMP(SLJIT_GREATER, TMP2, 0, STR_END, 0));
    else
      {
      jump[1] = CMP(SLJIT_LESS_EQUAL, TMP2, 0, STR_END, 0);
      /* STR_PTR = STR_END - IN_UCHARS(1) */
      add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, (common->newline >> 8) & 0xff));
      check_partial(common, TRUE);
      add_jump(compiler, backtracks, JUMP(SLJIT_JUMP));
      JUMPHERE(jump[1]);
      }

    OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(1));
    add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, (common->newline >> 8) & 0xff));
    add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, TMP2, 0, SLJIT_IMM, common->newline & 0xff));
    }
  else
    {
    peek_char(common, common->nlmax);
    check_newlinechar(common, common->nltype, backtracks, FALSE);
    }
  JUMPHERE(jump[0]);
  return cc;

  case OP_CIRC:
  OP1(SLJIT_MOV, TMP2, 0, ARGUMENTS, 0);
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(TMP2), SLJIT_OFFSETOF(jit_arguments, begin));
  add_jump(compiler, backtracks, CMP(SLJIT_GREATER, STR_PTR, 0, TMP1, 0));
  OP2(SLJIT_AND32 | SLJIT_SET_E, SLJIT_UNUSED, 0, SLJIT_MEM1(TMP2), SLJIT_OFFSETOF(jit_arguments, options), SLJIT_IMM, PCRE2_NOTBOL);
  add_jump(compiler, backtracks, JUMP(SLJIT_NOT_ZERO));
  return cc;

  case OP_CIRCM:
  OP1(SLJIT_MOV, TMP2, 0, ARGUMENTS, 0);
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(TMP2), SLJIT_OFFSETOF(jit_arguments, begin));
  jump[1] = CMP(SLJIT_GREATER, STR_PTR, 0, TMP1, 0);
  OP2(SLJIT_AND32 | SLJIT_SET_E, SLJIT_UNUSED, 0, SLJIT_MEM1(TMP2), SLJIT_OFFSETOF(jit_arguments, options), SLJIT_IMM, PCRE2_NOTBOL);
  add_jump(compiler, backtracks, JUMP(SLJIT_NOT_ZERO));
  jump[0] = JUMP(SLJIT_JUMP);
  JUMPHERE(jump[1]);

  if (!common->alt_circumflex)
    add_jump(compiler, backtracks, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));

  if (common->nltype == NLTYPE_FIXED && common->newline > 255)
    {
    OP2(SLJIT_SUB, TMP2, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(2));
    add_jump(compiler, backtracks, CMP(SLJIT_LESS, TMP2, 0, TMP1, 0));
    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-2));
    OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-1));
    add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, (common->newline >> 8) & 0xff));
    add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, TMP2, 0, SLJIT_IMM, common->newline & 0xff));
    }
  else
    {
    skip_char_back(common);
    read_char_range(common, common->nlmin, common->nlmax, TRUE);
    check_newlinechar(common, common->nltype, backtracks, FALSE);
    }
  JUMPHERE(jump[0]);
  return cc;

  case OP_REVERSE:
  length = GET(cc, 0);
  if (length == 0)
    return cc + LINK_SIZE;
  OP1(SLJIT_MOV, TMP1, 0, ARGUMENTS, 0);
#ifdef SUPPORT_UNICODE
  if (common->utf)
    {
    OP1(SLJIT_MOV, TMP3, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, begin));
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, length);
    label = LABEL();
    add_jump(compiler, backtracks, CMP(SLJIT_LESS_EQUAL, STR_PTR, 0, TMP3, 0));
    skip_char_back(common);
    OP2(SLJIT_SUB | SLJIT_SET_E, TMP2, 0, TMP2, 0, SLJIT_IMM, 1);
    JUMPTO(SLJIT_NOT_ZERO, label);
    }
  else
#endif
    {
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, begin));
    OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(length));
    add_jump(compiler, backtracks, CMP(SLJIT_LESS, STR_PTR, 0, TMP1, 0));
    }
  check_start_used_ptr(common);
  return cc + LINK_SIZE;
  }
SLJIT_ASSERT_STOP();
return cc;
}

static PCRE2_SPTR compile_char1_matchingpath(compiler_common *common, PCRE2_UCHAR type, PCRE2_SPTR cc, jump_list **backtracks, BOOL check_str_ptr)
{
DEFINE_COMPILER;
int length;
unsigned int c, oc, bit;
compare_context context;
struct sljit_jump *jump[3];
jump_list *end_list;
#ifdef SUPPORT_UNICODE
struct sljit_label *label;
PCRE2_UCHAR propdata[5];
#endif /* SUPPORT_UNICODE */

switch(type)
  {
  case OP_NOT_DIGIT:
  case OP_DIGIT:
  /* Digits are usually 0-9, so it is worth to optimize them. */
  if (check_str_ptr)
    detect_partial_match(common, backtracks);
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
  if (common->utf && is_char7_bitset((const sljit_u8*)common->ctypes - cbit_length + cbit_digit, FALSE))
    read_char7_type(common, type == OP_NOT_DIGIT);
  else
#endif
    read_char8_type(common, type == OP_NOT_DIGIT);
    /* Flip the starting bit in the negative case. */
  OP2(SLJIT_AND | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, ctype_digit);
  add_jump(compiler, backtracks, JUMP(type == OP_DIGIT ? SLJIT_ZERO : SLJIT_NOT_ZERO));
  return cc;

  case OP_NOT_WHITESPACE:
  case OP_WHITESPACE:
  if (check_str_ptr)
    detect_partial_match(common, backtracks);
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
  if (common->utf && is_char7_bitset((const sljit_u8*)common->ctypes - cbit_length + cbit_space, FALSE))
    read_char7_type(common, type == OP_NOT_WHITESPACE);
  else
#endif
    read_char8_type(common, type == OP_NOT_WHITESPACE);
  OP2(SLJIT_AND | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, ctype_space);
  add_jump(compiler, backtracks, JUMP(type == OP_WHITESPACE ? SLJIT_ZERO : SLJIT_NOT_ZERO));
  return cc;

  case OP_NOT_WORDCHAR:
  case OP_WORDCHAR:
  if (check_str_ptr)
    detect_partial_match(common, backtracks);
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
  if (common->utf && is_char7_bitset((const sljit_u8*)common->ctypes - cbit_length + cbit_word, FALSE))
    read_char7_type(common, type == OP_NOT_WORDCHAR);
  else
#endif
    read_char8_type(common, type == OP_NOT_WORDCHAR);
  OP2(SLJIT_AND | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, ctype_word);
  add_jump(compiler, backtracks, JUMP(type == OP_WORDCHAR ? SLJIT_ZERO : SLJIT_NOT_ZERO));
  return cc;

  case OP_ANY:
  if (check_str_ptr)
    detect_partial_match(common, backtracks);
  read_char_range(common, common->nlmin, common->nlmax, TRUE);
  if (common->nltype == NLTYPE_FIXED && common->newline > 255)
    {
    jump[0] = CMP(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, (common->newline >> 8) & 0xff);
    end_list = NULL;
    if (common->mode != PCRE2_JIT_PARTIAL_HARD)
      add_jump(compiler, &end_list, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));
    else
      check_str_end(common, &end_list);

    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), 0);
    add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, common->newline & 0xff));
    set_jumps(end_list, LABEL());
    JUMPHERE(jump[0]);
    }
  else
    check_newlinechar(common, common->nltype, backtracks, TRUE);
  return cc;

  case OP_ALLANY:
  if (check_str_ptr)
    detect_partial_match(common, backtracks);
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
  if (common->utf)
    {
    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), 0);
    OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
#if PCRE2_CODE_UNIT_WIDTH == 8 || PCRE2_CODE_UNIT_WIDTH == 16
#if PCRE2_CODE_UNIT_WIDTH == 8
    jump[0] = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0xc0);
    OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(TMP1), (sljit_sw)PRIV(utf8_table4) - 0xc0);
    OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);
#elif PCRE2_CODE_UNIT_WIDTH == 16
    jump[0] = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0xd800);
    OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xfc00);
    OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, SLJIT_IMM, 0xd800);
    OP_FLAGS(SLJIT_MOV, TMP1, 0, SLJIT_UNUSED, 0, SLJIT_EQUAL);
    OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 1);
    OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);
#endif
    JUMPHERE(jump[0]);
#endif /* PCRE2_CODE_UNIT_WIDTH == [8|16] */
    return cc;
    }
#endif
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  return cc;

  case OP_ANYBYTE:
  if (check_str_ptr)
    detect_partial_match(common, backtracks);
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  return cc;

#ifdef SUPPORT_UNICODE
  case OP_NOTPROP:
  case OP_PROP:
  propdata[0] = XCL_HASPROP;
  propdata[1] = type == OP_NOTPROP ? XCL_NOTPROP : XCL_PROP;
  propdata[2] = cc[0];
  propdata[3] = cc[1];
  propdata[4] = XCL_END;
  if (check_str_ptr)
    detect_partial_match(common, backtracks);
  compile_xclass_matchingpath(common, propdata, backtracks);
  return cc + 2;
#endif

  case OP_ANYNL:
  if (check_str_ptr)
    detect_partial_match(common, backtracks);
  read_char_range(common, common->bsr_nlmin, common->bsr_nlmax, FALSE);
  jump[0] = CMP(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, CHAR_CR);
  /* We don't need to handle soft partial matching case. */
  end_list = NULL;
  if (common->mode != PCRE2_JIT_PARTIAL_HARD)
    add_jump(compiler, &end_list, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));
  else
    check_str_end(common, &end_list);
  OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), 0);
  jump[1] = CMP(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, CHAR_NL);
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  jump[2] = JUMP(SLJIT_JUMP);
  JUMPHERE(jump[0]);
  check_newlinechar(common, common->bsr_nltype, backtracks, FALSE);
  set_jumps(end_list, LABEL());
  JUMPHERE(jump[1]);
  JUMPHERE(jump[2]);
  return cc;

  case OP_NOT_HSPACE:
  case OP_HSPACE:
  if (check_str_ptr)
    detect_partial_match(common, backtracks);
  read_char_range(common, 0x9, 0x3000, type == OP_NOT_HSPACE);
  add_jump(compiler, &common->hspace, JUMP(SLJIT_FAST_CALL));
  add_jump(compiler, backtracks, JUMP(type == OP_NOT_HSPACE ? SLJIT_NOT_ZERO : SLJIT_ZERO));
  return cc;

  case OP_NOT_VSPACE:
  case OP_VSPACE:
  if (check_str_ptr)
    detect_partial_match(common, backtracks);
  read_char_range(common, 0xa, 0x2029, type == OP_NOT_VSPACE);
  add_jump(compiler, &common->vspace, JUMP(SLJIT_FAST_CALL));
  add_jump(compiler, backtracks, JUMP(type == OP_NOT_VSPACE ? SLJIT_NOT_ZERO : SLJIT_ZERO));
  return cc;

#ifdef SUPPORT_UNICODE
  case OP_EXTUNI:
  if (check_str_ptr)
    detect_partial_match(common, backtracks);
  read_char(common);
  add_jump(compiler, &common->getucd, JUMP(SLJIT_FAST_CALL));
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, (sljit_sw)PRIV(ucd_records) + SLJIT_OFFSETOF(ucd_record, gbprop));
  /* Optimize register allocation: use a real register. */
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCALS0, STACK_TOP, 0);
  OP1(SLJIT_MOV_U8, STACK_TOP, 0, SLJIT_MEM2(TMP1, TMP2), 3);

  label = LABEL();
  jump[0] = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);
  OP1(SLJIT_MOV, TMP3, 0, STR_PTR, 0);
  read_char(common);
  add_jump(compiler, &common->getucd, JUMP(SLJIT_FAST_CALL));
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, (sljit_sw)PRIV(ucd_records) + SLJIT_OFFSETOF(ucd_record, gbprop));
  OP1(SLJIT_MOV_U8, TMP2, 0, SLJIT_MEM2(TMP1, TMP2), 3);

  OP2(SLJIT_SHL, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, 2);
  OP1(SLJIT_MOV_U32, TMP1, 0, SLJIT_MEM1(STACK_TOP), (sljit_sw)PRIV(ucp_gbtable));
  OP1(SLJIT_MOV, STACK_TOP, 0, TMP2, 0);
  OP2(SLJIT_SHL, TMP2, 0, SLJIT_IMM, 1, TMP2, 0);
  OP2(SLJIT_AND | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, TMP2, 0);
  JUMPTO(SLJIT_NOT_ZERO, label);

  OP1(SLJIT_MOV, STR_PTR, 0, TMP3, 0);
  JUMPHERE(jump[0]);
  OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), LOCALS0);

  if (common->mode == PCRE2_JIT_PARTIAL_HARD)
    {
    jump[0] = CMP(SLJIT_LESS, STR_PTR, 0, STR_END, 0);
    /* Since we successfully read a char above, partial matching must occure. */
    check_partial(common, TRUE);
    JUMPHERE(jump[0]);
    }
  return cc;
#endif

  case OP_CHAR:
  case OP_CHARI:
  length = 1;
#ifdef SUPPORT_UNICODE
  if (common->utf && HAS_EXTRALEN(*cc)) length += GET_EXTRALEN(*cc);
#endif
  if (common->mode == PCRE2_JIT_COMPLETE && check_str_ptr
      && (type == OP_CHAR || !char_has_othercase(common, cc) || char_get_othercase_bit(common, cc) != 0))
    {
    OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(length));
    add_jump(compiler, backtracks, CMP(SLJIT_GREATER, STR_PTR, 0, STR_END, 0));

    context.length = IN_UCHARS(length);
    context.sourcereg = -1;
#if defined SLJIT_UNALIGNED && SLJIT_UNALIGNED
    context.ucharptr = 0;
#endif
    return byte_sequence_compare(common, type == OP_CHARI, cc, &context, backtracks);
    }

  if (check_str_ptr)
    detect_partial_match(common, backtracks);
#ifdef SUPPORT_UNICODE
  if (common->utf)
    {
    GETCHAR(c, cc);
    }
  else
#endif
    c = *cc;

  if (type == OP_CHAR || !char_has_othercase(common, cc))
    {
    read_char_range(common, c, c, FALSE);
    add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, c));
    return cc + length;
    }
  oc = char_othercase(common, c);
  read_char_range(common, c < oc ? c : oc, c > oc ? c : oc, FALSE);
  bit = c ^ oc;
  if (is_powerof2(bit))
    {
    OP2(SLJIT_OR, TMP1, 0, TMP1, 0, SLJIT_IMM, bit);
    add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, c | bit));
    return cc + length;
    }
  jump[0] = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, c);
  add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, oc));
  JUMPHERE(jump[0]);
  return cc + length;

  case OP_NOT:
  case OP_NOTI:
  if (check_str_ptr)
    detect_partial_match(common, backtracks);

  length = 1;
#ifdef SUPPORT_UNICODE
  if (common->utf)
    {
#if PCRE2_CODE_UNIT_WIDTH == 8
    c = *cc;
    if (c < 128)
      {
      OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(STR_PTR), 0);
      if (type == OP_NOT || !char_has_othercase(common, cc))
        add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, c));
      else
        {
        /* Since UTF8 code page is fixed, we know that c is in [a-z] or [A-Z] range. */
        OP2(SLJIT_OR, TMP2, 0, TMP1, 0, SLJIT_IMM, 0x20);
        add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP2, 0, SLJIT_IMM, c | 0x20));
        }
      /* Skip the variable-length character. */
      OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
      jump[0] = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0xc0);
      OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(TMP1), (sljit_sw)PRIV(utf8_table4) - 0xc0);
      OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);
      JUMPHERE(jump[0]);
      return cc + 1;
      }
    else
#endif /* PCRE2_CODE_UNIT_WIDTH == 8 */
      {
      GETCHARLEN(c, cc, length);
      }
    }
  else
#endif /* SUPPORT_UNICODE */
    c = *cc;

  if (type == OP_NOT || !char_has_othercase(common, cc))
    {
    read_char_range(common, c, c, TRUE);
    add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, c));
    }
  else
    {
    oc = char_othercase(common, c);
    read_char_range(common, c < oc ? c : oc, c > oc ? c : oc, TRUE);
    bit = c ^ oc;
    if (is_powerof2(bit))
      {
      OP2(SLJIT_OR, TMP1, 0, TMP1, 0, SLJIT_IMM, bit);
      add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, c | bit));
      }
    else
      {
      add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, c));
      add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, oc));
      }
    }
  return cc + length;

  case OP_CLASS:
  case OP_NCLASS:
  if (check_str_ptr)
    detect_partial_match(common, backtracks);

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
  bit = (common->utf && is_char7_bitset((const sljit_u8 *)cc, type == OP_NCLASS)) ? 127 : 255;
  read_char_range(common, 0, bit, type == OP_NCLASS);
#else
  read_char_range(common, 0, 255, type == OP_NCLASS);
#endif

  if (check_class_ranges(common, (const sljit_u8 *)cc, type == OP_NCLASS, FALSE, backtracks))
    return cc + 32 / sizeof(PCRE2_UCHAR);

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
  jump[0] = NULL;
  if (common->utf)
    {
    jump[0] = CMP(SLJIT_GREATER, TMP1, 0, SLJIT_IMM, bit);
    if (type == OP_CLASS)
      {
      add_jump(compiler, backtracks, jump[0]);
      jump[0] = NULL;
      }
    }
#elif PCRE2_CODE_UNIT_WIDTH != 8
  jump[0] = CMP(SLJIT_GREATER, TMP1, 0, SLJIT_IMM, 255);
  if (type == OP_CLASS)
    {
    add_jump(compiler, backtracks, jump[0]);
    jump[0] = NULL;
    }
#endif /* SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8 */

  OP2(SLJIT_AND, TMP2, 0, TMP1, 0, SLJIT_IMM, 0x7);
  OP2(SLJIT_LSHR, TMP1, 0, TMP1, 0, SLJIT_IMM, 3);
  OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(TMP1), (sljit_sw)cc);
  OP2(SLJIT_SHL, TMP2, 0, SLJIT_IMM, 1, TMP2, 0);
  OP2(SLJIT_AND | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP1, 0, TMP2, 0);
  add_jump(compiler, backtracks, JUMP(SLJIT_ZERO));

#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH != 8
  if (jump[0] != NULL)
    JUMPHERE(jump[0]);
#endif
  return cc + 32 / sizeof(PCRE2_UCHAR);

#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32
  case OP_XCLASS:
  if (check_str_ptr)
    detect_partial_match(common, backtracks);
  compile_xclass_matchingpath(common, cc + LINK_SIZE, backtracks);
  return cc + GET(cc, 0) - 1;
#endif
  }
SLJIT_ASSERT_STOP();
return cc;
}

static SLJIT_INLINE PCRE2_SPTR compile_charn_matchingpath(compiler_common *common, PCRE2_SPTR cc, PCRE2_SPTR ccend, jump_list **backtracks)
{
/* This function consumes at least one input character. */
/* To decrease the number of length checks, we try to concatenate the fixed length character sequences. */
DEFINE_COMPILER;
PCRE2_SPTR ccbegin = cc;
compare_context context;
int size;

context.length = 0;
do
  {
  if (cc >= ccend)
    break;

  if (*cc == OP_CHAR)
    {
    size = 1;
#ifdef SUPPORT_UNICODE
    if (common->utf && HAS_EXTRALEN(cc[1]))
      size += GET_EXTRALEN(cc[1]);
#endif
    }
  else if (*cc == OP_CHARI)
    {
    size = 1;
#ifdef SUPPORT_UNICODE
    if (common->utf)
      {
      if (char_has_othercase(common, cc + 1) && char_get_othercase_bit(common, cc + 1) == 0)
        size = 0;
      else if (HAS_EXTRALEN(cc[1]))
        size += GET_EXTRALEN(cc[1]);
      }
    else
#endif
    if (char_has_othercase(common, cc + 1) && char_get_othercase_bit(common, cc + 1) == 0)
      size = 0;
    }
  else
    size = 0;

  cc += 1 + size;
  context.length += IN_UCHARS(size);
  }
while (size > 0 && context.length <= 128);

cc = ccbegin;
if (context.length > 0)
  {
  /* We have a fixed-length byte sequence. */
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, context.length);
  add_jump(compiler, backtracks, CMP(SLJIT_GREATER, STR_PTR, 0, STR_END, 0));

  context.sourcereg = -1;
#if defined SLJIT_UNALIGNED && SLJIT_UNALIGNED
  context.ucharptr = 0;
#endif
  do cc = byte_sequence_compare(common, *cc == OP_CHARI, cc + 1, &context, backtracks); while (context.length > 0);
  return cc;
  }

/* A non-fixed length character will be checked if length == 0. */
return compile_char1_matchingpath(common, *cc, cc + 1, backtracks, TRUE);
}

/* Forward definitions. */
static void compile_matchingpath(compiler_common *, PCRE2_SPTR, PCRE2_SPTR, backtrack_common *);
static void compile_backtrackingpath(compiler_common *, struct backtrack_common *);

#define PUSH_BACKTRACK(size, ccstart, error) \
  do \
    { \
    backtrack = sljit_alloc_memory(compiler, (size)); \
    if (SLJIT_UNLIKELY(sljit_get_compiler_error(compiler))) \
      return error; \
    memset(backtrack, 0, size); \
    backtrack->prev = parent->top; \
    backtrack->cc = (ccstart); \
    parent->top = backtrack; \
    } \
  while (0)

#define PUSH_BACKTRACK_NOVALUE(size, ccstart) \
  do \
    { \
    backtrack = sljit_alloc_memory(compiler, (size)); \
    if (SLJIT_UNLIKELY(sljit_get_compiler_error(compiler))) \
      return; \
    memset(backtrack, 0, size); \
    backtrack->prev = parent->top; \
    backtrack->cc = (ccstart); \
    parent->top = backtrack; \
    } \
  while (0)

#define BACKTRACK_AS(type) ((type *)backtrack)

static void compile_dnref_search(compiler_common *common, PCRE2_SPTR cc, jump_list **backtracks)
{
/* The OVECTOR offset goes to TMP2. */
DEFINE_COMPILER;
int count = GET2(cc, 1 + IMM2_SIZE);
PCRE2_SPTR slot = common->name_table + GET2(cc, 1) * common->name_entry_size;
unsigned int offset;
jump_list *found = NULL;

SLJIT_ASSERT(*cc == OP_DNREF || *cc == OP_DNREFI);

OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(1));

count--;
while (count-- > 0)
  {
  offset = GET2(slot, 0) << 1;
  GET_LOCAL_BASE(TMP2, 0, OVECTOR(offset));
  add_jump(compiler, &found, CMP(SLJIT_NOT_EQUAL, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset), TMP1, 0));
  slot += common->name_entry_size;
  }

offset = GET2(slot, 0) << 1;
GET_LOCAL_BASE(TMP2, 0, OVECTOR(offset));
if (backtracks != NULL && !common->unset_backref)
  add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset), TMP1, 0));

set_jumps(found, LABEL());
}

static void compile_ref_matchingpath(compiler_common *common, PCRE2_SPTR cc, jump_list **backtracks, BOOL withchecks, BOOL emptyfail)
{
DEFINE_COMPILER;
BOOL ref = (*cc == OP_REF || *cc == OP_REFI);
int offset = 0;
struct sljit_jump *jump = NULL;
struct sljit_jump *partial;
struct sljit_jump *nopartial;

if (ref)
  {
  offset = GET2(cc, 1) << 1;
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset));
  /* OVECTOR(1) contains the "string begin - 1" constant. */
  if (withchecks && !common->unset_backref)
    add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(1)));
  }
else
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(TMP2), 0);

#if defined SUPPORT_UNICODE
if (common->utf && *cc == OP_REFI)
  {
  SLJIT_ASSERT(TMP1 == SLJIT_R0 && STACK_TOP == SLJIT_R1 && TMP2 == SLJIT_R2);
  if (ref)
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1));
  else
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(TMP2), sizeof(sljit_sw));

  if (withchecks)
    jump = CMP(SLJIT_EQUAL, TMP1, 0, TMP2, 0);

  /* Needed to save important temporary registers. */
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCALS0, STACK_TOP, 0);
  OP1(SLJIT_MOV, SLJIT_R1, 0, ARGUMENTS, 0);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_R1), SLJIT_OFFSETOF(jit_arguments, startchar_ptr), STR_PTR, 0);
  sljit_emit_ijump(compiler, SLJIT_CALL3, SLJIT_IMM, SLJIT_FUNC_OFFSET(do_utf_caselesscmp));
  OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), LOCALS0);
  if (common->mode == PCRE2_JIT_COMPLETE)
    add_jump(compiler, backtracks, CMP(SLJIT_LESS_EQUAL, SLJIT_RETURN_REG, 0, SLJIT_IMM, 1));
  else
    {
    add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, SLJIT_RETURN_REG, 0, SLJIT_IMM, 0));
    nopartial = CMP(SLJIT_NOT_EQUAL, SLJIT_RETURN_REG, 0, SLJIT_IMM, 1);
    check_partial(common, FALSE);
    add_jump(compiler, backtracks, JUMP(SLJIT_JUMP));
    JUMPHERE(nopartial);
    }
  OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_RETURN_REG, 0);
  }
else
#endif /* SUPPORT_UNICODE */
  {
  if (ref)
    OP2(SLJIT_SUB | SLJIT_SET_E, TMP2, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1), TMP1, 0);
  else
    OP2(SLJIT_SUB | SLJIT_SET_E, TMP2, 0, SLJIT_MEM1(TMP2), sizeof(sljit_sw), TMP1, 0);

  if (withchecks)
    jump = JUMP(SLJIT_ZERO);

  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP2, 0);
  partial = CMP(SLJIT_GREATER, STR_PTR, 0, STR_END, 0);
  if (common->mode == PCRE2_JIT_COMPLETE)
    add_jump(compiler, backtracks, partial);

  add_jump(compiler, *cc == OP_REF ? &common->casefulcmp : &common->caselesscmp, JUMP(SLJIT_FAST_CALL));
  add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, TMP2, 0, SLJIT_IMM, 0));

  if (common->mode != PCRE2_JIT_COMPLETE)
    {
    nopartial = JUMP(SLJIT_JUMP);
    JUMPHERE(partial);
    /* TMP2 -= STR_END - STR_PTR */
    OP2(SLJIT_SUB, TMP2, 0, TMP2, 0, STR_PTR, 0);
    OP2(SLJIT_ADD, TMP2, 0, TMP2, 0, STR_END, 0);
    partial = CMP(SLJIT_EQUAL, TMP2, 0, SLJIT_IMM, 0);
    OP1(SLJIT_MOV, STR_PTR, 0, STR_END, 0);
    add_jump(compiler, *cc == OP_REF ? &common->casefulcmp : &common->caselesscmp, JUMP(SLJIT_FAST_CALL));
    add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, TMP2, 0, SLJIT_IMM, 0));
    JUMPHERE(partial);
    check_partial(common, FALSE);
    add_jump(compiler, backtracks, JUMP(SLJIT_JUMP));
    JUMPHERE(nopartial);
    }
  }

if (jump != NULL)
  {
  if (emptyfail)
    add_jump(compiler, backtracks, jump);
  else
    JUMPHERE(jump);
  }
}

static SLJIT_INLINE PCRE2_SPTR compile_ref_iterator_matchingpath(compiler_common *common, PCRE2_SPTR cc, backtrack_common *parent)
{
DEFINE_COMPILER;
BOOL ref = (*cc == OP_REF || *cc == OP_REFI);
backtrack_common *backtrack;
PCRE2_UCHAR type;
int offset = 0;
struct sljit_label *label;
struct sljit_jump *zerolength;
struct sljit_jump *jump = NULL;
PCRE2_SPTR ccbegin = cc;
int min = 0, max = 0;
BOOL minimize;

PUSH_BACKTRACK(sizeof(ref_iterator_backtrack), cc, NULL);

if (ref)
  offset = GET2(cc, 1) << 1;
else
  cc += IMM2_SIZE;
type = cc[1 + IMM2_SIZE];

SLJIT_COMPILE_ASSERT((OP_CRSTAR & 0x1) == 0, crstar_opcode_must_be_even);
minimize = (type & 0x1) != 0;
switch(type)
  {
  case OP_CRSTAR:
  case OP_CRMINSTAR:
  min = 0;
  max = 0;
  cc += 1 + IMM2_SIZE + 1;
  break;
  case OP_CRPLUS:
  case OP_CRMINPLUS:
  min = 1;
  max = 0;
  cc += 1 + IMM2_SIZE + 1;
  break;
  case OP_CRQUERY:
  case OP_CRMINQUERY:
  min = 0;
  max = 1;
  cc += 1 + IMM2_SIZE + 1;
  break;
  case OP_CRRANGE:
  case OP_CRMINRANGE:
  min = GET2(cc, 1 + IMM2_SIZE + 1);
  max = GET2(cc, 1 + IMM2_SIZE + 1 + IMM2_SIZE);
  cc += 1 + IMM2_SIZE + 1 + 2 * IMM2_SIZE;
  break;
  default:
  SLJIT_ASSERT_STOP();
  break;
  }

if (!minimize)
  {
  if (min == 0)
    {
    allocate_stack(common, 2);
    if (ref)
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset));
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), STR_PTR, 0);
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(1), SLJIT_IMM, 0);
    /* Temporary release of STR_PTR. */
    OP2(SLJIT_SUB, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, sizeof(sljit_sw));
    /* Handles both invalid and empty cases. Since the minimum repeat,
    is zero the invalid case is basically the same as an empty case. */
    if (ref)
      zerolength = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1));
    else
      {
      compile_dnref_search(common, ccbegin, NULL);
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(TMP2), 0);
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), POSSESSIVE1, TMP2, 0);
      zerolength = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_MEM1(TMP2), sizeof(sljit_sw));
      }
    /* Restore if not zero length. */
    OP2(SLJIT_ADD, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, sizeof(sljit_sw));
    }
  else
    {
    allocate_stack(common, 1);
    if (ref)
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset));
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), SLJIT_IMM, 0);
    if (ref)
      {
      add_jump(compiler, &backtrack->topbacktracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(1)));
      zerolength = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1));
      }
    else
      {
      compile_dnref_search(common, ccbegin, &backtrack->topbacktracks);
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(TMP2), 0);
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), POSSESSIVE1, TMP2, 0);
      zerolength = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_MEM1(TMP2), sizeof(sljit_sw));
      }
    }

  if (min > 1 || max > 1)
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), POSSESSIVE0, SLJIT_IMM, 0);

  label = LABEL();
  if (!ref)
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), POSSESSIVE1);
  compile_ref_matchingpath(common, ccbegin, &backtrack->topbacktracks, FALSE, FALSE);

  if (min > 1 || max > 1)
    {
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), POSSESSIVE0);
    OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, 1);
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), POSSESSIVE0, TMP1, 0);
    if (min > 1)
      CMPTO(SLJIT_LESS, TMP1, 0, SLJIT_IMM, min, label);
    if (max > 1)
      {
      jump = CMP(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, max);
      allocate_stack(common, 1);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), STR_PTR, 0);
      JUMPTO(SLJIT_JUMP, label);
      JUMPHERE(jump);
      }
    }

  if (max == 0)
    {
    /* Includes min > 1 case as well. */
    allocate_stack(common, 1);
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), STR_PTR, 0);
    JUMPTO(SLJIT_JUMP, label);
    }

  JUMPHERE(zerolength);
  BACKTRACK_AS(ref_iterator_backtrack)->matchingpath = LABEL();

  count_match(common);
  return cc;
  }

allocate_stack(common, ref ? 2 : 3);
if (ref)
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset));
OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), SLJIT_IMM, 0);
if (type != OP_CRMINSTAR)
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(1), SLJIT_IMM, 0);

if (min == 0)
  {
  /* Handles both invalid and empty cases. Since the minimum repeat,
  is zero the invalid case is basically the same as an empty case. */
  if (ref)
    zerolength = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1));
  else
    {
    compile_dnref_search(common, ccbegin, NULL);
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(TMP2), 0);
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(2), TMP2, 0);
    zerolength = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_MEM1(TMP2), sizeof(sljit_sw));
    }
  /* Length is non-zero, we can match real repeats. */
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), STR_PTR, 0);
  jump = JUMP(SLJIT_JUMP);
  }
else
  {
  if (ref)
    {
    add_jump(compiler, &backtrack->topbacktracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(1)));
    zerolength = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1));
    }
  else
    {
    compile_dnref_search(common, ccbegin, &backtrack->topbacktracks);
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(TMP2), 0);
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(2), TMP2, 0);
    zerolength = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_MEM1(TMP2), sizeof(sljit_sw));
    }
  }

BACKTRACK_AS(ref_iterator_backtrack)->matchingpath = LABEL();
if (max > 0)
  add_jump(compiler, &backtrack->topbacktracks, CMP(SLJIT_GREATER_EQUAL, SLJIT_MEM1(STACK_TOP), STACK(1), SLJIT_IMM, max));

if (!ref)
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(STACK_TOP), STACK(2));
compile_ref_matchingpath(common, ccbegin, &backtrack->topbacktracks, TRUE, TRUE);
OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), STR_PTR, 0);

if (min > 1)
  {
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(1));
  OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, 1);
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(1), TMP1, 0);
  CMPTO(SLJIT_LESS, TMP1, 0, SLJIT_IMM, min, BACKTRACK_AS(ref_iterator_backtrack)->matchingpath);
  }
else if (max > 0)
  OP2(SLJIT_ADD, SLJIT_MEM1(STACK_TOP), STACK(1), SLJIT_MEM1(STACK_TOP), STACK(1), SLJIT_IMM, 1);

if (jump != NULL)
  JUMPHERE(jump);
JUMPHERE(zerolength);

count_match(common);
return cc;
}

static SLJIT_INLINE PCRE2_SPTR compile_recurse_matchingpath(compiler_common *common, PCRE2_SPTR cc, backtrack_common *parent)
{
DEFINE_COMPILER;
backtrack_common *backtrack;
recurse_entry *entry = common->entries;
recurse_entry *prev = NULL;
sljit_sw start = GET(cc, 1);
PCRE2_SPTR start_cc;
BOOL needs_control_head;

PUSH_BACKTRACK(sizeof(recurse_backtrack), cc, NULL);

/* Inlining simple patterns. */
if (get_framesize(common, common->start + start, NULL, TRUE, &needs_control_head) == no_stack)
  {
  start_cc = common->start + start;
  compile_matchingpath(common, next_opcode(common, start_cc), bracketend(start_cc) - (1 + LINK_SIZE), backtrack);
  BACKTRACK_AS(recurse_backtrack)->inlined_pattern = TRUE;
  return cc + 1 + LINK_SIZE;
  }

while (entry != NULL)
  {
  if (entry->start == start)
    break;
  prev = entry;
  entry = entry->next;
  }

if (entry == NULL)
  {
  entry = sljit_alloc_memory(compiler, sizeof(recurse_entry));
  if (SLJIT_UNLIKELY(sljit_get_compiler_error(compiler)))
    return NULL;
  entry->next = NULL;
  entry->entry = NULL;
  entry->calls = NULL;
  entry->start = start;

  if (prev != NULL)
    prev->next = entry;
  else
    common->entries = entry;
  }

if (common->has_set_som && common->mark_ptr != 0)
  {
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(0));
  allocate_stack(common, 2);
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->mark_ptr);
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), TMP2, 0);
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(1), TMP1, 0);
  }
else if (common->has_set_som || common->mark_ptr != 0)
  {
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), common->has_set_som ? (int)(OVECTOR(0)) : common->mark_ptr);
  allocate_stack(common, 1);
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), TMP2, 0);
  }

if (entry->entry == NULL)
  add_jump(compiler, &entry->calls, JUMP(SLJIT_FAST_CALL));
else
  JUMPTO(SLJIT_FAST_CALL, entry->entry);
/* Leave if the match is failed. */
add_jump(compiler, &backtrack->topbacktracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, 0));
return cc + 1 + LINK_SIZE;
}

static int SLJIT_CALL do_callout(struct jit_arguments *arguments, pcre2_callout_block *callout_block, PCRE2_SPTR *jit_ovector)
{
PCRE2_SPTR begin = arguments->begin;
PCRE2_SIZE *ovector = arguments->match_data->ovector;
sljit_u32 oveccount = arguments->oveccount;
sljit_u32 i;

if (arguments->callout == NULL)
  return 0;

callout_block->version = 1;

/* Offsets in subject. */
callout_block->subject_length = arguments->end - arguments->begin;
callout_block->start_match = (PCRE2_SPTR)callout_block->subject - arguments->begin;
callout_block->current_position = (PCRE2_SPTR)callout_block->offset_vector - arguments->begin;
callout_block->subject = begin;

/* Convert and copy the JIT offset vector to the ovector array. */
callout_block->capture_top = 0;
callout_block->offset_vector = ovector;
for (i = 2; i < oveccount; i += 2)
  {
  ovector[i] = jit_ovector[i] - begin;
  ovector[i + 1] = jit_ovector[i + 1] - begin;
  if (jit_ovector[i] >= begin)
    callout_block->capture_top = i;
  }

callout_block->capture_top = (callout_block->capture_top >> 1) + 1;
ovector[0] = PCRE2_UNSET;
ovector[1] = PCRE2_UNSET;
return (arguments->callout)(callout_block, arguments->callout_data);
}

/* Aligning to 8 byte. */
#define CALLOUT_ARG_SIZE \
    (((int)sizeof(pcre2_callout_block) + 7) & ~7)

#define CALLOUT_ARG_OFFSET(arg) \
    (-CALLOUT_ARG_SIZE + SLJIT_OFFSETOF(pcre2_callout_block, arg))

static SLJIT_INLINE PCRE2_SPTR compile_callout_matchingpath(compiler_common *common, PCRE2_SPTR cc, backtrack_common *parent)
{
DEFINE_COMPILER;
backtrack_common *backtrack;
sljit_s32 mov_opcode;
unsigned int callout_length = (*cc == OP_CALLOUT)
    ? PRIV(OP_lengths)[OP_CALLOUT] : GET(cc, 1 + 2 * LINK_SIZE);
sljit_sw value1;
sljit_sw value2;
sljit_sw value3;

PUSH_BACKTRACK(sizeof(backtrack_common), cc, NULL);

allocate_stack(common, CALLOUT_ARG_SIZE / sizeof(sljit_sw));

SLJIT_ASSERT(common->capture_last_ptr != 0);
OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), common->capture_last_ptr);
OP1(SLJIT_MOV, TMP1, 0, ARGUMENTS, 0);
value1 = (*cc == OP_CALLOUT) ? cc[1 + 2 * LINK_SIZE] : 0;
OP1(SLJIT_MOV_U32, SLJIT_MEM1(STACK_TOP), CALLOUT_ARG_OFFSET(callout_number), SLJIT_IMM, value1);
OP1(SLJIT_MOV_U32, SLJIT_MEM1(STACK_TOP), CALLOUT_ARG_OFFSET(capture_last), TMP2, 0);

/* These pointer sized fields temporarly stores internal variables. */
OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(0));
OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), CALLOUT_ARG_OFFSET(offset_vector), STR_PTR, 0);
OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), CALLOUT_ARG_OFFSET(subject), TMP2, 0);

if (common->mark_ptr != 0)
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, mark_ptr));
mov_opcode = (sizeof(PCRE2_SIZE) == 4) ? SLJIT_MOV_U32 : SLJIT_MOV;
OP1(mov_opcode, SLJIT_MEM1(STACK_TOP), CALLOUT_ARG_OFFSET(pattern_position), SLJIT_IMM, GET(cc, 1));
OP1(mov_opcode, SLJIT_MEM1(STACK_TOP), CALLOUT_ARG_OFFSET(next_item_length), SLJIT_IMM, GET(cc, 1 + LINK_SIZE));

if (*cc == OP_CALLOUT)
  {
  value1 = 0;
  value2 = 0;
  value3 = 0;
  }
else
  {
  value1 = (sljit_sw) (cc + (1 + 4*LINK_SIZE) + 1);
  value2 = (callout_length - (1 + 4*LINK_SIZE + 2));
  value3 = (sljit_sw) (GET(cc, 1 + 3*LINK_SIZE));
  }

OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), CALLOUT_ARG_OFFSET(callout_string), SLJIT_IMM, value1);
OP1(mov_opcode, SLJIT_MEM1(STACK_TOP), CALLOUT_ARG_OFFSET(callout_string_length), SLJIT_IMM, value2);
OP1(mov_opcode, SLJIT_MEM1(STACK_TOP), CALLOUT_ARG_OFFSET(callout_string_offset), SLJIT_IMM, value3);
OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), CALLOUT_ARG_OFFSET(mark), (common->mark_ptr != 0) ? TMP2 : SLJIT_IMM, 0);

/* Needed to save important temporary registers. */
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCALS0, STACK_TOP, 0);
OP2(SLJIT_SUB, SLJIT_R1, 0, STACK_TOP, 0, SLJIT_IMM, CALLOUT_ARG_SIZE);
GET_LOCAL_BASE(SLJIT_R2, 0, OVECTOR_START);
sljit_emit_ijump(compiler, SLJIT_CALL3, SLJIT_IMM, SLJIT_FUNC_OFFSET(do_callout));
OP1(SLJIT_MOV_S32, SLJIT_RETURN_REG, 0, SLJIT_RETURN_REG, 0);
OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), LOCALS0);
free_stack(common, CALLOUT_ARG_SIZE / sizeof(sljit_sw));

/* Check return value. */
OP2(SLJIT_SUB | SLJIT_SET_S, SLJIT_UNUSED, 0, SLJIT_RETURN_REG, 0, SLJIT_IMM, 0);
add_jump(compiler, &backtrack->topbacktracks, JUMP(SLJIT_SIG_GREATER));
if (common->forced_quit_label == NULL)
  add_jump(compiler, &common->forced_quit, JUMP(SLJIT_SIG_LESS));
else
  JUMPTO(SLJIT_SIG_LESS, common->forced_quit_label);
return cc + callout_length;
}

#undef CALLOUT_ARG_SIZE
#undef CALLOUT_ARG_OFFSET

static SLJIT_INLINE BOOL assert_needs_str_ptr_saving(PCRE2_SPTR cc)
{
while (TRUE)
  {
  switch (*cc)
    {
    case OP_CALLOUT_STR:
    cc += GET(cc, 1 + 2*LINK_SIZE);
    break;

    case OP_NOT_WORD_BOUNDARY:
    case OP_WORD_BOUNDARY:
    case OP_CIRC:
    case OP_CIRCM:
    case OP_DOLL:
    case OP_DOLLM:
    case OP_CALLOUT:
    case OP_ALT:
    cc += PRIV(OP_lengths)[*cc];
    break;

    case OP_KET:
    return FALSE;

    default:
    return TRUE;
    }
  }
}

static PCRE2_SPTR compile_assert_matchingpath(compiler_common *common, PCRE2_SPTR cc, assert_backtrack *backtrack, BOOL conditional)
{
DEFINE_COMPILER;
int framesize;
int extrasize;
BOOL needs_control_head;
int private_data_ptr;
backtrack_common altbacktrack;
PCRE2_SPTR ccbegin;
PCRE2_UCHAR opcode;
PCRE2_UCHAR bra = OP_BRA;
jump_list *tmp = NULL;
jump_list **target = (conditional) ? &backtrack->condfailed : &backtrack->common.topbacktracks;
jump_list **found;
/* Saving previous accept variables. */
BOOL save_local_exit = common->local_exit;
BOOL save_positive_assert = common->positive_assert;
then_trap_backtrack *save_then_trap = common->then_trap;
struct sljit_label *save_quit_label = common->quit_label;
struct sljit_label *save_accept_label = common->accept_label;
jump_list *save_quit = common->quit;
jump_list *save_positive_assert_quit = common->positive_assert_quit;
jump_list *save_accept = common->accept;
struct sljit_jump *jump;
struct sljit_jump *brajump = NULL;

/* Assert captures then. */
common->then_trap = NULL;

if (*cc == OP_BRAZERO || *cc == OP_BRAMINZERO)
  {
  SLJIT_ASSERT(!conditional);
  bra = *cc;
  cc++;
  }
private_data_ptr = PRIVATE_DATA(cc);
SLJIT_ASSERT(private_data_ptr != 0);
framesize = get_framesize(common, cc, NULL, FALSE, &needs_control_head);
backtrack->framesize = framesize;
backtrack->private_data_ptr = private_data_ptr;
opcode = *cc;
SLJIT_ASSERT(opcode >= OP_ASSERT && opcode <= OP_ASSERTBACK_NOT);
found = (opcode == OP_ASSERT || opcode == OP_ASSERTBACK) ? &tmp : target;
ccbegin = cc;
cc += GET(cc, 1);

if (bra == OP_BRAMINZERO)
  {
  /* This is a braminzero backtrack path. */
  OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
  free_stack(common, 1);
  brajump = CMP(SLJIT_EQUAL, STR_PTR, 0, SLJIT_IMM, 0);
  }

if (framesize < 0)
  {
  extrasize = 1;
  if (bra == OP_BRA && !assert_needs_str_ptr_saving(ccbegin + 1 + LINK_SIZE))
    extrasize = 0;

  if (needs_control_head)
    extrasize++;

  if (framesize == no_frame)
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, STACK_TOP, 0);

  if (extrasize > 0)
    allocate_stack(common, extrasize);

  if (needs_control_head)
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr);

  if (extrasize > 0)
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), STR_PTR, 0);

  if (needs_control_head)
    {
    SLJIT_ASSERT(extrasize == 2);
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, SLJIT_IMM, 0);
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(1), TMP1, 0);
    }
  }
else
  {
  extrasize = needs_control_head ? 3 : 2;
  allocate_stack(common, framesize + extrasize);

  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
  OP2(SLJIT_SUB, TMP2, 0, STACK_TOP, 0, SLJIT_IMM, (framesize + extrasize) * sizeof(sljit_sw));
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, TMP2, 0);
  if (needs_control_head)
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr);
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), STR_PTR, 0);

  if (needs_control_head)
    {
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(2), TMP1, 0);
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(1), TMP2, 0);
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, SLJIT_IMM, 0);
    }
  else
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(1), TMP1, 0);

  init_frame(common, ccbegin, NULL, framesize + extrasize - 1, extrasize, FALSE);
  }

memset(&altbacktrack, 0, sizeof(backtrack_common));
if (opcode == OP_ASSERT_NOT || opcode == OP_ASSERTBACK_NOT)
  {
  /* Negative assert is stronger than positive assert. */
  common->local_exit = TRUE;
  common->quit_label = NULL;
  common->quit = NULL;
  common->positive_assert = FALSE;
  }
else
  common->positive_assert = TRUE;
common->positive_assert_quit = NULL;

while (1)
  {
  common->accept_label = NULL;
  common->accept = NULL;
  altbacktrack.top = NULL;
  altbacktrack.topbacktracks = NULL;

  if (*ccbegin == OP_ALT && extrasize > 0)
    OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(0));

  altbacktrack.cc = ccbegin;
  compile_matchingpath(common, ccbegin + 1 + LINK_SIZE, cc, &altbacktrack);
  if (SLJIT_UNLIKELY(sljit_get_compiler_error(compiler)))
    {
    if (opcode == OP_ASSERT_NOT || opcode == OP_ASSERTBACK_NOT)
      {
      common->local_exit = save_local_exit;
      common->quit_label = save_quit_label;
      common->quit = save_quit;
      }
    common->positive_assert = save_positive_assert;
    common->then_trap = save_then_trap;
    common->accept_label = save_accept_label;
    common->positive_assert_quit = save_positive_assert_quit;
    common->accept = save_accept;
    return NULL;
    }
  common->accept_label = LABEL();
  if (common->accept != NULL)
    set_jumps(common->accept, common->accept_label);

  /* Reset stack. */
  if (framesize < 0)
    {
    if (framesize == no_frame)
      OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
    else if (extrasize > 0)
      free_stack(common, extrasize);

    if (needs_control_head)
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, SLJIT_MEM1(STACK_TOP), 0);
    }
  else
    {
    if ((opcode != OP_ASSERT_NOT && opcode != OP_ASSERTBACK_NOT) || conditional)
      {
      /* We don't need to keep the STR_PTR, only the previous private_data_ptr. */
      OP2(SLJIT_ADD, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr, SLJIT_IMM, (framesize + 1) * sizeof(sljit_sw));
      if (needs_control_head)
        OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, SLJIT_MEM1(STACK_TOP), 0);
      }
    else
      {
      OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
      if (needs_control_head)
        OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, SLJIT_MEM1(STACK_TOP), (framesize + 1) * sizeof(sljit_sw));
      add_jump(compiler, &common->revertframes, JUMP(SLJIT_FAST_CALL));
      }
    }

  if (opcode == OP_ASSERT_NOT || opcode == OP_ASSERTBACK_NOT)
    {
    /* We know that STR_PTR was stored on the top of the stack. */
    if (conditional)
      {
      if (extrasize > 0)
        OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), needs_control_head ? sizeof(sljit_sw) : 0);
      }
    else if (bra == OP_BRAZERO)
      {
      if (framesize < 0)
        OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), (extrasize - 1) * sizeof(sljit_sw));
      else
        {
        OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), framesize * sizeof(sljit_sw));
        OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), (framesize + extrasize - 1) * sizeof(sljit_sw));
        OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, TMP1, 0);
        }
      OP2(SLJIT_ADD, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, sizeof(sljit_sw));
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), SLJIT_IMM, 0);
      }
    else if (framesize >= 0)
      {
      /* For OP_BRA and OP_BRAMINZERO. */
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, SLJIT_MEM1(STACK_TOP), framesize * sizeof(sljit_sw));
      }
    }
  add_jump(compiler, found, JUMP(SLJIT_JUMP));

  compile_backtrackingpath(common, altbacktrack.top);
  if (SLJIT_UNLIKELY(sljit_get_compiler_error(compiler)))
    {
    if (opcode == OP_ASSERT_NOT || opcode == OP_ASSERTBACK_NOT)
      {
      common->local_exit = save_local_exit;
      common->quit_label = save_quit_label;
      common->quit = save_quit;
      }
    common->positive_assert = save_positive_assert;
    common->then_trap = save_then_trap;
    common->accept_label = save_accept_label;
    common->positive_assert_quit = save_positive_assert_quit;
    common->accept = save_accept;
    return NULL;
    }
  set_jumps(altbacktrack.topbacktracks, LABEL());

  if (*cc != OP_ALT)
    break;

  ccbegin = cc;
  cc += GET(cc, 1);
  }

if (opcode == OP_ASSERT_NOT || opcode == OP_ASSERTBACK_NOT)
  {
  SLJIT_ASSERT(common->positive_assert_quit == NULL);
  /* Makes the check less complicated below. */
  common->positive_assert_quit = common->quit;
  }

/* None of them matched. */
if (common->positive_assert_quit != NULL)
  {
  jump = JUMP(SLJIT_JUMP);
  set_jumps(common->positive_assert_quit, LABEL());
  SLJIT_ASSERT(framesize != no_stack);
  if (framesize < 0)
    OP2(SLJIT_ADD, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr, SLJIT_IMM, extrasize * sizeof(sljit_sw));
  else
    {
    OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
    add_jump(compiler, &common->revertframes, JUMP(SLJIT_FAST_CALL));
    OP2(SLJIT_ADD, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, (framesize + extrasize) * sizeof(sljit_sw));
    }
  JUMPHERE(jump);
  }

if (needs_control_head)
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, SLJIT_MEM1(STACK_TOP), STACK(1));

if (opcode == OP_ASSERT || opcode == OP_ASSERTBACK)
  {
  /* Assert is failed. */
  if ((conditional && extrasize > 0) || bra == OP_BRAZERO)
    OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(0));

  if (framesize < 0)
    {
    /* The topmost item should be 0. */
    if (bra == OP_BRAZERO)
      {
      if (extrasize == 2)
        free_stack(common, 1);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), SLJIT_IMM, 0);
      }
    else if (extrasize > 0)
      free_stack(common, extrasize);
    }
  else
    {
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(extrasize - 1));
    /* The topmost item should be 0. */
    if (bra == OP_BRAZERO)
      {
      free_stack(common, framesize + extrasize - 1);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), SLJIT_IMM, 0);
      }
    else
      free_stack(common, framesize + extrasize);
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, TMP1, 0);
    }
  jump = JUMP(SLJIT_JUMP);
  if (bra != OP_BRAZERO)
    add_jump(compiler, target, jump);

  /* Assert is successful. */
  set_jumps(tmp, LABEL());
  if (framesize < 0)
    {
    /* We know that STR_PTR was stored on the top of the stack. */
    if (extrasize > 0)
      OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), (extrasize - 1) * sizeof(sljit_sw));

    /* Keep the STR_PTR on the top of the stack. */
    if (bra == OP_BRAZERO)
      {
      OP2(SLJIT_ADD, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, sizeof(sljit_sw));
      if (extrasize == 2)
        OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), STR_PTR, 0);
      }
    else if (bra == OP_BRAMINZERO)
      {
      OP2(SLJIT_ADD, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, sizeof(sljit_sw));
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), SLJIT_IMM, 0);
      }
    }
  else
    {
    if (bra == OP_BRA)
      {
      /* We don't need to keep the STR_PTR, only the previous private_data_ptr. */
      OP2(SLJIT_ADD, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr, SLJIT_IMM, (framesize + 1) * sizeof(sljit_sw));
      OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), (extrasize - 2) * sizeof(sljit_sw));
      }
    else
      {
      /* We don't need to keep the STR_PTR, only the previous private_data_ptr. */
      OP2(SLJIT_ADD, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr, SLJIT_IMM, (framesize + 2) * sizeof(sljit_sw));
      if (extrasize == 2)
        {
        OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
        if (bra == OP_BRAMINZERO)
          OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), SLJIT_IMM, 0);
        }
      else
        {
        OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), 0);
        OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), bra == OP_BRAZERO ? STR_PTR : SLJIT_IMM, 0);
        }
      }
    }

  if (bra == OP_BRAZERO)
    {
    backtrack->matchingpath = LABEL();
    SET_LABEL(jump, backtrack->matchingpath);
    }
  else if (bra == OP_BRAMINZERO)
    {
    JUMPTO(SLJIT_JUMP, backtrack->matchingpath);
    JUMPHERE(brajump);
    if (framesize >= 0)
      {
      OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
      add_jump(compiler, &common->revertframes, JUMP(SLJIT_FAST_CALL));
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, SLJIT_MEM1(STACK_TOP), framesize * sizeof(sljit_sw));
      }
    set_jumps(backtrack->common.topbacktracks, LABEL());
    }
  }
else
  {
  /* AssertNot is successful. */
  if (framesize < 0)
    {
    if (extrasize > 0)
      OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(0));

    if (bra != OP_BRA)
      {
      if (extrasize == 2)
        free_stack(common, 1);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), SLJIT_IMM, 0);
      }
    else if (extrasize > 0)
      free_stack(common, extrasize);
    }
  else
    {
    OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(extrasize - 1));
    /* The topmost item should be 0. */
    if (bra != OP_BRA)
      {
      free_stack(common, framesize + extrasize - 1);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), SLJIT_IMM, 0);
      }
    else
      free_stack(common, framesize + extrasize);
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, TMP1, 0);
    }

  if (bra == OP_BRAZERO)
    backtrack->matchingpath = LABEL();
  else if (bra == OP_BRAMINZERO)
    {
    JUMPTO(SLJIT_JUMP, backtrack->matchingpath);
    JUMPHERE(brajump);
    }

  if (bra != OP_BRA)
    {
    SLJIT_ASSERT(found == &backtrack->common.topbacktracks);
    set_jumps(backtrack->common.topbacktracks, LABEL());
    backtrack->common.topbacktracks = NULL;
    }
  }

if (opcode == OP_ASSERT_NOT || opcode == OP_ASSERTBACK_NOT)
  {
  common->local_exit = save_local_exit;
  common->quit_label = save_quit_label;
  common->quit = save_quit;
  }
common->positive_assert = save_positive_assert;
common->then_trap = save_then_trap;
common->accept_label = save_accept_label;
common->positive_assert_quit = save_positive_assert_quit;
common->accept = save_accept;
return cc + 1 + LINK_SIZE;
}

static SLJIT_INLINE void match_once_common(compiler_common *common, PCRE2_UCHAR ket, int framesize, int private_data_ptr, BOOL has_alternatives, BOOL needs_control_head)
{
DEFINE_COMPILER;
int stacksize;

if (framesize < 0)
  {
  if (framesize == no_frame)
    OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
  else
    {
    stacksize = needs_control_head ? 1 : 0;
    if (ket != OP_KET || has_alternatives)
      stacksize++;

    if (stacksize > 0)
      free_stack(common, stacksize);
    }

  if (needs_control_head)
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), (ket != OP_KET || has_alternatives) ? sizeof(sljit_sw) : 0);

  /* TMP2 which is set here used by OP_KETRMAX below. */
  if (ket == OP_KETRMAX)
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(STACK_TOP), 0);
  else if (ket == OP_KETRMIN)
    {
    /* Move the STR_PTR to the private_data_ptr. */
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, SLJIT_MEM1(STACK_TOP), 0);
    }
  }
else
  {
  stacksize = (ket != OP_KET || has_alternatives) ? 2 : 1;
  OP2(SLJIT_ADD, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr, SLJIT_IMM, (framesize + stacksize) * sizeof(sljit_sw));
  if (needs_control_head)
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), 0);

  if (ket == OP_KETRMAX)
    {
    /* TMP2 which is set here used by OP_KETRMAX below. */
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
    }
  }
if (needs_control_head)
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, TMP1, 0);
}

static SLJIT_INLINE int match_capture_common(compiler_common *common, int stacksize, int offset, int private_data_ptr)
{
DEFINE_COMPILER;

if (common->capture_last_ptr != 0)
  {
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->capture_last_ptr);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->capture_last_ptr, SLJIT_IMM, offset >> 1);
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(stacksize), TMP1, 0);
  stacksize++;
  }
if (common->optimized_cbracket[offset >> 1] == 0)
  {
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset));
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1));
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(stacksize), TMP1, 0);
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(stacksize + 1), TMP2, 0);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1), STR_PTR, 0);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset), TMP1, 0);
  stacksize += 2;
  }
return stacksize;
}

/*
  Handling bracketed expressions is probably the most complex part.

  Stack layout naming characters:
    S - Push the current STR_PTR
    0 - Push a 0 (NULL)
    A - Push the current STR_PTR. Needed for restoring the STR_PTR
        before the next alternative. Not pushed if there are no alternatives.
    M - Any values pushed by the current alternative. Can be empty, or anything.
    C - Push the previous OVECTOR(i), OVECTOR(i+1) and OVECTOR_PRIV(i) to the stack.
    L - Push the previous local (pointed by localptr) to the stack
   () - opional values stored on the stack
  ()* - optonal, can be stored multiple times

  The following list shows the regular expression templates, their PCRE byte codes
  and stack layout supported by pcre-sljit.

  (?:)                     OP_BRA     | OP_KET                A M
  ()                       OP_CBRA    | OP_KET                C M
  (?:)+                    OP_BRA     | OP_KETRMAX        0   A M S   ( A M S )*
                           OP_SBRA    | OP_KETRMAX        0   L M S   ( L M S )*
  (?:)+?                   OP_BRA     | OP_KETRMIN        0   A M S   ( A M S )*
                           OP_SBRA    | OP_KETRMIN        0   L M S   ( L M S )*
  ()+                      OP_CBRA    | OP_KETRMAX        0   C M S   ( C M S )*
                           OP_SCBRA   | OP_KETRMAX        0   C M S   ( C M S )*
  ()+?                     OP_CBRA    | OP_KETRMIN        0   C M S   ( C M S )*
                           OP_SCBRA   | OP_KETRMIN        0   C M S   ( C M S )*
  (?:)?    OP_BRAZERO    | OP_BRA     | OP_KET            S ( A M 0 )
  (?:)??   OP_BRAMINZERO | OP_BRA     | OP_KET            S ( A M 0 )
  ()?      OP_BRAZERO    | OP_CBRA    | OP_KET            S ( C M 0 )
  ()??     OP_BRAMINZERO | OP_CBRA    | OP_KET            S ( C M 0 )
  (?:)*    OP_BRAZERO    | OP_BRA     | OP_KETRMAX      S 0 ( A M S )*
           OP_BRAZERO    | OP_SBRA    | OP_KETRMAX      S 0 ( L M S )*
  (?:)*?   OP_BRAMINZERO | OP_BRA     | OP_KETRMIN      S 0 ( A M S )*
           OP_BRAMINZERO | OP_SBRA    | OP_KETRMIN      S 0 ( L M S )*
  ()*      OP_BRAZERO    | OP_CBRA    | OP_KETRMAX      S 0 ( C M S )*
           OP_BRAZERO    | OP_SCBRA   | OP_KETRMAX      S 0 ( C M S )*
  ()*?     OP_BRAMINZERO | OP_CBRA    | OP_KETRMIN      S 0 ( C M S )*
           OP_BRAMINZERO | OP_SCBRA   | OP_KETRMIN      S 0 ( C M S )*


  Stack layout naming characters:
    A - Push the alternative index (starting from 0) on the stack.
        Not pushed if there is no alternatives.
    M - Any values pushed by the current alternative. Can be empty, or anything.

  The next list shows the possible content of a bracket:
  (|)     OP_*BRA    | OP_ALT ...         M A
  (?()|)  OP_*COND   | OP_ALT             M A
  (?>|)   OP_ONCE    | OP_ALT ...         [stack trace] M A
  (?>|)   OP_ONCE_NC | OP_ALT ...         [stack trace] M A
                                          Or nothing, if trace is unnecessary
*/

static PCRE2_SPTR compile_bracket_matchingpath(compiler_common *common, PCRE2_SPTR cc, backtrack_common *parent)
{
DEFINE_COMPILER;
backtrack_common *backtrack;
PCRE2_UCHAR opcode;
int private_data_ptr = 0;
int offset = 0;
int i, stacksize;
int repeat_ptr = 0, repeat_length = 0;
int repeat_type = 0, repeat_count = 0;
PCRE2_SPTR ccbegin;
PCRE2_SPTR matchingpath;
PCRE2_SPTR slot;
PCRE2_UCHAR bra = OP_BRA;
PCRE2_UCHAR ket;
assert_backtrack *assert;
BOOL has_alternatives;
BOOL needs_control_head = FALSE;
struct sljit_jump *jump;
struct sljit_jump *skip;
struct sljit_label *rmax_label = NULL;
struct sljit_jump *braminzero = NULL;

PUSH_BACKTRACK(sizeof(bracket_backtrack), cc, NULL);

if (*cc == OP_BRAZERO || *cc == OP_BRAMINZERO)
  {
  bra = *cc;
  cc++;
  opcode = *cc;
  }

opcode = *cc;
ccbegin = cc;
matchingpath = bracketend(cc) - 1 - LINK_SIZE;
ket = *matchingpath;
if (ket == OP_KET && PRIVATE_DATA(matchingpath) != 0)
  {
  repeat_ptr = PRIVATE_DATA(matchingpath);
  repeat_length = PRIVATE_DATA(matchingpath + 1);
  repeat_type = PRIVATE_DATA(matchingpath + 2);
  repeat_count = PRIVATE_DATA(matchingpath + 3);
  SLJIT_ASSERT(repeat_length != 0 && repeat_type != 0 && repeat_count != 0);
  if (repeat_type == OP_UPTO)
    ket = OP_KETRMAX;
  if (repeat_type == OP_MINUPTO)
    ket = OP_KETRMIN;
  }

matchingpath = ccbegin + 1 + LINK_SIZE;
SLJIT_ASSERT(ket == OP_KET || ket == OP_KETRMAX || ket == OP_KETRMIN);
SLJIT_ASSERT(!((bra == OP_BRAZERO && ket == OP_KETRMIN) || (bra == OP_BRAMINZERO && ket == OP_KETRMAX)));
cc += GET(cc, 1);

has_alternatives = *cc == OP_ALT;
if (SLJIT_UNLIKELY(opcode == OP_COND || opcode == OP_SCOND))
  {
  SLJIT_COMPILE_ASSERT(OP_DNRREF == OP_RREF + 1 && OP_FALSE == OP_RREF + 2 && OP_TRUE == OP_RREF + 3,
    compile_time_checks_must_be_grouped_together);
  has_alternatives = ((*matchingpath >= OP_RREF && *matchingpath <= OP_TRUE) || *matchingpath == OP_FAIL) ? FALSE : TRUE;
  }

if (SLJIT_UNLIKELY(opcode == OP_COND) && (*cc == OP_KETRMAX || *cc == OP_KETRMIN))
  opcode = OP_SCOND;
if (SLJIT_UNLIKELY(opcode == OP_ONCE_NC))
  opcode = OP_ONCE;

if (opcode == OP_CBRA || opcode == OP_SCBRA)
  {
  /* Capturing brackets has a pre-allocated space. */
  offset = GET2(ccbegin, 1 + LINK_SIZE);
  if (common->optimized_cbracket[offset] == 0)
    {
    private_data_ptr = OVECTOR_PRIV(offset);
    offset <<= 1;
    }
  else
    {
    offset <<= 1;
    private_data_ptr = OVECTOR(offset);
    }
  BACKTRACK_AS(bracket_backtrack)->private_data_ptr = private_data_ptr;
  matchingpath += IMM2_SIZE;
  }
else if (opcode == OP_ONCE || opcode == OP_SBRA || opcode == OP_SCOND)
  {
  /* Other brackets simply allocate the next entry. */
  private_data_ptr = PRIVATE_DATA(ccbegin);
  SLJIT_ASSERT(private_data_ptr != 0);
  BACKTRACK_AS(bracket_backtrack)->private_data_ptr = private_data_ptr;
  if (opcode == OP_ONCE)
    BACKTRACK_AS(bracket_backtrack)->u.framesize = get_framesize(common, ccbegin, NULL, FALSE, &needs_control_head);
  }

/* Instructions before the first alternative. */
stacksize = 0;
if (ket == OP_KETRMAX || (ket == OP_KETRMIN && bra != OP_BRAMINZERO))
  stacksize++;
if (bra == OP_BRAZERO)
  stacksize++;

if (stacksize > 0)
  allocate_stack(common, stacksize);

stacksize = 0;
if (ket == OP_KETRMAX || (ket == OP_KETRMIN && bra != OP_BRAMINZERO))
  {
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(stacksize), SLJIT_IMM, 0);
  stacksize++;
  }

if (bra == OP_BRAZERO)
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(stacksize), STR_PTR, 0);

if (bra == OP_BRAMINZERO)
  {
  /* This is a backtrack path! (Since the try-path of OP_BRAMINZERO matches to the empty string) */
  OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
  if (ket != OP_KETRMIN)
    {
    free_stack(common, 1);
    braminzero = CMP(SLJIT_EQUAL, STR_PTR, 0, SLJIT_IMM, 0);
    }
  else
    {
    if (opcode == OP_ONCE || opcode >= OP_SBRA)
      {
      jump = CMP(SLJIT_NOT_EQUAL, STR_PTR, 0, SLJIT_IMM, 0);
      OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(1));
      /* Nothing stored during the first run. */
      skip = JUMP(SLJIT_JUMP);
      JUMPHERE(jump);
      /* Checking zero-length iteration. */
      if (opcode != OP_ONCE || BACKTRACK_AS(bracket_backtrack)->u.framesize < 0)
        {
        /* When we come from outside, private_data_ptr contains the previous STR_PTR. */
        braminzero = CMP(SLJIT_EQUAL, STR_PTR, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
        }
      else
        {
        /* Except when the whole stack frame must be saved. */
        OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
        braminzero = CMP(SLJIT_EQUAL, STR_PTR, 0, SLJIT_MEM1(TMP1), (BACKTRACK_AS(bracket_backtrack)->u.framesize + 1) * sizeof(sljit_sw));
        }
      JUMPHERE(skip);
      }
    else
      {
      jump = CMP(SLJIT_NOT_EQUAL, STR_PTR, 0, SLJIT_IMM, 0);
      OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(1));
      JUMPHERE(jump);
      }
    }
  }

if (repeat_type != 0)
  {
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), repeat_ptr, SLJIT_IMM, repeat_count);
  if (repeat_type == OP_EXACT)
    rmax_label = LABEL();
  }

if (ket == OP_KETRMIN)
  BACKTRACK_AS(bracket_backtrack)->recursive_matchingpath = LABEL();

if (ket == OP_KETRMAX)
  {
  rmax_label = LABEL();
  if (has_alternatives && opcode != OP_ONCE && opcode < OP_SBRA && repeat_type == 0)
    BACKTRACK_AS(bracket_backtrack)->alternative_matchingpath = rmax_label;
  }

/* Handling capturing brackets and alternatives. */
if (opcode == OP_ONCE)
  {
  stacksize = 0;
  if (needs_control_head)
    {
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr);
    stacksize++;
    }

  if (BACKTRACK_AS(bracket_backtrack)->u.framesize < 0)
    {
    /* Neither capturing brackets nor recursions are found in the block. */
    if (ket == OP_KETRMIN)
      {
      stacksize += 2;
      if (!needs_control_head)
        OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
      }
    else
      {
      if (BACKTRACK_AS(bracket_backtrack)->u.framesize == no_frame)
        OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, STACK_TOP, 0);
      if (ket == OP_KETRMAX || has_alternatives)
        stacksize++;
      }

    if (stacksize > 0)
      allocate_stack(common, stacksize);

    stacksize = 0;
    if (needs_control_head)
      {
      stacksize++;
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), TMP2, 0);
      }

    if (ket == OP_KETRMIN)
      {
      if (needs_control_head)
        OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(stacksize), STR_PTR, 0);
      if (BACKTRACK_AS(bracket_backtrack)->u.framesize == no_frame)
        OP2(SLJIT_SUB, SLJIT_MEM1(SLJIT_SP), private_data_ptr, STACK_TOP, 0, SLJIT_IMM, needs_control_head ? (2 * sizeof(sljit_sw)) : sizeof(sljit_sw));
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(stacksize + 1), TMP2, 0);
      }
    else if (ket == OP_KETRMAX || has_alternatives)
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(stacksize), STR_PTR, 0);
    }
  else
    {
    if (ket != OP_KET || has_alternatives)
      stacksize++;

    stacksize += BACKTRACK_AS(bracket_backtrack)->u.framesize + 1;
    allocate_stack(common, stacksize);

    if (needs_control_head)
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), TMP2, 0);

    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
    OP2(SLJIT_SUB, TMP2, 0, STACK_TOP, 0, SLJIT_IMM, stacksize * sizeof(sljit_sw));

    stacksize = needs_control_head ? 1 : 0;
    if (ket != OP_KET || has_alternatives)
      {
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(stacksize), STR_PTR, 0);
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, TMP2, 0);
      stacksize++;
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(stacksize), TMP1, 0);
      }
    else
      {
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, TMP2, 0);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(stacksize), TMP1, 0);
      }
    init_frame(common, ccbegin, NULL, BACKTRACK_AS(bracket_backtrack)->u.framesize + stacksize, stacksize + 1, FALSE);
    }
  }
else if (opcode == OP_CBRA || opcode == OP_SCBRA)
  {
  /* Saving the previous values. */
  if (common->optimized_cbracket[offset >> 1] != 0)
    {
    SLJIT_ASSERT(private_data_ptr == OVECTOR(offset));
    allocate_stack(common, 2);
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr + sizeof(sljit_sw));
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, STR_PTR, 0);
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), TMP1, 0);
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(1), TMP2, 0);
    }
  else
    {
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
    allocate_stack(common, 1);
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, STR_PTR, 0);
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), TMP2, 0);
    }
  }
else if (opcode == OP_SBRA || opcode == OP_SCOND)
  {
  /* Saving the previous value. */
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
  allocate_stack(common, 1);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, STR_PTR, 0);
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), TMP2, 0);
  }
else if (has_alternatives)
  {
  /* Pushing the starting string pointer. */
  allocate_stack(common, 1);
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), STR_PTR, 0);
  }

/* Generating code for the first alternative. */
if (opcode == OP_COND || opcode == OP_SCOND)
  {
  if (*matchingpath == OP_CREF)
    {
    SLJIT_ASSERT(has_alternatives);
    add_jump(compiler, &(BACKTRACK_AS(bracket_backtrack)->u.condfailed),
      CMP(SLJIT_EQUAL, SLJIT_MEM1(SLJIT_SP), OVECTOR(GET2(matchingpath, 1) << 1), SLJIT_MEM1(SLJIT_SP), OVECTOR(1)));
    matchingpath += 1 + IMM2_SIZE;
    }
  else if (*matchingpath == OP_DNCREF)
    {
    SLJIT_ASSERT(has_alternatives);

    i = GET2(matchingpath, 1 + IMM2_SIZE);
    slot = common->name_table + GET2(matchingpath, 1) * common->name_entry_size;
    OP1(SLJIT_MOV, TMP3, 0, STR_PTR, 0);
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(1));
    OP2(SLJIT_SUB | SLJIT_SET_E, TMP2, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(GET2(slot, 0) << 1), TMP1, 0);
    slot += common->name_entry_size;
    i--;
    while (i-- > 0)
      {
      OP2(SLJIT_SUB, STR_PTR, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(GET2(slot, 0) << 1), TMP1, 0);
      OP2(SLJIT_OR | SLJIT_SET_E, TMP2, 0, TMP2, 0, STR_PTR, 0);
      slot += common->name_entry_size;
      }
    OP1(SLJIT_MOV, STR_PTR, 0, TMP3, 0);
    add_jump(compiler, &(BACKTRACK_AS(bracket_backtrack)->u.condfailed), JUMP(SLJIT_ZERO));
    matchingpath += 1 + 2 * IMM2_SIZE;
    }
  else if ((*matchingpath >= OP_RREF && *matchingpath <= OP_TRUE) || *matchingpath == OP_FAIL)
    {
    /* Never has other case. */
    BACKTRACK_AS(bracket_backtrack)->u.condfailed = NULL;
    SLJIT_ASSERT(!has_alternatives);

    if (*matchingpath == OP_TRUE)
      {
      stacksize = 1;
      matchingpath++;
      }
    else if (*matchingpath == OP_FALSE || *matchingpath == OP_FAIL)
      stacksize = 0;
    else if (*matchingpath == OP_RREF)
      {
      stacksize = GET2(matchingpath, 1);
      if (common->currententry == NULL)
        stacksize = 0;
      else if (stacksize == RREF_ANY)
        stacksize = 1;
      else if (common->currententry->start == 0)
        stacksize = stacksize == 0;
      else
        stacksize = stacksize == (int)GET2(common->start, common->currententry->start + 1 + LINK_SIZE);

      if (stacksize != 0)
        matchingpath += 1 + IMM2_SIZE;
      }
    else
      {
      if (common->currententry == NULL || common->currententry->start == 0)
        stacksize = 0;
      else
        {
        stacksize = GET2(matchingpath, 1 + IMM2_SIZE);
        slot = common->name_table + GET2(matchingpath, 1) * common->name_entry_size;
        i = (int)GET2(common->start, common->currententry->start + 1 + LINK_SIZE);
        while (stacksize > 0)
          {
          if ((int)GET2(slot, 0) == i)
            break;
          slot += common->name_entry_size;
          stacksize--;
          }
        }

      if (stacksize != 0)
        matchingpath += 1 + 2 * IMM2_SIZE;
      }

      /* The stacksize == 0 is a common "else" case. */
      if (stacksize == 0)
        {
        if (*cc == OP_ALT)
          {
          matchingpath = cc + 1 + LINK_SIZE;
          cc += GET(cc, 1);
          }
        else
          matchingpath = cc;
        }
    }
  else
    {
    SLJIT_ASSERT(has_alternatives && *matchingpath >= OP_ASSERT && *matchingpath <= OP_ASSERTBACK_NOT);
    /* Similar code as PUSH_BACKTRACK macro. */
    assert = sljit_alloc_memory(compiler, sizeof(assert_backtrack));
    if (SLJIT_UNLIKELY(sljit_get_compiler_error(compiler)))
      return NULL;
    memset(assert, 0, sizeof(assert_backtrack));
    assert->common.cc = matchingpath;
    BACKTRACK_AS(bracket_backtrack)->u.assert = assert;
    matchingpath = compile_assert_matchingpath(common, matchingpath, assert, TRUE);
    }
  }

compile_matchingpath(common, matchingpath, cc, backtrack);
if (SLJIT_UNLIKELY(sljit_get_compiler_error(compiler)))
  return NULL;

if (opcode == OP_ONCE)
  match_once_common(common, ket, BACKTRACK_AS(bracket_backtrack)->u.framesize, private_data_ptr, has_alternatives, needs_control_head);

stacksize = 0;
if (repeat_type == OP_MINUPTO)
  {
  /* We need to preserve the counter. TMP2 will be used below. */
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), repeat_ptr);
  stacksize++;
  }
if (ket != OP_KET || bra != OP_BRA)
  stacksize++;
if (offset != 0)
  {
  if (common->capture_last_ptr != 0)
    stacksize++;
  if (common->optimized_cbracket[offset >> 1] == 0)
    stacksize += 2;
  }
if (has_alternatives && opcode != OP_ONCE)
  stacksize++;

if (stacksize > 0)
  allocate_stack(common, stacksize);

stacksize = 0;
if (repeat_type == OP_MINUPTO)
  {
  /* TMP2 was set above. */
  OP2(SLJIT_SUB, SLJIT_MEM1(STACK_TOP), STACK(stacksize), TMP2, 0, SLJIT_IMM, 1);
  stacksize++;
  }

if (ket != OP_KET || bra != OP_BRA)
  {
  if (ket != OP_KET)
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(stacksize), STR_PTR, 0);
  else
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(stacksize), SLJIT_IMM, 0);
  stacksize++;
  }

if (offset != 0)
  stacksize = match_capture_common(common, stacksize, offset, private_data_ptr);

if (has_alternatives)
  {
  if (opcode != OP_ONCE)
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(stacksize), SLJIT_IMM, 0);
  if (ket != OP_KETRMAX)
    BACKTRACK_AS(bracket_backtrack)->alternative_matchingpath = LABEL();
  }

/* Must be after the matchingpath label. */
if (offset != 0 && common->optimized_cbracket[offset >> 1] != 0)
  {
  SLJIT_ASSERT(private_data_ptr == OVECTOR(offset + 0));
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1), STR_PTR, 0);
  }

if (ket == OP_KETRMAX)
  {
  if (repeat_type != 0)
    {
    if (has_alternatives)
      BACKTRACK_AS(bracket_backtrack)->alternative_matchingpath = LABEL();
    OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_MEM1(SLJIT_SP), repeat_ptr, SLJIT_MEM1(SLJIT_SP), repeat_ptr, SLJIT_IMM, 1);
    JUMPTO(SLJIT_NOT_ZERO, rmax_label);
    /* Drop STR_PTR for greedy plus quantifier. */
    if (opcode != OP_ONCE)
      free_stack(common, 1);
    }
  else if (opcode == OP_ONCE || opcode >= OP_SBRA)
    {
    if (has_alternatives)
      BACKTRACK_AS(bracket_backtrack)->alternative_matchingpath = LABEL();
    /* Checking zero-length iteration. */
    if (opcode != OP_ONCE)
      {
      CMPTO(SLJIT_NOT_EQUAL, SLJIT_MEM1(SLJIT_SP), private_data_ptr, STR_PTR, 0, rmax_label);
      /* Drop STR_PTR for greedy plus quantifier. */
      if (bra != OP_BRAZERO)
        free_stack(common, 1);
      }
    else
      /* TMP2 must contain the starting STR_PTR. */
      CMPTO(SLJIT_NOT_EQUAL, TMP2, 0, STR_PTR, 0, rmax_label);
    }
  else
    JUMPTO(SLJIT_JUMP, rmax_label);
  BACKTRACK_AS(bracket_backtrack)->recursive_matchingpath = LABEL();
  }

if (repeat_type == OP_EXACT)
  {
  count_match(common);
  OP2(SLJIT_SUB | SLJIT_SET_E, SLJIT_MEM1(SLJIT_SP), repeat_ptr, SLJIT_MEM1(SLJIT_SP), repeat_ptr, SLJIT_IMM, 1);
  JUMPTO(SLJIT_NOT_ZERO, rmax_label);
  }
else if (repeat_type == OP_UPTO)
  {
  /* We need to preserve the counter. */
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), repeat_ptr);
  allocate_stack(common, 1);
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), TMP2, 0);
  }

if (bra == OP_BRAZERO)
  BACKTRACK_AS(bracket_backtrack)->zero_matchingpath = LABEL();

if (bra == OP_BRAMINZERO)
  {
  /* This is a backtrack path! (From the viewpoint of OP_BRAMINZERO) */
  JUMPTO(SLJIT_JUMP, ((braminzero_backtrack *)parent)->matchingpath);
  if (braminzero != NULL)
    {
    JUMPHERE(braminzero);
    /* We need to release the end pointer to perform the
    backtrack for the zero-length iteration. When
    framesize is < 0, OP_ONCE will do the release itself. */
    if (opcode == OP_ONCE && BACKTRACK_AS(bracket_backtrack)->u.framesize >= 0)
      {
      OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
      add_jump(compiler, &common->revertframes, JUMP(SLJIT_FAST_CALL));
      }
    else if (ket == OP_KETRMIN && opcode != OP_ONCE)
      free_stack(common, 1);
    }
  /* Continue to the normal backtrack. */
  }

if ((ket != OP_KET && bra != OP_BRAMINZERO) || bra == OP_BRAZERO)
  count_match(common);

/* Skip the other alternatives. */
while (*cc == OP_ALT)
  cc += GET(cc, 1);
cc += 1 + LINK_SIZE;

if (opcode == OP_ONCE)
  {
  /* We temporarily encode the needs_control_head in the lowest bit.
     Note: on the target architectures of SLJIT the ((x << 1) >> 1) returns
     the same value for small signed numbers (including negative numbers). */
  BACKTRACK_AS(bracket_backtrack)->u.framesize = (BACKTRACK_AS(bracket_backtrack)->u.framesize << 1) | (needs_control_head ? 1 : 0);
  }
return cc + repeat_length;
}

static PCRE2_SPTR compile_bracketpos_matchingpath(compiler_common *common, PCRE2_SPTR cc, backtrack_common *parent)
{
DEFINE_COMPILER;
backtrack_common *backtrack;
PCRE2_UCHAR opcode;
int private_data_ptr;
int cbraprivptr = 0;
BOOL needs_control_head;
int framesize;
int stacksize;
int offset = 0;
BOOL zero = FALSE;
PCRE2_SPTR ccbegin = NULL;
int stack; /* Also contains the offset of control head. */
struct sljit_label *loop = NULL;
struct jump_list *emptymatch = NULL;

PUSH_BACKTRACK(sizeof(bracketpos_backtrack), cc, NULL);
if (*cc == OP_BRAPOSZERO)
  {
  zero = TRUE;
  cc++;
  }

opcode = *cc;
private_data_ptr = PRIVATE_DATA(cc);
SLJIT_ASSERT(private_data_ptr != 0);
BACKTRACK_AS(bracketpos_backtrack)->private_data_ptr = private_data_ptr;
switch(opcode)
  {
  case OP_BRAPOS:
  case OP_SBRAPOS:
  ccbegin = cc + 1 + LINK_SIZE;
  break;

  case OP_CBRAPOS:
  case OP_SCBRAPOS:
  offset = GET2(cc, 1 + LINK_SIZE);
  /* This case cannot be optimized in the same was as
  normal capturing brackets. */
  SLJIT_ASSERT(common->optimized_cbracket[offset] == 0);
  cbraprivptr = OVECTOR_PRIV(offset);
  offset <<= 1;
  ccbegin = cc + 1 + LINK_SIZE + IMM2_SIZE;
  break;

  default:
  SLJIT_ASSERT_STOP();
  break;
  }

framesize = get_framesize(common, cc, NULL, FALSE, &needs_control_head);
BACKTRACK_AS(bracketpos_backtrack)->framesize = framesize;
if (framesize < 0)
  {
  if (offset != 0)
    {
    stacksize = 2;
    if (common->capture_last_ptr != 0)
      stacksize++;
    }
  else
    stacksize = 1;

  if (needs_control_head)
    stacksize++;
  if (!zero)
    stacksize++;

  BACKTRACK_AS(bracketpos_backtrack)->stacksize = stacksize;
  allocate_stack(common, stacksize);
  if (framesize == no_frame)
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, STACK_TOP, 0);

  stack = 0;
  if (offset != 0)
    {
    stack = 2;
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset));
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1));
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), TMP1, 0);
    if (common->capture_last_ptr != 0)
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->capture_last_ptr);
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(1), TMP2, 0);
    if (needs_control_head)
      OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr);
    if (common->capture_last_ptr != 0)
      {
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(2), TMP1, 0);
      stack = 3;
      }
    }
  else
    {
    if (needs_control_head)
      OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr);
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), STR_PTR, 0);
    stack = 1;
    }

  if (needs_control_head)
    stack++;
  if (!zero)
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(stack), SLJIT_IMM, 1);
  if (needs_control_head)
    {
    stack--;
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(stack), TMP2, 0);
    }
  }
else
  {
  stacksize = framesize + 1;
  if (!zero)
    stacksize++;
  if (needs_control_head)
    stacksize++;
  if (offset == 0)
    stacksize++;
  BACKTRACK_AS(bracketpos_backtrack)->stacksize = stacksize;

  allocate_stack(common, stacksize);
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
  if (needs_control_head)
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr);
  OP2(SLJIT_SUB, SLJIT_MEM1(SLJIT_SP), private_data_ptr, STACK_TOP, 0, SLJIT_IMM, -STACK(stacksize - 1));

  stack = 0;
  if (!zero)
    {
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), SLJIT_IMM, 1);
    stack = 1;
    }
  if (needs_control_head)
    {
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(stack), TMP2, 0);
    stack++;
    }
  if (offset == 0)
    {
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(stack), STR_PTR, 0);
    stack++;
    }
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(stack), TMP1, 0);
  init_frame(common, cc, NULL, stacksize - 1, stacksize - framesize, FALSE);
  stack -= 1 + (offset == 0);
  }

if (offset != 0)
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), cbraprivptr, STR_PTR, 0);

loop = LABEL();
while (*cc != OP_KETRPOS)
  {
  backtrack->top = NULL;
  backtrack->topbacktracks = NULL;
  cc += GET(cc, 1);

  compile_matchingpath(common, ccbegin, cc, backtrack);
  if (SLJIT_UNLIKELY(sljit_get_compiler_error(compiler)))
    return NULL;

  if (framesize < 0)
    {
    if (framesize == no_frame)
      OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);

    if (offset != 0)
      {
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), cbraprivptr);
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1), STR_PTR, 0);
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), cbraprivptr, STR_PTR, 0);
      if (common->capture_last_ptr != 0)
        OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->capture_last_ptr, SLJIT_IMM, offset >> 1);
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset), TMP1, 0);
      }
    else
      {
      if (opcode == OP_SBRAPOS)
        OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), STR_PTR, 0);
      }

    /* Even if the match is empty, we need to reset the control head. */
    if (needs_control_head)
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, SLJIT_MEM1(STACK_TOP), STACK(stack));

    if (opcode == OP_SBRAPOS || opcode == OP_SCBRAPOS)
      add_jump(compiler, &emptymatch, CMP(SLJIT_EQUAL, TMP1, 0, STR_PTR, 0));

    if (!zero)
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(stacksize - 1), SLJIT_IMM, 0);
    }
  else
    {
    if (offset != 0)
      {
      OP2(SLJIT_ADD, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr, SLJIT_IMM, stacksize * sizeof(sljit_sw));
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), cbraprivptr);
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1), STR_PTR, 0);
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), cbraprivptr, STR_PTR, 0);
      if (common->capture_last_ptr != 0)
        OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->capture_last_ptr, SLJIT_IMM, offset >> 1);
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset), TMP1, 0);
      }
    else
      {
      OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
      OP2(SLJIT_ADD, STACK_TOP, 0, TMP2, 0, SLJIT_IMM, stacksize * sizeof(sljit_sw));
      if (opcode == OP_SBRAPOS)
        OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(TMP2), (framesize + 1) * sizeof(sljit_sw));
      OP1(SLJIT_MOV, SLJIT_MEM1(TMP2), (framesize + 1) * sizeof(sljit_sw), STR_PTR, 0);
      }

    /* Even if the match is empty, we need to reset the control head. */
    if (needs_control_head)
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, SLJIT_MEM1(STACK_TOP), STACK(stack));

    if (opcode == OP_SBRAPOS || opcode == OP_SCBRAPOS)
      add_jump(compiler, &emptymatch, CMP(SLJIT_EQUAL, TMP1, 0, STR_PTR, 0));

    if (!zero)
      {
      if (framesize < 0)
        OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(stacksize - 1), SLJIT_IMM, 0);
      else
        OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), SLJIT_IMM, 0);
      }
    }

  JUMPTO(SLJIT_JUMP, loop);
  flush_stubs(common);

  compile_backtrackingpath(common, backtrack->top);
  if (SLJIT_UNLIKELY(sljit_get_compiler_error(compiler)))
    return NULL;
  set_jumps(backtrack->topbacktracks, LABEL());

  if (framesize < 0)
    {
    if (offset != 0)
      OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(SLJIT_SP), cbraprivptr);
    else
      OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
    }
  else
    {
    if (offset != 0)
      {
      /* Last alternative. */
      if (*cc == OP_KETRPOS)
        OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
      OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(SLJIT_SP), cbraprivptr);
      }
    else
      {
      OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
      OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(TMP2), (framesize + 1) * sizeof(sljit_sw));
      }
    }

  if (*cc == OP_KETRPOS)
    break;
  ccbegin = cc + 1 + LINK_SIZE;
  }

/* We don't have to restore the control head in case of a failed match. */

backtrack->topbacktracks = NULL;
if (!zero)
  {
  if (framesize < 0)
    add_jump(compiler, &backtrack->topbacktracks, CMP(SLJIT_NOT_EQUAL, SLJIT_MEM1(STACK_TOP), STACK(stacksize - 1), SLJIT_IMM, 0));
  else /* TMP2 is set to [private_data_ptr] above. */
    add_jump(compiler, &backtrack->topbacktracks, CMP(SLJIT_NOT_EQUAL, SLJIT_MEM1(TMP2), (stacksize - 1) * sizeof(sljit_sw), SLJIT_IMM, 0));
  }

/* None of them matched. */
set_jumps(emptymatch, LABEL());
count_match(common);
return cc + 1 + LINK_SIZE;
}

static SLJIT_INLINE PCRE2_SPTR get_iterator_parameters(compiler_common *common, PCRE2_SPTR cc, PCRE2_UCHAR *opcode, PCRE2_UCHAR *type, sljit_u32 *max, sljit_u32 *exact, PCRE2_SPTR *end)
{
int class_len;

*opcode = *cc;
*exact = 0;

if (*opcode >= OP_STAR && *opcode <= OP_POSUPTO)
  {
  cc++;
  *type = OP_CHAR;
  }
else if (*opcode >= OP_STARI && *opcode <= OP_POSUPTOI)
  {
  cc++;
  *type = OP_CHARI;
  *opcode -= OP_STARI - OP_STAR;
  }
else if (*opcode >= OP_NOTSTAR && *opcode <= OP_NOTPOSUPTO)
  {
  cc++;
  *type = OP_NOT;
  *opcode -= OP_NOTSTAR - OP_STAR;
  }
else if (*opcode >= OP_NOTSTARI && *opcode <= OP_NOTPOSUPTOI)
  {
  cc++;
  *type = OP_NOTI;
  *opcode -= OP_NOTSTARI - OP_STAR;
  }
else if (*opcode >= OP_TYPESTAR && *opcode <= OP_TYPEPOSUPTO)
  {
  cc++;
  *opcode -= OP_TYPESTAR - OP_STAR;
  *type = OP_END;
  }
else
  {
  SLJIT_ASSERT(*opcode == OP_CLASS || *opcode == OP_NCLASS || *opcode == OP_XCLASS);
  *type = *opcode;
  cc++;
  class_len = (*type < OP_XCLASS) ? (int)(1 + (32 / sizeof(PCRE2_UCHAR))) : GET(cc, 0);
  *opcode = cc[class_len - 1];

  if (*opcode >= OP_CRSTAR && *opcode <= OP_CRMINQUERY)
    {
    *opcode -= OP_CRSTAR - OP_STAR;
    *end = cc + class_len;

    if (*opcode == OP_PLUS || *opcode == OP_MINPLUS)
      {
      *exact = 1;
      *opcode -= OP_PLUS - OP_STAR;
      }
    }
  else if (*opcode >= OP_CRPOSSTAR && *opcode <= OP_CRPOSQUERY)
    {
    *opcode -= OP_CRPOSSTAR - OP_POSSTAR;
    *end = cc + class_len;

    if (*opcode == OP_POSPLUS)
      {
      *exact = 1;
      *opcode = OP_POSSTAR;
      }
    }
  else
    {
    SLJIT_ASSERT(*opcode == OP_CRRANGE || *opcode == OP_CRMINRANGE || *opcode == OP_CRPOSRANGE);
    *max = GET2(cc, (class_len + IMM2_SIZE));
    *exact = GET2(cc, class_len);

    if (*max == 0)
      {
      if (*opcode == OP_CRPOSRANGE)
        *opcode = OP_POSSTAR;
      else
        *opcode -= OP_CRRANGE - OP_STAR;
      }
    else
      {
      *max -= *exact;
      if (*max == 0)
        *opcode = OP_EXACT;
      else if (*max == 1)
        {
        if (*opcode == OP_CRPOSRANGE)
          *opcode = OP_POSQUERY;
        else
          *opcode -= OP_CRRANGE - OP_QUERY;
        }
      else
        {
        if (*opcode == OP_CRPOSRANGE)
          *opcode = OP_POSUPTO;
        else
          *opcode -= OP_CRRANGE - OP_UPTO;
        }
      }
    *end = cc + class_len + 2 * IMM2_SIZE;
    }
  return cc;
  }

switch(*opcode)
  {
  case OP_EXACT:
  *exact = GET2(cc, 0);
  cc += IMM2_SIZE;
  break;

  case OP_PLUS:
  case OP_MINPLUS:
  *exact = 1;
  *opcode -= OP_PLUS - OP_STAR;
  break;

  case OP_POSPLUS:
  *exact = 1;
  *opcode = OP_POSSTAR;
  break;

  case OP_UPTO:
  case OP_MINUPTO:
  case OP_POSUPTO:
  *max = GET2(cc, 0);
  cc += IMM2_SIZE;
  break;
  }

if (*type == OP_END)
  {
  *type = *cc;
  *end = next_opcode(common, cc);
  cc++;
  return cc;
  }

*end = cc + 1;
#ifdef SUPPORT_UNICODE
if (common->utf && HAS_EXTRALEN(*cc)) *end += GET_EXTRALEN(*cc);
#endif
return cc;
}

static PCRE2_SPTR compile_iterator_matchingpath(compiler_common *common, PCRE2_SPTR cc, backtrack_common *parent)
{
DEFINE_COMPILER;
backtrack_common *backtrack;
PCRE2_UCHAR opcode;
PCRE2_UCHAR type;
sljit_u32 max = 0, exact;
BOOL fast_fail;
sljit_s32 fast_str_ptr;
BOOL charpos_enabled;
PCRE2_UCHAR charpos_char;
unsigned int charpos_othercasebit;
PCRE2_SPTR end;
jump_list *no_match = NULL;
jump_list *no_char1_match = NULL;
struct sljit_jump *jump = NULL;
struct sljit_label *label;
int private_data_ptr = PRIVATE_DATA(cc);
int base = (private_data_ptr == 0) ? SLJIT_MEM1(STACK_TOP) : SLJIT_MEM1(SLJIT_SP);
int offset0 = (private_data_ptr == 0) ? STACK(0) : private_data_ptr;
int offset1 = (private_data_ptr == 0) ? STACK(1) : private_data_ptr + (int)sizeof(sljit_sw);
int tmp_base, tmp_offset;

PUSH_BACKTRACK(sizeof(char_iterator_backtrack), cc, NULL);

fast_str_ptr = PRIVATE_DATA(cc + 1);
fast_fail = TRUE;

SLJIT_ASSERT(common->fast_forward_bc_ptr == NULL || fast_str_ptr == 0 || cc == common->fast_forward_bc_ptr);

if (cc == common->fast_forward_bc_ptr)
  fast_fail = FALSE;
else if (common->fast_fail_start_ptr == 0)
  fast_str_ptr = 0;

SLJIT_ASSERT(common->fast_forward_bc_ptr != NULL || fast_str_ptr == 0
  || (fast_str_ptr >= common->fast_fail_start_ptr && fast_str_ptr <= common->fast_fail_end_ptr));

cc = get_iterator_parameters(common, cc, &opcode, &type, &max, &exact, &end);

if (type != OP_EXTUNI)
  {
  tmp_base = TMP3;
  tmp_offset = 0;
  }
else
  {
  tmp_base = SLJIT_MEM1(SLJIT_SP);
  tmp_offset = POSSESSIVE0;
  }

if (fast_fail && fast_str_ptr != 0)
  add_jump(compiler, &backtrack->topbacktracks, CMP(SLJIT_LESS_EQUAL, STR_PTR, 0, SLJIT_MEM1(SLJIT_SP), fast_str_ptr));

/* Handle fixed part first. */
if (exact > 1)
  {
  SLJIT_ASSERT(fast_str_ptr == 0);
  if (common->mode == PCRE2_JIT_COMPLETE
#ifdef SUPPORT_UNICODE
      && !common->utf
#endif
      )
    {
    OP2(SLJIT_ADD, TMP1, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(exact));
    add_jump(compiler, &backtrack->topbacktracks, CMP(SLJIT_GREATER, TMP1, 0, STR_END, 0));
    OP1(SLJIT_MOV, tmp_base, tmp_offset, SLJIT_IMM, exact);
    label = LABEL();
    compile_char1_matchingpath(common, type, cc, &backtrack->topbacktracks, FALSE);
    OP2(SLJIT_SUB | SLJIT_SET_E, tmp_base, tmp_offset, tmp_base, tmp_offset, SLJIT_IMM, 1);
    JUMPTO(SLJIT_NOT_ZERO, label);
    }
  else
    {
    OP1(SLJIT_MOV, tmp_base, tmp_offset, SLJIT_IMM, exact);
    label = LABEL();
    compile_char1_matchingpath(common, type, cc, &backtrack->topbacktracks, TRUE);
    OP2(SLJIT_SUB | SLJIT_SET_E, tmp_base, tmp_offset, tmp_base, tmp_offset, SLJIT_IMM, 1);
    JUMPTO(SLJIT_NOT_ZERO, label);
    }
  }
else if (exact == 1)
  compile_char1_matchingpath(common, type, cc, &backtrack->topbacktracks, TRUE);

switch(opcode)
  {
  case OP_STAR:
  case OP_UPTO:
  SLJIT_ASSERT(fast_str_ptr == 0 || opcode == OP_STAR);

  if (type == OP_ANYNL || type == OP_EXTUNI)
    {
    SLJIT_ASSERT(private_data_ptr == 0);
    SLJIT_ASSERT(fast_str_ptr == 0);

    allocate_stack(common, 2);
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), STR_PTR, 0);
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(1), SLJIT_IMM, 0);

    if (opcode == OP_UPTO)
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), POSSESSIVE0, SLJIT_IMM, max);

    label = LABEL();
    compile_char1_matchingpath(common, type, cc, &BACKTRACK_AS(char_iterator_backtrack)->u.backtracks, TRUE);
    if (opcode == OP_UPTO)
      {
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), POSSESSIVE0);
      OP2(SLJIT_SUB | SLJIT_SET_E, TMP1, 0, TMP1, 0, SLJIT_IMM, 1);
      jump = JUMP(SLJIT_ZERO);
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), POSSESSIVE0, TMP1, 0);
      }

    /* We cannot use TMP3 because of this allocate_stack. */
    allocate_stack(common, 1);
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), STR_PTR, 0);
    JUMPTO(SLJIT_JUMP, label);
    if (jump != NULL)
      JUMPHERE(jump);
    }
  else
    {
    charpos_enabled = FALSE;
    charpos_char = 0;
    charpos_othercasebit = 0;

    if ((type != OP_CHAR && type != OP_CHARI) && (*end == OP_CHAR || *end == OP_CHARI))
      {
      charpos_enabled = TRUE;
#ifdef SUPPORT_UNICODE
      charpos_enabled = !common->utf || !HAS_EXTRALEN(end[1]);
#endif
      if (charpos_enabled && *end == OP_CHARI && char_has_othercase(common, end + 1))
        {
        charpos_othercasebit = char_get_othercase_bit(common, end + 1);
        if (charpos_othercasebit == 0)
          charpos_enabled = FALSE;
        }

      if (charpos_enabled)
        {
        charpos_char = end[1];
        /* Consumpe the OP_CHAR opcode. */
        end += 2;
#if PCRE2_CODE_UNIT_WIDTH == 8
        SLJIT_ASSERT((charpos_othercasebit >> 8) == 0);
#elif PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32
        SLJIT_ASSERT((charpos_othercasebit >> 9) == 0);
        if ((charpos_othercasebit & 0x100) != 0)
          charpos_othercasebit = (charpos_othercasebit & 0xff) << 8;
#endif
        if (charpos_othercasebit != 0)
          charpos_char |= charpos_othercasebit;

        BACKTRACK_AS(char_iterator_backtrack)->u.charpos.enabled = TRUE;
        BACKTRACK_AS(char_iterator_backtrack)->u.charpos.chr = charpos_char;
        BACKTRACK_AS(char_iterator_backtrack)->u.charpos.othercasebit = charpos_othercasebit;
        }
      }

    if (charpos_enabled)
      {
      if (opcode == OP_UPTO)
        OP1(SLJIT_MOV, tmp_base, tmp_offset, SLJIT_IMM, max + 1);

      /* Search the first instance of charpos_char. */
      jump = JUMP(SLJIT_JUMP);
      label = LABEL();
      if (opcode == OP_UPTO)
        {
        OP2(SLJIT_SUB | SLJIT_SET_E, tmp_base, tmp_offset, tmp_base, tmp_offset, SLJIT_IMM, 1);
        add_jump(compiler, &backtrack->topbacktracks, JUMP(SLJIT_ZERO));
        }
      compile_char1_matchingpath(common, type, cc, &backtrack->topbacktracks, FALSE);
      if (fast_str_ptr != 0)
        OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), fast_str_ptr, STR_PTR, 0);
      JUMPHERE(jump);

      detect_partial_match(common, &backtrack->topbacktracks);
      OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
      if (charpos_othercasebit != 0)
        OP2(SLJIT_OR, TMP1, 0, TMP1, 0, SLJIT_IMM, charpos_othercasebit);
      CMPTO(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, charpos_char, label);

      if (private_data_ptr == 0)
        allocate_stack(common, 2);
      OP1(SLJIT_MOV, base, offset0, STR_PTR, 0);
      OP1(SLJIT_MOV, base, offset1, STR_PTR, 0);
      if (opcode == OP_UPTO)
        {
        OP2(SLJIT_SUB | SLJIT_SET_E, tmp_base, tmp_offset, tmp_base, tmp_offset, SLJIT_IMM, 1);
        add_jump(compiler, &no_match, JUMP(SLJIT_ZERO));
        }

      /* Search the last instance of charpos_char. */
      label = LABEL();
      compile_char1_matchingpath(common, type, cc, &no_match, FALSE);
      if (fast_str_ptr != 0)
        OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), fast_str_ptr, STR_PTR, 0);
      detect_partial_match(common, &no_match);
      OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
      if (charpos_othercasebit != 0)
        OP2(SLJIT_OR, TMP1, 0, TMP1, 0, SLJIT_IMM, charpos_othercasebit);
      if (opcode == OP_STAR)
        {
        CMPTO(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, charpos_char, label);
        OP1(SLJIT_MOV, base, offset0, STR_PTR, 0);
        }
      else
        {
        jump = CMP(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, charpos_char);
        OP1(SLJIT_MOV, base, offset0, STR_PTR, 0);
        JUMPHERE(jump);
        }

      if (opcode == OP_UPTO)
        {
        OP2(SLJIT_SUB | SLJIT_SET_E, tmp_base, tmp_offset, tmp_base, tmp_offset, SLJIT_IMM, 1);
        JUMPTO(SLJIT_NOT_ZERO, label);
        }
      else
        JUMPTO(SLJIT_JUMP, label);

      set_jumps(no_match, LABEL());
      OP1(SLJIT_MOV, STR_PTR, 0, base, offset0);
      OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
      OP1(SLJIT_MOV, base, offset0, STR_PTR, 0);
      }
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
    else if (common->utf)
      {
      if (private_data_ptr == 0)
        allocate_stack(common, 2);

      OP1(SLJIT_MOV, base, offset0, STR_PTR, 0);
      OP1(SLJIT_MOV, base, offset1, STR_PTR, 0);

      if (opcode == OP_UPTO)
        OP1(SLJIT_MOV, tmp_base, tmp_offset, SLJIT_IMM, max);

      label = LABEL();
      compile_char1_matchingpath(common, type, cc, &no_match, TRUE);
      OP1(SLJIT_MOV, base, offset0, STR_PTR, 0);

      if (opcode == OP_UPTO)
        {
        OP2(SLJIT_SUB | SLJIT_SET_E, tmp_base, tmp_offset, tmp_base, tmp_offset, SLJIT_IMM, 1);
        JUMPTO(SLJIT_NOT_ZERO, label);
        }
      else
        JUMPTO(SLJIT_JUMP, label);

      set_jumps(no_match, LABEL());
      OP1(SLJIT_MOV, STR_PTR, 0, base, offset0);
      if (fast_str_ptr != 0)
        OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), fast_str_ptr, STR_PTR, 0);
      }
#endif
    else
      {
      if (private_data_ptr == 0)
        allocate_stack(common, 2);

      OP1(SLJIT_MOV, base, offset1, STR_PTR, 0);
      if (opcode == OP_UPTO)
        OP1(SLJIT_MOV, tmp_base, tmp_offset, SLJIT_IMM, max);

      label = LABEL();
      detect_partial_match(common, &no_match);
      compile_char1_matchingpath(common, type, cc, &no_char1_match, FALSE);
      if (opcode == OP_UPTO)
        {
        OP2(SLJIT_SUB | SLJIT_SET_E, tmp_base, tmp_offset, tmp_base, tmp_offset, SLJIT_IMM, 1);
        JUMPTO(SLJIT_NOT_ZERO, label);
        OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
        }
      else
        JUMPTO(SLJIT_JUMP, label);

      set_jumps(no_char1_match, LABEL());
      OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
      set_jumps(no_match, LABEL());
      OP1(SLJIT_MOV, base, offset0, STR_PTR, 0);
      if (fast_str_ptr != 0)
        OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), fast_str_ptr, STR_PTR, 0);
      }
    }
  BACKTRACK_AS(char_iterator_backtrack)->matchingpath = LABEL();
  break;

  case OP_MINSTAR:
  if (private_data_ptr == 0)
    allocate_stack(common, 1);
  OP1(SLJIT_MOV, base, offset0, STR_PTR, 0);
  BACKTRACK_AS(char_iterator_backtrack)->matchingpath = LABEL();
  if (fast_str_ptr != 0)
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), fast_str_ptr, STR_PTR, 0);
  break;

  case OP_MINUPTO:
  SLJIT_ASSERT(fast_str_ptr == 0);
  if (private_data_ptr == 0)
    allocate_stack(common, 2);
  OP1(SLJIT_MOV, base, offset0, STR_PTR, 0);
  OP1(SLJIT_MOV, base, offset1, SLJIT_IMM, max + 1);
  BACKTRACK_AS(char_iterator_backtrack)->matchingpath = LABEL();
  break;

  case OP_QUERY:
  case OP_MINQUERY:
  SLJIT_ASSERT(fast_str_ptr == 0);
  if (private_data_ptr == 0)
    allocate_stack(common, 1);
  OP1(SLJIT_MOV, base, offset0, STR_PTR, 0);
  if (opcode == OP_QUERY)
    compile_char1_matchingpath(common, type, cc, &BACKTRACK_AS(char_iterator_backtrack)->u.backtracks, TRUE);
  BACKTRACK_AS(char_iterator_backtrack)->matchingpath = LABEL();
  break;

  case OP_EXACT:
  break;

  case OP_POSSTAR:
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
  if (common->utf)
    {
    OP1(SLJIT_MOV, tmp_base, tmp_offset, STR_PTR, 0);
    label = LABEL();
    compile_char1_matchingpath(common, type, cc, &no_match, TRUE);
    OP1(SLJIT_MOV, tmp_base, tmp_offset, STR_PTR, 0);
    JUMPTO(SLJIT_JUMP, label);
    set_jumps(no_match, LABEL());
    OP1(SLJIT_MOV, STR_PTR, 0, tmp_base, tmp_offset);
    if (fast_str_ptr != 0)
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), fast_str_ptr, STR_PTR, 0);
    break;
    }
#endif
  label = LABEL();
  detect_partial_match(common, &no_match);
  compile_char1_matchingpath(common, type, cc, &no_char1_match, FALSE);
  JUMPTO(SLJIT_JUMP, label);
  set_jumps(no_char1_match, LABEL());
  OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  set_jumps(no_match, LABEL());
  if (fast_str_ptr != 0)
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), fast_str_ptr, STR_PTR, 0);
  break;

  case OP_POSUPTO:
  SLJIT_ASSERT(fast_str_ptr == 0);
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
  if (common->utf)
    {
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), POSSESSIVE1, STR_PTR, 0);
    OP1(SLJIT_MOV, tmp_base, tmp_offset, SLJIT_IMM, max);
    label = LABEL();
    compile_char1_matchingpath(common, type, cc, &no_match, TRUE);
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), POSSESSIVE1, STR_PTR, 0);
    OP2(SLJIT_SUB | SLJIT_SET_E, tmp_base, tmp_offset, tmp_base, tmp_offset, SLJIT_IMM, 1);
    JUMPTO(SLJIT_NOT_ZERO, label);
    set_jumps(no_match, LABEL());
    OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(SLJIT_SP), POSSESSIVE1);
    break;
    }
#endif
  OP1(SLJIT_MOV, tmp_base, tmp_offset, SLJIT_IMM, max);
  label = LABEL();
  detect_partial_match(common, &no_match);
  compile_char1_matchingpath(common, type, cc, &no_char1_match, FALSE);
  OP2(SLJIT_SUB | SLJIT_SET_E, tmp_base, tmp_offset, tmp_base, tmp_offset, SLJIT_IMM, 1);
  JUMPTO(SLJIT_NOT_ZERO, label);
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  set_jumps(no_char1_match, LABEL());
  OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  set_jumps(no_match, LABEL());
  break;

  case OP_POSQUERY:
  SLJIT_ASSERT(fast_str_ptr == 0);
  OP1(SLJIT_MOV, tmp_base, tmp_offset, STR_PTR, 0);
  compile_char1_matchingpath(common, type, cc, &no_match, TRUE);
  OP1(SLJIT_MOV, tmp_base, tmp_offset, STR_PTR, 0);
  set_jumps(no_match, LABEL());
  OP1(SLJIT_MOV, STR_PTR, 0, tmp_base, tmp_offset);
  break;

  default:
  SLJIT_ASSERT_STOP();
  break;
  }

count_match(common);
return end;
}

static SLJIT_INLINE PCRE2_SPTR compile_fail_accept_matchingpath(compiler_common *common, PCRE2_SPTR cc, backtrack_common *parent)
{
DEFINE_COMPILER;
backtrack_common *backtrack;

PUSH_BACKTRACK(sizeof(backtrack_common), cc, NULL);

if (*cc == OP_FAIL)
  {
  add_jump(compiler, &backtrack->topbacktracks, JUMP(SLJIT_JUMP));
  return cc + 1;
  }

if (*cc == OP_ASSERT_ACCEPT || common->currententry != NULL || !common->might_be_empty)
  {
  /* No need to check notempty conditions. */
  if (common->accept_label == NULL)
    add_jump(compiler, &common->accept, JUMP(SLJIT_JUMP));
  else
    JUMPTO(SLJIT_JUMP, common->accept_label);
  return cc + 1;
  }

if (common->accept_label == NULL)
  add_jump(compiler, &common->accept, CMP(SLJIT_NOT_EQUAL, STR_PTR, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(0)));
else
  CMPTO(SLJIT_NOT_EQUAL, STR_PTR, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(0), common->accept_label);
OP1(SLJIT_MOV, TMP1, 0, ARGUMENTS, 0);
OP1(SLJIT_MOV_U32, TMP2, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, options));
OP2(SLJIT_AND | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP2, 0, SLJIT_IMM, PCRE2_NOTEMPTY);
add_jump(compiler, &backtrack->topbacktracks, JUMP(SLJIT_NOT_ZERO));
OP2(SLJIT_AND | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP2, 0, SLJIT_IMM, PCRE2_NOTEMPTY_ATSTART);
if (common->accept_label == NULL)
  add_jump(compiler, &common->accept, JUMP(SLJIT_ZERO));
else
  JUMPTO(SLJIT_ZERO, common->accept_label);
OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, str));
if (common->accept_label == NULL)
  add_jump(compiler, &common->accept, CMP(SLJIT_NOT_EQUAL, TMP2, 0, STR_PTR, 0));
else
  CMPTO(SLJIT_NOT_EQUAL, TMP2, 0, STR_PTR, 0, common->accept_label);
add_jump(compiler, &backtrack->topbacktracks, JUMP(SLJIT_JUMP));
return cc + 1;
}

static SLJIT_INLINE PCRE2_SPTR compile_close_matchingpath(compiler_common *common, PCRE2_SPTR cc)
{
DEFINE_COMPILER;
int offset = GET2(cc, 1);
BOOL optimized_cbracket = common->optimized_cbracket[offset] != 0;

/* Data will be discarded anyway... */
if (common->currententry != NULL)
  return cc + 1 + IMM2_SIZE;

if (!optimized_cbracket)
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR_PRIV(offset));
offset <<= 1;
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1), STR_PTR, 0);
if (!optimized_cbracket)
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset), TMP1, 0);
return cc + 1 + IMM2_SIZE;
}

static SLJIT_INLINE PCRE2_SPTR compile_control_verb_matchingpath(compiler_common *common, PCRE2_SPTR cc, backtrack_common *parent)
{
DEFINE_COMPILER;
backtrack_common *backtrack;
PCRE2_UCHAR opcode = *cc;
PCRE2_SPTR ccend = cc + 1;

if (opcode == OP_PRUNE_ARG || opcode == OP_SKIP_ARG || opcode == OP_THEN_ARG)
  ccend += 2 + cc[1];

PUSH_BACKTRACK(sizeof(backtrack_common), cc, NULL);

if (opcode == OP_SKIP)
  {
  allocate_stack(common, 1);
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), STR_PTR, 0);
  return ccend;
  }

if (opcode == OP_PRUNE_ARG || opcode == OP_THEN_ARG)
  {
  OP1(SLJIT_MOV, TMP1, 0, ARGUMENTS, 0);
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, (sljit_sw)(cc + 2));
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->mark_ptr, TMP2, 0);
  OP1(SLJIT_MOV, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, mark_ptr), TMP2, 0);
  }

return ccend;
}

static PCRE2_UCHAR then_trap_opcode[1] = { OP_THEN_TRAP };

static SLJIT_INLINE void compile_then_trap_matchingpath(compiler_common *common, PCRE2_SPTR cc, PCRE2_SPTR ccend, backtrack_common *parent)
{
DEFINE_COMPILER;
backtrack_common *backtrack;
BOOL needs_control_head;
int size;

PUSH_BACKTRACK_NOVALUE(sizeof(then_trap_backtrack), cc);
common->then_trap = BACKTRACK_AS(then_trap_backtrack);
BACKTRACK_AS(then_trap_backtrack)->common.cc = then_trap_opcode;
BACKTRACK_AS(then_trap_backtrack)->start = (sljit_sw)(cc - common->start);
BACKTRACK_AS(then_trap_backtrack)->framesize = get_framesize(common, cc, ccend, FALSE, &needs_control_head);

size = BACKTRACK_AS(then_trap_backtrack)->framesize;
size = 3 + (size < 0 ? 0 : size);

OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr);
allocate_stack(common, size);
if (size > 3)
  OP2(SLJIT_SUB, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, STACK_TOP, 0, SLJIT_IMM, (size - 3) * sizeof(sljit_sw));
else
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, STACK_TOP, 0);
OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(size - 1), SLJIT_IMM, BACKTRACK_AS(then_trap_backtrack)->start);
OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(size - 2), SLJIT_IMM, type_then_trap);
OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(size - 3), TMP2, 0);

size = BACKTRACK_AS(then_trap_backtrack)->framesize;
if (size >= 0)
  init_frame(common, cc, ccend, size - 1, 0, FALSE);
}

static void compile_matchingpath(compiler_common *common, PCRE2_SPTR cc, PCRE2_SPTR ccend, backtrack_common *parent)
{
DEFINE_COMPILER;
backtrack_common *backtrack;
BOOL has_then_trap = FALSE;
then_trap_backtrack *save_then_trap = NULL;

SLJIT_ASSERT(*ccend == OP_END || (*ccend >= OP_ALT && *ccend <= OP_KETRPOS));

if (common->has_then && common->then_offsets[cc - common->start] != 0)
  {
  SLJIT_ASSERT(*ccend != OP_END && common->control_head_ptr != 0);
  has_then_trap = TRUE;
  save_then_trap = common->then_trap;
  /* Tail item on backtrack. */
  compile_then_trap_matchingpath(common, cc, ccend, parent);
  }

while (cc < ccend)
  {
  switch(*cc)
    {
    case OP_SOD:
    case OP_SOM:
    case OP_NOT_WORD_BOUNDARY:
    case OP_WORD_BOUNDARY:
    case OP_EODN:
    case OP_EOD:
    case OP_DOLL:
    case OP_DOLLM:
    case OP_CIRC:
    case OP_CIRCM:
    case OP_REVERSE:
    cc = compile_simple_assertion_matchingpath(common, *cc, cc + 1, parent->top != NULL ? &parent->top->nextbacktracks : &parent->topbacktracks);
    break;

    case OP_NOT_DIGIT:
    case OP_DIGIT:
    case OP_NOT_WHITESPACE:
    case OP_WHITESPACE:
    case OP_NOT_WORDCHAR:
    case OP_WORDCHAR:
    case OP_ANY:
    case OP_ALLANY:
    case OP_ANYBYTE:
    case OP_NOTPROP:
    case OP_PROP:
    case OP_ANYNL:
    case OP_NOT_HSPACE:
    case OP_HSPACE:
    case OP_NOT_VSPACE:
    case OP_VSPACE:
    case OP_EXTUNI:
    case OP_NOT:
    case OP_NOTI:
    cc = compile_char1_matchingpath(common, *cc, cc + 1, parent->top != NULL ? &parent->top->nextbacktracks : &parent->topbacktracks, TRUE);
    break;

    case OP_SET_SOM:
    PUSH_BACKTRACK_NOVALUE(sizeof(backtrack_common), cc);
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(0));
    allocate_stack(common, 1);
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(0), STR_PTR, 0);
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), TMP2, 0);
    cc++;
    break;

    case OP_CHAR:
    case OP_CHARI:
    if (common->mode == PCRE2_JIT_COMPLETE)
      cc = compile_charn_matchingpath(common, cc, ccend, parent->top != NULL ? &parent->top->nextbacktracks : &parent->topbacktracks);
    else
      cc = compile_char1_matchingpath(common, *cc, cc + 1, parent->top != NULL ? &parent->top->nextbacktracks : &parent->topbacktracks, TRUE);
    break;

    case OP_STAR:
    case OP_MINSTAR:
    case OP_PLUS:
    case OP_MINPLUS:
    case OP_QUERY:
    case OP_MINQUERY:
    case OP_UPTO:
    case OP_MINUPTO:
    case OP_EXACT:
    case OP_POSSTAR:
    case OP_POSPLUS:
    case OP_POSQUERY:
    case OP_POSUPTO:
    case OP_STARI:
    case OP_MINSTARI:
    case OP_PLUSI:
    case OP_MINPLUSI:
    case OP_QUERYI:
    case OP_MINQUERYI:
    case OP_UPTOI:
    case OP_MINUPTOI:
    case OP_EXACTI:
    case OP_POSSTARI:
    case OP_POSPLUSI:
    case OP_POSQUERYI:
    case OP_POSUPTOI:
    case OP_NOTSTAR:
    case OP_NOTMINSTAR:
    case OP_NOTPLUS:
    case OP_NOTMINPLUS:
    case OP_NOTQUERY:
    case OP_NOTMINQUERY:
    case OP_NOTUPTO:
    case OP_NOTMINUPTO:
    case OP_NOTEXACT:
    case OP_NOTPOSSTAR:
    case OP_NOTPOSPLUS:
    case OP_NOTPOSQUERY:
    case OP_NOTPOSUPTO:
    case OP_NOTSTARI:
    case OP_NOTMINSTARI:
    case OP_NOTPLUSI:
    case OP_NOTMINPLUSI:
    case OP_NOTQUERYI:
    case OP_NOTMINQUERYI:
    case OP_NOTUPTOI:
    case OP_NOTMINUPTOI:
    case OP_NOTEXACTI:
    case OP_NOTPOSSTARI:
    case OP_NOTPOSPLUSI:
    case OP_NOTPOSQUERYI:
    case OP_NOTPOSUPTOI:
    case OP_TYPESTAR:
    case OP_TYPEMINSTAR:
    case OP_TYPEPLUS:
    case OP_TYPEMINPLUS:
    case OP_TYPEQUERY:
    case OP_TYPEMINQUERY:
    case OP_TYPEUPTO:
    case OP_TYPEMINUPTO:
    case OP_TYPEEXACT:
    case OP_TYPEPOSSTAR:
    case OP_TYPEPOSPLUS:
    case OP_TYPEPOSQUERY:
    case OP_TYPEPOSUPTO:
    cc = compile_iterator_matchingpath(common, cc, parent);
    break;

    case OP_CLASS:
    case OP_NCLASS:
    if (cc[1 + (32 / sizeof(PCRE2_UCHAR))] >= OP_CRSTAR && cc[1 + (32 / sizeof(PCRE2_UCHAR))] <= OP_CRPOSRANGE)
      cc = compile_iterator_matchingpath(common, cc, parent);
    else
      cc = compile_char1_matchingpath(common, *cc, cc + 1, parent->top != NULL ? &parent->top->nextbacktracks : &parent->topbacktracks, TRUE);
    break;

#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32
    case OP_XCLASS:
    if (*(cc + GET(cc, 1)) >= OP_CRSTAR && *(cc + GET(cc, 1)) <= OP_CRPOSRANGE)
      cc = compile_iterator_matchingpath(common, cc, parent);
    else
      cc = compile_char1_matchingpath(common, *cc, cc + 1, parent->top != NULL ? &parent->top->nextbacktracks : &parent->topbacktracks, TRUE);
    break;
#endif

    case OP_REF:
    case OP_REFI:
    if (cc[1 + IMM2_SIZE] >= OP_CRSTAR && cc[1 + IMM2_SIZE] <= OP_CRPOSRANGE)
      cc = compile_ref_iterator_matchingpath(common, cc, parent);
    else
      {
      compile_ref_matchingpath(common, cc, parent->top != NULL ? &parent->top->nextbacktracks : &parent->topbacktracks, TRUE, FALSE);
      cc += 1 + IMM2_SIZE;
      }
    break;

    case OP_DNREF:
    case OP_DNREFI:
    if (cc[1 + 2 * IMM2_SIZE] >= OP_CRSTAR && cc[1 + 2 * IMM2_SIZE] <= OP_CRPOSRANGE)
      cc = compile_ref_iterator_matchingpath(common, cc, parent);
    else
      {
      compile_dnref_search(common, cc, parent->top != NULL ? &parent->top->nextbacktracks : &parent->topbacktracks);
      compile_ref_matchingpath(common, cc, parent->top != NULL ? &parent->top->nextbacktracks : &parent->topbacktracks, TRUE, FALSE);
      cc += 1 + 2 * IMM2_SIZE;
      }
    break;

    case OP_RECURSE:
    cc = compile_recurse_matchingpath(common, cc, parent);
    break;

    case OP_CALLOUT:
    case OP_CALLOUT_STR:
    cc = compile_callout_matchingpath(common, cc, parent);
    break;

    case OP_ASSERT:
    case OP_ASSERT_NOT:
    case OP_ASSERTBACK:
    case OP_ASSERTBACK_NOT:
    PUSH_BACKTRACK_NOVALUE(sizeof(assert_backtrack), cc);
    cc = compile_assert_matchingpath(common, cc, BACKTRACK_AS(assert_backtrack), FALSE);
    break;

    case OP_BRAMINZERO:
    PUSH_BACKTRACK_NOVALUE(sizeof(braminzero_backtrack), cc);
    cc = bracketend(cc + 1);
    if (*(cc - 1 - LINK_SIZE) != OP_KETRMIN)
      {
      allocate_stack(common, 1);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), STR_PTR, 0);
      }
    else
      {
      allocate_stack(common, 2);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), SLJIT_IMM, 0);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(1), STR_PTR, 0);
      }
    BACKTRACK_AS(braminzero_backtrack)->matchingpath = LABEL();
    count_match(common);
    break;

    case OP_ONCE:
    case OP_ONCE_NC:
    case OP_BRA:
    case OP_CBRA:
    case OP_COND:
    case OP_SBRA:
    case OP_SCBRA:
    case OP_SCOND:
    cc = compile_bracket_matchingpath(common, cc, parent);
    break;

    case OP_BRAZERO:
    if (cc[1] > OP_ASSERTBACK_NOT)
      cc = compile_bracket_matchingpath(common, cc, parent);
    else
      {
      PUSH_BACKTRACK_NOVALUE(sizeof(assert_backtrack), cc);
      cc = compile_assert_matchingpath(common, cc, BACKTRACK_AS(assert_backtrack), FALSE);
      }
    break;

    case OP_BRAPOS:
    case OP_CBRAPOS:
    case OP_SBRAPOS:
    case OP_SCBRAPOS:
    case OP_BRAPOSZERO:
    cc = compile_bracketpos_matchingpath(common, cc, parent);
    break;

    case OP_MARK:
    PUSH_BACKTRACK_NOVALUE(sizeof(backtrack_common), cc);
    SLJIT_ASSERT(common->mark_ptr != 0);
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), common->mark_ptr);
    allocate_stack(common, common->has_skip_arg ? 5 : 1);
    OP1(SLJIT_MOV, TMP1, 0, ARGUMENTS, 0);
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(common->has_skip_arg ? 4 : 0), TMP2, 0);
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, (sljit_sw)(cc + 2));
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->mark_ptr, TMP2, 0);
    OP1(SLJIT_MOV, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, mark_ptr), TMP2, 0);
    if (common->has_skip_arg)
      {
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr);
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, STACK_TOP, 0);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(1), SLJIT_IMM, type_mark);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(2), SLJIT_IMM, (sljit_sw)(cc + 2));
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(3), STR_PTR, 0);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), TMP1, 0);
      }
    cc += 1 + 2 + cc[1];
    break;

    case OP_PRUNE:
    case OP_PRUNE_ARG:
    case OP_SKIP:
    case OP_SKIP_ARG:
    case OP_THEN:
    case OP_THEN_ARG:
    case OP_COMMIT:
    cc = compile_control_verb_matchingpath(common, cc, parent);
    break;

    case OP_FAIL:
    case OP_ACCEPT:
    case OP_ASSERT_ACCEPT:
    cc = compile_fail_accept_matchingpath(common, cc, parent);
    break;

    case OP_CLOSE:
    cc = compile_close_matchingpath(common, cc);
    break;

    case OP_SKIPZERO:
    cc = bracketend(cc + 1);
    break;

    default:
    SLJIT_ASSERT_STOP();
    return;
    }
  if (cc == NULL)
    return;
  }

if (has_then_trap)
  {
  /* Head item on backtrack. */
  PUSH_BACKTRACK_NOVALUE(sizeof(then_trap_backtrack), cc);
  BACKTRACK_AS(then_trap_backtrack)->common.cc = then_trap_opcode;
  BACKTRACK_AS(then_trap_backtrack)->then_trap = common->then_trap;
  common->then_trap = save_then_trap;
  }
SLJIT_ASSERT(cc == ccend);
}

#undef PUSH_BACKTRACK
#undef PUSH_BACKTRACK_NOVALUE
#undef BACKTRACK_AS

#define COMPILE_BACKTRACKINGPATH(current) \
  do \
    { \
    compile_backtrackingpath(common, (current)); \
    if (SLJIT_UNLIKELY(sljit_get_compiler_error(compiler))) \
      return; \
    } \
  while (0)

#define CURRENT_AS(type) ((type *)current)

static void compile_iterator_backtrackingpath(compiler_common *common, struct backtrack_common *current)
{
DEFINE_COMPILER;
PCRE2_SPTR cc = current->cc;
PCRE2_UCHAR opcode;
PCRE2_UCHAR type;
sljit_u32 max = 0, exact;
struct sljit_label *label = NULL;
struct sljit_jump *jump = NULL;
jump_list *jumplist = NULL;
PCRE2_SPTR end;
int private_data_ptr = PRIVATE_DATA(cc);
int base = (private_data_ptr == 0) ? SLJIT_MEM1(STACK_TOP) : SLJIT_MEM1(SLJIT_SP);
int offset0 = (private_data_ptr == 0) ? STACK(0) : private_data_ptr;
int offset1 = (private_data_ptr == 0) ? STACK(1) : private_data_ptr + (int)sizeof(sljit_sw);

cc = get_iterator_parameters(common, cc, &opcode, &type, &max, &exact, &end);

switch(opcode)
  {
  case OP_STAR:
  case OP_UPTO:
  if (type == OP_ANYNL || type == OP_EXTUNI)
    {
    SLJIT_ASSERT(private_data_ptr == 0);
    set_jumps(CURRENT_AS(char_iterator_backtrack)->u.backtracks, LABEL());
    OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
    free_stack(common, 1);
    CMPTO(SLJIT_NOT_EQUAL, STR_PTR, 0, SLJIT_IMM, 0, CURRENT_AS(char_iterator_backtrack)->matchingpath);
    }
  else
    {
    if (CURRENT_AS(char_iterator_backtrack)->u.charpos.enabled)
      {
      OP1(SLJIT_MOV, STR_PTR, 0, base, offset0);
      OP1(SLJIT_MOV, TMP2, 0, base, offset1);
      OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

      jump = CMP(SLJIT_LESS_EQUAL, STR_PTR, 0, TMP2, 0);
      label = LABEL();
      OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-1));
      OP1(SLJIT_MOV, base, offset0, STR_PTR, 0);
      if (CURRENT_AS(char_iterator_backtrack)->u.charpos.othercasebit != 0)
        OP2(SLJIT_OR, TMP1, 0, TMP1, 0, SLJIT_IMM, CURRENT_AS(char_iterator_backtrack)->u.charpos.othercasebit);
      CMPTO(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, CURRENT_AS(char_iterator_backtrack)->u.charpos.chr, CURRENT_AS(char_iterator_backtrack)->matchingpath);
      skip_char_back(common);
      CMPTO(SLJIT_GREATER, STR_PTR, 0, TMP2, 0, label);
      }
    else
      {
      OP1(SLJIT_MOV, STR_PTR, 0, base, offset0);
      jump = CMP(SLJIT_LESS_EQUAL, STR_PTR, 0, base, offset1);
      skip_char_back(common);
      OP1(SLJIT_MOV, base, offset0, STR_PTR, 0);
      JUMPTO(SLJIT_JUMP, CURRENT_AS(char_iterator_backtrack)->matchingpath);
      }
    JUMPHERE(jump);
    if (private_data_ptr == 0)
      free_stack(common, 2);
    }
  break;

  case OP_MINSTAR:
  OP1(SLJIT_MOV, STR_PTR, 0, base, offset0);
  compile_char1_matchingpath(common, type, cc, &jumplist, TRUE);
  OP1(SLJIT_MOV, base, offset0, STR_PTR, 0);
  JUMPTO(SLJIT_JUMP, CURRENT_AS(char_iterator_backtrack)->matchingpath);
  set_jumps(jumplist, LABEL());
  if (private_data_ptr == 0)
    free_stack(common, 1);
  break;

  case OP_MINUPTO:
  OP1(SLJIT_MOV, TMP1, 0, base, offset1);
  OP1(SLJIT_MOV, STR_PTR, 0, base, offset0);
  OP2(SLJIT_SUB | SLJIT_SET_E, TMP1, 0, TMP1, 0, SLJIT_IMM, 1);
  add_jump(compiler, &jumplist, JUMP(SLJIT_ZERO));

  OP1(SLJIT_MOV, base, offset1, TMP1, 0);
  compile_char1_matchingpath(common, type, cc, &jumplist, TRUE);
  OP1(SLJIT_MOV, base, offset0, STR_PTR, 0);
  JUMPTO(SLJIT_JUMP, CURRENT_AS(char_iterator_backtrack)->matchingpath);

  set_jumps(jumplist, LABEL());
  if (private_data_ptr == 0)
    free_stack(common, 2);
  break;

  case OP_QUERY:
  OP1(SLJIT_MOV, STR_PTR, 0, base, offset0);
  OP1(SLJIT_MOV, base, offset0, SLJIT_IMM, 0);
  CMPTO(SLJIT_NOT_EQUAL, STR_PTR, 0, SLJIT_IMM, 0, CURRENT_AS(char_iterator_backtrack)->matchingpath);
  jump = JUMP(SLJIT_JUMP);
  set_jumps(CURRENT_AS(char_iterator_backtrack)->u.backtracks, LABEL());
  OP1(SLJIT_MOV, STR_PTR, 0, base, offset0);
  OP1(SLJIT_MOV, base, offset0, SLJIT_IMM, 0);
  JUMPTO(SLJIT_JUMP, CURRENT_AS(char_iterator_backtrack)->matchingpath);
  JUMPHERE(jump);
  if (private_data_ptr == 0)
    free_stack(common, 1);
  break;

  case OP_MINQUERY:
  OP1(SLJIT_MOV, STR_PTR, 0, base, offset0);
  OP1(SLJIT_MOV, base, offset0, SLJIT_IMM, 0);
  jump = CMP(SLJIT_EQUAL, STR_PTR, 0, SLJIT_IMM, 0);
  compile_char1_matchingpath(common, type, cc, &jumplist, TRUE);
  JUMPTO(SLJIT_JUMP, CURRENT_AS(char_iterator_backtrack)->matchingpath);
  set_jumps(jumplist, LABEL());
  JUMPHERE(jump);
  if (private_data_ptr == 0)
    free_stack(common, 1);
  break;

  case OP_EXACT:
  case OP_POSSTAR:
  case OP_POSQUERY:
  case OP_POSUPTO:
  break;

  default:
  SLJIT_ASSERT_STOP();
  break;
  }

set_jumps(current->topbacktracks, LABEL());
}

static SLJIT_INLINE void compile_ref_iterator_backtrackingpath(compiler_common *common, struct backtrack_common *current)
{
DEFINE_COMPILER;
PCRE2_SPTR cc = current->cc;
BOOL ref = (*cc == OP_REF || *cc == OP_REFI);
PCRE2_UCHAR type;

type = cc[ref ? 1 + IMM2_SIZE : 1 + 2 * IMM2_SIZE];

if ((type & 0x1) == 0)
  {
  /* Maximize case. */
  set_jumps(current->topbacktracks, LABEL());
  OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
  free_stack(common, 1);
  CMPTO(SLJIT_NOT_EQUAL, STR_PTR, 0, SLJIT_IMM, 0, CURRENT_AS(ref_iterator_backtrack)->matchingpath);
  return;
  }

OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
CMPTO(SLJIT_NOT_EQUAL, STR_PTR, 0, SLJIT_IMM, 0, CURRENT_AS(ref_iterator_backtrack)->matchingpath);
set_jumps(current->topbacktracks, LABEL());
free_stack(common, ref ? 2 : 3);
}

static SLJIT_INLINE void compile_recurse_backtrackingpath(compiler_common *common, struct backtrack_common *current)
{
DEFINE_COMPILER;

if (CURRENT_AS(recurse_backtrack)->inlined_pattern)
  compile_backtrackingpath(common, current->top);
set_jumps(current->topbacktracks, LABEL());
if (CURRENT_AS(recurse_backtrack)->inlined_pattern)
  return;

if (common->has_set_som && common->mark_ptr != 0)
  {
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(1));
  free_stack(common, 2);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(0), TMP2, 0);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->mark_ptr, TMP1, 0);
  }
else if (common->has_set_som || common->mark_ptr != 0)
  {
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
  free_stack(common, 1);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->has_set_som ? (int)(OVECTOR(0)) : common->mark_ptr, TMP2, 0);
  }
}

static void compile_assert_backtrackingpath(compiler_common *common, struct backtrack_common *current)
{
DEFINE_COMPILER;
PCRE2_SPTR cc = current->cc;
PCRE2_UCHAR bra = OP_BRA;
struct sljit_jump *brajump = NULL;

SLJIT_ASSERT(*cc != OP_BRAMINZERO);
if (*cc == OP_BRAZERO)
  {
  bra = *cc;
  cc++;
  }

if (bra == OP_BRAZERO)
  {
  SLJIT_ASSERT(current->topbacktracks == NULL);
  OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
  }

if (CURRENT_AS(assert_backtrack)->framesize < 0)
  {
  set_jumps(current->topbacktracks, LABEL());

  if (bra == OP_BRAZERO)
    {
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), SLJIT_IMM, 0);
    CMPTO(SLJIT_NOT_EQUAL, STR_PTR, 0, SLJIT_IMM, 0, CURRENT_AS(assert_backtrack)->matchingpath);
    free_stack(common, 1);
    }
  return;
  }

if (bra == OP_BRAZERO)
  {
  if (*cc == OP_ASSERT_NOT || *cc == OP_ASSERTBACK_NOT)
    {
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), SLJIT_IMM, 0);
    CMPTO(SLJIT_NOT_EQUAL, STR_PTR, 0, SLJIT_IMM, 0, CURRENT_AS(assert_backtrack)->matchingpath);
    free_stack(common, 1);
    return;
    }
  free_stack(common, 1);
  brajump = CMP(SLJIT_EQUAL, STR_PTR, 0, SLJIT_IMM, 0);
  }

if (*cc == OP_ASSERT || *cc == OP_ASSERTBACK)
  {
  OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), CURRENT_AS(assert_backtrack)->private_data_ptr);
  add_jump(compiler, &common->revertframes, JUMP(SLJIT_FAST_CALL));
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), CURRENT_AS(assert_backtrack)->private_data_ptr, SLJIT_MEM1(STACK_TOP), CURRENT_AS(assert_backtrack)->framesize * sizeof(sljit_sw));

  set_jumps(current->topbacktracks, LABEL());
  }
else
  set_jumps(current->topbacktracks, LABEL());

if (bra == OP_BRAZERO)
  {
  /* We know there is enough place on the stack. */
  OP2(SLJIT_ADD, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, sizeof(sljit_sw));
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), SLJIT_IMM, 0);
  JUMPTO(SLJIT_JUMP, CURRENT_AS(assert_backtrack)->matchingpath);
  JUMPHERE(brajump);
  }
}

static void compile_bracket_backtrackingpath(compiler_common *common, struct backtrack_common *current)
{
DEFINE_COMPILER;
int opcode, stacksize, alt_count, alt_max;
int offset = 0;
int private_data_ptr = CURRENT_AS(bracket_backtrack)->private_data_ptr;
int repeat_ptr = 0, repeat_type = 0, repeat_count = 0;
PCRE2_SPTR cc = current->cc;
PCRE2_SPTR ccbegin;
PCRE2_SPTR ccprev;
PCRE2_UCHAR bra = OP_BRA;
PCRE2_UCHAR ket;
assert_backtrack *assert;
sljit_uw *next_update_addr = NULL;
BOOL has_alternatives;
BOOL needs_control_head = FALSE;
struct sljit_jump *brazero = NULL;
struct sljit_jump *alt1 = NULL;
struct sljit_jump *alt2 = NULL;
struct sljit_jump *once = NULL;
struct sljit_jump *cond = NULL;
struct sljit_label *rmin_label = NULL;
struct sljit_label *exact_label = NULL;

if (*cc == OP_BRAZERO || *cc == OP_BRAMINZERO)
  {
  bra = *cc;
  cc++;
  }

opcode = *cc;
ccbegin = bracketend(cc) - 1 - LINK_SIZE;
ket = *ccbegin;
if (ket == OP_KET && PRIVATE_DATA(ccbegin) != 0)
  {
  repeat_ptr = PRIVATE_DATA(ccbegin);
  repeat_type = PRIVATE_DATA(ccbegin + 2);
  repeat_count = PRIVATE_DATA(ccbegin + 3);
  SLJIT_ASSERT(repeat_type != 0 && repeat_count != 0);
  if (repeat_type == OP_UPTO)
    ket = OP_KETRMAX;
  if (repeat_type == OP_MINUPTO)
    ket = OP_KETRMIN;
  }
ccbegin = cc;
cc += GET(cc, 1);
has_alternatives = *cc == OP_ALT;
if (SLJIT_UNLIKELY(opcode == OP_COND) || SLJIT_UNLIKELY(opcode == OP_SCOND))
  has_alternatives = (ccbegin[1 + LINK_SIZE] >= OP_ASSERT && ccbegin[1 + LINK_SIZE] <= OP_ASSERTBACK_NOT) || CURRENT_AS(bracket_backtrack)->u.condfailed != NULL;
if (opcode == OP_CBRA || opcode == OP_SCBRA)
  offset = (GET2(ccbegin, 1 + LINK_SIZE)) << 1;
if (SLJIT_UNLIKELY(opcode == OP_COND) && (*cc == OP_KETRMAX || *cc == OP_KETRMIN))
  opcode = OP_SCOND;
if (SLJIT_UNLIKELY(opcode == OP_ONCE_NC))
  opcode = OP_ONCE;

alt_max = has_alternatives ? no_alternatives(ccbegin) : 0;

/* Decoding the needs_control_head in framesize. */
if (opcode == OP_ONCE)
  {
  needs_control_head = (CURRENT_AS(bracket_backtrack)->u.framesize & 0x1) != 0;
  CURRENT_AS(bracket_backtrack)->u.framesize >>= 1;
  }

if (ket != OP_KET && repeat_type != 0)
  {
  /* TMP1 is used in OP_KETRMIN below. */
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
  free_stack(common, 1);
  if (repeat_type == OP_UPTO)
    OP2(SLJIT_ADD, SLJIT_MEM1(SLJIT_SP), repeat_ptr, TMP1, 0, SLJIT_IMM, 1);
  else
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), repeat_ptr, TMP1, 0);
  }

if (ket == OP_KETRMAX)
  {
  if (bra == OP_BRAZERO)
    {
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
    free_stack(common, 1);
    brazero = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, 0);
    }
  }
else if (ket == OP_KETRMIN)
  {
  if (bra != OP_BRAMINZERO)
    {
    OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
    if (repeat_type != 0)
      {
      /* TMP1 was set a few lines above. */
      CMPTO(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, 0, CURRENT_AS(bracket_backtrack)->recursive_matchingpath);
      /* Drop STR_PTR for non-greedy plus quantifier. */
      if (opcode != OP_ONCE)
        free_stack(common, 1);
      }
    else if (opcode >= OP_SBRA || opcode == OP_ONCE)
      {
      /* Checking zero-length iteration. */
      if (opcode != OP_ONCE || CURRENT_AS(bracket_backtrack)->u.framesize < 0)
        CMPTO(SLJIT_NOT_EQUAL, STR_PTR, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr, CURRENT_AS(bracket_backtrack)->recursive_matchingpath);
      else
        {
        OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
        CMPTO(SLJIT_NOT_EQUAL, STR_PTR, 0, SLJIT_MEM1(TMP1), (CURRENT_AS(bracket_backtrack)->u.framesize + 1) * sizeof(sljit_sw), CURRENT_AS(bracket_backtrack)->recursive_matchingpath);
        }
      /* Drop STR_PTR for non-greedy plus quantifier. */
      if (opcode != OP_ONCE)
        free_stack(common, 1);
      }
    else
      JUMPTO(SLJIT_JUMP, CURRENT_AS(bracket_backtrack)->recursive_matchingpath);
    }
  rmin_label = LABEL();
  if (repeat_type != 0)
    OP2(SLJIT_ADD, SLJIT_MEM1(SLJIT_SP), repeat_ptr, SLJIT_MEM1(SLJIT_SP), repeat_ptr, SLJIT_IMM, 1);
  }
else if (bra == OP_BRAZERO)
  {
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
  free_stack(common, 1);
  brazero = CMP(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, 0);
  }
else if (repeat_type == OP_EXACT)
  {
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), repeat_ptr, SLJIT_IMM, 1);
  exact_label = LABEL();
  }

if (offset != 0)
  {
  if (common->capture_last_ptr != 0)
    {
    SLJIT_ASSERT(common->optimized_cbracket[offset >> 1] == 0);
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(STACK_TOP), STACK(1));
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->capture_last_ptr, TMP1, 0);
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(2));
    free_stack(common, 3);
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset), TMP2, 0);
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1), TMP1, 0);
    }
  else if (common->optimized_cbracket[offset >> 1] == 0)
    {
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(STACK_TOP), STACK(1));
    free_stack(common, 2);
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset), TMP1, 0);
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1), TMP2, 0);
    }
  }

if (SLJIT_UNLIKELY(opcode == OP_ONCE))
  {
  if (CURRENT_AS(bracket_backtrack)->u.framesize >= 0)
    {
    OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
    add_jump(compiler, &common->revertframes, JUMP(SLJIT_FAST_CALL));
    }
  once = JUMP(SLJIT_JUMP);
  }
else if (SLJIT_UNLIKELY(opcode == OP_COND) || SLJIT_UNLIKELY(opcode == OP_SCOND))
  {
  if (has_alternatives)
    {
    /* Always exactly one alternative. */
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
    free_stack(common, 1);

    alt_max = 2;
    alt1 = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, sizeof(sljit_uw));
    }
  }
else if (has_alternatives)
  {
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
  free_stack(common, 1);

  if (alt_max > 4)
    {
    /* Table jump if alt_max is greater than 4. */
    next_update_addr = allocate_read_only_data(common, alt_max * sizeof(sljit_uw));
    if (SLJIT_UNLIKELY(next_update_addr == NULL))
      return;
    sljit_emit_ijump(compiler, SLJIT_JUMP, SLJIT_MEM1(TMP1), (sljit_sw)next_update_addr);
    add_label_addr(common, next_update_addr++);
    }
  else
    {
    if (alt_max == 4)
      alt2 = CMP(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 2 * sizeof(sljit_uw));
    alt1 = CMP(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, sizeof(sljit_uw));
    }
  }

COMPILE_BACKTRACKINGPATH(current->top);
if (current->topbacktracks)
  set_jumps(current->topbacktracks, LABEL());

if (SLJIT_UNLIKELY(opcode == OP_COND) || SLJIT_UNLIKELY(opcode == OP_SCOND))
  {
  /* Conditional block always has at most one alternative. */
  if (ccbegin[1 + LINK_SIZE] >= OP_ASSERT && ccbegin[1 + LINK_SIZE] <= OP_ASSERTBACK_NOT)
    {
    SLJIT_ASSERT(has_alternatives);
    assert = CURRENT_AS(bracket_backtrack)->u.assert;
    if (assert->framesize >= 0 && (ccbegin[1 + LINK_SIZE] == OP_ASSERT || ccbegin[1 + LINK_SIZE] == OP_ASSERTBACK))
      {
      OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), assert->private_data_ptr);
      add_jump(compiler, &common->revertframes, JUMP(SLJIT_FAST_CALL));
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), assert->private_data_ptr, SLJIT_MEM1(STACK_TOP), assert->framesize * sizeof(sljit_sw));
      }
    cond = JUMP(SLJIT_JUMP);
    set_jumps(CURRENT_AS(bracket_backtrack)->u.assert->condfailed, LABEL());
    }
  else if (CURRENT_AS(bracket_backtrack)->u.condfailed != NULL)
    {
    SLJIT_ASSERT(has_alternatives);
    cond = JUMP(SLJIT_JUMP);
    set_jumps(CURRENT_AS(bracket_backtrack)->u.condfailed, LABEL());
    }
  else
    SLJIT_ASSERT(!has_alternatives);
  }

if (has_alternatives)
  {
  alt_count = sizeof(sljit_uw);
  do
    {
    current->top = NULL;
    current->topbacktracks = NULL;
    current->nextbacktracks = NULL;
    /* Conditional blocks always have an additional alternative, even if it is empty. */
    if (*cc == OP_ALT)
      {
      ccprev = cc + 1 + LINK_SIZE;
      cc += GET(cc, 1);
      if (opcode != OP_COND && opcode != OP_SCOND)
        {
        if (opcode != OP_ONCE)
          {
          if (private_data_ptr != 0)
            OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
          else
            OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
          }
        else
          OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(needs_control_head ? 1 : 0));
        }
      compile_matchingpath(common, ccprev, cc, current);
      if (SLJIT_UNLIKELY(sljit_get_compiler_error(compiler)))
        return;
      }

    /* Instructions after the current alternative is successfully matched. */
    /* There is a similar code in compile_bracket_matchingpath. */
    if (opcode == OP_ONCE)
      match_once_common(common, ket, CURRENT_AS(bracket_backtrack)->u.framesize, private_data_ptr, has_alternatives, needs_control_head);

    stacksize = 0;
    if (repeat_type == OP_MINUPTO)
      {
      /* We need to preserve the counter. TMP2 will be used below. */
      OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), repeat_ptr);
      stacksize++;
      }
    if (ket != OP_KET || bra != OP_BRA)
      stacksize++;
    if (offset != 0)
      {
      if (common->capture_last_ptr != 0)
        stacksize++;
      if (common->optimized_cbracket[offset >> 1] == 0)
        stacksize += 2;
      }
    if (opcode != OP_ONCE)
      stacksize++;

    if (stacksize > 0)
      allocate_stack(common, stacksize);

    stacksize = 0;
    if (repeat_type == OP_MINUPTO)
      {
      /* TMP2 was set above. */
      OP2(SLJIT_SUB, SLJIT_MEM1(STACK_TOP), STACK(stacksize), TMP2, 0, SLJIT_IMM, 1);
      stacksize++;
      }

    if (ket != OP_KET || bra != OP_BRA)
      {
      if (ket != OP_KET)
        OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(stacksize), STR_PTR, 0);
      else
        OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(stacksize), SLJIT_IMM, 0);
      stacksize++;
      }

    if (offset != 0)
      stacksize = match_capture_common(common, stacksize, offset, private_data_ptr);

    if (opcode != OP_ONCE)
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(stacksize), SLJIT_IMM, alt_count);

    if (offset != 0 && ket == OP_KETRMAX && common->optimized_cbracket[offset >> 1] != 0)
      {
      /* If ket is not OP_KETRMAX, this code path is executed after the jump to alternative_matchingpath. */
      SLJIT_ASSERT(private_data_ptr == OVECTOR(offset + 0));
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1), STR_PTR, 0);
      }

    JUMPTO(SLJIT_JUMP, CURRENT_AS(bracket_backtrack)->alternative_matchingpath);

    if (opcode != OP_ONCE)
      {
      if (alt_max > 4)
        add_label_addr(common, next_update_addr++);
      else
        {
        if (alt_count != 2 * sizeof(sljit_uw))
          {
          JUMPHERE(alt1);
          if (alt_max == 3 && alt_count == sizeof(sljit_uw))
            alt2 = CMP(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 2 * sizeof(sljit_uw));
          }
        else
          {
          JUMPHERE(alt2);
          if (alt_max == 4)
            alt1 = CMP(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 3 * sizeof(sljit_uw));
          }
        }
      alt_count += sizeof(sljit_uw);
      }

    COMPILE_BACKTRACKINGPATH(current->top);
    if (current->topbacktracks)
      set_jumps(current->topbacktracks, LABEL());
    SLJIT_ASSERT(!current->nextbacktracks);
    }
  while (*cc == OP_ALT);

  if (cond != NULL)
    {
    SLJIT_ASSERT(opcode == OP_COND || opcode == OP_SCOND);
    assert = CURRENT_AS(bracket_backtrack)->u.assert;
    if ((ccbegin[1 + LINK_SIZE] == OP_ASSERT_NOT || ccbegin[1 + LINK_SIZE] == OP_ASSERTBACK_NOT) && assert->framesize >= 0)
      {
      OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), assert->private_data_ptr);
      add_jump(compiler, &common->revertframes, JUMP(SLJIT_FAST_CALL));
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), assert->private_data_ptr, SLJIT_MEM1(STACK_TOP), assert->framesize * sizeof(sljit_sw));
      }
    JUMPHERE(cond);
    }

  /* Free the STR_PTR. */
  if (private_data_ptr == 0)
    free_stack(common, 1);
  }

if (offset != 0)
  {
  /* Using both tmp register is better for instruction scheduling. */
  if (common->optimized_cbracket[offset >> 1] != 0)
    {
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(STACK_TOP), STACK(1));
    free_stack(common, 2);
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset), TMP1, 0);
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1), TMP2, 0);
    }
  else
    {
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
    free_stack(common, 1);
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, TMP1, 0);
    }
  }
else if (opcode == OP_SBRA || opcode == OP_SCOND)
  {
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, SLJIT_MEM1(STACK_TOP), STACK(0));
  free_stack(common, 1);
  }
else if (opcode == OP_ONCE)
  {
  cc = ccbegin + GET(ccbegin, 1);
  stacksize = needs_control_head ? 1 : 0;

  if (CURRENT_AS(bracket_backtrack)->u.framesize >= 0)
    {
    /* Reset head and drop saved frame. */
    stacksize += CURRENT_AS(bracket_backtrack)->u.framesize + ((ket != OP_KET || *cc == OP_ALT) ? 2 : 1);
    }
  else if (ket == OP_KETRMAX || (*cc == OP_ALT && ket != OP_KETRMIN))
    {
    /* The STR_PTR must be released. */
    stacksize++;
    }

  if (stacksize > 0)
    free_stack(common, stacksize);

  JUMPHERE(once);
  /* Restore previous private_data_ptr */
  if (CURRENT_AS(bracket_backtrack)->u.framesize >= 0)
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, SLJIT_MEM1(STACK_TOP), CURRENT_AS(bracket_backtrack)->u.framesize * sizeof(sljit_sw));
  else if (ket == OP_KETRMIN)
    {
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(1));
    /* See the comment below. */
    free_stack(common, 2);
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, TMP1, 0);
    }
  }

if (repeat_type == OP_EXACT)
  {
  OP2(SLJIT_ADD, TMP1, 0, SLJIT_MEM1(SLJIT_SP), repeat_ptr, SLJIT_IMM, 1);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), repeat_ptr, TMP1, 0);
  CMPTO(SLJIT_LESS_EQUAL, TMP1, 0, SLJIT_IMM, repeat_count, exact_label);
  }
else if (ket == OP_KETRMAX)
  {
  OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
  if (bra != OP_BRAZERO)
    free_stack(common, 1);

  CMPTO(SLJIT_NOT_EQUAL, STR_PTR, 0, SLJIT_IMM, 0, CURRENT_AS(bracket_backtrack)->recursive_matchingpath);
  if (bra == OP_BRAZERO)
    {
    OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(1));
    JUMPTO(SLJIT_JUMP, CURRENT_AS(bracket_backtrack)->zero_matchingpath);
    JUMPHERE(brazero);
    free_stack(common, 1);
    }
  }
else if (ket == OP_KETRMIN)
  {
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(0));

  /* OP_ONCE removes everything in case of a backtrack, so we don't
  need to explicitly release the STR_PTR. The extra release would
  affect badly the free_stack(2) above. */
  if (opcode != OP_ONCE)
    free_stack(common, 1);
  CMPTO(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, 0, rmin_label);
  if (opcode == OP_ONCE)
    free_stack(common, bra == OP_BRAMINZERO ? 2 : 1);
  else if (bra == OP_BRAMINZERO)
    free_stack(common, 1);
  }
else if (bra == OP_BRAZERO)
  {
  OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
  JUMPTO(SLJIT_JUMP, CURRENT_AS(bracket_backtrack)->zero_matchingpath);
  JUMPHERE(brazero);
  }
}

static SLJIT_INLINE void compile_bracketpos_backtrackingpath(compiler_common *common, struct backtrack_common *current)
{
DEFINE_COMPILER;
int offset;
struct sljit_jump *jump;

if (CURRENT_AS(bracketpos_backtrack)->framesize < 0)
  {
  if (*current->cc == OP_CBRAPOS || *current->cc == OP_SCBRAPOS)
    {
    offset = (GET2(current->cc, 1 + LINK_SIZE)) << 1;
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(STACK_TOP), STACK(1));
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset), TMP1, 0);
    if (common->capture_last_ptr != 0)
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(2));
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1), TMP2, 0);
    if (common->capture_last_ptr != 0)
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->capture_last_ptr, TMP1, 0);
    }
  set_jumps(current->topbacktracks, LABEL());
  free_stack(common, CURRENT_AS(bracketpos_backtrack)->stacksize);
  return;
  }

OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), CURRENT_AS(bracketpos_backtrack)->private_data_ptr);
add_jump(compiler, &common->revertframes, JUMP(SLJIT_FAST_CALL));

if (current->topbacktracks)
  {
  jump = JUMP(SLJIT_JUMP);
  set_jumps(current->topbacktracks, LABEL());
  /* Drop the stack frame. */
  free_stack(common, CURRENT_AS(bracketpos_backtrack)->stacksize);
  JUMPHERE(jump);
  }
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), CURRENT_AS(bracketpos_backtrack)->private_data_ptr, SLJIT_MEM1(STACK_TOP), CURRENT_AS(bracketpos_backtrack)->framesize * sizeof(sljit_sw));
}

static SLJIT_INLINE void compile_braminzero_backtrackingpath(compiler_common *common, struct backtrack_common *current)
{
assert_backtrack backtrack;

current->top = NULL;
current->topbacktracks = NULL;
current->nextbacktracks = NULL;
if (current->cc[1] > OP_ASSERTBACK_NOT)
  {
  /* Manual call of compile_bracket_matchingpath and compile_bracket_backtrackingpath. */
  compile_bracket_matchingpath(common, current->cc, current);
  compile_bracket_backtrackingpath(common, current->top);
  }
else
  {
  memset(&backtrack, 0, sizeof(backtrack));
  backtrack.common.cc = current->cc;
  backtrack.matchingpath = CURRENT_AS(braminzero_backtrack)->matchingpath;
  /* Manual call of compile_assert_matchingpath. */
  compile_assert_matchingpath(common, current->cc, &backtrack, FALSE);
  }
SLJIT_ASSERT(!current->nextbacktracks && !current->topbacktracks);
}

static SLJIT_INLINE void compile_control_verb_backtrackingpath(compiler_common *common, struct backtrack_common *current)
{
DEFINE_COMPILER;
PCRE2_UCHAR opcode = *current->cc;
struct sljit_label *loop;
struct sljit_jump *jump;

if (opcode == OP_THEN || opcode == OP_THEN_ARG)
  {
  if (common->then_trap != NULL)
    {
    SLJIT_ASSERT(common->control_head_ptr != 0);

    OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr);
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, type_then_trap);
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, common->then_trap->start);
    jump = JUMP(SLJIT_JUMP);

    loop = LABEL();
    OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(STACK_TOP), -(int)sizeof(sljit_sw));
    JUMPHERE(jump);
    CMPTO(SLJIT_NOT_EQUAL, SLJIT_MEM1(STACK_TOP), -(int)(2 * sizeof(sljit_sw)), TMP1, 0, loop);
    CMPTO(SLJIT_NOT_EQUAL, SLJIT_MEM1(STACK_TOP), -(int)(3 * sizeof(sljit_sw)), TMP2, 0, loop);
    add_jump(compiler, &common->then_trap->quit, JUMP(SLJIT_JUMP));
    return;
    }
  else if (common->positive_assert)
    {
    add_jump(compiler, &common->positive_assert_quit, JUMP(SLJIT_JUMP));
    return;
    }
  }

if (common->local_exit)
  {
  if (common->quit_label == NULL)
    add_jump(compiler, &common->quit, JUMP(SLJIT_JUMP));
  else
    JUMPTO(SLJIT_JUMP, common->quit_label);
  return;
  }

if (opcode == OP_SKIP_ARG)
  {
  SLJIT_ASSERT(common->control_head_ptr != 0);
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCALS0, STACK_TOP, 0);
  OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_IMM, (sljit_sw)(current->cc + 2));
  sljit_emit_ijump(compiler, SLJIT_CALL2, SLJIT_IMM, SLJIT_FUNC_OFFSET(do_search_mark));
  OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), LOCALS0);

  OP1(SLJIT_MOV, STR_PTR, 0, TMP1, 0);
  add_jump(compiler, &common->reset_match, CMP(SLJIT_NOT_EQUAL, STR_PTR, 0, SLJIT_IMM, -1));
  return;
  }

if (opcode == OP_SKIP)
  OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
else
  OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_IMM, 0);
add_jump(compiler, &common->reset_match, JUMP(SLJIT_JUMP));
}

static SLJIT_INLINE void compile_then_trap_backtrackingpath(compiler_common *common, struct backtrack_common *current)
{
DEFINE_COMPILER;
struct sljit_jump *jump;
int size;

if (CURRENT_AS(then_trap_backtrack)->then_trap)
  {
  common->then_trap = CURRENT_AS(then_trap_backtrack)->then_trap;
  return;
  }

size = CURRENT_AS(then_trap_backtrack)->framesize;
size = 3 + (size < 0 ? 0 : size);

OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(size - 3));
free_stack(common, size);
jump = JUMP(SLJIT_JUMP);

set_jumps(CURRENT_AS(then_trap_backtrack)->quit, LABEL());
/* STACK_TOP is set by THEN. */
if (CURRENT_AS(then_trap_backtrack)->framesize >= 0)
  add_jump(compiler, &common->revertframes, JUMP(SLJIT_FAST_CALL));
OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
free_stack(common, 3);

JUMPHERE(jump);
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, TMP1, 0);
}

static void compile_backtrackingpath(compiler_common *common, struct backtrack_common *current)
{
DEFINE_COMPILER;
then_trap_backtrack *save_then_trap = common->then_trap;

while (current)
  {
  if (current->nextbacktracks != NULL)
    set_jumps(current->nextbacktracks, LABEL());
  switch(*current->cc)
    {
    case OP_SET_SOM:
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
    free_stack(common, 1);
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(0), TMP1, 0);
    break;

    case OP_STAR:
    case OP_MINSTAR:
    case OP_PLUS:
    case OP_MINPLUS:
    case OP_QUERY:
    case OP_MINQUERY:
    case OP_UPTO:
    case OP_MINUPTO:
    case OP_EXACT:
    case OP_POSSTAR:
    case OP_POSPLUS:
    case OP_POSQUERY:
    case OP_POSUPTO:
    case OP_STARI:
    case OP_MINSTARI:
    case OP_PLUSI:
    case OP_MINPLUSI:
    case OP_QUERYI:
    case OP_MINQUERYI:
    case OP_UPTOI:
    case OP_MINUPTOI:
    case OP_EXACTI:
    case OP_POSSTARI:
    case OP_POSPLUSI:
    case OP_POSQUERYI:
    case OP_POSUPTOI:
    case OP_NOTSTAR:
    case OP_NOTMINSTAR:
    case OP_NOTPLUS:
    case OP_NOTMINPLUS:
    case OP_NOTQUERY:
    case OP_NOTMINQUERY:
    case OP_NOTUPTO:
    case OP_NOTMINUPTO:
    case OP_NOTEXACT:
    case OP_NOTPOSSTAR:
    case OP_NOTPOSPLUS:
    case OP_NOTPOSQUERY:
    case OP_NOTPOSUPTO:
    case OP_NOTSTARI:
    case OP_NOTMINSTARI:
    case OP_NOTPLUSI:
    case OP_NOTMINPLUSI:
    case OP_NOTQUERYI:
    case OP_NOTMINQUERYI:
    case OP_NOTUPTOI:
    case OP_NOTMINUPTOI:
    case OP_NOTEXACTI:
    case OP_NOTPOSSTARI:
    case OP_NOTPOSPLUSI:
    case OP_NOTPOSQUERYI:
    case OP_NOTPOSUPTOI:
    case OP_TYPESTAR:
    case OP_TYPEMINSTAR:
    case OP_TYPEPLUS:
    case OP_TYPEMINPLUS:
    case OP_TYPEQUERY:
    case OP_TYPEMINQUERY:
    case OP_TYPEUPTO:
    case OP_TYPEMINUPTO:
    case OP_TYPEEXACT:
    case OP_TYPEPOSSTAR:
    case OP_TYPEPOSPLUS:
    case OP_TYPEPOSQUERY:
    case OP_TYPEPOSUPTO:
    case OP_CLASS:
    case OP_NCLASS:
#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH != 8
    case OP_XCLASS:
#endif
    compile_iterator_backtrackingpath(common, current);
    break;

    case OP_REF:
    case OP_REFI:
    case OP_DNREF:
    case OP_DNREFI:
    compile_ref_iterator_backtrackingpath(common, current);
    break;

    case OP_RECURSE:
    compile_recurse_backtrackingpath(common, current);
    break;

    case OP_ASSERT:
    case OP_ASSERT_NOT:
    case OP_ASSERTBACK:
    case OP_ASSERTBACK_NOT:
    compile_assert_backtrackingpath(common, current);
    break;

    case OP_ONCE:
    case OP_ONCE_NC:
    case OP_BRA:
    case OP_CBRA:
    case OP_COND:
    case OP_SBRA:
    case OP_SCBRA:
    case OP_SCOND:
    compile_bracket_backtrackingpath(common, current);
    break;

    case OP_BRAZERO:
    if (current->cc[1] > OP_ASSERTBACK_NOT)
      compile_bracket_backtrackingpath(common, current);
    else
      compile_assert_backtrackingpath(common, current);
    break;

    case OP_BRAPOS:
    case OP_CBRAPOS:
    case OP_SBRAPOS:
    case OP_SCBRAPOS:
    case OP_BRAPOSZERO:
    compile_bracketpos_backtrackingpath(common, current);
    break;

    case OP_BRAMINZERO:
    compile_braminzero_backtrackingpath(common, current);
    break;

    case OP_MARK:
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(common->has_skip_arg ? 4 : 0));
    if (common->has_skip_arg)
      OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
    free_stack(common, common->has_skip_arg ? 5 : 1);
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->mark_ptr, TMP1, 0);
    if (common->has_skip_arg)
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, TMP2, 0);
    break;

    case OP_THEN:
    case OP_THEN_ARG:
    case OP_PRUNE:
    case OP_PRUNE_ARG:
    case OP_SKIP:
    case OP_SKIP_ARG:
    compile_control_verb_backtrackingpath(common, current);
    break;

    case OP_COMMIT:
    if (!common->local_exit)
      OP1(SLJIT_MOV, SLJIT_RETURN_REG, 0, SLJIT_IMM, PCRE2_ERROR_NOMATCH);
    if (common->quit_label == NULL)
      add_jump(compiler, &common->quit, JUMP(SLJIT_JUMP));
    else
      JUMPTO(SLJIT_JUMP, common->quit_label);
    break;

    case OP_CALLOUT:
    case OP_CALLOUT_STR:
    case OP_FAIL:
    case OP_ACCEPT:
    case OP_ASSERT_ACCEPT:
    set_jumps(current->topbacktracks, LABEL());
    break;

    case OP_THEN_TRAP:
    /* A virtual opcode for then traps. */
    compile_then_trap_backtrackingpath(common, current);
    break;

    default:
    SLJIT_ASSERT_STOP();
    break;
    }
  current = current->prev;
  }
common->then_trap = save_then_trap;
}

static SLJIT_INLINE void compile_recurse(compiler_common *common)
{
DEFINE_COMPILER;
PCRE2_SPTR cc = common->start + common->currententry->start;
PCRE2_SPTR ccbegin = cc + 1 + LINK_SIZE + (*cc == OP_BRA ? 0 : IMM2_SIZE);
PCRE2_SPTR ccend = bracketend(cc) - (1 + LINK_SIZE);
BOOL needs_control_head;
int framesize = get_framesize(common, cc, NULL, TRUE, &needs_control_head);
int private_data_size = get_private_data_copy_length(common, ccbegin, ccend, needs_control_head);
int alternativesize;
BOOL needs_frame;
backtrack_common altbacktrack;
struct sljit_jump *jump;

/* Recurse captures then. */
common->then_trap = NULL;

SLJIT_ASSERT(*cc == OP_BRA || *cc == OP_CBRA || *cc == OP_CBRAPOS || *cc == OP_SCBRA || *cc == OP_SCBRAPOS);
needs_frame = framesize >= 0;
if (!needs_frame)
  framesize = 0;
alternativesize = *(cc + GET(cc, 1)) == OP_ALT ? 1 : 0;

SLJIT_ASSERT(common->currententry->entry == NULL && common->recursive_head_ptr != 0);
common->currententry->entry = LABEL();
set_jumps(common->currententry->calls, common->currententry->entry);

sljit_emit_fast_enter(compiler, TMP2, 0);
count_match(common);
allocate_stack(common, private_data_size + framesize + alternativesize);
OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(private_data_size + framesize + alternativesize - 1), TMP2, 0);
copy_private_data(common, ccbegin, ccend, TRUE, private_data_size + framesize + alternativesize, framesize + alternativesize, needs_control_head);
if (needs_control_head)
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, SLJIT_IMM, 0);
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->recursive_head_ptr, STACK_TOP, 0);
if (needs_frame)
  init_frame(common, cc, NULL, framesize + alternativesize - 1, alternativesize, TRUE);

if (alternativesize > 0)
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), STR_PTR, 0);

memset(&altbacktrack, 0, sizeof(backtrack_common));
common->quit_label = NULL;
common->accept_label = NULL;
common->quit = NULL;
common->accept = NULL;
altbacktrack.cc = ccbegin;
cc += GET(cc, 1);
while (1)
  {
  altbacktrack.top = NULL;
  altbacktrack.topbacktracks = NULL;

  if (altbacktrack.cc != ccbegin)
    OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(0));

  compile_matchingpath(common, altbacktrack.cc, cc, &altbacktrack);
  if (SLJIT_UNLIKELY(sljit_get_compiler_error(compiler)))
    return;

  add_jump(compiler, &common->accept, JUMP(SLJIT_JUMP));

  compile_backtrackingpath(common, altbacktrack.top);
  if (SLJIT_UNLIKELY(sljit_get_compiler_error(compiler)))
    return;
  set_jumps(altbacktrack.topbacktracks, LABEL());

  if (*cc != OP_ALT)
    break;

  altbacktrack.cc = cc + 1 + LINK_SIZE;
  cc += GET(cc, 1);
  }

/* None of them matched. */
OP1(SLJIT_MOV, TMP3, 0, SLJIT_IMM, 0);
jump = JUMP(SLJIT_JUMP);

if (common->quit != NULL)
  {
  set_jumps(common->quit, LABEL());
  OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), common->recursive_head_ptr);
  if (needs_frame)
    {
    OP2(SLJIT_SUB, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, (framesize + alternativesize) * sizeof(sljit_sw));
    add_jump(compiler, &common->revertframes, JUMP(SLJIT_FAST_CALL));
    OP2(SLJIT_ADD, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, (framesize + alternativesize) * sizeof(sljit_sw));
    }
  OP1(SLJIT_MOV, TMP3, 0, SLJIT_IMM, 0);
  common->quit = NULL;
  add_jump(compiler, &common->quit, JUMP(SLJIT_JUMP));
  }

set_jumps(common->accept, LABEL());
OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), common->recursive_head_ptr);
if (needs_frame)
  {
  OP2(SLJIT_SUB, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, (framesize + alternativesize) * sizeof(sljit_sw));
  add_jump(compiler, &common->revertframes, JUMP(SLJIT_FAST_CALL));
  OP2(SLJIT_ADD, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, (framesize + alternativesize) * sizeof(sljit_sw));
  }
OP1(SLJIT_MOV, TMP3, 0, SLJIT_IMM, 1);

JUMPHERE(jump);
if (common->quit != NULL)
  set_jumps(common->quit, LABEL());
copy_private_data(common, ccbegin, ccend, FALSE, private_data_size + framesize + alternativesize, framesize + alternativesize, needs_control_head);
free_stack(common, private_data_size + framesize + alternativesize);
if (needs_control_head)
  {
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), 2 * sizeof(sljit_sw));
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(STACK_TOP), sizeof(sljit_sw));
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->recursive_head_ptr, TMP1, 0);
  OP1(SLJIT_MOV, TMP1, 0, TMP3, 0);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, TMP2, 0);
  }
else
  {
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(STACK_TOP), sizeof(sljit_sw));
  OP1(SLJIT_MOV, TMP1, 0, TMP3, 0);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->recursive_head_ptr, TMP2, 0);
  }
sljit_emit_fast_return(compiler, SLJIT_MEM1(STACK_TOP), 0);
}

#undef COMPILE_BACKTRACKINGPATH
#undef CURRENT_AS

static int jit_compile(pcre2_code *code, sljit_u32 mode)
{
pcre2_real_code *re = (pcre2_real_code *)code;
struct sljit_compiler *compiler;
backtrack_common rootbacktrack;
compiler_common common_data;
compiler_common *common = &common_data;
const sljit_u8 *tables = re->tables;
void *allocator_data = &re->memctl;
int private_data_size;
PCRE2_SPTR ccend;
executable_functions *functions;
void *executable_func;
sljit_uw executable_size;
sljit_uw total_length;
label_addr_list *label_addr;
struct sljit_label *mainloop_label = NULL;
struct sljit_label *continue_match_label;
struct sljit_label *empty_match_found_label = NULL;
struct sljit_label *empty_match_backtrack_label = NULL;
struct sljit_label *reset_match_label;
struct sljit_label *quit_label;
struct sljit_jump *jump;
struct sljit_jump *minlength_check_failed = NULL;
struct sljit_jump *reqbyte_notfound = NULL;
struct sljit_jump *empty_match = NULL;

SLJIT_ASSERT(tables);

memset(&rootbacktrack, 0, sizeof(backtrack_common));
memset(common, 0, sizeof(compiler_common));
common->name_table = (PCRE2_SPTR)((uint8_t *)re + sizeof(pcre2_real_code));
rootbacktrack.cc = common->name_table + re->name_count * re->name_entry_size;

common->start = rootbacktrack.cc;
common->read_only_data_head = NULL;
common->fcc = tables + fcc_offset;
common->lcc = (sljit_sw)(tables + lcc_offset);
common->mode = mode;
common->might_be_empty = re->minlength == 0;
common->nltype = NLTYPE_FIXED;
switch(re->newline_convention)
  {
  case PCRE2_NEWLINE_CR: common->newline = CHAR_CR; break;
  case PCRE2_NEWLINE_LF: common->newline = CHAR_NL; break;
  case PCRE2_NEWLINE_CRLF: common->newline = (CHAR_CR << 8) | CHAR_NL; break;
  case PCRE2_NEWLINE_ANY: common->newline = (CHAR_CR << 8) | CHAR_NL; common->nltype = NLTYPE_ANY; break;
  case PCRE2_NEWLINE_ANYCRLF: common->newline = (CHAR_CR << 8) | CHAR_NL; common->nltype = NLTYPE_ANYCRLF; break;
  default: return PCRE2_ERROR_INTERNAL;
  }
common->nlmax = READ_CHAR_MAX;
common->nlmin = 0;
if (re->bsr_convention == PCRE2_BSR_UNICODE)
  common->bsr_nltype = NLTYPE_ANY;
else if (re->bsr_convention == PCRE2_BSR_ANYCRLF)
  common->bsr_nltype = NLTYPE_ANYCRLF;
else
  {
#ifdef BSR_ANYCRLF
  common->bsr_nltype = NLTYPE_ANYCRLF;
#else
  common->bsr_nltype = NLTYPE_ANY;
#endif
  }
common->bsr_nlmax = READ_CHAR_MAX;
common->bsr_nlmin = 0;
common->endonly = (re->overall_options & PCRE2_DOLLAR_ENDONLY) != 0;
common->ctypes = (sljit_sw)(tables + ctypes_offset);
common->name_count = re->name_count;
common->name_entry_size = re->name_entry_size;
common->unset_backref = (re->overall_options & PCRE2_MATCH_UNSET_BACKREF) != 0;
common->alt_circumflex = (re->overall_options & PCRE2_ALT_CIRCUMFLEX) != 0;
#ifdef SUPPORT_UNICODE
/* PCRE_UTF[16|32] have the same value as PCRE_UTF8. */
common->utf = (re->overall_options & PCRE2_UTF) != 0;
common->use_ucp = (re->overall_options & PCRE2_UCP) != 0;
if (common->utf)
  {
  if (common->nltype == NLTYPE_ANY)
    common->nlmax = 0x2029;
  else if (common->nltype == NLTYPE_ANYCRLF)
    common->nlmax = (CHAR_CR > CHAR_NL) ? CHAR_CR : CHAR_NL;
  else
    {
    /* We only care about the first newline character. */
    common->nlmax = common->newline & 0xff;
    }

  if (common->nltype == NLTYPE_FIXED)
    common->nlmin = common->newline & 0xff;
  else
    common->nlmin = (CHAR_CR < CHAR_NL) ? CHAR_CR : CHAR_NL;

  if (common->bsr_nltype == NLTYPE_ANY)
    common->bsr_nlmax = 0x2029;
  else
    common->bsr_nlmax = (CHAR_CR > CHAR_NL) ? CHAR_CR : CHAR_NL;
  common->bsr_nlmin = (CHAR_CR < CHAR_NL) ? CHAR_CR : CHAR_NL;
  }
#endif /* SUPPORT_UNICODE */
ccend = bracketend(common->start);

/* Calculate the local space size on the stack. */
common->ovector_start = LIMIT_MATCH + sizeof(sljit_sw);
common->optimized_cbracket = (sljit_u8 *)SLJIT_MALLOC(re->top_bracket + 1, allocator_data);
if (!common->optimized_cbracket)
  return PCRE2_ERROR_NOMEMORY;
#if defined DEBUG_FORCE_UNOPTIMIZED_CBRAS && DEBUG_FORCE_UNOPTIMIZED_CBRAS == 1
memset(common->optimized_cbracket, 0, re->top_bracket + 1);
#else
memset(common->optimized_cbracket, 1, re->top_bracket + 1);
#endif

SLJIT_ASSERT(*common->start == OP_BRA && ccend[-(1 + LINK_SIZE)] == OP_KET);
#if defined DEBUG_FORCE_UNOPTIMIZED_CBRAS && DEBUG_FORCE_UNOPTIMIZED_CBRAS == 2
common->capture_last_ptr = common->ovector_start;
common->ovector_start += sizeof(sljit_sw);
#endif
if (!check_opcode_types(common, common->start, ccend))
  {
  SLJIT_FREE(common->optimized_cbracket, allocator_data);
  return PCRE2_ERROR_NOMEMORY;
  }

/* Checking flags and updating ovector_start. */
if (mode == PCRE2_JIT_COMPLETE && (re->flags & PCRE2_LASTSET) != 0 && (re->overall_options & PCRE2_NO_START_OPTIMIZE) == 0)
  {
  common->req_char_ptr = common->ovector_start;
  common->ovector_start += sizeof(sljit_sw);
  }
if (mode != PCRE2_JIT_COMPLETE)
  {
  common->start_used_ptr = common->ovector_start;
  common->ovector_start += sizeof(sljit_sw);
  if (mode == PCRE2_JIT_PARTIAL_SOFT)
    {
    common->hit_start = common->ovector_start;
    common->ovector_start += sizeof(sljit_sw);
    }
  }
if ((re->overall_options & (PCRE2_FIRSTLINE | PCRE2_USE_OFFSET_LIMIT)) != 0)
  {
  common->match_end_ptr = common->ovector_start;
  common->ovector_start += sizeof(sljit_sw);
  }
#if defined DEBUG_FORCE_CONTROL_HEAD && DEBUG_FORCE_CONTROL_HEAD
common->control_head_ptr = 1;
#endif
if (common->control_head_ptr != 0)
  {
  common->control_head_ptr = common->ovector_start;
  common->ovector_start += sizeof(sljit_sw);
  }
if (common->has_set_som)
  {
  /* Saving the real start pointer is necessary. */
  common->start_ptr = common->ovector_start;
  common->ovector_start += sizeof(sljit_sw);
  }

/* Aligning ovector to even number of sljit words. */
if ((common->ovector_start & sizeof(sljit_sw)) != 0)
  common->ovector_start += sizeof(sljit_sw);

if (common->start_ptr == 0)
  common->start_ptr = OVECTOR(0);

/* Capturing brackets cannot be optimized if callouts are allowed. */
if (common->capture_last_ptr != 0)
  memset(common->optimized_cbracket, 0, re->top_bracket + 1);

SLJIT_ASSERT(!(common->req_char_ptr != 0 && common->start_used_ptr != 0));
common->cbra_ptr = OVECTOR_START + (re->top_bracket + 1) * 2 * sizeof(sljit_sw);

total_length = ccend - common->start;
common->private_data_ptrs = (sljit_s32 *)SLJIT_MALLOC(total_length * (sizeof(sljit_s32) + (common->has_then ? 1 : 0)), allocator_data);
if (!common->private_data_ptrs)
  {
  SLJIT_FREE(common->optimized_cbracket, allocator_data);
  return PCRE2_ERROR_NOMEMORY;
  }
memset(common->private_data_ptrs, 0, total_length * sizeof(sljit_s32));

private_data_size = common->cbra_ptr + (re->top_bracket + 1) * sizeof(sljit_sw);
set_private_data_ptrs(common, &private_data_size, ccend);
if ((re->overall_options & PCRE2_ANCHORED) == 0 && (re->overall_options & PCRE2_NO_START_OPTIMIZE) == 0)
  {
  if (!detect_fast_forward_skip(common, &private_data_size) && !common->has_skip_in_assert_back)
    detect_fast_fail(common, common->start, &private_data_size, 4);
  }

SLJIT_ASSERT(common->fast_fail_start_ptr <= common->fast_fail_end_ptr);

if (private_data_size > SLJIT_MAX_LOCAL_SIZE)
  {
  SLJIT_FREE(common->private_data_ptrs, allocator_data);
  SLJIT_FREE(common->optimized_cbracket, allocator_data);
  return PCRE2_ERROR_NOMEMORY;
  }

if (common->has_then)
  {
  common->then_offsets = (sljit_u8 *)(common->private_data_ptrs + total_length);
  memset(common->then_offsets, 0, total_length);
  set_then_offsets(common, common->start, NULL);
  }

compiler = sljit_create_compiler(allocator_data);
if (!compiler)
  {
  SLJIT_FREE(common->optimized_cbracket, allocator_data);
  SLJIT_FREE(common->private_data_ptrs, allocator_data);
  return PCRE2_ERROR_NOMEMORY;
  }
common->compiler = compiler;

/* Main pcre_jit_exec entry. */
sljit_emit_enter(compiler, 0, 1, 5, 5, 0, 0, private_data_size);

/* Register init. */
reset_ovector(common, (re->top_bracket + 1) * 2);
if (common->req_char_ptr != 0)
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->req_char_ptr, SLJIT_R0, 0);

OP1(SLJIT_MOV, ARGUMENTS, 0, SLJIT_S0, 0);
OP1(SLJIT_MOV, TMP1, 0, SLJIT_S0, 0);
OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, str));
OP1(SLJIT_MOV, STR_END, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, end));
OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, stack));
OP1(SLJIT_MOV_U32, TMP1, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, limit_match));
OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(TMP2), SLJIT_OFFSETOF(struct sljit_stack, base));
OP1(SLJIT_MOV, STACK_LIMIT, 0, SLJIT_MEM1(TMP2), SLJIT_OFFSETOF(struct sljit_stack, limit));
OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, 1);
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LIMIT_MATCH, TMP1, 0);

if (common->fast_fail_start_ptr < common->fast_fail_end_ptr)
  reset_fast_fail(common);

if (mode == PCRE2_JIT_PARTIAL_SOFT)
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->hit_start, SLJIT_IMM, -1);
if (common->mark_ptr != 0)
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->mark_ptr, SLJIT_IMM, 0);
if (common->control_head_ptr != 0)
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, SLJIT_IMM, 0);

/* Main part of the matching */
if ((re->overall_options & PCRE2_ANCHORED) == 0)
  {
  mainloop_label = mainloop_entry(common, (re->flags & PCRE2_HASCRORLF) != 0, re->overall_options);
  continue_match_label = LABEL();
  /* Forward search if possible. */
  if ((re->overall_options & PCRE2_NO_START_OPTIMIZE) == 0)
    {
    if (mode == PCRE2_JIT_COMPLETE && fast_forward_first_n_chars(common))
      ;
    else if ((re->flags & PCRE2_FIRSTSET) != 0)
      fast_forward_first_char(common, (PCRE2_UCHAR)(re->first_codeunit), (re->flags & PCRE2_FIRSTCASELESS) != 0);
    else if ((re->flags & PCRE2_STARTLINE) != 0)
      fast_forward_newline(common);
    else if ((re->flags & PCRE2_FIRSTMAPSET) != 0)
      fast_forward_start_bits(common, re->start_bitmap);
    }
  }
else
  continue_match_label = LABEL();

if (mode == PCRE2_JIT_COMPLETE && re->minlength > 0 && (re->overall_options & PCRE2_NO_START_OPTIMIZE) == 0)
  {
  OP1(SLJIT_MOV, SLJIT_RETURN_REG, 0, SLJIT_IMM, PCRE2_ERROR_NOMATCH);
  OP2(SLJIT_ADD, TMP2, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(re->minlength));
  minlength_check_failed = CMP(SLJIT_GREATER, TMP2, 0, STR_END, 0);
  }
if (common->req_char_ptr != 0)
  reqbyte_notfound = search_requested_char(common, (PCRE2_UCHAR)(re->last_codeunit), (re->flags & PCRE2_LASTCASELESS) != 0, (re->flags & PCRE2_FIRSTSET) != 0);

/* Store the current STR_PTR in OVECTOR(0). */
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(0), STR_PTR, 0);
/* Copy the limit of allowed recursions. */
OP1(SLJIT_MOV, COUNT_MATCH, 0, SLJIT_MEM1(SLJIT_SP), LIMIT_MATCH);
if (common->capture_last_ptr != 0)
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->capture_last_ptr, SLJIT_IMM, 0);
if (common->fast_forward_bc_ptr != NULL)
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), PRIVATE_DATA(common->fast_forward_bc_ptr + 1), STR_PTR, 0);

if (common->start_ptr != OVECTOR(0))
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->start_ptr, STR_PTR, 0);

/* Copy the beginning of the string. */
if (mode == PCRE2_JIT_PARTIAL_SOFT)
  {
  jump = CMP(SLJIT_NOT_EQUAL, SLJIT_MEM1(SLJIT_SP), common->hit_start, SLJIT_IMM, -1);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->start_used_ptr, STR_PTR, 0);
  JUMPHERE(jump);
  }
else if (mode == PCRE2_JIT_PARTIAL_HARD)
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->start_used_ptr, STR_PTR, 0);

compile_matchingpath(common, common->start, ccend, &rootbacktrack);
if (SLJIT_UNLIKELY(sljit_get_compiler_error(compiler)))
  {
  sljit_free_compiler(compiler);
  SLJIT_FREE(common->optimized_cbracket, allocator_data);
  SLJIT_FREE(common->private_data_ptrs, allocator_data);
  PRIV(jit_free_rodata)(common->read_only_data_head, compiler->allocator_data);
  return PCRE2_ERROR_NOMEMORY;
  }

if (common->might_be_empty)
  {
  empty_match = CMP(SLJIT_EQUAL, STR_PTR, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(0));
  empty_match_found_label = LABEL();
  }

common->accept_label = LABEL();
if (common->accept != NULL)
  set_jumps(common->accept, common->accept_label);

/* This means we have a match. Update the ovector. */
copy_ovector(common, re->top_bracket + 1);
common->quit_label = common->forced_quit_label = LABEL();
if (common->quit != NULL)
  set_jumps(common->quit, common->quit_label);
if (common->forced_quit != NULL)
  set_jumps(common->forced_quit, common->forced_quit_label);
if (minlength_check_failed != NULL)
  SET_LABEL(minlength_check_failed, common->forced_quit_label);
sljit_emit_return(compiler, SLJIT_MOV, SLJIT_RETURN_REG, 0);

if (mode != PCRE2_JIT_COMPLETE)
  {
  common->partialmatchlabel = LABEL();
  set_jumps(common->partialmatch, common->partialmatchlabel);
  return_with_partial_match(common, common->quit_label);
  }

if (common->might_be_empty)
  empty_match_backtrack_label = LABEL();
compile_backtrackingpath(common, rootbacktrack.top);
if (SLJIT_UNLIKELY(sljit_get_compiler_error(compiler)))
  {
  sljit_free_compiler(compiler);
  SLJIT_FREE(common->optimized_cbracket, allocator_data);
  SLJIT_FREE(common->private_data_ptrs, allocator_data);
  PRIV(jit_free_rodata)(common->read_only_data_head, compiler->allocator_data);
  return PCRE2_ERROR_NOMEMORY;
  }

SLJIT_ASSERT(rootbacktrack.prev == NULL);
reset_match_label = LABEL();

if (mode == PCRE2_JIT_PARTIAL_SOFT)
  {
  /* Update hit_start only in the first time. */
  jump = CMP(SLJIT_NOT_EQUAL, SLJIT_MEM1(SLJIT_SP), common->hit_start, SLJIT_IMM, 0);
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->start_ptr);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->start_used_ptr, SLJIT_IMM, -1);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->hit_start, TMP1, 0);
  JUMPHERE(jump);
  }

/* Check we have remaining characters. */
if ((re->overall_options & PCRE2_ANCHORED) == 0 && common->match_end_ptr != 0)
  {
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->match_end_ptr);
  }

OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(SLJIT_SP),
    (common->fast_forward_bc_ptr != NULL) ? (PRIVATE_DATA(common->fast_forward_bc_ptr + 1)) : common->start_ptr);

if ((re->overall_options & PCRE2_ANCHORED) == 0)
  {
  if (common->ff_newline_shortcut != NULL)
    {
    /* There cannot be more newlines if PCRE2_FIRSTLINE is set. */
    if ((re->overall_options & PCRE2_FIRSTLINE) == 0)
      {
      if (common->match_end_ptr != 0)
        {
        OP1(SLJIT_MOV, TMP3, 0, STR_END, 0);
        OP1(SLJIT_MOV, STR_END, 0, TMP1, 0);
        CMPTO(SLJIT_LESS, STR_PTR, 0, TMP1, 0, common->ff_newline_shortcut);
        OP1(SLJIT_MOV, STR_END, 0, TMP3, 0);
        }
      else
        CMPTO(SLJIT_LESS, STR_PTR, 0, STR_END, 0, common->ff_newline_shortcut);
      }
    }
  else
    CMPTO(SLJIT_LESS, STR_PTR, 0, (common->match_end_ptr == 0) ? STR_END : TMP1, 0, mainloop_label);
  }

/* No more remaining characters. */
if (reqbyte_notfound != NULL)
  JUMPHERE(reqbyte_notfound);

if (mode == PCRE2_JIT_PARTIAL_SOFT)
  CMPTO(SLJIT_NOT_EQUAL, SLJIT_MEM1(SLJIT_SP), common->hit_start, SLJIT_IMM, -1, common->partialmatchlabel);

OP1(SLJIT_MOV, SLJIT_RETURN_REG, 0, SLJIT_IMM, PCRE2_ERROR_NOMATCH);
JUMPTO(SLJIT_JUMP, common->quit_label);

flush_stubs(common);

if (common->might_be_empty)
  {
  JUMPHERE(empty_match);
  OP1(SLJIT_MOV, TMP1, 0, ARGUMENTS, 0);
  OP1(SLJIT_MOV_U32, TMP2, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, options));
  OP2(SLJIT_AND | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP2, 0, SLJIT_IMM, PCRE2_NOTEMPTY);
  JUMPTO(SLJIT_NOT_ZERO, empty_match_backtrack_label);
  OP2(SLJIT_AND | SLJIT_SET_E, SLJIT_UNUSED, 0, TMP2, 0, SLJIT_IMM, PCRE2_NOTEMPTY_ATSTART);
  JUMPTO(SLJIT_ZERO, empty_match_found_label);
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, str));
  CMPTO(SLJIT_NOT_EQUAL, TMP2, 0, STR_PTR, 0, empty_match_found_label);
  JUMPTO(SLJIT_JUMP, empty_match_backtrack_label);
  }

common->fast_forward_bc_ptr = NULL;
common->fast_fail_start_ptr = 0;
common->fast_fail_end_ptr = 0;
common->currententry = common->entries;
common->local_exit = TRUE;
quit_label = common->quit_label;
while (common->currententry != NULL)
  {
  /* Might add new entries. */
  compile_recurse(common);
  if (SLJIT_UNLIKELY(sljit_get_compiler_error(compiler)))
    {
    sljit_free_compiler(compiler);
    SLJIT_FREE(common->optimized_cbracket, allocator_data);
    SLJIT_FREE(common->private_data_ptrs, allocator_data);
    PRIV(jit_free_rodata)(common->read_only_data_head, compiler->allocator_data);
    return PCRE2_ERROR_NOMEMORY;
    }
  flush_stubs(common);
  common->currententry = common->currententry->next;
  }
common->local_exit = FALSE;
common->quit_label = quit_label;

/* Allocating stack, returns with PCRE_ERROR_JIT_STACKLIMIT if fails. */
/* This is a (really) rare case. */
set_jumps(common->stackalloc, LABEL());
/* RETURN_ADDR is not a saved register. */
sljit_emit_fast_enter(compiler, SLJIT_MEM1(SLJIT_SP), LOCALS0);
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCALS1, TMP2, 0);
OP1(SLJIT_MOV, TMP1, 0, ARGUMENTS, 0);
OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, stack));
OP1(SLJIT_MOV, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(struct sljit_stack, top), STACK_TOP, 0);
OP2(SLJIT_ADD, TMP2, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(struct sljit_stack, limit), SLJIT_IMM, STACK_GROWTH_RATE);

sljit_emit_ijump(compiler, SLJIT_CALL2, SLJIT_IMM, SLJIT_FUNC_OFFSET(sljit_stack_resize));
jump = CMP(SLJIT_NOT_EQUAL, SLJIT_RETURN_REG, 0, SLJIT_IMM, 0);
OP1(SLJIT_MOV, TMP1, 0, ARGUMENTS, 0);
OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, stack));
OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(struct sljit_stack, top));
OP1(SLJIT_MOV, STACK_LIMIT, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(struct sljit_stack, limit));
OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), LOCALS1);
sljit_emit_fast_return(compiler, SLJIT_MEM1(SLJIT_SP), LOCALS0);

/* Allocation failed. */
JUMPHERE(jump);
/* We break the return address cache here, but this is a really rare case. */
OP1(SLJIT_MOV, SLJIT_RETURN_REG, 0, SLJIT_IMM, PCRE2_ERROR_JIT_STACKLIMIT);
JUMPTO(SLJIT_JUMP, common->quit_label);

/* Call limit reached. */
set_jumps(common->calllimit, LABEL());
OP1(SLJIT_MOV, SLJIT_RETURN_REG, 0, SLJIT_IMM, PCRE2_ERROR_MATCHLIMIT);
JUMPTO(SLJIT_JUMP, common->quit_label);

if (common->revertframes != NULL)
  {
  set_jumps(common->revertframes, LABEL());
  do_revertframes(common);
  }
if (common->wordboundary != NULL)
  {
  set_jumps(common->wordboundary, LABEL());
  check_wordboundary(common);
  }
if (common->anynewline != NULL)
  {
  set_jumps(common->anynewline, LABEL());
  check_anynewline(common);
  }
if (common->hspace != NULL)
  {
  set_jumps(common->hspace, LABEL());
  check_hspace(common);
  }
if (common->vspace != NULL)
  {
  set_jumps(common->vspace, LABEL());
  check_vspace(common);
  }
if (common->casefulcmp != NULL)
  {
  set_jumps(common->casefulcmp, LABEL());
  do_casefulcmp(common);
  }
if (common->caselesscmp != NULL)
  {
  set_jumps(common->caselesscmp, LABEL());
  do_caselesscmp(common);
  }
if (common->reset_match != NULL)
  {
  set_jumps(common->reset_match, LABEL());
  do_reset_match(common, (re->top_bracket + 1) * 2);
  CMPTO(SLJIT_GREATER, STR_PTR, 0, TMP1, 0, continue_match_label);
  OP1(SLJIT_MOV, STR_PTR, 0, TMP1, 0);
  JUMPTO(SLJIT_JUMP, reset_match_label);
  }
#ifdef SUPPORT_UNICODE
#if PCRE2_CODE_UNIT_WIDTH == 8
if (common->utfreadchar != NULL)
  {
  set_jumps(common->utfreadchar, LABEL());
  do_utfreadchar(common);
  }
if (common->utfreadchar16 != NULL)
  {
  set_jumps(common->utfreadchar16, LABEL());
  do_utfreadchar16(common);
  }
if (common->utfreadtype8 != NULL)
  {
  set_jumps(common->utfreadtype8, LABEL());
  do_utfreadtype8(common);
  }
#endif /* PCRE2_CODE_UNIT_WIDTH == 8 */
if (common->getucd != NULL)
  {
  set_jumps(common->getucd, LABEL());
  do_getucd(common);
  }
#endif /* SUPPORT_UNICODE */

SLJIT_FREE(common->optimized_cbracket, allocator_data);
SLJIT_FREE(common->private_data_ptrs, allocator_data);

executable_func = sljit_generate_code(compiler);
executable_size = sljit_get_generated_code_size(compiler);
label_addr = common->label_addrs;
while (label_addr != NULL)
  {
  *label_addr->update_addr = sljit_get_label_addr(label_addr->label);
  label_addr = label_addr->next;
  }
sljit_free_compiler(compiler);
if (executable_func == NULL)
  {
  PRIV(jit_free_rodata)(common->read_only_data_head, compiler->allocator_data);
  return PCRE2_ERROR_NOMEMORY;
  }

/* Reuse the function descriptor if possible. */
if (re->executable_jit != NULL)
  functions = (executable_functions *)re->executable_jit;
else
  {
  functions = SLJIT_MALLOC(sizeof(executable_functions), allocator_data);
  if (functions == NULL)
    {
    /* This case is highly unlikely since we just recently
    freed a lot of memory. Not impossible though. */
    sljit_free_code(executable_func);
    PRIV(jit_free_rodata)(common->read_only_data_head, compiler->allocator_data);
    return PCRE2_ERROR_NOMEMORY;
    }
  memset(functions, 0, sizeof(executable_functions));
  functions->top_bracket = re->top_bracket + 1;
  functions->limit_match = re->limit_match;
  re->executable_jit = functions;
  }

/* Turn mode into an index. */
if (mode == PCRE2_JIT_COMPLETE)
  mode = 0;
else
  mode = (mode == PCRE2_JIT_PARTIAL_SOFT) ? 1 : 2;

SLJIT_ASSERT(mode < JIT_NUMBER_OF_COMPILE_MODES);
functions->executable_funcs[mode] = executable_func;
functions->read_only_data_heads[mode] = common->read_only_data_head;
functions->executable_sizes[mode] = executable_size;
return 0;
}

#endif

/*************************************************
*        JIT compile a Regular Expression        *
*************************************************/

/* This function used JIT to convert a previously-compiled pattern into machine
code.

Arguments:
  code          a compiled pattern
  options       JIT option bits

Returns:        0: success or (*NOJIT) was used
               <0: an error code
*/

#define PUBLIC_JIT_COMPILE_OPTIONS \
  (PCRE2_JIT_COMPLETE|PCRE2_JIT_PARTIAL_SOFT|PCRE2_JIT_PARTIAL_HARD)

PCRE2_EXP_DEFN int PCRE2_CALL_CONVENTION
pcre2_jit_compile(pcre2_code *code, uint32_t options)
{
#ifndef SUPPORT_JIT

(void)code;
(void)options;
return PCRE2_ERROR_JIT_BADOPTION;

#else  /* SUPPORT_JIT */

pcre2_real_code *re = (pcre2_real_code *)code;
executable_functions *functions;
int result;

if (code == NULL)
  return PCRE2_ERROR_NULL;

if ((options & ~PUBLIC_JIT_COMPILE_OPTIONS) != 0)
  return PCRE2_ERROR_JIT_BADOPTION;

if ((re->flags & PCRE2_NOJIT) != 0) return 0;

functions = (executable_functions *)re->executable_jit;

if ((options & PCRE2_JIT_COMPLETE) != 0 && (functions == NULL
    || functions->executable_funcs[0] == NULL)) {
  result = jit_compile(code, PCRE2_JIT_COMPLETE);
  if (result != 0)
    return result;
  }

if ((options & PCRE2_JIT_PARTIAL_SOFT) != 0 && (functions == NULL
    || functions->executable_funcs[1] == NULL)) {
  result = jit_compile(code, PCRE2_JIT_PARTIAL_SOFT);
  if (result != 0)
    return result;
  }

if ((options & PCRE2_JIT_PARTIAL_HARD) != 0 && (functions == NULL
    || functions->executable_funcs[2] == NULL)) {
  result = jit_compile(code, PCRE2_JIT_PARTIAL_HARD);
  if (result != 0)
    return result;
  }

return 0;

#endif  /* SUPPORT_JIT */
}

/* JIT compiler uses an all-in-one approach. This improves security,
   since the code generator functions are not exported. */

#define INCLUDED_FROM_PCRE2_JIT_COMPILE

#include "pcre2_jit_match.c"
#include "pcre2_jit_misc.c"

/* End of pcre2_jit_compile.c */
