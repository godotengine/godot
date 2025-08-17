/*************************************************
*      Perl-Compatible Regular Expressions       *
*************************************************/

/* PCRE is a library of functions to support regular expressions whose syntax
and semantics are as close as possible to those of the Perl 5 language.

                       Written by Philip Hazel
                    This module by Zoltan Herczeg
     Original API code Copyright (c) 1997-2012 University of Cambridge
          New API code Copyright (c) 2016-2024 University of Cambridge

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

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#include <sanitizer/msan_interface.h>
#endif /* __has_feature(memory_sanitizer) */
#endif /* defined(__has_feature) */

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

#include "../deps/sljit/sljit_src/sljitLir.c"

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

enum frame_types {
  no_frame = -1,
  no_stack = -2
};

enum control_types {
  type_mark = 0,
  type_then_trap = 1
};

enum  early_fail_types {
  type_skip = 0,
  type_fail = 1,
  type_fail_range = 2
};

typedef int (SLJIT_FUNC *jit_function)(jit_arguments *args);

/* The following structure is the key data type for the recursive
code generator. It is allocated by compile_matchingpath, and contains
the arguments for compile_backtrackingpath. Must be the first member
of its descendants. */
typedef struct backtrack_common {
  /* Backtracking path of an opcode, which falls back
     to our opcode, if it cannot resume matching. */
  struct backtrack_common *prev;
  /* Backtracks for opcodes without backtracking path.
     These opcodes are between 'prev' and the current
     opcode, and they never resume the match. */
  jump_list *simple_backtracks;
  /* Internal backtracking list for block constructs
     which contains other opcodes, such as brackets,
     asserts, conditionals, etc. */
  struct backtrack_common *top;
  /* Backtracks used internally by the opcode. For component
     opcodes, this list is also used by those opcodes without
     backtracking path which follows the 'top' backtrack. */
  jump_list *own_backtracks;
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
    /* Both for OP_COND, OP_SCOND, OP_ASSERT_SCS. */
    jump_list *no_capture;
    assert_backtrack *assert;
    /* For OP_ONCE. Less than 0 if not needed. */
    int framesize;
  } u;
  /* For brackets with >3 alternatives. */
  struct sljit_jump *matching_mov_addr;
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
  /* Creating a range based on the next character. */
  struct {
    unsigned int othercasebit;
    PCRE2_UCHAR chr;
    BOOL charpos_enabled;
  } charpos;
} char_iterator_backtrack;

typedef struct ref_iterator_backtrack {
  backtrack_common common;
  /* Next iteration. */
  struct sljit_label *matchingpath;
} ref_iterator_backtrack;

typedef struct recurse_entry {
  struct recurse_entry *next;
  /* Contains the function entry label. */
  struct sljit_label *entry_label;
  /* Contains the function entry label. */
  struct sljit_label *backtrack_label;
  /* Collects the entry calls until the function is not created. */
  jump_list *entry_calls;
  /* Collects the backtrack calls until the function is not created. */
  jump_list *backtrack_calls;
  /* Points to the starting opcode. */
  sljit_sw start;
} recurse_entry;

typedef struct recurse_backtrack {
  backtrack_common common;
  /* Return to the matching path. */
  struct sljit_label *matchingpath;
  /* Recursive pattern. */
  recurse_entry *entry;
  /* Pattern is inlined. */
  BOOL inlined_pattern;
} recurse_backtrack;

typedef struct vreverse_backtrack {
  backtrack_common common;
  /* Return to the matching path. */
  struct sljit_label *matchingpath;
} vreverse_backtrack;

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

#define MAX_N_CHARS 12
#define MAX_DIFF_CHARS 5

typedef struct fast_forward_char_data {
  /* Number of characters in the chars array, 255 for any character. */
  sljit_u8 count;
  /* Number of last UTF-8 characters in the chars array. */
  sljit_u8 last_count;
  /* Available characters in the current position. */
  PCRE2_UCHAR chars[MAX_DIFF_CHARS];
} fast_forward_char_data;

#define MAX_CLASS_RANGE_SIZE 4
#define MAX_CLASS_CHARS_SIZE 3

typedef struct compiler_common {
  /* The sljit ceneric compiler. */
  struct sljit_compiler *compiler;
  /* Compiled regular expression. */
  pcre2_real_code *re;
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
#if defined SLJIT_DEBUG && SLJIT_DEBUG
  /* End offset of locals for assertions. */
  sljit_s32 locals_size;
#endif
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
  /* Head of the recursive control verb management chain.
     Each item must have a previous offset and type
     (see control_types) values. See do_search_mark. */
  sljit_s32 control_head_ptr;
  /* The offset of the saved STR_END in the outermost
     scan substring block. Since scan substring restores
     STR_END after a match, it is enough to restore
     STR_END inside a scan substring block. */
  sljit_s32 restore_end_ptr;
  /* Points to the last matched capture block index. */
  sljit_s32 capture_last_ptr;
  /* Fast forward skipping byte code pointer. */
  PCRE2_SPTR fast_forward_bc_ptr;
  /* Locals used by fast fail optimization. */
  sljit_s32 early_fail_start_ptr;
  sljit_s32 early_fail_end_ptr;
  /* Variables used by recursive call generator. */
  sljit_s32 recurse_bitset_size;
  uint8_t *recurse_bitset;

  /* Flipped and lower case tables. */
  const sljit_u8 *fcc;
  sljit_sw lcc;
  /* Mode can be PCRE2_JIT_COMPLETE and others. */
  int mode;
  /* TRUE, when empty match is accepted for partial matching. */
  BOOL allow_empty_partial;
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
  /* Quit is redirected by recurse, negative assertion, or positive assertion in conditional block. */
  BOOL local_quit_available;
  /* Currently in a positive assertion. */
  BOOL in_positive_assertion;
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
  struct sljit_label *abort_label;
  struct sljit_label *accept_label;
  struct sljit_label *ff_newline_shortcut;
  stub_list *stubs;
  recurse_entry *entries;
  recurse_entry *currententry;
  jump_list *partialmatch;
  jump_list *quit;
  jump_list *positive_assertion_quit;
  jump_list *abort;
  jump_list *failed_match;
  jump_list *accept;
  jump_list *calllimit;
  jump_list *stackalloc;
  jump_list *revertframes;
  jump_list *wordboundary;
  jump_list *ucp_wordboundary;
  jump_list *anynewline;
  jump_list *hspace;
  jump_list *vspace;
  jump_list *casefulcmp;
  jump_list *caselesscmp;
  jump_list *reset_match;
  /* Same as reset_match, but resets the STR_PTR as well. */
  jump_list *restart_match;
  BOOL unset_backref;
  BOOL alt_circumflex;
#ifdef SUPPORT_UNICODE
  BOOL utf;
  BOOL invalid_utf;
  BOOL ucp;
  /* Points to saving area for iref. */
  jump_list *getucd;
  jump_list *getucdtype;
#if PCRE2_CODE_UNIT_WIDTH == 8
  jump_list *utfreadchar;
  jump_list *utfreadtype8;
  jump_list *utfpeakcharback;
#endif
#if PCRE2_CODE_UNIT_WIDTH == 8 || PCRE2_CODE_UNIT_WIDTH == 16
  jump_list *utfreadchar_invalid;
  jump_list *utfreadnewline_invalid;
  jump_list *utfmoveback_invalid;
  jump_list *utfpeakcharback_invalid;
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
#define STACK(i)      ((i) * SSIZE_OF(sw))

#ifdef SLJIT_PREF_SHIFT_REG
#if SLJIT_PREF_SHIFT_REG == SLJIT_R2
/* Nothing. */
#elif SLJIT_PREF_SHIFT_REG == SLJIT_R3
#define SHIFT_REG_IS_R3
#else
#error "Unsupported shift register"
#endif
#endif

#define TMP1          SLJIT_R0
#ifdef SHIFT_REG_IS_R3
#define TMP2          SLJIT_R3
#define TMP3          SLJIT_R2
#else
#define TMP2          SLJIT_R2
#define TMP3          SLJIT_R3
#endif
#define STR_PTR       SLJIT_R1
#define STR_END       SLJIT_S0
#define STACK_TOP     SLJIT_S1
#define STACK_LIMIT   SLJIT_S2
#define COUNT_MATCH   SLJIT_S3
#define ARGUMENTS     SLJIT_S4
#define RETURN_ADDR   SLJIT_R4

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
#define HAS_VIRTUAL_REGISTERS 1
#else
#define HAS_VIRTUAL_REGISTERS 0
#endif

/* Local space layout. */
/* Max limit of recursions. */
#define LIMIT_MATCH      (0 * sizeof(sljit_sw))
/* Local variables. Their number is computed by check_opcode_types. */
#define LOCAL0           (1 * sizeof(sljit_sw))
#define LOCAL1           (2 * sizeof(sljit_sw))
#define LOCAL2           (3 * sizeof(sljit_sw))
#define LOCAL3           (4 * sizeof(sljit_sw))
#define LOCAL4           (5 * sizeof(sljit_sw))
/* The output vector is stored on the stack, and contains pointers
to characters. The vector data is divided into two groups: the first
group contains the start / end character pointers, and the second is
the start pointers when the end of the capturing group has not yet reached. */
#define OVECTOR_START    (common->ovector_start)
#define OVECTOR(i)       (OVECTOR_START + (i) * SSIZE_OF(sw))
#define OVECTOR_PRIV(i)  (common->cbra_ptr + (i) * SSIZE_OF(sw))
#define PRIVATE_DATA(cc) (common->private_data_ptrs[(cc) - common->start])

#if PCRE2_CODE_UNIT_WIDTH == 8
#define MOV_UCHAR  SLJIT_MOV_U8
#define IN_UCHARS(x) (x)
#elif PCRE2_CODE_UNIT_WIDTH == 16
#define MOV_UCHAR  SLJIT_MOV_U16
#define UCHAR_SHIFT (1)
#define IN_UCHARS(x) ((x) * 2)
#elif PCRE2_CODE_UNIT_WIDTH == 32
#define MOV_UCHAR  SLJIT_MOV_U32
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
#define OP2U(op, src1, src1w, src2, src2w) \
  sljit_emit_op2u(compiler, (op), (src1), (src1w), (src2), (src2w))
#define OP_SRC(op, src, srcw) \
  sljit_emit_op_src(compiler, (op), (src), (srcw))
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
#define OP_FLAGS(op, dst, dstw, type) \
  sljit_emit_op_flags(compiler, (op), (dst), (dstw), (type))
#define SELECT(type, dst_reg, src1, src1w, src2_reg) \
  sljit_emit_select(compiler, (type), (dst_reg), (src1), (src1w), (src2_reg))
#define GET_LOCAL_BASE(dst, dstw, offset) \
  sljit_get_local_base(compiler, (dst), (dstw), (offset))

#define READ_CHAR_MAX ((sljit_u32)0xffffffff)

#define INVALID_UTF_CHAR -1
#define UNASSIGNED_UTF_CHAR 888

#if defined SUPPORT_UNICODE
#if PCRE2_CODE_UNIT_WIDTH == 8

#define GETCHARINC_INVALID(c, ptr, end, invalid_action) \
  { \
  if (ptr[0] <= 0x7f) \
    c = *ptr++; \
  else if (ptr + 1 < end && ptr[1] >= 0x80 && ptr[1] < 0xc0) \
    { \
    c = ptr[1] - 0x80; \
    \
    if (ptr[0] >= 0xc2 && ptr[0] <= 0xdf) \
      { \
      c |= (ptr[0] - 0xc0) << 6; \
      ptr += 2; \
      } \
    else if (ptr + 2 < end && ptr[2] >= 0x80 && ptr[2] < 0xc0) \
      { \
      c = c << 6 | (ptr[2] - 0x80); \
      \
      if (ptr[0] >= 0xe0 && ptr[0] <= 0xef) \
        { \
        c |= (ptr[0] - 0xe0) << 12; \
        ptr += 3; \
        \
        if (c < 0x800 || (c >= 0xd800 && c < 0xe000)) \
          { \
          invalid_action; \
          } \
        } \
      else if (ptr + 3 < end && ptr[3] >= 0x80 && ptr[3] < 0xc0) \
        { \
        c = c << 6 | (ptr[3] - 0x80); \
        \
        if (ptr[0] >= 0xf0 && ptr[0] <= 0xf4) \
          { \
          c |= (ptr[0] - 0xf0) << 18; \
          ptr += 4; \
          \
          if (c >= 0x110000 || c < 0x10000) \
            { \
            invalid_action; \
            } \
          } \
        else \
          { \
          invalid_action; \
          } \
        } \
      else \
        { \
        invalid_action; \
        } \
      } \
    else \
      { \
      invalid_action; \
      } \
    } \
  else \
    { \
    invalid_action; \
    } \
  }

#define GETCHARBACK_INVALID(c, ptr, start, invalid_action) \
  { \
  c = ptr[-1]; \
  if (c <= 0x7f) \
    ptr--; \
  else if (ptr - 1 > start && ptr[-1] >= 0x80 && ptr[-1] < 0xc0) \
    { \
    c -= 0x80; \
    \
    if (ptr[-2] >= 0xc2 && ptr[-2] <= 0xdf) \
      { \
      c |= (ptr[-2] - 0xc0) << 6; \
      ptr -= 2; \
      } \
    else if (ptr - 2 > start && ptr[-2] >= 0x80 && ptr[-2] < 0xc0) \
      { \
      c = c << 6 | (ptr[-2] - 0x80); \
      \
      if (ptr[-3] >= 0xe0 && ptr[-3] <= 0xef) \
        { \
        c |= (ptr[-3] - 0xe0) << 12; \
        ptr -= 3; \
        \
        if (c < 0x800 || (c >= 0xd800 && c < 0xe000)) \
          { \
          invalid_action; \
          } \
        } \
      else if (ptr - 3 > start && ptr[-3] >= 0x80 && ptr[-3] < 0xc0) \
        { \
        c = c << 6 | (ptr[-3] - 0x80); \
        \
        if (ptr[-4] >= 0xf0 && ptr[-4] <= 0xf4) \
          { \
          c |= (ptr[-4] - 0xf0) << 18; \
          ptr -= 4; \
          \
          if (c >= 0x110000 || c < 0x10000) \
            { \
            invalid_action; \
            } \
          } \
        else \
          { \
          invalid_action; \
          } \
        } \
      else \
        { \
        invalid_action; \
        } \
      } \
    else \
      { \
      invalid_action; \
      } \
    } \
  else \
    { \
    invalid_action; \
    } \
  }

#elif PCRE2_CODE_UNIT_WIDTH == 16

#define GETCHARINC_INVALID(c, ptr, end, invalid_action) \
  { \
  if (ptr[0] < 0xd800 || ptr[0] >= 0xe000) \
    c = *ptr++; \
  else if (ptr[0] < 0xdc00 && ptr + 1 < end && ptr[1] >= 0xdc00 && ptr[1] < 0xe000) \
    { \
    c = (((ptr[0] - 0xd800) << 10) | (ptr[1] - 0xdc00)) + 0x10000; \
    ptr += 2; \
    } \
  else \
    { \
    invalid_action; \
    } \
  }

#define GETCHARBACK_INVALID(c, ptr, start, invalid_action) \
  { \
  c = ptr[-1]; \
  if (c < 0xd800 || c >= 0xe000) \
    ptr--; \
  else if (c >= 0xdc00 && ptr - 1 > start && ptr[-2] >= 0xd800 && ptr[-2] < 0xdc00) \
    { \
    c = (((ptr[-2] - 0xd800) << 10) | (c - 0xdc00)) + 0x10000; \
    ptr -= 2; \
    } \
  else \
    { \
    invalid_action; \
    } \
  }


#elif PCRE2_CODE_UNIT_WIDTH == 32

#define GETCHARINC_INVALID(c, ptr, end, invalid_action) \
  { \
  if (ptr[0] < 0xd800 || (ptr[0] >= 0xe000 && ptr[0] < 0x110000)) \
    c = *ptr++; \
  else \
    { \
    invalid_action; \
    } \
  }

#define GETCHARBACK_INVALID(c, ptr, start, invalid_action) \
  { \
  c = ptr[-1]; \
  if (ptr[-1] < 0xd800 || (ptr[-1] >= 0xe000 && ptr[-1] < 0x110000)) \
    ptr--; \
  else \
    { \
    invalid_action; \
    } \
  }

#endif /* PCRE2_CODE_UNIT_WIDTH == [8|16|32] */
#endif /* SUPPORT_UNICODE */

static PCRE2_SPTR bracketend(PCRE2_SPTR cc)
{
SLJIT_ASSERT((*cc >= OP_ASSERT && *cc <= OP_ASSERT_SCS) || (*cc >= OP_ONCE && *cc <= OP_SCOND));
do cc += GET(cc, 1); while (*cc == OP_ALT);
SLJIT_ASSERT(*cc >= OP_KET && *cc <= OP_KETRPOS);
cc += 1 + LINK_SIZE;
return cc;
}

static int no_alternatives(PCRE2_SPTR cc)
{
int count = 0;
SLJIT_ASSERT((*cc >= OP_ASSERT && *cc <= OP_ASSERT_SCS) || (*cc >= OP_ONCE && *cc <= OP_SCOND));
do
  {
  cc += GET(cc, 1);
  count++;
  }
while (*cc == OP_ALT);
SLJIT_ASSERT(*cc >= OP_KET && *cc <= OP_KETRPOS);
return count;
}

static BOOL find_vreverse(PCRE2_SPTR cc)
{
  SLJIT_ASSERT(*cc == OP_ASSERTBACK || *cc == OP_ASSERTBACK_NOT ||  *cc == OP_ASSERTBACK_NA);

  do
    {
    if (cc[1 + LINK_SIZE] == OP_VREVERSE)
      return TRUE;
    cc += GET(cc, 1);
    }
  while (*cc == OP_ALT);

  return FALSE;
}

/* Functions whose might need modification for all new supported opcodes:
 next_opcode
 check_opcode_types
 set_private_data_ptrs
 get_framesize
 init_frame
 get_recurse_data_length
 copy_recurse_data
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
  case OP_VREVERSE:
  case OP_ASSERT:
  case OP_ASSERT_NOT:
  case OP_ASSERTBACK:
  case OP_ASSERTBACK_NOT:
  case OP_ASSERT_NA:
  case OP_ASSERTBACK_NA:
  case OP_ASSERT_SCS:
  case OP_ONCE:
  case OP_SCRIPT_RUN:
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
  case OP_NOT_UCP_WORD_BOUNDARY:
  case OP_UCP_WORD_BOUNDARY:
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
  case OP_ECLASS:
  case OP_XCLASS:
  SLJIT_COMPILE_ASSERT(OP_XCLASS + 1 == OP_ECLASS && OP_CLASS + 1 == OP_NCLASS && OP_NCLASS < OP_XCLASS, class_byte_code_order);
  return cc + GET(cc, 1);
#endif

  case OP_MARK:
  case OP_COMMIT_ARG:
  case OP_PRUNE_ARG:
  case OP_SKIP_ARG:
  case OP_THEN_ARG:
  return cc + 1 + 2 + cc[1];

  default:
  SLJIT_UNREACHABLE();
  return NULL;
  }
}

static sljit_s32 ref_update_local_size(compiler_common *common, PCRE2_SPTR cc, sljit_s32 current_locals_size)
{
/* Depends on do_casefulcmp(), do_caselesscmp(), and compile_ref_matchingpath() */
int locals_size = 2 * SSIZE_OF(sw);
SLJIT_UNUSED_ARG(common);

#ifdef SUPPORT_UNICODE
if ((*cc == OP_REFI || *cc == OP_DNREFI) && (common->utf || common->ucp))
  locals_size = 3 * SSIZE_OF(sw);
#endif

cc += PRIV(OP_lengths)[*cc];
/* Although do_casefulcmp() uses only one local, the allocate_stack()
calls during the repeat destroys LOCAL1 variables. */
if (*cc >= OP_CRSTAR && *cc <= OP_CRPOSRANGE)
  locals_size += 2 * SSIZE_OF(sw);

return (current_locals_size >= locals_size) ? current_locals_size : locals_size;
}

static BOOL check_opcode_types(compiler_common *common, PCRE2_SPTR cc, PCRE2_SPTR ccend)
{
int count;
PCRE2_SPTR slot;
PCRE2_SPTR assert_back_end = cc - 1;
PCRE2_SPTR assert_na_end = cc - 1;
sljit_s32 locals_size = 2 * SSIZE_OF(sw);
BOOL set_recursive_head = FALSE;
BOOL set_capture_last = FALSE;
BOOL set_mark = FALSE;

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

    case OP_TYPEUPTO:
    case OP_TYPEEXACT:
    if (cc[1 + IMM2_SIZE] == OP_EXTUNI && locals_size <= 3 * SSIZE_OF(sw))
      locals_size = 3 * SSIZE_OF(sw);
    cc += (2 + IMM2_SIZE) - 1;
    break;

    case OP_TYPEPOSSTAR:
    case OP_TYPEPOSPLUS:
    case OP_TYPEPOSQUERY:
    if (cc[1] == OP_EXTUNI && locals_size <= 3 * SSIZE_OF(sw))
      locals_size = 3 * SSIZE_OF(sw);
    cc += 2 - 1;
    break;

    case OP_TYPEPOSUPTO:
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
    if (common->utf && locals_size <= 3 * SSIZE_OF(sw))
      locals_size = 3 * SSIZE_OF(sw);
#endif
    if (cc[1 + IMM2_SIZE] == OP_EXTUNI && locals_size <= 3 * SSIZE_OF(sw))
      locals_size = 3 * SSIZE_OF(sw);
    cc += (2 + IMM2_SIZE) - 1;
    break;

    case OP_REFI:
    case OP_REF:
    locals_size = ref_update_local_size(common, cc, locals_size);
    common->optimized_cbracket[GET2(cc, 1)] = 0;
    cc += PRIV(OP_lengths)[*cc];
    break;

    case OP_ASSERT_NA:
    case OP_ASSERTBACK_NA:
    case OP_ASSERT_SCS:
    slot = bracketend(cc);
    if (slot > assert_na_end)
      assert_na_end = slot;
    cc += 1 + LINK_SIZE;
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

    case OP_DNREFI:
    case OP_DNREF:
    locals_size = ref_update_local_size(common, cc, locals_size);
    /* Fall through */
    case OP_DNCREF:
    count = GET2(cc, 1 + IMM2_SIZE);
    slot = common->name_table + GET2(cc, 1) * common->name_entry_size;
    while (count-- > 0)
      {
      common->optimized_cbracket[GET2(slot, 0)] = 0;
      slot += common->name_entry_size;
      }
    cc += PRIV(OP_lengths)[*cc];
    break;

    case OP_RECURSE:
    /* Set its value only once. */
    set_recursive_head = TRUE;
    cc += 1 + LINK_SIZE;
    break;

    case OP_CALLOUT:
    case OP_CALLOUT_STR:
    set_capture_last = TRUE;
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

    case OP_COMMIT_ARG:
    case OP_PRUNE_ARG:
    case OP_MARK:
    set_mark = TRUE;
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

    case OP_ASSERT_ACCEPT:
    if (cc < assert_na_end)
      return FALSE;
    cc++;
    break;

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
    case OP_CRPOSRANGE:
    /* The second value can be 0 for infinite repeats. */
    if (common->utf && GET2(cc, 1) != GET2(cc, 1 + IMM2_SIZE) && locals_size <= 3 * SSIZE_OF(sw))
      locals_size = 3 * SSIZE_OF(sw);
    cc += 1 + 2 * IMM2_SIZE;
    break;

    case OP_POSUPTO:
    case OP_POSUPTOI:
    case OP_NOTPOSUPTO:
    case OP_NOTPOSUPTOI:
    if (common->utf && locals_size <= 3 * SSIZE_OF(sw))
      locals_size = 3 * SSIZE_OF(sw);
#endif
    /* Fall through */
    default:
    cc = next_opcode(common, cc);
    if (cc == NULL)
      return FALSE;
    break;
    }
  }

SLJIT_ASSERT((locals_size & (SSIZE_OF(sw) - 1)) == 0);
#if defined SLJIT_DEBUG && SLJIT_DEBUG
common->locals_size = locals_size;
#endif

if (locals_size > 0)
  common->ovector_start += locals_size;

if (set_mark)
  {
  SLJIT_ASSERT(common->mark_ptr == 0);
  common->mark_ptr = common->ovector_start;
  common->ovector_start += sizeof(sljit_sw);
  }

if (set_recursive_head)
  {
  SLJIT_ASSERT(common->recursive_head_ptr == 0);
  common->recursive_head_ptr = common->ovector_start;
  common->ovector_start += sizeof(sljit_sw);
  }

if (set_capture_last)
  {
  SLJIT_ASSERT(common->capture_last_ptr == 0);
  common->capture_last_ptr = common->ovector_start;
  common->ovector_start += sizeof(sljit_sw);
  }

return TRUE;
}

#define EARLY_FAIL_ENHANCE_MAX (3 + 3)

/*
  Start represent the number of allowed early fail enhancements

  The 0-2 values has a special meaning:
    0 - skip is allowed for all iterators
    1 - fail is allowed for all iterators
    2 - fail is allowed for greedy iterators
    3 - only ranged early fail is allowed
  >3 - (start - 3) number of remaining ranged early fails allowed

return: the updated value of start
*/
static int detect_early_fail(compiler_common *common, PCRE2_SPTR cc,
   int *private_data_start, sljit_s32 depth, int start)
{
PCRE2_SPTR begin = cc;
PCRE2_SPTR next_alt;
PCRE2_SPTR end;
PCRE2_SPTR accelerated_start;
int result = 0;
int count, prev_count;

SLJIT_ASSERT(*cc == OP_ONCE || *cc == OP_BRA || *cc == OP_CBRA);
SLJIT_ASSERT(*cc != OP_CBRA || common->optimized_cbracket[GET2(cc, 1 + LINK_SIZE)] != 0);
SLJIT_ASSERT(start < EARLY_FAIL_ENHANCE_MAX);

next_alt = cc + GET(cc, 1);
if (*next_alt == OP_ALT && start < 1)
  start = 1;

do
  {
  count = start;
  cc += 1 + LINK_SIZE + ((*cc == OP_CBRA) ? IMM2_SIZE : 0);

  while (TRUE)
    {
    accelerated_start = NULL;

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
      case OP_NOT_UCP_WORD_BOUNDARY:
      case OP_UCP_WORD_BOUNDARY:
      /* Zero width assertions. */
      cc++;
      continue;

      case OP_NOT_DIGIT:
      case OP_DIGIT:
      case OP_NOT_WHITESPACE:
      case OP_WHITESPACE:
      case OP_NOT_WORDCHAR:
      case OP_WORDCHAR:
      case OP_ANY:
      case OP_ALLANY:
      case OP_ANYBYTE:
      case OP_NOT_HSPACE:
      case OP_HSPACE:
      case OP_NOT_VSPACE:
      case OP_VSPACE:
      if (count < 1)
        count = 1;
      cc++;
      continue;

      case OP_ANYNL:
      case OP_EXTUNI:
      if (count < 3)
        count = 3;
      cc++;
      continue;

      case OP_NOTPROP:
      case OP_PROP:
      if (count < 1)
        count = 1;
      cc += 1 + 2;
      continue;

      case OP_CHAR:
      case OP_CHARI:
      case OP_NOT:
      case OP_NOTI:
      if (count < 1)
        count = 1;
      cc += 2;
#ifdef SUPPORT_UNICODE
      if (common->utf && HAS_EXTRALEN(cc[-1])) cc += GET_EXTRALEN(cc[-1]);
#endif
      continue;

      case OP_TYPEMINSTAR:
      case OP_TYPEMINPLUS:
      if (count == 2)
        count = 3;
      /* Fall through */

      case OP_TYPESTAR:
      case OP_TYPEPLUS:
      case OP_TYPEPOSSTAR:
      case OP_TYPEPOSPLUS:
      /* The type or prop opcode is skipped in the next iteration. */
      cc += 1;

      if (cc[0] != OP_ANYNL && cc[0] != OP_EXTUNI)
        {
        accelerated_start = cc - 1;
        break;
        }

      if (count < 3)
        count = 3;
      continue;

      case OP_TYPEEXACT:
      if (count < 1)
        count = 1;
      cc += 1 + IMM2_SIZE;
      continue;

      case OP_TYPEUPTO:
      case OP_TYPEMINUPTO:
      case OP_TYPEPOSUPTO:
      cc += IMM2_SIZE;
      /* Fall through */

      case OP_TYPEQUERY:
      case OP_TYPEMINQUERY:
      case OP_TYPEPOSQUERY:
      /* The type or prop opcode is skipped in the next iteration. */
      if (count < 3)
        count = 3;
      cc += 1;
      continue;

      case OP_MINSTAR:
      case OP_MINPLUS:
      case OP_MINSTARI:
      case OP_MINPLUSI:
      case OP_NOTMINSTAR:
      case OP_NOTMINPLUS:
      case OP_NOTMINSTARI:
      case OP_NOTMINPLUSI:
      if (count == 2)
        count = 3;
      /* Fall through */

      case OP_STAR:
      case OP_PLUS:
      case OP_POSSTAR:
      case OP_POSPLUS:

      case OP_STARI:
      case OP_PLUSI:
      case OP_POSSTARI:
      case OP_POSPLUSI:

      case OP_NOTSTAR:
      case OP_NOTPLUS:
      case OP_NOTPOSSTAR:
      case OP_NOTPOSPLUS:

      case OP_NOTSTARI:
      case OP_NOTPLUSI:
      case OP_NOTPOSSTARI:
      case OP_NOTPOSPLUSI:
      accelerated_start = cc;
      cc += 2;
#ifdef SUPPORT_UNICODE
      if (common->utf && HAS_EXTRALEN(cc[-1])) cc += GET_EXTRALEN(cc[-1]);
#endif
      break;

      case OP_EXACT:
      if (count < 1)
        count = 1;
      cc += 2 + IMM2_SIZE;
#ifdef SUPPORT_UNICODE
      if (common->utf && HAS_EXTRALEN(cc[-1])) cc += GET_EXTRALEN(cc[-1]);
#endif
      continue;

      case OP_UPTO:
      case OP_MINUPTO:
      case OP_POSUPTO:
      case OP_UPTOI:
      case OP_MINUPTOI:
      case OP_EXACTI:
      case OP_POSUPTOI:
      case OP_NOTUPTO:
      case OP_NOTMINUPTO:
      case OP_NOTEXACT:
      case OP_NOTPOSUPTO:
      case OP_NOTUPTOI:
      case OP_NOTMINUPTOI:
      case OP_NOTEXACTI:
      case OP_NOTPOSUPTOI:
      cc += IMM2_SIZE;
      /* Fall through */

      case OP_QUERY:
      case OP_MINQUERY:
      case OP_POSQUERY:
      case OP_QUERYI:
      case OP_MINQUERYI:
      case OP_POSQUERYI:
      case OP_NOTQUERY:
      case OP_NOTMINQUERY:
      case OP_NOTPOSQUERY:
      case OP_NOTQUERYI:
      case OP_NOTMINQUERYI:
      case OP_NOTPOSQUERYI:
      if (count < 3)
        count = 3;
      cc += 2;
#ifdef SUPPORT_UNICODE
      if (common->utf && HAS_EXTRALEN(cc[-1])) cc += GET_EXTRALEN(cc[-1]);
#endif
      continue;

      case OP_CLASS:
      case OP_NCLASS:
#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH != 8
      case OP_XCLASS:
      case OP_ECLASS:
      accelerated_start = cc;
      cc += (*cc >= OP_XCLASS) ? GET(cc, 1) : (unsigned int)(1 + (32 / sizeof(PCRE2_UCHAR)));
#else
      accelerated_start = cc;
      cc += (1 + (32 / sizeof(PCRE2_UCHAR)));
#endif

      switch (*cc)
        {
        case OP_CRMINSTAR:
        case OP_CRMINPLUS:
        if (count == 2)
          count = 3;
        /* Fall through */

        case OP_CRSTAR:
        case OP_CRPLUS:
        case OP_CRPOSSTAR:
        case OP_CRPOSPLUS:
        cc++;
        break;

        case OP_CRRANGE:
        case OP_CRMINRANGE:
        case OP_CRPOSRANGE:
        if (GET2(cc, 1) == GET2(cc, 1 + IMM2_SIZE))
          {
          /* Exact repeat. */
          cc += 1 + 2 * IMM2_SIZE;
          if (count < 1)
            count = 1;
          continue;
          }

        cc += 2 * IMM2_SIZE;
        /* Fall through */
        case OP_CRQUERY:
        case OP_CRMINQUERY:
        case OP_CRPOSQUERY:
        cc++;
        if (count < 3)
          count = 3;
        continue;

        default:
        /* No repeat. */
        if (count < 1)
          count = 1;
        continue;
        }
      break;

      case OP_BRA:
      case OP_CBRA:
      prev_count = count;
      if (count < 1)
        count = 1;

      if (depth >= 4)
        break;

      if (count < 3 && cc[GET(cc, 1)] == OP_ALT)
        count = 3;

      end = bracketend(cc);
      if (end[-1 - LINK_SIZE] != OP_KET || (*cc == OP_CBRA && common->optimized_cbracket[GET2(cc, 1 + LINK_SIZE)] == 0))
        break;

      prev_count = detect_early_fail(common, cc, private_data_start, depth + 1, prev_count);

      if (prev_count > count)
        count = prev_count;

      if (PRIVATE_DATA(cc) != 0)
        common->private_data_ptrs[begin - common->start] = 1;

      if (count < EARLY_FAIL_ENHANCE_MAX)
        {
        cc = end;
        continue;
        }
      break;

      case OP_KET:
      SLJIT_ASSERT(PRIVATE_DATA(cc) == 0);
      if (cc >= next_alt)
        break;
      cc += 1 + LINK_SIZE;
      continue;
      }

    if (accelerated_start == NULL)
      break;

    if (count == 0)
      {
      common->fast_forward_bc_ptr = accelerated_start;
      common->private_data_ptrs[(accelerated_start + 1) - common->start] = ((*private_data_start) << 3) | type_skip;
      *private_data_start += sizeof(sljit_sw);
      count = 4;
      }
    else if (count < 3)
      {
      common->private_data_ptrs[(accelerated_start + 1) - common->start] = ((*private_data_start) << 3) | type_fail;

      if (common->early_fail_start_ptr == 0)
        common->early_fail_start_ptr = *private_data_start;

      *private_data_start += sizeof(sljit_sw);
      common->early_fail_end_ptr = *private_data_start;

      if (*private_data_start > SLJIT_MAX_LOCAL_SIZE)
        return EARLY_FAIL_ENHANCE_MAX;

      count = 4;
      }
    else
      {
      common->private_data_ptrs[(accelerated_start + 1) - common->start] = ((*private_data_start) << 3) | type_fail_range;

      if (common->early_fail_start_ptr == 0)
        common->early_fail_start_ptr = *private_data_start;

      *private_data_start += 2 * sizeof(sljit_sw);
      common->early_fail_end_ptr = *private_data_start;

      if (*private_data_start > SLJIT_MAX_LOCAL_SIZE)
        return EARLY_FAIL_ENHANCE_MAX;

      count++;
      }

    /* Cannot be part of a repeat. */
    common->private_data_ptrs[begin - common->start] = 1;

    if (count >= EARLY_FAIL_ENHANCE_MAX)
      break;
    }

  if (*cc != OP_ALT && *cc != OP_KET)
    result = EARLY_FAIL_ENHANCE_MAX;
  else if (result < count)
    result = count;

  cc = next_alt;
  next_alt = cc + GET(cc, 1);
  }
while (*cc == OP_ALT);

return result;
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
  if (max > (sljit_u32)(*cc == OP_CRRANGE ? 0 : 1))
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
if (end[-(1 + LINK_SIZE)] != OP_KET || PRIVATE_DATA(begin) != 0)
  return FALSE;

/* /(?:AB){4,6}/ is currently converted to /(?:AB){3}(?AB){1,3}/
 * Skip the check of the second part. */
if (PRIVATE_DATA(end - LINK_SIZE) != 0)
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

  /* When the bracket is prefixed by a zero iteration, skip the repeat check (at this point). */
  if (repeat_check && (*cc == OP_ONCE || *cc == OP_BRA || *cc == OP_CBRA || *cc == OP_COND))
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
    case OP_ASSERT_NA:
    case OP_ONCE:
    case OP_SCRIPT_RUN:
    case OP_BRAPOS:
    case OP_SBRA:
    case OP_SBRAPOS:
    case OP_SCOND:
    common->private_data_ptrs[cc - common->start] = private_data_ptr;
    private_data_ptr += sizeof(sljit_sw);
    bracketlen = 1 + LINK_SIZE;
    break;

    case OP_ASSERTBACK_NA:
    common->private_data_ptrs[cc - common->start] = private_data_ptr;
    private_data_ptr += sizeof(sljit_sw);

    if (find_vreverse(cc))
      {
      common->private_data_ptrs[cc + 1 - common->start] = 1;
      private_data_ptr += sizeof(sljit_sw);
      }

    bracketlen = 1 + LINK_SIZE;
    break;

    case OP_ASSERT_SCS:
    common->private_data_ptrs[cc - common->start] = private_data_ptr;
    private_data_ptr += 2 * sizeof(sljit_sw);
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
    common->private_data_ptrs[cc - common->start] = 0;
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
    size = 1;
    repeat_check = FALSE;
    break;

    CASE_ITERATOR_PRIVATE_DATA_1
    size = -2;
    space = 1;
    break;

    CASE_ITERATOR_PRIVATE_DATA_2A
    size = -2;
    space = 2;
    break;

    CASE_ITERATOR_PRIVATE_DATA_2B
    size = -(2 + IMM2_SIZE);
    space = 2;
    break;

    CASE_ITERATOR_TYPE_PRIVATE_DATA_1
    size = 1;
    space = 1;
    break;

    CASE_ITERATOR_TYPE_PRIVATE_DATA_2A
    size = 1;
    if (cc[1] != OP_EXTUNI)
      space = 2;
    break;

    case OP_TYPEUPTO:
    size = 1 + IMM2_SIZE;
    if (cc[1 + IMM2_SIZE] != OP_EXTUNI)
      space = 2;
    break;

    case OP_TYPEMINUPTO:
    size = 1 + IMM2_SIZE;
    space = 2;
    break;

    case OP_CLASS:
    case OP_NCLASS:
    size = 1 + 32 / sizeof(PCRE2_UCHAR);
    space = get_class_iterator_size(cc + size);
    break;

#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH != 8
    case OP_XCLASS:
    case OP_ECLASS:
    size = GET(cc, 1);
    space = get_class_iterator_size(cc + size);
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
    case OP_COMMIT_ARG:
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
    case OP_ECLASS:

    case OP_CALLOUT:
    case OP_CALLOUT_STR:

    case OP_NOT_UCP_WORD_BOUNDARY:
    case OP_UCP_WORD_BOUNDARY:

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

static void init_frame(compiler_common *common, PCRE2_SPTR cc, PCRE2_SPTR ccend, int stackpos, int stacktop)
{
DEFINE_COMPILER;
BOOL setsom_found = FALSE;
BOOL setmark_found = FALSE;
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
  if (*cc != OP_CBRAPOS && *cc != OP_SCBRAPOS)
    cc = next_opcode(common, cc);
  }

/* The data is restored by do_revertframes(). */
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
      stackpos -= SSIZE_OF(sw);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, TMP1, 0);
      stackpos -= SSIZE_OF(sw);
      setsom_found = TRUE;
      }
    cc += 1;
    break;

    case OP_MARK:
    case OP_COMMIT_ARG:
    case OP_PRUNE_ARG:
    case OP_THEN_ARG:
    SLJIT_ASSERT(common->mark_ptr != 0);
    if (!setmark_found)
      {
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->mark_ptr);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, SLJIT_IMM, -common->mark_ptr);
      stackpos -= SSIZE_OF(sw);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, TMP1, 0);
      stackpos -= SSIZE_OF(sw);
      setmark_found = TRUE;
      }
    cc += 1 + 2 + cc[1];
    break;

    case OP_RECURSE:
    if (common->has_set_som && !setsom_found)
      {
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(0));
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, SLJIT_IMM, -OVECTOR(0));
      stackpos -= SSIZE_OF(sw);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, TMP1, 0);
      stackpos -= SSIZE_OF(sw);
      setsom_found = TRUE;
      }
    if (common->mark_ptr != 0 && !setmark_found)
      {
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->mark_ptr);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, SLJIT_IMM, -common->mark_ptr);
      stackpos -= SSIZE_OF(sw);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, TMP1, 0);
      stackpos -= SSIZE_OF(sw);
      setmark_found = TRUE;
      }
    if (common->capture_last_ptr != 0 && !capture_last_found)
      {
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->capture_last_ptr);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, SLJIT_IMM, -common->capture_last_ptr);
      stackpos -= SSIZE_OF(sw);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, TMP1, 0);
      stackpos -= SSIZE_OF(sw);
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
      stackpos -= SSIZE_OF(sw);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, TMP1, 0);
      stackpos -= SSIZE_OF(sw);
      capture_last_found = TRUE;
      }
    offset = (GET2(cc, 1 + LINK_SIZE)) << 1;
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, SLJIT_IMM, OVECTOR(offset));
    stackpos -= SSIZE_OF(sw);
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset));
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1));
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, TMP1, 0);
    stackpos -= SSIZE_OF(sw);
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), stackpos, TMP2, 0);
    stackpos -= SSIZE_OF(sw);

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

#define RECURSE_TMP_REG_COUNT 3

typedef struct delayed_mem_copy_status {
  struct sljit_compiler *compiler;
  int store_bases[RECURSE_TMP_REG_COUNT];
  int store_offsets[RECURSE_TMP_REG_COUNT];
  int tmp_regs[RECURSE_TMP_REG_COUNT];
  int saved_tmp_regs[RECURSE_TMP_REG_COUNT];
  int next_tmp_reg;
} delayed_mem_copy_status;

static void delayed_mem_copy_init(delayed_mem_copy_status *status, compiler_common *common)
{
int i;

for (i = 0; i < RECURSE_TMP_REG_COUNT; i++)
  {
  SLJIT_ASSERT(status->tmp_regs[i] >= 0);
  SLJIT_ASSERT(sljit_get_register_index(SLJIT_GP_REGISTER, status->saved_tmp_regs[i]) < 0 || status->tmp_regs[i] == status->saved_tmp_regs[i]);

  status->store_bases[i] = -1;
  }
status->next_tmp_reg = 0;
status->compiler = common->compiler;
}

static void delayed_mem_copy_move(delayed_mem_copy_status *status, int load_base, sljit_sw load_offset,
  int store_base, sljit_sw store_offset)
{
struct sljit_compiler *compiler = status->compiler;
int next_tmp_reg = status->next_tmp_reg;
int tmp_reg = status->tmp_regs[next_tmp_reg];

SLJIT_ASSERT(load_base > 0 && store_base > 0);

if (status->store_bases[next_tmp_reg] == -1)
  {
  /* Preserve virtual registers. */
  if (sljit_get_register_index(SLJIT_GP_REGISTER, status->saved_tmp_regs[next_tmp_reg]) < 0)
    OP1(SLJIT_MOV, status->saved_tmp_regs[next_tmp_reg], 0, tmp_reg, 0);
  }
else
  OP1(SLJIT_MOV, SLJIT_MEM1(status->store_bases[next_tmp_reg]), status->store_offsets[next_tmp_reg], tmp_reg, 0);

OP1(SLJIT_MOV, tmp_reg, 0, SLJIT_MEM1(load_base), load_offset);
status->store_bases[next_tmp_reg] = store_base;
status->store_offsets[next_tmp_reg] = store_offset;

status->next_tmp_reg = (next_tmp_reg + 1) % RECURSE_TMP_REG_COUNT;
}

static void delayed_mem_copy_finish(delayed_mem_copy_status *status)
{
struct sljit_compiler *compiler = status->compiler;
int next_tmp_reg = status->next_tmp_reg;
int tmp_reg, saved_tmp_reg, i;

for (i = 0; i < RECURSE_TMP_REG_COUNT; i++)
  {
  if (status->store_bases[next_tmp_reg] != -1)
    {
    tmp_reg = status->tmp_regs[next_tmp_reg];
    saved_tmp_reg = status->saved_tmp_regs[next_tmp_reg];

    OP1(SLJIT_MOV, SLJIT_MEM1(status->store_bases[next_tmp_reg]), status->store_offsets[next_tmp_reg], tmp_reg, 0);

    /* Restore virtual registers. */
    if (sljit_get_register_index(SLJIT_GP_REGISTER, saved_tmp_reg) < 0)
      OP1(SLJIT_MOV, tmp_reg, 0, saved_tmp_reg, 0);
    }

  next_tmp_reg = (next_tmp_reg + 1) % RECURSE_TMP_REG_COUNT;
  }
}

#undef RECURSE_TMP_REG_COUNT

static BOOL recurse_check_bit(compiler_common *common, sljit_sw bit_index)
{
uint8_t *byte;
uint8_t mask;

SLJIT_ASSERT((bit_index & (sizeof(sljit_sw) - 1)) == 0);

bit_index >>= SLJIT_WORD_SHIFT;

SLJIT_ASSERT((bit_index >> 3) < common->recurse_bitset_size);

mask = 1 << (bit_index & 0x7);
byte = common->recurse_bitset + (bit_index >> 3);

if (*byte & mask)
  return FALSE;

*byte |= mask;
return TRUE;
}

enum get_recurse_flags {
  recurse_flag_quit_found = (1 << 0),
  recurse_flag_accept_found = (1 << 1),
  recurse_flag_setsom_found = (1 << 2),
  recurse_flag_setmark_found = (1 << 3),
  recurse_flag_control_head_found = (1 << 4),
};

static int get_recurse_data_length(compiler_common *common, PCRE2_SPTR cc, PCRE2_SPTR ccend, uint32_t *result_flags)
{
int length = 1;
int size, offset;
PCRE2_SPTR alternative;
uint32_t recurse_flags = 0;

memset(common->recurse_bitset, 0, common->recurse_bitset_size);

#if defined DEBUG_FORCE_CONTROL_HEAD && DEBUG_FORCE_CONTROL_HEAD
SLJIT_ASSERT(common->control_head_ptr != 0);
recurse_flags |= recurse_flag_control_head_found;
#endif

/* Calculate the sum of the private machine words. */
while (cc < ccend)
  {
  size = 0;
  switch(*cc)
    {
    case OP_SET_SOM:
    SLJIT_ASSERT(common->has_set_som);
    recurse_flags |= recurse_flag_setsom_found;
    cc += 1;
    break;

    case OP_RECURSE:
    if (common->has_set_som)
      recurse_flags |= recurse_flag_setsom_found;
    if (common->mark_ptr != 0)
      recurse_flags |= recurse_flag_setmark_found;
    if (common->capture_last_ptr != 0 && recurse_check_bit(common, common->capture_last_ptr))
      length++;
    cc += 1 + LINK_SIZE;
    break;

    case OP_KET:
    offset = PRIVATE_DATA(cc);
    if (offset != 0)
      {
      if (recurse_check_bit(common, offset))
        length++;
      SLJIT_ASSERT(PRIVATE_DATA(cc + 1) != 0);
      cc += PRIVATE_DATA(cc + 1);
      }
    cc += 1 + LINK_SIZE;
    break;

    case OP_ASSERT:
    case OP_ASSERT_NOT:
    case OP_ASSERTBACK:
    case OP_ASSERTBACK_NOT:
    case OP_ASSERT_NA:
    case OP_ASSERTBACK_NA:
    case OP_ONCE:
    case OP_SCRIPT_RUN:
    case OP_BRAPOS:
    case OP_SBRA:
    case OP_SBRAPOS:
    case OP_SCOND:
    SLJIT_ASSERT(PRIVATE_DATA(cc) != 0);
    if (recurse_check_bit(common, PRIVATE_DATA(cc)))
      length++;
    cc += 1 + LINK_SIZE;
    break;

    case OP_ASSERT_SCS:
    SLJIT_ASSERT(PRIVATE_DATA(cc) != 0);
    if (recurse_check_bit(common, PRIVATE_DATA(cc)))
      length += 2;
    cc += 1 + LINK_SIZE;
    break;

    case OP_CBRA:
    case OP_SCBRA:
    offset = GET2(cc, 1 + LINK_SIZE);
    if (recurse_check_bit(common, OVECTOR(offset << 1)))
      {
      SLJIT_ASSERT(recurse_check_bit(common, OVECTOR((offset << 1) + 1)));
      length += 2;
      }
    if (common->optimized_cbracket[offset] == 0 && recurse_check_bit(common, OVECTOR_PRIV(offset)))
      length++;
    if (common->capture_last_ptr != 0 && recurse_check_bit(common, common->capture_last_ptr))
      length++;
    cc += 1 + LINK_SIZE + IMM2_SIZE;
    break;

    case OP_CBRAPOS:
    case OP_SCBRAPOS:
    offset = GET2(cc, 1 + LINK_SIZE);
    if (recurse_check_bit(common, OVECTOR(offset << 1)))
      {
      SLJIT_ASSERT(recurse_check_bit(common, OVECTOR((offset << 1) + 1)));
      length += 2;
      }
    if (recurse_check_bit(common, OVECTOR_PRIV(offset)))
      length++;
    if (recurse_check_bit(common, PRIVATE_DATA(cc)))
      length++;
    if (common->capture_last_ptr != 0 && recurse_check_bit(common, common->capture_last_ptr))
      length++;
    cc += 1 + LINK_SIZE + IMM2_SIZE;
    break;

    case OP_COND:
    /* Might be a hidden SCOND. */
    alternative = cc + GET(cc, 1);
    if ((*alternative == OP_KETRMAX || *alternative == OP_KETRMIN) && recurse_check_bit(common, PRIVATE_DATA(cc)))
      length++;
    cc += 1 + LINK_SIZE;
    break;

    CASE_ITERATOR_PRIVATE_DATA_1
    offset = PRIVATE_DATA(cc);
    if (offset != 0 && recurse_check_bit(common, offset))
      length++;
    cc += 2;
#ifdef SUPPORT_UNICODE
    if (common->utf && HAS_EXTRALEN(cc[-1])) cc += GET_EXTRALEN(cc[-1]);
#endif
    break;

    CASE_ITERATOR_PRIVATE_DATA_2A
    offset = PRIVATE_DATA(cc);
    if (offset != 0 && recurse_check_bit(common, offset))
      {
      SLJIT_ASSERT(recurse_check_bit(common, offset + sizeof(sljit_sw)));
      length += 2;
      }
    cc += 2;
#ifdef SUPPORT_UNICODE
    if (common->utf && HAS_EXTRALEN(cc[-1])) cc += GET_EXTRALEN(cc[-1]);
#endif
    break;

    CASE_ITERATOR_PRIVATE_DATA_2B
    offset = PRIVATE_DATA(cc);
    if (offset != 0 && recurse_check_bit(common, offset))
      {
      SLJIT_ASSERT(recurse_check_bit(common, offset + sizeof(sljit_sw)));
      length += 2;
      }
    cc += 2 + IMM2_SIZE;
#ifdef SUPPORT_UNICODE
    if (common->utf && HAS_EXTRALEN(cc[-1])) cc += GET_EXTRALEN(cc[-1]);
#endif
    break;

    CASE_ITERATOR_TYPE_PRIVATE_DATA_1
    offset = PRIVATE_DATA(cc);
    if (offset != 0 && recurse_check_bit(common, offset))
      length++;
    cc += 1;
    break;

    CASE_ITERATOR_TYPE_PRIVATE_DATA_2A
    offset = PRIVATE_DATA(cc);
    if (offset != 0 && recurse_check_bit(common, offset))
      {
      SLJIT_ASSERT(recurse_check_bit(common, offset + sizeof(sljit_sw)));
      length += 2;
      }
    cc += 1;
    break;

    CASE_ITERATOR_TYPE_PRIVATE_DATA_2B
    offset = PRIVATE_DATA(cc);
    if (offset != 0 && recurse_check_bit(common, offset))
      {
      SLJIT_ASSERT(recurse_check_bit(common, offset + sizeof(sljit_sw)));
      length += 2;
      }
    cc += 1 + IMM2_SIZE;
    break;

    case OP_CLASS:
    case OP_NCLASS:
#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH != 8
    case OP_XCLASS:
    case OP_ECLASS:
    size = (*cc >= OP_XCLASS) ? GET(cc, 1) : 1 + 32 / (int)sizeof(PCRE2_UCHAR);
#else
    size = 1 + 32 / (int)sizeof(PCRE2_UCHAR);
#endif

    offset = PRIVATE_DATA(cc);
    if (offset != 0 && recurse_check_bit(common, offset))
      length += get_class_iterator_size(cc + size);
    cc += size;
    break;

    case OP_MARK:
    case OP_COMMIT_ARG:
    case OP_PRUNE_ARG:
    case OP_THEN_ARG:
    SLJIT_ASSERT(common->mark_ptr != 0);
    recurse_flags |= recurse_flag_setmark_found;
    if (common->control_head_ptr != 0)
      recurse_flags |= recurse_flag_control_head_found;
    if (*cc != OP_MARK)
      recurse_flags |= recurse_flag_quit_found;

    cc += 1 + 2 + cc[1];
    break;

    case OP_PRUNE:
    case OP_SKIP:
    case OP_COMMIT:
    recurse_flags |= recurse_flag_quit_found;
    cc++;
    break;

    case OP_SKIP_ARG:
    recurse_flags |= recurse_flag_quit_found;
    cc += 1 + 2 + cc[1];
    break;

    case OP_THEN:
    SLJIT_ASSERT(common->control_head_ptr != 0);
    recurse_flags |= recurse_flag_quit_found | recurse_flag_control_head_found;
    cc++;
    break;

    case OP_ACCEPT:
    case OP_ASSERT_ACCEPT:
    recurse_flags |= recurse_flag_accept_found;
    cc++;
    break;

    default:
    cc = next_opcode(common, cc);
    SLJIT_ASSERT(cc != NULL);
    break;
    }
  }
SLJIT_ASSERT(cc == ccend);

if (recurse_flags & recurse_flag_control_head_found)
  length++;
if (recurse_flags & recurse_flag_quit_found)
  {
  if (recurse_flags & recurse_flag_setsom_found)
    length++;
  if (recurse_flags & recurse_flag_setmark_found)
    length++;
  }

*result_flags = recurse_flags;
return length;
}

enum copy_recurse_data_types {
  recurse_copy_from_global,
  recurse_copy_private_to_global,
  recurse_copy_shared_to_global,
  recurse_copy_kept_shared_to_global,
  recurse_swap_global
};

static void copy_recurse_data(compiler_common *common, PCRE2_SPTR cc, PCRE2_SPTR ccend,
  int type, int stackptr, int stacktop, uint32_t recurse_flags)
{
delayed_mem_copy_status status;
PCRE2_SPTR alternative;
sljit_sw private_srcw[2];
sljit_sw shared_srcw[3];
sljit_sw kept_shared_srcw[2];
int private_count, shared_count, kept_shared_count;
int from_sp, base_reg, offset, i;

memset(common->recurse_bitset, 0, common->recurse_bitset_size);

#if defined DEBUG_FORCE_CONTROL_HEAD && DEBUG_FORCE_CONTROL_HEAD
SLJIT_ASSERT(common->control_head_ptr != 0);
recurse_check_bit(common, common->control_head_ptr);
#endif

switch (type)
  {
  case recurse_copy_from_global:
  from_sp = TRUE;
  base_reg = STACK_TOP;
  break;

  case recurse_copy_private_to_global:
  case recurse_copy_shared_to_global:
  case recurse_copy_kept_shared_to_global:
  from_sp = FALSE;
  base_reg = STACK_TOP;
  break;

  default:
  SLJIT_ASSERT(type == recurse_swap_global);
  from_sp = FALSE;
  base_reg = TMP2;
  break;
  }

stackptr = STACK(stackptr);
stacktop = STACK(stacktop);

status.tmp_regs[0] = TMP1;
status.saved_tmp_regs[0] = TMP1;

if (base_reg != TMP2)
  {
  status.tmp_regs[1] = TMP2;
  status.saved_tmp_regs[1] = TMP2;
  }
else
  {
  status.saved_tmp_regs[1] = RETURN_ADDR;
  if (HAS_VIRTUAL_REGISTERS)
    status.tmp_regs[1] = STR_PTR;
  else
    status.tmp_regs[1] = RETURN_ADDR;
  }

status.saved_tmp_regs[2] = TMP3;
if (HAS_VIRTUAL_REGISTERS)
  status.tmp_regs[2] = STR_END;
else
  status.tmp_regs[2] = TMP3;

delayed_mem_copy_init(&status, common);

if (type != recurse_copy_shared_to_global && type != recurse_copy_kept_shared_to_global)
  {
  SLJIT_ASSERT(type == recurse_copy_from_global || type == recurse_copy_private_to_global || type == recurse_swap_global);

  if (!from_sp)
    delayed_mem_copy_move(&status, base_reg, stackptr, SLJIT_SP, common->recursive_head_ptr);

  if (from_sp || type == recurse_swap_global)
    delayed_mem_copy_move(&status, SLJIT_SP, common->recursive_head_ptr, base_reg, stackptr);
  }

stackptr += sizeof(sljit_sw);

#if defined DEBUG_FORCE_CONTROL_HEAD && DEBUG_FORCE_CONTROL_HEAD
if (type != recurse_copy_shared_to_global)
  {
  if (!from_sp)
    delayed_mem_copy_move(&status, base_reg, stackptr, SLJIT_SP, common->control_head_ptr);

  if (from_sp || type == recurse_swap_global)
    delayed_mem_copy_move(&status, SLJIT_SP, common->control_head_ptr, base_reg, stackptr);
  }

stackptr += sizeof(sljit_sw);
#endif

while (cc < ccend)
  {
  private_count = 0;
  shared_count = 0;
  kept_shared_count = 0;

  switch(*cc)
    {
    case OP_SET_SOM:
    SLJIT_ASSERT(common->has_set_som);
    if ((recurse_flags & recurse_flag_quit_found) && recurse_check_bit(common, OVECTOR(0)))
      {
      kept_shared_srcw[0] = OVECTOR(0);
      kept_shared_count = 1;
      }
    cc += 1;
    break;

    case OP_RECURSE:
    if (recurse_flags & recurse_flag_quit_found)
      {
      if (common->has_set_som && recurse_check_bit(common, OVECTOR(0)))
        {
        kept_shared_srcw[0] = OVECTOR(0);
        kept_shared_count = 1;
        }
      if (common->mark_ptr != 0 && recurse_check_bit(common, common->mark_ptr))
        {
        kept_shared_srcw[kept_shared_count] = common->mark_ptr;
        kept_shared_count++;
        }
      }
    if (common->capture_last_ptr != 0 && recurse_check_bit(common, common->capture_last_ptr))
      {
      shared_srcw[0] = common->capture_last_ptr;
      shared_count = 1;
      }
    cc += 1 + LINK_SIZE;
    break;

    case OP_KET:
    private_srcw[0] = PRIVATE_DATA(cc);
    if (private_srcw[0] != 0)
      {
      if (recurse_check_bit(common, private_srcw[0]))
        private_count = 1;
      SLJIT_ASSERT(PRIVATE_DATA(cc + 1) != 0);
      cc += PRIVATE_DATA(cc + 1);
      }
    cc += 1 + LINK_SIZE;
    break;

    case OP_ASSERT:
    case OP_ASSERT_NOT:
    case OP_ASSERTBACK:
    case OP_ASSERTBACK_NOT:
    case OP_ASSERT_NA:
    case OP_ASSERTBACK_NA:
    case OP_ONCE:
    case OP_SCRIPT_RUN:
    case OP_BRAPOS:
    case OP_SBRA:
    case OP_SBRAPOS:
    case OP_SCOND:
    private_srcw[0] = PRIVATE_DATA(cc);
    if (recurse_check_bit(common, private_srcw[0]))
      private_count = 1;
    cc += 1 + LINK_SIZE;
    break;

    case OP_ASSERT_SCS:
    private_srcw[0] = PRIVATE_DATA(cc);
    private_srcw[1] = private_srcw[0] + sizeof(sljit_sw);
    if (recurse_check_bit(common, private_srcw[0]))
      private_count = 2;
    cc += 1 + LINK_SIZE;
    break;

    case OP_CBRA:
    case OP_SCBRA:
    offset = GET2(cc, 1 + LINK_SIZE);
    shared_srcw[0] = OVECTOR(offset << 1);
    if (recurse_check_bit(common, shared_srcw[0]))
      {
      shared_srcw[1] = shared_srcw[0] + sizeof(sljit_sw);
      SLJIT_ASSERT(recurse_check_bit(common, shared_srcw[1]));
      shared_count = 2;
      }

    if (common->capture_last_ptr != 0 && recurse_check_bit(common, common->capture_last_ptr))
      {
      shared_srcw[shared_count] = common->capture_last_ptr;
      shared_count++;
      }

    if (common->optimized_cbracket[offset] == 0)
      {
      private_srcw[0] = OVECTOR_PRIV(offset);
      if (recurse_check_bit(common, private_srcw[0]))
        private_count = 1;
      }

    cc += 1 + LINK_SIZE + IMM2_SIZE;
    break;

    case OP_CBRAPOS:
    case OP_SCBRAPOS:
    offset = GET2(cc, 1 + LINK_SIZE);
    shared_srcw[0] = OVECTOR(offset << 1);
    if (recurse_check_bit(common, shared_srcw[0]))
      {
      shared_srcw[1] = shared_srcw[0] + sizeof(sljit_sw);
      SLJIT_ASSERT(recurse_check_bit(common, shared_srcw[1]));
      shared_count = 2;
      }

    if (common->capture_last_ptr != 0 && recurse_check_bit(common, common->capture_last_ptr))
      {
      shared_srcw[shared_count] = common->capture_last_ptr;
      shared_count++;
      }

    private_srcw[0] = PRIVATE_DATA(cc);
    if (recurse_check_bit(common, private_srcw[0]))
      private_count = 1;

    offset = OVECTOR_PRIV(offset);
    if (recurse_check_bit(common, offset))
      {
      private_srcw[private_count] = offset;
      private_count++;
      }
    cc += 1 + LINK_SIZE + IMM2_SIZE;
    break;

    case OP_COND:
    /* Might be a hidden SCOND. */
    alternative = cc + GET(cc, 1);
    if (*alternative == OP_KETRMAX || *alternative == OP_KETRMIN)
      {
      private_srcw[0] = PRIVATE_DATA(cc);
      if (recurse_check_bit(common, private_srcw[0]))
        private_count = 1;
      }
    cc += 1 + LINK_SIZE;
    break;

    CASE_ITERATOR_PRIVATE_DATA_1
    private_srcw[0] = PRIVATE_DATA(cc);
    if (private_srcw[0] != 0 && recurse_check_bit(common, private_srcw[0]))
      private_count = 1;
    cc += 2;
#ifdef SUPPORT_UNICODE
    if (common->utf && HAS_EXTRALEN(cc[-1])) cc += GET_EXTRALEN(cc[-1]);
#endif
    break;

    CASE_ITERATOR_PRIVATE_DATA_2A
    private_srcw[0] = PRIVATE_DATA(cc);
    if (private_srcw[0] != 0 && recurse_check_bit(common, private_srcw[0]))
      {
      private_count = 2;
      private_srcw[1] = private_srcw[0] + sizeof(sljit_sw);
      SLJIT_ASSERT(recurse_check_bit(common, private_srcw[1]));
      }
    cc += 2;
#ifdef SUPPORT_UNICODE
    if (common->utf && HAS_EXTRALEN(cc[-1])) cc += GET_EXTRALEN(cc[-1]);
#endif
    break;

    CASE_ITERATOR_PRIVATE_DATA_2B
    private_srcw[0] = PRIVATE_DATA(cc);
    if (private_srcw[0] != 0 && recurse_check_bit(common, private_srcw[0]))
      {
      private_count = 2;
      private_srcw[1] = private_srcw[0] + sizeof(sljit_sw);
      SLJIT_ASSERT(recurse_check_bit(common, private_srcw[1]));
      }
    cc += 2 + IMM2_SIZE;
#ifdef SUPPORT_UNICODE
    if (common->utf && HAS_EXTRALEN(cc[-1])) cc += GET_EXTRALEN(cc[-1]);
#endif
    break;

    CASE_ITERATOR_TYPE_PRIVATE_DATA_1
    private_srcw[0] = PRIVATE_DATA(cc);
    if (private_srcw[0] != 0 && recurse_check_bit(common, private_srcw[0]))
      private_count = 1;
    cc += 1;
    break;

    CASE_ITERATOR_TYPE_PRIVATE_DATA_2A
    private_srcw[0] = PRIVATE_DATA(cc);
    if (private_srcw[0] != 0 && recurse_check_bit(common, private_srcw[0]))
      {
      private_count = 2;
      private_srcw[1] = private_srcw[0] + sizeof(sljit_sw);
      SLJIT_ASSERT(recurse_check_bit(common, private_srcw[1]));
      }
    cc += 1;
    break;

    CASE_ITERATOR_TYPE_PRIVATE_DATA_2B
    private_srcw[0] = PRIVATE_DATA(cc);
    if (private_srcw[0] != 0 && recurse_check_bit(common, private_srcw[0]))
      {
      private_count = 2;
      private_srcw[1] = private_srcw[0] + sizeof(sljit_sw);
      SLJIT_ASSERT(recurse_check_bit(common, private_srcw[1]));
      }
    cc += 1 + IMM2_SIZE;
    break;

    case OP_CLASS:
    case OP_NCLASS:
#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH != 8
    case OP_XCLASS:
    case OP_ECLASS:
    i = (*cc >= OP_XCLASS) ? GET(cc, 1) : 1 + 32 / (int)sizeof(PCRE2_UCHAR);
#else
    i = 1 + 32 / (int)sizeof(PCRE2_UCHAR);
#endif
    if (PRIVATE_DATA(cc) != 0)
      {
      private_count = 1;
      private_srcw[0] = PRIVATE_DATA(cc);
      switch(get_class_iterator_size(cc + i))
        {
        case 1:
        break;

        case 2:
        if (recurse_check_bit(common, private_srcw[0]))
          {
          private_count = 2;
          private_srcw[1] = private_srcw[0] + sizeof(sljit_sw);
          SLJIT_ASSERT(recurse_check_bit(common, private_srcw[1]));
          }
        break;

        default:
        SLJIT_UNREACHABLE();
        break;
        }
      }
    cc += i;
    break;

    case OP_MARK:
    case OP_COMMIT_ARG:
    case OP_PRUNE_ARG:
    case OP_THEN_ARG:
    SLJIT_ASSERT(common->mark_ptr != 0);
    if ((recurse_flags & recurse_flag_quit_found) && recurse_check_bit(common, common->mark_ptr))
      {
      kept_shared_srcw[0] = common->mark_ptr;
      kept_shared_count = 1;
      }
    if (common->control_head_ptr != 0 && recurse_check_bit(common, common->control_head_ptr))
      {
      private_srcw[0] = common->control_head_ptr;
      private_count = 1;
      }
    cc += 1 + 2 + cc[1];
    break;

    case OP_THEN:
    SLJIT_ASSERT(common->control_head_ptr != 0);
    if (recurse_check_bit(common, common->control_head_ptr))
      {
      private_srcw[0] = common->control_head_ptr;
      private_count = 1;
      }
    cc++;
    break;

    default:
    cc = next_opcode(common, cc);
    SLJIT_ASSERT(cc != NULL);
    continue;
    }

  if (type != recurse_copy_shared_to_global && type != recurse_copy_kept_shared_to_global)
    {
    SLJIT_ASSERT(type == recurse_copy_from_global || type == recurse_copy_private_to_global || type == recurse_swap_global);

    for (i = 0; i < private_count; i++)
      {
      SLJIT_ASSERT(private_srcw[i] != 0);

      if (!from_sp)
        delayed_mem_copy_move(&status, base_reg, stackptr, SLJIT_SP, private_srcw[i]);

      if (from_sp || type == recurse_swap_global)
        delayed_mem_copy_move(&status, SLJIT_SP, private_srcw[i], base_reg, stackptr);

      stackptr += sizeof(sljit_sw);
      }
    }
  else
    stackptr += sizeof(sljit_sw) * private_count;

  if (type != recurse_copy_private_to_global && type != recurse_copy_kept_shared_to_global)
    {
    SLJIT_ASSERT(type == recurse_copy_from_global || type == recurse_copy_shared_to_global || type == recurse_swap_global);

    for (i = 0; i < shared_count; i++)
      {
      SLJIT_ASSERT(shared_srcw[i] != 0);

      if (!from_sp)
        delayed_mem_copy_move(&status, base_reg, stackptr, SLJIT_SP, shared_srcw[i]);

      if (from_sp || type == recurse_swap_global)
        delayed_mem_copy_move(&status, SLJIT_SP, shared_srcw[i], base_reg, stackptr);

      stackptr += sizeof(sljit_sw);
      }
    }
  else
    stackptr += sizeof(sljit_sw) * shared_count;

  if (type != recurse_copy_private_to_global && type != recurse_swap_global)
    {
    SLJIT_ASSERT(type == recurse_copy_from_global || type == recurse_copy_shared_to_global || type == recurse_copy_kept_shared_to_global);

    for (i = 0; i < kept_shared_count; i++)
      {
      SLJIT_ASSERT(kept_shared_srcw[i] != 0);

      if (!from_sp)
        delayed_mem_copy_move(&status, base_reg, stackptr, SLJIT_SP, kept_shared_srcw[i]);

      if (from_sp || type == recurse_swap_global)
        delayed_mem_copy_move(&status, SLJIT_SP, kept_shared_srcw[i], base_reg, stackptr);

      stackptr += sizeof(sljit_sw);
      }
    }
  else
    stackptr += sizeof(sljit_sw) * kept_shared_count;
  }

SLJIT_ASSERT(cc == ccend && stackptr == stacktop);

delayed_mem_copy_finish(&status);
}

static SLJIT_INLINE PCRE2_SPTR set_then_offsets(compiler_common *common, PCRE2_SPTR cc, sljit_u8 *current_offset)
{
PCRE2_SPTR end = bracketend(cc);
BOOL has_alternatives = cc[GET(cc, 1)] == OP_ALT;

/* Assert captures *THEN verb even if it has no alternatives. */
if (*cc >= OP_ASSERT && *cc <= OP_ASSERTBACK_NOT)
  current_offset = NULL;
else if (*cc >= OP_ASSERT_NA && *cc <= OP_ASSERT_SCS)
  has_alternatives = TRUE;
/* Conditional block does never capture. */
else if (*cc == OP_COND || *cc == OP_SCOND)
  has_alternatives = FALSE;

cc = next_opcode(common, cc);

if (has_alternatives)
  {
  switch (*cc)
    {
    case OP_REVERSE:
    case OP_CREF:
      cc += 1 + IMM2_SIZE;
      break;
    case OP_VREVERSE:
    case OP_DNCREF:
      cc += 1 + 2 * IMM2_SIZE;
      break;
    }

  current_offset = common->then_offsets + (cc - common->start);
  }

while (cc < end)
  {
  if (*cc >= OP_ASSERT && *cc <= OP_SCOND)
    {
    cc = set_then_offsets(common, cc, current_offset);
    continue;
    }

  if (*cc == OP_ALT && has_alternatives)
    {
    cc += 1 + LINK_SIZE;

    if (*cc == OP_REVERSE)
      cc += 1 + IMM2_SIZE;
    else if (*cc == OP_VREVERSE)
      cc += 1 + 2 * IMM2_SIZE;

    current_offset = common->then_offsets + (cc - common->start);
    continue;
    }

  if (*cc >= OP_THEN && *cc <= OP_THEN_ARG && current_offset != NULL)
    *current_offset = 1;
  cc = next_opcode(common, cc);
  }

cc = end - 1 - LINK_SIZE;

/* Ignore repeats. */
if (*cc == OP_KET && PRIVATE_DATA(cc) != 0)
  end += PRIVATE_DATA(cc + 1);

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
while (list != NULL)
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

static SLJIT_INLINE void count_match(compiler_common *common)
{
DEFINE_COMPILER;

OP2(SLJIT_SUB | SLJIT_SET_Z, COUNT_MATCH, 0, COUNT_MATCH, 0, SLJIT_IMM, 1);
add_jump(compiler, &common->calllimit, JUMP(SLJIT_ZERO));
}

static SLJIT_INLINE void allocate_stack(compiler_common *common, int size)
{
/* May destroy all locals and registers except TMP2. */
DEFINE_COMPILER;

SLJIT_ASSERT(size > 0);
OP2(SLJIT_SUB, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, size * SSIZE_OF(sw));
#ifdef DESTROY_REGISTERS
OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, 12345);
OP1(SLJIT_MOV, TMP3, 0, TMP1, 0);
OP1(SLJIT_MOV, RETURN_ADDR, 0, TMP1, 0);
#if defined SLJIT_DEBUG && SLJIT_DEBUG
SLJIT_ASSERT(common->locals_size >= 2 * SSIZE_OF(sw));
/* These two are also used by the stackalloc calls. */
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCAL0, TMP1, 0);
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCAL1, TMP1, 0);
#endif
#endif
add_stub(common, CMP(SLJIT_LESS, STACK_TOP, 0, STACK_LIMIT, 0));
}

static SLJIT_INLINE void free_stack(compiler_common *common, int size)
{
DEFINE_COMPILER;

SLJIT_ASSERT(size > 0);
OP2(SLJIT_ADD, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, size * SSIZE_OF(sw));
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
  if (sljit_emit_mem_update(compiler, SLJIT_MOV | SLJIT_MEM_SUPP | SLJIT_MEM_STORE | SLJIT_MEM_PRE, SLJIT_R0, SLJIT_MEM1(SLJIT_R1), sizeof(sljit_sw)) == SLJIT_SUCCESS)
    {
    GET_LOCAL_BASE(SLJIT_R1, 0, OVECTOR_START);
    OP1(SLJIT_MOV, SLJIT_R2, 0, SLJIT_IMM, length - 1);
    loop = LABEL();
    sljit_emit_mem_update(compiler, SLJIT_MOV | SLJIT_MEM_STORE | SLJIT_MEM_PRE, SLJIT_R0, SLJIT_MEM1(SLJIT_R1), sizeof(sljit_sw));
    OP2(SLJIT_SUB | SLJIT_SET_Z, SLJIT_R2, 0, SLJIT_R2, 0, SLJIT_IMM, 1);
    JUMPTO(SLJIT_NOT_ZERO, loop);
    }
  else
    {
    GET_LOCAL_BASE(SLJIT_R1, 0, OVECTOR_START + sizeof(sljit_sw));
    OP1(SLJIT_MOV, SLJIT_R2, 0, SLJIT_IMM, length - 1);
    loop = LABEL();
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_R1), 0, SLJIT_R0, 0);
    OP2(SLJIT_ADD, SLJIT_R1, 0, SLJIT_R1, 0, SLJIT_IMM, sizeof(sljit_sw));
    OP2(SLJIT_SUB | SLJIT_SET_Z, SLJIT_R2, 0, SLJIT_R2, 0, SLJIT_IMM, 1);
    JUMPTO(SLJIT_NOT_ZERO, loop);
    }
  }
}

static SLJIT_INLINE void reset_early_fail(compiler_common *common)
{
DEFINE_COMPILER;
sljit_u32 size = (sljit_u32)(common->early_fail_end_ptr - common->early_fail_start_ptr);
sljit_u32 uncleared_size;
sljit_s32 src = SLJIT_IMM;
sljit_s32 i;
struct sljit_label *loop;

SLJIT_ASSERT(common->early_fail_start_ptr < common->early_fail_end_ptr);

if (size == sizeof(sljit_sw))
  {
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->early_fail_start_ptr, SLJIT_IMM, 0);
  return;
  }

if (sljit_get_register_index(SLJIT_GP_REGISTER, TMP3) >= 0 && !sljit_has_cpu_feature(SLJIT_HAS_ZERO_REGISTER))
  {
  OP1(SLJIT_MOV, TMP3, 0, SLJIT_IMM, 0);
  src = TMP3;
  }

if (size <= 6 * sizeof(sljit_sw))
  {
  for (i = common->early_fail_start_ptr; i < common->early_fail_end_ptr; i += sizeof(sljit_sw))
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), i, src, 0);
  return;
  }

GET_LOCAL_BASE(TMP1, 0, common->early_fail_start_ptr);

uncleared_size = ((size / sizeof(sljit_sw)) % 3) * sizeof(sljit_sw);

OP2(SLJIT_ADD, TMP2, 0, TMP1, 0, SLJIT_IMM, size - uncleared_size);

loop = LABEL();
OP1(SLJIT_MOV, SLJIT_MEM1(TMP1), 0, src, 0);
OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, 3 * sizeof(sljit_sw));
OP1(SLJIT_MOV, SLJIT_MEM1(TMP1), -2 * SSIZE_OF(sw), src, 0);
OP1(SLJIT_MOV, SLJIT_MEM1(TMP1), -1 * SSIZE_OF(sw), src, 0);
CMPTO(SLJIT_LESS, TMP1, 0, TMP2, 0, loop);

if (uncleared_size >= sizeof(sljit_sw))
  OP1(SLJIT_MOV, SLJIT_MEM1(TMP1), 0, src, 0);

if (uncleared_size >= 2 * sizeof(sljit_sw))
  OP1(SLJIT_MOV, SLJIT_MEM1(TMP1), sizeof(sljit_sw), src, 0);
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
  if (sljit_emit_mem_update(compiler, SLJIT_MOV | SLJIT_MEM_SUPP | SLJIT_MEM_STORE | SLJIT_MEM_PRE, TMP1, SLJIT_MEM1(TMP2), sizeof(sljit_sw)) == SLJIT_SUCCESS)
    {
    GET_LOCAL_BASE(TMP2, 0, OVECTOR_START + sizeof(sljit_sw));
    OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_IMM, length - 2);
    loop = LABEL();
    sljit_emit_mem_update(compiler, SLJIT_MOV | SLJIT_MEM_STORE | SLJIT_MEM_PRE, TMP1, SLJIT_MEM1(TMP2), sizeof(sljit_sw));
    OP2(SLJIT_SUB | SLJIT_SET_Z, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, 1);
    JUMPTO(SLJIT_NOT_ZERO, loop);
    }
  else
    {
    GET_LOCAL_BASE(TMP2, 0, OVECTOR_START + 2 * sizeof(sljit_sw));
    OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_IMM, length - 2);
    loop = LABEL();
    OP1(SLJIT_MOV, SLJIT_MEM1(TMP2), 0, TMP1, 0);
    OP2(SLJIT_ADD, TMP2, 0, TMP2, 0, SLJIT_IMM, sizeof(sljit_sw));
    OP2(SLJIT_SUB | SLJIT_SET_Z, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, 1);
    JUMPTO(SLJIT_NOT_ZERO, loop);
    }
  }

if (!HAS_VIRTUAL_REGISTERS)
  OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(ARGUMENTS), SLJIT_OFFSETOF(jit_arguments, stack));
else
  OP1(SLJIT_MOV, STACK_TOP, 0, ARGUMENTS, 0);

if (common->mark_ptr != 0)
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->mark_ptr, SLJIT_IMM, 0);
if (common->control_head_ptr != 0)
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, SLJIT_IMM, 0);
if (HAS_VIRTUAL_REGISTERS)
  OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(STACK_TOP), SLJIT_OFFSETOF(jit_arguments, stack));

OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->start_ptr);
OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(STACK_TOP), SLJIT_OFFSETOF(struct sljit_stack, end));
}

static sljit_sw SLJIT_FUNC do_search_mark(sljit_sw *current, PCRE2_SPTR skip_arg)
{
while (current != NULL)
  {
  switch (current[1])
    {
    case type_then_trap:
    break;

    case type_mark:
    if (PRIV(strcmp)(skip_arg, (PCRE2_SPTR)current[2]) == 0)
      return current[3];
    break;

    default:
    SLJIT_UNREACHABLE();
    break;
    }
  SLJIT_ASSERT(current[0] == 0 || current < (sljit_sw*)current[0]);
  current = (sljit_sw*)current[0];
  }
return 0;
}

static SLJIT_INLINE void copy_ovector(compiler_common *common, int topbracket)
{
DEFINE_COMPILER;
struct sljit_label *loop;
BOOL has_pre;

/* At this point we can freely use all registers. */
OP1(SLJIT_MOV, SLJIT_S2, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(1));
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(1), STR_PTR, 0);

if (HAS_VIRTUAL_REGISTERS)
  {
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
  }
else
  {
  OP1(SLJIT_MOV, SLJIT_S0, 0, SLJIT_MEM1(SLJIT_SP), common->start_ptr);
  OP1(SLJIT_MOV, SLJIT_R2, 0, SLJIT_MEM1(ARGUMENTS), SLJIT_OFFSETOF(jit_arguments, match_data));
  if (common->mark_ptr != 0)
    OP1(SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(SLJIT_SP), common->mark_ptr);
  OP1(SLJIT_MOV_U32, SLJIT_R1, 0, SLJIT_MEM1(ARGUMENTS), SLJIT_OFFSETOF(jit_arguments, oveccount));
  OP1(SLJIT_MOV, SLJIT_MEM1(ARGUMENTS), SLJIT_OFFSETOF(jit_arguments, startchar_ptr), SLJIT_S0, 0);
  if (common->mark_ptr != 0)
    OP1(SLJIT_MOV, SLJIT_MEM1(ARGUMENTS), SLJIT_OFFSETOF(jit_arguments, mark_ptr), SLJIT_R0, 0);
  OP2(SLJIT_ADD, SLJIT_R2, 0, SLJIT_R2, 0, SLJIT_IMM, SLJIT_OFFSETOF(pcre2_match_data, ovector) - sizeof(PCRE2_SIZE));
  }

has_pre = sljit_emit_mem_update(compiler, SLJIT_MOV | SLJIT_MEM_SUPP | SLJIT_MEM_PRE, SLJIT_S1, SLJIT_MEM1(SLJIT_S0), sizeof(sljit_sw)) == SLJIT_SUCCESS;

GET_LOCAL_BASE(SLJIT_S0, 0, OVECTOR_START - (has_pre ? sizeof(sljit_sw) : 0));
OP1(SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(HAS_VIRTUAL_REGISTERS ? SLJIT_R0 : ARGUMENTS), SLJIT_OFFSETOF(jit_arguments, begin));

loop = LABEL();

if (has_pre)
  sljit_emit_mem_update(compiler, SLJIT_MOV | SLJIT_MEM_PRE, SLJIT_S1, SLJIT_MEM1(SLJIT_S0), sizeof(sljit_sw));
else
  {
  OP1(SLJIT_MOV, SLJIT_S1, 0, SLJIT_MEM1(SLJIT_S0), 0);
  OP2(SLJIT_ADD, SLJIT_S0, 0, SLJIT_S0, 0, SLJIT_IMM, sizeof(sljit_sw));
  }

OP2(SLJIT_ADD, SLJIT_R2, 0, SLJIT_R2, 0, SLJIT_IMM, sizeof(PCRE2_SIZE));
OP2(SLJIT_SUB, SLJIT_S1, 0, SLJIT_S1, 0, SLJIT_R0, 0);
/* Copy the integer value to the output buffer */
#if PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32
OP2(SLJIT_ASHR, SLJIT_S1, 0, SLJIT_S1, 0, SLJIT_IMM, UCHAR_SHIFT);
#endif

SLJIT_ASSERT(sizeof(PCRE2_SIZE) == 4 || sizeof(PCRE2_SIZE) == 8);
OP1(((sizeof(PCRE2_SIZE) == 4) ? SLJIT_MOV_U32 : SLJIT_MOV), SLJIT_MEM1(SLJIT_R2), 0, SLJIT_S1, 0);

OP2(SLJIT_SUB | SLJIT_SET_Z, SLJIT_R1, 0, SLJIT_R1, 0, SLJIT_IMM, 1);
JUMPTO(SLJIT_NOT_ZERO, loop);

/* Calculate the return value, which is the maximum ovector value. */
if (topbracket > 1)
  {
  if (sljit_emit_mem_update(compiler, SLJIT_MOV | SLJIT_MEM_SUPP | SLJIT_MEM_PRE, SLJIT_R2, SLJIT_MEM1(SLJIT_R0), -(2 * SSIZE_OF(sw))) == SLJIT_SUCCESS)
    {
    GET_LOCAL_BASE(SLJIT_R0, 0, OVECTOR_START + topbracket * 2 * sizeof(sljit_sw));
    OP1(SLJIT_MOV, SLJIT_R1, 0, SLJIT_IMM, topbracket + 1);

    /* OVECTOR(0) is never equal to SLJIT_S2. */
    loop = LABEL();
    sljit_emit_mem_update(compiler, SLJIT_MOV | SLJIT_MEM_PRE, SLJIT_R2, SLJIT_MEM1(SLJIT_R0), -(2 * SSIZE_OF(sw)));
    OP2(SLJIT_SUB, SLJIT_R1, 0, SLJIT_R1, 0, SLJIT_IMM, 1);
    CMPTO(SLJIT_EQUAL, SLJIT_R2, 0, SLJIT_S2, 0, loop);
    OP1(SLJIT_MOV, SLJIT_RETURN_REG, 0, SLJIT_R1, 0);
    }
  else
    {
    GET_LOCAL_BASE(SLJIT_R0, 0, OVECTOR_START + (topbracket - 1) * 2 * sizeof(sljit_sw));
    OP1(SLJIT_MOV, SLJIT_R1, 0, SLJIT_IMM, topbracket + 1);

    /* OVECTOR(0) is never equal to SLJIT_S2. */
    loop = LABEL();
    OP1(SLJIT_MOV, SLJIT_R2, 0, SLJIT_MEM1(SLJIT_R0), 0);
    OP2(SLJIT_SUB, SLJIT_R0, 0, SLJIT_R0, 0, SLJIT_IMM, 2 * SSIZE_OF(sw));
    OP2(SLJIT_SUB, SLJIT_R1, 0, SLJIT_R1, 0, SLJIT_IMM, 1);
    CMPTO(SLJIT_EQUAL, SLJIT_R2, 0, SLJIT_S2, 0, loop);
    OP1(SLJIT_MOV, SLJIT_RETURN_REG, 0, SLJIT_R1, 0);
    }
  }
else
  OP1(SLJIT_MOV, SLJIT_RETURN_REG, 0, SLJIT_IMM, 1);
}

static SLJIT_INLINE void return_with_partial_match(compiler_common *common, struct sljit_label *quit)
{
DEFINE_COMPILER;
sljit_s32 mov_opcode;
sljit_s32 arguments_reg = !HAS_VIRTUAL_REGISTERS ? ARGUMENTS : SLJIT_R1;

SLJIT_COMPILE_ASSERT(STR_END == SLJIT_S0, str_end_must_be_saved_reg0);
SLJIT_ASSERT(common->start_used_ptr != 0 && common->start_ptr != 0
  && (common->mode == PCRE2_JIT_PARTIAL_SOFT ? common->hit_start != 0 : common->hit_start == 0));

if (arguments_reg != ARGUMENTS)
  OP1(SLJIT_MOV, arguments_reg, 0, ARGUMENTS, 0);
OP1(SLJIT_MOV, SLJIT_R2, 0, SLJIT_MEM1(SLJIT_SP),
  common->mode == PCRE2_JIT_PARTIAL_SOFT ? common->hit_start : common->start_ptr);
OP1(SLJIT_MOV, SLJIT_RETURN_REG, 0, SLJIT_IMM, PCRE2_ERROR_PARTIAL);

/* Store match begin and end. */
OP1(SLJIT_MOV, SLJIT_S1, 0, SLJIT_MEM1(arguments_reg), SLJIT_OFFSETOF(jit_arguments, begin));
OP1(SLJIT_MOV, SLJIT_MEM1(arguments_reg), SLJIT_OFFSETOF(jit_arguments, startchar_ptr), SLJIT_R2, 0);
OP1(SLJIT_MOV, SLJIT_R1, 0, SLJIT_MEM1(arguments_reg), SLJIT_OFFSETOF(jit_arguments, match_data));

mov_opcode = (sizeof(PCRE2_SIZE) == 4) ? SLJIT_MOV_U32 : SLJIT_MOV;

OP2(SLJIT_SUB, SLJIT_R2, 0, SLJIT_R2, 0, SLJIT_S1, 0);
#if PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32
OP2(SLJIT_ASHR, SLJIT_R2, 0, SLJIT_R2, 0, SLJIT_IMM, UCHAR_SHIFT);
#endif
OP1(mov_opcode, SLJIT_MEM1(SLJIT_R1), SLJIT_OFFSETOF(pcre2_match_data, ovector), SLJIT_R2, 0);

OP2(SLJIT_SUB, STR_END, 0, STR_END, 0, SLJIT_S1, 0);
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
if (common->utf || common->ucp)
  {
  if (common->utf)
    {
    GETCHAR(c, cc);
    }
  else
    c = *cc;

  if (c > 127)
    return c != UCD_OTHERCASE(c);

  return common->fcc[c] != c;
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
if ((common->utf || common->ucp) && c > 127)
  return UCD_OTHERCASE(c);
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
if (common->utf || common->ucp)
  {
  if (common->utf)
    {
    GETCHAR(c, cc);
    }
  else
    c = *cc;

  if (c <= 127)
    oc = common->fcc[c];
  else
    oc = UCD_OTHERCASE(c);
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
  if (bit >= (1u << 10))
    bit >>= 10;
  else
    return (bit < 256) ? ((2 << 8) | bit) : ((3 << 8) | (bit >> 8));
  }
#endif /* SUPPORT_UNICODE */
return (bit < 256) ? ((0u << 8) | bit) : ((1u << 8) | (bit >> 8));

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

if (!force && !common->allow_empty_partial)
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
if (!common->allow_empty_partial)
  add_jump(compiler, backtracks, CMP(SLJIT_GREATER_EQUAL, SLJIT_MEM1(SLJIT_SP), common->start_used_ptr, STR_PTR, 0));
else if (common->mode == PCRE2_JIT_PARTIAL_SOFT)
  add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, SLJIT_MEM1(SLJIT_SP), common->start_used_ptr, SLJIT_IMM, -1));

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

static void process_partial_match(compiler_common *common)
{
DEFINE_COMPILER;
struct sljit_jump *jump;

/* Partial matching mode. */
if (common->mode == PCRE2_JIT_PARTIAL_SOFT)
  {
  jump = CMP(SLJIT_GREATER_EQUAL, SLJIT_MEM1(SLJIT_SP), common->start_used_ptr, STR_PTR, 0);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->hit_start, SLJIT_IMM, 0);
  JUMPHERE(jump);
  }
else if (common->mode == PCRE2_JIT_PARTIAL_HARD)
  {
  if (common->partialmatchlabel != NULL)
    CMPTO(SLJIT_LESS, SLJIT_MEM1(SLJIT_SP), common->start_used_ptr, STR_PTR, 0, common->partialmatchlabel);
  else
    add_jump(compiler, &common->partialmatch, CMP(SLJIT_LESS, SLJIT_MEM1(SLJIT_SP), common->start_used_ptr, STR_PTR, 0));
  }
}

static void detect_partial_match_to(compiler_common *common, struct sljit_label *label)
{
DEFINE_COMPILER;

CMPTO(SLJIT_LESS, STR_PTR, 0, STR_END, 0, label);
process_partial_match(common);
}

static void peek_char(compiler_common *common, sljit_u32 max, sljit_s32 dst, sljit_sw dstw, jump_list **backtracks)
{
/* Reads the character into TMP1, keeps STR_PTR.
Does not check STR_END. TMP2, dst, RETURN_ADDR Destroyed. */
DEFINE_COMPILER;
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
struct sljit_jump *jump;
#endif /* SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32 */

SLJIT_UNUSED_ARG(max);
SLJIT_UNUSED_ARG(dst);
SLJIT_UNUSED_ARG(dstw);
SLJIT_UNUSED_ARG(backtracks);

OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));

#ifdef SUPPORT_UNICODE
#if PCRE2_CODE_UNIT_WIDTH == 8
if (common->utf)
  {
  if (max < 128) return;

  jump = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0x80);
  OP1(SLJIT_MOV, dst, dstw, STR_PTR, 0);
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  add_jump(compiler, common->invalid_utf ? &common->utfreadchar_invalid : &common->utfreadchar, JUMP(SLJIT_FAST_CALL));
  OP1(SLJIT_MOV, STR_PTR, 0, dst, dstw);
  if (backtracks && common->invalid_utf)
    add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, INVALID_UTF_CHAR));
  JUMPHERE(jump);
  }
#elif PCRE2_CODE_UNIT_WIDTH == 16
if (common->utf)
  {
  if (max < 0xd800) return;

  OP2(SLJIT_SUB, TMP2, 0, TMP1, 0, SLJIT_IMM, 0xd800);

  if (common->invalid_utf)
    {
    jump = CMP(SLJIT_GREATER_EQUAL, TMP2, 0, SLJIT_IMM, 0xe000 - 0xd800);
    OP1(SLJIT_MOV, dst, dstw, STR_PTR, 0);
    OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
    add_jump(compiler, &common->utfreadchar_invalid, JUMP(SLJIT_FAST_CALL));
    OP1(SLJIT_MOV, STR_PTR, 0, dst, dstw);
    if (backtracks && common->invalid_utf)
      add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, INVALID_UTF_CHAR));
    }
  else
    {
    /* TMP2 contains the high surrogate. */
    jump = CMP(SLJIT_GREATER_EQUAL, TMP2, 0, SLJIT_IMM, 0xdc00 - 0xd800);
    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(1));
    OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, 10);
    OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x10000 - 0xdc00);
    OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, TMP2, 0);
    }

  JUMPHERE(jump);
  }
#elif PCRE2_CODE_UNIT_WIDTH == 32
if (common->invalid_utf)
  {
  if (max < 0xd800) return;

  if (backtracks != NULL)
    {
    OP2(SLJIT_SUB, TMP2, 0, TMP1, 0, SLJIT_IMM, 0xd800);
    add_jump(compiler, backtracks, CMP(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 0x110000));
    add_jump(compiler, backtracks, CMP(SLJIT_LESS, TMP2, 0, SLJIT_IMM, 0xe000 - 0xd800));
    }
  else
    {
    OP2(SLJIT_SUB, TMP2, 0, TMP1, 0, SLJIT_IMM, 0xd800);
    OP2U(SLJIT_SUB | SLJIT_SET_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 0x110000);
    SELECT(SLJIT_GREATER_EQUAL, TMP1, SLJIT_IMM, INVALID_UTF_CHAR, TMP1);
    OP2U(SLJIT_SUB | SLJIT_SET_LESS, TMP2, 0, SLJIT_IMM, 0xe000 - 0xd800);
    SELECT(SLJIT_LESS, TMP1, SLJIT_IMM, INVALID_UTF_CHAR, TMP1);
    }
  }
#endif /* PCRE2_CODE_UNIT_WIDTH == [8|16|32] */
#endif /* SUPPORT_UNICODE */
}

static void peek_char_back(compiler_common *common, sljit_u32 max, jump_list **backtracks)
{
/* Reads one character back without moving STR_PTR. TMP2 must
contain the start of the subject buffer. Affects TMP1, TMP2, and RETURN_ADDR. */
DEFINE_COMPILER;

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
struct sljit_jump *jump;
#endif /* SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32 */

SLJIT_UNUSED_ARG(max);
SLJIT_UNUSED_ARG(backtracks);

OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-1));

#ifdef SUPPORT_UNICODE
#if PCRE2_CODE_UNIT_WIDTH == 8
if (common->utf)
  {
  if (max < 128) return;

  jump = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0x80);
  if (common->invalid_utf)
    {
    add_jump(compiler, &common->utfpeakcharback_invalid, JUMP(SLJIT_FAST_CALL));
    if (backtracks != NULL)
      add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, INVALID_UTF_CHAR));
    }
  else
    add_jump(compiler, &common->utfpeakcharback, JUMP(SLJIT_FAST_CALL));
  JUMPHERE(jump);
  }
#elif PCRE2_CODE_UNIT_WIDTH == 16
if (common->utf)
  {
  if (max < 0xd800) return;

  if (common->invalid_utf)
    {
    jump = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0xd800);
    add_jump(compiler, &common->utfpeakcharback_invalid, JUMP(SLJIT_FAST_CALL));
    if (backtracks != NULL)
      add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, INVALID_UTF_CHAR));
    }
  else
    {
    OP2(SLJIT_SUB, TMP2, 0, TMP1, 0, SLJIT_IMM, 0xdc00);
    jump = CMP(SLJIT_GREATER_EQUAL, TMP2, 0, SLJIT_IMM, 0xe000 - 0xdc00);
    /* TMP2 contains the low surrogate. */
    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-2));
    OP2(SLJIT_ADD, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x10000);
    OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xd800);
    OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 10);
    OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, TMP2, 0);
    }
    JUMPHERE(jump);
  }
#elif PCRE2_CODE_UNIT_WIDTH == 32
if (common->invalid_utf)
  {
  OP2(SLJIT_SUB, TMP2, 0, TMP1, 0, SLJIT_IMM, 0xd800);
  add_jump(compiler, backtracks, CMP(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 0x110000));
  add_jump(compiler, backtracks, CMP(SLJIT_LESS, TMP2, 0, SLJIT_IMM, 0xe000 - 0xd800));
  }
#endif /* PCRE2_CODE_UNIT_WIDTH == [8|16|32] */
#endif /* SUPPORT_UNICODE */
}

#define READ_CHAR_UPDATE_STR_PTR 0x1
#define READ_CHAR_UTF8_NEWLINE 0x2
#define READ_CHAR_NEWLINE (READ_CHAR_UPDATE_STR_PTR | READ_CHAR_UTF8_NEWLINE)
#define READ_CHAR_VALID_UTF 0x4

static void read_char(compiler_common *common, sljit_u32 min, sljit_u32 max,
  jump_list **backtracks, sljit_u32 options)
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

SLJIT_UNUSED_ARG(min);
SLJIT_UNUSED_ARG(max);
SLJIT_UNUSED_ARG(backtracks);
SLJIT_UNUSED_ARG(options);
SLJIT_ASSERT(min <= max);

OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

#ifdef SUPPORT_UNICODE
#if PCRE2_CODE_UNIT_WIDTH == 8
if (common->utf)
  {
  if (max < 128 && !(options & READ_CHAR_UPDATE_STR_PTR)) return;

  if (common->invalid_utf && !(options & READ_CHAR_VALID_UTF))
    {
    jump = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0x80);

    if (options & READ_CHAR_UTF8_NEWLINE)
      add_jump(compiler, &common->utfreadnewline_invalid, JUMP(SLJIT_FAST_CALL));
    else
      add_jump(compiler, &common->utfreadchar_invalid, JUMP(SLJIT_FAST_CALL));

    if (backtracks != NULL)
      add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, INVALID_UTF_CHAR));
    JUMPHERE(jump);
    return;
    }

  jump = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0xc0);
  if (min >= 0x10000)
    {
    OP2(SLJIT_SUB, TMP2, 0, TMP1, 0, SLJIT_IMM, 0xf0);
    if (options & READ_CHAR_UPDATE_STR_PTR)
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
    if (!(options & READ_CHAR_UPDATE_STR_PTR))
      OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(3));
    OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 6);
    OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x3f);
    OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);
    JUMPHERE(jump2);
    if (options & READ_CHAR_UPDATE_STR_PTR)
      OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, RETURN_ADDR, 0);
    }
  else if (min >= 0x800 && max <= 0xffff)
    {
    OP2(SLJIT_SUB, TMP2, 0, TMP1, 0, SLJIT_IMM, 0xe0);
    if (options & READ_CHAR_UPDATE_STR_PTR)
      OP1(SLJIT_MOV_U8, RETURN_ADDR, 0, SLJIT_MEM1(TMP1), (sljit_sw)PRIV(utf8_table4) - 0xc0);
    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
    jump2 = CMP(SLJIT_GREATER, TMP2, 0, SLJIT_IMM, 0xf);
    OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, 6);
    OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x3f);
    OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);
    OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(1));
    if (!(options & READ_CHAR_UPDATE_STR_PTR))
      OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(2));
    OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 6);
    OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x3f);
    OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);
    JUMPHERE(jump2);
    if (options & READ_CHAR_UPDATE_STR_PTR)
      OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, RETURN_ADDR, 0);
    }
  else if (max >= 0x800)
    {
    add_jump(compiler, &common->utfreadchar, JUMP(SLJIT_FAST_CALL));
    }
  else if (max < 128)
    {
    OP1(SLJIT_MOV_U8, TMP2, 0, SLJIT_MEM1(TMP1), (sljit_sw)PRIV(utf8_table4) - 0xc0);
    OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP2, 0);
    }
  else
    {
    OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
    if (!(options & READ_CHAR_UPDATE_STR_PTR))
      OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
    else
      OP1(SLJIT_MOV_U8, RETURN_ADDR, 0, SLJIT_MEM1(TMP1), (sljit_sw)PRIV(utf8_table4) - 0xc0);
    OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x3f);
    OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 6);
    OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x3f);
    OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);
    if (options & READ_CHAR_UPDATE_STR_PTR)
      OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, RETURN_ADDR, 0);
    }
  JUMPHERE(jump);
  }
#elif PCRE2_CODE_UNIT_WIDTH == 16
if (common->utf)
  {
  if (max < 0xd800 && !(options & READ_CHAR_UPDATE_STR_PTR)) return;

  if (common->invalid_utf && !(options & READ_CHAR_VALID_UTF))
    {
    OP2(SLJIT_SUB, TMP2, 0, TMP1, 0, SLJIT_IMM, 0xd800);
    jump = CMP(SLJIT_GREATER_EQUAL, TMP2, 0, SLJIT_IMM, 0xe000 - 0xd800);

    if (options & READ_CHAR_UTF8_NEWLINE)
      add_jump(compiler, &common->utfreadnewline_invalid, JUMP(SLJIT_FAST_CALL));
    else
      add_jump(compiler, &common->utfreadchar_invalid, JUMP(SLJIT_FAST_CALL));

    if (backtracks != NULL)
      add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, INVALID_UTF_CHAR));
    JUMPHERE(jump);
    return;
    }

  if (max >= 0x10000)
    {
    OP2(SLJIT_SUB, TMP2, 0, TMP1, 0, SLJIT_IMM, 0xd800);
    jump = CMP(SLJIT_GREATER_EQUAL, TMP2, 0, SLJIT_IMM, 0xdc00 - 0xd800);
    /* TMP2 contains the high surrogate. */
    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
    OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, 10);
    OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
    OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x10000 - 0xdc00);
    OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, TMP2, 0);
    JUMPHERE(jump);
    return;
    }

  /* Skip low surrogate if necessary. */
  OP2(SLJIT_SUB, TMP2, 0, TMP1, 0, SLJIT_IMM, 0xd800);

  if (sljit_has_cpu_feature(SLJIT_HAS_CMOV) && !HAS_VIRTUAL_REGISTERS)
    {
    if (options & READ_CHAR_UPDATE_STR_PTR)
      OP2(SLJIT_ADD, RETURN_ADDR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
    OP2U(SLJIT_SUB | SLJIT_SET_LESS, TMP2, 0, SLJIT_IMM, 0x400);
    if (options & READ_CHAR_UPDATE_STR_PTR)
      SELECT(SLJIT_LESS, STR_PTR, RETURN_ADDR, 0, STR_PTR);
    if (max >= 0xd800)
      SELECT(SLJIT_LESS, TMP1, SLJIT_IMM, 0x10000, TMP1);
    }
  else
    {
    jump = CMP(SLJIT_GREATER_EQUAL, TMP2, 0, SLJIT_IMM, 0x400);
    if (options & READ_CHAR_UPDATE_STR_PTR)
      OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
    if (max >= 0xd800)
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, 0x10000);
    JUMPHERE(jump);
    }
  }
#elif PCRE2_CODE_UNIT_WIDTH == 32
if (common->invalid_utf)
  {
  if (backtracks != NULL)
    {
    OP2(SLJIT_SUB, TMP2, 0, TMP1, 0, SLJIT_IMM, 0xd800);
    add_jump(compiler, backtracks, CMP(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 0x110000));
    add_jump(compiler, backtracks, CMP(SLJIT_LESS, TMP2, 0, SLJIT_IMM, 0xe000 - 0xd800));
    }
  else
    {
    OP2(SLJIT_SUB, TMP2, 0, TMP1, 0, SLJIT_IMM, 0xd800);
    OP2U(SLJIT_SUB | SLJIT_SET_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 0x110000);
    SELECT(SLJIT_GREATER_EQUAL, TMP1, SLJIT_IMM, INVALID_UTF_CHAR, TMP1);
    OP2U(SLJIT_SUB | SLJIT_SET_LESS, TMP2, 0, SLJIT_IMM, 0xe000 - 0xd800);
    SELECT(SLJIT_LESS, TMP1, SLJIT_IMM, INVALID_UTF_CHAR, TMP1);
    }
  }
#endif /* PCRE2_CODE_UNIT_WIDTH == [8|16|32] */
#endif /* SUPPORT_UNICODE */
}

static void skip_valid_char(compiler_common *common)
{
DEFINE_COMPILER;
#if (defined SUPPORT_UNICODE) && (PCRE2_CODE_UNIT_WIDTH == 8 || PCRE2_CODE_UNIT_WIDTH == 16)
struct sljit_jump *jump;
#endif

#if (defined SUPPORT_UNICODE) && (PCRE2_CODE_UNIT_WIDTH == 8 || PCRE2_CODE_UNIT_WIDTH == 16)
  if (common->utf)
    {
    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), 0);
    OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
#if PCRE2_CODE_UNIT_WIDTH == 8
    jump = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0xc0);
    OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(TMP1), (sljit_sw)PRIV(utf8_table4) - 0xc0);
    OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);
#elif PCRE2_CODE_UNIT_WIDTH == 16
    jump = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0xd800);
    OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xfc00);
    OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, 0xd800);
    OP_FLAGS(SLJIT_MOV, TMP1, 0, SLJIT_EQUAL);
    OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 1);
    OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);
#endif /* PCRE2_CODE_UNIT_WIDTH == 8 */
    JUMPHERE(jump);
    return;
    }
#endif /* SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == [8|16] */
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
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

static void read_char7_type(compiler_common *common, jump_list **backtracks, BOOL negated)
{
/* Reads the precise character type of a character into TMP1, if the character
is less than 128. Otherwise it returns with zero. Does not check STR_END. The
full_read argument tells whether characters above max are accepted or not. */
DEFINE_COMPILER;
struct sljit_jump *jump;

SLJIT_ASSERT(common->utf);

OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), 0);
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

/* All values > 127 are zero in ctypes. */
OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(TMP2), common->ctypes);

if (negated)
  {
  jump = CMP(SLJIT_LESS, TMP2, 0, SLJIT_IMM, 0x80);

  if (common->invalid_utf)
    {
    OP1(SLJIT_MOV, TMP1, 0, TMP2, 0);
    add_jump(compiler, &common->utfreadchar_invalid, JUMP(SLJIT_FAST_CALL));
    add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, INVALID_UTF_CHAR));
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, 0);
    }
  else
    {
    OP1(SLJIT_MOV_U8, TMP2, 0, SLJIT_MEM1(TMP2), (sljit_sw)PRIV(utf8_table4) - 0xc0);
    OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP2, 0);
    }
  JUMPHERE(jump);
  }
}

#endif /* SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8 */

static void read_char8_type(compiler_common *common, jump_list **backtracks, BOOL negated)
{
/* Reads the character type into TMP1, updates STR_PTR. Does not check STR_END. */
DEFINE_COMPILER;
#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH != 8
struct sljit_jump *jump;
#endif
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
struct sljit_jump *jump2;
#endif

SLJIT_UNUSED_ARG(backtracks);
SLJIT_UNUSED_ARG(negated);

OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), 0);
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
if (common->utf)
  {
  /* The result of this read may be unused, but saves an "else" part. */
  OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(TMP2), common->ctypes);
  jump = CMP(SLJIT_LESS, TMP2, 0, SLJIT_IMM, 0x80);

  if (!negated)
    {
    if (common->invalid_utf)
      add_jump(compiler, backtracks, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));

    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
    OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
    OP2(SLJIT_SUB, TMP2, 0, TMP2, 0, SLJIT_IMM, 0xc2);
    if (common->invalid_utf)
      add_jump(compiler, backtracks, CMP(SLJIT_GREATER_EQUAL, TMP2, 0, SLJIT_IMM, 0xe0 - 0xc2));

    OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, 6);
    OP2(SLJIT_ADD, TMP2, 0, TMP2, 0, TMP1, 0);
    OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x80);
    if (common->invalid_utf)
      add_jump(compiler, backtracks, CMP(SLJIT_GREATER_EQUAL, TMP2, 0, SLJIT_IMM, 0x40));

    OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, 0);
    jump2 = CMP(SLJIT_GREATER, TMP2, 0, SLJIT_IMM, 255);
    OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(TMP2), common->ctypes);
    JUMPHERE(jump2);
    }
  else if (common->invalid_utf)
    {
    add_jump(compiler, &common->utfreadchar_invalid, JUMP(SLJIT_FAST_CALL));
    OP1(SLJIT_MOV, TMP2, 0, TMP1, 0);
    add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, INVALID_UTF_CHAR));

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

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 32
if (common->invalid_utf && negated)
  add_jump(compiler, backtracks, CMP(SLJIT_GREATER_EQUAL, TMP2, 0, SLJIT_IMM, 0x110000));
#endif /* SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 32 */

#if PCRE2_CODE_UNIT_WIDTH != 8
/* The ctypes array contains only 256 values. */
OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, 0);
jump = CMP(SLJIT_GREATER, TMP2, 0, SLJIT_IMM, 255);
#endif /* PCRE2_CODE_UNIT_WIDTH != 8 */
OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(TMP2), common->ctypes);
#if PCRE2_CODE_UNIT_WIDTH != 8
JUMPHERE(jump);
#endif /* PCRE2_CODE_UNIT_WIDTH != 8 */

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 16
if (common->utf && negated)
  {
  /* Skip low surrogate if necessary. */
  if (!common->invalid_utf)
    {
    OP2(SLJIT_SUB, TMP2, 0, TMP2, 0, SLJIT_IMM, 0xd800);

    if (sljit_has_cpu_feature(SLJIT_HAS_CMOV) && !HAS_VIRTUAL_REGISTERS)
      {
      OP2(SLJIT_ADD, RETURN_ADDR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
      OP2U(SLJIT_SUB | SLJIT_SET_LESS, TMP2, 0, SLJIT_IMM, 0x400);
      SELECT(SLJIT_LESS, STR_PTR, RETURN_ADDR, 0, STR_PTR);
      }
    else
      {
      jump = CMP(SLJIT_GREATER_EQUAL, TMP2, 0, SLJIT_IMM, 0x400);
      OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
      JUMPHERE(jump);
      }
    return;
    }

  OP2(SLJIT_SUB, TMP2, 0, TMP2, 0, SLJIT_IMM, 0xd800);
  jump = CMP(SLJIT_GREATER_EQUAL, TMP2, 0, SLJIT_IMM, 0xe000 - 0xd800);
  add_jump(compiler, backtracks, CMP(SLJIT_GREATER_EQUAL, TMP2, 0, SLJIT_IMM, 0x400));
  add_jump(compiler, backtracks, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));

  OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  OP2(SLJIT_SUB, TMP2, 0, TMP2, 0, SLJIT_IMM, 0xdc00);
  add_jump(compiler, backtracks, CMP(SLJIT_GREATER_EQUAL, TMP2, 0, SLJIT_IMM, 0x400));

  JUMPHERE(jump);
  return;
  }
#endif /* SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 16 */
}

static void move_back(compiler_common *common, jump_list **backtracks, BOOL must_be_valid)
{
/* Goes one character back. Affects STR_PTR and TMP1. If must_be_valid is TRUE,
TMP2 is not used. Otherwise TMP2 must contain the start of the subject buffer,
and it is destroyed. Does not modify STR_PTR for invalid character sequences. */
DEFINE_COMPILER;

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
struct sljit_jump *jump;
#endif

#ifdef SUPPORT_UNICODE
#if PCRE2_CODE_UNIT_WIDTH == 8
struct sljit_label *label;

if (common->utf)
  {
  if (!must_be_valid && common->invalid_utf)
    {
    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), -IN_UCHARS(1));
    OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
    jump = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0x80);
    add_jump(compiler, &common->utfmoveback_invalid, JUMP(SLJIT_FAST_CALL));
    if (backtracks != NULL)
      add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, 0));
    JUMPHERE(jump);
    return;
    }

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

  if (!must_be_valid && common->invalid_utf)
    {
    OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xd800);
    jump = CMP(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 0xe000 - 0xd800);
    add_jump(compiler, &common->utfmoveback_invalid, JUMP(SLJIT_FAST_CALL));
    if (backtracks != NULL)
      add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, 0));
    JUMPHERE(jump);
    return;
    }

  /* Skip low surrogate if necessary. */
  OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xfc00);
  OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, 0xdc00);
  OP_FLAGS(SLJIT_MOV, TMP1, 0, SLJIT_EQUAL);
  OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, UCHAR_SHIFT);
  OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, TMP1, 0);
  return;
  }
#elif PCRE2_CODE_UNIT_WIDTH == 32
if (common->invalid_utf && !must_be_valid)
  {
  OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), -IN_UCHARS(1));
  if (backtracks != NULL)
    {
    add_jump(compiler, backtracks, CMP(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 0x110000));
    OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
    return;
    }

  OP2U(SLJIT_SUB | SLJIT_SET_LESS, TMP1, 0, SLJIT_IMM, 0x110000);
  OP_FLAGS(SLJIT_MOV, TMP1, 0, SLJIT_LESS);
  OP2(SLJIT_SHL,  TMP1, 0, TMP1, 0, SLJIT_IMM, UCHAR_SHIFT);
  OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, TMP1, 0);
  return;
  }
#endif /* PCRE2_CODE_UNIT_WIDTH == [8|16|32] */
#endif /* SUPPORT_UNICODE */

SLJIT_UNUSED_ARG(backtracks);
SLJIT_UNUSED_ARG(must_be_valid);

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
  sljit_set_current_flags(compiler, SLJIT_SET_Z);
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
of the character (>= 0xc0). Return char value in TMP1. */
DEFINE_COMPILER;
struct sljit_jump *jump;

sljit_emit_op_dst(compiler, SLJIT_FAST_ENTER, RETURN_ADDR, 0);
OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 6);
OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x3f);
OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);

/* Searching for the first zero. */
OP2U(SLJIT_AND | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, 0x800);
jump = JUMP(SLJIT_NOT_ZERO);
/* Two byte sequence. */
OP2(SLJIT_XOR, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x3000);
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);

JUMPHERE(jump);
OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(1));
OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 6);
OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x3f);
OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);

OP2U(SLJIT_AND | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, 0x10000);
jump = JUMP(SLJIT_NOT_ZERO);
/* Three byte sequence. */
OP2(SLJIT_XOR, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xe0000);
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(2));
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);

/* Four byte sequence. */
JUMPHERE(jump);
OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(2));
OP2(SLJIT_XOR, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xf0000);
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(3));
OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 6);
OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x3f);
OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);
}

static void do_utfreadtype8(compiler_common *common)
{
/* Fast decoding a UTF-8 character type. TMP2 contains the first byte
of the character (>= 0xc0). Return value in TMP1. */
DEFINE_COMPILER;
struct sljit_jump *jump;
struct sljit_jump *compare;

sljit_emit_op_dst(compiler, SLJIT_FAST_ENTER, RETURN_ADDR, 0);

OP2U(SLJIT_AND | SLJIT_SET_Z, TMP2, 0, SLJIT_IMM, 0x20);
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
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);

JUMPHERE(compare);
OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, 0);
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);

/* We only have types for characters less than 256. */
JUMPHERE(jump);
OP1(SLJIT_MOV_U8, TMP2, 0, SLJIT_MEM1(TMP2), (sljit_sw)PRIV(utf8_table4) - 0xc0);
OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, 0);
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP2, 0);
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);
}

static void do_utfreadchar_invalid(compiler_common *common)
{
/* Slow decoding a UTF-8 character. TMP1 contains the first byte
of the character (>= 0xc0). Return char value in TMP1. STR_PTR is
undefined for invalid characters. */
DEFINE_COMPILER;
sljit_s32 i;
sljit_s32 has_cmov = sljit_has_cpu_feature(SLJIT_HAS_CMOV);
struct sljit_jump *jump;
struct sljit_jump *buffer_end_close;
struct sljit_label *three_byte_entry;
struct sljit_label *exit_invalid_label;
struct sljit_jump *exit_invalid[11];

sljit_emit_op_dst(compiler, SLJIT_FAST_ENTER, RETURN_ADDR, 0);

OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xc2);

/* Usually more than 3 characters remained in the subject buffer. */
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(3));

/* Not a valid start of a multi-byte sequence, no more bytes read. */
exit_invalid[0] = CMP(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 0xf5 - 0xc2);

buffer_end_close = CMP(SLJIT_GREATER, STR_PTR, 0, STR_END, 0);

OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-3));
OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 6);
/* If TMP2 is in 0x80-0xbf range, TMP1 is also increased by (0x2 << 6). */
OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, TMP2, 0);
OP2(SLJIT_SUB, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x80);
exit_invalid[1] = CMP(SLJIT_GREATER_EQUAL, TMP2, 0, SLJIT_IMM, 0x40);

OP2U(SLJIT_AND | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, 0x800);
jump = JUMP(SLJIT_NOT_ZERO);

OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(2));
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);

JUMPHERE(jump);

/* Three-byte sequence. */
OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-2));
OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 6);
OP2(SLJIT_SUB, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x80);
OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);
if (has_cmov)
  {
  OP2U(SLJIT_SUB | SLJIT_SET_GREATER_EQUAL, TMP2, 0, SLJIT_IMM, 0x40);
  SELECT(SLJIT_GREATER_EQUAL, TMP1, SLJIT_IMM, 0x20000, TMP1);
  exit_invalid[2] = NULL;
  }
else
  exit_invalid[2] = CMP(SLJIT_GREATER_EQUAL, TMP2, 0, SLJIT_IMM, 0x40);

OP2U(SLJIT_AND | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, 0x10000);
jump = JUMP(SLJIT_NOT_ZERO);

three_byte_entry = LABEL();

OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x2d800);
if (has_cmov)
  {
  OP2U(SLJIT_SUB | SLJIT_SET_LESS, TMP1, 0, SLJIT_IMM, 0x800);
  SELECT(SLJIT_LESS, TMP1, SLJIT_IMM, INVALID_UTF_CHAR - 0xd800, TMP1);
  exit_invalid[3] = NULL;
  }
else
  exit_invalid[3] = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0x800);
OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xd800);
OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

if (has_cmov)
  {
  OP2U(SLJIT_SUB | SLJIT_SET_LESS, TMP1, 0, SLJIT_IMM, 0x800);
  SELECT(SLJIT_LESS, TMP1, SLJIT_IMM, INVALID_UTF_CHAR, TMP1);
  exit_invalid[4] = NULL;
  }
else
  exit_invalid[4] = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0x800);
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);

JUMPHERE(jump);

/* Four-byte sequence. */
OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-1));
OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 6);
OP2(SLJIT_SUB, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x80);
OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);
if (has_cmov)
  {
  OP2U(SLJIT_SUB | SLJIT_SET_GREATER_EQUAL, TMP2, 0, SLJIT_IMM, 0x40);
  SELECT(SLJIT_GREATER_EQUAL, TMP1, SLJIT_IMM, 0, TMP1);
  exit_invalid[5] = NULL;
  }
else
  exit_invalid[5] = CMP(SLJIT_GREATER_EQUAL, TMP2, 0, SLJIT_IMM, 0x40);

OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xc10000);
if (has_cmov)
  {
  OP2U(SLJIT_SUB | SLJIT_SET_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 0x100000);
  SELECT(SLJIT_GREATER_EQUAL, TMP1, SLJIT_IMM, INVALID_UTF_CHAR - 0x10000, TMP1);
  exit_invalid[6] = NULL;
  }
else
  exit_invalid[6] = CMP(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 0x100000);

OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x10000);
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);

JUMPHERE(buffer_end_close);
OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(2));
exit_invalid[7] = CMP(SLJIT_GREATER, STR_PTR, 0, STR_END, 0);

/* Two-byte sequence. */
OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-1));
OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 6);
/* If TMP2 is in 0x80-0xbf range, TMP1 is also increased by (0x2 << 6). */
OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, TMP2, 0);
OP2(SLJIT_SUB, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x80);
exit_invalid[8] = CMP(SLJIT_GREATER_EQUAL, TMP2, 0, SLJIT_IMM, 0x40);

OP2U(SLJIT_AND | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, 0x800);
jump = JUMP(SLJIT_NOT_ZERO);

OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);

/* Three-byte sequence. */
JUMPHERE(jump);
exit_invalid[9] = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);

OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 6);
OP2(SLJIT_SUB, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x80);
OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);
if (has_cmov)
  {
  OP2U(SLJIT_SUB | SLJIT_SET_GREATER_EQUAL, TMP2, 0, SLJIT_IMM, 0x40);
  SELECT(SLJIT_GREATER_EQUAL, TMP1, SLJIT_IMM, INVALID_UTF_CHAR, TMP1);
  exit_invalid[10] = NULL;
  }
else
  exit_invalid[10] = CMP(SLJIT_GREATER_EQUAL, TMP2, 0, SLJIT_IMM, 0x40);

/* One will be substracted from STR_PTR later. */
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(2));

/* Four byte sequences are not possible. */
CMPTO(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0x30000, three_byte_entry);

exit_invalid_label = LABEL();
for (i = 0; i < 11; i++)
  sljit_set_label(exit_invalid[i], exit_invalid_label);

OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, INVALID_UTF_CHAR);
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);
}

static void do_utfreadnewline_invalid(compiler_common *common)
{
/* Slow decoding a UTF-8 character, specialized for newlines.
TMP1 contains the first byte of the character (>= 0xc0). Return
char value in TMP1. */
DEFINE_COMPILER;
struct sljit_label *loop;
struct sljit_label *skip_start;
struct sljit_label *three_byte_exit;
struct sljit_jump *jump[5];

sljit_emit_op_dst(compiler, SLJIT_FAST_ENTER, RETURN_ADDR, 0);

if (common->nltype != NLTYPE_ANY)
  {
  SLJIT_ASSERT(common->nltype != NLTYPE_FIXED || common->newline < 128);

  /* All newlines are ascii, just skip intermediate octets. */
  jump[0] = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);
  loop = LABEL();
  if (sljit_emit_mem_update(compiler, MOV_UCHAR | SLJIT_MEM_SUPP | SLJIT_MEM_POST, TMP2, SLJIT_MEM1(STR_PTR), IN_UCHARS(1)) == SLJIT_SUCCESS)
    sljit_emit_mem_update(compiler, MOV_UCHAR | SLJIT_MEM_POST, TMP2, SLJIT_MEM1(STR_PTR), IN_UCHARS(1));
  else
    {
    OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
    OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
    }

  OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, 0xc0);
  CMPTO(SLJIT_EQUAL, TMP2, 0, SLJIT_IMM, 0x80, loop);
  OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

  JUMPHERE(jump[0]);

  OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, INVALID_UTF_CHAR);
  OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);
  return;
  }

jump[0] = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);
OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

jump[1] = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, 0xc2);
jump[2] = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, 0xe2);

skip_start = LABEL();
OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, 0xc0);
jump[3] = CMP(SLJIT_NOT_EQUAL, TMP2, 0, SLJIT_IMM, 0x80);

/* Skip intermediate octets. */
loop = LABEL();
jump[4] = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);
OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, 0xc0);
CMPTO(SLJIT_EQUAL, TMP2, 0, SLJIT_IMM, 0x80, loop);

JUMPHERE(jump[3]);
OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

three_byte_exit = LABEL();
JUMPHERE(jump[0]);
JUMPHERE(jump[4]);

OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, INVALID_UTF_CHAR);
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);

/* Two byte long newline: 0x85. */
JUMPHERE(jump[1]);
CMPTO(SLJIT_NOT_EQUAL, TMP2, 0, SLJIT_IMM, 0x85, skip_start);

OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, 0x85);
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);

/* Three byte long newlines: 0x2028 and 0x2029. */
JUMPHERE(jump[2]);
CMPTO(SLJIT_NOT_EQUAL, TMP2, 0, SLJIT_IMM, 0x80, skip_start);
CMPTO(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0, three_byte_exit);

OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

OP2(SLJIT_SUB, TMP1, 0, TMP2, 0, SLJIT_IMM, 0x80);
CMPTO(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 0x40, skip_start);

OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, 0x2000);
OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);
}

static void do_utfmoveback_invalid(compiler_common *common)
{
/* Goes one character back. */
DEFINE_COMPILER;
sljit_s32 i;
struct sljit_jump *jump;
struct sljit_jump *buffer_start_close;
struct sljit_label *exit_ok_label;
struct sljit_label *exit_invalid_label;
struct sljit_jump *exit_invalid[7];

sljit_emit_op_dst(compiler, SLJIT_FAST_ENTER, RETURN_ADDR, 0);

OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(3));
exit_invalid[0] = CMP(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 0xc0);

/* Two-byte sequence. */
buffer_start_close = CMP(SLJIT_LESS, STR_PTR, 0, TMP2, 0);

OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(2));

OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xc0);
jump = CMP(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 0x20);

OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, 1);
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(2));
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);

/* Three-byte sequence. */
JUMPHERE(jump);
exit_invalid[1] = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, -0x40);

OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(1));

OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xe0);
jump = CMP(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 0x10);

OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, 1);
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);

/* Four-byte sequence. */
JUMPHERE(jump);
OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xe0 - 0x80);
exit_invalid[2] = CMP(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 0x40);

OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xf0);
exit_invalid[3] = CMP(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 0x05);

exit_ok_label = LABEL();
OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, 1);
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);

/* Two-byte sequence. */
JUMPHERE(buffer_start_close);
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(2));

exit_invalid[4] = CMP(SLJIT_LESS, STR_PTR, 0, TMP2, 0);

OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));

OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xc0);
CMPTO(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0x20, exit_ok_label);

/* Three-byte sequence. */
OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
exit_invalid[5] = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, -0x40);
exit_invalid[6] = CMP(SLJIT_LESS, STR_PTR, 0, TMP2, 0);

OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));

OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xe0);
CMPTO(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0x10, exit_ok_label);

/* Four-byte sequences are not possible. */

exit_invalid_label = LABEL();
sljit_set_label(exit_invalid[5], exit_invalid_label);
sljit_set_label(exit_invalid[6], exit_invalid_label);
OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, 0);
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(3));
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);

JUMPHERE(exit_invalid[4]);
/* -2 + 4 = 2 */
OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(2));

exit_invalid_label = LABEL();
for (i = 0; i < 4; i++)
  sljit_set_label(exit_invalid[i], exit_invalid_label);
OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, 0);
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(4));
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);
}

static void do_utfpeakcharback(compiler_common *common)
{
/* Peak a character back. Does not modify STR_PTR. */
DEFINE_COMPILER;
struct sljit_jump *jump[2];

sljit_emit_op_dst(compiler, SLJIT_FAST_ENTER, RETURN_ADDR, 0);

OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-2));
OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xc0);
jump[0] = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0x20);

OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-3));
OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xe0);
jump[1] = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0x10);

OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-4));
OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xe0 - 0x80);
OP2(SLJIT_SUB, TMP2, 0, TMP2, 0, SLJIT_IMM, 0xf0);
OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, 6);
OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);

JUMPHERE(jump[1]);
OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-2));
OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 6);
OP2(SLJIT_SUB, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x80);
OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);

JUMPHERE(jump[0]);
OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-1));
OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 6);
OP2(SLJIT_SUB, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x80);
OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);

OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);
}

static void do_utfpeakcharback_invalid(compiler_common *common)
{
/* Peak a character back. Does not modify STR_PTR. */
DEFINE_COMPILER;
sljit_s32 i;
sljit_s32 has_cmov = sljit_has_cpu_feature(SLJIT_HAS_CMOV);
struct sljit_jump *jump[2];
struct sljit_label *two_byte_entry;
struct sljit_label *three_byte_entry;
struct sljit_label *exit_invalid_label;
struct sljit_jump *exit_invalid[8];

sljit_emit_op_dst(compiler, SLJIT_FAST_ENTER, RETURN_ADDR, 0);

OP2(SLJIT_ADD, TMP2, 0, TMP2, 0, SLJIT_IMM, IN_UCHARS(3));
exit_invalid[0] = CMP(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 0xc0);
jump[0] = CMP(SLJIT_GREATER_EQUAL, TMP2, 0, STR_PTR, 0);

/* Two-byte sequence. */
OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-2));
OP2(SLJIT_SUB, TMP2, 0, TMP2, 0, SLJIT_IMM, 0xc2);
jump[1] = CMP(SLJIT_GREATER_EQUAL, TMP2, 0, SLJIT_IMM, 0x1e);

two_byte_entry = LABEL();
OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, 6);
/* If TMP1 is in 0x80-0xbf range, TMP1 is also increased by (0x2 << 6). */
OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, TMP2, 0);
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);

JUMPHERE(jump[1]);
OP2(SLJIT_ADD, TMP2, 0, TMP2, 0, SLJIT_IMM, 0xc2 - 0x80);
OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x80);
exit_invalid[1] = CMP(SLJIT_GREATER_EQUAL, TMP2, 0, SLJIT_IMM, 0x40);
OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, 6);
OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);

/* Three-byte sequence. */
OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-3));
OP2(SLJIT_SUB, TMP2, 0, TMP2, 0, SLJIT_IMM, 0xe0);
jump[1] = CMP(SLJIT_GREATER_EQUAL, TMP2, 0, SLJIT_IMM, 0x10);

three_byte_entry = LABEL();
OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, 12);
OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);

OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xd800);
if (has_cmov)
  {
  OP2U(SLJIT_SUB | SLJIT_SET_LESS, TMP1, 0, SLJIT_IMM, 0x800);
  SELECT(SLJIT_LESS, TMP1, SLJIT_IMM, -0xd800, TMP1);
  exit_invalid[2] = NULL;
  }
else
  exit_invalid[2] = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0x800);

OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xd800);
if (has_cmov)
  {
  OP2U(SLJIT_SUB | SLJIT_SET_LESS, TMP1, 0, SLJIT_IMM, 0x800);
  SELECT(SLJIT_LESS, TMP1, SLJIT_IMM, INVALID_UTF_CHAR, TMP1);
  exit_invalid[3] = NULL;
  }
else
  exit_invalid[3] = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0x800);

OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);

JUMPHERE(jump[1]);
OP2(SLJIT_ADD, TMP2, 0, TMP2, 0, SLJIT_IMM, 0xe0 - 0x80);
exit_invalid[4] = CMP(SLJIT_GREATER_EQUAL, TMP2, 0, SLJIT_IMM, 0x40);
OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, 12);
OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);

/* Four-byte sequence. */
OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-4));
OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x10000);
OP2(SLJIT_SUB, TMP2, 0, TMP2, 0, SLJIT_IMM, 0xf0);
OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, 18);
/* ADD is used instead of OR because of the SUB 0x10000 above. */
OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, TMP2, 0);

if (has_cmov)
  {
  OP2U(SLJIT_SUB | SLJIT_SET_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 0x100000);
  SELECT(SLJIT_GREATER_EQUAL, TMP1, SLJIT_IMM, INVALID_UTF_CHAR - 0x10000, TMP1);
  exit_invalid[5] = NULL;
  }
else
  exit_invalid[5] = CMP(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 0x100000);

OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x10000);
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);

JUMPHERE(jump[0]);
OP2(SLJIT_SUB, TMP2, 0, TMP2, 0, SLJIT_IMM, IN_UCHARS(1));
jump[0] = CMP(SLJIT_GREATER_EQUAL, TMP2, 0, STR_PTR, 0);

/* Two-byte sequence. */
OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-2));
OP2(SLJIT_SUB, TMP2, 0, TMP2, 0, SLJIT_IMM, 0xc2);
CMPTO(SLJIT_LESS, TMP2, 0, SLJIT_IMM, 0x1e, two_byte_entry);

OP2(SLJIT_ADD, TMP2, 0, TMP2, 0, SLJIT_IMM, 0xc2 - 0x80);
OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x80);
exit_invalid[6] = CMP(SLJIT_GREATER_EQUAL, TMP2, 0, SLJIT_IMM, 0x40);
OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, 6);
OP2(SLJIT_OR, TMP1, 0, TMP1, 0, TMP2, 0);

/* Three-byte sequence. */
OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-3));
OP2(SLJIT_SUB, TMP2, 0, TMP2, 0, SLJIT_IMM, 0xe0);
CMPTO(SLJIT_LESS, TMP2, 0, SLJIT_IMM, 0x10, three_byte_entry);

OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, INVALID_UTF_CHAR);
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);

JUMPHERE(jump[0]);
exit_invalid[7] = CMP(SLJIT_GREATER, TMP2, 0, STR_PTR, 0);

/* Two-byte sequence. */
OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-2));
OP2(SLJIT_SUB, TMP2, 0, TMP2, 0, SLJIT_IMM, 0xc2);
CMPTO(SLJIT_LESS, TMP2, 0, SLJIT_IMM, 0x1e, two_byte_entry);

exit_invalid_label = LABEL();
for (i = 0; i < 8; i++)
  sljit_set_label(exit_invalid[i], exit_invalid_label);

OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, INVALID_UTF_CHAR);
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);
}

#endif /* PCRE2_CODE_UNIT_WIDTH == 8 */

#if PCRE2_CODE_UNIT_WIDTH == 16

static void do_utfreadchar_invalid(compiler_common *common)
{
/* Slow decoding a UTF-16 character. TMP1 contains the first half
of the character (>= 0xd800). Return char value in TMP1. STR_PTR is
undefined for invalid characters. */
DEFINE_COMPILER;
struct sljit_jump *exit_invalid[3];

sljit_emit_op_dst(compiler, SLJIT_FAST_ENTER, RETURN_ADDR, 0);

/* TMP2 contains the high surrogate. */
exit_invalid[0] = CMP(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 0xdc00);
exit_invalid[1] = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);

OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, 10);
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xdc00);
OP2(SLJIT_ADD, TMP2, 0, TMP2, 0, SLJIT_IMM, 0x10000);
exit_invalid[2] = CMP(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 0x400);

OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, TMP2, 0);
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);

JUMPHERE(exit_invalid[0]);
JUMPHERE(exit_invalid[1]);
JUMPHERE(exit_invalid[2]);
OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, INVALID_UTF_CHAR);
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);
}

static void do_utfreadnewline_invalid(compiler_common *common)
{
/* Slow decoding a UTF-16 character, specialized for newlines.
TMP1 contains the first half of the character (>= 0xd800). Return
char value in TMP1. */

DEFINE_COMPILER;
struct sljit_jump *exit_invalid[2];

sljit_emit_op_dst(compiler, SLJIT_FAST_ENTER, RETURN_ADDR, 0);

/* TMP2 contains the high surrogate. */
exit_invalid[0] = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);

OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
exit_invalid[1] = CMP(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 0xdc00);

OP2(SLJIT_SUB, TMP2, 0, TMP2, 0, SLJIT_IMM, 0xdc00);
OP2U(SLJIT_SUB | SLJIT_SET_LESS, TMP2, 0, SLJIT_IMM, 0x400);
OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_LESS);
OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, 0x10000);
OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, UCHAR_SHIFT);
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP2, 0);

OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);

JUMPHERE(exit_invalid[0]);
JUMPHERE(exit_invalid[1]);
OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, INVALID_UTF_CHAR);
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);
}

static void do_utfmoveback_invalid(compiler_common *common)
{
/* Goes one character back. */
DEFINE_COMPILER;
struct sljit_jump *exit_invalid[3];

sljit_emit_op_dst(compiler, SLJIT_FAST_ENTER, RETURN_ADDR, 0);

exit_invalid[0] = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0x400);
exit_invalid[1] = CMP(SLJIT_GREATER_EQUAL, TMP2, 0, STR_PTR, 0);

OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-1));
OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xd800);
exit_invalid[2] = CMP(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 0x400);

OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, 1);
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);

JUMPHERE(exit_invalid[0]);
JUMPHERE(exit_invalid[1]);
JUMPHERE(exit_invalid[2]);

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, 0);
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);
}

static void do_utfpeakcharback_invalid(compiler_common *common)
{
/* Peak a character back. Does not modify STR_PTR. */
DEFINE_COMPILER;
struct sljit_jump *jump;
struct sljit_jump *exit_invalid[3];

sljit_emit_op_dst(compiler, SLJIT_FAST_ENTER, RETURN_ADDR, 0);

jump = CMP(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 0xe000);
OP2(SLJIT_ADD, TMP2, 0, TMP2, 0, SLJIT_IMM, IN_UCHARS(1));
exit_invalid[0] = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0xdc00);
exit_invalid[1] = CMP(SLJIT_GREATER_EQUAL, TMP2, 0, STR_PTR, 0);

OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-2));
OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x10000 - 0xdc00);
OP2(SLJIT_SUB, TMP2, 0, TMP2, 0, SLJIT_IMM, 0xd800);
exit_invalid[2] = CMP(SLJIT_GREATER_EQUAL, TMP2, 0, SLJIT_IMM, 0x400);
OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, 10);
OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, TMP2, 0);

JUMPHERE(jump);
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);

JUMPHERE(exit_invalid[0]);
JUMPHERE(exit_invalid[1]);
JUMPHERE(exit_invalid[2]);

OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, INVALID_UTF_CHAR);
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);
}

#endif /* PCRE2_CODE_UNIT_WIDTH == 16 */

/* UCD_BLOCK_SIZE must be 128 (see the assert below). */
#define UCD_BLOCK_MASK 127
#define UCD_BLOCK_SHIFT 7

static void do_getucd(compiler_common *common)
{
/* Search the UCD record for the character comes in TMP1.
Returns chartype in TMP1 and UCD offset in TMP2. */
DEFINE_COMPILER;
#if PCRE2_CODE_UNIT_WIDTH == 32
struct sljit_jump *jump;
#endif

#if defined SLJIT_DEBUG && SLJIT_DEBUG
/* dummy_ucd_record */
const ucd_record *record = GET_UCD(UNASSIGNED_UTF_CHAR);
SLJIT_ASSERT(record->script == ucp_Unknown && record->chartype == ucp_Cn && record->gbprop == ucp_gbOther);
SLJIT_ASSERT(record->caseset == 0 && record->other_case == 0);
#endif

SLJIT_ASSERT(UCD_BLOCK_SIZE == 128 && sizeof(ucd_record) == 12);

sljit_emit_op_dst(compiler, SLJIT_FAST_ENTER, RETURN_ADDR, 0);

#if PCRE2_CODE_UNIT_WIDTH == 32
if (!common->utf)
  {
  jump = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, MAX_UTF_CODE_POINT + 1);
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, UNASSIGNED_UTF_CHAR);
  JUMPHERE(jump);
  }
#endif

OP2(SLJIT_LSHR, TMP2, 0, TMP1, 0, SLJIT_IMM, UCD_BLOCK_SHIFT);
OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, 1);
OP1(SLJIT_MOV_U16, TMP2, 0, SLJIT_MEM1(TMP2), (sljit_sw)PRIV(ucd_stage1));
OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, UCD_BLOCK_MASK);
OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, UCD_BLOCK_SHIFT);
OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, TMP2, 0);
OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, (sljit_sw)PRIV(ucd_stage2));
OP1(SLJIT_MOV_U16, TMP2, 0, SLJIT_MEM2(TMP2, TMP1), 1);
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);
}

static void do_getucdtype(compiler_common *common)
{
/* Search the UCD record for the character comes in TMP1.
Returns chartype in TMP1 and UCD offset in TMP2. */
DEFINE_COMPILER;
#if PCRE2_CODE_UNIT_WIDTH == 32
struct sljit_jump *jump;
#endif

#if defined SLJIT_DEBUG && SLJIT_DEBUG
/* dummy_ucd_record */
const ucd_record *record = GET_UCD(UNASSIGNED_UTF_CHAR);
SLJIT_ASSERT(record->script == ucp_Unknown && record->chartype == ucp_Cn && record->gbprop == ucp_gbOther);
SLJIT_ASSERT(record->caseset == 0 && record->other_case == 0);
#endif

SLJIT_ASSERT(UCD_BLOCK_SIZE == 128 && sizeof(ucd_record) == 12);

sljit_emit_op_dst(compiler, SLJIT_FAST_ENTER, RETURN_ADDR, 0);

#if PCRE2_CODE_UNIT_WIDTH == 32
if (!common->utf)
  {
  jump = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, MAX_UTF_CODE_POINT + 1);
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, UNASSIGNED_UTF_CHAR);
  JUMPHERE(jump);
  }
#endif

OP2(SLJIT_LSHR, TMP2, 0, TMP1, 0, SLJIT_IMM, UCD_BLOCK_SHIFT);
OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, 1);
OP1(SLJIT_MOV_U16, TMP2, 0, SLJIT_MEM1(TMP2), (sljit_sw)PRIV(ucd_stage1));
OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, UCD_BLOCK_MASK);
OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, UCD_BLOCK_SHIFT);
OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, TMP2, 0);
OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, (sljit_sw)PRIV(ucd_stage2));
OP1(SLJIT_MOV_U16, TMP2, 0, SLJIT_MEM2(TMP2, TMP1), 1);

/* TMP2 is multiplied by 12. Same as (TMP2 << 2) + ((TMP2 << 2) << 1). */
OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, (sljit_sw)PRIV(ucd_records) + SLJIT_OFFSETOF(ucd_record, chartype));
OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, 2);
OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, TMP2, 0);
OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM2(TMP1, TMP2), 1);

OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);
}

#endif /* SUPPORT_UNICODE */

static SLJIT_INLINE struct sljit_label *mainloop_entry(compiler_common *common)
{
DEFINE_COMPILER;
struct sljit_label *mainloop;
struct sljit_label *newlinelabel = NULL;
struct sljit_jump *start;
struct sljit_jump *end = NULL;
struct sljit_jump *end2 = NULL;
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
struct sljit_label *loop;
struct sljit_jump *jump;
#endif /* SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32 */
jump_list *newline = NULL;
sljit_u32 overall_options = common->re->overall_options;
BOOL hascrorlf = (common->re->flags & PCRE2_HASCRORLF) != 0;
BOOL newlinecheck = FALSE;
BOOL readuchar = FALSE;

if (!(hascrorlf || (overall_options & PCRE2_FIRSTLINE) != 0)
    && (common->nltype == NLTYPE_ANY || common->nltype == NLTYPE_ANYCRLF || common->newline > 255))
  newlinecheck = TRUE;

SLJIT_ASSERT(common->abort_label == NULL);

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
    read_char(common, common->nlmin, common->nlmax, NULL, READ_CHAR_NEWLINE);
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

  if (HAS_VIRTUAL_REGISTERS)
    {
    OP1(SLJIT_MOV, TMP1, 0, ARGUMENTS, 0);
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, offset_limit));
    }
  else
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(ARGUMENTS), SLJIT_OFFSETOF(jit_arguments, offset_limit));

  OP1(SLJIT_MOV, TMP2, 0, STR_END, 0);
  end = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, (sljit_sw) PCRE2_UNSET);
  if (HAS_VIRTUAL_REGISTERS)
    OP1(SLJIT_MOV, TMP2, 0, ARGUMENTS, 0);
  else
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(ARGUMENTS), SLJIT_OFFSETOF(jit_arguments, begin));

#if PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32
  OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, UCHAR_SHIFT);
#endif /* PCRE2_CODE_UNIT_WIDTH == [16|32] */
  if (HAS_VIRTUAL_REGISTERS)
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(TMP2), SLJIT_OFFSETOF(jit_arguments, begin));

  OP2(SLJIT_ADD, TMP2, 0, TMP2, 0, TMP1, 0);
  end2 = CMP(SLJIT_LESS_EQUAL, TMP2, 0, STR_END, 0);
  OP1(SLJIT_MOV, TMP2, 0, STR_END, 0);
  JUMPHERE(end2);
  OP1(SLJIT_MOV, SLJIT_RETURN_REG, 0, SLJIT_IMM, PCRE2_ERROR_NOMATCH);
  add_jump(compiler, &common->abort, CMP(SLJIT_LESS, TMP2, 0, STR_PTR, 0));
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
  OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, common->newline & 0xff);
  OP_FLAGS(SLJIT_MOV, TMP1, 0, SLJIT_EQUAL);
#if PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32
  OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, UCHAR_SHIFT);
#endif /* PCRE2_CODE_UNIT_WIDTH == [16|32] */
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);
  end2 = JUMP(SLJIT_JUMP);
  }

mainloop = LABEL();

/* Increasing the STR_PTR here requires one less jump in the most common case. */
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
if (common->utf && !common->invalid_utf) readuchar = TRUE;
#endif /* SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32 */
if (newlinecheck) readuchar = TRUE;

if (readuchar)
  OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), 0);

if (newlinecheck)
  CMPTO(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, (common->newline >> 8) & 0xff, newlinelabel);

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
#if PCRE2_CODE_UNIT_WIDTH == 8
if (common->invalid_utf)
  {
  /* Skip continuation code units. */
  loop = LABEL();
  jump = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);
  OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), 0);
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x80);
  CMPTO(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0x40, loop);
  OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  JUMPHERE(jump);
  }
else if (common->utf)
  {
  jump = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0xc0);
  OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(TMP1), (sljit_sw)PRIV(utf8_table4) - 0xc0);
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);
  JUMPHERE(jump);
  }
#elif PCRE2_CODE_UNIT_WIDTH == 16
if (common->invalid_utf)
  {
  /* Skip continuation code units. */
  loop = LABEL();
  jump = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);
  OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), 0);
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xdc00);
  CMPTO(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0x400, loop);
  OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  JUMPHERE(jump);
  }
else if (common->utf)
  {
  OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0xd800);

  if (sljit_has_cpu_feature(SLJIT_HAS_CMOV))
    {
    OP2(SLJIT_ADD, TMP2, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
    OP2U(SLJIT_SUB | SLJIT_SET_LESS, TMP1, 0, SLJIT_IMM, 0x400);
    SELECT(SLJIT_LESS, STR_PTR, TMP2, 0, STR_PTR);
    }
  else
    {
    OP2U(SLJIT_SUB | SLJIT_SET_LESS, TMP1, 0, SLJIT_IMM, 0x400);
    OP_FLAGS(SLJIT_MOV, TMP1, 0, SLJIT_LESS);
    OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, UCHAR_SHIFT);
    OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);
    }
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


static SLJIT_INLINE void add_prefix_char(PCRE2_UCHAR chr, fast_forward_char_data *chars, BOOL last)
{
sljit_u32 i, count = chars->count;

if (count == 255)
  return;

if (count == 0)
  {
  chars->count = 1;
  chars->chars[0] = chr;

  if (last)
    chars->last_count = 1;
  return;
  }

for (i = 0; i < count; i++)
  if (chars->chars[i] == chr)
    return;

if (count >= MAX_DIFF_CHARS)
  {
  chars->count = 255;
  return;
  }

chars->chars[count] = chr;
chars->count = count + 1;

if (last)
  chars->last_count++;
}

/* Value can be increased if needed. Patterns
such as /(a|){33}b/ can exhaust the stack.

Note: /(a|){29}b/ already stops scan_prefix()
because it reaches the maximum step_count. */
#define SCAN_PREFIX_STACK_END 32

/*
Scan prefix stores the prefix string in the chars array.
The elements of the chars array is either small character
sets or "any" (count is set to 255).

Examples (the chars array is represented by a simple regex):

/(abc|xbyd)/ prefix: /[ax]b[cy]/ (length: 3)
/a[a-z]b+c/ prefix: a.b (length: 3)
/ab?cd/ prefix: a[bc][cd] (length: 3)
/(ab|cd)|(ef|gh)/ prefix: [aceg][bdfh] (length: 2)

The length is returned by scan_prefix(). The length is
less than or equal than the minimum length of the pattern.
*/

static int scan_prefix(compiler_common *common, PCRE2_SPTR cc, fast_forward_char_data *chars)
{
fast_forward_char_data *chars_start = chars;
fast_forward_char_data *chars_end = chars + MAX_N_CHARS;
PCRE2_SPTR cc_stack[SCAN_PREFIX_STACK_END];
fast_forward_char_data *chars_stack[SCAN_PREFIX_STACK_END];
sljit_u8 next_alternative_stack[SCAN_PREFIX_STACK_END];
BOOL last, any, class, caseless;
int stack_ptr, step_count, repeat, len, len_save;
sljit_u32 chr; /* Any unicode character. */
sljit_u8 *bytes, *bytes_end, byte;
PCRE2_SPTR alternative, cc_save, oc;
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
PCRE2_UCHAR othercase[4];
#elif defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 16
PCRE2_UCHAR othercase[2];
#else
PCRE2_UCHAR othercase[1];
#endif

repeat = 1;
stack_ptr = 0;
step_count = 10000;
while (TRUE)
  {
  if (--step_count == 0)
    return 0;

  SLJIT_ASSERT(chars <= chars_start + MAX_N_CHARS);

  if (chars >= chars_end)
    {
    if (stack_ptr == 0)
      return (int)(chars_end - chars_start);

    --stack_ptr;
    cc = cc_stack[stack_ptr];
    chars = chars_stack[stack_ptr];

    if (chars >= chars_end)
      continue;

    if (next_alternative_stack[stack_ptr] != 0)
      {
      /* When an alternative is processed, the
      next alternative is pushed onto the stack. */
      SLJIT_ASSERT(*cc == OP_ALT);
      alternative = cc + GET(cc, 1);
      if (*alternative == OP_ALT)
        {
        SLJIT_ASSERT(stack_ptr < SCAN_PREFIX_STACK_END);
        SLJIT_ASSERT(chars_stack[stack_ptr] == chars);
        SLJIT_ASSERT(next_alternative_stack[stack_ptr] == 1);
        cc_stack[stack_ptr] = alternative;
        stack_ptr++;
        }
      cc += 1 + LINK_SIZE;
      }
    }

  last = TRUE;
  any = FALSE;
  class = FALSE;
  caseless = FALSE;

  switch (*cc)
    {
    case OP_CHARI:
    caseless = TRUE;
    /* Fall through */
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
    case OP_NOT_UCP_WORD_BOUNDARY:
    case OP_UCP_WORD_BOUNDARY:
    /* Zero width assertions. */
    cc++;
    continue;

    case OP_ASSERT:
    case OP_ASSERT_NOT:
    case OP_ASSERTBACK:
    case OP_ASSERTBACK_NOT:
    case OP_ASSERT_NA:
    case OP_ASSERTBACK_NA:
    case OP_ASSERT_SCS:
    cc = bracketend(cc);
    continue;

    case OP_PLUSI:
    case OP_MINPLUSI:
    case OP_POSPLUSI:
    caseless = TRUE;
    /* Fall through */
    case OP_PLUS:
    case OP_MINPLUS:
    case OP_POSPLUS:
    cc++;
    break;

    case OP_EXACTI:
    caseless = TRUE;
    /* Fall through */
    case OP_EXACT:
    repeat = GET2(cc, 1);
    last = FALSE;
    cc += 1 + IMM2_SIZE;
    break;

    case OP_QUERYI:
    case OP_MINQUERYI:
    case OP_POSQUERYI:
    caseless = TRUE;
    /* Fall through */
    case OP_QUERY:
    case OP_MINQUERY:
    case OP_POSQUERY:
    len = 1;
    cc++;
#ifdef SUPPORT_UNICODE
    if (common->utf && HAS_EXTRALEN(*cc)) len += GET_EXTRALEN(*cc);
#endif
    if (stack_ptr >= SCAN_PREFIX_STACK_END)
      {
      chars_end = chars;
      continue;
      }

    cc_stack[stack_ptr] = cc + len;
    chars_stack[stack_ptr] = chars;
    next_alternative_stack[stack_ptr] = 0;
    stack_ptr++;

    last = FALSE;
    break;

    case OP_KET:
    cc += 1 + LINK_SIZE;
    continue;

    case OP_ALT:
    cc += GET(cc, 1);
    continue;

    case OP_ONCE:
    case OP_BRA:
    case OP_BRAPOS:
    case OP_CBRA:
    case OP_CBRAPOS:
    alternative = cc + GET(cc, 1);
    if (*alternative == OP_ALT)
      {
      if (stack_ptr >= SCAN_PREFIX_STACK_END)
        {
        chars_end = chars;
        continue;
        }

      cc_stack[stack_ptr] = alternative;
      chars_stack[stack_ptr] = chars;
      next_alternative_stack[stack_ptr] = 1;
      stack_ptr++;
      }

    if (*cc == OP_CBRA || *cc == OP_CBRAPOS)
      cc += IMM2_SIZE;
    cc += 1 + LINK_SIZE;
    continue;

    case OP_CLASS:
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
    if (common->utf && !is_char7_bitset((const sljit_u8 *)(cc + 1), FALSE))
      {
      chars_end = chars;
      continue;
      }
#endif
    class = TRUE;
    break;

    case OP_NCLASS:
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
    if (common->utf)
      {
      chars_end = chars;
      continue;
      }
#endif
    class = TRUE;
    break;

#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH != 8
    case OP_XCLASS:
    case OP_ECLASS:
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
    if (common->utf)
      {
      chars_end = chars;
      continue;
      }
#endif
    any = TRUE;
    cc += GET(cc, 1);
    break;
#endif

    case OP_DIGIT:
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
    if (common->utf && !is_char7_bitset((const sljit_u8 *)common->ctypes - cbit_length + cbit_digit, FALSE))
      {
      chars_end = chars;
      continue;
      }
#endif
    any = TRUE;
    cc++;
    break;

    case OP_WHITESPACE:
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
    if (common->utf && !is_char7_bitset((const sljit_u8 *)common->ctypes - cbit_length + cbit_space, FALSE))
      {
      chars_end = chars;
      continue;
      }
#endif
    any = TRUE;
    cc++;
    break;

    case OP_WORDCHAR:
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
    if (common->utf && !is_char7_bitset((const sljit_u8 *)common->ctypes - cbit_length + cbit_word, FALSE))
      {
      chars_end = chars;
      continue;
      }
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
    if (common->utf)
      {
      chars_end = chars;
      continue;
      }
#endif
    any = TRUE;
    cc++;
    break;

#ifdef SUPPORT_UNICODE
    case OP_NOTPROP:
    case OP_PROP:
#if PCRE2_CODE_UNIT_WIDTH != 32
    if (common->utf)
      {
      chars_end = chars;
      continue;
      }
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
    if (common->utf)
      {
      chars_end = chars;
      continue;
      }
#endif
    any = TRUE;
    repeat = GET2(cc, 1);
    cc += 1 + IMM2_SIZE + 1;
    break;

    default:
    chars_end = chars;
    continue;
    }

  SLJIT_ASSERT(chars < chars_end);

  if (any)
    {
    do
      {
      chars->count = 255;
      chars++;
      }
    while (--repeat > 0 && chars < chars_end);

    repeat = 1;
    continue;
    }

  if (class)
    {
    bytes = (sljit_u8*) (cc + 1);
    cc += 1 + 32 / sizeof(PCRE2_UCHAR);

    SLJIT_ASSERT(last == TRUE && repeat == 1);
    switch (*cc)
      {
      case OP_CRQUERY:
      case OP_CRMINQUERY:
      case OP_CRPOSQUERY:
      last = FALSE;
      /* Fall through */
      case OP_CRSTAR:
      case OP_CRMINSTAR:
      case OP_CRPOSSTAR:
      if (stack_ptr >= SCAN_PREFIX_STACK_END)
        {
        chars_end = chars;
        continue;
        }

      cc_stack[stack_ptr] = ++cc;
      chars_stack[stack_ptr] = chars;
      next_alternative_stack[stack_ptr] = 0;
      stack_ptr++;
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
        {
        chars_end = chars;
        continue;
        }

      last = (repeat != (int)GET2(cc, 1 + IMM2_SIZE));
      cc += 1 + 2 * IMM2_SIZE;
      break;
      }

    do
      {
      if (bytes[31] & 0x80)
        chars->count = 255;
      else if (chars->count != 255)
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
                add_prefix_char(chr, chars, TRUE);
              byte >>= 1;
              chr++;
              }
            while (byte != 0);
            chr = (chr + 7) & (sljit_u32)(~7);
            }
          }
        while (chars->count != 255 && bytes < bytes_end);
        bytes = bytes_end - 32;
        }

      chars++;
      }
    while (--repeat > 0 && chars < chars_end);

    repeat = 1;
    if (last)
      chars_end = chars;
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
        {
        chars_end = chars;
        continue;
        }
      }
    else
#endif
      {
      chr = *cc;
#ifdef SUPPORT_UNICODE
      if (common->ucp && chr > 127)
        {
        chr = UCD_OTHERCASE(chr);
        othercase[0] = (chr == (PCRE2_UCHAR)chr) ? chr : *cc;
        }
      else
#endif
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
      len--;

      chr = *cc;
      add_prefix_char(*cc, chars, len == 0);

      if (caseless)
        add_prefix_char(*oc, chars, len == 0);

      chars++;
      cc++;
      oc++;
      }
    while (len > 0 && chars < chars_end);

    if (--repeat == 0 || chars >= chars_end)
      break;

    len = len_save;
    cc = cc_save;
    }

  repeat = 1;
  if (last)
    chars_end = chars;
  }
}

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
static void jumpto_if_not_utf_char_start(struct sljit_compiler *compiler, sljit_s32 reg, struct sljit_label *label)
{
#if PCRE2_CODE_UNIT_WIDTH == 8
OP2(SLJIT_AND, reg, 0, reg, 0, SLJIT_IMM, 0xc0);
CMPTO(SLJIT_EQUAL, reg, 0, SLJIT_IMM, 0x80, label);
#elif PCRE2_CODE_UNIT_WIDTH == 16
OP2(SLJIT_AND, reg, 0, reg, 0, SLJIT_IMM, 0xfc00);
CMPTO(SLJIT_EQUAL, reg, 0, SLJIT_IMM, 0xdc00, label);
#else
#error "Unknown code width"
#endif
}
#endif

#include "pcre2_jit_simd_inc.h"

#ifdef JIT_HAS_FAST_FORWARD_CHAR_PAIR_SIMD

static BOOL check_fast_forward_char_pair_simd(compiler_common *common, fast_forward_char_data *chars, int max)
{
  sljit_s32 i, j, max_i = 0, max_j = 0;
  sljit_u32 max_pri = 0;
  sljit_s32 max_offset = max_fast_forward_char_pair_offset();
  PCRE2_UCHAR a1, a2, a_pri, b1, b2, b_pri;

  for (i = max - 1; i >= 1; i--)
    {
    if (chars[i].last_count > 2)
      {
      a1 = chars[i].chars[0];
      a2 = chars[i].chars[1];
      a_pri = chars[i].last_count;

      j = i - max_offset;
      if (j < 0)
        j = 0;

      while (j < i)
        {
        b_pri = chars[j].last_count;
        if (b_pri > 2 && (sljit_u32)a_pri + (sljit_u32)b_pri >= max_pri)
          {
          b1 = chars[j].chars[0];
          b2 = chars[j].chars[1];

          if (a1 != b1 && a1 != b2 && a2 != b1 && a2 != b2)
            {
            max_pri = a_pri + b_pri;
            max_i = i;
            max_j = j;
            }
          }
        j++;
        }
      }
    }

if (max_pri == 0)
  return FALSE;

fast_forward_char_pair_simd(common, max_i, chars[max_i].chars[0], chars[max_i].chars[1], max_j, chars[max_j].chars[0], chars[max_j].chars[1]);
return TRUE;
}

#endif /* JIT_HAS_FAST_FORWARD_CHAR_PAIR_SIMD */

static void fast_forward_first_char2(compiler_common *common, PCRE2_UCHAR char1, PCRE2_UCHAR char2, sljit_s32 offset)
{
DEFINE_COMPILER;
struct sljit_label *start;
struct sljit_jump *match;
struct sljit_jump *partial_quit;
PCRE2_UCHAR mask;
BOOL has_match_end = (common->match_end_ptr != 0);

SLJIT_ASSERT(common->mode == PCRE2_JIT_COMPLETE || offset == 0);

if (has_match_end)
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->match_end_ptr);

if (offset > 0)
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(offset));

if (has_match_end)
  {
  OP1(SLJIT_MOV, TMP3, 0, STR_END, 0);

  OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, IN_UCHARS(offset + 1));
  OP2U(SLJIT_SUB | SLJIT_SET_GREATER, STR_END, 0, TMP1, 0);
  SELECT(SLJIT_GREATER, STR_END, TMP1, 0, STR_END);
  }

#ifdef JIT_HAS_FAST_FORWARD_CHAR_SIMD

if (JIT_HAS_FAST_FORWARD_CHAR_SIMD)
  {
  fast_forward_char_simd(common, char1, char2, offset);

  if (offset > 0)
    OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(offset));

  if (has_match_end)
    OP1(SLJIT_MOV, STR_END, 0, TMP3, 0);
  return;
  }

#endif

start = LABEL();

partial_quit = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);
if (common->mode == PCRE2_JIT_COMPLETE)
  add_jump(compiler, &common->failed_match, partial_quit);

OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), 0);
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

if (char1 == char2)
  CMPTO(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, char1, start);
else
  {
  mask = char1 ^ char2;
  if (is_powerof2(mask))
    {
    OP2(SLJIT_OR, TMP1, 0, TMP1, 0, SLJIT_IMM, mask);
    CMPTO(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, char1 | mask, start);
    }
  else
    {
    match = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, char1);
    CMPTO(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, char2, start);
    JUMPHERE(match);
    }
  }

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
if (common->utf && offset > 0)
  {
  OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-(offset + 1)));
  jumpto_if_not_utf_char_start(compiler, TMP1, start);
  }
#endif

OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(offset + 1));

if (common->mode != PCRE2_JIT_COMPLETE)
  JUMPHERE(partial_quit);

if (has_match_end)
  OP1(SLJIT_MOV, STR_END, 0, TMP3, 0);
}

static SLJIT_INLINE BOOL fast_forward_first_n_chars(compiler_common *common)
{
DEFINE_COMPILER;
struct sljit_label *start;
struct sljit_jump *match;
fast_forward_char_data chars[MAX_N_CHARS];
sljit_s32 offset;
PCRE2_UCHAR mask;
PCRE2_UCHAR *char_set, *char_set_end;
int i, max, from;
int range_right = -1, range_len;
sljit_u8 *update_table = NULL;
BOOL in_range;

for (i = 0; i < MAX_N_CHARS; i++)
  {
  chars[i].count = 0;
  chars[i].last_count = 0;
  }

max = scan_prefix(common, common->start, chars);

if (max < 1)
  return FALSE;

/* Convert last_count to priority. */
for (i = 0; i < max; i++)
  {
  SLJIT_ASSERT(chars[i].last_count <= chars[i].count);

  switch (chars[i].count)
    {
    case 0:
    chars[i].count = 255;
    chars[i].last_count = 0;
    break;

    case 1:
    chars[i].last_count = (chars[i].last_count == 1) ? 7 : 5;
    /* Simplifies algorithms later. */
    chars[i].chars[1] = chars[i].chars[0];
    break;

    case 2:
    SLJIT_ASSERT(chars[i].chars[0] != chars[i].chars[1]);

    if (is_powerof2(chars[i].chars[0] ^ chars[i].chars[1]))
      chars[i].last_count = (chars[i].last_count == 2) ? 6 : 4;
    else
      chars[i].last_count = (chars[i].last_count == 2) ? 3 : 2;
    break;

    default:
    chars[i].last_count = (chars[i].count == 255) ? 0 : 1;
    break;
    }
  }

#ifdef JIT_HAS_FAST_FORWARD_CHAR_PAIR_SIMD
if (JIT_HAS_FAST_FORWARD_CHAR_PAIR_SIMD && check_fast_forward_char_pair_simd(common, chars, max))
  return TRUE;
#endif

in_range = FALSE;
/* Prevent compiler "uninitialized" warning */
from = 0;
range_len = 4 /* minimum length */ - 1;
for (i = 0; i <= max; i++)
  {
  if (in_range && (i - from) > range_len && (chars[i - 1].count < 255))
    {
    range_len = i - from;
    range_right = i - 1;
    }

  if (i < max && chars[i].count < 255)
    {
    SLJIT_ASSERT(chars[i].count > 0);
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
    SLJIT_ASSERT(chars[range_right - i].count > 0 && chars[range_right - i].count < 255);

    char_set = chars[range_right - i].chars;
    char_set_end = char_set + chars[range_right - i].count;
    do
      {
      if (update_table[(*char_set) & 0xff] > IN_UCHARS(i))
        update_table[(*char_set) & 0xff] = IN_UCHARS(i);
      char_set++;
      }
    while (char_set < char_set_end);
    }
  }

offset = -1;
/* Scan forward. */
for (i = 0; i < max; i++)
  {
  if (range_right == i)
    continue;

  if (offset == -1)
    {
    if (chars[i].last_count >= 2)
      offset = i;
    }
  else if (chars[offset].last_count < chars[i].last_count)
    offset = i;
  }

SLJIT_ASSERT(offset == -1 || (chars[offset].count >= 1 && chars[offset].count <= 2));

if (range_right < 0)
  {
  if (offset < 0)
    return FALSE;
  /* Works regardless the value is 1 or 2. */
  fast_forward_first_char2(common, chars[offset].chars[0], chars[offset].chars[1], offset);
  return TRUE;
  }

SLJIT_ASSERT(range_right != offset);

if (common->match_end_ptr != 0)
  {
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->match_end_ptr);
  OP1(SLJIT_MOV, TMP3, 0, STR_END, 0);
  OP2(SLJIT_SUB | SLJIT_SET_LESS, STR_END, 0, STR_END, 0, SLJIT_IMM, IN_UCHARS(max));
  add_jump(compiler, &common->failed_match, JUMP(SLJIT_LESS));
  OP2U(SLJIT_SUB | SLJIT_SET_GREATER, STR_END, 0, TMP1, 0);
  SELECT(SLJIT_GREATER, STR_END, TMP1, 0, STR_END);
  }
else
  {
  OP2(SLJIT_SUB | SLJIT_SET_LESS, STR_END, 0, STR_END, 0, SLJIT_IMM, IN_UCHARS(max));
  add_jump(compiler, &common->failed_match, JUMP(SLJIT_LESS));
  }

SLJIT_ASSERT(range_right >= 0);

if (!HAS_VIRTUAL_REGISTERS)
  OP1(SLJIT_MOV, RETURN_ADDR, 0, SLJIT_IMM, (sljit_sw)update_table);

start = LABEL();
add_jump(compiler, &common->failed_match, CMP(SLJIT_GREATER, STR_PTR, 0, STR_END, 0));

#if PCRE2_CODE_UNIT_WIDTH == 8 || (defined SLJIT_LITTLE_ENDIAN && SLJIT_LITTLE_ENDIAN)
OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(range_right));
#else
OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(range_right + 1) - 1);
#endif

if (!HAS_VIRTUAL_REGISTERS)
  OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM2(RETURN_ADDR, TMP1), 0);
else
  OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(TMP1), (sljit_sw)update_table);

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);
CMPTO(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, 0, start);

if (offset >= 0)
  {
  OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(offset));
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

  if (chars[offset].count == 1)
    CMPTO(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, chars[offset].chars[0], start);
  else
    {
    mask = chars[offset].chars[0] ^ chars[offset].chars[1];
    if (is_powerof2(mask))
      {
      OP2(SLJIT_OR, TMP1, 0, TMP1, 0, SLJIT_IMM, mask);
      CMPTO(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, chars[offset].chars[0] | mask, start);
      }
    else
      {
      match = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, chars[offset].chars[0]);
      CMPTO(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, chars[offset].chars[1], start);
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

  jumpto_if_not_utf_char_start(compiler, TMP1, start);

  if (offset < 0)
    OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  }
#endif

if (offset >= 0)
  OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

if (common->match_end_ptr != 0)
  OP1(SLJIT_MOV, STR_END, 0, TMP3, 0);
else
  OP2(SLJIT_ADD, STR_END, 0, STR_END, 0, SLJIT_IMM, IN_UCHARS(max));
return TRUE;
}

static SLJIT_INLINE void fast_forward_first_char(compiler_common *common)
{
PCRE2_UCHAR first_char = (PCRE2_UCHAR)(common->re->first_codeunit);
PCRE2_UCHAR oc;

oc = first_char;
if ((common->re->flags & PCRE2_FIRSTCASELESS) != 0)
  {
  oc = TABLE_GET(first_char, common->fcc, first_char);
#if defined SUPPORT_UNICODE
  if (first_char > 127 && (common->utf || common->ucp))
    oc = UCD_OTHERCASE(first_char);
#endif
  }

fast_forward_first_char2(common, first_char, oc, 0);
}

static SLJIT_INLINE void fast_forward_newline(compiler_common *common)
{
DEFINE_COMPILER;
struct sljit_label *loop;
struct sljit_jump *lastchar = NULL;
struct sljit_jump *firstchar;
struct sljit_jump *quit = NULL;
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
#ifdef JIT_HAS_FAST_FORWARD_CHAR_PAIR_SIMD
  if (JIT_HAS_FAST_FORWARD_CHAR_PAIR_SIMD && common->mode == PCRE2_JIT_COMPLETE)
    {
    if (HAS_VIRTUAL_REGISTERS)
      {
      OP1(SLJIT_MOV, TMP1, 0, ARGUMENTS, 0);
      OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, str));
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, begin));
      }
    else
      {
      OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(ARGUMENTS), SLJIT_OFFSETOF(jit_arguments, str));
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(ARGUMENTS), SLJIT_OFFSETOF(jit_arguments, begin));
      }
    firstchar = CMP(SLJIT_LESS_EQUAL, STR_PTR, 0, TMP2, 0);

    OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
    OP2U(SLJIT_SUB | SLJIT_SET_Z, STR_PTR, 0, TMP1, 0);
    OP_FLAGS(SLJIT_MOV, TMP1, 0, SLJIT_NOT_EQUAL);
#if PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32
    OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, UCHAR_SHIFT);
#endif
    OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, TMP1, 0);

    fast_forward_char_pair_simd(common, 1, common->newline & 0xff, common->newline & 0xff, 0, (common->newline >> 8) & 0xff, (common->newline >> 8) & 0xff);
    OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(2));
    }
  else
#endif /* JIT_HAS_FAST_FORWARD_CHAR_PAIR_SIMD */
    {
    lastchar = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);
    if (HAS_VIRTUAL_REGISTERS)
      {
      OP1(SLJIT_MOV, TMP1, 0, ARGUMENTS, 0);
      OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, str));
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, begin));
      }
    else
      {
      OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(ARGUMENTS), SLJIT_OFFSETOF(jit_arguments, str));
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(ARGUMENTS), SLJIT_OFFSETOF(jit_arguments, begin));
      }
    firstchar = CMP(SLJIT_LESS_EQUAL, STR_PTR, 0, TMP2, 0);

    OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, IN_UCHARS(2));
    OP2U(SLJIT_SUB | SLJIT_SET_GREATER_EQUAL, STR_PTR, 0, TMP1, 0);
    OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_GREATER_EQUAL);
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
    JUMPHERE(lastchar);
    }

  JUMPHERE(firstchar);

  if (common->match_end_ptr != 0)
    OP1(SLJIT_MOV, STR_END, 0, TMP3, 0);
  return;
  }

if (HAS_VIRTUAL_REGISTERS)
  {
  OP1(SLJIT_MOV, TMP1, 0, ARGUMENTS, 0);
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, str));
  }
else
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(ARGUMENTS), SLJIT_OFFSETOF(jit_arguments, str));

/* Example: match /^/ to \r\n from offset 1. */
firstchar = CMP(SLJIT_LESS_EQUAL, STR_PTR, 0, TMP2, 0);

if (common->nltype == NLTYPE_ANY)
  move_back(common, NULL, FALSE);
else
  OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

loop = LABEL();
common->ff_newline_shortcut = loop;

#ifdef JIT_HAS_FAST_FORWARD_CHAR_SIMD
if (JIT_HAS_FAST_FORWARD_CHAR_SIMD && (common->nltype == NLTYPE_FIXED || common->nltype == NLTYPE_ANYCRLF))
  {
  if (common->nltype == NLTYPE_ANYCRLF)
    {
    fast_forward_char_simd(common, CHAR_CR, CHAR_LF, 0);
    if (common->mode != PCRE2_JIT_COMPLETE)
      lastchar = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);

    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), 0);
    OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
    quit = CMP(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, CHAR_CR);
    }
   else
    {
    fast_forward_char_simd(common, common->newline, common->newline, 0);

    OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
    if (common->mode != PCRE2_JIT_COMPLETE)
      {
      OP2U(SLJIT_SUB | SLJIT_SET_GREATER, STR_PTR, 0, STR_END, 0);
      SELECT(SLJIT_GREATER, STR_PTR, STR_END, 0, STR_PTR);
      }
    }
  }
else
#endif /* JIT_HAS_FAST_FORWARD_CHAR_SIMD */
  {
  read_char(common, common->nlmin, common->nlmax, NULL, READ_CHAR_NEWLINE);
  lastchar = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);
  if (common->nltype == NLTYPE_ANY || common->nltype == NLTYPE_ANYCRLF)
    foundcr = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, CHAR_CR);
  check_newlinechar(common, common->nltype, &newline, FALSE);
  set_jumps(newline, loop);
  }

if (common->nltype == NLTYPE_ANY || common->nltype == NLTYPE_ANYCRLF)
  {
  if (quit == NULL)
    {
    quit = JUMP(SLJIT_JUMP);
    JUMPHERE(foundcr);
    }

  notfoundnl = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);
  OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), 0);
  OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, CHAR_NL);
  OP_FLAGS(SLJIT_MOV, TMP1, 0, SLJIT_EQUAL);
#if PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32
  OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, UCHAR_SHIFT);
#endif
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);
  JUMPHERE(notfoundnl);
  JUMPHERE(quit);
  }

if (lastchar)
  JUMPHERE(lastchar);
JUMPHERE(firstchar);

if (common->match_end_ptr != 0)
  OP1(SLJIT_MOV, STR_END, 0, TMP3, 0);
}

static BOOL optimize_class(compiler_common *common, const sljit_u8 *bits, BOOL nclass, BOOL invert, jump_list **backtracks);

static SLJIT_INLINE void fast_forward_start_bits(compiler_common *common)
{
DEFINE_COMPILER;
const sljit_u8 *start_bits = common->re->start_bitmap;
struct sljit_label *start;
struct sljit_jump *partial_quit;
#if PCRE2_CODE_UNIT_WIDTH != 8
struct sljit_jump *found = NULL;
#endif
jump_list *matches = NULL;

if (common->match_end_ptr != 0)
  {
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->match_end_ptr);
  OP1(SLJIT_MOV, RETURN_ADDR, 0, STR_END, 0);
  OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, IN_UCHARS(1));
  OP2U(SLJIT_SUB | SLJIT_SET_GREATER, STR_END, 0, TMP1, 0);
  SELECT(SLJIT_GREATER, STR_END, TMP1, 0, STR_END);
  }

start = LABEL();

partial_quit = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);
if (common->mode == PCRE2_JIT_COMPLETE)
  add_jump(compiler, &common->failed_match, partial_quit);

OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), 0);
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

if (!optimize_class(common, start_bits, (start_bits[31] & 0x80) != 0, FALSE, &matches))
  {
#if PCRE2_CODE_UNIT_WIDTH != 8
  if ((start_bits[31] & 0x80) != 0)
    found = CMP(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 255);
  else
    CMPTO(SLJIT_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 255, start);
#elif defined SUPPORT_UNICODE
  if (common->utf && is_char7_bitset(start_bits, FALSE))
    CMPTO(SLJIT_GREATER, TMP1, 0, SLJIT_IMM, 127, start);
#endif
  OP2(SLJIT_AND, TMP2, 0, TMP1, 0, SLJIT_IMM, 0x7);
  OP2(SLJIT_LSHR, TMP1, 0, TMP1, 0, SLJIT_IMM, 3);
  OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(TMP1), (sljit_sw)start_bits);
  if (!HAS_VIRTUAL_REGISTERS)
    {
    OP2(SLJIT_SHL, TMP3, 0, SLJIT_IMM, 1, TMP2, 0);
    OP2U(SLJIT_AND | SLJIT_SET_Z, TMP1, 0, TMP3, 0);
    }
  else
    {
    OP2(SLJIT_SHL, TMP2, 0, SLJIT_IMM, 1, TMP2, 0);
    OP2U(SLJIT_AND | SLJIT_SET_Z, TMP1, 0, TMP2, 0);
    }
  JUMPTO(SLJIT_ZERO, start);
  }
else
  set_jumps(matches, start);

#if PCRE2_CODE_UNIT_WIDTH != 8
if (found != NULL)
  JUMPHERE(found);
#endif

OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

if (common->mode != PCRE2_JIT_COMPLETE)
  JUMPHERE(partial_quit);

if (common->match_end_ptr != 0)
  OP1(SLJIT_MOV, STR_END, 0, RETURN_ADDR, 0);
}

static SLJIT_INLINE jump_list *search_requested_char(compiler_common *common, PCRE2_UCHAR req_char, BOOL caseless, BOOL has_firstchar)
{
DEFINE_COMPILER;
struct sljit_label *loop;
struct sljit_jump *toolong;
struct sljit_jump *already_found;
struct sljit_jump *found;
struct sljit_jump *found_oc = NULL;
jump_list *not_found = NULL;
sljit_u32 oc, bit;

SLJIT_ASSERT(common->req_char_ptr != 0);
OP2(SLJIT_ADD, TMP2, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(REQ_CU_MAX) * 100);
OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->req_char_ptr);
toolong = CMP(SLJIT_LESS, TMP2, 0, STR_END, 0);
already_found = CMP(SLJIT_LESS, STR_PTR, 0, TMP1, 0);

if (has_firstchar)
  OP2(SLJIT_ADD, TMP1, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
else
  OP1(SLJIT_MOV, TMP1, 0, STR_PTR, 0);

oc = req_char;
if (caseless)
  {
  oc = TABLE_GET(req_char, common->fcc, req_char);
#if defined SUPPORT_UNICODE
  if (req_char > 127 && (common->utf || common->ucp))
    oc = UCD_OTHERCASE(req_char);
#endif
  }

#ifdef JIT_HAS_FAST_REQUESTED_CHAR_SIMD
if (JIT_HAS_FAST_REQUESTED_CHAR_SIMD)
  {
  not_found = fast_requested_char_simd(common, req_char, oc);
  }
else
#endif
  {
  loop = LABEL();
  add_jump(compiler, &not_found, CMP(SLJIT_GREATER_EQUAL, TMP1, 0, STR_END, 0));

  OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(TMP1), 0);

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
      found_oc = CMP(SLJIT_EQUAL, TMP2, 0, SLJIT_IMM, oc);
      }
    }
  OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, IN_UCHARS(1));
  JUMPTO(SLJIT_JUMP, loop);

  JUMPHERE(found);
  if (found_oc)
    JUMPHERE(found_oc);
  }

OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->req_char_ptr, TMP1, 0);

JUMPHERE(already_found);
JUMPHERE(toolong);
return not_found;
}

static void do_revertframes(compiler_common *common)
{
DEFINE_COMPILER;
struct sljit_jump *jump;
struct sljit_label *mainloop;

sljit_emit_op_dst(compiler, SLJIT_FAST_ENTER, RETURN_ADDR, 0);
GET_LOCAL_BASE(TMP1, 0, 0);

/* Drop frames until we reach STACK_TOP. */
mainloop = LABEL();
OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(STACK_TOP), -SSIZE_OF(sw));
OP2U(SLJIT_SUB | SLJIT_SET_SIG_LESS_EQUAL | SLJIT_SET_Z, TMP2, 0, SLJIT_IMM, 0);
jump = JUMP(SLJIT_SIG_LESS_EQUAL);

OP2(SLJIT_ADD, TMP2, 0, TMP2, 0, TMP1, 0);
if (HAS_VIRTUAL_REGISTERS)
  {
  OP1(SLJIT_MOV, SLJIT_MEM1(TMP2), 0, SLJIT_MEM1(STACK_TOP), -(2 * SSIZE_OF(sw)));
  OP1(SLJIT_MOV, SLJIT_MEM1(TMP2), sizeof(sljit_sw), SLJIT_MEM1(STACK_TOP), -(3 * SSIZE_OF(sw)));
  OP2(SLJIT_SUB, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, 3 * SSIZE_OF(sw));
  }
else
  {
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), -(2 * SSIZE_OF(sw)));
  OP1(SLJIT_MOV, TMP3, 0, SLJIT_MEM1(STACK_TOP), -(3 * SSIZE_OF(sw)));
  OP2(SLJIT_SUB, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, 3 * SSIZE_OF(sw));
  OP1(SLJIT_MOV, SLJIT_MEM1(TMP2), 0, TMP1, 0);
  GET_LOCAL_BASE(TMP1, 0, 0);
  OP1(SLJIT_MOV, SLJIT_MEM1(TMP2), sizeof(sljit_sw), TMP3, 0);
  }
JUMPTO(SLJIT_JUMP, mainloop);

JUMPHERE(jump);
sljit_set_current_flags(compiler, SLJIT_CURRENT_FLAGS_SUB | SLJIT_CURRENT_FLAGS_COMPARE | SLJIT_SET_SIG_LESS_EQUAL | SLJIT_SET_Z);
jump = JUMP(SLJIT_NOT_ZERO /* SIG_LESS */);
/* End of reverting values. */
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);

JUMPHERE(jump);
OP2(SLJIT_SUB, TMP2, 0, TMP1, 0, TMP2, 0);
if (HAS_VIRTUAL_REGISTERS)
  {
  OP1(SLJIT_MOV, SLJIT_MEM1(TMP2), 0, SLJIT_MEM1(STACK_TOP), -(2 * SSIZE_OF(sw)));
  OP2(SLJIT_SUB, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, 2 * SSIZE_OF(sw));
  }
else
  {
  OP1(SLJIT_MOV, TMP3, 0, SLJIT_MEM1(STACK_TOP), -(2 * SSIZE_OF(sw)));
  OP2(SLJIT_SUB, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, 2 * SSIZE_OF(sw));
  OP1(SLJIT_MOV, SLJIT_MEM1(TMP2), 0, TMP3, 0);
  }
JUMPTO(SLJIT_JUMP, mainloop);
}

#ifdef SUPPORT_UNICODE
#define UCPCAT(bit) (1 << (bit))
#define UCPCAT2(bit1, bit2) (UCPCAT(bit1) | UCPCAT(bit2))
#define UCPCAT3(bit1, bit2, bit3) (UCPCAT(bit1) | UCPCAT(bit2) | UCPCAT(bit3))
#define UCPCAT_RANGE(start, end) (((1 << ((end) + 1)) - 1) - ((1 << (start)) - 1))
#define UCPCAT_L UCPCAT_RANGE(ucp_Ll, ucp_Lu)
#define UCPCAT_N UCPCAT_RANGE(ucp_Nd, ucp_No)
#define UCPCAT_ALL ((1 << (ucp_Zs + 1)) - 1)
#endif

static void check_wordboundary(compiler_common *common, BOOL ucp)
{
DEFINE_COMPILER;
struct sljit_jump *skipread;
jump_list *skipread_list = NULL;
#ifdef SUPPORT_UNICODE
struct sljit_label *valid_utf;
jump_list *invalid_utf1 = NULL;
#endif /* SUPPORT_UNICODE */
jump_list *invalid_utf2 = NULL;
#if PCRE2_CODE_UNIT_WIDTH != 8 || defined SUPPORT_UNICODE
struct sljit_jump *jump;
#endif /* PCRE2_CODE_UNIT_WIDTH != 8 || SUPPORT_UNICODE */

SLJIT_UNUSED_ARG(ucp);
SLJIT_COMPILE_ASSERT(ctype_word == 0x10, ctype_word_must_be_16);

SLJIT_ASSERT(common->locals_size >= 2 * SSIZE_OF(sw));
sljit_emit_op_dst(compiler, SLJIT_FAST_ENTER, SLJIT_MEM1(SLJIT_SP), LOCAL0);
/* Get type of the previous char, and put it to TMP3. */
OP1(SLJIT_MOV, TMP1, 0, ARGUMENTS, 0);
OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, begin));
OP1(SLJIT_MOV, TMP3, 0, SLJIT_IMM, 0);
skipread = CMP(SLJIT_LESS_EQUAL, STR_PTR, 0, TMP2, 0);

#ifdef SUPPORT_UNICODE
if (common->invalid_utf)
  {
  peek_char_back(common, READ_CHAR_MAX, &invalid_utf1);

  if (common->mode != PCRE2_JIT_COMPLETE)
    {
    OP1(SLJIT_MOV, RETURN_ADDR, 0, TMP1, 0);
    OP1(SLJIT_MOV, TMP2, 0, STR_PTR, 0);
    move_back(common, NULL, TRUE);
    check_start_used_ptr(common);
    OP1(SLJIT_MOV, TMP1, 0, RETURN_ADDR, 0);
    OP1(SLJIT_MOV, STR_PTR, 0, TMP2, 0);
    }
  }
else
#endif /* SUPPORT_UNICODE */
  {
  if (common->mode == PCRE2_JIT_COMPLETE)
    peek_char_back(common, READ_CHAR_MAX, NULL);
  else
    {
    move_back(common, NULL, TRUE);
    check_start_used_ptr(common);
    read_char(common, 0, READ_CHAR_MAX, NULL, READ_CHAR_UPDATE_STR_PTR);
    }
  }

/* Testing char type. */
#ifdef SUPPORT_UNICODE
if (ucp)
  {
  add_jump(compiler, &common->getucdtype, JUMP(SLJIT_FAST_CALL));
  OP2(SLJIT_SHL, TMP2, 0, SLJIT_IMM, 1, TMP1, 0);
  OP2U(SLJIT_AND | SLJIT_SET_Z, TMP2, 0, SLJIT_IMM, UCPCAT2(ucp_Mn, ucp_Pc) | UCPCAT_L | UCPCAT_N);
  OP_FLAGS(SLJIT_MOV, TMP3, 0, SLJIT_NOT_ZERO);
  }
else
#endif /* SUPPORT_UNICODE */
  {
#if PCRE2_CODE_UNIT_WIDTH != 8
  jump = CMP(SLJIT_GREATER, TMP1, 0, SLJIT_IMM, 255);
#elif defined SUPPORT_UNICODE
  /* Here TMP3 has already been zeroed. */
  jump = NULL;
  if (common->utf)
    jump = CMP(SLJIT_GREATER, TMP1, 0, SLJIT_IMM, 255);
#endif /* PCRE2_CODE_UNIT_WIDTH == 8 */
  OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(TMP1), common->ctypes);
  OP2(SLJIT_LSHR, TMP1, 0, TMP1, 0, SLJIT_IMM, 4 /* ctype_word */);
  OP2(SLJIT_AND, TMP3, 0, TMP1, 0, SLJIT_IMM, 1);
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
peek_char(common, READ_CHAR_MAX, SLJIT_MEM1(SLJIT_SP), LOCAL1, &invalid_utf2);

/* Testing char type. This is a code duplication. */
#ifdef SUPPORT_UNICODE

valid_utf = LABEL();

if (ucp)
  {
  add_jump(compiler, &common->getucdtype, JUMP(SLJIT_FAST_CALL));
  OP2(SLJIT_SHL, TMP2, 0, SLJIT_IMM, 1, TMP1, 0);
  OP2U(SLJIT_AND | SLJIT_SET_Z, TMP2, 0, SLJIT_IMM, UCPCAT2(ucp_Mn, ucp_Pc) | UCPCAT_L | UCPCAT_N);
  OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_NOT_ZERO);
  }
else
#endif /* SUPPORT_UNICODE */
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

OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), LOCAL0);
OP2(SLJIT_XOR | SLJIT_SET_Z, TMP2, 0, TMP2, 0, TMP3, 0);
OP_SRC(SLJIT_FAST_RETURN, TMP1, 0);

#ifdef SUPPORT_UNICODE
if (common->invalid_utf)
  {
  set_jumps(invalid_utf1, LABEL());

  peek_char(common, READ_CHAR_MAX, SLJIT_MEM1(SLJIT_SP), LOCAL1, NULL);
  CMPTO(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, INVALID_UTF_CHAR, valid_utf);

  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), LOCAL0);
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, -1);
  OP_SRC(SLJIT_FAST_RETURN, TMP1, 0);

  set_jumps(invalid_utf2, LABEL());
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), LOCAL0);
  OP1(SLJIT_MOV, TMP2, 0, TMP3, 0);
  OP_SRC(SLJIT_FAST_RETURN, TMP1, 0);
  }
#endif /* SUPPORT_UNICODE */
}

static BOOL optimize_class_ranges(compiler_common *common, const sljit_u8 *bits, BOOL nclass, BOOL invert, jump_list **backtracks)
{
/* May destroy TMP1. */
DEFINE_COMPILER;
int ranges[MAX_CLASS_RANGE_SIZE];
sljit_u8 bit, cbit, all;
int i, byte, length = 0;

bit = bits[0] & 0x1;
/* All bits will be zero or one (since bit is zero or one). */
all = (sljit_u8)-bit;

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
      if (length >= MAX_CLASS_RANGE_SIZE)
        return FALSE;
      ranges[length] = i;
      length++;
      bit = cbit;
      all = (sljit_u8)-cbit; /* sign extend bit into byte */
      }
    i++;
    }
  }

if (((bit == 0) && nclass) || ((bit == 1) && !nclass))
  {
  if (length >= MAX_CLASS_RANGE_SIZE)
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
  SLJIT_UNREACHABLE();
  return FALSE;
  }
}

static BOOL optimize_class_chars(compiler_common *common, const sljit_u8 *bits, BOOL nclass, BOOL invert, jump_list **backtracks)
{
/* May destroy TMP1. */
DEFINE_COMPILER;
uint16_t char_list[MAX_CLASS_CHARS_SIZE];
uint8_t byte;
sljit_s32 type;
int i, j, k, len, c;

if (!sljit_has_cpu_feature(SLJIT_HAS_CMOV))
  return FALSE;

len = 0;

for (i = 0; i < 32; i++)
  {
  byte = bits[i];

  if (nclass)
    byte = (sljit_u8)~byte;

  j = 0;
  while (byte != 0)
    {
    if (byte & 0x1)
      {
      c = i * 8 + j;

      k = len;

      if ((c & 0x20) != 0)
        {
        for (k = 0; k < len; k++)
          if (char_list[k] == c - 0x20)
            {
            char_list[k] |= 0x120;
            break;
            }
        }

      if (k == len)
        {
        if (len >= MAX_CLASS_CHARS_SIZE)
          return FALSE;

        char_list[len++] = (uint16_t) c;
        }
      }

    byte >>= 1;
    j++;
    }
  }

if (len == 0) return FALSE;  /* Should never occur, but stops analyzers complaining. */

i = 0;
j = 0;

if (char_list[0] == 0)
  {
  i++;
  OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, 0);
  OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_ZERO);
  }
else
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, 0);

while (i < len)
  {
  if ((char_list[i] & 0x100) != 0)
    j++;
  else
    {
    OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, char_list[i]);
    SELECT(SLJIT_ZERO, TMP2, TMP1, 0, TMP2);
    }
  i++;
  }

if (j != 0)
  {
  OP2(SLJIT_OR, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x20);

  for (i = 0; i < len; i++)
    if ((char_list[i] & 0x100) != 0)
      {
      j--;
      OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, char_list[i] & 0xff);
      SELECT(SLJIT_ZERO, TMP2, TMP1, 0, TMP2);
      }
  }

if (invert)
  nclass = !nclass;

type = nclass ? SLJIT_NOT_EQUAL : SLJIT_EQUAL;
add_jump(compiler, backtracks, CMP(type, TMP2, 0, SLJIT_IMM, 0));
return TRUE;
}

static BOOL optimize_class(compiler_common *common, const sljit_u8 *bits, BOOL nclass, BOOL invert, jump_list **backtracks)
{
/* May destroy TMP1. */
if (optimize_class_ranges(common, bits, nclass, invert, backtracks))
  return TRUE;
return optimize_class_chars(common, bits, nclass, invert, backtracks);
}

static void check_anynewline(compiler_common *common)
{
/* Check whether TMP1 contains a newline character. TMP2 destroyed. */
DEFINE_COMPILER;

sljit_emit_op_dst(compiler, SLJIT_FAST_ENTER, RETURN_ADDR, 0);

OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x0a);
OP2U(SLJIT_SUB | SLJIT_SET_LESS_EQUAL, TMP1, 0, SLJIT_IMM, 0x0d - 0x0a);
OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_LESS_EQUAL);
OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, 0x85 - 0x0a);
#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32
#if PCRE2_CODE_UNIT_WIDTH == 8
if (common->utf)
  {
#endif
  OP_FLAGS(SLJIT_OR, TMP2, 0, SLJIT_EQUAL);
  OP2(SLJIT_OR, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x1);
  OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, 0x2029 - 0x0a);
#if PCRE2_CODE_UNIT_WIDTH == 8
  }
#endif
#endif /* SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH == [16|32] */
OP_FLAGS(SLJIT_OR | SLJIT_SET_Z, TMP2, 0, SLJIT_EQUAL);
OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);
}

static void check_hspace(compiler_common *common)
{
/* Check whether TMP1 contains a newline character. TMP2 destroyed. */
DEFINE_COMPILER;

sljit_emit_op_dst(compiler, SLJIT_FAST_ENTER, RETURN_ADDR, 0);

OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, 0x09);
OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_EQUAL);
OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, 0x20);
OP_FLAGS(SLJIT_OR, TMP2, 0, SLJIT_EQUAL);
OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, 0xa0);
#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32
#if PCRE2_CODE_UNIT_WIDTH == 8
if (common->utf)
  {
#endif
  OP_FLAGS(SLJIT_OR, TMP2, 0, SLJIT_EQUAL);
  OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, 0x1680);
  OP_FLAGS(SLJIT_OR, TMP2, 0, SLJIT_EQUAL);
  OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, 0x180e);
  OP_FLAGS(SLJIT_OR, TMP2, 0, SLJIT_EQUAL);
  OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x2000);
  OP2U(SLJIT_SUB | SLJIT_SET_LESS_EQUAL, TMP1, 0, SLJIT_IMM, 0x200A - 0x2000);
  OP_FLAGS(SLJIT_OR, TMP2, 0, SLJIT_LESS_EQUAL);
  OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, 0x202f - 0x2000);
  OP_FLAGS(SLJIT_OR, TMP2, 0, SLJIT_EQUAL);
  OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, 0x205f - 0x2000);
  OP_FLAGS(SLJIT_OR, TMP2, 0, SLJIT_EQUAL);
  OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, 0x3000 - 0x2000);
#if PCRE2_CODE_UNIT_WIDTH == 8
  }
#endif
#endif /* SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH == [16|32] */
OP_FLAGS(SLJIT_OR | SLJIT_SET_Z, TMP2, 0, SLJIT_EQUAL);

OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);
}

static void check_vspace(compiler_common *common)
{
/* Check whether TMP1 contains a newline character. TMP2 destroyed. */
DEFINE_COMPILER;

sljit_emit_op_dst(compiler, SLJIT_FAST_ENTER, RETURN_ADDR, 0);

OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x0a);
OP2U(SLJIT_SUB | SLJIT_SET_LESS_EQUAL, TMP1, 0, SLJIT_IMM, 0x0d - 0x0a);
OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_LESS_EQUAL);
OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, 0x85 - 0x0a);
#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32
#if PCRE2_CODE_UNIT_WIDTH == 8
if (common->utf)
  {
#endif
  OP_FLAGS(SLJIT_OR, TMP2, 0, SLJIT_EQUAL);
  OP2(SLJIT_OR, TMP1, 0, TMP1, 0, SLJIT_IMM, 0x1);
  OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, 0x2029 - 0x0a);
#if PCRE2_CODE_UNIT_WIDTH == 8
  }
#endif
#endif /* SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH == [16|32] */
OP_FLAGS(SLJIT_OR | SLJIT_SET_Z, TMP2, 0, SLJIT_EQUAL);

OP_SRC(SLJIT_FAST_RETURN, RETURN_ADDR, 0);
}

static void do_casefulcmp(compiler_common *common)
{
DEFINE_COMPILER;
struct sljit_jump *jump;
struct sljit_label *label;
int char1_reg;
int char2_reg;

if (HAS_VIRTUAL_REGISTERS)
  {
  char1_reg = STR_END;
  char2_reg = STACK_TOP;
  }
else
  {
  char1_reg = TMP3;
  char2_reg = RETURN_ADDR;
  }

/* Update ref_update_local_size() when this changes. */
SLJIT_ASSERT(common->locals_size >= SSIZE_OF(sw));
sljit_emit_op_dst(compiler, SLJIT_FAST_ENTER, SLJIT_MEM1(SLJIT_SP), LOCAL0);
OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, TMP2, 0);

if (char1_reg == STR_END)
  {
  OP1(SLJIT_MOV, TMP3, 0, char1_reg, 0);
  OP1(SLJIT_MOV, RETURN_ADDR, 0, char2_reg, 0);
  }

if (sljit_emit_mem_update(compiler, MOV_UCHAR | SLJIT_MEM_SUPP | SLJIT_MEM_POST, char1_reg, SLJIT_MEM1(TMP1), IN_UCHARS(1)) == SLJIT_SUCCESS)
  {
  label = LABEL();
  sljit_emit_mem_update(compiler, MOV_UCHAR | SLJIT_MEM_POST, char1_reg, SLJIT_MEM1(TMP1), IN_UCHARS(1));
  sljit_emit_mem_update(compiler, MOV_UCHAR | SLJIT_MEM_POST, char2_reg, SLJIT_MEM1(STR_PTR), IN_UCHARS(1));
  jump = CMP(SLJIT_NOT_EQUAL, char1_reg, 0, char2_reg, 0);
  OP2(SLJIT_SUB | SLJIT_SET_Z, TMP2, 0, TMP2, 0, SLJIT_IMM, IN_UCHARS(1));
  JUMPTO(SLJIT_NOT_ZERO, label);

  JUMPHERE(jump);
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), LOCAL0);
  }
else if (sljit_emit_mem_update(compiler, MOV_UCHAR | SLJIT_MEM_SUPP | SLJIT_MEM_PRE, char1_reg, SLJIT_MEM1(TMP1), IN_UCHARS(1)) == SLJIT_SUCCESS)
  {
  OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, IN_UCHARS(1));
  OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

  label = LABEL();
  sljit_emit_mem_update(compiler, MOV_UCHAR | SLJIT_MEM_PRE, char1_reg, SLJIT_MEM1(TMP1), IN_UCHARS(1));
  sljit_emit_mem_update(compiler, MOV_UCHAR | SLJIT_MEM_PRE, char2_reg, SLJIT_MEM1(STR_PTR), IN_UCHARS(1));
  jump = CMP(SLJIT_NOT_EQUAL, char1_reg, 0, char2_reg, 0);
  OP2(SLJIT_SUB | SLJIT_SET_Z, TMP2, 0, TMP2, 0, SLJIT_IMM, IN_UCHARS(1));
  JUMPTO(SLJIT_NOT_ZERO, label);

  JUMPHERE(jump);
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), LOCAL0);
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  }
else
  {
  label = LABEL();
  OP1(MOV_UCHAR, char1_reg, 0, SLJIT_MEM1(TMP1), 0);
  OP1(MOV_UCHAR, char2_reg, 0, SLJIT_MEM1(STR_PTR), 0);
  OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, IN_UCHARS(1));
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  jump = CMP(SLJIT_NOT_EQUAL, char1_reg, 0, char2_reg, 0);
  OP2(SLJIT_SUB | SLJIT_SET_Z, TMP2, 0, TMP2, 0, SLJIT_IMM, IN_UCHARS(1));
  JUMPTO(SLJIT_NOT_ZERO, label);

  JUMPHERE(jump);
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), LOCAL0);
  }

if (char1_reg == STR_END)
  {
  OP1(SLJIT_MOV, char1_reg, 0, TMP3, 0);
  OP1(SLJIT_MOV, char2_reg, 0, RETURN_ADDR, 0);
  }

OP_SRC(SLJIT_FAST_RETURN, TMP1, 0);
}

static void do_caselesscmp(compiler_common *common)
{
DEFINE_COMPILER;
struct sljit_jump *jump;
struct sljit_label *label;
int char1_reg = STR_END;
int char2_reg;
int lcc_table;
int opt_type = 0;

if (HAS_VIRTUAL_REGISTERS)
  {
  char2_reg = STACK_TOP;
  lcc_table = STACK_LIMIT;
  }
else
  {
  char2_reg = RETURN_ADDR;
  lcc_table = TMP3;
  }

if (sljit_emit_mem_update(compiler, MOV_UCHAR | SLJIT_MEM_SUPP | SLJIT_MEM_POST, char1_reg, SLJIT_MEM1(TMP1), IN_UCHARS(1)) == SLJIT_SUCCESS)
  opt_type = 1;
else if (sljit_emit_mem_update(compiler, MOV_UCHAR | SLJIT_MEM_SUPP | SLJIT_MEM_PRE, char1_reg, SLJIT_MEM1(TMP1), IN_UCHARS(1)) == SLJIT_SUCCESS)
  opt_type = 2;

/* Update ref_update_local_size() when this changes. */
SLJIT_ASSERT(common->locals_size >= 2 * SSIZE_OF(sw));
sljit_emit_op_dst(compiler, SLJIT_FAST_ENTER, SLJIT_MEM1(SLJIT_SP), LOCAL0);
OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, TMP2, 0);

OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCAL1, char1_reg, 0);

if (char2_reg == STACK_TOP)
  {
  OP1(SLJIT_MOV, TMP3, 0, char2_reg, 0);
  OP1(SLJIT_MOV, RETURN_ADDR, 0, lcc_table, 0);
  }

OP1(SLJIT_MOV, lcc_table, 0, SLJIT_IMM, common->lcc);

if (opt_type == 1)
  {
  label = LABEL();
  sljit_emit_mem_update(compiler, MOV_UCHAR | SLJIT_MEM_POST, char1_reg, SLJIT_MEM1(TMP1), IN_UCHARS(1));
  sljit_emit_mem_update(compiler, MOV_UCHAR | SLJIT_MEM_POST, char2_reg, SLJIT_MEM1(STR_PTR), IN_UCHARS(1));
  }
else if (opt_type == 2)
  {
  OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, IN_UCHARS(1));
  OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

  label = LABEL();
  sljit_emit_mem_update(compiler, MOV_UCHAR | SLJIT_MEM_PRE, char1_reg, SLJIT_MEM1(TMP1), IN_UCHARS(1));
  sljit_emit_mem_update(compiler, MOV_UCHAR | SLJIT_MEM_PRE, char2_reg, SLJIT_MEM1(STR_PTR), IN_UCHARS(1));
  }
else
  {
  label = LABEL();
  OP1(MOV_UCHAR, char1_reg, 0, SLJIT_MEM1(TMP1), 0);
  OP1(MOV_UCHAR, char2_reg, 0, SLJIT_MEM1(STR_PTR), 0);
  OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, IN_UCHARS(1));
  }

#if PCRE2_CODE_UNIT_WIDTH != 8
jump = CMP(SLJIT_GREATER, char1_reg, 0, SLJIT_IMM, 255);
#endif
OP1(SLJIT_MOV_U8, char1_reg, 0, SLJIT_MEM2(lcc_table, char1_reg), 0);
#if PCRE2_CODE_UNIT_WIDTH != 8
JUMPHERE(jump);
jump = CMP(SLJIT_GREATER, char2_reg, 0, SLJIT_IMM, 255);
#endif
OP1(SLJIT_MOV_U8, char2_reg, 0, SLJIT_MEM2(lcc_table, char2_reg), 0);
#if PCRE2_CODE_UNIT_WIDTH != 8
JUMPHERE(jump);
#endif

if (opt_type == 0)
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

jump = CMP(SLJIT_NOT_EQUAL, char1_reg, 0, char2_reg, 0);
OP2(SLJIT_SUB | SLJIT_SET_Z, TMP2, 0, TMP2, 0, SLJIT_IMM, IN_UCHARS(1));
JUMPTO(SLJIT_NOT_ZERO, label);

JUMPHERE(jump);
OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), LOCAL0);

if (opt_type == 2)
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

if (char2_reg == STACK_TOP)
  {
  OP1(SLJIT_MOV, char2_reg, 0, TMP3, 0);
  OP1(SLJIT_MOV, lcc_table, 0, RETURN_ADDR, 0);
  }

OP1(SLJIT_MOV, char1_reg, 0, SLJIT_MEM1(SLJIT_SP), LOCAL1);
OP_SRC(SLJIT_FAST_RETURN, TMP1, 0);
}

#include "pcre2_jit_char_inc.h"

static PCRE2_SPTR compile_simple_assertion_matchingpath(compiler_common *common, PCRE2_UCHAR type, PCRE2_SPTR cc, jump_list **backtracks)
{
DEFINE_COMPILER;
struct sljit_jump *jump[4];

switch(type)
  {
  case OP_SOD:
  if (HAS_VIRTUAL_REGISTERS)
    {
    OP1(SLJIT_MOV, TMP1, 0, ARGUMENTS, 0);
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, begin));
    }
  else
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(ARGUMENTS), SLJIT_OFFSETOF(jit_arguments, begin));
  add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, STR_PTR, 0, TMP1, 0));
  return cc;

  case OP_SOM:
  if (HAS_VIRTUAL_REGISTERS)
    {
    OP1(SLJIT_MOV, TMP1, 0, ARGUMENTS, 0);
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, str));
    }
  else
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(ARGUMENTS), SLJIT_OFFSETOF(jit_arguments, str));
  add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, STR_PTR, 0, TMP1, 0));
  return cc;

  case OP_NOT_WORD_BOUNDARY:
  case OP_WORD_BOUNDARY:
  case OP_NOT_UCP_WORD_BOUNDARY:
  case OP_UCP_WORD_BOUNDARY:
  add_jump(compiler, (type == OP_NOT_WORD_BOUNDARY || type == OP_WORD_BOUNDARY) ? &common->wordboundary : &common->ucp_wordboundary, JUMP(SLJIT_FAST_CALL));
#ifdef SUPPORT_UNICODE
  if (common->invalid_utf)
    {
    add_jump(compiler, backtracks, CMP((type == OP_NOT_WORD_BOUNDARY || type == OP_NOT_UCP_WORD_BOUNDARY) ? SLJIT_NOT_EQUAL : SLJIT_SIG_LESS_EQUAL, TMP2, 0, SLJIT_IMM, 0));
    return cc;
    }
#endif /* SUPPORT_UNICODE */
  sljit_set_current_flags(compiler, SLJIT_SET_Z);
  add_jump(compiler, backtracks, JUMP((type == OP_NOT_WORD_BOUNDARY || type == OP_NOT_UCP_WORD_BOUNDARY) ? SLJIT_NOT_ZERO : SLJIT_ZERO));
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
      OP2U(SLJIT_SUB | SLJIT_SET_LESS, TMP2, 0, STR_END, 0);
      OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_LESS);
      OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, (common->newline >> 8) & 0xff);
      OP_FLAGS(SLJIT_OR | SLJIT_SET_Z, TMP2, 0, SLJIT_NOT_EQUAL);
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
    OP2U(SLJIT_SUB | SLJIT_SET_Z | SLJIT_SET_GREATER, TMP2, 0, STR_END, 0);
    jump[2] = JUMP(SLJIT_GREATER);
    add_jump(compiler, backtracks, JUMP(SLJIT_NOT_EQUAL) /* LESS */);
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
      OP1(SLJIT_MOV, TMP3, 0, STR_PTR, 0);
      read_char(common, common->nlmin, common->nlmax, backtracks, READ_CHAR_UPDATE_STR_PTR);
      add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, STR_PTR, 0, STR_END, 0));
      add_jump(compiler, &common->anynewline, JUMP(SLJIT_FAST_CALL));
      sljit_set_current_flags(compiler, SLJIT_SET_Z);
      add_jump(compiler, backtracks, JUMP(SLJIT_ZERO));
      OP1(SLJIT_MOV, STR_PTR, 0, TMP3, 0);
      }
    JUMPHERE(jump[2]);
    JUMPHERE(jump[3]);
    }
  JUMPHERE(jump[0]);
  if (common->mode != PCRE2_JIT_COMPLETE)
    check_partial(common, TRUE);
  return cc;

  case OP_EOD:
  add_jump(compiler, backtracks, CMP(SLJIT_LESS, STR_PTR, 0, STR_END, 0));
  if (common->mode != PCRE2_JIT_COMPLETE)
    check_partial(common, TRUE);
  return cc;

  case OP_DOLL:
  if (HAS_VIRTUAL_REGISTERS)
    {
    OP1(SLJIT_MOV, TMP2, 0, ARGUMENTS, 0);
    OP2U(SLJIT_AND32 | SLJIT_SET_Z, SLJIT_MEM1(TMP2), SLJIT_OFFSETOF(jit_arguments, options), SLJIT_IMM, PCRE2_NOTEOL);
    }
  else
    OP2U(SLJIT_AND32 | SLJIT_SET_Z, SLJIT_MEM1(ARGUMENTS), SLJIT_OFFSETOF(jit_arguments, options), SLJIT_IMM, PCRE2_NOTEOL);
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
  if (HAS_VIRTUAL_REGISTERS)
    {
    OP1(SLJIT_MOV, TMP2, 0, ARGUMENTS, 0);
    OP2U(SLJIT_AND32 | SLJIT_SET_Z, SLJIT_MEM1(TMP2), SLJIT_OFFSETOF(jit_arguments, options), SLJIT_IMM, PCRE2_NOTEOL);
    }
  else
    OP2U(SLJIT_AND32 | SLJIT_SET_Z, SLJIT_MEM1(ARGUMENTS), SLJIT_OFFSETOF(jit_arguments, options), SLJIT_IMM, PCRE2_NOTEOL);
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
    peek_char(common, common->nlmax, TMP3, 0, NULL);
    check_newlinechar(common, common->nltype, backtracks, FALSE);
    }
  JUMPHERE(jump[0]);
  return cc;

  case OP_CIRC:
  if (HAS_VIRTUAL_REGISTERS)
    {
    OP1(SLJIT_MOV, TMP2, 0, ARGUMENTS, 0);
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(TMP2), SLJIT_OFFSETOF(jit_arguments, begin));
    add_jump(compiler, backtracks, CMP(SLJIT_GREATER, STR_PTR, 0, TMP1, 0));
    OP2U(SLJIT_AND32 | SLJIT_SET_Z, SLJIT_MEM1(TMP2), SLJIT_OFFSETOF(jit_arguments, options), SLJIT_IMM, PCRE2_NOTBOL);
    add_jump(compiler, backtracks, JUMP(SLJIT_NOT_ZERO));
    }
  else
    {
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(ARGUMENTS), SLJIT_OFFSETOF(jit_arguments, begin));
    add_jump(compiler, backtracks, CMP(SLJIT_GREATER, STR_PTR, 0, TMP1, 0));
    OP2U(SLJIT_AND32 | SLJIT_SET_Z, SLJIT_MEM1(ARGUMENTS), SLJIT_OFFSETOF(jit_arguments, options), SLJIT_IMM, PCRE2_NOTBOL);
    add_jump(compiler, backtracks, JUMP(SLJIT_NOT_ZERO));
    }
  return cc;

  case OP_CIRCM:
  /* TMP2 might be used by peek_char_back. */
  if (HAS_VIRTUAL_REGISTERS)
    {
    OP1(SLJIT_MOV, TMP1, 0, ARGUMENTS, 0);
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, begin));
    jump[1] = CMP(SLJIT_GREATER, STR_PTR, 0, TMP2, 0);
    OP2U(SLJIT_AND32 | SLJIT_SET_Z, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, options), SLJIT_IMM, PCRE2_NOTBOL);
    }
  else
    {
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(ARGUMENTS), SLJIT_OFFSETOF(jit_arguments, begin));
    jump[1] = CMP(SLJIT_GREATER, STR_PTR, 0, TMP2, 0);
    OP2U(SLJIT_AND32 | SLJIT_SET_Z, SLJIT_MEM1(ARGUMENTS), SLJIT_OFFSETOF(jit_arguments, options), SLJIT_IMM, PCRE2_NOTBOL);
    }
  add_jump(compiler, backtracks, JUMP(SLJIT_NOT_ZERO));
  jump[0] = JUMP(SLJIT_JUMP);
  JUMPHERE(jump[1]);

  if (!common->alt_circumflex)
    add_jump(compiler, backtracks, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));

  if (common->nltype == NLTYPE_FIXED && common->newline > 255)
    {
    OP2(SLJIT_SUB, TMP1, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(2));
    add_jump(compiler, backtracks, CMP(SLJIT_LESS, TMP1, 0, TMP2, 0));
    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-2));
    OP1(MOV_UCHAR, TMP2, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-1));
    add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, (common->newline >> 8) & 0xff));
    add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, TMP2, 0, SLJIT_IMM, common->newline & 0xff));
    }
  else
    {
    peek_char_back(common, common->nlmax, backtracks);
    check_newlinechar(common, common->nltype, backtracks, FALSE);
    }
  JUMPHERE(jump[0]);
  return cc;
  }
SLJIT_UNREACHABLE();
return cc;
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
#if defined SUPPORT_UNICODE
struct sljit_label *loop;
struct sljit_label *caseless_loop;
struct sljit_jump *turkish_ascii_i = NULL;
struct sljit_jump *turkish_non_ascii_i = NULL;
jump_list *no_match = NULL;
int source_reg = COUNT_MATCH;
int source_end_reg = ARGUMENTS;
int char1_reg = STACK_LIMIT;
PCRE2_UCHAR refi_flag = 0;

if (*cc == OP_REFI || *cc == OP_DNREFI)
  refi_flag = cc[PRIV(OP_lengths)[*cc] - 1];
#endif /* SUPPORT_UNICODE */

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
if ((common->utf || common->ucp) && (*cc == OP_REFI || *cc == OP_DNREFI))
  {
  /* Update ref_update_local_size() when this changes. */
  SLJIT_ASSERT(common->locals_size >= 3 * SSIZE_OF(sw));

  if (ref)
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1));
  else
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(TMP2), sizeof(sljit_sw));

  if (withchecks && emptyfail)
    add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, TMP2, 0));

  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCAL0, source_reg, 0);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCAL1, source_end_reg, 0);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCAL2, char1_reg, 0);

  OP1(SLJIT_MOV, source_reg, 0, TMP1, 0);
  OP1(SLJIT_MOV, source_end_reg, 0, TMP2, 0);

  loop = LABEL();
  jump = CMP(SLJIT_GREATER_EQUAL, source_reg, 0, source_end_reg, 0);
  partial = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);

  /* Read original character. It must be a valid UTF character. */
  OP1(SLJIT_MOV, TMP3, 0, STR_PTR, 0);
  OP1(SLJIT_MOV, STR_PTR, 0, source_reg, 0);

  read_char(common, 0, READ_CHAR_MAX, NULL, READ_CHAR_UPDATE_STR_PTR | READ_CHAR_VALID_UTF);

  OP1(SLJIT_MOV, source_reg, 0, STR_PTR, 0);
  OP1(SLJIT_MOV, STR_PTR, 0, TMP3, 0);
  OP1(SLJIT_MOV, char1_reg, 0, TMP1, 0);

  /* Read second character. */
  read_char(common, 0, READ_CHAR_MAX, &no_match, READ_CHAR_UPDATE_STR_PTR);

  CMPTO(SLJIT_EQUAL, TMP1, 0, char1_reg, 0, loop);

  if ((refi_flag & (REFI_FLAG_TURKISH_CASING|REFI_FLAG_CASELESS_RESTRICT)) ==
        REFI_FLAG_TURKISH_CASING)
    {
    OP2(SLJIT_OR, SLJIT_TMP_DEST_REG, 0, char1_reg, 0, SLJIT_IMM, 0x20);
    turkish_ascii_i = CMP(SLJIT_EQUAL, SLJIT_TMP_DEST_REG, 0, SLJIT_IMM, 0x69);

    OP2(SLJIT_OR, SLJIT_TMP_DEST_REG, 0, char1_reg, 0, SLJIT_IMM, 0x1);
    turkish_non_ascii_i = CMP(SLJIT_EQUAL, SLJIT_TMP_DEST_REG, 0, SLJIT_IMM, 0x131);
    }

  OP1(SLJIT_MOV, TMP3, 0, TMP1, 0);

  add_jump(compiler, &common->getucd, JUMP(SLJIT_FAST_CALL));

  OP2(SLJIT_SHL, TMP1, 0, TMP2, 0, SLJIT_IMM, 2);
  OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, 3);
  OP2(SLJIT_ADD, TMP2, 0, TMP2, 0, TMP1, 0);

  OP2(SLJIT_ADD, TMP2, 0, TMP2, 0, SLJIT_IMM, (sljit_sw)PRIV(ucd_records));

  OP1(SLJIT_MOV_S32, TMP1, 0, SLJIT_MEM1(TMP2), SLJIT_OFFSETOF(ucd_record, other_case));
  OP1(SLJIT_MOV_U8, TMP2, 0, SLJIT_MEM1(TMP2), SLJIT_OFFSETOF(ucd_record, caseset));
  OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, TMP3, 0);
  CMPTO(SLJIT_EQUAL, TMP1, 0, char1_reg, 0, loop);

  add_jump(compiler, &no_match, CMP(SLJIT_EQUAL, TMP2, 0, SLJIT_IMM, 0));
  OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, 2);
  OP2(SLJIT_ADD, TMP2, 0, TMP2, 0, SLJIT_IMM, (sljit_sw)PRIV(ucd_caseless_sets));

  if (refi_flag & REFI_FLAG_CASELESS_RESTRICT)
    add_jump(compiler, &no_match, CMP(SLJIT_LESS | SLJIT_32, SLJIT_MEM1(TMP2), 0, SLJIT_IMM, 128));

  caseless_loop = LABEL();
  OP1(SLJIT_MOV_U32, TMP1, 0, SLJIT_MEM1(TMP2), 0);
  OP2(SLJIT_ADD, TMP2, 0, TMP2, 0, SLJIT_IMM, sizeof(uint32_t));
  OP2U(SLJIT_SUB | SLJIT_SET_Z | SLJIT_SET_LESS, TMP1, 0, char1_reg, 0);
  JUMPTO(SLJIT_EQUAL, loop);
  JUMPTO(SLJIT_LESS, caseless_loop);

  if ((refi_flag & (REFI_FLAG_TURKISH_CASING|REFI_FLAG_CASELESS_RESTRICT)) ==
        REFI_FLAG_TURKISH_CASING)
    {
    add_jump(compiler, &no_match, JUMP(SLJIT_JUMP));
    JUMPHERE(turkish_ascii_i);

    OP2(SLJIT_LSHR, char1_reg, 0, char1_reg, 0, SLJIT_IMM, 5);
    OP2(SLJIT_AND, char1_reg, 0, char1_reg, 0, SLJIT_IMM, 1);
    OP2(SLJIT_XOR, char1_reg, 0, char1_reg, 0, SLJIT_IMM, 1);
    OP2(SLJIT_ADD, char1_reg, 0, char1_reg, 0, SLJIT_IMM, 0x130);
    CMPTO(SLJIT_EQUAL, TMP1, 0, char1_reg, 0, loop);

    add_jump(compiler, &no_match, JUMP(SLJIT_JUMP));
    JUMPHERE(turkish_non_ascii_i);

    OP2(SLJIT_AND, char1_reg, 0, char1_reg, 0, SLJIT_IMM, 1);
    OP2(SLJIT_XOR, char1_reg, 0, char1_reg, 0, SLJIT_IMM, 1);
    OP2(SLJIT_SHL, char1_reg, 0, char1_reg, 0, SLJIT_IMM, 5);
    OP2(SLJIT_ADD, char1_reg, 0, char1_reg, 0, SLJIT_IMM, 0x49);
    CMPTO(SLJIT_EQUAL, TMP1, 0, char1_reg, 0, loop);
    }

  set_jumps(no_match, LABEL());
  if (common->mode == PCRE2_JIT_COMPLETE)
    JUMPHERE(partial);

  OP1(SLJIT_MOV, source_reg, 0, SLJIT_MEM1(SLJIT_SP), LOCAL0);
  OP1(SLJIT_MOV, source_end_reg, 0, SLJIT_MEM1(SLJIT_SP), LOCAL1);
  OP1(SLJIT_MOV, char1_reg, 0, SLJIT_MEM1(SLJIT_SP), LOCAL2);
  add_jump(compiler, backtracks, JUMP(SLJIT_JUMP));

  if (common->mode != PCRE2_JIT_COMPLETE)
    {
    JUMPHERE(partial);
    OP1(SLJIT_MOV, source_reg, 0, SLJIT_MEM1(SLJIT_SP), LOCAL0);
    OP1(SLJIT_MOV, source_end_reg, 0, SLJIT_MEM1(SLJIT_SP), LOCAL1);
    OP1(SLJIT_MOV, char1_reg, 0, SLJIT_MEM1(SLJIT_SP), LOCAL2);

    check_partial(common, FALSE);
    add_jump(compiler, backtracks, JUMP(SLJIT_JUMP));
    }

  JUMPHERE(jump);
  OP1(SLJIT_MOV, source_reg, 0, SLJIT_MEM1(SLJIT_SP), LOCAL0);
  OP1(SLJIT_MOV, source_end_reg, 0, SLJIT_MEM1(SLJIT_SP), LOCAL1);
  OP1(SLJIT_MOV, char1_reg, 0, SLJIT_MEM1(SLJIT_SP), LOCAL2);
  return;
  }
else
#endif /* SUPPORT_UNICODE */
  {
  if (ref)
    OP2(SLJIT_SUB | SLJIT_SET_Z, TMP2, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1), TMP1, 0);
  else
    OP2(SLJIT_SUB | SLJIT_SET_Z, TMP2, 0, SLJIT_MEM1(TMP2), sizeof(sljit_sw), TMP1, 0);

  if (withchecks)
    jump = JUMP(SLJIT_ZERO);

  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP2, 0);
  partial = CMP(SLJIT_GREATER, STR_PTR, 0, STR_END, 0);
  if (common->mode == PCRE2_JIT_COMPLETE)
    add_jump(compiler, backtracks, partial);

  add_jump(compiler, (*cc == OP_REF || *cc == OP_DNREF) ? &common->casefulcmp : &common->caselesscmp, JUMP(SLJIT_FAST_CALL));
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
    add_jump(compiler, (*cc == OP_REF || *cc == OP_DNREF) ? &common->casefulcmp : &common->caselesscmp, JUMP(SLJIT_FAST_CALL));
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
int local_start = LOCAL2;
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

if (*ccbegin == OP_REFI || *ccbegin == OP_DNREFI)
  {
  cc += 1;
#ifdef SUPPORT_UNICODE
  if (common->utf || common->ucp)
    local_start = LOCAL3;
#endif
  }

type = cc[1 + IMM2_SIZE];

SLJIT_COMPILE_ASSERT((OP_CRSTAR & 0x1) == 0, crstar_opcode_must_be_even);
/* Update ref_update_local_size() when this changes. */
SLJIT_ASSERT(local_start + 2 * SSIZE_OF(sw) <= (int)LOCAL0 + common->locals_size);
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
  SLJIT_UNREACHABLE();
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
    OP2(SLJIT_ADD, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, sizeof(sljit_sw));
    /* Handles both invalid and empty cases. Since the minimum repeat,
    is zero the invalid case is basically the same as an empty case. */
    if (ref)
      zerolength = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1));
    else
      {
      compile_dnref_search(common, ccbegin, NULL);
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(TMP2), 0);
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), local_start + SSIZE_OF(sw), TMP2, 0);
      zerolength = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_MEM1(TMP2), sizeof(sljit_sw));
      }
    /* Restore if not zero length. */
    OP2(SLJIT_SUB, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, sizeof(sljit_sw));
    }
  else
    {
    allocate_stack(common, 1);
    if (ref)
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset));
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), SLJIT_IMM, 0);

    if (ref)
      {
      if (!common->unset_backref)
        add_jump(compiler, &backtrack->own_backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(1)));
      zerolength = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1));
      }
    else
      {
      compile_dnref_search(common, ccbegin, &backtrack->own_backtracks);
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(TMP2), 0);
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), local_start + SSIZE_OF(sw), TMP2, 0);
      zerolength = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_MEM1(TMP2), sizeof(sljit_sw));
      }
    }

  if (min > 1 || max > 1)
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), local_start, SLJIT_IMM, 0);

  label = LABEL();
  if (!ref)
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), local_start + SSIZE_OF(sw));
  compile_ref_matchingpath(common, ccbegin, &backtrack->own_backtracks, FALSE, FALSE);

  if (min > 1 || max > 1)
    {
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), local_start);
    OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, 1);
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), local_start, TMP1, 0);
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
    if (!common->unset_backref)
      add_jump(compiler, &backtrack->own_backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(1)));
    zerolength = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1));
    }
  else
    {
    compile_dnref_search(common, ccbegin, &backtrack->own_backtracks);
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(TMP2), 0);
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(2), TMP2, 0);
    zerolength = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_MEM1(TMP2), sizeof(sljit_sw));
    }
  }

BACKTRACK_AS(ref_iterator_backtrack)->matchingpath = LABEL();
if (max > 0)
  add_jump(compiler, &backtrack->own_backtracks, CMP(SLJIT_GREATER_EQUAL, SLJIT_MEM1(STACK_TOP), STACK(1), SLJIT_IMM, max));

if (!ref)
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(STACK_TOP), STACK(2));
compile_ref_matchingpath(common, ccbegin, &backtrack->own_backtracks, TRUE, TRUE);
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
  entry->entry_label = NULL;
  entry->backtrack_label = NULL;
  entry->entry_calls = NULL;
  entry->backtrack_calls = NULL;
  entry->start = start;

  if (prev != NULL)
    prev->next = entry;
  else
    common->entries = entry;
  }

BACKTRACK_AS(recurse_backtrack)->entry = entry;

if (entry->entry_label == NULL)
  add_jump(compiler, &entry->entry_calls, JUMP(SLJIT_FAST_CALL));
else
  JUMPTO(SLJIT_FAST_CALL, entry->entry_label);
/* Leave if the match is failed. */
add_jump(compiler, &backtrack->own_backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, 0));
BACKTRACK_AS(recurse_backtrack)->matchingpath = LABEL();
return cc + 1 + LINK_SIZE;
}

static sljit_s32 SLJIT_FUNC do_callout_jit(struct jit_arguments *arguments, pcre2_callout_block *callout_block, PCRE2_SPTR *jit_ovector)
{
PCRE2_SPTR begin;
PCRE2_SIZE *ovector;
sljit_u32 oveccount, capture_top;

if (arguments->callout == NULL)
  return 0;

SLJIT_COMPILE_ASSERT(sizeof (PCRE2_SIZE) <= sizeof (sljit_sw), pcre2_size_must_be_lower_than_sljit_sw_size);

begin = arguments->begin;
ovector = (PCRE2_SIZE*)(callout_block + 1);
oveccount = callout_block->capture_top;

SLJIT_ASSERT(oveccount >= 1);

callout_block->version = 2;
callout_block->callout_flags = 0;

/* Offsets in subject. */
callout_block->subject_length = arguments->end - arguments->begin;
callout_block->start_match = jit_ovector[0] - begin;
callout_block->current_position = (PCRE2_SPTR)callout_block->offset_vector - begin;
callout_block->subject = begin;

/* Convert and copy the JIT offset vector to the ovector array. */
callout_block->capture_top = 1;
callout_block->offset_vector = ovector;

ovector[0] = PCRE2_UNSET;
ovector[1] = PCRE2_UNSET;
ovector += 2;
jit_ovector += 2;
capture_top = 1;

/* Convert pointers to sizes. */
while (--oveccount != 0)
  {
  capture_top++;

  ovector[0] = (PCRE2_SIZE)(jit_ovector[0] - begin);
  ovector[1] = (PCRE2_SIZE)(jit_ovector[1] - begin);

  if (ovector[0] != PCRE2_UNSET)
    callout_block->capture_top = capture_top;

  ovector += 2;
  jit_ovector += 2;
  }

return (arguments->callout)(callout_block, arguments->callout_data);
}

#define CALLOUT_ARG_OFFSET(arg) \
    SLJIT_OFFSETOF(pcre2_callout_block, arg)

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
sljit_uw callout_arg_size = (common->re->top_bracket + 1) * 2 * SSIZE_OF(sw);

PUSH_BACKTRACK(sizeof(backtrack_common), cc, NULL);

callout_arg_size = (sizeof(pcre2_callout_block) + callout_arg_size + sizeof(sljit_sw) - 1) / sizeof(sljit_sw);

allocate_stack(common, callout_arg_size);

SLJIT_ASSERT(common->capture_last_ptr != 0);
OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), common->capture_last_ptr);
OP1(SLJIT_MOV, TMP1, 0, ARGUMENTS, 0);
value1 = (*cc == OP_CALLOUT) ? cc[1 + 2 * LINK_SIZE] : 0;
OP1(SLJIT_MOV_U32, SLJIT_MEM1(STACK_TOP), CALLOUT_ARG_OFFSET(callout_number), SLJIT_IMM, value1);
OP1(SLJIT_MOV_U32, SLJIT_MEM1(STACK_TOP), CALLOUT_ARG_OFFSET(capture_last), TMP2, 0);
OP1(SLJIT_MOV_U32, SLJIT_MEM1(STACK_TOP), CALLOUT_ARG_OFFSET(capture_top), SLJIT_IMM, common->re->top_bracket + 1);

/* These pointer sized fields temporarly stores internal variables. */
OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), CALLOUT_ARG_OFFSET(offset_vector), STR_PTR, 0);

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

SLJIT_ASSERT(TMP1 == SLJIT_R0 && STR_PTR == SLJIT_R1);

/* Needed to save important temporary registers. */
SLJIT_ASSERT(common->locals_size >= SSIZE_OF(sw));
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCAL0, STR_PTR, 0);
/* SLJIT_R0 = arguments */
OP1(SLJIT_MOV, SLJIT_R1, 0, STACK_TOP, 0);
GET_LOCAL_BASE(SLJIT_R2, 0, OVECTOR_START);
sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_ARGS3(32, W, W, W), SLJIT_IMM, SLJIT_FUNC_ADDR(do_callout_jit));
OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(SLJIT_SP), LOCAL0);
free_stack(common, callout_arg_size);

/* Check return value. */
OP2U(SLJIT_SUB32 | SLJIT_SET_Z | SLJIT_SET_SIG_GREATER, SLJIT_RETURN_REG, 0, SLJIT_IMM, 0);
add_jump(compiler, &backtrack->own_backtracks, JUMP(SLJIT_SIG_GREATER));
if (common->abort_label == NULL)
  add_jump(compiler, &common->abort, JUMP(SLJIT_NOT_EQUAL) /* SIG_LESS */);
else
  JUMPTO(SLJIT_NOT_EQUAL /* SIG_LESS */, common->abort_label);
return cc + callout_length;
}

#undef CALLOUT_ARG_SIZE
#undef CALLOUT_ARG_OFFSET

static PCRE2_SPTR compile_reverse_matchingpath(compiler_common *common, PCRE2_SPTR cc, backtrack_common *parent)
{
DEFINE_COMPILER;
backtrack_common *backtrack = NULL;
jump_list **reverse_failed;
unsigned int lmin, lmax;
#ifdef SUPPORT_UNICODE
struct sljit_jump *jump;
struct sljit_label *label;
#endif

SLJIT_ASSERT(parent->top == NULL);

if (*cc == OP_REVERSE)
  {
  reverse_failed = &parent->own_backtracks;
  lmin = GET2(cc, 1);
  lmax = lmin;
  cc += 1 + IMM2_SIZE;

  SLJIT_ASSERT(lmin > 0);
  }
else
  {
  SLJIT_ASSERT(*cc == OP_VREVERSE);
  PUSH_BACKTRACK(sizeof(vreverse_backtrack), cc, NULL);

  reverse_failed = &backtrack->own_backtracks;
  lmin = GET2(cc, 1);
  lmax = GET2(cc, 1 + IMM2_SIZE);
  cc += 1 + 2 * IMM2_SIZE;

  SLJIT_ASSERT(lmin < lmax);
  }

if (HAS_VIRTUAL_REGISTERS)
  {
  OP1(SLJIT_MOV, TMP1, 0, ARGUMENTS, 0);
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, begin));
  }
else
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(ARGUMENTS), SLJIT_OFFSETOF(jit_arguments, begin));

#ifdef SUPPORT_UNICODE
if (common->utf)
  {
  if (lmin > 0)
    {
    OP1(SLJIT_MOV, TMP3, 0, SLJIT_IMM, lmin);
    label = LABEL();
    add_jump(compiler, reverse_failed, CMP(SLJIT_LESS_EQUAL, STR_PTR, 0, TMP2, 0));
    move_back(common, reverse_failed, FALSE);
    OP2(SLJIT_SUB | SLJIT_SET_Z, TMP3, 0, TMP3, 0, SLJIT_IMM, 1);
    JUMPTO(SLJIT_NOT_ZERO, label);
    }

  if (lmin < lmax)
    {
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(3), STR_PTR, 0);

    OP1(SLJIT_MOV, TMP3, 0, SLJIT_IMM, lmax - lmin);
    label = LABEL();
    jump = CMP(SLJIT_LESS_EQUAL, STR_PTR, 0, TMP2, 0);
    move_back(common, reverse_failed, FALSE);
    OP2(SLJIT_SUB | SLJIT_SET_Z, TMP3, 0, TMP3, 0, SLJIT_IMM, 1);
    JUMPTO(SLJIT_NOT_ZERO, label);

    JUMPHERE(jump);
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(2), STR_PTR, 0);
    }
  }
else
#endif
  {
  if (lmin > 0)
    {
    OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(lmin));
    add_jump(compiler, reverse_failed, CMP(SLJIT_LESS, STR_PTR, 0, TMP2, 0));
    }

  if (lmin < lmax)
    {
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(3), STR_PTR, 0);

    OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(lmax - lmin));
    OP2U(SLJIT_SUB | SLJIT_SET_LESS, STR_PTR, 0, TMP2, 0);
    SELECT(SLJIT_LESS, STR_PTR, TMP2, 0, STR_PTR);

    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(2), STR_PTR, 0);
    }
  }

check_start_used_ptr(common);

if (lmin < lmax)
  BACKTRACK_AS(vreverse_backtrack)->matchingpath = LABEL();

return cc;
}

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
    case OP_NOT_UCP_WORD_BOUNDARY:
    case OP_UCP_WORD_BOUNDARY:
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
BOOL local_quit_available = FALSE;
BOOL needs_control_head;
BOOL end_block_size = 0;
BOOL has_vreverse;
int private_data_ptr;
backtrack_common altbacktrack;
PCRE2_SPTR ccbegin;
PCRE2_UCHAR opcode;
PCRE2_UCHAR bra = OP_BRA;
jump_list *tmp = NULL;
jump_list **target = (conditional) ? &backtrack->condfailed : &backtrack->common.own_backtracks;
jump_list **found;
/* Saving previous accept variables. */
BOOL save_local_quit_available = common->local_quit_available;
BOOL save_in_positive_assertion = common->in_positive_assertion;
sljit_s32 save_restore_end_ptr = common->restore_end_ptr;
then_trap_backtrack *save_then_trap = common->then_trap;
struct sljit_label *save_quit_label = common->quit_label;
struct sljit_label *save_accept_label = common->accept_label;
jump_list *save_quit = common->quit;
jump_list *save_positive_assertion_quit = common->positive_assertion_quit;
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

if ((opcode == OP_ASSERTBACK || opcode == OP_ASSERTBACK_NOT) && find_vreverse(ccbegin))
  end_block_size = 3;

if (framesize < 0)
  {
  extrasize = 1;
  if (bra == OP_BRA && !assert_needs_str_ptr_saving(ccbegin + 1 + LINK_SIZE))
    extrasize = 0;

  extrasize += end_block_size;

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
    SLJIT_ASSERT(extrasize == end_block_size + 2);
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, SLJIT_IMM, 0);
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(end_block_size + 1), TMP1, 0);
    }
  }
else
  {
  extrasize = (needs_control_head ? 3 : 2) + end_block_size;

  OP1(SLJIT_MOV, TMP2, 0, STACK_TOP, 0);
  allocate_stack(common, framesize + extrasize);

  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, TMP2, 0);
  if (needs_control_head)
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr);
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), STR_PTR, 0);

  if (needs_control_head)
    {
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(end_block_size + 2), TMP1, 0);
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(end_block_size + 1), TMP2, 0);
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, SLJIT_IMM, 0);
    }
  else
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(end_block_size + 1), TMP1, 0);

  init_frame(common, ccbegin, NULL, framesize + extrasize - 1, extrasize);
  }

if (end_block_size > 0)
  {
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(1), STR_END, 0);
  OP1(SLJIT_MOV, STR_END, 0, STR_PTR, 0);
  }

memset(&altbacktrack, 0, sizeof(backtrack_common));
if (conditional || (opcode == OP_ASSERT_NOT || opcode == OP_ASSERTBACK_NOT))
  {
  /* Control verbs cannot escape from these asserts. */
  local_quit_available = TRUE;
  common->restore_end_ptr = 0;
  common->local_quit_available = TRUE;
  common->quit_label = NULL;
  common->quit = NULL;
  }

common->in_positive_assertion = (opcode == OP_ASSERT || opcode == OP_ASSERTBACK);
common->positive_assertion_quit = NULL;

while (1)
  {
  common->accept_label = NULL;
  common->accept = NULL;
  altbacktrack.top = NULL;
  altbacktrack.own_backtracks = NULL;

  if (*ccbegin == OP_ALT && extrasize > 0)
    OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(0));

  altbacktrack.cc = ccbegin;
  ccbegin += 1 + LINK_SIZE;

  has_vreverse = (*ccbegin == OP_VREVERSE);
  if (*ccbegin == OP_REVERSE || has_vreverse)
    ccbegin = compile_reverse_matchingpath(common, ccbegin, &altbacktrack);

  compile_matchingpath(common, ccbegin, cc, &altbacktrack);
  if (SLJIT_UNLIKELY(sljit_get_compiler_error(compiler)))
    {
    if (local_quit_available)
      {
      common->local_quit_available = save_local_quit_available;
      common->quit_label = save_quit_label;
      common->quit = save_quit;
      }
    common->in_positive_assertion = save_in_positive_assertion;
    common->restore_end_ptr = save_restore_end_ptr;
    common->then_trap = save_then_trap;
    common->accept_label = save_accept_label;
    common->positive_assertion_quit = save_positive_assertion_quit;
    common->accept = save_accept;
    return NULL;
    }

  if (has_vreverse)
    {
    SLJIT_ASSERT(altbacktrack.top != NULL);
    add_jump(compiler, &altbacktrack.top->simple_backtracks, CMP(SLJIT_LESS, STR_PTR, 0, STR_END, 0));
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

    if (end_block_size > 0)
      OP1(SLJIT_MOV, STR_END, 0, SLJIT_MEM1(STACK_TOP), STACK(-extrasize + 1));

    if (needs_control_head)
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, SLJIT_MEM1(STACK_TOP), STACK(-1));
    }
  else
    {
    if ((opcode != OP_ASSERT_NOT && opcode != OP_ASSERTBACK_NOT) || conditional)
      {
      /* We don't need to keep the STR_PTR, only the previous private_data_ptr. */
      OP2(SLJIT_SUB, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr, SLJIT_IMM, (framesize + 1) * sizeof(sljit_sw));

      if (end_block_size > 0)
        OP1(SLJIT_MOV, STR_END, 0, SLJIT_MEM1(STACK_TOP), STACK(-extrasize + 2));

      if (needs_control_head)
        OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, SLJIT_MEM1(STACK_TOP), STACK(-1));
      }
    else
      {
      OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);

      if (end_block_size > 0)
        OP1(SLJIT_MOV, STR_END, 0, SLJIT_MEM1(STACK_TOP), STACK(-framesize - extrasize + 1));

      if (needs_control_head)
        OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, SLJIT_MEM1(STACK_TOP), STACK(-framesize - 2));
      add_jump(compiler, &common->revertframes, JUMP(SLJIT_FAST_CALL));
      OP2(SLJIT_ADD, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, (framesize - 1) * sizeof(sljit_sw));
      }
    }

  if (opcode == OP_ASSERT_NOT || opcode == OP_ASSERTBACK_NOT)
    {
    /* We know that STR_PTR was stored on the top of the stack. */
    if (conditional)
      {
      if (extrasize > 0)
        OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(-end_block_size - (needs_control_head ? 2 : 1)));
      }
    else if (bra == OP_BRAZERO)
      {
      if (framesize < 0)
        OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(-extrasize));
      else
        {
        OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(-framesize - 1));
        OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(-framesize - extrasize));
        OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, TMP1, 0);
        }
      OP2(SLJIT_SUB, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, sizeof(sljit_sw));
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), SLJIT_IMM, 0);
      }
    else if (framesize >= 0)
      {
      /* For OP_BRA and OP_BRAMINZERO. */
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, SLJIT_MEM1(STACK_TOP), STACK(-framesize - 1));
      }
    }
  add_jump(compiler, found, JUMP(SLJIT_JUMP));

  compile_backtrackingpath(common, altbacktrack.top);
  if (SLJIT_UNLIKELY(sljit_get_compiler_error(compiler)))
    {
    if (local_quit_available)
      {
      common->local_quit_available = save_local_quit_available;
      common->quit_label = save_quit_label;
      common->quit = save_quit;
      }
    common->in_positive_assertion = save_in_positive_assertion;
    common->restore_end_ptr = save_restore_end_ptr;
    common->then_trap = save_then_trap;
    common->accept_label = save_accept_label;
    common->positive_assertion_quit = save_positive_assertion_quit;
    common->accept = save_accept;
    return NULL;
    }
  set_jumps(altbacktrack.own_backtracks, LABEL());

  if (*cc != OP_ALT)
    break;

  ccbegin = cc;
  cc += GET(cc, 1);
  }

if (local_quit_available)
  {
  SLJIT_ASSERT(common->positive_assertion_quit == NULL);
  /* Makes the check less complicated below. */
  common->positive_assertion_quit = common->quit;
  }

/* None of them matched. */
if (common->positive_assertion_quit != NULL)
  {
  jump = JUMP(SLJIT_JUMP);
  set_jumps(common->positive_assertion_quit, LABEL());
  SLJIT_ASSERT(framesize != no_stack);
  if (framesize < 0)
    OP2(SLJIT_SUB, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr, SLJIT_IMM, extrasize * sizeof(sljit_sw));
  else
    {
    OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
    add_jump(compiler, &common->revertframes, JUMP(SLJIT_FAST_CALL));
    OP2(SLJIT_SUB, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, (extrasize + 1) * sizeof(sljit_sw));
    }
  JUMPHERE(jump);
  }

if (end_block_size > 0)
  OP1(SLJIT_MOV, STR_END, 0, SLJIT_MEM1(STACK_TOP), STACK(1));

if (needs_control_head)
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, SLJIT_MEM1(STACK_TOP), STACK(end_block_size + 1));

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
      if (extrasize >= 2)
        free_stack(common, extrasize - 1);
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
      OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(-extrasize));

    /* Keep the STR_PTR on the top of the stack. */
    if (bra == OP_BRAZERO)
      {
      /* This allocation is always successful. */
      OP2(SLJIT_SUB, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, sizeof(sljit_sw));
      if (extrasize >= 2)
        OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), STR_PTR, 0);
      }
    else if (bra == OP_BRAMINZERO)
      {
      OP2(SLJIT_SUB, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, sizeof(sljit_sw));
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), SLJIT_IMM, 0);
      }
    }
  else
    {
    if (bra == OP_BRA)
      {
      /* We don't need to keep the STR_PTR, only the previous private_data_ptr. */
      OP2(SLJIT_SUB, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr, SLJIT_IMM, (framesize + 1) * sizeof(sljit_sw));
      OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(-extrasize + 1));
      }
    else
      {
      /* We don't need to keep the STR_PTR, only the previous private_data_ptr. */
      OP2(SLJIT_SUB, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr, SLJIT_IMM, (framesize + end_block_size + 2) * sizeof(sljit_sw));

      if (extrasize == 2 + end_block_size)
        {
        OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
        if (bra == OP_BRAMINZERO)
          OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), SLJIT_IMM, 0);
        }
      else
        {
        SLJIT_ASSERT(extrasize == 3 + end_block_size);
        OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(-1));
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
    SLJIT_ASSERT(framesize != 0);
    if (framesize > 0)
      {
      OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
      add_jump(compiler, &common->revertframes, JUMP(SLJIT_FAST_CALL));
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(-2));
      OP2(SLJIT_ADD, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, (framesize - 1) * sizeof(sljit_sw));
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, TMP1, 0);
      }
    set_jumps(backtrack->common.own_backtracks, LABEL());
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
      if (extrasize >= 2)
        free_stack(common, extrasize - 1);
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
    SLJIT_ASSERT(found == &backtrack->common.own_backtracks);
    set_jumps(backtrack->common.own_backtracks, LABEL());
    backtrack->common.own_backtracks = NULL;
    }
  }

if (local_quit_available)
  {
  common->local_quit_available = save_local_quit_available;
  common->quit_label = save_quit_label;
  common->quit = save_quit;
  }

common->in_positive_assertion = save_in_positive_assertion;
common->restore_end_ptr = save_restore_end_ptr;
common->then_trap = save_then_trap;
common->accept_label = save_accept_label;
common->positive_assertion_quit = save_positive_assertion_quit;
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
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), (ket != OP_KET || has_alternatives) ? STACK(-2) : STACK(-1));

  /* TMP2 which is set here used by OP_KETRMAX below. */
  if (ket == OP_KETRMAX)
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(STACK_TOP), STACK(-1));
  else if (ket == OP_KETRMIN)
    {
    /* Move the STR_PTR to the private_data_ptr. */
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, SLJIT_MEM1(STACK_TOP), STACK(-1));
    }
  }
else
  {
  stacksize = (ket != OP_KET || has_alternatives) ? 2 : 1;
  OP2(SLJIT_SUB, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr, SLJIT_IMM, (framesize + stacksize) * sizeof(sljit_sw));
  if (needs_control_head)
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(-1));

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

static PCRE2_SPTR SLJIT_FUNC do_script_run(PCRE2_SPTR ptr, PCRE2_SPTR endptr)
{
  if (PRIV(script_run)(ptr, endptr, FALSE))
    return endptr;
  return NULL;
}

#ifdef SUPPORT_UNICODE

static PCRE2_SPTR SLJIT_FUNC do_script_run_utf(PCRE2_SPTR ptr, PCRE2_SPTR endptr)
{
  if (PRIV(script_run)(ptr, endptr, TRUE))
    return endptr;
  return NULL;
}

#endif /* SUPPORT_UNICODE */

static void match_script_run_common(compiler_common *common, int private_data_ptr, backtrack_common *parent)
{
DEFINE_COMPILER;

SLJIT_ASSERT(TMP1 == SLJIT_R0 && STR_PTR == SLJIT_R1);

OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
#ifdef SUPPORT_UNICODE
sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_ARGS2(W, W, W), SLJIT_IMM,
  common->utf ? SLJIT_FUNC_ADDR(do_script_run_utf) : SLJIT_FUNC_ADDR(do_script_run));
#else
sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_ARGS2(W, W, W), SLJIT_IMM, SLJIT_FUNC_ADDR(do_script_run));
#endif

OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_RETURN_REG, 0);
add_jump(compiler, parent->top != NULL ? &parent->top->simple_backtracks : &parent->own_backtracks, CMP(SLJIT_EQUAL, SLJIT_RETURN_REG, 0, SLJIT_IMM, 0));
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
BOOL has_vreverse = FALSE;
struct sljit_jump *jump;
struct sljit_jump *skip;
jump_list *jumplist;
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
else if (opcode == OP_ASSERT_NA || opcode == OP_ASSERTBACK_NA || opcode == OP_ONCE
         || opcode == OP_ASSERT_SCS || opcode == OP_SCRIPT_RUN || opcode == OP_SBRA || opcode == OP_SCOND)
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
  else if (opcode == OP_ONCE || opcode >= OP_SBRA)
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
      braminzero = CMP(SLJIT_EQUAL, STR_PTR, 0, SLJIT_MEM1(TMP1), STACK(-BACKTRACK_AS(bracket_backtrack)->u.framesize - 2));
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
  if (has_alternatives && opcode >= OP_BRA && opcode < OP_SBRA && repeat_type == 0)
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
        OP2(SLJIT_ADD, SLJIT_MEM1(SLJIT_SP), private_data_ptr, STACK_TOP, 0, SLJIT_IMM, needs_control_head ? (2 * sizeof(sljit_sw)) : sizeof(sljit_sw));
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
    OP2(SLJIT_ADD, TMP2, 0, STACK_TOP, 0, SLJIT_IMM, stacksize * sizeof(sljit_sw));

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
    init_frame(common, ccbegin, NULL, BACKTRACK_AS(bracket_backtrack)->u.framesize + stacksize, stacksize + 1);
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
else if (opcode == OP_ASSERTBACK_NA && PRIVATE_DATA(ccbegin + 1))
  {
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
  allocate_stack(common, 4);
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr + sizeof(sljit_sw));
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, STR_PTR, 0);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr + sizeof(sljit_sw), STR_END, 0);
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), TMP2, 0);
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(1), TMP1, 0);
  OP1(SLJIT_MOV, STR_END, 0, STR_PTR, 0);

  has_vreverse = (*matchingpath == OP_VREVERSE);
  if (*matchingpath == OP_REVERSE || has_vreverse)
    matchingpath = compile_reverse_matchingpath(common, matchingpath, backtrack);
  }
else if (opcode == OP_ASSERT_NA || opcode == OP_ASSERTBACK_NA || opcode == OP_SCRIPT_RUN || opcode == OP_SBRA || opcode == OP_SCOND)
  {
  /* Saving the previous value. */
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
  allocate_stack(common, 1);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, STR_PTR, 0);
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), TMP2, 0);

  if (*matchingpath == OP_REVERSE)
    matchingpath = compile_reverse_matchingpath(common, matchingpath, backtrack);
  }
else if (opcode == OP_ASSERT_SCS)
  {
  /* Nested scs blocks will not update this variable. */
  if (common->restore_end_ptr == 0)
    common->restore_end_ptr = private_data_ptr + sizeof(sljit_sw);

  if (*matchingpath == OP_CREF && (matchingpath[1 + IMM2_SIZE] != OP_CREF && matchingpath[1 + IMM2_SIZE] != OP_DNCREF))
    {
    /* Optimized case for a single capture reference. */
    i = OVECTOR(GET2(matchingpath, 1) << 1);

    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), i);

    add_jump(compiler, &(BACKTRACK_AS(bracket_backtrack)->u.no_capture), CMP(SLJIT_EQUAL, TMP2, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(1)));
    matchingpath += 1 + IMM2_SIZE;

    allocate_stack(common, has_alternatives ? 3 : 2);

    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
    OP1(SLJIT_MOV, SLJIT_TMP_DEST_REG, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr + sizeof(sljit_sw));
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr + sizeof(sljit_sw), STR_END, 0);
    OP1(SLJIT_MOV, STR_END, 0, SLJIT_MEM1(SLJIT_SP), i + sizeof(sljit_sw));
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, STR_PTR, 0);
    OP1(SLJIT_MOV, STR_PTR, 0, TMP2, 0);
    }
  else
    {
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(1));
    jumplist = NULL;

    while (TRUE)
      {
      if (*matchingpath == OP_CREF)
        {
        sljit_get_local_base(compiler, TMP2, 0, OVECTOR(GET2(matchingpath, 1) << 1));
        matchingpath += 1 + IMM2_SIZE;
        }
      else
        {
        SLJIT_ASSERT(*matchingpath == OP_DNCREF);

        i = GET2(matchingpath, 1 + IMM2_SIZE);
        slot = common->name_table + GET2(matchingpath, 1) * common->name_entry_size;

        while (i-- > 1)
          {
          sljit_get_local_base(compiler, TMP2, 0, OVECTOR(GET2(slot, 0) << 1));
          add_jump(compiler, &jumplist, CMP(SLJIT_NOT_EQUAL, SLJIT_MEM1(TMP2), 0, TMP1, 0));
          slot += common->name_entry_size;
          }

        sljit_get_local_base(compiler, TMP2, 0, OVECTOR(GET2(slot, 0) << 1));
        matchingpath += 1 + 2 * IMM2_SIZE;
        }

      if (*matchingpath != OP_CREF && *matchingpath != OP_DNCREF)
        break;

      add_jump(compiler, &jumplist, CMP(SLJIT_NOT_EQUAL, SLJIT_MEM1(TMP2), 0, TMP1, 0));
      }

    add_jump(compiler, &(BACKTRACK_AS(bracket_backtrack)->u.no_capture),
      CMP(SLJIT_EQUAL, SLJIT_MEM1(TMP2), 0, TMP1, 0));

    set_jumps(jumplist, LABEL());

    allocate_stack(common, has_alternatives ? 3 : 2);

    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
    OP1(SLJIT_MOV, SLJIT_TMP_DEST_REG, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr + sizeof(sljit_sw));
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, STR_PTR, 0);
    OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(TMP2), 0);
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr + sizeof(sljit_sw), STR_END, 0);
    OP1(SLJIT_MOV, STR_END, 0, SLJIT_MEM1(TMP2), sizeof(sljit_sw));
    }

  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), TMP1, 0);
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(1), SLJIT_TMP_DEST_REG, 0);

  if (has_alternatives)
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(2), STR_PTR, 0);
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
    add_jump(compiler, &(BACKTRACK_AS(bracket_backtrack)->u.no_capture),
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
    OP2(SLJIT_SUB | SLJIT_SET_Z, TMP2, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(GET2(slot, 0) << 1), TMP1, 0);
    slot += common->name_entry_size;
    i--;
    while (i-- > 0)
      {
      OP2(SLJIT_SUB, STR_PTR, 0, SLJIT_MEM1(SLJIT_SP), OVECTOR(GET2(slot, 0) << 1), TMP1, 0);
      OP2(SLJIT_OR | SLJIT_SET_Z, TMP2, 0, TMP2, 0, STR_PTR, 0);
      slot += common->name_entry_size;
      }
    OP1(SLJIT_MOV, STR_PTR, 0, TMP3, 0);
    add_jump(compiler, &(BACKTRACK_AS(bracket_backtrack)->u.no_capture), JUMP(SLJIT_ZERO));
    matchingpath += 1 + 2 * IMM2_SIZE;
    }
  else if ((*matchingpath >= OP_RREF && *matchingpath <= OP_TRUE) || *matchingpath == OP_FAIL)
    {
    /* Never has other case. */
    BACKTRACK_AS(bracket_backtrack)->u.no_capture = NULL;
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

switch (opcode)
  {
  case OP_ASSERTBACK_NA:
    if (has_vreverse)
      {
      SLJIT_ASSERT(backtrack->top != NULL && PRIVATE_DATA(ccbegin + 1));
      add_jump(compiler, &backtrack->top->simple_backtracks, CMP(SLJIT_LESS, STR_PTR, 0, STR_END, 0));
      }

    if (PRIVATE_DATA(ccbegin + 1))
      OP1(SLJIT_MOV, STR_END, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr + sizeof(sljit_sw));
    break;
  case OP_ONCE:
    match_once_common(common, ket, BACKTRACK_AS(bracket_backtrack)->u.framesize, private_data_ptr, has_alternatives, needs_control_head);
    break;
  case OP_SCRIPT_RUN:
    match_script_run_common(common, private_data_ptr, backtrack);
    break;
  }

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

/* Skip and count the other alternatives. */
i = 1;
while (*cc == OP_ALT)
  {
  cc += GET(cc, 1);
  i++;
  }

if (has_alternatives)
  {
  if (opcode != OP_ONCE)
    {
    if (i <= 3)
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(stacksize), SLJIT_IMM, 0);
    else
      BACKTRACK_AS(bracket_backtrack)->matching_mov_addr = sljit_emit_mov_addr(compiler, SLJIT_MEM1(STACK_TOP), STACK(stacksize));
    }
  if (ket != OP_KETRMAX)
    BACKTRACK_AS(bracket_backtrack)->alternative_matchingpath = LABEL();
  }

/* Must be after the matchingpath label. */
if (offset != 0 && common->optimized_cbracket[offset >> 1] != 0)
  {
  SLJIT_ASSERT(private_data_ptr == OVECTOR(offset + 0));
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1), STR_PTR, 0);
  }
else switch (opcode)
  {
  case OP_ASSERT_NA:
    OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
    break;
  case OP_ASSERT_SCS:
    OP1(SLJIT_MOV, TMP1, 0, STR_END, 0);
    OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
    OP1(SLJIT_MOV, STR_END, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr + sizeof(sljit_sw));
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr + sizeof(sljit_sw), TMP1, 0);

    /* Nested scs blocks will not update this variable. */
    if (common->restore_end_ptr == private_data_ptr + SSIZE_OF(sw))
      common->restore_end_ptr = 0;
    break;
  }

if (ket == OP_KETRMAX)
  {
  if (repeat_type != 0)
    {
    if (has_alternatives)
      BACKTRACK_AS(bracket_backtrack)->alternative_matchingpath = LABEL();
    OP2(SLJIT_SUB | SLJIT_SET_Z, SLJIT_MEM1(SLJIT_SP), repeat_ptr, SLJIT_MEM1(SLJIT_SP), repeat_ptr, SLJIT_IMM, 1);
    JUMPTO(SLJIT_NOT_ZERO, rmax_label);
    /* Drop STR_PTR for greedy plus quantifier. */
    if (opcode != OP_ONCE)
      free_stack(common, 1);
    }
  else if (opcode < OP_BRA || opcode >= OP_SBRA)
    {
    if (has_alternatives)
      BACKTRACK_AS(bracket_backtrack)->alternative_matchingpath = LABEL();

    /* Checking zero-length iteration. */
    if (opcode != OP_ONCE)
      {
      /* This case includes opcodes such as OP_SCRIPT_RUN. */
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
  OP2(SLJIT_SUB | SLJIT_SET_Z, SLJIT_MEM1(SLJIT_SP), repeat_ptr, SLJIT_MEM1(SLJIT_SP), repeat_ptr, SLJIT_IMM, 1);
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
    if (opcode == OP_ONCE)
      {
      int framesize = BACKTRACK_AS(bracket_backtrack)->u.framesize;

      SLJIT_ASSERT(framesize != 0);
      if (framesize > 0)
        {
        OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
        add_jump(compiler, &common->revertframes, JUMP(SLJIT_FAST_CALL));
        OP2(SLJIT_ADD, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, (framesize - 1) * sizeof(sljit_sw));
        }
      }
    else if (ket == OP_KETRMIN)
      free_stack(common, 1);
    }
  /* Continue to the normal backtrack. */
  }

if ((ket != OP_KET && bra != OP_BRAMINZERO) || bra == OP_BRAZERO || (has_alternatives && repeat_type != OP_EXACT))
  count_match(common);

cc += 1 + LINK_SIZE;

if (opcode == OP_ONCE)
  {
  int data;
  int framesize = BACKTRACK_AS(bracket_backtrack)->u.framesize;

  SLJIT_ASSERT(SHRT_MIN <= framesize && framesize < SHRT_MAX/2);
  /* We temporarily encode the needs_control_head in the lowest bit.
     The real value should be short enough for this operation to work
     without triggering Undefined Behaviour. */
  data = (int)((short)((unsigned short)framesize << 1) | (needs_control_head ? 1 : 0));
  BACKTRACK_AS(bracket_backtrack)->u.framesize = data;
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
  /* This case cannot be optimized in the same way as
  normal capturing brackets. */
  SLJIT_ASSERT(common->optimized_cbracket[offset] == 0);
  cbraprivptr = OVECTOR_PRIV(offset);
  offset <<= 1;
  ccbegin = cc + 1 + LINK_SIZE + IMM2_SIZE;
  break;

  default:
  SLJIT_UNREACHABLE();
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
  OP2(SLJIT_ADD, SLJIT_MEM1(SLJIT_SP), private_data_ptr, STACK_TOP, 0, SLJIT_IMM, stacksize * sizeof(sljit_sw));

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
  init_frame(common, cc, NULL, stacksize - 1, stacksize - framesize);
  stack -= 1 + (offset == 0);
  }

if (offset != 0)
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), cbraprivptr, STR_PTR, 0);

loop = LABEL();
while (*cc != OP_KETRPOS)
  {
  backtrack->top = NULL;
  backtrack->own_backtracks = NULL;
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
      OP2(SLJIT_SUB, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr, SLJIT_IMM, stacksize * sizeof(sljit_sw));
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
      OP2(SLJIT_SUB, STACK_TOP, 0, TMP2, 0, SLJIT_IMM, stacksize * sizeof(sljit_sw));
      if (opcode == OP_SBRAPOS)
        OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(TMP2), STACK(-framesize - 2));
      OP1(SLJIT_MOV, SLJIT_MEM1(TMP2), STACK(-framesize - 2), STR_PTR, 0);
      }

    /* Even if the match is empty, we need to reset the control head. */
    if (needs_control_head)
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, SLJIT_MEM1(STACK_TOP), STACK(stack));

    if (opcode == OP_SBRAPOS || opcode == OP_SCBRAPOS)
      add_jump(compiler, &emptymatch, CMP(SLJIT_EQUAL, TMP1, 0, STR_PTR, 0));

    if (!zero)
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), SLJIT_IMM, 0);
    }

  JUMPTO(SLJIT_JUMP, loop);
  flush_stubs(common);

  compile_backtrackingpath(common, backtrack->top);
  if (SLJIT_UNLIKELY(sljit_get_compiler_error(compiler)))
    return NULL;
  set_jumps(backtrack->own_backtracks, LABEL());

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
      OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(TMP2), STACK(-framesize - 2));
      }
    }

  if (*cc == OP_KETRPOS)
    break;
  ccbegin = cc + 1 + LINK_SIZE;
  }

/* We don't have to restore the control head in case of a failed match. */

backtrack->own_backtracks = NULL;
if (!zero)
  {
  if (framesize < 0)
    add_jump(compiler, &backtrack->own_backtracks, CMP(SLJIT_NOT_EQUAL, SLJIT_MEM1(STACK_TOP), STACK(stacksize - 1), SLJIT_IMM, 0));
  else /* TMP2 is set to [private_data_ptr] above. */
    add_jump(compiler, &backtrack->own_backtracks, CMP(SLJIT_NOT_EQUAL, SLJIT_MEM1(TMP2), STACK(-stacksize), SLJIT_IMM, 0));
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
  SLJIT_ASSERT(*opcode == OP_CLASS || *opcode == OP_NCLASS || *opcode == OP_XCLASS || *opcode == OP_ECLASS);
  *type = *opcode;
  class_len = (*type < OP_XCLASS) ? (int)(1 + (32 / sizeof(PCRE2_UCHAR))) : GET(cc, 1);
  *opcode = cc[class_len];
  cc++;

  if (*opcode >= OP_CRSTAR && *opcode <= OP_CRMINQUERY)
    {
    *opcode -= OP_CRSTAR - OP_STAR;
    *end = cc + class_len;

    if (*opcode == OP_PLUS || *opcode == OP_MINPLUS)
      {
      *exact = 1;
      *opcode -= OP_PLUS - OP_STAR;
      }
    return cc;
    }

  if (*opcode >= OP_CRPOSSTAR && *opcode <= OP_CRPOSQUERY)
    {
    *opcode -= OP_CRPOSSTAR - OP_POSSTAR;
    *end = cc + class_len;

    if (*opcode == OP_POSPLUS)
      {
      *exact = 1;
      *opcode = OP_POSSTAR;
      }
    return cc;
    }

  SLJIT_ASSERT(*opcode == OP_CRRANGE || *opcode == OP_CRMINRANGE || *opcode == OP_CRPOSRANGE);
  *max = GET2(cc, (class_len + IMM2_SIZE));
  *exact = GET2(cc, class_len);
  *end = cc + class_len + 2 * IMM2_SIZE;

  if (*max == 0)
    {
    SLJIT_ASSERT(*exact > 1);
    if (*opcode == OP_CRRANGE)
      *opcode = OP_UPTO;
    else if (*opcode == OP_CRPOSRANGE)
      *opcode = OP_POSUPTO;
    else
      *opcode = OP_MINSTAR;
    return cc;
    }

  *max -= *exact;
  if (*max == 0)
    *opcode = OP_EXACT;
  else
    {
    SLJIT_ASSERT(*exact > 0 || *max > 1);
    if (*opcode == OP_CRRANGE)
      *opcode = OP_UPTO;
    else if (*opcode == OP_CRPOSRANGE)
      *opcode = OP_POSUPTO;
    else if (*max == 1)
      *opcode = OP_MINQUERY;
    else
      *opcode = OP_MINUPTO;
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

static PCRE2_SPTR compile_iterator_matchingpath(compiler_common *common, PCRE2_SPTR cc, backtrack_common *parent, jump_list **prev_backtracks)
{
DEFINE_COMPILER;
backtrack_common *backtrack = NULL;
PCRE2_SPTR begin = cc;
PCRE2_UCHAR opcode;
PCRE2_UCHAR type;
sljit_u32 max = 0, exact;
sljit_s32 early_fail_ptr = PRIVATE_DATA(cc + 1);
sljit_s32 early_fail_type;
BOOL charpos_enabled, use_tmp;
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
int offset1 = (private_data_ptr == 0) ? STACK(1) : private_data_ptr + SSIZE_OF(sw);
int tmp_base, tmp_offset;

early_fail_type = (early_fail_ptr & 0x7);
early_fail_ptr >>= 3;

/* During recursion, these optimizations are disabled. */
if (common->early_fail_start_ptr == 0 && common->fast_forward_bc_ptr == NULL)
  {
  early_fail_ptr = 0;
  early_fail_type = type_skip;
  }

SLJIT_ASSERT(common->fast_forward_bc_ptr != NULL || early_fail_ptr == 0
  || (early_fail_ptr >= common->early_fail_start_ptr && early_fail_ptr <= common->early_fail_end_ptr));

if (early_fail_type == type_fail)
  add_jump(compiler, prev_backtracks, CMP(SLJIT_LESS_EQUAL, STR_PTR, 0, SLJIT_MEM1(SLJIT_SP), early_fail_ptr));

cc = get_iterator_parameters(common, cc, &opcode, &type, &max, &exact, &end);

if (type != OP_EXTUNI)
  {
  tmp_base = TMP3;
  tmp_offset = 0;
  }
else
  {
  tmp_base = SLJIT_MEM1(SLJIT_SP);
  tmp_offset = LOCAL2;
  }

if (opcode == OP_EXACT)
  {
  SLJIT_ASSERT(early_fail_ptr == 0 && exact >= 2);

  if (common->mode == PCRE2_JIT_COMPLETE
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
      && !common->utf
#endif
      && type != OP_ANYNL && type != OP_EXTUNI)
    {
    OP2(SLJIT_SUB, TMP1, 0, STR_END, 0, STR_PTR, 0);
    add_jump(compiler, prev_backtracks, CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, IN_UCHARS(exact)));

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 32
    if (type == OP_ALLANY && !common->invalid_utf)
#else
    if (type == OP_ALLANY)
#endif
      OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(exact));
    else
      {
      OP1(SLJIT_MOV, tmp_base, tmp_offset, SLJIT_IMM, exact);
      label = LABEL();
      compile_char1_matchingpath(common, type, cc, prev_backtracks, FALSE);
      OP2(SLJIT_SUB | SLJIT_SET_Z, tmp_base, tmp_offset, tmp_base, tmp_offset, SLJIT_IMM, 1);
      JUMPTO(SLJIT_NOT_ZERO, label);
      }
    }
  else
    {
    SLJIT_ASSERT(tmp_base == TMP3 || common->locals_size >= 3 * SSIZE_OF(sw));
    OP1(SLJIT_MOV, tmp_base, tmp_offset, SLJIT_IMM, exact);
    label = LABEL();
    compile_char1_matchingpath(common, type, cc, prev_backtracks, TRUE);
    OP2(SLJIT_SUB | SLJIT_SET_Z, tmp_base, tmp_offset, tmp_base, tmp_offset, SLJIT_IMM, 1);
    JUMPTO(SLJIT_NOT_ZERO, label);
    }
  }

if (early_fail_type == type_fail_range)
  {
  /* Range end first, followed by range start. */
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), early_fail_ptr);
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), early_fail_ptr + SSIZE_OF(sw));
  OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, TMP2, 0);
  OP2(SLJIT_SUB, TMP2, 0, STR_PTR, 0, TMP2, 0);
  add_jump(compiler, prev_backtracks, CMP(SLJIT_LESS_EQUAL, TMP2, 0, TMP1, 0));

  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), early_fail_ptr, STR_PTR, 0);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), early_fail_ptr + SSIZE_OF(sw), STR_PTR, 0);
  }

if (opcode < OP_EXACT)
  PUSH_BACKTRACK(sizeof(char_iterator_backtrack), begin, NULL);

switch(opcode)
  {
  case OP_STAR:
  case OP_UPTO:
  SLJIT_ASSERT(backtrack != NULL && (early_fail_ptr == 0 || opcode == OP_STAR));
  max += exact;

  if (type == OP_EXTUNI)
    {
    SLJIT_ASSERT(private_data_ptr == 0);
    SLJIT_ASSERT(early_fail_ptr == 0);

    if (exact == 1)
      {
      SLJIT_ASSERT(opcode == OP_STAR);
      allocate_stack(common, 1);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), SLJIT_IMM, 0);
      }
    else
      {
      /* If OP_EXTUNI is present, it has a separate EXACT opcode. */
      SLJIT_ASSERT(exact == 0);

      allocate_stack(common, 2);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), STR_PTR, 0);
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(1), SLJIT_IMM, 0);
      }

    if (opcode == OP_UPTO)
      {
      SLJIT_ASSERT(common->locals_size >= 3 * SSIZE_OF(sw));
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCAL2, SLJIT_IMM, max);
      }

    label = LABEL();
    compile_char1_matchingpath(common, type, cc, &backtrack->own_backtracks, TRUE);
    if (opcode == OP_UPTO)
      {
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), LOCAL2);
      OP2(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, TMP1, 0, SLJIT_IMM, 1);
      jump = JUMP(SLJIT_ZERO);
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCAL2, TMP1, 0);
      }

    /* We cannot use TMP3 because of allocate_stack. */
    allocate_stack(common, 1);
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), STR_PTR, 0);
    JUMPTO(SLJIT_JUMP, label);
    if (jump != NULL)
      JUMPHERE(jump);
    BACKTRACK_AS(char_iterator_backtrack)->matchingpath = LABEL();
    break;
    }
#ifdef SUPPORT_UNICODE
  else if (type == OP_ALLANY && !common->invalid_utf)
#else
  else if (type == OP_ALLANY)
#endif
    {
    if (opcode == OP_STAR)
      {
      if (exact == 1)
        detect_partial_match(common, prev_backtracks);

      if (private_data_ptr == 0)
        allocate_stack(common, 2);

      OP1(SLJIT_MOV, base, offset0, STR_END, 0);
      OP1(SLJIT_MOV, base, offset1, STR_PTR, 0);

      OP1(SLJIT_MOV, STR_PTR, 0, STR_END, 0);
      process_partial_match(common);

      if (early_fail_ptr != 0)
        OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), early_fail_ptr, STR_END, 0);
      BACKTRACK_AS(char_iterator_backtrack)->matchingpath = LABEL();
      break;
      }
#ifdef SUPPORT_UNICODE
    else if (!common->utf)
#else
    else
#endif
      {
      /* If OP_ALLANY is present, it has a separate EXACT opcode. */
      SLJIT_ASSERT(exact == 0);

      if (private_data_ptr == 0)
        allocate_stack(common, 2);

      OP1(SLJIT_MOV, base, offset1, STR_PTR, 0);
      OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(max));

      if (common->mode == PCRE2_JIT_COMPLETE)
        {
        OP2U(SLJIT_SUB | SLJIT_SET_GREATER, STR_PTR, 0, STR_END, 0);
        SELECT(SLJIT_GREATER, STR_PTR, STR_END, 0, STR_PTR);
        }
      else
        {
        jump = CMP(SLJIT_LESS_EQUAL, STR_PTR, 0, STR_END, 0);
        process_partial_match(common);
        JUMPHERE(jump);
        }

      OP1(SLJIT_MOV, base, offset0, STR_PTR, 0);

      if (early_fail_ptr != 0)
        OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), early_fail_ptr, STR_PTR, 0);
      BACKTRACK_AS(char_iterator_backtrack)->matchingpath = LABEL();
      break;
      }
    }

  charpos_enabled = FALSE;
  charpos_char = 0;
  charpos_othercasebit = 0;

  SLJIT_ASSERT(tmp_base == TMP3);
  if ((type != OP_CHAR && type != OP_CHARI) && (*end == OP_CHAR || *end == OP_CHARI))
    {
#ifdef SUPPORT_UNICODE
    charpos_enabled = !common->utf || !HAS_EXTRALEN(end[1]);
#else
    charpos_enabled = TRUE;
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
      /* Consume the OP_CHAR opcode. */
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

      BACKTRACK_AS(char_iterator_backtrack)->charpos.charpos_enabled = TRUE;
      BACKTRACK_AS(char_iterator_backtrack)->charpos.chr = charpos_char;
      BACKTRACK_AS(char_iterator_backtrack)->charpos.othercasebit = charpos_othercasebit;

      if (private_data_ptr == 0)
        allocate_stack(common, 2);

      use_tmp = (opcode == OP_STAR);

      if (use_tmp)
        {
        OP1(SLJIT_MOV, TMP3, 0, SLJIT_IMM, 0);
        OP1(SLJIT_MOV, base, offset0, TMP3, 0);
        }
      else
        {
        OP1(SLJIT_MOV, base, offset1, COUNT_MATCH, 0);
        OP1(SLJIT_MOV, COUNT_MATCH, 0, SLJIT_IMM, 0);
        OP1(SLJIT_MOV, base, offset0, COUNT_MATCH, 0);
        OP1(SLJIT_MOV, TMP3, 0, SLJIT_IMM, exact == max ? 0 : (max + 1));
        }

      /* Search the first instance of charpos_char. */
      if (exact > 0)
        detect_partial_match(common, &no_match);
      else
        jump = JUMP(SLJIT_JUMP);

      label = LABEL();

      if (opcode == OP_UPTO)
        {
        if (exact == max)
          OP2(SLJIT_ADD, TMP3, 0, TMP3, 0, SLJIT_IMM, 1);
        else
          {
          OP2(SLJIT_SUB | SLJIT_SET_Z, TMP3, 0, TMP3, 0, SLJIT_IMM, 1);
          add_jump(compiler, &no_match, JUMP(SLJIT_ZERO));
          }
        }

      compile_char1_matchingpath(common, type, cc, &no_match, FALSE);

      if (early_fail_ptr != 0)
        OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), early_fail_ptr, STR_PTR, 0);

      if (exact == 0)
        JUMPHERE(jump);

      detect_partial_match(common, &no_match);

      if (opcode == OP_UPTO && exact > 0)
        {
        if (exact == max)
          CMPTO(SLJIT_LESS, TMP3, 0, SLJIT_IMM, exact, label);
        else
          CMPTO(SLJIT_GREATER, TMP3, 0, SLJIT_IMM, (max + 1) - exact, label);
        }

      OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
      if (charpos_othercasebit != 0)
        OP2(SLJIT_OR, TMP1, 0, TMP1, 0, SLJIT_IMM, charpos_othercasebit);
      CMPTO(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, charpos_char, label);

      OP1(SLJIT_MOV, base, offset0, STR_PTR, 0);
      if (use_tmp)
        {
        OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP3, 0, SLJIT_IMM, 0);
        SELECT(SLJIT_EQUAL, TMP3, STR_PTR, 0, TMP3);
        }
      else
        {
        OP2U(SLJIT_SUB | SLJIT_SET_Z, COUNT_MATCH, 0, SLJIT_IMM, 0);
        SELECT(SLJIT_EQUAL, COUNT_MATCH, STR_PTR, 0, COUNT_MATCH);
        }
      JUMPTO(SLJIT_JUMP, label);

      set_jumps(no_match, LABEL());
      OP1(SLJIT_MOV, STR_PTR, 0, base, offset0);
      if (use_tmp)
        OP1(SLJIT_MOV, base, offset1, TMP3, 0);
      else
        {
        OP1(SLJIT_MOV, TMP1, 0, base, offset1);
        OP1(SLJIT_MOV, base, offset1, COUNT_MATCH, 0);
        OP1(SLJIT_MOV, COUNT_MATCH, 0, TMP1, 0);
        }

      add_jump(compiler, &backtrack->own_backtracks, CMP(SLJIT_EQUAL, STR_PTR, 0, SLJIT_IMM, 0));

      BACKTRACK_AS(char_iterator_backtrack)->matchingpath = LABEL();
      OP1(SLJIT_MOV, base, offset0, STR_PTR, 0);
      OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
      break;
      }
    }

  if (private_data_ptr == 0)
    allocate_stack(common, 2);

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
  use_tmp = (opcode == OP_STAR);

  if (common->utf)
    {
    if (!use_tmp)
      OP1(SLJIT_MOV, base, offset0, COUNT_MATCH, 0);

    OP1(SLJIT_MOV, use_tmp ? TMP3 : COUNT_MATCH, 0, STR_PTR, 0);
    }
#endif

  if (opcode == OP_UPTO)
    OP1(SLJIT_MOV, TMP3, 0, SLJIT_IMM, exact == max ? -(sljit_sw)exact : (sljit_sw)max);

  if (opcode == OP_UPTO && exact > 0)
    {
    label = LABEL();
    detect_partial_match(common, &no_match);
    compile_char1_matchingpath(common, type, cc, &no_char1_match, FALSE);
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
    if (common->utf)
      OP1(SLJIT_MOV, use_tmp ? TMP3 : COUNT_MATCH, 0, STR_PTR, 0);
#endif

    if (exact == max)
      {
      OP2(SLJIT_ADD | SLJIT_SET_Z, TMP3, 0, TMP3, 0, SLJIT_IMM, 1);
      JUMPTO(SLJIT_NOT_ZERO, label);
      }
    else
      {
      OP2(SLJIT_SUB | SLJIT_SET_Z, TMP3, 0, TMP3, 0, SLJIT_IMM, 1);
      add_jump(compiler, &no_match, JUMP(SLJIT_ZERO));
      CMPTO(SLJIT_NOT_EQUAL, TMP3, 0, SLJIT_IMM, max - exact, label);
      }

    OP1(SLJIT_MOV, base, offset1, STR_PTR, 0);
    JUMPTO(SLJIT_JUMP, label);
    }
  else
    {
    OP1(SLJIT_MOV, base, offset1, STR_PTR, 0);

    detect_partial_match(common, &no_match);
    label = LABEL();
    compile_char1_matchingpath(common, type, cc, &no_char1_match, FALSE);
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
    if (common->utf)
      OP1(SLJIT_MOV, use_tmp ? TMP3 : COUNT_MATCH, 0, STR_PTR, 0);
#endif

    if (opcode == OP_UPTO)
      {
      OP2(SLJIT_SUB | SLJIT_SET_Z, TMP3, 0, TMP3, 0, SLJIT_IMM, 1);
      add_jump(compiler, &no_match, JUMP(SLJIT_ZERO));
      }

    detect_partial_match_to(common, label);
    }

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
  if (common->utf)
    {
    set_jumps(no_char1_match, LABEL());
    set_jumps(no_match, LABEL());
    if (use_tmp)
      {
      OP1(SLJIT_MOV, STR_PTR, 0, TMP3, 0);
      OP1(SLJIT_MOV, base, offset0, TMP3, 0);
      }
    else
      {
      OP1(SLJIT_MOV, STR_PTR, 0, COUNT_MATCH, 0);
      OP1(SLJIT_MOV, COUNT_MATCH, 0, base, offset0);
      OP1(SLJIT_MOV, base, offset0, STR_PTR, 0);
      }
    }
  else
#endif
    {
    if (opcode != OP_UPTO || exact == 0)
      OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
    set_jumps(no_char1_match, LABEL());

    OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
    set_jumps(no_match, LABEL());
    OP1(SLJIT_MOV, base, offset0, STR_PTR, 0);
    }

  if (opcode == OP_UPTO)
    {
    if (exact > 0)
      {
      if (max == exact)
        jump = CMP(SLJIT_GREATER_EQUAL, TMP3, 0, SLJIT_IMM, -(sljit_sw)exact);
      else
        jump = CMP(SLJIT_GREATER, TMP3, 0, SLJIT_IMM, max - exact);

      add_jump(compiler, &backtrack->own_backtracks, jump);
      }
    }
  else if (exact == 1)
    add_jump(compiler, &backtrack->own_backtracks, CMP(SLJIT_EQUAL, base, offset1, STR_PTR, 0));

  if (early_fail_ptr != 0)
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), early_fail_ptr, STR_PTR, 0);

  BACKTRACK_AS(char_iterator_backtrack)->matchingpath = LABEL();
  break;

  case OP_QUERY:
  SLJIT_ASSERT(backtrack != NULL && early_fail_ptr == 0);
  if (private_data_ptr == 0)
    allocate_stack(common, 1);
  OP1(SLJIT_MOV, base, offset0, STR_PTR, 0);
  compile_char1_matchingpath(common, type, cc, &backtrack->own_backtracks, TRUE);
  BACKTRACK_AS(char_iterator_backtrack)->matchingpath = LABEL();
  break;

  case OP_MINSTAR:
  case OP_MINQUERY:
  SLJIT_ASSERT(backtrack != NULL && (opcode == OP_MINSTAR || early_fail_ptr == 0));
  if (private_data_ptr == 0)
    allocate_stack(common, 1);

  if (exact >= 1)
    {
    if (exact >= 2)
      {
      /* Extuni has a separate exact opcode. */
      SLJIT_ASSERT(tmp_base == TMP3 && early_fail_ptr == 0);
      OP1(SLJIT_MOV, TMP3, 0, SLJIT_IMM, exact);
      }

    if (opcode == OP_MINQUERY)
      OP1(SLJIT_MOV, base, offset0, SLJIT_IMM, -1);

    label = LABEL();
    BACKTRACK_AS(char_iterator_backtrack)->matchingpath = label;

    compile_char1_matchingpath(common, type, cc, &backtrack->own_backtracks, TRUE);

    if (exact >= 2)
      {
      OP2(SLJIT_SUB | SLJIT_SET_Z, TMP3, 0, TMP3, 0, SLJIT_IMM, 1);
      JUMPTO(SLJIT_NOT_ZERO, label);
      }

    if (opcode == OP_MINQUERY)
      OP2(SLJIT_AND, base, offset0, base, offset0, STR_PTR, 0);
    else
      OP1(SLJIT_MOV, base, offset0, STR_PTR, 0);
    }
  else
    {
    OP1(SLJIT_MOV, base, offset0, STR_PTR, 0);
    BACKTRACK_AS(char_iterator_backtrack)->matchingpath = LABEL();
    }

  if (early_fail_ptr != 0)
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), early_fail_ptr, STR_PTR, 0);
  break;

  case OP_MINUPTO:
  SLJIT_ASSERT(backtrack != NULL && early_fail_ptr == 0);
  if (private_data_ptr == 0)
    allocate_stack(common, 2);

  OP1(SLJIT_MOV, base, offset1, SLJIT_IMM, max + 1);

  if (exact == 0)
    {
    OP1(SLJIT_MOV, base, offset0, STR_PTR, 0);
    BACKTRACK_AS(char_iterator_backtrack)->matchingpath = LABEL();
    break;
    }

  if (exact >= 2)
    {
    /* Extuni has a separate exact opcode. */
    SLJIT_ASSERT(tmp_base == TMP3);
    OP1(SLJIT_MOV, TMP3, 0, SLJIT_IMM, exact);
    }

  label = LABEL();
  BACKTRACK_AS(char_iterator_backtrack)->matchingpath = label;

  compile_char1_matchingpath(common, type, cc, &backtrack->own_backtracks, TRUE);

  if (exact >= 2)
    {
    OP2(SLJIT_SUB | SLJIT_SET_Z, TMP3, 0, TMP3, 0, SLJIT_IMM, 1);
    JUMPTO(SLJIT_NOT_ZERO, label);
    }

  OP1(SLJIT_MOV, base, offset0, STR_PTR, 0);
  break;

  case OP_EXACT:
  SLJIT_ASSERT(backtrack == NULL);
  break;

  case OP_POSSTAR:
  SLJIT_ASSERT(backtrack == NULL);
#if defined SUPPORT_UNICODE
  if (type == OP_ALLANY && !common->invalid_utf)
#else
  if (type == OP_ALLANY)
#endif
    {
    if (exact == 1)
      detect_partial_match(common, prev_backtracks);

    OP1(SLJIT_MOV, STR_PTR, 0, STR_END, 0);
    process_partial_match(common);
    if (early_fail_ptr != 0)
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), early_fail_ptr, STR_END, 0);
    break;
    }

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
  if (common->utf)
    {
    SLJIT_ASSERT(tmp_base == TMP3 || common->locals_size >= 3 * SSIZE_OF(sw));

    if (tmp_base != TMP3)
      {
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCAL2, COUNT_MATCH, 0);
      tmp_base = COUNT_MATCH;
      }

    OP1(SLJIT_MOV, tmp_base, 0, exact == 1 ? SLJIT_IMM : STR_PTR, 0);
    detect_partial_match(common, &no_match);
    label = LABEL();
    compile_char1_matchingpath(common, type, cc, &no_match, FALSE);
    OP1(SLJIT_MOV, tmp_base, 0, STR_PTR, 0);
    detect_partial_match_to(common, label);

    set_jumps(no_match, LABEL());
    OP1(SLJIT_MOV, STR_PTR, 0, tmp_base, 0);

    if (tmp_base != TMP3)
      OP1(SLJIT_MOV, COUNT_MATCH, 0, SLJIT_MEM1(SLJIT_SP), LOCAL2);

    if (exact == 1)
      add_jump(compiler, prev_backtracks, CMP(SLJIT_EQUAL, STR_PTR, 0, SLJIT_IMM, 0));

    if (early_fail_ptr != 0)
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), early_fail_ptr, STR_PTR, 0);
    break;
    }
#endif

  if (exact == 1)
    OP1(SLJIT_MOV, tmp_base, tmp_offset, STR_PTR, 0);

  detect_partial_match(common, &no_match);
  label = LABEL();
  /* Extuni never fails, so no_char1_match is not used in that case.
     Anynl optionally reads an extra character on success. */
  compile_char1_matchingpath(common, type, cc, &no_char1_match, FALSE);
  detect_partial_match_to(common, label);
  if (type != OP_EXTUNI)
    OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

  set_jumps(no_char1_match, LABEL());
  if (type != OP_EXTUNI)
    OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

  set_jumps(no_match, LABEL());

  if (exact == 1)
    add_jump(compiler, prev_backtracks, CMP(SLJIT_EQUAL, tmp_base, tmp_offset, STR_PTR, 0));

  if (early_fail_ptr != 0)
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), early_fail_ptr, STR_PTR, 0);
  break;

  case OP_POSUPTO:
  SLJIT_ASSERT(backtrack == NULL && early_fail_ptr == 0);
  max += exact;

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
  if (type == OP_EXTUNI || common->utf)
#else
  if (type == OP_EXTUNI)
#endif
    {
    SLJIT_ASSERT(common->locals_size >= 3 * SSIZE_OF(sw));

    /* Count match is not modified by compile_char1_matchingpath. */
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCAL2, COUNT_MATCH, 0);
    OP1(SLJIT_MOV, COUNT_MATCH, 0, SLJIT_IMM, exact == max ? 0 : max);

    label = LABEL();
    /* Extuni only modifies TMP3 on successful match. */
    OP1(SLJIT_MOV, TMP3, 0, STR_PTR, 0);
    compile_char1_matchingpath(common, type, cc, &no_match, TRUE);

    if (exact == max)
      {
      OP2(SLJIT_ADD, COUNT_MATCH, 0, COUNT_MATCH, 0, SLJIT_IMM, 1);
      JUMPTO(SLJIT_JUMP, label);
      }
    else
      {
      OP2(SLJIT_SUB | SLJIT_SET_Z, COUNT_MATCH, 0, COUNT_MATCH, 0, SLJIT_IMM, 1);
      JUMPTO(SLJIT_NOT_ZERO, label);
      OP1(SLJIT_MOV, TMP3, 0, STR_PTR, 0);
      }

    set_jumps(no_match, LABEL());

    if (exact > 0)
      {
      if (exact == max)
        OP2U(SLJIT_SUB | SLJIT_SET_LESS, COUNT_MATCH, 0, SLJIT_IMM, exact);
      else
        OP2U(SLJIT_SUB | SLJIT_SET_GREATER, COUNT_MATCH, 0, SLJIT_IMM, max - exact);
      }

    OP1(SLJIT_MOV, COUNT_MATCH, 0, SLJIT_MEM1(SLJIT_SP), LOCAL2);

    if (exact > 0)
      add_jump(compiler, prev_backtracks, JUMP(exact == max ? SLJIT_LESS : SLJIT_GREATER));
    OP1(SLJIT_MOV, STR_PTR, 0, TMP3, 0);
    break;
    }

  SLJIT_ASSERT(tmp_base == TMP3);

  OP1(SLJIT_MOV, TMP3, 0, SLJIT_IMM, exact == max ? 0 : max);

  detect_partial_match(common, &no_match);
  label = LABEL();
  compile_char1_matchingpath(common, type, cc, &no_char1_match, FALSE);

  if (exact == max)
    OP2(SLJIT_ADD, TMP3, 0, TMP3, 0, SLJIT_IMM, 1);
  else
    {
    OP2(SLJIT_SUB | SLJIT_SET_Z, TMP3, 0, TMP3, 0, SLJIT_IMM, 1);
    add_jump(compiler, &no_match, JUMP(SLJIT_ZERO));
    }
  detect_partial_match_to(common, label);
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));

  set_jumps(no_char1_match, LABEL());
  OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  set_jumps(no_match, LABEL());

  if (exact > 0)
    {
    if (exact == max)
      jump = CMP(SLJIT_LESS, TMP3, 0, SLJIT_IMM, exact);
    else
      jump = CMP(SLJIT_GREATER, TMP3, 0, SLJIT_IMM, max - exact);

    add_jump(compiler, prev_backtracks, jump);
    }
  break;

  case OP_POSQUERY:
  SLJIT_ASSERT(backtrack == NULL && early_fail_ptr == 0);
  SLJIT_ASSERT(tmp_base == TMP3 || common->locals_size >= 3 * SSIZE_OF(sw));
  OP1(SLJIT_MOV, tmp_base, tmp_offset, STR_PTR, 0);
  compile_char1_matchingpath(common, type, cc, &no_match, TRUE);
  OP1(SLJIT_MOV, tmp_base, tmp_offset, STR_PTR, 0);
  set_jumps(no_match, LABEL());
  OP1(SLJIT_MOV, STR_PTR, 0, tmp_base, tmp_offset);
  break;

  default:
  SLJIT_UNREACHABLE();
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
  add_jump(compiler, &backtrack->own_backtracks, JUMP(SLJIT_JUMP));
  return cc + 1;
  }

if (*cc == OP_ACCEPT && common->currententry == NULL && (common->re->overall_options & PCRE2_ENDANCHORED) != 0)
  add_jump(compiler, &common->restart_match, CMP(SLJIT_NOT_EQUAL, STR_PTR, 0, STR_END, 0));

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

if (HAS_VIRTUAL_REGISTERS)
  {
  OP1(SLJIT_MOV, TMP1, 0, ARGUMENTS, 0);
  OP1(SLJIT_MOV_U32, TMP2, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, options));
  }
else
  OP1(SLJIT_MOV_U32, TMP2, 0, SLJIT_MEM1(ARGUMENTS), SLJIT_OFFSETOF(jit_arguments, options));

OP2U(SLJIT_AND | SLJIT_SET_Z, TMP2, 0, SLJIT_IMM, PCRE2_NOTEMPTY);
add_jump(compiler, &backtrack->own_backtracks, JUMP(SLJIT_NOT_ZERO));
OP2U(SLJIT_AND | SLJIT_SET_Z, TMP2, 0, SLJIT_IMM, PCRE2_NOTEMPTY_ATSTART);
if (common->accept_label == NULL)
  add_jump(compiler, &common->accept, JUMP(SLJIT_ZERO));
else
  JUMPTO(SLJIT_ZERO, common->accept_label);

OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(HAS_VIRTUAL_REGISTERS ? TMP1 : ARGUMENTS), SLJIT_OFFSETOF(jit_arguments, str));
if (common->accept_label == NULL)
  add_jump(compiler, &common->accept, CMP(SLJIT_NOT_EQUAL, TMP2, 0, STR_PTR, 0));
else
  CMPTO(SLJIT_NOT_EQUAL, TMP2, 0, STR_PTR, 0, common->accept_label);
add_jump(compiler, &backtrack->own_backtracks, JUMP(SLJIT_JUMP));
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

if (opcode == OP_COMMIT_ARG || opcode == OP_PRUNE_ARG ||
    opcode == OP_SKIP_ARG || opcode == OP_THEN_ARG)
  ccend += 2 + cc[1];

PUSH_BACKTRACK(sizeof(backtrack_common), cc, NULL);

if (opcode == OP_SKIP)
  {
  allocate_stack(common, 1);
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), STR_PTR, 0);
  return ccend;
  }

if (opcode == OP_COMMIT_ARG || opcode == OP_PRUNE_ARG || opcode == OP_THEN_ARG)
  {
  if (HAS_VIRTUAL_REGISTERS)
    OP1(SLJIT_MOV, TMP1, 0, ARGUMENTS, 0);
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, (sljit_sw)(cc + 2));
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->mark_ptr, TMP2, 0);
  OP1(SLJIT_MOV, SLJIT_MEM1(HAS_VIRTUAL_REGISTERS ? TMP1 : ARGUMENTS), SLJIT_OFFSETOF(jit_arguments, mark_ptr), TMP2, 0);
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
  OP2(SLJIT_ADD, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, STACK_TOP, 0, SLJIT_IMM, (size - 3) * sizeof(sljit_sw));
else
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, STACK_TOP, 0);
OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(size - 1), SLJIT_IMM, BACKTRACK_AS(then_trap_backtrack)->start);
OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(size - 2), SLJIT_IMM, type_then_trap);
OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(size - 3), TMP2, 0);

size = BACKTRACK_AS(then_trap_backtrack)->framesize;
if (size >= 0)
  init_frame(common, cc, ccend, size - 1, 0);
}

static void compile_matchingpath(compiler_common *common, PCRE2_SPTR cc, PCRE2_SPTR ccend, backtrack_common *parent)
{
DEFINE_COMPILER;
backtrack_common *backtrack;
BOOL has_then_trap = FALSE;
then_trap_backtrack *save_then_trap = NULL;
size_t op_len;

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
    case OP_NOT_UCP_WORD_BOUNDARY:
    case OP_UCP_WORD_BOUNDARY:
    cc = compile_simple_assertion_matchingpath(common, *cc, cc + 1, parent->top != NULL ? &parent->top->simple_backtracks : &parent->own_backtracks);
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
    cc = compile_char1_matchingpath(common, *cc, cc + 1, parent->top != NULL ? &parent->top->simple_backtracks : &parent->own_backtracks, TRUE);
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
      cc = compile_charn_matchingpath(common, cc, ccend, parent->top != NULL ? &parent->top->simple_backtracks : &parent->own_backtracks);
    else
      cc = compile_char1_matchingpath(common, *cc, cc + 1, parent->top != NULL ? &parent->top->simple_backtracks : &parent->own_backtracks, TRUE);
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
    cc = compile_iterator_matchingpath(common, cc, parent, parent->top != NULL ? &parent->top->simple_backtracks : &parent->own_backtracks);
    break;

    case OP_CLASS:
    case OP_NCLASS:
    if (cc[1 + (32 / sizeof(PCRE2_UCHAR))] >= OP_CRSTAR && cc[1 + (32 / sizeof(PCRE2_UCHAR))] <= OP_CRPOSRANGE)
      cc = compile_iterator_matchingpath(common, cc, parent, parent->top != NULL ? &parent->top->simple_backtracks : &parent->own_backtracks);
    else
      cc = compile_char1_matchingpath(common, *cc, cc + 1, parent->top != NULL ? &parent->top->simple_backtracks : &parent->own_backtracks, TRUE);
    break;

#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32
    case OP_XCLASS:
    case OP_ECLASS:
    op_len = GET(cc, 1);
    if (cc[op_len] >= OP_CRSTAR && cc[op_len] <= OP_CRPOSRANGE)
      cc = compile_iterator_matchingpath(common, cc, parent, parent->top != NULL ? &parent->top->simple_backtracks : &parent->own_backtracks);
    else
      cc = compile_char1_matchingpath(common, *cc, cc + 1, parent->top != NULL ? &parent->top->simple_backtracks : &parent->own_backtracks, TRUE);
    break;
#endif

    case OP_REF:
    case OP_REFI:
    op_len = PRIV(OP_lengths)[*cc];
    if (cc[op_len] >= OP_CRSTAR && cc[op_len] <= OP_CRPOSRANGE)
      cc = compile_ref_iterator_matchingpath(common, cc, parent);
    else
      {
      compile_ref_matchingpath(common, cc, parent->top != NULL ? &parent->top->simple_backtracks : &parent->own_backtracks, TRUE, FALSE);
      cc += op_len;
      }
    break;

    case OP_DNREF:
    case OP_DNREFI:
    op_len = PRIV(OP_lengths)[*cc];
    if (cc[op_len] >= OP_CRSTAR && cc[op_len] <= OP_CRPOSRANGE)
      cc = compile_ref_iterator_matchingpath(common, cc, parent);
    else
      {
      compile_dnref_search(common, cc, parent->top != NULL ? &parent->top->simple_backtracks : &parent->own_backtracks);
      compile_ref_matchingpath(common, cc, parent->top != NULL ? &parent->top->simple_backtracks : &parent->own_backtracks, TRUE, FALSE);
      cc += op_len;
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

    case OP_ASSERT_NA:
    case OP_ASSERTBACK_NA:
    case OP_ASSERT_SCS:
    case OP_ONCE:
    case OP_SCRIPT_RUN:
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
    if (HAS_VIRTUAL_REGISTERS)
      OP1(SLJIT_MOV, TMP1, 0, ARGUMENTS, 0);
    OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(common->has_skip_arg ? 4 : 0), TMP2, 0);
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, (sljit_sw)(cc + 2));
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->mark_ptr, TMP2, 0);
    OP1(SLJIT_MOV, SLJIT_MEM1(HAS_VIRTUAL_REGISTERS ? TMP1 : ARGUMENTS), SLJIT_OFFSETOF(jit_arguments, mark_ptr), TMP2, 0);
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
    case OP_COMMIT_ARG:
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
    SLJIT_UNREACHABLE();
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

static void compile_newline_move_back(compiler_common *common)
{
DEFINE_COMPILER;
struct sljit_jump *jump;

OP2(SLJIT_SUB, TMP1, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
jump = CMP(SLJIT_LESS_EQUAL, TMP1, 0, TMP2, 0);
/* All newlines are single byte, or their last byte
is not equal to CHAR_NL/CHAR_CR even if UTF is enabled. */
OP1(MOV_UCHAR, SLJIT_TMP_DEST_REG, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-2));
OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-1));
OP2(SLJIT_SHL, SLJIT_TMP_DEST_REG, 0, SLJIT_TMP_DEST_REG, 0, SLJIT_IMM, 8);
OP2(SLJIT_OR, TMP1, 0, TMP1, 0, SLJIT_TMP_DEST_REG, 0);
OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, CHAR_CR << 8 | CHAR_NL);
OP_FLAGS(SLJIT_MOV, TMP1, 0, SLJIT_EQUAL);
#if PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32
OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, UCHAR_SHIFT);
#endif
OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, TMP1, 0);
JUMPHERE(jump);
}

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
int offset1 = (private_data_ptr == 0) ? STACK(1) : private_data_ptr + SSIZE_OF(sw);

cc = get_iterator_parameters(common, cc, &opcode, &type, &max, &exact, &end);

switch(opcode)
  {
  case OP_STAR:
  case OP_UPTO:
  if (type == OP_EXTUNI)
    {
    SLJIT_ASSERT(private_data_ptr == 0);
    set_jumps(current->own_backtracks, LABEL());
    OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
    free_stack(common, 1);
    CMPTO(SLJIT_NOT_EQUAL, STR_PTR, 0, SLJIT_IMM, 0, CURRENT_AS(char_iterator_backtrack)->matchingpath);
    }
  else
    {
    if (CURRENT_AS(char_iterator_backtrack)->charpos.charpos_enabled)
      {
      OP1(SLJIT_MOV, STR_PTR, 0, base, offset0);
      OP1(SLJIT_MOV, TMP2, 0, base, offset1);

      jump = CMP(SLJIT_LESS_EQUAL, STR_PTR, 0, TMP2, 0);
      label = LABEL();
      if (type == OP_ANYNL)
        compile_newline_move_back(common);
      move_back(common, NULL, TRUE);

      OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(0));
      if (CURRENT_AS(char_iterator_backtrack)->charpos.othercasebit != 0)
        OP2(SLJIT_OR, TMP1, 0, TMP1, 0, SLJIT_IMM, CURRENT_AS(char_iterator_backtrack)->charpos.othercasebit);
      CMPTO(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, CURRENT_AS(char_iterator_backtrack)->charpos.chr, CURRENT_AS(char_iterator_backtrack)->matchingpath);
      /* The range beginning must match, no need to compare. */
      JUMPTO(SLJIT_JUMP, label);

      set_jumps(current->own_backtracks, LABEL());
      current->own_backtracks = NULL;
      }
    else
      {
      OP1(SLJIT_MOV, STR_PTR, 0, base, offset0);

      if (opcode == OP_STAR && exact == 1)
        {
        if (type == OP_ANYNL)
          {
          OP1(SLJIT_MOV, TMP2, 0, base, offset1);
          compile_newline_move_back(common);
          }

        move_back(common, NULL, TRUE);
        jump = CMP(SLJIT_LESS_EQUAL, STR_PTR, 0, base, offset1);
        }
      else
        {
        if (type == OP_ANYNL)
          {
          OP1(SLJIT_MOV, TMP2, 0, base, offset1);
          jump = CMP(SLJIT_LESS_EQUAL, STR_PTR, 0, TMP2, 0);
          compile_newline_move_back(common);
          }
        else
          jump = CMP(SLJIT_LESS_EQUAL, STR_PTR, 0, base, offset1);

        move_back(common, NULL, TRUE);
        }

      OP1(SLJIT_MOV, base, offset0, STR_PTR, 0);
      JUMPTO(SLJIT_JUMP, CURRENT_AS(char_iterator_backtrack)->matchingpath);

      set_jumps(current->own_backtracks, LABEL());
      }

    JUMPHERE(jump);
    if (private_data_ptr == 0)
      free_stack(common, 2);
    }
  break;

  case OP_QUERY:
  OP1(SLJIT_MOV, STR_PTR, 0, base, offset0);
  OP1(SLJIT_MOV, base, offset0, SLJIT_IMM, 0);
  CMPTO(SLJIT_NOT_EQUAL, STR_PTR, 0, SLJIT_IMM, 0, CURRENT_AS(char_iterator_backtrack)->matchingpath);
  jump = JUMP(SLJIT_JUMP);
  set_jumps(current->own_backtracks, LABEL());
  OP1(SLJIT_MOV, STR_PTR, 0, base, offset0);
  OP1(SLJIT_MOV, base, offset0, SLJIT_IMM, 0);
  JUMPTO(SLJIT_JUMP, CURRENT_AS(char_iterator_backtrack)->matchingpath);
  JUMPHERE(jump);
  if (private_data_ptr == 0)
    free_stack(common, 1);
  break;

  case OP_MINSTAR:
  OP1(SLJIT_MOV, STR_PTR, 0, base, offset0);
  if (exact == 0)
    {
    compile_char1_matchingpath(common, type, cc, &jumplist, TRUE);
    OP1(SLJIT_MOV, base, offset0, STR_PTR, 0);
    }
  else if (exact > 1)
    OP1(SLJIT_MOV, TMP3, 0, SLJIT_IMM, 1);

  JUMPTO(SLJIT_JUMP, CURRENT_AS(char_iterator_backtrack)->matchingpath);
  set_jumps(exact > 0 ? current->own_backtracks : jumplist, LABEL());
  if (private_data_ptr == 0)
    free_stack(common, 1);
  break;

  case OP_MINUPTO:
  OP1(SLJIT_MOV, TMP1, 0, base, offset1);
  OP1(SLJIT_MOV, STR_PTR, 0, base, offset0);
  OP2(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, TMP1, 0, SLJIT_IMM, 1);

  if (exact == 0)
    {
    add_jump(compiler, &jumplist, JUMP(SLJIT_ZERO));

    OP1(SLJIT_MOV, base, offset1, TMP1, 0);
    compile_char1_matchingpath(common, type, cc, &jumplist, TRUE);
    OP1(SLJIT_MOV, base, offset0, STR_PTR, 0);
    JUMPTO(SLJIT_JUMP, CURRENT_AS(char_iterator_backtrack)->matchingpath);

    set_jumps(jumplist, LABEL());
    }
  else
    {
    if (exact > 1)
      OP1(SLJIT_MOV, TMP3, 0, SLJIT_IMM, 1);
    OP1(SLJIT_MOV, base, offset1, TMP1, 0);
    JUMPTO(SLJIT_NOT_ZERO, CURRENT_AS(char_iterator_backtrack)->matchingpath);

    set_jumps(current->own_backtracks, LABEL());
    }

  if (private_data_ptr == 0)
    free_stack(common, 2);
  break;

  case OP_MINQUERY:
  OP1(SLJIT_MOV, STR_PTR, 0, base, offset0);
  OP1(SLJIT_MOV, base, offset0, SLJIT_IMM, 0);

  if (exact >= 1)
    {
    if (exact >= 2)
      OP1(SLJIT_MOV, TMP3, 0, SLJIT_IMM, 1);
    CMPTO(SLJIT_NOT_EQUAL, STR_PTR, 0, SLJIT_IMM, 0, CURRENT_AS(char_iterator_backtrack)->matchingpath);
    set_jumps(current->own_backtracks, LABEL());
    }
  else
    {
    jump = CMP(SLJIT_EQUAL, STR_PTR, 0, SLJIT_IMM, 0);
    compile_char1_matchingpath(common, type, cc, &jumplist, TRUE);
    JUMPTO(SLJIT_JUMP, CURRENT_AS(char_iterator_backtrack)->matchingpath);
    set_jumps(jumplist, LABEL());
    JUMPHERE(jump);
    }

  if (private_data_ptr == 0)
    free_stack(common, 1);
  break;

  default:
  SLJIT_UNREACHABLE();
  break;
  }
}

static SLJIT_INLINE void compile_ref_iterator_backtrackingpath(compiler_common *common, struct backtrack_common *current)
{
DEFINE_COMPILER;
PCRE2_SPTR cc = current->cc;
BOOL ref = (*cc == OP_REF || *cc == OP_REFI);
PCRE2_UCHAR type;

type = cc[PRIV(OP_lengths)[*cc]];

if ((type & 0x1) == 0)
  {
  /* Maximize case. */
  set_jumps(current->own_backtracks, LABEL());
  OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
  free_stack(common, 1);
  CMPTO(SLJIT_NOT_EQUAL, STR_PTR, 0, SLJIT_IMM, 0, CURRENT_AS(ref_iterator_backtrack)->matchingpath);
  return;
  }

OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
CMPTO(SLJIT_NOT_EQUAL, STR_PTR, 0, SLJIT_IMM, 0, CURRENT_AS(ref_iterator_backtrack)->matchingpath);
set_jumps(current->own_backtracks, LABEL());
free_stack(common, ref ? 2 : 3);
}

static SLJIT_INLINE void compile_recurse_backtrackingpath(compiler_common *common, struct backtrack_common *current)
{
DEFINE_COMPILER;
recurse_entry *entry;

if (!CURRENT_AS(recurse_backtrack)->inlined_pattern)
  {
  entry = CURRENT_AS(recurse_backtrack)->entry;
  if (entry->backtrack_label == NULL)
    add_jump(compiler, &entry->backtrack_calls, JUMP(SLJIT_FAST_CALL));
  else
    JUMPTO(SLJIT_FAST_CALL, entry->backtrack_label);
  CMPTO(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, 0, CURRENT_AS(recurse_backtrack)->matchingpath);
  }
else
  compile_backtrackingpath(common, current->top);

set_jumps(current->own_backtracks, LABEL());
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
  SLJIT_ASSERT(current->own_backtracks == NULL);
  OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
  }

if (CURRENT_AS(assert_backtrack)->framesize < 0)
  {
  set_jumps(current->own_backtracks, LABEL());

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
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(-2));
  OP2(SLJIT_ADD, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, (CURRENT_AS(assert_backtrack)->framesize - 1) * sizeof(sljit_sw));
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), CURRENT_AS(assert_backtrack)->private_data_ptr, TMP1, 0);

  set_jumps(current->own_backtracks, LABEL());
  }
else
  set_jumps(current->own_backtracks, LABEL());

if (bra == OP_BRAZERO)
  {
  /* We know there is enough place on the stack. */
  OP2(SLJIT_SUB, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, sizeof(sljit_sw));
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
const assert_backtrack *assert;
BOOL has_alternatives;
BOOL needs_control_head = FALSE;
BOOL has_vreverse;
struct sljit_jump *brazero = NULL;
struct sljit_jump *next_alt = NULL;
struct sljit_jump *once = NULL;
struct sljit_jump *cond = NULL;
struct sljit_label *rmin_label = NULL;
struct sljit_label *exact_label = NULL;
struct sljit_jump *mov_addr = NULL;

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
  has_alternatives = (ccbegin[1 + LINK_SIZE] >= OP_ASSERT && ccbegin[1 + LINK_SIZE] <= OP_ASSERTBACK_NOT) || CURRENT_AS(bracket_backtrack)->u.no_capture != NULL;
if (opcode == OP_CBRA || opcode == OP_SCBRA)
  offset = (GET2(ccbegin, 1 + LINK_SIZE)) << 1;
if (SLJIT_UNLIKELY(opcode == OP_COND) && (*cc == OP_KETRMAX || *cc == OP_KETRMIN))
  opcode = OP_SCOND;

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
        CMPTO(SLJIT_NOT_EQUAL, STR_PTR, 0, SLJIT_MEM1(TMP1), STACK(-CURRENT_AS(bracket_backtrack)->u.framesize - 2), CURRENT_AS(bracket_backtrack)->recursive_matchingpath);
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
else if (SLJIT_UNLIKELY(opcode == OP_ASSERT_SCS))
  {
  OP1(SLJIT_MOV, TMP1, 0, STR_END, 0);
  OP1(SLJIT_MOV, STR_END, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr + sizeof(sljit_sw));
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr + sizeof(sljit_sw), TMP1, 0);

  /* Nested scs blocks will not update this variable. */
  if (common->restore_end_ptr == 0)
    common->restore_end_ptr = private_data_ptr + sizeof(sljit_sw);
  }

if (SLJIT_UNLIKELY(opcode == OP_ONCE))
  {
  int framesize = CURRENT_AS(bracket_backtrack)->u.framesize;

  SLJIT_ASSERT(framesize != 0);
  if (framesize > 0)
    {
    OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
    add_jump(compiler, &common->revertframes, JUMP(SLJIT_FAST_CALL));
    OP2(SLJIT_ADD, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, (framesize - 1) * sizeof(sljit_sw));
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
    next_alt = CMP(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, 0);
    }
  }
else if (has_alternatives)
  {
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
  free_stack(common, 1);

  if (alt_max > 3)
    {
    sljit_emit_ijump(compiler, SLJIT_JUMP, TMP1, 0);

    SLJIT_ASSERT(CURRENT_AS(bracket_backtrack)->matching_mov_addr != NULL);
    sljit_set_label(CURRENT_AS(bracket_backtrack)->matching_mov_addr, LABEL());
    sljit_emit_op0(compiler, SLJIT_ENDBR);
    }
  else
    next_alt = CMP(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, 0);
  }

COMPILE_BACKTRACKINGPATH(current->top);
if (current->own_backtracks)
  set_jumps(current->own_backtracks, LABEL());

if (SLJIT_UNLIKELY(opcode == OP_COND) || SLJIT_UNLIKELY(opcode == OP_SCOND))
  {
  /* Conditional block always has at most one alternative. */
  if (ccbegin[1 + LINK_SIZE] >= OP_ASSERT && ccbegin[1 + LINK_SIZE] <= OP_ASSERTBACK_NOT)
    {
    SLJIT_ASSERT(has_alternatives);
    assert = CURRENT_AS(bracket_backtrack)->u.assert;
    SLJIT_ASSERT(assert->framesize != 0);
    if (assert->framesize > 0 && (ccbegin[1 + LINK_SIZE] == OP_ASSERT || ccbegin[1 + LINK_SIZE] == OP_ASSERTBACK))
      {
      OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), assert->private_data_ptr);
      add_jump(compiler, &common->revertframes, JUMP(SLJIT_FAST_CALL));
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(-2));
      OP2(SLJIT_ADD, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, (assert->framesize - 1) * sizeof(sljit_sw));
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), assert->private_data_ptr, TMP1, 0);
      }
    cond = JUMP(SLJIT_JUMP);
    set_jumps(CURRENT_AS(bracket_backtrack)->u.assert->condfailed, LABEL());
    }
  else if (CURRENT_AS(bracket_backtrack)->u.no_capture != NULL)
    {
    SLJIT_ASSERT(has_alternatives);
    cond = JUMP(SLJIT_JUMP);
    set_jumps(CURRENT_AS(bracket_backtrack)->u.no_capture, LABEL());
    }
  else
    SLJIT_ASSERT(!has_alternatives);
  }

if (has_alternatives)
  {
  alt_count = 1;
  do
    {
    current->top = NULL;
    current->own_backtracks = NULL;
    current->simple_backtracks = NULL;
    /* Conditional blocks always have an additional alternative, even if it is empty. */
    if (*cc == OP_ALT)
      {
      ccprev = cc + 1 + LINK_SIZE;
      cc += GET(cc, 1);

      has_vreverse = FALSE;

      switch (opcode)
        {
        case OP_ASSERTBACK:
        case OP_ASSERTBACK_NA:
          SLJIT_ASSERT(private_data_ptr != 0);
          OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);

          has_vreverse = (*ccprev == OP_VREVERSE);
          if (*ccprev == OP_REVERSE || has_vreverse)
            ccprev = compile_reverse_matchingpath(common, ccprev, current);
          break;
        case OP_ASSERT_SCS:
          OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(2));
          break;
        case OP_ONCE:
          OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(needs_control_head ? 1 : 0));
          break;
        case OP_COND:
        case OP_SCOND:
          break;
        default:
          if (private_data_ptr != 0)
            OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
          else
            OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
          break;
        }

      compile_matchingpath(common, ccprev, cc, current);
      if (SLJIT_UNLIKELY(sljit_get_compiler_error(compiler)))
        return;

      switch (opcode)
        {
        case OP_ASSERTBACK_NA:
          if (has_vreverse)
            {
            SLJIT_ASSERT(current->top != NULL && PRIVATE_DATA(ccbegin + 1));
            add_jump(compiler, &current->top->simple_backtracks, CMP(SLJIT_LESS, STR_PTR, 0, STR_END, 0));
            }

          if (PRIVATE_DATA(ccbegin + 1))
            OP1(SLJIT_MOV, STR_END, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr + sizeof(sljit_sw));
          break;
        case OP_ASSERT_NA:
          OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr);
          break;
        case OP_SCRIPT_RUN:
          match_script_run_common(common, private_data_ptr, current);
          break;
        }
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
      {
      if (alt_max <= 3)
        OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(stacksize), SLJIT_IMM, alt_count);
      else
        mov_addr = sljit_emit_mov_addr(compiler, SLJIT_MEM1(STACK_TOP), STACK(stacksize));
      }

    if (offset != 0 && ket == OP_KETRMAX && common->optimized_cbracket[offset >> 1] != 0)
      {
      /* If ket is not OP_KETRMAX, this code path is executed after the jump to alternative_matchingpath. */
      SLJIT_ASSERT(private_data_ptr == OVECTOR(offset + 0));
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1), STR_PTR, 0);
      }

    JUMPTO(SLJIT_JUMP, CURRENT_AS(bracket_backtrack)->alternative_matchingpath);

    if (opcode != OP_ONCE)
      {
      if (alt_max <= 3)
        {
        JUMPHERE(next_alt);
        alt_count++;
        if (alt_count < alt_max)
          {
          SLJIT_ASSERT(alt_count == 2 && alt_max == 3);
          next_alt = CMP(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, 1);
          }
        }
      else
        {
        sljit_set_label(mov_addr, LABEL());
        sljit_emit_op0(compiler, SLJIT_ENDBR);
        }
      }

    COMPILE_BACKTRACKINGPATH(current->top);
    if (current->own_backtracks)
      set_jumps(current->own_backtracks, LABEL());
    SLJIT_ASSERT(!current->simple_backtracks);
    }
  while (*cc == OP_ALT);

  if (cond != NULL)
    {
    SLJIT_ASSERT(opcode == OP_COND || opcode == OP_SCOND);
    if (ccbegin[1 + LINK_SIZE] == OP_ASSERT_NOT || ccbegin[1 + LINK_SIZE] == OP_ASSERTBACK_NOT)
      {
      assert = CURRENT_AS(bracket_backtrack)->u.assert;
      SLJIT_ASSERT(assert->framesize != 0);
      if (assert->framesize > 0)
        {
        OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), assert->private_data_ptr);
        add_jump(compiler, &common->revertframes, JUMP(SLJIT_FAST_CALL));
        OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(-2));
        OP2(SLJIT_ADD, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, (assert->framesize - 1) * sizeof(sljit_sw));
        OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), assert->private_data_ptr, TMP1, 0);
        }
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
else if (opcode == OP_ASSERTBACK_NA && PRIVATE_DATA(ccbegin + 1))
  {
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(STACK_TOP), STACK(1));
  OP1(SLJIT_MOV, STR_END, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr + sizeof(sljit_sw));
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, TMP1, 0);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr + sizeof(sljit_sw), TMP2, 0);
  free_stack(common, 4);
  }
else if (opcode == OP_ASSERT_NA || opcode == OP_ASSERTBACK_NA || opcode == OP_SCRIPT_RUN || opcode == OP_SBRA || opcode == OP_SCOND)
  {
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, SLJIT_MEM1(STACK_TOP), STACK(0));
  free_stack(common, 1);
  }
else if (opcode == OP_ASSERT_SCS)
  {
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(STACK_TOP), STACK(1));
  OP1(SLJIT_MOV, STR_END, 0, SLJIT_MEM1(SLJIT_SP), private_data_ptr + sizeof(sljit_sw));
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, TMP1, 0);
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr + sizeof(sljit_sw), TMP2, 0);
  free_stack(common, has_alternatives ? 3 : 2);

  set_jumps(CURRENT_AS(bracket_backtrack)->u.no_capture, LABEL());

  /* Nested scs blocks will not update this variable. */
  if (common->restore_end_ptr == private_data_ptr + SSIZE_OF(sw))
    common->restore_end_ptr = 0;
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
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), private_data_ptr, SLJIT_MEM1(STACK_TOP), STACK(-CURRENT_AS(bracket_backtrack)->u.framesize - 1));
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
PCRE2_SPTR cc;

/* No retry on backtrack, just drop everything. */
if (CURRENT_AS(bracketpos_backtrack)->framesize < 0)
  {
  cc = current->cc;

  if (*cc == OP_BRAPOSZERO)
    cc++;

  if (*cc == OP_CBRAPOS || *cc == OP_SCBRAPOS)
    {
    offset = (GET2(cc, 1 + LINK_SIZE)) << 1;
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(STACK_TOP), STACK(1));
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset), TMP1, 0);
    if (common->capture_last_ptr != 0)
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(2));
    OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(offset + 1), TMP2, 0);
    if (common->capture_last_ptr != 0)
      OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->capture_last_ptr, TMP1, 0);
    }
  set_jumps(current->own_backtracks, LABEL());
  free_stack(common, CURRENT_AS(bracketpos_backtrack)->stacksize);
  return;
  }

OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), CURRENT_AS(bracketpos_backtrack)->private_data_ptr);
add_jump(compiler, &common->revertframes, JUMP(SLJIT_FAST_CALL));
OP2(SLJIT_ADD, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, (CURRENT_AS(bracketpos_backtrack)->framesize - 1) * sizeof(sljit_sw));

if (current->own_backtracks)
  {
  jump = JUMP(SLJIT_JUMP);
  set_jumps(current->own_backtracks, LABEL());
  /* Drop the stack frame. */
  free_stack(common, CURRENT_AS(bracketpos_backtrack)->stacksize);
  JUMPHERE(jump);
  }
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), CURRENT_AS(bracketpos_backtrack)->private_data_ptr, SLJIT_MEM1(STACK_TOP), STACK(-CURRENT_AS(bracketpos_backtrack)->framesize - 1));
}

static SLJIT_INLINE void compile_braminzero_backtrackingpath(compiler_common *common, struct backtrack_common *current)
{
assert_backtrack backtrack;

current->top = NULL;
current->own_backtracks = NULL;
current->simple_backtracks = NULL;
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
SLJIT_ASSERT(!current->simple_backtracks && !current->own_backtracks);
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
    OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
    JUMPHERE(jump);
    CMPTO(SLJIT_NOT_EQUAL, SLJIT_MEM1(STACK_TOP), STACK(1), TMP1, 0, loop);
    CMPTO(SLJIT_NOT_EQUAL, SLJIT_MEM1(STACK_TOP), STACK(2), TMP2, 0, loop);
    add_jump(compiler, &common->then_trap->quit, JUMP(SLJIT_JUMP));
    return;
    }
  else if (!common->local_quit_available && common->in_positive_assertion)
    {
    add_jump(compiler, &common->positive_assertion_quit, JUMP(SLJIT_JUMP));
    return;
    }
  }

if (common->restore_end_ptr != 0 && opcode != OP_SKIP_ARG)
  OP1(SLJIT_MOV, STR_END, 0, SLJIT_MEM1(SLJIT_SP), common->restore_end_ptr);

if (common->local_quit_available)
  {
  /* Abort match with a fail. */
  if (common->quit_label == NULL)
    add_jump(compiler, &common->quit, JUMP(SLJIT_JUMP));
  else
    JUMPTO(SLJIT_JUMP, common->quit_label);
  return;
  }

if (opcode == OP_SKIP_ARG)
  {
  SLJIT_ASSERT(common->control_head_ptr != 0 && TMP1 == SLJIT_R0 && STR_PTR == SLJIT_R1);
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr);
  OP1(SLJIT_MOV, SLJIT_R1, 0, SLJIT_IMM, (sljit_sw)(current->cc + 2));
  sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_ARGS2(W, W, W), SLJIT_IMM, SLJIT_FUNC_ADDR(do_search_mark));

  if (common->restore_end_ptr == 0)
    {
    OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_R0, 0);
    add_jump(compiler, &common->reset_match, CMP(SLJIT_NOT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, 0));
    return;
    }

  jump = CMP(SLJIT_EQUAL, SLJIT_R0, 0, SLJIT_IMM, 0);
  OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_R0, 0);
  OP1(SLJIT_MOV, STR_END, 0, SLJIT_MEM1(SLJIT_SP), common->restore_end_ptr);
  add_jump(compiler, &common->reset_match, JUMP(SLJIT_JUMP));
  JUMPHERE(jump);
  return;
  }

if (opcode == OP_SKIP)
  OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
else
  OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_IMM, 0);
add_jump(compiler, &common->reset_match, JUMP(SLJIT_JUMP));
}

static SLJIT_INLINE void compile_vreverse_backtrackingpath(compiler_common *common, struct backtrack_common *current)
{
DEFINE_COMPILER;
struct sljit_jump *jump;
struct sljit_label *label;

OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(2));
jump = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(3));
skip_valid_char(common);
OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(2), STR_PTR, 0);
JUMPTO(SLJIT_JUMP, CURRENT_AS(vreverse_backtrack)->matchingpath);

label = LABEL();
sljit_set_label(jump, label);
set_jumps(current->own_backtracks, label);
}

static SLJIT_INLINE void compile_then_trap_backtrackingpath(compiler_common *common, struct backtrack_common *current)
{
DEFINE_COMPILER;
struct sljit_jump *jump;
int framesize;
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

framesize = CURRENT_AS(then_trap_backtrack)->framesize;
SLJIT_ASSERT(framesize != 0);

/* STACK_TOP is set by THEN. */
if (framesize > 0)
  {
  add_jump(compiler, &common->revertframes, JUMP(SLJIT_FAST_CALL));
  OP2(SLJIT_ADD, STACK_TOP, 0, STACK_TOP, 0, SLJIT_IMM, (framesize - 1) * sizeof(sljit_sw));
  }
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
  if (current->simple_backtracks != NULL)
    set_jumps(current->simple_backtracks, LABEL());
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
    /* Since classes has no backtracking path, this
    backtrackingpath was pushed by an iterator. */
    case OP_CLASS:
    case OP_NCLASS:
#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH != 8
    case OP_XCLASS:
    case OP_ECLASS:
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

    case OP_ASSERT_NA:
    case OP_ASSERTBACK_NA:
    case OP_ASSERT_SCS:
    case OP_ONCE:
    case OP_SCRIPT_RUN:
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
    case OP_COMMIT_ARG:
    if (common->restore_end_ptr != 0)
      OP1(SLJIT_MOV, STR_END, 0, SLJIT_MEM1(SLJIT_SP), common->restore_end_ptr);

    if (!common->local_quit_available)
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
    set_jumps(current->own_backtracks, LABEL());
    break;

    case OP_VREVERSE:
    compile_vreverse_backtrackingpath(common, current);
    break;

    case OP_THEN_TRAP:
    /* A virtual opcode for then traps. */
    compile_then_trap_backtrackingpath(common, current);
    break;

    default:
    SLJIT_UNREACHABLE();
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
uint32_t recurse_flags = 0;
int private_data_size = get_recurse_data_length(common, ccbegin, ccend, &recurse_flags);
int alt_count, alt_max, local_size;
backtrack_common altbacktrack;
jump_list *match = NULL;
struct sljit_jump *next_alt = NULL;
struct sljit_jump *accept_exit = NULL;
struct sljit_label *quit;
struct sljit_jump *mov_addr = NULL;

/* Recurse captures then. */
common->then_trap = NULL;

SLJIT_ASSERT(*cc == OP_BRA || *cc == OP_CBRA || *cc == OP_CBRAPOS || *cc == OP_SCBRA || *cc == OP_SCBRAPOS);

alt_max = no_alternatives(cc);
alt_count = 0;

/* Matching path. */
SLJIT_ASSERT(common->currententry->entry_label == NULL && common->recursive_head_ptr != 0);
common->currententry->entry_label = LABEL();
set_jumps(common->currententry->entry_calls, common->currententry->entry_label);

sljit_emit_op_dst(compiler, SLJIT_FAST_ENTER, TMP2, 0);
count_match(common);

local_size = (alt_max > 1) ? 2 : 1;

/* (Reversed) stack layout:
   [private data][return address][optional: str ptr] ... [optional: alternative index][recursive_head_ptr] */

allocate_stack(common, private_data_size + local_size);
/* Save return address. */
OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(local_size - 1), TMP2, 0);

copy_recurse_data(common, ccbegin, ccend, recurse_copy_from_global, local_size, private_data_size + local_size, recurse_flags);

/* This variable is saved and restored all time when we enter or exit from a recursive context. */
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->recursive_head_ptr, STACK_TOP, 0);

if (recurse_flags & recurse_flag_control_head_found)
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, SLJIT_IMM, 0);

if (alt_max > 1)
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
  altbacktrack.own_backtracks = NULL;

  if (altbacktrack.cc != ccbegin)
    OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(STACK_TOP), STACK(0));

  compile_matchingpath(common, altbacktrack.cc, cc, &altbacktrack);
  if (SLJIT_UNLIKELY(sljit_get_compiler_error(compiler)))
    return;

  allocate_stack(common, (alt_max > 1 || (recurse_flags & recurse_flag_accept_found)) ? 2 : 1);
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), common->recursive_head_ptr);

  if (alt_max > 1 || (recurse_flags & recurse_flag_accept_found))
    {
    if (alt_max > 3)
      mov_addr = sljit_emit_mov_addr(compiler, SLJIT_MEM1(STACK_TOP), STACK(1));
    else
      OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(1), SLJIT_IMM, alt_count);
    }

  add_jump(compiler, &match, JUMP(SLJIT_JUMP));

  if (alt_count == 0)
    {
    /* Backtracking path entry. */
    SLJIT_ASSERT(common->currententry->backtrack_label == NULL);
    common->currententry->backtrack_label = LABEL();
    set_jumps(common->currententry->backtrack_calls, common->currententry->backtrack_label);

    sljit_emit_op_dst(compiler, SLJIT_FAST_ENTER, TMP1, 0);

    if (recurse_flags & recurse_flag_accept_found)
      accept_exit = CMP(SLJIT_EQUAL, SLJIT_MEM1(STACK_TOP), STACK(1), SLJIT_IMM, -1);

    OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(STACK_TOP), STACK(0));
    /* Save return address. */
    OP1(SLJIT_MOV, SLJIT_MEM1(TMP2), STACK(local_size - 1), TMP1, 0);

    copy_recurse_data(common, ccbegin, ccend, recurse_swap_global, local_size, private_data_size + local_size, recurse_flags);

    if (alt_max > 1)
      {
      OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(STACK_TOP), STACK(1));
      free_stack(common, 2);

      if (alt_max > 3)
        {
        sljit_emit_ijump(compiler, SLJIT_JUMP, TMP1, 0);
        sljit_set_label(mov_addr, LABEL());
        sljit_emit_op0(compiler, SLJIT_ENDBR);
        }
      else
        next_alt = CMP(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, 0);
      }
    else
      free_stack(common, (recurse_flags & recurse_flag_accept_found) ? 2 : 1);
    }
  else if (alt_max > 3)
    {
    sljit_set_label(mov_addr, LABEL());
    sljit_emit_op0(compiler, SLJIT_ENDBR);
    }
  else
    {
    JUMPHERE(next_alt);
    if (alt_count + 1 < alt_max)
      {
      SLJIT_ASSERT(alt_count == 1 && alt_max == 3);
      next_alt = CMP(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, 1);
      }
    }

  alt_count++;

  compile_backtrackingpath(common, altbacktrack.top);
  if (SLJIT_UNLIKELY(sljit_get_compiler_error(compiler)))
    return;
  set_jumps(altbacktrack.own_backtracks, LABEL());

  if (*cc != OP_ALT)
    break;

  altbacktrack.cc = cc + 1 + LINK_SIZE;
  cc += GET(cc, 1);
  }

/* No alternative is matched. */

quit = LABEL();

copy_recurse_data(common, ccbegin, ccend, recurse_copy_private_to_global, local_size, private_data_size + local_size, recurse_flags);

OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(STACK_TOP), STACK(local_size - 1));
free_stack(common, private_data_size + local_size);
OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, 0);
OP_SRC(SLJIT_FAST_RETURN, TMP2, 0);

if (common->quit != NULL)
  {
  SLJIT_ASSERT(recurse_flags & recurse_flag_quit_found);

  set_jumps(common->quit, LABEL());
  OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), common->recursive_head_ptr);
  copy_recurse_data(common, ccbegin, ccend, recurse_copy_shared_to_global, local_size, private_data_size + local_size, recurse_flags);
  JUMPTO(SLJIT_JUMP, quit);
  }

if (recurse_flags & recurse_flag_accept_found)
  {
  JUMPHERE(accept_exit);
  free_stack(common, 2);

  /* Save return address. */
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(local_size - 1), TMP1, 0);

  copy_recurse_data(common, ccbegin, ccend, recurse_copy_kept_shared_to_global, local_size, private_data_size + local_size, recurse_flags);

  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(STACK_TOP), STACK(local_size - 1));
  free_stack(common, private_data_size + local_size);
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, 0);
  OP_SRC(SLJIT_FAST_RETURN, TMP2, 0);
  }

if (common->accept != NULL)
  {
  SLJIT_ASSERT(recurse_flags & recurse_flag_accept_found);

  set_jumps(common->accept, LABEL());

  OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(SLJIT_SP), common->recursive_head_ptr);
  OP1(SLJIT_MOV, TMP2, 0, STACK_TOP, 0);

  allocate_stack(common, 2);
  OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(1), SLJIT_IMM, -1);
  }

set_jumps(match, LABEL());

OP1(SLJIT_MOV, SLJIT_MEM1(STACK_TOP), STACK(0), TMP2, 0);

copy_recurse_data(common, ccbegin, ccend, recurse_swap_global, local_size, private_data_size + local_size, recurse_flags);

OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(TMP2), STACK(local_size - 1));
OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, 1);
OP_SRC(SLJIT_FAST_RETURN, TMP2, 0);
}

#undef COMPILE_BACKTRACKINGPATH
#undef CURRENT_AS

#define PUBLIC_JIT_COMPILE_CONFIGURATION_OPTIONS \
  (PCRE2_JIT_INVALID_UTF)

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
sljit_uw executable_size, private_data_length, total_length;
struct sljit_label *mainloop_label = NULL;
struct sljit_label *continue_match_label;
struct sljit_label *empty_match_found_label = NULL;
struct sljit_label *empty_match_backtrack_label = NULL;
struct sljit_label *reset_match_label;
struct sljit_label *quit_label;
struct sljit_jump *jump;
struct sljit_jump *minlength_check_failed = NULL;
struct sljit_jump *empty_match = NULL;
struct sljit_jump *end_anchor_failed = NULL;
jump_list *reqcu_not_found = NULL;

SLJIT_ASSERT(tables);

#if HAS_VIRTUAL_REGISTERS == 1
SLJIT_ASSERT(sljit_get_register_index(SLJIT_GP_REGISTER, TMP3) < 0 && sljit_get_register_index(SLJIT_GP_REGISTER, ARGUMENTS) < 0 && sljit_get_register_index(SLJIT_GP_REGISTER, RETURN_ADDR) < 0);
#elif HAS_VIRTUAL_REGISTERS == 0
SLJIT_ASSERT(sljit_get_register_index(SLJIT_GP_REGISTER, TMP3) >= 0 && sljit_get_register_index(SLJIT_GP_REGISTER, ARGUMENTS) >= 0 && sljit_get_register_index(SLJIT_GP_REGISTER, RETURN_ADDR) >= 0);
#else
#error "Invalid value for HAS_VIRTUAL_REGISTERS"
#endif

memset(&rootbacktrack, 0, sizeof(backtrack_common));
memset(common, 0, sizeof(compiler_common));
common->re = re;
common->name_table = (PCRE2_SPTR)((uint8_t *)re + sizeof(pcre2_real_code));
rootbacktrack.cc = (PCRE2_SPTR)((uint8_t *)re + re->code_start);

#ifdef SUPPORT_UNICODE
common->invalid_utf = (mode & PCRE2_JIT_INVALID_UTF) != 0;
#endif /* SUPPORT_UNICODE */
mode &= ~PUBLIC_JIT_COMPILE_CONFIGURATION_OPTIONS;

common->start = rootbacktrack.cc;
common->read_only_data_head = NULL;
common->fcc = tables + fcc_offset;
common->lcc = (sljit_sw)(tables + lcc_offset);
common->mode = mode;
common->might_be_empty = (re->minlength == 0) || (re->flags & PCRE2_MATCH_EMPTY);
common->allow_empty_partial = (re->max_lookbehind > 0) || (re->flags & PCRE2_MATCH_EMPTY);
common->nltype = NLTYPE_FIXED;
switch(re->newline_convention)
  {
  case PCRE2_NEWLINE_CR: common->newline = CHAR_CR; break;
  case PCRE2_NEWLINE_LF: common->newline = CHAR_NL; break;
  case PCRE2_NEWLINE_CRLF: common->newline = (CHAR_CR << 8) | CHAR_NL; break;
  case PCRE2_NEWLINE_ANY: common->newline = (CHAR_CR << 8) | CHAR_NL; common->nltype = NLTYPE_ANY; break;
  case PCRE2_NEWLINE_ANYCRLF: common->newline = (CHAR_CR << 8) | CHAR_NL; common->nltype = NLTYPE_ANYCRLF; break;
  case PCRE2_NEWLINE_NUL: common->newline = CHAR_NUL; break;
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
/* PCRE2_UTF[16|32] have the same value as PCRE2_UTF8. */
common->utf = (re->overall_options & PCRE2_UTF) != 0;
common->ucp = (re->overall_options & PCRE2_UCP) != 0;
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
else
  common->invalid_utf = FALSE;
#endif /* SUPPORT_UNICODE */
ccend = bracketend(common->start);

/* Calculate the local space size on the stack. */
common->ovector_start = LOCAL0;
/* Allocate space for temporary data structures. */
private_data_length = ccend - common->start;
/* The chance of overflow is very low, but might happen on 32 bit. */
if (private_data_length > ~(sljit_uw)0 / sizeof(sljit_s32))
  return PCRE2_ERROR_NOMEMORY;

private_data_length *= sizeof(sljit_s32);
/* Align to 32 bit. */
total_length = ((re->top_bracket + 1) + (sljit_uw)(sizeof(sljit_s32) - 1)) & ~(sljit_uw)(sizeof(sljit_s32) - 1);
if (~(sljit_uw)0 - private_data_length < total_length)
  return PCRE2_ERROR_NOMEMORY;

total_length += private_data_length;
common->private_data_ptrs = (sljit_s32*)SLJIT_MALLOC(total_length, allocator_data);
if (!common->private_data_ptrs)
  return PCRE2_ERROR_NOMEMORY;

memset(common->private_data_ptrs, 0, private_data_length);
common->optimized_cbracket = ((sljit_u8 *)common->private_data_ptrs) + private_data_length;
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
  SLJIT_FREE(common->private_data_ptrs, allocator_data);
  return PCRE2_ERROR_JIT_UNSUPPORTED;
  }

/* Checking flags and updating ovector_start. */
if (mode == PCRE2_JIT_COMPLETE &&
    (re->flags & PCRE2_LASTSET) != 0 &&
    (re->optimization_flags & PCRE2_OPTIM_START_OPTIMIZE) != 0)
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
private_data_size = common->cbra_ptr + (re->top_bracket + 1) * sizeof(sljit_sw);

if ((re->overall_options & PCRE2_ANCHORED) == 0 &&
    (re->optimization_flags & PCRE2_OPTIM_START_OPTIMIZE) != 0 &&
    !common->has_skip_in_assert_back)
  detect_early_fail(common, common->start, &private_data_size, 0, 0);

set_private_data_ptrs(common, &private_data_size, ccend);

SLJIT_ASSERT(common->early_fail_start_ptr <= common->early_fail_end_ptr);

if (private_data_size > 65536)
  {
  SLJIT_FREE(common->private_data_ptrs, allocator_data);
  return PCRE2_ERROR_JIT_UNSUPPORTED;
  }

if (common->has_then)
  {
  total_length = ccend - common->start;
  common->then_offsets = (sljit_u8 *)SLJIT_MALLOC(total_length, allocator_data);
  if (!common->then_offsets)
    {
    SLJIT_FREE(common->private_data_ptrs, allocator_data);
    return PCRE2_ERROR_NOMEMORY;
    }
  memset(common->then_offsets, 0, total_length);
  set_then_offsets(common, common->start, NULL);
  }

compiler = sljit_create_compiler(allocator_data);
if (!compiler)
  {
  SLJIT_FREE(common->private_data_ptrs, allocator_data);
  if (common->has_then)
    SLJIT_FREE(common->then_offsets, allocator_data);
  return PCRE2_ERROR_NOMEMORY;
  }
common->compiler = compiler;

/* Main pcre2_jit_exec entry. */
SLJIT_ASSERT((private_data_size & (sizeof(sljit_sw) - 1)) == 0);
sljit_emit_enter(compiler, 0, SLJIT_ARGS1(W, W), 5 | SLJIT_ENTER_VECTOR(SLJIT_NUMBER_OF_SCRATCH_VECTOR_REGISTERS), 5, private_data_size);

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
OP1(SLJIT_MOV, STACK_TOP, 0, SLJIT_MEM1(TMP2), SLJIT_OFFSETOF(struct sljit_stack, end));
OP1(SLJIT_MOV, STACK_LIMIT, 0, SLJIT_MEM1(TMP2), SLJIT_OFFSETOF(struct sljit_stack, start));
OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, 1);
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LIMIT_MATCH, TMP1, 0);

if (common->early_fail_start_ptr < common->early_fail_end_ptr)
  reset_early_fail(common);

if (mode == PCRE2_JIT_PARTIAL_SOFT)
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->hit_start, SLJIT_IMM, -1);
if (common->mark_ptr != 0)
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->mark_ptr, SLJIT_IMM, 0);
if (common->control_head_ptr != 0)
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->control_head_ptr, SLJIT_IMM, 0);

/* Main part of the matching */
if ((re->overall_options & PCRE2_ANCHORED) == 0)
  {
  mainloop_label = mainloop_entry(common);
  continue_match_label = LABEL();
  /* Forward search if possible. */
  if ((re->optimization_flags & PCRE2_OPTIM_START_OPTIMIZE) != 0)
    {
    if (mode == PCRE2_JIT_COMPLETE && fast_forward_first_n_chars(common))
      ;
    else if ((re->flags & PCRE2_FIRSTSET) != 0)
      fast_forward_first_char(common);
    else if ((re->flags & PCRE2_STARTLINE) != 0)
      fast_forward_newline(common);
    else if ((re->flags & PCRE2_FIRSTMAPSET) != 0)
      fast_forward_start_bits(common);
    }
  }
else
  continue_match_label = LABEL();

if (mode == PCRE2_JIT_COMPLETE && re->minlength > 0 &&
    (re->optimization_flags & PCRE2_OPTIM_START_OPTIMIZE) != 0)
  {
  OP1(SLJIT_MOV, SLJIT_RETURN_REG, 0, SLJIT_IMM, PCRE2_ERROR_NOMATCH);
  OP2(SLJIT_ADD, TMP2, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(re->minlength));
  minlength_check_failed = CMP(SLJIT_GREATER, TMP2, 0, STR_END, 0);
  }
if (common->req_char_ptr != 0)
  reqcu_not_found = search_requested_char(common, (PCRE2_UCHAR)(re->last_codeunit), (re->flags & PCRE2_LASTCASELESS) != 0, (re->flags & PCRE2_FIRSTSET) != 0);

/* Store the current STR_PTR in OVECTOR(0). */
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), OVECTOR(0), STR_PTR, 0);
/* Copy the limit of allowed recursions. */
OP1(SLJIT_MOV, COUNT_MATCH, 0, SLJIT_MEM1(SLJIT_SP), LIMIT_MATCH);
if (common->capture_last_ptr != 0)
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), common->capture_last_ptr, SLJIT_IMM, 0);
if (common->fast_forward_bc_ptr != NULL)
  OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), PRIVATE_DATA(common->fast_forward_bc_ptr + 1) >> 3, STR_PTR, 0);

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
  SLJIT_FREE(common->private_data_ptrs, allocator_data);
  if (common->has_then)
    SLJIT_FREE(common->then_offsets, allocator_data);
  PRIV(jit_free_rodata)(common->read_only_data_head, allocator_data);
  return PCRE2_ERROR_NOMEMORY;
  }

if ((re->overall_options & PCRE2_ENDANCHORED) != 0)
  end_anchor_failed = CMP(SLJIT_NOT_EQUAL, STR_PTR, 0, STR_END, 0);

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
common->quit_label = common->abort_label = LABEL();
if (common->quit != NULL)
  set_jumps(common->quit, common->quit_label);
if (common->abort != NULL)
  set_jumps(common->abort, common->abort_label);
if (minlength_check_failed != NULL)
  SET_LABEL(minlength_check_failed, common->abort_label);

sljit_emit_op0(compiler, SLJIT_SKIP_FRAMES_BEFORE_RETURN);
sljit_emit_return(compiler, SLJIT_MOV, SLJIT_RETURN_REG, 0);

if (common->failed_match != NULL)
  {
  SLJIT_ASSERT(common->mode == PCRE2_JIT_COMPLETE);
  set_jumps(common->failed_match, LABEL());
  OP1(SLJIT_MOV, SLJIT_RETURN_REG, 0, SLJIT_IMM, PCRE2_ERROR_NOMATCH);
  JUMPTO(SLJIT_JUMP, common->abort_label);
  }

if ((re->overall_options & PCRE2_ENDANCHORED) != 0)
  JUMPHERE(end_anchor_failed);

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
  SLJIT_FREE(common->private_data_ptrs, allocator_data);
  if (common->has_then)
    SLJIT_FREE(common->then_offsets, allocator_data);
  PRIV(jit_free_rodata)(common->read_only_data_head, allocator_data);
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
    (common->fast_forward_bc_ptr != NULL) ? (PRIVATE_DATA(common->fast_forward_bc_ptr + 1) >> 3) : common->start_ptr);

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
if (reqcu_not_found != NULL)
  set_jumps(reqcu_not_found, LABEL());

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
  OP2U(SLJIT_AND | SLJIT_SET_Z, TMP2, 0, SLJIT_IMM, PCRE2_NOTEMPTY);
  JUMPTO(SLJIT_NOT_ZERO, empty_match_backtrack_label);
  OP2U(SLJIT_AND | SLJIT_SET_Z, TMP2, 0, SLJIT_IMM, PCRE2_NOTEMPTY_ATSTART);
  JUMPTO(SLJIT_ZERO, empty_match_found_label);
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(TMP1), SLJIT_OFFSETOF(jit_arguments, str));
  CMPTO(SLJIT_NOT_EQUAL, TMP2, 0, STR_PTR, 0, empty_match_found_label);
  JUMPTO(SLJIT_JUMP, empty_match_backtrack_label);
  }

common->fast_forward_bc_ptr = NULL;
common->early_fail_start_ptr = 0;
common->early_fail_end_ptr = 0;
common->currententry = common->entries;
common->local_quit_available = TRUE;
quit_label = common->quit_label;
SLJIT_ASSERT(common->restore_end_ptr == 0);

if (common->currententry != NULL)
  {
  /* A free bit for each private data. */
  common->recurse_bitset_size = ((private_data_size / SSIZE_OF(sw)) + 7) >> 3;
  SLJIT_ASSERT(common->recurse_bitset_size > 0);
  common->recurse_bitset = (sljit_u8*)SLJIT_MALLOC(common->recurse_bitset_size, allocator_data);;

  if (common->recurse_bitset != NULL)
    {
    do
      {
      /* Might add new entries. */
      compile_recurse(common);
      if (SLJIT_UNLIKELY(sljit_get_compiler_error(compiler)))
        break;
      flush_stubs(common);
      common->currententry = common->currententry->next;
      }
    while (common->currententry != NULL);

    SLJIT_FREE(common->recurse_bitset, allocator_data);
    }

  if (common->currententry != NULL)
    {
    /* The common->recurse_bitset has been freed. */
    SLJIT_ASSERT(sljit_get_compiler_error(compiler) || common->recurse_bitset == NULL);

    sljit_free_compiler(compiler);
    SLJIT_FREE(common->private_data_ptrs, allocator_data);
    if (common->has_then)
      SLJIT_FREE(common->then_offsets, allocator_data);
    PRIV(jit_free_rodata)(common->read_only_data_head, allocator_data);
    return PCRE2_ERROR_NOMEMORY;
    }
  }

common->local_quit_available = FALSE;
common->quit_label = quit_label;
SLJIT_ASSERT(common->restore_end_ptr == 0);

/* Allocating stack, returns with PCRE2_ERROR_JIT_STACKLIMIT if fails. */
/* This is a (really) rare case. */
set_jumps(common->stackalloc, LABEL());
/* RETURN_ADDR is not a saved register. */
SLJIT_ASSERT(common->locals_size >= 2 * SSIZE_OF(sw));
sljit_emit_op_dst(compiler, SLJIT_FAST_ENTER, SLJIT_MEM1(SLJIT_SP), LOCAL0);

SLJIT_ASSERT(TMP1 == SLJIT_R0 && STR_PTR == SLJIT_R1);

OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCAL1, STR_PTR, 0);
OP1(SLJIT_MOV, SLJIT_R0, 0, ARGUMENTS, 0);
OP2(SLJIT_SUB, SLJIT_R1, 0, STACK_LIMIT, 0, SLJIT_IMM, STACK_GROWTH_RATE);
OP1(SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(SLJIT_R0), SLJIT_OFFSETOF(jit_arguments, stack));
OP1(SLJIT_MOV, STACK_LIMIT, 0, TMP2, 0);

sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_ARGS2(W, W, W), SLJIT_IMM, SLJIT_FUNC_ADDR(sljit_stack_resize));

jump = CMP(SLJIT_EQUAL, SLJIT_RETURN_REG, 0, SLJIT_IMM, 0);
OP1(SLJIT_MOV, TMP2, 0, STACK_LIMIT, 0);
OP1(SLJIT_MOV, STACK_LIMIT, 0, SLJIT_RETURN_REG, 0);
OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), LOCAL0);
OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(SLJIT_SP), LOCAL1);
OP_SRC(SLJIT_FAST_RETURN, TMP1, 0);

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
  check_wordboundary(common, FALSE);
  }
if (common->ucp_wordboundary != NULL)
  {
  set_jumps(common->ucp_wordboundary, LABEL());
  check_wordboundary(common, TRUE);
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
if (common->reset_match != NULL || common->restart_match != NULL)
  {
  if (common->restart_match != NULL)
    {
    set_jumps(common->restart_match, LABEL());
    OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(SLJIT_SP), common->start_ptr);
    }

  set_jumps(common->reset_match, LABEL());
  do_reset_match(common, (re->top_bracket + 1) * 2);
  /* The value of restart_match is in TMP1. */
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
if (common->utfreadtype8 != NULL)
  {
  set_jumps(common->utfreadtype8, LABEL());
  do_utfreadtype8(common);
  }
if (common->utfpeakcharback != NULL)
  {
  set_jumps(common->utfpeakcharback, LABEL());
  do_utfpeakcharback(common);
  }
#endif /* PCRE2_CODE_UNIT_WIDTH == 8 */
#if PCRE2_CODE_UNIT_WIDTH == 8 || PCRE2_CODE_UNIT_WIDTH == 16
if (common->utfreadchar_invalid != NULL)
  {
  set_jumps(common->utfreadchar_invalid, LABEL());
  do_utfreadchar_invalid(common);
  }
if (common->utfreadnewline_invalid != NULL)
  {
  set_jumps(common->utfreadnewline_invalid, LABEL());
  do_utfreadnewline_invalid(common);
  }
if (common->utfmoveback_invalid)
  {
  set_jumps(common->utfmoveback_invalid, LABEL());
  do_utfmoveback_invalid(common);
  }
if (common->utfpeakcharback_invalid)
  {
  set_jumps(common->utfpeakcharback_invalid, LABEL());
  do_utfpeakcharback_invalid(common);
  }
#endif /* PCRE2_CODE_UNIT_WIDTH == 8 || PCRE2_CODE_UNIT_WIDTH == 16 */
if (common->getucd != NULL)
  {
  set_jumps(common->getucd, LABEL());
  do_getucd(common);
  }
if (common->getucdtype != NULL)
  {
  set_jumps(common->getucdtype, LABEL());
  do_getucdtype(common);
  }
#endif /* SUPPORT_UNICODE */

SLJIT_FREE(common->private_data_ptrs, allocator_data);
if (common->has_then)
  SLJIT_FREE(common->then_offsets, allocator_data);

executable_func = sljit_generate_code(compiler, 0, NULL);
executable_size = sljit_get_generated_code_size(compiler);
sljit_free_compiler(compiler);

if (executable_func == NULL)
  {
  PRIV(jit_free_rodata)(common->read_only_data_head, allocator_data);
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
    sljit_free_code(executable_func, NULL);
    PRIV(jit_free_rodata)(common->read_only_data_head, allocator_data);
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
  (PCRE2_JIT_COMPLETE|PCRE2_JIT_PARTIAL_SOFT|PCRE2_JIT_PARTIAL_HARD|PCRE2_JIT_INVALID_UTF)

PCRE2_EXP_DEFN int PCRE2_CALL_CONVENTION
pcre2_jit_compile(pcre2_code *code, uint32_t options)
{
pcre2_real_code *re = (pcre2_real_code *)code;
#ifdef SUPPORT_JIT
void *exec_memory;
executable_functions *functions;
static int executable_allocator_is_working = -1;

if (executable_allocator_is_working == -1)
  {
  /* Checks whether the executable allocator is working. This check
     might run multiple times in multi-threaded environments, but the
     result should not be affected by it. */
  exec_memory = SLJIT_MALLOC_EXEC(32, NULL);
  if (exec_memory != NULL)
    {
    SLJIT_FREE_EXEC(((sljit_u8*)(exec_memory)) + SLJIT_EXEC_OFFSET(exec_memory), NULL);
    executable_allocator_is_working = 1;
    }
  else executable_allocator_is_working = 0;
  }
#endif

if (options & PCRE2_JIT_TEST_ALLOC)
  {
  if (options != PCRE2_JIT_TEST_ALLOC)
    return PCRE2_ERROR_JIT_BADOPTION;

#ifdef SUPPORT_JIT
  return executable_allocator_is_working ? 0 : PCRE2_ERROR_NOMEMORY;
#else
  return PCRE2_ERROR_JIT_UNSUPPORTED;
#endif
  }

if (code == NULL)
  return PCRE2_ERROR_NULL;

if ((options & ~PUBLIC_JIT_COMPILE_OPTIONS) != 0)
  return PCRE2_ERROR_JIT_BADOPTION;

/* Support for invalid UTF was first introduced in JIT, with the option
PCRE2_JIT_INVALID_UTF. Later, support was added to the interpreter, and the
compile-time option PCRE2_MATCH_INVALID_UTF was created. This is now the
preferred feature, with the earlier option deprecated. However, for backward
compatibility, if the earlier option is set, it forces the new option so that
if JIT matching falls back to the interpreter, there is still support for
invalid UTF. However, if this function has already been successfully called
without PCRE2_JIT_INVALID_UTF and without PCRE2_MATCH_INVALID_UTF (meaning that
non-invalid-supporting JIT code was compiled), give an error.

If in the future support for PCRE2_JIT_INVALID_UTF is withdrawn, the following
actions are needed:

  1. Remove the definition from pcre2.h.in and from the list in
     PUBLIC_JIT_COMPILE_OPTIONS above.

  2. Replace PCRE2_JIT_INVALID_UTF with a local flag in this module.

  3. Replace PCRE2_JIT_INVALID_UTF in pcre2_jit_test.c.

  4. Delete the following short block of code. The setting of "re" and
     "functions" can be moved into the JIT-only block below, but if that is
     done, (void)re and (void)functions will be needed in the non-JIT case, to
     avoid compiler warnings.
*/

#ifdef SUPPORT_JIT
functions = (executable_functions *)re->executable_jit;
#endif

if ((options & PCRE2_JIT_INVALID_UTF) != 0)
  {
  if ((re->overall_options & PCRE2_MATCH_INVALID_UTF) == 0)
    {
#ifdef SUPPORT_JIT
    if (functions != NULL) return PCRE2_ERROR_JIT_BADOPTION;
#endif
    re->overall_options |= PCRE2_MATCH_INVALID_UTF;
    }
  }

/* The above tests are run with and without JIT support. This means that
PCRE2_JIT_INVALID_UTF propagates back into the regex options (ensuring
interpreter support) even in the absence of JIT. But now, if there is no JIT
support, give an error return. */

#ifndef SUPPORT_JIT
return PCRE2_ERROR_JIT_BADOPTION;
#else  /* SUPPORT_JIT */

/* There is JIT support. Do the necessary. */

if ((re->flags & PCRE2_NOJIT) != 0) return 0;

if (!executable_allocator_is_working)
  return PCRE2_ERROR_NOMEMORY;

if ((re->overall_options & PCRE2_MATCH_INVALID_UTF) != 0)
  options |= PCRE2_JIT_INVALID_UTF;

if ((options & PCRE2_JIT_COMPLETE) != 0 && (functions == NULL
    || functions->executable_funcs[0] == NULL)) {
  uint32_t excluded_options = (PCRE2_JIT_PARTIAL_SOFT | PCRE2_JIT_PARTIAL_HARD);
  int result = jit_compile(code, options & ~excluded_options);
  if (result != 0)
    return result;
  }

if ((options & PCRE2_JIT_PARTIAL_SOFT) != 0 && (functions == NULL
    || functions->executable_funcs[1] == NULL)) {
  uint32_t excluded_options = (PCRE2_JIT_COMPLETE | PCRE2_JIT_PARTIAL_HARD);
  int result = jit_compile(code, options & ~excluded_options);
  if (result != 0)
    return result;
  }

if ((options & PCRE2_JIT_PARTIAL_HARD) != 0 && (functions == NULL
    || functions->executable_funcs[2] == NULL)) {
  uint32_t excluded_options = (PCRE2_JIT_COMPLETE | PCRE2_JIT_PARTIAL_SOFT);
  int result = jit_compile(code, options & ~excluded_options);
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
