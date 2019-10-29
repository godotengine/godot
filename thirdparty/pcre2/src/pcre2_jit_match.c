/*************************************************
*      Perl-Compatible Regular Expressions       *
*************************************************/

/* PCRE is a library of functions to support regular expressions whose syntax
and semantics are as close as possible to those of the Perl 5 language.

                       Written by Philip Hazel
     Original API code Copyright (c) 1997-2012 University of Cambridge
          New API code Copyright (c) 2016-2018 University of Cambridge

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

#ifndef INCLUDED_FROM_PCRE2_JIT_COMPILE
#error This file must be included from pcre2_jit_compile.c.
#endif

#ifdef SUPPORT_JIT

static SLJIT_NOINLINE int jit_machine_stack_exec(jit_arguments *arguments, jit_function executable_func)
{
sljit_u8 local_space[MACHINE_STACK_SIZE];
struct sljit_stack local_stack;

local_stack.min_start = local_space;
local_stack.start = local_space;
local_stack.end = local_space + MACHINE_STACK_SIZE;
local_stack.top = local_space + MACHINE_STACK_SIZE;
arguments->stack = &local_stack;
return executable_func(arguments);
}

#endif


/*************************************************
*              Do a JIT pattern match            *
*************************************************/

/* This function runs a JIT pattern match.

Arguments:
  code            points to the compiled expression
  subject         points to the subject string
  length          length of subject string (may contain binary zeros)
  start_offset    where to start in the subject string
  options         option bits
  match_data      points to a match_data block
  mcontext        points to a match context
  jit_stack       points to a JIT stack

Returns:          > 0 => success; value is the number of ovector pairs filled
                  = 0 => success, but ovector is not big enough
                   -1 => failed to match (PCRE_ERROR_NOMATCH)
                 < -1 => some kind of unexpected problem
*/

PCRE2_EXP_DEFN int PCRE2_CALL_CONVENTION
pcre2_jit_match(const pcre2_code *code, PCRE2_SPTR subject, PCRE2_SIZE length,
  PCRE2_SIZE start_offset, uint32_t options, pcre2_match_data *match_data,
  pcre2_match_context *mcontext)
{
#ifndef SUPPORT_JIT

(void)code;
(void)subject;
(void)length;
(void)start_offset;
(void)options;
(void)match_data;
(void)mcontext;
return PCRE2_ERROR_JIT_BADOPTION;

#else  /* SUPPORT_JIT */

pcre2_real_code *re = (pcre2_real_code *)code;
executable_functions *functions = (executable_functions *)re->executable_jit;
pcre2_jit_stack *jit_stack;
uint32_t oveccount = match_data->oveccount;
uint32_t max_oveccount;
union {
   void *executable_func;
   jit_function call_executable_func;
} convert_executable_func;
jit_arguments arguments;
int rc;
int index = 0;

if ((options & PCRE2_PARTIAL_HARD) != 0)
  index = 2;
else if ((options & PCRE2_PARTIAL_SOFT) != 0)
  index = 1;

if (functions == NULL || functions->executable_funcs[index] == NULL)
  return PCRE2_ERROR_JIT_BADOPTION;

/* Sanity checks should be handled by pcre_exec. */
arguments.str = subject + start_offset;
arguments.begin = subject;
arguments.end = subject + length;
arguments.match_data = match_data;
arguments.startchar_ptr = subject;
arguments.mark_ptr = NULL;
arguments.options = options;

if (mcontext != NULL)
  {
  arguments.callout = mcontext->callout;
  arguments.callout_data = mcontext->callout_data;
  arguments.offset_limit = mcontext->offset_limit;
  arguments.limit_match = (mcontext->match_limit < re->limit_match)?
    mcontext->match_limit : re->limit_match;
  if (mcontext->jit_callback != NULL)
    jit_stack = mcontext->jit_callback(mcontext->jit_callback_data);
  else
    jit_stack = (pcre2_jit_stack *)mcontext->jit_callback_data;
  }
else
  {
  arguments.callout = NULL;
  arguments.callout_data = NULL;
  arguments.offset_limit = PCRE2_UNSET;
  arguments.limit_match = (MATCH_LIMIT < re->limit_match)?
    MATCH_LIMIT : re->limit_match;
  jit_stack = NULL;
  }


max_oveccount = functions->top_bracket;
if (oveccount > max_oveccount)
  oveccount = max_oveccount;
arguments.oveccount = oveccount << 1;


convert_executable_func.executable_func = functions->executable_funcs[index];
if (jit_stack != NULL)
  {
  arguments.stack = (struct sljit_stack *)(jit_stack->stack);
  rc = convert_executable_func.call_executable_func(&arguments);
  }
else
  rc = jit_machine_stack_exec(&arguments, convert_executable_func.call_executable_func);

if (rc > (int)oveccount)
  rc = 0;
match_data->code = re;
match_data->subject = (rc >= 0 || rc == PCRE2_ERROR_PARTIAL)? subject : NULL;
match_data->rc = rc;
match_data->startchar = arguments.startchar_ptr - subject;
match_data->leftchar = 0;
match_data->rightchar = 0;
match_data->mark = arguments.mark_ptr;
match_data->matchedby = PCRE2_MATCHEDBY_JIT;

return match_data->rc;

#endif  /* SUPPORT_JIT */
}

/* End of pcre2_jit_match.c */
