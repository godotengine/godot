/*************************************************
*      Perl-Compatible Regular Expressions       *
*************************************************/

/* PCRE is a library of functions to support regular expressions whose syntax
and semantics are as close as possible to those of the Perl 5 language.

                       Written by Philip Hazel
     Original API code Copyright (c) 1997-2012 University of Cambridge
          New API code Copyright (c) 2016-2021 University of Cambridge

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

/* This module contains the function for checking a script run. */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "pcre2_internal.h"


/*************************************************
*                Check script run                *
*************************************************/

/* A script run is conceptually a sequence of characters all in the same
Unicode script. However, it isn't quite that simple. There are special rules
for scripts that are commonly used together, and also special rules for digits.
This function implements the appropriate checks, which is possible only when
PCRE2 is compiled with Unicode support. The function returns TRUE if there is
no Unicode support; however, it should never be called in that circumstance
because an error is given by pcre2_compile() if a script run is called for in a
version of PCRE2 compiled without Unicode support.

Arguments:
  pgr       point to the first character
  endptr    point after the last character
  utf       TRUE if in UTF mode

Returns:    TRUE if this is a valid script run
*/

/* These are states in the checking process. */

enum { SCRIPT_UNSET,          /* Requirement as yet unknown */
       SCRIPT_MAP,            /* Bitmap contains acceptable scripts */
       SCRIPT_HANPENDING,     /* Have had only Han characters */
       SCRIPT_HANHIRAKATA,    /* Expect Han or Hirikata */
       SCRIPT_HANBOPOMOFO,    /* Expect Han or Bopomofo */
       SCRIPT_HANHANGUL       /* Expect Han or Hangul */
       };

#define UCD_MAPSIZE (ucp_Unknown/32 + 1)
#define FULL_MAPSIZE (ucp_Script_Count/32 + 1)

BOOL
PRIV(script_run)(PCRE2_SPTR ptr, PCRE2_SPTR endptr, BOOL utf)
{
#ifdef SUPPORT_UNICODE
uint32_t require_state = SCRIPT_UNSET;
uint32_t require_map[FULL_MAPSIZE];
uint32_t map[FULL_MAPSIZE];
uint32_t require_digitset = 0;
uint32_t c;

#if PCRE2_CODE_UNIT_WIDTH == 32
(void)utf;    /* Avoid compiler warning */
#endif

/* Any string containing fewer than 2 characters is a valid script run. */

if (ptr >= endptr) return TRUE;
GETCHARINCTEST(c, ptr);
if (ptr >= endptr) return TRUE;

/* Initialize the require map. This is a full-size bitmap that has a bit for
every script, as opposed to the maps in ucd_script_sets, which only have bits
for scripts less than ucp_Unknown - those that appear in script extension
lists. */

for (int i = 0; i < FULL_MAPSIZE; i++) require_map[i] = 0;

/* Scan strings of two or more characters, checking the Unicode characteristics
of each code point. There is special code for scripts that can be combined with
characters from the Han Chinese script. This may be used in conjunction with
four other scripts in these combinations:

. Han with Hiragana and Katakana is allowed (for Japanese).
. Han with Bopomofo is allowed (for Taiwanese Mandarin).
. Han with Hangul is allowed (for Korean).

If the first significant character's script is one of the four, the required
script type is immediately known. However, if the first significant
character's script is Han, we have to keep checking for a non-Han character.
Hence the SCRIPT_HANPENDING state. */

for (;;)
  {
  const ucd_record *ucd = GET_UCD(c);
  uint32_t script = ucd->script;

  /* If the script is Unknown, the string is not a valid script run. Such
  characters can only form script runs of length one (see test above). */

  if (script == ucp_Unknown) return FALSE;

  /* A character without any script extensions whose script is Inherited or
  Common is always accepted with any script. If there are extensions, the
  following processing happens for all scripts. */

  if (UCD_SCRIPTX_PROP(ucd) != 0 || (script != ucp_Inherited && script != ucp_Common))
    {
    BOOL OK;

    /* Set up a full-sized map for this character that can include bits for all
    scripts. Copy the scriptx map for this character (which covers those
    scripts that appear in script extension lists), set the remaining values to
    zero, and then, except for Common or Inherited, add this script's bit to
    the map. */

    memcpy(map, PRIV(ucd_script_sets) + UCD_SCRIPTX_PROP(ucd), UCD_MAPSIZE * sizeof(uint32_t));
    memset(map + UCD_MAPSIZE, 0, (FULL_MAPSIZE - UCD_MAPSIZE) * sizeof(uint32_t));
    if (script != ucp_Common && script != ucp_Inherited) MAPSET(map, script);

    /* Handle the different checking states */

    switch(require_state)
      {
      /* First significant character - it might follow Common or Inherited
      characters that do not have any script extensions. */

      case SCRIPT_UNSET:
      switch(script)
        {
        case ucp_Han:
        require_state = SCRIPT_HANPENDING;
        break;

        case ucp_Hiragana:
        case ucp_Katakana:
        require_state = SCRIPT_HANHIRAKATA;
        break;

        case ucp_Bopomofo:
        require_state = SCRIPT_HANBOPOMOFO;
        break;

        case ucp_Hangul:
        require_state = SCRIPT_HANHANGUL;
        break;

        default:
        memcpy(require_map, map, FULL_MAPSIZE * sizeof(uint32_t));
        require_state = SCRIPT_MAP;
        break;
        }
      break;

      /* The first significant character was Han. An inspection of the Unicode
      11.0.0 files shows that there are the following types of Script Extension
      list that involve the Han, Bopomofo, Hiragana, Katakana, and Hangul
      scripts:

      . Bopomofo + Han
      . Han + Hiragana + Katakana
      . Hiragana + Katakana
      . Bopopmofo + Hangul + Han + Hiragana + Katakana

      The following code tries to make sense of this. */

#define FOUND_BOPOMOFO 1
#define FOUND_HIRAGANA 2
#define FOUND_KATAKANA 4
#define FOUND_HANGUL   8

      case SCRIPT_HANPENDING:
      if (script != ucp_Han)   /* Another Han does nothing */
        {
        uint32_t chspecial = 0;

        if (MAPBIT(map, ucp_Bopomofo) != 0) chspecial |= FOUND_BOPOMOFO;
        if (MAPBIT(map, ucp_Hiragana) != 0) chspecial |= FOUND_HIRAGANA;
        if (MAPBIT(map, ucp_Katakana) != 0) chspecial |= FOUND_KATAKANA;
        if (MAPBIT(map, ucp_Hangul) != 0)   chspecial |= FOUND_HANGUL;

        if (chspecial == 0) return FALSE;   /* Not allowed with Han */

        if (chspecial == FOUND_BOPOMOFO)
          require_state = SCRIPT_HANBOPOMOFO;
        else if (chspecial == (FOUND_HIRAGANA|FOUND_KATAKANA))
          require_state = SCRIPT_HANHIRAKATA;

        /* Otherwise this character must be allowed with all of them, so remain
        in the pending state. */
        }
      break;

      /* Previously encountered one of the "with Han" scripts. Check that
      this character is appropriate. */

      case SCRIPT_HANHIRAKATA:
      if (MAPBIT(map, ucp_Han) + MAPBIT(map, ucp_Hiragana) +
          MAPBIT(map, ucp_Katakana) == 0) return FALSE;
      break;

      case SCRIPT_HANBOPOMOFO:
      if (MAPBIT(map, ucp_Han) + MAPBIT(map, ucp_Bopomofo) == 0) return FALSE;
      break;

      case SCRIPT_HANHANGUL:
      if (MAPBIT(map, ucp_Han) + MAPBIT(map, ucp_Hangul) == 0) return FALSE;
      break;

      /* Previously encountered one or more characters that are allowed with a
      list of scripts. */

      case SCRIPT_MAP:
      OK = FALSE;

      for (int i = 0; i < FULL_MAPSIZE; i++)
        {
        if ((require_map[i] & map[i]) != 0)
          {
          OK = TRUE;
          break;
          }
        }

      if (!OK) return FALSE;

      /* The rest of the string must be in this script, but we have to
      allow for the Han complications. */

      switch(script)
        {
        case ucp_Han:
        require_state = SCRIPT_HANPENDING;
        break;

        case ucp_Hiragana:
        case ucp_Katakana:
        require_state = SCRIPT_HANHIRAKATA;
        break;

        case ucp_Bopomofo:
        require_state = SCRIPT_HANBOPOMOFO;
        break;

        case ucp_Hangul:
        require_state = SCRIPT_HANHANGUL;
        break;

        /* Compute the intersection of the required list of scripts and the
        allowed scripts for this character. */

        default:
        for (int i = 0; i < FULL_MAPSIZE; i++) require_map[i] &= map[i];
        break;
        }

      break;
      }
    }   /* End checking character's script and extensions. */

  /* The character is in an acceptable script. We must now ensure that all
  decimal digits in the string come from the same set. Some scripts (e.g.
  Common, Arabic) have more than one set of decimal digits. This code does
  not allow mixing sets, even within the same script. The vector called
  PRIV(ucd_digit_sets)[] contains, in its first element, the number of
  following elements, and then, in ascending order, the code points of the
  '9' characters in every set of 10 digits. Each set is identified by the
  offset in the vector of its '9' character. An initial check of the first
  value picks up ASCII digits quickly. Otherwise, a binary chop is used. */

  if (ucd->chartype == ucp_Nd)
    {
    uint32_t digitset;

    if (c <= PRIV(ucd_digit_sets)[1]) digitset = 1; else
      {
      int mid;
      int bot = 1;
      int top = PRIV(ucd_digit_sets)[0];
      for (;;)
        {
        if (top <= bot + 1)    /* <= rather than == is paranoia */
          {
          digitset = top;
          break;
          }
        mid = (top + bot) / 2;
        if (c <= PRIV(ucd_digit_sets)[mid]) top = mid; else bot = mid;
        }
      }

    /* A required value of 0 means "unset". */

    if (require_digitset == 0) require_digitset = digitset;
      else if (digitset != require_digitset) return FALSE;
    }   /* End digit handling */

  /* If we haven't yet got to the end, pick up the next character. */

  if (ptr >= endptr) return TRUE;
  GETCHARINCTEST(c, ptr);
  }  /* End checking loop */

#else   /* NOT SUPPORT_UNICODE */
(void)ptr;
(void)endptr;
(void)utf;
return TRUE;
#endif  /* SUPPORT_UNICODE */
}

/* End of pcre2_script_run.c */
