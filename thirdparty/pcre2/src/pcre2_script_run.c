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

/* These dummy values must be less than the negation of the largest offset in
the PRIV(ucd_script_sets) vector, which is held in a 16-bit field in UCD
records (and is only likely to be a few hundred). */

#define SCRIPT_UNSET        (-99999)
#define SCRIPT_HANPENDING   (-99998)
#define SCRIPT_HANHIRAKATA  (-99997)
#define SCRIPT_HANBOPOMOFO  (-99996)
#define SCRIPT_HANHANGUL    (-99995)
#define SCRIPT_LIST         (-99994)

#define INTERSECTION_LIST_SIZE 50

BOOL
PRIV(script_run)(PCRE2_SPTR ptr, PCRE2_SPTR endptr, BOOL utf)
{
#ifdef SUPPORT_UNICODE
int require_script = SCRIPT_UNSET;
uint8_t intersection_list[INTERSECTION_LIST_SIZE];
const uint8_t *require_list = NULL;
uint32_t require_digitset = 0;
uint32_t c;

#if PCRE2_CODE_UNIT_WIDTH == 32
(void)utf;    /* Avoid compiler warning */
#endif

/* Any string containing fewer than 2 characters is a valid script run. */

if (ptr >= endptr) return TRUE;
GETCHARINCTEST(c, ptr);
if (ptr >= endptr) return TRUE;

/* Scan strings of two or more characters, checking the Unicode characteristics
of each code point. We make use of the Script Extensions property. There is
special code for scripts that can be combined with characters from the Han
Chinese script. This may be used in conjunction with four other scripts in
these combinations:

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
  int32_t scriptx = ucd->scriptx;

  /* If the script extension is Unknown, the string is not a valid script run.
  Such characters can only form script runs of length one. */

  if (scriptx == ucp_Unknown) return FALSE;

  /* A character whose script extension is Inherited is always accepted with
  any script, and plays no further part in this testing. A character whose
  script is Common is always accepted, but must still be tested for a digit
  below. The scriptx value at this point is non-zero, because zero is
  ucp_Unknown, tested for above. */

  if (scriptx != ucp_Inherited)
    {
    if (scriptx != ucp_Common)
      {
      /* If the script extension value is positive, the character is not a mark
      that can be used with many scripts. In the simple case we either set or
      compare with the required script. However, handling the scripts that can
      combine with Han are more complicated, as is the case when the previous
      characters have been man-script marks. */

      if (scriptx > 0)
        {
        switch(require_script)
          {
          /* Either the first significant character (require_script unset) or
          after only Han characters. */

          case SCRIPT_UNSET:
          case SCRIPT_HANPENDING:
          switch(scriptx)
            {
            case ucp_Han:
            require_script = SCRIPT_HANPENDING;
            break;

            case ucp_Hiragana:
            case ucp_Katakana:
            require_script = SCRIPT_HANHIRAKATA;
            break;

            case ucp_Bopomofo:
            require_script = SCRIPT_HANBOPOMOFO;
            break;

            case ucp_Hangul:
            require_script = SCRIPT_HANHANGUL;
            break;

            /* Not a Han-related script. If expecting one, fail. Otherise set
            the requirement to this script. */

            default:
            if (require_script == SCRIPT_HANPENDING) return FALSE;
            require_script = scriptx;
            break;
            }
          break;

          /* Previously encountered one of the "with Han" scripts. Check that
          this character is appropriate. */

          case SCRIPT_HANHIRAKATA:
          if (scriptx != ucp_Han && scriptx != ucp_Hiragana && 
              scriptx != ucp_Katakana)
            return FALSE;
          break;

          case SCRIPT_HANBOPOMOFO:
          if (scriptx != ucp_Han && scriptx != ucp_Bopomofo) return FALSE;
          break;

          case SCRIPT_HANHANGUL:
          if (scriptx != ucp_Han && scriptx != ucp_Hangul) return FALSE;
          break;

          /* We have a list of scripts to check that is derived from one or
          more previous characters. This is either one of the lists in
          ucd_script_sets[] (for one previous character) or the intersection of
          several lists for multiple characters. */

          case SCRIPT_LIST:
            {
            const uint8_t *list;
            for (list = require_list; *list != 0; list++)
              {
              if (*list == scriptx) break;
              }
            if (*list == 0) return FALSE;
            }

          /* The rest of the string must be in this script, but we have to 
          allow for the Han complications. */
          
          switch(scriptx)
            {
            case ucp_Han:
            require_script = SCRIPT_HANPENDING;
            break;

            case ucp_Hiragana:
            case ucp_Katakana:
            require_script = SCRIPT_HANHIRAKATA;
            break;

            case ucp_Bopomofo:
            require_script = SCRIPT_HANBOPOMOFO;
            break;

            case ucp_Hangul:
            require_script = SCRIPT_HANHANGUL;
            break;

            default:
            require_script = scriptx;
            break;
            }  
          break;

          /* This is the easy case when a single script is required. */

          default:
          if (scriptx != require_script) return FALSE;
          break;
          }
        }  /* End of handing positive scriptx */

      /* If scriptx is negative, this character is a mark-type character that
      has a list of permitted scripts. */

      else
        {
        uint32_t chspecial;
        const uint8_t *clist, *rlist;
        const uint8_t *list = PRIV(ucd_script_sets) - scriptx;
        
        switch(require_script)
          {
          case SCRIPT_UNSET:
          require_list = PRIV(ucd_script_sets) - scriptx;
          require_script = SCRIPT_LIST;
          break;

          /* An inspection of the Unicode 11.0.0 files shows that there are the
          following types of Script Extension list that involve the Han,
          Bopomofo, Hiragana, Katakana, and Hangul scripts:

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
          chspecial = 0;
          for (; *list != 0; list++)
            {
            switch (*list)
              {
              case ucp_Bopomofo: chspecial |= FOUND_BOPOMOFO; break;
              case ucp_Hiragana: chspecial |= FOUND_HIRAGANA; break;
              case ucp_Katakana: chspecial |= FOUND_KATAKANA; break;
              case ucp_Hangul:   chspecial |= FOUND_HANGUL; break;
              default: break;
              }
            }

           if (chspecial == 0) return FALSE;

           if (chspecial == FOUND_BOPOMOFO)
             {
             require_script = SCRIPT_HANBOPOMOFO;
             }
           else if (chspecial == (FOUND_HIRAGANA|FOUND_KATAKANA))
             {
             require_script = SCRIPT_HANHIRAKATA;
             }

          /* Otherwise it must be allowed with all of them, so remain in
          the pending state. */

          break;

          case SCRIPT_HANHIRAKATA:
          for (; *list != 0; list++)
            {
            if (*list == ucp_Hiragana || *list == ucp_Katakana) break;
            }
          if (*list == 0) return FALSE;
          break;

          case SCRIPT_HANBOPOMOFO:
          for (; *list != 0; list++)
            {
            if (*list == ucp_Bopomofo) break;
            }
          if (*list == 0) return FALSE;
          break;

          case SCRIPT_HANHANGUL:
          for (; *list != 0; list++)
            {
            if (*list == ucp_Hangul) break;
            }
          if (*list == 0) return FALSE;
          break;

          /* Previously encountered one or more characters that are allowed
          with a list of scripts. Build the intersection of the required list
          with this character's list in intersection_list[]. This code is
          written so that it still works OK if the required list is already in
          that vector. */

          case SCRIPT_LIST:
            {
            int i = 0;
            for (rlist = require_list; *rlist != 0; rlist++)
              {
              for (clist = list; *clist != 0; clist++)
                {
                if (*rlist == *clist)
                  {
                  intersection_list[i++] = *rlist;
                  break;
                  }
                }
              }
            if (i == 0) return FALSE;  /* No scripts in common */

            /* If there's just one script in common, we can set it as the
            unique required script. Otherwise, terminate the intersection list
            and make it the required list. */

            if (i == 1)
              {
              require_script = intersection_list[0];
              }
            else
              {
              intersection_list[i] = 0;
              require_list = intersection_list;
              }
            }
          break;

          /* The previously set required script is a single script, not
          Han-related. Check that it is in this character's list. */

          default:
          for (; *list != 0; list++)
            {
            if (*list == require_script) break;
            }
          if (*list == 0) return FALSE;
          break;
          }
        }  /* End of handling negative scriptx */
      }    /* End of checking non-Common character */

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
    }     /* End checking non-Inherited character */

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
