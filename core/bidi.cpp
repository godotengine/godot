/*************************************************************************/
/*  bidi.cpp                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

/**
	@author Masoud BaniHashemian <masoudbh3@gmail.com>
*/

#include "bidi.h"

enum BidiLinearEnum {
  LTR,
  RTL,
  AL,
  EN,
  AN,
  ES,
  ET,
  CS,
  NSM,
  BN,
  BS,
  SS,
  WS,
  ON,
  LRE,
  RLE,
  LRO,
  RLO,
  PDF,
};
enum BidiLinearARPEnum {
  U,
  R,
  D,
  C,
  T,
  L,
  G,
};

#include "bidi-char-type.tab.i"
#include "bidi-mirroring.tab.i"
#include "bidi-joining-type.tab.i"
#include "bidi-arabic-shaping.tab.i"

#define BIDI_SENTINEL -1
#define BIDI_CHAR_FILL 0xFEFF
#define BIDI_MAX_EXPLICIT_LEVEL 61

#define BIDI_TEST_BITS(x, mask) (((x) & (mask)) ? 1 : 0)
#define BIDI_SET_BITS(x, mask) ((x) |= (mask))
#define BIDI_UNSET_BITS(x, mask) ((x) &= ~(mask))

#define BIDI_IS_LETTER(p) ((p) & BIDI_MASK_LETTER)
#define BIDI_IS_NUMBER(p)   ((p) & BIDI_MASK_NUMBER)
#define BIDI_IS_NUMBER_SEPARATOR_OR_TERMINATOR(p) ((p) & BIDI_MASK_NUMSEPTER)
#define BIDI_IS_STRONG(p)   ((p) & BIDI_MASK_STRONG)
#define BIDI_IS_WEAK(p)     ((p) & BIDI_MASK_WEAK)
#define BIDI_IS_NEUTRAL(p)  ((p) & BIDI_MASK_NEUTRAL)
#define BIDI_IS_SENTINEL(p) ((p) & BIDI_MASK_SENTINEL)
#define BIDI_IS_OVERRIDE(p) ((p) & BIDI_MASK_OVERRIDE)
#define BIDI_LEVEL_IS_RTL(lev) ((lev) & 1)
#define BIDI_LEVEL_TO_DIR(lev) (BIDI_LEVEL_IS_RTL (lev) ? BidiDefs::CHAR_TYPE_RTL : BidiDefs::CHAR_TYPE_LTR)
#define BIDI_IS_RTL(p)      ((p) & BIDI_MASK_RTL)
#define BIDI_IS_ARABIC(p)   ((p) & BIDI_MASK_ARABIC)
#define BIDI_IS_SEPARATOR(p) ((p) & BIDI_MASK_SEPARATOR)
#define BIDI_ARAB_SHAPES(p) ((p) & BIDI_MASK_ARAB_SHAPES)
#define BIDI_JOIN_SHAPE(p) ((p) & ( BIDI_MASK_JOINS_RIGHT | BIDI_MASK_JOINS_LEFT ))
#define BIDI_DIR_TO_LEVEL(dir) (BIDI_IS_RTL (dir) ? 1 : 0)

#define BIDI_LEVELS_MATCH(i, j) ((i) == (j) || (i) == BIDI_SENTINEL || (j) == BIDI_SENTINEL)
#define BIDI_IS_JOINING_TYPE_G(p) (BIDI_MASK_IGNORED == ((p) & ( BIDI_MASK_TRANSPARENT | BIDI_MASK_IGNORED )))
#define BIDI_IS_JOIN_SKIPPED(p) ((p) & (BIDI_MASK_TRANSPARENT | BIDI_MASK_IGNORED))
#define BIDI_JOINS_PRECEDING_MASK(level) (BIDI_LEVEL_IS_RTL (level)?BIDI_MASK_JOINS_RIGHT:BIDI_MASK_JOINS_LEFT)
#define BIDI_JOINS_FOLLOWING_MASK(level) (BIDI_LEVEL_IS_RTL (level) ? BIDI_MASK_JOINS_LEFT:BIDI_MASK_JOINS_RIGHT)

#define BIDI_IS_EXPLICIT_OR_SEPARATOR_OR_BN_OR_WS(p) ((p) & (BIDI_MASK_EXPLICIT | BIDI_MASK_SEPARATOR \
	| BIDI_MASK_BN | BIDI_MASK_WS))
#define BIDI_IS_EXPLICIT_OR_BN_OR_NSM(p) \
	((p) & (BIDI_MASK_EXPLICIT | BIDI_MASK_BN | BIDI_MASK_NSM))
#define BIDI_IS_ES_OR_CS(p) ((p) & (BIDI_MASK_ES | BIDI_MASK_CS))
#define BIDI_EXPLICIT_TO_OVERRIDE_DIR(p) \
	(BIDI_IS_OVERRIDE(p) ? BIDI_LEVEL_TO_DIR(BIDI_DIR_TO_LEVEL(p)) : BidiDefs::CHAR_TYPE_ON)
#define BIDI_CHANGE_NUMBER_TO_RTL(p) (BIDI_IS_NUMBER(p) ? BidiDefs::CHAR_TYPE_RTL : (p))
#define BIDI_IS_EXPLICIT_OR_BN(p) ((p) & (BIDI_MASK_EXPLICIT | BIDI_MASK_BN))
#define BIDI_IS_EXPLICIT_OR_BN_OR_WS(p) ((p) & (BIDI_MASK_EXPLICIT | BIDI_MASK_BN | BIDI_MASK_WS))
#define BIDI_EMBEDDING_DIRECTION(link) BIDI_LEVEL_TO_DIR(link->level)


#define swap(a,b) void *t; (t) = (a); (a) = (b); (b) = (t);
#define for_run_list(x,list) \
	for ((x) = (list)->next; (x)->type != BidiDefs::CHAR_TYPE_SENTINEL; (x) = (x)->next)
#define merge_run_lists(a,b) swap((a)->prev->next, (b)->prev->next); swap((a)->prev, (b)->prev);
#define delete_run(x) (x)->prev->next = (x)->next; (x)->next->prev = (x)->prev;
#define insert_run_before(x, list) (x)->prev = (list)->prev; (list)->prev->next = (x); \
	(x)->next = (list); (list)->prev = (x);
#define move_run_before(x, list) if ((x)->prev) { delete_run(x); } insert_run_before((x), (list));

void Bidi::new_input(const String& str, bool shape_arabic, bool shape_mirroring) {
  m_max_level=0;
  m_base_level=0;
  m_base_dir=BIDI_LEVEL_TO_DIR (m_base_level);
  
  if (!str.empty())
  {
    static const BidiDefs::BidiCharType linear_enum_to_types_enum[] = {
      BidiDefs::CHAR_TYPE_LTR,
      BidiDefs::CHAR_TYPE_RTL,
      BidiDefs::CHAR_TYPE_AL,
      BidiDefs::CHAR_TYPE_EN,
      BidiDefs::CHAR_TYPE_AN,
      BidiDefs::CHAR_TYPE_ES,
      BidiDefs::CHAR_TYPE_ET,
      BidiDefs::CHAR_TYPE_CS,
      BidiDefs::CHAR_TYPE_NSM,
      BidiDefs::CHAR_TYPE_BN,
      BidiDefs::CHAR_TYPE_BS,
      BidiDefs::CHAR_TYPE_SS,
      BidiDefs::CHAR_TYPE_WS,
      BidiDefs::CHAR_TYPE_ON,
      BidiDefs::CHAR_TYPE_LRE,
      BidiDefs::CHAR_TYPE_RLE,
      BidiDefs::CHAR_TYPE_LRO,
      BidiDefs::CHAR_TYPE_RLO,
      BidiDefs::CHAR_TYPE_PDF,
    };
    static const BidiDefs::BidiJoiningType linear_enum_to_join_enum[] = {
      BidiDefs::ARABIC_JOIN_NUN,
      BidiDefs::ARABIC_JOIN_RIGHT,
      BidiDefs::ARABIC_JOIN_DUAL,
      BidiDefs::ARABIC_JOIN_CAUSING,
      BidiDefs::ARABIC_JOIN_TRANSPARENT,
      BidiDefs::ARABIC_JOIN_LEFT,
      BidiDefs::ARABIC_JOIN_IGNORED,
    };
    int i;
    BidiRunP *main_run_list = NULL, *explicits_list = NULL, *i_run;
    resize(str.length());
    for (i=0 ; i<str.length() ; i++) {
      BidiChar ch = {
	//input_char
	str[i],
	//visual_char
	str[i],
	//arabic_props
	shape_arabic?linear_enum_to_join_enum[BIDI_GET_JOINING_TYPE (str[i])]
		    :BidiDefs::ARABIC_JOIN_IGNORED,
	//bidi_char_type
	linear_enum_to_types_enum[BIDI_GET_CHAR_TYPE (str[i])],
	//embedding_level
	0,
	//visual_index
	i,
      };
      set(i,ch);
    }
    
    //move pointer to first bidichar
    BidiChar *dst = &operator[](0);
    
    m_max_level=0;
    explicits_list = new_run(BIDI_SENTINEL,BIDI_SENTINEL,
				   BidiDefs::CHAR_TYPE_SENTINEL,
				   BIDI_SENTINEL,NULL,NULL);
    explicits_list->prev=explicits_list->next=explicits_list;
    
    /* Fill main_run_list */
    {
      BidiRunP *last_run = new_run(BIDI_SENTINEL,BIDI_SENTINEL,
				   BidiDefs::CHAR_TYPE_SENTINEL,
				   BIDI_SENTINEL,NULL,NULL);
      last_run->prev=last_run->next=last_run;
      main_run_list=last_run;
      for (i = 0; i < size(); i++)
      {
	if (dst[i].bidi_char_type != last_run->type)
	{
	  BidiRunP *run = new_run(i,0,dst[i].bidi_char_type,0,last_run,NULL);
	  run->next = run;
	  last_run->next = run;
	  last_run->length = i - last_run->position;
	  last_run = run;
	}
      }
      last_run->length = size() - last_run->position;
      last_run->next = main_run_list;
    }
    
    /* P2. P3. Search for first strong character and use its direction as base direction */
    for_run_list(i_run,main_run_list)
    {
      if(BIDI_IS_LETTER(i_run->type))
      {
	m_base_level = BIDI_DIR_TO_LEVEL (i_run->type);
	m_base_dir = BIDI_LEVEL_TO_DIR (m_base_level);
	break;
      }
    }

    /* Explicit Levels and Directions */
    {
      int level, new_level, i, j, over_pushed, first_interval, stack_size;
      BidiDefs::BidiCharType c_override, new_override;
      
      /* X1. Begin by setting the current embedding level to the paragraph
	embedding level. Set the directional override status to neutral.
	Process each character iteratively, applying rules X2 through X9.
	Only embedding levels from 0 to 61 are valid in this phase. */
      level = m_base_level;
      c_override = BidiDefs::CHAR_TYPE_ON;
      Vector<Status> status_stack;
      
      for_run_list(i_run,main_run_list)
      {
	BidiDefs::BidiCharType this_type = i_run->type;
	if (BIDI_IS_EXPLICIT_OR_BN (this_type))
	{
	  if (BIDI_IS_STRONG (this_type))
	  {
	    /* 1. Explicit Embeddings */
	    /*   X2. With each RLE, compute the least greater odd
		embedding level. */
	    /*   X3. With each LRE, compute the least greater even
		embedding level. */
	    /* 2. Explicit Overrides */
	    /*   X4. With each RLO, compute the least greater odd
		embedding level. */
	    /*   X5. With each LRO, compute the least greater even
		embedding level. */
	    new_override = BIDI_EXPLICIT_TO_OVERRIDE_DIR (this_type);
	    for (j = i_run->length; j; j--)
	    {
	      new_level = ((level + BIDI_DIR_TO_LEVEL (this_type) + 2) & ~1) - BIDI_DIR_TO_LEVEL (this_type);
	      if (new_level <= BIDI_MAX_EXPLICIT_LEVEL)
	      {
		if (level == BIDI_MAX_EXPLICIT_LEVEL - 1)
		  first_interval = over_pushed;
		Status st = { level, c_override };
		status_stack.push_back(st);
		level = new_level;
		c_override = new_override;
	      } else over_pushed++; 
	    }
	  }
	  else if (this_type == BidiDefs::CHAR_TYPE_PDF)
	  {
	    /* 3. Terminating Embeddings and overrides */
	    /*   X7. With each PDF, determine the matching embedding or
		override code. */
	    for (j = i_run->length; j; j--)
	    {
	      stack_size=status_stack.size();
	      if (stack_size)
	      {
		if (over_pushed > first_interval)
		  over_pushed--;
		else
		  {
		    if (over_pushed == first_interval)
		      first_interval = 0;
		    stack_size--;
		    level = status_stack[stack_size].level;
		    c_override = status_stack[stack_size].c_override;
		    status_stack.remove(stack_size);
		  }
	      }
	    }
	  }

	  /* X9. Remove all RLE, LRE, RLO, LRO, PDF, and BN codes. */
	  /* Remove element and add it to explicits_list */
	  i_run->level = 0;
	  move_run_before(i_run,explicits_list);
	  delete_run(i_run);
	}
	else if (this_type == BidiDefs::CHAR_TYPE_BS)
	{
	  /* X8. All explicit directional embeddings and overrides are
	      completely terminated at the end of each paragraph. Paragraph
	      separators are not included in the embedding. */
	  break;
	}
	else
	{
	  /* X6. For all types besides RLE, LRE, RLO, LRO, and PDF:
	      a. Set the level of the current character to the current
	      embedding level.
	      b. Whenever the directional override status is not neutral,
	      reset the current character type to the directional override
	      status. */
	  i_run->level = level;
	  if (!BIDI_IS_NEUTRAL (c_override))
	    i_run->type = c_override;
	}
      }
      
      /* Implementing X8. It has no effect on a single paragraph! */
      level = m_base_level;
      c_override = BidiDefs::CHAR_TYPE_ON;
      stack_size = 0;
      over_pushed = 0;
      
      status_stack.~Vector();
    }
    
    /* X10. The remaining rules are applied to each run of characters at the
      same level. For each run, determine the start-of-level-run (sor) and
      end-of-level-run (eor) type, either L or R. This depends on the
      higher of the two levels on either side of the boundary (at the start
      or end of the paragraph, the level of the 'other' run is the base
      embedding level). If the higher level is odd, the type is R, otherwise
      it is L. */
    
    compact_run_list(main_run_list);
    
    /* 4. Resolving weak types */
    {
      BidiDefs::BidiCharType last_strong, prev_type_orig;
      bool w4;

      last_strong = m_base_dir;

      for_run_list(i_run,main_run_list)
      {
	BidiDefs::BidiCharType prev_type, this_type, next_type;

	prev_type = i_run->prev->type;
	this_type = i_run->type;
	next_type = i_run->next->type;

	if (BIDI_IS_STRONG (prev_type))
	  last_strong = prev_type;

	/* W1. NSM
	  Examine each non-spacing mark (NSM) in the level run, and change the
	  type of the NSM to the type of the previous character. If the NSM
	  is at the start of the level run, it will get the type of sor. */
	if (this_type == BidiDefs::CHAR_TYPE_NSM)
	{
	  if (i_run->prev->level == i_run->level)
	    i_run = merge_run_with_prev (i_run);
	  else
	   i_run->type = prev_type;
	  if (prev_type == next_type && i_run->level == i_run->next->level)
	    {
	      i_run = merge_run_with_prev (i_run->next);
	    }
	  continue;
	}

	/* W2: European numbers. */
	if (this_type == BidiDefs::CHAR_TYPE_EN && last_strong == BidiDefs::CHAR_TYPE_AL)
	{
	  i_run->type = BidiDefs::CHAR_TYPE_AN;
	  if (next_type == BidiDefs::CHAR_TYPE_NSM)
	    i_run->next->type = BidiDefs::CHAR_TYPE_AN;
	}
      }


      last_strong = m_base_dir;
      w4 = true;
      prev_type_orig = BidiDefs::CHAR_TYPE_ON;

      for_run_list(i_run,main_run_list)
      {
	BidiDefs::BidiCharType prev_type, this_type, next_type;

	prev_type = i_run->prev->type;
	this_type = i_run->type;
	next_type = i_run->next->type;

	if (BIDI_IS_STRONG (prev_type))
	  last_strong = prev_type;

	/* W3: Change ALs to R. */
	if (this_type == BidiDefs::CHAR_TYPE_AL)
	{
	  i_run->type = BidiDefs::CHAR_TYPE_RTL;
	  w4 = true;
	  prev_type_orig = BidiDefs::CHAR_TYPE_ON;
	  continue;
	}

	/* W4. A single european separator changes to a european number.*/
	if (w4
	    && i_run->length == 1 && BIDI_IS_ES_OR_CS (this_type)
	    && BIDI_IS_NUMBER (prev_type_orig)
	    && prev_type_orig == next_type
	    && (prev_type_orig == BidiDefs::CHAR_TYPE_EN
		|| this_type == BidiDefs::CHAR_TYPE_CS))
	  {
	    i_run->type = prev_type;
	    this_type = i_run->type;
	  }
	w4 = true;

	/* W5. A sequence of European terminators adjacent to European
	  numbers changes to All European numbers. */
	if (this_type == BidiDefs::CHAR_TYPE_ET
	    && (prev_type_orig == BidiDefs::CHAR_TYPE_EN
		|| next_type == BidiDefs::CHAR_TYPE_EN))
	  {
	    i_run->type = BidiDefs::CHAR_TYPE_EN;
	    w4 = false;
	    this_type = i_run->type;
	  }

	/* W6. Otherwise change separators and terminators to other neutral. */
	if (BIDI_IS_NUMBER_SEPARATOR_OR_TERMINATOR (this_type))
	  i_run->type = BidiDefs::CHAR_TYPE_ON;

	/* W7. Change european numbers to L. */
	if (this_type == BidiDefs::CHAR_TYPE_EN && last_strong == BidiDefs::CHAR_TYPE_LTR)
	  {
	    i_run->type = BidiDefs::CHAR_TYPE_LTR;
	    prev_type_orig = (i_run->level == i_run->next->level ?
			      BidiDefs::CHAR_TYPE_EN : BidiDefs::CHAR_TYPE_ON);
	  }
	else
	  prev_type_orig = i_run->next->prev->type;
      }
    }
    
    compact_run_list_neutrals(main_run_list);
    
    /* 5. Resolving Neutral Types */
    {
      /* N1. and N2.*/
      for_run_list (i_run, main_run_list)
      {
	BidiDefs::BidiCharType prev_type, this_type, next_type;

	this_type = BIDI_CHANGE_NUMBER_TO_RTL (i_run->type);
	prev_type = BIDI_CHANGE_NUMBER_TO_RTL (i_run->prev->type);
	next_type = BIDI_CHANGE_NUMBER_TO_RTL (i_run->next->type);

	if (BIDI_IS_NEUTRAL (this_type))
	  i_run->type = (prev_type == next_type) ?
	    /* N1. */ prev_type :
	    /* N2. */ BIDI_EMBEDDING_DIRECTION (i_run);
      }
    }

    compact_run_list (main_run_list);
    
    /* 6. Resolving implicit levels */
    {
      m_max_level = m_base_level;

      for_run_list (i_run, main_run_list)
      {
	BidiDefs::BidiCharType this_type;
	int level;

	this_type = i_run->type;
	level = i_run->level;

	if (BIDI_IS_NUMBER (this_type))
	  i_run->level = (level + 2) & ~1;
	else
	  i_run->level =
	    level +
	    (BIDI_LEVEL_IS_RTL (level) ^ BIDI_DIR_TO_LEVEL (this_type));

	if (i_run->level > m_max_level)
	  m_max_level = i_run->level;
      }
    }

    compact_run_list (main_run_list);
    
    /* Reinsert the explicit codes & BN's that are already removed, from the
      explicits_list to main_run_list. */
    if (explicits_list->next != explicits_list)
    {
      shadow_run_list (main_run_list, explicits_list, true);
      explicits_list = NULL;
      i_run = main_run_list->next;
      if (i_run != main_run_list && i_run->level == BIDI_SENTINEL)
	i_run->level = m_base_level;
      for_run_list (i_run, main_run_list) if (i_run->level == BIDI_SENTINEL)
	i_run->level = i_run->prev->level;
    }
    
    /* L1. Reset the embedding levels of some chars:
	1. segment separators,
	2. paragraph separators,
	3. any sequence of whitespace characters. */
    {
      int j, state, pos;
      BidiDefs::BidiCharType char_type;
      BidiRunP *p, *q, *list;

      list = new_run(BIDI_SENTINEL,BIDI_SENTINEL,
				   BidiDefs::CHAR_TYPE_SENTINEL,
				   BIDI_SENTINEL,NULL,NULL);
      list->prev=list->next=list;
      q = list;
      state = 1;
      pos = size() - 1;
      for (j = size() - 1; j >= -1; j--)
      {
	if (j >= 0)
	  char_type = dst[j].bidi_char_type;
	else
	  char_type = BidiDefs::CHAR_TYPE_ON;
	if (!state && BIDI_IS_SEPARATOR (char_type))
	{
	  state = 1;
	  pos = j;
	}
	else if (state && !BIDI_IS_EXPLICIT_OR_SEPARATOR_OR_BN_OR_WS(char_type))
	{
	  state = 0;
	  p = new_run (j + 1, pos - j, m_base_dir, m_base_level, NULL, NULL);
	  move_run_before (p, q);
	  q = p;
	}
      }
      shadow_run_list (main_run_list, list, false);
    }
    
    /* Fill embedding level */
    {
      int pos = 0;
      for_run_list (i_run, main_run_list)
      {
	int l, level = i_run->level;
	for (l = i_run->length; l; l--)
	  dst[pos++].embedding_level = level;
      }
    }
    
    /* Create run vector */
    {
      int runs_count=0;
      for_run_list (i_run, main_run_list) runs_count++;
      m_runs.resize(runs_count);
      runs_count=0;
      for_run_list (i_run, main_run_list)
      {
	BidiRun run = {
	  i_run->position,
	  i_run->length,
	  i_run->type,
	  i_run->level,
	};
	m_runs.set(runs_count,run);
	runs_count++;
      }
    }
    
    if (main_run_list)  free_run_list (main_run_list);
    if (explicits_list) free_run_list (explicits_list);
    
    //move pointer to first bidichar
    dst = &operator[](0);
    
    /* L4. Mirror all characters that are in odd levels and have mirrors. */
    if (shape_mirroring)
    {
      for (i = size() - 1; i >= 0; i--)
	if (BIDI_LEVEL_IS_RTL (dst[i].embedding_level))
	{
	  CharType mirrored_ch;
	  mirrored_ch = BIDI_GET_MIRRORING (dst[i].input_char);
	  if (dst[i].input_char != mirrored_ch)
	    dst[i].input_char = mirrored_ch;
	}
    }
    
    /* Shape arabic glyphs. */
    if (shape_arabic)
    {
      /* Joining arabic */
      {
	int saved = 0;
	int saved_level = BIDI_SENTINEL;
	bool saved_shapes = false;
	int saved_joins_following_mask = BidiDefs::ARABIC_JOIN_NUN;
	bool joins = false;

	for (i = 0; i < size(); i++)
	  if (!BIDI_IS_JOINING_TYPE_G (dst[i].arabic_props))
	  {
	    bool disjoin = false;
	    bool shapes = BIDI_ARAB_SHAPES (dst[i].arabic_props);
	    int level = BIDI_IS_EXPLICIT_OR_BN (dst[i].bidi_char_type)?BIDI_SENTINEL:dst[i].embedding_level;

	    if (joins && !BIDI_LEVELS_MATCH (saved_level, level))
	    {
	      disjoin = true;
	      joins = false;
	    }

	    if (!BIDI_IS_JOIN_SKIPPED (dst[i].arabic_props))
	    {
	      const int joins_preceding_mask = BIDI_JOINS_PRECEDING_MASK (level);
	      if (!joins)
	      {
		if (shapes)
		  BIDI_UNSET_BITS (dst[i].arabic_props, joins_preceding_mask);
	      }
	      else if (!BIDI_TEST_BITS (dst[i].arabic_props, joins_preceding_mask))
	      {
		disjoin = true;
	      }
	    }

	    if (disjoin && saved_shapes)
	      BIDI_UNSET_BITS (dst[saved].arabic_props, saved_joins_following_mask);

	    if (!BIDI_IS_JOIN_SKIPPED (dst[i].arabic_props))
	    {
	      saved = i;
	      saved_level = level;
	      saved_shapes = shapes;
	      saved_joins_following_mask = BIDI_JOINS_FOLLOWING_MASK (level);
	      joins = BIDI_TEST_BITS (dst[i].arabic_props, saved_joins_following_mask);
	    }
	  }
	  if ((joins) && saved_shapes)
	    BIDI_UNSET_BITS (dst[saved].arabic_props, saved_joins_following_mask);
      }
      
      for (i = 0; i < size(); i++)
	if (BIDI_ARAB_SHAPES(dst[i].arabic_props))
	  dst[i].visual_char = BIDI_GET_ARABIC_SHAPE(dst[i].input_char, BIDI_JOIN_SHAPE (dst[i].arabic_props));
      
      int table_size=sizeof(mandatory_liga_table)/sizeof(mandatory_liga_table[0]);
      for (i = 0; i < size() - 1; i++) {
	CharType c;
	if ( dst[i].visual_char < mandatory_liga_table[0][0] || 
	      dst[i].visual_char > mandatory_liga_table[table_size-1][0] ) c=0;
	else
	{
	  for (int j=0;j<table_size;j++)
	  {
	    if (mandatory_liga_table[j][0]==dst[i].visual_char && 
		mandatory_liga_table[j][1]==dst[i+1].visual_char) {
	      c = mandatory_liga_table[j][2];
	      break;
	    }
	  }
	}
	if (BIDI_LEVEL_IS_RTL(dst[i].embedding_level) &&
	    dst[i].embedding_level == dst[i+1].embedding_level && c )
	  {
	    dst[i].visual_char = BIDI_CHAR_FILL;
	    BIDI_SET_BITS (dst[i].arabic_props, BIDI_MASK_LIGATURED);
	    dst[i+1].visual_char = c;
	  }
      }
    }

    /* Reorder each line */
    {
      int start = -1;
      for (i=0 ; i < size() ; i++) {
	if (dst[i].input_char!=0x000A && start==-1) start = i;
	if (dst[i].input_char==0x000A && start>-1) {
	  reorder_line(start,i-start);
	  start = -1;
	}
      }
      if (start>-1) reorder_line(start,size()-start);
    }

  }
}

void Bidi::new_input(const String& str) {
  new_input(str,true,true);
}

Bidi::Bidi(const String& str, bool shape_arabic, bool shape_mirroring) {
  new_input(str,shape_arabic,shape_mirroring);
}

Bidi::Bidi(const String& str) {
  new_input(str);
}



void Bidi::bidi_chars_reverse (int start, int len) {
  int i;
  BidiChar *dst = ptr();
  for (i = 0; i < len / 2; i++)
  {
    CharType tmp = dst[i + start].visual_char;
    dst[i + start].visual_char = dst[len - 1 - i + start].visual_char;
    dst[len - 1 - i + start].visual_char = tmp;
    int tmp2 = dst[i + start].visual_index;
    dst[i + start].visual_index = dst[len - 1 - i + start].visual_index;
    dst[len - 1 - i + start].visual_index = tmp2;
  }
}

void Bidi::free_run_list(BidiRunP *run_list) {
  BidiRunP *i_run;
  i_run = run_list;
  i_run->prev->next = NULL;
  while (i_run)
  {
    BidiRunP *run;

    run = i_run;
    i_run = i_run->next;
    delete run;
  };
}

Bidi::BidiRunP *Bidi::new_run(int position, int length, BidiDefs::BidiCharType type,
			 int level, BidiRunP *prev, BidiRunP *next) {
  BidiRunP *run = new BidiRunP;
  run->position=position;
  run->length=length;
  run->type=type;
  run->level=level;
  run->prev=prev;
  run->next=next;
  return run;
}

void Bidi::shadow_run_list(BidiRunP *base, BidiRunP *over, bool preserve_length) {
  BidiRunP *p = base, *q, *r, *s, *t;
  int pos = 0, pos2;

  for_run_list (q, over)
  {
    if (!q->length || q->position < pos) continue;
    pos = q->position;
    while (p->next->type != BidiDefs::CHAR_TYPE_SENTINEL && p->next->position <= pos)
      p = p->next;
    pos2 = pos + q->length;
    r = p;
    while (r->next->type != BidiDefs::CHAR_TYPE_SENTINEL && r->next->position < pos2)
      r = r->next;
    if (preserve_length)
      r->length += q->length;
    if (p == r)
    {
      if (p->position + p->length > pos2)
      {
	r = new_run (pos2, p->position + p->length - pos2, p->type, p->level, NULL, p->next);
	p->next->prev = r;
      }
      else
	r = r->next;

      if (p->position + p->length >= pos)
      {
	if (p->position < pos)
	  p->length = pos - p->position;
	else
	  {
	    t = p;
	    p = p->prev;
	    delete t;
	  }
      }
    }
    else
    {
      if (p->position + p->length >= pos)
      {
	if (p->position < pos)
	  p->length = pos - p->position;
	else
	  p = p->prev;
      }

      if (r->position + r->length > pos2)
      {
	r->length = r->position + r->length - pos2;
	r->position = pos2;
      }
      else
	r = r->next;

      for (s = p->next; s != r;)
      {
	t = s;
	s = s->next;
	delete t;
      }
    }
    t = q;
    q = q->prev;
    delete_run (t);
    p->next = t;
    t->prev = p;
    t->next = r;
    r->prev = t;
  }
  
  free_run_list (over);
}

void Bidi::compact_run_list_neutrals(BidiRunP *list) {
  if (list->next)
    for_run_list (list,list)
      if ( list->prev->level == list->level  && ( (
	  list->prev->type == list->type || 
	  ( BIDI_IS_NEUTRAL (list->prev->type) && BIDI_IS_NEUTRAL (list->type) )
	  ) ) )
	list = merge_run_with_prev (list);
}

void Bidi::compact_run_list(BidiRunP *list) {
  if (list->next)
    for_run_list (list,list)
	if (list->prev->type == list->type
	    && list->prev->level == list->level)
	  list = merge_run_with_prev (list);
}

Bidi::BidiRunP *Bidi::merge_run_with_prev(BidiRunP * second) {
  BidiRunP *first;
  first = second->prev;
  first->next = second->next;
  first->next->prev = first;
  first->length += second->length;
  delete second;
  return first;
}

void Bidi::reorder_line(int start, int length) {
  int max_level = 0;
  BidiChar *dst = &operator[](0);

  /* L1. Reset the embedding levels of some chars:
      4. any sequence of white space characters at the end of the line. */
  {
    int i;
    for (i = start + length - 1; i >= start && BIDI_IS_EXPLICIT_OR_BN_OR_WS (dst[i].bidi_char_type) ; i--)
      dst[i].embedding_level = BIDI_DIR_TO_LEVEL (m_base_dir);
  }

  /* 7. Reordering resolved levels */
  {
    int level, i;
    /* Reorder both the outstring and the order array */
    {
      if (BIDI_REORDER_NSM)
      {
	/* L3. Reorder NSMs. */
	for (i = start + length - 1; i >= start; i--)
	  if (BIDI_LEVEL_IS_RTL (dst[i].embedding_level)
	      && dst[i].bidi_char_type == BidiDefs::CHAR_TYPE_NSM)
	  {
	    int seq_end = i;
	    level = dst[i].embedding_level;

	    for (i--; i >= start && BIDI_IS_EXPLICIT_OR_BN_OR_NSM (dst[i].bidi_char_type)
		  && dst[i].embedding_level == level; i--)
	      ;
	    if (i < start || dst[i].embedding_level != level) i++;
	    bidi_chars_reverse (i, seq_end - i + 1);
	  }
      }
      
      /* Find max_level of the line.  */
      for (i = start + length - 1; i > start; i--)
	if (dst[i].embedding_level > max_level)
	  max_level = dst[i].embedding_level;
      
      /* L2. Reorder. */
      for (level = max_level; level > 0; level--)
	for (i = start + length - 1; i >= start; i--)
	  if (dst[i].embedding_level >= level)
	  {
	    /* Find all stretches that are >= level_idx */
	    int seq_end = i;
	    for (i--; i >= start && dst[i].embedding_level >= level; i--)
	      ;
	    bidi_chars_reverse (i + 1, seq_end - i);
	  }
    }
  }

}



String Bidi::get_input_string() const {
  if(empty()) return String();
  const BidiChar *dst = ptr();
  CharType *chars = new CharType[size()+1];
  for (int i = 0; i < size(); i ++) {
    chars[i]=dst[i].input_char;
  }
  chars[size()]=0;
  return String(chars);
}

String Bidi::get_visual_string() const {
  if(empty()) return String();
  const BidiChar *dst = ptr();
  CharType *chars = new CharType[size()+1];
  for (int i = 0; i < size() ; i ++) {
    chars[i]=dst[i].visual_char;
  }
  chars[size()]=0;
  return String(chars);
}

String Bidi::bidi_visual_string(const String& str) {
  Bidi *b = new Bidi(str);
  return b->get_visual_string();
}

uint32_t Bidi::hash() const {
	
	/* simple djb2 hashing */
	uint32_t hashv = 5381;
	if(empty()) return hashv;
	const BidiChar * chr = &operator[](0);
	CharType c;
	
	while ((c = (*chr++).visual_char))
		hashv = ((hashv << 5) + hashv) + c; /* hash * 33 + c */
	
	return hashv;
}
