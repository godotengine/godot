/*************************************************************************/
/*  bidi.h                                                               */
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

#ifndef H
#define H

/**
	@author Masoud BaniHashemian <masoudbh3@gmail.com>
*/

#include "ustring.h"

#define BIDI_REORDER_NSM	1

#define BIDI_MASK_RTL		0x00000001L	/* Is right to left */
#define BIDI_MASK_ARABIC	0x00000002L	/* Is arabic */
/* Each char can be only one of the three following. */
#define BIDI_MASK_STRONG	0x00000010L	/* Is strong */
#define BIDI_MASK_WEAK		0x00000020L	/* Is weak */
#define BIDI_MASK_NEUTRAL	0x00000040L	/* Is neutral */
#define BIDI_MASK_SENTINEL	0x00000080L	/* Is sentinel */
/* Each char can be only one of the five following. */
#define BIDI_MASK_LETTER	0x00000100L	/* Is letter: L, R, AL */
#define BIDI_MASK_NUMBER	0x00000200L	/* Is number: EN, AN */
#define BIDI_MASK_NUMSEPTER	0x00000400L	/* Is separator or terminator: ES, ET, CS */
#define BIDI_MASK_SPACE		0x00000800L	/* Is space: BN, BS, SS, WS */
#define BIDI_MASK_EXPLICIT	0x00001000L	/* Is expilict mark: LRE, RLE, LRO, RLO, PDF */
/* Can be set only if BIDI_MASK_SPACE is also set. */
#define BIDI_MASK_SEPARATOR	0x00002000L	/* Is text separator: BS, SS */
/* Can be set only if BIDI_MASK_EXPLICIT is also set. */
#define BIDI_MASK_OVERRIDE	0x00004000L	/* Is explicit override: LRO, RLO */
#define BIDI_MASK_ES		0x00010000L
#define BIDI_MASK_ET		0x00020000L
#define BIDI_MASK_CS		0x00040000L
#define BIDI_MASK_NSM		0x00080000L
#define BIDI_MASK_BN		0x00100000L
#define BIDI_MASK_BS		0x00200000L
#define BIDI_MASK_SS		0x00400000L
#define BIDI_MASK_WS		0x00800000L
#define BIDI_MASK_JOINS_RIGHT	0x01	/* May join to right */
#define BIDI_MASK_JOINS_LEFT	0x02	/* May join to right */
#define BIDI_MASK_ARAB_SHAPES	0x04	/* May Arabic shape */
#define BIDI_MASK_TRANSPARENT	0x08	/* Is transparent */
#define BIDI_MASK_IGNORED	0x10	/* Is ignored */
#define BIDI_MASK_LIGATURED	0x20	/* Is ligatured */

namespace BidiDefs {
  
  enum BidiCharType {
    /* Left-To-Right letter */
    CHAR_TYPE_LTR=( BIDI_MASK_STRONG | BIDI_MASK_LETTER ),
    /* Right-To-Left letter */
    CHAR_TYPE_RTL=( BIDI_MASK_STRONG | BIDI_MASK_LETTER | BIDI_MASK_RTL),
    /* Arabic Letter */
    CHAR_TYPE_AL=( BIDI_MASK_STRONG | BIDI_MASK_LETTER | BIDI_MASK_RTL | BIDI_MASK_ARABIC ),
    /* European Numeral */
    CHAR_TYPE_EN=( BIDI_MASK_WEAK | BIDI_MASK_NUMBER ),
    /* Arabic Numeral */
    CHAR_TYPE_AN=( BIDI_MASK_WEAK | BIDI_MASK_NUMBER | BIDI_MASK_ARABIC ),
    /* European number Separator */
    CHAR_TYPE_ES=( BIDI_MASK_WEAK | BIDI_MASK_NUMSEPTER | BIDI_MASK_ES ),
    /* European number Terminator */
    CHAR_TYPE_ET=( BIDI_MASK_WEAK | BIDI_MASK_NUMSEPTER | BIDI_MASK_ET ),
    /* Common Separator */
    CHAR_TYPE_CS=( BIDI_MASK_WEAK | BIDI_MASK_NUMSEPTER | BIDI_MASK_CS ),
    /* Non Spacing Mark */
    CHAR_TYPE_NSM=( BIDI_MASK_WEAK | BIDI_MASK_NSM ),
    /* Boundary Neutral */
    CHAR_TYPE_BN=( BIDI_MASK_WEAK | BIDI_MASK_SPACE | BIDI_MASK_BN ),
    /* Block Separator */
    CHAR_TYPE_BS=( BIDI_MASK_NEUTRAL | BIDI_MASK_SPACE | BIDI_MASK_SEPARATOR | BIDI_MASK_BS ),
    /* Segment Separator */
    CHAR_TYPE_SS=( BIDI_MASK_NEUTRAL | BIDI_MASK_SPACE | BIDI_MASK_SEPARATOR | BIDI_MASK_SS ),
    /* WhiteSpace */
    CHAR_TYPE_WS=( BIDI_MASK_NEUTRAL | BIDI_MASK_SPACE | BIDI_MASK_WS ),
    /* Other Neutral */
    CHAR_TYPE_ON=( BIDI_MASK_NEUTRAL ),
    /* Left-to-Right Embedding */
    CHAR_TYPE_LRE=( BIDI_MASK_STRONG | BIDI_MASK_EXPLICIT),
    /* Right-to-Left Embedding */
    CHAR_TYPE_RLE=( BIDI_MASK_STRONG | BIDI_MASK_EXPLICIT | BIDI_MASK_RTL ),
    /* Left-to-Right Override */
    CHAR_TYPE_LRO=( BIDI_MASK_STRONG | BIDI_MASK_EXPLICIT | BIDI_MASK_OVERRIDE ),
    /* Right-to-Left Override */
    CHAR_TYPE_RLO=( BIDI_MASK_STRONG | BIDI_MASK_EXPLICIT | BIDI_MASK_RTL | BIDI_MASK_OVERRIDE ),
    /* Pop Directional Flag */
    CHAR_TYPE_PDF=( BIDI_MASK_WEAK | BIDI_MASK_EXPLICIT ),
    /* Don't use this */
    CHAR_TYPE_SENTINEL=( BIDI_MASK_SENTINEL ),
  };
  
  enum BidiJoiningType {
    /* nUn-joining, e.g. Full Stop */
    ARABIC_JOIN_NUN=( 0 ),
    /* Right-joining, e.g. Arabic Letter Dal */
    ARABIC_JOIN_RIGHT=( BIDI_MASK_JOINS_RIGHT | BIDI_MASK_ARAB_SHAPES ),
    /* Dual-joining, e.g. Arabic Letter Ain */
    ARABIC_JOIN_DUAL=( BIDI_MASK_JOINS_RIGHT | BIDI_MASK_JOINS_LEFT | BIDI_MASK_ARAB_SHAPES ),
    /* join-Causing, e.g. Tatweel, ZWJ */
    ARABIC_JOIN_CAUSING=( BIDI_MASK_JOINS_RIGHT | BIDI_MASK_JOINS_LEFT ),
    /* Transparent, e.g. Arabic Fatha */
    ARABIC_JOIN_TRANSPARENT=( BIDI_MASK_TRANSPARENT | BIDI_MASK_ARAB_SHAPES ),
    /* Left-joining, i.e. fictional */
    ARABIC_JOIN_LEFT=( BIDI_MASK_JOINS_LEFT | BIDI_MASK_ARAB_SHAPES ),
    /* iGnored, e.g. LRE, RLE, ZWNBSP */
    ARABIC_JOIN_IGNORED=( BIDI_MASK_IGNORED ),
  };
  
};

struct BidiChar {

  CharType input_char;
  CharType visual_char;
  int arabic_props;
  BidiDefs::BidiCharType bidi_char_type;
  int embedding_level;
  int visual_index;

};

struct BidiRun {

  int position;
  int length;
  BidiDefs::BidiCharType type;
  int level;

};

class Bidi : public Vector<BidiChar> {
  
private:
  struct BidiRunP {
    
    int position;
    int length;
    BidiDefs::BidiCharType type;
    int level;
    
    BidiRunP *prev;
    BidiRunP *next;
    
  };
  struct Status {
    int level;
    BidiDefs::BidiCharType c_override;	/* only LTR, RTL and ON are valid */
  };
  
  BidiDefs::BidiCharType m_base_dir;
  int m_base_level;
  int m_max_level;
  Vector<BidiRun> m_runs;
  
  void bidi_chars_reverse (int start, int len);
  void free_run_list(BidiRunP *list);
  BidiRunP *new_run(int position, int length, BidiDefs::BidiCharType type,
		    int level, BidiRunP *prev, BidiRunP *next);
  void shadow_run_list(BidiRunP *base, BidiRunP *over, bool preserve_length);
  void compact_run_list(BidiRunP *list);
  void compact_run_list_neutrals(BidiRunP *list);
  BidiRunP *merge_run_with_prev(BidiRunP *second);
  void reorder_line(int start, int length);
  
public:
  inline Bidi() {
    m_base_dir = BidiDefs::CHAR_TYPE_SENTINEL;
    m_base_level = 0;
    m_max_level = 0;
  }
  Bidi(const String& str, bool shape_arabic, bool shape_mirroring);
  Bidi(const String& str);
  void new_input(const String& str, bool shape_arabic, bool shape_mirroring);
  void new_input(const String& str);
  
  int get_max_level() const;
  int get_base_level() const;
  int get_base_dir() const;
  Vector<BidiRun> get_runs() const;
  
  uint32_t hash() const;
  String get_input_string() const;
  String get_visual_string() const;
  static String bidi_visual_string(const String& str);
  
};

inline Vector<BidiRun> Bidi::get_runs() const {
  return m_runs;
}

inline int Bidi::get_max_level() const {
  return m_max_level;
}

inline int Bidi::get_base_level() const {
  return m_base_level;
}

inline int Bidi::get_base_dir() const {
  return m_base_dir;
}


#endif