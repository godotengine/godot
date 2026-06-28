// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2013, SIL International, All rights reserved.

#pragma once

// numbers are explicitly assigned for future proofing

namespace graphite2
{

class Error
{
public:
    Error() : _e(0) {};
    operator bool() { return (_e != 0); }
    int error() { return _e; }
    void error(int e) { _e = e; }
    bool test(bool pr, int err) { return (_e = pr ? err : 0); }

private:
    int _e;
};

enum errcontext {
    EC_READGLYPHS = 1,      // while reading glyphs
    EC_READSILF = 2,        // in Silf table
    EC_ASILF = 3,           // in Silf %d
    EC_APASS = 4,           // in Silf %d, pass %d
    EC_PASSCCODE = 5,       // in pass constraint code for Silf %d, pass %d
    EC_ARULE = 6,           // in Silf %d, pass %d, rule %d
    EC_ASTARTS = 7,         // in Silf %d, pass %d, start state %d
    EC_ATRANS = 8,          // in Silf %d, pass %d, fsm state %d
    EC_ARULEMAP = 9         // in Silf %d, pass %d, state %d
};

enum error {
    E_OUTOFMEM = 1,         // Out of memory
    E_NOGLYPHS = 2,         // There are no glyphs in the font
    E_BADUPEM = 3,          // The units per em for the font is bad (0)
    E_BADCMAP = 4,          // The font does not contain any useful cmaps
    E_NOSILF = 5,           // Missing Silf table
    E_TOOOLD = 6,           // Silf table version is too old
    E_BADSIZE = 7,          // context object has the wrong structural size
// Silf Subtable Errors take a Silf subtable number * 256 in the context
    E_BADMAXGLYPH = 8,      // Silf max glyph id is too high
    E_BADNUMJUSTS = 9,      // Number of Silf justification blocks is too high
    E_BADENDJUSTS = 10,     // Silf justification blocks take too much of the Silf table space
    E_BADCRITFEATURES = 11, // Critical features section in a Silf table is too big
    E_BADSCRIPTTAGS = 12,   // Silf script tags area is too big
    E_BADAPSEUDO = 13,      // The pseudo glyph attribute number is too high
    E_BADABREAK = 14,       // The linebreak glyph attribute number is too high
    E_BADABIDI = 15,        // The bidi glyph attribute number is too high
    E_BADAMIRROR = 16,      // The mirrored glyph attribute number is too high
    E_BADNUMPASSES = 17,    // The number of passes is > 128
    E_BADPASSESSTART = 18,  // The Silf table is too small to hold any passes
    E_BADPASSBOUND = 19,    // The positioning pass number is too low or the substitution pass number is too high
    E_BADPPASS = 20,        // The positioning pass number is too high
    E_BADSPASS = 21,        // the substitution pass number is too high
    E_BADJPASSBOUND = 22,   // the justification pass must be higher than the positioning pass
    E_BADJPASS = 23,        // the justification pass is too high
    E_BADALIG = 24,         // the number of initial ligature component glyph attributes is too high
    E_BADBPASS = 25,        // the bidi pass number is specified and is either too high or too low
    E_BADNUMPSEUDO = 26,    // The number of pseudo glyphs is too high
    E_BADCLASSSIZE = 27,    // The size of the classes block is bad
    E_TOOMANYLINEAR = 28,   // The number of linear classes in the silf table is too high
    E_CLASSESTOOBIG = 29,   // There are too many classes for the space allocated in the Silf subtable
    E_MISALIGNEDCLASSES = 30,   // The class offsets in the class table don't line up with the number of classes
    E_HIGHCLASSOFFSET = 31, // The class offsets point out of the class table
    E_BADCLASSOFFSET = 32,  // A class offset is less than one following it
    E_BADCLASSLOOKUPINFO = 33,  // The search header info for a non-linear class has wrong values in it
// Pass subtable errors. Context has pass number * 65536
    E_BADPASSSTART = 34,    // The start offset for a particular pass is bad
    E_BADPASSEND = 35,      // The end offset for a particular pass is bad
    E_BADPASSLENGTH = 36,   // The length of the pass is too small
    E_BADNUMTRANS = 37,     // The number of transition states in the fsm is bad
    E_BADNUMSUCCESS = 38,   // The number of success states in the fsm is bad
    E_BADNUMSTATES = 39,    // The number of states in the fsm is bad
    E_NORANGES = 40,        // There are no columns in the fsm
    E_BADRULEMAPLEN = 41,   // The size of the success state to rule mapping is bad
    E_BADCTXTLENBOUNDS = 42,    // The precontext maximum is greater than its minimum
    E_BADCTXTLENS = 43,     // The lists of rule lengths or pre context lengths is bad
    E_BADPASSCCODEPTR = 44, // The pass constraint code position does not align with where the forward reference says it should be
    E_BADRULECCODEPTR = 45, // The rule constraint code position does not align with where the forward reference says it should be
    E_BADCCODELEN = 46,     // Bad rule/pass constraint code length
    E_BADACTIONCODEPTR = 47,    // The action code position does not align with where the forward reference says it should be
    E_MUTABLECCODE = 48,    // Constraint code edits slots. It shouldn't.
    E_BADSTATE = 49,        // Bad state transition referencing an illegal state
    E_BADRULEMAPPING = 50,  // The structure of the rule mapping is bad
    E_BADRANGE = 51,        // Bad column range structure including a glyph in more than one column
    E_BADRULENUM = 52,      // A reference to a rule is out of range (too high)
    E_BADACOLLISION = 53,   // Bad Silf table collision attribute number (too high)
    E_BADEMPTYPASS = 54,    // Can't have empty passes (no rules) except for collision passes
    E_BADSILFVERSION = 55,  // The Silf table has a bad version (probably too high)
    E_BADCOLLISIONPASS = 56,    // Collision flags set on a non positioning pass
    E_BADNUMCOLUMNS = 57,   // Arbitrarily limit number of columns in fsm
// Code errors
    E_CODEFAILURE = 60,     // Base code error. The following subcodes must align with Machine::Code::status_t in Code.h
    E_CODEALLOC = 61,       // Out of memory
    E_INVALIDOPCODE = 62,   // Invalid op code
    E_UNIMPOPCODE = 63,     // Unimplemented op code encountered
    E_OUTOFRANGECODE = 64,  // Code argument out of range
    E_BADJUMPCODE = 65,     // Code jumps past end of op codes
    E_CODEBADARGS = 66,     // Code arguments exhausted
    E_CODENORETURN = 67,    // Missing return type op code at end of code
    E_CODENESTEDCTXT = 68,  // Nested context encountered in code
// Compression errors
    E_BADSCHEME = 69,
    E_SHRINKERFAILED = 70,
};

}
