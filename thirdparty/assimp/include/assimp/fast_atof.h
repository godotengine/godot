#pragma once

// Copyright (C) 2002-2007 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine" and the "irrXML" project.
// For conditions of distribution and use, see copyright notice in irrlicht.h and irrXML.h

// ------------------------------------------------------------------------------------
// Original description: (Schrompf)
// Adapted to the ASSIMP library because the builtin atof indeed takes AGES to parse a
// float inside a large string. Before parsing, it does a strlen on the given point.
// Changes:
//  22nd October 08 (Aramis_acg): Added temporary cast to double, added strtoul10_64
//     to ensure long numbers are handled correctly
// ------------------------------------------------------------------------------------


#ifndef FAST_A_TO_F_H_INCLUDED
#define FAST_A_TO_F_H_INCLUDED

#include <cmath>
#include <limits>
#include <stdint.h>
#include <stdexcept>
#include <assimp/defs.h>

#include "StringComparison.h"
#include <assimp/DefaultLogger.hpp>

#ifdef _MSC_VER
#  include <stdint.h>
#else
#  include <assimp/Compiler/pstdint.h>
#endif

namespace Assimp {

const double fast_atof_table[16] =  {  // we write [16] here instead of [] to work around a swig bug
    0.0,
    0.1,
    0.01,
    0.001,
    0.0001,
    0.00001,
    0.000001,
    0.0000001,
    0.00000001,
    0.000000001,
    0.0000000001,
    0.00000000001,
    0.000000000001,
    0.0000000000001,
    0.00000000000001,
    0.000000000000001
};


// ------------------------------------------------------------------------------------
// Convert a string in decimal format to a number
// ------------------------------------------------------------------------------------
inline
unsigned int strtoul10( const char* in, const char** out=0) {
    unsigned int value = 0;

    for ( ;; ) {
        if ( *in < '0' || *in > '9' ) {
            break;
        }

        value = ( value * 10 ) + ( *in - '0' );
        ++in;
    }
    if ( out ) {
        *out = in;
    }
    return value;
}

// ------------------------------------------------------------------------------------
// Convert a string in octal format to a number
// ------------------------------------------------------------------------------------
inline
unsigned int strtoul8( const char* in, const char** out=0) {
    unsigned int value( 0 );
    for ( ;; ) {
        if ( *in < '0' || *in > '7' ) {
            break;
        }

        value = ( value << 3 ) + ( *in - '0' );
        ++in;
    }
    if ( out ) {
        *out = in;
    }
    return value;
}

// ------------------------------------------------------------------------------------
// Convert a string in hex format to a number
// ------------------------------------------------------------------------------------
inline
unsigned int strtoul16( const char* in, const char** out=0) {
    unsigned int value( 0 );
    for ( ;; ) {
        if ( *in >= '0' && *in <= '9' ) {
            value = ( value << 4u ) + ( *in - '0' );
        } else if (*in >= 'A' && *in <= 'F') {
            value = ( value << 4u ) + ( *in - 'A' ) + 10;
        } else if (*in >= 'a' && *in <= 'f') {
            value = ( value << 4u ) + ( *in - 'a' ) + 10;
        } else {
            break;
        }
        ++in;
    }
    if ( out ) {
        *out = in;
    }
    return value;
}

// ------------------------------------------------------------------------------------
// Convert just one hex digit
// Return value is UINT_MAX if the input character is not a hex digit.
// ------------------------------------------------------------------------------------
inline
unsigned int HexDigitToDecimal(char in) {
    unsigned int out( UINT_MAX );
    if ( in >= '0' && in <= '9' ) {
        out = in - '0';
    } else if ( in >= 'a' && in <= 'f' ) {
        out = 10u + in - 'a';
    } else if ( in >= 'A' && in <= 'F' ) {
        out = 10u + in - 'A';
    }

    // return value is UINT_MAX if the input is not a hex digit
    return out;
}

// ------------------------------------------------------------------------------------
// Convert a hex-encoded octet (2 characters, i.e. df or 1a).
// ------------------------------------------------------------------------------------
inline
uint8_t HexOctetToDecimal(const char* in) {
    return ((uint8_t)HexDigitToDecimal(in[0])<<4)+(uint8_t)HexDigitToDecimal(in[1]);
}

// ------------------------------------------------------------------------------------
// signed variant of strtoul10
// ------------------------------------------------------------------------------------
inline
int strtol10( const char* in, const char** out=0) {
    bool inv = (*in=='-');
    if ( inv || *in == '+' ) {
        ++in;
    }

    int value = strtoul10(in,out);
    if (inv) {
        value = -value;
    }
    return value;
}

// ------------------------------------------------------------------------------------
// Parse a C++-like integer literal - hex and oct prefixes.
// 0xNNNN - hex
// 0NNN   - oct
// NNN    - dec
// ------------------------------------------------------------------------------------
inline
unsigned int strtoul_cppstyle( const char* in, const char** out=0) {
    if ('0' == in[0]) {
        return 'x' == in[1] ? strtoul16(in+2,out) : strtoul8(in+1,out);
    }
    return strtoul10(in, out);
}

// ------------------------------------------------------------------------------------
// Special version of the function, providing higher accuracy and safety
// It is mainly used by fast_atof to prevent ugly and unwanted integer overflows.
// ------------------------------------------------------------------------------------
inline
uint64_t strtoul10_64( const char* in, const char** out=0, unsigned int* max_inout=0) {
    unsigned int cur = 0;
    uint64_t value = 0;

    if ( *in < '0' || *in > '9' ) {
        throw std::invalid_argument( std::string( "The string \"" ) + in + "\" cannot be converted into a value." );
    }

    for ( ;; ) {
        if ( *in < '0' || *in > '9' ) {
            break;
        }

        const uint64_t new_value = ( value * (uint64_t) 10 ) + ( (uint64_t) ( *in - '0' ) );

        // numeric overflow, we rely on you
        if ( new_value < value ) {
            ASSIMP_LOG_WARN_F( "Converting the string \"", in, "\" into a value resulted in overflow." );
            return 0;
        }

        value = new_value;

        ++in;
        ++cur;

        if (max_inout && *max_inout == cur) {
            if (out) { /* skip to end */
                while ( *in >= '0' && *in <= '9' ) {
                    ++in;
                }
                *out = in;
            }

            return value;
        }
    }
    if ( out ) {
        *out = in;
    }

    if ( max_inout ) {
        *max_inout = cur;
    }

    return value;
}

// ------------------------------------------------------------------------------------
// signed variant of strtoul10_64
// ------------------------------------------------------------------------------------
inline
int64_t strtol10_64(const char* in, const char** out = 0, unsigned int* max_inout = 0) {
    bool inv = (*in == '-');
    if ( inv || *in == '+' ) {
        ++in;
    }

    int64_t value = strtoul10_64(in, out, max_inout);
    if (inv) {
        value = -value;
    }
    return value;
}

// Number of relevant decimals for floating-point parsing.
#define AI_FAST_ATOF_RELAVANT_DECIMALS 15

// ------------------------------------------------------------------------------------
//! Provides a fast function for converting a string into a float,
//! about 6 times faster than atof in win32.
// If you find any bugs, please send them to me, niko (at) irrlicht3d.org.
// ------------------------------------------------------------------------------------
template<typename Real>
inline
const char* fast_atoreal_move(const char* c, Real& out, bool check_comma = true) {
    Real f = 0;

    bool inv = (*c == '-');
    if (inv || *c == '+') {
        ++c;
    }

    if ((c[0] == 'N' || c[0] == 'n') && ASSIMP_strincmp(c, "nan", 3) == 0) {
        out = std::numeric_limits<Real>::quiet_NaN();
        c += 3;
        return c;
    }

    if ((c[0] == 'I' || c[0] == 'i') && ASSIMP_strincmp(c, "inf", 3) == 0) {
        out = std::numeric_limits<Real>::infinity();
        if (inv) {
            out = -out;
        }
        c += 3;
        if ((c[0] == 'I' || c[0] == 'i') && ASSIMP_strincmp(c, "inity", 5) == 0) {
            c += 5;
        }
        return c;
     }

    if (!(c[0] >= '0' && c[0] <= '9') &&
            !((c[0] == '.' || (check_comma && c[0] == ',')) && c[1] >= '0' && c[1] <= '9')) {
        throw std::invalid_argument("Cannot parse string "
                                    "as real number: does not start with digit "
                                    "or decimal point followed by digit.");
    }

    if (*c != '.' && (! check_comma || c[0] != ',')) {
        f = static_cast<Real>( strtoul10_64 ( c, &c) );
    }

    if ((*c == '.' || (check_comma && c[0] == ',')) && c[1] >= '0' && c[1] <= '9') {
        ++c;

        // NOTE: The original implementation is highly inaccurate here. The precision of a single
        // IEEE 754 float is not high enough, everything behind the 6th digit tends to be more
        // inaccurate than it would need to be. Casting to double seems to solve the problem.
        // strtol_64 is used to prevent integer overflow.

        // Another fix: this tends to become 0 for long numbers if we don't limit the maximum
        // number of digits to be read. AI_FAST_ATOF_RELAVANT_DECIMALS can be a value between
        // 1 and 15.
        unsigned int diff = AI_FAST_ATOF_RELAVANT_DECIMALS;
        double pl = static_cast<double>( strtoul10_64 ( c, &c, &diff ));

        pl *= fast_atof_table[diff];
        f += static_cast<Real>( pl );
    }
    // For backwards compatibility: eat trailing dots, but not trailing commas.
    else if (*c == '.') {
        ++c;
    }

    // A major 'E' must be allowed. Necessary for proper reading of some DXF files.
    // Thanks to Zhao Lei to point out that this if() must be outside the if (*c == '.' ..)
    if (*c == 'e' || *c == 'E') {
        ++c;
        const bool einv = (*c=='-');
        if (einv || *c=='+') {
            ++c;
        }

        // The reason float constants are used here is that we've seen cases where compilers
        // would perform such casts on compile-time constants at runtime, which would be
        // bad considering how frequently fast_atoreal_move<float> is called in Assimp.
        Real exp = static_cast<Real>( strtoul10_64(c, &c) );
        if (einv) {
            exp = -exp;
        }
        f *= std::pow(static_cast<Real>(10.0), exp);
    }

    if (inv) {
        f = -f;
    }
    out = f;
    return c;
}

// ------------------------------------------------------------------------------------
// The same but more human.
inline
ai_real fast_atof(const char* c) {
    ai_real ret(0.0);
    fast_atoreal_move<ai_real>(c, ret);

    return ret;
}

inline
ai_real fast_atof( const char* c, const char** cout) {
    ai_real ret(0.0);
    *cout = fast_atoreal_move<ai_real>(c, ret);

    return ret;
}

inline
ai_real fast_atof( const char** inout) {
    ai_real ret(0.0);
    *inout = fast_atoreal_move<ai_real>(*inout, ret);

    return ret;
}

} //! namespace Assimp

#endif // FAST_A_TO_F_H_INCLUDED
