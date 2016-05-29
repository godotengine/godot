//  NREX: Node RegEx
//  Version 0.2
//
//  Copyright (c) 2015-2016, Zher Huei Lee
//  All rights reserved.
//
//  This software is provided 'as-is', without any express or implied
//  warranty.  In no event will the authors be held liable for any damages
//  arising from the use of this software.
//
//  Permission is granted to anyone to use this software for any purpose,
//  including commercial applications, and to alter it and redistribute it
//  freely, subject to the following restrictions:
//
//   1. The origin of this software must not be misrepresented; you must not
//      claim that you wrote the original software. If you use this software
//      in a product, an acknowledgment in the product documentation would
//      be appreciated but is not required.
//
//   2. Altered source versions must be plainly marked as such, and must not
//      be misrepresented as being the original software.
//
//   3. This notice may not be removed or altered from any source
//      distribution.
//

#ifndef NREX_HPP
#define NREX_HPP

#include "nrex_config.h"

#ifdef NREX_UNICODE
typedef wchar_t nrex_char;
#else
typedef char nrex_char;
#endif

/*!
 * \brief Struct to contain the range of a capture result
 *
 * The range provided is relative to the begining of the searched string.
 *
 * \see nrex_node::match()
 */
struct nrex_result
{
    public:
        int start; /*!< Start of text range */
        int length; /*!< Length of text range */
};

class nrex_node;

/*!
 * \brief Holds the compiled regex pattern
 */
class nrex
{
    private:
        unsigned int _capturing;
        unsigned int _lookahead_depth;
        nrex_node* _root;
    public:

        /*!
         * \brief Initialises an empty regex container
         */
        nrex();

        /*!
         * \brief Initialises and compiles the regex pattern
         *
         * This calls nrex::compile() with the same arguments. To check whether
         * the compilation was successfull, use nrex::valid().
         *
         * If the NREX_THROW_ERROR was defined it would automatically throw a
         * runtime error nrex_compile_error if it encounters a problem when
         * parsing the pattern.
         *
         * \param pattern   The regex pattern
         * \param captures  The maximum number of capture groups to allow. Any
         *                  extra would be converted to non-capturing groups.
         *                  If negative, no limit would be imposed. Defaults
         *                  to 9.
         *
         * \see nrex::compile()
         */
        nrex(const nrex_char* pattern, int captures = 9);

        ~nrex();

        /*!
         * \brief Removes the compiled regex and frees up the memory
         */
        void reset();

        /*!
         * \brief Checks if there is a compiled regex being stored
         * \return True if present, False if not present
         */
        bool valid() const;

        /*!
         * \brief Provides number of captures the compiled regex uses
         *
         * This is used to provide the array size of the captures needed for
         * nrex::match() to work. The size is actually the number of capture
         * groups + one for the matching of the entire pattern. This can be
         * capped using the extra argument given in nrex::compile()
         * (default 10).
         *
         * \return The number of captures
         */
        int capture_size() const;

        /*!
         * \brief Compiles the provided regex pattern
         *
         * This automatically removes the existing compiled regex if already
         * present.
         *
         * If the NREX_THROW_ERROR was defined it would automatically throw a
         * runtime error nrex_compile_error if it encounters a problem when
         * parsing the pattern.
         *
         * \param pattern   The regex pattern
         * \param captures  The maximum number of capture groups to allow. Any
         *                  extra would be converted to non-capturing groups.
         *                  If negative, no limit would be imposed. Defaults
         *                  to 9.
         * \return True if the pattern was succesfully compiled
         */
        bool compile(const nrex_char* pattern, int captures = 9);

        /*!
         * \brief Uses the pattern to search through the provided string
         * \param str       The text to search through. It only needs to be
         *                  null terminated if the end point is not provided.
         *                  This also determines the starting anchor.
         * \param captures  The array of results to store the capture results.
         *                  The size of that array needs to be the same as the
         *                  size given in nrex::capture_size(). As it matches
         *                  the function fills the array with the results. 0 is
         *                  the result for the entire pattern, 1 and above
         *                  corresponds to the regex capture group if present.
         * \param offset    The starting point of the search. This does not move
         *                  the starting anchor. Defaults to 0.
         * \param end       The end point of the search. This also determines
         *                  the ending anchor. If a number less than the offset
         *                  is provided, the search would be done until null
         *                  termination. Defaults to -1.
         * \return          True if a match was found. False otherwise.
         */
        bool match(const nrex_char* str, nrex_result* captures, int offset = 0, int end = -1) const;
};

#ifdef NREX_THROW_ERROR

#include <stdexcept>

class nrex_compile_error : std::runtime_error
{
    public:
        nrex_compile_error(const char* message)
            : std::runtime_error(message)
        {
        }

        ~nrex_compile_error() throw()
        {
        }
};

#endif

#endif // NREX_HPP
