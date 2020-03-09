/*
---------------------------------------------------------------------------
Open Asset Import Library (assimp)
---------------------------------------------------------------------------

Copyright (c) 2006-2019, assimp team



All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the following
conditions are met:

* Redistributions of source code must retain the above
  copyright notice, this list of conditions and the
  following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the
  following disclaimer in the documentation and/or other
  materials provided with the distribution.

* Neither the name of the assimp team, nor the names of its
  contributors may be used to endorse or promote products
  derived from this software without specific prior
  written permission of the assimp team.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
---------------------------------------------------------------------------
*/

/** @file Implementation of the helper class to quickly find vertices close to a given position */

#include <assimp/SpatialSort.h>
#include <assimp/ai_assert.h>

using namespace Assimp;

// CHAR_BIT seems to be defined under MVSC, but not under GCC. Pray that the correct value is 8.
#ifndef CHAR_BIT
#   define CHAR_BIT 8
#endif

// ------------------------------------------------------------------------------------------------
// Constructs a spatially sorted representation from the given position array.
SpatialSort::SpatialSort( const aiVector3D* pPositions, unsigned int pNumPositions,
    unsigned int pElementOffset)

    // define the reference plane. We choose some arbitrary vector away from all basic axises
    // in the hope that no model spreads all its vertices along this plane.
    : mPlaneNormal(0.8523f, 0.34321f, 0.5736f)
{
    mPlaneNormal.Normalize();
    Fill(pPositions,pNumPositions,pElementOffset);
}

// ------------------------------------------------------------------------------------------------
SpatialSort :: SpatialSort()
: mPlaneNormal(0.8523f, 0.34321f, 0.5736f)
{
    mPlaneNormal.Normalize();
}

// ------------------------------------------------------------------------------------------------
// Destructor
SpatialSort::~SpatialSort()
{
    // nothing to do here, everything destructs automatically
}

// ------------------------------------------------------------------------------------------------
void SpatialSort::Fill( const aiVector3D* pPositions, unsigned int pNumPositions,
    unsigned int pElementOffset,
    bool pFinalize /*= true */)
{
    mPositions.clear();
    Append(pPositions,pNumPositions,pElementOffset,pFinalize);
}

// ------------------------------------------------------------------------------------------------
void SpatialSort :: Finalize()
{
    std::sort( mPositions.begin(), mPositions.end());
}

// ------------------------------------------------------------------------------------------------
void SpatialSort::Append( const aiVector3D* pPositions, unsigned int pNumPositions,
    unsigned int pElementOffset,
    bool pFinalize /*= true */)
{
    // store references to all given positions along with their distance to the reference plane
    const size_t initial = mPositions.size();
    mPositions.reserve(initial + (pFinalize?pNumPositions:pNumPositions*2));
    for( unsigned int a = 0; a < pNumPositions; a++)
    {
        const char* tempPointer = reinterpret_cast<const char*> (pPositions);
        const aiVector3D* vec   = reinterpret_cast<const aiVector3D*> (tempPointer + a * pElementOffset);

        // store position by index and distance
        ai_real distance = *vec * mPlaneNormal;
        mPositions.push_back( Entry( static_cast<unsigned int>(a+initial), *vec, distance));
    }

    if (pFinalize) {
        // now sort the array ascending by distance.
        Finalize();
    }
}

// ------------------------------------------------------------------------------------------------
// Returns an iterator for all positions close to the given position.
void SpatialSort::FindPositions( const aiVector3D& pPosition,
    ai_real pRadius, std::vector<unsigned int>& poResults) const
{
    const ai_real dist = pPosition * mPlaneNormal;
    const ai_real minDist = dist - pRadius, maxDist = dist + pRadius;

    // clear the array
    poResults.clear();

    // quick check for positions outside the range
    if( mPositions.size() == 0)
        return;
    if( maxDist < mPositions.front().mDistance)
        return;
    if( minDist > mPositions.back().mDistance)
        return;

    // do a binary search for the minimal distance to start the iteration there
    unsigned int index = (unsigned int)mPositions.size() / 2;
    unsigned int binaryStepSize = (unsigned int)mPositions.size() / 4;
    while( binaryStepSize > 1)
    {
        if( mPositions[index].mDistance < minDist)
            index += binaryStepSize;
        else
            index -= binaryStepSize;

        binaryStepSize /= 2;
    }

    // depending on the direction of the last step we need to single step a bit back or forth
    // to find the actual beginning element of the range
    while( index > 0 && mPositions[index].mDistance > minDist)
        index--;
    while( index < (mPositions.size() - 1) && mPositions[index].mDistance < minDist)
        index++;

    // Mow start iterating from there until the first position lays outside of the distance range.
    // Add all positions inside the distance range within the given radius to the result aray
    std::vector<Entry>::const_iterator it = mPositions.begin() + index;
    const ai_real pSquared = pRadius*pRadius;
    while( it->mDistance < maxDist)
    {
        if( (it->mPosition - pPosition).SquareLength() < pSquared)
            poResults.push_back( it->mIndex);
        ++it;
        if( it == mPositions.end())
            break;
    }

    // that's it
}

namespace {

    // Binary, signed-integer representation of a single-precision floating-point value.
    // IEEE 754 says: "If two floating-point numbers in the same format are ordered then they are
    //  ordered the same way when their bits are reinterpreted as sign-magnitude integers."
    // This allows us to convert all floating-point numbers to signed integers of arbitrary size
    //  and then use them to work with ULPs (Units in the Last Place, for high-precision
    //  computations) or to compare them (integer comparisons are faster than floating-point
    //  comparisons on many platforms).
    typedef ai_int BinFloat;

    // --------------------------------------------------------------------------------------------
    // Converts the bit pattern of a floating-point number to its signed integer representation.
    BinFloat ToBinary( const ai_real & pValue) {

        // If this assertion fails, signed int is not big enough to store a float on your platform.
        //  Please correct the declaration of BinFloat a few lines above - but do it in a portable,
        //  #ifdef'd manner!
        static_assert( sizeof(BinFloat) >= sizeof(ai_real), "sizeof(BinFloat) >= sizeof(ai_real)");

        #if defined( _MSC_VER)
            // If this assertion fails, Visual C++ has finally moved to ILP64. This means that this
            //  code has just become legacy code! Find out the current value of _MSC_VER and modify
            //  the #if above so it evaluates false on the current and all upcoming VC versions (or
            //  on the current platform, if LP64 or LLP64 are still used on other platforms).
            static_assert( sizeof(BinFloat) == sizeof(ai_real), "sizeof(BinFloat) == sizeof(ai_real)");

            // This works best on Visual C++, but other compilers have their problems with it.
            const BinFloat binValue = reinterpret_cast<BinFloat const &>(pValue);
        #else
            // On many compilers, reinterpreting a float address as an integer causes aliasing
            // problems. This is an ugly but more or less safe way of doing it.
            union {
                ai_real     asFloat;
                BinFloat    asBin;
            } conversion;
            conversion.asBin    = 0; // zero empty space in case sizeof(BinFloat) > sizeof(float)
            conversion.asFloat  = pValue;
            const BinFloat binValue = conversion.asBin;
        #endif

        // floating-point numbers are of sign-magnitude format, so find out what signed number
        //  representation we must convert negative values to.
        // See http://en.wikipedia.org/wiki/Signed_number_representations.

        // Two's complement?
        if( (-42 == (~42 + 1)) && (binValue & 0x80000000))
            return BinFloat(1 << (CHAR_BIT * sizeof(BinFloat) - 1)) - binValue;
        // One's complement?
        else if ( (-42 == ~42) && (binValue & 0x80000000))
            return BinFloat(-0) - binValue;
        // Sign-magnitude?
        else if( (-42 == (42 | (-0))) && (binValue & 0x80000000)) // -0 = 1000... binary
            return binValue;
        else
            return binValue;
    }

} // namespace

// ------------------------------------------------------------------------------------------------
// Fills an array with indices of all positions identical to the given position. In opposite to
// FindPositions(), not an epsilon is used but a (very low) tolerance of four floating-point units.
void SpatialSort::FindIdenticalPositions( const aiVector3D& pPosition,
    std::vector<unsigned int>& poResults) const
{
    // Epsilons have a huge disadvantage: they are of constant precision, while floating-point
    //  values are of log2 precision. If you apply e=0.01 to 100, the epsilon is rather small, but
    //  if you apply it to 0.001, it is enormous.

    // The best way to overcome this is the unit in the last place (ULP). A precision of 2 ULPs
    //  tells us that a float does not differ more than 2 bits from the "real" value. ULPs are of
    //  logarithmic precision - around 1, they are 1*(2^24) and around 10000, they are 0.00125.

    // For standard C math, we can assume a precision of 0.5 ULPs according to IEEE 754. The
    //  incoming vertex positions might have already been transformed, probably using rather
    //  inaccurate SSE instructions, so we assume a tolerance of 4 ULPs to safely identify
    //  identical vertex positions.
    static const int toleranceInULPs = 4;
    // An interesting point is that the inaccuracy grows linear with the number of operations:
    //  multiplying to numbers, each inaccurate to four ULPs, results in an inaccuracy of four ULPs
    //  plus 0.5 ULPs for the multiplication.
    // To compute the distance to the plane, a dot product is needed - that is a multiplication and
    //  an addition on each number.
    static const int distanceToleranceInULPs = toleranceInULPs + 1;
    // The squared distance between two 3D vectors is computed the same way, but with an additional
    //  subtraction.
    static const int distance3DToleranceInULPs = distanceToleranceInULPs + 1;

    // Convert the plane distance to its signed integer representation so the ULPs tolerance can be
    //  applied. For some reason, VC won't optimize two calls of the bit pattern conversion.
    const BinFloat minDistBinary = ToBinary( pPosition * mPlaneNormal) - distanceToleranceInULPs;
    const BinFloat maxDistBinary = minDistBinary + 2 * distanceToleranceInULPs;

    // clear the array in this strange fashion because a simple clear() would also deallocate
    // the array which we want to avoid
    poResults.resize( 0 );

    // do a binary search for the minimal distance to start the iteration there
    unsigned int index = (unsigned int)mPositions.size() / 2;
    unsigned int binaryStepSize = (unsigned int)mPositions.size() / 4;
    while( binaryStepSize > 1)
    {
        // Ugly, but conditional jumps are faster with integers than with floats
        if( minDistBinary > ToBinary(mPositions[index].mDistance))
            index += binaryStepSize;
        else
            index -= binaryStepSize;

        binaryStepSize /= 2;
    }

    // depending on the direction of the last step we need to single step a bit back or forth
    // to find the actual beginning element of the range
    while( index > 0 && minDistBinary < ToBinary(mPositions[index].mDistance) )
        index--;
    while( index < (mPositions.size() - 1) && minDistBinary > ToBinary(mPositions[index].mDistance))
        index++;

    // Now start iterating from there until the first position lays outside of the distance range.
    // Add all positions inside the distance range within the tolerance to the result array
    std::vector<Entry>::const_iterator it = mPositions.begin() + index;
    while( ToBinary(it->mDistance) < maxDistBinary)
    {
        if( distance3DToleranceInULPs >= ToBinary((it->mPosition - pPosition).SquareLength()))
            poResults.push_back(it->mIndex);
        ++it;
        if( it == mPositions.end())
            break;
    }

    // that's it
}

// ------------------------------------------------------------------------------------------------
unsigned int SpatialSort::GenerateMappingTable(std::vector<unsigned int>& fill, ai_real pRadius) const
{
    fill.resize(mPositions.size(),UINT_MAX);
    ai_real dist, maxDist;

    unsigned int t=0;
    const ai_real pSquared = pRadius*pRadius;
    for (size_t i = 0; i < mPositions.size();) {
        dist = mPositions[i].mPosition * mPlaneNormal;
        maxDist = dist + pRadius;

        fill[mPositions[i].mIndex] = t;
        const aiVector3D& oldpos = mPositions[i].mPosition;
        for (++i; i < fill.size() && mPositions[i].mDistance < maxDist
            && (mPositions[i].mPosition - oldpos).SquareLength() < pSquared; ++i)
        {
            fill[mPositions[i].mIndex] = t;
        }
        ++t;
    }

#ifdef ASSIMP_BUILD_DEBUG

    // debug invariant: mPositions[i].mIndex values must range from 0 to mPositions.size()-1
    for (size_t i = 0; i < fill.size(); ++i) {
        ai_assert(fill[i]<mPositions.size());
    }

#endif
    return t;
}
