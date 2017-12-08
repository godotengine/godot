// This code is in the public domain -- castanyo@yahoo.es

#pragma once
#ifndef NV_MATH_RANDOM_H
#define NV_MATH_RANDOM_H

#include "nvmath.h"
#include "nvcore/Utils.h" // nextPowerOfTwo


namespace nv
{

    /// Interface of the random number generators.
    class Rand
    {
    public:

        virtual ~Rand() {}

        enum time_e { Time };

        /// Provide a new seed.
        virtual void seed( uint s ) { /* empty */ };

        /// Get an integer random number.
        virtual uint get() = 0;

        /// Get a random number on [0, max] interval.
        uint getRange( uint max )
        {
            if (max == 0) return 0;
            if (max == NV_UINT32_MAX) return get();

            const uint np2 = nextPowerOfTwo( max+1 ); // @@ This fails if max == NV_UINT32_MAX
            const uint mask = np2 - 1;
            uint n;
            do { n = get() & mask; } while( n > max );
            return n;
        }

        /// Random number on [0.0, 1.0] interval.
        float getFloat()
        {
            union
            {
                uint32 i;
                float f;
            } pun;

            pun.i = 0x3f800000UL | (get() & 0x007fffffUL);
            return pun.f - 1.0f;
        }

        float getFloatRange(float min, float max) {
            return getFloat() * (max - min) + min;
        }

        /*
        /// Random number on [0.0, 1.0] interval.
        double getReal()
        {
        return double(get()) * (1.0/4294967295.0); // 2^32-1
        }

        /// Random number on [0.0, 1.0) interval.
        double getRealExclusive()
        {
        return double(get()) * (1.0/4294967296.0); // 2^32
        }
        */

        /// Get the max value of the random number.
        uint max() const { return NV_UINT32_MAX; }

        // Get a random seed.
        static uint randomSeed();

    };


    /// Very simple random number generator with low storage requirements.
    class SimpleRand : public Rand
    {
    public:

        /// Constructor that uses the current time as the seed.
        SimpleRand( time_e )
        {
            seed(randomSeed());
        }

        /// Constructor that uses the given seed.
        SimpleRand( uint s = 0 )
        {
            seed(s);
        }

        /// Set the given seed.
        virtual void seed( uint s )
        {
            current = s;
        }

        /// Get a random number.
        virtual uint get()
        {
            return current = current * 1103515245 + 12345;
        }

    private:

        uint current;

    };


    /// Mersenne twister random number generator.
    class MTRand : public Rand
    {
    public:

        enum { N = 624 };       // length of state vector
        enum { M = 397 };

        /// Constructor that uses the current time as the seed.
        MTRand( time_e )
        {
            seed(randomSeed());
        }

        /// Constructor that uses the given seed.
        MTRand( uint s = 0 )
        {
            seed(s);
        }

        /// Constructor that uses the given seeds.
        NVMATH_API MTRand( const uint * seed_array, uint length );


        /// Provide a new seed.
        virtual void seed( uint s )
        {
            initialize(s);
            reload();
        }	

        /// Get a random number between 0 - 65536.
        virtual uint get()
        {
            // Pull a 32-bit integer from the generator state
            // Every other access function simply transforms the numbers extracted here
            if( left == 0 ) { 
                reload(); 
            }
            left--;

            uint s1;
            s1 = *next++;
            s1 ^= (s1 >> 11);
            s1 ^= (s1 <<  7) & 0x9d2c5680U;
            s1 ^= (s1 << 15) & 0xefc60000U;
            return ( s1 ^ (s1 >> 18) );		
        };


    private:

        NVMATH_API void initialize( uint32 seed );
        NVMATH_API void reload();

        uint hiBit( uint u ) const { return u & 0x80000000U; }
        uint loBit( uint u ) const { return u & 0x00000001U; }
        uint loBits( uint u ) const { return u & 0x7fffffffU; }
        uint mixBits( uint u, uint v ) const { return hiBit(u) | loBits(v); }
        uint twist( uint m, uint s0, uint s1 ) const { return m ^ (mixBits(s0,s1)>>1) ^ ((~loBit(s1)+1) & 0x9908b0dfU); }

    private:

        uint state[N];	// internal state
        uint * next;	// next value to get from state
        int left;		// number of values left before reload needed		

    };



    /** George Marsaglia's random number generator. 
    * Code based on Thatcher Ulrich public domain source code:
    * http://cvs.sourceforge.net/viewcvs.py/tu-testbed/tu-testbed/base/tu_random.cpp?rev=1.7&view=auto
    *
    * PRNG code adapted from the complimentary-multiply-with-carry
    * code in the article: George Marsaglia, "Seeds for Random Number
    * Generators", Communications of the ACM, May 2003, Vol 46 No 5,
    * pp90-93.
    * 
    * The article says:
    * 
    * "Any one of the choices for seed table size and multiplier will
    * provide a RNG that has passed extensive tests of randomness,
    * particularly those in [3], yet is simple and fast --
    * approximately 30 million random 32-bit integers per second on a
    * 850MHz PC.  The period is a*b^n, where a is the multiplier, n
    * the size of the seed table and b=2^32-1.  (a is chosen so that
    * b is a primitive root of the prime a*b^n + 1.)"
    * 
    * [3] Marsaglia, G., Zaman, A., and Tsang, W.  Toward a universal
    * random number generator.  _Statistics and Probability Letters
    * 8_ (1990), 35-39.
    */
    class GMRand : public Rand
    {
    public:

        enum { SEED_COUNT = 8 };

        //	const uint64 a = 123471786;		// for SEED_COUNT=1024
        //	const uint64 a = 123554632;		// for SEED_COUNT=512
        //	const uint64 a = 8001634;		// for SEED_COUNT=255
        //	const uint64 a = 8007626;		// for SEED_COUNT=128
        //	const uint64 a = 647535442;		// for SEED_COUNT=64
        //	const uint64 a = 547416522;		// for SEED_COUNT=32
        //	const uint64 a = 487198574;		// for SEED_COUNT=16
        //	const uint64 a = 716514398U;	// for SEED_COUNT=8
        enum { a = 716514398U };


        GMRand( time_e )
        {
            seed(randomSeed());
        }

        GMRand(uint s = 987654321)
        {
            seed(s);
        }


        /// Provide a new seed.
        virtual void seed( uint s )
        {
            c = 362436;
            i = SEED_COUNT - 1;

            for(int i = 0; i < SEED_COUNT; i++) {
                s = s ^ (s << 13);
                s = s ^ (s >> 17);
                s = s ^ (s << 5);
                Q[i] = s;
            }
        }

        /// Get a random number between 0 - 65536.
        virtual uint get()
        {
            const uint32 r = 0xFFFFFFFE;		

            uint64 t;
            uint32 x;

            i = (i + 1) & (SEED_COUNT - 1);
            t = a * Q[i] + c;
            c = uint32(t >> 32);
            x = uint32(t + c);

            if( x < c ) {
                x++;
                c++;
            }

            uint32  val = r - x;
            Q[i] = val;
            return val;
        };


    private:

        uint32 c;
        uint32 i;
        uint32 Q[8];

    };


    /** Random number implementation from the GNU Sci. Lib. (GSL).
    * Adapted from Nicholas Chapman version:
    * 
    * Copyright (C) 1996, 1997, 1998, 1999, 2000 James Theiler, Brian Gough
    * This is the Unix rand48() generator. The generator returns the
    * upper 32 bits from each term of the sequence,
    * 
    * x_{n+1} = (a x_n + c) mod m 
    * 
    * using 48-bit unsigned arithmetic, with a = 0x5DEECE66D , c = 0xB
    * and m = 2^48. The seed specifies the upper 32 bits of the initial
    * value, x_1, with the lower 16 bits set to 0x330E.
    * 
    * The theoretical value of x_{10001} is 244131582646046.
    * 
    * The period of this generator is ? FIXME (probably around 2^48). 
    */
    class Rand48 : public Rand
    {
    public:

        Rand48( time_e )
        {
            seed(randomSeed());
        }

        Rand48( uint s = 0x1234ABCD )
        {
            seed(s);
        }	


        /** Set the given seed. */
        virtual void seed( uint s ) {
            vstate.x0 = 0x330E;
            vstate.x1 = uint16(s & 0xFFFF);
            vstate.x2 = uint16((s >> 16) & 0xFFFF);
        }

        /** Get a random number. */
        virtual uint get() {

            advance();

            uint x1 = vstate.x1;
            uint x2 = vstate.x2;
            return (x2 << 16) + x1;
        }


    private:

        void advance()
        {
            /* work with unsigned long ints throughout to get correct integer
            promotions of any unsigned short ints */
            const uint32 x0 = vstate.x0;
            const uint32 x1 = vstate.x1;
            const uint32 x2 = vstate.x2;

            uint32 a;
            a = a0 * x0 + c0;

            vstate.x0 = uint16(a & 0xFFFF);
            a >>= 16;

            /* although the next line may overflow we only need the top 16 bits
            in the following stage, so it does not matter */

            a += a0 * x1 + a1 * x0; 
            vstate.x1 = uint16(a & 0xFFFF);

            a >>= 16;
            a += a0 * x2 + a1 * x1 + a2 * x0;
            vstate.x2 = uint16(a & 0xFFFF);
        }


    private:	
        NVMATH_API static const uint16 a0, a1, a2, c0;

        struct rand48_state_t { 
            uint16 x0, x1, x2; 
        } vstate;

    };

} // nv namespace

#endif // NV_MATH_RANDOM_H
