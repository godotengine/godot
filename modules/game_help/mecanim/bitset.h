#pragma once


#include "./defs.h"
#include "./types.h"

namespace mecanim
{
    template<uint32_t Bits> class bitset
    {
    public:
        typedef uint32_t type;
        enum
        {
            digits = Bits,
            // parameters for packing bits into words
            Bitsperword = (int)(8 * sizeof(type)),  // bits in each word
            Words = (int)(Bits == 0 ? 0 : (Bits - 1) / Bitsperword)
        };

        bitset(type value = 0)
        {
            init(value);
        }

        bitset<Bits>& set()
        {
            // set all bits true
            init((type) ~0);
            return (*this);
        }

        bitset<Bits>& set(type pos, bool val = true)
        {
            // set bit at _Pos to _Val
            if (pos < Bits)
            {
                if (val)
                    mArray[pos / Bitsperword] |= (type)1 << pos % Bitsperword;
                else
                    mArray[pos / Bitsperword] &= ~((type)1 << pos % Bitsperword);
            }
            return (*this);
        }

        bitset<Bits>& reset()
        {
            // set all bits false
            init();
            return (*this);
        }

        bitset<Bits>& reset(type pos)
        {
            // set bit at pos to false
            return set(pos, false);
        }

        bitset<Bits>& flip()
        {
            // flip all bits
            for (int pos = Words; 0 <= pos; --pos)
                mArray[pos] = (type) ~mArray[pos];

            trim();
            return *this;
        }

        bitset<Bits>& flip(type pos)
        {
            // flip bit at pos
            if (pos < Bits)
                mArray[pos / Bitsperword] ^= (type)1 << pos % Bitsperword;
            return (*this);
        }

        size_t count() const
        {
            // count number of set bits
            static char Bitsperhex[] = "\0\1\1\2\1\2\2\3\1\2\2\3\2\3\3\4";
            type val = 0;
            for (int pos = Words; 0 <= pos; --pos)
                for (type Wordval = mArray[pos]; Wordval != 0; Wordval >>= 4)
                    val += Bitsperhex[Wordval & 0xF];
            return val;
        }

        type size() const
        {
            // return size of bitset
            return (Bits);
        }

        bool test(uint32_t pos) const
        {
            // test if bit at pos is set
            if (pos < Bits)
                return ((mArray[pos / Bitsperword] & ((type)1 << pos % Bitsperword)) != 0);
            return false;
        }

        bool any() const
        {
            // test if any bits are set
            for (int pos = Words; 0 <= pos; --pos)
                if (mArray[pos] != 0)
                    return true;
            return false;
        }

        bool none() const
        {
            // test if no bits are set
            return !any();
        }

        bool operator==(const bitset<Bits>& right) const
        {
            // test for bitset equality
            for (int pos = Words; 0 <= pos; --pos)
                if (mArray[pos] != right.word(pos))
                    return false;
            return true;
        }

        type const& word(uint32_t pos) const
        {
            // get word at pos
            return mArray[pos];
        }

        type& word(uint32_t pos)
        {
            // get word at pos
            return mArray[pos];
        }

    protected:
        void init(type value = 0)
        {
            // set all words to value
            for (int pos = Words; 0 <= pos; --pos)
                mArray[pos] = value;
            if (value != 0)
                trim();
        }

        void trim()
        {
            // clear any trailing bits in last word
            if (Bits % Bitsperword != 0)
                mArray[Words] &= ((type)1 << Bits % Bitsperword) - 1;
        }

        type mArray[Words + 1];
    };
}
