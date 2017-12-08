// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#pragma once
#ifndef NV_CORE_BITARRAY_H
#define NV_CORE_BITARRAY_H

#include "nvcore.h"
#include "Array.inl"

namespace nv
{

    // @@ Uh, this could be much faster.
    inline uint countSetBits(uint32 x) {
        uint count = 0;
        for(; x != 0; x >>= 1) {
            count += (x & 1);
        }
        return count;
    }

    // @@ This is even more lame. What was I thinking?
    inline uint countSetBits(uint32 x, int bits) {
        uint count = 0;
        for(; x != 0 && bits != 0; x >>= 1, bits--) {
            count += (x & 1);
        }
        return count;
    }

    // See "Conditionally set or clear bits without branching" at http://graphics.stanford.edu/~seander/bithacks.html
    inline uint setBits(uint w, uint m, bool b) {
        return (w & ~m) | (-int(b) & m);
    }



    // Simple bit array.
    class BitArray
    {
    public:

        BitArray() {}
        BitArray(uint sz) {
            resize(sz);
        }

        uint size() const { return m_size; }
        void clear() { resize(0); }

        void resize(uint new_size)
        {
            m_size = new_size;
            m_wordArray.resize( (m_size + 31) >> 5 );
        }

        void resize(uint new_size, bool init)
        {
            //if (new_size == m_size) return;

            uint old_size = m_size;
            uint size_mod_32 = old_size & 31;
            uint last_word_index = ((old_size + 31) >> 5) - 1;
            uint mask = (1 << size_mod_32) - 1;

            uint init_dword;
            if (init) {
                if (size_mod_32) m_wordArray[last_word_index] |= ~mask;
                init_dword = ~0;
            }
            else {
                if (size_mod_32) m_wordArray[last_word_index] &= mask;
                init_dword = 0;
            }

            m_size = new_size;
            m_wordArray.resize((new_size + 31) >> 5, init_dword);

            // Make sure new bits are initialized correctly.
            /*for (uint i = old_size; i < new_size; i++) {
                nvCheck(bitAt(i) == init);
            }*/
        }


        /// Get bit.
        bool bitAt(uint b) const
        {
            nvDebugCheck( b < m_size );
            return (m_wordArray[b >> 5] & (1 << (b & 31))) != 0;
        }

        // It may be useful to pack mulitple bit arrays together interleaving their bits.
        uint bitsAt(uint idx, uint count) const
        {
            //nvDebugCheck(count == 2 || count == 4 || count == 8 || count == 16 || count == 32);
            nvDebugCheck(count == 2);   // @@ Hardcoded for two.
            uint b = idx * count;
            nvDebugCheck(b < m_size);
            return (m_wordArray[b >> 5] & (0x3 << (b & 31))) >> (b & 31);
        }

        // It would be useful to have a function to set two bits simultaneously.
        /*void setBitsAt(uint idx, uint count, uint bits) const
        {
            //nvDebugCheck(count == 2 || count == 4 || count == 8 || count == 16 || count == 32);
            nvDebugCheck(count == 2);   // @@ Hardcoded for two.
            uint b = idx * count;
            nvDebugCheck(b < m_size);
            return (m_wordArray[b >> 5] & (0x3 << (b & 31))) >> (b & 31);
        }*/



        // Set a bit.
        void setBitAt(uint idx)
        {
            nvDebugCheck(idx < m_size);
            m_wordArray[idx >> 5] |=  (1 << (idx & 31));
        }

        // Clear a bit.
        void clearBitAt(uint idx)
        {
            nvDebugCheck(idx < m_size);
            m_wordArray[idx >> 5] &= ~(1 << (idx & 31));
        }

        // Toggle a bit.
        void toggleBitAt(uint idx)
        {
            nvDebugCheck(idx < m_size);
            m_wordArray[idx >> 5] ^= (1 << (idx & 31));
        }

        // Set a bit to the given value. @@ Rename modifyBitAt? 
        void setBitAt(uint idx, bool b)
        {
            nvDebugCheck(idx < m_size);
            m_wordArray[idx >> 5] = setBits(m_wordArray[idx >> 5], 1 << (idx & 31), b);
            nvDebugCheck(bitAt(idx) == b);
        }

        void append(bool value)
        {
            resize(m_size + 1);
            setBitAt(m_size - 1, value);
        }


        // Clear all the bits.
        void clearAll()
        {
            memset(m_wordArray.buffer(), 0, m_wordArray.size() * sizeof(uint));
        }

        // Set all the bits.
        void setAll()
        {
            memset(m_wordArray.buffer(), 0xFF, m_wordArray.size() * sizeof(uint));
        }

        // Toggle all the bits.
        void toggleAll()
        {
            const uint wordCount = m_wordArray.count();
            for(uint b = 0; b < wordCount; b++) {
                m_wordArray[b] ^= 0xFFFFFFFF;
            }
        }

        // Count the number of bits set.
        uint countSetBits() const
        {
            const uint num = m_wordArray.size();
            if( num == 0 ) {
                return 0;
            }

            uint count = 0;				
            for(uint i = 0; i < num - 1; i++) {
                count += nv::countSetBits(m_wordArray[i]);
            }
            count += nv::countSetBits(m_wordArray[num - 1], m_size & 31);

            //piDebugCheck(count + countClearBits() == m_size);
            return count;
        }

        // Count the number of bits clear.
        uint countClearBits() const {

            const uint num = m_wordArray.size();
            if( num == 0 ) {
                return 0;
            }

            uint count = 0;
            for(uint i = 0; i < num - 1; i++) {
                count += nv::countSetBits(~m_wordArray[i]);
            }
            count += nv::countSetBits(~m_wordArray[num - 1], m_size & 31);

            //piDebugCheck(count + countSetBits() == m_size);
            return count;
        }

        friend void swap(BitArray & a, BitArray & b)
        {
            swap(a.m_size, b.m_size);
            swap(a.m_wordArray, b.m_wordArray);
        }

        void operator &= (const BitArray & other) {
            if (other.m_size != m_size) {
                resize(other.m_size);
            }

            const uint wordCount = m_wordArray.count();
            for (uint i = 0; i < wordCount; i++) {
                m_wordArray[i] &= other.m_wordArray[i];
            }
        }

        void operator |= (const BitArray & other) {
            if (other.m_size != m_size) {
                resize(other.m_size);
            }

            const uint wordCount = m_wordArray.count();
            for (uint i = 0; i < wordCount; i++) {
                m_wordArray[i] |= other.m_wordArray[i];
            }
        }


    private:

        // Number of bits stored.
        uint m_size;

        // Array of bits.
        Array<uint> m_wordArray;

    };

} // nv namespace

#endif // NV_CORE_BITARRAY_H

