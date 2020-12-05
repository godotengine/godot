#ifndef GIM_PAIR_H
#define GIM_PAIR_H


//! Overlapping pair
struct GIM_PAIR
{
        int m_index1;
        int m_index2;
        GIM_PAIR()
        {
        }

        GIM_PAIR(const GIM_PAIR& p)
        {
                m_index1 = p.m_index1;
                m_index2 = p.m_index2;
        }

        GIM_PAIR(int index1, int index2)
        {
                m_index1 = index1;
                m_index2 = index2;
        }
};

#endif //GIM_PAIR_H

