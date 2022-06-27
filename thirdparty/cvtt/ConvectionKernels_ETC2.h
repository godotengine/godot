#include <stdint.h>

namespace cvtt
{
    namespace Tables
    {
        namespace ETC2
        {
            const int16_t g_thModifierTable[8] =
            {
                3, 6, 11, 16, 23, 32, 41, 64
            };

            const int16_t g_alphaModifierTablePositive[16][4] =
            {
                { 2, 5, 8, 14, },
                { 2, 6, 9, 12, },
                { 1, 4, 7, 12, },
                { 1, 3, 5, 12, },
                { 2, 5, 7, 11, },
                { 2, 6, 8, 10, },
                { 3, 6, 7, 10, },
                { 2, 4, 7, 10, },
                { 1, 5, 7, 9, },
                { 1, 4, 7, 9, },
                { 1, 3, 7, 9, },
                { 1, 4, 6, 9, },
                { 2, 3, 6, 9, },
                { 0, 1, 2, 9, },
                { 3, 5, 7, 8, },
                { 2, 4, 6, 8, },
            };
        }
    }
}
