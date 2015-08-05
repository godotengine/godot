#pragma once

namespace Javelin {

class BitUtility {
public:
    static bool IsPowerOf2(unsigned int x) {
        return (x & (x - 1)) == 0;
    }

    static unsigned int RotateRight(unsigned int value, unsigned int shift) {
        if ((shift &= sizeof(value) * 8 - 1) == 0) {
            return value;
        }
        return (value >> shift) | (value << (sizeof(value) * 8 - shift));
    }
};

}
