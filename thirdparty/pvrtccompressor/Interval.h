#pragma once

namespace Javelin {

template<typename T>
class Interval {
public:
    T min;
    T max;

    Interval() {
    }

    Interval<T> &operator|=(const T &x) {
        min.SetMin(x); 
        max.SetMax(x);
        return *this;
    }
};

}
