#pragma once

namespace Javelin {

template<typename T>
class Point2 {
public:
    T x;
    T y;

    Point2(int a, int b)
        : x(a)
        , y(b) {
    }
};

}
