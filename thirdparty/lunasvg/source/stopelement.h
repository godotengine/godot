#ifndef STOPELEMENT_H
#define STOPELEMENT_H

#include "styledelement.h"

namespace lunasvg {

class StopElement : public StyledElement
{
public:
    StopElement();

    double offset() const;
    Color stopColorWithOpacity() const;

    std::unique_ptr<Node> clone() const;
};

} // namespace lunasvg

#endif // STOPELEMENT_H
