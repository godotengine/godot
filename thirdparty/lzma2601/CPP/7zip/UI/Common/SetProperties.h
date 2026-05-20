// SetProperties.h

#ifndef ZIP7_INC_SETPROPERTIES_H
#define ZIP7_INC_SETPROPERTIES_H

#include "Property.h"

HRESULT SetProperties(IUnknown *unknown, const CObjectVector<CProperty> &properties);

#endif
