// BenchCon.h

#ifndef ZIP7_INC_BENCH_CON_H
#define ZIP7_INC_BENCH_CON_H

#include <stdio.h>

#include "../../Common/CreateCoder.h"
#include "../../UI/Common/Property.h"

HRESULT BenchCon(DECL_EXTERNAL_CODECS_LOC_VARS
    const CObjectVector<CProperty> &props, UInt32 numIterations, FILE *f);

#endif
