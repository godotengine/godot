/*
 * Copyright (c) 2020 - 2023 the ThorVG project. All rights reserved.

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "tvgSwCommon.h"


/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

SwOutline* mpoolReqOutline(SwMpool* mpool, unsigned idx)
{
    return &mpool->outline[idx];
}


void mpoolRetOutline(SwMpool* mpool, unsigned idx)
{
    mpool->outline[idx].pts.clear();
    mpool->outline[idx].cntrs.clear();
    mpool->outline[idx].types.clear();
    mpool->outline[idx].closed.clear();
}


SwOutline* mpoolReqStrokeOutline(SwMpool* mpool, unsigned idx)
{
    return &mpool->strokeOutline[idx];
}


void mpoolRetStrokeOutline(SwMpool* mpool, unsigned idx)
{
    mpool->strokeOutline[idx].pts.clear();
    mpool->strokeOutline[idx].cntrs.clear();
    mpool->strokeOutline[idx].types.clear();
    mpool->strokeOutline[idx].closed.clear();
}


SwOutline* mpoolReqDashOutline(SwMpool* mpool, unsigned idx)
{
    return &mpool->dashOutline[idx];
}


void mpoolRetDashOutline(SwMpool* mpool, unsigned idx)
{
    mpool->dashOutline[idx].pts.clear();
    mpool->dashOutline[idx].cntrs.clear();
    mpool->dashOutline[idx].types.clear();
    mpool->dashOutline[idx].closed.clear();
}


SwMpool* mpoolInit(unsigned threads)
{
    auto allocSize = threads + 1;

    auto mpool = static_cast<SwMpool*>(calloc(sizeof(SwMpool), 1));
    mpool->outline = static_cast<SwOutline*>(calloc(1, sizeof(SwOutline) * allocSize));
    mpool->strokeOutline = static_cast<SwOutline*>(calloc(1, sizeof(SwOutline) * allocSize));
    mpool->dashOutline = static_cast<SwOutline*>(calloc(1, sizeof(SwOutline) * allocSize));
    mpool->allocSize = allocSize;

    return mpool;
}


bool mpoolClear(SwMpool* mpool)
{
    for (unsigned i = 0; i < mpool->allocSize; ++i) {
        mpool->outline[i].pts.reset();
        mpool->outline[i].cntrs.reset();
        mpool->outline[i].types.reset();
        mpool->outline[i].closed.reset();

        mpool->strokeOutline[i].pts.reset();
        mpool->strokeOutline[i].cntrs.reset();
        mpool->strokeOutline[i].types.reset();
        mpool->strokeOutline[i].closed.reset();

        mpool->dashOutline[i].pts.reset();
        mpool->dashOutline[i].cntrs.reset();
        mpool->dashOutline[i].types.reset();
        mpool->dashOutline[i].closed.reset();
    }

    return true;
}


bool mpoolTerm(SwMpool* mpool)
{
    if (!mpool) return false;

    mpoolClear(mpool);

    free(mpool->outline);
    free(mpool->strokeOutline);
    free(mpool->dashOutline);
    free(mpool);

    return true;
}
