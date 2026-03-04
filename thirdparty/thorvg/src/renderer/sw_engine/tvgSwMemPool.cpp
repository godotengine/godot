/*
 * Copyright (c) 2020 - 2026 ThorVG project. All rights reserved.

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
/* External Class Implementation                                        */
/************************************************************************/

SwOutline* mpoolReqOutline(SwMpool* mpool, unsigned idx)
{
    mpool->outline[idx].pts.clear();
    mpool->outline[idx].cntrs.clear();
    mpool->outline[idx].types.clear();
    mpool->outline[idx].closed.clear();

    return &mpool->outline[idx];
}


SwStrokeBorder* mpoolReqStrokeLBorder(SwMpool* mpool, unsigned idx)
{
    mpool->leftBorder[idx].pts.clear();
    mpool->leftBorder[idx].start = -1;
    return &mpool->leftBorder[idx];
}


SwStrokeBorder* mpoolReqStrokeRBorder(SwMpool* mpool, unsigned idx)
{
    mpool->rightBorder[idx].pts.clear();
    mpool->rightBorder[idx].start = -1;
    return &mpool->rightBorder[idx];
}


SwCellPool* mpoolReqCellPool(SwMpool* mpool, unsigned idx)
{
    return &mpool->cellPool[idx];
}


SwMpool* mpoolInit(uint32_t threads)
{
    auto allocSize = threads + 1;

    auto mpool = tvg::malloc<SwMpool>(sizeof(SwMpool));
    mpool->outline = new SwOutline[allocSize];
    mpool->leftBorder = new SwStrokeBorder[allocSize];
    mpool->rightBorder = new SwStrokeBorder[allocSize];
    mpool->cellPool = new SwCellPool[allocSize];

    mpool->allocSize = allocSize;

    return mpool;
}


void mpoolTerm(SwMpool* mpool)
{
    if (!mpool) return;

    delete[](mpool->outline);
    delete[](mpool->leftBorder);
    delete[](mpool->rightBorder);
    delete[](mpool->cellPool);

    tvg::free(mpool);
}
