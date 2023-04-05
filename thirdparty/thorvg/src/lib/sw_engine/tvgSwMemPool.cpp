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
    mpool->outline[idx].cntrsCnt = 0;
    mpool->outline[idx].ptsCnt = 0;
}


SwOutline* mpoolReqStrokeOutline(SwMpool* mpool, unsigned idx)
{
    return &mpool->strokeOutline[idx];
}


void mpoolRetStrokeOutline(SwMpool* mpool, unsigned idx)
{
    mpool->strokeOutline[idx].cntrsCnt = 0;
    mpool->strokeOutline[idx].ptsCnt = 0;
}


SwMpool* mpoolInit(unsigned threads)
{
    if (threads == 0) threads = 1;

    auto mpool = static_cast<SwMpool*>(calloc(sizeof(SwMpool), 1));
    mpool->outline = static_cast<SwOutline*>(calloc(1, sizeof(SwOutline) * threads));
    if (!mpool->outline) goto err;

    mpool->strokeOutline = static_cast<SwOutline*>(calloc(1, sizeof(SwOutline) * threads));
    if (!mpool->strokeOutline) goto err;

    mpool->allocSize = threads;

    return mpool;

err:
    if (mpool->outline) {
        free(mpool->outline);
        mpool->outline = nullptr;
    }

    if (mpool->strokeOutline) {
        free(mpool->strokeOutline);
        mpool->strokeOutline = nullptr;
    }
    free(mpool);
    return nullptr;
}


bool mpoolClear(SwMpool* mpool)
{
    SwOutline* p;

    for (unsigned i = 0; i < mpool->allocSize; ++i) {

        //Outline
        p = &mpool->outline[i];

        free(p->cntrs);
        p->cntrs = nullptr;

        free(p->pts);
        p->pts = nullptr;

        free(p->types);
        p->types = nullptr;

        free(p->closed);
        p->closed = nullptr;

        p->cntrsCnt = p->reservedCntrsCnt = 0;
        p->ptsCnt = p->reservedPtsCnt = 0;

        //StrokeOutline
        p = &mpool->strokeOutline[i];

        free(p->cntrs);
        p->cntrs = nullptr;

        free(p->pts);
        p->pts = nullptr;

        free(p->types);
        p->types = nullptr;

        free(p->closed);
        p->closed = nullptr;

        p->cntrsCnt = p->reservedCntrsCnt = 0;
        p->ptsCnt = p->reservedPtsCnt = 0;
    }

    return true;
}


bool mpoolTerm(SwMpool* mpool)
{
    if (!mpool) return false;

    mpoolClear(mpool);

    if (mpool->outline) {
        free(mpool->outline);
        mpool->outline = nullptr;
    }

    if (mpool->strokeOutline) {
        free(mpool->strokeOutline);
        mpool->strokeOutline = nullptr;
    }

    free(mpool);

    return true;
}
