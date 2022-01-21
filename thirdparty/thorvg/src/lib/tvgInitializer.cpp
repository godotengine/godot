/*
 * Copyright (c) 2020-2021 Samsung Electronics Co., Ltd. All rights reserved.

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
#include "tvgCommon.h"
#include "tvgTaskScheduler.h"
#include "tvgLoader.h"

#ifdef _WIN32
    #include <cstring>
#endif

#ifdef THORVG_SW_RASTER_SUPPORT
    #include "tvgSwRenderer.h"
#endif

#ifdef THORVG_GL_RASTER_SUPPORT
    #include "tvgGlRenderer.h"
#endif


/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

static int _initCnt = 0;
static uint16_t _version = 0;


static bool _buildVersionInfo()
{
    auto SRC = THORVG_VERSION_STRING;   //ex) 0.3.99
    auto p = SRC;
    const char* x;

    char major[3];
    x = strchr(p, '.');
    if (!x) return false;
    memcpy(major, p, x - p);
    major[x - p] = '\0';
    p = x + 1;

    char minor[3];
    x = strchr(p, '.');
    if (!x) return false;
    memcpy(minor, p, x - p);
    minor[x - p] = '\0';
    p = x + 1;

    char micro[3];
    x = SRC + strlen(THORVG_VERSION_STRING);
    memcpy(micro, p, x - p);
    micro[x - p] = '\0';

    char sum[7];
    snprintf(sum, sizeof(sum), "%s%s%s", major, minor, micro);

    _version = atoi(sum);

    return true;
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

Result Initializer::init(CanvasEngine engine, uint32_t threads) noexcept
{
    auto nonSupport = true;

    if (static_cast<uint32_t>(engine) & static_cast<uint32_t>(CanvasEngine::Sw)) {
        #ifdef THORVG_SW_RASTER_SUPPORT
            if (!SwRenderer::init(threads)) return Result::FailedAllocation;
            nonSupport = false;
        #endif
    } else if (static_cast<uint32_t>(engine) & static_cast<uint32_t>(CanvasEngine::Gl)) {
        #ifdef THORVG_GL_RASTER_SUPPORT
            if (!GlRenderer::init(threads)) return Result::FailedAllocation;
            nonSupport = false;
        #endif
    } else {
        return Result::InvalidArguments;
    }

    if (nonSupport) return Result::NonSupport;

    if (_initCnt++ > 0) return Result::Success;

    if (!_buildVersionInfo()) return Result::Unknown;

    if (!LoaderMgr::init()) return Result::Unknown;

    TaskScheduler::init(threads);

    return Result::Success;
}


Result Initializer::term(CanvasEngine engine) noexcept
{
    if (_initCnt == 0) return Result::InsufficientCondition;

    auto nonSupport = true;

    if (static_cast<uint32_t>(engine) & static_cast<uint32_t>(CanvasEngine::Sw)) {
        #ifdef THORVG_SW_RASTER_SUPPORT
            if (!SwRenderer::term()) return Result::InsufficientCondition;
            nonSupport = false;
        #endif
    } else if (static_cast<uint32_t>(engine) & static_cast<uint32_t>(CanvasEngine::Gl)) {
        #ifdef THORVG_GL_RASTER_SUPPORT
            if (!GlRenderer::term()) return Result::InsufficientCondition;
            nonSupport = false;
        #endif
    } else {
        return Result::InvalidArguments;
    }

    if (nonSupport) return Result::NonSupport;

    if (--_initCnt > 0) return Result::Success;

    TaskScheduler::term();

    if (!LoaderMgr::term()) return Result::Unknown;

    return Result::Success;
}


uint16_t THORVG_VERSION_NUMBER()
{
    return _version;
}
