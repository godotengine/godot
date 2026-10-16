/*
 * Copyright (c) 2021 - 2026 ThorVG project. All rights reserved.

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

#include "tvgAccessor.h"
#include "tvgCompressor.h"
#include "tvgPicture.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

#define IMPL static_cast<AccessorImpl*>(this)

Accessor::Accessor() = default;

struct AccessorImpl : Accessor
{
    Paint* paint;
    AccessorCallback cb;
    bool accessible = false;  // special searching for picture nested scenes with id maps

    bool access(AccessorIterator* it)
    {
        while (auto child = it->next()) {
            // access the child
            if (!cb.func(child, cb.data)) return false;

            // access the children of the child
            if (auto it2 = AccessorIterator::iterator(child)) {
                if (!access(it2)) {
                    delete (it2);
                    return false;
                }
                delete(it2);
            }
        }
        return true;
    }

    // pre-order tree search
    void search()
    {
        // root
        if (!cb.func(paint, cb.data)) return;

        // children
        if (auto it = AccessorIterator::iterator(paint)) {
            access(it);
            delete (it);
        }
    }
};

/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

Result Accessor::set(Paint* paint, function<bool(const Paint* paint, void* data)> func, void* data) noexcept
{
    if (paint && func) {
        IMPL->paint = paint;
        IMPL->cb = {func, data};
        IMPL->accessible = (paint->type() == Type::Picture && static_cast<Picture*>(paint)->accessible);

        paint->ref();

        if (IMPL->accessible) to<PictureImpl>(IMPL->paint)->access(IMPL->cb);
        else IMPL->search();

        paint->unref(false);

        IMPL->accessible = false;

        return Result::Success;
    }
    return Result::InvalidArguments;
}

const char* Accessor::name(uint32_t id) noexcept
{
    if (IMPL->accessible) {
        auto entity = to<PictureImpl>(IMPL->paint)->access(id);
        if (entity) return entity->name;
    } else TVGLOG("RENDERER", "Did you enable Picture::accessible?");
    return nullptr;
}

uint32_t Accessor::id(const char* name) noexcept
{
    return djb2Encode(name);
}

Accessor::~Accessor()
{
}

Accessor* Accessor::gen() noexcept
{
    return new AccessorImpl;
}
