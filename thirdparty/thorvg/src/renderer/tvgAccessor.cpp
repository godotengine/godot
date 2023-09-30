/*
 * Copyright (c) 2021 - 2023 the ThorVG project. All rights reserved.

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

#include "tvgIteratorAccessor.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

static bool accessChildren(Iterator* it, function<bool(const Paint* paint)> func)
{
    while (auto child = it->next()) {
        //Access the child
        if (!func(child)) return false;

        //Access the children of the child
        if (auto it2 = IteratorAccessor::iterator(child)) {
            if (!accessChildren(it2, func)) {
                delete(it2);
                return false;
            }
            delete(it2);
        }
    }
    return true;
}

/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

unique_ptr<Picture> Accessor::set(unique_ptr<Picture> picture, function<bool(const Paint* paint)> func) noexcept
{
    auto p = picture.get();
    if (!p || !func) return picture;

    //Use the Preorder Tree-Search

    //Root
    if (!func(p)) return picture;

    //Children
    if (auto it = IteratorAccessor::iterator(p)) {
        accessChildren(it, func);
        delete(it);
    }
    return picture;
}


Accessor::~Accessor()
{

}


Accessor::Accessor() : pImpl(nullptr)
{

}


unique_ptr<Accessor> Accessor::gen() noexcept
{
    return unique_ptr<Accessor>(new Accessor);
}
