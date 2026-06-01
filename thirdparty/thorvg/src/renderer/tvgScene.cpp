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

#include "tvgScene.h"


Scene::Scene() = default;


Scene* Scene::gen() noexcept
{
    return new SceneImpl;
}


Type Scene::type() const noexcept
{
    return Type::Scene;
}


Result Scene::add(Paint* target, Paint* at) noexcept
{
    return to<SceneImpl>(this)->insert(target, at);
}


Result Scene::remove(Paint* paint) noexcept
{
    if (paint) return to<SceneImpl>(this)->remove(paint);
    else return to<SceneImpl>(this)->clearPaints();
}


const list<Paint*>& Scene::paints() const noexcept
{
    return to<SceneImpl>(this)->paints;
}


Result Scene::add(SceneEffect effect, ...) noexcept
{
    va_list args;
    va_start(args, effect);
    auto ret = to<SceneImpl>(this)->add(effect, args);
    va_end(args);
    return ret;
}
