// Copyright (c) 2017-2025 The Khronos Group Inc.
// Copyright (c) 2017-2019 Valve Corporation
// Copyright (c) 2017-2019 LunarG, Inc.
// Copyright (c) 2019 Collabora, Ltd.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Initial Author: Rylie Pavlik <rylie.pavlik@collabora.com>
//

/*!
 * @file
 *
 * Additional functions along the lines of the standard library algorithms.
 */

#pragma once

#include <algorithm>
#include <vector>

/// Like std::remove_if, except it works on associative containers and it actually removes this.
///
/// The iterator stuff in here is subtle - .erase() invalidates only that iterator, but it returns a non-invalidated iterator to the
/// next valid element which we can use instead of incrementing.
template <typename T, typename Pred>
static inline void map_erase_if(T &container, Pred &&predicate) {
    for (auto it = container.begin(); it != container.end();) {
        if (predicate(*it)) {
            it = container.erase(it);
        } else {
            ++it;
        }
    }
}

/*!
 * Moves all elements matching the predicate to the end of the vector then erases them.
 *
 * Combines the two parts of the erase-remove idiom to simplify things and avoid accidentally using the wrong erase overload.
 */
template <typename T, typename Alloc, typename Pred>
static inline void vector_remove_if_and_erase(std::vector<T, Alloc> &vec, Pred &&predicate) {
    auto b = vec.begin();
    auto e = vec.end();
    vec.erase(std::remove_if(b, e, std::forward<Pred>(predicate)), e);
}
