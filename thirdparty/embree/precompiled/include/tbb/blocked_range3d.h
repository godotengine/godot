/*
    Copyright (c) 2005-2020 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#ifndef __TBB_blocked_range3d_H
#define __TBB_blocked_range3d_H

#include "tbb_stddef.h"
#include "blocked_range.h"

namespace tbb {

//! A 3-dimensional range that models the Range concept.
/** @ingroup algorithms */
template<typename PageValue, typename RowValue=PageValue, typename ColValue=RowValue>
class blocked_range3d {
public:
    //! Type for size of an iteration range
    typedef blocked_range<PageValue> page_range_type;
    typedef blocked_range<RowValue>  row_range_type;
    typedef blocked_range<ColValue>  col_range_type;

private:
    page_range_type my_pages;
    row_range_type  my_rows;
    col_range_type  my_cols;

public:

    blocked_range3d( PageValue page_begin, PageValue page_end,
                     RowValue  row_begin,  RowValue row_end,
                     ColValue  col_begin,  ColValue col_end ) :
        my_pages(page_begin,page_end),
        my_rows(row_begin,row_end),
        my_cols(col_begin,col_end)
    {}

    blocked_range3d( PageValue page_begin, PageValue page_end, typename page_range_type::size_type page_grainsize,
                     RowValue  row_begin,  RowValue row_end,   typename row_range_type::size_type row_grainsize,
                     ColValue  col_begin,  ColValue col_end,   typename col_range_type::size_type col_grainsize ) :
        my_pages(page_begin,page_end,page_grainsize),
        my_rows(row_begin,row_end,row_grainsize),
        my_cols(col_begin,col_end,col_grainsize)
    {}

    //! True if range is empty
    bool empty() const {
        // Range is empty if at least one dimension is empty.
        return my_pages.empty() || my_rows.empty() || my_cols.empty();
    }

    //! True if range is divisible into two pieces.
    bool is_divisible() const {
        return  my_pages.is_divisible() || my_rows.is_divisible() || my_cols.is_divisible();
    }

    blocked_range3d( blocked_range3d& r, split ) :
        my_pages(r.my_pages),
        my_rows(r.my_rows),
        my_cols(r.my_cols)
    {
        split split_obj;
        do_split(r, split_obj);
    }

#if __TBB_USE_PROPORTIONAL_SPLIT_IN_BLOCKED_RANGES
    //! Static field to support proportional split
    static const bool is_splittable_in_proportion = true;

    blocked_range3d( blocked_range3d& r, proportional_split& proportion ) :
        my_pages(r.my_pages),
        my_rows(r.my_rows),
        my_cols(r.my_cols)
    {
        do_split(r, proportion);
    }
#endif /* __TBB_USE_PROPORTIONAL_SPLIT_IN_BLOCKED_RANGES */

    //! The pages of the iteration space
    const page_range_type& pages() const {return my_pages;}

    //! The rows of the iteration space
    const row_range_type& rows() const {return my_rows;}

    //! The columns of the iteration space
    const col_range_type& cols() const {return my_cols;}

private:

    template <typename Split>
    void do_split( blocked_range3d& r, Split& split_obj)
    {
        if ( my_pages.size()*double(my_rows.grainsize()) < my_rows.size()*double(my_pages.grainsize()) ) {
            if ( my_rows.size()*double(my_cols.grainsize()) < my_cols.size()*double(my_rows.grainsize()) ) {
                my_cols.my_begin = col_range_type::do_split(r.my_cols, split_obj);
            } else {
                my_rows.my_begin = row_range_type::do_split(r.my_rows, split_obj);
            }
        } else {
            if ( my_pages.size()*double(my_cols.grainsize()) < my_cols.size()*double(my_pages.grainsize()) ) {
                my_cols.my_begin = col_range_type::do_split(r.my_cols, split_obj);
            } else {
                my_pages.my_begin = page_range_type::do_split(r.my_pages, split_obj);
            }
        }
    }
};

} // namespace tbb

#endif /* __TBB_blocked_range3d_H */
