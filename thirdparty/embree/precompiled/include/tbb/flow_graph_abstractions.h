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

#ifndef __TBB_flow_graph_abstractions_H
#define __TBB_flow_graph_abstractions_H

namespace tbb {
namespace flow {
namespace interface11 {

//! Pure virtual template classes that define interfaces for async communication
class graph_proxy {
public:
    //! Inform a graph that messages may come from outside, to prevent premature graph completion
    virtual void reserve_wait() = 0;

    //! Inform a graph that a previous call to reserve_wait is no longer in effect
    virtual void release_wait() = 0;

    virtual ~graph_proxy() {}
};

template <typename Input>
class receiver_gateway : public graph_proxy {
public:
    //! Type of inputing data into FG.
    typedef Input input_type;

    //! Submit signal from an asynchronous activity to FG.
    virtual bool try_put(const input_type&) = 0;
};

} //interfaceX

using interface11::graph_proxy;
using interface11::receiver_gateway;

} //flow
} //tbb
#endif
