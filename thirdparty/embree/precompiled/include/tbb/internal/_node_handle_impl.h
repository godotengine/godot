/*
    Copyright (c) 2019-2020 Intel Corporation

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

#ifndef __TBB_node_handle_H
#define __TBB_node_handle_H

#include "_allocator_traits.h"
#include "../tbb_config.h"


namespace tbb {

// This classes must be declared here for correct friendly relationship
// TODO: Consider creation some internal class to access node_handle private fields without any friendly classes
namespace interface5 {
namespace internal {
    template <typename T, typename Allocator>
    class split_ordered_list;
    template <typename Traits>
    class concurrent_unordered_base;
}
}

namespace interface10{
namespace internal {
    template<typename Traits>
    class concurrent_skip_list;
}
}

namespace internal {

template<typename Value, typename Node, typename Allocator>
class node_handle_base {
public:
    typedef Allocator allocator_type;
protected:
    typedef Node node;
    typedef tbb::internal::allocator_traits<allocator_type> traits_type;
public:

    node_handle_base() : my_node(NULL), my_allocator() {}
    node_handle_base(node_handle_base&& nh) : my_node(nh.my_node),
                                              my_allocator(std::move(nh.my_allocator)) {
        nh.my_node = NULL;
    }

    bool empty() const { return my_node == NULL; }
    explicit operator bool() const { return my_node != NULL; }

    ~node_handle_base() { internal_destroy(); }

    node_handle_base& operator=(node_handle_base&& nh) {
        internal_destroy();
        my_node = nh.my_node;
        typedef typename traits_type::propagate_on_container_move_assignment pocma_type;
        tbb::internal::allocator_move_assignment(my_allocator, nh.my_allocator, pocma_type());
        nh.deactivate();
        return *this;
    }

    void swap(node_handle_base& nh) {
        std::swap(my_node, nh.my_node);
        typedef typename traits_type::propagate_on_container_swap pocs_type;
        tbb::internal::allocator_swap(my_allocator, nh.my_allocator, pocs_type());
    }

    allocator_type get_allocator() const {
        return my_allocator;
    }

protected:
    node_handle_base(node* n) : my_node(n) {}

    void internal_destroy() {
        if(my_node) {
            traits_type::destroy(my_allocator, my_node->storage());
            typename tbb::internal::allocator_rebind<allocator_type, node>::type node_allocator;
            node_allocator.deallocate(my_node, 1);
        }
    }

    void deactivate() { my_node = NULL; }

    node* my_node;
    allocator_type my_allocator;
};

// node handle for maps
template<typename Key, typename Value, typename Node, typename Allocator>
class node_handle : public node_handle_base<Value, Node, Allocator> {
    typedef node_handle_base<Value, Node, Allocator> base_type;
public:
    typedef Key key_type;
    typedef typename Value::second_type mapped_type;
    typedef typename base_type::allocator_type allocator_type;

    node_handle() : base_type() {}

    key_type& key() const {
        __TBB_ASSERT(!this->empty(), "Cannot get key from the empty node_type object");
        return *const_cast<key_type*>(&(this->my_node->value().first));
    }

    mapped_type& mapped() const {
        __TBB_ASSERT(!this->empty(), "Cannot get mapped value from the empty node_type object");
        return this->my_node->value().second;
    }

private:
    template<typename T, typename A>
    friend class tbb::interface5::internal::split_ordered_list;

    template<typename Traits>
    friend class tbb::interface5::internal::concurrent_unordered_base;

    template<typename Traits>
    friend class tbb::interface10::internal::concurrent_skip_list;

    node_handle(typename base_type::node* n) : base_type(n) {}
};

// node handle for sets
template<typename Key, typename Node, typename Allocator>
class node_handle<Key, Key, Node, Allocator> : public node_handle_base<Key, Node, Allocator> {
    typedef node_handle_base<Key, Node, Allocator> base_type;
public:
    typedef Key value_type;
    typedef typename base_type::allocator_type allocator_type;

    node_handle() : base_type() {}

    value_type& value() const {
        __TBB_ASSERT(!this->empty(), "Cannot get value from the empty node_type object");
        return *const_cast<value_type*>(&(this->my_node->value()));
    }

private:
    template<typename T, typename A>
    friend class tbb::interface5::internal::split_ordered_list;

    template<typename Traits>
    friend class tbb::interface5::internal::concurrent_unordered_base;

    template<typename Traits>
    friend class tbb::interface10::internal::concurrent_skip_list;

    node_handle(typename base_type::node* n) : base_type(n) {}
};


}// namespace internal
}// namespace tbb

#endif /*__TBB_node_handle_H*/
