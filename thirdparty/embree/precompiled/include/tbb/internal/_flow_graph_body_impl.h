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

#ifndef __TBB__flow_graph_body_impl_H
#define __TBB__flow_graph_body_impl_H

#include "tbb/internal/_template_helpers.h"

#ifndef __TBB_flow_graph_H
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

// included in namespace tbb::flow::interfaceX (in flow_graph.h)

namespace internal {

typedef tbb::internal::uint64_t tag_value;

using tbb::internal::strip;

#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT

template<typename ... Policies> struct Policy {};

template<typename ... Policies> struct has_policy;

template<typename ExpectedPolicy, typename FirstPolicy, typename ...Policies>
struct has_policy<ExpectedPolicy, FirstPolicy, Policies...> :
    tbb::internal::bool_constant<has_policy<ExpectedPolicy, FirstPolicy>::value ||
                                 has_policy<ExpectedPolicy, Policies...>::value> {};

template<typename ExpectedPolicy, typename SinglePolicy>
struct has_policy<ExpectedPolicy, SinglePolicy> :
    tbb::internal::bool_constant<tbb::internal::is_same_type<ExpectedPolicy, SinglePolicy>::value> {};

template<typename ExpectedPolicy, typename ...Policies>
struct has_policy<ExpectedPolicy, Policy<Policies...> > : has_policy<ExpectedPolicy, Policies...> {};

#else

template<typename P1, typename P2 = void> struct Policy {};

template<typename ExpectedPolicy, typename SinglePolicy>
struct has_policy : tbb::internal::bool_constant<tbb::internal::is_same_type<ExpectedPolicy, SinglePolicy>::value> {};

template<typename ExpectedPolicy, typename P>
struct has_policy<ExpectedPolicy, Policy<P> > : has_policy<ExpectedPolicy, P> {};

template<typename ExpectedPolicy, typename P1, typename P2>
struct has_policy<ExpectedPolicy, Policy<P1, P2> > :
    tbb::internal::bool_constant<has_policy<ExpectedPolicy, P1>::value || has_policy<ExpectedPolicy, P2>::value> {};

#endif

namespace graph_policy_namespace {

    struct rejecting { };
    struct reserving { };
    struct queueing  { };
    struct lightweight  { };

    // K == type of field used for key-matching.  Each tag-matching port will be provided
    // functor that, given an object accepted by the port, will return the
    /// field of type K being used for matching.
    template<typename K, typename KHash=tbb_hash_compare<typename strip<K>::type > >
    struct key_matching {
        typedef K key_type;
        typedef typename strip<K>::type base_key_type;
        typedef KHash hash_compare_type;
    };

    // old tag_matching join's new specifier
    typedef key_matching<tag_value> tag_matching;

    // Aliases for Policy combinations
    typedef interface11::internal::Policy<queueing, lightweight> queueing_lightweight;
    typedef interface11::internal::Policy<rejecting, lightweight>  rejecting_lightweight;

} // namespace graph_policy_namespace

// -------------- function_body containers ----------------------
//! A functor that takes no input and generates a value of type Output
template< typename Output >
class input_body : tbb::internal::no_assign {
public:
    virtual ~input_body() {}

#if TBB_DEPRECATED_INPUT_NODE_BODY
    virtual bool operator()(Output &output) = 0;
#else
    virtual Output operator()(flow_control& fc) = 0;
#endif
    virtual input_body* clone() = 0;
};

template <typename Body>
void check_input_node_body_input_type_impl(Body) {
    __TBB_STATIC_ASSERT((tbb::internal::is_same_type<typename tbb::internal::body_arg_detector<Body>::arg_type, flow_control&>::value),
        "TBB Warning: input_node body requirements have been changed."
        "To temporarily enforce deprecated API specify TBB_DEPRECATED_INPUT_NODE_BODY.");
}

template <typename Body>
void check_input_node_body_input_type(Body) {
    check_input_node_body_input_type_impl(&Body::operator());
}

template <typename ReturnType, typename T>
void check_input_node_body_input_type(ReturnType(*)(T)) {
    __TBB_STATIC_ASSERT((tbb::internal::is_same_type<T, flow_control&>::value),
        "TBB Warning: input_node body requirements have been changed."
        "To temporarily enforce deprecated API specify TBB_DEPRECATED_INPUT_NODE_BODY.");
}

//! The leaf for input_body
template< typename Output, typename Body>
class input_body_leaf : public input_body<Output> {
public:
    input_body_leaf( const Body &_body ) : body(_body) { }

#if TBB_DEPRECATED_INPUT_NODE_BODY
    bool operator()(Output &output) __TBB_override { return body( output ); }
#else
    Output operator()(flow_control& fc) __TBB_override {
        check_input_node_body_input_type(body);
        return body(fc);
    }
#endif
    input_body_leaf* clone() __TBB_override {
        return new input_body_leaf< Output, Body >(body);
    }
    Body get_body() { return body; }
private:
    Body body;
};

template< typename Output >
class source_body : tbb::internal::no_assign {
public:
    virtual ~source_body() {}
    virtual bool operator()(Output &output) = 0;
    virtual source_body* clone() = 0;
};

//! The leaf for source_body
template< typename Output, typename Body>
class source_body_leaf : public source_body<Output> {
public:
    source_body_leaf( const Body &_body ) : body(_body) { }

    bool operator()(Output &output) __TBB_override { return body( output ); }

    source_body_leaf* clone() __TBB_override {
        return new source_body_leaf< Output, Body >(body);
    }

    Body get_body() { return body; }
private:
    Body body;
};

//! A functor that takes an Input and generates an Output
template< typename Input, typename Output >
class function_body : tbb::internal::no_assign {
public:
    virtual ~function_body() {}
    virtual Output operator()(const Input &input) = 0;
    virtual function_body* clone() = 0;
};

//! the leaf for function_body
template <typename Input, typename Output, typename B>
class function_body_leaf : public function_body< Input, Output > {
public:
    function_body_leaf( const B &_body ) : body(_body) { }
    Output operator()(const Input &i) __TBB_override { return body(i); }
    B get_body() { return body; }
    function_body_leaf* clone() __TBB_override {
        return new function_body_leaf< Input, Output, B >(body);
    }
private:
    B body;
};

//! the leaf for function_body specialized for Input and output of continue_msg
template <typename B>
class function_body_leaf< continue_msg, continue_msg, B> : public function_body< continue_msg, continue_msg > {
public:
    function_body_leaf( const B &_body ) : body(_body) { }
    continue_msg operator()( const continue_msg &i ) __TBB_override {
        body(i);
        return i;
    }
    B get_body() { return body; }
    function_body_leaf* clone() __TBB_override {
        return new function_body_leaf< continue_msg, continue_msg, B >(body);
    }
private:
    B body;
};

//! the leaf for function_body specialized for Output of continue_msg
template <typename Input, typename B>
class function_body_leaf< Input, continue_msg, B> : public function_body< Input, continue_msg > {
public:
    function_body_leaf( const B &_body ) : body(_body) { }
    continue_msg operator()(const Input &i) __TBB_override {
        body(i);
        return continue_msg();
    }
    B get_body() { return body; }
    function_body_leaf* clone() __TBB_override {
        return new function_body_leaf< Input, continue_msg, B >(body);
    }
private:
    B body;
};

//! the leaf for function_body specialized for Input of continue_msg
template <typename Output, typename B>
class function_body_leaf< continue_msg, Output, B > : public function_body< continue_msg, Output > {
public:
    function_body_leaf( const B &_body ) : body(_body) { }
    Output operator()(const continue_msg &i) __TBB_override {
        return body(i);
    }
    B get_body() { return body; }
    function_body_leaf* clone() __TBB_override {
        return new function_body_leaf< continue_msg, Output, B >(body);
    }
private:
    B body;
};

//! function_body that takes an Input and a set of output ports
template<typename Input, typename OutputSet>
class multifunction_body : tbb::internal::no_assign {
public:
    virtual ~multifunction_body () {}
    virtual void operator()(const Input &/* input*/, OutputSet &/*oset*/) = 0;
    virtual multifunction_body* clone() = 0;
    virtual void* get_body_ptr() = 0;
};

//! leaf for multifunction.  OutputSet can be a std::tuple or a vector.
template<typename Input, typename OutputSet, typename B >
class multifunction_body_leaf : public multifunction_body<Input, OutputSet> {
public:
    multifunction_body_leaf(const B &_body) : body(_body) { }
    void operator()(const Input &input, OutputSet &oset) __TBB_override {
        body(input, oset); // body may explicitly put() to one or more of oset.
    }
    void* get_body_ptr() __TBB_override { return &body; }
    multifunction_body_leaf* clone() __TBB_override {
        return new multifunction_body_leaf<Input, OutputSet,B>(body);
    }

private:
    B body;
};

// ------ function bodies for hash_buffers and key-matching joins.

template<typename Input, typename Output>
class type_to_key_function_body : tbb::internal::no_assign {
    public:
        virtual ~type_to_key_function_body() {}
        virtual Output operator()(const Input &input) = 0;  // returns an Output
        virtual type_to_key_function_body* clone() = 0;
};

// specialization for ref output
template<typename Input, typename Output>
class type_to_key_function_body<Input,Output&> : tbb::internal::no_assign {
    public:
        virtual ~type_to_key_function_body() {}
        virtual const Output & operator()(const Input &input) = 0;  // returns a const Output&
        virtual type_to_key_function_body* clone() = 0;
};

template <typename Input, typename Output, typename B>
class type_to_key_function_body_leaf : public type_to_key_function_body<Input, Output> {
public:
    type_to_key_function_body_leaf( const B &_body ) : body(_body) { }
    Output operator()(const Input &i) __TBB_override { return body(i); }
    B get_body() { return body; }
    type_to_key_function_body_leaf* clone() __TBB_override {
        return new type_to_key_function_body_leaf< Input, Output, B>(body);
    }
private:
    B body;
};

template <typename Input, typename Output, typename B>
class type_to_key_function_body_leaf<Input,Output&,B> : public type_to_key_function_body< Input, Output&> {
public:
    type_to_key_function_body_leaf( const B &_body ) : body(_body) { }
    const Output& operator()(const Input &i) __TBB_override {
        return body(i);
    }
    B get_body() { return body; }
    type_to_key_function_body_leaf* clone() __TBB_override {
        return new type_to_key_function_body_leaf< Input, Output&, B>(body);
    }
private:
    B body;
};

// --------------------------- end of function_body containers ------------------------

// --------------------------- node task bodies ---------------------------------------

//! A task that calls a node's forward_task function
template< typename NodeType >
class forward_task_bypass : public graph_task {

    NodeType &my_node;

public:

    forward_task_bypass( NodeType &n
#if __TBB_PREVIEW_FLOW_GRAPH_PRIORITIES
                         , node_priority_t node_priority = no_priority
    ) : graph_task(node_priority),
#else
    ) :
#endif
    my_node(n) {}

    task *execute() __TBB_override {
        task * new_task = my_node.forward_task();
        if (new_task == SUCCESSFULLY_ENQUEUED) new_task = NULL;
        return new_task;
    }
};

//! A task that calls a node's apply_body_bypass function, passing in an input of type Input
//  return the task* unless it is SUCCESSFULLY_ENQUEUED, in which case return NULL
template< typename NodeType, typename Input >
class apply_body_task_bypass : public graph_task {

    NodeType &my_node;
    Input my_input;

public:

    apply_body_task_bypass( NodeType &n, const Input &i
#if __TBB_PREVIEW_FLOW_GRAPH_PRIORITIES
                            , node_priority_t node_priority = no_priority
    ) : graph_task(node_priority),
#else
    ) :
#endif
        my_node(n), my_input(i) {}

    task *execute() __TBB_override {
        task * next_task = my_node.apply_body_bypass( my_input );
        if(next_task == SUCCESSFULLY_ENQUEUED) next_task = NULL;
        return next_task;
    }
};

//! A task that calls a node's apply_body_bypass function with no input
template< typename NodeType >
class source_task_bypass : public graph_task {

    NodeType &my_node;

public:

    source_task_bypass( NodeType &n ) : my_node(n) {}

    task *execute() __TBB_override {
        task *new_task = my_node.apply_body_bypass( );
        if(new_task == SUCCESSFULLY_ENQUEUED) return NULL;
        return new_task;
    }
};

// ------------------------ end of node task bodies -----------------------------------

//! An empty functor that takes an Input and returns a default constructed Output
template< typename Input, typename Output >
struct empty_body {
    Output operator()( const Input & ) const { return Output(); }
};

template<typename T, typename DecrementType, typename DummyType = void>
class decrementer;

template<typename T, typename DecrementType>
class decrementer<T, DecrementType,
                  typename tbb::internal::enable_if<
                      tbb::internal::is_integral<DecrementType>::value, void>::type
                  > : public receiver<DecrementType>, tbb::internal::no_copy {
    T* my_node;
protected:

    task* try_put_task( const DecrementType& value ) __TBB_override {
        task* result = my_node->decrement_counter( value );
        if( !result )
            result = SUCCESSFULLY_ENQUEUED;
        return result;
    }

    graph& graph_reference() const __TBB_override {
        return my_node->my_graph;
    }

    template<typename U, typename V> friend class tbb::flow::interface11::limiter_node;
    void reset_receiver( reset_flags f ) __TBB_override {
#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
        if (f & rf_clear_edges)
            my_built_predecessors.clear();
#else
        tbb::internal::suppress_unused_warning( f );
#endif
    }

public:
    // Since decrementer does not make use of possibly unconstructed owner inside its
    // constructor, my_node can be directly initialized with 'this' pointer passed from the
    // owner, hence making method 'set_owner' needless.
    decrementer() : my_node(NULL) {}
    void set_owner( T *node ) { my_node = node; }

#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
    spin_mutex my_mutex;
    //! The predecessor type for this node
    typedef typename receiver<DecrementType>::predecessor_type predecessor_type;

    typedef internal::edge_container<predecessor_type> built_predecessors_type;
    typedef typename built_predecessors_type::edge_list_type predecessor_list_type;
    built_predecessors_type &built_predecessors() __TBB_override { return my_built_predecessors; }

    void internal_add_built_predecessor( predecessor_type &s) __TBB_override {
        spin_mutex::scoped_lock l(my_mutex);
        my_built_predecessors.add_edge( s );
    }

    void internal_delete_built_predecessor( predecessor_type &s) __TBB_override {
        spin_mutex::scoped_lock l(my_mutex);
        my_built_predecessors.delete_edge(s);
    }

    void copy_predecessors( predecessor_list_type &v) __TBB_override {
        spin_mutex::scoped_lock l(my_mutex);
        my_built_predecessors.copy_edges(v);
    }

    size_t predecessor_count() __TBB_override {
        spin_mutex::scoped_lock l(my_mutex);
        return my_built_predecessors.edge_count();
    }
protected:
    built_predecessors_type my_built_predecessors;
#endif  /* TBB_DEPRECATED_FLOW_NODE_EXTRACTION */
};

template<typename T>
class decrementer<T, continue_msg, void> : public continue_receiver, tbb::internal::no_copy {

    T *my_node;

    task *execute() __TBB_override {
        return my_node->decrement_counter( 1 );
    }

protected:

    graph& graph_reference() const __TBB_override {
        return my_node->my_graph;
    }

public:

    typedef continue_msg input_type;
    typedef continue_msg output_type;
    decrementer( int number_of_predecessors = 0 )
        : continue_receiver(
            __TBB_FLOW_GRAPH_PRIORITY_ARG1(number_of_predecessors, tbb::flow::internal::no_priority)
        )
          // Since decrementer does not make use of possibly unconstructed owner inside its
          // constructor, my_node can be directly initialized with 'this' pointer passed from the
          // owner, hence making method 'set_owner' needless.
        , my_node(NULL)
    {}
    void set_owner( T *node ) { my_node = node; }
};

} // namespace internal

#endif // __TBB__flow_graph_body_impl_H

