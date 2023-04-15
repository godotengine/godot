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

#ifndef __TBB__flow_graph_join_impl_H
#define __TBB__flow_graph_join_impl_H

#ifndef __TBB_flow_graph_H
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

namespace internal {

    struct forwarding_base : tbb::internal::no_assign {
        forwarding_base(graph &g) : graph_ref(g) {}
        virtual ~forwarding_base() {}
        // decrement_port_count may create a forwarding task.  If we cannot handle the task
        // ourselves, ask decrement_port_count to deal with it.
        virtual task * decrement_port_count(bool handle_task) = 0;
        virtual void increment_port_count() = 0;
        // moved here so input ports can queue tasks
        graph& graph_ref;
    };

    // specialization that lets us keep a copy of the current_key for building results.
    // KeyType can be a reference type.
    template<typename KeyType>
    struct matching_forwarding_base : public forwarding_base {
        typedef typename tbb::internal::strip<KeyType>::type current_key_type;
        matching_forwarding_base(graph &g) : forwarding_base(g) { }
        virtual task * increment_key_count(current_key_type const & /*t*/, bool /*handle_task*/) = 0; // {return NULL;}
        current_key_type current_key; // so ports can refer to FE's desired items
    };

    template< int N >
    struct join_helper {

        template< typename TupleType, typename PortType >
        static inline void set_join_node_pointer(TupleType &my_input, PortType *port) {
            tbb::flow::get<N-1>( my_input ).set_join_node_pointer(port);
            join_helper<N-1>::set_join_node_pointer( my_input, port );
        }
        template< typename TupleType >
        static inline void consume_reservations( TupleType &my_input ) {
            tbb::flow::get<N-1>( my_input ).consume();
            join_helper<N-1>::consume_reservations( my_input );
        }

        template< typename TupleType >
        static inline void release_my_reservation( TupleType &my_input ) {
            tbb::flow::get<N-1>( my_input ).release();
        }

        template <typename TupleType>
        static inline void release_reservations( TupleType &my_input) {
            join_helper<N-1>::release_reservations(my_input);
            release_my_reservation(my_input);
        }

        template< typename InputTuple, typename OutputTuple >
        static inline bool reserve( InputTuple &my_input, OutputTuple &out) {
            if ( !tbb::flow::get<N-1>( my_input ).reserve( tbb::flow::get<N-1>( out ) ) ) return false;
            if ( !join_helper<N-1>::reserve( my_input, out ) ) {
                release_my_reservation( my_input );
                return false;
            }
            return true;
        }

        template<typename InputTuple, typename OutputTuple>
        static inline bool get_my_item( InputTuple &my_input, OutputTuple &out) {
            bool res = tbb::flow::get<N-1>(my_input).get_item(tbb::flow::get<N-1>(out) ); // may fail
            return join_helper<N-1>::get_my_item(my_input, out) && res;       // do get on other inputs before returning
        }

        template<typename InputTuple, typename OutputTuple>
        static inline bool get_items(InputTuple &my_input, OutputTuple &out) {
            return get_my_item(my_input, out);
        }

        template<typename InputTuple>
        static inline void reset_my_port(InputTuple &my_input) {
            join_helper<N-1>::reset_my_port(my_input);
            tbb::flow::get<N-1>(my_input).reset_port();
        }

        template<typename InputTuple>
        static inline void reset_ports(InputTuple& my_input) {
            reset_my_port(my_input);
        }

        template<typename InputTuple, typename KeyFuncTuple>
        static inline void set_key_functors(InputTuple &my_input, KeyFuncTuple &my_key_funcs) {
            tbb::flow::get<N-1>(my_input).set_my_key_func(tbb::flow::get<N-1>(my_key_funcs));
            tbb::flow::get<N-1>(my_key_funcs) = NULL;
            join_helper<N-1>::set_key_functors(my_input, my_key_funcs);
        }

        template< typename KeyFuncTuple>
        static inline void copy_key_functors(KeyFuncTuple &my_inputs, KeyFuncTuple &other_inputs) {
            if(tbb::flow::get<N-1>(other_inputs).get_my_key_func()) {
                tbb::flow::get<N-1>(my_inputs).set_my_key_func(tbb::flow::get<N-1>(other_inputs).get_my_key_func()->clone());
            }
            join_helper<N-1>::copy_key_functors(my_inputs, other_inputs);
        }

        template<typename InputTuple>
        static inline void reset_inputs(InputTuple &my_input, reset_flags f) {
            join_helper<N-1>::reset_inputs(my_input, f);
            tbb::flow::get<N-1>(my_input).reset_receiver(f);
        }

#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
        template<typename InputTuple>
        static inline void extract_inputs(InputTuple &my_input) {
            join_helper<N-1>::extract_inputs(my_input);
            tbb::flow::get<N-1>(my_input).extract_receiver();
        }
#endif
    };  // join_helper<N>

    template< >
    struct join_helper<1> {

        template< typename TupleType, typename PortType >
        static inline void set_join_node_pointer(TupleType &my_input, PortType *port) {
            tbb::flow::get<0>( my_input ).set_join_node_pointer(port);
        }

        template< typename TupleType >
        static inline void consume_reservations( TupleType &my_input ) {
            tbb::flow::get<0>( my_input ).consume();
        }

        template< typename TupleType >
        static inline void release_my_reservation( TupleType &my_input ) {
            tbb::flow::get<0>( my_input ).release();
        }

        template<typename TupleType>
        static inline void release_reservations( TupleType &my_input) {
            release_my_reservation(my_input);
        }

        template< typename InputTuple, typename OutputTuple >
        static inline bool reserve( InputTuple &my_input, OutputTuple &out) {
            return tbb::flow::get<0>( my_input ).reserve( tbb::flow::get<0>( out ) );
        }

        template<typename InputTuple, typename OutputTuple>
        static inline bool get_my_item( InputTuple &my_input, OutputTuple &out) {
            return tbb::flow::get<0>(my_input).get_item(tbb::flow::get<0>(out));
        }

        template<typename InputTuple, typename OutputTuple>
        static inline bool get_items(InputTuple &my_input, OutputTuple &out) {
            return get_my_item(my_input, out);
        }

        template<typename InputTuple>
        static inline void reset_my_port(InputTuple &my_input) {
            tbb::flow::get<0>(my_input).reset_port();
        }

        template<typename InputTuple>
        static inline void reset_ports(InputTuple& my_input) {
            reset_my_port(my_input);
        }

        template<typename InputTuple, typename KeyFuncTuple>
        static inline void set_key_functors(InputTuple &my_input, KeyFuncTuple &my_key_funcs) {
            tbb::flow::get<0>(my_input).set_my_key_func(tbb::flow::get<0>(my_key_funcs));
            tbb::flow::get<0>(my_key_funcs) = NULL;
        }

        template< typename KeyFuncTuple>
        static inline void copy_key_functors(KeyFuncTuple &my_inputs, KeyFuncTuple &other_inputs) {
            if(tbb::flow::get<0>(other_inputs).get_my_key_func()) {
                tbb::flow::get<0>(my_inputs).set_my_key_func(tbb::flow::get<0>(other_inputs).get_my_key_func()->clone());
            }
        }
        template<typename InputTuple>
        static inline void reset_inputs(InputTuple &my_input, reset_flags f) {
            tbb::flow::get<0>(my_input).reset_receiver(f);
        }

#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
        template<typename InputTuple>
        static inline void extract_inputs(InputTuple &my_input) {
            tbb::flow::get<0>(my_input).extract_receiver();
        }
#endif
    };  // join_helper<1>

    //! The two-phase join port
    template< typename T >
    class reserving_port : public receiver<T> {
    public:
        typedef T input_type;
        typedef typename receiver<input_type>::predecessor_type predecessor_type;
#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
        typedef typename receiver<input_type>::predecessor_list_type predecessor_list_type;
        typedef typename receiver<input_type>::built_predecessors_type built_predecessors_type;
#endif
    private:
        // ----------- Aggregator ------------
        enum op_type { reg_pred, rem_pred, res_item, rel_res, con_res
#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
            , add_blt_pred, del_blt_pred, blt_pred_cnt, blt_pred_cpy
#endif
        };
        typedef reserving_port<T> class_type;

        class reserving_port_operation : public aggregated_operation<reserving_port_operation> {
        public:
            char type;
            union {
                T *my_arg;
                predecessor_type *my_pred;
#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
                size_t cnt_val;
                predecessor_list_type *plist;
#endif
            };
            reserving_port_operation(const T& e, op_type t) :
                type(char(t)), my_arg(const_cast<T*>(&e)) {}
            reserving_port_operation(const predecessor_type &s, op_type t) : type(char(t)),
                my_pred(const_cast<predecessor_type *>(&s)) {}
            reserving_port_operation(op_type t) : type(char(t)) {}
        };

        typedef internal::aggregating_functor<class_type, reserving_port_operation> handler_type;
        friend class internal::aggregating_functor<class_type, reserving_port_operation>;
        aggregator<handler_type, reserving_port_operation> my_aggregator;

        void handle_operations(reserving_port_operation* op_list) {
            reserving_port_operation *current;
            bool no_predecessors;
            while(op_list) {
                current = op_list;
                op_list = op_list->next;
                switch(current->type) {
                case reg_pred:
                    no_predecessors = my_predecessors.empty();
                    my_predecessors.add(*(current->my_pred));
                    if ( no_predecessors ) {
                        (void) my_join->decrement_port_count(true); // may try to forward
                    }
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
                case rem_pred:
                    my_predecessors.remove(*(current->my_pred));
                    if(my_predecessors.empty()) my_join->increment_port_count();
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
                case res_item:
                    if ( reserved ) {
                        __TBB_store_with_release(current->status, FAILED);
                    }
                    else if ( my_predecessors.try_reserve( *(current->my_arg) ) ) {
                        reserved = true;
                        __TBB_store_with_release(current->status, SUCCEEDED);
                    } else {
                        if ( my_predecessors.empty() ) {
                            my_join->increment_port_count();
                        }
                        __TBB_store_with_release(current->status, FAILED);
                    }
                    break;
                case rel_res:
                    reserved = false;
                    my_predecessors.try_release( );
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
                case con_res:
                    reserved = false;
                    my_predecessors.try_consume( );
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
                case add_blt_pred:
                    my_predecessors.internal_add_built_predecessor(*(current->my_pred));
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
                case del_blt_pred:
                    my_predecessors.internal_delete_built_predecessor(*(current->my_pred));
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
                case blt_pred_cnt:
                    current->cnt_val = my_predecessors.predecessor_count();
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
                case blt_pred_cpy:
                    my_predecessors.copy_predecessors(*(current->plist));
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
#endif  /* TBB_DEPRECATED_FLOW_NODE_EXTRACTION */
                }
            }
        }

    protected:
        template< typename R, typename B > friend class run_and_put_task;
        template<typename X, typename Y> friend class internal::broadcast_cache;
        template<typename X, typename Y> friend class internal::round_robin_cache;
        task *try_put_task( const T & ) __TBB_override {
            return NULL;
        }

        graph& graph_reference() const __TBB_override {
            return my_join->graph_ref;
        }

    public:

        //! Constructor
        reserving_port() : reserved(false) {
            my_join = NULL;
            my_predecessors.set_owner( this );
            my_aggregator.initialize_handler(handler_type(this));
        }

        // copy constructor
        reserving_port(const reserving_port& /* other */) : receiver<T>() {
            reserved = false;
            my_join = NULL;
            my_predecessors.set_owner( this );
            my_aggregator.initialize_handler(handler_type(this));
        }

        void set_join_node_pointer(forwarding_base *join) {
            my_join = join;
        }

        //! Add a predecessor
        bool register_predecessor( predecessor_type &src ) __TBB_override {
            reserving_port_operation op_data(src, reg_pred);
            my_aggregator.execute(&op_data);
            return op_data.status == SUCCEEDED;
        }

        //! Remove a predecessor
        bool remove_predecessor( predecessor_type &src ) __TBB_override {
            reserving_port_operation op_data(src, rem_pred);
            my_aggregator.execute(&op_data);
            return op_data.status == SUCCEEDED;
        }

        //! Reserve an item from the port
        bool reserve( T &v ) {
            reserving_port_operation op_data(v, res_item);
            my_aggregator.execute(&op_data);
            return op_data.status == SUCCEEDED;
        }

        //! Release the port
        void release( ) {
            reserving_port_operation op_data(rel_res);
            my_aggregator.execute(&op_data);
        }

        //! Complete use of the port
        void consume( ) {
            reserving_port_operation op_data(con_res);
            my_aggregator.execute(&op_data);
        }

#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
        built_predecessors_type &built_predecessors() __TBB_override { return my_predecessors.built_predecessors(); }
        void internal_add_built_predecessor(predecessor_type &src) __TBB_override {
            reserving_port_operation op_data(src, add_blt_pred);
            my_aggregator.execute(&op_data);
        }

        void internal_delete_built_predecessor(predecessor_type &src) __TBB_override {
            reserving_port_operation op_data(src, del_blt_pred);
            my_aggregator.execute(&op_data);
        }

        size_t predecessor_count() __TBB_override {
            reserving_port_operation op_data(blt_pred_cnt);
            my_aggregator.execute(&op_data);
            return op_data.cnt_val;
        }

        void copy_predecessors(predecessor_list_type &l) __TBB_override {
            reserving_port_operation op_data(blt_pred_cpy);
            op_data.plist = &l;
            my_aggregator.execute(&op_data);
        }

        void extract_receiver() {
            my_predecessors.built_predecessors().receiver_extract(*this);
        }

#endif  /* TBB_DEPRECATED_FLOW_NODE_EXTRACTION */

        void reset_receiver( reset_flags f) __TBB_override {
            if(f & rf_clear_edges) my_predecessors.clear();
            else
            my_predecessors.reset();
            reserved = false;
            __TBB_ASSERT(!(f&rf_clear_edges) || my_predecessors.empty(), "port edges not removed");
        }

    private:
#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
        friend class get_graph_helper;
#endif

        forwarding_base *my_join;
        reservable_predecessor_cache< T, null_mutex > my_predecessors;
        bool reserved;
    };  // reserving_port

    //! queueing join_port
    template<typename T>
    class queueing_port : public receiver<T>, public item_buffer<T> {
    public:
        typedef T input_type;
        typedef typename receiver<input_type>::predecessor_type predecessor_type;
        typedef queueing_port<T> class_type;
#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
        typedef typename receiver<input_type>::built_predecessors_type built_predecessors_type;
        typedef typename receiver<input_type>::predecessor_list_type predecessor_list_type;
#endif

    // ----------- Aggregator ------------
    private:
        enum op_type { get__item, res_port, try__put_task
#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
            , add_blt_pred, del_blt_pred, blt_pred_cnt, blt_pred_cpy
#endif
        };

        class queueing_port_operation : public aggregated_operation<queueing_port_operation> {
        public:
            char type;
            T my_val;
            T *my_arg;
#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
            predecessor_type *pred;
            size_t cnt_val;
            predecessor_list_type *plist;
#endif
            task * bypass_t;
            // constructor for value parameter
            queueing_port_operation(const T& e, op_type t) :
                type(char(t)), my_val(e)
                , bypass_t(NULL)
            {}
            // constructor for pointer parameter
            queueing_port_operation(const T* p, op_type t) :
                type(char(t)), my_arg(const_cast<T*>(p))
                , bypass_t(NULL)
            {}
            // constructor with no parameter
            queueing_port_operation(op_type t) : type(char(t))
                , bypass_t(NULL)
            {}
        };

        typedef internal::aggregating_functor<class_type, queueing_port_operation> handler_type;
        friend class internal::aggregating_functor<class_type, queueing_port_operation>;
        aggregator<handler_type, queueing_port_operation> my_aggregator;

        void handle_operations(queueing_port_operation* op_list) {
            queueing_port_operation *current;
            bool was_empty;
            while(op_list) {
                current = op_list;
                op_list = op_list->next;
                switch(current->type) {
                case try__put_task: {
                        task *rtask = NULL;
                        was_empty = this->buffer_empty();
                        this->push_back(current->my_val);
                        if (was_empty) rtask = my_join->decrement_port_count(false);
                        else
                            rtask = SUCCESSFULLY_ENQUEUED;
                        current->bypass_t = rtask;
                        __TBB_store_with_release(current->status, SUCCEEDED);
                    }
                    break;
                case get__item:
                    if(!this->buffer_empty()) {
                        *(current->my_arg) = this->front();
                        __TBB_store_with_release(current->status, SUCCEEDED);
                    }
                    else {
                        __TBB_store_with_release(current->status, FAILED);
                    }
                    break;
                case res_port:
                    __TBB_ASSERT(this->my_item_valid(this->my_head), "No item to reset");
                    this->destroy_front();
                    if(this->my_item_valid(this->my_head)) {
                        (void)my_join->decrement_port_count(true);
                    }
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
                case add_blt_pred:
                    my_built_predecessors.add_edge(*(current->pred));
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
                case del_blt_pred:
                    my_built_predecessors.delete_edge(*(current->pred));
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
                case blt_pred_cnt:
                    current->cnt_val = my_built_predecessors.edge_count();
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
                case blt_pred_cpy:
                    my_built_predecessors.copy_edges(*(current->plist));
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
#endif  /* TBB_DEPRECATED_FLOW_NODE_EXTRACTION */
                }
            }
        }
    // ------------ End Aggregator ---------------

    protected:
        template< typename R, typename B > friend class run_and_put_task;
        template<typename X, typename Y> friend class internal::broadcast_cache;
        template<typename X, typename Y> friend class internal::round_robin_cache;
        task *try_put_task(const T &v) __TBB_override {
            queueing_port_operation op_data(v, try__put_task);
            my_aggregator.execute(&op_data);
            __TBB_ASSERT(op_data.status == SUCCEEDED || !op_data.bypass_t, "inconsistent return from aggregator");
            if(!op_data.bypass_t) return SUCCESSFULLY_ENQUEUED;
            return op_data.bypass_t;
        }

        graph& graph_reference() const __TBB_override {
            return my_join->graph_ref;
        }

    public:

        //! Constructor
        queueing_port() : item_buffer<T>() {
            my_join = NULL;
            my_aggregator.initialize_handler(handler_type(this));
        }

        //! copy constructor
        queueing_port(const queueing_port& /* other */) : receiver<T>(), item_buffer<T>() {
            my_join = NULL;
            my_aggregator.initialize_handler(handler_type(this));
        }

        //! record parent for tallying available items
        void set_join_node_pointer(forwarding_base *join) {
            my_join = join;
        }

        bool get_item( T &v ) {
            queueing_port_operation op_data(&v, get__item);
            my_aggregator.execute(&op_data);
            return op_data.status == SUCCEEDED;
        }

        // reset_port is called when item is accepted by successor, but
        // is initiated by join_node.
        void reset_port() {
            queueing_port_operation op_data(res_port);
            my_aggregator.execute(&op_data);
            return;
        }

#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
        built_predecessors_type &built_predecessors() __TBB_override { return my_built_predecessors; }

        void internal_add_built_predecessor(predecessor_type &p) __TBB_override {
            queueing_port_operation op_data(add_blt_pred);
            op_data.pred = &p;
            my_aggregator.execute(&op_data);
        }

        void internal_delete_built_predecessor(predecessor_type &p) __TBB_override {
            queueing_port_operation op_data(del_blt_pred);
            op_data.pred = &p;
            my_aggregator.execute(&op_data);
        }

        size_t predecessor_count() __TBB_override {
            queueing_port_operation op_data(blt_pred_cnt);
            my_aggregator.execute(&op_data);
            return op_data.cnt_val;
        }

        void copy_predecessors(predecessor_list_type &l) __TBB_override {
            queueing_port_operation op_data(blt_pred_cpy);
            op_data.plist = &l;
            my_aggregator.execute(&op_data);
        }

        void extract_receiver() {
            item_buffer<T>::reset();
            my_built_predecessors.receiver_extract(*this);
        }
#endif  /* TBB_DEPRECATED_FLOW_NODE_EXTRACTION */

        void reset_receiver(reset_flags f) __TBB_override {
            tbb::internal::suppress_unused_warning(f);
            item_buffer<T>::reset();
#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
            if (f & rf_clear_edges)
                my_built_predecessors.clear();
#endif
        }

    private:
#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
        friend class get_graph_helper;
#endif

        forwarding_base *my_join;
#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
        edge_container<predecessor_type> my_built_predecessors;
#endif
    };  // queueing_port

#include "_flow_graph_tagged_buffer_impl.h"

    template<typename K>
    struct count_element {
        K my_key;
        size_t my_value;
    };

    // method to access the key in the counting table
    // the ref has already been removed from K
    template< typename K >
    struct key_to_count_functor {
        typedef count_element<K> table_item_type;
        const K& operator()(const table_item_type& v) { return v.my_key; }
    };

    // the ports can have only one template parameter.  We wrap the types needed in
    // a traits type
    template< class TraitsType >
    class key_matching_port :
        public receiver<typename TraitsType::T>,
        public hash_buffer< typename TraitsType::K, typename TraitsType::T, typename TraitsType::TtoK,
                typename TraitsType::KHash > {
    public:
        typedef TraitsType traits;
        typedef key_matching_port<traits> class_type;
        typedef typename TraitsType::T input_type;
        typedef typename TraitsType::K key_type;
        typedef typename tbb::internal::strip<key_type>::type noref_key_type;
        typedef typename receiver<input_type>::predecessor_type predecessor_type;
        typedef typename TraitsType::TtoK type_to_key_func_type;
        typedef typename TraitsType::KHash hash_compare_type;
        typedef hash_buffer< key_type, input_type, type_to_key_func_type, hash_compare_type > buffer_type;
#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
        typedef typename receiver<input_type>::built_predecessors_type built_predecessors_type;
        typedef typename receiver<input_type>::predecessor_list_type predecessor_list_type;
#endif
    private:
// ----------- Aggregator ------------
    private:
        enum op_type { try__put, get__item, res_port
#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
           , add_blt_pred, del_blt_pred, blt_pred_cnt, blt_pred_cpy
#endif
        };

        class key_matching_port_operation : public aggregated_operation<key_matching_port_operation> {
        public:
            char type;
            input_type my_val;
            input_type *my_arg;
#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
            predecessor_type *pred;
            size_t cnt_val;
            predecessor_list_type *plist;
#endif
            // constructor for value parameter
            key_matching_port_operation(const input_type& e, op_type t) :
                type(char(t)), my_val(e) {}
            // constructor for pointer parameter
            key_matching_port_operation(const input_type* p, op_type t) :
                type(char(t)), my_arg(const_cast<input_type*>(p)) {}
            // constructor with no parameter
            key_matching_port_operation(op_type t) : type(char(t)) {}
        };

        typedef internal::aggregating_functor<class_type, key_matching_port_operation> handler_type;
        friend class internal::aggregating_functor<class_type, key_matching_port_operation>;
        aggregator<handler_type, key_matching_port_operation> my_aggregator;

        void handle_operations(key_matching_port_operation* op_list) {
            key_matching_port_operation *current;
            while(op_list) {
                current = op_list;
                op_list = op_list->next;
                switch(current->type) {
                case try__put: {
                        bool was_inserted = this->insert_with_key(current->my_val);
                        // return failure if a duplicate insertion occurs
                        __TBB_store_with_release(current->status, was_inserted ? SUCCEEDED : FAILED);
                    }
                    break;
                case get__item:
                    // use current_key from FE for item
                    if(!this->find_with_key(my_join->current_key, *(current->my_arg))) {
                        __TBB_ASSERT(false, "Failed to find item corresponding to current_key.");
                    }
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
                case res_port:
                    // use current_key from FE for item
                    this->delete_with_key(my_join->current_key);
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
                case add_blt_pred:
                    my_built_predecessors.add_edge(*(current->pred));
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
                case del_blt_pred:
                    my_built_predecessors.delete_edge(*(current->pred));
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
                case blt_pred_cnt:
                    current->cnt_val = my_built_predecessors.edge_count();
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
                case blt_pred_cpy:
                    my_built_predecessors.copy_edges(*(current->plist));
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
#endif
                }
            }
        }
// ------------ End Aggregator ---------------
    protected:
        template< typename R, typename B > friend class run_and_put_task;
        template<typename X, typename Y> friend class internal::broadcast_cache;
        template<typename X, typename Y> friend class internal::round_robin_cache;
        task *try_put_task(const input_type& v) __TBB_override {
            key_matching_port_operation op_data(v, try__put);
            task *rtask = NULL;
            my_aggregator.execute(&op_data);
            if(op_data.status == SUCCEEDED) {
                rtask = my_join->increment_key_count((*(this->get_key_func()))(v), false);  // may spawn
                // rtask has to reflect the return status of the try_put
                if(!rtask) rtask = SUCCESSFULLY_ENQUEUED;
            }
            return rtask;
        }

        graph& graph_reference() const __TBB_override {
            return my_join->graph_ref;
        }

    public:

        key_matching_port() : receiver<input_type>(), buffer_type() {
            my_join = NULL;
            my_aggregator.initialize_handler(handler_type(this));
        }

        // copy constructor
        key_matching_port(const key_matching_port& /*other*/) : receiver<input_type>(), buffer_type() {
            my_join = NULL;
            my_aggregator.initialize_handler(handler_type(this));
        }

        ~key_matching_port() { }

        void set_join_node_pointer(forwarding_base *join) {
            my_join = dynamic_cast<matching_forwarding_base<key_type>*>(join);
        }

        void set_my_key_func(type_to_key_func_type *f) { this->set_key_func(f); }

        type_to_key_func_type* get_my_key_func() { return this->get_key_func(); }

        bool get_item( input_type &v ) {
            // aggregator uses current_key from FE for Key
            key_matching_port_operation op_data(&v, get__item);
            my_aggregator.execute(&op_data);
            return op_data.status == SUCCEEDED;
        }

#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
        built_predecessors_type &built_predecessors() __TBB_override { return my_built_predecessors; }

        void internal_add_built_predecessor(predecessor_type &p) __TBB_override {
            key_matching_port_operation op_data(add_blt_pred);
            op_data.pred = &p;
            my_aggregator.execute(&op_data);
        }

        void internal_delete_built_predecessor(predecessor_type &p) __TBB_override {
            key_matching_port_operation op_data(del_blt_pred);
            op_data.pred = &p;
            my_aggregator.execute(&op_data);
        }

        size_t predecessor_count() __TBB_override {
            key_matching_port_operation op_data(blt_pred_cnt);
            my_aggregator.execute(&op_data);
            return op_data.cnt_val;
        }

        void copy_predecessors(predecessor_list_type &l) __TBB_override {
            key_matching_port_operation op_data(blt_pred_cpy);
            op_data.plist = &l;
            my_aggregator.execute(&op_data);
        }
#endif

        // reset_port is called when item is accepted by successor, but
        // is initiated by join_node.
        void reset_port() {
            key_matching_port_operation op_data(res_port);
            my_aggregator.execute(&op_data);
            return;
        }

#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
        void extract_receiver() {
            buffer_type::reset();
            my_built_predecessors.receiver_extract(*this);
        }
#endif
        void reset_receiver(reset_flags f ) __TBB_override {
            tbb::internal::suppress_unused_warning(f);
            buffer_type::reset();
#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
           if (f & rf_clear_edges)
              my_built_predecessors.clear();
#endif
        }

    private:
        // my_join forwarding base used to count number of inputs that
        // received key.
        matching_forwarding_base<key_type> *my_join;
#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
        edge_container<predecessor_type> my_built_predecessors;
#endif
    };  // key_matching_port

    using namespace graph_policy_namespace;

    template<typename JP, typename InputTuple, typename OutputTuple>
    class join_node_base;

    //! join_node_FE : implements input port policy
    template<typename JP, typename InputTuple, typename OutputTuple>
    class join_node_FE;

    template<typename InputTuple, typename OutputTuple>
    class join_node_FE<reserving, InputTuple, OutputTuple> : public forwarding_base {
    public:
        static const int N = tbb::flow::tuple_size<OutputTuple>::value;
        typedef OutputTuple output_type;
        typedef InputTuple input_type;
        typedef join_node_base<reserving, InputTuple, OutputTuple> base_node_type; // for forwarding

        join_node_FE(graph &g) : forwarding_base(g), my_node(NULL) {
            ports_with_no_inputs = N;
            join_helper<N>::set_join_node_pointer(my_inputs, this);
        }

        join_node_FE(const join_node_FE& other) : forwarding_base((other.forwarding_base::graph_ref)), my_node(NULL) {
            ports_with_no_inputs = N;
            join_helper<N>::set_join_node_pointer(my_inputs, this);
        }

        void set_my_node(base_node_type *new_my_node) { my_node = new_my_node; }

       void increment_port_count() __TBB_override {
            ++ports_with_no_inputs;
        }

        // if all input_ports have predecessors, spawn forward to try and consume tuples
        task * decrement_port_count(bool handle_task) __TBB_override {
            if(ports_with_no_inputs.fetch_and_decrement() == 1) {
                if(internal::is_graph_active(this->graph_ref)) {
                    task *rtask = new ( task::allocate_additional_child_of( *(this->graph_ref.root_task()) ) )
                        forward_task_bypass<base_node_type>(*my_node);
                    if(!handle_task) return rtask;
                    internal::spawn_in_graph_arena(this->graph_ref, *rtask);
                }
            }
            return NULL;
        }

        input_type &input_ports() { return my_inputs; }

    protected:

        void reset(  reset_flags f) {
            // called outside of parallel contexts
            ports_with_no_inputs = N;
            join_helper<N>::reset_inputs(my_inputs, f);
        }

#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
        void extract( ) {
            // called outside of parallel contexts
            ports_with_no_inputs = N;
            join_helper<N>::extract_inputs(my_inputs);
        }
#endif

        // all methods on input ports should be called under mutual exclusion from join_node_base.

        bool tuple_build_may_succeed() {
            return !ports_with_no_inputs;
        }

        bool try_to_make_tuple(output_type &out) {
            if(ports_with_no_inputs) return false;
            return join_helper<N>::reserve(my_inputs, out);
        }

        void tuple_accepted() {
            join_helper<N>::consume_reservations(my_inputs);
        }
        void tuple_rejected() {
            join_helper<N>::release_reservations(my_inputs);
        }

        input_type my_inputs;
        base_node_type *my_node;
        atomic<size_t> ports_with_no_inputs;
    };  // join_node_FE<reserving, ... >

    template<typename InputTuple, typename OutputTuple>
    class join_node_FE<queueing, InputTuple, OutputTuple> : public forwarding_base {
    public:
        static const int N = tbb::flow::tuple_size<OutputTuple>::value;
        typedef OutputTuple output_type;
        typedef InputTuple input_type;
        typedef join_node_base<queueing, InputTuple, OutputTuple> base_node_type; // for forwarding

        join_node_FE(graph &g) : forwarding_base(g), my_node(NULL) {
            ports_with_no_items = N;
            join_helper<N>::set_join_node_pointer(my_inputs, this);
        }

        join_node_FE(const join_node_FE& other) : forwarding_base((other.forwarding_base::graph_ref)), my_node(NULL) {
            ports_with_no_items = N;
            join_helper<N>::set_join_node_pointer(my_inputs, this);
        }

        // needed for forwarding
        void set_my_node(base_node_type *new_my_node) { my_node = new_my_node; }

        void reset_port_count() {
            ports_with_no_items = N;
        }

        // if all input_ports have items, spawn forward to try and consume tuples
        task * decrement_port_count(bool handle_task) __TBB_override
        {
            if(ports_with_no_items.fetch_and_decrement() == 1) {
                if(internal::is_graph_active(this->graph_ref)) {
                    task *rtask = new ( task::allocate_additional_child_of( *(this->graph_ref.root_task()) ) )
                        forward_task_bypass <base_node_type>(*my_node);
                    if(!handle_task) return rtask;
                    internal::spawn_in_graph_arena(this->graph_ref, *rtask);
                }
            }
            return NULL;
        }

        void increment_port_count() __TBB_override { __TBB_ASSERT(false, NULL); }  // should never be called

        input_type &input_ports() { return my_inputs; }

    protected:

        void reset(  reset_flags f) {
            reset_port_count();
            join_helper<N>::reset_inputs(my_inputs, f );
        }

#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
        void extract() {
            reset_port_count();
            join_helper<N>::extract_inputs(my_inputs);
        }
#endif
        // all methods on input ports should be called under mutual exclusion from join_node_base.

        bool tuple_build_may_succeed() {
            return !ports_with_no_items;
        }

        bool try_to_make_tuple(output_type &out) {
            if(ports_with_no_items) return false;
            return join_helper<N>::get_items(my_inputs, out);
        }

        void tuple_accepted() {
            reset_port_count();
            join_helper<N>::reset_ports(my_inputs);
        }
        void tuple_rejected() {
            // nothing to do.
        }

        input_type my_inputs;
        base_node_type *my_node;
        atomic<size_t> ports_with_no_items;
    };  // join_node_FE<queueing, ...>

    // key_matching join front-end.
    template<typename InputTuple, typename OutputTuple, typename K, typename KHash>
    class join_node_FE<key_matching<K,KHash>, InputTuple, OutputTuple> : public matching_forwarding_base<K>,
             // buffer of key value counts
              public hash_buffer<   // typedefed below to key_to_count_buffer_type
                  typename tbb::internal::strip<K>::type&,        // force ref type on K
                  count_element<typename tbb::internal::strip<K>::type>,
                  internal::type_to_key_function_body<
                      count_element<typename tbb::internal::strip<K>::type>,
                      typename tbb::internal::strip<K>::type& >,
                  KHash >,
             // buffer of output items
             public item_buffer<OutputTuple> {
    public:
        static const int N = tbb::flow::tuple_size<OutputTuple>::value;
        typedef OutputTuple output_type;
        typedef InputTuple input_type;
        typedef K key_type;
        typedef typename tbb::internal::strip<key_type>::type unref_key_type;
        typedef KHash key_hash_compare;
        // must use K without ref.
        typedef count_element<unref_key_type> count_element_type;
        // method that lets us refer to the key of this type.
        typedef key_to_count_functor<unref_key_type> key_to_count_func;
        typedef internal::type_to_key_function_body< count_element_type, unref_key_type&> TtoK_function_body_type;
        typedef internal::type_to_key_function_body_leaf<count_element_type, unref_key_type&, key_to_count_func> TtoK_function_body_leaf_type;
        // this is the type of the special table that keeps track of the number of discrete
        // elements corresponding to each key that we've seen.
        typedef hash_buffer< unref_key_type&, count_element_type, TtoK_function_body_type, key_hash_compare >
                 key_to_count_buffer_type;
        typedef item_buffer<output_type> output_buffer_type;
        typedef join_node_base<key_matching<key_type,key_hash_compare>, InputTuple, OutputTuple> base_node_type; // for forwarding
        typedef matching_forwarding_base<key_type> forwarding_base_type;

// ----------- Aggregator ------------
        // the aggregator is only needed to serialize the access to the hash table.
        // and the output_buffer_type base class
    private:
        enum op_type { res_count, inc_count, may_succeed, try_make };
        typedef join_node_FE<key_matching<key_type,key_hash_compare>, InputTuple, OutputTuple> class_type;

        class key_matching_FE_operation : public aggregated_operation<key_matching_FE_operation> {
        public:
            char type;
            unref_key_type my_val;
            output_type* my_output;
            task *bypass_t;
            bool enqueue_task;
            // constructor for value parameter
            key_matching_FE_operation(const unref_key_type& e , bool q_task , op_type t) : type(char(t)), my_val(e),
                 my_output(NULL), bypass_t(NULL), enqueue_task(q_task) {}
            key_matching_FE_operation(output_type *p, op_type t) : type(char(t)), my_output(p), bypass_t(NULL),
                 enqueue_task(true) {}
            // constructor with no parameter
            key_matching_FE_operation(op_type t) : type(char(t)), my_output(NULL), bypass_t(NULL), enqueue_task(true) {}
        };

        typedef internal::aggregating_functor<class_type, key_matching_FE_operation> handler_type;
        friend class internal::aggregating_functor<class_type, key_matching_FE_operation>;
        aggregator<handler_type, key_matching_FE_operation> my_aggregator;

        // called from aggregator, so serialized
        // returns a task pointer if the a task would have been enqueued but we asked that
        // it be returned.  Otherwise returns NULL.
        task * fill_output_buffer(unref_key_type &t, bool should_enqueue, bool handle_task) {
            output_type l_out;
            task *rtask = NULL;
            bool do_fwd = should_enqueue && this->buffer_empty() && internal::is_graph_active(this->graph_ref);
            this->current_key = t;
            this->delete_with_key(this->current_key);   // remove the key
            if(join_helper<N>::get_items(my_inputs, l_out)) {  //  <== call back
                this->push_back(l_out);
                if(do_fwd) {  // we enqueue if receiving an item from predecessor, not if successor asks for item
                    rtask = new ( task::allocate_additional_child_of( *(this->graph_ref.root_task()) ) )
                        forward_task_bypass<base_node_type>(*my_node);
                    if(handle_task) {
                        internal::spawn_in_graph_arena(this->graph_ref, *rtask);
                        rtask = NULL;
                    }
                    do_fwd = false;
                }
                // retire the input values
                join_helper<N>::reset_ports(my_inputs);  //  <== call back
            }
            else {
                __TBB_ASSERT(false, "should have had something to push");
            }
            return rtask;
        }

        void handle_operations(key_matching_FE_operation* op_list) {
            key_matching_FE_operation *current;
            while(op_list) {
                current = op_list;
                op_list = op_list->next;
                switch(current->type) {
                case res_count:  // called from BE
                    {
                        this->destroy_front();
                        __TBB_store_with_release(current->status, SUCCEEDED);
                    }
                    break;
                case inc_count: {  // called from input ports
                        count_element_type *p = 0;
                        unref_key_type &t = current->my_val;
                        bool do_enqueue = current->enqueue_task;
                        if(!(this->find_ref_with_key(t,p))) {
                            count_element_type ev;
                            ev.my_key = t;
                            ev.my_value = 0;
                            this->insert_with_key(ev);
                            if(!(this->find_ref_with_key(t,p))) {
                                __TBB_ASSERT(false, "should find key after inserting it");
                            }
                        }
                        if(++(p->my_value) == size_t(N)) {
                            task *rtask = fill_output_buffer(t, true, do_enqueue);
                            __TBB_ASSERT(!rtask || !do_enqueue, "task should not be returned");
                            current->bypass_t = rtask;
                        }
                    }
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
                case may_succeed:  // called from BE
                    __TBB_store_with_release(current->status, this->buffer_empty() ? FAILED : SUCCEEDED);
                    break;
                case try_make:  // called from BE
                    if(this->buffer_empty()) {
                        __TBB_store_with_release(current->status, FAILED);
                    }
                    else {
                        *(current->my_output) = this->front();
                        __TBB_store_with_release(current->status, SUCCEEDED);
                    }
                    break;
                }
            }
        }
// ------------ End Aggregator ---------------

    public:
        template<typename FunctionTuple>
        join_node_FE(graph &g, FunctionTuple &TtoK_funcs) : forwarding_base_type(g), my_node(NULL) {
            join_helper<N>::set_join_node_pointer(my_inputs, this);
            join_helper<N>::set_key_functors(my_inputs, TtoK_funcs);
            my_aggregator.initialize_handler(handler_type(this));
                    TtoK_function_body_type *cfb = new TtoK_function_body_leaf_type(key_to_count_func());
            this->set_key_func(cfb);
        }

        join_node_FE(const join_node_FE& other) : forwarding_base_type((other.forwarding_base_type::graph_ref)), key_to_count_buffer_type(),
        output_buffer_type() {
            my_node = NULL;
            join_helper<N>::set_join_node_pointer(my_inputs, this);
            join_helper<N>::copy_key_functors(my_inputs, const_cast<input_type &>(other.my_inputs));
            my_aggregator.initialize_handler(handler_type(this));
            TtoK_function_body_type *cfb = new TtoK_function_body_leaf_type(key_to_count_func());
            this->set_key_func(cfb);
        }

        // needed for forwarding
        void set_my_node(base_node_type *new_my_node) { my_node = new_my_node; }

        void reset_port_count() {  // called from BE
            key_matching_FE_operation op_data(res_count);
            my_aggregator.execute(&op_data);
            return;
        }

        // if all input_ports have items, spawn forward to try and consume tuples
        // return a task if we are asked and did create one.
        task *increment_key_count(unref_key_type const & t, bool handle_task) __TBB_override {  // called from input_ports
            key_matching_FE_operation op_data(t, handle_task, inc_count);
            my_aggregator.execute(&op_data);
            return op_data.bypass_t;
        }

        task *decrement_port_count(bool /*handle_task*/) __TBB_override { __TBB_ASSERT(false, NULL); return NULL; }

        void increment_port_count() __TBB_override { __TBB_ASSERT(false, NULL); }  // should never be called

        input_type &input_ports() { return my_inputs; }

    protected:

        void reset(  reset_flags f ) {
            // called outside of parallel contexts
            join_helper<N>::reset_inputs(my_inputs, f);

            key_to_count_buffer_type::reset();
            output_buffer_type::reset();
        }

#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
        void extract() {
            // called outside of parallel contexts
            join_helper<N>::extract_inputs(my_inputs);
            key_to_count_buffer_type::reset();  // have to reset the tag counts
            output_buffer_type::reset();  // also the queue of outputs
            // my_node->current_tag = NO_TAG;
        }
#endif
        // all methods on input ports should be called under mutual exclusion from join_node_base.

        bool tuple_build_may_succeed() {  // called from back-end
            key_matching_FE_operation op_data(may_succeed);
            my_aggregator.execute(&op_data);
            return op_data.status == SUCCEEDED;
        }

        // cannot lock while calling back to input_ports.  current_key will only be set
        // and reset under the aggregator, so it will remain consistent.
        bool try_to_make_tuple(output_type &out) {
            key_matching_FE_operation op_data(&out,try_make);
            my_aggregator.execute(&op_data);
            return op_data.status == SUCCEEDED;
        }

        void tuple_accepted() {
            reset_port_count();  // reset current_key after ports reset.
        }

        void tuple_rejected() {
            // nothing to do.
        }

        input_type my_inputs;  // input ports
        base_node_type *my_node;
    }; // join_node_FE<key_matching<K,KHash>, InputTuple, OutputTuple>

    //! join_node_base
    template<typename JP, typename InputTuple, typename OutputTuple>
    class join_node_base : public graph_node, public join_node_FE<JP, InputTuple, OutputTuple>,
                           public sender<OutputTuple> {
    protected:
        using graph_node::my_graph;
    public:
        typedef OutputTuple output_type;

        typedef typename sender<output_type>::successor_type successor_type;
        typedef join_node_FE<JP, InputTuple, OutputTuple> input_ports_type;
        using input_ports_type::tuple_build_may_succeed;
        using input_ports_type::try_to_make_tuple;
        using input_ports_type::tuple_accepted;
        using input_ports_type::tuple_rejected;
#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
        typedef typename sender<output_type>::built_successors_type built_successors_type;
        typedef typename sender<output_type>::successor_list_type successor_list_type;
#endif

    private:
        // ----------- Aggregator ------------
        enum op_type { reg_succ, rem_succ, try__get, do_fwrd, do_fwrd_bypass
#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
            , add_blt_succ, del_blt_succ, blt_succ_cnt, blt_succ_cpy
#endif
        };
        typedef join_node_base<JP,InputTuple,OutputTuple> class_type;

        class join_node_base_operation : public aggregated_operation<join_node_base_operation> {
        public:
            char type;
            union {
                output_type *my_arg;
                successor_type *my_succ;
#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
                size_t cnt_val;
                successor_list_type *slist;
#endif
            };
            task *bypass_t;
            join_node_base_operation(const output_type& e, op_type t) : type(char(t)),
                my_arg(const_cast<output_type*>(&e)), bypass_t(NULL) {}
            join_node_base_operation(const successor_type &s, op_type t) : type(char(t)),
                my_succ(const_cast<successor_type *>(&s)), bypass_t(NULL) {}
            join_node_base_operation(op_type t) : type(char(t)), bypass_t(NULL) {}
        };

        typedef internal::aggregating_functor<class_type, join_node_base_operation> handler_type;
        friend class internal::aggregating_functor<class_type, join_node_base_operation>;
        bool forwarder_busy;
        aggregator<handler_type, join_node_base_operation> my_aggregator;

        void handle_operations(join_node_base_operation* op_list) {
            join_node_base_operation *current;
            while(op_list) {
                current = op_list;
                op_list = op_list->next;
                switch(current->type) {
                case reg_succ: {
                        my_successors.register_successor(*(current->my_succ));
                        if(tuple_build_may_succeed() && !forwarder_busy && internal::is_graph_active(my_graph)) {
                            task *rtask = new ( task::allocate_additional_child_of(*(my_graph.root_task())) )
                                    forward_task_bypass
                                    <join_node_base<JP,InputTuple,OutputTuple> >(*this);
                            internal::spawn_in_graph_arena(my_graph, *rtask);
                            forwarder_busy = true;
                        }
                        __TBB_store_with_release(current->status, SUCCEEDED);
                    }
                    break;
                case rem_succ:
                    my_successors.remove_successor(*(current->my_succ));
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
                case try__get:
                    if(tuple_build_may_succeed()) {
                        if(try_to_make_tuple(*(current->my_arg))) {
                            tuple_accepted();
                            __TBB_store_with_release(current->status, SUCCEEDED);
                        }
                        else __TBB_store_with_release(current->status, FAILED);
                    }
                    else __TBB_store_with_release(current->status, FAILED);
                    break;
                case do_fwrd_bypass: {
                        bool build_succeeded;
                        task *last_task = NULL;
                        output_type out;
                        if(tuple_build_may_succeed()) {  // checks output queue of FE
                            do {
                                build_succeeded = try_to_make_tuple(out);  // fetch front_end of queue
                                if(build_succeeded) {
                                    task *new_task = my_successors.try_put_task(out);
                                    last_task = combine_tasks(my_graph, last_task, new_task);
                                    if(new_task) {
                                        tuple_accepted();
                                    }
                                    else {
                                        tuple_rejected();
                                        build_succeeded = false;
                                    }
                                }
                            } while(build_succeeded);
                        }
                        current->bypass_t = last_task;
                        __TBB_store_with_release(current->status, SUCCEEDED);
                        forwarder_busy = false;
                    }
                    break;
#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
                case add_blt_succ:
                    my_successors.internal_add_built_successor(*(current->my_succ));
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
                case del_blt_succ:
                    my_successors.internal_delete_built_successor(*(current->my_succ));
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
                case blt_succ_cnt:
                    current->cnt_val = my_successors.successor_count();
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
                case blt_succ_cpy:
                    my_successors.copy_successors(*(current->slist));
                    __TBB_store_with_release(current->status, SUCCEEDED);
                    break;
#endif  /* TBB_DEPRECATED_FLOW_NODE_EXTRACTION */
                }
            }
        }
        // ---------- end aggregator -----------
    public:
        join_node_base(graph &g) : graph_node(g), input_ports_type(g), forwarder_busy(false) {
            my_successors.set_owner(this);
            input_ports_type::set_my_node(this);
            my_aggregator.initialize_handler(handler_type(this));
        }

        join_node_base(const join_node_base& other) :
            graph_node(other.graph_node::my_graph), input_ports_type(other),
            sender<OutputTuple>(), forwarder_busy(false), my_successors() {
            my_successors.set_owner(this);
            input_ports_type::set_my_node(this);
            my_aggregator.initialize_handler(handler_type(this));
        }

        template<typename FunctionTuple>
        join_node_base(graph &g, FunctionTuple f) : graph_node(g), input_ports_type(g, f), forwarder_busy(false) {
            my_successors.set_owner(this);
            input_ports_type::set_my_node(this);
            my_aggregator.initialize_handler(handler_type(this));
        }

        bool register_successor(successor_type &r) __TBB_override {
            join_node_base_operation op_data(r, reg_succ);
            my_aggregator.execute(&op_data);
            return op_data.status == SUCCEEDED;
        }

        bool remove_successor( successor_type &r) __TBB_override {
            join_node_base_operation op_data(r, rem_succ);
            my_aggregator.execute(&op_data);
            return op_data.status == SUCCEEDED;
        }

        bool try_get( output_type &v) __TBB_override {
            join_node_base_operation op_data(v, try__get);
            my_aggregator.execute(&op_data);
            return op_data.status == SUCCEEDED;
        }

#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
        built_successors_type &built_successors() __TBB_override { return my_successors.built_successors(); }

        void internal_add_built_successor( successor_type &r) __TBB_override {
            join_node_base_operation op_data(r, add_blt_succ);
            my_aggregator.execute(&op_data);
        }

        void internal_delete_built_successor( successor_type &r) __TBB_override {
            join_node_base_operation op_data(r, del_blt_succ);
            my_aggregator.execute(&op_data);
        }

        size_t successor_count() __TBB_override {
            join_node_base_operation op_data(blt_succ_cnt);
            my_aggregator.execute(&op_data);
            return op_data.cnt_val;
        }

        void copy_successors(successor_list_type &l) __TBB_override {
            join_node_base_operation op_data(blt_succ_cpy);
            op_data.slist = &l;
            my_aggregator.execute(&op_data);
        }
#endif  /* TBB_DEPRECATED_FLOW_NODE_EXTRACTION */

#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
        void extract() __TBB_override {
            input_ports_type::extract();
            my_successors.built_successors().sender_extract(*this);
        }
#endif

    protected:

        void reset_node(reset_flags f) __TBB_override {
            input_ports_type::reset(f);
            if(f & rf_clear_edges) my_successors.clear();
        }

    private:
        broadcast_cache<output_type, null_rw_mutex> my_successors;

        friend class forward_task_bypass< join_node_base<JP, InputTuple, OutputTuple> >;
        task *forward_task() {
            join_node_base_operation op_data(do_fwrd_bypass);
            my_aggregator.execute(&op_data);
            return op_data.bypass_t;
        }

    };  // join_node_base

    // join base class type generator
    template<int N, template<class> class PT, typename OutputTuple, typename JP>
    struct join_base {
        typedef typename internal::join_node_base<JP, typename wrap_tuple_elements<N,PT,OutputTuple>::type, OutputTuple> type;
    };

    template<int N, typename OutputTuple, typename K, typename KHash>
    struct join_base<N, key_matching_port, OutputTuple, key_matching<K,KHash> > {
        typedef key_matching<K, KHash> key_traits_type;
        typedef K key_type;
        typedef KHash key_hash_compare;
        typedef typename internal::join_node_base< key_traits_type,
                // ports type
                typename wrap_key_tuple_elements<N,key_matching_port,key_traits_type,OutputTuple>::type,
                OutputTuple > type;
    };

    //! unfolded_join_node : passes input_ports_type to join_node_base.  We build the input port type
    //  using tuple_element.  The class PT is the port type (reserving_port, queueing_port, key_matching_port)
    //  and should match the typename.

    template<int N, template<class> class PT, typename OutputTuple, typename JP>
    class unfolded_join_node : public join_base<N,PT,OutputTuple,JP>::type {
    public:
        typedef typename wrap_tuple_elements<N, PT, OutputTuple>::type input_ports_type;
        typedef OutputTuple output_type;
    private:
        typedef join_node_base<JP, input_ports_type, output_type > base_type;
    public:
        unfolded_join_node(graph &g) : base_type(g) {}
        unfolded_join_node(const unfolded_join_node &other) : base_type(other) {}
    };

#if __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING
    template <typename K, typename T>
    struct key_from_message_body {
        K operator()(const T& t) const {
            using tbb::flow::key_from_message;
            return key_from_message<K>(t);
        }
    };
    // Adds const to reference type
    template <typename K, typename T>
    struct key_from_message_body<K&,T> {
        const K& operator()(const T& t) const {
            using tbb::flow::key_from_message;
            return key_from_message<const K&>(t);
        }
    };
#endif /* __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING */
    // key_matching unfolded_join_node.  This must be a separate specialization because the constructors
    // differ.

    template<typename OutputTuple, typename K, typename KHash>
    class unfolded_join_node<2,key_matching_port,OutputTuple,key_matching<K,KHash> > : public
            join_base<2,key_matching_port,OutputTuple,key_matching<K,KHash> >::type {
        typedef typename tbb::flow::tuple_element<0, OutputTuple>::type T0;
        typedef typename tbb::flow::tuple_element<1, OutputTuple>::type T1;
    public:
        typedef typename wrap_key_tuple_elements<2,key_matching_port,key_matching<K,KHash>,OutputTuple>::type input_ports_type;
        typedef OutputTuple output_type;
    private:
        typedef join_node_base<key_matching<K,KHash>, input_ports_type, output_type > base_type;
        typedef typename internal::type_to_key_function_body<T0, K> *f0_p;
        typedef typename internal::type_to_key_function_body<T1, K> *f1_p;
        typedef typename tbb::flow::tuple< f0_p, f1_p > func_initializer_type;
    public:
#if __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING
        unfolded_join_node(graph &g) : base_type(g,
                func_initializer_type(
                    new internal::type_to_key_function_body_leaf<T0, K, key_from_message_body<K,T0> >(key_from_message_body<K,T0>()),
                    new internal::type_to_key_function_body_leaf<T1, K, key_from_message_body<K,T1> >(key_from_message_body<K,T1>())
                    ) ) {
        }
#endif /* __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING */
        template<typename Body0, typename Body1>
        unfolded_join_node(graph &g, Body0 body0, Body1 body1) : base_type(g,
                func_initializer_type(
                    new internal::type_to_key_function_body_leaf<T0, K, Body0>(body0),
                    new internal::type_to_key_function_body_leaf<T1, K, Body1>(body1)
                    ) ) {
            __TBB_STATIC_ASSERT(tbb::flow::tuple_size<OutputTuple>::value == 2, "wrong number of body initializers");
        }
        unfolded_join_node(const unfolded_join_node &other) : base_type(other) {}
    };

    template<typename OutputTuple, typename K, typename KHash>
    class unfolded_join_node<3,key_matching_port,OutputTuple,key_matching<K,KHash> > : public
            join_base<3,key_matching_port,OutputTuple,key_matching<K,KHash> >::type {
        typedef typename tbb::flow::tuple_element<0, OutputTuple>::type T0;
        typedef typename tbb::flow::tuple_element<1, OutputTuple>::type T1;
        typedef typename tbb::flow::tuple_element<2, OutputTuple>::type T2;
    public:
        typedef typename wrap_key_tuple_elements<3,key_matching_port,key_matching<K,KHash>,OutputTuple>::type input_ports_type;
        typedef OutputTuple output_type;
    private:
        typedef join_node_base<key_matching<K,KHash>, input_ports_type, output_type > base_type;
        typedef typename internal::type_to_key_function_body<T0, K> *f0_p;
        typedef typename internal::type_to_key_function_body<T1, K> *f1_p;
        typedef typename internal::type_to_key_function_body<T2, K> *f2_p;
        typedef typename tbb::flow::tuple< f0_p, f1_p, f2_p > func_initializer_type;
    public:
#if __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING
        unfolded_join_node(graph &g) : base_type(g,
                func_initializer_type(
                    new internal::type_to_key_function_body_leaf<T0, K, key_from_message_body<K,T0> >(key_from_message_body<K,T0>()),
                    new internal::type_to_key_function_body_leaf<T1, K, key_from_message_body<K,T1> >(key_from_message_body<K,T1>()),
                    new internal::type_to_key_function_body_leaf<T2, K, key_from_message_body<K,T2> >(key_from_message_body<K,T2>())
                    ) ) {
        }
#endif /* __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING */
        template<typename Body0, typename Body1, typename Body2>
        unfolded_join_node(graph &g, Body0 body0, Body1 body1, Body2 body2) : base_type(g,
                func_initializer_type(
                    new internal::type_to_key_function_body_leaf<T0, K, Body0>(body0),
                    new internal::type_to_key_function_body_leaf<T1, K, Body1>(body1),
                    new internal::type_to_key_function_body_leaf<T2, K, Body2>(body2)
                    ) ) {
            __TBB_STATIC_ASSERT(tbb::flow::tuple_size<OutputTuple>::value == 3, "wrong number of body initializers");
        }
        unfolded_join_node(const unfolded_join_node &other) : base_type(other) {}
    };

    template<typename OutputTuple, typename K, typename KHash>
    class unfolded_join_node<4,key_matching_port,OutputTuple,key_matching<K,KHash> > : public
            join_base<4,key_matching_port,OutputTuple,key_matching<K,KHash> >::type {
        typedef typename tbb::flow::tuple_element<0, OutputTuple>::type T0;
        typedef typename tbb::flow::tuple_element<1, OutputTuple>::type T1;
        typedef typename tbb::flow::tuple_element<2, OutputTuple>::type T2;
        typedef typename tbb::flow::tuple_element<3, OutputTuple>::type T3;
    public:
        typedef typename wrap_key_tuple_elements<4,key_matching_port,key_matching<K,KHash>,OutputTuple>::type input_ports_type;
        typedef OutputTuple output_type;
    private:
        typedef join_node_base<key_matching<K,KHash>, input_ports_type, output_type > base_type;
        typedef typename internal::type_to_key_function_body<T0, K> *f0_p;
        typedef typename internal::type_to_key_function_body<T1, K> *f1_p;
        typedef typename internal::type_to_key_function_body<T2, K> *f2_p;
        typedef typename internal::type_to_key_function_body<T3, K> *f3_p;
        typedef typename tbb::flow::tuple< f0_p, f1_p, f2_p, f3_p > func_initializer_type;
    public:
#if __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING
        unfolded_join_node(graph &g) : base_type(g,
                func_initializer_type(
                    new internal::type_to_key_function_body_leaf<T0, K, key_from_message_body<K,T0> >(key_from_message_body<K,T0>()),
                    new internal::type_to_key_function_body_leaf<T1, K, key_from_message_body<K,T1> >(key_from_message_body<K,T1>()),
                    new internal::type_to_key_function_body_leaf<T2, K, key_from_message_body<K,T2> >(key_from_message_body<K,T2>()),
                    new internal::type_to_key_function_body_leaf<T3, K, key_from_message_body<K,T3> >(key_from_message_body<K,T3>())
                    ) ) {
        }
#endif /* __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING */
        template<typename Body0, typename Body1, typename Body2, typename Body3>
        unfolded_join_node(graph &g, Body0 body0, Body1 body1, Body2 body2, Body3 body3) : base_type(g,
                func_initializer_type(
                    new internal::type_to_key_function_body_leaf<T0, K, Body0>(body0),
                    new internal::type_to_key_function_body_leaf<T1, K, Body1>(body1),
                    new internal::type_to_key_function_body_leaf<T2, K, Body2>(body2),
                    new internal::type_to_key_function_body_leaf<T3, K, Body3>(body3)
                    ) ) {
            __TBB_STATIC_ASSERT(tbb::flow::tuple_size<OutputTuple>::value == 4, "wrong number of body initializers");
        }
        unfolded_join_node(const unfolded_join_node &other) : base_type(other) {}
    };

    template<typename OutputTuple, typename K, typename KHash>
    class unfolded_join_node<5,key_matching_port,OutputTuple,key_matching<K,KHash> > : public
            join_base<5,key_matching_port,OutputTuple,key_matching<K,KHash> >::type {
        typedef typename tbb::flow::tuple_element<0, OutputTuple>::type T0;
        typedef typename tbb::flow::tuple_element<1, OutputTuple>::type T1;
        typedef typename tbb::flow::tuple_element<2, OutputTuple>::type T2;
        typedef typename tbb::flow::tuple_element<3, OutputTuple>::type T3;
        typedef typename tbb::flow::tuple_element<4, OutputTuple>::type T4;
    public:
        typedef typename wrap_key_tuple_elements<5,key_matching_port,key_matching<K,KHash>,OutputTuple>::type input_ports_type;
        typedef OutputTuple output_type;
    private:
        typedef join_node_base<key_matching<K,KHash> , input_ports_type, output_type > base_type;
        typedef typename internal::type_to_key_function_body<T0, K> *f0_p;
        typedef typename internal::type_to_key_function_body<T1, K> *f1_p;
        typedef typename internal::type_to_key_function_body<T2, K> *f2_p;
        typedef typename internal::type_to_key_function_body<T3, K> *f3_p;
        typedef typename internal::type_to_key_function_body<T4, K> *f4_p;
        typedef typename tbb::flow::tuple< f0_p, f1_p, f2_p, f3_p, f4_p > func_initializer_type;
    public:
#if __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING
        unfolded_join_node(graph &g) : base_type(g,
                func_initializer_type(
                    new internal::type_to_key_function_body_leaf<T0, K, key_from_message_body<K,T0> >(key_from_message_body<K,T0>()),
                    new internal::type_to_key_function_body_leaf<T1, K, key_from_message_body<K,T1> >(key_from_message_body<K,T1>()),
                    new internal::type_to_key_function_body_leaf<T2, K, key_from_message_body<K,T2> >(key_from_message_body<K,T2>()),
                    new internal::type_to_key_function_body_leaf<T3, K, key_from_message_body<K,T3> >(key_from_message_body<K,T3>()),
                    new internal::type_to_key_function_body_leaf<T4, K, key_from_message_body<K,T4> >(key_from_message_body<K,T4>())
                    ) ) {
        }
#endif /* __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING */
        template<typename Body0, typename Body1, typename Body2, typename Body3, typename Body4>
        unfolded_join_node(graph &g, Body0 body0, Body1 body1, Body2 body2, Body3 body3, Body4 body4) : base_type(g,
                func_initializer_type(
                    new internal::type_to_key_function_body_leaf<T0, K, Body0>(body0),
                    new internal::type_to_key_function_body_leaf<T1, K, Body1>(body1),
                    new internal::type_to_key_function_body_leaf<T2, K, Body2>(body2),
                    new internal::type_to_key_function_body_leaf<T3, K, Body3>(body3),
                    new internal::type_to_key_function_body_leaf<T4, K, Body4>(body4)
                    ) ) {
            __TBB_STATIC_ASSERT(tbb::flow::tuple_size<OutputTuple>::value == 5, "wrong number of body initializers");
        }
        unfolded_join_node(const unfolded_join_node &other) : base_type(other) {}
    };

#if __TBB_VARIADIC_MAX >= 6
    template<typename OutputTuple, typename K, typename KHash>
    class unfolded_join_node<6,key_matching_port,OutputTuple,key_matching<K,KHash> > : public
            join_base<6,key_matching_port,OutputTuple,key_matching<K,KHash> >::type {
        typedef typename tbb::flow::tuple_element<0, OutputTuple>::type T0;
        typedef typename tbb::flow::tuple_element<1, OutputTuple>::type T1;
        typedef typename tbb::flow::tuple_element<2, OutputTuple>::type T2;
        typedef typename tbb::flow::tuple_element<3, OutputTuple>::type T3;
        typedef typename tbb::flow::tuple_element<4, OutputTuple>::type T4;
        typedef typename tbb::flow::tuple_element<5, OutputTuple>::type T5;
    public:
        typedef typename wrap_key_tuple_elements<6,key_matching_port,key_matching<K,KHash>,OutputTuple>::type input_ports_type;
        typedef OutputTuple output_type;
    private:
        typedef join_node_base<key_matching<K,KHash> , input_ports_type, output_type > base_type;
        typedef typename internal::type_to_key_function_body<T0, K> *f0_p;
        typedef typename internal::type_to_key_function_body<T1, K> *f1_p;
        typedef typename internal::type_to_key_function_body<T2, K> *f2_p;
        typedef typename internal::type_to_key_function_body<T3, K> *f3_p;
        typedef typename internal::type_to_key_function_body<T4, K> *f4_p;
        typedef typename internal::type_to_key_function_body<T5, K> *f5_p;
        typedef typename tbb::flow::tuple< f0_p, f1_p, f2_p, f3_p, f4_p, f5_p > func_initializer_type;
    public:
#if __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING
        unfolded_join_node(graph &g) : base_type(g,
                func_initializer_type(
                    new internal::type_to_key_function_body_leaf<T0, K, key_from_message_body<K,T0> >(key_from_message_body<K,T0>()),
                    new internal::type_to_key_function_body_leaf<T1, K, key_from_message_body<K,T1> >(key_from_message_body<K,T1>()),
                    new internal::type_to_key_function_body_leaf<T2, K, key_from_message_body<K,T2> >(key_from_message_body<K,T2>()),
                    new internal::type_to_key_function_body_leaf<T3, K, key_from_message_body<K,T3> >(key_from_message_body<K,T3>()),
                    new internal::type_to_key_function_body_leaf<T4, K, key_from_message_body<K,T4> >(key_from_message_body<K,T4>()),
                    new internal::type_to_key_function_body_leaf<T5, K, key_from_message_body<K,T5> >(key_from_message_body<K,T5>())
                    ) ) {
        }
#endif /* __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING */
        template<typename Body0, typename Body1, typename Body2, typename Body3, typename Body4, typename Body5>
        unfolded_join_node(graph &g, Body0 body0, Body1 body1, Body2 body2, Body3 body3, Body4 body4, Body5 body5)
                : base_type(g, func_initializer_type(
                    new internal::type_to_key_function_body_leaf<T0, K, Body0>(body0),
                    new internal::type_to_key_function_body_leaf<T1, K, Body1>(body1),
                    new internal::type_to_key_function_body_leaf<T2, K, Body2>(body2),
                    new internal::type_to_key_function_body_leaf<T3, K, Body3>(body3),
                    new internal::type_to_key_function_body_leaf<T4, K, Body4>(body4),
                    new internal::type_to_key_function_body_leaf<T5, K, Body5>(body5)
                    ) ) {
            __TBB_STATIC_ASSERT(tbb::flow::tuple_size<OutputTuple>::value == 6, "wrong number of body initializers");
        }
        unfolded_join_node(const unfolded_join_node &other) : base_type(other) {}
    };
#endif

#if __TBB_VARIADIC_MAX >= 7
    template<typename OutputTuple, typename K, typename KHash>
    class unfolded_join_node<7,key_matching_port,OutputTuple,key_matching<K,KHash> > : public
            join_base<7,key_matching_port,OutputTuple,key_matching<K,KHash> >::type {
        typedef typename tbb::flow::tuple_element<0, OutputTuple>::type T0;
        typedef typename tbb::flow::tuple_element<1, OutputTuple>::type T1;
        typedef typename tbb::flow::tuple_element<2, OutputTuple>::type T2;
        typedef typename tbb::flow::tuple_element<3, OutputTuple>::type T3;
        typedef typename tbb::flow::tuple_element<4, OutputTuple>::type T4;
        typedef typename tbb::flow::tuple_element<5, OutputTuple>::type T5;
        typedef typename tbb::flow::tuple_element<6, OutputTuple>::type T6;
    public:
        typedef typename wrap_key_tuple_elements<7,key_matching_port,key_matching<K,KHash>,OutputTuple>::type input_ports_type;
        typedef OutputTuple output_type;
    private:
        typedef join_node_base<key_matching<K,KHash> , input_ports_type, output_type > base_type;
        typedef typename internal::type_to_key_function_body<T0, K> *f0_p;
        typedef typename internal::type_to_key_function_body<T1, K> *f1_p;
        typedef typename internal::type_to_key_function_body<T2, K> *f2_p;
        typedef typename internal::type_to_key_function_body<T3, K> *f3_p;
        typedef typename internal::type_to_key_function_body<T4, K> *f4_p;
        typedef typename internal::type_to_key_function_body<T5, K> *f5_p;
        typedef typename internal::type_to_key_function_body<T6, K> *f6_p;
        typedef typename tbb::flow::tuple< f0_p, f1_p, f2_p, f3_p, f4_p, f5_p, f6_p > func_initializer_type;
    public:
#if __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING
        unfolded_join_node(graph &g) : base_type(g,
                func_initializer_type(
                    new internal::type_to_key_function_body_leaf<T0, K, key_from_message_body<K,T0> >(key_from_message_body<K,T0>()),
                    new internal::type_to_key_function_body_leaf<T1, K, key_from_message_body<K,T1> >(key_from_message_body<K,T1>()),
                    new internal::type_to_key_function_body_leaf<T2, K, key_from_message_body<K,T2> >(key_from_message_body<K,T2>()),
                    new internal::type_to_key_function_body_leaf<T3, K, key_from_message_body<K,T3> >(key_from_message_body<K,T3>()),
                    new internal::type_to_key_function_body_leaf<T4, K, key_from_message_body<K,T4> >(key_from_message_body<K,T4>()),
                    new internal::type_to_key_function_body_leaf<T5, K, key_from_message_body<K,T5> >(key_from_message_body<K,T5>()),
                    new internal::type_to_key_function_body_leaf<T6, K, key_from_message_body<K,T6> >(key_from_message_body<K,T6>())
                    ) ) {
        }
#endif /* __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING */
        template<typename Body0, typename Body1, typename Body2, typename Body3, typename Body4,
                 typename Body5, typename Body6>
        unfolded_join_node(graph &g, Body0 body0, Body1 body1, Body2 body2, Body3 body3, Body4 body4,
                Body5 body5, Body6 body6) : base_type(g, func_initializer_type(
                    new internal::type_to_key_function_body_leaf<T0, K, Body0>(body0),
                    new internal::type_to_key_function_body_leaf<T1, K, Body1>(body1),
                    new internal::type_to_key_function_body_leaf<T2, K, Body2>(body2),
                    new internal::type_to_key_function_body_leaf<T3, K, Body3>(body3),
                    new internal::type_to_key_function_body_leaf<T4, K, Body4>(body4),
                    new internal::type_to_key_function_body_leaf<T5, K, Body5>(body5),
                    new internal::type_to_key_function_body_leaf<T6, K, Body6>(body6)
                    ) ) {
            __TBB_STATIC_ASSERT(tbb::flow::tuple_size<OutputTuple>::value == 7, "wrong number of body initializers");
        }
        unfolded_join_node(const unfolded_join_node &other) : base_type(other) {}
    };
#endif

#if __TBB_VARIADIC_MAX >= 8
    template<typename OutputTuple, typename K, typename KHash>
    class unfolded_join_node<8,key_matching_port,OutputTuple,key_matching<K,KHash> > : public
            join_base<8,key_matching_port,OutputTuple,key_matching<K,KHash> >::type {
        typedef typename tbb::flow::tuple_element<0, OutputTuple>::type T0;
        typedef typename tbb::flow::tuple_element<1, OutputTuple>::type T1;
        typedef typename tbb::flow::tuple_element<2, OutputTuple>::type T2;
        typedef typename tbb::flow::tuple_element<3, OutputTuple>::type T3;
        typedef typename tbb::flow::tuple_element<4, OutputTuple>::type T4;
        typedef typename tbb::flow::tuple_element<5, OutputTuple>::type T5;
        typedef typename tbb::flow::tuple_element<6, OutputTuple>::type T6;
        typedef typename tbb::flow::tuple_element<7, OutputTuple>::type T7;
    public:
        typedef typename wrap_key_tuple_elements<8,key_matching_port,key_matching<K,KHash>,OutputTuple>::type input_ports_type;
        typedef OutputTuple output_type;
    private:
        typedef join_node_base<key_matching<K,KHash> , input_ports_type, output_type > base_type;
        typedef typename internal::type_to_key_function_body<T0, K> *f0_p;
        typedef typename internal::type_to_key_function_body<T1, K> *f1_p;
        typedef typename internal::type_to_key_function_body<T2, K> *f2_p;
        typedef typename internal::type_to_key_function_body<T3, K> *f3_p;
        typedef typename internal::type_to_key_function_body<T4, K> *f4_p;
        typedef typename internal::type_to_key_function_body<T5, K> *f5_p;
        typedef typename internal::type_to_key_function_body<T6, K> *f6_p;
        typedef typename internal::type_to_key_function_body<T7, K> *f7_p;
        typedef typename tbb::flow::tuple< f0_p, f1_p, f2_p, f3_p, f4_p, f5_p, f6_p, f7_p > func_initializer_type;
    public:
#if __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING
        unfolded_join_node(graph &g) : base_type(g,
                func_initializer_type(
                    new internal::type_to_key_function_body_leaf<T0, K, key_from_message_body<K,T0> >(key_from_message_body<K,T0>()),
                    new internal::type_to_key_function_body_leaf<T1, K, key_from_message_body<K,T1> >(key_from_message_body<K,T1>()),
                    new internal::type_to_key_function_body_leaf<T2, K, key_from_message_body<K,T2> >(key_from_message_body<K,T2>()),
                    new internal::type_to_key_function_body_leaf<T3, K, key_from_message_body<K,T3> >(key_from_message_body<K,T3>()),
                    new internal::type_to_key_function_body_leaf<T4, K, key_from_message_body<K,T4> >(key_from_message_body<K,T4>()),
                    new internal::type_to_key_function_body_leaf<T5, K, key_from_message_body<K,T5> >(key_from_message_body<K,T5>()),
                    new internal::type_to_key_function_body_leaf<T6, K, key_from_message_body<K,T6> >(key_from_message_body<K,T6>()),
                    new internal::type_to_key_function_body_leaf<T7, K, key_from_message_body<K,T7> >(key_from_message_body<K,T7>())
                    ) ) {
        }
#endif /* __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING */
        template<typename Body0, typename Body1, typename Body2, typename Body3, typename Body4,
                 typename Body5, typename Body6, typename Body7>
        unfolded_join_node(graph &g, Body0 body0, Body1 body1, Body2 body2, Body3 body3, Body4 body4,
                Body5 body5, Body6 body6, Body7 body7) : base_type(g, func_initializer_type(
                    new internal::type_to_key_function_body_leaf<T0, K, Body0>(body0),
                    new internal::type_to_key_function_body_leaf<T1, K, Body1>(body1),
                    new internal::type_to_key_function_body_leaf<T2, K, Body2>(body2),
                    new internal::type_to_key_function_body_leaf<T3, K, Body3>(body3),
                    new internal::type_to_key_function_body_leaf<T4, K, Body4>(body4),
                    new internal::type_to_key_function_body_leaf<T5, K, Body5>(body5),
                    new internal::type_to_key_function_body_leaf<T6, K, Body6>(body6),
                    new internal::type_to_key_function_body_leaf<T7, K, Body7>(body7)
                    ) ) {
            __TBB_STATIC_ASSERT(tbb::flow::tuple_size<OutputTuple>::value == 8, "wrong number of body initializers");
        }
        unfolded_join_node(const unfolded_join_node &other) : base_type(other) {}
    };
#endif

#if __TBB_VARIADIC_MAX >= 9
    template<typename OutputTuple, typename K, typename KHash>
    class unfolded_join_node<9,key_matching_port,OutputTuple,key_matching<K,KHash> > : public
            join_base<9,key_matching_port,OutputTuple,key_matching<K,KHash> >::type {
        typedef typename tbb::flow::tuple_element<0, OutputTuple>::type T0;
        typedef typename tbb::flow::tuple_element<1, OutputTuple>::type T1;
        typedef typename tbb::flow::tuple_element<2, OutputTuple>::type T2;
        typedef typename tbb::flow::tuple_element<3, OutputTuple>::type T3;
        typedef typename tbb::flow::tuple_element<4, OutputTuple>::type T4;
        typedef typename tbb::flow::tuple_element<5, OutputTuple>::type T5;
        typedef typename tbb::flow::tuple_element<6, OutputTuple>::type T6;
        typedef typename tbb::flow::tuple_element<7, OutputTuple>::type T7;
        typedef typename tbb::flow::tuple_element<8, OutputTuple>::type T8;
    public:
        typedef typename wrap_key_tuple_elements<9,key_matching_port,key_matching<K,KHash>,OutputTuple>::type input_ports_type;
        typedef OutputTuple output_type;
    private:
        typedef join_node_base<key_matching<K,KHash> , input_ports_type, output_type > base_type;
        typedef typename internal::type_to_key_function_body<T0, K> *f0_p;
        typedef typename internal::type_to_key_function_body<T1, K> *f1_p;
        typedef typename internal::type_to_key_function_body<T2, K> *f2_p;
        typedef typename internal::type_to_key_function_body<T3, K> *f3_p;
        typedef typename internal::type_to_key_function_body<T4, K> *f4_p;
        typedef typename internal::type_to_key_function_body<T5, K> *f5_p;
        typedef typename internal::type_to_key_function_body<T6, K> *f6_p;
        typedef typename internal::type_to_key_function_body<T7, K> *f7_p;
        typedef typename internal::type_to_key_function_body<T8, K> *f8_p;
        typedef typename tbb::flow::tuple< f0_p, f1_p, f2_p, f3_p, f4_p, f5_p, f6_p, f7_p, f8_p > func_initializer_type;
    public:
#if __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING
        unfolded_join_node(graph &g) : base_type(g,
                func_initializer_type(
                    new internal::type_to_key_function_body_leaf<T0, K, key_from_message_body<K,T0> >(key_from_message_body<K,T0>()),
                    new internal::type_to_key_function_body_leaf<T1, K, key_from_message_body<K,T1> >(key_from_message_body<K,T1>()),
                    new internal::type_to_key_function_body_leaf<T2, K, key_from_message_body<K,T2> >(key_from_message_body<K,T2>()),
                    new internal::type_to_key_function_body_leaf<T3, K, key_from_message_body<K,T3> >(key_from_message_body<K,T3>()),
                    new internal::type_to_key_function_body_leaf<T4, K, key_from_message_body<K,T4> >(key_from_message_body<K,T4>()),
                    new internal::type_to_key_function_body_leaf<T5, K, key_from_message_body<K,T5> >(key_from_message_body<K,T5>()),
                    new internal::type_to_key_function_body_leaf<T6, K, key_from_message_body<K,T6> >(key_from_message_body<K,T6>()),
                    new internal::type_to_key_function_body_leaf<T7, K, key_from_message_body<K,T7> >(key_from_message_body<K,T7>()),
                    new internal::type_to_key_function_body_leaf<T8, K, key_from_message_body<K,T8> >(key_from_message_body<K,T8>())
                    ) ) {
        }
#endif /* __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING */
        template<typename Body0, typename Body1, typename Body2, typename Body3, typename Body4,
                 typename Body5, typename Body6, typename Body7, typename Body8>
        unfolded_join_node(graph &g, Body0 body0, Body1 body1, Body2 body2, Body3 body3, Body4 body4,
                Body5 body5, Body6 body6, Body7 body7, Body8 body8) : base_type(g, func_initializer_type(
                    new internal::type_to_key_function_body_leaf<T0, K, Body0>(body0),
                    new internal::type_to_key_function_body_leaf<T1, K, Body1>(body1),
                    new internal::type_to_key_function_body_leaf<T2, K, Body2>(body2),
                    new internal::type_to_key_function_body_leaf<T3, K, Body3>(body3),
                    new internal::type_to_key_function_body_leaf<T4, K, Body4>(body4),
                    new internal::type_to_key_function_body_leaf<T5, K, Body5>(body5),
                    new internal::type_to_key_function_body_leaf<T6, K, Body6>(body6),
                    new internal::type_to_key_function_body_leaf<T7, K, Body7>(body7),
                    new internal::type_to_key_function_body_leaf<T8, K, Body8>(body8)
                    ) ) {
            __TBB_STATIC_ASSERT(tbb::flow::tuple_size<OutputTuple>::value == 9, "wrong number of body initializers");
        }
        unfolded_join_node(const unfolded_join_node &other) : base_type(other) {}
    };
#endif

#if __TBB_VARIADIC_MAX >= 10
    template<typename OutputTuple, typename K, typename KHash>
    class unfolded_join_node<10,key_matching_port,OutputTuple,key_matching<K,KHash> > : public
            join_base<10,key_matching_port,OutputTuple,key_matching<K,KHash> >::type {
        typedef typename tbb::flow::tuple_element<0, OutputTuple>::type T0;
        typedef typename tbb::flow::tuple_element<1, OutputTuple>::type T1;
        typedef typename tbb::flow::tuple_element<2, OutputTuple>::type T2;
        typedef typename tbb::flow::tuple_element<3, OutputTuple>::type T3;
        typedef typename tbb::flow::tuple_element<4, OutputTuple>::type T4;
        typedef typename tbb::flow::tuple_element<5, OutputTuple>::type T5;
        typedef typename tbb::flow::tuple_element<6, OutputTuple>::type T6;
        typedef typename tbb::flow::tuple_element<7, OutputTuple>::type T7;
        typedef typename tbb::flow::tuple_element<8, OutputTuple>::type T8;
        typedef typename tbb::flow::tuple_element<9, OutputTuple>::type T9;
    public:
        typedef typename wrap_key_tuple_elements<10,key_matching_port,key_matching<K,KHash>,OutputTuple>::type input_ports_type;
        typedef OutputTuple output_type;
    private:
        typedef join_node_base<key_matching<K,KHash> , input_ports_type, output_type > base_type;
        typedef typename internal::type_to_key_function_body<T0, K> *f0_p;
        typedef typename internal::type_to_key_function_body<T1, K> *f1_p;
        typedef typename internal::type_to_key_function_body<T2, K> *f2_p;
        typedef typename internal::type_to_key_function_body<T3, K> *f3_p;
        typedef typename internal::type_to_key_function_body<T4, K> *f4_p;
        typedef typename internal::type_to_key_function_body<T5, K> *f5_p;
        typedef typename internal::type_to_key_function_body<T6, K> *f6_p;
        typedef typename internal::type_to_key_function_body<T7, K> *f7_p;
        typedef typename internal::type_to_key_function_body<T8, K> *f8_p;
        typedef typename internal::type_to_key_function_body<T9, K> *f9_p;
        typedef typename tbb::flow::tuple< f0_p, f1_p, f2_p, f3_p, f4_p, f5_p, f6_p, f7_p, f8_p, f9_p > func_initializer_type;
    public:
#if __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING
        unfolded_join_node(graph &g) : base_type(g,
                func_initializer_type(
                    new internal::type_to_key_function_body_leaf<T0, K, key_from_message_body<K,T0> >(key_from_message_body<K,T0>()),
                    new internal::type_to_key_function_body_leaf<T1, K, key_from_message_body<K,T1> >(key_from_message_body<K,T1>()),
                    new internal::type_to_key_function_body_leaf<T2, K, key_from_message_body<K,T2> >(key_from_message_body<K,T2>()),
                    new internal::type_to_key_function_body_leaf<T3, K, key_from_message_body<K,T3> >(key_from_message_body<K,T3>()),
                    new internal::type_to_key_function_body_leaf<T4, K, key_from_message_body<K,T4> >(key_from_message_body<K,T4>()),
                    new internal::type_to_key_function_body_leaf<T5, K, key_from_message_body<K,T5> >(key_from_message_body<K,T5>()),
                    new internal::type_to_key_function_body_leaf<T6, K, key_from_message_body<K,T6> >(key_from_message_body<K,T6>()),
                    new internal::type_to_key_function_body_leaf<T7, K, key_from_message_body<K,T7> >(key_from_message_body<K,T7>()),
                    new internal::type_to_key_function_body_leaf<T8, K, key_from_message_body<K,T8> >(key_from_message_body<K,T8>()),
                    new internal::type_to_key_function_body_leaf<T9, K, key_from_message_body<K,T9> >(key_from_message_body<K,T9>())
                    ) ) {
        }
#endif /* __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING */
        template<typename Body0, typename Body1, typename Body2, typename Body3, typename Body4,
            typename Body5, typename Body6, typename Body7, typename Body8, typename Body9>
        unfolded_join_node(graph &g, Body0 body0, Body1 body1, Body2 body2, Body3 body3, Body4 body4,
                Body5 body5, Body6 body6, Body7 body7, Body8 body8, Body9 body9) : base_type(g, func_initializer_type(
                    new internal::type_to_key_function_body_leaf<T0, K, Body0>(body0),
                    new internal::type_to_key_function_body_leaf<T1, K, Body1>(body1),
                    new internal::type_to_key_function_body_leaf<T2, K, Body2>(body2),
                    new internal::type_to_key_function_body_leaf<T3, K, Body3>(body3),
                    new internal::type_to_key_function_body_leaf<T4, K, Body4>(body4),
                    new internal::type_to_key_function_body_leaf<T5, K, Body5>(body5),
                    new internal::type_to_key_function_body_leaf<T6, K, Body6>(body6),
                    new internal::type_to_key_function_body_leaf<T7, K, Body7>(body7),
                    new internal::type_to_key_function_body_leaf<T8, K, Body8>(body8),
                    new internal::type_to_key_function_body_leaf<T9, K, Body9>(body9)
                    ) ) {
            __TBB_STATIC_ASSERT(tbb::flow::tuple_size<OutputTuple>::value == 10, "wrong number of body initializers");
        }
        unfolded_join_node(const unfolded_join_node &other) : base_type(other) {}
    };
#endif

    //! templated function to refer to input ports of the join node
    template<size_t N, typename JNT>
    typename tbb::flow::tuple_element<N, typename JNT::input_ports_type>::type &input_port(JNT &jn) {
        return tbb::flow::get<N>(jn.input_ports());
    }

}
#endif // __TBB__flow_graph_join_impl_H

