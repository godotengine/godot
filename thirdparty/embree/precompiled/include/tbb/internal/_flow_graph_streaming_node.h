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

#ifndef __TBB_flow_graph_streaming_H
#define __TBB_flow_graph_streaming_H

#ifndef __TBB_flow_graph_H
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

#if __TBB_PREVIEW_STREAMING_NODE

// Included in namespace tbb::flow::interfaceX (in flow_graph.h)

namespace internal {

template <int N1, int N2>
struct port_ref_impl {
    // "+1" since the port_ref range is a closed interval (includes its endpoints).
    static const int size = N2 - N1 + 1;
};

} // internal

// The purpose of the port_ref_impl is the pretty syntax: the deduction of a compile-time constant is processed from the return type.
// So it is possible to use this helper without parentheses, e.g. "port_ref<0>".
template <int N1, int N2 = N1>
__TBB_DEPRECATED internal::port_ref_impl<N1,N2> port_ref() {
    return internal::port_ref_impl<N1,N2>();
};

namespace internal {

template <typename T>
struct num_arguments {
    static const int value = 1;
};

template <int N1, int N2>
struct num_arguments<port_ref_impl<N1,N2>(*)()> {
    static const int value = port_ref_impl<N1,N2>::size;
};

template <int N1, int N2>
struct num_arguments<port_ref_impl<N1,N2>> {
    static const int value = port_ref_impl<N1,N2>::size;
};

template <typename... Args>
void ignore_return_values( Args&&... ) {}

template <typename T>
T or_return_values( T&& t ) { return t; }
template <typename T, typename... Rest>
T or_return_values( T&& t, Rest&&... rest ) {
    return t | or_return_values( std::forward<Rest>(rest)... );
}

template<typename JP>
struct key_from_policy {
    typedef size_t type;
    typedef std::false_type is_key_matching;
};

template<typename Key>
struct key_from_policy< key_matching<Key> > {
    typedef Key type;
    typedef std::true_type is_key_matching;
};

template<typename Key>
struct key_from_policy< key_matching<Key&> > {
    typedef const Key &type;
    typedef std::true_type is_key_matching;
};

template<typename Device, typename Key>
class streaming_device_with_key {
    Device my_device;
    typename std::decay<Key>::type my_key;
public:
    // TODO: investigate why default constructor is required
    streaming_device_with_key() {}
    streaming_device_with_key( const Device& d, Key k ) : my_device( d ), my_key( k ) {}
    Key key() const { return my_key; }
    const Device& device() const { return my_device; }
};

// --------- Kernel argument helpers --------- //
template <typename T>
struct is_port_ref_impl {
    typedef std::false_type type;
};

template <int N1, int N2>
struct is_port_ref_impl< port_ref_impl<N1, N2> > {
    typedef std::true_type type;
};

template <int N1, int N2>
struct is_port_ref_impl< port_ref_impl<N1, N2>( * )()  > {
    typedef std::true_type type;
};

template <typename T>
struct is_port_ref {
    typedef typename is_port_ref_impl< typename tbb::internal::strip<T>::type >::type type;
};

template <typename ...Args1>
struct convert_and_call_impl;

template <typename A1, typename ...Args1>
struct convert_and_call_impl<A1, Args1...> {
    static const size_t my_delta = 1; // Index 0 contains device

    template <typename F, typename Tuple, typename ...Args2>
    static void doit(F& f, Tuple& t, A1& a1, Args1&... args1, Args2&... args2) {
        convert_and_call_impl<A1, Args1...>::doit_impl(typename is_port_ref<A1>::type(), f, t, a1, args1..., args2...);
    }
    template <typename F, typename Tuple, typename ...Args2>
    static void doit_impl(std::false_type, F& f, Tuple& t, A1& a1, Args1&... args1, Args2&... args2) {
        convert_and_call_impl<Args1...>::doit(f, t, args1..., args2..., a1);
    }
    template <typename F, typename Tuple, int N1, int N2, typename ...Args2>
    static void doit_impl(std::true_type x, F& f, Tuple& t, port_ref_impl<N1, N2>, Args1&... args1, Args2&... args2) {
        convert_and_call_impl<port_ref_impl<N1 + 1,N2>, Args1...>::doit_impl(x, f, t, port_ref<N1 + 1, N2>(), args1...,
            args2..., std::get<N1 + my_delta>(t));
    }
    template <typename F, typename Tuple, int N, typename ...Args2>
    static void doit_impl(std::true_type, F& f, Tuple& t, port_ref_impl<N, N>, Args1&... args1, Args2&... args2) {
        convert_and_call_impl<Args1...>::doit(f, t, args1..., args2..., std::get<N + my_delta>(t));
    }

    template <typename F, typename Tuple, int N1, int N2, typename ...Args2>
    static void doit_impl(std::true_type x, F& f, Tuple& t, port_ref_impl<N1, N2>(* fn)(), Args1&... args1, Args2&... args2) {
        doit_impl(x, f, t, fn(), args1..., args2...);
    }
    template <typename F, typename Tuple, int N, typename ...Args2>
    static void doit_impl(std::true_type x, F& f, Tuple& t, port_ref_impl<N, N>(* fn)(), Args1&... args1, Args2&... args2) {
        doit_impl(x, f, t, fn(), args1..., args2...);
    }
};

template <>
struct convert_and_call_impl<> {
    template <typename F, typename Tuple, typename ...Args2>
    static void doit(F& f, Tuple&, Args2&... args2) {
        f(args2...);
    }
};
// ------------------------------------------- //

template<typename JP, typename StreamFactory, typename... Ports>
struct streaming_node_traits {
    // Do not use 'using' instead of 'struct' because Microsoft Visual C++ 12.0 fails to compile.
    template <typename T>
    struct async_msg_type {
        typedef typename StreamFactory::template async_msg_type<T> type;
    };

    typedef tuple< typename async_msg_type<Ports>::type... > input_tuple;
    typedef input_tuple output_tuple;
    typedef tuple< streaming_device_with_key< typename StreamFactory::device_type, typename key_from_policy<JP>::type >,
        typename async_msg_type<Ports>::type... > kernel_input_tuple;

    // indexer_node parameters pack expansion workaround for VS2013 for streaming_node
    typedef indexer_node< typename async_msg_type<Ports>::type... > indexer_node_type;
};

// Default empty implementation
template<typename StreamFactory, typename KernelInputTuple, typename = void>
class kernel_executor_helper {
    typedef typename StreamFactory::device_type device_type;
    typedef typename StreamFactory::kernel_type kernel_type;
    typedef KernelInputTuple kernel_input_tuple;
protected:
    template <typename ...Args>
    void enqueue_kernel_impl( kernel_input_tuple&, StreamFactory& factory, device_type device, const kernel_type& kernel, Args&... args ) const {
        factory.send_kernel( device, kernel, args... );
    }
};

// Implementation for StreamFactory supporting range
template<typename StreamFactory, typename KernelInputTuple>
class kernel_executor_helper<StreamFactory, KernelInputTuple, typename tbb::internal::void_t< typename StreamFactory::range_type >::type > {
    typedef typename StreamFactory::device_type device_type;
    typedef typename StreamFactory::kernel_type kernel_type;
    typedef KernelInputTuple kernel_input_tuple;

    typedef typename StreamFactory::range_type range_type;

    // Container for randge. It can contain either port references or real range.
    struct range_wrapper {
        virtual range_type get_range( const kernel_input_tuple &ip ) const = 0;
        virtual range_wrapper *clone() const = 0;
        virtual ~range_wrapper() {}
    };

    struct range_value : public range_wrapper {
        range_value( const range_type& value ) : my_value(value) {}

        range_value( range_type&& value ) : my_value(std::move(value)) {}

        range_type get_range( const kernel_input_tuple & ) const __TBB_override {
            return my_value;
        }

        range_wrapper *clone() const __TBB_override {
            return new range_value(my_value);
        }
    private:
        range_type my_value;
    };

    template <int N>
    struct range_mapper : public range_wrapper {
        range_mapper() {}

        range_type get_range( const kernel_input_tuple &ip ) const __TBB_override {
            // "+1" since get<0>(ip) is StreamFactory::device.
            return get<N + 1>(ip).data(false);
        }

        range_wrapper *clone() const __TBB_override {
            return new range_mapper<N>;
        }
    };

protected:
    template <typename ...Args>
    void enqueue_kernel_impl( kernel_input_tuple& ip, StreamFactory& factory, device_type device, const kernel_type& kernel, Args&... args ) const {
        __TBB_ASSERT(my_range_wrapper, "Range is not set. Call set_range() before running streaming_node.");
        factory.send_kernel( device, kernel, my_range_wrapper->get_range(ip), args... );
    }

public:
    kernel_executor_helper() : my_range_wrapper(NULL) {}

    kernel_executor_helper(const kernel_executor_helper& executor) : my_range_wrapper(executor.my_range_wrapper ? executor.my_range_wrapper->clone() : NULL) {}

    kernel_executor_helper(kernel_executor_helper&& executor) : my_range_wrapper(executor.my_range_wrapper) {
        // Set moving holder mappers to NULL to prevent double deallocation
        executor.my_range_wrapper = NULL;
    }

    ~kernel_executor_helper() {
        if (my_range_wrapper) delete my_range_wrapper;
    }

    void set_range(const range_type& work_size) {
        my_range_wrapper = new range_value(work_size);
    }

    void set_range(range_type&& work_size) {
        my_range_wrapper = new range_value(std::move(work_size));
    }

    template <int N>
    void set_range(port_ref_impl<N, N>) {
        my_range_wrapper = new range_mapper<N>;
    }

    template <int N>
    void set_range(port_ref_impl<N, N>(*)()) {
        my_range_wrapper = new range_mapper<N>;
    }

private:
    range_wrapper* my_range_wrapper;
};

} // internal

/*
/---------------------------------------- streaming_node ------------------------------------\
|                                                                                            |
|   /--------------\   /----------------------\   /-----------\   /----------------------\   |
|   |              |   |    (device_with_key) O---O           |   |                      |   |
|   |              |   |                      |   |           |   |                      |   |
O---O indexer_node O---O device_selector_node O---O join_node O---O      kernel_node     O---O
|   |              |   | (multifunction_node) |   |           |   | (multifunction_node) |   |
O---O              |   |                      O---O           |   |                      O---O
|   \--------------/   \----------------------/   \-----------/   \----------------------/   |
|                                                                                            |
\--------------------------------------------------------------------------------------------/
*/
template<typename... Args>
class __TBB_DEPRECATED streaming_node;

template<typename... Ports, typename JP, typename StreamFactory>
class __TBB_DEPRECATED
streaming_node< tuple<Ports...>, JP, StreamFactory >
    : public composite_node < typename internal::streaming_node_traits<JP, StreamFactory, Ports...>::input_tuple,
                              typename internal::streaming_node_traits<JP, StreamFactory, Ports...>::output_tuple >
    , public internal::kernel_executor_helper< StreamFactory, typename internal::streaming_node_traits<JP, StreamFactory, Ports...>::kernel_input_tuple >
{
    typedef typename internal::streaming_node_traits<JP, StreamFactory, Ports...>::input_tuple input_tuple;
    typedef typename internal::streaming_node_traits<JP, StreamFactory, Ports...>::output_tuple output_tuple;
    typedef typename internal::key_from_policy<JP>::type key_type;
protected:
    typedef typename StreamFactory::device_type device_type;
    typedef typename StreamFactory::kernel_type kernel_type;
private:
    typedef internal::streaming_device_with_key<device_type, key_type> device_with_key_type;
    typedef composite_node<input_tuple, output_tuple> base_type;
    static const size_t NUM_INPUTS = tuple_size<input_tuple>::value;
    static const size_t NUM_OUTPUTS = tuple_size<output_tuple>::value;

    typedef typename internal::make_sequence<NUM_INPUTS>::type input_sequence;
    typedef typename internal::make_sequence<NUM_OUTPUTS>::type output_sequence;

    typedef typename internal::streaming_node_traits<JP, StreamFactory, Ports...>::indexer_node_type indexer_node_type;
    typedef typename indexer_node_type::output_type indexer_node_output_type;
    typedef typename internal::streaming_node_traits<JP, StreamFactory, Ports...>::kernel_input_tuple kernel_input_tuple;
    typedef multifunction_node<indexer_node_output_type, kernel_input_tuple> device_selector_node;
    typedef multifunction_node<kernel_input_tuple, output_tuple> kernel_multifunction_node;

    template <int... S>
    typename base_type::input_ports_type get_input_ports( internal::sequence<S...> ) {
        return std::tie( internal::input_port<S>( my_indexer_node )... );
    }

    template <int... S>
    typename base_type::output_ports_type get_output_ports( internal::sequence<S...> ) {
        return std::tie( internal::output_port<S>( my_kernel_node )... );
    }

    typename base_type::input_ports_type get_input_ports() {
        return get_input_ports( input_sequence() );
    }

    typename base_type::output_ports_type get_output_ports() {
        return get_output_ports( output_sequence() );
    }

    template <int N>
    int make_Nth_edge() {
        make_edge( internal::output_port<N>( my_device_selector_node ), internal::input_port<N>( my_join_node ) );
        return 0;
    }

    template <int... S>
    void make_edges( internal::sequence<S...> ) {
        make_edge( my_indexer_node, my_device_selector_node );
        make_edge( my_device_selector_node, my_join_node );
        internal::ignore_return_values( make_Nth_edge<S + 1>()... );
        make_edge( my_join_node, my_kernel_node );
    }

    void make_edges() {
        make_edges( input_sequence() );
    }

    class device_selector_base {
    public:
        virtual void operator()( const indexer_node_output_type &v, typename device_selector_node::output_ports_type &op ) = 0;
        virtual device_selector_base *clone( streaming_node &n ) const = 0;
        virtual ~device_selector_base() {}
    };

    template <typename UserFunctor>
    class device_selector : public device_selector_base, tbb::internal::no_assign {
    public:
        device_selector( UserFunctor uf, streaming_node &n, StreamFactory &f )
            : my_dispatch_funcs( create_dispatch_funcs( input_sequence() ) )
            , my_user_functor( uf ), my_node(n), my_factory( f )
        {
            my_port_epoches.fill( 0 );
        }

        void operator()( const indexer_node_output_type &v, typename device_selector_node::output_ports_type &op ) __TBB_override {
            (this->*my_dispatch_funcs[ v.tag() ])( my_port_epoches[ v.tag() ], v, op );
            __TBB_ASSERT( (tbb::internal::is_same_type<typename internal::key_from_policy<JP>::is_key_matching, std::false_type>::value)
                || my_port_epoches[v.tag()] == 0, "Epoch is changed when key matching is requested" );
        }

        device_selector_base *clone( streaming_node &n ) const __TBB_override {
            return new device_selector( my_user_functor, n, my_factory );
        }
    private:
        typedef void(device_selector<UserFunctor>::*send_and_put_fn_type)(size_t &, const indexer_node_output_type &, typename device_selector_node::output_ports_type &);
        typedef std::array < send_and_put_fn_type, NUM_INPUTS > dispatch_funcs_type;

        template <int... S>
        static dispatch_funcs_type create_dispatch_funcs( internal::sequence<S...> ) {
            dispatch_funcs_type dispatch = { { &device_selector<UserFunctor>::send_and_put_impl<S>... } };
            return dispatch;
        }

        template <typename T>
        key_type get_key( std::false_type, const T &, size_t &epoch ) {
            __TBB_STATIC_ASSERT( (tbb::internal::is_same_type<key_type, size_t>::value), "" );
            return epoch++;
        }

        template <typename T>
        key_type get_key( std::true_type, const T &t, size_t &/*epoch*/ ) {
            using tbb::flow::key_from_message;
            return key_from_message<key_type>( t );
        }

        template <int N>
        void send_and_put_impl( size_t &epoch, const indexer_node_output_type &v, typename device_selector_node::output_ports_type &op ) {
            typedef typename tuple_element<N + 1, typename device_selector_node::output_ports_type>::type::output_type elem_type;
            elem_type e = internal::cast_to<elem_type>( v );
            device_type device = get_device( get_key( typename internal::key_from_policy<JP>::is_key_matching(), e, epoch ), get<0>( op ) );
            my_factory.send_data( device, e );
            get<N + 1>( op ).try_put( e );
        }

        template< typename DevicePort >
        device_type get_device( key_type key, DevicePort& dp ) {
            typename std::unordered_map<typename std::decay<key_type>::type, epoch_desc>::iterator it = my_devices.find( key );
            if ( it == my_devices.end() ) {
                device_type d = my_user_functor( my_factory );
                std::tie( it, std::ignore ) = my_devices.insert( std::make_pair( key, d ) );
                bool res = dp.try_put( device_with_key_type( d, key ) );
                __TBB_ASSERT_EX( res, NULL );
                my_node.notify_new_device( d );
            }
            epoch_desc &e = it->second;
            device_type d = e.my_device;
            if ( ++e.my_request_number == NUM_INPUTS ) my_devices.erase( it );
            return d;
        }

        struct epoch_desc {
            epoch_desc(device_type d ) : my_device( d ), my_request_number( 0 ) {}
            device_type my_device;
            size_t my_request_number;
        };

        std::unordered_map<typename std::decay<key_type>::type, epoch_desc> my_devices;
        std::array<size_t, NUM_INPUTS> my_port_epoches;
        dispatch_funcs_type my_dispatch_funcs;
        UserFunctor my_user_functor;
        streaming_node &my_node;
        StreamFactory &my_factory;
    };

    class device_selector_body {
    public:
        device_selector_body( device_selector_base *d ) : my_device_selector( d ) {}

        void operator()( const indexer_node_output_type &v, typename device_selector_node::output_ports_type &op ) {
            (*my_device_selector)(v, op);
        }
    private:
        device_selector_base *my_device_selector;
    };

    // TODO: investigate why copy-construction is disallowed
    class args_storage_base : tbb::internal::no_copy {
    public:
        typedef typename kernel_multifunction_node::output_ports_type output_ports_type;

        virtual void enqueue( kernel_input_tuple &ip, output_ports_type &op, const streaming_node &n ) = 0;
        virtual void send( device_type d ) = 0;
        virtual args_storage_base *clone() const = 0;
        virtual ~args_storage_base () {}

    protected:
        args_storage_base( const kernel_type& kernel, StreamFactory &f )
            : my_kernel( kernel ), my_factory( f )
        {}

        args_storage_base( const args_storage_base &k )
            : tbb::internal::no_copy(), my_kernel( k.my_kernel ), my_factory( k.my_factory )
        {}

        const kernel_type my_kernel;
        StreamFactory &my_factory;
    };

    template <typename... Args>
    class args_storage : public args_storage_base {
        typedef typename args_storage_base::output_ports_type output_ports_type;

        // ---------- Update events helpers ---------- //
        template <int N>
        bool do_try_put( const kernel_input_tuple& ip, output_ports_type &op ) const {
            const auto& t = get<N + 1>( ip );
            auto &port = get<N>( op );
            return port.try_put( t );
        }

        template <int... S>
        bool do_try_put( const kernel_input_tuple& ip, output_ports_type &op, internal::sequence<S...> ) const {
            return internal::or_return_values( do_try_put<S>( ip, op )... );
        }

        // ------------------------------------------- //
        class run_kernel_func : tbb::internal::no_assign {
        public:
            run_kernel_func( kernel_input_tuple &ip, const streaming_node &node, const args_storage& storage )
                : my_kernel_func( ip, node, storage, get<0>(ip).device() ) {}

            // It is immpossible to use Args... because a function pointer cannot be casted to a function reference implicitly.
            // Allow the compiler to deduce types for function pointers automatically.
            template <typename... FnArgs>
            void operator()( FnArgs&... args ) {
                internal::convert_and_call_impl<FnArgs...>::doit( my_kernel_func, my_kernel_func.my_ip, args... );
            }
        private:
            struct kernel_func : tbb::internal::no_copy {
                kernel_input_tuple &my_ip;
                const streaming_node &my_node;
                const args_storage& my_storage;
                device_type my_device;

                kernel_func( kernel_input_tuple &ip, const streaming_node &node, const args_storage& storage, device_type device )
                    : my_ip( ip ), my_node( node ), my_storage( storage ), my_device( device )
                {}

                template <typename... FnArgs>
                void operator()( FnArgs&... args ) {
                    my_node.enqueue_kernel( my_ip, my_storage.my_factory, my_device, my_storage.my_kernel, args... );
                }
            } my_kernel_func;
        };

        template<typename FinalizeFn>
        class run_finalize_func : tbb::internal::no_assign {
        public:
            run_finalize_func( kernel_input_tuple &ip, StreamFactory &factory, FinalizeFn fn )
                : my_ip( ip ), my_finalize_func( factory, get<0>(ip).device(), fn ) {}

            // It is immpossible to use Args... because a function pointer cannot be casted to a function reference implicitly.
            // Allow the compiler to deduce types for function pointers automatically.
            template <typename... FnArgs>
            void operator()( FnArgs&... args ) {
                internal::convert_and_call_impl<FnArgs...>::doit( my_finalize_func, my_ip, args... );
            }
        private:
            kernel_input_tuple &my_ip;

            struct finalize_func : tbb::internal::no_assign {
                StreamFactory &my_factory;
                device_type my_device;
                FinalizeFn my_fn;

                finalize_func( StreamFactory &factory, device_type device, FinalizeFn fn )
                    : my_factory(factory), my_device(device), my_fn(fn) {}

                template <typename... FnArgs>
                void operator()( FnArgs&... args ) {
                    my_factory.finalize( my_device, my_fn, args... );
                }
            } my_finalize_func;
        };

        template<typename FinalizeFn>
        static run_finalize_func<FinalizeFn> make_run_finalize_func( kernel_input_tuple &ip, StreamFactory &factory, FinalizeFn fn ) {
            return run_finalize_func<FinalizeFn>( ip, factory, fn );
        }

        class send_func : tbb::internal::no_assign {
        public:
            send_func( StreamFactory &factory, device_type d )
                : my_factory(factory), my_device( d ) {}

            template <typename... FnArgs>
            void operator()( FnArgs&... args ) {
                my_factory.send_data( my_device, args... );
            }
        private:
            StreamFactory &my_factory;
            device_type my_device;
        };

    public:
        args_storage( const kernel_type& kernel, StreamFactory &f, Args&&... args )
            : args_storage_base( kernel, f )
            , my_args_pack( std::forward<Args>(args)... )
        {}

        args_storage( const args_storage &k ) : args_storage_base( k ), my_args_pack( k.my_args_pack ) {}

        args_storage( const args_storage_base &k, Args&&... args ) : args_storage_base( k ), my_args_pack( std::forward<Args>(args)... ) {}

        void enqueue( kernel_input_tuple &ip, output_ports_type &op, const streaming_node &n ) __TBB_override {
            // Make const qualified args_pack (from non-const)
            const args_pack_type& const_args_pack = my_args_pack;
            // factory.enqure_kernel() gets
            //  - 'ip' tuple elements by reference and updates it (and 'ip') with dependencies
            //  - arguments (from my_args_pack) by const-reference via const_args_pack
            tbb::internal::call( run_kernel_func( ip, n, *this ), const_args_pack );

            if (! do_try_put( ip, op, input_sequence() ) ) {
                graph& g = n.my_graph;
                // No one message was passed to successors so set a callback to extend the graph lifetime until the kernel completion.
                g.increment_wait_count();

                // factory.finalize() gets
                //  - 'ip' tuple elements by reference, so 'ip' might be changed
                //  - arguments (from my_args_pack) by const-reference via const_args_pack
                tbb::internal::call( make_run_finalize_func(ip, this->my_factory, [&g] {
                    g.decrement_wait_count();
                }), const_args_pack );
            }
        }

        void send( device_type d ) __TBB_override {
            // factory.send() gets arguments by reference and updates these arguments with dependencies
            // (it gets but usually ignores port_ref-s)
            tbb::internal::call( send_func( this->my_factory, d ), my_args_pack );
        }

        args_storage_base *clone() const __TBB_override {
            // Create new args_storage with copying constructor.
            return new args_storage<Args...>( *this );
        }

    private:
        typedef tbb::internal::stored_pack<Args...> args_pack_type;
        args_pack_type my_args_pack;
    };

    // Body for kernel_multifunction_node.
    class kernel_body : tbb::internal::no_assign {
    public:
        kernel_body( const streaming_node &node ) : my_node( node ) {}

        void operator()( kernel_input_tuple ip, typename args_storage_base::output_ports_type &op ) {
            __TBB_ASSERT( (my_node.my_args_storage != NULL), "No arguments storage" );
            // 'ip' is passed by value to create local copy for updating inside enqueue_kernel()
            my_node.my_args_storage->enqueue( ip, op, my_node );
        }
    private:
        const streaming_node &my_node;
    };

    template <typename T, typename U = typename internal::is_port_ref<T>::type >
    struct wrap_to_async {
        typedef T type; // Keep port_ref as it is
    };

    template <typename T>
    struct wrap_to_async<T, std::false_type> {
        typedef typename StreamFactory::template async_msg_type< typename tbb::internal::strip<T>::type > type;
    };

    template <typename... Args>
    args_storage_base *make_args_storage(const args_storage_base& storage, Args&&... args) const {
        // In this variadic template convert all simple types 'T' into 'async_msg_type<T>'
        return new args_storage<Args...>(storage, std::forward<Args>(args)...);
    }

    void notify_new_device( device_type d ) {
        my_args_storage->send( d );
    }

    template <typename ...Args>
    void enqueue_kernel( kernel_input_tuple& ip, StreamFactory& factory, device_type device, const kernel_type& kernel, Args&... args ) const {
        this->enqueue_kernel_impl( ip, factory, device, kernel, args... );
    }

public:
    template <typename DeviceSelector>
    streaming_node( graph &g, const kernel_type& kernel, DeviceSelector d, StreamFactory &f )
        : base_type( g )
        , my_indexer_node( g )
        , my_device_selector( new device_selector<DeviceSelector>( d, *this, f ) )
        , my_device_selector_node( g, serial, device_selector_body( my_device_selector ) )
        , my_join_node( g )
        , my_kernel_node( g, serial, kernel_body( *this ) )
        // By default, streaming_node maps all its ports to the kernel arguments on a one-to-one basis.
        , my_args_storage( make_args_storage( args_storage<>(kernel, f), port_ref<0, NUM_INPUTS - 1>() ) )
    {
        base_type::set_external_ports( get_input_ports(), get_output_ports() );
        make_edges();
    }

    streaming_node( const streaming_node &node )
        : base_type( node.my_graph )
        , my_indexer_node( node.my_indexer_node )
        , my_device_selector( node.my_device_selector->clone( *this ) )
        , my_device_selector_node( node.my_graph, serial, device_selector_body( my_device_selector ) )
        , my_join_node( node.my_join_node )
        , my_kernel_node( node.my_graph, serial, kernel_body( *this ) )
        , my_args_storage( node.my_args_storage->clone() )
    {
        base_type::set_external_ports( get_input_ports(), get_output_ports() );
        make_edges();
    }

    streaming_node( streaming_node &&node )
        : base_type( node.my_graph )
        , my_indexer_node( std::move( node.my_indexer_node ) )
        , my_device_selector( node.my_device_selector->clone(*this) )
        , my_device_selector_node( node.my_graph, serial, device_selector_body( my_device_selector ) )
        , my_join_node( std::move( node.my_join_node ) )
        , my_kernel_node( node.my_graph, serial, kernel_body( *this ) )
        , my_args_storage( node.my_args_storage )
    {
        base_type::set_external_ports( get_input_ports(), get_output_ports() );
        make_edges();
        // Set moving node mappers to NULL to prevent double deallocation.
        node.my_args_storage = NULL;
    }

    ~streaming_node() {
        if ( my_args_storage ) delete my_args_storage;
        if ( my_device_selector ) delete my_device_selector;
    }

    template <typename... Args>
    void set_args( Args&&... args ) {
        // Copy the base class of args_storage and create new storage for "Args...".
        args_storage_base * const new_args_storage = make_args_storage( *my_args_storage, typename wrap_to_async<Args>::type(std::forward<Args>(args))...);
        delete my_args_storage;
        my_args_storage = new_args_storage;
    }

protected:
    void reset_node( reset_flags = rf_reset_protocol ) __TBB_override { __TBB_ASSERT( false, "Not implemented yet" ); }

private:
    indexer_node_type my_indexer_node;
    device_selector_base *my_device_selector;
    device_selector_node my_device_selector_node;
    join_node<kernel_input_tuple, JP> my_join_node;
    kernel_multifunction_node my_kernel_node;

    args_storage_base *my_args_storage;
};

#endif // __TBB_PREVIEW_STREAMING_NODE
#endif // __TBB_flow_graph_streaming_H
