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

#ifndef _FGT_GRAPH_TRACE_IMPL_H
#define _FGT_GRAPH_TRACE_IMPL_H

#include "../tbb_profiling.h"
#if (_MSC_VER >= 1900)
    #include <intrin.h>
#endif

namespace tbb {
    namespace internal {

#if TBB_USE_THREADING_TOOLS
    #if TBB_PREVIEW_FLOW_GRAPH_TRACE
        #if (_MSC_VER >= 1900)
            #define CODEPTR() (_ReturnAddress())
        #elif __TBB_GCC_VERSION >= 40800
            #define CODEPTR() ( __builtin_return_address(0))
        #else
            #define CODEPTR() NULL
        #endif
    #else
        #define CODEPTR() NULL
    #endif /* TBB_PREVIEW_FLOW_GRAPH_TRACE */

static inline void fgt_alias_port(void *node, void *p, bool visible) {
    if(visible)
        itt_relation_add( ITT_DOMAIN_FLOW, node, FLOW_NODE, __itt_relation_is_parent_of, p, FLOW_NODE );
    else
        itt_relation_add( ITT_DOMAIN_FLOW, p, FLOW_NODE, __itt_relation_is_child_of, node, FLOW_NODE );
}

static inline void fgt_composite ( void* codeptr, void *node, void *graph ) {
    itt_make_task_group( ITT_DOMAIN_FLOW, node, FLOW_NODE, graph, FLOW_GRAPH, FLOW_COMPOSITE_NODE );
    suppress_unused_warning( codeptr );
#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    if (codeptr != NULL) {
        register_node_addr(ITT_DOMAIN_FLOW, node, FLOW_NODE, CODE_ADDRESS, &codeptr);
    }
#endif
}

static inline void fgt_internal_alias_input_port( void *node, void *p, string_index name_index ) {
    itt_make_task_group( ITT_DOMAIN_FLOW, p, FLOW_INPUT_PORT, node, FLOW_NODE, name_index );
    itt_relation_add( ITT_DOMAIN_FLOW, node, FLOW_NODE, __itt_relation_is_parent_of, p, FLOW_INPUT_PORT );
}

static inline void fgt_internal_alias_output_port( void *node, void *p, string_index name_index ) {
    itt_make_task_group( ITT_DOMAIN_FLOW, p, FLOW_OUTPUT_PORT, node, FLOW_NODE, name_index );
    itt_relation_add( ITT_DOMAIN_FLOW, node, FLOW_NODE, __itt_relation_is_parent_of, p, FLOW_OUTPUT_PORT );
}

template<typename InputType>
void alias_input_port(void *node, tbb::flow::receiver<InputType>* port, string_index name_index) {
    // TODO: Make fgt_internal_alias_input_port a function template?
    fgt_internal_alias_input_port( node, port, name_index);
}

template < typename PortsTuple, int N >
struct fgt_internal_input_alias_helper {
    static void alias_port( void *node, PortsTuple &ports ) {
        alias_input_port( node, &(tbb::flow::get<N-1>(ports)), static_cast<tbb::internal::string_index>(FLOW_INPUT_PORT_0 + N - 1) );
        fgt_internal_input_alias_helper<PortsTuple, N-1>::alias_port( node, ports );
    }
};

template < typename PortsTuple >
struct fgt_internal_input_alias_helper<PortsTuple, 0> {
    static void alias_port( void * /* node */, PortsTuple & /* ports */ ) { }
};

template<typename OutputType>
void alias_output_port(void *node, tbb::flow::sender<OutputType>* port, string_index name_index) {
    // TODO: Make fgt_internal_alias_output_port a function template?
    fgt_internal_alias_output_port( node, static_cast<void *>(port), name_index);
}

template < typename PortsTuple, int N >
struct fgt_internal_output_alias_helper {
    static void alias_port( void *node, PortsTuple &ports ) {
        alias_output_port( node, &(tbb::flow::get<N-1>(ports)), static_cast<tbb::internal::string_index>(FLOW_OUTPUT_PORT_0 + N - 1) );
        fgt_internal_output_alias_helper<PortsTuple, N-1>::alias_port( node, ports );
    }
};

template < typename PortsTuple >
struct fgt_internal_output_alias_helper<PortsTuple, 0> {
    static void alias_port( void * /*node*/, PortsTuple &/*ports*/ ) {
    }
};

static inline void fgt_internal_create_input_port( void *node, void *p, string_index name_index ) {
    itt_make_task_group( ITT_DOMAIN_FLOW, p, FLOW_INPUT_PORT, node, FLOW_NODE, name_index );
}

static inline void fgt_internal_create_output_port( void* codeptr, void *node, void *p, string_index name_index ) {
    itt_make_task_group(ITT_DOMAIN_FLOW, p, FLOW_OUTPUT_PORT, node, FLOW_NODE, name_index);
    suppress_unused_warning( codeptr );
#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    if (codeptr != NULL) {
        register_node_addr(ITT_DOMAIN_FLOW, node, FLOW_NODE, CODE_ADDRESS, &codeptr);
    }
#endif
}

template<typename InputType>
void register_input_port(void *node, tbb::flow::receiver<InputType>* port, string_index name_index) {
    // TODO: Make fgt_internal_create_input_port a function template?
    // In C++03 dependent name lookup from the template definition context
    // works only for function declarations with external linkage:
    // http://www.open-std.org/JTC1/SC22/WG21/docs/cwg_defects.html#561
    fgt_internal_create_input_port(node, static_cast<void*>(port), name_index);
}

template < typename PortsTuple, int N >
struct fgt_internal_input_helper {
    static void register_port( void *node, PortsTuple &ports ) {
        register_input_port( node, &(tbb::flow::get<N-1>(ports)), static_cast<tbb::internal::string_index>(FLOW_INPUT_PORT_0 + N - 1) );
        fgt_internal_input_helper<PortsTuple, N-1>::register_port( node, ports );
    }
};

template < typename PortsTuple >
struct fgt_internal_input_helper<PortsTuple, 1> {
    static void register_port( void *node, PortsTuple &ports ) {
        register_input_port( node, &(tbb::flow::get<0>(ports)), FLOW_INPUT_PORT_0 );
    }
};

template<typename OutputType>
void register_output_port(void* codeptr, void *node, tbb::flow::sender<OutputType>* port, string_index name_index) {
    // TODO: Make fgt_internal_create_output_port a function template?
    fgt_internal_create_output_port( codeptr, node, static_cast<void *>(port), name_index);
}

template < typename PortsTuple, int N >
struct fgt_internal_output_helper {
    static void register_port( void* codeptr, void *node, PortsTuple &ports ) {
        register_output_port( codeptr, node, &(tbb::flow::get<N-1>(ports)), static_cast<tbb::internal::string_index>(FLOW_OUTPUT_PORT_0 + N - 1) );
        fgt_internal_output_helper<PortsTuple, N-1>::register_port( codeptr, node, ports );
    }
};

template < typename PortsTuple >
struct fgt_internal_output_helper<PortsTuple,1> {
    static void register_port( void* codeptr, void *node, PortsTuple &ports ) {
        register_output_port( codeptr, node, &(tbb::flow::get<0>(ports)), FLOW_OUTPUT_PORT_0 );
    }
};

template< typename NodeType >
void fgt_multioutput_node_desc( const NodeType *node, const char *desc ) {
    void *addr =  (void *)( static_cast< tbb::flow::receiver< typename NodeType::input_type > * >(const_cast< NodeType *>(node)) );
    itt_metadata_str_add( ITT_DOMAIN_FLOW, addr, FLOW_NODE, FLOW_OBJECT_NAME, desc );
}

template< typename NodeType >
void fgt_multiinput_multioutput_node_desc( const NodeType *node, const char *desc ) {
    void *addr =  const_cast<NodeType *>(node);
    itt_metadata_str_add( ITT_DOMAIN_FLOW, addr, FLOW_NODE, FLOW_OBJECT_NAME, desc );
}

template< typename NodeType >
static inline void fgt_node_desc( const NodeType *node, const char *desc ) {
    void *addr =  (void *)( static_cast< tbb::flow::sender< typename NodeType::output_type > * >(const_cast< NodeType *>(node)) );
    itt_metadata_str_add( ITT_DOMAIN_FLOW, addr, FLOW_NODE, FLOW_OBJECT_NAME, desc );
}

static inline void fgt_graph_desc( void *g, const char *desc ) {
    itt_metadata_str_add( ITT_DOMAIN_FLOW, g, FLOW_GRAPH, FLOW_OBJECT_NAME, desc );
}

static inline void fgt_body( void *node, void *body ) {
    itt_relation_add( ITT_DOMAIN_FLOW, body, FLOW_BODY, __itt_relation_is_child_of, node, FLOW_NODE );
}

template< int N, typename PortsTuple >
static inline void fgt_multioutput_node(void* codeptr, string_index t, void *g, void *input_port, PortsTuple &ports ) {
    itt_make_task_group( ITT_DOMAIN_FLOW, input_port, FLOW_NODE, g, FLOW_GRAPH, t );
    fgt_internal_create_input_port( input_port, input_port, FLOW_INPUT_PORT_0 );
    fgt_internal_output_helper<PortsTuple, N>::register_port(codeptr, input_port, ports );
}

template< int N, typename PortsTuple >
static inline void fgt_multioutput_node_with_body( void* codeptr, string_index t, void *g, void *input_port, PortsTuple &ports, void *body ) {
    itt_make_task_group( ITT_DOMAIN_FLOW, input_port, FLOW_NODE, g, FLOW_GRAPH, t );
    fgt_internal_create_input_port( input_port, input_port, FLOW_INPUT_PORT_0 );
    fgt_internal_output_helper<PortsTuple, N>::register_port( codeptr, input_port, ports );
    fgt_body( input_port, body );
}

template< int N, typename PortsTuple >
static inline void fgt_multiinput_node( void* codeptr, string_index t, void *g, PortsTuple &ports, void *output_port) {
    itt_make_task_group( ITT_DOMAIN_FLOW, output_port, FLOW_NODE, g, FLOW_GRAPH, t );
    fgt_internal_create_output_port( codeptr, output_port, output_port, FLOW_OUTPUT_PORT_0 );
    fgt_internal_input_helper<PortsTuple, N>::register_port( output_port, ports );
}

static inline void fgt_multiinput_multioutput_node( void* codeptr, string_index t, void *n, void *g ) {
    itt_make_task_group( ITT_DOMAIN_FLOW, n, FLOW_NODE, g, FLOW_GRAPH, t );
    suppress_unused_warning( codeptr );
#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    if (codeptr != NULL) {
        register_node_addr(ITT_DOMAIN_FLOW, n, FLOW_NODE, CODE_ADDRESS, &codeptr);
    }
#endif
}

static inline void fgt_node( void* codeptr, string_index t, void *g, void *output_port ) {
    itt_make_task_group( ITT_DOMAIN_FLOW, output_port, FLOW_NODE, g, FLOW_GRAPH, t );
    fgt_internal_create_output_port( codeptr, output_port, output_port, FLOW_OUTPUT_PORT_0 );
}

static void fgt_node_with_body( void* codeptr, string_index t, void *g, void *output_port, void *body ) {
    itt_make_task_group( ITT_DOMAIN_FLOW, output_port, FLOW_NODE, g, FLOW_GRAPH, t );
    fgt_internal_create_output_port(codeptr, output_port, output_port, FLOW_OUTPUT_PORT_0 );
    fgt_body( output_port, body );
}

static inline void fgt_node( void* codeptr, string_index t, void *g, void *input_port, void *output_port ) {
    fgt_node( codeptr, t, g, output_port );
    fgt_internal_create_input_port( output_port, input_port, FLOW_INPUT_PORT_0 );
}

static inline void  fgt_node_with_body( void* codeptr, string_index t, void *g, void *input_port, void *output_port, void *body ) {
    fgt_node_with_body( codeptr, t, g, output_port, body );
    fgt_internal_create_input_port( output_port, input_port, FLOW_INPUT_PORT_0 );
}


static inline void  fgt_node( void* codeptr, string_index t, void *g, void *input_port, void *decrement_port, void *output_port ) {
    fgt_node( codeptr, t, g, input_port, output_port );
    fgt_internal_create_input_port( output_port, decrement_port, FLOW_INPUT_PORT_1 );
}

static inline void fgt_make_edge( void *output_port, void *input_port ) {
    itt_relation_add( ITT_DOMAIN_FLOW, output_port, FLOW_OUTPUT_PORT, __itt_relation_is_predecessor_to, input_port, FLOW_INPUT_PORT);
}

static inline void fgt_remove_edge( void *output_port, void *input_port ) {
    itt_relation_add( ITT_DOMAIN_FLOW, output_port, FLOW_OUTPUT_PORT, __itt_relation_is_sibling_of, input_port, FLOW_INPUT_PORT);
}

static inline void fgt_graph( void *g ) {
    itt_make_task_group( ITT_DOMAIN_FLOW, g, FLOW_GRAPH, NULL, FLOW_NULL, FLOW_GRAPH );
}

static inline void fgt_begin_body( void *body ) {
    itt_task_begin( ITT_DOMAIN_FLOW, body, FLOW_BODY, NULL, FLOW_NULL, FLOW_BODY );
}

static inline void fgt_end_body( void * ) {
    itt_task_end( ITT_DOMAIN_FLOW );
}

static inline void fgt_async_try_put_begin( void *node, void *port ) {
    itt_task_begin( ITT_DOMAIN_FLOW, port, FLOW_OUTPUT_PORT, node, FLOW_NODE, FLOW_OUTPUT_PORT );
}

static inline void fgt_async_try_put_end( void *, void * ) {
    itt_task_end( ITT_DOMAIN_FLOW );
}

static inline void fgt_async_reserve( void *node, void *graph ) {
    itt_region_begin( ITT_DOMAIN_FLOW, node, FLOW_NODE, graph, FLOW_GRAPH, FLOW_NULL );
}

static inline void fgt_async_commit( void *node, void * /*graph*/) {
    itt_region_end( ITT_DOMAIN_FLOW, node, FLOW_NODE );
}

static inline void fgt_reserve_wait( void *graph ) {
    itt_region_begin( ITT_DOMAIN_FLOW, graph, FLOW_GRAPH, NULL, FLOW_NULL, FLOW_NULL );
}

static inline void fgt_release_wait( void *graph ) {
    itt_region_end( ITT_DOMAIN_FLOW, graph, FLOW_GRAPH );
}

#else // TBB_USE_THREADING_TOOLS

#define CODEPTR() NULL

static inline void fgt_alias_port(void * /*node*/, void * /*p*/, bool /*visible*/ ) { }

static inline void fgt_composite ( void* /*codeptr*/, void * /*node*/, void * /*graph*/ ) { }

static inline void fgt_graph( void * /*g*/ ) { }

template< typename NodeType >
static inline void fgt_multioutput_node_desc( const NodeType * /*node*/, const char * /*desc*/ ) { }

template< typename NodeType >
static inline void fgt_node_desc( const NodeType * /*node*/, const char * /*desc*/ ) { }

static inline void fgt_graph_desc( void * /*g*/, const char * /*desc*/ ) { }

static inline void fgt_body( void * /*node*/, void * /*body*/ ) { }

template< int N, typename PortsTuple >
static inline void fgt_multioutput_node( void* /*codeptr*/, string_index /*t*/, void * /*g*/, void * /*input_port*/, PortsTuple & /*ports*/ ) { }

template< int N, typename PortsTuple >
static inline void fgt_multioutput_node_with_body( void* /*codeptr*/, string_index /*t*/, void * /*g*/, void * /*input_port*/, PortsTuple & /*ports*/, void * /*body*/ ) { }

template< int N, typename PortsTuple >
static inline void fgt_multiinput_node( void* /*codeptr*/, string_index /*t*/, void * /*g*/, PortsTuple & /*ports*/, void * /*output_port*/ ) { }

static inline void fgt_multiinput_multioutput_node( void* /*codeptr*/, string_index /*t*/, void * /*node*/, void * /*graph*/ ) { }

static inline void fgt_node( void* /*codeptr*/, string_index /*t*/, void * /*g*/, void * /*output_port*/ ) { }
static inline void fgt_node( void* /*codeptr*/, string_index /*t*/, void * /*g*/, void * /*input_port*/, void * /*output_port*/ ) { }
static inline void  fgt_node( void* /*codeptr*/, string_index /*t*/, void * /*g*/, void * /*input_port*/, void * /*decrement_port*/, void * /*output_port*/ ) { }

static inline void fgt_node_with_body( void* /*codeptr*/, string_index /*t*/, void * /*g*/, void * /*output_port*/, void * /*body*/ ) { }
static inline void fgt_node_with_body( void* /*codeptr*/, string_index /*t*/, void * /*g*/, void * /*input_port*/, void * /*output_port*/, void * /*body*/ ) { }

static inline void fgt_make_edge( void * /*output_port*/, void * /*input_port*/ ) { }
static inline void fgt_remove_edge( void * /*output_port*/, void * /*input_port*/ ) { }

static inline void fgt_begin_body( void * /*body*/ ) { }
static inline void fgt_end_body( void *  /*body*/) { }

static inline void fgt_async_try_put_begin( void * /*node*/, void * /*port*/ ) { }
static inline void fgt_async_try_put_end( void * /*node*/ , void * /*port*/ ) { }
static inline void fgt_async_reserve( void * /*node*/, void * /*graph*/ ) { }
static inline void fgt_async_commit( void * /*node*/, void * /*graph*/ ) { }
static inline void fgt_reserve_wait( void * /*graph*/ ) { }
static inline void fgt_release_wait( void * /*graph*/ ) { }

template< typename NodeType >
void fgt_multiinput_multioutput_node_desc( const NodeType * /*node*/, const char * /*desc*/ ) { }

template < typename PortsTuple, int N >
struct fgt_internal_input_alias_helper {
    static void alias_port( void * /*node*/, PortsTuple & /*ports*/ ) { }
};

template < typename PortsTuple, int N >
struct fgt_internal_output_alias_helper {
    static void alias_port( void * /*node*/, PortsTuple & /*ports*/ ) { }
};

#endif // TBB_USE_THREADING_TOOLS

    } // namespace internal
} // namespace tbb

#endif
