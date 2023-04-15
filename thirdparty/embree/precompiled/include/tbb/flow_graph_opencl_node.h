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

#include "internal/_deprecated_header_message_guard.h"

#if !defined(__TBB_show_deprecation_message_flow_graph_opencl_node_H) && defined(__TBB_show_deprecated_header_message)
#define  __TBB_show_deprecation_message_flow_graph_opencl_node_H
#pragma message("TBB Warning: tbb/flow_graph_opencl_node.h is deprecated. For details, please see Deprecated Features appendix in the TBB reference manual.")
#endif

#if defined(__TBB_show_deprecated_header_message)
#undef __TBB_show_deprecated_header_message
#endif

#ifndef __TBB_flow_graph_opencl_node_H
#define __TBB_flow_graph_opencl_node_H

#define __TBB_flow_graph_opencl_node_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#include "tbb/tbb_config.h"
#if __TBB_PREVIEW_OPENCL_NODE

#include "flow_graph.h"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <map>
#include <mutex>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace tbb {
namespace flow {

namespace interface11 {

template <typename DeviceFilter>
class opencl_factory;

namespace opencl_info {
class default_opencl_factory;
}

template <typename Factory>
class opencl_program;

inline void enforce_cl_retcode(cl_int err, std::string msg) {
    if (err != CL_SUCCESS) {
        std::cerr << msg << "; error code: " << err << std::endl;
        throw msg;
    }
}

template <typename T>
T event_info(cl_event e, cl_event_info i) {
    T res;
    enforce_cl_retcode(clGetEventInfo(e, i, sizeof(res), &res, NULL), "Failed to get OpenCL event information");
    return res;
}

template <typename T>
T device_info(cl_device_id d, cl_device_info i) {
    T res;
    enforce_cl_retcode(clGetDeviceInfo(d, i, sizeof(res), &res, NULL), "Failed to get OpenCL device information");
    return res;
}

template <>
inline std::string device_info<std::string>(cl_device_id d, cl_device_info i) {
    size_t required;
    enforce_cl_retcode(clGetDeviceInfo(d, i, 0, NULL, &required), "Failed to get OpenCL device information");

    char *buff = (char*)alloca(required);
    enforce_cl_retcode(clGetDeviceInfo(d, i, required, buff, NULL), "Failed to get OpenCL device information");

    return buff;
}

template <typename T>
T platform_info(cl_platform_id p, cl_platform_info i) {
    T res;
    enforce_cl_retcode(clGetPlatformInfo(p, i, sizeof(res), &res, NULL), "Failed to get OpenCL platform information");
    return res;
}

template <>
inline std::string platform_info<std::string>(cl_platform_id p, cl_platform_info  i) {
    size_t required;
    enforce_cl_retcode(clGetPlatformInfo(p, i, 0, NULL, &required), "Failed to get OpenCL platform information");

    char *buff = (char*)alloca(required);
    enforce_cl_retcode(clGetPlatformInfo(p, i, required, buff, NULL), "Failed to get OpenCL platform information");

    return buff;
}


class __TBB_DEPRECATED_IN_VERBOSE_MODE opencl_device {
public:
    typedef size_t device_id_type;
    enum : device_id_type {
        unknown = device_id_type( -2 ),
        host = device_id_type( -1 )
    };

    opencl_device() : my_device_id( unknown ), my_cl_device_id( NULL ), my_cl_command_queue( NULL ) {}

    opencl_device( cl_device_id d_id ) : my_device_id( unknown ), my_cl_device_id( d_id ), my_cl_command_queue( NULL ) {}

    opencl_device( cl_device_id cl_d_id, device_id_type device_id ) : my_device_id( device_id ), my_cl_device_id( cl_d_id ), my_cl_command_queue( NULL ) {}

    std::string platform_profile() const {
        return platform_info<std::string>( platform_id(), CL_PLATFORM_PROFILE );
    }
    std::string platform_version() const {
        return platform_info<std::string>( platform_id(), CL_PLATFORM_VERSION );
    }
    std::string platform_name() const {
        return platform_info<std::string>( platform_id(), CL_PLATFORM_NAME );
    }
    std::string platform_vendor() const {
        return platform_info<std::string>( platform_id(), CL_PLATFORM_VENDOR );
    }
    std::string platform_extensions() const {
        return platform_info<std::string>( platform_id(), CL_PLATFORM_EXTENSIONS );
    }

    template <typename T>
    void info( cl_device_info i, T &t ) const {
        t = device_info<T>( my_cl_device_id, i );
    }
    std::string version() const {
        // The version string format: OpenCL<space><major_version.minor_version><space><vendor-specific information>
        return device_info<std::string>( my_cl_device_id, CL_DEVICE_VERSION );
    }
    int major_version() const {
        int major;
        std::sscanf( version().c_str(), "OpenCL %d", &major );
        return major;
    }
    int minor_version() const {
        int major, minor;
        std::sscanf( version().c_str(), "OpenCL %d.%d", &major, &minor );
        return minor;
    }
    bool out_of_order_exec_mode_on_host_present() const {
#if CL_VERSION_2_0
        if ( major_version() >= 2 )
            return (device_info<cl_command_queue_properties>( my_cl_device_id, CL_DEVICE_QUEUE_ON_HOST_PROPERTIES ) & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0;
        else
#endif /* CL_VERSION_2_0 */
            return (device_info<cl_command_queue_properties>( my_cl_device_id, CL_DEVICE_QUEUE_PROPERTIES ) & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0;
    }
    bool out_of_order_exec_mode_on_device_present() const {
#if CL_VERSION_2_0
        if ( major_version() >= 2 )
            return (device_info<cl_command_queue_properties>( my_cl_device_id, CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES ) & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0;
        else
#endif /* CL_VERSION_2_0 */
            return false;
    }
    std::array<size_t, 3> max_work_item_sizes() const {
        return device_info<std::array<size_t, 3>>( my_cl_device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES );
    }
    size_t max_work_group_size() const {
        return device_info<size_t>( my_cl_device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE );
    }
    bool built_in_kernel_available( const std::string& k ) const {
        const std::string semi = ";";
        // Added semicolumns to force an exact match (to avoid a partial match, e.g. "add" is partly matched with "madd").
        return (semi + built_in_kernels() + semi).find( semi + k + semi ) != std::string::npos;
    }
    std::string built_in_kernels() const {
        return device_info<std::string>( my_cl_device_id, CL_DEVICE_BUILT_IN_KERNELS );
    }
    std::string name() const {
        return device_info<std::string>( my_cl_device_id, CL_DEVICE_NAME );
    }
    cl_bool available() const {
        return device_info<cl_bool>( my_cl_device_id, CL_DEVICE_AVAILABLE );
    }
    cl_bool compiler_available() const {
        return device_info<cl_bool>( my_cl_device_id, CL_DEVICE_COMPILER_AVAILABLE );
    }
    cl_bool linker_available() const {
        return device_info<cl_bool>( my_cl_device_id, CL_DEVICE_LINKER_AVAILABLE );
    }
    bool extension_available( const std::string &ext ) const {
        const std::string space = " ";
        // Added space to force an exact match (to avoid a partial match, e.g. "ext" is partly matched with "ext2").
        return (space + extensions() + space).find( space + ext + space ) != std::string::npos;
    }
    std::string extensions() const {
        return device_info<std::string>( my_cl_device_id, CL_DEVICE_EXTENSIONS );
    }

    cl_device_type type() const {
        return device_info<cl_device_type>( my_cl_device_id, CL_DEVICE_TYPE );
    }

    std::string vendor() const {
        return device_info<std::string>( my_cl_device_id, CL_DEVICE_VENDOR );
    }

    cl_uint address_bits() const {
        return device_info<cl_uint>( my_cl_device_id, CL_DEVICE_ADDRESS_BITS );
    }

    cl_device_id device_id() const {
        return my_cl_device_id;
    }

    cl_command_queue command_queue() const {
        return my_cl_command_queue;
    }

    void set_command_queue( cl_command_queue cmd_queue ) {
        my_cl_command_queue = cmd_queue;
    }

    cl_platform_id platform_id() const {
        return device_info<cl_platform_id>( my_cl_device_id, CL_DEVICE_PLATFORM );
    }

private:

    device_id_type my_device_id;
    cl_device_id my_cl_device_id;
    cl_command_queue my_cl_command_queue;

    friend bool operator==(opencl_device d1, opencl_device d2) { return d1.my_cl_device_id == d2.my_cl_device_id; }

    template <typename DeviceFilter>
    friend class opencl_factory;
    template <typename Factory>
    friend class opencl_memory;
    template <typename Factory>
    friend class opencl_program;

#if TBB_USE_ASSERT
    template <typename T, typename Factory>
    friend class opencl_buffer;
#endif
};

class __TBB_DEPRECATED_IN_VERBOSE_MODE opencl_device_list {
    typedef std::vector<opencl_device> container_type;
public:
    typedef container_type::iterator iterator;
    typedef container_type::const_iterator const_iterator;
    typedef container_type::size_type size_type;

    opencl_device_list() {}
    opencl_device_list( std::initializer_list<opencl_device> il ) : my_container( il ) {}

    void add( opencl_device d ) { my_container.push_back( d ); }
    size_type size() const { return my_container.size(); }
    bool empty() const { return my_container.empty(); }
    iterator begin() { return my_container.begin(); }
    iterator end() { return my_container.end(); }
    const_iterator begin() const { return my_container.begin(); }
    const_iterator end() const { return my_container.end(); }
    const_iterator cbegin() const { return my_container.cbegin(); }
    const_iterator cend() const { return my_container.cend(); }

private:
    container_type my_container;
};

namespace internal {

// Retrieve all OpenCL devices from machine
inline opencl_device_list find_available_devices() {
    opencl_device_list opencl_devices;

    cl_uint num_platforms;
    enforce_cl_retcode(clGetPlatformIDs(0, NULL, &num_platforms), "clGetPlatformIDs failed");

    std::vector<cl_platform_id> platforms(num_platforms);
    enforce_cl_retcode(clGetPlatformIDs(num_platforms, platforms.data(), NULL), "clGetPlatformIDs failed");

    cl_uint num_devices;
    std::vector<cl_platform_id>::iterator platforms_it = platforms.begin();
    cl_uint num_all_devices = 0;
    while (platforms_it != platforms.end()) {
        cl_int err = clGetDeviceIDs(*platforms_it, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        if (err == CL_DEVICE_NOT_FOUND) {
            platforms_it = platforms.erase(platforms_it);
        }
        else {
            enforce_cl_retcode(err, "clGetDeviceIDs failed");
            num_all_devices += num_devices;
            ++platforms_it;
        }
    }

    std::vector<cl_device_id> devices(num_all_devices);
    std::vector<cl_device_id>::iterator devices_it = devices.begin();
    for (auto p = platforms.begin(); p != platforms.end(); ++p) {
        enforce_cl_retcode(clGetDeviceIDs((*p), CL_DEVICE_TYPE_ALL, (cl_uint)std::distance(devices_it, devices.end()), &*devices_it, &num_devices), "clGetDeviceIDs failed");
        devices_it += num_devices;
    }

    for (auto d = devices.begin(); d != devices.end(); ++d) {
        opencl_devices.add(opencl_device((*d)));
    }

    return opencl_devices;
}

} // namespace internal

// TODO: consider this namespace as public API
namespace opencl_info {

    inline const opencl_device_list& available_devices() {
        // Static storage for all available OpenCL devices on machine
        static const opencl_device_list my_devices = internal::find_available_devices();
        return my_devices;
    }

} // namespace opencl_info


class callback_base : tbb::internal::no_copy {
public:
    virtual void call() = 0;
    virtual ~callback_base() {}
};

template <typename Callback, typename T>
class callback : public callback_base {
    Callback my_callback;
    T my_data;
public:
    callback( Callback c, const T& t ) : my_callback( c ), my_data( t ) {}

    void call() __TBB_override {
        my_callback( my_data );
    }
};

template <typename T, typename Factory = opencl_info::default_opencl_factory>
class __TBB_DEPRECATED_IN_VERBOSE_MODE opencl_async_msg : public async_msg<T> {
public:
    typedef T value_type;

    opencl_async_msg() : my_callback_flag_ptr( std::make_shared< tbb::atomic<bool>>() ) {
        my_callback_flag_ptr->store<tbb::relaxed>(false);
    }

    explicit opencl_async_msg( const T& data ) : my_data(data), my_callback_flag_ptr( std::make_shared<tbb::atomic<bool>>() ) {
        my_callback_flag_ptr->store<tbb::relaxed>(false);
    }

    opencl_async_msg( const T& data, cl_event event ) : my_data(data), my_event(event), my_is_event(true), my_callback_flag_ptr( std::make_shared<tbb::atomic<bool>>() ) {
        my_callback_flag_ptr->store<tbb::relaxed>(false);
        enforce_cl_retcode( clRetainEvent( my_event ), "Failed to retain an event" );
    }

    T& data( bool wait = true ) {
        if ( my_is_event && wait ) {
            enforce_cl_retcode( clWaitForEvents( 1, &my_event ), "Failed to wait for an event" );
            enforce_cl_retcode( clReleaseEvent( my_event ), "Failed to release an event" );
            my_is_event = false;
        }
        return my_data;
    }

    const T& data( bool wait = true ) const {
        if ( my_is_event && wait ) {
            enforce_cl_retcode( clWaitForEvents( 1, &my_event ), "Failed to wait for an event" );
            enforce_cl_retcode( clReleaseEvent( my_event ), "Failed to release an event" );
            my_is_event = false;
        }
        return my_data;
    }

    opencl_async_msg( const opencl_async_msg &dmsg ) : async_msg<T>(dmsg),
        my_data(dmsg.my_data), my_event(dmsg.my_event), my_is_event( dmsg.my_is_event ),
        my_callback_flag_ptr(dmsg.my_callback_flag_ptr)
    {
        if ( my_is_event )
            enforce_cl_retcode( clRetainEvent( my_event ), "Failed to retain an event" );
    }

    opencl_async_msg( opencl_async_msg &&dmsg ) : async_msg<T>(std::move(dmsg)),
        my_data(std::move(dmsg.my_data)), my_event(dmsg.my_event), my_is_event(dmsg.my_is_event),
        my_callback_flag_ptr( std::move(dmsg.my_callback_flag_ptr) )
    {
        dmsg.my_is_event = false;
    }

    opencl_async_msg& operator=(const opencl_async_msg &dmsg) {
        async_msg<T>::operator =(dmsg);

        // Release original event
        if ( my_is_event )
            enforce_cl_retcode( clReleaseEvent( my_event ), "Failed to retain an event" );

        my_data = dmsg.my_data;
        my_event = dmsg.my_event;
        my_is_event = dmsg.my_is_event;

        // Retain copied event
        if ( my_is_event )
            enforce_cl_retcode( clRetainEvent( my_event ), "Failed to retain an event" );

        my_callback_flag_ptr = dmsg.my_callback_flag_ptr;
        return *this;
    }

    ~opencl_async_msg() {
        if ( my_is_event )
            enforce_cl_retcode( clReleaseEvent( my_event ), "Failed to release an event" );
    }

    cl_event const * get_event() const { return my_is_event ? &my_event : NULL; }
    void set_event( cl_event e ) const {
        if ( my_is_event ) {
            cl_command_queue cq = event_info<cl_command_queue>( my_event, CL_EVENT_COMMAND_QUEUE );
            if ( cq != event_info<cl_command_queue>( e, CL_EVENT_COMMAND_QUEUE ) )
                enforce_cl_retcode( clFlush( cq ), "Failed to flush an OpenCL command queue" );
            enforce_cl_retcode( clReleaseEvent( my_event ), "Failed to release an event" );
        }
        my_is_event = true;
        my_event = e;
        clRetainEvent( my_event );
    }

    void clear_event() const {
        if ( my_is_event ) {
            enforce_cl_retcode( clFlush( event_info<cl_command_queue>( my_event, CL_EVENT_COMMAND_QUEUE ) ), "Failed to flush an OpenCL command queue" );
            enforce_cl_retcode( clReleaseEvent( my_event ), "Failed to release an event" );
        }
        my_is_event = false;
    }

    template <typename Callback>
    void register_callback( Callback c ) const {
        __TBB_ASSERT( my_is_event, "The OpenCL event is not set" );
        enforce_cl_retcode( clSetEventCallback( my_event, CL_COMPLETE, register_callback_func, new callback<Callback, T>( c, my_data ) ), "Failed to set an OpenCL callback" );
    }

    operator T&() { return data(); }
    operator const T&() const { return data(); }

protected:
    // Overridden in this derived class to inform that
    // async calculation chain is over
    void finalize() const __TBB_override {
        receive_if_memory_object(*this);
        if (! my_callback_flag_ptr->fetch_and_store(true)) {
            opencl_async_msg a(*this);
            if (my_is_event) {
                register_callback([a](const T& t) mutable {
                    a.set(t);
                });
            }
            else {
                a.set(my_data);
            }
        }
        clear_event();
    }

private:
    static void CL_CALLBACK register_callback_func( cl_event, cl_int event_command_exec_status, void *data ) {
        tbb::internal::suppress_unused_warning( event_command_exec_status );
        __TBB_ASSERT( event_command_exec_status == CL_COMPLETE, NULL );
        __TBB_ASSERT( data, NULL );
        callback_base *c = static_cast<callback_base*>(data);
        c->call();
        delete c;
    }

    T my_data;
    mutable cl_event my_event;
    mutable bool my_is_event = false;

    std::shared_ptr< tbb::atomic<bool> > my_callback_flag_ptr;
};

template <typename K, typename T, typename Factory>
K key_from_message( const opencl_async_msg<T, Factory> &dmsg ) {
    using tbb::flow::key_from_message;
    const T &t = dmsg.data( false );
    __TBB_STATIC_ASSERT( true, "" );
    return key_from_message<K, T>( t );
}

template <typename Factory>
class opencl_memory {
public:
    opencl_memory() {}
    opencl_memory( Factory &f ) : my_host_ptr( NULL ), my_factory( &f ), my_sending_event_present( false ) {
        my_curr_device_id = my_factory->devices().begin()->my_device_id;
    }

    virtual ~opencl_memory() {
        if ( my_sending_event_present ) enforce_cl_retcode( clReleaseEvent( my_sending_event ), "Failed to release an event for the OpenCL buffer" );
        enforce_cl_retcode( clReleaseMemObject( my_cl_mem ), "Failed to release an memory object" );
    }

    cl_mem get_cl_mem() const {
        return my_cl_mem;
    }

    void* get_host_ptr() {
        if ( !my_host_ptr ) {
            opencl_async_msg<void*, Factory> d = receive( NULL );
            d.data();
            __TBB_ASSERT( d.data() == my_host_ptr, NULL );
        }
        return my_host_ptr;
    }

    Factory *factory() const { return my_factory; }

    opencl_async_msg<void*, Factory> receive(const cl_event *e) {
        opencl_async_msg<void*, Factory> d;
        if (e) {
            d = opencl_async_msg<void*, Factory>(my_host_ptr, *e);
        } else {
            d = opencl_async_msg<void*, Factory>(my_host_ptr);
        }

        // Concurrent receives are prohibited so we do not worry about synchronization.
        if (my_curr_device_id.load<tbb::relaxed>() != opencl_device::host) {
            map_memory(*my_factory->devices().begin(), d);
            my_curr_device_id.store<tbb::relaxed>(opencl_device::host);
            my_host_ptr = d.data(false);
        }
        // Release the sending event
        if (my_sending_event_present) {
            enforce_cl_retcode(clReleaseEvent(my_sending_event), "Failed to release an event");
            my_sending_event_present = false;
        }
        return d;
    }

    opencl_async_msg<void*, Factory> send(opencl_device device, const cl_event *e) {
        opencl_device::device_id_type device_id = device.my_device_id;
        if (!my_factory->is_same_context(my_curr_device_id.load<tbb::acquire>(), device_id)) {
            {
                tbb::spin_mutex::scoped_lock lock(my_sending_lock);
                if (!my_factory->is_same_context(my_curr_device_id.load<tbb::relaxed>(), device_id)) {
                    __TBB_ASSERT(my_host_ptr, "The buffer has not been mapped");
                    opencl_async_msg<void*, Factory> d(my_host_ptr);
                    my_factory->enqueue_unmap_buffer(device, *this, d);
                    my_sending_event = *d.get_event();
                    my_sending_event_present = true;
                    enforce_cl_retcode(clRetainEvent(my_sending_event), "Failed to retain an event");
                    my_host_ptr = NULL;
                    my_curr_device_id.store<tbb::release>(device_id);
                }
            }
            __TBB_ASSERT(my_sending_event_present, NULL);
        }

        // !e means that buffer has come from the host
        if (!e && my_sending_event_present) e = &my_sending_event;

        __TBB_ASSERT(!my_host_ptr, "The buffer has not been unmapped");
        return e ? opencl_async_msg<void*, Factory>(NULL, *e) : opencl_async_msg<void*, Factory>(NULL);
    }

    virtual void map_memory( opencl_device, opencl_async_msg<void*, Factory> & ) = 0;
protected:
    cl_mem my_cl_mem;
    tbb::atomic<opencl_device::device_id_type> my_curr_device_id;
    void* my_host_ptr;
    Factory *my_factory;

    tbb::spin_mutex my_sending_lock;
    bool my_sending_event_present;
    cl_event my_sending_event;
};

template <typename Factory>
class opencl_buffer_impl : public opencl_memory<Factory> {
    size_t my_size;
public:
    opencl_buffer_impl( size_t size, Factory& f ) : opencl_memory<Factory>( f ), my_size( size ) {
        cl_int err;
        this->my_cl_mem = clCreateBuffer( this->my_factory->context(), CL_MEM_ALLOC_HOST_PTR, size, NULL, &err );
        enforce_cl_retcode( err, "Failed to create an OpenCL buffer" );
    }

    // The constructor for subbuffers.
    opencl_buffer_impl( cl_mem m, size_t index, size_t size, Factory& f ) : opencl_memory<Factory>( f ), my_size( size ) {
        cl_int err;
        cl_buffer_region region = { index, size };
        this->my_cl_mem = clCreateSubBuffer( m, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err );
        enforce_cl_retcode( err, "Failed to create an OpenCL subbuffer" );
    }

    size_t size() const {
        return my_size;
    }

    void map_memory( opencl_device device, opencl_async_msg<void*, Factory> &dmsg ) __TBB_override {
        this->my_factory->enqueue_map_buffer( device, *this, dmsg );
    }

#if TBB_USE_ASSERT
    template <typename, typename>
    friend class opencl_buffer;
#endif
};

enum access_type {
    read_write,
    write_only,
    read_only
};

template <typename T, typename Factory = opencl_info::default_opencl_factory>
class __TBB_DEPRECATED_IN_VERBOSE_MODE
opencl_subbuffer;

template <typename T, typename Factory = opencl_info::default_opencl_factory>
class __TBB_DEPRECATED_IN_VERBOSE_MODE
opencl_buffer {
public:
    typedef cl_mem native_object_type;
    typedef opencl_buffer memory_object_type;
    typedef Factory opencl_factory_type;

    template<access_type a> using iterator = T*;

    template <access_type a>
    iterator<a> access() const {
        T* ptr = (T*)my_impl->get_host_ptr();
        __TBB_ASSERT( ptr, NULL );
        return iterator<a>( ptr );
    }

    T* data() const { return &access<read_write>()[0]; }

    template <access_type a = read_write>
    iterator<a> begin() const { return access<a>(); }

    template <access_type a = read_write>
    iterator<a> end() const { return access<a>()+my_impl->size()/sizeof(T); }

    size_t size() const { return my_impl->size()/sizeof(T); }

    T& operator[] ( ptrdiff_t k ) { return begin()[k]; }

    opencl_buffer() {}
    opencl_buffer( size_t size );
    opencl_buffer( Factory &f, size_t size ) : my_impl( std::make_shared<impl_type>( size*sizeof(T), f ) ) {}

    cl_mem native_object() const {
        return my_impl->get_cl_mem();
    }

    const opencl_buffer& memory_object() const {
        return *this;
    }

    void send( opencl_device device, opencl_async_msg<opencl_buffer, Factory> &dependency ) const {
        __TBB_ASSERT( dependency.data( /*wait = */false ) == *this, NULL );
        opencl_async_msg<void*, Factory> d = my_impl->send( device, dependency.get_event() );
        const cl_event *e = d.get_event();
        if ( e ) dependency.set_event( *e );
        else dependency.clear_event();
    }
    void receive( const opencl_async_msg<opencl_buffer, Factory> &dependency ) const {
        __TBB_ASSERT( dependency.data( /*wait = */false ) == *this, NULL );
        opencl_async_msg<void*, Factory> d = my_impl->receive( dependency.get_event() );
        const cl_event *e = d.get_event();
        if ( e ) dependency.set_event( *e );
        else dependency.clear_event();
    }

    opencl_subbuffer<T, Factory> subbuffer( size_t index, size_t size ) const;
private:
    // The constructor for subbuffers.
    opencl_buffer( Factory &f, cl_mem m, size_t index, size_t size ) : my_impl( std::make_shared<impl_type>( m, index*sizeof(T), size*sizeof(T), f ) ) {}

    typedef opencl_buffer_impl<Factory> impl_type;

    std::shared_ptr<impl_type> my_impl;

    friend bool operator==(const opencl_buffer<T, Factory> &lhs, const opencl_buffer<T, Factory> &rhs) {
        return lhs.my_impl == rhs.my_impl;
    }

    template <typename>
    friend class opencl_factory;
    template <typename, typename>
    friend class opencl_subbuffer;
};

template <typename T, typename Factory>
class __TBB_DEPRECATED_IN_VERBOSE_MODE
opencl_subbuffer : public opencl_buffer<T, Factory> {
    opencl_buffer<T, Factory> my_owner;
public:
    opencl_subbuffer() {}
    opencl_subbuffer( const opencl_buffer<T, Factory> &owner, size_t index, size_t size ) :
        opencl_buffer<T, Factory>( *owner.my_impl->factory(), owner.native_object(), index, size ), my_owner( owner ) {}
};

template <typename T, typename Factory>
opencl_subbuffer<T, Factory> opencl_buffer<T, Factory>::subbuffer( size_t index, size_t size ) const {
    return opencl_subbuffer<T, Factory>( *this, index, size );
}


#define is_typedef(type)                                                    \
    template <typename T>                                                   \
    struct is_##type {                                                      \
        template <typename C>                                               \
        static std::true_type check( typename C::type* );                   \
        template <typename C>                                               \
        static std::false_type check( ... );                                \
                                                                            \
        static const bool value = decltype(check<T>(0))::value;             \
    }

is_typedef( native_object_type );
is_typedef( memory_object_type );

template <typename T>
typename std::enable_if<is_native_object_type<T>::value, typename T::native_object_type>::type get_native_object( const T &t ) {
    return t.native_object();
}

template <typename T>
typename std::enable_if<!is_native_object_type<T>::value, T>::type get_native_object( T t ) {
    return t;
}

// send_if_memory_object checks if the T type has memory_object_type and call the send method for the object.
template <typename T, typename Factory>
typename std::enable_if<is_memory_object_type<T>::value>::type send_if_memory_object( opencl_device device, opencl_async_msg<T, Factory> &dmsg ) {
    const T &t = dmsg.data( false );
    typedef typename T::memory_object_type mem_obj_t;
    mem_obj_t mem_obj = t.memory_object();
    opencl_async_msg<mem_obj_t, Factory> d( mem_obj );
    if ( dmsg.get_event() ) d.set_event( *dmsg.get_event() );
    mem_obj.send( device, d );
    if ( d.get_event() ) dmsg.set_event( *d.get_event() );
}

template <typename T>
typename std::enable_if<is_memory_object_type<T>::value>::type send_if_memory_object( opencl_device device, T &t ) {
    typedef typename T::memory_object_type mem_obj_t;
    mem_obj_t mem_obj = t.memory_object();
    opencl_async_msg<mem_obj_t, typename mem_obj_t::opencl_factory_type> dmsg( mem_obj );
    mem_obj.send( device, dmsg );
}

template <typename T>
typename std::enable_if<!is_memory_object_type<T>::value>::type send_if_memory_object( opencl_device, T& ) {};

// receive_if_memory_object checks if the T type has memory_object_type and call the receive method for the object.
template <typename T, typename Factory>
typename std::enable_if<is_memory_object_type<T>::value>::type receive_if_memory_object( const opencl_async_msg<T, Factory> &dmsg ) {
    const T &t = dmsg.data( false );
    typedef typename T::memory_object_type mem_obj_t;
    mem_obj_t mem_obj = t.memory_object();
    opencl_async_msg<mem_obj_t, Factory> d( mem_obj );
    if ( dmsg.get_event() ) d.set_event( *dmsg.get_event() );
    mem_obj.receive( d );
    if ( d.get_event() ) dmsg.set_event( *d.get_event() );
}

template <typename T>
typename std::enable_if<!is_memory_object_type<T>::value>::type  receive_if_memory_object( const T& ) {}

class __TBB_DEPRECATED_IN_VERBOSE_MODE opencl_range {
public:
    typedef size_t range_index_type;
    typedef std::array<range_index_type, 3> nd_range_type;

    template <typename G = std::initializer_list<int>, typename L = std::initializer_list<int>,
        typename = typename std::enable_if<!std::is_same<typename std::decay<G>::type, opencl_range>::value>::type>
    opencl_range(G&& global_work = std::initializer_list<int>({ 0 }), L&& local_work = std::initializer_list<int>({ 0, 0, 0 })) {
        auto g_it = global_work.begin();
        auto l_it = local_work.begin();
        my_global_work_size = { {size_t(-1), size_t(-1), size_t(-1)} };
        // my_local_work_size is still uninitialized
        for (int s = 0; s < 3 && g_it != global_work.end(); ++g_it, ++l_it, ++s) {
            __TBB_ASSERT(l_it != local_work.end(), "global_work & local_work must have same size");
            my_global_work_size[s] = *g_it;
            my_local_work_size[s] = *l_it;
        }
    }

    const nd_range_type& global_range() const { return my_global_work_size; }
    const nd_range_type& local_range() const { return my_local_work_size; }

private:
    nd_range_type my_global_work_size;
    nd_range_type my_local_work_size;
};

template <typename DeviceFilter>
class __TBB_DEPRECATED_IN_VERBOSE_MODE opencl_factory {
public:
    template<typename T> using async_msg_type = opencl_async_msg<T, opencl_factory<DeviceFilter>>;
    typedef opencl_device device_type;

    class kernel : tbb::internal::no_assign {
    public:
        kernel( const kernel& k ) : my_factory( k.my_factory ) {
            // Clone my_cl_kernel via opencl_program
            size_t ret_size = 0;

            std::vector<char> kernel_name;
            for ( size_t curr_size = 32;; curr_size <<= 1 ) {
                kernel_name.resize( curr_size <<= 1 );
                enforce_cl_retcode( clGetKernelInfo( k.my_cl_kernel, CL_KERNEL_FUNCTION_NAME, curr_size, kernel_name.data(), &ret_size ), "Failed to get kernel info" );
                if ( ret_size < curr_size ) break;
            }

            cl_program program;
            enforce_cl_retcode( clGetKernelInfo( k.my_cl_kernel, CL_KERNEL_PROGRAM, sizeof(program), &program, &ret_size ), "Failed to get kernel info" );
            __TBB_ASSERT( ret_size == sizeof(program), NULL );

            my_cl_kernel = opencl_program< factory_type >( my_factory, program ).get_cl_kernel( kernel_name.data() );
        }

        ~kernel() {
            enforce_cl_retcode( clReleaseKernel( my_cl_kernel ), "Failed to release a kernel" );
        }

    private:
        typedef opencl_factory<DeviceFilter> factory_type;

        kernel( const cl_kernel& k, factory_type& f ) : my_cl_kernel( k ), my_factory( f ) {}

        // Data
        cl_kernel my_cl_kernel;
        factory_type& my_factory;

        template <typename DeviceFilter_>
        friend class opencl_factory;

        template <typename Factory>
        friend class opencl_program;
    };

    typedef kernel kernel_type;

    // 'range_type' enables kernel_executor with range support
    // it affects expectations for enqueue_kernel(.....) interface method
    typedef opencl_range range_type;

    opencl_factory() {}
    ~opencl_factory() {
        if ( my_devices.size() ) {
            for ( auto d = my_devices.begin(); d != my_devices.end(); ++d ) {
                enforce_cl_retcode( clReleaseCommandQueue( (*d).my_cl_command_queue ), "Failed to release a command queue" );
            }
            enforce_cl_retcode( clReleaseContext( my_cl_context ), "Failed to release a context" );
        }
    }

    bool init( const opencl_device_list &device_list ) {
        tbb::spin_mutex::scoped_lock lock( my_devices_mutex );
        if ( !my_devices.size() ) {
            my_devices = device_list;
            return true;
        }
        return false;
    }


private:
    template <typename Factory>
    void enqueue_map_buffer( opencl_device device, opencl_buffer_impl<Factory> &buffer, opencl_async_msg<void*, Factory>& dmsg ) {
        cl_event const* e1 = dmsg.get_event();
        cl_event e2;
        cl_int err;
        void *ptr = clEnqueueMapBuffer( device.my_cl_command_queue, buffer.get_cl_mem(), false, CL_MAP_READ | CL_MAP_WRITE, 0, buffer.size(),
            e1 == NULL ? 0 : 1, e1, &e2, &err );
        enforce_cl_retcode( err, "Failed to map a buffer" );
        dmsg.data( false ) = ptr;
        dmsg.set_event( e2 );
        enforce_cl_retcode( clReleaseEvent( e2 ), "Failed to release an event" );
    }


    template <typename Factory>
    void enqueue_unmap_buffer( opencl_device device, opencl_memory<Factory> &memory, opencl_async_msg<void*, Factory>& dmsg ) {
        cl_event const* e1 = dmsg.get_event();
        cl_event e2;
        enforce_cl_retcode(
            clEnqueueUnmapMemObject( device.my_cl_command_queue, memory.get_cl_mem(), memory.get_host_ptr(), e1 == NULL ? 0 : 1, e1, &e2 ),
           "Failed to unmap a buffer" );
        dmsg.set_event( e2 );
        enforce_cl_retcode( clReleaseEvent( e2 ), "Failed to release an event" );
    }

    // --------- Kernel argument & event list helpers --------- //
    template <size_t NUM_ARGS, typename T>
    void process_one_arg( const kernel_type& kernel, std::array<cl_event, NUM_ARGS>&, int&, int& place, const T& t ) {
        auto p = get_native_object(t);
        enforce_cl_retcode( clSetKernelArg(kernel.my_cl_kernel, place++, sizeof(p), &p), "Failed to set a kernel argument" );
    }

    template <size_t NUM_ARGS, typename T, typename F>
    void process_one_arg( const kernel_type& kernel, std::array<cl_event, NUM_ARGS>& events, int& num_events, int& place, const opencl_async_msg<T, F>& msg ) {
        __TBB_ASSERT((static_cast<typename std::array<cl_event, NUM_ARGS>::size_type>(num_events) < events.size()), NULL);

        const cl_event * const e = msg.get_event();
        if (e != NULL) {
            events[num_events++] = *e;
        }

        process_one_arg( kernel, events, num_events, place, msg.data(false) );
    }

    template <size_t NUM_ARGS, typename T, typename ...Rest>
    void process_arg_list( const kernel_type& kernel, std::array<cl_event, NUM_ARGS>& events, int& num_events, int& place, const T& t, const Rest&... args ) {
        process_one_arg( kernel, events, num_events, place, t );
        process_arg_list( kernel, events, num_events, place, args... );
    }

    template <size_t NUM_ARGS>
    void process_arg_list( const kernel_type&, std::array<cl_event, NUM_ARGS>&, int&, int& ) {}
    // ------------------------------------------- //
    template <typename T>
    void update_one_arg( cl_event, T& ) {}

    template <typename T, typename F>
    void update_one_arg( cl_event e, opencl_async_msg<T, F>& msg ) {
        msg.set_event( e );
    }

    template <typename T, typename ...Rest>
    void update_arg_list( cl_event e, T& t, Rest&... args ) {
        update_one_arg( e, t );
        update_arg_list( e, args... );
    }

    void update_arg_list( cl_event ) {}
    // ------------------------------------------- //
public:
    template <typename ...Args>
    void send_kernel( opencl_device device, const kernel_type& kernel, const range_type& work_size, Args&... args ) {
        std::array<cl_event, sizeof...(Args)> events;
        int num_events = 0;
        int place = 0;
        process_arg_list( kernel, events, num_events, place, args... );

        const cl_event e = send_kernel_impl( device, kernel.my_cl_kernel, work_size, num_events, events.data() );

        update_arg_list(e, args...);

        // Release our own reference to cl_event
        enforce_cl_retcode( clReleaseEvent(e), "Failed to release an event" );
    }

    // ------------------------------------------- //
    template <typename T, typename ...Rest>
    void send_data(opencl_device device, T& t, Rest&... args) {
        send_if_memory_object( device, t );
        send_data( device, args... );
    }

    void send_data(opencl_device) {}
    // ------------------------------------------- //

private:
    cl_event send_kernel_impl( opencl_device device, const cl_kernel& kernel,
        const range_type& work_size, cl_uint num_events, cl_event* event_list ) {
        const typename range_type::nd_range_type g_offset = { { 0, 0, 0 } };
        const typename range_type::nd_range_type& g_size = work_size.global_range();
        const typename range_type::nd_range_type& l_size = work_size.local_range();
        cl_uint s;
        for ( s = 1; s < 3 && g_size[s] != size_t(-1); ++s) {}
        cl_event event;
        enforce_cl_retcode(
            clEnqueueNDRangeKernel( device.my_cl_command_queue, kernel, s,
                g_offset.data(), g_size.data(), l_size[0] ? l_size.data() : NULL, num_events, num_events ? event_list : NULL, &event ),
            "Failed to enqueue a kernel" );
        return event;
    }

    // ------------------------------------------- //
    template <typename T>
    bool get_event_from_one_arg( cl_event&, const T& ) {
        return false;
    }

    template <typename T, typename F>
    bool get_event_from_one_arg( cl_event& e, const opencl_async_msg<T, F>& msg) {
        cl_event const *e_ptr = msg.get_event();

        if ( e_ptr != NULL ) {
            e = *e_ptr;
            return true;
        }

        return false;
    }

    template <typename T, typename ...Rest>
    bool get_event_from_args( cl_event& e, const T& t, const Rest&... args ) {
        if ( get_event_from_one_arg( e, t ) ) {
            return true;
        }

        return get_event_from_args( e, args... );
    }

    bool get_event_from_args( cl_event& ) {
        return false;
    }
    // ------------------------------------------- //

    struct finalize_fn : tbb::internal::no_assign {
        virtual ~finalize_fn() {}
        virtual void operator() () {}
    };

    template<typename Fn>
    struct finalize_fn_leaf : public finalize_fn {
        Fn my_fn;
        finalize_fn_leaf(Fn fn) : my_fn(fn) {}
        void operator() () __TBB_override { my_fn(); }
    };

    static void CL_CALLBACK finalize_callback(cl_event, cl_int event_command_exec_status, void *data) {
        tbb::internal::suppress_unused_warning(event_command_exec_status);
        __TBB_ASSERT(event_command_exec_status == CL_COMPLETE, NULL);

        finalize_fn * const fn_ptr = static_cast<finalize_fn*>(data);
        __TBB_ASSERT(fn_ptr != NULL, "Invalid finalize function pointer");
        (*fn_ptr)();

        // Function pointer was created by 'new' & this callback must be called once only
        delete fn_ptr;
    }
public:
    template <typename FinalizeFn, typename ...Args>
    void finalize( opencl_device device, FinalizeFn fn, Args&... args ) {
        cl_event e;

        if ( get_event_from_args( e, args... ) ) {
            enforce_cl_retcode( clSetEventCallback( e, CL_COMPLETE, finalize_callback,
                new finalize_fn_leaf<FinalizeFn>(fn) ), "Failed to set a callback" );
        }

        enforce_cl_retcode( clFlush( device.my_cl_command_queue ), "Failed to flush an OpenCL command queue" );
    }

    const opencl_device_list& devices() {
        std::call_once( my_once_flag, &opencl_factory::init_once, this );
        return my_devices;
    }

private:
    bool is_same_context( opencl_device::device_id_type d1, opencl_device::device_id_type d2 ) {
        __TBB_ASSERT( d1 != opencl_device::unknown && d2 != opencl_device::unknown, NULL );
        // Currently, factory supports only one context so if the both devices are not host it means the are in the same context.
        if ( d1 != opencl_device::host && d2 != opencl_device::host )
            return true;
        return d1 == d2;
    }
private:
    opencl_factory( const opencl_factory& );
    opencl_factory& operator=(const opencl_factory&);

    cl_context context() {
        std::call_once( my_once_flag, &opencl_factory::init_once, this );
        return my_cl_context;
    }

    void init_once() {
        {
            tbb::spin_mutex::scoped_lock lock(my_devices_mutex);
            if (!my_devices.size())
                my_devices = DeviceFilter()( opencl_info::available_devices() );
        }

        enforce_cl_retcode(my_devices.size() ? CL_SUCCESS : CL_INVALID_DEVICE, "No devices in the device list");
        cl_platform_id platform_id = my_devices.begin()->platform_id();
        for (opencl_device_list::iterator it = ++my_devices.begin(); it != my_devices.end(); ++it)
            enforce_cl_retcode(it->platform_id() == platform_id ? CL_SUCCESS : CL_INVALID_PLATFORM, "All devices should be in the same platform");

        std::vector<cl_device_id> cl_device_ids;
        for (auto d = my_devices.begin(); d != my_devices.end(); ++d) {
            cl_device_ids.push_back((*d).my_cl_device_id);
        }

        cl_context_properties context_properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, (cl_context_properties)NULL };
        cl_int err;
        cl_context ctx = clCreateContext(context_properties,
            (cl_uint)cl_device_ids.size(),
            cl_device_ids.data(),
            NULL, NULL, &err);
        enforce_cl_retcode(err, "Failed to create context");
        my_cl_context = ctx;

        size_t device_counter = 0;
        for (auto d = my_devices.begin(); d != my_devices.end(); d++) {
            (*d).my_device_id = device_counter++;
            cl_int err2;
            cl_command_queue cq;
#if CL_VERSION_2_0
            if ((*d).major_version() >= 2) {
                if ((*d).out_of_order_exec_mode_on_host_present()) {
                    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0 };
                    cq = clCreateCommandQueueWithProperties(ctx, (*d).my_cl_device_id, props, &err2);
                } else {
                    cl_queue_properties props[] = { 0 };
                    cq = clCreateCommandQueueWithProperties(ctx, (*d).my_cl_device_id, props, &err2);
                }
            } else
#endif
            {
                cl_command_queue_properties props = (*d).out_of_order_exec_mode_on_host_present() ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : 0;
                // Suppress "declared deprecated" warning for the next line.
#if __TBB_GCC_WARNING_SUPPRESSION_PRESENT
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#if _MSC_VER || __INTEL_COMPILER
#pragma warning( push )
#if __INTEL_COMPILER
#pragma warning (disable: 1478)
#else
#pragma warning (disable: 4996)
#endif
#endif
                cq = clCreateCommandQueue(ctx, (*d).my_cl_device_id, props, &err2);
#if _MSC_VER || __INTEL_COMPILER
#pragma warning( pop )
#endif
#if __TBB_GCC_WARNING_SUPPRESSION_PRESENT
#pragma GCC diagnostic pop
#endif
            }
            enforce_cl_retcode(err2, "Failed to create command queue");
            (*d).my_cl_command_queue = cq;
        }
    }

    std::once_flag my_once_flag;
    opencl_device_list my_devices;
    cl_context my_cl_context;

    tbb::spin_mutex my_devices_mutex;

    template <typename Factory>
    friend class opencl_program;
    template <typename Factory>
    friend class opencl_buffer_impl;
    template <typename Factory>
    friend class opencl_memory;
}; // class opencl_factory

// TODO: consider this namespace as public API
namespace opencl_info {

// Default types

template <typename Factory>
struct default_device_selector {
    opencl_device operator()(Factory& f) {
        __TBB_ASSERT(!f.devices().empty(), "No available devices");
        return *(f.devices().begin());
    }
};

struct default_device_filter {
    opencl_device_list operator()(const opencl_device_list &devices) {
        opencl_device_list dl;
        cl_platform_id platform_id = devices.begin()->platform_id();
        for (opencl_device_list::const_iterator it = devices.cbegin(); it != devices.cend(); ++it) {
            if (it->platform_id() == platform_id) {
                dl.add(*it);
            }
        }
        return dl;
    }
};

class default_opencl_factory : public opencl_factory < default_device_filter >, tbb::internal::no_copy {
public:
    template<typename T> using async_msg_type = opencl_async_msg<T, default_opencl_factory>;

    friend default_opencl_factory& default_factory();

private:
    default_opencl_factory() = default;
};

inline default_opencl_factory& default_factory() {
    static default_opencl_factory default_factory;
    return default_factory;
}

} // namespace opencl_info

template <typename T, typename Factory>
opencl_buffer<T, Factory>::opencl_buffer( size_t size ) : my_impl( std::make_shared<impl_type>( size*sizeof(T), opencl_info::default_factory() ) ) {}


enum class opencl_program_type {
    SOURCE,
    PRECOMPILED,
    SPIR
};

template <typename Factory = opencl_info::default_opencl_factory>
class __TBB_DEPRECATED_IN_VERBOSE_MODE opencl_program : tbb::internal::no_assign {
public:
    typedef typename Factory::kernel_type kernel_type;

    opencl_program( Factory& factory, opencl_program_type type, const std::string& program_name ) : my_factory( factory ), my_type(type) , my_arg_str( program_name) {}
    opencl_program( Factory& factory, const char* program_name ) : opencl_program( factory, std::string( program_name ) ) {}
    opencl_program( Factory& factory, const std::string& program_name ) : opencl_program( factory, opencl_program_type::SOURCE, program_name ) {}

    opencl_program( opencl_program_type type, const std::string& program_name ) : opencl_program( opencl_info::default_factory(), type, program_name ) {}
    opencl_program( const char* program_name ) : opencl_program( opencl_info::default_factory(), program_name ) {}
    opencl_program( const std::string& program_name ) : opencl_program( opencl_info::default_factory(), program_name ) {}
    opencl_program( opencl_program_type type ) : opencl_program( opencl_info::default_factory(), type ) {}

    opencl_program( const opencl_program &src ) : my_factory( src.my_factory ), my_type( src.type ), my_arg_str( src.my_arg_str ), my_cl_program( src.my_cl_program ) {
        // Set my_do_once_flag to the called state.
        std::call_once( my_do_once_flag, [](){} );
    }

    kernel_type get_kernel( const std::string& k ) const {
        return kernel_type( get_cl_kernel(k), my_factory );
    }

private:
    opencl_program( Factory& factory, cl_program program ) : my_factory( factory ), my_cl_program( program ) {
        // Set my_do_once_flag to the called state.
        std::call_once( my_do_once_flag, [](){} );
    }

    cl_kernel get_cl_kernel( const std::string& k ) const {
        std::call_once( my_do_once_flag, [this, &k](){ this->init( k ); } );
        cl_int err;
        cl_kernel kernel = clCreateKernel( my_cl_program, k.c_str(), &err );
        enforce_cl_retcode( err, std::string( "Failed to create kernel: " ) + k );
        return kernel;
    }

    class file_reader {
    public:
        file_reader( const std::string& filepath ) {
            std::ifstream file_descriptor( filepath, std::ifstream::binary );
            if ( !file_descriptor.is_open() ) {
                std::string str = std::string( "Could not open file: " ) + filepath;
                std::cerr << str << std::endl;
                throw str;
            }
            file_descriptor.seekg( 0, file_descriptor.end );
            size_t length = size_t( file_descriptor.tellg() );
            file_descriptor.seekg( 0, file_descriptor.beg );
            my_content.resize( length );
            char* begin = &*my_content.begin();
            file_descriptor.read( begin, length );
            file_descriptor.close();
        }
        const char* content() { return &*my_content.cbegin(); }
        size_t length() { return my_content.length(); }
    private:
        std::string my_content;
    };

    class opencl_program_builder {
    public:
        typedef void (CL_CALLBACK *cl_callback_type)(cl_program, void*);
        opencl_program_builder( Factory& f, const std::string& name, cl_program program,
                                cl_uint num_devices, cl_device_id* device_list,
                                const char* options, cl_callback_type callback,
                                void* user_data ) {
            cl_int err = clBuildProgram( program, num_devices, device_list, options,
                                         callback, user_data );
            if( err == CL_SUCCESS )
                return;
            std::string str = std::string( "Failed to build program: " ) + name;
            if ( err == CL_BUILD_PROGRAM_FAILURE ) {
                const opencl_device_list &devices = f.devices();
                for ( auto d = devices.begin(); d != devices.end(); ++d ) {
                    std::cerr << "Build log for device: " << (*d).name() << std::endl;
                    size_t log_size;
                    cl_int query_err = clGetProgramBuildInfo(
                        program, (*d).my_cl_device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
                        &log_size );
                    enforce_cl_retcode( query_err, "Failed to get build log size" );
                    if( log_size ) {
                        std::vector<char> output;
                        output.resize( log_size );
                        query_err = clGetProgramBuildInfo(
                            program, (*d).my_cl_device_id, CL_PROGRAM_BUILD_LOG,
                            output.size(), output.data(), NULL );
                        enforce_cl_retcode( query_err, "Failed to get build output" );
                        std::cerr << output.data() << std::endl;
                    } else {
                        std::cerr << "No build log available" << std::endl;
                    }
                }
            }
            enforce_cl_retcode( err, str );
        }
    };

    class opencl_device_filter {
    public:
        template<typename Filter>
        opencl_device_filter( cl_uint& num_devices, cl_device_id* device_list,
                              Filter filter, const char* message ) {
            for ( cl_uint i = 0; i < num_devices; ++i )
                if ( filter(device_list[i]) ) {
                    device_list[i--] = device_list[--num_devices];
                }
            if ( !num_devices )
                enforce_cl_retcode( CL_DEVICE_NOT_AVAILABLE, message );
        }
    };

    void init( const std::string& ) const {
        cl_uint num_devices;
        enforce_cl_retcode( clGetContextInfo( my_factory.context(), CL_CONTEXT_NUM_DEVICES, sizeof( num_devices ), &num_devices, NULL ),
            "Failed to get OpenCL context info" );
        if ( !num_devices )
            enforce_cl_retcode( CL_DEVICE_NOT_FOUND, "No supported devices found" );
        cl_device_id *device_list = (cl_device_id *)alloca( num_devices*sizeof( cl_device_id ) );
        enforce_cl_retcode( clGetContextInfo( my_factory.context(), CL_CONTEXT_DEVICES, num_devices*sizeof( cl_device_id ), device_list, NULL ),
            "Failed to get OpenCL context info" );
        const char *options = NULL;
        switch ( my_type ) {
        case opencl_program_type::SOURCE: {
            file_reader fr( my_arg_str );
            const char *s[] = { fr.content() };
            const size_t l[] = { fr.length() };
            cl_int err;
            my_cl_program = clCreateProgramWithSource( my_factory.context(), 1, s, l, &err );
            enforce_cl_retcode( err, std::string( "Failed to create program: " ) + my_arg_str );
            opencl_device_filter(
                num_devices, device_list,
                []( const opencl_device& d ) -> bool {
                    return !d.compiler_available() || !d.linker_available();
                }, "No one device supports building program from sources" );
            opencl_program_builder(
                my_factory, my_arg_str, my_cl_program, num_devices, device_list,
                options, /*callback*/ NULL, /*user data*/NULL );
            break;
        }
        case opencl_program_type::SPIR:
            options = "-x spir";
        case opencl_program_type::PRECOMPILED: {
            file_reader fr( my_arg_str );
            std::vector<const unsigned char*> s(
                num_devices, reinterpret_cast<const unsigned char*>(fr.content()) );
            std::vector<size_t> l( num_devices, fr.length() );
            std::vector<cl_int> bin_statuses( num_devices, -1 );
            cl_int err;
            my_cl_program = clCreateProgramWithBinary( my_factory.context(), num_devices,
                                                       device_list, l.data(), s.data(),
                                                       bin_statuses.data(), &err );
            if( err != CL_SUCCESS ) {
                std::string statuses_str;
                for (auto st = bin_statuses.begin(); st != bin_statuses.end(); ++st) {
                    statuses_str += std::to_string((*st));
                }

                enforce_cl_retcode( err, std::string( "Failed to create program, error " + std::to_string( err ) + " : " ) + my_arg_str +
                                    std::string( ", binary_statuses = " ) + statuses_str );
            }
            opencl_program_builder(
                my_factory, my_arg_str, my_cl_program, num_devices, device_list,
                options, /*callback*/ NULL, /*user data*/NULL );
            break;
        }
        default:
            __TBB_ASSERT( false, "Unsupported program type" );
        }
    }

    Factory& my_factory;
    opencl_program_type my_type;
    std::string my_arg_str;
    mutable cl_program my_cl_program;
    mutable std::once_flag my_do_once_flag;

    template <typename DeviceFilter>
    friend class opencl_factory;

    friend class Factory::kernel;
};

template<typename... Args>
class __TBB_DEPRECATED_IN_VERBOSE_MODE opencl_node;

template<typename JP, typename Factory, typename... Ports>
class __TBB_DEPRECATED_IN_VERBOSE_MODE
opencl_node< tuple<Ports...>, JP, Factory > : public streaming_node< tuple<Ports...>, JP, Factory > {
    typedef streaming_node < tuple<Ports...>, JP, Factory > base_type;
public:
    typedef typename base_type::kernel_type kernel_type;

    opencl_node( graph &g, const kernel_type& kernel )
        : base_type( g, kernel, opencl_info::default_device_selector< opencl_info::default_opencl_factory >(), opencl_info::default_factory() )
    {
        tbb::internal::fgt_multiinput_multioutput_node( CODEPTR(), tbb::internal::FLOW_OPENCL_NODE, this, &this->my_graph );
    }

    opencl_node( graph &g, const kernel_type& kernel, Factory &f )
        : base_type( g, kernel, opencl_info::default_device_selector <Factory >(), f )
    {
        tbb::internal::fgt_multiinput_multioutput_node( CODEPTR(), tbb::internal::FLOW_OPENCL_NODE, this, &this->my_graph );
    }

    template <typename DeviceSelector>
    opencl_node( graph &g, const kernel_type& kernel, DeviceSelector d, Factory &f)
        : base_type( g, kernel, d, f)
    {
        tbb::internal::fgt_multiinput_multioutput_node( CODEPTR(), tbb::internal::FLOW_OPENCL_NODE, this, &this->my_graph );
    }
};

template<typename JP, typename... Ports>
class __TBB_DEPRECATED_IN_VERBOSE_MODE
opencl_node< tuple<Ports...>, JP > : public opencl_node < tuple<Ports...>, JP, opencl_info::default_opencl_factory > {
    typedef opencl_node < tuple<Ports...>, JP, opencl_info::default_opencl_factory > base_type;
public:
    typedef typename base_type::kernel_type kernel_type;

    opencl_node( graph &g, const kernel_type& kernel )
        : base_type( g, kernel, opencl_info::default_device_selector< opencl_info::default_opencl_factory >(), opencl_info::default_factory() )
    {}

    template <typename DeviceSelector>
    opencl_node( graph &g, const kernel_type& kernel, DeviceSelector d )
        : base_type( g, kernel, d, opencl_info::default_factory() )
    {}
};

template<typename... Ports>
class __TBB_DEPRECATED_IN_VERBOSE_MODE
opencl_node< tuple<Ports...> > : public opencl_node < tuple<Ports...>, queueing, opencl_info::default_opencl_factory > {
    typedef opencl_node < tuple<Ports...>, queueing, opencl_info::default_opencl_factory > base_type;
public:
    typedef typename base_type::kernel_type kernel_type;

    opencl_node( graph &g, const kernel_type& kernel )
        : base_type( g, kernel, opencl_info::default_device_selector< opencl_info::default_opencl_factory >(), opencl_info::default_factory() )
    {}

    template <typename DeviceSelector>
    opencl_node( graph &g, const kernel_type& kernel, DeviceSelector d )
        : base_type( g, kernel, d, opencl_info::default_factory() )
    {}
};

} // namespace interfaceX

using interface11::opencl_node;
using interface11::read_only;
using interface11::read_write;
using interface11::write_only;
using interface11::opencl_buffer;
using interface11::opencl_subbuffer;
using interface11::opencl_device;
using interface11::opencl_device_list;
using interface11::opencl_program;
using interface11::opencl_program_type;
using interface11::opencl_async_msg;
using interface11::opencl_factory;
using interface11::opencl_range;

} // namespace flow
} // namespace tbb
#endif /* __TBB_PREVIEW_OPENCL_NODE */

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_flow_graph_opencl_node_H_include_area

#endif // __TBB_flow_graph_opencl_node_H
