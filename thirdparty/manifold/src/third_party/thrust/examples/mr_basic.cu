#include <thrust/mr/allocator.h>
#include <thrust/mr/new.h>
#include <thrust/mr/pool.h>
#include <thrust/mr/disjoint_pool.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include <cassert>

template<typename Vec>
void do_stuff_with_vector(typename Vec::allocator_type alloc)
{
    Vec v1(alloc);
    v1.push_back(1);
    assert(v1.back() == 1);

    Vec v2(alloc);
    v2 = v1;

    v1.swap(v2);

    v1.clear();
    v1.resize(2);
    assert(v1.size() == 2);
}

int main()
{
    thrust::mr::new_delete_resource memres;

    {
        // no virtual calls will be issued
        typedef thrust::mr::allocator<int, thrust::mr::new_delete_resource> Alloc;
        Alloc alloc(&memres);

        do_stuff_with_vector<thrust::host_vector<int, Alloc> >(alloc);
    }

    {
        // virtual calls will be issued - wrapping in a polymorphic wrapper
        thrust::mr::polymorphic_adaptor_resource<void *> adaptor(&memres);
        typedef thrust::mr::polymorphic_allocator<int, void *> Alloc;
        Alloc alloc(&adaptor);

        do_stuff_with_vector<thrust::host_vector<int, Alloc> >(alloc);
    }

    {
        // use the global device_ptr-flavored device memory resource
        typedef thrust::device_ptr_memory_resource<thrust::device_memory_resource> Resource;
        thrust::mr::polymorphic_adaptor_resource<thrust::device_ptr<void> > adaptor(
            thrust::mr::get_global_resource<Resource>()
        );
        typedef thrust::mr::polymorphic_allocator<int, thrust::device_ptr<void> > Alloc;
        Alloc alloc(&adaptor);

        do_stuff_with_vector<thrust::device_vector<int, Alloc> >(alloc);
    }

    typedef thrust::mr::unsynchronized_pool_resource<
        thrust::mr::new_delete_resource
    > Pool;
    Pool pool(&memres);
    {
        typedef thrust::mr::allocator<int, Pool> Alloc;
        Alloc alloc(&pool);

        do_stuff_with_vector<thrust::host_vector<int, Alloc> >(alloc);
    }

    typedef thrust::mr::disjoint_unsynchronized_pool_resource<
        thrust::mr::new_delete_resource,
        thrust::mr::new_delete_resource
    > DisjointPool;
    DisjointPool disjoint_pool(&memres, &memres);
    {
        typedef thrust::mr::allocator<int, DisjointPool> Alloc;
        Alloc alloc(&disjoint_pool);

        do_stuff_with_vector<thrust::host_vector<int, Alloc> >(alloc);
    }
}
