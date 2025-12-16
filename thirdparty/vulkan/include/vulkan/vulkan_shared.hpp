// Copyright 2015-2025 The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//

// This header is generated from the Khronos Vulkan XML API Registry.

#ifndef VULKAN_SHARED_HPP
#define VULKAN_SHARED_HPP

#include <vulkan/vulkan.hpp>

#if !defined( VULKAN_HPP_CXX_MODULE )
#  include <atomic>  // std::atomic_size_t
#endif

namespace VULKAN_HPP_NAMESPACE
{
#if !defined( VULKAN_HPP_NO_SMART_HANDLE )
  template <typename HandleType>
  class SharedHandleTraits;

  class NoDestructor
  {
  };

  template <typename HandleType, typename = void>
  struct HasDestructorType : std::false_type
  {
  };

  template <typename HandleType>
  struct HasDestructorType<HandleType, decltype( (void)typename SharedHandleTraits<HandleType>::DestructorType() )> : std::true_type
  {
  };

  template <typename HandleType, typename Enable = void>
  struct GetDestructorType
  {
    using type = NoDestructor;
  };

  template <typename HandleType>
  struct GetDestructorType<HandleType, typename std::enable_if<HasDestructorType<HandleType>::value>::type>
  {
    using type = typename SharedHandleTraits<HandleType>::DestructorType;
  };

  template <class HandleType>
  using DestructorTypeOf = typename GetDestructorType<HandleType>::type;

  template <class HandleType>
  struct HasDestructor : std::integral_constant<bool, !std::is_same<DestructorTypeOf<HandleType>, NoDestructor>::value>
  {
  };

  template <typename HandleType, typename = void>
  struct HasPoolType : std::false_type
  {
  };

  template <typename HandleType>
  struct HasPoolType<HandleType, decltype( (void)typename SharedHandleTraits<HandleType>::deleter::PoolTypeExport() )> : std::true_type
  {
  };

  template <typename HandleType, typename Enable = void>
  struct GetPoolType
  {
    using type = NoDestructor;
  };

  template <typename HandleType>
  struct GetPoolType<HandleType, typename std::enable_if<HasPoolType<HandleType>::value>::type>
  {
    using type = typename SharedHandleTraits<HandleType>::deleter::PoolTypeExport;
  };

  //=====================================================================================================================

  template <typename HandleType>
  class SharedHandle;

  template <typename DestructorType, typename Deleter>
  struct SharedHeader
  {
    SharedHeader( SharedHandle<DestructorType> parent, Deleter deleter VULKAN_HPP_DEFAULT_ASSIGNMENT( Deleter() ) ) VULKAN_HPP_NOEXCEPT
      : parent( std::move( parent ) )
      , deleter( std::move( deleter ) )
    {
    }

    SharedHandle<DestructorType> parent;
    Deleter                      deleter;
  };

  template <typename Deleter>
  struct SharedHeader<NoDestructor, Deleter>
  {
    SharedHeader( Deleter deleter VULKAN_HPP_DEFAULT_ASSIGNMENT( Deleter() ) ) VULKAN_HPP_NOEXCEPT : deleter( std::move( deleter ) ) {}

    Deleter deleter;
  };

  //=====================================================================================================================

  template <typename HeaderType>
  class ReferenceCounter
  {
  public:
    template <typename... Args>
    ReferenceCounter( Args &&... control_args ) : m_header( std::forward<Args>( control_args )... )
    {
    }

    ReferenceCounter( const ReferenceCounter & )             = delete;
    ReferenceCounter & operator=( const ReferenceCounter & ) = delete;

  public:
    size_t addRef() VULKAN_HPP_NOEXCEPT
    {
      // Relaxed memory order is sufficient since this does not impose any ordering on other operations
      return m_ref_cnt.fetch_add( 1, std::memory_order_relaxed );
    }

    size_t release() VULKAN_HPP_NOEXCEPT
    {
      // A release memory order to ensure that all releases are ordered
      return m_ref_cnt.fetch_sub( 1, std::memory_order_release );
    }

  public:
    std::atomic_size_t m_ref_cnt{ 1 };
    HeaderType         m_header;
  };

  //=====================================================================================================================

  template <typename HandleType, typename HeaderType, typename ForwardType = SharedHandle<HandleType>>
  class SharedHandleBase
  {
  public:
    SharedHandleBase() = default;

    template <typename... Args>
    SharedHandleBase( HandleType handle, Args &&... control_args )
      : m_control( new ReferenceCounter<HeaderType>( std::forward<Args>( control_args )... ) ), m_handle( handle )
    {
    }

    SharedHandleBase( const SharedHandleBase & o ) VULKAN_HPP_NOEXCEPT
    {
      o.addRef();
      m_handle  = o.m_handle;
      m_control = o.m_control;
    }

    SharedHandleBase( SharedHandleBase && o ) VULKAN_HPP_NOEXCEPT
      : m_control( o.m_control )
      , m_handle( o.m_handle )
    {
      o.m_handle  = nullptr;
      o.m_control = nullptr;
    }

    SharedHandleBase & operator=( const SharedHandleBase & o ) VULKAN_HPP_NOEXCEPT
    {
      SharedHandleBase( o ).swap( *this );
      return *this;
    }

    SharedHandleBase & operator=( SharedHandleBase && o ) VULKAN_HPP_NOEXCEPT
    {
      SharedHandleBase( std::move( o ) ).swap( *this );
      return *this;
    }

    ~SharedHandleBase()
    {
      // only this function owns the last reference to the control block
      // the same principle is used in the default deleter of std::shared_ptr
      if ( m_control && ( m_control->release() == 1 ) )
      {
        // noop in x86, but does thread synchronization in ARM
        // it is required to ensure that last thread is getting to destroy the control block
        // by ordering all atomic operations before this fence
        std::atomic_thread_fence( std::memory_order_acquire );
        ForwardType::internalDestroy( getHeader(), m_handle );
        delete m_control;
      }
    }

  public:
    HandleType get() const VULKAN_HPP_NOEXCEPT
    {
      return m_handle;
    }

    HandleType operator*() const VULKAN_HPP_NOEXCEPT
    {
      return m_handle;
    }

    explicit operator bool() const VULKAN_HPP_NOEXCEPT
    {
      return bool( m_handle );
    }

#  if defined( VULKAN_HPP_SMART_HANDLE_IMPLICIT_CAST )
    operator HandleType() const VULKAN_HPP_NOEXCEPT
    {
      return m_handle;
    }
#  endif

    const HandleType * operator->() const VULKAN_HPP_NOEXCEPT
    {
      return &m_handle;
    }

    HandleType * operator->() VULKAN_HPP_NOEXCEPT
    {
      return &m_handle;
    }

    void reset() VULKAN_HPP_NOEXCEPT
    {
      SharedHandleBase().swap( *this );
    }

    void swap( SharedHandleBase & o ) VULKAN_HPP_NOEXCEPT
    {
      std::swap( m_handle, o.m_handle );
      std::swap( m_control, o.m_control );
    }

    template <typename T = HandleType>
    typename std::enable_if<HasDestructor<T>::value, const SharedHandle<DestructorTypeOf<HandleType>> &>::type getDestructorType() const VULKAN_HPP_NOEXCEPT
    {
      return getHeader().parent;
    }

  protected:
    template <typename T = HandleType>
    static typename std::enable_if<!HasDestructor<T>::value, void>::type internalDestroy( const HeaderType & control, HandleType handle ) VULKAN_HPP_NOEXCEPT
    {
      control.deleter.destroy( handle );
    }

    template <typename T = HandleType>
    static typename std::enable_if<HasDestructor<T>::value, void>::type internalDestroy( const HeaderType & control, HandleType handle ) VULKAN_HPP_NOEXCEPT
    {
      control.deleter.destroy( control.parent.get(), handle );
    }

    const HeaderType & getHeader() const VULKAN_HPP_NOEXCEPT
    {
      return m_control->m_header;
    }

  private:
    void addRef() const VULKAN_HPP_NOEXCEPT
    {
      if ( m_control )
        m_control->addRef();
    }

  protected:
    ReferenceCounter<HeaderType> * m_control = nullptr;
    HandleType                     m_handle{};
  };

  template <typename HandleType>
  class SharedHandle : public SharedHandleBase<HandleType, SharedHeader<DestructorTypeOf<HandleType>, typename SharedHandleTraits<HandleType>::deleter>>
  {
  private:
    using BaseType    = SharedHandleBase<HandleType, SharedHeader<DestructorTypeOf<HandleType>, typename SharedHandleTraits<HandleType>::deleter>>;
    using DeleterType = typename SharedHandleTraits<HandleType>::deleter;
    friend BaseType;

  public:
    SharedHandle() = default;

    template <typename T = HandleType, typename = typename std::enable_if<HasDestructor<T>::value && !HasPoolType<T>::value>::type>
    explicit SharedHandle( HandleType                                 handle,
                           SharedHandle<DestructorTypeOf<HandleType>> parent,
                           DeleterType deleter                        VULKAN_HPP_DEFAULT_ASSIGNMENT( DeleterType() ) ) VULKAN_HPP_NOEXCEPT
      : BaseType( handle, std::move( parent ), std::move( deleter ) )
    {
    }

    template <typename Dispatcher = VULKAN_HPP_DEFAULT_DISPATCHER_TYPE,
              typename T          = HandleType,
              typename            = typename std::enable_if<HasDestructor<T>::value && HasPoolType<T>::value>::type>
    explicit SharedHandle( HandleType                                           handle,
                           SharedHandle<DestructorTypeOf<HandleType>>           parent,
                           SharedHandle<typename GetPoolType<HandleType>::type> pool,
                           const Dispatcher & dispatch                          VULKAN_HPP_DEFAULT_DISPATCHER_ASSIGNMENT ) VULKAN_HPP_NOEXCEPT
      : BaseType( handle, std::move( parent ), DeleterType{ std::move( pool ), dispatch } )
    {
    }

    template <typename T = HandleType, typename = typename std::enable_if<!HasDestructor<T>::value>::type>
    explicit SharedHandle( HandleType handle, DeleterType deleter VULKAN_HPP_DEFAULT_ASSIGNMENT( DeleterType() ) ) VULKAN_HPP_NOEXCEPT
      : BaseType( handle, std::move( deleter ) )
    {
    }

  protected:
    using BaseType::internalDestroy;
  };

  namespace detail
  {
// Silence the function cast warnings.
#  if defined( __GNUC__ ) && !defined( __clang__ ) && !defined( __INTEL_COMPILER )
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wcast-function-type"
#  elif defined( __clang__ ) && ( __clang_major__ >= 13 ) && !defined( __INTEL_COMPILER )
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wcast-function-type"
#  endif

    template <typename HandleType, typename Dispatcher = VULKAN_HPP_DEFAULT_DISPATCHER_TYPE>
    class ObjectDestroyShared
    {
    public:
      using DestructorType = typename SharedHandleTraits<HandleType>::DestructorType;

      using DestroyFunctionPointerType =
        typename std::conditional<HasDestructor<HandleType>::value,
                                  void ( DestructorType::* )( HandleType, const AllocationCallbacks *, const Dispatcher & ) const,
                                  void ( HandleType::* )( const AllocationCallbacks *, const Dispatcher & ) const>::type;

      using SelectorType = typename std::conditional<HasDestructor<HandleType>::value, DestructorType, HandleType>::type;

      ObjectDestroyShared( Optional<const AllocationCallbacks> allocationCallbacks VULKAN_HPP_DEFAULT_ASSIGNMENT( nullptr ),
                           const Dispatcher & dispatch                             VULKAN_HPP_DEFAULT_DISPATCHER_ASSIGNMENT )
        : m_destroy( reinterpret_cast<decltype( m_destroy )>( static_cast<DestroyFunctionPointerType>( &SelectorType::destroy ) ) )
        , m_dispatch( &dispatch )
        , m_allocationCallbacks( allocationCallbacks )
      {
      }

    public:
      template <typename T = HandleType>
      typename std::enable_if<HasDestructor<T>::value, void>::type destroy( DestructorType parent, HandleType handle ) const VULKAN_HPP_NOEXCEPT
      {
        VULKAN_HPP_ASSERT( m_destroy && m_dispatch );
        ( parent.*m_destroy )( handle, m_allocationCallbacks, *m_dispatch );
      }

      template <typename T = HandleType>
      typename std::enable_if<!HasDestructor<T>::value, void>::type destroy( HandleType handle ) const VULKAN_HPP_NOEXCEPT
      {
        VULKAN_HPP_ASSERT( m_destroy && m_dispatch );
        ( handle.*m_destroy )( m_allocationCallbacks, *m_dispatch );
      }

    private:
      DestroyFunctionPointerType          m_destroy             = nullptr;
      const Dispatcher *                  m_dispatch            = nullptr;
      Optional<const AllocationCallbacks> m_allocationCallbacks = nullptr;
    };

    template <typename HandleType, typename Dispatcher = VULKAN_HPP_DEFAULT_DISPATCHER_TYPE>
    class ObjectFreeShared
    {
    public:
      using DestructorType = typename SharedHandleTraits<HandleType>::DestructorType;

      using DestroyFunctionPointerType = void ( DestructorType::* )( HandleType, const AllocationCallbacks *, const Dispatcher & ) const;

      ObjectFreeShared( Optional<const AllocationCallbacks> allocationCallbacks VULKAN_HPP_DEFAULT_ASSIGNMENT( nullptr ),
                        const Dispatcher & dispatch                             VULKAN_HPP_DEFAULT_DISPATCHER_ASSIGNMENT )
        : m_destroy( reinterpret_cast<decltype( m_destroy )>( static_cast<DestroyFunctionPointerType>( &DestructorType::free ) ) )
        , m_dispatch( &dispatch )
        , m_allocationCallbacks( allocationCallbacks )
      {
      }

    public:
      void destroy( DestructorType parent, HandleType handle ) const VULKAN_HPP_NOEXCEPT
      {
        VULKAN_HPP_ASSERT( m_destroy && m_dispatch );
        ( parent.*m_destroy )( handle, m_allocationCallbacks, *m_dispatch );
      }

    private:
      DestroyFunctionPointerType          m_destroy             = nullptr;
      const Dispatcher *                  m_dispatch            = nullptr;
      Optional<const AllocationCallbacks> m_allocationCallbacks = nullptr;
    };

    template <typename HandleType, typename Dispatcher = VULKAN_HPP_DEFAULT_DISPATCHER_TYPE>
    class ObjectReleaseShared
    {
    public:
      using DestructorType = typename SharedHandleTraits<HandleType>::DestructorType;

      using DestroyFunctionPointerType = void ( DestructorType::* )( HandleType, const Dispatcher & ) const;

      ObjectReleaseShared( const Dispatcher & dispatch VULKAN_HPP_DEFAULT_DISPATCHER_ASSIGNMENT )
        : m_destroy( reinterpret_cast<decltype( m_destroy )>( static_cast<DestroyFunctionPointerType>( &DestructorType::release ) ) ), m_dispatch( &dispatch )
      {
      }

    public:
      void destroy( DestructorType parent, HandleType handle ) const VULKAN_HPP_NOEXCEPT
      {
        VULKAN_HPP_ASSERT( m_destroy && m_dispatch );
        ( parent.*m_destroy )( handle, *m_dispatch );
      }

    private:
      DestroyFunctionPointerType m_destroy  = nullptr;
      const Dispatcher *         m_dispatch = nullptr;
    };

    template <typename HandleType, typename PoolType, typename Dispatcher = VULKAN_HPP_DEFAULT_DISPATCHER_TYPE>
    class PoolFreeShared
    {
    public:
      using DestructorType = typename SharedHandleTraits<HandleType>::DestructorType;

      using PoolTypeExport = PoolType;

      using ReturnType = decltype( std::declval<DestructorType>().free( PoolType(), 0u, nullptr, Dispatcher() ) );

      using DestroyFunctionPointerType = ReturnType ( DestructorType::* )( PoolType, uint32_t, const HandleType *, const Dispatcher & ) const;

      PoolFreeShared() = default;

      PoolFreeShared( SharedHandle<PoolType> pool, const Dispatcher & dispatch VULKAN_HPP_DEFAULT_DISPATCHER_ASSIGNMENT )
        : m_destroy( reinterpret_cast<decltype( m_destroy )>( static_cast<DestroyFunctionPointerType>( &DestructorType::free ) ) )
        , m_dispatch( &dispatch )
        , m_pool( std::move( pool ) )
      {
      }

    public:
      void destroy( DestructorType parent, HandleType handle ) const VULKAN_HPP_NOEXCEPT
      {
        VULKAN_HPP_ASSERT( m_destroy && m_dispatch && m_pool );
        ( parent.*m_destroy )( m_pool.get(), 1u, &handle, *m_dispatch );
      }

    private:
      DestroyFunctionPointerType m_destroy  = nullptr;
      const Dispatcher *         m_dispatch = nullptr;
      SharedHandle<PoolType>     m_pool{};
    };

#  if defined( __GNUC__ ) && !defined( __clang__ ) && !defined( __INTEL_COMPILER )
#    pragma GCC diagnostic pop
#  elif defined( __clang__ ) && ( __clang_major__ >= 13 ) && !defined( __INTEL_COMPILER )
#    pragma clang diagnostic pop
#  endif
  }  // namespace detail

  //======================
  //=== SHARED HANDLEs ===
  //======================

  //=== VK_VERSION_1_0 ===
  template <>
  class SharedHandleTraits<Instance>
  {
  public:
    using DestructorType = NoDestructor;
    using deleter        = detail::ObjectDestroyShared<Instance>;
  };

  using SharedInstance = SharedHandle<Instance>;

  template <>
  class SharedHandleTraits<Device>
  {
  public:
    using DestructorType = NoDestructor;
    using deleter        = detail::ObjectDestroyShared<Device>;
  };

  using SharedDevice = SharedHandle<Device>;

  template <>
  class SharedHandleTraits<DeviceMemory>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectFreeShared<DeviceMemory>;
  };

  using SharedDeviceMemory = SharedHandle<DeviceMemory>;

  template <>
  class SharedHandleTraits<Fence>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<Fence>;
  };

  using SharedFence = SharedHandle<Fence>;

  template <>
  class SharedHandleTraits<Semaphore>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<Semaphore>;
  };

  using SharedSemaphore = SharedHandle<Semaphore>;

  template <>
  class SharedHandleTraits<QueryPool>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<QueryPool>;
  };

  using SharedQueryPool = SharedHandle<QueryPool>;

  template <>
  class SharedHandleTraits<Buffer>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<Buffer>;
  };

  using SharedBuffer = SharedHandle<Buffer>;

  template <>
  class SharedHandleTraits<Image>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<Image>;
  };

  using SharedImage = SharedHandle<Image>;

  template <>
  class SharedHandleTraits<ImageView>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<ImageView>;
  };

  using SharedImageView = SharedHandle<ImageView>;

  template <>
  class SharedHandleTraits<CommandPool>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<CommandPool>;
  };

  using SharedCommandPool = SharedHandle<CommandPool>;

  template <>
  class SharedHandleTraits<CommandBuffer>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::PoolFreeShared<CommandBuffer, CommandPool>;
  };

  using SharedCommandBuffer = SharedHandle<CommandBuffer>;

  template <>
  class SharedHandleTraits<Event>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<Event>;
  };

  using SharedEvent = SharedHandle<Event>;

  template <>
  class SharedHandleTraits<BufferView>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<BufferView>;
  };

  using SharedBufferView = SharedHandle<BufferView>;

  template <>
  class SharedHandleTraits<ShaderModule>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<ShaderModule>;
  };

  using SharedShaderModule = SharedHandle<ShaderModule>;

  template <>
  class SharedHandleTraits<PipelineCache>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<PipelineCache>;
  };

  using SharedPipelineCache = SharedHandle<PipelineCache>;

  template <>
  class SharedHandleTraits<Pipeline>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<Pipeline>;
  };

  using SharedPipeline = SharedHandle<Pipeline>;

  template <>
  class SharedHandleTraits<PipelineLayout>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<PipelineLayout>;
  };

  using SharedPipelineLayout = SharedHandle<PipelineLayout>;

  template <>
  class SharedHandleTraits<Sampler>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<Sampler>;
  };

  using SharedSampler = SharedHandle<Sampler>;

  template <>
  class SharedHandleTraits<DescriptorPool>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<DescriptorPool>;
  };

  using SharedDescriptorPool = SharedHandle<DescriptorPool>;

  template <>
  class SharedHandleTraits<DescriptorSet>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::PoolFreeShared<DescriptorSet, DescriptorPool>;
  };

  using SharedDescriptorSet = SharedHandle<DescriptorSet>;

  template <>
  class SharedHandleTraits<DescriptorSetLayout>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<DescriptorSetLayout>;
  };

  using SharedDescriptorSetLayout = SharedHandle<DescriptorSetLayout>;

  template <>
  class SharedHandleTraits<Framebuffer>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<Framebuffer>;
  };

  using SharedFramebuffer = SharedHandle<Framebuffer>;

  template <>
  class SharedHandleTraits<RenderPass>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<RenderPass>;
  };

  using SharedRenderPass = SharedHandle<RenderPass>;

  //=== VK_VERSION_1_1 ===
  template <>
  class SharedHandleTraits<DescriptorUpdateTemplate>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<DescriptorUpdateTemplate>;
  };

  using SharedDescriptorUpdateTemplate    = SharedHandle<DescriptorUpdateTemplate>;
  using SharedDescriptorUpdateTemplateKHR = SharedHandle<DescriptorUpdateTemplate>;

  template <>
  class SharedHandleTraits<SamplerYcbcrConversion>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<SamplerYcbcrConversion>;
  };

  using SharedSamplerYcbcrConversion    = SharedHandle<SamplerYcbcrConversion>;
  using SharedSamplerYcbcrConversionKHR = SharedHandle<SamplerYcbcrConversion>;

  //=== VK_VERSION_1_3 ===
  template <>
  class SharedHandleTraits<PrivateDataSlot>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<PrivateDataSlot>;
  };

  using SharedPrivateDataSlot    = SharedHandle<PrivateDataSlot>;
  using SharedPrivateDataSlotEXT = SharedHandle<PrivateDataSlot>;

  //=== VK_KHR_surface ===
  template <>
  class SharedHandleTraits<SurfaceKHR>
  {
  public:
    using DestructorType = Instance;
    using deleter        = detail::ObjectDestroyShared<SurfaceKHR>;
  };

  using SharedSurfaceKHR = SharedHandle<SurfaceKHR>;

  //=== VK_KHR_swapchain ===
  template <>
  class SharedHandleTraits<SwapchainKHR>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<SwapchainKHR>;
  };

  using SharedSwapchainKHR = SharedHandle<SwapchainKHR>;

  //=== VK_KHR_display ===
  template <>
  class SharedHandleTraits<DisplayKHR>
  {
  public:
    using DestructorType = PhysicalDevice;
    using deleter        = detail::ObjectDestroyShared<DisplayKHR>;
  };

  using SharedDisplayKHR = SharedHandle<DisplayKHR>;

  //=== VK_EXT_debug_report ===
  template <>
  class SharedHandleTraits<DebugReportCallbackEXT>
  {
  public:
    using DestructorType = Instance;
    using deleter        = detail::ObjectDestroyShared<DebugReportCallbackEXT>;
  };

  using SharedDebugReportCallbackEXT = SharedHandle<DebugReportCallbackEXT>;

  //=== VK_KHR_video_queue ===
  template <>
  class SharedHandleTraits<VideoSessionKHR>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<VideoSessionKHR>;
  };

  using SharedVideoSessionKHR = SharedHandle<VideoSessionKHR>;

  template <>
  class SharedHandleTraits<VideoSessionParametersKHR>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<VideoSessionParametersKHR>;
  };

  using SharedVideoSessionParametersKHR = SharedHandle<VideoSessionParametersKHR>;

  //=== VK_NVX_binary_import ===
  template <>
  class SharedHandleTraits<CuModuleNVX>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<CuModuleNVX>;
  };

  using SharedCuModuleNVX = SharedHandle<CuModuleNVX>;

  template <>
  class SharedHandleTraits<CuFunctionNVX>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<CuFunctionNVX>;
  };

  using SharedCuFunctionNVX = SharedHandle<CuFunctionNVX>;

  //=== VK_EXT_debug_utils ===
  template <>
  class SharedHandleTraits<DebugUtilsMessengerEXT>
  {
  public:
    using DestructorType = Instance;
    using deleter        = detail::ObjectDestroyShared<DebugUtilsMessengerEXT>;
  };

  using SharedDebugUtilsMessengerEXT = SharedHandle<DebugUtilsMessengerEXT>;

  //=== VK_KHR_acceleration_structure ===
  template <>
  class SharedHandleTraits<AccelerationStructureKHR>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<AccelerationStructureKHR>;
  };

  using SharedAccelerationStructureKHR = SharedHandle<AccelerationStructureKHR>;

  //=== VK_EXT_validation_cache ===
  template <>
  class SharedHandleTraits<ValidationCacheEXT>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<ValidationCacheEXT>;
  };

  using SharedValidationCacheEXT = SharedHandle<ValidationCacheEXT>;

  //=== VK_NV_ray_tracing ===
  template <>
  class SharedHandleTraits<AccelerationStructureNV>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<AccelerationStructureNV>;
  };

  using SharedAccelerationStructureNV = SharedHandle<AccelerationStructureNV>;

  //=== VK_INTEL_performance_query ===
  template <>
  class SharedHandleTraits<PerformanceConfigurationINTEL>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<PerformanceConfigurationINTEL>;
  };

  using SharedPerformanceConfigurationINTEL = SharedHandle<PerformanceConfigurationINTEL>;

  //=== VK_KHR_deferred_host_operations ===
  template <>
  class SharedHandleTraits<DeferredOperationKHR>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<DeferredOperationKHR>;
  };

  using SharedDeferredOperationKHR = SharedHandle<DeferredOperationKHR>;

  //=== VK_NV_device_generated_commands ===
  template <>
  class SharedHandleTraits<IndirectCommandsLayoutNV>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<IndirectCommandsLayoutNV>;
  };

  using SharedIndirectCommandsLayoutNV = SharedHandle<IndirectCommandsLayoutNV>;

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_NV_cuda_kernel_launch ===
  template <>
  class SharedHandleTraits<CudaModuleNV>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<CudaModuleNV>;
  };

  using SharedCudaModuleNV = SharedHandle<CudaModuleNV>;

  template <>
  class SharedHandleTraits<CudaFunctionNV>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<CudaFunctionNV>;
  };

  using SharedCudaFunctionNV = SharedHandle<CudaFunctionNV>;
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_buffer_collection ===
  template <>
  class SharedHandleTraits<BufferCollectionFUCHSIA>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<BufferCollectionFUCHSIA>;
  };

  using SharedBufferCollectionFUCHSIA = SharedHandle<BufferCollectionFUCHSIA>;
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

  //=== VK_EXT_opacity_micromap ===
  template <>
  class SharedHandleTraits<MicromapEXT>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<MicromapEXT>;
  };

  using SharedMicromapEXT = SharedHandle<MicromapEXT>;

  //=== VK_ARM_tensors ===
  template <>
  class SharedHandleTraits<TensorARM>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<TensorARM>;
  };

  using SharedTensorARM = SharedHandle<TensorARM>;

  template <>
  class SharedHandleTraits<TensorViewARM>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<TensorViewARM>;
  };

  using SharedTensorViewARM = SharedHandle<TensorViewARM>;

  //=== VK_NV_optical_flow ===
  template <>
  class SharedHandleTraits<OpticalFlowSessionNV>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<OpticalFlowSessionNV>;
  };

  using SharedOpticalFlowSessionNV = SharedHandle<OpticalFlowSessionNV>;

  //=== VK_EXT_shader_object ===
  template <>
  class SharedHandleTraits<ShaderEXT>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<ShaderEXT>;
  };

  using SharedShaderEXT = SharedHandle<ShaderEXT>;

  //=== VK_KHR_pipeline_binary ===
  template <>
  class SharedHandleTraits<PipelineBinaryKHR>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<PipelineBinaryKHR>;
  };

  using SharedPipelineBinaryKHR = SharedHandle<PipelineBinaryKHR>;

  //=== VK_ARM_data_graph ===
  template <>
  class SharedHandleTraits<DataGraphPipelineSessionARM>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<DataGraphPipelineSessionARM>;
  };

  using SharedDataGraphPipelineSessionARM = SharedHandle<DataGraphPipelineSessionARM>;

  //=== VK_NV_external_compute_queue ===
  template <>
  class SharedHandleTraits<ExternalComputeQueueNV>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<ExternalComputeQueueNV>;
  };

  using SharedExternalComputeQueueNV = SharedHandle<ExternalComputeQueueNV>;

  //=== VK_EXT_device_generated_commands ===
  template <>
  class SharedHandleTraits<IndirectCommandsLayoutEXT>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<IndirectCommandsLayoutEXT>;
  };

  using SharedIndirectCommandsLayoutEXT = SharedHandle<IndirectCommandsLayoutEXT>;

  template <>
  class SharedHandleTraits<IndirectExecutionSetEXT>
  {
  public:
    using DestructorType = Device;
    using deleter        = detail::ObjectDestroyShared<IndirectExecutionSetEXT>;
  };

  using SharedIndirectExecutionSetEXT = SharedHandle<IndirectExecutionSetEXT>;

  // a number of SharedHandle specializations
  enum class SwapchainOwns
  {
    no,
    yes,
  };

  struct ImageHeader : SharedHeader<DestructorTypeOf<Image>, typename SharedHandleTraits<Image>::deleter>
  {
    ImageHeader( SharedHandle<DestructorTypeOf<Image>>               parent,
                 typename SharedHandleTraits<Image>::deleter deleter VULKAN_HPP_DEFAULT_ASSIGNMENT( typename SharedHandleTraits<Image>::deleter() ),
                 SwapchainOwns                                       swapchainOwned = SwapchainOwns::no ) VULKAN_HPP_NOEXCEPT
      : SharedHeader<DestructorTypeOf<Image>, typename SharedHandleTraits<Image>::deleter>( std::move( parent ), std::move( deleter ) )
      , swapchainOwned( swapchainOwned )
    {
    }

    SwapchainOwns swapchainOwned = SwapchainOwns::no;
  };

  template <>
  class SharedHandle<Image> : public SharedHandleBase<Image, ImageHeader>
  {
    using BaseType    = SharedHandleBase<Image, ImageHeader>;
    using DeleterType = typename SharedHandleTraits<Image>::deleter;
    friend BaseType;

  public:
    SharedHandle() = default;

    explicit SharedHandle( Image                                 handle,
                           SharedHandle<DestructorTypeOf<Image>> parent,
                           SwapchainOwns swapchain_owned         VULKAN_HPP_DEFAULT_ASSIGNMENT( SwapchainOwns::no ),
                           DeleterType deleter                   VULKAN_HPP_DEFAULT_ASSIGNMENT( DeleterType() ) ) VULKAN_HPP_NOEXCEPT
      : BaseType( handle, std::move( parent ), std::move( deleter ), swapchain_owned )
    {
    }

  protected:
    static void internalDestroy( const ImageHeader & control, Image handle ) VULKAN_HPP_NOEXCEPT
    {
      if ( control.swapchainOwned == SwapchainOwns::no )
      {
        control.deleter.destroy( control.parent.get(), handle );
      }
    }
  };

  struct SwapchainHeader
  {
    SwapchainHeader( SharedHandle<SurfaceKHR>                     surface,
                     SharedHandle<DestructorTypeOf<SwapchainKHR>> parent,
                     typename SharedHandleTraits<SwapchainKHR>::deleter deleter
                       VULKAN_HPP_DEFAULT_ASSIGNMENT( typename SharedHandleTraits<SwapchainKHR>::deleter() ) ) VULKAN_HPP_NOEXCEPT
      : surface( std::move( surface ) )
      , parent( std::move( parent ) )
      , deleter( std::move( deleter ) )
    {
    }

    SharedHandle<SurfaceKHR>                           surface;
    SharedHandle<DestructorTypeOf<SwapchainKHR>>       parent;
    typename SharedHandleTraits<SwapchainKHR>::deleter deleter;
  };

  template <>
  class SharedHandle<SwapchainKHR> : public SharedHandleBase<SwapchainKHR, SwapchainHeader>
  {
    using BaseType    = SharedHandleBase<SwapchainKHR, SwapchainHeader>;
    using DeleterType = typename SharedHandleTraits<SwapchainKHR>::deleter;
    friend BaseType;

  public:
    SharedHandle() = default;

    explicit SharedHandle( SwapchainKHR                                 handle,
                           SharedHandle<DestructorTypeOf<SwapchainKHR>> parent,
                           SharedHandle<SurfaceKHR>                     surface,
                           DeleterType deleter                          VULKAN_HPP_DEFAULT_ASSIGNMENT( DeleterType() ) ) VULKAN_HPP_NOEXCEPT
      : BaseType( handle, std::move( surface ), std::move( parent ), std::move( deleter ) )
    {
    }

  public:
    const SharedHandle<SurfaceKHR> & getSurface() const VULKAN_HPP_NOEXCEPT
    {
      return getHeader().surface;
    }

  protected:
    using BaseType::internalDestroy;
  };

  template <typename HandleType, typename DestructorType>
  class SharedHandleBaseNoDestroy : public SharedHandleBase<HandleType, DestructorType>
  {
  public:
    using SharedHandleBase<HandleType, DestructorType>::SharedHandleBase;

    const DestructorType & getDestructorType() const VULKAN_HPP_NOEXCEPT
    {
      return SharedHandleBase<HandleType, DestructorType>::getHeader();
    }

  protected:
    static void internalDestroy( const DestructorType &, HandleType ) VULKAN_HPP_NOEXCEPT {}
  };

  //=== VK_VERSION_1_0 ===

  template <>
  class SharedHandle<PhysicalDevice> : public SharedHandleBaseNoDestroy<PhysicalDevice, SharedInstance>
  {
    friend SharedHandleBase<PhysicalDevice, SharedInstance>;

  public:
    SharedHandle() = default;

    explicit SharedHandle( PhysicalDevice handle, SharedInstance parent ) noexcept
      : SharedHandleBaseNoDestroy<PhysicalDevice, SharedInstance>( handle, std::move( parent ) )
    {
    }
  };

  using SharedPhysicalDevice = SharedHandle<PhysicalDevice>;

  template <>
  class SharedHandle<Queue> : public SharedHandleBaseNoDestroy<Queue, SharedDevice>
  {
    friend SharedHandleBase<Queue, SharedDevice>;

  public:
    SharedHandle() = default;

    explicit SharedHandle( Queue handle, SharedDevice parent ) noexcept : SharedHandleBaseNoDestroy<Queue, SharedDevice>( handle, std::move( parent ) ) {}
  };

  using SharedQueue = SharedHandle<Queue>;

  //=== VK_KHR_display ===

  template <>
  class SharedHandle<DisplayModeKHR> : public SharedHandleBaseNoDestroy<DisplayModeKHR, SharedDisplayKHR>
  {
    friend SharedHandleBase<DisplayModeKHR, SharedDisplayKHR>;

  public:
    SharedHandle() = default;

    explicit SharedHandle( DisplayModeKHR handle, SharedDisplayKHR parent ) noexcept
      : SharedHandleBaseNoDestroy<DisplayModeKHR, SharedDisplayKHR>( handle, std::move( parent ) )
    {
    }
  };

  using SharedDisplayModeKHR = SharedHandle<DisplayModeKHR>;
#endif  // !VULKAN_HPP_NO_SMART_HANDLE
}  // namespace VULKAN_HPP_NAMESPACE
#endif  // VULKAN_SHARED_HPP
