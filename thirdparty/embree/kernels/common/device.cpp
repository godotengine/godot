// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "device.h"
#include "../hash.h"
#include "scene_triangle_mesh.h"
#include "scene_user_geometry.h"
#include "scene_instance.h"
#include "scene_curves.h"
#include "scene_subdiv_mesh.h"

#include "../subdiv/tessellation_cache.h"

#include "acceln.h"
#include "geometry.h"

#include "../geometry/cylinder.h"

#include "../bvh/bvh4_factory.h"
#include "../bvh/bvh8_factory.h"

#include "../../common/tasking/taskscheduler.h"
#include "../../common/sys/alloc.h"

namespace embree
{
  /*! some global variables that can be set via rtcSetParameter1i for debugging purposes */
  ssize_t Device::debug_int0 = 0;
  ssize_t Device::debug_int1 = 0;
  ssize_t Device::debug_int2 = 0;
  ssize_t Device::debug_int3 = 0;

  DECLARE_SYMBOL2(RayStreamFilterFuncs,rayStreamFilterFuncs);

  static MutexSys g_mutex;
  static std::map<Device*,size_t> g_cache_size_map;
  static std::map<Device*,size_t> g_num_threads_map;

  Device::Device (const char* cfg)
  {
    /* check that CPU supports lowest ISA */
    if (!hasISA(ISA)) {
      throw_RTCError(RTC_ERROR_UNSUPPORTED_CPU,"CPU does not support " ISA_STR);
    }

    /* set default frequency level for detected CPU */
    switch (getCPUModel()) {
    case CPU::UNKNOWN:         frequency_level = FREQUENCY_SIMD256; break;
    case CPU::XEON_ICE_LAKE:   frequency_level = FREQUENCY_SIMD256; break;
    case CPU::CORE_ICE_LAKE:   frequency_level = FREQUENCY_SIMD256; break;
    case CPU::CORE_TIGER_LAKE: frequency_level = FREQUENCY_SIMD128; break;
    case CPU::CORE_COMET_LAKE: frequency_level = FREQUENCY_SIMD128; break;
    case CPU::CORE_CANNON_LAKE:frequency_level = FREQUENCY_SIMD128; break;
    case CPU::CORE_KABY_LAKE:  frequency_level = FREQUENCY_SIMD128; break;
    case CPU::XEON_SKY_LAKE:   frequency_level = FREQUENCY_SIMD128; break;
    case CPU::CORE_SKY_LAKE:   frequency_level = FREQUENCY_SIMD128; break;
    case CPU::XEON_BROADWELL:  frequency_level = FREQUENCY_SIMD256; break;
    case CPU::CORE_BROADWELL:  frequency_level = FREQUENCY_SIMD256; break;
    case CPU::XEON_HASWELL:    frequency_level = FREQUENCY_SIMD256; break;
    case CPU::CORE_HASWELL:    frequency_level = FREQUENCY_SIMD256; break;
    case CPU::XEON_IVY_BRIDGE: frequency_level = FREQUENCY_SIMD256; break;
    case CPU::CORE_IVY_BRIDGE: frequency_level = FREQUENCY_SIMD256; break;
    case CPU::SANDY_BRIDGE:    frequency_level = FREQUENCY_SIMD256; break;
    case CPU::NEHALEM:         frequency_level = FREQUENCY_SIMD128; break;
    case CPU::CORE2:           frequency_level = FREQUENCY_SIMD128; break;
    case CPU::CORE1:           frequency_level = FREQUENCY_SIMD128; break;
    case CPU::XEON_PHI_KNIGHTS_MILL   : frequency_level = FREQUENCY_SIMD512; break;
    case CPU::XEON_PHI_KNIGHTS_LANDING: frequency_level = FREQUENCY_SIMD512; break;
#if defined(__APPLE__)
    case CPU::ARM:             frequency_level = FREQUENCY_SIMD256; break; // Apple M1 supports high throughput for SIMD4
#else
    case CPU::ARM:             frequency_level = FREQUENCY_SIMD128; break;
#endif
    }

    /* initialize global state */
#if defined(EMBREE_CONFIG)
    State::parseString(EMBREE_CONFIG);
#endif
    State::parseString(cfg);
    State::verify();

    /* check whether selected ISA is supported by the HW, as the user could have forced an unsupported ISA */    
    if (!checkISASupport()) {
      throw_RTCError(RTC_ERROR_UNSUPPORTED_CPU,"CPU does not support selected ISA");
    }    
    
    /*! do some internal tests */
    assert(isa::Cylinder::verify());

    /*! enable huge page support if desired */
#if defined(__WIN32__)
    if (State::enable_selockmemoryprivilege)
      State::hugepages_success &= win_enable_selockmemoryprivilege(State::verbosity(3));
#endif
    State::hugepages_success &= os_init(State::hugepages,State::verbosity(3));
    
    /*! set tessellation cache size */
    setCacheSize( State::tessellation_cache_size );

    /*! enable some floating point exceptions to catch bugs */
    if (State::float_exceptions)
    {
      int exceptions = _MM_MASK_MASK;
      //exceptions &= ~_MM_MASK_INVALID;
      exceptions &= ~_MM_MASK_DENORM;
      exceptions &= ~_MM_MASK_DIV_ZERO;
      //exceptions &= ~_MM_MASK_OVERFLOW;
      //exceptions &= ~_MM_MASK_UNDERFLOW;
      //exceptions &= ~_MM_MASK_INEXACT;
      _MM_SET_EXCEPTION_MASK(exceptions);
    }
    
    /* print info header */
    if (State::verbosity(1))
      print();
    if (State::verbosity(2)) 
      State::print();

    /* register all algorithms */
    bvh4_factory = make_unique(new BVH4Factory(enabled_builder_cpu_features, enabled_cpu_features));

#if defined(EMBREE_TARGET_SIMD8)
    bvh8_factory = make_unique(new BVH8Factory(enabled_builder_cpu_features, enabled_cpu_features));
#endif

    /* setup tasking system */
    initTaskingSystem(numThreads);

    /* ray stream SOA to AOS conversion */
#if defined(EMBREE_RAY_PACKETS)
    RayStreamFilterFuncsType rayStreamFilterFuncs;
    SELECT_SYMBOL_DEFAULT_SSE42_AVX_AVX2_AVX512(enabled_cpu_features,rayStreamFilterFuncs);
    rayStreamFilters = rayStreamFilterFuncs();
#endif
  }

  Device::~Device ()
  {
    setCacheSize(0);
    exitTaskingSystem();
  }

  std::string getEnabledTargets()
  {
    std::string v;
#if defined(EMBREE_TARGET_SSE2)
    v += "SSE2 ";
#endif
#if defined(EMBREE_TARGET_SSE42)
    v += "SSE4.2 ";
#endif
#if defined(EMBREE_TARGET_AVX)
    v += "AVX ";
#endif
#if defined(EMBREE_TARGET_AVX2)
    v += "AVX2 ";
#endif
#if defined(EMBREE_TARGET_AVX512)
    v += "AVX512 ";
#endif
    return v;
  }

  std::string getEmbreeFeatures()
  {
    std::string v;
#if defined(EMBREE_RAY_MASK)
    v += "raymasks ";
#endif
#if defined (EMBREE_BACKFACE_CULLING)
    v += "backfaceculling ";
#endif
#if defined (EMBREE_BACKFACE_CULLING_CURVES)
    v += "backfacecullingcurves ";
#endif
#if defined(EMBREE_FILTER_FUNCTION)
    v += "intersection_filter ";
#endif
#if defined (EMBREE_COMPACT_POLYS)
    v += "compact_polys ";
#endif
    return v;
  }

  void Device::print()
  {
    const int cpu_features = getCPUFeatures();
    std::cout << std::endl;
    std::cout << "Embree Ray Tracing Kernels " << RTC_VERSION_STRING << " (" << RTC_HASH << ")" << std::endl;
    std::cout << "  Compiler  : " << getCompilerName() << std::endl;
    std::cout << "  Build     : ";
#if defined(DEBUG)
    std::cout << "Debug " << std::endl;
#else
    std::cout << "Release " << std::endl;
#endif
    std::cout << "  Platform  : " << getPlatformName() << std::endl;
    std::cout << "  CPU       : " << stringOfCPUModel(getCPUModel()) << " (" << getCPUVendor() << ")" << std::endl;
    std::cout << "   Threads  : " << getNumberOfLogicalThreads() << std::endl;
    std::cout << "   ISA      : " << stringOfCPUFeatures(cpu_features) << std::endl;
    std::cout << "   Targets  : " << supportedTargetList(cpu_features) << std::endl;
    const bool hasFTZ = _mm_getcsr() & _MM_FLUSH_ZERO_ON;
    const bool hasDAZ = _mm_getcsr() & _MM_DENORMALS_ZERO_ON;
    std::cout << "   MXCSR    : " << "FTZ=" << hasFTZ << ", DAZ=" << hasDAZ << std::endl;
    std::cout << "  Config" << std::endl;
    std::cout << "    Threads : " << (numThreads ? toString(numThreads) : std::string("default")) << std::endl;
    std::cout << "    ISA     : " << stringOfCPUFeatures(enabled_cpu_features) << std::endl;
    std::cout << "    Targets : " << supportedTargetList(enabled_cpu_features) << " (supported)" << std::endl;
    std::cout << "              " << getEnabledTargets() << " (compile time enabled)" << std::endl;
    std::cout << "    Features: " << getEmbreeFeatures() << std::endl;
    std::cout << "    Tasking : ";
#if defined(TASKING_TBB)
    std::cout << "TBB" << TBB_VERSION_MAJOR << "." << TBB_VERSION_MINOR << " ";
  #if TBB_INTERFACE_VERSION >= 12002
    std::cout << "TBB_header_interface_" << TBB_INTERFACE_VERSION << " TBB_lib_interface_" << TBB_runtime_interface_version() << " ";
  #else
    std::cout << "TBB_header_interface_" << TBB_INTERFACE_VERSION << " TBB_lib_interface_" << tbb::TBB_runtime_interface_version() << " ";
  #endif
#endif
#if defined(TASKING_INTERNAL)
    std::cout << "internal_tasking_system ";
#endif
#if defined(TASKING_PPL)
	std::cout << "PPL ";
#endif
    std::cout << std::endl;

    /* check of FTZ and DAZ flags are set in CSR */
    if (!hasFTZ || !hasDAZ) 
    {
#if !defined(_DEBUG)
      if (State::verbosity(1)) 
#endif
      {
        std::cout << std::endl;
        std::cout << "================================================================================" << std::endl;
        std::cout << "  WARNING: \"Flush to Zero\" or \"Denormals are Zero\" mode not enabled "         << std::endl 
                  << "           in the MXCSR control and status register. This can have a severe "     << std::endl
                  << "           performance impact. Please enable these modes for each application "   << std::endl
                  << "           thread the following way:" << std::endl
                  << std::endl 
                  << "           #include \"xmmintrin.h\"" << std::endl 
                  << "           #include \"pmmintrin.h\"" << std::endl 
                  << std::endl 
                  << "           _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);" << std::endl 
                  << "           _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);" << std::endl;
        std::cout << "================================================================================" << std::endl;
        std::cout << std::endl;
      }
    }
    std::cout << std::endl;
  }

  void Device::setDeviceErrorCode(RTCError error)
  {
    RTCError* stored_error = errorHandler.error();
    if (*stored_error == RTC_ERROR_NONE)
      *stored_error = error;
  }

  RTCError Device::getDeviceErrorCode()
  {
    RTCError* stored_error = errorHandler.error();
    RTCError error = *stored_error;
    *stored_error = RTC_ERROR_NONE;
    return error;
  }

  void Device::setThreadErrorCode(RTCError error)
  {
    RTCError* stored_error = g_errorHandler.error();
    if (*stored_error == RTC_ERROR_NONE)
      *stored_error = error;
  }

  RTCError Device::getThreadErrorCode()
  {
    RTCError* stored_error = g_errorHandler.error();
    RTCError error = *stored_error;
    *stored_error = RTC_ERROR_NONE;
    return error;
  }

  void Device::process_error(Device* device, RTCError error, const char* str)
  { 
    /* store global error code when device construction failed */
    if (!device)
      return setThreadErrorCode(error);

    /* print error when in verbose mode */
    if (device->verbosity(1)) 
    {
      switch (error) {
      case RTC_ERROR_NONE         : std::cerr << "Embree: No error"; break;
      case RTC_ERROR_UNKNOWN    : std::cerr << "Embree: Unknown error"; break;
      case RTC_ERROR_INVALID_ARGUMENT : std::cerr << "Embree: Invalid argument"; break;
      case RTC_ERROR_INVALID_OPERATION: std::cerr << "Embree: Invalid operation"; break;
      case RTC_ERROR_OUT_OF_MEMORY    : std::cerr << "Embree: Out of memory"; break;
      case RTC_ERROR_UNSUPPORTED_CPU  : std::cerr << "Embree: Unsupported CPU"; break;
      default                   : std::cerr << "Embree: Invalid error code"; break;                   
      };
      if (str) std::cerr << ", (" << str << ")";
      std::cerr << std::endl;
    }

    /* call user specified error callback */
    if (device->error_function) 
      device->error_function(device->error_function_userptr,error,str); 

    /* record error code */
    device->setDeviceErrorCode(error);
  }

  void Device::memoryMonitor(ssize_t bytes, bool post)
  {
    if (State::memory_monitor_function && bytes != 0) {
      if (!State::memory_monitor_function(State::memory_monitor_userptr,bytes,post)) {
        if (bytes > 0) { // only throw exception when we allocate memory to never throw inside a destructor
          throw_RTCError(RTC_ERROR_OUT_OF_MEMORY,"memory monitor forced termination");
        }
      }
    }
  }

  size_t getMaxNumThreads()
  {
    size_t maxNumThreads = 0;
    for (std::map<Device*,size_t>::iterator i=g_num_threads_map.begin(); i != g_num_threads_map.end(); i++)
      maxNumThreads = max(maxNumThreads, (*i).second);
    if (maxNumThreads == 0)
      maxNumThreads = std::numeric_limits<size_t>::max();
    return maxNumThreads;
  }

  size_t getMaxCacheSize()
  {
    size_t maxCacheSize = 0;
    for (std::map<Device*,size_t>::iterator i=g_cache_size_map.begin(); i!= g_cache_size_map.end(); i++)
      maxCacheSize = max(maxCacheSize, (*i).second);
    return maxCacheSize;
  }
 
  void Device::setCacheSize(size_t bytes) 
  {
#if defined(EMBREE_GEOMETRY_SUBDIVISION)
    Lock<MutexSys> lock(g_mutex);
    if (bytes == 0) g_cache_size_map.erase(this);
    else            g_cache_size_map[this] = bytes;
    
    size_t maxCacheSize = getMaxCacheSize();
    resizeTessellationCache(maxCacheSize);
#endif
  }

  void Device::initTaskingSystem(size_t numThreads) 
  {
    Lock<MutexSys> lock(g_mutex);
    if (numThreads == 0) 
      g_num_threads_map[this] = std::numeric_limits<size_t>::max();
    else 
      g_num_threads_map[this] = numThreads;

    /* create task scheduler */
    size_t maxNumThreads = getMaxNumThreads();
    TaskScheduler::create(maxNumThreads,State::set_affinity,State::start_threads);
#if USE_TASK_ARENA
    const size_t nThreads = min(maxNumThreads,TaskScheduler::threadCount());
    const size_t uThreads = min(max(numUserThreads,(size_t)1),nThreads);
    arena = make_unique(new tbb::task_arena((int)nThreads,(unsigned int)uThreads));
#endif
  }

  void Device::exitTaskingSystem() 
  {
    Lock<MutexSys> lock(g_mutex);
    g_num_threads_map.erase(this);

    /* terminate tasking system */
    if (g_num_threads_map.size() == 0) {
      TaskScheduler::destroy();
    } 
    /* or configure new number of threads */
    else {
      size_t maxNumThreads = getMaxNumThreads();
      TaskScheduler::create(maxNumThreads,State::set_affinity,State::start_threads);
    }
#if USE_TASK_ARENA
    arena.reset();
#endif
  }

  void Device::setProperty(const RTCDeviceProperty prop, ssize_t val)
  {
    /* hidden internal properties */
    switch ((size_t)prop)
    {
    case 1000000: debug_int0 = val; return;
    case 1000001: debug_int1 = val; return;
    case 1000002: debug_int2 = val; return;
    case 1000003: debug_int3 = val; return;
    }

    throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "unknown writable property");
  }

  ssize_t Device::getProperty(const RTCDeviceProperty prop)
  {
    size_t iprop = (size_t)prop;

    /* get name of internal regression test */
    if (iprop >= 2000000 && iprop < 3000000)
    {
      RegressionTest* test = getRegressionTest(iprop-2000000);
      if (test) return (ssize_t) test->name.c_str();
      else      return 0;
    }

    /* run internal regression test */
    if (iprop >= 3000000 && iprop < 4000000)
    {
      RegressionTest* test = getRegressionTest(iprop-3000000);
      if (test) return test->run();
      else      return 0;
    }

    /* documented properties */
    switch (prop) 
    {
    case RTC_DEVICE_PROPERTY_VERSION_MAJOR: return RTC_VERSION_MAJOR;
    case RTC_DEVICE_PROPERTY_VERSION_MINOR: return RTC_VERSION_MINOR;
    case RTC_DEVICE_PROPERTY_VERSION_PATCH: return RTC_VERSION_PATCH;
    case RTC_DEVICE_PROPERTY_VERSION      : return RTC_VERSION;

#if defined(EMBREE_TARGET_SIMD4) && defined(EMBREE_RAY_PACKETS)
    case RTC_DEVICE_PROPERTY_NATIVE_RAY4_SUPPORTED:  return hasISA(SSE2);
#else
    case RTC_DEVICE_PROPERTY_NATIVE_RAY4_SUPPORTED:  return 0;
#endif

#if defined(EMBREE_TARGET_SIMD8) && defined(EMBREE_RAY_PACKETS)
    case RTC_DEVICE_PROPERTY_NATIVE_RAY8_SUPPORTED:  return hasISA(AVX);
#else
    case RTC_DEVICE_PROPERTY_NATIVE_RAY8_SUPPORTED:  return 0;
#endif

#if defined(EMBREE_TARGET_SIMD16) && defined(EMBREE_RAY_PACKETS)
    case RTC_DEVICE_PROPERTY_NATIVE_RAY16_SUPPORTED: return hasISA(AVX512);
#else
    case RTC_DEVICE_PROPERTY_NATIVE_RAY16_SUPPORTED: return 0;
#endif

#if defined(EMBREE_RAY_PACKETS)
    case RTC_DEVICE_PROPERTY_RAY_STREAM_SUPPORTED:  return 1;
#else
    case RTC_DEVICE_PROPERTY_RAY_STREAM_SUPPORTED:  return 0;
#endif
    
#if defined(EMBREE_RAY_MASK)
    case RTC_DEVICE_PROPERTY_RAY_MASK_SUPPORTED: return 1;
#else
    case RTC_DEVICE_PROPERTY_RAY_MASK_SUPPORTED: return 0;
#endif

#if defined(EMBREE_BACKFACE_CULLING)
    case RTC_DEVICE_PROPERTY_BACKFACE_CULLING_ENABLED: return 1;
#else
    case RTC_DEVICE_PROPERTY_BACKFACE_CULLING_ENABLED: return 0;
#endif

#if defined(EMBREE_BACKFACE_CULLING_CURVES)
    case RTC_DEVICE_PROPERTY_BACKFACE_CULLING_CURVES_ENABLED: return 1;
#else
    case RTC_DEVICE_PROPERTY_BACKFACE_CULLING_CURVES_ENABLED: return 0;
#endif

#if defined(EMBREE_COMPACT_POLYS)
    case RTC_DEVICE_PROPERTY_COMPACT_POLYS_ENABLED: return 1;
#else
    case RTC_DEVICE_PROPERTY_COMPACT_POLYS_ENABLED: return 0;
#endif

#if defined(EMBREE_FILTER_FUNCTION)
    case RTC_DEVICE_PROPERTY_FILTER_FUNCTION_SUPPORTED: return 1;
#else
    case RTC_DEVICE_PROPERTY_FILTER_FUNCTION_SUPPORTED: return 0;
#endif

#if defined(EMBREE_IGNORE_INVALID_RAYS)
    case RTC_DEVICE_PROPERTY_IGNORE_INVALID_RAYS_ENABLED: return 1;
#else
    case RTC_DEVICE_PROPERTY_IGNORE_INVALID_RAYS_ENABLED: return 0;
#endif

#if defined(TASKING_INTERNAL)
    case RTC_DEVICE_PROPERTY_TASKING_SYSTEM: return 0;
#endif

#if defined(TASKING_TBB)
    case RTC_DEVICE_PROPERTY_TASKING_SYSTEM: return 1;
#endif

#if defined(TASKING_PPL)
    case RTC_DEVICE_PROPERTY_TASKING_SYSTEM: return 2;
#endif

#if defined(EMBREE_GEOMETRY_TRIANGLE)
    case RTC_DEVICE_PROPERTY_TRIANGLE_GEOMETRY_SUPPORTED: return 1;
#else
    case RTC_DEVICE_PROPERTY_TRIANGLE_GEOMETRY_SUPPORTED: return 0;
#endif
        
#if defined(EMBREE_GEOMETRY_QUAD)
    case RTC_DEVICE_PROPERTY_QUAD_GEOMETRY_SUPPORTED: return 1;
#else
    case RTC_DEVICE_PROPERTY_QUAD_GEOMETRY_SUPPORTED: return 0;
#endif

#if defined(EMBREE_GEOMETRY_CURVE)
    case RTC_DEVICE_PROPERTY_CURVE_GEOMETRY_SUPPORTED: return 1;
#else
    case RTC_DEVICE_PROPERTY_CURVE_GEOMETRY_SUPPORTED: return 0;
#endif

#if defined(EMBREE_GEOMETRY_SUBDIVISION)
    case RTC_DEVICE_PROPERTY_SUBDIVISION_GEOMETRY_SUPPORTED: return 1;
#else
    case RTC_DEVICE_PROPERTY_SUBDIVISION_GEOMETRY_SUPPORTED: return 0;
#endif

#if defined(EMBREE_GEOMETRY_USER)
    case RTC_DEVICE_PROPERTY_USER_GEOMETRY_SUPPORTED: return 1;
#else
    case RTC_DEVICE_PROPERTY_USER_GEOMETRY_SUPPORTED: return 0;
#endif

#if defined(EMBREE_GEOMETRY_POINT)
    case RTC_DEVICE_PROPERTY_POINT_GEOMETRY_SUPPORTED: return 1;
#else
    case RTC_DEVICE_PROPERTY_POINT_GEOMETRY_SUPPORTED: return 0;
#endif

#if defined(TASKING_PPL)
    case RTC_DEVICE_PROPERTY_JOIN_COMMIT_SUPPORTED: return 0;
#elif defined(TASKING_TBB) && (TBB_INTERFACE_VERSION_MAJOR < 8)
    case RTC_DEVICE_PROPERTY_JOIN_COMMIT_SUPPORTED: return 0;
#else
    case RTC_DEVICE_PROPERTY_JOIN_COMMIT_SUPPORTED: return 1;
#endif

#if defined(TASKING_TBB) && TASKING_TBB_USE_TASK_ISOLATION
    case RTC_DEVICE_PROPERTY_PARALLEL_COMMIT_SUPPORTED: return 1;
#else
    case RTC_DEVICE_PROPERTY_PARALLEL_COMMIT_SUPPORTED: return 0;
#endif

    default: throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "unknown readable property"); break;
    };
  }
}
