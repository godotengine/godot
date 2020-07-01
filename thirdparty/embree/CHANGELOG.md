Version History
---------------

### New Features in Embree 3.5.2
-   Added EMBREE_ISA_NAMESPACE cmake option that allows to put all Embree API functions
    inside a user defined namespace.
-   Added EMBREE_LIBRARY_NAME cmake option that allows to rename the Embree library.
-   When Embree is compiled as static library, EMBREE_STATIC_LIB has no longer to get
    defined before including the Embree API headers.
-   Added CPU frequency_level device configuration to allow an application to specify the
    frequency level it wants to run on. This forces Embree to not use optimizations that
    may reduce the CPU frequency below that level. By default Embree is configured to the
    the AVX-heavy frequency level, thus if the application uses solely non-AVX code, configuring
    the Embree device with "frequency_level=simd128" may give better performance.
-   Fixed a bug in the spatial split builder which caused it to fail
    for scenes with more than 2^24 geometries.

### New Features in Embree 3.5.1
-   Fixed ray/sphere intersector to work also for non-normalized rays.
-   Fixed self intersection avoidance for ray oriented discs when
    non-normalized rays were used.
-   Increased maximal face valence for subdiv patch to 64 and reduced stack size
    requirement for subdiv patch evaluation.

### New Features in Embree 3.5.0
-   Changed normal oriented curve definition to fix waving artefacts.
-   Fixed bounding issue for normal oriented motion blurred curves.
-   Fixed performance issue with motion blurred point geometry.
-   Fixed generation of documentation with new pandoc versions.

### New Features in Embree 3.4.0
-   Added point primitives (spheres, ray-oriented discs, normal-oriented discs).
-   Fixed crash triggered by scenes with only invalid primitives.
-   Improved robustness of quad/grid-based intersectors.
-   Upgraded to TBB 2019.2 for release builds.

### New Features in Embree 3.3.0
-   Added support for motion blur time range per geometry. This way geometries
    can appear and disappear during the camera shutter and time steps do not have
    to start and end at camera shutter interval boundaries.
-   Fixed crash with pathtracer when using --triangle-sphere command line.
-   Fixed crash with pathtracer when using --shader ao command line.
-   Fixed tutorials showing a black window on macOS 10.14 until moved.

### New Features in Embree 3.2.4
-   Fixed compile issues with ICC 2019.
-   Released ZIP files for Windows are now provided in a
    version linked against Visual Studio 2013 and Visual Studio 2015.

### New Features in Embree 3.2.3
-   Fixed crash when using curves with RTC_SCENE_FLAG_DYNAMIC
    combined with RTC_BUILD_QUALITY_MEDIUM.

### New Features in Embree 3.2.2
-   Fixed intersection distance for unnormalized rays with line segments.
-   Removed libmmd.dll dependency in release builds for Windows.
-   Fixed detection of AppleClang compiler under MacOSX.

### New Features in Embree 3.2.1
-   Bugfix in flat mode for hermite curves.
-   Added EMBREE_CURVE_SELF_INTERSECTION_AVOIDANCE_FACTOR cmake option to
    control self intersection avoidance for flat curves.
-   Performance fix when instantiating motion blurred scenes. The application
    should best use two (or more) time steps for an instance that instantiates
    a motion blurred scene.
-   Fixed AVX512 compile issue with GCC 6.1.1.
-   Fixed performance issue with rtcGetGeometryUserData when used
    during rendering.
-   Bugfix in length of derivatives for grid geometry.
-   Added BVH8 support for motion blurred curves and lines. For some workloads
    this increases performance by up to 7%.
-   Fixed rtcGetGeometryTransform to return the local to world transform.
-   Fixed bug in multi segment motion blur that caused missing of perfectly
    axis aligned geometry.
-   Reduced memory consumption of small scenes by 4x.
-   Reduced temporal storage of grid builder.

### New Features in Embree 3.2.0
-   Improved watertightness of robust mode.
-   Line segments, and other curves are now all contained in a single
    BVH which improves performance when these are both used in a scene.
-   Performance improvement of up to 20% for line segments.
-   Bugfix to Embree2 to Embree3 conversion script.
-   Added support for Hermite curve basis.
-   Semantics of normal buffer for normal oriented curves has
    changed to simplify usage. Please see documentation for details.
-   Using GLFW and imgui in tutorials.
-   Fixed floating point exception in static variable initialization.
-   Fixed invalid memory access in rtcGetGeometryTransform for non-motion
    blur instances.
-   Improved self intersection avoidance for flat curves. Transparency rays
    with tnear set to previous hit distance do not need curve radius
    based self intersection avoidance as same hit is calculated again. For this
    reason self intersection avoidance is now only applied to ray origin.

### New Features in Embree 3.1.0
-   Added new normal-oriented curve primitive for ray tracing of grass-like
    structures.
-   Added new grid primitive for ray tracing tessellated and displaced surfaces
    in very memory efficient manner.
-   Fixed bug of ribbon curve intersector when derivative was zero.
-   Installing all static libraries when EMBREE_STATIC_LIB is enabled.
-   Added API functions to access topology of subdivision mesh.
-   Reduced memory consumption of instances.
-   Improved performance of instances by 8%.
-   Reduced memory consumption of curves by up to 2x.
-   Up to 5% higher performance on AVX-512 architectures.
-   Added native support for multiple curve basis functions. Internal
    basis conversions are no longer performed, which saves additional
    memory when multiple bases are used.
-   Fixed issue with non thread safe local static variable initialization
    in VS2013.
-   Bugfix in rtcSetNewGeometry. Vertex buffers did not get properly
    overallocated.
-   Replaced ImageMagick with OpenImageIO in the tutorials.

### New Features in Embree 3.0.0
-   Switched to a new version of the API which provides improved
    flexibility but is not backward compatible. Please see "Upgrading
    from Embree 2 to Embree 3" section of the documentation for upgrade
    instructions. In particular, we provide a Python script that performs
    most of the transition work.
-   User geometries inside an instanced scene and a top-level scene no
    longer need to handle the instID field of the ray differently. They
    both just need to copy the context.instID into the ray.instID field.
-   Support for context filter functions that can be assigned to a ray
    query.
-   User geometries can now invoke filter functions using the
    rtcFilterIntersection and rtcFilterOcclusion calls.
-   Higher flexibility through specifying build quality per scene and
    geometry.
-   Geometry normal uses commonly used right-hand rule from now on.
-   Added self-intersection avoidance to ribbon curves and lines.
    Applications do not have to implement self-intersection workarounds
    for these primitive types anymore.
-   Added support for 4 billion primitives in a single scene.
-   Removed the RTC_MAX_USER_VERTEX_BUFFERS and RTC_MAX_INDEX_BUFFERS
    limitations.
-   Reduced memory consumption by 192 bytes per instance.
-   Fixed some performance issues on AVX-512 architectures.
-   Individual Contributor License Agreement (ICLA) and Corporate
    Contributor License Agreement (CCLA) no longer required to
    contribute to the project.

### New Features in Embree 2.17.5
-   Improved watertightness of robust mode.
-   Fixed floating point exception in static variable initialization.
-   Fixed AVX512 compile issue with GCC 6.1.1.

### New Features in Embree 2.17.4
-   Fixed AVX512 compile issue with GCC 7.
-   Fixed issue with not thread safe local static variable
    initialization in VS2013.
-   Fixed bug in the 4 and 8-wide packet intersection of instances with
    multi-segment motion blur on AVX-512 architectures.
-   Fixed bug in rtcOccluded4/8/16 when only AVX-512 ISA was enabled.

### New Features in Embree 2.17.3
-   Fixed GCC compile warning in debug mode.
-   Fixed bug of ribbon curve intersector when derivative was zero.
-   Installing all static libraries when EMBREE_STATIC_LIB is enabled.

### New Features in Embree 2.17.2
-   Made BVH build of curve geometry deterministic.

### New Features in Embree 2.17.1
-   Improved performance of occlusion ray packets by up to 50%.
-   Fixed detection of Clang for CMake 3 under MacOSX
-   Fixed AVX code compilation issue with GCC 7 compiler caused by
    explicit use of vzeroupper intrinsics.
-   Fixed an issue where Clang address sanitizer reported an error in
    the internal tasking system.
-   Added fix to compile on 32 bit Linux distribution.
-   Fixed some wrong relative include paths in Embree.
-   Improved performance of robust single ray mode by 5%.
-   Added EMBREE_INSTALL_DEPENDENCIES option (default OFF) to enable
    installing of Embree dependencies.
-   Fixed performance regression for occlusion ray streams.
-   Reduced temporary memory requirements of BVH builder for curves and
    line segments.
-   Fixed performance regression for user geometries and packet ray tracing.
-   Fixed bug where wrong closest hit was reported for very curvy hair segment.

### New Features in Embree 2.17.0
-   Improved packet ray tracing performance for coherent rays by 10-60%
    (requires RTC_INTERSECT_COHERENT flag).
-   Improved ray tracing performance for incoherent rays on
    AVX-512 architectures by 5%.
-   Improved ray tracing performance for streams of incoherent rays
    by 5-15%.
-   Fixed tbb_debug.lib linking error under Windows.
-   Fast coherent ray stream and packet code paths now also work in robust mode.
-   Using less agressive prefetching for large BVH nodes which
    results in 1-2% higher ray tracing performance.
-   Precompiled binaries have stack-protector enabled, except for
    traversal kernels. BVH builders can be slightly slower due to this
    change. If you want stack-protectors disabled please turn off
    EMBREE_STACK_PROTECTOR in cmake and build the binaries yourself.
-   When enabling ISAs individually, the 8-wide BVH was previously only
    available when the AVX ISA was also selected. This issue is now
    fixed, and one can enable only AVX2 and still get best
    performance by using an 8-wide BVH.
-   Fixed rtcOccluded1 and rtcOccluded1Ex API functions which were
    broken in ISPC.
-   Providing MSI installer for Windows.

### New Features in Embree 2.16.5
-   Bugfix in the robust triangle intersector that rarely caused NaNs.
-   Fixed bug in hybrid traversal kernel when BVH leaf was entered with no
    active rays. This rarely caused crashes when used with instancing.
-   Fixed bug introduced in Embree 2.16.2 which caused instancing not to
    work properly when a smaller than the native SIMD width was
    used in ray packet mode.
-   Fixed bug in the curve geometry intersector that caused rendering
    artefacts for Bézier curves with p0=p1 and/or p2=p3.
-   Fixed bug in the curve geometry intersector that caused hit results
    with NaNs to be reported.
-   Fixed masking bug that caused rare cracks in curve geometry.
-   Enabled support for SSE2 in precompiled binaries again.

### New Features in Embree 2.16.4
-   Bugfix in the ribbon intersector for hair primitives. Non-normalized
    rays caused wrong intersection distance to be reported.

### New Features in Embree 2.16.3
-   Increased accuracy for handling subdivision surfaces. This fixes
    cracks when using displacement mapping but reduces performance
    at irregular vertices.
-   Fixed a bug where subdivision geometry was not properly updated
    when modifying only the tesselation rate and vertex array.

### New Features in Embree 2.16.2
-   Fixed bug that caused NULL intersection context in intersection
    filter when instancing was used.
-   Fixed an issue where uv's where outside the triangle (or quad) for
    very small triangles (or quads). In robust mode we improved the uv
    calculation to avoid that issue, in fast mode we accept that
    inconsistency for better performance.
-   Changed UV encoding for non-quad subdivision patches to
    allow a subpatch UV range of `[-0.5,1.5[`. Using this new encoding
    one can use finite differences to calculate derivatives if required.
    Please adjust your code in case you rely on the old encoding.

### New Features in Embree 2.16.1
-   Workaround for compile issues with Visual Studio 2017
-   Fixed bug in subdiv code for static scenes when using tessellation
    levels larger than 50.
-   Fixed low performance when adding many geometries to a scene.
-   Fixed high memory consumption issue when using instances in
    dynamic scene (by disabling two level builder for user geometries
    and instances).

### New Features in Embree 2.16.0
-   Improved multi-segment motion blur support for scenes with
    different number of time steps per mesh.
-   New top level BVH builder that improves build times and BVH quality
    of two-level BVHs.
-   Added support to enable only a single ISA. Previously code was
    always compiled for SSE2.
-   Improved single ray tracing performance for incoherent rays on
    AVX-512 architectures by 5-10%.
-   Improved packet/hybrid ray tracing performance for incoherent rays
    on AVX-512 architectures by 10-30%.
-   Improved stream ray tracing performance for coherent rays in
    structure-of-pointers layout by 40-70%.
-   BVH builder for compact scenes of triangles and quads needs
    essentially no temporary memory anymore. This doubles the
    maximal scene size that can be rendered in compact mode.
-   Triangles no longer store the geometry normal in fast/default mode
    which reduces memory consumption by up to 20%.
-   Compact mode uses BVH4 now consistently which reduces memory
    consumption by up to 10%.
-   Reduced memory consumption for small scenes (of 10k-100k primitives)
    and dynamic scenes.
-   Improved performance of user geometries and instances through BVH8
    support.
-   The API supports now specifying the geometry ID of a geometry at
    construction time. This way matching the geometry ID used by
    Embree and the application is simplified.
-   Fixed a bug that would have caused a failure of the BVH builder
    for dynamic scenes when run on a machine with more then 1000 threads.
-   Fixed a bug that could have been triggered when reaching the maximal
    number of mappings under Linux (`vm.max_map_count`). This could have
    happened when creating a large number of small static scenes.
-   Added huge page support for Windows and MacOSX (experimental).
-   Added support for Visual Studio 2017.
-   Removed support for Visual Studio 2012.
-   Precompiled binaries now require a CPU supporting at least the
    SSE4.2 ISA.
-   We no longer provide precompiled binaries for 32-bit on Windows.
-   Under Windows one now has to use the platform toolset option in
    CMake to switch to Clang or the Intel® Compiler.
-   Fixed a bug for subdivision meshes when using the incoherent scene
    flag.
-   Fixed a bug in the line geometry intersection, that caused reporting
    an invalid line segment intersection with primID -1.
-   Buffer stride for vertex buffers of different time steps of triangle
    and quad meshes have to be identical now.
-   Fixed a bug in the curve geometry intersection code when passed a
    perfect cylinder.

### New Features in Embree 2.15.0

-   Added `rtcCommitJoin` mode that allows thread to join a build
    operation. When using the internal tasking system this allows
    Embree to solely use the threads that called `rtcCommitJoin` to
    build the scene, while previously also normal worker threads
    participated in the build. You should no longer use `rtcCommit`
    to join a build.
-   Added `rtcDeviceSetErrorFunction2` API call, which sets an error
    callback function which additionally gets passed a user provided
    pointer (`rtcDeviceSetErrorFunction` is now deprecated).
-   Added `rtcDeviceSetMemoryMonitorFunction2` API call, which sets a
    memory monitor callback function which additionally get passed a
    user provided pointer. (`rtcDeviceSetMemoryMonitorFunction` is now
    deprecated).
-   Build performance for hair geometry improved by up to 2×.
-   Standard BVH build performance increased by 5%.
-   Added API extension to use internal Morton-code based builder, the
    standard binned-SAH builder, and the spatial split-based SAH builder.
-   Added support for BSpline hair and curves. Embree uses
    either the Bézier or BSpline basis internally, and converts other
    curves, which requires more memory during rendering. For reduced
    memory consumption set the `EMBREE_NATIVE_SPLINE_BASIS` to the basis
    your application uses (which is set to `BEZIER` by default).
-   Setting the number of threads through `tbb::taskscheduler_init`
    object on the application side is now working properly.
-   Windows and Linux releases are build using AVX-512 support.
-   Implemented hybrid traversal for hair and line segments for
    improved ray packet performance.
-   AVX-512 code compiles with Clang 4.0.0
-   Fixed crash when ray packets were disabled in CMake.

### New Features in Embree 2.14.0

-   Added `ignore_config_files` option to init flags that allows the
    application to ignore Embree configuration files.
-   Face-varying interpolation is now supported for subdivision surfaces.
-   Up to 16 user vertex buffers are supported for vertex
    attribute interpolation.
-   Deprecated `rtcSetBoundaryMode` function, please use the new
    `rtcSetSubdivisionMode` function.
-   Added `RTC_SUBDIV_PIN_BOUNDARY` mode for handling boundaries of
    subdivision meshes.
-   Added `RTC_SUBDIV_PIN_ALL` mode to enforce linear interpolation
    for subdivision meshes.
-   Optimized object generation performance for dynamic scenes.
-   Reduced memory consumption when using lots of small dynamic objects.
-   Fixed bug for subdivision surfaces using low tessellation rates.
-   Hair geometry now uses a new ribbon intersector that intersects with
    ray-facing quads. The new intersector also returns the v-coordinate
    of the hair intersection, and fixes artefacts at junction points
    between segments, at the cost of a small performance hit.
-   Added `rtcSetBuffer2` function, that additionally gets the number of
    elements of a buffer. In dynamic scenes, this function allows to
    quickly change buffer sizes, making it possible to change the number
    of primitives of a mesh or the number of crease features for
    subdivision surfaces.
-   Added simple 'viewer_anim' tutorial for rendering key
    frame animations and 'buildbench' for measuring BVH (re-)build
    performance for static and dynamic scenes.
-   Added more AVX-512 optimizations for future architectures.

### New Features in Embree 2.13.0

-   Improved performance for compact (but not robust) scenes.
-   Added robust mode for motion blurred triangles and quads.
-   Added fast dynamic mode for user geometries.
-   Up to 20% faster BVH build performance on the second generation
    Intel® Xeon Phi™ processor codenamed Knights Landing.
-   Improved quality of the spatial split builder.
-   Improved performance for coherent streams of ray packets (SOA
    layout), e.g. for fast primary visibility.
-   Various bug fixes in tessellation cache, quad-based spatial
    split builder, etc.

### New Features in Embree 2.12.0

-   Added support for multi-segment motion blur for all primitive types.
-   API support for stream of pointers to single rays (`rtcIntersect1Mp`
    and `rtcOccluded1Mp`)
-   Improved BVH refitting performance for dynamic scenes.
-   Improved high-quality mode for quads (added spatial split builder
    for quads)
-   Faster dynamic scenes for triangle and quad-based meshes on AVX2
    enabled machines.
-   Performance and correctness bugfix in optimization for streams of
    coherent (single) rays.
-   Fixed large memory consumption (issue introduced in Embree v2.11.0).
    If you use Embree v2.11.0 please upgrade to Embree v2.12.0.
-   Reduced memory consumption for dynamic scenes containing small
    meshes.
-   Added support to start and affinitize TBB worker threads by passing
    "`start_threads=1,set_affinity=1`" to `rtcNewDevice`. These settings
    are recommended on systems with a high thread count.
-   `rtcInterpolate2` can now be called within a displacement shader.
-   Added initial support for Microsoft's Parallel Pattern Library (PPL)
    as tasking system alternative (for optimal performance TBB is
    highly recommended).
-   Updated to TBB 2017 which is released under the Apache v2.0 license.
-   Dropped support for Visual Studio 2012 Win32 compiler. Visual Studio
    2012 x64 is still supported.

### New Features in Embree 2.11.0

-   Improved performance for streams of coherent (single) rays flagged
    with `RTC_INTERSECT_COHERENT`. For such coherent ray streams, e.g.
    primary rays, the performance typically improves by 1.3-2×.
-   New spatial split BVH builder for triangles, which is 2-6× faster
    than the previous version and more memory conservative.
-   Improved performance and scalability of all standard BVH builders on
    systems with large core counts.
-   Fixed `rtcGetBounds` for motion blur scenes.
-   Thread affinity is now on by default when running on the latest
    Intel® Xeon Phi™ processor.
-   Added AVX-512 support for future Intel® Xeon processors.

### New Features in Embree 2.10.0

-   Added a new curve geometry which renders the sweep surface of a
    circle along a Bézier curve.
-   Intersection filters can update the `tfar` ray distance.
-   Geometry types can get disabled at compile time.
-   Modified and extended the ray stream API.
-   Added new callback mechanism for the ray stream API.
-   Improved ray stream performance (up to 5-10%).
-   Up to 20% faster morton builder on machines with large core counts.
-   Lots of optimizations for the second generation Intel® Xeon Phi™
    processor codenamed Knights Landing.
-   Added experimental support for compressed BVH nodes (reduces node
    size to 56-62% of uncompressed size). Compression introduces a
    typical performance overhead of ~10%.
-   Bugfix in backface culling mode. We do now properly cull the
    backfaces and not the frontfaces.
-   Feature freeze for the first generation Intel® Xeon Phi™ coprocessor
    codenamed Knights Corner. We will still maintain and add bug fixes
    to Embree v2.9.0, but Embree 2.10 and future versions will no longer
    support it.

### New Features in Embree 2.9.0

-   Improved shadow ray performance (10-100% depending on the scene).
-   Added initial support for ray streams (10-30% higher performance
    depending on ray coherence in the stream).
-   Added support to calculate second order derivatives using the
    `rtcInterpolate2` function.
-   Changed the parametrization for triangular subdivision faces to
    the same scheme used for pentagons.
-   Added support to query the Embree configuration using the
    `rtcDeviceGetParameter` function.

### New Features in Embree 2.8.1

-   Added support for setting per geometry tessellation rate (supported
    for subdivision and Bézier geometries).
-   Added support for motion blurred instances.

### New Features in Embree 2.8.0

-   Added support for line segment geometry.
-   Added support for quad geometry (replaces triangle-pairs feature).
-   Added support for linear motion blur of user geometries.
-   Improved performance through AVX-512 optimizations.
-   Improved performance of lazy scene build (when using TBB 4.4 update
    2).
-   Improved performance through huge page support under linux.

### New Features in Embree 2.7.1

-   Internal tasking system supports cancellation of build operations.
-   ISPC mode for robust and compact scenes got significantly faster
    (implemented hybrid traversal for bvh4.triangle4v and
    bvh4.triangle4i).
-   Hair rendering got faster as we fixed some issues with the SAH
    heuristic cost factors.
-   BVH8 got slight faster for single ray traversal (improved sorting
    when hitting more than 4 boxes).
-   BVH build performance got up to 30% faster on CPUs with high core
    counts (improved parallel partition code).
-   High quality build mode again working properly (spatial splits had
    been deactivated in v2.7.0 due to some bug).
-   Support for merging two adjacent triangles sharing a common edge
    into a triangle-pair primitive (can reduce memory consumption and
    BVH build times by up to 50% for mostly quad-based input meshes).
-   Internal cleanups (reduced number of traversal kernels by more
    templating).
-   Reduced stack size requirements of BVH builders.
-   Fixed crash for dynamic scenes, triggered by deleting all
    geometries from the scene.

### New Features in Embree 2.7.0

-   Added device concept to Embree to allow different components of an
    application to use Embree without interfering with each other.
-   Fixed memory leak in twolevel builder used for dynamic scenes.
-   Fixed bug in tesselation cache that caused crashes for subdivision
    surfaces.
-   Fixed bug in internal task scheduler that caused deadlocks when
    using `rtcCommitThread`.
-   Improved hit-distance accuracy for thin triangles in robust mode.
-   Added support to disable ray packet support in cmake.

### New Features in Embree 2.6.2

-   Fixed bug triggered by instantiating motion blur geometry.
-   Fixed bug in hit UV coordinates of static subdivision geometries.
-   Performance improvements when only changing tessellation levels for
    subdivision geometry per frame.
-   Added ray packet intersectors for subdivision geometry, resulting in
    improved performance for coherent rays.
-   Reduced virtual address space usage for static geometries.
-   Fixed some AVX2 code paths when compiling with GCC or Clang.
-   Bugfix for subdiv patches with non-matching winding order.
-   Bugfix in ISA detection of AVX-512.

### New Features in Embree 2.6.1

-   Major performance improvements for ray tracing subdivision surfaces,
    e.g. up to 2× faster for scenes where only the tessellation levels
    are changing per frame, and up to 3× faster for scenes with lots of
    crease features
-   Initial support for architectures supporting the new 16-wide AVX-512
    ISA
-   Implemented intersection filter callback support for subdivision
    surfaces
-   Added `RTC_IGNORE_INVALID_RAYS` CMake option which makes the ray
    intersectors more robust against full tree traversal caused by
    invalid ray inputs (e.g. INF, NaN, etc)

### New Features in Embree 2.6.0

-   Added `rtcInterpolate` function to interpolate per vertex
    attributes
-   Added `rtcSetBoundaryMode` function that can be used to select the
    boundary handling for subdivision surfaces
-   Fixed a traversal bug that caused rays with very small ray
    direction components to miss geometry
-   Performance improvements for the robust traversal mode
-   Fixed deadlock when calling `rtcCommit` from multiple
    threads on same scene

### New Features in Embree 2.5.1

-   On dual socket workstations, the initial BVH build performance
    almost doubled through a better memory allocation scheme
-   Reduced memory usage for subdivision surface objects with crease
    features
-   `rtcCommit` performance is robust against unset "flush to zero" and
    "denormals are zero" flags. However, enabling these flags in your
    application is still recommended
-   Reduced memory usage for subdivision surfaces with borders and
    infinitely sharp creases
-   Lots of internal cleanups and bug fixes for both Intel® Xeon® and
    Intel® Xeon Phi™

### New Features in Embree 2.5.0

-   Improved hierarchy build performance on both Intel Xeon and Intel
    Xeon Phi
-   Vastly improved tessellation cache for ray tracing subdivision
    surfaces
-   Added `rtcGetUserData` API call to query per geometry user pointer
    set through `rtcSetUserData`
-   Added support for memory monitor callback functions to track and
    limit memory consumption
-   Added support for progress monitor callback functions to track build
    progress and cancel long build operations
-   BVH builders can be used to build user defined hierarchies inside
    the application (see tutorial [BVH Builder])
-   Switched to TBB as default tasking system on Xeon to get even faster
    hierarchy build times and better integration for applications that
    also use TBB
-   `rtcCommit` can get called from multiple TBB threads to join the
    hierarchy build operations

### New Features in Embree 2.4

-   Support for Catmull Clark subdivision surfaces (triangle/quad base
    primitives)
-   Support for vector displacements on Catmull Clark subdivision
    surfaces
-   Various bug fixes (e.g. 4-byte alignment of vertex buffers works)

### New Features in Embree 2.3.3

-   BVH builders more robustly handle invalid input data (Intel Xeon
    processor family)
-   Motion blur support for hair geometry (Xeon)
-   Improved motion blur performance for triangle geometry (Xeon)
-   Improved robust ray tracing mode (Xeon)
-   Added `rtcCommitThread` API call for easier integration into
    existing tasking systems (Xeon and Intel Xeon Phi coprocessor)
-   Added support for recording and replaying all
    `rtcIntersect`/`rtcOccluded` calls (Xeon and Xeon Phi)

### New Features in Embree 2.3.2

-   Improved mixed AABB/OBB-BVH for hair geometry (Xeon Phi)
-   Reduced amount of pre-allocated memory for BVH builders (Xeon Phi)
-   New 64-bit Morton code-based BVH builder (Xeon Phi)
-   (Enhanced) Morton code-based BVH builders use now tree rotations to
    improve BVH quality (Xeon Phi)
-   Bug fixes (Xeon and Xeon Phi)

### New Features in Embree 2.3.1

-   High quality BVH mode improves spatial splits which result in up to
    30% performance improvement for some scenes (Xeon)
-   Compile time enabled intersection filter functions do not reduce
    performance if no intersection filter is used in the scene (Xeon and
    Xeon Phi)
-   Improved ray tracing performance for hair geometry by \>20% on Xeon
    Phi. BVH for hair geometry requires 20% less memory
-   BVH8 for AVX/AVX2 targets improves performance for single ray
    tracing on Haswell by up to 12% and by up to 5% for hybrid (Xeon)
-   Memory conservative BVH for Xeon Phi now uses BVH node quantization
    to lower memory footprint (requires half the memory footprint of the
    default BVH)

### New Features in Embree 2.3

-   Support for ray tracing hair geometry (Xeon and Xeon Phi)
-   Catching errors through error callback function
-   Faster hybrid traversal (Xeon and Xeon Phi)
-   New memory conservative BVH for Xeon Phi
-   Faster Morton code-based builder on Xeon
-   Faster binned-SAH builder on Xeon Phi
-   Lots of code cleanups/simplifications/improvements (Xeon and Xeon
    Phi)

### New Features in Embree 2.2

-   Support for motion blur on Xeon Phi
-   Support for intersection filter callback functions
-   Support for buffer sharing with the application
-   Lots of AVX2 optimizations, e.g. \~20% faster 8-wide hybrid
    traversal
-   Experimental support for 8-wide (AVX/AVX2) and 16-wide BVHs (Xeon
    Phi)

### New Features in Embree 2.1

-   New future proof API with a strong focus on supporting dynamic
    scenes
-   Lots of optimizations for 8-wide AVX2 (Haswell architecture)
-   Automatic runtime code selection for SSE, AVX, and AVX2
-   Support for user-defined geometry
-   New and improved BVH builders:
    -   Fast adaptive Morton code-based builder (without SAH-based
        top-level rebuild)
    -   Both the SAH and Morton code-based builders got faster (Xeon
        Phi)
    -   New variant of the SAH-based builder using triangle pre-splits
        (Xeon Phi)

### New Features in Embree 2.0

-   Support for the Intel® Xeon Phi™ coprocessor platform
-   Support for high-performance "packet" kernels on SSE, AVX, and Xeon
    Phi
-   Integration with the Intel® SPMD Program Compiler (ISPC)
-   Instantiation and fast BVH reconstruction
-   Example photo-realistic rendering engine for both C++ and ISPC

