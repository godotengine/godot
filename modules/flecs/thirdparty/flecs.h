// Comment out this line when using as DLL
#define flecs_STATIC
/**
 * @file flecs.h
 * @brief Flecs public API.
 *
 * This file contains the public API for Flecs.
 */

#ifndef FLECS_H
#define FLECS_H

/**
 * @defgroup c C API
 *
 * @{
 * @}
 */

/**
 * @defgroup core Core
 * @ingroup c
 * Core ECS functionality (entities, storage, queries).
 *
 * @{
 */

/**
 * @defgroup options API defines
 * Defines for customizing compile time features.
 *
 * @{
 */

/* Flecs version macros */
#define FLECS_VERSION_MAJOR 4  /**< Flecs major version. */
#define FLECS_VERSION_MINOR 0  /**< Flecs minor version. */
#define FLECS_VERSION_PATCH 4  /**< Flecs patch version. */

/** Flecs version. */
#define FLECS_VERSION FLECS_VERSION_IMPL(\
    FLECS_VERSION_MAJOR, FLECS_VERSION_MINOR, FLECS_VERSION_PATCH)

/** @def FLECS_CONFIG_HEADER
 * Allows for including a user-customizable header that specifies compile-time 
 * features. */
#ifdef FLECS_CONFIG_HEADER
#include "flecs_config.h"
#endif

/** @def ecs_float_t
 * Customizable precision for floating point operations */
#ifndef ecs_float_t
#define ecs_float_t float
#endif

/** @def ecs_ftime_t
 * Customizable precision for scalar time values. Change to double precision for
 * processes that can run for a long time (e.g. longer than a day). */
#ifndef ecs_ftime_t
#define ecs_ftime_t ecs_float_t
#endif

/** @def FLECS_LEGACY
 * Define when building for C89
 */
// #define FLECS_LEGACY

/** @def FLECS_ACCURATE_COUNTERS
 * Define to ensure that global counters used for statistics (such as the
 * allocation counters in the OS API) are accurate in multithreaded
 * applications, at the cost of increased overhead.
 */
// #define FLECS_ACCURATE_COUNTERS

/** @def FLECS_DISABLE_COUNTERS
 * Disables counters used for statistics. Improves performance, but
 * will prevent some features that rely on statistics from working,
 * like the statistics pages in the explorer.
 */
// #define FLECS_DISABLE_COUNTERS

/* Make sure provided configuration is valid */
#if defined(FLECS_DEBUG) && defined(FLECS_NDEBUG)
#error "invalid configuration: cannot both define FLECS_DEBUG and FLECS_NDEBUG"
#endif
#if defined(FLECS_DEBUG) && defined(NDEBUG)
#error "invalid configuration: cannot both define FLECS_DEBUG and NDEBUG"
#endif

/** @def FLECS_DEBUG
 * Used for input parameter checking and cheap sanity checks. There are lots of
 * asserts in every part of the code, so this will slow down applications.
 */
#if !defined(FLECS_DEBUG) && !defined(FLECS_NDEBUG)
#if defined(NDEBUG)
#define FLECS_NDEBUG
#else
#define FLECS_DEBUG
#endif
#endif

/** @def FLECS_SANITIZE
 * Enables expensive checks that can detect issues early. Recommended for
 * running tests or when debugging issues. This will severely slow down code.
 */
#ifdef FLECS_SANITIZE
#ifndef FLECS_DEBUG
#define FLECS_DEBUG /* If sanitized mode is enabled, so is debug mode */
#endif
#endif

/* Tip: if you see weird behavior that you think might be a bug, make sure to
 * test with the FLECS_DEBUG or FLECS_SANITIZE flags enabled. There's a good
 * chance that this gives you more information about the issue! */

/** @def FLECS_SOFT_ASSERT
 * Define to not abort for recoverable errors, like invalid parameters. An error
 * is still thrown to the console. This is recommended for when running inside a
 * third party runtime, such as the Unreal editor.
 *
 * Note that internal sanity checks (ECS_INTERNAL_ERROR) will still abort a
 * process, as this gives more information than a (likely) subsequent crash.
 *
 * When a soft assert occurs, the code will attempt to minimize the number of
 * side effects of the failed operation, but this may not always be possible.
 * Even though an application may still be able to continue running after a soft
 * assert, it should be treated as if in an undefined state.
 */
// #define FLECS_SOFT_ASSERT

/** @def FLECS_KEEP_ASSERT
 * By default asserts are disabled in release mode, when either FLECS_NDEBUG or
 * NDEBUG is defined. Defining FLECS_KEEP_ASSERT ensures that asserts are not
 * disabled. This define can be combined with FLECS_SOFT_ASSERT.
 */
// #define FLECS_KEEP_ASSERT

/** @def FLECS_CPP_NO_AUTO_REGISTRATION
 * When set, the C++ API will require that components are registered before they
 * are used. This is useful in multithreaded applications, where components need
 * to be registered beforehand, and to catch issues in projects where component 
 * registration is mandatory. Disabling automatic component registration also
 * slightly improves performance.
 * The C API is not affected by this feature.
 */
// #define FLECS_CPP_NO_AUTO_REGISTRATION

/** @def FLECS_CUSTOM_BUILD
 * This macro lets you customize which addons to build flecs with.
 * Without any addons Flecs is just a minimal ECS storage, but addons add
 * features such as systems, scheduling and reflection. If an addon is disabled,
 * it is excluded from the build, so that it consumes no resources. By default
 * all addons are enabled.
 *
 * You can customize a build by either whitelisting or blacklisting addons. To
 * whitelist addons, first define the FLECS_CUSTOM_BUILD macro, which disables
 * all addons. You can then manually select the addons you need by defining
 * their macro, like "FLECS_SYSTEM".
 *
 * To blacklist an addon, make sure to *not* define FLECS_CUSTOM_BUILD, and
 * instead define the addons you don't need by defining FLECS_NO_<addon>, for
 * example "FLECS_NO_SYSTEM". If there are any addons that depend on the
 * blacklisted addon, an error will be thrown during the build.
 *
 * Note that addons can have dependencies on each other. Addons will
 * automatically enable their dependencies. To see the list of addons that was
 * compiled in a build, enable tracing before creating the world by doing:
 *
 * @code
 * ecs_log_set_level(0);
 * @endcode
 *
 * which outputs the full list of addons Flecs was compiled with.
 */
// #define FLECS_CUSTOM_BUILD

#ifndef FLECS_CUSTOM_BUILD
#define FLECS_ALERTS         /**< Monitor conditions for errors */
#define FLECS_APP            /**< Application addon */
// #define FLECS_C           /**< C API convenience macros, always enabled */
#define FLECS_CPP            /**< C++ API */
#define FLECS_DOC            /**< Document entities & components */
// #define FLECS_JOURNAL     /**< Journaling addon (disabled by default) */
#define FLECS_JSON           /**< Parsing JSON to/from component values */
#define FLECS_HTTP           /**< Tiny HTTP server for connecting to remote UI */
#define FLECS_LOG            /**< When enabled ECS provides more detailed logs */
#define FLECS_META           /**< Reflection support */
#define FLECS_METRICS        /**< Expose component data as statistics */
#define FLECS_MODULE         /**< Module support */
#define FLECS_OS_API_IMPL    /**< Default implementation for OS API */
// #define FLECS_PERF_TRACE  /**< Enable performance tracing (disabled by default) */
#define FLECS_PIPELINE       /**< Pipeline support */
#define FLECS_REST           /**< REST API for querying application data */
#define FLECS_SCRIPT         /**< Flecs entity notation language */
// #define FLECS_SCRIPT_MATH /**< Math functions for flecs script (may require linking with libm) */
#define FLECS_SYSTEM         /**< System support */
#define FLECS_STATS          /**< Track runtime statistics */
#define FLECS_TIMER          /**< Timer support */
#define FLECS_UNITS          /**< Builtin standard units */
#endif // ifndef FLECS_CUSTOM_BUILD

/** @def FLECS_LOW_FOOTPRINT
 * Set a number of constants to values that decrease memory footprint, at the
 * cost of decreased performance. */
// #define FLECS_LOW_FOOTPRINT
#ifdef FLECS_LOW_FOOTPRINT
#define FLECS_HI_COMPONENT_ID (16)
#define FLECS_HI_ID_RECORD_ID (16)
#define FLECS_SPARSE_PAGE_BITS (4)
#define FLECS_ENTITY_PAGE_BITS (6)
#define FLECS_USE_OS_ALLOC
#endif

/** @def FLECS_HI_COMPONENT_ID
 * This constant can be used to balance between performance and memory
 * utilization. The constant is used in two ways:
 * - Entity ids 0..FLECS_HI_COMPONENT_ID are reserved for component ids.
 * - Used as lookup array size in table edges.
 *
 * Increasing this value increases the size of the lookup array, which allows
 * fast table traversal, which improves performance of ECS add/remove
 * operations. Component ids that fall outside of this range use a regular map
 * lookup, which is slower but more memory efficient. */
#ifndef FLECS_HI_COMPONENT_ID
#define FLECS_HI_COMPONENT_ID (256)
#endif

/** @def FLECS_HI_ID_RECORD_ID
 * This constant can be used to balance between performance and memory
 * utilization. The constant is used to determine the size of the id record
 * lookup array. Id values that fall outside of this range use a regular map
 * lookup, which is slower but more memory efficient.
 */
#ifndef FLECS_HI_ID_RECORD_ID
#define FLECS_HI_ID_RECORD_ID (1024)
#endif

/** @def FLECS_SPARSE_PAGE_BITS
 * This constant is used to determine the number of bits of an id that is used
 * to determine the page index when used with a sparse set. The number of bits
 * determines the page size, which is (1 << bits).
 * Lower values decrease memory utilization, at the cost of more allocations. */
#ifndef FLECS_SPARSE_PAGE_BITS
#define FLECS_SPARSE_PAGE_BITS (6)
#endif

/** @def FLECS_ENTITY_PAGE_BITS
 * Same as FLECS_SPARSE_PAGE_BITS, but for the entity index. */
#ifndef FLECS_ENTITY_PAGE_BITS
#define FLECS_ENTITY_PAGE_BITS (12)
#endif

/** @def FLECS_USE_OS_ALLOC
 * When enabled, Flecs will use the OS allocator provided in the OS API directly
 * instead of the builtin block allocator. This can decrease memory utilization
 * as memory will be freed more often, at the cost of decreased performance. */
// #define FLECS_USE_OS_ALLOC

/** @def FLECS_ID_DESC_MAX
 * Maximum number of ids to add ecs_entity_desc_t / ecs_bulk_desc_t */
#ifndef FLECS_ID_DESC_MAX
#define FLECS_ID_DESC_MAX (32)
#endif

/** @def FLECS_EVENT_DESC_MAX
 * Maximum number of events in ecs_observer_desc_t */
#ifndef FLECS_EVENT_DESC_MAX
#define FLECS_EVENT_DESC_MAX (8)
#endif

/** @def FLECS_VARIABLE_COUNT_MAX
 * Maximum number of query variables per query */
#define FLECS_VARIABLE_COUNT_MAX (64)

/** @def FLECS_TERM_COUNT_MAX 
 * Maximum number of terms in queries. Should not exceed 64. */
#ifndef FLECS_TERM_COUNT_MAX
#define FLECS_TERM_COUNT_MAX 32
#endif

/** @def FLECS_TERM_ARG_COUNT_MAX 
 * Maximum number of arguments for a term. */
#ifndef FLECS_TERM_ARG_COUNT_MAX
#define FLECS_TERM_ARG_COUNT_MAX (16)
#endif

/** @def FLECS_QUERY_VARIABLE_COUNT_MAX
 * Maximum number of query variables per query. Should not exceed 128. */
#ifndef FLECS_QUERY_VARIABLE_COUNT_MAX
#define FLECS_QUERY_VARIABLE_COUNT_MAX (64)
#endif

/** @def FLECS_QUERY_SCOPE_NESTING_MAX
 * Maximum nesting depth of query scopes */
#ifndef FLECS_QUERY_SCOPE_NESTING_MAX
#define FLECS_QUERY_SCOPE_NESTING_MAX (8)
#endif

/** @def FLECS_DAG_DEPTH_MAX
 * Maximum of levels in a DAG (acyclic relationship graph). If a graph with a
 * depth larger than this is encountered, a CYCLE_DETECTED panic is thrown.
 */
#ifndef FLECS_DAG_DEPTH_MAX
#define FLECS_DAG_DEPTH_MAX (128)
#endif

/** @} */

/**
 * @file api_defines.h
 * @brief Supporting defines for the public API.
 *
 * This file contains constants / macros that are typically not used by an
 * application but support the public API, and therefore must be exposed. This
 * header should not be included by itself.
 */

#ifndef FLECS_API_DEFINES_H
#define FLECS_API_DEFINES_H

/**
 * @file api_flags.h
 * @brief Bitset flags used by internals.
 */

#ifndef FLECS_API_FLAGS_H
#define FLECS_API_FLAGS_H

#ifdef __cplusplus
extern "C" {
#endif


////////////////////////////////////////////////////////////////////////////////
//// World flags
////////////////////////////////////////////////////////////////////////////////

#define EcsWorldQuitWorkers           (1u << 0)
#define EcsWorldReadonly              (1u << 1)
#define EcsWorldInit                  (1u << 2)
#define EcsWorldQuit                  (1u << 3)
#define EcsWorldFini                  (1u << 4)
#define EcsWorldMeasureFrameTime      (1u << 5)
#define EcsWorldMeasureSystemTime     (1u << 6)
#define EcsWorldMultiThreaded         (1u << 7)
#define EcsWorldFrameInProgress       (1u << 8)

////////////////////////////////////////////////////////////////////////////////
//// OS API flags
////////////////////////////////////////////////////////////////////////////////

#define EcsOsApiHighResolutionTimer   (1u << 0)
#define EcsOsApiLogWithColors         (1u << 1)
#define EcsOsApiLogWithTimeStamp      (1u << 2)
#define EcsOsApiLogWithTimeDelta      (1u << 3)


////////////////////////////////////////////////////////////////////////////////
//// Entity flags (set in upper bits of ecs_record_t::row)
////////////////////////////////////////////////////////////////////////////////

#define EcsEntityIsId                 (1u << 31)
#define EcsEntityIsTarget             (1u << 30)
#define EcsEntityIsTraversable        (1u << 29)


////////////////////////////////////////////////////////////////////////////////
//// Id flags (used by ecs_id_record_t::flags)
////////////////////////////////////////////////////////////////////////////////

#define EcsIdOnDeleteRemove            (1u << 0)
#define EcsIdOnDeleteDelete            (1u << 1)
#define EcsIdOnDeletePanic             (1u << 2)
#define EcsIdOnDeleteMask\
    (EcsIdOnDeletePanic|EcsIdOnDeleteRemove|EcsIdOnDeleteDelete)

#define EcsIdOnDeleteObjectRemove      (1u << 3)
#define EcsIdOnDeleteObjectDelete      (1u << 4)
#define EcsIdOnDeleteObjectPanic       (1u << 5)
#define EcsIdOnDeleteObjectMask\
    (EcsIdOnDeleteObjectPanic|EcsIdOnDeleteObjectRemove|\
        EcsIdOnDeleteObjectDelete)

#define EcsIdOnInstantiateOverride     (1u << 6)
#define EcsIdOnInstantiateInherit      (1u << 7)
#define EcsIdOnInstantiateDontInherit  (1u << 8)
#define EcsIdOnInstantiateMask\
    (EcsIdOnInstantiateOverride|EcsIdOnInstantiateInherit|\
        EcsIdOnInstantiateDontInherit)

#define EcsIdExclusive                 (1u << 9)
#define EcsIdTraversable               (1u << 10)
#define EcsIdTag                       (1u << 11)
#define EcsIdWith                      (1u << 12)
#define EcsIdCanToggle                 (1u << 13)
#define EcsIdIsTransitive              (1u << 14)

#define EcsIdHasOnAdd                  (1u << 16) /* Same values as table flags */
#define EcsIdHasOnRemove               (1u << 17) 
#define EcsIdHasOnSet                  (1u << 18)
#define EcsIdHasOnTableFill            (1u << 19)
#define EcsIdHasOnTableEmpty           (1u << 20)
#define EcsIdHasOnTableCreate          (1u << 21)
#define EcsIdHasOnTableDelete          (1u << 22)
#define EcsIdIsSparse                  (1u << 23)
#define EcsIdIsUnion                   (1u << 24)
#define EcsIdEventMask\
    (EcsIdHasOnAdd|EcsIdHasOnRemove|EcsIdHasOnSet|\
        EcsIdHasOnTableFill|EcsIdHasOnTableEmpty|EcsIdHasOnTableCreate|\
            EcsIdHasOnTableDelete|EcsIdIsSparse|EcsIdIsUnion)

#define EcsIdMarkedForDelete           (1u << 30)

/* Utilities for converting from flags to delete policies and vice versa */
#define ECS_ID_ON_DELETE(flags) \
    ((ecs_entity_t[]){0, EcsRemove, EcsDelete, 0, EcsPanic}\
        [((flags) & EcsIdOnDeleteMask)])
#define ECS_ID_ON_DELETE_TARGET(flags) ECS_ID_ON_DELETE(flags >> 3)
#define ECS_ID_ON_DELETE_FLAG(id) (1u << ((id) - EcsRemove))
#define ECS_ID_ON_DELETE_TARGET_FLAG(id) (1u << (3 + ((id) - EcsRemove)))

/* Utilities for converting from flags to instantiate policies and vice versa */
#define ECS_ID_ON_INSTANTIATE(flags) \
    ((ecs_entity_t[]){EcsOverride, EcsOverride, EcsInherit, 0, EcsDontInherit}\
        [(((flags) & EcsIdOnInstantiateMask) >> 6)])
#define ECS_ID_ON_INSTANTIATE_FLAG(id) (1u << (6 + ((id) - EcsOverride)))


////////////////////////////////////////////////////////////////////////////////
//// Iterator flags (used by ecs_iter_t::flags)
////////////////////////////////////////////////////////////////////////////////

#define EcsIterIsValid                 (1u << 0u)  /* Does iterator contain valid result */
#define EcsIterNoData                  (1u << 1u)  /* Does iterator provide (component) data */
#define EcsIterNoResults               (1u << 3u)  /* Iterator has no results */
#define EcsIterIgnoreThis              (1u << 4u)  /* Only evaluate non-this terms */
#define EcsIterHasCondSet              (1u << 6u)  /* Does iterator have conditionally set fields */
#define EcsIterProfile                 (1u << 7u)  /* Profile iterator performance */
#define EcsIterTrivialSearch           (1u << 8u)  /* Trivial iterator mode */
#define EcsIterTrivialTest             (1u << 11u) /* Trivial test mode (constrained $this) */
#define EcsIterTrivialCached           (1u << 14u) /* Trivial search for cached query */
#define EcsIterCacheSearch             (1u << 15u) /* Cache search */
#define EcsIterFixedInChangeComputed   (1u << 16u) /* Change detection for fixed in terms is done */
#define EcsIterFixedInChanged          (1u << 17u) /* Fixed in terms changed */
#define EcsIterSkip                    (1u << 18u) /* Result was skipped for change detection */
#define EcsIterCppEach                 (1u << 19u) /* Uses C++ 'each' iterator */

/* Same as event flags */
#define EcsIterTableOnly               (1u << 20u)  /* Result only populates table */


////////////////////////////////////////////////////////////////////////////////
//// Event flags (used by ecs_event_decs_t::flags)
////////////////////////////////////////////////////////////////////////////////

#define EcsEventTableOnly              (1u << 20u) /* Table event (no data, same as iter flags) */
#define EcsEventNoOnSet                (1u << 16u) /* Don't emit OnSet for inherited ids */


////////////////////////////////////////////////////////////////////////////////
//// Query flags (used by ecs_query_t::flags)
////////////////////////////////////////////////////////////////////////////////

/* Flags that can only be set by the query implementation */
#define EcsQueryMatchThis             (1u << 11u) /* Query has terms with $this source */
#define EcsQueryMatchOnlyThis         (1u << 12u) /* Query only has terms with $this source */
#define EcsQueryMatchOnlySelf         (1u << 13u) /* Query has no terms with up traversal */
#define EcsQueryMatchWildcards        (1u << 14u) /* Query matches wildcards */
#define EcsQueryMatchNothing          (1u << 15u) /* Query matches nothing */
#define EcsQueryHasCondSet            (1u << 16u) /* Query has conditionally set fields */
#define EcsQueryHasPred               (1u << 17u) /* Query has equality predicates */
#define EcsQueryHasScopes             (1u << 18u) /* Query has query scopes */
#define EcsQueryHasRefs               (1u << 19u) /* Query has terms with static source */
#define EcsQueryHasOutTerms           (1u << 20u) /* Query has [out] terms */
#define EcsQueryHasNonThisOutTerms    (1u << 21u) /* Query has [out] terms with no $this source */
#define EcsQueryHasMonitor            (1u << 22u) /* Query has monitor for change detection */
#define EcsQueryIsTrivial             (1u << 23u) /* Query can use trivial evaluation function */
#define EcsQueryHasCacheable          (1u << 24u) /* Query has cacheable terms */
#define EcsQueryIsCacheable           (1u << 25u) /* All terms of query are cacheable */
#define EcsQueryHasTableThisVar       (1u << 26u) /* Does query have $this table var */
#define EcsQueryCacheYieldEmptyTables (1u << 27u) /* Does query cache empty tables */
#define EcsQueryNested                (1u << 28u) /* Query created by a query (for observer, cache) */

////////////////////////////////////////////////////////////////////////////////
//// Term flags (used by ecs_term_t::flags_)
////////////////////////////////////////////////////////////////////////////////

#define EcsTermMatchAny               (1u << 0)
#define EcsTermMatchAnySrc            (1u << 1)
#define EcsTermTransitive             (1u << 2)
#define EcsTermReflexive              (1u << 3)
#define EcsTermIdInherited            (1u << 4)
#define EcsTermIsTrivial              (1u << 5)
#define EcsTermIsCacheable            (1u << 7)
#define EcsTermIsScope                (1u << 8)
#define EcsTermIsMember               (1u << 9)
#define EcsTermIsToggle               (1u << 10)
#define EcsTermKeepAlive              (1u << 11)
#define EcsTermIsSparse               (1u << 12)
#define EcsTermIsUnion                (1u << 13)
#define EcsTermIsOr                   (1u << 14)


////////////////////////////////////////////////////////////////////////////////
//// Observer flags (used by ecs_observer_t::flags)
////////////////////////////////////////////////////////////////////////////////

#define EcsObserverIsMulti             (1u << 1u)  /* Does observer have multiple terms */
#define EcsObserverIsMonitor           (1u << 2u)  /* Is observer a monitor */
#define EcsObserverIsDisabled          (1u << 3u)  /* Is observer entity disabled */
#define EcsObserverIsParentDisabled    (1u << 4u)  /* Is module parent of observer disabled  */
#define EcsObserverBypassQuery         (1u << 5u)  /* Don't evaluate query for multi-component observer*/
#define EcsObserverYieldOnCreate       (1u << 6u)  /* Yield matching entities when creating observer */
#define EcsObserverYieldOnDelete       (1u << 7u)  /* Yield matching entities when deleting observer */


////////////////////////////////////////////////////////////////////////////////
//// Table flags (used by ecs_table_t::flags)
////////////////////////////////////////////////////////////////////////////////

#define EcsTableHasBuiltins            (1u << 1u)  /* Does table have builtin components */
#define EcsTableIsPrefab               (1u << 2u)  /* Does the table store prefabs */
#define EcsTableHasIsA                 (1u << 3u)  /* Does the table have IsA relationship */
#define EcsTableHasChildOf             (1u << 4u)  /* Does the table type ChildOf relationship */
#define EcsTableHasName                (1u << 5u)  /* Does the table type have (Identifier, Name) */
#define EcsTableHasPairs               (1u << 6u)  /* Does the table type have pairs */
#define EcsTableHasModule              (1u << 7u)  /* Does the table have module data */
#define EcsTableIsDisabled             (1u << 8u)  /* Does the table type has EcsDisabled */
#define EcsTableNotQueryable           (1u << 9u)  /* Table should never be returned by queries */
#define EcsTableHasCtors               (1u << 10u)
#define EcsTableHasDtors               (1u << 11u)
#define EcsTableHasCopy                (1u << 12u)
#define EcsTableHasMove                (1u << 13u)
#define EcsTableHasToggle              (1u << 14u)
#define EcsTableHasOverrides           (1u << 15u)

#define EcsTableHasOnAdd               (1u << 16u) /* Same values as id flags */
#define EcsTableHasOnRemove            (1u << 17u)
#define EcsTableHasOnSet               (1u << 18u)
#define EcsTableHasOnTableFill         (1u << 19u)
#define EcsTableHasOnTableEmpty        (1u << 20u)
#define EcsTableHasOnTableCreate       (1u << 21u)
#define EcsTableHasOnTableDelete       (1u << 22u)
#define EcsTableHasSparse              (1u << 23u)
#define EcsTableHasUnion               (1u << 24u)

#define EcsTableHasTraversable         (1u << 26u)
#define EcsTableMarkedForDelete        (1u << 30u)

/* Composite table flags */
#define EcsTableHasLifecycle     (EcsTableHasCtors | EcsTableHasDtors)
#define EcsTableIsComplex        (EcsTableHasLifecycle | EcsTableHasToggle | EcsTableHasSparse)
#define EcsTableHasAddActions    (EcsTableHasIsA | EcsTableHasCtors | EcsTableHasOnAdd | EcsTableHasOnSet)
#define EcsTableHasRemoveActions (EcsTableHasIsA | EcsTableHasDtors | EcsTableHasOnRemove)
#define EcsTableEdgeFlags        (EcsTableHasOnAdd | EcsTableHasOnRemove | EcsTableHasSparse | EcsTableHasUnion)
#define EcsTableAddEdgeFlags     (EcsTableHasOnAdd | EcsTableHasSparse | EcsTableHasUnion)
#define EcsTableRemoveEdgeFlags  (EcsTableHasOnRemove | EcsTableHasSparse | EcsTableHasUnion)

////////////////////////////////////////////////////////////////////////////////
//// Aperiodic action flags (used by ecs_run_aperiodic)
////////////////////////////////////////////////////////////////////////////////

#define EcsAperiodicComponentMonitors  (1u << 2u)  /* Process component monitors */
#define EcsAperiodicEmptyQueries       (1u << 4u)  /* Process empty queries */

#ifdef __cplusplus
}
#endif

#endif


#if defined(_WIN32) || defined(_MSC_VER)
#define ECS_TARGET_WINDOWS
#elif defined(__ANDROID__)
#define ECS_TARGET_ANDROID
#define ECS_TARGET_POSIX
#elif defined(__linux__)
#define ECS_TARGET_LINUX
#define ECS_TARGET_POSIX
#elif defined(__FreeBSD__)
#define ECS_TARGET_FREEBSD
#define ECS_TARGET_POSIX
#elif defined(__APPLE__) && defined(__MACH__)
#define ECS_TARGET_DARWIN
#define ECS_TARGET_POSIX
#elif defined(__EMSCRIPTEN__)
#define ECS_TARGET_EM
#define ECS_TARGET_POSIX
#endif

#if defined(__MINGW32__) || defined(__MINGW64__)
#define ECS_TARGET_MINGW
#endif

#if defined(_MSC_VER)
#ifndef __clang__
#define ECS_TARGET_MSVC
#endif
#endif

#if defined(__clang__)
#define ECS_TARGET_CLANG
#endif

#if defined(__GNUC__)
#define ECS_TARGET_GNU
#endif

/* Map between clang and apple clang versions, as version 13 has a difference in
 * the format of __PRETTY_FUNCTION__ which enum reflection depends on. */
#if defined(__clang__)
    #if defined(__APPLE__)
        #if __clang_major__ == 13
            #if __clang_minor__ < 1
                #define ECS_CLANG_VERSION 12
            #else
                #define ECS_CLANG_VERSION 13
            #endif
        #else
            #define ECS_CLANG_VERSION __clang_major__
        #endif
    #else
        #define ECS_CLANG_VERSION __clang_major__
    #endif
#endif

/* Define noreturn attribute only for GCC or Clang. */
#if defined(ECS_TARGET_GNU) || defined(ECS_TARGET_CLANG)
    #define ECS_NORETURN __attribute__((noreturn))
#else
    #define ECS_NORETURN
#endif

/* Ignored warnings */
#if defined(ECS_TARGET_CLANG)
/* Ignore unknown options so we don't have to care about the compiler version */
#pragma clang diagnostic ignored "-Wunknown-warning-option"
/* Warns for double or redundant semicolons. There are legitimate cases where a
 * semicolon after an empty statement is useful, for example after a macro that
 * is replaced with a code block. With this warning enabled, semicolons would 
 * only have to be added after macro's that are not code blocks, which in some
 * cases isn't possible as the implementation of a macro can be different in
 * debug/release mode. */
#pragma clang diagnostic ignored "-Wextra-semi-stmt"
/* This is valid in C99, and Flecs must be compiled as C99. */
#pragma clang diagnostic ignored "-Wdeclaration-after-statement"
/* Clang attribute to detect fallthrough isn't supported on older versions. 
 * Implicit fallthrough is still detected by gcc and ignored with "fall through"
 * comments */
#pragma clang diagnostic ignored "-Wimplicit-fallthrough"
/* This warning prevents adding a default case when all enum constants are part
 * of the switch. In C however an enum type can assume any value in the range of
 * the type, and this warning makes it harder to catch invalid enum values. */
#pragma clang diagnostic ignored "-Wcovered-switch-default"
/* This warning prevents some casts of function results to a different kind of
 * type, e.g. casting an int result to double. Not very useful in practice, as
 * it just forces the code to assign to a variable first, then cast. */
#pragma clang diagnostic ignored "-Wbad-function-cast"
/* Format strings can be passed down from other functions. */
#pragma clang diagnostic ignored "-Wformat-nonliteral"
/* Useful, but not reliable enough. It can incorrectly flag macro's as unused
 * in standalone builds. */
#pragma clang diagnostic ignored "-Wunused-macros"
/* This warning gets thrown by clang even when a code is handling all case
 * values but not all cases (for example, when the switch contains a LastValue
 * case). Adding a "default" case fixes the warning, but silences future 
 * warnings about unhandled cases, which is worse. */
#pragma clang diagnostic ignored "-Wswitch-default"
#if __clang_major__ == 13
/* clang 13 can throw this warning for a define in ctype.h */
#pragma clang diagnostic ignored "-Wreserved-identifier"
#endif
/* Filenames aren't consistent across targets as they can use different casing 
 * (e.g. WinSock2 vs winsock2). */
#pragma clang diagnostic ignored "-Wnonportable-system-include-path"
/* Very difficult to workaround this warning in C, especially for an ECS. */
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
/* This warning gets thrown when trying to cast pointer returned from dlproc */
#pragma clang diagnostic ignored "-Wcast-function-type-strict"
/* This warning can get thrown for expressions that evaluate to constants
 * in debug/release mode. */
#pragma clang diagnostic ignored "-Wconstant-logical-operand"
/* With soft asserts enabled the code won't abort, which in some cases means
 * code paths are reached where values are uninitialized. */
#ifdef FLECS_SOFT_ASSERT
#pragma clang diagnostic ignored "-Wsometimes-uninitialized"
#endif

/* Allows for enum reflection support on legacy compilers */
#if __clang_major__ < 16
#pragma clang diagnostic ignored "-Wenum-constexpr-conversion"
#endif

#elif defined(ECS_TARGET_GNU)
#ifndef __cplusplus
#pragma GCC diagnostic ignored "-Wdeclaration-after-statement"
#pragma GCC diagnostic ignored "-Wbad-function-cast"
#endif
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#pragma GCC diagnostic ignored "-Wunused-macros"
/* This warning gets thrown *sometimes* when not all members for a struct are
 * provided in an initializer. Flecs heavily relies on descriptor structs that
 * only require partly initialization, so this warning isn't useful.
 * It doesn't introduce any safety issues (fields are guaranteed to be 0 
 * initialized), and later versions of gcc (>=11) seem to no longer throw this 
 * warning. */
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
/* Produces false positives in addons/cpp/delegate.hpp. */
#pragma GCC diagnostic ignored "-Warray-bounds"
/* Produces false positives in queries/src/cache.c */
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#pragma GCC diagnostic ignored "-Wrestrict"

#elif defined(ECS_TARGET_MSVC)
/* recursive on all control paths, function will cause runtime stack overflow
 * This warning is incorrectly thrown on enum reflection code. */
#pragma warning(disable: 4717)
#endif

/* Allows for enum reflection support on legacy compilers */
#if defined(__GNUC__) && __GNUC__ <= 10
#pragma GCC diagnostic ignored "-Wconversion"
#endif

/* Standard library dependencies */
#include <assert.h>
#include <stdarg.h>
#include <string.h>

/* Non-standard but required. If not provided by platform, add manually. */
#include <stdint.h>

/* Contains macros for importing / exporting symbols */
/*
                                   )
                                  (.)
                                  .|.
                                  | |
                              _.--| |--._
                           .-';  ;`-'& ; `&.
                          \   &  ;    &   &_/
                           |"""---...---"""|
                           \ | | | | | | | /
                            `---.|.|.|.---'

 * This file is generated by bake.lang.c for your convenience. Headers of
 * dependencies will automatically show up in this file. Include bake_config.h
 * in your main project file. Do not edit! */

#ifndef FLECS_BAKE_CONFIG_H
#define FLECS_BAKE_CONFIG_H

/* Headers of public dependencies */
/* No dependencies */

/* Convenience macro for exporting symbols */
#ifndef flecs_STATIC
#if defined(flecs_EXPORTS) && (defined(_MSC_VER) || defined(__MINGW32__))
  #define FLECS_API __declspec(dllexport)
#elif defined(flecs_EXPORTS)
  #define FLECS_API __attribute__((__visibility__("default")))
#elif defined(_MSC_VER)
  #define FLECS_API __declspec(dllimport)
#else
  #define FLECS_API
#endif
#else
  #define FLECS_API
#endif

#endif



#ifdef __cplusplus
extern "C" {
#endif

#ifdef __BAKE_LEGACY__
#define FLECS_LEGACY
#endif

/* Some symbols are only exported when building in debug build, to enable
 * white-box testing of internal data structures */
#ifndef FLECS_NDEBUG
#define FLECS_DBG_API FLECS_API
#else
#define FLECS_DBG_API
#endif


////////////////////////////////////////////////////////////////////////////////
//// Language support defines
////////////////////////////////////////////////////////////////////////////////

#ifndef FLECS_LEGACY
#include <stdbool.h>
#endif

#ifndef NULL
#define NULL ((void*)0)
#endif

/* The API uses the native bool type in C++, or a custom one in C */
#if !defined(__cplusplus) && !defined(__bool_true_false_are_defined)
#undef bool
#undef true
#undef false
typedef char bool;
#define false 0
#define true !false
#endif

/* Utility types to indicate usage as bitmask */
typedef uint8_t ecs_flags8_t;
typedef uint16_t ecs_flags16_t;
typedef uint32_t ecs_flags32_t;
typedef uint64_t ecs_flags64_t;

/* Bitmask type with compile-time defined size */
#define ecs_flagsn_t_(bits) ecs_flags##bits##_t
#define ecs_flagsn_t(bits) ecs_flagsn_t_(bits)

/* Bitset type that can store exactly as many bits as there are terms */
#define ecs_termset_t ecs_flagsn_t(FLECS_TERM_COUNT_MAX)

/* Utility macro's for setting/clearing termset bits */
#define ECS_TERMSET_SET(set, flag) ((set) |= (ecs_termset_t)(flag))
#define ECS_TERMSET_CLEAR(set, flag) ((set) &= (ecs_termset_t)~(flag))
#define ECS_TERMSET_COND(set, flag, cond) ((cond) \
    ? (ECS_TERMSET_SET(set, flag)) \
    : (ECS_TERMSET_CLEAR(set, flag)))

/* Keep unsigned integers out of the codebase as they do more harm than good */
typedef int32_t ecs_size_t;

/* Allocator type */
typedef struct ecs_allocator_t ecs_allocator_t;

#define ECS_SIZEOF(T) ECS_CAST(ecs_size_t, sizeof(T))

/* Use alignof in C++, or a trick in C. */
#ifdef __cplusplus
#define ECS_ALIGNOF(T) static_cast<int64_t>(alignof(T))
#elif defined(ECS_TARGET_MSVC)
#define ECS_ALIGNOF(T) (int64_t)__alignof(T)
#elif defined(ECS_TARGET_GNU)
#define ECS_ALIGNOF(T) (int64_t)__alignof__(T)
#elif defined(ECS_TARGET_CLANG)
#define ECS_ALIGNOF(T) (int64_t)__alignof__(T)
#else
#define ECS_ALIGNOF(T) ((int64_t)&((struct { char c; T d; } *)0)->d)
#endif

#ifndef FLECS_NO_DEPRECATED_WARNINGS
#if defined(ECS_TARGET_GNU)
#define ECS_DEPRECATED(msg) __attribute__((deprecated(msg)))
#elif defined(ECS_TARGET_MSVC)
#define ECS_DEPRECATED(msg) __declspec(deprecated(msg))
#else
#define ECS_DEPRECATED(msg)
#endif
#else
#define ECS_DEPRECATED(msg)
#endif

#define ECS_ALIGN(size, alignment) (ecs_size_t)((((((size_t)size) - 1) / ((size_t)alignment)) + 1) * ((size_t)alignment))

/* Simple utility for determining the max of two values */
#define ECS_MAX(a, b) (((a) > (b)) ? a : b)
#define ECS_MIN(a, b) (((a) < (b)) ? a : b)

/* Abstraction on top of C-style casts so that C functions can be used in C++
 * code without producing warnings */
#ifndef __cplusplus
#define ECS_CAST(T, V) ((T)(V))
#else
#define ECS_CAST(T, V) (static_cast<T>(V))
#endif

/* Utility macro for doing const casts without warnings */
#ifndef __cplusplus
#define ECS_CONST_CAST(type, value) ((type)(uintptr_t)(value))
#else
#define ECS_CONST_CAST(type, value) (const_cast<type>(value))
#endif

/* Utility macro for doing pointer casts without warnings */
#ifndef __cplusplus
#define ECS_PTR_CAST(type, value) ((type)(uintptr_t)(value))
#else
#define ECS_PTR_CAST(type, value) (reinterpret_cast<type>(value))
#endif

/* Utility macro's to do bitwise comparisons between floats without warnings */
#define ECS_EQ(a, b) (ecs_os_memcmp(&(a), &(b), sizeof(a)) == 0)
#define ECS_NEQ(a, b) (!ECS_EQ(a, b))
#define ECS_EQZERO(a) ECS_EQ(a, (uint64_t){0})
#define ECS_NEQZERO(a) ECS_NEQ(a, (uint64_t){0})

/* Utilities to convert flecs version to string */
#define FLECS_VERSION_IMPLSTR(major, minor, patch) #major "." #minor "." #patch
#define FLECS_VERSION_IMPL(major, minor, patch) \
    FLECS_VERSION_IMPLSTR(major, minor, patch)

#define ECS_CONCAT(a, b) a ## b

////////////////////////////////////////////////////////////////////////////////
//// Magic numbers for sanity checking
////////////////////////////////////////////////////////////////////////////////

/* Magic number to identify the type of the object */
#define ecs_world_t_magic     (0x65637377)
#define ecs_stage_t_magic     (0x65637373)
#define ecs_query_t_magic     (0x65637375)
#define ecs_observer_t_magic  (0x65637362)


////////////////////////////////////////////////////////////////////////////////
//// Entity id macros
////////////////////////////////////////////////////////////////////////////////

#define ECS_ROW_MASK                  (0x0FFFFFFFu)
#define ECS_ROW_FLAGS_MASK            (~ECS_ROW_MASK)
#define ECS_RECORD_TO_ROW(v)          (ECS_CAST(int32_t, (ECS_CAST(uint32_t, v) & ECS_ROW_MASK)))
#define ECS_RECORD_TO_ROW_FLAGS(v)    (ECS_CAST(uint32_t, v) & ECS_ROW_FLAGS_MASK)
#define ECS_ROW_TO_RECORD(row, flags) (ECS_CAST(uint32_t, (ECS_CAST(uint32_t, row) | (flags))))

#define ECS_ID_FLAGS_MASK             (0xFFull << 60)
#define ECS_ENTITY_MASK               (0xFFFFFFFFull)
#define ECS_GENERATION_MASK           (0xFFFFull << 32)
#define ECS_GENERATION(e)             ((e & ECS_GENERATION_MASK) >> 32)
#define ECS_GENERATION_INC(e)         ((e & ~ECS_GENERATION_MASK) | ((0xFFFF & (ECS_GENERATION(e) + 1)) << 32))
#define ECS_COMPONENT_MASK            (~ECS_ID_FLAGS_MASK)
#define ECS_HAS_ID_FLAG(e, flag)      ((e) & ECS_##flag)
#define ECS_IS_PAIR(id)               (((id) & ECS_ID_FLAGS_MASK) == ECS_PAIR)
#define ECS_PAIR_FIRST(e)             (ecs_entity_t_hi(e & ECS_COMPONENT_MASK))
#define ECS_PAIR_SECOND(e)            (ecs_entity_t_lo(e))
#define ECS_HAS_RELATION(e, rel)      (ECS_HAS_ID_FLAG(e, PAIR) && (ECS_PAIR_FIRST(e) == rel))

#define ECS_TERM_REF_FLAGS(ref)       ((ref)->id & EcsTermRefFlags)
#define ECS_TERM_REF_ID(ref)          ((ref)->id & ~EcsTermRefFlags)

////////////////////////////////////////////////////////////////////////////////
//// Convert between C typenames and variables
////////////////////////////////////////////////////////////////////////////////

/** Translate C type to id. */
#define ecs_id(T) FLECS_ID##T##ID_


////////////////////////////////////////////////////////////////////////////////
//// Utilities for working with pair identifiers
////////////////////////////////////////////////////////////////////////////////

#define ecs_entity_t_lo(value) ECS_CAST(uint32_t, value)
#define ecs_entity_t_hi(value) ECS_CAST(uint32_t, (value) >> 32)
#define ecs_entity_t_comb(lo, hi) ((ECS_CAST(uint64_t, hi) << 32) + ECS_CAST(uint32_t, lo))

#define ecs_pair(pred, obj) (ECS_PAIR | ecs_entity_t_comb(obj, pred))
#define ecs_pair_t(pred, obj) (ECS_PAIR | ecs_entity_t_comb(obj, ecs_id(pred)))
#define ecs_pair_first(world, pair) ecs_get_alive(world, ECS_PAIR_FIRST(pair))
#define ecs_pair_second(world, pair) ecs_get_alive(world, ECS_PAIR_SECOND(pair))
#define ecs_pair_relation ecs_pair_first
#define ecs_pair_target ecs_pair_second

#define flecs_poly_id(tag) ecs_pair(ecs_id(EcsPoly), tag)


////////////////////////////////////////////////////////////////////////////////
//// Debug macros
////////////////////////////////////////////////////////////////////////////////

#ifndef FLECS_NDEBUG
#define ECS_TABLE_LOCK(world, table) ecs_table_lock(world, table)
#define ECS_TABLE_UNLOCK(world, table) ecs_table_unlock(world, table)
#else
#define ECS_TABLE_LOCK(world, table)
#define ECS_TABLE_UNLOCK(world, table)
#endif


////////////////////////////////////////////////////////////////////////////////
//// Actions that drive iteration
////////////////////////////////////////////////////////////////////////////////

#define EcsIterNextYield  (0)   /* Move to next table, yield current */
#define EcsIterYield      (-1)  /* Stay on current table, yield */
#define EcsIterNext  (1)   /* Move to next table, don't yield */

////////////////////////////////////////////////////////////////////////////////
//// Convenience macros for ctor, dtor, move and copy
////////////////////////////////////////////////////////////////////////////////

#ifndef FLECS_LEGACY

/* Constructor/Destructor convenience macro */
#define ECS_XTOR_IMPL(type, postfix, var, ...)\
    void type##_##postfix(\
        void *_ptr,\
        int32_t _count,\
        const ecs_type_info_t *type_info)\
    {\
        (void)_ptr;\
        (void)_count;\
        (void)type_info;\
        for (int32_t i = 0; i < _count; i ++) {\
            type *var = &((type*)_ptr)[i];\
            (void)var;\
            __VA_ARGS__\
        }\
    }

/* Copy convenience macro */
#define ECS_COPY_IMPL(type, dst_var, src_var, ...)\
    void type##_##copy(\
        void *_dst_ptr,\
        const void *_src_ptr,\
        int32_t _count,\
        const ecs_type_info_t *type_info)\
    {\
        (void)_dst_ptr;\
        (void)_src_ptr;\
        (void)_count;\
        (void)type_info;\
        for (int32_t i = 0; i < _count; i ++) {\
            type *dst_var = &((type*)_dst_ptr)[i];\
            const type *src_var = &((const type*)_src_ptr)[i];\
            (void)dst_var;\
            (void)src_var;\
            __VA_ARGS__\
        }\
    }

/* Move convenience macro */
#define ECS_MOVE_IMPL(type, dst_var, src_var, ...)\
    void type##_##move(\
        void *_dst_ptr,\
        void *_src_ptr,\
        int32_t _count,\
        const ecs_type_info_t *type_info)\
    {\
        (void)_dst_ptr;\
        (void)_src_ptr;\
        (void)_count;\
        (void)type_info;\
        for (int32_t i = 0; i < _count; i ++) {\
            type *dst_var = &((type*)_dst_ptr)[i];\
            type *src_var = &((type*)_src_ptr)[i];\
            (void)dst_var;\
            (void)src_var;\
            __VA_ARGS__\
        }\
    }

#define ECS_HOOK_IMPL(type, func, var, ...)\
    void func(ecs_iter_t *_it)\
    {\
        for (int32_t i = 0; i < _it->count; i ++) {\
            ecs_entity_t entity = _it->entities[i];\
            type *var = ecs_field(_it, type, 0);\
            (void)entity;\
            (void)var;\
            __VA_ARGS__\
        }\
    }

#endif

#ifdef __cplusplus
}
#endif

#endif

/**
 * @file vec.h
 * @brief Vector with allocator support.
 */

#ifndef FLECS_VEC_H
#define FLECS_VEC_H


#ifdef __cplusplus
extern "C" {
#endif

/** A component column. */
typedef struct ecs_vec_t {
    void *array;
    int32_t count;
    int32_t size;
#ifdef FLECS_SANITIZE
    ecs_size_t elem_size;
    const char *type_name;
#endif
} ecs_vec_t;

FLECS_API
void ecs_vec_init(
    struct ecs_allocator_t *allocator,
    ecs_vec_t *vec,
    ecs_size_t size,
    int32_t elem_count);

FLECS_API
void ecs_vec_init_w_dbg_info(
    struct ecs_allocator_t *allocator,
    ecs_vec_t *vec,
    ecs_size_t size,
    int32_t elem_count,
    const char *type_name);

#define ecs_vec_init_t(allocator, vec, T, elem_count) \
    ecs_vec_init_w_dbg_info(allocator, vec, ECS_SIZEOF(T), elem_count, "vec<"#T">")

FLECS_API
void ecs_vec_init_if(
    ecs_vec_t *vec,
    ecs_size_t size);

#define ecs_vec_init_if_t(vec, T) \
    ecs_vec_init_if(vec, ECS_SIZEOF(T))

FLECS_API
void ecs_vec_fini(
    struct ecs_allocator_t *allocator,
    ecs_vec_t *vec,
    ecs_size_t size);

#define ecs_vec_fini_t(allocator, vec, T) \
    ecs_vec_fini(allocator, vec, ECS_SIZEOF(T))

FLECS_API
ecs_vec_t* ecs_vec_reset(
    struct ecs_allocator_t *allocator,
    ecs_vec_t *vec,
    ecs_size_t size);

#define ecs_vec_reset_t(allocator, vec, T) \
    ecs_vec_reset(allocator, vec, ECS_SIZEOF(T))

FLECS_API
void ecs_vec_clear(
    ecs_vec_t *vec);

FLECS_API
void* ecs_vec_append(
    struct ecs_allocator_t *allocator,
    ecs_vec_t *vec,
    ecs_size_t size);

#define ecs_vec_append_t(allocator, vec, T) \
    ECS_CAST(T*, ecs_vec_append(allocator, vec, ECS_SIZEOF(T)))

FLECS_API
void ecs_vec_remove(
    ecs_vec_t *vec,
    ecs_size_t size,
    int32_t elem);

#define ecs_vec_remove_t(vec, T, elem) \
    ecs_vec_remove(vec, ECS_SIZEOF(T), elem)

FLECS_API
void ecs_vec_remove_last(
    ecs_vec_t *vec);

FLECS_API
ecs_vec_t ecs_vec_copy(
    struct ecs_allocator_t *allocator,
    const ecs_vec_t *vec,
    ecs_size_t size);

#define ecs_vec_copy_t(allocator, vec, T) \
    ecs_vec_copy(allocator, vec, ECS_SIZEOF(T))

FLECS_API
ecs_vec_t ecs_vec_copy_shrink(
    struct ecs_allocator_t *allocator,
    const ecs_vec_t *vec,
    ecs_size_t size);

#define ecs_vec_copy_shrink_t(allocator, vec, T) \
    ecs_vec_copy_shrink(allocator, vec, ECS_SIZEOF(T))

FLECS_API
void ecs_vec_reclaim(
    struct ecs_allocator_t *allocator,
    ecs_vec_t *vec,
    ecs_size_t size);

#define ecs_vec_reclaim_t(allocator, vec, T) \
    ecs_vec_reclaim(allocator, vec, ECS_SIZEOF(T))

FLECS_API
void ecs_vec_set_size(
    struct ecs_allocator_t *allocator,
    ecs_vec_t *vec,
    ecs_size_t size,
    int32_t elem_count);

#define ecs_vec_set_size_t(allocator, vec, T, elem_count) \
    ecs_vec_set_size(allocator, vec, ECS_SIZEOF(T), elem_count)

FLECS_API
void ecs_vec_set_min_size(
    struct ecs_allocator_t *allocator,
    ecs_vec_t *vec,
    ecs_size_t size,
    int32_t elem_count);

#define ecs_vec_set_min_size_t(allocator, vec, T, elem_count) \
    ecs_vec_set_min_size(allocator, vec, ECS_SIZEOF(T), elem_count)

FLECS_API
void ecs_vec_set_min_count(
    struct ecs_allocator_t *allocator,
    ecs_vec_t *vec,
    ecs_size_t size,
    int32_t elem_count);

#define ecs_vec_set_min_count_t(allocator, vec, T, elem_count) \
    ecs_vec_set_min_count(allocator, vec, ECS_SIZEOF(T), elem_count)

FLECS_API
void ecs_vec_set_min_count_zeromem(
    struct ecs_allocator_t *allocator,
    ecs_vec_t *vec,
    ecs_size_t size,
    int32_t elem_count);

#define ecs_vec_set_min_count_zeromem_t(allocator, vec, T, elem_count) \
    ecs_vec_set_min_count_zeromem(allocator, vec, ECS_SIZEOF(T), elem_count)

FLECS_API
void ecs_vec_set_count(
    struct ecs_allocator_t *allocator,
    ecs_vec_t *vec,
    ecs_size_t size,
    int32_t elem_count);

#define ecs_vec_set_count_t(allocator, vec, T, elem_count) \
    ecs_vec_set_count(allocator, vec, ECS_SIZEOF(T), elem_count)

FLECS_API
void* ecs_vec_grow(
    struct ecs_allocator_t *allocator,
    ecs_vec_t *vec,
    ecs_size_t size,
    int32_t elem_count);

#define ecs_vec_grow_t(allocator, vec, T, elem_count) \
    ecs_vec_grow(allocator, vec, ECS_SIZEOF(T), elem_count)

FLECS_API
int32_t ecs_vec_count(
    const ecs_vec_t *vec);

FLECS_API
int32_t ecs_vec_size(
    const ecs_vec_t *vec);

FLECS_API
void* ecs_vec_get(
    const ecs_vec_t *vec,
    ecs_size_t size,
    int32_t index);

#define ecs_vec_get_t(vec, T, index) \
    ECS_CAST(T*, ecs_vec_get(vec, ECS_SIZEOF(T), index))

FLECS_API
void* ecs_vec_first(
    const ecs_vec_t *vec);

#define ecs_vec_first_t(vec, T) \
    ECS_CAST(T*, ecs_vec_first(vec))

FLECS_API
void* ecs_vec_last(
    const ecs_vec_t *vec,
    ecs_size_t size);

#define ecs_vec_last_t(vec, T) \
    ECS_CAST(T*, ecs_vec_last(vec, ECS_SIZEOF(T)))

#ifdef __cplusplus
}
#endif

#endif

/**
 * @file sparse.h
 * @brief Sparse set data structure.
 */

#ifndef FLECS_SPARSE_H
#define FLECS_SPARSE_H


#ifdef __cplusplus
extern "C" {
#endif

/** The number of elements in a single page */
#define FLECS_SPARSE_PAGE_SIZE (1 << FLECS_SPARSE_PAGE_BITS)

/** Compute the page index from an id by stripping the first 12 bits */
#define FLECS_SPARSE_PAGE(index) ((int32_t)((uint32_t)index >> FLECS_SPARSE_PAGE_BITS))

/** This computes the offset of an index inside a page */
#define FLECS_SPARSE_OFFSET(index) ((int32_t)index & (FLECS_SPARSE_PAGE_SIZE - 1))

typedef struct ecs_sparse_t {
    ecs_vec_t dense;         /* Dense array with indices to sparse array. The
                              * dense array stores both alive and not alive
                              * sparse indices. The 'count' member keeps
                              * track of which indices are alive. */

    ecs_vec_t pages;         /* Chunks with sparse arrays & data */
    ecs_size_t size;         /* Element size */
    int32_t count;           /* Number of alive entries */
    uint64_t max_id;         /* Local max index (if no global is set) */
    struct ecs_allocator_t *allocator;
    struct ecs_block_allocator_t *page_allocator;
} ecs_sparse_t;

/** Initialize sparse set */
FLECS_DBG_API
void flecs_sparse_init(
    ecs_sparse_t *result,
    struct ecs_allocator_t *allocator,
    struct ecs_block_allocator_t *page_allocator,
    ecs_size_t size);

#define flecs_sparse_init_t(result, allocator, page_allocator, T)\
    flecs_sparse_init(result, allocator, page_allocator, ECS_SIZEOF(T))

FLECS_DBG_API
void flecs_sparse_fini(
    ecs_sparse_t *sparse);

/** Remove all elements from sparse set */
FLECS_DBG_API
void flecs_sparse_clear(
    ecs_sparse_t *sparse);

/** Add element to sparse set, this generates or recycles an id */
FLECS_DBG_API
void* flecs_sparse_add(
    ecs_sparse_t *sparse,
    ecs_size_t elem_size);

#define flecs_sparse_add_t(sparse, T)\
    ECS_CAST(T*, flecs_sparse_add(sparse, ECS_SIZEOF(T)))

/** Get last issued id. */
FLECS_DBG_API
uint64_t flecs_sparse_last_id(
    const ecs_sparse_t *sparse);

/** Generate or recycle a new id. */
FLECS_DBG_API
uint64_t flecs_sparse_new_id(
    ecs_sparse_t *sparse);

/** Remove an element */
FLECS_DBG_API
void flecs_sparse_remove(
    ecs_sparse_t *sparse,
    ecs_size_t elem_size,
    uint64_t id);

#define flecs_sparse_remove_t(sparse, T, id)\
    flecs_sparse_remove(sparse, ECS_SIZEOF(T), id)

/** Remove an element without liveliness checking */
FLECS_DBG_API
void* flecs_sparse_remove_fast(
    ecs_sparse_t *sparse,
    ecs_size_t size,
    uint64_t index);

/** Test if id is alive, which requires the generation count to match. */
FLECS_DBG_API
bool flecs_sparse_is_alive(
    const ecs_sparse_t *sparse,
    uint64_t id);

/** Get value from sparse set by dense id. This function is useful in 
 * combination with flecs_sparse_count for iterating all values in the set. */
FLECS_DBG_API
void* flecs_sparse_get_dense(
    const ecs_sparse_t *sparse,
    ecs_size_t elem_size,
    int32_t index);

#define flecs_sparse_get_dense_t(sparse, T, index)\
    ECS_CAST(T*, flecs_sparse_get_dense(sparse, ECS_SIZEOF(T), index))

/** Get the number of alive elements in the sparse set. */
FLECS_DBG_API
int32_t flecs_sparse_count(
    const ecs_sparse_t *sparse);

/** Get element by (sparse) id. The returned pointer is stable for the duration
 * of the sparse set, as it is stored in the sparse array. */
FLECS_DBG_API
void* flecs_sparse_get(
    const ecs_sparse_t *sparse,
    ecs_size_t elem_size,
    uint64_t id);

#define flecs_sparse_get_t(sparse, T, index)\
    ECS_CAST(T*, flecs_sparse_get(sparse, ECS_SIZEOF(T), index))

/** Same as flecs_sparse_get, but doesn't assert if id is not alive. */
FLECS_DBG_API
void* flecs_sparse_try(
    const ecs_sparse_t *sparse,
    ecs_size_t elem_size,
    uint64_t id);

#define flecs_sparse_try_t(sparse, T, index)\
    ECS_CAST(T*, flecs_sparse_try(sparse, ECS_SIZEOF(T), index))

/** Like get_sparse, but don't care whether element is alive or not. */
FLECS_DBG_API
void* flecs_sparse_get_any(
    const ecs_sparse_t *sparse,
    ecs_size_t elem_size,
    uint64_t id);

#define flecs_sparse_get_any_t(sparse, T, index)\
    ECS_CAST(T*, flecs_sparse_get_any(sparse, ECS_SIZEOF(T), index))

/** Get or create element by (sparse) id. */
FLECS_DBG_API
void* flecs_sparse_ensure(
    ecs_sparse_t *sparse,
    ecs_size_t elem_size,
    uint64_t id);

#define flecs_sparse_ensure_t(sparse, T, index)\
    ECS_CAST(T*, flecs_sparse_ensure(sparse, ECS_SIZEOF(T), index))

/** Fast version of ensure, no liveliness checking */
FLECS_DBG_API
void* flecs_sparse_ensure_fast(
    ecs_sparse_t *sparse,
    ecs_size_t elem_size,
    uint64_t id);

#define flecs_sparse_ensure_fast_t(sparse, T, index)\
    ECS_CAST(T*, flecs_sparse_ensure_fast(sparse, ECS_SIZEOF(T), index))

/** Get pointer to ids (alive and not alive). Use with count() or size(). */
FLECS_DBG_API
const uint64_t* flecs_sparse_ids(
    const ecs_sparse_t *sparse);

/* Publicly exposed APIs 
 * These APIs are not part of the public API and as a result may change without
 * notice (though they haven't changed in a long time). */

FLECS_API
void ecs_sparse_init(
    ecs_sparse_t *sparse,
    ecs_size_t elem_size);

#define ecs_sparse_init_t(sparse, T)\
    ecs_sparse_init(sparse, ECS_SIZEOF(T))

FLECS_API
void* ecs_sparse_add(
    ecs_sparse_t *sparse,
    ecs_size_t elem_size);

#define ecs_sparse_add_t(sparse, T)\
    ECS_CAST(T*, ecs_sparse_add(sparse, ECS_SIZEOF(T)))

FLECS_API
uint64_t ecs_sparse_last_id(
    const ecs_sparse_t *sparse);

FLECS_API
int32_t ecs_sparse_count(
    const ecs_sparse_t *sparse);

FLECS_API
void* ecs_sparse_get_dense(
    const ecs_sparse_t *sparse,
    ecs_size_t elem_size,
    int32_t index);

#define ecs_sparse_get_dense_t(sparse, T, index)\
    ECS_CAST(T*, ecs_sparse_get_dense(sparse, ECS_SIZEOF(T), index))

FLECS_API
void* ecs_sparse_get(
    const ecs_sparse_t *sparse,
    ecs_size_t elem_size,
    uint64_t id);

#define ecs_sparse_get_t(sparse, T, index)\
    ECS_CAST(T*, ecs_sparse_get(sparse, ECS_SIZEOF(T), index))

#ifdef __cplusplus
}
#endif

#endif

/**
 * @file block_allocator.h
 * @brief Block allocator.
 */

#ifndef FLECS_BLOCK_ALLOCATOR_H
#define FLECS_BLOCK_ALLOCATOR_H


typedef struct ecs_map_t ecs_map_t;

typedef struct ecs_block_allocator_block_t {
    void *memory;
    struct ecs_block_allocator_block_t *next;
} ecs_block_allocator_block_t;

typedef struct ecs_block_allocator_chunk_header_t {
    struct ecs_block_allocator_chunk_header_t *next;
} ecs_block_allocator_chunk_header_t;

typedef struct ecs_block_allocator_t {
    ecs_block_allocator_chunk_header_t *head;
    ecs_block_allocator_block_t *block_head;
    ecs_block_allocator_block_t *block_tail;
    int32_t chunk_size;
    int32_t data_size;
    int32_t chunks_per_block;
    int32_t block_size;
#ifdef FLECS_SANITIZE
    int32_t alloc_count;
    ecs_map_t *outstanding;
#endif
} ecs_block_allocator_t;

FLECS_API
void flecs_ballocator_init(
    ecs_block_allocator_t *ba,
    ecs_size_t size);

#define flecs_ballocator_init_t(ba, T)\
    flecs_ballocator_init(ba, ECS_SIZEOF(T))
#define flecs_ballocator_init_n(ba, T, count)\
    flecs_ballocator_init(ba, ECS_SIZEOF(T) * count)

FLECS_API
ecs_block_allocator_t* flecs_ballocator_new(
    ecs_size_t size);

#define flecs_ballocator_new_t(T)\
    flecs_ballocator_new(ECS_SIZEOF(T))
#define flecs_ballocator_new_n(T, count)\
    flecs_ballocator_new(ECS_SIZEOF(T) * count)

FLECS_API
void flecs_ballocator_fini(
    ecs_block_allocator_t *ba);

FLECS_API
void flecs_ballocator_free(
    ecs_block_allocator_t *ba);

FLECS_API
void* flecs_balloc(
    ecs_block_allocator_t *allocator);

FLECS_API
void* flecs_balloc_w_dbg_info(
    ecs_block_allocator_t *allocator,
    const char *type_name);

FLECS_API
void* flecs_bcalloc(
    ecs_block_allocator_t *allocator);

FLECS_API
void* flecs_bcalloc_w_dbg_info(
    ecs_block_allocator_t *allocator,
    const char *type_name);

FLECS_API
void flecs_bfree(
    ecs_block_allocator_t *allocator, 
    void *memory);

FLECS_API
void flecs_bfree_w_dbg_info(
    ecs_block_allocator_t *allocator, 
    void *memory,
    const char *type_name);

FLECS_API
void* flecs_brealloc(
    ecs_block_allocator_t *dst, 
    ecs_block_allocator_t *src, 
    void *memory);

FLECS_API
void* flecs_brealloc_w_dbg_info(
    ecs_block_allocator_t *dst, 
    ecs_block_allocator_t *src, 
    void *memory,
    const char *type_name);

FLECS_API
void* flecs_bdup(
    ecs_block_allocator_t *ba, 
    void *memory);

#endif

/**
 * @file datastructures/stack_allocator.h
 * @brief Stack allocator.
 */

#ifndef FLECS_STACK_ALLOCATOR_H
#define FLECS_STACK_ALLOCATOR_H

/** Stack allocator for quick allocation of small temporary values */
#define ECS_STACK_PAGE_SIZE (4096)

typedef struct ecs_stack_page_t {
    void *data;
    struct ecs_stack_page_t *next;
    int16_t sp;
    uint32_t id;
} ecs_stack_page_t;

typedef struct ecs_stack_cursor_t {
    struct ecs_stack_cursor_t *prev;
    struct ecs_stack_page_t *page;
    int16_t sp;
    bool is_free;
#ifdef FLECS_DEBUG
    struct ecs_stack_t *owner;
#endif
} ecs_stack_cursor_t;

typedef struct ecs_stack_t {
    ecs_stack_page_t *first;
    ecs_stack_page_t *tail_page;
    ecs_stack_cursor_t *tail_cursor;
#ifdef FLECS_DEBUG
    int32_t cursor_count;
#endif
} ecs_stack_t;

FLECS_DBG_API
void flecs_stack_init(
    ecs_stack_t *stack);

FLECS_DBG_API
void flecs_stack_fini(
    ecs_stack_t *stack);

FLECS_DBG_API
void* flecs_stack_alloc(
    ecs_stack_t *stack, 
    ecs_size_t size,
    ecs_size_t align);

#define flecs_stack_alloc_t(stack, T)\
    flecs_stack_alloc(stack, ECS_SIZEOF(T), ECS_ALIGNOF(T))

#define flecs_stack_alloc_n(stack, T, count)\
    flecs_stack_alloc(stack, ECS_SIZEOF(T) * count, ECS_ALIGNOF(T))

FLECS_DBG_API
void* flecs_stack_calloc(
    ecs_stack_t *stack, 
    ecs_size_t size,
    ecs_size_t align);

#define flecs_stack_calloc_t(stack, T)\
    flecs_stack_calloc(stack, ECS_SIZEOF(T), ECS_ALIGNOF(T))

#define flecs_stack_calloc_n(stack, T, count)\
    flecs_stack_calloc(stack, ECS_SIZEOF(T) * count, ECS_ALIGNOF(T))

FLECS_DBG_API
void flecs_stack_free(
    void *ptr,
    ecs_size_t size);

#define flecs_stack_free_t(ptr, T)\
    flecs_stack_free(ptr, ECS_SIZEOF(T))

#define flecs_stack_free_n(ptr, T, count)\
    flecs_stack_free(ptr, ECS_SIZEOF(T) * count)

void flecs_stack_reset(
    ecs_stack_t *stack);

FLECS_DBG_API
ecs_stack_cursor_t* flecs_stack_get_cursor(
    ecs_stack_t *stack);

FLECS_DBG_API
void flecs_stack_restore_cursor(
    ecs_stack_t *stack,
    ecs_stack_cursor_t *cursor);

#endif

/**
 * @file map.h
 * @brief Map data structure.
 */

#ifndef FLECS_MAP_H
#define FLECS_MAP_H


#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t ecs_map_data_t;
typedef ecs_map_data_t ecs_map_key_t;
typedef ecs_map_data_t ecs_map_val_t;

/* Map type */
typedef struct ecs_bucket_entry_t {
    ecs_map_key_t key;
    ecs_map_val_t value;
    struct ecs_bucket_entry_t *next;
} ecs_bucket_entry_t;

typedef struct ecs_bucket_t {
    ecs_bucket_entry_t *first;
} ecs_bucket_t;

struct ecs_map_t {
    uint8_t bucket_shift;
    bool shared_allocator;
    ecs_bucket_t *buckets;
    int32_t bucket_count;
    int32_t count;
    struct ecs_block_allocator_t *entry_allocator;
    struct ecs_allocator_t *allocator;
};

typedef struct ecs_map_iter_t {
    const ecs_map_t *map;
    ecs_bucket_t *bucket;
    ecs_bucket_entry_t *entry;
    ecs_map_data_t *res;
} ecs_map_iter_t;

typedef struct ecs_map_params_t {
    struct ecs_allocator_t *allocator;
    struct ecs_block_allocator_t entry_allocator;
} ecs_map_params_t;

/* Function/macro postfixes meaning:
 *   _ptr:    access ecs_map_val_t as void*
 *   _ref:    access ecs_map_val_t* as T**
 *   _deref:  dereferences a _ref
 *   _alloc:  if _ptr is NULL, alloc
 *   _free:   if _ptr is not NULL, free
 */

FLECS_API
void ecs_map_params_init(
    ecs_map_params_t *params,
    struct ecs_allocator_t *allocator);

FLECS_API
void ecs_map_params_fini(
    ecs_map_params_t *params);

/** Initialize new map. */
FLECS_API
void ecs_map_init(
    ecs_map_t *map,
    struct ecs_allocator_t *allocator);

/** Initialize new map. */
FLECS_API
void ecs_map_init_w_params(
    ecs_map_t *map,
    ecs_map_params_t *params);

/** Initialize new map if uninitialized, leave as is otherwise */
FLECS_API
void ecs_map_init_if(
    ecs_map_t *map,
    struct ecs_allocator_t *allocator);

FLECS_API
void ecs_map_init_w_params_if(
    ecs_map_t *result,
    ecs_map_params_t *params);

/** Deinitialize map. */
FLECS_API
void ecs_map_fini(
    ecs_map_t *map);

/** Get element for key, returns NULL if they key doesn't exist. */
FLECS_API
ecs_map_val_t* ecs_map_get(
    const ecs_map_t *map,
    ecs_map_key_t key);

/* Get element as pointer (auto-dereferences _ptr) */
FLECS_API
void* ecs_map_get_deref_(
    const ecs_map_t *map,
    ecs_map_key_t key);

/** Get or insert element for key. */
FLECS_API
ecs_map_val_t* ecs_map_ensure(
    ecs_map_t *map,
    ecs_map_key_t key);

/** Get or insert pointer element for key, allocate if the pointer is NULL */
FLECS_API
void* ecs_map_ensure_alloc(
    ecs_map_t *map,
    ecs_size_t elem_size,
    ecs_map_key_t key);

/** Insert element for key. */
FLECS_API
void ecs_map_insert(
    ecs_map_t *map,
    ecs_map_key_t key,
    ecs_map_val_t value);

/** Insert pointer element for key, populate with new allocation. */
FLECS_API
void* ecs_map_insert_alloc(
    ecs_map_t *map,
    ecs_size_t elem_size,
    ecs_map_key_t key);

/** Remove key from map. */
FLECS_API
ecs_map_val_t ecs_map_remove(
    ecs_map_t *map,
    ecs_map_key_t key);

/* Remove pointer element, free if not NULL */
FLECS_API
void ecs_map_remove_free(
    ecs_map_t *map,
    ecs_map_key_t key);

/** Remove all elements from map. */
FLECS_API
void ecs_map_clear(
    ecs_map_t *map);

/** Return number of elements in map. */
#define ecs_map_count(map) ((map) ? (map)->count : 0)

/** Is map initialized */
#define ecs_map_is_init(map) ((map) ? (map)->bucket_shift != 0 : false)

/** Return iterator to map contents. */
FLECS_API
ecs_map_iter_t ecs_map_iter(
    const ecs_map_t *map);

/** Obtain next element in map from iterator. */
FLECS_API
bool ecs_map_next(
    ecs_map_iter_t *iter);

/** Copy map. */
FLECS_API
void ecs_map_copy(
    ecs_map_t *dst,
    const ecs_map_t *src);

#define ecs_map_get_ref(m, T, k) ECS_CAST(T**, ecs_map_get(m, k))
#define ecs_map_get_deref(m, T, k) ECS_CAST(T*, ecs_map_get_deref_(m, k))
#define ecs_map_ensure_ref(m, T, k) ECS_CAST(T**, ecs_map_ensure(m, k))

#define ecs_map_insert_ptr(m, k, v) ecs_map_insert(m, k, ECS_CAST(ecs_map_val_t, ECS_PTR_CAST(uintptr_t, v)))
#define ecs_map_insert_alloc_t(m, T, k) ECS_CAST(T*, ecs_map_insert_alloc(m, ECS_SIZEOF(T), k))
#define ecs_map_ensure_alloc_t(m, T, k) ECS_PTR_CAST(T*, (uintptr_t)ecs_map_ensure_alloc(m, ECS_SIZEOF(T), k))
#define ecs_map_remove_ptr(m, k) (ECS_PTR_CAST(void*, ECS_CAST(uintptr_t, (ecs_map_remove(m, k)))))

#define ecs_map_key(it) ((it)->res[0])
#define ecs_map_value(it) ((it)->res[1])
#define ecs_map_ptr(it) ECS_PTR_CAST(void*, ECS_CAST(uintptr_t, ecs_map_value(it)))
#define ecs_map_ref(it, T) (ECS_CAST(T**, &((it)->res[1])))

#ifdef __cplusplus
}
#endif

#endif

/**
 * @file switch_list.h
 * @brief Interleaved linked list for storing mutually exclusive values.
 */

#ifndef FLECS_SWITCH_LIST_H
#define FLECS_SWITCH_LIST_H


#ifdef __cplusplus
extern "C" {
#endif

typedef struct ecs_switch_node_t {
    uint32_t next;      /* Next node in list */
    uint32_t prev;      /* Prev node in list */
} ecs_switch_node_t;

typedef struct ecs_switch_page_t {
    ecs_vec_t nodes;    /* vec<ecs_switch_node_t> */
    ecs_vec_t values;   /* vec<uint64_t> */
} ecs_switch_page_t;

typedef struct ecs_switch_t {
    ecs_map_t hdrs;     /* map<uint64_t, uint32_t> */
    ecs_vec_t pages;    /* vec<ecs_switch_page_t> */
} ecs_switch_t;

/** Init new switch. */
FLECS_DBG_API
void flecs_switch_init(
    ecs_switch_t* sw,
    ecs_allocator_t *allocator);

/** Fini switch. */
FLECS_DBG_API
void flecs_switch_fini(
    ecs_switch_t *sw);

/** Set value of element. */
FLECS_DBG_API
bool flecs_switch_set(
    ecs_switch_t *sw,
    uint32_t element,
    uint64_t value);

/** Reset value of element. */
FLECS_DBG_API
bool flecs_switch_reset(
    ecs_switch_t *sw,
    uint32_t element);

/** Get value for element. */
FLECS_DBG_API
uint64_t flecs_switch_get(
    const ecs_switch_t *sw,
    uint32_t element);

/** Get first element for value. */
FLECS_DBG_API
uint32_t flecs_switch_first(
    const ecs_switch_t *sw,
    uint64_t value);

/** Get next element. */
FLECS_DBG_API
uint32_t flecs_switch_next(
    const ecs_switch_t *sw,
    uint32_t previous);

/** Get target iterator. */
FLECS_DBG_API
ecs_map_iter_t flecs_switch_targets(
    const ecs_switch_t *sw);

#ifdef __cplusplus
}
#endif

#endif

/**
 * @file allocator.h
 * @brief Allocator that returns memory objects of any size. 
 */

#ifndef FLECS_ALLOCATOR_H
#define FLECS_ALLOCATOR_H


FLECS_DBG_API extern int64_t ecs_block_allocator_alloc_count;
FLECS_DBG_API extern int64_t ecs_block_allocator_free_count;
FLECS_DBG_API extern int64_t ecs_stack_allocator_alloc_count;
FLECS_DBG_API extern int64_t ecs_stack_allocator_free_count;

struct ecs_allocator_t {
    ecs_block_allocator_t chunks;
    struct ecs_sparse_t sizes; /* <size, block_allocator_t> */
};

FLECS_API
void flecs_allocator_init(
    ecs_allocator_t *a);

FLECS_API
void flecs_allocator_fini(
    ecs_allocator_t *a);

FLECS_API
ecs_block_allocator_t* flecs_allocator_get(
    ecs_allocator_t *a, 
    ecs_size_t size);

FLECS_API
char* flecs_strdup(
    ecs_allocator_t *a, 
    const char* str);

FLECS_API
void flecs_strfree(
    ecs_allocator_t *a, 
    char* str);

FLECS_API
void* flecs_dup(
    ecs_allocator_t *a,
    ecs_size_t size,
    const void *src);

#define flecs_allocator(obj) (&obj->allocators.dyn)

#define flecs_alloc(a, size) flecs_balloc(flecs_allocator_get(a, size))
#define flecs_alloc_w_dbg_info(a, size, type_name) flecs_balloc_w_dbg_info(flecs_allocator_get(a, size), type_name)
#define flecs_alloc_t(a, T) flecs_alloc_w_dbg_info(a, ECS_SIZEOF(T), #T)
#define flecs_alloc_n(a, T, count) flecs_alloc_w_dbg_info(a, ECS_SIZEOF(T) * (count), #T)

#define flecs_calloc(a, size) flecs_bcalloc(flecs_allocator_get(a, size))
#define flecs_calloc_w_dbg_info(a, size, type_name) flecs_bcalloc_w_dbg_info(flecs_allocator_get(a, size), type_name)
#define flecs_calloc_t(a, T) flecs_calloc_w_dbg_info(a, ECS_SIZEOF(T), #T)
#define flecs_calloc_n(a, T, count) flecs_calloc_w_dbg_info(a, ECS_SIZEOF(T) * (count), #T)

#define flecs_free(a, size, ptr)\
    flecs_bfree((ptr) ? flecs_allocator_get(a, size) : NULL, ptr)
#define flecs_free_t(a, T, ptr)\
    flecs_bfree_w_dbg_info((ptr) ? flecs_allocator_get(a, ECS_SIZEOF(T)) : NULL, ptr, #T)
#define flecs_free_n(a, T, count, ptr)\
    flecs_bfree_w_dbg_info((ptr) ? flecs_allocator_get(a, ECS_SIZEOF(T) * (count)) : NULL\
        , ptr, #T)

#define flecs_realloc(a, size_dst, size_src, ptr)\
    flecs_brealloc(flecs_allocator_get(a, size_dst),\
    flecs_allocator_get(a, size_src),\
    ptr)
#define flecs_realloc_w_dbg_info(a, size_dst, size_src, ptr, type_name)\
    flecs_brealloc_w_dbg_info(flecs_allocator_get(a, size_dst),\
    flecs_allocator_get(a, size_src),\
    ptr,\
    type_name)
#define flecs_realloc_n(a, T, count_dst, count_src, ptr)\
    flecs_realloc(a, ECS_SIZEOF(T) * (count_dst), ECS_SIZEOF(T) * (count_src), ptr)

#define flecs_dup_n(a, T, count, ptr) flecs_dup(a, ECS_SIZEOF(T) * (count), ptr)

#endif

/**
 * @file strbuf.h
 * @brief Utility for constructing strings.
 */

#ifndef FLECS_STRBUF_H_
#define FLECS_STRBUF_H_


#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
/* Fixes missing field initializer warning on g++ */
#define ECS_STRBUF_INIT (ecs_strbuf_t){}
#else
#define ECS_STRBUF_INIT (ecs_strbuf_t){0}
#endif

#define ECS_STRBUF_SMALL_STRING_SIZE (512)
#define ECS_STRBUF_MAX_LIST_DEPTH (32)

typedef struct ecs_strbuf_list_elem {
    int32_t count;
    const char *separator;
} ecs_strbuf_list_elem;

typedef struct ecs_strbuf_t {
    char *content;
    ecs_size_t length;
    ecs_size_t size;

    ecs_strbuf_list_elem list_stack[ECS_STRBUF_MAX_LIST_DEPTH];
    int32_t list_sp;

    char small_string[ECS_STRBUF_SMALL_STRING_SIZE];
} ecs_strbuf_t;

/* Append format string to a buffer.
 * Returns false when max is reached, true when there is still space */
FLECS_API
void ecs_strbuf_append(
    ecs_strbuf_t *buffer,
    const char *fmt,
    ...);

/* Append format string with argument list to a buffer.
 * Returns false when max is reached, true when there is still space */
FLECS_API
void ecs_strbuf_vappend(
    ecs_strbuf_t *buffer,
    const char *fmt,
    va_list args);

/* Append string to buffer.
 * Returns false when max is reached, true when there is still space */
FLECS_API
void ecs_strbuf_appendstr(
    ecs_strbuf_t *buffer,
    const char *str);

/* Append character to buffer.
 * Returns false when max is reached, true when there is still space */
FLECS_API
void ecs_strbuf_appendch(
    ecs_strbuf_t *buffer,
    char ch);

/* Append int to buffer.
 * Returns false when max is reached, true when there is still space */
FLECS_API
void ecs_strbuf_appendint(
    ecs_strbuf_t *buffer,
    int64_t v);

/* Append float to buffer.
 * Returns false when max is reached, true when there is still space */
FLECS_API
void ecs_strbuf_appendflt(
    ecs_strbuf_t *buffer,
    double v,
    char nan_delim);

/* Append boolean to buffer.
 * Returns false when max is reached, true when there is still space */
FLECS_API
void ecs_strbuf_appendbool(
    ecs_strbuf_t *buffer,
    bool v);

/* Append source buffer to destination buffer.
 * Returns false when max is reached, true when there is still space */
FLECS_API
void ecs_strbuf_mergebuff(
    ecs_strbuf_t *dst_buffer,
    ecs_strbuf_t *src_buffer);

/* Append n characters to buffer.
 * Returns false when max is reached, true when there is still space */
FLECS_API
void ecs_strbuf_appendstrn(
    ecs_strbuf_t *buffer,
    const char *str,
    int32_t n);

/* Return result string */
FLECS_API
char* ecs_strbuf_get(
    ecs_strbuf_t *buffer);

/* Return small string from first element (appends \0) */
FLECS_API
char* ecs_strbuf_get_small(
    ecs_strbuf_t *buffer);

/* Reset buffer without returning a string */
FLECS_API
void ecs_strbuf_reset(
    ecs_strbuf_t *buffer);

/* Push a list */
FLECS_API
void ecs_strbuf_list_push(
    ecs_strbuf_t *buffer,
    const char *list_open,
    const char *separator);

/* Pop a new list */
FLECS_API
void ecs_strbuf_list_pop(
    ecs_strbuf_t *buffer,
    const char *list_close);

/* Insert a new element in list */
FLECS_API
void ecs_strbuf_list_next(
    ecs_strbuf_t *buffer);

/* Append character to as new element in list. */
FLECS_API
void ecs_strbuf_list_appendch(
    ecs_strbuf_t *buffer,
    char ch);

/* Append formatted string as a new element in list */
FLECS_API
void ecs_strbuf_list_append(
    ecs_strbuf_t *buffer,
    const char *fmt,
    ...);

/* Append string as a new element in list */
FLECS_API
void ecs_strbuf_list_appendstr(
    ecs_strbuf_t *buffer,
    const char *str);

/* Append string as a new element in list */
FLECS_API
void ecs_strbuf_list_appendstrn(
    ecs_strbuf_t *buffer,
    const char *str,
    int32_t n);

FLECS_API
int32_t ecs_strbuf_written(
    const ecs_strbuf_t *buffer);

#define ecs_strbuf_appendlit(buf, str)\
    ecs_strbuf_appendstrn(buf, str, (int32_t)(sizeof(str) - 1))

#define ecs_strbuf_list_appendlit(buf, str)\
    ecs_strbuf_list_appendstrn(buf, str, (int32_t)(sizeof(str) - 1))

#ifdef __cplusplus
}
#endif

#endif

/**
 * @file os_api.h
 * @brief Operating system abstraction API.
 *
 * This file contains the operating system abstraction API. The flecs core
 * library avoids OS/runtime specific API calls as much as possible. Instead it
 * provides an interface that can be implemented by applications.
 *
 * Examples for how to implement this interface can be found in the
 * examples/os_api folder.
 */

#ifndef FLECS_OS_API_H
#define FLECS_OS_API_H

/**
 * @defgroup c_os_api OS API
 * @ingroup c
 * Interface for providing OS specific functionality.
 *
 * @{
 */

#include <stdarg.h>
#include <errno.h>
#include <stdio.h>

#if defined(ECS_TARGET_WINDOWS)
#include <malloc.h>
#elif defined(ECS_TARGET_FREEBSD)
#include <stdlib.h>
#else
#include <alloca.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** Time type. */
typedef struct ecs_time_t {
    uint32_t sec;                                 /**< Second part. */
    uint32_t nanosec;                             /**< Nanosecond part. */
} ecs_time_t;

/* Allocation counters */
extern int64_t ecs_os_api_malloc_count;            /**< malloc count. */
extern int64_t ecs_os_api_realloc_count;           /**< realloc count. */
extern int64_t ecs_os_api_calloc_count;            /**< calloc count. */
extern int64_t ecs_os_api_free_count;              /**< free count. */

/* Use handle types that _at least_ can store pointers */
typedef uintptr_t ecs_os_thread_t;                 /**< OS thread. */
typedef uintptr_t ecs_os_cond_t;                   /**< OS cond. */
typedef uintptr_t ecs_os_mutex_t;                  /**< OS mutex. */
typedef uintptr_t ecs_os_dl_t;                     /**< OS dynamic library. */
typedef uintptr_t ecs_os_sock_t;                   /**< OS socket. */

/** 64 bit thread id. */
typedef uint64_t ecs_os_thread_id_t;

/** Generic function pointer type. */
typedef void (*ecs_os_proc_t)(void);

/** OS API init. */
typedef
void (*ecs_os_api_init_t)(void);

/** OS API deinit. */
typedef
void (*ecs_os_api_fini_t)(void);

/** OS API malloc function type. */
typedef
void* (*ecs_os_api_malloc_t)(
    ecs_size_t size);

/** OS API free function type. */
typedef
void (*ecs_os_api_free_t)(
    void *ptr);

/** OS API realloc function type. */
typedef
void* (*ecs_os_api_realloc_t)(
    void *ptr,
    ecs_size_t size);

/** OS API calloc function type. */
typedef
void* (*ecs_os_api_calloc_t)(
    ecs_size_t size);

/** OS API strdup function type. */
typedef
char* (*ecs_os_api_strdup_t)(
    const char *str);

/** OS API thread_callback function type. */
typedef
void* (*ecs_os_thread_callback_t)(
    void*);

/** OS API thread_new function type. */
typedef
ecs_os_thread_t (*ecs_os_api_thread_new_t)(
    ecs_os_thread_callback_t callback,
    void *param);

/** OS API thread_join function type. */
typedef
void* (*ecs_os_api_thread_join_t)(
    ecs_os_thread_t thread);

/** OS API thread_self function type. */
typedef
ecs_os_thread_id_t (*ecs_os_api_thread_self_t)(void);

/** OS API task_new function type. */
typedef
ecs_os_thread_t (*ecs_os_api_task_new_t)(
    ecs_os_thread_callback_t callback,
    void *param);

/** OS API task_join function type. */
typedef
void* (*ecs_os_api_task_join_t)(
    ecs_os_thread_t thread);

/* Atomic increment / decrement */
/** OS API ainc function type. */
typedef
int32_t (*ecs_os_api_ainc_t)(
    int32_t *value);

/** OS API lainc function type. */
typedef
int64_t (*ecs_os_api_lainc_t)(
    int64_t *value);

/* Mutex */
/** OS API mutex_new function type. */
typedef
ecs_os_mutex_t (*ecs_os_api_mutex_new_t)(
    void);

/** OS API mutex_lock function type. */
typedef
void (*ecs_os_api_mutex_lock_t)(
    ecs_os_mutex_t mutex);

/** OS API mutex_unlock function type. */
typedef
void (*ecs_os_api_mutex_unlock_t)(
    ecs_os_mutex_t mutex);

/** OS API mutex_free function type. */
typedef
void (*ecs_os_api_mutex_free_t)(
    ecs_os_mutex_t mutex);

/* Condition variable */
/** OS API cond_new function type. */
typedef
ecs_os_cond_t (*ecs_os_api_cond_new_t)(
    void);

/** OS API cond_free function type. */
typedef
void (*ecs_os_api_cond_free_t)(
    ecs_os_cond_t cond);

/** OS API cond_signal function type. */
typedef
void (*ecs_os_api_cond_signal_t)(
    ecs_os_cond_t cond);

/** OS API cond_broadcast function type. */
typedef
void (*ecs_os_api_cond_broadcast_t)(
    ecs_os_cond_t cond);

/** OS API cond_wait function type. */
typedef
void (*ecs_os_api_cond_wait_t)(
    ecs_os_cond_t cond,
    ecs_os_mutex_t mutex);

/** OS API sleep function type. */
typedef
void (*ecs_os_api_sleep_t)(
    int32_t sec,
    int32_t nanosec);

/** OS API enable_high_timer_resolution function type. */
typedef
void (*ecs_os_api_enable_high_timer_resolution_t)(
    bool enable);

/** OS API get_time function type. */
typedef
void (*ecs_os_api_get_time_t)(
    ecs_time_t *time_out);

/** OS API now function type. */
typedef
uint64_t (*ecs_os_api_now_t)(void);

/** OS API log function type. */
typedef
void (*ecs_os_api_log_t)(
    int32_t level,     /* Logging level */
    const char *file,  /* File where message was logged */
    int32_t line,      /* Line it was logged */
    const char *msg);

/** OS API abort function type. */
typedef
void (*ecs_os_api_abort_t)(
    void);

/** OS API dlopen function type. */
typedef
ecs_os_dl_t (*ecs_os_api_dlopen_t)(
    const char *libname);

/** OS API dlproc function type. */
typedef
ecs_os_proc_t (*ecs_os_api_dlproc_t)(
    ecs_os_dl_t lib,
    const char *procname);

/** OS API dlclose function type. */
typedef
void (*ecs_os_api_dlclose_t)(
    ecs_os_dl_t lib);

/** OS API module_to_path function type. */
typedef
char* (*ecs_os_api_module_to_path_t)(
    const char *module_id);

/* Performance tracing */
typedef void (*ecs_os_api_perf_trace_t)(
    const char *filename,
    size_t line,
    const char *name);

/* Prefix members of struct with 'ecs_' as some system headers may define
 * macros for functions like "strdup", "log" or "_free" */

/** OS API interface. */
typedef struct ecs_os_api_t {
    /* API init / deinit */
    ecs_os_api_init_t init_;                       /**< init callback. */
    ecs_os_api_fini_t fini_;                       /**< fini callback. */

    /* Memory management */
    ecs_os_api_malloc_t malloc_;                   /**< malloc callback. */
    ecs_os_api_realloc_t realloc_;                 /**< realloc callback. */
    ecs_os_api_calloc_t calloc_;                   /**< calloc callback. */
    ecs_os_api_free_t free_;                       /**< free callback. */

    /* Strings */
    ecs_os_api_strdup_t strdup_;                   /**< strdup callback. */

    /* Threads */
    ecs_os_api_thread_new_t thread_new_;           /**< thread_new callback. */
    ecs_os_api_thread_join_t thread_join_;         /**< thread_join callback. */
    ecs_os_api_thread_self_t thread_self_;         /**< thread_self callback. */

    /* Tasks */
    ecs_os_api_thread_new_t task_new_;             /**< task_new callback. */
    ecs_os_api_thread_join_t task_join_;           /**< task_join callback. */

    /* Atomic increment / decrement */
    ecs_os_api_ainc_t ainc_;                       /**< ainc callback. */
    ecs_os_api_ainc_t adec_;                       /**< adec callback. */
    ecs_os_api_lainc_t lainc_;                     /**< lainc callback. */
    ecs_os_api_lainc_t ladec_;                     /**< ladec callback. */

    /* Mutex */
    ecs_os_api_mutex_new_t mutex_new_;             /**< mutex_new callback. */
    ecs_os_api_mutex_free_t mutex_free_;           /**< mutex_free callback. */
    ecs_os_api_mutex_lock_t mutex_lock_;           /**< mutex_lock callback. */
    ecs_os_api_mutex_lock_t mutex_unlock_;         /**< mutex_unlock callback. */

    /* Condition variable */
    ecs_os_api_cond_new_t cond_new_;               /**< cond_new callback. */
    ecs_os_api_cond_free_t cond_free_;             /**< cond_free callback. */
    ecs_os_api_cond_signal_t cond_signal_;         /**< cond_signal callback. */
    ecs_os_api_cond_broadcast_t cond_broadcast_;   /**< cond_broadcast callback. */
    ecs_os_api_cond_wait_t cond_wait_;             /**< cond_wait callback. */

    /* Time */
    ecs_os_api_sleep_t sleep_;                     /**< sleep callback. */
    ecs_os_api_now_t now_;                         /**< now callback. */
    ecs_os_api_get_time_t get_time_;               /**< get_time callback. */

    /* Logging */
    ecs_os_api_log_t log_; /**< log callback.
                            * The level should be interpreted as:
                            * >0: Debug tracing. Only enabled in debug builds.
                            *  0: Tracing. Enabled in debug/release builds.
                            * -2: Warning. An issue occurred, but operation was successful.
                            * -3: Error. An issue occurred, and operation was unsuccessful.
                            * -4: Fatal. An issue occurred, and application must quit. */

    /* Application termination */
    ecs_os_api_abort_t abort_;                     /**< abort callback. */

    /* Dynamic library loading */
    ecs_os_api_dlopen_t dlopen_;                   /**< dlopen callback. */
    ecs_os_api_dlproc_t dlproc_;                   /**< dlproc callback. */
    ecs_os_api_dlclose_t dlclose_;                 /**< dlclose callback. */

    /* Overridable function that translates from a logical module id to a
     * shared library filename */
    ecs_os_api_module_to_path_t module_to_dl_;     /**< module_to_dl callback. */

    /* Overridable function that translates from a logical module id to a
     * path that contains module-specif resources or assets */
    ecs_os_api_module_to_path_t module_to_etc_;    /**< module_to_etc callback. */

    /* Performance tracing */
    ecs_os_api_perf_trace_t perf_trace_push_;

    /* Performance tracing */
    ecs_os_api_perf_trace_t perf_trace_pop_;

    int32_t log_level_;                            /**< Tracing level. */
    int32_t log_indent_;                           /**< Tracing indentation level. */
    int32_t log_last_error_;                       /**< Last logged error code. */
    int64_t log_last_timestamp_;                   /**< Last logged timestamp. */

    ecs_flags32_t flags_;                          /**< OS API flags */

    FILE *log_out_;                                /**< File used for logging output 
                                                    * (hint, log_ decides where to write) */
} ecs_os_api_t;

/** Static OS API variable with configured callbacks. */
FLECS_API
extern ecs_os_api_t ecs_os_api;

/** Initialize OS API. 
 * This operation is not usually called by an application. To override callbacks
 * of the OS API, use the following pattern:
 * 
 * @code
 * ecs_os_set_api_defaults();
 * ecs_os_api_t os_api = ecs_os_get_api();
 * os_api.abort_ = my_abort;
 * ecs_os_set_api(&os_api);
 * @endcode
 */
FLECS_API
void ecs_os_init(void);

/** Deinitialize OS API. 
 * This operation is not usually called by an application.
 */
FLECS_API
void ecs_os_fini(void);

/** Override OS API.
 * This overrides the OS API struct with new values for callbacks. See 
 * ecs_os_init() on how to use the function.
 * 
 * @param os_api Pointer to struct with values to set.
 */
FLECS_API
void ecs_os_set_api(
    ecs_os_api_t *os_api);

/** Get OS API. 
 * 
 * @return A value with the current OS API callbacks 
 * @see ecs_os_init()
 */
FLECS_API
ecs_os_api_t ecs_os_get_api(void);

/** Set default values for OS API.
 * This initializes the OS API struct with default values for callbacks like
 * malloc and free.
 * 
 * @see ecs_os_init()
 */
FLECS_API
void ecs_os_set_api_defaults(void);

/** Macro utilities 
 * \cond
 */

/* Memory management */
#ifndef ecs_os_malloc
#define ecs_os_malloc(size) ecs_os_api.malloc_(size)
#endif
#ifndef ecs_os_free
#define ecs_os_free(ptr) ecs_os_api.free_(ptr)
#endif
#ifndef ecs_os_realloc
#define ecs_os_realloc(ptr, size) ecs_os_api.realloc_(ptr, size)
#endif
#ifndef ecs_os_calloc
#define ecs_os_calloc(size) ecs_os_api.calloc_(size)
#endif
#if defined(ECS_TARGET_WINDOWS)
#define ecs_os_alloca(size) _alloca((size_t)(size))
#else
#define ecs_os_alloca(size) alloca((size_t)(size))
#endif

#define ecs_os_malloc_t(T) ECS_CAST(T*, ecs_os_malloc(ECS_SIZEOF(T)))
#define ecs_os_malloc_n(T, count) ECS_CAST(T*, ecs_os_malloc(ECS_SIZEOF(T) * (count)))
#define ecs_os_calloc_t(T) ECS_CAST(T*, ecs_os_calloc(ECS_SIZEOF(T)))
#define ecs_os_calloc_n(T, count) ECS_CAST(T*, ecs_os_calloc(ECS_SIZEOF(T) * (count)))

#define ecs_os_realloc_t(ptr, T) ECS_CAST(T*, ecs_os_realloc(ptr, ECS_SIZEOF(T)))
#define ecs_os_realloc_n(ptr, T, count) ECS_CAST(T*, ecs_os_realloc(ptr, ECS_SIZEOF(T) * (count)))
#define ecs_os_alloca_t(T) ECS_CAST(T*, ecs_os_alloca(ECS_SIZEOF(T)))
#define ecs_os_alloca_n(T, count) ECS_CAST(T*, ecs_os_alloca(ECS_SIZEOF(T) * (count)))

/* Strings */
#ifndef ecs_os_strdup
#define ecs_os_strdup(str) ecs_os_api.strdup_(str)
#endif

#ifdef __cplusplus
#define ecs_os_strlen(str) static_cast<ecs_size_t>(strlen(str))
#define ecs_os_strncmp(str1, str2, num) strncmp(str1, str2, static_cast<size_t>(num))
#define ecs_os_memcmp(ptr1, ptr2, num) memcmp(ptr1, ptr2, static_cast<size_t>(num))
#define ecs_os_memcpy(ptr1, ptr2, num) memcpy(ptr1, ptr2, static_cast<size_t>(num))
#define ecs_os_memset(ptr, value, num) memset(ptr, value, static_cast<size_t>(num))
#define ecs_os_memmove(dst, src, size) memmove(dst, src, static_cast<size_t>(size))
#else
#define ecs_os_strlen(str) (ecs_size_t)strlen(str)
#define ecs_os_strncmp(str1, str2, num) strncmp(str1, str2, (size_t)(num))
#define ecs_os_memcmp(ptr1, ptr2, num) memcmp(ptr1, ptr2, (size_t)(num))
#define ecs_os_memcpy(ptr1, ptr2, num) memcpy(ptr1, ptr2, (size_t)(num))
#define ecs_os_memset(ptr, value, num) memset(ptr, value, (size_t)(num))
#define ecs_os_memmove(dst, src, size) memmove(dst, src, (size_t)(size))
#endif

#define ecs_os_memcpy_t(ptr1, ptr2, T) ecs_os_memcpy(ptr1, ptr2, ECS_SIZEOF(T))
#define ecs_os_memcpy_n(ptr1, ptr2, T, count) ecs_os_memcpy(ptr1, ptr2, ECS_SIZEOF(T) * (size_t)count)
#define ecs_os_memcmp_t(ptr1, ptr2, T) ecs_os_memcmp(ptr1, ptr2, ECS_SIZEOF(T))

#define ecs_os_memmove_t(ptr1, ptr2, T) ecs_os_memmove(ptr1, ptr2, ECS_SIZEOF(T))
#define ecs_os_memmove_n(ptr1, ptr2, T, count) ecs_os_memmove(ptr1, ptr2, ECS_SIZEOF(T) * (size_t)count)
#define ecs_os_memmove_t(ptr1, ptr2, T) ecs_os_memmove(ptr1, ptr2, ECS_SIZEOF(T))

#define ecs_os_strcmp(str1, str2) strcmp(str1, str2)
#define ecs_os_memset_t(ptr, value, T) ecs_os_memset(ptr, value, ECS_SIZEOF(T))
#define ecs_os_memset_n(ptr, value, T, count) ecs_os_memset(ptr, value, ECS_SIZEOF(T) * (size_t)count)
#define ecs_os_zeromem(ptr) ecs_os_memset(ptr, 0, ECS_SIZEOF(*ptr))

#define ecs_os_memdup_t(ptr, T) ecs_os_memdup(ptr, ECS_SIZEOF(T))
#define ecs_os_memdup_n(ptr, T, count) ecs_os_memdup(ptr, ECS_SIZEOF(T) * count)

#define ecs_offset(ptr, T, index)\
    ECS_CAST(T*, ECS_OFFSET(ptr, ECS_SIZEOF(T) * index))

#if !defined(ECS_TARGET_POSIX) && !defined(ECS_TARGET_MINGW)
#define ecs_os_strcat(str1, str2) strcat_s(str1, INT_MAX, str2)
#define ecs_os_snprintf(ptr, len, ...) sprintf_s(ptr, ECS_CAST(size_t, len), __VA_ARGS__)
#define ecs_os_vsnprintf(ptr, len, fmt, args) vsnprintf(ptr, ECS_CAST(size_t, len), fmt, args)
#define ecs_os_strcpy(str1, str2) strcpy_s(str1, INT_MAX, str2)
#define ecs_os_strncpy(str1, str2, len) strncpy_s(str1, INT_MAX, str2, ECS_CAST(size_t, len))
#else
#define ecs_os_strcat(str1, str2) strcat(str1, str2)
#define ecs_os_snprintf(ptr, len, ...) snprintf(ptr, ECS_CAST(size_t, len), __VA_ARGS__)
#define ecs_os_vsnprintf(ptr, len, fmt, args) vsnprintf(ptr, ECS_CAST(size_t, len), fmt, args)
#define ecs_os_strcpy(str1, str2) strcpy(str1, str2)
#define ecs_os_strncpy(str1, str2, len) strncpy(str1, str2, ECS_CAST(size_t, len))
#endif

/* Files */
#ifndef ECS_TARGET_POSIX
#define ecs_os_fopen(result, file, mode) fopen_s(result, file, mode)
#else
#define ecs_os_fopen(result, file, mode) (*(result)) = fopen(file, mode)
#endif

/* Threads */
#define ecs_os_thread_new(callback, param) ecs_os_api.thread_new_(callback, param)
#define ecs_os_thread_join(thread) ecs_os_api.thread_join_(thread)
#define ecs_os_thread_self() ecs_os_api.thread_self_()

/* Tasks */
#define ecs_os_task_new(callback, param) ecs_os_api.task_new_(callback, param)
#define ecs_os_task_join(thread) ecs_os_api.task_join_(thread)

/* Atomic increment / decrement */
#define ecs_os_ainc(value) ecs_os_api.ainc_(value)
#define ecs_os_adec(value) ecs_os_api.adec_(value)
#define ecs_os_lainc(value) ecs_os_api.lainc_(value)
#define ecs_os_ladec(value) ecs_os_api.ladec_(value)

/* Mutex */
#define ecs_os_mutex_new() ecs_os_api.mutex_new_()
#define ecs_os_mutex_free(mutex) ecs_os_api.mutex_free_(mutex)
#define ecs_os_mutex_lock(mutex) ecs_os_api.mutex_lock_(mutex)
#define ecs_os_mutex_unlock(mutex) ecs_os_api.mutex_unlock_(mutex)

/* Condition variable */
#define ecs_os_cond_new() ecs_os_api.cond_new_()
#define ecs_os_cond_free(cond) ecs_os_api.cond_free_(cond)
#define ecs_os_cond_signal(cond) ecs_os_api.cond_signal_(cond)
#define ecs_os_cond_broadcast(cond) ecs_os_api.cond_broadcast_(cond)
#define ecs_os_cond_wait(cond, mutex) ecs_os_api.cond_wait_(cond, mutex)

/* Time */
#define ecs_os_sleep(sec, nanosec) ecs_os_api.sleep_(sec, nanosec)
#define ecs_os_now() ecs_os_api.now_()
#define ecs_os_get_time(time_out) ecs_os_api.get_time_(time_out)

#ifndef FLECS_DISABLE_COUNTERS
#ifdef FLECS_ACCURATE_COUNTERS
#define ecs_os_inc(v)  (ecs_os_ainc(v))
#define ecs_os_linc(v) (ecs_os_lainc(v))
#define ecs_os_dec(v)  (ecs_os_adec(v))
#define ecs_os_ldec(v) (ecs_os_ladec(v))
#else
#define ecs_os_inc(v)  (++(*v))
#define ecs_os_linc(v) (++(*v))
#define ecs_os_dec(v)  (--(*v))
#define ecs_os_ldec(v) (--(*v))
#endif
#else
#define ecs_os_inc(v)
#define ecs_os_linc(v)
#define ecs_os_dec(v)
#define ecs_os_ldec(v)
#endif


#ifdef ECS_TARGET_MINGW
/* mingw bug: without this a conversion error is thrown, but isnan/isinf should
 * accept float, double and long double. */
#define ecs_os_isnan(val) (isnan((float)val))
#define ecs_os_isinf(val) (isinf((float)val))
#else
#define ecs_os_isnan(val) (isnan(val))
#define ecs_os_isinf(val) (isinf(val))
#endif

/* Application termination */
#define ecs_os_abort() ecs_os_api.abort_()

/* Dynamic libraries */
#define ecs_os_dlopen(libname) ecs_os_api.dlopen_(libname)
#define ecs_os_dlproc(lib, procname) ecs_os_api.dlproc_(lib, procname)
#define ecs_os_dlclose(lib) ecs_os_api.dlclose_(lib)

/* Module id translation */
#define ecs_os_module_to_dl(lib) ecs_os_api.module_to_dl_(lib)
#define ecs_os_module_to_etc(lib) ecs_os_api.module_to_etc_(lib)

/** Macro utilities 
 * \endcond
 */


/* Logging */

/** Log at debug level.
 * 
 * @param file The file to log.
 * @param line The line to log.
 * @param msg The message to log.
*/
FLECS_API
void ecs_os_dbg(
    const char *file, 
    int32_t line, 
    const char *msg);

/** Log at trace level.
 * 
 * @param file The file to log.
 * @param line The line to log.
 * @param msg The message to log.
*/
FLECS_API
void ecs_os_trace(
    const char *file, 
    int32_t line, 
    const char *msg);

/** Log at warning level.
 * 
 * @param file The file to log.
 * @param line The line to log.
 * @param msg The message to log.
*/
FLECS_API
void ecs_os_warn(
    const char *file, 
    int32_t line, 
    const char *msg);

/** Log at error level.
 * 
 * @param file The file to log.
 * @param line The line to log.
 * @param msg The message to log.
*/
FLECS_API
void ecs_os_err(
    const char *file, 
    int32_t line, 
    const char *msg);

/** Log at fatal level.
 * 
 * @param file The file to log.
 * @param line The line to log.
 * @param msg The message to log.
*/
FLECS_API
void ecs_os_fatal(
    const char *file, 
    int32_t line, 
    const char *msg);

/** Convert errno to string.
 * 
 * @param err The error number.
 * @return A string describing the error.
 */
FLECS_API
const char* ecs_os_strerror(
    int err);

/** Utility for assigning strings. 
 * This operation frees an existing string and duplicates the input string.
 * 
 * @param str Pointer to a string value.
 * @param value The string value to assign.
 */
FLECS_API
void ecs_os_strset(
    char **str, 
    const char *value);

/* Profile tracing */
#ifdef FLECS_PERF_TRACE
#define ecs_os_perf_trace_push(name) ecs_os_perf_trace_push_(__FILE__, __LINE__, name)
#define ecs_os_perf_trace_pop(name) ecs_os_perf_trace_pop_(__FILE__, __LINE__, name)
#else
#define ecs_os_perf_trace_push(name)
#define ecs_os_perf_trace_pop(name)
#endif

void ecs_os_perf_trace_push_(
    const char *file,
    size_t line,
    const char *name);

void ecs_os_perf_trace_pop_(
    const char *file,
    size_t line,
    const char *name);

/** Sleep with floating point time. 
 * 
 * @param t The time in seconds.
 */
FLECS_API
void ecs_sleepf(
    double t);

/** Measure time since provided timestamp. 
 * Use with a time value initialized to 0 to obtain the number of seconds since
 * the epoch. The operation will write the current timestamp in start.
 * 
 * Usage:
 * @code
 * ecs_time_t t = {};
 * ecs_time_measure(&t);
 * // code
 * double elapsed = ecs_time_measure(&t);
 * @endcode
 * 
 * @param start The starting timestamp.
 * @return The time elapsed since start.
 */
FLECS_API
double ecs_time_measure(
    ecs_time_t *start);

/** Calculate difference between two timestamps. 
 * 
 * @param t1 The first timestamp.
 * @param t2 The first timestamp.
 * @return The difference between timestamps.
 */
FLECS_API
ecs_time_t ecs_time_sub(
    ecs_time_t t1,
    ecs_time_t t2);

/** Convert time value to a double. 
 * 
 * @param t The timestamp.
 * @return The timestamp converted to a double.
 */
FLECS_API
double ecs_time_to_double(
    ecs_time_t t);

/** Return newly allocated memory that contains a copy of src. 
 * 
 * @param src The source pointer.
 * @param size The number of bytes to copy.
 * @return The duplicated memory.
 */
FLECS_API
void* ecs_os_memdup(
    const void *src,
    ecs_size_t size);

/** Are heap functions available? */
FLECS_API
bool ecs_os_has_heap(void);

/** Are threading functions available? */
FLECS_API
bool ecs_os_has_threading(void);

/** Are task functions available? */
FLECS_API
bool ecs_os_has_task_support(void);

/** Are time functions available? */
FLECS_API
bool ecs_os_has_time(void);

/** Are logging functions available? */
FLECS_API
bool ecs_os_has_logging(void);

/** Are dynamic library functions available? */
FLECS_API
bool ecs_os_has_dl(void);

/** Are module path functions available? */
FLECS_API
bool ecs_os_has_modules(void);

#ifdef __cplusplus
}
#endif

/** @} */

#endif


#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup api_types API types
 * Public API types.
 *
 * @{
 */

/**
 * @defgroup core_types Core API Types
 * Types for core API objects.
 *
 * @{
 */

/** Ids are the things that can be added to an entity.
 * An id can be an entity or pair, and can have optional id flags. */
typedef uint64_t ecs_id_t;

/** An entity identifier.
 * Entity ids consist out of a number unique to the entity in the lower 32 bits,
 * and a counter used to track entity liveliness in the upper 32 bits. When an
 * id is recycled, its generation count is increased. This causes recycled ids
 * to be very large (>4 billion), which is normal. */
typedef ecs_id_t ecs_entity_t;

/** A type is a list of (component) ids.
 * Types are used to communicate the "type" of an entity. In most type systems a
 * typeof operation returns a single type. In ECS however, an entity can have
 * multiple components, which is why an ECS type consists of a vector of ids.
 *
 * The component ids of a type are sorted, which ensures that it doesn't matter
 * in which order components are added to an entity. For example, if adding
 * Position then Velocity would result in type [Position, Velocity], first
 * adding Velocity then Position would also result in type [Position, Velocity].
 *
 * Entities are grouped together by type in the ECS storage in tables. The
 * storage has exactly one table per unique type that is created by the
 * application that stores all entities and components for that type. This is
 * also referred to as an archetype.
 */
typedef struct {
    ecs_id_t *array;    /**< Array with ids. */
    int32_t count;      /**< Number of elements in array. */
} ecs_type_t;

/** A world is the container for all ECS data and supporting features.
 * Applications can have multiple worlds, though in most cases will only need
 * one. Worlds are isolated from each other, and can have separate sets of
 * systems, components, modules etc.
 *
 * If an application has multiple worlds with overlapping components, it is
 * common (though not strictly required) to use the same component ids across
 * worlds, which can be achieved by declaring a global component id variable.
 * To do this in the C API, see the entities/fwd_component_decl example. The
 * C++ API automatically synchronizes component ids between worlds.
 *
 * Component id conflicts between worlds can occur when a world has already used
 * an id for something else. There are a few ways to avoid this:
 *
 * - Ensure to register the same components in each world, in the same order.
 * - Create a dummy world in which all components are preregistered which
 *   initializes the global id variables.
 *
 * In some use cases, typically when writing tests, multiple worlds are created
 * and deleted with different components, registered in different order. To
 * ensure isolation between tests, the C++ API has a `flecs::reset` function
 * that forces the API to ignore the old component ids. */
typedef struct ecs_world_t ecs_world_t;

/** A stage enables modification while iterating and from multiple threads */
typedef struct ecs_stage_t ecs_stage_t;

/** A table stores entities and components for a specific type. */
typedef struct ecs_table_t ecs_table_t;

/** A term is a single element in a query. */
typedef struct ecs_term_t ecs_term_t;

/** A query returns entities matching a list of constraints. */
typedef struct ecs_query_t ecs_query_t;

/** An observer is a system that is invoked when an event matches its query.
 * Observers allow applications to respond to specific events, such as adding or
 * removing a component. Observers are created by both specifying a query and
 * a list of event kinds that should be listened for. An example of an observer
 * that triggers when a Position component is added to an entity (in C++):
 *
 * @code
 * world.observer<Position>()
 *   .event(flecs::OnAdd)
 *   .each([](Position& p) {
 *     // called when Position is added to an entity
 *   });
 * @endcode
 *
 * Observers only trigger when the source of the event matches the full observer 
 * query. For example, an OnAdd observer for Position, Velocity will only 
 * trigger after both components have been added to the entity. */
typedef struct ecs_observer_t ecs_observer_t;

/** An observable produces events that can be listened for by an observer.
 * Currently only the world is observable. In the future, queries will become
 * observable objects as well. */
typedef struct ecs_observable_t ecs_observable_t;

/** Type used for iterating iterable objects.
 * Iterators are objects that provide applications with information
 * about the currently iterated result, and store any state required for the
 * iteration. */
typedef struct ecs_iter_t ecs_iter_t;

/** A ref is a fast way to fetch a component for a specific entity.
 * Refs are a faster alternative to repeatedly calling ecs_get() for the same
 * entity/component combination. When comparing the performance of getting a ref
 * to calling ecs_get(), a ref is typically 3-5x faster.
 *
 * Refs achieve this performance by caching internal data structures associated
 * with the entity and component on the ecs_ref_t object that otherwise would
 * have to be looked up. */
typedef struct ecs_ref_t ecs_ref_t;

/** Type hooks are callbacks associated with component lifecycle events.
 * Typical examples of lifecycle events are construction, destruction, copying
 * and moving of components. */
typedef struct ecs_type_hooks_t ecs_type_hooks_t;

/** Type information.
 * Contains information about a (component) type, such as its size and
 * alignment and type hooks. */
typedef struct ecs_type_info_t ecs_type_info_t;

/** Information about an entity, like its table and row. */
typedef struct ecs_record_t ecs_record_t;

/** Information about a (component) id, such as type info and tables with the id */
typedef struct ecs_id_record_t ecs_id_record_t;

/** A poly object.
 * A poly (short for polymorph) object is an object that has a variable list of
 * capabilities, determined by a mixin table. This is the current list of types
 * in the flecs API that can be used as an ecs_poly_t:
 *
 * - ecs_world_t
 * - ecs_stage_t
 * - ecs_query_t
 *
 * Functions that accept an ecs_poly_t argument can accept objects of these
 * types. If the object does not have the requested mixin the API will throw an
 * assert.
 *
 * The poly/mixin framework enables partially overlapping features to be
 * implemented once, and enables objects of different types to interact with
 * each other depending on what mixins they have, rather than their type
 * (in some ways it's like a mini-ECS). Additionally, each poly object has a
 * header that enables the API to do sanity checking on the input arguments.
 */
typedef void ecs_poly_t;

/** Type that stores poly mixins */
typedef struct ecs_mixins_t ecs_mixins_t;

/** Header for ecs_poly_t objects. */
typedef struct ecs_header_t {
    int32_t magic;              /**< Magic number verifying it's a flecs object */
    int32_t type;               /**< Magic number indicating which type of flecs object */
    int32_t refcount;           /**< Refcount, to enable RAII handles */
    ecs_mixins_t *mixins;       /**< Table with offsets to (optional) mixins */
} ecs_header_t;

/** Record for entity index */
struct ecs_record_t {
    ecs_id_record_t *idr;       /**< Id record to (*, entity) for target entities */
    ecs_table_t *table;         /**< Identifies a type (and table) in world */
    uint32_t row;               /**< Table row of the entity */
    int32_t dense;              /**< Index in dense array of entity index */    
};

/** Header for table cache elements. */
typedef struct ecs_table_cache_hdr_t {
    struct ecs_table_cache_t *cache;  /**< Table cache of element. Of type ecs_id_record_t* for component index elements. */
    ecs_table_t *table;               /**< Table associated with element. */
    struct ecs_table_cache_hdr_t *prev, *next; /**< Next/previous elements for id in table cache. */
} ecs_table_cache_hdr_t;

/** Metadata describing where a component id is stored in a table.
 * This type is used as element type for the component index table cache. One
 * record exists per table/component in the table. Only records for wildcard ids
 * can have a count > 1. */
typedef struct ecs_table_record_t {
    ecs_table_cache_hdr_t hdr;  /**< Table cache header */
    int16_t index;              /**< First type index where id occurs in table */
    int16_t count;              /**< Number of times id occurs in table */
    int16_t column;             /**< First column index where id occurs */
} ecs_table_record_t;

/** @} */

/**
 * @defgroup function_types Function types.
 * Function callback types.
 *
 * @{
 */

/** Function prototype for runnables (systems, observers).
 * The run callback overrides the default behavior for iterating through the
 * results of a runnable object.
 *
 * The default runnable iterates the iterator, and calls an iter_action (see
 * below) for each returned result.
 *
 * @param it The iterator to be iterated by the runnable.
 */
typedef void (*ecs_run_action_t)(
    ecs_iter_t *it);

/** Function prototype for iterables.
 * A system may invoke a callback multiple times, typically once for each
 * matched table.
 *
 * @param it The iterator containing the data for the current match.
 */
typedef void (*ecs_iter_action_t)(
    ecs_iter_t *it);

/** Function prototype for iterating an iterator.
 * Stored inside initialized iterators. This allows an application to iterate
 * an iterator without needing to know what created it.
 *
 * @param it The iterator to iterate.
 * @return True if iterator has no more results, false if it does.
 */
typedef bool (*ecs_iter_next_action_t)(
    ecs_iter_t *it);

/** Function prototype for freeing an iterator.
 * Free iterator resources.
 *
 * @param it The iterator to free.
 */
typedef void (*ecs_iter_fini_action_t)(
    ecs_iter_t *it);

/** Callback used for comparing components */
typedef int (*ecs_order_by_action_t)(
    ecs_entity_t e1,
    const void *ptr1,
    ecs_entity_t e2,
    const void *ptr2);

/** Callback used for sorting the entire table of components */
typedef void (*ecs_sort_table_action_t)(
    ecs_world_t* world,
    ecs_table_t* table,
    ecs_entity_t* entities,
    void* ptr,
    int32_t size,
    int32_t lo,
    int32_t hi,
    ecs_order_by_action_t order_by);

/** Callback used for grouping tables in a query */
typedef uint64_t (*ecs_group_by_action_t)(
    ecs_world_t *world,
    ecs_table_t *table,
    ecs_id_t group_id,
    void *ctx);

/** Callback invoked when a query creates a new group. */
typedef void* (*ecs_group_create_action_t)(
    ecs_world_t *world,
    uint64_t group_id,
    void *group_by_ctx); /* from ecs_query_desc_t */

/** Callback invoked when a query deletes an existing group. */
typedef void (*ecs_group_delete_action_t)(
    ecs_world_t *world,
    uint64_t group_id,
    void *group_ctx,     /* return value from ecs_group_create_action_t */
    void *group_by_ctx); /* from ecs_query_desc_t */

/** Initialization action for modules */
typedef void (*ecs_module_action_t)(
    ecs_world_t *world);

/** Action callback on world exit */
typedef void (*ecs_fini_action_t)(
    ecs_world_t *world,
    void *ctx);

/** Function to cleanup context data */
typedef void (*ecs_ctx_free_t)(
    void *ctx);

/** Callback used for sorting values */
typedef int (*ecs_compare_action_t)(
    const void *ptr1,
    const void *ptr2);

/** Callback used for hashing values */
typedef uint64_t (*ecs_hash_value_action_t)(
    const void *ptr);

/** Constructor/destructor callback */
typedef void (*ecs_xtor_t)(
    void *ptr,
    int32_t count,
    const ecs_type_info_t *type_info);

/** Copy is invoked when a component is copied into another component. */
typedef void (*ecs_copy_t)(
    void *dst_ptr,
    const void *src_ptr,
    int32_t count,
    const ecs_type_info_t *type_info);

/** Move is invoked when a component is moved to another component. */
typedef void (*ecs_move_t)(
    void *dst_ptr,
    void *src_ptr,
    int32_t count,
    const ecs_type_info_t *type_info);

/** Destructor function for poly objects. */
typedef void (*flecs_poly_dtor_t)(
    ecs_poly_t *poly);

/** @} */

/**
 * @defgroup query_types Query descriptor types.
 * Types used to describe queries.
 *
 * @{
 */

/** Specify read/write access for term */
typedef enum ecs_inout_kind_t {
    EcsInOutDefault,  /**< InOut for regular terms, In for shared terms */
    EcsInOutNone,     /**< Term is neither read nor written */
    EcsInOutFilter,   /**< Same as InOutNone + prevents term from triggering observers */
    EcsInOut,         /**< Term is both read and written */
    EcsIn,            /**< Term is only read */
    EcsOut,           /**< Term is only written */
} ecs_inout_kind_t;

/** Specify operator for term */
typedef enum ecs_oper_kind_t {
    EcsAnd,           /**< The term must match */
    EcsOr,            /**< One of the terms in an or chain must match */
    EcsNot,           /**< The term must not match */
    EcsOptional,      /**< The term may match */
    EcsAndFrom,       /**< Term must match all components from term id */
    EcsOrFrom,        /**< Term must match at least one component from term id */
    EcsNotFrom,       /**< Term must match none of the components from term id */
} ecs_oper_kind_t;

/** Specify cache policy for query */
typedef enum ecs_query_cache_kind_t {
    EcsQueryCacheDefault,   /**< Behavior determined by query creation context */
    EcsQueryCacheAuto,      /**< Cache query terms that are cacheable */
    EcsQueryCacheAll,       /**< Require that all query terms can be cached */
    EcsQueryCacheNone,      /**< No caching */
} ecs_query_cache_kind_t;

/* Term id flags  */

/** Match on self.
 * Can be combined with other term flags on the ecs_term_t::flags_ field.
 * \ingroup queries
 */
#define EcsSelf                       (1llu << 63)

/** Match by traversing upwards.
 * Can be combined with other term flags on the ecs_term_ref_t::id field.
 * \ingroup queries
 */
#define EcsUp                         (1llu << 62)

/** Traverse relationship transitively.
 * Can be combined with other term flags on the ecs_term_ref_t::id field.
 * \ingroup queries
 */
#define EcsTrav                       (1llu << 61)

/** Sort results breadth first.
 * Can be combined with other term flags on the ecs_term_ref_t::id field.
 * \ingroup queries
 */
#define EcsCascade                    (1llu << 60)

/** Iterate groups in descending order.
 * Can be combined with other term flags on the ecs_term_ref_t::id field.
 * \ingroup queries
 */
#define EcsDesc                       (1llu << 59)

/** Term id is a variable.
 * Can be combined with other term flags on the ecs_term_ref_t::id field.
 * \ingroup queries
 */
#define EcsIsVariable                 (1llu << 58)

/** Term id is an entity.
 * Can be combined with other term flags on the ecs_term_ref_t::id field.
 * \ingroup queries
 */
#define EcsIsEntity                   (1llu << 57)

/** Term id is a name (don't attempt to lookup as entity).
 * Can be combined with other term flags on the ecs_term_ref_t::id field.
 * \ingroup queries
 */
#define EcsIsName                     (1llu << 56)

/** All term traversal flags.
 * Can be combined with other term flags on the ecs_term_ref_t::id field.
 * \ingroup queries
 */
#define EcsTraverseFlags              (EcsSelf|EcsUp|EcsTrav|EcsCascade|EcsDesc)

/** All term reference kind flags.
 * Can be combined with other term flags on the ecs_term_ref_t::id field.
 * \ingroup queries
 */
#define EcsTermRefFlags               (EcsTraverseFlags|EcsIsVariable|EcsIsEntity|EcsIsName)

/** Type that describes a reference to an entity or variable in a term. */
typedef struct ecs_term_ref_t {
    ecs_entity_t id;            /**< Entity id. If left to 0 and flags does not 
                                 * specify whether id is an entity or a variable
                                 * the id will be initialized to #EcsThis.
                                 * To explicitly set the id to 0, leave the id
                                 * member to 0 and set #EcsIsEntity in flags. */

    const char *name;           /**< Name. This can be either the variable name
                                 * (when the #EcsIsVariable flag is set) or an
                                 * entity name. When ecs_term_t::move is true,
                                 * the API assumes ownership over the string and
                                 * will free it when the term is destroyed. */
} ecs_term_ref_t;

/** Type that describes a term (single element in a query). */
struct ecs_term_t {
    ecs_id_t id;                /**< Component id to be matched by term. Can be
                                 * set directly, or will be populated from the
                                 * first/second members, which provide more
                                 * flexibility. */

    ecs_term_ref_t src;          /**< Source of term */
    ecs_term_ref_t first;        /**< Component or first element of pair */
    ecs_term_ref_t second;       /**< Second element of pair */

    ecs_entity_t trav;          /**< Relationship to traverse when looking for the
                                 * component. The relationship must have
                                 * the `Traversable` property. Default is `IsA`. */

    int16_t inout;              /**< Access to contents matched by term */
    int16_t oper;               /**< Operator of term */

    int8_t field_index;         /**< Index of field for term in iterator */
    ecs_flags16_t flags_;       /**< Flags that help eval, set by ecs_query_init() */
};

/** Queries are lists of constraints (terms) that match entities. 
 * Created with ecs_query_init().
 */
struct ecs_query_t {
    ecs_header_t hdr;           /**< Object header */

    ecs_term_t terms[FLECS_TERM_COUNT_MAX]; /**< Query terms */
    int32_t sizes[FLECS_TERM_COUNT_MAX]; /**< Component sizes. Indexed by field */
    ecs_id_t ids[FLECS_TERM_COUNT_MAX]; /**< Component ids. Indexed by field */

    ecs_flags32_t flags;        /**< Query flags */
    int8_t var_count;           /**< Number of query variables */
    int8_t term_count;          /**< Number of query terms */
    int8_t field_count;         /**< Number of fields returned by query */

    /* Bitmasks for quick field information lookups */
    ecs_termset_t fixed_fields; /**< Fields with a fixed source */
    ecs_termset_t var_fields;   /**< Fields with non-$this variable source */
    ecs_termset_t static_id_fields; /**< Fields with a static (component) id */
    ecs_termset_t data_fields;  /**< Fields that have data */
    ecs_termset_t write_fields; /**< Fields that write data */
    ecs_termset_t read_fields;  /**< Fields that read data */
    ecs_termset_t row_fields;   /**< Fields that must be acquired with field_at */
    ecs_termset_t shared_readonly_fields; /**< Fields that don't write shared data */
    ecs_termset_t set_fields;   /**< Fields that will be set */

    ecs_query_cache_kind_t cache_kind;  /**< Caching policy of query */
    
    char **vars;                /**< Array with variable names for iterator */

    void *ctx;                  /**< User context to pass to callback */
    void *binding_ctx;          /**< Context to be used for language bindings */

    ecs_entity_t entity;        /**< Entity associated with query (optional) */
    ecs_world_t *real_world;    /**< Actual world. */
    ecs_world_t *world;         /**< World or stage query was created with. */

    int32_t eval_count;         /**< Number of times query is evaluated */
};

/** An observer reacts to events matching a query.
 * Created with ecs_observer_init().
 */
struct ecs_observer_t {
    ecs_header_t hdr;           /**< Object header */
    
    ecs_query_t *query;         /**< Observer query */

    /** Observer events */
    ecs_entity_t events[FLECS_EVENT_DESC_MAX];
    int32_t event_count;        /**< Number of events */

    ecs_iter_action_t callback; /**< See ecs_observer_desc_t::callback */
    ecs_run_action_t run;       /**< See ecs_observer_desc_t::run */

    void *ctx;                  /**< Observer context */
    void *callback_ctx;         /**< Callback language binding context */
    void *run_ctx;              /**< Run language binding context */

    ecs_ctx_free_t ctx_free;    /**< Callback to free ctx */
    ecs_ctx_free_t callback_ctx_free; /**< Callback to free callback_ctx */
    ecs_ctx_free_t run_ctx_free; /**< Callback to free run_ctx */

    ecs_observable_t *observable; /**< Observable for observer */

    ecs_world_t *world;         /**< The world */
    ecs_entity_t entity;        /**< Entity associated with observer */
};

/** @} */

/** Type that contains component lifecycle callbacks.
 *
 * @ingroup components
 */

/* Flags that can be used to check which hooks a type has set */
#define ECS_TYPE_HOOK_CTOR                   (1 << 0)
#define ECS_TYPE_HOOK_DTOR                   (1 << 1)
#define ECS_TYPE_HOOK_COPY                   (1 << 2)
#define ECS_TYPE_HOOK_MOVE                   (1 << 3)
#define ECS_TYPE_HOOK_COPY_CTOR              (1 << 4)
#define ECS_TYPE_HOOK_MOVE_CTOR              (1 << 5)
#define ECS_TYPE_HOOK_CTOR_MOVE_DTOR         (1 << 6)
#define ECS_TYPE_HOOK_MOVE_DTOR              (1 << 7)

/* Flags that can be used to set/check which hooks of a type are invalid */
#define ECS_TYPE_HOOK_CTOR_ILLEGAL           (1 << 8)
#define ECS_TYPE_HOOK_DTOR_ILLEGAL           (1 << 9)
#define ECS_TYPE_HOOK_COPY_ILLEGAL           (1 << 10)
#define ECS_TYPE_HOOK_MOVE_ILLEGAL           (1 << 11)
#define ECS_TYPE_HOOK_COPY_CTOR_ILLEGAL      (1 << 12)
#define ECS_TYPE_HOOK_MOVE_CTOR_ILLEGAL      (1 << 13)
#define ECS_TYPE_HOOK_CTOR_MOVE_DTOR_ILLEGAL (1 << 14)
#define ECS_TYPE_HOOK_MOVE_DTOR_ILLEGAL      (1 << 15)

/* All valid hook flags */
#define ECS_TYPE_HOOKS (ECS_TYPE_HOOK_CTOR|ECS_TYPE_HOOK_DTOR|\
    ECS_TYPE_HOOK_COPY|ECS_TYPE_HOOK_MOVE|ECS_TYPE_HOOK_COPY_CTOR|\
    ECS_TYPE_HOOK_MOVE_CTOR|ECS_TYPE_HOOK_CTOR_MOVE_DTOR|\
    ECS_TYPE_HOOK_MOVE_DTOR)

/* All invalid hook flags */
#define ECS_TYPE_HOOKS_ILLEGAL (ECS_TYPE_HOOK_CTOR_ILLEGAL|\
    ECS_TYPE_HOOK_DTOR_ILLEGAL|ECS_TYPE_HOOK_COPY_ILLEGAL|\
    ECS_TYPE_HOOK_MOVE_ILLEGAL|ECS_TYPE_HOOK_COPY_CTOR_ILLEGAL|\
    ECS_TYPE_HOOK_MOVE_CTOR_ILLEGAL|ECS_TYPE_HOOK_CTOR_MOVE_DTOR_ILLEGAL|\
    ECS_TYPE_HOOK_MOVE_DTOR_ILLEGAL)

struct ecs_type_hooks_t {
    ecs_xtor_t ctor;            /**< ctor */
    ecs_xtor_t dtor;            /**< dtor */
    ecs_copy_t copy;            /**< copy assignment */
    ecs_move_t move;            /**< move assignment */

    /** Ctor + copy */
    ecs_copy_t copy_ctor;

    /** Ctor + move */
    ecs_move_t move_ctor;

    /** Ctor + move + dtor (or move_ctor + dtor).
     * This combination is typically used when a component is moved from one
     * location to a new location, like when it is moved to a new table. If
     * not set explicitly it will be derived from other callbacks. */
    ecs_move_t ctor_move_dtor;

    /** Move + dtor.
     * This combination is typically used when a component is moved from one
     * location to an existing location, like what happens during a remove. If
     * not set explicitly it will be derived from other callbacks. */
    ecs_move_t move_dtor;

    /** Hook flags.
     * Indicates which hooks are set for the type, and which hooks are illegal.
     * When an ILLEGAL flag is set when calling ecs_set_hooks() a hook callback
     * will be set that panics when called. */
    ecs_flags32_t flags;

    /** Callback that is invoked when an instance of a component is added. This
     * callback is invoked before triggers are invoked. */
    ecs_iter_action_t on_add;

    /** Callback that is invoked when an instance of the component is set. This
     * callback is invoked before triggers are invoked, and enable the component
     * to respond to changes on itself before others can. */
    ecs_iter_action_t on_set;

    /** Callback that is invoked when an instance of the component is removed.
     * This callback is invoked after the triggers are invoked, and before the
     * destructor is invoked. */
    ecs_iter_action_t on_remove;

    void *ctx;                         /**< User defined context */
    void *binding_ctx;                 /**< Language binding context */
    void *lifecycle_ctx;               /**< Component lifecycle context (see meta add-on)*/

    ecs_ctx_free_t ctx_free;           /**< Callback to free ctx */
    ecs_ctx_free_t binding_ctx_free;   /**< Callback to free binding_ctx */
    ecs_ctx_free_t lifecycle_ctx_free; /**< Callback to free lifecycle_ctx */
};

/** Type that contains component information (passed to ctors/dtors/...)
 *
 * @ingroup components
 */
struct ecs_type_info_t {
    ecs_size_t size;         /**< Size of type */
    ecs_size_t alignment;    /**< Alignment of type */
    ecs_type_hooks_t hooks;  /**< Type hooks */
    ecs_entity_t component;  /**< Handle to component (do not set) */
    const char *name;        /**< Type name. */
};

/**
 * @file api_types.h
 * @brief Supporting types for the public API.
 *
 * This file contains types that are typically not used by an application but 
 * support the public API, and therefore must be exposed. This header should not
 * be included by itself.
 */

#ifndef FLECS_API_TYPES_H
#define FLECS_API_TYPES_H


#ifdef __cplusplus
extern "C" {
#endif

////////////////////////////////////////////////////////////////////////////////
//// Opaque types
////////////////////////////////////////////////////////////////////////////////

/** Table data */
typedef struct ecs_data_t ecs_data_t;

/* Cached query table data */
typedef struct ecs_query_cache_table_match_t ecs_query_cache_table_match_t;

////////////////////////////////////////////////////////////////////////////////
//// Non-opaque types
////////////////////////////////////////////////////////////////////////////////

/** All observers for a specific event */
typedef struct ecs_event_record_t {
    struct ecs_event_id_record_t *any;
    struct ecs_event_id_record_t *wildcard;
    struct ecs_event_id_record_t *wildcard_pair;
    ecs_map_t event_ids; /* map<id, ecs_event_id_record_t> */
    ecs_entity_t event;
} ecs_event_record_t;

struct ecs_observable_t {
    ecs_event_record_t on_add;
    ecs_event_record_t on_remove;
    ecs_event_record_t on_set;
    ecs_event_record_t on_wildcard;
    ecs_sparse_t events;  /* sparse<event, ecs_event_record_t> */
    uint64_t last_observer_id;
};

/** Range in table */
typedef struct ecs_table_range_t {
    ecs_table_t *table;
    int32_t offset;       /* Leave both members to 0 to cover entire table */
    int32_t count;       
} ecs_table_range_t;

/** Value of query variable */
typedef struct ecs_var_t {
    ecs_table_range_t range; /* Set when variable stores a range of entities */
    ecs_entity_t entity;     /* Set when variable stores single entity */

    /* Most entities can be stored as a range by setting range.count to 1, 
     * however in order to also be able to store empty entities in variables, 
     * a separate entity member is needed. Both range and entity may be set at
     * the same time, as long as they are consistent. */
} ecs_var_t;

/** Cached reference. */
struct ecs_ref_t {
    ecs_entity_t entity;    /* Entity */
    ecs_entity_t id;        /* Component id */
    uint64_t table_id;      /* Table id for detecting ABA issues */
    struct ecs_table_record_t *tr; /* Table record for component */
    ecs_record_t *record;   /* Entity index record */
};


/* Page-iterator specific data */
typedef struct ecs_page_iter_t {
    int32_t offset;
    int32_t limit;
    int32_t remaining;
} ecs_page_iter_t;

/* Worker-iterator specific data */
typedef struct ecs_worker_iter_t {
    int32_t index;
    int32_t count;
} ecs_worker_iter_t;

/* Convenience struct to iterate table array for id */
typedef struct ecs_table_cache_iter_t {
    struct ecs_table_cache_hdr_t *cur, *next;
    bool iter_fill;
    bool iter_empty;
} ecs_table_cache_iter_t;

/** Each iterator */
typedef struct ecs_each_iter_t {
    ecs_table_cache_iter_t it;

    /* Storage for iterator fields */
    ecs_id_t ids;
    ecs_entity_t sources;
    ecs_size_t sizes;
    int32_t columns;
    const ecs_table_record_t* trs;
} ecs_each_iter_t;

typedef struct ecs_query_op_profile_t {
    int32_t count[2]; /* 0 = enter, 1 = redo */
} ecs_query_op_profile_t;

/** Query iterator */
typedef struct ecs_query_iter_t {
    const ecs_query_t *query;
    struct ecs_var_t *vars;               /* Variable storage */
    const struct ecs_query_var_t *query_vars;
    const struct ecs_query_op_t *ops;
    struct ecs_query_op_ctx_t *op_ctx;    /* Operation-specific state */
    ecs_query_cache_table_match_t *node, *prev, *last; /* For cached iteration */
    uint64_t *written;
    int32_t skip_count;

    ecs_query_op_profile_t *profile;

    int16_t op;
    int16_t sp;
} ecs_query_iter_t;

/* Bits for tracking whether a cache was used/whether the array was allocated.
 * Used by flecs_iter_init, flecs_iter_validate and ecs_iter_fini. 
 * Constants are named to enable easy macro substitution. */
#define flecs_iter_cache_ids           (1u << 0u)
#define flecs_iter_cache_trs           (1u << 1u)
#define flecs_iter_cache_sources       (1u << 2u)
#define flecs_iter_cache_ptrs          (1u << 3u)
#define flecs_iter_cache_variables     (1u << 4u)
#define flecs_iter_cache_all           (255)

/* Inline iterator arrays to prevent allocations for small array sizes */
typedef struct ecs_iter_cache_t {
    ecs_stack_cursor_t *stack_cursor; /* Stack cursor to restore to */
    ecs_flags8_t used;       /* For which fields is the cache used */
    ecs_flags8_t allocated;  /* Which fields are allocated */
} ecs_iter_cache_t;

/* Private iterator data. Used by iterator implementations to keep track of
 * progress & to provide builtin storage. */
typedef struct ecs_iter_private_t {
    union {
        ecs_query_iter_t query;
        ecs_page_iter_t page;
        ecs_worker_iter_t worker;
        ecs_each_iter_t each;
    } iter;                       /* Iterator specific data */

    void *entity_iter;            /* Query applied after matching a table */
    ecs_iter_cache_t cache;       /* Inline arrays to reduce allocations */
} ecs_iter_private_t;

#ifdef __cplusplus
}
#endif

#endif


/**
 * @file api_support.h
 * @brief Support functions and constants.
 *
 * Supporting types and functions that need to be exposed either in support of 
 * the public API or for unit tests, but that may change between minor / patch 
 * releases. 
 */

#ifndef FLECS_API_SUPPORT_H
#define FLECS_API_SUPPORT_H


#ifdef __cplusplus
extern "C" {
#endif

/** This is the largest possible component id. Components for the most part
 * occupy the same id range as entities, however they are not allowed to overlap
 * with (8) bits reserved for id flags. */
#define ECS_MAX_COMPONENT_ID (~((uint32_t)(ECS_ID_FLAGS_MASK >> 32)))

/** The maximum number of nested function calls before the core will throw a
 * cycle detected error */
#define ECS_MAX_RECURSION (512)

/** Maximum length of a parser token (used by parser-related addons) */
#define ECS_MAX_TOKEN_SIZE (256)

FLECS_API
char* flecs_module_path_from_c(
    const char *c_name);

bool flecs_identifier_is_0(
    const char *id);

/* Constructor that zeromem's a component value */
FLECS_API
void flecs_default_ctor(
    void *ptr, 
    int32_t count, 
    const ecs_type_info_t *ctx);

/* Create allocated string from format */
FLECS_DBG_API
char* flecs_vasprintf(
    const char *fmt,
    va_list args);

/* Create allocated string from format */
FLECS_API
char* flecs_asprintf(
    const char *fmt,
    ...);

/** Write an escaped character.
 * Write a character to an output string, insert escape character if necessary.
 *
 * @param out The string to write the character to.
 * @param in The input character.
 * @param delimiter The delimiter used (for example '"')
 * @return Pointer to the character after the last one written.
 */
FLECS_API
char* flecs_chresc(
    char *out,
    char in,
    char delimiter);

/** Parse an escaped character.
 * Parse a character with a potential escape sequence.
 *
 * @param in Pointer to character in input string.
 * @param out Output string.
 * @return Pointer to the character after the last one read.
 */
const char* flecs_chrparse(
    const char *in,
    char *out);

/** Write an escaped string.
 * Write an input string to an output string, escape characters where necessary.
 * To determine the size of the output string, call the operation with a NULL
 * argument for 'out', and use the returned size to allocate a string that is
 * large enough.
 *
 * @param out Pointer to output string (must be).
 * @param size Maximum number of characters written to output.
 * @param delimiter The delimiter used (for example '"').
 * @param in The input string.
 * @return The number of characters that (would) have been written.
 */
FLECS_API
ecs_size_t flecs_stresc(
    char *out,
    ecs_size_t size,
    char delimiter,
    const char *in);

/** Return escaped string.
 * Return escaped version of input string. Same as flecs_stresc(), but returns an
 * allocated string of the right size.
 *
 * @param delimiter The delimiter used (for example '"').
 * @param in The input string.
 * @return Escaped string.
 */
FLECS_API
char* flecs_astresc(
    char delimiter,
    const char *in);

/** Skip whitespace and newline characters.
 * This function skips whitespace characters.
 *
 * @param ptr Pointer to (potential) whitespaces to skip.
 * @return Pointer to the next non-whitespace character.
 */
FLECS_API
const char* flecs_parse_ws_eol(
    const char *ptr);

/** Parse digit.
 * This function will parse until the first non-digit character is found. The
 * provided expression must contain at least one digit character.
 *
 * @param ptr The expression to parse.
 * @param token The output buffer.
 * @return Pointer to the first non-digit character.
 */
FLECS_API
const char* flecs_parse_digit(
    const char *ptr,
    char *token);

/* Convert identifier to snake case */
FLECS_API
char* flecs_to_snake_case(
    const char *str);

FLECS_DBG_API
int32_t flecs_table_observed_count(
    const ecs_table_t *table);

FLECS_DBG_API
void flecs_dump_backtrace(
    void *stream);

FLECS_API
int32_t flecs_poly_claim_(
    ecs_poly_t *poly);

FLECS_API
int32_t flecs_poly_release_(
    ecs_poly_t *poly);

FLECS_API
int32_t flecs_poly_refcount(
    ecs_poly_t *poly);

FLECS_API
int32_t flecs_component_ids_index_get(void);

FLECS_API
ecs_entity_t flecs_component_ids_get(
    const ecs_world_t *world, 
    int32_t index);

FLECS_API
ecs_entity_t flecs_component_ids_get_alive(
    const ecs_world_t *stage_world, 
    int32_t index);

FLECS_API
void flecs_component_ids_set(
    ecs_world_t *world, 
    int32_t index,
    ecs_entity_t id);

#define flecs_poly_claim(poly) \
    flecs_poly_claim_(ECS_CONST_CAST(void*, reinterpret_cast<const void*>(poly)))

#define flecs_poly_release(poly) \
    flecs_poly_release_(ECS_CONST_CAST(void*, reinterpret_cast<const void*>(poly)))


/** Calculate offset from address */
#ifdef __cplusplus
#define ECS_OFFSET(o, offset) reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(o)) + (static_cast<uintptr_t>(offset)))
#else
#define ECS_OFFSET(o, offset) (void*)(((uintptr_t)(o)) + ((uintptr_t)(offset)))
#endif
#define ECS_OFFSET_T(o, T) ECS_OFFSET(o, ECS_SIZEOF(T))

#define ECS_ELEM(ptr, size, index) ECS_OFFSET(ptr, (size) * (index))
#define ECS_ELEM_T(o, T, index) ECS_ELEM(o, ECS_SIZEOF(T), index)

/** Enable/disable bitsets */
#define ECS_BIT_SET(flags, bit) (flags) |= (bit)
#define ECS_BIT_CLEAR(flags, bit) (flags) &= ~(bit) 
#define ECS_BIT_COND(flags, bit, cond) ((cond) \
    ? (ECS_BIT_SET(flags, bit)) \
    : (ECS_BIT_CLEAR(flags, bit)))

#define ECS_BIT_CLEAR16(flags, bit) (flags) &= (ecs_flags16_t)~(bit)   
#define ECS_BIT_COND16(flags, bit, cond) ((cond) \
    ? (ECS_BIT_SET(flags, bit)) \
    : (ECS_BIT_CLEAR16(flags, bit)))

#define ECS_BIT_IS_SET(flags, bit) ((flags) & (bit))

#define ECS_BIT_SETN(flags, n) ECS_BIT_SET(flags, 1llu << n)
#define ECS_BIT_CLEARN(flags, n) ECS_BIT_CLEAR(flags, 1llu << n)
#define ECS_BIT_CONDN(flags, n, cond) ECS_BIT_COND(flags, 1llu << n, cond)

#ifdef __cplusplus
}
#endif

#endif

/**
 * @file hashmap.h
 * @brief Hashmap data structure.
 */

#ifndef FLECS_HASHMAP_H
#define FLECS_HASHMAP_H


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    ecs_vec_t keys;
    ecs_vec_t values;
} ecs_hm_bucket_t;

typedef struct {
    ecs_hash_value_action_t hash;
    ecs_compare_action_t compare;
    ecs_size_t key_size;
    ecs_size_t value_size;
    ecs_block_allocator_t *hashmap_allocator;
    ecs_block_allocator_t bucket_allocator;
    ecs_map_t impl;
} ecs_hashmap_t;

typedef struct {
    ecs_map_iter_t it;
    ecs_hm_bucket_t *bucket;
    int32_t index;
} flecs_hashmap_iter_t;

typedef struct {
    void *key;
    void *value;
    uint64_t hash;
} flecs_hashmap_result_t;

FLECS_DBG_API
void flecs_hashmap_init_(
    ecs_hashmap_t *hm,
    ecs_size_t key_size,
    ecs_size_t value_size,
    ecs_hash_value_action_t hash,
    ecs_compare_action_t compare,
    ecs_allocator_t *allocator);

#define flecs_hashmap_init(hm, K, V, hash, compare, allocator)\
    flecs_hashmap_init_(hm, ECS_SIZEOF(K), ECS_SIZEOF(V), hash, compare, allocator)

FLECS_DBG_API
void flecs_hashmap_fini(
    ecs_hashmap_t *map);

FLECS_DBG_API
void* flecs_hashmap_get_(
    const ecs_hashmap_t *map,
    ecs_size_t key_size,
    const void *key,
    ecs_size_t value_size);

#define flecs_hashmap_get(map, key, V)\
    (V*)flecs_hashmap_get_(map, ECS_SIZEOF(*key), key, ECS_SIZEOF(V))

FLECS_DBG_API
flecs_hashmap_result_t flecs_hashmap_ensure_(
    ecs_hashmap_t *map,
    ecs_size_t key_size,
    const void *key,
    ecs_size_t value_size);

#define flecs_hashmap_ensure(map, key, V)\
    flecs_hashmap_ensure_(map, ECS_SIZEOF(*key), key, ECS_SIZEOF(V))

FLECS_DBG_API
void flecs_hashmap_set_(
    ecs_hashmap_t *map,
    ecs_size_t key_size,
    void *key,
    ecs_size_t value_size,
    const void *value);

#define flecs_hashmap_set(map, key, value)\
    flecs_hashmap_set_(map, ECS_SIZEOF(*key), key, ECS_SIZEOF(*value), value)

FLECS_DBG_API
void flecs_hashmap_remove_(
    ecs_hashmap_t *map,
    ecs_size_t key_size,
    const void *key,
    ecs_size_t value_size);

#define flecs_hashmap_remove(map, key, V)\
    flecs_hashmap_remove_(map, ECS_SIZEOF(*key), key, ECS_SIZEOF(V))

FLECS_DBG_API
void flecs_hashmap_remove_w_hash_(
    ecs_hashmap_t *map,
    ecs_size_t key_size,
    const void *key,
    ecs_size_t value_size,
    uint64_t hash);

#define flecs_hashmap_remove_w_hash(map, key, V, hash)\
    flecs_hashmap_remove_w_hash_(map, ECS_SIZEOF(*key), key, ECS_SIZEOF(V), hash)

FLECS_DBG_API
ecs_hm_bucket_t* flecs_hashmap_get_bucket(
    const ecs_hashmap_t *map,
    uint64_t hash);

FLECS_DBG_API
void flecs_hm_bucket_remove(
    ecs_hashmap_t *map,
    ecs_hm_bucket_t *bucket,
    uint64_t hash,
    int32_t index);

FLECS_DBG_API
void flecs_hashmap_copy(
    ecs_hashmap_t *dst,
    const ecs_hashmap_t *src);

FLECS_DBG_API
flecs_hashmap_iter_t flecs_hashmap_iter(
    ecs_hashmap_t *map);

FLECS_DBG_API
void* flecs_hashmap_next_(
    flecs_hashmap_iter_t *it,
    ecs_size_t key_size,
    void *key_out,
    ecs_size_t value_size);

#define flecs_hashmap_next(map, V)\
    (V*)flecs_hashmap_next_(map, 0, NULL, ECS_SIZEOF(V))

#define flecs_hashmap_next_w_key(map, K, key, V)\
    (V*)flecs_hashmap_next_(map, ECS_SIZEOF(K), key, ECS_SIZEOF(V))

#ifdef __cplusplus
}
#endif

#endif


/** Utility to hold a value of a dynamic type. */
typedef struct ecs_value_t {
    ecs_entity_t type;      /**< Type of value. */
    void *ptr;              /**< Pointer to value. */
} ecs_value_t;

/** Used with ecs_entity_init().
 *
 * @ingroup entities
 */
typedef struct ecs_entity_desc_t {
    int32_t _canary;      /**< Used for validity testing. Must be 0. */

    ecs_entity_t id;      /**< Set to modify existing entity (optional) */

    ecs_entity_t parent;  /**< Parent entity. */

    const char *name;     /**< Name of the entity. If no entity is provided, an
                           * entity with this name will be looked up first. When
                           * an entity is provided, the name will be verified
                           * with the existing entity. */

    const char *sep;      /**< Optional custom separator for hierarchical names.
                           * Leave to NULL for default ('.') separator. Set to
                           * an empty string to prevent tokenization of name. */

    const char *root_sep; /**< Optional, used for identifiers relative to root */

    const char *symbol;   /**< Optional entity symbol. A symbol is an unscoped
                           * identifier that can be used to lookup an entity. The
                           * primary use case for this is to associate the entity
                           * with a language identifier, such as a type or
                           * function name, where these identifiers differ from
                           * the name they are registered with in flecs. For
                           * example, C type "EcsPosition" might be registered
                           * as "flecs.components.transform.Position", with the
                           * symbol set to "EcsPosition". */

    bool use_low_id;      /**< When set to true, a low id (typically reserved for
                           * components) will be used to create the entity, if
                           * no id is specified. */

    /** 0-terminated array of ids to add to the entity. */
    const ecs_id_t *add;

    /** 0-terminated array of values to set on the entity. */
    const ecs_value_t *set;

    /** String expression with components to add */
    const char *add_expr;
} ecs_entity_desc_t;

/** Used with ecs_bulk_init().
 *
 * @ingroup entities
 */
typedef struct ecs_bulk_desc_t {
    int32_t _canary;        /**< Used for validity testing. Must be 0. */

    ecs_entity_t *entities; /**< Entities to bulk insert. Entity ids provided by
                             * the application must be empty (cannot
                             * have components). If no entity ids are provided, the
                             * operation will create 'count' new entities. */

    int32_t count;     /**< Number of entities to create/populate */

    ecs_id_t ids[FLECS_ID_DESC_MAX]; /**< Ids to create the entities with */

    void **data;       /**< Array with component data to insert. Each element in
                        * the array must correspond with an element in the ids
                        * array. If an element in the ids array is a tag, the
                        * data array must contain a NULL. An element may be
                        * set to NULL for a component, in which case the
                        * component will not be set by the operation. */

    ecs_table_t *table; /**< Table to insert the entities into. Should not be set
                         * at the same time as ids. When 'table' is set at the
                         * same time as 'data', the elements in the data array
                         * must correspond with the ids in the table's type. */

} ecs_bulk_desc_t;

/** Used with ecs_component_init().
 *
 * @ingroup components
 */
typedef struct ecs_component_desc_t {
    int32_t _canary;        /**< Used for validity testing. Must be 0. */

    /** Existing entity to associate with observer (optional) */
    ecs_entity_t entity;

    /** Parameters for type (size, hooks, ...) */
    ecs_type_info_t type;
} ecs_component_desc_t;

/** Iterator.
 * Used for iterating queries. The ecs_iter_t type contains all the information
 * that is provided by a query, and contains all the state required for the
 * iterator code.
 * 
 * Functions that create iterators accept as first argument the world, and as
 * second argument the object they iterate. For example:
 * 
 * @code
 * ecs_iter_t it = ecs_query_iter(world, q);
 * @endcode
 * 
 * When this code is called from a system, it is important to use the world
 * provided by its iterator object to ensure thread safety. For example:
 * 
 * @code
 * void Collide(ecs_iter_t *it) {
 *   ecs_iter_t qit = ecs_query_iter(it->world, Colliders);
 * }
 * @endcode
 * 
 * An iterator contains resources that need to be released. By default this 
 * is handled by the last call to next() that returns false. When iteration is
 * ended before iteration has completed, an application has to manually call
 * ecs_iter_fini() to release the iterator resources:
 * 
 * @code
 * ecs_iter_t it = ecs_query_iter(world, q);
 * while (ecs_query_next(&it)) {
 *   if (cond) {
 *     ecs_iter_fini(&it);
 *     break;
 *   }
 * }
 * @endcode
 *
 * @ingroup queries
 */
struct ecs_iter_t {
    /* World */
    ecs_world_t *world;           /**< The world. Can point to stage when in deferred/readonly mode. */
    ecs_world_t *real_world;      /**< Actual world. Never points to a stage. */

    /* Matched data */
    const ecs_entity_t *entities; /**< Entity identifiers */
    const ecs_size_t *sizes;      /**< Component sizes */
    ecs_table_t *table;           /**< Current table */
    ecs_table_t *other_table;     /**< Prev or next table when adding/removing */
    ecs_id_t *ids;                /**< (Component) ids */
    ecs_var_t *variables;         /**< Values of variables (if any) */
    const ecs_table_record_t **trs; /**< Info on where to find field in table */
    ecs_entity_t *sources;        /**< Entity on which the id was matched (0 if same as entities) */
    ecs_flags64_t constrained_vars; /**< Bitset that marks constrained variables */
    uint64_t group_id;            /**< Group id for table, if group_by is used */
    ecs_termset_t set_fields;     /**< Fields that are set */
    ecs_termset_t ref_fields;     /**< Bitset with fields that aren't component arrays */
    ecs_termset_t row_fields;     /**< Fields that must be obtained with field_at */
    ecs_termset_t up_fields;      /**< Bitset with fields matched through up traversal */

    /* Input information */
    ecs_entity_t system;          /**< The system (if applicable) */
    ecs_entity_t event;           /**< The event (if applicable) */
    ecs_id_t event_id;            /**< The (component) id for the event */
    int32_t event_cur;            /**< Unique event id. Used to dedup observer calls */

    /* Query information */
    int8_t field_count;           /**< Number of fields in iterator */
    int8_t term_index;            /**< Index of term that emitted an event.
                                   * This field will be set to the 'index' field
                                   * of an observer term. */
    int8_t variable_count;        /**< Number of variables for query */
    const ecs_query_t *query;     /**< Query being evaluated */
    char **variable_names;        /**< Names of variables (if any) */

    /* Context */
    void *param;                  /**< Param passed to ecs_run */
    void *ctx;                    /**< System context */
    void *binding_ctx;            /**< System binding context */
    void *callback_ctx;           /**< Callback language binding context */
    void *run_ctx;                /**< Run language binding context */

    /* Time */
    ecs_ftime_t delta_time;       /**< Time elapsed since last frame */
    ecs_ftime_t delta_system_time;/**< Time elapsed since last system invocation */

    /* Iterator counters */
    int32_t frame_offset;         /**< Offset relative to start of iteration */
    int32_t offset;               /**< Offset relative to current table */
    int32_t count;                /**< Number of entities to iterate */

    /* Misc */
    ecs_flags32_t flags;          /**< Iterator flags */
    ecs_entity_t interrupted_by;  /**< When set, system execution is interrupted */
    ecs_iter_private_t priv_;     /**< Private data */

    /* Chained iterators */
    ecs_iter_next_action_t next;  /**< Function to progress iterator */
    ecs_iter_action_t callback;   /**< Callback of system or observer */
    ecs_iter_fini_action_t fini;  /**< Function to cleanup iterator resources */
    ecs_iter_t *chain_it;         /**< Optional, allows for creating iterator chains */
};


/** Query must match prefabs.
 * Can be combined with other query flags on the ecs_query_desc_t::flags field.
 * \ingroup queries
 */
#define EcsQueryMatchPrefab           (1u << 1u)

/** Query must match disabled entities.
 * Can be combined with other query flags on the ecs_query_desc_t::flags field.
 * \ingroup queries
 */
#define EcsQueryMatchDisabled         (1u << 2u)

/** Query must match empty tables.
 * Can be combined with other query flags on the ecs_query_desc_t::flags field.
 * \ingroup queries
 */
#define EcsQueryMatchEmptyTables      (1u << 3u)

/** Query may have unresolved entity identifiers.
 * Can be combined with other query flags on the ecs_query_desc_t::flags field.
 * \ingroup queries
 */
#define EcsQueryAllowUnresolvedByName (1u << 6u)

/** Query only returns whole tables (ignores toggle/member fields).
 * Can be combined with other query flags on the ecs_query_desc_t::flags field.
 * \ingroup queries
 */
#define EcsQueryTableOnly             (1u << 7u)


/** Used with ecs_query_init().
 * 
 * \ingroup queries
 */
typedef struct ecs_query_desc_t {
    /** Used for validity testing. Must be 0. */
    int32_t _canary;

    /** Query terms */
    ecs_term_t terms[FLECS_TERM_COUNT_MAX];

    /** Query DSL expression (optional) */
    const char *expr;

    /** Caching policy of query */
    ecs_query_cache_kind_t cache_kind;

    /** Flags for enabling query features */
    ecs_flags32_t flags;

    /** Callback used for ordering query results. If order_by_id is 0, the
     * pointer provided to the callback will be NULL. If the callback is not
     * set, results will not be ordered. */
    ecs_order_by_action_t order_by_callback;

    /** Callback used for ordering query results. Same as order_by_callback,
     * but more efficient. */
    ecs_sort_table_action_t order_by_table_callback;

    /** Component to sort on, used together with order_by_callback or
     * order_by_table_callback. */
    ecs_entity_t order_by;

    /** Component id to be used for grouping. Used together with the
     * group_by_callback. */
    ecs_id_t group_by;

    /** Callback used for grouping results. If the callback is not set, results
     * will not be grouped. When set, this callback will be used to calculate a
     * "rank" for each entity (table) based on its components. This rank is then
     * used to sort entities (tables), so that entities (tables) of the same
     * rank are "grouped" together when iterated. */
    ecs_group_by_action_t group_by_callback;

    /** Callback that is invoked when a new group is created. The return value of
     * the callback is stored as context for a group. */
    ecs_group_create_action_t on_group_create;

    /** Callback that is invoked when an existing group is deleted. The return
     * value of the on_group_create callback is passed as context parameter. */
    ecs_group_delete_action_t on_group_delete;

    /** Context to pass to group_by */
    void *group_by_ctx;

    /** Function to free group_by_ctx */
    ecs_ctx_free_t group_by_ctx_free;

    /** User context to pass to callback */
    void *ctx;

    /** Context to be used for language bindings */
    void *binding_ctx;

    /** Callback to free ctx */
    ecs_ctx_free_t ctx_free;

    /** Callback to free binding_ctx */
    ecs_ctx_free_t binding_ctx_free;

    /** Entity associated with query (optional) */
    ecs_entity_t entity;
} ecs_query_desc_t;

/** Used with ecs_observer_init().
 *
 * @ingroup observers
 */
typedef struct ecs_observer_desc_t {
    /** Used for validity testing. Must be 0. */
    int32_t _canary;

    /** Existing entity to associate with observer (optional) */
    ecs_entity_t entity;

    /** Query for observer */
    ecs_query_desc_t query;

    /** Events to observe (OnAdd, OnRemove, OnSet) */
    ecs_entity_t events[FLECS_EVENT_DESC_MAX];

    /** When observer is created, generate events from existing data. For example,
     * #EcsOnAdd `Position` would match all existing instances of `Position`. */
    bool yield_existing;

    /** Callback to invoke on an event, invoked when the observer matches. */
    ecs_iter_action_t callback;

    /** Callback invoked on an event. When left to NULL the default runner
     * is used which matches the event with the observer's query, and calls
     * 'callback' when it matches.
     * A reason to override the run function is to improve performance, if there
     * are more efficient way to test whether an event matches the observer than
     * the general purpose query matcher. */
    ecs_run_action_t run;

    /** User context to pass to callback */
    void *ctx;

    /** Callback to free ctx */
    ecs_ctx_free_t ctx_free;

    /** Context associated with callback (for language bindings). */
    void *callback_ctx;

    /** Callback to free callback ctx. */
    ecs_ctx_free_t callback_ctx_free;

    /** Context associated with run (for language bindings). */
    void *run_ctx;

    /** Callback to free run ctx. */
    ecs_ctx_free_t run_ctx_free;

    /** Observable with which to register the observer */
    ecs_poly_t *observable;

    /** Optional shared last event id for multiple observers. Ensures only one
     * of the observers with the shared id gets triggered for an event */
    int32_t *last_event_id;

    /** Used for internal purposes */
    int8_t term_index_;
    ecs_flags32_t flags_;
} ecs_observer_desc_t;

/** Used with ecs_emit().
 *
 * @ingroup observers
 */
typedef struct ecs_event_desc_t {
    /** The event id. Only observers for the specified event will be notified */
    ecs_entity_t event;

    /** Component ids. Only observers with a matching component id will be
     * notified. Observers are guaranteed to get notified once, even if they
     * match more than one id. */
    const ecs_type_t *ids;

    /** The table for which to notify. */
    ecs_table_t *table;

    /** Optional 2nd table to notify. This can be used to communicate the
     * previous or next table, in case an entity is moved between tables. */
    ecs_table_t *other_table;

    /** Limit notified entities to ones starting from offset (row) in table */
    int32_t offset;

    /** Limit number of notified entities to count. offset+count must be less
     * than the total number of entities in the table. If left to 0, it will be
     * automatically determined by doing `ecs_table_count(table) - offset`. */
    int32_t count;

    /** Single-entity alternative to setting table / offset / count */
    ecs_entity_t entity;

    /** Optional context.
     * The type of the param must be the event, where the event is a component.
     * When an event is enqueued, the value of param is coped to a temporary
     * storage of the event type. */
    void *param;

    /** Same as param, but with the guarantee that the value won't be modified.
     * When an event with a const parameter is enqueued, the value of the param
     * is copied to a temporary storage of the event type. */
    const void *const_param;

    /** Observable (usually the world) */
    ecs_poly_t *observable;

    /** Event flags */
    ecs_flags32_t flags;
} ecs_event_desc_t;


/**
 * @defgroup misc_types Miscellaneous types
 * Types used to create entities, observers, queries and more.
 *
 * @{
 */

/** Type with information about the current Flecs build */
typedef struct ecs_build_info_t {
    const char *compiler;           /**< Compiler used to compile flecs */
    const char **addons;            /**< Addons included in build */
    const char *version;            /**< Stringified version */
    int16_t version_major;          /**< Major flecs version */
    int16_t version_minor;          /**< Minor flecs version */
    int16_t version_patch;          /**< Patch flecs version */
    bool debug;                     /**< Is this a debug build */
    bool sanitize;                  /**< Is this a sanitize build */
    bool perf_trace;                /**< Is this a perf tracing build */
} ecs_build_info_t;

/** Type that contains information about the world. */
typedef struct ecs_world_info_t {
    ecs_entity_t last_component_id;   /**< Last issued component entity id */
    ecs_entity_t min_id;              /**< First allowed entity id */
    ecs_entity_t max_id;              /**< Last allowed entity id */

    ecs_ftime_t delta_time_raw;       /**< Raw delta time (no time scaling) */
    ecs_ftime_t delta_time;           /**< Time passed to or computed by ecs_progress() */
    ecs_ftime_t time_scale;           /**< Time scale applied to delta_time */
    ecs_ftime_t target_fps;           /**< Target fps */
    ecs_ftime_t frame_time_total;     /**< Total time spent processing a frame */
    ecs_ftime_t system_time_total;    /**< Total time spent in systems */
    ecs_ftime_t emit_time_total;      /**< Total time spent notifying observers */
    ecs_ftime_t merge_time_total;     /**< Total time spent in merges */
    ecs_ftime_t rematch_time_total;   /**< Time spent on query rematching */
    double world_time_total;          /**< Time elapsed in simulation */
    double world_time_total_raw;      /**< Time elapsed in simulation (no scaling) */

    int64_t frame_count_total;        /**< Total number of frames */
    int64_t merge_count_total;        /**< Total number of merges */
    int64_t eval_comp_monitors_total; /**< Total number of monitor evaluations */
    int64_t rematch_count_total;      /**< Total number of rematches */

    int64_t id_create_total;          /**< Total number of times a new id was created */
    int64_t id_delete_total;          /**< Total number of times an id was deleted */
    int64_t table_create_total;       /**< Total number of times a table was created */
    int64_t table_delete_total;       /**< Total number of times a table was deleted */
    int64_t pipeline_build_count_total; /**< Total number of pipeline builds */
    int64_t systems_ran_frame;        /**< Total number of systems ran in last frame */
    int64_t observers_ran_frame;      /**< Total number of times observer was invoked */

    int32_t tag_id_count;             /**< Number of tag (no data) ids in the world */
    int32_t component_id_count;       /**< Number of component (data) ids in the world */
    int32_t pair_id_count;            /**< Number of pair ids in the world */

    int32_t table_count;              /**< Number of tables */

    /* -- Command counts -- */
    struct {
        int64_t add_count;             /**< Add commands processed */
        int64_t remove_count;          /**< Remove commands processed */
        int64_t delete_count;          /**< Selete commands processed */
        int64_t clear_count;           /**< Clear commands processed */
        int64_t set_count;             /**< Set commands processed */
        int64_t ensure_count;          /**< Ensure/emplace commands processed */
        int64_t modified_count;        /**< Modified commands processed */
        int64_t discard_count;         /**< Commands discarded, happens when entity is no longer alive when running the command */
        int64_t event_count;           /**< Enqueued custom events */
        int64_t other_count;           /**< Other commands processed */
        int64_t batched_entity_count;  /**< Entities for which commands were batched */
        int64_t batched_command_count; /**< Commands batched */
    } cmd;                             /**< Command statistics. */

    const char *name_prefix;          /**< Value set by ecs_set_name_prefix(). Used
                                       * to remove library prefixes of symbol
                                       * names (such as `Ecs`, `ecs_`) when
                                       * registering them as names. */
} ecs_world_info_t;

/** Type that contains information about a query group. */
typedef struct ecs_query_group_info_t {
    int32_t match_count;  /**< How often tables have been matched/unmatched */
    int32_t table_count;  /**< Number of tables in group */
    void *ctx;            /**< Group context, returned by on_group_create */
} ecs_query_group_info_t;

/** @} */

/**
 * @defgroup builtin_components Builtin component types.
 * Types that represent builtin components.
 *
 * @{
 */

/** A (string) identifier. Used as pair with #EcsName and #EcsSymbol tags */
typedef struct EcsIdentifier {
    char *value;          /**< Identifier string */
    ecs_size_t length;    /**< Length of identifier */
    uint64_t hash;        /**< Hash of current value */
    uint64_t index_hash;  /**< Hash of existing record in current index */
    ecs_hashmap_t *index; /**< Current index */
} EcsIdentifier;

/** Component information. */
typedef struct EcsComponent {
    ecs_size_t size;           /**< Component size */
    ecs_size_t alignment;      /**< Component alignment */
} EcsComponent;

/** Component for storing a poly object */
typedef struct EcsPoly {
    ecs_poly_t *poly;          /**< Pointer to poly object */
} EcsPoly;

/** When added to an entity this informs serialization formats which component 
 * to use when a value is assigned to an entity without specifying the 
 * component. This is intended as a hint, serialization formats are not required 
 * to use it. Adding this component does not change the behavior of core ECS 
 * operations. */
typedef struct EcsDefaultChildComponent {
    ecs_id_t component;  /**< Default component id. */
} EcsDefaultChildComponent;

/** @} */
/** @} */

/* Only include deprecated definitions if deprecated addon is required */
#ifdef FLECS_DEPRECATED
/**
 * @file addons/deprecated.h
 * @brief The deprecated addon contains deprecated operations.
 */

#ifdef FLECS_DEPRECATED

#ifndef FLECS_DEPRECATED_H
#define FLECS_DEPRECATED_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif

#endif

#endif

/**
 * @defgroup api_constants API Constants
 * Public API constants.
 *
 * @{
 */

/**
 * @defgroup id_flags Component id flags.
 * Id flags are bits that can be set on an id (ecs_id_t).
 *
 * @{
 */

/** Indicates that the id is a pair. */
FLECS_API extern const ecs_id_t ECS_PAIR;

/** Automatically override component when it is inherited */
FLECS_API extern const ecs_id_t ECS_AUTO_OVERRIDE;

/** Adds bitset to storage which allows component to be enabled/disabled */
FLECS_API extern const ecs_id_t ECS_TOGGLE;

/** @} */

/**
 * @defgroup builtin_tags Builtin component ids.
 * @{
 */

/* Builtin component ids */

/** Component component id. */
FLECS_API extern const ecs_entity_t ecs_id(EcsComponent);

/** Identifier component id. */
FLECS_API extern const ecs_entity_t ecs_id(EcsIdentifier);

/** Poly component id. */
FLECS_API extern const ecs_entity_t ecs_id(EcsPoly);

/** DefaultChildComponent component id. */
FLECS_API extern const ecs_entity_t ecs_id(EcsDefaultChildComponent);

/** Tag added to queries. */
FLECS_API extern const ecs_entity_t EcsQuery;

/** Tag added to observers. */
FLECS_API extern const ecs_entity_t EcsObserver;

/** Tag added to systems. */
FLECS_API extern const ecs_entity_t EcsSystem;

/** TickSource component id. */
FLECS_API extern const ecs_entity_t ecs_id(EcsTickSource);

/** Pipeline module component ids */
FLECS_API extern const ecs_entity_t ecs_id(EcsPipelineQuery);

/** Timer component id. */
FLECS_API extern const ecs_entity_t ecs_id(EcsTimer);

/** RateFilter component id. */
FLECS_API extern const ecs_entity_t ecs_id(EcsRateFilter);

/** Root scope for builtin flecs entities */
FLECS_API extern const ecs_entity_t EcsFlecs;

/** Core module scope */
FLECS_API extern const ecs_entity_t EcsFlecsCore;

/** Entity associated with world (used for "attaching" components to world) */
FLECS_API extern const ecs_entity_t EcsWorld;

/** Wildcard entity ("*"). Matches any id, returns all matches. */
FLECS_API extern const ecs_entity_t EcsWildcard;

/** Any entity ("_"). Matches any id, returns only the first. */
FLECS_API extern const ecs_entity_t EcsAny;

/** This entity. Default source for queries. */
FLECS_API extern const ecs_entity_t EcsThis;

/** Variable entity ("$"). Used in expressions to prefix variable names */
FLECS_API extern const ecs_entity_t EcsVariable;

/** Shortcut as EcsVariable is typically used as source for singleton terms */
#define EcsSingleton EcsVariable

/** Marks a relationship as transitive.
 * Behavior:
 *
 * @code
 *   if R(X, Y) and R(Y, Z) then R(X, Z)
 * @endcode
 */
FLECS_API extern const ecs_entity_t EcsTransitive;

/** Marks a relationship as reflexive.
 * Behavior:
 *
 * @code
 *   R(X, X) == true
 * @endcode
 */
FLECS_API extern const ecs_entity_t EcsReflexive;

/** Ensures that entity/component cannot be used as target in `IsA` relationship.
 * Final can improve the performance of queries as they will not attempt to 
 * substitute a final component with its subsets.
 *
 * Behavior:
 *
 * @code
 *   if IsA(X, Y) and Final(Y) throw error
 * @endcode
 */
FLECS_API extern const ecs_entity_t EcsFinal;

/** Relationship that specifies component inheritance behavior. */
FLECS_API extern const ecs_entity_t EcsOnInstantiate;

/** Override component on instantiate. 
 * This will copy the component from the base entity `(IsA target)` to the
 * instance. The base component will never be inherited from the prefab. */
FLECS_API extern const ecs_entity_t EcsOverride;

/** Inherit component on instantiate. 
 * This will inherit (share) the component from the base entity `(IsA target)`.
 * The component can be manually overridden by adding it to the instance. */
FLECS_API extern const ecs_entity_t EcsInherit;

/** Never inherit component on instantiate. 
 * This will not copy or share the component from the base entity `(IsA target)`.
 * When the component is added to an instance, its value will never be copied 
 * from the base entity. */
FLECS_API extern const ecs_entity_t EcsDontInherit;

/** Marks relationship as commutative.
 * Behavior:
 *
 * @code
 *   if R(X, Y) then R(Y, X)
 * @endcode
 */
FLECS_API extern const ecs_entity_t EcsSymmetric;

/** Can be added to relationship to indicate that the relationship can only occur
 * once on an entity. Adding a 2nd instance will replace the 1st.
 *
 * Behavior:
 *
 * @code
 *   R(X, Y) + R(X, Z) = R(X, Z)
 * @endcode
 */
FLECS_API extern const ecs_entity_t EcsExclusive;

/** Marks a relationship as acyclic. Acyclic relationships may not form cycles. */
FLECS_API extern const ecs_entity_t EcsAcyclic;

/** Marks a relationship as traversable. Traversable relationships may be
 * traversed with "up" queries. Traversable relationships are acyclic. */
FLECS_API extern const ecs_entity_t EcsTraversable;

/** Ensure that a component always is added together with another component.
 *
 * Behavior:
 *
 * @code
 *   If With(R, O) and R(X) then O(X)
 *   If With(R, O) and R(X, Y) then O(X, Y)
 * @endcode
 */
FLECS_API extern const ecs_entity_t EcsWith;

/** Ensure that relationship target is child of specified entity.
 *
 * Behavior:
 *
 * @code
 *   If OneOf(R, O) and R(X, Y), Y must be a child of O
 *   If OneOf(R) and R(X, Y), Y must be a child of R
 * @endcode
 */
FLECS_API extern const ecs_entity_t EcsOneOf;

/** Mark a component as toggleable with ecs_enable_id(). */
FLECS_API extern const ecs_entity_t EcsCanToggle;

/** Can be added to components to indicate it is a trait. Traits are components
 * and/or tags that are added to other components to modify their behavior.
 */
FLECS_API extern const ecs_entity_t EcsTrait;

/** Ensure that an entity is always used in pair as relationship.
 *
 * Behavior:
 *
 * @code
 *   e.add(R) panics
 *   e.add(X, R) panics, unless X has the "Trait" trait
 * @endcode
 */
FLECS_API extern const ecs_entity_t EcsRelationship;

/** Ensure that an entity is always used in pair as target.
 *
 * Behavior:
 *
 * @code
 *   e.add(T) panics
 *   e.add(T, X) panics
 * @endcode
 */
FLECS_API extern const ecs_entity_t EcsTarget;

/** Can be added to relationship to indicate that it should never hold data, 
 * even when it or the relationship target is a component. */
FLECS_API extern const ecs_entity_t EcsPairIsTag;

/** Tag to indicate name identifier */
FLECS_API extern const ecs_entity_t EcsName;

/** Tag to indicate symbol identifier */
FLECS_API extern const ecs_entity_t EcsSymbol;

/** Tag to indicate alias identifier */
FLECS_API extern const ecs_entity_t EcsAlias;

/** Used to express parent-child relationships. */
FLECS_API extern const ecs_entity_t EcsChildOf;

/** Used to express inheritance relationships. */
FLECS_API extern const ecs_entity_t EcsIsA;

/** Used to express dependency relationships */
FLECS_API extern const ecs_entity_t EcsDependsOn;

/** Used to express a slot (used with prefab inheritance) */
FLECS_API extern const ecs_entity_t EcsSlotOf;

/** Tag added to module entities */
FLECS_API extern const ecs_entity_t EcsModule;

/** Tag to indicate an entity/component/system is private to a module */
FLECS_API extern const ecs_entity_t EcsPrivate;

/** Tag added to prefab entities. Any entity with this tag is automatically
 * ignored by queries, unless #EcsPrefab is explicitly queried for. */
FLECS_API extern const ecs_entity_t EcsPrefab;

/** When this tag is added to an entity it is skipped by queries, unless
 * #EcsDisabled is explicitly queried for. */
FLECS_API extern const ecs_entity_t EcsDisabled;

/** Trait added to entities that should never be returned by queries. Reserved
 * for internal entities that have special meaning to the query engine, such as
 * #EcsThis, #EcsWildcard, #EcsAny. */
FLECS_API extern const ecs_entity_t EcsNotQueryable;

/** Event that triggers when an id is added to an entity */
FLECS_API extern const ecs_entity_t EcsOnAdd;

/** Event that triggers when an id is removed from an entity */
FLECS_API extern const ecs_entity_t EcsOnRemove;

/** Event that triggers when a component is set for an entity */
FLECS_API extern const ecs_entity_t EcsOnSet;

/** Event that triggers observer when an entity starts/stops matching a query */
FLECS_API extern const ecs_entity_t EcsMonitor;

/** Event that triggers when a table is created. */
FLECS_API extern const ecs_entity_t EcsOnTableCreate;

/** Event that triggers when a table is deleted. */
FLECS_API extern const ecs_entity_t EcsOnTableDelete;

/** Relationship used for specifying cleanup behavior. */
FLECS_API extern const ecs_entity_t EcsOnDelete;

/** Relationship used to define what should happen when a target entity (second
 * element of a pair) is deleted. */
FLECS_API extern const ecs_entity_t EcsOnDeleteTarget;

/** Remove cleanup policy. Must be used as target in pair with #EcsOnDelete or
 * #EcsOnDeleteTarget. */
FLECS_API extern const ecs_entity_t EcsRemove;

/** Delete cleanup policy. Must be used as target in pair with #EcsOnDelete or
 * #EcsOnDeleteTarget. */
FLECS_API extern const ecs_entity_t EcsDelete;

/** Panic cleanup policy. Must be used as target in pair with #EcsOnDelete or
 * #EcsOnDeleteTarget. */
FLECS_API extern const ecs_entity_t EcsPanic;

/** Mark component as sparse */
FLECS_API extern const ecs_entity_t EcsSparse;

/** Mark relationship as union */
FLECS_API extern const ecs_entity_t EcsUnion;

/** Marker used to indicate `$var == ...` matching in queries. */
FLECS_API extern const ecs_entity_t EcsPredEq;

/** Marker used to indicate `$var == "name"` matching in queries. */
FLECS_API extern const ecs_entity_t EcsPredMatch;

/** Marker used to indicate `$var ~= "pattern"` matching in queries. */
FLECS_API extern const ecs_entity_t EcsPredLookup;

/** Marker used to indicate the start of a scope (`{`) in queries. */
FLECS_API extern const ecs_entity_t EcsScopeOpen;

/** Marker used to indicate the end of a scope (`}`) in queries. */
FLECS_API extern const ecs_entity_t EcsScopeClose;

/** Tag used to indicate query is empty.
 * This tag is removed automatically when a query becomes non-empty, and is not
 * automatically re-added when it becomes empty.
 */
FLECS_API extern const ecs_entity_t EcsEmpty;

FLECS_API extern const ecs_entity_t ecs_id(EcsPipeline); /**< Pipeline component id. */
FLECS_API extern const ecs_entity_t EcsOnStart;     /**< OnStart pipeline phase. */
FLECS_API extern const ecs_entity_t EcsPreFrame;    /**< PreFrame pipeline phase. */
FLECS_API extern const ecs_entity_t EcsOnLoad;      /**< OnLoad pipeline phase. */
FLECS_API extern const ecs_entity_t EcsPostLoad;    /**< PostLoad pipeline phase. */
FLECS_API extern const ecs_entity_t EcsPreUpdate;   /**< PreUpdate pipeline phase. */
FLECS_API extern const ecs_entity_t EcsOnUpdate;    /**< OnUpdate pipeline phase. */
FLECS_API extern const ecs_entity_t EcsOnValidate;  /**< OnValidate pipeline phase. */
FLECS_API extern const ecs_entity_t EcsPostUpdate;  /**< PostUpdate pipeline phase. */
FLECS_API extern const ecs_entity_t EcsPreStore;    /**< PreStore pipeline phase. */
FLECS_API extern const ecs_entity_t EcsOnStore;     /**< OnStore pipeline phase. */
FLECS_API extern const ecs_entity_t EcsPostFrame;   /**< PostFrame pipeline phase. */
FLECS_API extern const ecs_entity_t EcsPhase;       /**< Phase pipeline phase. */

/** Value used to quickly check if component is builtin. This is used to quickly
 * filter out tables with builtin components (for example for ecs_delete()) */
#define EcsLastInternalComponentId (ecs_id(EcsPoly))

/** The first user-defined component starts from this id. Ids up to this number
 * are reserved for builtin components */
#define EcsFirstUserComponentId (8)

/** The first user-defined entity starts from this id. Ids up to this number
 * are reserved for builtin entities */
#define EcsFirstUserEntityId (FLECS_HI_COMPONENT_ID + 128)

/* When visualized the reserved id ranges look like this:
 * - [1..8]: Builtin components
 * - [9..FLECS_HI_COMPONENT_ID]: Low ids reserved for application components
 * - [FLECS_HI_COMPONENT_ID + 1..EcsFirstUserEntityId]: Builtin entities
 */

/** @} */
/** @} */

/**
 * @defgroup world_api World
 * Functions for working with `ecs_world_t`.
 *
 * @{
 */

/**
 * @defgroup world_creation_deletion Creation & Deletion
 * @{
 */

/** Create a new world.
 * This operation automatically imports modules from addons Flecs has been built
 * with, except when the module specifies otherwise.
 *
 * @return A new world
 */
FLECS_API
ecs_world_t* ecs_init(void);

/** Create a new world with just the core module.
 * Same as ecs_init(), but doesn't import modules from addons. This operation is
 * faster than ecs_init() and results in less memory utilization.
 *
 * @return A new tiny world
 */
FLECS_API
ecs_world_t* ecs_mini(void);

/** Create a new world with arguments.
 * Same as ecs_init(), but allows passing in command line arguments. Command line
 * arguments are used to:
 * - automatically derive the name of the application from argv[0]
 *
 * @return A new world
 */
FLECS_API
ecs_world_t* ecs_init_w_args(
    int argc,
    char *argv[]);

/** Delete a world.
 * This operation deletes the world, and everything it contains.
 *
 * @param world The world to delete.
 * @return Zero if successful, non-zero if failed.
 */
FLECS_API
int ecs_fini(
    ecs_world_t *world);

/** Returns whether the world is being deleted.
 * This operation can be used in callbacks like type hooks or observers to
 * detect if they are invoked while the world is being deleted.
 *
 * @param world The world.
 * @return True if being deleted, false if not.
 */
FLECS_API
bool ecs_is_fini(
    const ecs_world_t *world);

/** Register action to be executed when world is destroyed.
 * Fini actions are typically used when a module needs to clean up before a
 * world shuts down.
 *
 * @param world The world.
 * @param action The function to execute.
 * @param ctx Userdata to pass to the function */
FLECS_API
void ecs_atfini(
    ecs_world_t *world,
    ecs_fini_action_t action,
    void *ctx);

/** Type returned by ecs_get_entities(). */
typedef struct ecs_entities_t {
    const ecs_entity_t *ids; /**< Array with all entity ids in the world. */
    int32_t count;           /**< Total number of entity ids. */
    int32_t alive_count;     /**< Number of alive entity ids. */
} ecs_entities_t;

/** Return entity identifiers in world.
 * This operation returns an array with all entity ids that exist in the world.
 * Note that the returned array will change and may get invalidated as a result
 * of entity creation & deletion.
 * 
 * To iterate all alive entity ids, do:
 * @code
 * ecs_entities_t entities = ecs_get_entities(world);
 * for (int i = 0; i < entities.alive_count; i ++) {
 *   ecs_entity_t id = entities.ids[i];
 * }
 * @endcode
 * 
 * To iterate not-alive ids, do:
 * @code
 * for (int i = entities.alive_count + 1; i < entities.count; i ++) {
 *   ecs_entity_t id = entities.ids[i];
 * }
 * @endcode
 * 
 * The returned array does not need to be freed. Mutating the returned array
 * will return in undefined behavior (and likely crashes).
 * 
 * @param world The world.
 * @return Struct with entity id array.
 */
FLECS_API
ecs_entities_t ecs_get_entities(
    const ecs_world_t *world);

/** Get flags set on the world.
 * This operation returns the internal flags (see api_flags.h) that are
 * set on the world.
 *
 * @param world The world.
 * @return Flags set on the world.
 */
FLECS_API
ecs_flags32_t ecs_world_get_flags(
    const ecs_world_t *world);

/** @} */

/**
 * @defgroup world_frame Frame functions
 * @{
 */

/** Begin frame.
 * When an application does not use ecs_progress() to control the main loop, it
 * can still use Flecs features such as FPS limiting and time measurements. This
 * operation needs to be invoked whenever a new frame is about to get processed.
 *
 * Calls to ecs_frame_begin() must always be followed by ecs_frame_end().
 *
 * The function accepts a delta_time parameter, which will get passed to
 * systems. This value is also used to compute the amount of time the function
 * needs to sleep to ensure it does not exceed the target_fps, when it is set.
 * When 0 is provided for delta_time, the time will be measured.
 *
 * This function should only be ran from the main thread.
 *
 * @param world The world.
 * @param delta_time Time elapsed since the last frame.
 * @return The provided delta_time, or measured time if 0 was provided.
 */
FLECS_API
ecs_ftime_t ecs_frame_begin(
    ecs_world_t *world,
    ecs_ftime_t delta_time);

/** End frame.
 * This operation must be called at the end of the frame, and always after
 * ecs_frame_begin().
 *
 * @param world The world.
 */
FLECS_API
void ecs_frame_end(
    ecs_world_t *world);

/** Register action to be executed once after frame.
 * Post frame actions are typically used for calling operations that cannot be
 * invoked during iteration, such as changing the number of threads.
 *
 * @param world The world.
 * @param action The function to execute.
 * @param ctx Userdata to pass to the function */
FLECS_API
void ecs_run_post_frame(
    ecs_world_t *world,
    ecs_fini_action_t action,
    void *ctx);

/** Signal exit
 * This operation signals that the application should quit. It will cause
 * ecs_progress() to return false.
 *
 * @param world The world to quit.
 */
FLECS_API
void ecs_quit(
    ecs_world_t *world);

/** Return whether a quit has been requested.
 *
 * @param world The world.
 * @return Whether a quit has been requested.
 * @see ecs_quit()
 */
FLECS_API
bool ecs_should_quit(
    const ecs_world_t *world);

/** Measure frame time.
 * Frame time measurements measure the total time passed in a single frame, and
 * how much of that time was spent on systems and on merging.
 *
 * Frame time measurements add a small constant-time overhead to an application.
 * When an application sets a target FPS, frame time measurements are enabled by
 * default.
 *
 * @param world The world.
 * @param enable Whether to enable or disable frame time measuring.
 */
FLECS_API void ecs_measure_frame_time(
    ecs_world_t *world,
    bool enable);

/** Measure system time.
 * System time measurements measure the time spent in each system.
 *
 * System time measurements add overhead to every system invocation and
 * therefore have a small but measurable impact on application performance.
 * System time measurements must be enabled before obtaining system statistics.
 *
 * @param world The world.
 * @param enable Whether to enable or disable system time measuring.
 */
FLECS_API void ecs_measure_system_time(
    ecs_world_t *world,
    bool enable);

/** Set target frames per second (FPS) for application.
 * Setting the target FPS ensures that ecs_progress() is not invoked faster than
 * the specified FPS. When enabled, ecs_progress() tracks the time passed since
 * the last invocation, and sleeps the remaining time of the frame (if any).
 *
 * This feature ensures systems are ran at a consistent interval, as well as
 * conserving CPU time by not running systems more often than required.
 *
 * Note that ecs_progress() only sleeps if there is time left in the frame. Both
 * time spent in flecs as time spent outside of flecs are taken into
 * account.
 *
 * @param world The world.
 * @param fps The target FPS.
 */
FLECS_API
void ecs_set_target_fps(
    ecs_world_t *world,
    ecs_ftime_t fps);

/** Set default query flags. 
 * Set a default value for the ecs_filter_desc_t::flags field. Default flags
 * are applied in addition to the flags provided in the descriptor. For a
 * list of available flags, see include/flecs/private/api_flags.h. Typical flags
 * to use are:
 *
 *  - `EcsQueryMatchEmptyTables`
 *  - `EcsQueryMatchDisabled`
 *  - `EcsQueryMatchPrefab`
 * 
 * @param world The world.
 * @param flags The query flags.
 */
FLECS_API
void ecs_set_default_query_flags(
    ecs_world_t *world,
    ecs_flags32_t flags);

/** @} */

/**
 * @defgroup commands Commands
 * @{
 */

/** Begin readonly mode.
 * This operation puts the world in readonly mode, which disallows mutations on
 * the world. Readonly mode exists so that internal mechanisms can implement
 * optimizations that certain aspects of the world to not change, while also 
 * providing a mechanism for applications to prevent accidental mutations in, 
 * for example, multithreaded applications.
 * 
 * Readonly mode is a stronger version of deferred mode. In deferred mode
 * ECS operations such as add/remove/set/delete etc. are added to a command 
 * queue to be executed later. In readonly mode, operations that could break
 * scheduler logic (such as creating systems, queries) are also disallowed.
 * 
 * Readonly mode itself has a single threaded and a multi threaded mode. In
 * single threaded mode certain mutations on the world are still allowed, for 
 * example:
 * - Entity liveliness operations (such as new, make_alive), so that systems are
 *   able to create new entities.
 * - Implicit component registration, so that this works from systems
 * - Mutations to supporting data structures for the evaluation of uncached 
 *   queries (filters), so that these can be created on the fly.
 * 
 * These mutations are safe in a single threaded applications, but for
 * multithreaded applications the world needs to be entirely immutable. For this
 * purpose multi threaded readonly mode exists, which disallows all mutations on
 * the world. This means that in multi threaded applications, entity liveliness
 * operations, implicit component registration, and on-the-fly query creation
 * are not guaranteed to work.
 * 
 * While in readonly mode, applications can still enqueue ECS operations on a
 * stage. Stages are managed automatically when using the pipeline addon and 
 * ecs_progress(), but they can also be configured manually as shown here:
 * 
 * @code
 * // Number of stages typically corresponds with number of threads
 * ecs_set_stage_count(world, 2);
 * ecs_stage_t *stage = ecs_get_stage(world, 1);
 *
 * ecs_readonly_begin(world);
 * ecs_add(world, e, Tag); // readonly assert
 * ecs_add(stage, e, Tag); // OK
 * @endcode
 * 
 * When an attempt is made to perform an operation on a world in readonly mode,
 * the code will throw an assert saying that the world is in readonly mode.
 * 
 * A call to ecs_readonly_begin() must be followed up with ecs_readonly_end().
 * When ecs_readonly_end() is called, all enqueued commands from configured 
 * stages are merged back into the world. Calls to ecs_readonly_begin() and
 * ecs_readonly_end() should always happen from a context where the code has
 * exclusive access to the world. The functions themselves are not thread safe.
 * 
 * In a typical application, a (non-exhaustive) call stack that uses 
 * ecs_readonly_begin() and ecs_readonly_end() will look like this:
 * 
 * @code
 * ecs_progress()
 *   ecs_readonly_begin()
 *     ecs_defer_begin()
 * 
 *       // user code
 * 
 *   ecs_readonly_end()
 *     ecs_defer_end()
 *@endcode
 *
 * @param world The world
 * @param multi_threaded Whether to enable readonly/multi threaded mode.
 * @return Whether world is in readonly mode.
 */
FLECS_API
bool ecs_readonly_begin(
    ecs_world_t *world,
    bool multi_threaded);

/** End readonly mode.
 * This operation ends readonly mode, and must be called after
 * ecs_readonly_begin(). Operations that were deferred while the world was in
 * readonly mode will be flushed.
 *
 * @param world The world
 */
FLECS_API
void ecs_readonly_end(
    ecs_world_t *world);

/** Merge world or stage.
 * When automatic merging is disabled, an application can call this
 * operation on either an individual stage, or on the world which will merge
 * all stages. This operation may only be called when staging is not enabled
 * (either after ecs_progress() or after ecs_readonly_end()).
 *
 * This operation may be called on an already merged stage or world.
 *
 * @param world The world.
 */
FLECS_API
void ecs_merge(
    ecs_world_t *world);

/** Defer operations until end of frame.
 * When this operation is invoked while iterating, operations inbetween the
 * ecs_defer_begin() and ecs_defer_end() operations are executed at the end
 * of the frame.
 *
 * This operation is thread safe.
 *
 * @param world The world.
 * @return true if world changed from non-deferred mode to deferred mode.
 *
 * @see ecs_defer_end()
 * @see ecs_is_deferred()
 * @see ecs_defer_resume()
 * @see ecs_defer_suspend()
 */
FLECS_API
bool ecs_defer_begin(
    ecs_world_t *world);

/** Test if deferring is enabled for current stage.
 *
 * @param world The world.
 * @return True if deferred, false if not.
 *
 * @see ecs_defer_begin()
 * @see ecs_defer_end()
 * @see ecs_defer_resume()
 * @see ecs_defer_suspend()
 */
FLECS_API
bool ecs_is_deferred(
    const ecs_world_t *world);

/** End block of operations to defer.
 * See ecs_defer_begin().
 *
 * This operation is thread safe.
 *
 * @param world The world.
 * @return true if world changed from deferred mode to non-deferred mode.
 *
 * @see ecs_defer_begin()
 * @see ecs_defer_is_deferred()
 * @see ecs_defer_resume()
 * @see ecs_defer_suspend()
 */
FLECS_API
bool ecs_defer_end(
    ecs_world_t *world);

/** Suspend deferring but do not flush queue.
 * This operation can be used to do an undeferred operation while not flushing
 * the operations in the queue.
 *
 * An application should invoke ecs_defer_resume() before ecs_defer_end() is called.
 * The operation may only be called when deferring is enabled.
 *
 * @param world The world.
 *
 * @see ecs_defer_begin()
 * @see ecs_defer_end()
 * @see ecs_defer_is_deferred()
 * @see ecs_defer_resume()
 */
FLECS_API
void ecs_defer_suspend(
    ecs_world_t *world);

/** Resume deferring.
 * See ecs_defer_suspend().
 *
 * @param world The world.
 *
 * @see ecs_defer_begin()
 * @see ecs_defer_end()
 * @see ecs_defer_is_deferred()
 * @see ecs_defer_suspend()
 */
FLECS_API
void ecs_defer_resume(
    ecs_world_t *world);

/** Configure world to have N stages.
 * This initializes N stages, which allows applications to defer operations to
 * multiple isolated defer queues. This is typically used for applications with
 * multiple threads, where each thread gets its own queue, and commands are
 * merged when threads are synchronized.
 *
 * Note that the ecs_set_threads() function already creates the appropriate
 * number of stages. The ecs_set_stage_count() operation is useful for applications
 * that want to manage their own stages and/or threads.
 *
 * @param world The world.
 * @param stages The number of stages.
 */
FLECS_API
void ecs_set_stage_count(
    ecs_world_t *world,
    int32_t stages);

/** Get number of configured stages.
 * Return number of stages set by ecs_set_stage_count().
 *
 * @param world The world.
 * @return The number of stages used for threading.
 */
FLECS_API
int32_t ecs_get_stage_count(
    const ecs_world_t *world);

/** Get stage-specific world pointer.
 * Flecs threads can safely invoke the API as long as they have a private
 * context to write to, also referred to as the stage. This function returns a
 * pointer to a stage, disguised as a world pointer.
 *
 * Note that this function does not(!) create a new world. It simply wraps the
 * existing world in a thread-specific context, which the API knows how to
 * unwrap. The reason the stage is returned as an ecs_world_t is so that it
 * can be passed transparently to the existing API functions, vs. having to
 * create a dedicated API for threading.
 *
 * @param world The world.
 * @param stage_id The index of the stage to retrieve.
 * @return A thread-specific pointer to the world.
 */
FLECS_API
ecs_world_t* ecs_get_stage(
    const ecs_world_t *world,
    int32_t stage_id);

/** Test whether the current world is readonly.
 * This function allows the code to test whether the currently used world
 * is readonly or whether it allows for writing.
 *
 * @param world A pointer to a stage or the world.
 * @return True if the world or stage is readonly.
 */
FLECS_API
bool ecs_stage_is_readonly(
    const ecs_world_t *world);

/** Create unmanaged stage.
 * Create a stage whose lifecycle is not managed by the world. Must be freed
 * with ecs_stage_free().
 *
 * @param world The world.
 * @return The stage.
 */
FLECS_API
ecs_world_t* ecs_stage_new(
    ecs_world_t *world);

/** Free unmanaged stage.
 *
 * @param stage The stage to free.
 */
FLECS_API
void ecs_stage_free(
    ecs_world_t *stage);

/** Get stage id.
 * The stage id can be used by an application to learn about which stage it is
 * using, which typically corresponds with the worker thread id.
 *
 * @param world The world.
 * @return The stage id.
 */
FLECS_API
int32_t ecs_stage_get_id(
    const ecs_world_t *world);

/** @} */

/**
 * @defgroup world_misc Misc
 * @{
 */

/** Set a world context.
 * This operation allows an application to register custom data with a world
 * that can be accessed anywhere where the application has the world.
 *
 * @param world The world.
 * @param ctx A pointer to a user defined structure.
 * @param ctx_free A function that is invoked with ctx when the world is freed.
 */
FLECS_API
void ecs_set_ctx(
    ecs_world_t *world,
    void *ctx,
    ecs_ctx_free_t ctx_free);

/** Set a world binding context.
 * Same as ecs_set_ctx() but for binding context. A binding context is intended
 * specifically for language bindings to store binding specific data.
 *
 * @param world The world.
 * @param ctx A pointer to a user defined structure.
 * @param ctx_free A function that is invoked with ctx when the world is freed.
 */
FLECS_API
void ecs_set_binding_ctx(
    ecs_world_t *world,
    void *ctx,
    ecs_ctx_free_t ctx_free);

/** Get the world context.
 * This operation retrieves a previously set world context.
 *
 * @param world The world.
 * @return The context set with ecs_set_ctx(). If no context was set, the
 *         function returns NULL.
 */
FLECS_API
void* ecs_get_ctx(
    const ecs_world_t *world);

/** Get the world binding context.
 * This operation retrieves a previously set world binding context.
 *
 * @param world The world.
 * @return The context set with ecs_set_binding_ctx(). If no context was set, the
 *         function returns NULL.
 */
FLECS_API
void* ecs_get_binding_ctx(
    const ecs_world_t *world);

/** Get build info.
 *  Returns information about the current Flecs build.
 * 
 * @return A struct with information about the current Flecs build.
 */
FLECS_API
const ecs_build_info_t* ecs_get_build_info(void);

/** Get world info.
 *
 * @param world The world.
 * @return Pointer to the world info. Valid for as long as the world exists.
 */
FLECS_API
const ecs_world_info_t* ecs_get_world_info(
    const ecs_world_t *world);

/** Dimension the world for a specified number of entities.
 * This operation will preallocate memory in the world for the specified number
 * of entities. Specifying a number lower than the current number of entities in
 * the world will have no effect.
 *
 * @param world The world.
 * @param entity_count The number of entities to preallocate.
 */
FLECS_API
void ecs_dim(
    ecs_world_t *world,
    int32_t entity_count);

/** Set a range for issuing new entity ids.
 * This function constrains the entity identifiers returned by ecs_new_w() to the
 * specified range. This operation can be used to ensure that multiple processes
 * can run in the same simulation without requiring a central service that
 * coordinates issuing identifiers.
 *
 * If `id_end` is set to 0, the range is infinite. If `id_end` is set to a non-zero
 * value, it has to be larger than `id_start`. If `id_end` is set and ecs_new() is
 * invoked after an id is issued that is equal to `id_end`, the application will
 * abort.
 *
 * @param world The world.
 * @param id_start The start of the range.
 * @param id_end The end of the range.
 */
FLECS_API
void ecs_set_entity_range(
    ecs_world_t *world,
    ecs_entity_t id_start,
    ecs_entity_t id_end);

/** Enable/disable range limits.
 * When an application is both a receiver of range-limited entities and a
 * producer of range-limited entities, range checking needs to be temporarily
 * disabled when inserting received entities. Range checking is disabled on a
 * stage, so setting this value is thread safe.
 *
 * @param world The world.
 * @param enable True if range checking should be enabled, false to disable.
 * @return The previous value.
 */
FLECS_API
bool ecs_enable_range_check(
    ecs_world_t *world,
    bool enable);

/** Get the largest issued entity id (not counting generation).
 *
 * @param world The world.
 * @return The largest issued entity id.
 */
FLECS_API
ecs_entity_t ecs_get_max_id(
    const ecs_world_t *world);

/** Force aperiodic actions.
 * The world may delay certain operations until they are necessary for the
 * application to function correctly. This may cause observable side effects
 * such as delayed triggering of events, which can be inconvenient when for
 * example running a test suite.
 *
 * The flags parameter specifies which aperiodic actions to run. Specify 0 to
 * run all actions. Supported flags start with 'EcsAperiodic'. Flags identify
 * internal mechanisms and may change unannounced.
 *
 * @param world The world.
 * @param flags The flags specifying which actions to run.
 */
FLECS_API
void ecs_run_aperiodic(
    ecs_world_t *world,
    ecs_flags32_t flags);

/** Used with ecs_delete_empty_tables(). */
typedef struct ecs_delete_empty_tables_desc_t {
    /** Optional component filter for the tables to evaluate. */
    ecs_id_t id;

    /** Free table data when generation > clear_generation. */
    uint16_t clear_generation;

    /** Delete table when generation > delete_generation. */
    uint16_t delete_generation;

    /** Minimum number of component ids the table should have. */
    int32_t min_id_count;

    /** Amount of time operation is allowed to spend. */
    double time_budget_seconds;
} ecs_delete_empty_tables_desc_t;

/** Cleanup empty tables.
 * This operation cleans up empty tables that meet certain conditions. Having
 * large amounts of empty tables does not negatively impact performance of the
 * ECS, but can take up considerable amounts of memory, especially in
 * applications with many components, and many components per entity.
 *
 * The generation specifies the minimum number of times this operation has
 * to be called before an empty table is cleaned up. If a table becomes non
 * empty, the generation is reset.
 *
 * The operation allows for both a "clear" generation and a "delete"
 * generation. When the clear generation is reached, the table's
 * resources are freed (like component arrays) but the table itself is not
 * deleted. When the delete generation is reached, the empty table is deleted.
 *
 * By specifying a non-zero id the cleanup logic can be limited to tables with
 * a specific (component) id. The operation will only increase the generation
 * count of matching tables.
 *
 * The min_id_count specifies a lower bound for the number of components a table
 * should have. Often the more components a table has, the more specific it is
 * and therefore less likely to be reused.
 *
 * The time budget specifies how long the operation should take at most.
 *
 * @param world The world.
 * @param desc Configuration parameters.
 * @return Number of deleted tables.
 */
FLECS_API
int32_t ecs_delete_empty_tables(
    ecs_world_t *world,
    const ecs_delete_empty_tables_desc_t *desc);

/** Get world from poly.
 *
 * @param poly A pointer to a poly object.
 * @return The world.
 */
FLECS_API
const ecs_world_t* ecs_get_world(
    const ecs_poly_t *poly);

/** Get entity from poly.
 *
 * @param poly A pointer to a poly object.
 * @return Entity associated with the poly object.
 */
FLECS_API
ecs_entity_t ecs_get_entity(
    const ecs_poly_t *poly);

/** Test if pointer is of specified type.
 * Usage:
 *
 * @code
 * flecs_poly_is(ptr, ecs_world_t)
 * @endcode
 *
 * This operation only works for poly types.
 *
 * @param object The object to test.
 * @param type The id of the type.
 * @return True if the pointer is of the specified type.
 */
FLECS_API
bool flecs_poly_is_(
    const ecs_poly_t *object,
    int32_t type);

/** Test if pointer is of specified type.
 * @see flecs_poly_is_()
 */
#define flecs_poly_is(object, type)\
    flecs_poly_is_(object, type##_magic)

/** Make a pair id.
 * This function is equivalent to using the ecs_pair() macro, and is added for
 * convenience to make it easier for non C/C++ bindings to work with pairs.
 *
 * @param first The first element of the pair of the pair.
 * @param second The target of the pair.
 * @return A pair id.
 */
FLECS_API
ecs_id_t ecs_make_pair(
    ecs_entity_t first,
    ecs_entity_t second);

/** @} */

/** @} */

/**
 * @defgroup entities Entities
 * Functions for working with `ecs_entity_t`.
 *
 * @{
 */

/**
 * @defgroup creating_entities Creating & Deleting
 * Functions for creating and deleting entities.
 *
 * @{
 */

/** Create new entity id.
 * This operation returns an unused entity id. This operation is guaranteed to
 * return an empty entity as it does not use values set by ecs_set_scope() or
 * ecs_set_with().
 *
 * @param world The world.
 * @return The new entity id.
 */
FLECS_API
ecs_entity_t ecs_new(
    ecs_world_t *world);

/** Create new low id.
 * This operation returns a new low id. Entity ids start after the
 * FLECS_HI_COMPONENT_ID constant. This reserves a range of low ids for things
 * like components, and allows parts of the code to optimize operations.
 *
 * Note that FLECS_HI_COMPONENT_ID does not represent the maximum number of
 * components that can be created, only the maximum number of components that
 * can take advantage of these optimizations.
 *
 * This operation is guaranteed to return an empty entity as it does not use
 * values set by ecs_set_scope() or ecs_set_with().
 *
 * This operation does not recycle ids.
 *
 * @param world The world.
 * @return The new component id.
 */
FLECS_API
ecs_entity_t ecs_new_low_id(
    ecs_world_t *world);

/** Create new entity with (component) id.
 * This operation creates a new entity with an optional (component) id. When 0
 * is passed to the id parameter, no component is added to the new entity.
 *
 * @param world The world.
 * @param id The component id to initialize the new entity with.
 * @return The new entity.
 */
FLECS_API
ecs_entity_t ecs_new_w_id(
    ecs_world_t *world,
    ecs_id_t id);

/** Create new entity in table.
 * This operation creates a new entity in the specified table.
 *
 * @param world The world.
 * @param table The table to which to add the new entity.
 * @return The new entity.
 */
FLECS_API
ecs_entity_t ecs_new_w_table(
    ecs_world_t *world,
    ecs_table_t *table);

/** Find or create an entity.
 * This operation creates a new entity, or modifies an existing one. When a name
 * is set in the ecs_entity_desc_t::name field and ecs_entity_desc_t::entity is
 * not set, the operation will first attempt to find an existing entity by that
 * name. If no entity with that name can be found, it will be created.
 *
 * If both a name and entity handle are provided, the operation will check if
 * the entity name matches with the provided name. If the names do not match,
 * the function will fail and return 0.
 *
 * If an id to a non-existing entity is provided, that entity id become alive.
 *
 * See the documentation of ecs_entity_desc_t for more details.
 *
 * @param world The world.
 * @param desc Entity init parameters.
 * @return A handle to the new or existing entity, or 0 if failed.
 */
FLECS_API
ecs_entity_t ecs_entity_init(
    ecs_world_t *world,
    const ecs_entity_desc_t *desc);

/** Bulk create/populate new entities.
 * This operation bulk inserts a list of new or predefined entities into a
 * single table.
 *
 * The operation does not take ownership of component arrays provided by the
 * application. Components that are non-trivially copyable will be moved into
 * the storage.
 *
 * The operation will emit OnAdd events for each added id, and OnSet events for
 * each component that has been set.
 *
 * If no entity ids are provided by the application, the returned array of ids
 * points to an internal data structure which changes when new entities are
 * created/deleted.
 *
 * If as a result of the operation triggers are invoked that deletes
 * entities and no entity ids were provided by the application, the returned
 * array of identifiers may be incorrect. To avoid this problem, an application
 * can first call ecs_bulk_init() to create empty entities, copy the array to one
 * that is owned by the application, and then use this array to populate the
 * entities.
 *
 * @param world The world.
 * @param desc Bulk creation parameters.
 * @return Array with the list of entity ids created/populated.
 */
FLECS_API
const ecs_entity_t* ecs_bulk_init(
    ecs_world_t *world,
    const ecs_bulk_desc_t *desc);

/** Create N new entities.
 * This operation is the same as ecs_new_w_id(), but creates N entities
 * instead of one.
 *
 * @param world The world.
 * @param id The component id to create the entities with.
 * @param count The number of entities to create.
 * @return The first entity id of the newly created entities.
 */
FLECS_API
const ecs_entity_t* ecs_bulk_new_w_id(
    ecs_world_t *world,
    ecs_id_t id,
    int32_t count);

/** Clone an entity
 * This operation clones the components of one entity into another entity. If
 * no destination entity is provided, a new entity will be created. Component
 * values are not copied unless copy_value is true.
 *
 * If the source entity has a name, it will not be copied to the destination
 * entity. This is to prevent having two entities with the same name under the
 * same parent, which is not allowed.
 *
 * @param world The world.
 * @param dst The entity to copy the components to.
 * @param src The entity to copy the components from.
 * @param copy_value If true, the value of components will be copied to dst.
 * @return The destination entity.
 */
FLECS_API
ecs_entity_t ecs_clone(
    ecs_world_t *world,
    ecs_entity_t dst,
    ecs_entity_t src,
    bool copy_value);

/** Delete an entity.
 * This operation will delete an entity and all of its components. The entity id
 * will be made available for recycling. If the entity passed to ecs_delete() is
 * not alive, the operation will have no side effects.
 *
 * @param world The world.
 * @param entity The entity.
 */
FLECS_API
void ecs_delete(
    ecs_world_t *world,
    ecs_entity_t entity);

/** Delete all entities with the specified id.
 * This will delete all entities (tables) that have the specified id. The id
 * may be a wildcard and/or a pair.
 *
 * @param world The world.
 * @param id The id.
 */
FLECS_API
void ecs_delete_with(
    ecs_world_t *world,
    ecs_id_t id);

/** @} */

/**
 * @defgroup adding_removing Adding & Removing
 * Functions for adding and removing components.
 *
 * @{
 */

/** Add a (component) id to an entity.
 * This operation adds a single (component) id to an entity. If the entity
 * already has the id, this operation will have no side effects.
 *
 * @param world The world.
 * @param entity The entity.
 * @param id The id to add.
 */
FLECS_API
void ecs_add_id(
    ecs_world_t *world,
    ecs_entity_t entity,
    ecs_id_t id);

/** Remove a (component) id from an entity.
 * This operation removes a single (component) id to an entity. If the entity
 * does not have the id, this operation will have no side effects.
 *
 * @param world The world.
 * @param entity The entity.
 * @param id The id to remove.
 */
FLECS_API
void ecs_remove_id(
    ecs_world_t *world,
    ecs_entity_t entity,
    ecs_id_t id);

/** Add auto override for (component) id.
 * An auto override is a component that is automatically added to an entity when
 * it is instantiated from a prefab. Auto overrides are added to the entity that
 * is inherited from (usually a prefab). For example:
 * 
 * @code
 * ecs_entity_t prefab = ecs_insert(world,
 *   ecs_value(Position, {10, 20}),
 *   ecs_value(Mass, {100}));
 * 
 * ecs_auto_override(world, prefab, Position);
 * 
 * ecs_entity_t inst = ecs_new_w_pair(world, EcsIsA, prefab);
 * assert(ecs_owns(world, inst, Position)); // true
 * assert(ecs_owns(world, inst, Mass)); // false
 * @endcode
 * 
 * An auto override is equivalent to a manual override:
 * 
 * @code
 * ecs_entity_t prefab = ecs_insert(world,
 *   ecs_value(Position, {10, 20}),
 *   ecs_value(Mass, {100}));
 * 
 * ecs_entity_t inst = ecs_new_w_pair(world, EcsIsA, prefab);
 * assert(ecs_owns(world, inst, Position)); // false
 * ecs_add(world, inst, Position); // manual override
 * assert(ecs_owns(world, inst, Position)); // true
 * assert(ecs_owns(world, inst, Mass)); // false
 * @endcode
 * 
 * This operation is equivalent to manually adding the id with the AUTO_OVERRIDE
 * bit applied:
 *
 * @code
 * ecs_add_id(world, entity, ECS_AUTO_OVERRIDE | id);
 * @endcode
 * 
 * When a component is overridden and inherited from a prefab, the value from 
 * the prefab component is copied to the instance. When the component is not
 * inherited from a prefab, it is added to the instance as if using ecs_add_id().
 * 
 * Overriding is the default behavior on prefab instantiation. Auto overriding
 * is only useful for components with the `(OnInstantiate, Inherit)` trait.
 * When a component has the `(OnInstantiate, DontInherit)` trait and is overridden
 * the component is added, but the value from the prefab will not be copied.
 *
 * @param world The world.
 * @param entity The entity.
 * @param id The (component) id to auto override.
 */
FLECS_API
void ecs_auto_override_id(
    ecs_world_t *world,
    ecs_entity_t entity,
    ecs_id_t id);

/** Clear all components.
 * This operation will remove all components from an entity.
 *
 * @param world The world.
 * @param entity The entity.
 */
FLECS_API
void ecs_clear(
    ecs_world_t *world,
    ecs_entity_t entity);

/** Remove all instances of the specified (component) id.
 * This will remove the specified id from all entities (tables). The id may be
 * a wildcard and/or a pair.
 *
 * @param world The world.
 * @param id The id.
 */
FLECS_API
void ecs_remove_all(
    ecs_world_t *world,
    ecs_id_t id);

/** Set current with id.
 * New entities are automatically created with the specified id.
 *
 * @param world The world.
 * @param id The id.
 * @return The previous id.
 */
FLECS_API
ecs_entity_t ecs_set_with(
    ecs_world_t *world,
    ecs_id_t id);

/** Get current with id.
 * Get the id set with ecs_set_with().
 *
 * @param world The world.
 * @return The last id provided to ecs_set_with().
 */
FLECS_API
ecs_id_t ecs_get_with(
    const ecs_world_t *world);

/** @} */

/**
 * @defgroup enabling_disabling Enabling & Disabling
 * Functions for enabling/disabling entities and components.
 *
 * @{
 */

/** Enable or disable entity.
 * This operation enables or disables an entity by adding or removing the
 * #EcsDisabled tag. A disabled entity will not be matched with any systems,
 * unless the system explicitly specifies the #EcsDisabled tag.
 *
 * @param world The world.
 * @param entity The entity to enable or disable.
 * @param enabled true to enable the entity, false to disable.
 */
FLECS_API
void ecs_enable(
    ecs_world_t *world,
    ecs_entity_t entity,
    bool enabled);

/** Enable or disable component.
 * Enabling or disabling a component does not add or remove a component from an
 * entity, but prevents it from being matched with queries. This operation can
 * be useful when a component must be temporarily disabled without destroying
 * its value. It is also a more performant operation for when an application
 * needs to add/remove components at high frequency, as enabling/disabling is
 * cheaper than a regular add or remove.
 *
 * @param world The world.
 * @param entity The entity.
 * @param id The component.
 * @param enable True to enable the component, false to disable.
 */
FLECS_API
void ecs_enable_id(
    ecs_world_t *world,
    ecs_entity_t entity,
    ecs_id_t id,
    bool enable);

/** Test if component is enabled.
 * Test whether a component is currently enabled or disabled. This operation
 * will return true when the entity has the component and if it has not been
 * disabled by ecs_enable_component().
 *
 * @param world The world.
 * @param entity The entity.
 * @param id The component.
 * @return True if the component is enabled, otherwise false.
 */
FLECS_API
bool ecs_is_enabled_id(
    const ecs_world_t *world,
    ecs_entity_t entity,
    ecs_id_t id);

/** @} */

/**
 * @defgroup getting Getting & Setting
 * Functions for getting/setting components.
 *
 * @{
 */

/** Get an immutable pointer to a component.
 * This operation obtains a const pointer to the requested component. The
 * operation accepts the component entity id.
 * 
 * This operation can return inherited components reachable through an `IsA`
 * relationship.
 *
 * @param world The world.
 * @param entity The entity.
 * @param id The id of the component to get.
 * @return The component pointer, NULL if the entity does not have the component.
 *
 * @see ecs_get_mut_id()
 */
FLECS_API
const void* ecs_get_id(
    const ecs_world_t *world,
    ecs_entity_t entity,
    ecs_id_t id);

/** Get a mutable pointer to a component.
 * This operation obtains a mutable pointer to the requested component. The
 * operation accepts the component entity id.
 * 
 * Unlike ecs_get_id(), this operation does not return inherited components.
 *
 * @param world The world.
 * @param entity The entity.
 * @param id The id of the component to get.
 * @return The component pointer, NULL if the entity does not have the component.
 */
FLECS_API
void* ecs_get_mut_id(
    const ecs_world_t *world,
    ecs_entity_t entity,
    ecs_id_t id);

/** Get a mutable pointer to a component.
 * This operation returns a mutable pointer to a component. If the component did
 * not yet exist, it will be added.
 *
 * If ensure is called when the world is in deferred/readonly mode, the
 * function will:
 * - return a pointer to a temp storage if the component does not yet exist, or
 * - return a pointer to the existing component if it exists
 *
 * @param world The world.
 * @param entity The entity.
 * @param id The entity id of the component to obtain.
 * @return The component pointer.
 *
 * @see ecs_ensure_modified_id()
 * @see ecs_emplace_id()
 */
FLECS_API
void* ecs_ensure_id(
    ecs_world_t *world,
    ecs_entity_t entity,
    ecs_id_t id);

/** Combines ensure + modified in single operation.
 * This operation is a more efficient alternative to calling ecs_ensure_id() and
 * ecs_modified_id() separately. This operation is only valid when the world is in
 * deferred mode, which ensures that the Modified event is not emitted before
 * the modification takes place.
 *
 * @param world The world.
 * @param entity The entity.
 * @param id The id of the component to obtain.
 * @return The component pointer.
 */
FLECS_API
void* ecs_ensure_modified_id(
    ecs_world_t *world,
    ecs_entity_t entity,
    ecs_id_t id);

/** Create a component ref.
 * A ref is a handle to an entity + component which caches a small amount of
 * data to reduce overhead of repeatedly accessing the component. Use
 * ecs_ref_get() to get the component data.
 *
 * @param world The world.
 * @param entity The entity.
 * @param id The id of the component.
 * @return The reference.
 */
FLECS_API
ecs_ref_t ecs_ref_init_id(
    const ecs_world_t *world,
    ecs_entity_t entity,
    ecs_id_t id);

/** Get component from ref.
 * Get component pointer from ref. The ref must be created with ecs_ref_init().
 *
 * @param world The world.
 * @param ref The ref.
 * @param id The component id.
 * @return The component pointer, NULL if the entity does not have the component.
 */
FLECS_API
void* ecs_ref_get_id(
    const ecs_world_t *world,
    ecs_ref_t *ref,
    ecs_id_t id);

/** Update ref.
 * Ensures contents of ref are up to date. Same as ecs_ref_get_id(), but does not
 * return pointer to component id.
 *
 * @param world The world.
 * @param ref The ref.
 */
FLECS_API
void ecs_ref_update(
    const ecs_world_t *world,
    ecs_ref_t *ref);

/** Find record for entity. 
 * An entity record contains the table and row for the entity.
 * 
 * @param world The world.
 * @param entity The entity.
 * @return The record, NULL if the entity does not exist.
 */
FLECS_API
ecs_record_t* ecs_record_find(
    const ecs_world_t *world,
    ecs_entity_t entity);

/** Begin exclusive write access to entity.
 * This operation provides safe exclusive access to the components of an entity
 * without the overhead of deferring operations.
 *
 * When this operation is called simultaneously for the same entity more than
 * once it will throw an assert. Note that for this to happen, asserts must be
 * enabled. It is up to the application to ensure that access is exclusive, for
 * example by using a read-write mutex.
 *
 * Exclusive access is enforced at the table level, so only one entity can be
 * exclusively accessed per table. The exclusive access check is thread safe.
 *
 * This operation must be followed up with ecs_write_end().
 *
 * @param world The world.
 * @param entity The entity.
 * @return A record to the entity.
 */
FLECS_API
ecs_record_t* ecs_write_begin(
    ecs_world_t *world,
    ecs_entity_t entity);

/** End exclusive write access to entity.
 * This operation ends exclusive access, and must be called after
 * ecs_write_begin().
 *
 * @param record Record to the entity.
 */
FLECS_API
void ecs_write_end(
    ecs_record_t *record);

/** Begin read access to entity.
 * This operation provides safe read access to the components of an entity.
 * Multiple simultaneous reads are allowed per entity.
 *
 * This operation ensures that code attempting to mutate the entity's table will
 * throw an assert. Note that for this to happen, asserts must be enabled. It is
 * up to the application to ensure that this does not happen, for example by
 * using a read-write mutex.
 *
 * This operation does *not* provide the same guarantees as a read-write mutex,
 * as it is possible to call ecs_read_begin() after calling ecs_write_begin(). It is
 * up to application has to ensure that this does not happen.
 *
 * This operation must be followed up with ecs_read_end().
 *
 * @param world The world.
 * @param entity The entity.
 * @return A record to the entity.
 */
FLECS_API
const ecs_record_t* ecs_read_begin(
    ecs_world_t *world,
    ecs_entity_t entity);

/** End read access to entity.
 * This operation ends read access, and must be called after ecs_read_begin().
 *
 * @param record Record to the entity.
 */
FLECS_API
void ecs_read_end(
    const ecs_record_t *record);

/** Get entity corresponding with record.
 * This operation only works for entities that are not empty.
 *
 * @param record The record for which to obtain the entity id.
 * @return The entity id for the record.
 */
FLECS_API
ecs_entity_t ecs_record_get_entity(
    const ecs_record_t *record);

/** Get component from entity record.
 * This operation returns a pointer to a component for the entity
 * associated with the provided record. For safe access to the component, obtain
 * the record with ecs_read_begin() or ecs_write_begin().
 *
 * Obtaining a component from a record is faster than obtaining it from the
 * entity handle, as it reduces the number of lookups required.
 *
 * @param world The world.
 * @param record Record to the entity.
 * @param id The (component) id.
 * @return Pointer to component, or NULL if entity does not have the component.
 *
 * @see ecs_record_ensure_id()
 */
FLECS_API
const void* ecs_record_get_id(
    const ecs_world_t *world,
    const ecs_record_t *record,
    ecs_id_t id);

/** Same as ecs_record_get_id(), but returns a mutable pointer.
 * For safe access to the component, obtain the record with ecs_write_begin().
 *
 * @param world The world.
 * @param record Record to the entity.
 * @param id The (component) id.
 * @return Pointer to component, or NULL if entity does not have the component.
 */
FLECS_API
void* ecs_record_ensure_id(
    ecs_world_t *world,
    ecs_record_t *record,
    ecs_id_t id);

/** Test if entity for record has a (component) id.
 *
 * @param world The world.
 * @param record Record to the entity.
 * @param id The (component) id.
 * @return Whether the entity has the component.
 */
FLECS_API
bool ecs_record_has_id(
    ecs_world_t *world,
    const ecs_record_t *record,
    ecs_id_t id);

/** Get component pointer from column/record. 
 * This returns a pointer to the component using a table column index. The
 * table's column index can be found with ecs_table_get_column_index().
 * 
 * Usage:
 * @code
 * ecs_record_t *r = ecs_record_find(world, entity);
 * int32_t column = ecs_table_get_column_index(world, table, ecs_id(Position));
 * Position *ptr = ecs_record_get_by_column(r, column, sizeof(Position));
 * @endcode
 * 
 * @param record The record.
 * @param column The column index in the entity's table.
 * @param size The component size.
 * @return The component pointer.
 */
FLECS_API
void* ecs_record_get_by_column(
    const ecs_record_t *record,
    int32_t column,
    size_t size);

/** Emplace a component.
 * Emplace is similar to ecs_ensure_id() except that the component constructor
 * is not invoked for the returned pointer, allowing the component to be 
 * constructed directly in the storage.
 * 
 * When the `is_new` parameter is not provided, the operation will assert when the
 * component already exists. When the `is_new` parameter is provided, it will
 * indicate whether the returned storage has been constructed.
 * 
 * When `is_new` indicates that the storage has not yet been constructed, it must
 * be constructed by the code invoking this operation. Not constructing the
 * component will result in undefined behavior.
 *
 * @param world The world.
 * @param entity The entity.
 * @param id The component to obtain.
 * @param is_new Whether this is an existing or new component.
 * @return The (uninitialized) component pointer.
 */
FLECS_API
void* ecs_emplace_id(
    ecs_world_t *world,
    ecs_entity_t entity,
    ecs_id_t id,
    bool *is_new);

/** Signal that a component has been modified.
 * This operation is usually used after modifying a component value obtained by
 * ecs_ensure_id(). The operation will mark the component as dirty, and invoke
 * OnSet observers and hooks.
 *
 * @param world The world.
 * @param entity The entity.
 * @param id The id of the component that was modified.
 */
FLECS_API
void ecs_modified_id(
    ecs_world_t *world,
    ecs_entity_t entity,
    ecs_id_t id);

/** Set the value of a component.
 * This operation allows an application to set the value of a component. The
 * operation is equivalent to calling ecs_ensure_id() followed by
 * ecs_modified_id(). The operation will not modify the value of the passed in
 * component. If the component has a copy hook registered, it will be used to
 * copy in the component.
 *
 * If the provided entity is 0, a new entity will be created.
 *
 * @param world The world.
 * @param entity The entity.
 * @param id The id of the component to set.
 * @param size The size of the pointed-to value.
 * @param ptr The pointer to the value.
 */
FLECS_API
void ecs_set_id(
    ecs_world_t *world,
    ecs_entity_t entity,
    ecs_id_t id,
    size_t size,
    const void *ptr);

/** @} */

/**
 * @defgroup liveliness Entity Liveliness
 * Functions for testing and modifying entity liveliness.
 *
 * @{
 */

/** Test whether an entity is valid.
 * Entities that are valid can be used with API functions. Using invalid
 * entities with API operations will cause the function to panic.
 *
 * An entity is valid if it is not 0 and if it is alive.
 *
 * ecs_is_valid() will return true for ids that don't exist (alive or not alive). This
 * allows for using ids that have never been created by ecs_new_w() or similar. In
 * this the function differs from ecs_is_alive(), which will return false for
 * entities that do not yet exist.
 *
 * The operation will return false for an id that exists and is not alive, as
 * using this id with an API operation would cause it to assert.
 *
 * @param world The world.
 * @param e The entity.
 * @return True if the entity is valid, false if the entity is not valid.
 */
FLECS_API
bool ecs_is_valid(
    const ecs_world_t *world,
    ecs_entity_t e);

/** Test whether an entity is alive.
 * Entities are alive after they are created, and become not alive when they are
 * deleted. Operations that return alive ids are (amongst others) ecs_new(),
 * ecs_new_low_id() and ecs_entity_init(). Ids can be made alive with the ecs_make_alive()
 * function.
 *
 * After an id is deleted it can be recycled. Recycled ids are different from
 * the original id in that they have a different generation count. This makes it
 * possible for the API to distinguish between the two. An example:
 *
 * @code
 * ecs_entity_t e1 = ecs_new(world);
 * ecs_is_alive(world, e1);             // true
 * ecs_delete(world, e1);
 * ecs_is_alive(world, e1);             // false
 *
 * ecs_entity_t e2 = ecs_new(world);    // recycles e1
 * ecs_is_alive(world, e2);             // true
 * ecs_is_alive(world, e1);             // false
 * @endcode
 *
 * @param world The world.
 * @param e The entity.
 * @return True if the entity is alive, false if the entity is not alive.
 */
FLECS_API
bool ecs_is_alive(
    const ecs_world_t *world,
    ecs_entity_t e);

/** Remove generation from entity id.
 *
 * @param e The entity id.
 * @return The entity id without the generation count.
 */
FLECS_API
ecs_id_t ecs_strip_generation(
    ecs_entity_t e);

/** Get alive identifier.
 * In some cases an application may need to work with identifiers from which
 * the generation has been stripped. A typical scenario in which this happens is
 * when iterating relationships in an entity type.
 *
 * For example, when obtaining the parent id from a `ChildOf` relationship, the parent
 * (second element of the pair) will have been stored in a 32 bit value, which
 * cannot store the entity generation. This function can retrieve the identifier
 * with the current generation for that id.
 *
 * If the provided identifier is not alive, the function will return 0.
 *
 * @param world The world.
 * @param e The for which to obtain the current alive entity id.
 * @return The alive entity id if there is one, or 0 if the id is not alive.
 */
FLECS_API
ecs_entity_t ecs_get_alive(
    const ecs_world_t *world,
    ecs_entity_t e);

/** Ensure id is alive.
 * This operation ensures that the provided id is alive. This is useful in
 * scenarios where an application has an existing id that has not been created
 * with ecs_new_w() (such as a global constant or an id from a remote application).
 *
 * When this operation is successful it guarantees that the provided id exists,
 * is valid and is alive.
 *
 * Before this operation the id must either not be alive or have a generation
 * that is equal to the passed in entity.
 *
 * If the provided id has a non-zero generation count and the id does not exist
 * in the world, the id will be created with the specified generation.
 *
 * If the provided id is alive and has a generation count that does not match
 * the provided id, the operation will fail.
 *
 * @param world The world.
 * @param entity The entity id to make alive.
 *
 * @see ecs_make_alive_id()
 */
FLECS_API
void ecs_make_alive(
    ecs_world_t *world,
    ecs_entity_t entity);

/** Same as ecs_make_alive(), but for (component) ids.
 * An id can be an entity or pair, and can contain id flags. This operation
 * ensures that the entity (or entities, for a pair) are alive.
 *
 * When this operation is successful it guarantees that the provided id can be
 * used in operations that accept an id.
 *
 * Since entities in a pair do not encode their generation ids, this operation
 * will not fail when an entity with non-zero generation count already exists in
 * the world.
 *
 * This is different from ecs_make_alive(), which will fail if attempted with an id
 * that has generation 0 and an entity with a non-zero generation is currently
 * alive.
 *
 * @param world The world.
 * @param id The id to make alive.
 */
FLECS_API
void ecs_make_alive_id(
    ecs_world_t *world,
    ecs_id_t id);

/** Test whether an entity exists.
 * Similar as ecs_is_alive(), but ignores entity generation count.
 *
 * @param world The world.
 * @param entity The entity.
 * @return True if the entity exists, false if the entity does not exist.
 */
FLECS_API
bool ecs_exists(
    const ecs_world_t *world,
    ecs_entity_t entity);

/** Override the generation of an entity.
 * The generation count of an entity is increased each time an entity is deleted
 * and is used to test whether an entity id is alive.
 *
 * This operation overrides the current generation of an entity with the
 * specified generation, which can be useful if an entity is externally managed,
 * like for external pools, savefiles or netcode.
 * 
 * This operation is similar to ecs_make_alive(), except that it will also
 * override the generation of an alive entity.
 *
 * @param world The world.
 * @param entity Entity for which to set the generation with the new generation.
 */
FLECS_API
void ecs_set_version(
    ecs_world_t *world,
    ecs_entity_t entity);

/** @} */

/**
 * @defgroup entity_info Entity Information.
 * Get information from entity.
 *
 * @{
 */

/** Get the type of an entity.
 *
 * @param world The world.
 * @param entity The entity.
 * @return The type of the entity, NULL if the entity has no components.
 */
FLECS_API
const ecs_type_t* ecs_get_type(
    const ecs_world_t *world,
    ecs_entity_t entity);

/** Get the table of an entity.
 *
 * @param world The world.
 * @param entity The entity.
 * @return The table of the entity, NULL if the entity has no components/tags.
 */
FLECS_API
ecs_table_t* ecs_get_table(
    const ecs_world_t *world,
    ecs_entity_t entity);

/** Convert type to string.
 * The result of this operation must be freed with ecs_os_free().
 *
 * @param world The world.
 * @param type The type.
 * @return The stringified type.
 */
FLECS_API
char* ecs_type_str(
    const ecs_world_t *world,
    const ecs_type_t* type);

/** Convert table to string.
 * Same as `ecs_type_str(world, ecs_table_get_type(table))`. The result of this
 * operation must be freed with ecs_os_free().
 *
 * @param world The world.
 * @param table The table.
 * @return The stringified table type.
 *
 * @see ecs_table_get_type()
 * @see ecs_type_str()
 */
FLECS_API
char* ecs_table_str(
    const ecs_world_t *world,
    const ecs_table_t *table);

/** Convert entity to string.
 * Same as combining:
 * - ecs_get_path(world, entity)
 * - ecs_type_str(world, ecs_get_type(world, entity))
 *
 * The result of this operation must be freed with ecs_os_free().
 *
 * @param world The world.
 * @param entity The entity.
 * @return The entity path with stringified type.
 *
 * @see ecs_get_path()
 * @see ecs_type_str()
 */
FLECS_API
char* ecs_entity_str(
    const ecs_world_t *world,
    ecs_entity_t entity);

/** Test if an entity has an id.
 * This operation returns true if the entity has or inherits the specified id.
 *
 * @param world The world.
 * @param entity The entity.
 * @param id The id to test for.
 * @return True if the entity has the id, false if not.
 *
 * @see ecs_owns_id()
 */
FLECS_API
bool ecs_has_id(
    const ecs_world_t *world,
    ecs_entity_t entity,
    ecs_id_t id);

/** Test if an entity owns an id.
 * This operation returns true if the entity has the specified id. The operation
 * behaves the same as ecs_has_id(), except that it will return false for
 * components that are inherited through an `IsA` relationship.
 *
 * @param world The world.
 * @param entity The entity.
 * @param id The id to test for.
 * @return True if the entity has the id, false if not.
 */
FLECS_API
bool ecs_owns_id(
    const ecs_world_t *world,
    ecs_entity_t entity,
    ecs_id_t id);

/** Get the target of a relationship.
 * This will return a target (second element of a pair) of the entity for the
 * specified relationship. The index allows for iterating through the targets,
 * if a single entity has multiple targets for the same relationship.
 *
 * If the index is larger than the total number of instances the entity has for
 * the relationship, the operation will return 0.
 *
 * @param world The world.
 * @param entity The entity.
 * @param rel The relationship between the entity and the target.
 * @param index The index of the relationship instance.
 * @return The target for the relationship at the specified index.
 */
FLECS_API
ecs_entity_t ecs_get_target(
    const ecs_world_t *world,
    ecs_entity_t entity,
    ecs_entity_t rel,
    int32_t index);

/** Get parent (target of `ChildOf` relationship) for entity.
 * This operation is the same as calling:
 *
 * @code
 * ecs_get_target(world, entity, EcsChildOf, 0);
 * @endcode
 *
 * @param world The world.
 * @param entity The entity.
 * @return The parent of the entity, 0 if the entity has no parent.
 *
 * @see ecs_get_target()
 */
FLECS_API
ecs_entity_t ecs_get_parent(
    const ecs_world_t *world,
    ecs_entity_t entity);

/** Get the target of a relationship for a given id.
 * This operation returns the first entity that has the provided id by following
 * the specified relationship. If the entity itself has the id then entity will
 * be returned. If the id cannot be found on the entity or by following the
 * relationship, the operation will return 0.
 *
 * This operation can be used to lookup, for example, which prefab is providing
 * a component by specifying the `IsA` relationship:
 *
 * @code
 * // Is Position provided by the entity or one of its base entities?
 * ecs_get_target_for_id(world, entity, EcsIsA, ecs_id(Position))
 * @endcode
 *
 * @param world The world.
 * @param entity The entity.
 * @param rel The relationship to follow.
 * @param id The id to lookup.
 * @return The entity for which the target has been found.
 */
FLECS_API
ecs_entity_t ecs_get_target_for_id(
    const ecs_world_t *world,
    ecs_entity_t entity,
    ecs_entity_t rel,
    ecs_id_t id);

/** Return depth for entity in tree for the specified relationship.
 * Depth is determined by counting the number of targets encountered while
 * traversing up the relationship tree for rel. Only acyclic relationships are
 * supported.
 *
 * @param world The world.
 * @param entity The entity.
 * @param rel The relationship.
 * @return The depth of the entity in the tree.
 */
FLECS_API
int32_t ecs_get_depth(
    const ecs_world_t *world,
    ecs_entity_t entity,
    ecs_entity_t rel);

/** Count entities that have the specified id.
 * Returns the number of entities that have the specified id.
 *
 * @param world The world.
 * @param entity The id to search for.
 * @return The number of entities that have the id.
 */
FLECS_API
int32_t ecs_count_id(
    const ecs_world_t *world,
    ecs_id_t entity);

/** @} */


/**
 * @defgroup paths Entity Names
 * Functions for working with entity names and paths.
 *
 * @{
 */

/** Get the name of an entity.
 * This will return the name stored in `(EcsIdentifier, EcsName)`.
 *
 * @param world The world.
 * @param entity The entity.
 * @return The type of the entity, NULL if the entity has no name.
 *
 * @see ecs_set_name()
 */
FLECS_API
const char* ecs_get_name(
    const ecs_world_t *world,
    ecs_entity_t entity);

/** Get the symbol of an entity.
 * This will return the symbol stored in `(EcsIdentifier, EcsSymbol)`.
 *
 * @param world The world.
 * @param entity The entity.
 * @return The type of the entity, NULL if the entity has no name.
 *
 * @see ecs_set_symbol()
 */
FLECS_API
const char* ecs_get_symbol(
    const ecs_world_t *world,
    ecs_entity_t entity);

/** Set the name of an entity.
 * This will set or overwrite the name of an entity. If no entity is provided,
 * a new entity will be created.
 *
 * The name is stored in `(EcsIdentifier, EcsName)`.
 *
 * @param world The world.
 * @param entity The entity.
 * @param name The name.
 * @return The provided entity, or a new entity if 0 was provided.
 *
 * @see ecs_get_name()
 */
FLECS_API
ecs_entity_t ecs_set_name(
    ecs_world_t *world,
    ecs_entity_t entity,
    const char *name);

/** Set the symbol of an entity.
 * This will set or overwrite the symbol of an entity. If no entity is provided,
 * a new entity will be created.
 *
 * The symbol is stored in (EcsIdentifier, EcsSymbol).
 *
 * @param world The world.
 * @param entity The entity.
 * @param symbol The symbol.
 * @return The provided entity, or a new entity if 0 was provided.
 *
 * @see ecs_get_symbol()
 */
FLECS_API
ecs_entity_t ecs_set_symbol(
    ecs_world_t *world,
    ecs_entity_t entity,
    const char *symbol);

/** Set alias for entity.
 * An entity can be looked up using its alias from the root scope without
 * providing the fully qualified name if its parent. An entity can only have
 * a single alias.
 *
 * The symbol is stored in `(EcsIdentifier, EcsAlias)`.
 *
 * @param world The world.
 * @param entity The entity.
 * @param alias The alias.
 */
FLECS_API
void ecs_set_alias(
    ecs_world_t *world,
    ecs_entity_t entity,
    const char *alias);

/** Lookup an entity by it's path.
 * This operation is equivalent to calling:
 *
 * @code
 * ecs_lookup_path_w_sep(world, 0, path, ".", NULL, true);
 * @endcode
 *
 * @param world The world.
 * @param path The entity path.
 * @return The entity with the specified path, or 0 if no entity was found.
 *
 * @see ecs_lookup_child()
 * @see ecs_lookup_path_w_sep()
 * @see ecs_lookup_symbol()
 */
FLECS_API
ecs_entity_t ecs_lookup(
    const ecs_world_t *world,
    const char *path);

/** Lookup a child entity by name.
 * Returns an entity that matches the specified name. Only looks for entities in
 * the provided parent. If no parent is provided, look in the current scope (
 * root if no scope is provided).
 *
 * @param world The world.
 * @param parent The parent for which to lookup the child.
 * @param name The entity name.
 * @return The entity with the specified name, or 0 if no entity was found.
 *
 * @see ecs_lookup()
 * @see ecs_lookup_path_w_sep()
 * @see ecs_lookup_symbol()
 */
FLECS_API
ecs_entity_t ecs_lookup_child(
    const ecs_world_t *world,
    ecs_entity_t parent,
    const char *name);

/** Lookup an entity from a path.
 * Lookup an entity from a provided path, relative to the provided parent. The
 * operation will use the provided separator to tokenize the path expression. If
 * the provided path contains the prefix, the search will start from the root.
 *
 * If the entity is not found in the provided parent, the operation will
 * continue to search in the parent of the parent, until the root is reached. If
 * the entity is still not found, the lookup will search in the flecs.core
 * scope. If the entity is not found there either, the function returns 0.
 *
 * @param world The world.
 * @param parent The entity from which to resolve the path.
 * @param path The path to resolve.
 * @param sep The path separator.
 * @param prefix The path prefix.
 * @param recursive Recursively traverse up the tree until entity is found.
 * @return The entity if found, else 0.
 *
 * @see ecs_lookup()
 * @see ecs_lookup_child()
 * @see ecs_lookup_symbol()
 */
FLECS_API
ecs_entity_t ecs_lookup_path_w_sep(
    const ecs_world_t *world,
    ecs_entity_t parent,
    const char *path,
    const char *sep,
    const char *prefix,
    bool recursive);

/** Lookup an entity by its symbol name.
 * This looks up an entity by symbol stored in `(EcsIdentifier, EcsSymbol)`. The
 * operation does not take into account hierarchies.
 *
 * This operation can be useful to resolve, for example, a type by its C
 * identifier, which does not include the Flecs namespacing.
 *
 * @param world The world.
 * @param symbol The symbol.
 * @param lookup_as_path If not found as a symbol, lookup as path.
 * @param recursive If looking up as path, recursively traverse up the tree.
 * @return The entity if found, else 0.
 *
 * @see ecs_lookup()
 * @see ecs_lookup_child()
 * @see ecs_lookup_path_w_sep()
 */
FLECS_API
ecs_entity_t ecs_lookup_symbol(
    const ecs_world_t *world,
    const char *symbol,
    bool lookup_as_path,
    bool recursive);

/** Get a path identifier for an entity.
 * This operation creates a path that contains the names of the entities from
 * the specified parent to the provided entity, separated by the provided
 * separator. If no parent is provided the path will be relative to the root. If
 * a prefix is provided, the path will be prefixed by the prefix.
 *
 * If the parent is equal to the provided child, the operation will return an
 * empty string. If a nonzero component is provided, the path will be created by
 * looking for parents with that component.
 *
 * The returned path should be freed by the application.
 *
 * @param world The world.
 * @param parent The entity from which to create the path.
 * @param child The entity to which to create the path.
 * @param sep The separator to use between path elements.
 * @param prefix The initial character to use for root elements.
 * @return The relative entity path.
 *
 * @see ecs_get_path_w_sep_buf()
 */
FLECS_API
char* ecs_get_path_w_sep(
    const ecs_world_t *world,
    ecs_entity_t parent,
    ecs_entity_t child,
    const char *sep,
    const char *prefix);

/** Write path identifier to buffer.
 * Same as ecs_get_path_w_sep(), but writes result to an ecs_strbuf_t.
 *
 * @param world The world.
 * @param parent The entity from which to create the path.
 * @param child The entity to which to create the path.
 * @param sep The separator to use between path elements.
 * @param prefix The initial character to use for root elements.
 * @param buf The buffer to write to.
 *
 * @see ecs_get_path_w_sep()
 */
void ecs_get_path_w_sep_buf(
    const ecs_world_t *world,
    ecs_entity_t parent,
    ecs_entity_t child,
    const char *sep,
    const char *prefix,
    ecs_strbuf_t *buf,
    bool escape);

/** Find or create entity from path.
 * This operation will find or create an entity from a path, and will create any
 * intermediate entities if required. If the entity already exists, no entities
 * will be created.
 *
 * If the path starts with the prefix, then the entity will be created from the
 * root scope.
 *
 * @param world The world.
 * @param parent The entity relative to which the entity should be created.
 * @param path The path to create the entity for.
 * @param sep The separator used in the path.
 * @param prefix The prefix used in the path.
 * @return The entity.
 */
FLECS_API
ecs_entity_t ecs_new_from_path_w_sep(
    ecs_world_t *world,
    ecs_entity_t parent,
    const char *path,
    const char *sep,
    const char *prefix);

/** Add specified path to entity.
 * This operation is similar to ecs_new_from_path(), but will instead add the path
 * to an existing entity.
 *
 * If an entity already exists for the path, it will be returned instead.
 *
 * @param world The world.
 * @param entity The entity to which to add the path.
 * @param parent The entity relative to which the entity should be created.
 * @param path The path to create the entity for.
 * @param sep The separator used in the path.
 * @param prefix The prefix used in the path.
 * @return The entity.
 */
FLECS_API
ecs_entity_t ecs_add_path_w_sep(
    ecs_world_t *world,
    ecs_entity_t entity,
    ecs_entity_t parent,
    const char *path,
    const char *sep,
    const char *prefix);

/** Set the current scope.
 * This operation sets the scope of the current stage to the provided entity.
 * As a result new entities will be created in this scope, and lookups will be
 * relative to the provided scope.
 *
 * It is considered good practice to restore the scope to the old value.
 *
 * @param world The world.
 * @param scope The entity to use as scope.
 * @return The previous scope.
 *
 * @see ecs_get_scope()
 */
FLECS_API
ecs_entity_t ecs_set_scope(
    ecs_world_t *world,
    ecs_entity_t scope);

/** Get the current scope.
 * Get the scope set by ecs_set_scope(). If no scope is set, this operation will
 * return 0.
 *
 * @param world The world.
 * @return The current scope.
 */
FLECS_API
ecs_entity_t ecs_get_scope(
    const ecs_world_t *world);

/** Set a name prefix for newly created entities.
 * This is a utility that lets C modules use prefixed names for C types and
 * C functions, while using names for the entity names that do not have the
 * prefix. The name prefix is currently only used by ECS_COMPONENT.
 *
 * @param world The world.
 * @param prefix The name prefix to use.
 * @return The previous prefix.
 */
FLECS_API
const char* ecs_set_name_prefix(
    ecs_world_t *world,
    const char *prefix);

/** Set search path for lookup operations.
 * This operation accepts an array of entity ids that will be used as search
 * scopes by lookup operations. The operation returns the current search path.
 * It is good practice to restore the old search path.
 *
 * The search path will be evaluated starting from the last element.
 *
 * The default search path includes flecs.core. When a custom search path is
 * provided it overwrites the existing search path. Operations that rely on
 * looking up names from flecs.core without providing the namespace may fail if
 * the custom search path does not include flecs.core (EcsFlecsCore).
 *
 * The search path array is not copied into managed memory. The application must
 * ensure that the provided array is valid for as long as it is used as the
 * search path.
 *
 * The provided array must be terminated with a 0 element. This enables an
 * application to push/pop elements to an existing array without invoking the
 * ecs_set_lookup_path() operation again.
 *
 * @param world The world.
 * @param lookup_path 0-terminated array with entity ids for the lookup path.
 * @return Current lookup path array.
 *
 * @see ecs_get_lookup_path()
 */
FLECS_API
ecs_entity_t* ecs_set_lookup_path(
    ecs_world_t *world,
    const ecs_entity_t *lookup_path);

/** Get current lookup path.
 * Returns value set by ecs_set_lookup_path().
 *
 * @param world The world.
 * @return The current lookup path.
 */
FLECS_API
ecs_entity_t* ecs_get_lookup_path(
    const ecs_world_t *world);

/** @} */

/** @} */

/**
 * @defgroup components Components
 * Functions for registering and working with components.
 *
 * @{
 */

/** Find or create a component.
 * This operation creates a new component, or finds an existing one. The find or
 * create behavior is the same as ecs_entity_init().
 *
 * When an existing component is found, the size and alignment are verified with
 * the provided values. If the values do not match, the operation will fail.
 *
 * See the documentation of ecs_component_desc_t for more details.
 *
 * @param world The world.
 * @param desc Component init parameters.
 * @return A handle to the new or existing component, or 0 if failed.
 */
FLECS_API
ecs_entity_t ecs_component_init(
    ecs_world_t *world,
    const ecs_component_desc_t *desc);

/** Get the type for an id.
 * This function returns the type information for an id. The specified id can be
 * any valid id. For the rules on how type information is determined based on
 * id, see ecs_get_typeid().
 *
 * @param world The world.
 * @param id The id.
 * @return The type information of the id.
 */
FLECS_API
const ecs_type_info_t* ecs_get_type_info(
    const ecs_world_t *world,
    ecs_id_t id);

/** Register hooks for component.
 * Hooks allow for the execution of user code when components are constructed,
 * copied, moved, destructed, added, removed or set. Hooks can be assigned as
 * as long as a component has not yet been used (added to an entity).
 *
 * The hooks that are currently set can be accessed with ecs_get_type_info().
 *
 * @param world The world.
 * @param id The component id for which to register the actions
 * @param hooks Type that contains the component actions.
 */
FLECS_API
void ecs_set_hooks_id(
    ecs_world_t *world,
    ecs_entity_t id,
    const ecs_type_hooks_t *hooks);

/** Get hooks for component.
 *
 * @param world The world.
 * @param id The component id for which to retrieve the hooks.
 * @return The hooks for the component, or NULL if not registered.
 */
FLECS_API
const ecs_type_hooks_t* ecs_get_hooks_id(
    const ecs_world_t *world,
    ecs_entity_t id);

/** @} */

/**
 * @defgroup ids Ids
 * Functions for working with `ecs_id_t`.
 *
 * @{
 */

/** Returns whether specified id a tag.
 * This operation returns whether the specified type is a tag (a component
 * without data/size).
 *
 * An id is a tag when:
 * - it is an entity without the EcsComponent component
 * - it has an EcsComponent with size member set to 0
 * - it is a pair where both elements are a tag
 * - it is a pair where the first element has the #EcsPairIsTag tag
 *
 * @param world The world.
 * @param id The id.
 * @return Whether the provided id is a tag.
 */
FLECS_API
bool ecs_id_is_tag(
    const ecs_world_t *world,
    ecs_id_t id);

/** Returns whether specified id is in use.
 * This operation returns whether an id is in use in the world. An id is in use
 * if it has been added to one or more tables.
 *
 * @param world The world.
 * @param id The id.
 * @return Whether the id is in use.
 */
FLECS_API
bool ecs_id_in_use(
    const ecs_world_t *world,
    ecs_id_t id);

/** Get the type for an id.
 * This operation returns the component id for an id, if the id is associated
 * with a type. For a regular component with a non-zero size (an entity with the
 * EcsComponent component) the operation will return the entity itself.
 *
 * For an entity that does not have the EcsComponent component, or with an
 * EcsComponent value with size 0, the operation will return 0.
 *
 * For a pair id the operation will return the type associated with the pair, by
 * applying the following queries in order:
 * - The first pair element is returned if it is a component
 * - 0 is returned if the relationship entity has the Tag property
 * - The second pair element is returned if it is a component
 * - 0 is returned.
 *
 * @param world The world.
 * @param id The id.
 * @return The type id of the id.
 */
FLECS_API
ecs_entity_t ecs_get_typeid(
    const ecs_world_t *world,
    ecs_id_t id);

/** Utility to match an id with a pattern.
 * This operation returns true if the provided pattern matches the provided
 * id. The pattern may contain a wildcard (or wildcards, when a pair).
 *
 * @param id The id.
 * @param pattern The pattern to compare with.
 * @return Whether the id matches the pattern.
 */
FLECS_API
bool ecs_id_match(
    ecs_id_t id,
    ecs_id_t pattern);

/** Utility to check if id is a pair.
 *
 * @param id The id.
 * @return True if id is a pair.
 */
FLECS_API
bool ecs_id_is_pair(
    ecs_id_t id);

/** Utility to check if id is a wildcard.
 *
 * @param id The id.
 * @return True if id is a wildcard or a pair containing a wildcard.
 */
FLECS_API
bool ecs_id_is_wildcard(
    ecs_id_t id);

/** Utility to check if id is valid.
 * A valid id is an id that can be added to an entity. Invalid ids are:
 * - ids that contain wildcards
 * - ids that contain invalid entities
 * - ids that are 0 or contain 0 entities
 *
 * Note that the same rules apply to removing from an entity, with the exception
 * of wildcards.
 *
 * @param world The world.
 * @param id The id.
 * @return True if the id is valid.
 */
FLECS_API
bool ecs_id_is_valid(
    const ecs_world_t *world,
    ecs_id_t id);

/** Get flags associated with id.
 * This operation returns the internal flags (see api_flags.h) that are
 * associated with the provided id.
 *
 * @param world The world.
 * @param id The id.
 * @return Flags associated with the id, or 0 if the id is not in use.
 */
FLECS_API
ecs_flags32_t ecs_id_get_flags(
    const ecs_world_t *world,
    ecs_id_t id);

/** Convert id flag to string.
 * This operation converts an id flag to a string.
 *
 * @param id_flags The id flag.
 * @return The id flag string, or NULL if no valid id is provided.
 */
FLECS_API
const char* ecs_id_flag_str(
    ecs_id_t id_flags);

/** Convert (component) id to string.
 * This operation interprets the structure of an id and converts it to a string.
 *
 * @param world The world.
 * @param id The id to convert to a string.
 * @return The id converted to a string.
 */
FLECS_API
char* ecs_id_str(
    const ecs_world_t *world,
    ecs_id_t id);

/** Write (component) id string to buffer.
 * Same as ecs_id_str() but writes result to ecs_strbuf_t.
 *
 * @param world The world.
 * @param id The id to convert to a string.
 * @param buf The buffer to write to.
 */
FLECS_API
void ecs_id_str_buf(
    const ecs_world_t *world,
    ecs_id_t id,
    ecs_strbuf_t *buf);

/** Convert string to a (component) id.
 * This operation is the reverse of ecs_id_str(). The FLECS_SCRIPT addon
 * is required for this operation to work.
 *
 * @param world The world.
 * @param expr The string to convert to an id.
 */
FLECS_API
ecs_id_t ecs_id_from_str(
    const ecs_world_t *world,
    const char *expr);

/** @} */

/**
 * @defgroup queries Queries
 * @brief Functions for working with `ecs_term_t` and `ecs_query_t`.
 * @{
 */

/** Test whether term id is set.
 *
 * @param id The term id.
 * @return True when set, false when not set.
 */
FLECS_API 
bool ecs_term_ref_is_set(
    const ecs_term_ref_t *id);

/** Test whether a term is set.
 * This operation can be used to test whether a term has been initialized with
 * values or whether it is empty.
 *
 * An application generally does not need to invoke this operation. It is useful
 * when initializing a 0-initialized array of terms (like in ecs_term_desc_t) as
 * this operation can be used to find the last initialized element.
 *
 * @param term The term.
 * @return True when set, false when not set.
 */
FLECS_API
bool ecs_term_is_initialized(
    const ecs_term_t *term);

/** Is term matched on $this variable.
 * This operation checks whether a term is matched on the $this variable, which
 * is the default source for queries.
 *
 * A term has a $this source when:
 * - ecs_term_t::src::id is EcsThis
 * - ecs_term_t::src::flags is EcsIsVariable
 *
 * If ecs_term_t::src is not populated, it will be automatically initialized to
 * the $this source for the created query.
 *
 * @param term The term.
 * @return True if term matches $this, false if not.
 */
FLECS_API
bool ecs_term_match_this(
    const ecs_term_t *term);

/** Is term matched on 0 source.
 * This operation checks whether a term is matched on a 0 source. A 0 source is
 * a term that isn't matched against anything, and can be used just to pass
 * (component) ids to a query iterator.
 *
 * A term has a 0 source when:
 * - ecs_term_t::src::id is 0
 * - ecs_term_t::src::flags has EcsIsEntity set
 *
 * @param term The term.
 * @return True if term has 0 source, false if not.
 */
FLECS_API
bool ecs_term_match_0(
    const ecs_term_t *term);

/** Convert term to string expression.
 * Convert term to a string expression. The resulting expression is equivalent
 * to the same term, with the exception of And & Or operators.
 *
 * @param world The world.
 * @param term The term.
 * @return The term converted to a string.
 */
FLECS_API
char* ecs_term_str(
    const ecs_world_t *world,
    const ecs_term_t *term);

/** Convert query to string expression.
 * Convert query to a string expression. The resulting expression can be
 * parsed to create the same query.
 * 
 * @param query The query.
 * @return The query converted to a string.
 */
FLECS_API 
char* ecs_query_str(
    const ecs_query_t *query); 

/** @} */

/**
 * @defgroup each_iter Each iterator
 * @brief Find all entities that have a single (component) id.
 * @{
 */

/** Iterate all entities with specified (component id). 
 * This returns an iterator that yields all entities with a single specified
 * component. This is a much lighter weight operation than creating and 
 * iterating a query.
 * 
 * Usage:
 * @code
 * ecs_iter_t it = ecs_each(world, Player);
 * while (ecs_each_next(&it)) {
 *   for (int i = 0; i < it.count; i ++) {
 *     // Iterate as usual.
 *   }
 * }
 * @endcode
 * 
 * If the specified id is a component, it is possible to access the component
 * pointer with ecs_field just like with regular queries:
 * 
 * @code
 * ecs_iter_t it = ecs_each(world, Position);
 * while (ecs_each_next(&it)) {
 *   Position *p = ecs_field(&it, Position, 0);
 *   for (int i = 0; i < it.count; i ++) {
 *     // Iterate as usual.
 *   }
 * }
 * @endcode
 * 
 * @param world The world.
 * @param id The (component) id to iterate.
 * @return An iterator that iterates all entities with the (component) id.
*/
FLECS_API
ecs_iter_t ecs_each_id(
    const ecs_world_t *world,
    ecs_id_t id);

/** Progress an iterator created with ecs_each_id().
 * 
 * @param it The iterator.
 * @return True if the iterator has more results, false if not.
 */
FLECS_API
bool ecs_each_next(
    ecs_iter_t *it);

/** Iterate children of parent.
 * Equivalent to:
 * @code
 * ecs_iter_t it = ecs_each_id(world, ecs_pair(EcsChildOf, parent));
 * @endcode
 * 
 * @param world The world.
 * @param parent The parent.
 * @return An iterator that iterates all children of the parent.
 *
 * @see ecs_each_id()
*/
FLECS_API
ecs_iter_t ecs_children(
    const ecs_world_t *world,
    ecs_entity_t parent);

/** Progress an iterator created with ecs_children().
 * 
 * @param it The iterator.
 * @return True if the iterator has more results, false if not.
 */
FLECS_API
bool ecs_children_next(
    ecs_iter_t *it);

/** @} */

/**
 * @defgroup queries Queries
 * Functions for working with `ecs_query_t`.
 *
 * @{
 */

/** Create a query.
 * 
 * @param world The world.
 * @param desc The descriptor (see ecs_query_desc_t)
 * @return The query.
 */
FLECS_API
ecs_query_t* ecs_query_init(
    ecs_world_t *world,
    const ecs_query_desc_t *desc);

/** Delete a query.
 *
 * @param query The query.
 */
FLECS_API
void ecs_query_fini(
    ecs_query_t *query);

/** Find variable index.
 * This operation looks up the index of a variable in the query. This index can
 * be used in operations like ecs_iter_set_var() and ecs_iter_get_var().
 *
 * @param query The query.
 * @param name The variable name.
 * @return The variable index.
 */
FLECS_API
int32_t ecs_query_find_var(
    const ecs_query_t *query,
    const char *name);    

/** Get variable name.
 * This operation returns the variable name for an index.
 *
 * @param query The query.
 * @param var_id The variable index.
 * @return The variable name.
 */
FLECS_API
const char* ecs_query_var_name(
    const ecs_query_t *query,
    int32_t var_id);

/** Test if variable is an entity.
 * Internally the query engine has entity variables and table variables. When
 * iterating through query variables (by using ecs_query_variable_count()) only
 * the values for entity variables are accessible. This operation enables an
 * application to check if a variable is an entity variable.
 *
 * @param query The query.
 * @param var_id The variable id.
 * @return Whether the variable is an entity variable.
 */
FLECS_API
bool ecs_query_var_is_entity(
    const ecs_query_t *query,
    int32_t var_id);  

/** Create a query iterator.
 * Use an iterator to iterate through the entities that match an entity. Queries
 * can return multiple results, and have to be iterated by repeatedly calling
 * ecs_query_next() until the operation returns false.
 * 
 * Depending on the query, a single result can contain an entire table, a range
 * of entities in a table, or a single entity. Iteration code has an inner and
 * an outer loop. The outer loop loops through the query results, and typically
 * corresponds with a table. The inner loop loops entities in the result.
 * 
 * Example:
 * @code
 * ecs_iter_t it = ecs_query_iter(world, q);
 * 
 * while (ecs_query_next(&it)) {
 *   Position *p = ecs_field(&it, Position, 0);
 *   Velocity *v = ecs_field(&it, Velocity, 1);
 * 
 *   for (int i = 0; i < it.count; i ++) {
 *     p[i].x += v[i].x;
 *     p[i].y += v[i].y;
 *   }
 * }
 * @endcode
 * 
 * The world passed into the operation must be either the actual world or the
 * current stage, when iterating from a system. The stage is accessible through
 * the it.world member.
 * 
 * Example:
 * @code
 * void MySystem(ecs_iter_t *it) {
 *   ecs_query_t *q = it->ctx; // Query passed as system context
 * 
 *   // Create query iterator from system stage
 *   ecs_iter_t qit = ecs_query_iter(it->world, q);
 *   while (ecs_query_next(&qit)) {
 *     // Iterate as usual
 *   }
 * }
 * @endcode
 * 
 * If query iteration is stopped without the last call to ecs_query_next() 
 * returning false, iterator resources need to be cleaned up explicitly
 * with ecs_iter_fini().
 * 
 * Example:
 * @code
 * ecs_iter_t it = ecs_query_iter(world, q);
 * 
 * while (ecs_query_next(&it)) {
 *   if (!ecs_field_is_set(&it, 0)) {
 *     ecs_iter_fini(&it); // Free iterator resources
 *     break;
 *   }
 * 
 *   for (int i = 0; i < it.count; i ++) {
 *     // ...
 *   }
 * }
 * @endcode
 *
 * @param world The world.
 * @param query The query.
 * @return An iterator.
 *
 * @see ecs_query_next()
 */
FLECS_API
ecs_iter_t ecs_query_iter(
    const ecs_world_t *world,
    const ecs_query_t *query);

/** Progress query iterator.
 *
 * @param it The iterator.
 * @return True if the iterator has more results, false if not.
 *
 * @see ecs_query_iter()
 */
FLECS_API
bool ecs_query_next(
    ecs_iter_t *it);

/** Match entity with query.
 * This operation matches an entity with a query and returns the result of the
 * match in the "it" out parameter. An application should free the iterator
 * resources with ecs_iter_fini() if this function returns true.
 * 
 * Usage:
 * @code
 * ecs_iter_t it;
 * if (ecs_query_has(q, e, &it)) {
 *   ecs_iter_fini(&it);
 * }
 * @endcode
 * 
 * @param query The query.
 * @param entity The entity to match
 * @param it The iterator with matched data.
 * @return True if entity matches the query, false if not.
 */
FLECS_API
bool ecs_query_has(
    ecs_query_t *query,
    ecs_entity_t entity,
    ecs_iter_t *it);

/** Match table with query.
 * This operation matches a table with a query and returns the result of the
 * match in the "it" out parameter. An application should free the iterator
 * resources with ecs_iter_fini() if this function returns true.
 * 
 * Usage:
 * @code
 * ecs_iter_t it;
 * if (ecs_query_has_table(q, t, &it)) {
 *   ecs_iter_fini(&it);
 * }
 * @endcode
 * 
 * @param query The query.
 * @param table The table to match
 * @param it The iterator with matched data.
 * @return True if table matches the query, false if not.
 */
FLECS_API
bool ecs_query_has_table(
    ecs_query_t *query,
    ecs_table_t *table,
    ecs_iter_t *it);

/** Match range with query.
 * This operation matches a range with a query and returns the result of the
 * match in the "it" out parameter. An application should free the iterator
 * resources with ecs_iter_fini() if this function returns true.
 * 
 * The entire range must match the query for the operation to return true.
 * 
 * Usage:
 * @code
 * ecs_table_range_t range = {
 *   .table = table,
 *   .offset = 1,
 *   .count = 2
 * };
 * 
 * ecs_iter_t it;
 * if (ecs_query_has_range(q, &range, &it)) {
 *   ecs_iter_fini(&it);
 * }
 * @endcode
 * 
 * @param query The query.
 * @param range The range to match
 * @param it The iterator with matched data.
 * @return True if range matches the query, false if not.
 */
FLECS_API
bool ecs_query_has_range(
    ecs_query_t *query,
    ecs_table_range_t *range,
    ecs_iter_t *it);

/** Returns how often a match event happened for a cached query. 
 * This operation can be used to determine whether the query cache has been 
 * updated with new tables.
 * 
 * @param query The query.
 * @return The number of match events happened.
 */
FLECS_API
int32_t ecs_query_match_count(
    const ecs_query_t *query);

/** Convert query to a string.
 * This will convert the query program to a string which can aid in debugging
 * the behavior of a query.
 *
 * The returned string must be freed with ecs_os_free().
 *
 * @param query The query.
 * @return The query plan.
 */
FLECS_API
char* ecs_query_plan(
    const ecs_query_t *query);

/** Convert query to string with profile.
 * To use this you must set the EcsIterProfile flag on an iterator before
 * starting iteration:
 *
 * @code
 *   it.flags |= EcsIterProfile
 * @endcode
 * 
 * The returned string must be freed with ecs_os_free().
 *
 * @param query The query.
 * @param it The iterator with profile data.
 * @return The query plan with profile data.
 */
FLECS_API
char* ecs_query_plan_w_profile(
    const ecs_query_t *query,
    const ecs_iter_t *it);

/** Populate variables from key-value string.
 * Convenience function to set query variables from a key-value string separated
 * by comma's. The string must have the following format:
 *
 * @code
 *   var_a: value, var_b: value
 * @endcode
 *
 * The key-value list may optionally be enclosed in parenthesis.
 * 
 * This function uses the script addon.
 *
 * @param query The query.
 * @param it The iterator for which to set the variables.
 * @param expr The key-value expression.
 * @return Pointer to the next character after the last parsed one.
 */
FLECS_API
const char* ecs_query_args_parse(
    ecs_query_t *query,
    ecs_iter_t *it,
    const char *expr);

/** Returns whether the query data changed since the last iteration.
 * The operation will return true after:
 * - new entities have been matched with
 * - new tables have been matched/unmatched with
 * - matched entities were deleted
 * - matched components were changed
 *
 * The operation will not return true after a write-only (EcsOut) or filter
 * (EcsInOutNone) term has changed, when a term is not matched with the
 * current table (This subject) or for tag terms.
 *
 * The changed state of a table is reset after it is iterated. If an iterator was
 * not iterated until completion, tables may still be marked as changed.
 *
 * If no iterator is provided the operation will return the changed state of the
 * all matched tables of the query.
 *
 * If an iterator is provided, the operation will return the changed state of
 * the currently returned iterator result. The following preconditions must be
 * met before using an iterator with change detection:
 *
 * - The iterator is a query iterator (created with ecs_query_iter())
 * - The iterator must be valid (ecs_query_next() must have returned true)
 *
 * @param query The query (optional if 'it' is provided).
 * @return true if entities changed, otherwise false.
 */
FLECS_API
bool ecs_query_changed(
    ecs_query_t *query);

/** Get query object.
 * Returns the query object. Can be used to access various information about
 * the query.
 *
 * @param world The world.
 * @param query The query.
 * @return The query object.
 */
FLECS_API
const ecs_query_t* ecs_query_get(
    const ecs_world_t *world,
    ecs_entity_t query);

/** Skip a table while iterating.
 * This operation lets the query iterator know that a table was skipped while
 * iterating. A skipped table will not reset its changed state, and the query
 * will not update the dirty flags of the table for its out columns.
 *
 * Only valid iterators must be provided (next has to be called at least once &
 * return true) and the iterator must be a query iterator.
 *
 * @param it The iterator result to skip.
 */
FLECS_API
void ecs_iter_skip(
    ecs_iter_t *it);

/** Set group to iterate for query iterator.
 * This operation limits the results returned by the query to only the selected
 * group id. The query must have a group_by function, and the iterator must
 * be a query iterator.
 *
 * Groups are sets of tables that are stored together in the query cache based
 * on a group id, which is calculated per table by the group_by function. To
 * iterate a group, an iterator only needs to know the first and last cache node
 * for that group, which can both be found in a fast O(1) operation.
 *
 * As a result, group iteration is one of the most efficient mechanisms to
 * filter out large numbers of entities, even if those entities are distributed
 * across many tables. This makes it a good fit for things like dividing up
 * a world into cells, and only iterating cells close to a player.
 *
 * The group to iterate must be set before the first call to ecs_query_next(). No
 * operations that can add/remove components should be invoked between calling
 * ecs_iter_set_group() and ecs_query_next().
 *
 * @param it The query iterator.
 * @param group_id The group to iterate.
 */
FLECS_API
void ecs_iter_set_group(
    ecs_iter_t *it,
    uint64_t group_id);

/** Get context of query group.
 * This operation returns the context of a query group as returned by the
 * on_group_create callback.
 *
 * @param query The query.
 * @param group_id The group for which to obtain the context.
 * @return The group context, NULL if the group doesn't exist.
 */
FLECS_API
void* ecs_query_get_group_ctx(
    const ecs_query_t *query,
    uint64_t group_id);

/** Get information about query group.
 * This operation returns information about a query group, including the group
 * context returned by the on_group_create callback.
 *
 * @param query The query.
 * @param group_id The group for which to obtain the group info.
 * @return The group info, NULL if the group doesn't exist.
 */
FLECS_API
const ecs_query_group_info_t* ecs_query_get_group_info(
    const ecs_query_t *query,
    uint64_t group_id);

/** Struct returned by ecs_query_count(). */
typedef struct ecs_query_count_t {
    int32_t results;      /**< Number of results returned by query. */
    int32_t entities;     /**< Number of entities returned by query. */
    int32_t tables;       /**< Number of tables returned by query. */
    int32_t empty_tables; /**< Number of empty tables returned by query. */
} ecs_query_count_t;

/** Returns number of entities and results the query matches with.
 * Only entities matching the $this variable as source are counted.
 *
 * @param query The query.
 * @return The number of matched entities.
 */
FLECS_API
ecs_query_count_t ecs_query_count(
    const ecs_query_t *query);

/** Does query return one or more results. 
 * 
 * @param query The query.
 * @return True if query matches anything, false if not.
 */
FLECS_API
bool ecs_query_is_true(
    const ecs_query_t *query);

/** Get query used to populate cache.
 * This operation returns the query that is used to populate the query cache.
 * For queries that are can be entirely cached, the returned query will be 
 * equivalent to the query passed to ecs_query_get_cache_query().
 *
 * @param query The query.
 * @return The query used to populate the cache, NULL if query is not cached.
 */
FLECS_API
const ecs_query_t* ecs_query_get_cache_query(
    const ecs_query_t *query);

/** @} */

/**
 * @defgroup observers Observers
 * Functions for working with events and observers.
 *
 * @{
 */

/** Send event.
 * This sends an event to matching triggers & is the mechanism used by flecs
 * itself to send `OnAdd`, `OnRemove`, etc events.
 *
 * Applications can use this function to send custom events, where a custom
 * event can be any regular entity.
 *
 * Applications should not send builtin flecs events, as this may violate
 * assumptions the code makes about the conditions under which those events are
 * sent.
 *
 * Triggers are invoked synchronously. It is therefore safe to use stack-based
 * data as event context, which can be set in the "param" member.
 *
 * @param world The world.
 * @param desc Event parameters.
 *
 * @see ecs_enqueue()
 */
FLECS_API
void ecs_emit(
    ecs_world_t *world,
    ecs_event_desc_t *desc);

/** Enqueue event.
 * Same as ecs_emit(), but enqueues an event in the command queue instead. The
 * event will be emitted when ecs_defer_end() is called.
 * 
 * If this operation is called when the provided world is not in deferred mode
 * it behaves just like ecs_emit().
 * 
 * @param world The world.
 * @param desc Event parameters.
*/
FLECS_API
void ecs_enqueue(
    ecs_world_t *world,
    ecs_event_desc_t *desc);

/** Create observer.
 * Observers are like triggers, but can subscribe for multiple terms. An
 * observer only triggers when the source of the event meets all terms.
 *
 * See the documentation for ecs_observer_desc_t for more details.
 *
 * @param world The world.
 * @param desc The observer creation parameters.
 * @return The observer, or 0 if the operation failed.
 */
FLECS_API
ecs_entity_t ecs_observer_init(
    ecs_world_t *world,
    const ecs_observer_desc_t *desc);

/** Get observer object.
 * Returns the observer object. Can be used to access various information about
 * the observer, like the query and context.
 *
 * @param world The world.
 * @param observer The observer.
 * @return The observer object.
 */
FLECS_API
const ecs_observer_t* ecs_observer_get(
    const ecs_world_t *world,
    ecs_entity_t observer);

/** @} */

/**
 * @defgroup iterator Iterators
 * Functions for working with `ecs_iter_t`.
 *
 * @{
 */

/** Progress any iterator.
 * This operation is useful in combination with iterators for which it is not
 * known what created them. Example use cases are functions that should accept
 * any kind of iterator (such as serializers) or iterators created from poly
 * objects.
 *
 * This operation is slightly slower than using a type-specific iterator (e.g.
 * ecs_query_next, ecs_query_next) as it has to call a function pointer which
 * introduces a level of indirection.
 *
 * @param it The iterator.
 * @return True if iterator has more results, false if not.
 */
FLECS_API
bool ecs_iter_next(
    ecs_iter_t *it);

/** Cleanup iterator resources.
 * This operation cleans up any resources associated with the iterator.
 *
 * This operation should only be used when an iterator is not iterated until
 * completion (next has not yet returned false). When an iterator is iterated
 * until completion, resources are automatically freed.
 *
 * @param it The iterator.
 */
FLECS_API
void ecs_iter_fini(
    ecs_iter_t *it);

/** Count number of matched entities in query.
 * This operation returns the number of matched entities. If a query contains no
 * matched entities but still yields results (e.g. it has no terms with This
 * sources) the operation will return 0.
 *
 * To determine the number of matched entities, the operation iterates the
 * iterator until it yields no more results.
 *
 * @param it The iterator.
 * @return True if iterator has more results, false if not.
 */
FLECS_API
int32_t ecs_iter_count(
    ecs_iter_t *it);

/** Test if iterator is true.
 * This operation will return true if the iterator returns at least one result.
 * This is especially useful in combination with fact-checking queries (see the
 * queries addon).
 *
 * The operation requires a valid iterator. After the operation is invoked, the
 * application should no longer invoke next on the iterator and should treat it
 * as if the iterator is iterated until completion.
 *
 * @param it The iterator.
 * @return true if the iterator returns at least one result.
 */
FLECS_API
bool ecs_iter_is_true(
    ecs_iter_t *it);

/** Get first matching entity from iterator.
 * After this operation the application should treat the iterator as if it has
 * been iterated until completion.
 *
 * @param it The iterator.
 * @return The first matching entity, or 0 if no entities were matched.
 */
FLECS_API
ecs_entity_t ecs_iter_first(
    ecs_iter_t *it);

/** Set value for iterator variable.
 * This constrains the iterator to return only results for which the variable
 * equals the specified value. The default value for all variables is
 * EcsWildcard, which means the variable can assume any value.
 *
 * Example:
 *
 * @code
 * // Query that matches (Eats, *)
 * ecs_query_t *q = ecs_query(world, {
 *   .terms = {
 *     { .first.id = Eats, .second.name = "$food" }
 *   }
 * });
 * 
 * int food_var = ecs_query_find_var(r, "food");
 * 
 * // Set Food to Apples, so we're only matching (Eats, Apples)
 * ecs_iter_t it = ecs_query_iter(world, q);
 * ecs_iter_set_var(&it, food_var, Apples);
 * 
 * while (ecs_query_next(&it)) {
 *   for (int i = 0; i < it.count; i ++) {
 *     // iterate as usual
 *   }
 * }
 * @endcode
 *
 * The variable must be initialized after creating the iterator and before the
 * first call to next.
 *
 * @param it The iterator.
 * @param var_id The variable index.
 * @param entity The entity variable value.
 *
 * @see ecs_iter_set_var_as_range()
 * @see ecs_iter_set_var_as_table()
 */
FLECS_API
void ecs_iter_set_var(
    ecs_iter_t *it,
    int32_t var_id,
    ecs_entity_t entity);

/** Same as ecs_iter_set_var(), but for a table.
 * This constrains the variable to all entities in a table.
 *
 * @param it The iterator.
 * @param var_id The variable index.
 * @param table The table variable value.
 *
 * @see ecs_iter_set_var()
 * @see ecs_iter_set_var_as_range()
 */
FLECS_API
void ecs_iter_set_var_as_table(
    ecs_iter_t *it,
    int32_t var_id,
    const ecs_table_t *table);

/** Same as ecs_iter_set_var(), but for a range of entities
 * This constrains the variable to a range of entities in a table.
 *
 * @param it The iterator.
 * @param var_id The variable index.
 * @param range The range variable value.
 *
 * @see ecs_iter_set_var()
 * @see ecs_iter_set_var_as_table()
 */
FLECS_API
void ecs_iter_set_var_as_range(
    ecs_iter_t *it,
    int32_t var_id,
    const ecs_table_range_t *range);

/** Get value of iterator variable as entity.
 * A variable can be interpreted as entity if it is set to an entity, or if it
 * is set to a table range with count 1.
 *
 * This operation can only be invoked on valid iterators. The variable index
 * must be smaller than the total number of variables provided by the iterator
 * (as set in ecs_iter_t::variable_count).
 *
 * @param it The iterator.
 * @param var_id The variable index.
 * @return The variable value.
 */
FLECS_API
ecs_entity_t ecs_iter_get_var(
    ecs_iter_t *it,
    int32_t var_id);

/** Get value of iterator variable as table.
 * A variable can be interpreted as table if it is set as table range with
 * both offset and count set to 0, or if offset is 0 and count matches the
 * number of elements in the table.
 *
 * This operation can only be invoked on valid iterators. The variable index
 * must be smaller than the total number of variables provided by the iterator
 * (as set in ecs_iter_t::variable_count).
 *
 * @param it The iterator.
 * @param var_id The variable index.
 * @return The variable value.
 */
FLECS_API
ecs_table_t* ecs_iter_get_var_as_table(
    ecs_iter_t *it,
    int32_t var_id);

/** Get value of iterator variable as table range.
 * A value can be interpreted as table range if it is set as table range, or if
 * it is set to an entity with a non-empty type (the entity must have at least
 * one component, tag or relationship in its type).
 *
 * This operation can only be invoked on valid iterators. The variable index
 * must be smaller than the total number of variables provided by the iterator
 * (as set in ecs_iter_t::variable_count).
 *
 * @param it The iterator.
 * @param var_id The variable index.
 * @return The variable value.
 */
FLECS_API
ecs_table_range_t ecs_iter_get_var_as_range(
    ecs_iter_t *it,
    int32_t var_id);

/** Returns whether variable is constrained.
 * This operation returns true for variables set by one of the ecs_iter_set_var*
 * operations.
 *
 * A constrained variable is guaranteed not to change values while results are
 * being iterated.
 *
 * @param it The iterator.
 * @param var_id The variable index.
 * @return Whether the variable is constrained to a specified value.
 */
FLECS_API
bool ecs_iter_var_is_constrained(
    ecs_iter_t *it,
    int32_t var_id);

/** Returns whether current iterator result has changed.
 * This operation must be used in combination with a query that supports change
 * detection (e.g. is cached). The operation returns whether the currently
 * iterated result has changed since the last time it was iterated by the query.
 * 
 * Change detection works on a per-table basis. Changes to individual entities
 * cannot be detected this way.
 * 
 * @param it The iterator.
 * @return True if the result changed, false if it didn't.
*/
FLECS_API
bool ecs_iter_changed(
    ecs_iter_t *it);

/** Convert iterator to string.
 * Prints the contents of an iterator to a string. Useful for debugging and/or
 * testing the output of an iterator.
 *
 * The function only converts the currently iterated data to a string. To
 * convert all data, the application has to manually call the next function and
 * call ecs_iter_str() on each result.
 *
 * @param it The iterator.
 * @return A string representing the contents of the iterator.
 */
FLECS_API
char* ecs_iter_str(
    const ecs_iter_t *it);

/** Create a paged iterator.
 * Paged iterators limit the results to those starting from 'offset', and will
 * return at most 'limit' results.
 *
 * The iterator must be iterated with ecs_page_next().
 *
 * A paged iterator acts as a passthrough for data exposed by the parent
 * iterator, so that any data provided by the parent will also be provided by
 * the paged iterator.
 *
 * @param it The source iterator.
 * @param offset The number of entities to skip.
 * @param limit The maximum number of entities to iterate.
 * @return A page iterator.
 */
FLECS_API
ecs_iter_t ecs_page_iter(
    const ecs_iter_t *it,
    int32_t offset,
    int32_t limit);

/** Progress a paged iterator.
 * Progresses an iterator created by ecs_page_iter().
 *
 * @param it The iterator.
 * @return true if iterator has more results, false if not.
 */
FLECS_API
bool ecs_page_next(
    ecs_iter_t *it);

/** Create a worker iterator.
 * Worker iterators can be used to equally divide the number of matched entities
 * across N resources (usually threads). Each resource will process the total
 * number of matched entities divided by 'count'.
 *
 * Entities are distributed across resources such that the distribution is
 * stable between queries. Two queries that match the same table are guaranteed
 * to match the same entities in that table.
 *
 * The iterator must be iterated with ecs_worker_next().
 *
 * A worker iterator acts as a passthrough for data exposed by the parent
 * iterator, so that any data provided by the parent will also be provided by
 * the worker iterator.
 *
 * @param it The source iterator.
 * @param index The index of the current resource.
 * @param count The total number of resources to divide entities between.
 * @return A worker iterator.
 */
FLECS_API
ecs_iter_t ecs_worker_iter(
    const ecs_iter_t *it,
    int32_t index,
    int32_t count);

/** Progress a worker iterator.
 * Progresses an iterator created by ecs_worker_iter().
 *
 * @param it The iterator.
 * @return true if iterator has more results, false if not.
 */
FLECS_API
bool ecs_worker_next(
    ecs_iter_t *it);

/** Get data for field.
 * This operation retrieves a pointer to an array of data that belongs to the
 * term in the query. The index refers to the location of the term in the query,
 * and starts counting from zero.
 *
 * For example, the query `"Position, Velocity"` will return the `Position` array
 * for index 0, and the `Velocity` array for index 1.
 *
 * When the specified field is not owned by the entity this function returns a
 * pointer instead of an array. This happens when the source of a field is not
 * the entity being iterated, such as a shared component (from a prefab), a
 * component from a parent, or another entity. The ecs_field_is_self() operation
 * can be used to test dynamically if a field is owned.
 * 
 * When a field contains a sparse component, use the ecs_field_at function. When
 * a field is guaranteed to be set and owned, the ecs_field_self() function can be
 * used. ecs_field_self() has slightly better performance, and provides stricter 
 * validity checking.
 *
 * The provided size must be either 0 or must match the size of the type
 * of the returned array. If the size does not match, the operation may assert.
 * The size can be dynamically obtained with ecs_field_size().
 * 
 * An example:
 * 
 * @code
 * while (ecs_query_next(&it)) {
 *   Position *p = ecs_field(&it, Position, 0);
 *   Velocity *v = ecs_field(&it, Velocity, 1);
 *   for (int32_t i = 0; i < it->count; i ++) {
 *     p[i].x += v[i].x;
 *     p[i].y += v[i].y;
 *   }
 * }
 * @endcode
 *
 * @param it The iterator.
 * @param size The size of the field type.
 * @param index The index of the field.
 * @return A pointer to the data of the field.
 */
FLECS_API
void* ecs_field_w_size(
    const ecs_iter_t *it,
    size_t size,
    int8_t index);

/** Get data for field at specified row.
 * This operation should be used instead of ecs_field_w_size for sparse 
 * component fields. This operation should be called for each returned row in a
 * result. In the following example the Velocity component is sparse:
 * 
 * @code
 * while (ecs_query_next(&it)) {
 *   Position *p = ecs_field(&it, Position, 0);
 *   for (int32_t i = 0; i < it->count; i ++) {
 *     Velocity *v = ecs_field_at(&it, Velocity, 1);
 *     p[i].x += v->x;
 *     p[i].y += v->y;
 *   }
 * }
 * @endcode
 * 
 * @param it the iterator.
 * @param size The size of the field type.
 * @param index The index of the field.
 * @return A pointer to the data of the field.
 */
FLECS_API
void* ecs_field_at_w_size(
    const ecs_iter_t *it,
    size_t size,
    int8_t index,
    int32_t row);

/** Test whether the field is readonly.
 * This operation returns whether the field is readonly. Readonly fields are
 * annotated with [in], or are added as a const type in the C++ API.
 *
 * @param it The iterator.
 * @param index The index of the field in the iterator.
 * @return Whether the field is readonly.
 */
FLECS_API
bool ecs_field_is_readonly(
    const ecs_iter_t *it,
    int8_t index);

/** Test whether the field is writeonly.
 * This operation returns whether this is a writeonly field. Writeonly terms are
 * annotated with [out].
 *
 * Serializers are not required to serialize the values of a writeonly field.
 *
 * @param it The iterator.
 * @param index The index of the field in the iterator.
 * @return Whether the field is writeonly.
 */
FLECS_API
bool ecs_field_is_writeonly(
    const ecs_iter_t *it,
    int8_t index);

/** Test whether field is set.
 *
 * @param it The iterator.
 * @param index The index of the field in the iterator.
 * @return Whether the field is set.
 */
FLECS_API
bool ecs_field_is_set(
    const ecs_iter_t *it,
    int8_t index);

/** Return id matched for field.
 *
 * @param it The iterator.
 * @param index The index of the field in the iterator.
 * @return The id matched for the field.
 */
FLECS_API
ecs_id_t ecs_field_id(
    const ecs_iter_t *it,
    int8_t index);

/** Return index of matched table column.
 * This function only returns column indices for fields that have been matched
 * on the $this variable. Fields matched on other tables will return -1.
 *
 * @param it The iterator.
 * @param index The index of the field in the iterator.
 * @return The index of the matched column, -1 if not matched.
 */
FLECS_API
int32_t ecs_field_column(
    const ecs_iter_t *it,
    int8_t index);

/** Return field source.
 * The field source is the entity on which the field was matched.
 *
 * @param it The iterator.
 * @param index The index of the field in the iterator.
 * @return The source for the field.
 */
FLECS_API
ecs_entity_t ecs_field_src(
    const ecs_iter_t *it,
    int8_t index);

/** Return field type size.
 * Return type size of the field. Returns 0 if the field has no data.
 *
 * @param it The iterator.
 * @param index The index of the field in the iterator.
 * @return The type size for the field.
 */
FLECS_API
size_t ecs_field_size(
    const ecs_iter_t *it,
    int8_t index);

/** Test whether the field is matched on self.
 * This operation returns whether the field is matched on the currently iterated
 * entity. This function will return false when the field is owned by another
 * entity, such as a parent or a prefab.
 *
 * When this operation returns false, the field must be accessed as a single
 * value instead of an array. Fields for which this operation returns true
 * return arrays with it->count values.
 *
 * @param it The iterator.
 * @param index The index of the field in the iterator.
 * @return Whether the field is matched on self.
 */
FLECS_API
bool ecs_field_is_self(
    const ecs_iter_t *it,
    int8_t index);

/** @} */

/**
 * @defgroup tables Tables
 * Functions for working with `ecs_table_t`.
 *
 * @{
 */

/** Get type for table.
 * The table type is a vector that contains all component, tag and pair ids.
 *
 * @param table The table.
 * @return The type of the table.
 */
FLECS_API
const ecs_type_t* ecs_table_get_type(
    const ecs_table_t *table);

/** Get type index for id.
 * This operation returns the index for an id in the table's type.
 *
 * @param world The world.
 * @param table The table.
 * @param id The id.
 * @return The index of the id in the table type, or -1 if not found.
 *
 * @see ecs_table_has_id()
 */
FLECS_API
int32_t ecs_table_get_type_index(
    const ecs_world_t *world,
    const ecs_table_t *table,
    ecs_id_t id);

/** Get column index for id.
 * This operation returns the column index for an id in the table's type. If the
 * id is not a component, the function will return -1.
 *
 * @param world The world.
 * @param table The table.
 * @param id The component id.
 * @return The column index of the id, or -1 if not found/not a component.
 */
FLECS_API
int32_t ecs_table_get_column_index(
    const ecs_world_t *world,
    const ecs_table_t *table,
    ecs_id_t id);

/** Return number of columns in table.
 * Similar to `ecs_table_get_type(table)->count`, except that the column count
 * only counts the number of components in a table.
 *
 * @param table The table.
 * @return The number of columns in the table.
 */
FLECS_API
int32_t ecs_table_column_count(
    const ecs_table_t *table);

/** Convert type index to column index.
 * Tables have an array of columns for each component in the table. This array
 * does not include elements for tags, which means that the index for a
 * component in the table type is not necessarily the same as the index in the
 * column array. This operation converts from an index in the table type to an
 * index in the column array.
 *
 * @param table The table.
 * @param index The index in the table type.
 * @return The index in the table column array.
 *
 * @see ecs_table_column_to_type_index()
 */
FLECS_API
int32_t ecs_table_type_to_column_index(
    const ecs_table_t *table,
    int32_t index);

/** Convert column index to type index.
 * Same as ecs_table_type_to_column_index(), but converts from an index in the
 * column array to an index in the table type.
 *
 * @param table The table.
 * @param index The column index.
 * @return The index in the table type.
 */
FLECS_API
int32_t ecs_table_column_to_type_index(
    const ecs_table_t *table,
    int32_t index);

/** Get column from table by column index.
 * This operation returns the component array for the provided index.
 *
 * @param table The table.
 * @param index The column index.
 * @param offset The index of the first row to return (0 for entire column).
 * @return The component array, or NULL if the index is not a component.
 */
FLECS_API
void* ecs_table_get_column(
    const ecs_table_t *table,
    int32_t index,
    int32_t offset);

/** Get column from table by component id.
 * This operation returns the component array for the provided component  id.
 *
 * @param world The world.
 * @param table The table.
 * @param id The component id for the column.
 * @param offset The index of the first row to return (0 for entire column).
 * @return The component array, or NULL if the index is not a component.
 */
FLECS_API
void* ecs_table_get_id(
    const ecs_world_t *world,
    const ecs_table_t *table,
    ecs_id_t id,
    int32_t offset);

/** Get column size from table.
 * This operation returns the component size for the provided index.
 *
 * @param table The table.
 * @param index The column index.
 * @return The component size, or 0 if the index is not a component.
 */
FLECS_API
size_t ecs_table_get_column_size(
    const ecs_table_t *table,
    int32_t index);

/** Returns the number of entities in the table.
 * This operation returns the number of entities in the table.
 *
 * @param table The table.
 * @return The number of entities in the table.
 */
FLECS_API
int32_t ecs_table_count(
    const ecs_table_t *table);

/** Returns allocated size of table.
 * This operation returns the number of elements allocated in the table 
 * per column.
 * 
 * @param table The table.
 * @return The number of allocated elements in the table.
 */
FLECS_API
int32_t ecs_table_size(
    const ecs_table_t *table);

/** Returns array with entity ids for table.
 * The size of the returned array is the result of ecs_table_count().
 * 
 * @param table The table.
 * @return Array with entity ids for table.
 */
FLECS_API
const ecs_entity_t* ecs_table_entities(
    const ecs_table_t *table);

/** Test if table has id.
 * Same as `ecs_table_get_type_index(world, table, id) != -1`.
 *
 * @param world The world.
 * @param table The table.
 * @param id The id.
 * @return True if the table has the id, false if the table doesn't.
 *
 * @see ecs_table_get_type_index()
 */
FLECS_API
bool ecs_table_has_id(
    const ecs_world_t *world,
    const ecs_table_t *table,
    ecs_id_t id);

/** Return depth for table in tree for relationship rel.
 * Depth is determined by counting the number of targets encountered while
 * traversing up the relationship tree for rel. Only acyclic relationships are
 * supported.
 *
 * @param world The world.
 * @param table The table.
 * @param rel The relationship.
 * @return The depth of the table in the tree.
 */
FLECS_API
int32_t ecs_table_get_depth(
    const ecs_world_t *world,
    const ecs_table_t *table,
    ecs_entity_t rel);

/** Get table that has all components of current table plus the specified id.
 * If the provided table already has the provided id, the operation will return
 * the provided table.
 *
 * @param world The world.
 * @param table The table.
 * @param id The id to add.
 * @result The resulting table.
 */
FLECS_API
ecs_table_t* ecs_table_add_id(
    ecs_world_t *world,
    ecs_table_t *table,
    ecs_id_t id);

/** Find table from id array.
 * This operation finds or creates a table with the specified array of
 * (component) ids. The ids in the array must be sorted, and it may not contain
 * duplicate elements.
 *
 * @param world The world.
 * @param ids The id array.
 * @param id_count The number of elements in the id array.
 * @return The table with the specified (component) ids.
 */
FLECS_API
ecs_table_t* ecs_table_find(
    ecs_world_t *world,
    const ecs_id_t *ids,
    int32_t id_count);

/** Get table that has all components of current table minus the specified id.
 * If the provided table doesn't have the provided id, the operation will return
 * the provided table.
 *
 * @param world The world.
 * @param table The table.
 * @param id The id to remove.
 * @result The resulting table.
 */
FLECS_API
ecs_table_t* ecs_table_remove_id(
    ecs_world_t *world,
    ecs_table_t *table,
    ecs_id_t id);

/** Lock a table.
 * When a table is locked, modifications to it will throw an assert. When the
 * table is locked recursively, it will take an equal amount of unlock
 * operations to actually unlock the table.
 *
 * Table locks can be used to build safe iterators where it is guaranteed that
 * the contents of a table are not modified while it is being iterated.
 *
 * The operation only works when called on the world, and has no side effects
 * when called on a stage. The assumption is that when called on a stage,
 * operations are deferred already.
 *
 * @param world The world.
 * @param table The table to lock.
 */
FLECS_API
void ecs_table_lock(
    ecs_world_t *world,
    ecs_table_t *table);

/** Unlock a table.
 * Must be called after calling ecs_table_lock().
 *
 * @param world The world.
 * @param table The table to unlock.
 */
FLECS_API
void ecs_table_unlock(
    ecs_world_t *world,
    ecs_table_t *table);

/** Test table for flags.
 * Test if table has all of the provided flags. See
 * include/flecs/private/api_flags.h for a list of table flags that can be used
 * with this function.
 *
 * @param table The table.
 * @param flags The flags to test for.
 * @return Whether the specified flags are set for the table.
 */
FLECS_API
bool ecs_table_has_flags(
    ecs_table_t *table,
    ecs_flags32_t flags);

/** Swaps two elements inside the table. This is useful for implementing custom
 * table sorting algorithms.
 * @param world The world
 * @param table The table to swap elements in
 * @param row_1 Table element to swap with row_2
 * @param row_2 Table element to swap with row_1
*/
FLECS_API
void ecs_table_swap_rows(
    ecs_world_t* world,
    ecs_table_t* table,
    int32_t row_1,
    int32_t row_2);

/** Commit (move) entity to a table.
 * This operation moves an entity from its current table to the specified
 * table. This may cause the following actions:
 * - Ctor for each component in the target table
 * - Move for each overlapping component
 * - Dtor for each component in the source table.
 * - `OnAdd` triggers for non-overlapping components in the target table
 * - `OnRemove` triggers for non-overlapping components in the source table.
 *
 * This operation is a faster than adding/removing components individually.
 *
 * The application must explicitly provide the difference in components between
 * tables as the added/removed parameters. This can usually be derived directly
 * from the result of ecs_table_add_id() and ecs_table_remove_id(). These arrays are
 * required to properly execute `OnAdd`/`OnRemove` triggers.
 *
 * @param world The world.
 * @param entity The entity to commit.
 * @param record The entity's record (optional, providing it saves a lookup).
 * @param table The table to commit the entity to.
 * @return True if the entity got moved, false otherwise.
 */
FLECS_API
bool ecs_commit(
    ecs_world_t *world,
    ecs_entity_t entity,
    ecs_record_t *record,
    ecs_table_t *table,
    const ecs_type_t *added,
    const ecs_type_t *removed);


/** Search for component id in table type.
 * This operation returns the index of first occurrence of the id in the table
 * type. The id may be a wildcard.
 *
 * When id_out is provided, the function will assign it with the found id. The
 * found id may be different from the provided id if it is a wildcard.
 *
 * This is a constant time operation.
 *
 * @param world The world.
 * @param table The table.
 * @param id The id to search for.
 * @param id_out If provided, it will be set to the found id (optional).
 * @return The index of the id in the table type.
 *
 * @see ecs_search_offset()
 * @see ecs_search_relation()
 */
FLECS_API
int32_t ecs_search(
    const ecs_world_t *world,
    const ecs_table_t *table,
    ecs_id_t id,
    ecs_id_t *id_out);

/** Search for component id in table type starting from an offset.
 * This operation is the same as ecs_search(), but starts searching from an offset
 * in the table type.
 *
 * This operation is typically called in a loop where the resulting index is
 * used in the next iteration as offset:
 *
 * @code
 * int32_t index = -1;
 * while ((index = ecs_search_offset(world, table, offset, id, NULL))) {
 *   // do stuff
 * }
 * @endcode
 *
 * Depending on how the operation is used it is either linear or constant time.
 * When the id has the form `(id)` or `(rel, *)` and the operation is invoked as
 * in the above example, it is guaranteed to be constant time.
 *
 * If the provided id has the form `(*, tgt)` the operation takes linear time. The
 * reason for this is that ids for an target are not packed together, as they
 * are sorted relationship first.
 *
 * If the id at the offset does not match the provided id, the operation will do
 * a linear search to find a matching id.
 *
 * @param world The world.
 * @param table The table.
 * @param offset Offset from where to start searching.
 * @param id The id to search for.
 * @param id_out If provided, it will be set to the found id (optional).
 * @return The index of the id in the table type.
 *
 * @see ecs_search()
 * @see ecs_search_relation()
 */
FLECS_API
int32_t ecs_search_offset(
    const ecs_world_t *world,
    const ecs_table_t *table,
    int32_t offset,
    ecs_id_t id,
    ecs_id_t *id_out);

/** Search for component/relationship id in table type starting from an offset.
 * This operation is the same as ecs_search_offset(), but has the additional
 * capability of traversing relationships to find a component. For example, if
 * an application wants to find a component for either the provided table or a
 * prefab (using the `IsA` relationship) of that table, it could use the operation
 * like this:
 *
 * @code
 * int32_t index = ecs_search_relation(
 *   world,            // the world
 *   table,            // the table
 *   0,                // offset 0
 *   ecs_id(Position), // the component id
 *   EcsIsA,           // the relationship to traverse
 *   0,                // start at depth 0 (the table itself)
 *   0,                // no depth limit
 *   NULL,             // (optional) entity on which component was found
 *   NULL,             // see above
 *   NULL);            // internal type with information about matched id
 * @endcode
 *
 * The operation searches depth first. If a table type has 2 `IsA` relationships, the
 * operation will first search the `IsA` tree of the first relationship.
 *
 * When choosing between ecs_search(), ecs_search_offset() and ecs_search_relation(),
 * the simpler the function the better its performance.
 *
 * @param world The world.
 * @param table The table.
 * @param offset Offset from where to start searching.
 * @param id The id to search for.
 * @param rel The relationship to traverse (optional).
 * @param flags Whether to search EcsSelf and/or EcsUp.
 * @param subject_out If provided, it will be set to the matched entity.
 * @param id_out If provided, it will be set to the found id (optional).
 * @param tr_out Internal datatype.
 * @return The index of the id in the table type.
 *
 * @see ecs_search()
 * @see ecs_search_offset()
 */
FLECS_API
int32_t ecs_search_relation(
    const ecs_world_t *world,
    const ecs_table_t *table,
    int32_t offset,
    ecs_id_t id,
    ecs_entity_t rel,
    ecs_flags64_t flags, /* EcsSelf and/or EcsUp */
    ecs_entity_t *subject_out,
    ecs_id_t *id_out,
    struct ecs_table_record_t **tr_out);

/** Remove all entities in a table. Does not deallocate table memory. 
 * Retaining table memory can be efficient when planning 
 * to refill the table with operations like ecs_bulk_init
 *
 * @param world The world.
 * @param table The table to clear.
 */
FLECS_API
void ecs_table_clear_entities(
    ecs_world_t* world,
    ecs_table_t* table);
    
/** @} */

/**
 * @defgroup values Values
 * Construct, destruct, copy and move dynamically created values.
 *
 * @{
 */

/** Construct a value in existing storage
 *
 * @param world The world.
 * @param type The type of the value to create.
 * @param ptr Pointer to a value of type 'type'
 * @return Zero if success, nonzero if failed.
 */
FLECS_API
int ecs_value_init(
    const ecs_world_t *world,
    ecs_entity_t type,
    void *ptr);

/** Construct a value in existing storage
 *
 * @param world The world.
 * @param ti The type info of the type to create.
 * @param ptr Pointer to a value of type 'type'
 * @return Zero if success, nonzero if failed.
 */
FLECS_API
int ecs_value_init_w_type_info(
    const ecs_world_t *world,
    const ecs_type_info_t *ti,
    void *ptr);

/** Construct a value in new storage
 *
 * @param world The world.
 * @param type The type of the value to create.
 * @return Pointer to type if success, NULL if failed.
 */
FLECS_API
void* ecs_value_new(
    ecs_world_t *world,
    ecs_entity_t type);

/** Construct a value in new storage
 *
 * @param world The world.
 * @param ti The type info of the type to create.
 * @return Pointer to type if success, NULL if failed.
 */
void* ecs_value_new_w_type_info(
    ecs_world_t *world,
    const ecs_type_info_t *ti);

/** Destruct a value
 *
 * @param world The world.
 * @param ti Type info of the value to destruct.
 * @param ptr Pointer to constructed value of type 'type'.
 * @return Zero if success, nonzero if failed.
 */
int ecs_value_fini_w_type_info(
    const ecs_world_t *world,
    const ecs_type_info_t *ti,
    void *ptr);

/** Destruct a value
 *
 * @param world The world.
 * @param type The type of the value to destruct.
 * @param ptr Pointer to constructed value of type 'type'.
 * @return Zero if success, nonzero if failed.
 */
FLECS_API
int ecs_value_fini(
    const ecs_world_t *world,
    ecs_entity_t type,
    void* ptr);

/** Destruct a value, free storage
 *
 * @param world The world.
 * @param type The type of the value to destruct.
 * @param ptr A pointer to the value.
 * @return Zero if success, nonzero if failed.
 */
FLECS_API
int ecs_value_free(
    ecs_world_t *world,
    ecs_entity_t type,
    void* ptr);

/** Copy value.
 *
 * @param world The world.
 * @param ti Type info of the value to copy.
 * @param dst Pointer to the storage to copy to.
 * @param src Pointer to the value to copy.
 * @return Zero if success, nonzero if failed.
 */
FLECS_API
int ecs_value_copy_w_type_info(
    const ecs_world_t *world,
    const ecs_type_info_t *ti,
    void* dst,
    const void *src);

/** Copy value.
 *
 * @param world The world.
 * @param type The type of the value to copy.
 * @param dst Pointer to the storage to copy to.
 * @param src Pointer to the value to copy.
 * @return Zero if success, nonzero if failed.
 */
FLECS_API
int ecs_value_copy(
    const ecs_world_t *world,
    ecs_entity_t type,
    void* dst,
    const void *src);

/** Move value.
 *
 * @param world The world.
 * @param ti Type info of the value to move.
 * @param dst Pointer to the storage to move to.
 * @param src Pointer to the value to move.
 * @return Zero if success, nonzero if failed.
 */
int ecs_value_move_w_type_info(
    const ecs_world_t *world,
    const ecs_type_info_t *ti,
    void* dst,
    void *src);

/** Move value.
 *
 * @param world The world.
 * @param type The type of the value to move.
 * @param dst Pointer to the storage to move to.
 * @param src Pointer to the value to move.
 * @return Zero if success, nonzero if failed.
 */
int ecs_value_move(
    const ecs_world_t *world,
    ecs_entity_t type,
    void* dst,
    void *src);

/** Move construct value.
 *
 * @param world The world.
 * @param ti Type info of the value to move.
 * @param dst Pointer to the storage to move to.
 * @param src Pointer to the value to move.
 * @return Zero if success, nonzero if failed.
 */
int ecs_value_move_ctor_w_type_info(
    const ecs_world_t *world,
    const ecs_type_info_t *ti,
    void* dst,
    void *src);

/** Move construct value.
 *
 * @param world The world.
 * @param type The type of the value to move.
 * @param dst Pointer to the storage to move to.
 * @param src Pointer to the value to move.
 * @return Zero if success, nonzero if failed.
 */
int ecs_value_move_ctor(
    const ecs_world_t *world,
    ecs_entity_t type,
    void* dst,
    void *src);

/** @} */

/** @} */

/**
 * @defgroup c_addons Addons
 * @ingroup c
 * C APIs for addons.
 *
 * @{
 * @}
 */

/**
 * @file addons/flecs_c.h
 * @brief Extends the core API with convenience macros for C applications.
 */

#ifndef FLECS_C_
#define FLECS_C_

/**
 * @defgroup flecs_c Macro API
 * @ingroup c
 * Convenience macro's for C API
 *
 * @{
 */

/**
 * @defgroup flecs_c_creation Creation macro's
 * Convenience macro's for creating entities, components and observers
 *
 * @{
 */

/* Use for declaring entity, tag, prefab / any other entity identifier */
#define ECS_DECLARE(id)\
    ecs_entity_t id, ecs_id(id)

/** Forward declare an entity. */
#define ECS_ENTITY_DECLARE ECS_DECLARE

/** Define a forward declared entity.
 *
 * Example:
 *
 * @code
 * ECS_ENTITY_DEFINE(world, MyEntity, Position, Velocity);
 * @endcode
 */
#define ECS_ENTITY_DEFINE(world, id_, ...) \
    { \
        ecs_entity_desc_t desc = {0}; \
        desc.id = id_; \
        desc.name = #id_; \
        desc.add_expr = #__VA_ARGS__; \
        id_ = ecs_entity_init(world, &desc); \
        ecs_id(id_) = id_; \
        ecs_assert(id_ != 0, ECS_INVALID_PARAMETER, "failed to create entity %s", #id_); \
    } \
    (void)id_; \
    (void)ecs_id(id_)

/** Declare & define an entity.
 *
 * Example:
 *
 * @code
 * ECS_ENTITY(world, MyEntity, Position, Velocity);
 * @endcode
 */
#define ECS_ENTITY(world, id, ...) \
    ecs_entity_t ecs_id(id); \
    ecs_entity_t id = 0; \
    ECS_ENTITY_DEFINE(world, id, __VA_ARGS__)

/** Forward declare a tag. */
#define ECS_TAG_DECLARE ECS_DECLARE

/** Define a forward declared tag.
 *
 * Example:
 *
 * @code
 * ECS_TAG_DEFINE(world, MyTag);
 * @endcode
 */
#define ECS_TAG_DEFINE(world, id) ECS_ENTITY_DEFINE(world, id, 0)

/** Declare & define a tag.
 *
 * Example:
 *
 * @code
 * ECS_TAG(world, MyTag);
 * @endcode
 */
#define ECS_TAG(world, id) ECS_ENTITY(world, id, 0)

/** Forward declare a prefab. */
#define ECS_PREFAB_DECLARE ECS_DECLARE

/** Define a forward declared prefab.
 *
 * Example:
 *
 * @code
 * ECS_PREFAB_DEFINE(world, MyPrefab, Position, Velocity);
 * @endcode
 */
#define ECS_PREFAB_DEFINE(world, id, ...) ECS_ENTITY_DEFINE(world, id, Prefab, __VA_ARGS__)

/** Declare & define a prefab.
 *
 * Example:
 *
 * @code
 * ECS_PREFAB(world, MyPrefab, Position, Velocity);
 * @endcode
 */
#define ECS_PREFAB(world, id, ...) ECS_ENTITY(world, id, Prefab, __VA_ARGS__)

/** Forward declare a component. */
#define ECS_COMPONENT_DECLARE(id)         ecs_entity_t ecs_id(id)

/** Define a forward declared component.
 *
 * Example:
 *
 * @code
 * ECS_COMPONENT_DEFINE(world, Position);
 * @endcode
 */
#define ECS_COMPONENT_DEFINE(world, id_) \
    {\
        ecs_component_desc_t desc = {0}; \
        ecs_entity_desc_t edesc = {0}; \
        edesc.id = ecs_id(id_); \
        edesc.use_low_id = true; \
        edesc.name = #id_; \
        edesc.symbol = #id_; \
        desc.entity = ecs_entity_init(world, &edesc); \
        desc.type.size = ECS_SIZEOF(id_); \
        desc.type.alignment = ECS_ALIGNOF(id_); \
        ecs_id(id_) = ecs_component_init(world, &desc);\
    }\
    ecs_assert(ecs_id(id_) != 0, ECS_INVALID_PARAMETER, "failed to create component %s", #id_)

/** Declare & define a component.
 *
 * Example:
 *
 * @code
 * ECS_COMPONENT(world, Position);
 * @endcode
 */
#define ECS_COMPONENT(world, id)\
    ecs_entity_t ecs_id(id) = 0;\
    ECS_COMPONENT_DEFINE(world, id);\
    (void)ecs_id(id)

/* Forward declare an observer. */
#define ECS_OBSERVER_DECLARE(id)         ecs_entity_t ecs_id(id)

/** Define a forward declared observer.
 *
 * Example:
 *
 * @code
 * ECS_OBSERVER_DEFINE(world, AddPosition, EcsOnAdd, Position);
 * @endcode
 */
#define ECS_OBSERVER_DEFINE(world, id_, kind, ...)\
    {\
        ecs_observer_desc_t desc = {0};\
        ecs_entity_desc_t edesc = {0}; \
        edesc.id = ecs_id(id_); \
        edesc.name = #id_; \
        desc.entity = ecs_entity_init(world, &edesc); \
        desc.callback = id_;\
        desc.query.expr = #__VA_ARGS__;\
        desc.events[0] = kind;\
        ecs_id(id_) = ecs_observer_init(world, &desc);\
        ecs_assert(ecs_id(id_) != 0, ECS_INVALID_PARAMETER, "failed to create observer %s", #id_);\
    }

/** Declare & define an observer.
 *
 * Example:
 *
 * @code
 * ECS_OBSERVER(world, AddPosition, EcsOnAdd, Position);
 * @endcode
 */
#define ECS_OBSERVER(world, id, kind, ...)\
    ecs_entity_t ecs_id(id) = 0; \
    ECS_OBSERVER_DEFINE(world, id, kind, __VA_ARGS__);\
    ecs_entity_t id = ecs_id(id);\
    (void)ecs_id(id);\
    (void)id

/* Forward declare a query. */
#define ECS_QUERY_DECLARE(name)         ecs_query_t* name

/** Define a forward declared observer.
 *
 * Example:
 *
 * @code
 * ECS_QUERY_DEFINE(world, AddPosition, Position);
 * @endcode
 */
#define ECS_QUERY_DEFINE(world, name_, ...)\
    {\
        ecs_query_desc_t desc = {0};\
        ecs_entity_desc_t edesc = {0}; \
        edesc.name = #name_; \
        desc.entity = ecs_entity_init(world, &edesc); \
        desc.expr = #__VA_ARGS__;\
        name_ = ecs_query_init(world, &desc);\
        ecs_assert(name_ != NULL, ECS_INVALID_PARAMETER, "failed to create query %s", #name_);\
    }

/** Declare & define an observer.
 *
 * Example:
 *
 * @code
 * ECS_OBSERVER(world, AddPosition, EcsOnAdd, Position);
 * @endcode
 */
#define ECS_QUERY(world, name, ...)\
    ecs_query_t* name = NULL; \
    ECS_QUERY_DEFINE(world, name, __VA_ARGS__);\
    (void)name

/** Shorthand for creating an entity with ecs_entity_init().
 *
 * Example:
 *
 * @code
 * ecs_entity(world, {
 *   .name = "MyEntity"
 * });
 * @endcode
 */
#define ecs_entity(world, ...)\
    ecs_entity_init(world, &(ecs_entity_desc_t) __VA_ARGS__ )

/** Shorthand for creating a component with ecs_component_init().
 *
 * Example:
 *
 * @code
 * ecs_component(world, {
 *   .type.size = 4,
 *   .type.alignment = 4
 * });
 * @endcode
 */
#define ecs_component(world, ...)\
    ecs_component_init(world, &(ecs_component_desc_t) __VA_ARGS__ )

/** Shorthand for creating a component from a type.
 *
 * Example:
 *
 * @code
 * ecs_component_t(world, Position);
 * @endcode
 */
#define ecs_component_t(world, T)\
    ecs_component_init(world, &(ecs_component_desc_t) { \
        .entity = ecs_entity(world, { \
            .name = #T, \
            .symbol = #T, \
            .use_low_id = true \
        }), \
        .type.size = ECS_SIZEOF(T), \
        .type.alignment = ECS_ALIGNOF(T) \
    })

/** Shorthand for creating a query with ecs_query_cache_init.
 *
 * Example:
 *   ecs_query(world, {
 *     .terms = {{ ecs_id(Position) }}
 *   });
 */
#define ecs_query(world, ...)\
    ecs_query_init(world, &(ecs_query_desc_t) __VA_ARGS__ )

/** Shorthand for creating an observer with ecs_observer_init().
 *
 * Example:
 *
 * @code
 * ecs_observer(world, {
 *   .terms = {{ ecs_id(Position) }},
 *   .events = { EcsOnAdd },
 *   .callback = AddPosition
 * });
 * @endcode
 */
#define ecs_observer(world, ...)\
    ecs_observer_init(world, &(ecs_observer_desc_t) __VA_ARGS__ )

/** @} */

/**
 * @defgroup flecs_c_type_safe Type Safe API
 * Macro's that wrap around core functions to provide a "type safe" API in C
 *
 * @{
 */

/**
 * @defgroup flecs_c_entities Entity API
 * @{
 */

/**
 * @defgroup flecs_c_creation_deletion Creation & Deletion
 * @{
 */

#define ecs_new_w(world, T) ecs_new_w_id(world, ecs_id(T))

#define ecs_new_w_pair(world, first, second)\
    ecs_new_w_id(world, ecs_pair(first, second))

#define ecs_bulk_new(world, component, count)\
    ecs_bulk_new_w_id(world, ecs_id(component), count)

/** @} */

/**
 * @defgroup flecs_c_adding_removing Adding & Removing
 * @{
 */

#define ecs_add(world, entity, T)\
    ecs_add_id(world, entity, ecs_id(T))

#define ecs_add_pair(world, subject, first, second)\
    ecs_add_id(world, subject, ecs_pair(first, second))


#define ecs_remove(world, entity, T)\
    ecs_remove_id(world, entity, ecs_id(T))

#define ecs_remove_pair(world, subject, first, second)\
    ecs_remove_id(world, subject, ecs_pair(first, second))


#define ecs_auto_override(world, entity, T)\
    ecs_auto_override_id(world, entity, ecs_id(T))

#define ecs_auto_override_pair(world, subject, first, second)\
    ecs_auto_override_id(world, subject, ecs_pair(first, second))

/** @} */

/**
 * @defgroup flecs_c_getting_setting Getting & Setting
 * @{
 */

/* insert */
#define ecs_insert(world, ...)\
    ecs_entity(world, { .set = ecs_values(__VA_ARGS__)})

/* set */

#define ecs_set_ptr(world, entity, component, ptr)\
    ecs_set_id(world, entity, ecs_id(component), sizeof(component), ptr)

#define ecs_set(world, entity, component, ...)\
    ecs_set_id(world, entity, ecs_id(component), sizeof(component), &(component)__VA_ARGS__)

#define ecs_set_pair(world, subject, First, second, ...)\
    ecs_set_id(world, subject,\
        ecs_pair(ecs_id(First), second),\
        sizeof(First), &(First)__VA_ARGS__)

#define ecs_set_pair_second(world, subject, first, Second, ...)\
    ecs_set_id(world, subject,\
        ecs_pair(first, ecs_id(Second)),\
        sizeof(Second), &(Second)__VA_ARGS__)

#define ecs_set_override(world, entity, T, ...)\
    ecs_add_id(world, entity, ECS_AUTO_OVERRIDE | ecs_id(T));\
    ecs_set(world, entity, T, __VA_ARGS__)

/* emplace */

#define ecs_emplace(world, entity, T, is_new)\
    (ECS_CAST(T*, ecs_emplace_id(world, entity, ecs_id(T), is_new)))

#define ecs_emplace_pair(world, entity, First, second, is_new)\
    (ECS_CAST(First*, ecs_emplace_id(world, entity, ecs_pair_t(First, second), is_new)))

/* get */

#define ecs_get(world, entity, T)\
    (ECS_CAST(const T*, ecs_get_id(world, entity, ecs_id(T))))

#define ecs_get_pair(world, subject, First, second)\
    (ECS_CAST(const First*, ecs_get_id(world, subject,\
        ecs_pair(ecs_id(First), second))))

#define ecs_get_pair_second(world, subject, first, Second)\
    (ECS_CAST(const Second*, ecs_get_id(world, subject,\
        ecs_pair(first, ecs_id(Second)))))

/* get_mut */

#define ecs_get_mut(world, entity, T)\
    (ECS_CAST(T*, ecs_get_mut_id(world, entity, ecs_id(T))))

#define ecs_get_mut_pair(world, subject, First, second)\
    (ECS_CAST(First*, ecs_get_mut_id(world, subject,\
        ecs_pair(ecs_id(First), second))))

#define ecs_get_mut_pair_second(world, subject, first, Second)\
    (ECS_CAST(Second*, ecs_get_mut_id(world, subject,\
        ecs_pair(first, ecs_id(Second)))))

#define ecs_get_mut(world, entity, T)\
    (ECS_CAST(T*, ecs_get_mut_id(world, entity, ecs_id(T))))

/* ensure */

#define ecs_ensure(world, entity, T)\
    (ECS_CAST(T*, ecs_ensure_id(world, entity, ecs_id(T))))

#define ecs_ensure_pair(world, subject, First, second)\
    (ECS_CAST(First*, ecs_ensure_id(world, subject,\
        ecs_pair(ecs_id(First), second))))

#define ecs_ensure_pair_second(world, subject, first, Second)\
    (ECS_CAST(Second*, ecs_ensure_id(world, subject,\
        ecs_pair(first, ecs_id(Second)))))

#define ecs_ensure(world, entity, T)\
    (ECS_CAST(T*, ecs_ensure_id(world, entity, ecs_id(T))))

#define ecs_ensure_pair(world, subject, First, second)\
    (ECS_CAST(First*, ecs_ensure_id(world, subject,\
        ecs_pair(ecs_id(First), second))))

#define ecs_ensure_pair_second(world, subject, first, Second)\
    (ECS_CAST(Second*, ecs_ensure_id(world, subject,\
        ecs_pair(first, ecs_id(Second)))))

/* modified */

#define ecs_modified(world, entity, component)\
    ecs_modified_id(world, entity, ecs_id(component))

#define ecs_modified_pair(world, subject, first, second)\
    ecs_modified_id(world, subject, ecs_pair(first, second))

/* record */

#define ecs_record_get(world, record, T)\
    (ECS_CAST(const T*, ecs_record_get_id(world, record, ecs_id(T))))

#define ecs_record_has(world, record, T)\
    (ecs_record_has_id(world, record, ecs_id(T)))

#define ecs_record_get_pair(world, record, First, second)\
    (ECS_CAST(const First*, ecs_record_get_id(world, record, \
        ecs_pair(ecs_id(First), second))))

#define ecs_record_get_pair_second(world, record, first, Second)\
    (ECS_CAST(const Second*, ecs_record_get_id(world, record,\
        ecs_pair(first, ecs_id(Second)))))

#define ecs_record_ensure(world, record, T)\
    (ECS_CAST(T*, ecs_record_ensure_id(world, record, ecs_id(T))))

#define ecs_record_ensure_pair(world, record, First, second)\
    (ECS_CAST(First*, ecs_record_ensure_id(world, record, \
        ecs_pair(ecs_id(First), second))))

#define ecs_record_ensure_pair_second(world, record, first, Second)\
    (ECS_CAST(Second*, ecs_record_ensure_id(world, record,\
        ecs_pair(first, ecs_id(Second)))))

#define ecs_ref_init(world, entity, T)\
    ecs_ref_init_id(world, entity, ecs_id(T))

#define ecs_ref_get(world, ref, T)\
    (ECS_CAST(T*, ecs_ref_get_id(world, ref, ecs_id(T))))

/** @} */

/**
 * @defgroup flecs_c_singletons Singletons
 * @{
 */

#define ecs_singleton_add(world, comp)\
    ecs_add(world, ecs_id(comp), comp)

#define ecs_singleton_remove(world, comp)\
    ecs_remove(world, ecs_id(comp), comp)

#define ecs_singleton_get(world, comp)\
    ecs_get(world, ecs_id(comp), comp)

#define ecs_singleton_set_ptr(world, comp, ptr)\
    ecs_set_ptr(world, ecs_id(comp), comp, ptr)

#define ecs_singleton_set(world, comp, ...)\
    ecs_set(world, ecs_id(comp), comp, __VA_ARGS__)

#define ecs_singleton_ensure(world, comp)\
    ecs_ensure(world, ecs_id(comp), comp)

#define ecs_singleton_modified(world, comp)\
    ecs_modified(world, ecs_id(comp), comp)

/** @} */

/**
 * @defgroup flecs_c_has Has, Owns, Shares
 * @{
 */

#define ecs_has(world, entity, T)\
    ecs_has_id(world, entity, ecs_id(T))

#define ecs_has_pair(world, entity, first, second)\
    ecs_has_id(world, entity, ecs_pair(first, second))

#define ecs_owns_pair(world, entity, first, second)\
    ecs_owns_id(world, entity, ecs_pair(first, second))

#define ecs_owns(world, entity, T)\
    ecs_owns_id(world, entity, ecs_id(T))

#define ecs_shares_id(world, entity, id)\
    (ecs_search_relation(world, ecs_get_table(world, entity), 0, ecs_id(id), \
        EcsIsA, 1, 0, 0, 0, 0) != -1)

#define ecs_shares_pair(world, entity, first, second)\
    (ecs_shares_id(world, entity, ecs_pair(first, second)))

#define ecs_shares(world, entity, T)\
    (ecs_shares_id(world, entity, ecs_id(T)))

#define ecs_get_target_for(world, entity, rel, T)\
    ecs_get_target_for_id(world, entity, rel, ecs_id(T))

/** @} */

/**
 * @defgroup flecs_c_enable_disable Enabling & Disabling
 * @{
 */

#define ecs_enable_component(world, entity, T, enable)\
    ecs_enable_id(world, entity, ecs_id(T), enable)

#define ecs_is_enabled(world, entity, T)\
    ecs_is_enabled_id(world, entity, ecs_id(T))

#define ecs_enable_pair(world, entity, First, second, enable)\
    ecs_enable_id(world, entity, ecs_pair(ecs_id(First), second), enable)

#define ecs_is_enabled_pair(world, entity, First, second)\
    ecs_is_enabled_id(world, entity, ecs_pair(ecs_id(First), second))

/** @} */

/**
 * @defgroup flecs_c_entity_names Entity Names
 * @{
 */

#define ecs_lookup_from(world, parent, path)\
    ecs_lookup_path_w_sep(world, parent, path, ".", NULL, true)

#define ecs_get_path_from(world, parent, child)\
    ecs_get_path_w_sep(world, parent, child, ".", NULL)

#define ecs_get_path(world, child)\
    ecs_get_path_w_sep(world, 0, child, ".", NULL)

#define ecs_get_path_buf(world, child, buf)\
    ecs_get_path_w_sep_buf(world, 0, child, ".", NULL, buf, false)

#define ecs_new_from_path(world, parent, path)\
    ecs_new_from_path_w_sep(world, parent, path, ".", NULL)

#define ecs_add_path(world, entity, parent, path)\
    ecs_add_path_w_sep(world, entity, parent, path, ".", NULL)

#define ecs_add_fullpath(world, entity, path)\
    ecs_add_path_w_sep(world, entity, 0, path, ".", NULL)

/** @} */

/** @} */

/**
 * @defgroup flecs_c_components Component API
 * @{
 */

#define ecs_set_hooks(world, T, ...)\
    ecs_set_hooks_id(world, ecs_id(T), &(ecs_type_hooks_t)__VA_ARGS__)

#define ecs_get_hooks(world, T)\
    ecs_get_hooks_id(world, ecs_id(T));

/** Declare a constructor.
 * Example:
 *
 * @code
 * ECS_CTOR(MyType, ptr, { ptr->value = NULL; });
 * @endcode
 */
#define ECS_CTOR(type, var, ...)\
    ECS_XTOR_IMPL(type, ctor, var, __VA_ARGS__)

/** Declare a destructor.
 * Example:
 *
 * @code
 * ECS_DTOR(MyType, ptr, { free(ptr->value); });
 * @endcode
 */
#define ECS_DTOR(type, var, ...)\
    ECS_XTOR_IMPL(type, dtor, var, __VA_ARGS__)

/** Declare a copy action.
 * Example:
 *
 * @code
 * ECS_COPY(MyType, dst, src, { dst->value = strdup(src->value); });
 * @endcode
 */
#define ECS_COPY(type, dst_var, src_var, ...)\
    ECS_COPY_IMPL(type, dst_var, src_var, __VA_ARGS__)

/** Declare a move action.
 * Example:
 *
 * @code
 * ECS_MOVE(MyType, dst, src, { dst->value = src->value; src->value = 0; });
 * @endcode
 */
#define ECS_MOVE(type, dst_var, src_var, ...)\
    ECS_MOVE_IMPL(type, dst_var, src_var, __VA_ARGS__)

/** Declare component hooks.
 * Example:
 *
 * @code
 * ECS_ON_SET(MyType, ptr, { printf("%d\n", ptr->value); });
 * @endcode
 */
#define ECS_ON_ADD(type, ptr, ...)\
    ECS_HOOK_IMPL(type, ecs_on_add(type), ptr, __VA_ARGS__)
#define ECS_ON_REMOVE(type, ptr, ...)\
    ECS_HOOK_IMPL(type, ecs_on_remove(type), ptr, __VA_ARGS__)
#define ECS_ON_SET(type, ptr, ...)\
    ECS_HOOK_IMPL(type, ecs_on_set(type), ptr, __VA_ARGS__)

/* Map from typename to function name of component lifecycle action */
#define ecs_ctor(type) type##_ctor
#define ecs_dtor(type) type##_dtor
#define ecs_copy(type) type##_copy
#define ecs_move(type) type##_move
#define ecs_on_set(type) type##_on_set
#define ecs_on_add(type) type##_on_add
#define ecs_on_remove(type) type##_on_remove

/** @} */

/**
 * @defgroup flecs_c_ids Id API
 * @{
 */

#define ecs_count(world, type)\
    ecs_count_id(world, ecs_id(type))

/** @} */

/**
 * @defgroup flecs_c_iterators Iterator API
 * @{
 */

#define ecs_field(it, T, index)\
    (ECS_CAST(T*, ecs_field_w_size(it, sizeof(T), index)))

#define ecs_field_self(it, T, index)\
    (ECS_CAST(T*, ecs_field_self_w_size(it, sizeof(T), index)))

#define ecs_field_at(it, T, index, row)\
    (ECS_CAST(T*, ecs_field_at_w_size(it, sizeof(T), index, row)))

/** @} */

/**
 * @defgroup flecs_c_tables Table API
 * @{
 */

#define ecs_table_get(world, table, T, offset)\
    (ECS_CAST(T*, ecs_table_get_id(world, table, ecs_id(T), offset)))

#define ecs_table_get_pair(world, table, First, second, offset)\
    (ECS_CAST(First*, ecs_table_get_id(world, table, ecs_pair(ecs_id(First), second), offset)))

#define ecs_table_get_pair_second(world, table, first, Second, offset)\
    (ECS_CAST(Second*, ecs_table_get_id(world, table, ecs_pair(first, ecs_id(Second)), offset)))

/** @} */

/**
 * @defgroup flecs_c_values Value API
 * @{
 */

/** Convenience macro for creating compound literal id array */
#define ecs_ids(...) (ecs_id_t[]){ __VA_ARGS__, 0 }

/** Convenience macro for creating compound literal values array */
#define ecs_values(...) (ecs_value_t[]){ __VA_ARGS__, {0, 0}}

/** Convenience macro for creating compound literal value */
#define ecs_value_ptr(T, ptr) ((ecs_value_t){ecs_id(T), ptr})

/** Convenience macro for creating compound literal pair value */
#define ecs_value_pair(R, t, ...) ((ecs_value_t){ecs_pair_t(R, t), &(R)__VA_ARGS__})

/** Convenience macro for creating compound literal pair value */
#define ecs_value_pair_2nd(r, T, ...) ((ecs_value_t){ecs_pair(r, ecs_id(T)), &(T)__VA_ARGS__})

/** Convenience macro for creating heap allocated value */
#define ecs_value_new_t(world, T) ecs_value_new(world, ecs_id(T))

/** Convenience macro for creating compound literal value literal */
#define ecs_value(T, ...) ((ecs_value_t){ecs_id(T), &(T)__VA_ARGS__})

/** @} */

/** @} */

/**
 * @defgroup flecs_c_table_sorting Table sorting
 * Convenience macro's for sorting tables.
 *
 * @{
 */
#define ecs_sort_table(id) ecs_id(id##_sort_table)

#define ecs_compare(id) ecs_id(id##_compare_fn)

/* Declare efficient table sorting operation that uses provided compare function.
 * For best results use LTO or make the function body visible in the same compilation unit.
 * Variadic arguments are prepended before generated functions, use it to declare static
 *   or exported functions.
 * Parameters of the comparison function:
 *   ecs_entity_t e1, const void* ptr1,
 *   ecs_entity_t e2, const void* ptr2
 * Parameters of the sort functions:
 *   ecs_world_t *world
 *   ecs_table_t *table
 *   ecs_entity_t *entities
 *   void *ptr
 *   int32_t elem_size
 *   int32_t lo
 *   int32_t hi
 *   ecs_order_by_action_t order_by - Pointer to the original comparison function. You are not supposed to use it.
 * Example:
 *
 * @code
 * int CompareMyType(ecs_entity_t e1, const void* ptr1, ecs_entity_t e2, const void* ptr2) { const MyType* p1 = ptr1; const MyType* p2 = ptr2; return p1->value - p2->value; }
 * ECS_SORT_TABLE_WITH_COMPARE(MyType, MyCustomCompare, CompareMyType)
 * @endcode
 */
#define ECS_SORT_TABLE_WITH_COMPARE(id, op_name, compare_fn, ...) \
    static int32_t ECS_CONCAT(op_name, _partition)( \
        ecs_world_t *world, \
        ecs_table_t *table, \
        ecs_entity_t *entities, \
        void *ptr, \
        int32_t elem_size, \
        int32_t lo, \
        int32_t hi, \
        ecs_order_by_action_t order_by) \
    { \
        (void)(order_by); \
        int32_t p = (hi + lo) / 2; \
        void *pivot = ECS_ELEM(ptr, elem_size, p); \
        ecs_entity_t pivot_e = entities[p]; \
        int32_t i = lo - 1, j = hi + 1; \
        void *el; \
    repeat: \
        { \
            do { \
                i ++; \
                el = ECS_ELEM(ptr, elem_size, i); \
            } while ( compare_fn(entities[i], el, pivot_e, pivot) < 0); \
            do { \
                j --; \
                el = ECS_ELEM(ptr, elem_size, j); \
            } while ( compare_fn(entities[j], el, pivot_e, pivot) > 0); \
            if (i >= j) { \
                return j; \
            } \
            ecs_table_swap_rows(world, table, i, j); \
            if (p == i) { \
                pivot = ECS_ELEM(ptr, elem_size, j); \
                pivot_e = entities[j]; \
            } else if (p == j) { \
                pivot = ECS_ELEM(ptr, elem_size, i); \
                pivot_e = entities[i]; \
            } \
            goto repeat; \
        } \
    } \
    __VA_ARGS__ void op_name( \
        ecs_world_t *world, \
        ecs_table_t *table, \
        ecs_entity_t *entities, \
        void *ptr, \
        int32_t size, \
        int32_t lo, \
        int32_t hi, \
        ecs_order_by_action_t order_by) \
    { \
        if ((hi - lo) < 1)  { \
            return; \
        } \
        int32_t p = ECS_CONCAT(op_name, _partition)(world, table, entities, ptr, size, lo, hi, order_by); \
        op_name(world, table, entities, ptr, size, lo, p, order_by); \
        op_name(world, table, entities, ptr, size, p + 1, hi, order_by); \
    }

/* Declare efficient table sorting operation that uses default component comparison operator.
 * For best results use LTO or make the comparison operator visible in the same compilation unit.
 * Variadic arguments are prepended before generated functions, use it to declare static
 *   or exported functions.
 * Example:
 *
 * @code
 * ECS_COMPARE(MyType, { const MyType* p1 = ptr1; const MyType* p2 = ptr2; return p1->value - p2->value; });
 * ECS_SORT_TABLE(MyType)
 * @endcode
 */
#define ECS_SORT_TABLE(id, ...) \
    ECS_SORT_TABLE_WITH_COMPARE(id, ecs_sort_table(id), ecs_compare(id), __VA_ARGS__)

/* Declare component comparison operations.
 * Parameters:
 *   ecs_entity_t e1, const void* ptr1,
 *   ecs_entity_t e2, const void* ptr2
 * Example:
 *
 * @code
 * ECS_COMPARE(MyType, { const MyType* p1 = ptr1; const MyType* p2 = ptr2; return p1->value - p2->value; });
 * @endcode
 */
#define ECS_COMPARE(id, ...) \
    int ecs_compare(id)(ecs_entity_t e1, const void* ptr1, ecs_entity_t e2, const void* ptr2) { \
        __VA_ARGS__ \
    }

/** @} */

/**
 * @defgroup flecs_c_misc Misc
 * Misc convenience macro's.
 *
 * @{
 */

#define ecs_isa(e)       ecs_pair(EcsIsA, e)
#define ecs_childof(e)   ecs_pair(EcsChildOf, e)
#define ecs_dependson(e) ecs_pair(EcsDependsOn, e)
#define ecs_with(e)      ecs_pair(EcsWith, e)

#define ecs_each(world, id) ecs_each_id(world, ecs_id(id))
#define ecs_each_pair(world, r, t) ecs_each_id(world, ecs_pair(r, t))
#define ecs_each_pair_t(world, R, t) ecs_each_id(world, ecs_pair(ecs_id(R), t))

/** @} */

/** @} */

#endif // FLECS_C_


#ifdef __cplusplus
}
#endif

/**
 * @file addons.h
 * @brief Include enabled addons.
 *
 * This file should only be included by the main flecs.h header.
 */

#ifndef FLECS_ADDONS_H
#define FLECS_ADDONS_H

/* Blacklist macros */
#ifdef FLECS_NO_CPP
#undef FLECS_CPP
#endif
#ifdef FLECS_NO_MODULE
#undef FLECS_MODULE
#endif
#ifdef FLECS_NO_SCRIPT
#undef FLECS_SCRIPT
#endif
#ifdef FLECS_NO_SCRIPT_MATH
#undef FLECS_SCRIPT_MATH
#endif
#ifdef FLECS_NO_STATS
#undef FLECS_STATS
#endif
#ifdef FLECS_NO_SYSTEM
#undef FLECS_SYSTEM
#endif
#ifdef FLECS_NO_ALERTS
#undef FLECS_ALERTS
#endif
#ifdef FLECS_NO_PIPELINE
#undef FLECS_PIPELINE
#endif
#ifdef FLECS_NO_TIMER
#undef FLECS_TIMER
#endif
#ifdef FLECS_NO_META
#undef FLECS_META
#endif
#ifdef FLECS_NO_UNITS
#undef FLECS_UNITS
#endif
#ifdef FLECS_NO_JSON
#undef FLECS_JSON
#endif
#ifdef FLECS_NO_DOC
#undef FLECS_DOC
#endif
#ifdef FLECS_NO_LOG
#undef FLECS_LOG
#endif
#ifdef FLECS_NO_APP
#undef FLECS_APP
#endif
#ifdef FLECS_NO_OS_API_IMPL
#undef FLECS_OS_API_IMPL
#endif
#ifdef FLECS_NO_HTTP
#undef FLECS_HTTP
#endif
#ifdef FLECS_NO_REST
#undef FLECS_REST
#endif
#ifdef FLECS_NO_JOURNAL
#undef FLECS_JOURNAL
#endif

/* Always included, if disabled functions are replaced with dummy macros */
/**
 * @file addons/journal.h
 * @brief Journaling addon that logs API functions.
 *
 * The journaling addon traces API calls. The trace is formatted as runnable
 * C code, which allows for (partially) reproducing the behavior of an app
 * with the journaling trace.
 *
 * The journaling addon is disabled by default. Enabling it can have a
 * significant impact on performance.
 */

#ifdef FLECS_JOURNAL

#ifndef FLECS_LOG
#define FLECS_LOG
#endif

#ifndef FLECS_JOURNAL_H
#define FLECS_JOURNAL_H

/**
 * @defgroup c_addons_journal Journal
 * @ingroup c_addons
 * Journaling addon (disabled by default).
 *
 *
 * @{
 */

/* Trace when log level is at or higher than level */
#define FLECS_JOURNAL_LOG_LEVEL (0)

#ifdef __cplusplus
extern "C" {
#endif

/* Journaling API, meant to be used by internals. */

typedef enum ecs_journal_kind_t {
    EcsJournalNew,
    EcsJournalMove,
    EcsJournalClear,
    EcsJournalDelete,
    EcsJournalDeleteWith,
    EcsJournalRemoveAll,
    EcsJournalTableEvents
} ecs_journal_kind_t;

FLECS_DBG_API
void flecs_journal_begin(
    ecs_world_t *world,
    ecs_journal_kind_t kind,
    ecs_entity_t entity,
    ecs_type_t *add,
    ecs_type_t *remove);

FLECS_DBG_API
void flecs_journal_end(void);

#define flecs_journal(...)\
    flecs_journal_begin(__VA_ARGS__);\
    flecs_journal_end();

#ifdef __cplusplus
}
#endif // __cplusplus
/** @} */
#endif // FLECS_JOURNAL_H
#else
#define flecs_journal_begin(...)
#define flecs_journal_end(...)
#define flecs_journal(...)

#endif // FLECS_JOURNAL

/**
 * @file addons/log.h
 * @brief Logging addon.
 *
 * The logging addon provides an API for (debug) tracing and reporting errors
 * at various levels. When enabled, the logging addon can provide more detailed
 * information about the state of the ECS and any errors that may occur.
 *
 * The logging addon can be disabled to reduce footprint of the library, but
 * limits information logged to only file, line and error code.
 *
 * When enabled the logging addon can be configured to exclude levels of tracing
 * from the build to reduce the impact on performance. By default all debug
 * tracing is enabled for debug builds, tracing is enabled at release builds.
 *
 * Applications can change the logging level at runtime with ecs_log_set_level(),
 * but what is actually logged depends on what is compiled (when compiled
 * without debug tracing, setting the runtime level to debug won't have an
 * effect).
 *
 * The logging addon uses the OS API log_ function for all tracing.
 *
 * Note that even when the logging addon is not enabled, its header/source must
 * be included in a build. To prevent unused variable warnings in the code, some
 * API functions are included when the addon is disabled, but have empty bodies.
 */

#ifndef FLECS_LOG_H
#define FLECS_LOG_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef FLECS_LOG

/**
 * @defgroup c_addons_log Log
 * @ingroup c_addons
 * Logging functions.
 *
 * @{
 */

////////////////////////////////////////////////////////////////////////////////
//// Tracing
////////////////////////////////////////////////////////////////////////////////

/** Log message indicating an operation is deprecated. */
FLECS_API
void ecs_deprecated_(
    const char *file,
    int32_t line,
    const char *msg);

/** Increase log stack.
 * This operation increases the indent_ value of the OS API and can be useful to
 * make nested behavior more visible.
 *
 * @param level The log level.
 */
FLECS_API
void ecs_log_push_(int32_t level);

/** Decrease log stack.
 * This operation decreases the indent_ value of the OS API and can be useful to
 * make nested behavior more visible.
 *
 * @param level The log level.
 */
FLECS_API
void ecs_log_pop_(int32_t level);

/** Should current level be logged.
 * This operation returns true when the specified log level should be logged
 * with the current log level.
 *
 * @param level The log level to check for.
 * @return Whether logging is enabled for the current level.
 */
FLECS_API
bool ecs_should_log(int32_t level);

////////////////////////////////////////////////////////////////////////////////
//// Error reporting
////////////////////////////////////////////////////////////////////////////////

/** Get description for error code */
FLECS_API
const char* ecs_strerror(
    int32_t error_code);

#else // FLECS_LOG

////////////////////////////////////////////////////////////////////////////////
//// Dummy macros for when logging is disabled
////////////////////////////////////////////////////////////////////////////////

#define ecs_deprecated_(file, line, msg)\
    (void)file;\
    (void)line;\
    (void)msg

#define ecs_log_push_(level)
#define ecs_log_pop_(level)
#define ecs_should_log(level) false

#define ecs_strerror(error_code)\
    (void)error_code

#endif // FLECS_LOG


////////////////////////////////////////////////////////////////////////////////
//// Logging functions (do nothing when logging is enabled)
////////////////////////////////////////////////////////////////////////////////

FLECS_API
void ecs_print_(
    int32_t level,
    const char *file,
    int32_t line,
    const char *fmt,
    ...);

FLECS_API
void ecs_printv_(
    int level,
    const char *file,
    int32_t line,
    const char *fmt,
    va_list args);

FLECS_API
void ecs_log_(
    int32_t level,
    const char *file,
    int32_t line,
    const char *fmt,
    ...);

FLECS_API
void ecs_logv_(
    int level,
    const char *file,
    int32_t line,
    const char *fmt,
    va_list args);

FLECS_API
void ecs_abort_(
    int32_t error_code,
    const char *file,
    int32_t line,
    const char *fmt,
    ...);

FLECS_API
void ecs_assert_log_(
    int32_t error_code,
    const char *condition_str,
    const char *file,
    int32_t line,
    const char *fmt,
    ...);

FLECS_API
void ecs_parser_error_(
    const char *name,
    const char *expr,
    int64_t column,
    const char *fmt,
    ...);

FLECS_API
void ecs_parser_errorv_(
    const char *name,
    const char *expr,
    int64_t column,
    const char *fmt,
    va_list args);

FLECS_API
void ecs_parser_warning_(
    const char *name,
    const char *expr,
    int64_t column,
    const char *fmt,
    ...);

FLECS_API
void ecs_parser_warningv_(
    const char *name,
    const char *expr,
    int64_t column,
    const char *fmt,
    va_list args);


////////////////////////////////////////////////////////////////////////////////
//// Logging macros
////////////////////////////////////////////////////////////////////////////////

#ifndef FLECS_LEGACY /* C89 doesn't support variadic macros */

/* Base logging function. Accepts a custom level */
#define ecs_print(level, ...)\
    ecs_print_(level, __FILE__, __LINE__, __VA_ARGS__)

#define ecs_printv(level, fmt, args)\
    ecs_printv_(level, __FILE__, __LINE__, fmt, args)

#define ecs_log(level, ...)\
    ecs_log_(level, __FILE__, __LINE__, __VA_ARGS__)

#define ecs_logv(level, fmt, args)\
    ecs_logv_(level, __FILE__, __LINE__, fmt, args)

/* Tracing. Used for logging of infrequent events  */
#define ecs_trace_(file, line, ...) ecs_log_(0, file, line, __VA_ARGS__)
#define ecs_trace(...) ecs_trace_(__FILE__, __LINE__, __VA_ARGS__)

/* Warning. Used when an issue occurs, but operation is successful */
#define ecs_warn_(file, line, ...) ecs_log_(-2, file, line, __VA_ARGS__)
#define ecs_warn(...) ecs_warn_(__FILE__, __LINE__, __VA_ARGS__)

/* Error. Used when an issue occurs, and operation failed. */
#define ecs_err_(file, line, ...) ecs_log_(-3, file, line, __VA_ARGS__)
#define ecs_err(...) ecs_err_(__FILE__, __LINE__, __VA_ARGS__)

/* Fatal. Used when an issue occurs, and the application cannot continue. */
#define ecs_fatal_(file, line, ...) ecs_log_(-4, file, line, __VA_ARGS__)
#define ecs_fatal(...) ecs_fatal_(__FILE__, __LINE__, __VA_ARGS__)

/* Optionally include warnings about using deprecated features */
#ifndef FLECS_NO_DEPRECATED_WARNINGS
#define ecs_deprecated(...)\
    ecs_deprecated_(__FILE__, __LINE__, __VA_ARGS__)
#else
#define ecs_deprecated(...)
#endif // FLECS_NO_DEPRECATED_WARNINGS

/* If no tracing verbosity is defined, pick default based on build config */
#if !(defined(FLECS_LOG_0) || defined(FLECS_LOG_1) || defined(FLECS_LOG_2) || defined(FLECS_LOG_3))
#if !defined(FLECS_NDEBUG)
#define FLECS_LOG_3 /* Enable all tracing in debug mode. May slow things down */
#else
#define FLECS_LOG_0 /* Only enable infrequent tracing in release mode */
#endif // !defined(FLECS_NDEBUG)
#endif // !(defined(FLECS_LOG_0) || defined(FLECS_LOG_1) || defined(FLECS_LOG_2) || defined(FLECS_LOG_3))


/* Define/undefine macros based on compiled-in tracing level. This can optimize
 * out tracing statements from a build, which improves performance. */

#if defined(FLECS_LOG_3) /* All debug tracing enabled */
#define ecs_dbg_1(...) ecs_log(1, __VA_ARGS__);
#define ecs_dbg_2(...) ecs_log(2, __VA_ARGS__);
#define ecs_dbg_3(...) ecs_log(3, __VA_ARGS__);

#define ecs_log_push_1() ecs_log_push_(1);
#define ecs_log_push_2() ecs_log_push_(2);
#define ecs_log_push_3() ecs_log_push_(3);

#define ecs_log_pop_1() ecs_log_pop_(1);
#define ecs_log_pop_2() ecs_log_pop_(2);
#define ecs_log_pop_3() ecs_log_pop_(3);

#define ecs_should_log_1() ecs_should_log(1)
#define ecs_should_log_2() ecs_should_log(2)
#define ecs_should_log_3() ecs_should_log(3)

#define FLECS_LOG_2
#define FLECS_LOG_1
#define FLECS_LOG_0

#elif defined(FLECS_LOG_2) /* Level 2 and below debug tracing enabled */
#define ecs_dbg_1(...) ecs_log(1, __VA_ARGS__);
#define ecs_dbg_2(...) ecs_log(2, __VA_ARGS__);
#define ecs_dbg_3(...)

#define ecs_log_push_1() ecs_log_push_(1);
#define ecs_log_push_2() ecs_log_push_(2);
#define ecs_log_push_3()

#define ecs_log_pop_1() ecs_log_pop_(1);
#define ecs_log_pop_2() ecs_log_pop_(2);
#define ecs_log_pop_3()

#define ecs_should_log_1() ecs_should_log(1)
#define ecs_should_log_2() ecs_should_log(2)
#define ecs_should_log_3() false

#define FLECS_LOG_1
#define FLECS_LOG_0

#elif defined(FLECS_LOG_1) /* Level 1 debug tracing enabled */
#define ecs_dbg_1(...) ecs_log(1, __VA_ARGS__);
#define ecs_dbg_2(...)
#define ecs_dbg_3(...)

#define ecs_log_push_1() ecs_log_push_(1);
#define ecs_log_push_2()
#define ecs_log_push_3()

#define ecs_log_pop_1() ecs_log_pop_(1);
#define ecs_log_pop_2()
#define ecs_log_pop_3()

#define ecs_should_log_1() ecs_should_log(1)
#define ecs_should_log_2() false
#define ecs_should_log_3() false

#define FLECS_LOG_0

#elif defined(FLECS_LOG_0) /* No debug tracing enabled */
#define ecs_dbg_1(...)
#define ecs_dbg_2(...)
#define ecs_dbg_3(...)

#define ecs_log_push_1()
#define ecs_log_push_2()
#define ecs_log_push_3()

#define ecs_log_pop_1()
#define ecs_log_pop_2()
#define ecs_log_pop_3()

#define ecs_should_log_1() false
#define ecs_should_log_2() false
#define ecs_should_log_3() false

#else /* No tracing enabled */
#undef ecs_trace
#define ecs_trace(...)
#define ecs_dbg_1(...)
#define ecs_dbg_2(...)
#define ecs_dbg_3(...)

#define ecs_log_push_1()
#define ecs_log_push_2()
#define ecs_log_push_3()

#define ecs_log_pop_1()
#define ecs_log_pop_2()
#define ecs_log_pop_3()

#endif // defined(FLECS_LOG_3)

/* Default debug tracing is at level 1 */
#define ecs_dbg ecs_dbg_1

/* Default level for push/pop is 0 */
#define ecs_log_push() ecs_log_push_(0)
#define ecs_log_pop() ecs_log_pop_(0)

/** Abort.
 * Unconditionally aborts process. */
#define ecs_abort(error_code, ...)\
    ecs_abort_(error_code, __FILE__, __LINE__, __VA_ARGS__);\
    ecs_os_abort(); abort(); /* satisfy compiler/static analyzers */

/** Assert.
 * Aborts if condition is false, disabled in debug mode. */
#if defined(FLECS_NDEBUG) && !defined(FLECS_KEEP_ASSERT)
#define ecs_assert(condition, error_code, ...)
#else
#define ecs_assert(condition, error_code, ...)\
    if (!(condition)) {\
        ecs_assert_log_(error_code, #condition, __FILE__, __LINE__, __VA_ARGS__);\
        ecs_os_abort();\
    }\
    assert(condition) /* satisfy compiler/static analyzers */
#endif // FLECS_NDEBUG

#define ecs_assert_var(var, error_code, ...)\
    ecs_assert(var, error_code, __VA_ARGS__);\
    (void)var

/** Debug assert.
 * Assert that is only valid in debug mode (ignores FLECS_KEEP_ASSERT) */
#ifndef FLECS_NDEBUG
#define ecs_dbg_assert(condition, error_code, ...) ecs_assert(condition, error_code, __VA_ARGS__)
#else
#define ecs_dbg_assert(condition, error_code, ...)
#endif

/** Sanitize assert.
 * Assert that is only valid in sanitized mode (ignores FLECS_KEEP_ASSERT) */
#ifdef FLECS_SANITIZE
#define ecs_san_assert(condition, error_code, ...) ecs_assert(condition, error_code, __VA_ARGS__)
#else
#define ecs_san_assert(condition, error_code, ...)
#endif


/* Silence dead code/unused label warnings when compiling without checks. */
#define ecs_dummy_check\
    if ((false)) {\
        goto error;\
    }

/** Check.
 * goto error if condition is false. */
#if defined(FLECS_NDEBUG) && !defined(FLECS_KEEP_ASSERT)
#define ecs_check(condition, error_code, ...) ecs_dummy_check
#else
#ifdef FLECS_SOFT_ASSERT
#define ecs_check(condition, error_code, ...)\
    if (!(condition)) {\
        ecs_assert_log_(error_code, #condition, __FILE__, __LINE__, __VA_ARGS__);\
        goto error;\
    }
#else // FLECS_SOFT_ASSERT
#define ecs_check(condition, error_code, ...)\
    ecs_assert(condition, error_code, __VA_ARGS__);\
    ecs_dummy_check
#endif
#endif // FLECS_NDEBUG

/** Panic.
 * goto error when FLECS_SOFT_ASSERT is defined, otherwise abort */
#if defined(FLECS_NDEBUG) && !defined(FLECS_KEEP_ASSERT)
#define ecs_throw(error_code, ...) ecs_dummy_check
#else
#ifdef FLECS_SOFT_ASSERT
#define ecs_throw(error_code, ...)\
    ecs_abort_(error_code, __FILE__, __LINE__, __VA_ARGS__);\
    goto error;
#else
#define ecs_throw(error_code, ...)\
    ecs_abort(error_code, __VA_ARGS__);\
    ecs_dummy_check
#endif
#endif // FLECS_NDEBUG

/** Parser error */
#define ecs_parser_error(name, expr, column, ...)\
    ecs_parser_error_(name, expr, column, __VA_ARGS__)

#define ecs_parser_errorv(name, expr, column, fmt, args)\
    ecs_parser_errorv_(name, expr, column, fmt, args)

#define ecs_parser_warning(name, expr, column, ...)\
    ecs_parser_warning_(name, expr, column, __VA_ARGS__)

#define ecs_parser_warningv(name, expr, column, fmt, args)\
    ecs_parser_warningv_(name, expr, column, fmt, args)

#endif // FLECS_LEGACY


////////////////////////////////////////////////////////////////////////////////
//// Functions that are always available
////////////////////////////////////////////////////////////////////////////////

/** Enable or disable log.
 * This will enable builtin log. For log to work, it will have to be
 * compiled in which requires defining one of the following macros:
 *
 * FLECS_LOG_0 - All log is disabled
 * FLECS_LOG_1 - Enable log level 1
 * FLECS_LOG_2 - Enable log level 2 and below
 * FLECS_LOG_3 - Enable log level 3 and below
 *
 * If no log level is defined and this is a debug build, FLECS_LOG_3 will
 * have been automatically defined.
 *
 * The provided level corresponds with the log level. If -1 is provided as
 * value, warnings are disabled. If -2 is provided, errors are disabled as well.
 *
 * @param level Desired tracing level.
 * @return Previous log level.
 */
FLECS_API
int ecs_log_set_level(
    int level);

/** Get current log level.
 *
 * @return Previous log level.
 */
FLECS_API
int ecs_log_get_level(void);

/** Enable/disable tracing with colors.
 * By default colors are enabled.
 *
 * @param enabled Whether to enable tracing with colors.
 * @return Previous color setting.
 */
FLECS_API
bool ecs_log_enable_colors(
    bool enabled);

/** Enable/disable logging timestamp.
 * By default timestamps are disabled. Note that enabling timestamps introduces
 * overhead as the logging code will need to obtain the current time.
 *
 * @param enabled Whether to enable tracing with timestamps.
 * @return Previous timestamp setting.
 */
FLECS_API
bool ecs_log_enable_timestamp(
    bool enabled);

/** Enable/disable logging time since last log.
 * By default deltatime is disabled. Note that enabling timestamps introduces
 * overhead as the logging code will need to obtain the current time.
 *
 * When enabled, this logs the amount of time in seconds passed since the last
 * log, when this amount is non-zero. The format is a '+' character followed by
 * the number of seconds:
 *
 *     +1 trace: log message
 *
 * @param enabled Whether to enable tracing with timestamps.
 * @return Previous timestamp setting.
 */
FLECS_API
bool ecs_log_enable_timedelta(
    bool enabled);

/** Get last logged error code.
 * Calling this operation resets the error code.
 *
 * @return Last error, 0 if none was logged since last call to last_error.
 */
FLECS_API
int ecs_log_last_error(void);


////////////////////////////////////////////////////////////////////////////////
//// Error codes
////////////////////////////////////////////////////////////////////////////////

#define ECS_INVALID_OPERATION (1)
#define ECS_INVALID_PARAMETER (2)
#define ECS_CONSTRAINT_VIOLATED (3)
#define ECS_OUT_OF_MEMORY (4)
#define ECS_OUT_OF_RANGE (5)
#define ECS_UNSUPPORTED (6)
#define ECS_INTERNAL_ERROR (7)
#define ECS_ALREADY_DEFINED (8)
#define ECS_MISSING_OS_API (9)
#define ECS_OPERATION_FAILED (10)
#define ECS_INVALID_CONVERSION (11)
#define ECS_ID_IN_USE (12)
#define ECS_CYCLE_DETECTED (13)
#define ECS_LEAK_DETECTED (14)
#define ECS_DOUBLE_FREE (15)

#define ECS_INCONSISTENT_NAME (20)
#define ECS_NAME_IN_USE (21)
#define ECS_NOT_A_COMPONENT (22)
#define ECS_INVALID_COMPONENT_SIZE (23)
#define ECS_INVALID_COMPONENT_ALIGNMENT (24)
#define ECS_COMPONENT_NOT_REGISTERED (25)
#define ECS_INCONSISTENT_COMPONENT_ID (26)
#define ECS_INCONSISTENT_COMPONENT_ACTION (27)
#define ECS_MODULE_UNDEFINED (28)
#define ECS_MISSING_SYMBOL (29)
#define ECS_ALREADY_IN_USE (30)

#define ECS_ACCESS_VIOLATION (40)
#define ECS_COLUMN_INDEX_OUT_OF_RANGE (41)
#define ECS_COLUMN_IS_NOT_SHARED (42)
#define ECS_COLUMN_IS_SHARED (43)
#define ECS_COLUMN_TYPE_MISMATCH (45)

#define ECS_INVALID_WHILE_READONLY (70)
#define ECS_LOCKED_STORAGE (71)
#define ECS_INVALID_FROM_WORKER (72)


////////////////////////////////////////////////////////////////////////////////
//// Used when logging with colors is enabled
////////////////////////////////////////////////////////////////////////////////

#define ECS_BLACK   "\033[1;30m"
#define ECS_RED     "\033[0;31m"
#define ECS_GREEN   "\033[0;32m"
#define ECS_YELLOW  "\033[0;33m"
#define ECS_BLUE    "\033[0;34m"
#define ECS_MAGENTA "\033[0;35m"
#define ECS_CYAN    "\033[0;36m"
#define ECS_WHITE   "\033[1;37m"
#define ECS_GREY    "\033[0;37m"
#define ECS_NORMAL  "\033[0;49m"
#define ECS_BOLD    "\033[1;49m"

#ifdef __cplusplus
}
#endif

/** @} */

#endif // FLECS_LOG_H


/* Handle addon dependencies that need declarations to be visible in header */
#ifdef FLECS_STATS
#ifndef FLECS_PIPELINE
#define FLECS_PIPELINE
#endif
#ifndef FLECS_TIMER
#define FLECS_TIMER
#endif
#endif

#ifdef FLECS_REST
#ifndef FLECS_HTTP
#define FLECS_HTTP
#endif
#endif

#ifdef FLECS_APP
#ifdef FLECS_NO_APP
#error "FLECS_NO_APP failed: APP is required by other addons"
#endif
/**
 * @file addons/app.h
 * @brief App addon.
 *
 * The app addon is a wrapper around the application's main loop. Its main
 * purpose is to provide a hook to modules that need to take control of the
 * main loop, as is for example the case with native applications that use
 * emscripten with webGL.
 */

#ifdef FLECS_APP

#ifndef FLECS_PIPELINE
#define FLECS_PIPELINE
#endif

#ifndef FLECS_APP_H
#define FLECS_APP_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup c_addons_app App
 * @ingroup c_addons
 * Optional addon for running the main application loop.
 *
 * @{
 */

/** Callback type for init action. */
typedef int(*ecs_app_init_action_t)(
    ecs_world_t *world);

/** Used with ecs_app_run(). */
typedef struct ecs_app_desc_t {
    ecs_ftime_t target_fps;   /**< Target FPS. */
    ecs_ftime_t delta_time;   /**< Frame time increment (0 for measured values) */
    int32_t threads;          /**< Number of threads. */
    int32_t frames;           /**< Number of frames to run (0 for infinite) */
    bool enable_rest;         /**< Enables ECS access over HTTP, necessary for explorer */
    bool enable_stats;      /**< Periodically collect statistics */
    uint16_t port;            /**< HTTP port used by REST API */

    ecs_app_init_action_t init; /**< If set, function is ran before starting the
                                 * main loop. */

    void *ctx;                /**< Reserved for custom run/frame actions */
} ecs_app_desc_t;

/** Callback type for run action. */
typedef int(*ecs_app_run_action_t)(
    ecs_world_t *world,
    ecs_app_desc_t *desc);

/** Callback type for frame action. */
typedef int(*ecs_app_frame_action_t)(
    ecs_world_t *world,
    const ecs_app_desc_t *desc);

/** Run application.
 * This will run the application with the parameters specified in desc. After
 * the application quits (ecs_quit() is called) the world will be cleaned up.
 *
 * If a custom run action is set, it will be invoked by this operation. The
 * default run action calls the frame action in a loop until it returns a
 * non-zero value.
 *
 * @param world The world.
 * @param desc Application parameters.
 */
FLECS_API
int ecs_app_run(
    ecs_world_t *world,
    ecs_app_desc_t *desc);

/** Default frame callback.
 * This operation will run a single frame. By default this operation will invoke
 * ecs_progress() directly, unless a custom frame action is set.
 *
 * @param world The world.
 * @param desc The desc struct passed to ecs_app_run().
 * @return value returned by ecs_progress()
 */
FLECS_API
int ecs_app_run_frame(
    ecs_world_t *world,
    const ecs_app_desc_t *desc);

/** Set custom run action.
 * See ecs_app_run().
 *
 * @param callback The run action.
 */
FLECS_API
int ecs_app_set_run_action(
    ecs_app_run_action_t callback);

/** Set custom frame action.
 * See ecs_app_run_frame().
 *
 * @param callback The frame action.
 */
FLECS_API
int ecs_app_set_frame_action(
    ecs_app_frame_action_t callback);

/** @} */

#ifdef __cplusplus
}
#endif

#endif

#endif // FLECS_APP

#endif

#ifdef FLECS_HTTP
#ifdef FLECS_NO_HTTP
#error "FLECS_NO_HTTP failed: HTTP is required by other addons"
#endif
/**
 * @file addons/http.h
 * @brief HTTP addon.
 *
 * Minimalistic HTTP server that can receive and reply to simple HTTP requests.
 * The main goal of this addon is to enable remotely connecting to a running
 * Flecs application (for example, with a web-based UI) and request/visualize
 * data from the ECS world.
 *
 * Each server instance creates a single thread used for receiving requests.
 * Receiving requests are enqueued and handled when the application calls
 * ecs_http_server_dequeue(). This increases latency of request handling vs.
 * responding directly in the receive thread, but is better suited for
 * retrieving data from ECS applications, as requests can be processed by an ECS
 * system without having to lock the world.
 *
 * This server is intended to be used in a development environment.
 */

#ifdef FLECS_HTTP

/**
 * @defgroup c_addons_http Http
 * @ingroup c_addons
 * Simple HTTP server used for serving up REST API.
 *
 * @{
 */

#if !defined(FLECS_OS_API_IMPL) && !defined(FLECS_NO_OS_API_IMPL)
#define FLECS_OS_API_IMPL
#endif

#ifndef FLECS_HTTP_H
#define FLECS_HTTP_H

/** Maximum number of headers in request. */
#define ECS_HTTP_HEADER_COUNT_MAX (32)

/** Maximum number of query parameters in request. */
#define ECS_HTTP_QUERY_PARAM_COUNT_MAX (32)

#ifdef __cplusplus
extern "C" {
#endif

/** HTTP server. */
typedef struct ecs_http_server_t ecs_http_server_t;

/** A connection manages communication with the remote host. */
typedef struct {
    uint64_t id;
    ecs_http_server_t *server;

    char host[128];
    char port[16];
} ecs_http_connection_t;

/** Helper type used for headers & URL query parameters. */
typedef struct {
    const char *key;
    const char *value;
} ecs_http_key_value_t;

/** Supported request methods. */
typedef enum {
    EcsHttpGet,
    EcsHttpPost,
    EcsHttpPut,
    EcsHttpDelete,
    EcsHttpOptions,
    EcsHttpMethodUnsupported
} ecs_http_method_t;

/** An HTTP request. */
typedef struct {
    uint64_t id;

    ecs_http_method_t method;
    char *path;
    char *body;
    ecs_http_key_value_t headers[ECS_HTTP_HEADER_COUNT_MAX];
    ecs_http_key_value_t params[ECS_HTTP_HEADER_COUNT_MAX];
    int32_t header_count;
    int32_t param_count;

    ecs_http_connection_t *conn;
} ecs_http_request_t;

/** An HTTP reply. */
typedef struct {
    int code;                   /**< default = 200 */
    ecs_strbuf_t body;          /**< default = "" */
    const char* status;         /**< default = OK */
    const char* content_type;   /**< default = application/json */
    ecs_strbuf_t headers;       /**< default = "" */
} ecs_http_reply_t;

#define ECS_HTTP_REPLY_INIT \
    (ecs_http_reply_t){200, ECS_STRBUF_INIT, "OK", "application/json", ECS_STRBUF_INIT}

/* Global HTTP statistics. */
extern int64_t ecs_http_request_received_count;       /**< Total number of HTTP requests received. */
extern int64_t ecs_http_request_invalid_count;        /**< Total number of invalid HTTP requests. */
extern int64_t ecs_http_request_handled_ok_count;     /**< Total number of successful HTTP requests. */
extern int64_t ecs_http_request_handled_error_count;  /**< Total number of HTTP requests with errors. */
extern int64_t ecs_http_request_not_handled_count;    /**< Total number of HTTP requests with an unknown endpoint. */
extern int64_t ecs_http_request_preflight_count;      /**< Total number of preflight HTTP requests received. */
extern int64_t ecs_http_send_ok_count;                /**< Total number of HTTP replies successfully sent. */
extern int64_t ecs_http_send_error_count;             /**< Total number of HTTP replies that failed to send. */
extern int64_t ecs_http_busy_count;                   /**< Total number of HTTP busy replies. */

/** Request callback.
 * Invoked for each valid request. The function should populate the reply and
 * return true. When the function returns false, the server will reply with a
 * 404 (Not found) code. */
typedef bool (*ecs_http_reply_action_t)(
    const ecs_http_request_t* request,
    ecs_http_reply_t *reply,
    void *ctx);

/** Used with ecs_http_server_init(). */
typedef struct {
    ecs_http_reply_action_t callback; /**< Function called for each request  */
    void *ctx;                        /**< Passed to callback (optional) */
    uint16_t port;                    /**< HTTP port */
    const char *ipaddr;               /**< Interface to listen on (optional) */
    int32_t send_queue_wait_ms;       /**< Send queue wait time when empty */
    double cache_timeout;             /**< Cache invalidation timeout (0 disables caching) */
    double cache_purge_timeout;       /**< Cache purge timeout (for purging cache entries) */
} ecs_http_server_desc_t;

/** Create server.
 * Use ecs_http_server_start() to start receiving requests.
 *
 * @param desc Server configuration parameters.
 * @return The new server, or NULL if creation failed.
 */
FLECS_API
ecs_http_server_t* ecs_http_server_init(
    const ecs_http_server_desc_t *desc);

/** Destroy server.
 * This operation will stop the server if it was still running.
 *
 * @param server The server to destroy.
 */
FLECS_API
void ecs_http_server_fini(
    ecs_http_server_t* server);

/** Start server.
 * After this operation the server will be able to accept requests.
 *
 * @param server The server to start.
 * @return Zero if successful, non-zero if failed.
 */
FLECS_API
int ecs_http_server_start(
    ecs_http_server_t* server);

/** Process server requests.
 * This operation invokes the reply callback for each received request. No new
 * requests will be enqueued while processing requests.
 *
 * @param server The server for which to process requests.
 */
FLECS_API
void ecs_http_server_dequeue(
    ecs_http_server_t* server,
    ecs_ftime_t delta_time);

/** Stop server.
 * After this operation no new requests can be received.
 *
 * @param server The server.
 */
FLECS_API
void ecs_http_server_stop(
    ecs_http_server_t* server);

/** Emulate a request.
 * The request string must be a valid HTTP request. A minimal example:
 *
 *     GET /entity/flecs/core/World?label=true HTTP/1.1
 *
 * @param srv The server.
 * @param req The request.
 * @param len The length of the request (optional).
 * @return The reply.
 */
FLECS_API
int ecs_http_server_http_request(
    ecs_http_server_t* srv,
    const char *req,
    ecs_size_t len,
    ecs_http_reply_t *reply_out);

/** Convenience wrapper around ecs_http_server_http_request(). */
FLECS_API
int ecs_http_server_request(
    ecs_http_server_t* srv,
    const char *method,
    const char *req,
    const char *body,
    ecs_http_reply_t *reply_out);

/** Get context provided in ecs_http_server_desc_t */
FLECS_API
void* ecs_http_server_ctx(
    ecs_http_server_t* srv);

/** Find header in request.
 *
 * @param req The request.
 * @param name name of the header to find
 * @return The header value, or NULL if not found.
*/
FLECS_API
const char* ecs_http_get_header(
    const ecs_http_request_t* req,
    const char* name);

/** Find query parameter in request.
 *
 * @param req The request.
 * @param name The parameter name.
 * @return The decoded parameter value, or NULL if not found.
 */
FLECS_API
const char* ecs_http_get_param(
    const ecs_http_request_t* req,
    const char* name);

#ifdef __cplusplus
}
#endif

/** @} */

#endif // FLECS_HTTP_H

#endif // FLECS_HTTP

#endif

#ifdef FLECS_REST
#ifdef FLECS_NO_REST
#error "FLECS_NO_REST failed: REST is required by other addons"
#endif
/**
 * @file addons/rest.h
 * @brief REST API addon.
 *
 * A small REST API that uses the HTTP server and JSON serializer to provide
 * access to application data for remote applications.
 *
 * A description of the API can be found in docs/FlecsRemoteApi.md
 */

#ifdef FLECS_REST

/**
 * @defgroup c_addons_rest Rest
 * @ingroup c_addons
 * REST API for querying and mutating entities.
 *
 * @{
 */

/* Used for the HTTP server */
#ifndef FLECS_HTTP
#define FLECS_HTTP
#endif

/* Used for building the JSON replies */
#ifndef FLECS_JSON
#define FLECS_JSON
#endif

/* For the REST system */
#ifndef FLECS_PIPELINE
#define FLECS_PIPELINE
#endif

#ifndef FLECS_REST_H
#define FLECS_REST_H

#ifdef __cplusplus
extern "C" {
#endif

#define ECS_REST_DEFAULT_PORT (27750)

/** Component that instantiates the REST API. */
FLECS_API extern const ecs_entity_t ecs_id(EcsRest);

/** Component that creates a REST API server when instantiated. */
typedef struct {
    uint16_t port;      /**< Port of server (optional, default = 27750) */
    char *ipaddr;       /**< Interface address (optional, default = 0.0.0.0) */
    void *impl;
} EcsRest;

/** Create HTTP server for REST API.
 * This allows for the creation of a REST server that can be managed by the
 * application without using Flecs systems.
 *
 * @param world The world.
 * @param desc The HTTP server descriptor.
 * @return The HTTP server, or NULL if failed.
 */
FLECS_API
ecs_http_server_t* ecs_rest_server_init(
    ecs_world_t *world,
    const ecs_http_server_desc_t *desc);

/** Cleanup REST HTTP server.
 * The server must have been created with ecs_rest_server_init().
 */
FLECS_API
void ecs_rest_server_fini(
    ecs_http_server_t *srv);

/** Rest module import function.
 * Usage:
 * @code
 * ECS_IMPORT(world, FlecsRest)
 * @endcode
 * 
 * @param world The world.
 */
FLECS_API
void FlecsRestImport(
    ecs_world_t *world);

#ifdef __cplusplus
}
#endif

#endif

/** @} */

#endif

#endif

#ifdef FLECS_TIMER
#ifdef FLECS_NO_TIMER
#error "FLECS_NO_TIMER failed: TIMER is required by other addons"
#endif
/**
 * @file addons/timer.h
 * @brief Timer module.
 *
 * Timers can be used to trigger actions at periodic or one-shot intervals. They
 * are typically used together with systems and pipelines.
 */

#ifdef FLECS_TIMER

/**
 * @defgroup c_addons_timer Timer
 * @ingroup c_addons
 * Run systems at a time interval.
 *
 * @{
 */

#ifndef FLECS_MODULE
#define FLECS_MODULE
#endif

#ifndef FLECS_PIPELINE
#define FLECS_PIPELINE
#endif

#ifndef FLECS_TIMER_H
#define FLECS_TIMER_H

#ifdef __cplusplus
extern "C" {
#endif

/** Component used for one shot/interval timer functionality */
typedef struct EcsTimer {
    ecs_ftime_t timeout;         /**< Timer timeout period */
    ecs_ftime_t time;            /**< Incrementing time value */
    ecs_ftime_t overshoot;       /**< Used to correct returned interval time */
    int32_t fired_count;         /**< Number of times ticked */
    bool active;                 /**< Is the timer active or not */
    bool single_shot;            /**< Is this a single shot timer */
} EcsTimer;

/** Apply a rate filter to a tick source */
typedef struct EcsRateFilter {
    ecs_entity_t src;            /**< Source of the rate filter */
    int32_t rate;                /**< Rate of the rate filter */
    int32_t tick_count;          /**< Number of times the rate filter ticked */
    ecs_ftime_t time_elapsed;    /**< Time elapsed since last tick */
} EcsRateFilter;


/** Set timer timeout.
 * This operation executes any systems associated with the timer after the
 * specified timeout value. If the entity contains an existing timer, the
 * timeout value will be reset. The timer can be started and stopped with
 * ecs_start_timer() and ecs_stop_timer().
 *
 * The timer is synchronous, and is incremented each frame by delta_time.
 *
 * The tick_source entity will be a tick source after this operation. Tick
 * sources can be read by getting the EcsTickSource component. If the tick
 * source ticked this frame, the 'tick' member will be true. When the tick
 * source is a system, the system will tick when the timer ticks.
 *
 * @param world The world.
 * @param tick_source The timer for which to set the timeout (0 to create one).
 * @param timeout The timeout value.
 * @return The timer entity.
 */
FLECS_API
ecs_entity_t ecs_set_timeout(
    ecs_world_t *world,
    ecs_entity_t tick_source,
    ecs_ftime_t timeout);

/** Get current timeout value for the specified timer.
 * This operation returns the value set by ecs_set_timeout(). If no timer is
 * active for this entity, the operation returns 0.
 *
 * After the timeout expires the EcsTimer component is removed from the entity.
 * This means that if ecs_get_timeout() is invoked after the timer is expired, the
 * operation will return 0.
 *
 * The timer is synchronous, and is incremented each frame by delta_time.
 *
 * The tick_source entity will be a tick source after this operation. Tick
 * sources can be read by getting the EcsTickSource component. If the tick
 * source ticked this frame, the 'tick' member will be true. When the tick
 * source is a system, the system will tick when the timer ticks.
 *
 * @param world The world.
 * @param tick_source The timer.
 * @return The current timeout value, or 0 if no timer is active.
 */
FLECS_API
ecs_ftime_t ecs_get_timeout(
    const ecs_world_t *world,
    ecs_entity_t tick_source);

/** Set timer interval.
 * This operation will continuously invoke systems associated with the timer
 * after the interval period expires. If the entity contains an existing timer,
 * the interval value will be reset.
 *
 * The timer is synchronous, and is incremented each frame by delta_time.
 *
 * The tick_source entity will be a tick source after this operation. Tick
 * sources can be read by getting the EcsTickSource component. If the tick
 * source ticked this frame, the 'tick' member will be true. When the tick
 * source is a system, the system will tick when the timer ticks.
 *
 * @param world The world.
 * @param tick_source The timer for which to set the interval (0 to create one).
 * @param interval The interval value.
 * @return The timer entity.
 */
FLECS_API
ecs_entity_t ecs_set_interval(
    ecs_world_t *world,
    ecs_entity_t tick_source,
    ecs_ftime_t interval);

/** Get current interval value for the specified timer.
 * This operation returns the value set by ecs_set_interval(). If the entity is
 * not a timer, the operation will return 0.
 *
 * @param world The world.
 * @param tick_source The timer for which to set the interval.
 * @return The current interval value, or 0 if no timer is active.
 */
FLECS_API
ecs_ftime_t ecs_get_interval(
    const ecs_world_t *world,
    ecs_entity_t tick_source);

/** Start timer.
 * This operation resets the timer and starts it with the specified timeout.
 *
 * @param world The world.
 * @param tick_source The timer to start.
 */
FLECS_API
void ecs_start_timer(
    ecs_world_t *world,
    ecs_entity_t tick_source);

/** Stop timer
 * This operation stops a timer from triggering.
 *
 * @param world The world.
 * @param tick_source The timer to stop.
 */
FLECS_API
void ecs_stop_timer(
    ecs_world_t *world,
    ecs_entity_t tick_source);

/** Reset time value of timer to 0.
 * This operation resets the timer value to 0.
 *
 * @param world The world.
 * @param tick_source The timer to reset.
 */
FLECS_API
void ecs_reset_timer(
    ecs_world_t *world,
    ecs_entity_t tick_source);

/** Enable randomizing initial time value of timers.
 * Initializes timers with a random time value, which can improve scheduling as
 * systems/timers for the same interval don't all happen on the same tick.
 *
 * @param world The world.
 */
FLECS_API
void ecs_randomize_timers(
    ecs_world_t *world);

/** Set rate filter.
 * This operation initializes a rate filter. Rate filters sample tick sources
 * and tick at a configurable multiple. A rate filter is a tick source itself,
 * which means that rate filters can be chained.
 *
 * Rate filters enable deterministic system execution which cannot be achieved
 * with interval timers alone. For example, if timer A has interval 2.0 and
 * timer B has interval 4.0, it is not guaranteed that B will tick at exactly
 * twice the multiple of A. This is partly due to the indeterministic nature of
 * timers, and partly due to floating point rounding errors.
 *
 * Rate filters can be combined with timers (or other rate filters) to ensure
 * that a system ticks at an exact multiple of a tick source (which can be
 * another system). If a rate filter is created with a rate of 1 it will tick
 * at the exact same time as its source.
 *
 * If no tick source is provided, the rate filter will use the frame tick as
 * source, which corresponds with the number of times ecs_progress() is called.
 *
 * The tick_source entity will be a tick source after this operation. Tick
 * sources can be read by getting the EcsTickSource component. If the tick
 * source ticked this frame, the 'tick' member will be true. When the tick
 * source is a system, the system will tick when the timer ticks.
 *
 * @param world The world.
 * @param tick_source The rate filter entity (0 to create one).
 * @param rate The rate to apply.
 * @param source The tick source (0 to use frames)
 * @return The filter entity.
 */
FLECS_API
ecs_entity_t ecs_set_rate(
    ecs_world_t *world,
    ecs_entity_t tick_source,
    int32_t rate,
    ecs_entity_t source);

/** Assign tick source to system.
 * Systems can be their own tick source, which can be any of the tick sources
 * (one shot timers, interval times and rate filters). However, in some cases it
 * is must be guaranteed that different systems tick on the exact same frame.
 *
 * This cannot be guaranteed by giving two systems the same interval/rate filter
 * as it is possible that one system is (for example) disabled, which would
 * cause the systems to go out of sync. To provide these guarantees, systems
 * must use the same tick source, which is what this operation enables.
 *
 * When two systems share the same tick source, it is guaranteed that they tick
 * in the same frame. The provided tick source can be any entity that is a tick
 * source, including another system. If the provided entity is not a tick source
 * the system will not be ran.
 *
 * To disassociate a tick source from a system, use 0 for the tick_source
 * parameter.
 *
 * @param world The world.
 * @param system The system to associate with the timer.
 * @param tick_source The tick source to associate with the system.
 */
FLECS_API
void ecs_set_tick_source(
    ecs_world_t *world,
    ecs_entity_t system,
    ecs_entity_t tick_source);


////////////////////////////////////////////////////////////////////////////////
//// Module
////////////////////////////////////////////////////////////////////////////////

/** Timer module import function.
 * Usage:
 * @code
 * ECS_IMPORT(world, FlecsTimer)
 * @endcode
 * 
 * @param world The world.
 */
FLECS_API
void FlecsTimerImport(
    ecs_world_t *world);

#ifdef __cplusplus
}
#endif

#endif

/** @} */

#endif

#endif

#ifdef FLECS_PIPELINE
#ifdef FLECS_NO_PIPELINE
#error "FLECS_NO_PIPELINE failed: PIPELINE is required by other addons"
#endif
/**
 * @file addons/pipeline.h
 * @brief Pipeline module.
 *
 * The pipeline module provides support for running systems automatically and
 * on multiple threads. A pipeline is a collection of tags that can be added to
 * systems. When ran, a pipeline will query for all systems that have the tags
 * that belong to a pipeline, and run them.
 *
 * The module defines a number of builtin tags (EcsPreUpdate, EcsOnUpdate,
 * EcsPostUpdate etc.) that are registered with the builtin pipeline. The
 * builtin pipeline is ran by default when calling ecs_progress(). An
 * application can set a custom pipeline with the ecs_set_pipeline() function.
 */

#ifdef FLECS_PIPELINE

/**
 * @defgroup c_addons_pipeline Pipeline
 * @ingroup c_addons
 * Pipelines order and schedule systems for execution.
 *
 * @{
 */

#ifndef FLECS_MODULE
#define FLECS_MODULE
#endif

#ifndef FLECS_SYSTEM
#define FLECS_SYSTEM
#endif

#if !defined(FLECS_OS_API_IMPL) && !defined(FLECS_NO_OS_API_IMPL)
#define FLECS_OS_API_IMPL
#endif

#ifndef FLECS_PIPELINE_H
#define FLECS_PIPELINE_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef FLECS_LEGACY

/** Convenience macro to create a predeclared pipeline. 
 * Usage:
 * @code
 * ECS_ENTITY_DECLARE(MyPipeline);
 * ECS_PIPELINE_DEFINE(world, MyPipeline, Update || Physics || Render)
 * @endcode
 */
#define ECS_PIPELINE_DEFINE(world, id_, ...) \
    { \
        ecs_pipeline_desc_t desc = {0}; \
        ecs_entity_desc_t edesc = {0}; \
        edesc.id = id_;\
        edesc.name = #id_;\
        desc.entity = ecs_entity_init(world, &edesc);\
        desc.query.expr = #__VA_ARGS__; \
        id_ = ecs_pipeline_init(world, &desc); \
        ecs_id(id_) = id_;\
    } \
    ecs_assert(id_ != 0, ECS_INVALID_PARAMETER, "failed to create pipeline");

/** Convenience macro to create a pipeline. 
 * Usage:
 * @code
 * ECS_PIPELINE(world, MyPipeline, Update || Physics || Render)
 * @endcode
 * 
 */
#define ECS_PIPELINE(world, id, ...) \
    ecs_entity_t id = 0, ecs_id(id) = 0; ECS_PIPELINE_DEFINE(world, id, __VA_ARGS__);\
    (void)id;\
    (void)ecs_id(id);

/** Convenience macro to create a pipeline. 
 * See ecs_pipeline_init().
 */
#define ecs_pipeline(world, ...)\
    ecs_pipeline_init(world, &(ecs_pipeline_desc_t) __VA_ARGS__ )

#endif

/** Pipeline descriptor, used with ecs_pipeline_init(). */
typedef struct ecs_pipeline_desc_t {
    /** Existing entity to associate with pipeline (optional). */
    ecs_entity_t entity;

    /** The pipeline query. 
     * Pipelines are queries that are matched with system entities. Pipeline
     * queries are the same as regular queries, which means the same query rules
     * apply. A common mistake is to try a pipeline that matches systems in a
     * list of phases by specifying all the phases, like:
     *   OnUpdate, OnPhysics, OnRender
     * 
     * That however creates a query that matches entities with OnUpdate _and_
     * OnPhysics _and_ OnRender tags, which is likely undesired. Instead, a
     * query could use the or operator match a system that has one of the
     * specified phases:
     *   OnUpdate || OnPhysics || OnRender
     * 
     * This will return the correct set of systems, but they likely won't be in
     * the correct order. To make sure systems are returned in the correct order
     * two query ordering features can be used:
     * - group_by
     * - order_by
     * 
     * Take a look at the system manual for a more detailed explanation of
     * how query features can be applied to pipelines, and how the builtin
     * pipeline query works.
    */
    ecs_query_desc_t query;
} ecs_pipeline_desc_t;

/** Create a custom pipeline.
 * 
 * @param world The world.
 * @param desc The pipeline descriptor.
 * @return The pipeline, 0 if failed.
 */
FLECS_API
ecs_entity_t ecs_pipeline_init(
    ecs_world_t *world,
    const ecs_pipeline_desc_t *desc);

/** Set a custom pipeline.
 * This operation sets the pipeline to run when ecs_progress() is invoked.
 *
 * @param world The world.
 * @param pipeline The pipeline to set.
 */
FLECS_API
void ecs_set_pipeline(
    ecs_world_t *world,
    ecs_entity_t pipeline);

/** Get the current pipeline.
 * This operation gets the current pipeline.
 *
 * @param world The world.
 * @return The current pipeline.
 */
FLECS_API
ecs_entity_t ecs_get_pipeline(
    const ecs_world_t *world);

/** Progress a world.
 * This operation progresses the world by running all systems that are both
 * enabled and periodic on their matching entities.
 *
 * An application can pass a delta_time into the function, which is the time
 * passed since the last frame. This value is passed to systems so they can
 * update entity values proportional to the elapsed time since their last
 * invocation.
 *
 * When an application passes 0 to delta_time, ecs_progress() will automatically
 * measure the time passed since the last frame. If an application does not uses
 * time management, it should pass a non-zero value for delta_time (1.0 is
 * recommended). That way, no time will be wasted measuring the time.
 *
 * @param world The world to progress.
 * @param delta_time The time passed since the last frame.
 * @return false if ecs_quit() has been called, true otherwise.
 */
FLECS_API
bool ecs_progress(
    ecs_world_t *world,
    ecs_ftime_t delta_time);

/** Set time scale.
 * Increase or decrease simulation speed by the provided multiplier.
 *
 * @param world The world.
 * @param scale The scale to apply (default = 1).
 */
FLECS_API
void ecs_set_time_scale(
    ecs_world_t *world,
    ecs_ftime_t scale);

/** Reset world clock.
 * Reset the clock that keeps track of the total time passed in the simulation.
 *
 * @param world The world.
 */
FLECS_API
void ecs_reset_clock(
    ecs_world_t *world);

/** Run pipeline.
 * This will run all systems in the provided pipeline. This operation may be
 * invoked from multiple threads, and only when staging is disabled, as the
 * pipeline manages staging and, if necessary, synchronization between threads.
 *
 * If 0 is provided for the pipeline id, the default pipeline will be ran (this
 * is either the builtin pipeline or the pipeline set with set_pipeline()).
 *
 * When using progress() this operation will be invoked automatically for the
 * default pipeline (either the builtin pipeline or the pipeline set with
 * set_pipeline()). An application may run additional pipelines.
 *
 * @param world The world.
 * @param pipeline The pipeline to run.
 * @param delta_time The delta_time to pass to systems.
 */
FLECS_API
void ecs_run_pipeline(
    ecs_world_t *world,
    ecs_entity_t pipeline,
    ecs_ftime_t delta_time);


////////////////////////////////////////////////////////////////////////////////
//// Threading
////////////////////////////////////////////////////////////////////////////////

/** Set number of worker threads.
 * Setting this value to a value higher than 1 will start as many threads and
 * will cause systems to evenly distribute matched entities across threads. The
 * operation may be called multiple times to reconfigure the number of threads
 * used, but never while running a system / pipeline.
 * Calling ecs_set_threads() will also end the use of task threads setup with
 * ecs_set_task_threads() and vice-versa.
 * 
 * @param world The world.
 * @param threads The number of threads to create. 
 */
FLECS_API
void ecs_set_threads(
    ecs_world_t *world,
    int32_t threads);

/** Set number of worker task threads.
 * ecs_set_task_threads() is similar to ecs_set_threads(), except threads are treated
 * as short-lived tasks and will be created and joined around each update of the world.
 * Creation and joining of these tasks will use the os_api_t tasks APIs rather than the
 * the standard thread API functions, although they may be the same if desired.
 * This function is useful for multithreading world updates using an external
 * asynchronous job system rather than long running threads by providing the APIs
 * to create tasks for your job system and then wait on their conclusion.
 * The operation may be called multiple times to reconfigure the number of task threads
 * used, but never while running a system / pipeline.
 * Calling ecs_set_task_threads() will also end the use of threads setup with
 * ecs_set_threads() and vice-versa 
 * 
 * @param world The world.
 * @param task_threads The number of task threads to create. 
 */
FLECS_API
void ecs_set_task_threads(
    ecs_world_t *world,
    int32_t task_threads);

/** Returns true if task thread use have been requested. 
 * 
 * @param world The world.
 * @result Whether the world is using task threads.
 */
FLECS_API
bool ecs_using_task_threads(
    ecs_world_t *world);

////////////////////////////////////////////////////////////////////////////////
//// Module
////////////////////////////////////////////////////////////////////////////////

/** Pipeline module import function.
 * Usage:
 * @code
 * ECS_IMPORT(world, FlecsPipeline)
 * @endcode
 * 
 * @param world The world.
 */
FLECS_API
void FlecsPipelineImport(
    ecs_world_t *world);

#ifdef __cplusplus
}
#endif

#endif

/** @} */

#endif

#endif

#ifdef FLECS_SYSTEM
#ifdef FLECS_NO_SYSTEM
#error "FLECS_NO_SYSTEM failed: SYSTEM is required by other addons"
#endif
/**
 * @file addons/system.h
 * @brief System module.
 *
 * The system module allows for creating and running systems. A system is a
 * query in combination with a callback function. In addition systems have
 * support for time management and can be monitored by the stats addon.
 */

#ifdef FLECS_SYSTEM

/**
 * @defgroup c_addons_system System
 * @ingroup c_addons
 * Systems are a query + function that can be ran manually or by a pipeline.
 *
 * @{
 */

#ifndef FLECS_MODULE
#define FLECS_MODULE
#endif

#ifndef FLECS_SYSTEM_H
#define FLECS_SYSTEM_H

#ifdef __cplusplus
extern "C" {
#endif

/** Component used to provide a tick source to systems */
typedef struct EcsTickSource {
    bool tick;                 /**< True if providing tick */
    ecs_ftime_t time_elapsed;  /**< Time elapsed since last tick */
} EcsTickSource;

/** Use with ecs_system_init() to create or update a system. */
typedef struct ecs_system_desc_t {
    int32_t _canary;

    /** Existing entity to associate with system (optional) */
    ecs_entity_t entity;

    /** System query parameters */
    ecs_query_desc_t query;

    /** Callback that is ran for each result returned by the system's query. This
     * means that this callback can be invoked multiple times per system per
     * frame, typically once for each matching table. */
    ecs_iter_action_t callback;

    /** Callback that is invoked when a system is ran.
     * When left to NULL, the default system runner is used, which calls the
     * "callback" action for each result returned from the system's query.
     *
     * It should not be assumed that the input iterator can always be iterated
     * with ecs_query_next(). When a system is multithreaded and/or paged, the
     * iterator can be either a worker or paged iterator. The correct function 
     * to use for iteration is ecs_iter_next().
     *
     * An implementation can test whether the iterator is a query iterator by
     * testing whether the it->next value is equal to ecs_query_next(). */
    ecs_run_action_t run;

    /** Context to be passed to callback (as ecs_iter_t::param) */
    void *ctx;

    /** Callback to free ctx. */
    ecs_ctx_free_t ctx_free;

    /** Context associated with callback (for language bindings). */
    void *callback_ctx;

    /** Callback to free callback ctx. */
    ecs_ctx_free_t callback_ctx_free;

    /** Context associated with run (for language bindings). */
    void *run_ctx;

    /** Callback to free run ctx. */
    ecs_ctx_free_t run_ctx_free;
    
    /** Interval in seconds at which the system should run */
    ecs_ftime_t interval;

    /** Rate at which the system should run */
    int32_t rate;

    /** External tick source that determines when system ticks */
    ecs_entity_t tick_source;

    /** If true, system will be ran on multiple threads */
    bool multi_threaded;

    /** If true, system will have access to the actual world. Cannot be true at the
     * same time as multi_threaded. */
    bool immediate;
} ecs_system_desc_t;

/** Create a system */
FLECS_API
ecs_entity_t ecs_system_init(
    ecs_world_t *world,
    const ecs_system_desc_t *desc);

/** System type, get with ecs_system_get() */
typedef struct ecs_system_t {
    ecs_header_t hdr;

    /** See ecs_system_desc_t */
    ecs_run_action_t run;

    /** See ecs_system_desc_t */
    ecs_iter_action_t action;

    /** System query */
    ecs_query_t *query;

    /** Entity associated with query */
    ecs_entity_t query_entity;

    /** Tick source associated with system */
    ecs_entity_t tick_source;

    /** Is system multithreaded */
    bool multi_threaded;

    /** Is system ran in immediate mode */
    bool immediate;

    /** Cached system name (for perf tracing) */
    const char *name;

    /** Userdata for system */
    void *ctx;

    /** Callback language binding context */
    void *callback_ctx;

    /** Run language binding context */
    void *run_ctx;

    /** Callback to free ctx. */
    ecs_ctx_free_t ctx_free;

    /** Callback to free callback ctx. */
    ecs_ctx_free_t callback_ctx_free;

    /** Callback to free run ctx. */
    ecs_ctx_free_t run_ctx_free;

    /** Time spent on running system */
    ecs_ftime_t time_spent;

    /** Time passed since last invocation */
    ecs_ftime_t time_passed;

    /** Last frame for which the system was considered */
    int64_t last_frame;

    /* Mixins */
    ecs_world_t *world;
    ecs_entity_t entity;
    flecs_poly_dtor_t dtor;      
} ecs_system_t;

/** Get system object.
 * Returns the system object. Can be used to access various information about
 * the system, like the query and context.
 *
 * @param world The world.
 * @param system The system.
 * @return The system object.
 */
FLECS_API
const ecs_system_t* ecs_system_get(
    const ecs_world_t *world,
    ecs_entity_t system);

#ifndef FLECS_LEGACY

/** Forward declare a system. */
#define ECS_SYSTEM_DECLARE(id) ecs_entity_t ecs_id(id)

/** Define a forward declared system.
 *
 * Example:
 *
 * @code
 * ECS_SYSTEM_DEFINE(world, Move, EcsOnUpdate, Position, Velocity);
 * @endcode
 */
#define ECS_SYSTEM_DEFINE(world, id_, phase, ...) \
    { \
        ecs_system_desc_t desc = {0}; \
        ecs_entity_desc_t edesc = {0}; \
        ecs_id_t add_ids[3] = {\
            ((phase) ? ecs_pair(EcsDependsOn, (phase)) : 0), \
            (phase), \
            0 \
        };\
        edesc.id = ecs_id(id_);\
        edesc.name = #id_;\
        edesc.add = add_ids;\
        desc.entity = ecs_entity_init(world, &edesc);\
        desc.query.expr = #__VA_ARGS__; \
        desc.callback = id_; \
        ecs_id(id_) = ecs_system_init(world, &desc); \
    } \
    ecs_assert(ecs_id(id_) != 0, ECS_INVALID_PARAMETER, "failed to create system %s", #id_)

/** Declare & define a system.
 *
 * Example:
 *
 * @code
 * ECS_SYSTEM(world, Move, EcsOnUpdate, Position, Velocity);
 * @endcode
 */
#define ECS_SYSTEM(world, id, phase, ...) \
    ecs_entity_t ecs_id(id) = 0; ECS_SYSTEM_DEFINE(world, id, phase, __VA_ARGS__);\
    ecs_entity_t id = ecs_id(id);\
    (void)ecs_id(id);\
    (void)id

/** Shorthand for creating a system with ecs_system_init().
 *
 * Example:
 *
 * @code
 * ecs_system(world, {
 *   .entity = ecs_entity(world, {
 *     .name = "MyEntity",
 *     .add = ecs_ids( ecs_dependson(EcsOnUpdate) )
 *   }),
 *   .query.terms = {
 *     { ecs_id(Position) },
 *     { ecs_id(Velocity) }
 *   },
 *   .callback = Move
 * });
 * @endcode
 */
#define ecs_system(world, ...)\
    ecs_system_init(world, &(ecs_system_desc_t) __VA_ARGS__ )

#endif

/** Run a specific system manually.
 * This operation runs a single system manually. It is an efficient way to
 * invoke logic on a set of entities, as manual systems are only matched to
 * tables at creation time or after creation time, when a new table is created.
 *
 * Manual systems are useful to evaluate lists of pre-matched entities at
 * application defined times. Because none of the matching logic is evaluated
 * before the system is invoked, manual systems are much more efficient than
 * manually obtaining a list of entities and retrieving their components.
 *
 * An application may pass custom data to a system through the param parameter.
 * This data can be accessed by the system through the param member in the
 * ecs_iter_t value that is passed to the system callback.
 *
 * Any system may interrupt execution by setting the interrupted_by member in
 * the ecs_iter_t value. This is particularly useful for manual systems, where
 * the value of interrupted_by is returned by this operation. This, in
 * combination with the param argument lets applications use manual systems
 * to lookup entities: once the entity has been found its handle is passed to
 * interrupted_by, which is then subsequently returned.
 *
 * @param world The world.
 * @param system The system to run.
 * @param delta_time The time passed since the last system invocation.
 * @param param A user-defined parameter to pass to the system.
 * @return handle to last evaluated entity if system was interrupted.
 */
FLECS_API
ecs_entity_t ecs_run(
    ecs_world_t *world,
    ecs_entity_t system,
    ecs_ftime_t delta_time,
    void *param);

/** Same as ecs_run(), but subdivides entities across number of provided stages.
 *
 * @param world The world.
 * @param system The system to run.
 * @param stage_current The id of the current stage.
 * @param stage_count The total number of stages.
 * @param delta_time The time passed since the last system invocation.
 * @param param A user-defined parameter to pass to the system.
 * @return handle to last evaluated entity if system was interrupted.
 */
FLECS_API
ecs_entity_t ecs_run_worker(
    ecs_world_t *world,
    ecs_entity_t system,
    int32_t stage_current,
    int32_t stage_count,
    ecs_ftime_t delta_time,
    void *param);

/** System module import function.
 * Usage:
 * @code
 * ECS_IMPORT(world, FlecsSystem)
 * @endcode
 * 
 * @param world The world.
 */
FLECS_API
void FlecsSystemImport(
    ecs_world_t *world);

#ifdef __cplusplus
}
#endif

#endif

/** @} */

#endif

#endif

#ifdef FLECS_STATS
#ifdef FLECS_NO_STATS
#error "FLECS_NO_STATS failed: STATS is required by other addons"
#endif
/**
 * @file addons/stats.h
 * @brief Statistics addon.
 *
 * The stats addon tracks high resolution statistics for the world, systems and
 * pipelines. The addon can be used as an API where an application calls
 * functions to obtain statistics directly and as a module where statistics are
 * automatically tracked. The latter is required for statistics tracking in the
 * explorer.
 * 
 * When the addon is imported as module, statistics are tracked for each frame,
 * second, minute, hour, day and week with 60 datapoints per tier.
 */

#ifdef FLECS_STATS

/**
 * @defgroup c_addons_stats Stats
 * @ingroup c_addons
 * Collection of statistics for world, queries, systems and pipelines.
 *
 * @{
 */

#ifndef FLECS_STATS_H
#define FLECS_STATS_H

#ifndef FLECS_MODULE
#define FLECS_MODULE
#endif

#ifndef FLECS_PIPELINE
#define FLECS_PIPELINE
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define ECS_STAT_WINDOW (60)

/** Simple value that indicates current state */
typedef struct ecs_gauge_t {
    ecs_float_t avg[ECS_STAT_WINDOW];
    ecs_float_t min[ECS_STAT_WINDOW];
    ecs_float_t max[ECS_STAT_WINDOW];
} ecs_gauge_t;

/** Monotonically increasing counter */
typedef struct ecs_counter_t {
    ecs_gauge_t rate;                     /**< Keep track of deltas too */
    double value[ECS_STAT_WINDOW];
} ecs_counter_t;

/** Make all metrics the same size, so we can iterate over fields */
typedef union ecs_metric_t {
    ecs_gauge_t gauge;
    ecs_counter_t counter;
} ecs_metric_t;

typedef struct ecs_world_stats_t {
    int64_t first_;

    /* Entities */
    struct {
        ecs_metric_t count;               /**< Number of entities */
        ecs_metric_t not_alive_count;     /**< Number of not alive (recyclable) entity ids */
    } entities;

    /* Component ids */
    struct {
        ecs_metric_t tag_count;           /**< Number of tag ids (ids without data) */
        ecs_metric_t component_count;     /**< Number of components ids (ids with data) */
        ecs_metric_t pair_count;          /**< Number of pair ids */
        ecs_metric_t type_count;          /**< Number of registered types */
        ecs_metric_t create_count;        /**< Number of times id has been created */
        ecs_metric_t delete_count;        /**< Number of times id has been deleted */
    } components;

    /* Tables */
    struct {
        ecs_metric_t count;                /**< Number of tables */
        ecs_metric_t empty_count;          /**< Number of empty tables */
        ecs_metric_t create_count;         /**< Number of times table has been created */
        ecs_metric_t delete_count;         /**< Number of times table has been deleted */
    } tables;

    /* Queries & events */
    struct {
        ecs_metric_t query_count;          /**< Number of queries */
        ecs_metric_t observer_count;       /**< Number of observers */
        ecs_metric_t system_count;         /**< Number of systems */
    } queries;

    /* Commands */
    struct {
        ecs_metric_t add_count;
        ecs_metric_t remove_count;
        ecs_metric_t delete_count;
        ecs_metric_t clear_count;
        ecs_metric_t set_count;
        ecs_metric_t ensure_count;
        ecs_metric_t modified_count;
        ecs_metric_t other_count;
        ecs_metric_t discard_count;
        ecs_metric_t batched_entity_count;
        ecs_metric_t batched_count;
    } commands;

    /* Frame data */
    struct {
        ecs_metric_t frame_count;          /**< Number of frames processed. */
        ecs_metric_t merge_count;          /**< Number of merges executed. */
        ecs_metric_t rematch_count;        /**< Number of query rematches */
        ecs_metric_t pipeline_build_count; /**< Number of system pipeline rebuilds (occurs when an inactive system becomes active). */
        ecs_metric_t systems_ran;          /**< Number of systems ran. */
        ecs_metric_t observers_ran;        /**< Number of times an observer was invoked. */
        ecs_metric_t event_emit_count;     /**< Number of events emitted */
    } frame;

    /* Timing */
    struct {
        ecs_metric_t world_time_raw;       /**< Actual time passed since simulation start (first time progress() is called) */
        ecs_metric_t world_time;           /**< Simulation time passed since simulation start. Takes into account time scaling */
        ecs_metric_t frame_time;           /**< Time spent processing a frame. Smaller than world_time_total when load is not 100% */
        ecs_metric_t system_time;          /**< Time spent on running systems. */
        ecs_metric_t emit_time;            /**< Time spent on notifying observers. */
        ecs_metric_t merge_time;           /**< Time spent on merging commands. */
        ecs_metric_t rematch_time;         /**< Time spent on rematching. */
        ecs_metric_t fps;                  /**< Frames per second. */
        ecs_metric_t delta_time;           /**< Delta_time. */
    } performance;

    struct {
        /* Memory allocation data */
        ecs_metric_t alloc_count;          /**< Allocs per frame */
        ecs_metric_t realloc_count;        /**< Reallocs per frame */
        ecs_metric_t free_count;           /**< Frees per frame */
        ecs_metric_t outstanding_alloc_count; /**< Difference between allocs & frees */

        /* Memory allocator data */
        ecs_metric_t block_alloc_count;    /**< Block allocations per frame */
        ecs_metric_t block_free_count;     /**< Block frees per frame */
        ecs_metric_t block_outstanding_alloc_count; /**< Difference between allocs & frees */
        ecs_metric_t stack_alloc_count;    /**< Page allocations per frame */
        ecs_metric_t stack_free_count;     /**< Page frees per frame */
        ecs_metric_t stack_outstanding_alloc_count; /**< Difference between allocs & frees */
    } memory;

    /* HTTP statistics */
    struct {
        ecs_metric_t request_received_count;
        ecs_metric_t request_invalid_count;
        ecs_metric_t request_handled_ok_count;
        ecs_metric_t request_handled_error_count;
        ecs_metric_t request_not_handled_count;
        ecs_metric_t request_preflight_count;
        ecs_metric_t send_ok_count;
        ecs_metric_t send_error_count;
        ecs_metric_t busy_count;
    } http;

    int64_t last_;

    /** Current position in ring buffer */
    int32_t t;
} ecs_world_stats_t;

/** Statistics for a single query (use ecs_query_cache_stats_get) */
typedef struct ecs_query_stats_t {
    int64_t first_;
    ecs_metric_t result_count;              /**< Number of query results */
    ecs_metric_t matched_table_count;       /**< Number of matched tables */
    ecs_metric_t matched_entity_count;      /**< Number of matched entities */
    int64_t last_;

    /** Current position in ringbuffer */
    int32_t t; 
} ecs_query_stats_t;

/** Statistics for a single system (use ecs_system_stats_get()) */
typedef struct ecs_system_stats_t {
    int64_t first_;
    ecs_metric_t time_spent;       /**< Time spent processing a system */
    int64_t last_;

    bool task;                     /**< Is system a task */

    ecs_query_stats_t query;
} ecs_system_stats_t;

/** Statistics for sync point */
typedef struct ecs_sync_stats_t {
    int64_t first_;
    ecs_metric_t time_spent;
    ecs_metric_t commands_enqueued;
    int64_t last_;

    int32_t system_count;
    bool multi_threaded;
    bool immediate;
} ecs_sync_stats_t;

/** Statistics for all systems in a pipeline. */
typedef struct ecs_pipeline_stats_t {
    /* Allow for initializing struct with {0} */
    int8_t canary_;

    /** Vector with system ids of all systems in the pipeline. The systems are
     * stored in the order they are executed. Merges are represented by a 0. */
    ecs_vec_t systems;

    /** Vector with sync point stats */
    ecs_vec_t sync_points;

    /** Current position in ring buffer */
    int32_t t;

    int32_t system_count;        /**< Number of systems in pipeline */
    int32_t active_system_count; /**< Number of active systems in pipeline */
    int32_t rebuild_count;       /**< Number of times pipeline has rebuilt */
} ecs_pipeline_stats_t;

/** Get world statistics.
 *
 * @param world The world.
 * @param stats Out parameter for statistics.
 */
FLECS_API
void ecs_world_stats_get(
    const ecs_world_t *world,
    ecs_world_stats_t *stats);

/** Reduce source measurement window into single destination measurement. */
FLECS_API
void ecs_world_stats_reduce(
    ecs_world_stats_t *dst,
    const ecs_world_stats_t *src);

/** Reduce last measurement into previous measurement, restore old value. */
FLECS_API
void ecs_world_stats_reduce_last(
    ecs_world_stats_t *stats,
    const ecs_world_stats_t *old,
    int32_t count);

/** Repeat last measurement. */
FLECS_API
void ecs_world_stats_repeat_last(
    ecs_world_stats_t *stats);

/** Copy last measurement from source to destination. */
FLECS_API
void ecs_world_stats_copy_last(
    ecs_world_stats_t *dst,
    const ecs_world_stats_t *src);

FLECS_API
void ecs_world_stats_log(
    const ecs_world_t *world,
    const ecs_world_stats_t *stats);

/** Get query statistics.
 * Obtain statistics for the provided query.
 *
 * @param world The world.
 * @param query The query.
 * @param stats Out parameter for statistics.
 */
FLECS_API
void ecs_query_stats_get(
    const ecs_world_t *world,
    const ecs_query_t *query,
    ecs_query_stats_t *stats);

/** Reduce source measurement window into single destination measurement. */
FLECS_API 
void ecs_query_cache_stats_reduce(
    ecs_query_stats_t *dst,
    const ecs_query_stats_t *src);

/** Reduce last measurement into previous measurement, restore old value. */
FLECS_API
void ecs_query_cache_stats_reduce_last(
    ecs_query_stats_t *stats,
    const ecs_query_stats_t *old,
    int32_t count);

/** Repeat last measurement. */
FLECS_API
void ecs_query_cache_stats_repeat_last(
    ecs_query_stats_t *stats);

/** Copy last measurement from source to destination. */
FLECS_API
void ecs_query_cache_stats_copy_last(
    ecs_query_stats_t *dst,
    const ecs_query_stats_t *src);

/** Get system statistics.
 * Obtain statistics for the provided system.
 *
 * @param world The world.
 * @param system The system.
 * @param stats Out parameter for statistics.
 * @return true if success, false if not a system.
 */
FLECS_API
bool ecs_system_stats_get(
    const ecs_world_t *world,
    ecs_entity_t system,
    ecs_system_stats_t *stats);

/** Reduce source measurement window into single destination measurement */
FLECS_API
void ecs_system_stats_reduce(
    ecs_system_stats_t *dst,
    const ecs_system_stats_t *src);

/** Reduce last measurement into previous measurement, restore old value. */
FLECS_API
void ecs_system_stats_reduce_last(
    ecs_system_stats_t *stats,
    const ecs_system_stats_t *old,
    int32_t count);

/** Repeat last measurement. */
FLECS_API
void ecs_system_stats_repeat_last(
    ecs_system_stats_t *stats);

/** Copy last measurement from source to destination. */
FLECS_API
void ecs_system_stats_copy_last(
    ecs_system_stats_t *dst,
    const ecs_system_stats_t *src);

/** Get pipeline statistics.
 * Obtain statistics for the provided pipeline.
 *
 * @param world The world.
 * @param pipeline The pipeline.
 * @param stats Out parameter for statistics.
 * @return true if success, false if not a pipeline.
 */
FLECS_API
bool ecs_pipeline_stats_get(
    ecs_world_t *world,
    ecs_entity_t pipeline,
    ecs_pipeline_stats_t *stats);

/** Free pipeline stats.
 *
 * @param stats The stats to free.
 */
FLECS_API
void ecs_pipeline_stats_fini(
    ecs_pipeline_stats_t *stats);

/** Reduce source measurement window into single destination measurement */
FLECS_API
void ecs_pipeline_stats_reduce(
    ecs_pipeline_stats_t *dst,
    const ecs_pipeline_stats_t *src);

/** Reduce last measurement into previous measurement, restore old value. */
FLECS_API
void ecs_pipeline_stats_reduce_last(
    ecs_pipeline_stats_t *stats,
    const ecs_pipeline_stats_t *old,
    int32_t count);

/** Repeat last measurement. */
FLECS_API
void ecs_pipeline_stats_repeat_last(
    ecs_pipeline_stats_t *stats);

/** Copy last measurement to destination.
 * This operation copies the last measurement into the destination. It does not
 * modify the cursor.
 *
 * @param dst The metrics.
 * @param src The metrics to copy.
 */
FLECS_API
void ecs_pipeline_stats_copy_last(
    ecs_pipeline_stats_t *dst,
    const ecs_pipeline_stats_t *src);

/** Reduce all measurements from a window into a single measurement. */
FLECS_API
void ecs_metric_reduce(
    ecs_metric_t *dst,
    const ecs_metric_t *src,
    int32_t t_dst,
    int32_t t_src);

/** Reduce last measurement into previous measurement */
FLECS_API
void ecs_metric_reduce_last(
    ecs_metric_t *m,
    int32_t t,
    int32_t count);

/** Copy measurement */
FLECS_API
void ecs_metric_copy(
    ecs_metric_t *m,
    int32_t dst,
    int32_t src);

FLECS_API extern ECS_COMPONENT_DECLARE(FlecsStats);        /**< Flecs stats module. */
FLECS_API extern ECS_COMPONENT_DECLARE(EcsWorldStats);     /**< Component id for EcsWorldStats. */
FLECS_API extern ECS_COMPONENT_DECLARE(EcsWorldSummary);   /**< Component id for EcsWorldSummary. */
FLECS_API extern ECS_COMPONENT_DECLARE(EcsSystemStats);    /**< Component id for EcsSystemStats. */
FLECS_API extern ECS_COMPONENT_DECLARE(EcsPipelineStats);  /**< Component id for EcsPipelineStats. */

FLECS_API extern ecs_entity_t EcsPeriod1s;                 /**< Tag used for metrics collected in last second. */
FLECS_API extern ecs_entity_t EcsPeriod1m;                 /**< Tag used for metrics collected in last minute. */
FLECS_API extern ecs_entity_t EcsPeriod1h;                 /**< Tag used for metrics collected in last hour. */
FLECS_API extern ecs_entity_t EcsPeriod1d;                 /**< Tag used for metrics collected in last day. */
FLECS_API extern ecs_entity_t EcsPeriod1w;                 /**< Tag used for metrics collected in last week. */

/** Common data for statistics. */
typedef struct {
    ecs_ftime_t elapsed;
    int32_t reduce_count;
} EcsStatsHeader;

/** Component that stores world statistics. */
typedef struct {
    EcsStatsHeader hdr;
    ecs_world_stats_t stats;
} EcsWorldStats;

/** Component that stores system statistics. */
typedef struct {
    EcsStatsHeader hdr;
    ecs_map_t stats;
} EcsSystemStats;

/** Component that stores pipeline statistics. */
typedef struct {
    EcsStatsHeader hdr;
    ecs_map_t stats;
} EcsPipelineStats;

/** Component that stores a summary of world statistics. */
typedef struct {
    /* Time */
    double target_fps;          /**< Target FPS */
    double time_scale;          /**< Simulation time scale */

    /* Total time */
    double frame_time_total;    /**< Total time spent processing a frame */
    double system_time_total;   /**< Total time spent in systems */
    double merge_time_total;    /**< Total time spent in merges */

    /* Last frame time */
    double frame_time_last;     /**< Time spent processing a frame */
    double system_time_last;    /**< Time spent in systems */
    double merge_time_last;     /**< Time spent in merges */

    int64_t frame_count;        /**< Number of frames processed */
    int64_t command_count;      /**< Number of commands processed */

    /* Build info */
    ecs_build_info_t build_info; /**< Build info */
} EcsWorldSummary;

/** Stats module import function.
 * Usage:
 * @code
 * ECS_IMPORT(world, FlecsStats)
 * @endcode
 * 
 * @param world The world.
 */
FLECS_API
void FlecsStatsImport(
    ecs_world_t *world);

#ifdef __cplusplus
}
#endif

#endif

/** @} */

#endif

#endif

#ifdef FLECS_METRICS
#ifdef FLECS_NO_METRICS
#error "FLECS_NO_METRICS failed: METRICS is required by other addons"
#endif
/**
 * @file addons/metrics.h
 * @brief Metrics module.
 *
 * The metrics module extracts metrics from components and makes them available
 * through a unified component interface.
 */

#ifdef FLECS_METRICS

/**
 * @defgroup c_addons_metrics Metrics
 * @ingroup c_addons
 * Collect user-defined metrics from ECS data.
 *
 * @{
 */

#ifndef FLECS_METRICS_H
#define FLECS_METRICS_H

#ifndef FLECS_META
#define FLECS_META
#endif

#ifndef FLECS_UNITS
#define FLECS_UNITS
#endif

#ifndef FLECS_PIPELINE
#define FLECS_PIPELINE
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** Flecs metrics module. */
FLECS_API extern ECS_COMPONENT_DECLARE(FlecsMetrics);

/** Tag added to metrics, and used as first element of metric kind pair. */
FLECS_API extern ECS_TAG_DECLARE(EcsMetric);

/** Metric that has monotonically increasing value. */
FLECS_API extern ECS_TAG_DECLARE(EcsCounter);

/** Counter metric that is auto-incremented by source value. */
FLECS_API extern ECS_TAG_DECLARE(EcsCounterIncrement);

/** Counter metric that counts the number of entities with an id. */
FLECS_API extern ECS_TAG_DECLARE(EcsCounterId);

/** Metric that represents current value. */
FLECS_API extern ECS_TAG_DECLARE(EcsGauge);

/** Tag added to metric instances. */
FLECS_API extern ECS_TAG_DECLARE(EcsMetricInstance);

/** Component with metric instance value. */
FLECS_API extern ECS_COMPONENT_DECLARE(EcsMetricValue);

/** Component with entity source of metric instance. */
FLECS_API extern ECS_COMPONENT_DECLARE(EcsMetricSource);

/** Component that stores metric value. */
typedef struct EcsMetricValue {
    double value;
} EcsMetricValue;

/** Component that stores metric source. */
typedef struct EcsMetricSource {
    ecs_entity_t entity;
} EcsMetricSource;

/** Used with ecs_metric_init to create metric. */
typedef struct ecs_metric_desc_t {
    int32_t _canary;

    /** Entity associated with metric */
    ecs_entity_t entity;

    /** Entity associated with member that stores metric value. Must not be set
     * at the same time as id. Cannot be combined with EcsCounterId. */
    ecs_entity_t member;

    /* Member dot expression. Can be used instead of member and supports nested
     * members. Must be set together with id and should not be set at the same
     * time as member. */
    const char *dotmember;

    /** Tracks whether entities have the specified component id. Must not be set
     * at the same time as member. */
    ecs_id_t id;

    /** If id is a (R, *) wildcard and relationship R has the OneOf property,
     * setting this value to true will track individual targets.
     * If the kind is EcsCountId and the id is a (R, *) wildcard, this value
     * will create a metric per target. */
    bool targets;

    /** Must be EcsGauge, EcsCounter, EcsCounterIncrement or EcsCounterId */
    ecs_entity_t kind;

    /** Description of metric. Will only be set if FLECS_DOC addon is enabled */
    const char *brief;
} ecs_metric_desc_t;

/** Create a new metric.
 * Metrics are entities that store values measured from a range of different
 * properties in the ECS storage. Metrics provide a single unified interface to
 * discovering and reading these values, which can be useful for monitoring
 * utilities, or for debugging.
 *
 * Examples of properties that can be measured by metrics are:
 *  - Component member values
 *  - How long an entity has had a specific component
 *  - How long an entity has had a specific target for a relationship
 *  - How many entities have a specific component
 *
 * Metrics can either be created as a "gauge" or "counter". A gauge is a metric
 * that represents the value of something at a specific point in time, for
 * example "velocity". A counter metric represents a value that is monotonically
 * increasing, for example "miles driven".
 *
 * There are three different kinds of counter metric kinds:
 * - EcsCounter
 *   When combined with a member, this will store the actual value of the member
 *   in the metric. This is useful for values that are already counters, such as
 *   a MilesDriven component.
 *   This kind creates a metric per entity that has the member/id.
 *
 * - EcsCounterIncrement
 *   When combined with a member, this will increment the value of the metric by
 *   the value of the member * delta_time. This is useful for values that are
 *   not counters, such as a Velocity component.
 *   This kind creates a metric per entity that has the member.
 *
 * - EcsCounterId
 *   This metric kind will count the number of entities with a specific
 *   (component) id. This kind creates a single metric instance for regular ids,
 *   and a metric instance per target for wildcard ids when targets is set.
 *
 * @param world The world.
 * @param desc Metric description.
 * @return The metric entity.
 */
FLECS_API
ecs_entity_t ecs_metric_init(
    ecs_world_t *world,
    const ecs_metric_desc_t *desc);

/** Shorthand for creating a metric with ecs_metric_init().
 *
 * Example:
 *
 * @code
 * ecs_metric(world, {
 *   .member = ecs_lookup(world, "Position.x")
 *   .kind = EcsGauge
 * });
 * @endcode
 */
#define ecs_metric(world, ...)\
    ecs_metric_init(world, &(ecs_metric_desc_t) __VA_ARGS__ )

/** Metrics module import function.
 * Usage:
 * @code
 * ECS_IMPORT(world, FlecsMetrics)
 * @endcode
 * 
 * @param world The world.
 */
FLECS_API
void FlecsMetricsImport(
    ecs_world_t *world);

#ifdef __cplusplus
}
#endif

#endif

/** @} */

#endif

#endif

#ifdef FLECS_ALERTS
#ifdef FLECS_NO_ALERTS
#error "FLECS_NO_ALERTS failed: ALERTS is required by other addons"
#endif
/**
 * @file addons/alerts.h
 * @brief Alerts module.
 *
 * The alerts module enables applications to register alerts for when certain
 * conditions are met. Alerts are registered as queries, and automatically
 * become active when entities match the alert query.
 */

#ifdef FLECS_ALERTS

/**
 * @defgroup c_addons_alerts Alerts
 * @ingroup c_addons
 * Create alerts from monitoring queries.
 *
 * @{
 */

#ifndef FLECS_ALERTS_H
#define FLECS_ALERTS_H

#ifndef FLECS_PIPELINE
#define FLECS_PIPELINE
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define ECS_ALERT_MAX_SEVERITY_FILTERS (4)

/** Module id. */
FLECS_API extern ECS_COMPONENT_DECLARE(FlecsAlerts);

/* Module components */

FLECS_API extern ECS_COMPONENT_DECLARE(EcsAlert);          /**< Component added to alert, and used as first element of alert severity pair. */
FLECS_API extern ECS_COMPONENT_DECLARE(EcsAlertInstance);  /**< Component added to alert instance. */
FLECS_API extern ECS_COMPONENT_DECLARE(EcsAlertsActive);   /**< Component added to alert source which tracks how many active alerts there are. */
FLECS_API extern ECS_COMPONENT_DECLARE(EcsAlertTimeout);   /**< Component added to alert which tracks how long an alert has been inactive. */

/* Alert severity tags */
FLECS_API extern ECS_TAG_DECLARE(EcsAlertInfo);            /**< Info alert severity. */
FLECS_API extern ECS_TAG_DECLARE(EcsAlertWarning);         /**< Warning alert severity. */
FLECS_API extern ECS_TAG_DECLARE(EcsAlertError);           /**< Error alert severity. */
FLECS_API extern ECS_TAG_DECLARE(EcsAlertCritical);        /**< Critical alert severity. */

/** Component added to alert instance. */
typedef struct EcsAlertInstance {
    char *message; /**< Generated alert message */
} EcsAlertInstance;

/** Map with active alerts for entity. */
typedef struct EcsAlertsActive {
    int32_t info_count;    /**< Number of alerts for source with info severity */
    int32_t warning_count; /**< Number of alerts for source with warning severity */
    int32_t error_count;   /**< Number of alerts for source with error severity */
    ecs_map_t alerts;
} EcsAlertsActive;

/** Alert severity filter. 
 * A severity filter can adjust the severity of an alert based on whether an
 * entity in the alert query has a specific component. For example, a filter
 * could check if an entity has the "Production" tag, and increase the default
 * severity of an alert from Warning to Error.
 */
typedef struct ecs_alert_severity_filter_t {
    ecs_entity_t severity; /* Severity kind */
    ecs_id_t with;         /* Component to match */
    const char *var;       /* Variable to match component on. Do not include the
                            * '$' character. Leave to NULL for $this. */
    int32_t _var_index;    /* Index of variable in filter (do not set) */
} ecs_alert_severity_filter_t;

/** Alert descriptor, used with ecs_alert_init(). */
typedef struct ecs_alert_desc_t {
    int32_t _canary;

    /** Entity associated with alert */
    ecs_entity_t entity;

    /** Alert query. An alert will be created for each entity that matches the
     * specified query. The query must have at least one term that uses the
     * $this variable (default). */
    ecs_query_desc_t query;

    /** Template for alert message. This string is used to generate the alert
     * message and may refer to variables in the query result. The format for
     * the template expressions is as specified by ecs_script_string_interpolate().
     *
     * Examples:
     *
     *     "$this has Position but not Velocity"
     *     "$this has a parent entity $parent without Position"
     */
    const char *message;

    /** User friendly name. Will only be set if FLECS_DOC addon is enabled. */
    const char *doc_name;

    /** Description of alert. Will only be set if FLECS_DOC addon is enabled */
    const char *brief;

    /** Metric kind. Must be EcsAlertInfo, EcsAlertWarning, EcsAlertError or
     * EcsAlertCritical. Defaults to EcsAlertError. */
    ecs_entity_t severity;

    /** Severity filters can be used to assign different severities to the same
     * alert. This prevents having to create multiple alerts, and allows
     * entities to transition between severities without resetting the
     * alert duration (optional). */
    ecs_alert_severity_filter_t severity_filters[ECS_ALERT_MAX_SEVERITY_FILTERS];

    /** The retain period specifies how long an alert must be inactive before it
     * is cleared. This makes it easier to track noisy alerts. While an alert is
     * inactive its duration won't increase.
     * When the retain period is 0, the alert will clear immediately after it no
     * longer matches the alert query. */
    ecs_ftime_t retain_period;

    /** Alert when member value is out of range. Uses the warning/error ranges
     * assigned to the member in the MemberRanges component (optional). */
    ecs_entity_t member;

    /** (Component) id of member to monitor. If left to 0 this will be set to
     * the parent entity of the member (optional). */
    ecs_id_t id;

    /** Variable from which to fetch the member (optional). When left to NULL
     * 'id' will be obtained from $this. */
    const char *var;
} ecs_alert_desc_t;

/** Create a new alert.
 * An alert is a query that is evaluated periodically and creates alert
 * instances for each entity that matches the query. Alerts can be used to
 * automate detection of errors in an application.
 *
 * Alerts are automatically cleared when a query is no longer true for an alert
 * instance. At most one alert instance will be created per matched entity.
 *
 * Alert instances have three components:
 * - AlertInstance: contains the alert message for the instance
 * - MetricSource: contains the entity that triggered the alert
 * - MetricValue: contains how long the alert has been active
 *
 * Alerts reuse components from the metrics addon so that alert instances can be
 * tracked and discovered as metrics. Just like metrics, alert instances are
 * created as children of the alert.
 *
 * When an entity has active alerts, it will have the EcsAlertsActive component
 * which contains a map with active alerts for the entity. This component
 * will be automatically removed once all alerts are cleared for the entity.
 *
 * @param world The world.
 * @param desc Alert description.
 * @return The alert entity.
 */
FLECS_API
ecs_entity_t ecs_alert_init(
    ecs_world_t *world,
    const ecs_alert_desc_t *desc);

/** Create a new alert.
 * @see ecs_alert_init()
 */
#define ecs_alert(world, ...)\
    ecs_alert_init(world, &(ecs_alert_desc_t)__VA_ARGS__)

/** Return number of active alerts for entity.
 * When a valid alert entity is specified for the alert parameter, the operation
 * will return whether the specified alert is active for the entity. When no
 * alert is specified, the operation will return the total number of active
 * alerts for the entity.
 *
 * @param world The world.
 * @param entity The entity.
 * @param alert The alert to test for (optional).
 * @return The number of active alerts for the entity.
 */
FLECS_API
int32_t ecs_get_alert_count(
    const ecs_world_t *world,
    ecs_entity_t entity,
    ecs_entity_t alert);

/** Return alert instance for specified alert.
 * This operation returns the alert instance for the specified alert. If the
 * alert is not active for the entity, the operation will return 0.
 *
 * @param world The world.
 * @param entity The entity.
 * @param alert The alert to test for.
 * @return The alert instance for the specified alert.
 */
FLECS_API
ecs_entity_t ecs_get_alert(
    const ecs_world_t *world,
    ecs_entity_t entity,
    ecs_entity_t alert);

/** Alert module import function.
 * Usage:
 * @code
 * ECS_IMPORT(world, FlecsAlerts)
 * @endcode
 * 
 * @param world The world.
 */
FLECS_API
void FlecsAlertsImport(
    ecs_world_t *world);

#ifdef __cplusplus
}
#endif

#endif

/** @} */

#endif

#endif

#ifdef FLECS_JSON
#ifdef FLECS_NO_JSON
#error "FLECS_NO_JSON failed: JSON is required by other addons"
#endif
/**
 * @file addons/json.h
 * @brief JSON parser addon.
 *
 * Parse expression strings into component values. Entity identifiers,
 * enumerations and bitmasks are encoded as strings.
 *
 * See docs/FlecsRemoteApi.md for a description of the JSON format.
 */

#ifdef FLECS_JSON

#ifndef FLECS_META
#define FLECS_META
#endif

#ifndef FLECS_SCRIPT
#define FLECS_SCRIPT
#endif

#ifndef FLECS_JSON_H
#define FLECS_JSON_H

/**
 * @defgroup c_addons_json Json
 * @ingroup c_addons
 * Functions for serializing to/from JSON.
 *
 * @{
 */

#ifdef __cplusplus
extern "C" {
#endif

/** Used with ecs_ptr_from_json(), ecs_entity_from_json(). */
typedef struct ecs_from_json_desc_t {
    const char *name; /**< Name of expression (used for logging) */
    const char *expr; /**< Full expression (used for logging) */

    /** Callback that allows for specifying a custom lookup function. The
     * default behavior uses ecs_lookup() */
    ecs_entity_t (*lookup_action)(
        const ecs_world_t*,
        const char *value,
        void *ctx);
    void *lookup_ctx;

    /** Require components to be registered with reflection data. When not
     * in strict mode, values for components without reflection are ignored. */
    bool strict;
} ecs_from_json_desc_t;

/** Parse JSON string into value.
 * This operation parses a JSON expression into the provided pointer. The
 * memory pointed to must be large enough to contain a value of the used type.
 *
 * @param world The world.
 * @param type The type of the expression to parse.
 * @param ptr Pointer to the memory to write to.
 * @param json The JSON expression to parse.
 * @param desc Configuration parameters for deserializer.
 * @return Pointer to the character after the last one read, or NULL if failed.
 */
FLECS_API
const char* ecs_ptr_from_json(
    const ecs_world_t *world,
    ecs_entity_t type,
    void *ptr,
    const char *json,
    const ecs_from_json_desc_t *desc);

/** Parse JSON object with multiple component values into entity. The format
 * is the same as the one outputted by ecs_entity_to_json(), but at the moment
 * only supports the "ids" and "values" member.
 *
 * @param world The world.
 * @param entity The entity to serialize to.
 * @param json The JSON expression to parse (see entity in JSON format manual).
 * @param desc Configuration parameters for deserializer.
 * @return Pointer to the character after the last one read, or NULL if failed.
 */
FLECS_API
const char* ecs_entity_from_json(
    ecs_world_t *world,
    ecs_entity_t entity,
    const char *json,
    const ecs_from_json_desc_t *desc);

/** Parse JSON object with multiple entities into the world. The format is the
 * same as the one outputted by ecs_world_to_json().
 *
 * @param world The world.
 * @param json The JSON expression to parse (see iterator in JSON format manual).
 * @param desc Deserialization parameters.
 * @return Last deserialized character, NULL if failed.
 */
FLECS_API
const char* ecs_world_from_json(
    ecs_world_t *world,
    const char *json,
    const ecs_from_json_desc_t *desc);

/** Same as ecs_world_from_json(), but loads JSON from file.
 *
 * @param world The world.
 * @param filename The file from which to load the JSON.
 * @param desc Deserialization parameters.
 * @return Last deserialized character, NULL if failed.
 */
FLECS_API
const char* ecs_world_from_json_file(
    ecs_world_t *world,
    const char *filename,
    const ecs_from_json_desc_t *desc);

/** Serialize array into JSON string.
 * This operation serializes a value of the provided type to a JSON string. The
 * memory pointed to must be large enough to contain a value of the used type.
 *
 * If count is 0, the function will serialize a single value, not wrapped in
 * array brackets. If count is >= 1, the operation will serialize values to a
 * a comma-separated list inside of array brackets.
 *
 * @param world The world.
 * @param type The type of the value to serialize.
 * @param data The value to serialize.
 * @param count The number of elements to serialize.
 * @return String with JSON expression, or NULL if failed.
 */
FLECS_API
char* ecs_array_to_json(
    const ecs_world_t *world,
    ecs_entity_t type,
    const void *data,
    int32_t count);

/** Serialize array into JSON string buffer.
 * Same as ecs_array_to_json(), but serializes to an ecs_strbuf_t instance.
 *
 * @param world The world.
 * @param type The type of the value to serialize.
 * @param data The value to serialize.
 * @param count The number of elements to serialize.
 * @param buf_out The strbuf to append the string to.
 * @return Zero if success, non-zero if failed.
 */
FLECS_API
int ecs_array_to_json_buf(
    const ecs_world_t *world,
    ecs_entity_t type,
    const void *data,
    int32_t count,
    ecs_strbuf_t *buf_out);

/** Serialize value into JSON string.
 * Same as ecs_array_to_json(), with count = 0.
 *
 * @param world The world.
 * @param type The type of the value to serialize.
 * @param data The value to serialize.
 * @return String with JSON expression, or NULL if failed.
 */
FLECS_API
char* ecs_ptr_to_json(
    const ecs_world_t *world,
    ecs_entity_t type,
    const void *data);

/** Serialize value into JSON string buffer.
 * Same as ecs_ptr_to_json(), but serializes to an ecs_strbuf_t instance.
 *
 * @param world The world.
 * @param type The type of the value to serialize.
 * @param data The value to serialize.
 * @param buf_out The strbuf to append the string to.
 * @return Zero if success, non-zero if failed.
 */
FLECS_API
int ecs_ptr_to_json_buf(
    const ecs_world_t *world,
    ecs_entity_t type,
    const void *data,
    ecs_strbuf_t *buf_out);

/** Serialize type info to JSON.
 * This serializes type information to JSON, and can be used to store/transmit
 * the structure of a (component) value.
 *
 * If the provided type does not have reflection data, "0" will be returned.
 *
 * @param world The world.
 * @param type The type to serialize to JSON.
 * @return A JSON string with the serialized type info, or NULL if failed.
 */
FLECS_API
char* ecs_type_info_to_json(
    const ecs_world_t *world,
    ecs_entity_t type);

/** Serialize type info into JSON string buffer.
 * Same as ecs_type_info_to_json(), but serializes to an ecs_strbuf_t instance.
 *
 * @param world The world.
 * @param type The type to serialize.
 * @param buf_out The strbuf to append the string to.
 * @return Zero if success, non-zero if failed.
 */
FLECS_API
int ecs_type_info_to_json_buf(
    const ecs_world_t *world,
    ecs_entity_t type,
    ecs_strbuf_t *buf_out);

/** Used with ecs_iter_to_json(). */
typedef struct ecs_entity_to_json_desc_t {
    bool serialize_entity_id;  /**< Serialize entity id */
    bool serialize_doc;        /**< Serialize doc attributes */
    bool serialize_full_paths; /**< Serialize full paths for tags, components and pairs */
    bool serialize_inherited;  /**< Serialize base components */
    bool serialize_values;     /**< Serialize component values */
    bool serialize_builtin;    /**< Serialize builtin data as components (e.g. "name", "parent") */
    bool serialize_type_info;  /**< Serialize type info (requires serialize_values) */
    bool serialize_alerts;     /**< Serialize active alerts for entity */
    ecs_entity_t serialize_refs; /**< Serialize references (incoming edges) for relationship */
    bool serialize_matches;    /**< Serialize which queries entity matches with */
} ecs_entity_to_json_desc_t;

/** Utility used to initialize JSON entity serializer. */
#ifndef __cplusplus
#define ECS_ENTITY_TO_JSON_INIT (ecs_entity_to_json_desc_t){\
    .serialize_entity_id = false, \
    .serialize_doc = false, \
    .serialize_full_paths = true, \
    .serialize_inherited = false, \
    .serialize_values = true, \
    .serialize_builtin = false, \
    .serialize_type_info = false, \
    .serialize_alerts = false, \
    .serialize_refs = 0, \
    .serialize_matches = false, \
}
#else
#define ECS_ENTITY_TO_JSON_INIT {\
    false, \
    false, \
    true, \
    false, \
    true, \
    false, \
    false, \
    false, \
    0, \
    false, \
}
#endif

/** Serialize entity into JSON string.
 * This creates a JSON object with the entity's (path) name, which components
 * and tags the entity has, and the component values.
 *
 * The operation may fail if the entity contains components with invalid values.
 *
 * @param world The world.
 * @param entity The entity to serialize to JSON.
 * @return A JSON string with the serialized entity data, or NULL if failed.
 */
FLECS_API
char* ecs_entity_to_json(
    const ecs_world_t *world,
    ecs_entity_t entity,
    const ecs_entity_to_json_desc_t *desc);

/** Serialize entity into JSON string buffer.
 * Same as ecs_entity_to_json(), but serializes to an ecs_strbuf_t instance.
 *
 * @param world The world.
 * @param entity The entity to serialize.
 * @param buf_out The strbuf to append the string to.
 * @return Zero if success, non-zero if failed.
 */
FLECS_API
int ecs_entity_to_json_buf(
    const ecs_world_t *world,
    ecs_entity_t entity,
    ecs_strbuf_t *buf_out,
    const ecs_entity_to_json_desc_t *desc);

/** Used with ecs_iter_to_json(). */
typedef struct ecs_iter_to_json_desc_t {
    bool serialize_entity_ids;      /**< Serialize entity ids */
    bool serialize_values;          /**< Serialize component values */
    bool serialize_builtin;         /**< Serialize builtin data as components (e.g. "name", "parent") */
    bool serialize_doc;             /**< Serialize doc attributes */
    bool serialize_full_paths;      /**< Serialize full paths for tags, components and pairs */
    bool serialize_fields;          /**< Serialize field data */
    bool serialize_inherited;       /**< Serialize inherited components */
    bool serialize_table;           /**< Serialize entire table vs. matched components */
    bool serialize_type_info;       /**< Serialize type information */
    bool serialize_field_info;      /**< Serialize metadata for fields returned by query */
    bool serialize_query_info;      /**< Serialize query terms */
    bool serialize_query_plan;      /**< Serialize query plan */
    bool serialize_query_profile;   /**< Profile query performance */
    bool dont_serialize_results;    /**< If true, query won't be evaluated */
    bool serialize_alerts;          /**< Serialize active alerts for entity */
    ecs_entity_t serialize_refs;    /**< Serialize references (incoming edges) for relationship */
    bool serialize_matches;         /**< Serialize which queries entity matches with */
    ecs_poly_t *query;            /**< Query object (required for serialize_query_[plan|profile]). */
} ecs_iter_to_json_desc_t;

/** Utility used to initialize JSON iterator serializer. */
#ifndef __cplusplus
#define ECS_ITER_TO_JSON_INIT (ecs_iter_to_json_desc_t){\
    .serialize_entity_ids =      false, \
    .serialize_values =          true, \
    .serialize_builtin =         false, \
    .serialize_doc =             false, \
    .serialize_full_paths =      true, \
    .serialize_fields =          true, \
    .serialize_inherited =       false, \
    .serialize_table =           false, \
    .serialize_type_info =       false, \
    .serialize_field_info =      false, \
    .serialize_query_info =      false, \
    .serialize_query_plan =      false, \
    .serialize_query_profile =   false, \
    .dont_serialize_results =    false, \
    .serialize_alerts =          false, \
    .serialize_refs =            false, \
    .serialize_matches =         false, \
    .query =                     NULL \
}
#else
#define ECS_ITER_TO_JSON_INIT {\
    false, \
    true, \
    false, \
    false, \
    true, \
    true, \
    false, \
    false, \
    false, \
    false, \
    false, \
    false, \
    false, \
    false, \
    false, \
    false, \
    false, \
    nullptr \
}
#endif

/** Serialize iterator into JSON string.
 * This operation will iterate the contents of the iterator and serialize them
 * to JSON. The function accepts iterators from any source.
 *
 * @param iter The iterator to serialize to JSON.
 * @return A JSON string with the serialized iterator data, or NULL if failed.
 */
FLECS_API
char* ecs_iter_to_json(
    ecs_iter_t *iter,
    const ecs_iter_to_json_desc_t *desc);

/** Serialize iterator into JSON string buffer.
 * Same as ecs_iter_to_json(), but serializes to an ecs_strbuf_t instance.
 *
 * @param iter The iterator to serialize.
 * @param buf_out The strbuf to append the string to.
 * @return Zero if success, non-zero if failed.
 */
FLECS_API
int ecs_iter_to_json_buf(
    ecs_iter_t *iter,
    ecs_strbuf_t *buf_out,
    const ecs_iter_to_json_desc_t *desc);

/** Used with ecs_iter_to_json(). */
typedef struct ecs_world_to_json_desc_t {
    bool serialize_builtin;    /**< Exclude flecs modules & contents */
    bool serialize_modules;    /**< Exclude modules & contents */
} ecs_world_to_json_desc_t;

/** Serialize world into JSON string.
 * This operation iterates the contents of the world to JSON. The operation is
 * equivalent to the following code:
 *
 * @code
 * ecs_query_t *f = ecs_query(world, {
 *   .terms = {{ .id = EcsAny }}
 * });
 *
 * ecs_iter_t it = ecs_query_init(world, &f);
 * ecs_iter_to_json_desc_t desc = { .serialize_table = true };
 * ecs_iter_to_json(iter, &desc);
 * @endcode
 *
 * @param world The world to serialize.
 * @return A JSON string with the serialized iterator data, or NULL if failed.
 */
FLECS_API
char* ecs_world_to_json(
    ecs_world_t *world,
    const ecs_world_to_json_desc_t *desc);

/** Serialize world into JSON string buffer.
 * Same as ecs_world_to_json(), but serializes to an ecs_strbuf_t instance.
 *
 * @param world The world to serialize.
 * @param buf_out The strbuf to append the string to.
 * @return Zero if success, non-zero if failed.
 */
FLECS_API
int ecs_world_to_json_buf(
    ecs_world_t *world,
    ecs_strbuf_t *buf_out,
    const ecs_world_to_json_desc_t *desc);

#ifdef __cplusplus
}
#endif

#endif

/** @} */

#endif

#endif

#ifdef FLECS_UNITS
#ifdef FLECS_NO_UNITS
#error "FLECS_NO_UNITS failed: UNITS is required by other addons"
#endif
/**
 * @file addons/units.h
 * @brief Units module.
 *
 * Builtin standard units. The units addon is not imported by default, even if
 * the addon is included in the build. To import the module, do:
 *
 * In C:
 *
 * @code
 * ECS_IMPORT(world, FlecsUnits);
 * @endcode
 *
 * In C++:
 *
 * @code
 * world.import<flecs::units>();
 * @endcode
 *
 * As a result this module behaves just like an application-defined module,
 * which means that the ids generated for the entities inside the module are not
 * fixed, and depend on the order in which the module is imported.
 */

#ifdef FLECS_UNITS

/**
 * @defgroup c_addons_units Units.
 * @ingroup c_addons
 * Common unit annotations for reflection framework.
 *
 * @{
 */

#ifndef FLECS_MODULE
#define FLECS_MODULE
#endif

#ifndef FLECS_META
#define FLECS_META
#endif

#ifndef FLECS_UNITS_H
#define FLECS_UNITS_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup c_addons_units_prefixes Prefixes
 * @ingroup c_addons_units
 * Prefixes to indicate unit count (e.g. Kilo, Mega)
 *
 * @{
 */

FLECS_API extern ecs_entity_t EcsUnitPrefixes; /**< Parent scope for prefixes. */

FLECS_API extern ecs_entity_t EcsYocto;  /**< Yocto unit prefix. */
FLECS_API extern ecs_entity_t EcsZepto;  /**< Zepto unit prefix. */
FLECS_API extern ecs_entity_t EcsAtto;   /**< Atto unit prefix. */
FLECS_API extern ecs_entity_t EcsFemto;  /**< Femto unit prefix. */
FLECS_API extern ecs_entity_t EcsPico;   /**< Pico unit prefix. */
FLECS_API extern ecs_entity_t EcsNano;   /**< Nano unit prefix. */
FLECS_API extern ecs_entity_t EcsMicro;  /**< Micro unit prefix. */
FLECS_API extern ecs_entity_t EcsMilli;  /**< Milli unit prefix. */
FLECS_API extern ecs_entity_t EcsCenti;  /**< Centi unit prefix. */
FLECS_API extern ecs_entity_t EcsDeci;   /**< Deci unit prefix. */
FLECS_API extern ecs_entity_t EcsDeca;   /**< Deca unit prefix. */
FLECS_API extern ecs_entity_t EcsHecto;  /**< Hecto unit prefix. */
FLECS_API extern ecs_entity_t EcsKilo;   /**< Kilo unit prefix. */
FLECS_API extern ecs_entity_t EcsMega;   /**< Mega unit prefix. */
FLECS_API extern ecs_entity_t EcsGiga;   /**< Giga unit prefix. */
FLECS_API extern ecs_entity_t EcsTera;   /**< Tera unit prefix. */
FLECS_API extern ecs_entity_t EcsPeta;   /**< Peta unit prefix. */
FLECS_API extern ecs_entity_t EcsExa;    /**< Exa unit prefix. */
FLECS_API extern ecs_entity_t EcsZetta;  /**< Zetta unit prefix. */
FLECS_API extern ecs_entity_t EcsYotta;  /**< Yotta unit prefix. */

FLECS_API extern ecs_entity_t EcsKibi;   /**< Kibi unit prefix. */
FLECS_API extern ecs_entity_t EcsMebi;   /**< Mebi unit prefix. */
FLECS_API extern ecs_entity_t EcsGibi;   /**< Gibi unit prefix. */
FLECS_API extern ecs_entity_t EcsTebi;   /**< Tebi unit prefix. */
FLECS_API extern ecs_entity_t EcsPebi;   /**< Pebi unit prefix. */
FLECS_API extern ecs_entity_t EcsExbi;   /**< Exbi unit prefix. */
FLECS_API extern ecs_entity_t EcsZebi;   /**< Zebi unit prefix. */
FLECS_API extern ecs_entity_t EcsYobi;   /**< Yobi unit prefix. */

/** @} */

/**
 * @defgroup c_addons_units_duration Duration
 * @ingroup c_addons_units
 * @{
 */

FLECS_API extern ecs_entity_t EcsDuration;         /**< Duration quantity. */
FLECS_API extern     ecs_entity_t EcsPicoSeconds;  /**< PicoSeconds duration unit. */
FLECS_API extern     ecs_entity_t EcsNanoSeconds;  /**< NanoSeconds duration unit. */
FLECS_API extern     ecs_entity_t EcsMicroSeconds; /**< MicroSeconds duration unit. */
FLECS_API extern     ecs_entity_t EcsMilliSeconds; /**< MilliSeconds duration unit. */
FLECS_API extern     ecs_entity_t EcsSeconds;      /**< Seconds duration unit. */
FLECS_API extern     ecs_entity_t EcsMinutes;      /**< Minutes duration unit. */
FLECS_API extern     ecs_entity_t EcsHours;        /**< Hours duration unit. */
FLECS_API extern     ecs_entity_t EcsDays;         /**< Days duration unit. */

/** @} */

/**
 * @defgroup c_addons_units_time Time
 * @ingroup c_addons_units
 * @{
 */

FLECS_API extern ecs_entity_t EcsTime;             /**< Time quantity. */
FLECS_API extern     ecs_entity_t EcsDate;         /**< Date unit. */

/** @} */

/**
 * @defgroup c_addons_units_mass Mass
 * @ingroup c_addons_units
 * @{
 */

FLECS_API extern ecs_entity_t EcsMass;             /**< Mass quantity. */
FLECS_API extern     ecs_entity_t EcsGrams;        /**< Grams unit. */
FLECS_API extern     ecs_entity_t EcsKiloGrams;    /**< KiloGrams unit. */

/** @} */

/**
 * @defgroup c_addons_units_electric_Current Electric Current
 * @ingroup c_addons_units
 * @{
 */

FLECS_API extern ecs_entity_t EcsElectricCurrent;  /**< ElectricCurrent quantity. */
FLECS_API extern     ecs_entity_t EcsAmpere;       /**< Ampere unit. */

/** @} */

/**
 * @defgroup c_addons_units_amount Amount
 * @ingroup c_addons_units
 * @{
 */

FLECS_API extern ecs_entity_t EcsAmount;           /**< Amount quantity. */
FLECS_API extern     ecs_entity_t EcsMole;         /**< Mole unit. */

/** @} */

/**
 * @defgroup c_addons_units_luminous_intensity Luminous Intensity
 * @ingroup c_addons_units
 * @{
 */

FLECS_API extern ecs_entity_t EcsLuminousIntensity; /**< LuminousIntensity quantity. */
FLECS_API extern     ecs_entity_t EcsCandela;       /**< Candela unit. */

/** @} */

/**
 * @defgroup c_addons_units_force Force
 * @ingroup c_addons_units
 * @{
 */

FLECS_API extern ecs_entity_t EcsForce;            /**< Force quantity. */
FLECS_API extern     ecs_entity_t EcsNewton;       /**< Newton unit. */

/** @} */

/**
 * @defgroup c_addons_units_length Length
 * @ingroup c_addons_units
 * @{
 */

FLECS_API extern ecs_entity_t EcsLength;              /**< Length quantity. */
FLECS_API extern     ecs_entity_t EcsMeters;          /**< Meters unit. */
FLECS_API extern         ecs_entity_t EcsPicoMeters;  /**< PicoMeters unit. */
FLECS_API extern         ecs_entity_t EcsNanoMeters;  /**< NanoMeters unit. */
FLECS_API extern         ecs_entity_t EcsMicroMeters; /**< MicroMeters unit. */
FLECS_API extern         ecs_entity_t EcsMilliMeters; /**< MilliMeters unit. */
FLECS_API extern         ecs_entity_t EcsCentiMeters; /**< CentiMeters unit. */
FLECS_API extern         ecs_entity_t EcsKiloMeters;  /**< KiloMeters unit. */
FLECS_API extern     ecs_entity_t EcsMiles;           /**< Miles unit. */
FLECS_API extern     ecs_entity_t EcsPixels;          /**< Pixels unit. */

/** @} */

/**
 * @defgroup c_addons_units_pressure Pressure
 * @ingroup c_addons_units
 * @{
 */

FLECS_API extern ecs_entity_t EcsPressure;          /**< Pressure quantity. */
FLECS_API extern     ecs_entity_t EcsPascal;        /**< Pascal unit. */
FLECS_API extern     ecs_entity_t EcsBar;           /**< Bar unit. */

/** @} */

/**
 * @defgroup c_addons_units_speed Speed
 * @ingroup c_addons_units
 * @{
 */

FLECS_API extern ecs_entity_t EcsSpeed;                   /**< Speed quantity. */
FLECS_API extern     ecs_entity_t EcsMetersPerSecond;     /**< MetersPerSecond unit. */
FLECS_API extern     ecs_entity_t EcsKiloMetersPerSecond; /**< KiloMetersPerSecond unit. */
FLECS_API extern     ecs_entity_t EcsKiloMetersPerHour;   /**< KiloMetersPerHour unit. */
FLECS_API extern     ecs_entity_t EcsMilesPerHour;        /**< MilesPerHour unit. */

/** @} */

/**
 * @defgroup c_addons_units_temperature Temperature
 * @ingroup c_addons_units
 * @{
 */

FLECS_API extern ecs_entity_t EcsTemperature;       /**< Temperature quantity. */
FLECS_API extern     ecs_entity_t EcsKelvin;        /**< Kelvin unit. */
FLECS_API extern     ecs_entity_t EcsCelsius;       /**< Celsius unit. */
FLECS_API extern     ecs_entity_t EcsFahrenheit;    /**< Fahrenheit unit. */

/** @} */

/**
 * @defgroup c_addons_units_data Data
 * @ingroup c_addons_units
 * @{
 */

FLECS_API extern ecs_entity_t EcsData;               /**< Data quantity. */
FLECS_API extern     ecs_entity_t EcsBits;           /**< Bits unit. */
FLECS_API extern         ecs_entity_t EcsKiloBits;   /**< KiloBits unit. */
FLECS_API extern         ecs_entity_t EcsMegaBits;   /**< MegaBits unit. */
FLECS_API extern         ecs_entity_t EcsGigaBits;   /**< GigaBits unit. */
FLECS_API extern     ecs_entity_t EcsBytes;          /**< Bytes unit. */
FLECS_API extern         ecs_entity_t EcsKiloBytes;  /**< KiloBytes unit. */
FLECS_API extern         ecs_entity_t EcsMegaBytes;  /**< MegaBytes unit. */
FLECS_API extern         ecs_entity_t EcsGigaBytes;  /**< GigaBytes unit. */
FLECS_API extern         ecs_entity_t EcsKibiBytes;  /**< KibiBytes unit. */
FLECS_API extern         ecs_entity_t EcsMebiBytes;  /**< MebiBytes unit. */
FLECS_API extern         ecs_entity_t EcsGibiBytes;  /**< GibiBytes unit. */

/** @} */

/**
 * @defgroup c_addons_units_datarate Data Rate
 * @ingroup c_addons_units
 * @{
 */

FLECS_API extern ecs_entity_t EcsDataRate;               /**< DataRate quantity. */
FLECS_API extern     ecs_entity_t EcsBitsPerSecond;      /**< BitsPerSecond unit. */
FLECS_API extern     ecs_entity_t EcsKiloBitsPerSecond;  /**< KiloBitsPerSecond unit. */
FLECS_API extern     ecs_entity_t EcsMegaBitsPerSecond;  /**< MegaBitsPerSecond unit. */
FLECS_API extern     ecs_entity_t EcsGigaBitsPerSecond;  /**< GigaBitsPerSecond unit. */
FLECS_API extern     ecs_entity_t EcsBytesPerSecond;     /**< BytesPerSecond unit. */
FLECS_API extern     ecs_entity_t EcsKiloBytesPerSecond; /**< KiloBytesPerSecond unit. */
FLECS_API extern     ecs_entity_t EcsMegaBytesPerSecond; /**< MegaBytesPerSecond unit. */
FLECS_API extern     ecs_entity_t EcsGigaBytesPerSecond; /**< GigaBytesPerSecond unit. */

/** @} */

/**
 * @defgroup c_addons_units_duration Duration
 * @ingroup c_addons_units
 * @{
 */

FLECS_API extern ecs_entity_t EcsAngle;            /**< Angle quantity. */
FLECS_API extern     ecs_entity_t EcsRadians;      /**< Radians unit. */
FLECS_API extern     ecs_entity_t EcsDegrees;      /**< Degrees unit. */

/** @} */

/**
 * @defgroup c_addons_units_angle Angle
 * @ingroup c_addons_units
 * @{
 */

FLECS_API extern ecs_entity_t EcsFrequency;        /**< Frequency quantity. */
FLECS_API extern     ecs_entity_t EcsHertz;        /**< Hertz unit. */
FLECS_API extern     ecs_entity_t EcsKiloHertz;    /**< KiloHertz unit. */
FLECS_API extern     ecs_entity_t EcsMegaHertz;    /**< MegaHertz unit. */
FLECS_API extern     ecs_entity_t EcsGigaHertz;    /**< GigaHertz unit. */

/** @} */

/**
 * @defgroup c_addons_units_uri Uri
 * @ingroup c_addons_units
 * @{
 */

FLECS_API extern ecs_entity_t EcsUri;              /**< URI quantity. */
FLECS_API extern     ecs_entity_t EcsUriHyperlink; /**< UriHyperlink unit. */
FLECS_API extern     ecs_entity_t EcsUriImage;     /**< UriImage unit. */
FLECS_API extern     ecs_entity_t EcsUriFile;      /**< UriFile unit. */

/** @} */

/**
 * @defgroup c_addons_units_color Color
 * @ingroup c_addons_units
 * @{
 */

FLECS_API extern ecs_entity_t EcsColor;            /**< Color quantity. */
FLECS_API extern     ecs_entity_t EcsColorRgb;     /**< ColorRgb unit. */
FLECS_API extern     ecs_entity_t EcsColorHsl;     /**< ColorHsl unit. */
FLECS_API extern     ecs_entity_t EcsColorCss;     /**< ColorCss unit. */

/** @} */


FLECS_API extern ecs_entity_t EcsAcceleration;     /**< Acceleration unit. */
FLECS_API extern ecs_entity_t EcsPercentage;       /**< Percentage unit. */
FLECS_API extern ecs_entity_t EcsBel;              /**< Bel unit. */
FLECS_API extern ecs_entity_t EcsDeciBel;          /**< DeciBel unit. */

////////////////////////////////////////////////////////////////////////////////
//// Module
////////////////////////////////////////////////////////////////////////////////

/** Units module import function.
 * Usage:
 * @code
 * ECS_IMPORT(world, FlecsUnits)
 * @endcode
 * 
 * @param world The world.
 */
FLECS_API
void FlecsUnitsImport(
    ecs_world_t *world);

#ifdef __cplusplus
}
#endif

#endif

/** @} */

#endif

#endif

#ifdef FLECS_SCRIPT_MATH
#ifdef FLECS_NO_SCRIPT_MATH
#error "FLECS_NO_SCRIPT_MATH failed: SCRIPT_MATH is required by other addons"
#endif
/**
 * @file addons/script_math.h
 * @brief Math functions for flecs script.
 */

#ifdef FLECS_SCRIPT_MATH

#ifndef FLECS_SCRIPT
#define FLECS_SCRIPT
#endif

/**
 * @defgroup c_addons_script_math Script Math
 * @ingroup c_addons
 * Math functions for flecs script.
 * @{
 */

#ifndef FLECS_SCRIPT_MATH_H
#define FLECS_SCRIPT_MATH_H

#ifdef __cplusplus
extern "C" {
#endif

FLECS_API
extern ECS_COMPONENT_DECLARE(EcsScriptRng);

/* Randon number generator */
typedef struct {
    uint64_t seed;
    void *impl;
} EcsScriptRng;

/** Script math import function.
 * Usage:
 * @code
 * ECS_IMPORT(world, FlecsScriptMath)
 * @endcode
 * 
 * @param world The world.
 */
FLECS_API
void FlecsScriptMathImport(
    ecs_world_t *world);

#ifdef __cplusplus
}
#endif

#endif

/** @} */

#endif

#endif

#ifdef FLECS_SCRIPT
#ifdef FLECS_NO_SCRIPT
#error "FLECS_NO_SCRIPT failed: SCRIPT is required by other addons"
#endif
/**
 * @file addons/script.h
 * @brief Flecs script module.
 *
 * For script, see examples/script.
 */

#ifdef FLECS_SCRIPT

/**
 * @defgroup c_addons_script Flecs script
 * @ingroup c_addons
 * DSL for loading scenes, assets and configuration.
 *
 * @{
 */

#ifndef FLECS_META
#define FLECS_META
#endif

#ifndef FLECS_DOC
#define FLECS_DOC
#endif


#ifndef FLECS_SCRIPT_H
#define FLECS_SCRIPT_H

#ifdef __cplusplus
extern "C" {
#endif

#define FLECS_SCRIPT_FUNCTION_ARGS_MAX (16)

FLECS_API
extern ECS_COMPONENT_DECLARE(EcsScript);

FLECS_API
extern ECS_DECLARE(EcsScriptTemplate);

FLECS_API
extern ECS_COMPONENT_DECLARE(EcsScriptConstVar);

FLECS_API
extern ECS_COMPONENT_DECLARE(EcsScriptFunction);

FLECS_API
extern ECS_COMPONENT_DECLARE(EcsScriptMethod);

/* Script template. */
typedef struct ecs_script_template_t ecs_script_template_t;

/** Script variable. */
typedef struct ecs_script_var_t {
    const char *name;
    ecs_value_t value;
    const ecs_type_info_t *type_info;
    int32_t sp;
    bool is_const;
} ecs_script_var_t;

/** Script variable scope. */
typedef struct ecs_script_vars_t {
    struct ecs_script_vars_t *parent;
    int32_t sp;

    ecs_hashmap_t var_index;
    ecs_vec_t vars;

    const ecs_world_t *world;
    struct ecs_stack_t *stack;
    ecs_stack_cursor_t *cursor;
    ecs_allocator_t *allocator;
} ecs_script_vars_t;

/** Script object. */
typedef struct ecs_script_t {
    ecs_world_t *world;
    const char *name;
    const char *code;
} ecs_script_t;

/* Runtime for executing scripts */
typedef struct ecs_script_runtime_t ecs_script_runtime_t;

/** Script component. 
 * This component is added to the entities of managed scripts and templates.
 */
typedef struct EcsScript {
    ecs_script_t *script;
    ecs_script_template_t *template_; /* Only set for template scripts */
} EcsScript;

/** Script function context. */
typedef struct ecs_function_ctx_t {
    ecs_world_t *world;
    ecs_entity_t function;
    void *ctx;
} ecs_function_ctx_t;

/** Script function callback. */
typedef void(*ecs_function_callback_t)(
    const ecs_function_ctx_t *ctx,
    int32_t argc,
    const ecs_value_t *argv,
    ecs_value_t *result);

/** Function argument type. */
typedef struct ecs_script_parameter_t {
    const char *name;
    ecs_entity_t type;
} ecs_script_parameter_t;

/** Const component.
 * This component describes a const variable that can be used from scripts.
 */
typedef struct EcsScriptConstVar {
    ecs_value_t value;
    const ecs_type_info_t *type_info;
} EcsScriptConstVar;

/** Function component.
 * This component describes a function that can be called from a script.
 */
typedef struct EcsScriptFunction {
    ecs_entity_t return_type;
    ecs_vec_t params; /* vec<ecs_script_parameter_t> */
    ecs_function_callback_t callback;
    void *ctx;
} EcsScriptFunction;

/** Method component. 
 * This component describes a method that can be called from a script. Methods
 * are functions that can be called on instances of a type. A method entity is
 * stored in the scope of the type it belongs to.
 */
typedef struct EcsScriptMethod {
    ecs_entity_t return_type;
    ecs_vec_t params; /* vec<ecs_script_parameter_t> */
    ecs_function_callback_t callback;
    void *ctx;
} EcsScriptMethod;

/* Parsing & running scripts */

/** Used with ecs_script_parse() and ecs_script_eval() */
typedef struct ecs_script_eval_desc_t {
    ecs_script_vars_t *vars;       /**< Variables used by script */
    ecs_script_runtime_t *runtime; /**< Reusable runtime (optional) */
} ecs_script_eval_desc_t;

/** Parse script.
 * This operation parses a script and returns a script object upon success. To
 * run the script, call ecs_script_eval().
 * 
 * If the script uses outside variables, an ecs_script_vars_t object must be
 * provided in the vars member of the desc object that defines all variables 
 * with the correct types.
 * 
 * @param world The world.
 * @param name Name of the script (typically a file/module name).
 * @param code The script code.
 * @param desc Parameters for script runtime.
 * @return Script object if success, NULL if failed.
*/
FLECS_API
ecs_script_t* ecs_script_parse(
    ecs_world_t *world,
    const char *name,
    const char *code,
    const ecs_script_eval_desc_t *desc);

/** Evaluate script.
 * This operation evaluates (runs) a parsed script.
 * 
 * If variables were provided to ecs_script_parse(), an application may pass
 * a different ecs_script_vars_t object to ecs_script_eval(), as long as the
 * object has all referenced variables and they are of the same type.
 * 
 * @param script The script.
 * @param desc Parameters for script runtime.
 * @return Zero if success, non-zero if failed.
*/
FLECS_API
int ecs_script_eval(
    const ecs_script_t *script,
    const ecs_script_eval_desc_t *desc);

/** Free script.
 * This operation frees a script object.
 * 
 * Templates created by the script rely upon resources in the script object,
 * and for that reason keep the script alive until all templates created by the
 * script are deleted.
 *
 * @param script The script.
 */
FLECS_API
void ecs_script_free(
    ecs_script_t *script);

/** Parse script.
 * This parses a script and instantiates the entities in the world.
 * This operation is the equivalent to doing:
 * 
 * @code
 * ecs_script_t *script = ecs_script_parse(world, name, code);
 * ecs_script_eval(script);
 * ecs_script_free(script);
 * @endcode
 * 
 * @param world The world.
 * @param name The script name (typically the file).
 * @param code The script.
 * @return Zero if success, non-zero otherwise.
 */
FLECS_API
int ecs_script_run(
    ecs_world_t *world,
    const char *name,
    const char *code);

/** Parse script file.
 * This parses a script file and instantiates the entities in the world. This
 * operation is equivalent to loading the file contents and passing it to
 * ecs_script_run().
 *
 * @param world The world.
 * @param filename The script file name.
 * @return Zero if success, non-zero if failed.
 */
FLECS_API
int ecs_script_run_file(
    ecs_world_t *world,
    const char *filename);

/** Create runtime for script.
 * A script runtime is a container for any data created during script 
 * evaluation. By default calling ecs_script_run() or ecs_script_eval() will
 * create a runtime on the spot. A runtime can be created in advance and reused
 * across multiple script evaluations to improve performance.
 * 
 * When scripts are evaluated on multiple threads, each thread should have its
 * own script runtime.
 * 
 * A script runtime must be deleted with ecs_script_runtime_free().
 * 
 * @return A new script runtime.
 */
FLECS_API
ecs_script_runtime_t* ecs_script_runtime_new(void);

/** Free script runtime.
 * This operation frees a script runtime created by ecs_script_runtime_new().
 * 
 * @param runtime The runtime to free.
 */
FLECS_API
void ecs_script_runtime_free(
    ecs_script_runtime_t *runtime);

/** Convert script AST to string.
 * This operation converts the script abstract syntax tree to a string, which
 * can be used to debug a script.
 * 
 * @param script The script.
 * @param buf The buffer to write to.
 * @return Zero if success, non-zero if failed.
 */
FLECS_API
int ecs_script_ast_to_buf(
    ecs_script_t *script,
    ecs_strbuf_t *buf,
    bool colors);

/** Convert script AST to string.
 * This operation converts the script abstract syntax tree to a string, which
 * can be used to debug a script.
 * 
 * @param script The script.
 * @return The string if success, NULL if failed.
 */
FLECS_API
char* ecs_script_ast_to_str(
    ecs_script_t *script,
    bool colors);


/* Managed scripts (script associated with entity that outlives the function) */

/** Used with ecs_script_init() */
typedef struct ecs_script_desc_t {
    ecs_entity_t entity;   /* Set to customize entity handle associated with script */
    const char *filename;  /* Set to load script from file */
    const char *code;      /* Set to parse script from string */
} ecs_script_desc_t;

/** Load managed script.
 * A managed script tracks which entities it creates, and keeps those entities
 * synchronized when the contents of the script are updated. When the script is
 * updated, entities that are no longer in the new version will be deleted.
 *
 * This feature is experimental.
 *
 * @param world The world.
 * @param desc Script descriptor.
 */
FLECS_API
ecs_entity_t ecs_script_init(
    ecs_world_t *world,
    const ecs_script_desc_t *desc);

#define ecs_script(world, ...)\
    ecs_script_init(world, &(ecs_script_desc_t) __VA_ARGS__)

/** Update script with new code.
 *
 * @param world The world.
 * @param script The script entity.
 * @param instance An template instance (optional).
 * @param code The script code.
 */
FLECS_API
int ecs_script_update(
    ecs_world_t *world,
    ecs_entity_t script,
    ecs_entity_t instance,
    const char *code);

/** Clear all entities associated with script.
 *
 * @param world The world.
 * @param script The script entity.
 * @param instance The script instance.
 */
FLECS_API
void ecs_script_clear(
    ecs_world_t *world,
    ecs_entity_t script,
    ecs_entity_t instance);


/* Script variables */

/** Create new variable scope.
 * Create root variable scope. A variable scope contains one or more variables. 
 * Scopes can be nested, which allows variables in different scopes to have the 
 * same name. Variables from parent scopes will be shadowed by variables in 
 * child scopes with the same name.
 * 
 * Use the `ecs_script_vars_push()` and `ecs_script_vars_pop()` functions to
 * push and pop variable scopes.
 * 
 * When a variable contains allocated resources (e.g. a string), its resources
 * will be freed when `ecs_script_vars_pop()` is called on the scope, the
 * ecs_script_vars_t::type_info field is initialized for the variable, and 
 * `ecs_type_info_t::hooks::dtor` is set.
 * 
 * @param world The world.
 */
FLECS_API
ecs_script_vars_t* ecs_script_vars_init(
    ecs_world_t *world);

/** Free variable scope.
 * Free root variable scope. The provided scope should not have a parent. This
 * operation calls `ecs_script_vars_pop()` on the scope.
 *
 * @param vars The variable scope.
 */
FLECS_API
void ecs_script_vars_fini(
    ecs_script_vars_t *vars);

/** Push new variable scope.
 * 
 * Scopes created with ecs_script_vars_push() must be cleaned up with
 * ecs_script_vars_pop().
 * 
 * If the stack and allocator arguments are left to NULL, their values will be
 * copied from the parent.
 *
 * @param parent The parent scope (provide NULL for root scope).
 * @return The new variable scope.
 */
FLECS_API
ecs_script_vars_t* ecs_script_vars_push(
    ecs_script_vars_t *parent);

/** Pop variable scope.
 * This frees up the resources for a variable scope. The scope must be at the
 * top of a vars stack. Calling ecs_script_vars_pop() on a scope that is not the
 * last scope causes undefined behavior.
 *
 * @param vars The scope to free.
 * @return The parent scope.
 */
FLECS_API
ecs_script_vars_t* ecs_script_vars_pop(
    ecs_script_vars_t *vars);

/** Declare a variable.
 * This operation declares a new variable in the current scope. If a variable
 * with the specified name already exists, the operation will fail.
 * 
 * This operation does not allocate storage for the variable. This is done to
 * allow for variables that point to existing storage, which prevents having
 * to copy existing values to a variable scope.
 * 
 * @param vars The variable scope.
 * @param name The variable name.
 * @return The new variable, or NULL if the operation failed.
 */
FLECS_API
ecs_script_var_t* ecs_script_vars_declare(
    ecs_script_vars_t *vars,
    const char *name);

/** Define a variable.
 * This operation calls `ecs_script_vars_declare()` and allocates storage for
 * the variable. If the type has a ctor, it will be called on the new storage.
 * 
 * The scope's stack allocator will be used to allocate the storage. After 
 * `ecs_script_vars_pop()` is called on the scope, the variable storage will no
 * longer be valid.
 * 
 * The operation will fail if the type argument is not a type.
 * 
 * @param vars The variable scope.
 * @param name The variable name.
 * @param type The variable type.
 * @return The new variable, or NULL if the operation failed.
 */
FLECS_API
ecs_script_var_t* ecs_script_vars_define_id(
    ecs_script_vars_t *vars,
    const char *name,
    ecs_entity_t type);

#define ecs_script_vars_define(vars, name, type)\
    ecs_script_vars_define_id(vars, name, ecs_id(type))

/** Lookup a variable.
 * This operation looks up a variable in the current scope. If the variable 
 * can't be found in the current scope, the operation will recursively search
 * the parent scopes.
 * 
 * @param vars The variable scope.
 * @param name The variable name.
 * @return The variable, or NULL if one with the provided name does not exist.
 */
FLECS_API
ecs_script_var_t* ecs_script_vars_lookup(
    const ecs_script_vars_t *vars,
    const char *name);

/** Lookup a variable by stack pointer.
 * This operation provides a faster way to lookup variables that are always 
 * declared in the same order in a ecs_script_vars_t scope.
 * 
 * The stack pointer of a variable can be obtained from the ecs_script_var_t 
 * type. The provided frame offset must be valid for the provided variable  
 * stack. If the frame offset is not valid, this operation will panic.
 * 
 * @param vars The variable scope.
 * @param sp The stack pointer to the variable.
 * @return The variable.
 */
FLECS_API
ecs_script_var_t* ecs_script_vars_from_sp(
    const ecs_script_vars_t *vars,
    int32_t sp);

/** Print variables.
 * This operation prints all variables in the vars scope and parent scopes.asm
 * 
 * @param vars The variable scope.
 */
FLECS_API
void ecs_script_vars_print(
    const ecs_script_vars_t *vars);

/** Preallocate space for variables.
 * This operation preallocates space for the specified number of variables. This
 * is a performance optimization only, and is not necessary before declaring
 * variables in a scope.
 * 
 * @param vars The variable scope.
 * @param count The number of variables to preallocate space for.
 */
FLECS_API
void ecs_script_vars_set_size(
    ecs_script_vars_t *vars,
    int32_t count);

/** Convert iterator to vars
 * This operation converts an iterator to a variable array. This allows for
 * using iterator results in expressions. The operation only converts a
 * single result at a time, and does not progress the iterator.
 *
 * Iterator fields with data will be made available as variables with as name
 * the field index (e.g. "$1"). The operation does not check if reflection data
 * is registered for a field type. If no reflection data is registered for the
 * type, using the field variable in expressions will fail.
 *
 * Field variables will only contain single elements, even if the iterator
 * returns component arrays. The offset parameter can be used to specify which
 * element in the component arrays to return. The offset parameter must be
 * smaller than it->count.
 *
 * The operation will create a variable for query variables that contain a
 * single entity.
 *
 * The operation will attempt to use existing variables. If a variable does not
 * yet exist, the operation will create it. If an existing variable exists with
 * a mismatching type, the operation will fail.
 *
 * Accessing variables after progressing the iterator or after the iterator is
 * destroyed will result in undefined behavior.
 *
 * If vars contains a variable that is not present in the iterator, the variable
 * will not be modified.
 *
 * @param it The iterator to convert to variables.
 * @param vars The variables to write to.
 * @param offset The offset to the current element.
 */
FLECS_API
void ecs_script_vars_from_iter(
    const ecs_iter_t *it,
    ecs_script_vars_t *vars,
    int offset);


/* Standalone expression evaluation */

/** Used with ecs_expr_run(). */
typedef struct ecs_expr_eval_desc_t {
    const char *name;                /**< Script name */
    const char *expr;                /**< Full expression string */
    const ecs_script_vars_t *vars;   /**< Variables accessible in expression */
    ecs_entity_t type;               /**< Type of parsed value (optional) */
    ecs_entity_t (*lookup_action)(   /**< Function for resolving entity identifiers */
        const ecs_world_t*,
        const char *value,
        void *ctx);
    void *lookup_ctx;                /**< Context passed to lookup function */

    /** Disable constant folding (slower evaluation, faster parsing) */
    bool disable_folding;

    /** This option instructs the expression runtime to lookup variables by 
     * stack pointer instead of by name, which improves performance. Only enable 
     * when provided variables are always declared in the same order. */
    bool disable_dynamic_variable_binding;

    /** Allow for unresolved identifiers when parsing. Useful when entities can
     * be created in between parsing & evaluating. */
    bool allow_unresolved_identifiers;

    ecs_script_runtime_t *runtime;   /**< Reusable runtime (optional) */
} ecs_expr_eval_desc_t;

/** Run expression.
 * This operation runs an expression and stores the result in the provided 
 * value. If the value contains a type that is different from the type of the
 * expression, the expression will be cast to the value.
 *
 * If the provided value for value.ptr is NULL, the value must be freed with 
 * ecs_value_free() afterwards.
 *
 * @param world The world.
 * @param ptr The pointer to the expression to parse.
 * @param value The value containing type & pointer to write to.
 * @param desc Configuration parameters for the parser.
 * @return Pointer to the character after the last one read, or NULL if failed.
 */
FLECS_API
const char* ecs_expr_run(
    ecs_world_t *world,
    const char *ptr,
    ecs_value_t *value,
    const ecs_expr_eval_desc_t *desc);

/** Parse expression.
 * This operation parses an expression and returns an object that can be 
 * evaluated multiple times with ecs_expr_eval().
 * 
 * @param world The world.
 * @param expr The expression string.
 * @param desc Configuration parameters for the parser.
 * @return A script object if parsing is successful, NULL if parsing failed.
 */
FLECS_API
ecs_script_t* ecs_expr_parse(
    ecs_world_t *world,
    const char *expr,
    const ecs_expr_eval_desc_t *desc);

/** Evaluate expression.
 * This operation evaluates an expression parsed with ecs_expr_parse() 
 * and stores the result in the provided value. If the value contains a type 
 * that is different from the type of the expression, the expression will be 
 * cast to the value.
 * 
 * If the provided value for value.ptr is NULL, the value must be freed with 
 * ecs_value_free() afterwards.
 * 
 * @param script The script containing the expression.
 * @param value The value in which to store the expression result.
 * @param desc Configuration parameters for the parser.
 * @return Zero if successful, non-zero if failed.
 */
FLECS_API
int ecs_expr_eval(
    const ecs_script_t *script,
    ecs_value_t *value,
    const ecs_expr_eval_desc_t *desc);

/** Evaluate interpolated expressions in string.
 * This operation evaluates expressions in a string, and replaces them with
 * their evaluated result. Supported expression formats are:
 *  - $variable_name
 *  - {expression}
 *
 * The $, { and } characters can be escaped with a backslash (\).
 *
 * @param world The world.
 * @param str The string to evaluate.
 * @param vars The variables to use for evaluation.
 */
FLECS_API
char* ecs_script_string_interpolate(
    ecs_world_t *world,
    const char *str,
    const ecs_script_vars_t *vars);


/* Global const variables */

/** Used with ecs_const_var_init */
typedef struct ecs_const_var_desc_t {
    /* Variable name. */
    const char *name;

    /* Variable parent (namespace). */
    ecs_entity_t parent;

    /* Variable type. */
    ecs_entity_t type;

    /* Pointer to value of variable. The value will be copied to an internal
     * storage and does not need to be kept alive. */
    void *value;
} ecs_const_var_desc_t;

/** Create a const variable that can be accessed by scripts. 
 * 
 * @param world The world.
 * @param desc Const var parameters.
 * @return The const var, or 0 if failed.
 */
FLECS_API
ecs_entity_t ecs_const_var_init(
    ecs_world_t *world,
    ecs_const_var_desc_t *desc);

#define ecs_const_var(world, ...)\
    ecs_const_var_init(world, &(ecs_const_var_desc_t)__VA_ARGS__)

/* Functions */

/** Used with ecs_function_init and ecs_method_init */
typedef struct ecs_function_desc_t {
    /** Function name. */
    const char *name;
    
    /** Parent of function. For methods the parent is the type for which the 
     * method will be registered. */
    ecs_entity_t parent;

    /** Function parameters. */
    ecs_script_parameter_t params[FLECS_SCRIPT_FUNCTION_ARGS_MAX];

    /** Function return type. */
    ecs_entity_t return_type;

    /** Function implementation. */
    ecs_function_callback_t callback;

    /** Context passed to function implementation. */
    void *ctx;
} ecs_function_desc_t;

/** Create new function. 
 * This operation creates a new function that can be called from a script.
 * 
 * @param world The world.
 * @param desc Function init parameters.
 * @return The function, or 0 if failed.
*/
FLECS_API
ecs_entity_t ecs_function_init(
    ecs_world_t *world,
    const ecs_function_desc_t *desc);

#define ecs_function(world, ...)\
    ecs_function_init(world, &(ecs_function_desc_t)__VA_ARGS__)

/** Create new method. 
 * This operation creates a new method that can be called from a script. A 
 * method is like a function, except that it can be called on every instance of
 * a type.
 * 
 * Methods automatically receive the instance on which the method is invoked as
 * first argument.
 * 
 * @param world Method The world.
 * @param desc Method init parameters.
 * @return The function, or 0 if failed.
*/
FLECS_API
ecs_entity_t ecs_method_init(
    ecs_world_t *world,
    const ecs_function_desc_t *desc);

#define ecs_method(world, ...)\
    ecs_method_init(world, &(ecs_function_desc_t)__VA_ARGS__)


/* Value serialization */

/** Serialize value into expression string.
 * This operation serializes a value of the provided type to a string. The
 * memory pointed to must be large enough to contain a value of the used type.
 *
 * @param world The world.
 * @param type The type of the value to serialize.
 * @param data The value to serialize.
 * @return String with expression, or NULL if failed.
 */
FLECS_API
char* ecs_ptr_to_expr(
    const ecs_world_t *world,
    ecs_entity_t type,
    const void *data);

/** Serialize value into expression buffer.
 * Same as ecs_ptr_to_expr(), but serializes to an ecs_strbuf_t instance.
 *
 * @param world The world.
 * @param type The type of the value to serialize.
 * @param data The value to serialize.
 * @param buf The strbuf to append the string to.
 * @return Zero if success, non-zero if failed.
 */
FLECS_API
int ecs_ptr_to_expr_buf(
    const ecs_world_t *world,
    ecs_entity_t type,
    const void *data,
    ecs_strbuf_t *buf);

/** Similar as ecs_ptr_to_expr(), but serializes values to string.
 * Whereas the output of ecs_ptr_to_expr() is a valid expression, the output of
 * ecs_ptr_to_str() is a string representation of the value. In most cases the
 * output of the two operations is the same, but there are some differences:
 * - Strings are not quoted
 *
 * @param world The world.
 * @param type The type of the value to serialize.
 * @param data The value to serialize.
 * @return String with result, or NULL if failed.
 */
FLECS_API
char* ecs_ptr_to_str(
    const ecs_world_t *world,
    ecs_entity_t type,
    const void *data);

/** Serialize value into string buffer.
 * Same as ecs_ptr_to_str(), but serializes to an ecs_strbuf_t instance.
 *
 * @param world The world.
 * @param type The type of the value to serialize.
 * @param data The value to serialize.
 * @param buf The strbuf to append the string to.
 * @return Zero if success, non-zero if failed.
 */
FLECS_API
int ecs_ptr_to_str_buf(
    const ecs_world_t *world,
    ecs_entity_t type,
    const void *data,
    ecs_strbuf_t *buf);

typedef struct ecs_expr_node_t ecs_expr_node_t; 

/** Script module import function.
 * Usage:
 * @code
 * ECS_IMPORT(world, FlecsScript)
 * @endcode
 * 
 * @param world The world.
 */
FLECS_API
void FlecsScriptImport(
    ecs_world_t *world);

#ifdef __cplusplus
}
#endif

#endif

/** @} */

#endif

#endif

#ifdef FLECS_DOC
#ifdef FLECS_NO_DOC
#error "FLECS_NO_DOC failed: DOC is required by other addons"
#endif
/**
 * @file addons/doc.h
 * @brief Doc module.
 *
 * The doc module allows for documenting entities (and thus components, systems)
 * by adding brief and/or detailed descriptions as components. Documentation
 * added with the doc module can be retrieved at runtime, and can be used by
 * tooling such as UIs or documentation frameworks.
 */

#ifdef FLECS_DOC

#ifndef FLECS_DOC_H
#define FLECS_DOC_H

#ifndef FLECS_MODULE
#define FLECS_MODULE
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup c_addons_doc Doc
 * @ingroup c_addons
 * Utilities for documenting entities, components and systems.
 *
 * @{
 */

FLECS_API extern const ecs_entity_t ecs_id(EcsDocDescription); /**< Component id for EcsDocDescription. */

/** Tag for adding a UUID to entities. 
 * Added to an entity as (EcsDocDescription, EcsUuid) by ecs_doc_set_uuid().
 */
FLECS_API extern const ecs_entity_t EcsDocUuid;

/** Tag for adding brief descriptions to entities. 
 * Added to an entity as (EcsDocDescription, EcsBrief) by ecs_doc_set_brief().
 */
FLECS_API extern const ecs_entity_t EcsDocBrief;

/** Tag for adding detailed descriptions to entities. 
 * Added to an entity as (EcsDocDescription, EcsDocDetail) by ecs_doc_set_detail().
 */
FLECS_API extern const ecs_entity_t EcsDocDetail;

/** Tag for adding a link to entities. 
 * Added to an entity as (EcsDocDescription, EcsDocLink) by ecs_doc_set_link().
 */
FLECS_API extern const ecs_entity_t EcsDocLink;

/** Tag for adding a color to entities. 
 * Added to an entity as (EcsDocDescription, EcsDocColor) by ecs_doc_set_link().
 */
FLECS_API extern const ecs_entity_t EcsDocColor;

/** Component that stores description.
 * Used as pair together with the following tags to store entity documentation:
 * - EcsName
 * - EcsDocBrief
 * - EcsDocDetail
 * - EcsDocLink
 * - EcsDocColor
 */
typedef struct EcsDocDescription {
    char *value;
} EcsDocDescription;

/** Add UUID to entity.
 * Associate entity with an (external) UUID.
 *
 * @param world The world.
 * @param entity The entity to which to add the UUID.
 * @param uuid The UUID to add.
 *
 * @see ecs_doc_get_uuid()
 * @see flecs::doc::set_uuid()
 * @see flecs::entity_builder::set_doc_uuid()
 */
FLECS_API
void ecs_doc_set_uuid(
    ecs_world_t *world,
    ecs_entity_t entity,
    const char *uuid);

/** Add human-readable name to entity.
 * Contrary to entity names, human readable names do not have to be unique and
 * can contain special characters used in the query language like '*'.
 *
 * @param world The world.
 * @param entity The entity to which to add the name.
 * @param name The name to add.
 *
 * @see ecs_doc_get_name()
 * @see flecs::doc::set_name()
 * @see flecs::entity_builder::set_doc_name()
 */
FLECS_API
void ecs_doc_set_name(
    ecs_world_t *world,
    ecs_entity_t entity,
    const char *name);

/** Add brief description to entity.
 *
 * @param world The world.
 * @param entity The entity to which to add the description.
 * @param description The description to add.
 *
 * @see ecs_doc_get_brief()
 * @see flecs::doc::set_brief()
 * @see flecs::entity_builder::set_doc_brief()
 */
FLECS_API
void ecs_doc_set_brief(
    ecs_world_t *world,
    ecs_entity_t entity,
    const char *description);

/** Add detailed description to entity.
 *
 * @param world The world.
 * @param entity The entity to which to add the description.
 * @param description The description to add.
 *
 * @see ecs_doc_get_detail()
 * @see flecs::doc::set_detail()
 * @see flecs::entity_builder::set_doc_detail()
 */
FLECS_API
void ecs_doc_set_detail(
    ecs_world_t *world,
    ecs_entity_t entity,
    const char *description);

/** Add link to external documentation to entity.
 *
 * @param world The world.
 * @param entity The entity to which to add the link.
 * @param link The link to add.
 *
 * @see ecs_doc_get_link()
 * @see flecs::doc::set_link()
 * @see flecs::entity_builder::set_doc_link()
 */
FLECS_API
void ecs_doc_set_link(
    ecs_world_t *world,
    ecs_entity_t entity,
    const char *link);

/** Add color to entity.
 * UIs can use color as hint to improve visualizing entities.
 *
 * @param world The world.
 * @param entity The entity to which to add the link.
 * @param color The color to add.
 *
 * @see ecs_doc_get_color()
 * @see flecs::doc::set_color()
 * @see flecs::entity_builder::set_doc_color()
 */
FLECS_API
void ecs_doc_set_color(
    ecs_world_t *world,
    ecs_entity_t entity,
    const char *color);

/** Get UUID from entity.
 * @param world The world.
 * @param entity The entity from which to get the UUID.
 * @return The UUID.
 *
 * @see ecs_doc_set_uuid()
 * @see flecs::doc::get_uuid()
 * @see flecs::entity_view::get_doc_uuid()
 */
FLECS_API
const char* ecs_doc_get_uuid(
    const ecs_world_t *world,
    ecs_entity_t entity);

/** Get human readable name from entity.
 * If entity does not have an explicit human readable name, this operation will
 * return the entity name.
 *
 * To test if an entity has a human readable name, use:
 *
 * @code
 * ecs_has_pair(world, e, ecs_id(EcsDocDescription), EcsName);
 * @endcode
 *
 * Or in C++:
 *
 * @code
 * e.has<flecs::doc::Description>(flecs::Name);
 * @endcode
 *
 * @param world The world.
 * @param entity The entity from which to get the name.
 * @return The name.
 *
 * @see ecs_doc_set_name()
 * @see flecs::doc::get_name()
 * @see flecs::entity_view::get_doc_name()
 */
FLECS_API
const char* ecs_doc_get_name(
    const ecs_world_t *world,
    ecs_entity_t entity);

/** Get brief description from entity.
 *
 * @param world The world.
 * @param entity The entity from which to get the description.
 * @return The description.
 *
 * @see ecs_doc_set_brief()
 * @see flecs::doc::get_brief()
 * @see flecs::entity_view::get_doc_brief()
 */
FLECS_API
const char* ecs_doc_get_brief(
    const ecs_world_t *world,
    ecs_entity_t entity);

/** Get detailed description from entity.
 *
 * @param world The world.
 * @param entity The entity from which to get the description.
 * @return The description.
 *
 * @see ecs_doc_set_detail()
 * @see flecs::doc::get_detail()
 * @see flecs::entity_view::get_doc_detail()
 */
FLECS_API
const char* ecs_doc_get_detail(
    const ecs_world_t *world,
    ecs_entity_t entity);

/** Get link to external documentation from entity.
 *
 * @param world The world.
 * @param entity The entity from which to get the link.
 * @return The link.
 *
 * @see ecs_doc_set_link()
 * @see flecs::doc::get_link()
 * @see flecs::entity_view::get_doc_link()
 */
FLECS_API
const char* ecs_doc_get_link(
    const ecs_world_t *world,
    ecs_entity_t entity);

/** Get color from entity.
 *
 * @param world The world.
 * @param entity The entity from which to get the color.
 * @return The color.
 *
 * @see ecs_doc_set_color()
 * @see flecs::doc::get_color()
 * @see flecs::entity_view::get_doc_color()
 */
FLECS_API
const char* ecs_doc_get_color(
    const ecs_world_t *world,
    ecs_entity_t entity);

/** Doc module import function.
 * Usage:
 * @code
 * ECS_IMPORT(world, FlecsDoc)
 * @endcode
 * 
 * @param world The world.
 */
FLECS_API
void FlecsDocImport(
    ecs_world_t *world);

/** @} */

#ifdef __cplusplus
}
#endif

#endif

#endif

#endif

#ifdef FLECS_META
#ifdef FLECS_NO_META
#error "FLECS_NO_META failed: META is required by other addons"
#endif
/**
 * @file addons/meta.h
 * @brief Meta addon.
 *
 * The meta addon enables reflecting on component data. Types are stored as
 * entities, with components that store the reflection data. A type has at least
 * two components:
 *
 * - EcsComponent: core component, contains size & alignment
 * - EcsType:  component that indicates what kind of type the entity is
 *
 * Additionally the type may have an additional component that contains the
 * reflection data for the type. For example, structs have these components:
 *
 * - EcsComponent
 * - EcsType
 * - EcsStruct
 *
 * Structs can be populated by adding child entities with the EcsMember
 * component. Adding a child with a Member component to an entity will
 * automatically add the EcsStruct component to the parent.
 *
 * Enums/bitmasks can be populated by adding child entities with the Constant
 * tag. By default constants are automatically assigned values when they are
 * added to the enum/bitmask. The parent entity must have the EcsEnum or
 * EcsBitmask component before adding the constants.
 *
 * To create enum constants with a manual value, set (Constant, i32) to the
 * desired value. To create bitmask constants with a manual value, set
 * (Constant, u32) to the desired value. Constants with manual values should not
 * conflict with other constants.
 *
 * The _init APIs are convenience wrappers around creating the entities and
 * components for the types.
 *
 * When a type is created it automatically receives the EcsComponent and
 * EcsType components. The former means that the resulting type can be
 * used as a regular component:
 *
 * @code
 * // Create Position type
 * ecs_entity_t pos = ecs_struct_init(world, &(ecs_struct_desc_t){
 *  .entity.name = "Position",
 *  .members = {
 *       {"x", ecs_id(ecs_f32_t)},
 *       {"y", ecs_id(ecs_f32_t)}
 *   }
 * });
 *
 * // Create entity with Position component
 * ecs_entity_t e = ecs_new_w_id(world, pos);
 * @endcode
 *
 * Type entities do not have to be named.
 */

#ifdef FLECS_META

/**
 * @defgroup c_addons_meta Meta
 * @ingroup c_addons
 * Flecs reflection framework.
 *
 * @{
 */

#include <stddef.h>

#ifndef FLECS_MODULE
#define FLECS_MODULE
#endif

#ifndef FLECS_META_H
#define FLECS_META_H

#ifdef __cplusplus
extern "C" {
#endif

/** Max number of constants/members that can be specified in desc structs. */
#define ECS_MEMBER_DESC_CACHE_SIZE (32)

/** Primitive type definitions.
 * These typedefs allow the builtin primitives to be used as regular components:
 *
 * @code
 * ecs_set(world, e, ecs_i32_t, {10});
 * @endcode
 *
 * Or a more useful example (create an enum constant with a manual value):
 *
 * @code
 * ecs_set_pair_second(world, e, EcsConstant, ecs_i32_t, {10});
 * @endcode
 */

typedef bool ecs_bool_t;                                        /**< Builtin bool type */
typedef char ecs_char_t;                                        /**< Builtin char type */
typedef unsigned char ecs_byte_t;                               /**< Builtin  ecs_byte type */
typedef uint8_t ecs_u8_t;                                       /**< Builtin u8 type */
typedef uint16_t ecs_u16_t;                                     /**< Builtin u16 type */
typedef uint32_t ecs_u32_t;                                     /**< Builtin u32 type */
typedef uint64_t ecs_u64_t;                                     /**< Builtin u64 type */
typedef uintptr_t ecs_uptr_t;                                   /**< Builtin uptr type */
typedef int8_t ecs_i8_t;                                        /**< Builtin i8 type */
typedef int16_t ecs_i16_t;                                      /**< Builtin i16 type */
typedef int32_t ecs_i32_t;                                      /**< Builtin i32 type */
typedef int64_t ecs_i64_t;                                      /**< Builtin i64 type */
typedef intptr_t ecs_iptr_t;                                    /**< Builtin iptr type */
typedef float ecs_f32_t;                                        /**< Builtin f32 type */
typedef double ecs_f64_t;                                       /**< Builtin f64 type */
typedef char* ecs_string_t;                                     /**< Builtin string type */

/* Meta module component ids */
FLECS_API extern const ecs_entity_t ecs_id(EcsType);            /**< Id for component added to all types with reflection data. */
FLECS_API extern const ecs_entity_t ecs_id(EcsTypeSerializer);  /**< Id for component that stores a type specific serializer. */
FLECS_API extern const ecs_entity_t ecs_id(EcsPrimitive);       /**< Id for component that stores reflection data for a primitive type. */
FLECS_API extern const ecs_entity_t ecs_id(EcsEnum);            /**< Id for component that stores reflection data for an enum type. */
FLECS_API extern const ecs_entity_t ecs_id(EcsBitmask);         /**< Id for component that stores reflection data for a bitmask type. */
FLECS_API extern const ecs_entity_t ecs_id(EcsMember);          /**< Id for component that stores reflection data for struct members. */
FLECS_API extern const ecs_entity_t ecs_id(EcsMemberRanges);    /**< Id for component that stores min/max ranges for member values. */
FLECS_API extern const ecs_entity_t ecs_id(EcsStruct);          /**< Id for component that stores reflection data for a struct type. */
FLECS_API extern const ecs_entity_t ecs_id(EcsArray);           /**< Id for component that stores reflection data for an array type. */
FLECS_API extern const ecs_entity_t ecs_id(EcsVector);          /**< Id for component that stores reflection data for a vector type. */
FLECS_API extern const ecs_entity_t ecs_id(EcsOpaque);          /**< Id for component that stores reflection data for an opaque type. */
FLECS_API extern const ecs_entity_t ecs_id(EcsUnit);            /**< Id for component that stores unit data. */
FLECS_API extern const ecs_entity_t ecs_id(EcsUnitPrefix);      /**< Id for component that stores unit prefix data. */
FLECS_API extern const ecs_entity_t EcsConstant;                /**< Tag added to enum/bitmask constants. */
FLECS_API extern const ecs_entity_t EcsQuantity;                /**< Tag added to unit quantities. */

/* Primitive type component ids */

FLECS_API extern const ecs_entity_t ecs_id(ecs_bool_t);         /**< Builtin boolean type. */
FLECS_API extern const ecs_entity_t ecs_id(ecs_char_t);         /**< Builtin char type. */
FLECS_API extern const ecs_entity_t ecs_id(ecs_byte_t);         /**< Builtin byte type. */
FLECS_API extern const ecs_entity_t ecs_id(ecs_u8_t);           /**< Builtin 8 bit unsigned int type. */
FLECS_API extern const ecs_entity_t ecs_id(ecs_u16_t);          /**< Builtin 16 bit unsigned int type. */
FLECS_API extern const ecs_entity_t ecs_id(ecs_u32_t);          /**< Builtin 32 bit unsigned int type. */
FLECS_API extern const ecs_entity_t ecs_id(ecs_u64_t);          /**< Builtin 64 bit unsigned int type. */
FLECS_API extern const ecs_entity_t ecs_id(ecs_uptr_t);         /**< Builtin pointer sized unsigned int type. */
FLECS_API extern const ecs_entity_t ecs_id(ecs_i8_t);           /**< Builtin 8 bit signed int type. */
FLECS_API extern const ecs_entity_t ecs_id(ecs_i16_t);          /**< Builtin 16 bit signed int type. */
FLECS_API extern const ecs_entity_t ecs_id(ecs_i32_t);          /**< Builtin 32 bit signed int type. */
FLECS_API extern const ecs_entity_t ecs_id(ecs_i64_t);          /**< Builtin 64 bit signed int type. */
FLECS_API extern const ecs_entity_t ecs_id(ecs_iptr_t);         /**< Builtin pointer sized signed int type. */
FLECS_API extern const ecs_entity_t ecs_id(ecs_f32_t);          /**< Builtin 32 bit floating point type. */
FLECS_API extern const ecs_entity_t ecs_id(ecs_f64_t);          /**< Builtin 64 bit floating point type. */
FLECS_API extern const ecs_entity_t ecs_id(ecs_string_t);       /**< Builtin string type. */
FLECS_API extern const ecs_entity_t ecs_id(ecs_entity_t);       /**< Builtin entity type. */
FLECS_API extern const ecs_entity_t ecs_id(ecs_id_t);           /**< Builtin (component) id type. */

/** Type kinds supported by meta addon */
typedef enum ecs_type_kind_t {
    EcsPrimitiveType,
    EcsBitmaskType,
    EcsEnumType,
    EcsStructType,
    EcsArrayType,
    EcsVectorType,
    EcsOpaqueType,
    EcsTypeKindLast = EcsOpaqueType
} ecs_type_kind_t;

/** Component that is automatically added to every type with the right kind. */
typedef struct EcsType {
    ecs_type_kind_t kind;  /**< Type kind. */
    bool existing;         /**< Did the type exist or is it populated from reflection */
    bool partial;          /**< Is the reflection data a partial type description */
} EcsType;

/** Primitive type kinds supported by meta addon */
typedef enum ecs_primitive_kind_t {
    EcsBool = 1,
    EcsChar,
    EcsByte,
    EcsU8,
    EcsU16,
    EcsU32,
    EcsU64,
    EcsI8,
    EcsI16,
    EcsI32,
    EcsI64,
    EcsF32,
    EcsF64,
    EcsUPtr,
    EcsIPtr,
    EcsString,
    EcsEntity,
    EcsId,
    EcsPrimitiveKindLast = EcsId
} ecs_primitive_kind_t;

/** Component added to primitive types */
typedef struct EcsPrimitive {
    ecs_primitive_kind_t kind;                     /**< Primitive type kind. */
} EcsPrimitive;

/** Component added to member entities */
typedef struct EcsMember {
    ecs_entity_t type;                             /**< Member type. */
    int32_t count;                                 /**< Number of elements (for inline arrays). */
    ecs_entity_t unit;                             /**< Member unit. */
    int32_t offset;                                /**< Member offset. */
    bool use_offset;                               /**< If offset should be explicitly used. */
} EcsMember;

/** Type expressing a range for a member value */
typedef struct ecs_member_value_range_t {
    double min;                                    /**< Min member value. */
    double max;                                    /**< Max member value. */
} ecs_member_value_range_t;

/** Component added to member entities to express valid value ranges */
typedef struct EcsMemberRanges {
    ecs_member_value_range_t value;                /**< Member value range. */
    ecs_member_value_range_t warning;              /**< Member value warning range. */
    ecs_member_value_range_t error;                /**< Member value error range. */
} EcsMemberRanges;

/** Element type of members vector in EcsStruct */
typedef struct ecs_member_t {
    /** Must be set when used with ecs_struct_desc_t */
    const char *name;

    /** Member type. */
    ecs_entity_t type;

    /** Element count (for inline arrays). May be set when used with ecs_struct_desc_t */
    int32_t count;

    /** May be set when used with ecs_struct_desc_t. Member offset. */
    int32_t offset;

    /** May be set when used with ecs_struct_desc_t, will be auto-populated if
     * type entity is also a unit */
    ecs_entity_t unit;

    /** Set to true to prevent automatic offset computation. This option should
     * be used when members are registered out of order or where calculation of
     * member offsets doesn't match C type offsets. */
    bool use_offset;

    /** Numerical range that specifies which values member can assume. This
     * range may be used by UI elements such as a progress bar or slider. The
     * value of a member should not exceed this range. */
    ecs_member_value_range_t range;

    /** Numerical range outside of which the value represents an error. This
     * range may be used by UI elements to style a value. */
    ecs_member_value_range_t error_range;

    /** Numerical range outside of which the value represents an warning. This
     * range may be used by UI elements to style a value. */
    ecs_member_value_range_t warning_range;

    /** Should not be set by ecs_struct_desc_t */
    ecs_size_t size;

    /** Should not be set by ecs_struct_desc_t */
    ecs_entity_t member;
} ecs_member_t;

/** Component added to struct type entities */
typedef struct EcsStruct {
    /** Populated from child entities with Member component */
    ecs_vec_t members; /* vector<ecs_member_t> */
} EcsStruct;

/** Type that describes an enum constant */
typedef struct ecs_enum_constant_t {
    /** Must be set when used with ecs_enum_desc_t */
    const char *name;

    /** May be set when used with ecs_enum_desc_t */
    int64_t value;

    /** For when the underlying type is unsigned */
    uint64_t value_unsigned;

    /** Should not be set by ecs_enum_desc_t */
    ecs_entity_t constant;
} ecs_enum_constant_t;

/** Component added to enum type entities */
typedef struct EcsEnum {
    ecs_entity_t underlying_type;

    /** Populated from child entities with Constant component */
    ecs_map_t constants; /**< map<i32_t, ecs_enum_constant_t> */
} EcsEnum;

/** Type that describes an bitmask constant */
typedef struct ecs_bitmask_constant_t {
    /** Must be set when used with ecs_bitmask_desc_t */
    const char *name;

    /** May be set when used with ecs_bitmask_desc_t */
    ecs_flags64_t value;

    /** Keep layout the same with ecs_enum_constant_t */
    int64_t _unused;

    /** Should not be set by ecs_bitmask_desc_t */
    ecs_entity_t constant;
} ecs_bitmask_constant_t;

/** Component added to bitmask type entities */
typedef struct EcsBitmask {
    /* Populated from child entities with Constant component */
    ecs_map_t constants; /**< map<u32_t, ecs_bitmask_constant_t> */
} EcsBitmask;

/** Component added to array type entities */
typedef struct EcsArray {
    ecs_entity_t type; /**< Element type */
    int32_t count;     /**< Number of elements */
} EcsArray;

/** Component added to vector type entities */
typedef struct EcsVector {
    ecs_entity_t type; /**< Element type */
} EcsVector;


/* Opaque type support */

#if !defined(__cplusplus) || !defined(FLECS_CPP)

/** Serializer interface */
typedef struct ecs_serializer_t {
    /* Serialize value */
    int (*value)(
        const struct ecs_serializer_t *ser, /**< Serializer */
        ecs_entity_t type,                  /**< Type of the value to serialize */
        const void *value);                 /**< Pointer to the value to serialize */

    /* Serialize member */
    int (*member)(
        const struct ecs_serializer_t *ser, /**< Serializer */
        const char *member);                /**< Member name */

    const ecs_world_t *world;               /**< The world. */
    void *ctx;                              /**< Serializer context. */
} ecs_serializer_t;

#elif defined(__cplusplus)

} /* extern "C" { */

/** Serializer interface (same layout as C, but with convenience methods) */
typedef struct ecs_serializer_t {
    /* Serialize value */
    int (*value_)(
        const struct ecs_serializer_t *ser,
        ecs_entity_t type,
        const void *value);

    /* Serialize member */
    int (*member_)(
        const struct ecs_serializer_t *ser,
        const char *name);

    /* Serialize value */
    int value(ecs_entity_t type, const void *value) const;

    /* Serialize value */
    template <typename T>
    int value(const T& value) const;

    /* Serialize member */
    int member(const char *name) const;

    const ecs_world_t *world;
    void *ctx;
} ecs_serializer_t;

extern "C" {
#endif

/** Callback invoked serializing an opaque type. */
typedef int (*ecs_meta_serialize_t)(
    const ecs_serializer_t *ser,
    const void *src);                  /**< Pointer to value to serialize */

/** Opaque type reflection data. 
 * An opaque type is a type with an unknown layout that can be mapped to a type
 * known to the reflection framework. See the opaque type reflection examples.
 */
typedef struct EcsOpaque {
    ecs_entity_t as_type;              /**< Type that describes the serialized output */
    ecs_meta_serialize_t serialize;    /**< Serialize action */

    /* Deserializer interface
     * Only override the callbacks that are valid for the opaque type. If a
     * deserializer attempts to assign a value type that is not supported by the
     * interface, a conversion error is thrown.
     */

    /** Assign bool value */
    void (*assign_bool)(
        void *dst,
        bool value);

    /** Assign char value */
    void (*assign_char)(
        void *dst,
        char value);

    /** Assign int value */
    void (*assign_int)(
        void *dst,
        int64_t value);

    /** Assign unsigned int value */
    void (*assign_uint)(
        void *dst,
        uint64_t value);

    /** Assign float value */
    void (*assign_float)(
        void *dst,
        double value);

    /** Assign string value */
    void (*assign_string)(
        void *dst,
        const char *value);

    /** Assign entity value */
    void (*assign_entity)(
        void *dst,
        ecs_world_t *world,
        ecs_entity_t entity);

    /** Assign (component) id value */
    void (*assign_id)(
        void *dst,
        ecs_world_t *world,
        ecs_id_t id);

    /** Assign null value */
    void (*assign_null)(
        void *dst);

    /** Clear collection elements */
    void (*clear)(
        void *dst);

    /** Ensure & get collection element */
    void* (*ensure_element)(
        void *dst,
        size_t elem);

    /** Ensure & get element */
    void* (*ensure_member)(
        void *dst,
        const char *member);

    /** Return number of elements */
    size_t (*count)(
        const void *dst);

    /** Resize to number of elements */
    void (*resize)(
        void *dst,
        size_t count);
} EcsOpaque;


/* Units */

/** Helper type to describe translation between two units. Note that this
 * is not intended as a generic approach to unit conversions (e.g. from celsius
 * to fahrenheit) but to translate between units that derive from the same base
 * (e.g. meters to kilometers).
 *
 * Note that power is applied to the factor. When describing a translation of
 * 1000, either use {factor = 1000, power = 1} or {factor = 1, power = 3}. */
typedef struct ecs_unit_translation_t {
    int32_t factor; /**< Factor to apply (e.g. "1000", "1000000", "1024") */
    int32_t power;  /**< Power to apply to factor (e.g. "1", "3", "-9") */
} ecs_unit_translation_t;

/** Component that stores unit data. */
typedef struct EcsUnit {
    char *symbol;                                  /**< Unit symbol. */
    ecs_entity_t prefix;                           /**< Order of magnitude prefix relative to derived */
    ecs_entity_t base;                             /**< Base unit (e.g. "meters") */
    ecs_entity_t over;                             /**< Over unit (e.g. "per second") */
    ecs_unit_translation_t translation;            /**< Translation for derived unit */
} EcsUnit;

/** Component that stores unit prefix data. */
typedef struct EcsUnitPrefix {
    char *symbol;                                 /**< Symbol of prefix (e.g. "K", "M", "Ki") */
    ecs_unit_translation_t translation;           /**< Translation of prefix */
} EcsUnitPrefix;


/* Serializer utilities */

/** Serializer instruction opcodes. 
 * The meta type serializer works by generating a flattened array with 
 * instructions that tells a serializer what kind of fields can be found in a
 * type at which offsets.
*/
typedef enum ecs_meta_type_op_kind_t {
    EcsOpArray,
    EcsOpVector,
    EcsOpOpaque,
    EcsOpPush,
    EcsOpPop,

    EcsOpScope, /**< Marks last constant that can open/close a scope */

    EcsOpEnum,
    EcsOpBitmask,

    EcsOpPrimitive, /**< Marks first constant that's a primitive */

    EcsOpBool,
    EcsOpChar,
    EcsOpByte,
    EcsOpU8,
    EcsOpU16,
    EcsOpU32,
    EcsOpU64,
    EcsOpI8,
    EcsOpI16,
    EcsOpI32,
    EcsOpI64,
    EcsOpF32,
    EcsOpF64,
    EcsOpUPtr,
    EcsOpIPtr,
    EcsOpString,
    EcsOpEntity,
    EcsOpId,
    EcsMetaTypeOpKindLast = EcsOpId
} ecs_meta_type_op_kind_t;

/** Meta type serializer instruction data. */
typedef struct ecs_meta_type_op_t {
    ecs_meta_type_op_kind_t kind;                  /**< Instruction opcode. */
    ecs_size_t offset;                             /**< Offset of current field */
    int32_t count;                                 /**< Number of elements (for inline arrays). */
    const char *name;                              /**< Name of value (only used for struct members) */
    int32_t op_count;                              /**< Number of operations until next field or end */
    ecs_size_t size;                               /**< Size of type of operation */
    ecs_entity_t type;                             /**< Type entity */
    int32_t member_index;                          /**< Index of member in struct */
    ecs_hashmap_t *members;                        /**< string -> member index (structs only) */
} ecs_meta_type_op_t;

/** Component that stores the type serializer.
 * Added to all types with reflection data.
 */
typedef struct EcsTypeSerializer {
    ecs_vec_t ops;      /**< vector<ecs_meta_type_op_t> */
} EcsTypeSerializer;


/* Deserializer utilities */

/** Maximum level of type nesting. 
 * >32 levels of nesting is not sane.
 */
#define ECS_META_MAX_SCOPE_DEPTH (32)

/** Type with information about currently serialized scope. */
typedef struct ecs_meta_scope_t {
    ecs_entity_t type;                             /**< The type being iterated */
    ecs_meta_type_op_t *ops;                       /**< The type operations (see ecs_meta_type_op_t) */
    int32_t op_count;                              /**< Number of operations in ops array to process */
    int32_t op_cur;                                /**< Current operation */
    int32_t elem_cur;                              /**< Current element (for collections) */
    int32_t prev_depth;                            /**< Depth to restore, in case dotmember was used */
    void *ptr;                                     /**< Pointer to the value being iterated */
    const EcsComponent *comp;                      /**< Pointer to component, in case size/alignment is needed */
    const EcsOpaque *opaque;                       /**< Opaque type interface */
    ecs_vec_t *vector;                             /**< Current vector, in case a vector is iterated */
    ecs_hashmap_t *members;                        /**< string -> member index */
    bool is_collection;                            /**< Is the scope iterating elements? */
    bool is_inline_array;                          /**< Is the scope iterating an inline array? */
    bool is_empty_scope;                           /**< Was scope populated (for collections) */
} ecs_meta_scope_t;

/** Type that enables iterating/populating a value using reflection data. */
typedef struct ecs_meta_cursor_t {
    const ecs_world_t *world;                      /**< The world. */
    ecs_meta_scope_t scope[ECS_META_MAX_SCOPE_DEPTH]; /**< Cursor scope stack. */
    int32_t depth;                                 /**< Current scope depth. */
    bool valid;                                    /**< Does the cursor point to a valid field. */
    bool is_primitive_scope;                       /**< If in root scope, this allows for a push for primitive types */

    /** Custom entity lookup action for overriding default ecs_lookup */
    ecs_entity_t (*lookup_action)(const ecs_world_t*, const char*, void*);
    void *lookup_ctx;                              /**< Context for lookup_action */
} ecs_meta_cursor_t;

/** Create meta cursor.
 * A meta cursor allows for walking over, reading and writing a value without
 * having to know its type at compile time.
 * 
 * When a value is assigned through the cursor API, it will get converted to
 * the actual value of the underlying type. This allows the underlying type to
 * change without having to update the serialized data. For example, an integer
 * field can be set by a string, a floating point can be set as integer etc.
 * 
 * @param world The world.
 * @param type The type of the value.
 * @param ptr Pointer to the value.
 * @return A meta cursor for the value.
 */
FLECS_API
ecs_meta_cursor_t ecs_meta_cursor(
    const ecs_world_t *world,
    ecs_entity_t type,
    void *ptr);

/** Get pointer to current field.
 * 
 * @param cursor The cursor.
 * @return A pointer to the current field.
 */
FLECS_API
void* ecs_meta_get_ptr(
    ecs_meta_cursor_t *cursor);

/** Move cursor to next field.
 * 
 * @param cursor The cursor.
 * @return Zero if success, non-zero if failed.
 */
FLECS_API
int ecs_meta_next(
    ecs_meta_cursor_t *cursor);

/** Move cursor to a field.
 * 
 * @param cursor The cursor.
 * @return Zero if success, non-zero if failed.
 */
FLECS_API
int ecs_meta_elem(
    ecs_meta_cursor_t *cursor,
    int32_t elem);

/** Move cursor to member.
 * 
 * @param cursor The cursor.
 * @param name The name of the member.
 * @return Zero if success, non-zero if failed.
 */
FLECS_API
int ecs_meta_member(
    ecs_meta_cursor_t *cursor,
    const char *name);

/** Move cursor to member.
 * Same as ecs_meta_member(), but with support for "foo.bar" syntax.
 * 
 * @param cursor The cursor.
 * @param name The name of the member.
 * @return Zero if success, non-zero if failed.
 */
FLECS_API
int ecs_meta_dotmember(
    ecs_meta_cursor_t *cursor,
    const char *name);

/** Push a scope (required/only valid for structs & collections).
 * 
 * @param cursor The cursor.
 * @return Zero if success, non-zero if failed.
 */
FLECS_API
int ecs_meta_push(
    ecs_meta_cursor_t *cursor);

/** Pop a struct or collection scope (must follow a push).
 * 
 * @param cursor The cursor.
 * @return Zero if success, non-zero if failed.
 */
FLECS_API
int ecs_meta_pop(
    ecs_meta_cursor_t *cursor);

/** Is the current scope a collection?.
 * 
 * @param cursor The cursor.
 * @return True if current scope is a collection, false if not.
 */
FLECS_API
bool ecs_meta_is_collection(
    const ecs_meta_cursor_t *cursor);

/** Get type of current field.
 * 
 * @param cursor The cursor.
 * @return The type of the current field.
 */
FLECS_API
ecs_entity_t ecs_meta_get_type(
    const ecs_meta_cursor_t *cursor);

/** Get unit of current field.
 * 
 * @param cursor The cursor.
 * @return The unit of the current field.
 */
FLECS_API
ecs_entity_t ecs_meta_get_unit(
    const ecs_meta_cursor_t *cursor);

/** Get member name of current field.
 * 
 * @param cursor The cursor.
 * @return The member name of the current field.
 */
FLECS_API
const char* ecs_meta_get_member(
    const ecs_meta_cursor_t *cursor);

/** Get member entity of current field.
 * 
 * @param cursor The cursor.
 * @return The member entity of the current field.
 */
FLECS_API
ecs_entity_t ecs_meta_get_member_id(
    const ecs_meta_cursor_t *cursor);

/* The set functions assign the field with the specified value. If the value
 * does not have the same type as the field, it will be cased to the field type.
 * If no valid conversion is available, the operation will fail. */

/** Set field with boolean value.
 * 
 * @param cursor The cursor.
 * @param value The value to set.
 * @return Zero if success, non-zero if failed.
 */
FLECS_API
int ecs_meta_set_bool(
    ecs_meta_cursor_t *cursor,
    bool value);

/** Set field with char value.
 * 
 * @param cursor The cursor.
 * @param value The value to set.
 * @return Zero if success, non-zero if failed.
 */
FLECS_API
int ecs_meta_set_char(
    ecs_meta_cursor_t *cursor,
    char value);

/** Set field with int value.
 * 
 * @param cursor The cursor.
 * @param value The value to set.
 * @return Zero if success, non-zero if failed.
 */
FLECS_API
int ecs_meta_set_int(
    ecs_meta_cursor_t *cursor,
    int64_t value);

/** Set field with uint value.
 * 
 * @param cursor The cursor.
 * @param value The value to set.
 * @return Zero if success, non-zero if failed.
 */
FLECS_API
int ecs_meta_set_uint(
    ecs_meta_cursor_t *cursor,
    uint64_t value);

/** Set field with float value.
 * 
 * @param cursor The cursor.
 * @param value The value to set.
 * @return Zero if success, non-zero if failed.
 */
FLECS_API
int ecs_meta_set_float(
    ecs_meta_cursor_t *cursor,
    double value);

/** Set field with string value.
 * 
 * @param cursor The cursor.
 * @param value The value to set.
 * @return Zero if success, non-zero if failed.
 */
FLECS_API
int ecs_meta_set_string(
    ecs_meta_cursor_t *cursor,
    const char *value);

/** Set field with string literal value (has enclosing "").
 * 
 * @param cursor The cursor.
 * @param value The value to set.
 * @return Zero if success, non-zero if failed.
 */
FLECS_API
int ecs_meta_set_string_literal(
    ecs_meta_cursor_t *cursor,
    const char *value);

/** Set field with entity value.
 * 
 * @param cursor The cursor.
 * @param value The value to set.
 * @return Zero if success, non-zero if failed.
 */
FLECS_API
int ecs_meta_set_entity(
    ecs_meta_cursor_t *cursor,
    ecs_entity_t value);

/** Set field with (component) id value.
 * 
 * @param cursor The cursor.
 * @param value The value to set.
 * @return Zero if success, non-zero if failed.
 */
FLECS_API
int ecs_meta_set_id(
    ecs_meta_cursor_t *cursor,
    ecs_id_t value);

/** Set field with null value.
 * 
 * @param cursor The cursor.
 * @return Zero if success, non-zero if failed.
 */
FLECS_API
int ecs_meta_set_null(
    ecs_meta_cursor_t *cursor);

/** Set field with dynamic value.
 * 
 * @param cursor The cursor.
 * @param value The value to set.
 * @return Zero if success, non-zero if failed.
 */
FLECS_API
int ecs_meta_set_value(
    ecs_meta_cursor_t *cursor,
    const ecs_value_t *value);

/* Functions for getting members. */

/** Get field value as boolean.
 * 
 * @param cursor The cursor.
 * @return The value of the current field.
 */
FLECS_API
bool ecs_meta_get_bool(
    const ecs_meta_cursor_t *cursor);

/** Get field value as char.
 * 
 * @param cursor The cursor.
 * @return The value of the current field.
 */
FLECS_API
char ecs_meta_get_char(
    const ecs_meta_cursor_t *cursor);

/** Get field value as signed integer.
 * 
 * @param cursor The cursor.
 * @return The value of the current field.
 */
FLECS_API
int64_t ecs_meta_get_int(
    const ecs_meta_cursor_t *cursor);

/** Get field value as unsigned integer.
 * 
 * @param cursor The cursor.
 * @return The value of the current field.
 */
FLECS_API
uint64_t ecs_meta_get_uint(
    const ecs_meta_cursor_t *cursor);

/** Get field value as float.
 * 
 * @param cursor The cursor.
 * @return The value of the current field.
 */
FLECS_API
double ecs_meta_get_float(
    const ecs_meta_cursor_t *cursor);

/** Get field value as string.
 * This operation does not perform conversions. If the field is not a string,
 * this operation will fail.
 * 
 * @param cursor The cursor.
 * @return The value of the current field.
 */
FLECS_API
const char* ecs_meta_get_string(
    const ecs_meta_cursor_t *cursor);

/** Get field value as entity.
 * This operation does not perform conversions. 
 * 
 * @param cursor The cursor.
 * @return The value of the current field.
 */
FLECS_API
ecs_entity_t ecs_meta_get_entity(
    const ecs_meta_cursor_t *cursor);

/** Get field value as (component) id.
 * This operation can convert from an entity. 
 * 
 * @param cursor The cursor.
 * @return The value of the current field.
 */
ecs_id_t ecs_meta_get_id(
    const ecs_meta_cursor_t *cursor);

/** Convert pointer of primitive kind to float. 
 * 
 * @param type_kind The primitive type kind of the value.
 * @param ptr Pointer to a value of a primitive type.
 * @return The value in floating point format.
 */
FLECS_API
double ecs_meta_ptr_to_float(
    ecs_primitive_kind_t type_kind,
    const void *ptr);

/* API functions for creating meta types */

/** Used with ecs_primitive_init(). */
typedef struct ecs_primitive_desc_t {
    ecs_entity_t entity;       /**< Existing entity to use for type (optional). */
    ecs_primitive_kind_t kind; /**< Primitive type kind. */
} ecs_primitive_desc_t;

/** Create a new primitive type. 
 * 
 * @param world The world.
 * @param desc The type descriptor.
 * @return The new type, 0 if failed.
 */
FLECS_API
ecs_entity_t ecs_primitive_init(
    ecs_world_t *world,
    const ecs_primitive_desc_t *desc);


/** Used with ecs_enum_init(). */
typedef struct ecs_enum_desc_t {
    ecs_entity_t entity;       /**< Existing entity to use for type (optional). */
    ecs_enum_constant_t constants[ECS_MEMBER_DESC_CACHE_SIZE]; /**< Enum constants. */
    ecs_entity_t underlying_type;
} ecs_enum_desc_t;

/** Create a new enum type. 
 * 
 * @param world The world.
 * @param desc The type descriptor.
 * @return The new type, 0 if failed.
 */
FLECS_API
ecs_entity_t ecs_enum_init(
    ecs_world_t *world,
    const ecs_enum_desc_t *desc);


/** Used with ecs_bitmask_init(). */
typedef struct ecs_bitmask_desc_t {
    ecs_entity_t entity;       /**< Existing entity to use for type (optional). */
    ecs_bitmask_constant_t constants[ECS_MEMBER_DESC_CACHE_SIZE]; /**< Bitmask constants. */
} ecs_bitmask_desc_t;

/** Create a new bitmask type. 
 * 
 * @param world The world.
 * @param desc The type descriptor.
 * @return The new type, 0 if failed.
 */
FLECS_API
ecs_entity_t ecs_bitmask_init(
    ecs_world_t *world,
    const ecs_bitmask_desc_t *desc);


/** Used with ecs_array_init(). */
typedef struct ecs_array_desc_t {
    ecs_entity_t entity;  /**< Existing entity to use for type (optional). */
    ecs_entity_t type;    /**< Element type. */
    int32_t count;        /**< Number of elements. */
} ecs_array_desc_t;

/** Create a new array type. 
 * 
 * @param world The world.
 * @param desc The type descriptor.
 * @return The new type, 0 if failed.
 */
FLECS_API
ecs_entity_t ecs_array_init(
    ecs_world_t *world,
    const ecs_array_desc_t *desc);


/** Used with ecs_vector_init(). */
typedef struct ecs_vector_desc_t {
    ecs_entity_t entity;  /**< Existing entity to use for type (optional). */
    ecs_entity_t type;    /**< Element type. */
} ecs_vector_desc_t;

/** Create a new vector type. 
 * 
 * @param world The world.
 * @param desc The type descriptor.
 * @return The new type, 0 if failed.
 */
FLECS_API
ecs_entity_t ecs_vector_init(
    ecs_world_t *world,
    const ecs_vector_desc_t *desc);


/** Used with ecs_struct_init(). */
typedef struct ecs_struct_desc_t {
    ecs_entity_t entity; /**< Existing entity to use for type (optional). */
    ecs_member_t members[ECS_MEMBER_DESC_CACHE_SIZE]; /**< Struct members. */
} ecs_struct_desc_t;

/** Create a new struct type. 
 * 
 * @param world The world.
 * @param desc The type descriptor.
 * @return The new type, 0 if failed.
 */
FLECS_API
ecs_entity_t ecs_struct_init(
    ecs_world_t *world,
    const ecs_struct_desc_t *desc);


/** Used with ecs_opaque_init(). */
typedef struct ecs_opaque_desc_t {
    ecs_entity_t entity;  /**< Existing entity to use for type (optional). */
    EcsOpaque type;       /**< Type that the opaque type maps to. */
} ecs_opaque_desc_t;

/** Create a new opaque type.
 * Opaque types are types of which the layout doesn't match what can be modelled
 * with the primitives of the meta framework, but which have a structure
 * that can be described with meta primitives. Typical examples are STL types
 * such as std::string or std::vector, types with a nontrivial layout, and types
 * that only expose getter/setter methods.
 *
 * An opaque type is a combination of a serialization function, and a handle to
 * a meta type which describes the structure of the serialized output. For
 * example, an opaque type for std::string would have a serializer function that
 * accesses .c_str(), and with type ecs_string_t.
 *
 * The serializer callback accepts a serializer object and a pointer to the
 * value of the opaque type to be serialized. The serializer has two methods:
 *
 * - value, which serializes a value (such as .c_str())
 * - member, which specifies a member to be serialized (in the case of a struct)
 * 
 * @param world The world.
 * @param desc The type descriptor.
 * @return The new type, 0 if failed.
 */
FLECS_API
ecs_entity_t ecs_opaque_init(
    ecs_world_t *world,
    const ecs_opaque_desc_t *desc);


/** Used with ecs_unit_init(). */
typedef struct ecs_unit_desc_t {
    /** Existing entity to associate with unit (optional). */
    ecs_entity_t entity;

    /** Unit symbol, e.g. "m", "%", "g". (optional). */
    const char *symbol;

    /** Unit quantity, e.g. distance, percentage, weight. (optional). */
    ecs_entity_t quantity;

    /** Base unit, e.g. "meters" (optional). */
    ecs_entity_t base;

    /** Over unit, e.g. "per second" (optional). */
    ecs_entity_t over;

    /** Translation to apply to derived unit (optional). */
    ecs_unit_translation_t translation;

    /** Prefix indicating order of magnitude relative to the derived unit. If set
     * together with "translation", the values must match. If translation is not
     * set, setting prefix will auto-populate it.
     * Additionally, setting the prefix will enforce that the symbol (if set)
     * is consistent with the prefix symbol + symbol of the derived unit. If the
     * symbol is not set, it will be auto populated. */
    ecs_entity_t prefix;
} ecs_unit_desc_t;

/** Create a new unit. 
 * 
 * @param world The world.
 * @param desc The unit descriptor.
 * @return The new unit, 0 if failed.
 */
FLECS_API
ecs_entity_t ecs_unit_init(
    ecs_world_t *world,
    const ecs_unit_desc_t *desc);


/** Used with ecs_unit_prefix_init(). */
typedef struct ecs_unit_prefix_desc_t {
    /** Existing entity to associate with unit prefix (optional). */
    ecs_entity_t entity;

    /** Unit symbol, e.g. "m", "%", "g". (optional). */
    const char *symbol;

    /** Translation to apply to derived unit (optional). */
    ecs_unit_translation_t translation;
} ecs_unit_prefix_desc_t;

/** Create a new unit prefix. 
 * 
 * @param world The world.
 * @param desc The type descriptor.
 * @return The new unit prefix, 0 if failed.
 */
FLECS_API
ecs_entity_t ecs_unit_prefix_init(
    ecs_world_t *world,
    const ecs_unit_prefix_desc_t *desc);


/** Create a new quantity. 
 * 
 * @param world The world.
 * @param desc The quantity descriptor.
 * @return The new quantity, 0 if failed.
 */
FLECS_API
ecs_entity_t ecs_quantity_init(
    ecs_world_t *world,
    const ecs_entity_desc_t *desc);

/* Convenience macros */

/** Create a primitive type. */
#define ecs_primitive(world, ...)\
    ecs_primitive_init(world, &(ecs_primitive_desc_t) __VA_ARGS__ )

/** Create an enum type. */
#define ecs_enum(world, ...)\
    ecs_enum_init(world, &(ecs_enum_desc_t) __VA_ARGS__ )

/** Create a bitmask type. */
#define ecs_bitmask(world, ...)\
    ecs_bitmask_init(world, &(ecs_bitmask_desc_t) __VA_ARGS__ )

/** Create an array type. */
#define ecs_array(world, ...)\
    ecs_array_init(world, &(ecs_array_desc_t) __VA_ARGS__ )

/** Create a vector type. */
#define ecs_vector(world, ...)\
    ecs_vector_init(world, &(ecs_vector_desc_t) __VA_ARGS__ )

/** Create an opaque type. */
#define ecs_opaque(world, ...)\
    ecs_opaque_init(world, &(ecs_opaque_desc_t) __VA_ARGS__ )

/** Create a struct type. */
#define ecs_struct(world, ...)\
    ecs_struct_init(world, &(ecs_struct_desc_t) __VA_ARGS__ )

/** Create a unit. */
#define ecs_unit(world, ...)\
    ecs_unit_init(world, &(ecs_unit_desc_t) __VA_ARGS__ )

/** Create a unit prefix. */
#define ecs_unit_prefix(world, ...)\
    ecs_unit_prefix_init(world, &(ecs_unit_prefix_desc_t) __VA_ARGS__ )

/** Create a unit quantity. */
#define ecs_quantity(world, ...)\
    ecs_quantity_init(world, &(ecs_entity_desc_t) __VA_ARGS__ )


/** Meta module import function.
 * Usage:
 * @code
 * ECS_IMPORT(world, FlecsMeta)
 * @endcode
 * 
 * @param world The world.
 */
FLECS_API
void FlecsMetaImport(
    ecs_world_t *world);

#ifdef __cplusplus
}
#endif

/**
 * @file addons/meta_c.h
 * @brief Utility macros for populating reflection data in C.
 */

#ifdef FLECS_META

/**
 * @defgroup c_addons_meta_c Meta Utilities
 * @ingroup c_addons
 * Macro utilities to automatically insert reflection data.
 *
 * @{
 */

#ifndef FLECS_META_C_H
#define FLECS_META_C_H

#ifdef __cplusplus
extern "C" {
#endif

/* Macro that controls behavior of API. Usually set in module header. When the
 * macro is not defined, it defaults to IMPL. */

/* Define variables used by reflection utilities. This should only be defined
 * by the module itself, not by the code importing the module */
/* #define ECS_META_IMPL IMPL */

/* Don't define variables used by reflection utilities but still declare the
 * variable for the component id. This enables the reflection utilities to be
 * used for global component variables, even if no reflection is used. */
/* #define ECS_META_IMPL DECLARE */

/* Don't define variables used by reflection utilities. This generates an extern
 * variable for the component identifier. */
/* #define ECS_META_IMPL EXTERN */

/** Declare component with descriptor. */
#define ECS_META_COMPONENT(world, name)\
    ECS_COMPONENT_DEFINE(world, name);\
    ecs_meta_from_desc(world, ecs_id(name),\
        FLECS__##name##_kind, FLECS__##name##_desc)

/** ECS_STRUCT(name, body). */
#define ECS_STRUCT(name, ...)\
    ECS_META_IMPL_CALL(ECS_STRUCT_, ECS_META_IMPL, name, #__VA_ARGS__);\
    ECS_STRUCT_TYPE(name, __VA_ARGS__)

/** ECS_ENUM(name, body). */
#define ECS_ENUM(name, ...)\
    ECS_META_IMPL_CALL(ECS_ENUM_, ECS_META_IMPL, name, #__VA_ARGS__);\
    ECS_ENUM_TYPE(name, __VA_ARGS__)

/** ECS_BITMASK(name, body). */
#define ECS_BITMASK(name, ...)\
    ECS_META_IMPL_CALL(ECS_BITMASK_, ECS_META_IMPL, name, #__VA_ARGS__);\
    ECS_ENUM_TYPE(name, __VA_ARGS__)

/** Macro used to mark part of type for which no reflection data is created. */
#define ECS_PRIVATE

/** Populate meta information from type descriptor. */
FLECS_API
int ecs_meta_from_desc(
    ecs_world_t *world,
    ecs_entity_t component,
    ecs_type_kind_t kind,
    const char *desc);


/** \cond
 * Private utilities to switch between meta IMPL, DECLARE and EXTERN variants.
 */

#define ECS_META_IMPL_CALL_INNER(base, impl, name, type_desc)\
    base ## impl(name, type_desc)

#define ECS_META_IMPL_CALL(base, impl, name, type_desc)\
    ECS_META_IMPL_CALL_INNER(base, impl, name, type_desc)

/* ECS_STRUCT implementation */
#define ECS_STRUCT_TYPE(name, ...)\
    typedef struct __VA_ARGS__ name

#define ECS_STRUCT_ECS_META_IMPL ECS_STRUCT_IMPL

#define ECS_STRUCT_IMPL(name, type_desc)\
    extern ECS_COMPONENT_DECLARE(name);\
    static const char *FLECS__##name##_desc = type_desc;\
    static ecs_type_kind_t FLECS__##name##_kind = EcsStructType;\
    ECS_COMPONENT_DECLARE(name) = 0

#define ECS_STRUCT_DECLARE(name, type_desc)\
    extern ECS_COMPONENT_DECLARE(name);\
    ECS_COMPONENT_DECLARE(name) = 0

#define ECS_STRUCT_EXTERN(name, type_desc)\
    extern ECS_COMPONENT_DECLARE(name)


/* ECS_ENUM implementation */
#define ECS_ENUM_TYPE(name, ...)\
    typedef enum __VA_ARGS__ name

#define ECS_ENUM_ECS_META_IMPL ECS_ENUM_IMPL

#define ECS_ENUM_IMPL(name, type_desc)\
    extern ECS_COMPONENT_DECLARE(name);\
    static const char *FLECS__##name##_desc = type_desc;\
    static ecs_type_kind_t FLECS__##name##_kind = EcsEnumType;\
    ECS_COMPONENT_DECLARE(name) = 0

#define ECS_ENUM_DECLARE(name, type_desc)\
    extern ECS_COMPONENT_DECLARE(name);\
    ECS_COMPONENT_DECLARE(name) = 0

#define ECS_ENUM_EXTERN(name, type_desc)\
    extern ECS_COMPONENT_DECLARE(name)


/* ECS_BITMASK implementation */
#define ECS_BITMASK_TYPE(name, ...)\
    typedef enum __VA_ARGS__ name

#define ECS_BITMASK_ECS_META_IMPL ECS_BITMASK_IMPL

#define ECS_BITMASK_IMPL(name, type_desc)\
    extern ECS_COMPONENT_DECLARE(name);\
    static const char *FLECS__##name##_desc = type_desc;\
    static ecs_type_kind_t FLECS__##name##_kind = EcsBitmaskType;\
    ECS_COMPONENT_DECLARE(name) = 0

#define ECS_BITMASK_DECLARE(name, type_desc)\
    extern ECS_COMPONENT_DECLARE(name);\
    ECS_COMPONENT_DECLARE(name) = 0

#define ECS_BITMASK_EXTERN(name, type_desc)\
    extern ECS_COMPONENT_DECLARE(name)

/** \endcond */

#ifdef __cplusplus
}
#endif

#endif // FLECS_META_H

/** @} */

#endif // FLECS_META


#endif

/** @} */

#endif

#endif

#ifdef FLECS_OS_API_IMPL
#ifdef FLECS_NO_OS_API_IMPL
#error "FLECS_NO_OS_API_IMPL failed: OS_API_IMPL is required by other addons"
#endif
/**
 * @file addons/os_api_impl.h
 * @brief Default OS API implementation.
 */

#ifdef FLECS_OS_API_IMPL

/**
 * @defgroup c_addons_os_api_impl OS API Implementation
 * @ingroup c_addons
 * Default implementation for OS API interface.
 *
 * @{
 */

#ifndef FLECS_OS_API_IMPL_H
#define FLECS_OS_API_IMPL_H

#ifdef __cplusplus
extern "C" {
#endif

FLECS_API
void ecs_set_os_api_impl(void);

#ifdef __cplusplus
}
#endif

#endif // FLECS_OS_API_IMPL_H

/** @} */

#endif // FLECS_OS_API_IMPL

#endif

#ifdef FLECS_MODULE
#ifdef FLECS_NO_MODULE
#error "FLECS_NO_MODULE failed: MODULE is required by other addons"
#endif
/**
 * @file addons/module.h
 * @brief Module addon.
 *
 * The module addon allows for creating and importing modules. Flecs modules
 * enable applications to organize components and systems into reusable units of
 * code that can easily be across projects.
 */

#ifdef FLECS_MODULE

/**
 * @defgroup c_addons_module Module
 * @ingroup c_addons
 * Modules organize components, systems and more in reusable units of code.
 *
 * @{
 */

#ifndef FLECS_MODULE_H
#define FLECS_MODULE_H

#ifdef __cplusplus
extern "C" {
#endif

/** Import a module.
 * This operation will load a modules and store the public module handles in the
 * handles_out out parameter. The module name will be used to verify if the
 * module was already loaded, in which case it won't be reimported. The name
 * will be translated from PascalCase to an entity path (pascal.case) before the
 * lookup occurs.
 *
 * Module contents will be stored as children of the module entity. This
 * prevents modules from accidentally defining conflicting identifiers. This is
 * enforced by setting the scope before and after loading the module to the
 * module entity id.
 *
 * A more convenient way to import a module is by using the ECS_IMPORT macro.
 *
 * @param world The world.
 * @param module The module import function.
 * @param module_name The name of the module.
 * @return The module entity.
 */
FLECS_API
ecs_entity_t ecs_import(
    ecs_world_t *world,
    ecs_module_action_t module,
    const char *module_name);

/** Same as ecs_import(), but with name to scope conversion.
 * PascalCase names are automatically converted to scoped names.
 *
 * @param world The world.
 * @param module The module import function.
 * @param module_name_c The name of the module.
 * @return The module entity.
 */
FLECS_API
ecs_entity_t ecs_import_c(
    ecs_world_t *world,
    ecs_module_action_t module,
    const char *module_name_c);

/** Import a module from a library.
 * Similar to ecs_import(), except that this operation will attempt to load the
 * module from a dynamic library.
 *
 * A library may contain multiple modules, which is why both a library name and
 * a module name need to be provided. If only a library name is provided, the
 * library name will be reused for the module name.
 *
 * The library will be looked up using a canonical name, which is in the same
 * form as a module, like `flecs.components.transform`. To transform this
 * identifier to a platform specific library name, the operation relies on the
 * module_to_dl callback of the os_api which the application has to override if
 * the default does not yield the correct library name.
 *
 * @param world The world.
 * @param library_name The name of the library to load.
 * @param module_name The name of the module to load.
 */
FLECS_API
ecs_entity_t ecs_import_from_library(
    ecs_world_t *world,
    const char *library_name,
    const char *module_name);

/** Register a new module. */
FLECS_API
ecs_entity_t ecs_module_init(
    ecs_world_t *world,
    const char *c_name,
    const ecs_component_desc_t *desc);

/** Define module. */
#define ECS_MODULE_DEFINE(world, id)\
    {\
        ecs_component_desc_t desc = {0};\
        desc.entity = ecs_id(id);\
        ecs_id(id) = ecs_module_init(world, #id, &desc);\
        ecs_set_scope(world, ecs_id(id));\
    }

/** Create a module. */
#define ECS_MODULE(world, id)\
    ecs_entity_t ecs_id(id) = 0; ECS_MODULE_DEFINE(world, id)\
    (void)ecs_id(id)

/** Wrapper around ecs_import().
 * This macro provides a convenient way to load a module with the world. It can
 * be used like this:
 *
 * @code
 * ECS_IMPORT(world, FlecsSystemsPhysics);
 * @endcode
 */
#define ECS_IMPORT(world, id) ecs_import_c(world, id##Import, #id)

#ifdef __cplusplus
}
#endif

#endif

/** @} */

#endif

#endif

#ifdef FLECS_CPP
#ifdef FLECS_NO_CPP
#error "FLECS_NO_CPP failed: CPP is required by other addons"
#endif
/**
 * @file addons/flecs_cpp.h
 * @brief C++ utility functions
 *
 * This header contains utility functions that are accessible from both C and
 * C++ code. These functions are not part of the public API and are not meant
 * to be used directly by applications.
 */

#ifdef FLECS_CPP

#ifndef FLECS_CPP_H
#define FLECS_CPP_H

#ifdef __cplusplus
extern "C" {
#endif

// The functions in this file can be used from C or C++, but these macros are only relevant to C++.
#ifdef __cplusplus

#if defined(__clang__)
#define ECS_FUNC_NAME_FRONT(type, name) ((sizeof(#type) + sizeof(" flecs::_::() [T = ") + sizeof(#name)) - 3u)
#define ECS_FUNC_NAME_BACK (sizeof("]") - 1u)
#define ECS_FUNC_NAME __PRETTY_FUNCTION__
#elif defined(__GNUC__)
#define ECS_FUNC_NAME_FRONT(type, name) ((sizeof(#type) + sizeof(" flecs::_::() [with T = ") + sizeof(#name)) - 3u)
#define ECS_FUNC_NAME_BACK (sizeof("]") - 1u)
#define ECS_FUNC_NAME __PRETTY_FUNCTION__
#elif defined(_WIN32)
#define ECS_FUNC_NAME_FRONT(type, name) ((sizeof(#type) + sizeof(" __cdecl flecs::_::<") + sizeof(#name)) - 3u)
#define ECS_FUNC_NAME_BACK (sizeof(">(void)") - 1u)
#define ECS_FUNC_NAME __FUNCSIG__
#else
#error "implicit component registration not supported"
#endif

#define ECS_FUNC_TYPE_LEN(type, name, str)\
    (flecs::string::length(str) - (ECS_FUNC_NAME_FRONT(type, name) + ECS_FUNC_NAME_BACK))

#endif

FLECS_API
char* ecs_cpp_get_type_name(
    char *type_name, 
    const char *func_name,
    size_t len,
    size_t front_len);

FLECS_API
char* ecs_cpp_get_symbol_name(
    char *symbol_name,
    const char *type_name,
    size_t len);

FLECS_API
char* ecs_cpp_get_constant_name(
    char *constant_name,
    const char *func_name,
    size_t len,
    size_t back_len);

FLECS_API
const char* ecs_cpp_trim_module(
    ecs_world_t *world,
    const char *type_name);

FLECS_API
ecs_entity_t ecs_cpp_component_find(
    ecs_world_t *world,
    ecs_entity_t id,
    const char *name,
    const char *symbol,
    size_t size,
    size_t alignment,
    bool implicit_name,
    bool *existing_out);

FLECS_API
ecs_entity_t ecs_cpp_component_register(
    ecs_world_t *world,
    ecs_entity_t s_id,
    ecs_entity_t id,
    const char *name,
    const char *type_name,
    const char *symbol,
    size_t size,
    size_t alignment,
    bool is_component,
    bool *existing_out);

FLECS_API
void ecs_cpp_enum_init(
    ecs_world_t *world,
    ecs_entity_t id,
    ecs_entity_t underlying_type);

FLECS_API
ecs_entity_t ecs_cpp_enum_constant_register(
    ecs_world_t *world,
    ecs_entity_t parent,
    ecs_entity_t id,
    const char *name,
    void *value,
    ecs_entity_t value_type,
    size_t value_size);

#ifdef FLECS_META
FLECS_API
const ecs_member_t* ecs_cpp_last_member(
    const ecs_world_t *world, 
    ecs_entity_t type);
#endif

#ifdef __cplusplus
}
#endif

#endif // FLECS_CPP_H

#endif // FLECS_CPP


#ifdef __cplusplus
/**
 * @file addons/cpp/flecs.hpp
 * @brief Flecs C++11 API.
 */

#pragma once

// STL includes
#include <type_traits>

/**
 * @defgroup cpp C++ API
 * @{
 */

namespace flecs
{

struct world;
struct world_async_stage;
struct iter;
struct entity_view;
struct entity;
struct type;
struct table;
struct table_range;
struct untyped_component;

template <typename T>
struct component;

template <typename T>
struct ref;

namespace _
{
template <typename T, typename U = int>
struct type;

template <typename Func, typename ... Components>
struct each_delegate;

} // namespace _
} // namespace flecs

// Types imported from C API
/**
 * @file addons/cpp/c_types.hpp
 * @brief Aliases for types/constants from C API
 */

#pragma once

namespace flecs {

/**
 * @defgroup cpp_globals API Types & Globals
 * @ingroup cpp_core
 * Types & constants bridged from C API.
 *
 * @{
 */

using world_t = ecs_world_t;
using world_info_t = ecs_world_info_t;
using id_t = ecs_id_t;
using entity_t = ecs_entity_t;
using type_t = ecs_type_t;
using table_t = ecs_table_t;
using term_t = ecs_term_t;
using query_t = ecs_query_t;
using query_group_info_t = ecs_query_group_info_t;
using observer_t = ecs_observer_t;
using iter_t = ecs_iter_t;
using ref_t = ecs_ref_t;
using type_info_t = ecs_type_info_t;
using type_hooks_t = ecs_type_hooks_t;
using flags32_t = ecs_flags32_t;

enum inout_kind_t {
    InOutDefault = EcsInOutDefault,
    InOutNone = EcsInOutNone,
    InOutFilter = EcsInOutFilter,
    InOut = EcsInOut,
    In = EcsIn,
    Out = EcsOut
};

enum oper_kind_t {
    And = EcsAnd,
    Or = EcsOr,
    Not = EcsNot,
    Optional = EcsOptional,
    AndFrom = EcsAndFrom,
    OrFrom = EcsOrFrom,
    NotFrom = EcsNotFrom
};

enum query_cache_kind_t {
    QueryCacheDefault = EcsQueryCacheDefault,
    QueryCacheAuto = EcsQueryCacheAuto,
    QueryCacheAll = EcsQueryCacheAll,
    QueryCacheNone = EcsQueryCacheNone
};

/** Id bit flags */
static const flecs::entity_t PAIR = ECS_PAIR;
static const flecs::entity_t AUTO_OVERRIDE = ECS_AUTO_OVERRIDE;
static const flecs::entity_t TOGGLE = ECS_TOGGLE;

////////////////////////////////////////////////////////////////////////////////
//// Builtin components and tags
////////////////////////////////////////////////////////////////////////////////

/* Builtin components */
using Component = EcsComponent;
using Identifier = EcsIdentifier;
using Poly = EcsPoly;
using DefaultChildComponent = EcsDefaultChildComponent;

/* Builtin tags */
static const flecs::entity_t Query = EcsQuery;
static const flecs::entity_t Observer = EcsObserver;
static const flecs::entity_t Private = EcsPrivate;
static const flecs::entity_t Module = EcsModule;
static const flecs::entity_t Prefab = EcsPrefab;
static const flecs::entity_t Disabled = EcsDisabled;
static const flecs::entity_t Empty = EcsEmpty;
static const flecs::entity_t Monitor = EcsMonitor;
static const flecs::entity_t System = EcsSystem;
static const flecs::entity_t Pipeline = ecs_id(EcsPipeline);
static const flecs::entity_t Phase = EcsPhase;

/* Builtin event tags */
static const flecs::entity_t OnAdd = EcsOnAdd;
static const flecs::entity_t OnRemove = EcsOnRemove;
static const flecs::entity_t OnSet = EcsOnSet;
static const flecs::entity_t OnTableCreate = EcsOnTableCreate;
static const flecs::entity_t OnTableDelete = EcsOnTableDelete;

/* Builtin term flags */
static const uint64_t Self = EcsSelf;
static const uint64_t Up = EcsUp;
static const uint64_t Trav = EcsTrav;
static const uint64_t Cascade = EcsCascade;
static const uint64_t Desc = EcsDesc;
static const uint64_t IsVariable = EcsIsVariable;
static const uint64_t IsEntity = EcsIsEntity;
static const uint64_t IsName = EcsIsName;
static const uint64_t TraverseFlags = EcsTraverseFlags;
static const uint64_t TermRefFlags = EcsTermRefFlags;

/* Builtin entity ids */
static const flecs::entity_t Flecs = EcsFlecs;
static const flecs::entity_t FlecsCore = EcsFlecsCore;
static const flecs::entity_t World = EcsWorld;

/* Component traits */
static const flecs::entity_t Wildcard = EcsWildcard;
static const flecs::entity_t Any = EcsAny;
static const flecs::entity_t This = EcsThis;
static const flecs::entity_t Transitive = EcsTransitive;
static const flecs::entity_t Reflexive = EcsReflexive;
static const flecs::entity_t Final = EcsFinal;
static const flecs::entity_t PairIsTag = EcsPairIsTag;
static const flecs::entity_t Exclusive = EcsExclusive;
static const flecs::entity_t Acyclic = EcsAcyclic;
static const flecs::entity_t Traversable = EcsTraversable;
static const flecs::entity_t Symmetric = EcsSymmetric;
static const flecs::entity_t With = EcsWith;
static const flecs::entity_t OneOf = EcsOneOf;
static const flecs::entity_t Trait = EcsTrait;
static const flecs::entity_t Relationship = EcsRelationship;
static const flecs::entity_t Target = EcsTarget;
static const flecs::entity_t CanToggle = EcsCanToggle;

/* OnInstantiate trait */
static const flecs::entity_t OnInstantiate = EcsOnInstantiate;
static const flecs::entity_t Override = EcsOverride;
static const flecs::entity_t Inherit = EcsInherit;
static const flecs::entity_t DontInherit = EcsDontInherit;

/* OnDelete/OnDeleteTarget traits */
static const flecs::entity_t OnDelete = EcsOnDelete;
static const flecs::entity_t OnDeleteTarget = EcsOnDeleteTarget;
static const flecs::entity_t Remove = EcsRemove;
static const flecs::entity_t Delete = EcsDelete;
static const flecs::entity_t Panic = EcsPanic;

/* Builtin relationships */
static const flecs::entity_t IsA = EcsIsA;
static const flecs::entity_t ChildOf = EcsChildOf;
static const flecs::entity_t DependsOn = EcsDependsOn;
static const flecs::entity_t SlotOf = EcsSlotOf;

/* Builtin identifiers */
static const flecs::entity_t Name = EcsName;
static const flecs::entity_t Symbol = EcsSymbol;

/* Storage */
static const flecs::entity_t Sparse = EcsSparse;
static const flecs::entity_t Union = EcsUnion;

/* Builtin predicates for comparing entity ids in queries. */
static const flecs::entity_t PredEq = EcsPredEq;
static const flecs::entity_t PredMatch = EcsPredMatch;
static const flecs::entity_t PredLookup = EcsPredLookup;

/* Builtin marker entities for query scopes */
static const flecs::entity_t ScopeOpen = EcsScopeOpen;
static const flecs::entity_t ScopeClose = EcsScopeClose;

/** @} */

}


// C++ utilities
/**
 * @file addons/cpp/utils/utils.hpp
 * @brief Flecs STL (FTL?)
 * 
 * Flecs STL (FTL?)
 * Minimalistic utilities that allow for STL like functionality without having
 * to depend on the actual STL.
 */

// Macros so that C++ new calls can allocate using ecs_os_api memory allocation functions
// Rationale:
//  - Using macros here instead of a templated function bc clients might override ecs_os_malloc
//    to contain extra debug info like source tracking location. Using a template function
//    in that scenario would collapse all source location into said function vs. the
//    actual call site
//  - FLECS_PLACEMENT_NEW(): exists to remove any naked new calls/make it easy to identify any regressions
//    by grepping for new/delete

#define FLECS_PLACEMENT_NEW(_ptr, _type)  ::new(flecs::_::placement_new_tag, _ptr) _type
#define FLECS_NEW(_type)                  FLECS_PLACEMENT_NEW(ecs_os_malloc(sizeof(_type)), _type)
#define FLECS_DELETE(_ptr)          \
  do {                              \
    if (_ptr) {                     \
      flecs::_::destruct_obj(_ptr); \
      ecs_os_free(_ptr);            \
    }                               \
  } while (false)

/* Faster (compile time) alternatives to std::move / std::forward. From:
 *   https://www.foonathan.net/2020/09/move-forward/
 */

#define FLECS_MOV(...) \
  static_cast<flecs::remove_reference_t<decltype(__VA_ARGS__)>&&>(__VA_ARGS__)

#define FLECS_FWD(...) \
  static_cast<decltype(__VA_ARGS__)&&>(__VA_ARGS__)

namespace flecs 
{

namespace _
{

// Dummy Placement new tag to disambiguate from any other operator new overrides
struct placement_new_tag_t{};
constexpr placement_new_tag_t placement_new_tag{};
template<class Ty> inline void destruct_obj(Ty* _ptr) { _ptr->~Ty(); }
template<class Ty> inline void free_obj(void* _ptr) { 
    if (_ptr) {
        destruct_obj(static_cast<Ty*>(_ptr)); 
        ecs_os_free(_ptr); 
    }
}

} // namespace _

} // namespace flecs

// Allows overriding flecs_static_assert, which is useful when testing
#ifndef flecs_static_assert
#define flecs_static_assert(cond, str) static_assert(cond, str)
#endif

inline void* operator new(size_t,   flecs::_::placement_new_tag_t, void* _ptr) noexcept { return _ptr; }
inline void  operator delete(void*, flecs::_::placement_new_tag_t, void*)      noexcept {              }

namespace flecs
{

// C++11/C++14 convenience template replacements

template <bool V, typename T, typename F>
using conditional_t = typename std::conditional<V, T, F>::type;

template <typename T>
using decay_t = typename std::decay<T>::type;

template <bool V, typename T = void>
using enable_if_t = typename std::enable_if<V, T>::type;

template <typename T>
using remove_pointer_t = typename std::remove_pointer<T>::type;

template <typename T>
using remove_reference_t = typename std::remove_reference<T>::type;

template <typename T>
using underlying_type_t = typename std::underlying_type<T>::type;

using std::is_base_of;
using std::is_empty;
using std::is_const;
using std::is_pointer;
using std::is_reference;
using std::is_volatile;
using std::is_same;
using std::is_enum;

// Determine constness even if T is a pointer type
template <typename T>
using is_const_p = is_const< remove_pointer_t<T> >;

// Apply cv modifiers from source type to destination type
// (from: https://stackoverflow.com/questions/52559336/add-const-to-type-if-template-arg-is-const)
template<class Src, class Dst>
using transcribe_const_t = conditional_t<is_const<Src>::value, Dst const, Dst>;

template<class Src, class Dst>
using transcribe_volatile_t = conditional_t<is_volatile<Src>::value, Dst volatile, Dst>;

template<class Src, class Dst>
using transcribe_cv_t = transcribe_const_t< Src, transcribe_volatile_t< Src, Dst> >;

template<class Src, class Dst>
using transcribe_pointer_t = conditional_t<is_pointer<Src>::value, Dst*, Dst>;

template<class Src, class Dst>
using transcribe_cvp_t = transcribe_cv_t< Src, transcribe_pointer_t< Src, Dst> >;


// More convenience templates. The if_*_t templates use int as default type
// instead of void. This enables writing code that's a bit less cluttered when
// the templates are used in a template declaration:
//
//     enable_if_t<true>* = nullptr
// vs:
//     if_t<true> = 0

template <bool V>
using if_t = enable_if_t<V, int>;

template <bool V>
using if_not_t = enable_if_t<false == V, int>;

namespace _
{

// Utility to prevent static assert from immediately triggering
template <class... T>
struct always_false {
    static const bool value = false;
};

} // namespace _

} // namespace flecs

#include <stdlib.h>
/**
 * @file addons/cpp/utils/array.hpp
 * @brief Array class.
 * 
 * Array class. Simple std::array like utility that is mostly there to aid
 * template code where template expansion would lead to an array with size 0.
 */

namespace flecs {

template <typename T>
struct array_iterator
{
    explicit array_iterator(T* value, int index) {
        value_ = value;
        index_ = index;
    }

    bool operator!=(array_iterator const& other) const
    {
        return index_ != other.index_;
    }

    T & operator*() const
    {
        return value_[index_];
    }

    array_iterator& operator++()
    {
        ++index_;
        return *this;
    }

private:
    T* value_;
    int index_;
};

template <typename T, size_t Size, class Enable = void> 
struct array final { };

template <typename T, size_t Size>
struct array<T, Size, enable_if_t<Size != 0> > final {
    array() {};

    array(const T (&elems)[Size]) {
        int i = 0;
        for (auto it = this->begin(); it != this->end(); ++ it) {
            *it = elems[i ++];
        }
    }

    T& operator[](int index) {
        return array_[index];
    }

    T& operator[](size_t index) {
        return array_[index];
    }

    array_iterator<T> begin() {
        return array_iterator<T>(array_, 0);
    }

    array_iterator<T> end() {
        return array_iterator<T>(array_, Size);
    }

    size_t size() {
        return Size;
    }

    T* ptr() {
        return array_;
    }

    template <typename Func>
    void each(const Func& func) {
        for (auto& elem : *this) {
            func(elem);
        }
    }

private:
    T array_[Size];
};

template<typename T, size_t Size>
array<T, Size> to_array(const T (&elems)[Size]) {
    return array<T, Size>(elems);
}

// Specialized class for zero-sized array
template <typename T, size_t Size>
struct array<T, Size, enable_if_t<Size == 0>> final {
    array() {};
    array(const T* (&elems)) { (void)elems; }
    T operator[](size_t index) { ecs_os_abort(); (void)index; return T(); }
    array_iterator<T> begin() { return array_iterator<T>(nullptr, 0); }
    array_iterator<T> end() { return array_iterator<T>(nullptr, 0); }

    size_t size() {
        return 0;
    }

    T* ptr() {
        return NULL;
    }
};

}

/**
 * @file addons/cpp/utils/string.hpp
 * @brief String utility that doesn't implicitly allocate memory.
 */

namespace flecs {

struct string_view;

// This removes dependencies on std::string (and therefore STL) and allows the 
// API to return allocated strings without incurring additional allocations when
// wrapping in an std::string.
struct string {
    explicit string() 
        : str_(nullptr)
        , const_str_("")
        , length_(0) { }

    explicit string(char *str) 
        : str_(str)
        , const_str_(str ? str : "")
        , length_(str ? ecs_os_strlen(str) : 0) { }

    ~string() {
        // If flecs is included in a binary but is not used, it is possible that
        // the OS API is not initialized. Calling ecs_os_free in that case could
        // crash the application during exit. However, if a string has been set
        // flecs has been used, and OS API should have been initialized.
        if (str_) {
            ecs_os_free(str_);
        }
    }

    string(string&& str) noexcept {
        ecs_os_free(str_);
        str_ = str.str_;
        const_str_ = str.const_str_;
        length_ = str.length_;
        str.str_ = nullptr;
    }

    operator const char*() const {
        return const_str_;
    }

    string& operator=(string&& str) noexcept {
        ecs_os_free(str_);
        str_ = str.str_;
        const_str_ = str.const_str_;
        length_ = str.length_;
        str.str_ = nullptr;
        return *this;
    }

    // Ban implicit copies/allocations
    string& operator=(const string& str) = delete;
    string(const string& str) = delete;

    bool operator==(const flecs::string& str) const {
        if (str.const_str_ == const_str_) {
            return true;
        }

        if (!const_str_ || !str.const_str_) {
            return false;
        }

        if (str.length_ != length_) {
            return false;
        }

        return ecs_os_strcmp(str, const_str_) == 0;
    }

    bool operator!=(const flecs::string& str) const {
        return !(*this == str);
    }    

    bool operator==(const char *str) const {
        if (const_str_ == str) {
            return true;
        }

        if (!const_str_ || !str) {
            return false;
        }

        return ecs_os_strcmp(str, const_str_) == 0;
    }

    bool operator!=(const char *str) const {
        return !(*this == str);
    }    

    const char* c_str() const {
        return const_str_;
    }

    std::size_t length() const {
        return static_cast<std::size_t>(length_);
    }

    template <size_t N>
    static constexpr size_t length( char const (&)[N] ) {
        return N - 1;
    }

    std::size_t size() const {
        return length();
    }

    void clear() {
        ecs_os_free(str_);
        str_ = nullptr;
        const_str_ = nullptr;
    }

    bool contains(const char *substr) {
        if (const_str_) {
            return strstr(const_str_, substr) != nullptr;
        } else {
            return false;
        }
    }

protected:
    // Must be constructed through string_view. This allows for using the string
    // class for both owned and non-owned strings, which can reduce allocations
    // when code conditionally should store a literal or an owned string.
    // Making this constructor private forces the code to explicitly create a
    // string_view which emphasizes that the string won't be freed by the class.
    string(const char *str)
        : str_(nullptr)
        , const_str_(str ? str : "")
        , length_(str ? ecs_os_strlen(str) : 0) { }

    char *str_ = nullptr;
    const char *const_str_;
    ecs_size_t length_;
};

// For consistency, the API returns a string_view where it could have returned
// a const char*, so an application won't have to think about whether to call
// c_str() or not. The string_view is a thin wrapper around a string that forces
// the API to indicate explicitly when a string is owned or not.
struct string_view : string {
    explicit string_view(const char *str)
        : string(str) { }
};

}

/**
 * @file addons/cpp/utils/enum.hpp
 * @brief Compile time enum reflection utilities.
 * 
 * Discover at compile time valid enumeration constants for an enumeration type
 * and their names. This is used to automatically register enum constants.
 */

#include <string.h>
#include <limits>

// 126, so that FLECS_ENUM_MAX_COUNT is 127 which is the largest value 
// representable by an int8_t.
#define FLECS_ENUM_MAX(T) _::to_constant<T, 126>::value
#define FLECS_ENUM_MAX_COUNT (FLECS_ENUM_MAX(int) + 1)

#ifndef FLECS_CPP_ENUM_REFLECTION_SUPPORT
#if !defined(__clang__) && defined(__GNUC__)
#if __GNUC__ > 7 || (__GNUC__ == 7 && __GNUC_MINOR__ >= 5)
#define FLECS_CPP_ENUM_REFLECTION_SUPPORT 1
#else
#define FLECS_CPP_ENUM_REFLECTION_SUPPORT 0
#endif
#else
#define FLECS_CPP_ENUM_REFLECTION_SUPPORT 1
#endif
#endif

#if defined(__clang__) && __clang_major__ >= 16
// https://reviews.llvm.org/D130058, https://reviews.llvm.org/D131307
#define flecs_enum_cast(T, v) __builtin_bit_cast(T, v)
#elif defined(__GNUC__) && __GNUC__ > 10
#define flecs_enum_cast(T, v) __builtin_bit_cast(T, v)
#else
#define flecs_enum_cast(T, v) static_cast<T>(v)
#endif

namespace flecs {

/** Int to enum */
namespace _ {
template <typename E, underlying_type_t<E> Value>
struct to_constant {
    static constexpr E value = flecs_enum_cast(E, Value);
};

template <typename E, underlying_type_t<E> Value>
constexpr E to_constant<E, Value>::value;
}

/** Convenience type with enum reflection data */
template <typename E>
struct enum_data;

template <typename E>
static enum_data<E> enum_type(flecs::world_t *world);

template <typename E>
struct enum_last {
    static constexpr E value = FLECS_ENUM_MAX(E);
};

/* Utility macro to override enum_last trait */
#define FLECS_ENUM_LAST(T, Last)\
    namespace flecs {\
    template<>\
    struct enum_last<T> {\
        static constexpr T value = Last;\
    };\
    }

namespace _ {

#if INTPTR_MAX == INT64_MAX
    #ifdef ECS_TARGET_MSVC
        #if _MSC_VER >= 1929
            #define ECS_SIZE_T_STR "unsigned __int64"
        #else
            #define ECS_SIZE_T_STR "unsigned int"
        #endif 
    #elif defined(__clang__)
        #define ECS_SIZE_T_STR "size_t"
    #else
        #ifdef ECS_TARGET_WINDOWS
            #define ECS_SIZE_T_STR "constexpr size_t; size_t = long long unsigned int"
        #else
            #define ECS_SIZE_T_STR "constexpr size_t; size_t = long unsigned int"
        #endif
    #endif
#else
    #ifdef ECS_TARGET_MSVC
        #if _MSC_VER >= 1929
            #define ECS_SIZE_T_STR "unsigned __int32"
        #else
            #define ECS_SIZE_T_STR "unsigned int"
        #endif 
    #elif defined(__clang__)
        #define ECS_SIZE_T_STR "size_t"
    #else
        #ifdef ECS_TARGET_WINDOWS
            #define ECS_SIZE_T_STR "constexpr size_t; size_t = unsigned int"
        #else
            #define ECS_SIZE_T_STR "constexpr size_t; size_t = unsigned int"
        #endif
    #endif
#endif

template <typename E>
constexpr size_t enum_type_len() {
    return ECS_FUNC_TYPE_LEN(, enum_type_len, ECS_FUNC_NAME) 
        - (sizeof(ECS_SIZE_T_STR) - 1u);
}

/** Test if value is valid for enumeration.
 * This function leverages that when a valid value is provided, 
 * __PRETTY_FUNCTION__ contains the enumeration name, whereas if a value is
 * invalid, the string contains a number or a negative (-) symbol. */
#if defined(ECS_TARGET_CLANG)
#if ECS_CLANG_VERSION < 13
template <typename E, E C>
constexpr bool enum_constant_is_valid() {
    return !((
        (ECS_FUNC_NAME[ECS_FUNC_NAME_FRONT(bool, enum_constant_is_valid) +
            enum_type_len<E>() + 6 /* ', C = ' */] >= '0') &&
        (ECS_FUNC_NAME[ECS_FUNC_NAME_FRONT(bool, enum_constant_is_valid) +
            enum_type_len<E>() + 6 /* ', C = ' */] <= '9')) ||
        (ECS_FUNC_NAME[ECS_FUNC_NAME_FRONT(bool, enum_constant_is_valid) +
            enum_type_len<E>() + 6 /* ', C = ' */] == '-'));
}
#else
template <typename E, E C>
constexpr bool enum_constant_is_valid() {
    return (ECS_FUNC_NAME[ECS_FUNC_NAME_FRONT(bool, enum_constant_is_valid) +
        enum_type_len<E>() + 6 /* ', E C = ' */] != '(');
}
#endif
#elif defined(ECS_TARGET_GNU)
template <typename E, E C>
constexpr bool enum_constant_is_valid() {
    return (ECS_FUNC_NAME[ECS_FUNC_NAME_FRONT(constexpr bool, enum_constant_is_valid) +
        enum_type_len<E>() + 8 /* ', E C = ' */] != '(');
}
#else
/* Use different trick on MSVC, since it uses hexadecimal representation for
 * invalid enum constants. We can leverage that msvc inserts a C-style cast
 * into the name, and the location of its first character ('(') is known. */
template <typename E, E C>
constexpr bool enum_constant_is_valid() {
    return ECS_FUNC_NAME[ECS_FUNC_NAME_FRONT(bool, enum_constant_is_valid) +
        enum_type_len<E>() + 1] != '(';
}
#endif

/* Without this wrapper __builtin_bit_cast doesn't work */
template <typename E, underlying_type_t<E> C>
constexpr bool enum_constant_is_valid_wrap() {
    return enum_constant_is_valid<E, flecs_enum_cast(E, C)>();
}

template <typename E, E C>
struct enum_is_valid {
    static constexpr bool value = enum_constant_is_valid<E, C>();
};

/** Extract name of constant from string */
template <typename E, E C>
static const char* enum_constant_to_name() {
    static const size_t len = ECS_FUNC_TYPE_LEN(const char*, enum_constant_to_name, ECS_FUNC_NAME);
    static char result[len + 1] = {};
    return ecs_cpp_get_constant_name(
        result, ECS_FUNC_NAME, string::length(ECS_FUNC_NAME),
            ECS_FUNC_NAME_BACK);
}

/** Enumeration constant data */
template<typename T>
struct enum_constant_data {
    int32_t index; // Global index used to obtain world local entity id
    T offset;
};

/**
 * @brief Provides utilities for enum reflection.
 * 
 * This struct provides static functions for enum reflection, including conversion
 * between enum values and their underlying integral types, and iteration over enum
 * values.
 * 
 * @tparam E The enum type.
 * @tparam Handler The handler for enum reflection operations.
 */
template <typename E, typename Handler>
struct enum_reflection {
    using U = underlying_type_t<E>;

    /**
     * @brief Iterates over the range [Low, High] of enum values between Low and High.
     *
     * Recursively divide and conquers the search space to reduce the template-depth. Once
     * recursive division is complete, calls Handle<E>::handle_constant in ascending order,
     * passing the values computed up the chain.
     * 
     * @tparam Low The lower bound of the search range, inclusive.
     * @tparam High The upper bound of the search range, inclusive.
     * @tparam Args Additional arguments to be passed through to Handler::handle_constant
     * @param last_value The last value processed in the iteration.
     * @param args Additional arguments to be passed through to Handler::handle_constant
     * @return constexpr U The result of the iteration.
     */
    template <U Low, U High, typename... Args>
    static constexpr U each_enum_range(U last_value, Args... args) {
        return High - Low <= 1
            ? High == Low
                ? Handler::template handle_constant<Low>(last_value, args...)
                : Handler::template handle_constant<High>(Handler::template handle_constant<Low>(last_value, args...), args...)
            : each_enum_range<(Low + High) / 2 + 1, High>(
                    each_enum_range<Low, (Low + High) / 2>(last_value, args...),
                    args...
              );
    }

    /**
     * @brief Iterates over the mask range (Low, High] of enum values between Low and High.
     *
     * Recursively iterates the search space, looking for enums defined as multiple-of-2 
     * bitmasks. Each iteration, shifts bit to the right until it hits Low, then calls
     * Handler::handle_constant for each bitmask in ascending order.
     * 
     * @tparam Low The lower bound of the search range, not inclusive
     * @tparam High The upper bound of the search range, inclusive.
     * @tparam Args Additional arguments to be passed through to Handler::handle_constant
     * @param last_value The last value processed in the iteration.
     * @param args Additional arguments to be passed through to Handler::handle_constant
     * @return constexpr U The result of the iteration.
     */
    template <U Low, U High, typename... Args>
    static constexpr U each_mask_range(U last_value, Args... args) {
        // If Low shares any bits with Current Flag, or if High is less than/equal to Low (and High isn't negative because max-flag signed)
        return (Low & High) || (High <= Low && High != high_bit)
            ? last_value
            : Handler::template handle_constant<High>(
                each_mask_range<Low, ((High >> 1) & ~high_bit)>(last_value, args...),
                args...
              );
    }

    /**
     * @brief Handles enum iteration for gathering reflection data.
     *
     * Iterates over all enum values up to a specified maximum value 
     * (each_enum_range<0, Value>), then iterates the rest of the possible bitmasks
     * (each_mask_range<Value, high_bit>).
     * 
     * @tparam Value The maximum enum value to iterate up to.
     * @tparam Args Additional arguments to be passed through to Handler::handle_constant
     * @param args Additional arguments to be passed through to Handler::handle_constant
     * @return constexpr U The result of the iteration.
     */
    template <U Value = static_cast<U>(FLECS_ENUM_MAX(E)), typename... Args>
    static constexpr U each_enum(Args... args) {
        return each_mask_range<Value, high_bit>(each_enum_range<0, Value>(0, args...), args...);
    }

    static const U high_bit = static_cast<U>(1) << (sizeof(U) * 8 - 1);
};

/** Enumeration type data */
template<typename E>
struct enum_data_impl {
private:
    using U = underlying_type_t<E>;

    /**
     * @brief Handler struct for generating compile-time count of enum constants.
     */
    struct reflection_count {
        template <U Value, flecs::if_not_t< enum_constant_is_valid_wrap<E, Value>() > = 0>
        static constexpr U handle_constant(U last_value) {
            return last_value;
        }

        template <U Value, flecs::if_t< enum_constant_is_valid_wrap<E, Value>() > = 0>
        static constexpr U handle_constant(U last_value) {
            return 1 + last_value;
        }
    };

public:
    int min;
    int max;
    bool has_contiguous;
	// If enum constants start not-sparse, contiguous_until will be the index of the first sparse value, or end of the constants array
    U contiguous_until;
	// Compile-time generated count of enum constants.
    static constexpr unsigned int constants_size = enum_reflection<E, reflection_count>::template each_enum< static_cast<U>(enum_last<E>::value) >();
    // Constants array is sized to the number of found-constants, or 1 (to avoid 0-sized array)
    enum_constant_data<U> constants[constants_size? constants_size: 1];
};

/** Class that scans an enum for constants, extracts names & creates entities */
template <typename E>
struct enum_type {
private:
    using U = underlying_type_t<E>;

    /**
     * @brief Helper struct for filling enum_type's static `enum_data_impl<E>` member with reflection data.
     *
     * Because reflection occurs in-order, we can use current value/last value to determine continuity, and
     * use that as a lookup heuristic later on.
     */
    struct reflection_init {
        template <U Value, flecs::if_not_t< enum_constant_is_valid_wrap<E, Value>() > = 0>
        static U handle_constant(U last_value, flecs::world_t*) {
            // Search for constant failed. Pass last valid value through.
            return last_value;
        }

        template <U Value, flecs::if_t< enum_constant_is_valid_wrap<E, Value>() > = 0>
        static U handle_constant(U last_value, flecs::world_t *world) {
            // Constant is valid, so fill reflection data.
            auto v = Value;
            const char *name = enum_constant_to_name<E, flecs_enum_cast(E, Value)>();

            ++enum_type<E>::data.max; // Increment cursor as we build constants array.

            // If the enum was previously contiguous, and continues to be through the current value...
            if (enum_type<E>::data.has_contiguous && static_cast<U>(enum_type<E>::data.max) == v && enum_type<E>::data.contiguous_until == v) {
                ++enum_type<E>::data.contiguous_until;
            }
            // else, if the enum was never contiguous and hasn't been set as not contiguous...
            else if (!enum_type<E>::data.contiguous_until && enum_type<E>::data.has_contiguous) {
                enum_type<E>::data.has_contiguous = false;
            }

            ecs_assert(!(last_value > 0 && v < std::numeric_limits<U>::min() + last_value), ECS_UNSUPPORTED,
                "Signed integer enums causes integer overflow when recording offset from high positive to"
                " low negative. Consider using unsigned integers as underlying type.");
            enum_type<E>::data.constants[enum_type<E>::data.max].offset = v - last_value;
            if (!enum_type<E>::data.constants[enum_type<E>::data.max].index) {
                enum_type<E>::data.constants[enum_type<E>::data.max].index = 
                    flecs_component_ids_index_get();
            }
            
            flecs::entity_t constant = ecs_cpp_enum_constant_register(
                world, type<E>::id(world), 0, name, &v, type<U>::id(world), sizeof(U));
            flecs_component_ids_set(world, 
                enum_type<E>::data.constants[enum_type<E>::data.max].index, 
                constant);

            return v;
        }
    };
public:

    static enum_data_impl<E> data;

    static enum_type<E>& get() {
        static _::enum_type<E> instance;
        return instance;
    }

    flecs::entity_t entity(E value) const {
        int index = index_by_value(value);
        if (index >= 0) {
            return data.constants[index].id;
        }
        return 0;
    }

    void init(flecs::world_t *world, flecs::entity_t id) {
#if !FLECS_CPP_ENUM_REFLECTION_SUPPORT
        ecs_abort(ECS_UNSUPPORTED, "enum reflection requires gcc 7.5 or higher")
#endif
        // Initialize/reset reflection data values to default state.
        data.min = 0;
        data.max = -1;
        data.has_contiguous = true;
        data.contiguous_until = 0;

        ecs_log_push();
        ecs_cpp_enum_init(world, id, type<U>::id(world));
        // data.id = id;

        // Generate reflection data
        enum_reflection<E, reflection_init>::template each_enum< static_cast<U>(enum_last<E>::value) >(world);
        ecs_log_pop();
    }
};

template <typename E>
enum_data_impl<E> enum_type<E>::data;

template <typename E, if_t< is_enum<E>::value > = 0>
inline static void init_enum(flecs::world_t *world, flecs::entity_t id) {
    _::enum_type<E>::get().init(world, id);
}

template <typename E, if_not_t< is_enum<E>::value > = 0>
inline static void init_enum(flecs::world_t*, flecs::entity_t) { }

} // namespace _

/** Enumeration type data wrapper with world pointer */
template <typename E>
struct enum_data {
    using U = underlying_type_t<E>;

    enum_data(flecs::world_t *world, _::enum_data_impl<E>& impl)
        : world_(world)
        , impl_(impl) { }
    
	/**
     * @brief Checks if a given integral value is a valid enum value.
     * 
     * @param value The integral value.
     * @return true If the value is a valid enum value.
     * @return false If the value is not a valid enum value.
     */
    bool is_valid(U value) {
        int index = index_by_value(value);
        if (index < 0) {
            return false;
        }
        return impl_.constants[index].index != 0;
    }

    /**
     * @brief Checks if a given enum value is valid.
     * 
     * @param value The enum value.
     * @return true If the value is valid.
     * @return false If the value is not valid.
     */
    bool is_valid(E value) {
        return is_valid(static_cast<U>(value));
    }

    /**
     * @brief Finds the index into the constants array for a value, if one exists
     * 
     * @param value The enum value.
     * @return int The index of the enum value.
     */
    int index_by_value(U value) const {
        if (impl_.max < 0) {
            return -1;
        }
        // Check if value is in contiguous lookup section
        if (impl_.has_contiguous && value < impl_.contiguous_until && value >= 0) {
            return static_cast<int>(value);
        }
        U accumulator = impl_.contiguous_until? impl_.contiguous_until - 1: 0;
        for (int i = static_cast<int>(impl_.contiguous_until); i <= impl_.max; ++i) {
            accumulator += impl_.constants[i].offset;
            if (accumulator == value) {
                return i;
            }
        }
        return -1;
    }

    /**
     * @brief Finds the index into the constants array for an enum value, if one exists
     * 
     * @param value The enum value.
     * @return int The index of the enum value.
     */
    int index_by_value(E value) const {
        return index_by_value(static_cast<U>(value));
    }

    int first() const {
        return impl_.min;
    }

    int last() const {
        return impl_.max;
    }

    int next(int cur) const {
        return cur + 1;
    }

    flecs::entity entity() const;
    flecs::entity entity(U value) const;
    flecs::entity entity(E value) const;

    flecs::world_t *world_;
    _::enum_data_impl<E>& impl_;
};

/** Convenience function for getting enum reflection data */
template <typename E>
enum_data<E> enum_type(flecs::world_t *world) {
    _::type<E>::id(world); // Ensure enum is registered
    auto& ref = _::enum_type<E>::get();
    return enum_data<E>(world, ref.data);
}

} // namespace flecs

/**
 * @file addons/cpp/utils/stringstream.hpp
 * @brief Wrapper around ecs_strbuf_t that provides a simple stringstream like API.
 */

namespace flecs {

struct stringstream {
    explicit stringstream() 
        : buf_({}) { }

    ~stringstream() {
        ecs_strbuf_reset(&buf_);
    }

    stringstream(stringstream&& str) noexcept {
        ecs_strbuf_reset(&buf_);
        buf_ = str.buf_;
        str.buf_ = {};
    }

    stringstream& operator=(stringstream&& str) noexcept {
        ecs_strbuf_reset(&buf_);
        buf_ = str.buf_;
        str.buf_ = {};
        return *this;
    }

    // Ban implicit copies/allocations
    stringstream& operator=(const stringstream& str) = delete;
    stringstream(const stringstream& str) = delete;    

    stringstream& operator<<(const char* str) {
        ecs_strbuf_appendstr(&buf_, str);
        return *this;
    }

    flecs::string str() {
        return flecs::string(ecs_strbuf_get(&buf_));
    }

private:
    ecs_strbuf_t buf_;
};

}

/**
 * @file addons/cpp/utils/function_traits.hpp
 * @brief Compile time utilities to inspect properties of functions.
 *
 * Code from: https://stackoverflow.com/questions/27024238/c-template-mechanism-to-get-the-number-of-function-arguments-which-would-work
 */

namespace flecs {
namespace _ {

template <typename ... Args>
struct arg_list { };

// Base type that contains the traits
template <typename ReturnType, typename... Args>
struct function_traits_defs
{
    static constexpr bool is_callable = true;
    static constexpr size_t arity = sizeof...(Args);
    using return_type = ReturnType;
    using args = arg_list<Args ...>;
};

// Primary template for function_traits_impl
template <typename T>
struct function_traits_impl {
    static constexpr bool is_callable = false;
};

// Template specializations for the different kinds of function types (whew)
template <typename ReturnType, typename... Args>
struct function_traits_impl<ReturnType(Args...)>
    : function_traits_defs<ReturnType, Args...> {};

template <typename ReturnType, typename... Args>
struct function_traits_impl<ReturnType(*)(Args...)>
    : function_traits_defs<ReturnType, Args...> {};

template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits_impl<ReturnType(ClassType::*)(Args...)>
    : function_traits_defs<ReturnType, Args...> {};

template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits_impl<ReturnType(ClassType::*)(Args...) const>
    : function_traits_defs<ReturnType, Args...> {};    

template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits_impl<ReturnType(ClassType::*)(Args...) const&>
    : function_traits_defs<ReturnType, Args...> {};
    
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits_impl<ReturnType(ClassType::*)(Args...) const&&>
    : function_traits_defs<ReturnType, Args...> {};
    
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits_impl<ReturnType(ClassType::*)(Args...) volatile>
    : function_traits_defs<ReturnType, Args...> {};

template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits_impl<ReturnType(ClassType::*)(Args...) volatile&>
    : function_traits_defs<ReturnType, Args...> {};
    
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits_impl<ReturnType(ClassType::*)(Args...) volatile&&>
    : function_traits_defs<ReturnType, Args...> {};

template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits_impl<ReturnType(ClassType::*)(Args...) const volatile>
    : function_traits_defs<ReturnType, Args...> {};

template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits_impl<ReturnType(ClassType::*)(Args...) const volatile&>
    : function_traits_defs<ReturnType, Args...> {};
    
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits_impl<ReturnType(ClassType::*)(Args...) const volatile&&>
    : function_traits_defs<ReturnType, Args...> {};

// Primary template for function_traits_no_cv. If T is not a function, the
// compiler will attempt to instantiate this template and fail, because its base
// is undefined.
template <typename T, typename V = void>
struct function_traits_no_cv
    : function_traits_impl<T> {};

// Specialized template for function types
template <typename T>
struct function_traits_no_cv<T, decltype((void)&T::operator())>
    : function_traits_impl<decltype(&T::operator())> {};
 
// Front facing template that decays T before ripping it apart.
template <typename T>
struct function_traits
    : function_traits_no_cv< decay_t<T> > {};

} // _


template <typename T>
struct is_callable {
    static constexpr bool value = _::function_traits<T>::is_callable;
};

template <typename T>
struct arity {
    static constexpr int value = _::function_traits<T>::arity;
};

template <typename T>
using return_type_t = typename _::function_traits<T>::return_type;

template <typename T>
using arg_list_t = typename _::function_traits<T>::args;

// First arg
template<typename Func, typename ... Args>
struct first_arg_impl;

template<typename Func, typename T, typename ... Args>
struct first_arg_impl<Func, _::arg_list<T, Args ...> > {
    using type = T;
};

template<typename Func>
struct first_arg {
    using type = typename first_arg_impl<Func, arg_list_t<Func>>::type;
};

template <typename Func>
using first_arg_t = typename first_arg<Func>::type;

// Last arg
template<typename Func, typename ... Args>
struct second_arg_impl;

template<typename Func, typename First, typename T, typename ... Args>
struct second_arg_impl<Func, _::arg_list<First, T, Args ...> > {
    using type = T;
};

template<typename Func>
struct second_arg {
    using type = typename second_arg_impl<Func, arg_list_t<Func>>::type;
};

template <typename Func>
using second_arg_t = typename second_arg<Func>::type;

} // flecs



// Mixin forward declarations
/**
 * @file addons/cpp/mixins/id/decl.hpp
 * @brief Id class.
 */

#pragma once

namespace flecs {

struct id;
struct entity;

/**
 * @defgroup cpp_ids Ids
 * @ingroup cpp_core
 * Class for working with entity, component, tag and pair ids.
 *
 * @{
 */

/** Class that wraps around a flecs::id_t.
 * A flecs id is an identifier that can be added to entities. Ids can be:
 * - entities (including components, tags)
 * - pair ids
 * - entities with id flags set (like flecs::AUTO_OVERRIDE, flecs::TOGGLE)
 */
struct id {
    id()
        : world_(nullptr)
        , id_(0) { }

    explicit id(flecs::id_t value)
        : world_(nullptr)
        , id_(value) { }

    explicit id(flecs::world_t *world, flecs::id_t value = 0)
        : world_(world)
        , id_(value) { }

    explicit id(flecs::world_t *world, flecs::id_t first, flecs::id_t second)
        : world_(world)
        , id_(ecs_pair(first, second)) { }

    explicit id(flecs::world_t *world, const char *expr)
        : world_(world)
        , id_(ecs_id_from_str(world, expr)) { }

    explicit id(flecs::id_t first, flecs::id_t second)
        : world_(nullptr)
        , id_(ecs_pair(first, second)) { }

    explicit id(const flecs::id& first, const flecs::id& second)
        : world_(first.world_)
        , id_(ecs_pair(first.id_, second.id_)) { }

    /** Test if id is pair (has first, second) */
    bool is_pair() const {
        return (id_ & ECS_ID_FLAGS_MASK) == flecs::PAIR;
    }

    /** Test if id is a wildcard */
    bool is_wildcard() const {
        return ecs_id_is_wildcard(id_);
    }

    /** Test if id is entity */
    bool is_entity() const {
        return !(id_ & ECS_ID_FLAGS_MASK);
    }

    /** Return id as entity (only allowed when id is valid entity) */
    flecs::entity entity() const;

    /** Return id with role added */
    flecs::entity add_flags(flecs::id_t flags) const;

    /** Return id with role removed */
    flecs::entity remove_flags(flecs::id_t flags) const;

    /** Return id without role */
    flecs::entity remove_flags() const;

    /** Return id without role */
    flecs::entity remove_generation() const;

    /** Return component type of id */
    flecs::entity type_id() const;

    /** Test if id has specified role */
    bool has_flags(flecs::id_t flags) const {
        return ((id_ & flags) == flags);
    }

    /** Test if id has any role */
    bool has_flags() const {
        return (id_ & ECS_ID_FLAGS_MASK) != 0;
    }

    /** Return id flags set on id */
    flecs::entity flags() const;

    /** Test if id has specified first */
    bool has_relation(flecs::id_t first) const {
        if (!is_pair()) {
            return false;
        }
        return ECS_PAIR_FIRST(id_) == first;
    }

    /** Get first element from a pair.
     * If the id is not a pair, this operation will fail. When the id has a
     * world, the operation will ensure that the returned id has the correct
     * generation count. */
    flecs::entity first() const;

    /** Get second element from a pair.
     * If the id is not a pair, this operation will fail. When the id has a
     * world, the operation will ensure that the returned id has the correct
     * generation count. */
    flecs::entity second() const;

    /* Convert id to string */
    flecs::string str() const {
        return flecs::string(ecs_id_str(world_, id_));
    }

    /** Convert role of id to string. */
    flecs::string flags_str() const {
        return flecs::string_view( ecs_id_flag_str(id_ & ECS_ID_FLAGS_MASK));
    }

    /** Return flecs::id_t value */
    flecs::id_t raw_id() const {
        return id_;
    }

    operator flecs::id_t() const {
        return id_;
    }

    flecs::world world() const;

protected:
    /* World is optional, but guarantees that entity identifiers extracted from
     * the id are valid */
    flecs::world_t *world_;
    flecs::id_t id_;
};

/** @} */

}

/**
 * @file addons/cpp/mixins/term/decl.hpp
 * @brief Term declarations.
 */

#pragma once

namespace flecs {

/**
 * @ingroup cpp_core_queries
 *
 * @{
 */

struct term;
struct term_builder;

/** @} */

}

/**
 * @file addons/cpp/mixins/query/decl.hpp
 * @brief Query declarations.
 */

#pragma once

namespace flecs {

/**
 * @defgroup cpp_core_queries Queries
 * @ingroup cpp_core
 *
 * @{
 */

struct query_base;

template<typename ... Components>
struct query;

template<typename ... Components>
struct query_builder;

/** @} */

}

/**
 * @file addons/cpp/mixins/event/decl.hpp
 * @brief Event declarations.
 */

#pragma once

/**
 * @file addons/cpp/mixins/event/builder.hpp
 * @brief Event builder.
 */

#pragma once

#define ECS_EVENT_DESC_ID_COUNT_MAX (8)

namespace flecs {

/**
 * @ingroup cpp_addons_event
 * @{
 */

/** Event builder interface */
template <typename Base, typename E>
struct event_builder_base {
    event_builder_base(flecs::world_t *world, flecs::entity_t event)
        : world_(world)
        , desc_{}
        , ids_{}
        , ids_array_{}
    {
        desc_.event = event;
    }

    /** Add component to emit for */
    template <typename T>
    Base& id() {
        ids_.array = ids_array_;
        ids_.array[ids_.count] = _::type<T>().id(world_);
        ids_.count ++;
        return *this;
    }
    
    /** 
     * Add pair to emit for
     * @tparam First The first element of the pair.
     * @tparam Second the second element of a pair.
     */
    template <typename First, typename Second>
    Base& id() {
        return id(
            ecs_pair(_::type<First>::id(this->world_), 
                _::type<Second>::id(this->world_)));
    }

    /** 
     * Add pair to emit for
     * @tparam First The first element of the pair.
     * @param second The second element of the pair id.
     */
    template <typename First>
    Base& id(entity_t second) {
        return id(ecs_pair(_::type<First>::id(this->world_), second));
    }

    /** 
     * Add pair to emit for
     * @param first The first element of the pair type.
     * @param second The second element of the pair id.
     */
    Base& id(entity_t first, entity_t second) {
        return id(ecs_pair(first, second));
    }

    template <typename Enum, if_t<is_enum<Enum>::value> = 0>
    Base& id(Enum value) {
        const auto& et = enum_type<Enum>(this->world_);
        flecs::entity_t target = et.entity(value);
        return id(et.entity(), target);
    }

    /** Add (component) id to emit for */
    Base& id(flecs::id_t id) {
        ids_.array = ids_array_;
        ids_.array[ids_.count] = id;
        ids_.count ++;
        return *this;
    }

    /** Set entity for which to emit event */
    Base& entity(flecs::entity_t e) {
        desc_.entity = e;
        return *this;
    }

    /* Set table for which to emit event */
    Base& table(flecs::table_t *t, int32_t offset = 0, int32_t count = 0) {
        desc_.table = t;
        desc_.offset = offset;
        desc_.count = count;
        return *this;
    }

    /* Set event data */
    Base& ctx(const E* ptr) {
        desc_.const_param = ptr;
        return *this;
    }

    /* Set event data */
    Base& ctx(E* ptr) {
        desc_.param = ptr;
        return *this;
    }

    void emit() {
        ids_.array = ids_array_;
        desc_.ids = &ids_;
        desc_.observable = const_cast<flecs::world_t*>(ecs_get_world(world_));
        ecs_emit(world_, &desc_);
    }

    void enqueue() {
        ids_.array = ids_array_;
        desc_.ids = &ids_;
        desc_.observable = const_cast<flecs::world_t*>(ecs_get_world(world_));
        ecs_enqueue(world_, &desc_);
    }

protected:
    flecs::world_t *world_;
    ecs_event_desc_t desc_;
    flecs::type_t ids_;
    flecs::id_t ids_array_[ECS_EVENT_DESC_ID_COUNT_MAX];

private:
    operator Base&() {
        return *static_cast<Base*>(this);
    }
};

struct event_builder : event_builder_base<event_builder, void> {
    using event_builder_base::event_builder_base;
};

template <typename E>
struct event_builder_typed : event_builder_base<event_builder_typed<E>, E> {
private:
    using Class = event_builder_typed<E>;

public:
    using event_builder_base<Class, E>::event_builder_base;

    /* Set event data */
    Class& ctx(const E& ptr) {
        this->desc_.const_param = &ptr;
        return *this;
    }

    /* Set event data */
    Class& ctx(E&& ptr) {
        this->desc_.param = &ptr;
        return *this;
    }
};

/** @} */

}


namespace flecs {
namespace _ {

// Utility to derive event type from function
template  <typename Func, typename U = int>
struct event_from_func;

// Specialization for observer callbacks with a single argument
template  <typename Func>
struct event_from_func<Func, if_t< arity<Func>::value == 1>> {
    using type = decay_t<first_arg_t<Func>>;
};

// Specialization for observer callbacks with an initial entity src argument
template  <typename Func>
struct event_from_func<Func, if_t< arity<Func>::value == 2>> {
    using type = decay_t<second_arg_t<Func>>;
};

template <typename Func>
using event_from_func_t = typename event_from_func<Func>::type;

}
}

/**
 * @file addons/cpp/mixins/observer/decl.hpp
 * @brief Observer declarations.
 */

#pragma once

namespace flecs {

/**
 * @defgroup cpp_observers Observers
 * @ingroup cpp_core
 * Observers let applications register callbacks for ECS events.
 *
 * @{
 */

struct observer;

template<typename ... Components>
struct observer_builder;

/** @} */

}

#ifdef FLECS_SYSTEM
/**
 * @file addons/cpp/mixins/system/decl.hpp
 * @brief System module declarations.
 */

#pragma once

namespace flecs {

/**
 * @defgroup cpp_addons_systems Systems
 * @ingroup cpp_addons
 * Systems are a query + function that can be ran manually or by a pipeline.
 *
 * @{
 */

using TickSource = EcsTickSource;

struct system;

template<typename ... Components>
struct system_builder;

namespace _ {

void system_init(flecs::world& world);

/** @} */

} // namespace _
} // namespace flecs

#endif
#ifdef FLECS_PIPELINE
/**
 * @file addons/cpp/mixins/pipeline/decl.hpp
 * @brief Pipeline module declarations.
 */

#pragma once

namespace flecs {

/**
 * @defgroup cpp_pipelines Pipelines
 * @ingroup cpp_addons
 * Pipelines order and schedule systems for execution.
 *
 * @{
 */

template <typename ... Components>
struct pipeline;

template <typename ... Components>
struct pipeline_builder;

/* Builtin pipeline tags */
static const flecs::entity_t OnStart = EcsOnStart;
static const flecs::entity_t PreFrame = EcsPreFrame;
static const flecs::entity_t OnLoad = EcsOnLoad;
static const flecs::entity_t PostLoad = EcsPostLoad;
static const flecs::entity_t PreUpdate = EcsPreUpdate;
static const flecs::entity_t OnUpdate = EcsOnUpdate;
static const flecs::entity_t OnValidate = EcsOnValidate;
static const flecs::entity_t PostUpdate = EcsPostUpdate;
static const flecs::entity_t PreStore = EcsPreStore;
static const flecs::entity_t OnStore = EcsOnStore;
static const flecs::entity_t PostFrame = EcsPostFrame;

/** @} */

}

#endif
#ifdef FLECS_TIMER
/**
 * @file addons/cpp/mixins/timer/decl.hpp
 * @brief Timer module declarations.
 */

#pragma once

namespace flecs {

/**
 * @defgroup cpp_addons_timer Timer
 * @ingroup cpp_addons
 * Run systems at a time interval.
 *
 * @{
 */

using Timer = EcsTimer;
using RateFilter = EcsRateFilter;

struct timer;

/** @} */

namespace _ {

void timer_init(flecs::world& world);

} // namespace _
} // namespace flecs

#endif
#ifdef FLECS_DOC
/**
 * @file addons/cpp/mixins/doc/decl.hpp
 * @brief Doc mixin declarations.
 */

#pragma once

namespace flecs {
namespace doc {

/**
 * @defgroup cpp_addons_doc Doc
 * @ingroup cpp_addons
 * Utilities for documenting entities, components and systems.
 *
 * @{
 */

/** flecs.doc.Description component */
using Description = EcsDocDescription;

/** flecs.doc.Uuid component */
static const flecs::entity_t Uuid = EcsDocUuid;

/** flecs.doc.Brief component */
static const flecs::entity_t Brief = EcsDocBrief;

/** flecs.doc.Detail component */
static const flecs::entity_t Detail = EcsDocDetail;

/** flecs.doc.Link component */
static const flecs::entity_t Link = EcsDocLink;

/** flecs.doc.Color component */
static const flecs::entity_t Color = EcsDocColor;

/** @private */
namespace _ {
/** @private */
void init(flecs::world& world);
}

/** @} */

}
}

#endif
#ifdef FLECS_REST
/**
 * @file addons/cpp/mixins/rest/decl.hpp
 * @brief Rest module declarations.
 */

#pragma once

namespace flecs {

/**
 * @defgroup cpp_addons_rest Rest
 * @ingroup cpp_addons
 * REST API for querying and mutating entities.
 *
 * @{
 */

using Rest = EcsRest;

namespace rest {

namespace _ {

void init(flecs::world& world);

}
}

/** @} */

}

#endif
#ifdef FLECS_META
/**
 * @file addons/cpp/mixins/meta/decl.hpp
 * @brief Meta declarations.
 */

#pragma once

namespace flecs {

/**
 * @defgroup cpp_addons_meta Meta
 * @ingroup cpp_addons
 * Flecs reflection framework.
 *
 * @{
 */

/* Primitive type aliases */
using bool_t = ecs_bool_t;
using char_t = ecs_char_t;
using u8_t = ecs_u8_t;
using u16_t = ecs_u16_t;
using u32_t = ecs_u32_t;
using u64_t = ecs_u64_t;
using uptr_t = ecs_uptr_t;
using i8_t = ecs_i8_t;
using i16_t = ecs_i16_t;
using i32_t = ecs_i32_t;
using i64_t = ecs_i64_t;
using iptr_t = ecs_iptr_t;
using f32_t = ecs_f32_t;
using f64_t = ecs_f64_t;

/* Embedded type aliases */
using member_t = ecs_member_t;
using enum_constant_t = ecs_enum_constant_t;
using bitmask_constant_t = ecs_bitmask_constant_t;

/* Components */
using Type = EcsType;
using TypeSerializer = EcsTypeSerializer;
using Primitive = EcsPrimitive;
using Enum = EcsEnum;
using Bitmask = EcsBitmask;
using Member = EcsMember;
using MemberRanges = EcsMemberRanges;
using Struct = EcsStruct;
using Array = EcsArray;
using Vector = EcsVector;
using Unit = EcsUnit;

/** Base type for bitmasks */
struct bitmask {
    uint32_t value;
};

/* Handles to builtin reflection types */
static const flecs::entity_t Bool = ecs_id(ecs_bool_t);
static const flecs::entity_t Char = ecs_id(ecs_char_t);
static const flecs::entity_t Byte = ecs_id(ecs_byte_t);
static const flecs::entity_t U8 = ecs_id(ecs_u8_t);
static const flecs::entity_t U16 = ecs_id(ecs_u16_t);
static const flecs::entity_t U32 = ecs_id(ecs_u32_t);
static const flecs::entity_t U64 = ecs_id(ecs_u64_t);
static const flecs::entity_t Uptr = ecs_id(ecs_uptr_t);
static const flecs::entity_t I8 = ecs_id(ecs_i8_t);
static const flecs::entity_t I16 = ecs_id(ecs_i16_t);
static const flecs::entity_t I32 = ecs_id(ecs_i32_t);
static const flecs::entity_t I64 = ecs_id(ecs_i64_t);
static const flecs::entity_t Iptr = ecs_id(ecs_iptr_t);
static const flecs::entity_t F32 = ecs_id(ecs_f32_t);
static const flecs::entity_t F64 = ecs_id(ecs_f64_t);
static const flecs::entity_t String = ecs_id(ecs_string_t);
static const flecs::entity_t Entity = ecs_id(ecs_entity_t);
static const flecs::entity_t Constant = EcsConstant;
static const flecs::entity_t Quantity = EcsQuantity;

namespace meta {

/* Type kinds supported by reflection system */
using type_kind_t = ecs_type_kind_t;
static const type_kind_t PrimitiveType = EcsPrimitiveType;
static const type_kind_t BitmaskType = EcsBitmaskType;
static const type_kind_t EnumType = EcsEnumType;
static const type_kind_t StructType = EcsStructType;
static const type_kind_t ArrayType = EcsArrayType;
static const type_kind_t VectorType = EcsVectorType;
static const type_kind_t CustomType = EcsOpaqueType;
static const type_kind_t TypeKindLast = EcsTypeKindLast;

/* Primitive type kinds supported by reflection system */
using primitive_kind_t = ecs_primitive_kind_t;
static const primitive_kind_t Bool = EcsBool;
static const primitive_kind_t Char = EcsChar;
static const primitive_kind_t Byte = EcsByte;
static const primitive_kind_t U8 = EcsU8;
static const primitive_kind_t U16 = EcsU16;
static const primitive_kind_t U32 = EcsU32;
static const primitive_kind_t U64 = EcsU64;
static const primitive_kind_t I8 = EcsI8;
static const primitive_kind_t I16 = EcsI16;
static const primitive_kind_t I32 = EcsI32;
static const primitive_kind_t I64 = EcsI64;
static const primitive_kind_t F32 = EcsF32;
static const primitive_kind_t F64 = EcsF64;
static const primitive_kind_t UPtr = EcsUPtr;
static const primitive_kind_t IPtr = EcsIPtr;
static const primitive_kind_t String = EcsString;
static const primitive_kind_t Entity = EcsEntity;
static const primitive_kind_t PrimitiveKindLast = EcsPrimitiveKindLast;

/** @} */

namespace _ {

void init(flecs::world& world);

} // namespace _
} // namespace meta
} // namespace flecs

/**
 * @file addons/cpp/mixins/meta/opaque.hpp
 * @brief Helpers for opaque type registration.
 */

#pragma once

namespace flecs {

/**
 * @defgroup cpp_addons_meta Meta
 * @ingroup cpp_addons
 * Flecs reflection framework.
 *
 * @{
 */

/** Class for reading/writing dynamic values.
 *
 * @ingroup cpp_addons_meta
 */
struct cursor {
    cursor(flecs::world_t *world, flecs::entity_t type_id, void *ptr) {
        cursor_ = ecs_meta_cursor(world, type_id, ptr);
    }

    /** Push value scope (such as a nested struct) */
    int push() {
        return ecs_meta_push(&cursor_);
    }

    /** Pop value scope */
    int pop() {
        return ecs_meta_pop(&cursor_);
    }

    /** Move to next member/element */
    int next() {
        return ecs_meta_next(&cursor_);
    }

    /** Move to member by name */
    int member(const char *name) {
        return ecs_meta_member(&cursor_, name);
    }

    /** Move to element by index */
    int elem(int32_t elem) {
        return ecs_meta_elem(&cursor_, elem);
    }

    /** Test if current scope is a collection type */
    bool is_collection() {
        return ecs_meta_is_collection(&cursor_);
    }

    /** Get member name */
    flecs::string_view get_member() const {
        return flecs::string_view(ecs_meta_get_member(&cursor_));
    }

    /** Get type of value */
    flecs::entity get_type() const;

    /** Get unit of value */
    flecs::entity get_unit() const;

    /** Get untyped pointer to value */
    void* get_ptr() {
        return ecs_meta_get_ptr(&cursor_);
    }

    /** Set boolean value */
    int set_bool(bool value) {
        return ecs_meta_set_bool(&cursor_, value);
    }

    /** Set char value */
    int set_char(char value) {
        return ecs_meta_set_char(&cursor_, value);
    }

    /** Set signed int value */
    int set_int(int64_t value) {
        return ecs_meta_set_int(&cursor_, value);
    }

    /** Set unsigned int value */
    int set_uint(uint64_t value) {
        return ecs_meta_set_uint(&cursor_, value);
    }

    /** Set float value */
    int set_float(double value) {
        return ecs_meta_set_float(&cursor_, value);
    }

    /** Set string value */
    int set_string(const char *value) {
        return ecs_meta_set_string(&cursor_, value);
    }

    /** Set string literal value */
    int set_string_literal(const char *value) {
        return ecs_meta_set_string_literal(&cursor_, value);
    }

    /** Set entity value */
    int set_entity(flecs::entity_t value) {
        return ecs_meta_set_entity(&cursor_, value);
    }

    /** Set (component) id value */
    int set_id(flecs::id_t value) {
        return ecs_meta_set_id(&cursor_, value);
    }

    /** Set null value */
    int set_null() {
        return ecs_meta_set_null(&cursor_);
    }

    /** Get boolean value */
    bool get_bool() const {
        return ecs_meta_get_bool(&cursor_);
    }

    /** Get char value */
    char get_char() const {
        return ecs_meta_get_char(&cursor_);
    }

    /** Get signed int value */
    int64_t get_int() const {
        return ecs_meta_get_int(&cursor_);
    }

    /** Get unsigned int value */
    uint64_t get_uint() const {
        return ecs_meta_get_uint(&cursor_);
    }

    /** Get float value */
    double get_float() const {
        return ecs_meta_get_float(&cursor_);
    }

    /** Get string value */
    const char *get_string() const {
        return ecs_meta_get_string(&cursor_);
    }

    /** Get entity value */
    flecs::entity get_entity() const;

    /** Cursor object */
    ecs_meta_cursor_t cursor_;
};

/** @} */

}

/**
 * @file addons/cpp/mixins/meta/opaque.hpp
 * @brief Helpers for opaque type registration.
 */

#pragma once

#include <stdio.h>

namespace flecs {

/**
 * @defgroup cpp_addons_meta Meta
 * @ingroup cpp_addons
 * Flecs reflection framework.
 *
 * @{
 */

/** Serializer object, used for serializing opaque types */
using serializer = ecs_serializer_t;

/** Serializer function, used to serialize opaque types */
using serialize_t = ecs_meta_serialize_t;

/** Type safe variant of serializer function */
template <typename T>
using serialize = int(*)(const serializer *, const T*);

/** Type safe interface for opaque types */
template <typename T, typename ElemType = void>
struct opaque {
    opaque(flecs::world_t *w = nullptr) : world(w) {
        if (world) {
            desc.entity = _::type<T>::id(world);
        }
    }

    /** Type that describes the type kind/structure of the opaque type */
    opaque& as_type(flecs::id_t func) {
        this->desc.type.as_type = func;
        return *this;
    }

    /** Serialize function */
    opaque& serialize(flecs::serialize<T> func) {
        this->desc.type.serialize =
            reinterpret_cast<decltype(
                this->desc.type.serialize)>(func);
        return *this;
    }

    /** Assign bool value */
    opaque& assign_bool(void (*func)(T *dst, bool value)) {
        this->desc.type.assign_bool =
            reinterpret_cast<decltype(
                this->desc.type.assign_bool)>(func);
        return *this;
    }

    /** Assign char value */
    opaque& assign_char(void (*func)(T *dst, char value)) {
        this->desc.type.assign_char =
            reinterpret_cast<decltype(
                this->desc.type.assign_char)>(func);
        return *this;
    }

    /** Assign int value */
    opaque& assign_int(void (*func)(T *dst, int64_t value)) {
        this->desc.type.assign_int =
            reinterpret_cast<decltype(
                this->desc.type.assign_int)>(func);
        return *this;
    }

    /** Assign unsigned int value */
    opaque& assign_uint(void (*func)(T *dst, uint64_t value)) {
        this->desc.type.assign_uint =
            reinterpret_cast<decltype(
                this->desc.type.assign_uint)>(func);
        return *this;
    }

    /** Assign float value */
    opaque& assign_float(void (*func)(T *dst, double value)) {
        this->desc.type.assign_float =
            reinterpret_cast<decltype(
                this->desc.type.assign_float)>(func);
        return *this;
    }

    /** Assign string value */
    opaque& assign_string(void (*func)(T *dst, const char *value)) {
        this->desc.type.assign_string =
            reinterpret_cast<decltype(
                this->desc.type.assign_string)>(func);
        return *this;
    }

    /** Assign entity value */
    opaque& assign_entity(
        void (*func)(T *dst, ecs_world_t *world, ecs_entity_t entity))
    {
        this->desc.type.assign_entity =
            reinterpret_cast<decltype(
                this->desc.type.assign_entity)>(func);
        return *this;
    }

    /** Assign (component) id value */
    opaque& assign_id(
        void (*func)(T *dst, ecs_world_t *world, ecs_id_t id))
    {
        this->desc.type.assign_id =
            reinterpret_cast<decltype(
                this->desc.type.assign_id)>(func);
        return *this;
    }

    /** Assign null value */
    opaque& assign_null(void (*func)(T *dst)) {
        this->desc.type.assign_null =
            reinterpret_cast<decltype(
                this->desc.type.assign_null)>(func);
        return *this;
    }

    /** Clear collection elements */
    opaque& clear(void (*func)(T *dst)) {
        this->desc.type.clear =
            reinterpret_cast<decltype(
                this->desc.type.clear)>(func);
        return *this;
    }

    /** Ensure & get collection element */
    opaque& ensure_element(ElemType* (*func)(T *dst, size_t elem)) {
        this->desc.type.ensure_element =
            reinterpret_cast<decltype(
                this->desc.type.ensure_element)>(func);
        return *this;
    }

    /** Ensure & get element */
    opaque& ensure_member(void* (*func)(T *dst, const char *member)) {
        this->desc.type.ensure_member =
            reinterpret_cast<decltype(
                this->desc.type.ensure_member)>(func);
        return *this;
    }

    /** Return number of elements */
    opaque& count(size_t (*func)(const T *dst)) {
        this->desc.type.count =
            reinterpret_cast<decltype(
                this->desc.type.count)>(func);
        return *this;
    }

    /** Resize to number of elements */
    opaque& resize(void (*func)(T *dst, size_t count)) {
        this->desc.type.resize =
            reinterpret_cast<decltype(
                this->desc.type.resize)>(func);
        return *this;
    }

    ~opaque() {
        if (world) {
            ecs_opaque_init(world, &desc);
        }
    }

    /** Opaque type descriptor */
    flecs::world_t *world = nullptr;
    ecs_opaque_desc_t desc = {};
};

/** @} */

}


#endif
#ifdef FLECS_UNITS
/**
 * @file addons/cpp/mixins/units/decl.hpp
 * @brief Units module declarations.
 */

#pragma once

namespace flecs {
struct units {

/**
 * @defgroup cpp_addons_units Units
 * @ingroup cpp_addons
 * Common unit annotations for reflection framework.
 *
 * @{
 */

struct Prefixes { };

/**
 * @defgroup cpp_addons_units_prefixes Prefixes
 * @ingroup cpp_addons_units
 * Prefixes to indicate unit count (e.g. Kilo, Mega)
 *
 * @{
 */

struct Yocto { };
struct Zepto { };
struct Atto { };
struct Femto { };
struct Pico { };
struct Nano { };
struct Micro { };
struct Milli { };
struct Centi { };
struct Deci { };
struct Deca { };
struct Hecto { };
struct Kilo { };
struct Mega { };
struct Giga { };
struct Tera { };
struct Peta { };
struct Exa { };
struct Zetta { };
struct Yotta { };
struct Kibi { };
struct Mebi { };
struct Gibi { };
struct Tebi { };
struct Pebi { };
struct Exbi { };
struct Zebi { };
struct Yobi { };

/** @} */

/**
 * @defgroup cpp_addons_units_quantities Quantities
 * @ingroup cpp_addons_units
 * Quantities that group units (e.g. Length)
 *
 * @{
 */

struct Duration { };
struct Time { };
struct Mass { };
struct ElectricCurrent { };
struct LuminousIntensity { };
struct Force { };
struct Amount { };
struct Length { };
struct Pressure { };
struct Speed { };
struct Temperature { };
struct Data { };
struct DataRate { };
struct Angle { };
struct Frequency { };
struct Uri { };
struct Color { };

/** @} */

struct duration {
/**
 * @defgroup cpp_addons_units_duration Duration
 * @ingroup cpp_addons_units
 * @{
 */

struct PicoSeconds { };
struct NanoSeconds { };
struct MicroSeconds { };
struct MilliSeconds { };
struct Seconds { };
struct Minutes { };
struct Hours { };
struct Days { };

/** @} */
};

struct angle {
/**
 * @defgroup cpp_addons_units_angle Angle
 * @ingroup cpp_addons_units
 * @{
 */

struct Radians { };
struct Degrees { };

/** @} */
};


struct time {
/**
 * @defgroup cpp_addons_units_time Time
 * @ingroup cpp_addons_units
 * @{
 */

struct Date { };

/** @} */
};


struct mass {
/**
 * @defgroup cpp_addons_units_mass Mass
 * @ingroup cpp_addons_units
 * @{
 */

struct Grams { };
struct KiloGrams { };

/** @} */
};


struct electric_current {
/**
 * @defgroup cpp_addons_units_electric_current Electric Current
 * @ingroup cpp_addons_units
 * @{
 */

struct Ampere { };

/** @} */
};


struct amount {
/**
 * @defgroup cpp_addons_units_amount Amount
 * @ingroup cpp_addons_units
 * @{
 */

struct Mole { };

/** @} */
};


struct luminous_intensity {
/**
 * @defgroup cpp_addons_units_luminous_intensity Luminous Intensity
 * @ingroup cpp_addons_units
 * @{
 */

struct Candela { };

/** @} */
};


struct force {
/**
 * @defgroup cpp_addons_units_force Force
 * @ingroup cpp_addons_units
 * @{
 */

struct Newton { };

/** @} */
};


struct length {
/**
 * @defgroup cpp_addons_units_length Length
 * @ingroup cpp_addons_units
 * @{
 */

struct Meters { };
struct PicoMeters { };
struct NanoMeters { };
struct MicroMeters { };
struct MilliMeters { };
struct CentiMeters { };
struct KiloMeters { };
struct Miles { };
struct Pixels { };

/** @} */
};


struct pressure {
/**
 * @defgroup cpp_addons_units_pressure Pressure
 * @ingroup cpp_addons_units
 * @{
 */

struct Pascal { };
struct Bar { };

/** @} */
};


struct speed {
/**
 * @defgroup cpp_addons_units_speed Speed
 * @ingroup cpp_addons_units
 * @{
 */

struct MetersPerSecond { };
struct KiloMetersPerSecond { };
struct KiloMetersPerHour { };
struct MilesPerHour { };

/** @} */
};


struct temperature {
/**
 * @defgroup cpp_addons_units_temperature Temperature
 * @ingroup cpp_addons_units
 * @{
 */

struct Kelvin { };
struct Celsius { };
struct Fahrenheit { };

/** @} */
};


struct data {
/**
 * @defgroup cpp_addons_units_data Data
 * @ingroup cpp_addons_units
 * @{
 */

struct Bits { };
struct KiloBits { };
struct MegaBits { };
struct GigaBits { };
struct Bytes { };
struct KiloBytes { };
struct MegaBytes { };
struct GigaBytes { };
struct KibiBytes { };
struct MebiBytes { };
struct GibiBytes { };

/** @} */
};

struct datarate {
/**
 * @defgroup cpp_addons_units_datarate Data Rate
 * @ingroup cpp_addons_units
 * @{
 */

struct BitsPerSecond { };
struct KiloBitsPerSecond { };
struct MegaBitsPerSecond { };
struct GigaBitsPerSecond { };
struct BytesPerSecond { };
struct KiloBytesPerSecond { };
struct MegaBytesPerSecond { };
struct GigaBytesPerSecond { };

/** @} */
};


struct frequency {
/**
 * @defgroup cpp_addons_units_frequency Frequency
 * @ingroup cpp_addons_units
 * @{
 */

struct Hertz { };
struct KiloHertz { };
struct MegaHertz { };
struct GigaHertz { };

/** @} */
};


struct uri {
/**
 * @defgroup cpp_addons_units_uri Uri
 * @ingroup cpp_addons_units
 * @{
 */

struct Hyperlink { };
struct Image { };
struct File { };

/** @} */
};


struct color {
/**
 * @defgroup cpp_addons_units_color Color
 * @ingroup cpp_addons_units
 * @{
 */

struct Rgb { };
struct Hsl { };
struct Css { };

/** @} */
};

struct Percentage { };
struct Bel { };
struct DeciBel { };

units(flecs::world& world);

/** @} */

};
}

#endif
#ifdef FLECS_STATS
/**
 * @file addons/cpp/mixins/stats/decl.hpp
 * @brief Stats module declarations.
 */

#pragma once

namespace flecs {

/**
 * @defgroup cpp_addons_stats Stats
 * @ingroup cpp_addons
 * The stats addon tracks statistics for the world and systems.
 *
 * @{
 */

/** Component that stores world statistics */
using WorldStats = EcsWorldStats;

/** Component that stores system/pipeline statistics */
using PipelineStats = EcsPipelineStats;

/** Component with world summary stats */
using WorldSummary = EcsWorldSummary;

struct stats {
    stats(flecs::world& world);
};

/** @} */

}

#endif
#ifdef FLECS_METRICS
/**
 * @file addons/cpp/mixins/metrics/decl.hpp
 * @brief Metrics declarations.
 */

#pragma once

/**
 * @file addons/cpp/mixins/metrics/builder.hpp
 * @brief Metric builder.
 */

#pragma once

#define ECS_EVENT_DESC_ID_COUNT_MAX (8)

namespace flecs {

/**
 * @ingroup cpp_addons_metrics
 * @{
 */

/** Event builder interface */
struct metric_builder {
    metric_builder(flecs::world_t *world, flecs::entity_t entity) 
        : world_(world) 
    {
        desc_.entity = entity;
    }

    ~metric_builder();

    metric_builder& member(flecs::entity_t e) {
        desc_.member = e;
        return *this;
    }

    metric_builder& member(const char *name);

    template <typename T>
    metric_builder& member(const char *name);

    metric_builder& dotmember(const char *name);

    template <typename T>
    metric_builder& dotmember(const char *name);

    metric_builder& id(flecs::id_t the_id) {
        desc_.id = the_id;
        return *this;
    }

    metric_builder& id(flecs::entity_t first, flecs::entity_t second) {
        desc_.id = ecs_pair(first, second);
        return *this;
    }

    template <typename T>
    metric_builder& id() {
        return id(_::type<T>::id(world_));
    }

    template <typename First>
    metric_builder& id(flecs::entity_t second) {
        return id(_::type<First>::id(world_), second);
    }

    template <typename Second>
    metric_builder& id_second(flecs::entity_t first) {
        return id(first, _::type<Second>::id(world_));
    }

    template <typename First, typename Second>
    metric_builder& id() {
        return id<First>(_::type<Second>::id(world_));
    }

    metric_builder& targets(bool value = true) {
        desc_.targets = value;
        return *this;
    }

    metric_builder& kind(flecs::entity_t the_kind) {
        desc_.kind = the_kind;
        return *this;
    }

    template <typename Kind>
    metric_builder& kind() {
        return kind(_::type<Kind>::id(world_));
    }

    metric_builder& brief(const char *b) {
        desc_.brief = b;
        return *this;
    }

    operator flecs::entity();

protected:
    flecs::world_t *world_;
    ecs_metric_desc_t desc_ = {};
    bool created_ = false;
};

/**
 * @}
 */

}


namespace flecs {

/**
 * @defgroup cpp_addons_metrics Metrics
 * @ingroup cpp_addons
 * The metrics module extracts metrics from components and makes them available
 * through a unified component interface.
 *
 * @{
 */

struct metrics {
    using Value = EcsMetricValue;
    using Source = EcsMetricSource;

    struct Instance { };
    struct Metric { };
    struct Counter { };
    struct CounterIncrement { };
    struct CounterId { };
    struct Gauge { };

    metrics(flecs::world& world);
};

/** @} */

}

#endif
#ifdef FLECS_ALERTS
/**
 * @file addons/cpp/mixins/alerts/decl.hpp
 * @brief Alert declarations.
 */

#pragma once

namespace flecs {

/**
 * @defgroup cpp_addons_alerts Alerts
 * @ingroup cpp_addons
 * Alert implementation.
 *
 * @{
 */

/** Module */
struct alerts {
    using AlertsActive = EcsAlertsActive;
    using Instance = EcsAlertInstance;

    struct Alert { };
    struct Info { };
    struct Warning { };
    struct Error { };

    alerts(flecs::world& world);
};

template <typename ... Components>
struct alert;

template <typename ... Components>
struct alert_builder;

/** @} */

}

#endif
#ifdef FLECS_JSON
/**
 * @file addons/cpp/mixins/json/decl.hpp
 * @brief JSON addon declarations.
 */

#pragma once

namespace flecs {

/**
 * @defgroup cpp_addons_json Json
 * @ingroup cpp_addons
 * Functions for serializing to/from JSON.
 *
 * @{
 */

using from_json_desc_t = ecs_from_json_desc_t;
using entity_to_json_desc_t = ecs_entity_to_json_desc_t;
using iter_to_json_desc_t = ecs_iter_to_json_desc_t;

/** @} */

}

#endif
#ifdef FLECS_APP
/**
 * @file addons/cpp/mixins/app/decl.hpp
 * @brief App addon declarations.
 */

#pragma once

/**
 * @file addons/cpp/mixins/app/builder.hpp
 * @brief App builder.
 */

#pragma once

namespace flecs {

/**
 * @defgroup cpp_addons_app App
 * @ingroup cpp_addons
 * Optional addon for running the main application loop.
 *
 * @{
 */

/** App builder interface */
struct app_builder {
    app_builder(flecs::world_t *world)
        : world_(world)
        , desc_{}
    {
        const ecs_world_info_t *stats = ecs_get_world_info(world);
        desc_.target_fps = stats->target_fps;
        ecs_ftime_t t_zero = 0.0;
        if (ECS_EQ(desc_.target_fps, t_zero)) {
            desc_.target_fps = 60;
        }
    }

    app_builder& target_fps(ecs_ftime_t value) {
        desc_.target_fps = value;
        return *this;
    }

    app_builder& delta_time(ecs_ftime_t value) {
        desc_.delta_time = value;
        return *this;
    }

    app_builder& threads(int32_t value) {
        desc_.threads = value;
        return *this;
    }

    app_builder& frames(int32_t value) {
        desc_.frames = value;
        return *this;
    }

    app_builder& enable_rest(uint16_t port = 0) {
        desc_.enable_rest = true;
        desc_.port = port;
        return *this;
    }

    app_builder& enable_stats(bool value = true) {
        desc_.enable_stats = value;
        return *this;
    }

    app_builder& init(ecs_app_init_action_t value) {
        desc_.init = value;
        return *this;
    }

    app_builder& ctx(void *value) {
        desc_.ctx = value;
        return *this;
    }

    int run() {
        int result = ecs_app_run(world_, &desc_);
        if (ecs_should_quit(world_)) {
            // Only free world if quit flag is set. This ensures that we won't
            // try to cleanup the world if the app is used in an environment
            // that takes over the main loop, like with emscripten.
            if (!flecs_poly_release(world_)) {
                ecs_fini(world_);
            }
        }
        return result;
    }

private:
    flecs::world_t *world_;
    ecs_app_desc_t desc_;
};

/** @} */

}


#endif
#ifdef FLECS_SCRIPT
/**
 * @file addons/cpp/mixins/script/decl.hpp
 * @brief Script declarations.
 */

#pragma once

/**
 * @file addons/cpp/mixins/script/builder.hpp
 * @brief Script builder.
 */

#pragma once

namespace flecs {

/**
 * @ingroup cpp_addons_script
 * @{
 */

/** Script builder interface */
struct script_builder {
    script_builder(flecs::world_t *world, const char *name = nullptr)
        : world_(world)
        , desc_{}
    {
        if (name != nullptr) {
            ecs_entity_desc_t entity_desc = {};
            entity_desc.name = name;
            entity_desc.sep = "::";
            entity_desc.root_sep = "::";
            this->desc_.entity = ecs_entity_init(world, &entity_desc);
        }
    }

    script_builder& code(const char *str) {
        desc_.code = str;
        return *this;
    }

    script_builder& filename(const char *str) {
        desc_.filename = str;
        return *this;
    }

    flecs::entity run() const;

protected:
    flecs::world_t *world_;
    ecs_script_desc_t desc_;
};

}


namespace flecs {

/**
 * @defgroup cpp_addons_script Script
 * @ingroup cpp_addons
 *
 * @{
 */

struct script_builder;

/** @} */

}

#endif

/**
 * @file addons/cpp/log.hpp
 * @brief Logging functions.
 */

#pragma once

namespace flecs {
namespace log {

/**
 * @defgroup cpp_log Logging
 * @ingroup cpp_addons
 * Logging functions.
 *
 * @{
 */

/** Set log level */
inline void set_level(int level) {
    ecs_log_set_level(level);
}

inline int get_level() {
    return ecs_log_get_level();
}

/** Enable colors in logging */
inline void enable_colors(bool enabled = true) {
    ecs_log_enable_colors(enabled);
}

/** Enable timestamps in logging */
inline void enable_timestamp(bool enabled = true) {
    ecs_log_enable_timestamp(enabled);
}

/** Enable time delta in logging */
inline void enable_timedelta(bool enabled = true) {
    ecs_log_enable_timedelta(enabled);
}

/** Debug trace (level 1) */
inline void dbg(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    ecs_logv(1, fmt, args);
    va_end(args);
}

/** Trace (level 0) */
inline void trace(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    ecs_logv(0, fmt, args);
    va_end(args);
}

/** Trace (level -2) */
inline void warn(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    ecs_logv(-2, fmt, args);
    va_end(args);
}

/** Trace (level -3) */
inline void err(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    ecs_logv(-3, fmt, args);
    va_end(args);
}

/** Increase log indentation */
inline void push(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    ecs_logv(0, fmt, args);
    va_end(args);
    ecs_log_push();
}

/** Increase log indentation */
inline void push() {
    ecs_log_push();
}

/** Increase log indentation */
inline void pop() {
    ecs_log_pop();
}

/** @} */

}
}

/**
 * @file addons/cpp/pair.hpp
 * @brief Utilities for working with compile time pairs.
 */

#pragma once

namespace flecs {

namespace _ {
    struct pair_base { };
} // _


/**
 * @defgroup cpp_pair_type Pair type
 * @ingroup cpp_core
 * Compile time utilities for working with relationship pairs.
 *
 * @{
 */

/** Type that represents a pair.
 * The pair type can be used to represent a pair at compile time, and is able
 * to automatically derive the storage type associated with the pair, accessible
 * through pair::type.
 *
 * The storage type is derived using the following rules:
 * - if pair::first is non-empty, the storage type is pair::first
 * - if pair::first is empty and pair::second is non-empty, the storage type is pair::second
 *
 * The pair type can hold a temporary value so that it can be used in the
 * signatures of queries
 */
template <typename First, typename Second>
struct pair : _::pair_base {
    using type = conditional_t<!is_empty<First>::value || is_empty<Second>::value, First, Second>;
    using first = First;
    using second = Second;

    pair(type& v) : ref_(v) { }

    // This allows the class to be used as a temporary object
    pair(const type& v) : ref_(const_cast<type&>(v)) { }

    operator type&() {
        return ref_;
    }

    operator const type&() const {
        return ref_;
    }

    type* operator->() {
        return &ref_;
    }

    const type* operator->() const {
        return &ref_;
    }

    type& operator*() {
        return ref_;
    }

    const type& operator*() const {
        return ref_;
    }

private:
    type& ref_;
};

template <typename First, typename Second, if_t<is_empty<First>::value> = 0>
using pair_object = pair<First, Second>;

template <typename T>
using raw_type_t = remove_pointer_t<remove_reference_t<T>>;

/** Test if type is a pair. */
template <typename T>
struct is_pair {
    static constexpr bool value = is_base_of<_::pair_base, raw_type_t<T> >::value;
};

/** Get pair::first from pair while preserving cv qualifiers. */
template <typename P>
using pair_first_t = transcribe_cv_t<remove_reference_t<P>, typename raw_type_t<P>::first>;

/** Get pair::second from pair while preserving cv qualifiers. */
template <typename P>
using pair_second_t = transcribe_cv_t<remove_reference_t<P>, typename raw_type_t<P>::second>;

/** Get pair::type type from pair while preserving cv qualifiers and pointer type. */
template <typename P>
using pair_type_t = transcribe_cvp_t<remove_reference_t<P>, typename raw_type_t<P>::type>;

/** Get actual type from a regular type or pair. */
template <typename T, typename U = int>
struct actual_type;

template <typename T>
struct actual_type<T, if_not_t< is_pair<T>::value >> {
    using type = T;
};

template <typename T>
struct actual_type<T, if_t< is_pair<T>::value >> {
    using type = pair_type_t<T>;
};

template <typename T>
using actual_type_t = typename actual_type<T>::type;


// Get type without const, *, &
template<typename T>
struct base_type {
    using type = decay_t< actual_type_t<T> >;
};

template <typename T>
using base_type_t = typename base_type<T>::type;


// Get type without *, & (retains const which is useful for function args)
template<typename T>
struct base_arg_type {
    using type = remove_pointer_t< remove_reference_t< actual_type_t<T> > >;
};

template <typename T>
using base_arg_type_t = typename base_arg_type<T>::type;


// Test if type is the same as its actual type
template <typename T>
struct is_actual {
    static constexpr bool value =
        std::is_same<T, actual_type_t<T> >::value && !is_enum<T>::value;
};

} // flecs

/**
 * @file addons/cpp/lifecycle_traits.hpp
 * @brief Utilities for discovering and registering component lifecycle hooks.
 */

#pragma once

namespace flecs 
{

namespace _ 
{

// T()
// Can't coexist with T(flecs::entity) or T(flecs::world, flecs::entity)
template <typename T>
void ctor_impl(void *ptr, int32_t count, const ecs_type_info_t *info) {
    (void)info; ecs_assert(info->size == ECS_SIZEOF(T),
        ECS_INTERNAL_ERROR, NULL);
    T *arr = static_cast<T*>(ptr);
    for (int i = 0; i < count; i ++) {
        FLECS_PLACEMENT_NEW(&arr[i], T);
    }
}

// ~T()
template <typename T>
void dtor_impl(void *ptr, int32_t count, const ecs_type_info_t *info) {
    (void)info; ecs_assert(info->size == ECS_SIZEOF(T), 
        ECS_INTERNAL_ERROR, NULL);
    T *arr = static_cast<T*>(ptr);
    for (int i = 0; i < count; i ++) {
        arr[i].~T();
    }
}

// T& operator=(const T&)
template <typename T>
void copy_impl(void *dst_ptr, const void *src_ptr, int32_t count, 
    const ecs_type_info_t *info)
{
    (void)info; ecs_assert(info->size == ECS_SIZEOF(T), 
        ECS_INTERNAL_ERROR, NULL);
    T *dst_arr = static_cast<T*>(dst_ptr);
    const T *src_arr = static_cast<const T*>(src_ptr);
    for (int i = 0; i < count; i ++) {
        dst_arr[i] = src_arr[i];
    }
}

// T& operator=(T&&)
template <typename T>
void move_impl(void *dst_ptr, void *src_ptr, int32_t count, 
    const ecs_type_info_t *info)
{
    (void)info; ecs_assert(info->size == ECS_SIZEOF(T), 
        ECS_INTERNAL_ERROR, NULL);
    T *dst_arr = static_cast<T*>(dst_ptr);
    T *src_arr = static_cast<T*>(src_ptr);
    for (int i = 0; i < count; i ++) {
        dst_arr[i] = FLECS_MOV(src_arr[i]);
    }
}

// T(T&)
template <typename T>
void copy_ctor_impl(void *dst_ptr, const void *src_ptr, int32_t count, 
    const ecs_type_info_t *info)
{
    (void)info; ecs_assert(info->size == ECS_SIZEOF(T), 
        ECS_INTERNAL_ERROR, NULL);
    T *dst_arr = static_cast<T*>(dst_ptr);
    const T *src_arr = static_cast<const T*>(src_ptr);
    for (int i = 0; i < count; i ++) {
        FLECS_PLACEMENT_NEW(&dst_arr[i], T(src_arr[i]));
    }
}

// T(T&&)
template <typename T>
void move_ctor_impl(void *dst_ptr, void *src_ptr, int32_t count, 
    const ecs_type_info_t *info)
{
    (void)info; ecs_assert(info->size == ECS_SIZEOF(T), 
        ECS_INTERNAL_ERROR, NULL);
    T *dst_arr = static_cast<T*>(dst_ptr);
    T *src_arr = static_cast<T*>(src_ptr);
    for (int i = 0; i < count; i ++) {
        FLECS_PLACEMENT_NEW(&dst_arr[i], T(FLECS_MOV(src_arr[i])));
    }
}

// T(T&&), ~T()
// Typically used when moving to a new table, and removing from the old table
template <typename T>
void ctor_move_dtor_impl(void *dst_ptr, void *src_ptr, int32_t count, 
    const ecs_type_info_t *info)
{
    (void)info; ecs_assert(info->size == ECS_SIZEOF(T), 
        ECS_INTERNAL_ERROR, NULL);
    T *dst_arr = static_cast<T*>(dst_ptr);
    T *src_arr = static_cast<T*>(src_ptr);
    for (int i = 0; i < count; i ++) {
        FLECS_PLACEMENT_NEW(&dst_arr[i], T(FLECS_MOV(src_arr[i])));
        src_arr[i].~T();
    }
}

// Move assign + dtor (non-trivial move assignment)
// Typically used when moving a component to a deleted component
template <typename T, if_not_t<
    std::is_trivially_move_assignable<T>::value > = 0>
void move_dtor_impl(void *dst_ptr, void *src_ptr, int32_t count, 
    const ecs_type_info_t *info)
{
    (void)info; ecs_assert(info->size == ECS_SIZEOF(T), 
        ECS_INTERNAL_ERROR, NULL);
    T *dst_arr = static_cast<T*>(dst_ptr);
    T *src_arr = static_cast<T*>(src_ptr);
    for (int i = 0; i < count; i ++) {
        // Move assignment should free dst & assign dst to src
        dst_arr[i] = FLECS_MOV(src_arr[i]);
        // Destruct src. Move should have left object in a state where it no
        // longer holds resources, but it still needs to be destructed.
        src_arr[i].~T();
    }
}

// Move assign + dtor (trivial move assignment)
// Typically used when moving a component to a deleted component
template <typename T, if_t<
    std::is_trivially_move_assignable<T>::value > = 0>
void move_dtor_impl(void *dst_ptr, void *src_ptr, int32_t count, 
    const ecs_type_info_t *info)
{
    (void)info; ecs_assert(info->size == ECS_SIZEOF(T), 
        ECS_INTERNAL_ERROR, NULL);
    T *dst_arr = static_cast<T*>(dst_ptr);
    T *src_arr = static_cast<T*>(src_ptr);
    for (int i = 0; i < count; i ++) {
        // Cleanup resources of dst
        dst_arr[i].~T();
        // Copy src to dst
        dst_arr[i] = FLECS_MOV(src_arr[i]);
        // No need to destruct src. Since this is a trivial move the code
        // should be agnostic to the address of the component which means we
        // can pretend nothing got destructed.
    }
}

} // _

// Trait to test if type is constructible by flecs
template <typename T>
struct is_flecs_constructible {
    static constexpr bool value = 
        std::is_default_constructible<actual_type_t<T>>::value;
};

namespace _
{

// Trivially constructible
template <typename T, if_t< std::is_trivially_constructible<T>::value > = 0>
ecs_xtor_t ctor(ecs_flags32_t &) {
    return nullptr;
}

// Not constructible by flecs
template <typename T, if_t< 
    ! std::is_default_constructible<T>::value > = 0>
ecs_xtor_t ctor(ecs_flags32_t &flags) {
    flags |= ECS_TYPE_HOOK_CTOR_ILLEGAL;
    return nullptr;
}

// Default constructible
template <typename T, if_t<
    ! std::is_trivially_constructible<T>::value &&
    std::is_default_constructible<T>::value > = 0>
ecs_xtor_t ctor(ecs_flags32_t &) {
    return ctor_impl<T>;
}

// No dtor
template <typename T, if_t< std::is_trivially_destructible<T>::value > = 0>
ecs_xtor_t dtor(ecs_flags32_t &) {
    return nullptr;
}

// Dtor
template <typename T, if_t<
    std::is_destructible<T>::value &&
    ! std::is_trivially_destructible<T>::value > = 0>
ecs_xtor_t dtor(ecs_flags32_t &) {
    return dtor_impl<T>;
}

// Assert when the type cannot be destructed
template <typename T, if_not_t< std::is_destructible<T>::value > = 0>
ecs_xtor_t dtor(ecs_flags32_t &flags) {
    flecs_static_assert(always_false<T>::value, 
        "component type must be destructible");
    flags |= ECS_TYPE_HOOK_DTOR_ILLEGAL;
    return nullptr;
}

// Trivially copyable
template <typename T, if_t< std::is_trivially_copyable<T>::value > = 0>
ecs_copy_t copy(ecs_flags32_t &) {
    return nullptr;
}

// Not copyable
template <typename T, if_t<
    ! std::is_trivially_copyable<T>::value &&
    ! std::is_copy_assignable<T>::value > = 0>
ecs_copy_t copy(ecs_flags32_t &flags) {
    flags |= ECS_TYPE_HOOK_COPY_ILLEGAL;
    return nullptr;
}

// Copy assignment
template <typename T, if_t<
    std::is_copy_assignable<T>::value &&
    ! std::is_trivially_copyable<T>::value > = 0>
ecs_copy_t copy(ecs_flags32_t &) {
    return copy_impl<T>;
}

// Trivially move assignable
template <typename T, if_t< std::is_trivially_move_assignable<T>::value > = 0>
ecs_move_t move(ecs_flags32_t &) {
    return nullptr;
}

// Component types must be move assignable
template <typename T, if_not_t< std::is_move_assignable<T>::value > = 0>
ecs_move_t move(ecs_flags32_t &flags) {
    flags |= ECS_TYPE_HOOK_MOVE_ILLEGAL;
    return nullptr;
}

// Move assignment
template <typename T, if_t<
    std::is_move_assignable<T>::value &&
    ! std::is_trivially_move_assignable<T>::value > = 0>
ecs_move_t move(ecs_flags32_t &) {
    return move_impl<T>;
}

// Trivially copy constructible
template <typename T, if_t<
    std::is_trivially_copy_constructible<T>::value > = 0>
ecs_copy_t copy_ctor(ecs_flags32_t &) {
    return nullptr;
}

// No copy ctor
template <typename T, if_t< ! std::is_copy_constructible<T>::value > = 0>
ecs_copy_t copy_ctor(ecs_flags32_t &flags) {
       flags |= ECS_TYPE_HOOK_COPY_CTOR_ILLEGAL;
    return nullptr;

}

// Copy ctor
template <typename T, if_t<
    std::is_copy_constructible<T>::value &&
    ! std::is_trivially_copy_constructible<T>::value > = 0>
ecs_copy_t copy_ctor(ecs_flags32_t &) {
    return copy_ctor_impl<T>;
}

// Trivially move constructible
template <typename T, if_t<
    std::is_trivially_move_constructible<T>::value > = 0>
ecs_move_t move_ctor(ecs_flags32_t &) {
    return nullptr;
}

// Component types must be move constructible
template <typename T, if_not_t< std::is_move_constructible<T>::value > = 0>
ecs_move_t move_ctor(ecs_flags32_t &flags) {
    flags |= ECS_TYPE_HOOK_MOVE_CTOR_ILLEGAL;
    return nullptr;
}

// Move ctor
template <typename T, if_t<
    std::is_move_constructible<T>::value &&
    ! std::is_trivially_move_constructible<T>::value > = 0>
ecs_move_t move_ctor(ecs_flags32_t &) {
    return move_ctor_impl<T>;
}

// Trivial merge (move assign + dtor)
template <typename T, if_t<
    std::is_trivially_move_constructible<T>::value  &&
    std::is_trivially_destructible<T>::value > = 0>
ecs_move_t ctor_move_dtor(ecs_flags32_t &) {
    return nullptr;
}

// Component types must be move constructible and destructible
template <typename T, if_t<
    ! std::is_move_constructible<T>::value ||
    ! std::is_destructible<T>::value > = 0>
ecs_move_t ctor_move_dtor(ecs_flags32_t &flags) {
    flags |= ECS_TYPE_HOOK_CTOR_MOVE_DTOR_ILLEGAL;
    return nullptr;
}

// Merge ctor + dtor
template <typename T, if_t<
    !(std::is_trivially_move_constructible<T>::value &&
      std::is_trivially_destructible<T>::value) &&
    std::is_move_constructible<T>::value &&
    std::is_destructible<T>::value > = 0>
ecs_move_t ctor_move_dtor(ecs_flags32_t &) {
    return ctor_move_dtor_impl<T>;
}

// Trivial merge (move assign + dtor)
template <typename T, if_t<
    std::is_trivially_move_assignable<T>::value  &&
    std::is_trivially_destructible<T>::value > = 0>
ecs_move_t move_dtor(ecs_flags32_t &) {
    return nullptr;
}

// Component types must be move constructible and destructible
template <typename T, if_t<
    ! std::is_move_assignable<T>::value ||
    ! std::is_destructible<T>::value > = 0>
ecs_move_t move_dtor(ecs_flags32_t &flags) {
    flags |= ECS_TYPE_HOOK_MOVE_DTOR_ILLEGAL;
    return nullptr;
}

// Merge assign + dtor
template <typename T, if_t<
    !(std::is_trivially_move_assignable<T>::value &&
      std::is_trivially_destructible<T>::value) &&
    std::is_move_assignable<T>::value &&
    std::is_destructible<T>::value > = 0>
ecs_move_t move_dtor(ecs_flags32_t &) {
    return move_dtor_impl<T>;
}

} // _
} // flecs

/**
 * @file addons/cpp/world.hpp
 * @brief World class.
 */

#pragma once

namespace flecs
{

/* Static helper functions to assign a component value */

// set(T&&), T = constructible
template <typename T, if_t< is_flecs_constructible<T>::value > = 0>
inline void set(world_t *world, flecs::entity_t entity, T&& value, flecs::id_t id) {
    ecs_assert(_::type<T>::size() != 0, ECS_INVALID_PARAMETER,
            "operation invalid for empty type");

    if (!ecs_is_deferred(world)) {
        T& dst = *static_cast<T*>(ecs_ensure_id(world, entity, id));
        dst = FLECS_MOV(value);

        ecs_modified_id(world, entity, id);
    } else {
        T& dst = *static_cast<T*>(ecs_ensure_modified_id(world, entity, id));
        dst = FLECS_MOV(value);
    }
}

// set(const T&), T = constructible
template <typename T, if_t< is_flecs_constructible<T>::value > = 0>
inline void set(world_t *world, flecs::entity_t entity, const T& value, flecs::id_t id) {
    ecs_assert(_::type<T>::size() != 0, ECS_INVALID_PARAMETER,
            "operation invalid for empty type");

    if (!ecs_is_deferred(world)) {
        T& dst = *static_cast<T*>(ecs_ensure_id(world, entity, id));
        dst = FLECS_MOV(value);

        ecs_modified_id(world, entity, id);
    } else {
        T& dst = *static_cast<T*>(ecs_ensure_modified_id(world, entity, id));
        dst = FLECS_MOV(value);
    }
}

// set(T&&), T = not constructible
template <typename T, if_not_t< is_flecs_constructible<T>::value > = 0>
inline void set(world_t *world, flecs::entity_t entity, T&& value, flecs::id_t id) {
    ecs_assert(_::type<T>::size() != 0, ECS_INVALID_PARAMETER,
            "operation invalid for empty type");

    if (!ecs_is_deferred(world)) {
        T& dst = *static_cast<remove_reference_t<T>*>(ecs_ensure_id(world, entity, id));
        dst = FLECS_MOV(value);

        ecs_modified_id(world, entity, id);
    } else {
        T& dst = *static_cast<remove_reference_t<T>*>(ecs_ensure_modified_id(world, entity, id));
        dst = FLECS_MOV(value);
    }
}

// set(const T&), T = not constructible
template <typename T, if_not_t< is_flecs_constructible<T>::value > = 0>
inline void set(world_t *world, flecs::entity_t entity, const T& value, flecs::id_t id) {
    ecs_assert(_::type<T>::size() != 0, ECS_INVALID_PARAMETER,
            "operation invalid for empty type");

    if (!ecs_is_deferred(world)) {
        T& dst = *static_cast<remove_reference_t<T>*>(ecs_ensure_id(world, entity, id));
        dst = FLECS_MOV(value);

        ecs_modified_id(world, entity, id);
    } else {
        T& dst = *static_cast<remove_reference_t<T>*>(ecs_ensure_modified_id(world, entity, id));
        dst = FLECS_MOV(value);
    }
}

// emplace for T(Args...)
template <typename T, typename ... Args, if_t<
    std::is_constructible<actual_type_t<T>, Args...>::value ||
    std::is_default_constructible<actual_type_t<T>>::value > = 0>
inline void emplace(world_t *world, flecs::entity_t entity, flecs::id_t id, Args&&... args) {
    ecs_assert(_::type<T>::size() != 0, ECS_INVALID_PARAMETER,
            "operation invalid for empty type");
    T& dst = *static_cast<T*>(ecs_emplace_id(world, entity, id, nullptr));

    FLECS_PLACEMENT_NEW(&dst, T{FLECS_FWD(args)...});

    ecs_modified_id(world, entity, id);
}

// set(T&&)
template <typename T, typename A>
inline void set(world_t *world, entity_t entity, A&& value) {
    id_t id = _::type<T>::id(world);
    flecs::set(world, entity, FLECS_FWD(value), id);
}

// set(const T&)
template <typename T, typename A>
inline void set(world_t *world, entity_t entity, const A& value) {
    id_t id = _::type<T>::id(world);
    flecs::set(world, entity, value, id);
}

/** Return id without generation.
 *
 * @see ecs_strip_generation()
 */
inline flecs::id_t strip_generation(flecs::entity_t e) {
    return ecs_strip_generation(e);
}

/** Return entity generation.
 */
inline uint32_t get_generation(flecs::entity_t e) {
    return ECS_GENERATION(e);
}

struct scoped_world;

/**
 * @defgroup cpp_world World
 * @ingroup cpp_core
 * World operations.
 *
 * @{
 */

/** The world.
 * The world is the container of all ECS data and systems. If the world is
 * deleted, all data in the world will be deleted as well.
 */
struct world {
    /** Create world.
     */
    explicit world()
        : world_( ecs_init() ) { 
            init_builtin_components(); 
        }

    /** Create world with command line arguments.
     * Currently command line arguments are not interpreted, but they may be
     * used in the future to configure Flecs parameters.
     */
    explicit world(int argc, char *argv[])
        : world_( ecs_init_w_args(argc, argv) ) { 
            init_builtin_components(); 
        }

    /** Create world from C world.
     */
    explicit world(world_t *w)
        : world_( w ) { 
            if (w) {
                flecs_poly_claim(w);
            }
        }

    /** Not allowed to copy a world. May only take a reference.
     */
    world(const world& obj) {
        this->world_ = obj.world_;
        flecs_poly_claim(this->world_);
    }

    world& operator=(const world& obj) noexcept {
        release();
        this->world_ = obj.world_;
        flecs_poly_claim(this->world_);
        return *this;
    }

    world(world&& obj) noexcept {
        world_ = obj.world_;
        obj.world_ = nullptr;
    }

    world& operator=(world&& obj) noexcept {
        release();
        world_ = obj.world_;
        obj.world_ = nullptr;
        return *this;
    }

    /* Releases the underlying world object. If this is the last handle, the world
       will be finalized. */
    void release() {
        if (world_) {
            if (!flecs_poly_release(world_)) {
                if (ecs_stage_get_id(world_) == -1) {
                    ecs_stage_free(world_);
                } else {
                    // before we call ecs_fini(), we increment the reference count back to 1
                    // otherwise, copies of this object created during ecs_fini (e.g. a component on_remove hook)
                    // would call again this destructor and ecs_fini().
                    flecs_poly_claim(world_);
                    ecs_fini(world_);
                }
            }
            world_ = nullptr;
        }        
    }

    ~world() {
        release();
    }

    /* Implicit conversion to world_t* */
    operator world_t*() const { return world_; }

    /** Make current world object owner of the world. This may only be called on
     * one flecs::world object, an may only be called  once. Failing to do so
     * will result in undefined behavior.
     * 
     * This operation allows a custom (C) world to be wrapped by a C++ object,
     * and transfer ownership so that the world is automatically cleaned up.
     */
    void make_owner() {
        flecs_poly_release(world_);
    }

    /** Deletes and recreates the world. */
    void reset() {
        /* Make sure there's only one reference to the world */
        ecs_assert(flecs_poly_refcount(world_) == 1, ECS_INVALID_OPERATION,
            "reset would invalidate other handles");
        ecs_fini(world_);
        world_ = ecs_init();
    }

    /** Obtain pointer to C world object.
     */
    world_t* c_ptr() const {
        return world_;
    }

    /** Signal application should quit.
     * After calling this operation, the next call to progress() returns false.
     */
    void quit() const {
        ecs_quit(world_);
    }

    /** Register action to be executed when world is destroyed.
     */
    void atfini(ecs_fini_action_t action, void *ctx = nullptr) const {
        ecs_atfini(world_, action, ctx);
    }

    /** Test if quit() has been called.
     */
    bool should_quit() const {
        return ecs_should_quit(world_);
    }

    /** Begin frame.
     * When an application does not use progress() to control the main loop, it
     * can still use Flecs features such as FPS limiting and time measurements.
     * This operation needs to be invoked whenever a new frame is about to get
     * processed.
     *
     * Calls to frame_begin() must always be followed by frame_end().
     *
     * The function accepts a delta_time parameter, which will get passed to
     * systems. This value is also used to compute the amount of time the
     * function needs to sleep to ensure it does not exceed the target_fps, when
     * it is set. When 0 is provided for delta_time, the time will be measured.
     *
     * This function should only be ran from the main thread.
     *
     * @param delta_time Time elapsed since the last frame.
     * @return The provided delta_time, or measured time if 0 was provided.
     *
     * @see ecs_frame_begin()
     * @see flecs::world::frame_end()
     */
    ecs_ftime_t frame_begin(float delta_time = 0) const {
        return ecs_frame_begin(world_, delta_time);
    }

    /** End frame.
     * This operation must be called at the end of the frame, and always after
     * frame_begin().
     *
     * This function should only be ran from the main thread.
     *
     * @see ecs_frame_end()
     * @see flecs::world::frame_begin()
     */
    void frame_end() const {
        ecs_frame_end(world_);
    }

    /** Begin readonly mode.
     *
     * @param multi_threaded Whether to enable readonly/multi threaded mode.
     * 
     * @return Whether world is currently readonly.
     *
     * @see ecs_readonly_begin()
     * @see flecs::world::is_readonly()
     * @see flecs::world::readonly_end()
     */
    bool readonly_begin(bool multi_threaded = false) const {
        return ecs_readonly_begin(world_, multi_threaded);
    }

    /** End readonly mode.
     * 
     * @see ecs_readonly_end()
     * @see flecs::world::is_readonly()
     * @see flecs::world::readonly_begin()
     */
    void readonly_end() const {
        ecs_readonly_end(world_);
    }

    /** Defer operations until end of frame.
     * When this operation is invoked while iterating, operations inbetween the
     * defer_begin() and defer_end() operations are executed at the end of the frame.
     *
     * This operation is thread safe.
     *
     * @return true if world changed from non-deferred mode to deferred mode.
     *
     * @see ecs_defer_begin()
     * @see flecs::world::defer()
     * @see flecs::world::defer_end()
     * @see flecs::world::is_deferred()
     * @see flecs::world::defer_resume()
     * @see flecs::world::defer_suspend()
     */
    bool defer_begin() const {
        return ecs_defer_begin(world_);
    }

    /** End block of operations to defer.
     * See defer_begin().
     *
     * This operation is thread safe.
     *
     * @return true if world changed from deferred mode to non-deferred mode.
     *
     * @see ecs_defer_end()
     * @see flecs::world::defer()
     * @see flecs::world::defer_begin()
     * @see flecs::world::is_deferred()
     * @see flecs::world::defer_resume()
     * @see flecs::world::defer_suspend()
     */
    bool defer_end() const {
        return ecs_defer_end(world_);
    }

    /** Test whether deferring is enabled.
     *
     * @return True if deferred, false if not.
     *
     * @see ecs_is_deferred()
     * @see flecs::world::defer()
     * @see flecs::world::defer_begin()
     * @see flecs::world::defer_end()
     * @see flecs::world::defer_resume()
     * @see flecs::world::defer_suspend()
     */
    bool is_deferred() const {
        return ecs_is_deferred(world_);
    }

    /** Configure world to have N stages.
     * This initializes N stages, which allows applications to defer operations to
     * multiple isolated defer queues. This is typically used for applications with
     * multiple threads, where each thread gets its own queue, and commands are
     * merged when threads are synchronized.
     *
     * Note that set_threads() already creates the appropriate number of stages.
     * The set_stage_count() operation is useful for applications that want to manage
     * their own stages and/or threads.
     *
     * @param stages The number of stages.
     *
     * @see ecs_set_stage_count()
     * @see flecs::world::get_stage_count()
     */
    void set_stage_count(int32_t stages) const {
        ecs_set_stage_count(world_, stages);
    }

    /** Get number of configured stages.
     * Return number of stages set by set_stage_count().
     *
     * @return The number of stages used for threading.
     *
     * @see ecs_get_stage_count()
     * @see flecs::world::set_stage_count()
     */
    int32_t get_stage_count() const {
        return ecs_get_stage_count(world_);
    }

    /** Get current stage id.
     * The stage id can be used by an application to learn about which stage it
     * is using, which typically corresponds with the worker thread id.
     *
     * @return The stage id.
     */
    int32_t get_stage_id() const {
        return ecs_stage_get_id(world_);
    }

    /** Test if is a stage.
     * If this function returns false, it is guaranteed that this is a valid
     * world object.
     *
     * @return True if the world is a stage, false if not.
     */
    bool is_stage() const {
        ecs_assert(
            flecs_poly_is(world_, ecs_world_t) ||
            flecs_poly_is(world_, ecs_stage_t), 
                ECS_INVALID_PARAMETER, 
                "flecs::world instance contains invalid reference to world or stage");
        return flecs_poly_is(world_, ecs_stage_t);
    }

    /** Merge world or stage.
     * When automatic merging is disabled, an application can call this
     * operation on either an individual stage, or on the world which will merge
     * all stages. This operation may only be called when staging is not enabled
     * (either after progress() or after readonly_end()).
     *
     * This operation may be called on an already merged stage or world.
     *
     * @see ecs_merge()
     */
    void merge() const {
        ecs_merge(world_);
    }

    /** Get stage-specific world pointer.
     * Flecs threads can safely invoke the API as long as they have a private
     * context to write to, also referred to as the stage. This function returns a
     * pointer to a stage, disguised as a world pointer.
     *
     * Note that this function does not(!) create a new world. It simply wraps the
     * existing world in a thread-specific context, which the API knows how to
     * unwrap. The reason the stage is returned as an ecs_world_t is so that it
     * can be passed transparently to the existing API functions, vs. having to
     * create a dediated API for threading.
     *
     * @param stage_id The index of the stage to retrieve.
     * @return A thread-specific pointer to the world.
     */
    flecs::world get_stage(int32_t stage_id) const {
        return flecs::world(ecs_get_stage(world_, stage_id));
    }

    /** Create asynchronous stage.
     * An asynchronous stage can be used to asynchronously queue operations for
     * later merging with the world. An asynchronous stage is similar to a regular
     * stage, except that it does not allow reading from the world.
     *
     * Asynchronous stages are never merged automatically, and must therefore be
     * manually merged with the ecs_merge function. It is not necessary to call
     * defer_begin or defer_end before and after enqueuing commands, as an
     * asynchronous stage unconditionally defers operations.
     *
     * The application must ensure that no commands are added to the stage while the
     * stage is being merged.
     *
     * @return The stage.
     */
    flecs::world async_stage() const {
        ecs_world_t *as = ecs_stage_new(world_);
        flecs_poly_release(as); // world object will claim
        return flecs::world(as);
    }

    /** Get actual world.
     * If the current object points to a stage, this operation will return the
     * actual world.
     *
     * @return The actual world.
     */
    flecs::world get_world() const {
        /* Safe cast, mutability is checked */
        return flecs::world(
            world_ ? const_cast<flecs::world_t*>(ecs_get_world(world_)) : nullptr);
    }

    /** Test whether the current world object is readonly.
     * This function allows the code to test whether the currently used world
     * object is readonly or whether it allows for writing.
     *
     * @return True if the world or stage is readonly.
     *
     * @see ecs_stage_is_readonly()
     * @see flecs::world::readonly_begin()
     * @see flecs::world::readonly_end()
     */
    bool is_readonly() const {
        return ecs_stage_is_readonly(world_);
    }

    /** Set world context.
     * Set a context value that can be accessed by anyone that has a reference
     * to the world.
     *
     * @param ctx A pointer to a user defined structure.
     * @param ctx_free A function that is invoked with ctx when the world is freed.
     *
     *
     * @see ecs_set_ctx()
     * @see flecs::world::get_ctx()
     */
    void set_ctx(void* ctx, ecs_ctx_free_t ctx_free = nullptr) const {
        ecs_set_ctx(world_, ctx, ctx_free);
    }

    /** Get world context.
     * This operation retrieves a previously set world context.
     *
     * @return The context set with set_binding_ctx(). If no context was set, the
     *         function returns NULL.
     *
     * @see ecs_get_ctx()
     * @see flecs::world::set_ctx()
     */
    void* get_ctx() const {
        return ecs_get_ctx(world_);
    }

    /** Set world binding context.
     *
     * Same as set_ctx() but for binding context. A binding context is intended
     * specifically for language bindings to store binding specific data.
     *
     * @param ctx A pointer to a user defined structure.
     * @param ctx_free A function that is invoked with ctx when the world is freed.
     *
     * @see ecs_set_binding_ctx()
     * @see flecs::world::get_binding_ctx()
     */
    void set_binding_ctx(void* ctx, ecs_ctx_free_t ctx_free = nullptr) const {
        ecs_set_binding_ctx(world_, ctx, ctx_free);
    }

    /** Get world binding context.
     * This operation retrieves a previously set world binding context.
     *
     * @return The context set with set_binding_ctx(). If no context was set, the
     *         function returns NULL.
     *
     * @see ecs_get_binding_ctx()
     * @see flecs::world::set_binding_ctx()
     */
    void* get_binding_ctx() const {
        return ecs_get_binding_ctx(world_);
    }

    /** Preallocate memory for number of entities.
     * This function preallocates memory for the entity index.
     *
     * @param entity_count Number of entities to preallocate memory for.
     *
     * @see ecs_dim()
     */
    void dim(int32_t entity_count) const {
        ecs_dim(world_, entity_count);
    }

    /** Set entity range.
     * This function limits the range of issued entity ids between min and max.
     *
     * @param min Minimum entity id issued.
     * @param max Maximum entity id issued.
     *
     * @see ecs_set_entity_range()
     */
    void set_entity_range(entity_t min, entity_t max) const {
        ecs_set_entity_range(world_, min, max);
    }

    /** Enforce that operations cannot modify entities outside of range.
     * This function ensures that only entities within the specified range can
     * be modified. Use this function if specific parts of the code only are
     * allowed to modify a certain set of entities, as could be the case for
     * networked applications.
     *
     * @param enabled True if range check should be enabled, false if not.
     *
     * @see ecs_enable_range_check()
     */
    void enable_range_check(bool enabled = true) const {
        ecs_enable_range_check(world_, enabled);
    }

    /** Set current scope.
     *
     * @param scope The scope to set.
     * @return The current scope;
     *
     * @see ecs_set_scope()
     * @see flecs::world::get_scope()
     */
    flecs::entity set_scope(const flecs::entity_t scope) const;

    /** Get current scope.
     *
     * @return The current scope.
     *
     * @see ecs_get_scope()
     * @see flecs::world::set_scope()
     */
    flecs::entity get_scope() const;

    /** Same as set_scope but with type.
     *
     * @see ecs_set_scope()
     * @see flecs::world::get_scope()
     */
    template <typename T>
    flecs::entity set_scope() const;

    /** Set search path.
     *
     * @see ecs_set_lookup_path()
     * @see flecs::world::lookup()
     */
    flecs::entity_t* set_lookup_path(const flecs::entity_t *search_path) const {
        return ecs_set_lookup_path(world_, search_path);
    }

    /** Lookup entity by name.
     *
     * @param name Entity name.
     * @param recursive When false, only the current scope is searched.
     * @result The entity if found, or 0 if not found.
     */
    flecs::entity lookup(const char *name, const char *sep = "::", const char *root_sep = "::", bool recursive = true) const;

    /** Set singleton component.
     */
    template <typename T, if_t< !is_callable<T>::value > = 0>
    void set(const T& value) const {
        flecs::set<T>(world_, _::type<T>::id(world_), value);
    }

    /** Set singleton component.
     */
    template <typename T, if_t< !is_callable<T>::value > = 0>
    void set(T&& value) const {
        flecs::set<T>(world_, _::type<T>::id(world_),
            FLECS_FWD(value));
    }

    /** Set singleton pair.
     */
    template <typename First, typename Second, typename P = flecs::pair<First, Second>,
        typename A = actual_type_t<P>, if_not_t< flecs::is_pair<First>::value> = 0>
    void set(const A& value) const {
        flecs::set<P>(world_, _::type<First>::id(world_), value);
    }

    /** Set singleton pair.
     */
    template <typename First, typename Second, typename P = flecs::pair<First, Second>,
        typename A = actual_type_t<P>, if_not_t< flecs::is_pair<First>::value> = 0>
    void set(A&& value) const {
        flecs::set<P>(world_, _::type<First>::id(world_), FLECS_FWD(value));
    }

    /** Set singleton pair.
     */
    template <typename First, typename Second>
    void set(Second second, const First& value) const;

    /** Set singleton pair.
     */
    template <typename First, typename Second>
    void set(Second second, First&& value) const;

    /** Set singleton component inside a callback.
     */
    template <typename Func, if_t< is_callable<Func>::value > = 0 >
    void set(const Func& func) const;

    template <typename T, typename ... Args>
    void emplace(Args&&... args) const {
        flecs::id_t component_id = _::type<T>::id(world_);
        flecs::emplace<T>(world_, component_id, component_id, FLECS_FWD(args)...);
    }

    /** Ensure singleton component.
     */
    #ifndef ensure
    template <typename T>
    T& ensure() const;
    #endif

    /** Mark singleton component as modified.
     */
    template <typename T>
    void modified() const;

    /** Get ref singleton component.
     */
    template <typename T>
    ref<T> get_ref() const;

    /** Get singleton component.
     */
    template <typename T>
    const T* get() const;

    /** Get singleton pair.
     */
    template <typename First, typename Second, typename P = flecs::pair<First, Second>,
        typename A = actual_type_t<P>>
    const A* get() const;

    /** Get singleton pair.
     */
    template <typename First, typename Second>
    const First* get(Second second) const;

    /** Get singleton component inside a callback.
     */
    template <typename Func, if_t< is_callable<Func>::value > = 0 >
    void get(const Func& func) const;

    /** Get mutable singleton component.
     */
    template <typename T>
    T* get_mut() const;

    /** Get mutable singleton pair.
     */
    template <typename First, typename Second, typename P = flecs::pair<First, Second>,
        typename A = actual_type_t<P>>
    A* get_mut() const;

    /** Get mutable singleton pair.
     */
    template <typename First, typename Second>
    First* get_mut(Second second) const;

    /** Test if world has singleton component.
     */
    template <typename T>
    bool has() const;

    /** Test if world has the provided pair.
     *
     * @tparam First The first element of the pair
     * @tparam Second The second element of the pair
     */
    template <typename First, typename Second>
    bool has() const;

    /** Test if world has the provided pair.
     *
     * @tparam First The first element of the pair
     * @param second The second element of the pair.
     */
    template <typename First>
    bool has(flecs::id_t second) const;

    /** Test if world has the provided pair.
     *
     * @param first The first element of the pair
     * @param second The second element of the pair
     */
    bool has(flecs::id_t first, flecs::id_t second) const;

    /** Add singleton component.
     */
    template <typename T>
    void add() const;

    /** Adds a pair to the singleton component.
     *
     * @tparam First The first element of the pair
     * @tparam Second The second element of the pair
     */
    template <typename First, typename Second>
    void add() const;

    /** Adds a pair to the singleton component.
     *
     * @tparam First The first element of the pair
     * @param second The second element of the pair.
     */
    template <typename First>
    void add(flecs::entity_t second) const;

    /** Adds a pair to the singleton entity.
     *
     * @param first The first element of the pair
     * @param second The second element of the pair
     */
    void add(flecs::entity_t first, flecs::entity_t second) const;

    /** Remove singleton component.
     */
    template <typename T>
    void remove() const;

    /** Removes the pair singleton component.
     *
     * @tparam First The first element of the pair
     * @tparam Second The second element of the pair
     */
    template <typename First, typename Second>
    void remove() const;

    /** Removes the pair singleton component.
     *
     * @tparam First The first element of the pair
     * @param second The second element of the pair.
     */
    template <typename First>
    void remove(flecs::entity_t second) const;

    /** Removes the pair singleton component.
     *
     * @param first The first element of the pair
     * @param second The second element of the pair
     */
    void remove(flecs::entity_t first, flecs::entity_t second) const;

    /** Iterate entities in root of world
     * Accepts a callback with the following signature:
     *
     * @code
     * void(*)(flecs::entity e);
     * @endcode
     */
    template <typename Func>
    void children(Func&& f) const;

    /** Get singleton entity for type.
     */
    template <typename T>
    flecs::entity singleton() const;

    /** Get target for a given pair from a singleton entity.
     * This operation returns the target for a given pair. The optional
     * index can be used to iterate through targets, in case the entity has
     * multiple instances for the same relationship.
     *
     * @tparam First The first element of the pair.
     * @param index The index (0 for the first instance of the relationship).
     */
    template<typename First>
    flecs::entity target(int32_t index = 0) const;

    /** Get target for a given pair from a singleton entity.
     * This operation returns the target for a given pair. The optional
     * index can be used to iterate through targets, in case the entity has
     * multiple instances for the same relationship.
     *
     * @param first The first element of the pair for which to retrieve the target.
     * @param index The index (0 for the first instance of the relationship).
     */
    template<typename T>
    flecs::entity target(flecs::entity_t first, int32_t index = 0) const;

    /** Get target for a given pair from a singleton entity.
     * This operation returns the target for a given pair. The optional
     * index can be used to iterate through targets, in case the entity has
     * multiple instances for the same relationship.
     *
     * @param first The first element of the pair for which to retrieve the target.
     * @param index The index (0 for the first instance of the relationship).
     */
    flecs::entity target(flecs::entity_t first, int32_t index = 0) const;

    /** Create alias for component.
     *
     * @tparam T to create an alias for.
     * @param alias Alias for the component.
     * @return Entity representing the component.
     */
    template <typename T>
    flecs::entity use(const char *alias = nullptr) const;

    /** Create alias for entity.
     *
     * @param name Name of the entity.
     * @param alias Alias for the entity.
     */
    flecs::entity use(const char *name, const char *alias = nullptr) const;

    /** Create alias for entity.
     *
     * @param entity Entity for which to create the alias.
     * @param alias Alias for the entity.
     */
    void use(flecs::entity entity, const char *alias = nullptr) const;

    /** Count entities matching a component.
     *
     * @param component_id The component id.
     */
    int count(flecs::id_t component_id) const {
        return ecs_count_id(world_, component_id);
    }

    /** Count entities matching a pair.
     *
     * @param first The first element of the pair.
     * @param second The second element of the pair.
     */
    int count(flecs::entity_t first, flecs::entity_t second) const {
        return ecs_count_id(world_, ecs_pair(first, second));
    }

    /** Count entities matching a component.
     *
     * @tparam T The component type.
     */
    template <typename T>
    int count() const {
        return count(_::type<T>::id(world_));
    }

    /** Count entities matching a pair.
     *
     * @tparam First The first element of the pair.
     * @param second The second element of the pair.
     */
    template <typename First>
    int count(flecs::entity_t second) const {
        return count(_::type<First>::id(world_), second);
    }

    /** Count entities matching a pair.
     *
     * @tparam First The first element of the pair.
     * @tparam Second The second element of the pair.
     */
    template <typename First, typename Second>
    int count() const {
        return count(
            _::type<First>::id(world_),
            _::type<Second>::id(world_));
    }

    /** All entities created in function are created with id.
     */
    template <typename Func>
    void with(id_t with_id, const Func& func) const {
        ecs_id_t prev = ecs_set_with(world_, with_id);
        func();
        ecs_set_with(world_, prev);
    }

    /** All entities created in function are created with type.
     */
    template <typename T, typename Func>
    void with(const Func& func) const {
        with(this->id<T>(), func);
    }

    /** All entities created in function are created with pair.
     */
    template <typename First, typename Second, typename Func>
    void with(const Func& func) const {
        with(ecs_pair(this->id<First>(), this->id<Second>()), func);
    }

    /** All entities created in function are created with pair.
     */
    template <typename First, typename Func>
    void with(id_t second, const Func& func) const {
        with(ecs_pair(this->id<First>(), second), func);
    }

    /** All entities created in function are created with pair.
     */
    template <typename Func>
    void with(id_t first, id_t second, const Func& func) const {
        with(ecs_pair(first, second), func);
    }

    /** All entities created in function are created in scope. All operations
     * called in function (such as lookup) are relative to scope.
     */
    template <typename Func>
    void scope(id_t parent, const Func& func) const {
        ecs_entity_t prev = ecs_set_scope(world_, parent);
        func();
        ecs_set_scope(world_, prev);
    }

    /** Same as scope(parent, func), but with T as parent.
     */
    template <typename T, typename Func>
    void scope(const Func& func) const {
        flecs::id_t parent = _::type<T>::id(world_);
        scope(parent, func);
    }

    /** Use provided scope for operations ran on returned world.
     * Operations need to be ran in a single statement.
     */
    flecs::scoped_world scope(id_t parent) const;

    template <typename T>
    flecs::scoped_world scope() const;

    flecs::scoped_world scope(const char* name) const;

    /** Delete all entities with specified id. */
    void delete_with(id_t the_id) const {
        ecs_delete_with(world_, the_id);
    }

    /** Delete all entities with specified pair. */
    void delete_with(entity_t first, entity_t second) const {
        delete_with(ecs_pair(first, second));
    }

    /** Delete all entities with specified component. */
    template <typename T>
    void delete_with() const {
        delete_with(_::type<T>::id(world_));
    }

    /** Delete all entities with specified pair. */
    template <typename First, typename Second>
    void delete_with() const {
        delete_with(_::type<First>::id(world_), _::type<Second>::id(world_));
    }

    /** Delete all entities with specified pair. */
    template <typename First>
    void delete_with(entity_t second) const {
        delete_with(_::type<First>::id(world_), second);
    }

    /** Remove all instances of specified id. */
    void remove_all(id_t the_id) const {
        ecs_remove_all(world_, the_id);
    }

    /** Remove all instances of specified pair. */
    void remove_all(entity_t first, entity_t second) const {
        remove_all(ecs_pair(first, second));
    }

    /** Remove all instances of specified component. */
    template <typename T>
    void remove_all() const {
        remove_all(_::type<T>::id(world_));
    }

    /** Remove all instances of specified pair. */
    template <typename First, typename Second>
    void remove_all() const {
        remove_all(_::type<First>::id(world_), _::type<Second>::id(world_));
    }

    /** Remove all instances of specified pair. */
    template <typename First>
    void remove_all(entity_t second) const {
        remove_all(_::type<First>::id(world_), second);
    }

    /** Defer all operations called in function.
     *
     * @see flecs::world::defer_begin()
     * @see flecs::world::defer_end()
     * @see flecs::world::defer_is_deferred()
     * @see flecs::world::defer_resume()
     * @see flecs::world::defer_suspend()
     */
    template <typename Func>
    void defer(const Func& func) const {
        ecs_defer_begin(world_);
        func();
        ecs_defer_end(world_);
    }

    /** Suspend deferring operations.
     *
     * @see ecs_defer_suspend()
     * @see flecs::world::defer()
     * @see flecs::world::defer_begin()
     * @see flecs::world::defer_end()
     * @see flecs::world::defer_is_deferred()
     * @see flecs::world::defer_resume()
     */
    void defer_suspend() const {
        ecs_defer_suspend(world_);
    }

    /** Resume deferring operations.
     *
     * @see ecs_defer_resume()
     * @see flecs::world::defer()
     * @see flecs::world::defer_begin()
     * @see flecs::world::defer_end()
     * @see flecs::world::defer_is_deferred()
     * @see flecs::world::defer_suspend()
     */
    void defer_resume() const {
        ecs_defer_resume(world_);
    }

    /** Check if entity id exists in the world.
     *
     * @see ecs_exists()
     * @see flecs::world::is_alive()
     * @see flecs::world::is_valid()
     */
    bool exists(flecs::entity_t e) const {
        return ecs_exists(world_, e);
    }

    /** Check if entity id exists in the world.
     *
     * @see ecs_is_alive()
     * @see flecs::world::exists()
     * @see flecs::world::is_valid()
     */
    bool is_alive(flecs::entity_t e) const {
        return ecs_is_alive(world_, e);
    }

    /** Check if entity id is valid.
     * Invalid entities cannot be used with API functions.
     *
     * @see ecs_is_valid()
     * @see flecs::world::exists()
     * @see flecs::world::is_alive()
     */
    bool is_valid(flecs::entity_t e) const {
        return ecs_is_valid(world_, e);
    }

    /** Get alive entity for id.
     * Returns the entity with the current generation.
     *
     * @see ecs_get_alive()
     */
    flecs::entity get_alive(flecs::entity_t e) const;

    /**
     * @see ecs_make_alive()
     */
    flecs::entity make_alive(flecs::entity_t e) const;

    /** Set version of entity to provided.
     * 
     * @see ecs_set_version()
     */
    void set_version(flecs::entity_t e) const {
        ecs_set_version(world_, e);
    }

    /* Run callback after completing frame */
    void run_post_frame(ecs_fini_action_t action, void *ctx) const {
        ecs_run_post_frame(world_, action, ctx);
    }

    /** Get the world info.
     *
     * @see ecs_get_world_info()
     */
    const flecs::world_info_t* get_info() const{
        return ecs_get_world_info(world_);
    }

    /** Get delta_time */
    ecs_ftime_t delta_time() const {
        return get_info()->delta_time;
    }

/**
 * @file addons/cpp/mixins/id/mixin.inl
 * @brief Id world mixin.
 */

/** Get id from a type.
 * 
 * @memberof flecs::world
 */
template <typename T>
flecs::id id() const;

/** Id factory.
 * 
 * @memberof flecs::world
 */
template <typename ... Args>
flecs::id id(Args&&... args) const;

/** Get pair id from relationship, object.
 * 
 * @memberof flecs::world
 */
template <typename First, typename Second>
flecs::id pair() const;

/** Get pair id from relationship, object.
 * 
 * @memberof flecs::world
 */
template <typename First>
flecs::id pair(entity_t o) const;

/** Get pair id from relationship, object.
 * 
 * @memberof flecs::world
 */
flecs::id pair(entity_t r, entity_t o) const;

/**
 * @file addons/cpp/mixins/component/mixin.inl
 * @brief Component mixin.
 */

/** Find or register component.
 * 
 * @ingroup cpp_components
 * @memberof flecs::world
 */
template <typename T, typename... Args>
flecs::component<T> component(Args &&... args) const;

/** Find or register untyped component.
 * Method available on flecs::world class.
 * 
 * @ingroup cpp_components
 * @memberof flecs::world
 */
template <typename... Args>
flecs::untyped_component component(Args &&... args) const;

/**
 * @file addons/cpp/mixins/entity/mixin.inl
 * @brief Entity world mixin.
 */

/** Create an entity.
 * 
 * @memberof flecs::world
 * @ingroup cpp_entities
 */
template <typename... Args>
flecs::entity entity(Args &&... args) const;

/** Convert enum constant to entity.
 * 
 * @memberof flecs::world
 * @ingroup cpp_entities
 */
template <typename E, if_t< is_enum<E>::value > = 0>
flecs::id id(E value) const;

/** Convert enum constant to entity.
 * 
 * @memberof flecs::world
 * @ingroup cpp_entities
 */
template <typename E, if_t< is_enum<E>::value > = 0>
flecs::entity entity(E value) const;

/** Create a prefab.
 * 
 * @memberof flecs::world
 * @ingroup cpp_entities
 */
template <typename... Args>
flecs::entity prefab(Args &&... args) const;

/** Create an entity that's associated with a type.
 * 
 * @memberof flecs::world
 * @ingroup cpp_entities
 */
template <typename T>
flecs::entity entity(const char *name = nullptr) const;

/** Create a prefab that's associated with a type.
 * 
 * @memberof flecs::world
 * @ingroup cpp_entities
 */
template <typename T>
flecs::entity prefab(const char *name = nullptr) const;

/**
 * @file addons/cpp/mixins/event/mixin.inl
 * @brief Event world mixin.
 */

/**
 * @defgroup cpp_addons_event Events
 * @ingroup cpp_addons
 * API for emitting events.
 *
 * @{
 */

/** Create a new event.
 *
 * @memberof flecs::world
 *
 * @param evt The event id.
 * @return Event builder.
 */
flecs::event_builder event(flecs::entity_t evt) const;

/** Create a new event.
 *
 * @memberof flecs::world
 *
 * @tparam E The event type.
 * @return Event builder.
 */
template <typename E>
flecs::event_builder_typed<E> event() const;

/** @} */

/**
 * @file addons/cpp/mixins/term/mixin.inl
 * @brief Term world mixin.
 */

/**
 * @memberof flecs::world
 * @ingroup cpp_core_queries
 *
 * @{
 */

/** Create a term.
 * 
 */
template<typename... Args>
flecs::term term(Args &&... args) const;

/** Create a term for a (component) type.
 */
template<typename T>
flecs::term term() const;  

/** Create a term for a pair.
 */
template<typename First, typename Second>
flecs::term term() const;

/** @} */

/**
 * @file addons/cpp/mixins/observer/mixin.inl
 * @brief Observer world mixin.
 */

/** Observer builder.
 * 
 * @memberof flecs::world
 * @ingroup cpp_observers
 *
 * @{
 */

/** Upcast entity to an observer.
 * The provided entity must be an observer.
 * 
 * @param e The entity.
 * @return An observer object.
 */
flecs::observer observer(flecs::entity e) const;

/** Create a new observer.
 * 
 * @tparam Components The components to match on.
 * @tparam Args Arguments passed to the constructor of flecs::observer_builder.
 * @return Observer builder.
 */
template <typename... Components, typename... Args>
flecs::observer_builder<Components...> observer(Args &&... args) const;

/** @} */

/**
 * @file addons/cpp/mixins/query/mixin.inl
 * @brief Query world mixin.
 */

/**
 * @memberof flecs::world
 * @ingroup cpp_core_queries
 *
 * @{
 */

/** Create a query.
 * 
 * @see ecs_query_init
 */
template <typename... Comps, typename... Args>
flecs::query<Comps...> query(Args &&... args) const;

/** Create a query from entity.
 * 
 * @see ecs_query_init
 */
flecs::query<> query(flecs::entity query_entity) const;

/** Create a query builder.
 * 
 * @see ecs_query_init
 */
template <typename... Comps, typename... Args>
flecs::query_builder<Comps...> query_builder(Args &&... args) const;

/** Iterate over all entities with components in argument list of function.
 * The function parameter must match the following signature:
 *
 * @code
 * void(*)(T&, U&, ...)
 * @endcode
 *
 * or:
 *
 * @code
 * void(*)(flecs::entity, T&, U&, ...)
 * @endcode
 * 
 */
template <typename Func>
void each(Func&& func) const;

/** Iterate over all entities with provided component.
 * The function parameter must match the following signature:
 *
 * @code
 * void(*)(T&)
 * @endcode
 *
 * or:
 *
 * @code
 * void(*)(flecs::entity, T&)
 * @endcode
 * 
 */
template <typename T, typename Func>
void each(Func&& func) const;

/** Iterate over all entities with provided (component) id. */
template <typename Func>
void each(flecs::id_t term_id, Func&& func) const;

/** @} */

/**
 * @file addons/cpp/mixins/enum/mixin.inl
 * @brief Enum world mixin.
 */

/** Convert enum constant to entity.
 * 
 * @memberof flecs::world
 * @ingroup cpp_entities
 */
template <typename E, if_t< is_enum<E>::value > = 0>
flecs::entity to_entity(E constant) const;


#   ifdef FLECS_MODULE
/**
 * @file addons/cpp/mixins/module/mixin.inl
 * @brief Module world mixin.
 */

/** 
 * @memberof flecs::world
 * @ingroup cpp_addons_modules
 * 
 * @{
 */

/** Define a module.
 * This operation is not mandatory, but can be called inside the module ctor to
 * obtain the entity associated with the module, or override the module name.
 * 
 * @tparam Module module class.
 * @return Module entity.
 */
template <typename Module>
flecs::entity module(const char *name = nullptr) const;

/** Import a module.
 * 
 * @tparam Module module class.
 * @return Module entity.
 */
template <typename Module>
flecs::entity import();

/** @} */

#   endif
#   ifdef FLECS_PIPELINE
/**
 * @file addons/cpp/mixins/pipeline/mixin.inl
 * @brief Pipeline world mixin.
 */

/**
 * @memberof flecs::world
 * @ingroup cpp_pipelines
 *
 * @{
 */

/** Create a new pipeline.
 *
 * @return A pipeline builder.
 */
flecs::pipeline_builder<> pipeline() const;

/** Create a new pipeline.
 *
 * @tparam Pipeline Type associated with pipeline.
 * @return A pipeline builder.
 */
template <typename Pipeline, if_not_t< is_enum<Pipeline>::value > = 0>
flecs::pipeline_builder<> pipeline() const;

/** Set pipeline.
 * @see ecs_set_pipeline
 */
void set_pipeline(const flecs::entity pip) const;

/** Set pipeline.
 * @see ecs_set_pipeline
 */
template <typename Pipeline>
void set_pipeline() const;

/** Get pipeline.
 * @see ecs_get_pipeline
 */
flecs::entity get_pipeline() const;

/** Progress world one tick.
 * @see ecs_progress
 */
bool progress(ecs_ftime_t delta_time = 0.0) const;

/** Run pipeline.
 * @see ecs_run_pipeline
 */
void run_pipeline(const flecs::entity_t pip, ecs_ftime_t delta_time = 0.0) const;

/** Run pipeline.
 * @tparam Pipeline Type associated with pipeline.
 * @see ecs_run_pipeline
 */
template <typename Pipeline, if_not_t< is_enum<Pipeline>::value > = 0>
void run_pipeline(ecs_ftime_t delta_time = 0.0) const;

/** Set timescale.
 * @see ecs_set_time_scale
 */
void set_time_scale(ecs_ftime_t mul) const;

/** Set target FPS.
 * @see ecs_set_target_fps
 */
void set_target_fps(ecs_ftime_t target_fps) const;

/** Reset simulation clock.
 * @see ecs_reset_clock
 */
void reset_clock() const;

/** Set number of threads.
 * @see ecs_set_threads
 */
void set_threads(int32_t threads) const;

/** Set number of threads.
 * @see ecs_get_stage_count
 */
int32_t get_threads() const;

/** Set number of task threads.
 * @see ecs_set_task_threads
 */
void set_task_threads(int32_t task_threads) const;

/** Returns true if task thread use has been requested.
 * @see ecs_using_task_threads
 */
bool using_task_threads() const;

/** @} */

#   endif
#   ifdef FLECS_SYSTEM
/**
 * @file addons/cpp/mixins/system/mixin.inl
 * @brief System module world mixin.
 */

/** 
 * @memberof flecs::world
 * @ingroup cpp_addons_systems
 *
 * @{
*/

/** Upcast entity to a system.
 * The provided entity must be a system.
 * 
 * @param e The entity.
 * @return A system object.
 */
flecs::system system(flecs::entity e) const;

/** Create a new system.
 * 
 * @tparam Components The components to match on.
 * @tparam Args Arguments passed to the constructor of flecs::system_builder.
 * @return System builder.
 */
template <typename... Components, typename... Args>
flecs::system_builder<Components...> system(Args &&... args) const;

/** @} */

#   endif
#   ifdef FLECS_TIMER
/**
 * @file addons/cpp/mixins/timer/mixin.inl
 * @brief Timer module mixin.
 */

/**
 * @memberof flecs::world
 * @ingroup cpp_addons_timer
 */

/** Find or register a singleton timer. */
template <typename T>
flecs::timer timer() const;

/** Find or register a timer. */
template <typename... Args>
flecs::timer timer(Args &&... args) const;

/** Enable randomization of initial time values for timers.
 * @see ecs_randomize_timers
 */
void randomize_timers() const;

#   endif
#   ifdef FLECS_SCRIPT
/**
 * @file addons/cpp/mixins/script/mixin.inl
 * @brief Script world mixin.
 */

/**
 * @defgroup cpp_addons_script Script
 * @ingroup cpp_addons
 * Data definition format for loading entity data.
 *
 * @{
 */

/** Run script.
 * @see ecs_script_run
 */
int script_run(const char *name, const char *str) const {
    return ecs_script_run(world_, name, str);
}

/** Run script from file.
 * @see ecs_script_run_file
 */
int script_run_file(const char *filename) const {
    return ecs_script_run_file(world_, filename);
}

/** Build script.
 * @see ecs_script_init
 */
script_builder script(const char *name = nullptr) const {
    return script_builder(world_, name);
}

/** Convert value to string */
flecs::string to_expr(flecs::entity_t tid, const void* value) {
    char *expr = ecs_ptr_to_expr(world_, tid, value);
    return flecs::string(expr);
}

/** Convert value to string */
template <typename T>
flecs::string to_expr(const T* value) {
    flecs::entity_t tid = _::type<T>::id(world_);
    return to_expr(tid, value);
}


/** @} */

#   endif
#   ifdef FLECS_META
/**
 * @file addons/cpp/mixins/meta/world.inl
 * @brief Meta world mixin.
 */

/**
 * @memberof flecs::world
 * @ingroup cpp_addons_meta
 * 
 * @{
 */

/** Return meta cursor to value */
flecs::cursor cursor(flecs::entity_t tid, void *ptr) {
    return flecs::cursor(world_, tid, ptr);
}

/** Return meta cursor to value */
template <typename T>
flecs::cursor cursor(void *ptr) {
    flecs::entity_t tid = _::type<T>::id(world_);
    return cursor(tid, ptr);
}

/** Create primitive type */
flecs::entity primitive(flecs::meta::primitive_kind_t kind);

/** Create array type. */
flecs::entity array(flecs::entity_t elem_id, int32_t array_count);

/** Create array type. */
template <typename T>
flecs::entity array(int32_t array_count);

/** Create vector type. */
flecs::entity vector(flecs::entity_t elem_id);

/** Create vector type. */
template <typename T>
flecs::entity vector();

/** @} */

#   endif
#   ifdef FLECS_JSON
/**
 * @file addons/cpp/mixins/json/world.inl
 * @brief JSON world mixin.
 */

/** Serialize untyped value to JSON.
 * 
 * @memberof flecs::world
 * @ingroup cpp_addons_json
 */
flecs::string to_json(flecs::entity_t tid, const void* value) {
    char *json = ecs_ptr_to_json(world_, tid, value);
    return flecs::string(json);
}

/** Serialize value to JSON.
 * 
 * @memberof flecs::world
 * @ingroup cpp_addons_json
 */
template <typename T>
flecs::string to_json(const T* value) {
    flecs::entity_t tid = _::type<T>::id(world_);
    return to_json(tid, value);
}

/** Serialize world to JSON.
 * 
 * @memberof flecs::world
 * @ingroup cpp_addons_json
 */
flecs::string to_json() {
    return flecs::string( ecs_world_to_json(world_, nullptr) );
}

/** Deserialize value from JSON.
 * 
 * @memberof flecs::world
 * @ingroup cpp_addons_json
 */
const char* from_json(flecs::entity_t tid, void* value, const char *json, flecs::from_json_desc_t *desc = nullptr) {
    return ecs_ptr_from_json(world_, tid, value, json, desc);
}

/** Deserialize value from JSON.
 * 
 * @memberof flecs::world
 * @ingroup cpp_addons_json
 */
template <typename T>
const char* from_json(T* value, const char *json, flecs::from_json_desc_t *desc = nullptr) {
    return ecs_ptr_from_json(world_, _::type<T>::id(world_),
        value, json, desc);
}

/** Deserialize JSON into world.
 * 
 * @memberof flecs::world
 * @ingroup cpp_addons_json
 */
const char* from_json(const char *json, flecs::from_json_desc_t *desc = nullptr) {
    return ecs_world_from_json(world_, json, desc);
}

/** Deserialize JSON file into world.
 * 
 * @memberof flecs::world
 * @ingroup cpp_addons_json
 */
const char* from_json_file(const char *json, flecs::from_json_desc_t *desc = nullptr) {
    return ecs_world_from_json_file(world_, json, desc);
}

#   endif
#   ifdef FLECS_APP
/**
 * @file addons/cpp/mixins/app/mixin.inl
 * @brief App world addon mixin.
 */

/**
 * @ingroup cpp_addons_app
 * @memberof flecs::world
 *
 * @{
 */

/** Return app builder.
 * The app builder is a convenience wrapper around a loop that runs 
 * world::progress. An app allows for writing platform agnostic code,
 * as it provides hooks to modules for overtaking the main loop which is 
 * required for frameworks like emscripten.
 */
flecs::app_builder app() {
    flecs::world_t *w = world_;
    world_ = nullptr; // Take ownership
    return flecs::app_builder(w);
}

/** @} */

#   endif
#   ifdef FLECS_METRICS

/** Create metric.
 * 
 * @ingroup cpp_addons_metrics
 * @memberof flecs::world
 */
template <typename... Args>
flecs::metric_builder metric(Args &&... args) const;

#   endif
#   ifdef FLECS_ALERTS

/** Create alert.
 * 
 * @ingroup cpp_addons_alerts
 * @memberof flecs::world
 */
template <typename... Comps, typename... Args>
flecs::alert_builder<Comps...> alert(Args &&... args) const;

#   endif

public:
    void init_builtin_components();

    world_t *world_;
};

/** Scoped world.
 * Utility class used by the world::scope method to create entities in a scope.
 */
struct scoped_world : world {
    scoped_world(
        flecs::world_t *w,
        flecs::entity_t s) : world(w)
    {
        prev_scope_ = ecs_set_scope(w, s);
    }

    ~scoped_world() {
        ecs_set_scope(world_, prev_scope_);
    }

    scoped_world(const scoped_world& obj) : world(nullptr) {
        prev_scope_ = obj.prev_scope_;
        world_ = obj.world_;
        flecs_poly_claim(world_);
    }

    flecs::entity_t prev_scope_;
};

/** @} */

} // namespace flecs


/**
 * @file addons/cpp/field.hpp
 * @brief Wrapper classes for fields returned by flecs::iter.
 */

#pragma once

/**
 * @defgroup cpp_field Fields
 * @ingroup cpp_core
 * Field helper types.
 *
 * @{
 */

namespace flecs
{

/** Unsafe wrapper class around a field.
 * This class can be used when a system does not know the type of a field at
 * compile time.
 *
 * @ingroup cpp_iterator
 */
struct untyped_field {
    untyped_field(void* array, size_t size, size_t count, bool is_shared = false)
        : data_(array)
        , size_(size)
        , count_(count)
        , is_shared_(is_shared) {}

    /** Return element in component array.
     * This operator may only be used if the field is not shared.
     *
     * @param index Index of element.
     * @return Reference to element.
     */
    void* operator[](size_t index) const {
        ecs_assert(!is_shared_ || !index, ECS_INVALID_PARAMETER,
            "invalid usage of [] operator for shared component field");
        ecs_assert(index < count_, ECS_COLUMN_INDEX_OUT_OF_RANGE,
            "index %d out of range for field", index);
        return ECS_OFFSET(data_, size_ * index);
    }

protected:
    void* data_;
    size_t size_;
    size_t count_;
    bool is_shared_;
};

/** Wrapper class around a field.
 *
 * @tparam T component type of the field.
 *
 * @ingroup cpp_iterator
 */
template <typename T>
struct field {
    static_assert(std::is_empty<T>::value == false,
        "invalid type for field, cannot iterate empty type");

    /** Create field from component array.
     *
     * @param array Pointer to the component array.
     * @param count Number of elements in component array.
     * @param is_shared Is the component shared or not.
     */
    field(T* array, size_t count, bool is_shared = false)
        : data_(array)
        , count_(count)
        , is_shared_(is_shared) {}

    /** Create field from iterator.
     *
     * @param iter Iterator object.
     * @param field Index of the signature of the query being iterated over.
     */
    field(iter &iter, int field);

    /** Return element in component array.
     * This operator may only be used if the field is not shared.
     *
     * @param index Index of element.
     * @return Reference to element.
     */
    T& operator[](size_t index) const;

    /** Return first element of component array.
     * This operator is typically used when the field is shared.
     *
     * @return Reference to the first element.
     */
    T& operator*() const;

    /** Return first element of component array.
     * This operator is typically used when the field is shared.
     *
     * @return Pointer to the first element.
     */
    T* operator->() const;

protected:
    T* data_;
    size_t count_;
    bool is_shared_;
};

} // namespace flecs

/** @} */

/**
 * @file addons/cpp/iter.hpp
 * @brief Wrapper classes for ecs_iter_t and component arrays.
 */

#pragma once

/**
 * @defgroup cpp_iterator Iterators
 * @ingroup cpp_core
 * Iterator operations.
 *
 * @{
 */

namespace flecs
{

////////////////////////////////////////////////////////////////////////////////

namespace _ {

////////////////////////////////////////////////////////////////////////////////

/** Iterate over an integer range (used to iterate over entity range).
 *
 * @tparam T of the iterator
 */
template <typename T>
struct range_iterator
{
    explicit range_iterator(T value)
        : value_(value){}

    bool operator!=(range_iterator const& other) const
    {
        return value_ != other.value_;
    }

    T const& operator*() const
    {
        return value_;
    }

    range_iterator& operator++()
    {
        ++value_;
        return *this;
    }

private:
    T value_;
};

} // namespace _

} // namespace flecs

namespace flecs
{

////////////////////////////////////////////////////////////////////////////////

/** Class for iterating over query results.
 *
 * @ingroup cpp_iterator
 */
struct iter {
private:
    using row_iterator = _::range_iterator<size_t>;

public:
    /** Construct iterator from C iterator object.
     * This operation is typically not invoked directly by the user.
     *
     * @param it Pointer to C iterator.
     */
    iter(ecs_iter_t *it) : iter_(it) { }

    row_iterator begin() const {
        return row_iterator(0);
    }

    row_iterator end() const {
        return row_iterator(static_cast<size_t>(iter_->count));
    }

    flecs::entity system() const;

    flecs::entity event() const;

    flecs::id event_id() const;

    flecs::world world() const;

    const flecs::iter_t* c_ptr() const {
        return iter_;
    }

    size_t count() const {
        ecs_check(iter_->flags & EcsIterIsValid, ECS_INVALID_PARAMETER,
            "operation invalid before calling next()");
        return static_cast<size_t>(iter_->count);
    error:
        return 0;
    }

    ecs_ftime_t delta_time() const {
        return iter_->delta_time;
    }

    ecs_ftime_t delta_system_time() const {
        return iter_->delta_system_time;
    }

    flecs::type type() const;

    flecs::table table() const;

    flecs::table other_table() const;

    flecs::table_range range() const;

    /** Access ctx.
     * ctx contains the context pointer assigned to a system.
     */
    void* ctx() {
        return iter_->ctx;
    }

    /** Access ctx.
     * ctx contains the context pointer assigned to a system.
     */
    template <typename T>
    T* ctx() {
        return static_cast<T*>(iter_->ctx);
    }

    /** Access param.
     * param contains the pointer passed to the param argument of system::run
     */
    void* param() {
        return iter_->param;
    }

    /** Access param.
     * param contains the pointer passed to the param argument of system::run
     */
    template <typename T>
    T* param() {
        /* TODO: type check */
        return static_cast<T*>(iter_->param);
    }

    /** Obtain mutable handle to entity being iterated over.
     *
     * @param row Row being iterated over.
     */
    flecs::entity entity(size_t row) const;

    /** Returns whether field is matched on self.
     *
     * @param index The field index.
     */
    bool is_self(int8_t index) const {
        return ecs_field_is_self(iter_, index);
    }

    /** Returns whether field is set.
     *
     * @param index The field index.
     */
    bool is_set(int8_t index) const {
        return ecs_field_is_set(iter_, index);
    }

    /** Returns whether field is readonly.
     *
     * @param index The field index.
     */
    bool is_readonly(int8_t index) const {
        return ecs_field_is_readonly(iter_, index);
    }

    /** Number of fields in iterator.
     */
    int32_t field_count() const {
        return iter_->field_count;
    }

    /** Size of field data type.
     *
     * @param index The field id.
     */
    size_t size(int8_t index) const {
        return ecs_field_size(iter_, index);
    }

    /** Obtain field source (0 if This).
     *
     * @param index The field index.
     */
    flecs::entity src(int8_t index) const;

    /** Obtain id matched for field.
     *
     * @param index The field index.
     */
    flecs::id id(int8_t index) const;

    /** Obtain pair id matched for field.
     * This operation will fail if the id is not a pair.
     *
     * @param index The field index.
     */
    flecs::id pair(int8_t index) const;

    /** Obtain column index for field.
     *
     * @param index The field index.
     */
    int32_t column_index(int8_t index) const {
        return ecs_field_column(iter_, index);
    }

    /** Obtain term that triggered an observer
     */
    int8_t term_index() const {
        return iter_->term_index;
    }

    /** Convert current iterator result to string.
     */
    flecs::string str() const {
        char *s = ecs_iter_str(iter_);
        return flecs::string(s);
    }

    /** Get readonly access to field data.
     * If the specified field index does not match with the provided type, the
     * function will assert.
     * 
     * This function should not be used in each() callbacks, unless it is to
     * access a shared field. For access to non-shared fields in each(), use
     * field_at.
     *
     * @tparam T Type of the field.
     * @param index The field index.
     * @return The field data.
     */
    template <typename T, typename A = actual_type_t<T>,
        typename std::enable_if<std::is_const<T>::value, void>::type* = nullptr>
    flecs::field<A> field(int8_t index) const;

    /** Get read/write access to field data.
     * If the matched id for the specified field does not match with the provided
     * type or if the field is readonly, the function will assert.
     * 
     * This function should not be used in each() callbacks, unless it is to
     * access a shared field. For access to non-shared fields in each(), use
     * field_at.
     *
     * @tparam T Type of the field.
     * @param index The field index.
     * @return The field data.
     */
    template <typename T, typename A = actual_type_t<T>,
        typename std::enable_if<
            std::is_const<T>::value == false, void>::type* = nullptr>
    flecs::field<A> field(int8_t index) const;

    /** Get unchecked access to field data.
     * Unchecked access is required when a system does not know the type of a
     * field at compile time.
     * 
     * This function should not be used in each() callbacks, unless it is to
     * access a shared field. For access to non-shared fields in each(), use
     * field_at.
     *
     * @param index The field index.
     */
    flecs::untyped_field field(int8_t index) const {
        ecs_assert(!(iter_->flags & EcsIterCppEach) || 
               ecs_field_src(iter_, index) != 0, ECS_INVALID_OPERATION,
            "cannot .field from .each, use .field_at(%d, row) instead", index);
        return get_unchecked_field(index);
    }

    /** Get pointer to field at row. 
     * This function may be used to access shared fields when row is set to 0.
     */
    void* field_at(int8_t index, size_t row) const {
        if (iter_->row_fields & (1llu << index)) {
            return get_unchecked_field_at(index, row)[0];
        } else {
            return get_unchecked_field(index)[row];
        }
    }

    /** Get reference to field at row. 
     * This function may be used to access shared fields when row is set to 0.
     */
    template <typename T, typename A = actual_type_t<T>,
        typename std::enable_if<std::is_const<T>::value, void>::type* = nullptr>
    const A& field_at(int8_t index, size_t row) const {
        if (iter_->row_fields & (1llu << index)) {
            return get_field_at<A>(index, row)[0];
        } else {
            return get_field<A>(index)[row];
        }
    }

    /** Get reference to field at row. 
     * This function may be used to access shared fields when row is set to 0.
     */
    template <typename T, typename A = actual_type_t<T>,
        typename std::enable_if<
            std::is_const<T>::value == false, void>::type* = nullptr>
    A& field_at(int8_t index, size_t row) const {
        ecs_assert(!ecs_field_is_readonly(iter_, index),
            ECS_ACCESS_VIOLATION, NULL);
        if (iter_->row_fields & (1llu << index)) {
            return get_field_at<A>(index, row)[0];
        } else {
            return get_field<A>(index)[row];
        }
    }

    /** Get readonly access to entity ids.
     *
     * @return The entity ids.
     */
    flecs::field<const flecs::entity_t> entities() const {
        return flecs::field<const flecs::entity_t>(
            iter_->entities, static_cast<size_t>(iter_->count), false);
    }

    /** Check if the current table has changed since the last iteration.
     * Can only be used when iterating queries and/or systems. */
    bool changed() {
        return ecs_iter_changed(iter_);
    }

    /** Skip current table.
     * This indicates to the query that the data in the current table is not
     * modified. By default, iterating a table with a query will mark the
     * iterated components as dirty if they are annotated with InOut or Out.
     *
     * When this operation is invoked, the components of the current table will
     * not be marked dirty. */
    void skip() {
        ecs_iter_skip(iter_);
    }

    /* Return group id for current table (grouped queries only) */
    uint64_t group_id() const {
        return iter_->group_id;
    }

    /** Get value of variable by id.
     * Get value of a query variable for current result.
     */
    flecs::entity get_var(int var_id) const;

    /** Get value of variable by name.
     * Get value of a query variable for current result.
     */
    flecs::entity get_var(const char *name) const;

    /** Progress iterator.
     * This operation should only be called from a context where the iterator is
     * not being progressed automatically. An example of a valid context is
     * inside of a run() callback. An example of an invalid context is inside of
     * an each() callback.
     */
    bool next() {
        if (iter_->flags & EcsIterIsValid && iter_->table) {
            ECS_TABLE_UNLOCK(iter_->world, iter_->table);
        }
        bool result = iter_->next(iter_);
        iter_->flags |= EcsIterIsValid;
        if (result && iter_->table) {
            ECS_TABLE_LOCK(iter_->world, iter_->table);
        }
        return result;
    }

    /** Forward to each.
     * If a system has an each callback registered, this operation will forward
     * the current iterator to the each callback.
     */
    void each() {
        iter_->callback(iter_);
    }

    /** Iterate targets for pair field.
     * 
     * @param index The field index.
     * @param func Callback invoked for each target
     */
    template <typename Func>
    void targets(int8_t index, const Func& func);

    /** Free iterator resources.
     * This operation only needs to be called when the iterator is not iterated
     * until completion (e.g. the last call to next() did not return false).
     * 
     * Failing to call this operation on an unfinished iterator will throw a
     * fatal LEAK_DETECTED error.
     * 
     * @see ecs_iter_fini()
     */
    void fini() {
        if (iter_->flags & EcsIterIsValid && iter_->table) {
            ECS_TABLE_UNLOCK(iter_->world, iter_->table);
        }
        ecs_iter_fini(iter_);
    }

private:
    /* Get field, check if correct type is used */
    template <typename T, typename A = actual_type_t<T>>
    flecs::field<T> get_field(int8_t index) const {

#ifndef FLECS_NDEBUG
        ecs_entity_t term_id = ecs_field_id(iter_, index);
        ecs_assert(ECS_HAS_ID_FLAG(term_id, PAIR) ||
            term_id == _::type<T>::id(iter_->world),
            ECS_COLUMN_TYPE_MISMATCH, NULL);
#endif

        size_t count;
        bool is_shared = !ecs_field_is_self(iter_, index);

        /* If a shared column is retrieved with 'column', there will only be a
         * single value. Ensure that the application does not accidentally read
         * out of bounds. */
        if (is_shared) {
            count = 1;
        } else {
            /* If column is owned, there will be as many values as there are
             * entities. */
            count = static_cast<size_t>(iter_->count);
        }

        return flecs::field<A>(
            static_cast<T*>(ecs_field_w_size(iter_, sizeof(A), index)),
            count, is_shared);
    }

    /* Get field, check if correct type is used */
    template <typename T, typename A = actual_type_t<T>>
    flecs::field<T> get_field_at(int8_t index, int32_t row) const {

#ifndef FLECS_NDEBUG
        ecs_entity_t term_id = ecs_field_id(iter_, index);
        ecs_assert(ECS_HAS_ID_FLAG(term_id, PAIR) ||
            term_id == _::type<T>::id(iter_->world),
            ECS_COLUMN_TYPE_MISMATCH, NULL);
#endif

        return flecs::field<A>(
            static_cast<T*>(ecs_field_at_w_size(iter_, sizeof(A), index, row)),
                1, false);
    }

    flecs::untyped_field get_unchecked_field(int8_t index) const {
        size_t count;
        size_t size = ecs_field_size(iter_, index);
        bool is_shared = !ecs_field_is_self(iter_, index);

        /* If a shared column is retrieved with 'column', there will only be a
         * single value. Ensure that the application does not accidentally read
         * out of bounds. */
        if (is_shared) {
            count = 1;
        } else {
            /* If column is owned, there will be as many values as there are
             * entities. */
            count = static_cast<size_t>(iter_->count);
        }

        return flecs::untyped_field(
            ecs_field_w_size(iter_, 0, index), size, count, is_shared);
    }

    flecs::untyped_field get_unchecked_field_at(int8_t index, size_t row) const {
        size_t size = ecs_field_size(iter_, index);
        return flecs::untyped_field(
            ecs_field_at_w_size(iter_, 0, index, static_cast<int32_t>(row)), 
                size, 1, false);
    }

    flecs::iter_t *iter_;
};

} // namespace flecs

/** @} */

/**
 * @file addons/cpp/entity.hpp
 * @brief Entity class.
 *
 * This class provides read/write access to entities.
 */

#pragma once

/**
 * @file addons/cpp/entity_view.hpp
 * @brief Entity class with only readonly operations.
 * 
 * This class provides readonly access to entities. Using this class to store 
 * entities in components ensures valid handles, as this class will always store
 * the actual world vs. a stage. The constructors of this class will never 
 * create a new entity.
 *
 * To obtain a mutable handle to the entity, use the "mut" function.
 */

#pragma once

/**
 * @ingroup cpp_entities
 * @{
 */

namespace flecs
{

/** Entity view.
 * Class with read operations for entities. Base for flecs::entity.
 * 
 * @ingroup cpp_entities
 */
struct entity_view : public id {

    entity_view() : flecs::id() { }

    /** Wrap an existing entity id.
     *
     * @param world The world in which the entity is created.
     * @param id The entity id.
     */
    explicit entity_view(flecs::world_t *world, flecs::id_t id)
        : flecs::id(world 
            ? const_cast<flecs::world_t*>(ecs_get_world(world))
            : nullptr
        , id ) { }

    /** Implicit conversion from flecs::entity_t to flecs::entity_view. */
    entity_view(entity_t id) 
        : flecs::id( nullptr, id ) { }

    /** Get entity id.
     * @return The integer entity id.
     */
    entity_t id() const {
        return id_;
    }

    /** Check if entity is valid.
     *
     * @return True if the entity is alive, false otherwise.
     */
    bool is_valid() const {
        return world_ && ecs_is_valid(world_, id_);
    }
  
    explicit operator bool() const {
        return is_valid();
    }

    /** Check if entity is alive.
     *
     * @return True if the entity is alive, false otherwise.
     */
    bool is_alive() const {
        return world_ && ecs_is_alive(world_, id_);
    }

    /** Return the entity name.
     *
     * @return The entity name.
     */
    flecs::string_view name() const {
        return flecs::string_view(ecs_get_name(world_, id_));
    }

    /** Return the entity symbol.
     *
     * @return The entity symbol.
     */
    flecs::string_view symbol() const {
        return flecs::string_view(ecs_get_symbol(world_, id_));
    }

    /** Return the entity path.
     *
     * @return The hierarchical entity path.
     */
    flecs::string path(const char *sep = "::", const char *init_sep = "::") const {
        return path_from(0, sep, init_sep);
    }   

    /** Return the entity path relative to a parent.
     *
     * @return The relative hierarchical entity path.
     */
    flecs::string path_from(flecs::entity_t parent, const char *sep = "::", const char *init_sep = "::") const {
        char *path = ecs_get_path_w_sep(world_, parent, id_, sep, init_sep);
        return flecs::string(path);
    }

    /** Return the entity path relative to a parent.
     *
     * @return The relative hierarchical entity path.
     */
    template <typename Parent>
    flecs::string path_from(const char *sep = "::", const char *init_sep = "::") const {
        return path_from(_::type<Parent>::id(world_), sep, init_sep);
    }

    bool enabled() const {
        return !ecs_has_id(world_, id_, flecs::Disabled);
    }

    /** Get the entity's type.
     *
     * @return The entity's type.
     */
    flecs::type type() const;

    /** Get the entity's table.
     *
     * @return Returns the entity's table.
     */
    flecs::table table() const;

    /** Get table range for the entity.
     * Returns a range with the entity's row as offset and count set to 1. If
     * the entity is not stored in a table, the function returns a range with
     * count 0.
     *
     * @return Returns the entity's table range.
     */
    flecs::table_range range() const;

    /** Iterate (component) ids of an entity.
     * The function parameter must match the following signature:
     *
     * @code
     * void(*)(flecs::id id)
     * @endcode
     *
     * @param func The function invoked for each id.
     */
    template <typename Func>
    void each(const Func& func) const;

    /** Iterate matching pair ids of an entity.
     * The function parameter must match the following signature:
     *
     * @code
     * void(*)(flecs::id id)
     * @endcode
     *
     * @param func The function invoked for each id.
     */
    template <typename Func>
    void each(flecs::id_t first, flecs::id_t second, const Func& func) const;

    /** Iterate targets for a given relationship.
     * The function parameter must match the following signature:
     *
     * @code
     * void(*)(flecs::entity target)
     * @endcode
     *
     * @param rel The relationship for which to iterate the targets.
     * @param func The function invoked for each target.
     */
    template <typename Func>
    void each(const flecs::entity_view& rel, const Func& func) const;

    /** Iterate targets for a given relationship.
     * The function parameter must match the following signature:
     *
     * @code
     * void(*)(flecs::entity target)
     * @endcode
     *
     * @tparam First The relationship for which to iterate the targets.
     * @param func The function invoked for each target.     
     */
    template <typename First, typename Func>
    void each(const Func& func) const { 
        return each(_::type<First>::id(world_), func);
    }

    /** Iterate children for entity.
     * The function parameter must match the following signature:
     *
     * @code
     * void(*)(flecs::entity target)
     * @endcode
     *
     * @param rel The relationship to follow.
     * @param func The function invoked for each child.     
     */
    template <typename Func>
    void children(flecs::entity_t rel, Func&& func) const {
        /* When the entity is a wildcard, this would attempt to query for all
         * entities with (ChildOf, *) or (ChildOf, _) instead of querying for
         * the children of the wildcard entity. */
        if (id_ == flecs::Wildcard || id_ == flecs::Any) {
            /* This is correct, wildcard entities don't have children */
            return;
        }

        flecs::world world(world_);

        ecs_iter_t it = ecs_each_id(world_, ecs_pair(rel, id_));
        while (ecs_each_next(&it)) {
            _::each_delegate<Func>(FLECS_MOV(func)).invoke(&it);
        }
    }

    /** Iterate children for entity.
     * The function parameter must match the following signature:
     *
     * @code
     * void(*)(flecs::entity target)
     * @endcode
     *
     * @tparam Rel The relationship to follow.
     * @param func The function invoked for each child.     
     */
    template <typename Rel, typename Func>
    void children(Func&& func) const {
        children(_::type<Rel>::id(world_), FLECS_MOV(func));
    }

    /** Iterate children for entity.
     * The function parameter must match the following signature:
     *
     * @code
     * void(*)(flecs::entity target)
     * @endcode
     * 
     * This operation follows the ChildOf relationship.
     *
     * @param func The function invoked for each child.     
     */
    template <typename Func>
    void children(Func&& func) const {
        children(flecs::ChildOf, FLECS_MOV(func));
    }

    /** Get component value.
     * 
     * @tparam T The component to get.
     * @return Pointer to the component value, nullptr if the entity does not
     *         have the component.
     */
    template <typename T, if_t< is_actual<T>::value > = 0>
    const T* get() const {
        auto comp_id = _::type<T>::id(world_);
        ecs_assert(_::type<T>::size() != 0, ECS_INVALID_PARAMETER,
            "operation invalid for empty type");
        return static_cast<const T*>(ecs_get_id(world_, id_, comp_id));
    }

    /** Get component value.
     * Overload for when T is not the same as the actual type, which happens
     * when using pair types.
     * 
     * @tparam T The component to get.
     * @return Pointer to the component value, nullptr if the entity does not
     *         have the component.
     */
    template <typename T, typename A = actual_type_t<T>, 
        if_t< flecs::is_pair<T>::value > = 0>
    const A* get() const {
        auto comp_id = _::type<T>::id(world_);
        ecs_assert(_::type<A>::size() != 0, ECS_INVALID_PARAMETER,
            "operation invalid for empty type");
        return static_cast<const A*>(ecs_get_id(world_, id_, comp_id));
    }

    /** Get a pair.
     * This operation gets the value for a pair from the entity.
     *
     * @tparam First The first element of the pair.
     * @tparam Second the second element of a pair.
     */
    template <typename First, typename Second, typename P = pair<First, Second>, 
        typename A = actual_type_t<P>, if_not_t< flecs::is_pair<First>::value > = 0>
    const A* get() const {
        return this->get<P>();
    }

    /** Get a pair.
     * This operation gets the value for a pair from the entity. 
     *
     * @tparam First The first element of the pair.
     * @param second The second element of the pair.
     */
    template<typename First, typename Second, if_not_t< is_enum<Second>::value> = 0>
    const First* get(Second second) const {
        auto first = _::type<First>::id(world_);
        ecs_assert(_::type<First>::size() != 0, ECS_INVALID_PARAMETER,
            "operation invalid for empty type");
        return static_cast<const First*>(
            ecs_get_id(world_, id_, ecs_pair(first, second)));
    }

    /** Get a pair.
     * This operation gets the value for a pair from the entity. 
     *
     * @tparam First The first element of the pair.
     * @param constant the enum constant.
     */
    template<typename First, typename Second, if_t<is_enum<Second>::value> = 0>
    const First* get(Second constant) const {
        const auto& et = enum_type<Second>(this->world_);
        flecs::entity_t target = et.entity(constant);
        return get<First>(target);
    }

    /** Get component value (untyped).
     * 
     * @param comp The component to get.
     * @return Pointer to the component value, nullptr if the entity does not
     *         have the component.
     */
    const void* get(flecs::id_t comp) const {
        return ecs_get_id(world_, id_, comp);
    }

    /** Get a pair (untyped).
     * This operation gets the value for a pair from the entity. If neither the
     * first nor the second part of the pair are components, the operation 
     * will fail.
     *
     * @param first The first element of the pair.
     * @param second The second element of the pair.
     */
    const void* get(flecs::entity_t first, flecs::entity_t second) const {
        return ecs_get_id(world_, id_, ecs_pair(first, second));
    }

    /** Get 1..N components.
     * This operation accepts a callback with as arguments the components to
     * retrieve. The callback will only be invoked when the entity has all
     * the components.
     *
     * This operation is faster than individually calling get for each component
     * as it only obtains entity metadata once.
     * 
     * While the callback is invoked the table in which the components are
     * stored is locked, which prevents mutations that could cause invalidation
     * of the component references. Note that this is not an actual lock: 
     * invalid access causes a runtime panic and so it is still up to the 
     * application to ensure access is protected.
     * 
     * The component arguments must be references and can be either const or
     * non-const. When all arguments are const, the function will read-lock the
     * table (see ecs_read_begin). If one or more arguments are non-const the
     * function will write-lock the table (see ecs_write_begin).
     * 
     * Example:
     *
     * @code
     * e.get([](Position& p, Velocity& v) { // write lock
     *   p.x += v.x;
     * });
     * 
     * e.get([](const Position& p) {        // read lock
     *   std::cout << p.x << std::endl;
     * });
     * @endcode
     *
     * @param func The callback to invoke.
     * @return True if the entity has all components, false if not.
     */
    template <typename Func, if_t< is_callable<Func>::value > = 0>
    bool get(const Func& func) const;

    /** Get enum constant.
     * 
     * @tparam T The enum type for which to get the constant
     * @return Constant entity if found, 0 entity if not.
     */
    template <typename T, if_t< is_enum<T>::value > = 0>
    const T* get() const;

    /** Get the second part for a pair.
     * This operation gets the value for a pair from the entity. The first
     * part of the pair should not be a component.
     *
     * @tparam Second the second element of a pair.
     * @param first The first part of the pair.
     */
    template<typename Second>
    const Second* get_second(flecs::entity_t first) const {
        auto second = _::type<Second>::id(world_);
        ecs_assert( ecs_get_type_info(world_, ecs_pair(first, second)) != NULL,
            ECS_INVALID_PARAMETER, "pair is not a component");
        ecs_assert( ecs_get_type_info(world_, ecs_pair(first, second))->component == second,
            ECS_INVALID_PARAMETER, "type of pair is not Second");
        ecs_assert(_::type<Second>::size() != 0, ECS_INVALID_PARAMETER,
            "operation invalid for empty type");
        return static_cast<const Second*>(
            ecs_get_id(world_, id_, ecs_pair(first, second)));
    }

    /** Get the second part for a pair.
     * This operation gets the value for a pair from the entity. The first
     * part of the pair should not be a component.
     *
     * @tparam First The first element of the pair.
     * @tparam Second the second element of a pair.
     */
    template<typename First, typename Second>
    const Second* get_second() const {
        return get<pair_object<First, Second>>();
    }

    /** Get mutable component value.
     * 
     * @tparam T The component to get.
     * @return Pointer to the component value, nullptr if the entity does not
     *         have the component.
     */
    template <typename T, if_t< is_actual<T>::value > = 0>
    T* get_mut() const {
        auto comp_id = _::type<T>::id(world_);
        ecs_assert(_::type<T>::size() != 0, ECS_INVALID_PARAMETER,
            "operation invalid for empty type");
        return static_cast<T*>(ecs_get_mut_id(world_, id_, comp_id));
    }

    /** Get mutable component value.
     * Overload for when T is not the same as the actual type, which happens
     * when using pair types.
     * 
     * @tparam T The component to get.
     * @return Pointer to the component value, nullptr if the entity does not
     *         have the component.
     */
    template <typename T, typename A = actual_type_t<T>, 
        if_t< flecs::is_pair<T>::value > = 0>
    A* get_mut() const {
        auto comp_id = _::type<T>::id(world_);
        ecs_assert(_::type<A>::size() != 0, ECS_INVALID_PARAMETER,
            "operation invalid for empty type");
        return static_cast<A*>(ecs_get_mut_id(world_, id_, comp_id));
    }

    /** Get a mutable pair.
     * This operation gets the value for a pair from the entity.
     *
     * @tparam First The first element of the pair.
     * @tparam Second the second element of a pair.
     */
    template <typename First, typename Second, typename P = pair<First, Second>, 
        typename A = actual_type_t<P>, if_not_t< flecs::is_pair<First>::value > = 0>
    A* get_mut() const {
        return this->get_mut<P>();
    }

    /** Get a mutable pair.
     * This operation gets the value for a pair from the entity. 
     *
     * @tparam First The first element of the pair.
     * @param second The second element of the pair.
     */
    template<typename First, typename Second, if_not_t< is_enum<Second>::value> = 0>
    First* get_mut(Second second) const {
        auto first = _::type<First>::id(world_);
        ecs_assert(_::type<First>::size() != 0, ECS_INVALID_PARAMETER, 
            "operation invalid for empty type");
        return static_cast<First*>(
            ecs_get_mut_id(world_, id_, ecs_pair(first, second)));
    }

    /** Get a mutable pair.
     * This operation gets the value for a pair from the entity. 
     *
     * @tparam First The first element of the pair.
     * @param constant the enum constant.
     */
    template<typename First, typename Second, if_t<is_enum<Second>::value> = 0>
    First* get_mut(Second constant) const {
        const auto& et = enum_type<Second>(this->world_);
        flecs::entity_t target = et.entity(constant);
        return get_mut<First>(target);
    }

    /** Get mutable component value (untyped).
     * 
     * @param comp The component to get.
     * @return Pointer to the component value, nullptr if the entity does not
     *         have the component.
     */
    void* get_mut(flecs::id_t comp) const {
        return ecs_get_mut_id(world_, id_, comp);
    }

    /** Get a mutable pair (untyped).
     * This operation gets the value for a pair from the entity. If neither the
     * first nor the second part of the pair are components, the operation 
     * will fail.
     *
     * @param first The first element of the pair.
     * @param second The second element of the pair.
     */
    void* get_mut(flecs::entity_t first, flecs::entity_t second) const {
        return ecs_get_mut_id(world_, id_, ecs_pair(first, second));
    }

    /** Get the second part for a pair.
     * This operation gets the value for a pair from the entity. The first
     * part of the pair should not be a component.
     *
     * @tparam Second the second element of a pair.
     * @param first The first part of the pair.
     */
    template<typename Second>
    Second* get_mut_second(flecs::entity_t first) const {
        auto second = _::type<Second>::id(world_);
        ecs_assert( ecs_get_type_info(world_, ecs_pair(first, second)) != NULL,
            ECS_INVALID_PARAMETER, "pair is not a component");
        ecs_assert( ecs_get_type_info(world_, ecs_pair(first, second))->component == second,
            ECS_INVALID_PARAMETER, "type of pair is not Second");
        ecs_assert(_::type<Second>::size() != 0, ECS_INVALID_PARAMETER, 
            "operation invalid for empty type");
        return static_cast<Second*>(
            ecs_get_mut_id(world_, id_, ecs_pair(first, second)));
    }

    /** Get the second part for a pair.
     * This operation gets the value for a pair from the entity. The first
     * part of the pair should not be a component.
     *
     * @tparam First The first element of the pair.
     * @tparam Second the second element of a pair.
     */
    template<typename First, typename Second>
    Second* get_mut_second() const {
        return get_mut<pair_object<First, Second>>();
    }

    /** Get target for a given pair.
     * This operation returns the target for a given pair. The optional
     * index can be used to iterate through targets, in case the entity has
     * multiple instances for the same relationship.
     *
     * @tparam First The first element of the pair.
     * @param index The index (0 for the first instance of the relationship).
     */
    template<typename First>
    flecs::entity target(int32_t index = 0) const;

    /** Get target for a given pair.
     * This operation returns the target for a given pair. The optional
     * index can be used to iterate through targets, in case the entity has
     * multiple instances for the same relationship.
     *
     * @param first The first element of the pair for which to retrieve the target.
     * @param index The index (0 for the first instance of the relationship).
     */
    flecs::entity target(flecs::entity_t first, int32_t index = 0) const;

    /** Get the target of a pair for a given relationship id.
     * This operation returns the first entity that has the provided id by following
     * the specified relationship. If the entity itself has the id then entity will
     * be returned. If the id cannot be found on the entity or by following the
     * relationship, the operation will return 0.
     * 
     * This operation can be used to lookup, for example, which prefab is providing
     * a component by specifying the IsA pair:
     * 
     * @code
     * // Is Position provided by the entity or one of its base entities?
     * ecs_get_target_for_id(world, entity, EcsIsA, ecs_id(Position))
     * @endcode
     * 
     * @param relationship The relationship to follow.
     * @param id The id to lookup.
     * @return The entity for which the target has been found.
     */
    flecs::entity target_for(flecs::entity_t relationship, flecs::id_t id) const;

    template <typename T>
    flecs::entity target_for(flecs::entity_t relationship) const;

    template <typename First, typename Second>
    flecs::entity target_for(flecs::entity_t relationship) const;

    /** Get depth for given relationship.
     *
     * @param rel The relationship.
     * @return The depth.
     */
    int32_t depth(flecs::entity_t rel) const {
        return ecs_get_depth(world_, id_, rel);
    }

    /** Get depth for given relationship.
     *
     * @tparam Rel The relationship.
     * @return The depth.
     */
    template<typename Rel>
    int32_t depth() const {
        return this->depth(_::type<Rel>::id(world_));
    }

    /** Get parent of entity.
     * Short for target(flecs::ChildOf).
     * 
     * @return The parent of the entity.
     */
    flecs::entity parent() const;
    
    /** Lookup an entity by name.
     * Lookup an entity in the scope of this entity. The provided path may
     * contain double colons as scope separators, for example: "Foo::Bar".
     *
     * @param path The name of the entity to lookup.
     * @param search_path When false, only the entity's scope is searched.
     * @return The found entity, or entity::null if no entity matched.
     */
    flecs::entity lookup(const char *path, bool search_path = false) const;

    /** Check if entity has the provided entity.
     *
     * @param e The entity to check.
     * @return True if the entity has the provided entity, false otherwise.
     */
    bool has(flecs::id_t e) const {
        return ecs_has_id(world_, id_, e);
    }     

    /** Check if entity has the provided component.
     *
     * @tparam T The component to check.
     * @return True if the entity has the provided component, false otherwise.
     */
    template <typename T>
    bool has() const {
        flecs::id_t cid = _::type<T>::id(world_);
        bool result = ecs_has_id(world_, id_, cid);
        if (result) {
            return result;
        }

        if (is_enum<T>::value) {
            return ecs_has_pair(world_, id_, cid, flecs::Wildcard);
        }

        return false;
    }

    /** Check if entity has the provided enum constant.
     *
     * @tparam E The enum type (can be deduced).
     * @param value The enum constant to check. 
     * @return True if the entity has the provided constant, false otherwise.
     */
    template <typename E, if_t< is_enum<E>::value > = 0>
    bool has(E value) const {
        auto r = _::type<E>::id(world_);
        auto o = enum_type<E>(world_).entity(value);
        ecs_assert(o, ECS_INVALID_PARAMETER,
            "Constant was not found in Enum reflection data."
            " Did you mean to use has<E>() instead of has(E)?");
        return ecs_has_pair(world_, id_, r, o);
    }

    /** Check if entity has the provided pair.
     *
     * @tparam First The first element of the pair.
     * @tparam Second The second element of the pair.
     * @return True if the entity has the provided component, false otherwise.
     */
    template <typename First, typename Second>
    bool has() const {
        return this->has<First>(_::type<Second>::id(world_));
    }

    /** Check if entity has the provided pair.
     *
     * @tparam First The first element of the pair.
     * @param second The second element of the pair.
     * @return True if the entity has the provided component, false otherwise.
     */
    template<typename First, typename Second, if_not_t< is_enum<Second>::value > = 0>
    bool has(Second second) const {
        auto comp_id = _::type<First>::id(world_);
        return ecs_has_id(world_, id_, ecs_pair(comp_id, second));
    }

    /** Check if entity has the provided pair.
     *
     * @tparam Second The second element of the pair.
     * @param first The first element of the pair.
     * @return True if the entity has the provided component, false otherwise.
     */
    template <typename Second>
    bool has_second(flecs::entity_t first) const {
        return this->has(first, _::type<Second>::id(world_));
    }

    /** Check if entity has the provided pair.
     *
     * @tparam First The first element of the pair.
     * @param value The enum constant.
     * @return True if the entity has the provided component, false otherwise.
     */
    template<typename First, typename E, if_t< is_enum<E>::value > = 0>
    bool has(E value) const {
        const auto& et = enum_type<E>(this->world_);
        flecs::entity_t second = et.entity(value);
        return has<First>(second);
    }

    /** Check if entity has the provided pair.
     *
     * @param first The first element of the pair.
     * @param second The second element of the pair.
     * @return True if the entity has the provided component, false otherwise.
     */
    bool has(flecs::id_t first, flecs::id_t second) const {
        return ecs_has_id(world_, id_, ecs_pair(first, second));
    }

    /** Check if entity owns the provided entity.
     * An entity is owned if it is not shared from a base entity.
     *
     * @param e The entity to check.
     * @return True if the entity owns the provided entity, false otherwise.
     */
    bool owns(flecs::id_t e) const {
        return ecs_owns_id(world_, id_, e);
    }

    /** Check if entity owns the provided pair.
     *
     * @tparam First The first element of the pair.
     * @param second The second element of the pair.
     * @return True if the entity owns the provided component, false otherwise.
     */
    template <typename First>
    bool owns(flecs::id_t second) const {
        auto comp_id = _::type<First>::id(world_);
        return owns(ecs_pair(comp_id, second));
    }

    /** Check if entity owns the provided pair.
     *
     * @param first The first element of the pair.
     * @param second The second element of the pair.
     * @return True if the entity owns the provided component, false otherwise.
     */
    bool owns(flecs::id_t first, flecs::id_t second) const {
        return owns(ecs_pair(first, second));
    }

    /** Check if entity owns the provided component.
     * An component is owned if it is not shared from a base entity.
     *
     * @tparam T The component to check.
     * @return True if the entity owns the provided component, false otherwise.
     */
    template <typename T>
    bool owns() const {
        return owns(_::type<T>::id(world_));
    }

    /** Check if entity owns the provided pair.
     * An pair is owned if it is not shared from a base entity.
     *
     * @tparam First The first element of the pair.
     * @tparam Second The second element of the pair.
     * @return True if the entity owns the provided pair, false otherwise.
     */
    template <typename First, typename Second>
    bool owns() const {
        return owns(
            _::type<First>::id(world_),
            _::type<Second>::id(world_));
    }

    /** Test if id is enabled.
     *
     * @param id The id to test.
     * @return True if enabled, false if not.
     */
    bool enabled(flecs::id_t id) const {
        return ecs_is_enabled_id(world_, id_, id);
    }

    /** Test if component is enabled.
     *
     * @tparam T The component to test.
     * @return True if enabled, false if not.
     */
    template<typename T>
    bool enabled() const {
        return this->enabled(_::type<T>::id(world_));
    }

    /** Test if pair is enabled.
     *
     * @param first The first element of the pair.
     * @param second The second element of the pair.
     * @return True if enabled, false if not.
     */
    bool enabled(flecs::id_t first, flecs::id_t second) const {
        return this->enabled(ecs_pair(first, second));
    }

    /** Test if pair is enabled.
     *
     * @tparam First The first element of the pair.
     * @param second The second element of the pair.
     * @return True if enabled, false if not.
     */
    template <typename First>
    bool enabled(flecs::id_t second) const {
        return this->enabled(_::type<First>::id(world_), second);
    }

    /** Test if pair is enabled.
     *
     * @tparam First The first element of the pair.
     * @tparam Second The second element of the pair.
     * @return True if enabled, false if not.
     */
    template <typename First, typename Second>
    bool enabled() const {
        return this->enabled<First>(_::type<Second>::id(world_));
    }

    flecs::entity clone(bool clone_value = true, flecs::entity_t dst_id = 0) const;

    /** Return mutable entity handle for current stage 
     * When an entity handle created from the world is used while the world is
     * in staged mode, it will only allow for readonly operations since 
     * structural changes are not allowed on the world while in staged mode.
     * 
     * To do mutations on the entity, this operation provides a handle to the
     * entity that uses the stage instead of the actual world.
     *
     * Note that staged entity handles should never be stored persistently, in
     * components or elsewhere. An entity handle should always point to the
     * main world.
     *
     * Also note that this operation is not necessary when doing mutations on an
     * entity outside of a system. It is allowed to do entity operations 
     * directly on the world, as long as the world is not in staged mode.
     *
     * @param stage The current stage.
     * @return An entity handle that allows for mutations in the current stage.
     */
    flecs::entity mut(const flecs::world& stage) const;

    /** Same as mut(world), but for iterator.
     * This operation allows for the construction of a mutable entity handle
     * from an iterator.
     *
     * @param it An iterator that contains a reference to the world or stage.
     * @return An entity handle that allows for mutations in the current stage.
     */
    flecs::entity mut(const flecs::iter& it) const;

    /** Same as mut(world), but for entity.
     * This operation allows for the construction of a mutable entity handle
     * from another entity. This is useful in each() functions, which only 
     * provide a handle to the entity being iterated over.
     *
     * @param e Another mutable entity.
     * @return An entity handle that allows for mutations in the current stage.
     */
    flecs::entity mut(const flecs::entity_view& e) const;

#   ifdef FLECS_JSON
/**
 * @file addons/cpp/mixins/json/entity_view.inl
 * @brief JSON entity mixin.
 */

/** Serialize entity to JSON.
 * 
 * @memberof flecs::entity_view
 * @ingroup cpp_addons_json
 */
flecs::string to_json(const flecs::entity_to_json_desc_t *desc = nullptr) const {
    char *json = ecs_entity_to_json(world_, id_, desc);
    return flecs::string(json);
}

#   endif
#   ifdef FLECS_DOC
/**
 * @file addons/cpp/mixins/doc/entity_view.inl
 * @brief Doc entity view mixin.
 */

/** Get human readable name.
 *
 * @see ecs_doc_get_name()
 * @see flecs::doc::get_name()
 * @see flecs::entity_builder::set_doc_name()
 *
 * @memberof flecs::entity_view
 * @ingroup cpp_addons_doc
 */
const char* doc_name() const {
    return ecs_doc_get_name(world_, id_);
}

/** Get brief description.
 *
 * @see ecs_doc_get_brief()
 * @see flecs::doc::get_brief()
 * @see flecs::entity_builder::set_doc_brief()
 *
 * @memberof flecs::entity_view
 * @ingroup cpp_addons_doc
 */
const char* doc_brief() const {
    return ecs_doc_get_brief(world_, id_);
}

/** Get detailed description.
 *
 * @see ecs_doc_get_detail()
 * @see flecs::doc::get_detail()
 * @see flecs::entity_builder::set_doc_detail()
 *
 * @memberof flecs::entity_view
 * @ingroup cpp_addons_doc
 */
const char* doc_detail() const {
    return ecs_doc_get_detail(world_, id_);
}

/** Get link to external documentation.
 *
 * @see ecs_doc_get_link()
 * @see flecs::doc::get_link()
 * @see flecs::entity_builder::set_doc_link()
 *
 * @memberof flecs::entity_view
 * @ingroup cpp_addons_doc
 */
const char* doc_link() const {
    return ecs_doc_get_link(world_, id_);
}

/** Get color.
 *
 * @see ecs_doc_get_color()
 * @see flecs::doc::get_color()
 * @see flecs::entity_builder::set_doc_color()
 *
 * @memberof flecs::entity_view
 * @ingroup cpp_addons_doc
 */
const char* doc_color() const {
    return ecs_doc_get_color(world_, id_);
}

/** Get UUID.
 *
 * @see ecs_doc_get_uuid()
 * @see flecs::doc::get_uuid()
 * @see flecs::entity_builder::set_doc_uuid()
 *
 * @memberof flecs::entity_view
 * @ingroup cpp_addons_doc
 */
const char* doc_uuid() const {
    return ecs_doc_get_uuid(world_, id_);
}

#   endif
#   ifdef FLECS_ALERTS
/**
 * @file addons/cpp/mixins/alerts/entity_view.inl
 * @brief Alerts entity mixin.
 */

/** Return number of alerts for entity.
 * 
 * @memberof flecs::entity_view
 * @ingroup cpp_addons_alerts
 */
int32_t alert_count(flecs::entity_t alert = 0) const {
    return ecs_get_alert_count(world_, id_, alert);
}

#   endif

/**
 * @file addons/cpp/mixins/enum/entity_view.inl
 * @brief Enum entity view mixin.
 */

/** Convert entity to enum constant.
 * 
 * @memberof flecs::entity_view
 * @ingroup cpp_entities
 */
template <typename E>
E to_constant() const;


/**
 * @file addons/cpp/mixins/event/entity_view.inl
 * @brief Event entity mixin.
 */

/** Emit event for entity.
 * 
 * @memberof flecs::entity_view
 * 
 * @param evt The event to emit.
 */
void emit(flecs::entity_t evt) const {
    flecs::world(world_)
        .event(evt)
        .entity(id_)
        .emit();
}

/** Emit event for entity.
 * 
 * @memberof flecs::entity_view
 * 
 * @param evt The event to emit.
 */
void emit(flecs::entity evt) const;

/** Emit event for entity.
 * 
 * @memberof flecs::entity_view
 * 
 * @tparam Evt The event to emit.
 */
template <typename Evt, if_t<is_empty<Evt>::value> = 0>
void emit() const {
    this->emit(_::type<Evt>::id(world_));
}

/** Emit event with payload for entity.
 * 
 * @memberof flecs::entity_view
 * 
 * @tparam Evt The event to emit.
 */
template <typename Evt, if_not_t<is_empty<Evt>::value> = 0>
void emit(const Evt& payload) const {
    flecs::world(world_)
        .event(_::type<Evt>::id(world_))
        .entity(id_)
        .ctx(&payload)
        .emit();
}


/** Enqueue event for entity.
 * 
 * @memberof flecs::entity_view
 * 
 * @param evt The event to enqueue.
 */
void enqueue(flecs::entity_t evt) const {
    flecs::world(world_)
        .event(evt)
        .entity(id_)
        .enqueue();
}

/** Enqueue event for entity.
 * 
 * @memberof flecs::entity_view
 * 
 * @param evt The event to enqueue.
 */
void enqueue(flecs::entity evt) const;

/** Enqueue event for entity.
 * 
 * @memberof flecs::entity_view
 * 
 * @tparam Evt The event to enqueue.
 */
template <typename Evt, if_t<is_empty<Evt>::value> = 0>
void enqueue() const {
    this->enqueue(_::type<Evt>::id(world_));
}

/** Enqueue event with payload for entity.
 * 
 * @memberof flecs::entity_view
 * 
 * @tparam Evt The event to enqueue.
 */
template <typename Evt, if_not_t<is_empty<Evt>::value> = 0>
void enqueue(const Evt& payload) const {
    flecs::world(world_)
        .event(_::type<Evt>::id(world_))
        .entity(id_)
        .ctx(&payload)
        .enqueue();
}


private:
    flecs::entity set_stage(world_t *stage);
};

}

/** @} */

/**
 * @file addons/cpp/mixins/entity/builder.hpp
 * @brief Entity builder.
 */

#pragma once

namespace flecs
{

/** Entity builder. 
 * @ingroup cpp_entities
 */
template <typename Self>
struct entity_builder : entity_view {

    using entity_view::entity_view;

    /** Add a component to an entity.
     * To ensure the component is initialized, it should have a constructor.
     * 
     * @tparam T the component type to add.
     */
    template <typename T>
    const Self& add() const  {
        flecs_static_assert(is_flecs_constructible<T>::value,
            "cannot default construct type: add T::T() or use emplace<T>()");
        ecs_add_id(this->world_, this->id_, _::type<T>::id(this->world_));
        return to_base();
    }

     /** Add pair for enum constant.
     * This operation will add a pair to the entity where the first element is
     * the enumeration type, and the second element the enumeration constant.
     * 
     * The operation may be used with regular (C style) enumerations as well as
     * enum classes.
     * 
     * @param value The enumeration value.
     */
    template <typename E, if_t< is_enum<E>::value > = 0>
    const Self& add(E value) const  {
        flecs::entity_t first = _::type<E>::id(this->world_);
        const auto& et = enum_type<E>(this->world_);
        flecs::entity_t second = et.entity(value);

        ecs_assert(second, ECS_INVALID_PARAMETER, "Component was not found in reflection data.");
        return this->add(first, second);
    }

    /** Add an entity to an entity.
     * Add an entity to the entity. This is typically used for tagging.
     *
     * @param component The component to add.
     */
    const Self& add(id_t component) const  {
        ecs_add_id(this->world_, this->id_, component);
        return to_base();
    }

    /** Add a pair.
     * This operation adds a pair to the entity.
     *
     * @param first The first element of the pair.
     * @param second The second element of the pair.
     */
    const Self& add(entity_t first, entity_t second) const  {
        ecs_add_pair(this->world_, this->id_, first, second);
        return to_base();
    }

    /** Add a pair.
     * This operation adds a pair to the entity.
     *
     * @tparam First The first element of the pair
     * @tparam Second The second element of the pair
     */
    template<typename First, typename Second>
    const Self& add() const  {
        return this->add<First>(_::type<Second>::id(this->world_));
    }

    /** Add a pair.
     * This operation adds a pair to the entity.
     *
     * @tparam First The first element of the pair
     * @param second The second element of the pair.
     */
    template<typename First, typename Second, if_not_t< is_enum<Second>::value > = 0>
    const Self& add(Second second) const  {
        flecs_static_assert(is_flecs_constructible<First>::value,
            "cannot default construct type: add T::T() or use emplace<T>()");
        return this->add(_::type<First>::id(this->world_), second);
    }

    /** Add a pair.
     * This operation adds a pair to the entity that consists out of a tag
     * combined with an enum constant.
     *
     * @tparam First The first element of the pair
     * @param constant the enum constant.
     */
    template<typename First, typename Second, if_t< is_enum<Second>::value > = 0>
    const Self& add(Second constant) const  {
        flecs_static_assert(is_flecs_constructible<First>::value,
            "cannot default construct type: add T::T() or use emplace<T>()");
        const auto& et = enum_type<Second>(this->world_);
        return this->add<First>(et.entity(constant));
    }

    /** Add a pair.
     * This operation adds a pair to the entity.
     *
     * @param first The first element of the pair
     * @tparam Second The second element of the pair
     */
    template<typename Second>
    const Self& add_second(flecs::entity_t first) const  {
        return this->add(first, _::type<Second>::id(this->world_));
    }

    /** Conditional add.
     * This operation adds if condition is true, removes if condition is false.
     * 
     * @param cond The condition to evaluate.
     * @param component The component to add.
     */
    const Self& add_if(bool cond, flecs::id_t component) const  {
        if (cond) {
            return this->add(component);
        } else {
            return this->remove(component);
        }
    }

    /** Conditional add.
     * This operation adds if condition is true, removes if condition is false.
     * 
     * @tparam T The component to add.
     * @param cond The condition to evaluate.
     */
    template <typename T>
    const Self& add_if(bool cond) const  {
        if (cond) {
            return this->add<T>();
        } else {
            return this->remove<T>();
        }
    }

    /** Conditional add.
     * This operation adds if condition is true, removes if condition is false.
     * 
     * @param cond The condition to evaluate.
     * @param first The first element of the pair.
     * @param second The second element of the pair.
     */
    const Self& add_if(bool cond, flecs::entity_t first, flecs::entity_t second) const  {
        if (cond) {
            return this->add(first, second);
        } else {
            /* If second is 0 or if relationship is exclusive, use wildcard for
             * second which will remove all instances of the relationship.
             * Replacing 0 with Wildcard will make it possible to use the second
             * as the condition. */
            if (!second || ecs_has_id(this->world_, first, flecs::Exclusive)) {
                second = flecs::Wildcard;
            }
            return this->remove(first, second);
        }
    }

    /** Conditional add.
     * This operation adds if condition is true, removes if condition is false.
     * 
     * @tparam First The first element of the pair
     * @param cond The condition to evaluate.
     * @param second The second element of the pair.
     */
    template <typename First>
    const Self& add_if(bool cond, flecs::entity_t second) const  {
        return this->add_if(cond, _::type<First>::id(this->world_), second);
    }

    /** Conditional add.
     * This operation adds if condition is true, removes if condition is false.
     * 
     * @tparam First The first element of the pair
     * @tparam Second The second element of the pair
     * @param cond The condition to evaluate.
     */
    template <typename First, typename Second>
    const Self& add_if(bool cond) const  {
        return this->add_if<First>(cond, _::type<Second>::id(this->world_));
    }

    /** Conditional add.
     * This operation adds if condition is true, removes if condition is false.
     * 
     * @param cond The condition to evaluate.
     * @param constant The enumeration constant.
     */
    template <typename E, if_t< is_enum<E>::value > = 0>
    const Self& add_if(bool cond, E constant) const  {
        const auto& et = enum_type<E>(this->world_);
        return this->add_if<E>(cond, et.entity(constant));
    }

    /** Shortcut for `add(IsA, entity)`.
     *
     * @param second The second element of the pair.
     */
    const Self& is_a(entity_t second) const  {
        return this->add(flecs::IsA, second);
    }

    /** Shortcut for `add(IsA, entity)`.
     *
     * @tparam T the type associated with the entity.
     */
    template <typename T>
    const Self& is_a() const  {
        return this->add(flecs::IsA, _::type<T>::id(this->world_));
    }

    /** Shortcut for `add(ChildOf, entity)`.
     *
     * @param second The second element of the pair.
     */
    const Self& child_of(entity_t second) const  {
        return this->add(flecs::ChildOf, second);
    }

    /** Shortcut for `add(DependsOn, entity)`.
     *
     * @param second The second element of the pair.
     */
    const Self& depends_on(entity_t second) const  {
        return this->add(flecs::DependsOn, second);
    }

     /** Shortcut for `add(DependsOn, entity)`.
     *
     * @param second The second element of the pair.
     */
    template <typename E, if_t<is_enum<E>::value> = 0>
    const Self& depends_on(E second) const {
        const auto& et = enum_type<E>(this->world_);
        flecs::entity_t target = et.entity(second);
        return depends_on(target);
    }

    /** Shortcut for `add(SlotOf, entity)`.
     *
     * @param second The second element of the pair.
     */
    const Self& slot_of(entity_t second) const  {
        return this->add(flecs::SlotOf, second);
    }

    /** Shortcut for `add(SlotOf, target(ChildOf))`.
     */
    const Self& slot() const  {
        ecs_check(ecs_get_target(world_, id_, flecs::ChildOf, 0), 
            ECS_INVALID_PARAMETER, "add ChildOf pair before using slot()");
        return this->slot_of(this->target(flecs::ChildOf));
    error:
        return to_base();
    }

    /** Shortcut for `add(ChildOf, entity)`.
     *
     * @tparam T the type associated with the entity.
     */
    template <typename T>
    const Self& child_of() const  {
        return this->child_of(_::type<T>::id(this->world_));
    }
 
    /** Shortcut for `add(DependsOn, entity)`.
     *
     * @tparam T the type associated with the entity.
     */
    template <typename T>
    const Self& depends_on() const  {
        return this->depends_on(_::type<T>::id(this->world_));
    }

    /** Shortcut for `add(SlotOf, entity)`.
     *
     * @tparam T the type associated with the entity.
     */
    template <typename T>
    const Self& slot_of() const  {
        return this->slot_of(_::type<T>::id(this->world_));
    }

    /** Remove a component from an entity.
     *
     * @tparam T the type of the component to remove.
     */
    template <typename T, if_not_t< is_enum<T>::value > = 0>
    const Self& remove() const {
        ecs_remove_id(this->world_, this->id_, _::type<T>::id(this->world_));
        return to_base();
    }

     /** Remove pair for enum.
     * This operation will remove any `(Enum, *)` pair from the entity.
     * 
     * @tparam E The enumeration type.
     */
    template <typename E, if_t< is_enum<E>::value > = 0>
    const Self& remove() const  {
        flecs::entity_t first = _::type<E>::id(this->world_);
        return this->remove(first, flecs::Wildcard);
    }

    /** Remove an entity from an entity.
     *
     * @param entity The entity to remove.
     */
    const Self& remove(entity_t entity) const  {
        ecs_remove_id(this->world_, this->id_, entity);
        return to_base();
    }

    /** Remove a pair.
     * This operation removes a pair from the entity.
     *
     * @param first The first element of the pair.
     * @param second The second element of the pair.
     */
    const Self& remove(entity_t first, entity_t second) const  {
        ecs_remove_pair(this->world_, this->id_, first, second);
        return to_base();
    }

    /** Removes a pair.
     * This operation removes a pair from the entity.
     *
     * @tparam First The first element of the pair
     * @tparam Second The second element of the pair
     */
    template<typename First, typename Second>
    const Self& remove() const  {
        return this->remove<First>(_::type<Second>::id(this->world_));
    }

    /** Remove a pair.
     * This operation removes the pair from the entity.
     *
     * @tparam First The first element of the pair
     * @param second The second element of the pair.
     */
    template<typename First, typename Second, if_not_t< is_enum<Second>::value > = 0>
    const Self& remove(Second second) const  {
        return this->remove(_::type<First>::id(this->world_), second);
    }

    /** Removes a pair.
     * This operation removes a pair from the entity.
     *
     * @tparam Second The second element of the pair
     * @param first The first element of the pair
     */
    template<typename Second>
    const Self& remove_second(flecs::entity_t first) const  {
        return this->remove(first, _::type<Second>::id(this->world_));
    }

    /** Remove a pair.
     * This operation removes the pair from the entity.
     *
     * @tparam First The first element of the pair
     * @param constant the enum constant.
     */
    template<typename First, typename Second, if_t< is_enum<Second>::value > = 0>
    const Self& remove(Second constant) const  {
        const auto& et = enum_type<Second>(this->world_);
        flecs::entity_t second = et.entity(constant);
        return this->remove<First>(second);
    }  

    /** Mark id for auto-overriding.
     * When an entity inherits from a base entity (using the `IsA` relationship)
     * any ids marked for auto-overriding on the base will be overridden
     * automatically by the entity.
     *
     * @param id The id to mark for overriding.
     */
    const Self& auto_override(flecs::id_t id) const  {
        return this->add(ECS_AUTO_OVERRIDE | id);
    }

    /** Mark pair for auto-overriding.
     * @see auto_override(flecs::id_t) const
     *
     * @param first The first element of the pair.
     * @param second The second element of the pair.
     */
    const Self& auto_override(flecs::entity_t first, flecs::entity_t second) const  {
        return this->auto_override(ecs_pair(first, second));
    }

    /** Mark component for auto-overriding.
     * @see auto_override(flecs::id_t) const
     *
     * @tparam T The component to mark for overriding.
     */
    template <typename T>
    const Self& auto_override() const  {
        return this->auto_override(_::type<T>::id(this->world_));
    }

    /** Mark pair for auto-overriding.
     * @see auto_override(flecs::id_t) const
     *
     * @tparam First The first element of the pair.
     * @param second The second element of the pair.
     */
    template <typename First>
    const Self& auto_override(flecs::entity_t second) const  {
        return this->auto_override(_::type<First>::id(this->world_), second);
    }

    /** Mark pair for auto-overriding.
     * @see auto_override(flecs::id_t) const
     *
     * @tparam First The first element of the pair.
     * @tparam Second The second element of the pair.
     */
    template <typename First, typename Second>
    const Self& auto_override() const  {
        return this->auto_override<First>(_::type<Second>::id(this->world_));
    }

    /** Set component, mark component for auto-overriding.
     * @see auto_override(flecs::id_t) const
     *
     * @tparam T The component to set and for which to add the OVERRIDE flag
     * @param val The value to set.
     */
    template <typename T>
    const Self& set_auto_override(const T& val) const  {
        this->auto_override<T>();
        return this->set<T>(val);
    }

    /** Set component, mark component for auto-overriding.
     * @see auto_override(flecs::id_t) const
     *
     * @tparam T The component to set and for which to add the OVERRIDE flag
     * @param val The value to set.
     */
    template <typename T>
    const Self& set_auto_override(T&& val) const  {
        this->auto_override<T>();
        return this->set<T>(FLECS_FWD(val));
    }

    /** Set pair, mark component for auto-overriding.
     * @see auto_override(flecs::id_t) const
     *
     * @tparam First The first element of the pair.
     * @param second The second element of the pair.
     * @param val The value to set.
     */
    template <typename First>
    const Self& set_auto_override(flecs::entity_t second, const First& val) const  {
        this->auto_override<First>(second);
        return this->set<First>(second, val);
    }

    /** Set pair, mark component for auto-overriding.
     * @see auto_override(flecs::id_t) const
     *
     * @tparam First The first element of the pair.
     * @param second The second element of the pair.
     * @param val The value to set.
     */
    template <typename First>
    const Self& set_auto_override(flecs::entity_t second, First&& val) const  {
        this->auto_override<First>(second);
        return this->set<First>(second, FLECS_FWD(val));
    }

    /** Set component, mark component for auto-overriding.
     * @see auto_override(flecs::id_t) const
     *
     * @tparam First The first element of the pair.
     * @tparam Second The second element of the pair.
     * @param val The value to set.
     */
    template <typename First, typename Second, typename P = pair<First, Second>, 
        typename A = actual_type_t<P>, if_not_t< flecs::is_pair<First>::value> = 0>    
    const Self& set_auto_override(const A& val) const  {
        this->auto_override<First, Second>();
        return this->set<First, Second>(val);
    }

    /** Set component, mark component for auto-overriding.
     * @see auto_override(flecs::id_t) const
     *
     * @tparam First The first element of the pair.
     * @tparam Second The second element of the pair.
     * @param val The value to set.
     */
    template <typename First, typename Second, typename P = pair<First, Second>, 
        typename A = actual_type_t<P>, if_not_t< flecs::is_pair<First>::value> = 0>    
    const Self& set_auto_override(A&& val) const  {
        this->auto_override<First, Second>();
        return this->set<First, Second>(FLECS_FWD(val));
    }

    /** Emplace component, mark component for auto-overriding.
     * @see auto_override(flecs::id_t) const
     *
     * @tparam T The component to emplace and override.
     * @param args The arguments to pass to the constructor of `T`.
     */
    template <typename T, typename ... Args>
    const Self& emplace_auto_override(Args&&... args) const  {
        this->auto_override<T>();

        flecs::emplace<T>(this->world_, this->id_, 
            _::type<T>::id(this->world_), FLECS_FWD(args)...);

        return to_base();  
    }

    /** Emplace pair, mark pair for auto-overriding.
     * @see auto_override(flecs::id_t) const
     *
     * @tparam First The first element of the pair to emplace and override.
     * @tparam Second The second element of the pair to emplace and override.
     * @param args The arguments to pass to the constructor of `Second`.
     */
    template <typename First, typename Second, typename P = pair<First, Second>, 
        typename A = actual_type_t<P>, if_not_t< flecs::is_pair<First>::value> = 0,
            typename ... Args>
    const Self& emplace_auto_override(Args&&... args) const  {
        this->auto_override<First, Second>();

        flecs::emplace<A>(this->world_, this->id_, 
            ecs_pair(_::type<First>::id(this->world_),
                _::type<Second>::id(this->world_)),
            FLECS_FWD(args)...);

        return to_base();  
    }

    /** Enable an entity.
     * Enabled entities are matched with systems and can be searched with
     * queries.
     */
    const Self& enable() const  {
        ecs_enable(this->world_, this->id_, true);
        return to_base();
    }

    /** Disable an entity.
     * Disabled entities are not matched with systems and cannot be searched 
     * with queries, unless explicitly specified in the query expression.
     */
    const Self& disable() const  {
        ecs_enable(this->world_, this->id_, false);
        return to_base();
    }

    /** Enable an id.
     * This sets the enabled bit for this component. If this is the first time
     * the component is enabled or disabled, the bitset is added.
     * 
     * @param id The id to enable.
     * @param toggle True to enable, false to disable (default = true).
     *
     * @see ecs_enable_id()
     */
    const Self& enable(flecs::id_t id, bool toggle = true) const  {
        ecs_enable_id(this->world_, this->id_, id, toggle);
        return to_base();       
    }

    /** Enable a component.
     * @see enable(flecs::id_t) const
     *
     * @tparam T The component to enable.
     */
    template<typename T>
    const Self& enable() const  {
        return this->enable(_::type<T>::id(this->world_));
    }

    /** Enable a pair.
     * @see enable(flecs::id_t) const
     *
     * @param first The first element of the pair.
     * @param second The second element of the pair.
     */
    const Self& enable(flecs::id_t first, flecs::id_t second) const  {
        return this->enable(ecs_pair(first, second));
    }

    /** Enable a pair.
     * @see enable(flecs::id_t) const
     *
     * @tparam First The first element of the pair.
     * @param second The second element of the pair.
     */
    template<typename First>
    const Self& enable(flecs::id_t second) const  {
        return this->enable(_::type<First>::id(world_), second);
    }

    /** Enable a pair.
     * @see enable(flecs::id_t) const
     *
     * @tparam First The first element of the pair.
     * @tparam Second The second element of the pair.
     */
    template<typename First, typename Second>
    const Self& enable() const  {
        return this->enable<First>(_::type<Second>::id(world_));
    }

    /** Disable an id.
     * This sets the enabled bit for this id. If this is the first time
     * the id is enabled or disabled, the bitset is added.
     *
     * @param id The id to disable.
     *
     * @see ecs_enable_id()
     * @see enable(flecs::id_t) const
     */
    const Self& disable(flecs::id_t id) const  {
        return this->enable(id, false);
    }

    /** Disable a component.
     * @see disable(flecs::id_t) const
     *
     * @tparam T The component to enable.
     */
    template<typename T>
    const Self& disable() const  {
        return this->disable(_::type<T>::id(world_));
    }

    /** Disable a pair.
     * @see disable(flecs::id_t) const
     *
     * @param first The first element of the pair.
     * @param second The second element of the pair.
     */
    const Self& disable(flecs::id_t first, flecs::id_t second) const  {
        return this->disable(ecs_pair(first, second));
    }

    /** Disable a pair.
     * @see disable(flecs::id_t) const
     *
     * @tparam First The first element of the pair.
     * @param second The second element of the pair.
     */
    template<typename First>
    const Self& disable(flecs::id_t second) const  {
        return this->disable(_::type<First>::id(world_), second);
    }

    /** Disable a pair.
     * @see disable(flecs::id_t) const
     *
     * @tparam First The first element of the pair.
     * @tparam Second The second element of the pair.
     */
    template<typename First, typename Second>
    const Self& disable() const  {
        return this->disable<First>(_::type<Second>::id(world_));
    }

    const Self& set_ptr(entity_t comp, size_t size, const void *ptr) const  {
        ecs_set_id(this->world_, this->id_, comp, size, ptr);
        return to_base();
    }

    const Self& set_ptr(entity_t comp, const void *ptr) const  {
        const flecs::Component *cptr = ecs_get(
            this->world_, comp, EcsComponent);

        /* Can't set if it's not a component */
        ecs_assert(cptr != NULL, ECS_INVALID_PARAMETER, NULL);

        return set_ptr(comp, cptr->size, ptr);
    }

    template<typename T, if_t<is_actual<T>::value> = 0 >
    const Self& set(T&& value) const  {
        flecs::set<T>(this->world_, this->id_, FLECS_FWD(value));
        return to_base();
    }

    template<typename T, if_t<is_actual<T>::value > = 0>
    const Self& set(const T& value) const  {
        flecs::set<T>(this->world_, this->id_, value);
        return to_base();
    }

    template<typename T, typename A = actual_type_t<T>, if_not_t< 
        is_actual<T>::value > = 0>
    const Self& set(A&& value) const  {
        flecs::set<T>(this->world_, this->id_, FLECS_FWD(value));
        return to_base();
    }

    template<typename T, typename A = actual_type_t<T>, if_not_t<
        is_actual<T>::value > = 0>
    const Self& set(const A& value) const  {
        flecs::set<T>(this->world_, this->id_, value);
        return to_base();
    }

    /** Set a pair for an entity.
     * This operation sets the pair value, and uses First as type. If the
     * entity did not yet have the pair, it will be added.
     *
     * @tparam First The first element of the pair.
     * @tparam Second The second element of the pair
     * @param value The value to set.
     */
    template <typename First, typename Second, typename P = pair<First, Second>, 
        typename A = actual_type_t<P>, if_not_t< flecs::is_pair<First>::value> = 0>
    const Self& set(A&& value) const  {
        flecs::set<P>(this->world_, this->id_, FLECS_FWD(value));
        return to_base();
    }

    /** Set a pair for an entity.
     * This operation sets the pair value, and uses First as type. If the
     * entity did not yet have the pair, it will be added.
     *
     * @tparam First The first element of the pair.
     * @tparam Second The second element of the pair
     * @param value The value to set.
     */
    template <typename First, typename Second, typename P = pair<First, Second>, 
        typename A = actual_type_t<P>, if_not_t< flecs::is_pair<First>::value> = 0>
    const Self& set(const A& value) const  {
        flecs::set<P>(this->world_, this->id_, value);
        return to_base();
    }

    /** Set a pair for an entity.
     * This operation sets the pair value, and uses First as type. If the
     * entity did not yet have the pair, it will be added.
     *
     * @tparam First The first element of the pair.
     * @param second The second element of the pair.
     * @param value The value to set.
     */
    template <typename First, typename Second, if_not_t< is_enum<Second>::value > = 0>
    const Self& set(Second second, const First& value) const  {
        auto first = _::type<First>::id(this->world_);
        flecs::set(this->world_, this->id_, value, 
            ecs_pair(first, second));
        return to_base();
    }

    /** Set a pair for an entity.
     * This operation sets the pair value, and uses First as type. If the
     * entity did not yet have the pair, it will be added.
     *
     * @tparam First The first element of the pair.
     * @param second The second element of the pair.
     * @param value The value to set.
     */
    template <typename First, typename Second, if_not_t< is_enum<Second>::value > = 0>
    const Self& set(Second second, First&& value) const  {
        auto first = _::type<First>::id(this->world_);
        flecs::set(this->world_, this->id_, FLECS_FWD(value), 
            ecs_pair(first, second));
        return to_base();
    }

    /** Set a pair for an entity.
     * This operation sets the pair value, and uses First as type. If the
     * entity did not yet have the pair, it will be added.
     *
     * @tparam First The first element of the pair.
     * @param constant The enum constant.
     * @param value The value to set.
     */
    template <typename First, typename Second, if_t< is_enum<Second>::value > = 0>
    const Self& set(Second constant, const First& value) const  {
        const auto& et = enum_type<Second>(this->world_);
        flecs::entity_t second = et.entity(constant);
        return set<First>(second, value);
    }

    /** Set a pair for an entity.
     * This operation sets the pair value, and uses Second as type. If the
     * entity did not yet have the pair, it will be added.
     *
     * @tparam Second The second element of the pair
     * @param first The first element of the pair.
     * @param value The value to set.
     */
    template <typename Second>
    const Self& set_second(entity_t first, const Second& value) const  {
        auto second = _::type<Second>::id(this->world_);
        ecs_assert( ecs_get_type_info(world_, ecs_pair(first, second)) != NULL,
            ECS_INVALID_PARAMETER, "pair is not a component");
        ecs_assert( ecs_get_type_info(world_, ecs_pair(first, second))->component == second,
            ECS_INVALID_PARAMETER, "type of pair is not Second");
        flecs::set(this->world_, this->id_, value, 
            ecs_pair(first, second));
        return to_base();
    }

    /** Set a pair for an entity.
     * This operation sets the pair value, and uses Second as type. If the
     * entity did not yet have the pair, it will be added.
     *
     * @tparam Second The second element of the pair
     * @param first The first element of the pair.
     * @param value The value to set.
     */
    template <typename Second>
    const Self& set_second(entity_t first, Second&& value) const  {
        auto second = _::type<Second>::id(this->world_);
        ecs_assert( ecs_get_type_info(world_, ecs_pair(first, second)) != NULL,
            ECS_INVALID_PARAMETER, "pair is not a component");
        ecs_assert( ecs_get_type_info(world_, ecs_pair(first, second))->component == second,
            ECS_INVALID_PARAMETER, "type of pair is not Second");
        flecs::set(this->world_, this->id_, FLECS_FWD(value), 
            ecs_pair(first, second));
        return to_base();
    }

    template <typename First, typename Second>
    const Self& set_second(const Second& value) const  {
        flecs::set<pair_object<First, Second>>(this->world_, this->id_, value);
        return to_base();
    }    

    /** Set 1..N components.
     * This operation accepts a callback with as arguments the components to
     * set. If the entity does not have all of the provided components, they
     * will be added.
     *
     * This operation is faster than individually calling get for each component
     * as it only obtains entity metadata once. When this operation is called
     * while deferred, its performance is equivalent to that of calling ensure
     * for each component separately.
     *
     * The operation will invoke modified for each component after the callback
     * has been invoked.
     *
     * @param func The callback to invoke.
     */
    template <typename Func>
    const Self& insert(const Func& func) const;

    /** Emplace component.
     * Emplace constructs a component in the storage, which prevents calling the
     * destructor on the value passed into the function.
     *
     * Emplace attempts the following signatures to construct the component:
     *
     * @code
     * T{Args...}
     * T{flecs::entity, Args...}
     * @endcode
     *
     * If the second signature matches, emplace will pass in the current entity 
     * as argument to the constructor, which is useful if the component needs
     * to be aware of the entity to which it has been added.
     *
     * Emplace may only be called for components that have not yet been added
     * to the entity.
     *
     * @tparam T the component to emplace
     * @param args The arguments to pass to the constructor of T
     */
    template<typename T, typename ... Args, typename A = actual_type_t<T>>
    const Self& emplace(Args&&... args) const  {
        flecs::emplace<A>(this->world_, this->id_, 
            _::type<T>::id(this->world_), FLECS_FWD(args)...);
        return to_base();
    }

    template <typename First, typename Second, typename ... Args, typename P = pair<First, Second>, 
        typename A = actual_type_t<P>, if_not_t< flecs::is_pair<First>::value> = 0>
    const Self& emplace(Args&&... args) const  {
        flecs::emplace<A>(this->world_, this->id_, 
            ecs_pair(_::type<First>::id(this->world_),
                _::type<Second>::id(this->world_)),
            FLECS_FWD(args)...);
        return to_base();
    }

    template <typename First, typename ... Args>
    const Self& emplace_first(flecs::entity_t second, Args&&... args) const  {
        auto first = _::type<First>::id(this->world_);
        flecs::emplace<First>(this->world_, this->id_, 
            ecs_pair(first, second),
            FLECS_FWD(args)...);
        return to_base();
    }

    template <typename Second, typename ... Args>
    const Self& emplace_second(flecs::entity_t first, Args&&... args) const  {
        auto second = _::type<Second>::id(this->world_);
        ecs_assert( ecs_get_type_info(world_, ecs_pair(first, second)) != NULL,
            ECS_INVALID_PARAMETER, "pair is not a component");
        ecs_assert( ecs_get_type_info(world_, ecs_pair(first, second))->component == second,
            ECS_INVALID_PARAMETER, "type of pair is not Second");
        flecs::emplace<Second>(this->world_, this->id_, 
            ecs_pair(first, second),
            FLECS_FWD(args)...);
        return to_base();
    }

    /** Entities created in function will have the current entity.
     * This operation is thread safe.
     *
     * @param func The function to call.
     */
    template <typename Func>
    const Self& with(const Func& func) const  {
        ecs_id_t prev = ecs_set_with(this->world_, this->id_);
        func();
        ecs_set_with(this->world_, prev);
        return to_base();
    }

    /** Entities created in function will have `(First, this)`.
     * This operation is thread safe.
     *
     * @tparam First The first element of the pair
     * @param func The function to call.
     */
    template <typename First, typename Func>
    const Self& with(const Func& func) const  {
        with(_::type<First>::id(this->world_), func);
        return to_base();
    }

    /** Entities created in function will have `(first, this)`.
     * This operation is thread safe.
     *
     * @param first The first element of the pair.
     * @param func The function to call.
     */
    template <typename Func>
    const Self& with(entity_t first, const Func& func) const  {
        ecs_id_t prev = ecs_set_with(this->world_, 
            ecs_pair(first, this->id_));
        func();
        ecs_set_with(this->world_, prev);
        return to_base();
    }

    /** The function will be ran with the scope set to the current entity. */
    template <typename Func>
    const Self& scope(const Func& func) const  {
        ecs_entity_t prev = ecs_set_scope(this->world_, this->id_);
        func();
        ecs_set_scope(this->world_, prev);
        return to_base();
    }

    /** Return world scoped to entity */
    scoped_world scope() const {
        return scoped_world(world_, id_);
    }

    /* Set the entity name.
     */
    const Self& set_name(const char *name) const  {
        ecs_set_name(this->world_, this->id_, name);
        return to_base();
    }

    /* Set entity alias.
     */
    const Self& set_alias(const char *name) const  {
        ecs_set_alias(this->world_, this->id_, name);
        return to_base();
    }

#   ifdef FLECS_DOC
/**
 * @file addons/cpp/mixins/doc/entity_builder.inl
 * @brief Doc entity builder mixin.
 */

/** Set human readable name.
 * This adds `(flecs.doc.Description, flecs.Name)` to the entity.
 *
 * @see ecs_doc_set_name()
 * @see flecs::doc::set_name()
 * @see flecs::entity_view::doc_name()
 *
 * @memberof flecs::entity_builder
 * @ingroup cpp_addons_doc
 */
const Self& set_doc_name(const char *name) const {
    ecs_doc_set_name(world_, id_, name);
    return to_base();
}

/** Set brief description.
 * This adds `(flecs.doc.Description, flecs.doc.Brief)` to the entity.
 *
 * @see ecs_doc_set_brief()
 * @see flecs::doc::set_brief()
 * @see flecs::entity_view::doc_brief()
 *
 * @memberof flecs::entity_builder
 * @ingroup cpp_addons_doc
 */
const Self& set_doc_brief(const char *brief) const {
    ecs_doc_set_brief(world_, id_, brief);
    return to_base();
}

/** Set detailed description.
 * This adds `(flecs.doc.Description, flecs.doc.Detail)` to the entity.
 *
 * @see ecs_doc_set_detail()
 * @see flecs::doc::set_detail()
 * @see flecs::entity_view::doc_detail()
 *
 * @memberof flecs::entity_builder
 * @ingroup cpp_addons_doc
 */
const Self& set_doc_detail(const char *detail) const {
    ecs_doc_set_detail(world_, id_, detail);
    return to_base();
}

/** Set link to external documentation.
 * This adds `(flecs.doc.Description, flecs.doc.Link)` to the entity.
 *
 * @see ecs_doc_set_link()
 * @see flecs::doc::set_link()
 * @see flecs::entity_view::doc_link()
 *
 * @memberof flecs::entity_builder
 * @ingroup cpp_addons_doc
 */
const Self& set_doc_link(const char *link) const {
    ecs_doc_set_link(world_, id_, link);
    return to_base();
}

/** Set doc color.
 * This adds `(flecs.doc.Description, flecs.doc.Color)` to the entity.
 *
 * @see ecs_doc_set_color()
 * @see flecs::doc::set_color()
 * @see flecs::entity_view::doc_color()
 *
 * @memberof flecs::entity_builder
 * @ingroup cpp_addons_doc
 */
const Self& set_doc_color(const char *color) const {
    ecs_doc_set_color(world_, id_, color);
    return to_base();
}

/** Set doc UUID.
 * This adds `(flecs.doc.Description, flecs.doc.Uuid)` to the entity.
 *
 * @see ecs_doc_set_uuid()
 * @see flecs::doc::set_uuid()
 * @see flecs::entity_view::doc_uuid()
 *
 * @memberof flecs::entity_builder
 * @ingroup cpp_addons_doc
 */
const Self& set_doc_uuid(const char *uuid) const {
    ecs_doc_set_uuid(world_, id_, uuid);
    return to_base();
}

#   endif

#   ifdef FLECS_META
/**
 * @file addons/cpp/mixins/meta/entity_builder.inl
 * @brief Meta entity builder mixin.
 */

/**
 * @memberof flecs::entity_view
 * @ingroup cpp_addons_meta
 * 
 * @{
 */

/** Make entity a unit */
const Self& unit(
    const char *symbol, 
    flecs::entity_t prefix = 0,
    flecs::entity_t base = 0,
    flecs::entity_t over = 0,
    int32_t factor = 0,
    int32_t power = 0) const
{
    ecs_unit_desc_t desc = {};
    desc.entity = this->id_;
    desc.symbol = const_cast<char*>(symbol); /* safe, will be copied in */
    desc.base = base;
    desc.over = over;
    desc.prefix = prefix;
    desc.translation.factor = factor;
    desc.translation.power = power;
    ecs_unit_init(this->world(), &desc);

    return to_base();
}

/** Make entity a derived unit */
const Self& unit( 
    flecs::entity_t prefix = 0,
    flecs::entity_t base = 0,
    flecs::entity_t over = 0,
    int32_t factor = 0,
    int32_t power = 0) const
{
    ecs_unit_desc_t desc = {};
    desc.entity = this->id_;
    desc.base = base;
    desc.over = over;
    desc.prefix = prefix;
    desc.translation.factor = factor;
    desc.translation.power = power;
    ecs_unit_init(this->world(), &desc);

    return to_base();
}

/** Make entity a derived unit */
const Self& unit_prefix( 
    const char *symbol,
    int32_t factor = 0,
    int32_t power = 0) const
{
    ecs_unit_prefix_desc_t desc = {};
    desc.entity = this->id_;
    desc.symbol = const_cast<char*>(symbol); /* safe, will be copied in */
    desc.translation.factor = factor;
    desc.translation.power = power;
    ecs_unit_prefix_init(this->world(), &desc);

    return to_base();
}

/** Add quantity to unit */
const Self& quantity(flecs::entity_t quantity) const {
    ecs_add_pair(this->world(), this->id(), flecs::Quantity, quantity);
    return to_base();
}

/** Make entity a unity prefix */
template <typename Quantity>
const Self& quantity() const {
    return this->quantity(_::type<Quantity>::id(this->world()));
}

/** Make entity a quantity */
const Self& quantity() const {
    ecs_add_id(this->world(), this->id(), flecs::Quantity);
    return to_base();
}

/** @} */

#   endif

#   ifdef FLECS_JSON
/**
 * @file addons/cpp/mixins/json/entity_builder.inl
 * @brief JSON entity mixin.
 */

/** Set component from JSON.
 * 
 * @memberof flecs::entity_builder
 * @ingroup cpp_addons_json
 */
const Self& set_json(
    flecs::id_t e, 
    const char *json, 
    flecs::from_json_desc_t *desc = nullptr) const
{
    flecs::entity_t type = ecs_get_typeid(world_, e);
    if (!type) {
        ecs_err("id is not a type");
        return to_base();
    }

    void *ptr = ecs_ensure_id(world_, id_, e);
    ecs_assert(ptr != NULL, ECS_INTERNAL_ERROR, NULL);
    ecs_ptr_from_json(world_, type, ptr, json, desc);
    ecs_modified_id(world_, id_, e);

    return to_base();
}

/** Set pair from JSON.
 * 
 * @memberof flecs::entity_builder
 * @ingroup cpp_addons_json
 */
const Self& set_json(
    flecs::entity_t r, 
    flecs::entity_t t,
    const char *json, 
    flecs::from_json_desc_t *desc = nullptr) const
{
    return set_json(ecs_pair(r, t), json, desc);
}

/** Set component from JSON.
 * 
 * @memberof flecs::entity_builder
 * @ingroup cpp_addons_json
 */
template <typename T>
const Self& set_json(
    const char *json, 
    flecs::from_json_desc_t *desc = nullptr) const
{
    return set_json(_::type<T>::id(world_), json, desc);
}

/** Set pair from JSON.
 * 
 * @memberof flecs::entity_builder
 * @ingroup cpp_addons_json
 */
template <typename R, typename T>
const Self& set_json(
    const char *json, 
    flecs::from_json_desc_t *desc = nullptr) const
{
    return set_json(
        _::type<R>::id(world_), 
        _::type<T>::id(world_),
        json, desc);
}

/** Set pair from JSON.
 * 
 * @memberof flecs::entity_builder
 * @ingroup cpp_addons_json
 */
template <typename R>
const Self& set_json(
    flecs::entity_t t,
    const char *json, 
    flecs::from_json_desc_t *desc = nullptr) const
{
    return set_json(
        _::type<R>::id(world_), t,
        json, desc);
}

/** Set pair from JSON.
 * 
 * @memberof flecs::entity_builder
 * @ingroup cpp_addons_json
 */
template <typename T>
const Self& set_json_second(
    flecs::entity_t r,
    const char *json, 
    flecs::from_json_desc_t *desc = nullptr) const
{
    return set_json(
        r, _::type<T>::id(world_),
        json, desc);
}

#   endif

/**
 * @file addons/cpp/mixins/event/entity_builder.inl
 * @brief Event entity mixin.
 */

/** Observe event on entity
 * 
 * @memberof flecs::entity_builder
 * 
 * @param evt The event id.
 * @param callback The observer callback.
 * @return Event builder.
 */
template <typename Func>
const Self& observe(flecs::entity_t evt, Func&& callback) const;

/** Observe event on entity
 * 
 * @memberof flecs::entity_builder
 * 
 * @tparam Evt The event type.
 * @param callback The observer callback.
 * @return Event builder.
 */
template <typename Evt, typename Func>
const Self& observe(Func&& callback) const;

/** Observe event on entity
 * 
 * @memberof flecs::entity_builder
 *
 * @param callback The observer callback.
 * @return Event builder.
 */
template <typename Func>
const Self& observe(Func&& callback) const;




protected:
    const Self& to_base() const  {
        return *static_cast<const Self*>(this);
    }
};

}


/**
 * @defgroup cpp_entities Entities
 * @ingroup cpp_core
 * Entity operations.
 *
 * @{
 */

namespace flecs
{

/** Entity.
 * Class with read/write operations for entities.
 *
 * @ingroup cpp_entities
*/
struct entity : entity_builder<entity>
{
    entity() : entity_builder<entity>() { }

    /** Create entity.
     *
     * @param world The world in which to create the entity.
     */
    explicit entity(world_t *world)
        : entity_builder()
    {
        world_ = world;
        if (!ecs_get_scope(world_) && !ecs_get_with(world_)) {
            id_ = ecs_new(world);
        } else {
            ecs_entity_desc_t desc = {};
            id_ = ecs_entity_init(world_, &desc);
        }
    }

    /** Wrap an existing entity id.
     *
     * @param world The world in which the entity is created.
     * @param id The entity id.
     */
    explicit entity(const flecs::world_t *world, flecs::entity_t id) {
        world_ = const_cast<flecs::world_t*>(world);
        id_ = id;
    }

    /** Create a named entity.
     * Named entities can be looked up with the lookup functions. Entity names
     * may be scoped, where each element in the name is separated by "::".
     * For example: "Foo::Bar". If parts of the hierarchy in the scoped name do
     * not yet exist, they will be automatically created.
     *
     * @param world The world in which to create the entity.
     * @param name The entity name.
     */
    explicit entity(world_t *world, const char *name)
        : entity_builder()
    {
        world_ = world;

        ecs_entity_desc_t desc = {};
        desc.name = name;
        desc.sep = "::";
        desc.root_sep = "::";
        id_ = ecs_entity_init(world, &desc);
    }

    /** Conversion from flecs::entity_t to flecs::entity.
     *
     * @param id The entity_t value to convert.
     */
    explicit entity(entity_t id)
        : entity_builder( nullptr, id ) { }

    #ifndef ensure

    /** Get mutable component value.
     * This operation returns a mutable pointer to the component. If the entity
     * did not yet have the component, it will be added. If a base entity had
     * the component, it will be overridden, and the value of the base component
     * will be copied to the entity before this function returns.
     *
     * @tparam T The component to get.
     * @return Pointer to the component value.
     */
    template <typename T>
    T& ensure() const {
        auto comp_id = _::type<T>::id(world_);
        ecs_assert(_::type<T>::size() != 0, ECS_INVALID_PARAMETER,
            "operation invalid for empty type");
        return *static_cast<T*>(ecs_ensure_id(world_, id_, comp_id));
    }

    /** Get mutable component value (untyped).
     * This operation returns a mutable pointer to the component. If the entity
     * did not yet have the component, it will be added. If a base entity had
     * the component, it will be overridden, and the value of the base component
     * will be copied to the entity before this function returns.
     *
     * @param comp The component to get.
     * @return Pointer to the component value.
     */
    void* ensure(entity_t comp) const {
        return ecs_ensure_id(world_, id_, comp);
    }

    /** Get mutable pointer for a pair.
     * This operation gets the value for a pair from the entity.
     *
     * @tparam First The first part of the pair.
     * @tparam Second the second part of the pair.
     */
    template <typename First, typename Second, typename P = pair<First, Second>,
        typename A = actual_type_t<P>, if_not_t< flecs::is_pair<First>::value> = 0>
    A& ensure() const {
        return *static_cast<A*>(ecs_ensure_id(world_, id_, ecs_pair(
            _::type<First>::id(world_),
            _::type<Second>::id(world_))));
    }

    /** Get mutable pointer for the first element of a pair.
     * This operation gets the value for a pair from the entity.
     *
     * @tparam First The first part of the pair.
     * @param second The second element of the pair.
     */
    template <typename First>
    First& ensure(entity_t second) const {
        auto first = _::type<First>::id(world_);
        ecs_assert(_::type<First>::size() != 0, ECS_INVALID_PARAMETER,
            "operation invalid for empty type");
        return *static_cast<First*>(
            ecs_ensure_id(world_, id_, ecs_pair(first, second)));
    }

    /** Get mutable pointer for a pair (untyped).
     * This operation gets the value for a pair from the entity. If neither the
     * first nor second element of the pair is a component, the operation will
     * fail.
     *
     * @param first The first element of the pair.
     * @param second The second element of the pair.
     */
    void* ensure(entity_t first, entity_t second) const {
        return ecs_ensure_id(world_, id_, ecs_pair(first, second));
    }

    /** Get mutable pointer for the second element of a pair.
     * This operation gets the value for a pair from the entity.
     *
     * @tparam Second The second element of the pair.
     * @param first The first element of the pair.
     */
    template <typename Second>
    Second& ensure_second(entity_t first) const {
        auto second = _::type<Second>::id(world_);
        ecs_assert( ecs_get_type_info(world_, ecs_pair(first, second)) != NULL,
            ECS_INVALID_PARAMETER, "pair is not a component");
        ecs_assert( ecs_get_type_info(world_, ecs_pair(first, second))->component == second,
            ECS_INVALID_PARAMETER, "type of pair is not Second");
        ecs_assert(_::type<Second>::size() != 0, ECS_INVALID_PARAMETER,
            "operation invalid for empty type");
        return *static_cast<Second*>(
            ecs_ensure_id(world_, id_, ecs_pair(first, second)));
    }

    #endif

    /** Signal that component was modified.
     *
     * @tparam T component that was modified.
     */
    template <typename T>
    void modified() const {
        auto comp_id = _::type<T>::id(world_);
        ecs_assert(_::type<T>::size() != 0, ECS_INVALID_PARAMETER,
            "operation invalid for empty type");
        this->modified(comp_id);
    }

    /** Signal that the first element of a pair was modified.
     *
     * @tparam First The first part of the pair.
     * @tparam Second the second part of the pair.
     */
    template <typename First, typename Second, typename A = actual_type_t<flecs::pair<First, Second>>>
    void modified() const {
        auto first = _::type<First>::id(world_);
        auto second = _::type<Second>::id(world_);
        ecs_assert(_::type<A>::size() != 0, ECS_INVALID_PARAMETER,
            "operation invalid for empty type");
        this->modified(first, second);
    }

    /** Signal that the first part of a pair was modified.
     *
     * @tparam First The first part of the pair.
     * @param second The second element of the pair.
     */
    template <typename First>
    void modified(entity_t second) const {
        auto first = _::type<First>::id(world_);
        ecs_assert(_::type<First>::size() != 0, ECS_INVALID_PARAMETER,
            "operation invalid for empty type");
        this->modified(first, second);
    }

    /** Signal that a pair has modified (untyped).
     * If neither the first or second element of the pair are a component, the
     * operation will fail.
     *
     * @param first The first element of the pair.
     * @param second The second element of the pair.
     */
    void modified(entity_t first, entity_t second) const {
        this->modified(ecs_pair(first, second));
    }

    /** Signal that component was modified.
     *
     * @param comp component that was modified.
     */
    void modified(entity_t comp) const {
        ecs_modified_id(world_, id_, comp);
    }

    /** Get reference to component.
     * A reference allows for quick and safe access to a component value, and is
     * a faster alternative to repeatedly calling 'get' for the same component.
     *
     * @tparam T component for which to get a reference.
     * @return The reference.
     */
    template <typename T, if_t< is_actual<T>::value > = 0>
    ref<T> get_ref() const {
        return ref<T>(world_, id_, _::type<T>::id(world_));
    }

    /** Get reference to component.
     * Overload for when T is not the same as the actual type, which happens
     * when using pair types.
     * A reference allows for quick and safe access to a component value, and is
     * a faster alternative to repeatedly calling 'get' for the same component.
     *
     * @tparam T component for which to get a reference.
     * @return The reference.
     */
    template <typename T, typename A = actual_type_t<T>, if_t< flecs::is_pair<T>::value > = 0>
    ref<A> get_ref() const {
        return ref<A>(world_, id_,
                      ecs_pair(_::type<typename T::first>::id(world_),
                               _::type<typename T::second>::id(world_)));
    }


    template <typename First, typename Second, typename P = flecs::pair<First, Second>,
        typename A = actual_type_t<P>>
    ref<A> get_ref() const {
        return ref<A>(world_, id_,
            ecs_pair(_::type<First>::id(world_), _::type<Second>::id(world_)));
    }

    template <typename First>
    ref<First> get_ref(flecs::entity_t second) const {
        auto first = _::type<First>::id(world_);
        return ref<First>(world_, id_, ecs_pair(first, second));
    }

    template <typename Second>
    ref<Second> get_ref_second(flecs::entity_t first) const {
        auto second = _::type<Second>::id(world_);
        ecs_assert( ecs_get_type_info(world_, ecs_pair(first, second)) != NULL,
            ECS_INVALID_PARAMETER, "pair is not a component");
        ecs_assert( ecs_get_type_info(world_, ecs_pair(first, second))->component == second,
            ECS_INVALID_PARAMETER, "type of pair is not Second");
        return ref<Second>(world_, id_, ecs_pair(first, second));
    }

    /** Clear an entity.
     * This operation removes all components from an entity without recycling
     * the entity id.
     *
     * @see ecs_clear()
     */
    void clear() const {
        ecs_clear(world_, id_);
    }

    /** Delete an entity.
     * Entities have to be deleted explicitly, and are not deleted when the
     * entity object goes out of scope.
     *
     * @see ecs_delete()
     */
    void destruct() const {
        ecs_delete(world_, id_);
    }

    /** Return entity as entity_view.
     * This returns an entity_view instance for the entity which is a readonly
     * version of the entity class.
     *
     * This is similar to a regular upcast, except that this method ensures that
     * the entity_view instance is instantiated with a world vs. a stage, which
     * a regular upcast does not guarantee.
     */
    flecs::entity_view view() const {
        return flecs::entity_view(
            const_cast<flecs::world_t*>(ecs_get_world(world_)), id_);
    }

    /** Entity id 0.
     * This function is useful when the API must provide an entity that
     * belongs to a world, but the entity id is 0.
     *
     * @param world The world.
     */
    static
    flecs::entity null(const flecs::world_t *world) {
        flecs::entity result;
        result.world_ = const_cast<flecs::world_t*>(world);
        return result;
    }

    static
    flecs::entity null() {
        return flecs::entity();
    }

#   ifdef FLECS_JSON

/** Deserialize entity to JSON.
 * 
 * @memberof flecs::entity
 * @ingroup cpp_addons_json
 */
const char* from_json(const char *json) {
    return ecs_entity_from_json(world_, id_, json, nullptr);
}

#   endif
};

} // namespace flecs

/** @} */

/**
 * @file addons/cpp/delegate.hpp
 * @brief Wrappers around C++ functions that provide callbacks for C APIs.
 */

#pragma once

#include <utility> // std::declval

namespace flecs
{

namespace _ 
{

// Binding ctx for component hooks
struct component_binding_ctx {
    void *on_add = nullptr;
    void *on_remove = nullptr;
    void *on_set = nullptr;
    ecs_ctx_free_t free_on_add = nullptr;
    ecs_ctx_free_t free_on_remove = nullptr;
    ecs_ctx_free_t free_on_set = nullptr;

    ~component_binding_ctx() {
        if (on_add && free_on_add) {
            free_on_add(on_add);
        }
        if (on_remove && free_on_remove) {
            free_on_remove(on_remove);
        }
        if (on_set && free_on_set) {
            free_on_set(on_set);
        }
    }
};

// Utility to convert template argument pack to array of term ptrs
struct field_ptr {
    void *ptr = nullptr;
    int8_t index = 0;
    bool is_ref = false;
    bool is_row = false;
};

template <typename ... Components>
struct field_ptrs {
    using array = flecs::array<_::field_ptr, sizeof...(Components)>;

    void populate(const ecs_iter_t *iter) {
        populate(iter, 0, static_cast<
            remove_reference_t<
                remove_pointer_t<Components>>
                    *>(nullptr)...);
    }

    void populate_self(const ecs_iter_t *iter) {
        populate_self(iter, 0, static_cast<
            remove_reference_t<
                remove_pointer_t<Components>>
                    *>(nullptr)...);
    }

    array fields_;

private:
    void populate(const ecs_iter_t*, size_t) { }

    template <typename T, typename... Targs, 
        typename A = remove_pointer_t<actual_type_t<T>>,
            if_not_t< is_empty<A>::value > = 0>
    void populate(const ecs_iter_t *iter, size_t index, T, Targs... comps) {
        if (iter->row_fields & (1llu << index)) {
            /* Need to fetch the value with ecs_field_at() */
            fields_[index].is_row = true;
            fields_[index].is_ref = true;
            fields_[index].index = static_cast<int8_t>(index);
        } else {
            fields_[index].ptr = ecs_field_w_size(iter, sizeof(A), 
                static_cast<int8_t>(index));
            fields_[index].is_ref = iter->sources[index] != 0;
        }

        populate(iter, index + 1, comps ...);
    }

    template <typename T, typename... Targs, 
        typename A = remove_pointer_t<actual_type_t<T>>,
            if_t< is_empty<A>::value > = 0>
    void populate(const ecs_iter_t *iter, size_t index, T, Targs... comps) {
        populate(iter, index + 1, comps ...);
    }

    void populate_self(const ecs_iter_t*, size_t) { }

    template <typename T, typename... Targs, 
        typename A = remove_pointer_t<actual_type_t<T>>,
            if_not_t< is_empty<A>::value > = 0>
    void populate_self(const ecs_iter_t *iter, size_t index, T, Targs... comps) {
        fields_[index].ptr = ecs_field_w_size(iter, sizeof(A), 
            static_cast<int8_t>(index));
        fields_[index].is_ref = false;
        ecs_assert(iter->sources[index] == 0, ECS_INTERNAL_ERROR, NULL);
        populate_self(iter, index + 1, comps ...);
    }

    template <typename T, typename... Targs,
        typename A = remove_pointer_t<actual_type_t<T>>,
            if_t< is_empty<A>::value > = 0>
    void populate_self(const ecs_iter_t *iter, size_t index, T, Targs... comps) {
        populate(iter, index + 1, comps ...);
    }
};

struct delegate { };

// Template that figures out from the template parameters of a query/system
// how to pass the value to the each callback
template <typename T, typename = int>
struct each_field { };

// Base class
struct each_column_base {
    each_column_base(const _::field_ptr& field, size_t row) 
        : field_(field), row_(row) {
    }

protected:
    const _::field_ptr& field_;
    size_t row_;    
};

// If type is not a pointer, return a reference to the type (default case)
template <typename T>
struct each_field<T, if_t< !is_pointer<T>::value && 
        !is_empty<actual_type_t<T>>::value && is_actual<T>::value > > 
    : each_column_base 
{
    each_field(const flecs::iter_t*, _::field_ptr& field, size_t row) 
        : each_column_base(field, row) { }

    T& get_row() {
        return static_cast<T*>(this->field_.ptr)[this->row_];
    }  
};

// If argument type is not the same as actual component type, return by value.
// This requires that the actual type can be converted to the type.
// A typical scenario where this happens is when using flecs::pair types.
template <typename T>
struct each_field<T, if_t< !is_pointer<T>::value &&
        !is_empty<actual_type_t<T>>::value && !is_actual<T>::value> > 
    : each_column_base 
{
    each_field(const flecs::iter_t*, _::field_ptr& field, size_t row) 
        : each_column_base(field, row) { }

    T get_row() {
        return static_cast<actual_type_t<T>*>(this->field_.ptr)[this->row_];
    }  
};

// If type is empty (indicating a tag) the query will pass a nullptr. To avoid
// returning nullptr to reference arguments, return a temporary value.
template <typename T>
struct each_field<T, if_t< is_empty<actual_type_t<T>>::value && 
        !is_pointer<T>::value > > 
    : each_column_base 
{
    each_field(const flecs::iter_t*, _::field_ptr& field, size_t row) 
        : each_column_base(field, row) { }

    T get_row() {
        return actual_type_t<T>();
    }
};

// If type is a pointer (indicating an optional value) don't index with row if
// the field is not set.
template <typename T>
struct each_field<T, if_t< is_pointer<T>::value && 
        !is_empty<actual_type_t<T>>::value > > 
    : each_column_base 
{
    each_field(const flecs::iter_t*, _::field_ptr& field, size_t row) 
        : each_column_base(field, row) { }

    actual_type_t<T> get_row() {
        if (this->field_.ptr) {
            return &static_cast<actual_type_t<T>>(this->field_.ptr)[this->row_];
        } else {
            // optional argument doesn't have a value
            return nullptr;
        }
    }
};

// If the query contains component references to other entities, check if the
// current argument is one.
template <typename T, typename = int>
struct each_ref_field : public each_field<T> {
    each_ref_field(const flecs::iter_t *iter, _::field_ptr& field, size_t row)
        : each_field<T>(iter, field, row) {

        if (field.is_ref) {
            // If this is a reference, set the row to 0 as a ref always is a
            // single value, not an array. This prevents the application from
            // having to do an if-check on whether the column is owned.
            //
            // This check only happens when the current table being iterated
            // over caused the query to match a reference. The check is
            // performed once per iterated table.
            this->row_ = 0;
        }

        if (field.is_row) {
            field.ptr = ecs_field_at_w_size(iter, sizeof(T), field.index, 
                static_cast<int32_t>(row));
        }
    }
};

// Type that handles passing components to each callbacks
template <typename Func, typename ... Components>
struct each_delegate : public delegate {
    using Terms = typename field_ptrs<Components ...>::array;

    template < if_not_t< is_same< decay_t<Func>, decay_t<Func>& >::value > = 0>
    explicit each_delegate(Func&& func) noexcept 
        : func_(FLECS_MOV(func)) { }

    explicit each_delegate(const Func& func) noexcept 
        : func_(func) { }

    // Invoke object directly. This operation is useful when the calling
    // function has just constructed the delegate, such as what happens when
    // iterating a query.
    void invoke(ecs_iter_t *iter) const {
        field_ptrs<Components...> terms;

        iter->flags |= EcsIterCppEach;

        if (iter->ref_fields | iter->up_fields) {
            terms.populate(iter);
            invoke_unpack< each_ref_field >(iter, func_, 0, terms.fields_);
        } else {
            terms.populate_self(iter);
            invoke_unpack< each_field >(iter, func_, 0, terms.fields_);
        }
    }

    // Static function that can be used as callback for systems/triggers
    static void run(ecs_iter_t *iter) {
        auto self = static_cast<const each_delegate*>(iter->callback_ctx);
        ecs_assert(self != nullptr, ECS_INTERNAL_ERROR, NULL);
        self->invoke(iter);
    }

    // Create instance of delegate
    static each_delegate* make(const Func& func) {
        return FLECS_NEW(each_delegate)(func);
    }

    // Function that can be used as callback to free delegate
    static void destruct(void *obj) {
        _::free_obj<each_delegate>(obj);
    }

    // Static function to call for component on_add hook
    static void run_add(ecs_iter_t *iter) {
        component_binding_ctx *ctx = reinterpret_cast<component_binding_ctx*>(
            iter->callback_ctx);
        iter->callback_ctx = ctx->on_add;
        run(iter);
    }

    // Static function to call for component on_remove hook
    static void run_remove(ecs_iter_t *iter) {
        component_binding_ctx *ctx = reinterpret_cast<component_binding_ctx*>(
            iter->callback_ctx);
        iter->callback_ctx = ctx->on_remove;
        run(iter);
    }

    // Static function to call for component on_set hook
    static void run_set(ecs_iter_t *iter) {
        component_binding_ctx *ctx = reinterpret_cast<component_binding_ctx*>(
            iter->callback_ctx);
        iter->callback_ctx = ctx->on_set;
        run(iter);
    }

private:
    // func(flecs::entity, Components...)
    template <template<typename X, typename = int> class ColumnType, 
        typename... Args,
        typename Fn = Func,
        decltype(std::declval<const Fn&>()(
            std::declval<flecs::entity>(),
            std::declval<ColumnType< remove_reference_t<Components> > >().get_row()...), 0) = 0>
    static void invoke_callback(
        ecs_iter_t *iter, const Func& func, size_t i, Args... comps) 
    {
        func(flecs::entity(iter->world, iter->entities[i]),
            (ColumnType< remove_reference_t<Components> >(iter, comps, i)
                .get_row())...);
    }

    // func(flecs::iter&, size_t row, Components...)
    template <template<typename X, typename = int> class ColumnType, 
        typename... Args,
        typename Fn = Func,
        decltype(std::declval<const Fn&>()(
            std::declval<flecs::iter&>(),
            std::declval<size_t&>(),
            std::declval<ColumnType< remove_reference_t<Components> > >().get_row()...), 0) = 0>
    static void invoke_callback(
        ecs_iter_t *iter, const Func& func, size_t i, Args... comps) 
    {
        flecs::iter it(iter);
        func(it, i, (ColumnType< remove_reference_t<Components> >(iter, comps, i)
            .get_row())...);
    }

    // func(Components...)
    template <template<typename X, typename = int> class ColumnType, 
        typename... Args,
        typename Fn = Func,
        decltype(std::declval<const Fn&>()(
            std::declval<ColumnType< remove_reference_t<Components> > >().get_row()...), 0) = 0>
    static void invoke_callback(
        ecs_iter_t *iter, const Func& func, size_t i, Args... comps) 
    {
        func((ColumnType< remove_reference_t<Components> >(iter, comps, i)
            .get_row())...);
    }

    template <template<typename X, typename = int> class ColumnType, 
        typename... Args, if_t< 
            sizeof...(Components) == sizeof...(Args)> = 0>
    static void invoke_unpack(
        ecs_iter_t *iter, const Func& func, size_t, Terms&, Args... comps) 
    {
        ECS_TABLE_LOCK(iter->world, iter->table);

        size_t count = static_cast<size_t>(iter->count);
        if (count == 0 && !iter->table) {
            // If query has no This terms, count can be 0. Since each does not
            // have an entity parameter, just pass through components
            count = 1;
        }

        for (size_t i = 0; i < count; i ++) {
            invoke_callback<ColumnType>(iter, func, i, comps...);
        }

        ECS_TABLE_UNLOCK(iter->world, iter->table);
    }

    template <template<typename X, typename = int> class ColumnType, 
        typename... Args, if_t< sizeof...(Components) != sizeof...(Args) > = 0>
    static void invoke_unpack(ecs_iter_t *iter, const Func& func, 
        size_t index, Terms& columns, Args... comps) 
    {
        invoke_unpack<ColumnType>(
            iter, func, index + 1, columns, comps..., columns[index]);
    }    

public:
    Func func_;
};

template <typename Func, typename ... Components>
struct find_delegate : public delegate {
    using Terms = typename field_ptrs<Components ...>::array;

    template < if_not_t< is_same< decay_t<Func>, decay_t<Func>& >::value > = 0>
    explicit find_delegate(Func&& func) noexcept 
        : func_(FLECS_MOV(func)) { }

    explicit find_delegate(const Func& func) noexcept 
        : func_(func) { }

    // Invoke object directly. This operation is useful when the calling
    // function has just constructed the delegate, such as what happens when
    // iterating a query.
    flecs::entity invoke(ecs_iter_t *iter) const {
        field_ptrs<Components...> terms;

        iter->flags |= EcsIterCppEach;

        if (iter->ref_fields | iter->up_fields) {
            terms.populate(iter);
            return invoke_callback< each_ref_field >(iter, func_, 0, terms.fields_);
        } else {
            terms.populate_self(iter);
            return invoke_callback< each_field >(iter, func_, 0, terms.fields_);
        }
    }

private:
    // Number of function arguments is one more than number of components, pass
    // entity as argument.
    template <template<typename X, typename = int> class ColumnType,
        typename... Args,
        typename Fn = Func,
        if_t<sizeof...(Components) == sizeof...(Args)> = 0,
        decltype(bool(std::declval<const Fn&>()(
            std::declval<flecs::entity>(),
            std::declval<ColumnType< remove_reference_t<Components> > >().get_row()...))) = true>
    static flecs::entity invoke_callback(
        ecs_iter_t *iter, const Func& func, size_t, Terms&, Args... comps) 
    {
        ECS_TABLE_LOCK(iter->world, iter->table);

        ecs_world_t *world = iter->world;
        size_t count = static_cast<size_t>(iter->count);
        flecs::entity result;

        for (size_t i = 0; i < count; i ++) {
            if (func(flecs::entity(world, iter->entities[i]),
                (ColumnType< remove_reference_t<Components> >(iter, comps, i)
                    .get_row())...))
            {
                result = flecs::entity(world, iter->entities[i]);
                break;
            }
        }

        ECS_TABLE_UNLOCK(iter->world, iter->table);

        return result;
    }

    // Number of function arguments is two more than number of components, pass
    // iter + index as argument.
    template <template<typename X, typename = int> class ColumnType,
        typename... Args,
        typename Fn = Func,
        if_t<sizeof...(Components) == sizeof...(Args)> = 0,
        decltype(bool(std::declval<const Fn&>()(
            std::declval<flecs::iter&>(),
            std::declval<size_t&>(),
            std::declval<ColumnType< remove_reference_t<Components> > >().get_row()...))) = true>
    static flecs::entity invoke_callback(
        ecs_iter_t *iter, const Func& func, size_t, Terms&, Args... comps) 
    {
        size_t count = static_cast<size_t>(iter->count);
        if (count == 0) {
            // If query has no This terms, count can be 0. Since each does not
            // have an entity parameter, just pass through components
            count = 1;
        }

        flecs::iter it(iter);
        flecs::entity result;

        ECS_TABLE_LOCK(iter->world, iter->table);

        for (size_t i = 0; i < count; i ++) {
            if (func(it, i, 
                (ColumnType< remove_reference_t<Components> >(iter, comps, i)
                    .get_row())...))
            {
                result = flecs::entity(iter->world, iter->entities[i]);
                break;
            }
        }

        ECS_TABLE_UNLOCK(iter->world, iter->table);

        return result;
    }

    // Number of function arguments is equal to number of components, no entity
    template <template<typename X, typename = int> class ColumnType,
        typename... Args,
        typename Fn = Func,
        if_t<sizeof...(Components) == sizeof...(Args)> = 0,
        decltype(bool(std::declval<const Fn&>()(
            std::declval<ColumnType< remove_reference_t<Components> > >().get_row()...))) = true>
    static flecs::entity invoke_callback(
        ecs_iter_t *iter, const Func& func, size_t, Terms&, Args... comps) 
    {
        size_t count = static_cast<size_t>(iter->count);
        if (count == 0) {
            // If query has no This terms, count can be 0. Since each does not
            // have an entity parameter, just pass through components
            count = 1;
        }

        flecs::iter it(iter);
        flecs::entity result;

        ECS_TABLE_LOCK(iter->world, iter->table);

        for (size_t i = 0; i < count; i ++) {
            if (func(
                (ColumnType< remove_reference_t<Components> >(iter, comps, i)
                    .get_row())...))
            {
                result = flecs::entity(iter->world, iter->entities[i]);
                break;
            }
        }

        ECS_TABLE_UNLOCK(iter->world, iter->table);

        return result;
    }

    template <template<typename X, typename = int> class ColumnType, 
        typename... Args, if_t< sizeof...(Components) != sizeof...(Args) > = 0>
    static flecs::entity invoke_callback(ecs_iter_t *iter, const Func& func, 
        size_t index, Terms& columns, Args... comps) 
    {
        return invoke_callback<ColumnType>(
            iter, func, index + 1, columns, comps..., columns[index]);
    }

    Func func_;
};

////////////////////////////////////////////////////////////////////////////////
//// Utility class to invoke a system iterate action
////////////////////////////////////////////////////////////////////////////////

template <typename Func>
struct run_delegate : delegate {
    template < if_not_t< is_same< decay_t<Func>, decay_t<Func>& >::value > = 0>
    explicit run_delegate(Func&& func) noexcept 
        : func_(FLECS_MOV(func)) { }

    explicit run_delegate(const Func& func) noexcept 
        : func_(func) { }

    // Invoke object directly. This operation is useful when the calling
    // function has just constructed the delegate, such as what happens when
    // iterating a query.
    void invoke(ecs_iter_t *iter) const {
        flecs::iter it(iter);
        iter->flags &= ~EcsIterIsValid;
        func_(it);
    }

    // Static function that can be used as callback for systems/triggers
    static void run(ecs_iter_t *iter) {
        auto self = static_cast<const run_delegate*>(iter->run_ctx);
        ecs_assert(self != nullptr, ECS_INTERNAL_ERROR, NULL);
        self->invoke(iter);
    }

    Func func_;
};


////////////////////////////////////////////////////////////////////////////////
//// Utility class to invoke an entity observer delegate
////////////////////////////////////////////////////////////////////////////////

template <typename Func>
struct entity_observer_delegate : delegate {
    explicit entity_observer_delegate(Func&& func) noexcept 
        : func_(FLECS_MOV(func)) { }

    // Static function that can be used as callback for systems/triggers
    static void run(ecs_iter_t *iter) {
        invoke<Func>(iter);
    }

private:
    template <typename F,
        decltype(std::declval<const F&>()(std::declval<flecs::entity>()), 0) = 0>
    static void invoke(ecs_iter_t *iter) {
        auto self = static_cast<const entity_observer_delegate*>(iter->callback_ctx);
        ecs_assert(self != nullptr, ECS_INTERNAL_ERROR, NULL);
        self->func_(flecs::entity(iter->world, ecs_field_src(iter, 0)));
    }

    template <typename F,
        decltype(std::declval<const F&>()(), 0) = 0>
    static void invoke(ecs_iter_t *iter) {
        auto self = static_cast<const entity_observer_delegate*>(iter->callback_ctx);
        ecs_assert(self != nullptr, ECS_INTERNAL_ERROR, NULL);
        self->func_();
    }

    Func func_;
};

template <typename Func, typename Event>
struct entity_payload_observer_delegate : delegate {
    explicit entity_payload_observer_delegate(Func&& func) noexcept 
        : func_(FLECS_MOV(func)) { }

    // Static function that can be used as callback for systems/triggers
    static void run(ecs_iter_t *iter) {
        invoke<Func>(iter);
    }

private:
    template <typename F,
        decltype(std::declval<const F&>()(
            std::declval<Event&>()), 0) = 0>
    static void invoke(ecs_iter_t *iter) {
        auto self = static_cast<const entity_payload_observer_delegate*>(
            iter->callback_ctx);
        ecs_assert(self != nullptr, ECS_INTERNAL_ERROR, NULL);
        ecs_assert(iter->param != nullptr, ECS_INVALID_OPERATION, 
            "entity observer invoked without payload");

        Event *data = static_cast<Event*>(iter->param);
        self->func_(*data);
    }

    template <typename F,
        decltype(std::declval<const F&>()(
            std::declval<flecs::entity>(),
            std::declval<Event&>()), 0) = 0>
    static void invoke(ecs_iter_t *iter) {
        auto self = static_cast<const entity_payload_observer_delegate*>(
            iter->callback_ctx);
        ecs_assert(self != nullptr, ECS_INTERNAL_ERROR, NULL);
        ecs_assert(iter->param != nullptr, ECS_INVALID_OPERATION, 
            "entity observer invoked without payload");

        Event *data = static_cast<Event*>(iter->param);
        self->func_(flecs::entity(iter->world, ecs_field_src(iter, 0)), *data);
    }

    Func func_;
};


////////////////////////////////////////////////////////////////////////////////
//// Utility to invoke callback on entity if it has components in signature
////////////////////////////////////////////////////////////////////////////////

template<typename ... Args>
struct entity_with_delegate_impl;

template<typename ... Args>
struct entity_with_delegate_impl<arg_list<Args ...>> {
    using ColumnArray = flecs::array<int32_t, sizeof...(Args)>;
    using ArrayType = flecs::array<void*, sizeof...(Args)>;
    using DummyArray = flecs::array<int, sizeof...(Args)>;
    using IdArray = flecs::array<id_t, sizeof...(Args)>;

    static bool const_args() {
        static flecs::array<bool, sizeof...(Args)> is_const_args ({
            flecs::is_const<flecs::remove_reference_t<Args>>::value...
        });

        for (auto is_const : is_const_args) {
            if (!is_const) {
                return false;
            }
        }
        return true;
    }

    static 
    bool get_ptrs(world_t *world, flecs::entity_t e, const ecs_record_t *r, ecs_table_t *table,
        ArrayType& ptrs) 
    {
        ecs_assert(table != NULL, ECS_INTERNAL_ERROR, NULL);
        if (!ecs_table_column_count(table) && 
            !ecs_table_has_flags(table, EcsTableHasSparse)) 
        {
            return false;
        }

        /* table_index_of needs real world */
        const flecs::world_t *real_world = ecs_get_world(world);

        IdArray ids ({
            _::type<Args>().id(world)...
        });

        /* Get column indices for components */
        ColumnArray columns ({
            ecs_table_get_column_index(real_world, table, 
                _::type<Args>().id(world))...
        });

        /* Get pointers for columns for entity */
        size_t i = 0;
        for (int32_t column : columns) {
            if (column == -1) {
                /* Component could be sparse */
                void *ptr = ecs_get_mut_id(world, e, ids[i]);
                if (!ptr) {
                    return false;
                }

                ptrs[i ++] = ptr;
                continue;
            }

            ptrs[i ++] = ecs_record_get_by_column(r, column, 0);
        }

        return true;
    }

    static bool ensure_ptrs(world_t *world, ecs_entity_t e, ArrayType& ptrs) {
        /* Get pointers w/ensure */
        size_t i = 0;
        DummyArray dummy ({
            (ptrs[i ++] = ecs_ensure_id(world, e, 
                _::type<Args>().id(world)), 0)...
        });

        return true;
    }    

    template <typename Func>
    static bool invoke_read(world_t *world, entity_t e, const Func& func) {
        const ecs_record_t *r = ecs_read_begin(world, e);
        if (!r) {
            return false;
        }

        ecs_table_t *table = r->table;
        if (!table) {
            return false;
        }

        ArrayType ptrs;
        bool has_components = get_ptrs(world, e, r, table, ptrs);
        if (has_components) {
            invoke_callback(func, 0, ptrs);
        }

        ecs_read_end(r);

        return has_components;
    }

    template <typename Func>
    static bool invoke_write(world_t *world, entity_t e, const Func& func) {
        ecs_record_t *r = ecs_write_begin(world, e);
        if (!r) {
            return false;
        }

        ecs_table_t *table = r->table;
        if (!table) {
            return false;
        }

        ArrayType ptrs;
        bool has_components = get_ptrs(world, e, r, table, ptrs);
        if (has_components) {
            invoke_callback(func, 0, ptrs);
        }

        ecs_write_end(r);

        return has_components;
    }

    template <typename Func>
    static bool invoke_get(world_t *world, entity_t e, const Func& func) {
        if (const_args()) {
            return invoke_read(world, e, func);
        } else {
            return invoke_write(world, e, func);
        }
    }

    // Utility for storing id in array in pack expansion
    static size_t store_added(IdArray& added, size_t elem, ecs_table_t *prev, 
        ecs_table_t *next, id_t id) 
    {
        // Array should only contain ids for components that are actually added,
        // so check if the prev and next tables are different.
        if (prev != next) {
            added[elem] = id;
            elem ++;
        }
        return elem;
    }

    template <typename Func>
    static bool invoke_ensure(world_t *world, entity_t id, const Func& func) {
        flecs::world w(world);

        ArrayType ptrs;
        ecs_table_t *table = NULL;

        // When not deferred take the fast path.
        if (!w.is_deferred()) {
            // Bit of low level code so we only do at most one table move & one
            // entity lookup for the entire operation.

            // Make sure the object is not a stage. Operations on a stage are
            // only allowed when the stage is in deferred mode, which is when
            // the world is in readonly mode.
            ecs_assert(!w.is_stage(), ECS_INVALID_PARAMETER, NULL);

            // Find table for entity
            ecs_record_t *r = ecs_record_find(world, id);
            if (r) {
                table = r->table;
            }

            // Find destination table that has all components
            ecs_table_t *prev = table, *next;
            size_t elem = 0;
            IdArray added;

            // Iterate components, only store added component ids in added array
            DummyArray dummy_before ({ (
                next = ecs_table_add_id(world, prev, w.id<Args>()),
                elem = store_added(added, elem, prev, next, w.id<Args>()),
                prev = next, 0
            )... });

            (void)dummy_before;

            // If table is different, move entity straight to it
            if (table != next) {
                ecs_type_t ids;
                ids.array = added.ptr();
                ids.count = static_cast<ecs_size_t>(elem);
                ecs_commit(world, id, r, next, &ids, NULL);
                table = next;
            }

            if (!get_ptrs(w, id, r, table, ptrs)) {
                ecs_abort(ECS_INTERNAL_ERROR, NULL);
            }

            ECS_TABLE_LOCK(world, table);

        // When deferred, obtain pointers with regular ensure
        } else {
            ensure_ptrs(world, id, ptrs);
        }

        invoke_callback(func, 0, ptrs);

        if (!w.is_deferred()) {
            ECS_TABLE_UNLOCK(world, table);
        }

        // Call modified on each component
        DummyArray dummy_after ({
            ( ecs_modified_id(world, id, w.id<Args>()), 0)...
        });
        (void)dummy_after;

        return true;
    }    

private:
    template <typename Func, typename ... TArgs, 
        if_t<sizeof...(TArgs) == sizeof...(Args)> = 0>
    static void invoke_callback(
        const Func& f, size_t, ArrayType&, TArgs&& ... comps) 
    {
        f(*static_cast<typename base_arg_type<Args>::type*>(comps)...);
    }

    template <typename Func, typename ... TArgs, 
        if_t<sizeof...(TArgs) != sizeof...(Args)> = 0>
    static void invoke_callback(const Func& f, size_t arg, ArrayType& ptrs, 
        TArgs&& ... comps) 
    {
        invoke_callback(f, arg + 1, ptrs, comps..., ptrs[arg]);
    }
};

template <typename Func, typename U = int>
struct entity_with_delegate {
    static_assert(function_traits<Func>::value, "type is not callable");
};

template <typename Func>
struct entity_with_delegate<Func, if_t< is_callable<Func>::value > >
    : entity_with_delegate_impl< arg_list_t<Func> >
{
    static_assert(function_traits<Func>::arity > 0,
        "function must have at least one argument");
};

} // namespace _

// Experimental: allows using the each delegate for use cases outside of flecs
template <typename Func, typename ... Args>
using delegate = _::each_delegate<typename std::decay<Func>::type, Args...>;

} // namespace flecs

/**
 * @file addons/cpp/component.hpp
 * @brief Registering/obtaining info from components.
 */

#pragma once

#include <ctype.h>
#include <stdio.h>

/**
 * @defgroup cpp_components Components
 * @ingroup cpp_core
 * Registering and working with components.
 *
 * @{
 */

namespace flecs {

namespace _ {

// Trick to obtain typename from type, as described here
// https://blog.molecular-matters.com/2015/12/11/getting-the-type-of-a-template-argument-as-string-without-rtti/
//
// The code from the link has been modified to work with more types, and across
// multiple compilers. The resulting string should be the same on all platforms
// for all compilers.
//

#if defined(__GNUC__) || defined(_WIN32)
template <typename T>
inline const char* type_name() {
    static const size_t len = ECS_FUNC_TYPE_LEN(const char*, type_name, ECS_FUNC_NAME);
    static char result[len + 1] = {};
    static const size_t front_len = ECS_FUNC_NAME_FRONT(const char*, type_name);
    static const char* cppTypeName = ecs_cpp_get_type_name(result, ECS_FUNC_NAME, len, front_len);
    return cppTypeName;
}
#else
#error "implicit component registration not supported"
#endif

// Translate a typename into a language-agnostic identifier. This allows for
// registration of components/modules across language boundaries.
template <typename T>
inline const char* symbol_name() {
    static const size_t len = ECS_FUNC_TYPE_LEN(const char*, symbol_name, ECS_FUNC_NAME);
    static char result[len + 1] = {};
    static const char* cppSymbolName = ecs_cpp_get_symbol_name(result, type_name<T>(), len);
    return cppSymbolName;
}

template <> inline const char* symbol_name<uint8_t>() {
    return "u8";
}
template <> inline const char* symbol_name<uint16_t>() {
    return "u16";
}
template <> inline const char* symbol_name<uint32_t>() {
    return "u32";
}
template <> inline const char* symbol_name<uint64_t>() {
    return "u64";
}
template <> inline const char* symbol_name<int8_t>() {
    return "i8";
}
template <> inline const char* symbol_name<int16_t>() {
    return "i16";
}
template <> inline const char* symbol_name<int32_t>() {
    return "i32";
}
template <> inline const char* symbol_name<int64_t>() {
    return "i64";
}
template <> inline const char* symbol_name<float>() {
    return "f32";
}
template <> inline const char* symbol_name<double>() {
    return "f64";
}

// If type is trivial, don't register lifecycle actions. While the functions
// that obtain the lifecycle callback do detect whether the callback is required
// adding a special case for trivial types eases the burden a bit on the
// compiler as it reduces the number of templates to evaluate.
template<typename T, enable_if_t<
    std::is_trivial<T>::value == true
        >* = nullptr>
void register_lifecycle_actions(ecs_world_t*, ecs_entity_t) { }

// If the component is non-trivial, register component lifecycle actions.
// Depending on the type not all callbacks may be available.
template<typename T, enable_if_t<
    std::is_trivial<T>::value == false
        >* = nullptr>
void register_lifecycle_actions(
    ecs_world_t *world,
    ecs_entity_t component)
{
    ecs_type_hooks_t cl{};
    cl.ctor = ctor<T>(cl.flags);
    cl.dtor = dtor<T>(cl.flags);

    cl.copy = copy<T>(cl.flags);
    cl.copy_ctor = copy_ctor<T>(cl.flags);
    cl.move = move<T>(cl.flags);
    cl.move_ctor = move_ctor<T>(cl.flags);

    cl.ctor_move_dtor = ctor_move_dtor<T>(cl.flags);
    cl.move_dtor = move_dtor<T>(cl.flags);

    ecs_set_hooks_id(world, component, &cl);

    if (cl.flags & (ECS_TYPE_HOOK_MOVE_ILLEGAL|ECS_TYPE_HOOK_MOVE_CTOR_ILLEGAL))
    {
        ecs_add_id(world, component, flecs::Sparse);
    }
}

template <typename T>
struct type_impl {
    static_assert(is_pointer<T>::value == false,
        "pointer types are not allowed for components");

    // Initialize component identifier
    static void init(
        bool allow_tag = true)
    {
        s_index = flecs_component_ids_index_get();
        s_allow_tag = allow_tag;
        s_size = sizeof(T);
        s_alignment = alignof(T);
        if (is_empty<T>::value && allow_tag) {
            s_size = 0;
            s_alignment = 0;
        }
    }

    static void init_builtin(
        flecs::world_t *world,
        flecs::entity_t id,
        bool allow_tag = true)
    {
        init(allow_tag);
        flecs_component_ids_set(world, s_index, id);
    }

    // Register component id.
    static entity_t register_id(world_t *world,
        const char *name = nullptr, bool allow_tag = true, flecs::id_t id = 0,
        bool is_component = false, bool implicit_name = true, const char *n = nullptr, 
        flecs::entity_t module = 0)
    {
        if (!s_index) {
            // This is the first time (in this binary image) that this type is
            // being used. Generate a static index that will identify the type
            // across worlds.
            init(allow_tag);
            ecs_assert(s_index != 0, ECS_INTERNAL_ERROR, NULL);
        }

        flecs::entity_t c = flecs_component_ids_get(world, s_index);

        if (!c || !ecs_is_alive(world, c)) {
            // When a component is implicitly registered, ensure that it's not
            // registered in the current scope of the application/that "with"
            // components get added to the component entity.
            ecs_entity_t prev_scope = ecs_set_scope(world, module);
            ecs_entity_t prev_with = ecs_set_with(world, 0);

            // At this point it is possible that the type was already registered
            // with the world, just not for this binary. The registration code
            // uses the type symbol to check if it was already registered. Note
            // that the symbol is separate from the typename, as an application
            // can override a component name when registering a type.
            bool existing = false;
            c = ecs_cpp_component_find(
                world, id, n, symbol_name<T>(), size(), alignment(), 
                implicit_name, &existing);

            const char *symbol = nullptr;
            if (c) {
                symbol = ecs_get_symbol(world, c);
            }
            if (!symbol) {
                symbol = symbol_name<T>();
            }

            c = ecs_cpp_component_register(world, c, c, name, type_name<T>(), 
                symbol, size(), alignment(), is_component, &existing);

            ecs_set_with(world, prev_with);
            ecs_set_scope(world, prev_scope);

            // Register lifecycle callbacks, but only if the component has a
            // size. Components that don't have a size are tags, and tags don't
            // require construction/destruction/copy/move's.
            if (size() && !existing) {
                register_lifecycle_actions<T>(world, c);
            }

            // Set world local component id
            flecs_component_ids_set(world, s_index, c);

            // If component is enum type, register constants. Make sure to do 
            // this after setting the component id, because the enum code will
            // be calling type<T>::id().
            #if FLECS_CPP_ENUM_REFLECTION_SUPPORT
            _::init_enum<T>(world, c);
            #endif
        }

        ecs_assert(c != 0, ECS_INTERNAL_ERROR, NULL);

        return c;
    }

    // Get type (component) id.
    // If type was not yet registered and automatic registration is allowed,
    // this function will also register the type.
    static entity_t id(world_t *world)
    {
#ifdef FLECS_CPP_NO_AUTO_REGISTRATION
        ecs_assert(registered(world), ECS_INVALID_OPERATION, 
            "component '%s' must be registered before use",
            type_name<T>());

        flecs::entity_t c = flecs_component_ids_get(world, s_index);
        ecs_assert(c != 0, ECS_INTERNAL_ERROR, NULL);
        ecs_assert(ecs_is_alive(world, c), ECS_INVALID_OPERATION,
            "component '%s' was deleted, reregister before using",
            type_name<T>());
#else
        flecs::entity_t c = flecs_component_ids_get_alive(world, s_index);
        if (!c) {
            c = register_id(world);
        }
#endif
        return c;
    }

    // Return the size of a component.
    static size_t size() {
        ecs_assert(s_index != 0, ECS_INTERNAL_ERROR, NULL);
        return s_size;
    }

    // Return the alignment of a component.
    static size_t alignment() {
        ecs_assert(s_index != 0, ECS_INTERNAL_ERROR, NULL);
        return s_alignment;
    }

    // Was the component already registered.
    static bool registered(flecs::world_t *world) {
        ecs_assert(world != nullptr, ECS_INVALID_PARAMETER, NULL);

        if (s_index == 0) {
            return false;
        }

        if (!flecs_component_ids_get(world, s_index)) {
            return false;
        }

        return true;
    }

    // This function is only used to test cross-translation unit features. No
    // code other than test cases should invoke this function.
    static void reset() {
        s_index = 0;
        s_size = 0;
        s_alignment = 0;
        s_allow_tag = true;
    }

    static int32_t s_index;
    static size_t s_size;
    static size_t s_alignment;
    static bool s_allow_tag;
};

// Global templated variables that hold component identifier and other info
template <typename T> int32_t  type_impl<T>::s_index;
template <typename T> size_t   type_impl<T>::s_size;
template <typename T> size_t   type_impl<T>::s_alignment;
template <typename T> bool     type_impl<T>::s_allow_tag( true );

// Front facing class for implicitly registering a component & obtaining
// static component data

// Regular type
template <typename T>
struct type<T, if_not_t< is_pair<T>::value >>
    : type_impl<base_type_t<T>> { };

// Pair type
template <typename T>
struct type<T, if_t< is_pair<T>::value >>
{
    // Override id method to return id of pair
    static id_t id(world_t *world = nullptr) {
        return ecs_pair(
            type< pair_first_t<T> >::id(world),
            type< pair_second_t<T> >::id(world));
    }
};

} // namespace _

/** Untyped component class.
 * Generic base class for flecs::component.
 *
 * @ingroup cpp_components
 */
struct untyped_component : entity {
    using entity::entity;

#   ifdef FLECS_META
/**
 * @file addons/cpp/mixins/meta/untyped_component.inl
 * @brief Meta component mixin.
 */

/**
 * @memberof flecs::component
 * @ingroup cpp_addons_meta
 * 
 * @{
 */

private:

/** Private method that adds member to component. */
untyped_component& internal_member(
    flecs::entity_t type_id, 
    flecs::entity_t unit, 
    const char *name, 
    int32_t count = 0, 
    size_t offset = 0, 
    bool use_offset = false) 
{
    ecs_entity_desc_t desc = {};
    desc.name = name;
    desc.parent = id_;
    ecs_entity_t eid = ecs_entity_init(world_, &desc);
    ecs_assert(eid != 0, ECS_INTERNAL_ERROR, NULL);

    flecs::entity e(world_, eid);

    Member m = {};
    m.type = type_id;
    m.unit = unit;
    m.count = count;
    m.offset = static_cast<int32_t>(offset);
    m.use_offset = use_offset;
    e.set<Member>(m);

    return *this;
}

public: 

/** Add member with unit. */
untyped_component& member(
    flecs::entity_t type_id, 
    flecs::entity_t unit, 
    const char *name, 
    int32_t count = 0) 
{
    return internal_member(type_id, unit, name, count, 0, false);
}

/** Add member with unit, count and offset. */
untyped_component& member(
    flecs::entity_t type_id, 
    flecs::entity_t unit, 
    const char *name, 
    int32_t count, 
    size_t offset) 
{
    return internal_member(type_id, unit, name, count, offset, true);
}

/** Add member. */
untyped_component& member(
    flecs::entity_t type_id, 
    const char* name,
    int32_t count = 0) 
{
    return member(type_id, 0, name, count);
}

/** Add member with count and offset. */
untyped_component& member(
    flecs::entity_t type_id, 
    const char* name, 
    int32_t count, 
    size_t offset) 
{
    return member(type_id, 0, name, count, offset);
}

/** Add member. */
template <typename MemberType>
untyped_component& member(
    const char *name,
    int32_t count = 0) 
{
    flecs::entity_t type_id = _::type<MemberType>::id(world_);
    return member(type_id, name, count);
}

/** Add member. */
template <typename MemberType>
untyped_component& member(
    const char *name,
    int32_t count, 
    size_t offset) 
{
    flecs::entity_t type_id = _::type<MemberType>::id(world_);
    return member(type_id, name, count, offset);
}

/** Add member with unit. */
template <typename MemberType>
untyped_component& member(
    flecs::entity_t unit,
    const char *name, 
    int32_t count = 0) 
{
    flecs::entity_t type_id = _::type<MemberType>::id(world_);
    return member(type_id, unit, name, count);
}

/** Add member with unit. */
template <typename MemberType>
untyped_component& member(
    flecs::entity_t unit,
    const char *name, 
    int32_t count, 
    size_t offset) 
{
    flecs::entity_t type_id = _::type<MemberType>::id(world_);
    return member(type_id, unit, name, count, offset);
}

/** Add member with unit. */
template <typename MemberType, typename UnitType>
untyped_component& member(
    const char *name,
    int32_t count = 0) 
{
    flecs::entity_t type_id = _::type<MemberType>::id(world_);
    flecs::entity_t unit_id = _::type<UnitType>::id(world_);
    return member(type_id, unit_id, name, count);
}

/** Add member with unit. */
template <typename MemberType, typename UnitType>
untyped_component& member(
    const char *name, 
    int32_t count, 
    size_t offset) 
{
    flecs::entity_t type_id = _::type<MemberType>::id(world_);
    flecs::entity_t unit_id = _::type<UnitType>::id(world_);
    return member(type_id, unit_id, name, count, offset);
}

/** Add member using pointer-to-member. */
template <typename MemberType, typename ComponentType, 
    typename RealType = typename std::remove_extent<MemberType>::type>
untyped_component& member(
    const char* name, 
    const MemberType ComponentType::* ptr) 
{
    flecs::entity_t type_id = _::type<RealType>::id(world_);
    size_t offset = reinterpret_cast<size_t>(&(static_cast<ComponentType*>(nullptr)->*ptr));
    return member(type_id, name, std::extent<MemberType>::value, offset);
}

/** Add member with unit using pointer-to-member. */
template <typename MemberType, typename ComponentType, 
    typename RealType = typename std::remove_extent<MemberType>::type>
untyped_component& member(
    flecs::entity_t unit, 
    const char* name, 
    const MemberType ComponentType::* ptr) 
{
    flecs::entity_t type_id = _::type<RealType>::id(world_);
    size_t offset = reinterpret_cast<size_t>(&(static_cast<ComponentType*>(nullptr)->*ptr));
    return member(type_id, unit, name, std::extent<MemberType>::value, offset);
}

/** Add member with unit using pointer-to-member. */
template <typename UnitType, typename MemberType, typename ComponentType, 
    typename RealType = typename std::remove_extent<MemberType>::type>
untyped_component& member(
    const char* name, 
    const MemberType ComponentType::* ptr) 
{
    flecs::entity_t type_id = _::type<RealType>::id(world_);
    flecs::entity_t unit_id = _::type<UnitType>::id(world_);
    size_t offset = reinterpret_cast<size_t>(&(static_cast<ComponentType*>(nullptr)->*ptr));
    return member(type_id, unit_id, name, std::extent<MemberType>::value, offset);
}

/** Add constant. */
untyped_component& constant(
    const char *name,
    int32_t value) 
{
    ecs_add_id(world_, id_, _::type<flecs::Enum>::id(world_));

    ecs_entity_desc_t desc = {};
    desc.name = name;
    desc.parent = id_;
    ecs_entity_t eid = ecs_entity_init(world_, &desc);
    ecs_assert(eid != 0, ECS_INTERNAL_ERROR, NULL);

    ecs_set_id(world_, eid, 
        ecs_pair(flecs::Constant, flecs::I32), sizeof(int32_t),
        &value);

    return *this;
}

/** Add bitmask constant. */
untyped_component& bit(
    const char *name, 
    uint32_t value) 
{
    ecs_add_id(world_, id_, _::type<flecs::Bitmask>::id(world_));

    ecs_entity_desc_t desc = {};
    desc.name = name;
    desc.parent = id_;
    ecs_entity_t eid = ecs_entity_init(world_, &desc);
    ecs_assert(eid != 0, ECS_INTERNAL_ERROR, NULL);

    ecs_set_id(world_, eid, 
        ecs_pair(flecs::Constant, flecs::U32), sizeof(uint32_t),
        &value);

    return *this;
}

/** Register array metadata for component */
template <typename Elem>
untyped_component& array(
    int32_t elem_count) 
{
    ecs_array_desc_t desc = {};
    desc.entity = id_;
    desc.type = _::type<Elem>::id(world_);
    desc.count = elem_count;
    ecs_array_init(world_, &desc);
    return *this;
}

/** Add member value range */
untyped_component& range(
    double min,
    double max) 
{
    const flecs::member_t *m = ecs_cpp_last_member(world_, id_);
    if (!m) {
        return *this;
    }

    flecs::world w(world_);
    flecs::entity me = w.entity(m->member);

    // Don't use C++ ensure because Unreal defines a macro called ensure
    flecs::MemberRanges *mr = static_cast<flecs::MemberRanges*>(
        ecs_ensure_id(w, me, w.id<flecs::MemberRanges>()));
    mr->value.min = min;
    mr->value.max = max;
    me.modified<flecs::MemberRanges>();
    return *this;
}

/** Add member warning range */
untyped_component& warning_range(
    double min,
    double max) 
{
    const flecs::member_t *m = ecs_cpp_last_member(world_, id_);
    if (!m) {
        return *this;
    }

    flecs::world w(world_);
    flecs::entity me = w.entity(m->member);

    // Don't use C++ ensure because Unreal defines a macro called ensure
    flecs::MemberRanges *mr = static_cast<flecs::MemberRanges*>(
        ecs_ensure_id(w, me, w.id<flecs::MemberRanges>()));
    mr->warning.min = min;
    mr->warning.max = max;
    me.modified<flecs::MemberRanges>();
    return *this;
}

/** Add member error range */
untyped_component& error_range(
    double min,
    double max) 
{
    const flecs::member_t *m = ecs_cpp_last_member(world_, id_);
    if (!m) {
        return *this;
    }

    flecs::world w(world_);
    flecs::entity me = w.entity(m->member);

    // Don't use C++ ensure because Unreal defines a macro called ensure
    flecs::MemberRanges *mr = static_cast<flecs::MemberRanges*>(ecs_ensure_id(
        w, me, w.id<flecs::MemberRanges>()));
    mr->error.min = min;
    mr->error.max = max;
    me.modified<flecs::MemberRanges>();
    return *this;
}

/** @} */

#   endif
#   ifdef FLECS_METRICS
/**
 * @file addons/cpp/mixins/meta/untyped_component.inl
 * @brief Metrics component mixin.
 */

/**
 * @memberof flecs::component
 * @ingroup cpp_addons_metrics
 * 
 * @{
 */

/** Register member as metric.
 * When no explicit name is provided, this operation will derive the metric name
 * from the member name. When the member name is "value", the operation will use
 * the name of the component.
 * 
 * When the brief parameter is provided, it is set on the metric as if 
 * set_doc_brief is used. The brief description can be obtained with 
 * get_doc_brief.
 * 
 * @tparam Kind Metric kind (Counter, CounterIncrement or Gauge).
 * @param parent Parent entity of the metric (optional).
 * @param brief Description for metric (optional).
 * @param name Name of metric (optional).
 */
template <typename Kind>
untyped_component& metric(
    flecs::entity_t parent = 0, 
    const char *brief = nullptr, 
    const char *name = nullptr);

/** @} */

#   endif
};

/** Component class.
 * Class used to register components and component metadata.
 *
 * @ingroup cpp_components
 */
template <typename T>
struct component : untyped_component {
    /** Register a component.
     * If the component was already registered, this operation will return a handle
     * to the existing component.
     *
     * @param world The world for which to register the component.
     * @param name Optional name (overrides typename).
     * @param allow_tag If true, empty types will be registered with size 0.
     * @param id Optional id to register component with.
     */
    component(
        flecs::world_t *world,
        const char *name = nullptr,
        bool allow_tag = true,
        flecs::id_t id = 0)
    {
        const char *n = name;
        bool implicit_name = false;
        if (!n) {
            n = _::type_name<T>();

            // Keep track of whether name was explicitly set. If not, and the
            // component was already registered, just use the registered name.
            // The registered name may differ from the typename as the registered
            // name includes the flecs scope. This can in theory be different from
            // the C++ namespace though it is good practice to keep them the same */
            implicit_name = true;
        }

        // If component is registered by module, ensure it's registered in the
        // scope of the module.
        flecs::entity_t module = ecs_get_scope(world);

        // Strip off the namespace part of the component name, unless a name was
        // explicitly provided by the application.
        if (module && implicit_name) {
            // If the type is a template type, make sure to ignore
            // inside the template parameter list.
            const char *start = strchr(n, '<'), *last_elem = NULL;
            if (start) {
                const char *ptr = start;
                while (ptr[0] && (ptr[0] != ':') && (ptr > n)) {
                    ptr --;
                }
                if (ptr[0] == ':') {
                    last_elem = ptr;
                }
            }

            if (last_elem) {
                name = last_elem + 1;
            }
        }

        world_ = world;
        id_ = _::type<T>::register_id(
            world, name, allow_tag, id, true, implicit_name, n, module);
    }

    /** Register on_add hook. */
    template <typename Func>
    component<T>& on_add(Func&& func) {
        using Delegate = typename _::each_delegate<typename std::decay<Func>::type, T>;
        flecs::type_hooks_t h = get_hooks();
        ecs_assert(h.on_add == nullptr, ECS_INVALID_OPERATION,
            "on_add hook is already set");
        BindingCtx *ctx = get_binding_ctx(h);
        h.on_add = Delegate::run_add;
        ctx->on_add = FLECS_NEW(Delegate)(FLECS_FWD(func));
        ctx->free_on_add = _::free_obj<Delegate>;
        ecs_set_hooks_id(world_, id_, &h);
        return *this;
    }

    /** Register on_remove hook. */
    template <typename Func>
    component<T>& on_remove(Func&& func) {
        using Delegate = typename _::each_delegate<
            typename std::decay<Func>::type, T>;
        flecs::type_hooks_t h = get_hooks();
        ecs_assert(h.on_remove == nullptr, ECS_INVALID_OPERATION,
            "on_remove hook is already set");
        BindingCtx *ctx = get_binding_ctx(h);
        h.on_remove = Delegate::run_remove;
        ctx->on_remove = FLECS_NEW(Delegate)(FLECS_FWD(func));
        ctx->free_on_remove = _::free_obj<Delegate>;
        ecs_set_hooks_id(world_, id_, &h);
        return *this;
    }

    /** Register on_set hook. */
    template <typename Func>
    component<T>& on_set(Func&& func) {
        using Delegate = typename _::each_delegate<
            typename std::decay<Func>::type, T>;
        flecs::type_hooks_t h = get_hooks();
        ecs_assert(h.on_set == nullptr, ECS_INVALID_OPERATION,
            "on_set hook is already set");
        BindingCtx *ctx = get_binding_ctx(h);
        h.on_set = Delegate::run_set;
        ctx->on_set = FLECS_NEW(Delegate)(FLECS_FWD(func));
        ctx->free_on_set = _::free_obj<Delegate>;
        ecs_set_hooks_id(world_, id_, &h);
        return *this;
    }

#   ifdef FLECS_META

/** Register opaque type interface */
template <typename Func>
component& opaque(const Func& type_support) {
    flecs::world world(world_);
    auto ts = type_support(world);
    ts.desc.entity = _::type<T>::id(world_);
    ecs_opaque_init(world_, &ts.desc);
    return *this;
}

flecs::opaque<T> opaque(flecs::entity_t as_type) {
    return flecs::opaque<T>(world_).as_type(as_type);
}

flecs::opaque<T> opaque(flecs::entity as_type) {
    return this->opaque(as_type.id());
}

flecs::opaque<T> opaque(flecs::untyped_component as_type) {
    return this->opaque(as_type.id());
}

/** Return opaque type builder for collection type */
template <typename ElemType>
flecs::opaque<T, ElemType> opaque(flecs::id_t as_type) {
    return flecs::opaque<T, ElemType>(world_).as_type(as_type);
}

/** Add constant. */
component<T>& constant(const char *name, T value) {
    using U = typename std::underlying_type<T>::type;

    ecs_add_id(world_, id_, _::type<flecs::Enum>::id(world_));

    ecs_entity_desc_t desc = {};
    desc.name = name;
    desc.parent = id_;
    ecs_entity_t eid = ecs_entity_init(world_, &desc);
    ecs_assert(eid != 0, ECS_INTERNAL_ERROR, NULL);

    flecs::id_t pair = ecs_pair(flecs::Constant, _::type<U>::id(world_));
    U *ptr = static_cast<U*>(ecs_ensure_id(world_, eid, pair));
    *ptr = static_cast<U>(value);
    ecs_modified_id(world_, eid, pair);

    return *this;
}

#   endif

private:
    using BindingCtx = _::component_binding_ctx;

    BindingCtx* get_binding_ctx(flecs::type_hooks_t& h){
        BindingCtx *result = static_cast<BindingCtx*>(h.binding_ctx);
        if (!result) {
            result = FLECS_NEW(BindingCtx);
            h.binding_ctx = result;
            h.binding_ctx_free = _::free_obj<BindingCtx>;
        }
        return result;
    }

    flecs::type_hooks_t get_hooks() {
        const flecs::type_hooks_t* h = ecs_get_hooks_id(world_, id_);
        if (h) {
            return *h;
        } else {
            return {};
        }
    }
};

}

/** @} */

/**
 * @file addons/cpp/ref.hpp
 * @brief Class that caches data to speedup get operations.
 */

#pragma once

namespace flecs
{

/**
 * @defgroup cpp_ref Refs
 * @ingroup cpp_core
 * Refs are a fast mechanism for referring to a specific entity/component.
 *
 * @{
 */

/** Component reference.
 * Reference to a component from a specific entity.
 */
template <typename T>
struct ref {
    ref() : world_(nullptr), ref_{} { }

    ref(world_t *world, entity_t entity, flecs::id_t id = 0)
        : ref_()
    {
        // the world we were called with may be a stage; convert it to a world
        // here if that is the case
        world_ = world ? const_cast<flecs::world_t *>(ecs_get_world(world))
            : nullptr;
        if (!id) {
            id = _::type<T>::id(world);
        }

        ecs_assert(_::type<T>::size() != 0, ECS_INVALID_PARAMETER,
            "operation invalid for empty type");

        ref_ = ecs_ref_init_id(world_, entity, id);
    }

    ref(flecs::entity entity, flecs::id_t id = 0)
        : ref(entity.world(), entity.id(), id) { }

    T* operator->() {
        T* result = static_cast<T*>(ecs_ref_get_id(
            world_, &ref_, this->ref_.id));

        ecs_assert(result != NULL, ECS_INVALID_PARAMETER,
            "nullptr dereference by flecs::ref");

        return result;
    }

    T* get() {
        return static_cast<T*>(ecs_ref_get_id(
            world_, &ref_, this->ref_.id));
    }

    T* try_get() {
        if (!world_ || !ref_.entity) {
            return nullptr;
        }

        return get();
    }

    bool has() {
        return !!try_get();
    }

    /** implicit conversion to bool.  return true if there is a valid T* being referred to **/
    operator bool() {
        return has();
    }

    flecs::entity entity() const;

private:
    world_t *world_;
    flecs::ref_t ref_;
};

/** @} */

}

/**
 * @file addons/cpp/type.hpp
 * @brief Utility functions for id vector.
 */

#pragma once

namespace flecs {

/**
 * @defgroup cpp_types Types
 * @ingroup cpp_core
 * @brief Type operations.
 *
 * @{
 */

/** Type class.
 * A type is a vector of component ids which can be requested from entities or tables.
 */
struct type {
    type() : world_(nullptr), type_(nullptr) { }

    type(world_t *world, const type_t *t)
        : world_(world)
        , type_(t) { }

    /** Convert type to comma-separated string */
    flecs::string str() const {
        return flecs::string(ecs_type_str(world_, type_));
    }

    /** Return number of ids in type */
    int32_t count() const {
        if (!type_) {
            return 0;
        }
        return type_->count;
    }

    /** Return pointer to array. */
    flecs::id_t* array() const {
        if (!type_) {
            return nullptr;
        }
        return type_->array;
    }

    /** Get id at specified index in type */
    flecs::id get(int32_t index) const {
        ecs_assert(type_ != NULL, ECS_INVALID_PARAMETER, NULL);
        ecs_assert(type_->count > index, ECS_OUT_OF_RANGE, NULL);
        if (!type_) {
            return flecs::id();
        }
        return flecs::id(world_, type_->array[index]);
    }

    const flecs::id_t* begin() const {
        if (type_ && type_->count) {
            return type_->array;
        } else {
            return &empty_;
        }
    }

    const flecs::id_t* end() const {
        if (type_ && type_->count) {
            return &type_->array[type_->count];
        } else {
            return &empty_;
        }
    }

    /** Implicit conversion to type_t */
    operator const type_t*() const {
        return type_;
    }
private:
    world_t *world_;
    const type_t *type_;
    flecs::id_t empty_;
};

/** #} */

}

/**
 * @file addons/cpp/table.hpp
 * @brief Direct access to table data.
 */

#pragma once

namespace flecs {

/**
 * @defgroup cpp_tables Tables
 * @ingroup cpp_core
 * Table operations.
 *
 * @{
 */

struct table {
    table() : world_(nullptr), table_(nullptr) { }

    table(world_t *world, table_t *t)
        : world_(world)
        , table_(t) { }

    virtual ~table() { }

    /** Convert table type to string. */
    flecs::string str() const {
        return flecs::string(ecs_table_str(world_, table_));
    }

    /** Get table type. */
    flecs::type type() const {
        return flecs::type(world_, ecs_table_get_type(table_));
    }

    /** Get table count. */
    int32_t count() const {
        return ecs_table_count(table_);
    }

    /** Find type index for (component) id.
     *
     * @param id The (component) id.
     * @return The index of the id in the table type, -1 if not found/
     */
    int32_t type_index(flecs::id_t id) const {
        return ecs_table_get_type_index(world_, table_, id);
    }

    /** Find type index for type.
     *
     * @tparam T The type.
     * @return True if the table has the type, false if not.
     */
    template <typename T>
    int32_t type_index() const {
        return type_index(_::type<T>::id(world_));
    }

    /** Find type index for pair.
     * @param first First element of pair.
     * @param second Second element of pair.
     * @return True if the table has the pair, false if not.
     */
    int32_t type_index(flecs::entity_t first, flecs::entity_t second) const {
        return type_index(ecs_pair(first, second));
    }

    /** Find type index for pair.
     * @tparam First First element of pair.
     * @param second Second element of pair.
     * @return True if the table has the pair, false if not.
     */
    template <typename First>
    int32_t type_index(flecs::entity_t second) const {
        return type_index(_::type<First>::id(world_), second);
    }

    /** Find type index for pair.
     * @tparam First First element of pair.
     * @tparam Second Second element of pair.
     * @return True if the table has the pair, false if not.
     */
    template <typename First, typename Second>
    int32_t type_index() const {
        return type_index<First>(_::type<Second>::id(world_));
    }

    /** Find column index for (component) id.
     *
     * @param id The (component) id.
     * @return The index of the id in the table type, -1 if not found/
     */
    int32_t column_index(flecs::id_t id) const {
        return ecs_table_get_column_index(world_, table_, id);
    }

    /** Find column index for type.
     *
     * @tparam T The type.
     * @return True if the table has the type, false if not.
     */
    template <typename T>
    int32_t column_index() const {
        return column_index(_::type<T>::id(world_));
    }

    /** Find column index for pair.
     * @param first First element of pair.
     * @param second Second element of pair.
     * @return True if the table has the pair, false if not.
     */
    int32_t column_index(flecs::entity_t first, flecs::entity_t second) const {
        return column_index(ecs_pair(first, second));
    }

    /** Find column index for pair.
     * @tparam First First element of pair.
     * @param second Second element of pair.
     * @return True if the table has the pair, false if not.
     */
    template <typename First>
    int32_t column_index(flecs::entity_t second) const {
        return column_index(_::type<First>::id(world_), second);
    }

    /** Find column index for pair.
     * @tparam First First element of pair.
     * @tparam Second Second element of pair.
     * @return True if the table has the pair, false if not.
     */
    template <typename First, typename Second>
    int32_t column_index() const {
        return column_index<First>(_::type<Second>::id(world_));
    }

    /** Test if table has (component) id.
     *
     * @param id The (component) id.
     * @return True if the table has the id, false if not.
     */
    bool has(flecs::id_t id) const {
        return type_index(id) != -1;
    }

    /** Test if table has the type.
     *
     * @tparam T The type.
     * @return True if the table has the type, false if not.
     */
    template <typename T>
    bool has() const {
        return type_index<T>() != -1;
    }

    /** Test if table has the pair.
     *
     * @param first First element of pair.
     * @param second Second element of pair.
     * @return True if the table has the pair, false if not.
     */
    bool has(flecs::entity_t first, flecs::entity_t second) const {
        return type_index(first, second) != -1;
    }

    /** Test if table has the pair.
     *
     * @tparam First First element of pair.
     * @param second Second element of pair.
     * @return True if the table has the pair, false if not.
     */
    template <typename First>
    bool has(flecs::entity_t second) const {
        return type_index<First>(second) != -1;
    }

    /** Test if table has the pair.
     *
     * @tparam First First element of pair.
     * @tparam Second Second element of pair.
     * @return True if the table has the pair, false if not.
     */
    template <typename First, typename Second>
    bool has() const {
        return type_index<First, Second>() != -1;
    }

    /** Get pointer to component array by column index.
     *
     * @param index The column index.
     * @return Pointer to the column, NULL if not a component.
     */
    virtual void* get_column(int32_t index) const {
        return ecs_table_get_column(table_, index, 0);
    }

    /** Get pointer to component array by component.
     *
     * @param id The component id.
     * @return Pointer to the column, NULL if not found.
     */
    void* get(flecs::id_t id) const {
        int32_t index = column_index(id);
        if (index == -1) {
            return NULL;
        }
        return get_column(index);
    }

    /** Get pointer to component array by pair.
     *
     * @param first The first element of the pair.
     * @param second The second element of the pair.
     * @return Pointer to the column, NULL if not found.
     */
    void* get(flecs::entity_t first, flecs::entity_t second) const {
        return get(ecs_pair(first, second));
    }

    /** Get pointer to component array by component.
     *
     * @tparam T The component.
     * @return Pointer to the column, NULL if not found.
     */
    template <typename T, if_t< is_actual<T>::value > = 0>
    T* get() const {
        return static_cast<T*>(get(_::type<T>::id(world_)));
    }

    /** Get pointer to component array by (enum) component.
     *
     * @tparam T The (enum) component.
     * @return Pointer to the column, NULL if not found.
     */
    template <typename T, if_t< is_enum<T>::value > = 0>
    T* get() const {
        return static_cast<T*>(get(_::type<T>::id(world_)));
    }

    /** Get pointer to component array by component.
     *
     * @tparam T The component.
     * @return Pointer to the column, NULL if not found.
     */
    template <typename T, typename A = actual_type_t<T>,
        if_t< flecs::is_pair<T>::value > = 0>
    A* get() const {
        return static_cast<A*>(get(_::type<T>::id(world_)));
    }

    /** Get pointer to component array by pair.
     *
     * @tparam First The first element of the pair.
     * @param second The second element of the pair.
     * @return Pointer to the column, NULL if not found.
     */
    template <typename First>
    First* get(flecs::entity_t second) const {
        return static_cast<First*>(get(_::type<First>::id(world_), second));
    }

    /** Get pointer to component array by pair.
     *
     * @tparam First The first element of the pair.
     * @tparam Second The second element of the pair.
     * @return Pointer to the column, NULL if not found.
     */
    template <typename First, typename Second, typename P = flecs::pair<First, Second>,
        typename A = actual_type_t<P>, if_not_t< flecs::is_pair<First>::value> = 0>
    A* get() const {
        return static_cast<A*>(get<First>(_::type<Second>::id(world_)));
    }

    /** Get column size */
    size_t column_size(int32_t index) {
        return ecs_table_get_column_size(table_, index);
    }

    /** Get depth for given relationship.
     *
     * @param rel The relationship.
     * @return The depth.
     */
    int32_t depth(flecs::entity_t rel) {
        return ecs_table_get_depth(world_, table_, rel);
    }

    /** Get depth for given relationship.
     *
     * @tparam Rel The relationship.
     * @return The depth.
     */
    template <typename Rel>
    int32_t depth() {
        return depth(_::type<Rel>::id(world_));
    }

    /** Get table.
     *
     * @return The table.
     */
    table_t* get_table() const {
        return table_;
    }

    /* Implicit conversion to table_t */
    operator table_t*() const {
        return table_;
    }

protected:
    world_t *world_;
    table_t *table_;
};

struct table_range : table {
    table_range()
        : table()
        , offset_(0)
        , count_(0) { }

    table_range(world_t *world, table_t *t, int32_t offset, int32_t count)
        : table(world, t)
        , offset_(offset)
        , count_(count) { }

    int32_t offset() const {
        return offset_;
    }

    int32_t count() const {
        return count_;
    }

    /** Get pointer to component array by column index.
     *
     * @param index The column index.
     * @return Pointer to the column, NULL if not a component.
     */
    void* get_column(int32_t index) const override {
        return ecs_table_get_column(table_, index, offset_);
    }

private:
    int32_t offset_ = 0;
    int32_t count_ = 0;
};

/** @} */

}

/**
 * @file addons/cpp/utils/iterable.hpp
 * @brief Base class for iterable objects, like queries.
 */

namespace flecs {

template <typename ... Components>
struct iter_iterable;

template <typename ... Components>
struct page_iterable;

template <typename ... Components>
struct worker_iterable; 

template <typename ... Components>
struct iterable {

    /** Each iterator.
     * The "each" iterator accepts a function that is invoked for each matching
     * entity. The following function signatures are valid:
     *  - func(flecs::entity e, Components& ...)
     *  - func(flecs::iter& it, size_t index, Components& ....)
     *  - func(Components& ...)
     */
    template <typename Func>
    void each(Func&& func) const {
        ecs_iter_t it = this->get_iter(nullptr);
        ecs_iter_next_action_t next = this->next_action();
        while (next(&it)) {
            _::each_delegate<Func, Components...>(func).invoke(&it);
        }
    }

    /** Run iterator.
     * The "each" iterator accepts a function that is invoked once for a query
     * with a valid iterator. The following signature is valid:
     *  - func(flecs::iter&)
     */
    template <typename Func>
    void run(Func&& func) const {
        ecs_iter_t it = this->get_iter(nullptr);
        _::run_delegate<Func>(func).invoke(&it);
    }

    template <typename Func>
    flecs::entity find(Func&& func) const {
        ecs_iter_t it = this->get_iter(nullptr);
        ecs_iter_next_action_t next = this->next_action();

        flecs::entity result;
        while (!result && next(&it)) {
            result = _::find_delegate<Func, Components...>(func).invoke(&it);
        }

        if (result) {
            ecs_iter_fini(&it);
        }

        return result;
    }

    /** Create iterator.
     * Create an iterator object that can be modified before iterating.
     */
    iter_iterable<Components...> iter(flecs::world_t *world = nullptr) const;

    /** Create iterator.
     * Create an iterator object that can be modified before iterating.
     */
    iter_iterable<Components...> iter(flecs::iter& iter) const;

    /** Create iterator.
     * Create an iterator object that can be modified before iterating.
     */
    iter_iterable<Components...> iter(flecs::entity e) const;

    /** Page iterator.
     * Create an iterator that limits the returned entities with offset/limit.
     * 
     * @param offset How many entities to skip.
     * @param limit The maximum number of entities to return.
     * @return Iterable that can be iterated with each/iter.
     */
    page_iterable<Components...> page(int32_t offset, int32_t limit);

    /** Worker iterator.
     * Create an iterator that divides the number of matched entities across
     * a number of resources.
     * 
     * @param index The index of the current resource.
     * @param count The total number of resources to divide entities between.
     * @return Iterable that can be iterated with each/iter.
     */
    worker_iterable<Components...> worker(int32_t index, int32_t count);

    /** Return number of entities matched by iterable. */
    int32_t count() const {
        return this->iter().count();
    }

    /** Return whether iterable has any matches. */
    bool is_true() const {
        return this->iter().is_true();
    }

    /** Return first entity matched by iterable. */
    flecs::entity first() const {
        return this->iter().first();
    }

    iter_iterable<Components...> set_var(int var_id, flecs::entity_t value) const {
        return this->iter().set_var(var_id, value);
    }

    iter_iterable<Components...> set_var(const char *name, flecs::entity_t value) const {
        return this->iter().set_var(name, value);
    }

    iter_iterable<Components...> set_var(const char *name, flecs::table_t *value) const {
        return this->iter().set_var(name, value);
    }

    iter_iterable<Components...> set_var(const char *name, ecs_table_range_t value) const {
        return this->iter().set_var(name, value);
    }

    iter_iterable<Components...> set_var(const char *name, flecs::table_range value) const {
        return this->iter().set_var(name, value);
    }

    // Limit results to tables with specified group id (grouped queries only)
    iter_iterable<Components...> set_group(uint64_t group_id) const {
        return this->iter().set_group(group_id);
    }

    // Limit results to tables with specified group id (grouped queries only)
    template <typename Group>
    iter_iterable<Components...> set_group() const {
        return this->iter().template set_group<Group>();
    }

    virtual ~iterable() { }
protected:
    friend iter_iterable<Components...>;
    friend page_iterable<Components...>;
    friend worker_iterable<Components...>;

    virtual ecs_iter_t get_iter(flecs::world_t *stage) const = 0;
    virtual ecs_iter_next_action_t next_action() const = 0;
};

template <typename ... Components>
struct iter_iterable final : iterable<Components...> {
    template <typename Iterable>
    iter_iterable(Iterable *it, flecs::world_t *world) 
    {
        it_ = it->get_iter(world);
        next_ = it->next_action();
        next_each_ = it->next_action();
        ecs_assert(next_ != nullptr, ECS_INTERNAL_ERROR, NULL);
        ecs_assert(next_each_ != nullptr, ECS_INTERNAL_ERROR, NULL);
    }

    iter_iterable<Components...>& set_var(int var_id, flecs::entity_t value) {
        ecs_assert(var_id != -1, ECS_INVALID_PARAMETER, 0);
        ecs_iter_set_var(&it_, var_id, value);
        return *this;
    }

    iter_iterable<Components...>& set_var(const char *name, flecs::entity_t value) {
        ecs_query_iter_t *qit = &it_.priv_.iter.query;
        int var_id = ecs_query_find_var(qit->query, name);
        ecs_assert(var_id != -1, ECS_INVALID_PARAMETER, name);
        ecs_iter_set_var(&it_, var_id, value);
        return *this;
    }

    iter_iterable<Components...>& set_var(const char *name, flecs::table_t *value) {
        ecs_query_iter_t *qit = &it_.priv_.iter.query;
        int var_id = ecs_query_find_var(qit->query, name);
        ecs_assert(var_id != -1, ECS_INVALID_PARAMETER, name);
        ecs_iter_set_var_as_table(&it_, var_id, value);
        return *this;
    }

    iter_iterable<Components...>& set_var(const char *name, ecs_table_range_t value) {
        ecs_query_iter_t *qit = &it_.priv_.iter.query;
        int var_id = ecs_query_find_var(qit->query, name);
        ecs_assert(var_id != -1, ECS_INVALID_PARAMETER, name);
        ecs_iter_set_var_as_range(&it_, var_id, &value);
        return *this;
    }

    iter_iterable<Components...>& set_var(const char *name, flecs::table_range value) {
        ecs_table_range_t range;
        range.table = value.get_table();
        range.offset = value.offset();
        range.count = value.count();
        return set_var(name, range);
    }

#   ifdef FLECS_JSON
/**
 * @file addons/cpp/mixins/json/iterable.inl
 * @brief JSON iterable mixin.
 */

/** Serialize iterator result to JSON.
 * 
 * @memberof flecs::iter
 * @ingroup cpp_addons_json
 */
flecs::string to_json(flecs::iter_to_json_desc_t *desc = nullptr) {
    char *json = ecs_iter_to_json(&it_, desc);
    return flecs::string(json);
}

#   endif

    // Return total number of entities in result.
    int32_t count() {
        int32_t result = 0;
        while (next_each_(&it_)) {
            result += it_.count;
        }
        return result;
    }

    // Returns true if iterator yields at least once result.
    bool is_true() {
        bool result = next_each_(&it_);
        if (result) {
            ecs_iter_fini(&it_);
        }
        return result;
    }

    // Return first matching entity.
    flecs::entity first() {
        flecs::entity result;
        if (next_each_(&it_) && it_.count) {
            result = flecs::entity(it_.world, it_.entities[0]);
            ecs_iter_fini(&it_);
        }
        return result;
    }

    // Limit results to tables with specified group id (grouped queries only)
    iter_iterable<Components...>& set_group(uint64_t group_id) {
        ecs_iter_set_group(&it_, group_id);
        return *this;
    }

    // Limit results to tables with specified group id (grouped queries only)
    template <typename Group>
    iter_iterable<Components...>& set_group() {
        ecs_iter_set_group(&it_, _::type<Group>().id(it_.real_world));
        return *this;
    }

protected:
    ecs_iter_t get_iter(flecs::world_t *world) const override {
        if (world) {
            ecs_iter_t result = it_;
            result.world = world;
            return result;
        }
        return it_;
    }

    ecs_iter_next_action_t next_action() const override {
        return next_;
    }

private:
    ecs_iter_t it_;
    ecs_iter_next_action_t next_;
    ecs_iter_next_action_t next_each_;
};

template <typename ... Components>
iter_iterable<Components...> iterable<Components...>::iter(flecs::world_t *world) const
{
    return iter_iterable<Components...>(this, world);
}

template <typename ... Components>
iter_iterable<Components...> iterable<Components...>::iter(flecs::iter& it) const
{
    return iter_iterable<Components...>(this, it.world());
}

template <typename ... Components>
iter_iterable<Components...> iterable<Components...>::iter(flecs::entity e) const
{
    return iter_iterable<Components...>(this, e.world());
}

template <typename ... Components>
struct page_iterable final : iterable<Components...> {
    template <typename Iterable>
    page_iterable(int32_t offset, int32_t limit, Iterable *it) 
        : offset_(offset)
        , limit_(limit)
    {
        chain_it_ = it->get_iter(nullptr);
    }

protected:
    ecs_iter_t get_iter(flecs::world_t*) const {
        return ecs_page_iter(&chain_it_, offset_, limit_);
    }

    ecs_iter_next_action_t next_action() const {
        return ecs_page_next;
    }

private:
    ecs_iter_t chain_it_;
    int32_t offset_;
    int32_t limit_;
};

template <typename ... Components>
page_iterable<Components...> iterable<Components...>::page(
    int32_t offset, 
    int32_t limit) 
{
    return page_iterable<Components...>(offset, limit, this);
}

template <typename ... Components>
struct worker_iterable final : iterable<Components...> {
    worker_iterable(int32_t offset, int32_t limit, iterable<Components...> *it) 
        : offset_(offset)
        , limit_(limit)
    {
        chain_it_ = it->get_iter(nullptr);
    }

protected:
    ecs_iter_t get_iter(flecs::world_t*) const {
        return ecs_worker_iter(&chain_it_, offset_, limit_);
    }

    ecs_iter_next_action_t next_action() const {
        return ecs_worker_next;
    }

private:
    ecs_iter_t chain_it_;
    int32_t offset_;
    int32_t limit_;
};

template <typename ... Components>
worker_iterable<Components...> iterable<Components...>::worker(
    int32_t index, 
    int32_t count) 
{
    return worker_iterable<Components...>(index, count, this);
}

}


// Mixin implementations
/**
 * @file addons/cpp/mixins/id/impl.hpp
 * @brief Id class implementation.
 */

#pragma once

namespace flecs {

inline flecs::entity id::entity() const {
    ecs_assert(!is_pair(), ECS_INVALID_OPERATION, NULL);
    ecs_assert(!flags(), ECS_INVALID_OPERATION, NULL);
    return flecs::entity(world_, id_);
}

inline flecs::entity id::flags() const {
    return flecs::entity(world_, id_ & ECS_ID_FLAGS_MASK);
}

inline flecs::entity id::first() const {
    ecs_assert(is_pair(), ECS_INVALID_OPERATION, NULL);

    flecs::entity_t e = ECS_PAIR_FIRST(id_);
    if (world_) {
        return flecs::entity(world_, ecs_get_alive(world_, e));
    } else {
        return flecs::entity(e);
    }
}

inline flecs::entity id::second() const {
    flecs::entity_t e = ECS_PAIR_SECOND(id_);
    if (world_) {
        return flecs::entity(world_, ecs_get_alive(world_, e));
    } else {
        return flecs::entity(e);
    }
}

inline flecs::entity id::add_flags(flecs::id_t flags) const {
    return flecs::entity(world_, id_ | flags);
}

inline flecs::entity id::remove_flags(flecs::id_t flags) const {
    (void)flags;
    ecs_assert((id_ & ECS_ID_FLAGS_MASK) == flags, ECS_INVALID_PARAMETER, NULL);
    return flecs::entity(world_, id_ & ECS_COMPONENT_MASK);
}

inline flecs::entity id::remove_flags() const {
    return flecs::entity(world_, id_ & ECS_COMPONENT_MASK);
}

inline flecs::entity id::remove_generation() const {
    return flecs::entity(world_, static_cast<uint32_t>(id_));
}

inline flecs::world id::world() const {
    return flecs::world(world_);
}

inline flecs::entity id::type_id() const {
    return flecs::entity(world_, ecs_get_typeid(world_, id_));
}


// Id mixin implementation

template <typename T>
inline flecs::id world::id() const {
    return flecs::id(world_, _::type<T>::id(world_));
}

template <typename ... Args>
inline flecs::id world::id(Args&&... args) const {
    return flecs::id(world_, FLECS_FWD(args)...);
}

template <typename First, typename Second>
inline flecs::id world::pair() const {
    return flecs::id(
        world_, 
        ecs_pair(
            _::type<First>::id(world_), 
            _::type<Second>::id(world_)));
}

template <typename First>
inline flecs::id world::pair(entity_t o) const {
    ecs_assert(!ECS_IS_PAIR(o), ECS_INVALID_PARAMETER, 
        "cannot create nested pairs");

    return flecs::id(
        world_,
        ecs_pair(
            _::type<First>::id(world_), 
            o));
}

inline flecs::id world::pair(entity_t r, entity_t o) const {
    ecs_assert(!ECS_IS_PAIR(r) && !ECS_IS_PAIR(o), ECS_INVALID_PARAMETER, 
        "cannot create nested pairs");

    return flecs::id(
        world_,
        ecs_pair(r, o));
}

}

/**
 * @file addons/cpp/mixins/entity/impl.hpp
 * @brief Entity implementation.
 */

#pragma once

namespace flecs {

template <typename T>
flecs::entity ref<T>::entity() const {
    return flecs::entity(world_, ref_.entity);
}

template <typename Self>
template <typename Func>
inline const Self& entity_builder<Self>::insert(const Func& func) const  {
    _::entity_with_delegate<Func>::invoke_ensure(
        this->world_, this->id_, func);
    return to_base();
}

template <typename T, if_t< is_enum<T>::value > >
const T* entity_view::get() const {
    entity_t r = _::type<T>::id(world_);
    entity_t c = ecs_get_target(world_, id_, r, 0);

    if (c) {
#ifdef FLECS_META
        using U = typename std::underlying_type<T>::type;
        const T* v = static_cast<const T*>(
            ecs_get_id(world_, c, ecs_pair(flecs::Constant, _::type<U>::id(world_))));
        ecs_assert(v != NULL, ECS_INTERNAL_ERROR, "missing enum constant value");
        return v;
#else
        // Fallback if we don't have the reflection addon
        return static_cast<const T*>(ecs_get_id(world_, id_, r));
#endif
    } else {
        // If there is no matching pair for (r, *), try just r
        return static_cast<const T*>(ecs_get_id(world_, id_, r));
    }
}

template<typename First>
inline flecs::entity entity_view::target(int32_t index) const 
{
    return flecs::entity(world_, 
        ecs_get_target(world_, id_, _::type<First>::id(world_), index));
}

inline flecs::entity entity_view::target(
    flecs::entity_t relationship, 
    int32_t index) const 
{
    return flecs::entity(world_, 
        ecs_get_target(world_, id_, relationship, index));
}

inline flecs::entity entity_view::target_for(
    flecs::entity_t relationship, 
    flecs::id_t id) const 
{
    return flecs::entity(world_, 
        ecs_get_target_for_id(world_, id_, relationship, id));
}

template <typename T>
inline flecs::entity entity_view::target_for(flecs::entity_t relationship) const {
    return target_for(relationship, _::type<T>::id(world_));
}

template <typename First, typename Second>
inline flecs::entity entity_view::target_for(flecs::entity_t relationship) const {
    return target_for(relationship, _::type<First, Second>::id(world_));
}

inline flecs::entity entity_view::parent() const {
    return target(flecs::ChildOf);
}

inline flecs::entity entity_view::mut(const flecs::world& stage) const {
    ecs_assert(!stage.is_readonly(), ECS_INVALID_PARAMETER, 
        "cannot use readonly world/stage to create mutable handle");
    return flecs::entity(id_).set_stage(stage.c_ptr());
}

inline flecs::entity entity_view::mut(const flecs::iter& it) const {
    ecs_assert(!it.world().is_readonly(), ECS_INVALID_PARAMETER, 
        "cannot use iterator created for readonly world/stage to create mutable handle");
    return flecs::entity(id_).set_stage(it.world().c_ptr());
}

inline flecs::entity entity_view::mut(const flecs::entity_view& e) const {
    ecs_assert(!e.world().is_readonly(), ECS_INVALID_PARAMETER, 
        "cannot use entity created for readonly world/stage to create mutable handle");
    return flecs::entity(id_).set_stage(e.world_);
}

inline flecs::entity entity_view::set_stage(world_t *stage) {
    return flecs::entity(stage, id_);
}   

inline flecs::type entity_view::type() const {
    return flecs::type(world_, ecs_get_type(world_, id_));
}

inline flecs::table entity_view::table() const {
    return flecs::table(world_, ecs_get_table(world_, id_));
}

inline flecs::table_range entity_view::range() const {
    ecs_record_t *r = ecs_record_find(world_, id_);
    if (r) {
        return flecs::table_range(world_, r->table, 
            ECS_RECORD_TO_ROW(r->row), 1);
    }
    return flecs::table_range();
}

template <typename Func>
inline void entity_view::each(const Func& func) const {
    const ecs_type_t *type = ecs_get_type(world_, id_);
    if (!type) {
        return;
    }

    const ecs_id_t *ids = type->array;
    int32_t count = type->count;

    for (int i = 0; i < count; i ++) {
        ecs_id_t id = ids[i];
        flecs::id ent(world_, id);
        func(ent); 
    }
}

template <typename Func>
inline void entity_view::each(flecs::id_t pred, flecs::id_t obj, const Func& func) const {
    flecs::world_t *real_world = const_cast<flecs::world_t*>(
        ecs_get_world(world_));

    const ecs_table_t *table = ecs_get_table(world_, id_);
    if (!table) {
        return;
    }

    const ecs_type_t *type = ecs_table_get_type(table);
    if (!type) {
        return;
    }

    flecs::id_t pattern = pred;
    if (obj) {
        pattern = ecs_pair(pred, obj);
    }

    int32_t cur = 0;
    id_t *ids = type->array;
    
    while (-1 != (cur = ecs_search_offset(real_world, table, cur, pattern, 0)))
    {
        flecs::id ent(world_, ids[cur]);
        func(ent);
        cur ++;
    }
}

template <typename Func>
inline void entity_view::each(const flecs::entity_view& rel, const Func& func) const {
    return this->each(rel, flecs::Wildcard, [&](flecs::id id) {
        flecs::entity obj = id.second();
        func(obj);
    });
}

template <typename Func, if_t< is_callable<Func>::value > >
inline bool entity_view::get(const Func& func) const {
    return _::entity_with_delegate<Func>::invoke_get(world_, id_, func);
} 

inline flecs::entity entity_view::lookup(const char *path, bool search_path) const {
    ecs_assert(id_ != 0, ECS_INVALID_PARAMETER, "invalid lookup from null handle");
    auto id = ecs_lookup_path_w_sep(world_, id_, path, "::", "::", search_path);
    return flecs::entity(world_, id);
}

inline flecs::entity entity_view::clone(bool copy_value, flecs::entity_t dst_id) const {
    if (!dst_id) {
        dst_id = ecs_new(world_);
    }

    flecs::entity dst = flecs::entity(world_, dst_id);
    ecs_clone(world_, dst_id, id_, copy_value);
    return dst;
}

// Entity mixin implementation
template <typename... Args>
inline flecs::entity world::entity(Args &&... args) const {
    return flecs::entity(world_, FLECS_FWD(args)...);
}

template <typename E, if_t< is_enum<E>::value >>
inline flecs::id world::id(E value) const {
    flecs::entity_t constant = enum_type<E>(world_).entity(value);
    return flecs::id(world_, constant);
}

template <typename E, if_t< is_enum<E>::value >>
inline flecs::entity world::entity(E value) const {
    flecs::entity_t constant = enum_type<E>(world_).entity(value);
    return flecs::entity(world_, constant);
}

template <typename T>
inline flecs::entity world::entity(const char *name) const {
    return flecs::entity(world_, _::type<T>::register_id(world_, name, true) );
}

template <typename... Args>
inline flecs::entity world::prefab(Args &&... args) const {
    flecs::entity result = flecs::entity(world_, FLECS_FWD(args)...);
    result.add(flecs::Prefab);
    return result;
}

template <typename T>
inline flecs::entity world::prefab(const char *name) const {
    flecs::entity result = flecs::component<T>(world_, name, true);
    result.add(flecs::Prefab);
    return result;
}

}

/**
 * @file addons/cpp/mixins/component/impl.hpp
 * @brief Component mixin implementation
 */

#pragma once

namespace flecs {

template <typename T, typename... Args>
inline flecs::component<T> world::component(Args &&... args) const {
    return flecs::component<T>(world_, FLECS_FWD(args)...);
}

template <typename... Args>
inline flecs::untyped_component world::component(Args &&... args) const {
    return flecs::untyped_component(world_, FLECS_FWD(args)...);
}

} // namespace flecs

/**
 * @file addons/cpp/mixins/term/impl.hpp
 * @brief Term implementation.
 */

#pragma once

/**
 * @file addons/cpp/mixins/term/builder_i.hpp
 * @brief Term builder interface.
 */

#pragma once

/**
 * @file addons/cpp/utils/signature.hpp
 * @brief Compile time utilities for deriving query attributes from param pack.
 */

#pragma once

#include <stdio.h>

namespace flecs {
namespace _ {

    template <typename T, if_t< is_const_p<T>::value > = 0>
    constexpr flecs::inout_kind_t type_to_inout() {
        return flecs::In;
    }

    template <typename T, if_t< is_reference<T>::value > = 0>
    constexpr flecs::inout_kind_t type_to_inout() {
        return flecs::InOut;
    }

    template <typename T, if_not_t< 
        is_const_p<T>::value || is_reference<T>::value > = 0>
    constexpr flecs::inout_kind_t type_to_inout() {
        return flecs::InOutDefault;
    }

    template <typename T, if_t< is_pointer<T>::value > = 0>
    constexpr flecs::oper_kind_t type_to_oper() {
        return flecs::Optional;
    }

    template <typename T, if_not_t< is_pointer<T>::value > = 0>
    constexpr flecs::oper_kind_t type_to_oper() {
        return flecs::And;
    }

    template <typename ... Components>
    struct sig {
        sig(flecs::world_t *world) 
            : world_(world)
            , ids({ (_::type<remove_pointer_t<Components>>::id(world))... })
            , inout ({ (type_to_inout<Components>())... })
            , oper ({ (type_to_oper<Components>())... }) 
        { }

        flecs::world_t *world_;
        flecs::array<flecs::id_t, sizeof...(Components)> ids;
        flecs::array<flecs::inout_kind_t, sizeof...(Components)> inout;
        flecs::array<flecs::oper_kind_t, sizeof...(Components)> oper;

        template <typename Builder>
        void populate(const Builder& b) {
            size_t i = 0;
            for (auto id : ids) {
               if (!(id & ECS_ID_FLAGS_MASK)) {
                    const flecs::type_info_t *ti = ecs_get_type_info(world_, id);
                    if (ti) {
                        // Union relationships always return a value of type
                        // flecs::entity_t which holds the target id of the 
                        // union relationship.
                        // If a union component with a non-zero size (like an 
                        // enum) is added to the query signature, the each/iter
                        // functions would accept a parameter of the component
                        // type instead of flecs::entity_t, which would cause
                        // an assert.
                        ecs_assert(
                            !ti->size || !ecs_has_id(world_, id, flecs::Union),
                            ECS_INVALID_PARAMETER,
                            "use with() method to add union relationship");
                    }
                }

                b->with(id).inout(inout[i]).oper(oper[i]);
                i ++;
            }
        }
    };

} // namespace _
} // namespace flecs


namespace flecs 
{

/** Term identifier builder.
 * A term identifier describes a single identifier in a term. Identifier
 * descriptions can reference entities by id, name or by variable, which means
 * the entity will be resolved when the term is evaluated.
 * 
 * @ingroup cpp_core_queries
 */
template<typename Base>
struct term_ref_builder_i {
    term_ref_builder_i() : term_ref_(nullptr) { }

    virtual ~term_ref_builder_i() { }

    /* The self flag indicates the term identifier itself is used */
    Base& self() {
        this->assert_term_ref();
        term_ref_->id |= flecs::Self;
        return *this;
    }

    /* Specify value of identifier by id */
    Base& id(flecs::entity_t id) {
        this->assert_term_ref();
        term_ref_->id = id;
        return *this;
    }

    /* Specify value of identifier by id. Almost the same as id(entity), but this
     * operation explicitly sets the flecs::IsEntity flag. This forces the id to 
     * be interpreted as entity, whereas not setting the flag would implicitly
     * convert ids for builtin variables such as flecs::This to a variable.
     * 
     * This function can also be used to disambiguate id(0), which would match
     * both id(entity_t) and id(const char*).
     */
    Base& entity(flecs::entity_t entity) {
        this->assert_term_ref();
        term_ref_->id = entity | flecs::IsEntity;
        return *this;
    }

    /* Specify value of identifier by name */
    Base& name(const char *name) {
        this->assert_term_ref();
        term_ref_->id |= flecs::IsEntity;
        term_ref_->name = const_cast<char*>(name);
        return *this;
    }

    /* Specify identifier is a variable (resolved at query evaluation time) */
    Base& var(const char *var_name) {
        this->assert_term_ref();
        term_ref_->id |= flecs::IsVariable;
        term_ref_->name = const_cast<char*>(var_name);
        return *this;
    }

    /* Override term id flags */
    Base& flags(flecs::flags32_t flags) {
        this->assert_term_ref();
        term_ref_->id = flags;
        return *this;
    }

    ecs_term_ref_t *term_ref_;

protected:
    virtual flecs::world_t* world_v() = 0;

    void assert_term_ref() {
        ecs_assert(term_ref_ != NULL, ECS_INVALID_PARAMETER, 
            "no active term (call .with() first)");
    }

private:
    operator Base&() {
        return *static_cast<Base*>(this);
    }
};

/** Term builder interface. 
 * A term is a single element of a query expression. 
 * 
 * @ingroup cpp_core_queries
 */
template<typename Base>
struct term_builder_i : term_ref_builder_i<Base> {
    term_builder_i() : term_(nullptr) { }

    term_builder_i(ecs_term_t *term_ptr) { 
        set_term(term_ptr);
    }

    Base& term(id_t id) {
        return this->id(id);
    }

    /* Call prior to setting values for src identifier */
    Base& src() {
        this->assert_term();
        this->term_ref_ = &term_->src;
        return *this;
    }

    /* Call prior to setting values for first identifier. This is either the
     * component identifier, or first element of a pair (in case second is
     * populated as well). */
    Base& first() {
        this->assert_term();
        this->term_ref_ = &term_->first;
        return *this;
    }

    /* Call prior to setting values for second identifier. This is the second 
     * element of a pair. Requires that first() is populated as well. */
    Base& second() {
        this->assert_term();
        this->term_ref_ = &term_->second;
        return *this;
    }

    /* Select src identifier, initialize it with entity id */
    Base& src(flecs::entity_t id) {
        this->src();
        this->id(id);
        return *this;
    }

    /* Select src identifier, initialize it with id associated with type */
    template<typename T>
    Base& src() {
        this->src(_::type<T>::id(this->world_v()));
        return *this;
    }

    /* Select src identifier, initialize it with name. If name starts with a $
     * the name is interpreted as a variable. */
    Base& src(const char *name) {
        ecs_assert(name != NULL, ECS_INVALID_PARAMETER, NULL);
        this->src();
        if (name[0] == '$') {
            this->var(&name[1]);
        } else {
            this->name(name);
        }
        return *this;
    }

    /* Select first identifier, initialize it with entity id */
    Base& first(flecs::entity_t id) {
        this->first();
        this->id(id);
        return *this;
    }

    /* Select first identifier, initialize it with id associated with type */
    template<typename T>
    Base& first() {
        this->first(_::type<T>::id(this->world_v()));
        return *this;
    }

    /* Select first identifier, initialize it with name. If name starts with a $
     * the name is interpreted as a variable. */
    Base& first(const char *name) {
        ecs_assert(name != NULL, ECS_INVALID_PARAMETER, NULL);
        this->first();
        if (name[0] == '$') {
            this->var(&name[1]);
        } else {
            this->name(name);
        }
        return *this;
    }

    /* Select second identifier, initialize it with entity id */
    Base& second(flecs::entity_t id) {
        this->second();
        this->id(id);
        return *this;
    }

    /* Select second identifier, initialize it with id associated with type */
    template<typename T>
    Base& second() {
        this->second(_::type<T>::id(this->world_v()));
        return *this;
    }

    /* Select second identifier, initialize it with name. If name starts with a $
     * the name is interpreted as a variable. */
    Base& second(const char *name) {
        ecs_assert(name != NULL, ECS_INVALID_PARAMETER, NULL);
        this->second();
        if (name[0] == '$') {
            this->var(&name[1]);
        } else {
            this->name(name);
        }
        return *this;
    }

    /* The up flag indicates that the term identifier may be substituted by
     * traversing a relationship upwards. For example: substitute the identifier
     * with its parent by traversing the ChildOf relationship. */
    Base& up(flecs::entity_t trav = 0) {
        this->assert_term_ref();
        ecs_check(this->term_ref_ != &term_->first, ECS_INVALID_PARAMETER,
            "up traversal can only be applied to term source");
        ecs_check(this->term_ref_ != &term_->second, ECS_INVALID_PARAMETER,
            "up traversal can only be applied to term source");
        this->term_ref_->id |= flecs::Up;
        if (trav) {
            term_->trav = trav;
        }
    error:
        return *this;
    }

    template <typename Trav>
    Base& up() {
        return this->up(_::type<Trav>::id(this->world_v()));
    }

    /* The cascade flag is like up, but returns results in breadth-first order.
     * Only supported for flecs::query */
    Base& cascade(flecs::entity_t trav = 0) {
        this->assert_term_ref();
        this->up();
        this->term_ref_->id |= flecs::Cascade;
        if (trav) {
            term_->trav = trav;
        }
        return *this;
    }

    template <typename Trav>
    Base& cascade() {
        return this->cascade(_::type<Trav>::id(this->world_v()));
    }

    /* Use with cascade to iterate results in descending (bottom -> top) order */
    Base& desc() {
        this->assert_term_ref();
        this->term_ref_->id |= flecs::Desc;
        return *this;
    }

    /* Same as up(), exists for backwards compatibility */
    Base& parent() {
        return this->up();
    }

    /* Specify relationship to traverse, and flags to indicate direction */
    Base& trav(flecs::entity_t trav, flecs::flags32_t flags = 0) {
        this->assert_term_ref();
        term_->trav = trav;
        this->term_ref_->id |= flags;
        return *this;
    }

    /** Set id flags for term. */
    Base& id_flags(id_t flags) {
        this->assert_term();
        term_->id |= flags;
        return *this;
    }

    /** Set read/write access of term. */
    Base& inout(flecs::inout_kind_t inout) {
        this->assert_term();
        term_->inout = static_cast<int16_t>(inout);
        return *this;
    }

    /** Set read/write access for stage. Use this when a system reads or writes
     * components other than the ones provided by the query. This information 
     * can be used by schedulers to insert sync/merge points between systems
     * where deferred operations are flushed.
     * 
     * Setting this is optional. If not set, the value of the accessed component
     * may be out of sync for at most one frame.
     */
    Base& inout_stage(flecs::inout_kind_t inout) {
        this->assert_term();
        term_->inout = static_cast<int16_t>(inout);
        if (term_->oper != EcsNot) {
            this->src().entity(0);
        }
        return *this;
    }

    /** Short for inout_stage(flecs::Out). 
     *   Use when system uses add, remove or set. 
     */
    Base& write() {
        return this->inout_stage(flecs::Out);
    }

    /** Short for inout_stage(flecs::In).
     *   Use when system uses get.
     */
    Base& read() {
        return this->inout_stage(flecs::In);
    }

    /** Short for inout_stage(flecs::InOut).
     *   Use when system uses ensure.
     */
    Base& read_write() {
        return this->inout_stage(flecs::InOut);
    }

    /** Short for inout(flecs::In) */
    Base& in() {
        return this->inout(flecs::In);
    }

    /** Short for inout(flecs::Out) */
    Base& out() {
        return this->inout(flecs::Out);
    }

    /** Short for inout(flecs::InOut) */
    Base& inout() {
        return this->inout(flecs::InOut);
    }

    /** Short for inout(flecs::In) */
    Base& inout_none() {
        return this->inout(flecs::InOutNone);
    }

    /** Set operator of term. */
    Base& oper(flecs::oper_kind_t oper) {
        this->assert_term();
        term_->oper = static_cast<int16_t>(oper);
        return *this;
    }

    /* Short for oper(flecs::And) */
    Base& and_() {
        return this->oper(flecs::And);
    }

    /* Short for oper(flecs::Or) */
    Base& or_() {
        return this->oper(flecs::Or);
    }

    /* Short for oper(flecs::Or) */
    Base& not_() {
        return this->oper(flecs::Not);
    }

    /* Short for oper(flecs::Or) */
    Base& optional() {
        return this->oper(flecs::Optional);
    }

    /* Short for oper(flecs::AndFrom) */
    Base& and_from() {
        return this->oper(flecs::AndFrom);
    }

    /* Short for oper(flecs::OrFrom) */
    Base& or_from() {
        return this->oper(flecs::OrFrom);
    }

    /* Short for oper(flecs::NotFrom) */
    Base& not_from() {
        return this->oper(flecs::NotFrom);
    }

    /** Match singleton. */
    Base& singleton() {
        this->assert_term();
        ecs_assert(term_->id || term_->first.id, ECS_INVALID_PARAMETER, 
                "no component specified for singleton");
        
        flecs::id_t sid = term_->id;
        if (!sid) {
            sid = term_->first.id;
        }

        ecs_assert(sid != 0, ECS_INVALID_PARAMETER, NULL);

        if (!ECS_IS_PAIR(sid)) {
            term_->src.id = sid;
        } else {
            term_->src.id = ecs_pair_first(world(), sid);
        }
        return *this;
    }

    /* Query terms are not triggered on by observers */
    Base& filter() {
        term_->inout = EcsInOutFilter;
        return *this;
    }

    ecs_term_t *term_;

protected:
    virtual flecs::world_t* world_v() override = 0;

    void set_term(ecs_term_t *term) {
        term_ = term;
        if (term) {
            this->term_ref_ = &term_->src; // default to subject
        } else {
            this->term_ref_ = nullptr;
        }
    }

private:
    void assert_term() {
        ecs_assert(term_ != NULL, ECS_INVALID_PARAMETER, 
            "no active term (call .with() first)");
    }

    operator Base&() {
        return *static_cast<Base*>(this);
    }   
};

}


namespace flecs {

/** Class that describes a term.
 * 
 * @ingroup cpp_core_queries
 */
struct term final : term_builder_i<term> {
    term()
        : term_builder_i<term>(&value)
        , value({})
        , world_(nullptr) { }

    term(flecs::world_t *world_ptr) 
        : term_builder_i<term>(&value)
        , value({})
        , world_(world_ptr) { }

    term(flecs::world_t *world_ptr, ecs_term_t t)
        : term_builder_i<term>(&value)
        , value({})
        , world_(world_ptr) {
            value = t;
            this->set_term(&value);
        }

    term(flecs::world_t *world_ptr, id_t id)
        : term_builder_i<term>(&value)
        , value({})
        , world_(world_ptr) {
            if (id & ECS_ID_FLAGS_MASK) {
                value.id = id;
            } else {
                value.first.id = id;
            }
            this->set_term(&value);
        }

    term(flecs::world_t *world_ptr, entity_t r, entity_t o)
        : term_builder_i<term>(&value)
        , value({})
        , world_(world_ptr) {
            value.id = ecs_pair(r, o);
            this->set_term(&value);
        }

    term(id_t id) 
        : term_builder_i<term>(&value)
        , value({})
        , world_(nullptr) { 
            if (id & ECS_ID_FLAGS_MASK) {
                value.id = id;
            } else {
                value.first.id = id;
            }
        }

    term(id_t r, id_t o) 
        : term_builder_i<term>(&value)
        , value({})
        , world_(nullptr) { 
            value.id = ecs_pair(r, o);
        }

    void reset() {
        value = {};
        this->set_term(nullptr);
    }

    bool is_set() {
        return ecs_term_is_initialized(&value);
    }

    flecs::id id() {
        return flecs::id(world_, value.id);
    }

    flecs::inout_kind_t inout() {
        return static_cast<flecs::inout_kind_t>(value.inout);
    }

    flecs::oper_kind_t oper() {
        return static_cast<flecs::oper_kind_t>(value.oper);
    }

    flecs::entity get_src() {
        return flecs::entity(world_, ECS_TERM_REF_ID(&value.src));
    }

    flecs::entity get_first() {
        return flecs::entity(world_, ECS_TERM_REF_ID(&value.first));
    }

    flecs::entity get_second() {
        return flecs::entity(world_, ECS_TERM_REF_ID(&value.second));
    }

    operator flecs::term_t() const {
        return value;
    }

    flecs::term_t value;

protected:
    flecs::world_t* world_v() override { return world_; }

private:
    flecs::world_t *world_;
};

// Term mixin implementation
template <typename... Args>
inline flecs::term world::term(Args &&... args) const {
    return flecs::term(world_, FLECS_FWD(args)...);
}

template <typename T>
inline flecs::term world::term() const {
    return flecs::term(world_, _::type<T>::id(world_));
}

template <typename First, typename Second>
inline flecs::term world::term() const {
    return flecs::term(world_, ecs_pair(
        _::type<First>::id(world_),
        _::type<Second>::id(world_)));
}

}

/**
 * @file addons/cpp/mixins/query/impl.hpp
 * @brief Query implementation.
 */

#pragma once

/**
 * @file addons/cpp/mixins/query/builder.hpp
 * @brief Query builder.
 */

#pragma once

/**
 * @file addons/cpp/utils/builder.hpp
 * @brief Builder base class.
 * 
 * Generic functionality for builder classes.
 */

#pragma once

namespace flecs {
namespace _ {

// Macros for template types so we don't go cross-eyed
#define FLECS_TBUILDER template<typename ... Components> class
#define FLECS_IBUILDER template<typename IBase, typename ... Components> class

template<FLECS_TBUILDER T, typename TDesc, typename Base, FLECS_IBUILDER IBuilder, typename ... Components>
struct builder : IBuilder<Base, Components ...>
{
    using IBase = IBuilder<Base, Components ...>;

public:
    builder(flecs::world_t *world)
        : IBase(&desc_)
        , desc_{}
        , world_(world) { }

    builder(const builder& f) 
        : IBase(&desc_, f.term_index_)
    {
        world_ = f.world_;
        desc_ = f.desc_;
    }

    builder(builder&& f)  noexcept
        : builder<T, TDesc, Base, IBuilder, Components...>(f) { }

    operator TDesc*() {
        return &desc_;
    }

    T<Components ...> build() {
        return T<Components...>(world_, *static_cast<Base*>(this));
    }

protected:
    flecs::world_t* world_v() override { return world_; }
    TDesc desc_;
    flecs::world_t *world_;
};

#undef FLECS_TBUILDER
#undef FLECS_IBUILDER

} // namespace _
} // namespace flecs

/**
 * @file addons/cpp/mixins/query/builder_i.hpp
 * @brief Query builder interface.
 */

#pragma once


namespace flecs 
{

/** Query builder interface.
 * 
 * @ingroup cpp_core_queries
 */
template<typename Base, typename ... Components>
struct query_builder_i : term_builder_i<Base> {
    query_builder_i(ecs_query_desc_t *desc, int32_t term_index = 0) 
        : term_index_(term_index)
        , expr_count_(0)
        , desc_(desc) { }

    Base& query_flags(ecs_flags32_t flags) {
        desc_->flags |= flags;
        return *this;
    }

    Base& cache_kind(query_cache_kind_t kind) {
        desc_->cache_kind = static_cast<ecs_query_cache_kind_t>(kind);
        return *this;
    }

    Base& cached() {
        return cache_kind(flecs::QueryCacheAuto);
    }

    Base& expr(const char *expr) {
        ecs_check(expr_count_ == 0, ECS_INVALID_OPERATION,
            "query_builder::expr() called more than once");
        desc_->expr = expr;
        expr_count_ ++;

    error:
        return *this;
    }

    /* With methods */

    template<typename T>
    Base& with() {
        this->term();
        *this->term_ = flecs::term(_::type<T>::id(this->world_v()));
        this->term_->inout = static_cast<ecs_inout_kind_t>(
            _::type_to_inout<T>());
            if (this->term_->inout == EcsInOutDefault) {
            this->inout_none();
        }
        return *this;
    }

    Base& with(id_t id) {
        this->term();
        *this->term_ = flecs::term(id);
        if (this->term_->inout == EcsInOutDefault) {
            this->inout_none();
        }
        return *this;
    }

    Base& with(const char *name) {
        this->term();
        *this->term_ = flecs::term().first(name);
        if (this->term_->inout == EcsInOutDefault) {
            this->inout_none();
        }
        return *this;
    }

    Base& with(const char *first, const char *second) {
        this->term();
        *this->term_ = flecs::term().first(first).second(second);
        if (this->term_->inout == EcsInOutDefault) {
            this->inout_none();
        }
        return *this;
    }

    Base& with(entity_t r, entity_t o) {
        this->term();
        *this->term_ = flecs::term(r, o);
        if (this->term_->inout == EcsInOutDefault) {
            this->inout_none();
        }
        return *this;
    }

    Base& with(entity_t r, const char *o) {
        this->term();
        *this->term_ = flecs::term(r).second(o);
        if (this->term_->inout == EcsInOutDefault) {
            this->inout_none();
        }
        return *this;
    }

    Base& with(const char *r, entity_t o) {
        this->term();
        *this->term_ = flecs::term().first(r).second(o);
        if (this->term_->inout == EcsInOutDefault) {
            this->inout_none();
        }
        return *this;
    }

    template<typename First>
    Base& with(id_t o) {
        return this->with(_::type<First>::id(this->world_v()), o);
    }

    template<typename First>
    Base& with(const char *second) {
        return this->with(_::type<First>::id(this->world_v())).second(second);
    }

    template<typename First, typename Second>
    Base& with() {
        return this->with<First>(_::type<Second>::id(this->world_v()));
    }

    template <typename E, if_t< is_enum<E>::value > = 0>
    Base& with(E value) {
        flecs::entity_t r = _::type<E>::id(this->world_v());
        auto o = enum_type<E>(this->world_v()).entity(value);
        return this->with(r, o);
    }

    Base& with(flecs::term& term) {
        this->term();
        *this->term_ = term;
        return *this;
    }

    Base& with(flecs::term&& term) {
        this->term();
        *this->term_ = term;
        return *this;
    }

    /* Without methods, shorthand for .with(...).not_(). */

    template <typename ... Args>
    Base& without(Args&&... args) {
        return this->with(FLECS_FWD(args)...).not_();
    }

    template <typename T, typename ... Args>
    Base& without(Args&&... args) {
        return this->with<T>(FLECS_FWD(args)...).not_();
    }

    template <typename First, typename Second>
    Base& without() {
        return this->with<First, Second>().not_();
    }

    /* Write/read methods */

    Base& write() {
        term_builder_i<Base>::write();
        return *this;
    }

    template <typename ... Args>
    Base& write(Args&&... args) {
        return this->with(FLECS_FWD(args)...).write();
    }

    template <typename T, typename ... Args>
    Base& write(Args&&... args) {
        return this->with<T>(FLECS_FWD(args)...).write();
    }

    template <typename First, typename Second>
    Base& write() {
        return this->with<First, Second>().write();
    }

    Base& read() {
        term_builder_i<Base>::read();
        return *this;
    }

    template <typename ... Args>
    Base& read(Args&&... args) {
        return this->with(FLECS_FWD(args)...).read();
    }

    template <typename T, typename ... Args>
    Base& read(Args&&... args) {
        return this->with<T>(FLECS_FWD(args)...).read();
    }

    template <typename First, typename Second>
    Base& read() {
        return this->with<First, Second>().read();
    }

    /* Scope_open/scope_close shorthand notation. */
    Base& scope_open() {
        return this->with(flecs::ScopeOpen).entity(0);
    }

    Base& scope_close() {
        return this->with(flecs::ScopeClose).entity(0);
    }

    /* Term notation for more complex query features */

    Base& term() {
        if (this->term_) {
            ecs_check(ecs_term_is_initialized(this->term_), 
                ECS_INVALID_OPERATION, 
                    "query_builder::term() called without initializing term");
        }

        ecs_check(term_index_ < FLECS_TERM_COUNT_MAX, 
            ECS_INVALID_PARAMETER, "maximum number of terms exceeded");

        this->set_term(&desc_->terms[term_index_]);

        term_index_ ++;
    
    error:
        return *this;
    }

    Base& term_at(int32_t term_index) {
        ecs_assert(term_index >= 0, ECS_INVALID_PARAMETER, NULL);
        int32_t prev_index = term_index_;
        term_index_ = term_index;
        this->term();
        term_index_ = prev_index;
        ecs_assert(ecs_term_is_initialized(this->term_), 
            ECS_INVALID_PARAMETER, NULL);
        return *this;
    }

    /** Sort the output of a query.
     * This enables sorting of entities across matched tables. As a result of this
     * operation, the order of entities in the matched tables may be changed. 
     * Resorting happens when a query iterator is obtained, and only if the table
     * data has changed.
     *
     * If multiple queries that match the same (down)set of tables specify different 
     * sorting functions, resorting is likely to happen every time an iterator is
     * obtained, which can significantly slow down iterations.
     *
     * The sorting function will be applied to the specified component. Resorting
     * only happens if that component has changed, or when the entity order in the
     * table has changed. If no component is provided, resorting only happens when
     * the entity order changes.
     *
     * @tparam T The component used to sort.
     * @param compare The compare function used to sort the components.
     */      
    template <typename T>
    Base& order_by(int(*compare)(flecs::entity_t, const T*, flecs::entity_t, const T*)) {
        ecs_order_by_action_t cmp = reinterpret_cast<ecs_order_by_action_t>(compare);
        return this->order_by(_::type<T>::id(this->world_v()), cmp);
    }

    /** Sort the output of a query.
     * Same as order_by<T>, but with component identifier.
     *
     * @param component The component used to sort.
     * @param compare The compare function used to sort the components.
     */    
    Base& order_by(flecs::entity_t component, int(*compare)(flecs::entity_t, const void*, flecs::entity_t, const void*)) {
        desc_->order_by_callback = reinterpret_cast<ecs_order_by_action_t>(compare);
        desc_->order_by = component;
        return *this;
    }

    /** Group and sort matched tables.
     * Similar to ecs_query_order_by(), but instead of sorting individual entities, this
     * operation only sorts matched tables. This can be useful of a query needs to
     * enforce a certain iteration order upon the tables it is iterating, for 
     * example by giving a certain component or tag a higher priority.
     *
     * The sorting function assigns a "rank" to each type, which is then used to
     * sort the tables. Tables with higher ranks will appear later in the iteration.
     * 
     * Resorting happens when a query iterator is obtained, and only if the set of
     * matched tables for a query has changed. If table sorting is enabled together
     * with entity sorting, table sorting takes precedence, and entities will be
     * sorted within each set of tables that are assigned the same rank.
     *
     * @tparam T The component used to determine the group rank.
     * @param group_by_action Callback that determines group id for table.
     */
    template <typename T>
    Base& group_by(uint64_t(*group_by_action)(flecs::world_t*, flecs::table_t *table, flecs::id_t id, void* ctx)) {
        ecs_group_by_action_t action = reinterpret_cast<ecs_group_by_action_t>(group_by_action);
        return this->group_by(_::type<T>::id(this->world_v()), action);
    }

    /** Group and sort matched tables.
     * Same as group_by<T>, but with component identifier.
     *
     * @param component The component used to determine the group rank.
     * @param group_by_action Callback that determines group id for table.
     */
    Base& group_by(flecs::entity_t component, uint64_t(*group_by_action)(flecs::world_t*, flecs::table_t *table, flecs::id_t id, void* ctx)) {
        desc_->group_by_callback = reinterpret_cast<ecs_group_by_action_t>(group_by_action);
        desc_->group_by = component;
        return *this;
    }

    /** Group and sort matched tables.
     * Same as group_by<T>, but with default group_by action.
     *
     * @tparam T The component used to determine the group rank.
     */
    template <typename T>
    Base& group_by() {
        return this->group_by(_::type<T>::id(this->world_v()), nullptr);
    }

    /** Group and sort matched tables.
     * Same as group_by, but with default group_by action.
     *
     * @param component The component used to determine the group rank.
     */
    Base& group_by(flecs::entity_t component) {
        return this->group_by(component, nullptr);
    }

    /** Specify context to be passed to group_by function.
     *
     * @param ctx Context to pass to group_by function.
     * @param ctx_free Function to cleanup context (called when query is deleted).
     */
    Base& group_by_ctx(void *ctx, ecs_ctx_free_t ctx_free = nullptr) {
        desc_->group_by_ctx = ctx;
        desc_->group_by_ctx_free = ctx_free;
        return *this;
    }

    /** Specify on_group_create action.
     */
    Base& on_group_create(ecs_group_create_action_t action) {
        desc_->on_group_create = action;
        return *this;
    }

    /** Specify on_group_delete action.
     */
    Base& on_group_delete(ecs_group_delete_action_t action) {
        desc_->on_group_delete = action;
        return *this;
    }

protected:
    virtual flecs::world_t* world_v() override = 0;
    int32_t term_index_;
    int32_t expr_count_;

private:
    operator Base&() {
        return *static_cast<Base*>(this);
    }

    ecs_query_desc_t *desc_;
};

}


namespace flecs {
namespace _ {
    template <typename ... Components>
    using query_builder_base = builder<
        query, ecs_query_desc_t, query_builder<Components...>, 
        query_builder_i, Components ...>;
}

/** Query builder.
 * 
 * @ingroup cpp_core_queries
 */
template <typename ... Components>
struct query_builder final : _::query_builder_base<Components...> {
    query_builder(flecs::world_t* world, flecs::entity query_entity)
        : _::query_builder_base<Components...>(world)
    {
        _::sig<Components...>(world).populate(this);
        this->desc_.entity = query_entity.id();
    }

    query_builder(flecs::world_t* world, const char *name = nullptr)
        : _::query_builder_base<Components...>(world)
    {
        _::sig<Components...>(world).populate(this);
        if (name != nullptr) {
            ecs_entity_desc_t entity_desc = {};
            entity_desc.name = name;
            entity_desc.sep = "::";
            entity_desc.root_sep = "::";
            this->desc_.entity = ecs_entity_init(world, &entity_desc);
        }
    }

    template <typename Func>
    void each(Func&& func) {
        this->build().each(FLECS_FWD(func));
    }
};

}


namespace flecs 
{

struct query_base {
    query_base() { }

    query_base(query_t *q)
        : query_(q) { 
            flecs_poly_claim(q);
        }

    query_base(const query_t *q)
        : query_(ECS_CONST_CAST(query_t*, q)) { 
            flecs_poly_claim(q);
        }

    query_base(world_t *world, ecs_query_desc_t *desc) {
        if (desc->entity && desc->terms[0].id == 0) {
            const flecs::Poly *query_poly = ecs_get_pair(
                world, desc->entity, EcsPoly, EcsQuery);
            if (query_poly) {
                query_ = static_cast<flecs::query_t*>(query_poly->poly);
                flecs_poly_claim(query_);
                return;
            }
        }

        query_ = ecs_query_init(world, desc);
    }

    query_base(const query_base& obj) {
        this->query_ = obj.query_;
        flecs_poly_claim(this->query_);
    }

    query_base& operator=(const query_base& obj) {
        this->query_ = obj.query_;
        flecs_poly_claim(this->query_);
        return *this; 
    }

    query_base(query_base&& obj) noexcept {
        this->query_ = obj.query_;
        obj.query_ = nullptr;
    }

    query_base& operator=(query_base&& obj) noexcept {
        this->query_ = obj.query_;
        obj.query_ = nullptr;
        return *this; 
    }

    flecs::entity entity() {
        return flecs::entity(query_->world, query_->entity);
    }

    const flecs::query_t* c_ptr() const {
        return query_;
    }

    operator const flecs::query_t*() const {
        return query_;
    }

    operator bool() const {
        return query_ != nullptr;
    }

    /** Free persistent query.
     * A persistent query is a query that is associated with an entity, such as
     * system queries and named queries. Persistent queries must be deleted with
     * destruct(), or will be deleted automatically at world cleanup. 
     */
    void destruct() {
        ecs_assert(query_->entity != 0, ECS_INVALID_OPERATION, "destruct() "
            "should only be called on queries associated with entities");
        ecs_query_fini(query_);
        query_ = nullptr;
    }

    ~query_base() {
        /* Only free if query is not associated with entity, such as system
         * queries and named queries. Named queries have to be either explicitly
         * deleted with the .destruct() method, or will be deleted when the
         * world is deleted. */
        if (query_ && !query_->entity) {
            if (!flecs_poly_release(query_)) {
                ecs_query_fini(query_);
                query_ = nullptr;
            }
        }
    }

    /** Returns whether the query data changed since the last iteration.
     * This operation must be invoked before obtaining the iterator, as this will
     * reset the changed state. The operation will return true after:
     * - new entities have been matched with
     * - matched entities were deleted
     * - matched components were changed
     * 
     * @return true if entities changed, otherwise false.
     */
    bool changed() const {
        return ecs_query_changed(query_);
    }

    /** Get info for group. 
     * 
     * @param group_id The group id for which to retrieve the info.
     * @return The group info.
     */
    const flecs::query_group_info_t* group_info(uint64_t group_id) const {
        return ecs_query_get_group_info(query_, group_id);
    }

    /** Get context for group. 
     * 
     * @param group_id The group id for which to retrieve the context.
     * @return The group context.
     */
    void* group_ctx(uint64_t group_id) const {
        const flecs::query_group_info_t *gi = group_info(group_id);
        if (gi) {
            return gi->ctx;
        } else {
            return NULL;
        }
    }

    template <typename Func>
    void each_term(const Func& func) {
        for (int i = 0; i < query_->term_count; i ++) {
            flecs::term t(query_->world, query_->terms[i]);
            func(t);
            t.reset(); // prevent freeing resources
        }
    }

    flecs::term term(int32_t index) {
        return flecs::term(query_->world, query_->terms[index]);
    }

    int32_t term_count() {
        return query_->term_count;
    }

    int32_t field_count() {
        return query_->field_count;
    }

    int32_t find_var(const char *name) {
        return ecs_query_find_var(query_, name);
    }

    flecs::string str() {
        char *result = ecs_query_str(query_);
        return flecs::string(result);
    }

    /** Returns a string representing the query plan.
     * This can be used to analyze the behavior & performance of the query.
     * @see ecs_query_plan
     */
    flecs::string plan() const {
        char *result = ecs_query_plan(query_);
        return flecs::string(result);
    }

    operator query<>() const;

#   ifdef FLECS_JSON

/** Serialize query to JSON.
 * 
 * @memberof flecs::query_base
 * @ingroup cpp_addons_json
 */
flecs::string to_json(flecs::iter_to_json_desc_t *desc = nullptr) {
    ecs_iter_t it = ecs_query_iter(ecs_get_world(query_), query_);
    char *json = ecs_iter_to_json(&it, desc);
    return flecs::string(json);
}
#   endif

protected:
    query_t *query_ = nullptr;
};

template<typename ... Components>
struct query : query_base, iterable<Components...> {
private:
    using Fields = typename _::field_ptrs<Components...>::array;

public:
    using query_base::query_base;

    query() : query_base() { } // necessary not to confuse msvc

    query(const query& obj) : query_base(obj) { }

    query& operator=(const query& obj) {
        query_base::operator=(obj);
        return *this;
    }

    query(query&& obj) noexcept : query_base(FLECS_MOV(obj)) { }

    query& operator=(query&& obj) noexcept {
        query_base::operator=(FLECS_FWD(obj));
        return *this;
    }

private:
    ecs_iter_t get_iter(flecs::world_t *world) const override {
        ecs_assert(query_ != nullptr, ECS_INVALID_PARAMETER, 
            "cannot iterate invalid query");
        if (!world) {
            world = query_->world;
        }
        return ecs_query_iter(world, query_);
    }

    ecs_iter_next_action_t next_action() const override {
        return ecs_query_next;
    }
};

// World mixin implementation
template <typename... Comps, typename... Args>
inline flecs::query<Comps...> world::query(Args &&... args) const {
    return flecs::query_builder<Comps...>(world_, FLECS_FWD(args)...)
        .build();
}

inline flecs::query<> world::query(flecs::entity query_entity) const {
    ecs_query_desc_t desc = {};
    desc.entity = query_entity;
    return flecs::query<>(world_, &desc);
}

template <typename... Comps, typename... Args>
inline flecs::query_builder<Comps...> world::query_builder(Args &&... args) const {
    return flecs::query_builder<Comps...>(world_, FLECS_FWD(args)...);
}

// world::each
namespace _ {

// Each with entity parameter
template<typename Func, typename ... Args>
struct query_delegate_w_ent;

template<typename Func, typename E, typename ... Args>
struct query_delegate_w_ent<Func, arg_list<E, Args ...> >
{
    query_delegate_w_ent(const flecs::world& world, Func&& func) {
        auto f = world.query<Args ...>();
        f.each(FLECS_MOV(func));
    }
};

// Each without entity parameter
template<typename Func, typename ... Args>
struct query_delegate_no_ent;

template<typename Func, typename ... Args>
struct query_delegate_no_ent<Func, arg_list<Args ...> >
{
    query_delegate_no_ent(const flecs::world& world, Func&& func) {
        auto f = world.query<Args ...>();
        f.each(FLECS_MOV(func));
    }
};

// Switch between function with & without entity parameter
template<typename Func, typename T = int>
struct query_delegate;

template <typename Func>
struct query_delegate<Func, if_t<is_same<first_arg_t<Func>, flecs::entity>::value> > {
    query_delegate(const flecs::world& world, Func&& func) {
        query_delegate_w_ent<Func, arg_list_t<Func>>(world, FLECS_MOV(func));
    }
};

template <typename Func>
struct query_delegate<Func, if_not_t<is_same<first_arg_t<Func>, flecs::entity>::value> > {
    query_delegate(const flecs::world& world, Func&& func) {
        query_delegate_no_ent<Func, arg_list_t<Func>>(world, FLECS_MOV(func));
    }
};

}

template <typename Func>
inline void world::each(Func&& func) const {
    _::query_delegate<Func> f_delegate(*this, FLECS_MOV(func));
}

template <typename T, typename Func>
inline void world::each(Func&& func) const {
    ecs_iter_t it = ecs_each_id(world_, _::type<T>::id(world_));

    while (ecs_each_next(&it)) {
        _::each_delegate<Func, T>(func).invoke(&it);
    }
}

template <typename Func>
inline void world::each(flecs::id_t each_id, Func&& func) const {
    ecs_iter_t it = ecs_each_id(world_, each_id);

    while (ecs_each_next(&it)) {
        _::each_delegate<Func>(func).invoke(&it);
    }
}

// query_base implementation
inline query_base::operator flecs::query<> () const {
    return flecs::query<>(query_);
}

}

/**
 * @file addons/cpp/mixins/observer/impl.hpp
 * @brief Observer implementation.
 */

#pragma once

/**
 * @file addons/cpp/mixins/observer/builder.hpp
 * @brief Observer builder.
 */

#pragma once

/**
 * @file addons/cpp/utils/node_builder.hpp
 * @brief Base builder class for node objects, like systems, observers.
 */

#pragma once

namespace flecs {
namespace _ {

// Macros for template types so we don't go cross-eyed
#define FLECS_IBUILDER template<typename IBase, typename ... Components> class

template<typename T, typename TDesc, typename Base, FLECS_IBUILDER IBuilder, typename ... Components>
struct node_builder : IBuilder<Base, Components ...>
{
    using IBase = IBuilder<Base, Components ...>;

public:
    explicit node_builder(flecs::world_t* world, const char *name = nullptr)
        : IBase(&desc_)
        , desc_{}
        , world_(world)
    {
        ecs_entity_desc_t entity_desc = {};
        entity_desc.name = name;
        entity_desc.sep = "::";
        entity_desc.root_sep = "::";
        desc_.entity = ecs_entity_init(world_, &entity_desc);
    }

    template <typename Func>
    T run(Func&& func) {
        using Delegate = typename _::run_delegate<
            typename std::decay<Func>::type>;

        auto ctx = FLECS_NEW(Delegate)(FLECS_FWD(func));
        desc_.run = Delegate::run;
        desc_.run_ctx = ctx;
        desc_.run_ctx_free = _::free_obj<Delegate>;
        return T(world_, &desc_);
    }

    template <typename Func, typename EachFunc>
    T run(Func&& func, EachFunc&& each_func) {
        using Delegate = typename _::run_delegate<
            typename std::decay<Func>::type>;

        auto ctx = FLECS_NEW(Delegate)(FLECS_FWD(func));
        desc_.run = Delegate::run;
        desc_.run_ctx = ctx;
        desc_.run_ctx_free = _::free_obj<Delegate>;
        return each(FLECS_FWD(each_func));
    }

    template <typename Func>
    T each(Func&& func) {
        using Delegate = typename _::each_delegate<
            typename std::decay<Func>::type, Components...>;
        auto ctx = FLECS_NEW(Delegate)(FLECS_FWD(func));
        desc_.callback = Delegate::run;
        desc_.callback_ctx = ctx;
        desc_.callback_ctx_free = _::free_obj<Delegate>;
        return T(world_, &desc_);
    }

protected:
    flecs::world_t* world_v() override { return world_; }
    TDesc desc_;
    flecs::world_t *world_;
};

#undef FLECS_IBUILDER

} // namespace _
} // namespace flecs

/**
 * @file addons/cpp/mixins/observer/builder_i.hpp
 * @brief Observer builder interface.
 */

#pragma once


namespace flecs {

/** Observer builder interface.
 * 
 * @ingroup cpp_observers
 */
template<typename Base, typename ... Components>
struct observer_builder_i : query_builder_i<Base, Components ...> {
    using BaseClass = query_builder_i<Base, Components ...>;
    observer_builder_i()
        : BaseClass(nullptr)
        , desc_(nullptr)
        , event_count_(0) { }

    observer_builder_i(ecs_observer_desc_t *desc) 
        : BaseClass(&desc->query)
        , desc_(desc)
        , event_count_(0) { }

    /** Specify the event(s) for when the observer should run.
     * @param evt The event.
     */
    Base& event(entity_t evt) {
        desc_->events[event_count_ ++] = evt;
        return *this;
    }

    /** Specify the event(s) for when the observer should run.
     * @tparam E The event.
     */
    template <typename E>
    Base& event() {
        desc_->events[event_count_ ++] = _::type<E>().id(world_v());
        return *this;
    }

    /** Invoke observer for anything that matches its query on creation */
    Base& yield_existing(bool value = true) {
        desc_->yield_existing = value;
        return *this;
    }

    /** Set observer flags */
    Base& observer_flags(ecs_flags32_t flags) {
        desc_->flags_ |= flags;
        return *this;
    }

    /** Set observer context */
    Base& ctx(void *ptr) {
        desc_->ctx = ptr;
        return *this;
    }

    /** Set observer run callback */
    Base& run(ecs_iter_action_t action) {
        desc_->run = action;
        return *this;
    }

protected:
    virtual flecs::world_t* world_v() override = 0;

private:
    operator Base&() {
        return *static_cast<Base*>(this);
    }

    ecs_observer_desc_t *desc_;
    int32_t event_count_;
};

}


namespace flecs {
namespace _ {
    template <typename ... Components>
    using observer_builder_base = node_builder<
        observer, ecs_observer_desc_t, observer_builder<Components...>, 
        observer_builder_i, Components ...>;
}

/** Observer builder.
 * 
 * @ingroup cpp_observers
 */
template <typename ... Components>
struct observer_builder final : _::observer_builder_base<Components...> {
    observer_builder(flecs::world_t* world, const char *name = nullptr)
        : _::observer_builder_base<Components...>(world, name)
    {
        _::sig<Components...>(world).populate(this);
    }
};

}


namespace flecs 
{

struct observer final : entity
{
    using entity::entity;

    explicit observer() : entity() { }

    observer(flecs::world_t *world, ecs_observer_desc_t *desc) {
        world_ = world;
        id_ = ecs_observer_init(world, desc);
    }

    void ctx(void *ctx) {
        ecs_observer_desc_t desc = {};
        desc.entity = id_;
        desc.ctx = ctx;
        ecs_observer_init(world_, &desc);
    }

    void* ctx() const {
        return ecs_observer_get(world_, id_)->ctx;
    }

    flecs::query<> query() const {
        return flecs::query<>(ecs_observer_get(world_, id_)->query);
    }
};

// Mixin implementation
inline observer world::observer(flecs::entity e) const {
    return flecs::observer(world_, e);
}

template <typename... Comps, typename... Args>
inline observer_builder<Comps...> world::observer(Args &&... args) const {
    return flecs::observer_builder<Comps...>(world_, FLECS_FWD(args)...);
}

} // namespace flecs

/**
 * @file addons/cpp/mixins/event/impl.hpp
 * @brief Event implementation.
 */

#pragma once


namespace flecs 
{

// Mixin implementation

inline flecs::event_builder world::event(flecs::entity_t evt) const {
    return flecs::event_builder(world_, evt);
}

template <typename E>
inline flecs::event_builder_typed<E> world::event() const {
    return flecs::event_builder_typed<E>(world_, _::type<E>().id(world_));
}

namespace _ {
    inline void entity_observer_create(
        flecs::world_t *world,
        flecs::entity_t event,
        flecs::entity_t entity,
        ecs_iter_action_t callback,
        void *callback_ctx,
        ecs_ctx_free_t callback_ctx_free) 
    {
        ecs_observer_desc_t desc = {};
        desc.events[0] = event;
        desc.query.terms[0].id = EcsAny;
        desc.query.terms[0].src.id = entity;
        desc.callback = callback;
        desc.callback_ctx = callback_ctx;
        desc.callback_ctx_free = callback_ctx_free;

        flecs::entity_t o = ecs_observer_init(world, &desc);
        ecs_add_pair(world, o, EcsChildOf, entity);
    }

    template <typename Func>
    struct entity_observer_factory {
        template <typename Evt, if_t<is_empty<Evt>::value> = 0>
        static void create(
            flecs::world_t *world,
            flecs::entity_t entity,
            Func&& f)
        {
            using Delegate = _::entity_observer_delegate<Func>;
            auto ctx = FLECS_NEW(Delegate)(FLECS_FWD(f));
            entity_observer_create(world, _::type<Evt>::id(world), entity, Delegate::run, ctx, _::free_obj<Delegate>);
        }

        template <typename Evt, if_not_t<is_empty<Evt>::value> = 0>
        static void create(
            flecs::world_t *world,
            flecs::entity_t entity,
            Func&& f)
        {
            using Delegate = _::entity_payload_observer_delegate<Func, Evt>;
            auto ctx = FLECS_NEW(Delegate)(FLECS_FWD(f));
            entity_observer_create(world, _::type<Evt>::id(world), entity, Delegate::run, ctx, _::free_obj<Delegate>);
        }
    };
}

template <typename Self>
template <typename Func>
inline const Self& entity_builder<Self>::observe(flecs::entity_t evt, Func&& f) const {
    using Delegate = _::entity_observer_delegate<Func>;
    auto ctx = FLECS_NEW(Delegate)(FLECS_FWD(f));

    _::entity_observer_create(world_, evt, id_, Delegate::run, ctx, _::free_obj<Delegate>);

    return to_base();
}

template <typename Self>
template <typename Evt, typename Func>
inline const Self& entity_builder<Self>::observe(Func&& f) const {
    _::entity_observer_factory<Func>::template create<Evt>(
        world_, id_, FLECS_FWD(f));
    return to_base();
}

template <typename Self>
template <typename Func>
inline const Self& entity_builder<Self>::observe(Func&& f) const {
    return this->observe<_::event_from_func_t<Func>>(FLECS_FWD(f));
}

inline void entity_view::emit(flecs::entity evt) const {
    this->emit(evt.id());
}

inline void entity_view::enqueue(flecs::entity evt) const {
    this->enqueue(evt.id());
}

} // namespace flecs

/**
 * @file addons/cpp/mixins/enum/impl.hpp
 * @brief Enum implementation.
 */

#pragma once

namespace flecs {

template <typename E>
inline E entity_view::to_constant() const {
#ifdef FLECS_META
    using U = typename std::underlying_type<E>::type;
    const E* ptr = static_cast<const E*>(ecs_get_id(world_, id_, 
        ecs_pair(flecs::Constant, _::type<U>::id(world_))));
    ecs_assert(ptr != NULL, ECS_INVALID_PARAMETER, "entity is not a constant");
    return ptr[0];
#else
    ecs_assert(false, ECS_UNSUPPORTED,
        "operation not supported without FLECS_META addon");
    return E();
#endif
}

template <typename E, if_t< is_enum<E>::value >>
inline flecs::entity world::to_entity(E constant) const {
    const auto& et = enum_type<E>(world_);
    return flecs::entity(world_, et.entity(constant));
}

}
#ifdef FLECS_MODULE
/**
 * @file addons/cpp/mixins/module/impl.hpp
 * @brief Module implementation.
 */

#pragma once

namespace flecs {

namespace _ {

template <typename T>
ecs_entity_t do_import(world& world, const char *symbol) {
    ecs_trace("#[magenta]import#[reset] %s", _::type_name<T>());
    ecs_log_push();

    ecs_entity_t scope = ecs_set_scope(world, 0);

    // Initialize module component type & don't allow it to be registered as a
    // tag, as this would prevent calling emplace()
    auto c_ = component<T>(world, nullptr, false);

    // Make module component sparse so that it'll never move in memory. This
    // guarantees that a module destructor can be reliably used to cleanup
    // module resources.
    c_.add(flecs::Sparse);

    ecs_set_scope(world, c_);
    world.emplace<T>(world);
    ecs_set_scope(world, scope);

    ecs_add_id(world, c_, EcsModule);

    // It should now be possible to lookup the module
    ecs_entity_t m = ecs_lookup_symbol(world, symbol, false, false);
    ecs_assert(m != 0, ECS_MODULE_UNDEFINED, symbol);
    ecs_assert(m == c_, ECS_INTERNAL_ERROR, NULL);

    ecs_log_pop();

    return m;
}

template <typename T>
flecs::entity import(world& world) {
    const char *symbol = _::symbol_name<T>();

    ecs_entity_t m = ecs_lookup_symbol(world, symbol, true, false);

    if (!_::type<T>::registered(world)) {
        /* Module is registered with world, initialize static data */
        if (m) {
            _::type<T>::init_builtin(world, m, false);

        /* Module is not yet registered, register it now */
        } else {
            m = _::do_import<T>(world, symbol);
        }

    /* Module has been registered, but could have been for another world. Import
     * if module hasn't been registered for this world. */
    } else if (!m) {
        m = _::do_import<T>(world, symbol);
    }

    return flecs::entity(world, m);
}

}

/**
 * @defgroup cpp_addons_modules Modules
 * @ingroup cpp_addons
 * Modules organize components, systems and more in reusable units of code.
 *
 * @{
 */

template <typename Module>
inline flecs::entity world::module(const char *name) const {
    flecs::entity result = this->entity(_::type<Module>::register_id(
        world_, nullptr, false));

    if (name) {
        flecs::entity prev_parent = result.parent();
        ecs_add_path_w_sep(world_, result, 0, name, "::", "::");
        flecs::entity parent = result.parent();
        if (prev_parent != parent) {
            // Module was reparented, cleanup old parent(s)
            flecs::entity cur = prev_parent, next;
            while (cur) {
                next = cur.parent();

                ecs_iter_t it = ecs_each_id(world_, ecs_pair(EcsChildOf, cur));
                if (!ecs_iter_is_true(&it)) {
                    cur.destruct();

                    // Prevent increasing the generation count of the temporary
                    // parent. This allows entities created during 
                    // initialization to keep non-recycled ids.
                    this->set_version(cur);
                }

                cur = next;
            }
        }
    }

    return result;
}

template <typename Module>
inline flecs::entity world::import() {
    return flecs::_::import<Module>(*this);
}

/** @} */

}

#endif
#ifdef FLECS_SYSTEM
/**
 * @file addons/cpp/mixins/system/impl.hpp
 * @brief System module implementation.
 */

#pragma once

/**
 * @file addons/cpp/mixins/system/builder.hpp
 * @brief System builder.
 */

#pragma once

/**
 * @file addons/cpp/mixins/system/builder_i.hpp
 * @brief System builder interface.
 */

#pragma once


namespace flecs 
{

/** System builder interface.
 * 
 * @ingroup cpp_addons_systems
 */
template<typename Base, typename ... Components>
struct system_builder_i : query_builder_i<Base, Components ...> {
private:
    using BaseClass = query_builder_i<Base, Components ...>;

public:
    system_builder_i(ecs_system_desc_t *desc) 
        : BaseClass(&desc->query)
        , desc_(desc) { }

    /** Specify in which phase the system should run.
     *
     * @param phase The phase.
     */
    Base& kind(entity_t phase) {
        flecs::entity_t cur_phase = ecs_get_target(
            world_v(), desc_->entity, EcsDependsOn, 0);
        if (cur_phase) {
            ecs_remove_id(world_v(), desc_->entity, ecs_dependson(cur_phase));
            ecs_remove_id(world_v(), desc_->entity, cur_phase);
        }
        if (phase) {
            ecs_add_id(world_v(), desc_->entity, ecs_dependson(phase));
            ecs_add_id(world_v(), desc_->entity, phase);
        }
        return *this;
    }

    template <typename E, if_t<is_enum<E>::value> = 0>
    Base& kind(E phase)
    {
        const auto& et = enum_type<E>(this->world_v());
        flecs::entity_t target = et.entity(phase);
        return this->kind(target);
    }

    /** Specify in which phase the system should run.
     *
     * @tparam Phase The phase.
     */
    template <typename Phase>
    Base& kind() {
        return this->kind(_::type<Phase>::id(world_v()));
    }

    /** Specify whether system can run on multiple threads.
     *
     * @param value If false system will always run on a single thread.
     */
    Base& multi_threaded(bool value = true) {
        desc_->multi_threaded = value;
        return *this;
    }

    /** Specify whether system should be ran in staged context.
     *
     * @param value If false system will always run staged.
     */
    Base& immediate(bool value = true) {
        desc_->immediate = value;
        return *this;
    }

    /** Set system interval.
     * This operation will cause the system to be ran at the specified interval.
     *
     * The timer is synchronous, and is incremented each frame by delta_time.
     *
     * @param interval The interval value.
     */
    Base& interval(ecs_ftime_t interval) {
        desc_->interval = interval;
        return *this;
    }

    /** Set system rate.
     * This operation will cause the system to be ran at a multiple of the 
     * provided tick source. The tick source may be any entity, including
     * another system.
     *
     * @param tick_source The tick source.
     * @param rate The multiple at which to run the system.
     */
    Base& rate(const entity_t tick_source, int32_t rate) {
        desc_->rate = rate;
        desc_->tick_source = tick_source;
        return *this;
    }

    /** Set system rate.
     * This operation will cause the system to be ran at a multiple of the 
     * frame tick frequency. If a tick source was provided, this just updates
     * the rate of the system.
     *
     * @param rate The multiple at which to run the system.
     */
    Base& rate(int32_t rate) {
        desc_->rate = rate;
        return *this;
    }

    /** Set tick source.
     * This operation sets a shared tick source for the system.
     *
     * @tparam T The type associated with the singleton tick source to use for the system.
     */
    template<typename T>
    Base& tick_source() {
        desc_->tick_source = _::type<T>::id(world_v());
        return *this;
    }

    /** Set tick source.
     * This operation sets a shared tick source for the system.
     *
     * @param tick_source The tick source to use for the system.
     */
    Base& tick_source(flecs::entity_t tick_source) {
        desc_->tick_source = tick_source;
        return *this;
    }

    /** Set system context */
    Base& ctx(void *ptr) {
        desc_->ctx = ptr;
        return *this;
    }

    /** Set system run callback */
    Base& run(ecs_iter_action_t action) {
        desc_->run = action;
        return *this;
    }

protected:
    virtual flecs::world_t* world_v() override = 0;

private:
    operator Base&() {
        return *static_cast<Base*>(this);
    }

    ecs_system_desc_t *desc_;
};

}


namespace flecs {
namespace _ {
    template <typename ... Components>
    using system_builder_base = node_builder<
        system, ecs_system_desc_t, system_builder<Components...>, 
        system_builder_i, Components ...>;
}

/** System builder.
 * 
 * @ingroup cpp_addons_systems
 */
template <typename ... Components>
struct system_builder final : _::system_builder_base<Components...> {
    system_builder(flecs::world_t* world, const char *name = nullptr)
        : _::system_builder_base<Components...>(world, name)
    {
        _::sig<Components...>(world).populate(this);

#ifdef FLECS_PIPELINE
        ecs_add_id(world, this->desc_.entity, ecs_dependson(flecs::OnUpdate));
        ecs_add_id(world, this->desc_.entity, flecs::OnUpdate);
#endif
    }
};

}


namespace flecs 
{

struct system_runner_fluent {
    system_runner_fluent(
        world_t *world, 
        entity_t id, 
        int32_t stage_current, 
        int32_t stage_count, 
        ecs_ftime_t delta_time, 
        void *param)
        : stage_(world)
        , id_(id)
        , delta_time_(delta_time)
        , param_(param)
        , stage_current_(stage_current)
        , stage_count_(stage_count) { }

    system_runner_fluent& offset(int32_t offset) {
        offset_ = offset;
        return *this;
    }

    system_runner_fluent& limit(int32_t limit) {
        limit_ = limit;
        return *this;
    }

    system_runner_fluent& stage(flecs::world& stage) {
        stage_ = stage.c_ptr();
        return *this;
    }

    ~system_runner_fluent() {
        if (stage_count_) {
            ecs_run_worker(
                stage_, id_, stage_current_, stage_count_, delta_time_,
                param_);            
        } else {
            ecs_run(stage_, id_, delta_time_, param_);
        }
    }

private:
    world_t *stage_;
    entity_t id_;
    ecs_ftime_t delta_time_;
    void *param_;
    int32_t offset_;
    int32_t limit_;
    int32_t stage_current_;
    int32_t stage_count_;
};

struct system final : entity
{
    using entity::entity;

    explicit system() {
        id_ = 0;
        world_ = nullptr;
    }

    explicit system(flecs::world_t *world, ecs_system_desc_t *desc) {
        world_ = world;
        id_ = ecs_system_init(world, desc);
    }

    void ctx(void *ctx) {
        ecs_system_desc_t desc = {};
        desc.entity = id_;
        desc.ctx = ctx;
        ecs_system_init(world_, &desc);
    }

    void* ctx() const {
        return ecs_system_get(world_, id_)->ctx;
    }

    flecs::query<> query() const {
        return flecs::query<>(ecs_system_get(world_, id_)->query);
    }

    system_runner_fluent run(ecs_ftime_t delta_time = 0.0f, void *param = nullptr) const {
        return system_runner_fluent(world_, id_, 0, 0, delta_time, param);
    }

    system_runner_fluent run_worker(
        int32_t stage_current, 
        int32_t stage_count, 
        ecs_ftime_t delta_time = 0.0f, 
        void *param = nullptr) const 
    {
        return system_runner_fluent(
            world_, id_, stage_current, stage_count, delta_time, param);
    }

#   ifdef FLECS_TIMER
/**
 * @file addons/cpp/mixins/timer/system_mixin.inl
 * @brief Timer module system mixin.
 */

/**
 * @memberof flecs::system
 * @ingroup cpp_addons_timer
 *
 * @{
 */

/** Set interval.
 * @see ecs_set_interval
 */
void interval(ecs_ftime_t interval);

/** Get interval.
 * @see ecs_get_interval.
 */
ecs_ftime_t interval();

/** Set timeout.
 * @see ecs_set_timeout
 */
void timeout(ecs_ftime_t timeout);

/** Get timeout.
 * @see ecs_get_timeout
 */
ecs_ftime_t timeout();

/** Set system rate (system is its own tick source).
 * @see ecs_set_rate
 */
void rate(int32_t rate);

/** Start timer.
 * @see ecs_start_timer
 */
void start();

/** Stop timer.
 * @see ecs_start_timer
 */
void stop();

/** Set external tick source.
 * @see ecs_set_tick_source
 */
template<typename T>
void set_tick_source();

/** Set external tick source.
 * @see ecs_set_tick_source
 */
void set_tick_source(flecs::entity e);

/** @} */

#   endif

};

// Mixin implementation
inline system world::system(flecs::entity e) const {
    return flecs::system(world_, e);
}

template <typename... Comps, typename... Args>
inline system_builder<Comps...> world::system(Args &&... args) const {
    return flecs::system_builder<Comps...>(world_, FLECS_FWD(args)...);
}

namespace _ {

inline void system_init(flecs::world& world) {
    world.component<TickSource>("flecs::system::TickSource");
}

} // namespace _
} // namespace flecs

#endif
#ifdef FLECS_PIPELINE
/**
 * @file addons/cpp/mixins/pipeline/impl.hpp
 * @brief Pipeline module implementation.
 */

#pragma once

/**
 * @file addons/cpp/mixins/pipeline/builder.hpp
 * @brief Pipeline builder.
 */

#pragma once

/**
 * @file addons/cpp/mixins/pipeline/builder_i.hpp
 * @brief Pipeline builder interface.
 */

#pragma once


namespace flecs {

/** Pipeline builder interface.
 * 
 * @ingroup cpp_pipelines
 */
template<typename Base>
struct pipeline_builder_i : query_builder_i<Base> {
    pipeline_builder_i(ecs_pipeline_desc_t *desc, int32_t term_index = 0) 
        : query_builder_i<Base>(&desc->query, term_index)
        , desc_(desc) { }

private:
    ecs_pipeline_desc_t *desc_;
};

}


namespace flecs {
namespace _ {
    template <typename ... Components>
    using pipeline_builder_base = builder<
        pipeline, ecs_pipeline_desc_t, pipeline_builder<Components...>, 
        pipeline_builder_i, Components ...>;
}

/** Pipeline builder.
 * 
 * @ingroup cpp_pipelines
 */
template <typename ... Components>
struct pipeline_builder final : _::pipeline_builder_base<Components...> {
    pipeline_builder(flecs::world_t* world, flecs::entity_t id = 0)
        : _::pipeline_builder_base<Components...>(world)
    {
        _::sig<Components...>(world).populate(this);
        this->desc_.entity = id;
    }
};

}


namespace flecs {

template <typename ... Components>
struct pipeline : entity {
    pipeline(world_t *world, ecs_pipeline_desc_t *desc) 
        : entity(world)
    {
        id_ = ecs_pipeline_init(world, desc);

        if (!id_) {
            ecs_abort(ECS_INVALID_PARAMETER, NULL);
        }
    }
};

inline flecs::pipeline_builder<> world::pipeline() const {
    return flecs::pipeline_builder<>(world_);
}

template <typename Pipeline, if_not_t< is_enum<Pipeline>::value >>
inline flecs::pipeline_builder<> world::pipeline() const {
    return flecs::pipeline_builder<>(world_, _::type<Pipeline>::id(world_));
}

inline void world::set_pipeline(const flecs::entity pip) const {
    return ecs_set_pipeline(world_, pip);
}

template <typename Pipeline>
inline void world::set_pipeline() const {
    return ecs_set_pipeline(world_, _::type<Pipeline>::id(world_));
}

inline flecs::entity world::get_pipeline() const {
    return flecs::entity(world_, ecs_get_pipeline(world_));
}

inline bool world::progress(ecs_ftime_t delta_time) const {
    return ecs_progress(world_, delta_time);
}

inline void world::run_pipeline(const flecs::entity_t pip, ecs_ftime_t delta_time) const {
    return ecs_run_pipeline(world_, pip, delta_time);
}

template <typename Pipeline, if_not_t< is_enum<Pipeline>::value >>
inline void world::run_pipeline(ecs_ftime_t delta_time) const {
    return ecs_run_pipeline(world_, _::type<Pipeline>::id(world_), delta_time);
}

inline void world::set_time_scale(ecs_ftime_t mul) const {
    ecs_set_time_scale(world_, mul);
}

inline void world::set_target_fps(ecs_ftime_t target_fps) const {
    ecs_set_target_fps(world_, target_fps);
}

inline void world::reset_clock() const {
    ecs_reset_clock(world_);
}

inline void world::set_threads(int32_t threads) const {
    ecs_set_threads(world_, threads);
}

inline int32_t world::get_threads() const {
    return ecs_get_stage_count(world_);
}

inline void world::set_task_threads(int32_t task_threads) const {
    ecs_set_task_threads(world_, task_threads);
}

inline bool world::using_task_threads() const {
    return ecs_using_task_threads(world_);
}

}

#endif
#ifdef FLECS_TIMER
/**
 * @file addons/cpp/mixins/timer/impl.hpp
 * @brief Timer module implementation.
 */

#pragma once

namespace flecs {

// Timer class
struct timer final : entity {
    using entity::entity;

    timer& interval(ecs_ftime_t interval) {
        ecs_set_interval(world_, id_, interval);
        return *this;
    }

    ecs_ftime_t interval() {
        return ecs_get_interval(world_, id_);
    }

    timer& timeout(ecs_ftime_t timeout) {
        ecs_set_timeout(world_, id_, timeout);
        return *this;
    }

    ecs_ftime_t timeout() {
        return ecs_get_timeout(world_, id_);
    }

    timer& rate(int32_t rate, flecs::entity_t tick_source = 0) {
        ecs_set_rate(world_, id_, rate, tick_source);
        return *this;
    }

    void start() {
        ecs_start_timer(world_, id_);
    }

    void stop() {
        ecs_stop_timer(world_, id_);
    }
};

template <typename T>
inline flecs::timer world::timer() const {
    return flecs::timer(world_, _::type<T>::id(world_));
}

template <typename... Args>
inline flecs::timer world::timer(Args &&... args) const {
    return flecs::timer(world_, FLECS_FWD(args)...);
}

inline void world::randomize_timers() const {
    ecs_randomize_timers(world_);
}

inline void system::interval(ecs_ftime_t interval) {
    ecs_set_interval(world_, id_, interval);
}

inline ecs_ftime_t system::interval() {
    return ecs_get_interval(world_, id_);
}

inline void system::timeout(ecs_ftime_t timeout) {
    ecs_set_timeout(world_, id_, timeout);
}

inline ecs_ftime_t system::timeout() {
    return ecs_get_timeout(world_, id_);
}

inline void system::rate(int32_t rate) {
    ecs_set_rate(world_, id_, rate, 0);
}

inline void system::start() {
    ecs_start_timer(world_, id_);
}

inline void system::stop() {
    ecs_stop_timer(world_, id_);
}

template<typename T>
inline void system::set_tick_source() {
    ecs_set_tick_source(world_, id_, _::type<T>::id(world_));
}

inline void system::set_tick_source(flecs::entity e) {
    ecs_set_tick_source(world_, id_, e);
}

namespace _ {

inline void timer_init(flecs::world& world) {
    world.component<RateFilter>("flecs::timer::RateFilter");
    world.component<Timer>("flecs::timer::Timer");
}

}
}

#endif
#ifdef FLECS_DOC
/**
 * @file addons/cpp/mixins/doc/impl.hpp
 * @brief Doc mixin implementation.
 */

#pragma once

namespace flecs {
namespace doc {

/** Get UUID for an entity.
 *
 * @see ecs_doc_get_uuid()
 * @see flecs::doc::set_uuid()
 * @see flecs::entity_view::doc_uuid()
 *
 * @ingroup cpp_addons_doc
 */
inline const char* get_uuid(const flecs::entity_view& e) {
    return ecs_doc_get_uuid(e.world(), e);
}

/** Get human readable name for an entity.
 *
 * @see ecs_doc_get_name()
 * @see flecs::doc::set_name()
 * @see flecs::entity_view::doc_name()
 *
 * @ingroup cpp_addons_doc
 */
inline const char* get_name(const flecs::entity_view& e) {
    return ecs_doc_get_name(e.world(), e);
}

/** Get brief description for an entity.
 *
 * @see ecs_doc_get_brief()
 * @see flecs::doc::set_brief()
 * @see flecs::entity_view::doc_brief()
 *
 * @ingroup cpp_addons_doc
 */
inline const char* get_brief(const flecs::entity_view& e) {
    return ecs_doc_get_brief(e.world(), e);
}

/** Get detailed description for an entity.
 *
 * @see ecs_doc_get_detail()
 * @see flecs::doc::set_detail()
 * @see flecs::entity_view::doc_detail()
 *
 * @ingroup cpp_addons_doc
 */
inline const char* get_detail(const flecs::entity_view& e) {
    return ecs_doc_get_detail(e.world(), e);
}

/** Get link to external documentation for an entity.
 *
 * @see ecs_doc_get_link()
 * @see flecs::doc::set_link()
 * @see flecs::entity_view::doc_link()
 *
 * @ingroup cpp_addons_doc
 */
inline const char* get_link(const flecs::entity_view& e) {
    return ecs_doc_get_link(e.world(), e);
}

/** Get color for an entity.
 *
 * @see ecs_doc_get_color()
 * @see flecs::doc::set_color()
 * @see flecs::entity_view::doc_color()
 *
 * @ingroup cpp_addons_doc
 */
inline const char* get_color(const flecs::entity_view& e) {
    return ecs_doc_get_color(e.world(), e);
}

/** Set UUID for an entity.
 *
 * @see ecs_doc_set_uuid()
 * @see flecs::doc::get_uuid()
 * @see flecs::entity_builder::set_doc_uuid()
 *
 * @ingroup cpp_addons_doc
 */
inline void set_uuid(flecs::entity& e, const char *uuid) {
    ecs_doc_set_uuid(e.world(), e, uuid);
}

/** Set human readable name for an entity.
 *
 * @see ecs_doc_set_name()
 * @see flecs::doc::get_name()
 * @see flecs::entity_builder::set_doc_name()
 *
 * @ingroup cpp_addons_doc
 */
inline void set_name(flecs::entity& e, const char *name) {
    ecs_doc_set_name(e.world(), e, name);
}

/** Set brief description for an entity.
 *
 * @see ecs_doc_set_brief()
 * @see flecs::doc::get_brief()
 * @see flecs::entity_builder::set_doc_brief()
 *
 * @ingroup cpp_addons_doc
 */
inline void set_brief(flecs::entity& e, const char *description) {
    ecs_doc_set_brief(e.world(), e, description);
}

/** Set detailed description for an entity.
 *
 * @see ecs_doc_set_detail()
 * @see flecs::doc::get_detail()
 * @see flecs::entity_builder::set_doc_detail()
 *
 * @ingroup cpp_addons_doc
 */
inline void set_detail(flecs::entity& e, const char *description) {
    ecs_doc_set_detail(e.world(), e, description);
}

/** Set link to external documentation for an entity.
 *
 * @see ecs_doc_set_link()
 * @see flecs::doc::get_link()
 * @see flecs::entity_builder::set_doc_link()
 *
 * @ingroup cpp_addons_doc
 */
inline void set_link(flecs::entity& e, const char *link) {
    ecs_doc_set_link(e.world(), e, link);
}

/** Set color for an entity.
 *
 * @see ecs_doc_set_color()
 * @see flecs::doc::get_color()
 * @see flecs::entity_builder::set_doc_color()
 *
 * @ingroup cpp_addons_doc
 */
inline void set_color(flecs::entity& e, const char *color) {
    ecs_doc_set_color(e.world(), e, color);
}

/** @private */
namespace _ {

/** @private */
inline void init(flecs::world& world) {
    world.component<doc::Description>("flecs::doc::Description");
}

} // namespace _
} // namespace doc
} // namespace flecs

#endif
#ifdef FLECS_DOC
#endif
#ifdef FLECS_REST
/**
 * @file addons/cpp/mixins/rest/impl.hpp
 * @brief Rest module implementation.
 */

#pragma once

namespace flecs {
namespace rest {
namespace _ {

inline void init(flecs::world& world) {
    world.component<Rest>("flecs::rest::Rest");
}
 
} // namespace _
} // namespace rest
} // namespace flecs

#endif
#ifdef FLECS_META
/**
 * @file addons/cpp/mixins/meta/impl.hpp
 * @brief Meta implementation.
 */

#pragma once

FLECS_ENUM_LAST(flecs::meta::type_kind_t, flecs::meta::TypeKindLast)
FLECS_ENUM_LAST(flecs::meta::primitive_kind_t, flecs::meta::PrimitiveKindLast)

namespace flecs {
namespace meta {
namespace _ {

/* Type support for entity wrappers */
template <typename EntityType>
inline flecs::opaque<EntityType> flecs_entity_support(flecs::world&) {
    return flecs::opaque<EntityType>()
        .as_type(flecs::Entity)
        .serialize([](const flecs::serializer *ser, const EntityType *data) {
            flecs::entity_t id = data->id();
            return ser->value(flecs::Entity, &id);
        })
        .assign_entity(
            [](EntityType *dst, flecs::world_t *world, flecs::entity_t e) {
                *dst = EntityType(world, e);
            });
}

inline void init(flecs::world& world) {
    world.component<bool_t>("flecs::meta::bool");
    world.component<char_t>("flecs::meta::char");
    world.component<u8_t>("flecs::meta::u8");
    world.component<u16_t>("flecs::meta::u16");
    world.component<u32_t>("flecs::meta::u32");
    world.component<u64_t>("flecs::meta::u64");
    world.component<i8_t>("flecs::meta::i8");
    world.component<i16_t>("flecs::meta::i16");
    world.component<i32_t>("flecs::meta::i32");
    world.component<i64_t>("flecs::meta::i64");
    world.component<f32_t>("flecs::meta::f32");
    world.component<f64_t>("flecs::meta::f64");

    world.component<type_kind_t>("flecs::meta::type_kind");
    world.component<primitive_kind_t>("flecs::meta::primitive_kind");
    world.component<member_t>("flecs::meta::member_t");
    world.component<enum_constant_t>("flecs::meta::enum_constant");
    world.component<bitmask_constant_t>("flecs::meta::bitmask_constant");

    world.component<Type>("flecs::meta::type");
    world.component<TypeSerializer>("flecs::meta::TypeSerializer");
    world.component<Primitive>("flecs::meta::primitive");
    world.component<Enum>("flecs::meta::enum");
    world.component<Bitmask>("flecs::meta::bitmask");
    world.component<Member>("flecs::meta::member");
    world.component<MemberRanges>("flecs::meta::member_ranges");
    world.component<Struct>("flecs::meta::struct");
    world.component<Array>("flecs::meta::array");
    world.component<Vector>("flecs::meta::vector");

    world.component<Unit>("flecs::meta::unit");

    // To support member<uintptr_t> and member<intptr_t> register components
    // (that do not have conflicting symbols with builtin ones) for platform
    // specific types.

    if (!flecs::is_same<i32_t, iptr_t>() && !flecs::is_same<i64_t, iptr_t>()) {
        flecs::_::type<iptr_t>::init_builtin(world, flecs::Iptr, true);
        // Remove symbol to prevent validation errors, as it doesn't match with 
        // the typename
        ecs_remove_pair(world, flecs::Iptr, ecs_id(EcsIdentifier), EcsSymbol);
    }

    if (!flecs::is_same<u32_t, uptr_t>() && !flecs::is_same<u64_t, uptr_t>()) {
        flecs::_::type<uptr_t>::init_builtin(world, flecs::Uptr, true);
        // Remove symbol to prevent validation errors, as it doesn't match with 
        // the typename
        ecs_remove_pair(world, flecs::Uptr, ecs_id(EcsIdentifier), EcsSymbol);
    }

    // Register opaque type support for C++ entity wrappers
    world.entity("::flecs::cpp").add(flecs::Module).scope([&]{
        world.component<flecs::entity_view>()
            .opaque(flecs_entity_support<flecs::entity_view>);
        world.component<flecs::entity>()
            .opaque(flecs_entity_support<flecs::entity>);
    });
}

} // namespace _

} // namespace meta


inline flecs::entity cursor::get_type() const {
    return flecs::entity(cursor_.world, ecs_meta_get_type(&cursor_));
}

inline flecs::entity cursor::get_unit() const {
    return flecs::entity(cursor_.world, ecs_meta_get_unit(&cursor_));
}

inline flecs::entity cursor::get_entity() const {
    return flecs::entity(cursor_.world, ecs_meta_get_entity(&cursor_));
}

/** Create primitive type */
inline flecs::entity world::primitive(flecs::meta::primitive_kind_t kind) {
    ecs_primitive_desc_t desc = {};
    desc.kind = kind;
    flecs::entity_t eid = ecs_primitive_init(world_, &desc);
    ecs_assert(eid != 0, ECS_INVALID_OPERATION, NULL);
    return flecs::entity(world_, eid);
}

/** Create array type. */
inline flecs::entity world::array(flecs::entity_t elem_id, int32_t array_count) {
    ecs_array_desc_t desc = {};
    desc.type = elem_id;
    desc.count = array_count;
    flecs::entity_t eid = ecs_array_init(world_, &desc);
    ecs_assert(eid != 0, ECS_INVALID_OPERATION, NULL);
    return flecs::entity(world_, eid);
}

/** Create array type. */
template <typename T>
inline flecs::entity world::array(int32_t array_count) {
    return this->array(_::type<T>::id(world_), array_count);
}

inline flecs::entity world::vector(flecs::entity_t elem_id) {
    ecs_vector_desc_t desc = {};
    desc.type = elem_id;
    flecs::entity_t eid = ecs_vector_init(world_, &desc);
    ecs_assert(eid != 0, ECS_INVALID_OPERATION, NULL);
    return flecs::entity(world_, eid);
}

template <typename T>
inline flecs::entity world::vector() {
    return this->vector(_::type<T>::id(world_));
}

} // namespace flecs

inline int ecs_serializer_t::value(ecs_entity_t type, const void *v) const {
    return this->value_(this, type, v);
}

template <typename T>
inline int ecs_serializer_t::value(const T& v) const {
    return this->value(flecs::_::type<T>::id(
        const_cast<flecs::world_t*>(this->world)), &v);
}

inline int ecs_serializer_t::member(const char *name) const {
    return this->member_(this, name);
}

#endif
#ifdef FLECS_UNITS
/**
 * @file addons/cpp/mixins/units/impl.hpp
 * @brief Units module implementation.
 */

#pragma once

namespace flecs {

inline units::units(flecs::world& world) {
    /* Import C module  */
    FlecsUnitsImport(world);

    /* Bridge between C++ types and flecs.units entities */
    world.module<units>();

    // Initialize world.entity(prefixes) scope
    world.entity<Prefixes>("::flecs::units::prefixes");

    // Initialize prefixes
    world.entity<Yocto>("::flecs::units::prefixes::Yocto");
    world.entity<Zepto>("::flecs::units::prefixes::Zepto");
    world.entity<Atto>("::flecs::units::prefixes::Atto");
    world.entity<Femto>("::flecs::units::prefixes::Femto");
    world.entity<Pico>("::flecs::units::prefixes::Pico");
    world.entity<Nano>("::flecs::units::prefixes::Nano");
    world.entity<Micro>("::flecs::units::prefixes::Micro");
    world.entity<Milli>("::flecs::units::prefixes::Milli");
    world.entity<Centi>("::flecs::units::prefixes::Centi");
    world.entity<Deci>("::flecs::units::prefixes::Deci");
    world.entity<Deca>("::flecs::units::prefixes::Deca");
    world.entity<Hecto>("::flecs::units::prefixes::Hecto");
    world.entity<Kilo>("::flecs::units::prefixes::Kilo");
    world.entity<Mega>("::flecs::units::prefixes::Mega");
    world.entity<Giga>("::flecs::units::prefixes::Giga");
    world.entity<Tera>("::flecs::units::prefixes::Tera");
    world.entity<Peta>("::flecs::units::prefixes::Peta");
    world.entity<Exa>("::flecs::units::prefixes::Exa");
    world.entity<Zetta>("::flecs::units::prefixes::Zetta");
    world.entity<Yotta>("::flecs::units::prefixes::Yotta");
    world.entity<Kibi>("::flecs::units::prefixes::Kibi");
    world.entity<Mebi>("::flecs::units::prefixes::Mebi");
    world.entity<Gibi>("::flecs::units::prefixes::Gibi");
    world.entity<Tebi>("::flecs::units::prefixes::Tebi");
    world.entity<Pebi>("::flecs::units::prefixes::Pebi");
    world.entity<Exbi>("::flecs::units::prefixes::Exbi");
    world.entity<Zebi>("::flecs::units::prefixes::Zebi");
    world.entity<Yobi>("::flecs::units::prefixes::Yobi");

    // Initialize quantities
    world.entity<Duration>("::flecs::units::Duration");
    world.entity<Time>("::flecs::units::Time");
    world.entity<Mass>("::flecs::units::Mass");
    world.entity<Force>("::flecs::units::Force");
    world.entity<ElectricCurrent>("::flecs::units::ElectricCurrent");
    world.entity<Amount>("::flecs::units::Amount");
    world.entity<LuminousIntensity>("::flecs::units::LuminousIntensity");
    world.entity<Length>("::flecs::units::Length");
    world.entity<Pressure>("::flecs::units::Pressure");
    world.entity<Speed>("::flecs::units::Speed");
    world.entity<Temperature>("::flecs::units::Temperature");
    world.entity<Data>("::flecs::units::Data");
    world.entity<DataRate>("::flecs::units::DataRate");
    world.entity<Angle>("::flecs::units::Angle");
    world.entity<Frequency>("::flecs::units::Frequency");
    world.entity<Uri>("::flecs::units::Uri");
    world.entity<Color>("::flecs::units::Color");

    // Initialize duration units
    world.entity<duration::PicoSeconds>(
        "::flecs::units::Duration::PicoSeconds");
    world.entity<duration::NanoSeconds>(
        "::flecs::units::Duration::NanoSeconds");
    world.entity<duration::MicroSeconds>(
        "::flecs::units::Duration::MicroSeconds");
    world.entity<duration::MilliSeconds>(
        "::flecs::units::Duration::MilliSeconds");
    world.entity<duration::Seconds>(
        "::flecs::units::Duration::Seconds");
    world.entity<duration::Minutes>(
        "::flecs::units::Duration::Minutes");
    world.entity<duration::Hours>(
        "::flecs::units::Duration::Hours");
    world.entity<duration::Days>(
        "::flecs::units::Duration::Days");

    // Initialize time units
    world.entity<time::Date>("::flecs::units::Time::Date");

    // Initialize mass units
    world.entity<mass::Grams>("::flecs::units::Mass::Grams");
    world.entity<mass::KiloGrams>("::flecs::units::Mass::KiloGrams");

    // Initialize current units
    world.entity<electric_current::Ampere>
    ("::flecs::units::ElectricCurrent::Ampere");  

    // Initialize amount units
    world.entity<amount::Mole>("::flecs::units::Amount::Mole");

    // Initialize luminous intensity units
    world.entity<luminous_intensity::Candela>(
        "::flecs::units::LuminousIntensity::Candela");

    // Initialize force units
    world.entity<force::Newton>("::flecs::units::Force::Newton");

    // Initialize length units
    world.entity<length::Meters>("::flecs::units::Length::Meters");
    world.entity<length::PicoMeters>("::flecs::units::Length::PicoMeters");
    world.entity<length::NanoMeters>("::flecs::units::Length::NanoMeters");
    world.entity<length::MicroMeters>("::flecs::units::Length::MicroMeters");
    world.entity<length::MilliMeters>("::flecs::units::Length::MilliMeters");
    world.entity<length::CentiMeters>("::flecs::units::Length::CentiMeters");
    world.entity<length::KiloMeters>("::flecs::units::Length::KiloMeters");
    world.entity<length::Miles>("::flecs::units::Length::Miles");
    world.entity<length::Pixels>("::flecs::units::Length::Pixels");

    // Initialize pressure units
    world.entity<pressure::Pascal>("::flecs::units::Pressure::Pascal");
    world.entity<pressure::Bar>("::flecs::units::Pressure::Bar");

    // Initialize speed units
    world.entity<speed::MetersPerSecond>(
        "::flecs::units::Speed::MetersPerSecond");
    world.entity<speed::KiloMetersPerSecond>(
        "::flecs::units::Speed::KiloMetersPerSecond");
    world.entity<speed::KiloMetersPerHour>(
        "::flecs::units::Speed::KiloMetersPerHour");
    world.entity<speed::MilesPerHour>(
        "::flecs::units::Speed::MilesPerHour");

    // Initialize temperature units
    world.entity<temperature::Kelvin>(
        "::flecs::units::Temperature::Kelvin");
    world.entity<temperature::Celsius>(
        "::flecs::units::Temperature::Celsius");
    world.entity<temperature::Fahrenheit>(
        "::flecs::units::Temperature::Fahrenheit");

    // Initialize data units
    world.entity<data::Bits>(
        "::flecs::units::Data::Bits");
    world.entity<data::KiloBits>(
        "::flecs::units::Data::KiloBits");
    world.entity<data::MegaBits>(
        "::flecs::units::Data::MegaBits");
    world.entity<data::GigaBits>(
        "::flecs::units::Data::GigaBits");
    world.entity<data::Bytes>(
        "::flecs::units::Data::Bytes");
    world.entity<data::KiloBytes>(
        "::flecs::units::Data::KiloBytes");
    world.entity<data::MegaBytes>(
        "::flecs::units::Data::MegaBytes");
    world.entity<data::GigaBytes>(
        "::flecs::units::Data::GigaBytes");
    world.entity<data::KibiBytes>(
        "::flecs::units::Data::KibiBytes");
    world.entity<data::MebiBytes>(
        "::flecs::units::Data::MebiBytes");
    world.entity<data::GibiBytes>(
        "::flecs::units::Data::GibiBytes");

    // Initialize datarate units
    world.entity<datarate::BitsPerSecond>(
        "::flecs::units::DataRate::BitsPerSecond");
    world.entity<datarate::KiloBitsPerSecond>(
        "::flecs::units::DataRate::KiloBitsPerSecond");
    world.entity<datarate::MegaBitsPerSecond>(
        "::flecs::units::DataRate::MegaBitsPerSecond");
    world.entity<datarate::GigaBitsPerSecond>(
        "::flecs::units::DataRate::GigaBitsPerSecond");
    world.entity<datarate::BytesPerSecond>(
        "::flecs::units::DataRate::BytesPerSecond");
    world.entity<datarate::KiloBytesPerSecond>(
        "::flecs::units::DataRate::KiloBytesPerSecond");
    world.entity<datarate::MegaBytesPerSecond>(
        "::flecs::units::DataRate::MegaBytesPerSecond");
    world.entity<datarate::GigaBytesPerSecond>(
        "::flecs::units::DataRate::GigaBytesPerSecond");

    // Initialize hertz units
    world.entity<frequency::Hertz>(
        "::flecs::units::Frequency::Hertz");
    world.entity<frequency::KiloHertz>(
        "::flecs::units::Frequency::KiloHertz");
    world.entity<frequency::MegaHertz>(
        "::flecs::units::Frequency::MegaHertz");
    world.entity<frequency::GigaHertz>(
        "::flecs::units::Frequency::GigaHertz");

    // Initialize uri units
    world.entity<uri::Hyperlink>(
        "::flecs::units::Uri::Hyperlink");
    world.entity<uri::Image>(
        "::flecs::units::Uri::Image");
    world.entity<uri::File>(
        "::flecs::units::Uri::File");

    // Initialize angles
    world.entity<angle::Radians>(
        "::flecs::units::Angle::Radians");
    world.entity<angle::Degrees>(
        "::flecs::units::Angle::Degrees");

    // Initialize color
    world.entity<color::Rgb>("::flecs::units::Color::Rgb");
    world.entity<color::Hsl>("::flecs::units::Color::Hsl");
    world.entity<color::Css>("::flecs::units::Color::Css");

    // Initialize percentage
    world.entity<Percentage>("::flecs::units::Percentage");

    // Initialize Bel
    world.entity<Bel>("::flecs::units::Bel");
    world.entity<DeciBel>("::flecs::units::DeciBel");
}

}

#endif
#ifdef FLECS_STATS
/**
 * @file addons/cpp/mixins/stats/impl.hpp
 * @brief Monitor module implementation.
 */

#pragma once

namespace flecs {

inline stats::stats(flecs::world& world) {
#ifdef FLECS_UNITS
    world.import<flecs::units>();
#endif

    /* Import C module  */
    FlecsStatsImport(world);

    world.component<WorldSummary>();
    world.component<WorldStats>();
    world.component<PipelineStats>();
}

}

#endif
#ifdef FLECS_METRICS
/**
 * @file addons/cpp/mixins/metrics/impl.hpp
 * @brief Metrics module implementation.
 */

#pragma once

namespace flecs {

inline metrics::metrics(flecs::world& world) {
    world.import<flecs::units>();

    /* Import C module  */
    FlecsMetricsImport(world);

    world.component<Value>();
    world.component<Source>();

    world.entity<metrics::Instance>("::flecs::metrics::Instance");
    world.entity<metrics::Metric>("::flecs::metrics::Metric");
    world.entity<metrics::Counter>("::flecs::metrics::Metric::Counter");
    world.entity<metrics::CounterId>("::flecs::metrics::Metric::CounterId");
    world.entity<metrics::CounterIncrement>("::flecs::metrics::Metric::CounterIncrement");
    world.entity<metrics::Gauge>("::flecs::metrics::Metric::Gauge");
}

inline metric_builder::~metric_builder() {
    if (!created_) {
        ecs_metric_init(world_, &desc_);
    }
}

inline metric_builder& metric_builder::member(const char *name) {
    flecs::entity m;
    if (desc_.id) {
        flecs::entity_t type = ecs_get_typeid(world_, desc_.id);
        m = flecs::entity(world_, type).lookup(name);
    } else {
        m = flecs::world(world_).lookup(name);
    }
    if (!m) {
        flecs::log::err("member '%s' not found", name);
    }
    return member(m);
}

template <typename T>
inline metric_builder& metric_builder::member(const char *name) {
    flecs::entity e (world_, _::type<T>::id(world_));
    flecs::entity_t m = e.lookup(name);
    if (!m) {
        flecs::log::err("member '%s' not found in type '%s'", 
            name, e.path().c_str());
        return *this;
    }
    return member(m);
}

inline metric_builder& metric_builder::dotmember(const char *expr) {
    desc_.dotmember = expr;
    return *this;
}

template <typename T>
inline metric_builder& metric_builder::dotmember(const char *expr) {
    desc_.dotmember = expr;
    desc_.id = _::type<T>::id(world_);
    return *this;
}

inline metric_builder::operator flecs::entity() {
    if (!created_) {
        created_ = true;
        flecs::entity result(world_, ecs_metric_init(world_, &desc_));
        desc_.entity = result;
        return result;
    } else {
        return flecs::entity(world_, desc_.entity);
    }
}

template <typename... Args>
inline flecs::metric_builder world::metric(Args &&... args) const {
    flecs::entity result(world_, FLECS_FWD(args)...);
    return flecs::metric_builder(world_, result);
}

template <typename Kind>
inline untyped_component& untyped_component::metric(
    flecs::entity_t parent, 
    const char *brief, 
    const char *metric_name) 
{
    flecs::world w(world_);
    flecs::entity e(world_, id_);

    const flecs::member_t *m = ecs_cpp_last_member(w, e);
    if (!m) {
        return *this;
    }

    flecs::entity me = w.entity(m->member);
    flecs::entity metric_entity = me;
    if (parent) {
        const char *component_name = e.name();
        if (!metric_name) {
            if (ecs_os_strcmp(m->name, "value") || !component_name) {
                metric_entity = w.scope(parent).entity(m->name);
            } else {
                // If name of member is "value", use name of type.
                char *snake_name = flecs_to_snake_case(component_name);
                metric_entity = w.scope(parent).entity(snake_name);
                ecs_os_free(snake_name);
            }
        } else {
            metric_entity = w.scope(parent).entity(metric_name);
        }
    }

    w.metric(metric_entity).member(me).kind<Kind>().brief(brief);

    return *this;
}

}

#endif
#ifdef FLECS_ALERTS
/**
 * @file addons/cpp/mixins/alerts/impl.hpp
 * @brief Alerts module implementation.
 */

#pragma once

/**
 * @file addons/cpp/mixins/alerts/builder.hpp
 * @brief Alert builder.
 */

#pragma once

/**
 * @file addons/cpp/mixins/alerts/builder_i.hpp
 * @brief Alert builder interface.
 */

#pragma once


namespace flecs {

/** Alert builder interface.
 * 
 * @ingroup cpp_addons_alerts
 */
template<typename Base, typename ... Components>
struct alert_builder_i : query_builder_i<Base, Components ...> {
private:
    using BaseClass = query_builder_i<Base, Components ...>;

public:
    alert_builder_i()
        : BaseClass(nullptr)
        , desc_(nullptr) { }

    alert_builder_i(ecs_alert_desc_t *desc, int32_t term_index = 0) 
        : BaseClass(&desc->query, term_index)
        , desc_(desc) { }

    /** Alert message.
     *
     * @see ecs_alert_desc_t::message
     */      
    Base& message(const char *message) {
        desc_->message = message;
        return *this;
    }

    /** Set brief description for alert.
     * 
     * @see ecs_alert_desc_t::brief
     */
    Base& brief(const char *brief) {
        desc_->brief = brief;
        return *this;
    }

    /** Set doc name for alert.
     * 
     * @see ecs_alert_desc_t::doc_name
     */
    Base& doc_name(const char *doc_name) {
        desc_->doc_name = doc_name;
        return *this;
    }

    /** Set severity of alert (default is Error) 
     * 
     * @see ecs_alert_desc_t::severity
     */
    Base& severity(flecs::entity_t kind) {
        desc_->severity = kind;
        return *this;
    }

    /* Set retain period of alert. 
     * 
     * @see ecs_alert_desc_t::retain_period
     */
    Base& retain_period(ecs_ftime_t period) {
        desc_->retain_period = period;
        return *this;
    }

    /** Set severity of alert (default is Error) 
     * 
     * @see ecs_alert_desc_t::severity
     */
    template <typename Severity>
    Base& severity() {
        return severity(_::type<Severity>::id(world_v()));
    }

    /** Add severity filter */
    Base& severity_filter(flecs::entity_t kind, flecs::id_t with, const char *var = nullptr) {
        ecs_assert(severity_filter_count < ECS_ALERT_MAX_SEVERITY_FILTERS, 
            ECS_INVALID_PARAMETER, "Maximum number of severity filters reached");

        ecs_alert_severity_filter_t *filter = 
            &desc_->severity_filters[severity_filter_count ++];

        filter->severity = kind;
        filter->with = with;
        filter->var = var;
        return *this;
    }

    /** Add severity filter */
    template <typename Severity>
    Base& severity_filter(flecs::id_t with, const char *var = nullptr) {
        return severity_filter(_::type<Severity>::id(world_v()), with, var);
    }

    /** Add severity filter */
    template <typename Severity, typename T, if_not_t< is_enum<T>::value > = 0>
    Base& severity_filter(const char *var = nullptr) {
        return severity_filter(_::type<Severity>::id(world_v()), 
            _::type<T>::id(world_v()), var);
    }

    /** Add severity filter */
    template <typename Severity, typename T, if_t< is_enum<T>::value > = 0 >
    Base& severity_filter(T with, const char *var = nullptr) {
        flecs::world w(world_v());
        flecs::entity constant = w.to_entity<T>(with);
        return severity_filter(_::type<Severity>::id(world_v()), 
            w.pair<T>(constant), var);
    }

    /** Set member to create an alert for out of range values */
    Base& member(flecs::entity_t m) {
        desc_->member = m;
        return *this;
    }

    /** Set (component) id for member (optional). If .member() is set and id
     * is not set, the id will default to the member parent. */
    Base& id(flecs::id_t id) {
        desc_->id = id;
        return *this;
    }

    /** Set member to create an alert for out of range values */
    template <typename T>
    Base& member(const char *m, const char *v = nullptr) {
        flecs::entity_t id = _::type<T>::id(world_v());
        flecs::entity_t mid = ecs_lookup_path_w_sep(
            world_v(), id, m, "::", "::", false);
        ecs_assert(m != 0, ECS_INVALID_PARAMETER, NULL);
        desc_->var = v;
        return this->member(mid);
    }

    /** Set source variable for member (optional, defaults to $this) */
    Base& var(const char *v) {
        desc_->var = v;
        return *this;
    }

protected:
    virtual flecs::world_t* world_v() = 0;

private:
    operator Base&() {
        return *static_cast<Base*>(this);
    }

    ecs_alert_desc_t *desc_;
    int32_t severity_filter_count = 0;
};

}


namespace flecs {
namespace _ {
    template <typename ... Components>
    using alert_builder_base = builder<
        alert, ecs_alert_desc_t, alert_builder<Components...>, 
        alert_builder_i, Components ...>;
}

/** Alert builder.
 * 
 * @ingroup cpp_addons_alerts
 */
template <typename ... Components>
struct alert_builder final : _::alert_builder_base<Components...> {
    alert_builder(flecs::world_t* world, const char *name = nullptr)
        : _::alert_builder_base<Components...>(world)
    {
        _::sig<Components...>(world).populate(this);
        if (name != nullptr) {
            ecs_entity_desc_t entity_desc = {};
            entity_desc.name = name;
            entity_desc.sep = "::";
            entity_desc.root_sep = "::";
            this->desc_.entity = ecs_entity_init(world, &entity_desc);
        }
    }
};

}


namespace flecs {

template <typename ... Components>
struct alert final : entity
{
    using entity::entity;

    explicit alert() {
        id_ = 0;
        world_ = nullptr;
    }

    explicit alert(flecs::world_t *world, ecs_alert_desc_t *desc) {
        world_ = world;
        id_ = ecs_alert_init(world, desc);
    }
};

inline alerts::alerts(flecs::world& world) {
    world.import<metrics>();

    /* Import C module  */
    FlecsAlertsImport(world);

    world.component<AlertsActive>();
    world.component<Instance>();

    world.entity<alerts::Alert>("::flecs::alerts::Alert");
    world.entity<alerts::Info>("::flecs::alerts::Info");
    world.entity<alerts::Warning>("::flecs::alerts::Warning");
    world.entity<alerts::Error>("::flecs::alerts::Error");
}

template <typename... Comps, typename... Args>
inline flecs::alert_builder<Comps...> world::alert(Args &&... args) const {
    return flecs::alert_builder<Comps...>(world_, FLECS_FWD(args)...);
}

}

#endif
#ifdef FLECS_SCRIPT
/**
 * @file addons/cpp/mixins/script/impl.hpp
 * @brief Script implementation.
 */

#pragma once


namespace flecs 
{

inline flecs::entity script_builder::run() const {
    ecs_entity_t e = ecs_script_init(world_, &desc_);
    return flecs::entity(world_, e);
}

}

#endif

/**
 * @file addons/cpp/impl/field.hpp
 * @brief Field implementation.
 */

#pragma once

namespace flecs
{

template <typename T>
inline field<T>::field(iter &iter, int32_t index) {
    *this = iter.field<T>(index);
}

template <typename T>
T& field<T>::operator[](size_t index) const {
    ecs_assert(data_ != nullptr, ECS_INVALID_OPERATION, 
        "invalid nullptr dereference of component type %s", 
            _::type_name<T>());
    ecs_assert(index < count_, ECS_COLUMN_INDEX_OUT_OF_RANGE,
        "index %d out of range for array of component type %s",
            index, _::type_name<T>());
    ecs_assert(!index || !is_shared_, ECS_INVALID_PARAMETER,
        "non-zero index invalid for shared field of component type %s",
            _::type_name<T>());
    return data_[index];
}

/** Return first element of component array.
 * This operator is typically used when the field is shared.
 *
 * @return Reference to the first element.
 */
template <typename T>
T& field<T>::operator*() const {
    ecs_assert(data_ != nullptr, ECS_INVALID_OPERATION, 
        "invalid nullptr dereference of component type %s", 
            _::type_name<T>());
    return *data_;
}

/** Return first element of component array.
 * This operator is typically used when the field is shared.
 *
 * @return Pointer to the first element.
 */
template <typename T>
T* field<T>::operator->() const {
    ecs_assert(data_ != nullptr, ECS_INVALID_OPERATION, 
        "invalid nullptr dereference of component type %s", 
            _::type_name<T>());
    ecs_assert(data_ != nullptr, ECS_INVALID_OPERATION, 
        "-> operator invalid for array with >1 element of "
        "component type %s, use [row] instead",
            _::type_name<T>());
    return data_;
}

}

/**
 * @file addons/cpp/impl/iter.hpp
 * @brief Iterator implementation.
 */

#pragma once

namespace flecs
{

inline flecs::entity iter::system() const {
    return flecs::entity(iter_->world, iter_->system);
}

inline flecs::entity iter::event() const {
    return flecs::entity(iter_->world, iter_->event);
}

inline flecs::id iter::event_id() const {
    return flecs::id(iter_->world, iter_->event_id);
}

inline flecs::world iter::world() const {
    return flecs::world(iter_->world);
}

inline flecs::entity iter::entity(size_t row) const {
    ecs_assert(row < static_cast<size_t>(iter_->count), 
        ECS_COLUMN_INDEX_OUT_OF_RANGE, NULL);
    return flecs::entity(iter_->world, iter_->entities[row]);
}

inline flecs::entity iter::src(int8_t index) const {
    return flecs::entity(iter_->world, ecs_field_src(iter_, index));
}

inline flecs::id iter::id(int8_t index) const {
    return flecs::id(iter_->world, ecs_field_id(iter_, index));
}

inline flecs::id iter::pair(int8_t index) const {
    flecs::id_t id = ecs_field_id(iter_, index);
    ecs_check(ECS_HAS_ID_FLAG(id, PAIR), ECS_INVALID_PARAMETER, NULL);
    return flecs::id(iter_->world, id);
error:
    return flecs::id();
}

inline flecs::type iter::type() const {
    return flecs::type(iter_->world, ecs_table_get_type(iter_->table));
}

inline flecs::table iter::table() const {
    return flecs::table(iter_->real_world, iter_->table);
}

inline flecs::table iter::other_table() const {
    return flecs::table(iter_->real_world, iter_->other_table);
}

inline flecs::table_range iter::range() const {
    return flecs::table_range(iter_->real_world, iter_->table, 
        iter_->offset, iter_->count);
}

template <typename T, typename A,
    typename std::enable_if<std::is_const<T>::value, void>::type*>
inline flecs::field<A> iter::field(int8_t index) const {
    ecs_assert(!(iter_->flags & EcsIterCppEach) || 
               ecs_field_src(iter_, index) != 0, ECS_INVALID_OPERATION,
        "cannot .field from .each, use .field_at<%s>(%d, row) instead",
            _::type_name<T>(), index);
    return get_field<A>(index);
}

template <typename T, typename A,
    typename std::enable_if<
        std::is_const<T>::value == false, void>::type*>
inline flecs::field<A> iter::field(int8_t index) const {
    ecs_assert(!(iter_->flags & EcsIterCppEach) || 
               ecs_field_src(iter_, index) != 0, ECS_INVALID_OPERATION,
        "cannot .field from .each, use .field_at<%s>(%d, row) instead",
            _::type_name<T>(), index);
    ecs_assert(!ecs_field_is_readonly(iter_, index),
        ECS_ACCESS_VIOLATION, NULL);
    return get_field<A>(index);
}

inline flecs::entity iter::get_var(int var_id) const {
    ecs_assert(var_id != -1, ECS_INVALID_PARAMETER, 0);
    return flecs::entity(iter_->world, ecs_iter_get_var(iter_, var_id));
}

/** Get value of variable by name.
 * Get value of a query variable for current result.
 */
inline flecs::entity iter::get_var(const char *name) const {
    ecs_query_iter_t *qit = &iter_->priv_.iter.query;
    const flecs::query_t *q = qit->query;
    int var_id = ecs_query_find_var(q, name);
    ecs_assert(var_id != -1, ECS_INVALID_PARAMETER, name);
    return flecs::entity(iter_->world, ecs_iter_get_var(iter_, var_id));
}

template <typename Func>
void iter::targets(int8_t index, const Func& func) {
    ecs_assert(iter_->table != nullptr, ECS_INVALID_OPERATION, NULL);
    ecs_assert(index < iter_->field_count, ECS_INVALID_PARAMETER, NULL);
    ecs_assert(ecs_field_is_set(iter_, index), ECS_INVALID_PARAMETER, NULL);
    const ecs_type_t *table_type = ecs_table_get_type(iter_->table);
    const ecs_table_record_t *tr = iter_->trs[index];
    int32_t i = tr->index, end = i + tr->count;
    for (; i < end; i ++) {
        ecs_id_t id = table_type->array[i];
        ecs_assert(ECS_IS_PAIR(id), ECS_INVALID_PARAMETER, 
            "field does not match a pair");
        flecs::entity tgt(iter_->world, 
            ecs_pair_second(iter_->real_world, id));
        func(tgt);
    }
}

} // namespace flecs

/**
 * @file addons/cpp/impl/world.hpp
 * @brief World implementation.
 */

#pragma once

namespace flecs 
{

inline void world::init_builtin_components() {
    this->component<Component>();
    this->component<Identifier>();
    this->component<Poly>();

#   ifdef FLECS_SYSTEM
    _::system_init(*this);
#   endif
#   ifdef FLECS_TIMER
    _::timer_init(*this);
#   endif
#   ifdef FLECS_DOC
    doc::_::init(*this);
#   endif
#   ifdef FLECS_REST
    rest::_::init(*this);
#   endif
#   ifdef FLECS_META
    meta::_::init(*this);
#   endif
}

template <typename T>
inline flecs::entity world::use(const char *alias) const {
    entity_t e = _::type<T>::id(world_);
    const char *name = alias;
    if (!name) {
        // If no name is defined, use the entity name without the scope
        name = ecs_get_name(world_, e);
    }
    ecs_set_alias(world_, e, name);
    return flecs::entity(world_, e);
}

inline flecs::entity world::use(const char *name, const char *alias) const {
    entity_t e = ecs_lookup_path_w_sep(world_, 0, name, "::", "::", true);
    ecs_assert(e != 0, ECS_INVALID_PARAMETER, NULL);

    ecs_set_alias(world_, e, alias);
    return flecs::entity(world_, e);
}

inline void world::use(flecs::entity e, const char *alias) const {
    entity_t eid = e.id();
    const char *name = alias;
    if (!name) {
        // If no name is defined, use the entity name without the scope
        name = ecs_get_name(world_, eid);
    }
    ecs_set_alias(world_, eid, name);
}

inline flecs::entity world::set_scope(const flecs::entity_t s) const {
    return flecs::entity(ecs_set_scope(world_, s));
}

inline flecs::entity world::get_scope() const {
    return flecs::entity(world_, ecs_get_scope(world_));
}

template <typename T>
inline flecs::entity world::set_scope() const {
    return set_scope( _::type<T>::id(world_) ); 
}

inline entity world::lookup(const char *name, const char *sep, const char *root_sep, bool recursive) const {
    auto e = ecs_lookup_path_w_sep(world_, 0, name, sep, root_sep, recursive);
    return flecs::entity(*this, e);
}

#ifndef ensure
template <typename T>
inline T& world::ensure() const {
    flecs::entity e(world_, _::type<T>::id(world_));
    return e.ensure<T>();
}
#endif

template <typename T>
inline void world::modified() const {
    flecs::entity e(world_, _::type<T>::id(world_));
    e.modified<T>();
}

template <typename First, typename Second>
inline void world::set(Second second, const First& value) const {
    flecs::entity e(world_, _::type<First>::id(world_));
    e.set<First>(second, value);
}

template <typename First, typename Second>
inline void world::set(Second second, First&& value) const {
    flecs::entity e(world_, _::type<First>::id(world_));
    e.set<First>(second, value);
}

template <typename T>
inline ref<T> world::get_ref() const {
    flecs::entity e(world_, _::type<T>::id(world_));
    return e.get_ref<T>();
}

template <typename T>
inline const T* world::get() const {
    flecs::entity e(world_, _::type<T>::id(world_));
    return e.get<T>();
}

template <typename First, typename Second, typename P, typename A>
const A* world::get() const {
    flecs::entity e(world_, _::type<First>::id(world_));
    return e.get<First, Second>();
}

template <typename First, typename Second>
const First* world::get(Second second) const {
    flecs::entity e(world_, _::type<First>::id(world_));
    return e.get<First>(second);
}

template <typename T>
T* world::get_mut() const {
    flecs::entity e(world_, _::type<T>::id(world_));
    return e.get_mut<T>();
}

template <typename First, typename Second, typename P, typename A>
A* world::get_mut() const {
    flecs::entity e(world_, _::type<First>::id(world_));
    return e.get_mut<First, Second>();
}

template <typename First, typename Second>
First* world::get_mut(Second second) const {
    flecs::entity e(world_, _::type<First>::id(world_));
    return e.get_mut<First>(second);
}

template <typename T>
inline bool world::has() const {
    flecs::entity e(world_, _::type<T>::id(world_));
    return e.has<T>();
}

template <typename First, typename Second>
inline bool world::has() const {
    flecs::entity e(world_, _::type<First>::id(world_));
    return e.has<First, Second>();
}

template <typename First>
inline bool world::has(flecs::id_t second) const {
    flecs::entity e(world_, _::type<First>::id(world_));
    return e.has<First>(second);
}

inline bool world::has(flecs::id_t first, flecs::id_t second) const {
    flecs::entity e(world_, first);
    return e.has(first, second);
}

template <typename T>
inline void world::add() const {
    flecs::entity e(world_, _::type<T>::id(world_));
    e.add<T>();
}

template <typename First, typename Second>
inline void world::add() const {
    flecs::entity e(world_, _::type<First>::id(world_));
    e.add<First, Second>();
}

template <typename First>
inline void world::add(flecs::entity_t second) const {
    flecs::entity e(world_, _::type<First>::id(world_));
    e.add<First>(second);
}

inline void world::add(flecs::entity_t first, flecs::entity_t second) const {
    flecs::entity e(world_, first);
    e.add(first, second);
}

template <typename T>
inline void world::remove() const {
    flecs::entity e(world_, _::type<T>::id(world_));
    e.remove<T>();
}

template <typename First, typename Second>
inline void world::remove() const {
    flecs::entity e(world_, _::type<First>::id(world_));
    e.remove<First, Second>();
}

template <typename First>
inline void world::remove(flecs::entity_t second) const {
    flecs::entity e(world_, _::type<First>::id(world_));
    e.remove<First>(second);
}

inline void world::remove(flecs::entity_t first, flecs::entity_t second) const {
    flecs::entity e(world_, first);
    e.remove(first, second);
}

template <typename Func>
inline void world::children(Func&& f) const {
    this->entity(0).children(FLECS_FWD(f));
}

template <typename T>
inline flecs::entity world::singleton() const {
    return flecs::entity(world_, _::type<T>::id(world_));
}

template <typename First>
inline flecs::entity world::target(int32_t index) const
{
    return flecs::entity(world_,
        ecs_get_target(world_, _::type<First>::id(world_), _::type<First>::id(world_), index));
}

template <typename T>
inline flecs::entity world::target(
    flecs::entity_t relationship,
    int32_t index) const
{
    return flecs::entity(world_,
        ecs_get_target(world_, _::type<T>::id(world_), relationship, index));
}

inline flecs::entity world::target(
    flecs::entity_t relationship,
    int32_t index) const
{
    return flecs::entity(world_,
        ecs_get_target(world_, relationship, relationship, index));
}

template <typename Func, if_t< is_callable<Func>::value > >
inline void world::get(const Func& func) const {
    static_assert(arity<Func>::value == 1, "singleton component must be the only argument");
    _::entity_with_delegate<Func>::invoke_get(
        this->world_, this->singleton<first_arg_t<Func>>(), func);
}

template <typename Func, if_t< is_callable<Func>::value > >
inline void world::set(const Func& func) const {
    static_assert(arity<Func>::value == 1, "singleton component must be the only argument");
    _::entity_with_delegate<Func>::invoke_ensure(
        this->world_, this->singleton<first_arg_t<Func>>(), func);
}

inline flecs::entity world::get_alive(flecs::entity_t e) const {
    e = ecs_get_alive(world_, e);
    return flecs::entity(world_, e);
}

inline flecs::entity world::make_alive(flecs::entity_t e) const {
    ecs_make_alive(world_, e);
    return flecs::entity(world_, e);
}

template <typename E>
inline flecs::entity enum_data<E>::entity() const {
    return flecs::entity(world_, _::type<E>::id(world_));
}

template <typename E>
inline flecs::entity enum_data<E>::entity(underlying_type_t<E> value) const {
    int index = index_by_value(value);
    if (index >= 0) {
        int32_t constant_i = impl_.constants[index].index;
        flecs::entity_t entity = flecs_component_ids_get(world_, constant_i);
        return flecs::entity(world_, entity);
    }
#ifdef FLECS_META
    // Reflection data lookup failed. Try value lookup amongst flecs::Constant relationships
    flecs::world world = flecs::world(world_);
    return world.query_builder()
        .with(flecs::ChildOf, world.id<E>())
        .with(flecs::Constant, world.id<int32_t>())
        .build()
        .find([value](flecs::entity constant) {
            const int32_t *constant_value = constant.get_second<int32_t>(flecs::Constant);
            ecs_assert(constant_value, ECS_INTERNAL_ERROR, NULL);
            return value == static_cast<underlying_type_t<E>>(*constant_value);
        });
#else
    return flecs::entity::null(world_);
#endif
}

template <typename E>
inline flecs::entity enum_data<E>::entity(E value) const {
    return entity(static_cast<underlying_type_t<E>>(value));
}

/** Use provided scope for operations ran on returned world.
 * Operations need to be ran in a single statement.
 */
inline flecs::scoped_world world::scope(id_t parent) const {
    return scoped_world(world_, parent);
}

template <typename T>
inline flecs::scoped_world world::scope() const {
    flecs::id_t parent = _::type<T>::id(world_);
    return scoped_world(world_, parent);
}

inline flecs::scoped_world world::scope(const char* name) const {
  return scope(entity(name));
}

} // namespace flecs


/**
 * @defgroup cpp_core Core
 * Core ECS functionality (entities, storage, queries)
 *
 * @{
 * @}
 */

/**
 * @defgroup cpp_addons Addons
 * C++ APIs for addons.
 *
 * @{
 * @}
 */

/** @} */

#endif // __cplusplus

#endif // FLECS_CPP

#endif


#endif

