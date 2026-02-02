/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 * Copyright 2011-2024  Jose Luis Blanco (joseluisblancoc@gmail.com).
 *   All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

/** \mainpage nanoflann C++ API documentation
 *  nanoflann is a C++ header-only library for building KD-Trees, mostly
 *  optimized for 2D or 3D point clouds.
 *
 *  nanoflann does not require compiling or installing, just an
 *  #include <nanoflann.hpp> in your code.
 *
 *  See:
 *   - [Online README](https://github.com/jlblancoc/nanoflann)
 *   - [C++ API documentation](https://jlblancoc.github.io/nanoflann/)
 */

// Modified to disable exception 

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>  // for abs()
#include <cstdint>
#include <cstdlib>  // for abs()
#include <functional>  // std::reference_wrapper
#include <future>
#include <istream>
#include <limits>  // std::numeric_limits
#include <ostream>

#if !defined(NANOFLANN_NO_EXCEPTIONS)
# if defined(_MSC_VER)
#  include <cstddef>    // for _HAS_EXCEPTIONS
# endif
# if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || (_HAS_EXCEPTIONS)
#  define NANOFLANN_NO_EXCEPTIONS  0
# else
#  define NANOFLANN_NO_EXCEPTIONS  1
# endif
#endif

#if !NANOFLANN_NO_EXCEPTIONS
#include <stdexcept>
#endif

#include <unordered_set>
#include <vector>

/** Library version: 0xMmP (M=Major,m=minor,P=patch) */
#define NANOFLANN_VERSION 0x155

// Avoid conflicting declaration of min/max macros in Windows headers
#if !defined(NOMINMAX) && \
    (defined(_WIN32) || defined(_WIN32_) || defined(WIN32) || defined(_WIN64))
#define NOMINMAX
#ifdef max
#undef max
#undef min
#endif
#endif
// Avoid conflicts with X11 headers
#ifdef None
#undef None
#endif

namespace nanoflann
{
/** @addtogroup nanoflann_grp nanoflann C++ library for KD-trees
 *  @{ */

/** the PI constant (required to avoid MSVC missing symbols) */
template <typename T>
T pi_const()
{
    return static_cast<T>(3.14159265358979323846);
}

/**
 * Traits if object is resizable and assignable (typically has a resize | assign
 * method)
 */
template <typename T, typename = int>
struct has_resize : std::false_type
{
};

template <typename T>
struct has_resize<T, decltype((void)std::declval<T>().resize(1), 0)>
    : std::true_type
{
};

template <typename T, typename = int>
struct has_assign : std::false_type
{
};

template <typename T>
struct has_assign<T, decltype((void)std::declval<T>().assign(1, 0), 0)>
    : std::true_type
{
};

/**
 * Free function to resize a resizable object
 */
template <typename Container>
inline typename std::enable_if<has_resize<Container>::value, void>::type resize(
    Container& c, const size_t nElements)
{
    c.resize(nElements);
}

/**
 * Free function that has no effects on non resizable containers (e.g.
 * std::array) It raises an exception if the expected size does not match
 */
template <typename Container>
inline typename std::enable_if<!has_resize<Container>::value, void>::type
    resize(Container& c, const size_t nElements)
{
    if (nElements != c.size()) {
#if !NANOFLANN_NO_EXCEPTIONS
        throw std::logic_error("Try to change the size of a std::array.");
#endif
    }
}

/**
 * Free function to assign to a container
 */
template <typename Container, typename T>
inline typename std::enable_if<has_assign<Container>::value, void>::type assign(
    Container& c, const size_t nElements, const T& value)
{
    c.assign(nElements, value);
}

/**
 * Free function to assign to a std::array
 */
template <typename Container, typename T>
inline typename std::enable_if<!has_assign<Container>::value, void>::type
    assign(Container& c, const size_t nElements, const T& value)
{
    for (size_t i = 0; i < nElements; i++) c[i] = value;
}

/** @addtogroup result_sets_grp Result set classes
 *  @{ */

/** Result set for KNN searches (N-closest neighbors) */
template <
    typename _DistanceType, typename _IndexType = size_t,
    typename _CountType = size_t>
class KNNResultSet
{
   public:
    using DistanceType = _DistanceType;
    using IndexType    = _IndexType;
    using CountType    = _CountType;

   private:
    IndexType*    indices;
    DistanceType* dists;
    CountType     capacity;
    CountType     count;

   public:
    explicit KNNResultSet(CountType capacity_)
        : indices(nullptr), dists(nullptr), capacity(capacity_), count(0)
    {
    }

    void init(IndexType* indices_, DistanceType* dists_)
    {
        indices = indices_;
        dists   = dists_;
        count   = 0;
        if (capacity)
            dists[capacity - 1] = (std::numeric_limits<DistanceType>::max)();
    }

    CountType size() const { return count; }
    bool      empty() const { return count == 0; }
    bool      full() const { return count == capacity; }

    /**
     * Called during search to add an element matching the criteria.
     * @return true if the search should be continued, false if the results are
     * sufficient
     */
    bool addPoint(DistanceType dist, IndexType index)
    {
        CountType i;
        for (i = count; i > 0; --i)
        {
            /** If defined and two points have the same distance, the one with
             *  the lowest-index will be returned first. */
#ifdef NANOFLANN_FIRST_MATCH
            if ((dists[i - 1] > dist) ||
                ((dist == dists[i - 1]) && (indices[i - 1] > index)))
            {
#else
            if (dists[i - 1] > dist)
            {
#endif
                if (i < capacity)
                {
                    dists[i]   = dists[i - 1];
                    indices[i] = indices[i - 1];
                }
            }
            else
                break;
        }
        if (i < capacity)
        {
            dists[i]   = dist;
            indices[i] = index;
        }
        if (count < capacity) count++;

        // tell caller that the search shall continue
        return true;
    }

    DistanceType worstDist() const { return dists[capacity - 1]; }
};

/** Result set for RKNN searches (N-closest neighbors with a maximum radius) */
template <
    typename _DistanceType, typename _IndexType = size_t,
    typename _CountType = size_t>
class RKNNResultSet
{
   public:
    using DistanceType = _DistanceType;
    using IndexType    = _IndexType;
    using CountType    = _CountType;

   private:
    IndexType*    indices;
    DistanceType* dists;
    CountType     capacity;
    CountType     count;
    DistanceType  maximumSearchDistanceSquared;

   public:
    explicit RKNNResultSet(
        CountType capacity_, DistanceType maximumSearchDistanceSquared_)
        : indices(nullptr),
          dists(nullptr),
          capacity(capacity_),
          count(0),
          maximumSearchDistanceSquared(maximumSearchDistanceSquared_)
    {
    }

    void init(IndexType* indices_, DistanceType* dists_)
    {
        indices = indices_;
        dists   = dists_;
        count   = 0;
        if (capacity) dists[capacity - 1] = maximumSearchDistanceSquared;
    }

    CountType size() const { return count; }
    bool      empty() const { return count == 0; }
    bool      full() const { return count == capacity; }

    /**
     * Called during search to add an element matching the criteria.
     * @return true if the search should be continued, false if the results are
     * sufficient
     */
    bool addPoint(DistanceType dist, IndexType index)
    {
        CountType i;
        for (i = count; i > 0; --i)
        {
            /** If defined and two points have the same distance, the one with
             *  the lowest-index will be returned first. */
#ifdef NANOFLANN_FIRST_MATCH
            if ((dists[i - 1] > dist) ||
                ((dist == dists[i - 1]) && (indices[i - 1] > index)))
            {
#else
            if (dists[i - 1] > dist)
            {
#endif
                if (i < capacity)
                {
                    dists[i]   = dists[i - 1];
                    indices[i] = indices[i - 1];
                }
            }
            else
                break;
        }
        if (i < capacity)
        {
            dists[i]   = dist;
            indices[i] = index;
        }
        if (count < capacity) count++;

        // tell caller that the search shall continue
        return true;
    }

    DistanceType worstDist() const { return dists[capacity - 1]; }
};

/** operator "<" for std::sort() */
struct IndexDist_Sorter
{
    /** PairType will be typically: ResultItem<IndexType,DistanceType> */
    template <typename PairType>
    bool operator()(const PairType& p1, const PairType& p2) const
    {
        return p1.second < p2.second;
    }
};

/**
 * Each result element in RadiusResultSet. Note that distances and indices
 * are named `first` and `second` to keep backward-compatibility with the
 * `std::pair<>` type used in the past. In contrast, this structure is ensured
 * to be `std::is_standard_layout` so it can be used in wrappers to other
 * languages.
 * See: https://github.com/jlblancoc/nanoflann/issues/166
 */
template <typename IndexType = size_t, typename DistanceType = double>
struct ResultItem
{
    ResultItem() = default;
    ResultItem(const IndexType index, const DistanceType distance)
        : first(index), second(distance)
    {
    }

    IndexType    first;  //!< Index of the sample in the dataset
    DistanceType second;  //!< Distance from sample to query point
};

/**
 * A result-set class used when performing a radius based search.
 */
template <typename _DistanceType, typename _IndexType = size_t>
class RadiusResultSet
{
   public:
    using DistanceType = _DistanceType;
    using IndexType    = _IndexType;

   public:
    const DistanceType radius;

    std::vector<ResultItem<IndexType, DistanceType>>& m_indices_dists;

    explicit RadiusResultSet(
        DistanceType                                      radius_,
        std::vector<ResultItem<IndexType, DistanceType>>& indices_dists)
        : radius(radius_), m_indices_dists(indices_dists)
    {
        init();
    }

    void init() { clear(); }
    void clear() { m_indices_dists.clear(); }

    size_t size() const { return m_indices_dists.size(); }
    size_t empty() const { return m_indices_dists.empty(); }

    bool full() const { return true; }

    /**
     * Called during search to add an element matching the criteria.
     * @return true if the search should be continued, false if the results are
     * sufficient
     */
    bool addPoint(DistanceType dist, IndexType index)
    {
        if (dist < radius) m_indices_dists.emplace_back(index, dist);
        return true;
    }

    DistanceType worstDist() const { return radius; }

    /**
     * Find the worst result (farthest neighbor) without copying or sorting
     * Pre-conditions: size() > 0
     */
    ResultItem<IndexType, DistanceType> worst_item() const
    {
        if (m_indices_dists.empty()) {
#if !NANOFLANN_NO_EXCEPTIONS
            throw std::runtime_error(
                "Cannot invoke RadiusResultSet::worst_item() on "
                "an empty list of results.");
#endif
        }
        auto it = std::max_element(
            m_indices_dists.begin(), m_indices_dists.end(), IndexDist_Sorter());
        return *it;
    }
};

/** @} */

/** @addtogroup loadsave_grp Load/save auxiliary functions
 * @{ */
template <typename T>
void save_value(std::ostream& stream, const T& value)
{
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
void save_value(std::ostream& stream, const std::vector<T>& value)
{
    size_t size = value.size();
    stream.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
    stream.write(reinterpret_cast<const char*>(value.data()), sizeof(T) * size);
}

template <typename T>
void load_value(std::istream& stream, T& value)
{
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
}

template <typename T>
void load_value(std::istream& stream, std::vector<T>& value)
{
    size_t size;
    stream.read(reinterpret_cast<char*>(&size), sizeof(size_t));
    value.resize(size);
    stream.read(reinterpret_cast<char*>(value.data()), sizeof(T) * size);
}
/** @} */

/** @addtogroup metric_grp Metric (distance) classes
 * @{ */

struct Metric
{
};

/** Manhattan distance functor (generic version, optimized for
 * high-dimensionality data sets). Corresponding distance traits:
 * nanoflann::metric_L1
 *
 * \tparam T Type of the elements (e.g. double, float, uint8_t)
 * \tparam DataSource Source of the data, i.e. where the vectors are stored
 * \tparam _DistanceType Type of distance variables (must be signed)
 * \tparam IndexType Type of the arguments with which the data can be
 * accessed (e.g. float, double, int64_t, T*)
 */
template <
    class T, class DataSource, typename _DistanceType = T,
    typename IndexType = uint32_t>
struct L1_Adaptor
{
    using ElementType  = T;
    using DistanceType = _DistanceType;

    const DataSource& data_source;

    L1_Adaptor(const DataSource& _data_source) : data_source(_data_source) {}

    DistanceType evalMetric(
        const T* a, const IndexType b_idx, size_t size,
        DistanceType worst_dist = -1) const
    {
        DistanceType result    = DistanceType();
        const T*     last      = a + size;
        const T*     lastgroup = last - 3;
        size_t       d         = 0;

        /* Process 4 items with each loop for efficiency. */
        while (a < lastgroup)
        {
            const DistanceType diff0 =
                std::abs(a[0] - data_source.kdtree_get_pt(b_idx, d++));
            const DistanceType diff1 =
                std::abs(a[1] - data_source.kdtree_get_pt(b_idx, d++));
            const DistanceType diff2 =
                std::abs(a[2] - data_source.kdtree_get_pt(b_idx, d++));
            const DistanceType diff3 =
                std::abs(a[3] - data_source.kdtree_get_pt(b_idx, d++));
            result += diff0 + diff1 + diff2 + diff3;
            a += 4;
            if ((worst_dist > 0) && (result > worst_dist)) { return result; }
        }
        /* Process last 0-3 components.  Not needed for standard vector lengths.
         */
        while (a < last)
        {
            result += std::abs(*a++ - data_source.kdtree_get_pt(b_idx, d++));
        }
        return result;
    }

    template <typename U, typename V>
    DistanceType accum_dist(const U a, const V b, const size_t) const
    {
        return std::abs(a - b);
    }
};

/** **Squared** Euclidean distance functor (generic version, optimized for
 * high-dimensionality data sets). Corresponding distance traits:
 * nanoflann::metric_L2
 *
 * \tparam T Type of the elements (e.g. double, float, uint8_t)
 * \tparam DataSource Source of the data, i.e. where the vectors are stored
 * \tparam _DistanceType Type of distance variables (must be signed)
 * \tparam IndexType Type of the arguments with which the data can be
 * accessed (e.g. float, double, int64_t, T*)
 */
template <
    class T, class DataSource, typename _DistanceType = T,
    typename IndexType = uint32_t>
struct L2_Adaptor
{
    using ElementType  = T;
    using DistanceType = _DistanceType;

    const DataSource& data_source;

    L2_Adaptor(const DataSource& _data_source) : data_source(_data_source) {}

    DistanceType evalMetric(
        const T* a, const IndexType b_idx, size_t size,
        DistanceType worst_dist = -1) const
    {
        DistanceType result    = DistanceType();
        const T*     last      = a + size;
        const T*     lastgroup = last - 3;
        size_t       d         = 0;

        /* Process 4 items with each loop for efficiency. */
        while (a < lastgroup)
        {
            const DistanceType diff0 =
                a[0] - data_source.kdtree_get_pt(b_idx, d++);
            const DistanceType diff1 =
                a[1] - data_source.kdtree_get_pt(b_idx, d++);
            const DistanceType diff2 =
                a[2] - data_source.kdtree_get_pt(b_idx, d++);
            const DistanceType diff3 =
                a[3] - data_source.kdtree_get_pt(b_idx, d++);
            result +=
                diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
            a += 4;
            if ((worst_dist > 0) && (result > worst_dist)) { return result; }
        }
        /* Process last 0-3 components.  Not needed for standard vector lengths.
         */
        while (a < last)
        {
            const DistanceType diff0 =
                *a++ - data_source.kdtree_get_pt(b_idx, d++);
            result += diff0 * diff0;
        }
        return result;
    }

    template <typename U, typename V>
    DistanceType accum_dist(const U a, const V b, const size_t) const
    {
        return (a - b) * (a - b);
    }
};

/** **Squared** Euclidean (L2) distance functor (suitable for low-dimensionality
 * datasets, like 2D or 3D point clouds) Corresponding distance traits:
 * nanoflann::metric_L2_Simple
 *
 * \tparam T Type of the elements (e.g. double, float, uint8_t)
 * \tparam DataSource Source of the data, i.e. where the vectors are stored
 * \tparam _DistanceType Type of distance variables (must be signed)
 * \tparam IndexType Type of the arguments with which the data can be
 * accessed (e.g. float, double, int64_t, T*)
 */
template <
    class T, class DataSource, typename _DistanceType = T,
    typename IndexType = uint32_t>
struct L2_Simple_Adaptor
{
    using ElementType  = T;
    using DistanceType = _DistanceType;

    const DataSource& data_source;

    L2_Simple_Adaptor(const DataSource& _data_source)
        : data_source(_data_source)
    {
    }

    DistanceType evalMetric(
        const T* a, const IndexType b_idx, size_t size) const
    {
        DistanceType result = DistanceType();
        for (size_t i = 0; i < size; ++i)
        {
            const DistanceType diff =
                a[i] - data_source.kdtree_get_pt(b_idx, i);
            result += diff * diff;
        }
        return result;
    }

    template <typename U, typename V>
    DistanceType accum_dist(const U a, const V b, const size_t) const
    {
        return (a - b) * (a - b);
    }
};

/** SO2 distance functor
 *  Corresponding distance traits: nanoflann::metric_SO2
 *
 * \tparam T Type of the elements (e.g. double, float, uint8_t)
 * \tparam DataSource Source of the data, i.e. where the vectors are stored
 * \tparam _DistanceType Type of distance variables (must be signed) (e.g.
 * float, double) orientation is constrained to be in [-pi, pi]
 * \tparam IndexType Type of the arguments with which the data can be
 * accessed (e.g. float, double, int64_t, T*)
 */
template <
    class T, class DataSource, typename _DistanceType = T,
    typename IndexType = uint32_t>
struct SO2_Adaptor
{
    using ElementType  = T;
    using DistanceType = _DistanceType;

    const DataSource& data_source;

    SO2_Adaptor(const DataSource& _data_source) : data_source(_data_source) {}

    DistanceType evalMetric(
        const T* a, const IndexType b_idx, size_t size) const
    {
        return accum_dist(
            a[size - 1], data_source.kdtree_get_pt(b_idx, size - 1), size - 1);
    }

    /** Note: this assumes that input angles are already in the range [-pi,pi]
     */
    template <typename U, typename V>
    DistanceType accum_dist(const U a, const V b, const size_t) const
    {
        DistanceType result = DistanceType();
        DistanceType PI     = pi_const<DistanceType>();
        result              = b - a;
        if (result > PI)
            result -= 2 * PI;
        else if (result < -PI)
            result += 2 * PI;
        return result;
    }
};

/** SO3 distance functor (Uses L2_Simple)
 *  Corresponding distance traits: nanoflann::metric_SO3
 *
 * \tparam T Type of the elements (e.g. double, float, uint8_t)
 * \tparam DataSource Source of the data, i.e. where the vectors are stored
 * \tparam _DistanceType Type of distance variables (must be signed) (e.g.
 * float, double)
 * \tparam IndexType Type of the arguments with which the data can be
 * accessed (e.g. float, double, int64_t, T*)
 */
template <
    class T, class DataSource, typename _DistanceType = T,
    typename IndexType = uint32_t>
struct SO3_Adaptor
{
    using ElementType  = T;
    using DistanceType = _DistanceType;

    L2_Simple_Adaptor<T, DataSource, DistanceType, IndexType>
        distance_L2_Simple;

    SO3_Adaptor(const DataSource& _data_source)
        : distance_L2_Simple(_data_source)
    {
    }

    DistanceType evalMetric(
        const T* a, const IndexType b_idx, size_t size) const
    {
        return distance_L2_Simple.evalMetric(a, b_idx, size);
    }

    template <typename U, typename V>
    DistanceType accum_dist(const U a, const V b, const size_t idx) const
    {
        return distance_L2_Simple.accum_dist(a, b, idx);
    }
};

/** Metaprogramming helper traits class for the L1 (Manhattan) metric */
struct metric_L1 : public Metric
{
    template <class T, class DataSource, typename IndexType = uint32_t>
    struct traits
    {
        using distance_t = L1_Adaptor<T, DataSource, T, IndexType>;
    };
};
/** Metaprogramming helper traits class for the L2 (Euclidean) **squared**
 * distance metric */
struct metric_L2 : public Metric
{
    template <class T, class DataSource, typename IndexType = uint32_t>
    struct traits
    {
        using distance_t = L2_Adaptor<T, DataSource, T, IndexType>;
    };
};
/** Metaprogramming helper traits class for the L2_simple (Euclidean)
 * **squared** distance metric */
struct metric_L2_Simple : public Metric
{
    template <class T, class DataSource, typename IndexType = uint32_t>
    struct traits
    {
        using distance_t = L2_Simple_Adaptor<T, DataSource, T, IndexType>;
    };
};
/** Metaprogramming helper traits class for the SO3_InnerProdQuat metric */
struct metric_SO2 : public Metric
{
    template <class T, class DataSource, typename IndexType = uint32_t>
    struct traits
    {
        using distance_t = SO2_Adaptor<T, DataSource, T, IndexType>;
    };
};
/** Metaprogramming helper traits class for the SO3_InnerProdQuat metric */
struct metric_SO3 : public Metric
{
    template <class T, class DataSource, typename IndexType = uint32_t>
    struct traits
    {
        using distance_t = SO3_Adaptor<T, DataSource, T, IndexType>;
    };
};

/** @} */

/** @addtogroup param_grp Parameter structs
 * @{ */

enum class KDTreeSingleIndexAdaptorFlags
{
    None                  = 0,
    SkipInitialBuildIndex = 1
};

inline std::underlying_type<KDTreeSingleIndexAdaptorFlags>::type operator&(
    KDTreeSingleIndexAdaptorFlags lhs, KDTreeSingleIndexAdaptorFlags rhs)
{
    using underlying =
        typename std::underlying_type<KDTreeSingleIndexAdaptorFlags>::type;
    return static_cast<underlying>(lhs) & static_cast<underlying>(rhs);
}

/**  Parameters (see README.md) */
struct KDTreeSingleIndexAdaptorParams
{
    KDTreeSingleIndexAdaptorParams(
        size_t                        _leaf_max_size = 10,
        KDTreeSingleIndexAdaptorFlags _flags =
            KDTreeSingleIndexAdaptorFlags::None,
        unsigned int _n_thread_build = 1)
        : leaf_max_size(_leaf_max_size),
          flags(_flags),
          n_thread_build(_n_thread_build)
    {
    }

    size_t                        leaf_max_size;
    KDTreeSingleIndexAdaptorFlags flags;
    unsigned int                  n_thread_build;
};

/** Search options for KDTreeSingleIndexAdaptor::findNeighbors() */
struct SearchParameters
{
    SearchParameters(float eps_ = 0, bool sorted_ = true)
        : eps(eps_), sorted(sorted_)
    {
    }

    float eps;  //!< search for eps-approximate neighbours (default: 0)
    bool  sorted;  //!< only for radius search, require neighbours sorted by
                  //!< distance (default: true)
};
/** @} */

/** @addtogroup memalloc_grp Memory allocation
 * @{ */

/**
 * Pooled storage allocator
 *
 * The following routines allow for the efficient allocation of storage in
 * small chunks from a specified pool.  Rather than allowing each structure
 * to be freed individually, an entire pool of storage is freed at once.
 * This method has two advantages over just using malloc() and free().  First,
 * it is far more efficient for allocating small objects, as there is
 * no overhead for remembering all the information needed to free each
 * object or consolidating fragmented memory.  Second, the decision about
 * how long to keep an object is made at the time of allocation, and there
 * is no need to track down all the objects to free them.
 *
 */
class PooledAllocator
{
    static constexpr size_t WORDSIZE  = 16;  // WORDSIZE must >= 8
    static constexpr size_t BLOCKSIZE = 8192;

    /* We maintain memory alignment to word boundaries by requiring that all
        allocations be in multiples of the machine wordsize.  */
    /* Size of machine word in bytes.  Must be power of 2. */
    /* Minimum number of bytes requested at a time from	the system.  Must be
     * multiple of WORDSIZE. */

    using Size = size_t;

    Size  remaining_ = 0;  //!< Number of bytes left in current block of storage
    void* base_ = nullptr;  //!< Pointer to base of current block of storage
    void* loc_  = nullptr;  //!< Current location in block to next allocate

    void internal_init()
    {
        remaining_   = 0;
        base_        = nullptr;
        usedMemory   = 0;
        wastedMemory = 0;
    }

   public:
    Size usedMemory   = 0;
    Size wastedMemory = 0;

    /**
        Default constructor. Initializes a new pool.
     */
    PooledAllocator() { internal_init(); }

    /**
     * Destructor. Frees all the memory allocated in this pool.
     */
    ~PooledAllocator() { free_all(); }

    /** Frees all allocated memory chunks */
    void free_all()
    {
        while (base_ != nullptr)
        {
            // Get pointer to prev block
            void* prev = *(static_cast<void**>(base_));
            ::free(base_);
            base_ = prev;
        }
        internal_init();
    }

    /**
     * Returns a pointer to a piece of new memory of the given size in bytes
     * allocated from the pool.
     */
    void* malloc(const size_t req_size)
    {
        /* Round size up to a multiple of wordsize.  The following expression
            only works for WORDSIZE that is a power of 2, by masking last bits
           of incremented size to zero.
         */
        const Size size = (req_size + (WORDSIZE - 1)) & ~(WORDSIZE - 1);

        /* Check whether a new block must be allocated.  Note that the first
           word of a block is reserved for a pointer to the previous block.
         */
        if (size > remaining_)
        {
            wastedMemory += remaining_;

            /* Allocate new storage. */
            const Size blocksize =
                size > BLOCKSIZE ? size + WORDSIZE : BLOCKSIZE + WORDSIZE;

            // use the standard C malloc to allocate memory
            void* m = ::malloc(blocksize);
            if (!m)
            {
                fprintf(stderr, "Failed to allocate memory.\n");
#if !NANOFLANN_NO_EXCEPTIONS
                throw std::bad_alloc();
#endif
            }

            /* Fill first word of new block with pointer to previous block. */
            static_cast<void**>(m)[0] = base_;
            base_                     = m;

            remaining_ = blocksize - WORDSIZE;
            loc_       = static_cast<char*>(m) + WORDSIZE;
        }
        void* rloc = loc_;
        loc_       = static_cast<char*>(loc_) + size;
        remaining_ -= size;

        usedMemory += size;

        return rloc;
    }

    /**
     * Allocates (using this pool) a generic type T.
     *
     * Params:
     *     count = number of instances to allocate.
     * Returns: pointer (of type T*) to memory buffer
     */
    template <typename T>
    T* allocate(const size_t count = 1)
    {
        T* mem = static_cast<T*>(this->malloc(sizeof(T) * count));
        return mem;
    }
};
/** @} */

/** @addtogroup nanoflann_metaprog_grp Auxiliary metaprogramming stuff
 * @{ */

/** Used to declare fixed-size arrays when DIM>0, dynamically-allocated vectors
 * when DIM=-1. Fixed size version for a generic DIM:
 */
template <int32_t DIM, typename T>
struct array_or_vector
{
    using type = std::array<T, DIM>;
};
/** Dynamic size version */
template <typename T>
struct array_or_vector<-1, T>
{
    using type = std::vector<T>;
};

/** @} */

/** kd-tree base-class
 *
 * Contains the member functions common to the classes KDTreeSingleIndexAdaptor
 * and KDTreeSingleIndexDynamicAdaptor_.
 *
 * \tparam Derived The name of the class which inherits this class.
 * \tparam DatasetAdaptor The user-provided adaptor, which must be ensured to
 *         have a lifetime equal or longer than the instance of this class.
 * \tparam Distance The distance metric to use, these are all classes derived
 * from nanoflann::Metric
 * \tparam DIM Dimensionality of data points (e.g. 3 for 3D points)
 * \tparam IndexType Type of the arguments with which the data can be
 * accessed (e.g. float, double, int64_t, T*)
 */
template <
    class Derived, typename Distance, class DatasetAdaptor, int32_t DIM = -1,
    typename IndexType = uint32_t>
class KDTreeBaseClass
{
   public:
    /** Frees the previously-built index. Automatically called within
     * buildIndex(). */
    void freeIndex(Derived& obj)
    {
        obj.pool_.free_all();
        obj.root_node_           = nullptr;
        obj.size_at_index_build_ = 0;
    }

    using ElementType  = typename Distance::ElementType;
    using DistanceType = typename Distance::DistanceType;

    /**
     *  Array of indices to vectors in the dataset_.
     */
    std::vector<IndexType> vAcc_;

    using Offset    = typename decltype(vAcc_)::size_type;
    using Size      = typename decltype(vAcc_)::size_type;
    using Dimension = int32_t;

    /*---------------------------
     * Internal Data Structures
     * --------------------------*/
    struct Node
    {
        /** Union used because a node can be either a LEAF node or a non-leaf
         * node, so both data fields are never used simultaneously */
        union
        {
            struct leaf
            {
                Offset left, right;  //!< Indices of points in leaf node
            } lr;
            struct nonleaf
            {
                Dimension divfeat;  //!< Dimension used for subdivision.
                /// The values used for subdivision.
                DistanceType divlow, divhigh;
            } sub;
        } node_type;

        /** Child nodes (both=nullptr mean its a leaf node) */
        Node *child1 = nullptr, *child2 = nullptr;
    };

    using NodePtr      = Node*;
    using NodeConstPtr = const Node*;

    struct Interval
    {
        ElementType low, high;
    };

    NodePtr root_node_ = nullptr;

    Size leaf_max_size_ = 0;

    /// Number of thread for concurrent tree build
    Size n_thread_build_ = 1;
    /// Number of current points in the dataset
    Size size_ = 0;
    /// Number of points in the dataset when the index was built
    Size      size_at_index_build_ = 0;
    Dimension dim_                 = 0;  //!< Dimensionality of each data point

    /** Define "BoundingBox" as a fixed-size or variable-size container
     * depending on "DIM" */
    using BoundingBox = typename array_or_vector<DIM, Interval>::type;

    /** Define "distance_vector_t" as a fixed-size or variable-size container
     * depending on "DIM" */
    using distance_vector_t = typename array_or_vector<DIM, DistanceType>::type;

    /** The KD-tree used to find neighbours */
    BoundingBox root_bbox_;

    /**
     * Pooled memory allocator.
     *
     * Using a pooled memory allocator is more efficient
     * than allocating memory directly when there is a large
     * number small of memory allocations.
     */
    PooledAllocator pool_;

    /** Returns number of points in dataset  */
    Size size(const Derived& obj) const { return obj.size_; }

    /** Returns the length of each point in the dataset */
    Size veclen(const Derived& obj) { return DIM > 0 ? DIM : obj.dim; }

    /// Helper accessor to the dataset points:
    ElementType dataset_get(
        const Derived& obj, IndexType element, Dimension component) const
    {
        return obj.dataset_.kdtree_get_pt(element, component);
    }

    /**
     * Computes the inde memory usage
     * Returns: memory used by the index
     */
    Size usedMemory(Derived& obj)
    {
        return obj.pool_.usedMemory + obj.pool_.wastedMemory +
               obj.dataset_.kdtree_get_point_count() *
                   sizeof(IndexType);  // pool memory and vind array memory
    }

    void computeMinMax(
        const Derived& obj, Offset ind, Size count, Dimension element,
        ElementType& min_elem, ElementType& max_elem)
    {
        min_elem = dataset_get(obj, vAcc_[ind], element);
        max_elem = min_elem;
        for (Offset i = 1; i < count; ++i)
        {
            ElementType val = dataset_get(obj, vAcc_[ind + i], element);
            if (val < min_elem) min_elem = val;
            if (val > max_elem) max_elem = val;
        }
    }

    /**
     * Create a tree node that subdivides the list of vecs from vind[first]
     * to vind[last].  The routine is called recursively on each sublist.
     *
     * @param left index of the first vector
     * @param right index of the last vector
     */
    NodePtr divideTree(
        Derived& obj, const Offset left, const Offset right, BoundingBox& bbox)
    {
        NodePtr node = obj.pool_.template allocate<Node>();  // allocate memory
        const auto dims = (DIM > 0 ? DIM : obj.dim_);

        /* If too few exemplars remain, then make this a leaf node. */
        if ((right - left) <= static_cast<Offset>(obj.leaf_max_size_))
        {
            node->child1 = node->child2 = nullptr; /* Mark as leaf node. */
            node->node_type.lr.left     = left;
            node->node_type.lr.right    = right;

            // compute bounding-box of leaf points
            for (Dimension i = 0; i < dims; ++i)
            {
                bbox[i].low  = dataset_get(obj, obj.vAcc_[left], i);
                bbox[i].high = dataset_get(obj, obj.vAcc_[left], i);
            }
            for (Offset k = left + 1; k < right; ++k)
            {
                for (Dimension i = 0; i < dims; ++i)
                {
                    const auto val = dataset_get(obj, obj.vAcc_[k], i);
                    if (bbox[i].low > val) bbox[i].low = val;
                    if (bbox[i].high < val) bbox[i].high = val;
                }
            }
        }
        else
        {
            Offset       idx;
            Dimension    cutfeat;
            DistanceType cutval;
            middleSplit_(obj, left, right - left, idx, cutfeat, cutval, bbox);

            node->node_type.sub.divfeat = cutfeat;

            BoundingBox left_bbox(bbox);
            left_bbox[cutfeat].high = cutval;
            node->child1 = this->divideTree(obj, left, left + idx, left_bbox);

            BoundingBox right_bbox(bbox);
            right_bbox[cutfeat].low = cutval;
            node->child2 = this->divideTree(obj, left + idx, right, right_bbox);

            node->node_type.sub.divlow  = left_bbox[cutfeat].high;
            node->node_type.sub.divhigh = right_bbox[cutfeat].low;

            for (Dimension i = 0; i < dims; ++i)
            {
                bbox[i].low  = std::min(left_bbox[i].low, right_bbox[i].low);
                bbox[i].high = std::max(left_bbox[i].high, right_bbox[i].high);
            }
        }

        return node;
    }

    /**
     * Create a tree node that subdivides the list of vecs from vind[first] to
     * vind[last] concurrently.  The routine is called recursively on each
     * sublist.
     *
     * @param left index of the first vector
     * @param right index of the last vector
     * @param thread_count count of std::async threads
     * @param mutex mutex for mempool allocation
     */
    NodePtr divideTreeConcurrent(
        Derived& obj, const Offset left, const Offset right, BoundingBox& bbox,
        std::atomic<unsigned int>& thread_count, std::mutex& mutex)
    {
        std::unique_lock<std::mutex> lock(mutex);
        NodePtr node = obj.pool_.template allocate<Node>();  // allocate memory
        lock.unlock();

        const auto dims = (DIM > 0 ? DIM : obj.dim_);

        /* If too few exemplars remain, then make this a leaf node. */
        if ((right - left) <= static_cast<Offset>(obj.leaf_max_size_))
        {
            node->child1 = node->child2 = nullptr; /* Mark as leaf node. */
            node->node_type.lr.left     = left;
            node->node_type.lr.right    = right;

            // compute bounding-box of leaf points
            for (Dimension i = 0; i < dims; ++i)
            {
                bbox[i].low  = dataset_get(obj, obj.vAcc_[left], i);
                bbox[i].high = dataset_get(obj, obj.vAcc_[left], i);
            }
            for (Offset k = left + 1; k < right; ++k)
            {
                for (Dimension i = 0; i < dims; ++i)
                {
                    const auto val = dataset_get(obj, obj.vAcc_[k], i);
                    if (bbox[i].low > val) bbox[i].low = val;
                    if (bbox[i].high < val) bbox[i].high = val;
                }
            }
        }
        else
        {
            Offset       idx;
            Dimension    cutfeat;
            DistanceType cutval;
            middleSplit_(obj, left, right - left, idx, cutfeat, cutval, bbox);

            node->node_type.sub.divfeat = cutfeat;

            std::future<NodePtr> left_future, right_future;

            BoundingBox left_bbox(bbox);
            left_bbox[cutfeat].high = cutval;
            if (++thread_count < n_thread_build_)
            {
                left_future = std::async(
                    std::launch::async, &KDTreeBaseClass::divideTreeConcurrent,
                    this, std::ref(obj), left, left + idx, std::ref(left_bbox),
                    std::ref(thread_count), std::ref(mutex));
            }
            else
            {
                --thread_count;
                node->child1 = this->divideTreeConcurrent(
                    obj, left, left + idx, left_bbox, thread_count, mutex);
            }

            BoundingBox right_bbox(bbox);
            right_bbox[cutfeat].low = cutval;
            if (++thread_count < n_thread_build_)
            {
                right_future = std::async(
                    std::launch::async, &KDTreeBaseClass::divideTreeConcurrent,
                    this, std::ref(obj), left + idx, right,
                    std::ref(right_bbox), std::ref(thread_count),
                    std::ref(mutex));
            }
            else
            {
                --thread_count;
                node->child2 = this->divideTreeConcurrent(
                    obj, left + idx, right, right_bbox, thread_count, mutex);
            }

            if (left_future.valid())
            {
                node->child1 = left_future.get();
                --thread_count;
            }
            if (right_future.valid())
            {
                node->child2 = right_future.get();
                --thread_count;
            }

            node->node_type.sub.divlow  = left_bbox[cutfeat].high;
            node->node_type.sub.divhigh = right_bbox[cutfeat].low;

            for (Dimension i = 0; i < dims; ++i)
            {
                bbox[i].low  = std::min(left_bbox[i].low, right_bbox[i].low);
                bbox[i].high = std::max(left_bbox[i].high, right_bbox[i].high);
            }
        }

        return node;
    }

    void middleSplit_(
        const Derived& obj, const Offset ind, const Size count, Offset& index,
        Dimension& cutfeat, DistanceType& cutval, const BoundingBox& bbox)
    {
        const auto  dims     = (DIM > 0 ? DIM : obj.dim_);
        const auto  EPS      = static_cast<DistanceType>(0.00001);
        ElementType max_span = bbox[0].high - bbox[0].low;
        for (Dimension i = 1; i < dims; ++i)
        {
            ElementType span = bbox[i].high - bbox[i].low;
            if (span > max_span) { max_span = span; }
        }
        ElementType max_spread = -1;
        cutfeat                = 0;
        ElementType min_elem = 0, max_elem = 0;
        for (Dimension i = 0; i < dims; ++i)
        {
            ElementType span = bbox[i].high - bbox[i].low;
            if (span > (1 - EPS) * max_span)
            {
                ElementType min_elem_, max_elem_;
                computeMinMax(obj, ind, count, i, min_elem_, max_elem_);
                ElementType spread = max_elem_ - min_elem_;
                if (spread > max_spread)
                {
                    cutfeat    = i;
                    max_spread = spread;
                    min_elem   = min_elem_;
                    max_elem   = max_elem_;
                }
            }
        }
        // split in the middle
        DistanceType split_val = (bbox[cutfeat].low + bbox[cutfeat].high) / 2;

        if (split_val < min_elem)
            cutval = min_elem;
        else if (split_val > max_elem)
            cutval = max_elem;
        else
            cutval = split_val;

        Offset lim1, lim2;
        planeSplit(obj, ind, count, cutfeat, cutval, lim1, lim2);

        if (lim1 > count / 2)
            index = lim1;
        else if (lim2 < count / 2)
            index = lim2;
        else
            index = count / 2;
    }

    /**
     *  Subdivide the list of points by a plane perpendicular on the axis
     * corresponding to the 'cutfeat' dimension at 'cutval' position.
     *
     *  On return:
     *  dataset[ind[0..lim1-1]][cutfeat]<cutval
     *  dataset[ind[lim1..lim2-1]][cutfeat]==cutval
     *  dataset[ind[lim2..count]][cutfeat]>cutval
     */
    void planeSplit(
        const Derived& obj, const Offset ind, const Size count,
        const Dimension cutfeat, const DistanceType& cutval, Offset& lim1,
        Offset& lim2)
    {
        /* Move vector indices for left subtree to front of list. */
        Offset left  = 0;
        Offset right = count - 1;
        for (;;)
        {
            while (left <= right &&
                   dataset_get(obj, vAcc_[ind + left], cutfeat) < cutval)
                ++left;
            while (right && left <= right &&
                   dataset_get(obj, vAcc_[ind + right], cutfeat) >= cutval)
                --right;
            if (left > right || !right)
                break;  // "!right" was added to support unsigned Index types
            std::swap(vAcc_[ind + left], vAcc_[ind + right]);
            ++left;
            --right;
        }
        /* If either list is empty, it means that all remaining features
         * are identical. Split in the middle to maintain a balanced tree.
         */
        lim1  = left;
        right = count - 1;
        for (;;)
        {
            while (left <= right &&
                   dataset_get(obj, vAcc_[ind + left], cutfeat) <= cutval)
                ++left;
            while (right && left <= right &&
                   dataset_get(obj, vAcc_[ind + right], cutfeat) > cutval)
                --right;
            if (left > right || !right)
                break;  // "!right" was added to support unsigned Index types
            std::swap(vAcc_[ind + left], vAcc_[ind + right]);
            ++left;
            --right;
        }
        lim2 = left;
    }

    DistanceType computeInitialDistances(
        const Derived& obj, const ElementType* vec,
        distance_vector_t& dists) const
    {
        assert(vec);
        DistanceType dist = DistanceType();

        for (Dimension i = 0; i < (DIM > 0 ? DIM : obj.dim_); ++i)
        {
            if (vec[i] < obj.root_bbox_[i].low)
            {
                dists[i] =
                    obj.distance_.accum_dist(vec[i], obj.root_bbox_[i].low, i);
                dist += dists[i];
            }
            if (vec[i] > obj.root_bbox_[i].high)
            {
                dists[i] =
                    obj.distance_.accum_dist(vec[i], obj.root_bbox_[i].high, i);
                dist += dists[i];
            }
        }
        return dist;
    }

    static void save_tree(
        const Derived& obj, std::ostream& stream, const NodeConstPtr tree)
    {
        save_value(stream, *tree);
        if (tree->child1 != nullptr) { save_tree(obj, stream, tree->child1); }
        if (tree->child2 != nullptr) { save_tree(obj, stream, tree->child2); }
    }

    static void load_tree(Derived& obj, std::istream& stream, NodePtr& tree)
    {
        tree = obj.pool_.template allocate<Node>();
        load_value(stream, *tree);
        if (tree->child1 != nullptr) { load_tree(obj, stream, tree->child1); }
        if (tree->child2 != nullptr) { load_tree(obj, stream, tree->child2); }
    }

    /**  Stores the index in a binary file.
     *   IMPORTANT NOTE: The set of data points is NOT stored in the file, so
     * when loading the index object it must be constructed associated to the
     * same source of data points used while building it. See the example:
     * examples/saveload_example.cpp \sa loadIndex  */
    void saveIndex(const Derived& obj, std::ostream& stream) const
    {
        save_value(stream, obj.size_);
        save_value(stream, obj.dim_);
        save_value(stream, obj.root_bbox_);
        save_value(stream, obj.leaf_max_size_);
        save_value(stream, obj.vAcc_);
        if (obj.root_node_) save_tree(obj, stream, obj.root_node_);
    }

    /**  Loads a previous index from a binary file.
     *   IMPORTANT NOTE: The set of data points is NOT stored in the file, so
     * the index object must be constructed associated to the same source of
     * data points used while building the index. See the example:
     * examples/saveload_example.cpp \sa loadIndex  */
    void loadIndex(Derived& obj, std::istream& stream)
    {
        load_value(stream, obj.size_);
        load_value(stream, obj.dim_);
        load_value(stream, obj.root_bbox_);
        load_value(stream, obj.leaf_max_size_);
        load_value(stream, obj.vAcc_);
        load_tree(obj, stream, obj.root_node_);
    }
};

/** @addtogroup kdtrees_grp KD-tree classes and adaptors
 * @{ */

/** kd-tree static index
 *
 * Contains the k-d trees and other information for indexing a set of points
 * for nearest-neighbor matching.
 *
 *  The class "DatasetAdaptor" must provide the following interface (can be
 * non-virtual, inlined methods):
 *
 *  \code
 *   // Must return the number of data poins
 *   size_t kdtree_get_point_count() const { ... }
 *
 *
 *   // Must return the dim'th component of the idx'th point in the class:
 *   T kdtree_get_pt(const size_t idx, const size_t dim) const { ... }
 *
 *   // Optional bounding-box computation: return false to default to a standard
 * bbox computation loop.
 *   //   Return true if the BBOX was already computed by the class and returned
 * in "bb" so it can be avoided to redo it again.
 *   //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3
 * for point clouds) template <class BBOX> bool kdtree_get_bbox(BBOX &bb) const
 *   {
 *      bb[0].low = ...; bb[0].high = ...;  // 0th dimension limits
 *      bb[1].low = ...; bb[1].high = ...;  // 1st dimension limits
 *      ...
 *      return true;
 *   }
 *
 *  \endcode
 *
 * \tparam DatasetAdaptor The user-provided adaptor, which must be ensured to
 *         have a lifetime equal or longer than the instance of this class.
 * \tparam Distance The distance metric to use: nanoflann::metric_L1,
 * nanoflann::metric_L2, nanoflann::metric_L2_Simple, etc. \tparam DIM
 * Dimensionality of data points (e.g. 3 for 3D points) \tparam IndexType Will
 * be typically size_t or int
 */
template <
    typename Distance, class DatasetAdaptor, int32_t DIM = -1,
    typename IndexType = uint32_t>
class KDTreeSingleIndexAdaptor
    : public KDTreeBaseClass<
          KDTreeSingleIndexAdaptor<Distance, DatasetAdaptor, DIM, IndexType>,
          Distance, DatasetAdaptor, DIM, IndexType>
{
   public:
    /** Deleted copy constructor*/
    explicit KDTreeSingleIndexAdaptor(
        const KDTreeSingleIndexAdaptor<
            Distance, DatasetAdaptor, DIM, IndexType>&) = delete;

    /** The data source used by this index */
    const DatasetAdaptor& dataset_;

    const KDTreeSingleIndexAdaptorParams indexParams;

    Distance distance_;

    using Base = typename nanoflann::KDTreeBaseClass<
        nanoflann::KDTreeSingleIndexAdaptor<
            Distance, DatasetAdaptor, DIM, IndexType>,
        Distance, DatasetAdaptor, DIM, IndexType>;

    using Offset    = typename Base::Offset;
    using Size      = typename Base::Size;
    using Dimension = typename Base::Dimension;

    using ElementType  = typename Base::ElementType;
    using DistanceType = typename Base::DistanceType;

    using Node    = typename Base::Node;
    using NodePtr = Node*;

    using Interval = typename Base::Interval;

    /** Define "BoundingBox" as a fixed-size or variable-size container
     * depending on "DIM" */
    using BoundingBox = typename Base::BoundingBox;

    /** Define "distance_vector_t" as a fixed-size or variable-size container
     * depending on "DIM" */
    using distance_vector_t = typename Base::distance_vector_t;

    /**
     * KDTree constructor
     *
     * Refer to docs in README.md or online in
     * https://github.com/jlblancoc/nanoflann
     *
     * The KD-Tree point dimension (the length of each point in the datase, e.g.
     * 3 for 3D points) is determined by means of:
     *  - The \a DIM template parameter if >0 (highest priority)
     *  - Otherwise, the \a dimensionality parameter of this constructor.
     *
     * @param inputData Dataset with the input features. Its lifetime must be
     *  equal or longer than that of the instance of this class.
     * @param params Basically, the maximum leaf node size
     *
     * Note that there is a variable number of optional additional parameters
     * which will be forwarded to the metric class constructor. Refer to example
     * `examples/pointcloud_custom_metric.cpp` for a use case.
     *
     */
    template <class... Args>
    explicit KDTreeSingleIndexAdaptor(
        const Dimension dimensionality, const DatasetAdaptor& inputData,
        const KDTreeSingleIndexAdaptorParams& params, Args&&... args)
        : dataset_(inputData),
          indexParams(params),
          distance_(inputData, std::forward<Args>(args)...)
    {
        init(dimensionality, params);
    }

    explicit KDTreeSingleIndexAdaptor(
        const Dimension dimensionality, const DatasetAdaptor& inputData,
        const KDTreeSingleIndexAdaptorParams& params = {})
        : dataset_(inputData), indexParams(params), distance_(inputData)
    {
        init(dimensionality, params);
    }

   private:
    void init(
        const Dimension                       dimensionality,
        const KDTreeSingleIndexAdaptorParams& params)
    {
        Base::size_                = dataset_.kdtree_get_point_count();
        Base::size_at_index_build_ = Base::size_;
        Base::dim_                 = dimensionality;
        if (DIM > 0) Base::dim_ = DIM;
        Base::leaf_max_size_ = params.leaf_max_size;
        if (params.n_thread_build > 0)
        {
            Base::n_thread_build_ = params.n_thread_build;
        }
        else
        {
            Base::n_thread_build_ =
                std::max(std::thread::hardware_concurrency(), 1u);
        }

        if (!(params.flags &
              KDTreeSingleIndexAdaptorFlags::SkipInitialBuildIndex))
        {
            // Build KD-tree:
            buildIndex();
        }
    }

   public:
    /**
     * Builds the index
     */
    void buildIndex()
    {
        Base::size_                = dataset_.kdtree_get_point_count();
        Base::size_at_index_build_ = Base::size_;
        init_vind();
        this->freeIndex(*this);
        Base::size_at_index_build_ = Base::size_;
        if (Base::size_ == 0) return;
        computeBoundingBox(Base::root_bbox_);
        // construct the tree
        if (Base::n_thread_build_ == 1)
        {
            Base::root_node_ =
                this->divideTree(*this, 0, Base::size_, Base::root_bbox_);
        }
        else
        {
            std::atomic<unsigned int> thread_count(0u);
            std::mutex                mutex;
            Base::root_node_ = this->divideTreeConcurrent(
                *this, 0, Base::size_, Base::root_bbox_, thread_count, mutex);
        }
    }

    /** \name Query methods
     * @{ */

    /**
     * Find set of nearest neighbors to vec[0:dim-1]. Their indices are stored
     * inside the result object.
     *
     * Params:
     *     result = the result object in which the indices of the
     * nearest-neighbors are stored vec = the vector for which to search the
     * nearest neighbors
     *
     * \tparam RESULTSET Should be any ResultSet<DistanceType>
     * \return  True if the requested neighbors could be found.
     * \sa knnSearch, radiusSearch
     *
     * \note If L2 norms are used, all returned distances are actually squared
     *       distances.
     */
    template <typename RESULTSET>
    bool findNeighbors(
        RESULTSET& result, const ElementType* vec,
        const SearchParameters& searchParams = {}) const
    {
        assert(vec);
        if (this->size(*this) == 0) return false;
        if (!Base::root_node_) {
#if !NANOFLANN_NO_EXCEPTIONS
            throw std::runtime_error(
                "[nanoflann] findNeighbors() called before building the "
                "index.");
#endif
        }
        float epsError = 1 + searchParams.eps;

        // fixed or variable-sized container (depending on DIM)
        distance_vector_t dists;
        // Fill it with zeros.
        auto zero = static_cast<decltype(result.worstDist())>(0);
        assign(dists, (DIM > 0 ? DIM : Base::dim_), zero);
        DistanceType dist = this->computeInitialDistances(*this, vec, dists);
        searchLevel(result, vec, Base::root_node_, dist, dists, epsError);
        return result.full();
    }

    /**
     * Find the "num_closest" nearest neighbors to the \a query_point[0:dim-1].
     * Their indices and distances are stored in the provided pointers to
     * array/vector.
     *
     * \sa radiusSearch, findNeighbors
     * \return Number `N` of valid points in the result set.
     *
     * \note If L2 norms are used, all returned distances are actually squared
     *       distances.
     *
     * \note Only the first `N` entries in `out_indices` and `out_distances`
     *       will be valid. Return is less than `num_closest` only if the
     *       number of elements in the tree is less than `num_closest`.
     */
    Size knnSearch(
        const ElementType* query_point, const Size num_closest,
        IndexType* out_indices, DistanceType* out_distances) const
    {
        nanoflann::KNNResultSet<DistanceType, IndexType> resultSet(num_closest);
        resultSet.init(out_indices, out_distances);
        findNeighbors(resultSet, query_point);
        return resultSet.size();
    }

    /**
     * Find all the neighbors to \a query_point[0:dim-1] within a maximum
     * radius. The output is given as a vector of pairs, of which the first
     * element is a point index and the second the corresponding distance.
     * Previous contents of \a IndicesDists are cleared.
     *
     *  If searchParams.sorted==true, the output list is sorted by ascending
     * distances.
     *
     *  For a better performance, it is advisable to do a .reserve() on the
     * vector if you have any wild guess about the number of expected matches.
     *
     *  \sa knnSearch, findNeighbors, radiusSearchCustomCallback
     * \return The number of points within the given radius (i.e. indices.size()
     * or dists.size() )
     *
     * \note If L2 norms are used, search radius and all returned distances
     *       are actually squared distances.
     */
    Size radiusSearch(
        const ElementType* query_point, const DistanceType& radius,
        std::vector<ResultItem<IndexType, DistanceType>>& IndicesDists,
        const SearchParameters& searchParams = {}) const
    {
        RadiusResultSet<DistanceType, IndexType> resultSet(
            radius, IndicesDists);
        const Size nFound =
            radiusSearchCustomCallback(query_point, resultSet, searchParams);
        if (searchParams.sorted)
            std::sort(
                IndicesDists.begin(), IndicesDists.end(), IndexDist_Sorter());
        return nFound;
    }

    /**
     * Just like radiusSearch() but with a custom callback class for each point
     * found in the radius of the query. See the source of RadiusResultSet<> as
     * a start point for your own classes. \sa radiusSearch
     */
    template <class SEARCH_CALLBACK>
    Size radiusSearchCustomCallback(
        const ElementType* query_point, SEARCH_CALLBACK& resultSet,
        const SearchParameters& searchParams = {}) const
    {
        findNeighbors(resultSet, query_point, searchParams);
        return resultSet.size();
    }

    /**
     * Find the first N neighbors to \a query_point[0:dim-1] within a maximum
     * radius. The output is given as a vector of pairs, of which the first
     * element is a point index and the second the corresponding distance.
     * Previous contents of \a IndicesDists are cleared.
     *
     * \sa radiusSearch, findNeighbors
     * \return Number `N` of valid points in the result set.
     *
     * \note If L2 norms are used, all returned distances are actually squared
     *       distances.
     *
     * \note Only the first `N` entries in `out_indices` and `out_distances`
     *       will be valid. Return is less than `num_closest` only if the
     *       number of elements in the tree is less than `num_closest`.
     */
    Size rknnSearch(
        const ElementType* query_point, const Size num_closest,
        IndexType* out_indices, DistanceType* out_distances,
        const DistanceType& radius) const
    {
        nanoflann::RKNNResultSet<DistanceType, IndexType> resultSet(
            num_closest, radius);
        resultSet.init(out_indices, out_distances);
        findNeighbors(resultSet, query_point);
        return resultSet.size();
    }

    /** @} */

   public:
    /** Make sure the auxiliary list \a vind has the same size than the current
     * dataset, and re-generate if size has changed. */
    void init_vind()
    {
        // Create a permutable array of indices to the input vectors.
        Base::size_ = dataset_.kdtree_get_point_count();
        if (Base::vAcc_.size() != Base::size_) Base::vAcc_.resize(Base::size_);
        for (Size i = 0; i < Base::size_; i++) Base::vAcc_[i] = i;
    }

    void computeBoundingBox(BoundingBox& bbox)
    {
        const auto dims = (DIM > 0 ? DIM : Base::dim_);
        resize(bbox, dims);
        if (dataset_.kdtree_get_bbox(bbox))
        {
            // Done! It was implemented in derived class
        }
        else
        {
            const Size N = dataset_.kdtree_get_point_count();
            if (!N) {
#if !NANOFLANN_NO_EXCEPTIONS
                throw std::runtime_error(
                    "[nanoflann] computeBoundingBox() called but "
                    "no data points found.");
#endif
            }
            for (Dimension i = 0; i < dims; ++i)
            {
                bbox[i].low = bbox[i].high =
                    this->dataset_get(*this, Base::vAcc_[0], i);
            }
            for (Offset k = 1; k < N; ++k)
            {
                for (Dimension i = 0; i < dims; ++i)
                {
                    const auto val =
                        this->dataset_get(*this, Base::vAcc_[k], i);
                    if (val < bbox[i].low) bbox[i].low = val;
                    if (val > bbox[i].high) bbox[i].high = val;
                }
            }
        }
    }

    /**
     * Performs an exact search in the tree starting from a node.
     * \tparam RESULTSET Should be any ResultSet<DistanceType>
     * \return true if the search should be continued, false if the results are
     * sufficient
     */
    template <class RESULTSET>
    bool searchLevel(
        RESULTSET& result_set, const ElementType* vec, const NodePtr node,
        DistanceType mindist, distance_vector_t& dists,
        const float epsError) const
    {
        /* If this is a leaf node, then do check and return. */
        if ((node->child1 == nullptr) && (node->child2 == nullptr))
        {
            DistanceType worst_dist = result_set.worstDist();
            for (Offset i = node->node_type.lr.left;
                 i < node->node_type.lr.right; ++i)
            {
                const IndexType accessor = Base::vAcc_[i];  // reorder... : i;
                DistanceType    dist     = distance_.evalMetric(
                    vec, accessor, (DIM > 0 ? DIM : Base::dim_));
                if (dist < worst_dist)
                {
                    if (!result_set.addPoint(dist, Base::vAcc_[i]))
                    {
                        // the resultset doesn't want to receive any more
                        // points, we're done searching!
                        return false;
                    }
                }
            }
            return true;
        }

        /* Which child branch should be taken first? */
        Dimension    idx   = node->node_type.sub.divfeat;
        ElementType  val   = vec[idx];
        DistanceType diff1 = val - node->node_type.sub.divlow;
        DistanceType diff2 = val - node->node_type.sub.divhigh;

        NodePtr      bestChild;
        NodePtr      otherChild;
        DistanceType cut_dist;
        if ((diff1 + diff2) < 0)
        {
            bestChild  = node->child1;
            otherChild = node->child2;
            cut_dist =
                distance_.accum_dist(val, node->node_type.sub.divhigh, idx);
        }
        else
        {
            bestChild  = node->child2;
            otherChild = node->child1;
            cut_dist =
                distance_.accum_dist(val, node->node_type.sub.divlow, idx);
        }

        /* Call recursively to search next level down. */
        if (!searchLevel(result_set, vec, bestChild, mindist, dists, epsError))
        {
            // the resultset doesn't want to receive any more points, we're done
            // searching!
            return false;
        }

        DistanceType dst = dists[idx];
        mindist          = mindist + cut_dist - dst;
        dists[idx]       = cut_dist;
        if (mindist * epsError <= result_set.worstDist())
        {
            if (!searchLevel(
                    result_set, vec, otherChild, mindist, dists, epsError))
            {
                // the resultset doesn't want to receive any more points, we're
                // done searching!
                return false;
            }
        }
        dists[idx] = dst;
        return true;
    }

   public:
    /**  Stores the index in a binary file.
     *   IMPORTANT NOTE: The set of data points is NOT stored in the file, so
     * when loading the index object it must be constructed associated to the
     * same source of data points used while building it. See the example:
     * examples/saveload_example.cpp \sa loadIndex  */
    void saveIndex(std::ostream& stream) const
    {
        Base::saveIndex(*this, stream);
    }

    /**  Loads a previous index from a binary file.
     *   IMPORTANT NOTE: The set of data points is NOT stored in the file, so
     * the index object must be constructed associated to the same source of
     * data points used while building the index. See the example:
     * examples/saveload_example.cpp \sa loadIndex  */
    void loadIndex(std::istream& stream) { Base::loadIndex(*this, stream); }

};  // class KDTree

/** kd-tree dynamic index
 *
 * Contains the k-d trees and other information for indexing a set of points
 * for nearest-neighbor matching.
 *
 * The class "DatasetAdaptor" must provide the following interface (can be
 * non-virtual, inlined methods):
 *
 *  \code
 *   // Must return the number of data poins
 *   size_t kdtree_get_point_count() const { ... }
 *
 *   // Must return the dim'th component of the idx'th point in the class:
 *   T kdtree_get_pt(const size_t idx, const size_t dim) const { ... }
 *
 *   // Optional bounding-box computation: return false to default to a standard
 * bbox computation loop.
 *   //   Return true if the BBOX was already computed by the class and returned
 * in "bb" so it can be avoided to redo it again.
 *   //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3
 * for point clouds) template <class BBOX> bool kdtree_get_bbox(BBOX &bb) const
 *   {
 *      bb[0].low = ...; bb[0].high = ...;  // 0th dimension limits
 *      bb[1].low = ...; bb[1].high = ...;  // 1st dimension limits
 *      ...
 *      return true;
 *   }
 *
 *  \endcode
 *
 * \tparam DatasetAdaptor The user-provided adaptor (see comments above).
 * \tparam Distance The distance metric to use: nanoflann::metric_L1,
 * nanoflann::metric_L2, nanoflann::metric_L2_Simple, etc.
 * \tparam DIM Dimensionality of data points (e.g. 3 for 3D points)
 * \tparam IndexType Type of the arguments with which the data can be
 * accessed (e.g. float, double, int64_t, T*)
 */
template <
    typename Distance, class DatasetAdaptor, int32_t DIM = -1,
    typename IndexType = uint32_t>
class KDTreeSingleIndexDynamicAdaptor_
    : public KDTreeBaseClass<
          KDTreeSingleIndexDynamicAdaptor_<
              Distance, DatasetAdaptor, DIM, IndexType>,
          Distance, DatasetAdaptor, DIM, IndexType>
{
   public:
    /**
     * The dataset used by this index
     */
    const DatasetAdaptor& dataset_;  //!< The source of our data

    KDTreeSingleIndexAdaptorParams index_params_;

    std::vector<int>& treeIndex_;

    Distance distance_;

    using Base = typename nanoflann::KDTreeBaseClass<
        nanoflann::KDTreeSingleIndexDynamicAdaptor_<
            Distance, DatasetAdaptor, DIM, IndexType>,
        Distance, DatasetAdaptor, DIM, IndexType>;

    using ElementType  = typename Base::ElementType;
    using DistanceType = typename Base::DistanceType;

    using Offset    = typename Base::Offset;
    using Size      = typename Base::Size;
    using Dimension = typename Base::Dimension;

    using Node    = typename Base::Node;
    using NodePtr = Node*;

    using Interval = typename Base::Interval;
    /** Define "BoundingBox" as a fixed-size or variable-size container
     * depending on "DIM" */
    using BoundingBox = typename Base::BoundingBox;

    /** Define "distance_vector_t" as a fixed-size or variable-size container
     * depending on "DIM" */
    using distance_vector_t = typename Base::distance_vector_t;

    /**
     * KDTree constructor
     *
     * Refer to docs in README.md or online in
     * https://github.com/jlblancoc/nanoflann
     *
     * The KD-Tree point dimension (the length of each point in the datase, e.g.
     * 3 for 3D points) is determined by means of:
     *  - The \a DIM template parameter if >0 (highest priority)
     *  - Otherwise, the \a dimensionality parameter of this constructor.
     *
     * @param inputData Dataset with the input features. Its lifetime must be
     *  equal or longer than that of the instance of this class.
     * @param params Basically, the maximum leaf node size
     */
    KDTreeSingleIndexDynamicAdaptor_(
        const Dimension dimensionality, const DatasetAdaptor& inputData,
        std::vector<int>&                     treeIndex,
        const KDTreeSingleIndexAdaptorParams& params =
            KDTreeSingleIndexAdaptorParams())
        : dataset_(inputData),
          index_params_(params),
          treeIndex_(treeIndex),
          distance_(inputData)
    {
        Base::size_                = 0;
        Base::size_at_index_build_ = 0;
        for (auto& v : Base::root_bbox_) v = {};
        Base::dim_ = dimensionality;
        if (DIM > 0) Base::dim_ = DIM;
        Base::leaf_max_size_ = params.leaf_max_size;
        if (params.n_thread_build > 0)
        {
            Base::n_thread_build_ = params.n_thread_build;
        }
        else
        {
            Base::n_thread_build_ =
                std::max(std::thread::hardware_concurrency(), 1u);
        }
    }

    /** Explicitly default the copy constructor */
    KDTreeSingleIndexDynamicAdaptor_(
        const KDTreeSingleIndexDynamicAdaptor_& rhs) = default;

    /** Assignment operator definiton */
    KDTreeSingleIndexDynamicAdaptor_ operator=(
        const KDTreeSingleIndexDynamicAdaptor_& rhs)
    {
        KDTreeSingleIndexDynamicAdaptor_ tmp(rhs);
        std::swap(Base::vAcc_, tmp.Base::vAcc_);
        std::swap(Base::leaf_max_size_, tmp.Base::leaf_max_size_);
        std::swap(index_params_, tmp.index_params_);
        std::swap(treeIndex_, tmp.treeIndex_);
        std::swap(Base::size_, tmp.Base::size_);
        std::swap(Base::size_at_index_build_, tmp.Base::size_at_index_build_);
        std::swap(Base::root_node_, tmp.Base::root_node_);
        std::swap(Base::root_bbox_, tmp.Base::root_bbox_);
        std::swap(Base::pool_, tmp.Base::pool_);
        return *this;
    }

    /**
     * Builds the index
     */
    void buildIndex()
    {
        Base::size_ = Base::vAcc_.size();
        this->freeIndex(*this);
        Base::size_at_index_build_ = Base::size_;
        if (Base::size_ == 0) return;
        computeBoundingBox(Base::root_bbox_);
        // construct the tree
        if (Base::n_thread_build_ == 1)
        {
            Base::root_node_ =
                this->divideTree(*this, 0, Base::size_, Base::root_bbox_);
        }
        else
        {
            std::atomic<unsigned int> thread_count(0u);
            std::mutex                mutex;
            Base::root_node_ = this->divideTreeConcurrent(
                *this, 0, Base::size_, Base::root_bbox_, thread_count, mutex);
        }
    }

    /** \name Query methods
     * @{ */

    /**
     * Find set of nearest neighbors to vec[0:dim-1]. Their indices are stored
     * inside the result object.
     * This is the core search function, all others are wrappers around this
     * one.
     *
     * \param result The result object in which the indices of the
     *               nearest-neighbors are stored.
     * \param vec    The vector of the query point for which to search the
     *               nearest neighbors.
     * \param searchParams Optional parameters for the search.
     *
     * \tparam RESULTSET Should be any ResultSet<DistanceType>
     * \return True if the requested neighbors could be found.
     *
     * \sa knnSearch(), radiusSearch(), radiusSearchCustomCallback()
     *
     * \note If L2 norms are used, all returned distances are actually squared
     *       distances.
     */
    template <typename RESULTSET>
    bool findNeighbors(
        RESULTSET& result, const ElementType* vec,
        const SearchParameters& searchParams = {}) const
    {
        assert(vec);
        if (this->size(*this) == 0) return false;
        if (!Base::root_node_) return false;
        float epsError = 1 + searchParams.eps;

        // fixed or variable-sized container (depending on DIM)
        distance_vector_t dists;
        // Fill it with zeros.
        assign(
            dists, (DIM > 0 ? DIM : Base::dim_),
            static_cast<typename distance_vector_t::value_type>(0));
        DistanceType dist = this->computeInitialDistances(*this, vec, dists);
        searchLevel(result, vec, Base::root_node_, dist, dists, epsError);
        return result.full();
    }

    /**
     * Find the "num_closest" nearest neighbors to the \a query_point[0:dim-1].
     * Their indices are stored inside the result object. \sa radiusSearch,
     * findNeighbors
     * \return Number `N` of valid points in
     * the result set.
     *
     * \note If L2 norms are used, all returned distances are actually squared
     *       distances.
     *
     * \note Only the first `N` entries in `out_indices` and `out_distances`
     *       will be valid. Return may be less than `num_closest` only if the
     *       number of elements in the tree is less than `num_closest`.
     */
    Size knnSearch(
        const ElementType* query_point, const Size num_closest,
        IndexType* out_indices, DistanceType* out_distances,
        const SearchParameters& searchParams = {}) const
    {
        nanoflann::KNNResultSet<DistanceType, IndexType> resultSet(num_closest);
        resultSet.init(out_indices, out_distances);
        findNeighbors(resultSet, query_point, searchParams);
        return resultSet.size();
    }

    /**
     * Find all the neighbors to \a query_point[0:dim-1] within a maximum
     * radius. The output is given as a vector of pairs, of which the first
     * element is a point index and the second the corresponding distance.
     * Previous contents of \a IndicesDists are cleared.
     *
     * If searchParams.sorted==true, the output list is sorted by ascending
     * distances.
     *
     * For a better performance, it is advisable to do a .reserve() on the
     * vector if you have any wild guess about the number of expected matches.
     *
     *  \sa knnSearch, findNeighbors, radiusSearchCustomCallback
     * \return The number of points within the given radius (i.e. indices.size()
     * or dists.size() )
     *
     * \note If L2 norms are used, search radius and all returned distances
     *       are actually squared distances.
     */
    Size radiusSearch(
        const ElementType* query_point, const DistanceType& radius,
        std::vector<ResultItem<IndexType, DistanceType>>& IndicesDists,
        const SearchParameters& searchParams = {}) const
    {
        RadiusResultSet<DistanceType, IndexType> resultSet(
            radius, IndicesDists);
        const size_t nFound =
            radiusSearchCustomCallback(query_point, resultSet, searchParams);
        if (searchParams.sorted)
            std::sort(
                IndicesDists.begin(), IndicesDists.end(), IndexDist_Sorter());
        return nFound;
    }

    /**
     * Just like radiusSearch() but with a custom callback class for each point
     * found in the radius of the query. See the source of RadiusResultSet<> as
     * a start point for your own classes. \sa radiusSearch
     */
    template <class SEARCH_CALLBACK>
    Size radiusSearchCustomCallback(
        const ElementType* query_point, SEARCH_CALLBACK& resultSet,
        const SearchParameters& searchParams = {}) const
    {
        findNeighbors(resultSet, query_point, searchParams);
        return resultSet.size();
    }

    /** @} */

   public:
    void computeBoundingBox(BoundingBox& bbox)
    {
        const auto dims = (DIM > 0 ? DIM : Base::dim_);
        resize(bbox, dims);

        if (dataset_.kdtree_get_bbox(bbox))
        {
            // Done! It was implemented in derived class
        }
        else
        {
            const Size N = Base::size_;
            if (!N) {
#if !NANOFLANN_NO_EXCEPTIONS
                throw std::runtime_error(
                    "[nanoflann] computeBoundingBox() called but "
                    "no data points found.");
#endif
            }
            for (Dimension i = 0; i < dims; ++i)
            {
                bbox[i].low = bbox[i].high =
                    this->dataset_get(*this, Base::vAcc_[0], i);
            }
            for (Offset k = 1; k < N; ++k)
            {
                for (Dimension i = 0; i < dims; ++i)
                {
                    const auto val =
                        this->dataset_get(*this, Base::vAcc_[k], i);
                    if (val < bbox[i].low) bbox[i].low = val;
                    if (val > bbox[i].high) bbox[i].high = val;
                }
            }
        }
    }

    /**
     * Performs an exact search in the tree starting from a node.
     * \tparam RESULTSET Should be any ResultSet<DistanceType>
     */
    template <class RESULTSET>
    void searchLevel(
        RESULTSET& result_set, const ElementType* vec, const NodePtr node,
        DistanceType mindist, distance_vector_t& dists,
        const float epsError) const
    {
        /* If this is a leaf node, then do check and return. */
        if ((node->child1 == nullptr) && (node->child2 == nullptr))
        {
            DistanceType worst_dist = result_set.worstDist();
            for (Offset i = node->node_type.lr.left;
                 i < node->node_type.lr.right; ++i)
            {
                const IndexType index = Base::vAcc_[i];  // reorder... : i;
                if (treeIndex_[index] == -1) continue;
                DistanceType dist = distance_.evalMetric(
                    vec, index, (DIM > 0 ? DIM : Base::dim_));
                if (dist < worst_dist)
                {
                    if (!result_set.addPoint(
                            static_cast<typename RESULTSET::DistanceType>(dist),
                            static_cast<typename RESULTSET::IndexType>(
                                Base::vAcc_[i])))
                    {
                        // the resultset doesn't want to receive any more
                        // points, we're done searching!
                        return;  // false;
                    }
                }
            }
            return;
        }

        /* Which child branch should be taken first? */
        Dimension    idx   = node->node_type.sub.divfeat;
        ElementType  val   = vec[idx];
        DistanceType diff1 = val - node->node_type.sub.divlow;
        DistanceType diff2 = val - node->node_type.sub.divhigh;

        NodePtr      bestChild;
        NodePtr      otherChild;
        DistanceType cut_dist;
        if ((diff1 + diff2) < 0)
        {
            bestChild  = node->child1;
            otherChild = node->child2;
            cut_dist =
                distance_.accum_dist(val, node->node_type.sub.divhigh, idx);
        }
        else
        {
            bestChild  = node->child2;
            otherChild = node->child1;
            cut_dist =
                distance_.accum_dist(val, node->node_type.sub.divlow, idx);
        }

        /* Call recursively to search next level down. */
        searchLevel(result_set, vec, bestChild, mindist, dists, epsError);

        DistanceType dst = dists[idx];
        mindist          = mindist + cut_dist - dst;
        dists[idx]       = cut_dist;
        if (mindist * epsError <= result_set.worstDist())
        {
            searchLevel(result_set, vec, otherChild, mindist, dists, epsError);
        }
        dists[idx] = dst;
    }

   public:
    /**  Stores the index in a binary file.
     *   IMPORTANT NOTE: The set of data points is NOT stored in the file, so
     * when loading the index object it must be constructed associated to the
     * same source of data points used while building it. See the example:
     * examples/saveload_example.cpp \sa loadIndex  */
    void saveIndex(std::ostream& stream) { saveIndex(*this, stream); }

    /**  Loads a previous index from a binary file.
     *   IMPORTANT NOTE: The set of data points is NOT stored in the file, so
     * the index object must be constructed associated to the same source of
     * data points used while building the index. See the example:
     * examples/saveload_example.cpp \sa loadIndex  */
    void loadIndex(std::istream& stream) { loadIndex(*this, stream); }
};

/** kd-tree dynaimic index
 *
 * class to create multiple static index and merge their results to behave as
 * single dynamic index as proposed in Logarithmic Approach.
 *
 *  Example of usage:
 *  examples/dynamic_pointcloud_example.cpp
 *
 * \tparam DatasetAdaptor The user-provided adaptor (see comments above).
 * \tparam Distance The distance metric to use: nanoflann::metric_L1,
 * nanoflann::metric_L2, nanoflann::metric_L2_Simple, etc. \tparam DIM
 * Dimensionality of data points (e.g. 3 for 3D points) \tparam IndexType
 * Will be typically size_t or int
 */
template <
    typename Distance, class DatasetAdaptor, int32_t DIM = -1,
    typename IndexType = uint32_t>
class KDTreeSingleIndexDynamicAdaptor
{
   public:
    using ElementType  = typename Distance::ElementType;
    using DistanceType = typename Distance::DistanceType;

    using Offset = typename KDTreeSingleIndexDynamicAdaptor_<
        Distance, DatasetAdaptor, DIM>::Offset;
    using Size = typename KDTreeSingleIndexDynamicAdaptor_<
        Distance, DatasetAdaptor, DIM>::Size;
    using Dimension = typename KDTreeSingleIndexDynamicAdaptor_<
        Distance, DatasetAdaptor, DIM>::Dimension;

   protected:
    Size leaf_max_size_;
    Size treeCount_;
    Size pointCount_;

    /**
     * The dataset used by this index
     */
    const DatasetAdaptor& dataset_;  //!< The source of our data

    /** treeIndex[idx] is the index of tree in which point at idx is stored.
     * treeIndex[idx]=-1 means that point has been removed. */
    std::vector<int>        treeIndex_;
    std::unordered_set<int> removedPoints_;

    KDTreeSingleIndexAdaptorParams index_params_;

    Dimension dim_;  //!< Dimensionality of each data point

    using index_container_t = KDTreeSingleIndexDynamicAdaptor_<
        Distance, DatasetAdaptor, DIM, IndexType>;
    std::vector<index_container_t> index_;

   public:
    /** Get a const ref to the internal list of indices; the number of indices
     * is adapted dynamically as the dataset grows in size. */
    const std::vector<index_container_t>& getAllIndices() const
    {
        return index_;
    }

   private:
    /** finds position of least significant unset bit */
    int First0Bit(IndexType num)
    {
        int pos = 0;
        while (num & 1)
        {
            num = num >> 1;
            pos++;
        }
        return pos;
    }

    /** Creates multiple empty trees to handle dynamic support */
    void init()
    {
        using my_kd_tree_t = KDTreeSingleIndexDynamicAdaptor_<
            Distance, DatasetAdaptor, DIM, IndexType>;
        std::vector<my_kd_tree_t> index(
            treeCount_,
            my_kd_tree_t(dim_ /*dim*/, dataset_, treeIndex_, index_params_));
        index_ = index;
    }

   public:
    Distance distance_;

    /**
     * KDTree constructor
     *
     * Refer to docs in README.md or online in
     * https://github.com/jlblancoc/nanoflann
     *
     * The KD-Tree point dimension (the length of each point in the datase, e.g.
     * 3 for 3D points) is determined by means of:
     *  - The \a DIM template parameter if >0 (highest priority)
     *  - Otherwise, the \a dimensionality parameter of this constructor.
     *
     * @param inputData Dataset with the input features. Its lifetime must be
     *  equal or longer than that of the instance of this class.
     * @param params Basically, the maximum leaf node size
     */
    explicit KDTreeSingleIndexDynamicAdaptor(
        const int dimensionality, const DatasetAdaptor& inputData,
        const KDTreeSingleIndexAdaptorParams& params =
            KDTreeSingleIndexAdaptorParams(),
        const size_t maximumPointCount = 1000000000U)
        : dataset_(inputData), index_params_(params), distance_(inputData)
    {
        treeCount_  = static_cast<size_t>(std::log2(maximumPointCount)) + 1;
        pointCount_ = 0U;
        dim_        = dimensionality;
        treeIndex_.clear();
        if (DIM > 0) dim_ = DIM;
        leaf_max_size_ = params.leaf_max_size;
        init();
        const size_t num_initial_points = dataset_.kdtree_get_point_count();
        if (num_initial_points > 0) { addPoints(0, num_initial_points - 1); }
    }

    /** Deleted copy constructor*/
    explicit KDTreeSingleIndexDynamicAdaptor(
        const KDTreeSingleIndexDynamicAdaptor<
            Distance, DatasetAdaptor, DIM, IndexType>&) = delete;

    /** Add points to the set, Inserts all points from [start, end] */
    void addPoints(IndexType start, IndexType end)
    {
        const Size count    = end - start + 1;
        int        maxIndex = 0;
        treeIndex_.resize(treeIndex_.size() + count);
        for (IndexType idx = start; idx <= end; idx++)
        {
            const int pos           = First0Bit(pointCount_);
            maxIndex                = std::max(pos, maxIndex);
            treeIndex_[pointCount_] = pos;

            const auto it = removedPoints_.find(idx);
            if (it != removedPoints_.end())
            {
                removedPoints_.erase(it);
                treeIndex_[idx] = pos;
            }

            for (int i = 0; i < pos; i++)
            {
                for (int j = 0; j < static_cast<int>(index_[i].vAcc_.size());
                     j++)
                {
                    index_[pos].vAcc_.push_back(index_[i].vAcc_[j]);
                    if (treeIndex_[index_[i].vAcc_[j]] != -1)
                        treeIndex_[index_[i].vAcc_[j]] = pos;
                }
                index_[i].vAcc_.clear();
            }
            index_[pos].vAcc_.push_back(idx);
            pointCount_++;
        }

        for (int i = 0; i <= maxIndex; ++i)
        {
            index_[i].freeIndex(index_[i]);
            if (!index_[i].vAcc_.empty()) index_[i].buildIndex();
        }
    }

    /** Remove a point from the set (Lazy Deletion) */
    void removePoint(size_t idx)
    {
        if (idx >= pointCount_) return;
        removedPoints_.insert(idx);
        treeIndex_[idx] = -1;
    }

    /**
     * Find set of nearest neighbors to vec[0:dim-1]. Their indices are stored
     * inside the result object.
     *
     * Params:
     *     result = the result object in which the indices of the
     * nearest-neighbors are stored vec = the vector for which to search the
     * nearest neighbors
     *
     * \tparam RESULTSET Should be any ResultSet<DistanceType>
     * \return  True if the requested neighbors could be found.
     * \sa knnSearch, radiusSearch
     *
     * \note If L2 norms are used, all returned distances are actually squared
     *       distances.
     */
    template <typename RESULTSET>
    bool findNeighbors(
        RESULTSET& result, const ElementType* vec,
        const SearchParameters& searchParams = {}) const
    {
        for (size_t i = 0; i < treeCount_; i++)
        {
            index_[i].findNeighbors(result, &vec[0], searchParams);
        }
        return result.full();
    }
};

/** An L2-metric KD-tree adaptor for working with data directly stored in an
 * Eigen Matrix, without duplicating the data storage. You can select whether a
 * row or column in the matrix represents a point in the state space.
 *
 * Example of usage:
 * \code
 * Eigen::Matrix<num_t,Eigen::Dynamic,Eigen::Dynamic>  mat;
 *
 * // Fill out "mat"...
 * using my_kd_tree_t = nanoflann::KDTreeEigenMatrixAdaptor<
 *   Eigen::Matrix<num_t,Dynamic,Dynamic>>;
 *
 * const int max_leaf = 10;
 * my_kd_tree_t mat_index(mat, max_leaf);
 * mat_index.index->...
 * \endcode
 *
 *  \tparam DIM If set to >0, it specifies a compile-time fixed dimensionality
 * for the points in the data set, allowing more compiler optimizations.
 * \tparam Distance The distance metric to use: nanoflann::metric_L1,
 * nanoflann::metric_L2, nanoflann::metric_L2_Simple, etc.
 * \tparam row_major If set to true the rows of the matrix are used as the
 *         points, if set to false  the columns of the matrix are used as the
 *         points.
 */
template <
    class MatrixType, int32_t DIM = -1, class Distance = nanoflann::metric_L2,
    bool row_major = true>
struct KDTreeEigenMatrixAdaptor
{
    using self_t =
        KDTreeEigenMatrixAdaptor<MatrixType, DIM, Distance, row_major>;
    using num_t     = typename MatrixType::Scalar;
    using IndexType = typename MatrixType::Index;
    using metric_t  = typename Distance::template traits<
        num_t, self_t, IndexType>::distance_t;

    using index_t = KDTreeSingleIndexAdaptor<
        metric_t, self_t,
        row_major ? MatrixType::ColsAtCompileTime
                  : MatrixType::RowsAtCompileTime,
        IndexType>;

    index_t* index_;  //! The kd-tree index for the user to call its methods as
                      //! usual with any other FLANN index.

    using Offset    = typename index_t::Offset;
    using Size      = typename index_t::Size;
    using Dimension = typename index_t::Dimension;

    /// Constructor: takes a const ref to the matrix object with the data points
    explicit KDTreeEigenMatrixAdaptor(
        const Dimension                                 dimensionality,
        const std::reference_wrapper<const MatrixType>& mat,
        const int                                       leaf_max_size = 10)
        : m_data_matrix(mat)
    {
        const auto dims = row_major ? mat.get().cols() : mat.get().rows();
        if (static_cast<Dimension>(dims) != dimensionality) {
#if !NANOFLANN_NO_EXCEPTIONS
            throw std::runtime_error(
                "Error: 'dimensionality' must match column count in data "
                "matrix");
#endif
        }

        if (DIM > 0 && static_cast<int32_t>(dims) != DIM) {
#if !NANOFLANN_NO_EXCEPTIONS
            throw std::runtime_error(
                "Data set dimensionality does not match the 'DIM' template "
                "argument");
#endif
        }
        index_ = new index_t(
            dims, *this /* adaptor */,
            nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size));
    }

   public:
    /** Deleted copy constructor */
    KDTreeEigenMatrixAdaptor(const self_t&) = delete;

    ~KDTreeEigenMatrixAdaptor() { delete index_; }

    const std::reference_wrapper<const MatrixType> m_data_matrix;

    /** Query for the \a num_closest closest points to a given point (entered as
     * query_point[0:dim-1]). Note that this is a short-cut method for
     * index->findNeighbors(). The user can also call index->... methods as
     * desired.
     *
     * \note If L2 norms are used, all returned distances are actually squared
     *       distances.
     */
    void query(
        const num_t* query_point, const Size num_closest,
        IndexType* out_indices, num_t* out_distances) const
    {
        nanoflann::KNNResultSet<num_t, IndexType> resultSet(num_closest);
        resultSet.init(out_indices, out_distances);
        index_->findNeighbors(resultSet, query_point);
    }

    /** @name Interface expected by KDTreeSingleIndexAdaptor
     * @{ */

    const self_t& derived() const { return *this; }
    self_t&       derived() { return *this; }

    // Must return the number of data points
    Size kdtree_get_point_count() const
    {
        if (row_major)
            return m_data_matrix.get().rows();
        else
            return m_data_matrix.get().cols();
    }

    // Returns the dim'th component of the idx'th point in the class:
    num_t kdtree_get_pt(const IndexType idx, size_t dim) const
    {
        if (row_major)
            return m_data_matrix.get().coeff(idx, IndexType(dim));
        else
            return m_data_matrix.get().coeff(IndexType(dim), idx);
    }

    // Optional bounding-box computation: return false to default to a standard
    // bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned
    //   in "bb" so it can be avoided to redo it again. Look at bb.size() to
    //   find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const
    {
        return false;
    }

    /** @} */

};  // end of KDTreeEigenMatrixAdaptor
/** @} */

/** @} */  // end of grouping
}  // namespace nanoflann
