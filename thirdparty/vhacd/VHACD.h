/* Copyright (c) 2011 Khaled Mamou (kmamou at gmail dot com)
 All rights reserved.
 
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 3. The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once
#ifndef VHACD_H
#    define VHACD_H

// Please view this slide deck which describes usage and how the algorithm works.
// https://docs.google.com/presentation/d/1OZ4mtZYrGEC8qffqb8F7Le2xzufiqvaPpRbLHKKgTIM/edit?usp=sharing

// VHACD is now a header only library.
// In just *one* of your CPP files *before* you include 'VHACD.h' you must declare
// #define ENABLE_VHACD_IMPLEMENTATION 1
// This will compile the implementation code into your project. If you don't
// have this define, you will get link errors since the implementation code will
// not be present. If you define it more than once in your code base, you will get
// link errors due to a duplicate implementation. This is the same pattern used by
// ImGui and StbLib and other popular open source libraries.

#    define VHACD_VERSION_MAJOR 4
#    define VHACD_VERSION_MINOR 1

// Changes for version 4.1
//
// Various minor tweaks mostly to the test application and some default values.

// Changes for version 4.0
//
// * The code has been significantly refactored to be cleaner and easier to maintain
//      * All OpenCL related code removed
//      * All Bullet code removed
//      * All SIMD code removed
//      * Old plane splitting code removed
// 
// * The code is now delivered as a single header file 'VHACD.h' which has both the API
// * declaration as well as the implementation.  Simply add '#define ENABLE_VHACD_IMPLEMENTATION 1'
// * to any CPP in your application prior to including 'VHACD.h'. Only do this in one CPP though.
// * If you do not have this define once, you will get link errors since the implementation code
// * will not be compiled in. If you have this define more than once, you are likely to get
// * duplicate symbol link errors.
//
// * Since the library is now delivered as a single header file, we do not provide binaries
// * or build scripts as these are not needed.
//
// * The old DebugView and test code has all been removed and replaced with a much smaller and
// * simpler test console application with some test meshes to work with.
//
// * The convex hull generation code has changed. The previous version came from Bullet. 
// * However, the new version is courtesy of Julio Jerez, the author of the Newton
// * physics engine. His new version is faster and more numerically stable.
//
// * The code can now detect if the input mesh is, itself, already a convex object and
// * can early out.
//
// * Significant performance improvements have been made to the code and it is now much
// * faster, stable, and is easier to tune than previous versions.
//
// * A bug was fixed with the shrink wrapping code (project hull vertices) that could
// * sometime produce artifacts in the results. The new version uses a 'closest point'
// * algorithm that is more reliable.
//
// * You can now select which 'fill mode' to use. For perfectly closed meshes, the default
// * behavior using a flood fill generally works fine. However, some meshes have small 
// * holes in them and therefore the flood fill will fail, treating the mesh as being
// * hollow. In these cases, you can use the 'raycast' fill option to determine which 
// * parts of the voxelized mesh are 'inside' versus being 'outside'. Finally, there
// * are some rare instances where a user might actually want the mesh to be treated as
// * hollow, in which case you can pass in 'surface' only.
// *
// * A new optional virtual interface called 'IUserProfiler' was provided.
// * This allows the user to provide an optional profiling callback interface to assist in
// * diagnosing performance issues. This change was made by Danny Couture at Epic for the UE4 integration.
// * Some profiling macros were also declared in support of this feature.
// *
// * Another new optional virtual interface called 'IUserTaskRunner' was provided.
// * This interface is used to run logical 'tasks' in a background thread. If none is provided
// * then a default implementation using std::thread will be executed.
// * This change was made by Danny Couture at Epic to speed up the voxelization step.
// *



// The history of V-HACD:
//
// The initial version was written by John W. Ratcliff and was called 'ACD'
// This version did not perform CSG operations on the source mesh, so if you 
// recursed too deeply it would produce hollow results.
//
// The next version was written by Khaled Mamou and was called 'HACD'
// In this version Khaled tried to perform a CSG operation on the source 
// mesh to produce more robust results. However, Khaled learned that the
// CSG library he was using had licensing issues so he started work on the
// next version.
//
// The next version was called 'V-HACD' because Khaled made the observation
// that plane splitting would be far easier to implement working in voxel space.
// 
// V-HACD has been integrated into UE4, Blender, and a number of other projects.
// This new release, version4, is a significant refactor of the code to fix
// some bugs, improve performance, and to make the codebase easier to maintain
// going forward.

#include <stdint.h>
#include <functional>

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>

namespace VHACD {

struct Vertex
{
    double mX;
    double mY;
    double mZ;

    Vertex() = default;
    Vertex(double x, double y, double z) : mX(x), mY(y), mZ(z) {}

    const double& operator[](size_t idx) const
    {
        switch(idx)
        {
            case 0: return mX;
            case 1: return mY;
            case 2: return mZ;
        };
        return mX;
    }
};

struct Triangle
{
    uint32_t mI0;
    uint32_t mI1;
    uint32_t mI2;

    Triangle() = default;
    Triangle(uint32_t i0, uint32_t i1, uint32_t i2) : mI0(i0), mI1(i1), mI2(i2) {}
};

template <typename T>
class Vector3
{
public:
    /*
    * Getters
    */
    T& operator[](size_t i);
    const T& operator[](size_t i) const;
    T& GetX();
    T& GetY();
    T& GetZ();
    const T& GetX() const;
    const T& GetY() const;
    const T& GetZ() const;

    /*
    * Normalize and norming
    */
    T Normalize();
    Vector3 Normalized();
    T GetNorm() const;
    T GetNormSquared() const;
    int LongestAxis() const;

    /*
    * Vector-vector operations
    */
    Vector3& operator=(const Vector3& rhs);
    Vector3& operator+=(const Vector3& rhs);
    Vector3& operator-=(const Vector3& rhs);

    Vector3 CWiseMul(const Vector3& rhs) const;
    Vector3 Cross(const Vector3& rhs) const;
    T Dot(const Vector3& rhs) const;
    Vector3 operator+(const Vector3& rhs) const;
    Vector3 operator-(const Vector3& rhs) const;

    /*
    * Vector-scalar operations
    */
    Vector3& operator-=(T a);
    Vector3& operator+=(T a);
    Vector3& operator/=(T a);
    Vector3& operator*=(T a);

    Vector3 operator*(T rhs) const;
    Vector3 operator/(T rhs) const;

    /*
    * Unary operations
    */
    Vector3 operator-() const;

    /*
    * Comparison operators
    */
    bool operator<(const Vector3& rhs) const;
    bool operator>(const Vector3& rhs) const;

    /*
     * Returns true if all elements of *this are greater than or equal to all elements of rhs, coefficient wise
     * LE is less than or equal
     */
    bool CWiseAllGE(const Vector3<T>& rhs) const;
    bool CWiseAllLE(const Vector3<T>& rhs) const;

    Vector3 CWiseMin(const Vector3& rhs) const;
    Vector3 CWiseMax(const Vector3& rhs) const;
    T MinCoeff() const;
    T MaxCoeff() const;

    T MinCoeff(uint32_t& idx) const;
    T MaxCoeff(uint32_t& idx) const;

    /*
    * Constructors
    */
    Vector3() = default;
    Vector3(T a);
    Vector3(T x, T y, T z);
    Vector3(const Vector3& rhs);
    ~Vector3() = default;

    template <typename U>
    Vector3(const Vector3<U>& rhs);

    Vector3(const VHACD::Vertex&);
    Vector3(const VHACD::Triangle&);

    operator VHACD::Vertex() const;

private:
    std::array<T, 3> m_data{ T(0.0) };
};

typedef VHACD::Vector3<double> Vect3;

struct BoundsAABB
{
    BoundsAABB() = default;
    BoundsAABB(const std::vector<VHACD::Vertex>& points);
    BoundsAABB(const Vect3& min,
               const Vect3& max);

    BoundsAABB Union(const BoundsAABB& b);

    bool Intersects(const BoundsAABB& b) const;

    double SurfaceArea() const;
    double Volume() const;

    BoundsAABB Inflate(double ratio) const;

    VHACD::Vect3 ClosestPoint(const VHACD::Vect3& p) const;

    VHACD::Vect3& GetMin();
    VHACD::Vect3& GetMax();
    const VHACD::Vect3& GetMin() const;
    const VHACD::Vect3& GetMax() const;

    VHACD::Vect3 GetSize() const;
    VHACD::Vect3 GetCenter() const;

    VHACD::Vect3 m_min{ double(0.0) };
    VHACD::Vect3 m_max{ double(0.0) };
};

/**
* This enumeration determines how the voxels as filled to create a solid
* object. The default should be 'FLOOD_FILL' which generally works fine 
* for closed meshes. However, if the mesh is not watertight, then using
* RAYCAST_FILL may be preferable as it will determine if a voxel is part 
* of the interior of the source mesh by raycasting around it.
* 
* Finally, there are some cases where you might actually want a convex 
* decomposition to treat the source mesh as being hollow. If that is the
* case you can pass in 'SURFACE_ONLY' and then the convex decomposition 
* will converge only onto the 'skin' of the surface mesh.
*/
enum class FillMode
{
    FLOOD_FILL, // This is the default behavior, after the voxelization step it uses a flood fill to determine 'inside'
                // from 'outside'. However, meshes with holes can fail and create hollow results.
    SURFACE_ONLY, // Only consider the 'surface', will create 'skins' with hollow centers.
    RAYCAST_FILL, // Uses raycasting to determine inside from outside.
};

class IVHACD
{
public:
    /**
    * This optional pure virtual interface is used to notify the caller of the progress
    * of convex decomposition as well as a signal when it is complete when running in
    * a background thread
    */
    class IUserCallback
    {
    public:
        virtual ~IUserCallback(){};

        /**
        * Notifies the application of the current state of the convex decomposition operation
        * 
        * @param overallProgress : Total progress from 0-100%
        * @param stageProgress : Progress of the current stage 0-100%
        * @param stage : A text description of the current stage we are in
        * @param operation : A text description of what operation is currently being performed.
        */
        virtual void Update(const double overallProgress,
                            const double stageProgress,
                            const char* const stage,
                            const char* operation) = 0;

        // This is an optional user callback which is only called when running V-HACD asynchronously.
        // This is a callback performed to notify the user that the
        // convex decomposition background process is completed. This call back will occur from
        // a different thread so the user should take that into account.
        virtual void NotifyVHACDComplete()
        {
        }
    };

    /**
    * Optional user provided pure virtual interface to be notified of warning or informational messages
    */
    class IUserLogger
    {
    public:
        virtual ~IUserLogger(){};
        virtual void Log(const char* const msg) = 0;
    };

    /**
    * An optional user provided pure virtual interface to perform a background task.
    * This was added by Danny Couture at Epic as they wanted to use their own
    * threading system instead of the standard library version which is the default.
    */
    class IUserTaskRunner
    {
    public:
        virtual ~IUserTaskRunner(){};
        virtual void* StartTask(std::function<void()> func) = 0;
        virtual void JoinTask(void* Task) = 0;
    };

    /**
    * A simple class that represents a convex hull as a triangle mesh with 
    * double precision vertices. Polygons are not currently provided.
    */
    class ConvexHull
    {
    public:
        std::vector<VHACD::Vertex>      m_points;
        std::vector<VHACD::Triangle>    m_triangles;

        double                          m_volume{ 0 };          // The volume of the convex hull
        VHACD::Vect3                    m_center{ 0, 0, 0 };    // The centroid of the convex hull
        uint32_t                        m_meshId{ 0 };          // A unique id for this convex hull
        VHACD::Vect3            mBmin;                  // Bounding box minimum of the AABB
        VHACD::Vect3            mBmax;                  // Bounding box maximum of the AABB
    };

    /**
    * This class provides the parameters controlling the convex decomposition operation
    */
    class Parameters
    {
    public:
        IUserCallback*      m_callback{nullptr};            // Optional user provided callback interface for progress
        IUserLogger*        m_logger{nullptr};              // Optional user provided callback interface for log messages
        IUserTaskRunner*    m_taskRunner{nullptr};          // Optional user provided interface for creating tasks
        uint32_t            m_maxConvexHulls{ 64 };         // The maximum number of convex hulls to produce
        uint32_t            m_resolution{ 400000 };         // The voxel resolution to use
        double              m_minimumVolumePercentErrorAllowed{ 1 }; // if the voxels are within 1% of the volume of the hull, we consider this a close enough approximation
        uint32_t            m_maxRecursionDepth{ 10 };        // The maximum recursion depth
        bool                m_shrinkWrap{true};             // Whether or not to shrinkwrap the voxel positions to the source mesh on output
        FillMode            m_fillMode{ FillMode::FLOOD_FILL }; // How to fill the interior of the voxelized mesh
        uint32_t            m_maxNumVerticesPerCH{ 64 };    // The maximum number of vertices allowed in any output convex hull
        bool                m_asyncACD{ true };             // Whether or not to run asynchronously, taking advantage of additional cores
        uint32_t            m_minEdgeLength{ 2 };           // Once a voxel patch has an edge length of less than 4 on all 3 sides, we don't keep recursing
        bool                m_findBestPlane{ false };       // Whether or not to attempt to split planes along the best location. Experimental feature. False by default.
    };

    /**
    * Will cause the convex decomposition operation to be canceled early. No results will be produced but the background operation will end as soon as it can.
    */
    virtual void Cancel() = 0;

    /**
    * Compute a convex decomposition of a triangle mesh using float vertices and the provided user parameters.
    * 
    * @param points : The vertices of the source mesh as floats in the form of X1,Y1,Z1,  X2,Y2,Z2,.. etc.
    * @param countPoints : The number of vertices in the source mesh.
    * @param triangles : The indices of triangles in the source mesh in the form of I1,I2,I3, .... 
    * @param countTriangles : The number of triangles in the source mesh
    * @param params : The convex decomposition parameters to apply
    * @return : Returns true if the convex decomposition operation can be started
    */
    virtual bool Compute(const float* const points,
                         const uint32_t countPoints,
                         const uint32_t* const triangles,
                         const uint32_t countTriangles,
                         const Parameters& params) = 0;

    /**
    * Compute a convex decomposition of a triangle mesh using double vertices and the provided user parameters.
    * 
    * @param points : The vertices of the source mesh as floats in the form of X1,Y1,Z1,  X2,Y2,Z2,.. etc.
    * @param countPoints : The number of vertices in the source mesh.
    * @param triangles : The indices of triangles in the source mesh in the form of I1,I2,I3, .... 
    * @param countTriangles : The number of triangles in the source mesh
    * @param params : The convex decomposition parameters to apply
    * @return : Returns true if the convex decomposition operation can be started
    */
    virtual bool Compute(const double* const points,
                         const uint32_t countPoints,
                         const uint32_t* const triangles,
                         const uint32_t countTriangles,
                         const Parameters& params) = 0;

    /**
    * Returns the number of convex hulls that were produced.
    * 
    * @return : Returns the number of convex hulls produced, or zero if it failed or was canceled
    */
    virtual uint32_t GetNConvexHulls() const = 0;

    /**
    * Retrieves one of the convex hulls in the solution set
    * 
    * @param index : Which convex hull to retrieve
    * @param ch : The convex hull descriptor to return
    * @return : Returns true if the convex hull exists and could be retrieved
    */
    virtual bool GetConvexHull(const uint32_t index,
                               ConvexHull& ch) const = 0;

    /**
    * Releases any memory allocated by the V-HACD class
    */
    virtual void Clean() = 0; // release internally allocated memory

    /**
    * Releases this instance of the V-HACD class
    */
    virtual void Release() = 0; // release IVHACD

    // Will compute the center of mass of the convex hull decomposition results and return it
    // in 'centerOfMass'.  Returns false if the center of mass could not be computed.
    virtual bool ComputeCenterOfMass(double centerOfMass[3]) const = 0;

    // In synchronous mode (non-multi-threaded) the state is always 'ready'
    // In asynchronous mode, this returns true if the background thread is not still actively computing
    // a new solution.  In an asynchronous config the 'IsReady' call will report any update or log
    // messages in the caller's current thread.
    virtual bool IsReady() const
    {
        return true;
    }

    /**
    * At the request of LegionFu : out_look@foxmail.com
    * This method will return which convex hull is closest to the source position.
    * You can use this method to figure out, for example, which vertices in the original
    * source mesh are best associated with which convex hull.
    * 
    * @param pos : The input 3d position to test against
    * 
    * @return : Returns which convex hull this position is closest to.
    */
    virtual uint32_t findNearestConvexHull(const double pos[3],
                                           double& distanceToHull) = 0;

protected:
    virtual ~IVHACD()
    {
    }
};
/*
 * Out of line definitions
 */

    template <typename T>
    T clamp(const T& v, const T& lo, const T& hi)
    {
        if (v < lo)
        {
            return lo;
        }
        if (v > hi)
        {
            return hi;
        }
        return v ;
    }

/*
 * Getters
 */
    template <typename T>
    inline T& Vector3<T>::operator[](size_t i)
    {
        return m_data[i];
    }

    template <typename T>
    inline const T& Vector3<T>::operator[](size_t i) const
    {
        return m_data[i];
    }

    template <typename T>
    inline T& Vector3<T>::GetX()
    {
        return m_data[0];
    }

    template <typename T>
    inline T& Vector3<T>::GetY()
    {
        return m_data[1];
    }

    template <typename T>
    inline T& Vector3<T>::GetZ()
    {
        return m_data[2];
    }

    template <typename T>
    inline const T& Vector3<T>::GetX() const
    {
        return m_data[0];
    }

    template <typename T>
    inline const T& Vector3<T>::GetY() const
    {
        return m_data[1];
    }

    template <typename T>
    inline const T& Vector3<T>::GetZ() const
    {
        return m_data[2];
    }

/*
 * Normalize and norming
 */
    template <typename T>
    inline T Vector3<T>::Normalize()
    {
        T n = GetNorm();
        if (n != T(0.0)) (*this) /= n;
        return n;
    }

    template <typename T>
    inline Vector3<T> Vector3<T>::Normalized()
    {
        Vector3<T> ret = *this;
        T n = GetNorm();
        if (n != T(0.0)) ret /= n;
        return ret;
    }

    template <typename T>
    inline T Vector3<T>::GetNorm() const
    {
        return std::sqrt(GetNormSquared());
    }

    template <typename T>
    inline T Vector3<T>::GetNormSquared() const
    {
        return this->Dot(*this);
    }

    template <typename T>
    inline int Vector3<T>::LongestAxis() const
    {
        auto it = std::max_element(m_data.begin(), m_data.end());
        return int(std::distance(m_data.begin(), it));
    }

/*
 * Vector-vector operations
 */
    template <typename T>
    inline Vector3<T>& Vector3<T>::operator=(const Vector3<T>& rhs)
    {
        GetX() = rhs.GetX();
        GetY() = rhs.GetY();
        GetZ() = rhs.GetZ();
        return *this;
    }

    template <typename T>
    inline Vector3<T>& Vector3<T>::operator+=(const Vector3<T>& rhs)
    {
        GetX() += rhs.GetX();
        GetY() += rhs.GetY();
        GetZ() += rhs.GetZ();
        return *this;
    }

    template <typename T>
    inline Vector3<T>& Vector3<T>::operator-=(const Vector3<T>& rhs)
    {
        GetX() -= rhs.GetX();
        GetY() -= rhs.GetY();
        GetZ() -= rhs.GetZ();
        return *this;
    }

    template <typename T>
    inline Vector3<T> Vector3<T>::CWiseMul(const Vector3<T>& rhs) const
    {
        return Vector3<T>(GetX() * rhs.GetX(),
                          GetY() * rhs.GetY(),
                          GetZ() * rhs.GetZ());
    }

    template <typename T>
    inline Vector3<T> Vector3<T>::Cross(const Vector3<T>& rhs) const
    {
        return Vector3<T>(GetY() * rhs.GetZ() - GetZ() * rhs.GetY(),
                          GetZ() * rhs.GetX() - GetX() * rhs.GetZ(),
                          GetX() * rhs.GetY() - GetY() * rhs.GetX());
    }

    template <typename T>
    inline T Vector3<T>::Dot(const Vector3<T>& rhs) const
    {
        return   GetX() * rhs.GetX()
                 + GetY() * rhs.GetY()
                 + GetZ() * rhs.GetZ();
    }

    template <typename T>
    inline Vector3<T> Vector3<T>::operator+(const Vector3<T>& rhs) const
    {
        return Vector3<T>(GetX() + rhs.GetX(),
                          GetY() + rhs.GetY(),
                          GetZ() + rhs.GetZ());
    }

    template <typename T>
    inline Vector3<T> Vector3<T>::operator-(const Vector3<T>& rhs) const
    {
        return Vector3<T>(GetX() - rhs.GetX(),
                          GetY() - rhs.GetY(),
                          GetZ() - rhs.GetZ());
    }

    template <typename T>
    inline Vector3<T> operator*(T lhs, const Vector3<T>& rhs)
    {
        return Vector3<T>(lhs * rhs.GetX(),
                          lhs * rhs.GetY(),
                          lhs * rhs.GetZ());
    }

/*
 * Vector-scalar operations
 */
    template <typename T>
    inline Vector3<T>& Vector3<T>::operator-=(T a)
    {
        GetX() -= a;
        GetY() -= a;
        GetZ() -= a;
        return *this;
    }

    template <typename T>
    inline Vector3<T>& Vector3<T>::operator+=(T a)
    {
        GetX() += a;
        GetY() += a;
        GetZ() += a;
        return *this;
    }

    template <typename T>
    inline Vector3<T>& Vector3<T>::operator/=(T a)
    {
        GetX() /= a;
        GetY() /= a;
        GetZ() /= a;
        return *this;
    }

    template <typename T>
    inline Vector3<T>& Vector3<T>::operator*=(T a)
    {
        GetX() *= a;
        GetY() *= a;
        GetZ() *= a;
        return *this;
    }

    template <typename T>
    inline Vector3<T> Vector3<T>::operator*(T rhs) const
    {
        return Vector3<T>(GetX() * rhs,
                          GetY() * rhs,
                          GetZ() * rhs);
    }

    template <typename T>
    inline Vector3<T> Vector3<T>::operator/(T rhs) const
    {
        return Vector3<T>(GetX() / rhs,
                          GetY() / rhs,
                          GetZ() / rhs);
    }

/*
 * Unary operations
 */
    template <typename T>
    inline Vector3<T> Vector3<T>::operator-() const
    {
        return Vector3<T>(-GetX(),
                          -GetY(),
                          -GetZ());
    }

/*
 * Comparison operators
 */
    template <typename T>
    inline bool Vector3<T>::operator<(const Vector3<T>& rhs) const
    {
        if (GetX() == rhs.GetX())
        {
            if (GetY() == rhs.GetY())
            {
                return (GetZ() < rhs.GetZ());
            }
            return (GetY() < rhs.GetY());
        }
        return (GetX() < rhs.GetX());
    }

    template <typename T>
    inline bool Vector3<T>::operator>(const Vector3<T>& rhs) const
    {
        if (GetX() == rhs.GetX())
        {
            if (GetY() == rhs.GetY())
            {
                return (GetZ() > rhs.GetZ());
            }
            return (GetY() > rhs.GetY());
        }
        return (GetX() > rhs.GetZ());
    }

    template <typename T>
    inline bool Vector3<T>::CWiseAllGE(const Vector3<T>& rhs) const
    {
        return    GetX() >= rhs.GetX()
                  && GetY() >= rhs.GetY()
                  && GetZ() >= rhs.GetZ();
    }

    template <typename T>
    inline bool Vector3<T>::CWiseAllLE(const Vector3<T>& rhs) const
    {
        return    GetX() <= rhs.GetX()
                  && GetY() <= rhs.GetY()
                  && GetZ() <= rhs.GetZ();
    }

    template <typename T>
    inline Vector3<T> Vector3<T>::CWiseMin(const Vector3<T>& rhs) const
    {
        return Vector3<T>(std::min(GetX(), rhs.GetX()),
                          std::min(GetY(), rhs.GetY()),
                          std::min(GetZ(), rhs.GetZ()));
    }

    template <typename T>
    inline Vector3<T> Vector3<T>::CWiseMax(const Vector3<T>& rhs) const
    {
        return Vector3<T>(std::max(GetX(), rhs.GetX()),
                          std::max(GetY(), rhs.GetY()),
                          std::max(GetZ(), rhs.GetZ()));
    }

    template <typename T>
    inline T Vector3<T>::MinCoeff() const
    {
        return *std::min_element(m_data.begin(), m_data.end());
    }

    template <typename T>
    inline T Vector3<T>::MaxCoeff() const
    {
        return *std::max_element(m_data.begin(), m_data.end());
    }

    template <typename T>
    inline T Vector3<T>::MinCoeff(uint32_t& idx) const
    {
        auto it = std::min_element(m_data.begin(), m_data.end());
        idx = uint32_t(std::distance(m_data.begin(), it));
        return *it;
    }

    template <typename T>
    inline T Vector3<T>::MaxCoeff(uint32_t& idx) const
    {
        auto it = std::max_element(m_data.begin(), m_data.end());
        idx = uint32_t(std::distance(m_data.begin(), it));
        return *it;
    }

/*
 * Constructors
 */
    template <typename T>
    inline Vector3<T>::Vector3(T a)
            : m_data{a, a, a}
    {
    }

    template <typename T>
    inline Vector3<T>::Vector3(T x, T y, T z)
            : m_data{x, y, z}
    {
    }

    template <typename T>
    inline Vector3<T>::Vector3(const Vector3& rhs)
            : m_data{rhs.m_data}
    {
    }

    template <typename T>
    template <typename U>
    inline Vector3<T>::Vector3(const Vector3<U>& rhs)
            : m_data{T(rhs.GetX()), T(rhs.GetY()), T(rhs.GetZ())}
    {
    }

    template <typename T>
    inline Vector3<T>::Vector3(const VHACD::Vertex& rhs)
            : Vector3<T>(rhs.mX, rhs.mY, rhs.mZ)
    {
        static_assert(std::is_same<T, double>::value, "Vertex to Vector3 constructor only enabled for double");
    }

    template <typename T>
    inline Vector3<T>::Vector3(const VHACD::Triangle& rhs)
            : Vector3<T>(rhs.mI0, rhs.mI1, rhs.mI2)
    {
        static_assert(std::is_same<T, uint32_t>::value, "Triangle to Vector3 constructor only enabled for uint32_t");
    }

    template <typename T>
    inline Vector3<T>::operator VHACD::Vertex() const
    {
        static_assert(std::is_same<T, double>::value, "Vector3 to Vertex conversion only enable for double");
        return ::VHACD::Vertex( GetX(), GetY(), GetZ());
    }

IVHACD* CreateVHACD();      // Create a synchronous (blocking) implementation of V-HACD
IVHACD* CreateVHACD_ASYNC();    // Create an asynchronous (non-blocking) implementation of V-HACD

} // namespace VHACD

#if ENABLE_VHACD_IMPLEMENTATION
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <limits.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4100 4127 4189 4244 4456 4701 4702 4996)
#endif // _MSC_VER

#ifdef __GNUC__
#pragma GCC diagnostic push
// Minimum set of warnings used for cleanup
// #pragma GCC diagnostic warning "-Wall"
// #pragma GCC diagnostic warning "-Wextra"
// #pragma GCC diagnostic warning "-Wpedantic"
// #pragma GCC diagnostic warning "-Wold-style-cast"
// #pragma GCC diagnostic warning "-Wnon-virtual-dtor"
// #pragma GCC diagnostic warning "-Wshadow"
#endif // __GNUC__

// Scoped Timer
namespace VHACD {

class Timer
{
public:
    Timer()
        : m_startTime(std::chrono::high_resolution_clock::now())
    {
    }

    void Reset()
    {
        m_startTime = std::chrono::high_resolution_clock::now();
    }

    double GetElapsedSeconds()
    {
        auto s = PeekElapsedSeconds();
        Reset();
        return s;
    }

    double PeekElapsedSeconds()
    {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = now - m_startTime;
        return diff.count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_startTime;
};

class ScopedTime
{
public:
    ScopedTime(const char* action,
               VHACD::IVHACD::IUserLogger* logger)
        : m_action(action)
        , m_logger(logger)
    {
        m_timer.Reset();
    }

    ~ScopedTime()
    {
        double dtime = m_timer.GetElapsedSeconds();
        if( m_logger )
        {
            char scratch[512];
            snprintf(scratch,
                        sizeof(scratch),"%s took %0.5f seconds",
                        m_action,
                        dtime);
            m_logger->Log(scratch);
        }
    }

    const char* m_action{ nullptr };
    Timer       m_timer;
    VHACD::IVHACD::IUserLogger* m_logger{ nullptr };
};
BoundsAABB::BoundsAABB(const std::vector<VHACD::Vertex>& points)
        : m_min(points[0])
        , m_max(points[0])
{
    for (uint32_t i = 1; i < points.size(); ++i)
    {
        const VHACD::Vertex& p = points[i];
        m_min = m_min.CWiseMin(p);
        m_max = m_max.CWiseMax(p);
    }
}

BoundsAABB::BoundsAABB(const VHACD::Vect3& min,
                              const VHACD::Vect3& max)
        : m_min(min)
        , m_max(max)
{
}

BoundsAABB BoundsAABB::Union(const BoundsAABB& b)
{
    return BoundsAABB(GetMin().CWiseMin(b.GetMin()),
                      GetMax().CWiseMax(b.GetMax()));
}

bool VHACD::BoundsAABB::Intersects(const VHACD::BoundsAABB& b) const
{
    if (   (  GetMin().GetX() > b.GetMax().GetX())
           || (b.GetMin().GetX() >   GetMax().GetX()))
        return false;
    if (   (  GetMin().GetY() > b.GetMax().GetY())
           || (b.GetMin().GetY() >   GetMax().GetY()))
        return false;
    if (   (  GetMin().GetZ() > b.GetMax().GetZ())
           || (b.GetMin().GetZ() >   GetMax().GetZ()))
        return false;
    return true;
}

double BoundsAABB::SurfaceArea() const
{
    VHACD::Vect3 d = GetMax() - GetMin();
    return double(2.0) * (d.GetX() * d.GetY() + d.GetX() * d.GetZ() + d.GetY() * d.GetZ());
}

double VHACD::BoundsAABB::Volume() const
{
    VHACD::Vect3 d = GetMax() - GetMin();
    return d.GetX() * d.GetY() * d.GetZ();
}

BoundsAABB VHACD::BoundsAABB::Inflate(double ratio) const
{
    double inflate = (GetMin() - GetMax()).GetNorm() * double(0.5) * ratio;
    return BoundsAABB(GetMin() - inflate,
                      GetMax() + inflate);
}

VHACD::Vect3 VHACD::BoundsAABB::ClosestPoint(const VHACD::Vect3& p) const
{
    return p.CWiseMax(GetMin()).CWiseMin(GetMax());
}

VHACD::Vect3& VHACD::BoundsAABB::GetMin()
{
    return m_min;
}

VHACD::Vect3& VHACD::BoundsAABB::GetMax()
{
    return m_max;
}

inline const VHACD::Vect3& VHACD::BoundsAABB::GetMin() const
{
    return m_min;
}

const VHACD::Vect3& VHACD::BoundsAABB::GetMax() const
{
    return m_max;
}

VHACD::Vect3 VHACD::BoundsAABB::GetSize() const
{
    return GetMax() - GetMin();
}

VHACD::Vect3 VHACD::BoundsAABB::GetCenter() const
{
    return (GetMin() + GetMax()) * double(0.5);
}

/*
 * Relies on three way comparison, which std::sort doesn't use
 */
template <class T, class dCompareKey>
void Sort(T* const array, int elements)
{
    const int batchSize = 8;
    int stack[1024][2];

    stack[0][0] = 0;
    stack[0][1] = elements - 1;
    int stackIndex = 1;
    const dCompareKey comparator;
    while (stackIndex)
    {
        stackIndex--;
        int lo = stack[stackIndex][0];
        int hi = stack[stackIndex][1];
        if ((hi - lo) > batchSize)
        {
            int mid = (lo + hi) >> 1;
            if (comparator.Compare(array[lo], array[mid]) > 0)
            {
                std::swap(array[lo],
                          array[mid]);
            }
            if (comparator.Compare(array[mid], array[hi]) > 0)
            {
                std::swap(array[mid],
                          array[hi]);
            }
            if (comparator.Compare(array[lo], array[mid]) > 0)
            {
                std::swap(array[lo],
                          array[mid]);
            }
            int i = lo + 1;
            int j = hi - 1;
            const T pivot(array[mid]);
            do
            {
                while (comparator.Compare(array[i], pivot) < 0)
                {
                    i++;
                }
                while (comparator.Compare(array[j], pivot) > 0)
                {
                    j--;
                }

                if (i <= j)
                {
                    std::swap(array[i],
                              array[j]);
                    i++;
                    j--;
                }
            } while (i <= j);

            if (i < hi)
            {
                stack[stackIndex][0] = i;
                stack[stackIndex][1] = hi;
                stackIndex++;
            }
            if (lo < j)
            {
                stack[stackIndex][0] = lo;
                stack[stackIndex][1] = j;
                stackIndex++;
            }
            assert(stackIndex < int(sizeof(stack) / (2 * sizeof(stack[0][0]))));
        }
    }

    int stride = batchSize + 1;
    if (elements < stride)
    {
        stride = elements;
    }
    for (int i = 1; i < stride; ++i)
    {
        if (comparator.Compare(array[0], array[i]) > 0)
        {
            std::swap(array[0],
                      array[i]);
        }
    }

    for (int i = 1; i < elements; ++i)
    {
        int j = i;
        const T tmp(array[i]);
        for (; comparator.Compare(array[j - 1], tmp) > 0; --j)
        {
            assert(j > 0);
            array[j] = array[j - 1];
        }
        array[j] = tmp;
    }
}

/*
Maintaining comment due to attribution
Purpose:

TRIANGLE_AREA_3D computes the area of a triangle in 3D.

Modified:

22 April 1999

Author:

John Burkardt

Parameters:

Input, double X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3, the (getX,getY,getZ)
coordinates of the corners of the triangle.

Output, double TRIANGLE_AREA_3D, the area of the triangle.
*/
double ComputeArea(const VHACD::Vect3& p1,
                   const VHACD::Vect3& p2,
                   const VHACD::Vect3& p3)
{
    /*
    Find the projection of (P3-P1) onto (P2-P1).
    */
    double base = (p2 - p1).GetNorm();
    /*
    The height of the triangle is the length of (P3-P1) after its
    projection onto (P2-P1) has been subtracted.
    */
    double height;
    if (base == double(0.0))
    {
        height = double(0.0);
    }
    else
    {
        double dot = (p3 - p1).Dot(p2 - p1);
        double alpha = dot / (base * base);

        VHACD::Vect3 a = p3 - p1 - alpha * (p2 - p1);
        height = a.GetNorm();
    }

    return double(0.5) * base * height;
}

bool ComputeCentroid(const std::vector<VHACD::Vertex>& points,
                     const std::vector<VHACD::Triangle>& indices,
                     VHACD::Vect3& center)

{
    bool ret = false;
    if (points.size())
    {
        center = VHACD::Vect3(0);

        VHACD::Vect3 numerator(0);
        double denominator = 0;

        for (uint32_t i = 0; i < indices.size(); i++)
        {
            uint32_t i1 = indices[i].mI0;
            uint32_t i2 = indices[i].mI1;
            uint32_t i3 = indices[i].mI2;

            const VHACD::Vect3& p1 = points[i1];
            const VHACD::Vect3& p2 = points[i2];
            const VHACD::Vect3& p3 = points[i3];

            // Compute the average of the sum of the three positions
            VHACD::Vect3 sum = (p1 + p2 + p3) / 3;

            // Compute the area of this triangle
            double area = ComputeArea(p1,
                                      p2,
                                      p3);

            numerator += (sum * area);

            denominator += area;
        }
        double recip = 1 / denominator;
        center = numerator * recip;
        ret = true;
    }
    return ret;
}

double Determinant3x3(const std::array<VHACD::Vect3, 3>& matrix,
                      double& error)
{
    double det = double(0.0);
    error = double(0.0);

    double a01xa12 = matrix[0].GetY() * matrix[1].GetZ();
    double a02xa11 = matrix[0].GetZ() * matrix[1].GetY();
    error += (std::abs(a01xa12) + std::abs(a02xa11)) * std::abs(matrix[2].GetX());
    det += (a01xa12 - a02xa11) * matrix[2].GetX();

    double a00xa12 = matrix[0].GetX() * matrix[1].GetZ();
    double a02xa10 = matrix[0].GetZ() * matrix[1].GetX();
    error += (std::abs(a00xa12) + std::abs(a02xa10)) * std::abs(matrix[2].GetY());
    det -= (a00xa12 - a02xa10) * matrix[2].GetY();

    double a00xa11 = matrix[0].GetX() * matrix[1].GetY();
    double a01xa10 = matrix[0].GetY() * matrix[1].GetX();
    error += (std::abs(a00xa11) + std::abs(a01xa10)) * std::abs(matrix[2].GetZ());
    det += (a00xa11 - a01xa10) * matrix[2].GetZ();

    return det;
}

double ComputeMeshVolume(const std::vector<VHACD::Vertex>& vertices,
                         const std::vector<VHACD::Triangle>& indices)
{
    double volume = 0;
    for (uint32_t i = 0; i < indices.size(); i++)
    {
        const std::array<VHACD::Vect3, 3> m = {
            vertices[indices[i].mI0],
            vertices[indices[i].mI1],
            vertices[indices[i].mI2]
        };
        double placeholder;
        volume += Determinant3x3(m,
                                 placeholder);
    }

    volume *= (double(1.0) / double(6.0));
    if (volume < 0)
        volume *= -1;
    return volume;
}

/*
 * To minimize memory allocations while maintaining pointer stability.
 * Used in KdTreeNode and ConvexHull, as both use tree data structures that rely on pointer stability
 * Neither rely on random access or iteration
 * They just dump elements into a memory pool, then refer to pointers to the elements
 * All elements are default constructed in NodeStorage's m_nodes array
 */
template <typename T, std::size_t MaxBundleSize = 1024>
class NodeBundle
{
    struct NodeStorage {
        bool IsFull() const;

        T& GetNextNode();

        std::size_t m_index;
        std::array<T, MaxBundleSize> m_nodes;
    };

    std::list<NodeStorage> m_list;
    typename std::list<NodeStorage>::iterator m_head{ m_list.end() };

public:
    T& GetNextNode();

    T& GetFirstNode();

    void Clear();
};

template <typename T, std::size_t MaxBundleSize>
bool NodeBundle<T, MaxBundleSize>::NodeStorage::IsFull() const
{
    return m_index == MaxBundleSize;
}

template <typename T, std::size_t MaxBundleSize>
T& NodeBundle<T, MaxBundleSize>::NodeStorage::GetNextNode()
{
    assert(m_index < MaxBundleSize);
    T& ret = m_nodes[m_index];
    m_index++;
    return ret;
}

template <typename T, std::size_t MaxBundleSize>
T& NodeBundle<T, MaxBundleSize>::GetNextNode()
{
    /*
     * || short circuits, so doesn't dereference if m_bundle == m_bundleHead.end()
     */
    if (   m_head == m_list.end()
        || m_head->IsFull())
    {
        m_head = m_list.emplace(m_list.end());
    }

    return m_head->GetNextNode();
}

template <typename T, std::size_t MaxBundleSize>
T& NodeBundle<T, MaxBundleSize>::GetFirstNode()
{
    assert(m_head != m_list.end());
    return m_list.front().m_nodes[0];
}

template <typename T, std::size_t MaxBundleSize>
void NodeBundle<T, MaxBundleSize>::Clear()
{
    m_list.clear();
}

/*
 * Returns index of highest set bit in x
 */
inline int dExp2(int x)
{
    int exp;
    for (exp = -1; x; x >>= 1)
    {
        exp++;
    }
    return exp;
}

/*
 * Reverses the order of the bits in v and returns the result
 * Does not put fill any of the bits higher than the highest bit in v
 * Only used to calculate index of ndNormalMap::m_normal when tessellating a triangle
 */
inline int dBitReversal(int v,
                        int base)
{
    int x = 0;
    int power = dExp2(base) - 1;
    do
    {
        x += (v & 1) << power;
        v >>= 1;
        power--;
    } while (v);
    return x;
}

class Googol
{
    #define VHACD_GOOGOL_SIZE 4
public:
    Googol() = default;
    Googol(double value);

    operator double() const;
    Googol operator+(const Googol &A) const;
    Googol operator-(const Googol &A) const;
    Googol operator*(const Googol &A) const;
    Googol operator/ (const Googol &A) const;

    Googol& operator+= (const Googol &A);
    Googol& operator-= (const Googol &A);

    bool operator>(const Googol &A) const;
    bool operator>=(const Googol &A) const;
    bool operator<(const Googol &A) const;
    bool operator<=(const Googol &A) const;
    bool operator==(const Googol &A) const;
    bool operator!=(const Googol &A) const;

    Googol Abs() const;
    Googol Floor() const;
    Googol InvSqrt() const;
    Googol Sqrt() const;

    void ToString(char* const string) const;

private:
    void NegateMantissa(std::array<uint64_t, VHACD_GOOGOL_SIZE>& mantissa) const;
    void CopySignedMantissa(std::array<uint64_t, VHACD_GOOGOL_SIZE>& mantissa) const;
    int NormalizeMantissa(std::array<uint64_t, VHACD_GOOGOL_SIZE>& mantissa) const;
    void ShiftRightMantissa(std::array<uint64_t, VHACD_GOOGOL_SIZE>& mantissa,
                            int bits) const;
    uint64_t CheckCarrier(uint64_t a, uint64_t b) const;

    int LeadingZeros(uint64_t a) const;
    void ExtendedMultiply(uint64_t a,
                          uint64_t b,
                          uint64_t& high,
                          uint64_t& low) const;
    void ScaleMantissa(uint64_t* out,
                       uint64_t scale) const;

    int m_sign{ 0 };
    int m_exponent{ 0 };
    std::array<uint64_t, VHACD_GOOGOL_SIZE> m_mantissa{ 0 };

public:
    static Googol m_zero;
    static Googol m_one;
    static Googol m_two;
    static Googol m_three;
    static Googol m_half;
};

Googol Googol::m_zero(double(0.0));
Googol Googol::m_one(double(1.0));
Googol Googol::m_two(double(2.0));
Googol Googol::m_three(double(3.0));
Googol Googol::m_half(double(0.5));

Googol::Googol(double value)
{
    int exp;
    double mantissa = fabs(frexp(value, &exp));

    m_exponent = exp;
    m_sign = (value >= 0) ? 0 : 1;

    m_mantissa[0] = uint64_t(double(uint64_t(1) << 62) * mantissa);
}

Googol::operator double() const
{
    double mantissa = (double(1.0) / double(uint64_t(1) << 62)) * double(m_mantissa[0]);
    mantissa = ldexp(mantissa, m_exponent) * (m_sign ? double(-1.0) : double(1.0));
    return mantissa;
}

Googol Googol::operator+(const Googol &A) const
{
    Googol tmp;
    if (m_mantissa[0] && A.m_mantissa[0])
    {
        std::array<uint64_t, VHACD_GOOGOL_SIZE> mantissa0;
        std::array<uint64_t, VHACD_GOOGOL_SIZE> mantissa1;
        std::array<uint64_t, VHACD_GOOGOL_SIZE> mantissa;

        CopySignedMantissa(mantissa0);
        A.CopySignedMantissa(mantissa1);

        int exponentDiff = m_exponent - A.m_exponent;
        int exponent = m_exponent;
        if (exponentDiff > 0)
        {
            ShiftRightMantissa(mantissa1,
                               exponentDiff);
        }
        else if (exponentDiff < 0)
        {
            exponent = A.m_exponent;
            ShiftRightMantissa(mantissa0,
                               -exponentDiff);
        }

        uint64_t carrier = 0;
        for (int i = VHACD_GOOGOL_SIZE - 1; i >= 0; i--)
        {
            uint64_t m0 = mantissa0[i];
            uint64_t m1 = mantissa1[i];
            mantissa[i] = m0 + m1 + carrier;
            carrier = CheckCarrier(m0, m1) | CheckCarrier(m0 + m1, carrier);
        }

        int sign = 0;
        if (int64_t(mantissa[0]) < 0)
        {
            sign = 1;
            NegateMantissa(mantissa);
        }

        int bits = NormalizeMantissa(mantissa);
        if (bits <= (-64 * VHACD_GOOGOL_SIZE))
        {
            tmp.m_sign = 0;
            tmp.m_exponent = 0;
        }
        else
        {
            tmp.m_sign = sign;
            tmp.m_exponent = int(exponent + bits);
        }

        tmp.m_mantissa = mantissa;
    }
    else if (A.m_mantissa[0])
    {
        tmp = A;
    }
    else
    {
        tmp = *this;
    }

    return tmp;
}

Googol Googol::operator-(const Googol &A) const
{
    Googol tmp(A);
    tmp.m_sign = !tmp.m_sign;
    return *this + tmp;
}

Googol Googol::operator*(const Googol &A) const
{
    if (m_mantissa[0] && A.m_mantissa[0])
    {
        std::array<uint64_t, VHACD_GOOGOL_SIZE * 2> mantissaAcc{ 0 };
        for (int i = VHACD_GOOGOL_SIZE - 1; i >= 0; i--)
        {
            uint64_t a = m_mantissa[i];
            if (a)
            {
                uint64_t mantissaScale[2 * VHACD_GOOGOL_SIZE] = { 0 };
                A.ScaleMantissa(&mantissaScale[i], a);

                uint64_t carrier = 0;
                for (int j = 0; j < 2 * VHACD_GOOGOL_SIZE; j++)
                {
                    const int k = 2 * VHACD_GOOGOL_SIZE - 1 - j;
                    uint64_t m0 = mantissaAcc[k];
                    uint64_t m1 = mantissaScale[k];
                    mantissaAcc[k] = m0 + m1 + carrier;
                    carrier = CheckCarrier(m0, m1) | CheckCarrier(m0 + m1, carrier);
                }
            }
        }

        uint64_t carrier = 0;
        int bits = LeadingZeros(mantissaAcc[0]) - 2;
        for (int i = 0; i < 2 * VHACD_GOOGOL_SIZE; i++)
        {
            const int k = 2 * VHACD_GOOGOL_SIZE - 1 - i;
            uint64_t a = mantissaAcc[k];
            mantissaAcc[k] = (a << uint64_t(bits)) | carrier;
            carrier = a >> uint64_t(64 - bits);
        }

        int exp = m_exponent + A.m_exponent - (bits - 2);

        Googol tmp;
        tmp.m_sign = m_sign ^ A.m_sign;
        tmp.m_exponent = exp;
        for (std::size_t i = 0; i < tmp.m_mantissa.size(); ++i)
        {
            tmp.m_mantissa[i] = mantissaAcc[i];
        }

        return tmp;
    }
    return Googol(double(0.0));
}

Googol Googol::operator/(const Googol &A) const
{
    Googol tmp(double(1.0) / A);
    tmp = tmp * (m_two - A * tmp);
    tmp = tmp * (m_two - A * tmp);
    bool test = false;
    int passes = 0;
    do
    {
        passes++;
        Googol tmp0(tmp);
        tmp = tmp * (m_two - A * tmp);
        test = tmp0 == tmp;
    } while (test && (passes < (2 * VHACD_GOOGOL_SIZE)));
    return (*this) * tmp;
}

Googol& Googol::operator+=(const Googol &A)
{
    *this = *this + A;
    return *this;
}

Googol& Googol::operator-=(const Googol &A)
{
    *this = *this - A;
    return *this;
}

bool Googol::operator>(const Googol &A) const
{
    Googol tmp(*this - A);
    return double(tmp) > double(0.0);
}

bool Googol::operator>=(const Googol &A) const
{
    Googol tmp(*this - A);
    return double(tmp) >= double(0.0);
}

bool Googol::operator<(const Googol &A) const
{
    Googol tmp(*this - A);
    return double(tmp) < double(0.0);
}

bool Googol::operator<=(const Googol &A) const
{
    Googol tmp(*this - A);
    return double(tmp) <= double(0.0);
}

bool Googol::operator==(const Googol &A) const
{
    return    m_sign == A.m_sign
           && m_exponent == A.m_exponent
           && m_mantissa == A.m_mantissa;
}

bool Googol::operator!=(const Googol &A) const
{
    return !(*this == A);
}

Googol Googol::Abs() const
{
    Googol tmp(*this);
    tmp.m_sign = 0;
    return tmp;
}

Googol Googol::Floor() const
{
    if (m_exponent < 1)
    {
        return Googol(double(0.0));
    }
    int bits = m_exponent + 2;
    int start = 0;
    while (bits >= 64)
    {
        bits -= 64;
        start++;
    }

    Googol tmp(*this);
    for (int i = VHACD_GOOGOL_SIZE - 1; i > start; i--)
    {
        tmp.m_mantissa[i] = 0;
    }
    // some compilers do no like this and I do not know why is that
    //uint64_t mask = (-1LL) << (64 - bits);
    uint64_t mask(~0ULL);
    mask <<= (64 - bits);
    tmp.m_mantissa[start] &= mask;
    return tmp;
}

Googol Googol::InvSqrt() const
{
    const Googol& me = *this;
    Googol x(double(1.0) / sqrt(me));

    int test = 0;
    int passes = 0;
    do
    {
        passes++;
        Googol tmp(x);
        x = m_half * x * (m_three - me * x * x);
        test = (x != tmp);
    } while (test && (passes < (2 * VHACD_GOOGOL_SIZE)));
    return x;
}

Googol Googol::Sqrt() const
{
    return *this * InvSqrt();
}

void Googol::ToString(char* const string) const
{
    Googol tmp(*this);
    Googol base(double(10.0));
    while (double(tmp) > double(1.0))
    {
        tmp = tmp / base;
    }

    int index = 0;
    while (tmp.m_mantissa[0])
    {
        tmp = tmp * base;
        Googol digit(tmp.Floor());
        tmp -= digit;
        double val = digit;
        string[index] = char(val) + '0';
        index++;
    }
    string[index] = 0;
}

void Googol::NegateMantissa(std::array<uint64_t, VHACD_GOOGOL_SIZE>& mantissa) const
{
    uint64_t carrier = 1;
    for (size_t i = mantissa.size() - 1; i < mantissa.size(); i--)
    {
        uint64_t a = ~mantissa[i] + carrier;
        if (a)
        {
            carrier = 0;
        }
        mantissa[i] = a;
    }
}

void Googol::CopySignedMantissa(std::array<uint64_t, VHACD_GOOGOL_SIZE>& mantissa) const
{
    mantissa = m_mantissa;
    if (m_sign)
    {
        NegateMantissa(mantissa);
    }
}

int Googol::NormalizeMantissa(std::array<uint64_t, VHACD_GOOGOL_SIZE>& mantissa) const
{
    int bits = 0;
    if (int64_t(mantissa[0] * 2) < 0)
    {
        bits = 1;
        ShiftRightMantissa(mantissa, 1);
    }
    else
    {
        while (!mantissa[0] && bits > (-64 * VHACD_GOOGOL_SIZE))
        {
            bits -= 64;
            for (int i = 1; i < VHACD_GOOGOL_SIZE; i++) {
                mantissa[i - 1] = mantissa[i];
            }
            mantissa[VHACD_GOOGOL_SIZE - 1] = 0;
        }

        if (bits > (-64 * VHACD_GOOGOL_SIZE))
        {
            int n = LeadingZeros(mantissa[0]) - 2;
            if (n > 0)
            {
                uint64_t carrier = 0;
                for (int i = VHACD_GOOGOL_SIZE - 1; i >= 0; i--)
                {
                    uint64_t a = mantissa[i];
                    mantissa[i] = (a << n) | carrier;
                    carrier = a >> (64 - n);
                }
                bits -= n;
            }
            else if (n < 0)
            {
                // this is very rare but it does happens, whee the leading zeros of the mantissa is an exact multiple of 64
                uint64_t carrier = 0;
                int shift = -n;
                for (int i = 0; i < VHACD_GOOGOL_SIZE; i++)
                {
                    uint64_t a = mantissa[i];
                    mantissa[i] = (a >> shift) | carrier;
                    carrier = a << (64 - shift);
                }
                bits -= n;
            }
        }
    }
    return bits;
}

void Googol::ShiftRightMantissa(std::array<uint64_t, VHACD_GOOGOL_SIZE>& mantissa,
                                int bits) const
{
    uint64_t carrier = 0;
    if (int64_t(mantissa[0]) < int64_t(0))
    {
        carrier = uint64_t(-1);
    }

    while (bits >= 64)
    {
        for (int i = VHACD_GOOGOL_SIZE - 2; i >= 0; i--)
        {
            mantissa[i + 1] = mantissa[i];
        }
        mantissa[0] = carrier;
        bits -= 64;
    }

    if (bits > 0)
    {
        carrier <<= (64 - bits);
        for (int i = 0; i < VHACD_GOOGOL_SIZE; i++)
        {
            uint64_t a = mantissa[i];
            mantissa[i] = (a >> bits) | carrier;
            carrier = a << (64 - bits);
        }
    }
}

uint64_t Googol::CheckCarrier(uint64_t a, uint64_t b) const
{
    return ((uint64_t(-1) - b) < a) ? uint64_t(1) : 0;
}

int Googol::LeadingZeros(uint64_t a) const
{
    #define VHACD_COUNTBIT(mask, add)	\
    do {								\
        uint64_t test = a & mask;		\
        n += test ? 0 : add;			\
        a = test ? test : (a & ~mask);	\
    } while (false)

    int n = 0;
    VHACD_COUNTBIT(0xffffffff00000000LL, 32);
    VHACD_COUNTBIT(0xffff0000ffff0000LL, 16);
    VHACD_COUNTBIT(0xff00ff00ff00ff00LL, 8);
    VHACD_COUNTBIT(0xf0f0f0f0f0f0f0f0LL, 4);
    VHACD_COUNTBIT(0xccccccccccccccccLL, 2);
    VHACD_COUNTBIT(0xaaaaaaaaaaaaaaaaLL, 1);

    return n;
}

void Googol::ExtendedMultiply(uint64_t a,
                              uint64_t b,
                              uint64_t& high,
                              uint64_t& low) const
{
    uint64_t bLow = b & 0xffffffff;
    uint64_t bHigh = b >> 32;
    uint64_t aLow = a & 0xffffffff;
    uint64_t aHigh = a >> 32;

    uint64_t l = bLow * aLow;

    uint64_t c1 = bHigh * aLow;
    uint64_t c2 = bLow * aHigh;
    uint64_t m = c1 + c2;
    uint64_t carrier = CheckCarrier(c1, c2) << 32;

    uint64_t h = bHigh * aHigh + carrier;

    uint64_t ml = m << 32;
    uint64_t ll = l + ml;
    uint64_t mh = (m >> 32) + CheckCarrier(l, ml);
    uint64_t hh = h + mh;

    low = ll;
    high = hh;
}

void Googol::ScaleMantissa(uint64_t* dst,
                           uint64_t scale) const
{
    uint64_t carrier = 0;
    for (int i = VHACD_GOOGOL_SIZE - 1; i >= 0; i--)
    {
        if (m_mantissa[i])
        {
            uint64_t low;
            uint64_t high;
            ExtendedMultiply(scale,
                             m_mantissa[i],
                             high,
                             low);
            uint64_t acc = low + carrier;
            carrier = CheckCarrier(low,
                                   carrier);
            carrier += high;
            dst[i + 1] = acc;
        }
        else
        {
            dst[i + 1] = carrier;
            carrier = 0;
        }

    }
    dst[0] = carrier;
}

Googol Determinant3x3(const std::array<VHACD::Vector3<Googol>, 3>& matrix)
{
    Googol det = double(0.0);

    Googol a01xa12 = matrix[0].GetY() * matrix[1].GetZ();
    Googol a02xa11 = matrix[0].GetZ() * matrix[1].GetY();
    det += (a01xa12 - a02xa11) * matrix[2].GetX();

    Googol a00xa12 = matrix[0].GetX() * matrix[1].GetZ();
    Googol a02xa10 = matrix[0].GetZ() * matrix[1].GetX();
    det -= (a00xa12 - a02xa10) * matrix[2].GetY();

    Googol a00xa11 = matrix[0].GetX() * matrix[1].GetY();
    Googol a01xa10 = matrix[0].GetY() * matrix[1].GetX();
    det += (a00xa11 - a01xa10) * matrix[2].GetZ();
    return det;
}

class HullPlane : public VHACD::Vect3
{
public:
    HullPlane(const HullPlane&) = default;
    HullPlane(double x,
              double y,
              double z,
              double w);

    HullPlane(const VHACD::Vect3& p,
              double w);

    HullPlane(const VHACD::Vect3& p0,
              const VHACD::Vect3& p1,
              const VHACD::Vect3& p2);

    HullPlane Scale(double s) const;

    HullPlane& operator=(const HullPlane& rhs);

    double Evalue(const VHACD::Vect3 &point) const;

    double& GetW();
    const double& GetW() const;

private:
    double m_w;
};

HullPlane::HullPlane(double x,
                     double y,
                     double z,
                     double w)
    : VHACD::Vect3(x, y, z)
    , m_w(w)
{
}

HullPlane::HullPlane(const VHACD::Vect3& p,
                     double w)
    : VHACD::Vect3(p)
    , m_w(w)
{
}

HullPlane::HullPlane(const VHACD::Vect3& p0,
                     const VHACD::Vect3& p1,
                     const VHACD::Vect3& p2)
    : VHACD::Vect3((p1 - p0).Cross(p2 - p0))
    , m_w(-Dot(p0))
{
}

HullPlane HullPlane::Scale(double s) const
{
    return HullPlane(*this * s,
                     m_w * s);
}

HullPlane& HullPlane::operator=(const HullPlane& rhs)
{
    GetX() = rhs.GetX();
    GetY() = rhs.GetY();
    GetZ() = rhs.GetZ();
    m_w = rhs.m_w;
    return *this;
}

double HullPlane::Evalue(const VHACD::Vect3& point) const
{
    return Dot(point) + m_w;
}

double& HullPlane::GetW()
{
    return m_w;
}

const double& HullPlane::GetW() const
{
    return m_w;
}

class ConvexHullFace
{
public:
    ConvexHullFace() = default;
    double Evalue(const std::vector<VHACD::Vect3>& pointArray,
                  const VHACD::Vect3& point) const;
    HullPlane GetPlaneEquation(const std::vector<VHACD::Vect3>& pointArray,
                               bool& isValid) const;

    std::array<int, 3> m_index;
private:
    int m_mark{ 0 };
    std::array<std::list<ConvexHullFace>::iterator, 3> m_twin;

    friend class ConvexHull;
};

double ConvexHullFace::Evalue(const std::vector<VHACD::Vect3>& pointArray,
                              const VHACD::Vect3& point) const
{
    const VHACD::Vect3& p0 = pointArray[m_index[0]];
    const VHACD::Vect3& p1 = pointArray[m_index[1]];
    const VHACD::Vect3& p2 = pointArray[m_index[2]];

    std::array<VHACD::Vect3, 3> matrix = { p2 - p0, p1 - p0, point - p0 };
    double error;
    double det = Determinant3x3(matrix,
                                error);

    // the code use double, however the threshold for accuracy test is the machine precision of a float.
    // by changing this to a smaller number, the code should run faster since many small test will be considered valid
    // the precision must be a power of two no smaller than the machine precision of a double, (1<<48)
    // float64(1<<30) can be a good value

    // double precision	= double (1.0f) / double (1<<30);
    double precision = double(1.0) / double(1 << 24);
    double errbound = error * precision;
    if (fabs(det) > errbound)
    {
        return det;
    }

    const VHACD::Vector3<Googol> p0g = pointArray[m_index[0]];
    const VHACD::Vector3<Googol> p1g = pointArray[m_index[1]];
    const VHACD::Vector3<Googol> p2g = pointArray[m_index[2]];
    const VHACD::Vector3<Googol> pointg = point;
    std::array<VHACD::Vector3<Googol>, 3> exactMatrix = { p2g - p0g, p1g - p0g, pointg - p0g };
    return Determinant3x3(exactMatrix);
}

HullPlane ConvexHullFace::GetPlaneEquation(const std::vector<VHACD::Vect3>& pointArray,
                                           bool& isvalid) const
{
    const VHACD::Vect3& p0 = pointArray[m_index[0]];
    const VHACD::Vect3& p1 = pointArray[m_index[1]];
    const VHACD::Vect3& p2 = pointArray[m_index[2]];
    HullPlane plane(p0, p1, p2);

    isvalid = false;
    double mag2 = plane.Dot(plane);
    if (mag2 > double(1.0e-16))
    {
        isvalid = true;
        plane = plane.Scale(double(1.0) / sqrt(mag2));
    }
    return plane;
}

class ConvexHullVertex : public VHACD::Vect3
{
public:
    ConvexHullVertex() = default;
    ConvexHullVertex(const ConvexHullVertex&) = default;
    ConvexHullVertex& operator=(const ConvexHullVertex& rhs) = default;
    using VHACD::Vect3::operator=;

    int m_mark{ 0 };
};


class ConvexHullAABBTreeNode
{
    #define VHACD_CONVEXHULL_3D_VERTEX_CLUSTER_SIZE 8
public:
    ConvexHullAABBTreeNode() = default;
    ConvexHullAABBTreeNode(ConvexHullAABBTreeNode* parent);

    VHACD::Vect3 m_box[2];
    ConvexHullAABBTreeNode* m_left{ nullptr };
    ConvexHullAABBTreeNode* m_right{ nullptr };
    ConvexHullAABBTreeNode* m_parent{ nullptr };

    size_t m_count;
    std::array<size_t, VHACD_CONVEXHULL_3D_VERTEX_CLUSTER_SIZE> m_indices;
};

ConvexHullAABBTreeNode::ConvexHullAABBTreeNode(ConvexHullAABBTreeNode* parent)
    : m_parent(parent)
{
}

class ConvexHull
{
    class ndNormalMap;

public:
    ConvexHull(const ConvexHull& source);
    ConvexHull(const std::vector<::VHACD::Vertex>& vertexCloud,
               double distTol,
               int maxVertexCount = 0x7fffffff);
    ~ConvexHull() = default;

    const std::vector<VHACD::Vect3>& GetVertexPool() const;

    const std::list<ConvexHullFace>& GetList() const { return m_list; }

private:
    void BuildHull(const std::vector<::VHACD::Vertex>& vertexCloud,
                   double distTol,
                   int maxVertexCount);

    void GetUniquePoints(std::vector<ConvexHullVertex>& points);
    int InitVertexArray(std::vector<ConvexHullVertex>& points,
                        NodeBundle<ConvexHullAABBTreeNode>& memoryPool);

    ConvexHullAABBTreeNode* BuildTreeNew(std::vector<ConvexHullVertex>& points,
                                         std::vector<ConvexHullAABBTreeNode>& memoryPool) const;
    ConvexHullAABBTreeNode* BuildTreeOld(std::vector<ConvexHullVertex>& points,
                                         NodeBundle<ConvexHullAABBTreeNode>& memoryPool);
    ConvexHullAABBTreeNode* BuildTreeRecurse(ConvexHullAABBTreeNode* const parent,
                                             ConvexHullVertex* const points,
                                             int count,
                                             int baseIndex,
                                             NodeBundle<ConvexHullAABBTreeNode>& memoryPool) const;

    std::list<ConvexHullFace>::iterator AddFace(int i0,
                                                int i1,
                                                int i2);

    void CalculateConvexHull3D(ConvexHullAABBTreeNode* vertexTree,
                               std::vector<ConvexHullVertex>& points,
                               int count,
                               double distTol,
                               int maxVertexCount);

    int SupportVertex(ConvexHullAABBTreeNode** const tree,
                      const std::vector<ConvexHullVertex>& points,
                      const VHACD::Vect3& dir,
                      const bool removeEntry = true) const;
    double TetrahedrumVolume(const VHACD::Vect3& p0,
                             const VHACD::Vect3& p1,
                             const VHACD::Vect3& p2,
                             const VHACD::Vect3& p3) const;

    std::list<ConvexHullFace> m_list;
    VHACD::Vect3 m_aabbP0{ 0 };
    VHACD::Vect3 m_aabbP1{ 0 };
    double m_diag{ 0.0 };
    std::vector<VHACD::Vect3> m_points;
};

class ConvexHull::ndNormalMap
{
public:
    ndNormalMap();

    static const ndNormalMap& GetNormalMap();

    void TessellateTriangle(int level,
                            const VHACD::Vect3& p0,
                            const VHACD::Vect3& p1,
                            const VHACD::Vect3& p2,
                            int& count);

    std::array<VHACD::Vect3, 128> m_normal;
    int m_count{ 128 };
};

const ConvexHull::ndNormalMap& ConvexHull::ndNormalMap::GetNormalMap()
{
    static ndNormalMap normalMap;
    return normalMap;
}

void ConvexHull::ndNormalMap::TessellateTriangle(int level,
                                                 const VHACD::Vect3& p0,
                                                 const VHACD::Vect3& p1,
                                                 const VHACD::Vect3& p2,
                                                 int& count)
{
    if (level)
    {
        assert(fabs(p0.Dot(p0) - double(1.0)) < double(1.0e-4));
        assert(fabs(p1.Dot(p1) - double(1.0)) < double(1.0e-4));
        assert(fabs(p2.Dot(p2) - double(1.0)) < double(1.0e-4));
        VHACD::Vect3 p01(p0 + p1);
        VHACD::Vect3 p12(p1 + p2);
        VHACD::Vect3 p20(p2 + p0);

        p01 = p01 * (double(1.0) / p01.GetNorm());
        p12 = p12 * (double(1.0) / p12.GetNorm());
        p20 = p20 * (double(1.0) / p20.GetNorm());

        assert(fabs(p01.GetNormSquared() - double(1.0)) < double(1.0e-4));
        assert(fabs(p12.GetNormSquared() - double(1.0)) < double(1.0e-4));
        assert(fabs(p20.GetNormSquared() - double(1.0)) < double(1.0e-4));

        TessellateTriangle(level - 1, p0,  p01, p20, count);
        TessellateTriangle(level - 1, p1,  p12, p01, count);
        TessellateTriangle(level - 1, p2,  p20, p12, count);
        TessellateTriangle(level - 1, p01, p12, p20, count);
    }
    else
    {
        /*
         * This is just m_normal[index] = n.Normalized(), but due to tiny floating point errors, causes
         * different outputs, so I'm leaving it
         */
        HullPlane n(p0, p1, p2);
        n = n.Scale(double(1.0) / n.GetNorm());
        n.GetW() = double(0.0);
        int index = dBitReversal(count,
                                 int(m_normal.size()));
        m_normal[index] = n;
        count++;
        assert(count <= int(m_normal.size()));
    }
}

ConvexHull::ndNormalMap::ndNormalMap()
{
    VHACD::Vect3 p0(double( 1.0), double( 0.0), double( 0.0));
    VHACD::Vect3 p1(double(-1.0), double( 0.0), double( 0.0));
    VHACD::Vect3 p2(double( 0.0), double( 1.0), double( 0.0));
    VHACD::Vect3 p3(double( 0.0), double(-1.0), double( 0.0));
    VHACD::Vect3 p4(double( 0.0), double( 0.0), double( 1.0));
    VHACD::Vect3 p5(double( 0.0), double( 0.0), double(-1.0));

    int count = 0;
    int subdivisions = 2;
    TessellateTriangle(subdivisions, p4, p0, p2, count);
    TessellateTriangle(subdivisions, p0, p5, p2, count);
    TessellateTriangle(subdivisions, p5, p1, p2, count);
    TessellateTriangle(subdivisions, p1, p4, p2, count);
    TessellateTriangle(subdivisions, p0, p4, p3, count);
    TessellateTriangle(subdivisions, p5, p0, p3, count);
    TessellateTriangle(subdivisions, p1, p5, p3, count);
    TessellateTriangle(subdivisions, p4, p1, p3, count);
}

ConvexHull::ConvexHull(const std::vector<::VHACD::Vertex>& vertexCloud,
                       double distTol,
                       int maxVertexCount)
{
    if (vertexCloud.size() >= 4)
    {
        BuildHull(vertexCloud,
                  distTol,
                  maxVertexCount);
    }
}

const std::vector<VHACD::Vect3>& ConvexHull::GetVertexPool() const
{
    return m_points;
}

void ConvexHull::BuildHull(const std::vector<::VHACD::Vertex>& vertexCloud,
                           double distTol,
                           int maxVertexCount)
{
    size_t treeCount = vertexCloud.size() / (VHACD_CONVEXHULL_3D_VERTEX_CLUSTER_SIZE >> 1);
    treeCount = std::max(treeCount, size_t(4)) * 2;

    std::vector<ConvexHullVertex> points(vertexCloud.size());
    /*
     * treePool provides a memory pool for the AABB tree
     * Each node is either a leaf or non-leaf node
     * Non-leaf nodes have up to 8 vertices
     * Vertices are specified by the m_indices array and are accessed via the points array
     *
     * Later on in ConvexHull::SupportVertex, the tree is used directly
     * It differentiates between ConvexHullAABBTreeNode and ConvexHull3DPointCluster by whether the m_left and m_right
     * pointers are null or not
     *
     * Pointers have to be stable
     */
    NodeBundle<ConvexHullAABBTreeNode> treePool;
    for (size_t i = 0; i < vertexCloud.size(); ++i)
    {
        points[i] = VHACD::Vect3(vertexCloud[i]);
    }
    int count = InitVertexArray(points,
                                treePool);

    if (m_points.size() >= 4)
    {
        CalculateConvexHull3D(&treePool.GetFirstNode(),
                              points,
                              count,
                              distTol,
                              maxVertexCount);
    }
}

void ConvexHull::GetUniquePoints(std::vector<ConvexHullVertex>& points)
{
    class CompareVertex
    {
        public:
        int Compare(const ConvexHullVertex& elementA, const ConvexHullVertex& elementB) const
        {
            for (int i = 0; i < 3; i++)
            {
                if (elementA[i] < elementB[i])
                {
                    return -1;
                }
                else if (elementA[i] > elementB[i])
                {
                    return 1;
                }
            }
            return 0;
        }
    };

    int count = int(points.size());
    Sort<ConvexHullVertex, CompareVertex>(points.data(),
                                          count);

    int indexCount = 0;
    CompareVertex compareVertex;
    for (int i = 1; i < count; ++i)
    {
        for (; i < count; ++i)
        {
            if (compareVertex.Compare(points[indexCount], points[i]))
            {
                indexCount++;
                points[indexCount] = points[i];
                break;
            }
        }
    }
    points.resize(indexCount + 1);
}

ConvexHullAABBTreeNode* ConvexHull::BuildTreeRecurse(ConvexHullAABBTreeNode* const parent,
                                                     ConvexHullVertex* const points,
                                                     int count,
                                                     int baseIndex,
                                                     NodeBundle<ConvexHullAABBTreeNode>& memoryPool) const
{
    ConvexHullAABBTreeNode* tree = nullptr;

    assert(count);
    VHACD::Vect3 minP( double(1.0e15));
    VHACD::Vect3 maxP(-double(1.0e15));
    if (count <= VHACD_CONVEXHULL_3D_VERTEX_CLUSTER_SIZE)
    {
        ConvexHullAABBTreeNode& clump = memoryPool.GetNextNode();

        clump.m_count = count;
        for (int i = 0; i < count; ++i)
        {
            clump.m_indices[i] = i + baseIndex;

            const VHACD::Vect3& p = points[i];
            minP = minP.CWiseMin(p);
            maxP = maxP.CWiseMax(p);
        }

        clump.m_left = nullptr;
        clump.m_right = nullptr;
        tree = &clump;
    }
    else
    {
        VHACD::Vect3 median(0);
        VHACD::Vect3 varian(0);
        for (int i = 0; i < count; ++i)
        {
            const VHACD::Vect3& p = points[i];
            minP = minP.CWiseMin(p);
            maxP = maxP.CWiseMax(p);
            median += p;
            varian += p.CWiseMul(p);
        }

        varian = varian * double(count) - median.CWiseMul(median);
        int index = 0;
        double maxVarian = double(-1.0e10);
        for (int i = 0; i < 3; ++i)
        {
            if (varian[i] > maxVarian)
            {
                index = i;
                maxVarian = varian[i];
            }
        }
        VHACD::Vect3 center(median * (double(1.0) / double(count)));

        double test = center[index];

        int i0 = 0;
        int i1 = count - 1;
        do
        {
            for (; i0 <= i1; i0++)
            {
                double val = points[i0][index];
                if (val > test)
                {
                    break;
                }
            }

            for (; i1 >= i0; i1--)
            {
                double val = points[i1][index];
                if (val < test)
                {
                    break;
                }
            }

            if (i0 < i1)
            {
                std::swap(points[i0],
                          points[i1]);
                i0++;
                i1--;
            }
        } while (i0 <= i1);

        if (i0 == 0)
        {
            i0 = count / 2;
        }
        if (i0 >= (count - 1))
        {
            i0 = count / 2;
        }

        tree = &memoryPool.GetNextNode();

        assert(i0);
        assert(count - i0);

        tree->m_left = BuildTreeRecurse(tree,
                                        points,
                                        i0,
                                        baseIndex,
                                        memoryPool);
        tree->m_right = BuildTreeRecurse(tree,
                                         &points[i0],
                                         count - i0,
                                         i0 + baseIndex,
                                         memoryPool);
    }

    assert(tree);
    tree->m_parent = parent;
    /*
     * WARNING: Changing the compiler conversion of 1.0e-3f changes the results of the convex decomposition
     * Inflate the tree's bounding box slightly
     */
    tree->m_box[0] = minP - VHACD::Vect3(double(1.0e-3f));
    tree->m_box[1] = maxP + VHACD::Vect3(double(1.0e-3f));
    return tree;
}

ConvexHullAABBTreeNode* ConvexHull::BuildTreeOld(std::vector<ConvexHullVertex>& points,
                                                 NodeBundle<ConvexHullAABBTreeNode>& memoryPool)
{
    GetUniquePoints(points);
    int count = int(points.size());
    if (count < 4)
    {
        return nullptr;
    }
    return BuildTreeRecurse(nullptr,
                            points.data(),
                            count,
                            0,
                            memoryPool);
}

ConvexHullAABBTreeNode* ConvexHull::BuildTreeNew(std::vector<ConvexHullVertex>& points,
                                                 std::vector<ConvexHullAABBTreeNode>& memoryPool) const
{
    class dCluster
    {
        public:
        VHACD::Vect3 m_sum{ double(0.0) };
        VHACD::Vect3 m_sum2{ double(0.0) };
        int m_start{ 0 };
        int m_count{ 0 };
    };

    dCluster firstCluster;
    firstCluster.m_count = int(points.size());

    for (int i = 0; i < firstCluster.m_count; ++i)
    {
        const VHACD::Vect3& p = points[i];
        firstCluster.m_sum += p;
        firstCluster.m_sum2 += p.CWiseMul(p);
    }

    int baseCount = 0;
    const int clusterSize = 16;

    if (firstCluster.m_count > clusterSize)
    {
        dCluster spliteStack[128];
        spliteStack[0] = firstCluster;
        size_t stack = 1;

        while (stack)
        {
            stack--;
            dCluster cluster (spliteStack[stack]);

            const VHACD::Vect3 origin(cluster.m_sum * (double(1.0) / cluster.m_count));
            const VHACD::Vect3 variance2(cluster.m_sum2 * (double(1.0) / cluster.m_count) - origin.CWiseMul(origin));
            double maxVariance2 = variance2.MaxCoeff();

            if (   (cluster.m_count <= clusterSize)
                || (stack > (sizeof(spliteStack) / sizeof(spliteStack[0]) - 4))
                || (maxVariance2 < 1.e-4f))
            {
                // no sure if this is beneficial,
                // the array is so small that seem too much overhead
                //int maxIndex = 0;
                //double min_x = 1.0e20f;
                //for (int i = 0; i < cluster.m_count; ++i)
                //{
                //	if (points[cluster.m_start + i].getX() < min_x)
                //	{
                //		maxIndex = i;
                //		min_x = points[cluster.m_start + i].getX();
                //	}
                //}
                //Swap(points[cluster.m_start], points[cluster.m_start + maxIndex]);
                //
                //for (int i = 2; i < cluster.m_count; ++i)
                //{
                //	int j = i;
                //	ConvexHullVertex tmp(points[cluster.m_start + i]);
                //	for (; points[cluster.m_start + j - 1].getX() > tmp.getX(); --j)
                //	{
                //		assert(j > 0);
                //		points[cluster.m_start + j] = points[cluster.m_start + j - 1];
                //	}
                //	points[cluster.m_start + j] = tmp;
                //}

                int count = cluster.m_count;
                for (int i = cluster.m_count - 1; i > 0; --i)
                {
                    for (int j = i - 1; j >= 0; --j)
                    {
                        VHACD::Vect3 error(points[cluster.m_start + j] - points[cluster.m_start + i]);
                        double mag2 = error.Dot(error);
                        if (mag2 < double(1.0e-6))
                        {
                            points[cluster.m_start + j] = points[cluster.m_start + i];
                            count--;
                            break;
                        }
                    }
                }

                assert(baseCount <= cluster.m_start);
                for (int i = 0; i < count; ++i)
                {
                    points[baseCount] = points[cluster.m_start + i];
                    baseCount++;
                }
            }
            else
            {
                const int firstSortAxis = variance2.LongestAxis();
                double axisVal = origin[firstSortAxis];

                int i0 = 0;
                int i1 = cluster.m_count - 1;

                const int start = cluster.m_start;
                while (i0 < i1)
                {
                    while (   (points[start + i0][firstSortAxis] <= axisVal)
                           && (i0 < i1))
                    {
                        ++i0;
                    };

                    while (   (points[start + i1][firstSortAxis] > axisVal)
                           && (i0 < i1))
                    {
                        --i1;
                    }

                    assert(i0 <= i1);
                    if (i0 < i1)
                    {
                        std::swap(points[start + i0],
                                  points[start + i1]);
                        ++i0;
                        --i1;
                    }
                }

                while (   (points[start + i0][firstSortAxis] <= axisVal)
                       && (i0 < cluster.m_count))
                {
                    ++i0;
                };

                #ifdef _DEBUG
                for (int i = 0; i < i0; ++i)
                {
                    assert(points[start + i][firstSortAxis] <= axisVal);
                }

                for (int i = i0; i < cluster.m_count; ++i)
                {
                    assert(points[start + i][firstSortAxis] > axisVal);
                }
                #endif

                VHACD::Vect3 xc(0);
                VHACD::Vect3 x2c(0);
                for (int i = 0; i < i0; ++i)
                {
                    const VHACD::Vect3& x = points[start + i];
                    xc += x;
                    x2c += x.CWiseMul(x);
                }

                dCluster cluster_i1(cluster);
                cluster_i1.m_start = start + i0;
                cluster_i1.m_count = cluster.m_count - i0;
                cluster_i1.m_sum -= xc;
                cluster_i1.m_sum2 -= x2c;
                spliteStack[stack] = cluster_i1;
                assert(cluster_i1.m_count > 0);
                stack++;

                dCluster cluster_i0(cluster);
                cluster_i0.m_start = start;
                cluster_i0.m_count = i0;
                cluster_i0.m_sum = xc;
                cluster_i0.m_sum2 = x2c;
                assert(cluster_i0.m_count > 0);
                spliteStack[stack] = cluster_i0;
                stack++;
            }
        }
    }

    points.resize(baseCount);
    if (baseCount < 4)
    {
        return nullptr;
    }

    VHACD::Vect3 sum(0);
    VHACD::Vect3 sum2(0);
    VHACD::Vect3 minP(double( 1.0e15));
    VHACD::Vect3 maxP(double(-1.0e15));
    class dTreeBox
    {
        public:
        VHACD::Vect3 m_min;
        VHACD::Vect3 m_max;
        VHACD::Vect3 m_sum;
        VHACD::Vect3 m_sum2;
        ConvexHullAABBTreeNode* m_parent;
        ConvexHullAABBTreeNode** m_child;
        int m_start;
        int m_count;
    };

    for (int i = 0; i < baseCount; ++i)
    {
        const VHACD::Vect3& p = points[i];
        sum += p;
        sum2 += p.CWiseMul(p);
        minP = minP.CWiseMin(p);
        maxP = maxP.CWiseMax(p);
    }

    dTreeBox treeBoxStack[128];
    treeBoxStack[0].m_start = 0;
    treeBoxStack[0].m_count = baseCount;
    treeBoxStack[0].m_sum = sum;
    treeBoxStack[0].m_sum2 = sum2;
    treeBoxStack[0].m_min = minP;
    treeBoxStack[0].m_max = maxP;
    treeBoxStack[0].m_child = nullptr;
    treeBoxStack[0].m_parent = nullptr;

    int stack = 1;
    ConvexHullAABBTreeNode* root = nullptr;
    while (stack)
    {
        stack--;
        dTreeBox box(treeBoxStack[stack]);
        if (box.m_count <= VHACD_CONVEXHULL_3D_VERTEX_CLUSTER_SIZE)
        {
            assert(memoryPool.size() != memoryPool.capacity()
                   && "memoryPool is going to be reallocated, pointers will be invalid");
            memoryPool.emplace_back();
            ConvexHullAABBTreeNode& clump = memoryPool.back();

            clump.m_count = box.m_count;
            for (int i = 0; i < box.m_count; ++i)
            {
                clump.m_indices[i] = i + box.m_start;
            }
            clump.m_box[0] = box.m_min;
            clump.m_box[1] = box.m_max;

            if (box.m_child)
            {
                *box.m_child = &clump;
            }

            if (!root)
            {
                root = &clump;
            }
        }
        else
        {
            const VHACD::Vect3 origin(box.m_sum * (double(1.0) / box.m_count));
            const VHACD::Vect3 variance2(box.m_sum2 * (double(1.0) / box.m_count) - origin.CWiseMul(origin));

            int firstSortAxis = 0;
            if ((variance2.GetY() >= variance2.GetX()) && (variance2.GetY() >= variance2.GetZ()))
            {
                firstSortAxis = 1;
            }
            else if ((variance2.GetZ() >= variance2.GetX()) && (variance2.GetZ() >= variance2.GetY()))
            {
                firstSortAxis = 2;
            }
            double axisVal = origin[firstSortAxis];

            int i0 = 0;
            int i1 = box.m_count - 1;

            const int start = box.m_start;
            while (i0 < i1)
            {
                while ((points[start + i0][firstSortAxis] <= axisVal) && (i0 < i1))
                {
                    ++i0;
                };

                while ((points[start + i1][firstSortAxis] > axisVal) && (i0 < i1))
                {
                    --i1;
                }

                assert(i0 <= i1);
                if (i0 < i1)
                {
                    std::swap(points[start + i0],
                              points[start + i1]);
                    ++i0;
                    --i1;
                }
            }

            while ((points[start + i0][firstSortAxis] <= axisVal) && (i0 < box.m_count))
            {
                ++i0;
            };

            #ifdef _DEBUG
            for (int i = 0; i < i0; ++i)
            {
                assert(points[start + i][firstSortAxis] <= axisVal);
            }

            for (int i = i0; i < box.m_count; ++i)
            {
                assert(points[start + i][firstSortAxis] > axisVal);
            }
            #endif

            assert(memoryPool.size() != memoryPool.capacity()
                   && "memoryPool is going to be reallocated, pointers will be invalid");
            memoryPool.emplace_back();
            ConvexHullAABBTreeNode& node = memoryPool.back();

            node.m_box[0] = box.m_min;
            node.m_box[1] = box.m_max;
            if (box.m_child)
            {
                *box.m_child = &node;
            }

            if (!root)
            {
                root = &node;
            }

            {
                VHACD::Vect3 xc(0);
                VHACD::Vect3 x2c(0);
                VHACD::Vect3 p0(double( 1.0e15));
                VHACD::Vect3 p1(double(-1.0e15));
                for (int i = i0; i < box.m_count; ++i)
                {
                    const VHACD::Vect3& p = points[start + i];
                    xc += p;
                    x2c += p.CWiseMul(p);
                    p0 = p0.CWiseMin(p);
                    p1 = p1.CWiseMax(p);
                }

                dTreeBox cluster_i1(box);
                cluster_i1.m_start = start + i0;
                cluster_i1.m_count = box.m_count - i0;
                cluster_i1.m_sum = xc;
                cluster_i1.m_sum2 = x2c;
                cluster_i1.m_min = p0;
                cluster_i1.m_max = p1;
                cluster_i1.m_parent = &node;
                cluster_i1.m_child = &node.m_right;
                treeBoxStack[stack] = cluster_i1;
                assert(cluster_i1.m_count > 0);
                stack++;
            }

            {
                VHACD::Vect3 xc(0);
                VHACD::Vect3 x2c(0);
                VHACD::Vect3 p0(double( 1.0e15));
                VHACD::Vect3 p1(double(-1.0e15));
                for (int i = 0; i < i0; ++i)
                {
                    const VHACD::Vect3& p = points[start + i];
                    xc += p;
                    x2c += p.CWiseMul(p);
                    p0 = p0.CWiseMin(p);
                    p1 = p1.CWiseMax(p);
                }

                dTreeBox cluster_i0(box);
                cluster_i0.m_start = start;
                cluster_i0.m_count = i0;
                cluster_i0.m_min = p0;
                cluster_i0.m_max = p1;
                cluster_i0.m_sum = xc;
                cluster_i0.m_sum2 = x2c;
                cluster_i0.m_parent = &node;
                cluster_i0.m_child = &node.m_left;
                assert(cluster_i0.m_count > 0);
                treeBoxStack[stack] = cluster_i0;
                stack++;
            }
        }
    }

    return root;
}

int ConvexHull::SupportVertex(ConvexHullAABBTreeNode** const treePointer,
                              const std::vector<ConvexHullVertex>& points,
                              const VHACD::Vect3& dirPlane,
                              const bool removeEntry) const
{
#define VHACD_STACK_DEPTH_3D 64
    double aabbProjection[VHACD_STACK_DEPTH_3D];
    ConvexHullAABBTreeNode* stackPool[VHACD_STACK_DEPTH_3D];

    VHACD::Vect3 dir(dirPlane);

    int index = -1;
    int stack = 1;
    stackPool[0] = *treePointer;
    aabbProjection[0] = double(1.0e20);
    double maxProj = double(-1.0e20);
    int ix = (dir[0] > double(0.0)) ? 1 : 0;
    int iy = (dir[1] > double(0.0)) ? 1 : 0;
    int iz = (dir[2] > double(0.0)) ? 1 : 0;
    while (stack)
    {
        stack--;
        double boxSupportValue = aabbProjection[stack];
        if (boxSupportValue > maxProj)
        {
            ConvexHullAABBTreeNode* me = stackPool[stack];

            /*
             * If the node is not a leaf node...
             */
            if (me->m_left && me->m_right)
            {
                const VHACD::Vect3 leftSupportPoint(me->m_left->m_box[ix].GetX(),
                                                    me->m_left->m_box[iy].GetY(),
                                                    me->m_left->m_box[iz].GetZ());
                double leftSupportDist = leftSupportPoint.Dot(dir);

                const VHACD::Vect3 rightSupportPoint(me->m_right->m_box[ix].GetX(),
                                                     me->m_right->m_box[iy].GetY(),
                                                     me->m_right->m_box[iz].GetZ());
                double rightSupportDist = rightSupportPoint.Dot(dir);

                /*
                 * ...push the shorter side first
                 * So we can explore the tree in the larger side first
                 */
                if (rightSupportDist >= leftSupportDist)
                {
                    aabbProjection[stack] = leftSupportDist;
                    stackPool[stack] = me->m_left;
                    stack++;
                    assert(stack < VHACD_STACK_DEPTH_3D);
                    aabbProjection[stack] = rightSupportDist;
                    stackPool[stack] = me->m_right;
                    stack++;
                    assert(stack < VHACD_STACK_DEPTH_3D);
                }
                else
                {
                    aabbProjection[stack] = rightSupportDist;
                    stackPool[stack] = me->m_right;
                    stack++;
                    assert(stack < VHACD_STACK_DEPTH_3D);
                    aabbProjection[stack] = leftSupportDist;
                    stackPool[stack] = me->m_left;
                    stack++;
                    assert(stack < VHACD_STACK_DEPTH_3D);
                }
            }
            /*
             * If it is a node...
             */
            else
            {
                ConvexHullAABBTreeNode* cluster = me;
                for (size_t i = 0; i < cluster->m_count; ++i)
                {
                    const ConvexHullVertex& p = points[cluster->m_indices[i]];
                    assert(p.GetX() >= cluster->m_box[0].GetX());
                    assert(p.GetX() <= cluster->m_box[1].GetX());
                    assert(p.GetY() >= cluster->m_box[0].GetY());
                    assert(p.GetY() <= cluster->m_box[1].GetY());
                    assert(p.GetZ() >= cluster->m_box[0].GetZ());
                    assert(p.GetZ() <= cluster->m_box[1].GetZ());
                    if (!p.m_mark)
                    {
                        //assert(p.m_w == double(0.0f));
                        double dist = p.Dot(dir);
                        if (dist > maxProj)
                        {
                            maxProj = dist;
                            index = cluster->m_indices[i];
                        }
                    }
                    else if (removeEntry)
                    {
                        cluster->m_indices[i] = cluster->m_indices[cluster->m_count - 1];
                        cluster->m_count = cluster->m_count - 1;
                        i--;
                    }
                }

                if (cluster->m_count == 0)
                {
                    ConvexHullAABBTreeNode* const parent = cluster->m_parent;
                    if (parent)
                    {
                        ConvexHullAABBTreeNode* const sibling = (parent->m_left != cluster) ? parent->m_left : parent->m_right;
                        assert(sibling != cluster);
                        ConvexHullAABBTreeNode* const grandParent = parent->m_parent;
                        if (grandParent)
                        {
                            sibling->m_parent = grandParent;
                            if (grandParent->m_right == parent)
                            {
                                grandParent->m_right = sibling;
                            }
                            else
                            {
                                grandParent->m_left = sibling;
                            }
                        }
                        else
                        {
                            sibling->m_parent = nullptr;
                            *treePointer = sibling;
                        }
                    }
                }
            }
        }
    }

    assert(index != -1);
    return index;
}

double ConvexHull::TetrahedrumVolume(const VHACD::Vect3& p0,
                                     const VHACD::Vect3& p1,
                                     const VHACD::Vect3& p2,
                                     const VHACD::Vect3& p3) const
{
    const VHACD::Vect3 p1p0(p1 - p0);
    const VHACD::Vect3 p2p0(p2 - p0);
    const VHACD::Vect3 p3p0(p3 - p0);
    return p3p0.Dot(p1p0.Cross(p2p0));
}

int ConvexHull::InitVertexArray(std::vector<ConvexHullVertex>& points,
                                NodeBundle<ConvexHullAABBTreeNode>& memoryPool)
//                                 std::vector<ConvexHullAABBTreeNode>& memoryPool)
{
#if 1
    ConvexHullAABBTreeNode* tree = BuildTreeOld(points,
                                                memoryPool);
#else
    ConvexHullAABBTreeNode* tree = BuildTreeNew(points, (char**)&memoryPool, maxMemSize);
#endif
    int count = int(points.size());
    if (count < 4)
    {
        m_points.resize(0);
        return 0;
    }

    m_points.resize(count);
    m_aabbP0 = tree->m_box[0];
    m_aabbP1 = tree->m_box[1];

    VHACD::Vect3 boxSize(tree->m_box[1] - tree->m_box[0]);
    m_diag = boxSize.GetNorm();
    const ndNormalMap& normalMap = ndNormalMap::GetNormalMap();

    int index0 = SupportVertex(&tree,
                               points,
                               normalMap.m_normal[0]);
    m_points[0] = points[index0];
    points[index0].m_mark = 1;

    bool validTetrahedrum = false;
    VHACD::Vect3 e1(double(0.0));
    for (int i = 1; i < normalMap.m_count; ++i)
    {
        int index = SupportVertex(&tree,
                                  points,
                                  normalMap.m_normal[i]);
        assert(index >= 0);

        e1 = points[index] - m_points[0];
        double error2 = e1.GetNormSquared();
        if (error2 > (double(1.0e-4) * m_diag * m_diag))
        {
            m_points[1] = points[index];
            points[index].m_mark = 1;
            validTetrahedrum = true;
            break;
        }
    }
    if (!validTetrahedrum)
    {
        m_points.resize(0);
        assert(0);
        return count;
    }

    validTetrahedrum = false;
    VHACD::Vect3 e2(double(0.0));
    VHACD::Vect3 normal(double(0.0));
    for (int i = 2; i < normalMap.m_count; ++i)
    {
        int index = SupportVertex(&tree,
                                  points,
                                  normalMap.m_normal[i]);
        assert(index >= 0);
        e2 = points[index] - m_points[0];
        normal = e1.Cross(e2);
        double error2 = normal.GetNorm();
        if (error2 > (double(1.0e-4) * m_diag * m_diag))
        {
            m_points[2] = points[index];
            points[index].m_mark = 1;
            validTetrahedrum = true;
            break;
        }
    }

    if (!validTetrahedrum)
    {
        m_points.resize(0);
        assert(0);
        return count;
    }

    // find the largest possible tetrahedron
    validTetrahedrum = false;
    VHACD::Vect3 e3(double(0.0));

    index0 = SupportVertex(&tree,
                           points,
                           normal);
    e3 = points[index0] - m_points[0];
    double err2 = normal.Dot(e3);
    if (fabs(err2) > (double(1.0e-6) * m_diag * m_diag))
    {
        // we found a valid tetrahedral, about and start build the hull by adding the rest of the points
        m_points[3] = points[index0];
        points[index0].m_mark = 1;
        validTetrahedrum = true;
    }
    if (!validTetrahedrum)
    {
        VHACD::Vect3 n(-normal);
        int index = SupportVertex(&tree,
                                  points,
                                  n);
        e3 = points[index] - m_points[0];
        double error2 = normal.Dot(e3);
        if (fabs(error2) > (double(1.0e-6) * m_diag * m_diag))
        {
            // we found a valid tetrahedral, about and start build the hull by adding the rest of the points
            m_points[3] = points[index];
            points[index].m_mark = 1;
            validTetrahedrum = true;
        }
    }
    if (!validTetrahedrum)
    {
        for (int i = 3; i < normalMap.m_count; ++i)
        {
            int index = SupportVertex(&tree,
                                      points,
                                      normalMap.m_normal[i]);
            assert(index >= 0);

            //make sure the volume of the fist tetrahedral is no negative
            e3 = points[index] - m_points[0];
            double error2 = normal.Dot(e3);
            if (fabs(error2) > (double(1.0e-6) * m_diag * m_diag))
            {
                // we found a valid tetrahedral, about and start build the hull by adding the rest of the points
                m_points[3] = points[index];
                points[index].m_mark = 1;
                validTetrahedrum = true;
                break;
            }
        }
    }
    if (!validTetrahedrum)
    {
        // the points do not form a convex hull
        m_points.resize(0);
        return count;
    }

    m_points.resize(4);
    double volume = TetrahedrumVolume(m_points[0],
                                      m_points[1],
                                      m_points[2],
                                      m_points[3]);
    if (volume > double(0.0))
    {
        std::swap(m_points[2],
                  m_points[3]);
    }
    assert(TetrahedrumVolume(m_points[0], m_points[1], m_points[2], m_points[3]) < double(0.0));
    return count;
}

std::list<ConvexHullFace>::iterator ConvexHull::AddFace(int i0,
                                                        int i1,
                                                        int i2)
{
    ConvexHullFace face;
    face.m_index[0] = i0;
    face.m_index[1] = i1;
    face.m_index[2] = i2;

    std::list<ConvexHullFace>::iterator node = m_list.emplace(m_list.end(), face);
    return node;
}

void ConvexHull::CalculateConvexHull3D(ConvexHullAABBTreeNode* vertexTree,
                                       std::vector<ConvexHullVertex>& points,
                                       int count,
                                       double distTol,
                                       int maxVertexCount)
{
    distTol = fabs(distTol) * m_diag;
    std::list<ConvexHullFace>::iterator f0Node = AddFace(0, 1, 2);
    std::list<ConvexHullFace>::iterator f1Node = AddFace(0, 2, 3);
    std::list<ConvexHullFace>::iterator f2Node = AddFace(2, 1, 3);
    std::list<ConvexHullFace>::iterator f3Node = AddFace(1, 0, 3);

    ConvexHullFace& f0 = *f0Node;
    ConvexHullFace& f1 = *f1Node;
    ConvexHullFace& f2 = *f2Node;
    ConvexHullFace& f3 = *f3Node;

    f0.m_twin[0] = f3Node;
    f0.m_twin[1] = f2Node;
    f0.m_twin[2] = f1Node;

    f1.m_twin[0] = f0Node;
    f1.m_twin[1] = f2Node;
    f1.m_twin[2] = f3Node;

    f2.m_twin[0] = f0Node;
    f2.m_twin[1] = f3Node;
    f2.m_twin[2] = f1Node;

    f3.m_twin[0] = f0Node;
    f3.m_twin[1] = f1Node;
    f3.m_twin[2] = f2Node;

    std::list<std::list<ConvexHullFace>::iterator> boundaryFaces;
    boundaryFaces.push_back(f0Node);
    boundaryFaces.push_back(f1Node);
    boundaryFaces.push_back(f2Node);
    boundaryFaces.push_back(f3Node);

    m_points.resize(count);

    count -= 4;
    maxVertexCount -= 4;
    int currentIndex = 4;

    /*
     * Some are iterators into boundaryFaces, others into m_list
     */
    std::vector<std::list<ConvexHullFace>::iterator> stack;
    std::vector<std::list<ConvexHullFace>::iterator> coneList;
    std::vector<std::list<ConvexHullFace>::iterator> deleteList;

    stack.reserve(1024 + count);
    coneList.reserve(1024 + count);
    deleteList.reserve(1024 + count);

    while (boundaryFaces.size() && count && (maxVertexCount > 0))
    {
        // my definition of the optimal convex hull of a given vertex count,
        // is the convex hull formed by a subset of the input vertex that minimizes the volume difference
        // between the perfect hull formed from all input vertex and the hull of the sub set of vertex.
        // When using a priority heap this algorithms will generate the an optimal of a fix vertex count.
        // Since all Newton's tools do not have a limit on the point count of a convex hull, I can use either a stack or a queue.
        // a stack maximize construction speed, a Queue tend to maximize the volume of the generated Hull approaching a perfect Hull.
        // For now we use a queue.
        // For general hulls it does not make a difference if we use a stack, queue, or a priority heap.
        // perfect optimal hull only apply for when build hull of a limited vertex count.
        //
        // Also when building Hulls of a limited vertex count, this function runs in constant time.
        // yes that is correct, it does not makes a difference if you build a N point hull from 100 vertex
        // or from 100000 vertex input array.

        // using a queue (some what slower by better hull when reduced vertex count is desired)
        bool isvalid;
        std::list<ConvexHullFace>::iterator faceNode = boundaryFaces.back();
        ConvexHullFace& face = *faceNode;
        HullPlane planeEquation(face.GetPlaneEquation(m_points, isvalid));

        int index = 0;
        double dist = 0;
        VHACD::Vect3 p;
        if (isvalid)
        {
            index = SupportVertex(&vertexTree,
                                  points,
                                  planeEquation);
            p = points[index];
            dist = planeEquation.Evalue(p);
        }

        if (   isvalid
            && (dist >= distTol)
            && (face.Evalue(m_points, p) < double(0.0)))
        {
            stack.push_back(faceNode);

            deleteList.clear();
            while (stack.size())
            {
                std::list<ConvexHullFace>::iterator node1 = stack.back();
                ConvexHullFace& face1 = *node1;

                stack.pop_back();

                if (!face1.m_mark && (face1.Evalue(m_points, p) < double(0.0)))
                {
                    #ifdef _DEBUG
                    for (const auto node : deleteList)
                    {
                        assert(node != node1);
                    }
                    #endif

                    deleteList.push_back(node1);
                    face1.m_mark = 1;
                    for (std::list<ConvexHullFace>::iterator& twinNode : face1.m_twin)
                    {
                        ConvexHullFace& twinFace = *twinNode;
                        if (!twinFace.m_mark)
                        {
                            stack.push_back(twinNode);
                        }
                    }
                }
            }

            m_points[currentIndex] = points[index];
            points[index].m_mark = 1;

            coneList.clear();
            for (std::list<ConvexHullFace>::iterator node1 : deleteList)
            {
                ConvexHullFace& face1 = *node1;
                assert(face1.m_mark == 1);
                for (std::size_t j0 = 0; j0 < face1.m_twin.size(); ++j0)
                {
                    std::list<ConvexHullFace>::iterator twinNode = face1.m_twin[j0];
                    ConvexHullFace& twinFace = *twinNode;
                    if (!twinFace.m_mark)
                    {
                        std::size_t j1 = (j0 == 2) ? 0 : j0 + 1;
                        std::list<ConvexHullFace>::iterator newNode = AddFace(currentIndex,
                                                                              face1.m_index[j0],
                                                                              face1.m_index[j1]);
                        boundaryFaces.push_front(newNode);
                        ConvexHullFace& newFace = *newNode;

                        newFace.m_twin[1] = twinNode;
                        for (std::size_t k = 0; k < twinFace.m_twin.size(); ++k)
                        {
                            if (twinFace.m_twin[k] == node1)
                            {
                                twinFace.m_twin[k] = newNode;
                            }
                        }
                        coneList.push_back(newNode);
                    }
                }
            }

            for (std::size_t i = 0; i < coneList.size() - 1; ++i)
            {
                std::list<ConvexHullFace>::iterator nodeA = coneList[i];
                ConvexHullFace& faceA = *nodeA;
                assert(faceA.m_mark == 0);
                for (std::size_t j = i + 1; j < coneList.size(); j++)
                {
                    std::list<ConvexHullFace>::iterator nodeB = coneList[j];
                    ConvexHullFace& faceB = *nodeB;
                    assert(faceB.m_mark == 0);
                    if (faceA.m_index[2] == faceB.m_index[1])
                    {
                        faceA.m_twin[2] = nodeB;
                        faceB.m_twin[0] = nodeA;
                        break;
                    }
                }

                for (std::size_t j = i + 1; j < coneList.size(); j++)
                {
                    std::list<ConvexHullFace>::iterator nodeB = coneList[j];
                    ConvexHullFace& faceB = *nodeB;
                    assert(faceB.m_mark == 0);
                    if (faceA.m_index[1] == faceB.m_index[2])
                    {
                        faceA.m_twin[0] = nodeB;
                        faceB.m_twin[2] = nodeA;
                        break;
                    }
                }
            }

            for (std::list<ConvexHullFace>::iterator node : deleteList)
            {
                auto it = std::find(boundaryFaces.begin(),
                                    boundaryFaces.end(),
                                    node);
                if (it != boundaryFaces.end())
                {
                    boundaryFaces.erase(it);
                }
                m_list.erase(node);
            }

            maxVertexCount--;
            currentIndex++;
            count--;
        }
        else
        {
            auto it = std::find(boundaryFaces.begin(),
                                boundaryFaces.end(),
                                faceNode);
            if (it != boundaryFaces.end())
            {
                boundaryFaces.erase(it);
            }
        }
    }
    m_points.resize(currentIndex);
}

//***********************************************************************************************
// End of ConvexHull generation code by Julio Jerez <jerezjulio0@gmail.com>
//***********************************************************************************************

class KdTreeNode;

enum Axes
{
    X_AXIS = 0,
    Y_AXIS = 1,
    Z_AXIS = 2
};

class KdTreeFindNode
{
public:
    KdTreeFindNode() = default;

    KdTreeNode* m_node{ nullptr };
    double m_distance{ 0.0 };
};

class KdTree
{
public:
    KdTree() = default;

    const VHACD::Vertex& GetPosition(uint32_t index) const;

    uint32_t Search(const VHACD::Vect3& pos,
                    double radius,
                    uint32_t maxObjects,
                    KdTreeFindNode* found) const;

    uint32_t Add(const VHACD::Vertex& v);

    KdTreeNode& GetNewNode(uint32_t index);

    uint32_t GetNearest(const VHACD::Vect3& pos,
                        double radius,
                        bool& _found) const; // returns the nearest possible neighbor's index.

    const std::vector<VHACD::Vertex>& GetVertices() const;
    std::vector<VHACD::Vertex>&& TakeVertices();

    uint32_t GetVCount() const;

private:
    KdTreeNode* m_root{ nullptr };
    NodeBundle<KdTreeNode> m_bundle;

    std::vector<VHACD::Vertex> m_vertices;
};

class KdTreeNode
{
public:
    KdTreeNode() = default;
    KdTreeNode(uint32_t index);

    void Add(KdTreeNode& node,
             Axes dim,
             const KdTree& iface);

    uint32_t GetIndex() const;

    void Search(Axes axis,
                const VHACD::Vect3& pos,
                double radius,
                uint32_t& count,
                uint32_t maxObjects,
                KdTreeFindNode* found,
                const KdTree& iface);

private:
    uint32_t m_index = 0;
    KdTreeNode* m_left = nullptr;
    KdTreeNode* m_right = nullptr;
};

const VHACD::Vertex& KdTree::GetPosition(uint32_t index) const
{
    assert(index < m_vertices.size());
    return m_vertices[index];
}

uint32_t KdTree::Search(const VHACD::Vect3& pos,
                        double radius,
                        uint32_t maxObjects,
                        KdTreeFindNode* found) const
{
    if (!m_root)
        return 0;
    uint32_t count = 0;
    m_root->Search(X_AXIS, pos, radius, count, maxObjects, found, *this);
    return count;
}

uint32_t KdTree::Add(const VHACD::Vertex& v)
{
    uint32_t ret = uint32_t(m_vertices.size());
    m_vertices.emplace_back(v);
    KdTreeNode& node = GetNewNode(ret);
    if (m_root)
    {
        m_root->Add(node,
                    X_AXIS,
                    *this);
    }
    else
    {
        m_root = &node;
    }
    return ret;
}

KdTreeNode& KdTree::GetNewNode(uint32_t index)
{
    KdTreeNode& node = m_bundle.GetNextNode();
    node = KdTreeNode(index);
    return node;
}

uint32_t KdTree::GetNearest(const VHACD::Vect3& pos,
                            double radius,
                            bool& _found) const // returns the nearest possible neighbor's index.
{
    uint32_t ret = 0;

    _found = false;
    KdTreeFindNode found;
    uint32_t count = Search(pos, radius, 1, &found);
    if (count)
    {
        KdTreeNode* node = found.m_node;
        ret = node->GetIndex();
        _found = true;
    }
    return ret;
}

const std::vector<VHACD::Vertex>& KdTree::GetVertices() const
{
    return m_vertices;
}

std::vector<VHACD::Vertex>&& KdTree::TakeVertices()
{
    return std::move(m_vertices);
}

uint32_t KdTree::GetVCount() const
{
    return uint32_t(m_vertices.size());
}

KdTreeNode::KdTreeNode(uint32_t index)
    : m_index(index)
{
}

void KdTreeNode::Add(KdTreeNode& node,
                     Axes dim,
                     const KdTree& tree)
{
    Axes axis = X_AXIS;
    uint32_t idx = 0;
    switch (dim)
    {
    case X_AXIS:
        idx = 0;
        axis = Y_AXIS;
        break;
    case Y_AXIS:
        idx = 1;
        axis = Z_AXIS;
        break;
    case Z_AXIS:
        idx = 2;
        axis = X_AXIS;
        break;
    }

    const VHACD::Vertex& nodePosition = tree.GetPosition(node.m_index);
    const VHACD::Vertex& position = tree.GetPosition(m_index);
    if (nodePosition[idx] <= position[idx])
    {
        if (m_left)
            m_left->Add(node, axis, tree);
        else
            m_left = &node;
    }
    else
    {
        if (m_right)
            m_right->Add(node, axis, tree);
        else
            m_right = &node;
    }
}

uint32_t KdTreeNode::GetIndex() const
{
    return m_index;
}

void KdTreeNode::Search(Axes axis,
                        const VHACD::Vect3& pos,
                        double radius,
                        uint32_t& count,
                        uint32_t maxObjects,
                        KdTreeFindNode* found,
                        const KdTree& iface)
{
    const VHACD::Vect3 position = iface.GetPosition(m_index);

    const VHACD::Vect3 d = pos - position;

    KdTreeNode* search1 = 0;
    KdTreeNode* search2 = 0;

    uint32_t idx = 0;
    switch (axis)
    {
    case X_AXIS:
        idx = 0;
        axis = Y_AXIS;
        break;
    case Y_AXIS:
        idx = 1;
        axis = Z_AXIS;
        break;
    case Z_AXIS:
        idx = 2;
        axis = X_AXIS;
        break;
    }

    if (d[idx] <= 0) // JWR  if we are to the left
    {
        search1 = m_left; // JWR  then search to the left
        if (-d[idx] < radius) // JWR  if distance to the right is less than our search radius, continue on the right
                            // as well.
            search2 = m_right;
    }
    else
    {
        search1 = m_right; // JWR  ok, we go down the left tree
        if (d[idx] < radius) // JWR  if the distance from the right is less than our search radius
            search2 = m_left;
    }

    double r2 = radius * radius;
    double m = d.GetNormSquared();

    if (m < r2)
    {
        switch (count)
        {
        case 0:
        {
            found[count].m_node = this;
            found[count].m_distance = m;
            break;
        }
        case 1:
        {
            if (m < found[0].m_distance)
            {
                if (maxObjects == 1)
                {
                    found[0].m_node = this;
                    found[0].m_distance = m;
                }
                else
                {
                    found[1] = found[0];
                    found[0].m_node = this;
                    found[0].m_distance = m;
                }
            }
            else if (maxObjects > 1)
            {
                found[1].m_node = this;
                found[1].m_distance = m;
            }
            break;
        }
        default:
        {
            bool inserted = false;

            for (uint32_t i = 0; i < count; i++)
            {
                if (m < found[i].m_distance) // if this one is closer than a pre-existing one...
                {
                    // insertion sort...
                    uint32_t scan = count;
                    if (scan >= maxObjects)
                        scan = maxObjects - 1;
                    for (uint32_t j = scan; j > i; j--)
                    {
                        found[j] = found[j - 1];
                    }
                    found[i].m_node = this;
                    found[i].m_distance = m;
                    inserted = true;
                    break;
                }
            }

            if (!inserted && count < maxObjects)
            {
                found[count].m_node = this;
                found[count].m_distance = m;
            }
        }
        break;
        }

        count++;

        if (count > maxObjects)
        {
            count = maxObjects;
        }
    }


    if (search1)
        search1->Search(axis, pos, radius, count, maxObjects, found, iface);

    if (search2)
        search2->Search(axis, pos, radius, count, maxObjects, found, iface);
}

class VertexIndex
{
public:
    VertexIndex(double granularity,
                bool snapToGrid);

    VHACD::Vect3 SnapToGrid(VHACD::Vect3 p);

    uint32_t GetIndex(VHACD::Vect3 p,
                      bool& newPos);

    const std::vector<VHACD::Vertex>& GetVertices() const;

    std::vector<VHACD::Vertex>&& TakeVertices();

    uint32_t GetVCount() const;

    bool SaveAsObj(const char* fname,
                   uint32_t tcount,
                   uint32_t* indices)
    {
        bool ret = false;

        FILE* fph = fopen(fname, "wb");
        if (fph)
        {
            ret = true;

            const std::vector<VHACD::Vertex>& v = GetVertices();
            for (uint32_t i = 0; i < v.size(); ++i)
            {
                fprintf(fph, "v %0.9f %0.9f %0.9f\r\n",
                        v[i].mX,
                        v[i].mY,
                        v[i].mZ);
            }

            for (uint32_t i = 0; i < tcount; i++)
            {
                uint32_t i1 = *indices++;
                uint32_t i2 = *indices++;
                uint32_t i3 = *indices++;
                fprintf(fph, "f %d %d %d\r\n",
                        i1 + 1,
                        i2 + 1,
                        i3 + 1);
            }
            fclose(fph);
        }

        return ret;
    }

private:
    bool m_snapToGrid : 1;
    double m_granularity;
    KdTree m_KdTree;
};

VertexIndex::VertexIndex(double granularity,
                         bool snapToGrid)
    : m_snapToGrid(snapToGrid)
    , m_granularity(granularity)
{
}

VHACD::Vect3 VertexIndex::SnapToGrid(VHACD::Vect3 p)
{
    for (int i = 0; i < 3; ++i)
    {
        double m = fmod(p[i], m_granularity);
        p[i] -= m;
    }
    return p;
}

uint32_t VertexIndex::GetIndex(VHACD::Vect3 p,
                               bool& newPos)
{
    uint32_t ret;

    newPos = false;

    if (m_snapToGrid)
    {
        p = SnapToGrid(p);
    }

    bool found;
    ret = m_KdTree.GetNearest(p, m_granularity, found);
    if (!found)
    {
        newPos = true;
        ret = m_KdTree.Add(VHACD::Vertex(p.GetX(), p.GetY(), p.GetZ()));
    }

    return ret;
}

const std::vector<VHACD::Vertex>& VertexIndex::GetVertices() const
{
    return m_KdTree.GetVertices();
}

std::vector<VHACD::Vertex>&& VertexIndex::TakeVertices()
{
    return std::move(m_KdTree.TakeVertices());
}

uint32_t VertexIndex::GetVCount() const
{
    return m_KdTree.GetVCount();
}

/*
 * A wrapper class for 3 10 bit integers packed into a 32 bit integer
 * Layout is [PAD][X][Y][Z]
 * Pad is bits 31-30, X is 29-20, Y is 19-10, and Z is 9-0
 */
class Voxel
{
    /*
     * Specify all of them for consistency
     */
    static constexpr int VoxelBitsZStart =  0;
    static constexpr int VoxelBitsYStart = 10;
    static constexpr int VoxelBitsXStart = 20;
    static constexpr int VoxelBitMask = 0x03FF; // bits 0 through 9 inclusive
public:
    Voxel() = default;

    Voxel(uint32_t index);

    Voxel(uint32_t x,
          uint32_t y,
          uint32_t z);

    bool operator==(const Voxel &v) const;

    VHACD::Vector3<uint32_t> GetVoxel() const;

    uint32_t GetX() const;
    uint32_t GetY() const;
    uint32_t GetZ() const;

    uint32_t GetVoxelAddress() const;

private:
    uint32_t m_voxel{ 0 };
};

Voxel::Voxel(uint32_t index)
    : m_voxel(index)
{
}

Voxel::Voxel(uint32_t x,
             uint32_t y,
             uint32_t z)
    : m_voxel((x << VoxelBitsXStart) | (y << VoxelBitsYStart) | (z << VoxelBitsZStart))
{
    assert(x < 1024 && "Voxel constructed with X outside of range");
    assert(y < 1024 && "Voxel constructed with Y outside of range");
    assert(z < 1024 && "Voxel constructed with Z outside of range");
}

bool Voxel::operator==(const Voxel& v) const
{
    return m_voxel == v.m_voxel;
}

VHACD::Vector3<uint32_t> Voxel::GetVoxel() const
{
    return VHACD::Vector3<uint32_t>(GetX(), GetY(), GetZ());
}

uint32_t Voxel::GetX() const
{
    return (m_voxel >> VoxelBitsXStart) & VoxelBitMask;
}

uint32_t Voxel::GetY() const
{
    return (m_voxel >> VoxelBitsYStart) & VoxelBitMask;
}

uint32_t Voxel::GetZ() const
{
    return (m_voxel >> VoxelBitsZStart) & VoxelBitMask;
}

uint32_t Voxel::GetVoxelAddress() const
{
    return m_voxel;
}

struct SimpleMesh
{
    std::vector<VHACD::Vertex> m_vertices;
    std::vector<VHACD::Triangle> m_indices;
};

/*======================== 0-tests ========================*/
inline bool IntersectRayAABB(const VHACD::Vect3& start,
                             const VHACD::Vect3& dir,
                             const VHACD::BoundsAABB& bounds,
                             double& t)
{
    //! calculate candidate plane on each axis
    bool inside = true;
    VHACD::Vect3 ta(double(-1.0));

    //! use unrolled loops
    for (uint32_t i = 0; i < 3; ++i)
    {
        if (start[i] < bounds.GetMin()[i])
        {
            if (dir[i] != double(0.0))
                ta[i] = (bounds.GetMin()[i] - start[i]) / dir[i];
            inside = false;
        }
        else if (start[i] > bounds.GetMax()[i])
        {
            if (dir[i] != double(0.0))
                ta[i] = (bounds.GetMax()[i] - start[i]) / dir[i];
            inside = false;
        }
    }

    //! if point inside all planes
    if (inside)
    {
        t = double(0.0);
        return true;
    }

    //! we now have t values for each of possible intersection planes
    //! find the maximum to get the intersection point
    uint32_t taxis;
    double tmax = ta.MaxCoeff(taxis);

    if (tmax < double(0.0))
        return false;

    //! check that the intersection point lies on the plane we picked
    //! we don't test the axis of closest intersection for precision reasons

    //! no eps for now
    double eps = double(0.0);

    VHACD::Vect3 hit = start + dir * tmax;

    if ((   hit.GetX() < bounds.GetMin().GetX() - eps
         || hit.GetX() > bounds.GetMax().GetX() + eps)
        && taxis != 0)
        return false;
    if ((   hit.GetY() < bounds.GetMin().GetY() - eps
         || hit.GetY() > bounds.GetMax().GetY() + eps)
        && taxis != 1)
        return false;
    if ((   hit.GetZ() < bounds.GetMin().GetZ() - eps
         || hit.GetZ() > bounds.GetMax().GetZ() + eps)
        && taxis != 2)
        return false;

    //! output results
    t = tmax;

    return true;
}

// Moller and Trumbore's method
inline bool IntersectRayTriTwoSided(const VHACD::Vect3& p,
                                    const VHACD::Vect3& dir,
                                    const VHACD::Vect3& a,
                                    const VHACD::Vect3& b,
                                    const VHACD::Vect3& c,
                                    double& t,
                                    double& u,
                                    double& v,
                                    double& w,
                                    double& sign,
                                    VHACD::Vect3* normal)
{
    VHACD::Vect3 ab = b - a;
    VHACD::Vect3 ac = c - a;
    VHACD::Vect3 n = ab.Cross(ac);

    double d = -dir.Dot(n);
    double ood = double(1.0) / d; // No need to check for division by zero here as infinity arithmetic will save us...
    VHACD::Vect3 ap = p - a;

    t = ap.Dot(n) * ood;
    if (t < double(0.0))
    {
        return false;
    }

    VHACD::Vect3 e = -dir.Cross(ap);
    v = ac.Dot(e) * ood;
    if (v < double(0.0) || v > double(1.0)) // ...here...
    {
        return false;
    }
    w = -ab.Dot(e) * ood;
    if (w < double(0.0) || v + w > double(1.0)) // ...and here
    {
        return false;
    }

    u = double(1.0) - v - w;
    if (normal)
    {
        *normal = n;
    }

    sign = d;

    return true;
}

// RTCD 5.1.5, page 142
inline VHACD::Vect3 ClosestPointOnTriangle(const VHACD::Vect3& a,
                                           const VHACD::Vect3& b,
                                           const VHACD::Vect3& c,
                                           const VHACD::Vect3& p,
                                           double& v,
                                           double& w)
{
    VHACD::Vect3 ab = b - a;
    VHACD::Vect3 ac = c - a;
    VHACD::Vect3 ap = p - a;

    double d1 = ab.Dot(ap);
    double d2 = ac.Dot(ap);
    if (   d1 <= double(0.0)
        && d2 <= double(0.0))
    {
        v = double(0.0);
        w = double(0.0);
        return a;
    }

    VHACD::Vect3 bp = p - b;
    double d3 = ab.Dot(bp);
    double d4 = ac.Dot(bp);
    if (   d3 >= double(0.0)
        && d4 <= d3)
    {
        v = double(1.0);
        w = double(0.0);
        return b;
    }

    double vc = d1 * d4 - d3 * d2;
    if (   vc <= double(0.0)
        && d1 >= double(0.0)
        && d3 <= double(0.0))
    {
        v = d1 / (d1 - d3);
        w = double(0.0);
        return a + v * ab;
    }

    VHACD::Vect3 cp = p - c;
    double d5 = ab.Dot(cp);
    double d6 = ac.Dot(cp);
    if (d6 >= double(0.0) && d5 <= d6)
    {
        v = double(0.0);
        w = double(1.0);
        return c;
    }

    double vb = d5 * d2 - d1 * d6;
    if (   vb <= double(0.0)
        && d2 >= double(0.0)
        && d6 <= double(0.0))
    {
        v = double(0.0);
        w = d2 / (d2 - d6);
        return a + w * ac;
    }

    double va = d3 * d6 - d5 * d4;
    if (   va <= double(0.0)
        && (d4 - d3) >= double(0.0)
        && (d5 - d6) >= double(0.0))
    {
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        v = double(1.0) - w;
        return b + w * (c - b);
    }

    double denom = double(1.0) / (va + vb + vc);
    v = vb * denom;
    w = vc * denom;
    return a + ab * v + ac * w;
}

class AABBTree
{
public:
    AABBTree() = default;
    AABBTree(AABBTree&&) = default;
    AABBTree& operator=(AABBTree&&) = default;

    AABBTree(const std::vector<VHACD::Vertex>& vertices,
             const std::vector<VHACD::Triangle>& indices);

    bool TraceRay(const VHACD::Vect3& start,
                  const VHACD::Vect3& to,
                  double& outT,
                  double& faceSign,
                  VHACD::Vect3& hitLocation) const;

    bool TraceRay(const VHACD::Vect3& start,
                  const VHACD::Vect3& dir,
                  uint32_t& insideCount,
                  uint32_t& outsideCount) const;

    bool TraceRay(const VHACD::Vect3& start,
                  const VHACD::Vect3& dir,
                  double& outT,
                  double& u,
                  double& v,
                  double& w,
                  double& faceSign,
                  uint32_t& faceIndex) const;

    VHACD::Vect3 GetCenter() const;
    VHACD::Vect3 GetMinExtents() const;
    VHACD::Vect3 GetMaxExtents() const;

    bool GetClosestPointWithinDistance(const VHACD::Vect3& point,
                                       double maxDistance,
                                       VHACD::Vect3& closestPoint) const;

private:
    struct Node
    {
        union
        {
            uint32_t m_children;
            uint32_t m_numFaces{ 0 };
        };

        uint32_t* m_faces{ nullptr };
        VHACD::BoundsAABB m_extents;
    };

    struct FaceSorter
    {
        FaceSorter(const std::vector<VHACD::Vertex>& positions,
                   const std::vector<VHACD::Triangle>& indices,
                   uint32_t axis);

        bool operator()(uint32_t lhs, uint32_t rhs) const;

        double GetCentroid(uint32_t face) const;

        const std::vector<VHACD::Vertex>& m_vertices;
        const std::vector<VHACD::Triangle>& m_indices;
        uint32_t m_axis;
    };

    // partition the objects and return the number of objects in the lower partition
    uint32_t PartitionMedian(Node& n,
                             uint32_t* faces,
                             uint32_t numFaces);
    uint32_t PartitionSAH(Node& n,
                          uint32_t* faces,
                          uint32_t numFaces);

    void Build();

    void BuildRecursive(uint32_t nodeIndex,
                        uint32_t* faces,
                        uint32_t numFaces);

    void TraceRecursive(uint32_t nodeIndex,
                        const VHACD::Vect3& start,
                        const VHACD::Vect3& dir,
                        double& outT,
                        double& u,
                        double& v,
                        double& w,
                        double& faceSign,
                        uint32_t& faceIndex) const;


    bool GetClosestPointWithinDistance(const VHACD::Vect3& point,
                                       const double maxDis,
                                       double& dis,
                                       double& v,
                                       double& w,
                                       uint32_t& faceIndex,
                                       VHACD::Vect3& closest) const;

    void GetClosestPointWithinDistanceSqRecursive(uint32_t nodeIndex,
                                                  const VHACD::Vect3& point,
                                                  double& outDisSq,
                                                  double& outV,
                                                  double& outW,
                                                  uint32_t& outFaceIndex,
                                                  VHACD::Vect3& closest) const;

    VHACD::BoundsAABB CalculateFaceBounds(uint32_t* faces,
                                          uint32_t numFaces);

    // track the next free node
    uint32_t m_freeNode;

    const std::vector<VHACD::Vertex>* m_vertices{ nullptr };
    const std::vector<VHACD::Triangle>* m_indices{ nullptr };

    std::vector<uint32_t> m_faces;
    std::vector<Node> m_nodes;
    std::vector<VHACD::BoundsAABB> m_faceBounds;

    // stats
    uint32_t m_treeDepth{ 0 };
    uint32_t m_innerNodes{ 0 };
    uint32_t m_leafNodes{ 0 };

    uint32_t s_depth{ 0 };
};

AABBTree::FaceSorter::FaceSorter(const std::vector<VHACD::Vertex>& positions,
                                 const std::vector<VHACD::Triangle>& indices,
                                 uint32_t axis)
    : m_vertices(positions)
    , m_indices(indices)
    , m_axis(axis)
{
}

inline bool AABBTree::FaceSorter::operator()(uint32_t lhs,
                                             uint32_t rhs) const
{
    double a = GetCentroid(lhs);
    double b = GetCentroid(rhs);

    if (a == b)
    {
        return lhs < rhs;
    }
    else
    {
        return a < b;
    }
}

inline double AABBTree::FaceSorter::GetCentroid(uint32_t face) const
{
    const VHACD::Vect3& a = m_vertices[m_indices[face].mI0];
    const VHACD::Vect3& b = m_vertices[m_indices[face].mI1];
    const VHACD::Vect3& c = m_vertices[m_indices[face].mI2];

    return (a[m_axis] + b[m_axis] + c[m_axis]) / double(3.0);
}

AABBTree::AABBTree(const std::vector<VHACD::Vertex>& vertices,
                   const std::vector<VHACD::Triangle>& indices)
    : m_vertices(&vertices)
    , m_indices(&indices)
{
    Build();
}

bool AABBTree::TraceRay(const VHACD::Vect3& start,
                        const VHACD::Vect3& to,
                        double& outT,
                        double& faceSign,
                        VHACD::Vect3& hitLocation) const
{
    VHACD::Vect3 dir = to - start;
    double distance = dir.Normalize();
    double u, v, w;
    uint32_t faceIndex;
    bool hit = TraceRay(start,
                        dir,
                        outT,
                        u,
                        v,
                        w,
                        faceSign,
                        faceIndex);
    if (hit)
    {
        hitLocation = start + dir * outT;
    }

    if (hit && outT > distance)
    {
        hit = false;
    }
    return hit;
}

bool AABBTree::TraceRay(const VHACD::Vect3& start,
                        const VHACD::Vect3& dir,
                        uint32_t& insideCount,
                        uint32_t& outsideCount) const
{
    double outT, u, v, w, faceSign;
    uint32_t faceIndex;
    bool hit = TraceRay(start,
                        dir,
                        outT,
                        u,
                        v,
                        w,
                        faceSign,
                        faceIndex);
    if (hit)
    {
        if (faceSign >= 0)
        {
            insideCount++;
        }
        else
        {
            outsideCount++;
        }
    }
    return hit;
}

bool AABBTree::TraceRay(const VHACD::Vect3& start,
                        const VHACD::Vect3& dir,
                        double& outT,
                        double& u,
                        double& v,
                        double& w,
                        double& faceSign,
                        uint32_t& faceIndex) const
{
    outT = FLT_MAX;
    TraceRecursive(0,
                   start,
                   dir,
                   outT,
                   u,
                   v,
                   w,
                   faceSign,
                   faceIndex);
    return (outT != FLT_MAX);
}

VHACD::Vect3 AABBTree::GetCenter() const
{
    return m_nodes[0].m_extents.GetCenter();
}

VHACD::Vect3 AABBTree::GetMinExtents() const
{
    return m_nodes[0].m_extents.GetMin();
}

VHACD::Vect3 AABBTree::GetMaxExtents() const
{
    return m_nodes[0].m_extents.GetMax();
}

bool AABBTree::GetClosestPointWithinDistance(const VHACD::Vect3& point,
                                             double maxDistance,
                                             VHACD::Vect3& closestPoint) const
{
    double dis, v, w;
    uint32_t faceIndex;
    bool hit = GetClosestPointWithinDistance(point,
                                             maxDistance,
                                             dis,
                                             v,
                                             w,
                                             faceIndex,
                                             closestPoint);
    return hit;
}

// partition faces around the median face
uint32_t AABBTree::PartitionMedian(Node& n,
                                   uint32_t* faces,
                                   uint32_t numFaces)
{
    FaceSorter predicate(*m_vertices,
                         *m_indices,
                         n.m_extents.GetSize().LongestAxis());
    std::nth_element(faces,
                     faces + numFaces / 2,
                     faces + numFaces,
                     predicate);

    return numFaces / 2;
}

// partition faces based on the surface area heuristic
uint32_t AABBTree::PartitionSAH(Node&,
                                uint32_t* faces,
                                uint32_t numFaces)
{
    uint32_t bestAxis = 0;
    uint32_t bestIndex = 0;
    double bestCost = FLT_MAX;

    for (uint32_t a = 0; a < 3; ++a)
    {
        // sort faces by centroids
        FaceSorter predicate(*m_vertices,
                             *m_indices,
                             a);
        std::sort(faces,
                  faces + numFaces,
                  predicate);

        // two passes over data to calculate upper and lower bounds
        std::vector<double> cumulativeLower(numFaces);
        std::vector<double> cumulativeUpper(numFaces);

        VHACD::BoundsAABB lower;
        VHACD::BoundsAABB upper;

        for (uint32_t i = 0; i < numFaces; ++i)
        {
            lower.Union(m_faceBounds[faces[i]]);
            upper.Union(m_faceBounds[faces[numFaces - i - 1]]);

            cumulativeLower[i] = lower.SurfaceArea();
            cumulativeUpper[numFaces - i - 1] = upper.SurfaceArea();
        }

        double invTotalSA = double(1.0) / cumulativeUpper[0];

        // test all split positions
        for (uint32_t i = 0; i < numFaces - 1; ++i)
        {
            double pBelow = cumulativeLower[i] * invTotalSA;
            double pAbove = cumulativeUpper[i] * invTotalSA;

            double cost = double(0.125) + (pBelow * i + pAbove * (numFaces - i));
            if (cost <= bestCost)
            {
                bestCost = cost;
                bestIndex = i;
                bestAxis = a;
            }
        }
    }

    // re-sort by best axis
    FaceSorter predicate(*m_vertices,
                         *m_indices,
                         bestAxis);
    std::sort(faces,
              faces + numFaces,
              predicate);

    return bestIndex + 1;
}

void AABBTree::Build()
{
    const uint32_t numFaces = uint32_t(m_indices->size());

    // build initial list of faces
    m_faces.reserve(numFaces);

    // calculate bounds of each face and store
    m_faceBounds.reserve(numFaces);

    std::vector<VHACD::BoundsAABB> stack;
    for (uint32_t i = 0; i < numFaces; ++i)
    {
        VHACD::BoundsAABB top = CalculateFaceBounds(&i,
                                                    1);

        m_faces.push_back(i);
        m_faceBounds.push_back(top);
    }

    m_nodes.reserve(uint32_t(numFaces * double(1.5)));

    // allocate space for all the nodes
    m_freeNode = 1;

    // start building
    BuildRecursive(0,
                   m_faces.data(),
                   numFaces);

    assert(s_depth == 0);
}

void AABBTree::BuildRecursive(uint32_t nodeIndex,
                              uint32_t* faces,
                              uint32_t numFaces)
{
    const uint32_t kMaxFacesPerLeaf = 6;

    // if we've run out of nodes allocate some more
    if (nodeIndex >= m_nodes.size())
    {
        uint32_t s = std::max(uint32_t(double(1.5) * m_nodes.size()), 512U);
        m_nodes.resize(s);
    }

    // a reference to the current node, need to be careful here as this reference may become invalid if array is resized
    Node& n = m_nodes[nodeIndex];

    // track max tree depth
    ++s_depth;
    m_treeDepth = std::max(m_treeDepth, s_depth);

    n.m_extents = CalculateFaceBounds(faces,
                                      numFaces);

    // calculate bounds of faces and add node
    if (numFaces <= kMaxFacesPerLeaf)
    {
        n.m_faces = faces;
        n.m_numFaces = numFaces;

        ++m_leafNodes;
    }
    else
    {
        ++m_innerNodes;

        // face counts for each branch
        const uint32_t leftCount = PartitionMedian(n, faces, numFaces);
        // const uint32_t leftCount = PartitionSAH(n, faces, numFaces);
        const uint32_t rightCount = numFaces - leftCount;

        // alloc 2 nodes
        m_nodes[nodeIndex].m_children = m_freeNode;

        // allocate two nodes
        m_freeNode += 2;

        // split faces in half and build each side recursively
        BuildRecursive(m_nodes[nodeIndex].m_children + 0, faces, leftCount);
        BuildRecursive(m_nodes[nodeIndex].m_children + 1, faces + leftCount, rightCount);
    }

    --s_depth;
}

void AABBTree::TraceRecursive(uint32_t nodeIndex,
                              const VHACD::Vect3& start,
                              const VHACD::Vect3& dir,
                              double& outT,
                              double& outU,
                              double& outV,
                              double& outW,
                              double& faceSign,
                              uint32_t& faceIndex) const
{
    const Node& node = m_nodes[nodeIndex];

    if (node.m_faces == NULL)
    {
        // find closest node
        const Node& leftChild = m_nodes[node.m_children + 0];
        const Node& rightChild = m_nodes[node.m_children + 1];

        double dist[2] = { FLT_MAX, FLT_MAX };

        IntersectRayAABB(start,
                         dir,
                         leftChild.m_extents,
                         dist[0]);
        IntersectRayAABB(start,
                         dir,
                         rightChild.m_extents,
                         dist[1]);

        uint32_t closest = 0;
        uint32_t furthest = 1;

        if (dist[1] < dist[0])
        {
            closest = 1;
            furthest = 0;
        }

        if (dist[closest] < outT)
        {
            TraceRecursive(node.m_children + closest,
                           start,
                           dir,
                           outT,
                           outU,
                           outV,
                           outW,
                           faceSign,
                           faceIndex);
        }

        if (dist[furthest] < outT)
        {
            TraceRecursive(node.m_children + furthest,
                           start,
                           dir,
                           outT,
                           outU,
                           outV,
                           outW,
                           faceSign,
                           faceIndex);
        }
    }
    else
    {
        double t, u, v, w, s;

        for (uint32_t i = 0; i < node.m_numFaces; ++i)
        {
            uint32_t indexStart = node.m_faces[i];

            const VHACD::Vect3& a = (*m_vertices)[(*m_indices)[indexStart].mI0];
            const VHACD::Vect3& b = (*m_vertices)[(*m_indices)[indexStart].mI1];
            const VHACD::Vect3& c = (*m_vertices)[(*m_indices)[indexStart].mI2];
            if (IntersectRayTriTwoSided(start, dir, a, b, c, t, u, v, w, s, NULL))
            {
                if (t < outT)
                {
                    outT = t;
                    outU = u;
                    outV = v;
                    outW = w;
                    faceSign = s;
                    faceIndex = node.m_faces[i];
                }
            }
        }
    }
}

bool AABBTree::GetClosestPointWithinDistance(const VHACD::Vect3& point,
                                             const double maxDis,
                                             double& dis,
                                             double& v,
                                             double& w,
                                             uint32_t& faceIndex,
                                             VHACD::Vect3& closest) const
{
    dis = maxDis;
    faceIndex = uint32_t(~0);
    double disSq = dis * dis;

    GetClosestPointWithinDistanceSqRecursive(0,
                                             point,
                                             disSq,
                                             v,
                                             w,
                                             faceIndex,
                                             closest);
    dis = sqrt(disSq);

    return (faceIndex < (~(static_cast<unsigned int>(0))));
}

void AABBTree::GetClosestPointWithinDistanceSqRecursive(uint32_t nodeIndex,
                                                        const VHACD::Vect3& point,
                                                        double& outDisSq,
                                                        double& outV,
                                                        double& outW,
                                                        uint32_t& outFaceIndex,
                                                        VHACD::Vect3& closestPoint) const
{
    const Node& node = m_nodes[nodeIndex];

    if (node.m_faces == nullptr)
    {
        // find closest node
        const Node& leftChild = m_nodes[node.m_children + 0];
        const Node& rightChild = m_nodes[node.m_children + 1];

        // double dist[2] = { FLT_MAX, FLT_MAX };
        VHACD::Vect3 lp = leftChild.m_extents.ClosestPoint(point);
        VHACD::Vect3 rp = rightChild.m_extents.ClosestPoint(point);


        uint32_t closest = 0;
        uint32_t furthest = 1;
        double dcSq = (point - lp).GetNormSquared();
        double dfSq = (point - rp).GetNormSquared();
        if (dfSq < dcSq)
        {
            closest = 1;
            furthest = 0;
            std::swap(dfSq, dcSq);
        }

        if (dcSq < outDisSq)
        {
            GetClosestPointWithinDistanceSqRecursive(node.m_children + closest,
                                                     point,
                                                     outDisSq,
                                                     outV,
                                                     outW,
                                                     outFaceIndex,
                                                     closestPoint);
        }

        if (dfSq < outDisSq)
        {
            GetClosestPointWithinDistanceSqRecursive(node.m_children + furthest,
                                                     point,
                                                     outDisSq,
                                                     outV,
                                                     outW,
                                                     outFaceIndex,
                                                     closestPoint);
        }
    }
    else
    {

        double v, w;
        for (uint32_t i = 0; i < node.m_numFaces; ++i)
        {
            uint32_t indexStart = node.m_faces[i];

            const VHACD::Vect3& a = (*m_vertices)[(*m_indices)[indexStart].mI0];
            const VHACD::Vect3& b = (*m_vertices)[(*m_indices)[indexStart].mI1];
            const VHACD::Vect3& c = (*m_vertices)[(*m_indices)[indexStart].mI2];

            VHACD::Vect3 cp = ClosestPointOnTriangle(a, b, c, point, v, w);
            double disSq = (cp - point).GetNormSquared();

            if (disSq < outDisSq)
            {
                closestPoint = cp;
                outDisSq = disSq;
                outV = v;
                outW = w;
                outFaceIndex = node.m_faces[i];
            }
        }
    }
}

VHACD::BoundsAABB AABBTree::CalculateFaceBounds(uint32_t* faces,
                                                uint32_t numFaces)
{
    VHACD::Vect3 minExtents( FLT_MAX);
    VHACD::Vect3 maxExtents(-FLT_MAX);

    // calculate face bounds
    for (uint32_t i = 0; i < numFaces; ++i)
    {
        VHACD::Vect3 a = (*m_vertices)[(*m_indices)[faces[i]].mI0];
        VHACD::Vect3 b = (*m_vertices)[(*m_indices)[faces[i]].mI1];
        VHACD::Vect3 c = (*m_vertices)[(*m_indices)[faces[i]].mI2];

        minExtents = a.CWiseMin(minExtents);
        maxExtents = a.CWiseMax(maxExtents);

        minExtents = b.CWiseMin(minExtents);
        maxExtents = b.CWiseMax(maxExtents);

        minExtents = c.CWiseMin(minExtents);
        maxExtents = c.CWiseMax(maxExtents);
    }

    return VHACD::BoundsAABB(minExtents,
                             maxExtents);
}

enum class VoxelValue : uint8_t
{
    PRIMITIVE_UNDEFINED = 0,
    PRIMITIVE_OUTSIDE_SURFACE_TOWALK = 1,
    PRIMITIVE_OUTSIDE_SURFACE = 2,
    PRIMITIVE_INSIDE_SURFACE = 3,
    PRIMITIVE_ON_SURFACE = 4
};

class Volume
{
public:
    void Voxelize(const std::vector<VHACD::Vertex>& points,
                  const std::vector<VHACD::Triangle>& triangles,
                  const size_t dim,
                  FillMode fillMode,
                  const AABBTree& aabbTree);

    void RaycastFill(const AABBTree& aabbTree);

    void SetVoxel(const size_t i,
                  const size_t j,
                  const size_t k,
                  VoxelValue value);

    VoxelValue& GetVoxel(const size_t i,
                         const size_t j,
                         const size_t k);

    const VoxelValue& GetVoxel(const size_t i,
                               const size_t j,
                               const size_t k) const;

    const std::vector<Voxel>& GetSurfaceVoxels() const;
    const std::vector<Voxel>& GetInteriorVoxels() const;

    double GetScale() const;
    const VHACD::BoundsAABB& GetBounds() const;
    const VHACD::Vector3<uint32_t>& GetDimensions() const;

    VHACD::BoundsAABB m_bounds;
    double m_scale{ 1.0 };
    VHACD::Vector3<uint32_t> m_dim{ 0 };
    size_t m_numVoxelsOnSurface{ 0 };
    size_t m_numVoxelsInsideSurface{ 0 };
    size_t m_numVoxelsOutsideSurface{ 0 };
    std::vector<VoxelValue> m_data;
private:

    void MarkOutsideSurface(const size_t i0,
                            const size_t j0,
                            const size_t k0,
                            const size_t i1,
                            const size_t j1,
                            const size_t k1);
    void FillOutsideSurface();

    void FillInsideSurface();

    std::vector<VHACD::Voxel> m_surfaceVoxels;
    std::vector<VHACD::Voxel> m_interiorVoxels;
};

bool PlaneBoxOverlap(const VHACD::Vect3& normal,
                     const VHACD::Vect3& vert,
                     const VHACD::Vect3& maxbox)
{
    int32_t q;
    VHACD::Vect3 vmin;
    VHACD::Vect3 vmax;
    double v;
    for (q = 0; q < 3; q++)
    {
        v = vert[q];
        if (normal[q] > double(0.0))
        {
            vmin[q] = -maxbox[q] - v;
            vmax[q] =  maxbox[q] - v;
        }
        else
        {
            vmin[q] =  maxbox[q] - v;
            vmax[q] = -maxbox[q] - v;
        }
    }
    if (normal.Dot(vmin) > double(0.0))
        return false;
    if (normal.Dot(vmax) >= double(0.0))
        return true;
    return false;
}

bool AxisTest(double  a, double  b, double fa, double fb,
              double v0, double v1, double v2, double v3,
              double boxHalfSize1,  double boxHalfSize2)
{
    double p0 = a * v0 + b * v1;
    double p1 = a * v2 + b * v3;

    double min = std::min(p0, p1);
    double max = std::max(p0, p1);

    double rad = fa * boxHalfSize1 + fb * boxHalfSize2;
    if (min > rad || max < -rad)
    {
        return false;
    }

    return true;
}

bool TriBoxOverlap(const VHACD::Vect3& boxCenter,
                   const VHACD::Vect3& boxHalfSize,
                   const VHACD::Vect3& triVer0,
                   const VHACD::Vect3& triVer1,
                   const VHACD::Vect3& triVer2)
{
    /*    use separating axis theorem to test overlap between triangle and box */
    /*    need to test for overlap in these directions: */
    /*    1) the {x,y,z}-directions (actually, since we use the AABB of the triangle */
    /*       we do not even need to test these) */
    /*    2) normal of the triangle */
    /*    3) crossproduct(edge from tri, {x,y,z}-direction) */
    /*       this gives 3x3=9 more tests */

    VHACD::Vect3 v0 = triVer0 - boxCenter;
    VHACD::Vect3 v1 = triVer1 - boxCenter;
    VHACD::Vect3 v2 = triVer2 - boxCenter;
    VHACD::Vect3 e0 = v1 - v0;
    VHACD::Vect3 e1 = v2 - v1;
    VHACD::Vect3 e2 = v0 - v2;

    /* This is the fastest branch on Sun */
    /* move everything so that the boxcenter is in (0,0,0) */

    /* Bullet 3:  */
    /*  test the 9 tests first (this was faster) */
    double fex = fabs(e0[0]);
    double fey = fabs(e0[1]);
    double fez = fabs(e0[2]);

    /*
     * These should use Get*() instead of subscript for consistency, but the function calls are long enough already
     */
    if (!AxisTest( e0[2], -e0[1], fez, fey, v0[1], v0[2], v2[1], v2[2], boxHalfSize[1], boxHalfSize[2])) return 0; // X01
    if (!AxisTest(-e0[2],  e0[0], fez, fex, v0[0], v0[2], v2[0], v2[2], boxHalfSize[0], boxHalfSize[2])) return 0; // Y02
    if (!AxisTest( e0[1], -e0[0], fey, fex, v1[0], v1[1], v2[0], v2[1], boxHalfSize[0], boxHalfSize[1])) return 0; // Z12

    fex = fabs(e1[0]);
    fey = fabs(e1[1]);
    fez = fabs(e1[2]);

    if (!AxisTest( e1[2], -e1[1], fez, fey, v0[1], v0[2], v2[1], v2[2], boxHalfSize[1], boxHalfSize[2])) return 0; // X01
    if (!AxisTest(-e1[2],  e1[0], fez, fex, v0[0], v0[2], v2[0], v2[2], boxHalfSize[0], boxHalfSize[2])) return 0; // Y02
    if (!AxisTest( e1[1], -e1[0], fey, fex, v0[0], v0[1], v1[0], v1[1], boxHalfSize[0], boxHalfSize[2])) return 0; // Z0

    fex = fabs(e2[0]);
    fey = fabs(e2[1]);
    fez = fabs(e2[2]);

    if (!AxisTest( e2[2], -e2[1], fez, fey, v0[1], v0[2], v1[1], v1[2], boxHalfSize[1], boxHalfSize[2])) return 0; // X2
    if (!AxisTest(-e2[2],  e2[0], fez, fex, v0[0], v0[2], v1[0], v1[2], boxHalfSize[0], boxHalfSize[2])) return 0; // Y1
    if (!AxisTest( e2[1], -e2[0], fey, fex, v1[0], v1[1], v2[0], v2[1], boxHalfSize[0], boxHalfSize[1])) return 0; // Z12

    /* Bullet 1: */
    /*  first test overlap in the {x,y,z}-directions */
    /*  find min, max of the triangle each direction, and test for overlap in */
    /*  that direction -- this is equivalent to testing a minimal AABB around */
    /*  the triangle against the AABB */

    /* test in 0-direction */
    double min = std::min({v0.GetX(), v1.GetX(), v2.GetX()});
    double max = std::max({v0.GetX(), v1.GetX(), v2.GetX()});
    if (min > boxHalfSize[0] || max < -boxHalfSize[0])
        return false;

    /* test in 1-direction */
    min = std::min({v0.GetY(), v1.GetY(), v2.GetY()});
    max = std::max({v0.GetY(), v1.GetY(), v2.GetY()});
    if (min > boxHalfSize[1] || max < -boxHalfSize[1])
        return false;

    /* test in getZ-direction */
    min = std::min({v0.GetZ(), v1.GetZ(), v2.GetZ()});
    max = std::max({v0.GetZ(), v1.GetZ(), v2.GetZ()});
    if (min > boxHalfSize[2] || max < -boxHalfSize[2])
        return false;

    /* Bullet 2: */
    /*  test if the box intersects the plane of the triangle */
    /*  compute plane equation of triangle: normal*x+d=0 */
    VHACD::Vect3 normal = e0.Cross(e1);

    if (!PlaneBoxOverlap(normal, v0, boxHalfSize))
        return false;
    return true; /* box and triangle overlaps */
}

void Volume::Voxelize(const std::vector<VHACD::Vertex>& points,
                      const std::vector<VHACD::Triangle>& indices,
                      const size_t dimensions,
                      FillMode fillMode,
                      const AABBTree& aabbTree)
{
    double a = std::pow(dimensions, 0.33);
    size_t dim = a * double(1.5);
    dim = std::max(dim, size_t(32));

    if (points.size() == 0)
    {
        return;
    }

    m_bounds = BoundsAABB(points);

    VHACD::Vect3 d = m_bounds.GetSize();
    double r;
    // Equal comparison is important here to avoid taking the last branch when d[0] == d[1] with d[2] being the smallest
    // dimension. That would lead to dimensions in i and j to be a lot bigger than expected and make the amount of
    // voxels in the volume totally unmanageable.
    if (d[0] >= d[1] && d[0] >= d[2])
    {
        r = d[0];
        m_dim[0] = uint32_t(dim);
        m_dim[1] = uint32_t(2 + static_cast<size_t>(dim * d[1] / d[0]));
        m_dim[2] = uint32_t(2 + static_cast<size_t>(dim * d[2] / d[0]));
    }
    else if (d[1] >= d[0] && d[1] >= d[2])
    {
        r = d[1];
        m_dim[1] = uint32_t(dim);
        m_dim[0] = uint32_t(2 + static_cast<size_t>(dim * d[0] / d[1]));
        m_dim[2] = uint32_t(2 + static_cast<size_t>(dim * d[2] / d[1]));
    }
    else
    {
        r = d[2];
        m_dim[2] = uint32_t(dim);
        m_dim[0] = uint32_t(2 + static_cast<size_t>(dim * d[0] / d[2]));
        m_dim[1] = uint32_t(2 + static_cast<size_t>(dim * d[1] / d[2]));
    }

    m_scale = r / (dim - 1);
    double invScale = (dim - 1) / r;

    m_data = std::vector<VoxelValue>(m_dim[0] * m_dim[1] * m_dim[2],
                                     VoxelValue::PRIMITIVE_UNDEFINED);
    m_numVoxelsOnSurface = 0;
    m_numVoxelsInsideSurface = 0;
    m_numVoxelsOutsideSurface = 0;

    VHACD::Vect3 p[3];
    VHACD::Vect3 boxcenter;
    VHACD::Vect3 pt;
    const VHACD::Vect3 boxhalfsize(double(0.5));
    for (size_t t = 0; t < indices.size(); ++t)
    {
        size_t i0, j0, k0;
        size_t i1, j1, k1;
        VHACD::Vector3<uint32_t> tri = indices[t];
        for (int32_t c = 0; c < 3; ++c)
        {
            pt = points[tri[c]];

            p[c] = (pt - m_bounds.GetMin()) * invScale;

            size_t i = static_cast<size_t>(p[c][0] + double(0.5));
            size_t j = static_cast<size_t>(p[c][1] + double(0.5));
            size_t k = static_cast<size_t>(p[c][2] + double(0.5));

            assert(i < m_dim[0] && j < m_dim[1] && k < m_dim[2]);

            if (c == 0)
            {
                i0 = i1 = i;
                j0 = j1 = j;
                k0 = k1 = k;
            }
            else
            {
                i0 = std::min(i0, i);
                j0 = std::min(j0, j);
                k0 = std::min(k0, k);

                i1 = std::max(i1, i);
                j1 = std::max(j1, j);
                k1 = std::max(k1, k);
            }
        }
        if (i0 > 0)
            --i0;
        if (j0 > 0)
            --j0;
        if (k0 > 0)
            --k0;
        if (i1 < m_dim[0])
            ++i1;
        if (j1 < m_dim[1])
            ++j1;
        if (k1 < m_dim[2])
            ++k1;
        for (size_t i_id = i0; i_id < i1; ++i_id)
        {
            boxcenter[0] = uint32_t(i_id);
            for (size_t j_id = j0; j_id < j1; ++j_id)
            {
                boxcenter[1] = uint32_t(j_id);
                for (size_t k_id = k0; k_id < k1; ++k_id)
                {
                    boxcenter[2] = uint32_t(k_id);
                    bool res = TriBoxOverlap(boxcenter,
                                             boxhalfsize,
                                             p[0],
                                             p[1],
                                             p[2]);
                    VoxelValue& value = GetVoxel(i_id,
                                                 j_id,
                                                 k_id);
                    if (   res
                        && value == VoxelValue::PRIMITIVE_UNDEFINED)
                    {
                        value = VoxelValue::PRIMITIVE_ON_SURFACE;
                        ++m_numVoxelsOnSurface;
                        m_surfaceVoxels.emplace_back(uint32_t(i_id),
                                                     uint32_t(j_id),
                                                     uint32_t(k_id));
                    }
                }
            }
        }
    }

    if (fillMode == FillMode::SURFACE_ONLY)
    {
        const size_t i0_local = m_dim[0];
        const size_t j0_local = m_dim[1];
        const size_t k0_local = m_dim[2];
        for (size_t i_id = 0; i_id < i0_local; ++i_id)
        {
            for (size_t j_id = 0; j_id < j0_local; ++j_id)
            {
                for (size_t k_id = 0; k_id < k0_local; ++k_id)
                {
                    const VoxelValue& voxel = GetVoxel(i_id,
                                                       j_id,
                                                       k_id);
                    if (voxel != VoxelValue::PRIMITIVE_ON_SURFACE)
                    {
                        SetVoxel(i_id,
                                 j_id,
                                 k_id,
                                 VoxelValue::PRIMITIVE_OUTSIDE_SURFACE);
                    }
                }
            }
        }
    }
    else if (fillMode == FillMode::FLOOD_FILL)
    {
        /*
         * Marking the outside edges of the voxel cube to be outside surfaces to walk
         */
        MarkOutsideSurface(0,            0,            0,            m_dim[0], m_dim[1], 1);
        MarkOutsideSurface(0,            0,            m_dim[2] - 1, m_dim[0], m_dim[1], m_dim[2]);
        MarkOutsideSurface(0,            0,            0,            m_dim[0], 1,        m_dim[2]);
        MarkOutsideSurface(0,            m_dim[1] - 1, 0,            m_dim[0], m_dim[1], m_dim[2]);
        MarkOutsideSurface(0,            0,            0,            1,        m_dim[1], m_dim[2]);
        MarkOutsideSurface(m_dim[0] - 1, 0,            0,            m_dim[0], m_dim[1], m_dim[2]);
        FillOutsideSurface();
        FillInsideSurface();
    }
    else if (fillMode == FillMode::RAYCAST_FILL)
    {
        RaycastFill(aabbTree);
    }
}

void Volume::RaycastFill(const AABBTree& aabbTree)
{
    const uint32_t i0 = m_dim[0];
    const uint32_t j0 = m_dim[1];
    const uint32_t k0 = m_dim[2];

    size_t maxSize = i0 * j0 * k0;

    std::vector<Voxel> temp;
    temp.reserve(maxSize);
    uint32_t count{ 0 };
    m_numVoxelsInsideSurface = 0;
    for (uint32_t i = 0; i < i0; ++i)
    {
        for (uint32_t j = 0; j < j0; ++j)
        {
            for (uint32_t k = 0; k < k0; ++k)
            {
                VoxelValue& voxel = GetVoxel(i, j, k);
                if (voxel != VoxelValue::PRIMITIVE_ON_SURFACE)
                {
                    VHACD::Vect3 start = VHACD::Vect3(i, j, k) * m_scale + m_bounds.GetMin();

                    uint32_t insideCount = 0;
                    uint32_t outsideCount = 0;

                    VHACD::Vect3 directions[6] = {
                        VHACD::Vect3( 1,  0,  0),
                        VHACD::Vect3(-1,  0,  0), // this was 1, 0, 0 in the original code, but looks wrong
                        VHACD::Vect3( 0,  1,  0),
                        VHACD::Vect3( 0, -1,  0),
                        VHACD::Vect3( 0,  0,  1),
                        VHACD::Vect3( 0,  0, -1)
                    };

                    for (uint32_t r = 0; r < 6; r++)
                    {
                        aabbTree.TraceRay(start,
                                          directions[r],
                                          insideCount,
                                          outsideCount);
                        // Early out if we hit the outside of the mesh
                        if (outsideCount)
                        {
                            break;
                        }
                        // Early out if we accumulated 3 inside hits
                        if (insideCount >= 3)
                        {
                            break;
                        }
                    }

                    if (outsideCount == 0 && insideCount >= 3)
                    {
                        voxel = VoxelValue::PRIMITIVE_INSIDE_SURFACE;
                        temp.emplace_back(i, j, k);
                        count++;
                        m_numVoxelsInsideSurface++;
                    }
                    else
                    {
                        voxel = VoxelValue::PRIMITIVE_OUTSIDE_SURFACE;
                    }
                }
            }
        }
    }

    if (count)
    {
        m_interiorVoxels = std::move(temp);
    }
}

void Volume::SetVoxel(const size_t i,
                      const size_t j,
                      const size_t k,
                      VoxelValue value)
{
    assert(i < m_dim[0]);
    assert(j < m_dim[1]);
    assert(k < m_dim[2]);

    m_data[k + j * m_dim[2] + i * m_dim[1] * m_dim[2]] = value;
}

VoxelValue& Volume::GetVoxel(const size_t i,
                             const size_t j,
                             const size_t k)
{
    assert(i < m_dim[0]);
    assert(j < m_dim[1]);
    assert(k < m_dim[2]);
    return m_data[k + j * m_dim[2] + i * m_dim[1] * m_dim[2]];
}

const VoxelValue& Volume::GetVoxel(const size_t i,
                                   const size_t j,
                                   const size_t k) const
{
    assert(i < m_dim[0]);
    assert(j < m_dim[1]);
    assert(k < m_dim[2]);
    return m_data[k + j * m_dim[2] + i * m_dim[1] * m_dim[2]];
}

const std::vector<Voxel>& Volume::GetSurfaceVoxels() const
{
    return m_surfaceVoxels;
}

const std::vector<Voxel>& Volume::GetInteriorVoxels() const
{
    return m_interiorVoxels;
}

double Volume::GetScale() const
{
    return m_scale;
}

const VHACD::BoundsAABB& Volume::GetBounds() const
{
    return m_bounds;
}

const VHACD::Vector3<uint32_t>& Volume::GetDimensions() const
{
    return m_dim;
}

void Volume::MarkOutsideSurface(const size_t i0,
                                const size_t j0,
                                const size_t k0,
                                const size_t i1,
                                const size_t j1,
                                const size_t k1)
{
    for (size_t i = i0; i < i1; ++i)
    {
        for (size_t j = j0; j < j1; ++j)
        {
            for (size_t k = k0; k < k1; ++k)
            {
                VoxelValue& v = GetVoxel(i, j, k);
                if (v == VoxelValue::PRIMITIVE_UNDEFINED)
                {
                    v = VoxelValue::PRIMITIVE_OUTSIDE_SURFACE_TOWALK;
                }
            }
        }
    }
}

inline void WalkForward(int64_t start,
                        int64_t end,
                        VoxelValue* ptr,
                        int64_t stride,
                        int64_t maxDistance)
{
    for (int64_t i = start, count = 0;
         count < maxDistance && i < end && *ptr == VoxelValue::PRIMITIVE_UNDEFINED;
         ++i, ptr += stride, ++count)
    {
        *ptr = VoxelValue::PRIMITIVE_OUTSIDE_SURFACE_TOWALK;
    }
}

inline void WalkBackward(int64_t start,
                         int64_t end,
                         VoxelValue* ptr,
                         int64_t stride,
                         int64_t maxDistance)
{
    for (int64_t i = start, count = 0;
         count < maxDistance && i >= end && *ptr == VoxelValue::PRIMITIVE_UNDEFINED;
         --i, ptr -= stride, ++count)
    {
        *ptr = VoxelValue::PRIMITIVE_OUTSIDE_SURFACE_TOWALK;
    }
}

void Volume::FillOutsideSurface()
{
    size_t voxelsWalked = 0;
    const int64_t i0 = m_dim[0];
    const int64_t j0 = m_dim[1];
    const int64_t k0 = m_dim[2];

    // Avoid striding too far in each direction to stay in L1 cache as much as possible.
    // The cache size required for the walk is roughly (4 * walkDistance * 64) since
    // the k direction doesn't count as it's walking byte per byte directly in a cache lines.
    // ~16k is required for a walk distance of 64 in each directions.
    const size_t walkDistance = 64;

    // using the stride directly instead of calling GetVoxel for each iterations saves
    // a lot of multiplications and pipeline stalls due to data dependencies on imul.
    const size_t istride = &GetVoxel(1, 0, 0) - &GetVoxel(0, 0, 0);
    const size_t jstride = &GetVoxel(0, 1, 0) - &GetVoxel(0, 0, 0);
    const size_t kstride = &GetVoxel(0, 0, 1) - &GetVoxel(0, 0, 0);

    // It might seem counter intuitive to go over the whole voxel range multiple times
    // but since we do the run in memory order, it leaves us with far fewer cache misses
    // than a BFS algorithm and it has the additional benefit of not requiring us to
    // store and manipulate a fifo for recursion that might become huge when the number
    // of voxels is large.
    // This will outperform the BFS algorithm by several orders of magnitude in practice.
    do
    {
        voxelsWalked = 0;
        for (int64_t i = 0; i < i0; ++i)
        {
            for (int64_t j = 0; j < j0; ++j)
            {
                for (int64_t k = 0; k < k0; ++k)
                {
                    VoxelValue& voxel = GetVoxel(i, j, k);
                    if (voxel == VoxelValue::PRIMITIVE_OUTSIDE_SURFACE_TOWALK)
                    {
                        voxelsWalked++;
                        voxel = VoxelValue::PRIMITIVE_OUTSIDE_SURFACE;

                        // walk in each direction to mark other voxel that should be walked.
                        // this will generate a 3d pattern that will help the overall
                        // algorithm converge faster while remaining cache friendly.
                        WalkForward(k + 1, k0, &voxel + kstride, kstride, walkDistance);
                        WalkBackward(k - 1, 0, &voxel - kstride, kstride, walkDistance);

                        WalkForward(j + 1, j0, &voxel + jstride, jstride, walkDistance);
                        WalkBackward(j - 1, 0, &voxel - jstride, jstride, walkDistance);

                        WalkForward(i + 1, i0, &voxel + istride, istride, walkDistance);
                        WalkBackward(i - 1, 0, &voxel - istride, istride, walkDistance);
                    }
                }
            }
        }

        m_numVoxelsOutsideSurface += voxelsWalked;
    } while (voxelsWalked != 0);
}

void Volume::FillInsideSurface()
{
    const uint32_t i0 = uint32_t(m_dim[0]);
    const uint32_t j0 = uint32_t(m_dim[1]);
    const uint32_t k0 = uint32_t(m_dim[2]);

    size_t maxSize = i0 * j0 * k0;

    std::vector<Voxel> temp;
    temp.reserve(maxSize);
    uint32_t count{ 0 };

    for (uint32_t i = 0; i < i0; ++i)
    {
        for (uint32_t j = 0; j < j0; ++j)
        {
            for (uint32_t k = 0; k < k0; ++k)
            {
                VoxelValue& v = GetVoxel(i, j, k);
                if (v == VoxelValue::PRIMITIVE_UNDEFINED)
                {
                    v = VoxelValue::PRIMITIVE_INSIDE_SURFACE;
                    temp.emplace_back(i, j, k);
                    count++;
                    ++m_numVoxelsInsideSurface;
                }
            }
        }
    }

    if ( count )
    {
        m_interiorVoxels = std::move(temp);
    }
}

//******************************************************************************************
//  ShrinkWrap helper class
//******************************************************************************************
// This is a code snippet which 'shrinkwraps' a convex hull
// to a source mesh.
//
// It is a somewhat complicated algorithm. It works as follows:
//
// * Step #1 : Compute the mean unit normal vector for each vertex in the convex hull
// * Step #2 : For each vertex in the conex hull we project is slightly outwards along the mean normal vector
// * Step #3 : We then raycast from this slightly extruded point back into the opposite direction of the mean normal vector
//             resulting in a raycast from slightly beyond the vertex in the hull into the source mesh we are trying
//             to 'shrink wrap' against
// * Step #4 : If the raycast fails we leave the original vertex alone
// * Step #5 : If the raycast hits a backface we leave the original vertex alone
// * Step #6 : If the raycast hits too far away (no more than a certain threshold distance) we live it alone
// * Step #7 : If the point we hit on the source mesh is not still within the convex hull, we reject it.
// * Step #8 : If all of the previous conditions are met, then we take the raycast hit location as the 'new position'
// * Step #9 : Once all points have been projected, if possible, we need to recompute the convex hull again based on these shrinkwrapped points
// * Step #10 : In theory that should work.. let's see...

//***********************************************************************************************
// QuickHull implementation
//***********************************************************************************************

//////////////////////////////////////////////////////////////////////////
// Quickhull base class holding the hull during construction
//////////////////////////////////////////////////////////////////////////
class QuickHull
{
public:
    uint32_t ComputeConvexHull(const std::vector<VHACD::Vertex>& vertices,
                               uint32_t maxHullVertices);

    const std::vector<VHACD::Vertex>& GetVertices() const;
    const std::vector<VHACD::Triangle>& GetIndices() const;

private:
    std::vector<VHACD::Vertex>   m_vertices;
    std::vector<VHACD::Triangle> m_indices;
};

uint32_t QuickHull::ComputeConvexHull(const std::vector<VHACD::Vertex>& vertices,
                                      uint32_t maxHullVertices)
{
    m_indices.clear();

    VHACD::ConvexHull ch(vertices,
                         double(0.0001),
                         maxHullVertices);

    auto& vlist = ch.GetVertexPool();
    if ( !vlist.empty() )
    {
        size_t vcount = vlist.size();
        m_vertices.resize(vcount);
        std::copy(vlist.begin(),
                  vlist.end(),
                  m_vertices.begin());
    }

    for (std::list<ConvexHullFace>::const_iterator node = ch.GetList().begin(); node != ch.GetList().end(); ++node)
    {
        const VHACD::ConvexHullFace& face = *node;
        m_indices.emplace_back(face.m_index[0],
                               face.m_index[1],
                               face.m_index[2]);
    }

    return uint32_t(m_indices.size());
}

const std::vector<VHACD::Vertex>& QuickHull::GetVertices() const
{
    return m_vertices;
}

const std::vector<VHACD::Triangle>& QuickHull::GetIndices() const
{
    return m_indices;
}

//******************************************************************************************
// Implementation of the ShrinkWrap function
//******************************************************************************************

void ShrinkWrap(SimpleMesh& sourceConvexHull,
                const AABBTree& aabbTree,
                uint32_t maxHullVertexCount,
                double distanceThreshold,
                bool doShrinkWrap)
{
    std::vector<VHACD::Vertex> verts; // New verts for the new convex hull
    verts.reserve(sourceConvexHull.m_vertices.size());
    // Examine each vertex and see if it is within the voxel distance.
    // If it is, then replace the point with the shrinkwrapped / projected point
    for (uint32_t j = 0; j < sourceConvexHull.m_vertices.size(); j++)
    {
        VHACD::Vertex& p = sourceConvexHull.m_vertices[j];
        if (doShrinkWrap)
        {
            VHACD::Vect3 closest;
            if (aabbTree.GetClosestPointWithinDistance(p, distanceThreshold, closest))
            {
                p = closest;
            }
        }
        verts.emplace_back(p);
    }
    // Final step is to recompute the convex hull
    VHACD::QuickHull qh;
    uint32_t tcount = qh.ComputeConvexHull(verts,
                                            maxHullVertexCount);
    if (tcount)
    {
        sourceConvexHull.m_vertices = qh.GetVertices();
        sourceConvexHull.m_indices = qh.GetIndices();
    }
}

//********************************************************************************************************************

#if !VHACD_DISABLE_THREADING

//********************************************************************************************************************
// Definition of the ThreadPool
//********************************************************************************************************************

class ThreadPool {
 public:
    ThreadPool();
    ThreadPool(int worker);
    ~ThreadPool();
    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&& ... args)
#ifndef __cpp_lib_is_invocable
        -> std::future< typename std::result_of< F( Args... ) >::type>;
#else
        -> std::future< typename std::invoke_result_t<F, Args...>>;
#endif
 private:
    std::vector<std::thread> workers;
    std::deque<std::function<void()>> tasks;
    std::mutex task_mutex;
    std::condition_variable cv;
    bool closed;
    // int count; // Unused upstream. See https://github.com/kmammou/v-hacd/issues/148
};

ThreadPool::ThreadPool()
    : ThreadPool(1)
{
}

ThreadPool::ThreadPool(int worker)
    : closed(false)
    // , count(0) // Unused upstream. See https://github.com/kmammou/v-hacd/issues/148
{
    workers.reserve(worker);
    for(int i=0; i<worker; i++) 
    {
        workers.emplace_back(
            [this]
            {
                std::unique_lock<std::mutex> lock(this->task_mutex);
                while(true) 
                {
                    while (this->tasks.empty()) 
                    {
                        if (this->closed) 
                        {
                            return;
                        }
                        this->cv.wait(lock);
                    }
                    auto task = this->tasks.front();
                    this->tasks.pop_front();
                    lock.unlock();
                    task();
                    lock.lock();
                }
            }
        );
    }
}

template<typename F, typename... Args>
auto ThreadPool::enqueue(F&& f, Args&& ... args)
#ifndef __cpp_lib_is_invocable
    -> std::future< typename std::result_of< F( Args... ) >::type>
#else
    -> std::future< typename std::invoke_result_t<F, Args...>>
#endif
{

#ifndef __cpp_lib_is_invocable
    using return_type = typename std::result_of< F( Args... ) >::type;
#else
    using return_type = typename std::invoke_result_t< F, Args... >;
#endif
    auto task = std::make_shared<std::packaged_task<return_type()> > (
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    auto result = task->get_future();

    {
        std::unique_lock<std::mutex> lock(task_mutex);
        if (!closed) 
        {
            tasks.emplace_back([task]
            { 
                (*task)();
            });
            cv.notify_one();
        }
    }

    return result;
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(task_mutex);
        closed = true;
    }
    cv.notify_all();
    for (auto && worker : workers) 
    {
        worker.join();
    }
}
#endif

enum class Stages
{
    COMPUTE_BOUNDS_OF_INPUT_MESH,
    REINDEXING_INPUT_MESH,
    CREATE_RAYCAST_MESH,
    VOXELIZING_INPUT_MESH,
    BUILD_INITIAL_CONVEX_HULL,
    PERFORMING_DECOMPOSITION,
    INITIALIZING_CONVEX_HULLS_FOR_MERGING,
    COMPUTING_COST_MATRIX,
    MERGING_CONVEX_HULLS,
    FINALIZING_RESULTS,
    NUM_STAGES
};

class VHACDCallbacks
{
public:
    virtual void ProgressUpdate(Stages stage,
                                double stageProgress,
                                const char *operation) = 0;
    virtual bool IsCanceled() const = 0;

    virtual ~VHACDCallbacks() = default;
};

enum class SplitAxis
{
    X_AXIS_NEGATIVE,
    X_AXIS_POSITIVE,
    Y_AXIS_NEGATIVE,
    Y_AXIS_POSITIVE,
    Z_AXIS_NEGATIVE,
    Z_AXIS_POSITIVE,
};

// This class represents a collection of voxels, the convex hull
// which surrounds them, and a triangle mesh representation of those voxels
class VoxelHull
{
public:

    // This method constructs a new VoxelHull based on a plane split of the parent
    // convex hull
    VoxelHull(const VoxelHull& parent,
              SplitAxis axis,
              uint32_t splitLoc);

    // Here we construct the initial convex hull around the
    // entire voxel set
    VoxelHull(Volume& voxels,
              const IVHACD::Parameters &params,
              VHACDCallbacks *callbacks);

    ~VoxelHull() = default;

    // Helper method to refresh the min/max voxel bounding region
    void MinMaxVoxelRegion(const Voxel &v);

    void BuildRaycastMesh();

    // We now compute the convex hull relative to a triangle mesh generated 
    // from the voxels
    void ComputeConvexHull();

    // Returns true if this convex hull should be considered done
    bool IsComplete();

    
    // Convert a voxel position into it's correct double precision location
    VHACD::Vect3 GetPoint(const int32_t x,
                                 const int32_t y,
                                 const int32_t z,
                                 const double scale,
                                 const VHACD::Vect3& bmin) const;

    // Sees if we have already got an index for this voxel position.
    // If the voxel position has already been indexed, we just return
    // that index value.
    // If not, then we convert it into the floating point position and
    // add it to the index map
    uint32_t GetVertexIndex(const VHACD::Vector3<uint32_t>& p);

    // This method will convert the voxels into an actual indexed triangle mesh of boxes
    // This serves two purposes.
    // The primary purpose is so that when we compute a convex hull it considered all of the points
    // for each voxel, not just the center point. If you don't do this, then the hulls don't fit the
    // mesh accurately enough.
    // The second reason we convert it into a triangle mesh is so that we can do raycasting against it
    // to search for the best splitting plane fairly quickly. That algorithm will be discussed in the 
    // method which computes the best splitting plane.
    void BuildVoxelMesh();

    // Convert a single voxel position into an actual 3d box mesh comprised
    // of 12 triangles
    void AddVoxelBox(const Voxel &v);
    
    // Add the triangle represented by these 3 indices into the 'box' set of vertices
    // to the output mesh
    void AddTri(const std::array<VHACD::Vector3<uint32_t>, 8>& box,
                uint32_t i1,
                uint32_t i2,
                uint32_t i3);

    // Here we convert from voxel space to a 3d position, index it, and add
    // the triangle positions and indices for the output mesh
    void AddTriangle(const VHACD::Vector3<uint32_t>& p1,
                     const VHACD::Vector3<uint32_t>& p2,
                     const VHACD::Vector3<uint32_t>& p3);

    // When computing the split plane, we start by simply 
    // taking the midpoint of the longest side. However,
    // we can also search the surface and look for the greatest
    // spot of concavity and use that as the split location.
    // This will make the convex decomposition more efficient
    // as it will tend to cut across the greatest point of
    // concavity on the surface.
    SplitAxis ComputeSplitPlane(uint32_t& location);

    VHACD::Vect3 GetPosition(const VHACD::Vector3<int32_t>& ip) const;

    double Raycast(const VHACD::Vector3<int32_t>& p1,
                   const VHACD::Vector3<int32_t>& p2) const;

    bool FindConcavity(uint32_t idx,
                       uint32_t& splitLoc);

    // Finding the greatest area of concavity..
    bool FindConcavityX(uint32_t& splitLoc);

    // Finding the greatest area of concavity..
    bool FindConcavityY(uint32_t& splitLoc);

    // Finding the greatest area of concavity..
    bool FindConcavityZ(uint32_t& splitLoc);

    // This operation is performed in a background thread.
    // It splits the voxels by a plane
    void PerformPlaneSplit();

    // Used only for debugging. Saves the voxelized mesh to disk
    // Optionally saves the original source mesh as well for comparison
    void SaveVoxelMesh(const SimpleMesh& inputMesh,
                       bool saveVoxelMesh,
                       bool saveSourceMesh);

    void SaveOBJ(const char* fname,
                 const VoxelHull* h);

    void SaveOBJ(const char* fname);

private:
    void WriteOBJ(FILE* fph,
                  const std::vector<VHACD::Vertex>& vertices,
                  const std::vector<VHACD::Triangle>& indices,
                  uint32_t baseIndex);
public:

    SplitAxis               m_axis{ SplitAxis::X_AXIS_NEGATIVE };
    Volume*                 m_voxels{ nullptr }; // The voxelized data set
    double                  m_voxelScale{ 0 };   // Size of a single voxel
    double                  m_voxelScaleHalf{ 0 }; // 1/2 of the size of a single voxel
    VHACD::BoundsAABB       m_voxelBounds;
    VHACD::Vect3            m_voxelAdjust;       // Minimum coordinates of the voxel space, with adjustment
    uint32_t                m_depth{ 0 };        // How deep in the recursion of the binary tree this hull is
    uint32_t                m_index{ 0 };        // Each convex hull is given a unique id to distinguish it from the others
    double                  m_volumeError{ 0 };  // The percentage error from the convex hull volume vs. the voxel volume
    double                  m_voxelVolume{ 0 };  // The volume of the voxels
    double                  m_hullVolume{ 0 };   // The volume of the enclosing convex hull

    std::unique_ptr<IVHACD::ConvexHull> m_convexHull{ nullptr }; // The convex hull which encloses this set of voxels.
    std::vector<Voxel>                  m_surfaceVoxels;     // The voxels which are on the surface of the source mesh.
    std::vector<Voxel>                  m_newSurfaceVoxels;  // Voxels which are on the surface as a result of a plane split
    std::vector<Voxel>                  m_interiorVoxels;    // Voxels which are part of the interior of the hull

    std::unique_ptr<VoxelHull>          m_hullA{ nullptr }; // hull resulting from one side of the plane split
    std::unique_ptr<VoxelHull>          m_hullB{ nullptr }; // hull resulting from the other side of the plane split

    // Defines the coordinates this convex hull comprises within the voxel volume
    // of the entire source
    VHACD::Vector3<uint32_t>                    m_1{ 0 };
    VHACD::Vector3<uint32_t>                    m_2{ 0 };
    AABBTree                                    m_AABBTree;
    std::unordered_map<uint32_t, uint32_t>      m_voxelIndexMap; // Maps from a voxel coordinate space into a vertex index space
    std::vector<VHACD::Vertex>                  m_vertices;
    std::vector<VHACD::Triangle>                m_indices;
    static uint32_t                             m_voxelHullCount;
    IVHACD::Parameters                          m_params;
    VHACDCallbacks*                             m_callbacks{ nullptr };
};

uint32_t VoxelHull::m_voxelHullCount = 0;

VoxelHull::VoxelHull(const VoxelHull& parent,
                     SplitAxis axis,
                     uint32_t splitLoc)
    : m_axis(axis)
    , m_voxels(parent.m_voxels)
    , m_voxelScale(m_voxels->GetScale())
    , m_voxelScaleHalf(m_voxelScale * double(0.5))
    , m_voxelBounds(m_voxels->GetBounds())
    , m_voxelAdjust(m_voxelBounds.GetMin() - m_voxelScaleHalf)
    , m_depth(parent.m_depth + 1)
    , m_index(++m_voxelHullCount)
    , m_1(parent.m_1)
    , m_2(parent.m_2)
    , m_params(parent.m_params)
{
    // Default copy the voxel region from the parent, but values will
    // be adjusted next based on the split axis and location
    switch ( m_axis )
    {
        case SplitAxis::X_AXIS_NEGATIVE:
            m_2.GetX() = splitLoc;
            break;
        case SplitAxis::X_AXIS_POSITIVE:
            m_1.GetX() = splitLoc + 1;
            break;
        case SplitAxis::Y_AXIS_NEGATIVE:
            m_2.GetY() = splitLoc;
            break;
        case SplitAxis::Y_AXIS_POSITIVE:
            m_1.GetY() = splitLoc + 1;
            break;
        case SplitAxis::Z_AXIS_NEGATIVE:
            m_2.GetZ() = splitLoc;
            break;
        case SplitAxis::Z_AXIS_POSITIVE:
            m_1.GetZ() = splitLoc + 1;
            break;
    }

    // First, we copy all of the interior voxels from our parent
    // which intersect our region
    for (auto& i : parent.m_interiorVoxels)
    {
        VHACD::Vector3<uint32_t> v = i.GetVoxel();
        if (v.CWiseAllGE(m_1) && v.CWiseAllLE(m_2))
        {
            bool newSurface = false;
            switch ( m_axis )
            {
                case SplitAxis::X_AXIS_NEGATIVE:
                    if ( v.GetX() == splitLoc )
                    {
                        newSurface = true;
                    }
                    break;
                case SplitAxis::X_AXIS_POSITIVE:
                    if ( v.GetX() == m_1.GetX() )
                    {
                        newSurface = true;
                    }
                    break;
                case SplitAxis::Y_AXIS_NEGATIVE:
                    if ( v.GetY() == splitLoc )
                    {
                        newSurface = true;
                    }
                    break;
                case SplitAxis::Y_AXIS_POSITIVE:
                    if ( v.GetY() == m_1.GetY() )
                    {
                        newSurface = true;
                    }
                    break;
                case SplitAxis::Z_AXIS_NEGATIVE:
                    if ( v.GetZ() == splitLoc )
                    {
                        newSurface = true;
                    }
                    break;
                case SplitAxis::Z_AXIS_POSITIVE:
                    if ( v.GetZ() == m_1.GetZ() )
                    {
                        newSurface = true;
                    }
                    break;
            }
            // If his interior voxels lie directly on the split plane then
            // these become new surface voxels for our patch
            if ( newSurface )
            {
                m_newSurfaceVoxels.push_back(i);
            }
            else
            {
                m_interiorVoxels.push_back(i);
            }
        }
    }
    // Next we copy all of the surface voxels which intersect our region
    for (auto& i : parent.m_surfaceVoxels)
    {
        VHACD::Vector3<uint32_t> v = i.GetVoxel();
        if (v.CWiseAllGE(m_1) && v.CWiseAllLE(m_2))
        {
            m_surfaceVoxels.push_back(i);
        }
    }
    // Our parent's new surface voxels become our new surface voxels so long as they intersect our region
    for (auto& i : parent.m_newSurfaceVoxels)
    {
        VHACD::Vector3<uint32_t> v = i.GetVoxel();
        if (v.CWiseAllGE(m_1) && v.CWiseAllLE(m_2))
        {
            m_newSurfaceVoxels.push_back(i);
        }
    }

    // Recompute the min-max bounding box which would be different after the split occurs
    m_1 = VHACD::Vector3<uint32_t>(0x7FFFFFFF);
    m_2 = VHACD::Vector3<uint32_t>(0);
    for (auto& i : m_surfaceVoxels)
    {
        MinMaxVoxelRegion(i);
    }
    for (auto& i : m_newSurfaceVoxels)
    {
        MinMaxVoxelRegion(i);
    }
    for (auto& i : m_interiorVoxels)
    {
        MinMaxVoxelRegion(i);
    }

    BuildVoxelMesh();
    BuildRaycastMesh(); // build a raycast mesh of the voxel mesh
    ComputeConvexHull();
}

VoxelHull::VoxelHull(Volume& voxels,
                     const IVHACD::Parameters& params,
                     VHACDCallbacks* callbacks)
    : m_voxels(&voxels)
    , m_voxelScale(m_voxels->GetScale())
    , m_voxelScaleHalf(m_voxelScale * double(0.5))
    , m_voxelBounds(m_voxels->GetBounds())
    , m_voxelAdjust(m_voxelBounds.GetMin() - m_voxelScaleHalf)
    , m_index(++m_voxelHullCount)
    // Here we get a copy of all voxels which lie on the surface mesh
    , m_surfaceVoxels(m_voxels->GetSurfaceVoxels())
    // Now we get a copy of all voxels which are considered part of the 'interior' of the source mesh
    , m_interiorVoxels(m_voxels->GetInteriorVoxels())
    , m_2(m_voxels->GetDimensions() - 1)
    , m_params(params)
    , m_callbacks(callbacks)
{
    BuildVoxelMesh();
    BuildRaycastMesh(); // build a raycast mesh of the voxel mesh
    ComputeConvexHull();
}

void VoxelHull::MinMaxVoxelRegion(const Voxel& v)
{
    VHACD::Vector3<uint32_t> x = v.GetVoxel();
    m_1 = m_1.CWiseMin(x);
    m_2 = m_2.CWiseMax(x);
}

void VoxelHull::BuildRaycastMesh()
{
    // Create a raycast mesh representation of the voxelized surface mesh
    if ( !m_indices.empty() )
    {
        m_AABBTree = AABBTree(m_vertices,
                              m_indices);
    }
}

void VoxelHull::ComputeConvexHull()
{
    if ( !m_vertices.empty() )
    {
        // we compute the convex hull as follows...
        VHACD::QuickHull qh;
        uint32_t tcount = qh.ComputeConvexHull(m_vertices,
                                               uint32_t(m_vertices.size()));
        if ( tcount )
        {
            m_convexHull = std::unique_ptr<IVHACD::ConvexHull>(new IVHACD::ConvexHull);

            m_convexHull->m_points = qh.GetVertices();
            m_convexHull->m_triangles = qh.GetIndices();

            VHACD::ComputeCentroid(m_convexHull->m_points,
                                   m_convexHull->m_triangles,
                                   m_convexHull->m_center);
            m_convexHull->m_volume = VHACD::ComputeMeshVolume(m_convexHull->m_points,
                                                              m_convexHull->m_triangles);
        }
    }
    if ( m_convexHull )
    {
        m_hullVolume = m_convexHull->m_volume;
    }
    // This is the volume of a single voxel
    double singleVoxelVolume = m_voxelScale * m_voxelScale * m_voxelScale;
    size_t voxelCount = m_interiorVoxels.size() + m_newSurfaceVoxels.size() + m_surfaceVoxels.size();
    m_voxelVolume = singleVoxelVolume * double(voxelCount);
    double diff = fabs(m_hullVolume - m_voxelVolume);
    m_volumeError = (diff * 100) / m_voxelVolume;
}

bool VoxelHull::IsComplete()
{
    bool ret = false;
    if ( m_convexHull == nullptr )
    {
        ret = true;
    }
    else if ( m_volumeError < m_params.m_minimumVolumePercentErrorAllowed )
    {
        ret = true;
    }
    else if ( m_depth > m_params.m_maxRecursionDepth )
    {
        ret = true;
    }
    else
    {
        // We compute the voxel width on all 3 axes and see if they are below the min threshold size
        VHACD::Vector3<uint32_t> d = m_2 - m_1;
        if ( d.GetX() <= m_params.m_minEdgeLength &&
             d.GetY() <= m_params.m_minEdgeLength &&
             d.GetZ() <= m_params.m_minEdgeLength )
        {
            ret = true;
        }
    }
    return ret;
}

VHACD::Vect3 VoxelHull::GetPoint(const int32_t x,
                                 const int32_t y,
                                 const int32_t z,
                                 const double scale,
                                 const VHACD::Vect3& bmin) const
{
    return VHACD::Vect3(x * scale + bmin.GetX(),
                        y * scale + bmin.GetY(),
                        z * scale + bmin.GetZ());
}

uint32_t VoxelHull::GetVertexIndex(const VHACD::Vector3<uint32_t>& p)
{
    uint32_t ret = 0;
    uint32_t address = (p.GetX() << 20) | (p.GetY() << 10) | p.GetZ();
    auto found = m_voxelIndexMap.find(address);
    if ( found != m_voxelIndexMap.end() )
    {
        ret = found->second;
    }
    else
    {
        VHACD::Vect3 vertex = GetPoint(p.GetX(),
                                       p.GetY(),
                                       p.GetZ(),
                                       m_voxelScale,
                                       m_voxelAdjust);
        ret = uint32_t(m_voxelIndexMap.size());
        m_voxelIndexMap[address] = ret;
        m_vertices.emplace_back(vertex);
    }
    return ret;
}

void VoxelHull::BuildVoxelMesh()
{
    // When we build the triangle mesh we do *not* need the interior voxels, only the ones
    // which lie upon the logical surface of the mesh.
    // Each time we perform a plane split, voxels which are along the splitting plane become
    // 'new surface voxels'.

    for (auto& i : m_surfaceVoxels)
    {
        AddVoxelBox(i);
    }
    for (auto& i : m_newSurfaceVoxels)
    {
        AddVoxelBox(i);
    }
}

void VoxelHull::AddVoxelBox(const Voxel &v)
{
    // The voxel position of the upper left corner of the box
    VHACD::Vector3<uint32_t> bmin(v.GetX(),
                                  v.GetY(),
                                  v.GetZ());
    // The voxel position of the lower right corner of the box
    VHACD::Vector3<uint32_t> bmax(bmin.GetX() + 1,
                                  bmin.GetY() + 1,
                                  bmin.GetZ() + 1);

    // Build the set of 8 voxel positions representing
    // the coordinates of the box
    std::array<VHACD::Vector3<uint32_t>, 8> box{{
        { bmin.GetX(), bmin.GetY(), bmin.GetZ() },
        { bmax.GetX(), bmin.GetY(), bmin.GetZ() },
        { bmax.GetX(), bmax.GetY(), bmin.GetZ() },
        { bmin.GetX(), bmax.GetY(), bmin.GetZ() },
        { bmin.GetX(), bmin.GetY(), bmax.GetZ() },
        { bmax.GetX(), bmin.GetY(), bmax.GetZ() },
        { bmax.GetX(), bmax.GetY(), bmax.GetZ() },
        { bmin.GetX(), bmax.GetY(), bmax.GetZ() }
    }};

    // Now add the 12 triangles comprising the 3d box
    AddTri(box, 2, 1, 0);
    AddTri(box, 3, 2, 0);

    AddTri(box, 7, 2, 3);
    AddTri(box, 7, 6, 2);

    AddTri(box, 5, 1, 2);
    AddTri(box, 5, 2, 6);

    AddTri(box, 5, 4, 1);
    AddTri(box, 4, 0, 1);

    AddTri(box, 4, 6, 7);
    AddTri(box, 4, 5, 6);

    AddTri(box, 4, 7, 0);
    AddTri(box, 7, 3, 0);
}

void VoxelHull::AddTri(const std::array<VHACD::Vector3<uint32_t>, 8>& box,
                       uint32_t i1,
                       uint32_t i2,
                       uint32_t i3)
{
    AddTriangle(box[i1], box[i2], box[i3]);
}

void VoxelHull::AddTriangle(const VHACD::Vector3<uint32_t>& p1,
                            const VHACD::Vector3<uint32_t>& p2,
                            const VHACD::Vector3<uint32_t>& p3)
{
    uint32_t i1 = GetVertexIndex(p1);
    uint32_t i2 = GetVertexIndex(p2);
    uint32_t i3 = GetVertexIndex(p3);

    m_indices.emplace_back(i1, i2, i3);
}

SplitAxis VoxelHull::ComputeSplitPlane(uint32_t& location)
{
    SplitAxis ret = SplitAxis::X_AXIS_NEGATIVE;

    VHACD::Vector3<uint32_t> d = m_2 - m_1;

    if ( d.GetX() >= d.GetY() && d.GetX() >= d.GetZ() )
    {
        ret = SplitAxis::X_AXIS_NEGATIVE;
        location = (m_2.GetX() + 1 + m_1.GetX()) / 2;
        uint32_t edgeLoc;
        if ( m_params.m_findBestPlane && FindConcavityX(edgeLoc) )
        {
            location = edgeLoc;
        }
    }
    else if ( d.GetY() >= d.GetX() && d.GetY() >= d.GetZ() )
    {
        ret = SplitAxis::Y_AXIS_NEGATIVE;
        location = (m_2.GetY() + 1 + m_1.GetY()) / 2;
        uint32_t edgeLoc;
        if ( m_params.m_findBestPlane && FindConcavityY(edgeLoc) )
        {
            location = edgeLoc;
        }
    }
    else
    {
        ret = SplitAxis::Z_AXIS_NEGATIVE;
        location = (m_2.GetZ() + 1 + m_1.GetZ()) / 2;
        uint32_t edgeLoc;
        if ( m_params.m_findBestPlane && FindConcavityZ(edgeLoc) )
        {
            location = edgeLoc;
        }
    }

    return ret;
}

VHACD::Vect3 VoxelHull::GetPosition(const VHACD::Vector3<int32_t>& ip) const
{
    return GetPoint(ip.GetX(),
                    ip.GetY(),
                    ip.GetZ(),
                    m_voxelScale,
                    m_voxelAdjust);
}

double VoxelHull::Raycast(const VHACD::Vector3<int32_t>& p1,
                          const VHACD::Vector3<int32_t>& p2) const
{
    double ret;
    VHACD::Vect3 from = GetPosition(p1);
    VHACD::Vect3 to = GetPosition(p2);

    double outT;
    double faceSign;
    VHACD::Vect3 hitLocation;
    if (m_AABBTree.TraceRay(from, to, outT, faceSign, hitLocation))
    {
        ret = (from - hitLocation).GetNorm();
    }
    else
    {
        ret = 0; // if it doesn't hit anything, just assign it to zero.
    }

    return ret;
}

bool VoxelHull::FindConcavity(uint32_t idx,
                              uint32_t& splitLoc)
{
    bool ret = false;

    int32_t d = (m_2[idx] - m_1[idx]) + 1; // The length of the getX axis in voxel space

    uint32_t idx1;
    uint32_t idx2;
    uint32_t idx3;
    switch (idx)
    {
        case 0: // X
            idx1 = 0;
            idx2 = 1;
            idx3 = 2;
            break;
        case 1: // Y
            idx1 = 1;
            idx2 = 0;
            idx3 = 2;
            break;
        case 2:
            idx1 = 2;
            idx2 = 1;
            idx3 = 0;
            break;
        default:
            /*
                * To silence uninitialized variable warnings
                */
            idx1 = 0;
            idx2 = 0;
            idx3 = 0;
            assert(0 && "findConcavity::idx must be 0, 1, or 2");
            break;
    }

    // We will compute the edge error on the XY plane and the XZ plane
    // searching for the greatest location of concavity
    std::vector<double> edgeError1 = std::vector<double>(d);
    std::vector<double> edgeError2 = std::vector<double>(d);

    // Counter of number of voxel samples on the XY plane we have accumulated
    uint32_t index1 = 0;

    // Compute Edge Error on the XY plane
    for (uint32_t i0 = m_1[idx1]; i0 <= m_2[idx1]; i0++)
    {
        double errorTotal = 0;
        // We now perform a raycast from the sides inward on the XY plane to
        // determine the total error (distance of the surface from the sides)
        // along this getX position.
        for (uint32_t i1 = m_1[idx2]; i1 <= m_2[idx2]; i1++)
        {
            VHACD::Vector3<int32_t> p1;
            VHACD::Vector3<int32_t> p2;
            switch (idx)
            {
                case 0:
                {
                    p1 = VHACD::Vector3<int32_t>(i0, i1, m_1.GetZ() - 2);
                    p2 = VHACD::Vector3<int32_t>(i0, i1, m_2.GetZ() + 2);
                    break;
                }
                case 1:
                {
                    p1 = VHACD::Vector3<int32_t>(i1, i0, m_1.GetZ() - 2);
                    p2 = VHACD::Vector3<int32_t>(i1, i0, m_2.GetZ() + 2);
                    break;
                }
                case 2:
                {
                    p1 = VHACD::Vector3<int32_t>(m_1.GetX() - 2, i1, i0);
                    p2 = VHACD::Vector3<int32_t>(m_2.GetX() + 2, i1, i0);
                    break;
                }
            }

            double e1 = Raycast(p1, p2);
            double e2 = Raycast(p2, p1);

            errorTotal = errorTotal + e1 + e2;
        }
        // The total amount of edge error along this voxel location
        edgeError1[index1] = errorTotal;
        index1++;
    }

    // Compute edge error along the XZ plane
    uint32_t index2 = 0;

    for (uint32_t i0 = m_1[idx1]; i0 <= m_2[idx1]; i0++)
    {
        double errorTotal = 0;

        for (uint32_t i1 = m_1[idx3]; i1 <= m_2[idx3]; i1++)
        {
            VHACD::Vector3<int32_t> p1;
            VHACD::Vector3<int32_t> p2;
            switch (idx)
            {
                case 0:
                {
                    p1 = VHACD::Vector3<int32_t>(i0, m_1.GetY() - 2, i1);
                    p2 = VHACD::Vector3<int32_t>(i0, m_2.GetY() + 2, i1);
                    break;
                }
                case 1:
                {
                    p1 = VHACD::Vector3<int32_t>(m_1.GetX() - 2, i0, i1);
                    p2 = VHACD::Vector3<int32_t>(m_2.GetX() + 2, i0, i1);
                    break;
                }
                case 2:
                {
                    p1 = VHACD::Vector3<int32_t>(i1, m_1.GetY() - 2, i0);
                    p2 = VHACD::Vector3<int32_t>(i1, m_2.GetY() + 2, i0);
                    break;
                }
            }

            double e1 = Raycast(p1, p2); // raycast from one side to the interior
            double e2 = Raycast(p2, p1); // raycast from the other side to the interior

            errorTotal = errorTotal + e1 + e2;
        }
        edgeError2[index2] = errorTotal;
        index2++;
    }


    // we now compute the first derivative to find the greatest spot of concavity on the XY plane
    double maxDiff = 0;
    uint32_t maxC = 0;
    for (uint32_t x = 1; x < index1; x++)
    {
        if ( edgeError1[x] > 0 &&  edgeError1[x - 1] > 0 )
        {
            double diff = abs(edgeError1[x] - edgeError1[x - 1]);
            if ( diff > maxDiff )
            {
                maxDiff = diff;
                maxC = x-1;
            }
        }
    }

    // Now see if there is a greater concavity on the XZ plane
    for (uint32_t x = 1; x < index2; x++)
    {
        if ( edgeError2[x] > 0 && edgeError2[x - 1] > 0 )
        {
            double diff = abs(edgeError2[x] - edgeError2[x - 1]);
            if ( diff > maxDiff )
            {
                maxDiff = diff;
                maxC = x - 1;
            }
        }
    }

    splitLoc = maxC + m_1[idx1];

    // we do not allow an edge split if it is too close to the ends
    if (    splitLoc > (m_1[idx1] + 4)
         && splitLoc < (m_2[idx1] - 4) )
    {
        ret = true;
    }

    return ret;
}

// Finding the greatest area of concavity..
bool VoxelHull::FindConcavityX(uint32_t& splitLoc)
{
    return FindConcavity(0, splitLoc);
}

// Finding the greatest area of concavity..
bool VoxelHull::FindConcavityY(uint32_t& splitLoc)
{
    return FindConcavity(1, splitLoc);
}

// Finding the greatest area of concavity..
bool VoxelHull::FindConcavityZ(uint32_t &splitLoc)
{
    return FindConcavity(2, splitLoc);
}

void VoxelHull::PerformPlaneSplit()
{
    if ( IsComplete() )
    {
    }
    else
    {
        uint32_t splitLoc;
        SplitAxis axis = ComputeSplitPlane(splitLoc);
        switch ( axis )
        {
            case SplitAxis::X_AXIS_NEGATIVE:
            case SplitAxis::X_AXIS_POSITIVE:
                // Split on the getX axis at this split location
                m_hullA = std::unique_ptr<VoxelHull>(new VoxelHull(*this, SplitAxis::X_AXIS_NEGATIVE, splitLoc));
                m_hullB = std::unique_ptr<VoxelHull>(new VoxelHull(*this, SplitAxis::X_AXIS_POSITIVE, splitLoc));
                break;
            case SplitAxis::Y_AXIS_NEGATIVE:
            case SplitAxis::Y_AXIS_POSITIVE:
                // Split on the 1 axis at this split location
                m_hullA = std::unique_ptr<VoxelHull>(new VoxelHull(*this, SplitAxis::Y_AXIS_NEGATIVE, splitLoc));
                m_hullB = std::unique_ptr<VoxelHull>(new VoxelHull(*this, SplitAxis::Y_AXIS_POSITIVE, splitLoc));
                break;
            case SplitAxis::Z_AXIS_NEGATIVE:
            case SplitAxis::Z_AXIS_POSITIVE:
                // Split on the getZ axis at this split location
                m_hullA = std::unique_ptr<VoxelHull>(new VoxelHull(*this, SplitAxis::Z_AXIS_NEGATIVE, splitLoc));
                m_hullB = std::unique_ptr<VoxelHull>(new VoxelHull(*this, SplitAxis::Z_AXIS_POSITIVE, splitLoc));
                break;
        }
    }
}

void VoxelHull::SaveVoxelMesh(const SimpleMesh &inputMesh,
                              bool saveVoxelMesh,
                              bool saveSourceMesh)
{
    char scratch[512];
    snprintf(scratch,
             sizeof(scratch),
             "voxel-mesh-%03d.obj",
             m_index);
    FILE *fph = fopen(scratch,
                      "wb");
    if ( fph )
    {
        uint32_t baseIndex = 1;
        if ( saveVoxelMesh )
        {
            WriteOBJ(fph,
                     m_vertices,
                     m_indices,
                     baseIndex);
            baseIndex += uint32_t(m_vertices.size());
        }
        if ( saveSourceMesh )
        {
            WriteOBJ(fph,
                     inputMesh.m_vertices,
                     inputMesh.m_indices,
                     baseIndex);
        }
        fclose(fph);
    }
}

void VoxelHull::SaveOBJ(const char* fname,
                        const VoxelHull* h)
{
    FILE *fph = fopen(fname,"wb");
    if ( fph )
    {
        uint32_t baseIndex = 1;
        WriteOBJ(fph,
                 m_vertices,
                 m_indices,
                 baseIndex);

        baseIndex += uint32_t(m_vertices.size());

        WriteOBJ(fph,
                 h->m_vertices,
                 h->m_indices,
                 baseIndex);
        fclose(fph);
    }
}

void VoxelHull::SaveOBJ(const char *fname)
{
    FILE *fph = fopen(fname, "wb");
    if ( fph )
    {
        printf("Saving '%s' with %d vertices and %d triangles\n",
                fname,
                uint32_t(m_vertices.size()),
                uint32_t(m_indices.size()));
        WriteOBJ(fph,
                 m_vertices,
                 m_indices,
                 1);
        fclose(fph);
    }
}

void VoxelHull::WriteOBJ(FILE* fph,
                         const std::vector<VHACD::Vertex>& vertices,
                         const std::vector<VHACD::Triangle>& indices,
                         uint32_t baseIndex)
{
    if (!fph)
    {
        return;
    }

    for (size_t i = 0; i < vertices.size(); ++i)
    {
        const VHACD::Vertex& v = vertices[i];
        fprintf(fph, "v %0.9f %0.9f %0.9f\n",
                v.mX,
                v.mY,
                v.mZ);
    }

    for (size_t i = 0; i < indices.size(); ++i)
    {
        const VHACD::Triangle& t = indices[i];
        fprintf(fph, "f %d %d %d\n",
                t.mI0 + baseIndex,
                t.mI1 + baseIndex,
                t.mI2 + baseIndex);
    }
}

class VHACDImpl;

// This class represents a single task to compute the volume error
// of two convex hulls combined
class CostTask
{
public:
    VHACDImpl*          m_this{ nullptr };
    IVHACD::ConvexHull* m_hullA{ nullptr };
    IVHACD::ConvexHull* m_hullB{ nullptr };
    double              m_concavity{ 0 }; // concavity of the two combined
    std::future<void>   m_future;
};

class HullPair
{
public:
    HullPair() = default;
    HullPair(uint32_t hullA,
             uint32_t hullB,
             double concavity);

    bool operator<(const HullPair &h) const;

    uint32_t    m_hullA{ 0 };
    uint32_t    m_hullB{ 0 };
    double      m_concavity{ 0 };
};

HullPair::HullPair(uint32_t hullA,
                   uint32_t hullB,
                   double concavity)
    : m_hullA(hullA)
    , m_hullB(hullB)
    , m_concavity(concavity)
{
}

bool HullPair::operator<(const HullPair &h) const
{
    return m_concavity > h.m_concavity ? true : false;
}

// void jobCallback(void* userPtr);

class VHACDImpl : public IVHACD, public VHACDCallbacks
{
    // Don't consider more than 100,000 convex hulls.
    static constexpr uint32_t MaxConvexHullFragments{ 100000 };
public:
    VHACDImpl() = default;

    /*
     * Overrides VHACD::IVHACD
     */
    ~VHACDImpl() override
    {
        Clean();
    }

    void Cancel() override final;

    bool Compute(const float* const points,
                 const uint32_t countPoints,
                 const uint32_t* const triangles,
                 const uint32_t countTriangles,
                 const Parameters& params) override final;

    bool Compute(const double* const points,
                 const uint32_t countPoints,
                 const uint32_t* const triangles,
                 const uint32_t countTriangles,
                 const Parameters& params) override final;

    uint32_t GetNConvexHulls() const override final;

    bool GetConvexHull(const uint32_t index,
                       ConvexHull& ch) const override final;

    void Clean() override final;  // release internally allocated memory

    void Release() override final;

    // Will compute the center of mass of the convex hull decomposition results and return it
    // in 'centerOfMass'.  Returns false if the center of mass could not be computed.
    bool ComputeCenterOfMass(double centerOfMass[3]) const override final;

    // In synchronous mode (non-multi-threaded) the state is always 'ready'
    // In asynchronous mode, this returns true if the background thread is not still actively computing
    // a new solution.  In an asynchronous config the 'IsReady' call will report any update or log
    // messages in the caller's current thread.
    bool IsReady(void) const override final;

    /**
    * At the request of LegionFu : out_look@foxmail.com
    * This method will return which convex hull is closest to the source position.
    * You can use this method to figure out, for example, which vertices in the original
    * source mesh are best associated with which convex hull.
    * 
    * @param pos : The input 3d position to test against
    * 
    * @return : Returns which convex hull this position is closest to.
    */
    uint32_t findNearestConvexHull(const double pos[3],
                                   double& distanceToHull) override final;

// private:
    bool Compute(const std::vector<VHACD::Vertex>& points,
                 const std::vector<VHACD::Triangle>& triangles,
                 const Parameters& params);

    // Take the source position, normalize it, and then convert it into an index position
    uint32_t GetIndex(VHACD::VertexIndex& vi,
                      const VHACD::Vertex& p);

    // This copies the input mesh while scaling the input positions
    // to fit into a normalized unit cube. It also re-indexes all of the
    // vertex positions in case they weren't clean coming in. 
    void CopyInputMesh(const std::vector<VHACD::Vertex>& points,
                       const std::vector<VHACD::Triangle>& triangles);

    void ScaleOutputConvexHull(ConvexHull &ch);

    void AddCostToPriorityQueue(CostTask& task);

    void ReleaseConvexHull(ConvexHull* ch);

    void PerformConvexDecomposition();

    double ComputeConvexHullVolume(const ConvexHull& sm);

    double ComputeVolume4(const VHACD::Vect3& a,
                          const VHACD::Vect3& b,
                          const VHACD::Vect3& c,
                          const VHACD::Vect3& d);

    double ComputeConcavity(double volumeSeparate,
                            double volumeCombined,
                            double volumeMesh);

    // See if we can compute the cost without having to actually merge convex hulls.
    // If the axis aligned bounding boxes (slightly inflated) of the two convex hulls
    // do not intersect, then we don't need to actually compute the merged convex hull
    // volume.
    bool DoFastCost(CostTask& mt);

    void PerformMergeCostTask(CostTask& mt);

    ConvexHull* ComputeReducedConvexHull(const ConvexHull& ch,
                                         uint32_t maxVerts,
                                         bool projectHullVertices);

    // Take the points in convex hull A and the points in convex hull B and generate
    // a new convex hull on the combined set of points.
    // Once completed, we create a SimpleMesh instance to hold the triangle mesh
    // and we compute an inflated AABB for it.
    ConvexHull* ComputeCombinedConvexHull(const ConvexHull& sm1,
                                          const ConvexHull& sm2);


    ConvexHull* GetHull(uint32_t index);

    bool RemoveHull(uint32_t index);

    ConvexHull* CopyConvexHull(const ConvexHull& source);

    const char* GetStageName(Stages stage) const;

    /*
     * Overrides VHACD::VHACDCallbacks
     */
    void ProgressUpdate(Stages stage,
                        double stageProgress,
                        const char* operation) override final;

    bool IsCanceled() const override final;

    std::atomic<bool>                                   m_canceled{ false };
    Parameters                                          m_params; // Convex decomposition parameters

    std::vector<IVHACD::ConvexHull*>                    m_convexHulls; // Finalized convex hulls
    std::vector<std::unique_ptr<VoxelHull>>             m_voxelHulls; // completed voxel hulls
    std::vector<std::unique_ptr<VoxelHull>>             m_pendingHulls;

    std::vector<std::unique_ptr<AABBTree>>              m_trees;
    VHACD::AABBTree                                     m_AABBTree;
    VHACD::Volume                                       m_voxelize;
    VHACD::Vect3                                        m_center;
    double                                              m_scale{ double(1.0) };
    double                                              m_recipScale{ double(1.0) };
    SimpleMesh                                          m_inputMesh; // re-indexed and normalized input mesh
    std::vector<VHACD::Vertex>                          m_vertices;
    std::vector<VHACD::Triangle>                        m_indices;

    double                                              m_overallHullVolume{ double(0.0) };
    double                                              m_voxelScale{ double(0.0) };
    double                                              m_voxelHalfScale{ double(0.0) };
    VHACD::Vect3                                        m_voxelBmin;
    VHACD::Vect3                                        m_voxelBmax;
    uint32_t                                            m_meshId{ 0 };
    std::priority_queue<HullPair>                       m_hullPairQueue;
#if !VHACD_DISABLE_THREADING
    std::unique_ptr<ThreadPool>                         m_threadPool{ nullptr };
#endif
    std::unordered_map<uint32_t, IVHACD::ConvexHull*>   m_hulls;

    double                                              m_overallProgress{ double(0.0) };
    double                                              m_stageProgress{ double(0.0) };
    double                                              m_operationProgress{ double(0.0) };
};

void VHACDImpl::Cancel()
{
    m_canceled = true;
}

bool VHACDImpl::Compute(const float* const points,
                        const uint32_t countPoints,
                        const uint32_t* const triangles,
                        const uint32_t countTriangles,
                        const Parameters& params)
{
    std::vector<VHACD::Vertex> v;
    v.reserve(countPoints);
    for (uint32_t i = 0; i < countPoints; ++i)
    {
        v.emplace_back(points[i * 3 + 0],
                       points[i * 3 + 1],
                       points[i * 3 + 2]);
    }

    std::vector<VHACD::Triangle> t;
    t.reserve(countTriangles);
    for (uint32_t i = 0; i < countTriangles; ++i)
    {
        t.emplace_back(triangles[i * 3 + 0],
                       triangles[i * 3 + 1],
                       triangles[i * 3 + 2]);
    }

    return Compute(v, t, params);
}

bool VHACDImpl::Compute(const double* const points,
                        const uint32_t countPoints,
                        const uint32_t* const triangles,
                        const uint32_t countTriangles,
                        const Parameters& params)
{
    std::vector<VHACD::Vertex> v;
    v.reserve(countPoints);
    for (uint32_t i = 0; i < countPoints; ++i)
    {
        v.emplace_back(points[i * 3 + 0],
                       points[i * 3 + 1],
                       points[i * 3 + 2]);
    }

    std::vector<VHACD::Triangle> t;
    t.reserve(countTriangles);
    for (uint32_t i = 0; i < countTriangles; ++i)
    {
        t.emplace_back(triangles[i * 3 + 0],
                       triangles[i * 3 + 1],
                       triangles[i * 3 + 2]);
    }

    return Compute(v, t, params);
}

uint32_t VHACDImpl::GetNConvexHulls() const
{
    return uint32_t(m_convexHulls.size());
}

bool VHACDImpl::GetConvexHull(const uint32_t index,
                              ConvexHull& ch) const
{
    bool ret = false;

    if ( index < uint32_t(m_convexHulls.size() ))
    {
        ch = *m_convexHulls[index];
        ret = true;
    }

    return ret;
}

void VHACDImpl::Clean()
{
#if !VHACD_DISABLE_THREADING
    m_threadPool = nullptr;
#endif

    m_trees.clear();

    for (auto& ch : m_convexHulls)
    {
        ReleaseConvexHull(ch);
    }
    m_convexHulls.clear();

    for (auto& ch : m_hulls)
    {
        ReleaseConvexHull(ch.second);
    }
    m_hulls.clear();

    m_voxelHulls.clear();

    m_pendingHulls.clear();

    m_vertices.clear();
    m_indices.clear();
}

void VHACDImpl::Release()
{
    delete this;
}

bool VHACDImpl::ComputeCenterOfMass(double centerOfMass[3]) const
{
    bool ret = false;

    return ret;
}

bool VHACDImpl::IsReady() const
{
    return true;
}

uint32_t VHACDImpl::findNearestConvexHull(const double pos[3],
                                          double& distanceToHull)
{
    uint32_t ret = 0; // The default return code is zero

    uint32_t hullCount = GetNConvexHulls();
    distanceToHull = 0;
    // First, make sure that we have valid and completed results
    if ( hullCount )
    {
        // See if we already have AABB trees created for each convex hull
        if ( m_trees.empty() )
        {
            // For each convex hull, we generate an AABB tree for fast closest point queries
            for (uint32_t i = 0; i < hullCount; i++)
            {
                VHACD::IVHACD::ConvexHull ch;
                GetConvexHull(i,ch);
                // Pass the triangle mesh to create an AABB tree instance based on it.
                m_trees.emplace_back(new AABBTree(ch.m_points,
                                                  ch.m_triangles));
            }
        }
        // We now compute the closest point to each convex hull and save the nearest one
        double closest = 1e99;
        for (uint32_t i = 0; i < hullCount; i++)
        {
            std::unique_ptr<AABBTree>& t = m_trees[i];
            if ( t )
            {
                VHACD::Vect3 closestPoint;
                VHACD::Vect3 position(pos[0],
                                      pos[1],
                                      pos[2]);
                if ( t->GetClosestPointWithinDistance(position, 1e99, closestPoint))
                {
                    VHACD::Vect3 d = position - closestPoint;
                    double distanceSquared = d.GetNormSquared();
                    if ( distanceSquared < closest )
                    {
                        closest = distanceSquared;
                        ret = i;
                    }
                }
            }
        }
        distanceToHull = sqrt(closest); // compute the distance to the nearest convex hull
    }

    return ret;
}

bool VHACDImpl::Compute(const std::vector<VHACD::Vertex>& points,
                        const std::vector<VHACD::Triangle>& triangles,
                        const Parameters& params)
{
    bool ret = false;

    m_params = params;
    m_canceled = false;

    Clean(); // release any previous results
#if !VHACD_DISABLE_THREADING
    if ( m_params.m_asyncACD )
    {
        m_threadPool = std::unique_ptr<ThreadPool>(new ThreadPool(8));
    }
#endif
    CopyInputMesh(points,
                  triangles);
    if ( !m_canceled )
    {
        // We now recursively perform convex decomposition until complete
        PerformConvexDecomposition();
    }

    if ( m_canceled )
    {
        Clean();
        ret = false;
        if ( m_params.m_logger )
        {
            m_params.m_logger->Log("VHACD operation canceled before it was complete.");
        }
    }
    else
    {
        ret = true;
    }
#if !VHACD_DISABLE_THREADING
    m_threadPool = nullptr;
#endif
    return ret;
}

uint32_t VHACDImpl::GetIndex(VHACD::VertexIndex& vi,
                             const VHACD::Vertex& p)
{
    VHACD::Vect3 pos = (VHACD::Vect3(p) - m_center) * m_recipScale;
    bool newPos;
    uint32_t ret = vi.GetIndex(pos,
                               newPos);
    return ret;
}

void VHACDImpl::CopyInputMesh(const std::vector<VHACD::Vertex>& points,
                              const std::vector<VHACD::Triangle>& triangles)
{
    m_vertices.clear();
    m_indices.clear();
    m_indices.reserve(triangles.size());

    // First we must find the bounding box of this input vertices and normalize them into a unit-cube
    VHACD::Vect3 bmin( FLT_MAX);
    VHACD::Vect3 bmax(-FLT_MAX);
    ProgressUpdate(Stages::COMPUTE_BOUNDS_OF_INPUT_MESH,
                   0,
                   "ComputingBounds");
    for (uint32_t i = 0; i < points.size(); i++)
    {
        const VHACD::Vertex& p = points[i];

        bmin = bmin.CWiseMin(p);
        bmax = bmax.CWiseMax(p);
    }
    ProgressUpdate(Stages::COMPUTE_BOUNDS_OF_INPUT_MESH,
                   100,
                   "ComputingBounds");

    m_center = (bmax + bmin) * double(0.5);

    VHACD::Vect3 scale = bmax - bmin;
    m_scale = scale.MaxCoeff();

    m_recipScale = m_scale > double(0.0) ? double(1.0) / m_scale : double(0.0);

    {
        VHACD::VertexIndex vi = VHACD::VertexIndex(double(0.001), false);

        uint32_t dcount = 0;

        for (uint32_t i = 0; i < triangles.size() && !m_canceled; ++i)
        {
            const VHACD::Triangle& t = triangles[i];
            const VHACD::Vertex& p1 = points[t.mI0];
            const VHACD::Vertex& p2 = points[t.mI1];
            const VHACD::Vertex& p3 = points[t.mI2];

            uint32_t i1 = GetIndex(vi, p1);
            uint32_t i2 = GetIndex(vi, p2);
            uint32_t i3 = GetIndex(vi, p3);

            if ( i1 == i2 || i1 == i3 || i2 == i3 )
            {
                dcount++;
            }
            else
            {
                m_indices.emplace_back(i1, i2, i3);
            }
        }

        if ( dcount )
        {
            if ( m_params.m_logger )
            {
                char scratch[512];
                snprintf(scratch,
                         sizeof(scratch),
                         "Skipped %d degenerate triangles", dcount);
                m_params.m_logger->Log(scratch);
            }
        }

        m_vertices = vi.TakeVertices();
    }

    // Create the raycast mesh
    if ( !m_canceled )
    {
        ProgressUpdate(Stages::CREATE_RAYCAST_MESH,
                       0,
                       "Building RaycastMesh");
        m_AABBTree = VHACD::AABBTree(m_vertices,
                                     m_indices);
        ProgressUpdate(Stages::CREATE_RAYCAST_MESH,
                       100,
                       "RaycastMesh completed");
    }
    if ( !m_canceled )
    {
        ProgressUpdate(Stages::VOXELIZING_INPUT_MESH,
                        0,
                        "Voxelizing Input Mesh");
        m_voxelize = VHACD::Volume();
        m_voxelize.Voxelize(m_vertices,
                            m_indices,
                            m_params.m_resolution,
                            m_params.m_fillMode,
                            m_AABBTree);
        m_voxelScale = m_voxelize.GetScale();
        m_voxelHalfScale = m_voxelScale * double(0.5);
        m_voxelBmin = m_voxelize.GetBounds().GetMin();
        m_voxelBmax = m_voxelize.GetBounds().GetMax();
        ProgressUpdate(Stages::VOXELIZING_INPUT_MESH,
                       100,
                       "Voxelization complete");
    }

    m_inputMesh.m_vertices = m_vertices;
    m_inputMesh.m_indices = m_indices;
    if ( !m_canceled )
    {
        ProgressUpdate(Stages::BUILD_INITIAL_CONVEX_HULL,
                        0,
                        "Build initial ConvexHull");
        std::unique_ptr<VoxelHull> vh = std::unique_ptr<VoxelHull>(new VoxelHull(m_voxelize,
                                                                                 m_params,
                                                                                 this));
        if ( vh->m_convexHull )
        {
            m_overallHullVolume = vh->m_convexHull->m_volume;
        }
        m_pendingHulls.push_back(std::move(vh));
        ProgressUpdate(Stages::BUILD_INITIAL_CONVEX_HULL,
                       100,
                       "Initial ConvexHull complete");
    }
}

void VHACDImpl::ScaleOutputConvexHull(ConvexHull& ch)
{
    for (uint32_t i = 0; i < ch.m_points.size(); i++)
    {
        VHACD::Vect3 p = ch.m_points[i];
        p = (p * m_scale) + m_center;
        ch.m_points[i] = p;
    }
    ch.m_volume = ComputeConvexHullVolume(ch); // get the combined volume
    VHACD::BoundsAABB b(ch.m_points);
    ch.mBmin = b.GetMin();
    ch.mBmax = b.GetMax();
    ComputeCentroid(ch.m_points,
                    ch.m_triangles,
                    ch.m_center);
}

void VHACDImpl::AddCostToPriorityQueue(CostTask& task)
{
    HullPair hp(task.m_hullA->m_meshId,
                task.m_hullB->m_meshId,
                task.m_concavity);
    m_hullPairQueue.push(hp);
}

void VHACDImpl::ReleaseConvexHull(ConvexHull* ch)
{
    if ( ch )
    {
        delete ch;
    }
}

void jobCallback(std::unique_ptr<VoxelHull>& userPtr)
{
    userPtr->PerformPlaneSplit();
}

void computeMergeCostTask(CostTask& ptr)
{
    ptr.m_this->PerformMergeCostTask(ptr);
}

void VHACDImpl::PerformConvexDecomposition()
{
    {
        ScopedTime st("Convex Decomposition",
                      m_params.m_logger);
        double maxHulls = pow(2, m_params.m_maxRecursionDepth);
        // We recursively split convex hulls until we can
        // no longer recurse further.
        Timer t;

        while ( !m_pendingHulls.empty() && !m_canceled )
        {
            size_t count = m_pendingHulls.size() + m_voxelHulls.size();
            double e = t.PeekElapsedSeconds();
            if ( e >= double(0.1) )
            {
                t.Reset();
                double stageProgress = (double(count) * double(100.0)) / maxHulls;
                ProgressUpdate(Stages::PERFORMING_DECOMPOSITION,
                               stageProgress,
                               "Performing recursive decomposition of convex hulls");
            }
            // First we make a copy of the hulls we are processing
            std::vector<std::unique_ptr<VoxelHull>> oldList = std::move(m_pendingHulls);
            // For each hull we want to split, we either
            // immediately perform the plane split or we post it as
            // a job to be performed in a background thread
            std::vector<std::future<void>> futures(oldList.size());
            uint32_t futureCount = 0;
            for (auto& i : oldList)
            {
                if ( i->IsComplete() || count > MaxConvexHullFragments )
                {
                }
                else
                {
#if !VHACD_DISABLE_THREADING
                    if ( m_threadPool )
                    {
                        futures[futureCount] = m_threadPool->enqueue([&i]
                        {
                            jobCallback(i);
                        });
                        futureCount++;
                    }
                    else
#endif
                    {
                        i->PerformPlaneSplit();
                    }
                }
            }
            // Wait for any outstanding jobs to complete in the background threads
            if ( futureCount )
            {
                for (uint32_t i = 0; i < futureCount; i++)
                {
                    futures[i].get();
                }
            }
            // Now, we rebuild the pending convex hulls list by
            // adding the two children to the output list if
            // we need to recurse them further
            for (auto& vh : oldList)
            {
                if ( vh->IsComplete() || count > MaxConvexHullFragments )
                {
                    if ( vh->m_convexHull )
                    {
                        m_voxelHulls.push_back(std::move(vh));
                    }
                }
                else
                {
                    if ( vh->m_hullA )
                    {
                        m_pendingHulls.push_back(std::move(vh->m_hullA));
                    }
                    if ( vh->m_hullB )
                    {
                        m_pendingHulls.push_back(std::move(vh->m_hullB));
                    }
                }
            }
        }
    }

    if ( !m_canceled )
    {
        // Give each convex hull a unique guid
        m_meshId = 0;
        m_hulls.clear();

        // Build the convex hull id map
        std::vector<ConvexHull*> hulls;

        ProgressUpdate(Stages::INITIALIZING_CONVEX_HULLS_FOR_MERGING,
                       0,
                       "Initializing ConvexHulls");
        for (auto& vh : m_voxelHulls)
        {
            if ( m_canceled )
            {
                break;
            }
            ConvexHull* ch = CopyConvexHull(*vh->m_convexHull);
            m_meshId++;
            ch->m_meshId = m_meshId;
            m_hulls[m_meshId] = ch;
            // Compute the volume of the convex hull
            ch->m_volume = ComputeConvexHullVolume(*ch);
            // Compute the AABB of the convex hull
            VHACD::BoundsAABB b = VHACD::BoundsAABB(ch->m_points).Inflate(double(0.1));
            ch->mBmin = b.GetMin();
            ch->mBmax = b.GetMax();

            ComputeCentroid(ch->m_points,
                            ch->m_triangles,
                            ch->m_center);

            hulls.push_back(ch);
        }
        ProgressUpdate(Stages::INITIALIZING_CONVEX_HULLS_FOR_MERGING,
                        100,
                        "ConvexHull initialization complete");

        m_voxelHulls.clear();

        // here we merge convex hulls as needed until the match the
        // desired maximum hull count.
        size_t hullCount = hulls.size();

        if ( hullCount > m_params.m_maxConvexHulls && !m_canceled)
        {
            size_t costMatrixSize = ((hullCount * hullCount) - hullCount) >> 1;
            std::vector<CostTask> tasks;
            tasks.reserve(costMatrixSize);

            ScopedTime st("Computing the Cost Matrix",
                          m_params.m_logger);
            // First thing we need to do is compute the cost matrix
            // This is computed as the volume error of any two convex hulls
            // combined
            ProgressUpdate(Stages::COMPUTING_COST_MATRIX,
                           0,
                           "Computing Hull Merge Cost Matrix");
            for (size_t i = 1; i < hullCount && !m_canceled; i++)
            {
                ConvexHull* chA = hulls[i];

                for (size_t j = 0; j < i && !m_canceled; j++)
                {
                    ConvexHull* chB = hulls[j];

                    CostTask t;
                    t.m_hullA = chA;
                    t.m_hullB = chB;
                    t.m_this = this;

                    if ( DoFastCost(t) )
                    {
                    }
                    else
                    {
                        tasks.push_back(std::move(t));
                        CostTask* task = &tasks.back();
#if !VHACD_DISABLE_THREADING
                        if ( m_threadPool )
                        {
                            task->m_future = m_threadPool->enqueue([task]
                            {
                                computeMergeCostTask(*task);
                            });
                        }
#endif
                    }
                }
            }

            if ( !m_canceled )
            {
#if !VHACD_DISABLE_THREADING
                if ( m_threadPool )
                {
                    for (CostTask& task : tasks)
                    {
                        task.m_future.get();
                    }

                    for (CostTask& task : tasks)
                    {
                        AddCostToPriorityQueue(task);
                    }
                }
                else
#endif
                {
                    for (CostTask& task : tasks)
                    {
                        PerformMergeCostTask(task);
                        AddCostToPriorityQueue(task);
                    }
                }
                ProgressUpdate(Stages::COMPUTING_COST_MATRIX,
                               100,
                               "Finished cost matrix");
            }

            if ( !m_canceled )
            {
                ScopedTime stMerging("Merging Convex Hulls",
                                     m_params.m_logger);
                Timer t;
                // Now that we know the cost to merge each hull, we can begin merging them.
                bool cancel = false;

                uint32_t maxMergeCount = uint32_t(m_hulls.size()) - m_params.m_maxConvexHulls;
                uint32_t startCount = uint32_t(m_hulls.size());

                while (    !cancel
                        && m_hulls.size() > m_params.m_maxConvexHulls
                        && !m_hullPairQueue.empty()
                        && !m_canceled)
                {
                    double e = t.PeekElapsedSeconds();
                    if ( e >= double(0.1) )
                    {
                        t.Reset();
                        uint32_t hullsProcessed = startCount - uint32_t(m_hulls.size() );
                        double stageProgress = double(hullsProcessed * 100) / double(maxMergeCount);
                        ProgressUpdate(Stages::MERGING_CONVEX_HULLS,
                                       stageProgress,
                                       "Merging Convex Hulls");
                    }

                    HullPair hp = m_hullPairQueue.top();
                    m_hullPairQueue.pop();

                    // It is entirely possible that the hull pair queue can
                    // have references to convex hulls that are no longer valid
                    // because they were previously merged. So we check for this
                    // and if either hull referenced in this pair no longer
                    // exists, then we skip it.

                    // Look up this pair of hulls by ID
                    ConvexHull* ch1 = GetHull(hp.m_hullA);
                    ConvexHull* ch2 = GetHull(hp.m_hullB);

                    // If both hulls are still valid, then we merge them, delete the old
                    // two hulls and recompute the cost matrix for the new combined hull
                    // we have created
                    if ( ch1 && ch2 )
                    {
                        // This is the convex hull which results from combining the
                        // vertices in the two source hulls
                        ConvexHull* combinedHull = ComputeCombinedConvexHull(*ch1,
                                                                                *ch2);
                        // The two old convex hulls are going to get removed
                        RemoveHull(hp.m_hullA);
                        RemoveHull(hp.m_hullB);

                        m_meshId++;
                        combinedHull->m_meshId = m_meshId;
                        tasks.clear();
                        tasks.reserve(m_hulls.size());

                        // Compute the cost between this new merged hull
                        // and all existing convex hulls and then
                        // add that to the priority queue
                        for (auto& i : m_hulls)
                        {
                            if ( m_canceled )
                            {
                                break;
                            }
                            ConvexHull* secondHull = i.second;
                            CostTask ct;
                            ct.m_hullA = combinedHull;
                            ct.m_hullB = secondHull;
                            ct.m_this = this;
                            if ( DoFastCost(ct) )
                            {
                            }
                            else
                            {
                                tasks.push_back(std::move(ct));
                            }
                        }
                        m_hulls[combinedHull->m_meshId] = combinedHull;
                        // See how many merge cost tasks were posted
                        // If there are 8 or more and we are running asynchronously, then do them that way.
#if !VHACD_DISABLE_THREADING
                        if ( m_threadPool && tasks.size() >= 2)
                        {
                            for (CostTask& task : tasks)
                            {
                                task.m_future = m_threadPool->enqueue([&task]
                                {
                                    computeMergeCostTask(task);
                                });
                            }

                            for (CostTask& task : tasks)
                            {
                                task.m_future.get();
                            }
                        }
                        else
#endif
                        {
                            for (CostTask& task : tasks)
                            {
                                PerformMergeCostTask(task);
                            }
                        }

                        for (CostTask& task : tasks)
                        {
                            AddCostToPriorityQueue(task);
                        }
                    }
                }
                // Ok...once we are done, we copy the results!
                m_meshId -= 0;
                ProgressUpdate(Stages::FINALIZING_RESULTS,
                               0,
                               "Finalizing results");
                for (auto& i : m_hulls)
                {
                    if ( m_canceled )
                    {
                        break;
                    }
                    ConvexHull* ch = i.second;
                    // We now must reduce the convex hull
                    if ( ch->m_points.size() > m_params.m_maxNumVerticesPerCH || m_params.m_shrinkWrap)
                    {
                        ConvexHull* reduce = ComputeReducedConvexHull(*ch,
                                                                      m_params.m_maxNumVerticesPerCH,
                                                                      m_params.m_shrinkWrap);
                        ReleaseConvexHull(ch);
                        ch = reduce;
                    }
                    ScaleOutputConvexHull(*ch);
                    ch->m_meshId = m_meshId;
                    m_meshId++;
                    m_convexHulls.push_back(ch);
                }
                m_hulls.clear(); // since the hulls were moved into the output list, we don't need to delete them from this container
                ProgressUpdate(Stages::FINALIZING_RESULTS,
                               100,
                               "Finalized results complete");
            }
        }
        else
        {
            ProgressUpdate(Stages::FINALIZING_RESULTS,
                           0,
                           "Finalizing results");
            m_meshId = 0;
            for (auto& ch : hulls)
            {
                // We now must reduce the convex hull
                if ( ch->m_points.size() > m_params.m_maxNumVerticesPerCH  || m_params.m_shrinkWrap )
                {
                    ConvexHull* reduce = ComputeReducedConvexHull(*ch,
                                                                  m_params.m_maxNumVerticesPerCH,
                                                                  m_params.m_shrinkWrap);
                    ReleaseConvexHull(ch);
                    ch = reduce;
                }
                ScaleOutputConvexHull(*ch);
                ch->m_meshId = m_meshId;
                m_meshId++;
                m_convexHulls.push_back(ch);
            }
            m_hulls.clear();
            ProgressUpdate(Stages::FINALIZING_RESULTS,
                           100,
                           "Finalized results");
        }
    }
}

double VHACDImpl::ComputeConvexHullVolume(const ConvexHull& sm)
{
    double totalVolume = 0;
    VHACD::Vect3 bary(0, 0, 0);
    for (uint32_t i = 0; i < sm.m_points.size(); i++)
    {
        VHACD::Vect3 p(sm.m_points[i]);
        bary += p;
    }
    bary /= double(sm.m_points.size());

    for (uint32_t i = 0; i < sm.m_triangles.size(); i++)
    {
        uint32_t i1 = sm.m_triangles[i].mI0;
        uint32_t i2 = sm.m_triangles[i].mI1;
        uint32_t i3 = sm.m_triangles[i].mI2;

        VHACD::Vect3 ver0(sm.m_points[i1]);
        VHACD::Vect3 ver1(sm.m_points[i2]);
        VHACD::Vect3 ver2(sm.m_points[i3]);

        totalVolume += ComputeVolume4(ver0,
                                      ver1,
                                      ver2,
                                      bary);

    }
    totalVolume = totalVolume / double(6.0);
    return totalVolume;
}

double VHACDImpl::ComputeVolume4(const VHACD::Vect3& a,
                                 const VHACD::Vect3& b,
                                 const VHACD::Vect3& c,
                                 const VHACD::Vect3& d)
{
    VHACD::Vect3 ad = a - d;
    VHACD::Vect3 bd = b - d;
    VHACD::Vect3 cd = c - d;
    VHACD::Vect3 bcd = bd.Cross(cd);
    double dot = ad.Dot(bcd);
    return dot;
}

double VHACDImpl::ComputeConcavity(double volumeSeparate,
                                   double volumeCombined,
                                   double volumeMesh)
{
    return fabs(volumeSeparate - volumeCombined) / volumeMesh;
}

bool VHACDImpl::DoFastCost(CostTask& mt)
{
    bool ret = false;

    ConvexHull* ch1 = mt.m_hullA;
    ConvexHull* ch2 = mt.m_hullB;

    VHACD::BoundsAABB ch1b(ch1->mBmin,
                           ch1->mBmax);
    VHACD::BoundsAABB ch2b(ch2->mBmin,
                           ch2->mBmax);
    if (!ch1b.Intersects(ch2b))
    {
        VHACD::BoundsAABB b = ch1b.Union(ch2b);

        double combinedVolume = b.Volume();
        double concavity = ComputeConcavity(ch1->m_volume + ch2->m_volume,
                                            combinedVolume,
                                            m_overallHullVolume);
        HullPair hp(ch1->m_meshId,
                    ch2->m_meshId,
                    concavity);
        m_hullPairQueue.push(hp);
        ret = true;
    }
    return ret;
}

void VHACDImpl::PerformMergeCostTask(CostTask& mt)
{
    ConvexHull* ch1 = mt.m_hullA;
    ConvexHull* ch2 = mt.m_hullB;

    double volume1 = ch1->m_volume;
    double volume2 = ch2->m_volume;

    ConvexHull* combined = ComputeCombinedConvexHull(*ch1,
                                                     *ch2); // Build the combined convex hull
    double combinedVolume = ComputeConvexHullVolume(*combined); // get the combined volume
    mt.m_concavity = ComputeConcavity(volume1 + volume2,
                                      combinedVolume,
                                      m_overallHullVolume);
    ReleaseConvexHull(combined);
}

IVHACD::ConvexHull* VHACDImpl::ComputeReducedConvexHull(const ConvexHull& ch,
                                                        uint32_t maxVerts,
                                                        bool projectHullVertices)
{
    SimpleMesh sourceConvexHull;

    sourceConvexHull.m_vertices = ch.m_points;
    sourceConvexHull.m_indices = ch.m_triangles;

    ShrinkWrap(sourceConvexHull,
               m_AABBTree,
               maxVerts,
               m_voxelScale * 4,
               projectHullVertices);

    ConvexHull *ret = new ConvexHull;

    ret->m_points = sourceConvexHull.m_vertices;
    ret->m_triangles = sourceConvexHull.m_indices;

    VHACD::BoundsAABB b = VHACD::BoundsAABB(ret->m_points).Inflate(double(0.1));
    ret->mBmin = b.GetMin();
    ret->mBmax = b.GetMax();
    ComputeCentroid(ret->m_points,
                    ret->m_triangles,
                    ret->m_center);

    ret->m_volume = ComputeConvexHullVolume(*ret);

    // Return the convex hull
    return ret;
}

IVHACD::ConvexHull* VHACDImpl::ComputeCombinedConvexHull(const ConvexHull& sm1,
                                                         const ConvexHull& sm2)
{
    uint32_t vcount = uint32_t(sm1.m_points.size() + sm2.m_points.size()); // Total vertices from both hulls
    std::vector<VHACD::Vertex> vertices(vcount);
    auto it = std::copy(sm1.m_points.begin(),
                        sm1.m_points.end(),
                        vertices.begin());
    std::copy(sm2.m_points.begin(),
                sm2.m_points.end(),
                it);

    VHACD::QuickHull qh;
    qh.ComputeConvexHull(vertices,
                         vcount);

    ConvexHull* ret = new ConvexHull;
    ret->m_points = qh.GetVertices();
    ret->m_triangles = qh.GetIndices();

    ret->m_volume = ComputeConvexHullVolume(*ret);

    VHACD::BoundsAABB b = VHACD::BoundsAABB(qh.GetVertices()).Inflate(double(0.1));
    ret->mBmin = b.GetMin();
    ret->mBmax = b.GetMax();
    ComputeCentroid(ret->m_points,
                    ret->m_triangles,
                    ret->m_center);

    // Return the convex hull
    return ret;
}

IVHACD::ConvexHull* VHACDImpl::GetHull(uint32_t index)
{
    ConvexHull* ret = nullptr;

    auto found = m_hulls.find(index);
    if ( found != m_hulls.end() )
    {
        ret = found->second;
    }

    return ret;
}

bool VHACDImpl::RemoveHull(uint32_t index)
{
    bool ret = false;
    auto found = m_hulls.find(index);
    if ( found != m_hulls.end() )
    {
        ret = true;
        ReleaseConvexHull(found->second);
        m_hulls.erase(found);
    }
    return ret;
}

IVHACD::ConvexHull* VHACDImpl::CopyConvexHull(const ConvexHull& source)
{
    ConvexHull *ch = new ConvexHull;
    *ch = source;

    return ch;
}

const char* VHACDImpl::GetStageName(Stages stage) const
{
    const char *ret = "unknown";
    switch ( stage )
    {
        case Stages::COMPUTE_BOUNDS_OF_INPUT_MESH:
            ret = "COMPUTE_BOUNDS_OF_INPUT_MESH";
            break;
        case Stages::REINDEXING_INPUT_MESH:
            ret = "REINDEXING_INPUT_MESH";
            break;
        case Stages::CREATE_RAYCAST_MESH:
            ret = "CREATE_RAYCAST_MESH";
            break;
        case Stages::VOXELIZING_INPUT_MESH:
            ret = "VOXELIZING_INPUT_MESH";
            break;
        case Stages::BUILD_INITIAL_CONVEX_HULL:
            ret = "BUILD_INITIAL_CONVEX_HULL";
            break;
        case Stages::PERFORMING_DECOMPOSITION:
            ret = "PERFORMING_DECOMPOSITION";
            break;
        case Stages::INITIALIZING_CONVEX_HULLS_FOR_MERGING:
            ret = "INITIALIZING_CONVEX_HULLS_FOR_MERGING";
            break;
        case Stages::COMPUTING_COST_MATRIX:
            ret = "COMPUTING_COST_MATRIX";
            break;
        case Stages::MERGING_CONVEX_HULLS:
            ret = "MERGING_CONVEX_HULLS";
            break;
        case Stages::FINALIZING_RESULTS:
            ret = "FINALIZING_RESULTS";
            break;
        case Stages::NUM_STAGES:
            // Should be unreachable, here to silence enumeration value not handled in switch warnings
            // GCC/Clang's -Wswitch
            break;
    }
    return ret;
}

void VHACDImpl::ProgressUpdate(Stages stage,
                               double stageProgress,
                               const char* operation)
{
    if ( m_params.m_callback )
    {
        double overallProgress = (double(stage) * 100) / double(Stages::NUM_STAGES);
        const char *s = GetStageName(stage);
        m_params.m_callback->Update(overallProgress,
                                    stageProgress,
                                    s,
                                    operation);
    }
}

bool VHACDImpl::IsCanceled() const
{
    return m_canceled;
}

IVHACD* CreateVHACD(void)
{
    VHACDImpl *ret = new VHACDImpl;
    return static_cast< IVHACD *>(ret);
}

IVHACD* CreateVHACD(void);

#if !VHACD_DISABLE_THREADING

class LogMessage
{
public:
    double  m_overallProgress{ double(-1.0) };
    double  m_stageProgress{ double(-1.0) };
    std::string m_stage;
    std::string m_operation;
};

class VHACDAsyncImpl : public VHACD::IVHACD,
                       public VHACD::IVHACD::IUserCallback,
                       VHACD::IVHACD::IUserLogger,
                       VHACD::IVHACD::IUserTaskRunner
{
public:
    VHACDAsyncImpl() = default;

    ~VHACDAsyncImpl() override;

    void Cancel() override final;

    bool Compute(const float* const points,
                 const uint32_t countPoints,
                 const uint32_t* const triangles,
                 const uint32_t countTriangles,
                 const Parameters& params) override final;

    bool Compute(const double* const points,
                 const uint32_t countPoints,
                 const uint32_t* const triangles,
                 const uint32_t countTriangles,
                 const Parameters& params) override final;

    bool GetConvexHull(const uint32_t index,
                       VHACD::IVHACD::ConvexHull& ch) const override final;

    uint32_t GetNConvexHulls() const override final;

    void Clean() override final; // release internally allocated memory

    void Release() override final; // release IVHACD

    // Will compute the center of mass of the convex hull decomposition results and return it
    // in 'centerOfMass'.  Returns false if the center of mass could not be computed.
    bool ComputeCenterOfMass(double centerOfMass[3]) const override;

    bool IsReady() const override final;

    /**
    * At the request of LegionFu : out_look@foxmail.com
    * This method will return which convex hull is closest to the source position.
    * You can use this method to figure out, for example, which vertices in the original
    * source mesh are best associated with which convex hull.
    *
    * @param pos : The input 3d position to test against
    *
    * @return : Returns which convex hull this position is closest to.
    */
    uint32_t findNearestConvexHull(const double pos[3],
                                   double& distanceToHull) override final;

    void Update(const double overallProgress,
                const double stageProgress,
                const char* const stage,
                const char *operation) override final;

    void Log(const char* const msg) override final;

    void* StartTask(std::function<void()> func) override;

    void JoinTask(void* Task) override;

    bool Compute(const Parameters params);

    bool ComputeNow(const std::vector<VHACD::Vertex>& points,
                    const std::vector<VHACD::Triangle>& triangles,
                    const Parameters& _desc);

    // As a convenience for the calling application we only send it update and log messages from it's own main
    // thread.  This reduces the complexity burden on the caller by making sure it only has to deal with log
    // messages in it's main application thread.
    void ProcessPendingMessages() const;

private:
    VHACD::VHACDImpl                m_VHACD;
    std::vector<VHACD::Vertex>      m_vertices;
    std::vector<VHACD::Triangle>    m_indices;
    VHACD::IVHACD::IUserCallback*   m_callback{ nullptr };
    VHACD::IVHACD::IUserLogger*     m_logger{ nullptr };
    VHACD::IVHACD::IUserTaskRunner* m_taskRunner{ nullptr };
    void*                           m_task{ nullptr };
    std::atomic<bool>               m_running{ false };
    std::atomic<bool>               m_cancel{ false };

    // Thread safe caching mechanism for messages and update status.
    // This is so that caller always gets messages in his own thread
    // Member variables are marked as 'mutable' since the message dispatch function
    // is called from const query methods.
    mutable std::mutex              m_messageMutex;
    mutable std::vector<LogMessage> m_messages;
    mutable std::atomic<bool>       m_haveMessages{ false };
};

VHACDAsyncImpl::~VHACDAsyncImpl()
{
    Cancel();
}

void VHACDAsyncImpl::Cancel()
{
    m_cancel = true;
    m_VHACD.Cancel();

    if (m_task)
    {
        m_taskRunner->JoinTask(m_task); // Wait for the thread to fully exit before we delete the instance
        m_task = nullptr;
    }
    m_cancel = false; // clear the cancel semaphore
}

bool VHACDAsyncImpl::Compute(const float* const points,
                             const uint32_t countPoints,
                             const uint32_t* const triangles,
                             const uint32_t countTriangles,
                             const Parameters& params)
{
    m_vertices.reserve(countPoints);
    for (uint32_t i = 0; i < countPoints; ++i)
    {
        m_vertices.emplace_back(points[i * 3 + 0],
                                points[i * 3 + 1],
                                points[i * 3 + 2]);
    }

    m_indices.reserve(countTriangles);
    for (uint32_t i = 0; i < countTriangles; ++i)
    {
        m_indices.emplace_back(triangles[i * 3 + 0],
                               triangles[i * 3 + 1],
                               triangles[i * 3 + 2]);
    }

    return Compute(params);
}

bool VHACDAsyncImpl::Compute(const double* const points,
                             const uint32_t countPoints,
                             const uint32_t* const triangles,
                             const uint32_t countTriangles,
                             const Parameters& params)
{
    // We need to copy the input vertices and triangles into our own buffers so we can operate
    // on them safely from the background thread.
    // Can't be local variables due to being asynchronous
    m_vertices.reserve(countPoints);
    for (uint32_t i = 0; i < countPoints; ++i)
    {
        m_vertices.emplace_back(points[i * 3 + 0],
                                points[i * 3 + 1],
                                points[i * 3 + 2]);
    }

    m_indices.reserve(countTriangles);
    for (uint32_t i = 0; i < countTriangles; ++i)
    {
        m_indices.emplace_back(triangles[i * 3 + 0],
                               triangles[i * 3 + 1],
                               triangles[i * 3 + 2]);
    }

    return Compute(params);
}

bool VHACDAsyncImpl::GetConvexHull(const uint32_t index,
                                   VHACD::IVHACD::ConvexHull& ch) const
{
    return m_VHACD.GetConvexHull(index,
                                 ch);
}

uint32_t VHACDAsyncImpl::GetNConvexHulls() const
{
    ProcessPendingMessages();
    return m_VHACD.GetNConvexHulls();
}

void VHACDAsyncImpl::Clean()
{
    Cancel();
    m_VHACD.Clean();
}

void VHACDAsyncImpl::Release()
{
    delete this;
}

bool VHACDAsyncImpl::ComputeCenterOfMass(double centerOfMass[3]) const
{
    bool ret = false;

    centerOfMass[0] = 0;
    centerOfMass[1] = 0;
    centerOfMass[2] = 0;

    if (IsReady())
    {
        ret = m_VHACD.ComputeCenterOfMass(centerOfMass);
    }
    return ret;
}

bool VHACDAsyncImpl::IsReady() const
{
    ProcessPendingMessages();
    return !m_running;
}

uint32_t VHACDAsyncImpl::findNearestConvexHull(const double pos[3],
                                               double& distanceToHull)
{
    uint32_t ret = 0; // The default return code is zero

    distanceToHull = 0;
    // First, make sure that we have valid and completed results
    if (IsReady() )
    {
        ret = m_VHACD.findNearestConvexHull(pos,distanceToHull);
    }

    return ret;
}

void VHACDAsyncImpl::Update(const double overallProgress,
                            const double stageProgress,
                            const char* const stage,
                            const char* operation)
{
    m_messageMutex.lock();
    LogMessage m;
    m.m_operation = std::string(operation);
    m.m_overallProgress = overallProgress;
    m.m_stageProgress = stageProgress;
    m.m_stage = std::string(stage);
    m_messages.push_back(m);
    m_haveMessages = true;
    m_messageMutex.unlock();
}

void VHACDAsyncImpl::Log(const char* const msg)
{
    m_messageMutex.lock();
    LogMessage m;
    m.m_operation = std::string(msg);
    m_haveMessages = true;
    m_messages.push_back(m);
    m_messageMutex.unlock();
}

void* VHACDAsyncImpl::StartTask(std::function<void()> func)
{
    return new std::thread(func);
}

void VHACDAsyncImpl::JoinTask(void* Task)
{
    std::thread* t = static_cast<std::thread*>(Task);
    t->join();
    delete t;
}

bool VHACDAsyncImpl::Compute(Parameters params)
{
    Cancel(); // if we previously had a solution running; cancel it.

    m_taskRunner = params.m_taskRunner ? params.m_taskRunner : this;
    params.m_taskRunner = m_taskRunner;

    m_running = true;
    m_task = m_taskRunner->StartTask([this, params]() {
        ComputeNow(m_vertices,
                   m_indices,
                   params);
        // If we have a user provided callback and the user did *not* call 'cancel' we notify him that the
        // task is completed. However..if the user selected 'cancel' we do not send a completed notification event.
        if (params.m_callback && !m_cancel)
        {
            params.m_callback->NotifyVHACDComplete();
        }
        m_running = false;
    });
    return true;
}

bool VHACDAsyncImpl::ComputeNow(const std::vector<VHACD::Vertex>& points,
                                const std::vector<VHACD::Triangle>& triangles,
                                const Parameters& _desc)
{
    uint32_t ret = 0;

    Parameters desc;
    m_callback = _desc.m_callback;
    m_logger = _desc.m_logger;

    desc = _desc;
    // Set our intercepting callback interfaces if non-null
    desc.m_callback = _desc.m_callback ? this : nullptr;
    desc.m_logger = _desc.m_logger ? this : nullptr;

    // If not task runner provided, then use the default one
    if (desc.m_taskRunner == nullptr)
    {
        desc.m_taskRunner = this;
    }

    bool ok = m_VHACD.Compute(points,
                              triangles,
                              desc);
    if (ok)
    {
        ret = m_VHACD.GetNConvexHulls();
    }

    return ret ? true : false;
}

void VHACDAsyncImpl::ProcessPendingMessages() const
{
    if (m_cancel)
    {
        return;
    }
    if ( m_haveMessages )
    {
        m_messageMutex.lock();
        for (auto& i : m_messages)
        {
            if ( i.m_overallProgress == -1 )
            {
                if ( m_logger )
                {
                    m_logger->Log(i.m_operation.c_str());
                }
            }
            else if ( m_callback )
            {
                m_callback->Update(i.m_overallProgress,
                                   i.m_stageProgress,
                                   i.m_stage.c_str(),
                                   i.m_operation.c_str());
            }
        }
        m_messages.clear();
        m_haveMessages = false;
        m_messageMutex.unlock();
    }
}

IVHACD* CreateVHACD_ASYNC()
{
    VHACDAsyncImpl* m = new VHACDAsyncImpl;
    return static_cast<IVHACD*>(m);
}
#endif

} // namespace VHACD

#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif // __GNUC__

#endif // ENABLE_VHACD_IMPLEMENTATION

#endif // VHACD_H
