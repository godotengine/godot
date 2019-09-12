/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2019, assimp team


All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the
following conditions are met:

* Redistributions of source code must retain the above
  copyright notice, this list of conditions and the
  following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the
  following disclaimer in the documentation and/or other
  materials provided with the distribution.

* Neither the name of the assimp team, nor the names of its
  contributors may be used to endorse or promote products
  derived from this software without specific prior
  written permission of the assimp team.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

----------------------------------------------------------------------
*/

/** @file Base class of all import post processing steps */
#ifndef INCLUDED_AI_BASEPROCESS_H
#define INCLUDED_AI_BASEPROCESS_H

#include <map>
#include <assimp/GenericProperty.h>

struct aiScene;

namespace Assimp    {

class Importer;

// ---------------------------------------------------------------------------
/** Helper class to allow post-processing steps to interact with each other.
 *
 *  The class maintains a simple property list that can be used by pp-steps
 *  to provide additional information to other steps. This is primarily
 *  intended for cross-step optimizations.
 */
class SharedPostProcessInfo
{
public:

    struct Base
    {
        virtual ~Base()
        {}
    };

    //! Represents data that is allocated on the heap, thus needs to be deleted
    template <typename T>
    struct THeapData : public Base
    {
        explicit THeapData(T* in)
            : data (in)
        {}

        ~THeapData()
        {
            delete data;
        }
        T* data;
    };

    //! Represents static, by-value data not allocated on the heap
    template <typename T>
    struct TStaticData : public Base
    {
        explicit TStaticData(T in)
            : data (in)
        {}

        ~TStaticData()
        {}

        T data;
    };

    // some typedefs for cleaner code
    typedef unsigned int KeyType;
    typedef std::map<KeyType, Base*>  PropertyMap;

public:

    //! Destructor
    ~SharedPostProcessInfo()
    {
        Clean();
    }

    //! Remove all stored properties from the table
    void Clean()
    {
        // invoke the virtual destructor for all stored properties
        for (PropertyMap::iterator it = pmap.begin(), end = pmap.end();
             it != end; ++it)
        {
            delete (*it).second;
        }
        pmap.clear();
    }

    //! Add a heap property to the list
    template <typename T>
    void AddProperty( const char* name, T* in ){
        AddProperty(name,(Base*)new THeapData<T>(in));
    }

    //! Add a static by-value property to the list
    template <typename T>
    void AddProperty( const char* name, T in ){
        AddProperty(name,(Base*)new TStaticData<T>(in));
    }


    //! Get a heap property
    template <typename T>
    bool GetProperty( const char* name, T*& out ) const
    {
        THeapData<T>* t = (THeapData<T>*)GetPropertyInternal(name);
        if(!t)
        {
            out = NULL;
            return false;
        }
        out = t->data;
        return true;
    }

    //! Get a static, by-value property
    template <typename T>
    bool GetProperty( const char* name, T& out ) const
    {
        TStaticData<T>* t = (TStaticData<T>*)GetPropertyInternal(name);
        if(!t)return false;
        out = t->data;
        return true;
    }

    //! Remove a property of a specific type
    void RemoveProperty( const char* name)  {
        SetGenericPropertyPtr<Base>(pmap,name,NULL);
    }

private:

    void AddProperty( const char* name, Base* data) {
        SetGenericPropertyPtr<Base>(pmap,name,data);
    }

    Base* GetPropertyInternal( const char* name) const  {
        return GetGenericProperty<Base*>(pmap,name,NULL);
    }

private:

    //! Map of all stored properties
    PropertyMap pmap;
};

#if 0

// ---------------------------------------------------------------------------
/** @brief Represents a dependency table for a postprocessing steps.
 *
 *  For future use.
 */
 struct PPDependencyTable
 {
     unsigned int execute_me_before_these;
     unsigned int execute_me_after_these;
     unsigned int only_if_these_are_not_specified;
     unsigned int mutually_exclusive_with;
 };

#endif


#define AI_SPP_SPATIAL_SORT "$Spat"

// ---------------------------------------------------------------------------
/** The BaseProcess defines a common interface for all post processing steps.
 * A post processing step is run after a successful import if the caller
 * specified the corresponding flag when calling ReadFile().
 * Enum #aiPostProcessSteps defines which flags are available.
 * After a successful import the Importer iterates over its internal array
 * of processes and calls IsActive() on each process to evaluate if the step
 * should be executed. If the function returns true, the class' Execute()
 * function is called subsequently.
 */
class ASSIMP_API_WINONLY BaseProcess {
    friend class Importer;

public:
    /** Constructor to be privately used by Importer */
    BaseProcess() AI_NO_EXCEPT;

    /** Destructor, private as well */
    virtual ~BaseProcess();

    // -------------------------------------------------------------------
    /** Returns whether the processing step is present in the given flag.
     * @param pFlags The processing flags the importer was called with. A
     *   bitwise combination of #aiPostProcessSteps.
     * @return true if the process is present in this flag fields,
     *   false if not.
    */
    virtual bool IsActive( unsigned int pFlags) const = 0;

    // -------------------------------------------------------------------
    /** Check whether this step expects its input vertex data to be
     *  in verbose format. */
    virtual bool RequireVerboseFormat() const;

    // -------------------------------------------------------------------
    /** Executes the post processing step on the given imported data.
    * The function deletes the scene if the postprocess step fails (
    * the object pointer will be set to NULL).
    * @param pImp Importer instance (pImp->mScene must be valid)
    */
    void ExecuteOnScene( Importer* pImp);

    // -------------------------------------------------------------------
    /** Called prior to ExecuteOnScene().
    * The function is a request to the process to update its configuration
    * basing on the Importer's configuration property list.
    */
    virtual void SetupProperties(const Importer* pImp);

    // -------------------------------------------------------------------
    /** Executes the post processing step on the given imported data.
    * A process should throw an ImportErrorException* if it fails.
    * This method must be implemented by deriving classes.
    * @param pScene The imported data to work at.
    */
    virtual void Execute( aiScene* pScene) = 0;


    // -------------------------------------------------------------------
    /** Assign a new SharedPostProcessInfo to the step. This object
     *  allows multiple postprocess steps to share data.
     * @param sh May be NULL
    */
    inline void SetSharedData(SharedPostProcessInfo* sh)    {
        shared = sh;
    }

    // -------------------------------------------------------------------
    /** Get the shared data that is assigned to the step.
    */
    inline SharedPostProcessInfo* GetSharedData()   {
        return shared;
    }

protected:

    /** See the doc of #SharedPostProcessInfo for more details */
    SharedPostProcessInfo* shared;

    /** Currently active progress handler */
    ProgressHandler* progress;
};


} // end of namespace Assimp

#endif // AI_BASEPROCESS_H_INC
