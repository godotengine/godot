//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#ifndef _SHHANDLE_INCLUDED_
#define _SHHANDLE_INCLUDED_

//
// Machine independent part of the compiler private objects
// sent as ShHandle to the driver.
//
// This should not be included by driver code.
//

#define SH_EXPORTING
#include "../Public/ShaderLang.h"
#include "../MachineIndependent/Versions.h"
#include "InfoSink.h"

class TCompiler;
class TLinker;
class TUniformMap;

//
// The base class used to back handles returned to the driver.
//
class TShHandleBase {
public:
    TShHandleBase() { pool = new glslang::TPoolAllocator; }
    virtual ~TShHandleBase() { delete pool; }
    virtual TCompiler* getAsCompiler() { return 0; }
    virtual TLinker* getAsLinker() { return 0; }
    virtual TUniformMap* getAsUniformMap() { return 0; }
    virtual glslang::TPoolAllocator* getPool() const { return pool; }
private:
    glslang::TPoolAllocator* pool;
};

//
// The base class for the machine dependent linker to derive from
// for managing where uniforms live.
//
class TUniformMap : public TShHandleBase {
public:
    TUniformMap() { }
    virtual ~TUniformMap() { }
    virtual TUniformMap* getAsUniformMap() { return this; }
    virtual int getLocation(const char* name) = 0;
    virtual TInfoSink& getInfoSink() { return infoSink; }
    TInfoSink infoSink;
};

class TIntermNode;

//
// The base class for the machine dependent compiler to derive from
// for managing object code from the compile.
//
class TCompiler : public TShHandleBase {
public:
    TCompiler(EShLanguage l, TInfoSink& sink) : infoSink(sink) , language(l), haveValidObjectCode(false) { }
    virtual ~TCompiler() { }
    EShLanguage getLanguage() { return language; }
    virtual TInfoSink& getInfoSink() { return infoSink; }

    virtual bool compile(TIntermNode* root, int version = 0, EProfile profile = ENoProfile) = 0;

    virtual TCompiler* getAsCompiler() { return this; }
    virtual bool linkable() { return haveValidObjectCode; }

    TInfoSink& infoSink;
protected:
    TCompiler& operator=(TCompiler&);

    EShLanguage language;
    bool haveValidObjectCode;
};

//
// Link operations are based on a list of compile results...
//
typedef glslang::TVector<TCompiler*> TCompilerList;
typedef glslang::TVector<TShHandleBase*> THandleList;

//
// The base class for the machine dependent linker to derive from
// to manage the resulting executable.
//

class TLinker : public TShHandleBase {
public:
    TLinker(EShExecutable e, TInfoSink& iSink) :
        infoSink(iSink),
        executable(e),
        haveReturnableObjectCode(false),
        appAttributeBindings(0),
        fixedAttributeBindings(0),
        excludedAttributes(0),
        excludedCount(0),
        uniformBindings(0) { }
    virtual TLinker* getAsLinker() { return this; }
    virtual ~TLinker() { }
    virtual bool link(TCompilerList&, TUniformMap*) = 0;
    virtual bool link(THandleList&) { return false; }
    virtual void setAppAttributeBindings(const ShBindingTable* t)   { appAttributeBindings = t; }
    virtual void setFixedAttributeBindings(const ShBindingTable* t) { fixedAttributeBindings = t; }
    virtual void getAttributeBindings(ShBindingTable const **t) const = 0;
    virtual void setExcludedAttributes(const int* attributes, int count) { excludedAttributes = attributes; excludedCount = count; }
    virtual ShBindingTable* getUniformBindings() const  { return uniformBindings; }
    virtual const void* getObjectCode() const { return 0; } // a real compiler would be returning object code here
    virtual TInfoSink& getInfoSink() { return infoSink; }
    TInfoSink& infoSink;
protected:
    TLinker& operator=(TLinker&);
    EShExecutable executable;
    bool haveReturnableObjectCode;  // true when objectCode is acceptable to send to driver

    const ShBindingTable* appAttributeBindings;
    const ShBindingTable* fixedAttributeBindings;
    const int* excludedAttributes;
    int excludedCount;
    ShBindingTable* uniformBindings;                // created by the linker
};

//
// This is the interface between the machine independent code
// and the machine dependent code.
//
// The machine dependent code should derive from the classes
// above. Then Construct*() and Delete*() will create and
// destroy the machine dependent objects, which contain the
// above machine independent information.
//
TCompiler* ConstructCompiler(EShLanguage, int);

TShHandleBase* ConstructLinker(EShExecutable, int);
TShHandleBase* ConstructBindings();
void DeleteLinker(TShHandleBase*);
void DeleteBindingList(TShHandleBase* bindingList);

TUniformMap* ConstructUniformMap();
void DeleteCompiler(TCompiler*);

void DeleteUniformMap(TUniformMap*);

#endif // _SHHANDLE_INCLUDED_
