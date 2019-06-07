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

//
// The top level algorithms for linking multiple
// shaders together.
//
#include "../Include/Common.h"
#include "../Include/ShHandle.h"

//
// Actual link object, derived from the shader handle base classes.
//
class TGenericLinker : public TLinker {
public:
    TGenericLinker(EShExecutable e, int dOptions) : TLinker(e, infoSink), debugOptions(dOptions) { }
    bool link(TCompilerList&, TUniformMap*) { return true; }
    void getAttributeBindings(ShBindingTable const **) const { }
    TInfoSink infoSink;
    int debugOptions;
};

//
// The internal view of a uniform/float object exchanged with the driver.
//
class TUniformLinkedMap : public TUniformMap {
public:
    TUniformLinkedMap() { }
    virtual int getLocation(const char*) { return 0; }
};

TShHandleBase* ConstructLinker(EShExecutable executable, int debugOptions)
{
    return new TGenericLinker(executable, debugOptions);
}

void DeleteLinker(TShHandleBase* linker)
{
    delete linker;
}

TUniformMap* ConstructUniformMap()
{
    return new TUniformLinkedMap();
}

void DeleteUniformMap(TUniformMap* map)
{
    delete map;
}

TShHandleBase* ConstructBindings()
{
    return 0;
}

void DeleteBindingList(TShHandleBase* bindingList)
{
    delete bindingList;
}
