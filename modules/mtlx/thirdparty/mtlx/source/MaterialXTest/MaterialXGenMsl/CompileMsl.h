//
// Copyright (c) 2023 Apple Inc.
// Licensed under the Apache License v2.0
//

#ifndef COMPILEMSL_H
#define COMPILEMSL_H

#if __APPLE__
void CompileMslShader(const char* pShaderFilePath, const char* pEntryFuncName);
#else
void CompileMslShader(const char*, const char* ) {}
#endif

#endif // COMPILEMSL_H
