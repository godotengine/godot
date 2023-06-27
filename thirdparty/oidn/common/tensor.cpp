// ======================================================================== //
// Copyright 2009-2019 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "exception.h"
#include "tensor.h"

namespace oidn {

  std::map<std::string, Tensor> parseTensors(void* buffer)
  {
    char* input = (char*)buffer;

    // Parse the magic value
    const int magic = *(unsigned short*)input;
    if (magic != 0x41D7)
      throw Exception(Error::InvalidOperation, "invalid tensor archive");
    input += sizeof(unsigned short);

    // Parse the version
    const int majorVersion = *(unsigned char*)input++;
    const int minorVersion = *(unsigned char*)input++;
    UNUSED(minorVersion);
    if (majorVersion > 1)
      throw Exception(Error::InvalidOperation, "unsupported tensor archive version");

    // Parse the number of tensors
    const int numTensors = *(int*)input;
    input += sizeof(int);

    // Parse the tensors
    std::map<std::string, Tensor> tensorMap;
    for (int i = 0; i < numTensors; ++i)
    {
      Tensor tensor;

      // Parse the name
      const int nameLen = *(unsigned char*)input++;
      std::string name(input, nameLen);
      input += nameLen;

      // Parse the number of dimensions
      const int ndims = *(unsigned char*)input++;

      // Parse the shape of the tensor
      tensor.dims.resize(ndims);
      for (int i = 0; i < ndims; ++i)
        tensor.dims[i] = ((int*)input)[i];
      input += ndims * sizeof(int);

      // Parse the format of the tensor
      tensor.format = std::string(input, input + ndims);
      input += ndims;

      // Parse the data type of the tensor
      const char type = *(unsigned char*)input++;
      if (type != 'f') // only float32 is supported
        throw Exception(Error::InvalidOperation, "unsupported tensor data type");

      // Skip the data
      tensor.data = (float*)input;
      input += tensor.size() * sizeof(float);

      // Add the tensor to the map
      tensorMap.emplace(name, std::move(tensor));
    }

    return tensorMap;
  }

} // namespace oidn
