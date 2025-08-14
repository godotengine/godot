# Metal Rendering Device

This document aims to describe the Metal rendering device implementation in Godot.

## Future work / ideas

* Use placement heaps
* Explicit hazard tracking
* [MetalFX] upscaling support?

## Acknowledgments

The Metal rendering owes a lot to the work of the [MoltenVK] project, which is a Vulkan implementation on top of Metal.
In accordance with the Apache 2.0 license, the following copyright notices have been included where applicable:

```
/**************************************************************************/
/*                                                                        */
/* Portions of this code were derived from MoltenVK.                      */
/*                                                                        */
/* Copyright (c) 2015-2023 The Brenwill Workshop Ltd.                     */
/* (http://www.brenwill.com)                                              */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/
```

[MoltenVK]: https://github.com/KhronosGroup/MoltenVK
[MetalFX]: https://developer.apple.com/documentation/metalfx?language=objc
