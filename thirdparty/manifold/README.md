[![codecov](https://codecov.io/github/elalish/manifold/branch/master/graph/badge.svg?token=IIA8G5HVS7)](https://codecov.io/github/elalish/manifold)

![A metallic Menger sponge](https://elalish.github.io/manifold/samples/models/mengerSponge3.webp "A metallic Menger sponge")

# Manifold

[**High-level Documentation**](https://elalish.blogspot.com/search/label/Manifold) | [**API Documentation**](https://elalish.github.io/manifold/modules.html) | [**Algorithm Documentation**](https://github.com/elalish/manifold/wiki/Manifold-Library)

[Manifold](https://github.com/elalish/manifold) is a geometry library dedicated to creating and operating on manifold triangle meshes. A [manifold mesh](https://github.com/elalish/manifold/wiki/Manifold-Library#manifoldness) is a mesh that represents a solid object, and so is very important in manufacturing, CAD, structural analysis, etc. Further information can be found on the [wiki](https://github.com/elalish/manifold/wiki/Manifold-Library).

## What's here

This library is intended to be fast with guaranteed manifold output. As such you need manifold meshes as input, which can be hard to come by since most 3D graphics meshes are not. This library can create simple primitive meshes but also links in Assimp, which will import many kinds of 3D files, but you'll get an error if the imported mesh isn't manifold. Various automated repair tools exist online for fixing non manifold models, usually for 3D printing. 

The most significant contribution here is a guaranteed-manifold [mesh Boolean](https://github.com/elalish/manifold/wiki/Manifold-Library#mesh-boolean) algorithm, which I believe is the first of its kind. If anyone knows of another, please tell me. Likewise, if the Boolean here ever fails you, please submit an issue! This Boolean forms the basis of a CAD kernel, as it allows simple shapes to be combined into more complex ones.

[Documentation](https://elalish.github.io/manifold/modules.html) is available through Doxygen for all of this library's classes and functions. Expect more detail to be added as time goes on.

To aid in speed, this library makes extensive use of parallelization, generally through Nvidia's Thrust library. You can switch between the CUDA, OMP and serial C++ backends by setting a CMake flag. Not everything is so parallelizable, for instance a [polygon triangulation](https://github.com/elalish/manifold/wiki/Manifold-Library#polygon-triangulation) algorithm is included which is serial. 

Look in the [samples](https://github.com/elalish/manifold/tree/master/samples) directory for examples of how to use this library to make interesting 3D models. You may notice that some of these examples bare a certain resemblance to my OpenSCAD designs on [Thingiverse](https://www.thingiverse.com/emmett), which is no accident. Much as I love OpenSCAD, my library is dramatically faster and the code is more flexible, though it could be improved even more with JS or Python bindings to avoid the syntax and compiling of C++. 

## Building

The canonical build instructions are in the [manifold.yml](https://github.com/elalish/manifold/blob/master/.github/workflows/manifold.yml) file, as that is what this project's continuous integration server uses to build and test. I have only built under Ubuntu Linux, and the CI uses Nvidia's Cuda 11 Docker image. Part of my [road map](https://github.com/elalish/manifold/wiki/Manifold-Library#road-map) is to migrate from Thrust to C++20 parallel algorithms, which will alleviate the need to install the Cuda Developer Kit to build.

## Python binding

> Note: This is still a WIP

The CMake script will build the python binding `pymanifold` automatically. To
use the extension, please add `$BUILD_DIR/tools` to your `PYTHONPATH`, where
`$BUILD_DIR` is the build directory for CMake. Examples using the python binding
can be found in `test/python`. Run the following code in the interpreter for
python binding documentation:

```
>>> import pymanifold
>>> help(manifold)
```

For more detailed documentation, please refer to the C++ API.

## Contributing

Contributions are welcome! A lower barrier contribution is to simply make a PR that adds a test, especially if it repros an issue you've found. Simply name it prepended with DISABLED_, so that it passes the CI. That will be a very strong signal to me to fix your issue. However, if you know how to fix it yourself, then including the fix in your PR would be much appreciated!

## About the author

This library is by [Emmett Lalish](https://elalish.blogspot.com/). I am currently a Google employee and this is my 20% project, not an official Google project. At my day job I'm the maintainer of [\<model-viewer\>](https://modelviewer.dev/). I was the first employee at a 3D video startup, [Omnivor](https://www.omnivor.io/), and before that I worked on 3D printing at Microsoft, including [3D Builder](https://www.microsoft.com/en-us/p/3d-builder/9wzdncrfj3t6?activetab=pivot%3Aoverviewtab). Originally an aerospace engineer, I started at a small DARPA contractor doing seedling projects, one of which became [Sea Hunter](https://en.wikipedia.org/wiki/Sea_Hunter). I earned my doctorate from the University of Washington in control theory and published some [papers](https://www.researchgate.net/scientific-contributions/75011026_Emmett_Lalish).
