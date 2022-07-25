# Godot GLTF import and export module

In a nutshell, the GLTF module works like this:

* The [`structures/`](structures/) folder contains GLTF structures, the
  small pieces that make up a GLTF file, represented as C++ classes.
* The [`extensions/`](extensions/) folder contains GLTF extensions, which
  are optional features that build on top of the base GLTF spec.
* [`GLTFState`](gltf_state.h) holds collections of structures and extensions.
* [`GLTFDocument`](gltf_document.h) operates on GLTFState and its elements.
* The [`editor/`](editor/) folder uses GLTFDocument to import and export 3D models.
