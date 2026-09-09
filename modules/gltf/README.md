# Godot glTF import and export module

In a nutshell, the glTF module works like this:

* The [`structures/`](structures/) folder contains glTF structures, the
  small pieces that make up a glTF file, represented as C++ classes.
* The [`extensions/`](extensions/) folder contains glTF extensions, which
  are optional features that build on top of the base glTF spec.
* [`GLTFState`](gltf_state.h) holds collections of structures and extensions.
* [`GLTFDocument`](gltf_document.h) operates on GLTFState and its elements.
* The [`editor/`](editor/) folder uses GLTFDocument to import and export 3D models.
