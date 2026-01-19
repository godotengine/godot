# GDScript Structs and Performance Optimizations

This branch implements GDScript struct types and VM/compiler optimizations.

## Overview

- **Struct Types**: Lightweight struct syntax with compile-time type checking
- **VM Optimizations**: Typed array/dictionary iteration (3-4x faster)
- **Compiler Optimizations**: Dead code elimination, lambda warnings
- **API Additions**: `Array.reserve()` for pre-allocation

## Documentation

See `doc/gdscript_structs/` for detailed documentation:

- **GDSCRIPT_STRUCTS_USAGE.md** - Type system reference
- **STRUCTS_QUICK_START.md** - Getting started guide
- **GDSCRIPT_PERFORMANCE_GUIDE.md** - Performance best practices
- **TESTING_QUICK_START.md** - Testing instructions

## Building and Testing

Standard Godot build process. See `TESTING_QUICK_START.md` for running benchmarks.

## Status

All features are implemented and tested. See `doc/gdscript_structs/GDSCRIPT_STRUCTS_ROADMAP.md` for implementation details.

## Issue Reference

Addresses: https://github.com/godotengine/godot/issues/7329
