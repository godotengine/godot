# GDScript ELF Golden Test Fixtures

This directory contains golden test fixtures for C99 code generation tests.

## Format

Each fixture file is named `<test_name>.golden.c` and contains the expected C99 code output for a specific GDScript function.

## Updating Fixtures

To update golden fixtures when the code generation changes:

1. Run tests with a flag to update fixtures (if implemented)
2. Or manually update the `.golden.c` files after verifying the generated code is correct

## Fixture Structure

Fixtures are normalized C99 code that should match the output of `GDScriptBytecodeCCodeGenerator::generate_c_code()` exactly (after normalization).
