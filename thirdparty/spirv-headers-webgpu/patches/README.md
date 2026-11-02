# Required SPIR-V extension enums

`0001-add-required-extension-enums.patch` adds extension definitions referenced by
the pinned SPIRV-Tools sources but missing from Dawn's pinned C++11 header:
`FragmentCoverageMESA`, `PredicatedIOINTEL` and `RoundedDivideSqrtINTEL`.

The other headers, upstream result/type flags, license and generator grammars
are used unchanged. The engine's `thirdparty/spirv-headers` remains separate.
