Testing allocators is difficult as bugs may only surface after particular
allocation patterns. The main approach to testing _mimalloc_ is therefore
to have extensive internal invariant checking (see `page_is_valid` in `page.c`
for example), which is enabled in debug mode with `-DMI_DEBUG_FULL=ON`.
The main testing strategy is then to run [`mimalloc-bench`][bench] using full
invariant checking to catch any potential problems over a wide range of intensive
allocation benchmarks and programs.

However, this does not test well for the entire API surface and this is tested
with `test-api.c` when using `make test` (from `out/debug` etc). (This is
not complete yet, please add to it.)

The `main.c` and `main-override.c` are there to test if building and overriding
from a local install works and therefore these build a separate `test/CMakeLists.txt`.

[bench]: https://github.com/daanx/mimalloc-bench
