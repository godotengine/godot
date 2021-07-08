xxHash - Extremely fast hash algorithm
======================================

xxHash is an Extremely fast Hash algorithm, running at RAM speed limits.
It successfully completes the [SMHasher](http://code.google.com/p/smhasher/wiki/SMHasher) test suite
which evaluates collision, dispersion and randomness qualities of hash functions.
Code is highly portable, and hashes are identical on all platforms (little / big endian).

|Branch      |Status   |
|------------|---------|
|master      | [![Build Status](https://travis-ci.org/Cyan4973/xxHash.svg?branch=master)](https://travis-ci.org/Cyan4973/xxHash?branch=master) |
|dev         | [![Build Status](https://travis-ci.org/Cyan4973/xxHash.svg?branch=dev)](https://travis-ci.org/Cyan4973/xxHash?branch=dev) |



Benchmarks
-------------------------

The benchmark uses SMHasher speed test, compiled with Visual 2010 on a Windows Seven 32-bit box.
The reference system uses a Core 2 Duo @3GHz


| Name          |   Speed  | Quality | Author           |
|---------------|----------|:-------:|------------------|
| [xxHash]      | 5.4 GB/s |   10    | Y.C.             |
| MurmurHash 3a | 2.7 GB/s |   10    | Austin Appleby   |
| SBox          | 1.4 GB/s |    9    | Bret Mulvey      |
| Lookup3       | 1.2 GB/s |    9    | Bob Jenkins      |
| CityHash64    | 1.05 GB/s|   10    | Pike & Alakuijala|
| FNV           | 0.55 GB/s|    5    | Fowler, Noll, Vo |
| CRC32         | 0.43 GB/s|    9    |                  |
| MD5-32        | 0.33 GB/s|   10    | Ronald L.Rivest  |
| SHA1-32       | 0.28 GB/s|   10    |                  |

[xxHash]: http://www.xxhash.com

Q.Score is a measure of quality of the hash function.
It depends on successfully passing SMHasher test set.
10 is a perfect score.
Algorithms with a score < 5 are not listed on this table.

A more recent version, XXH64, has been created thanks to [Mathias Westerdahl](https://github.com/JCash),
which offers superior speed and dispersion for 64-bit systems.
Note however that 32-bit applications will still run faster using the 32-bit version.

SMHasher speed test, compiled using GCC 4.8.2, on Linux Mint 64-bit.
The reference system uses a Core i5-3340M @2.7GHz

| Version    | Speed on 64-bit | Speed on 32-bit |
|------------|------------------|------------------|
| XXH64      | 13.8 GB/s        |  1.9 GB/s        |
| XXH32      |  6.8 GB/s        |  6.0 GB/s        |

This project also includes a command line utility, named `xxhsum`, offering similar features as `md5sum`,
thanks to [Takayuki Matsuoka](https://github.com/t-mat) contributions.


### License

The library files `xxhash.c` and `xxhash.h` are BSD licensed.
The utility `xxhsum` is GPL licensed.


### Build modifiers

The following macros can be set at compilation time,
they modify xxhash behavior. They are all disabled by default.

- `XXH_INLINE_ALL` : Make all functions `inline`, with bodies directly included within `xxhash.h`.
                     There is no need for an `xxhash.o` module in this case.
                     Inlining functions is generally beneficial for speed on small keys.
                     It's especially effective when key length is a compile time constant,
                     with observed performance improvement in the +200% range .
                     See [this article](https://fastcompression.blogspot.com/2018/03/xxhash-for-small-keys-impressive-power.html) for details.
- `XXH_ACCEPT_NULL_INPUT_POINTER` : if set to `1`, when input is a null-pointer,
                                    xxhash result is the same as a zero-length key
                                    (instead of a dereference segfault).
- `XXH_FORCE_MEMORY_ACCESS` : default method `0` uses a portable `memcpy()` notation.
                              Method `1` uses a gcc-specific `packed` attribute, which can provide better performance for some targets.
                              Method `2` forces unaligned reads, which is not standard compliant, but might sometimes be the only way to extract better performance.
- `XXH_CPU_LITTLE_ENDIAN` : by default, endianess is determined at compile time.
                            It's possible to skip auto-detection and force format to little-endian, by setting this macro to 1.
                            Setting it to 0 forces big-endian.
- `XXH_FORCE_NATIVE_FORMAT` : on big-endian systems : use native number representation.
                              Breaks consistency with little-endian results.
- `XXH_PRIVATE_API` : same impact as `XXH_INLINE_ALL`.
                      Name underlines that symbols will not be published on library public interface.
- `XXH_NAMESPACE` : prefix all symbols with the value of `XXH_NAMESPACE`.
                    Useful to evade symbol naming collisions,
                    in case of multiple inclusions of xxHash source code.
                    Client applications can still use regular function name,
                    symbols are automatically translated through `xxhash.h`.
- `XXH_STATIC_LINKING_ONLY` : gives access to state declaration for static allocation.
                              Incompatible with dynamic linking, due to risks of ABI changes.
- `XXH_NO_LONG_LONG` : removes support for XXH64,
                       for targets without 64-bit support.


### Example

Calling xxhash 64-bit variant from a C program :

```c
#include "xxhash.h"

unsigned long long calcul_hash(const void* buffer, size_t length)
{
    unsigned long long const seed = 0;   /* or any other value */
    unsigned long long const hash = XXH64(buffer, length, seed);
    return hash;
}
```

Using streaming variant is more involved, but makes it possible to provide data in multiple rounds :
```c
#include "stdlib.h"   /* abort() */
#include "xxhash.h"


unsigned long long calcul_hash_streaming(someCustomType handler)
{
    XXH64_state_t* const state = XXH64_createState();
    if (state==NULL) abort();

    size_t const bufferSize = SOME_VALUE;
    void* const buffer = malloc(bufferSize);
    if (buffer==NULL) abort();

    unsigned long long const seed = 0;   /* or any other value */
    XXH_errorcode const resetResult = XXH64_reset(state, seed);
    if (resetResult == XXH_ERROR) abort();

    (...)
    while ( /* any condition */ ) {
        size_t const length = get_more_data(buffer, bufferSize, handler);   /* undescribed */
        XXH_errorcode const addResult = XXH64_update(state, buffer, length);
        if (addResult == XXH_ERROR) abort();
        (...)
    }

    (...)
    unsigned long long const hash = XXH64_digest(state);

    free(buffer);
    XXH64_freeState(state);

    return hash;
}
```


### Other programming languages

Beyond the C reference version,
xxHash is also available on many programming languages,
thanks to great contributors.
They are [listed here](http://www.xxhash.com/#other-languages).


### Branch Policy

> - The "master" branch is considered stable, at all times.
> - The "dev" branch is the one where all contributions must be merged
    before being promoted to master.
>   + If you plan to propose a patch, please commit into the "dev" branch,
      or its own feature branch.
      Direct commit to "master" are not permitted.
