cmake_minimum_required(VERSION 3.0)
project(tinyprintf C)

macro(make_test t src PRINTF SPRINTF)
    add_executable(${t} ${src} tinyprintf.c)
    target_include_directories(${t} PUBLIC .)
    target_compile_definitions(${t} PRIVATE
        TINYPRINTF_OVERRIDE_LIBC=0
        TINYPRINTF_DEFINE_TFP_PRINTF=${PRINTF}
        TINYPRINTF_DEFINE_TFP_SPRINTF=${SPRINTF})
endmacro()


make_test(test-printf test/printf.c 1 0)
make_test(test-sprintf test/sprintf.c 0 1)
make_test(test-all test/all.c 1 1)
