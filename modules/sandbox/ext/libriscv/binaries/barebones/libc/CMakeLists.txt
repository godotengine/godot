option(LIBC_USE_STDLIB "Enable standard libraries" ON)
option(LIBC_WRAP_NATIVE "" OFF)

set(LIBC_SOURCES
	assert.cpp
	heap.cpp
	libc.cpp
	libcxx.cpp
	microthread.cpp
	print.cpp
  )
if (NOT LIBC_USE_STDLIB)
	list(APPEND LIBC_SOURCES
		start.cpp
		write.cpp
	)
endif()

set_source_files_properties(libc.cpp
	PROPERTIES COMPILE_FLAGS -fno-builtin)

add_library(libc STATIC ${LIBC_SOURCES})
target_include_directories(libc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(libc PUBLIC tinyprintf)
target_compile_definitions(libc PRIVATE NATIVE_MEM_SYSCALLS=1)

if (LIBC_USE_STDLIB)
	target_compile_definitions(libc PRIVATE USE_NEWLIB=1)
endif()
if (LIBC_WRAP_NATIVE)
	target_compile_definitions(libc PRIVATE
		WRAP_NATIVE_SYSCALLS=1)
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	set_source_files_properties(
			kernel/panic.cpp
		PROPERTIES COMPILE_FLAGS -Wno-frame-address)
endif()
