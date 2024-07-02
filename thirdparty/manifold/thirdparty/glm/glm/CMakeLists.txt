file(GLOB ROOT_SOURCE *.cpp)
file(GLOB ROOT_INLINE *.inl)
file(GLOB ROOT_HEADER *.hpp)
file(GLOB ROOT_TEXT ../*.txt)
file(GLOB ROOT_MD ../*.md)
file(GLOB ROOT_NAT ../util/glm.natvis)

file(GLOB_RECURSE CORE_SOURCE ./detail/*.cpp)
file(GLOB_RECURSE CORE_INLINE ./detail/*.inl)
file(GLOB_RECURSE CORE_HEADER ./detail/*.hpp)

file(GLOB_RECURSE EXT_SOURCE ./ext/*.cpp)
file(GLOB_RECURSE EXT_INLINE ./ext/*.inl)
file(GLOB_RECURSE EXT_HEADER ./ext/*.hpp)

file(GLOB_RECURSE GTC_SOURCE ./gtc/*.cpp)
file(GLOB_RECURSE GTC_INLINE ./gtc/*.inl)
file(GLOB_RECURSE GTC_HEADER ./gtc/*.hpp)

file(GLOB_RECURSE GTX_SOURCE ./gtx/*.cpp)
file(GLOB_RECURSE GTX_INLINE ./gtx/*.inl)
file(GLOB_RECURSE GTX_HEADER ./gtx/*.hpp)

file(GLOB_RECURSE SIMD_SOURCE ./simd/*.cpp)
file(GLOB_RECURSE SIMD_INLINE ./simd/*.inl)
file(GLOB_RECURSE SIMD_HEADER ./simd/*.h)

source_group("Text Files" FILES ${ROOT_TEXT} ${ROOT_MD})
source_group("Core Files" FILES ${CORE_SOURCE})
source_group("Core Files" FILES ${CORE_INLINE})
source_group("Core Files" FILES ${CORE_HEADER})
source_group("EXT Files" FILES ${EXT_SOURCE})
source_group("EXT Files" FILES ${EXT_INLINE})
source_group("EXT Files" FILES ${EXT_HEADER})
source_group("GTC Files" FILES ${GTC_SOURCE})
source_group("GTC Files" FILES ${GTC_INLINE})
source_group("GTC Files" FILES ${GTC_HEADER})
source_group("GTX Files" FILES ${GTX_SOURCE})
source_group("GTX Files" FILES ${GTX_INLINE})
source_group("GTX Files" FILES ${GTX_HEADER})
source_group("SIMD Files" FILES ${SIMD_SOURCE})
source_group("SIMD Files" FILES ${SIMD_INLINE})
source_group("SIMD Files" FILES ${SIMD_HEADER})

add_library(glm-header-only INTERFACE)
add_library(glm::glm-header-only ALIAS glm-header-only)

target_include_directories(glm-header-only INTERFACE
	"$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
	"$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

if (GLM_BUILD_LIBRARY)
	add_library(glm
		${ROOT_TEXT}      ${ROOT_MD}        ${ROOT_NAT}
		${ROOT_SOURCE}    ${ROOT_INLINE}    ${ROOT_HEADER}
		${CORE_SOURCE}    ${CORE_INLINE}    ${CORE_HEADER}
		${EXT_SOURCE}     ${EXT_INLINE}     ${EXT_HEADER}
		${GTC_SOURCE}     ${GTC_INLINE}     ${GTC_HEADER}
		${GTX_SOURCE}     ${GTX_INLINE}     ${GTX_HEADER}
		${SIMD_SOURCE}    ${SIMD_INLINE}    ${SIMD_HEADER}
	)
	add_library(glm::glm ALIAS glm)
	target_link_libraries(glm PUBLIC glm-header-only)
else()
	add_library(glm INTERFACE)
	add_library(glm::glm ALIAS glm)
	target_link_libraries(glm INTERFACE glm-header-only)
endif()
