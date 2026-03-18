#ifndef TEST_LOD_SYSTEM_H
#define TEST_LOD_SYSTEM_H

namespace GaussianSplatting {
namespace Tests {

// Main test runner
void run_lod_system_tests();

// Individual test functions
bool test_hierarchical_structure_build();
bool test_frustum_culling();
bool test_adaptive_lod();
bool test_splat_clustering();
bool test_streaming_lod();
bool test_lod_transitions();
bool test_scalability();

} // namespace Tests
} // namespace GaussianSplatting

#endif // TEST_LOD_SYSTEM_H