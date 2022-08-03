# Disable unreachable code warnings.
# This test unconditionally throws in some places, the compiler will detect that
# control flow will never reach some instructions. This is intentional.
target_link_libraries(${test_target} PRIVATE thrust.silence_unreachable_code_warnings)
