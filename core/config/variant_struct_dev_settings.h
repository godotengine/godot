

#pragma once

// NOTE: this .h is just for enabling different configurations/possible implementations
// this file will be removed when this PR is squashed for merging, and the behaviour is decided

// Enabled reference-type behaviour
// If commented out, COW behaviour is enabled
#define VSTRUCT_IS_REFERENCE_TYPE

// Moves all newly defined enums to the end of their respective lists, rather than where they make more intuitive sense
// If commented out, backwards compatability may break
#define ENUMS_SHOULD_NOT_BREAK_APIS
