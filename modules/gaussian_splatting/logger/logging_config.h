#ifndef GAUSSIAN_LOGGING_CONFIG_H
#define GAUSSIAN_LOGGING_CONFIG_H

// Centralized logging cadence for performance builds.
// Use these to gate high-volume logs (e.g., JSON diagnostics).
static constexpr int LOG_EVERY_FRAMES_VERBOSE = 0;      // never
static constexpr int LOG_EVERY_FRAMES_STATUS  = 120;    // ~2 seconds at 60 FPS
static constexpr int LOG_EVERY_FRAMES_RARE    = 1000;   // rare snapshots

#endif // GAUSSIAN_LOGGING_CONFIG_H
