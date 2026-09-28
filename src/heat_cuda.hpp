#pragma once

#include <vector>

// Advances the grid `u` (size (n+2)^2, boundary ring included) by `steps` FTCS steps on the GPU.
// On return `u` holds the final grid. Returns the time spent in the time-stepping loop only,
// in seconds, not counting the copies between host and device.
template <typename T>
double run_cuda(std::vector<T>& u, int n, long steps, T r);
