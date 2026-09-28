// 2D heat equation  u_t = alpha * (u_xx + u_yy)  on the unit square, u = 0 on the boundary.
// Explicit finite differences in time and space (FTCS scheme).
//
// With the initial condition u(x, y, 0) = sin(pi x) sin(pi y) the exact solution is
//   u(x, y, t) = exp(-2 pi^2 alpha t) sin(pi x) sin(pi y),
// so every run can report its error against the analytical answer.
//
// Each run prints one line of JSON so that the benchmarking tool can collect results.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef HEAT_WITH_CUDA
#include "heat_cuda.hpp"
#endif

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kAlpha = 1.0;  // thermal diffusivity

struct Config {
    int n = 512;                   // interior grid points per side
    long steps = -1;               // number of time steps (performance runs)
    double t_end = -1.0;           // final time (accuracy runs); used if steps < 0
    double r = 0.2;                // alpha * dt / h^2, must be <= 0.25 for stability
    std::string precision = "double";
    std::string backend = "serial";
    int threads = 0;               // 0 = OpenMP default
};

struct Result {
    long steps;
    double dt, t, seconds, max_err, rms_err;
    int threads;
};

[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "usage: heat [--n N] [--steps S | --t-end T] [--r R]\n"
                 "            [--precision float|double] [--backend serial|omp|cuda] [--threads P]\n");
    std::exit(1);
}

Config parse_args(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) usage();
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--n")) c.n = std::atoi(next());
        else if (!std::strcmp(argv[i], "--steps")) c.steps = std::atol(next());
        else if (!std::strcmp(argv[i], "--t-end")) c.t_end = std::atof(next());
        else if (!std::strcmp(argv[i], "--r")) c.r = std::atof(next());
        else if (!std::strcmp(argv[i], "--precision")) c.precision = next();
        else if (!std::strcmp(argv[i], "--backend")) c.backend = next();
        else if (!std::strcmp(argv[i], "--threads")) c.threads = std::atoi(next());
        else usage();
    }
    if (c.n < 1 || c.r <= 0.0 || c.r > 0.25) usage();
    if ((c.steps < 0) == (c.t_end < 0.0)) usage();  // exactly one of --steps / --t-end
    if (c.precision != "float" && c.precision != "double") usage();
    if (c.backend != "serial" && c.backend != "omp" && c.backend != "cuda") usage();
#ifndef HEAT_WITH_CUDA
    if (c.backend == "cuda") {
        std::fprintf(stderr, "this build has no CUDA back end\n");
        std::exit(2);
    }
#endif
    return c;
}

// One time step on an (n+2) x (n+2) grid stored row by row; the outer ring is the boundary.
// Each new value is the old value plus r times the discrete Laplacian (the 5-point stencil).
template <typename T>
void step_serial(const T* u, T* v, int n, T r) {
    const int w = n + 2;
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j) {
            const int k = i * w + j;
            v[k] = u[k] + r * (u[k - w] + u[k + w] + u[k - 1] + u[k + 1] - T(4) * u[k]);
        }
    }
}

// Same update, but rows are shared out between threads. Every point depends only on the
// previous time level, so rows can be computed in any order without data races.
template <typename T>
void step_omp(const T* u, T* v, int n, T r) {
    const int w = n + 2;
#pragma omp parallel for schedule(static)
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j) {
            const int k = i * w + j;
            v[k] = u[k] + r * (u[k - w] + u[k + w] + u[k - 1] + u[k + 1] - T(4) * u[k]);
        }
    }
}

template <typename T>
Result run(const Config& c) {
    const int n = c.n;
    const int w = n + 2;
    const double h = 1.0 / (n + 1);

    // Choose the step count and time step. For accuracy runs dt is shrunk slightly so that
    // the simulation lands exactly on t_end while keeping alpha*dt/h^2 <= r.
    long steps;
    double dt;
    if (c.steps >= 0) {
        steps = c.steps;
        dt = c.r * h * h / kAlpha;
    } else {
        const double dt_max = c.r * h * h / kAlpha;
        steps = static_cast<long>(std::ceil(c.t_end / dt_max));
        dt = c.t_end / steps;
    }
    const T r = static_cast<T>(kAlpha * dt / (h * h));

    std::vector<T> u(static_cast<size_t>(w) * w, T(0));
    std::vector<T> v(u.size(), T(0));
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= n; ++j)
            u[static_cast<size_t>(i) * w + j] =
                static_cast<T>(std::sin(kPi * j * h) * std::sin(kPi * i * h));

    int threads = 1;
#ifdef _OPENMP
    if (c.backend == "omp") {
        if (c.threads > 0) omp_set_num_threads(c.threads);
        threads = omp_get_max_threads();
    }
#endif

    double seconds;
#ifdef HEAT_WITH_CUDA
    if (c.backend == "cuda") {
        threads = 0;
        seconds = run_cuda(u, n, steps, r);
    } else
#endif
    {
        const auto start = std::chrono::steady_clock::now();
        for (long s = 0; s < steps; ++s) {
            if (c.backend == "omp") step_omp(u.data(), v.data(), n, r);
            else step_serial(u.data(), v.data(), n, r);
            std::swap(u, v);  // swaps the buffers, not the data
        }
        seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    // Compare with the exact solution (in double precision, whatever T is).
    const double t = steps * dt;
    const double decay = std::exp(-2.0 * kPi * kPi * kAlpha * t);
    double max_err = 0.0, sum_sq = 0.0;
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j) {
            const double exact = decay * std::sin(kPi * j * h) * std::sin(kPi * i * h);
            const double e = std::fabs(static_cast<double>(u[static_cast<size_t>(i) * w + j]) - exact);
            max_err = std::max(max_err, e);
            sum_sq += e * e;
        }
    }

    Result res;
    res.steps = steps;
    res.dt = dt;
    res.t = t;
    res.seconds = seconds;
    res.max_err = max_err;
    res.rms_err = std::sqrt(sum_sq / (static_cast<double>(n) * n));
    res.threads = threads;
    return res;
}

}  // namespace

int main(int argc, char** argv) {
    const Config c = parse_args(argc, argv);
    const Result res = c.precision == "float" ? run<float>(c) : run<double>(c);

    // Million lattice updates per second: grid points updated per step, times steps, per second.
    const double updates = static_cast<double>(c.n) * c.n * res.steps;
    const double mlups = res.seconds > 0.0 ? updates / res.seconds / 1e6 : 0.0;
    // Idealised memory traffic: each update reads one value and writes one value.
    const double bytes = updates * 2.0 * (c.precision == "float" ? 4.0 : 8.0);
    const double gbs = res.seconds > 0.0 ? bytes / res.seconds / 1e9 : 0.0;

    std::printf(
        "{\"backend\":\"%s\",\"precision\":\"%s\",\"n\":%d,\"threads\":%d,\"steps\":%ld,"
        "\"dt\":%.6e,\"t\":%.6e,\"seconds\":%.6e,\"mlups\":%.3f,\"gbs\":%.3f,"
        "\"max_err\":%.6e,\"rms_err\":%.6e}\n",
        c.backend.c_str(), c.precision.c_str(), c.n, res.threads, res.steps, res.dt, res.t,
        res.seconds, mlups, gbs, res.max_err, res.rms_err);
    return 0;
}
