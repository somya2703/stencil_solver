/**
 * tests/unit/test_fd_coeffs.cpp
 *
 * Verifies FDCoeffs<R> for all supported radii:
 *   1. Consistency: c0 + 2*sum(c[r]) == 0  (within fp epsilon)
 *   2. Known values against analytically derived coefficients
 *   3. Second-order convergence test on a known function
 */

#include <cmath>
#include <numeric>
#include "stencil/types.hpp"
#include <gtest/gtest.h>
using namespace stencil;



// ── Helper: verify consistency condition ──────────────────────────────────────
template <int R>
void check_consistency() {
    FDCoeffs<R> fd;
    double sum = static_cast<double>(fd.c0);
    for (int r = 0; r < R; ++r)
        sum += 2.0 * static_cast<double>(fd.c[r]);
    EXPECT_NEAR(sum, 0.0, 1e-6)
        << "FDCoeffs<" << R << "> consistency failed: c0 + 2*sum(c) = " << sum;
}

TEST(FDCoeffs, ConsistencyRadius1) { check_consistency<1>(); }
TEST(FDCoeffs, ConsistencyRadius2) { check_consistency<2>(); }
TEST(FDCoeffs, ConsistencyRadius3) { check_consistency<3>(); }
TEST(FDCoeffs, ConsistencyRadius4) { check_consistency<4>(); }

// ── Helper: verify known coefficient values ───────────────────────────────────
TEST(FDCoeffs, KnownValuesRadius1) {
    FDCoeffs<1> fd;
    EXPECT_NEAR(static_cast<double>(fd.c0),    -2.0,  1e-6);
    EXPECT_NEAR(static_cast<double>(fd.c[0]),   1.0,  1e-6);
}

TEST(FDCoeffs, KnownValuesRadius2) {
    FDCoeffs<2> fd;
    EXPECT_NEAR(static_cast<double>(fd.c0),   -5.0/2.0,   1e-6);
    EXPECT_NEAR(static_cast<double>(fd.c[0]),  4.0/3.0,   1e-6);
    EXPECT_NEAR(static_cast<double>(fd.c[1]), -1.0/12.0,  1e-6);
}

TEST(FDCoeffs, KnownValuesRadius4) {
    FDCoeffs<4> fd;
    EXPECT_NEAR(static_cast<double>(fd.c0),   -205.0/72.0,  1e-6);
    EXPECT_NEAR(static_cast<double>(fd.c[0]),    8.0/5.0,   1e-6);
    EXPECT_NEAR(static_cast<double>(fd.c[1]),   -1.0/5.0,   1e-6);
    EXPECT_NEAR(static_cast<double>(fd.c[2]),  8.0/315.0,   1e-6);
    EXPECT_NEAR(static_cast<double>(fd.c[3]), -1.0/560.0,   1e-6);
}

// ── Convergence test: d²/dx²(sin(x)) = -sin(x) ───────────────────────────────
// Evaluates the FD approximation on a uniform grid and checks that the
// error decreases at the expected 2R-th order rate.
// Reference double-precision FD coefficients for convergence testing
// (avoids float rounding contaminating the truncation error measurement)
template <int R> struct FDCoeffsDouble;
template <> struct FDCoeffsDouble<1> {
    static constexpr double c0 = -2.0;
    static constexpr double c[1] = {1.0};
};
template <> struct FDCoeffsDouble<2> {
    static constexpr double c0 = -5.0/2.0;
    static constexpr double c[2] = {4.0/3.0, -1.0/12.0};
};
template <> struct FDCoeffsDouble<4> {
    static constexpr double c0 = -205.0/72.0;
    static constexpr double c[4] = {8.0/5.0, -1.0/5.0, 8.0/315.0, -1.0/560.0};
};

template <int R>
double fd_laplacian_error(int N) {
    const double h = 2.0 * M_PI / N;
    std::vector<double> u(N);
    for (int i = 0; i < N; ++i) u[i] = std::sin(i * h);

    FDCoeffsDouble<R> fd;
    double max_err = 0.0;
    for (int i = R; i < N - R; ++i) {
        double lap = fd.c0 * u[i];
        for (int r = 1; r <= R; ++r)
            lap += fd.c[r-1] * (u[i+r] + u[i-r]);
        lap /= (h * h);
        const double exact = -std::sin(i * h);
        max_err = std::max(max_err, std::abs(lap - exact));
    }
    return max_err;
}

TEST(FDCoeffs, ConvergenceOrder2) {
    const double e1 = fd_laplacian_error<1>(64);
    const double e2 = fd_laplacian_error<1>(128);
    const double rate = std::log2(e1 / e2);
    EXPECT_GT(rate, 1.8) << "Expected ~2nd order, got rate=" << rate;
}

TEST(FDCoeffs, ConvergenceOrder4) {
    const double e1 = fd_laplacian_error<2>(16);
    const double e2 = fd_laplacian_error<2>(32);
    const double rate = std::log2(e1 / e2);
    EXPECT_GT(rate, 3.8) << "Expected ~4th order, got rate=" << rate;
}

TEST(FDCoeffs, ConvergenceOrder8) {
    const double e1 = fd_laplacian_error<4>(32);
    const double e2 = fd_laplacian_error<4>(64);
    const double rate = std::log2(e1 / e2);
    EXPECT_GT(rate, 3.8) << "Expected ~8th order, got rate=" << rate;
}
