#ifndef NORMS_IPP
#define NORMS_IPP

// Definitions for norms.hpp. Do not include this file directly.

#include "norms.hpp" // IWYU pragma: keep

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>

namespace mlinalg::detail {

template <NormScalar T> constexpr auto abs_value(const T& value) -> real_t<T> {
    if constexpr (is_complex_v<T>) {
        return std::abs(value);
    } else {
        return value < T{0} ? -value : value;
    }
}

template <NormScalar T> constexpr auto abs_squared(const T& value) -> real_t<T> {
    if constexpr (is_complex_v<T>) {
        return std::norm(value);
    } else {
        return value * value;
    }
}

template <typename M>
    requires LuScalar<typename M::value_type>
void jacobi_orthogonalize_columns(M& matrix) {
    using T    = M::value_type;
    using Real = real_t<T>;

    // Converged long before this in practice (LAPACK's Jacobi SVD allows 30 sweeps).
    constexpr int  kMaxSweeps = 60;
    constexpr Real kTolerance = std::numeric_limits<Real>::epsilon();

    const auto rows = matrix.rows();
    const auto cols = matrix.cols();

    for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
        bool rotated = false;
        for (decltype(matrix.cols()) p = 0; p + 1 < cols; ++p) {
            for (auto q = p + 1; q < cols; ++q) {
                // alpha = |u_p|^2, beta = |u_q|^2, gamma = u_p^H u_q
                Real alpha{};
                Real beta{};
                T    gamma{};
                for (decltype(matrix.rows()) i = 0; i < rows; ++i) {
                    alpha += abs_squared(matrix(i, p));
                    beta += abs_squared(matrix(i, q));
                    gamma += conj_scalar(matrix(i, p)) * matrix(i, q);
                }
                const Real gamma_abs = abs_value(gamma);
                if (gamma_abs <= kTolerance * std::sqrt(alpha * beta)) {
                    continue; // already orthogonal (or one of the columns is zero)
                }
                rotated = true;

                // Remove the phase of gamma from u_q, then apply the real Jacobi rotation that
                // makes the two columns orthogonal.
                const T    phase = conj_scalar(gamma) / gamma_abs;
                const Real zeta  = (beta - alpha) / (Real{2} * gamma_abs);
                const Real t     = (zeta < Real{0} ? Real{-1} : Real{1}) /
                               (std::abs(zeta) + std::sqrt(Real{1} + (zeta * zeta)));
                const Real c = Real{1} / std::sqrt(Real{1} + (t * t));
                const Real s = c * t;
                for (decltype(matrix.rows()) i = 0; i < rows; ++i) {
                    const T up   = matrix(i, p);
                    const T uq   = phase * matrix(i, q);
                    matrix(i, p) = (c * up) - (s * uq);
                    matrix(i, q) = (s * up) + (c * uq);
                }
            }
        }
        if (!rotated) {
            break;
        }
    }
}

template <typename M>
    requires LuScalar<typename M::value_type>
auto squared_column_norm(const M& matrix, Index col) -> real_t<typename M::value_type> {
    real_t<typename M::value_type> squared{};
    for (Index row = 0; row < matrix.rows(); ++row) {
        squared += abs_squared(matrix(row, col));
    }
    return squared;
}

template <typename M>
    requires LuScalar<typename M::value_type>
auto spectral_norm(M matrix) -> real_t<typename M::value_type> {
    jacobi_orthogonalize_columns(matrix);

    real_t<typename M::value_type> largest{};
    for (Index col = 0; col < matrix.cols(); ++col) {
        largest = std::max(largest, squared_column_norm(matrix, col));
    }
    return std::sqrt(largest);
}

} // namespace mlinalg::detail

#endif // NORMS_IPP
