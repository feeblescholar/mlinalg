#ifndef PARTIAL_PIV_LU_IPP
#define PARTIAL_PIV_LU_IPP

// Definitions for partial_piv_lu.hpp. Do not include this file directly.

#include "partial_piv_lu.hpp" // IWYU pragma: keep

#include <cassert>
#include <utility>

namespace mlinalg {

namespace detail {

template <LuScalar T> constexpr auto pivot_magnitude(const T& value) noexcept {
    if constexpr (is_complex_v<T>) {
        return pivot_magnitude(value.real()) + pivot_magnitude(value.imag());
    } else {
        return value < T{0} ? -value : value;
    }
}

} // namespace detail

template <typename M>
constexpr PartialPivLU<M>::PartialPivLU(const M& matrix)
    : m_lu(matrix), m_permutation(matrix.rows(), 1) {
    assert(matrix.rows() == matrix.cols() && "PartialPivLU: matrix must be square");
    compute();
}

template <typename M> constexpr auto PartialPivLU<M>::rows() const noexcept -> Index {
    return m_lu.rows();
}

template <typename M> constexpr auto PartialPivLU<M>::cols() const noexcept -> Index {
    return m_lu.cols();
}

template <typename M> constexpr auto PartialPivLU<M>::matrix_lu() const noexcept -> const M& {
    return m_lu;
}

template <typename M>
constexpr auto PartialPivLU<M>::permutation() const noexcept -> const permutation_type& {
    return m_permutation;
}

template <typename M> constexpr auto PartialPivLU<M>::permutation_sign() const noexcept -> int {
    return m_sign;
}

template <typename M> constexpr auto PartialPivLU<M>::determinant() const -> value_type {
    auto det = static_cast<value_type>(m_sign);
    for (Index i = 0; i < m_lu.rows(); ++i) {
        det *= m_lu(i, i);
    }
    return det;
}

template <typename M> constexpr auto PartialPivLU<M>::is_invertible() const -> bool {
    for (Index i = 0; i < m_lu.rows(); ++i) {
        if (m_lu(i, i) == value_type{0}) {
            return false;
        }
    }
    return true;
}

template <typename M> constexpr auto PartialPivLU<M>::inverse() const -> M {
    assert(is_invertible() && "PartialPivLU::inverse: matrix is singular");
    const Index n   = m_lu.rows();
    M           inv = m_lu; // same shape; every coefficient is overwritten below

    for (Index j = 0; j < n; ++j) {
        // Right-hand side: column j of P, i.e. a 1 in the row that was moved from row j.
        for (Index i = 0; i < n; ++i) {
            inv(i, j) = m_permutation[i] == j ? value_type{1} : value_type{0};
        }
        // Forward substitution with the unit lower triangle L, column-oriented.
        for (Index k = 0; k < n; ++k) {
            const value_type xk = inv(k, j);
            for (Index i = k + 1; i < n; ++i) {
                inv(i, j) -= m_lu(i, k) * xk;
            }
        }
        // Back substitution with the upper triangle U, column-oriented.
        for (Index k = n - 1; k >= 0; --k) {
            inv(k, j) /= m_lu(k, k);
            const value_type xk = inv(k, j);
            for (Index i = 0; i < k; ++i) {
                inv(i, j) -= m_lu(i, k) * xk;
            }
        }
    }
    return inv;
}

template <typename M> constexpr void PartialPivLU<M>::compute() {
    // Right-looking Doolittle elimination. Inner loops run down columns to match the
    // column-major layout.
    const Index n = m_lu.rows();
    for (Index i = 0; i < n; ++i) {
        m_permutation[i] = i;
    }

    for (Index k = 0; k < n; ++k) {
        Index pivot_row = k;
        auto  pivot_mag = detail::pivot_magnitude(m_lu(k, k));
        for (Index i = k + 1; i < n; ++i) {
            const auto mag = detail::pivot_magnitude(m_lu(i, k));
            if (mag > pivot_mag) {
                pivot_mag = mag;
                pivot_row = i;
            }
        }

        if (pivot_row != k) {
            for (Index j = 0; j < n; ++j) {
                std::swap(m_lu(k, j), m_lu(pivot_row, j));
            }
            std::swap(m_permutation[k], m_permutation[pivot_row]);
            m_sign = -m_sign;
        }

        // A zero pivot means the whole column below it is zero: nothing to eliminate.
        const value_type pivot = m_lu(k, k);
        if (pivot == value_type{0}) {
            continue;
        }
        for (Index i = k + 1; i < n; ++i) {
            m_lu(i, k) /= pivot;
        }
        for (Index j = k + 1; j < n; ++j) {
            const value_type ukj = m_lu(k, j);
            for (Index i = k + 1; i < n; ++i) {
                m_lu(i, j) -= m_lu(i, k) * ukj;
            }
        }
    }
}

} // namespace mlinalg

#endif // PARTIAL_PIV_LU_IPP
