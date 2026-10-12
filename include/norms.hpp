#ifndef NORMS_HPP
#define NORMS_HPP

// Module 4: building blocks of the matrix norms (Matrix::norm1, norm2, norminf, normF).
//
// The spectral norm (largest singular value) is computed with one-sided Jacobi rotations
// (Hestenes): pairs of columns are rotated until all columns are mutually orthogonal. The columns
// then have the singular values as their lengths. The method is deterministic and accurate, and
// each sweep over all column pairs costs O(rows * cols^2).
//
// Squared magnitudes are accumulated without scaling, so coefficients whose square overflows or
// underflows the real type (beyond roughly 1e+-154 for double) give inaccurate results.

#include "ast/conj_transpose_op.hpp"
#include "packet.hpp"
#include "partial_piv_lu.hpp"

#include <complex>
#include <concepts>

namespace mlinalg::detail {

/// Scalars the absolute-sum norms (norm1, norminf) are defined for.
template <typename T>
concept NormScalar = LuScalar<T> || std::signed_integral<T>;

/// The type of a magnitude of T: T itself for real scalars, the component type for complex ones.
template <typename T> struct real_type {
    using type = T;
};

template <typename T> struct real_type<std::complex<T>> {
    using type = T;
};

template <typename T> using real_t = real_type<T>::type;

/// |value|. constexpr for real scalars; the complex modulus needs a square root and is not.
template <NormScalar T> [[nodiscard]] constexpr auto abs_value(const T& value) -> real_t<T>;

/// |value|^2, without a square root for complex scalars.
template <NormScalar T> [[nodiscard]] constexpr auto abs_squared(const T& value) -> real_t<T>;

/// Largest singular value of `matrix`, which is taken by value and used as scratch space.
/// Fastest when the matrix has no more columns than rows.
template <typename M>
    requires LuScalar<typename M::value_type>
[[nodiscard]] auto spectral_norm(M matrix) -> real_t<typename M::value_type>;

} // namespace mlinalg::detail

#include "norms.ipp" // IWYU pragma: keep

#endif // NORMS_HPP
