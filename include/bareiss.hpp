#ifndef BAREISS_HPP
#define BAREISS_HPP

// Module 4: exact determinant and rank of integer matrices (Bareiss fraction-free elimination).
//
// Gaussian elimination on integers would need fractions. Bareiss' algorithm keeps every
// intermediate value an integer: step k computes
//
//   a(i, j) <- (a(i, j) * a(k, k) - a(i, k) * a(k, j)) / previous pivot
//
// where the division is always exact (by Sylvester's identity each a(i, j) is then a
// (k + 2) x (k + 2) minor of the original matrix). The last pivot is the determinant.
// A zero pivot is replaced by swapping in a lower row, which flips the sign.
//
// The same step reduces a rectangular matrix to row echelon form: a column without a nonzero
// entry at or below the current pivot row is skipped, and the division stays exact. The number of
// pivots found is the rank.
//
// Overflow: all arithmetic is done in T. Intermediates are minors of the input, and the product
// in the numerator is roughly the square of one, so it can overflow even when the determinant
// itself fits in T. Signed overflow is undefined behaviour.
// TODO: promote the intermediates to a wider type (e.g. __int128 for 64-bit T) internally.

#include <concepts>

namespace mlinalg::detail {

/// Scalars the Bareiss determinant is used for.
template <typename T>
concept BareissScalar = std::signed_integral<T>;

/// Determinant of the square matrix `matrix`, which is taken by value and used as scratch space.
/// M is typically a Matrix<T, N, N> with a signed integral T.
template <typename M>
    requires BareissScalar<typename M::value_type>
[[nodiscard]] constexpr auto bareiss_determinant(M matrix) -> M::value_type;

/// Exact rank of `matrix` (any shape), which is taken by value and used as scratch space.
/// The result type is the matrix's index type.
template <typename M>
    requires BareissScalar<typename M::value_type>
[[nodiscard]] constexpr auto bareiss_rank(M matrix) -> decltype(matrix.rows());

} // namespace mlinalg::detail

#include "bareiss.ipp" // IWYU pragma: keep

#endif // BAREISS_HPP
