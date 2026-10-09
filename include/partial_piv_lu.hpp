#ifndef PARTIAL_PIV_LU_HPP
#define PARTIAL_PIV_LU_HPP

// Module 4: LU decomposition with partial (row) pivoting.
//
// For a square matrix A, computes P * A = L * U where P is a row permutation, L is unit lower
// triangular and U is upper triangular. L and U are stored compactly in one matrix: U on and above
// the diagonal, L strictly below it (its unit diagonal is implicit).
//
// The pivot of column k is the entry of largest magnitude on or below the diagonal. Magnitude is
// |x| for real scalars and |re| + |im| for complex ones (the cheap norm LAPACK uses for pivoting).
// A singular matrix is not an error: its zero pivots are kept and the determinant is 0. It has no
// inverse, though: check is_invertible() before calling inverse().

#include "matrix_storage.hpp"
#include "packet.hpp"

#include <concepts>

namespace mlinalg {

namespace detail {

/// Scalars an LU decomposition is defined for: division must be exact up to rounding.
template <typename T>
concept LuScalar = std::floating_point<T> || is_complex_v<T>;

/// Pivot-selection magnitude: |x| for reals, |re| + |im| for complex. constexpr, unlike std::abs.
template <LuScalar T> [[nodiscard]] constexpr auto pivot_magnitude(const T& value) noexcept;

} // namespace detail

/// LU decomposition of a square matrix of type M (typically Matrix<T, N, N>) with partial
/// pivoting. Owns a copy of the matrix it was computed from.
///
/// The scalar requirement is a static_assert rather than a constraint on purpose: Matrix<T>::lu()
/// names PartialPivLU<Matrix<T>> in its signature for every T, including integers, and a
/// constraint could be checked as soon as that signature is instantiated.
template <typename M> class PartialPivLU {
    static_assert(detail::LuScalar<typename M::value_type>,
                  "PartialPivLU requires a floating-point or std::complex scalar type");

  public:
    using matrix_type = M;
    using value_type  = M::value_type;

    /// Compile-time extent of the (square) matrix, or Dynamic.
    static constexpr int SizeAtCompileTime =
        M::RowsAtCompileTime != Dynamic ? M::RowsAtCompileTime : M::ColsAtCompileTime;

    using permutation_type = MatrixStorage<Index, SizeAtCompileTime, 1>;

    /// Factorizes `matrix`, which must be square.
    constexpr explicit PartialPivLU(const M& matrix);

    [[nodiscard]] constexpr auto rows() const noexcept -> Index;
    [[nodiscard]] constexpr auto cols() const noexcept -> Index;

    /// L (strictly below the diagonal, unit diagonal implied) and U (on and above it).
    [[nodiscard]] constexpr auto matrix_lu() const noexcept -> const M&;

    /// Row i of P * A is row permutation()[i] of A.
    [[nodiscard]] constexpr auto permutation() const noexcept -> const permutation_type&;

    /// Sign of the permutation P: +1 for an even number of row swaps, -1 for an odd number.
    [[nodiscard]] constexpr auto permutation_sign() const noexcept -> int;

    /// det(A) = sign(P) * product of the diagonal of U.
    [[nodiscard]] constexpr auto determinant() const -> value_type;

    /// False if U has an exactly zero pivot. A matrix that is numerically close to singular still
    /// counts as invertible; its inverse is then dominated by rounding errors.
    [[nodiscard]] constexpr auto is_invertible() const -> bool;

    /// A^-1, computed column by column from L * U * x = P * e_j. A must be invertible (checked
    /// by an assertion only).
    [[nodiscard]] constexpr auto inverse() const -> M;

  private:
    constexpr void compute();

    M                m_lu;
    permutation_type m_permutation;
    int              m_sign = 1;
};

} // namespace mlinalg

#include "partial_piv_lu.ipp" // IWYU pragma: keep

#endif // PARTIAL_PIV_LU_HPP
