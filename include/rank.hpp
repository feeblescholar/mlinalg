#ifndef RANK_HPP
#define RANK_HPP

// Module 4: numerical rank of floating-point and complex matrices (Matrix::rank, Matrix::dim).
//
// In floating point, a matrix that is rank deficient on paper almost never has an exactly zero
// singular value, so the rank is defined with a threshold: it is the number of singular values
// larger than
//
//   max(rows, cols) * epsilon * (largest singular value)
//
// which is the convention of LAPACK-based tools (MATLAB, NumPy). The singular values come from the
// one-sided Jacobi iteration in norms.hpp.
//
// Integer matrices have an exact rank instead, see bareiss.hpp.

#include "matrix_storage.hpp"
#include "norms.hpp"
#include "partial_piv_lu.hpp"

namespace mlinalg::detail {

/// Number of singular values of `matrix` above the threshold described at the top of this file.
/// `matrix` is taken by value and used as scratch space. Fastest when the matrix has no more
/// columns than rows.
template <typename M>
    requires LuScalar<typename M::value_type>
[[nodiscard]] auto numerical_rank(M matrix) -> Index;

} // namespace mlinalg::detail

#include "rank.ipp" // IWYU pragma: keep

#endif // RANK_HPP
