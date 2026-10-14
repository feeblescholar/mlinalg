#ifndef RANK_IPP
#define RANK_IPP

// Definitions for rank.hpp. Do not include this file directly.

#include "rank.hpp" // IWYU pragma: keep

#include <algorithm>
#include <cmath>
#include <limits>

namespace mlinalg::detail {

template <typename M>
    requires LuScalar<typename M::value_type>
auto numerical_rank(M matrix) -> Index {
    using Real = real_t<typename M::value_type>;

    jacobi_orthogonalize_columns(matrix);

    // The squared column lengths are now the squared singular values.
    Real largest{};
    for (Index col = 0; col < matrix.cols(); ++col) {
        largest = std::max(largest, squared_column_norm(matrix, col));
    }
    const Real threshold = static_cast<Real>(std::max(matrix.rows(), matrix.cols())) *
                           std::numeric_limits<Real>::epsilon() * std::sqrt(largest);

    Index rank = 0;
    for (Index col = 0; col < matrix.cols(); ++col) {
        if (std::sqrt(squared_column_norm(matrix, col)) > threshold) {
            ++rank;
        }
    }
    return rank;
}

} // namespace mlinalg::detail

#endif // RANK_IPP
