#ifndef AST_OWNED_MATRIX_IPP
#define AST_OWNED_MATRIX_IPP

// Definitions for owned_matrix.hpp. Do not include this file directly.

#include <utility>

namespace mlinalg {

template <MatrixLeaf M>
constexpr OwnedMatrix<M>::OwnedMatrix(M matrix) : m_matrix(std::move(matrix)) {}

template <MatrixLeaf M> constexpr auto OwnedMatrix<M>::rows() const -> Index {
    if constexpr (M::RowsAtCompileTime != Dynamic) {
        return M::RowsAtCompileTime;
    } else {
        return static_cast<Index>(m_matrix.rows());
    }
}

template <MatrixLeaf M> constexpr auto OwnedMatrix<M>::cols() const -> Index {
    if constexpr (M::ColsAtCompileTime != Dynamic) {
        return M::ColsAtCompileTime;
    } else {
        return static_cast<Index>(m_matrix.cols());
    }
}

template <MatrixLeaf M>
constexpr auto OwnedMatrix<M>::coeff(Index row, Index col) const -> M::value_type {
    return m_matrix(row, col);
}

template <MatrixLeaf M> constexpr auto OwnedMatrix<M>::matrix() const noexcept -> const M& {
    return m_matrix;
}

} // namespace mlinalg

#endif // AST_OWNED_MATRIX_IPP
