#ifndef AST_MATRIX_WRAPPER_IPP
#define AST_MATRIX_WRAPPER_IPP

// Definitions for matrix_wrapper.hpp. Do not include this file directly.

namespace mlinalg {

template <MatrixLeaf M>
constexpr MatrixWrapper<M>::MatrixWrapper(const M& matrix) noexcept : m_matrix(&matrix) {}

template <MatrixLeaf M> constexpr auto MatrixWrapper<M>::rows() const -> Index {
    if constexpr (M::RowsAtCompileTime != Dynamic) {
        return M::RowsAtCompileTime;
    } else {
        return static_cast<Index>(m_matrix->rows());
    }
}

template <MatrixLeaf M> constexpr auto MatrixWrapper<M>::cols() const -> Index {
    if constexpr (M::ColsAtCompileTime != Dynamic) {
        return M::ColsAtCompileTime;
    } else {
        return static_cast<Index>(m_matrix->cols());
    }
}

template <MatrixLeaf M>
constexpr auto MatrixWrapper<M>::coeff(Index row, Index col) const -> M::value_type {
    return (*m_matrix)(row, col);
}

template <MatrixLeaf M> constexpr auto MatrixWrapper<M>::matrix() const noexcept -> const M& {
    return *m_matrix;
}

} // namespace mlinalg

#endif // AST_MATRIX_WRAPPER_IPP
