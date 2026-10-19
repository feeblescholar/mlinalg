#ifndef AST_TEMPORARY_IPP
#define AST_TEMPORARY_IPP

// Definitions for temporary.hpp. Do not include this file directly.

namespace mlinalg {

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
                                   template <typename E>
                 requires(!std::same_as<E, Temporary<T, Rows, Cols>>) && Expression<E> &&
                         std::same_as<expr_scalar_t<E>, T> && (expr_rows_v<E> == Rows) &&
                         (expr_cols_v<E> == Cols)
constexpr Temporary<T, Rows, Cols>::Temporary(const E& expr) : m_storage(expr.rows(), expr.cols()) {
    for (Index col = 0; col < cols(); ++col) {
        for (Index row = 0; row < rows(); ++row) {
            m_storage[(col * rows()) + row] = expr.coeff(row, col);
        }
    }
}

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
constexpr auto Temporary<T, Rows, Cols>::rows() const -> Index {
    return m_storage.rows();
}

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
constexpr auto Temporary<T, Rows, Cols>::cols() const -> Index {
    return m_storage.cols();
}

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
constexpr auto Temporary<T, Rows, Cols>::coeff(Index row, Index col) const -> T {
    return m_storage[(col * rows()) + row];
}

} // namespace mlinalg

#endif // AST_TEMPORARY_IPP
