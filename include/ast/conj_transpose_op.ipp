#ifndef AST_CONJ_TRANSPOSE_OP_IPP
#define AST_CONJ_TRANSPOSE_OP_IPP

// Definitions for conj_transpose_op.hpp. Do not include this file directly.

#include <complex>
#include <utility>

namespace mlinalg {

namespace detail {

template <typename T> constexpr auto conj_scalar(const T& value) -> T {
    return value;
}

template <typename T> constexpr auto conj_scalar(const std::complex<T>& value) -> std::complex<T> {
    return std::conj(value);
}

} // namespace detail

template <Expression E>
constexpr ConjTransposeOp<E>::ConjTransposeOp(E expr) : m_expr(std::move(expr)) {}

template <Expression E> constexpr auto ConjTransposeOp<E>::rows() const -> Index {
    if constexpr (expr_cols_v<E> != Dynamic) {
        return expr_cols_v<E>;
    } else {
        return m_expr.cols();
    }
}

template <Expression E> constexpr auto ConjTransposeOp<E>::cols() const -> Index {
    if constexpr (expr_rows_v<E> != Dynamic) {
        return expr_rows_v<E>;
    } else {
        return m_expr.rows();
    }
}

template <Expression E>
constexpr auto ConjTransposeOp<E>::coeff(Index row, Index col) const -> expr_scalar_t<E> {
    return detail::conj_scalar(m_expr.coeff(col, row));
}

template <Expression E> constexpr auto ConjTransposeOp<E>::expression() const noexcept -> const E& {
    return m_expr;
}

} // namespace mlinalg

#endif // AST_CONJ_TRANSPOSE_OP_IPP
