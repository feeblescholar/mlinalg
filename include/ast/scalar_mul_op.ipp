#ifndef AST_SCALAR_MUL_OP_IPP
#define AST_SCALAR_MUL_OP_IPP

// Definitions for scalar_mul_op.hpp. Do not include this file directly.

#include <utility>

namespace mlinalg {

template <Expression E>
constexpr ScalarMulOp<E>::ScalarMulOp(expr_scalar_t<E> scalar, E expr)
    : m_scalar(std::move(scalar)), m_expr(std::move(expr)) {}

template <Expression E> constexpr auto ScalarMulOp<E>::rows() const -> Index {
    if constexpr (expr_rows_v<E> != Dynamic) {
        return expr_rows_v<E>;
    } else {
        return m_expr.rows();
    }
}

template <Expression E> constexpr auto ScalarMulOp<E>::cols() const -> Index {
    if constexpr (expr_cols_v<E> != Dynamic) {
        return expr_cols_v<E>;
    } else {
        return m_expr.cols();
    }
}

template <Expression E>
constexpr auto ScalarMulOp<E>::coeff(Index row, Index col) const -> expr_scalar_t<E> {
    return m_scalar * m_expr.coeff(row, col);
}

template <Expression E>
constexpr auto ScalarMulOp<E>::scalar() const noexcept -> const expr_scalar_t<E>& {
    return m_scalar;
}

template <Expression E> constexpr auto ScalarMulOp<E>::expression() const noexcept -> const E& {
    return m_expr;
}

} // namespace mlinalg

#endif // AST_SCALAR_MUL_OP_IPP
