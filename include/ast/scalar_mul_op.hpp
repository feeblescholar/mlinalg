#ifndef AST_SCALAR_MUL_OP_HPP
#define AST_SCALAR_MUL_OP_HPP

// Module 3: ScalarMulOp - an expression scaled by a scalar.

#include "expr_traits.hpp"

namespace mlinalg {

/// scalar * expr, coefficient-wise. The scalar is stored as the expression's ScalarType.
template <Expression E> class ScalarMulOp {
  public:
    constexpr ScalarMulOp(expr_scalar_t<E> scalar, E expr);

    [[nodiscard]] constexpr auto rows() const -> Index;
    [[nodiscard]] constexpr auto cols() const -> Index;
    [[nodiscard]] constexpr auto coeff(Index row, Index col) const -> expr_scalar_t<E>;

    [[nodiscard]] constexpr auto scalar() const noexcept -> const expr_scalar_t<E>&;
    [[nodiscard]] constexpr auto expression() const noexcept -> const E&;

  private:
    expr_scalar_t<E> m_scalar;
    E                m_expr;
};

template <typename S, typename E> ScalarMulOp(S, E) -> ScalarMulOp<E>;

template <Expression E> struct expr_traits<ScalarMulOp<E>> {
    static constexpr int Rows = expr_rows_v<E>;
    static constexpr int Cols = expr_cols_v<E>;
    using ScalarType          = expr_scalar_t<E>;
};

} // namespace mlinalg

#include "scalar_mul_op.ipp" // IWYU pragma: keep

#endif // AST_SCALAR_MUL_OP_HPP
