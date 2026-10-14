#ifndef AST_TRANSPOSE_OP_HPP
#define AST_TRANSPOSE_OP_HPP

// Module 3: TransposeOp - transpose of an expression.

#include "expr_traits.hpp"

namespace mlinalg {

/// expr^T: coeff(i, j) is expr.coeff(j, i). Rows and columns are swapped, nothing is copied.
template <Expression E> class TransposeOp {
  public:
    constexpr explicit TransposeOp(E expr);

    [[nodiscard]] constexpr auto rows() const -> Index;
    [[nodiscard]] constexpr auto cols() const -> Index;
    [[nodiscard]] constexpr auto coeff(Index row, Index col) const -> expr_scalar_t<E>;

    [[nodiscard]] constexpr auto expression() const noexcept -> const E&;

  private:
    E m_expr;
};

template <typename E> TransposeOp(E) -> TransposeOp<E>;

template <Expression E> struct expr_traits<TransposeOp<E>> {
    static constexpr int Rows = expr_cols_v<E>;
    static constexpr int Cols = expr_rows_v<E>;
    using ScalarType          = expr_scalar_t<E>;
};

} // namespace mlinalg

#include "transpose_op.ipp" // IWYU pragma: keep

#endif // AST_TRANSPOSE_OP_HPP
