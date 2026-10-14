#ifndef AST_CONJ_TRANSPOSE_OP_HPP
#define AST_CONJ_TRANSPOSE_OP_HPP

// Module 3: ConjTransposeOp - conjugate (Hermitian) transpose of an expression.

#include "expr_traits.hpp"

#include <complex>

namespace mlinalg {

namespace detail {

/// Complex conjugate; the identity for every non-complex scalar type.
template <typename T> [[nodiscard]] constexpr auto conj_scalar(const T& value) -> T;
template <typename T>
[[nodiscard]] constexpr auto conj_scalar(const std::complex<T>& value) -> std::complex<T>;

} // namespace detail

/// expr^H: coeff(i, j) is the complex conjugate of expr.coeff(j, i). For real scalar types this
/// is the plain transpose.
template <Expression E> class ConjTransposeOp {
  public:
    constexpr explicit ConjTransposeOp(E expr);

    [[nodiscard]] constexpr auto rows() const -> Index;
    [[nodiscard]] constexpr auto cols() const -> Index;
    [[nodiscard]] constexpr auto coeff(Index row, Index col) const -> expr_scalar_t<E>;

    [[nodiscard]] constexpr auto expression() const noexcept -> const E&;

  private:
    E m_expr;
};

template <typename E> ConjTransposeOp(E) -> ConjTransposeOp<E>;

template <Expression E> struct expr_traits<ConjTransposeOp<E>> {
    static constexpr int Rows = expr_cols_v<E>;
    static constexpr int Cols = expr_rows_v<E>;
    using ScalarType          = expr_scalar_t<E>;
};

} // namespace mlinalg

#include "conj_transpose_op.ipp" // IWYU pragma: keep

#endif // AST_CONJ_TRANSPOSE_OP_HPP
