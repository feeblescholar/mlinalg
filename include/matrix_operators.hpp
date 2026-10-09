#ifndef MATRIX_OPERATORS_HPP
#define MATRIX_OPERATORS_HPP

// Module 4: overloaded +, -, * building AST nodes (Module 3), and the compound assignments
// +=, -=, *=.
//
// The binary operators never evaluate anything: they return proxy objects that are evaluated when
// they are assigned to a Matrix. An operand is either
//   * an AST expression, which is copied into the new node, or
//   * a concrete matrix (MatrixLeaf), which is wrapped in a non-owning MatrixWrapper. It must be an
//     lvalue that outlives the expression; temporaries are rejected to prevent dangling.
//
// Scalars must have exactly the expression's ScalarType: 2.0 * a is valid for a double matrix,
// 2 * a and 2.0f * a are not.
//
// Compound assignments follow the std semantics: a op= b evaluates a = a op b immediately and
// returns a as an lvalue. Because nothing outlives the statement, the right-hand side may also be
// a temporary matrix.

#include "ast/ast.hpp"

#include <concepts>
#include <type_traits>

namespace mlinalg {

namespace detail {

/// A concrete matrix used as an operand. AST nodes are never MatrixLeafs, but be explicit.
template <typename X>
concept MatrixOperand = MatrixLeaf<std::remove_cvref_t<X>> && !Expression<std::remove_cvref_t<X>>;

/// A forwarded operand (X as deduced from X&&) that can be stored in an AST node: any expression,
/// or a matrix lvalue. A matrix rvalue would leave the expression dangling.
template <typename X>
concept BindableOperand =
    Expression<std::remove_cvref_t<X>> || (MatrixOperand<X> && std::is_lvalue_reference_v<X>);

/// The AST node type an operand turns into.
template <typename X> struct operand_expr;

template <typename X>
    requires Expression<std::remove_cvref_t<X>>
struct operand_expr<X> {
    using type = std::remove_cvref_t<X>;
};

template <typename X>
    requires MatrixOperand<X>
struct operand_expr<X> {
    using type = MatrixWrapper<std::remove_cvref_t<X>>;
};

template <typename X> using operand_expr_t = operand_expr<X>::type;

/// Converts an operand into its AST node.
template <typename X>
    requires BindableOperand<X>
[[nodiscard]] constexpr auto as_expr(X&& operand) -> operand_expr_t<X>;

/// Operands of + and -.
template <typename L, typename R>
concept CwiseOperands = BindableOperand<L> && BindableOperand<R> &&
                        CwiseCompatible<operand_expr_t<L>, operand_expr_t<R>>;

/// Operands of the matrix product.
template <typename L, typename R>
concept ProductOperands = BindableOperand<L> && BindableOperand<R> &&
                          ProductCompatible<operand_expr_t<L>, operand_expr_t<R>>;

/// Scalar factor of a ScalarMulOp over operand X: exactly the operand's scalar type.
template <typename S, typename X>
concept ScalarFor = BindableOperand<X> && std::same_as<S, expr_scalar_t<operand_expr_t<X>>>;

} // namespace detail

/// Coefficient-wise sum.
template <typename L, typename R>
    requires detail::CwiseOperands<L, R>
[[nodiscard]] constexpr auto
operator+(L&& lhs, R&& rhs) -> AddOp<detail::operand_expr_t<L>, detail::operand_expr_t<R>>;

/// Coefficient-wise difference.
template <typename L, typename R>
    requires detail::CwiseOperands<L, R>
[[nodiscard]] constexpr auto
operator-(L&& lhs, R&& rhs) -> SubOp<detail::operand_expr_t<L>, detail::operand_expr_t<R>>;

/// Matrix product.
template <typename L, typename R>
    requires detail::ProductOperands<L, R>
[[nodiscard]] constexpr auto
operator*(L&& lhs, R&& rhs) -> MulOp<detail::operand_expr_t<L>, detail::operand_expr_t<R>>;

/// scalar * matrix.
template <typename S, typename X>
    requires detail::ScalarFor<S, X>
[[nodiscard]] constexpr auto operator*(const S& scalar,
                                       X&&      operand) -> ScalarMulOp<detail::operand_expr_t<X>>;

/// matrix * scalar. Evaluated as scalar * coefficient.
template <typename X, typename S>
    requires detail::ScalarFor<S, X>
[[nodiscard]] constexpr auto operator*(X&&      operand,
                                       const S& scalar) -> ScalarMulOp<detail::operand_expr_t<X>>;

namespace detail {

/// M is a matrix that `lhs = lhs + rhs` is valid for: same scalar type, and a result shape that
/// fits M. SubAssignable and MulAssignable are the same for - and *. The right-hand side is used
/// as an lvalue, so temporary matrices are accepted (see the top of this file).
template <typename M, typename R>
concept AddAssignable = MatrixOperand<M> && requires(M& lhs, const R& rhs) { lhs = lhs + rhs; };

template <typename M, typename R>
concept SubAssignable = MatrixOperand<M> && requires(M& lhs, const R& rhs) { lhs = lhs - rhs; };

/// Covers both the matrix product and scaling by a scalar.
template <typename M, typename R>
concept MulAssignable = MatrixOperand<M> && requires(M& lhs, const R& rhs) { lhs = lhs * rhs; };

} // namespace detail

/// lhs = lhs + rhs.
template <typename M, typename R>
    requires detail::AddAssignable<M, R>
constexpr auto operator+=(M& lhs, const R& rhs) -> M&;

/// lhs = lhs - rhs.
template <typename M, typename R>
    requires detail::SubAssignable<M, R>
constexpr auto operator-=(M& lhs, const R& rhs) -> M&;

/// lhs = lhs * rhs, where rhs is a matrix, an expression or a scalar. A matrix product may change
/// the shape of a Dynamic lhs.
template <typename M, typename R>
    requires detail::MulAssignable<M, R>
constexpr auto operator*=(M& lhs, const R& rhs) -> M&;

} // namespace mlinalg

#include "matrix_operators.ipp" // IWYU pragma: keep

#endif // MATRIX_OPERATORS_HPP
