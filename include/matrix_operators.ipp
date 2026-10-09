#ifndef MATRIX_OPERATORS_IPP
#define MATRIX_OPERATORS_IPP

// Definitions for matrix_operators.hpp. Do not include this file directly.

#include "matrix_operators.hpp" // IWYU pragma: keep

#include <utility>

namespace mlinalg {

namespace detail {

template <typename X>
    requires BindableOperand<X>
constexpr auto as_expr(X&& operand) -> operand_expr_t<X> {
    if constexpr (Expression<std::remove_cvref_t<X>>) {
        return std::forward<X>(operand);
    } else {
        return MatrixWrapper<std::remove_cvref_t<X>>(operand);
    }
}

} // namespace detail

template <typename L, typename R>
    requires detail::CwiseOperands<L, R>
constexpr auto operator+(L&& lhs,
                         R&& rhs) -> AddOp<detail::operand_expr_t<L>, detail::operand_expr_t<R>> {
    return {detail::as_expr(std::forward<L>(lhs)), detail::as_expr(std::forward<R>(rhs))};
}

template <typename L, typename R>
    requires detail::CwiseOperands<L, R>
constexpr auto operator-(L&& lhs,
                         R&& rhs) -> SubOp<detail::operand_expr_t<L>, detail::operand_expr_t<R>> {
    return {detail::as_expr(std::forward<L>(lhs)), detail::as_expr(std::forward<R>(rhs))};
}

template <typename L, typename R>
    requires detail::ProductOperands<L, R>
constexpr auto operator*(L&& lhs,
                         R&& rhs) -> MulOp<detail::operand_expr_t<L>, detail::operand_expr_t<R>> {
    return {detail::as_expr(std::forward<L>(lhs)), detail::as_expr(std::forward<R>(rhs))};
}

template <typename S, typename X>
    requires detail::ScalarFor<S, X>
constexpr auto operator*(const S& scalar, X&& operand) -> ScalarMulOp<detail::operand_expr_t<X>> {
    return {scalar, detail::as_expr(std::forward<X>(operand))};
}

template <typename X, typename S>
    requires detail::ScalarFor<S, X>
constexpr auto operator*(X&& operand, const S& scalar) -> ScalarMulOp<detail::operand_expr_t<X>> {
    return {scalar, detail::as_expr(std::forward<X>(operand))};
}

// Evaluation goes through Matrix::operator=, which uses a temporary, so lhs may appear in rhs.

template <typename M, typename R>
    requires detail::AddAssignable<M, R>
constexpr auto operator+=(M& lhs, const R& rhs) -> M& {
    lhs = lhs + rhs;
    return lhs;
}

template <typename M, typename R>
    requires detail::SubAssignable<M, R>
constexpr auto operator-=(M& lhs, const R& rhs) -> M& {
    lhs = lhs - rhs;
    return lhs;
}

template <typename M, typename R>
    requires detail::MulAssignable<M, R>
constexpr auto operator*=(M& lhs, const R& rhs) -> M& {
    lhs = lhs * rhs;
    return lhs;
}

} // namespace mlinalg

#endif // MATRIX_OPERATORS_IPP
