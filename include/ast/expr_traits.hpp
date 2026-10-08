#ifndef AST_EXPR_TRAITS_HPP
#define AST_EXPR_TRAITS_HPP

// Module 3: Expression Templates - shared compile-time machinery for AST nodes.
//
// Every AST node type E specializes expr_traits<E> with:
//   static constexpr int Rows;  // compile-time row count, or Dynamic
//   static constexpr int Cols;  // compile-time column count, or Dynamic
//   using ScalarType = ...;     // coefficient type
//
// and provides the runtime interface checked by the Expression concept:
//   rows(), cols()     current dimensions
//   coeff(row, col)    value of one coefficient (scalar semantics of the node)

#include "../matrix_storage.hpp"

#include <concepts>
#include <type_traits>

namespace mlinalg {

/// Compile-time properties of an AST node. Specialized by every node type.
template <typename E> struct expr_traits;

template <typename E> inline constexpr int expr_rows_v = expr_traits<std::remove_cvref_t<E>>::Rows;

template <typename E> inline constexpr int expr_cols_v = expr_traits<std::remove_cvref_t<E>>::Cols;

template <typename E> using expr_scalar_t = expr_traits<std::remove_cvref_t<E>>::ScalarType;

/// An AST node: a non-reference object type with expr_traits and coefficient access.
/// Nodes store their children by value, so they must be cheap to copy.
template <typename E>
concept Expression =
    std::is_object_v<E> && !std::is_const_v<E> && std::copy_constructible<E> && requires {
        { expr_traits<E>::Rows } -> std::convertible_to<int>;
        { expr_traits<E>::Cols } -> std::convertible_to<int>;
        typename expr_traits<E>::ScalarType;
    } && requires(const E& expr, Index row, Index col) {
        { expr.rows() } -> std::same_as<Index>;
        { expr.cols() } -> std::same_as<Index>;
        { expr.coeff(row, col) } -> std::convertible_to<typename expr_traits<E>::ScalarType>;
    };

namespace detail {

/// Result of merging two dimensions that must agree: Dynamic if either one is Dynamic.
[[nodiscard]] constexpr auto combine_dims(int lhs, int rhs) noexcept -> int;

/// True unless both dimensions are static and different.
[[nodiscard]] constexpr auto dims_compatible(int lhs, int rhs) noexcept -> bool;

/// Operands of a coefficient-wise binary node (AddOp, SubOp): same scalar type and shape.
template <typename L, typename R>
concept CwiseCompatible =
    Expression<L> && Expression<R> && std::same_as<expr_scalar_t<L>, expr_scalar_t<R>> &&
    dims_compatible(expr_rows_v<L>, expr_rows_v<R>) &&
    dims_compatible(expr_cols_v<L>, expr_cols_v<R>);

/// Operands of a matrix product (MulOp): same scalar type and lhs.cols() == rhs.rows().
template <typename L, typename R>
concept ProductCompatible =
    Expression<L> && Expression<R> && std::same_as<expr_scalar_t<L>, expr_scalar_t<R>> &&
    dims_compatible(expr_cols_v<L>, expr_rows_v<R>);

} // namespace detail

} // namespace mlinalg

#include "expr_traits.ipp" // IWYU pragma: keep

#endif // AST_EXPR_TRAITS_HPP
