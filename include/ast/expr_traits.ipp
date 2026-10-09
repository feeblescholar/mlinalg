#ifndef AST_EXPR_TRAITS_IPP
#define AST_EXPR_TRAITS_IPP

// Definitions for expr_traits.hpp. Do not include this file directly.

namespace mlinalg::detail {

constexpr auto combine_dims(int lhs, int rhs) noexcept -> int {
    return (lhs == Dynamic || rhs == Dynamic) ? Dynamic : lhs;
}

constexpr auto dims_compatible(int lhs, int rhs) noexcept -> bool {
    return lhs == Dynamic || rhs == Dynamic || lhs == rhs;
}

} // namespace mlinalg::detail

#endif // AST_EXPR_TRAITS_IPP
