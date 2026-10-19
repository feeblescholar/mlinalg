#ifndef AST_EVAL_TREE_HPP
#define AST_EVAL_TREE_HPP

// Module 5: rewrites an expression into the tree that is actually evaluated.
//
// MulOp::coeff reads every operand coefficient once per dot product, so a product operand that is
// an expression would be recomputed lhs.rows() or rhs.cols() times, and chained products blow up
// to O(N^k). Right before evaluation, every MulOp operand is checked:
//   * a MatrixWrapper is forwarded as it is (by reference),
//   * any other expression is evaluated, together with its whole subtree, into a Temporary.
// Every other node is rebuilt with its rewritten children and stays lazy. For example:
//   (A + B) * C            ->  Temporary(A + B) * C
//   (A - B) + C * (D * E)  ->  (A - B) + C * Temporary(D * E)
//   A * B * C * D          ->  Temporary(Temporary(A * B) * C) * D
//
// The rewrite runs when a Matrix evaluates an expression, not when the operators build it, so
// expressions stay lazy and cheap to copy, and the temporaries only live as long as the evaluation.

#include "add_op.hpp"
#include "conj_transpose_op.hpp"
#include "expr_traits.hpp"
#include "matrix_wrapper.hpp"
#include "mul_op.hpp"
#include "scalar_mul_op.hpp"
#include "sub_op.hpp"
#include "temporary.hpp"
#include "transpose_op.hpp"

#include <type_traits>

namespace mlinalg::detail {

/// The rewritten type of expression E. Specialized for every node type.
template <typename E> struct eval_tree;

template <typename E> using eval_tree_t = eval_tree<std::remove_cvref_t<E>>::type;

/// Product operands that are read directly instead of being evaluated into a Temporary.
template <typename E> inline constexpr bool is_direct_operand_v = false;

template <MatrixLeaf M> inline constexpr bool is_direct_operand_v<MatrixWrapper<M>> = true;

/// The rewritten type of a MulOp operand E.
template <typename E>
using product_operand_t =
    std::conditional_t<is_direct_operand_v<E>, E, temporary_for_t<eval_tree_t<E>>>;

template <MatrixLeaf M> struct eval_tree<MatrixWrapper<M>> {
    using type = MatrixWrapper<M>;
};

template <typename L, typename R> struct eval_tree<AddOp<L, R>> {
    using type = AddOp<eval_tree_t<L>, eval_tree_t<R>>;
};

template <typename L, typename R> struct eval_tree<SubOp<L, R>> {
    using type = SubOp<eval_tree_t<L>, eval_tree_t<R>>;
};

template <typename L, typename R> struct eval_tree<MulOp<L, R>> {
    using type = MulOp<product_operand_t<L>, product_operand_t<R>>;
};

template <typename E> struct eval_tree<ScalarMulOp<E>> {
    using type = ScalarMulOp<eval_tree_t<E>>;
};

template <typename E> struct eval_tree<TransposeOp<E>> {
    using type = TransposeOp<eval_tree_t<E>>;
};

template <typename E> struct eval_tree<ConjTransposeOp<E>> {
    using type = ConjTransposeOp<eval_tree_t<E>>;
};

/// Builds the evaluation tree of `expr`, computing the Temporaries it needs.
template <MatrixLeaf M>
[[nodiscard]] constexpr auto make_eval_tree(const MatrixWrapper<M>& expr) -> MatrixWrapper<M>;

template <typename L, typename R>
[[nodiscard]] constexpr auto make_eval_tree(const AddOp<L, R>& expr) -> eval_tree_t<AddOp<L, R>>;

template <typename L, typename R>
[[nodiscard]] constexpr auto make_eval_tree(const SubOp<L, R>& expr) -> eval_tree_t<SubOp<L, R>>;

template <typename L, typename R>
[[nodiscard]] constexpr auto make_eval_tree(const MulOp<L, R>& expr) -> eval_tree_t<MulOp<L, R>>;

template <typename E>
[[nodiscard]] constexpr auto
make_eval_tree(const ScalarMulOp<E>& expr) -> eval_tree_t<ScalarMulOp<E>>;

template <typename E>
[[nodiscard]] constexpr auto
make_eval_tree(const TransposeOp<E>& expr) -> eval_tree_t<TransposeOp<E>>;

template <typename E>
[[nodiscard]] constexpr auto
make_eval_tree(const ConjTransposeOp<E>& expr) -> eval_tree_t<ConjTransposeOp<E>>;

/// Rewrites a MulOp operand: forwarded if it is direct, evaluated into a Temporary otherwise.
template <typename E>
[[nodiscard]] constexpr auto make_product_operand(const E& expr) -> product_operand_t<E>;

} // namespace mlinalg::detail

#include "eval_tree.ipp" // IWYU pragma: keep

#endif // AST_EVAL_TREE_HPP
