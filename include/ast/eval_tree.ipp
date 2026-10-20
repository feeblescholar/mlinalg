#ifndef AST_EVAL_TREE_IPP
#define AST_EVAL_TREE_IPP

// Definitions for eval_tree.hpp. Do not include this file directly.

namespace mlinalg::detail {

template <MatrixLeaf M>
constexpr auto make_eval_tree(const MatrixWrapper<M>& expr) -> MatrixWrapper<M> {
    return expr;
}

template <MatrixLeaf M>
constexpr auto make_eval_tree(const OwnedMatrix<M>& expr) -> MatrixWrapper<M> {
    return MatrixWrapper<M>(expr.matrix());
}

template <typename L, typename R>
constexpr auto make_eval_tree(const AddOp<L, R>& expr) -> eval_tree_t<AddOp<L, R>> {
    return {make_eval_tree(expr.lhs()), make_eval_tree(expr.rhs())};
}

template <typename L, typename R>
constexpr auto make_eval_tree(const SubOp<L, R>& expr) -> eval_tree_t<SubOp<L, R>> {
    return {make_eval_tree(expr.lhs()), make_eval_tree(expr.rhs())};
}

template <typename L, typename R>
constexpr auto make_eval_tree(const MulOp<L, R>& expr) -> eval_tree_t<MulOp<L, R>> {
    return {make_product_operand(expr.lhs()), make_product_operand(expr.rhs())};
}

template <typename E>
constexpr auto make_eval_tree(const ScalarMulOp<E>& expr) -> eval_tree_t<ScalarMulOp<E>> {
    return {expr.scalar(), make_eval_tree(expr.expression())};
}

template <typename E>
constexpr auto make_eval_tree(const TransposeOp<E>& expr) -> eval_tree_t<TransposeOp<E>> {
    return eval_tree_t<TransposeOp<E>>(make_eval_tree(expr.expression()));
}

template <typename E>
constexpr auto make_eval_tree(const ConjTransposeOp<E>& expr) -> eval_tree_t<ConjTransposeOp<E>> {
    return eval_tree_t<ConjTransposeOp<E>>(make_eval_tree(expr.expression()));
}

template <typename E> constexpr auto make_product_operand(const E& expr) -> product_operand_t<E> {
    if constexpr (is_direct_operand_v<E>) {
        return make_eval_tree(expr);
    } else {
        // The subtree is rewritten first, so products nested inside it get their own Temporary.
        return product_operand_t<E>(make_eval_tree(expr));
    }
}

} // namespace mlinalg::detail

#endif // AST_EVAL_TREE_IPP
