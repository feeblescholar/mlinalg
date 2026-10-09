#include "../include/ast/ast.hpp"
#include "test.hpp"

#include <complex>
#include <cstddef>
#include <initializer_list>
#include <type_traits>

using mlinalg::AddOp;
using mlinalg::Dynamic;
using mlinalg::expr_cols_v;
using mlinalg::expr_rows_v;
using mlinalg::expr_scalar_t;
using mlinalg::Expression;
using mlinalg::Index;
using mlinalg::MatrixWrapper;
using mlinalg::MulOp;
using mlinalg::ScalarMulOp;
using mlinalg::SubOp;

namespace {

/// Minimal row-major matrix satisfying MatrixLeaf, standing in for the Module 4 Matrix.
template <typename T, int Rows, int Cols> class TestMatrix {
  public:
    using value_type                       = T;
    static constexpr int RowsAtCompileTime = Rows;
    static constexpr int ColsAtCompileTime = Cols;

    constexpr TestMatrix(Index rows, Index cols, std::initializer_list<T> values)
        : m_storage(rows, cols) {
        Index i = 0;
        for (const T& value : values) {
            m_storage[i++] = value;
        }
    }

    [[nodiscard]] constexpr auto rows() const -> Index { return m_storage.rows(); }
    [[nodiscard]] constexpr auto cols() const -> Index { return m_storage.cols(); }

    [[nodiscard]] constexpr auto operator()(Index row, Index col) const -> const T& {
        return m_storage[(row * cols()) + col];
    }
    [[nodiscard]] constexpr auto operator()(Index row, Index col) -> T& {
        return m_storage[(row * cols()) + col];
    }

  private:
    mlinalg::MatrixStorage<T, Rows, Cols> m_storage;
};

using M23  = TestMatrix<double, 2, 3>;
using M32  = TestMatrix<double, 3, 2>;
using M34  = TestMatrix<double, 3, 4>;
using MX3  = TestMatrix<double, Dynamic, 3>;
using MXX  = TestMatrix<double, Dynamic, Dynamic>;
using MF23 = TestMatrix<float, 2, 3>;

using W23  = MatrixWrapper<M23>;
using W32  = MatrixWrapper<M32>;
using W34  = MatrixWrapper<M34>;
using WX3  = MatrixWrapper<MX3>;
using WXX  = MatrixWrapper<MXX>;
using WF23 = MatrixWrapper<MF23>;

template <typename E, int R, int C, typename S>
constexpr bool has_traits =
    expr_rows_v<E> == R && expr_cols_v<E> == C && std::is_same_v<expr_scalar_t<E>, S>;

/// Fully materialize an expression's coefficients and compare against an expected row-major list.
template <Expression E> void expect_coeffs(const E& expr, std::initializer_list<double> expected) {
    ASSERT_EQ(static_cast<std::size_t>(expr.rows() * expr.cols()), expected.size());
    const auto* it = expected.begin();
    for (Index i = 0; i < expr.rows(); ++i) {
        for (Index j = 0; j < expr.cols(); ++j) {
            EXPECT_DOUBLE_EQ(expr.coeff(i, j), *it++) << "at (" << i << ", " << j << ")";
        }
    }
}

} // namespace

// ------------------------------------------------------------------------------------------------
// Compile-time traits
// ------------------------------------------------------------------------------------------------

static_assert(Expression<W23> && Expression<AddOp<W23, W23>> && Expression<SubOp<W23, W23>> &&
              Expression<MulOp<W23, W34>> && Expression<ScalarMulOp<W23>>);

// Leaves take their traits from the wrapped matrix.
static_assert(has_traits<W23, 2, 3, double>);
static_assert(has_traits<WX3, Dynamic, 3, double>);
static_assert(has_traits<WF23, 2, 3, float>);

// Coefficient-wise nodes: equal static sizes are kept; mixing Dynamic and static gives Dynamic.
static_assert(has_traits<AddOp<W23, W23>, 2, 3, double>);
static_assert(has_traits<SubOp<W23, W23>, 2, 3, double>);
static_assert(has_traits<AddOp<W23, WXX>, Dynamic, Dynamic, double>);
static_assert(has_traits<AddOp<WXX, W23>, Dynamic, Dynamic, double>);
static_assert(has_traits<SubOp<W23, WX3>, Dynamic, 3, double>);

// Product: rows from lhs, cols from rhs.
static_assert(has_traits<MulOp<W23, W34>, 2, 4, double>);
static_assert(has_traits<MulOp<WX3, W34>, Dynamic, 4, double>);
static_assert(has_traits<MulOp<W23, WXX>, 2, Dynamic, double>);

// Scalar multiplication preserves the operand's traits.
static_assert(has_traits<ScalarMulOp<W23>, 2, 3, double>);
static_assert(has_traits<ScalarMulOp<MulOp<WX3, W34>>, Dynamic, 4, double>);

// Nested trees propagate traits.
static_assert(
    has_traits<AddOp<W23, MulOp<W23, MatrixWrapper<TestMatrix<double, 3, 3>>>>, 2, 3, double>);

// Statically incompatible operands are rejected by the node constraints.
static_assert(!mlinalg::detail::CwiseCompatible<W23, W32>);   // shape mismatch
static_assert(!mlinalg::detail::CwiseCompatible<W23, WF23>);  // scalar type mismatch
static_assert(!mlinalg::detail::ProductCompatible<W23, W23>); // inner 3 != 2
static_assert(mlinalg::detail::ProductCompatible<W23, W32>);
static_assert(mlinalg::detail::CwiseCompatible<W23, WXX>); // decided at runtime

// A leaf must not bind to a temporary matrix.
static_assert(std::is_constructible_v<W23, const M23&>);
static_assert(!std::is_constructible_v<W23, M23&&>);

// ------------------------------------------------------------------------------------------------
// Constexpr construction and evaluation (static matrices only)
// ------------------------------------------------------------------------------------------------

static_assert([] {
    const M23 a(2, 3, {1, 2, 3, 4, 5, 6});
    const M23 b(2, 3, {6, 5, 4, 3, 2, 1});
    const M32 c(3, 2, {1, 0, 0, 1, 1, 1});

    // (a + b) * c - 2 * (a * c)
    const MulOp   product(AddOp(W23{a}, W23{b}), W32{c});
    const MulOp   ac(W23{a}, W32{c});
    const SubOp   expr(product, ScalarMulOp(2.0, ac));
    constexpr int rows = expr_rows_v<decltype(expr)>; // dimension deduction is a constant expr

    // (a + b) = all 7s, so (a + b) * c = [[14, 14], [14, 14]]; a * c = [[4, 5], [10, 11]]
    return rows == 2 && expr.rows() == 2 && expr.cols() == 2 && expr.coeff(0, 0) == 6.0 &&
           expr.coeff(0, 1) == 4.0 && expr.coeff(1, 0) == -6.0 && expr.coeff(1, 1) == -8.0;
}());

// ------------------------------------------------------------------------------------------------
// Runtime behaviour
// ------------------------------------------------------------------------------------------------

TEST(MatrixWrapper, ForwardsDimensionsAndCoefficients) {
    const MXX m(2, 3, {1, 2, 3, 4, 5, 6});
    const WXX w(m);
    EXPECT_EQ(w.rows(), 2);
    EXPECT_EQ(w.cols(), 3);
    EXPECT_EQ(&w.matrix(), &m);
    expect_coeffs(w, {1, 2, 3, 4, 5, 6});
}

TEST(AddOp, DynamicOperands) {
    const MXX   a(2, 2, {1, 2, 3, 4});
    const MXX   b(2, 2, {10, 20, 30, 40});
    const AddOp sum(WXX{a}, WXX{b});
    EXPECT_EQ(sum.rows(), 2);
    EXPECT_EQ(sum.cols(), 2);
    expect_coeffs(sum, {11, 22, 33, 44});
}

TEST(SubOp, MixedStaticAndDynamicOperands) {
    const M23   a(2, 3, {1, 2, 3, 4, 5, 6});
    const MX3   b(2, 3, {1, 1, 1, 2, 2, 2});
    const SubOp diff(W23{a}, WX3{b});
    static_assert(expr_rows_v<decltype(diff)> == Dynamic && expr_cols_v<decltype(diff)> == 3);
    EXPECT_EQ(diff.rows(), 2);
    EXPECT_EQ(diff.cols(), 3);
    expect_coeffs(diff, {0, 1, 2, 2, 3, 4});
}

TEST(MulOp, NonSquareProduct) {
    const MXX   a(2, 3, {1, 2, 3, 4, 5, 6});
    const MXX   b(3, 2, {7, 8, 9, 10, 11, 12});
    const MulOp product(WXX{a}, WXX{b});
    EXPECT_EQ(product.rows(), 2);
    EXPECT_EQ(product.cols(), 2);
    expect_coeffs(product, {58, 64, 139, 154});
}

TEST(MulOp, RowTimesColumnIsDotProduct) {
    const MXX row(1, 4, {1, 2, 3, 4});
    const MXX col(4, 1, {5, 6, 7, 8});
    expect_coeffs(MulOp(WXX{row}, WXX{col}), {70});
    expect_coeffs(MulOp(WXX{col}, WXX{row}),
                  {5, 10, 15, 20, 6, 12, 18, 24, 7, 14, 21, 28, 8, 16, 24, 32});
}

TEST(ScalarMulOp, ScalesEveryCoefficient) {
    const MXX         m(2, 2, {1, -2, 3, -4});
    const ScalarMulOp scaled(-0.5, WXX{m});
    EXPECT_EQ(scaled.scalar(), -0.5);
    expect_coeffs(scaled, {-0.5, 1, -1.5, 2});
}

TEST(Ast, SpecExampleBPlusCTimesD) {
    // A = B + C * D from the spec, as a single unevaluated tree.
    const MXX   b(2, 2, {1, 1, 1, 1});
    const MXX   c(2, 3, {1, 2, 3, 4, 5, 6});
    const MXX   d(3, 2, {1, 0, 0, 1, 1, 1});
    const AddOp expr(WXX{b}, MulOp(WXX{c}, WXX{d}));
    expect_coeffs(expr, {5, 6, 11, 12});
}

TEST(Ast, EvaluationIsLazy) {
    // Leaves refer to the matrices, so the tree sees updates made after it was built.
    MXX         a(1, 2, {1, 2});
    const MXX   b(1, 2, {10, 20});
    const AddOp sum(WXX{a}, WXX{b});
    a(0, 1) = 100;
    expect_coeffs(sum, {11, 120});
}

TEST(Ast, NodesAreCopyableValues) {
    const MXX   a(1, 2, {1, 2});
    const AddOp sum(WXX{a}, WXX{a});
    auto        copy = sum; // inner nodes are held by value, leaves by reference
    expect_coeffs(copy, {2, 4});
    EXPECT_EQ(&copy.lhs().matrix(), &a);
}

TEST(Ast, ComplexScalars) {
    using C   = std::complex<double>;
    using MC  = TestMatrix<C, Dynamic, Dynamic>;
    using WMC = MatrixWrapper<MC>;
    const MC    a(1, 2, {C(1, 1), C(0, 2)});
    const MC    b(2, 1, {C(2, 0), C(0, -1)});
    const MulOp product(WMC{a}, WMC{b});
    static_assert(std::is_same_v<expr_scalar_t<decltype(product)>, C>);
    // (1+i)*2 + (2i)*(-i) = 2 + 2i + 2
    EXPECT_EQ(product.coeff(0, 0), C(4, 2));
    EXPECT_EQ(ScalarMulOp(C(0, 1), WMC{a}).coeff(0, 0), C(-1, 1));
}
