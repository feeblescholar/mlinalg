#include "../include/matrix.hpp"
#include "test.hpp"

#include <complex>
#include <initializer_list>
#include <type_traits>
#include <utility>

using mlinalg::AddOp;
using mlinalg::Dynamic;
using mlinalg::expr_cols_v;
using mlinalg::expr_rows_v;
using mlinalg::Index;
using mlinalg::Matrix;
using mlinalg::MatrixWrapper;
using mlinalg::MulOp;
using mlinalg::RowVector;
using mlinalg::ScalarMulOp;
using mlinalg::SubOp;
using mlinalg::Vector;

namespace {

using M22  = Matrix<double, 2, 2>;
using M23  = Matrix<double, 2, 3>;
using M32  = Matrix<double, 3, 2>;
using M33  = Matrix<double, 3, 3>;
using MX3  = Matrix<double, Dynamic, 3>;
using MXX  = Matrix<double, Dynamic, Dynamic>;
using MF23 = Matrix<float, 2, 3>;
using VX   = Vector<double>;
using RX   = RowVector<double>;

using W23 = MatrixWrapper<M23>;
using W32 = MatrixWrapper<M32>;
using WXX = MatrixWrapper<MXX>;

template <typename L, typename R>
concept Addable = requires { std::declval<L>() + std::declval<R>(); };

template <typename L, typename R>
concept Subtractable = requires { std::declval<L>() - std::declval<R>(); };

template <typename L, typename R>
concept Multipliable = requires { std::declval<L>() * std::declval<R>(); };

template <typename L, typename R>
concept AddAssignable = requires { std::declval<L>() += std::declval<R>(); };

template <typename L, typename R>
concept SubAssignable = requires { std::declval<L>() -= std::declval<R>(); };

template <typename L, typename R>
concept MulAssignable = requires { std::declval<L>() *= std::declval<R>(); };

/// Compare every coefficient against an expected row-major list.
template <typename M> void expect_coeffs(const M& m, std::initializer_list<double> expected) {
    ASSERT_EQ(static_cast<std::size_t>(m.rows() * m.cols()), expected.size());
    const auto* it = expected.begin();
    for (Index i = 0; i < m.rows(); ++i) {
        for (Index j = 0; j < m.cols(); ++j) {
            EXPECT_DOUBLE_EQ(m(i, j), *it++) << "at (" << i << ", " << j << ")";
        }
    }
}

} // namespace

// ------------------------------------------------------------------------------------------------
// Operators return AST proxies, not matrices
// ------------------------------------------------------------------------------------------------

static_assert(std::is_same_v<decltype(std::declval<const M23&>() + std::declval<const M23&>()),
                             AddOp<W23, W23>>);
static_assert(
    std::is_same_v<decltype(std::declval<M23&>() - std::declval<M23&>()), SubOp<W23, W23>>);
static_assert(
    std::is_same_v<decltype(std::declval<M23&>() * std::declval<M32&>()), MulOp<W23, W32>>);
static_assert(std::is_same_v<decltype(2.0 * std::declval<M23&>()), ScalarMulOp<W23>>);
static_assert(std::is_same_v<decltype(std::declval<M23&>() * 2.0), ScalarMulOp<W23>>);

// Expressions are nested by value.
static_assert(
    std::is_same_v<decltype((std::declval<M23&>() + std::declval<M23&>()) * std::declval<M32&>()),
                   MulOp<AddOp<W23, W23>, W32>>);
static_assert(std::is_same_v<decltype(2.0 * (std::declval<MXX&>() - std::declval<MXX&>())),
                             ScalarMulOp<SubOp<WXX, WXX>>>);

// Shape traits flow through the operators (Dynamic wins in coefficient-wise ops).
static_assert(expr_rows_v<decltype(std::declval<M23&>() + std::declval<MXX&>())> == Dynamic);
static_assert(expr_rows_v<decltype(std::declval<MX3&>() * std::declval<M32&>())> == Dynamic);
static_assert(expr_cols_v<decltype(std::declval<MX3&>() * std::declval<M32&>())> == 2);

// Statically incompatible shapes are rejected.
static_assert(!Addable<M23&, M32&> && !Subtractable<M23&, M32&>);
static_assert(!Multipliable<M23&, M23&>);
static_assert(Addable<M23&, MXX&> && Multipliable<M23&, MXX&>); // checked at runtime

// No mixed scalar types, neither between matrices nor for scalar factors.
static_assert(!Addable<M23&, MF23&> && !Multipliable<M23&, Matrix<float, 3, 2>&>);
static_assert(!Multipliable<int, M23&> && !Multipliable<M23&, float>);
static_assert(Multipliable<float, MF23&> && Multipliable<double, M23&>);

// Temporary matrices would dangle; temporary expressions are copied into the node.
static_assert(!Addable<M23, M23&> && !Addable<M23&, M23> && !Multipliable<double, M23>);
static_assert(!Multipliable<M23&&, M32&> && !Multipliable<M23&, const M32&&>);
static_assert(Addable<AddOp<W23, W23>, M23&>);

// Compound assignments return the left-hand side as an lvalue.
static_assert(std::is_same_v<decltype(std::declval<MXX&>() += std::declval<MXX&>()), MXX&>);
static_assert(std::is_same_v<decltype(std::declval<M23&>() -= std::declval<M23&>()), M23&>);
static_assert(std::is_same_v<decltype(std::declval<M23&>() *= std::declval<M33&>()), M23&>);
static_assert(std::is_same_v<decltype(std::declval<M23&>() *= 2.0), M23&>);

// The left-hand side must be a modifiable matrix lvalue.
static_assert(!AddAssignable<const MXX&, MXX&> && !AddAssignable<MXX, MXX&>);
static_assert(!AddAssignable<AddOp<W23, W23>&, M23&>);

// The right-hand side may be a matrix, an expression or a temporary matrix (evaluated at once).
static_assert(AddAssignable<M23&, AddOp<W23, W23>> && SubAssignable<MXX&, const MXX&>);
static_assert(AddAssignable<MXX&, MXX> && MulAssignable<MXX&, MXX&&>);

// The result must fit the left-hand side's static shape.
static_assert(!AddAssignable<M22&, M23&> && !SubAssignable<M23&, M32&>);
static_assert(!MulAssignable<M22&, M23&>); // 2x2 * 2x3 = 2x3 does not fit M22
static_assert(!MulAssignable<M23&, M23&>); // inner dimension mismatch
static_assert(MulAssignable<M23&, M33&> && MulAssignable<M23&, MXX&>);

// No mixed scalar types.
static_assert(!AddAssignable<M23&, MF23&> && !MulAssignable<M23&, int> &&
              !MulAssignable<MF23&, double>);

// ------------------------------------------------------------------------------------------------
// Constexpr evaluation
// ------------------------------------------------------------------------------------------------

static_assert([] {
    const M22 a{{1, 2}, {3, 4}};
    const M22 b{{5, 6}, {7, 8}};
    const M22 c = (a + b) * a - 2.0 * b + a * 1.0;
    // (a + b) * a = [[6, 8], [10, 12]] * a = [[30, 44], [46, 68]]
    return c(0, 0) == 21.0 && c(0, 1) == 34.0 && c(1, 0) == 35.0 && c(1, 1) == 56.0;
}());

static_assert([] {
    M22       a{{1, 2}, {3, 4}};
    const M22 b{{1, 1}, {1, 1}};
    a += b;       // [[2, 3], [4, 5]]
    a -= 2.0 * b; // [[0, 1], [2, 3]]
    a *= b;       // [[1, 1], [5, 5]]
    a *= 3.0;     // [[3, 3], [15, 15]]
    return a(0, 0) == 3.0 && a(0, 1) == 3.0 && a(1, 0) == 15.0 && a(1, 1) == 15.0;
}());

// ------------------------------------------------------------------------------------------------
// Runtime behaviour
// ------------------------------------------------------------------------------------------------

TEST(MatrixOperators, AddAndSub) {
    const MXX a{{1, 2, 3}, {4, 5, 6}};
    const MXX b{{6, 5, 4}, {3, 2, 1}};
    const MXX sum  = a + b;
    const MXX diff = a - b;
    expect_coeffs(sum, {7, 7, 7, 7, 7, 7});
    expect_coeffs(diff, {-5, -3, -1, 1, 3, 5});
}

TEST(MatrixOperators, MixedFixedAndDynamicOperands) {
    const M23 a{{1, 2, 3}, {4, 5, 6}};
    const MX3 b{{1, 1, 1}, {2, 2, 2}};
    const MXX c{{1, 0}, {0, 1}, {1, 1}};
    const M22 result = (a - b) * c;
    expect_coeffs(result, {2, 3, 6, 7});
}

TEST(MatrixOperators, MatrixProduct) {
    const MXX a{{1, 2, 3}, {4, 5, 6}};
    const MXX b{{7, 8}, {9, 10}, {11, 12}};
    const MXX p = a * b;
    EXPECT_EQ(p.rows(), 2);
    EXPECT_EQ(p.cols(), 2);
    expect_coeffs(p, {58, 64, 139, 154});
}

TEST(MatrixOperators, ScalarOnEitherSide) {
    const MXX a{{1, -2}, {3, -4}};
    const MXX left  = 0.5 * a;
    const MXX right = a * 0.5;
    expect_coeffs(left, {0.5, -1, 1.5, -2});
    expect_coeffs(right, {0.5, -1, 1.5, -2});

    const double k      = 3.0;
    const MXX    scaled = k * (a + a);
    expect_coeffs(scaled, {6, -12, 18, -24});
}

TEST(MatrixOperators, SpecExampleBPlusCTimesD) {
    const M33 b{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    const M33 c{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    const M33 d{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}};
    M33       a;
    a = b + c * d;
    expect_coeffs(a, {2, 4, 9, 4, 11, 18, 7, 16, 28});
}

TEST(MatrixOperators, EvaluationIsLazy) {
    MXX        a{{1, 2}, {3, 4}};
    const MXX  b{{10, 20}, {30, 40}};
    const auto expr = a + b; // refers to a and b, nothing computed yet
    a(0, 0)         = 100.0;
    const MXX sum   = expr;
    expect_coeffs(sum, {110, 22, 33, 44});
}

TEST(MatrixOperators, ExpressionLvalueCanBeReused) {
    const MXX  a{{1, 2}, {3, 4}};
    const auto twice = a + a;
    const MXX  four  = twice + twice;
    const MXX  sq    = twice * twice;
    expect_coeffs(four, {4, 8, 12, 16});
    expect_coeffs(sq, {28, 40, 60, 88});
}

TEST(MatrixOperators, AliasedAssignment) {
    MXX       a{{1, 2}, {3, 4}};
    const MXX b{{5, 6}, {7, 8}};
    a = a * b;
    expect_coeffs(a, {19, 22, 43, 50});
    a = a - a * 0.5;
    expect_coeffs(a, {9.5, 11, 21.5, 25});
}

TEST(MatrixOperators, Vectors) {
    const MXX m{{1, 2}, {3, 4}};
    const VX  x{1, -1};
    const RX  r{2, 3};
    const VX  y = m * x + x;
    EXPECT_DOUBLE_EQ(y[0], 0.0);
    EXPECT_DOUBLE_EQ(y[1], -2.0);

    const Matrix<double, 1, 1> dot = r * x; // row * column
    EXPECT_DOUBLE_EQ(dot(0, 0), -1.0);

    const MXX outer = x * r; // column * row
    expect_coeffs(outer, {2, 3, -2, -3});
}

TEST(MatrixOperators, ComplexScalars) {
    using C = std::complex<double>;
    const Matrix<C, 2, 2> a{{C(1, 1), C(0, 0)}, {C(0, 0), C(1, -1)}};
    const Matrix<C, 2, 2> b = C(0, 1) * a + a * a;
    EXPECT_EQ(b(0, 0), C(-1, 1) + C(0, 2));
    EXPECT_EQ(b(1, 1), C(1, 1) + C(0, -2));
    EXPECT_EQ(b(0, 1), C(0, 0));
}

TEST(MatrixOperators, IntegerScalars) {
    const Matrix<int, 2, 2> a{{1, 2}, {3, 4}};
    const Matrix<int, 2, 2> b = 2 * a - a * a;
    EXPECT_EQ(b(0, 0), -5);
    EXPECT_EQ(b(0, 1), -6);
    EXPECT_EQ(b(1, 0), -9);
    EXPECT_EQ(b(1, 1), -14);
}

// ------------------------------------------------------------------------------------------------
// Compound assignment
// ------------------------------------------------------------------------------------------------

TEST(MatrixCompoundAssignment, AddAndSub) {
    MXX       a{{1, 2}, {3, 4}};
    const MXX b{{10, 20}, {30, 40}};
    a += b;
    expect_coeffs(a, {11, 22, 33, 44});
    a -= b - b * 0.5;
    expect_coeffs(a, {6, 12, 18, 24});
}

TEST(MatrixCompoundAssignment, ReturnsLhsAndChains) {
    MXX       a{{1, 2}, {3, 4}};
    const MXX b{{1, 1}, {1, 1}};
    MXX&      ref = (a += b) -= 2.0 * b;
    EXPECT_EQ(&ref, &a);
    expect_coeffs(a, {0, 1, 2, 3});
}

TEST(MatrixCompoundAssignment, ScalarMultiply) {
    MXX a{{1, -2}, {3, -4}};
    a *= -2.0;
    expect_coeffs(a, {-2, 4, -6, 8});
}

TEST(MatrixCompoundAssignment, MatrixProductChangesDynamicShape) {
    MXX       a{{1, 2, 3}, {4, 5, 6}};
    const MXX b{{1}, {0}, {1}};
    a *= b;
    EXPECT_EQ(a.rows(), 2);
    EXPECT_EQ(a.cols(), 1);
    expect_coeffs(a, {4, 10});
}

TEST(MatrixCompoundAssignment, FixedShapeProduct) {
    M23       a{{1, 2, 3}, {4, 5, 6}};
    const M33 p{{0, 0, 1}, {0, 1, 0}, {1, 0, 0}}; // reverses the columns
    a *= p;
    expect_coeffs(a, {3, 2, 1, 6, 5, 4});
}

TEST(MatrixCompoundAssignment, SelfAliasing) {
    MXX a{{1, 2}, {3, 4}};
    a += a;
    expect_coeffs(a, {2, 4, 6, 8});
    a *= a;
    expect_coeffs(a, {28, 40, 60, 88});
    a -= a;
    expect_coeffs(a, {0, 0, 0, 0});
}

TEST(MatrixCompoundAssignment, TemporaryRightHandSide) {
    MXX a{{1, 2}, {3, 4}};
    a += MXX{{1, 1}, {1, 1}};
    expect_coeffs(a, {2, 3, 4, 5});
    a *= MXX{{0, 1}, {1, 0}};
    expect_coeffs(a, {3, 2, 5, 4});
}

TEST(MatrixCompoundAssignment, Vectors) {
    VX       v{1, 2, 3};
    const VX w{1, 1, 1};
    v -= 2.0 * w;
    EXPECT_EQ(v[0], -1.0);
    EXPECT_EQ(v[1], 0.0);
    EXPECT_EQ(v[2], 1.0);
}
