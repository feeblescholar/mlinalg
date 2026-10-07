#include "../src/workflowtest.hpp"
#include <gtest/gtest.h>

TEST(workflowtest, addition) {
    EXPECT_EQ(add(2, 4), 6);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
