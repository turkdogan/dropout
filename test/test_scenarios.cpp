#include "catch.hpp"

#include "../scenario.h"

const float EPSILON = 0.01f;

TEST_CASE("no dropout", "[scenario]") {
    Scenario scenario("NA");

    SECTION("isEnabled()") {
        REQUIRE(scenario.isEnabled() == false);
    }

    SECTION("dropout values") {
        REQUIRE(scenario.size() == 0);
    }
}

TEST_CASE("constant dropout", "[scenario]") {

    float keep_rate = 0.5f;
    int epoch = 100;
    Scenario scenario("C", epoch, keep_rate);

    SECTION("isEnabled()") {
        REQUIRE(scenario.isEnabled() == true);
    }

    SECTION("dropout values") {
        REQUIRE(scenario.size() == epoch);

        REQUIRE(std::abs(scenario.getKeepRate(0) - keep_rate) < EPSILON);
        REQUIRE(std::abs(scenario.getKeepRate(epoch-1) - keep_rate) < EPSILON);
    }
}

TEST_CASE("linear increasing dropout", "[scenario]") {

    float dropout_begin = 0.5f;
    float dropout_end = 1.0f;
    int epoch = 100;
    auto fn = [](int epoch) {return epoch * 2;};
    Scenario scenario("L", epoch, dropout_begin, dropout_end, fn);

    SECTION("isEnabled()") {
        REQUIRE(scenario.isEnabled() == true);
    }

    SECTION("dropout values") {
        REQUIRE(scenario.size() == epoch);

        REQUIRE(std::abs(scenario.getKeepRate(0) - dropout_begin) < EPSILON);
        REQUIRE(std::abs(scenario.getKeepRate(epoch-1) - dropout_end) < EPSILON);

        for (int i = 0; i < epoch - 1; i++) {
            REQUIRE(scenario.getKeepRate(i) < scenario.getKeepRate(i+1));
        }
    }
}

TEST_CASE("concave increasing dropout", "[scenario]") {

    float dropout_begin = 0.5f;
    float dropout_end = 1.0f;
    int epoch = 100;

    auto fn = [](int epoch){return epoch*epoch;};
    Scenario scenario("C", epoch, dropout_begin, dropout_end, fn);

    SECTION("isEnabled()") {
        REQUIRE(scenario.isEnabled() == true);
    }

    SECTION("dropout values") {
        REQUIRE(scenario.size() == epoch);

        REQUIRE(std::abs(scenario.getKeepRate(0) - dropout_begin) < EPSILON);
        REQUIRE(std::abs(scenario.getKeepRate(epoch-1) - dropout_end) < EPSILON);

        for (int i = 0; i < epoch - 1; i++) {
            REQUIRE(scenario.getKeepRate(i) < scenario.getKeepRate(i+1));
        }
    }
}

TEST_CASE("convex increasing dropout", "[scenario]") {

    float dropout_begin = 0.5f;
    float dropout_end = 1.0f;
    int epoch = 100;

    auto fn = [](int epoch){return epoch*epoch;};
    Scenario scenario("C", epoch, dropout_begin, dropout_end, fn);

    SECTION("isEnabled()") {
        REQUIRE(scenario.isEnabled() == true);
    }

    SECTION("dropout values") {
        REQUIRE(scenario.size() == epoch);

        REQUIRE(std::abs(scenario.getKeepRate(0) - dropout_begin) < EPSILON);
        REQUIRE(std::abs(scenario.getKeepRate(epoch-1) - dropout_end) < EPSILON);

        for (int i = 0; i < epoch - 1; i++) {
            REQUIRE(scenario.getKeepRate(i) < scenario.getKeepRate(i+1));
        }
    }
}


TEST_CASE("concave decreasing dropout", "[scenario]") {

    float dropout_begin = 1.0f;
    float dropout_end = 0.5f;
    int epoch = 100;

    auto fn = [](int epoch){return sqrt(epoch);};
    Scenario scenario("C", epoch, dropout_begin, dropout_end, fn);

    SECTION("isEnabled()") {
        REQUIRE(scenario.isEnabled() == true);
    }

    SECTION("dropout values") {
        REQUIRE(scenario.size() == epoch);

        REQUIRE(std::abs(scenario.getKeepRate(0) - dropout_begin) < EPSILON);
        REQUIRE(std::abs(scenario.getKeepRate(epoch-1) - dropout_end) < EPSILON);

        for (int i = 0; i < epoch - 1; i++) {
            REQUIRE(scenario.getKeepRate(i) > scenario.getKeepRate(i+1));
        }
    }
}

TEST_CASE("convex decreasing dropout", "[scenario]") {

    float dropout_begin = 1.0f;
    float dropout_end = 0.5f;
    int epoch = 100;

    auto fn = [](int epoch){return epoch*epoch;};
    Scenario scenario("C", epoch, dropout_begin, dropout_end, fn);

    SECTION("isEnabled()") {
        REQUIRE(scenario.isEnabled() == true);
    }

    SECTION("dropout values") {
        REQUIRE(scenario.size() == epoch);

        REQUIRE(std::abs(scenario.getKeepRate(0) - dropout_begin) < EPSILON);
        REQUIRE(std::abs(scenario.getKeepRate(epoch-1) - dropout_end) < EPSILON);

        for (int i = 0; i < epoch - 1; i++) {
            REQUIRE(scenario.getKeepRate(i) > scenario.getKeepRate(i+1));
        }
    }
}

TEST_CASE("half concave increasing dropout", "[scenario]") {

    float dropout_begin = 0.5f;
    float dropout_end = 1.0f;
    int epoch = 100;

    auto fn = [](int epoch){return sqrt(epoch);};
    Scenario scenario("C", epoch, epoch/2, dropout_begin, dropout_end, fn);

    SECTION("isEnabled()") {
        REQUIRE(scenario.isEnabled() == true);
    }

    SECTION("dropout values") {
        REQUIRE(scenario.size() == epoch);

        REQUIRE(std::abs(scenario.getKeepRate(0) - 1.0) < EPSILON);
        REQUIRE(std::abs(scenario.getKeepRate(epoch/2-1) - 1.0) < EPSILON);

        REQUIRE(std::abs(scenario.getKeepRate(epoch-1) - dropout_end) < EPSILON);

        for (int i = epoch/2; i < epoch - 1; i++) {
            REQUIRE(scenario.getKeepRate(i) < scenario.getKeepRate(i+1));
        }
    }
}

TEST_CASE("half convex increasing dropout", "[scenario]") {

    float dropout_begin = 0.5f;
    float dropout_end = 1.0f;
    int epoch = 100;

    auto fn = [](int epoch){return epoch*epoch;};
    Scenario scenario("C", epoch, epoch/2, dropout_begin, dropout_end, fn);

    SECTION("isEnabled()") {
        REQUIRE(scenario.isEnabled() == true);
    }

    SECTION("dropout values") {
        REQUIRE(scenario.size() == epoch);

        REQUIRE(std::abs(scenario.getKeepRate(0) - 1.0) < EPSILON);
        REQUIRE(std::abs(scenario.getKeepRate(epoch/2-1) - 1.0) < EPSILON);

        REQUIRE(std::abs(scenario.getKeepRate(epoch-1) - dropout_end) < EPSILON);

        for (int i = epoch/2; i < epoch - 1; i++) {
            REQUIRE(scenario.getKeepRate(i) < scenario.getKeepRate(i+1));
        }
    }
}

TEST_CASE("half convex decreasing dropout", "[scenario]") {

    float dropout_begin = 1.0f;
    float dropout_end = .5f;
    int epoch = 100;

    auto fn = [](int epoch){return epoch*epoch;};
    Scenario scenario("HConvexD", epoch, epoch/2, dropout_begin, dropout_end, fn);

    SECTION("isEnabled()") {
        REQUIRE(scenario.isEnabled() == true);
    }

    SECTION("dropout values") {
        REQUIRE(scenario.size() == epoch);

        REQUIRE(std::abs(scenario.getKeepRate(0)- 1.0) < EPSILON);
        REQUIRE(std::abs(scenario.getKeepRate(epoch/2-1) - 1.0) < EPSILON);

        REQUIRE(std::abs(scenario.getKeepRate(epoch-1) - dropout_end) < EPSILON);

        for (int i = epoch/2; i < epoch - 1; i++) {
            REQUIRE(scenario.getKeepRate(i) > scenario.getKeepRate(i+1));
        }
    }
}

TEST_CASE("half concave decreasing dropout", "[scenario]") {

    float dropout_begin = 1.0f;
    float dropout_end = .5f;
    int epoch = 100;

    auto fn = [](int epoch){return sqrt(epoch);};
    Scenario scenario("HCD", epoch, epoch/2, dropout_begin, dropout_end, fn);

    SECTION("isEnabled()") {
        REQUIRE(scenario.isEnabled() == true);
    }

    SECTION("dropout values") {
        REQUIRE(scenario.size() == epoch);

        REQUIRE(std::abs(scenario.getKeepRate(0) - 1.0) < EPSILON);
        REQUIRE(std::abs(scenario.getKeepRate(epoch/2-1) - 1.0) < EPSILON);

        REQUIRE(std::abs(scenario.getKeepRate(epoch-1) - dropout_end) < EPSILON);

        for (int i = epoch/2; i < epoch - 1; i++) {
            REQUIRE(scenario.getKeepRate(i) > scenario.getKeepRate(i+1));
        }
    }
}
