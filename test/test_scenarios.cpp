#include "catch.hpp"

#include "../scenario.h"

const float EPSILON = 0.01f;

TEST_CASE("no dropout", "[scenario]") {

    DropoutScenario scenario = createNoDropoutScenario();

    SECTION("dont_drop") {
        REQUIRE(scenario.dont_drop == true);
    }

    SECTION("dropout values") {
        REQUIRE(scenario.dropouts.size() == 0);
    }
}

TEST_CASE("constant dropout", "[scenario]") {

    float keep_rate = 0.5f;
    int epoch = 100;
    DropoutScenario scenario = createConstantDropoutScenario(keep_rate, epoch);

    SECTION("dont_drop") {
        REQUIRE(scenario.dont_drop == false);
    }

    SECTION("dropout values") {
        REQUIRE(scenario.dropouts.size() == epoch);

        REQUIRE(std::abs(scenario.dropouts[0] - keep_rate) < EPSILON);
        REQUIRE(std::abs(scenario.dropouts[epoch-1] - keep_rate) < EPSILON);
    }
}

TEST_CASE("linear increasing dropout", "[scenario]") {

    float dropout_begin = 0.5f;
    float dropout_end = 1.0f;
    int epoch = 100;
    DropoutScenario scenario = createLinearDropoutScenario(dropout_begin, dropout_end, epoch);

    SECTION("dont_drop") {
        REQUIRE(scenario.dont_drop == false);
    }

    SECTION("dropout values") {
        REQUIRE(scenario.dropouts.size() == epoch);

        REQUIRE(std::abs(scenario.dropouts[0] - dropout_begin) < EPSILON);
        REQUIRE(std::abs(scenario.dropouts[epoch-1] - dropout_end) < EPSILON);

        for (int i = 0; i < epoch - 1; i++) {
            REQUIRE(scenario.dropouts[i] < scenario.dropouts[i+1]);
        }
    }
}

TEST_CASE("concave increasing dropout", "[scenario]") {

    float dropout_begin = 0.5f;
    float dropout_end = 1.0f;
    int epoch = 100;
    DropoutScenario scenario = createConcaveDropoutScenario(dropout_begin, dropout_end, epoch);

    SECTION("dont_drop") {
        REQUIRE(scenario.dont_drop == false);
    }

    SECTION("dropout values") {
        REQUIRE(scenario.dropouts.size() == epoch);

        REQUIRE(std::abs(scenario.dropouts[0] - dropout_begin) < EPSILON);
        REQUIRE(std::abs(scenario.dropouts[epoch-1] - dropout_end) < EPSILON);

        for (int i = 0; i < epoch - 1; i++) {
            REQUIRE(scenario.dropouts[i] < scenario.dropouts[i+1]);
        }
    }
}

TEST_CASE("convex increasing dropout", "[scenario]") {

    float dropout_begin = 0.5f;
    float dropout_end = 1.0f;
    int epoch = 100;
    DropoutScenario scenario = createConvexDropoutScenario(dropout_begin, dropout_end, epoch);

    SECTION("dont_drop") {
        REQUIRE(scenario.dont_drop == false);
    }

    SECTION("dropout values") {
        REQUIRE(scenario.dropouts.size() == epoch);

        REQUIRE(std::abs(scenario.dropouts[0] - dropout_begin) < EPSILON);
        REQUIRE(std::abs(scenario.dropouts[epoch-1] - dropout_end) < EPSILON);

        for (int i = 0; i < epoch - 1; i++) {
            REQUIRE(scenario.dropouts[i] < scenario.dropouts[i+1]);
        }
    }
}


TEST_CASE("concave decreasing dropout", "[scenario]") {

    float dropout_begin = 1.0f;
    float dropout_end = 0.5f;
    int epoch = 100;
    DropoutScenario scenario = createConcaveDecDropoutScenario(dropout_begin, dropout_end, epoch);

    SECTION("dont_drop") {
        REQUIRE(scenario.dont_drop == false);
    }

    SECTION("dropout values") {
        REQUIRE(scenario.dropouts.size() == epoch);

        REQUIRE(std::abs(scenario.dropouts[0] - dropout_begin) < EPSILON);
        REQUIRE(std::abs(scenario.dropouts[epoch-1] - dropout_end) < EPSILON);

        for (int i = 0; i < epoch - 1; i++) {
            REQUIRE(scenario.dropouts[i] > scenario.dropouts[i+1]);
        }
    }
}

TEST_CASE("convex decreasing dropout", "[scenario]") {

    float dropout_begin = 1.0f;
    float dropout_end = 0.5f;
    int epoch = 100;
    DropoutScenario scenario = createConvexDecDropoutScenario(dropout_begin, dropout_end, epoch);

    SECTION("dont_drop") {
        REQUIRE(scenario.dont_drop == false);
    }

    SECTION("dropout values") {
        REQUIRE(scenario.dropouts.size() == epoch);

        REQUIRE(std::abs(scenario.dropouts[0] - dropout_begin) < EPSILON);
        REQUIRE(std::abs(scenario.dropouts[epoch-1] - dropout_end) < EPSILON);

        for (int i = 0; i < epoch - 1; i++) {
            REQUIRE(scenario.dropouts[i] > scenario.dropouts[i+1]);
        }
    }
}

TEST_CASE("half concave increasing dropout", "[scenario]") {

    float dropout_begin = 0.5f;
    float dropout_end = 1.0f;
    int epoch = 100;
    DropoutScenario scenario = halfConcaveDropoutScenario(dropout_begin, dropout_end, epoch);

    SECTION("dont_drop") {
        REQUIRE(scenario.dont_drop == false);
    }

    SECTION("dropout values") {
        REQUIRE(scenario.dropouts.size() == epoch);

        REQUIRE(std::abs(scenario.dropouts[0] - 1.0) < EPSILON);
        REQUIRE(std::abs(scenario.dropouts[epoch/2-1] - 1.0) < EPSILON);

        REQUIRE(std::abs(scenario.dropouts[epoch-1] - dropout_end) < EPSILON);

        for (int i = epoch/2; i < epoch - 1; i++) {
            REQUIRE(scenario.dropouts[i] < scenario.dropouts[i+1]);
        }
    }
}

TEST_CASE("half convex increasing dropout", "[scenario]") {

    float dropout_begin = 0.5f;
    float dropout_end = 1.0f;
    int epoch = 100;
    DropoutScenario scenario = halfConvexDropoutScenario(dropout_begin, dropout_end, epoch);

    SECTION("dont_drop") {
        REQUIRE(scenario.dont_drop == false);
    }

    SECTION("dropout values") {
        REQUIRE(scenario.dropouts.size() == epoch);

        REQUIRE(std::abs(scenario.dropouts[0] - 1.0) < EPSILON);
        REQUIRE(std::abs(scenario.dropouts[epoch/2-1] - 1.0) < EPSILON);

        REQUIRE(std::abs(scenario.dropouts[epoch-1] - dropout_end) < EPSILON);

        for (int i = epoch/2; i < epoch - 1; i++) {
            REQUIRE(scenario.dropouts[i] < scenario.dropouts[i+1]);
        }
    }
}

TEST_CASE("half convex decreasing dropout", "[scenario]") {

    float dropout_begin = 1.0f;
    float dropout_end = .5f;
    int epoch = 100;
    DropoutScenario scenario = halfConvexDecDropoutScenario(dropout_begin, dropout_end, epoch);

    SECTION("dont_drop") {
        REQUIRE(scenario.dont_drop == false);
    }

    SECTION("dropout values") {
        REQUIRE(scenario.dropouts.size() == epoch);

        REQUIRE(std::abs(scenario.dropouts[0] - 1.0) < EPSILON);
        REQUIRE(std::abs(scenario.dropouts[epoch/2-1] - 1.0) < EPSILON);

        REQUIRE(std::abs(scenario.dropouts[epoch-1] - dropout_end) < EPSILON);

        for (int i = epoch/2; i < epoch - 1; i++) {
            REQUIRE(scenario.dropouts[i] > scenario.dropouts[i+1]);
        }
    }
}

TEST_CASE("half concave decreasing dropout", "[scenario]") {

    float dropout_begin = 1.0f;
    float dropout_end = .5f;
    int epoch = 100;
    DropoutScenario scenario = halfConcaveDecDropoutScenario(dropout_begin, dropout_end, epoch);

    SECTION("dont_drop") {
        REQUIRE(scenario.dont_drop == false);
    }

    SECTION("dropout values") {
        REQUIRE(scenario.dropouts.size() == epoch);

        REQUIRE(std::abs(scenario.dropouts[0] - 1.0) < EPSILON);
        REQUIRE(std::abs(scenario.dropouts[epoch/2-1] - 1.0) < EPSILON);

        REQUIRE(std::abs(scenario.dropouts[epoch-1] - dropout_end) < EPSILON);

        for (int i = epoch/2; i < epoch - 1; i++) {
            REQUIRE(scenario.dropouts[i] > scenario.dropouts[i+1]);
        }
    }
}
