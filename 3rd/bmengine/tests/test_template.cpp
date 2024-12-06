#include "bmengine/core/core.h"
#include "bmengine/functions/index_select.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <cuda.h>

using namespace bmengine;
using namespace bmengine::core;
using std::vector;

void test_xxx(const Context& ctx) {

}

int main() {
    core::Engine engine({
            {0, 1 * 1024 * 1024 * 1024},
     });
    auto ctx = engine.create_context();
    auto with_dev = ctx.with_device(0);

    test_xxx(ctx);
    return 0;
}