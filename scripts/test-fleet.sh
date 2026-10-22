#!/usr/bin/env bash
# Fast offline checks by default; --rpc also builds and runs real CPU RPC inference.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
mkdir -p build/fleet-tests
python3 -m unittest discover -s tests -p 'test_fleet_*.py' -v
cxx=${CXX:-c++}
"$cxx" -std=c++17 -Wall -Wextra -Werror -I ps5/app \
    tests/fleet_worker_config_test.cpp -o build/fleet-tests/worker-config
build/fleet-tests/worker-config
"$cxx" -std=c++17 -Wall -Wextra -Werror -I ps5/app -I third_party/llama.cpp/ggml/include \
    tests/fleet_worker_runtime_test.cpp ps5/app/fleet_worker.cpp -o build/fleet-tests/worker-runtime
build/fleet-tests/worker-runtime
"$cxx" -std=c++17 -O1 ps5/app/model_plan_test.cpp ps5/app/model_plan.cpp -o build/fleet-tests/existing-model-plan
build/fleet-tests/existing-model-plan
bash -n scripts/build-fleet-host.sh scripts/build-app.sh scripts/test-fleet.sh
if [[ ${1:-} == --rpc ]]; then
    scripts/build-fleet-host.sh
    out="${PS5LM_FLEET_BUILD:-$root/build/fleet-host}"
    "${CMAKE:-cmake}" -E env CTEST_OUTPUT_ON_FAILURE=1 \
        "$(dirname "$(command -v "${CMAKE:-cmake}")")/ctest" --test-dir "$out" -R '^test-rpc-multi-server$'
    python3 tests/fleet_rpc_no_comm_test.py --binary "$out/bin/ggml-rpc-server"
    python3 tests/fleet_rpc_smoke.py --build "$out"
elif [[ $# != 0 ]]; then
    echo 'usage: scripts/test-fleet.sh [--rpc]' >&2
    exit 2
fi
