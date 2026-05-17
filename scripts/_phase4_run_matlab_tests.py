"""Try to run test_expected_value_solver.m via a shared MATLAB engine
session. Skips if no MATLAB session is found."""

from pathlib import Path
import sys


def main() -> int:
    try:
        import matlab.engine
    except ImportError:
        print("matlab.engine not available -- skipping MATLAB test run")
        return 0

    sessions = list(matlab.engine.find_matlab())
    if not sessions:
        print("No shared MATLAB sessions found. In MATLAB run: matlab.engine.shareEngine")
        return 0

    print(f"Connecting to MATLAB session: {sessions[0]}")
    eng = matlab.engine.connect_matlab(sessions[0])

    rl_path = Path(__file__).parent.parent / "snowdrop_tangled_agents" / "matlab" / "rl"
    eng.addpath(str(rl_path), nargout=0)
    eng.eval("clear classes", nargout=0)

    print("Running test_expected_value_solver...")
    try:
        eng.eval("phase4_test_results = runtests('test_expected_value_solver');", nargout=0)
        n = int(eng.eval("numel(phase4_test_results)", nargout=1))
        passed = int(eng.eval("sum([phase4_test_results.Passed])", nargout=1))
        failed = int(eng.eval("sum([phase4_test_results.Failed])", nargout=1))
        incomplete = int(eng.eval("sum([phase4_test_results.Incomplete])", nargout=1))
        print(f"\nResult: {passed}/{n} passed, {failed} failed, {incomplete} incomplete")
        if failed > 0 or incomplete > 0:
            eng.eval("disp(table(phase4_test_results))", nargout=0)
            return 1
        return 0
    except Exception as e:
        print(f"MATLAB test run failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
