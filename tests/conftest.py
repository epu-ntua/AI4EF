import pytest

TEST_RESULTS = []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if report.when == "call":
        if report.passed:
            status, tag = "PASSED", "[OK]"
        elif report.failed:
            status, tag = "FAILED", "[FAIL]"
        else:
            status, tag = "SKIPPED", "[SKIP]"

        TEST_RESULTS.append({
            "name": item.name,
            "status": status,
            "tag": tag,
        })


def pytest_sessionfinish(session, exitstatus):
    print("\n" + "=" * 90)
    print("[SUMMARY] FINAL SUMMARY ")
    print("=" * 90)

    for r in TEST_RESULTS:
        print(f"{r['tag']} {r['name']} — {r['status']}")

    print("=" * 90)
    print(f"[OK]    Passed:  {sum(r['status'] == 'PASSED' for r in TEST_RESULTS)}")
    print(f"[SKIP]  Skipped: {sum(r['status'] == 'SKIPPED' for r in TEST_RESULTS)}")
    print(f"[FAIL]  Failed:  {sum(r['status'] == 'FAILED' for r in TEST_RESULTS)}")
    print("=" * 90)
