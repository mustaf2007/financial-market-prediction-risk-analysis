"""
V2.2.7 controlled active-trade acceptance test.
Tests the existing portfolio_returns() function with deterministic inputs.
This is a unit test only; it does not alter the model or historical holdout.
"""
import numpy as np
import financial_ai_system_v227 as system

def main():
    cfg = system.Config(
        cost_bps_per_side=5.0,
        target_volatility=0.15,
        max_position=1.0,
        annualization=252
    )

    # At 15% annualized volatility, position sizing should be 100%,
    # because the target daily volatility equals the supplied daily volatility.
    daily_vol = 0.15 / np.sqrt(252)
    probabilities = np.array([0.70, 0.70, 0.40, 0.80])
    execution_returns = np.array([0.010, -0.010, 0.020, 0.005])
    volatility = np.full(4, daily_vol)

    net, active, position, costs = system.portfolio_returns(
        probabilities, 0.60, execution_returns, volatility, cfg
    )

    expected_active = np.array([1.0, 1.0, 0.0, 1.0])
    expected_position = expected_active.copy()
    expected_costs = expected_active * 0.001  # 10 bps round-trip
    expected_gross = expected_position * execution_returns
    expected_net = expected_gross - expected_costs

    checks = {
        "active_trade_mask": np.array_equal(active, expected_active),
        "position_size": np.allclose(position, expected_position, atol=1e-10),
        "round_trip_costs": np.allclose(costs, expected_costs, atol=1e-10),
        "gross_and_net_returns": np.allclose(net, expected_net, atol=1e-10),
        "inactive_day_has_no_cost": costs[2] == 0.0,
        "active_days_charge_two_sides": np.allclose(costs[active == 1], 0.001)
    }

    print("\nV2.2.7 CONTROLLED ACTIVE-TRADE TEST")
    print("----------------------------------")
    print("Active mask:       ", active.tolist())
    print("Position sizes:    ", np.round(position, 6).tolist())
    print("Gross returns:     ", np.round(expected_gross, 6).tolist())
    print("Transaction costs: ", np.round(costs, 6).tolist())
    print("Net returns:       ", np.round(net, 6).tolist())
    print("\nCHECKS")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")

    if not all(checks.values()):
        raise SystemExit("\nACCEPTANCE RESULT: FAIL")
    print("\nACCEPTANCE RESULT: PASS")
    print("Expected net returns: [0.009, -0.011, 0.0, 0.004]")

if __name__ == "__main__":
    main()