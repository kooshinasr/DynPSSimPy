from collections import defaultdict
import tests.validation.validation_functions as val_fun
import dynpssimpy.dynamic as dps
import importlib
from scipy.integrate import RK45
import matplotlib.pyplot as plt
import pandas as pd
import sys
import time


if __name__ == '__main__':

    import ps_models.ieee39 as model_data
    model = model_data.load()

    [importlib.reload(mdl) for mdl in [model_data, dps, val_fun]]

    ps = dps.PowerSystemModel(model=model)
    ps.pf_max_it = 100
    ps.power_flow()
    ps.init_dyn_sim()
    # print(max(abs(ps.ode_fun(0, ps.x0))))

    t_end = 10
    max_step = 5e-3

    # PowerFactory result
    # pf_res = val_fun.load_pf_res('tests/validation/ieee39/powerfactory_res.csv')  # For interactive mode
    pf_res = val_fun.load_pf_res('powerfactory_res.csv')  # For interactive mode

    x0 = ps.x0

    sol = RK45(ps.ode_fun, 0, x0, t_end, max_step=5e-3)

    t = 0
    result_dict = defaultdict(list)
    sc_bus_idx = ps.gen_bus_idx[0]  # Needed when Kron reduction is bypassed
    t_0 = time.time()  # Timer

    print('Running dynamic simulation')
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t / (t_end) * 100))

        # Simulate next step
        result = sol.step()
        t = sol.t

        if t >= 1 and t <= 1.05:
            # print('Event!')
            ps.y_bus_red_mod[(sc_bus_idx,)*2] = 1e6
        else:
            ps.y_bus_red_mod[(sc_bus_idx,)*2] = 0

        # Store result variables
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, sol.y)]

    print('\nSimulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)

    # Computing the error
    val_fun.generate_plots(ps, result, pf_res, max_step, choose_plots='expanded')

    print(val_fun.compute_error(ps, result, pf_res, max_step))