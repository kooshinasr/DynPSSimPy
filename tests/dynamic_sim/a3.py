import dynpssimpy.dynamic as dps
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import RK23, RK45, solve_ivp
import importlib
import time


if __name__ == '__main__':

    # Load model
    import ps_models.smib_tet4180 as model_data
    # import ps_models.ieee39 as model_data
    # import ps_models.n44 as model_data

    [importlib.reload(mdl) for mdl in [dps, model_data]]
    model = model_data.load()

    t_0 = time.time()

    ps = dps.PowerSystemModel(model=model)
    ps.pf_max_it = 100
    # ps.use_numba = True
    ps.power_flow()
    print(ps.s_0)
    print(np.abs(ps.v_0))
    print(ps.v_0)
    ps.init_dyn_sim()
    print(ps.state_desc)
    print(ps.x0)
    ps.build_y_bus_red()
    ps.ode_fun(0.0, ps.x0)
    t_end = 1
    x0 = ps.x0.copy()
    #x0[ps.gen_mdls['GEN'].state_idx['angle'][0]] += 1

    sol = RK45(ps.ode_fun, 0, x0, t_end, max_step=1e-2)

    t = 0
    result_dict = defaultdict(list)

    event_flag = False
    event_flag2 = False
    while t < t_end:
        print(t)

        # Simulate next step
        result = sol.step()
        x = sol.y
        t = sol.t

        if t >= 1 and event_flag:
            event_flag = False
            ps.network_event('line', 'L1-2', 'disconnect')

        if t >= 1.3 and event_flag2:
            event_flag2 = False
            ps.network_event('line', 'L1-2', 'connect')

        # Store result
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)

    fig, ax = plt.subplots(2)
    p1 = result.xs(key='speed', axis='columns', level=1)
    p2 = result.xs(key='angle', axis='columns', level=1)

    ax[0].plot(result[('Global', 't')], p1)
    ax[0].legend(p1.columns)
    ax[1].plot(result[('Global', 't')], p2)
    ax[1].legend(p2.columns)
    print(result['G1'])
    #print(result.xs(key='speed', axis='columns', level=1).columns)
    plt.show()
