import dynpssimpy.dynamic as dps
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import RK23, RK45, solve_ivp
import importlib
import time
import sys




if __name__ == '__main__':

    # Load model
    import ps_models.smib_tet4180 as model_data
    # import ps_models.ieee39 as model_data
    # import ps_models.n44 as model_data

    [importlib.reload(mdl) for mdl in [dps, model_data]]
    model = model_data.load()
    t_0 = time.time()

    ps = dps.PowerSystemModel(model=model)
    ps.pf_max_it = 10 # Max iterations for power flow
    ps.power_flow()

    ps.init_dyn_sim()
    ## Print power flow results
    #print(ps.s_0)
    #print(np.abs(ps.v_0))
    #print(ps.v_0)
    ps.ode_fun(0.0, ps.x0)


    x0 = ps.x0.copy()
    #x0[ps.gen_mdls['GEN'].state_idx['angle'][0]] += 1


    t_end = 3 # End of simulation

    sol = RK45(ps.ode_fun, 0, x0, t_end, max_step=1e-2)
    t = 0
    result_dict = defaultdict(list)

    #avr_outs = []
    #gov_outs = []
    gen_vars = ['P_e', 'I_g']
    bus_vars = ['v']

    gen_var_desc = ps.var_desc('GEN',gen_vars)
    bus_var_desc = ps.var_desc('bus',bus_vars)

    event_flag = True
    event_flag2 = True
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        # Simulate next step
        result = sol.step()
        x = sol.y
        t = sol.t

        if t >= 1 and event_flag:
            event_flag = False
            ps.network_event('line', 'L1-2', 'disconnect')
            #ps.network_event('sc','B2', 'activate')

        if t >= 1.05 and event_flag2:
            event_flag2 = False
            #ps.network_event('line', 'L1-2', 'connect')
            #ps.network_event('sc', 'B2', 'deactivate')

        # Store result
        result_dict['Global', 't'].append(sol.t)                                                # Time
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]       # States
        ps.store_vars('GEN',gen_vars, gen_var_desc, result_dict)                                # Additional gen vars
        ps.store_vars('bus',bus_vars, bus_var_desc, result_dict)                                # Additional bus vars

    print('   Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))
    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)
    t_plot = result[('Global', 't')]

    # Plotting section
    fig, ax = plt.subplots(2, sharex = True)

    var1 = 'I_g'                                        # variable to plot
    p1 = result.xs(key=var1, axis='columns', level=1)   # time domain values for var1
    legnd1 = list(np.array(var1 + ': ')+p1.columns)     # legend for var1

    ax[0].plot(t_plot, p1)
    ax[0].legend(legnd1)
    #ax[0].set_ylabel('Electrical power')

    var2 = 'angle'                                      # variable to plot
    p2 = result.xs(key=var2, axis='columns', level=1)   # time domain values for var2
    legnd2 = list(np.array(var2 + ': ') + p2.columns)   # legend for var2

    var3 = 'P_e'  # variable to plot
    p3 = result.xs(key=var3, axis='columns', level=1)
    legnd3 = list(np.array(var3 + ': ') + p3.columns)

    ax[1].plot(t_plot, p2)
    #ax[1].plot(t_plot, p3)                             # Plotting two variables in same plot
    print(result)
    ax[1].legend(legnd2+ legnd3)

    # Plot different variables together in same subplot
    #ax[1].plot(t_plot, p2)
    #ax[1].plot(t_plot, p3)
    #ax[1].legend(legnd2 + legnd3)

    #p2 = list(p1.columns)+list(p2.columns)

    #ax[1].plot(t_plot, p1)
    #ax[1].plot(t_plot, p2)
    #ax[1].legend(p2)
    #ax[1].set_ylabel('Angle & El power')
    ##print(result.xs(key='speed', axis='columns', level=1).columns)

    fig.text(0.5, 0.04, 'Time [seconds]', ha='center')
    plt.show()
