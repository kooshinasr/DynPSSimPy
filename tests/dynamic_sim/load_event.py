import dynpssimpy.dynamic as dps
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import RK23, RK45, solve_ivp
import importlib
import time
import sys
import dynpssimpy.modal_analysis as dps_mdl
import dynpssimpy.plotting as dps_plt


if __name__ == '__main__':

    # Load model
    import ps_models.smib_tet4180 as model_data
    # import ps_models.ieee39 as model_data
    # import ps_models.n44 as model_data

    [importlib.reload(mdl) for mdl in [dps, model_data]]
    model = model_data.load()
    t_0 = time.time()

    if not model['gov_on']: model.pop('gov', None)
    if not model['avr_on']: model.pop('avr', None)
    if not model['pss_on']: model.pop('pss', None)

    ps = dps.PowerSystemModel(model=model)
    ps.pf_max_it = 10 # Max iterations for power flow
    ps.power_flow()

    # Print power flow results
    float_formatter = "{:.3f}".format
    np.set_printoptions(formatter={'float_kind': float_formatter})
    print('   Gen  |  Load |  Grid')
    print('P: {}'.format(np.real(ps.s_0)))
    print('Q: {}'.format(np.imag(ps.s_0)))
    print('V: {}'.format(np.abs(ps.v_0)))
    print('d: {}'.format(np.angle(ps.v_0)*180/np.pi))

    ps.init_dyn_sim()
    ps.ode_fun(0.0, ps.x0)
    x0 = ps.x0.copy()
    t = 0
    t_end = 30 # End of simulation
    sol = RK45(ps.ode_fun, 0, x0, t_end, max_step=2e-2)
    result_dict = defaultdict(list)

    # Additional plot variables below. All states are stored by default, but algebraic variables like powers and
    # currents have to be specified in the lists below. Currently supports GEN output and input variables, and
    # powers ['P_l', 'Q_l', 'S_l'] at load buses.
    #avr_outs = []
    #gov_outs = []
    gen_vars = ['P_e', 'I_g','P_m']
    load_vars = ['P_l', 'Q_l']  # l subscript means "load"

    gen_var_desc = ps.var_desc('GEN',gen_vars)
    load_var_desc = ps.var_desc('load',load_vars)

    event_flag = True
    event_flag2 = True
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/t_end*100))

        if t >= 1 and event_flag:
            event_flag = False

            ps.network_event('line','L1-2', 'disconnect')

        if t >= 15 and event_flag2:
            ps.network_event('load_change','L1', 'activate', dS = 5)  # dS in MVA (not pu), can be complex

            event_flag2 = False

        # Simulate next step
        result = sol.step()
        x = sol.y
        t = sol.t

        # Store result
        result_dict['Global', 't'].append(sol.t)                                                # Time
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]       # States
        ps.store_vars('GEN',gen_vars, gen_var_desc, result_dict)                                # Additional gen vars
        ps.store_vars('load',load_vars, load_var_desc, result_dict)                             # Load vars

    print('   Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))
    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)
    t_plot = result[('Global', 't')]
    # Plotting section
    fig, ax = plt.subplots(2, sharex = True)

    var1 = 'speed'                                      # variable to plot
    p1 = result.xs(key=var1, axis='columns', level=1)   # time domain values for var1
    #p1 = p1[['G1']]                                     # Double brackets to access specific devices (e.g. G1)
    legnd1 = list(np.array(var1 + ': ')+p1.columns)     # legend for var1

    var2 = 'P_l'                                      # variable to plot
    p2 = result.xs(key=var2, axis='columns', level=1)   # time domain values for var2
    #p2 = p2[['G1']]                                     # Double brackets to access specific devices (e.g. G1)
    legnd2 = list(np.array(var2 + ': ') + p2.columns)   # legend for var2

    var3 = 'P_e'  # variable to plot
    p3 = result.xs(key=var3, axis='columns', level=1)
    #p3 = p3[['G1']]
    legnd3 = list(np.array(var3 + ': ') + p3.columns)

    var4 = 'P_m'  # variable to plot
    p4 = result.xs(key=var4, axis='columns', level=1)
    p4 = p4[['G1']]
    legnd4 = list(np.array(var4 + ': ') + p4.columns)

    ax[0].plot(t_plot, p1)
    ax[0].legend(legnd1)
    ax[0].set_ylabel('Speed')


    ax[1].plot(t_plot, p2)
    ax[1].plot(t_plot, p3)                              # Plotting two variables in same plot
    ax[1].plot(t_plot, p4)
    ax[1].legend(legnd2 + legnd3 + legnd4)
    ax[1].set_ylabel('Power')

    fig.text(0.5, 0.04, 'Time [seconds]', ha='center')
    plt.show()