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


def linear_comparison(t_plot, k, x0, dx0, ps_lin):
    """
    Compares nonlinear response from simulation with linear response calculated from modal analysis and dx0.

    :param t_plot: From simulation
    :param k: Index for starting time for computing linear response
    :param x0: initial conditions
    :param dx0: delta x0 at t_plot[k]
    :param ps_lin: linear power system instance
    :return: x_lin: time response for all state variables for t[k:]
    """
    c_vec = np.dot(ps_lin.lev, dx0)

    t_arr = t_plot[k:].values-t_plot[k]
    x_lin = np.outer(np.ones(len(t_plot)),x0)

    exp_t = np.multiply(np.exp(np.outer(t_arr,ps_lin.eigs)),c_vec)

    for i, eig in enumerate(ps_lin.eigs):
        dx_i = np.dot(exp_t, ps_lin.rev[i,:])
        x_lin[k:,i] += np.real(dx_i)

    return x_lin


if __name__ == '__main__':

    # Load model
    import ps_models.k2a as model_data

    [importlib.reload(mdl) for mdl in [dps, model_data]]
    model = model_data.load()
    t_0 = time.time()

    if not model['gov_on']: model.pop('gov', None)
    if not model['avr_on']: model.pop('avr', None)
    if not model['pss_on']: model.pop('pss', None)

    ps = dps.PowerSystemModel(model=model)
    ps.pf_max_it = 10 # Max iterations for power flow
    ps.power_flow()

    ps.init_dyn_sim()
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize()
    ps.ode_fun(0.0, ps.x0)
    x0 = ps.x0.copy()
    t = 0
    t_end = 15 # End of simulation
    sol = RK45(ps.ode_fun, 0, x0, t_end, max_step=1e-2)
    result_dict = defaultdict(list)

    gen_vars = ['P_e', 'I_g', 'P_m', 'V_t_abs',  'Q']
    load_vars = ['P_l','Q_l']  # l subscript means "load"

    gen_var_desc = ps.var_desc('GEN',gen_vars)
    load_var_desc = ps.var_desc('load',load_vars)

    event_flag = True
    event_flag2 = True
    k = 0
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/t_end*100))

        if t >= 1 and event_flag:
            event_flag = False
            ps.network_event('sc','B7', 'activate')

        if t >= 1.02 and event_flag2:
            event_flag2 = False
            ps.network_event('sc', 'B7', 'deactivate')
            dx = x - x0
            t_ind = k-1

        # Simulate next step
        result = sol.step()
        x = sol.y
        t = sol.t
        k+=1
        # Store result
        result_dict['Global', 't'].append(sol.t)                                                # Time
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]       # States
        ps.store_vars('GEN',gen_vars, gen_var_desc, result_dict)                                # Additional gen vars
        ps.store_vars('load',load_vars, load_var_desc, result_dict)                             # Load vars

    print('   Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))
    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)
    t_plot = result[('Global', 't')]
    x_lin = linear_comparison(t_plot,t_ind,x0,dx,ps_lin)
    x_lin = pd.DataFrame(x_lin, columns=index[1:len(ps.state_desc)+1])

    # Plotting section
    fig, ax = plt.subplots(1)

    var1 = 'angle'
    p1 = result.xs(key=var1, axis='columns', level=1)
    # p1 = p1[['G1']]
    legnd1 = list(np.array(var1 + ': ')+p1.columns)

    var2 = 'angle'
    p2 = x_lin.xs(key=var2, axis='columns', level=1)
    legnd2 = list(np.array('linear ' + var2 + ': ') + p2.columns)

    ax.plot(t_plot, p1.sub(p1['G3'].values, axis=0))
    ax.plot(t_plot, p2.sub(p2['G3'].values, axis=0))
    ax.set_ylabel('Angle referenced to G3')
    ax.legend(legnd1+legnd2)

    fig.text(0.5, 0.04, 'Time [seconds]', ha='center')
    plt.show()