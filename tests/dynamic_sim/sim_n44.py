# For running simulation in N44


import dynpssimpy.dynamic as dps
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import dynpssimpy.utility_functions as dps_uf
import importlib
import sys
import time
import numpy as np

# Line outage when HYGOV is used
# Added for not making fatal changes to the original file


if __name__ == '__main__':
    importlib.reload(dps)

    # Load model
    import ps_models.n44 as model_data
    # import ps_models.ieee39 as model_data
    # import ps_models.sm_ib as model_data
    # import ps_models.sm_load as model_data
    model = model_data.load()


    # Calling model twice, first is to get the desired names and so on.
    # Could probably do it another way, but this works
    ps = dps.PowerSystemModel(model=model)

    # Add controls for all generators (not specified in model)

    model['gov'] = {'TGOV1':
                        [['name', 'gen', 'R', 'D_t', 'V_min', 'V_max', 'T_1', 'T_2', 'T_3']] +
                        [['GOV' + str(i), gen_name, 0.05, 0, 0, 1, 0.2, 1, 2] for i, (gen_name, gen_p) in
                         enumerate(zip(ps.generators['name'], ps.generators['P']))]
                    }

    model['avr'] = {'SEXS':
                        [['name', 'gen', 'K', 'T_a', 'T_b', 'T_e', 'E_min', 'E_max']] +
                        [['AVR' + str(i), gen_name, 100, 2.0, 10.0, 0.5, -3, 3] for i, gen_name in
                         enumerate(ps.generators['name'])]
                    }
    model.pop('avr')
    model['pss'] = {'STAB1':
                        [['name', 'gen', 'K', 'T', 'T_1', 'T_2', 'T_3', 'T_4', 'H_lim']] +
                       [['PSS' + str(i), gen_name, 50, 10.0, 0.5, 0.5, 0.05, 0.05, 0.03] for i, gen_name in
                         enumerate(ps.generators['name'])]
                    }
    model.pop('pss')

    # Power system with governos, avr and pss
    ps = dps.PowerSystemModel(model=model)
    ps.use_numba = True


    ps.pf_max_it = 100
    ps.power_flow()
    ps.init_dyn_sim()

    # Solver
    t_end = 300
    sol = dps_uf.ModifiedEuler(ps.ode_fun, 0, ps.x0, t_end, max_step=30e-3)

    t = 0
    result_dict = defaultdict(list)
    t_0 = time.time()

    gen_vars = ['P_e', 'I_g','P_m']
    load_vars = ['P_l']  # l subscript means "load"

    gen_var_desc = ps.var_desc('GEN',gen_vars)
    load_var_desc = ps.var_desc('load',load_vars)

    event_flag = True
    event_flag2 = True
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        # Simulate next step
        result = sol.step()
        x = sol.y
        t = sol.t

        if t > 2 and event_flag:
            event_flag = False
            #ps.network_event('load_increase', 'B9', 'connect')
            #ps.network_event('line', 'L7-8-1', 'disconnect')
            # Load change doesnt care about connect or disconnect, the sign on the value (MW) is whats interesting
            ps.network_event('load_change', 'L7100-1', 'connect', value=-1000)

        # Store result
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]

        # Store generator values
        ps.store_vars('GEN', gen_vars, gen_var_desc, result_dict)

        # Store ACE signals
        if bool(ps.ace_mdls):
            for key, dm in ps.ace_mdls.items():
                # Store ACE signals
                [result_dict[tuple([n, 'ace'])].append(ace) for n, ace in zip(dm.par['name'], dm.ace)]
                # Store instantaneous line flows
                [result_dict[tuple([n, 'p_tie'])].append(p_tie*ps.s_n) for n, p_tie in zip(dm.par['name'], dm.p_tie)]
                # Store scheduled P_tie values
                [result_dict[tuple([n, 'p_tie0'])].append(p_tie0 * ps.s_n) for n, p_tie0 in zip(dm.par['name'], dm.int_par['Ptie0'])]

    print('\nSimulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)

    fig, ax = plt.subplots(2)
    ax[0].plot(result[('Global', 't')], result.xs(key='speed', axis='columns', level=1))
    ax[0].set_title('Speed')
    ax[0].grid(True)
    ax[1].plot(result[('Global', 't')], result.xs(key='angle', axis='columns', level=1))

    # Plot ACEs if these are included
    if bool(ps.ace_mdls):
        fig2, ax2 = plt.subplots(2)
        ax2[0].plot(result[('Global', 't')], result.xs(key='ace', axis='columns', level=1))
        ax2[0].set_title('ACE (top) and P-tie (bottom)')
        ax2[0].set_ylabel('ACE')
        ax2[1].plot(result[('Global', 't')], result.xs(key='p_tie', axis='columns', level=1))
        ax2[1].set_ylabel('P [MW]')
        # Append plot of scheduled value of active power transfer
        ax2[1].plot(result[('Global', 't')], result.xs(key='p_tie0', axis='columns', level=1))
        ax2[0].grid(True)
        ax2[1].grid(True)

    # Plot Pe to see how much each generator outputs
    var3 = 'P_e'  # variable to plot
    p3 = result.xs(key=var3, axis='columns', level=1)
    legnd3 = list(np.array(var3 + ': ') + p3.columns)
    fig3, ax3 = plt.subplots(1)
    ax3.plot(result[('Global', 't')], p3)
    ax3.set_title('Pe')
    ax3.set_ylabel('Pe [MW]')
    ax3.set_xlabel('Time [s]')
    ax3.legend(legnd3)
    ax3.grid(True)


    plt.show()