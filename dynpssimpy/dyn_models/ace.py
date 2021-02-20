# Attempting to create a central regulator ACE (area control error)
import numpy as np
class ACE_FIRST:
    def __init__(self):
        self.state_list = ['x_1'] # integrator
        self.int_par_list = ['Ptie0', 'f']
        self.input_list = ['speed_dev', 'p_tie']
        self.output_list = ['P_ace']

    # Initialize done in dynamic.py/init_dyn_sim, similarily as for generators, i need index and
    # setting values for the int_par_list

    @staticmethod
    def _update(dx, x, input, output, p, int_par):
        speed_dev = input['speed_dev']
        p_tie = input['p_tie']

        ace = p['lambda']*speed_dev - (p_tie - int_par['Ptie0'])
        s_1 = p['K_p']*ace
        s_3 = x['x_1']
        delta_p_ref = s_1 + s_3

        # Add power reference signal change
        output['P_ace'][:] = p['alpha']*delta_p_ref

        # Update state variable
        dx['x_1'][:] = p['K_i']*ace

        # Returns ACE solely for plotting purposes
        # Added: returns p_tie for plotting too
        return ace, p_tie

