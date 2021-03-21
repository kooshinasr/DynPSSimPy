import numpy as np


class TGOV1:
    def __init__(self):
        self.state_list = ['x_1', 'x_2']
        self.int_par_list = ['x_1_bias']
        self.input_list = ['speed_dev']
        self.output_list = ['P_m']

    @staticmethod
    def initialize(x_0, input, output, p, int_par):
        v_2 = np.minimum(np.maximum(output['P_m'], p['V_min']), p['V_max'])
        v_1 = v_2
        v_3 = v_2

        int_par['x_1_bias'] = p['R'] * v_1

        x_0['x_1'][:] = v_2
        x_0['x_2'][:] = p['T_2'] * v_2 - p['T_3'] * v_3

    @staticmethod
    def _update(dx, x, input, output, p, int_par):

        speed_dev = input['speed_dev']
        v_1 = 1 / p['R'] * (speed_dev + int_par['x_1_bias'])
        v_2 = np.minimum(np.maximum(x['x_1'], p['V_min']), p['V_max'])
        v_3 = p['T_2'] / p['T_3'] * v_2 - 1 / p['T_3'] * x['x_2']
        delta_p_m = v_3 - p['D_t'] * speed_dev

        output['P_m'][:] = delta_p_m

        dx['x_1'][:] = 1 / p['T_1'] * (v_1 - v_2)
        dx['x_2'][:] = v_3 - v_2

        # Lims on state variable x_1 (clamping)
        lower_lim_idx = (x['x_1'] <= p['V_min']) & (dx['x_1'] < 0)
        dx['x_1'][lower_lim_idx] *= 0

        upper_lim_idx = (x['x_1'] >= p['V_max']) & (dx['x_1'] > 0)
        dx['x_1'][upper_lim_idx] *= 0


class MYGOV:
    def __init__(self):
        self.state_list = ['x_1', 'x_2']
        self.int_par_list = ['P0', 'wref']
        self.input_list = ['speed_dev']
        self.output_list = ['P_m']

    @staticmethod
    def initialize(x_0, input, output, p, int_par):
        x_0['x_1'][:] = output['P_m']
        x_0['x_2'][:] = 0
        int_par['P0'] = output['P_m']

    @staticmethod
    def _update(dx, x, input, output, p, int_par):

        speed_dev = int_par['wref'] + input['speed_dev']

        dP = x['x_1'] - int_par['P0'] - x['x_2']

        dx['x_1'][:] = (speed_dev-p['R']*dP)*p['K']
        dx['x_2'][:] = speed_dev*p['Kw']
        output['P_m'][:] = x['x_1']


class HYGOV:
    """
    purpose: Implementing a simple hydrogovernor model
    blockdiagram: droop, low-pass-filter, lead-lag filter, "lead-lag (RHP zero)", damping feedback signal
    input: speed deviation
    output: mechanical power reference
    """

    def __init__(self):
        self.state_list = ['x_1', 'x_2', 'x_3'] # three states
        self.int_par_list = ['x_1_bias']
        self.input_list = ['speed_dev']
        self.output_list = ['P_m']

    @staticmethod
    def initialize(x_0, input, output, p, int_par):
        v_2 = np.minimum(np.maximum(output['P_m'], p['V_min']), p['V_max'])
        v_1 = v_2
        v_3 = v_2
        v_4 = v_3

        int_par['x_1_bias'] = p['R'] * v_1

        x_0['x_1'][:] = v_2
        x_0['x_2'][:] = p['T_3'] * v_2 - p['T_4'] * v_3
        x_0['x_3'][:] = -p['P_m0']*p['T_w']*v_3 - p['P_m0']*p['T_w']/2*v_4

    @staticmethod
    def _update(dx, x, input, output, p, int_par):
        speed_dev = input['speed_dev']
        v_1 = 1 / p['R'] * (speed_dev + int_par['x_1_bias'])
        v_2 = np.minimum(np.maximum(x['x_1'], p['V_min']), p['V_max'])
        v_3 = p['T_3'] / p['T_4'] * v_2 - 1 / p['T_4'] * x['x_2']

        v_4 = -p['P_m0'] * p['T_w'] / (p['P_m0'] * p['T_w'] / 2) * v_3 - 1 / (
                    p['P_m0'] * p['T_w'] / 2) * x['x_3']
        delta_p_m = v_4 - p['D_t'] * speed_dev

        output['P_m'][:] = delta_p_m

        dx['x_1'][:] = 1 / p['T_2'] * (v_1 - v_2)
        dx['x_2'][:] = v_3 - v_2
        dx['x_3'][:] = v_4 - v_3

        # Lims on state variable x_1 (clamping)
        lower_lim_idx = (x['x_1'] <= p['V_min']) & (dx['x_1'] < 0)
        dx['x_1'][lower_lim_idx] *= 0

        upper_lim_idx = (x['x_1'] >= p['V_max']) & (dx['x_1'] > 0)
        dx['x_1'][upper_lim_idx] *= 0


class A8HYGOV:

    def __init__(self):
        self.state_list = ['c','xm','xf']
        self.int_par_list = ['Pm_init']
        self.input_list = ['speed_dev']
        self.output_list = ['P_m']

    @staticmethod
    def initialize(x_0, input, output, p, int_par):
        x_0['c'] = 0
        x_0['xm'] = 0
        x_0['xf'] = 0
        int_par['Pm_init'] = output['P_m']

    @staticmethod
    def _update(dx, x, input, output, p, int_par):
        speed_dev = input['speed_dev']
        u1 = speed_dev-p['R']*x['c']-x['xf']
        dx['c'][:] = 1/p['Tg']*u1
        dx['xm'][:] = 2/(p['Pm0']*p['Tw'])*(x['c']-x['xm']-p['Pm0']*p['Tw']/p['Tg']*u1)
        dx['xf'][:] = 1/p['Tr']*(p['delta']*p['Tr']/p['Tg']*u1-x['xf'])

        output['P_m'] = x['xm'] + int_par['Pm_init']


class HYGOV_LFC(HYGOV):
    """
    purpose: Implementing secondary control on the HYGOV.
             This is based on Local frequency error -> thereby being distributed frequency control.
    """
    def __init__(self):
        super().__init__()
        self.state_list.append('x_11')
        self.state_list.append('x_12')

    # Do not need to do anything with initialize, all new states are zero

    # Trying to override this methods to make use of the PID feedforward to eliminate steady-state error
    @staticmethod
    def _update(dx, x, input, output, p, int_par):

        # Primary control, calls parent class update function
        HYGOV._update(dx, x, input, output, p, int_par)

        # Secondary control loop
        speed_dev = input['speed_dev']
        s_1 = p['K_p']*speed_dev
        s_2 = (p['K_d']*speed_dev-x['x_11'])/(p['K_d']/10) # limiter time constant 1/10 of K_d
        s_3 = x['x_12']
        delta_p_m_secondary = s_1 + s_2 + s_3

        # Add power signal from secondary control
        output['P_m'][:] += delta_p_m_secondary

        # Update state variables correpsonding to the PID regulator
        dx['x_11'][:] = s_2
        dx['x_12'][:] = p['K_i']*speed_dev


