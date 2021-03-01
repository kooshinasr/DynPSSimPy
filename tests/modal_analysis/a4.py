import dynpssimpy.dynamic as dps
import dynpssimpy.modal_analysis as dps_mdl
import dynpssimpy.plotting as dps_plt
import ps_models.smib_tet4180 as model_data
import importlib
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    model = model_data.load()
    if not model['gov_on']: model.pop('gov', None)
    if not model['avr_on']: model.pop('avr', None)
    if not model['pss_on']: model.pop('pss', None)

    ps = dps.PowerSystemModel(model)
    ps.power_flow()
    ps.init_dyn_sim()

    # Perform system linearization
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize()

    pf, rev_Abs, rev_ang = ps_lin.pf_table()
    # Plot eigenvalues
    dps_plt.plot_eigs(ps_lin.eigs)
    print(ps_lin.eigs)

    plt.show()