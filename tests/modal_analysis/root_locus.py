import dynpssimpy.dynamic as dps
import dynpssimpy.modal_analysis as dps_mdl
import dynpssimpy.plotting as dps_plt
import ps_models.smib_tet4180 as model_data
import importlib
import numpy as np
import matplotlib.pyplot as plt
import dynpssimpy.utility_functions as dps_uf


model = model_data.load()
if not model['gov_on']: model.pop('gov', None)
if not model['avr_on']: model.pop('avr', None)
if not model['pss_on']: model.pop('pss', None)
fig, ax = plt.subplots(1)

ps = dps.PowerSystemModel(model)
ps.power_flow()
ps.init_dyn_sim()


# Index to access relevant models, e.g. generators, avrs, govs etc.
# Without index, the change is applied to all instances of specified model, e.g. all 'GEN' generators, or all 'TGR' avrs

index = dps_uf.lookup_strings('G1', ps.gen_mdls['GEN'].par['name'])  # Index for G1

# index = dps_uf.lookup_strings('AVR1', ps.avr_mdls['TGR'].par['name'])  # Index for AVR1

# index = dps_uf.lookup_strings(['G1','IB'], ps.gen_mdls['GEN'].par['name'])  # Indices for G1 and IB

for i in range(30):
    ps.init_dyn_sim()
    print(index)
    ps.avr_mdls['TGR'].par['Ka'][index] = 10 + i*30
    #ps.gen_mdls['GEN'].par['H'][index] = 2 + i*0.1
    # Perform system linearization
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize()

    #pf, rev_Abs, rev_ang = ps_lin.pf_table()
    # Plot eigenvalues
    #dps_plt.plot_eigs(ps_lin.eigs)
    sc = ax.scatter(ps_lin.eigs.real, ps_lin.eigs.imag)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.grid(True)
    #print(ps_lin.eigs)

plt.show()