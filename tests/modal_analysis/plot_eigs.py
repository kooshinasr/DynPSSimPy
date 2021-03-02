import dynpssimpy.dynamic as dps
import dynpssimpy.modal_analysis as dps_mdl
import dynpssimpy.plotting as dps_plt
import ps_models.k2a as model_data
import importlib
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import ps_models.k2a as model_data
    ps = dps.PowerSystemModel(model_data.load())
    ps.power_flow()
    ps.init_dyn_sim()

    # Perform system linearization
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize()

    # Alternatively:
    # ps_lin = ps.linearize()

    # Plot eigenvalues
    dps_plt.plot_eigs(ps_lin.eigs)

    # Get mode shape for electromechanical modes
    mode_idx = ps_lin.get_mode_idx(['em'], damp_threshold=0.3)
    rev = ps_lin.rev
    mode_shape = rev[np.ix_(ps.gen_mdls['GEN'].state_idx['speed'], mode_idx)]

    # NEW
    idx_tmp = 0
    eig_tmp = ps_lin.eigs[mode_idx]
    # Plot mode shape
    fig, ax = plt.subplots(1, mode_shape.shape[1], subplot_kw={'projection': 'polar'})
    for ax_, ms in zip(ax, mode_shape.T):
        dps_plt.plot_mode_shape(ms, ax=ax_, normalize=True)

        # Getting corresponding text
        re = eig_tmp[idx_tmp].real
        im = eig_tmp[idx_tmp].imag
        str = '{:.2f} Hz\n{:.2f}%'.format(im/(2*np.pi), -100*re/(np.sqrt(re*re+im*im)))
        ax_.set_title(str, fontsize=6)
        idx_tmp += 1
        ax_.grid(True)

    plt.show()