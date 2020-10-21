from PyQt5 import QtWidgets
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import time
import threading
from scipy.integrate import RK23
sys.path.append(r'C:/Users/lokal_hallvhau/Dropbox/Python/DynPSSimPy/')
import dynpssimpy.dynamic as dps
import importlib
from pyqtconsole.console import PythonConsole
import pandas as pd
import dynpssimpy.real_time_sim as dps_rts
import dynpssimpy.gui as gui


def main(rts):
    app = QtWidgets.QApplication(sys.argv)
    # main_win = gui.LivePlotter(rts, [])  # ['angle', 'speed'])
    phasor_plot = gui.PhasorPlot(rts)
    ts_plot = gui.TimeSeriesPlot(rts, ['speed'], update_freq=50)  # , 'speed', 'e_q_t', 'e_d_t', 'e_q_st', 'e_d_st'])

    # Add Control Widgets
    line_outage_ctrl = gui.LineOutageWidget(rts)
    excitation_ctrl = gui.GenCtrlWidget(rts)

    # main_win.show()
    app.exec_()

    return app
    # sys.exit(app.exec_())


if __name__ == '__main__':


    [importlib.reload(module) for module in [dps, dps_rts, gui]]

    import ps_models.k2a as model_data
    model = model_data.load()

    # model['pss'] = {}
    # model['gov'] = {}
    # model['avr'] = {}

    importlib.reload(dps)
    ps = dps.PowerSystemModel(model=model)

    ps.power_flow()
    ps.init_dyn_sim()
    ps.build_y_bus_red(ps.buses['name'])
    ps.x0[ps.angle_idx][0] += 1e-1
    rts = dps_rts.RealTimeSimulator(ps, dt=5e-3, speed=0.25)

    # gui.PhasorPlot(rts)
    rts.start()

    from threading import Thread
    app = main(rts)
    rts.stop()