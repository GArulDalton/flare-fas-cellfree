"""flarefas: FLARE port selection for fluid-antenna cell-free massive MIMO."""
from .system import (gen_batch, gen_batch_sub, port_corr, select, rzf, rates, sumrate_est,
                     sel_fixed, sel_random, sel_maxgain, sel_greedy)
from .model import (init_params, features, forward, loss_fn, sel_flare, sel_flare_ls,
                    train, save_params, load_params, n_params)

__version__ = "1.0.0"
