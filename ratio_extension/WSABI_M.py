from bayesquad.gps import WarpedGP, GP
from typing import Tuple, Union
import GPy


class WsabiMGP(WarpedGP):
    def __init__(self, gp: Union[GP, GPy.core.GP]):
        super().__init__(gp)

        self._unwarped_Y = [gp.Y**2 / 2]
        self._all_X = [gp.X]