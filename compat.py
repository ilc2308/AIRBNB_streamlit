"""Compatibilidad multiplataforma para cargar modelos XGBoost serializados en Linux."""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

_APLICADO = False


def aplicar_compatibilidad_xgboost() -> None:
    """Adapta el estado interno de XGBoost al deserializar en Windows.

    XGBoost guarda el estado del generador aleatorio con un formato dependiente
    de la plataforma. Los modelos entrenados en Linux no se pueden cargar
    directamente en Windows sin normalizar ese bloque. La función es idempotente
    y no hace nada fuera de Windows.
    """

    global _APLICADO
    if _APLICADO or sys.platform != 'win32':
        return

    _APLICADO = True

    try:
        import numpy as np
        import xgboost
        import xgboost.core
    except ImportError:
        logger.warning('XGBoost no está instalado; se omite la compatibilidad de Windows.')
        return

    try:
        original_setstate = xgboost.core.Booster.__setstate__

        def _win_setstate(self, state):
            buf = state.get('handle')
            if buf is not None and isinstance(buf, (bytearray, bytes)):
                b = bytes(buf)
                marker = b'rng_stateSL'
                idx = b.find(marker)
                if idx != -1:
                    str_len = int.from_bytes(b[idx + len(marker) : idx + len(marker) + 8], 'big')
                    str_start = idx + len(marker) + 8
                    old_str = b[str_start : str_start + str_len].decode('latin1')
                    parts = old_str.strip().split()
                    if len(parts) >= 624 and all(p.isdigit() for p in parts[:10]):
                        reg = xgboost.XGBRegressor(n_estimators=1, random_state=42)
                        reg.fit(np.array([[1], [2]]), np.array([1, 2]))
                        win_buf = bytes(reg.get_booster().__getstate__()['handle'])
                        win_idx = win_buf.find(marker)
                        win_len = int.from_bytes(win_buf[win_idx + len(marker) : win_idx + len(marker) + 8], 'big')
                        win_str_start = win_idx + len(marker) + 8
                        win_str_bytes = win_buf[win_str_start : win_str_start + win_len]
                        new_buf = (
                            b[: idx + len(marker)]
                            + win_len.to_bytes(8, 'big')
                            + win_str_bytes
                            + b[str_start + str_len :]
                        )
                        state['handle'] = bytearray(new_buf)
            return original_setstate(self, state)

        setattr(xgboost.core.Booster, '__setstate__', _win_setstate)  # noqa: B010
        logger.debug('Compatibilidad de XGBoost para Windows aplicada.')
    except Exception:
        logger.warning('No se pudo aplicar la compatibilidad de XGBoost para Windows.', exc_info=True)
