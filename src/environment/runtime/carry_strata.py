"""Shared historical/runtime bins for residual lifecycle risk."""
# Finer near-expiry ages prevent pooling a few remaining hours with nine days.
AGE_BOUNDS = (7, 14, 21, 22, 23, 24, 25, 26, 27, 28, 29)
SILENCE_BOUNDS = (1, 7, 14, 21)
DEPTH_BOUNDS = (3, 6, 10)
