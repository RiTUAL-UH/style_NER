import os

DEVICE = None
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR        = os.path.join(PROJ_DIR, 'data')
CHECKPOINT_DIR  = os.path.join(PROJ_DIR, 'checkpoints')
REPORT_DIR      = os.path.join(PROJ_DIR, 'reports')

FIGURE_DIR      = os.path.join(REPORT_DIR, 'figures')
HISTORY_DIR     = os.path.join(REPORT_DIR, 'history')
PREDICTIONS_DIR = os.path.join(REPORT_DIR, 'predictions')
