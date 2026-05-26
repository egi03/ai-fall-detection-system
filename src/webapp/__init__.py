"""FastAPI uploader that runs an uploaded video through the BiLSTM ensemble.

Routes live in :mod:`src.webapp.server`; the heavy lifting (pose -> features ->
ensemble -> alarm FSM -> annotated MP4) lives in :mod:`src.webapp.pipeline`.
"""
